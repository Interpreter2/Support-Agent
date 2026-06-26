"""
HTTP service surface.

Endpoints:
  POST /tickets              submit a ticket, runs the agent synchronously, returns the outcome
  GET  /tickets/{id}/audit   full audit trail for the most recent run of that ticket
  GET  /approvals            list pending human-approval requests
  POST /approvals/{id}/approve   commit the underlying action (refund/cancel)
  POST /approvals/{id}/reject    reject without committing

Kept intentionally small: this is a take-home, not a production API.
Notably absent on purpose (see DESIGN.md "what I'd change at
production scale"): auth/authn on the approval endpoints, persistence
beyond the JSONL audit file and in-memory backend, idempotency keys on
ticket submission, and a queue/worker split for ticket processing
(currently /tickets blocks for the duration of the agent loop, which
is fine for a demo but wrong for production traffic).
"""
from __future__ import annotations

import os

import asyncio

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from support_agent.llm.ollama_client import OllamaClient
from support_agent.runner import run_tickets_concurrently
from support_agent.service import Service, build_service
from support_agent.types import Ticket

app = FastAPI(title="Autonomous Support Agent")

_service: Service | None = None


def get_service() -> Service:
    global _service
    if _service is None:
        model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")
        if model == "fake":
            from support_agent.llm.fake import FakeLLMClient, ScriptedTurn
            script = [
                ScriptedTurn.call("search_kb", {"query": "general"}),
                ScriptedTurn.call("send_reply", {"message": "Hello from the frontend! I am the mock LLM responding perfectly to your ticket while the real AI model finishes downloading. Your UI looks great!"})
            ]
            llm = FakeLLMClient(script * 100)
        else:
            host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            llm = OllamaClient(model=model, host=host)
        _service = build_service(llm, audit_path=os.environ.get("AUDIT_PATH", "audit.jsonl"))
    return _service


class TicketIn(BaseModel):
    customer_id: str
    subject: str
    body: str


class OutcomeOut(BaseModel):
    ticket_id: str
    run_id: str
    resolution: str
    customer_reply: str | None
    escalation_reason: str | None
    iterations: int
    duration_s: float
    critique_occurred: bool = False
    improved_by_critique: bool = False


class ApprovalAction(BaseModel):
    note: str = ""


@app.post("/tickets", response_model=OutcomeOut)
def submit_ticket(payload: TicketIn) -> OutcomeOut:
    svc = get_service()
    ticket = Ticket.new(payload.customer_id, payload.subject, payload.body)
    outcome = svc.loop.run(ticket)
    return OutcomeOut(
        ticket_id=outcome.ticket_id, run_id=outcome.run_id,
        resolution=outcome.resolution.value, customer_reply=outcome.customer_reply,
        escalation_reason=outcome.escalation_reason, iterations=outcome.iterations,
        duration_s=outcome.duration_s, critique_occurred=outcome.critique_occurred,
        improved_by_critique=outcome.improved_by_critique,
    )


@app.post("/tickets/batch", response_model=list[OutcomeOut])
async def submit_ticket_batch(payloads: list[TicketIn]) -> list[OutcomeOut]:
    svc = get_service()
    tickets = [Ticket.new(p.customer_id, p.subject, p.body) for p in payloads]
    summary = await run_tickets_concurrently(svc, tickets)
    return [
        OutcomeOut(
            ticket_id=outcome.ticket_id, run_id=outcome.run_id,
            resolution=outcome.resolution.value, customer_reply=outcome.customer_reply,
            escalation_reason=outcome.escalation_reason, iterations=outcome.iterations,
            duration_s=outcome.duration_s, critique_occurred=outcome.critique_occurred,
            improved_by_critique=outcome.improved_by_critique,
        )
        for outcome in summary.outcomes
    ]


@app.get("/runs/{run_id}/audit")
def get_audit(run_id: str) -> list[dict]:
    svc = get_service()
    events = svc.audit.events_for_run(run_id)
    if not events:
        raise HTTPException(404, f"no audit events for run {run_id}")
    return events


@app.get("/runs/stream")
async def stream_runs(request: Request):
    q = asyncio.Queue()
    svc = get_service()
    
    loop = asyncio.get_event_loop()
    def _callback(line: str):
        loop.call_soon_threadsafe(q.put_nowait, line)
        
    svc.audit.subscribers.append(_callback)
    
    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                line = await q.get()
                yield f"data: {line}\n\n"
        finally:
            if _callback in svc.audit.subscribers:
                svc.audit.subscribers.remove(_callback)
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/approvals")
def list_pending_approvals() -> list[dict]:
    svc = get_service()
    return [
        {
            "request_id": r.request_id, "run_id": r.run_id, "ticket_id": r.ticket_id,
            "tool_name": r.tool_name, "arguments": r.arguments, "reason": r.reason,
            "created_at": r.created_at,
        }
        for r in svc.approvals.pending()
    ]


@app.post("/approvals/{request_id}/approve")
def approve(request_id: str, payload: ApprovalAction) -> dict:
    svc = get_service()
    req = svc.approvals.get(request_id)
    if req is None:
        raise HTTPException(404, f"no approval request {request_id}")

    def commit(r):
        if r.tool_name == "issue_refund":
            svc.backend.commit_refund(r.arguments["order_id"], r.arguments["amount_usd"])
        elif r.tool_name == "cancel_order":
            svc.backend.commit_cancel(r.arguments["order_id"])

    try:
        resolved = svc.approvals.resolve(request_id, approve=True, on_commit=commit,
                                          note=payload.note)
    except ValueError as e:
        raise HTTPException(409, str(e))

    svc.audit.log(req.run_id, req.ticket_id, "approval_committed",
                  request_id=request_id, tool_name=req.tool_name, arguments=req.arguments)
    return {"request_id": resolved.request_id, "status": resolved.status.value}


@app.post("/approvals/{request_id}/reject")
def reject(request_id: str, payload: ApprovalAction) -> dict:
    svc = get_service()
    req = svc.approvals.get(request_id)
    if req is None:
        raise HTTPException(404, f"no approval request {request_id}")
    try:
        resolved = svc.approvals.resolve(request_id, approve=False, note=payload.note)
    except ValueError as e:
        raise HTTPException(409, str(e))

    svc.audit.log(req.run_id, req.ticket_id, "approval_rejected",
                  request_id=request_id, tool_name=req.tool_name, note=payload.note)
    return {"request_id": resolved.request_id, "status": resolved.status.value}


@app.get("/runs")
def list_runs() -> list[dict]:
    """Summarize every run seen in the audit log, most recent first."""
    svc = get_service()
    events = svc.audit.all_events()
    by_run: dict[str, dict] = {}
    order: list[str] = []
    for e in events:
        rid = e.get("run_id")
        if rid is None:
            continue
        if rid not in by_run:
            by_run[rid] = {
                "run_id": rid, "ticket_id": e.get("ticket_id"),
                "started_at": None, "resolution": None, "reason": None,
                "iterations": None, "duration_s": None, "subject": None,
                "customer_id": None, "tool_calls": 0, "policy_denials": 0,
                "approvals_filed": 0,
            }
            order.append(rid)
        rec = by_run[rid]
        if e["event"] == "run_started":
            rec["started_at"] = e["ts"]
            rec["subject"] = e.get("subject")
            rec["customer_id"] = e.get("customer_id")
        elif e["event"] == "run_resolved":
            rec["resolution"] = "resolved"
            rec["iterations"] = e.get("iterations")
            rec["duration_s"] = e.get("duration_s")
        elif e["event"] == "run_escalated":
            rec["resolution"] = "escalated"
            rec["reason"] = e.get("reason")
            rec["iterations"] = e.get("iterations")
            rec["duration_s"] = e.get("duration_s")
        elif e["event"] in ("tool_ok", "tool_error"):
            rec["tool_calls"] += 1
        elif e["event"] == "policy_decision" and e.get("decision") == "deny":
            rec["policy_denials"] += 1
        elif e["event"] == "tool_ok" and e.get("tool") in ("issue_refund", "cancel_order"):
            rec["approvals_filed"] += 1

    return [by_run[rid] for rid in reversed(order)]


@app.get("/dashboard")
def dashboard() -> HTMLResponse:
    html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    with open(html_path) as f:
        return HTMLResponse(f.read())


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
