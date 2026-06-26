"""
API-level integration tests. We override get_service's global Service
with one built on FakeLLMClient so these run fully offline/deterministically,
exercising the same HTTP surface a real deployment would use.
"""
import pathlib

import support_agent.api as api_module
from fastapi.testclient import TestClient
from support_agent.llm.fake import FakeLLMClient, ScriptedTurn
from support_agent.service import build_service


def _client_with_script(script):
    llm = FakeLLMClient(script)
    audit_path = "/tmp/test_api_audit.jsonl"
    p = pathlib.Path(audit_path)
    if p.exists():
        p.unlink()
    api_module._service = build_service(llm, audit_path=audit_path)
    return TestClient(api_module.app), api_module._service


def test_submit_ticket_resolves():
    script = [
        ScriptedTurn.call("get_order", {"order_id": "ord_1001"}),
        ScriptedTurn.call("send_reply", {"message": "Your order is on the way."}),
    ]
    client, svc = _client_with_script(script)
    resp = client.post("/tickets", json={
        "customer_id": "cust_001", "subject": "Where's my stuff", "body": "check ord_1001",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["resolution"] == "resolved"
    assert "on the way" in data["customer_reply"]


def test_refund_flow_through_api_creates_pending_approval_then_commits_on_approve():
    script = [
        ScriptedTurn.call("get_order", {"order_id": "ord_1001"}),
        ScriptedTurn.call("issue_refund", {
            "order_id": "ord_1001", "amount_usd": 24.99, "justification": "broken",
        }),
        ScriptedTurn.call("send_reply", {"message": "Filed for approval."}),
    ]
    client, svc = _client_with_script(script)

    resp = client.post("/tickets", json={
        "customer_id": "cust_001", "subject": "Broken", "body": "ord_1001 arrived broken",
    })
    assert resp.status_code == 200

    pending_resp = client.get("/approvals")
    pending = pending_resp.json()
    assert len(pending) == 1
    request_id = pending[0]["request_id"]

    # Backend untouched before approval.
    assert svc.backend.get_order("ord_1001").refunded_usd == 0.0

    approve_resp = client.post(f"/approvals/{request_id}/approve", json={"note": "verified"})
    assert approve_resp.status_code == 200
    assert approve_resp.json()["status"] == "approved"

    # Now, and only now, backend state changes.
    assert svc.backend.get_order("ord_1001").refunded_usd == 24.99

    # Can't double-resolve.
    again = client.post(f"/approvals/{request_id}/approve", json={"note": "again"})
    assert again.status_code == 409


def test_audit_endpoint_returns_events_for_run():
    script = [
        ScriptedTurn.call("get_order", {"order_id": "ord_1001"}),
        ScriptedTurn.call("send_reply", {"message": "ok"}),
    ]
    client, svc = _client_with_script(script)
    resp = client.post("/tickets", json={
        "customer_id": "cust_001", "subject": "x", "body": "check ord_1001",
    })
    run_id = resp.json()["run_id"]
    audit_resp = client.get(f"/runs/{run_id}/audit")
    assert audit_resp.status_code == 200
    events = audit_resp.json()
    assert any(e["event"] == "run_resolved" for e in events)


def test_health():
    client, _ = _client_with_script([ScriptedTurn.say("noop")])
    assert client.get("/health").json() == {"status": "ok"}
