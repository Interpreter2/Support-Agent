"""
Integration test for the concurrent runner: many tickets processed at
once, bounded by both ticket-level and LLM-call-level concurrency, with
each ticket's FakeLLMClient scripted independently so per-ticket
correctness (including a mix of benign and abusive tickets in the same
batch) still holds under concurrency.
"""
import asyncio
import time

from support_agent.llm.base import LLMClient, LLMTurn
from support_agent.llm.fake import FakeLLMClient, ScriptedTurn
from support_agent.runner import run_tickets_concurrently
from support_agent.service import build_service
from support_agent.types import Resolution, Ticket


class _MultiTicketFakeLLM(LLMClient):
    """Routes next_turn() to a per-ticket FakeLLMClient based on the
    customer_id embedded in the first user message, so many tickets can
    be run through ONE Service/LLMClient concurrently, each following
    its own script."""

    def __init__(self, scripts_by_marker: dict[str, list[ScriptedTurn]]):
        self._clients = {k: FakeLLMClient(v) for k, v in scripts_by_marker.items()}
        self.call_count = 0
        self._lock = asyncio.Lock()

    def next_turn(self, messages, tools) -> LLMTurn:
        self.call_count += 1
        first_user = next(m["content"] for m in messages if m["role"] == "user")
        for marker, client in self._clients.items():
            if marker in first_user:
                return client.next_turn(messages, tools)
        raise AssertionError(f"no script matched ticket: {first_user[:80]}")


def test_concurrent_tickets_each_get_correct_independent_outcome():
    scripts = {
        "ord_1001": [
            ScriptedTurn.call("get_order", {"order_id": "ord_1001"}),
            ScriptedTurn.call("send_reply", {"message": "Order is on its way."}),
        ],
        "ord_2001": [
            ScriptedTurn.call("get_order", {"order_id": "ord_2001"}),
            ScriptedTurn.call("issue_refund", {
                "order_id": "ord_2001", "amount_usd": 89.0, "justification": "defective",
            }),
            ScriptedTurn.call("send_reply", {"message": "Filed for approval."}),
        ],
        "ord_3002": [  # abuse case: already fully refunded, model tries again
            ScriptedTurn.call("get_order", {"order_id": "ord_3002"}),
            ScriptedTurn.call("issue_refund", {
                "order_id": "ord_3002", "amount_usd": 45.0, "justification": "refund again",
            }),
            ScriptedTurn.call("send_reply", {"message": "Already refunded."}),
        ],
    }
    llm = _MultiTicketFakeLLM(scripts)
    svc = build_service(llm, audit_path="/tmp/test_audit_concurrent.jsonl")

    tickets = [
        Ticket.new("cust_001", "Status", "check ord_1001"),
        Ticket.new("cust_002", "Refund", "please refund ord_2001, it's defective"),
        Ticket.new("cust_003", "Refund again", "refund ord_3002 again please"),
    ]

    summary = asyncio.run(run_tickets_concurrently(
        svc, tickets, max_concurrent_tickets=3, max_concurrent_llm_calls=2,
    ))

    assert len(summary.outcomes) == 3
    assert all(o.resolution == Resolution.RESOLVED for o in summary.outcomes)

    # Safety must hold per-ticket even though they ran concurrently:
    assert svc.backend.get_order("ord_1001").refunded_usd == 0.0
    assert svc.backend.get_order("ord_2001").refunded_usd == 0.0  # pending, not committed
    assert svc.backend.get_order("ord_3002").refunded_usd == 45.0  # unchanged, no double refund

    pending = svc.approvals.pending()
    assert len(pending) == 1  # only the legit ord_2001 refund was queued
    assert pending[0].arguments["order_id"] == "ord_2001"


def test_llm_call_concurrency_is_actually_bounded():
    """Verify the semaphore really limits concurrent next_turn() calls,
    not just that the math works out by coincidence."""
    max_in_flight = 0
    current_in_flight = 0
    lock = asyncio.Lock()

    class SlowLLM(LLMClient):
        def next_turn(self, messages, tools) -> LLMTurn:
            nonlocal max_in_flight, current_in_flight
            current_in_flight += 1
            max_in_flight = max(max_in_flight, current_in_flight)
            time.sleep(0.05)
            current_in_flight -= 1
            return LLMTurn(tool_calls=[], text="done", raw=None)

        def __init__(self):
            pass

    # Use real tool-free scripted turns: model just replies with text
    # immediately (which the loop treats as a stall, then escalates
    # after the nudge) -- fine, we only care about LLM-call concurrency
    # here, not ticket resolution semantics.
    svc = build_service(SlowLLM(), audit_path="/tmp/test_audit_concurrency_bound.jsonl")
    tickets = [Ticket.new("cust_001", "x", "x") for _ in range(6)]

    asyncio.run(run_tickets_concurrently(
        svc, tickets, max_concurrent_tickets=6, max_concurrent_llm_calls=2,
    ))

    assert max_in_flight <= 2, f"expected at most 2 concurrent LLM calls, saw {max_in_flight}"
