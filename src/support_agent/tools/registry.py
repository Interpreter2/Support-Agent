"""
Tool implementations.

Each tool function:
  1. Takes plain-Python arguments (already validated against a schema
     by the registry).
  2. May raise ToolError to simulate a flaky external dependency --
     timeouts, 500s, whatever a real order/payments API would throw.
  3. For NEEDS_APPROVAL tools (issue_refund, cancel_order): NEVER
     mutates backend state directly. Always files an ApprovalRequest
     and returns a "pending" result. This is the second line of
     defense -- even if something upstream of this function (agent
     loop, policy engine) had a bug that let a call through it should
     not have, this function still cannot move money on its own. Only
     ApprovalQueue.resolve(..., approve=True) can do that, and that is
     never called from inside the agent loop.

Deliberate failure injection: get_order / get_customer / search_kb
randomly raise ToolError at backend.failure_rate, so the agent has to
demonstrate retry/fallback behavior on imperfect tools (a stated
requirement), not just on a hypothetically-perfect mock world.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from support_agent.tools.approvals import ApprovalQueue
from support_agent.tools.backend import KB_ARTICLES, MockBackend


class ToolError(Exception):
    """Raised by a tool to simulate a failing external dependency."""


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema, shared verbatim with the LLM
    fn: Callable[..., Any]


class ToolRegistry:
    def __init__(self, backend: MockBackend, approvals: ApprovalQueue):
        self.backend = backend
        self.approvals = approvals
        self._specs: dict[str, ToolSpec] = {}
        self._register_all()

    def _register_all(self) -> None:
        self._add(
            "get_customer",
            "Look up a customer's profile by customer_id.",
            {"type": "object", "properties": {
                "customer_id": {"type": "string"},
            }, "required": ["customer_id"]},
            self._get_customer,
        )
        self._add(
            "get_order",
            "Look up a single order by order_id, including total, status, and "
            "amount already refunded.",
            {"type": "object", "properties": {
                "order_id": {"type": "string"},
            }, "required": ["order_id"]},
            self._get_order,
        )
        self._add(
            "get_order_history",
            "List all orders belonging to a customer_id.",
            {"type": "object", "properties": {
                "customer_id": {"type": "string"},
            }, "required": ["customer_id"]},
            self._get_order_history,
        )
        self._add(
            "search_kb",
            "Search the internal knowledge base for help articles matching a query.",
            {"type": "object", "properties": {
                "query": {"type": "string"},
            }, "required": ["query"]},
            self._search_kb,
        )
        self._add(
            "send_reply",
            "Send the final customer-facing reply that resolves this ticket. "
            "Calling this ends the agent's turn with a RESOLVED outcome.",
            {"type": "object", "properties": {
                "message": {"type": "string"},
            }, "required": ["message"]},
            self._send_reply,
        )
        self._add(
            "issue_refund",
            "Request a refund for an order. This does NOT immediately refund the "
            "customer -- ALL refunds require human approval regardless of amount. "
            "This call files an approval request and returns its pending status.",
            {"type": "object", "properties": {
                "order_id": {"type": "string"},
                "amount_usd": {"type": "number"},
                "justification": {"type": "string"},
            }, "required": ["order_id", "amount_usd", "justification"]},
            self._issue_refund,
        )
        self._add(
            "cancel_order",
            "Request cancellation of an order that has not yet shipped. Requires "
            "human approval. Files an approval request and returns its pending status.",
            {"type": "object", "properties": {
                "order_id": {"type": "string"},
                "justification": {"type": "string"},
            }, "required": ["order_id", "justification"]},
            self._cancel_order,
        )

    def _add(self, name: str, desc: str, params: dict, fn: Callable) -> None:
        self._specs[name] = ToolSpec(name, desc, params, fn)

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    # -- tool bodies --------------------------------------------------

    def _maybe_fail(self, tool_name: str) -> None:
        if self.backend.should_inject_failure():
            raise ToolError(f"{tool_name}: upstream service timeout (injected failure)")

    def _get_customer(self, *, customer_id: str) -> dict:
        self._maybe_fail("get_customer")
        cust = self.backend.get_customer(customer_id)
        if not cust:
            return {"found": False, "customer_id": customer_id}
        return {"found": True, "customer_id": cust.customer_id, "name": cust.name, "email": cust.email}

    def _get_order(self, *, order_id: str) -> dict:
        self._maybe_fail("get_order")
        order = self.backend.get_order(order_id)
        if not order:
            return {"found": False, "order_id": order_id}
        return {
            "found": True,
            "order_id": order.order_id,
            "customer_id": order.customer_id,
            "item": order.item,
            "total_usd": order.total_usd,
            "status": order.status,
            "refunded_usd": order.refunded_usd,
        }

    def _get_order_history(self, *, customer_id: str) -> dict:
        self._maybe_fail("get_order_history")
        orders = self.backend.get_orders_for_customer(customer_id)
        return {"orders": [
            {"order_id": o.order_id, "item": o.item, "total_usd": o.total_usd,
             "status": o.status, "refunded_usd": o.refunded_usd}
            for o in orders
        ]}

    def _search_kb(self, *, query: str) -> dict:
        self._maybe_fail("search_kb")
        q = query.lower()
        hits = [a for a in KB_ARTICLES if q in a["title"].lower() or q in a["body"].lower()]
        if not hits:
            hits = KB_ARTICLES[:2]  # weak fallback so the agent still has something to read
        return {"results": hits}

    def _send_reply(self, *, message: str) -> dict:
        # No failure injection here: sending the final reply is the
        # resolution act itself, and the agent loop treats this call as
        # terminal (see agent/loop.py). We want that path deterministic.
        return {"sent": True, "message": message}

    def _issue_refund(self, *, order_id: str, amount_usd: float, justification: str,
                       run_id: str, ticket_id: str, approval_reason: str) -> dict:
        # NOTE: approval_reason/run_id/ticket_id are injected by the
        # executor (executor.py), not by the model -- see there for why.
        req = self.approvals.file(
            run_id=run_id, ticket_id=ticket_id, tool_name="issue_refund",
            arguments={"order_id": order_id, "amount_usd": amount_usd,
                       "justification": justification},
            reason=approval_reason,
        )
        return {
            "status": "pending_approval",
            "request_id": req.request_id,
            "message": (
                f"Refund of ${amount_usd:.2f} for order {order_id} has been filed "
                f"for human approval (request {req.request_id}). It has NOT been "
                f"issued yet."
            ),
        }

    def _cancel_order(self, *, order_id: str, justification: str,
                       run_id: str, ticket_id: str, approval_reason: str) -> dict:
        req = self.approvals.file(
            run_id=run_id, ticket_id=ticket_id, tool_name="cancel_order",
            arguments={"order_id": order_id, "justification": justification},
            reason=approval_reason,
        )
        return {
            "status": "pending_approval",
            "request_id": req.request_id,
            "message": (
                f"Cancellation of order {order_id} has been filed for human "
                f"approval (request {req.request_id}). It has NOT been cancelled yet."
            ),
        }
