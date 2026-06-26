"""
Human approval queue.

When the policy engine returns REQUIRES_APPROVAL for a tool call, the
tool implementation does NOT perform the side effect. Instead it
constructs an ApprovalRequest, files it here, and returns a tool result
to the model saying (truthfully) that the action is pending human
review. The model's loop continues/ends believing the action is
*pending*, never *done* -- there is no code path that lets the model
observe a refund as completed within its own turn.

A human (or a test harness standing in for one) later calls approve()
or reject() on a specific request_id, which is the only code path that
actually mutates backend money/order state.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class ApprovalRequest:
    request_id: str
    run_id: str
    ticket_id: str
    tool_name: str
    arguments: dict[str, Any]
    reason: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    resolver_note: Optional[str] = None


class ApprovalQueue:
    def __init__(self):
        self._lock = threading.Lock()
        self._requests: dict[str, ApprovalRequest] = {}

    def file(
        self, *, run_id: str, ticket_id: str, tool_name: str, arguments: dict, reason: str
    ) -> ApprovalRequest:
        req = ApprovalRequest(
            request_id=f"appr_{uuid.uuid4().hex[:10]}",
            run_id=run_id,
            ticket_id=ticket_id,
            tool_name=tool_name,
            arguments=arguments,
            reason=reason,
        )
        with self._lock:
            self._requests[req.request_id] = req
        return req

    def get(self, request_id: str) -> Optional[ApprovalRequest]:
        with self._lock:
            return self._requests.get(request_id)

    def pending(self) -> list[ApprovalRequest]:
        with self._lock:
            return [r for r in self._requests.values() if r.status == ApprovalStatus.PENDING]

    def resolve(
        self,
        request_id: str,
        approve: bool,
        *,
        on_commit: Optional[Callable[[ApprovalRequest], None]] = None,
        note: str = "",
    ) -> ApprovalRequest:
        """
        on_commit is invoked (to actually mutate backend state) only if
        approve=True. This keeps "decide" and "execute" as two distinct
        steps even on the human side, mirroring the agent/tool split.
        """
        with self._lock:
            req = self._requests.get(request_id)
            if req is None:
                raise KeyError(f"no approval request {request_id}")
            if req.status != ApprovalStatus.PENDING:
                raise ValueError(f"request {request_id} already {req.status.value}")
            req.status = ApprovalStatus.APPROVED if approve else ApprovalStatus.REJECTED
            req.resolved_at = time.time()
            req.resolver_note = note

        if approve and on_commit:
            on_commit(req)

        return req
