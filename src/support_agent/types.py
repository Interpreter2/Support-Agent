"""
Core domain types. Kept dependency-free (no LLM SDK imports here) so that
policy/tools/audit code can be unit tested without ever touching a model.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Tickets
# ---------------------------------------------------------------------------

@dataclass
class Ticket:
    ticket_id: str
    customer_id: str
    subject: str
    body: str
    created_at: float = field(default_factory=time.time)

    @staticmethod
    def new(customer_id: str, subject: str, body: str) -> "Ticket":
        return Ticket(
            ticket_id=f"tkt_{uuid.uuid4().hex[:10]}",
            customer_id=customer_id,
            subject=subject,
            body=body,
        )


# ---------------------------------------------------------------------------
# Tool risk tiers + policy decisions
# ---------------------------------------------------------------------------

class ToolTier(str, Enum):
    """
    SAFE: read-only or reversible. Auto-executes.
    NEEDS_APPROVAL: irreversible / money-moving. The tool implementation
        itself refuses to perform the side effect; it only files a pending
        approval request. A human must commit it out-of-band. This is
        enforced inside the tool, not just by the agent loop, so a model
        that bypasses orchestration logic still cannot move money.
    """
    SAFE = "safe"
    NEEDS_APPROVAL = "needs_approval"


class PolicyDecision(str, Enum):
    ALLOW = "allow"              # tool may run normally
    DENY = "deny"                 # tool call is blocked outright, never reaches the tool
    REQUIRES_APPROVAL = "requires_approval"  # tool runs in "file for approval" mode


@dataclass
class PolicyResult:
    decision: PolicyDecision
    reason: str


# ---------------------------------------------------------------------------
# Tool calls / results
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    call_id: str
    name: str
    ok: bool
    content: Any
    policy: Optional[PolicyResult] = None
    latency_s: float = 0.0
    attempt: int = 1


# ---------------------------------------------------------------------------
# Agent run outcome
# ---------------------------------------------------------------------------

class Resolution(str, Enum):
    RESOLVED = "resolved"
    ESCALATED = "escalated"


@dataclass
class AgentOutcome:
    ticket_id: str
    resolution: Resolution
    customer_reply: Optional[str]
    escalation_reason: Optional[str]
    run_id: str
    iterations: int
    duration_s: float
