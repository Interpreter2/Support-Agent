"""
Deterministic policy engine.

This is the answer to "how does this hold up against a manipulated
model": the policy layer never reads the model's reasoning/justification
text. It only ever looks at:

    (tool_name, structured_arguments, ground-truth backend state)

and applies fixed rules. A prompt-injected ticket can make the model
WANT to call issue_refund with a huge amount, or call it twice, or
target someone else's order -- but it cannot change what these rules
check, because the rules are plain Python that runs after the model's
turn, not instructions the model is asked to follow.

Defense in depth: this module is the *first* line of defense (agent
orchestration consults it before invoking a tool). The *second* line
lives inside the tool implementations themselves (see tools/), which
independently refuse to perform irreversible side effects regardless
of what the policy layer decided -- so even a bug here, or an agent
that calls a tool function directly bypassing this check, still cannot
move money.
"""
from __future__ import annotations

from dataclasses import dataclass

from support_agent.types import PolicyDecision, PolicyResult, ToolTier


# Tool name -> tier. This is the single source of truth for which tools
# are auto-executable vs. which require human sign-off. Defined here
# (not scattered across tool implementations) so it's auditable in one
# place.
TOOL_TIERS: dict[str, ToolTier] = {
    "get_customer": ToolTier.SAFE,
    "get_order": ToolTier.SAFE,
    "get_order_history": ToolTier.SAFE,
    "search_kb": ToolTier.SAFE,
    "send_reply": ToolTier.SAFE,
    "issue_refund": ToolTier.NEEDS_APPROVAL,
    "cancel_order": ToolTier.NEEDS_APPROVAL,
}

# Hard ceiling: even an *approval request* for an absurd amount is
# rejected outright rather than queued, on the theory that a refund
# request many multiples of any plausible order total is more likely
# a manipulated/confused model than a legitimate edge case, and a
# human reviewer's time shouldn't be spent on obviously-bogus requests.
ABSOLUTE_REFUND_CEILING_USD = 1000.00


@dataclass
class PolicyEngine:
    """
    Stateless except for what it's told about each call; safe to share
    across concurrent ticket workers.
    """

    def evaluate(
        self,
        tool_name: str,
        arguments: dict,
        *,
        order_total_usd: float | None = None,
        order_customer_id: str | None = None,
        requesting_customer_id: str | None = None,
        already_refunded_usd: float = 0.0,
    ) -> PolicyResult:
        tier = TOOL_TIERS.get(tool_name)

        if tier is None:
            # Unknown tool name. Could be a hallucinated tool, a typo,
            # or a model trying something it was never given. Deny by
            # default -- unknown is unsafe.
            return PolicyResult(PolicyDecision.DENY, f"unknown tool '{tool_name}'")

        if tier is ToolTier.SAFE:
            return PolicyResult(PolicyDecision.ALLOW, "safe tier: auto-allowed")

        # --- NEEDS_APPROVAL tier from here down ---

        if tool_name == "issue_refund":
            amount = arguments.get("amount_usd")
            order_id = arguments.get("order_id")

            if not isinstance(amount, (int, float)) or amount <= 0:
                return PolicyResult(PolicyDecision.DENY, "invalid or non-positive refund amount")

            if amount > ABSOLUTE_REFUND_CEILING_USD:
                return PolicyResult(
                    PolicyDecision.DENY,
                    f"refund amount ${amount:.2f} exceeds absolute ceiling "
                    f"${ABSOLUTE_REFUND_CEILING_USD:.2f}",
                )

            if order_total_usd is not None and amount > order_total_usd + 0.01:
                return PolicyResult(
                    PolicyDecision.DENY,
                    f"refund amount ${amount:.2f} exceeds order total ${order_total_usd:.2f}",
                )

            if (
                order_total_usd is not None
                and already_refunded_usd + amount > order_total_usd + 0.01
            ):
                return PolicyResult(
                    PolicyDecision.DENY,
                    f"refund would bring total refunded to "
                    f"${already_refunded_usd + amount:.2f}, exceeding order total "
                    f"${order_total_usd:.2f} (possible duplicate refund attempt)",
                )

            if (
                order_customer_id is not None
                and requesting_customer_id is not None
                and order_customer_id != requesting_customer_id
            ):
                return PolicyResult(
                    PolicyDecision.DENY,
                    "order does not belong to the customer who filed this ticket",
                )

            return PolicyResult(
                PolicyDecision.REQUIRES_APPROVAL,
                f"refund of ${amount:.2f} on order {order_id} requires human approval "
                f"(policy default: ALL refunds require approval, no auto-approve threshold)",
            )

        if tool_name == "cancel_order":
            order_id = arguments.get("order_id")
            if (
                order_customer_id is not None
                and requesting_customer_id is not None
                and order_customer_id != requesting_customer_id
            ):
                return PolicyResult(
                    PolicyDecision.DENY,
                    "order does not belong to the customer who filed this ticket",
                )
            return PolicyResult(
                PolicyDecision.REQUIRES_APPROVAL,
                f"cancellation of order {order_id} requires human approval",
            )

        # Should be unreachable given TOOL_TIERS, but fail closed.
        return PolicyResult(PolicyDecision.DENY, "no matching policy rule; failing closed")
