"""
Tool executor.

Sits between the agent loop and ToolRegistry. Responsibilities:

  1. Look up ground-truth facts needed for policy evaluation (order
     total, order owner) BEFORE invoking the tool, by calling the
     backend directly -- never trusting numbers the model might have
     echoed back to itself in conversation history. This is part of
     why a manipulated model can't talk its way past the refund check:
     the policy engine is fed facts this executor fetched itself, not
     facts the model asserts.
  2. Consult PolicyEngine. DENY -> tool body never runs, model gets a
     structured refusal. ALLOW -> run normally. REQUIRES_APPROVAL -> run
     the tool, but the tool itself only files an approval (registry.py
     enforces this independently as line-of-defense #2).
  3. Retry on ToolError (simulated flaky externals) with capped
     attempts and a small backoff, then fail with a clear "tool
     unavailable" result the model can react to (e.g. apologize/escalate)
     rather than crashing the run.
  4. Cache idempotent (SAFE, read-only) tool results within a single
     run, keyed on (tool_name, args), so a model that re-queries the
     same order three times doesn't hammer the backend or burn loop
     budget on repeat lookups. Mutating/approval tools are never
     cached, since "did I already file this approval" is exactly the
     kind of state we want freshly checked against the queue, and
     repeat identical refund requests are themselves a signal worth
     letting the policy layer see again (duplicate-refund detection
     lives in PolicyEngine via already_refunded_usd, not in the cache).
  5. Emit an audit event for every decision point: policy check result,
     tool attempt/retry/failure, final result.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from support_agent.audit.log import AuditLog
from support_agent.policy.engine import PolicyEngine
from support_agent.tools.backend import MockBackend
from support_agent.tools.registry import ToolError, ToolRegistry
from support_agent.types import PolicyDecision, PolicyResult, ToolCall, ToolResult

CACHEABLE_TOOLS = {"get_customer", "get_order", "get_order_history", "search_kb"}
MAX_RETRIES = 2
RETRY_BACKOFF_S = 0.05  # kept tiny on purpose: tests/eval should run fast


@dataclass
class Executor:
    registry: ToolRegistry
    policy: PolicyEngine
    backend: MockBackend
    audit: AuditLog

    def __post_init__(self):
        self._cache: dict[tuple, ToolResult] = {}

    def reset_cache(self) -> None:
        self._cache.clear()

    def execute(self, call: ToolCall, *, run_id: str, ticket_id: str,
                requesting_customer_id: str) -> ToolResult:
        spec = self.registry.get(call.name)
        if spec is None:
            self.audit.log(run_id, ticket_id, "tool_unknown", tool=call.name, arguments=call.arguments)
            return ToolResult(call.id, call.name, ok=False,
                               content={"error": f"unknown tool '{call.name}'"})

        cache_key = (call.name, _freeze(call.arguments))
        if call.name in CACHEABLE_TOOLS and cache_key in self._cache:
            cached = self._cache[cache_key]
            self.audit.log(run_id, ticket_id, "tool_cache_hit", tool=call.name, arguments=call.arguments)
            return ToolResult(call.id, call.name, ok=cached.ok, content=cached.content,
                               policy=cached.policy, latency_s=0.0, attempt=0)

        policy_result = self._evaluate_policy(call, requesting_customer_id)
        self.audit.log(
            run_id, ticket_id, "policy_decision",
            tool=call.name, arguments=call.arguments,
            decision=policy_result.decision.value, reason=policy_result.reason,
        )

        if policy_result.decision is PolicyDecision.DENY:
            result = ToolResult(
                call.id, call.name, ok=False,
                content={"error": "denied_by_policy", "reason": policy_result.reason},
                policy=policy_result,
            )
            if call.name in CACHEABLE_TOOLS:
                self._cache[cache_key] = result
            return result

        # ALLOW or REQUIRES_APPROVAL both reach the tool body; for
        # REQUIRES_APPROVAL the tool body itself only files an approval
        # (see registry.py) rather than performing the side effect.
        kwargs = dict(call.arguments)
        if call.name in {"issue_refund", "cancel_order"}:
            kwargs.update(run_id=run_id, ticket_id=ticket_id, approval_reason=policy_result.reason)

        result = self._run_with_retry(call, spec.fn, kwargs, run_id, ticket_id)
        result.policy = policy_result

        if call.name in CACHEABLE_TOOLS and result.ok:
            self._cache[cache_key] = result

        return result

    def _evaluate_policy(self, call: ToolCall, requesting_customer_id: str) -> PolicyResult:
        order_total_usd = None
        order_customer_id = None
        already_refunded_usd = 0.0

        order_id = call.arguments.get("order_id") if isinstance(call.arguments, dict) else None
        if order_id:
            order = self.backend.get_order(order_id)
            if order:
                order_total_usd = order.total_usd
                order_customer_id = order.customer_id
                already_refunded_usd = order.refunded_usd

        return self.policy.evaluate(
            call.name,
            call.arguments if isinstance(call.arguments, dict) else {},
            order_total_usd=order_total_usd,
            order_customer_id=order_customer_id,
            requesting_customer_id=requesting_customer_id,
            already_refunded_usd=already_refunded_usd,
        )

    def _run_with_retry(self, call: ToolCall, fn, kwargs, run_id, ticket_id) -> ToolResult:
        last_err = None
        for attempt in range(1, MAX_RETRIES + 2):  # MAX_RETRIES retries + 1 initial try
            start = time.time()
            try:
                content = fn(**kwargs)
                latency = time.time() - start
                self.audit.log(run_id, ticket_id, "tool_ok", tool=call.name,
                                attempt=attempt, latency_s=latency)
                return ToolResult(call.id, call.name, ok=True, content=content,
                                   latency_s=latency, attempt=attempt)
            except ToolError as e:
                last_err = str(e)
                latency = time.time() - start
                self.audit.log(run_id, ticket_id, "tool_error", tool=call.name,
                                attempt=attempt, error=last_err, latency_s=latency)
                if attempt <= MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_S)
                    continue
            except TypeError as e:
                # Malformed/missing arguments from the model -- don't retry,
                # this won't fix itself; surface clearly so the model can
                # correct its own call.
                last_err = f"invalid arguments: {e}"
                self.audit.log(run_id, ticket_id, "tool_bad_arguments", tool=call.name,
                                error=last_err)
                break

        return ToolResult(call.id, call.name, ok=False,
                           content={"error": "tool_unavailable", "detail": last_err},
                           attempt=MAX_RETRIES + 1)


def _freeze(arguments: dict) -> tuple:
    return tuple(sorted(arguments.items())) if isinstance(arguments, dict) else (str(arguments),)
