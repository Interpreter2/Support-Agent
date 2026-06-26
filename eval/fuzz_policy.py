"""
Randomized adversarial fuzz eval for the policy engine.

Hand-picked unit tests prove the policy boundary holds on cases we
thought of. This script instead generates a large number of randomized
(including deliberately extreme) refund/cancel attempts and asserts an
invariant that must hold for ALL of them:

    No combination of arguments the policy engine evaluates as ALLOW
    or REQUIRES_APPROVAL can result in a refund total that exceeds the
    order's total, or that targets an order belonging to a different
    customer than the one requesting it.

This is "demonstrably correct... especially on the abuse cases" in a
form that's stronger than enumerated examples: it searches the
argument space for a counterexample and fails loudly if it ever finds
one. Run with `python -m eval.fuzz_policy` (deterministic given
--seed, so failures are reproducible).
"""
from __future__ import annotations

import argparse
import random
import sys

from support_agent.policy.engine import PolicyEngine
from support_agent.types import PolicyDecision


def run_fuzz(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    engine = PolicyEngine()
    violations: list[str] = []

    customers = ["cust_001", "cust_002", "cust_003", "cust_999_attacker"]

    for i in range(n):
        order_total = rng.choice([0.01, 1.0, 24.99, 50.0, 89.0, 500.0, 999.99])
        already_refunded = rng.choice([0.0, 0.0, 0.0, order_total * 0.5, order_total])
        order_owner = rng.choice(customers[:3])
        requester = rng.choice(customers)

        # Bias generation toward adversarial/edge values rather than
        # uniformly "nice" ones -- this is where real bugs hide.
        amount = rng.choice([
            order_total,                              # exact match, legit
            order_total + 0.01,                        # just over
            order_total * rng.uniform(1.5, 100),        # wildly over
            -abs(rng.uniform(0, 100)),                  # negative
            0,                                           # zero
            rng.uniform(0, order_total) if order_total > 0 else 0,  # legit partial
            float("inf"),
            order_total - already_refunded,             # exactly tops it up: legit IF already_refunded < total
        ])

        args = {"order_id": f"ord_fuzz_{i}", "amount_usd": amount, "justification": "fuzz"}
        result = engine.evaluate(
            "issue_refund", args,
            order_total_usd=order_total, order_customer_id=order_owner,
            requesting_customer_id=requester, already_refunded_usd=already_refunded,
        )

        if result.decision in (PolicyDecision.ALLOW, PolicyDecision.REQUIRES_APPROVAL):
            # This call is about to be permitted to proceed (to filing,
            # for REQUIRES_APPROVAL -- recall SAFE/ALLOW never applies
            # to issue_refund at all per TOOL_TIERS). Check invariants.
            if requester != order_owner:
                violations.append(
                    f"[{i}] cross-customer refund permitted: requester={requester} "
                    f"owner={order_owner} amount={amount} decision={result.decision.value}"
                )
            if not isinstance(amount, (int, float)) or amount != amount or amount in (
                float("inf"), float("-inf")
            ):
                violations.append(f"[{i}] non-finite/invalid amount permitted: {amount}")
            elif amount <= 0:
                violations.append(f"[{i}] non-positive amount permitted: {amount}")
            elif already_refunded + amount > order_total + 0.01:
                violations.append(
                    f"[{i}] permitted refund would exceed order total: "
                    f"already={already_refunded} amount={amount} total={order_total}"
                )

    return violations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    violations = run_fuzz(args.n, args.seed)

    print(f"Ran {args.n} randomized refund-policy trials (seed={args.seed}).")
    if violations:
        print(f"\n{len(violations)} POLICY VIOLATIONS FOUND:\n")
        for v in violations[:50]:
            print(" -", v)
        if len(violations) > 50:
            print(f"   ... and {len(violations) - 50} more")
        return 1

    print("No violations found: every ALLOW/REQUIRES_APPROVAL decision respected "
          "ownership and refund-total invariants across the sampled space.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
