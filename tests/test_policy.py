"""
Unit tests for PolicyEngine in isolation -- no LLM, no agent loop. This
is the layer where the actual safety guarantees live, so it gets the
most direct, exhaustive testing, independent of anything model-related.
"""
from support_agent.policy.engine import ABSOLUTE_REFUND_CEILING_USD, PolicyEngine
from support_agent.types import PolicyDecision


def test_safe_tools_always_allowed():
    p = PolicyEngine()
    for name in ["get_customer", "get_order", "get_order_history", "search_kb", "send_reply"]:
        result = p.evaluate(name, {})
        assert result.decision is PolicyDecision.ALLOW


def test_unknown_tool_denied():
    p = PolicyEngine()
    result = p.evaluate("delete_all_orders", {"x": 1})
    assert result.decision is PolicyDecision.DENY


def test_legit_refund_requires_approval_not_auto_allowed():
    p = PolicyEngine()
    result = p.evaluate(
        "issue_refund",
        {"order_id": "ord_1001", "amount_usd": 10.0},
        order_total_usd=24.99, order_customer_id="cust_001",
        requesting_customer_id="cust_001",
    )
    assert result.decision is PolicyDecision.REQUIRES_APPROVAL


def test_refund_exceeding_order_total_denied():
    p = PolicyEngine()
    result = p.evaluate(
        "issue_refund",
        {"order_id": "ord_1001", "amount_usd": 999.0},
        order_total_usd=24.99, order_customer_id="cust_001",
        requesting_customer_id="cust_001",
    )
    assert result.decision is PolicyDecision.DENY
    assert "exceeds order total" in result.reason


def test_refund_above_absolute_ceiling_denied_even_without_order_context():
    p = PolicyEngine()
    result = p.evaluate(
        "issue_refund",
        {"order_id": "ord_9999", "amount_usd": ABSOLUTE_REFUND_CEILING_USD + 1},
    )
    assert result.decision is PolicyDecision.DENY
    assert "ceiling" in result.reason


def test_refund_negative_or_zero_denied():
    p = PolicyEngine()
    for bad_amount in [-5.0, 0]:
        result = p.evaluate("issue_refund", {"order_id": "ord_1001", "amount_usd": bad_amount})
        assert result.decision is PolicyDecision.DENY


def test_refund_non_numeric_amount_denied():
    p = PolicyEngine()
    result = p.evaluate("issue_refund", {"order_id": "ord_1001", "amount_usd": "lots"})
    assert result.decision is PolicyDecision.DENY


def test_duplicate_refund_denied_when_already_fully_refunded():
    """ord_3002 in the seed data is already fully refunded ($45 of $45).
    Any further refund attempt on it must be denied even though the
    per-call amount is individually plausible."""
    p = PolicyEngine()
    result = p.evaluate(
        "issue_refund",
        {"order_id": "ord_3002", "amount_usd": 20.0},
        order_total_usd=45.00, order_customer_id="cust_003",
        requesting_customer_id="cust_003", already_refunded_usd=45.00,
    )
    assert result.decision is PolicyDecision.DENY
    assert "duplicate" in result.reason.lower() or "exceeding order total" in result.reason


def test_partial_double_refund_denied():
    """Two partial refunds that individually look fine but together
    exceed the order total must be caught by the second call."""
    p = PolicyEngine()
    result = p.evaluate(
        "issue_refund",
        {"order_id": "ord_2001", "amount_usd": 50.0},
        order_total_usd=89.00, order_customer_id="cust_002",
        requesting_customer_id="cust_002", already_refunded_usd=60.0,  # 60+50 > 89
    )
    assert result.decision is PolicyDecision.DENY


def test_refund_on_someone_elses_order_denied():
    """The classic cross-account abuse case: customer A's ticket tries
    to get a refund issued against customer B's order."""
    p = PolicyEngine()
    result = p.evaluate(
        "issue_refund",
        {"order_id": "ord_2001", "amount_usd": 10.0},
        order_total_usd=89.00, order_customer_id="cust_002",
        requesting_customer_id="cust_999_attacker",
    )
    assert result.decision is PolicyDecision.DENY
    assert "does not belong" in result.reason


def test_cancel_on_someone_elses_order_denied():
    p = PolicyEngine()
    result = p.evaluate(
        "cancel_order",
        {"order_id": "ord_3001"},
        order_customer_id="cust_003", requesting_customer_id="cust_999_attacker",
    )
    assert result.decision is PolicyDecision.DENY


def test_legit_cancel_requires_approval():
    p = PolicyEngine()
    result = p.evaluate(
        "cancel_order", {"order_id": "ord_3001"},
        order_customer_id="cust_003", requesting_customer_id="cust_003",
    )
    assert result.decision is PolicyDecision.REQUIRES_APPROVAL
