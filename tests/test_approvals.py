"""
Tests for ApprovalQueue: the human-in-the-loop commit path. These are
deliberately separate from agent-loop tests because approving/
rejecting is something a human (or an ops script) does out-of-band,
not something the agent ever triggers itself.
"""
import pytest

from support_agent.tools.approvals import ApprovalQueue
from support_agent.tools.backend import MockBackend


def test_filing_does_not_mutate_backend():
    backend = MockBackend()
    q = ApprovalQueue()
    before = backend.get_order("ord_1001").refunded_usd
    q.file(run_id="r1", ticket_id="t1", tool_name="issue_refund",
           arguments={"order_id": "ord_1001", "amount_usd": 10.0}, reason="test")
    after = backend.get_order("ord_1001").refunded_usd
    assert before == after == 0.0


def test_approve_commits_refund_via_on_commit_callback():
    backend = MockBackend()
    q = ApprovalQueue()
    req = q.file(run_id="r1", ticket_id="t1", tool_name="issue_refund",
                 arguments={"order_id": "ord_1001", "amount_usd": 10.0}, reason="test")

    def commit(r):
        backend.commit_refund(r.arguments["order_id"], r.arguments["amount_usd"])

    resolved = q.resolve(req.request_id, approve=True, on_commit=commit, note="looks legit")
    assert resolved.status.value == "approved"
    assert backend.get_order("ord_1001").refunded_usd == 10.0


def test_reject_does_not_commit():
    backend = MockBackend()
    q = ApprovalQueue()
    req = q.file(run_id="r1", ticket_id="t1", tool_name="issue_refund",
                 arguments={"order_id": "ord_1001", "amount_usd": 10.0}, reason="test")

    committed = []
    resolved = q.resolve(req.request_id, approve=False,
                          on_commit=lambda r: committed.append(r), note="seems off")
    assert resolved.status.value == "rejected"
    assert committed == []
    assert backend.get_order("ord_1001").refunded_usd == 0.0


def test_cannot_resolve_twice():
    q = ApprovalQueue()
    req = q.file(run_id="r1", ticket_id="t1", tool_name="issue_refund",
                 arguments={}, reason="test")
    q.resolve(req.request_id, approve=True)
    with pytest.raises(ValueError):
        q.resolve(req.request_id, approve=False)


def test_pending_lists_only_unresolved():
    q = ApprovalQueue()
    r1 = q.file(run_id="r1", ticket_id="t1", tool_name="issue_refund", arguments={}, reason="a")
    r2 = q.file(run_id="r2", ticket_id="t2", tool_name="issue_refund", arguments={}, reason="b")
    q.resolve(r1.request_id, approve=True)
    pending = q.pending()
    assert len(pending) == 1
    assert pending[0].request_id == r2.request_id
