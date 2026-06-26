import sys
from support_agent.llm.fake import FakeLLMClient, ScriptedTurn
from support_agent.service import build_service
from support_agent.types import Ticket

def test_cancellation():
    print("--- Test 1: Order Cancellation ---")
    script = [
        ScriptedTurn.call("get_order", {"order_id": "ord_3001"}),
        ScriptedTurn.call("cancel_order", {
            "order_id": "ord_3001", "justification": "Customer requested cancellation because it's taking too long."
        }),
        ScriptedTurn.call("send_reply", {
            "message": "I've filed a cancellation request for your Monitor Arm (ord_3001). Since it hasn't shipped yet, this should be processed shortly once approved."
        }),
    ]
    llm = FakeLLMClient(script)
    svc = build_service(llm, audit_path="/tmp/test_audit_demo.jsonl")
    
    ticket = Ticket.new("cust_003", "Cancel my order", "I want to cancel my order ord_3001, it is taking too long.")
    outcome = svc.loop.run(ticket)
    
    print(f"Resolution: {outcome.resolution.value}")
    print(f"Reply: {outcome.customer_reply}")
    print(f"Pending Approvals: {len(svc.approvals.pending())}")
    for p in svc.approvals.pending():
        print(f"  -> Tool: {p.tool_name}, Arguments: {p.arguments}")
    print()

def test_inquiry():
    print("--- Test 2: Policy Inquiry ---")
    script = [
        ScriptedTurn.call("search_kb", {"query": "shipping times"}),
        ScriptedTurn.call("send_reply", {
            "message": "Standard shipping takes 3-5 business days domestically, while expedited shipping takes 1-2 business days."
        }),
    ]
    llm = FakeLLMClient(script)
    svc = build_service(llm, audit_path="/tmp/test_audit_demo.jsonl")
    
    ticket = Ticket.new("cust_002", "Shipping times", "How long does standard shipping take?")
    outcome = svc.loop.run(ticket)
    
    print(f"Resolution: {outcome.resolution.value}")
    print(f"Reply: {outcome.customer_reply}")
    print()

def test_status_check():
    print("--- Test 3: Status Check ---")
    script = [
        ScriptedTurn.call("get_order", {"order_id": "ord_1002"}),
        ScriptedTurn.call("send_reply", {
            "message": "Your USB-C Hub (ord_1002) has already shipped!"
        }),
    ]
    llm = FakeLLMClient(script)
    svc = build_service(llm, audit_path="/tmp/test_audit_demo.jsonl")
    
    ticket = Ticket.new("cust_001", "Where is my order", "Can you check the status of my order ord_1002?")
    outcome = svc.loop.run(ticket)
    
    print(f"Resolution: {outcome.resolution.value}")
    print(f"Reply: {outcome.customer_reply}")
    print()

if __name__ == "__main__":
    test_cancellation()
    test_inquiry()
    test_status_check()
