"""
Composition root. One place that wires concrete implementations
together, so tests/eval can swap the LLM client (real Ollama vs Fake)
without touching agent/policy/tool code at all.
"""
from __future__ import annotations

from dataclasses import dataclass

from support_agent.agent.loop import AgentLoop
from support_agent.audit.log import AuditLog
from support_agent.llm.base import LLMClient
from support_agent.policy.engine import PolicyEngine
from support_agent.tools.approvals import ApprovalQueue
from support_agent.tools.backend import MockBackend
from support_agent.tools.executor import Executor
from support_agent.tools.registry import ToolRegistry


@dataclass
class Service:
    backend: MockBackend
    approvals: ApprovalQueue
    registry: ToolRegistry
    policy: PolicyEngine
    audit: AuditLog
    loop: AgentLoop


def build_service(
    llm: LLMClient,
    *,
    audit_path: str = "audit.jsonl",
    tool_failure_rate: float = 0.0,
    backend_seed: int | None = 7,
) -> Service:
    backend = MockBackend(failure_rate=tool_failure_rate, seed=backend_seed)
    approvals = ApprovalQueue()
    registry = ToolRegistry(backend, approvals)
    policy = PolicyEngine()
    audit = AuditLog(audit_path)
    executor = Executor(registry=registry, policy=policy, backend=backend, audit=audit)
    loop = AgentLoop(llm=llm, registry=registry, executor=executor, audit=audit)
    return Service(backend=backend, approvals=approvals, registry=registry,
                    policy=policy, audit=audit, loop=loop)
