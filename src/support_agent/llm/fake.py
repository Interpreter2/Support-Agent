"""
Scriptable fake LLM client.

This is the load-bearing piece for "show, in some repeatable way, that
the system behaves the way you claim -- especially on the abuse
cases." A real model is non-deterministic and its tool-calling
reliability varies by model/quantization/day. We don't want our proof
of *safety* to depend on a model happening to behave on a given run.

FakeLLMClient lets a test/eval script script out an exact sequence of
model turns per ticket -- including deliberately adversarial ones
(oversized refund, duplicate refund, cross-customer order access,
malformed JSON arguments, an infinite-loop attempt, a prompt-injection
payload embedded in tool results that the script can assert the model
"saw" but the *policy layer* ignored). Running the same script twice
gives the same outcome every time, which is what makes the safety
claims testable in CI rather than "I tried it a few times."

It is also used to script known-tricky-but-benign flows (multi-step
lookups, recovering from an injected tool failure) so we can assert
the *agent loop's* control flow independent of any real model's quirks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from support_agent.llm.base import LLMClient, LLMTurn
from support_agent.types import ToolCall


@dataclass
class ScriptedTurn:
    """One forced model turn. Provide either tool_calls or text, matching
    what a real LLMTurn would contain."""
    tool_calls: list[ToolCall] = field(default_factory=list)
    text: Optional[str] = None

    @staticmethod
    def call(name: str, arguments: dict, call_id: str | None = None) -> "ScriptedTurn":
        return ScriptedTurn(tool_calls=[
            ToolCall(id=call_id or f"call_{name}_{id(arguments)}", name=name, arguments=arguments)
        ])

    @staticmethod
    def calls(*pairs: tuple[str, dict]) -> "ScriptedTurn":
        return ScriptedTurn(tool_calls=[
            ToolCall(id=f"call_{n}_{i}", name=n, arguments=a) for i, (n, a) in enumerate(pairs)
        ])

    @staticmethod
    def say(text: str) -> "ScriptedTurn":
        return ScriptedTurn(text=text)


class FakeLLMClient(LLMClient):
    """
    Plays back a fixed list of ScriptedTurns, one per call to
    next_turn(), regardless of the messages/tools passed in. If the
    script runs out, raises -- a test that exhausts its script has a
    bug in the script (the agent looped more than expected), which we
    want to fail loudly rather than silently fall back to anything.
    """

    def __init__(self, script: list[ScriptedTurn]):
        self._script = list(script)
        self._i = 0
        self.calls_seen: list[tuple[list[dict], list[dict]]] = []

    def next_turn(self, messages: list[dict], tools: list[dict]) -> LLMTurn:
        self.calls_seen.append((messages, tools))
        if self._i >= len(self._script):
            raise AssertionError(
                f"FakeLLMClient script exhausted after {self._i} turns; "
                f"agent loop made more calls than the test expected"
            )
        turn = self._script[self._i]
        self._i += 1
        return LLMTurn(tool_calls=list(turn.tool_calls), text=turn.text, raw={"scripted": True})

    @property
    def turns_consumed(self) -> int:
        return self._i
