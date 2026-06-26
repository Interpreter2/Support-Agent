"""
LLM client interface.

Everything above this layer (agent loop, policy, tools) is written
against LLMClient and never imports a specific provider. This is what
lets the eval harness drive the system with a fully deterministic
FakeLLMClient (see llm/fake.py) while production/dev runs use
OllamaClient (see llm/ollama_client.py) against a local model. Swapping
to Anthropic/OpenAI's API later is a new ~40-line class, not a rewrite.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from support_agent.types import ToolCall


@dataclass
class LLMTurn:
    """
    A single model turn. Either it wants to call tools, or it produced
    plain text with no tool calls (treated by the agent loop as a stall
    -- see agent/loop.py for why that's not silently treated as success).
    """
    tool_calls: list[ToolCall]
    text: Optional[str]
    raw: Any = None  # provider's raw response, kept for audit logging only


class LLMClient(ABC):
    @abstractmethod
    def next_turn(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> LLMTurn:
        """
        messages: OpenAI/Ollama-style chat messages (role/content, plus
            tool results appended as role="tool").
        tools: JSON-schema tool definitions in the standard
            {"type": "function", "function": {...}} shape.
        """
        ...
