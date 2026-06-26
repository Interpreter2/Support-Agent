"""
Real client for a local Ollama server.

Uses Ollama's /api/chat endpoint with the standard `tools` parameter
(shared schema across tool-calling-capable models pulled via Ollama,
e.g. qwen2.5-instruct, llama3.1). See:
https://github.com/ollama/ollama/blob/main/docs/api.md#chat-request-with-tools

Models vary in how reliably they emit well-formed tool_calls. We do
NOT try to paper over that here with prompt tricks -- this client's
job is just to talk to the HTTP API and translate its response shape
into our LLMTurn. Tolerating a model that occasionally emits malformed
or hallucinated tool calls is the agent loop's and policy engine's job
(see agent/loop.py, policy/engine.py), because that's the layer that
also has to tolerate it from any other provider, including a
deliberately adversarial one used in eval fixtures.
"""
from __future__ import annotations

import json
import uuid
from typing import Any

import httpx

from support_agent.llm.base import LLMClient, LLMTurn
from support_agent.types import ToolCall


class OllamaClient(LLMClient):
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        timeout_s: float = 60.0,
        temperature: float = 0.2,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s
        self.temperature = temperature
        self._client = httpx.Client(timeout=timeout_s)

    def next_turn(self, messages: list[dict], tools: list[dict]) -> LLMTurn:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        resp = self._client.post(f"{self.host}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()

        message = data.get("message", {})
        raw_calls = message.get("tool_calls") or []

        tool_calls: list[ToolCall] = []
        for rc in raw_calls:
            fn = rc.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            # Ollama sometimes returns arguments as a JSON string rather
            # than a dict depending on model/version -- normalize.
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_unparsed": args}
            tool_calls.append(ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=name, arguments=args))

        text = message.get("content") or None
        return LLMTurn(tool_calls=tool_calls, text=text, raw=data)

    def close(self) -> None:
        self._client.close()
