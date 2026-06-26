"""
Concurrent ticket processing with rate limiting.

Two independent limits, because they guard different things:

  - max_concurrent_tickets: bounds how many tickets are "in flight"
    (each ticket's full agent loop, including all its tool calls).
  - max_concurrent_llm_calls: separately bounds concurrent calls to
    the model specifically. Ollama is a single local process serving
    one model; even if we're happy running 20 tickets concurrently,
    hammering Ollama with 20 simultaneous generate requests will just
    thrash it (or queue inside Ollama unpredictably). A semaphore
    around the LLM call specifically keeps that bounded independent of
    ticket concurrency.

The loop's actual LLM calls happen inside AgentLoop.run() via
self.llm.next_turn(); to rate-limit those without threading a
semaphore through every layer, we wrap the LLMClient passed to the
Service in a small decorator that acquires the semaphore around
next_turn(). This keeps AgentLoop itself unaware of concurrency
concerns entirely -- it's correct to read top-to-bottom as if it were
the only ticket running.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from support_agent.llm.base import LLMClient, LLMTurn
from support_agent.service import Service
from support_agent.types import AgentOutcome, Ticket


class _RateLimitedLLM(LLMClient):
    """Wraps a real LLMClient, bounding concurrent next_turn() calls via
    a threading semaphore (agent loop runs in a worker thread per ticket)."""

    def __init__(self, inner: LLMClient, max_concurrent: int):
        import threading
        self._inner = inner
        self._sem = threading.Semaphore(max_concurrent)

    def next_turn(self, messages, tools) -> LLMTurn:
        with self._sem:
            return self._inner.next_turn(messages, tools)


@dataclass
class RunSummary:
    outcomes: list[AgentOutcome]

    @property
    def resolved_count(self) -> int:
        return sum(1 for o in self.outcomes if o.resolution.value == "resolved")

    @property
    def escalated_count(self) -> int:
        return sum(1 for o in self.outcomes if o.resolution.value == "escalated")


async def run_tickets_concurrently(
    service: Service,
    tickets: list[Ticket],
    *,
    max_concurrent_tickets: int = 5,
    max_concurrent_llm_calls: int = 2,
) -> RunSummary:
    # Swap in the rate-limited LLM wrapper for the duration of this run.
    original_llm = service.loop.llm
    service.loop.llm = _RateLimitedLLM(original_llm, max_concurrent_llm_calls)

    ticket_sem = asyncio.Semaphore(max_concurrent_tickets)
    loop = asyncio.get_event_loop()

    async def process(ticket: Ticket) -> AgentOutcome:
        async with ticket_sem:
            # AgentLoop.run is sync (it does blocking HTTP under the
            # hood for the real client); push it to a thread so many
            # tickets' loops can interleave without one blocking
            # another's event-loop turn.
            return await loop.run_in_executor(None, service.loop.run, ticket)

    try:
        outcomes = await asyncio.gather(*(process(t) for t in tickets))
    finally:
        service.loop.llm = original_llm

    return RunSummary(outcomes=list(outcomes))
