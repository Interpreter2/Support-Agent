# Autonomous Support Agent

Resolves customer support tickets by calling tools and an LLM. Money-moving
and irreversible actions (refunds, cancellations) always require human
approval — the agent can request them, but cannot complete them on its own.
See `DESIGN.md` for the architecture, safety reasoning, and trade-offs.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

For the live model, install [Ollama](https://ollama.com) and pull a model
that supports tool calling, e.g.:

```bash
ollama pull qwen2.5:7b-instruct
ollama serve   # if not already running
```

## Run the deterministic test suite (no model required)

This is the primary evidence that the safety machinery works — it runs
against a scriptable fake LLM, including deliberately adversarial scripts
(oversized refunds, cross-customer refund attempts, prompt injection,
infinite-loop attempts), so it's fully repeatable and doesn't depend on a
real model's non-determinism.

```bash
pytest tests/ -v
```

Also run the randomized policy fuzz eval, which searches the refund-argument
space for any case where the policy engine would permit a cross-customer
refund or a refund exceeding the order total:

```bash
python -m eval.fuzz_policy --n 20000
```

## Run the live service against Ollama

```bash
export OLLAMA_MODEL=qwen2.5:7b-instruct   # default
uvicorn support_agent.api:app --reload --port 8000
```

Submit a ticket:

```bash
curl -X POST localhost:8000/tickets -H "Content-Type: application/json" -d '{
  "customer_id": "cust_001",
  "subject": "Broken mouse",
  "body": "My wireless mouse (ord_1001) arrived broken, can I get a refund?"
}'
```

Open the dashboard to see the run, its full audit trail, and any pending
approvals:

```
http://localhost:8000/dashboard
```

Approve or reject a pending refund/cancellation:

```bash
curl -X POST localhost:8000/approvals/<request_id>/approve -H "Content-Type: application/json" -d '{"note": "verified with customer"}'
```

Seeded mock customers/orders (see `src/support_agent/tools/backend.py`):

| customer_id | orders |
|---|---|
| cust_001 | ord_1001 ($24.99, delivered), ord_1002 ($39.99, shipped) |
| cust_002 | ord_2001 ($89.00, delivered) |
| cust_003 | ord_3001 ($54.50, processing), ord_3002 ($45.00, already fully refunded) |

## Characterize the real model's behavior

Once Ollama is running, this exercises the live model against a small fixed
ticket set (including abuse attempts) multiple times and asserts that the
safety invariants hold on every run, while reporting how often the model
actually resolves vs. escalates (which is expected to vary):

```bash
python -m eval.live_ollama_smoke --model qwen2.5:7b-instruct --runs 3
```

## Repo layout

```
src/support_agent/
  types.py             shared domain types
  agent/loop.py         orchestration loop + termination guarantees
  policy/engine.py       deterministic safety boundary (the core of this design)
  tools/                 mock backend, tool implementations, approval queue, executor
  llm/                   LLMClient interface + Ollama + Fake (scriptable) implementations
  audit/log.py           append-only structured audit trail
  service.py             composition root
  runner.py              concurrent ticket processing + rate limiting
  api.py                 FastAPI HTTP service
  dashboard.html          run-inspection UI served at /dashboard
tests/                   deterministic unit + integration tests (FakeLLMClient)
eval/
  fuzz_policy.py          randomized adversarial search over the policy engine
  live_ollama_smoke.py    statistical characterization of the real model
DESIGN.md                 architecture, safety reasoning, trade-offs
```
