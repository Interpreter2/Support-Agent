"""
Structured, append-only audit trail.

Design choice: every event is a flat dict written as one JSON line
(JSONL). This is deliberately boring — greppable, diffable, append-only,
and trivial to replay or feed into the eval harness. A human reviewing
"what did the agent do and why" reads this file top to bottom for a
run_id; they do not need to parse model free-text to find out whether
a refund was attempted.

We log the model's free-text reasoning too (when present) but treat it
as commentary, never as evidence of safety -- the safety story rests on
the policy_decision events below, which are produced by deterministic
code, not by the model.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.subscribers = []

    def _write(self, event: dict[str, Any]) -> None:
        event["ts"] = time.time()
        line = json.dumps(event, default=_json_default)
        with self._lock:
            with open(self.path, "a") as f:
                f.write(line + "\n")
        for sub in self.subscribers:
            sub(line)

    def log(self, run_id: str, ticket_id: str, event_type: str, **fields: Any) -> None:
        self._write({
            "run_id": run_id,
            "ticket_id": ticket_id,
            "event": event_type,
            **fields,
        })

    def events_for_run(self, run_id: str) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("run_id") == run_id:
                    out.append(rec)
        return out

    def all_events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]


def _json_default(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if hasattr(obj, "value"):  # Enum
        return obj.value
    return str(obj)
