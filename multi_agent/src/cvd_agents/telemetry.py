"""Run logging: agent spans, LLM calls and tool calls, plus workload shares.

Events never contain feature values, user text, model output text or secrets.
LLM and tool events are captured by a LangChain callback handler, so calls made
inside framework agents (``create_agent``) are logged the same way as direct ones.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

COUNTED_KINDS = ("llm", "tool")


class RunRecorder:
    def __init__(self, session_id: str, runs_dir: Path | None, persist: bool = True):
        self.session_id = session_id
        self.turn = 0
        self.events: list[dict[str, Any]] = []
        self.path = (runs_dir / session_id) if (runs_dir is not None and persist) else None
        if self.path is not None:
            self.path.mkdir(parents=True, exist_ok=True)

    def log(self, *, agent: str, kind: str, name: str, status: str, duration_ms: float = 0.0, **extra: Any) -> None:
        event = {
            "ts": round(time.time(), 3),
            "session_id": self.session_id,
            "turn": self.turn,
            "agent": agent,
            "kind": kind,
            "name": name,
            "status": status,
            "duration_ms": round(duration_ms, 1),
            **{k: v for k, v in extra.items() if v is not None},
        }
        self.events.append(event)
        if self.path is not None:
            with (self.path / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    @contextmanager
    def agent_span(self, agent: str, name: str = "run") -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        except Exception as exc:
            self.log(agent=agent, kind="agent", name=name, status="error",
                     duration_ms=(time.perf_counter() - started) * 1000, error_type=type(exc).__name__)
            raise
        self.log(agent=agent, kind="agent", name=name, status="ok",
                 duration_ms=(time.perf_counter() - started) * 1000)

    def save_json(self, filename: str, payload: Any) -> None:
        if self.path is not None:
            (self.path / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def calculate_agent_load(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Share of LLM + tool calls per agent. Agent spans are reported but not counted."""
    counted = [e for e in events if e["kind"] in COUNTED_KINDS]
    total = len(counted)
    per_agent = Counter(e["agent"] for e in counted)
    by_kind = Counter((e["agent"], e["kind"]) for e in counted)
    spans = Counter(e["agent"] for e in events if e["kind"] == "agent")
    agents = {
        name: {
            "llm_calls": by_kind[(name, "llm")],
            "tool_calls": by_kind[(name, "tool")],
            "calls": count,
            "share": round(count / total, 4) if total else 0.0,
            "agent_invocations": spans[name],
        }
        for name, count in sorted(per_agent.items())
    }
    for name, count in spans.items():
        agents.setdefault(name, {"llm_calls": 0, "tool_calls": 0, "calls": 0, "share": 0.0, "agent_invocations": count})
    max_share = max((a["share"] for a in agents.values()), default=0.0)
    return {
        "total_llm_and_tool_calls": total,
        "llm_calls": sum(1 for e in counted if e["kind"] == "llm"),
        "tool_calls": sum(1 for e in counted if e["kind"] == "tool"),
        "failed_calls": sum(1 for e in counted if e["status"] != "ok"),
        "agents": agents,
        "max_share": max_share,
        "passes_40_percent_rule": max_share <= 0.4,
    }


def _usage(response: Any) -> dict[str, int] | None:
    try:
        message = response.generations[0][0].message
        usage = getattr(message, "usage_metadata", None)
        if usage:
            return {"input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0)}
    except (AttributeError, IndexError, TypeError):
        pass
    return None


class TelemetryCallback(BaseCallbackHandler):
    """Writes one event per LLM call and per tool call, attributed via run metadata."""

    def __init__(self, recorder: RunRecorder):
        self.recorder = recorder
        self._open: dict[UUID, tuple[str, str, str, float]] = {}

    def _start(self, run_id: UUID, kind: str, name: str, metadata: dict[str, Any] | None) -> None:
        agent = (metadata or {}).get("agent", "unattributed")
        self._open[run_id] = (agent, kind, name, time.perf_counter())

    def _end(self, run_id: UUID, status: str, **extra: Any) -> None:
        agent, kind, name, started = self._open.pop(run_id, ("unattributed", "unknown", "unknown", time.perf_counter()))
        self.recorder.log(agent=agent, kind=kind, name=name, status=status,
                          duration_ms=(time.perf_counter() - started) * 1000, **extra)

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):  # noqa: ANN001
        name = (kwargs.get("invocation_params") or {}).get("model") or (serialized or {}).get("name") or "chat_model"
        self._start(run_id, "llm", str(name), metadata)

    def on_llm_start(self, serialized, prompts, *, run_id, metadata=None, **kwargs):  # noqa: ANN001
        self._start(run_id, "llm", str((serialized or {}).get("name", "llm")), metadata)

    def on_llm_end(self, response, *, run_id, **kwargs):  # noqa: ANN001
        self._end(run_id, "ok", tokens=_usage(response))

    def on_llm_error(self, error, *, run_id, **kwargs):  # noqa: ANN001
        self._end(run_id, "error", error_type=type(error).__name__)

    def on_tool_start(self, serialized, input_str, *, run_id, metadata=None, **kwargs):  # noqa: ANN001
        self._start(run_id, "tool", str((serialized or {}).get("name") or kwargs.get("name") or "tool"), metadata)

    def on_tool_end(self, output, *, run_id, **kwargs):  # noqa: ANN001
        self._end(run_id, "ok")

    def on_tool_error(self, error, *, run_id, **kwargs):  # noqa: ANN001
        self._end(run_id, "error", error_type=type(error).__name__)
