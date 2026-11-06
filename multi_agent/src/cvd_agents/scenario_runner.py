"""Run dialogue scenarios offline (rule-based simulator) or live (real LLM API)."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, Literal

from . import tools as domain_tools
from .config import DEFAULT_LIMITS, SCENARIOS_PATH, Limits, model_spec, runs_dir
from .contracts import TurnResult
from .graph import CVDAssistant
from .llm import TransientLLMError, build_chat_model
from .offline_llm import looping_responder, simulated_llm
from .telemetry import calculate_agent_load

Mode = Literal["offline", "live"]


@contextmanager
def artifact_failure() -> Iterator[None]:
    """Make the Stage 9 verifier report FAIL (as on a SHA mismatch) without touching files."""
    original = domain_tools.validate_stage9_research_release
    domain_tools.verify_locked_release.cache_clear()
    domain_tools.validate_stage9_research_release = lambda root: {"validation_status": "FAIL"}
    try:
        yield
    finally:
        domain_tools.validate_stage9_research_release = original
        domain_tools.verify_locked_release.cache_clear()


def _build(scenario: dict[str, Any], mode: Mode, limits: Limits, out: Path) -> tuple[CVDAssistant, str]:
    fault = scenario.get("fault") or {}
    kind = fault.get("type")
    live = mode == "live" and not scenario.get("simulated_only")
    if kind == "step_limit":
        limits = replace(limits, max_agent_steps_per_turn=fault["max_agent_steps_per_turn"])
    limits = replace(limits, retry_base_delay_seconds=0.01 if not live else limits.retry_base_delay_seconds)
    if live:
        if kind == "llm_down":  # real provider call with an impossible timeout -> real timeout errors
            llm = build_chat_model(replace(limits, llm_timeout_seconds=0.001))
        else:
            llm = build_chat_model(limits)
        label = f"live:{model_spec()}"
    else:
        faults: list[Exception] = []
        if kind == "llm_transient":
            faults = [TransientLLMError("simulated 429/timeout")] * fault["count"]
        if kind == "llm_down":
            faults = [TransientLLMError("simulated outage")] * 50
        responder = looping_responder if kind == "evidence_loop" else None
        llm = simulated_llm(faults=faults, **({"responder": responder} if responder else {}))
        label = "simulated"
    return CVDAssistant(llm, limits=limits, session_id=scenario["id"], runs_path=out), label


def check(expect: dict[str, Any], result: TurnResult, events: list[dict[str, Any]], previous: TurnResult | None) -> list[str]:
    failures = []
    for key in ("status", "intent", "phase"):
        if key in expect and getattr(result, key) != expect[key]:
            failures.append(f"{key}={getattr(result, key)!r}, expected {expect[key]!r}")
    if "status_in" in expect and result.status not in expect["status_in"]:
        failures.append(f"status={result.status!r}, expected one of {expect['status_in']}")
    if "has_prediction" in expect and (result.prediction is not None) != expect["has_prediction"]:
        failures.append(f"has_prediction={result.prediction is not None}")
    for text in expect.get("reply_contains", []):
        if text.lower() not in result.reply.lower():
            failures.append(f"reply lacks {text!r}")
    for text in expect.get("reply_not_contains", []):
        if text.lower() in result.reply.lower():
            failures.append(f"reply contains forbidden {text!r}")
    if "min_facts" in expect and len(result.evidence_fact_ids) < expect["min_facts"]:
        failures.append(f"facts={len(result.evidence_fact_ids)}")
    if "facts_include" in expect and not any(expect["facts_include"] in f for f in result.evidence_fact_ids):
        failures.append(f"facts lack {expect['facts_include']!r}")
    if "safety_verdict" in expect and (result.safety is None or result.safety.verdict != expect["safety_verdict"]):
        failures.append(f"safety={result.safety.verdict if result.safety else None}")
    if "min_retries" in expect:
        retries = sum(1 for e in events if e["kind"] == "retry" and e["turn"] == result.turn)
        if retries < expect["min_retries"]:
            failures.append(f"retries={retries}")
    if expect.get("prediction_changed"):
        if previous is None or previous.prediction is None or result.prediction is None or \
                previous.prediction.probability == result.prediction.probability:
            failures.append("prediction did not change")
    return failures


def run_scenario(scenario: dict[str, Any], mode: Mode, limits: Limits = DEFAULT_LIMITS, out: Path | None = None) -> dict[str, Any]:
    out = out or runs_dir() / "scenarios" / mode
    assistant, label = _build(scenario, mode, limits, out)
    fault = (scenario.get("fault") or {}).get("type")
    turns: list[dict[str, Any]] = []
    failures: list[str] = []
    last_with_prediction: TurnResult | None = None
    expectations = scenario.get("expect_turns") or [{}] * (len(scenario["turns"]) - 1) + [scenario.get("expect", {})]
    with (artifact_failure() if fault == "artifact_failure" else _nullcontext()):
        for index, message in enumerate(scenario["turns"]):
            result = assistant.handle(message)
            problems = check(expectations[index], result, assistant.recorder.events, last_with_prediction)
            failures += [f"turn {index + 1}: {p}" for p in problems]
            if result.prediction is not None:
                last_with_prediction = result
            turns.append({
                "turn": result.turn, "user": message, "intent": result.intent, "status": result.status,
                "phase": result.phase, "agents_called": result.agents_called,
                "probability_percent": round(result.prediction.probability_percent, 4) if result.prediction else None,
                "safety_verdict": result.safety.verdict if result.safety else None,
                "safety_revisions": result.safety.revisions if result.safety else None,
                "rejected_draft": result.safety.rejected_draft if result.safety else None,
                "safety_issues": [i.detail for i in result.safety.issues] if result.safety else [],
                "evidence_fact_ids": result.evidence_fact_ids, "reply": result.reply, "error": result.error,
            })
    return {"id": scenario["id"], "title": scenario["title"], "llm": label, "passed": not failures,
            "failures": failures, "turns": turns, "load": calculate_agent_load(assistant.recorder.events)}


@contextmanager
def _nullcontext() -> Iterator[None]:
    yield


def run_all(mode: Mode, only: list[str] | None = None, out: Path | None = None) -> dict[str, Any]:
    spec = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    selected = [s for s in spec["scenarios"] if not only or s["id"] in only]
    results = [run_scenario(s, mode, out=out) for s in selected]
    typical = next((r for r in results if r["id"] == spec["typical_scenario_id"]), None)
    summary = {
        "mode": mode,
        "model": model_spec() if mode == "live" else "simulated (rule-based, not an LLM)",
        "total": len(results),
        "passed": sum(r["passed"] for r in results),
        "failed": [r["id"] for r in results if not r["passed"]],
        "typical_scenario_load": typical["load"] if typical else None,
        "results": results,
    }
    target = (out or runs_dir() / "scenarios" / mode)
    target.mkdir(parents=True, exist_ok=True)
    (target / "scenario_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
