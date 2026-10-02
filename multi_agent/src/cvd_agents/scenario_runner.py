"""Execute documented synthetic scenarios without an LLM provider."""

from __future__ import annotations

import json
from unittest.mock import patch

from .config import PROJECT_DIR, RUNS_DIR
from .workflow import OrchestratorAgent


def run_scenarios() -> dict:
    scenarios = json.loads((PROJECT_DIR / "scenarios/scenarios.json").read_text(encoding="utf-8"))
    results = []
    for case in scenarios:
        identifier = case["id"]
        if "question" in case["input"]:
            state = OrchestratorAgent().answer_question(case["input"]["question"], runs_dir=RUNS_DIR, synthetic=True)
        else:
            if identifier == "artifact_failure":
                with patch("cvd_agents.tools.validate_stage9_research_release", side_effect=ValueError("simulated SHA mismatch")):
                    state = OrchestratorAgent().assess(case["input"], runs_dir=RUNS_DIR, synthetic=True)
            else:
                state = OrchestratorAgent().assess(case["input"], runs_dir=RUNS_DIR, synthetic=True)
        status = state.status
        run_id = state.run_id
        events = [json.loads(line) for line in (RUNS_DIR / run_id / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        observed_agents = {row["agent"] for row in events if row["event_type"] == "agent_call"}
        load = json.loads((RUNS_DIR / run_id / "load_report.json").read_text(encoding="utf-8"))
        passed = status == case["expected_status"] and set(case["expected_agents"]).issubset(observed_agents)
        if status == "completed":
            passed = passed and load["passes_40_percent_rule"] and state.safety_review.passed
        results.append({"id": identifier, "expected_status": case["expected_status"], "actual_status": status, "passed": passed, "run_id": run_id})
    summary = {"total": len(results), "passed": sum(row["passed"] for row in results), "failed": sum(not row["passed"] for row in results), "results": results}
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "scenario_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = run_scenarios()
    print(f"{result['total']} scenarios, {result['passed']} passed, {result['failed']} failed")
    raise SystemExit(0 if result["failed"] == 0 else 1)
