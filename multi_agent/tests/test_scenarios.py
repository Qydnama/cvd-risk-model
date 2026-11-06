"""Scenario suite. Offline runs use the rule-based simulator; live runs need a real API key."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from cvd_agents.config import SCENARIOS_PATH, load_dotenv, model_spec
from cvd_agents.scenario_runner import run_scenario

SCENARIOS = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))["scenarios"]


def test_suite_size_meets_course_requirements():
    assert len(SCENARIOS) >= 10


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["id"] for s in SCENARIOS])
def test_scenario_offline(scenario, tmp_path: Path):
    result = run_scenario(scenario, "offline", out=tmp_path)
    assert result["passed"], result["failures"]


def test_typical_scenario_respects_40_percent_rule_offline(tmp_path: Path):
    typical = next(s for s in SCENARIOS if s["id"] == "typical_session")
    load = run_scenario(typical, "offline", out=tmp_path)["load"]
    assert load["passes_40_percent_rule"], load


@pytest.mark.live
def test_typical_scenario_live(tmp_path: Path):
    load_dotenv()
    if not model_spec() or not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        pytest.skip("No live LLM configured")
    typical = next(s for s in SCENARIOS if s["id"] == "typical_session")
    result = run_scenario(typical, "live", out=tmp_path)
    assert result["passed"], result["failures"]
