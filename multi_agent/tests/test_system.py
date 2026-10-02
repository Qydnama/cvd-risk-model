from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from cvd_agents.config import REPO_ROOT
from cvd_agents.contracts import PatientInput
from cvd_agents.deep import build_agent_graph, run_agent_assessment
from cvd_agents.tools import (
    ArtifactVerificationError,
    check_research_only_constraints,
    get_model_evidence,
    predict_cvd_mortality_risk,
    validate_patient_input,
    verify_model_artifacts,
)
from cvd_agents.workflow import OrchestratorAgent, calculate_agent_load
from cvd_risk.modeling.stage9_research_release import validate_minimal_v1_research_input


@pytest.fixture
def patient_data():
    return {"age": 58, "sex": "male", "bmi": 28.1, "smoking_status": "former", "hypertension_dx": True, "diabetes_dx": False, "general_health": "good"}


def test_schema_rejects_missing_invalid_and_fabrication(patient_data):
    assert validate_patient_input(patient_data).valid
    for mutation in ({"age": 39}, {"bmi": float("nan")}, {"smoking_status": "sometimes"}, {"hypertension_dx": "yes"}):
        assert not validate_patient_input({**patient_data, **mutation}).valid
    incomplete = patient_data.copy()
    del incomplete["bmi"]
    result = validate_patient_input(incomplete)
    assert not result.valid and result.normalized_input is None


def test_real_model_matches_canonical_calibrated_path(patient_data):
    patient = PatientInput.model_validate(patient_data)
    result = predict_cvd_mortality_risk(patient)
    schema = json.loads((REPO_ROOT / "artifacts/minimal_v1_research_release/input_schema.json").read_text(encoding="utf-8"))
    frame = pd.DataFrame([[getattr(patient, key) for key in schema["ordered_features"]]], columns=schema["ordered_features"])
    validate_minimal_v1_research_input(frame, schema)
    model = joblib.load(REPO_ROOT / "artifacts/minimal_v1/full_development_pipeline.joblib")
    expected = float(model.predict_proba(frame, calibrated=True)[0])
    assert result.probability == pytest.approx(expected, abs=1e-15)
    assert result.probability_percent == pytest.approx(expected * 100)
    assert np.isfinite(result.probability)


def test_artifact_failure_is_hard_error():
    with patch("cvd_agents.tools.validate_stage9_research_release", side_effect=ValueError("SHA mismatch")):
        with pytest.raises(ArtifactVerificationError, match="SHA mismatch"):
            verify_model_artifacts()


def test_evidence_is_repository_sourced():
    evidence = get_model_evidence()
    assert evidence.metrics["development_weighted_roc_auc"] == 0.82156
    assert evidence.metrics["development_weighted_brier_score"] == 0.00897681
    assert evidence.kazakhstan_validated is False
    assert evidence.clinical_deployment_allowed is False


def test_safety_rejects_treatment_and_false_claim(patient_data):
    prediction = predict_cvd_mortality_risk(PatientInput.model_validate(patient_data))
    evidence = get_model_evidence()
    bad = f"Five-year cardiovascular mortality is {prediction.probability_percent:.4f}%. Take medication. Clinically validated for Kazakhstan. Research only."
    assert "unsafe_medical_or_threshold_claim" in check_research_only_constraints(bad, prediction, evidence)


def test_workflow_logs_sanitized_state_and_balanced_load(patient_data, tmp_path):
    state = OrchestratorAgent().assess(patient_data, runs_dir=tmp_path, synthetic=True)
    assert state.status == "completed"
    folder = tmp_path / state.run_id
    snapshot = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    events_text = (folder / "events.jsonl").read_text(encoding="utf-8")
    assert snapshot["synthetic"] is True
    assert "28.1" not in json.dumps(snapshot) + events_text
    assert "male" not in json.dumps(snapshot) + events_text
    load = json.loads((folder / "load_report.json").read_text(encoding="utf-8"))
    assert load["passes_40_percent_rule"] and load["max_share"] <= 0.4
    assert len(load["agents"]) == 6


def test_workflow_invalid_input_stops_before_prediction(patient_data, tmp_path):
    del patient_data["bmi"]
    state = OrchestratorAgent().assess(patient_data, runs_dir=tmp_path)
    assert state.status == "invalid_input"
    assert state.prediction is None


def test_review_limit_is_one(patient_data, tmp_path):
    from cvd_agents.contracts import SafetyReview
    with patch("cvd_agents.workflow.SafetyReviewAgent.run", return_value=SafetyReview(passed=False, violations=["unsafe"])) as review:
        state = OrchestratorAgent().assess(patient_data, runs_dir=tmp_path)
    assert state.status == "safety_failed"
    assert state.revision_count == 1
    assert review.call_count == 2


def test_deep_agent_graph_builds_without_provider_key():
    class ToolAwareFake(FakeListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    graph = build_agent_graph(ToolAwareFake(responses=["research only"]))
    assert graph.name == "orchestrator_agent"
    result = graph.invoke({"messages": [{"role": "user", "content": "State the purpose."}]}, config={"recursion_limit": 12})
    assert result["messages"][-1].content == "research only"


def test_agent_assessment_requires_all_five_delegations(patient_data, tmp_path):
    state = OrchestratorAgent().assess(patient_data, runs_dir=tmp_path, synthetic=True)
    names = ["input_validation_agent", "risk_prediction_agent", "evidence_agent", "explanation_agent", "safety_review_agent"]

    class FakeGraph:
        async def ainvoke(self, *args, **kwargs):
            return {"messages": [SimpleNamespace(tool_calls=[{"name": "task", "args": {"subagent_type": name}} for name in names])]}

    with patch("cvd_agents.deep.build_agent_graph", return_value=FakeGraph()):
        result = run_agent_assessment(state)
    assert result["delegated_agents"] == names
    assert result["final_response"] == state.safety_review.final_response
    names.pop()
    with patch("cvd_agents.deep.build_agent_graph", return_value=FakeGraph()):
        with pytest.raises(RuntimeError, match="delegation incomplete"):
            run_agent_assessment(state)


def test_all_scenario_definitions():
    scenarios = json.loads((Path(__file__).parents[1] / "scenarios/scenarios.json").read_text(encoding="utf-8"))
    assert len(scenarios) >= 12
    assert len({case["id"] for case in scenarios}) == len(scenarios)
    assert all(case["expected_agents"] and case["expected_status"] and case["expected_behavior"] for case in scenarios)
