from __future__ import annotations

from cvd_agents.deep_prototype import build_prototype_graph
from cvd_agents.prototype import OrchestratorAgent
from cvd_agents.tools import validate_patient_input
from langchain_core.language_models.fake_chat_models import FakeListChatModel


VALID = {
    "age": 58,
    "sex": "male",
    "bmi": 28.1,
    "smoking_status": "former",
    "hypertension_dx": True,
    "diabetes_dx": False,
    "general_health": "good",
}


def test_two_agents_exchange_typed_data_using_real_locked_model():
    result = OrchestratorAgent().assess(VALID)
    assert result.status == "completed"
    assert result.validation.valid
    assert result.validation.normalized_input.age == 58
    assert result.prediction.model_id == "minimal_v1"
    assert 0 < result.prediction.probability < 1


def test_invalid_input_stops_before_prediction():
    result = OrchestratorAgent().assess({**VALID, "age": 39})
    assert result.status == "invalid_input"
    assert result.prediction is None


def test_missing_field_is_not_fabricated():
    data = VALID.copy()
    del data["bmi"]
    result = validate_patient_input(data)
    assert result.valid is False
    assert result.normalized_input is None


def test_deep_agents_prototype_configuration_builds_without_api_call():
    graph = build_prototype_graph(FakeListChatModel(responses=["research only"]))
    assert graph.name == "orchestrator_agent"
