"""Two implemented agents exchanging typed data through a coordinator."""

from __future__ import annotations

from typing import Any

from .contracts import PatientInput, PredictionResult, PrototypeState, ValidationResult
from .tools import predict_cvd_mortality_risk, validate_patient_input


class InputValidationAgent:
    name = "input_validation_agent"
    tools = ("load_input_schema", "validate_patient_input")

    def run(self, raw: dict[str, Any]) -> ValidationResult:
        return validate_patient_input(raw)


class RiskPredictionAgent:
    name = "risk_prediction_agent"
    tools = ("predict_cvd_mortality_risk",)

    def run(self, patient: PatientInput) -> PredictionResult:
        return predict_cvd_mortality_risk(patient)


class OrchestratorAgent:
    """Only coordinates. No validation, prediction, or explanation logic."""

    name = "orchestrator_agent"
    tools: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.input_agent = InputValidationAgent()
        self.prediction_agent = RiskPredictionAgent()

    def assess(self, raw: dict[str, Any]) -> PrototypeState:
        state = PrototypeState()
        try:
            state.validation = self.input_agent.run(raw)
            if not state.validation.valid or state.validation.normalized_input is None:
                state.status = "invalid_input"
                return state
            state.prediction = self.prediction_agent.run(state.validation.normalized_input)
            state.status = "completed"
        except Exception as exc:
            state.status = "error"
            state.error = f"{type(exc).__name__}: {exc}"
        return state
