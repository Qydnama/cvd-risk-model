"""LangChain tool wrappers for the input-validation and prediction agents.

The wrappers only adapt the deterministic functions in ``tools.py``; they add no logic.
Wrapping them as tools makes every call visible to the telemetry callback.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from .contracts import PatientInput
from .tools import (
    list_missing_fields,
    predict_cvd_mortality_risk,
    validate_partial_features,
    validate_patient_input,
    verify_locked_release,
)


@tool("validate_partial_features")
def validate_partial_features_tool(values: dict[str, str]) -> dict[str, Any]:
    """Normalize and check any subset of the seven fields against the release schema."""
    return validate_partial_features(values)


@tool("list_missing_fields")
def list_missing_fields_tool(known: list[str]) -> dict[str, Any]:
    """List required fields not yet provided, with allowed values."""
    return list_missing_fields(known)


@tool("validate_complete_input")
def validate_complete_input_tool(patient: dict[str, Any]) -> dict[str, Any]:
    """Validate all seven fields together with the strict release contract."""
    return validate_patient_input(patient).model_dump()


@tool("verify_locked_release")
def verify_locked_release_tool() -> dict[str, Any]:
    """Verify the Stage 9 research release (SHA-256, provenance, smoke test) before inference."""
    return verify_locked_release()


@tool("predict_cvd_mortality_risk")
def predict_cvd_mortality_risk_tool(patient: dict[str, Any]) -> dict[str, Any]:
    """Run the locked minimal_v1 pipeline on a validated patient. The LLM never computes risk."""
    return predict_cvd_mortality_risk(PatientInput.model_validate(patient)).model_dump()


INPUT_TOOLS = [validate_partial_features_tool, list_missing_fields_tool, validate_complete_input_tool]
PREDICTION_TOOLS = [verify_locked_release_tool, predict_cvd_mortality_risk_tool]
