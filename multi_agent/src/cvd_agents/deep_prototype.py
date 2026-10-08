"""Deep Agents configuration for the two implemented specialist roles."""

from __future__ import annotations

import json
import os
from typing import Any

from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool

from .tools import predict_cvd_mortality_risk, validate_patient_input


@tool
def validate_input_tool(patient_json: str) -> dict[str, Any]:
    """Validate seven supplied patient fields against the release schema."""
    raw = json.loads(patient_json)
    if not isinstance(raw, dict):
        return {"valid": False, "errors": ["Expected a JSON object"]}
    return validate_patient_input(raw).model_dump()


@tool
def locked_prediction_tool(patient_json: str) -> dict[str, Any]:
    """Predict with the verified locked pipeline, never with the LLM."""
    raw = json.loads(patient_json)
    if not isinstance(raw, dict):
        return {"error": "Expected a JSON object"}
    validation = validate_patient_input(raw)
    if not validation.valid or validation.normalized_input is None:
        return {"error": "Invalid patient input", "details": validation.errors}
    return predict_cvd_mortality_risk(validation.normalized_input).model_dump()


def build_prototype_graph(model: str | BaseChatModel | None = None):
    """Optional LLM-driven prototype; requires a configured model to invoke."""
    selected = model or os.getenv("AGENT_MODEL", "openai:gpt-5.1")
    return create_deep_agent(
        model=selected,
        tools=[],
        subagents=[
            {
                "name": "input_validation_agent",
                "description": "Validate supplied patient fields without filling missing values.",
                "system_prompt": "Use validate_input_tool and return structured validation. Do not invent inputs.",
                "tools": [validate_input_tool],
                "model": selected,
            },
            {
                "name": "risk_prediction_agent",
                "description": "Run the existing locked model for validated data.",
                "system_prompt": "Use locked_prediction_tool. Never calculate or guess risk yourself.",
                "tools": [locked_prediction_tool],
                "model": selected,
            },
        ],
        system_prompt="You are orchestrator_agent. Coordinate validation then prediction. Do no domain calculations. This is research only.",
        name="orchestrator_agent",
    )
