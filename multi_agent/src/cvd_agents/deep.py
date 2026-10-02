"""Optional Deep Agents interface; deterministic tools remain authoritative."""

from __future__ import annotations

import os
import asyncio
import json
from typing import Any

from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool

from .tools import get_model_evidence, predict_cvd_mortality_risk, validate_patient_input
from .config import TOTAL_RUN_TIMEOUT_SECONDS
from .contracts import CVDWorkflowState


@tool
def validate_patient_input_tool(patient_json: str) -> dict[str, Any]:
    """Validate all seven patient fields against the locked research schema."""
    import json

    try:
        raw = json.loads(patient_json)
    except json.JSONDecodeError:
        return {"valid": False, "errors": ["Invalid JSON"]}
    if not isinstance(raw, dict):
        return {"valid": False, "errors": ["Expected a JSON object"]}
    return validate_patient_input(raw).model_dump()


@tool
def predict_cvd_mortality_risk_tool(patient_json: str) -> dict[str, Any]:
    """Verify and run the locked minimal_v1 pipeline for validated patient JSON."""
    import json

    raw = json.loads(patient_json)
    validation = validate_patient_input(raw)
    if not validation.valid or validation.normalized_input is None:
        return {"error": "Invalid patient input", "details": validation.errors}
    return predict_cvd_mortality_risk(validation.normalized_input).model_dump()


@tool
def get_model_evidence_tool() -> dict[str, Any]:
    """Read repository model identity, development metrics, and limitations."""
    return get_model_evidence().model_dump()


def build_agent_graph(model: str | BaseChatModel | None = None):
    """Build six named Deep Agents roles with narrowly assigned domain tools.

    The graph is for interactive exploration. `OrchestratorAgent.assess` performs
    the enforced, auditable deterministic assessment and final safety gate.
    """
    selected = model or os.getenv("AGENT_MODEL", "openai:gpt-5.1")
    subagents = [
        {"name": "input_validation_agent", "description": "Validate all seven supplied patient fields; never fill missing values.", "system_prompt": "Return only validated structured input or errors. Use the validation tool.", "tools": [validate_patient_input_tool], "model": selected},
        {"name": "risk_prediction_agent", "description": "Run deterministic locked model inference only.", "system_prompt": "Use the prediction tool. Never estimate numerical risk yourself.", "tools": [predict_cvd_mortality_risk_tool], "model": selected},
        {"name": "evidence_agent", "description": "Fetch model facts and limitations from repository evidence.", "system_prompt": "Use the evidence tool. Never invent study facts.", "tools": [get_model_evidence_tool], "model": selected},
        {"name": "explanation_agent", "description": "Draft plain research explanation from verified structured prediction and evidence.", "system_prompt": "Explain five-year cardiovascular mortality only. No diagnosis, treatment, causal attribution, or clinical validation claim.", "tools": [], "model": selected},
        {"name": "safety_review_agent", "description": "Review research-only output for medical and evidence constraints.", "system_prompt": "Reject unsupported clinical advice, false Kazakhstan validation, wrong outcome, and unsupported metrics.", "tools": [], "model": selected},
    ]
    return create_deep_agent(
        model=selected,
        tools=[],
        subagents=subagents,
        system_prompt=(
            "You are orchestrator_agent. Coordinate subagents only. For risk assessments, "
            "delegate in order: validation, prediction, evidence, explanation, safety. "
            "Never calculate risk or invent evidence. Research use only."
        ),
        name="orchestrator_agent",
    )


def run_agent_assessment(state: CVDWorkflowState, model: str | BaseChatModel | None = None) -> dict[str, Any]:
    """Run optional LLM delegation after the locked deterministic assessment.

    The numerical result and final response are still taken from the verified
    deterministic workflow. Missing delegation is reported as failure.
    """
    if state.status != "completed" or not state.validation or not state.prediction or not state.evidence:
        raise ValueError("A completed verified assessment is required")
    graph = build_agent_graph(model)
    payload = {
        "patient": state.validation.normalized_input.model_dump(),
        "verified_prediction": state.prediction.model_dump(),
        "verified_evidence": state.evidence.model_dump(),
    }
    message = (
        "Perform a research-only assessment using the supplied verified data. "
        "Delegate once to each of the five named specialist agents in order. "
        "Numerical risk and evidence must match the supplied verified data. "
        f"Structured data: {json.dumps(payload)}"
    )

    async def invoke():
        return await asyncio.wait_for(
            graph.ainvoke({"messages": [{"role": "user", "content": message}]}, config={"recursion_limit": 12}),
            timeout=TOTAL_RUN_TIMEOUT_SECONDS,
        )

    for attempt in range(2):
        try:
            result = asyncio.run(invoke())
            break
        except Exception as exc:
            transient = isinstance(exc, (TimeoutError, ConnectionError)) or getattr(exc, "status_code", None) in {429, 500, 502, 503, 504} or type(exc).__name__ in {"APIConnectionError", "APITimeoutError", "RateLimitError"}
            if attempt or not transient:
                raise
    delegated = []
    for item in result.get("messages", []):
        for call in getattr(item, "tool_calls", []):
            if call.get("name") == "task":
                specialist = call.get("args", {}).get("subagent_type")
                if specialist:
                    delegated.append(specialist)
    expected = {"input_validation_agent", "risk_prediction_agent", "evidence_agent", "explanation_agent", "safety_review_agent"}
    missing = sorted(expected - set(delegated))
    if missing:
        raise RuntimeError(f"Deep Agents delegation incomplete: {', '.join(missing)}")
    return {"delegated_agents": delegated, "final_response": state.safety_review.final_response}
