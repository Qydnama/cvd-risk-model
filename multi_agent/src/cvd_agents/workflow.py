"""Bounded six-component research workflow and privacy-preserving run records."""

from __future__ import annotations

import json
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

from .config import MAX_AGENT_STEPS, MAX_REVIEW_REVISIONS, RUNS_DIR, TOTAL_RUN_TIMEOUT_SECONDS
from .contracts import CVDWorkflowState, EvidenceBundle, ExplanationResult, PatientInput, PredictionResult, SafetyReview
from .tools import check_research_only_constraints, get_model_evidence, predict_cvd_mortality_risk, validate_patient_input

T = TypeVar("T")


class RunRecorder:
    def __init__(self, run_id: str, runs_dir: Path = RUNS_DIR, synthetic: bool = False):
        self.run_id = run_id
        self.synthetic = synthetic
        self.path = runs_dir / run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.events: list[dict[str, Any]] = []

    def call(self, agent: str, fn: Callable[[], T], tool: str | None = None) -> T:
        start = time.monotonic()
        status = "success"
        error_type = None
        try:
            return fn()
        except Exception as exc:
            status = "error"
            error_type = type(exc).__name__
            raise
        finally:
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": self.run_id,
                "event_type": "tool_call" if tool else "agent_call",
                "agent": agent,
                "tool": tool,
                "status": status,
                "duration_ms": round((time.monotonic() - start) * 1000, 2),
                "error_type": error_type,
            }
            self.events.append(event)
            with (self.path / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")

    def save(self, state: CVDWorkflowState) -> None:
        # No request, patient fields, answer text, or individual prediction are persisted.
        snapshot = {
            "run_id": state.run_id,
            "status": state.status,
            "validation_passed": state.validation.valid if state.validation else None,
            "prediction_verified": state.prediction is not None,
            "evidence_loaded": state.evidence is not None,
            "safety_passed": state.safety_review.passed if state.safety_review else None,
            "revision_count": state.revision_count,
            "error_types": state.errors,
            "synthetic": self.synthetic,
        }
        (self.path / "state.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        (self.path / "load_report.json").write_text(json.dumps(calculate_agent_load(self.events), indent=2), encoding="utf-8")


def calculate_agent_load(events: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(event["agent"] for event in events if event["event_type"] in {"agent_call", "tool_call"})
    total = sum(counts.values())
    agents = {name: {"calls": count, "share": count / total} for name, count in sorted(counts.items())} if total else {}
    maximum = max((item["share"] for item in agents.values()), default=0.0)
    return {"total_calls": total, "agents": agents, "max_share": maximum, "passes_40_percent_rule": maximum <= 0.4}


class InputValidationAgent:
    name = "input_validation_agent"
    tools = ("validate_patient_input",)

    def run(self, raw: dict[str, Any], log: RunRecorder):
        return log.call(self.name, lambda: log.call(self.name, lambda: validate_patient_input(raw), "validate_patient_input"))


class RiskPredictionAgent:
    name = "risk_prediction_agent"
    tools = ("predict_cvd_mortality_risk",)

    def run(self, patient: PatientInput, log: RunRecorder) -> PredictionResult:
        return log.call(self.name, lambda: log.call(self.name, lambda: predict_cvd_mortality_risk(patient), "predict_cvd_mortality_risk"))


class EvidenceAgent:
    name = "evidence_agent"
    tools = ("get_model_evidence",)

    def run(self, log: RunRecorder) -> EvidenceBundle:
        return log.call(self.name, lambda: log.call(self.name, get_model_evidence, "get_model_evidence"))


class ExplanationAgent:
    name = "explanation_agent"
    tools: tuple[str, ...] = ()

    def run(self, patient: PatientInput, prediction: PredictionResult, evidence: EvidenceBundle, log: RunRecorder) -> ExplanationResult:
        def explain() -> ExplanationResult:
            text = (
                f"The locked {evidence.model_id} research model estimated a five-year cardiovascular mortality "
                f"probability of {prediction.probability_percent:.4f}%. "
                "Inputs used: age, sex, BMI, smoking status, prior hypertension and diabetes diagnoses, "
                "and self-rated general health. This is a model estimate, not a diagnosis or treatment recommendation. "
                f"Development evidence: {evidence.development_source}; temporal and external validation are "
                "underpowered and exploratory. The model is not validated for Kazakhstan. "
                "This system is for research and educational use only."
            )
            return ExplanationResult(text=text)
        return log.call(self.name, explain)


class SafetyReviewAgent:
    name = "safety_review_agent"
    tools = ("check_research_only_constraints",)

    def run(self, explanation: ExplanationResult, prediction: PredictionResult, evidence: EvidenceBundle, log: RunRecorder) -> SafetyReview:
        def review() -> SafetyReview:
            violations = log.call(self.name, lambda: check_research_only_constraints(explanation.text, prediction, evidence), "check_research_only_constraints")
            return SafetyReview(passed=not violations, violations=violations, final_response=explanation.text if not violations else None)
        return log.call(self.name, review)

    def review_information(self, text: str, log: RunRecorder) -> SafetyReview:
        def review() -> SafetyReview:
            lower = text.lower()
            violations = []
            if "research" not in lower:
                violations.append("missing_research_limitation")
            if any(phrase in lower for phrase in ("take medication", "start medication", "stop medication", "clinically validated")):
                violations.append("unsafe_medical_claim")
            return SafetyReview(passed=not violations, violations=violations, final_response=text if not violations else None)
        return log.call(self.name, review)


class OrchestratorAgent:
    name = "orchestrator_agent"
    tools: tuple[str, ...] = ()

    def __init__(self):
        self.input_agent = InputValidationAgent()
        self.prediction_agent = RiskPredictionAgent()
        self.evidence_agent = EvidenceAgent()
        self.explanation_agent = ExplanationAgent()
        self.safety_agent = SafetyReviewAgent()

    def assess(self, raw: dict[str, Any], user_request: str = "Estimate research risk", runs_dir: Path = RUNS_DIR, synthetic: bool = False) -> CVDWorkflowState:
        run_id = uuid.uuid4().hex
        log = RunRecorder(run_id, runs_dir, synthetic)
        state = CVDWorkflowState(run_id=run_id, user_request=user_request, raw_input=raw)
        started = time.monotonic()
        steps = 0

        def step(fn: Callable[[], T]) -> T:
            nonlocal steps
            steps += 1
            if steps > MAX_AGENT_STEPS or time.monotonic() - started > TOTAL_RUN_TIMEOUT_SECONDS:
                raise TimeoutError("Workflow step or time limit exceeded")
            return fn()

        try:
            def coordinate() -> None:
                state.validation = step(lambda: self.input_agent.run(raw, log))
                if not state.validation.valid or state.validation.normalized_input is None:
                    state.status = "invalid_input"
                    state.errors = state.validation.errors
                    return
                patient = state.validation.normalized_input
                state.prediction = step(lambda: self.prediction_agent.run(patient, log))
                state.evidence = step(lambda: self.evidence_agent.run(log))
                for revision in range(MAX_REVIEW_REVISIONS + 1):
                    state.explanation = step(lambda: self.explanation_agent.run(patient, state.prediction, state.evidence, log))
                    state.safety_review = step(lambda: self.safety_agent.run(state.explanation, state.prediction, state.evidence, log))
                    if state.safety_review.passed:
                        state.status = "completed"
                        return
                    if revision < MAX_REVIEW_REVISIONS:
                        state.revision_count += 1
                state.status = "safety_failed"
                state.errors = state.safety_review.violations
            log.call(self.name, coordinate)
        except Exception as exc:
            state.status = "error"
            state.errors = [f"{type(exc).__name__}: {exc}"]
        finally:
            log.save(state)
        return state

    def answer_question(self, question: str, runs_dir: Path = RUNS_DIR, synthetic: bool = False) -> CVDWorkflowState:
        run_id = uuid.uuid4().hex
        log = RunRecorder(run_id, runs_dir, synthetic)
        state = CVDWorkflowState(run_id=run_id, user_request=question)
        try:
            def coordinate() -> None:
                lower = question.lower()
                if any(word in lower for word in ("medication", "treatment", "medicine", "drug")):
                    draft = "I cannot recommend medication or treatment from this research model. This system is for research and educational use only."
                    state.status = "refused"
                else:
                    state.evidence = self.evidence_agent.run(log)
                    evidence = state.evidence
                    if "kazakhstan" in lower:
                        draft = "The model has not been validated for Kazakhstan. Its temporal and external validation is underpowered and exploratory. This system is for research and educational use only."
                    elif any(word in lower for word in ("auc", "brier", "metric", "performance")):
                        draft = (f"NHIS development survey-weighted ROC-AUC: {evidence.metrics['development_weighted_roc_auc']}; "
                                 f"Brier score: {evidence.metrics['development_weighted_brier_score']}. "
                                 "These are development results, not clinical validation. This system is for research and educational use only.")
                    else:
                        draft = (f"{evidence.model_id} is a research model developed on {evidence.development_source}. "
                                 "This system is for research and educational use only.")
                    state.status = "answered"
                state.safety_review = self.safety_agent.review_information(draft, log)
                if not state.safety_review.passed:
                    state.status = "safety_failed"
                    state.errors = state.safety_review.violations
            log.call(self.name, coordinate)
        except Exception as exc:
            state.status = "error"
            state.errors = [f"{type(exc).__name__}: {exc}"]
        finally:
            log.save(state)
        return state
