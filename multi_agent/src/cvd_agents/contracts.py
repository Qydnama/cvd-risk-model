"""Typed messages exchanged between agents and the explicit session state."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool

FEATURES: tuple[str, ...] = (
    "age",
    "sex",
    "bmi",
    "smoking_status",
    "hypertension_dx",
    "diabetes_dx",
    "general_health",
)
FeatureName = Literal[
    "age", "sex", "bmi", "smoking_status", "hypertension_dx", "diabetes_dx", "general_health"
]


# --- Assignment 2 contracts (unchanged) -------------------------------------------------


class PatientInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float = Field(strict=True)
    sex: str
    bmi: float = Field(strict=True)
    smoking_status: str
    hypertension_dx: StrictBool
    diabetes_dx: StrictBool
    general_health: str


class ValidationResult(BaseModel):
    valid: bool
    errors: list[str]
    normalized_input: PatientInput | None = None


class PredictionResult(BaseModel):
    model_id: Literal["minimal_v1"] = "minimal_v1"
    outcome: Literal["five_year_cvd_mortality"] = "five_year_cvd_mortality"
    probability: float = Field(ge=0, le=1)
    probability_percent: float = Field(ge=0, le=100)
    research_only: Literal[True] = True


class PrototypeState(BaseModel):
    """State of the single-shot ``assess`` command kept from Assignment 2."""

    status: Literal["started", "invalid_input", "completed", "error"] = "started"
    validation: ValidationResult | None = None
    prediction: PredictionResult | None = None
    error: str | None = None


# --- Orchestrator -------------------------------------------------------------------------

Intent = Literal[
    "calculate",
    "provide_features",
    "confirm",
    "reject",
    "question_model",
    "question_cvd_general",
    "question_result",
    "change_features",
    "reset",
    "out_of_scope",
]


class RoutingDecision(BaseModel):
    """Structured output of the orchestrator LLM call."""

    intent: Intent = Field(description="One intent from the allowed list")
    reason: str = Field(default="", description="One short sentence, no medical content")


# --- Input validation agent ---------------------------------------------------------------


class ExtractedValue(BaseModel):
    field: FeatureName
    value: str = Field(description="Value exactly as stated by the user, as text")
    quote: str = Field(description="Exact words from the user message that state this value")


class FeatureExtraction(BaseModel):
    """Values the user explicitly stated in the current message. Nothing inferred."""

    values: list[ExtractedValue] = Field(default_factory=list)


class FieldState(BaseModel):
    value: Any
    source_turn: int
    quote: str


class InputOutcome(BaseModel):
    accepted: list[str] = Field(default_factory=list)
    changed: list[str] = Field(default_factory=list)
    rejected_quotes: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    validation: ValidationResult | None = None
    reply: str = ""
    ready_for_confirmation: bool = False


# --- Evidence, explanation, safety -------------------------------------------------------

EvidenceKind = Literal["model", "medical", "result"]


class EvidenceFact(BaseModel):
    fact_id: str
    source_id: str
    source_path: str
    locator: str
    kind: EvidenceKind
    text: str


class EvidenceBundle(BaseModel):
    question: str
    question_kind: Literal["model", "medical", "result"]
    facts: list[EvidenceFact] = Field(default_factory=list)

    @property
    def fact_ids(self) -> set[str]:
        return {fact.fact_id for fact in self.facts}


class ExplanationDraft(BaseModel):
    text: str = Field(description="Answer in Russian. Use {probability_percent} placeholder for the model result")
    used_fact_ids: list[str] = Field(default_factory=list, description="IDs of facts the text relies on")


class SafetyIssue(BaseModel):
    check: str
    detail: str


class SafetyJudgement(BaseModel):
    """Structured output of the safety LLM review."""

    acceptable: bool
    issues: list[str] = Field(default_factory=list)


class SafetyReview(BaseModel):
    verdict: Literal["approved", "revise", "blocked"]
    issues: list[SafetyIssue] = Field(default_factory=list)
    revisions: int = 0
    rejected_draft: str | None = None  # kept for audit when the draft is not approved


# --- Session state ------------------------------------------------------------------------

Phase = Literal["idle", "collecting", "awaiting_confirmation", "result_ready"]
TurnStatus = Literal[
    "answered", "needs_input", "awaiting_confirmation", "refused", "error", "limit_exceeded"
]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    text: str


class SessionState(BaseModel):
    session_id: str
    turn: int = 0
    phase: Phase = "idle"
    features: dict[str, FieldState] = Field(default_factory=dict)
    invalid_attempts: dict[str, int] = Field(default_factory=dict)
    prediction: PredictionResult | None = None
    prediction_features_hash: str | None = None
    history: list[ChatMessage] = Field(default_factory=list)

    def feature_values(self) -> dict[str, Any]:
        return {name: state.value for name, state in self.features.items()}

    def features_hash(self) -> str:
        payload = json.dumps(self.feature_values(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def prediction_is_current(self) -> bool:
        return self.prediction is not None and self.prediction_features_hash == self.features_hash()

    def sanitized(self) -> dict[str, Any]:
        """Snapshot for logs: field names and statuses only, no health values or text."""
        return {
            "session_id": self.session_id,
            "turn": self.turn,
            "phase": self.phase,
            "known_features": sorted(self.features),
            "has_prediction": self.prediction is not None,
            "prediction_is_current": self.prediction_is_current(),
            "history_length": len(self.history),
        }


class TurnResult(BaseModel):
    session_id: str
    turn: int
    intent: Intent | None
    status: TurnStatus
    reply: str
    phase: Phase
    prediction: PredictionResult | None = None
    safety: SafetyReview | None = None
    evidence_fact_ids: list[str] = Field(default_factory=list)
    agents_called: list[str] = Field(default_factory=list)
    error: str | None = None
