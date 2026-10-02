"""Typed boundaries between the six workflow components."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool


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
    verified_pipeline_sha256: str


class EvidenceBundle(BaseModel):
    model_name: str
    model_id: str
    development_source: str
    predictor_count: int
    metrics: dict[str, float]
    limitations: list[str]
    kazakhstan_validated: Literal[False] = False
    clinical_deployment_allowed: Literal[False] = False
    sources: list[str]


class ExplanationResult(BaseModel):
    text: str


class SafetyReview(BaseModel):
    passed: bool
    violations: list[str]
    final_response: str | None = None


class CVDWorkflowState(BaseModel):
    run_id: str
    user_request: str
    raw_input: dict[str, Any] | None = None
    validation: ValidationResult | None = None
    prediction: PredictionResult | None = None
    evidence: EvidenceBundle | None = None
    explanation: ExplanationResult | None = None
    safety_review: SafetyReview | None = None
    status: str = "started"
    errors: list[str] = Field(default_factory=list)
    revision_count: int = 0
