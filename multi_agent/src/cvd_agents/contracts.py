"""Structured messages for the Assignment 2 prototype."""

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


class PrototypeState(BaseModel):
    """Explicit in-memory task state; persistence is planned for Assignment 4."""

    status: Literal["started", "invalid_input", "completed", "error"] = "started"
    validation: ValidationResult | None = None
    prediction: PredictionResult | None = None
    error: str | None = None
