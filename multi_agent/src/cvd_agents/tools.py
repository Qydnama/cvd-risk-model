"""Deterministic domain functions: the Stage 9 schema and the locked ML pipeline."""

from __future__ import annotations

import json
import math
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pydantic import ValidationError

from cvd_risk.modeling.stage9_research_release import (
    validate_minimal_v1_research_input,
    validate_stage9_research_release,
)

from .contracts import PatientInput, PredictionResult, ValidationResult

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "artifacts/minimal_v1_research_release/input_schema.json"
MODEL_PATH = REPO_ROOT / "artifacts/minimal_v1/full_development_pipeline.joblib"


def load_input_schema() -> dict[str, Any]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    if schema.get("model_id") != "minimal_v1" or len(schema.get("required", [])) != 7:
        raise ValueError("Unexpected model input schema")
    return schema


def validate_patient_input(raw: dict[str, Any]) -> ValidationResult:
    schema = load_input_schema()
    required = schema["required"]
    errors = [f"Missing field: {key}" for key in required if key not in raw]
    errors += [f"Unexpected field: {key}" for key in raw if key not in required]
    if errors:
        return ValidationResult(valid=False, errors=errors)
    try:
        patient = PatientInput.model_validate(raw)
    except ValidationError as exc:
        return ValidationResult(
            valid=False,
            errors=[f"Invalid type: {item['loc'][0]}" for item in exc.errors()],
        )
    for key, spec in schema["properties"].items():
        value = getattr(patient, key)
        if spec["type"] == "number":
            if not math.isfinite(value) or not spec["minimum"] <= value <= spec["maximum"]:
                errors.append(f"{key} must be between {spec['minimum']} and {spec['maximum']}")
        elif "enum" in spec and value not in spec["enum"]:
            errors.append(f"{key} must be one of: {', '.join(spec['enum'])}")
    if errors:
        return ValidationResult(valid=False, errors=errors)
    return ValidationResult(valid=True, errors=[], normalized_input=patient)


_TRUE = {"true", "yes", "да", "есть", "1"}
_FALSE = {"false", "no", "нет", "0"}


def normalize_and_check_field(name: str, raw_value: Any) -> tuple[Any, str | None]:
    """Convert one user-stated value to the schema type and check it. Never fills gaps."""
    spec = load_input_schema()["properties"].get(name)
    if spec is None:
        return None, f"Unknown field: {name}"
    text = str(raw_value).strip().lower()
    if spec["type"] == "number":
        try:
            value = float(text.replace(",", "."))
        except ValueError:
            return None, f"{name} must be a number"
        if not math.isfinite(value) or not spec["minimum"] <= value <= spec["maximum"]:
            return None, f"{name} must be between {spec['minimum']} and {spec['maximum']}"
        return value, None
    if spec["type"] == "boolean":
        if text in _TRUE:
            return True, None
        if text in _FALSE:
            return False, None
        return None, f"{name} must be yes or no"
    value = text.replace(" ", "_")
    if "enum" in spec and value not in spec["enum"]:
        return None, f"{name} must be one of: {', '.join(spec['enum'])}"
    return value, None


def validate_partial_features(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate any subset of the seven fields; returns accepted values and per-field errors."""
    accepted: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for name, raw_value in raw.items():
        value, error = normalize_and_check_field(name, raw_value)
        if error:
            errors[name] = error
        else:
            accepted[name] = value
    return {"accepted": accepted, "errors": errors}


def list_missing_fields(known: list[str]) -> dict[str, Any]:
    """Return required fields that are still missing, with their allowed values."""
    schema = load_input_schema()
    missing = [name for name in schema["ordered_features"] if name not in known]
    hints = {}
    for name in missing:
        spec = schema["properties"][name]
        if spec["type"] == "number":
            hints[name] = f"number {spec['minimum']}-{spec['maximum']}"
        elif spec["type"] == "boolean":
            hints[name] = "yes/no"
        else:
            hints[name] = "/".join(spec.get("enum", []))
    return {"missing": missing, "allowed": hints}


@lru_cache(maxsize=1)
def verify_locked_release() -> dict[str, Any]:
    """Run the Stage 9 release verifier once per process; failures are not cached."""
    release = validate_stage9_research_release(root=REPO_ROOT)
    if release["validation_status"] != "PASS":
        raise ValueError("Locked release verification failed")
    return {"validation_status": "PASS", "model_id": "minimal_v1"}


def predict_cvd_mortality_risk(patient: PatientInput) -> PredictionResult:
    """Use the root project's canonical verified calibrated inference path."""
    verify_locked_release()
    schema = load_input_schema()
    ordered = schema["ordered_features"]
    frame = pd.DataFrame([[getattr(patient, key) for key in ordered]], columns=ordered)
    validate_minimal_v1_research_input(frame, schema)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated in NumPy 2.5.*",
            category=DeprecationWarning,
            module=r"joblib\.numpy_pickle",
        )
        model = joblib.load(MODEL_PATH)
    if (model.name, model.route, model.probability_clip) != ("LogisticRegression", "minimal", 1e-6):
        raise ValueError("Locked pipeline identity changed")
    probability = np.asarray(model.predict_proba(frame, calibrated=True), dtype=np.float64)
    if probability.shape != (1,) or not np.isfinite(probability).all():
        raise ValueError("Invalid model prediction")
    value = float(probability[0])
    return PredictionResult(probability=value, probability_percent=100 * value)
