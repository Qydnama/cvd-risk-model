"""Two deterministic sources: the Stage 9 schema and locked ML pipeline."""

from __future__ import annotations

import json
import math
import warnings
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


def predict_cvd_mortality_risk(patient: PatientInput) -> PredictionResult:
    """Use the root project's canonical verified calibrated inference path."""
    release = validate_stage9_research_release(root=REPO_ROOT)
    if release["validation_status"] != "PASS":
        raise ValueError("Locked release verification failed")
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
