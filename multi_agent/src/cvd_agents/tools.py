"""Deterministic tools; no LLM is trusted for model inference or evidence."""

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

from .config import REPO_ROOT
from .contracts import EvidenceBundle, PatientInput, PredictionResult, ValidationResult


class ArtifactVerificationError(RuntimeError):
    pass


def _json(relative: str, root: Path = REPO_ROOT) -> dict[str, Any]:
    path = root / relative
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Evidence file unavailable or invalid: {relative}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Evidence root is not an object: {relative}")
    return value


def load_input_schema(root: Path = REPO_ROOT) -> dict[str, Any]:
    schema = _json("artifacts/minimal_v1_research_release/input_schema.json", root)
    if schema.get("model_id") != "minimal_v1" or len(schema.get("required", [])) != 7:
        raise ValueError("Unexpected minimal_v1 input schema")
    return schema


def validate_patient_input(data: dict[str, Any], root: Path = REPO_ROOT) -> ValidationResult:
    schema = load_input_schema(root)
    required = schema["required"]
    errors = [f"Missing required field: {key}" for key in required if key not in data]
    errors += [f"Unexpected field: {key}" for key in data if key not in required]
    if errors:
        return ValidationResult(valid=False, errors=errors)
    try:
        patient = PatientInput.model_validate(data)
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


def verify_model_artifacts(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Reuse the repository's full Stage 9 verifier before any unpickling."""
    try:
        result = validate_stage9_research_release(root=root)
    except Exception as exc:
        raise ArtifactVerificationError(f"Locked model verification failed: {type(exc).__name__}: {exc}") from exc
    if result.get("validation_status") != "PASS" or result.get("locked_binary_artifacts_verified") != 4:
        raise ArtifactVerificationError("Locked model verification did not pass")
    return result


def predict_cvd_mortality_risk(patient: PatientInput, root: Path = REPO_ROOT) -> PredictionResult:
    verified = verify_model_artifacts(root)
    schema = load_input_schema(root)
    frame = pd.DataFrame([[getattr(patient, key) for key in schema["ordered_features"]]], columns=schema["ordered_features"])
    validate_minimal_v1_research_input(frame, schema)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Setting the shape on a NumPy array has been deprecated in NumPy 2.5.*", category=DeprecationWarning, module=r"joblib\.numpy_pickle")
            model = joblib.load(root / "artifacts/minimal_v1/full_development_pipeline.joblib")
        if (model.name, model.route, model.probability_clip) != ("LogisticRegression", "minimal", 1e-6):
            raise ValueError("Locked pipeline identity mismatch")
        probability = np.asarray(model.predict_proba(frame, calibrated=True), dtype=np.float64)
    except Exception as exc:
        raise RuntimeError(f"Locked model prediction failed: {type(exc).__name__}") from exc
    if probability.shape != (1,) or not np.isfinite(probability).all() or not 0 <= probability[0] <= 1:
        raise RuntimeError("Locked model returned an invalid probability")
    value = float(probability[0])
    return PredictionResult(probability=value, probability_percent=value * 100, verified_pipeline_sha256=verified["locked_pipeline_sha256"])


def get_model_evidence(root: Path = REPO_ROOT) -> EvidenceBundle:
    """Read only repository-owned, manifest-bound evidence."""
    verify_model_artifacts(root)
    manifest = _json("artifacts/minimal_v1_research_release/release_manifest.json", root)
    card = _json("artifacts/minimal_v1_model_card/status.json", root)
    pipeline = _json("artifacts/minimal_v1/serialized_pipeline_manifest.json", root)
    schema = load_input_schema(root)
    card_path = root / "reports/minimal_v1_model_card.md"
    try:
        card_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError("Model card unavailable") from exc
    if (manifest.get("model_id"), card.get("model_id"), schema.get("model_id")) != ("minimal_v1",) * 3:
        raise ValueError("Evidence model IDs disagree")
    if pipeline["pipeline"]["name"] != card["model"]:
        raise ValueError("Evidence model names disagree")
    development = card["development"]
    return EvidenceBundle(
        model_name=card["model"], model_id=card["model_id"],
        development_source=f"{development['source']} {development['years']}",
        predictor_count=len(schema["ordered_features"]),
        metrics={"development_weighted_roc_auc": development["weighted_roc_auc"], "development_weighted_brier_score": development["weighted_brier_score"]},
        limitations=["Research use only", "Temporal and external validation are underpowered and exploratory", "No validation for Kazakhstan", "No clinical deployment approval"],
        sources=["reports/minimal_v1_model_card.md", "artifacts/minimal_v1_model_card/status.json", "artifacts/minimal_v1_research_release/input_schema.json", "artifacts/minimal_v1_research_release/release_manifest.json", "artifacts/minimal_v1/serialized_pipeline_manifest.json"],
    )


def check_research_only_constraints(text: str, prediction: PredictionResult | None, evidence: EvidenceBundle | None) -> list[str]:
    lower = text.lower()
    violations: list[str] = []
    if prediction is None or evidence is None or len(prediction.verified_pipeline_sha256) != 64:
        violations.append("missing_verified_model_or_evidence")
    if "five-year cardiovascular mortality" not in lower and "five year cardiovascular mortality" not in lower:
        violations.append("wrong_or_missing_outcome")
    if "research" not in lower:
        violations.append("missing_research_only_limit")
    if "kazakhstan" not in lower or "not validated" not in lower:
        violations.append("missing_kazakhstan_limitation")
    forbidden = ("take medication", "stop medication", "start medication", "diagnosed with", "clinically validated", "high risk", "low risk", "medium risk")
    if any(token in lower for token in forbidden):
        violations.append("unsafe_medical_or_threshold_claim")
    if prediction is not None and f"{prediction.probability_percent:.4f}%" not in text:
        violations.append("prediction_mismatch")
    return violations
