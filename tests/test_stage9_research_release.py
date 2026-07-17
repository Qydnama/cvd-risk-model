from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cvd_risk.modeling.stage9_research_release import (
    validate_minimal_v1_research_input,
    validate_stage9_research_release,
)


def _schema() -> dict[str, object]:
    return json.loads(
        Path("artifacts/minimal_v1_research_release/input_schema.json").read_text(
            encoding="utf-8"
        )
    )


def _valid_input() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [62.0],
            "sex": ["female"],
            "bmi": [26.1],
            "smoking_status": ["never"],
            "hypertension_dx": [True],
            "diabetes_dx": [False],
            "general_health": ["good"],
        }
    )


def test_stage9_release_matches_authorization_and_locked_pipeline() -> None:
    result = validate_stage9_research_release(root=Path.cwd())

    assert result["validation_status"] == "PASS"
    assert result["release_status"] == "RESEARCH_ONLY_SOURCE_SNAPSHOT_CANDIDATE"
    assert result["model_id"] == "minimal_v1"
    assert result["research_use_only"] is True
    assert result["refit_performed"] is False
    assert result["heldout_labels_accessed"] is False
    assert result["clinical_deployment_allowed"] is False
    assert result["api_or_web_route_created"] is False
    assert result["primary_export_gate"] == "FAIL"
    assert result["locked_binary_artifacts_verified"] == 4
    assert result["synthetic_smoke_records"] == 3
    assert result["synthetic_predictions_finite"] is True


def test_stage9_input_contract_accepts_complete_canonical_record() -> None:
    validate_minimal_v1_research_input(_valid_input(), _schema())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.drop(columns="general_health"), "exactly match"),
        (lambda frame: frame.assign(age=80), "outside the Stage 9"),
        (lambda frame: frame.assign(sex="unknown"), "non-canonical"),
        (lambda frame: frame.assign(bmi=pd.NA), "contains missing"),
        (lambda frame: frame.assign(extra_field=1), "exactly match"),
    ],
)
def test_stage9_input_contract_rejects_unsupported_records(
    mutation: object,
    message: str,
) -> None:
    changed = mutation(_valid_input())  # type: ignore[operator]
    with pytest.raises(ValueError, match=message):
        validate_minimal_v1_research_input(changed, _schema())


def test_stage9_creates_no_routing_or_production_artifact() -> None:
    release_dir = Path("artifacts/minimal_v1_research_release")

    assert not (release_dir / "routing_manifest.json").exists()
    assert not (release_dir / "model.joblib").exists()
    assert not (release_dir / "api_route.json").exists()
