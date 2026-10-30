from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from cvd_risk.devops_demo import predict_synthetic_records, verify_bundle

ROOT = Path(__file__).resolve().parents[1]


def records() -> list[dict[str, Any]]:
    return json.loads((ROOT / "devops/synthetic_inputs.json").read_text("utf-8"))


def test_demo_runs_existing_calibrated_model_without_refit() -> None:
    result = predict_synthetic_records(ROOT, records())
    assert result["status"] == "PASS"
    assert result["model_id"] == "minimal_v1"
    assert result["verified_binary_artifacts"] == 4
    assert result["refit_performed"] is False
    assert result["clinical_deployment_allowed"] is False
    predictions = result["predictions"]
    assert len(predictions) == 3
    assert all(0 <= row["five_year_cvd_mortality_probability"] <= 1 for row in predictions)
    assert predict_synthetic_records(ROOT, records())["predictions"] == predictions


@pytest.mark.parametrize("mutation", ["missing", "age", "extra", "unknown", "empty"])
def test_invalid_input_is_rejected_before_deserialization(mutation: str) -> None:
    data = records()
    if mutation == "missing":
        data[0]["bmi"] = None
    elif mutation == "age":
        data[0]["age"] = 80
    elif mutation == "extra":
        for row in data:
            row["extra"] = 1
    elif mutation == "unknown":
        data[0]["smoking_status"] = "unknown"
    else:
        data = []
    with patch("cvd_risk.devops_demo.joblib.load") as load:
        with pytest.raises(ValueError):
            predict_synthetic_records(ROOT, data)
        load.assert_not_called()


def test_modified_binary_is_rejected_before_deserialization() -> None:
    original = Path.read_bytes

    def read_bytes(path: Path) -> bytes:
        if path.name == "estimator.joblib":
            return b"modified binary"
        return original(path)

    with patch.object(Path, "read_bytes", read_bytes):
        with pytest.raises(ValueError, match="checksum mismatch"):
            verify_bundle(ROOT)
