from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cvd_risk.modeling.training import (
    STAGE3_REQUIRED_INPUTS,
    STAGE3_REQUIRED_OUTPUTS,
    load_modeling_config,
    load_nhis_development,
    verify_stage3_training_gate,
)
from cvd_risk.provenance import sha256_file


def test_locked_modeling_config_has_protocol_routes() -> None:
    config = load_modeling_config("configs/modeling.yaml")

    assert [entry["route"] for entry in config["models"]] == [
        "age_sex",
        "minimal_no_general_health",
        "minimal",
    ]
    assert config["cross_validation"]["splitter"] == "StratifiedGroupKFold"
    assert config["calibration"]["methods"] == ["none", "sigmoid"]


def test_development_loader_requires_pushdown_filter(monkeypatch) -> None:
    observed: dict[str, object] = {}
    development = pd.DataFrame(
        {
            "participant_id": ["p1", "p2"],
            "split": ["development", "development"],
            "survey_year": [1997, 1998],
            "strata": [1, 2],
            "psu": [1, 1],
            "analysis_weight": [1.0, 2.0],
            "cvd_death_5y": [0, 1],
            "age": [50, 60],
            "sex": ["female", "male"],
            "bmi": [24.0, 28.0],
            "smoking_status": ["never", "former"],
            "hypertension_dx": [False, True],
            "diabetes_dx": [False, False],
            "general_health": ["excellent", "good"],
        }
    )

    def fake_read_parquet(path, *, columns, filters):
        observed["path"] = path
        observed["columns"] = columns
        observed["filters"] = filters
        return development.loc[:, columns]

    monkeypatch.setattr("cvd_risk.modeling.training.pd.read_parquet", fake_read_parquet)

    loaded = load_nhis_development("cohort.parquet")

    assert len(loaded) == 2
    assert observed["filters"] == [("split", "==", "development")]


def test_stage3_gate_verifies_every_bound_digest(tmp_path: Path) -> None:
    bound = tmp_path / "bound.txt"
    bound.write_text("immutable", encoding="utf-8")
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text("schema_version: 2\n", encoding="utf-8")
    bound_entry = {"path": str(bound), "sha256": sha256_file(bound)}
    inputs = {name: dict(bound_entry) for name in STAGE3_REQUIRED_INPUTS}
    inputs["experiment_config"] = {
        "path": str(experiment.resolve()),
        "sha256": sha256_file(experiment),
    }
    outputs = {name: dict(bound_entry) for name in STAGE3_REQUIRED_OUTPUTS}
    status = {
        "schema_version": 1,
        "ready_for_modeling": True,
        "heldout_race_ethnicity_labels_sealed": True,
        "protocol_discrepancies_ready": True,
        "inputs": inputs,
        "outputs": outputs,
    }
    status_path = tmp_path / "stop_gate.json"
    status_path.write_text(json.dumps(status), encoding="utf-8")

    verified = verify_stage3_training_gate(experiment, status_path)
    assert verified["ready_for_modeling"] is True

    bound.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after Stage 3"):
        verify_stage3_training_gate(experiment, status_path)
