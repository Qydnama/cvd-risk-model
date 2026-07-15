from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cvd_risk.modeling.ml_candidates import fit_nested_ml_candidate
from cvd_risk.modeling.stage5 import load_stage5_config, verify_stage4_artifacts
from cvd_risk.provenance import sha256_file


def _development_fixture() -> pd.DataFrame:
    rng = np.random.default_rng(20260714)
    n = 360
    index = np.arange(n)
    age = 40 + index % 40
    probability = 1.0 / (1.0 + np.exp(-(-3.0 + 0.055 * (age - 40))))
    return pd.DataFrame(
        {
            "age": age,
            "sex": np.where(index % 2, "male", "female"),
            "bmi": 19.0 + index % 35,
            "smoking_status": np.asarray(["never", "former", "current"])[index % 3],
            "hypertension_dx": index % 4 == 0,
            "diabetes_dx": index % 8 == 0,
            "general_health": np.asarray(
                ["excellent", "very_good", "good", "fair", "poor"]
            )[index % 5],
            "cvd_death_5y": rng.binomial(1, probability),
            "analysis_weight": 1.0 + (index % 7) / 10.0,
            "survey_year": 1997 + index % 12,
        }
    )


def test_stage5_config_locks_models_and_weighted_brier() -> None:
    config = load_stage5_config("configs/stage5_ml.yaml")

    assert [entry["name"] for entry in config["models"]] == ["RandomForest", "XGBoost"]
    assert config["inner_tuning"]["primary_metric"] == "survey_weighted_brier"


@pytest.mark.parametrize(
    ("name", "estimator_name", "common", "candidates"),
    [
        (
            "RandomForest",
            "RandomForestClassifier",
            {
                "n_estimators": 12,
                "criterion": "log_loss",
                "max_features": "sqrt",
                "bootstrap": True,
                "n_jobs": 1,
            },
            [
                {"candidate_id": "rf_a", "max_depth": 4, "min_samples_leaf": 5},
                {"candidate_id": "rf_b", "max_depth": 6, "min_samples_leaf": 10},
            ],
        ),
        (
            "XGBoost",
            "XGBClassifier",
            {
                "n_estimators": 12,
                "learning_rate": 0.1,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "n_jobs": 1,
                "verbosity": 0,
            },
            [
                {"candidate_id": "xgb_a", "max_depth": 2, "min_child_weight": 2},
                {"candidate_id": "xgb_b", "max_depth": 3, "min_child_weight": 4},
            ],
        ),
    ],
)
def test_nested_ml_candidate_produces_complete_calibrated_oof(
    name: str,
    estimator_name: str,
    common: dict[str, object],
    candidates: list[dict[str, object]],
) -> None:
    frame = _development_fixture()

    fitted, predictions, tuning, folds = fit_nested_ml_candidate(
        frame,
        name=name,
        estimator_name=estimator_name,
        common_parameters=common,
        candidates=candidates,
        outcome_column="cvd_death_5y",
        weight_column="analysis_weight",
        group_column="survey_year",
        outer_splits=3,
        inner_splits=2,
        spline_quantiles=(0.05, 0.35, 0.65, 0.95),
        probability_clip=1.0e-6,
        seed=20260714,
    )

    assert predictions["fold"].between(1, 3).all()
    assert predictions["prediction_uncalibrated"].between(0, 1).all()
    assert predictions["prediction_sigmoid"].between(0, 1).all()
    assert tuning.groupby("outer_fold")["selected_for_outer_fold"].sum().eq(1).all()
    assert tuning["selected_for_final_fit"].any()
    assert len(folds) == 3
    assert fitted.predict_proba(frame).shape == (len(frame),)


def test_stage4_manifest_verifier_checks_model_binaries(tmp_path: Path) -> None:
    bound = tmp_path / "bound.bin"
    bound.write_bytes(b"bound")
    digest = sha256_file(bound)
    manifest = {
        "schema_version": 1,
        "stage": "stage4_baselines",
        "status": "BASELINES_FITTED_NOT_MODEL_LOCKED",
        "heldout_outcomes_accessed": False,
        "final_model_selected": False,
        "external_validation_lock_created": False,
        "source_bindings": {str(bound): digest},
        "outputs": {"oof_predictions": {"path": str(bound), "sha256": digest}},
        "models": [
            {"name": f"model_{index}", "path": str(bound), "sha256": digest}
            for index in range(3)
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    verified = verify_stage4_artifacts(path)
    assert verified["heldout_outcomes_accessed"] is False

    bound.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source binding changed"):
        verify_stage4_artifacts(path)
