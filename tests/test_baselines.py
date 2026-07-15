from __future__ import annotations

import numpy as np
import pandas as pd

from cvd_risk.modeling.baselines import (
    binary_prediction_metrics,
    fit_nested_oof_baseline,
    normalize_training_weights,
)


def _development_fixture() -> pd.DataFrame:
    rng = np.random.default_rng(20260714)
    n = 360
    index = np.arange(n)
    age = 40 + index % 40
    sex = np.where(index % 2, "male", "female")
    linear = -4.0 + 0.06 * (age - 40) + 0.35 * (sex == "male")
    outcome = rng.binomial(1, 1.0 / (1.0 + np.exp(-linear)))
    return pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "bmi": 20.0 + index % 30,
            "smoking_status": np.asarray(["never", "former", "current"])[index % 3],
            "hypertension_dx": index % 4 == 0,
            "diabetes_dx": index % 9 == 0,
            "general_health": np.asarray(
                ["excellent", "very_good", "good", "fair", "poor"]
            )[index % 5],
            "cvd_death_5y": outcome,
            "analysis_weight": 1.0 + (index % 11) / 10.0,
            "survey_year": 1997 + index % 12,
        }
    )


def test_training_weights_are_normalized_within_fold() -> None:
    normalized = normalize_training_weights(np.array([2.0, 3.0, 7.0]))

    assert normalized.mean() == 1.0
    assert np.all(normalized > 0)


def test_nested_grouped_oof_baseline_is_complete_and_year_disjoint() -> None:
    frame = _development_fixture()
    fitted, predictions, folds = fit_nested_oof_baseline(
        frame,
        name="AgeSexLogistic",
        route="age_sex",
        outcome_column="cvd_death_5y",
        weight_column="analysis_weight",
        group_column="survey_year",
        outer_splits=4,
        inner_splits=3,
        spline_quantiles=(0.05, 0.35, 0.65, 0.95),
        logistic_settings={
            "penalty": None,
            "solver": "lbfgs",
            "max_iter": 500,
            "tolerance": 1.0e-8,
        },
        probability_clip=1.0e-6,
        seed=20260714,
    )

    assert predictions["fold"].between(1, 4).all()
    assert predictions["prediction_uncalibrated"].between(0, 1).all()
    assert predictions["prediction_sigmoid"].between(0, 1).all()
    for row in folds.itertuples(index=False):
        train_years = set(str(row.train_years).split(";"))
        validation_years = set(str(row.validation_years).split(";"))
        assert train_years.isdisjoint(validation_years)
    final_prediction = fitted.predict_proba(frame)
    assert final_prediction.shape == (len(frame),)
    assert np.isfinite(final_prediction).all()


def test_binary_metrics_include_calibration_and_probability_scores() -> None:
    y = np.array([0, 0, 1, 1])
    prediction = np.array([0.1, 0.2, 0.7, 0.9])
    metrics = binary_prediction_metrics(
        y, prediction, np.ones(4), probability_clip=1.0e-6
    )

    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["brier_score"] < 0.1
    assert set(metrics) == {
        "roc_auc",
        "average_precision",
        "brier_score",
        "log_loss",
        "calibration_intercept",
        "calibration_slope",
        "observed_expected_ratio",
    }
