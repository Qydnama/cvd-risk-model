from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cvd_risk.modeling.operating_point import (
    compare_uncalibrated_and_sigmoid,
    load_operating_point_config,
    select_maximum_cutoff,
    stratified_cluster_bootstrap,
    weighted_operating_metrics,
)


def test_operating_point_config_locks_secondary_interpretation() -> None:
    config = load_operating_point_config("configs/operating_point.yaml")
    policy = config["secondary_screening_operating_point"]

    assert policy["target_survey_weighted_sensitivity"] == 0.80
    assert policy["primary_model_output"] == "continuous_five_year_risk"
    assert policy["role"] == "secondary_screening_operating_point"
    assert config["cluster_bootstrap"]["reselect_cutoff_in_each_replicate"] is False


def test_selects_maximum_cutoff_with_ties_at_target() -> None:
    risk = np.array([0.9, 0.8, 0.8, 0.4, 0.1])
    outcome = np.array([1, 0, 1, 1, 1])
    weight = np.ones(5)

    result = select_maximum_cutoff(
        risk, outcome, weight, target_sensitivity=0.50
    )

    assert result["cutoff"] == 0.8
    assert result["achieved_sensitivity"] == 0.5
    assert result["next_higher_cutoff"] == 0.9
    assert result["next_higher_cutoff_sensitivity"] == 0.25


def test_weighted_metrics_and_dca_use_fixed_cutoff() -> None:
    frame = pd.DataFrame(
        {
            "prediction_sigmoid": [0.9, 0.8, 0.4, 0.1],
            "cvd_death_5y": [1, 0, 1, 0],
            "analysis_weight": [1.0, 2.0, 3.0, 4.0],
        }
    )

    result = weighted_operating_metrics(frame, cutoff=0.5)

    assert result["tp"] == 1.0
    assert result["fp"] == 2.0
    assert result["fn"] == 3.0
    assert result["tn"] == 4.0
    assert result["sensitivity"] == pytest.approx(0.25)
    assert result["specificity"] == pytest.approx(2 / 3)
    assert result["ppv"] == pytest.approx(1 / 3)
    assert result["npv"] == pytest.approx(4 / 7)
    assert result["net_benefit_model"] == pytest.approx(-0.1)


def test_cluster_bootstrap_is_reproducible_and_stratified() -> None:
    frame = pd.DataFrame(
        {
            "strata": [1, 1, 1, 1, 2, 2, 2, 2],
            "psu": [1, 1, 2, 2, 1, 1, 2, 2],
            "prediction_sigmoid": [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4],
            "cvd_death_5y": [1, 0, 1, 0, 1, 0, 0, 1],
            "analysis_weight": np.ones(8),
        }
    )

    first = stratified_cluster_bootstrap(
        frame, cutoff=0.5, repetitions=50, seed=7
    )
    second = stratified_cluster_bootstrap(
        frame, cutoff=0.5, repetitions=50, seed=7
    )

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 50
    assert first["sensitivity"].between(0, 1).all()


def test_calibration_comparison_accepts_brier_gain_and_log_loss_tie() -> None:
    frame = pd.DataFrame(
        {
            "strata": [1, 1, 1, 1, 2, 2, 2, 2],
            "psu": [1, 1, 2, 2, 1, 1, 2, 2],
            "analysis_weight": np.ones(8),
            "cvd_death_5y": [1, 0, 1, 0, 1, 0, 0, 1],
            "prediction_uncalibrated": [0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.3, 0.7],
            "prediction_sigmoid": [0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.2, 0.8],
        }
    )
    comparison = pd.DataFrame(
        [
            {
                "model": "LogisticRegression",
                "weighting": "survey_weighted",
                "calibration": "none",
                "brier_score": 0.09,
                "log_loss": 0.35,
                "calibration_intercept": -0.2,
                "calibration_slope": 0.8,
                "observed_expected_ratio": 0.9,
            },
            {
                "model": "LogisticRegression",
                "weighting": "survey_weighted",
                "calibration": "sigmoid",
                "brier_score": 0.04,
                "log_loss": 0.22,
                "calibration_intercept": 0.05,
                "calibration_slope": 0.95,
                "observed_expected_ratio": 1.02,
            },
        ]
    )

    result = compare_uncalibrated_and_sigmoid(
        frame,
        comparison,
        repetitions=50,
        seed=7,
        confidence_level=0.95,
        numerical_tie_tolerance=1e-6,
    )

    assert result["passed"].all()
