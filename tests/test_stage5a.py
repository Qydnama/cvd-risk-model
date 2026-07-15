from __future__ import annotations

import numpy as np
import pytest

from cvd_risk.modeling.stage5a import (
    evaluate_stability,
    load_final_model_selection,
    load_stage5a_config,
    weighted_lp_summary,
)


def test_stage5a_configs_lock_selected_logistic_and_pmval_arguments() -> None:
    config = load_stage5a_config("configs/stage5a_planning.yaml")
    selection = load_final_model_selection(config["selected_model_config"])

    assert selection["primary"]["model"] == "LogisticRegression"
    assert selection["primary"]["calibration"] == "sigmoid"
    assert config["pmvalsampsize"]["initial_simobs"] == 1_000_000
    assert config["pmvalsampsize"]["repeat_simobs"] == 2_000_000


def test_weighted_lp_summary_uses_population_sd_and_clipping() -> None:
    prediction = np.array([0.0, 0.2, 0.8, 1.0])
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    outcome = np.array([0, 0, 1, 1])

    result = weighted_lp_summary(
        prediction,
        weights,
        outcome,
        probability_clip=1.0e-6,
    )

    bounded = np.clip(prediction, 1.0e-6, 1.0 - 1.0e-6)
    lp = np.log(bounded / (1.0 - bounded))
    expected_mean = np.average(lp, weights=weights)
    expected_sd = np.sqrt(np.average((lp - expected_mean) ** 2, weights=weights))
    assert result["weighted_lp_mean"] == pytest.approx(expected_mean)
    assert result["weighted_lp_sd"] == pytest.approx(expected_sd)
    assert result["weighted_prevalence"] == pytest.approx(0.7)
    assert result["n_clipped_low"] == 1
    assert result["n_clipped_high"] == 1


def test_pmval_stability_uses_larger_adjacent_requirement() -> None:
    stable = evaluate_stability(
        [{"required_n": 41084}, {"required_n": 41100}], tolerance=0.01
    )

    assert stable["stable"] is True
    assert stable["provisional_required_n"] == 41100
    assert stable["relative_difference"] == pytest.approx(16 / 41100)


def test_pmval_stability_detects_need_for_another_run() -> None:
    unstable = evaluate_stability(
        [{"required_n": 40000}, {"required_n": 42000}], tolerance=0.01
    )

    assert unstable["stable"] is False
