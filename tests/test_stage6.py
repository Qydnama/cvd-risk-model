from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from cvd_risk.modeling.stage6 import (
    PREDICTION_COLUMNS,
    _assert_label_free_columns,
    create_label_free_predictions,
    load_stage6_config,
    weighted_prediction_summary,
)


class _FrozenModelStub:
    name = "LogisticRegression"
    route = "minimal"

    def predict_proba(self, frame: pd.DataFrame, *, calibrated: bool = True) -> np.ndarray:
        assert calibrated is True
        return np.array([0.002, 0.02], dtype=np.float64)


def _label_free_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": ["a", "b"],
            "survey_year": [2010, 2011],
            "strata": [1, 2],
            "psu": [1, 1],
            "analysis_weight": [1.0, 3.0],
            "age": [50, 70],
            "sex": ["female", "male"],
            "bmi": [25.0, 30.0],
            "smoking_status": ["never", "former"],
            "hypertension_dx": [False, True],
            "diabetes_dx": [False, True],
            "general_health": ["excellent", "fair"],
        }
    )


@pytest.mark.parametrize(
    "column",
    ["cvd_death_5y", "MORTSTAT", "UCOD_LEADING", "followup_months", "outcome"],
)
def test_stage6_blocks_outcome_like_columns(column: str) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        _assert_label_free_columns(["age", column])


def test_stage6_config_has_explicit_cohorts_and_locked_arguments() -> None:
    config = load_stage6_config("configs/stage6_label_free.yaml")

    assert config["cohorts"]["nhis_temporal"]["years"] == [
        2010,
        2011,
        2012,
        2013,
        2014,
    ]
    assert len(config["cohorts"]["nhanes_external"]["cycles"]) == 8
    assert config["pmvalsampsize"]["seed"] == 20260714


def test_prediction_output_is_label_free_and_uses_fixed_cutoff() -> None:
    output = create_label_free_predictions(
        _label_free_frame(),
        cohort="nhis_temporal",
        model=cast(Any, _FrozenModelStub()),
        cutoff=0.007469058716065273,
        probability_clip=1.0e-6,
    )

    assert output.columns.tolist() == PREDICTION_COLUMNS
    assert output["secondary_screen_positive"].tolist() == [False, True]
    assert output["survey_cycle"].tolist() == ["", ""]
    assert not any("death" in column.lower() for column in output.columns)


def test_weighted_prediction_summary_uses_survey_weights() -> None:
    output = create_label_free_predictions(
        _label_free_frame(),
        cohort="nhis_temporal",
        model=cast(Any, _FrozenModelStub()),
        cutoff=0.01,
        probability_clip=1.0e-6,
    )

    summary = weighted_prediction_summary(output)

    assert summary["n_nominal"] == 2
    assert summary["weighted_prediction_mean"] == pytest.approx(0.0155)
    assert summary["weighted_screen_positive_fraction"] == pytest.approx(0.75)
