from __future__ import annotations

import pandas as pd
import pytest

from cvd_risk.data.cohort import (
    LeakageError,
    select_fast_features,
    validate_fast_predictors,
)
from cvd_risk.data.harmonize import FAST_FEATURES


@pytest.mark.parametrize(
    "leaking_column",
    [
        "MORTSTAT",
        "UCOD_LEADING",
        "PERMTH_INT",
        "death_year",
        "followup_quarters",
        "cvd_death_5y",
        "analysis_weight",
        "survey_year",
        "split",
    ],
)
def test_known_outcome_design_and_weight_columns_are_rejected(
    leaking_column: str,
) -> None:
    proposed = [*FAST_FEATURES, leaking_column]
    with pytest.raises(LeakageError):
        validate_fast_predictors(proposed)


def test_race_ethnicity_remains_audit_only() -> None:
    with pytest.raises(LeakageError, match="Audit-only"):
        validate_fast_predictors([*FAST_FEATURES, "race_ethnicity"])


def test_fast_feature_selection_uses_exact_locked_order() -> None:
    frame = pd.DataFrame(
        {
            **{feature: [1] for feature in FAST_FEATURES},
            "cvd_death_5y": [1],
            "race_ethnicity": ["audit-group"],
            "MORTSTAT": [1],
        }
    )

    selected = select_fast_features(frame)

    assert tuple(selected.columns) == FAST_FEATURES
    assert "cvd_death_5y" not in selected
    assert "race_ethnicity" not in selected


def test_incomplete_fast_feature_contract_fails() -> None:
    with pytest.raises(LeakageError, match="incomplete"):
        validate_fast_predictors(FAST_FEATURES[:-1])
