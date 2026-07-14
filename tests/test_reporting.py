from __future__ import annotations

import pandas as pd
import pytest

from cvd_risk.reporting.tables import (
    cohort_flow_table,
    evaluate_stop_gate,
    event_counts_table,
    missingness_reason_table,
    missingness_table,
    subgroup_event_counts_table,
)


def test_cohort_flow_rejects_growth() -> None:
    small = pd.DataFrame({"cvd_death_5y": [0]})
    large = pd.DataFrame({"cvd_death_5y": [0, 1]})
    with pytest.raises(ValueError, match="Cohort grew"):
        cohort_flow_table([("eligible", small), ("bad merge", large)])


def test_event_counts_supports_survey_weights() -> None:
    frame = pd.DataFrame({"cvd_death_5y": [1, 0, 0], "analysis_weight": [2.0, 1.0, 1.0]})
    table = event_counts_table(
        {"nhis_development": frame},
        weight_columns={"nhis_development": "analysis_weight"},
    )
    row = table.iloc[0]
    assert row["n_events"] == 1
    assert row["event_prevalence_unweighted"] == pytest.approx(1 / 3)
    assert row["event_prevalence_weighted"] == pytest.approx(0.5)


def test_event_counts_rejects_non_positive_weight_sum() -> None:
    frame = pd.DataFrame({"cvd_death_5y": [1, 0], "analysis_weight": [0.0, 0.0]})
    with pytest.raises(ValueError, match="non-positive"):
        event_counts_table(
            {"nhis_development": frame},
            weight_columns={"nhis_development": "analysis_weight"},
        )


def test_missingness_marks_absent_feature_unavailable() -> None:
    frame = pd.DataFrame({"age": [40, None]})
    table = missingness_table({"sample": frame}, features=("age", "bmi"))
    bmi = table.loc[table["feature"] == "bmi"].iloc[0]
    assert not bool(bmi["available"])
    assert bmi["n_missing"] == 2


def test_missingness_preserves_refused_reason() -> None:
    frame = pd.DataFrame(
        {
            "sex": pd.Series(["female", pd.NA], dtype="string"),
            "sex_missing_reason": pd.Series([pd.NA, "refused"], dtype="string"),
        }
    )
    table = missingness_reason_table({"nhis": frame}, features=("sex",))
    assert table.iloc[0]["missing_reason"] == "refused"
    assert table.iloc[0]["n_reason"] == 1


def test_subgroup_counts_flag_unreliable_categories() -> None:
    frame = pd.DataFrame(
        {
            "cvd_death_5y": [1, 0, 0, 0],
            "analysis_weight": [1.0] * 4,
            "race_ethnicity": ["a", "a", "b", pd.NA],
        }
    )
    table = subgroup_event_counts_table(
        {"external": frame},
        subgroup_col="race_ethnicity",
        minimum_events=2,
        minimum_non_events=2,
    )
    assert not table["eligible_for_performance_audit"].any()
    assert set(table["n_missing_subgroup"]) == {1}


def test_stop_gate_blocks_missing_external_cohort_and_model_df() -> None:
    counts = pd.DataFrame(
        {
            "cohort": ["nhis_development", "nhis_temporal_test"],
            "n_events": [501, 210],
        }
    )
    result = evaluate_stop_gate(counts)
    assert not result.ready_for_modeling
    assert "nhanes_external" in result.summary


def test_stop_gate_passes_when_all_requirements_are_met() -> None:
    counts = pd.DataFrame(
        {
            "cohort": [
                "nhis_development",
                "nhis_temporal_test",
                "nhanes_external",
            ],
            "n_events": [600, 250, 120],
        }
    )
    result = evaluate_stop_gate(counts, effective_degrees_of_freedom=25)
    assert result.ready_for_modeling
    assert set(result.details["status"]) == {"pass"}
