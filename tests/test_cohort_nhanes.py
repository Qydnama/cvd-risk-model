from __future__ import annotations

import pandas as pd
import pytest

from cvd_risk.data.cohort import OutcomeDefinitionError
from cvd_risk.data.cohort_nhanes import (
    NHANESCohortError,
    annotate_nhanes_cohort,
    build_nhanes_cohort_with_flow,
    derive_nhanes_cvd_outcome_with_complete_followup,
)


def _candidates() -> pd.DataFrame:
    identifiers = [
        "alive",
        "cvd-event",
        "competing-death",
        "too-young",
        "zero-weight",
        "prevalent",
        "baseline-unknown",
        "short-alive-followup",
        "ineligible",
    ]
    size = len(identifiers)
    frame = pd.DataFrame(
        {
            "participant_id": identifiers,
            "survey_cycle": ["2013-2014"] * size,
            "survey_year": [2013] * size,
            "age": [50, 60, 70, 39, 50, 50, 50, 50, 50],
            "analysis_weight": [1.0, 2.0, 3.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            "ELIGSTAT": [1, 1, 1, 1, 1, 1, 1, 1, 3],
            "MORTSTAT": [0, 1, 1, 0, 0, 0, 0, 0, pd.NA],
            "UCOD_LEADING": [
                pd.NA,
                "001",
                "002",
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
            ],
            "PERMTH_INT": [60, 60, 12, 100, 100, 100, 100, 59, pd.NA],
            "coronary_heart_disease": [
                False,
                False,
                False,
                False,
                False,
                True,
                pd.NA,
                False,
                False,
            ],
            "angina_pectoris": [False] * size,
            "myocardial_infarction": [False] * size,
            "stroke": [False] * size,
            "congestive_heart_failure": [False] * size,
        }
    )
    return frame


def test_exact_month_60_event_and_competing_death_are_observed_outcomes() -> None:
    result = derive_nhanes_cvd_outcome_with_complete_followup(_candidates())

    assert result.loc[0, "cvd_death_5y"] == 0
    assert result.loc[1, "cvd_death_5y"] == 1
    assert result.loc[2, "cvd_death_5y"] == 0
    assert bool(result.loc[2, "full_five_year_outcome_observed"])


def test_alive_participant_requires_at_least_60_months_followup() -> None:
    result = derive_nhanes_cvd_outcome_with_complete_followup(_candidates())

    assert pd.isna(result.loc[7, "cvd_death_5y"])
    assert result.loc[7, "outcome_missing_reason"] == "insufficient_five_year_followup"
    assert not bool(result.loc[7, "full_five_year_outcome_observed"])


def test_negative_followup_is_rejected_even_for_alive_participant() -> None:
    frame = _candidates().iloc[[0]].copy()
    frame["PERMTH_INT"] = -1

    with pytest.raises(OutcomeDefinitionError, match="Negative NHANES follow-up"):
        derive_nhanes_cvd_outcome_with_complete_followup(frame)


def test_locked_cohort_and_flow_exclude_zero_weight_unknown_cvd_and_short_followup() -> None:
    cohort, flow = build_nhanes_cohort_with_flow(_candidates())

    assert cohort["participant_id"].tolist() == ["alive", "cvd-event", "competing-death"]
    assert cohort["cvd_death_5y"].tolist() == [0, 1, 0]
    assert cohort["split"].tolist() == ["external_validation"] * 3
    assert flow["n_participants"].is_monotonic_decreasing
    assert flow.iloc[-1]["n_participants"] == 3


def test_annotation_preserves_distinct_exclusion_reasons() -> None:
    annotated = annotate_nhanes_cohort(_candidates())
    reasons = annotated.set_index("participant_id")["cohort_exclusion_reason"]

    assert reasons["zero-weight"] == "nonpositive_or_missing_mec_weight"
    assert reasons["prevalent"] == "prevalent_cvd"
    assert reasons["baseline-unknown"] == "baseline_cvd_unknown"
    assert reasons["short-alive-followup"] == "outcome_unknown_or_incomplete_followup"


def test_chf_is_excluded_only_from_the_nhanes_sensitivity_cohort() -> None:
    candidates = _candidates()
    candidates.loc[0, "congestive_heart_failure"] = True

    annotated = annotate_nhanes_cohort(candidates).set_index("participant_id")

    assert annotated.loc["alive", "baseline_cvd_status"] == "absent"
    assert annotated.loc["alive", "baseline_cvd_status_with_chf"] == "present"
    assert bool(annotated.loc["alive", "cohort_eligible"])
    assert not bool(annotated.loc["alive", "cohort_chf_sensitivity_eligible"])


def test_duplicate_participant_ids_are_rejected_before_flow() -> None:
    frame = _candidates()
    frame.loc[1, "participant_id"] = frame.loc[0, "participant_id"]

    with pytest.raises(NHANESCohortError, match="duplicate participant"):
        build_nhanes_cohort_with_flow(frame)
