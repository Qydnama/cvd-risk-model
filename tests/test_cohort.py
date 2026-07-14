from __future__ import annotations

import pandas as pd
import pytest

from cvd_risk.data.cohort import (
    annotate_nhis_cohort,
    assign_nhis_split,
    build_nhis_cohort,
    build_nhis_cohort_with_flow,
    derive_baseline_cvd_status,
)
from cvd_risk.data.harmonize import (
    BASELINE_CVD_COLUMNS,
    VERIFIED_CODEBOOK_STATUS,
    CodebookVerificationError,
)


def _cohort_candidates() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "participant_id": [
                "development",
                "temporal",
                "too-young",
                "prevalent",
                "baseline-unknown",
                "linkage-ineligible",
                "boundary",
            ],
            "survey_year": [2009, 2010, 2009, 2009, 2009, 2009, 2005],
            "interview_quarter": [1] * 7,
            "age": [40, 79, 39, 60, 60, 60, 60],
            "mortality_eligible": [True, True, True, True, True, False, True],
            "mortality_weight": [130.0, 50.0, 130.0, 130.0, 130.0, 130.0, 130.0],
            "deceased": [False, True, False, False, False, pd.NA, True],
            "ucod_leading": [pd.NA, "001", pd.NA, pd.NA, pd.NA, pd.NA, "005"],
            "death_year": [pd.NA, 2012, pd.NA, pd.NA, pd.NA, pd.NA, 2010],
            "death_quarter": [pd.NA, 1, pd.NA, pd.NA, pd.NA, pd.NA, 1],
            "coronary_heart_disease": [False, False, False, True, pd.NA, False, False],
            "angina_pectoris": [False] * 7,
            "myocardial_infarction": [False] * 7,
            "stroke": [False] * 7,
            "congestive_heart_failure": [False] * 7,
        }
    )
    return frame


def _mark_codebook_verified(frame: pd.DataFrame) -> pd.DataFrame:
    verified = frame.copy()
    verified.attrs["codebook_verification_status"] = VERIFIED_CODEBOOK_STATUS
    verified.attrs["extract_sha256"] = "a" * 64
    verified.attrs["codebook_sha256"] = "b" * 64
    verified.attrs["feature_mapping_sha256"] = "c" * 64
    return verified


def test_baseline_cvd_unknown_is_not_treated_as_absent() -> None:
    status = derive_baseline_cvd_status(_cohort_candidates())

    assert status.iloc[0] == "absent"
    assert status.iloc[3] == "present"
    assert status.iloc[4] == "unknown"


def test_pending_codebook_blocks_final_cohort() -> None:
    with pytest.raises(CodebookVerificationError, match="blocked"):
        build_nhis_cohort(_cohort_candidates())


def test_locked_cohort_filters_and_weight_scaling() -> None:
    cohort, flow = build_nhis_cohort_with_flow(_mark_codebook_verified(_cohort_candidates()))

    assert cohort["participant_id"].tolist() == ["development", "temporal"]
    assert cohort["split"].tolist() == ["development", "temporal_test"]
    assert cohort["cvd_death_5y"].tolist() == [0, 1]
    assert cohort["analysis_weight"].tolist() == pytest.approx([10.0, 10.0])
    assert flow["n_participants"].is_monotonic_decreasing
    assert flow.iloc[-1]["n_participants"] == 2


def test_boundary_record_is_only_available_in_sensitivity_cohort() -> None:
    candidates = _mark_codebook_verified(_cohort_candidates())
    primary = build_nhis_cohort(candidates)
    sensitivity = build_nhis_cohort(candidates, include_ambiguous_boundary=True)

    assert "boundary" not in set(primary["participant_id"])
    assert "boundary" in set(sensitivity["participant_id"])


def test_annotation_records_distinct_prevalent_and_unknown_exclusions() -> None:
    annotated = annotate_nhis_cohort(_cohort_candidates())
    reasons = annotated.set_index("participant_id")["cohort_exclusion_reason"]

    assert reasons["prevalent"] == "prevalent_cvd"
    assert reasons["baseline-unknown"] == "baseline_cvd_unknown"
    assert reasons["boundary"] == "outcome_unknown_or_ambiguous"


def test_split_is_fixed_by_year() -> None:
    splits = assign_nhis_split(pd.Series([1997, 2009, 2010, 2014]))
    assert splits.tolist() == [
        "development",
        "development",
        "temporal_test",
        "temporal_test",
    ]


def test_all_baseline_conditions_are_present_in_fixture() -> None:
    assert set(BASELINE_CVD_COLUMNS).issubset(_cohort_candidates().columns)
