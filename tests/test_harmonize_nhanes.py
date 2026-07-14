from __future__ import annotations

import pandas as pd
import pytest

from cvd_risk.data.cohort_nhanes import annotate_nhanes_cohort
from cvd_risk.data.harmonize_nhanes import (
    NHANESHarmonizationError,
    harmonize_nhanes,
)

MACHINE_ZERO = 5.397605346934028e-79


def _raw_nhanes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CYCLE": ["1999-2000", "2003-2004", "2005-2006"],
            "SEQN": [10, 10, 10],
            "RIDAGEYR": [MACHINE_ZERO, 50, 60],
            "RIAGENDR": [1, 2, 1],
            "RIDRETH1": [1, 3, 5],
            "WTMEC2YR": [MACHINE_ZERO, 800.0, 1600.0],
            "WTMEC4YR": [3200.0, pd.NA, pd.NA],
            "SDMVSTRA": [1, 29, 44],
            "SDMVPSU": [1, 2, 2],
            "BMXBMI": [pd.NA, 25.0, 30.0],
            "SMQ020": [2, 1, 1],
            "SMQ040": [pd.NA, 3, 1],
            "BPQ020": [2, 1, 7],
            "DIQ010": [2, 3, 1],
            "HUQ010": [1, pd.NA, pd.NA],
            "HSD010": [pd.NA, 4, 5],
            "MCQ160B": [2, 2, 2],
            "MCQ160C": [2, 2, 9],
            "MCQ160D": [2, 2, 2],
            "MCQ160E": [2, 2, 2],
            "MCQ160F": [2, 2, 2],
            "ELIGSTAT": [1, 1, 1],
            "MORTSTAT": [0, 1, 1],
            "UCOD_LEADING": [pd.NA, "001", "002"],
            "PERMTH_INT": [100, 60, 12],
            "BPXSY1": [pd.NA, 120, 130],
            "BPXSY2": [pd.NA, 118, 128],
            "BPXSY3": [pd.NA, pd.NA, 126],
            "BPXSY4": [pd.NA, 122, pd.NA],
            "LBXTC": [pd.NA, 200, 210],
            "LBDHDL": [50, pd.NA, pd.NA],
            "LBXHDD": [pd.NA, 55, pd.NA],
            "LBDHDD": [pd.NA, pd.NA, 60],
            "LBXGH": [pd.NA, 5.5, 6.5],
        }
    )


def test_harmonizes_fast_features_design_and_audit_race_with_provenance() -> None:
    result = harmonize_nhanes(_raw_nhanes())

    assert result["participant_id"].tolist() == [
        "1999-2000:10",
        "2003-2004:10",
        "2005-2006:10",
    ]
    assert result["survey_year"].tolist() == [1999, 2003, 2005]
    assert result["sex"].tolist() == ["male", "female", "male"]
    assert result["smoking_status"].tolist() == ["never", "former", "current"]
    assert result["general_health"].tolist() == ["excellent", "fair", "poor"]
    assert result["general_health_source"].tolist() == ["HUQ010", "HSD010", "HSD010"]
    assert result["race_ethnicity"].tolist() == [
        "mexican_american",
        "non_hispanic_white",
        "other_or_multiracial",
    ]
    assert result.loc[2, "coronary_heart_disease_missing_reason"] == "dont_know"
    assert pd.isna(result.loc[2, "coronary_heart_disease"])


def test_normalizes_legacy_xport_machine_zero_before_age_and_weight_logic() -> None:
    result = harmonize_nhanes(_raw_nhanes())

    assert result.loc[0, "age_raw"] == MACHINE_ZERO
    assert result.loc[0, "age"] == 0
    assert result.loc[0, "mec_weight_raw"] == MACHINE_ZERO
    assert result.loc[0, "mec_weight"] == 0
    assert result.loc[0, "analysis_weight"] == 800
    assert result.loc[0, "mec_four_year_weight"] == 3200


def test_amendment_2_4_uses_four_year_weight_for_early_cycles() -> None:
    raw = _raw_nhanes().iloc[[0]].copy()
    raw["RIDAGEYR"] = 50
    raw["WTMEC2YR"] = 800.0
    raw["WTMEC4YR"] = 3200.0

    result = harmonize_nhanes(raw)

    assert result.loc[0, "mec_four_year_weight"] == 3200.0
    assert result.loc[0, "mec_four_year_weight_source"] == "WTMEC4YR"
    assert result.loc[0, "analysis_weight"] == 800.0
    assert result.loc[0, "analysis_weight_source"] == "WTMEC4YR/4"
    assert result.loc[0, "analysis_weight_multiplier"] == 0.25
    assert result.loc[0, "weight_protocol"] == "amendment_2_4_source_specific_pooling"
    assert result.loc[0, "weight_estimand"] == "nhanes_external_validation_1999_2014"


def test_amendment_2_4_requires_four_year_weight_for_early_cycles() -> None:
    raw = _raw_nhanes().iloc[[0]].drop(columns="WTMEC4YR")
    raw["RIDAGEYR"] = 50

    with pytest.raises(NHANESHarmonizationError, match="WTMEC4YR is required"):
        harmonize_nhanes(raw)


def test_machine_zero_weight_is_excluded_even_when_participant_is_an_eligible_adult() -> None:
    raw = _raw_nhanes().iloc[[0]].copy()
    raw["RIDAGEYR"] = 50
    raw["WTMEC4YR"] = MACHINE_ZERO
    harmonized = harmonize_nhanes(raw)

    annotated = annotate_nhanes_cohort(harmonized)

    assert harmonized.loc[0, "analysis_weight"] == 0
    assert not bool(annotated.loc[0, "cohort_eligible"])
    assert annotated.loc[0, "cohort_exclusion_reason"] == "nonpositive_or_missing_mec_weight"


def test_extended_features_use_available_bp_readings_and_cycle_aware_labs() -> None:
    result = harmonize_nhanes(_raw_nhanes(), include_extended=True)

    assert pd.isna(result.loc[0, "systolic_bp"])
    assert result.loc[0, "systolic_bp_missing_reason"] == "true_missing"
    assert result.loc[1, "systolic_bp"] == pytest.approx(120.0)
    assert result.loc[2, "systolic_bp"] == pytest.approx(128.0)
    assert result["hdl_cholesterol"].tolist() == [50.0, 55.0, 60.0]
    assert result["hdl_cholesterol_source"].tolist() == [
        "LBDHDL",
        "LBXHDD",
        "LBDHDD",
    ]
    assert result["total_cholesterol"].tolist()[1:] == [200.0, 210.0]
    assert result["hba1c"].tolist()[1:] == [5.5, 6.5]


def test_valid_missing_codes_remain_missing_instead_of_becoming_reference_values() -> None:
    result = harmonize_nhanes(_raw_nhanes())

    assert pd.isna(result.loc[1, "diabetes_dx"])
    assert result.loc[1, "diabetes_dx_missing_reason"] == "borderline"
    assert pd.isna(result.loc[2, "hypertension_dx"])
    assert result.loc[2, "hypertension_dx_missing_reason"] == "refused"
    assert pd.isna(result.loc[0, "bmi"])
    assert result.loc[0, "bmi_missing_reason"] == "true_missing"


def test_unknown_nonmissing_code_fails_fast() -> None:
    raw = _raw_nhanes()
    raw.loc[1, "RIAGENDR"] = 3

    with pytest.raises(NHANESHarmonizationError, match="RIAGENDR: unknown code 3"):
        harmonize_nhanes(raw)


def test_1999_requires_huq010_not_later_hsd010() -> None:
    raw = _raw_nhanes().drop(columns="HUQ010")

    with pytest.raises(NHANESHarmonizationError, match="1999-2000.*HUQ010"):
        harmonize_nhanes(raw)


def test_duplicate_ids_within_cycle_are_rejected() -> None:
    raw = pd.concat([_raw_nhanes(), _raw_nhanes().iloc[[0]]], ignore_index=True)

    with pytest.raises(NHANESHarmonizationError, match="participant_id.*unique"):
        harmonize_nhanes(raw)


def test_systolic_source_range_is_strict_but_missing_readings_are_allowed() -> None:
    raw = _raw_nhanes()
    raw.loc[1, "BPXSY1"] = 999

    with pytest.raises(NHANESHarmonizationError, match="BPXSY1"):
        harmonize_nhanes(raw, include_extended=True)


def test_systolic_readings_must_follow_official_even_integer_encoding() -> None:
    raw = _raw_nhanes()
    raw.loc[1, "BPXSY1"] = 121

    with pytest.raises(NHANESHarmonizationError, match="even integer"):
        harmonize_nhanes(raw, include_extended=True)
