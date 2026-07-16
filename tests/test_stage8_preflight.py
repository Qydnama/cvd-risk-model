from __future__ import annotations

import numpy as np
import pandas as pd

from cvd_risk.modeling.stage8_clarification import materialize_sbp_alias
from cvd_risk.modeling.stage8_preflight import (
    clinical_masks,
    field_attrition_table,
    route_gate_table,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": ["a", "b", "c", "d"],
            "survey_cycle": ["a", "a", "b", "b"],
            "strata": [1, 1, 2, 2],
            "psu": [1, 2, 1, 2],
            "analysis_weight": [1.0, 2.0, 3.0, 4.0],
            "cvd_death_5y": [1, 1, 0, 0],
            "age": [50.0, 60.0, 70.0, 55.0],
            "sex": ["female", "male", "female", "male"],
            "bmi": [25.0, np.nan, 30.0, 27.0],
            "smoking_status": ["never", "former", "current", "never"],
            "hypertension_dx": ["no", "yes", "yes", "no"],
            "diabetes_dx": ["no", "no", "yes", "no"],
            "general_health": ["good", "fair", "good", "excellent"],
            "systolic_bp_mean": [120.0, 130.0, 140.0, np.nan],
            "total_cholesterol": [180.0, 190.0, 200.0, 210.0],
            "hdl_cholesterol": [50.0, 45.0, 40.0, 55.0],
            "hba1c": [5.0, 6.0, np.nan, 5.5],
        }
    )


def test_clinical_masks_preserve_both_protocol_interpretations() -> None:
    masks = clinical_masks(_frame())
    assert masks["strict_section_10_clinical_core"].tolist() == [True, False, True, False]
    assert masks["strict_section_10_clinical_hba1c_paired"].tolist() == [True, False, False, False]
    assert masks["route_policy_section_20_clinical_core"].tolist() == [True, True, True, False]
    assert masks["route_policy_section_20_clinical_hba1c_paired"].tolist() == [
        True,
        True,
        False,
        False,
    ]


def test_route_gate_can_diverge_between_protocol_interpretations() -> None:
    result = route_gate_table(_frame(), minimum_events=2).set_index("analysis")
    assert result.loc["strict_section_10_clinical_core", "events"] == 1
    assert result.loc["strict_section_10_clinical_core", "event_gate_status"] == "FAIL"
    assert result.loc["route_policy_section_20_clinical_core", "events"] == 2
    assert result.loc["route_policy_section_20_clinical_core", "event_gate_status"] == "PASS"


def test_field_attrition_reports_raw_and_weighted_availability() -> None:
    result = field_attrition_table(_frame()).set_index("field")
    assert result.loc["bmi", "n_missing"] == 1
    assert result.loc["bmi", "events_lost"] == 1
    assert result.loc["systolic_bp_mean", "canonical_field"] == "sbp_mean"
    assert result.loc["systolic_bp_mean", "weighted_observed_fraction"] == 0.6


def test_sbp_alias_preserves_values_dtype_and_missingness() -> None:
    source = pd.Series([120.5, np.nan, 141.0], dtype="Float64", name="systolic_bp_mean")
    alias = materialize_sbp_alias(source)
    assert alias.name == "sbp_mean"
    assert alias.dtype == np.dtype("float64")
    assert np.array_equal(
        source.to_numpy(dtype=np.float64, na_value=np.nan),
        alias.to_numpy(),
        equal_nan=True,
    )
