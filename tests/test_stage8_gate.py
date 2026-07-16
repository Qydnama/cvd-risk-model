from __future__ import annotations

import numpy as np
import pandas as pd

from cvd_risk.modeling.stage8_gate import (
    build_primary_clinical_core,
    gate_decision_table,
    missingness_validation_table,
    weight_validation_table,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": ["a", "b", "c"],
            "analysis_weight": [1.0, 2.0, 3.0],
            "analysis_weight_source": ["WTMEC4YR/4", "WTMEC2YR/8", "WTMEC2YR/8"],
            "weight_protocol": ["protocol"] * 3,
            "weight_estimand": ["estimand"] * 3,
            "strata": [1, 1, 2],
            "psu": [1, 2, 1],
            "cvd_death_5y": [1, 0, 1],
            "age": [50.0, 60.0, 70.0],
            "sex": ["female", "male", "female"],
            "bmi": [25.0, np.nan, 30.0],
            "smoking_status": ["never", "former", "current"],
            "hypertension_dx": ["no", "yes", "yes"],
            "diabetes_dx": ["no", "no", "yes"],
            "general_health": ["good", "fair", "good"],
            "systolic_bp_mean": pd.Series([120.0, 130.0, np.nan], dtype="Float64"),
            "total_cholesterol": [180.0, 190.0, 200.0],
            "hdl_cholesterol": [50.0, 45.0, 40.0],
        }
    )


def test_primary_core_excludes_only_missing_clinical_fields_and_aliases_sbp() -> None:
    result = build_primary_clinical_core(_frame())
    assert result["participant_id"].tolist() == ["a", "b"]
    assert result["bmi"].isna().sum() == 1
    assert result["sbp_mean"].tolist() == [120.0, 130.0]
    assert result["sbp_mean"].dtype == np.dtype("float64")


def test_weight_and_missingness_validation_enforce_route_policy() -> None:
    frame = build_primary_clinical_core(_frame())
    weights = weight_validation_table(
        frame,
        {
            "allowed_weight_sources": ["WTMEC4YR/4", "WTMEC2YR/8"],
            "weight_protocol": "protocol",
            "weight_estimand": "estimand",
        },
    )
    assert weights["status"].eq("PASS").all()
    missingness = missingness_validation_table(frame).set_index("field")
    assert missingness.loc["systolic_bp_mean", "n_missing"] == 0
    assert missingness.loc["bmi", "n_missing"] == 1
    assert missingness.loc["bmi", "status"] == "PASS"
    assert missingness.loc["bmi", "imputation_policy"] == "fit_inside_training_fold_only"


def test_gate_requires_every_component_to_pass() -> None:
    design = pd.DataFrame(
        [
            {
                "raw_events": 218,
                "n_eff_design": 1000.0,
                "pmsampsize_required_n": 900.0,
                "pass_information": True,
                "max_ci_half_width": 0.002,
                "pass_risk_precision": True,
                "design_df": 20,
                "pass_design_df": True,
            }
        ]
    )
    weights = pd.DataFrame({"status": ["PASS", "PASS"]})
    missingness = pd.DataFrame({"status": ["PASS"]})
    schema = pd.DataFrame({"equality_assertion": [True]})
    config = {
        "full_gate": {
            "raw_events_minimum": 200,
            "max_design_based_risk_ci_half_width": 0.0025,
            "minimum_design_df": 8,
        }
    }
    result = gate_decision_table(design, weights, missingness, schema, config)
    assert result["status"].eq("PASS").all()
    design.loc[0, "pass_information"] = False
    result = gate_decision_table(design, weights, missingness, schema, config)
    assert result.loc[result["check"].eq("n_eff_design"), "status"].item() == "FAIL"
