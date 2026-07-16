from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cvd_risk.modeling.stage7 import (
    Stage7Error,
    build_nhanes_recent_cohort,
    build_nhis_recent_cohort,
    fit_weighted_cause_specific_cox,
    ipcw_cumulative_incidence_metrics,
    load_stage7_config,
    predict_cif_from_cause_specific_cox,
    weighted_aalen_johansen,
)


def _nhis_recent_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": ["alive", "cvd", "competing", "prevalent"],
            "survey_year": pd.Series([2018, 2018, 2017, 2016], dtype="Int64"),
            "interview_quarter": pd.Series([4, 2, 3, 1], dtype="Int64"),
            "age": pd.Series([50, 60, 70, 55], dtype="Int64"),
            "sex": pd.Series(["female", "male", "female", "male"], dtype="string"),
            "bmi": [25.0, 30.0, 28.0, 26.0],
            "smoking_status": pd.Series(
                ["never", "former", "current", "never"], dtype="string"
            ),
            "hypertension_dx": pd.Series([False, True, True, False], dtype="boolean"),
            "diabetes_dx": pd.Series([False, True, False, False], dtype="boolean"),
            "general_health": pd.Series(
                ["excellent", "fair", "good", "very_good"], dtype="string"
            ),
            "mortality_eligible": pd.Series([True, True, True, True], dtype="boolean"),
            "mortality_weight": [4.0, 8.0, 12.0, 4.0],
            "deceased": pd.Series([False, True, True, False], dtype="boolean"),
            "ucod_leading": pd.Series([pd.NA, "001", "010", pd.NA], dtype="string"),
            "death_year": pd.Series([pd.NA, 2018, 2019, pd.NA], dtype="Int64"),
            "death_quarter": pd.Series([pd.NA, 2, 1, pd.NA], dtype="Int64"),
            "strata": pd.Series([1, 1, 2, 2], dtype="Int64"),
            "psu": pd.Series([1, 2, 1, 2], dtype="Int64"),
            "race_ethnicity": pd.Series(
                ["non_hispanic_white"] * 4, dtype="string"
            ),
            "coronary_heart_disease": pd.Series(
                [False, False, False, True], dtype="boolean"
            ),
            "angina_pectoris": pd.Series([False] * 4, dtype="boolean"),
            "myocardial_infarction": pd.Series([False] * 4, dtype="boolean"),
            "stroke": pd.Series([False] * 4, dtype="boolean"),
        }
    )


def test_nhis_recent_quarter_contract_and_competing_events() -> None:
    cohort, flow = build_nhis_recent_cohort(_nhis_recent_frame())
    assert cohort["participant_id"].tolist() == ["alive", "cvd", "competing"]
    assert cohort["event_type"].tolist() == [0, 1, 2]
    assert cohort["followup_time"].tolist() == [5, 1, 7]
    assert cohort["analysis_weight"].tolist() == [1.0, 2.0, 3.0]
    assert set(cohort["followup_time_unit"]) == {"quarter"}
    assert int(flow.iloc[-1]["n_participants"]) == 3


def test_nhis_recent_rejects_death_after_administrative_end() -> None:
    frame = _nhis_recent_frame()
    frame.loc[1, "death_year"] = 2020
    with pytest.raises(Stage7Error, match="after the 2019-Q4"):
        build_nhis_recent_cohort(frame)


def _nhanes_recent_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CYCLE": ["2015-2016", "2015-2016", "2017-2018", "2017-2018"],
            "SEQN": [1, 2, 3, 4],
            "RIDAGEYR": [50, 60, 70, 55],
            "RIAGENDR": [2, 1, 2, 1],
            "RIDRETH1": [3, 4, 1, 5],
            "WTMEC2YR": [2.0, 4.0, 6.0, 8.0],
            "SDMVSTRA": [119, 119, 120, 120],
            "SDMVPSU": [1, 2, 1, 2],
            "BMXBMI": [25.0, 30.0, 28.0, 26.0],
            "SMQ020": [2, 1, 1, 2],
            "SMQ040": [np.nan, 3, 1, np.nan],
            "BPQ020": [2, 1, 1, 2],
            "DIQ010": [2, 1, 2, 2],
            "HUQ010": [1, 4, 3, 2],
            "MCQ160B": [2, 2, 2, 2],
            "MCQ160C": [2, 2, 2, 1],
            "MCQ160D": [2, 2, 2, 2],
            "MCQ160E": [2, 2, 2, 2],
            "MCQ160F": [2, 2, 2, 2],
            "ELIGSTAT": [1, 1, 1, 1],
            "MORTSTAT": [0, 1, 1, 0],
            "UCOD_LEADING": [pd.NA, "001", "010", pd.NA],
            "PERMTH_INT": [48, 12, 24, 36],
        }
    )


def test_nhanes_recent_uses_two_cycle_weight_and_month_time() -> None:
    cohort, flow = build_nhanes_recent_cohort(_nhanes_recent_frame())
    assert cohort["participant_id"].tolist() == [
        "2015-2016:1",
        "2015-2016:2",
        "2017-2018:3",
    ]
    assert cohort["event_type"].tolist() == [0, 1, 2]
    assert cohort["analysis_weight"].tolist() == [1.0, 2.0, 3.0]
    assert set(cohort["followup_time_unit"]) == {"month"}
    assert int(flow.iloc[-1]["n_participants"]) == 3


def _analysis_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cohort": ["synthetic"] * 8,
            "participant_id": [f"p{i}" for i in range(8)],
            "strata": [1, 1, 1, 1, 2, 2, 2, 2],
            "psu": [1, 1, 2, 2, 1, 1, 2, 2],
            "analysis_weight": [1.0] * 8,
            "followup_time": [1, 2, 3, 4, 5, 6, 7, 8],
            "event_type": [1, 2, 1, 0, 2, 1, 0, 0],
            "predicted_risk": [0.55, 0.45, 0.40, 0.60, 0.35, 0.50, 0.30, 0.65],
            "linear_predictor": [0.2, -0.1, -0.3, 0.4, -0.5, 0.0, -0.7, 0.6],
        }
    )


def test_aalen_johansen_keeps_competing_death_in_cif() -> None:
    result = weighted_aalen_johansen(_analysis_frame(), [2, 8])
    cif = result[["cif_cvd", "cif_competing", "survival"]].to_numpy(dtype=float)
    assert cif[0, 0] > 0
    assert cif[0, 1] > 0
    total = float(cif[-1].sum())
    assert total == pytest.approx(1)


def test_weighted_cox_cif_and_ipcw_metrics_are_finite() -> None:
    frame = _analysis_frame()
    fit1 = fit_weighted_cause_specific_cox(frame, cause=1)
    fit2 = fit_weighted_cause_specific_cox(frame, cause=2)
    predictions = predict_cif_from_cause_specific_cox(
        frame["linear_predictor"].to_numpy(), fit1, fit2, [4, 8]
    )
    assert np.all(predictions[8] >= predictions[4])
    metrics = ipcw_cumulative_incidence_metrics(frame, predictions[4], 4)
    assert 0 <= float(metrics["time_dependent_auc"]) <= 1
    assert float(metrics["ipcw_brier"]) >= 0


def test_weighted_cox_is_invariant_to_common_weight_multiplier() -> None:
    frame = _analysis_frame()
    fit = fit_weighted_cause_specific_cox(frame, cause=1)
    scaled = frame.copy()
    scaled["analysis_weight"] *= 1_000_000
    scaled_fit = fit_weighted_cause_specific_cox(scaled, cause=1)
    assert scaled_fit.beta == pytest.approx(fit.beta)
    assert scaled_fit.standard_error == pytest.approx(fit.standard_error)


def test_stage7_config_locks_recent_periods_and_support_rule() -> None:
    config = load_stage7_config("configs/stage7_secondary.yaml")
    assert config["cohorts"]["nhis_recent"]["years"] == [2015, 2016, 2017, 2018]
    assert config["cohorts"]["nhanes_recent"]["cycles"] == [
        "2015-2016",
        "2017-2018",
    ]
    assert config["analysis"]["support_rule"]["minimum_cvd_events"] == 100
