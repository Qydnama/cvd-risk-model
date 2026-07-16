"""Stage 7 secondary censored and competing-risk analysis.

This module is deliberately isolated from the Stage 1-6 cohort builders and
the locked primary model code.  It applies the immutable minimal pipeline to
the 2015-2018 recent cohorts, but it never refits or serializes that model.
Only aggregate outputs are persisted.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from cvd_risk.data.cohort import derive_baseline_cvd_status
from cvd_risk.data.harmonize import (
    BASELINE_CVD_COLUMNS,
    harmonize_nhis,
    verify_codebook_binding,
)
from cvd_risk.data.load_nhanes import load_nhanes_cycles
from cvd_risk.data.load_nhis import discover_nhis_input, load_nhis
from cvd_risk.modeling.baselines import FittedBaselineModel
from cvd_risk.provenance import sha256_file

NHIS_RECENT_YEARS = (2015, 2016, 2017, 2018)
NHANES_RECENT_CYCLES = ("2015-2016", "2017-2018")
NHANES_RECENT_START_YEAR = {"2015-2016": 2015, "2017-2018": 2017}
CVD_CAUSES = frozenset({"001", "005"})
VALID_CAUSES = frozenset(f"{value:03d}" for value in range(1, 11))
FAST_FEATURES = (
    "age",
    "sex",
    "bmi",
    "smoking_status",
    "hypertension_dx",
    "diabetes_dx",
    "general_health",
)


class Stage7Error(ValueError):
    """Raised when the secondary-analysis contract is violated."""


@dataclass(frozen=True)
class CoxFit:
    """One-predictor weighted cause-specific Cox fit with Breslow ties."""

    cause: int
    beta: float
    standard_error: float
    score_center: float
    converged: bool
    iterations: int
    event_times: np.ndarray
    baseline_hazard_increments: np.ndarray


def _normalized_text_sha256(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise Stage7Error(f"{context} is missing required columns: {missing}")


def _numeric(series: pd.Series, name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    invalid = series.notna() & values.isna()
    if invalid.any():
        raise Stage7Error(f"{name} contains non-numeric values")
    return values


def _integer(series: pd.Series, name: str) -> pd.Series:
    values = _numeric(series, name)
    fractional = values.notna() & values.mod(1).ne(0)
    if fractional.any():
        raise Stage7Error(f"{name} contains non-integer values")
    return values.astype("Int64")


def _yes_no(series: pd.Series, name: str) -> pd.Series:
    values = _integer(series, name)
    unexpected = values.notna() & ~values.isin([1, 2, 7, 9])
    if unexpected.any():
        found = sorted(values.loc[unexpected].astype(int).unique().tolist())
        raise Stage7Error(f"{name} contains unexpected codes: {found}")
    output = pd.Series(pd.NA, index=series.index, dtype="boolean")
    output.loc[values.eq(1)] = True
    output.loc[values.eq(2)] = False
    return output


def _ucod(series: pd.Series, name: str) -> pd.Series:
    output = series.astype("string").str.strip()
    output = output.mask(output.isin(["", "."]))
    numeric = pd.to_numeric(output, errors="coerce")
    invalid_text = output.notna() & numeric.isna()
    if invalid_text.any():
        raise Stage7Error(f"{name} contains non-numeric cause codes")
    normalized = numeric.map(lambda value: pd.NA if pd.isna(value) else f"{int(value):03d}")
    invalid = normalized.notna() & ~normalized.isin(VALID_CAUSES)
    if invalid.any():
        found = sorted(normalized.loc[invalid].astype(str).unique().tolist())
        raise Stage7Error(f"{name} contains invalid cause codes: {found}")
    return normalized.astype("string")


def load_stage7_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise Stage7Error("Stage 7 config schema_version must be 1")
    if payload.get("stage") != "stage7_secondary_censored_competing_risk":
        raise Stage7Error("Unexpected Stage 7 identifier")
    immutable = payload.get("immutable_primary_model", {})
    required_false_changes = (
        "model_refit_forbidden",
        "calibrator_refit_forbidden",
        "cutoff_change_forbidden",
        "secondary_results_must_not_select_primary_model",
    )
    if not all(immutable.get(field) is True for field in required_false_changes):
        raise Stage7Error("Stage 7 must forbid changes to the primary model and selection")
    cohorts = payload.get("cohorts", {})
    if cohorts.get("nhis_recent", {}).get("years") != list(NHIS_RECENT_YEARS):
        raise Stage7Error("NHIS recent years must be explicitly enumerated")
    if cohorts.get("nhanes_recent", {}).get("cycles") != list(NHANES_RECENT_CYCLES):
        raise Stage7Error("NHANES recent cycles must be explicitly enumerated")
    if cohorts["nhis_recent"].get("horizons") != [4, 8, 12]:
        raise Stage7Error("NHIS horizons must remain 4/8/12 quarters")
    if cohorts["nhanes_recent"].get("horizons") != [12, 24, 36]:
        raise Stage7Error("NHANES horizons must remain 12/24/36 months")
    support = payload.get("analysis", {}).get("support_rule", {})
    if support.get("minimum_cvd_events") != 100 or support.get(
        "minimum_weighted_at_risk_proportion"
    ) != 0.20:
        raise Stage7Error("Stage 7 support rule changed")
    return payload


def verify_immutable_primary_bindings(config: dict[str, Any], root: Path) -> dict[str, str]:
    immutable = config["immutable_primary_model"]
    bindings = {
        "pipeline": sha256_file(root / immutable["pipeline_path"]),
        "lock": _normalized_text_sha256(root / immutable["lock_path"]),
        "amendment": _normalized_text_sha256(root / immutable["amendment_path"]),
    }
    expected = {
        "pipeline": immutable["pipeline_sha256"],
        "lock": immutable["lock_sha256_normalized_lf"],
        "amendment": immutable["amendment_sha256_normalized_lf"],
    }
    mismatches = [name for name in bindings if bindings[name] != expected[name]]
    if mismatches:
        raise Stage7Error(f"Immutable primary bindings changed: {mismatches}")
    return bindings


def _flow_table(cohort: str, flags: list[tuple[str, pd.Series]]) -> pd.DataFrame:
    if not flags:
        raise Stage7Error("Cohort flow requires at least one rule")
    retained = pd.Series(True, index=flags[0][1].index)
    rows = [
        {
            "cohort": cohort,
            "step_order": 0,
            "step": "raw recent records",
            "n_participants": len(retained),
            "removed_from_previous": 0,
        }
    ]
    previous = len(retained)
    for order, (label, flag) in enumerate(flags, start=1):
        retained &= flag.fillna(False).astype(bool)
        current = int(retained.sum())
        rows.append(
            {
                "cohort": cohort,
                "step_order": order,
                "step": label,
                "n_participants": current,
                "removed_from_previous": previous - current,
            }
        )
        previous = current
    return pd.DataFrame(rows)


def build_nhis_recent_cohort(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the distinct NHIS 2015-2018 quarter-precision recent cohort."""

    required = {
        "participant_id",
        "survey_year",
        "interview_quarter",
        "age",
        "sex",
        "bmi",
        "smoking_status",
        "hypertension_dx",
        "diabetes_dx",
        "general_health",
        "mortality_eligible",
        "mortality_weight",
        "deceased",
        "ucod_leading",
        "death_year",
        "death_quarter",
        "strata",
        "psu",
        "race_ethnicity",
        *BASELINE_CVD_COLUMNS,
    }
    _require_columns(frame, required, "NHIS recent input")
    result = frame.loc[_integer(frame["survey_year"], "survey_year").isin(NHIS_RECENT_YEARS)].copy()
    if result.empty:
        raise Stage7Error("NHIS recent input contains no 2015-2018 records")
    if result["participant_id"].isna().any() or result["participant_id"].duplicated().any():
        raise Stage7Error("NHIS recent participant identifiers must be unique and nonmissing")

    year = _integer(result["survey_year"], "survey_year")
    interview_quarter = _integer(result["interview_quarter"], "interview_quarter")
    invalid_interview_quarter = ~interview_quarter.isin([1, 2, 3, 4])
    if invalid_interview_quarter.any():
        raise Stage7Error("NHIS recent interview quarter is outside 1..4")
    deceased = result["deceased"].astype("boolean")
    death_year = _integer(result["death_year"], "death_year")
    death_quarter = _integer(result["death_quarter"], "death_quarter")
    invalid_death_quarter = deceased.eq(True) & ~death_quarter.isin([1, 2, 3, 4])
    if invalid_death_quarter.any():
        raise Stage7Error("NHIS recent death quarter is outside 1..4")
    if (deceased.eq(True) & death_year.isna()).any():
        raise Stage7Error("NHIS recent death year is missing for a deceased participant")

    admin_index = 4 * 2019 + 3
    interview_index = 4 * year.astype(int) + (interview_quarter.astype(int) - 1)
    death_index = 4 * death_year.fillna(2019).astype(int) + (
        death_quarter.fillna(4).astype(int) - 1
    )
    death_after_admin = deceased.eq(True) & death_index.gt(admin_index)
    death_before_interview = deceased.eq(True) & death_index.lt(interview_index)
    if death_after_admin.any():
        raise Stage7Error("NHIS recent death occurs after the 2019-Q4 administrative end")
    if death_before_interview.any():
        raise Stage7Error("NHIS recent death occurs before the interview quarter")

    time = pd.Series(admin_index - interview_index + 1, index=result.index, dtype="Int64")
    time.loc[deceased.eq(True)] = (
        death_index.loc[deceased.eq(True)] - interview_index.loc[deceased.eq(True)] + 1
    ).astype("Int64")
    if time.dropna().le(0).any():
        raise Stage7Error("NHIS recent follow-up time must be positive")

    cause = result["ucod_leading"].astype("string")
    invalid_cause = deceased.eq(True) & cause.notna() & ~cause.isin(VALID_CAUSES)
    if invalid_cause.any():
        raise Stage7Error("NHIS recent cause of death is outside the public LMF categories")
    event = pd.Series(pd.NA, index=result.index, dtype="Int8")
    event.loc[deceased.eq(False)] = 0
    event.loc[deceased.eq(True) & cause.isin(CVD_CAUSES)] = 1
    event.loc[deceased.eq(True) & cause.isin(VALID_CAUSES.difference(CVD_CAUSES))] = 2

    baseline = derive_baseline_cvd_status(result)
    age = _numeric(result["age"], "age")
    linkage = result["mortality_eligible"].astype("boolean")
    raw_weight = _numeric(result["mortality_weight"], "mortality_weight")
    positive_weight = raw_weight.notna() & raw_weight.gt(0)
    flags = [
        ("year in 2015-2018 recent protocol", year.isin(NHIS_RECENT_YEARS)),
        ("known age", age.notna()),
        ("age 40-79", age.between(40, 79, inclusive="both")),
        ("mortality linkage eligible", linkage.eq(True)),
        ("positive MORTWTSA", positive_weight),
        ("exclude prevalent CVD", baseline.ne("present")),
        ("baseline CVD known absent", baseline.eq("absent")),
        ("known follow-up time", time.notna()),
        ("known competing-event type", event.notna()),
    ]
    flow = _flow_table("nhis_recent", flags)
    eligible = pd.Series(True, index=result.index)
    for _, flag in flags:
        eligible &= flag.fillna(False)
    cohort = result.loc[eligible].copy()
    cohort["followup_time"] = time.loc[eligible].astype(int)
    cohort["followup_time_unit"] = "quarter"
    cohort["followup_time_source"] = "NHIS_INTERVIEW_AND_DEATH_QUARTER"
    cohort["time_interval_precision"] = "quarter"
    cohort["event_type"] = event.loc[eligible].astype(int)
    cohort["analysis_weight"] = (raw_weight.loc[eligible] / 4.0).astype(float)
    cohort["baseline_cvd_status"] = baseline.loc[eligible]
    cohort["source"] = "NHIS"
    cohort["cohort"] = "nhis_recent"
    cohort["survey_cycle"] = ""
    return cohort.reset_index(drop=True), flow


def _nhanes_smoking(frame: pd.DataFrame) -> pd.Series:
    ever = _integer(frame["SMQ020"], "SMQ020")
    current = _integer(frame["SMQ040"], "SMQ040")
    invalid_ever = ever.notna() & ~ever.isin([1, 2, 7, 9])
    invalid_current = current.notna() & ~current.isin([1, 2, 3, 7, 9])
    if invalid_ever.any() or invalid_current.any():
        raise Stage7Error("NHANES recent smoking codes are outside the documented values")
    output = pd.Series(pd.NA, index=frame.index, dtype="string")
    output.loc[ever.eq(2)] = "never"
    output.loc[ever.eq(1) & current.eq(3)] = "former"
    output.loc[ever.eq(1) & current.isin([1, 2])] = "current"
    return output


def build_nhanes_recent_cohort(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the distinct NHANES 2015-2018 month-precision recent cohort."""

    required = {
        "CYCLE",
        "SEQN",
        "RIDAGEYR",
        "RIAGENDR",
        "RIDRETH1",
        "WTMEC2YR",
        "SDMVSTRA",
        "SDMVPSU",
        "BMXBMI",
        "SMQ020",
        "SMQ040",
        "BPQ020",
        "DIQ010",
        "HUQ010",
        "MCQ160B",
        "MCQ160C",
        "MCQ160D",
        "MCQ160E",
        "MCQ160F",
        "ELIGSTAT",
        "MORTSTAT",
        "UCOD_LEADING",
        "PERMTH_INT",
    }
    _require_columns(frame, required, "NHANES recent input")
    cycle = frame["CYCLE"].astype("string")
    if set(cycle.unique()) != set(NHANES_RECENT_CYCLES):
        raise Stage7Error("NHANES recent input must contain exactly the two recent cycles")
    seqn = _integer(frame["SEQN"], "SEQN")
    participant_id = cycle + ":" + seqn.astype("string")
    if participant_id.isna().any() or participant_id.duplicated().any():
        raise Stage7Error("NHANES recent participant identifiers must be unique and nonmissing")

    age = _numeric(frame["RIDAGEYR"], "RIDAGEYR")
    sex_code = _integer(frame["RIAGENDR"], "RIAGENDR")
    sex = sex_code.map({1: "male", 2: "female"}).astype("string")
    race_code = _integer(frame["RIDRETH1"], "RIDRETH1")
    race = race_code.map(
        {
            1: "mexican_american",
            2: "other_hispanic",
            3: "non_hispanic_white",
            4: "non_hispanic_black",
            5: "other_or_multiracial",
        }
    ).astype("string")
    health_code = _integer(frame["HUQ010"], "HUQ010")
    health = health_code.map(
        {1: "excellent", 2: "very_good", 3: "good", 4: "fair", 5: "poor"}
    ).astype("string")
    hypertension = _yes_no(frame["BPQ020"], "BPQ020")
    diabetes_code = _integer(frame["DIQ010"], "DIQ010")
    invalid_diabetes = diabetes_code.notna() & ~diabetes_code.isin([1, 2, 3, 7, 9])
    if invalid_diabetes.any():
        raise Stage7Error("DIQ010 contains unexpected codes")
    diabetes = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    diabetes.loc[diabetes_code.eq(1)] = True
    diabetes.loc[diabetes_code.eq(2)] = False

    baseline_frame = pd.DataFrame(
        {
            "coronary_heart_disease": _yes_no(frame["MCQ160C"], "MCQ160C"),
            "angina_pectoris": _yes_no(frame["MCQ160D"], "MCQ160D"),
            "myocardial_infarction": _yes_no(frame["MCQ160E"], "MCQ160E"),
            "stroke": _yes_no(frame["MCQ160F"], "MCQ160F"),
        },
        index=frame.index,
    )
    baseline = derive_baseline_cvd_status(baseline_frame)
    eligibility = _integer(frame["ELIGSTAT"], "ELIGSTAT")
    status = _integer(frame["MORTSTAT"], "MORTSTAT")
    followup = _integer(frame["PERMTH_INT"], "PERMTH_INT")
    invalid_time = followup.notna() & ((followup < 0) | (followup > 374))
    if invalid_time.any():
        raise Stage7Error("NHANES recent PERMTH_INT is outside 0..374")
    cause = _ucod(frame["UCOD_LEADING"], "UCOD_LEADING")
    event = pd.Series(pd.NA, index=frame.index, dtype="Int8")
    event.loc[status.eq(0)] = 0
    event.loc[status.eq(1) & cause.isin(CVD_CAUSES)] = 1
    event.loc[status.eq(1) & cause.isin(VALID_CAUSES.difference(CVD_CAUSES))] = 2

    raw_weight = _numeric(frame["WTMEC2YR"], "WTMEC2YR")
    positive_weight = raw_weight.notna() & raw_weight.gt(0)
    flags = [
        ("cycle in 2015-2018 recent protocol", cycle.isin(NHANES_RECENT_CYCLES)),
        ("known age", age.notna()),
        ("age 40-79", age.between(40, 79, inclusive="both")),
        ("mortality linkage eligible", eligibility.eq(1)),
        ("positive WTMEC2YR", positive_weight),
        ("exclude prevalent CVD", baseline.ne("present")),
        ("baseline CVD known absent", baseline.eq("absent")),
        ("known PERMTH_INT", followup.notna()),
        ("known competing-event type", event.notna()),
    ]
    flow = _flow_table("nhanes_recent", flags)
    eligible = pd.Series(True, index=frame.index)
    for _, flag in flags:
        eligible &= flag.fillna(False)
    output = pd.DataFrame(index=frame.index)
    output["cohort"] = "nhanes_recent"
    output["source"] = "NHANES"
    output["participant_id"] = participant_id
    output["survey_cycle"] = cycle
    output["survey_year"] = cycle.map(NHANES_RECENT_START_YEAR).astype("Int64")
    output["strata"] = _integer(frame["SDMVSTRA"], "SDMVSTRA")
    output["psu"] = _integer(frame["SDMVPSU"], "SDMVPSU")
    output["analysis_weight"] = (raw_weight / 2.0).astype(float)
    # Keep the raw numeric representation until after eligibility filtering.
    # SAS XPORT can encode zero as a tiny machine-zero float for infants; those
    # rows are outside the locked 40-79 cohort and must not break construction.
    output["age"] = age.astype(float)
    output["sex"] = sex
    output["bmi"] = _numeric(frame["BMXBMI"], "BMXBMI").astype(float)
    output["smoking_status"] = _nhanes_smoking(frame)
    output["hypertension_dx"] = hypertension
    output["diabetes_dx"] = diabetes
    output["general_health"] = health
    output["race_ethnicity"] = race
    output["baseline_cvd_status"] = baseline
    output["followup_time"] = followup
    output["followup_time_unit"] = "month"
    output["followup_time_source"] = "LMF_PERMTH_INT"
    output["time_interval_precision"] = "month"
    output["event_type"] = event
    cohort = output.loc[eligible].copy().reset_index(drop=True)
    return cohort, flow


def _validate_analysis_frame(frame: pd.DataFrame) -> None:
    required = {
        "cohort",
        "participant_id",
        "strata",
        "psu",
        "analysis_weight",
        "followup_time",
        "event_type",
        "predicted_risk",
        "linear_predictor",
    }
    _require_columns(frame, required, "Stage 7 analysis frame")
    if frame.empty or frame["participant_id"].duplicated().any():
        raise Stage7Error("Stage 7 analysis frame is empty or has duplicate identifiers")
    arrays = frame[["analysis_weight", "followup_time", "predicted_risk", "linear_predictor"]]
    if not np.isfinite(arrays.to_numpy(dtype=float)).all():
        raise Stage7Error("Stage 7 analysis frame contains non-finite numeric values")
    if (frame["analysis_weight"] <= 0).any() or (frame["followup_time"] < 0).any():
        raise Stage7Error("Stage 7 weights must be positive and times nonnegative")
    if not set(frame["event_type"].astype(int).unique()).issubset({0, 1, 2}):
        raise Stage7Error("Stage 7 event_type must be 0, 1, or 2")


def weighted_aalen_johansen(frame: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Return weighted Aalen-Johansen CIF and cause-specific hazards."""

    _validate_analysis_frame(frame)
    time = frame["followup_time"].to_numpy(dtype=float)
    event = frame["event_type"].to_numpy(dtype=int)
    weight = frame["analysis_weight"].to_numpy(dtype=float)
    unique_times = np.sort(np.unique(time[event != 0]))
    survival = 1.0
    cif1 = 0.0
    cif2 = 0.0
    cumulative_hazard1 = 0.0
    cumulative_hazard2 = 0.0
    trajectory: list[dict[str, float]] = []
    for current in unique_times:
        risk_weight = float(weight[time >= current].sum())
        if risk_weight <= 0:
            raise Stage7Error("Aalen-Johansen risk set has nonpositive weight")
        d1 = float(weight[(time == current) & (event == 1)].sum())
        d2 = float(weight[(time == current) & (event == 2)].sum())
        hazard1 = d1 / risk_weight
        hazard2 = d2 / risk_weight
        if hazard1 + hazard2 > 1.0 + 1e-12:
            raise Stage7Error("Aalen-Johansen hazard increment exceeds one")
        cif1 += survival * hazard1
        cif2 += survival * hazard2
        survival *= max(0.0, 1.0 - hazard1 - hazard2)
        cumulative_hazard1 += hazard1
        cumulative_hazard2 += hazard2
        trajectory.append(
            {
                "time": float(current),
                "survival": survival,
                "cif_cvd": cif1,
                "cif_competing": cif2,
                "cumhaz_cvd": cumulative_hazard1,
                "cumhaz_competing": cumulative_hazard2,
            }
        )
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        available = [row for row in trajectory if row["time"] <= horizon]
        current = available[-1] if available else {
            "survival": 1.0,
            "cif_cvd": 0.0,
            "cif_competing": 0.0,
            "cumhaz_cvd": 0.0,
            "cumhaz_competing": 0.0,
        }
        rows.append(
            {
                "horizon": horizon,
                **{key: current[key] for key in current if key != "time"},
                "raw_at_risk": int((time >= horizon).sum()),
                "weighted_at_risk": float(weight[time >= horizon].sum()),
                "weighted_at_risk_proportion": float(
                    weight[time >= horizon].sum() / weight.sum()
                ),
                "raw_cvd_events_by_horizon": int(((event == 1) & (time <= horizon)).sum()),
                "raw_competing_events_by_horizon": int(
                    ((event == 2) & (time <= horizon)).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def fit_weighted_cause_specific_cox(
    frame: pd.DataFrame,
    *,
    cause: int,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> CoxFit:
    """Fit one weighted cause-specific Cox coefficient using Breslow ties."""

    if cause not in {1, 2}:
        raise Stage7Error("Cause-specific Cox cause must be 1 or 2")
    _validate_analysis_frame(frame)
    time = frame["followup_time"].to_numpy(dtype=float)
    event = frame["event_type"].to_numpy(dtype=int)
    raw_weight = frame["analysis_weight"].to_numpy(dtype=float)
    # Cox coefficients and Breslow baseline increments are invariant to a
    # common weight multiplier. Normalizing prevents public survey totals from
    # being misinterpreted as frequency-replicated sample size in the naive
    # model-based information matrix.
    weight = raw_weight / raw_weight.mean()
    score = frame["linear_predictor"].to_numpy(dtype=float)
    center = float(np.average(score, weights=weight))
    x = score - center
    event_times = np.sort(np.unique(time[event == cause]))
    if len(event_times) == 0:
        raise Stage7Error(f"No events are available for cause {cause}")
    beta = 0.0
    converged = False
    information = math.nan
    iterations = 0
    for current_iteration in range(1, max_iterations + 1):
        iterations = current_iteration
        linear = np.clip(beta * x, -50.0, 50.0)
        relative_risk = np.exp(linear)
        score_total = 0.0
        information = 0.0
        for current in event_times:
            risk = time >= current
            cases = (time == current) & (event == cause)
            weighted_risk = weight[risk] * relative_risk[risk]
            s0 = float(weighted_risk.sum())
            if s0 <= 0:
                raise Stage7Error("Cox risk denominator is nonpositive")
            s1 = float(np.sum(weighted_risk * x[risk]))
            s2 = float(np.sum(weighted_risk * x[risk] ** 2))
            d0 = float(weight[cases].sum())
            score_total += float(np.sum(weight[cases] * x[cases])) - d0 * s1 / s0
            information += d0 * (s2 / s0 - (s1 / s0) ** 2)
        if not math.isfinite(information) or information <= 0:
            raise Stage7Error("Cox information is nonpositive or non-finite")
        step = score_total / information
        beta += step
        if abs(step) <= tolerance:
            converged = True
            break
    if not converged:
        raise Stage7Error(f"Cause-specific Cox did not converge for cause {cause}")

    relative_risk = np.exp(np.clip(beta * x, -50.0, 50.0))
    increments = []
    for current in event_times:
        risk = time >= current
        cases = (time == current) & (event == cause)
        denominator = float(np.sum(weight[risk] * relative_risk[risk]))
        increments.append(float(weight[cases].sum()) / denominator)
    return CoxFit(
        cause=cause,
        beta=float(beta),
        standard_error=float(math.sqrt(1.0 / information)),
        score_center=center,
        converged=converged,
        iterations=iterations,
        event_times=event_times,
        baseline_hazard_increments=np.asarray(increments, dtype=float),
    )


def predict_cif_from_cause_specific_cox(
    score: np.ndarray,
    fit_cvd: CoxFit,
    fit_competing: CoxFit,
    horizons: list[int],
) -> dict[int, np.ndarray]:
    """Reconstruct individual CVD CIF jointly from both cause-specific hazards."""

    values = np.asarray(score, dtype=float)
    if not np.isfinite(values).all():
        raise Stage7Error("Cox prediction score contains non-finite values")
    rr1 = np.exp(np.clip(fit_cvd.beta * (values - fit_cvd.score_center), -50.0, 50.0))
    rr2 = np.exp(
        np.clip(fit_competing.beta * (values - fit_competing.score_center), -50.0, 50.0)
    )
    increments1 = dict(
        zip(fit_cvd.event_times.tolist(), fit_cvd.baseline_hazard_increments.tolist(), strict=True)
    )
    increments2 = dict(
        zip(
            fit_competing.event_times.tolist(),
            fit_competing.baseline_hazard_increments.tolist(),
            strict=True,
        )
    )
    event_times = sorted(set(increments1).union(increments2))
    survival = np.ones(len(values), dtype=float)
    cif = np.zeros(len(values), dtype=float)
    predictions: dict[int, np.ndarray] = {}
    for current in event_times:
        h1 = rr1 * increments1.get(current, 0.0)
        h2 = rr2 * increments2.get(current, 0.0)
        total = h1 + h2
        event_probability = 1.0 - np.exp(-total)
        share1 = np.divide(h1, total, out=np.zeros_like(h1), where=total > 0)
        cif += survival * event_probability * share1
        survival *= np.exp(-total)
        for horizon in horizons:
            if current <= horizon and horizon not in predictions:
                next_times = [value for value in event_times if current < value <= horizon]
                if not next_times:
                    predictions[horizon] = cif.copy()
    for horizon in horizons:
        default = (
            cif.copy()
            if event_times and horizon >= event_times[-1]
            else np.zeros(len(values))
        )
        predictions.setdefault(horizon, default)
        if not np.isfinite(predictions[horizon]).all():
            raise Stage7Error("Cox CIF prediction is non-finite")
    return predictions


def _censoring_survival(
    time: np.ndarray, event: np.ndarray, weight: np.ndarray
) -> tuple[dict[float, float], dict[float, float]]:
    before: dict[float, float] = {}
    after: dict[float, float] = {}
    survival = 1.0
    for current in np.sort(np.unique(time)):
        before[float(current)] = survival
        risk = float(weight[time >= current].sum())
        censored = float(weight[(time == current) & (event == 0)].sum())
        if risk <= 0:
            raise Stage7Error("Censoring risk set has nonpositive weight")
        survival *= max(0.0, 1.0 - censored / risk)
        after[float(current)] = survival
    return before, after


def _survival_before_horizon(after: dict[float, float], horizon: float) -> float:
    eligible = [time for time in after if time < horizon]
    return after[max(eligible)] if eligible else 1.0


def _weighted_auc(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float:
    if set(np.unique(labels)) != {0, 1}:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    y = labels[order]
    s = scores[order]
    w = weights[order]
    total_case = float(w[y == 1].sum())
    total_control = float(w[y == 0].sum())
    if total_case <= 0 or total_control <= 0:
        return math.nan
    contribution = 0.0
    control_before = 0.0
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and s[end] == s[start]:
            end += 1
        group_case = float(w[start:end][y[start:end] == 1].sum())
        group_control = float(w[start:end][y[start:end] == 0].sum())
        contribution += group_case * (control_before + 0.5 * group_control)
        control_before += group_control
        start = end
    return contribution / (total_case * total_control)


def ipcw_cumulative_incidence_metrics(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    horizon: int,
) -> dict[str, float | int]:
    """Calculate CIF AUC and Brier using reverse-KM administrative-censoring weights."""

    _validate_analysis_frame(frame)
    time = frame["followup_time"].to_numpy(dtype=float)
    event = frame["event_type"].to_numpy(dtype=int)
    survey_weight = frame["analysis_weight"].to_numpy(dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    if len(predicted) != len(frame) or not np.isfinite(predicted).all():
        raise Stage7Error("CIF predictions are invalid")
    before, after = _censoring_survival(time, event, survey_weight)
    g_horizon = _survival_before_horizon(after, float(horizon))
    if g_horizon <= 0:
        raise Stage7Error("Censoring survival is zero before the requested horizon")
    labels: list[int] = []
    scores: list[float] = []
    metric_weights: list[float] = []
    residual_sum = 0.0
    for index in range(len(frame)):
        if event[index] in {1, 2} and time[index] <= horizon:
            g = before[float(time[index])]
            if g <= 0:
                continue
            label = int(event[index] == 1)
            ipcw = survey_weight[index] / g
        elif time[index] >= horizon:
            label = 0
            ipcw = survey_weight[index] / g_horizon
        else:
            continue
        labels.append(label)
        scores.append(float(predicted[index]))
        metric_weights.append(float(ipcw))
        residual_sum += ipcw * (label - predicted[index]) ** 2
    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    metric_weight_array = np.asarray(metric_weights, dtype=float)
    return {
        "time_dependent_auc": float(
            _weighted_auc(labels_array, scores_array, metric_weight_array)
        ),
        "ipcw_brier": float(residual_sum / survey_weight.sum()),
        "metric_known_n": len(labels),
        "ipcw_weight_sum": float(metric_weight_array.sum()),
        "censoring_survival_before_horizon": float(g_horizon),
    }


def _weighted_quantile_boundaries(
    values: np.ndarray, weights: np.ndarray, bins: int
) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    targets = np.linspace(0.0, cumulative[-1], bins + 1)[1:-1]
    interior = np.interp(targets, cumulative, sorted_values)
    edges = np.concatenate(([-np.inf], np.unique(interior), [np.inf]))
    if len(edges) < 3:
        raise Stage7Error("Calibration predictions do not support at least two bins")
    return edges


def cumulative_incidence_calibration(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    horizon: int,
    *,
    bins: int,
) -> pd.DataFrame:
    weights = frame["analysis_weight"].to_numpy(dtype=float)
    values = np.asarray(prediction, dtype=float)
    edges = _weighted_quantile_boundaries(values, weights, bins)
    assignment = np.searchsorted(edges[1:-1], values, side="right") + 1
    rows: list[dict[str, Any]] = []
    for number in range(1, len(edges)):
        selected = assignment == number
        subset = frame.loc[selected].copy()
        if subset.empty:
            continue
        observed = weighted_aalen_johansen(subset, [horizon]).iloc[0]
        rows.append(
            {
                "horizon": horizon,
                "bin": number,
                "lower_boundary": edges[number - 1],
                "upper_boundary": edges[number],
                "n": int(selected.sum()),
                "events": int(
                    ((subset["event_type"] == 1) & (subset["followup_time"] <= horizon)).sum()
                ),
                "weight_sum": float(weights[selected].sum()),
                "weighted_mean_prediction": float(
                    np.average(values[selected], weights=weights[selected])
                ),
                "observed_cif": float(observed["cif_cvd"]),
            }
        )
    return pd.DataFrame(rows)


def _attach_frozen_predictions(
    cohort: pd.DataFrame,
    model: FittedBaselineModel,
    *,
    probability_clip: float,
) -> pd.DataFrame:
    if model.route != "minimal":
        raise Stage7Error("Stage 7 requires the immutable minimal route")
    _require_columns(cohort, FAST_FEATURES, "Stage 7 frozen-model input")
    probability = np.asarray(model.predict_proba(cohort, calibrated=True), dtype=float)
    if len(probability) != len(cohort) or not np.isfinite(probability).all():
        raise Stage7Error("Frozen model produced invalid recent-cohort probabilities")
    if ((probability < 0) | (probability > 1)).any():
        raise Stage7Error("Frozen model produced probabilities outside [0, 1]")
    bounded = np.clip(probability, probability_clip, 1.0 - probability_clip)
    output = cohort.copy()
    output["predicted_risk"] = probability
    output["linear_predictor"] = np.log(bounded / (1.0 - bounded))
    return output


def _event_count_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, group in frame.groupby("cohort", sort=False, observed=True):
        time = group["followup_time"].to_numpy(dtype=float)
        event = group["event_type"].to_numpy(dtype=int)
        weight = group["analysis_weight"].to_numpy(dtype=float)
        rows.append(
            {
                "cohort": cohort,
                "n": len(group),
                "censored": int((event == 0).sum()),
                "cvd_deaths": int((event == 1).sum()),
                "competing_deaths": int((event == 2).sum()),
                "weight_sum": float(weight.sum()),
                "followup_min": float(np.min(time)),
                "followup_q25": float(np.quantile(time, 0.25)),
                "followup_median": float(np.median(time)),
                "followup_q75": float(np.quantile(time, 0.75)),
                "followup_max": float(np.max(time)),
                "followup_time_unit": str(group["followup_time_unit"].iloc[0]),
                "followup_time_source": str(group["followup_time_source"].iloc[0]),
                "role": "secondary_censored_competing_risk_only",
            }
        )
    return pd.DataFrame(rows)


def _fit_summary_rows(
    cohort: str,
    time_unit: str,
    role: str,
    fit: CoxFit,
    horizons: list[int],
) -> list[dict[str, Any]]:
    rows = []
    for horizon in horizons:
        cumulative_baseline = float(
            fit.baseline_hazard_increments[fit.event_times <= horizon].sum()
        )
        rows.append(
            {
                "cohort": cohort,
                "time_unit": time_unit,
                "cause": "cvd_death" if fit.cause == 1 else "competing_death",
                "horizon": horizon,
                "cox_role": role,
                "predictor": "frozen_sigmoid_linear_predictor",
                "coefficient": fit.beta,
                "model_based_standard_error": fit.standard_error,
                "hazard_ratio_per_lp_unit": math.exp(fit.beta),
                "hazard_ratio_ci_lower_model_based": math.exp(
                    fit.beta - 1.96 * fit.standard_error
                ),
                "hazard_ratio_ci_upper_model_based": math.exp(
                    fit.beta + 1.96 * fit.standard_error
                ),
                "cumulative_baseline_hazard": cumulative_baseline,
                "converged": fit.converged,
                "iterations": fit.iterations,
                "ties": "breslow",
                "variance_note": "model_based_not_survey_cluster_robust",
            }
        )
    return rows


def _write_cif_figure(path: Path, aj: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cohorts = aj["cohort"].drop_duplicates().tolist()
    figure, axes = plt.subplots(1, len(cohorts), figsize=(6 * len(cohorts), 4.5), squeeze=False)
    for axis, cohort in zip(axes[0], cohorts, strict=True):
        selected = aj.loc[aj["cohort"].eq(cohort)].sort_values("horizon")
        axis.plot(selected["horizon"], selected["cif_cvd"], marker="o", label="CVD death")
        axis.plot(
            selected["horizon"],
            selected["cif_competing"],
            marker="o",
            label="Competing death",
        )
        axis.set_title(f"{cohort}: weighted Aalen-Johansen")
        axis.set_xlabel(str(selected["time_unit"].iloc[0]))
        axis.set_ylabel("Cumulative incidence")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Stage 7 secondary/exploratory competing-risk CIF")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_calibration_figure(path: Path, calibration: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        (cohort, int(horizon))
        for cohort, horizon in calibration[["cohort", "horizon"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ]
    columns = 3
    rows = math.ceil(len(panels) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.2 * rows), squeeze=False)
    for axis, (cohort, horizon) in zip(axes.flat, panels, strict=False):
        selected = calibration.loc[
            calibration["cohort"].eq(cohort) & calibration["horizon"].eq(horizon)
        ].sort_values("bin")
        maximum = max(
            float(selected["weighted_mean_prediction"].max()),
            float(selected["observed_cif"].max()),
            1e-4,
        )
        axis.plot([0, maximum], [0, maximum], linestyle="--", color="black", label="Identity")
        axis.plot(
            selected["weighted_mean_prediction"],
            selected["observed_cif"],
            marker="o",
            label="Weighted bins",
        )
        axis.set_title(f"{cohort}, t={horizon}")
        axis.set_xlabel("Model-based predicted CIF")
        axis.set_ylabel("Aalen-Johansen observed CIF")
        axis.grid(alpha=0.25)
    for axis in list(axes.flat)[len(panels) :]:
        axis.set_visible(False)
    figure.suptitle("Stage 7 cumulative-incidence calibration (secondary/exploratory)")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_report(
    path: Path,
    *,
    event_counts: pd.DataFrame,
    aj: pd.DataFrame,
    metrics: pd.DataFrame,
    support: pd.DataFrame,
    calibration_bins: int,
    bindings: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage 7 secondary censored/competing-risk analysis",
        "",
        "## Mandatory interpretation",
        "",
        "All results are **secondary/exploratory**. They do not replace the fixed five-year "
        "NHIS development model or the Stage 6 temporal/external validation. The immutable "
        "minimal pipeline was applied without refitting; preprocessing, sigmoid calibration, "
        "and the fixed cutoff were not changed. Stage 7 results were not used for post-hoc "
        "primary-model or threshold selection.",
        "",
        "The recent cohorts are distinct censored cohorts. No participant lacking complete "
        "five-year follow-up is represented as an ordinary negative case in a primary binary "
        "table.",
        "",
        "## Immutable bindings",
        "",
        f"- Pipeline SHA-256: `{bindings['pipeline']}`.",
        f"- Original lock normalized-LF SHA-256: `{bindings['lock']}`.",
        f"- Lock amendment normalized-LF SHA-256: `{bindings['amendment']}`.",
        "",
        "## Recent-cohort counts and follow-up",
        "",
        event_counts.to_markdown(index=False),
        "",
        "NHIS time is the observed quarter interval number from interview through death or "
        "the administrative end at 2019-Q4; `+1` gives positive time for a death in the "
        "interview quarter and does not imply monthly precision. NHANES uses public LMF "
        "`PERMTH_INT` in months. NHIS and NHANES were never pooled into one survival model.",
        "",
        "## Weighted Aalen-Johansen cumulative incidence",
        "",
        aj.loc[
            :,
            [
                "cohort",
                "horizon",
                "time_unit",
                "cif_cvd",
                "cif_competing",
                "survival",
                "raw_at_risk",
                "weighted_at_risk_proportion",
            ],
        ].to_markdown(index=False),
        "",
        "Competing death is retained as event type 2 in Aalen-Johansen estimation and is "
        "never treated as independent censoring for the CVD cumulative incidence.",
        "",
        "![Weighted competing-risk CIF](figures/stage7_competing_risk_cif.png)",
        "",
        "## Time-dependent model metrics",
        "",
        metrics.to_markdown(index=False),
        "",
        "AUC and Brier use survey weights and reverse-Kaplan-Meier IPCW for administrative "
        "censoring. Model-based CIF predictions are reconstructed jointly from weighted "
        "cause-specific CVD and non-CVD hazards with the frozen sigmoid linear predictor as "
        "the only model covariate. NHIS cause-specific Cox results are sensitivity analyses "
        "because quarter-discrete Aalen-Johansen is the primary NHIS secondary method.",
        "The cause-specific baseline hazards and score coefficients are estimated and "
        "evaluated in the same recent cohort. Consequently Brier, O/E, and calibration-bin "
        "results are apparent secondary-model diagnostics, not independent validation of a "
        "new horizon-specific model; near-unity overall O/E is partly fitted by construction.",
        "",
        "## Support rules",
        "",
        support.to_markdown(index=False),
        "",
        "At longer horizons, a model-performance claim requires at least 100 observed CVD "
        "deaths by the horizon and a weighted at-risk proportion of at least 0.20. Failed "
        "rows remain descriptive with CIF and at-risk counts only.",
        "",
        "## Cumulative-incidence calibration",
        "",
        f"Calibration uses {calibration_bins} survey-weighted equal-weight bins separately "
        "within each cohort and horizon. Boundaries are based on the model-based CIF for that "
        "horizon; observed values are weighted Aalen-Johansen CIF estimates.",
        "",
        "![Cumulative-incidence calibration](figures/stage7_cumulative_incidence_calibration.png)",
        "",
        "## Limitations",
        "",
        "- Follow-up is short and administratively limited at 31 December 2019.",
        "- NHIS has quarter-level interval precision, not exact monthly or daily time.",
        "- Public-use linked mortality files can contain synthetic follow-up time or cause "
        "  values for selected records, although vital status is not perturbed.",
        "- Cause-specific Cox standard errors are model-based and not survey-cluster robust; "
        "  they are reported only as secondary sensitivity diagnostics.",
        "- These U.S. population results do not establish clinical validity in Kazakhstan.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def run_stage7_secondary(
    config_path: str | Path = "configs/stage7_secondary.yaml",
) -> dict[str, Any]:
    """Execute Stage 7 once and persist aggregate outputs only."""

    config_path = Path(config_path)
    root = config_path.resolve().parents[1]
    config = load_stage7_config(config_path)
    bindings = verify_immutable_primary_bindings(config, root)
    sources = config["sources"]
    probability_clip = float(config["analysis"]["probability_clip"])

    nhis_inputs = discover_nhis_input(root / sources["nhis_raw_dir"])
    verify_codebook_binding(
        nhis_inputs.data_path,
        nhis_inputs.codebook_path,
        mapping_path=root / sources["nhis_mapping"],
    )
    nhis_raw = load_nhis(root / sources["nhis_raw_dir"], require_all_years=True)
    recent_mask = pd.to_numeric(nhis_raw["YEAR"], errors="coerce").isin(NHIS_RECENT_YEARS)
    nhis_recent_raw = nhis_raw.loc[recent_mask].copy()
    nhis_recent_raw.attrs = nhis_raw.attrs.copy()
    nhis_harmonized = harmonize_nhis(
        nhis_recent_raw,
        mapping_path=root / sources["nhis_mapping"],
        strict=True,
    )
    nhis, nhis_flow = build_nhis_recent_cohort(nhis_harmonized)
    del nhis_raw, nhis_recent_raw, nhis_harmonized

    nhanes_raw = load_nhanes_cycles(
        list(NHANES_RECENT_CYCLES),
        raw_dir=root / sources["nhanes_raw_dir"],
        config_path=root / sources["nhanes_inventory"],
        profile="fast",
        include_mortality=True,
    )
    nhanes, nhanes_flow = build_nhanes_recent_cohort(nhanes_raw)
    del nhanes_raw

    model = joblib.load(root / config["immutable_primary_model"]["pipeline_path"])
    if not isinstance(model, FittedBaselineModel):
        raise Stage7Error("Frozen pipeline has an unexpected serialized type")
    nhis = _attach_frozen_predictions(nhis, model, probability_clip=probability_clip)
    nhanes = _attach_frozen_predictions(nhanes, model, probability_clip=probability_clip)
    analysis_frames = {"nhis_recent": nhis, "nhanes_recent": nhanes}
    event_counts = _event_count_table(pd.concat([nhis, nhanes], ignore_index=True))
    flow = pd.concat([nhis_flow, nhanes_flow], ignore_index=True)

    aj_tables: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    calibration_tables: list[pd.DataFrame] = []
    hazard_rows: list[dict[str, Any]] = []
    minimum_events = int(config["analysis"]["support_rule"]["minimum_cvd_events"])
    minimum_at_risk = float(
        config["analysis"]["support_rule"]["minimum_weighted_at_risk_proportion"]
    )
    calibration_bins = int(config["analysis"]["calibration_bins"])

    for cohort_name, frame in analysis_frames.items():
        settings = config["cohorts"][cohort_name]
        horizons = [int(value) for value in settings["horizons"]]
        time_unit = str(settings["time_unit"])
        aj = weighted_aalen_johansen(frame, horizons)
        aj.insert(0, "cohort", cohort_name)
        aj.insert(2, "time_unit", time_unit)
        aj_tables.append(aj)

        fit_cvd = fit_weighted_cause_specific_cox(frame, cause=1)
        fit_competing = fit_weighted_cause_specific_cox(frame, cause=2)
        hazard_rows.extend(
            _fit_summary_rows(
                cohort_name,
                time_unit,
                str(settings["cause_specific_cox_role"]),
                fit_cvd,
                horizons,
            )
        )
        hazard_rows.extend(
            _fit_summary_rows(
                cohort_name,
                time_unit,
                str(settings["cause_specific_cox_role"]),
                fit_competing,
                horizons,
            )
        )
        predictions = predict_cif_from_cause_specific_cox(
            frame["linear_predictor"].to_numpy(dtype=float),
            fit_cvd,
            fit_competing,
            horizons,
        )
        primary_horizon = int(settings["primary_recent_horizon"])
        for horizon in horizons:
            aj_row = aj.loc[aj["horizon"].eq(horizon)].iloc[0]
            longer_horizon = horizon != primary_horizon
            passes_events = int(aj_row["raw_cvd_events_by_horizon"]) >= minimum_events
            passes_at_risk = float(aj_row["weighted_at_risk_proportion"]) >= minimum_at_risk
            claim_allowed = longer_horizon and passes_events and passes_at_risk
            support_status = (
                "PRIMARY_RECENT_EXPLORATORY"
                if not longer_horizon
                else ("PASS" if claim_allowed else "FAIL_INSUFFICIENT_SUPPORT")
            )
            support_rows.append(
                {
                    "cohort": cohort_name,
                    "horizon": horizon,
                    "time_unit": time_unit,
                    "raw_cvd_events_by_horizon": int(aj_row["raw_cvd_events_by_horizon"]),
                    "minimum_cvd_events": minimum_events,
                    "passes_event_rule": passes_events,
                    "weighted_at_risk_proportion": float(
                        aj_row["weighted_at_risk_proportion"]
                    ),
                    "minimum_weighted_at_risk_proportion": minimum_at_risk,
                    "passes_at_risk_rule": passes_at_risk,
                    "longer_horizon_support_rule_applies": longer_horizon,
                    "model_performance_claim_allowed": claim_allowed,
                    "status": support_status,
                }
            )
            metrics = ipcw_cumulative_incidence_metrics(
                frame, predictions[horizon], horizon
            )
            metric_rows.append(
                {
                    "cohort": cohort_name,
                    "horizon": horizon,
                    "time_unit": time_unit,
                    "time_dependent_auc": metrics["time_dependent_auc"],
                    "ipcw_brier": metrics["ipcw_brier"],
                    "metric_known_n": metrics["metric_known_n"],
                    "ipcw_weight_sum": metrics["ipcw_weight_sum"],
                    "censoring_survival_before_horizon": metrics[
                        "censoring_survival_before_horizon"
                    ],
                    "mean_model_based_cif": float(
                        np.average(
                            predictions[horizon],
                            weights=frame["analysis_weight"].to_numpy(dtype=float),
                        )
                    ),
                    "observed_aalen_johansen_cif": float(aj_row["cif_cvd"]),
                    "observed_expected_ratio": float(
                        aj_row["cif_cvd"]
                        / np.average(
                            predictions[horizon],
                            weights=frame["analysis_weight"].to_numpy(dtype=float),
                        )
                    ),
                    "support_status": support_status,
                    "interpretation": "secondary_exploratory",
                    "evaluation_role": "apparent_secondary_cause_specific_model",
                }
            )
            calibration = cumulative_incidence_calibration(
                frame,
                predictions[horizon],
                horizon,
                bins=calibration_bins,
            )
            calibration.insert(0, "cohort", cohort_name)
            calibration.insert(2, "time_unit", time_unit)
            calibration["support_status"] = support_status
            calibration["evaluation_role"] = "apparent_secondary_cause_specific_model"
            calibration_tables.append(calibration)

    aj_output = pd.concat(aj_tables, ignore_index=True)
    metrics_output = pd.DataFrame(metric_rows)
    support_output = pd.DataFrame(support_rows)
    calibration_output = pd.concat(calibration_tables, ignore_index=True)
    hazards_output = pd.DataFrame(hazard_rows)

    outputs = {name: root / value for name, value in config["outputs"].items()}
    _write_csv(flow, outputs["cohort_flow"])
    _write_csv(event_counts, outputs["event_counts"])
    _write_csv(aj_output, outputs["aalen_johansen"])
    _write_csv(hazards_output, outputs["cause_specific_hazards"])
    _write_csv(metrics_output, outputs["time_dependent_metrics"])
    _write_csv(calibration_output, outputs["calibration"])
    _write_csv(support_output, outputs["support"])
    _write_cif_figure(outputs["cif_figure"], aj_output)
    _write_calibration_figure(outputs["calibration_figure"], calibration_output)
    _write_report(
        outputs["report"],
        event_counts=event_counts,
        aj=aj_output,
        metrics=metrics_output,
        support=support_output,
        calibration_bins=calibration_bins,
        bindings=bindings,
    )

    output_bindings = {}
    for name, path in outputs.items():
        if name == "manifest":
            continue
        output_bindings[name] = {
            "repository_relative_path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "hash_mode": "raw_bytes",
        }
    manifest = {
        "schema_version": 1,
        "stage": "stage7_secondary_censored_competing_risk",
        "status": "COMPLETE_SECONDARY_EXPLORATORY",
        "completed_at": datetime.now(UTC).isoformat(),
        "interpretation": "secondary_exploratory",
        "primary_model_modified": False,
        "preprocessing_modified": False,
        "calibrator_modified": False,
        "cutoff_modified_or_reselected": False,
        "participant_level_output_persisted": False,
        "raw_bootstrap_replicates_persisted": False,
        "immutable_bindings": bindings,
        "config": {
            "repository_relative_path": config_path.resolve().relative_to(root).as_posix(),
            "sha256": sha256_file(config_path),
        },
        "recent_nhanes_inventory": {
            "repository_relative_path": sources["nhanes_inventory"],
            "sha256": sha256_file(root / sources["nhanes_inventory"]),
        },
        "raw_data_manifest": {
            "repository_relative_path": sources["raw_manifest"],
            "sha256": sha256_file(root / sources["raw_manifest"]),
        },
        "nhis_sources": {
            "extract": {
                "repository_relative_path": (
                    nhis_inputs.data_path.resolve().relative_to(root).as_posix()
                ),
                "sha256": sha256_file(nhis_inputs.data_path),
            },
            "codebook": {
                "repository_relative_path": (
                    nhis_inputs.codebook_path.resolve().relative_to(root).as_posix()
                ),
                "sha256": sha256_file(nhis_inputs.codebook_path),
            },
        },
        "implementation": {
            "module": {
                "repository_relative_path": "src/cvd_risk/modeling/stage7.py",
                "sha256": sha256_file(root / "src/cvd_risk/modeling/stage7.py"),
            },
            "runner": {
                "repository_relative_path": "scripts/19_stage7_secondary_competing_risk.py",
                "sha256": sha256_file(root / "scripts/19_stage7_secondary_competing_risk.py"),
            },
        },
        "cohort_counts": {
            str(record["cohort"]): {
                "n": int(record["n"]),
                "cvd_deaths": int(record["cvd_deaths"]),
                "competing_deaths": int(record["competing_deaths"]),
                "censored": int(record["censored"]),
            }
            for record in event_counts.to_dict(orient="records")
        },
        "outputs": output_bindings,
    }
    outputs["manifest"].parent.mkdir(parents=True, exist_ok=True)
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def validate_stage7_outputs(
    config_path: str | Path = "configs/stage7_secondary.yaml",
) -> dict[str, Any]:
    """Validate Stage 7 aggregate outputs without loading participant-level data."""

    config_path = Path(config_path)
    root = config_path.resolve().parents[1]
    config = load_stage7_config(config_path)
    immutable = verify_immutable_primary_bindings(config, root)
    manifest_path = root / config["outputs"]["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE_SECONDARY_EXPLORATORY":
        raise Stage7Error("Stage 7 manifest is not complete and exploratory")
    forbidden_true = (
        "primary_model_modified",
        "preprocessing_modified",
        "calibrator_modified",
        "cutoff_modified_or_reselected",
        "participant_level_output_persisted",
        "raw_bootstrap_replicates_persisted",
    )
    if any(manifest.get(field) is not False for field in forbidden_true):
        raise Stage7Error("Stage 7 manifest records a forbidden modification or persisted data")
    if manifest.get("immutable_bindings") != immutable:
        raise Stage7Error("Stage 7 immutable bindings differ from the current verified files")
    if sha256_file(config_path) != manifest.get("config", {}).get("sha256"):
        raise Stage7Error("Stage 7 config SHA changed")
    if sha256_file(root / config["sources"]["nhanes_inventory"]) != manifest.get(
        "recent_nhanes_inventory", {}
    ).get("sha256"):
        raise Stage7Error("Stage 7 recent NHANES inventory SHA changed")
    if sha256_file(root / config["sources"]["raw_manifest"]) != manifest.get(
        "raw_data_manifest", {}
    ).get("sha256"):
        raise Stage7Error("Stage 7 raw-data manifest SHA changed")
    for group in ("nhis_sources", "implementation"):
        bindings = manifest.get(group)
        if not isinstance(bindings, dict) or not bindings:
            raise Stage7Error(f"Stage 7 manifest is missing {group} bindings")
        for name, binding in bindings.items():
            if not isinstance(binding, dict):
                raise Stage7Error(f"Stage 7 {group}/{name} binding is invalid")
            path = root / str(binding.get("repository_relative_path"))
            if sha256_file(path) != binding.get("sha256"):
                raise Stage7Error(f"Stage 7 {group}/{name} SHA changed")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(config["outputs"]).difference(
        {"manifest"}
    ):
        raise Stage7Error("Stage 7 output inventory is incomplete")
    for name, binding in outputs.items():
        if not isinstance(binding, dict) or binding.get("hash_mode") != "raw_bytes":
            raise Stage7Error(f"Stage 7 output binding is invalid: {name}")
        relative = Path(str(binding.get("repository_relative_path")))
        if relative.suffix.lower() in {".parquet", ".feather", ".pkl", ".joblib"}:
            raise Stage7Error(
                f"Participant/model artifact type is forbidden in Stage 7: {relative}"
            )
        path = root / relative
        if sha256_file(path) != binding.get("sha256"):
            raise Stage7Error(f"Stage 7 output SHA changed: {name}")

    support = pd.read_csv(root / config["outputs"]["support"])
    required_support = {
        "cohort",
        "horizon",
        "raw_cvd_events_by_horizon",
        "weighted_at_risk_proportion",
        "model_performance_claim_allowed",
        "status",
    }
    _require_columns(support, required_support, "Stage 7 support output")
    primary = support["status"].eq("PRIMARY_RECENT_EXPLORATORY")
    if support.loc[primary, "model_performance_claim_allowed"].astype(bool).any():
        raise Stage7Error("Primary recent horizons cannot be labeled for performance claims")
    longer = support["longer_horizon_support_rule_applies"].astype(bool)
    expected_pass = (
        support["raw_cvd_events_by_horizon"].ge(100)
        & support["weighted_at_risk_proportion"].ge(0.20)
    )
    observed = support["model_performance_claim_allowed"].astype(bool)
    if not observed.eq(longer & expected_pass).all():
        raise Stage7Error("Stage 7 longer-horizon support rule is inconsistent")
    return {
        "status": manifest["status"],
        "cohorts": sorted(manifest["cohort_counts"]),
        "output_count": len(outputs),
        "pipeline_sha256": immutable["pipeline"],
    }
