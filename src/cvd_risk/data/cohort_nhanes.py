"""Protocol-locked construction of the NHANES external-validation cohort."""

from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd

from cvd_risk.data.cohort import (
    OutcomeDefinitionError,
    derive_baseline_cvd_status,
    derive_nhanes_cvd_outcome,
)
from cvd_risk.data.harmonize import (
    SENSITIVITY_BASELINE_CVD_COLUMNS,
)
from cvd_risk.data.harmonize_nhanes import NHANES_CYCLE_START_YEAR


class NHANESCohortError(ValueError):
    """Raised when the locked NHANES cohort contract is violated."""


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise NHANESCohortError(f"{context} is missing required columns: {missing}")


def derive_nhanes_cvd_outcome_with_complete_followup(frame: pd.DataFrame) -> pd.DataFrame:
    """Reuse the common outcome and correct non-events with less than 60 months.

    A linkage-eligible participant assumed alive is a known five-year non-event
    only when ``PERMTH_INT >= 60``. A non-CVD death before five years remains a
    valid competing-event zero, while CVD deaths retain the exact ``<= 60`` rule.
    """

    result = derive_nhanes_cvd_outcome(frame)
    eligibility = pd.to_numeric(result["ELIGSTAT"], errors="coerce")
    status = pd.to_numeric(result["MORTSTAT"], errors="coerce")
    months = pd.to_numeric(result["PERMTH_INT"], errors="coerce")
    eligible = eligibility.eq(1)
    negative = eligible & months.notna() & months.lt(0)
    if negative.any():
        raise OutcomeDefinitionError(f"Negative NHANES follow-up for {int(negative.sum())} records")

    alive = eligible & status.eq(0)
    incomplete_alive = alive & (months.isna() | months.lt(60))
    result.loc[incomplete_alive, "cvd_death_5y"] = pd.NA
    result.loc[incomplete_alive, "outcome_missing_reason"] = "insufficient_five_year_followup"
    result["full_five_year_outcome_observed"] = result["cvd_death_5y"].notna().astype(bool)
    return result


def annotate_nhanes_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    """Annotate every NHANES row with sequential protocol eligibility flags."""

    required = {
        "participant_id",
        "survey_cycle",
        "survey_year",
        "age",
        "analysis_weight",
        "ELIGSTAT",
        "MORTSTAT",
        "UCOD_LEADING",
        "PERMTH_INT",
        *SENSITIVITY_BASELINE_CVD_COLUMNS,
    }
    _require_columns(frame, required, "NHANES cohort input")
    if frame["participant_id"].isna().any():
        raise NHANESCohortError("NHANES participant_id cannot be missing")
    if frame["participant_id"].duplicated().any():
        raise NHANESCohortError("NHANES input contains duplicate participant identifiers")

    annotated = derive_nhanes_cvd_outcome_with_complete_followup(frame)
    annotated["baseline_cvd_status"] = derive_baseline_cvd_status(annotated)
    annotated["baseline_cvd_status_with_chf"] = derive_baseline_cvd_status(
        annotated,
        condition_columns=SENSITIVITY_BASELINE_CVD_COLUMNS,
    )
    cycle = annotated["survey_cycle"].astype("string")
    valid_cycle = cycle.isin(NHANES_CYCLE_START_YEAR)
    expected_year = cycle.map(NHANES_CYCLE_START_YEAR)
    survey_year = pd.to_numeric(annotated["survey_year"], errors="coerce")
    valid_period = valid_cycle & survey_year.eq(expected_year)

    age = pd.to_numeric(annotated["age"], errors="coerce")
    weight = pd.to_numeric(annotated["analysis_weight"], errors="coerce")
    finite_weight = weight.map(lambda value: False if pd.isna(value) else math.isfinite(value))
    positive_weight = finite_weight & weight.gt(0).fillna(False)
    eligibility = pd.to_numeric(annotated["ELIGSTAT"], errors="coerce")

    flags = {
        "valid_cycle_period": valid_period.fillna(False),
        "age_known": age.notna(),
        "age_40_79": age.between(40, 79, inclusive="both").fillna(False),
        "positive_mec_weight": positive_weight.fillna(False),
        "mortality_linkage_eligible": eligibility.eq(1).fillna(False),
        "no_prevalent_cvd": annotated["baseline_cvd_status"].ne("present"),
        "baseline_cvd_known_absent": annotated["baseline_cvd_status"].eq("absent"),
        "known_five_year_outcome": annotated["cvd_death_5y"].notna(),
    }
    for name, values in flags.items():
        annotated[f"cohort_rule_{name}"] = values.astype(bool)

    eligible = pd.Series(True, index=annotated.index)
    for values in flags.values():
        eligible &= values.astype(bool)
    annotated["cohort_eligible"] = eligible.astype(bool)
    annotated["chf_sensitivity_available"] = True
    annotated["cohort_chf_sensitivity_eligible"] = (
        eligible & annotated["baseline_cvd_status_with_chf"].eq("absent")
    ).astype(bool)

    exclusion_reason = pd.Series(pd.NA, index=annotated.index, dtype="string")
    ordered_reasons = (
        (~flags["valid_cycle_period"], "cycle_or_year_outside_protocol"),
        (~flags["age_known"], "age_unknown"),
        (flags["age_known"] & ~flags["age_40_79"], "age_outside_40_79"),
        (~flags["positive_mec_weight"], "nonpositive_or_missing_mec_weight"),
        (~flags["mortality_linkage_eligible"], "mortality_not_eligible_or_unknown"),
        (annotated["baseline_cvd_status"].eq("present"), "prevalent_cvd"),
        (annotated["baseline_cvd_status"].eq("unknown"), "baseline_cvd_unknown"),
        (~flags["known_five_year_outcome"], "outcome_unknown_or_incomplete_followup"),
    )
    for mask, label in ordered_reasons:
        use = exclusion_reason.isna() & mask.fillna(False)
        exclusion_reason.loc[use] = label
    annotated["cohort_exclusion_reason"] = exclusion_reason

    invalid_final_weight = eligible & (~finite_weight | weight.le(0).fillna(True))
    if invalid_final_weight.any():
        raise NHANESCohortError("Eligible NHANES participants have invalid MEC weights")
    return annotated


def nhanes_cohort_flow(annotated: pd.DataFrame) -> pd.DataFrame:
    """Build a monotone participant flow for the external-validation cohort."""

    rules = (
        ("cycle and start year in 1999-2014 protocol", "valid_cycle_period"),
        ("known age", "age_known"),
        ("age 40-79", "age_40_79"),
        ("positive MEC examination weight", "positive_mec_weight"),
        ("mortality linkage eligible", "mortality_linkage_eligible"),
        ("exclude prevalent CVD", "no_prevalent_cvd"),
        ("baseline CVD status known absent", "baseline_cvd_known_absent"),
        ("known five-year outcome with complete non-event follow-up", "known_five_year_outcome"),
    )
    _require_columns(
        annotated,
        (f"cohort_rule_{key}" for _, key in rules),
        "Annotated NHANES cohort",
    )
    rows = [
        {
            "step_order": 0,
            "step": "raw harmonized records",
            "n_participants": len(annotated),
            "removed_from_previous": 0,
        }
    ]
    retained = pd.Series(True, index=annotated.index)
    previous = len(annotated)
    for order, (label, key) in enumerate(rules, start=1):
        retained &= annotated[f"cohort_rule_{key}"].astype(bool)
        count = int(retained.sum())
        rows.append(
            {
                "step_order": order,
                "step": label,
                "n_participants": count,
                "removed_from_previous": previous - count,
            }
        )
        previous = count
    return pd.DataFrame(rows)


def build_nhanes_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    """Return records satisfying the locked NHANES external cohort definition."""

    annotated = annotate_nhanes_cohort(frame)
    cohort = annotated.loc[annotated["cohort_eligible"]].copy().reset_index(drop=True)
    if cohort["participant_id"].duplicated().any():
        raise NHANESCohortError("NHANES cohort contains duplicate participant identifiers")
    cohort["split"] = "external_validation"
    return cohort


def build_nhanes_cohort_with_flow(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the eligible NHANES cohort and exact sequential flow table."""

    annotated = annotate_nhanes_cohort(frame)
    flow = nhanes_cohort_flow(annotated)
    cohort = annotated.loc[annotated["cohort_eligible"]].copy().reset_index(drop=True)
    if cohort["participant_id"].duplicated().any():
        raise NHANESCohortError("NHANES cohort contains duplicate participant identifiers")
    cohort["split"] = "external_validation"
    return cohort, flow
