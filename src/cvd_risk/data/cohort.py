"""Protocol-locked outcome definitions, cohort filters, and leakage guards."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd

from cvd_risk.data.harmonize import (
    BASELINE_CVD_COLUMNS,
    require_verified_codebook,
)
from cvd_risk.schemas import FAST_FEATURES

CVD_UCOD_LEADING = frozenset({"001", "005"})
ALLOWED_UCOD_LEADING = frozenset(f"{value:03d}" for value in range(1, 11))
NHIS_DEVELOPMENT_YEARS = frozenset(range(1997, 2010))
NHIS_TEMPORAL_TEST_YEARS = frozenset(range(2010, 2015))

_AUDIT_ONLY_COLUMNS = frozenset({"race_ethnicity", "income", "education"})
_KNOWN_LEAKAGE_COLUMNS = frozenset(
    {
        "MORTSTAT",
        "MORTUCODLD",
        "MORTDODY",
        "MORTDODQ",
        "UCOD_LEADING",
        "PERMTH_INT",
        "PERMTH_EXM",
        "cvd_death_5y",
        "deceased",
        "ucod_leading",
        "death_year",
        "death_quarter",
        "followup_quarters",
        "followup_months",
        "mortality_eligible",
        "mortality_weight",
        "analysis_weight",
        "survey_year",
        "interview_quarter",
        "split",
        "participant_id",
        "strata",
        "psu",
    }
)


class OutcomeDefinitionError(ValueError):
    """Raised when mortality fields violate the pre-specified outcome rules."""


class CohortDefinitionError(ValueError):
    """Raised when a cohort cannot be constructed without changing protocol."""


class LeakageError(ValueError):
    """Raised when a proposed predictor set contains labels or audit-only data."""


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise KeyError(f"{context} is missing required columns: {missing}")


def _nullable_boolean(
    series: pd.Series,
    *,
    true_values: Sequence[Any],
    false_values: Sequence[Any],
    name: str,
) -> pd.Series:
    true_text = {str(value).strip().casefold() for value in true_values}
    false_text = {str(value).strip().casefold() for value in false_values}
    mapped_values: list[Any] = []
    unexpected: list[Any] = []
    for value in series:
        if pd.isna(value):
            mapped_values.append(pd.NA)
            continue
        normalized = str(value).strip().casefold()
        if normalized in true_text:
            mapped_values.append(True)
        elif normalized in false_text:
            mapped_values.append(False)
        else:
            unexpected.append(value)
            mapped_values.append(pd.NA)
    if unexpected:
        examples = list(dict.fromkeys(map(str, unexpected)))[:5]
        raise OutcomeDefinitionError(f"{name} has unknown status values: {examples}")
    return pd.Series(mapped_values, index=series.index, dtype="boolean")


def _ucod(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized_label = text.casefold().replace("-", "_").replace(" ", "_")
    if normalized_label in {"diseases_of_heart", "heart_disease"}:
        return "001"
    if normalized_label in {"cerebrovascular_diseases", "cerebrovascular_disease"}:
        return "005"
    try:
        number = float(text)
    except ValueError as exc:
        raise OutcomeDefinitionError(f"Unknown UCOD leading-cause label {value!r}") from exc
    if not number.is_integer():
        raise OutcomeDefinitionError(f"UCOD leading cause must be integral, got {value!r}")
    normalized = str(int(number)).zfill(3)
    if normalized not in ALLOWED_UCOD_LEADING:
        raise OutcomeDefinitionError(
            f"UCOD leading cause {value!r} is outside allowed public-LMF categories 001-010"
        )
    return normalized


def _strict_nullable_integer(series: pd.Series, *, name: str) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    malformed = series.notna() & converted.isna()
    if malformed.any():
        examples = series.loc[malformed].astype(str).head(5).tolist()
        raise OutcomeDefinitionError(f"{name} has non-numeric values: {examples}")
    fractional = converted.notna() & converted.mod(1).ne(0)
    if fractional.any():
        examples = converted.loc[fractional].head(5).tolist()
        raise OutcomeDefinitionError(f"{name} must contain whole months/codes: {examples}")
    return converted.astype("Int64")


def derive_nhis_cvd_outcome(
    frame: pd.DataFrame,
    *,
    include_ambiguous_boundary: bool = False,
    strict_negative_followup: bool = True,
) -> pd.DataFrame:
    """Apply the quarterly NHIS five-year outcome definition.

    Primary analysis includes CVD deaths at 0-19 quarters. CVD deaths at exactly
    20 quarters are marked missing/excluded; the sensitivity definition includes
    them when ``include_ambiguous_boundary=True``.
    """

    required = (
        "survey_year",
        "interview_quarter",
        "mortality_eligible",
        "deceased",
        "ucod_leading",
        "death_year",
        "death_quarter",
    )
    _require_columns(frame, required, "NHIS outcome input")
    result = frame.copy()
    eligible = _nullable_boolean(
        result["mortality_eligible"],
        true_values=(True, 1, "eligible", "yes"),
        false_values=(False, 0, 2, 3, "ineligible", "no"),
        name="mortality_eligible",
    )
    deceased = _nullable_boolean(
        result["deceased"],
        true_values=(True, 1, "deceased", "dead"),
        false_values=(False, 0, "alive"),
        name="deceased",
    )

    followup = pd.Series(pd.NA, index=result.index, dtype="Int64")
    complete_timing = (
        deceased.fillna(False) & result["death_year"].notna() & result["death_quarter"].notna()
    )
    if complete_timing.any():
        death_year = pd.to_numeric(result.loc[complete_timing, "death_year"], errors="raise")
        death_quarter = pd.to_numeric(result.loc[complete_timing, "death_quarter"], errors="raise")
        survey_year = pd.to_numeric(result.loc[complete_timing, "survey_year"], errors="raise")
        interview_quarter = pd.to_numeric(
            result.loc[complete_timing, "interview_quarter"], errors="raise"
        )
        calculated = 4 * (death_year - survey_year) + (death_quarter - interview_quarter)
        if (calculated % 1 != 0).any():
            raise OutcomeDefinitionError("NHIS follow-up quarter calculation is non-integral")
        followup.loc[complete_timing] = calculated.astype("int64")

    negative = followup.notna() & (followup < 0)
    if negative.any() and strict_negative_followup:
        identifiers = (
            result.loc[negative, "participant_id"].astype(str).head(5).tolist()
            if "participant_id" in result
            else result.index[negative].astype(str).tolist()[:5]
        )
        raise OutcomeDefinitionError(
            f"Negative NHIS follow-up for {int(negative.sum())} records; examples: {identifiers}"
        )

    causes = result["ucod_leading"].map(_ucod).astype("string")
    cvd_cause = causes.isin(CVD_UCOD_LEADING)
    target = pd.Series(pd.NA, index=result.index, dtype="Int8")
    reason = pd.Series(pd.NA, index=result.index, dtype="string")

    reason.loc[eligible.isna()] = "mortality_eligibility_unknown"
    reason.loc[eligible.eq(False).fillna(False)] = "mortality_not_eligible"
    eligible_known = eligible.eq(True).fillna(False)
    reason.loc[eligible_known & deceased.isna()] = "mortality_status_unknown"

    alive = eligible_known & deceased.eq(False).fillna(False)
    target.loc[alive] = 0
    reason.loc[alive] = pd.NA

    dead = eligible_known & deceased.eq(True).fillna(False)
    cause_unknown = dead & causes.isna()
    reason.loc[cause_unknown] = "cause_unknown"

    non_cvd_death = dead & causes.notna() & ~cvd_cause
    target.loc[non_cvd_death] = 0
    reason.loc[non_cvd_death] = pd.NA

    cvd_death = dead & cvd_cause
    timing_unknown = cvd_death & followup.isna()
    reason.loc[timing_unknown] = "death_timing_unknown"
    reason.loc[cvd_death & negative] = "negative_followup"

    valid_cvd_timing = cvd_death & followup.notna() & ~negative
    within_primary = valid_cvd_timing & followup.between(0, 19, inclusive="both")
    boundary = valid_cvd_timing & followup.eq(20)
    after_horizon = valid_cvd_timing & followup.gt(20)
    target.loc[within_primary] = 1
    target.loc[after_horizon] = 0
    reason.loc[within_primary | after_horizon] = pd.NA
    if include_ambiguous_boundary:
        target.loc[boundary] = 1
        reason.loc[boundary] = pd.NA
    else:
        reason.loc[boundary] = "ambiguous_20_quarter_boundary"

    result["followup_quarters"] = followup
    result["cvd_death_5y_boundary_20q"] = boundary.astype("boolean")
    result["cvd_death_5y"] = target
    result["outcome_missing_reason"] = reason
    return result


def derive_nhanes_cvd_outcome(
    frame: pd.DataFrame,
    *,
    eligibility_col: str = "ELIGSTAT",
    status_col: str = "MORTSTAT",
    cause_col: str = "UCOD_LEADING",
    months_col: str = "PERMTH_INT",
    strict_negative_followup: bool = True,
) -> pd.DataFrame:
    """Apply the exact-month NHANES definition, including month 60."""

    _require_columns(
        frame,
        (eligibility_col, status_col, cause_col, months_col),
        "NHANES outcome input",
    )
    result = frame.copy()
    eligible = _nullable_boolean(
        result[eligibility_col],
        true_values=(True, 1, "eligible"),
        false_values=(False, 0, 2, 3, "ineligible"),
        name=eligibility_col,
    )
    deceased = _nullable_boolean(
        result[status_col],
        true_values=(True, 1, "deceased", "dead"),
        false_values=(False, 0, "alive"),
        name=status_col,
    )
    months = _strict_nullable_integer(result[months_col], name=months_col)
    eligible_with_time = eligible.eq(True).fillna(False) & months.notna()
    negative = eligible_with_time & months.lt(0)
    if negative.any() and strict_negative_followup:
        raise OutcomeDefinitionError(f"Negative NHANES follow-up for {int(negative.sum())} records")

    causes = result[cause_col].map(_ucod).astype("string")
    cvd_cause = causes.isin(CVD_UCOD_LEADING)
    target = pd.Series(pd.NA, index=result.index, dtype="Int8")
    reason = pd.Series(pd.NA, index=result.index, dtype="string")
    reason.loc[eligible.isna()] = "mortality_eligibility_unknown"
    reason.loc[eligible.eq(False).fillna(False)] = "mortality_not_eligible"

    eligible_known = eligible.eq(True).fillna(False)
    reason.loc[eligible_known & deceased.isna()] = "mortality_status_unknown"
    alive = eligible_known & deceased.eq(False).fillna(False)
    alive_full_followup = alive & months.ge(60).fillna(False)
    alive_incomplete = alive & ~alive_full_followup
    target.loc[alive_full_followup] = 0
    reason.loc[alive_full_followup] = pd.NA
    reason.loc[alive_incomplete] = "insufficient_potential_followup"

    dead = eligible_known & deceased.eq(True).fillna(False)
    reason.loc[dead & causes.isna()] = "cause_unknown"
    non_cvd = dead & causes.notna() & ~cvd_cause
    target.loc[non_cvd] = 0
    reason.loc[non_cvd] = pd.NA

    cvd = dead & cvd_cause
    reason.loc[cvd & months.isna()] = "followup_months_unknown"
    reason.loc[cvd & negative] = "negative_followup"
    valid = cvd & months.notna() & ~negative
    target.loc[valid & months.le(60)] = 1
    target.loc[valid & months.gt(60)] = 0
    reason.loc[valid] = pd.NA

    result["followup_months"] = months
    result["cvd_death_5y"] = target
    result["outcome_missing_reason"] = reason
    return result


def derive_baseline_cvd_status(
    frame: pd.DataFrame,
    *,
    condition_columns: Sequence[str] = BASELINE_CVD_COLUMNS,
) -> pd.Series:
    """Return present/absent/unknown without ever treating unknown as No."""

    _require_columns(frame, condition_columns, "Baseline CVD input")
    conditions = pd.DataFrame(index=frame.index)
    for column in condition_columns:
        conditions[column] = _nullable_boolean(
            frame[column],
            true_values=(True, 1, "yes", "present"),
            false_values=(False, 0, "no", "absent"),
            name=column,
        )
    present = conditions.eq(True).any(axis=1)
    absent = conditions.notna().all(axis=1) & conditions.eq(False).all(axis=1)
    status = pd.Series("unknown", index=frame.index, dtype="string")
    status.loc[absent] = "absent"
    status.loc[present] = "present"
    return status


def assign_nhis_split(years: pd.Series, *, strict: bool = True) -> pd.Series:
    """Assign the fixed temporal protocol; never perform a random split."""

    numeric = pd.to_numeric(years, errors="coerce")
    split = pd.Series(pd.NA, index=years.index, dtype="string")
    split.loc[numeric.isin(NHIS_DEVELOPMENT_YEARS)] = "development"
    split.loc[numeric.isin(NHIS_TEMPORAL_TEST_YEARS)] = "temporal_test"
    invalid = years.notna() & split.isna()
    if strict and invalid.any():
        values = sorted(set(years.loc[invalid].astype(str)))
        raise CohortDefinitionError(f"Survey years outside the locked NHIS splits: {values}")
    return split


def annotate_nhis_cohort(
    frame: pd.DataFrame,
    *,
    include_ambiguous_boundary: bool = False,
) -> pd.DataFrame:
    """Annotate every source row with protocol eligibility and exclusion reason."""

    required = {
        "participant_id",
        "survey_year",
        "age",
        "mortality_eligible",
        "mortality_weight",
        *BASELINE_CVD_COLUMNS,
    }
    _require_columns(frame, required, "NHIS cohort input")
    annotated = derive_nhis_cvd_outcome(
        frame,
        include_ambiguous_boundary=include_ambiguous_boundary,
    )
    annotated["baseline_cvd_status"] = derive_baseline_cvd_status(annotated)
    annotated["split"] = assign_nhis_split(annotated["survey_year"], strict=False)

    age = pd.to_numeric(annotated["age"], errors="coerce")
    linkage = _nullable_boolean(
        annotated["mortality_eligible"],
        true_values=(True, 1, "eligible", "yes"),
        false_values=(False, 0, 2, 3, "ineligible", "no"),
        name="mortality_eligible",
    )
    flags = {
        "valid_split": annotated["split"].notna(),
        "age_known": age.notna(),
        "age_40_79": age.between(40, 79, inclusive="both").fillna(False),
        "mortality_linkage_eligible": linkage.eq(True).fillna(False),
        "no_prevalent_cvd": annotated["baseline_cvd_status"].ne("present"),
        "baseline_cvd_known_absent": annotated["baseline_cvd_status"].eq("absent"),
        "known_five_year_outcome": annotated["cvd_death_5y"].notna(),
    }
    for name, values in flags.items():
        annotated[f"cohort_rule_{name}"] = values.astype(bool)

    eligible = pd.Series(True, index=annotated.index)
    for values in flags.values():
        eligible &= values.astype(bool)
    annotated["cohort_eligible"] = eligible

    reason = pd.Series(pd.NA, index=annotated.index, dtype="string")
    ordered_reasons = (
        (~flags["valid_split"], "survey_year_outside_protocol"),
        (~flags["age_known"], "age_unknown"),
        (flags["age_known"] & ~flags["age_40_79"], "age_outside_40_79"),
        (~flags["mortality_linkage_eligible"], "mortality_not_eligible_or_unknown"),
        (annotated["baseline_cvd_status"].eq("present"), "prevalent_cvd"),
        (annotated["baseline_cvd_status"].eq("unknown"), "baseline_cvd_unknown"),
        (~flags["known_five_year_outcome"], "outcome_unknown_or_ambiguous"),
    )
    for mask, label in ordered_reasons:
        use = reason.isna() & mask.fillna(False)
        reason.loc[use] = label
    annotated["cohort_exclusion_reason"] = reason

    weights = pd.to_numeric(annotated["mortality_weight"], errors="coerce")
    invalid_weight = eligible & (weights.isna() | weights.lt(0))
    if invalid_weight.any():
        raise CohortDefinitionError("Eligible NHIS participants have invalid MORTWTSA weights")
    divisor = annotated["split"].map({"development": 13.0, "temporal_test": 5.0})
    annotated["analysis_weight"] = (weights / divisor).astype("Float64")
    return annotated


def nhis_cohort_flow(annotated: pd.DataFrame) -> pd.DataFrame:
    """Build the monotone stage-3 flow table from an annotated NHIS frame."""

    rules = (
        ("survey year in locked split", "valid_split"),
        ("known age", "age_known"),
        ("age 40-79", "age_40_79"),
        ("mortality linkage eligible", "mortality_linkage_eligible"),
        ("exclude prevalent CVD", "no_prevalent_cvd"),
        ("baseline CVD status known absent", "baseline_cvd_known_absent"),
        ("known unambiguous five-year outcome", "known_five_year_outcome"),
    )
    _require_columns(
        annotated,
        (f"cohort_rule_{key}" for _, key in rules),
        "Annotated NHIS cohort",
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


def build_nhis_cohort(
    frame: pd.DataFrame,
    *,
    include_ambiguous_boundary: bool = False,
    allow_unverified_codebook: bool = False,
) -> pd.DataFrame:
    """Return only records satisfying the locked NHIS cohort definition."""

    if not allow_unverified_codebook:
        require_verified_codebook(frame)
    annotated = annotate_nhis_cohort(
        frame,
        include_ambiguous_boundary=include_ambiguous_boundary,
    )
    cohort = annotated.loc[annotated["cohort_eligible"]].copy()
    if cohort["participant_id"].duplicated().any():
        raise CohortDefinitionError("NHIS cohort contains duplicate participant identifiers")
    return cohort.reset_index(drop=True)


def build_nhis_cohort_with_flow(
    frame: pd.DataFrame,
    *,
    include_ambiguous_boundary: bool = False,
    allow_unverified_codebook: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the eligible cohort and its exact sequential flow table."""

    if not allow_unverified_codebook:
        require_verified_codebook(frame)
    annotated = annotate_nhis_cohort(
        frame,
        include_ambiguous_boundary=include_ambiguous_boundary,
    )
    flow = nhis_cohort_flow(annotated)
    cohort = annotated.loc[annotated["cohort_eligible"]].copy().reset_index(drop=True)
    if cohort["participant_id"].duplicated().any():
        raise CohortDefinitionError("NHIS cohort contains duplicate participant identifiers")
    return cohort, flow


def validate_fast_predictors(
    columns: Sequence[str],
    *,
    require_exact: bool = True,
) -> tuple[str, ...]:
    """Reject outcome, design, weight, and audit-only columns as predictors."""

    proposed = tuple(columns)
    duplicates = sorted({column for column in proposed if proposed.count(column) > 1})
    if duplicates:
        raise LeakageError(f"Duplicate predictor columns: {duplicates}")
    audit_only = sorted(set(proposed).intersection(_AUDIT_ONLY_COLUMNS))
    if audit_only:
        raise LeakageError(f"Audit-only columns cannot be model predictors: {audit_only}")
    known_leakage = sorted(set(proposed).intersection(_KNOWN_LEAKAGE_COLUMNS))
    if known_leakage:
        raise LeakageError(f"Outcome/design leakage in predictors: {known_leakage}")
    unexpected = sorted(set(proposed).difference(FAST_FEATURES))
    if unexpected:
        raise LeakageError(f"Non-protocol fast-model predictors: {unexpected}")
    if require_exact:
        missing = sorted(set(FAST_FEATURES).difference(proposed))
        if missing:
            raise LeakageError(f"Fast-model predictor set is incomplete: {missing}")
    return proposed


def select_fast_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Select only the seven locked predictors in their canonical order."""

    _require_columns(frame, FAST_FEATURES, "Fast-model input")
    validate_fast_predictors(FAST_FEATURES)
    return frame.loc[:, FAST_FEATURES].copy()
