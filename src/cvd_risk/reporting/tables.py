"""Pure helpers for the mandatory pre-training cohort audit.

The external and temporal labels may be read here only to produce the
pre-specified feasibility counts.  These helpers deliberately contain no
model-selection or preprocessing logic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from cvd_risk.schemas import FAST_FEATURES


def _validated_binary_target(frame: pd.DataFrame, target_col: str) -> pd.Series:
    if target_col not in frame:
        raise KeyError(f"Missing target column: {target_col}")
    target = frame[target_col]
    observed = set(target.dropna().unique().tolist())
    if not observed.issubset({0, 1}):
        raise ValueError(
            f"{target_col} must be binary or missing; observed values: {sorted(map(str, observed))}"
        )
    return target


def cohort_flow_table(
    steps: Sequence[tuple[str, pd.DataFrame]],
    *,
    target_col: str = "cvd_death_5y",
) -> pd.DataFrame:
    """Summarize an ordered series of cohort filters.

    ``steps`` must contain snapshots after each filter. Participant counts are
    required to be non-increasing; violating this normally signals a bad join.
    """

    if not steps:
        raise ValueError("At least one cohort-flow step is required")

    rows: list[dict[str, Any]] = []
    previous_n: int | None = None
    for order, (step, frame) in enumerate(steps, start=1):
        if not step:
            raise ValueError("Cohort-flow step names cannot be empty")
        n = int(len(frame))
        if previous_n is not None and n > previous_n:
            raise ValueError(
                f"Cohort grew from {previous_n} to {n} at step {step!r}; "
                "check merge cardinality and duplicated participants"
            )
        if target_col in frame:
            target = _validated_binary_target(frame, target_col)
            events: Any = int(target.fillna(0).sum())
            unknown_outcomes = int(target.isna().sum())
        else:
            events = pd.NA
            unknown_outcomes = n
        rows.append(
            {
                "step_order": order,
                "step": step,
                "n_participants": n,
                "removed_from_previous": 0 if previous_n is None else previous_n - n,
                "n_events": events,
                "n_unknown_outcomes": unknown_outcomes,
            }
        )
        previous_n = n
    return pd.DataFrame(rows)


def event_counts_table(
    cohorts: Mapping[str, pd.DataFrame],
    *,
    target_col: str = "cvd_death_5y",
    weight_columns: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Return event counts and weighted/unweighted prevalence by named split."""

    if not cohorts:
        raise ValueError("At least one cohort is required")
    rows: list[dict[str, Any]] = []
    for cohort_name, frame in cohorts.items():
        target = _validated_binary_target(frame, target_col)
        if target.isna().any():
            raise ValueError(
                f"Cohort {cohort_name!r} still has {int(target.isna().sum())} unknown outcomes"
            )
        target_float = target.astype(float)
        n = int(len(frame))
        events = int(target_float.sum())
        row: dict[str, Any] = {
            "cohort": cohort_name,
            "n_participants": n,
            "n_events": events,
            "n_non_events": n - events,
            "event_prevalence_unweighted": events / n if n else np.nan,
            "weight_column": pd.NA,
            "weight_sum": np.nan,
            "weighted_event_sum": np.nan,
            "event_prevalence_weighted": np.nan,
        }
        weight_col = (weight_columns or {}).get(cohort_name)
        if weight_col is not None:
            if weight_col not in frame:
                raise KeyError(f"Missing weight column {weight_col!r} in {cohort_name!r}")
            weights = pd.to_numeric(frame[weight_col], errors="coerce")
            invalid = weights.isna() | ~np.isfinite(weights) | (weights < 0)
            if invalid.any():
                raise ValueError(
                    f"Cohort {cohort_name!r} has {int(invalid.sum())} missing, "
                    "non-finite, or negative survey weights"
                )
            weight_sum = float(weights.sum())
            if weight_sum <= 0:
                raise ValueError(f"Cohort {cohort_name!r} has a non-positive survey-weight sum")
            weighted_events = float((weights * target_float).sum())
            row.update(
                {
                    "weight_column": weight_col,
                    "weight_sum": weight_sum,
                    "weighted_event_sum": weighted_events,
                    "event_prevalence_weighted": (weighted_events / weight_sum),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def missingness_table(
    cohorts: Mapping[str, pd.DataFrame],
    *,
    features: Sequence[str] = FAST_FEATURES,
) -> pd.DataFrame:
    """Report missing values before any imputation."""

    rows: list[dict[str, Any]] = []
    for cohort_name, frame in cohorts.items():
        n = len(frame)
        for feature in features:
            available = feature in frame
            missing = int(frame[feature].isna().sum()) if available else n
            rows.append(
                {
                    "cohort": cohort_name,
                    "feature": feature,
                    "row_type": "summary",
                    "available": available,
                    "n_participants": n,
                    "n_missing": missing,
                    "missing_fraction": missing / n if n else np.nan,
                    "missing_reason": pd.NA,
                    "n_reason": pd.NA,
                    "reason_fraction": np.nan,
                }
            )
    return pd.DataFrame(rows)


def missingness_reason_table(
    cohorts: Mapping[str, pd.DataFrame],
    *,
    features: Sequence[str] = FAST_FEATURES,
) -> pd.DataFrame:
    """Break missing values down into refused/DK/NIU/source-specific reasons."""

    rows: list[dict[str, Any]] = []
    for cohort_name, frame in cohorts.items():
        n = len(frame)
        for feature in features:
            if feature not in frame:
                continue
            missing = frame[feature].isna()
            n_missing = int(missing.sum())
            if n_missing == 0:
                continue
            reason_col = f"{feature}_missing_reason"
            if reason_col not in frame:
                raise KeyError(
                    f"{cohort_name!r} has {n_missing} missing {feature!r} values "
                    f"but no {reason_col!r} provenance"
                )
            reason = frame[reason_col].astype("string")
            missing_without_reason = missing & reason.isna()
            reason_on_observed = ~missing & reason.notna()
            if missing_without_reason.any() or reason_on_observed.any():
                raise ValueError(
                    f"{cohort_name!r}/{feature!r} has inconsistent missing-reason provenance"
                )
            counts = reason.loc[missing].value_counts(dropna=False)
            for label, count in counts.items():
                rows.append(
                    {
                        "cohort": cohort_name,
                        "feature": feature,
                        "row_type": "reason",
                        "available": True,
                        "n_participants": n,
                        "n_missing": n_missing,
                        "missing_fraction": n_missing / n if n else np.nan,
                        "missing_reason": str(label),
                        "n_reason": int(count),
                        "reason_fraction": int(count) / n if n else np.nan,
                    }
                )
    return pd.DataFrame(
        rows,
        columns=(
            "cohort",
            "feature",
            "row_type",
            "available",
            "n_participants",
            "n_missing",
            "missing_fraction",
            "missing_reason",
            "n_reason",
            "reason_fraction",
        ),
    )


def subgroup_event_counts_table(
    cohorts: Mapping[str, pd.DataFrame],
    *,
    subgroup_col: str,
    target_col: str = "cvd_death_5y",
    weight_col: str = "analysis_weight",
    minimum_events: int = 20,
    minimum_non_events: int = 20,
) -> pd.DataFrame:
    """Assess whether audit-only subgroups have enough outcomes for evaluation."""

    rows: list[dict[str, Any]] = []
    for cohort_name, frame in cohorts.items():
        target = _validated_binary_target(frame, target_col)
        if target.isna().any():
            raise ValueError(f"{cohort_name!r} has unknown outcomes in subgroup audit")
        if weight_col not in frame:
            raise KeyError(f"{cohort_name!r} is missing {weight_col!r}")
        weights = pd.to_numeric(frame[weight_col], errors="coerce")
        if weights.isna().any() or (~np.isfinite(weights)).any() or (weights < 0).any():
            raise ValueError(f"{cohort_name!r} has invalid subgroup audit weights")
        if subgroup_col not in frame or frame[subgroup_col].notna().sum() == 0:
            rows.append(
                {
                    "cohort": cohort_name,
                    "subgroup": subgroup_col,
                    "category": pd.NA,
                    "status": "unavailable",
                    "n_participants": 0,
                    "n_events": 0,
                    "n_non_events": 0,
                    "n_missing_subgroup": len(frame),
                    "weighted_event_prevalence": np.nan,
                    "eligible_for_performance_audit": False,
                }
            )
            continue
        known = frame[subgroup_col].notna()
        for category in sorted(frame.loc[known, subgroup_col].astype(str).unique()):
            selected = known & frame[subgroup_col].astype("string").eq(category).fillna(False)
            category_target = target.loc[selected].astype(int)
            category_weights = weights.loc[selected]
            events = int(category_target.sum())
            n = int(selected.sum())
            non_events = n - events
            weight_sum = float(category_weights.sum())
            weighted_events = float((category_weights * category_target).sum())
            reliable = events >= minimum_events and non_events >= minimum_non_events
            rows.append(
                {
                    "cohort": cohort_name,
                    "subgroup": subgroup_col,
                    "category": category,
                    "status": "available",
                    "n_participants": n,
                    "n_events": events,
                    "n_non_events": non_events,
                    "n_missing_subgroup": int((~known).sum()),
                    "weighted_event_prevalence": (
                        weighted_events / weight_sum if weight_sum > 0 else np.nan
                    ),
                    "eligible_for_performance_audit": reliable,
                }
            )
    return pd.DataFrame(rows)


def feature_availability_table(
    cohorts: Mapping[str, pd.DataFrame],
    *,
    features: Sequence[str] = FAST_FEATURES,
) -> pd.DataFrame:
    """Return schema-level availability and observed non-null counts."""

    rows: list[dict[str, Any]] = []
    for cohort_name, frame in cohorts.items():
        for feature in features:
            available = feature in frame
            rows.append(
                {
                    "cohort": cohort_name,
                    "feature": feature,
                    "column_present": available,
                    "n_observed": int(frame[feature].notna().sum()) if available else 0,
                    "dtype": str(frame[feature].dtype) if available else pd.NA,
                }
            )
    return pd.DataFrame(rows)


def feature_availability_by_period_table(
    frame: pd.DataFrame,
    *,
    source: str,
    period_col: str,
    features: Sequence[str] = FAST_FEATURES,
    expected_periods: Sequence[Any] | None = None,
) -> pd.DataFrame:
    """Report feature availability for every survey year or two-year cycle."""

    if period_col not in frame:
        raise KeyError(f"Missing period column {period_col!r} for {source}")
    if frame[period_col].isna().any():
        raise ValueError(f"{source} contains missing values in {period_col!r}")
    rows: list[dict[str, Any]] = []
    periods = (
        list(expected_periods)
        if expected_periods is not None
        else frame[period_col].drop_duplicates().sort_values().tolist()
    )
    for period in periods:
        subset = frame.loc[frame[period_col] == period]
        for feature in features:
            present = feature in frame
            rows.append(
                {
                    "source": source,
                    "period_type": period_col,
                    "period": period,
                    "period_present": not subset.empty,
                    "feature": feature,
                    "column_present": present,
                    "n_participants": len(subset),
                    "n_observed": int(subset[feature].notna().sum()) if present else 0,
                    "observed_fraction": (
                        float(subset[feature].notna().mean()) if present else 0.0
                    ),
                    "dtype": str(subset[feature].dtype) if present else pd.NA,
                }
            )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class StopGateThresholds:
    """Pre-specified recommended event minima from the research protocol."""

    nhis_development_events: int = 500
    nhis_temporal_test_events: int = 200
    nhanes_external_events: int = 100
    events_per_effective_df: int = 20


@dataclass(frozen=True)
class StopGateResult:
    details: pd.DataFrame
    ready_for_modeling: bool
    summary: str


def evaluate_stop_gate(
    event_counts: pd.DataFrame,
    *,
    effective_degrees_of_freedom: int | None = None,
    thresholds: StopGateThresholds | None = None,
) -> StopGateResult:
    """Evaluate feasibility without relaxing any protocol criterion.

    Expected cohort names are ``nhis_development``, ``nhis_temporal_test`` and
    ``nhanes_external``. A missing cohort is a blocked check, never an implicit
    pass. The event-per-DF check remains blocked until a candidate logistic
    specification has a fixed effective degree of freedom.
    """

    thresholds = thresholds or StopGateThresholds()
    required_columns = {"cohort", "n_events"}
    missing_columns = required_columns.difference(event_counts.columns)
    if missing_columns:
        raise KeyError(f"Missing event-count columns: {sorted(missing_columns)}")

    event_requirements = {
        "nhis_development": thresholds.nhis_development_events,
        "nhis_temporal_test": thresholds.nhis_temporal_test_events,
        "nhanes_external": thresholds.nhanes_external_events,
    }
    by_cohort = event_counts.set_index("cohort")
    rows: list[dict[str, Any]] = []
    for cohort, minimum in event_requirements.items():
        if cohort not in by_cohort.index:
            observed: Any = pd.NA
            status = "blocked_missing_cohort"
        else:
            value = by_cohort.loc[cohort, "n_events"]
            if isinstance(value, pd.Series):
                raise ValueError(f"Duplicate event-count rows for {cohort!r}")
            observed = int(cast(Any, value))
            status = "pass" if observed >= minimum else "fail"
        rows.append(
            {
                "check": f"events:{cohort}",
                "observed": observed,
                "minimum": minimum,
                "status": status,
            }
        )

    development_events: int | None = None
    if "nhis_development" in by_cohort.index:
        value = by_cohort.loc["nhis_development", "n_events"]
        if not isinstance(value, pd.Series):
            development_events = int(cast(Any, value))
    if effective_degrees_of_freedom is None:
        epdf_observed: Any = pd.NA
        epdf_status = "blocked_model_df_not_fixed"
    elif effective_degrees_of_freedom <= 0:
        raise ValueError("effective_degrees_of_freedom must be positive")
    elif development_events is None:
        epdf_observed = pd.NA
        epdf_status = "blocked_missing_cohort"
    else:
        epdf_observed = development_events / effective_degrees_of_freedom
        epdf_status = "pass" if epdf_observed >= thresholds.events_per_effective_df else "fail"
    rows.append(
        {
            "check": "events_per_effective_df:nhis_development",
            "observed": epdf_observed,
            "minimum": thresholds.events_per_effective_df,
            "status": epdf_status,
        }
    )

    details = pd.DataFrame(rows)
    ready = bool((details["status"] == "pass").all())
    if ready:
        summary = "PASS: all pre-specified feasibility checks passed."
    else:
        failed = ", ".join(details.loc[details["status"] != "pass", "check"])
        summary = f"STOP: modeling is not authorized; unresolved checks: {failed}."
    return StopGateResult(details=details, ready_for_modeling=ready, summary=summary)
