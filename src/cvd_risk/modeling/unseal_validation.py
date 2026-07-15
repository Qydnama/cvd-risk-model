"""Pre-registered one-time held-out validation with a hard unseal gate.

The pure metric and join helpers are exercised only with synthetic labels before
authorization. Real outcome I/O is isolated in :func:`run_authorized_unseal_validation`
and cannot begin without a separately created authorization manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import yaml
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from cvd_risk.modeling.lock_amendment import portable_binding, repository_root
from cvd_risk.modeling.preprocessing import weighted_quantile
from cvd_risk.provenance import sha256_file
from cvd_risk.stage0 import _find_rscript

STAGE6_LABEL_FREE_COMMIT = "66041aa9880c36ec710137358d50f32b4f900d54"
FROZEN_CUTOFF = 0.007469058716065273
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 20260714
OUTCOME_COLUMN = "cvd_death_5y"
IDENTIFIER_COLUMN = "participant_id"
SUBGROUP_COLUMNS = ("age", "sex", "race_ethnicity")
OUTCOME_LOAD_COLUMNS = (
    "participant_id",
    "cvd_death_5y",
    "age",
    "sex",
    "race_ethnicity",
    "split",
)
PREDICTION_REQUIRED_COLUMNS = (
    "cohort",
    "participant_id",
    "survey_cycle",
    "survey_year",
    "strata",
    "psu",
    "analysis_weight",
    "predicted_risk",
    "linear_predictor",
    "secondary_screen_positive",
)
PRIMARY_METRICS = (
    "weighted_prevalence",
    "roc_auc",
    "average_precision",
    "brier_score",
    "log_loss",
    "calibration_intercept",
    "calibration_slope",
    "observed_expected_ratio",
    "log_observed_expected_ratio",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "flagged_fraction",
    "tp_per_10000",
    "fn_per_10000",
    "fp_per_10000",
    "tn_per_10000",
    "net_benefit_model",
    "net_benefit_treat_all",
    "net_benefit_treat_none",
)
SUBGROUP_METRICS = (
    "roc_auc",
    "brier_score",
    "calibration_intercept",
    "calibration_slope",
    "sensitivity",
    "ppv",
)
PLAN_TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".lock",
    ".md",
    ".py",
    ".r",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def _git(*args: str, root: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


def _plan_binding(path: str | Path, *, root: Path) -> dict[str, str]:
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Plan binding must be repository-relative: {relative}")
    target = root / relative
    payload = target.read_bytes()
    if relative.suffix.lower() in PLAN_TEXT_SUFFIXES:
        payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        hash_mode = "normalized_text_lf"
    else:
        hash_mode = "raw_bytes"
    return {
        "repository_relative_path": relative.as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "hash_mode": hash_mode,
    }


def load_unseal_validation_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unseal validation config schema_version must be 1")
    if payload.get("stage") != "stage6_preregistered_unseal_validation":
        raise ValueError("Unexpected unseal validation stage")
    if payload.get("status") != "planned_outcomes_sealed":
        raise ValueError("Unseal config must remain in the sealed planning state")
    bindings = payload.get("bindings", {})
    if bindings.get("stage6_label_free_commit") != STAGE6_LABEL_FREE_COMMIT:
        raise ValueError("Unseal plan is not bound to the accepted Stage 6 commit")
    analysis = payload.get("analysis", {})
    fixed = analysis.get("fixed_cutoff", {})
    if fixed.get("value") != FROZEN_CUTOFF:
        raise ValueError("Unseal plan cutoff differs from the immutable lock")
    if fixed.get("reselection_on_heldout_data") != "forbidden":
        raise ValueError("Held-out cutoff reselection is not forbidden")
    bootstrap = analysis.get("cluster_bootstrap", {})
    if (
        bootstrap.get("method")
        != "stratified_psu_with_replacement_within_strata"
        or bootstrap.get("repetitions") != BOOTSTRAP_REPETITIONS
        or bootstrap.get("seed") != BOOTSTRAP_SEED
    ):
        raise ValueError("Unseal cluster-bootstrap policy changed")
    contract = payload.get("outcome_contract", {})
    if contract.get("identifier_column") != IDENTIFIER_COLUMN:
        raise ValueError("Unexpected outcome identifier")
    if contract.get("outcome_columns") != [OUTCOME_COLUMN]:
        raise ValueError("Outcome allowlist must contain only cvd_death_5y")
    if tuple(contract.get("subgroup_columns", [])) != SUBGROUP_COLUMNS:
        raise ValueError("Unexpected subgroup/audit allowlist")
    if payload.get("join_contract", {}).get("cardinality") != "one_to_one":
        raise ValueError("Outcome join must be one-to-one")
    expected_cohorts = {"nhis_temporal", "nhanes_external"}
    cohorts = payload.get("cohorts", {})
    if set(cohorts) != expected_cohorts:
        raise ValueError("Unseal plan requires both locked validation cohorts")
    for name, settings in cohorts.items():
        if tuple(settings.get("outcome_load_allowlist", [])) != OUTCOME_LOAD_COLUMNS:
            raise ValueError(f"{name} outcome load allowlist changed")
        if int(settings.get("expected_n", 0)) <= 0:
            raise ValueError(f"{name} expected row count is invalid")
    immutable = analysis.get("immutable_components", {})
    if immutable != {
        "model": "no_change",
        "preprocessing": "no_change",
        "calibrator": "no_change",
        "cutoff": "no_change",
    }:
        raise ValueError("Unseal plan permits a locked component to change")
    if analysis.get("interpretation") != "underpowered_exploratory":
        raise ValueError("Validation must remain underpowered/exploratory")
    return payload


def join_outcomes_one_to_one(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    allowed_outcome_columns: tuple[str, ...] = OUTCOME_LOAD_COLUMNS,
) -> pd.DataFrame:
    """Join a strict outcome allowlist without tolerating any ID mismatch."""

    missing_prediction = set(PREDICTION_REQUIRED_COLUMNS).difference(predictions.columns)
    if missing_prediction:
        raise KeyError(f"Prediction fields are missing: {sorted(missing_prediction)}")
    unexpected_outcome = set(outcomes.columns).difference(allowed_outcome_columns)
    missing_outcome = set(allowed_outcome_columns).difference(outcomes.columns)
    if unexpected_outcome or missing_outcome:
        raise ValueError(
            "Outcome frame violates the exact allowlist: "
            f"unexpected={sorted(unexpected_outcome)} missing={sorted(missing_outcome)}"
        )
    for label, frame in (("prediction", predictions), ("outcome", outcomes)):
        identifiers = frame[IDENTIFIER_COLUMN]
        if identifiers.isna().any():
            raise ValueError(f"{label} participant IDs contain missing values")
        if identifiers.duplicated().any():
            raise ValueError(f"{label} participant IDs contain duplicates")
    prediction_ids = set(predictions[IDENTIFIER_COLUMN].astype(str))
    outcome_ids = set(outcomes[IDENTIFIER_COLUMN].astype(str))
    if prediction_ids != outcome_ids:
        raise ValueError(
            "Prediction/outcome ID sets differ: "
            f"unmatched_predictions={len(prediction_ids - outcome_ids)} "
            f"unmatched_outcomes={len(outcome_ids - prediction_ids)}"
        )
    ordered = predictions.copy()
    ordered["_prediction_order"] = np.arange(len(ordered), dtype=np.int64)
    joined = ordered.merge(
        outcomes,
        on=IDENTIFIER_COLUMN,
        how="left",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    if not joined["_merge"].eq("both").all() or len(joined) != len(predictions):
        raise RuntimeError("One-to-one join changed validation cohort membership")
    joined = joined.sort_values("_prediction_order", kind="stable").drop(
        columns=["_prediction_order", "_merge", "split"]
    )
    if not joined[IDENTIFIER_COLUMN].astype(str).reset_index(drop=True).equals(
        predictions[IDENTIFIER_COLUMN].astype(str).reset_index(drop=True)
    ):
        raise RuntimeError("Outcome join did not preserve prediction row order")
    target = joined[OUTCOME_COLUMN]
    if target.isna().any() or not set(target.unique()).issubset({0, 1}):
        raise ValueError("Locked validation outcome must be complete and binary")
    return joined.reset_index(drop=True)


def _validate_metric_arrays(
    prediction: np.ndarray, outcome: np.ndarray, weight: np.ndarray
) -> None:
    if not (len(prediction) == len(outcome) == len(weight)) or len(prediction) == 0:
        raise ValueError("Metric arrays must have the same non-zero length")
    if not np.isfinite(prediction).all() or not np.isfinite(weight).all():
        raise ValueError("Predictions and weights must be finite")
    if (prediction < 0).any() or (prediction > 1).any() or (weight <= 0).any():
        raise ValueError("Predictions must be in [0,1] and weights must be positive")
    if set(np.unique(outcome)) != {0, 1}:
        raise ValueError("Metrics require both binary outcome classes")


def _weighted_calibration_fit(
    linear_predictor: np.ndarray,
    outcome: np.ndarray,
    weight: np.ndarray,
    *,
    max_iter: int = 100,
    tolerance: float = 1.0e-10,
) -> tuple[float, float]:
    center = float(np.average(linear_predictor, weights=weight))
    scale = float(
        np.sqrt(np.average((linear_predictor - center) ** 2, weights=weight))
    )
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Calibration slope requires a non-constant linear predictor")
    standardized = (linear_predictor - center) / scale
    x = np.column_stack((np.ones(len(linear_predictor)), standardized))
    beta = np.array([0.0, scale], dtype=np.float64)
    scaled_weight = weight / np.mean(weight)

    def log_likelihood(candidate: np.ndarray) -> float:
        eta = x @ candidate
        return float(
            np.sum(scaled_weight * (outcome * eta - np.logaddexp(0.0, eta)))
        )

    for _ in range(max_iter):
        eta = np.clip(x @ beta, -35.0, 35.0)
        fitted = 1.0 / (1.0 + np.exp(-eta))
        score = x.T @ (scaled_weight * (outcome - fitted))
        information = x.T @ (
            (scaled_weight * fitted * (1.0 - fitted))[:, np.newaxis] * x
        )
        step = np.linalg.solve(information, score)
        current = log_likelihood(beta)
        factor = 1.0
        while factor >= 1.0e-8:
            candidate = beta + factor * step
            if log_likelihood(candidate) >= current:
                beta = candidate
                break
            factor /= 2.0
        if factor < 1.0e-8:
            raise RuntimeError("Weighted calibration line search failed")
        if float(np.max(np.abs(factor * step))) <= tolerance:
            slope = float(beta[1] / scale)
            intercept = float(beta[0] - slope * center)
            return intercept, slope
    raise RuntimeError("Weighted calibration regression did not converge")


def weighted_original_metrics(
    frame: pd.DataFrame,
    *,
    cutoff: float = FROZEN_CUTOFF,
    probability_clip: float = 1.0e-6,
) -> dict[str, float]:
    prediction = frame["predicted_risk"].to_numpy(dtype=np.float64)
    outcome = frame[OUTCOME_COLUMN].to_numpy(dtype=np.int8)
    weight = frame["analysis_weight"].to_numpy(dtype=np.float64)
    _validate_metric_arrays(prediction, outcome, weight)
    bounded = np.clip(prediction, probability_clip, 1.0 - probability_clip)
    lp = np.log(bounded / (1.0 - bounded))
    intercept, slope = _weighted_calibration_fit(lp, outcome, weight)
    observed = float(np.sum(weight * outcome))
    expected = float(np.sum(weight * prediction))
    oe = observed / expected
    positive = prediction >= cutoff
    tp = float(weight[positive & (outcome == 1)].sum())
    fn = float(weight[(~positive) & (outcome == 1)].sum())
    fp = float(weight[positive & (outcome == 0)].sum())
    tn = float(weight[(~positive) & (outcome == 0)].sum())
    total = tp + fn + fp + tn

    def safe_ratio(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator > 0 else float("nan")

    odds = cutoff / (1.0 - cutoff)
    net_benefit_model = tp / total - fp / total * odds
    net_benefit_treat_all = (tp + fn) / total - (fp + tn) / total * odds
    operating = {
        "sensitivity": safe_ratio(tp, tp + fn),
        "specificity": safe_ratio(tn, fp + tn),
        "ppv": safe_ratio(tp, tp + fp),
        "npv": safe_ratio(tn, fn + tn),
        "flagged_fraction": (tp + fp) / total,
        "tp_per_10000": tp / total * 10_000.0,
        "fn_per_10000": fn / total * 10_000.0,
        "fp_per_10000": fp / total * 10_000.0,
        "tn_per_10000": tn / total * 10_000.0,
        "net_benefit_model": net_benefit_model,
        "net_benefit_treat_all": net_benefit_treat_all,
        "net_benefit_treat_none": 0.0,
    }
    return {
        "weighted_prevalence": float(np.average(outcome, weights=weight)),
        "roc_auc": float(roc_auc_score(outcome, prediction, sample_weight=weight)),
        "average_precision": float(
            average_precision_score(outcome, prediction, sample_weight=weight)
        ),
        "brier_score": float(brier_score_loss(outcome, prediction, sample_weight=weight)),
        "log_loss": float(log_loss(outcome, bounded, sample_weight=weight, labels=[0, 1])),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "observed_expected_ratio": oe,
        "log_observed_expected_ratio": math.log(oe),
        **{key: float(operating[key]) for key in PRIMARY_METRICS if key in operating},
    }


def recalibration_in_the_large(
    frame: pd.DataFrame,
    *,
    probability_clip: float = 1.0e-6,
) -> tuple[float, np.ndarray]:
    prediction = frame["predicted_risk"].to_numpy(dtype=np.float64)
    outcome = frame[OUTCOME_COLUMN].to_numpy(dtype=np.int8)
    weight = frame["analysis_weight"].to_numpy(dtype=np.float64)
    _validate_metric_arrays(prediction, outcome, weight)
    bounded = np.clip(prediction, probability_clip, 1.0 - probability_clip)
    lp = np.log(bounded / (1.0 - bounded))
    target = float(np.sum(weight * outcome))
    lower, upper = -30.0, 30.0
    for _ in range(200):
        intercept = (lower + upper) / 2.0
        fitted = 1.0 / (1.0 + np.exp(-np.clip(lp + intercept, -35.0, 35.0)))
        if float(np.sum(weight * fitted)) < target:
            lower = intercept
        else:
            upper = intercept
        if upper - lower <= 1.0e-12:
            break
    intercept = (lower + upper) / 2.0
    recalibrated = 1.0 / (1.0 + np.exp(-np.clip(lp + intercept, -35.0, 35.0)))
    return float(intercept), recalibrated


def calibration_bin_boundaries(
    prediction: np.ndarray,
    weight: np.ndarray,
    *,
    requested_bins: int = 10,
) -> np.ndarray:
    if requested_bins < 2:
        raise ValueError("Calibration plot requires at least two requested bins")
    quantiles = np.arange(1, requested_bins, dtype=np.float64) / requested_bins
    internal = weighted_quantile(prediction, quantiles, weight)
    return np.r_[-np.inf, np.unique(internal), np.inf]


def calibration_bin_table(
    frame: pd.DataFrame,
    *,
    boundaries: np.ndarray | None = None,
    requested_bins: int = 10,
) -> pd.DataFrame:
    prediction = frame["predicted_risk"].to_numpy(dtype=np.float64)
    outcome = frame[OUTCOME_COLUMN].to_numpy(dtype=np.int8)
    weight = frame["analysis_weight"].to_numpy(dtype=np.float64)
    _validate_metric_arrays(prediction, outcome, weight)
    edges = (
        calibration_bin_boundaries(prediction, weight, requested_bins=requested_bins)
        if boundaries is None
        else np.asarray(boundaries, dtype=np.float64)
    )
    if len(edges) < 3 or not np.all(np.diff(edges) > 0):
        raise ValueError("Calibration-bin boundaries must be strictly increasing")
    bin_number = np.searchsorted(edges[1:-1], prediction, side="right") + 1
    rows: list[dict[str, Any]] = []
    for number in range(1, len(edges)):
        selected = bin_number == number
        if not selected.any():
            continue
        selected_weight = weight[selected]
        rows.append(
            {
                "bin": number,
                "lower_boundary": float(edges[number - 1]),
                "upper_boundary": float(edges[number]),
                "n": int(selected.sum()),
                "events": int(outcome[selected].sum()),
                "weight_sum": float(selected_weight.sum()),
                "weighted_mean_prediction": float(
                    np.average(prediction[selected], weights=selected_weight)
                ),
                "weighted_observed_fraction": float(
                    np.average(outcome[selected], weights=selected_weight)
                ),
            }
        )
    return pd.DataFrame(rows)


def _cluster_structure(frame: pd.DataFrame) -> list[np.ndarray]:
    clusters_by_stratum: list[np.ndarray] = []
    for _, stratum in frame.groupby("strata", sort=True, observed=True):
        clusters = stratum["psu"].drop_duplicates().to_numpy()
        if len(clusters) < 2:
            raise ValueError("Cluster bootstrap requires at least two PSUs per stratum")
        clusters_by_stratum.append(clusters)
    return clusters_by_stratum


def _bootstrap_multiplicity_by_groups(
    frame: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    multiplicity = np.zeros(len(frame), dtype=np.int16)
    for _, block in frame.groupby("strata", sort=True, observed=True):
        cluster_indices = [
            indices.to_numpy(dtype=np.int64)
            for _, indices in block.groupby("psu", sort=True, observed=True).groups.items()
        ]
        if len(cluster_indices) < 2:
            raise ValueError("Cluster bootstrap requires at least two PSUs per stratum")
        draws = rng.integers(0, len(cluster_indices), size=len(cluster_indices))
        counts = np.bincount(draws, minlength=len(cluster_indices))
        for indices, count in zip(cluster_indices, counts, strict=True):
            multiplicity[indices] = int(count)
    return multiplicity


def stratified_psu_metric_bootstrap(
    frame: pd.DataFrame,
    *,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
    cutoff: float = FROZEN_CUTOFF,
) -> pd.DataFrame:
    if repetitions < 2:
        raise ValueError("At least two bootstrap repetitions are required")
    _cluster_structure(frame)
    base_weight = frame["analysis_weight"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for replicate in range(1, repetitions + 1):
        multiplicity = _bootstrap_multiplicity_by_groups(frame, rng)
        selected = multiplicity > 0
        replicate_frame = frame.loc[selected].copy()
        replicate_frame["analysis_weight"] = (
            base_weight[selected] * multiplicity[selected].astype(np.float64)
        )
        row: dict[str, Any] = {"replicate": replicate, "valid": True}
        try:
            row.update(weighted_original_metrics(replicate_frame, cutoff=cutoff))
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            row["valid"] = False
            row.update({metric: np.nan for metric in PRIMARY_METRICS})
        rows.append(row)
    return pd.DataFrame(rows)


def stratified_psu_subgroup_bootstrap(
    frame: pd.DataFrame,
    eligibility: pd.DataFrame,
    *,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
    cutoff: float = FROZEN_CUTOFF,
) -> pd.DataFrame:
    if repetitions < 2:
        raise ValueError("At least two bootstrap repetitions are required")
    _cluster_structure(frame)
    eligible = eligibility.loc[eligibility["eligible_for_performance"]].copy()
    base = frame.copy()
    base["age_group"] = pd.cut(
        pd.to_numeric(base["age"], errors="coerce"),
        bins=[39, 54, 64, 79],
        labels=["40-54", "55-64", "65-79"],
        include_lowest=True,
        right=True,
    ).astype("string")
    base_weight = base["analysis_weight"].to_numpy(dtype=np.float64)
    masks = {
        (str(row.dimension), str(row.category)): base[str(row.dimension)]
        .astype("string")
        .eq(str(row.category))
        .fillna(False)
        .to_numpy()
        for row in eligible.itertuples(index=False)
    }
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for replicate in range(1, repetitions + 1):
        multiplicity = _bootstrap_multiplicity_by_groups(base, rng)
        for (dimension, category), group_mask in masks.items():
            selected = (multiplicity > 0) & group_mask
            record: dict[str, Any] = {
                "replicate": replicate,
                "dimension": dimension,
                "category": category,
                "valid": True,
            }
            replicate_frame = base.loc[selected].copy()
            replicate_frame["analysis_weight"] = (
                base_weight[selected] * multiplicity[selected].astype(np.float64)
            )
            try:
                metrics = weighted_original_metrics(replicate_frame, cutoff=cutoff)
                record.update({metric: metrics[metric] for metric in SUBGROUP_METRICS})
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                record["valid"] = False
                record.update({metric: np.nan for metric in SUBGROUP_METRICS})
            rows.append(record)
    return pd.DataFrame(rows)


def add_subgroup_percentile_intervals(
    point: pd.DataFrame,
    bootstrap: pd.DataFrame,
    *,
    confidence_level: float = 0.95,
    minimum_valid_replicates: int = 1900,
) -> pd.DataFrame:
    alpha = 1.0 - confidence_level
    rows: list[dict[str, Any]] = []
    for entry in point.itertuples(index=False):
        base = {
            "dimension": entry.dimension,
            "category": entry.category,
            "n": entry.n,
            "events": entry.events,
            "non_events": entry.non_events,
            "eligible_for_performance": entry.eligible_for_performance,
            "status": entry.status,
        }
        for metric in SUBGROUP_METRICS:
            if not entry.eligible_for_performance:
                rows.append(
                    {
                        **base,
                        "metric": metric,
                        "estimate": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "valid_replicates": 0,
                    }
                )
                continue
            selected = bootstrap.loc[
                bootstrap["dimension"].eq(entry.dimension)
                & bootstrap["category"].eq(entry.category),
                metric,
            ].to_numpy(dtype=np.float64)
            finite = selected[np.isfinite(selected)]
            valid = len(finite)
            enough = valid >= minimum_valid_replicates
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "estimate": float(getattr(entry, metric)),
                    "ci_lower": (
                        float(np.quantile(finite, alpha / 2.0)) if enough else np.nan
                    ),
                    "ci_upper": (
                        float(np.quantile(finite, 1.0 - alpha / 2.0))
                        if enough
                        else np.nan
                    ),
                    "valid_replicates": valid,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_calibration_bands(
    frame: pd.DataFrame,
    boundaries: np.ndarray,
    *,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
    confidence_level: float = 0.95,
    minimum_valid_replicates: int = 1900,
) -> pd.DataFrame:
    _cluster_structure(frame)
    point = calibration_bin_table(frame, boundaries=boundaries)
    base_weight = frame["analysis_weight"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    observed: dict[int, list[float]] = {int(value): [] for value in point["bin"]}
    predicted: dict[int, list[float]] = {int(value): [] for value in point["bin"]}
    for _ in range(repetitions):
        multiplicity = _bootstrap_multiplicity_by_groups(frame, rng)
        selected = multiplicity > 0
        replicate = frame.loc[selected].copy()
        replicate["analysis_weight"] = (
            base_weight[selected] * multiplicity[selected].astype(np.float64)
        )
        table = calibration_bin_table(replicate, boundaries=boundaries).set_index("bin")
        for number in observed:
            if number not in table.index:
                observed[number].append(np.nan)
                predicted[number].append(np.nan)
            else:
                observed[number].append(
                    float(
                        np.asarray(
                            table.loc[number, "weighted_observed_fraction"],
                            dtype=np.float64,
                        )
                    )
                )
                predicted[number].append(
                    float(
                        np.asarray(
                            table.loc[number, "weighted_mean_prediction"],
                            dtype=np.float64,
                        )
                    )
                )
    alpha = 1.0 - confidence_level
    point = point.copy()
    for label, values_by_bin in (("observed", observed), ("predicted", predicted)):
        lower: list[float] = []
        upper: list[float] = []
        valid_counts: list[int] = []
        for number in point["bin"].astype(int):
            values = np.asarray(values_by_bin[number], dtype=np.float64)
            finite = values[np.isfinite(values)]
            valid_counts.append(len(finite))
            if len(finite) < minimum_valid_replicates:
                lower.append(np.nan)
                upper.append(np.nan)
            else:
                lower.append(float(np.quantile(finite, alpha / 2.0)))
                upper.append(float(np.quantile(finite, 1.0 - alpha / 2.0)))
        point[f"{label}_ci_lower"] = lower
        point[f"{label}_ci_upper"] = upper
        point[f"{label}_valid_replicates"] = valid_counts
    point["requested_bins"] = 10
    point["actual_bins"] = len(point)
    return point


def period_stability_table(
    frame: pd.DataFrame,
    *,
    period_column: str,
    expected_periods: list[Any],
    cutoff: float = FROZEN_CUTOFF,
) -> pd.DataFrame:
    observed = sorted(frame[period_column].dropna().unique().tolist())
    if observed != sorted(expected_periods):
        raise ValueError("Validation periods differ from the pre-registered enumeration")
    rows: list[dict[str, Any]] = []
    for period in expected_periods:
        selected = frame.loc[frame[period_column].eq(period)]
        target = selected[OUTCOME_COLUMN].to_numpy(dtype=np.int8)
        row: dict[str, Any] = {
            "period": period,
            "n": len(selected),
            "events": int(target.sum()),
            "status": "descriptive_exploratory",
        }
        if set(np.unique(target)) == {0, 1}:
            row.update(weighted_original_metrics(selected, cutoff=cutoff))
        else:
            row.update({metric: np.nan for metric in PRIMARY_METRICS})
        rows.append(row)
    return pd.DataFrame(rows)


def recalibration_sensitivity_table(frame: pd.DataFrame) -> pd.DataFrame:
    original = weighted_original_metrics(frame)
    intercept, recalibrated = recalibration_in_the_large(frame)
    sensitivity_frame = frame.copy()
    sensitivity_frame["predicted_risk"] = recalibrated
    recalibrated_metrics = weighted_original_metrics(sensitivity_frame)
    metrics = ("brier_score", "log_loss", "calibration_intercept", "observed_expected_ratio")
    return pd.DataFrame(
        [
            {
                "analysis": "recalibration_in_the_large_sensitivity_only",
                "offset_intercept": intercept,
                "metric": metric,
                "original": original[metric],
                "recalibrated": recalibrated_metrics[metric],
                "primary_predictions_replaced": False,
                "serialized_calibrator_updated": False,
            }
            for metric in metrics
        ]
    )


def save_calibration_plot(table: pd.DataFrame, path: Path, *, cohort: str) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="Identity")
    axis.errorbar(
        table["weighted_mean_prediction"],
        table["weighted_observed_fraction"],
        yerr=np.vstack(
            (
                table["weighted_observed_fraction"] - table["observed_ci_lower"],
                table["observed_ci_upper"] - table["weighted_observed_fraction"],
            )
        ),
        fmt="o-",
        capsize=3,
        label="Survey-weighted bins",
    )
    maximum = float(
        max(
            table["weighted_mean_prediction"].max(),
            table["observed_ci_upper"].max(),
        )
    )
    axis.set_xlim(0, maximum * 1.08)
    axis.set_ylim(0, maximum * 1.08)
    axis.set_xlabel("Survey-weighted mean predicted risk")
    axis.set_ylabel("Survey-weighted observed event fraction")
    axis.set_title(f"Calibration: {cohort} (underpowered/exploratory)")
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _verify_stage6_binding(config: dict[str, Any], *, root: Path) -> dict[str, Any]:
    bindings = config["bindings"]
    stage6_commit = str(bindings["stage6_label_free_commit"])
    if _git("rev-parse", stage6_commit, root=root) != STAGE6_LABEL_FREE_COMMIT:
        raise ValueError("Accepted Stage 6 commit is unavailable")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", stage6_commit, "HEAD"],
        cwd=root,
        check=False,
    ).returncode != 0:
        raise ValueError("Accepted Stage 6 commit is not an ancestor of HEAD")
    manifest_path = root / bindings["stage6_label_free_manifest"]
    stage6 = json.loads(manifest_path.read_text(encoding="utf-8"))
    if stage6.get("status") != "LABEL_FREE_PREDICTIONS_FROZEN_OUTCOMES_STILL_SEALED":
        raise ValueError("Stage 6 label-free manifest has an unexpected status")
    if stage6.get("heldout_outcomes_accessed") is not False:
        raise ValueError("Stage 6 label-free manifest reports outcome access")
    if stage6.get("heldout_outcomes_unsealed") is not False:
        raise ValueError("Stage 6 label-free manifest reports outcome unsealing")
    if stage6["pipeline"]["sha256"] != bindings["pipeline_sha256"]:
        raise ValueError("Unseal plan pipeline SHA differs from Stage 6")
    for cohort, settings in config["cohorts"].items():
        stage6_entry = stage6["cohorts"][cohort]
        if stage6_entry["prediction_sha256"] != settings["prediction_sha256"]:
            raise ValueError(f"{cohort} prediction SHA differs from Stage 6")
        if stage6_entry["n"] != settings["expected_n"]:
            raise ValueError(f"{cohort} planned N differs from Stage 6")
    return stage6


def create_unseal_plan_manifest(
    *,
    config_path: str | Path = "configs/stage6_unseal_validation.yaml",
    output_path: str | Path = "artifacts/stage6_unseal_plan/unseal_plan_manifest.json",
    force: bool = False,
) -> dict[str, Any]:
    root = repository_root()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = load_unseal_validation_config(config_file)
    stage6 = _verify_stage6_binding(config, root=root)
    authorization_path = root / config["authorization"]["manifest_path"]
    receipt_path = root / config["authorization"]["access_receipt_path"]
    if authorization_path.exists() or receipt_path.exists():
        raise RuntimeError("Cannot preregister after authorization or unseal receipt exists")
    target = Path(output_path)
    if not target.is_absolute():
        target = root / target
    if target.exists() and not force:
        raise FileExistsError("Unseal plan manifest already exists")
    source_paths = [
        config_file,
        root / "src/cvd_risk/modeling/unseal_validation.py",
        root / "scripts/16_create_unseal_plan.py",
        root / "scripts/17_validate_unseal_plan.py",
        root / "scripts/18_external_validate_after_unseal.py",
        root / "scripts/r/stage6_design_risk_precision.R",
        root / "tests/test_unseal_validation.py",
    ]
    missing = [path for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Unseal plan source files are absent: {missing}")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "stage": "stage6_preregistered_unseal_validation",
        "status": "PREREGISTERED_OUTCOMES_REMAIN_SEALED",
        "created_on": datetime.now(UTC).isoformat(),
        "heldout_outcomes_loaded": False,
        "heldout_outcomes_joined": False,
        "heldout_outcomes_analyzed": False,
        "outcomes_unsealed": False,
        "real_data_execution_performed": False,
        "interpretation_locked": "underpowered_exploratory",
        "precision_gates": {
            "nhis_temporal": "FAIL",
            "nhanes_external": "FAIL",
        },
        "locked_bindings": {
            "original_lock": _plan_binding(
                config["bindings"]["original_lock"], root=root
            ),
            "lock_amendment": _plan_binding(
                config["bindings"]["lock_amendment"], root=root
            ),
            "stage6_label_free_manifest": _plan_binding(
                config["bindings"]["stage6_label_free_manifest"], root=root
            ),
            "stage6_label_free_commit": STAGE6_LABEL_FREE_COMMIT,
            "pipeline": _plan_binding(config["bindings"]["pipeline"], root=root),
            "prediction_sha256": {
                cohort: entry["prediction_sha256"]
                for cohort, entry in stage6["cohorts"].items()
            },
        },
        "outcome_contract": config["outcome_contract"],
        "join_contract": config["join_contract"],
        "cohort_contracts": {
            name: {
                "outcome_source_repository_relative_path": settings["outcome_source"],
                "outcome_source_sha256": settings["outcome_source_sha256"],
                "outcome_load_allowlist": settings["outcome_load_allowlist"],
                "split_value": settings["split_value"],
                "expected_n": settings["expected_n"],
                "expected_periods": settings["expected_periods"],
            }
            for name, settings in config["cohorts"].items()
        },
        "analysis_contract": config["analysis"],
        "authorization_contract": config["authorization"],
        "synthetic_label_tests": {
            "path": "tests/test_unseal_validation.py",
            "real_heldout_paths_used": False,
            "required_before_unseal": True,
        },
        "source_bindings": {
            path.relative_to(root).as_posix(): _plan_binding(
                path.relative_to(root), root=root
            )
            for path in source_paths
        },
        "forbidden_changes_after_unseal": [
            "model",
            "preprocessing",
            "calibrator",
            "cutoff",
            "prediction_artifacts",
            "outcome_contract",
            "metric_definitions",
            "bootstrap_seed_or_repetitions",
        ],
        "next_action": "request_separate_explicit_one_time_unseal_authorization",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_unseal_plan(
    *,
    config_path: str | Path = "configs/stage6_unseal_validation.yaml",
    manifest_path: str | Path = "artifacts/stage6_unseal_plan/unseal_plan_manifest.json",
    require_authorization_absent: bool = True,
) -> dict[str, Any]:
    root = repository_root()
    config_file = Path(config_path)
    manifest_file = Path(manifest_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    if not manifest_file.is_absolute():
        manifest_file = root / manifest_file
    config = load_unseal_validation_config(config_file)
    stage6 = _verify_stage6_binding(config, root=root)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("status") != "PREREGISTERED_OUTCOMES_REMAIN_SEALED":
        raise ValueError("Unexpected unseal plan manifest status")
    sealed_flags = (
        "heldout_outcomes_loaded",
        "heldout_outcomes_joined",
        "heldout_outcomes_analyzed",
        "outcomes_unsealed",
        "real_data_execution_performed",
    )
    if any(manifest.get(flag) is not False for flag in sealed_flags):
        raise ValueError("Unseal plan manifest reports held-out data access")
    if manifest.get("interpretation_locked") != "underpowered_exploratory":
        raise ValueError("Unseal plan interpretation changed")
    if manifest.get("outcome_contract") != config["outcome_contract"]:
        raise ValueError("Manifest outcome contract differs from config")
    if manifest.get("join_contract") != config["join_contract"]:
        raise ValueError("Manifest join contract differs from config")
    if manifest.get("analysis_contract") != config["analysis"]:
        raise ValueError("Manifest analysis contract differs from config")
    if (
        manifest["locked_bindings"]["stage6_label_free_commit"]
        != STAGE6_LABEL_FREE_COMMIT
    ):
        raise ValueError("Manifest Stage 6 commit changed")
    for name, relative in (
        ("original_lock", config["bindings"]["original_lock"]),
        ("lock_amendment", config["bindings"]["lock_amendment"]),
        ("stage6_label_free_manifest", config["bindings"]["stage6_label_free_manifest"]),
        ("pipeline", config["bindings"]["pipeline"]),
    ):
        if manifest["locked_bindings"][name] != _plan_binding(relative, root=root):
            raise ValueError(f"Locked unseal-plan binding changed: {name}")
    for cohort, entry in manifest["locked_bindings"]["prediction_sha256"].items():
        if entry != stage6["cohorts"][cohort]["prediction_sha256"]:
            raise ValueError(f"Manifest prediction SHA changed: {cohort}")
    for relative, entry in manifest["source_bindings"].items():
        current = _plan_binding(relative, root=root)
        if current != entry:
            raise ValueError(f"Unseal plan source binding changed: {relative}")
    if require_authorization_absent:
        authorization = root / config["authorization"]["manifest_path"]
        receipt = root / config["authorization"]["access_receipt_path"]
        if authorization.exists() or receipt.exists():
            raise RuntimeError("Pre-unseal validation found authorization/access state")
    return {
        "status": "UNSEAL_PLAN_VALID_OUTCOMES_SEALED",
        "stage6_commit": STAGE6_LABEL_FREE_COMMIT,
        "prediction_sha256": manifest["locked_bindings"]["prediction_sha256"],
        "heldout_outcomes_loaded": False,
        "outcomes_unsealed": False,
    }


def _plan_commit(root: Path, manifest_path: str) -> str:
    commit = _git("log", "-1", "--format=%H", "--", manifest_path, root=root)
    if len(commit) != 40:
        raise RuntimeError("Unseal plan manifest is not committed")
    return commit


def _verify_unseal_authorization(
    authorization: dict[str, Any],
    *,
    config: dict[str, Any],
    plan_manifest: dict[str, Any],
    root: Path,
) -> None:
    expected = {
        "schema_version": 1,
        "status": "AUTHORIZED_ONE_TIME_UNSEAL",
        "explicit_user_permission": True,
        "one_time_unseal": True,
        "stage6_label_free_commit": STAGE6_LABEL_FREE_COMMIT,
        "unseal_plan_commit": _plan_commit(
            root, "artifacts/stage6_unseal_plan/unseal_plan_manifest.json"
        ),
        "prediction_sha256": plan_manifest["locked_bindings"]["prediction_sha256"],
        "interpretation": "underpowered_exploratory",
        "model_change_authorized": False,
        "calibrator_change_authorized": False,
        "cutoff_change_authorized": False,
    }
    if any(authorization.get(key) != value for key, value in expected.items()):
        raise ValueError("Unseal authorization does not match the pre-registered scope")
    if authorization.get("authorized_by_user") is not True:
        raise ValueError("Unseal authorization lacks explicit user attribution")
    if not isinstance(authorization.get("authorized_at"), str):
        raise ValueError("Unseal authorization lacks an authorization timestamp")
    if config["analysis"]["fixed_cutoff"]["value"] != FROZEN_CUTOFF:
        raise ValueError("Frozen cutoff changed before unseal")


def _load_authorized_joined_cohort(
    name: str,
    settings: dict[str, Any],
    *,
    root: Path,
) -> pd.DataFrame:
    prediction_path = root / settings["prediction_path"]
    outcome_path = root / settings["outcome_source"]
    if sha256_file(prediction_path) != settings["prediction_sha256"]:
        raise ValueError(f"{name} frozen prediction SHA changed")
    if sha256_file(outcome_path) != settings["outcome_source_sha256"]:
        raise ValueError(f"{name} outcome source SHA changed")
    if pq.ParquetFile(prediction_path).schema_arrow.names != list(
        PREDICTION_REQUIRED_COLUMNS
    ):
        raise ValueError(f"{name} prediction schema changed")
    available = set(pq.ParquetFile(outcome_path).schema_arrow.names)
    allowlist = list(settings["outcome_load_allowlist"])
    missing = set(allowlist).difference(available)
    if missing:
        raise KeyError(f"{name} outcome allowlist fields are absent: {sorted(missing)}")
    predictions = pd.read_parquet(prediction_path)
    # This is the only real held-out outcome read in the project pipeline. PyArrow
    # receives the exact harmonized allowlist and the locked split filter; raw
    # mortality, cause-of-death and follow-up columns are never deserialized.
    outcomes = pd.read_parquet(
        outcome_path,
        columns=allowlist,
        filters=[("split", "==", settings["split_value"])],
    )
    predictions[IDENTIFIER_COLUMN] = predictions[IDENTIFIER_COLUMN].astype(str)
    outcomes[IDENTIFIER_COLUMN] = outcomes[IDENTIFIER_COLUMN].astype(str)
    if len(predictions) != int(settings["expected_n"]):
        raise ValueError(f"{name} prediction N differs from the plan")
    if len(outcomes) != int(settings["expected_n"]):
        raise ValueError(f"{name} outcome N differs from the plan")
    return join_outcomes_one_to_one(predictions, outcomes)


def _run_design_risk_precision(
    cohorts: dict[str, pd.DataFrame],
    *,
    root: Path,
    output_path: Path,
) -> pd.DataFrame:
    rscript = _find_rscript()
    if rscript is None:
        raise RuntimeError("BLOCKED_RSCRIPT_NOT_FOUND")
    r_script = root / "scripts/r/stage6_design_risk_precision.R"
    combined = pd.concat(
        [
            frame.assign(cohort=name).rename(
                columns={
                    "analysis_weight": "weight",
                    OUTCOME_COLUMN: "outcome",
                }
            )[["cohort", "outcome", "weight", "strata", "psu"]]
            for name, frame in cohorts.items()
        ],
        ignore_index=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cvd_stage6_unseal_design_") as directory:
        input_path = Path(directory) / "design_input.csv"
        combined.to_csv(input_path, index=False)
        completed = subprocess.run(
            [rscript, str(r_script), str(input_path), str(output_path)],
            capture_output=True,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            "R design-risk precision failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    return pd.read_csv(output_path)


def _write_validation_report(
    path: Path,
    *,
    overall: pd.DataFrame,
    intervals: pd.DataFrame,
    design_risk: pd.DataFrame,
) -> None:
    lines = [
        "# Stage 6 temporal and external validation",
        "",
        "## Mandatory interpretation",
        "",
        "All results are **underpowered/exploratory** because both pre-unseal "
        "survey precision gates failed. No claim of adequate precision is made. "
        "The frozen model, preprocessing, sigmoid calibrator and cutoff were not "
        "changed or reselected.",
        "",
        "## Original frozen-model overall metrics",
        "",
        overall.to_markdown(index=False),
        "",
        "## Stratified PSU cluster-bootstrap intervals",
        "",
        intervals.to_markdown(index=False),
        "",
        "## Design-based overall risk precision",
        "",
        design_risk.to_markdown(index=False),
        "",
        "Recalibration-in-the-large is reported separately as a sensitivity analysis "
        "and never replaces the primary predictions. The secondary cutoff remains "
        f"`{FROZEN_CUTOFF:.15f}` and is not a diagnosis or clinically endorsed boundary.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_authorized_unseal_validation(
    *,
    config_path: str | Path = "configs/stage6_unseal_validation.yaml",
    plan_manifest_path: str | Path = (
        "artifacts/stage6_unseal_plan/unseal_plan_manifest.json"
    ),
    authorization_path: str | Path = "artifacts/audit/unseal_authorization.json",
) -> dict[str, Any]:
    root = repository_root()
    config_file = root / config_path
    plan_file = root / plan_manifest_path
    authorization_file = root / authorization_path
    config = load_unseal_validation_config(config_file)
    validate_unseal_plan(
        config_path=config_file,
        manifest_path=plan_file,
        require_authorization_absent=False,
    )
    if not authorization_file.is_file():
        raise PermissionError(
            "UNSEAL_NOT_AUTHORIZED: a separately approved authorization manifest is required"
        )
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    authorization = json.loads(authorization_file.read_text(encoding="utf-8"))
    _verify_unseal_authorization(
        authorization,
        config=config,
        plan_manifest=plan,
        root=root,
    )
    receipt_path = root / config["authorization"]["access_receipt_path"]
    if receipt_path.exists():
        raise RuntimeError(
            "ONE_TIME_UNSEAL_ALREADY_STARTED: manual audit is required; no silent rerun"
        )
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "ONE_TIME_UNSEAL_ACCESS_STARTED",
        "started_at": datetime.now(UTC).isoformat(),
        "authorization": portable_binding(authorization_file.relative_to(root), root=root),
        "plan_commit": _plan_commit(
            root, "artifacts/stage6_unseal_plan/unseal_plan_manifest.json"
        ),
        "stage6_label_free_commit": STAGE6_LABEL_FREE_COMMIT,
        "outcomes_unsealed": True,
        "model_change_authorized": False,
        "calibrator_change_authorized": False,
        "cutoff_change_authorized": False,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    outputs = config["outputs_after_authorized_unseal"]
    artifact_dir = root / outputs["artifact_dir"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        key: root / value
        for key, value in outputs.items()
        if key not in {"artifact_dir", "calibration_figure_dir"}
    }
    known = [*output_paths.values(), artifact_dir / "manifest.json"]
    if any(path.exists() for path in known):
        raise FileExistsError("Post-unseal validation outputs already exist")

    try:
        cohorts = {
            name: _load_authorized_joined_cohort(name, settings, root=root)
            for name, settings in config["cohorts"].items()
        }
        overall_rows: list[dict[str, Any]] = []
        interval_tables: list[pd.DataFrame] = []
        calibration_tables: list[pd.DataFrame] = []
        subgroup_tables: list[pd.DataFrame] = []
        period_tables: list[pd.DataFrame] = []
        recalibration_tables: list[pd.DataFrame] = []
        bootstrap_config = config["analysis"]["cluster_bootstrap"]
        figure_dir = root / outputs["calibration_figure_dir"]
        figure_paths: dict[str, Path] = {}

        for name, frame in cohorts.items():
            point = weighted_original_metrics(frame)
            overall_rows.extend(
                {"cohort": name, "metric": metric, "estimate": point[metric]}
                for metric in PRIMARY_METRICS
            )
            bootstrap = stratified_psu_metric_bootstrap(
                frame,
                repetitions=int(bootstrap_config["repetitions"]),
                seed=int(bootstrap_config["seed"]),
            )
            intervals = percentile_interval_table(
                point,
                bootstrap,
                confidence_level=float(bootstrap_config["confidence_level"]),
                minimum_valid_replicates=int(
                    bootstrap_config["minimum_valid_replicates"]
                ),
            )
            width_targets = {
                "roc_auc": 0.10,
                "calibration_slope": 0.40,
                "observed_expected_ratio": 0.20,
            }
            intervals["full_ci_width"] = intervals["ci_upper"] - intervals["ci_lower"]
            intervals["target_full_ci_width"] = intervals["metric"].map(width_targets)
            has_target = intervals["target_full_ci_width"].notna()
            passes_width = pd.Series(pd.NA, index=intervals.index, dtype="boolean")
            passes_width.loc[has_target] = (
                intervals.loc[has_target, "full_ci_width"]
                <= intervals.loc[has_target, "target_full_ci_width"]
            ).to_numpy()
            intervals["passes_width_target"] = passes_width
            intervals.insert(0, "cohort", name)
            interval_tables.append(intervals)

            boundaries = calibration_bin_boundaries(
                frame["predicted_risk"].to_numpy(dtype=np.float64),
                frame["analysis_weight"].to_numpy(dtype=np.float64),
            )
            calibration = bootstrap_calibration_bands(
                frame,
                boundaries,
                repetitions=int(bootstrap_config["repetitions"]),
                seed=int(bootstrap_config["seed"]),
                confidence_level=float(bootstrap_config["confidence_level"]),
                minimum_valid_replicates=int(
                    bootstrap_config["minimum_valid_replicates"]
                ),
            )
            calibration.insert(0, "cohort", name)
            calibration_tables.append(calibration)
            figure_path = figure_dir / f"{name}_calibration.png"
            save_calibration_plot(
                calibration,
                figure_path,
                cohort=name,
            )
            figure_paths[f"calibration_figure_{name}"] = figure_path

            eligibility = subgroup_eligibility_table(frame)
            subgroup_point = subgroup_point_metrics(frame, eligibility)
            subgroup_bootstrap = stratified_psu_subgroup_bootstrap(
                frame,
                eligibility,
                repetitions=int(bootstrap_config["repetitions"]),
                seed=int(bootstrap_config["seed"]),
            )
            subgroup = add_subgroup_percentile_intervals(
                subgroup_point,
                subgroup_bootstrap,
                confidence_level=float(bootstrap_config["confidence_level"]),
                minimum_valid_replicates=int(
                    bootstrap_config["minimum_valid_replicates"]
                ),
            )
            subgroup.insert(0, "cohort", name)
            subgroup_tables.append(subgroup)

            settings = config["cohorts"][name]
            period = period_stability_table(
                frame,
                period_column=settings["period_column"],
                expected_periods=settings["expected_periods"],
            )
            period.insert(0, "cohort", name)
            period_tables.append(period)

            recalibration = recalibration_sensitivity_table(frame)
            recalibration.insert(0, "cohort", name)
            recalibration_tables.append(recalibration)

        overall_table = pd.DataFrame(overall_rows)
        interval_table = pd.concat(interval_tables, ignore_index=True)
        calibration_table = pd.concat(calibration_tables, ignore_index=True)
        subgroup_table = pd.concat(subgroup_tables, ignore_index=True)
        period_table = pd.concat(period_tables, ignore_index=True)
        recalibration_table = pd.concat(recalibration_tables, ignore_index=True)
        design_risk = _run_design_risk_precision(
            cohorts,
            root=root,
            output_path=output_paths["design_risk_precision"],
        )
        for key, table in (
            ("overall_metrics", overall_table),
            ("confidence_intervals", interval_table),
            ("calibration_bins", calibration_table),
            ("subgroup_metrics", subgroup_table),
            ("period_stability", period_table),
            ("recalibration_sensitivity", recalibration_table),
        ):
            output_paths[key].parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(output_paths[key], index=False)
        _write_validation_report(
            output_paths["report"],
            overall=overall_table,
            intervals=interval_table,
            design_risk=design_risk,
        )
        execution_manifest: dict[str, Any] = {
            "schema_version": 1,
            "stage": "stage6_authorized_unseal_validation",
            "status": "COMPLETE_UNDERPOWERED_EXPLORATORY",
            "completed_at": datetime.now(UTC).isoformat(),
            "outcomes_unsealed": True,
            "heldout_outcomes_loaded": True,
            "heldout_outcomes_joined": True,
            "heldout_outcomes_analyzed": True,
            "interpretation": "underpowered_exploratory",
            "model_modified": False,
            "calibrator_modified": False,
            "cutoff_modified_or_reselected": False,
            "cutoff": FROZEN_CUTOFF,
            "plan_commit": receipt["plan_commit"],
            "authorization": receipt["authorization"],
            "environment": {"python": platform.python_version()},
            "outputs": {
                key: portable_binding(path.relative_to(root), root=root)
                for key, path in {**output_paths, **figure_paths}.items()
            },
            "joined_individual_data_persisted": False,
        }
        execution_path = artifact_dir / "manifest.json"
        execution_path.write_text(
            json.dumps(execution_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        receipt.update(
            {
                "status": "ONE_TIME_UNSEAL_VALIDATION_COMPLETE",
                "completed_at": datetime.now(UTC).isoformat(),
                "execution_manifest": portable_binding(
                    execution_path.relative_to(root), root=root
                ),
            }
        )
        receipt_path.write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return execution_manifest
    except Exception:
        receipt.update(
            {
                "status": "UNSEAL_OCCURRED_VALIDATION_FAILED_REVIEW_REQUIRED",
                "failed_at": datetime.now(UTC).isoformat(),
            }
        )
        receipt_path.write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        raise


__all__ = [
    "BOOTSTRAP_REPETITIONS",
    "BOOTSTRAP_SEED",
    "FROZEN_CUTOFF",
    "OUTCOME_LOAD_COLUMNS",
    "PRIMARY_METRICS",
    "add_subgroup_percentile_intervals",
    "bootstrap_calibration_bands",
    "calibration_bin_boundaries",
    "calibration_bin_table",
    "create_unseal_plan_manifest",
    "join_outcomes_one_to_one",
    "load_unseal_validation_config",
    "percentile_interval_table",
    "period_stability_table",
    "recalibration_in_the_large",
    "recalibration_sensitivity_table",
    "run_authorized_unseal_validation",
    "stratified_psu_metric_bootstrap",
    "stratified_psu_subgroup_bootstrap",
    "subgroup_eligibility_table",
    "subgroup_point_metrics",
    "validate_unseal_plan",
    "weighted_original_metrics",
]


def percentile_interval_table(
    point: dict[str, float],
    bootstrap: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = PRIMARY_METRICS,
    confidence_level: float = 0.95,
    minimum_valid_replicates: int = 1900,
) -> pd.DataFrame:
    alpha = 1.0 - confidence_level
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        values = bootstrap[metric].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        valid = len(finite)
        eligible = valid >= minimum_valid_replicates
        rows.append(
            {
                "metric": metric,
                "estimate": point[metric],
                "ci_lower": (
                    float(np.quantile(finite, alpha / 2.0)) if eligible else np.nan
                ),
                "ci_upper": (
                    float(np.quantile(finite, 1.0 - alpha / 2.0)) if eligible else np.nan
                ),
                "confidence_level": confidence_level,
                "valid_replicates": valid,
                "status": "complete" if eligible else "insufficient_valid_replicates",
            }
        )
    return pd.DataFrame(rows)


def subgroup_eligibility_table(
    frame: pd.DataFrame,
    *,
    minimum_events: int = 20,
    minimum_non_events: int = 20,
) -> pd.DataFrame:
    working = frame.copy()
    age = pd.to_numeric(working["age"], errors="coerce")
    working["age_group"] = pd.cut(
        age,
        bins=[39, 54, 64, 79],
        labels=["40-54", "55-64", "65-79"],
        include_lowest=True,
        right=True,
    ).astype("string")
    rows: list[dict[str, Any]] = []
    for dimension in ("sex", "age_group", "race_ethnicity"):
        values = working[dimension].astype("string")
        missing = int(values.isna().sum())
        for category in sorted(values.dropna().unique().tolist()):
            selected = values.eq(category).fillna(False).to_numpy()
            target = working.loc[selected, OUTCOME_COLUMN].to_numpy(dtype=np.int8)
            events = int(target.sum())
            non_events = int(len(target) - events)
            eligible = events >= minimum_events and non_events >= minimum_non_events
            rows.append(
                {
                    "dimension": dimension,
                    "category": str(category),
                    "n": int(selected.sum()),
                    "events": events,
                    "non_events": non_events,
                    "missing_dimension": missing,
                    "eligible_for_performance": eligible,
                    "status": "eligible" if eligible else "counts_only_insufficient_events",
                }
            )
    return pd.DataFrame(rows)


def subgroup_point_metrics(
    frame: pd.DataFrame,
    eligibility: pd.DataFrame,
    *,
    cutoff: float = FROZEN_CUTOFF,
) -> pd.DataFrame:
    working = frame.copy()
    working["age_group"] = pd.cut(
        pd.to_numeric(working["age"], errors="coerce"),
        bins=[39, 54, 64, 79],
        labels=["40-54", "55-64", "65-79"],
        include_lowest=True,
        right=True,
    ).astype("string")
    rows: list[dict[str, Any]] = []
    for entry in eligibility.itertuples(index=False):
        base = {
            "dimension": entry.dimension,
            "category": entry.category,
            "n": entry.n,
            "events": entry.events,
            "non_events": entry.non_events,
            "eligible_for_performance": entry.eligible_for_performance,
            "status": entry.status,
        }
        if not entry.eligible_for_performance:
            rows.append({**base, **{metric: np.nan for metric in SUBGROUP_METRICS}})
            continue
        selected = (
            working[entry.dimension]
            .astype("string")
            .eq(str(entry.category))
            .fillna(False)
        )
        metrics = weighted_original_metrics(working.loc[selected], cutoff=cutoff)
        rows.append({**base, **{metric: metrics[metric] for metric in SUBGROUP_METRICS}})
    return pd.DataFrame(rows)
