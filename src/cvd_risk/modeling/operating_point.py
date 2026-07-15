"""Pre-lock audit of a fixed secondary screening operating point."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from cvd_risk.provenance import sha256_file

REQUIRED_BOOTSTRAP = {
    "method": "stratified_psu_with_replacement_within_strata",
    "repetitions": 2000,
    "seed": 20260714,
    "confidence_level": 0.95,
    "interval_method": "percentile",
    "reselect_cutoff_in_each_replicate": False,
}

CONFUSION_KEYS = ("tp", "fn", "fp", "tn")
CI_METRICS = (
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "flagged_fraction",
    "prevalence",
    "tp_per_10000",
    "fn_per_10000",
    "fp_per_10000",
    "tn_per_10000",
    "net_benefit_model",
    "net_benefit_treat_all",
    "net_benefit_treat_none",
    "net_benefit_vs_treat_all",
    "net_benefit_vs_treat_none",
)


def _scalar_float(value: Any) -> float:
    return float(value)


def load_operating_point_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Operating-point config schema_version must be 1")
    if payload.get("stage") != "stage5b_prelock_operating_point_audit":
        raise ValueError("Unexpected operating-point stage")
    policy = payload.get("secondary_screening_operating_point", {})
    required = {
        "approved_by_user_on": "2026-07-15",
        "target_survey_weighted_sensitivity": 0.80,
        "selection_rule": (
            "maximum_development_oof_cutoff_with_sensitivity_at_least_target"
        ),
        "classification_rule": "predicted_risk_greater_than_or_equal_to_cutoff",
        "approved_reference_cutoff": 0.007469,
        "approved_reference_rounding_decimals": 6,
        "role": "secondary_screening_operating_point",
        "primary_model_output": "continuous_five_year_risk",
        "interpretation": (
            "not_a_diagnosis_or_clinically_endorsed_high_risk_boundary"
        ),
    }
    if not isinstance(policy, dict) or any(
        policy.get(key) != value for key, value in required.items()
    ):
        raise ValueError("Operating-point policy differs from the user-approved policy")
    if payload.get("cluster_bootstrap") != REQUIRED_BOOTSTRAP:
        raise ValueError("Cluster-bootstrap policy differs from the pre-lock policy")
    return payload


def select_maximum_cutoff(
    prediction: np.ndarray,
    outcome: np.ndarray,
    sample_weight: np.ndarray,
    *,
    target_sensitivity: float,
) -> dict[str, float]:
    risk = np.asarray(prediction, dtype=np.float64)
    y = np.asarray(outcome, dtype=np.int8)
    weight = np.asarray(sample_weight, dtype=np.float64)
    _validate_arrays(risk, y, weight)
    if not 0.0 < target_sensitivity <= 1.0:
        raise ValueError("Target sensitivity must be in (0, 1]")

    order = np.argsort(-risk, kind="stable")
    sorted_risk = risk[order]
    positive_weight = weight[order] * y[order]
    group_ends = np.r_[
        np.flatnonzero(sorted_risk[:-1] != sorted_risk[1:]), len(sorted_risk) - 1
    ]
    cumulative_sensitivity = np.cumsum(positive_weight)[group_ends] / positive_weight.sum()
    crossing = np.flatnonzero(cumulative_sensitivity >= target_sensitivity)
    if len(crossing) == 0:
        raise RuntimeError("No cutoff achieves the requested sensitivity")
    index = int(crossing[0])
    cutoff = float(sorted_risk[group_ends[index]])
    previous_sensitivity = (
        float(cumulative_sensitivity[index - 1]) if index > 0 else 0.0
    )
    next_higher_cutoff = (
        float(sorted_risk[group_ends[index - 1]]) if index > 0 else float("nan")
    )
    return {
        "cutoff": cutoff,
        "achieved_sensitivity": float(cumulative_sensitivity[index]),
        "next_higher_cutoff": next_higher_cutoff,
        "next_higher_cutoff_sensitivity": previous_sensitivity,
    }


def _validate_arrays(risk: np.ndarray, y: np.ndarray, weight: np.ndarray) -> None:
    if not (len(risk) == len(y) == len(weight)) or len(risk) == 0:
        raise ValueError("Risk, outcome, and weight must have equal non-zero length")
    if not np.isfinite(risk).all() or not np.isfinite(weight).all():
        raise ValueError("Risk and weight must be finite")
    if (risk < 0).any() or (risk > 1).any() or (weight <= 0).any():
        raise ValueError("Risk must be in [0, 1] and weights must be positive")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Outcome must be binary")


def _metrics_from_totals(
    *,
    tp: np.ndarray | float,
    fn: np.ndarray | float,
    fp: np.ndarray | float,
    tn: np.ndarray | float,
    cutoff: float,
) -> dict[str, np.ndarray | float]:
    total = tp + fn + fp + tn
    event = tp + fn
    non_event = fp + tn
    flagged = tp + fp
    not_flagged = fn + tn
    odds = cutoff / (1.0 - cutoff)
    net_benefit_model = tp / total - fp / total * odds
    net_benefit_treat_all = event / total - non_event / total * odds
    zeros = np.zeros_like(np.asarray(total, dtype=np.float64))
    return {
        "sensitivity": tp / event,
        "specificity": tn / non_event,
        "ppv": tp / flagged,
        "npv": tn / not_flagged,
        "flagged_fraction": flagged / total,
        "prevalence": event / total,
        "tp_per_10000": tp / total * 10_000.0,
        "fn_per_10000": fn / total * 10_000.0,
        "fp_per_10000": fp / total * 10_000.0,
        "tn_per_10000": tn / total * 10_000.0,
        "net_benefit_model": net_benefit_model,
        "net_benefit_treat_all": net_benefit_treat_all,
        "net_benefit_treat_none": zeros,
        "net_benefit_vs_treat_all": net_benefit_model - net_benefit_treat_all,
        "net_benefit_vs_treat_none": net_benefit_model,
    }


def weighted_operating_metrics(
    frame: pd.DataFrame,
    *,
    cutoff: float,
    prediction_column: str = "prediction_sigmoid",
) -> dict[str, float]:
    risk = frame[prediction_column].to_numpy(dtype=np.float64)
    y = frame["cvd_death_5y"].to_numpy(dtype=np.int8)
    weight = frame["analysis_weight"].to_numpy(dtype=np.float64)
    _validate_arrays(risk, y, weight)
    screen_positive = risk >= cutoff
    totals = {
        "tp": float(weight[screen_positive & (y == 1)].sum()),
        "fn": float(weight[(~screen_positive) & (y == 1)].sum()),
        "fp": float(weight[screen_positive & (y == 0)].sum()),
        "tn": float(weight[(~screen_positive) & (y == 0)].sum()),
    }
    derived = _metrics_from_totals(**totals, cutoff=cutoff)
    return {
        "n": float(len(frame)),
        "events": float(y.sum()),
        "cutoff": cutoff,
        **totals,
        **{key: float(np.asarray(value)) for key, value in derived.items()},
    }


def stratified_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    cutoff: float,
    repetitions: int,
    seed: int,
    prediction_column: str = "prediction_sigmoid",
) -> pd.DataFrame:
    if repetitions < 2:
        raise ValueError("At least two bootstrap repetitions are required")
    working = frame[
        ["strata", "psu", "analysis_weight", "cvd_death_5y", prediction_column]
    ].copy()
    risk = working[prediction_column].to_numpy(dtype=np.float64)
    y = working["cvd_death_5y"].to_numpy(dtype=np.int8)
    weight = working["analysis_weight"].to_numpy(dtype=np.float64)
    _validate_arrays(risk, y, weight)
    positive = risk >= cutoff
    working["tp"] = weight * (positive & (y == 1))
    working["fn"] = weight * ((~positive) & (y == 1))
    working["fp"] = weight * (positive & (y == 0))
    working["tn"] = weight * ((~positive) & (y == 0))
    clusters = (
        working.groupby(["strata", "psu"], sort=True, observed=True)[
            list(CONFUSION_KEYS)
        ]
        .sum()
        .reset_index()
    )
    cluster_counts = clusters.groupby("strata", observed=True)["psu"].nunique()
    if (cluster_counts < 2).any():
        raise ValueError("Cluster bootstrap requires at least two PSUs in every stratum")

    rng = np.random.default_rng(seed)
    replicate_totals = np.zeros((repetitions, len(CONFUSION_KEYS)), dtype=np.float64)
    for _, block in clusters.groupby("strata", sort=True, observed=True):
        values = block[list(CONFUSION_KEYS)].to_numpy(dtype=np.float64)
        size = len(values)
        draws = rng.integers(0, size, size=(repetitions, size))
        replicate_totals += values[draws].sum(axis=1)
    derived = _metrics_from_totals(
        tp=replicate_totals[:, 0],
        fn=replicate_totals[:, 1],
        fp=replicate_totals[:, 2],
        tn=replicate_totals[:, 3],
        cutoff=cutoff,
    )
    output = pd.DataFrame(
        {key: np.asarray(value, dtype=np.float64) for key, value in derived.items()}
    )
    output.insert(0, "replicate", np.arange(1, repetitions + 1))
    return output


def confidence_interval_table(
    point: dict[str, float], bootstrap: pd.DataFrame, *, confidence_level: float
) -> pd.DataFrame:
    alpha = 1.0 - confidence_level
    rows: list[dict[str, float | str]] = []
    for metric in CI_METRICS:
        values = bootstrap[metric].to_numpy(dtype=np.float64)
        rows.append(
            {
                "metric": metric,
                "estimate": point[metric],
                "ci_lower": float(np.quantile(values, alpha / 2.0)),
                "ci_upper": float(np.quantile(values, 1.0 - alpha / 2.0)),
                "confidence_level": confidence_level,
            }
        )
    return pd.DataFrame(rows)


def compare_uncalibrated_and_sigmoid(
    frame: pd.DataFrame,
    comparison_table: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
    confidence_level: float,
    numerical_tie_tolerance: float,
) -> pd.DataFrame:
    rows = comparison_table.loc[
        comparison_table["model"].eq("LogisticRegression")
        & comparison_table["weighting"].eq("survey_weighted")
        & comparison_table["calibration"].isin(["none", "sigmoid"])
    ].set_index("calibration")
    if set(rows.index) != {"none", "sigmoid"}:
        raise ValueError("Paired LogisticRegression calibration rows are absent")
    working = frame[
        [
            "strata",
            "psu",
            "analysis_weight",
            "cvd_death_5y",
            "prediction_uncalibrated",
            "prediction_sigmoid",
        ]
    ].copy()
    y = working["cvd_death_5y"].to_numpy(dtype=np.float64)
    weight = working["analysis_weight"].to_numpy(dtype=np.float64)
    uncalibrated = np.clip(
        working["prediction_uncalibrated"].to_numpy(dtype=np.float64), 1e-15, 1 - 1e-15
    )
    sigmoid = np.clip(
        working["prediction_sigmoid"].to_numpy(dtype=np.float64), 1e-15, 1 - 1e-15
    )
    working["weight"] = weight
    working["brier_delta"] = weight * (
        (sigmoid - y) ** 2 - (uncalibrated - y) ** 2
    )
    uncalibrated_log_loss = -y * np.log(uncalibrated) - (1 - y) * np.log(
        1 - uncalibrated
    )
    sigmoid_log_loss = -y * np.log(sigmoid) - (1 - y) * np.log(1 - sigmoid)
    working["log_loss_delta"] = weight * (
        sigmoid_log_loss - uncalibrated_log_loss
    )
    clusters = (
        working.groupby(["strata", "psu"], sort=True, observed=True)[
            ["weight", "brier_delta", "log_loss_delta"]
        ]
        .sum()
        .reset_index()
    )
    counts = clusters.groupby("strata", observed=True)["psu"].nunique()
    if (counts < 2).any():
        raise ValueError("Paired calibration bootstrap requires two PSUs per stratum")
    rng = np.random.default_rng(seed)
    totals = np.zeros((repetitions, 3), dtype=np.float64)
    for _, block in clusters.groupby("strata", sort=True, observed=True):
        values = block[["weight", "brier_delta", "log_loss_delta"]].to_numpy(
            dtype=np.float64
        )
        size = len(values)
        draws = rng.integers(0, size, size=(repetitions, size))
        totals += values[draws].sum(axis=1)
    deltas = {
        "brier_score": totals[:, 1] / totals[:, 0],
        "log_loss": totals[:, 2] / totals[:, 0],
    }
    alpha = 1.0 - confidence_level
    output: list[dict[str, Any]] = []
    for metric in ("brier_score", "log_loss"):
        before = _scalar_float(rows.loc["none", metric])
        after = _scalar_float(rows.loc["sigmoid", metric])
        delta = after - before
        ci = np.quantile(deltas[metric], [alpha / 2.0, 1.0 - alpha / 2.0])
        if metric == "brier_score":
            passed = delta <= 0.0 and float(ci[1]) < 0.0
            decision = "improved_paired_cluster_bootstrap_ci_below_zero"
        else:
            improved = delta < -numerical_tie_tolerance and float(ci[1]) < 0.0
            numerical_tie = (
                abs(delta) <= numerical_tie_tolerance
                and float(ci[0]) <= 0 <= float(ci[1])
            )
            passed = improved or numerical_tie
            decision = (
                "improved_paired_cluster_bootstrap_ci_below_zero"
                if improved
                else "numerical_tie_no_bootstrap_evidence_of_worsening"
            )
        output.append(
            {
                "metric": metric,
                "uncalibrated": before,
                "sigmoid": after,
                "delta_sigmoid_minus_uncalibrated": delta,
                "delta_ci_lower": float(ci[0]),
                "delta_ci_upper": float(ci[1]),
                "decision": decision,
                "passed": passed,
            }
        )
    calibration_metrics = (
        ("absolute_calibration_intercept", "calibration_intercept", 0.0),
        ("absolute_calibration_slope_error", "calibration_slope", 1.0),
        ("absolute_observed_expected_error", "observed_expected_ratio", 1.0),
    )
    for label, column, target in calibration_metrics:
        before = abs(_scalar_float(rows.loc["none", column]) - target)
        after = abs(_scalar_float(rows.loc["sigmoid", column]) - target)
        output.append(
            {
                "metric": label,
                "uncalibrated": before,
                "sigmoid": after,
                "delta_sigmoid_minus_uncalibrated": after - before,
                "delta_ci_lower": np.nan,
                "delta_ci_upper": np.nan,
                "decision": "absolute_calibration_error_improved",
                "passed": after < before,
            }
        )
    result = pd.DataFrame(output)
    if not result["passed"].all():
        failed = result.loc[~result["passed"], "metric"].tolist()
        raise RuntimeError(f"Sigmoid calibration comparison blocker failed: {failed}")
    return result


def stability_table(
    frame: pd.DataFrame,
    *,
    group_column: str,
    cutoff: float,
    prediction_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, subset in frame.groupby(group_column, sort=True, observed=True):
        row: dict[str, Any] = {
            group_column: group,
            **weighted_operating_metrics(
                subset, cutoff=cutoff, prediction_column=prediction_column
            ),
        }
        rows.append(row)
    columns = [group_column] + [key for key in rows[0] if key != group_column]
    return pd.DataFrame(rows)[columns]


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


def verify_post_chf_lineage(
    *,
    config: dict[str, Any],
    selected_oof: pd.DataFrame,
    stage4: dict[str, Any],
    stage5: dict[str, Any],
) -> pd.DataFrame:
    experiment_path = Path(config["experiment_config"])
    discrepancy_path = Path(config["protocol_discrepancies"])
    stop_gate_path = Path(config["stop_gate"])
    cohort_path = Path(config["cohort"])
    experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    discrepancy_payload = yaml.safe_load(discrepancy_path.read_text(encoding="utf-8"))
    discrepancies = {
        entry["id"]: entry for entry in discrepancy_payload["discrepancies"]
    }
    chf = discrepancies["nhis_congestive_heart_failure_availability"]
    stop_gate = json.loads(stop_gate_path.read_text(encoding="utf-8"))
    primary_conditions = experiment["cohort"]["prevalent_cvd_conditions"]
    sensitivity_conditions = experiment["cohort"][
        "sensitivity_prevalent_cvd_conditions"
    ]
    expected_primary = [
        "coronary_heart_disease",
        "angina_pectoris",
        "myocardial_infarction",
        "stroke",
    ]
    cohort_sha = sha256_file(cohort_path)
    oof_path = Path(stage4["outputs"]["oof_predictions"]["path"])
    oof_sha = sha256_file(oof_path)
    stage3_commit = _git("log", "-1", "--format=%H", "--", "reports/stop_gate.json")
    stage4_commit = _git(
        "log", "-1", "--format=%H", "--", "artifacts/stage4_baselines/manifest.json"
    )
    base_commit = stage4["git"]["base_commit"]
    amendment_blob = _git("show", f"{base_commit}:configs/protocol_discrepancies.yaml")
    ancestor_status = subprocess.run(
        ["git", "merge-base", "--is-ancestor", stage3_commit, stage4_commit],
        check=False,
    ).returncode

    cohort = pd.read_parquet(cohort_path)
    required_chf_columns = {
        "congestive_heart_failure",
        "baseline_cvd_status_with_chf",
        "chf_sensitivity_available",
        "cohort_chf_sensitivity_eligible",
    }
    development = cohort.loc[cohort["split"].eq("development")].copy()
    cohort_keys = development[
        [
            "participant_id",
            "survey_year",
            "strata",
            "psu",
            "analysis_weight",
            "cvd_death_5y",
        ]
    ].sort_values("participant_id").reset_index(drop=True)
    oof_keys = selected_oof[
        [
            "participant_id",
            "survey_year",
            "strata",
            "psu",
            "analysis_weight",
            "cvd_death_5y",
        ]
    ].sort_values("participant_id").reset_index(drop=True)
    keys_identical = cohort_keys.equals(oof_keys)

    checks = [
        (
            "chf_amendment_resolved",
            chf["status"] == "resolved" and "2026-07-14" in chf["resolution"],
            chf["resolution"],
        ),
        (
            "primary_and_sensitivity_rules_locked",
            primary_conditions == expected_primary
            and sensitivity_conditions == expected_primary + ["congestive_heart_failure"]
            and experiment["cohort"]["nhis_chf_sensitivity_years"] == [1999, 2002],
            "four-condition primary exclusion; CHF sensitivity only in 1999/2002",
        ),
        (
            "cohort_contains_chf_amendment_fields",
            required_chf_columns.issubset(cohort.columns),
            ", ".join(sorted(required_chf_columns)),
        ),
        (
            "cohort_hash_bound_after_amendment",
            cohort_sha
            == stop_gate["inputs"]["nhis_cohort"]["sha256"]
            == stage4["input_cohort"]["sha256"]
            == stage5["input_cohort"]["sha256"],
            cohort_sha,
        ),
        (
            "oof_hash_matches_stage4_manifest",
            oof_sha == stage4["outputs"]["oof_predictions"]["sha256"],
            oof_sha,
        ),
        (
            "oof_rows_match_post_amendment_development_cohort",
            keys_identical and len(oof_keys) == int(stage4["n"]),
            f"cohort_n={len(cohort_keys)}; oof_n={len(oof_keys)}",
        ),
        (
            "stage4_based_on_chf_amendment_commit",
            base_commit == stage3_commit
            and "nhis_congestive_heart_failure_availability" in amendment_blob
            and "status: resolved" in amendment_blob,
            f"stage3/base={stage3_commit}; stage4_commit={stage4_commit}",
        ),
        (
            "stage4_commit_follows_stage3_amendment",
            ancestor_status == 0 and stage3_commit != stage4_commit,
            f"{stage3_commit} -> {stage4_commit}",
        ),
    ]
    result = pd.DataFrame(checks, columns=["check", "passed", "evidence"])
    if not result["passed"].all():
        failed = result.loc[~result["passed"], "check"].tolist()
        raise RuntimeError(f"Post-CHF lineage confirmation failed: {failed}")
    return result


def _verify_manifest(path: Path, *, expected_stage: str, expected_status: str) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("stage") != expected_stage or manifest.get("status") != expected_status:
        raise ValueError(f"Unexpected manifest state: {path}")
    if manifest.get("heldout_outcomes_accessed") is not False:
        raise ValueError(f"Held-out outcomes were not sealed in {path}")
    return manifest


def _binding(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _write_report(
    path: Path,
    *,
    cutoff_result: dict[str, float],
    metrics: pd.DataFrame,
    confusion: pd.DataFrame,
    fold: pd.DataFrame,
    year: pd.DataFrame,
    dca: pd.DataFrame,
    calibration_comparison: pd.DataFrame,
    lineage: pd.DataFrame,
    manifest_path: Path,
) -> None:
    metric_index = metrics.set_index("metric")
    lines = [
        "# Stage 5b: pre-lock secondary screening operating point",
        "",
        "## Interpretation",
        "",
        "The model's primary output remains continuous five-year CVD mortality risk. "
        "This cutoff is a secondary screening operating point; it is not a diagnosis "
        "or a clinically endorsed high-risk boundary.",
        "",
        "## Fixed development OOF cutoff",
        "",
        "- Selection rule: maximum cutoff with survey-weighted sensitivity >= 0.80.",
        f"- Exact cutoff: `{cutoff_result['cutoff']:.12f}`.",
        f"- Survey-weighted sensitivity: {cutoff_result['achieved_sensitivity']:.6f}.",
        f"- Next higher cutoff: `{cutoff_result['next_higher_cutoff']:.12f}` "
        f"(sensitivity {cutoff_result['next_higher_cutoff_sensitivity']:.6f}, below target).",
        "- Cutoff is fixed in every bootstrap, fold, year, and DCA calculation.",
        "",
        "## Overall metrics with stratified PSU cluster-bootstrap 95% CI",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Survey-weighted confusion matrix per 10,000",
        "",
        confusion.pivot(
            index="actual_outcome", columns="screen_class", values="estimate_per_10000"
        ).to_markdown(),
        "",
        "Cell-specific intervals are available in the machine-readable confusion table.",
        "",
        "## Decision curve at the fixed cutoff",
        "",
        dca.to_markdown(index=False),
        "",
        "## Uncalibrated LogisticRegression versus sigmoid",
        "",
        calibration_comparison.to_markdown(index=False),
        "",
        "The comparison already existed in the b92f921 source tree. Sigmoid improves "
        "weighted Brier with a paired cluster-bootstrap interval below zero. The log-loss "
        "difference is 3.67e-08 and its paired interval spans zero, so it is treated as a "
        "numerical tie rather than claimed as an improvement. Calibration intercept, slope "
        "error, and O/E error all improve.",
        "",
        "## Fold stability",
        "",
        fold[
            [
                "fold",
                "n",
                "events",
                "sensitivity",
                "specificity",
                "ppv",
                "npv",
                "flagged_fraction",
                "net_benefit_model",
            ]
        ].to_markdown(index=False),
        "",
        "## Year stability",
        "",
        year[
            [
                "survey_year",
                "n",
                "events",
                "sensitivity",
                "specificity",
                "ppv",
                "npv",
                "flagged_fraction",
                "net_benefit_model",
            ]
        ].to_markdown(index=False),
        "",
        "Fold/year estimates are descriptive stability checks. The cutoff was not "
        "reselected within any fold or year.",
        "",
        "## CHF amendment lineage",
        "",
        lineage.to_markdown(index=False),
        "",
        "All lineage checks passed. The cohort hash used by Stage 3, Stage 4 and "
        "Stage 5 is identical, the cohort contains the CHF sensitivity fields, and "
        "the selected OOF rows exactly match the post-amendment development cohort.",
        "",
        f"PPV: {metric_index.loc['ppv', 'estimate']:.6f}; "
        f"NPV: {metric_index.loc['npv', 'estimate']:.6f}.",
        "",
        "No held-out outcome was loaded or used. No model lock was created by this audit.",
        "",
        f"Machine-readable provenance: `{manifest_path.as_posix()}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_operating_point_audit(
    *, config_path: str | Path, force: bool = False
) -> dict[str, Any]:
    path = Path(config_path)
    config = load_operating_point_config(path)
    stage4_path = Path(config["stage4_manifest"])
    stage5_path = Path(config["stage5_manifest"])
    stage5a_path = Path(config["stage5a_manifest"])
    stage4 = _verify_manifest(
        stage4_path,
        expected_stage="stage4_baselines",
        expected_status="BASELINES_FITTED_NOT_MODEL_LOCKED",
    )
    stage5 = _verify_manifest(
        stage5_path,
        expected_stage="stage5_ml_candidates",
        expected_status="ML_CANDIDATES_FITTED_NOT_MODEL_LOCKED",
    )
    _verify_manifest(
        stage5a_path,
        expected_stage="stage5a_provisional_external_validation_planning",
        expected_status="PROVISIONAL_PLANNING_COMPLETE_NOT_LOCKED",
    )
    selection_path = Path(config["selected_model_config"])
    selection = yaml.safe_load(selection_path.read_text(encoding="utf-8"))
    primary = selection["primary"]
    if primary["model"] != "LogisticRegression" or primary["calibration"] != "sigmoid":
        raise ValueError("Unexpected selected model")
    oof_path = Path(primary["oof_predictions"])
    if sha256_file(oof_path) != stage4["outputs"]["oof_predictions"]["sha256"]:
        raise ValueError("OOF predictions differ from Stage 4 manifest")
    oof = pd.read_parquet(oof_path)
    selected = oof.loc[oof["model"].eq(primary["model"])].reset_index(drop=True)
    if not selected["survey_year"].between(1997, 2009, inclusive="both").all():
        raise ValueError("Operating-point audit contains held-out years")

    policy = config["secondary_screening_operating_point"]
    prediction_column = config["prediction_column"]
    cutoff_result = select_maximum_cutoff(
        selected[prediction_column].to_numpy(dtype=np.float64),
        selected["cvd_death_5y"].to_numpy(dtype=np.int8),
        selected["analysis_weight"].to_numpy(dtype=np.float64),
        target_sensitivity=float(policy["target_survey_weighted_sensitivity"]),
    )
    rounded = round(
        cutoff_result["cutoff"], int(policy["approved_reference_rounding_decimals"])
    )
    if rounded != float(policy["approved_reference_cutoff"]):
        raise RuntimeError(
            f"Derived cutoff {cutoff_result['cutoff']:.12f} does not match approval"
        )
    point = weighted_operating_metrics(
        selected, cutoff=cutoff_result["cutoff"], prediction_column=prediction_column
    )
    if point["sensitivity"] < float(policy["target_survey_weighted_sensitivity"]):
        raise RuntimeError("Derived cutoff does not achieve the approved sensitivity")

    comparison_config = config["calibration_comparison"]
    comparison_path = Path(comparison_config["evidence_table"])
    if sha256_file(comparison_path) != stage4["outputs"]["comparison_table"]["sha256"]:
        raise ValueError("Stage 4 calibration comparison differs from its manifest")
    evidence_blob = _git(
        "rev-parse",
        f"{comparison_config['evidence_commit']}:{comparison_path.as_posix()}",
    )
    if evidence_blob != _git("hash-object", comparison_path.as_posix()):
        raise ValueError("b92f921 does not contain the current calibration comparison")
    calibration_comparison = compare_uncalibrated_and_sigmoid(
        selected,
        pd.read_csv(comparison_path),
        repetitions=int(comparison_config["paired_cluster_bootstrap_repetitions"]),
        seed=int(comparison_config["seed"]),
        confidence_level=float(comparison_config["confidence_level"]),
        numerical_tie_tolerance=float(
            comparison_config["numerical_tie_absolute_tolerance"]
        ),
    )

    bootstrap_config = config["cluster_bootstrap"]
    bootstrap = stratified_cluster_bootstrap(
        selected,
        cutoff=cutoff_result["cutoff"],
        repetitions=int(bootstrap_config["repetitions"]),
        seed=int(bootstrap_config["seed"]),
        prediction_column=prediction_column,
    )
    metrics = confidence_interval_table(
        point,
        bootstrap,
        confidence_level=float(bootstrap_config["confidence_level"]),
    )
    metric_index = metrics.set_index("metric")
    confusion_rows = [
        ("CVD death", "screen positive", "tp_per_10000"),
        ("CVD death", "screen negative", "fn_per_10000"),
        ("no CVD death", "screen positive", "fp_per_10000"),
        ("no CVD death", "screen negative", "tn_per_10000"),
    ]
    confusion = pd.DataFrame(
        [
            {
                "actual_outcome": actual,
                "screen_class": screen,
                "estimate_per_10000": metric_index.loc[key, "estimate"],
                "ci_lower": metric_index.loc[key, "ci_lower"],
                "ci_upper": metric_index.loc[key, "ci_upper"],
                "confidence_level": metric_index.loc[key, "confidence_level"],
            }
            for actual, screen, key in confusion_rows
        ]
    )
    dca_rows = [
        ("model", "net_benefit_model"),
        ("treat_all", "net_benefit_treat_all"),
        ("treat_none", "net_benefit_treat_none"),
        ("model_minus_treat_all", "net_benefit_vs_treat_all"),
        ("model_minus_treat_none", "net_benefit_vs_treat_none"),
    ]
    dca = pd.DataFrame(
        [
            {
                "strategy": strategy,
                "cutoff": cutoff_result["cutoff"],
                "net_benefit": metric_index.loc[key, "estimate"],
                "ci_lower": metric_index.loc[key, "ci_lower"],
                "ci_upper": metric_index.loc[key, "ci_upper"],
                "confidence_level": metric_index.loc[key, "confidence_level"],
            }
            for strategy, key in dca_rows
        ]
    )
    fold = stability_table(
        selected,
        group_column="fold",
        cutoff=cutoff_result["cutoff"],
        prediction_column=prediction_column,
    )
    year = stability_table(
        selected,
        group_column="survey_year",
        cutoff=cutoff_result["cutoff"],
        prediction_column=prediction_column,
    )
    lineage = verify_post_chf_lineage(
        config=config, selected_oof=selected, stage4=stage4, stage5=stage5
    )

    outputs = config["outputs"]
    artifact_dir = Path(outputs["artifact_dir"])
    manifest_path = artifact_dir / "manifest.json"
    bootstrap_path = artifact_dir / "bootstrap_metrics.csv"
    output_paths = {
        key: Path(value)
        for key, value in outputs.items()
        if key not in {"artifact_dir", "report"}
    }
    report_path = Path(outputs["report"])
    known = [manifest_path, bootstrap_path, report_path, *output_paths.values()]
    if any(item.exists() for item in known) and not force:
        raise FileExistsError("Stage 5b outputs exist; pass --force to replace them")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for output_path in output_paths.values():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_paths["metrics"], index=False)
    confusion.to_csv(output_paths["confusion_per_10000"], index=False)
    fold.to_csv(output_paths["fold_stability"], index=False)
    year.to_csv(output_paths["year_stability"], index=False)
    dca.to_csv(output_paths["dca"], index=False)
    lineage.to_csv(output_paths["lineage"], index=False)
    calibration_comparison.to_csv(output_paths["calibration_comparison"], index=False)
    bootstrap.to_csv(bootstrap_path, index=False)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "stage": "stage5b_prelock_operating_point_audit",
        "status": "PRELOCK_OPERATING_POINT_AUDIT_COMPLETE",
        "heldout_outcomes_accessed": False,
        "external_validation_lock_created": False,
        "primary_output": "continuous_five_year_risk",
        "secondary_operating_point": {
            **policy,
            "exact_cutoff": cutoff_result["cutoff"],
            "achieved_survey_weighted_sensitivity": point["sensitivity"],
            "next_higher_cutoff_sensitivity": cutoff_result[
                "next_higher_cutoff_sensitivity"
            ],
            "next_higher_cutoff": cutoff_result["next_higher_cutoff"],
        },
        "cluster_bootstrap": bootstrap_config,
        "cohort_and_oof_post_chf_amendment": bool(lineage["passed"].all()),
        "sigmoid_comparison_blocker_closed": bool(
            calibration_comparison["passed"].all()
        ),
        "calibration_comparison_evidence": {
            "commit": comparison_config["evidence_commit"],
            "table_blob_sha1": evidence_blob,
            "seed": comparison_config["seed"],
            "paired_cluster_bootstrap_repetitions": comparison_config[
                "paired_cluster_bootstrap_repetitions"
            ],
            "numerical_tie_absolute_tolerance": comparison_config[
                "numerical_tie_absolute_tolerance"
            ],
        },
        "input_bindings": {
            "config": _binding(path),
            "selection": _binding(selection_path),
            "experiment": _binding(Path(config["experiment_config"])),
            "protocol_discrepancies": _binding(Path(config["protocol_discrepancies"])),
            "stop_gate": _binding(Path(config["stop_gate"])),
            "cohort": _binding(Path(config["cohort"])),
            "stage4_manifest": _binding(stage4_path),
            "stage5_manifest": _binding(stage5_path),
            "stage5a_manifest": _binding(stage5a_path),
            "oof_predictions": _binding(oof_path),
            "selected_model": _binding(Path(primary["artifact"])),
            "source": _binding(Path(__file__)),
            "stage4_calibration_comparison": _binding(comparison_path),
        },
        "point_metrics": point,
        "stability_ranges": {
            "fold_sensitivity_min": float(fold["sensitivity"].min()),
            "fold_sensitivity_max": float(fold["sensitivity"].max()),
            "year_sensitivity_min": float(year["sensitivity"].min()),
            "year_sensitivity_max": float(year["sensitivity"].max()),
        },
        "outputs": {
            **{key: _binding(value) for key, value in output_paths.items()},
            "bootstrap_metrics": _binding(bootstrap_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_report(
        report_path,
        cutoff_result=cutoff_result,
        metrics=metrics,
        confusion=confusion,
        fold=fold,
        year=year,
        dca=dca,
        calibration_comparison=calibration_comparison,
        lineage=lineage,
        manifest_path=manifest_path,
    )
    manifest["outputs"]["report"] = _binding(report_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


__all__ = [
    "confidence_interval_table",
    "compare_uncalibrated_and_sigmoid",
    "load_operating_point_config",
    "run_operating_point_audit",
    "select_maximum_cutoff",
    "stability_table",
    "stratified_cluster_bootstrap",
    "verify_post_chf_lineage",
    "weighted_operating_metrics",
]
