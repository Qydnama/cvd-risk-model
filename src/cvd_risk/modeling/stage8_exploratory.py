"""Underpowered exploratory paired OOF analysis for Stage 8 clinical_core."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.model_selection import StratifiedGroupKFold  # type: ignore[import-untyped]

from cvd_risk.modeling.baselines import normalize_training_weights
from cvd_risk.modeling.preprocessing import (
    RCS_QUANTILES,
    ROUTE_FEATURES,
    MinimalPreprocessor,
    restricted_cubic_spline_basis,
    weighted_quantile,
)
from cvd_risk.modeling.stage8_clarification import materialize_sbp_alias
from cvd_risk.modeling.unseal_validation import (
    _bootstrap_multiplicity_by_groups,
    weighted_original_metrics,
)
from cvd_risk.provenance import sha256_file

LABEL = "UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY"
EXPECTED_ADDENDUM_COMMIT = "024ad6af132affea3a88a60a56f0550e47cf2239"
EXPECTED_ADDENDUM_BLOB = "9a5b906631f12ad312e981344aab8f4455427f8c"
MINIMAL_FEATURES = ROUTE_FEATURES["minimal"]
CORE_FEATURES = (*MINIMAL_FEATURES, "sbp_mean", "total_cholesterol", "hdl_cholesterol")
METRICS = (
    "roc_auc",
    "average_precision",
    "brier_score",
    "log_loss",
    "calibration_intercept",
    "calibration_slope",
    "observed_expected_ratio",
)
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".yaml", ".yml"}


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def _binding(path: Path, root: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        normalized = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest = hashlib.sha256(normalized).hexdigest()
        mode = "normalized_text_lf"
    else:
        digest = hashlib.sha256(payload).hexdigest()
        mode = "raw_bytes"
    return {
        "repository_relative_path": path.resolve().relative_to(root.resolve()).as_posix(),
        "size_bytes": len(payload),
        "sha256": digest,
        "hash_mode": mode,
    }


def load_stage8_exploratory_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Stage 8 exploratory config schema_version must be 1")
    if payload.get("stage") != "stage8_underpowered_exploratory_development_only":
        raise ValueError("Unexpected Stage 8 exploratory stage")
    if payload.get("label") != LABEL:
        raise ValueError("Every Stage 8 exploratory output must use the required label")
    bindings = payload.get("bindings", {})
    if bindings.get("addendum_commit") != EXPECTED_ADDENDUM_COMMIT:
        raise ValueError("Exploratory analysis is not bound to the addendum commit")
    if bindings.get("addendum_blob_sha1") != EXPECTED_ADDENDUM_BLOB:
        raise ValueError("Exploratory analysis is not bound to the addendum blob")
    if payload["models"] != {
        "minimal": {"predictor_parameters": 15},
        "clinical_core": {
            "predictor_parameters": 20,
            "clinical_predictors": ["sbp_mean", "total_cholesterol", "hdl_cholesterol"],
        },
    }:
        raise ValueError("Stage 8 exploratory model specifications changed")
    if tuple(payload["restricted_cubic_splines"]["quantiles"]) != RCS_QUANTILES:
        raise ValueError("Stage 8 spline quantiles changed")
    if payload["cluster_bootstrap"] != {
        "method": "stratified_psu_with_replacement_within_strata_fixed_oof_predictions_no_refit",
        "repetitions": 2000,
        "seed": 20260714,
        "confidence_level": 0.95,
        "interval": "percentile",
        "minimum_valid_replicates": 1900,
    }:
        raise ValueError("Stage 8 paired bootstrap policy changed")
    return payload


class ClinicalCorePreprocessor:
    """Separate Stage 8 preprocessor with fold-fitted minimal imputation."""

    def __init__(self, spline_quantiles: tuple[float, ...] = RCS_QUANTILES) -> None:
        self.spline_quantiles = spline_quantiles

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        sample_weight: Sequence[float] | np.ndarray,
    ) -> ClinicalCorePreprocessor:
        missing = set(CORE_FEATURES).difference(frame.columns)
        if missing:
            raise ValueError(f"Clinical core frame is missing: {sorted(missing)}")
        weights = np.asarray(sample_weight, dtype=np.float64)
        if len(weights) != len(frame) or not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Clinical core training weights must be finite and positive")
        clinical = frame[["sbp_mean", "total_cholesterol", "hdl_cholesterol"]]
        if clinical.isna().any().any():
            raise ValueError(
                "Clinical fields cannot be imputed in the primary exploratory analysis"
            )
        if not all(pd.api.types.is_numeric_dtype(clinical[column]) for column in clinical):
            raise TypeError("Clinical fields must be numeric")

        self.minimal_preprocessor_ = MinimalPreprocessor(
            route="minimal", spline_quantiles=self.spline_quantiles
        )
        self.minimal_preprocessor_.fit(frame.loc[:, MINIMAL_FEATURES], sample_weight=weights)
        sbp = frame["sbp_mean"].to_numpy(dtype=np.float64)
        self.sbp_knots_ = weighted_quantile(sbp, self.spline_quantiles, weights)
        if len(np.unique(self.sbp_knots_)) != 4:
            raise ValueError("SBP does not support four distinct spline knots")
        sbp_basis = restricted_cubic_spline_basis(sbp, self.sbp_knots_)
        self.sbp_center_ = np.average(sbp_basis, axis=0, weights=weights)
        self.sbp_scale_ = np.sqrt(
            np.average((sbp_basis - self.sbp_center_) ** 2, axis=0, weights=weights)
        )
        if not np.isfinite(self.sbp_scale_).all() or (self.sbp_scale_ <= 0).any():
            raise ValueError("SBP spline basis has invalid variance")
        self.linear_center_: dict[str, float] = {}
        self.linear_scale_: dict[str, float] = {}
        for column in ("total_cholesterol", "hdl_cholesterol"):
            values = frame[column].to_numpy(dtype=np.float64)
            center = float(np.average(values, weights=weights))
            scale = float(np.sqrt(np.average((values - center) ** 2, weights=weights)))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"{column} has invalid weighted variance")
            self.linear_center_[column] = center
            self.linear_scale_[column] = scale
        self.feature_names_out_ = np.r_[
            self.minimal_preprocessor_.get_feature_names_out(),
            [
                "sbp_mean",
                "sbp_mean_rcs_1",
                "sbp_mean_rcs_2",
                "total_cholesterol",
                "hdl_cholesterol",
            ],
        ]
        if len(self.feature_names_out_) != 20:
            raise RuntimeError("Clinical core preprocessor must produce 20 parameters")
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "feature_names_out_"):
            raise RuntimeError("ClinicalCorePreprocessor is not fitted")
        clinical = frame[["sbp_mean", "total_cholesterol", "hdl_cholesterol"]]
        if clinical.isna().any().any():
            raise ValueError("Clinical fields cannot be imputed")
        minimal = self.minimal_preprocessor_.transform(frame.loc[:, MINIMAL_FEATURES])
        sbp_basis = restricted_cubic_spline_basis(
            frame["sbp_mean"].to_numpy(dtype=np.float64), self.sbp_knots_
        )
        columns = [minimal, (sbp_basis - self.sbp_center_) / self.sbp_scale_]
        for column in ("total_cholesterol", "hdl_cholesterol"):
            values = frame[column].to_numpy(dtype=np.float64)
            columns.append(
                ((values - self.linear_center_[column]) / self.linear_scale_[column]).reshape(-1, 1)
            )
        matrix = np.column_stack(columns).astype(np.float64, copy=False)
        if matrix.shape[1] != 20 or not np.isfinite(matrix).all():
            raise RuntimeError("Clinical core transformed matrix violates its specification")
        return matrix

    def get_feature_names_out(self) -> np.ndarray:
        if not hasattr(self, "feature_names_out_"):
            raise RuntimeError("ClinicalCorePreprocessor is not fitted")
        return self.feature_names_out_.copy()


def _new_logistic(config: Mapping[str, Any], seed: int) -> LogisticRegression:
    if config.get("penalty") is not None:
        raise ValueError("Stage 8 exploratory logistic regression must remain unpenalized")
    return LogisticRegression(
        C=np.inf,
        l1_ratio=0.0,
        solver=str(config["solver"]),
        max_iter=int(config["max_iter"]),
        tol=float(config["tolerance"]),
        random_state=seed,
    )


def _logit(probability: np.ndarray, clip: float) -> np.ndarray:
    bounded = np.clip(probability, clip, 1.0 - clip)
    return np.log(bounded / (1.0 - bounded))


def _fit_preprocessor(
    frame: pd.DataFrame,
    weights: np.ndarray,
    *,
    model: str,
) -> MinimalPreprocessor | ClinicalCorePreprocessor:
    if model == "minimal":
        processor = MinimalPreprocessor(route="minimal", spline_quantiles=RCS_QUANTILES)
        processor.fit(frame.loc[:, MINIMAL_FEATURES], sample_weight=weights)
        return processor
    if model == "clinical_core":
        return ClinicalCorePreprocessor().fit(frame.loc[:, CORE_FEATURES], sample_weight=weights)
    raise ValueError(f"Unknown exploratory model: {model}")


def _transform(
    processor: MinimalPreprocessor | ClinicalCorePreprocessor,
    frame: pd.DataFrame,
    *,
    model: str,
) -> np.ndarray:
    features = MINIMAL_FEATURES if model == "minimal" else CORE_FEATURES
    return processor.transform(frame.loc[:, features])


def _fit_base(
    frame: pd.DataFrame,
    outcome: np.ndarray,
    weights: np.ndarray,
    *,
    model: str,
    logistic_config: Mapping[str, Any],
    seed: int,
) -> tuple[MinimalPreprocessor | ClinicalCorePreprocessor, LogisticRegression]:
    normalized = normalize_training_weights(weights)
    processor = _fit_preprocessor(frame, normalized, model=model)
    matrix = _transform(processor, frame, model=model)
    estimator = _new_logistic(logistic_config, seed)
    estimator.fit(matrix, outcome, sample_weight=normalized)
    return processor, estimator


def _fit_sigmoid(
    raw_prediction: np.ndarray,
    outcome: np.ndarray,
    weights: np.ndarray,
    *,
    clip: float,
    logistic_config: Mapping[str, Any],
    seed: int,
) -> LogisticRegression:
    calibrator = _new_logistic(logistic_config, seed)
    calibrator.fit(
        _logit(raw_prediction, clip).reshape(-1, 1),
        outcome,
        sample_weight=normalize_training_weights(weights),
    )
    return calibrator


def load_fixed_exploratory_cohort(config: Mapping[str, Any], root: Path) -> pd.DataFrame:
    path = root / config["bindings"]["cohort"]
    if sha256_file(path) != config["bindings"]["cohort_sha256"]:
        raise ValueError("Stage 8 exploratory cohort binding changed")
    columns = [
        "participant_id",
        "survey_cycle",
        "strata",
        "psu",
        "analysis_weight",
        "cohort_eligible",
        "split",
        "cvd_death_5y",
        *MINIMAL_FEATURES,
        "systolic_bp_mean",
        "total_cholesterol",
        "hdl_cholesterol",
        "hba1c",
    ]
    frame = pd.read_parquet(path, columns=columns)
    policy = config["fixed_cohort"]
    frame = frame.loc[
        frame["cohort_eligible"].eq(policy["cohort_eligible"]) & frame["split"].eq(policy["split"])
    ].copy()
    complete = frame[policy["required_nonmissing_source_fields"]].notna().all(axis=1)
    frame = frame.loc[complete].reset_index(drop=True)
    if (len(frame), int(frame["cvd_death_5y"].sum())) != (
        policy["expected_n"],
        policy["expected_events"],
    ):
        raise ValueError("Stage 8 exploratory cohort counts changed")
    frame["sbp_mean"] = materialize_sbp_alias(frame["systolic_bp_mean"])
    if frame["participant_id"].duplicated().any():
        raise ValueError("Stage 8 exploratory participant IDs are not unique")
    return frame


def fit_paired_oof(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit paired fold models only; no full-cohort deployable estimator is created."""

    cv = config["cross_validation"]
    seed = int(cv["seed"])
    outcome = frame["cvd_death_5y"].to_numpy(dtype=np.int8)
    weights = frame["analysis_weight"].to_numpy(dtype=np.float64)
    groups = frame[cv["group"]].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=int(cv["outer_splits"]), shuffle=bool(cv["shuffle"]), random_state=seed
    )
    outer_splits = list(splitter.split(frame, outcome, groups))
    fold_id = np.zeros(len(frame), dtype=np.int16)
    fold_rows: list[dict[str, Any]] = []
    for fold, (train_index, validation_index) in enumerate(outer_splits, start=1):
        fold_id[validation_index] = fold
        train_groups = sorted(set(groups[train_index]))
        validation_groups = sorted(set(groups[validation_index]))
        if set(train_groups).intersection(validation_groups):
            raise RuntimeError("Outer grouped CV leaked a survey cycle")
        fold_rows.append(
            {
                "label": LABEL,
                "fold": fold,
                "train_cycles": ";".join(train_groups),
                "validation_cycles": ";".join(validation_groups),
                "n_train": len(train_index),
                "events_train": int(outcome[train_index].sum()),
                "n_validation": len(validation_index),
                "events_validation": int(outcome[validation_index].sum()),
                "participants_identical_for_models": True,
            }
        )
    if (fold_id == 0).any():
        raise RuntimeError("Outer grouped CV did not assign every participant")

    predictions = frame[
        ["participant_id", "survey_cycle", "strata", "psu", "analysis_weight", "cvd_death_5y"]
    ].copy()
    predictions.insert(0, "label", LABEL)
    predictions["fold"] = fold_id
    coefficients: list[dict[str, Any]] = []
    imputation: list[dict[str, Any]] = []
    clip = float(config["calibration"]["probability_clip"])

    for model in ("minimal", "clinical_core"):
        raw = np.full(len(frame), np.nan, dtype=np.float64)
        sigmoid = np.full(len(frame), np.nan, dtype=np.float64)
        for fold, (train_index, validation_index) in enumerate(outer_splits, start=1):
            train = frame.iloc[train_index]
            validation = frame.iloc[validation_index]
            processor, estimator = _fit_base(
                train,
                outcome[train_index],
                weights[train_index],
                model=model,
                logistic_config=config["logistic_regression"],
                seed=seed + fold,
            )
            validation_matrix = _transform(processor, validation, model=model)
            fold_raw = estimator.predict_proba(validation_matrix)[:, 1]
            raw[validation_index] = fold_raw

            inner_groups = groups[train_index]
            inner_splitter = StratifiedGroupKFold(
                n_splits=int(cv["inner_splits"]),
                shuffle=bool(cv["shuffle"]),
                random_state=seed + 100 * fold,
            )
            inner_raw = np.full(len(train_index), np.nan, dtype=np.float64)
            for inner_fold, (inner_train, inner_validation) in enumerate(
                inner_splitter.split(train, outcome[train_index], inner_groups), start=1
            ):
                inner_processor, inner_estimator = _fit_base(
                    train.iloc[inner_train],
                    outcome[train_index][inner_train],
                    weights[train_index][inner_train],
                    model=model,
                    logistic_config=config["logistic_regression"],
                    seed=seed + 1000 * fold + inner_fold,
                )
                inner_matrix = _transform(
                    inner_processor, train.iloc[inner_validation], model=model
                )
                inner_raw[inner_validation] = inner_estimator.predict_proba(inner_matrix)[:, 1]
            if not np.isfinite(inner_raw).all():
                raise RuntimeError("Nested grouped CV did not produce every calibration prediction")
            calibrator = _fit_sigmoid(
                inner_raw,
                outcome[train_index],
                weights[train_index],
                clip=clip,
                logistic_config=config["logistic_regression"],
                seed=seed + 10000 + fold,
            )
            sigmoid[validation_index] = calibrator.predict_proba(
                _logit(fold_raw, clip).reshape(-1, 1)
            )[:, 1]

            names = (
                processor.get_feature_names_out()
                if isinstance(processor, ClinicalCorePreprocessor)
                else processor.get_feature_names_out()
            )
            expected_parameters = int(config["models"][model]["predictor_parameters"])
            if len(names) != expected_parameters or estimator.coef_.shape[1] != expected_parameters:
                raise RuntimeError(f"{model} violates its fixed parameter budget")
            for feature, value in zip(names, estimator.coef_[0], strict=True):
                coefficients.append(
                    {
                        "label": LABEL,
                        "model": model,
                        "fold": fold,
                        "term": str(feature),
                        "coefficient": float(value),
                        "deployable_full_cohort_model": False,
                    }
                )
            minimal_processor = (
                processor.minimal_preprocessor_
                if isinstance(processor, ClinicalCorePreprocessor)
                else processor
            )
            for feature, value in minimal_processor.imputation_values_.items():
                imputation.append(
                    {
                        "label": LABEL,
                        "model": model,
                        "fold": fold,
                        "feature": feature,
                        "training_fold_value": value,
                        "fit_scope": "outer_training_fold_only",
                        "validation_cycles_excluded_from_fit": ";".join(
                            sorted(set(groups[validation_index]))
                        ),
                    }
                )
        if not np.isfinite(raw).all() or not np.isfinite(sigmoid).all():
            raise RuntimeError(f"{model} OOF predictions are incomplete")
        predictions[f"{model}_prediction_uncalibrated"] = raw
        predictions[f"{model}_prediction_sigmoid"] = sigmoid

    if not predictions["fold"].eq(fold_id).all() or len(predictions) != len(frame):
        raise RuntimeError("Paired predictions do not share identical participants and folds")
    return (
        predictions,
        pd.DataFrame(fold_rows),
        pd.DataFrame(coefficients),
        pd.DataFrame(imputation),
    )


def _metric_frame(predictions: pd.DataFrame, model: str) -> pd.DataFrame:
    return predictions[["strata", "psu", "analysis_weight", "cvd_death_5y"]].assign(
        predicted_risk=predictions[f"{model}_prediction_sigmoid"].to_numpy()
    )


def point_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in ("minimal", "clinical_core"):
        metrics = weighted_original_metrics(_metric_frame(predictions, model))
        for metric in METRICS:
            rows.append(
                {
                    "label": LABEL,
                    "model": model,
                    "metric": metric,
                    "estimate": metrics[metric],
                }
            )
    return pd.DataFrame(rows)


def net_benefit(
    outcome: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    threshold: float,
) -> float:
    positive = prediction >= threshold
    total = float(weight.sum())
    tp = float(weight[positive & (outcome == 1)].sum())
    fp = float(weight[positive & (outcome == 0)].sum())
    return tp / total - fp / total * threshold / (1.0 - threshold)


def point_dca_table(predictions: pd.DataFrame, thresholds: Sequence[float]) -> pd.DataFrame:
    outcome = predictions["cvd_death_5y"].to_numpy(dtype=np.int8)
    weights = predictions["analysis_weight"].to_numpy(dtype=np.float64)
    prevalence = float(np.average(outcome, weights=weights))
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        minimal = net_benefit(
            outcome,
            predictions["minimal_prediction_sigmoid"].to_numpy(dtype=np.float64),
            weights,
            threshold,
        )
        core = net_benefit(
            outcome,
            predictions["clinical_core_prediction_sigmoid"].to_numpy(dtype=np.float64),
            weights,
            threshold,
        )
        treat_all = prevalence - (1.0 - prevalence) * threshold / (1.0 - threshold)
        rows.extend(
            [
                {
                    "label": LABEL,
                    "threshold": threshold,
                    "quantity": "minimal_net_benefit",
                    "estimate": minimal,
                },
                {
                    "label": LABEL,
                    "threshold": threshold,
                    "quantity": "clinical_core_net_benefit",
                    "estimate": core,
                },
                {
                    "label": LABEL,
                    "threshold": threshold,
                    "quantity": "clinical_core_minus_minimal",
                    "estimate": core - minimal,
                },
                {
                    "label": LABEL,
                    "threshold": threshold,
                    "quantity": "treat_all_net_benefit",
                    "estimate": treat_all,
                },
                {
                    "label": LABEL,
                    "threshold": threshold,
                    "quantity": "treat_none_net_benefit",
                    "estimate": 0.0,
                },
            ]
        )
    return pd.DataFrame(rows)


def paired_cluster_bootstrap(
    predictions: pd.DataFrame,
    *,
    thresholds: Sequence[float],
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    base_weight = predictions["analysis_weight"].to_numpy(dtype=np.float64)
    outcome = predictions["cvd_death_5y"].to_numpy(dtype=np.int8)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for replicate in range(1, repetitions + 1):
        multiplicity = _bootstrap_multiplicity_by_groups(predictions, rng)
        selected = multiplicity > 0
        effective_weight = base_weight[selected] * multiplicity[selected].astype(np.float64)
        selected_outcome = outcome[selected]
        record: dict[str, Any] = {"replicate": replicate, "valid": True}
        model_metrics: dict[str, dict[str, float]] = {}
        try:
            for model in ("minimal", "clinical_core"):
                frame = pd.DataFrame(
                    {
                        "analysis_weight": effective_weight,
                        "cvd_death_5y": selected_outcome,
                        "predicted_risk": predictions.loc[
                            selected, f"{model}_prediction_sigmoid"
                        ].to_numpy(dtype=np.float64),
                    }
                )
                model_metrics[model] = weighted_original_metrics(frame)
                for metric in METRICS:
                    record[f"{model}__{metric}"] = model_metrics[model][metric]
                    record[f"diff__{metric}"] = (
                        model_metrics["clinical_core"][metric] - model_metrics["minimal"][metric]
                        if model == "clinical_core"
                        else np.nan
                    )
            for threshold in thresholds:
                suffix = format(float(threshold), ".6g")
                values: dict[str, float] = {}
                for model in ("minimal", "clinical_core"):
                    values[model] = net_benefit(
                        selected_outcome,
                        predictions.loc[selected, f"{model}_prediction_sigmoid"].to_numpy(
                            dtype=np.float64
                        ),
                        effective_weight,
                        float(threshold),
                    )
                    record[f"dca__{model}__{suffix}"] = values[model]
                record[f"dca__diff__{suffix}"] = values["clinical_core"] - values["minimal"]
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            record["valid"] = False
        rows.append(record)
    return pd.DataFrame(rows)


def add_percentile_intervals(
    point_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    *,
    confidence_level: float,
    minimum_valid_replicates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    alpha = 1.0 - confidence_level
    valid = bootstrap.loc[bootstrap["valid"]].copy()
    rows = []
    differences = []
    for entry in point_metrics.itertuples(index=False):
        values = valid[f"{entry.model}__{entry.metric}"].dropna().to_numpy(dtype=np.float64)
        enough = len(values) >= minimum_valid_replicates
        rows.append(
            {
                **entry._asdict(),
                "ci_lower": float(np.quantile(values, alpha / 2)) if enough else np.nan,
                "ci_upper": float(np.quantile(values, 1 - alpha / 2)) if enough else np.nan,
                "valid_replicates": len(values),
                "confidence_level": confidence_level,
            }
        )
    indexed = point_metrics.pivot(index="metric", columns="model", values="estimate")
    for metric in METRICS:
        values = valid[f"diff__{metric}"].dropna().to_numpy(dtype=np.float64)
        enough = len(values) >= minimum_valid_replicates
        differences.append(
            {
                "label": LABEL,
                "metric": metric,
                "difference_direction": "clinical_core_minus_minimal",
                "estimate": float(
                    indexed.loc[metric, "clinical_core"] - indexed.loc[metric, "minimal"]
                ),
                "ci_lower": float(np.quantile(values, alpha / 2)) if enough else np.nan,
                "ci_upper": float(np.quantile(values, 1 - alpha / 2)) if enough else np.nan,
                "valid_replicates": len(values),
                "confidence_level": confidence_level,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(differences)


def add_dca_intervals(
    point: pd.DataFrame,
    bootstrap: pd.DataFrame,
    *,
    confidence_level: float,
    minimum_valid_replicates: int,
) -> pd.DataFrame:
    alpha = 1.0 - confidence_level
    valid = bootstrap.loc[bootstrap["valid"]]
    rows = []
    mapping = {
        "minimal_net_benefit": "minimal",
        "clinical_core_net_benefit": "clinical_core",
        "clinical_core_minus_minimal": "diff",
    }
    for entry in point.itertuples(index=False):
        if entry.quantity in mapping:
            suffix = format(float(entry.threshold), ".6g")
            column = f"dca__{mapping[entry.quantity]}__{suffix}"
            values = valid[column].dropna().to_numpy(dtype=np.float64)
            enough = len(values) >= minimum_valid_replicates
            lower = float(np.quantile(values, alpha / 2)) if enough else np.nan
            upper = float(np.quantile(values, 1 - alpha / 2)) if enough else np.nan
            count = len(values)
        else:
            lower = upper = np.nan
            count = 0
        rows.append(
            {
                **entry._asdict(),
                "ci_lower": lower,
                "ci_upper": upper,
                "valid_replicates": count,
                "confidence_level": confidence_level,
            }
        )
    return pd.DataFrame(rows)


def _write_report(
    path: Path,
    *,
    metrics: pd.DataFrame,
    differences: pd.DataFrame,
    dca: pd.DataFrame,
    folds: pd.DataFrame,
    valid_replicates: int,
) -> None:
    display_metrics = metrics[
        ["model", "metric", "estimate", "ci_lower", "ci_upper", "valid_replicates"]
    ]
    display_differences = differences[
        ["metric", "estimate", "ci_lower", "ci_upper", "valid_replicates"]
    ]
    lines = [
        f"# Stage 8: {LABEL}",
        "",
        "## Обязательная интерпретация",
        "",
        f"**{LABEL}**",
        "",
        "Это underpowered NHANES internal development analysis. Он не является independent "
        "external validation и не разрешает export, production use, API или web routing.",
        "",
        "Primary/export gate остаётся неизменным: "
        "`FAIL_INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE`, n_eff_design=13061.96 < "
        "N_pmsampsize=13732.",
        "",
        "## Дизайн",
        "",
        "Обе модели получили OOF predictions для одних и тех же 18,987 участников и "
        "одинаковых четырёх folds, grouped по NHANES survey cycle. HbA1c использовался "
        "только для фиксированного availability mask и не входил ни в одну модель. "
        "Clinical fields были complete-case и не импутировались; imputation minimal "
        "predictors обучалась отдельно только внутри каждого training fold. Sigmoid "
        "calibration была nested и grouped.",
        "",
        "## Survey-weighted OOF metrics",
        "",
        display_metrics.to_markdown(index=False),
        "",
        "## Paired differences: clinical_core minus minimal",
        "",
        display_differences.to_markdown(index=False),
        "",
        "95% percentile intervals получены из pre-specified stratified PSU-within-strata "
        f"cluster bootstrap: {valid_replicates:,}/2,000 valid paired replicates, fixed OOF "
        "predictions, без refit.",
        "",
        "## Decision-curve analysis",
        "",
        dca.to_markdown(index=False),
        "",
        "## Identical outer folds",
        "",
        folds.to_markdown(index=False),
        "",
        "## Ограничения",
        "",
        "Результаты получены после aggregate gate failure и являются underpowered, "
        "development-only и exploratory. Они не переопределяют Section 10 FAIL, full-gate "
        "FAIL или ранее зафиксированные Stage 6/7 результаты. Полный-cohort estimator не "
        "обучался и сериализованный pipeline не создавался.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_stage8_exploratory(
    config_path: str | Path,
    *,
    root: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    repository = Path(root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = repository / config_file
    config = load_stage8_exploratory_config(config_file)
    commit = config["bindings"]["addendum_commit"]
    if (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=repository,
            check=False,
        ).returncode
        != 0
    ):
        raise ValueError("Exploratory addendum commit is not an ancestor of fitting")
    revision = f"{commit}:{config['bindings']['addendum_path']}"
    if _git(repository, "rev-parse", revision) != config["bindings"]["addendum_blob_sha1"]:
        raise ValueError("Exploratory addendum blob differs from its binding")
    primary_gate_paths = [
        "artifacts/stage8_clinical_gate/status.json",
        "reports/stage8_clinical_core_full_gate_report.md",
        "reports/tables/stage8_clinical_core_full_gate.csv",
    ]
    if _git(
        repository,
        "diff",
        "--name-only",
        config["bindings"]["primary_gate_commit"],
        "--",
        *primary_gate_paths,
    ):
        raise ValueError("Existing primary gate evidence changed before exploratory fitting")

    outputs = {key: repository / value for key, value in config["outputs"].items()}
    known = [path for key, path in outputs.items() if key != "artifact_dir"]
    if any(path.exists() for path in known) and not force:
        raise FileExistsError("Stage 8 exploratory outputs exist; pass --force to replace them")
    for path in known:
        path.parent.mkdir(parents=True, exist_ok=True)

    frame = load_fixed_exploratory_cohort(config, repository)
    predictions, folds, coefficients, imputation = fit_paired_oof(frame, config)
    point = point_metric_table(predictions)
    point_dca = point_dca_table(predictions, config["decision_curve"]["thresholds"])
    bootstrap_config = config["cluster_bootstrap"]
    bootstrap = paired_cluster_bootstrap(
        predictions,
        thresholds=config["decision_curve"]["thresholds"],
        repetitions=int(bootstrap_config["repetitions"]),
        seed=int(bootstrap_config["seed"]),
    )
    valid_replicates = int(bootstrap["valid"].sum())
    if valid_replicates < int(bootstrap_config["minimum_valid_replicates"]):
        raise RuntimeError("Too few valid paired Stage 8 bootstrap replicates")
    metrics, differences = add_percentile_intervals(
        point,
        bootstrap,
        confidence_level=float(bootstrap_config["confidence_level"]),
        minimum_valid_replicates=int(bootstrap_config["minimum_valid_replicates"]),
    )
    dca = add_dca_intervals(
        point_dca,
        bootstrap,
        confidence_level=float(bootstrap_config["confidence_level"]),
        minimum_valid_replicates=int(bootstrap_config["minimum_valid_replicates"]),
    )
    for table in (predictions, folds, coefficients, imputation, metrics, differences, dca):
        if "label" not in table or not table["label"].eq(LABEL).all():
            raise RuntimeError("An exploratory output lacks the mandatory label")
    predictions.to_parquet(outputs["oof_predictions"], index=False)
    bootstrap.insert(0, "label", LABEL)
    bootstrap.to_csv(outputs["bootstrap_metrics"], index=False)
    metrics.to_csv(outputs["metrics"], index=False)
    differences.to_csv(outputs["paired_differences"], index=False)
    dca.to_csv(outputs["decision_curve"], index=False)
    folds.to_csv(outputs["folds"], index=False)
    coefficients.to_csv(outputs["coefficients"], index=False)
    imputation.to_csv(outputs["imputation_audit"], index=False)
    _write_report(
        outputs["report"],
        metrics=metrics,
        differences=differences,
        dca=dca,
        folds=folds,
        valid_replicates=valid_replicates,
    )

    forbidden_suffixes = {".joblib", ".pkl"}
    forbidden = [
        path
        for path in outputs["artifact_dir"].rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    ]
    if forbidden:
        raise RuntimeError(f"Deployable Stage 8 artifacts are forbidden: {forbidden}")
    status: dict[str, Any] = {
        "schema_version": 1,
        "stage": "stage8_underpowered_exploratory_development_only",
        "label": LABEL,
        "PRIMARY_EXPORT_GATE": "FAIL",
        "PRIMARY_EXPORT_GATE_REASON": "FAIL_INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE",
        "EXPLORATORY_FITTING": "COMPLETE",
        "WEB_ROUTING_ALLOWED": False,
        "EXTERNAL_VALIDATION": "NOT_PERFORMED",
        "generated_at": datetime.now(UTC).isoformat(),
        "cohort_n": len(frame),
        "raw_cvd_events": int(frame["cvd_death_5y"].sum()),
        "models": ["minimal", "clinical_core"],
        "paired_identical_participants_and_folds": True,
        "valid_bootstrap_replicates": valid_replicates,
        "full_cohort_deployable_model_fitted": False,
        "serialized_model_or_pipeline_created": False,
        "api_route_created": False,
        "web_routing_configuration_created": False,
        "input_bindings": {
            "config": _binding(config_file, repository),
            "addendum": _binding(repository / config["bindings"]["addendum_path"], repository),
            "cohort": _binding(repository / config["bindings"]["cohort"], repository),
            "source": _binding(Path(__file__), repository),
        },
        "outputs": {
            key: _binding(path, repository)
            for key, path in outputs.items()
            if key not in {"artifact_dir", "status"}
        },
    }
    outputs["status"].write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return status


def validate_stage8_exploratory_status(
    status_path: str | Path,
    *,
    root: str | Path,
) -> dict[str, Any]:
    repository = Path(root).resolve()
    path = Path(status_path)
    if not path.is_absolute():
        path = repository / path
    status = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "PRIMARY_EXPORT_GATE": "FAIL",
        "EXPLORATORY_FITTING": "COMPLETE",
        "WEB_ROUTING_ALLOWED": False,
        "EXTERNAL_VALIDATION": "NOT_PERFORMED",
    }
    if any(status.get(key) != value for key, value in required.items()):
        raise ValueError("Stage 8 exploratory final status differs from the authorization")
    if status.get("label") != LABEL:
        raise ValueError("Stage 8 exploratory final status lacks its label")
    if status.get("full_cohort_deployable_model_fitted") is not False:
        raise ValueError("Stage 8 exploratory analysis cannot fit a deployable full model")
    for entry in (*status["input_bindings"].values(), *status["outputs"].values()):
        output = repository / entry["repository_relative_path"]
        current = _binding(output, repository)
        if current["sha256"] != entry["sha256"] or current["hash_mode"] != entry["hash_mode"]:
            raise ValueError(f"Stage 8 exploratory output changed: {output}")
    predictions = pd.read_parquet(
        repository / "artifacts/stage8_exploratory/oof_predictions.parquet"
    )
    if len(predictions) != 18987 or int(predictions["cvd_death_5y"].sum()) != 218:
        raise ValueError("Stage 8 exploratory OOF cohort differs")
    if not predictions["label"].eq(LABEL).all():
        raise ValueError("Stage 8 OOF predictions lack the mandatory label")
    if predictions["participant_id"].duplicated().any() or predictions["fold"].isna().any():
        raise ValueError("Stage 8 paired OOF identities/folds are invalid")
    for relative in (
        "reports/tables/stage8_exploratory_metrics.csv",
        "reports/tables/stage8_exploratory_paired_differences.csv",
        "reports/tables/stage8_exploratory_decision_curve.csv",
        "reports/tables/stage8_exploratory_folds.csv",
        "reports/tables/stage8_exploratory_fold_coefficients.csv",
        "reports/tables/stage8_exploratory_imputation_audit.csv",
        "artifacts/stage8_exploratory/bootstrap_metrics.csv",
    ):
        table = pd.read_csv(repository / relative)
        if "label" not in table or not table["label"].eq(LABEL).all():
            raise ValueError(f"Stage 8 exploratory table lacks its label: {relative}")
    imputation = pd.read_csv(repository / "reports/tables/stage8_exploratory_imputation_audit.csv")
    if set(imputation["feature"]).intersection(
        {"sbp_mean", "total_cholesterol", "hdl_cholesterol"}
    ):
        raise ValueError("Clinical fields appear in the imputation audit")
    paired_imputation = imputation.pivot(
        index=["fold", "feature"], columns="model", values="training_fold_value"
    )
    if set(paired_imputation.columns) != {"minimal", "clinical_core"}:
        raise ValueError("Both models must record paired minimal imputation values")
    for row in paired_imputation.itertuples(index=False):
        left, right = row
        try:
            equal = bool(np.isclose(float(left), float(right), equal_nan=True))
        except (TypeError, ValueError):
            equal = str(left) == str(right)
        if not equal:
            raise ValueError("Paired models used different fold-wise minimal imputation")
    primary_gate = json.loads(
        (repository / "artifacts/stage8_clinical_gate/status.json").read_text(encoding="utf-8")
    )
    if primary_gate.get("status") != "FAIL_FULL_GATE_CLINICAL_CORE_FITTING_FORBIDDEN":
        raise ValueError("Primary Stage 8 full-gate failure was changed")
    gate_paths = [
        "artifacts/stage8_clinical_gate/status.json",
        "reports/stage8_clinical_core_full_gate_report.md",
        "reports/tables/stage8_clinical_core_full_gate.csv",
    ]
    if _git(
        repository,
        "diff",
        "--name-only",
        "a3efc76cb0565ff4638c30a1c5cb747b95fa73a4",
        "--",
        *gate_paths,
    ):
        raise ValueError("Primary Stage 8 gate evidence differs from its gate commit")
    forbidden = []
    for pattern in ("*.joblib", "*.pkl", "*routing*.json"):
        forbidden.extend((repository / "artifacts/stage8_exploratory").glob(pattern))
    if forbidden:
        raise ValueError(f"Forbidden deployable/routing artifacts exist: {forbidden}")
    return {
        "status": "PASS",
        **required,
        "label": LABEL,
        "cohort_n": 18987,
        "events": 218,
        "deployable_artifacts": 0,
    }


__all__ = [
    "ClinicalCorePreprocessor",
    "fit_paired_oof",
    "load_fixed_exploratory_cohort",
    "load_stage8_exploratory_config",
    "run_stage8_exploratory",
    "validate_stage8_exploratory_status",
]
