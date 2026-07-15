"""Survey-weighted logistic baselines and nested cross-fitted calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold  # type: ignore[import-untyped]

from cvd_risk.modeling.preprocessing import ROUTE_FEATURES, MinimalPreprocessor


def normalize_training_weights(sample_weight: np.ndarray) -> np.ndarray:
    """Normalize positive survey weights to mean one within a training fold."""

    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError("sample_weight must be a non-empty one-dimensional array")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("sample_weight must contain only finite positive values")
    return weights / weights.mean()


def _logit(probability: np.ndarray, clip: float) -> np.ndarray:
    bounded = np.clip(np.asarray(probability, dtype=np.float64), clip, 1.0 - clip)
    return np.log(bounded / (1.0 - bounded))


def _new_logistic(settings: dict[str, Any], seed: int) -> LogisticRegression:
    penalty = settings.get("penalty")
    if penalty is not None:
        raise ValueError("Primary Stage 4 logistic baselines must remain unpenalized")
    return LogisticRegression(
        C=np.inf,
        l1_ratio=0.0,
        solver=str(settings.get("solver", "lbfgs")),
        max_iter=int(settings.get("max_iter", 1000)),
        tol=float(settings.get("tolerance", 1.0e-8)),
        random_state=seed,
    )


@dataclass
class FittedBaselineModel:
    """Serializable preprocessing, uncalibrated model, and Platt calibrator."""

    name: str
    route: str
    preprocessor: MinimalPreprocessor
    estimator: LogisticRegression
    calibrator: LogisticRegression
    probability_clip: float

    def predict_proba(self, frame: pd.DataFrame, *, calibrated: bool = True) -> np.ndarray:
        features = frame.loc[:, ROUTE_FEATURES[self.route]]
        matrix = self.preprocessor.transform(features)
        raw = self.estimator.predict_proba(matrix)[:, 1]
        if not calibrated:
            return raw
        linear_predictor = _logit(raw, self.probability_clip).reshape(-1, 1)
        return self.calibrator.predict_proba(linear_predictor)[:, 1]


def _fit_base_model(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    route: str,
    spline_quantiles: tuple[float, ...],
    logistic_settings: dict[str, Any],
    seed: int,
) -> tuple[MinimalPreprocessor, LogisticRegression]:
    normalized = normalize_training_weights(sample_weight)
    preprocessor = MinimalPreprocessor(route=route, spline_quantiles=spline_quantiles)
    route_frame = X.loc[:, ROUTE_FEATURES[route]]
    preprocessor.fit(route_frame, sample_weight=normalized)
    matrix = preprocessor.transform(route_frame)
    estimator = _new_logistic(logistic_settings, seed)
    estimator.fit(matrix, y, sample_weight=normalized)
    return preprocessor, estimator


def _fit_calibrator(
    raw_prediction: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    clip: float,
    logistic_settings: dict[str, Any],
    seed: int,
) -> LogisticRegression:
    calibrator = _new_logistic(logistic_settings, seed)
    calibrator.fit(
        _logit(raw_prediction, clip).reshape(-1, 1),
        y,
        sample_weight=normalize_training_weights(sample_weight),
    )
    return calibrator


def _inner_oof_predictions(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    groups: np.ndarray,
    *,
    route: str,
    spline_quantiles: tuple[float, ...],
    logistic_settings: dict[str, Any],
    seed: int,
    n_splits: int,
) -> np.ndarray:
    unique_groups = np.unique(groups)
    actual_splits = min(n_splits, len(unique_groups))
    if actual_splits < 2:
        raise ValueError("Nested calibration requires at least two training groups")
    splitter = StratifiedGroupKFold(
        n_splits=actual_splits,
        shuffle=True,
        random_state=seed,
    )
    predictions = np.full(len(X), np.nan, dtype=np.float64)
    for inner_fold, (train_index, validation_index) in enumerate(
        splitter.split(X, y, groups), start=1
    ):
        preprocessor, estimator = _fit_base_model(
            X.iloc[train_index],
            y[train_index],
            sample_weight[train_index],
            route=route,
            spline_quantiles=spline_quantiles,
            logistic_settings=logistic_settings,
            seed=seed + inner_fold,
        )
        validation_matrix = preprocessor.transform(
            X.iloc[validation_index].loc[:, ROUTE_FEATURES[route]]
        )
        predictions[validation_index] = estimator.predict_proba(validation_matrix)[:, 1]
    if not np.isfinite(predictions).all():
        raise RuntimeError("Nested cross-validation did not produce every prediction")
    return predictions


def fit_nested_oof_baseline(
    frame: pd.DataFrame,
    *,
    name: str,
    route: str,
    outcome_column: str,
    weight_column: str,
    group_column: str,
    outer_splits: int,
    inner_splits: int,
    spline_quantiles: tuple[float, ...],
    logistic_settings: dict[str, Any],
    probability_clip: float,
    seed: int,
) -> tuple[FittedBaselineModel, pd.DataFrame, pd.DataFrame]:
    """Fit nested OOF predictions, then a full-development deployable candidate."""

    required = {*ROUTE_FEATURES[route], outcome_column, weight_column, group_column}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Training frame is missing fields: {sorted(missing)}")
    X = frame.loc[:, ROUTE_FEATURES[route]].copy()
    y = frame[outcome_column].to_numpy(dtype=np.int8)
    weights = frame[weight_column].to_numpy(dtype=np.float64)
    groups = frame[group_column].to_numpy()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Development outcome must contain both binary classes")
    if len(np.unique(groups)) < outer_splits:
        raise ValueError("Not enough distinct groups for the configured outer CV")

    splitter = StratifiedGroupKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=seed,
    )
    raw_oof = np.full(len(frame), np.nan, dtype=np.float64)
    calibrated_oof = np.full(len(frame), np.nan, dtype=np.float64)
    fold_id = np.zeros(len(frame), dtype=np.int16)
    fold_rows: list[dict[str, Any]] = []

    for fold, (train_index, test_index) in enumerate(splitter.split(X, y, groups), start=1):
        train_groups = set(groups[train_index])
        test_groups = set(groups[test_index])
        if train_groups.intersection(test_groups):
            raise RuntimeError("Grouped CV placed a survey year in train and validation")
        preprocessor, estimator = _fit_base_model(
            X.iloc[train_index],
            y[train_index],
            weights[train_index],
            route=route,
            spline_quantiles=spline_quantiles,
            logistic_settings=logistic_settings,
            seed=seed + fold,
        )
        test_matrix = preprocessor.transform(X.iloc[test_index])
        fold_raw = estimator.predict_proba(test_matrix)[:, 1]
        raw_oof[test_index] = fold_raw

        inner_raw = _inner_oof_predictions(
            X.iloc[train_index].reset_index(drop=True),
            y[train_index],
            weights[train_index],
            groups[train_index],
            route=route,
            spline_quantiles=spline_quantiles,
            logistic_settings=logistic_settings,
            seed=seed + 100 * fold,
            n_splits=inner_splits,
        )
        calibrator = _fit_calibrator(
            inner_raw,
            y[train_index],
            weights[train_index],
            clip=probability_clip,
            logistic_settings=logistic_settings,
            seed=seed + 1000 + fold,
        )
        calibrated_oof[test_index] = calibrator.predict_proba(
            _logit(fold_raw, probability_clip).reshape(-1, 1)
        )[:, 1]
        fold_id[test_index] = fold
        fold_rows.append(
            {
                "fold": fold,
                "train_years": ";".join(map(str, sorted(train_groups))),
                "validation_years": ";".join(map(str, sorted(test_groups))),
                "n_train": len(train_index),
                "events_train": int(y[train_index].sum()),
                "n_validation": len(test_index),
                "events_validation": int(y[test_index].sum()),
            }
        )
    if not np.isfinite(raw_oof).all() or not np.isfinite(calibrated_oof).all():
        raise RuntimeError("Outer cross-validation did not produce every prediction")
    if (fold_id == 0).any():
        raise RuntimeError("Outer cross-validation did not assign every row")

    full_preprocessor, full_estimator = _fit_base_model(
        X,
        y,
        weights,
        route=route,
        spline_quantiles=spline_quantiles,
        logistic_settings=logistic_settings,
        seed=seed,
    )
    final_calibrator = _fit_calibrator(
        raw_oof,
        y,
        weights,
        clip=probability_clip,
        logistic_settings=logistic_settings,
        seed=seed,
    )
    fitted = FittedBaselineModel(
        name=name,
        route=route,
        preprocessor=full_preprocessor,
        estimator=full_estimator,
        calibrator=final_calibrator,
        probability_clip=probability_clip,
    )
    predictions = pd.DataFrame(
        {
            "model": name,
            "fold": fold_id,
            "prediction_uncalibrated": raw_oof,
            "prediction_sigmoid": calibrated_oof,
        },
        index=frame.index,
    )
    return fitted, predictions, pd.DataFrame(fold_rows)


def calibration_summary(
    y: np.ndarray,
    prediction: np.ndarray,
    sample_weight: np.ndarray,
    *,
    probability_clip: float,
) -> tuple[float, float]:
    """Estimate calibration intercept and slope on a prediction vector."""

    model = LogisticRegression(
        C=np.inf,
        l1_ratio=0.0,
        solver="lbfgs",
        max_iter=1000,
        tol=1.0e-10,
    )
    model.fit(
        _logit(prediction, probability_clip).reshape(-1, 1),
        y,
        sample_weight=normalize_training_weights(sample_weight),
    )
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def binary_prediction_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
    sample_weight: np.ndarray,
    *,
    probability_clip: float,
) -> dict[str, float]:
    """Compute pre-specified development metrics for one prediction vector."""

    weights = np.asarray(sample_weight, dtype=np.float64)
    bounded = np.clip(prediction, probability_clip, 1.0 - probability_clip)
    intercept, slope = calibration_summary(
        y, bounded, weights, probability_clip=probability_clip
    )
    return {
        "roc_auc": float(roc_auc_score(y, bounded, sample_weight=weights)),
        "average_precision": float(
            average_precision_score(y, bounded, sample_weight=weights)
        ),
        "brier_score": float(brier_score_loss(y, bounded, sample_weight=weights)),
        "log_loss": float(log_loss(y, bounded, sample_weight=weights, labels=[0, 1])),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "observed_expected_ratio": float(np.sum(weights * y) / np.sum(weights * bounded)),
    }


__all__ = [
    "FittedBaselineModel",
    "binary_prediction_metrics",
    "calibration_summary",
    "fit_nested_oof_baseline",
    "normalize_training_weights",
]
