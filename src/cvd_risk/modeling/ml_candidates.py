"""Nested grouped tuning for tree-based Stage 5 model candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import brier_score_loss, roc_auc_score  # type: ignore[import-untyped]
from sklearn.model_selection import StratifiedGroupKFold  # type: ignore[import-untyped]
from xgboost import XGBClassifier

from cvd_risk.modeling.baselines import normalize_training_weights
from cvd_risk.modeling.preprocessing import ROUTE_FEATURES, MinimalPreprocessor


def _logit(probability: np.ndarray, clip: float) -> np.ndarray:
    bounded = np.clip(np.asarray(probability, dtype=np.float64), clip, 1.0 - clip)
    return np.log(bounded / (1.0 - bounded))


def _fit_calibrator(
    prediction: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    clip: float,
    seed: int,
) -> LogisticRegression:
    calibrator = LogisticRegression(
        C=np.inf,
        l1_ratio=0.0,
        solver="lbfgs",
        max_iter=1000,
        tol=1.0e-10,
        random_state=seed,
    )
    calibrator.fit(
        _logit(prediction, clip).reshape(-1, 1),
        y,
        sample_weight=normalize_training_weights(sample_weight),
    )
    return calibrator


def _make_estimator(
    estimator_name: str,
    parameters: dict[str, Any],
    *,
    seed: int,
) -> Any:
    if estimator_name == "RandomForestClassifier":
        return RandomForestClassifier(random_state=seed, **parameters)
    if estimator_name == "XGBClassifier":
        return XGBClassifier(random_state=seed, **parameters)
    raise ValueError(f"Unsupported Stage 5 estimator: {estimator_name!r}")


@dataclass
class FittedMLCandidate:
    """Serializable Stage 5 preprocessor, estimator, and sigmoid calibrator."""

    name: str
    estimator_name: str
    route: str
    parameters: dict[str, Any]
    preprocessor: MinimalPreprocessor
    estimator: Any
    calibrator: LogisticRegression
    probability_clip: float

    def predict_proba(self, frame: pd.DataFrame, *, calibrated: bool = True) -> np.ndarray:
        matrix = self.preprocessor.transform(frame.loc[:, ROUTE_FEATURES[self.route]])
        raw = np.asarray(self.estimator.predict_proba(matrix)[:, 1], dtype=np.float64)
        if not calibrated:
            return raw
        return self.calibrator.predict_proba(
            _logit(raw, self.probability_clip).reshape(-1, 1)
        )[:, 1]


def _fit_estimator(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    estimator_name: str,
    parameters: dict[str, Any],
    spline_quantiles: tuple[float, ...],
    seed: int,
) -> tuple[MinimalPreprocessor, Any]:
    normalized = normalize_training_weights(sample_weight)
    preprocessor = MinimalPreprocessor(
        route="minimal", spline_quantiles=spline_quantiles
    ).fit(X, sample_weight=normalized)
    matrix = preprocessor.transform(X)
    estimator = _make_estimator(estimator_name, parameters, seed=seed)
    estimator.fit(matrix, y, sample_weight=normalized)
    return preprocessor, estimator


def _inner_predictions(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    groups: np.ndarray,
    *,
    estimator_name: str,
    parameters: dict[str, Any],
    spline_quantiles: tuple[float, ...],
    n_splits: int,
    seed: int,
) -> np.ndarray:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    prediction = np.full(len(X), np.nan, dtype=np.float64)
    for inner_fold, (train_index, validation_index) in enumerate(
        splitter.split(X, y, groups), start=1
    ):
        preprocessor, estimator = _fit_estimator(
            X.iloc[train_index],
            y[train_index],
            sample_weight[train_index],
            estimator_name=estimator_name,
            parameters=parameters,
            spline_quantiles=spline_quantiles,
            seed=seed + inner_fold,
        )
        matrix = preprocessor.transform(X.iloc[validation_index])
        prediction[validation_index] = estimator.predict_proba(matrix)[:, 1]
    if not np.isfinite(prediction).all():
        raise RuntimeError("Inner tuning did not predict every training participant")
    return prediction


def fit_nested_ml_candidate(
    frame: pd.DataFrame,
    *,
    name: str,
    estimator_name: str,
    common_parameters: dict[str, Any],
    candidates: list[dict[str, Any]],
    outcome_column: str,
    weight_column: str,
    group_column: str,
    outer_splits: int,
    inner_splits: int,
    spline_quantiles: tuple[float, ...],
    probability_clip: float,
    seed: int,
) -> tuple[FittedMLCandidate, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tune in each outer fold, calibrate from inner OOF, and fit a final candidate."""

    X = frame.loc[:, ROUTE_FEATURES["minimal"]].copy()
    y = frame[outcome_column].to_numpy(dtype=np.int8)
    weights = frame[weight_column].to_numpy(dtype=np.float64)
    groups = frame[group_column].to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=seed,
    )
    raw_oof = np.full(len(frame), np.nan, dtype=np.float64)
    calibrated_oof = np.full(len(frame), np.nan, dtype=np.float64)
    fold_id = np.zeros(len(frame), dtype=np.int16)
    tuning_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    candidate_specs: list[tuple[str, dict[str, Any]]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        candidate_parameters = {
            key: value for key, value in candidate.items() if key != "candidate_id"
        }
        candidate_specs.append((candidate_id, {**common_parameters, **candidate_parameters}))
    if len({candidate_id for candidate_id, _ in candidate_specs}) != len(candidate_specs):
        raise ValueError(f"{name} candidate_id values must be unique")

    for outer_fold, (train_index, validation_index) in enumerate(
        splitter.split(X, y, groups), start=1
    ):
        inner_results: list[tuple[float, str, dict[str, Any], np.ndarray, float]] = []
        outer_X = X.iloc[train_index].reset_index(drop=True)
        outer_y = y[train_index]
        outer_weights = weights[train_index]
        outer_groups = groups[train_index]
        for candidate_id, parameters in candidate_specs:
            inner_prediction = _inner_predictions(
                outer_X,
                outer_y,
                outer_weights,
                outer_groups,
                estimator_name=estimator_name,
                parameters=parameters,
                spline_quantiles=spline_quantiles,
                n_splits=inner_splits,
                seed=seed + outer_fold * 100,
            )
            inner_brier = float(
                brier_score_loss(
                    outer_y, inner_prediction, sample_weight=outer_weights
                )
            )
            inner_auc = float(
                roc_auc_score(outer_y, inner_prediction, sample_weight=outer_weights)
            )
            inner_results.append(
                (inner_brier, candidate_id, parameters, inner_prediction, inner_auc)
            )
        inner_results.sort(key=lambda item: (item[0], item[1]))
        selected_brier, selected_id, selected_parameters, selected_inner, selected_auc = (
            inner_results[0]
        )
        for inner_brier, candidate_id, parameters, _prediction, inner_auc in inner_results:
            tuning_rows.append(
                {
                    "model": name,
                    "outer_fold": outer_fold,
                    "candidate_id": candidate_id,
                    "inner_brier_weighted": inner_brier,
                    "inner_roc_auc_weighted": inner_auc,
                    "selected_for_outer_fold": candidate_id == selected_id,
                    "parameters": parameters,
                }
            )

        preprocessor, estimator = _fit_estimator(
            X.iloc[train_index],
            outer_y,
            outer_weights,
            estimator_name=estimator_name,
            parameters=selected_parameters,
            spline_quantiles=spline_quantiles,
            seed=seed + outer_fold,
        )
        outer_raw = np.asarray(
            estimator.predict_proba(preprocessor.transform(X.iloc[validation_index]))[:, 1],
            dtype=np.float64,
        )
        calibrator = _fit_calibrator(
            selected_inner,
            outer_y,
            outer_weights,
            clip=probability_clip,
            seed=seed + 1000 + outer_fold,
        )
        outer_calibrated = calibrator.predict_proba(
            _logit(outer_raw, probability_clip).reshape(-1, 1)
        )[:, 1]
        raw_oof[validation_index] = outer_raw
        calibrated_oof[validation_index] = outer_calibrated
        fold_id[validation_index] = outer_fold
        fold_rows.append(
            {
                "model": name,
                "fold": outer_fold,
                "selected_candidate_id": selected_id,
                "inner_brier_weighted": selected_brier,
                "inner_roc_auc_weighted": selected_auc,
                "n_validation": len(validation_index),
                "events_validation": int(y[validation_index].sum()),
            }
        )

    if not np.isfinite(raw_oof).all() or not np.isfinite(calibrated_oof).all():
        raise RuntimeError("Outer ML cross-validation did not produce every prediction")
    if (fold_id == 0).any():
        raise RuntimeError("Outer ML cross-validation did not assign every participant")

    tuning = pd.DataFrame(tuning_rows)
    mean_scores = (
        tuning.groupby("candidate_id", observed=True)["inner_brier_weighted"]
        .mean()
        .sort_values(kind="mergesort")
    )
    final_candidate_id = str(mean_scores.index[0])
    final_parameters = dict(
        next(
            parameters
            for candidate_id, parameters in candidate_specs
            if candidate_id == final_candidate_id
        )
    )
    tuning["selected_for_final_fit"] = tuning["candidate_id"].eq(final_candidate_id)
    final_preprocessor, final_estimator = _fit_estimator(
        X,
        y,
        weights,
        estimator_name=estimator_name,
        parameters=final_parameters,
        spline_quantiles=spline_quantiles,
        seed=seed,
    )
    final_calibrator = _fit_calibrator(
        raw_oof,
        y,
        weights,
        clip=probability_clip,
        seed=seed,
    )
    fitted = FittedMLCandidate(
        name=name,
        estimator_name=estimator_name,
        route="minimal",
        parameters=final_parameters,
        preprocessor=final_preprocessor,
        estimator=final_estimator,
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
    return fitted, predictions, tuning, pd.DataFrame(fold_rows)


__all__ = ["FittedMLCandidate", "fit_nested_ml_candidate"]
