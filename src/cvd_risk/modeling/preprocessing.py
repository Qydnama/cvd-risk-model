"""Fold-fitted preprocessing for the protocol-locked minimal model routes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore[import-untyped]
from sklearn.utils.validation import check_is_fitted  # type: ignore[import-untyped]

from cvd_risk.schemas import validate_fast_feature_schema

RCS_QUANTILES: tuple[float, ...] = (0.05, 0.35, 0.65, 0.95)
ROUTE_FEATURES: dict[str, tuple[str, ...]] = {
    "age_sex": ("age", "sex"),
    "minimal_no_general_health": (
        "age",
        "sex",
        "bmi",
        "smoking_status",
        "hypertension_dx",
        "diabetes_dx",
    ),
    "minimal": (
        "age",
        "sex",
        "bmi",
        "smoking_status",
        "hypertension_dx",
        "diabetes_dx",
        "general_health",
    ),
}
ROUTE_PARAMETER_COUNTS: dict[str, int] = {
    "age_sex": 4,
    "minimal_no_general_health": 11,
    "minimal": 15,
}


def _validate_weights(sample_weight: Sequence[float] | np.ndarray, n: int) -> np.ndarray:
    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.ndim != 1 or len(weights) != n:
        raise ValueError("sample_weight must be a one-dimensional array matching X")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("sample_weight must contain only finite positive values")
    return weights


def weighted_quantile(
    values: Sequence[float] | np.ndarray,
    quantiles: Sequence[float] | np.ndarray,
    sample_weight: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return deterministic weighted empirical quantiles for positive weights."""

    observed = np.asarray(values, dtype=np.float64)
    requested = np.asarray(quantiles, dtype=np.float64)
    weights = _validate_weights(sample_weight, len(observed))
    if observed.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if not np.isfinite(observed).all():
        raise ValueError("values must contain only finite values")
    if requested.ndim != 1 or (requested < 0).any() or (requested > 1).any():
        raise ValueError("quantiles must be a one-dimensional array within [0, 1]")
    order = np.argsort(observed, kind="mergesort")
    sorted_values = observed[order]
    sorted_weights = weights[order]
    positions = (np.cumsum(sorted_weights) - 0.5 * sorted_weights) / sorted_weights.sum()
    return np.interp(requested, positions, sorted_values)


def restricted_cubic_spline_basis(values: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Build a three-column Harrell restricted cubic spline basis for four knots."""

    x = np.asarray(values, dtype=np.float64).reshape(-1)
    k = np.asarray(knots, dtype=np.float64).reshape(-1)
    if len(k) != 4 or not np.isfinite(k).all() or not np.all(np.diff(k) > 0):
        raise ValueError("Restricted cubic splines require four distinct increasing knots")
    scale = (k[-1] - k[0]) ** 2
    nonlinear: list[np.ndarray] = []
    for knot in k[:-2]:
        term = (
            np.maximum(x - knot, 0.0) ** 3
            - np.maximum(x - k[-2], 0.0) ** 3 * (k[-1] - knot) / (k[-1] - k[-2])
            + np.maximum(x - k[-1], 0.0) ** 3 * (k[-2] - knot) / (k[-1] - k[-2])
        ) / scale
        nonlinear.append(term)
    return np.column_stack((x, *nonlinear)).astype(np.float64, copy=False)


def _weighted_mode(series: pd.Series, weights: np.ndarray, levels: Sequence[Any]) -> Any:
    available = series.notna().to_numpy()
    if not available.any():
        raise ValueError(f"Cannot fit imputation: {series.name!r} is entirely missing")
    totals = {
        level: float(weights[available & series.eq(level).fillna(False).to_numpy()].sum())
        for level in levels
    }
    return max(levels, key=lambda level: (totals[level], -levels.index(level)))


class MinimalPreprocessor(TransformerMixin, BaseEstimator):
    """Impute and encode a minimal route using training-fold information only."""

    def __init__(
        self,
        *,
        route: str = "minimal",
        spline_quantiles: tuple[float, ...] = RCS_QUANTILES,
    ) -> None:
        self.route = route
        self.spline_quantiles = spline_quantiles

    def fit(
        self,
        X: pd.DataFrame,
        y: Any = None,
        *,
        sample_weight: Sequence[float] | np.ndarray | None = None,
    ) -> MinimalPreprocessor:
        del y
        if self.route not in ROUTE_FEATURES:
            raise ValueError(f"Unknown model route: {self.route!r}")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MinimalPreprocessor requires a pandas DataFrame")
        missing = set(ROUTE_FEATURES[self.route]).difference(X.columns)
        if missing:
            raise ValueError(f"Route {self.route!r} is missing features: {sorted(missing)}")
        # Validate the complete canonical contract when it is present. Route subsets
        # are validated explicitly below to support AgeSexLogistic.
        if set(ROUTE_FEATURES["minimal"]).issubset(X.columns):
            validate_fast_feature_schema(X, allow_missing=True)
        weights = (
            np.ones(len(X), dtype=np.float64)
            if sample_weight is None
            else _validate_weights(sample_weight, len(X))
        )
        self.imputation_values_: dict[str, Any] = {}
        self.spline_knots_: dict[str, np.ndarray] = {}
        self.spline_centers_: dict[str, np.ndarray] = {}
        self.spline_scales_: dict[str, np.ndarray] = {}

        continuous = ["age"]
        if self.route != "age_sex":
            continuous.append("bmi")
        for column in continuous:
            numeric = pd.to_numeric(X[column], errors="coerce").astype(float)
            malformed = X[column].notna() & numeric.isna()
            if malformed.any():
                raise ValueError(f"{column} contains non-numeric values")
            observed = numeric.notna().to_numpy()
            if not observed.any():
                raise ValueError(f"Cannot fit imputation: {column!r} is entirely missing")
            median = weighted_quantile(
                numeric.loc[observed].to_numpy(), (0.5,), weights[observed]
            )[0]
            self.imputation_values_[column] = float(median)
            filled = numeric.fillna(float(median)).to_numpy(dtype=np.float64)
            knots = weighted_quantile(filled, self.spline_quantiles, weights)
            if len(np.unique(knots)) != 4:
                raise ValueError(f"{column} does not support four distinct spline knots")
            self.spline_knots_[column] = knots
            basis = restricted_cubic_spline_basis(filled, knots)
            center = np.average(basis, axis=0, weights=weights)
            variance = np.average((basis - center) ** 2, axis=0, weights=weights)
            scale = np.sqrt(variance)
            if not np.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError(f"{column} spline basis has zero or invalid variance")
            self.spline_centers_[column] = center
            self.spline_scales_[column] = scale

        self.imputation_values_["sex"] = _weighted_mode(
            X["sex"].astype("string"), weights, ["female", "male"]
        )
        if self.route != "age_sex":
            self.imputation_values_["smoking_status"] = _weighted_mode(
                X["smoking_status"].astype("string"),
                weights,
                ["never", "former", "current"],
            )
            for column in ("hypertension_dx", "diabetes_dx"):
                self.imputation_values_[column] = bool(
                    _weighted_mode(X[column].astype("boolean"), weights, [False, True])
                )
        if self.route == "minimal":
            self.imputation_values_["general_health"] = _weighted_mode(
                X["general_health"].astype("string"),
                weights,
                ["excellent", "very_good", "good", "fair", "poor"],
            )

        self.feature_names_out_ = self._feature_names()
        expected = ROUTE_PARAMETER_COUNTS[self.route]
        if len(self.feature_names_out_) != expected:
            raise RuntimeError(
                f"Route {self.route!r} produced {len(self.feature_names_out_)} "
                f"parameters instead of {expected}"
            )
        self.n_features_in_ = len(ROUTE_FEATURES[self.route])
        return self

    def _feature_names(self) -> np.ndarray:
        names = ["age", "age_rcs_1", "age_rcs_2", "sex_male"]
        if self.route != "age_sex":
            names.extend(
                [
                    "bmi",
                    "bmi_rcs_1",
                    "bmi_rcs_2",
                    "smoking_former",
                    "smoking_current",
                    "hypertension_yes",
                    "diabetes_yes",
                ]
            )
        if self.route == "minimal":
            names.extend(
                [
                    "health_very_good",
                    "health_good",
                    "health_fair",
                    "health_poor",
                ]
            )
        return np.asarray(names, dtype=object)

    @staticmethod
    def _require_levels(series: pd.Series, allowed: set[str], column: str) -> pd.Series:
        text = series.astype("string")
        unexpected = sorted(set(text.dropna().astype(str).unique()).difference(allowed))
        if unexpected:
            raise ValueError(f"{column} contains non-canonical categories: {unexpected}")
        return text

    def _numeric_with_imputation(self, frame: pd.DataFrame, column: str) -> pd.Series:
        numeric = pd.to_numeric(frame[column], errors="coerce").astype(float)
        malformed = frame[column].notna() & numeric.isna()
        if malformed.any():
            raise ValueError(f"{column} contains non-numeric values")
        return numeric.fillna(self.imputation_values_[column])

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(
            self,
            [
                "imputation_values_",
                "spline_knots_",
                "spline_centers_",
                "spline_scales_",
                "feature_names_out_",
            ],
        )
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MinimalPreprocessor requires a pandas DataFrame")
        missing = set(ROUTE_FEATURES[self.route]).difference(X.columns)
        if missing:
            raise ValueError(f"Route {self.route!r} is missing features: {sorted(missing)}")

        age = self._numeric_with_imputation(X, "age")
        age_basis = restricted_cubic_spline_basis(age.to_numpy(), self.spline_knots_["age"])
        columns = [
            (age_basis - self.spline_centers_["age"]) / self.spline_scales_["age"]
        ]
        sex = self._require_levels(X["sex"], {"female", "male"}, "sex").fillna(
            self.imputation_values_["sex"]
        )
        columns.append(sex.eq("male").to_numpy(dtype=np.float64).reshape(-1, 1))

        if self.route != "age_sex":
            bmi = self._numeric_with_imputation(X, "bmi")
            bmi_basis = restricted_cubic_spline_basis(
                bmi.to_numpy(), self.spline_knots_["bmi"]
            )
            columns.append(
                (bmi_basis - self.spline_centers_["bmi"]) / self.spline_scales_["bmi"]
            )
            smoking = self._require_levels(
                X["smoking_status"], {"never", "former", "current"}, "smoking_status"
            ).fillna(self.imputation_values_["smoking_status"])
            columns.extend(
                [
                    smoking.eq("former").to_numpy(dtype=np.float64).reshape(-1, 1),
                    smoking.eq("current").to_numpy(dtype=np.float64).reshape(-1, 1),
                ]
            )
            for column in ("hypertension_dx", "diabetes_dx"):
                boolean = X[column].astype("boolean").fillna(self.imputation_values_[column])
                columns.append(boolean.to_numpy(dtype=np.float64).reshape(-1, 1))
        if self.route == "minimal":
            health = self._require_levels(
                X["general_health"],
                {"excellent", "very_good", "good", "fair", "poor"},
                "general_health",
            ).fillna(self.imputation_values_["general_health"])
            for level in ("very_good", "good", "fair", "poor"):
                columns.append(health.eq(level).to_numpy(dtype=np.float64).reshape(-1, 1))

        transformed = np.column_stack(columns).astype(np.float64, copy=False)
        if transformed.shape[1] != ROUTE_PARAMETER_COUNTS[self.route]:
            raise RuntimeError("Transformed predictor matrix violates the parameter budget")
        if not np.isfinite(transformed).all():
            raise ValueError("Preprocessing produced non-finite values")
        return transformed

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        del input_features
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


__all__ = [
    "MinimalPreprocessor",
    "RCS_QUANTILES",
    "ROUTE_FEATURES",
    "ROUTE_PARAMETER_COUNTS",
    "restricted_cubic_spline_basis",
    "weighted_quantile",
]
