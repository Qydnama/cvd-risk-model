"""Canonical feature contract shared by NHIS, NHANES, and future inference."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

FAST_FEATURES: tuple[str, ...] = (
    "age",
    "sex",
    "bmi",
    "smoking_status",
    "hypertension_dx",
    "diabetes_dx",
    "general_health",
)
SEX_CATEGORIES: tuple[str, ...] = ("male", "female")
SMOKING_CATEGORIES: tuple[str, ...] = ("never", "former", "current")
GENERAL_HEALTH_CATEGORIES: tuple[str, ...] = (
    "excellent",
    "very_good",
    "good",
    "fair",
    "poor",
)

CATEGORY_LEVELS: Mapping[str, tuple[str, ...]] = {
    "sex": SEX_CATEGORIES,
    "smoking_status": SMOKING_CATEGORIES,
    "general_health": GENERAL_HEALTH_CATEGORIES,
}


class CanonicalSchemaError(ValueError):
    """Raised when harmonized values violate the shared model interface."""


def calculate_bmi(*, height_cm: float, weight_kg: float) -> float:
    """Calculate BMI once from future web-form height and weight inputs."""

    if not np.isfinite(height_cm) or not np.isfinite(weight_kg):
        raise CanonicalSchemaError("Height and weight must be finite")
    if not 100 <= height_cm <= 250:
        raise CanonicalSchemaError("Height must be between 100 and 250 cm")
    if not 25 <= weight_kg <= 350:
        raise CanonicalSchemaError("Weight must be between 25 and 350 kg")
    bmi = weight_kg / (height_cm / 100.0) ** 2
    if not 10 <= bmi <= 99.8:
        raise CanonicalSchemaError("Calculated BMI is outside the supported range 10–99.8")
    return bmi


def validate_fast_feature_schema(
    frame: pd.DataFrame,
    *,
    allow_missing: bool = True,
) -> None:
    """Validate names, numeric domains, booleans, and canonical category levels."""

    missing_columns = set(FAST_FEATURES).difference(frame.columns)
    if missing_columns:
        raise CanonicalSchemaError(f"Missing canonical fast features: {sorted(missing_columns)}")
    for column, allowed in CATEGORY_LEVELS.items():
        category_observed = set(frame[column].dropna().astype(str).unique())
        unexpected = sorted(category_observed.difference(allowed))
        if unexpected:
            raise CanonicalSchemaError(f"{column} contains non-canonical categories: {unexpected}")
    for column in ("hypertension_dx", "diabetes_dx"):
        boolean_observed: set[Any] = set(frame[column].dropna().unique().tolist())
        if not boolean_observed.issubset({False, True}):
            raise CanonicalSchemaError(f"{column} must contain only boolean values")
    # NHANES contains measured BMI values above the NHIS/future-inference
    # support (including 130.21 in 2005–2006). Preserve them for an explicit
    # implausible-value audit rather than silently dropping or clipping them.
    for column, minimum, maximum in (("age", 0, 120), ("bmi", 7, 150)):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        malformed = frame[column].notna() & numeric.isna()
        if malformed.any():
            raise CanonicalSchemaError(f"{column} contains non-numeric values")
        finite = numeric.dropna().map(np.isfinite)
        if not finite.all():
            raise CanonicalSchemaError(f"{column} contains non-finite values")
        if not numeric.dropna().between(minimum, maximum, inclusive="both").all():
            raise CanonicalSchemaError(
                f"{column} is outside the canonical range {minimum}–{maximum}"
            )
    if not allow_missing and frame.loc[:, FAST_FEATURES].isna().any().any():
        raise CanonicalSchemaError("Canonical fast-model input contains missing values")
