from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cvd_risk.modeling.preprocessing import (
    ROUTE_PARAMETER_COUNTS,
    MinimalPreprocessor,
    restricted_cubic_spline_basis,
    weighted_quantile,
)


def _features(n: int = 60) -> pd.DataFrame:
    index = np.arange(n)
    frame = pd.DataFrame(
        {
            "age": 40 + index % 40,
            "sex": np.where(index % 2, "male", "female"),
            "bmi": 18.0 + index * 0.3,
            "smoking_status": np.asarray(["never", "former", "current"])[index % 3],
            "hypertension_dx": index % 4 == 0,
            "diabetes_dx": index % 7 == 0,
            "general_health": np.asarray(
                ["excellent", "very_good", "good", "fair", "poor"]
            )[index % 5],
        }
    )
    frame.loc[3, "bmi"] = np.nan
    frame.loc[5, "smoking_status"] = None
    frame.loc[7, "general_health"] = None
    return frame


def test_weighted_quantile_responds_to_training_weights() -> None:
    values = np.array([0.0, 10.0, 20.0])

    unweighted = weighted_quantile(values, [0.5], np.ones(3))[0]
    weighted = weighted_quantile(values, [0.5], np.array([1.0, 1.0, 20.0]))[0]

    assert unweighted == pytest.approx(10.0)
    assert weighted > unweighted


def test_restricted_cubic_spline_has_protocol_parameter_count() -> None:
    basis = restricted_cubic_spline_basis(
        np.array([35.0, 45.0, 65.0, 85.0]), np.array([40.0, 50.0, 70.0, 80.0])
    )

    assert basis.shape == (4, 3)
    assert np.isfinite(basis).all()


@pytest.mark.parametrize("route", list(ROUTE_PARAMETER_COUNTS))
def test_preprocessor_fits_imputation_and_exact_budget(route: str) -> None:
    frame = _features()
    weights = np.linspace(0.5, 2.0, len(frame))
    preprocessor = MinimalPreprocessor(route=route)

    transformed = preprocessor.fit_transform(frame, sample_weight=weights)

    assert transformed.shape == (len(frame), ROUTE_PARAMETER_COUNTS[route])
    assert len(preprocessor.get_feature_names_out()) == ROUTE_PARAMETER_COUNTS[route]
    assert np.isfinite(transformed).all()
    assert np.all(np.diff(preprocessor.spline_knots_["age"]) > 0)
    normalized = weights / weights.sum()
    assert np.average(transformed[:, :3], axis=0, weights=normalized) == pytest.approx(
        np.zeros(3), abs=1.0e-12
    )


def test_preprocessor_rejects_unknown_live_category() -> None:
    training = _features()
    preprocessor = MinimalPreprocessor(route="minimal").fit(training)
    invalid = training.copy()
    invalid.loc[0, "smoking_status"] = "occasional"

    with pytest.raises(ValueError, match="non-canonical"):
        preprocessor.transform(invalid)


def test_preprocessor_rejects_malformed_numeric_value() -> None:
    training = _features()
    preprocessor = MinimalPreprocessor(route="minimal").fit(training)
    invalid = training.copy()
    invalid["bmi"] = invalid["bmi"].astype(object)
    invalid.loc[0, "bmi"] = "unknown"

    with pytest.raises(ValueError, match="non-numeric"):
        preprocessor.transform(invalid)
