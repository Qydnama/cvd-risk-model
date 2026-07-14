from __future__ import annotations

import pandas as pd
import pytest

from cvd_risk.schemas import (
    CanonicalSchemaError,
    calculate_bmi,
    validate_fast_feature_schema,
)


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [60],
            "sex": ["female"],
            "bmi": [25.0],
            "smoking_status": ["former"],
            "hypertension_dx": [True],
            "diabetes_dx": [False],
            "general_health": ["good"],
        }
    )


def test_calculate_bmi_for_future_eight_field_form() -> None:
    assert calculate_bmi(height_cm=180, weight_kg=81) == pytest.approx(25.0)


def test_validate_shared_fast_schema() -> None:
    validate_fast_feature_schema(_valid_frame())


def test_general_health_order_uses_only_canonical_labels() -> None:
    frame = _valid_frame()
    frame["general_health"] = "average"
    with pytest.raises(CanonicalSchemaError, match="general_health"):
        validate_fast_feature_schema(frame)
