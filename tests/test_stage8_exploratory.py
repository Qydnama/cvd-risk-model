from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cvd_risk.modeling.stage8_exploratory import ClinicalCorePreprocessor, net_benefit


def _clinical_frame() -> pd.DataFrame:
    n = 20
    return pd.DataFrame(
        {
            "age": pd.Series(np.linspace(40, 79, n), dtype="Float64"),
            "sex": pd.Series((["female", "male"] * 10), dtype="string"),
            "bmi": pd.Series(np.linspace(18, 38, n), dtype="Float64"),
            "smoking_status": pd.Series(
                (["never", "former", "current", "never", "former"] * 4),
                dtype="string",
            ),
            "hypertension_dx": pd.Series(([False, True] * 10), dtype="boolean"),
            "diabetes_dx": pd.Series(([False, False, True, False] * 5), dtype="boolean"),
            "general_health": pd.Series(
                (["excellent", "very_good", "good", "fair", "poor"] * 4),
                dtype="string",
            ),
            "sbp_mean": np.linspace(90, 180, n),
            "total_cholesterol": np.linspace(130, 280, n),
            "hdl_cholesterol": np.linspace(30, 80, n),
        }
    )


def test_clinical_core_preprocessor_has_exact_20_parameter_specification() -> None:
    frame = _clinical_frame()
    processor = ClinicalCorePreprocessor().fit(frame, sample_weight=np.ones(len(frame)))
    matrix = processor.transform(frame)
    assert matrix.shape == (20, 20)
    assert len(processor.get_feature_names_out()) == 20
    assert np.isfinite(matrix).all()


def test_minimal_imputation_is_frozen_from_training_data_only() -> None:
    train = _clinical_frame()
    train.loc[0, "bmi"] = pd.NA
    processor = ClinicalCorePreprocessor().fit(train, sample_weight=np.ones(len(train)))
    fitted_value = processor.minimal_preprocessor_.imputation_values_["bmi"]
    validation = _clinical_frame().iloc[:2].copy()
    validation.loc[:, "bmi"] = pd.NA
    validation.loc[:, "age"] = [40.0, 79.0]
    processor.transform(validation)
    assert processor.minimal_preprocessor_.imputation_values_["bmi"] == fitted_value


def test_clinical_fields_are_never_imputed() -> None:
    frame = _clinical_frame()
    frame.loc[0, "sbp_mean"] = np.nan
    with pytest.raises(ValueError, match="cannot be imputed"):
        ClinicalCorePreprocessor().fit(frame, sample_weight=np.ones(len(frame)))


def test_weighted_net_benefit() -> None:
    outcome = np.array([1, 0, 0], dtype=np.int8)
    prediction = np.array([0.2, 0.1, 0.01])
    weight = np.array([2.0, 1.0, 1.0])
    observed = net_benefit(outcome, prediction, weight, 0.05)
    expected = 2.0 / 4.0 - 1.0 / 4.0 * 0.05 / 0.95
    assert np.isclose(observed, expected)
