from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cvd_risk.modeling.unseal_validation import (
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    FROZEN_CUTOFF,
    OUTCOME_LOAD_COLUMNS,
    bootstrap_calibration_bands,
    calibration_bin_boundaries,
    join_outcomes_one_to_one,
    load_unseal_validation_config,
    percentile_interval_table,
    recalibration_in_the_large,
    run_authorized_unseal_validation,
    stratified_psu_metric_bootstrap,
    subgroup_eligibility_table,
    subgroup_point_metrics,
    weighted_original_metrics,
)
from cvd_risk.stage0 import _find_rscript


def _synthetic_frames(n_per_cluster: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    rows: list[dict[str, object]] = []
    for stratum in range(1, 5):
        for psu in (1, 2):
            for _within in range(n_per_cluster):
                index = len(rows)
                age = 40 + index % 40
                linear = -6.5 + 0.06 * (age - 40) + 0.30 * (psu - 1)
                risk = float(1.0 / (1.0 + np.exp(-linear)))
                rows.append(
                    {
                        "cohort": "synthetic",
                        "participant_id": f"synthetic-{index:04d}",
                        "survey_cycle": "",
                        "survey_year": 2010 + index % 5,
                        "strata": stratum,
                        "psu": psu,
                        "analysis_weight": float(1.0 + index % 7),
                        "predicted_risk": risk,
                        "linear_predictor": linear,
                        "secondary_screen_positive": risk >= FROZEN_CUTOFF,
                    }
                )
    predictions = pd.DataFrame(rows)
    probability = predictions["predicted_risk"].to_numpy(dtype=float)
    outcome_probability = 0.18 + 0.25 * probability / probability.max()
    outcome = rng.binomial(1, outcome_probability)
    # Guarantee both outcome classes within the synthetic fixture only.
    outcome[0] = 0
    outcome[1] = 1
    outcomes = pd.DataFrame(
        {
            "participant_id": predictions["participant_id"],
            "cvd_death_5y": outcome,
            "age": [40 + index % 40 for index in range(len(predictions))],
            "sex": ["female" if index % 2 == 0 else "male" for index in range(len(predictions))],
            "race_ethnicity": [
                "group_a" if index % 3 else "group_b" for index in range(len(predictions))
            ],
            "split": "synthetic_only",
        }
    )
    return predictions, outcomes


def test_preregistered_contract_is_exact_and_still_sealed() -> None:
    config = load_unseal_validation_config("configs/stage6_unseal_validation.yaml")

    assert config["status"] == "planned_outcomes_sealed"
    assert config["outcome_contract"]["outcome_columns"] == ["cvd_death_5y"]
    assert tuple(
        config["cohorts"]["nhis_temporal"]["outcome_load_allowlist"]
    ) == OUTCOME_LOAD_COLUMNS
    assert config["analysis"]["fixed_cutoff"]["value"] == FROZEN_CUTOFF
    assert config["analysis"]["cluster_bootstrap"]["repetitions"] == BOOTSTRAP_REPETITIONS
    assert config["analysis"]["cluster_bootstrap"]["seed"] == BOOTSTRAP_SEED
    assert config["authorization"]["explicit_user_permission_required"] is True


def test_one_to_one_join_preserves_prediction_order() -> None:
    predictions, outcomes = _synthetic_frames()
    shuffled = outcomes.sample(frac=1.0, random_state=1)

    joined = join_outcomes_one_to_one(predictions, shuffled)

    assert joined["participant_id"].tolist() == predictions["participant_id"].tolist()
    assert len(joined) == len(predictions)
    assert joined["cvd_death_5y"].isin([0, 1]).all()


@pytest.mark.parametrize("side", ["prediction", "outcome"])
def test_one_to_one_join_rejects_duplicate_ids(side: str) -> None:
    predictions, outcomes = _synthetic_frames()
    target = predictions if side == "prediction" else outcomes
    target.loc[1, "participant_id"] = target.loc[0, "participant_id"]

    with pytest.raises(ValueError, match="duplicates"):
        join_outcomes_one_to_one(predictions, outcomes)


def test_one_to_one_join_rejects_unmatched_ids() -> None:
    predictions, outcomes = _synthetic_frames()
    outcomes.loc[0, "participant_id"] = "not-in-predictions"

    with pytest.raises(ValueError, match="ID sets differ"):
        join_outcomes_one_to_one(predictions, outcomes)


def test_outcome_allowlist_rejects_raw_mortality_column() -> None:
    predictions, outcomes = _synthetic_frames()
    outcomes["MORTSTAT"] = 1

    with pytest.raises(ValueError, match="exact allowlist"):
        join_outcomes_one_to_one(predictions, outcomes)


def test_original_metrics_use_fixed_predictions_and_cutoff() -> None:
    predictions, outcomes = _synthetic_frames()
    joined = join_outcomes_one_to_one(predictions, outcomes)

    metrics = weighted_original_metrics(joined)

    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["average_precision"] <= 1.0
    assert metrics["brier_score"] >= 0.0
    assert metrics["sensitivity"] == pytest.approx(
        weighted_original_metrics(joined, cutoff=FROZEN_CUTOFF)["sensitivity"]
    )


def test_recalibration_in_the_large_is_offset_only_sensitivity() -> None:
    predictions, outcomes = _synthetic_frames()
    joined = join_outcomes_one_to_one(predictions, outcomes)

    intercept, recalibrated = recalibration_in_the_large(joined)
    weight = joined["analysis_weight"].to_numpy(dtype=float)
    target = joined["cvd_death_5y"].to_numpy(dtype=float)

    assert np.isfinite(intercept)
    assert np.average(recalibrated, weights=weight) == pytest.approx(
        np.average(target, weights=weight), abs=1.0e-10
    )
    assert not np.shares_memory(recalibrated, joined["predicted_risk"].to_numpy())


def test_cluster_bootstrap_is_stratified_psu_and_deterministic() -> None:
    predictions, outcomes = _synthetic_frames()
    joined = join_outcomes_one_to_one(predictions, outcomes)

    first = stratified_psu_metric_bootstrap(joined, repetitions=12, seed=20260714)
    second = stratified_psu_metric_bootstrap(joined, repetitions=12, seed=20260714)
    intervals = percentile_interval_table(
        weighted_original_metrics(joined),
        first,
        minimum_valid_replicates=8,
    )

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 12
    assert intervals["status"].eq("complete").all()


def test_calibration_bins_and_bootstrap_bands_use_fixed_boundaries() -> None:
    predictions, outcomes = _synthetic_frames()
    joined = join_outcomes_one_to_one(predictions, outcomes)
    boundaries = calibration_bin_boundaries(
        joined["predicted_risk"].to_numpy(dtype=float),
        joined["analysis_weight"].to_numpy(dtype=float),
    )

    table = bootstrap_calibration_bands(
        joined,
        boundaries,
        repetitions=10,
        seed=20260714,
        minimum_valid_replicates=6,
    )

    assert 2 <= table["actual_bins"].iloc[0] <= 10
    assert table["observed_valid_replicates"].ge(6).all()


def test_subgroup_thresholds_suppress_ineligible_synthetic_categories() -> None:
    predictions, outcomes = _synthetic_frames()
    joined = join_outcomes_one_to_one(predictions, outcomes)
    eligibility = subgroup_eligibility_table(
        joined,
        minimum_events=10_000,
        minimum_non_events=10_000,
    )

    table = subgroup_point_metrics(joined, eligibility)

    assert not table["eligible_for_performance"].any()
    assert table["roc_auc"].isna().all()
    assert table["status"].eq("counts_only_insufficient_events").all()


def test_preunseal_test_file_does_not_reference_real_data_paths() -> None:
    predictions, outcomes = _synthetic_frames()

    assert predictions["participant_id"].str.startswith("synthetic-").all()
    assert outcomes["split"].eq("synthetic_only").all()
    assert Path("data") not in [Path(value) for value in outcomes["participant_id"]]


def test_real_runner_refuses_before_separate_authorization(tmp_path: Path) -> None:
    missing_authorization = tmp_path / "authorization-does-not-exist.json"

    with pytest.raises(PermissionError, match="UNSEAL_NOT_AUTHORIZED"):
        run_authorized_unseal_validation(authorization_path=missing_authorization)


def test_r_design_risk_backend_uses_synthetic_labels_only(tmp_path: Path) -> None:
    rscript = _find_rscript()
    if rscript is None:
        pytest.skip("Rscript is unavailable")
    predictions, outcomes = _synthetic_frames()
    joined = join_outcomes_one_to_one(predictions, outcomes)
    design = joined.rename(
        columns={"cvd_death_5y": "outcome", "analysis_weight": "weight"}
    )[["outcome", "weight", "strata", "psu"]].assign(cohort="synthetic")
    input_path = tmp_path / "synthetic_design.csv"
    output_path = tmp_path / "synthetic_risk.csv"
    design.to_csv(input_path, index=False)

    completed = subprocess.run(
        [
            rscript,
            "scripts/r/stage6_design_risk_precision.R",
            str(input_path),
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    result = pd.read_csv(output_path)
    assert result.loc[0, "cohort"] == "synthetic"
    assert result.loc[0, "method"] == "survey_svyciprop_logit"
