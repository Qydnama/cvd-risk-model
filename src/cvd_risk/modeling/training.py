"""Stage-4 development-only EDA and baseline training orchestration."""

from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import sklearn  # type: ignore[import-untyped]
import yaml

from cvd_risk.modeling.baselines import binary_prediction_metrics, fit_nested_oof_baseline
from cvd_risk.modeling.preprocessing import (
    RCS_QUANTILES,
    ROUTE_PARAMETER_COUNTS,
)
from cvd_risk.provenance import sha256_file
from cvd_risk.schemas import FAST_FEATURES, validate_fast_feature_schema

STAGE3_REQUIRED_INPUTS = frozenset(
    {
        "nhis_cohort",
        "nhanes_cohort",
        "nhis_flow",
        "nhanes_flow",
        "experiment_config",
        "research_protocol",
        "feature_mapping",
        "harmonization_differences_input",
        "protocol_discrepancies",
    }
)
STAGE3_REQUIRED_OUTPUTS = frozenset(
    {
        "cohort_flow_nhis",
        "cohort_flow_nhanes",
        "event_counts",
        "missingness_by_source",
        "feature_availability",
        "data_discrepancies",
        "harmonization_differences",
        "race_ethnicity_event_counts",
        "protocol_discrepancies_table",
        "baseline_cvd_sensitivity",
        "survey_design_precision",
        "effective_sample_size",
        "sample_size_gate",
        "nhanes_design_collision_audit",
        "huq_hsd_disagreement",
        "stop_gate_table",
        "outcome_by_year_nhis_development",
        "age_distribution",
        "feasibility_report",
    }
)


def load_modeling_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Modeling config schema_version must be 1")
    if payload.get("stage") != "stage4_baselines":
        raise ValueError("Modeling config must target stage4_baselines")
    if tuple(payload["restricted_cubic_splines"]["quantiles"]) != RCS_QUANTILES:
        raise ValueError("Spline quantiles must remain [0.05, 0.35, 0.65, 0.95]")
    if payload.get("development_cohort") != "nhis_development":
        raise ValueError("Stage 4 may train only on NHIS development")
    configured = {entry["route"]: int(entry["predictor_parameters"]) for entry in payload["models"]}
    if configured != ROUTE_PARAMETER_COUNTS:
        raise ValueError(
            f"Model parameter budgets diverge from protocol: {configured} != "
            f"{ROUTE_PARAMETER_COUNTS}"
        )
    if payload["calibration"].get("methods") != ["none", "sigmoid"]:
        raise ValueError("Stage 4 must compare uncalibrated and sigmoid calibration")
    return payload


def _verify_bound_entries(entries: Mapping[str, Any], required: frozenset[str]) -> None:
    missing = required.difference(entries)
    if missing:
        raise ValueError(f"Stop-gate provenance is missing bindings: {sorted(missing)}")
    for name in required:
        entry = entries[name]
        if not isinstance(entry, Mapping) or "path" not in entry or "sha256" not in entry:
            raise ValueError(f"Stop-gate binding {name!r} is malformed")
        path = Path(str(entry["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"Audited file {name!r} is missing: {path}")
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Audited file {name!r} changed after Stage 3: {path}")


def verify_stage3_training_gate(
    experiment_config_path: str | Path,
    status_path: str | Path,
) -> dict[str, Any]:
    """Require an intact PASS gate without exposing held-out individual labels."""

    path = Path(status_path)
    if not path.is_file():
        raise FileNotFoundError("Stage 3 stop-gate status is absent")
    status = json.loads(path.read_text(encoding="utf-8"))
    if status.get("schema_version") != 1 or status.get("ready_for_modeling") is not True:
        raise ValueError(status.get("summary", "Stage 3 feasibility gate did not pass"))
    if status.get("heldout_race_ethnicity_labels_sealed") is not True:
        raise ValueError("Held-out subgroup labels are not recorded as sealed")
    if status.get("protocol_discrepancies_ready") is not True:
        raise ValueError("Blocking protocol discrepancies remain unresolved")
    inputs = status.get("inputs")
    outputs = status.get("outputs")
    if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
        raise ValueError("Stop-gate provenance inputs/outputs must be mappings")
    audited_config = Path(str(inputs["experiment_config"]["path"])).resolve()
    if Path(experiment_config_path).resolve() != audited_config:
        raise ValueError("Training config differs from the Stage 3 experiment config")
    _verify_bound_entries(inputs, STAGE3_REQUIRED_INPUTS)
    _verify_bound_entries(outputs, STAGE3_REQUIRED_OUTPUTS)
    return status


def load_nhis_development(path: str | Path) -> pd.DataFrame:
    """Read only development rows and columns; held-out outcomes never enter memory."""

    columns = [
        "participant_id",
        "split",
        "survey_year",
        "strata",
        "psu",
        "analysis_weight",
        "cvd_death_5y",
        *FAST_FEATURES,
    ]
    frame = pd.read_parquet(
        path,
        columns=columns,
        filters=[("split", "==", "development")],
    )
    if frame.empty or not frame["split"].eq("development").all():
        raise ValueError("Development-only Parquet filter failed")
    validate_fast_feature_schema(frame, allow_missing=True)
    if frame["cvd_death_5y"].isna().any():
        raise ValueError("NHIS development contains an unknown primary outcome")
    weights = frame["analysis_weight"].astype(float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("NHIS development survey weights must be finite and positive")
    return frame.reset_index(drop=True)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights))


def development_description(frame: pd.DataFrame) -> pd.DataFrame:
    """Return weighted and unweighted predictor descriptions without held-out data."""

    weights = frame["analysis_weight"].to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for feature in ("age", "bmi"):
        numeric = pd.to_numeric(frame[feature], errors="coerce")
        known = numeric.notna().to_numpy()
        observed = numeric.loc[known].to_numpy(dtype=np.float64)
        observed_weights = weights[known]
        rows.append(
            {
                "feature": feature,
                "category": pd.NA,
                "n_known": int(known.sum()),
                "unweighted_value": float(observed.mean()),
                "weighted_value": _weighted_mean(observed, observed_weights),
                "minimum": float(observed.min()),
                "maximum": float(observed.max()),
                "statistic": "mean",
            }
        )
    categorical = (
        "sex",
        "smoking_status",
        "hypertension_dx",
        "diabetes_dx",
        "general_health",
    )
    for feature in categorical:
        known = frame[feature].notna().to_numpy()
        known_weight = float(weights[known].sum())
        for category in sorted(frame.loc[known, feature].astype(str).unique()):
            selected = frame[feature].astype(str).eq(category).to_numpy() & known
            rows.append(
                {
                    "feature": feature,
                    "category": category,
                    "n_known": int(known.sum()),
                    "unweighted_value": float(selected.sum() / known.sum()),
                    "weighted_value": float(weights[selected].sum() / known_weight),
                    "minimum": pd.NA,
                    "maximum": pd.NA,
                    "statistic": "proportion",
                }
            )
    return pd.DataFrame(rows)


def development_missingness(frame: pd.DataFrame) -> pd.DataFrame:
    weights = frame["analysis_weight"].to_numpy(dtype=np.float64)
    total_weight = weights.sum()
    rows = []
    for feature in FAST_FEATURES:
        missing = frame[feature].isna().to_numpy()
        rows.append(
            {
                "feature": feature,
                "n": len(frame),
                "n_missing": int(missing.sum()),
                "missing_fraction_unweighted": float(missing.mean()),
                "missing_fraction_weighted": float(weights[missing].sum() / total_weight),
            }
        )
    return pd.DataFrame(rows)


def implausible_value_table(frame: pd.DataFrame) -> pd.DataFrame:
    checks = {
        "age_outside_protocol_40_79": ~frame["age"].between(40, 79, inclusive="both"),
        "bmi_below_future_input_10": frame["bmi"].lt(10),
        "bmi_above_future_input_99_8": frame["bmi"].gt(99.8),
        "non_finite_age": frame["age"].notna()
        & ~np.isfinite(pd.to_numeric(frame["age"], errors="coerce")),
        "non_finite_bmi": frame["bmi"].notna()
        & ~np.isfinite(pd.to_numeric(frame["bmi"], errors="coerce")),
    }
    return pd.DataFrame(
        [
            {"check": name, "n_flagged": int(flag.fillna(False).sum()), "n_total": len(frame)}
            for name, flag in checks.items()
        ]
    )


def predictor_drift_by_year(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe within-development predictor drift without outcome cross-tabs."""

    rows: list[dict[str, Any]] = []
    for year, group in frame.groupby("survey_year", observed=True):
        year_value = int(str(year))
        weights = group["analysis_weight"].to_numpy(dtype=np.float64)
        for feature in ("age", "bmi"):
            numeric = pd.to_numeric(group[feature], errors="coerce")
            known = numeric.notna().to_numpy()
            rows.append(
                {
                    "survey_year": year_value,
                    "feature": feature,
                    "category": pd.NA,
                    "statistic": "weighted_mean",
                    "value": _weighted_mean(
                        numeric.loc[known].to_numpy(dtype=np.float64), weights[known]
                    ),
                }
            )
            rows.append(
                {
                    "survey_year": year_value,
                    "feature": feature,
                    "category": pd.NA,
                    "statistic": "missing_fraction_weighted",
                    "value": float(weights[~known].sum() / weights.sum()),
                }
            )
        for feature in (
            "sex",
            "smoking_status",
            "hypertension_dx",
            "diabetes_dx",
            "general_health",
        ):
            known = group[feature].notna().to_numpy()
            denominator = weights[known].sum()
            for category in sorted(group.loc[known, feature].astype(str).unique()):
                selected = group[feature].astype(str).eq(category).to_numpy() & known
                rows.append(
                    {
                        "survey_year": year_value,
                        "feature": feature,
                        "category": category,
                        "statistic": "weighted_proportion",
                        "value": float(weights[selected].sum() / denominator),
                    }
                )
    return pd.DataFrame(rows)


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unavailable", True
    return {"base_commit": commit, "working_tree_dirty": dirty}


def _write_report(
    path: Path,
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    folds: pd.DataFrame,
    missingness: pd.DataFrame,
    implausible: pd.DataFrame,
    manifest_path: Path,
) -> None:
    weighted = metrics.loc[metrics["weighting"].eq("survey_weighted")]
    lines = [
        "# Stage 4: development EDA and logistic baselines",
        "",
        "## Scope and leakage policy",
        "",
        "Only NHIS development 1997–2009 was loaded. NHIS temporal and NHANES "
        "individual outcomes were not loaded, scored, or used for model decisions.",
        "",
        f"- Participants: {len(frame):,}",
        f"- Five-year CVD deaths: {int(frame['cvd_death_5y'].sum()):,}",
        "- Outer validation: StratifiedGroupKFold grouped by survey year",
        "- Calibration: sigmoid/Platt with nested grouped cross-fitting",
        "- Survey weights: normalized to mean 1 separately in every training fold",
        "- Imputation and weighted spline knots: fitted separately in every training fold",
        "- This stage does not create an external-validation lock or choose a final model",
        "",
        "## Survey-weighted out-of-fold metrics",
        "",
        weighted.to_markdown(index=False),
        "",
        "## Outer folds",
        "",
        folds.to_markdown(index=False),
        "",
        "## Development missingness",
        "",
        missingness.to_markdown(index=False),
        "",
        "## Implausible-value audit",
        "",
        implausible.to_markdown(index=False),
        "",
        "## Status",
        "",
        "Baseline candidates were fitted reproducibly. Random Forest, boosting, final "
        "model selection, threshold policy, and external-validation lock remain pending.",
        "",
        f"Machine-readable provenance: `{manifest_path.as_posix()}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_stage4_baselines(
    *,
    experiment_config_path: str | Path,
    modeling_config_path: str | Path,
    stop_gate_path: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    """Run development EDA and nested OOF logistic baselines."""

    modeling_path = Path(modeling_config_path)
    model_config = load_modeling_config(modeling_path)
    status = verify_stage3_training_gate(experiment_config_path, stop_gate_path)
    cohort_path = Path(status["inputs"]["nhis_cohort"]["path"])
    frame = load_nhis_development(cohort_path)
    event_table = pd.read_csv(status["outputs"]["event_counts"]["path"])
    audited = event_table.loc[event_table["cohort"].eq("nhis_development")].iloc[0]
    if len(frame) != int(audited["n_participants"]) or int(
        frame["cvd_death_5y"].sum()
    ) != int(audited["n_events"]):
        raise ValueError("Development rows differ from the sealed aggregate Stage 3 audit")

    outputs = model_config["outputs"]
    artifact_dir = Path(outputs["artifact_dir"])
    known_output_paths = [
        Path(outputs[key])
        for key in (
            "report",
            "comparison_table",
            "fold_table",
            "development_description",
            "development_missingness",
            "predictor_drift",
            "implausible_values",
            "coefficients",
        )
    ]
    artifact_paths = [
        artifact_dir / "oof_predictions.parquet",
        artifact_dir / "manifest.json",
        *[artifact_dir / f"{entry['name']}.joblib" for entry in model_config["models"]],
    ]
    existing = [path for path in [*known_output_paths, *artifact_paths] if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Stage 4 outputs already exist; pass --force to replace only known outputs: "
            + ", ".join(map(str, existing))
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for path in known_output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    description = development_description(frame)
    missingness = development_missingness(frame)
    implausible = implausible_value_table(frame)
    drift = predictor_drift_by_year(frame)
    description.to_csv(outputs["development_description"], index=False)
    missingness.to_csv(outputs["development_missingness"], index=False)
    implausible.to_csv(outputs["implausible_values"], index=False)
    drift.to_csv(outputs["predictor_drift"], index=False)

    cv = model_config["cross_validation"]
    calibration = model_config["calibration"]
    logistic = model_config["logistic_regression"]
    seed = int(model_config["seed"])
    quantiles = tuple(
        float(value) for value in model_config["restricted_cubic_splines"]["quantiles"]
    )
    outcome = str(model_config["outcome"])
    weight_column = str(model_config["weight"])
    group_column = str(model_config["group"])
    y = frame[outcome].to_numpy(dtype=np.int8)
    weights = frame[weight_column].to_numpy(dtype=np.float64)
    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    shared_folds: pd.DataFrame | None = None
    model_entries: list[dict[str, Any]] = []

    for entry in model_config["models"]:
        name = str(entry["name"])
        route = str(entry["route"])
        fitted, predictions, folds = fit_nested_oof_baseline(
            frame,
            name=name,
            route=route,
            outcome_column=outcome,
            weight_column=weight_column,
            group_column=group_column,
            outer_splits=int(cv["n_splits"]),
            inner_splits=int(calibration["inner_splits"]),
            spline_quantiles=quantiles,
            logistic_settings=logistic,
            probability_clip=float(calibration["probability_clip"]),
            seed=seed,
        )
        if shared_folds is None:
            shared_folds = folds
        elif not shared_folds.equals(folds):
            raise RuntimeError("Baseline candidates did not use identical outer folds")
        prediction_frames.append(predictions)
        for calibration_name, prediction_column in (
            ("none", "prediction_uncalibrated"),
            ("sigmoid", "prediction_sigmoid"),
        ):
            for weighting, metric_weight in (
                ("survey_weighted", weights),
                ("unweighted_sensitivity", np.ones(len(frame), dtype=np.float64)),
            ):
                values = binary_prediction_metrics(
                    y,
                    predictions[prediction_column].to_numpy(),
                    metric_weight,
                    probability_clip=float(calibration["probability_clip"]),
                )
                metric_rows.append(
                    {
                        "model": name,
                        "route": route,
                        "predictor_parameters": int(entry["predictor_parameters"]),
                        "calibration": calibration_name,
                        "weighting": weighting,
                        "n": len(frame),
                        "events": int(y.sum()),
                        **values,
                    }
                )
        feature_names = fitted.preprocessor.get_feature_names_out()
        for feature, coefficient in zip(feature_names, fitted.estimator.coef_[0], strict=True):
            coefficient_rows.append(
                {
                    "model": name,
                    "route": route,
                    "term": str(feature),
                    "coefficient": float(coefficient),
                    "odds_ratio": float(np.exp(coefficient)),
                }
            )
        coefficient_rows.append(
            {
                "model": name,
                "route": route,
                "term": "intercept",
                "coefficient": float(fitted.estimator.intercept_[0]),
                "odds_ratio": float(np.exp(fitted.estimator.intercept_[0])),
            }
        )
        model_path = artifact_dir / f"{name}.joblib"
        joblib.dump(fitted, model_path)
        model_entries.append(
            {
                "name": name,
                "route": route,
                "predictor_parameters": int(entry["predictor_parameters"]),
                "path": str(model_path.resolve()),
                "sha256": sha256_file(model_path),
                "feature_names": feature_names.tolist(),
                "spline_knots": {
                    feature: knots.tolist()
                    for feature, knots in fitted.preprocessor.spline_knots_.items()
                },
                "imputation_values": fitted.preprocessor.imputation_values_,
            }
        )

    assert shared_folds is not None
    metrics = pd.DataFrame(metric_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    metrics.to_csv(outputs["comparison_table"], index=False)
    shared_folds.to_csv(outputs["fold_table"], index=False)
    coefficients.to_csv(outputs["coefficients"], index=False)

    identifiers = frame.loc[
        :, ["participant_id", "survey_year", "strata", "psu", weight_column, outcome]
    ].reset_index(drop=True)
    oof_parts = [
        pd.concat([identifiers, item.reset_index(drop=True)], axis=1)
        for item in prediction_frames
    ]
    oof = pd.concat(oof_parts, ignore_index=True)
    oof_path = artifact_dir / "oof_predictions.parquet"
    oof.to_parquet(oof_path, index=False)

    source_paths = [
        Path(__file__),
        Path(__file__).with_name("baselines.py"),
        Path(__file__).with_name("preprocessing.py"),
        Path(experiment_config_path),
        modeling_path,
        Path("pyproject.toml"),
        Path("uv.lock"),
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "stage": "stage4_baselines",
        "status": "BASELINES_FITTED_NOT_MODEL_LOCKED",
        "scope": "NHIS development 1997-2009 only",
        "heldout_outcomes_accessed": False,
        "final_model_selected": False,
        "external_validation_lock_created": False,
        "seed": seed,
        "n": len(frame),
        "events": int(y.sum()),
        "input_cohort": {
            "path": str(cohort_path.resolve()),
            "sha256": sha256_file(cohort_path),
        },
        "source_bindings": {
            str(path.as_posix()): sha256_file(path) for path in source_paths
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "git": _git_metadata(),
        "cross_validation": {
            **cv,
            "group": group_column,
            "nested_calibration_splits": int(calibration["inner_splits"]),
        },
        "models": model_entries,
        "outputs": {
            "oof_predictions": {
                "path": str(oof_path.resolve()),
                "sha256": sha256_file(oof_path),
            },
            **{
                key: {
                    "path": str(Path(outputs[key]).resolve()),
                    "sha256": sha256_file(outputs[key]),
                }
                for key in (
                    "comparison_table",
                    "fold_table",
                    "development_description",
                    "development_missingness",
                    "predictor_drift",
                    "implausible_values",
                    "coefficients",
                )
            },
        },
    }
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    _write_report(
        Path(outputs["report"]),
        frame=frame,
        metrics=metrics,
        folds=shared_folds,
        missingness=missingness,
        implausible=implausible,
        manifest_path=manifest_path,
    )
    manifest["outputs"]["report"] = {
        "path": str(Path(outputs["report"]).resolve()),
        "sha256": sha256_file(outputs["report"]),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "development_description",
    "development_missingness",
    "implausible_value_table",
    "load_modeling_config",
    "load_nhis_development",
    "predictor_drift_by_year",
    "run_stage4_baselines",
    "verify_stage3_training_gate",
]
