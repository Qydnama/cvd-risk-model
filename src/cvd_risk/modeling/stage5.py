"""Stage 5 tree-candidate orchestration and development-only comparison."""

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
import xgboost
import yaml

from cvd_risk.modeling.baselines import binary_prediction_metrics
from cvd_risk.modeling.ml_candidates import fit_nested_ml_candidate
from cvd_risk.modeling.preprocessing import RCS_QUANTILES
from cvd_risk.modeling.training import load_nhis_development
from cvd_risk.provenance import sha256_file


def load_stage5_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Stage 5 config schema_version must be 1")
    if payload.get("stage") != "stage5_ml_candidates":
        raise ValueError("Stage 5 config has an unexpected stage identifier")
    if payload.get("development_cohort") != "nhis_development":
        raise ValueError("Stage 5 may train only on NHIS development")
    if payload.get("route") != "minimal":
        raise ValueError("Stage 5 tree candidates must use the locked minimal route")
    if tuple(payload["preprocessing"]["restricted_cubic_spline_quantiles"]) != RCS_QUANTILES:
        raise ValueError("Stage 5 spline quantiles diverge from Stage 4")
    tuning = payload["inner_tuning"]
    if tuning.get("primary_metric") != "survey_weighted_brier":
        raise ValueError("Inner tuning must minimize survey-weighted Brier score")
    if tuning.get("final_hyperparameters") != "lowest_mean_inner_score_across_outer_folds":
        raise ValueError("Unsupported final hyperparameter aggregation rule")
    expected = {
        "RandomForest": "RandomForestClassifier",
        "XGBoost": "XGBClassifier",
    }
    observed = {entry["name"]: entry["estimator"] for entry in payload["models"]}
    if observed != expected:
        raise ValueError(f"Stage 5 requires exactly one RF and one XGBoost candidate: {observed}")
    for entry in payload["models"]:
        if len(entry.get("candidates", [])) < 2:
            raise ValueError(f"{entry['name']} requires at least two tuning candidates")
    return payload


def verify_stage4_artifacts(manifest_path: str | Path) -> dict[str, Any]:
    """Verify every Stage 4 source/output/model binding before Stage 5."""

    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError("Stage 4 baseline manifest is absent")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required_flags = {
        "schema_version": 1,
        "stage": "stage4_baselines",
        "status": "BASELINES_FITTED_NOT_MODEL_LOCKED",
        "heldout_outcomes_accessed": False,
        "final_model_selected": False,
        "external_validation_lock_created": False,
    }
    for field, expected in required_flags.items():
        if manifest.get(field) != expected:
            raise ValueError(f"Stage 4 manifest {field!r} must be {expected!r}")
    for name, digest in manifest.get("source_bindings", {}).items():
        source = Path(name)
        if not source.is_file() or sha256_file(source) != digest:
            raise ValueError(f"Stage 4 source binding changed: {source}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("Stage 4 output bindings are missing")
    for name, entry in outputs.items():
        output = Path(str(entry["path"]))
        if not output.is_file() or sha256_file(output) != entry["sha256"]:
            raise ValueError(f"Stage 4 output binding changed: {name}")
    models = manifest.get("models")
    if not isinstance(models, list) or len(models) != 3:
        raise ValueError("Stage 4 must provide three verified baseline models")
    for entry in models:
        model_path = Path(str(entry["path"]))
        if not model_path.is_file() or sha256_file(model_path) != entry["sha256"]:
            raise ValueError(f"Stage 4 model binding changed: {entry.get('name')}")
    return manifest


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


def _metric_rows(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    route: str,
    probability_clip: float,
) -> list[dict[str, Any]]:
    y = frame["cvd_death_5y"].to_numpy(dtype=np.int8)
    weights = frame["analysis_weight"].to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for calibration, column in (
        ("none", "prediction_uncalibrated"),
        ("sigmoid", "prediction_sigmoid"),
    ):
        for weighting, metric_weight in (
            ("survey_weighted", weights),
            ("unweighted_sensitivity", np.ones(len(frame), dtype=np.float64)),
        ):
            metrics = binary_prediction_metrics(
                y,
                predictions[column].to_numpy(dtype=np.float64),
                metric_weight,
                probability_clip=probability_clip,
            )
            rows.append(
                {
                    "model": str(predictions["model"].iloc[0]),
                    "route": route,
                    "predictor_parameters": pd.NA,
                    "calibration": calibration,
                    "weighting": weighting,
                    "n": len(frame),
                    "events": int(y.sum()),
                    **metrics,
                    "model_family": "tree_ml",
                    "candidate_status": "development_candidate_not_locked",
                }
            )
    return rows


def _fold_metric_rows(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    probability_clip: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold, fold_prediction in predictions.groupby("fold", observed=True):
        index = fold_prediction.index.to_numpy(dtype=int)
        y = frame.loc[index, "cvd_death_5y"].to_numpy(dtype=np.int8)
        weights = frame.loc[index, "analysis_weight"].to_numpy(dtype=np.float64)
        for calibration, column in (
            ("none", "prediction_uncalibrated"),
            ("sigmoid", "prediction_sigmoid"),
        ):
            metrics = binary_prediction_metrics(
                y,
                fold_prediction[column].to_numpy(dtype=np.float64),
                weights,
                probability_clip=probability_clip,
            )
            rows.append(
                {
                    "model": str(fold_prediction["model"].iloc[0]),
                    "fold": int(str(fold)),
                    "calibration": calibration,
                    "n": len(index),
                    "events": int(y.sum()),
                    **metrics,
                }
            )
    return rows


def _write_report(
    path: Path,
    *,
    comparison: pd.DataFrame,
    tuning: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    manifest_path: Path,
) -> None:
    primary = comparison.loc[
        comparison["weighting"].eq("survey_weighted")
        & comparison["calibration"].eq("sigmoid")
    ].sort_values("brier_score", kind="mergesort")
    stability = (
        fold_metrics.loc[fold_metrics["calibration"].eq("sigmoid")]
        .groupby("model", observed=True)
        .agg(
            brier_mean=("brier_score", "mean"),
            brier_sd=("brier_score", "std"),
            auc_mean=("roc_auc", "mean"),
            auc_sd=("roc_auc", "std"),
        )
        .reset_index()
    )
    selected = tuning.loc[tuning["selected_for_final_fit"].astype(bool), [
        "model",
        "candidate_id",
        "parameters",
    ]].drop_duplicates()
    lines = [
        "# Stage 5: Random Forest and XGBoost development comparison",
        "",
        "## Scope",
        "",
        "Only NHIS development 1997–2009 was used. Hyperparameters were selected "
        "inside each outer fold by three-fold grouped tuning on survey-weighted Brier "
        "score. Sigmoid calibration used inner out-of-fold predictions.",
        "",
        "NHIS temporal and NHANES individual outcomes were not loaded. No final model "
        "was selected and no external-validation lock was created.",
        "",
        "## Survey-weighted sigmoid OOF comparison",
        "",
        primary.to_markdown(index=False),
        "",
        "## Fold stability",
        "",
        stability.to_markdown(index=False),
        "",
        "## Hyperparameters selected for full-development candidate artifacts",
        "",
        selected.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "These are development comparisons, not temporal or external validation. "
        "The lowest development Brier score does not automatically authorize choosing "
        "a final model; discrimination, calibration, stability, and simplicity must be "
        "reviewed before Stage 5a and model lock.",
        "",
        f"Machine-readable provenance: `{manifest_path.as_posix()}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_stage5_ml_candidates(
    *,
    stage5_config_path: str | Path,
    stage4_manifest_path: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    config_path = Path(stage5_config_path)
    config = load_stage5_config(config_path)
    stage4_path = Path(stage4_manifest_path)
    stage4 = verify_stage4_artifacts(stage4_path)
    cohort_path = Path(stage4["input_cohort"]["path"])
    if sha256_file(cohort_path) != stage4["input_cohort"]["sha256"]:
        raise ValueError("NHIS cohort changed after Stage 4")
    frame = load_nhis_development(cohort_path)
    if len(frame) != int(stage4["n"]) or int(frame["cvd_death_5y"].sum()) != int(
        stage4["events"]
    ):
        raise ValueError("NHIS development no longer matches Stage 4")

    outputs = config["outputs"]
    artifact_dir = Path(outputs["artifact_dir"])
    report_paths = [
        Path(outputs[key])
        for key in (
            "report",
            "comparison_table",
            "tuning_table",
            "fold_metrics_table",
            "feature_importance_table",
        )
    ]
    artifact_paths = [
        artifact_dir / "manifest.json",
        artifact_dir / "oof_predictions.parquet",
        *[artifact_dir / f"{entry['name']}.joblib" for entry in config["models"]],
    ]
    existing = [path for path in [*report_paths, *artifact_paths] if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Stage 5 outputs already exist; pass --force to replace known outputs: "
            + ", ".join(map(str, existing))
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for path in report_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    outer = config["outer_cross_validation"]
    inner = config["inner_tuning"]
    clip = float(config["calibration"]["probability_clip"])
    quantiles = tuple(
        float(value)
        for value in config["preprocessing"]["restricted_cubic_spline_quantiles"]
    )
    seed = int(config["seed"])
    all_predictions: list[pd.DataFrame] = []
    all_tuning: list[pd.DataFrame] = []
    fold_selection: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    model_bindings: list[dict[str, Any]] = []
    stage4_oof_path = Path(stage4["outputs"]["oof_predictions"]["path"])
    stage4_oof = pd.read_parquet(stage4_oof_path)
    baseline_models = set(stage4_oof["model"].astype(str).unique())
    if baseline_models != {
        "AgeSexLogistic",
        "LogisticRegressionNoGeneralHealth",
        "LogisticRegression",
    }:
        raise ValueError(f"Unexpected Stage 4 OOF models: {sorted(baseline_models)}")
    reference_oof = stage4_oof.loc[
        stage4_oof["model"].eq("AgeSexLogistic")
    ].reset_index(drop=True)
    if not reference_oof["participant_id"].astype(str).equals(
        frame["participant_id"].astype(str)
    ):
        raise ValueError("Stage 4 OOF participant order differs from development cohort")
    for baseline_name in sorted(baseline_models):
        baseline_prediction = stage4_oof.loc[
            stage4_oof["model"].eq(baseline_name),
            [
                "model",
                "fold",
                "prediction_uncalibrated",
                "prediction_sigmoid",
            ],
        ].reset_index(drop=True)
        fold_metric_rows.extend(
            _fold_metric_rows(frame, baseline_prediction, probability_clip=clip)
        )

    for entry in config["models"]:
        fitted, predictions, tuning, selected_folds = fit_nested_ml_candidate(
            frame,
            name=str(entry["name"]),
            estimator_name=str(entry["estimator"]),
            common_parameters=dict(entry["common_parameters"]),
            candidates=[dict(candidate) for candidate in entry["candidates"]],
            outcome_column=str(config["outcome"]),
            weight_column=str(config["weight"]),
            group_column=str(config["group"]),
            outer_splits=int(outer["n_splits"]),
            inner_splits=int(inner["n_splits"]),
            spline_quantiles=quantiles,
            probability_clip=clip,
            seed=seed,
        )
        all_predictions.append(predictions)
        if not predictions["fold"].reset_index(drop=True).equals(
            reference_oof["fold"].astype(predictions["fold"].dtype)
        ):
            raise RuntimeError(f"{entry['name']} outer folds differ from Stage 4")
        tuning = tuning.copy()
        tuning["parameters"] = tuning["parameters"].map(
            lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
        )
        all_tuning.append(tuning)
        fold_selection.append(selected_folds)
        metric_rows.extend(_metric_rows(frame, predictions, route="minimal", probability_clip=clip))
        fold_metric_rows.extend(_fold_metric_rows(frame, predictions, probability_clip=clip))
        feature_names = fitted.preprocessor.get_feature_names_out()
        importances = np.asarray(fitted.estimator.feature_importances_, dtype=np.float64)
        for feature, importance in zip(feature_names, importances, strict=True):
            importance_rows.append(
                {
                    "model": fitted.name,
                    "feature": str(feature),
                    "impurity_importance": float(importance),
                    "status": "diagnostic_not_final_interpretation",
                }
            )
        model_path = artifact_dir / f"{fitted.name}.joblib"
        joblib.dump(fitted, model_path)
        model_bindings.append(
            {
                "name": fitted.name,
                "estimator": fitted.estimator_name,
                "parameters": fitted.parameters,
                "path": str(model_path.resolve()),
                "sha256": sha256_file(model_path),
                "feature_names": feature_names.tolist(),
                "spline_knots": {
                    feature: knots.tolist()
                    for feature, knots in fitted.preprocessor.spline_knots_.items()
                },
            }
        )

    identifiers = frame.loc[
        :, ["participant_id", "survey_year", "strata", "psu", "analysis_weight", "cvd_death_5y"]
    ]
    oof_parts = [
        pd.concat([identifiers.reset_index(drop=True), item.reset_index(drop=True)], axis=1)
        for item in all_predictions
    ]
    oof = pd.concat(oof_parts, ignore_index=True)
    oof_path = artifact_dir / "oof_predictions.parquet"
    oof.to_parquet(oof_path, index=False)

    tree_metrics = pd.DataFrame(metric_rows)
    baseline_metrics_path = Path(stage4["outputs"]["comparison_table"]["path"])
    baseline_metrics = pd.read_csv(baseline_metrics_path)
    baseline_metrics["model_family"] = "logistic"
    baseline_metrics["candidate_status"] = "development_candidate_not_locked"
    comparison = pd.concat([baseline_metrics, tree_metrics], ignore_index=True)
    comparison.to_csv(outputs["comparison_table"], index=False)
    tuning_table = pd.concat(all_tuning, ignore_index=True)
    tuning_table.to_csv(outputs["tuning_table"], index=False)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    fold_metrics.to_csv(outputs["fold_metrics_table"], index=False)
    feature_importance = pd.DataFrame(importance_rows)
    feature_importance.to_csv(outputs["feature_importance_table"], index=False)

    source_paths = [
        Path(__file__),
        Path(__file__).with_name("ml_candidates.py"),
        Path(__file__).with_name("preprocessing.py"),
        Path(__file__).with_name("baselines.py"),
        config_path,
        stage4_path,
        Path("pyproject.toml"),
        Path("uv.lock"),
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "stage": "stage5_ml_candidates",
        "status": "ML_CANDIDATES_FITTED_NOT_MODEL_LOCKED",
        "scope": "NHIS development 1997-2009 only",
        "heldout_outcomes_accessed": False,
        "final_model_selected": False,
        "external_validation_lock_created": False,
        "seed": seed,
        "n": len(frame),
        "events": int(frame["cvd_death_5y"].sum()),
        "input_cohort": {
            "path": str(cohort_path.resolve()),
            "sha256": sha256_file(cohort_path),
        },
        "stage4_manifest": {
            "path": str(stage4_path.resolve()),
            "sha256": sha256_file(stage4_path),
        },
        "source_bindings": {
            str(path.as_posix()): sha256_file(path) for path in source_paths
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "git": _git_metadata(),
        "outer_cross_validation": outer,
        "inner_tuning": inner,
        "models": model_bindings,
        "outer_fold_selections": pd.concat(fold_selection, ignore_index=True).to_dict(
            orient="records"
        ),
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
                    "tuning_table",
                    "fold_metrics_table",
                    "feature_importance_table",
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
        comparison=comparison,
        tuning=tuning_table,
        fold_metrics=fold_metrics,
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


__all__ = ["load_stage5_config", "run_stage5_ml_candidates", "verify_stage4_artifacts"]
