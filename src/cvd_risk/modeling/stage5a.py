"""Provisional external-validation sample-size planning from development OOF risk."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from cvd_risk.provenance import sha256_file
from cvd_risk.stage0 import _find_rscript

LOCKED_PMVAL_ARGUMENTS: dict[str, Any] = {
    "package_version": "0.1.0",
    "type": "b",
    "cstatistic": 0.75,
    "cstatciwidth": 0.10,
    "cslope": 1.0,
    "csciwidth": 0.40,
    "oe": 1.0,
    "oeciwidth": 0.20,
    "initial_simobs": 1_000_000,
    "repeat_simobs": 2_000_000,
    "maximum_simobs": 16_000_000,
    "stability_relative_tolerance": 0.01,
    "seed": 20260714,
    "graph": False,
    "trace": False,
}


def load_stage5a_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Stage 5a config schema_version must be 1")
    if payload.get("stage") != "stage5a_provisional_external_validation_planning":
        raise ValueError("Stage 5a config has an unexpected stage identifier")
    if payload.get("prediction_column") != "prediction_sigmoid":
        raise ValueError("Stage 5a must use the selected sigmoid-calibrated OOF prediction")
    if payload.get("prevalence_source") != (
        "nhis_development_survey_weighted_outcome_prevalence"
    ):
        raise ValueError("Stage 5a prevalence source is not protocol-compatible")
    if payload.get("weighted_sd_definition") != "population":
        raise ValueError("Stage 5a LP SD must use the weighted population definition")
    if payload.get("pmvalsampsize") != LOCKED_PMVAL_ARGUMENTS:
        raise ValueError("Stage 5a pmvalsampsize arguments diverge from the locked plan")
    return payload


def load_final_model_selection(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Final model selection schema_version must be 1")
    primary = payload.get("primary", {})
    required_primary = {
        "model": "LogisticRegression",
        "route": "minimal",
        "calibration": "sigmoid",
    }
    if not isinstance(primary, dict) or any(
        primary.get(key) != value for key, value in required_primary.items()
    ):
        raise ValueError("Primary model selection is not sigmoid minimal LogisticRegression")
    if payload.get("heldout_outcomes_used_for_selection") is not False:
        raise ValueError("Final selection must not use held-out outcomes")
    if payload.get("external_validation_lock_created") is not False:
        raise ValueError("Stage 5a selection config must remain pre-lock")
    return payload


def _verify_manifest(path: Path, *, stage: str, status: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required manifest is absent: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("stage") != stage or manifest.get("status") != status:
        raise ValueError(f"Unexpected manifest state: {path}")
    if manifest.get("heldout_outcomes_accessed") is not False:
        raise ValueError(f"Held-out outcome access is not sealed in {path}")
    if manifest.get("external_validation_lock_created") is not False:
        raise ValueError(f"A model lock already exists according to {path}")
    for source, digest in manifest.get("source_bindings", {}).items():
        source_path = Path(source)
        if not source_path.is_file() or sha256_file(source_path) != digest:
            raise ValueError(f"Manifest source binding changed: {source_path}")
    for name, entry in manifest.get("outputs", {}).items():
        output = Path(entry["path"])
        if not output.is_file() or sha256_file(output) != entry["sha256"]:
            raise ValueError(f"Manifest output binding changed: {name}")
    for entry in manifest.get("models", []):
        model = Path(entry["path"])
        if not model.is_file() or sha256_file(model) != entry["sha256"]:
            raise ValueError(f"Manifest model binding changed: {entry.get('name')}")
    return manifest


def weighted_lp_summary(
    prediction: np.ndarray,
    sample_weight: np.ndarray,
    outcome: np.ndarray,
    *,
    probability_clip: float,
) -> dict[str, float]:
    probabilities = np.asarray(prediction, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64)
    y = np.asarray(outcome, dtype=np.float64)
    if not (len(probabilities) == len(weights) == len(y)) or len(y) == 0:
        raise ValueError("Prediction, weight, and outcome arrays must have equal non-zero length")
    if not np.isfinite(probabilities).all() or not np.isfinite(weights).all():
        raise ValueError("Stage 5a arrays must contain only finite values")
    if (weights <= 0).any() or not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("Stage 5a requires positive weights and a binary outcome")
    bounded = np.clip(probabilities, probability_clip, 1.0 - probability_clip)
    lp = np.log(bounded / (1.0 - bounded))
    mean = float(np.average(lp, weights=weights))
    sd = float(np.sqrt(np.average((lp - mean) ** 2, weights=weights)))
    return {
        "weighted_prevalence": float(np.average(y, weights=weights)),
        "weighted_lp_mean": mean,
        "weighted_lp_sd": sd,
        "prediction_min": float(probabilities.min()),
        "prediction_max": float(probabilities.max()),
        "n_clipped_low": int((probabilities < probability_clip).sum()),
        "n_clipped_high": int((probabilities > 1.0 - probability_clip).sum()),
    }


def _run_r_pmvalsampsize(
    *,
    rscript: str,
    r_script_path: Path,
    output_path: Path,
    prevalence: float,
    lp_mean: float,
    lp_sd: float,
    simobs: int,
    seed: int,
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            rscript,
            str(r_script_path),
            str(output_path),
            repr(prevalence),
            repr(lp_mean),
            repr(lp_sd),
            str(simobs),
            str(seed),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "R pmvalsampsize Stage 5a run failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    result = pd.read_csv(output_path)
    if len(result) != 1:
        raise RuntimeError("R pmvalsampsize output must contain exactly one row")
    return {str(key): value for key, value in result.iloc[0].to_dict().items()}


def evaluate_stability(rows: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    if len(rows) < 2:
        raise ValueError("At least two pmvalsampsize runs are required")
    previous = float(rows[-2]["required_n"])
    current = float(rows[-1]["required_n"])
    relative_difference = abs(current - previous) / max(current, previous)
    return {
        "previous_required_n": previous,
        "current_required_n": current,
        "relative_difference": relative_difference,
        "tolerance": tolerance,
        "stable": relative_difference <= tolerance,
        "provisional_required_n": int(np.ceil(max(previous, current))),
    }


def _write_report(
    path: Path,
    *,
    summary: dict[str, float],
    stability_table: pd.DataFrame,
    stability: dict[str, Any],
    manifest_path: Path,
) -> None:
    lines = [
        "# Stage 5a: provisional external-validation precision planning",
        "",
        "## Scope",
        "",
        "Planning uses only sigmoid-calibrated LogisticRegression minimal out-of-fold "
        "predictions from NHIS development 1997–2009. No temporal or NHANES outcome "
        "was loaded or used.",
        "",
        "The required prevalence argument is the survey-weighted NHIS development "
        "five-year CVD mortality prevalence. This pre-lock value remains fixed for "
        "provisional planning and is not estimated from held-out labels.",
        "",
        "## OOF linear-predictor distribution",
        "",
        f"- Weighted prevalence: {summary['weighted_prevalence']:.12f}",
        f"- Weighted LP mean: {summary['weighted_lp_mean']:.12f}",
        f"- Weighted LP population SD: {summary['weighted_lp_sd']:.12f}",
        f"- Predictions clipped low/high: {summary['n_clipped_low']}/{summary['n_clipped_high']}",
        "",
        "## pmvalsampsize stability",
        "",
        stability_table.to_markdown(index=False),
        "",
        f"Stability relative difference: {stability['relative_difference']:.6f} "
        f"(tolerance {stability['tolerance']:.2%}).",
        "",
        f"Provisional IID/SRS-equivalent required N: **{stability['provisional_required_n']:,}**.",
        "",
        "Survey-design correction is intentionally deferred to Stage 6, after frozen "
        "label-free held-out predictions provide DEFF_LP and DEFF_prediction.",
        "",
        "No external-validation lock was created in this stage. Threshold policy and "
        "remaining pre-lock metadata must be frozen first.",
        "",
        f"Machine-readable provenance: `{manifest_path.as_posix()}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_stage5a_planning(
    *,
    config_path: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    stage5a_path = Path(config_path)
    config = load_stage5a_config(stage5a_path)
    selection_path = Path(config["selected_model_config"])
    selection = load_final_model_selection(selection_path)
    stage4_path = Path(config["stage4_manifest"])
    stage5_path = Path(config["stage5_manifest"])
    stage4 = _verify_manifest(
        stage4_path,
        stage="stage4_baselines",
        status="BASELINES_FITTED_NOT_MODEL_LOCKED",
    )
    _verify_manifest(
        stage5_path,
        stage="stage5_ml_candidates",
        status="ML_CANDIDATES_FITTED_NOT_MODEL_LOCKED",
    )
    primary = selection["primary"]
    stage4_models = {entry["name"]: entry for entry in stage4["models"]}
    if primary["model"] not in stage4_models:
        raise ValueError("Selected primary model is absent from Stage 4 manifest")
    selected_model = stage4_models[primary["model"]]
    selected_path = Path(primary["artifact"])
    if selected_path.resolve() != Path(selected_model["path"]).resolve():
        raise ValueError("Selected artifact path differs from Stage 4 manifest")
    if sha256_file(selected_path) != selected_model["sha256"]:
        raise ValueError("Selected model artifact SHA differs from Stage 4 manifest")

    oof_path = Path(primary["oof_predictions"])
    stage4_oof = Path(stage4["outputs"]["oof_predictions"]["path"])
    if oof_path.resolve() != stage4_oof.resolve():
        raise ValueError("Selected OOF path differs from Stage 4 manifest")
    oof = pd.read_parquet(oof_path)
    selected_oof = oof.loc[oof["model"].eq(primary["model"])].reset_index(drop=True)
    if len(selected_oof) != int(stage4["n"]):
        raise ValueError("Selected OOF predictions do not cover NHIS development")
    if not selected_oof["survey_year"].between(1997, 2009, inclusive="both").all():
        raise ValueError("Stage 5a OOF file contains a held-out survey year")
    summary = weighted_lp_summary(
        selected_oof[config["prediction_column"]].to_numpy(dtype=np.float64),
        selected_oof["analysis_weight"].to_numpy(dtype=np.float64),
        selected_oof["cvd_death_5y"].to_numpy(dtype=np.int8),
        probability_clip=float(config["probability_clip"]),
    )

    outputs = config["outputs"]
    artifact_dir = Path(outputs["artifact_dir"])
    stability_path = Path(outputs["stability_table"])
    report_path = Path(outputs["report"])
    manifest_path = artifact_dir / "manifest.json"
    known = [stability_path, report_path, manifest_path]
    if any(path.exists() for path in known) and not force:
        raise FileExistsError("Stage 5a outputs already exist; pass --force to replace them")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stability_path.parent.mkdir(parents=True, exist_ok=True)
    rscript = _find_rscript()
    if rscript is None:
        raise RuntimeError("BLOCKED_RSCRIPT_NOT_FOUND")
    r_script_path = Path("scripts/r/pmvalsampsize_stage5a.R")
    arguments = config["pmvalsampsize"]
    simobs_values = [int(arguments["initial_simobs"]), int(arguments["repeat_simobs"])]
    rows: list[dict[str, Any]] = []
    while True:
        simobs = simobs_values[len(rows)]
        raw_output = artifact_dir / f"pmvalsampsize_{simobs}.csv"
        row = _run_r_pmvalsampsize(
            rscript=rscript,
            r_script_path=r_script_path,
            output_path=raw_output,
            prevalence=summary["weighted_prevalence"],
            lp_mean=summary["weighted_lp_mean"],
            lp_sd=summary["weighted_lp_sd"],
            simobs=simobs,
            seed=int(arguments["seed"]),
        )
        row["raw_output_path"] = str(raw_output.resolve())
        row["raw_output_sha256"] = sha256_file(raw_output)
        rows.append(row)
        if len(rows) < 2:
            continue
        stability = evaluate_stability(
            rows, float(arguments["stability_relative_tolerance"])
        )
        if stability["stable"]:
            break
        next_simobs = simobs * 2
        if next_simobs > int(arguments["maximum_simobs"]):
            raise RuntimeError("pmvalsampsize stability did not converge by maximum_simobs")
        simobs_values.append(next_simobs)

    stability_table = pd.DataFrame(rows)
    stability_table["relative_difference_from_previous"] = (
        stability_table["required_n"].astype(float).pct_change().abs()
    )
    stability_table.to_csv(stability_path, index=False)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "stage": "stage5a_provisional_external_validation_planning",
        "status": "PROVISIONAL_PLANNING_COMPLETE_NOT_LOCKED",
        "selected_model": primary,
        "heldout_outcomes_accessed": False,
        "external_validation_lock_created": False,
        "threshold_policy_locked": False,
        "input_bindings": {
            "selection": {
                "path": str(selection_path.resolve()),
                "sha256": sha256_file(selection_path),
            },
            "stage4_manifest": {
                "path": str(stage4_path.resolve()),
                "sha256": sha256_file(stage4_path),
            },
            "stage5_manifest": {
                "path": str(stage5_path.resolve()),
                "sha256": sha256_file(stage5_path),
            },
            "selected_model": {
                "path": str(selected_path.resolve()),
                "sha256": sha256_file(selected_path),
            },
            "oof_predictions": {
                "path": str(oof_path.resolve()),
                "sha256": sha256_file(oof_path),
            },
            "config": {
                "path": str(stage5a_path.resolve()),
                "sha256": sha256_file(stage5a_path),
            },
            "r_script": {
                "path": str(r_script_path.resolve()),
                "sha256": sha256_file(r_script_path),
            },
            "renv_lock": {
                "path": str(Path("renv.lock").resolve()),
                "sha256": sha256_file("renv.lock"),
            },
        },
        "environment": {
            "python": platform.python_version(),
            "rscript": rscript,
            "pmvalsampsize": arguments["package_version"],
        },
        "lp_summary": summary,
        "pmvalsampsize_arguments": arguments,
        "stability": stability,
        "design_correction_status": "DEFERRED_TO_STAGE6_LABEL_FREE_PREDICTIONS",
        "outputs": {
            "stability_table": {
                "path": str(stability_path.resolve()),
                "sha256": sha256_file(stability_path),
            },
            "raw_runs": [
                {
                    "simobs": int(row["simobs"]),
                    "path": row["raw_output_path"],
                    "sha256": row["raw_output_sha256"],
                }
                for row in rows
            ],
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    _write_report(
        report_path,
        summary=summary,
        stability_table=stability_table,
        stability=stability,
        manifest_path=manifest_path,
    )
    manifest["outputs"]["report"] = {
        "path": str(report_path.resolve()),
        "sha256": sha256_file(report_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "LOCKED_PMVAL_ARGUMENTS",
    "evaluate_stability",
    "load_final_model_selection",
    "load_stage5a_config",
    "run_stage5a_planning",
    "weighted_lp_summary",
]
