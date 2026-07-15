"""Pre-unseal Stage 6 predictions and survey precision gates."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import yaml

from cvd_risk.modeling.baselines import FittedBaselineModel
from cvd_risk.modeling.lock_amendment import (
    NHANES_EXTERNAL_CYCLES,
    NHIS_TEMPORAL_YEARS,
    portable_binding,
    repository_root,
    verify_amendment_content,
)
from cvd_risk.modeling.preprocessing import ROUTE_FEATURES
from cvd_risk.modeling.stage5a import (
    LOCKED_PMVAL_ARGUMENTS,
    _run_r_pmvalsampsize,
    evaluate_stability,
)
from cvd_risk.provenance import sha256_file
from cvd_risk.stage0 import _find_rscript

FORBIDDEN_COLUMN_TOKENS = (
    "outcome",
    "death",
    "deceased",
    "mort",
    "ucod",
    "permth",
    "followup",
    "follow_up",
)
PREDICTION_COLUMNS = [
    "cohort",
    "participant_id",
    "survey_cycle",
    "survey_year",
    "strata",
    "psu",
    "analysis_weight",
    "predicted_risk",
    "linear_predictor",
    "secondary_screen_positive",
]


def _assert_label_free_columns(columns: list[str]) -> None:
    blocked = [
        column
        for column in columns
        if any(token in column.lower() for token in FORBIDDEN_COLUMN_TOKENS)
    ]
    if blocked:
        raise ValueError(f"Outcome-like columns are forbidden before unseal: {blocked}")


def _nested(payload: dict[str, Any], dotted_field: str) -> Any:
    value: Any = payload
    for field in dotted_field.split("."):
        if not isinstance(value, dict) or field not in value:
            raise KeyError(f"Missing manifest field: {dotted_field}")
        value = value[field]
    return value


def load_stage6_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Stage 6 config schema_version must be 1")
    if payload.get("stage") != "stage6_preunseal_label_free_predictions":
        raise ValueError("Unexpected Stage 6 identifier")
    if payload.get("pmvalsampsize") != LOCKED_PMVAL_ARGUMENTS:
        raise ValueError("Stage 6 pmvalsampsize arguments differ from Stage 5a")
    cohorts = payload.get("cohorts")
    if not isinstance(cohorts, dict) or set(cohorts) != {
        "nhis_temporal",
        "nhanes_external",
    }:
        raise ValueError("Stage 6 requires the two locked validation cohorts")
    if cohorts["nhis_temporal"].get("years") != NHIS_TEMPORAL_YEARS:
        raise ValueError("NHIS temporal years must be explicitly enumerated")
    if cohorts["nhanes_external"].get("cycles") != NHANES_EXTERNAL_CYCLES:
        raise ValueError("NHANES external cycles must be explicitly enumerated")
    for settings in cohorts.values():
        loaded = settings.get("loaded_columns")
        if not isinstance(loaded, list) or not all(isinstance(item, str) for item in loaded):
            raise ValueError("Each cohort needs an explicit loaded_columns allowlist")
        _assert_label_free_columns(loaded)
        required = {*ROUTE_FEATURES["minimal"], "participant_id", "survey_year"}
        required.update({"strata", "psu", "analysis_weight", "split"})
        missing = required.difference(loaded)
        if missing:
            raise ValueError(f"Stage 6 allowlist is missing columns: {sorted(missing)}")
    return payload


def _verify_lock_and_amendment(
    config: dict[str, Any], *, root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    lock_path = root / config["lock"]["original"]
    amendment_path = root / config["lock"]["amendment"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    verify_amendment_content(amendment, lock=lock, root=root)
    pipeline_path = root / config["lock"]["pipeline"]
    if sha256_file(pipeline_path) != lock["model"]["artifact"]["sha256"]:
        raise ValueError("Frozen full-development pipeline SHA changed")
    if amendment.get("heldout_outcomes_unsealed") is not False:
        raise ValueError("Stage 6 label-free runner requires sealed held-out outcomes")
    return lock, amendment


def _load_label_free_cohort(
    name: str, settings: dict[str, Any], *, root: Path
) -> pd.DataFrame:
    path = root / settings["path"]
    if sha256_file(path) != settings["sha256"]:
        raise ValueError(f"{name} cohort SHA changed")
    columns = list(settings["loaded_columns"])
    _assert_label_free_columns(columns)
    available = set(pq.ParquetFile(path).schema_arrow.names)
    missing = set(columns).difference(available)
    if missing:
        raise KeyError(f"{name} cohort schema is missing: {sorted(missing)}")

    # Security boundary: pandas/pyarrow receives only the declared predictor/design
    # allowlist. No outcome or mortality column is deserialized before unseal.
    frame = pd.read_parquet(path, columns=columns)
    frame = frame.loc[frame["split"].eq(settings["split"])].copy()
    if len(frame) != int(settings["expected_n"]):
        raise ValueError(f"{name} row count differs from the sealed aggregate")
    years = sorted(frame["survey_year"].astype(int).unique().tolist())
    if years != list(settings["years"]):
        raise ValueError(f"{name} years differ from the explicit config")
    if settings["cycles"]:
        cycles = sorted(frame["survey_cycle"].astype(str).unique().tolist())
        if cycles != sorted(settings["cycles"]):
            raise ValueError(f"{name} cycles differ from the explicit config")
    if frame["participant_id"].duplicated().any():
        raise ValueError(f"{name} participant identifiers are not unique")
    if frame[["strata", "psu", "analysis_weight"]].isna().any().any():
        raise ValueError(f"{name} has missing survey-design fields")
    if (frame["analysis_weight"] <= 0).any():
        raise ValueError(f"{name} has non-positive survey weights")
    return frame.reset_index(drop=True)


def create_label_free_predictions(
    frame: pd.DataFrame,
    *,
    cohort: str,
    model: FittedBaselineModel,
    cutoff: float,
    probability_clip: float,
) -> pd.DataFrame:
    if model.route != "minimal":
        raise ValueError("Locked Stage 6 model must use the minimal route")
    probability = np.asarray(model.predict_proba(frame, calibrated=True), dtype=np.float64)
    if len(probability) != len(frame) or not np.isfinite(probability).all():
        raise ValueError("Frozen model produced invalid probabilities")
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("Frozen model produced values outside [0, 1]")
    bounded = np.clip(probability, probability_clip, 1.0 - probability_clip)
    cycle = (
        frame["survey_cycle"].astype(str)
        if "survey_cycle" in frame.columns
        else pd.Series([""] * len(frame), index=frame.index, dtype="string")
    )
    output = pd.DataFrame(
        {
            "cohort": cohort,
            "participant_id": frame["participant_id"].astype(str),
            "survey_cycle": cycle,
            "survey_year": frame["survey_year"].astype(int),
            "strata": frame["strata"],
            "psu": frame["psu"],
            "analysis_weight": frame["analysis_weight"].astype(float),
            "predicted_risk": probability,
            "linear_predictor": np.log(bounded / (1.0 - bounded)),
            "secondary_screen_positive": probability >= cutoff,
        }
    )
    if output.columns.tolist() != PREDICTION_COLUMNS:
        raise RuntimeError("Label-free prediction output schema changed")
    _assert_label_free_columns(output.columns.tolist())
    return output


def weighted_prediction_summary(frame: pd.DataFrame) -> dict[str, Any]:
    weights = frame["analysis_weight"].to_numpy(dtype=np.float64)
    risk = frame["predicted_risk"].to_numpy(dtype=np.float64)
    lp = frame["linear_predictor"].to_numpy(dtype=np.float64)
    lp_mean = float(np.average(lp, weights=weights))
    return {
        "cohort": str(frame["cohort"].iloc[0]),
        "n_nominal": len(frame),
        "weight_sum": float(weights.sum()),
        "weighted_prediction_mean": float(np.average(risk, weights=weights)),
        "weighted_prediction_sd": float(
            np.sqrt(np.average((risk - np.average(risk, weights=weights)) ** 2, weights=weights))
        ),
        "weighted_lp_mean": lp_mean,
        "weighted_lp_sd": float(
            np.sqrt(np.average((lp - lp_mean) ** 2, weights=weights))
        ),
        "prediction_min": float(risk.min()),
        "prediction_max": float(risk.max()),
        "weighted_screen_positive_fraction": float(
            np.average(frame["secondary_screen_positive"].astype(float), weights=weights)
        ),
    }


def _run_design_effects(
    predictions: list[pd.DataFrame],
    *,
    rscript: str,
    r_script_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    design_input = pd.concat(
        [
            frame.rename(
                columns={
                    "analysis_weight": "weight",
                    "predicted_risk": "prediction",
                }
            ).loc[:, ["cohort", "weight", "strata", "psu", "linear_predictor", "prediction"]]
            for frame in predictions
        ],
        ignore_index=True,
    )
    _assert_label_free_columns(design_input.columns.tolist())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cvd_stage6_label_free_") as directory:
        input_path = Path(directory) / "design_input.csv"
        design_input.to_csv(input_path, index=False)
        completed = subprocess.run(
            [rscript, str(r_script_path), str(input_path), str(output_path)],
            capture_output=True,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            "R label-free design-effect run failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    result = pd.read_csv(output_path)
    if len(result) != 4:
        raise RuntimeError("Expected LP and prediction design effects for two cohorts")
    return result


def _write_report(
    path: Path,
    *,
    summaries: pd.DataFrame,
    design_effects: pd.DataFrame,
    precision_gate: pd.DataFrame,
    manifest_path: Path,
) -> None:
    gate_lines = []
    for row in precision_gate.itertuples(index=False):
        status = "PASS" if row.pass_precision_gate else "FAIL / underpowered"
        gate_lines.append(
            f"- {row.cohort}: **{status}**; nominal N={row.n_nominal:,}, "
            f"DEFF gate={row.deff_gate:.4f}, effective N={row.n_eff_gate:,.1f}, "
            f"required nominal N={row.required_nominal_n:,}."
        )
    lines = [
        "# Stage 6: pre-unseal label-free predictions",
        "",
        "## Result",
        "",
        "The immutable sigmoid-calibrated minimal LogisticRegression pipeline was "
        "applied once to NHIS 2010–2014 and all eight NHANES 1999–2014 cycles. "
        "Only explicitly allowlisted predictor, identifier, and survey-design columns "
        "were deserialized. No held-out outcome, mortality, cause-of-death, or "
        "follow-up column was loaded or joined.",
        "",
        "The primary output remains continuous five-year risk. The fixed cutoff is a "
        "secondary screening operating point, not a diagnosis or clinically endorsed "
        "high-risk boundary.",
        "",
        "## Label-free prediction distribution",
        "",
        summaries.to_markdown(index=False),
        "",
        "## Survey design effects",
        "",
        design_effects.to_markdown(index=False),
        "",
        "## Precision gate",
        "",
        *gate_lines,
        "",
        "The pmvalsampsize prevalence argument remains the frozen survey-weighted "
        "NHIS development prevalence; held-out prevalence was not read. A failed gate "
        "does not permit model changes and marks subsequent validation as "
        "underpowered/exploratory with full confidence intervals.",
        "",
        "Held-out outcomes remain sealed. This report does not authorize or perform "
        "unsealing.",
        "",
        f"Machine-readable provenance: `{manifest_path.as_posix()}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_stage6_label_free(
    *, config_path: str | Path, force: bool = False
) -> dict[str, Any]:
    root = repository_root()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = load_stage6_config(config_file)
    lock, amendment = _verify_lock_and_amendment(config, root=root)
    outputs = config["outputs"]
    artifact_dir = root / outputs["artifact_dir"]
    prediction_dir = root / outputs["prediction_dir"]
    manifest_path = artifact_dir / "manifest.json"
    report_path = root / outputs["report"]
    table_paths = {
        "summary": root / outputs["summary_table"],
        "stability": root / outputs["stability_table"],
        "design_effects": root / outputs["design_effect_table"],
        "precision_gate": root / outputs["precision_gate_table"],
    }
    prediction_paths = {
        name: prediction_dir / f"{name}_predictions.parquet"
        for name in config["cohorts"]
    }
    known = [manifest_path, report_path, *table_paths.values(), *prediction_paths.values()]
    if any(path.exists() for path in known) and not force:
        raise FileExistsError("Stage 6 outputs exist; pass --force to replace them")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(root / config["lock"]["pipeline"])
    if not isinstance(model, FittedBaselineModel):
        raise TypeError("Frozen pipeline has an unexpected serialized type")
    if model.name != lock["model"]["name"] or model.route != lock["model"]["route"]:
        raise ValueError("Serialized pipeline identity differs from the lock")
    cutoff = float(
        lock["output_policy"]["secondary_screening_operating_point"]["cutoff"]
    )
    probability_clip = float(config["probability_clip"])

    predictions: dict[str, pd.DataFrame] = {}
    input_bindings: dict[str, Any] = {}
    for name, settings in config["cohorts"].items():
        frame = _load_label_free_cohort(name, settings, root=root)
        output = create_label_free_predictions(
            frame,
            cohort=name,
            model=model,
            cutoff=cutoff,
            probability_clip=probability_clip,
        )
        output.to_parquet(prediction_paths[name], index=False)
        predictions[name] = output
        input_bindings[name] = {
            "repository_relative_path": settings["path"],
            "sha256": settings["sha256"],
            "loaded_columns": settings["loaded_columns"],
            "outcome_columns_loaded": [],
            "predictor_missing_counts": {
                column: int(frame[column].isna().sum())
                for column in ROUTE_FEATURES["minimal"]
            },
            "missing_value_handling": "frozen_development_fitted_preprocessor",
            "split": settings["split"],
        }

    summary_table = pd.DataFrame(
        [weighted_prediction_summary(frame) for frame in predictions.values()]
    )
    table_paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    summary_table.to_csv(table_paths["summary"], index=False)

    rscript = _find_rscript()
    if rscript is None:
        raise RuntimeError("BLOCKED_RSCRIPT_NOT_FOUND")
    r_design_script = root / "scripts/r/label_free_design_effects.R"
    design_effects = _run_design_effects(
        list(predictions.values()),
        rscript=rscript,
        r_script_path=r_design_script,
        output_path=table_paths["design_effects"],
    )

    prevalence_manifest_path = root / config["prevalence_source"]["manifest"]
    prevalence_manifest = json.loads(prevalence_manifest_path.read_text(encoding="utf-8"))
    prevalence = float(
        _nested(prevalence_manifest, config["prevalence_source"]["field"])
    )
    arguments = config["pmvalsampsize"]
    r_pmval_script = root / "scripts/r/pmvalsampsize_stage5a.R"
    stability_rows: list[dict[str, Any]] = []
    stability_by_cohort: dict[str, dict[str, Any]] = {}
    for summary in summary_table.to_dict(orient="records"):
        cohort = str(summary["cohort"])
        rows: list[dict[str, Any]] = []
        simobs_values = [int(arguments["initial_simobs"]), int(arguments["repeat_simobs"])]
        while True:
            simobs = simobs_values[len(rows)]
            raw_path = artifact_dir / f"pmvalsampsize_{cohort}_{simobs}.csv"
            row = _run_r_pmvalsampsize(
                rscript=rscript,
                r_script_path=r_pmval_script,
                output_path=raw_path,
                prevalence=prevalence,
                lp_mean=float(summary["weighted_lp_mean"]),
                lp_sd=float(summary["weighted_lp_sd"]),
                simobs=simobs,
                seed=int(arguments["seed"]),
            )
            row["cohort"] = cohort
            row["raw_output_path"] = raw_path.relative_to(root).as_posix()
            row["raw_output_sha256"] = sha256_file(raw_path)
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
                raise RuntimeError(f"{cohort} pmvalsampsize stability did not converge")
            simobs_values.append(next_simobs)
        stability_by_cohort[cohort] = stability
        stability_rows.extend(rows)
    stability_table = pd.DataFrame(stability_rows)
    stability_table["relative_difference_from_previous"] = stability_table.groupby(
        "cohort"
    )["required_n"].pct_change().abs()
    stability_table.to_csv(table_paths["stability"], index=False)

    sealed_table_path = root / config["sealed_design_effect_table"]
    sealed = pd.read_csv(sealed_table_path)
    gate_rows: list[dict[str, Any]] = []
    for name, settings in config["cohorts"].items():
        sealed_name = str(settings["sealed_design_cohort"])
        sealed_row = sealed.loc[sealed["cohort"].eq(sealed_name)]
        if len(sealed_row) != 1:
            raise ValueError(f"Missing unique sealed DEFF_y for {name}")
        deff_y = float(sealed_row.iloc[0]["deff_y"])
        if not np.isclose(deff_y, float(settings["sealed_design_effect_y"]), rtol=1e-12):
            raise ValueError(f"Configured and sealed DEFF_y differ for {name}")
        cohort_deff = design_effects.loc[design_effects["cohort"].eq(name)]
        deff_lp = float(
            cohort_deff.loc[cohort_deff["variable"].eq("linear_predictor"), "design_effect"].iloc[0]
        )
        deff_prediction = float(
            cohort_deff.loc[cohort_deff["variable"].eq("prediction"), "design_effect"].iloc[0]
        )
        deff_gate = max(1.0, deff_y, deff_lp, deff_prediction)
        required = int(stability_by_cohort[name]["provisional_required_n"])
        n_nominal = int(settings["expected_n"])
        n_eff = n_nominal / deff_gate
        required_nominal = int(np.ceil(required * deff_gate))
        gate_rows.append(
            {
                "cohort": name,
                "n_nominal": n_nominal,
                "pmvalsampsize_required_n": required,
                "deff_y": deff_y,
                "deff_lp": deff_lp,
                "deff_prediction": deff_prediction,
                "deff_gate": deff_gate,
                "n_eff_gate": n_eff,
                "required_nominal_n": required_nominal,
                "pass_precision_gate": bool(n_eff >= required),
            }
        )
    precision_gate = pd.DataFrame(gate_rows)
    precision_gate.to_csv(table_paths["precision_gate"], index=False)

    _write_report(
        report_path,
        summaries=summary_table,
        design_effects=design_effects,
        precision_gate=precision_gate,
        manifest_path=manifest_path.relative_to(root),
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "stage": "stage6_preunseal_label_free_predictions",
        "status": "LABEL_FREE_PREDICTIONS_FROZEN_OUTCOMES_STILL_SEALED",
        "heldout_outcomes_accessed": False,
        "heldout_outcomes_unsealed": False,
        "model_retrained_or_modified": False,
        "original_lock": {
            "repository_relative_path": config["lock"]["original"],
            "sha256": amendment["original_lock"]["sha256"],
            "commit": amendment["original_lock"]["commit"],
        },
        "lock_amendment": {
            **portable_binding(config["lock"]["amendment"], root=root),
            "commit": config["lock"]["amendment_commit"],
        },
        "pipeline": {
            "repository_relative_path": config["lock"]["pipeline"],
            "sha256": lock["model"]["artifact"]["sha256"],
            "calibration": lock["model"]["calibration"],
        },
        "cutoff": cutoff,
        "input_bindings": input_bindings,
        "prevalence_argument": {
            "value": prevalence,
            "source": config["prevalence_source"]["interpretation"],
            "heldout_prevalence_used": False,
        },
        "pmvalsampsize_arguments": arguments,
        "cohorts": {
            name: {
                "n": len(frame),
                "prediction_path": prediction_paths[name].relative_to(root).as_posix(),
                "prediction_sha256": sha256_file(prediction_paths[name]),
                "prediction_columns": PREDICTION_COLUMNS,
                "outcome_columns": [],
                "stability": stability_by_cohort[name],
            }
            for name, frame in predictions.items()
        },
        "environment": {
            "python": platform.python_version(),
            "rscript": rscript,
            "survey": "4.5",
            "pmvalsampsize": arguments["package_version"],
        },
        "source_bindings": {
            path.relative_to(root).as_posix(): sha256_file(path)
            for path in (
                config_file,
                root / "src/cvd_risk/modeling/stage6.py",
                root / "scripts/14_stage6_label_free.py",
                root / "scripts/15_validate_stage6_label_free.py",
                r_design_script,
                r_pmval_script,
                prevalence_manifest_path,
                sealed_table_path,
            )
        },
        "outputs": {
            name: {
                "repository_relative_path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in {**table_paths, "report": report_path}.items()
        },
        "next_action_requires_explicit_unseal": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_stage6_label_free(
    manifest_path: str | Path = "artifacts/stage6_label_free/manifest.json",
) -> dict[str, Any]:
    root = repository_root()
    target = Path(manifest_path)
    if not target.is_absolute():
        target = root / target
    manifest = json.loads(target.read_text(encoding="utf-8"))
    if manifest.get("status") != "LABEL_FREE_PREDICTIONS_FROZEN_OUTCOMES_STILL_SEALED":
        raise ValueError("Unexpected Stage 6 label-free manifest status")
    if manifest.get("heldout_outcomes_accessed") is not False:
        raise ValueError("Stage 6 manifest reports held-out outcome access")
    if manifest.get("heldout_outcomes_unsealed") is not False:
        raise ValueError("Stage 6 manifest reports held-out outcome unsealing")
    if manifest.get("model_retrained_or_modified") is not False:
        raise ValueError("Stage 6 manifest reports a changed model")

    lock_path = root / manifest["original_lock"]["repository_relative_path"]
    amendment_path = root / manifest["lock_amendment"]["repository_relative_path"]
    pipeline_path = root / manifest["pipeline"]["repository_relative_path"]
    current_amendment_binding = portable_binding(
        manifest["lock_amendment"]["repository_relative_path"], root=root
    )
    if any(
        current_amendment_binding[field] != manifest["lock_amendment"][field]
        for field in ("repository_relative_path", "sha256", "hash_mode")
    ):
        raise ValueError("Stage 6 lock amendment SHA changed")
    amendment_commit = str(manifest["lock_amendment"].get("commit", ""))
    if len(amendment_commit) != 40 or any(
        character not in "0123456789abcdef" for character in amendment_commit
    ):
        raise ValueError("Stage 6 manifest lacks a full amendment commit SHA")
    committed_amendment = subprocess.run(
        [
            "git",
            "show",
            f"{amendment_commit}:{manifest['lock_amendment']['repository_relative_path']}",
        ],
        cwd=root,
        capture_output=True,
        check=False,
    )
    committed_payload = committed_amendment.stdout.replace(b"\r\n", b"\n").replace(
        b"\r", b"\n"
    )
    if committed_amendment.returncode != 0 or hashlib.sha256(
        committed_payload
    ).hexdigest() != manifest["lock_amendment"]["sha256"]:
        raise ValueError("Amendment commit does not contain the bound amendment")
    if sha256_file(pipeline_path) != manifest["pipeline"]["sha256"]:
        raise ValueError("Stage 6 frozen pipeline SHA changed")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    verify_amendment_content(amendment, lock=lock, root=root)

    for relative, digest in manifest["source_bindings"].items():
        source = root / relative
        if not source.is_file() or sha256_file(source) != digest:
            raise ValueError(f"Stage 6 source binding changed: {relative}")
    for name, entry in manifest["outputs"].items():
        output = root / entry["repository_relative_path"]
        if not output.is_file() or sha256_file(output) != entry["sha256"]:
            raise ValueError(f"Stage 6 output binding changed: {name}")
    for name, entry in manifest["cohorts"].items():
        prediction = root / entry["prediction_path"]
        if not prediction.is_file() or sha256_file(prediction) != entry["prediction_sha256"]:
            raise ValueError(f"Stage 6 prediction binding changed: {name}")
        columns = pq.ParquetFile(prediction).schema_arrow.names
        if columns != PREDICTION_COLUMNS or columns != entry["prediction_columns"]:
            raise ValueError(f"Stage 6 prediction schema changed: {name}")
        _assert_label_free_columns(columns)
        if entry.get("outcome_columns") != []:
            raise ValueError(f"Stage 6 prediction declares outcome columns: {name}")
    return {
        "status": "STAGE6_LABEL_FREE_VALID",
        "cohort_count": len(manifest["cohorts"]),
        "pipeline_sha256": manifest["pipeline"]["sha256"],
        "heldout_outcomes_accessed": False,
        "heldout_outcomes_unsealed": False,
    }


__all__ = [
    "FORBIDDEN_COLUMN_TOKENS",
    "PREDICTION_COLUMNS",
    "create_label_free_predictions",
    "load_stage6_config",
    "run_stage6_label_free",
    "validate_stage6_label_free",
    "weighted_prediction_summary",
]
