"""Corrected Stage 8 paired OOF analysis on the primary Section 20 cohort."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from cvd_risk.modeling.preprocessing import RCS_QUANTILES
from cvd_risk.modeling.stage8_clarification import materialize_sbp_alias
from cvd_risk.modeling.stage8_correction_addendum import (
    HISTORICAL_ANALYSIS_COMMIT,
    HISTORICAL_BLOBS,
    HISTORICAL_LABEL,
)
from cvd_risk.modeling.stage8_exploratory import (
    CORE_FEATURES,
    LABEL,
    MINIMAL_FEATURES,
    _binding,
    _git,
    add_dca_intervals,
    add_percentile_intervals,
    fit_paired_oof,
    paired_cluster_bootstrap,
    point_dca_table,
    point_metric_table,
)
from cvd_risk.provenance import sha256_file

CORRECTION_ADDENDUM_COMMIT = "dfc75d783b29420503d30c3f4eed528111633a7e"
CORRECTION_ADDENDUM_BLOB = "0c2947f2c592c8d2cc2ec760dc008cc4e07b6996"
PRIMARY_GATE_COMMIT = "a3efc76cb0565ff4638c30a1c5cb747b95fa73a4"
EXPECTED_VALIDATION_CYCLES = {
    1: ["2003-2004", "2009-2010"],
    2: ["2005-2006", "2013-2014"],
    3: ["2001-2002", "2011-2012"],
    4: ["1999-2000", "2007-2008"],
}


def load_corrected_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Corrected Stage 8 config schema_version must be 1")
    if payload.get("stage") != "stage8_corrected_underpowered_exploratory_development_only":
        raise ValueError("Unexpected corrected Stage 8 stage")
    if payload.get("label") != LABEL:
        raise ValueError("Corrected Stage 8 label changed")
    if payload.get("historical_analysis_label") != HISTORICAL_LABEL:
        raise ValueError("Historical analysis relabel changed")
    bindings = payload["bindings"]
    if bindings["correction_addendum_commit"] != CORRECTION_ADDENDUM_COMMIT:
        raise ValueError("Corrected workflow is not bound to the correction commit")
    if bindings["correction_addendum_blob_sha1"] != CORRECTION_ADDENDUM_BLOB:
        raise ValueError("Corrected workflow is not bound to the correction blob")
    if bindings["historical_analysis_commit"] != HISTORICAL_ANALYSIS_COMMIT:
        raise ValueError("Historical analysis commit changed")
    cohort = payload["fixed_cohort"]
    if cohort != {
        "split": "external_validation",
        "cohort_eligible": True,
        "required_nonmissing_source_fields": [
            "systolic_bp_mean",
            "total_cholesterol",
            "hdl_cholesterol",
        ],
        "hba1c_role": "not_required_not_loaded_not_a_predictor",
        "expected_n": 19018,
        "expected_events": 218,
    }:
        raise ValueError("Corrected primary Section 20 cohort policy changed")
    if payload["models"] != {
        "minimal": {"predictor_parameters": 15},
        "clinical_core": {
            "predictor_parameters": 20,
            "clinical_predictors": ["sbp_mean", "total_cholesterol", "hdl_cholesterol"],
        },
    }:
        raise ValueError("Corrected model specifications changed")
    if tuple(payload["restricted_cubic_splines"]["quantiles"]) != RCS_QUANTILES:
        raise ValueError("Corrected spline definitions changed")
    if payload["calibration"] != {
        "method": "nested_grouped_sigmoid",
        "probability_clip": 1.0e-6,
    }:
        raise ValueError("Corrected nested calibration changed")
    if payload["cluster_bootstrap"] != {
        "method": "stratified_psu_with_replacement_within_strata_fixed_oof_predictions_no_refit",
        "repetitions": 2000,
        "seed": 20260714,
        "confidence_level": 0.95,
        "interval": "percentile",
        "minimum_valid_replicates": 1900,
    }:
        raise ValueError("Corrected paired bootstrap policy changed")
    configured_cycles = {
        int(fold): [str(cycle) for cycle in cycles]
        for fold, cycles in payload["cross_validation"]["exact_validation_cycles"].items()
    }
    if configured_cycles != EXPECTED_VALIDATION_CYCLES:
        raise ValueError("Corrected workflow does not preserve the exact historical folds")
    return payload


def load_corrected_cohort(config: Mapping[str, Any], root: Path) -> pd.DataFrame:
    path = root / str(config["bindings"]["cohort"])
    if sha256_file(path) != config["bindings"]["cohort_sha256"]:
        raise ValueError("Corrected Stage 8 cohort binding changed")
    columns = [
        "participant_id",
        "survey_cycle",
        "strata",
        "psu",
        "analysis_weight",
        "cohort_eligible",
        "split",
        "cvd_death_5y",
        *MINIMAL_FEATURES,
        "systolic_bp_mean",
        "total_cholesterol",
        "hdl_cholesterol",
    ]
    if "hba1c" in columns:
        raise RuntimeError("Corrected workflow must not load or require HbA1c")
    frame = pd.read_parquet(path, columns=columns)
    policy = config["fixed_cohort"]
    frame = frame.loc[
        frame["cohort_eligible"].eq(policy["cohort_eligible"])
        & frame["split"].eq(policy["split"])
    ].copy()
    complete = frame[policy["required_nonmissing_source_fields"]].notna().all(axis=1)
    frame = frame.loc[complete].reset_index(drop=True)
    counts = (len(frame), int(frame["cvd_death_5y"].sum()))
    if counts != (policy["expected_n"], policy["expected_events"]):
        raise ValueError(f"Corrected Stage 8 cohort counts changed: {counts}")
    frame["sbp_mean"] = materialize_sbp_alias(frame["systolic_bp_mean"])
    if frame["participant_id"].duplicated().any():
        raise ValueError("Corrected Stage 8 participant IDs are not unique")
    return frame


def _assert_exact_folds(folds: pd.DataFrame) -> None:
    observed = {
        int(str(row.fold)): sorted(str(row.validation_cycles).split(";"))
        for row in folds.itertuples(index=False)
    }
    if observed != EXPECTED_VALIDATION_CYCLES:
        raise RuntimeError(f"Outer cycle assignment changed: {observed}")


def _write_corrected_report(
    path: Path,
    *,
    metrics: pd.DataFrame,
    differences: pd.DataFrame,
    dca: pd.DataFrame,
    folds: pd.DataFrame,
    valid_replicates: int,
) -> None:
    metric_view = metrics[
        ["model", "metric", "estimate", "ci_lower", "ci_upper", "valid_replicates"]
    ]
    difference_view = differences[
        ["metric", "estimate", "ci_lower", "ci_upper", "valid_replicates"]
    ]
    lines = [
        f"# Stage 8 corrected: {LABEL}",
        "",
        f"**{LABEL}**",
        "",
        "Это исправленный парный OOF-анализ на primary Section 20 clinical_core cohort: "
        "N=19,018, 218 raw CVD events. Требуются complete SBP, total cholesterol и HDL; "
        "HbA1c не требуется, не загружается и не является predictor.",
        "",
        "Предыдущий анализ N=18,987 / 218 events сохраняется без изменения и в соответствии "
        f"с correction addendum интерпретируется только как `{HISTORICAL_LABEL}`.",
        "",
        "Primary/export gate остаётся неизменным: "
        "`FAIL_INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE`, n_eff_design=13061.96 < "
        "N_pmsampsize=13732. Этот результат — underpowered NHANES internal development "
        "analysis, а не independent external validation.",
        "",
        "## Дизайн",
        "",
        "Minimal comparator и clinical_core рассчитаны для одних и тех же 19,018 участников "
        "и на одинаковых четырёх grouped folds. Использованы неизменённые спецификации 15 и "
        "20 predictor parameters. Imputation minimal predictors обучалась только внутри "
        "соответствующих training folds; clinical fields не импутировались. Calibration — "
        "та же nested grouped sigmoid calibration.",
        "",
        "## Survey-weighted OOF metrics",
        "",
        metric_view.to_markdown(index=False),
        "",
        "## Paired differences: clinical_core minus minimal",
        "",
        difference_view.to_markdown(index=False),
        "",
        "95% percentile CI рассчитаны тем же paired stratified PSU-within-strata cluster "
        f"bootstrap: {valid_replicates:,}/2,000 valid replicates, fixed OOF predictions, "
        "без refit.",
        "",
        "## Decision-curve analysis",
        "",
        dca.to_markdown(index=False),
        "",
        "## Идентичные outer folds",
        "",
        folds.to_markdown(index=False),
        "",
        "## Ограничения",
        "",
        "Не создавались full-cohort estimator, serialized pipeline, deployable/production "
        "artifact, API route или web routing configuration. Анализ не меняет Stage 8 full-gate "
        "FAIL и не является независимой внешней валидацией.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_corrected_stage8_exploratory(
    config_path: str | Path,
    *,
    root: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    repository = Path(root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = repository / config_file
    config = load_corrected_config(config_file)
    commit = config["bindings"]["correction_addendum_commit"]
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=repository,
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("Correction addendum commit is not an ancestor of corrected fitting")
    revision = f"{commit}:{config['bindings']['correction_addendum_path']}"
    if _git(repository, "rev-parse", revision) != CORRECTION_ADDENDUM_BLOB:
        raise ValueError("Correction addendum blob differs from its binding")
    historical_changes = _git(
        repository,
        "diff",
        "--name-only",
        HISTORICAL_ANALYSIS_COMMIT,
        "--",
        *HISTORICAL_BLOBS,
    )
    if historical_changes:
        raise ValueError(f"Historical HbA1c-available analysis changed: {historical_changes}")
    gate_paths = [
        "artifacts/stage8_clinical_gate/status.json",
        "reports/stage8_clinical_core_full_gate_report.md",
        "reports/tables/stage8_clinical_core_full_gate.csv",
    ]
    if _git(repository, "diff", "--name-only", PRIMARY_GATE_COMMIT, "--", *gate_paths):
        raise ValueError("Primary Stage 8 gate evidence changed before corrected fitting")

    outputs = {key: repository / value for key, value in config["outputs"].items()}
    known = [path for key, path in outputs.items() if key != "artifact_dir"]
    if any(path.exists() for path in known) and not force:
        raise FileExistsError("Corrected Stage 8 outputs exist; pass --force to replace them")
    for path in known:
        path.parent.mkdir(parents=True, exist_ok=True)

    frame = load_corrected_cohort(config, repository)
    predictions, folds, coefficients, imputation = fit_paired_oof(frame, config)
    _assert_exact_folds(folds)
    point = point_metric_table(predictions)
    point_dca = point_dca_table(predictions, config["decision_curve"]["thresholds"])
    bootstrap_config = config["cluster_bootstrap"]
    bootstrap = paired_cluster_bootstrap(
        predictions,
        thresholds=config["decision_curve"]["thresholds"],
        repetitions=int(bootstrap_config["repetitions"]),
        seed=int(bootstrap_config["seed"]),
    )
    valid_replicates = int(bootstrap["valid"].sum())
    if valid_replicates < int(bootstrap_config["minimum_valid_replicates"]):
        raise RuntimeError("Too few valid corrected paired bootstrap replicates")
    metrics, differences = add_percentile_intervals(
        point,
        bootstrap,
        confidence_level=float(bootstrap_config["confidence_level"]),
        minimum_valid_replicates=int(bootstrap_config["minimum_valid_replicates"]),
    )
    dca = add_dca_intervals(
        point_dca,
        bootstrap,
        confidence_level=float(bootstrap_config["confidence_level"]),
        minimum_valid_replicates=int(bootstrap_config["minimum_valid_replicates"]),
    )
    for table in (predictions, folds, coefficients, imputation, metrics, differences, dca):
        if "label" not in table or not table["label"].eq(LABEL).all():
            raise RuntimeError("A corrected output lacks the mandatory label")
    if "hba1c" in predictions.columns:
        raise RuntimeError("HbA1c must not appear in corrected OOF outputs")

    predictions.to_parquet(outputs["oof_predictions"], index=False)
    bootstrap.insert(0, "label", LABEL)
    bootstrap.to_csv(outputs["bootstrap_metrics"], index=False)
    metrics.to_csv(outputs["metrics"], index=False)
    differences.to_csv(outputs["paired_differences"], index=False)
    dca.to_csv(outputs["decision_curve"], index=False)
    folds.to_csv(outputs["folds"], index=False)
    coefficients.to_csv(outputs["coefficients"], index=False)
    imputation.to_csv(outputs["imputation_audit"], index=False)
    _write_corrected_report(
        outputs["report"],
        metrics=metrics,
        differences=differences,
        dca=dca,
        folds=folds,
        valid_replicates=valid_replicates,
    )

    forbidden_suffixes = {".joblib", ".pkl", ".pickle", ".onnx", ".bin"}
    forbidden = [
        path
        for path in outputs["artifact_dir"].rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    ]
    if forbidden:
        raise RuntimeError(f"Deployable corrected Stage 8 artifacts are forbidden: {forbidden}")
    status: dict[str, Any] = {
        "schema_version": 1,
        "stage": "stage8_corrected_underpowered_exploratory_development_only",
        "label": LABEL,
        "historical_analysis_label": HISTORICAL_LABEL,
        "historical_analysis_commit": HISTORICAL_ANALYSIS_COMMIT,
        "correction_addendum_commit": CORRECTION_ADDENDUM_COMMIT,
        "PRIMARY_EXPORT_GATE": "FAIL",
        "PRIMARY_EXPORT_GATE_REASON": "FAIL_INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE",
        "n_eff_design": 13061.9598240511,
        "N_pmsampsize": 13732,
        "EXPLORATORY_FITTING": "COMPLETE",
        "WEB_ROUTING_ALLOWED": False,
        "EXTERNAL_VALIDATION": "NOT_PERFORMED",
        "generated_at": datetime.now(UTC).isoformat(),
        "cohort_n": len(frame),
        "raw_cvd_events": int(frame["cvd_death_5y"].sum()),
        "hba1c_required": False,
        "hba1c_loaded": False,
        "models": ["minimal", "clinical_core"],
        "paired_identical_participants_and_folds": True,
        "valid_bootstrap_replicates": valid_replicates,
        "full_cohort_deployable_model_fitted": False,
        "serialized_model_or_pipeline_created": False,
        "api_route_created": False,
        "web_routing_configuration_created": False,
        "input_bindings": {
            "config": _binding(config_file, repository),
            "correction_addendum": _binding(
                repository / config["bindings"]["correction_addendum_path"], repository
            ),
            "cohort": _binding(repository / config["bindings"]["cohort"], repository),
            "source": _binding(Path(__file__), repository),
            "shared_prespecified_methodology_source": _binding(
                Path(__file__).with_name("stage8_exploratory.py"), repository
            ),
        },
        "outputs": {
            key: _binding(path, repository)
            for key, path in outputs.items()
            if key not in {"artifact_dir", "status"}
        },
    }
    outputs["status"].write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return status


def validate_corrected_stage8_status(
    status_path: str | Path,
    *,
    root: str | Path,
) -> dict[str, Any]:
    repository = Path(root).resolve()
    path = Path(status_path)
    if not path.is_absolute():
        path = repository / path
    status = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "PRIMARY_EXPORT_GATE": "FAIL",
        "EXPLORATORY_FITTING": "COMPLETE",
        "WEB_ROUTING_ALLOWED": False,
        "EXTERNAL_VALIDATION": "NOT_PERFORMED",
    }
    if any(status.get(key) != value for key, value in required.items()):
        raise ValueError("Corrected Stage 8 final status differs from authorization")
    if status.get("label") != LABEL or status.get("historical_analysis_label") != HISTORICAL_LABEL:
        raise ValueError("Corrected or historical analysis label changed")
    if status.get("n_eff_design") != 13061.9598240511 or status.get("N_pmsampsize") != 13732:
        raise ValueError("Primary/export effective-sample-size gate changed")
    for entry in (*status["input_bindings"].values(), *status["outputs"].values()):
        output = repository / entry["repository_relative_path"]
        current = _binding(output, repository)
        if current["sha256"] != entry["sha256"] or current["hash_mode"] != entry["hash_mode"]:
            raise ValueError(f"Corrected Stage 8 bound file changed: {output}")

    predictions = pd.read_parquet(
        repository / "artifacts/stage8_exploratory_corrected/oof_predictions.parquet"
    )
    if (len(predictions), int(predictions["cvd_death_5y"].sum())) != (19018, 218):
        raise ValueError("Corrected paired OOF cohort differs")
    if "hba1c" in predictions.columns or not predictions["label"].eq(LABEL).all():
        raise ValueError("Corrected OOF label or HbA1c prohibition failed")
    if predictions["participant_id"].duplicated().any() or predictions["fold"].isna().any():
        raise ValueError("Corrected paired OOF identities/folds are invalid")

    table_paths = (
        "reports/tables/stage8_corrected_exploratory_metrics.csv",
        "reports/tables/stage8_corrected_exploratory_paired_differences.csv",
        "reports/tables/stage8_corrected_exploratory_decision_curve.csv",
        "reports/tables/stage8_corrected_exploratory_folds.csv",
        "reports/tables/stage8_corrected_exploratory_fold_coefficients.csv",
        "reports/tables/stage8_corrected_exploratory_imputation_audit.csv",
        "artifacts/stage8_exploratory_corrected/bootstrap_metrics.csv",
    )
    for relative in table_paths:
        table = pd.read_csv(repository / relative)
        if "label" not in table or not table["label"].eq(LABEL).all():
            raise ValueError(f"Corrected table lacks its label: {relative}")
    folds = pd.read_csv(repository / table_paths[3])
    _assert_exact_folds(folds)
    imputation = pd.read_csv(repository / table_paths[5])
    if set(imputation["feature"]).intersection(CORE_FEATURES[-3:]):
        raise ValueError("Clinical fields appear in corrected imputation audit")
    paired = imputation.pivot(
        index=["fold", "feature"], columns="model", values="training_fold_value"
    )
    if set(paired.columns) != {"minimal", "clinical_core"}:
        raise ValueError("Both corrected models must record paired minimal imputation")
    for left, right in paired.itertuples(index=False, name=None):
        try:
            equal = bool(np.isclose(float(left), float(right), equal_nan=True))
        except (TypeError, ValueError):
            equal = str(left) == str(right)
        if not equal:
            raise ValueError("Corrected paired models used different minimal imputation")

    historical_changes = _git(
        repository,
        "diff",
        "--name-only",
        HISTORICAL_ANALYSIS_COMMIT,
        "--",
        *HISTORICAL_BLOBS,
    )
    if historical_changes:
        raise ValueError(f"Historical Stage 8 analysis was modified: {historical_changes}")
    primary_gate = json.loads(
        (repository / "artifacts/stage8_clinical_gate/status.json").read_text(encoding="utf-8")
    )
    if primary_gate.get("status") != "FAIL_FULL_GATE_CLINICAL_CORE_FITTING_FORBIDDEN":
        raise ValueError("Primary Stage 8 full-gate failure was changed")
    gate_paths = [
        "artifacts/stage8_clinical_gate/status.json",
        "reports/stage8_clinical_core_full_gate_report.md",
        "reports/tables/stage8_clinical_core_full_gate.csv",
    ]
    if _git(repository, "diff", "--name-only", PRIMARY_GATE_COMMIT, "--", *gate_paths):
        raise ValueError("Primary Stage 8 gate evidence differs from its gate commit")
    forbidden: list[Path] = []
    for pattern in ("*.joblib", "*.pkl", "*.pickle", "*.onnx", "*.bin", "*routing*.json"):
        forbidden.extend(
            (repository / "artifacts/stage8_exploratory_corrected").rglob(pattern)
        )
    if forbidden:
        raise ValueError(f"Forbidden corrected deployable/routing artifacts exist: {forbidden}")
    return {
        "status": "PASS",
        **required,
        "label": LABEL,
        "historical_analysis_label": HISTORICAL_LABEL,
        "cohort_n": 19018,
        "events": 218,
        "valid_bootstrap_replicates": int(status["valid_bootstrap_replicates"]),
        "deployable_artifacts": 0,
    }


__all__ = [
    "load_corrected_cohort",
    "load_corrected_config",
    "run_corrected_stage8_exploratory",
    "validate_corrected_stage8_status",
]
