"""Full pre-fitting information and survey-design gate for Stage 8 clinical_core."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from cvd_risk.modeling.stage8_clarification import (
    materialize_sbp_alias,
    validate_stage8_clarification,
)
from cvd_risk.modeling.stage8_preflight import load_stage8_cohort
from cvd_risk.provenance import sha256_file
from cvd_risk.stage0 import _find_rscript

EXPECTED_CLARIFICATION_COMMIT = "0b0a00a6cb41c0af71709d74aa89ec6fc4614610"
EXPECTED_CLARIFICATION_BLOB = "4d20cc80efcf80c4315e7f3c8f9756eb368a4e5d"
MINIMAL_FEATURES = (
    "age",
    "sex",
    "bmi",
    "smoking_status",
    "hypertension_dx",
    "diabetes_dx",
    "general_health",
)
CLINICAL_SOURCE_FIELDS = (
    "systolic_bp_mean",
    "total_cholesterol",
    "hdl_cholesterol",
)
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".r", ".yaml", ".yml"}


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


def _binding(path: Path, root: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest = hashlib.sha256(payload).hexdigest()
        hash_mode = "normalized_text_lf"
    else:
        digest = sha256_file(path)
        hash_mode = "raw_bytes"
    return {
        "repository_relative_path": path.resolve().relative_to(root.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": digest,
        "hash_mode": hash_mode,
    }


def load_stage8_gate_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Stage 8 gate config schema_version must be 1")
    if payload.get("stage") != "stage8_clinical_core_full_gate":
        raise ValueError("Unexpected Stage 8 gate stage")
    bindings = payload.get("bindings", {})
    if bindings.get("clarification_commit") != EXPECTED_CLARIFICATION_COMMIT:
        raise ValueError("Stage 8 gate is not bound to the clarification commit")
    if bindings.get("clarification_blob_sha1") != EXPECTED_CLARIFICATION_BLOB:
        raise ValueError("Stage 8 gate is not bound to the clarification artifact blob")
    gate = payload.get("full_gate", {})
    required = {
        "raw_events_minimum": 200,
        "predictor_parameters": 20,
        "expected_r2_fraction_of_max": 0.15,
        "shrinkage": 0.90,
        "max_design_based_risk_ci_half_width": 0.0025,
        "minimum_design_df": 8,
        "require_n_eff_design_at_least_pmsampsize": True,
        "require_all_checks": True,
    }
    if gate != required:
        raise ValueError("Stage 8 full-gate thresholds differ from the locked protocol")
    return payload


def verify_clarification_ancestry(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    commit = config["bindings"]["clarification_commit"]
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=root, check=False
    )
    if completed.returncode != 0:
        raise ValueError("Clarification commit is not an ancestor of the Stage 8 gate")
    revision = f"{commit}:{config['bindings']['clarification_path']}"
    if _git(root, "rev-parse", revision) != config["bindings"]["clarification_blob_sha1"]:
        raise ValueError("Clarification artifact differs from the committed blob")
    return validate_stage8_clarification(config["bindings"]["clarification_path"], root=root)


def build_primary_clinical_core(frame: pd.DataFrame) -> pd.DataFrame:
    mask = frame[list(CLINICAL_SOURCE_FIELDS)].notna().all(axis=1)
    selected = frame.loc[mask].copy().reset_index(drop=True)
    selected["sbp_mean"] = materialize_sbp_alias(selected["systolic_bp_mean"])
    source = selected["systolic_bp_mean"].to_numpy(dtype=np.float64)
    alias = selected["sbp_mean"].to_numpy(dtype=np.float64)
    if not np.array_equal(source, alias, equal_nan=True):
        raise RuntimeError("Stage 8 SBP schema alias changed values")
    return selected


def schema_mapping_table(frame: pd.DataFrame, feature_mapping_path: Path) -> pd.DataFrame:
    mapping = yaml.safe_load(feature_mapping_path.read_text(encoding="utf-8"))
    systolic = mapping["cross_source_features"]["systolic_bp"]
    if systolic.get("final_type") != "numeric_mmHg":
        raise ValueError("Feature mapping does not bind systolic BP to mmHg")
    source = frame["systolic_bp_mean"].to_numpy(dtype=np.float64)
    alias = frame["sbp_mean"].to_numpy(dtype=np.float64)
    return pd.DataFrame(
        [
            {
                "source_column": "systolic_bp_mean",
                "canonical_column": "sbp_mean",
                "mapping": "identity_alias_inside_stage8_pipeline_only",
                "source_dtype": str(frame["systolic_bp_mean"].dtype),
                "canonical_dtype": str(frame["sbp_mean"].dtype),
                "unit": "mmHg",
                "n": len(frame),
                "n_equal_including_missing": int(
                    np.sum((source == alias) | (np.isnan(source) & np.isnan(alias)))
                ),
                "minimum": float(np.min(alias)),
                "maximum": float(np.max(alias)),
                "equality_assertion": bool(np.array_equal(source, alias, equal_nan=True)),
                "source_cohort_modified": False,
            }
        ]
    )


def weight_validation_table(frame: pd.DataFrame, policy: Mapping[str, Any]) -> pd.DataFrame:
    weights = frame["analysis_weight"].to_numpy(dtype=np.float64)
    allowed_sources = set(policy["allowed_weight_sources"])
    checks = [
        ("weight_finite", bool(np.isfinite(weights).all()), int((~np.isfinite(weights)).sum())),
        ("weight_positive", bool((weights > 0).all()), int((weights <= 0).sum())),
        ("strata_complete", bool(frame["strata"].notna().all()), int(frame["strata"].isna().sum())),
        ("psu_complete", bool(frame["psu"].notna().all()), int(frame["psu"].isna().sum())),
        (
            "weight_source_allowlist",
            set(frame["analysis_weight_source"].dropna().astype(str).unique()) == allowed_sources,
            ";".join(sorted(frame["analysis_weight_source"].dropna().astype(str).unique())),
        ),
        (
            "weight_protocol",
            frame["weight_protocol"].eq(policy["weight_protocol"]).all(),
            ";".join(sorted(frame["weight_protocol"].astype(str).unique())),
        ),
        (
            "weight_estimand",
            frame["weight_estimand"].eq(policy["weight_estimand"]).all(),
            ";".join(sorted(frame["weight_estimand"].astype(str).unique())),
        ),
    ]
    return pd.DataFrame(
        [
            {"check": name, "status": "PASS" if passed else "FAIL", "observed": observed}
            for name, passed, observed in checks
        ]
    )


def missingness_validation_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for field in (*CLINICAL_SOURCE_FIELDS, *MINIMAL_FEATURES):
        missing = int(frame[field].isna().sum())
        clinical = field in CLINICAL_SOURCE_FIELDS
        passed = missing == 0 if clinical else True
        rows.append(
            {
                "field": field,
                "role": "required_clinical" if clinical else "minimal_fold_imputation_allowed",
                "n": len(frame),
                "n_missing": missing,
                "missing_fraction": float(missing / len(frame)),
                "imputation_policy": ("forbidden" if clinical else "fit_inside_training_fold_only"),
                "status": "PASS" if passed else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def _run_r_gate(
    frame: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    root: Path,
) -> pd.DataFrame:
    rscript = _find_rscript()
    if rscript is None:
        raise RuntimeError("Rscript is required for the Stage 8 full gate")
    gate = config["full_gate"]
    design_input = frame[["cvd_death_5y", "analysis_weight", "strata", "psu"]].copy()
    design_input.columns = ["outcome", "weight", "strata", "psu"]
    with tempfile.TemporaryDirectory(prefix="cvd-risk-stage8-gate-") as directory:
        input_path = Path(directory) / "input.csv"
        output_path = Path(directory) / "output.csv"
        design_input.to_csv(input_path, index=False)
        completed = subprocess.run(
            [
                rscript,
                str(root / "scripts/r/stage8_clinical_gate.R"),
                str(input_path),
                str(output_path),
                repr(gate["expected_r2_fraction_of_max"]),
                str(gate["predictor_parameters"]),
                repr(gate["shrinkage"]),
                repr(gate["max_design_based_risk_ci_half_width"]),
                str(gate["minimum_design_df"]),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
        result = pd.read_csv(output_path)
    if len(result) != 1:
        raise RuntimeError("Stage 8 R gate must return exactly one row")
    survey = config["survey_design"]
    row = result.iloc[0]
    if str(row["survey_version"]) != survey["survey_version"]:
        raise ValueError("R survey version differs from the Stage 8 gate lock")
    if str(row["pmsampsize_version"]) != survey["pmsampsize_version"]:
        raise ValueError("R pmsampsize version differs from the Stage 8 gate lock")
    return result


def gate_decision_table(
    design: pd.DataFrame,
    weights: pd.DataFrame,
    missingness: pd.DataFrame,
    schema: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    row = design.iloc[0]
    gate = config["full_gate"]
    checks = [
        (
            "raw_events",
            int(row["raw_events"]),
            gate["raw_events_minimum"],
            int(row["raw_events"]) >= gate["raw_events_minimum"],
        ),
        (
            "n_eff_design",
            float(row["n_eff_design"]),
            float(row["pmsampsize_required_n"]),
            bool(row["pass_information"]),
        ),
        (
            "risk_ci_half_width",
            float(row["max_ci_half_width"]),
            gate["max_design_based_risk_ci_half_width"],
            bool(row["pass_risk_precision"]),
        ),
        (
            "design_df",
            float(row["design_df"]),
            gate["minimum_design_df"],
            bool(row["pass_design_df"]),
        ),
        (
            "survey_weight_validation",
            int(weights["status"].eq("PASS").sum()),
            len(weights),
            weights["status"].eq("PASS").all(),
        ),
        (
            "missingness_validation",
            int(missingness["status"].eq("PASS").sum()),
            len(missingness),
            missingness["status"].eq("PASS").all(),
        ),
        (
            "sbp_schema_alias",
            bool(schema.iloc[0]["equality_assertion"]),
            True,
            bool(schema.iloc[0]["equality_assertion"]),
        ),
    ]
    return pd.DataFrame(
        [
            {
                "check": name,
                "observed": observed,
                "required": required,
                "status": "PASS" if passed else "FAIL",
            }
            for name, observed, required, passed in checks
        ]
    )


def _write_report(
    path: Path,
    *,
    decision: pd.DataFrame,
    design: pd.DataFrame,
    missingness: pd.DataFrame,
    status: str,
) -> None:
    row = design.iloc[0]
    lines = [
        "# Stage 8 clinical_core full pre-fitting gate",
        "",
        f"**{status}**",
        "",
        "Анализ относится только к NHANES internal development/exploratory clinical route "
        "и не является новой independent external validation.",
        "",
        "## Полный gate",
        "",
        decision.to_markdown(index=False),
        "",
        "## Design-based information",
        "",
        design.to_markdown(index=False),
        "",
        f"Primary Section 20 cohort содержит {int(row['n_nominal']):,} участников и "
        f"{int(row['raw_events']):,} raw events. Event floor сам по себе не определял "
        "решение: fitting разрешается только при PASS всех строк полного gate.",
        "",
        "Section 10 paired all-predictor complete-case сохраняется отдельно как "
        "sensitivity feasibility cohort: N=18,262, events=198, FAIL.",
        "",
        "## Minimal-predictor missingness",
        "",
        missingness.to_markdown(index=False),
        "",
        "Clinical fields не импутировались. Любая будущая imputation minimal predictors "
        "должна обучаться только на training partition каждого grouped fold.",
        "",
        "Исходный NHANES cohort, locked minimal_v1, Stage 6/7 artifacts и existing reports "
        "не изменялись.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_stage8_clinical_gate(
    config_path: str | Path,
    *,
    root: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    repository = Path(root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = repository / config_file
    config = load_stage8_gate_config(config_file)
    clarification = verify_clarification_ancestry(config, repository)

    preflight_config = yaml.safe_load(
        (repository / "configs/stage8_clinical_preflight.yaml").read_text(encoding="utf-8")
    )
    frame = load_stage8_cohort(preflight_config, root=repository)
    extra_columns = pd.read_parquet(
        repository / config["bindings"]["cohort"],
        columns=[
            "participant_id",
            "analysis_weight_source",
            "weight_protocol",
            "weight_estimand",
        ],
    ).set_index("participant_id")
    frame = frame.join(extra_columns, on="participant_id", validate="one_to_one")
    primary = build_primary_clinical_core(frame)
    schema = schema_mapping_table(primary, repository / config["bindings"]["feature_mapping"])
    weights = weight_validation_table(primary, config["survey_design"])
    missingness = missingness_validation_table(primary)
    design = _run_r_gate(primary, config=config, root=repository)
    decision = gate_decision_table(design, weights, missingness, schema, config)
    all_pass = bool(decision["status"].eq("PASS").all())
    status_label = (
        "PASS_ALL_GATES_CLINICAL_CORE_FITTING_AUTHORIZED"
        if all_pass
        else "FAIL_FULL_GATE_CLINICAL_CORE_FITTING_FORBIDDEN"
    )

    outputs = {key: repository / value for key, value in config["outputs"].items()}
    known = [path for key, path in outputs.items() if key != "artifact_dir"]
    if any(path.exists() for path in known) and not force:
        raise FileExistsError("Stage 8 full-gate outputs exist; pass --force to replace them")
    for path in known:
        path.parent.mkdir(parents=True, exist_ok=True)
    decision.to_csv(outputs["gate_table"], index=False)
    design.to_csv(outputs["design_metrics"], index=False)
    missingness.to_csv(outputs["missingness_validation"], index=False)
    weights.to_csv(outputs["weight_validation"], index=False)
    schema.to_csv(outputs["schema_mapping"], index=False)
    _write_report(
        outputs["report"],
        decision=decision,
        design=design,
        missingness=missingness,
        status=status_label,
    )
    status = {
        "schema_version": 1,
        "stage": "stage8_clinical_core_full_gate",
        "status": status_label,
        "generated_at": datetime.now(UTC).isoformat(),
        "interpretation": "nhanes_internal_development_exploratory_not_external_validation",
        "clarification_commit": EXPECTED_CLARIFICATION_COMMIT,
        "clarification_validation": clarification,
        "all_required_gates_pass": all_pass,
        "clinical_core_fitting_authorized": all_pass,
        "clinical_hba1c_fitting_authorized": False,
        "models_fitted": False,
        "input_bindings": {
            "config": _binding(config_file, repository),
            "cohort": _binding(repository / config["bindings"]["cohort"], repository),
            "clarification": _binding(
                repository / config["bindings"]["clarification_path"], repository
            ),
            "r_gate_source": _binding(repository / "scripts/r/stage8_clinical_gate.R", repository),
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


def validate_stage8_gate_status(
    status_path: str | Path,
    *,
    root: str | Path,
) -> dict[str, Any]:
    repository = Path(root).resolve()
    path = Path(status_path)
    if not path.is_absolute():
        path = repository / path
    status = json.loads(path.read_text(encoding="utf-8"))
    if status.get("stage") != "stage8_clinical_core_full_gate":
        raise ValueError("Unexpected Stage 8 gate status stage")
    if status.get("status") != "FAIL_FULL_GATE_CLINICAL_CORE_FITTING_FORBIDDEN":
        raise ValueError("Stage 8 gate is expected to preserve the information-gate failure")
    if status.get("all_required_gates_pass") is not False:
        raise ValueError("Stage 8 gate cannot claim all checks passed")
    if status.get("clinical_core_fitting_authorized") is not False:
        raise ValueError("Stage 8 gate cannot authorize clinical_core fitting")
    if status.get("models_fitted") is not False:
        raise ValueError("Stage 8 gate status unexpectedly records model fitting")
    for entry in status["outputs"].values():
        output_path = repository / entry["repository_relative_path"]
        current = _binding(output_path, repository)
        if current["sha256"] != entry["sha256"] or current["hash_mode"] != entry["hash_mode"]:
            raise ValueError(f"Stage 8 gate output changed: {output_path}")
    gate = pd.read_csv(repository / "reports/tables/stage8_clinical_core_full_gate.csv")
    failed = gate.loc[gate["status"].eq("FAIL")]
    if failed["check"].tolist() != ["n_eff_design"]:
        raise ValueError("Stage 8 full gate must have only the information check failing")
    info = failed.iloc[0]
    if not (
        np.isclose(float(info["observed"]), 13061.9598240511)
        and np.isclose(float(info["required"]), 13732.0)
    ):
        raise ValueError("Stage 8 information-gate values changed")
    fitted = []
    for pattern in ("*.joblib", "*oof*.parquet", "*prediction*.parquet"):
        fitted.extend((repository / "artifacts/stage8_clinical_gate").glob(pattern))
    if fitted:
        raise ValueError(f"Forbidden Stage 8 fitted artifacts exist: {fitted}")
    return {
        "status": "PASS",
        "gate_status": status["status"],
        "only_failed_check": "n_eff_design",
        "clinical_core_fitting_authorized": False,
        "models_fitted": False,
    }


__all__ = [
    "build_primary_clinical_core",
    "gate_decision_table",
    "load_stage8_gate_config",
    "missingness_validation_table",
    "run_stage8_clinical_gate",
    "validate_stage8_gate_status",
    "weight_validation_table",
]
