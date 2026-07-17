"""Validate the minimal_v1 model card against immutable aggregate evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from decimal import Decimal
from pathlib import Path
from typing import Any

EXPECTED_EVIDENCE = {
    "external_validation_lock": (
        "a75366b7ab29d7a29d18353dc18e2d29ef0b8218",
        "artifacts/minimal_v1/external_validation_lock.json",
        "a6703aea4ffcce53489a37770fb98006ca0f6880",
    ),
    "lock_amendment": (
        "907c8f5cb8062b5a0c83013c78935a6d547acf11",
        "artifacts/minimal_v1/external_validation_lock_amendment_001.json",
        "d6f8ffd1dc404ae2569233e95272857c93208392",
    ),
    "serialized_pipeline_manifest": (
        "5cace77785c3e7e3398ead12efc101ae235034ec",
        "artifacts/minimal_v1/serialized_pipeline_manifest.json",
        "d776c177fb59c3f8323237be2cfa9be6aef79b4f",
    ),
    "stage4_development_report": (
        "a486317de194a99ce16fa65ba3b40c63fda458ba",
        "reports/stage4_baseline_report.md",
        "318cd1b08f9f35b5b3c1817062a5e332baa3d7ee",
    ),
    "stage5b_operating_metrics": (
        "5cace77785c3e7e3398ead12efc101ae235034ec",
        "reports/tables/stage5b_operating_metrics.csv",
        "5c39f90811362d2b4a0ec9a6ff70b56df510ac86",
    ),
    "stage6_validation_manifest": (
        "280d8405a189512f48caf5768d5c3843df83968a",
        "artifacts/stage6_validation/manifest.json",
        "f15447fdd1eec78c952c832ad9f9ab6347ea2b4d",
    ),
    "stage6_final_report": (
        "280d8405a189512f48caf5768d5c3843df83968a",
        "reports/stage6_underpowered_exploratory_validation_final.md",
        "dd5e40e97f4d47b5f2882549a2101fa2cf1705ce",
    ),
    "stage6_overall_metrics": (
        "280d8405a189512f48caf5768d5c3843df83968a",
        "reports/tables/stage6_validation_overall_metrics.csv",
        "f1727216d16c2b206ae454a86aae1c69ca740cef",
    ),
    "stage6_bootstrap_intervals": (
        "280d8405a189512f48caf5768d5c3843df83968a",
        "reports/tables/stage6_validation_bootstrap_ci.csv",
        "2e079231d98df7b3917a31aebb8566b3da8a9286",
    ),
    "stage6_subgroups": (
        "280d8405a189512f48caf5768d5c3843df83968a",
        "reports/tables/stage6_validation_subgroups.csv",
        "56c1013fc39a0ee8205db810cfe64051422dc1f4",
    ),
    "stage8_closure_status": (
        "c25d351f9e9b73bbffa62d28e58a574c92e6a449",
        "artifacts/stage8_closure/status.json",
        "8366c8c5b6220bd3da067e74fc56969dd1f173b8",
    ),
    "stage8_closure_report": (
        "c25d351f9e9b73bbffa62d28e58a574c92e6a449",
        "reports/stage8_closure_report.md",
        "f173a4cae48db7cf1688a68674c8e8158bbb1560",
    ),
    "posthoc_high_sensitivity_report": (
        "17cf4853ab9fd574aa67e0e9beeab65087b545a2",
        "reports/posthoc_secondary_high_sensitivity_operating_points.md",
        "c44f089702507af318f61051d7df0ead9fd78d84",
    ),
}

PIPELINE_HASHES = {
    "full_development_pipeline.joblib": (
        "758d635310a43596655bd23a0ebcebc293a0f4bde4ab5d98f3852a84e023a010"
    ),
    "preprocessing.joblib": "6509ebab833c6ae449f9ba569c7ceaae53386a962914734597e24aa3a739d857",
    "estimator.joblib": "b6c1827a9703ef9d4714088109588159fed40ed1526cba2145ddc6956e1495f7",
    "calibrator.joblib": "ff2291dc373e7ce934b84b07a069ba9cc02c3bff0f94b81492da40de7f31d525",
}


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_metric_table(path: Path) -> dict[tuple[str, str], Decimal]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row["cohort"], row["metric"]): Decimal(row["estimate"])
            for row in csv.DictReader(handle)
        }


def validate_minimal_v1_model_card(
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
        "status": "DOCUMENTED_LOCKED_RESEARCH_MODEL_UNCHANGED",
        "model_id": "minimal_v1",
        "protocol_version": "2.4",
        "model": "LogisticRegression",
        "route": "minimal",
        "calibration": "sigmoid",
        "predictor_parameters": 15,
        "lock_changed": False,
        "refit_performed": False,
        "clinical_deployment_approved": False,
        "api_or_web_route_created": False,
    }
    if any(status.get(key) != value for key, value in required.items()):
        raise ValueError("minimal_v1 model-card identity or governance status changed")
    if status.get("pipeline_sha256") != PIPELINE_HASHES["full_development_pipeline.joblib"]:
        raise ValueError("minimal_v1 model-card pipeline SHA changed")
    if Decimal(str(status.get("locked_cutoff"))) != Decimal("0.007469058716065273"):
        raise ValueError("minimal_v1 model-card locked cutoff changed")

    evidence = status["evidence"]
    if set(evidence) != set(EXPECTED_EVIDENCE):
        raise ValueError("minimal_v1 model-card evidence inventory changed")
    for name, (commit, relative, blob) in EXPECTED_EVIDENCE.items():
        if evidence[name] != {"commit": commit, "path": relative, "blob_sha1": blob}:
            raise ValueError(f"minimal_v1 evidence binding changed: {name}")
        if _git(repository, "cat-file", "-t", commit) != "commit":
            raise ValueError(f"minimal_v1 evidence commit unavailable: {commit}")
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=repository,
            check=False,
        )
        if ancestor.returncode != 0:
            raise ValueError(f"minimal_v1 evidence is not an ancestor of HEAD: {commit}")
        if _git(repository, "rev-parse", f"{commit}:{relative}") != blob:
            raise ValueError(f"minimal_v1 evidence blob differs at commit: {relative}")
        if _git(repository, "diff", "--name-only", commit, "--", relative):
            raise ValueError(f"minimal_v1 locked evidence changed after commit: {relative}")

    lock = json.loads(
        (repository / "artifacts/minimal_v1/external_validation_lock.json").read_text(
            encoding="utf-8"
        )
    )
    model = lock["model"]
    if (model["name"], model["route"], model["calibration"]) != (
        "LogisticRegression",
        "minimal",
        "sigmoid",
    ):
        raise ValueError("minimal_v1 locked model specification changed")
    if model["predictor_parameters"] != 15:
        raise ValueError("minimal_v1 predictor parameter budget changed")
    expected_inputs = [
        "age",
        "sex",
        "bmi",
        "smoking_status",
        "hypertension_dx",
        "diabetes_dx",
        "general_health",
    ]
    if model["feature_schema"]["input_order"] != expected_inputs:
        raise ValueError("minimal_v1 input schema changed")
    locked_point = lock["output_policy"]["secondary_screening_operating_point"]
    if Decimal(str(locked_point["cutoff"])) != Decimal("0.007469058716065273"):
        raise ValueError("minimal_v1 lock operating point changed")

    artifact_dir = repository / "artifacts/minimal_v1"
    for filename, expected_hash in PIPELINE_HASHES.items():
        if _sha256(artifact_dir / filename) != expected_hash:
            raise ValueError(f"minimal_v1 binary artifact changed: {filename}")

    stage6 = json.loads(
        (repository / "artifacts/stage6_validation/manifest.json").read_text(encoding="utf-8")
    )
    if stage6.get("status") != "COMPLETE_UNDERPOWERED_EXPLORATORY":
        raise ValueError("minimal_v1 Stage 6 interpretation changed")
    for flag in ("model_modified", "calibrator_modified", "cutoff_modified_or_reselected"):
        if stage6.get(flag) is not False:
            raise ValueError(f"minimal_v1 Stage 6 lock flag changed: {flag}")

    metrics = _read_metric_table(
        repository / "reports/tables/stage6_validation_overall_metrics.csv"
    )
    expected_metrics = {
        ("nhis_temporal", "roc_auc"): Decimal("0.795011126045732"),
        ("nhis_temporal", "calibration_slope"): Decimal("0.9139772829378829"),
        ("nhis_temporal", "observed_expected_ratio"): Decimal("0.8551337575380161"),
        ("nhis_temporal", "sensitivity"): Decimal("0.7813542141239187"),
        ("nhanes_external", "roc_auc"): Decimal("0.8190802967942877"),
        ("nhanes_external", "calibration_slope"): Decimal("1.0311668095159172"),
        ("nhanes_external", "observed_expected_ratio"): Decimal("0.8216769374053172"),
        ("nhanes_external", "sensitivity"): Decimal("0.789659221733242"),
    }
    if any(metrics.get(key) != value for key, value in expected_metrics.items()):
        raise ValueError("minimal_v1 Stage 6 aggregate metrics changed")

    closure = json.loads(
        (repository / "artifacts/stage8_closure/status.json").read_text(encoding="utf-8")
    )
    if closure.get("CLINICAL_CORE_EXPORT_ALLOWED") is not False:
        raise ValueError("Model card incorrectly permits clinical_core export")
    if closure.get("CLINICAL_HBA1C_FITTING") != "NOT_AUTHORIZED_NOT_PERFORMED":
        raise ValueError("Model card clinical_hba1c status changed")

    card = (repository / "reports/minimal_v1_model_card.md").read_text(encoding="utf-8")
    required_tokens = (
        "DOCUMENTED_LOCKED_RESEARCH_MODEL_UNCHANGED",
        "758d635310a43596655bd23a0ebcebc293a0f4bde4ab5d98f3852a84e023a010",
        "0.007469058716065273",
        "underpowered/exploratory",
        "clinical deployment",
        "Казахстан",
        "CLINICAL_HBA1C_FITTING",
    )
    if any(token not in card for token in required_tokens):
        raise ValueError("minimal_v1 model card lacks a required disclosure")

    return {
        "validation_status": "PASS",
        **required,
        "pipeline_sha256": PIPELINE_HASHES["full_development_pipeline.joblib"],
        "locked_cutoff": "0.007469058716065273",
        "validation_interpretation": "UNDERPOWERED_EXPLORATORY",
        "locked_binary_artifacts_verified": len(PIPELINE_HASHES),
        "clinical_routes_exported": 0,
    }


__all__ = ["validate_minimal_v1_model_card"]
