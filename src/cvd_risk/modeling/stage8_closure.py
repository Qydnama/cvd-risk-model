"""Validate the immutable-evidence Stage 8 closure status and report."""

from __future__ import annotations

import csv
import json
import subprocess
from decimal import Decimal
from pathlib import Path
from typing import Any

EXPECTED_EVIDENCE = {
    "primary_gate": (
        "a3efc76cb0565ff4638c30a1c5cb747b95fa73a4",
        "artifacts/stage8_clinical_gate/status.json",
        "aa84bf5d83ea9acb5f4b7f9386cfb5c2c4e9afa6",
    ),
    "historical_hba1c_available_analysis_status": (
        "df6d157fd7753f463c0620b42e32fd2343bb4d6b",
        "artifacts/stage8_exploratory/status.json",
        "f1b204e54700035c10bacf2559c64760d41e7c8f",
    ),
    "historical_hba1c_available_analysis_report": (
        "df6d157fd7753f463c0620b42e32fd2343bb4d6b",
        "reports/stage8_underpowered_exploratory_development_report.md",
        "2a151149b1a7f230b6009d639607273a51fbffee",
    ),
    "correction_addendum": (
        "dfc75d783b29420503d30c3f4eed528111633a7e",
        "artifacts/stage8_exploratory_correction/addendum.json",
        "0c2947f2c592c8d2cc2ec760dc008cc4e07b6996",
    ),
    "corrected_analysis_status": (
        "aa74b6ab7589e8c7a6b93b663fca15e7244c88f3",
        "artifacts/stage8_exploratory_corrected/status.json",
        "acc307c7e67de8ead55cd783963801f3686314ac",
    ),
    "corrected_analysis_report": (
        "aa74b6ab7589e8c7a6b93b663fca15e7244c88f3",
        "reports/stage8_corrected_underpowered_exploratory_development_report.md",
        "cb529162d9dba3c19fb0311d21857adadbc5c23d",
    ),
    "posthoc_high_sensitivity_report": (
        "17cf4853ab9fd574aa67e0e9beeab65087b545a2",
        "reports/posthoc_secondary_high_sensitivity_operating_points.md",
        "c44f089702507af318f61051d7df0ead9fd78d84",
    ),
    "posthoc_high_sensitivity_table": (
        "17cf4853ab9fd574aa67e0e9beeab65087b545a2",
        "reports/tables/posthoc_high_sensitivity_operating_points.csv",
        "db1c723bfe6cd4c9cfef5e804142e0105d2be6b6",
    ),
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


def validate_stage8_closure(
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
        "status": "CLOSED_PRIMARY_GATE_FAILED_EXPLORATORY_ANALYSES_RETAINED",
        "PRIMARY_EXPORT_GATE": "FAIL",
        "PRIMARY_EXPORT_GATE_REASON": "FAIL_INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE",
        "EXPLORATORY_FITTING": "COMPLETE",
        "CLINICAL_CORE_EXPORT_ALLOWED": False,
        "CLINICAL_HBA1C_FITTING": "NOT_AUTHORIZED_NOT_PERFORMED",
        "CLINICAL_HBA1C_EXPORT_ALLOWED": False,
        "WEB_ROUTING_ALLOWED": False,
        "EXTERNAL_VALIDATION": "NOT_PERFORMED",
        "MINIMAL_V1_CHANGED_BY_STAGE8": False,
    }
    if any(status.get(key) != value for key, value in required.items()):
        raise ValueError("Stage 8 closure decision differs from the approved evidence")
    if status.get("n_eff_design") != 13061.9598240511:
        raise ValueError("Stage 8 closure n_eff_design changed")
    if status.get("N_pmsampsize") != 13732:
        raise ValueError("Stage 8 closure N_pmsampsize changed")

    evidence = status["evidence"]
    if set(evidence) != set(EXPECTED_EVIDENCE):
        raise ValueError("Stage 8 closure evidence inventory changed")
    for name, (commit, relative, blob) in EXPECTED_EVIDENCE.items():
        if evidence[name] != {"commit": commit, "path": relative, "blob_sha1": blob}:
            raise ValueError(f"Stage 8 closure evidence binding changed: {name}")
        if _git(repository, "cat-file", "-t", commit) != "commit":
            raise ValueError(f"Stage 8 evidence commit is unavailable: {commit}")
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=repository,
            check=False,
        )
        if ancestor.returncode != 0:
            raise ValueError(f"Stage 8 evidence is not an ancestor of HEAD: {commit}")
        if _git(repository, "rev-parse", f"{commit}:{relative}") != blob:
            raise ValueError(f"Stage 8 evidence blob differs at its commit: {relative}")
        if _git(repository, "diff", "--name-only", commit, "--", relative):
            raise ValueError(f"Stage 8 evidence file changed after its commit: {relative}")

    primary_gate = json.loads(
        (repository / "artifacts/stage8_clinical_gate/status.json").read_text(encoding="utf-8")
    )
    if primary_gate.get("status") != "FAIL_FULL_GATE_CLINICAL_CORE_FITTING_FORBIDDEN":
        raise ValueError("Primary Stage 8 full-gate failure changed")

    historical = json.loads(
        (repository / "artifacts/stage8_exploratory/status.json").read_text(encoding="utf-8")
    )
    if (historical.get("cohort_n"), historical.get("raw_cvd_events")) != (18987, 218):
        raise ValueError("Historical HbA1c-available cohort counts changed")
    correction = json.loads(
        (
            repository / "artifacts/stage8_exploratory_correction/addendum.json"
        ).read_text(encoding="utf-8")
    )
    historical_label = "HBA1C_AVAILABLE_PAIRED_COHORT_SENSITIVITY_ANALYSIS"
    if correction["historical_analysis_preservation"]["authoritative_relabel"] != (
        historical_label
    ):
        raise ValueError("Historical Stage 8 analysis relabel changed")

    corrected = json.loads(
        (
            repository / "artifacts/stage8_exploratory_corrected/status.json"
        ).read_text(encoding="utf-8")
    )
    if (corrected.get("cohort_n"), corrected.get("raw_cvd_events")) != (19018, 218):
        raise ValueError("Corrected primary Section 20 cohort counts changed")
    if corrected.get("label") != "UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY":
        raise ValueError("Corrected Stage 8 analysis label changed")
    if corrected.get("hba1c_required") is not False or corrected.get("hba1c_loaded") is not False:
        raise ValueError("Corrected Stage 8 HbA1c policy changed")

    with (repository / "reports/tables/posthoc_high_sensitivity_operating_points.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        operating_points = {row["operating_point"]: row for row in csv.DictReader(handle)}
    if Decimal(operating_points["original_locked_cutoff"]["cutoff"]) != Decimal(
        "0.007469058716065273"
    ):
        raise ValueError("Original locked cutoff changed in the post-hoc reference")
    if operating_points["high_sensitivity_90"]["label"] != (
        "POST_HOC_SECONDARY_HIGH_SENSITIVITY_OPERATING_POINT"
    ):
        raise ValueError("90% post-hoc threshold label changed")
    if operating_points["high_sensitivity_95"]["label"] != (
        "EXPLORATORY_EXTREME_SENSITIVITY_STRESS_TEST"
    ):
        raise ValueError("95% exploratory threshold label changed")

    report = (repository / "reports/stage8_closure_report.md").read_text(encoding="utf-8")
    required_report_tokens = (
        "CLOSED_PRIMARY_GATE_FAILED_EXPLORATORY_ANALYSES_RETAINED",
        "PRIMARY_EXPORT_GATE = FAIL",
        "HBA1C_AVAILABLE_PAIRED_COHORT_SENSITIVITY_ANALYSIS",
        "UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY",
        "CLINICAL_CORE_EXPORT_ALLOWED = false",
        "CLINICAL_HBA1C_FITTING = NOT_AUTHORIZED_NOT_PERFORMED",
        "WEB_ROUTING_ALLOWED = false",
        "EXTERNAL_VALIDATION = NOT_PERFORMED",
    )
    if any(token not in report for token in required_report_tokens):
        raise ValueError("Stage 8 closure report lacks a required decision token")

    closure_dir = repository / "artifacts/stage8_closure"
    forbidden: list[Path] = []
    for pattern in ("*.joblib", "*.pkl", "*.pickle", "*.onnx", "*.bin", "*routing*.json"):
        forbidden.extend(closure_dir.rglob(pattern))
    if forbidden:
        raise ValueError(f"Stage 8 closure created forbidden deployable artifacts: {forbidden}")
    return {
        "status": "PASS",
        **required,
        "historical_hba1c_available_cohort_n": 18987,
        "corrected_primary_section20_cohort_n": 19018,
        "raw_cvd_events": 218,
        "deployable_clinical_artifacts": 0,
    }


__all__ = ["validate_stage8_closure"]
