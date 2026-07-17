"""Validate the pre-fitting correction for the primary Section 20 cohort."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

HISTORICAL_ADDENDUM_COMMIT = "024ad6af132affea3a88a60a56f0550e47cf2239"
HISTORICAL_ANALYSIS_COMMIT = "df6d157fd7753f463c0620b42e32fd2343bb4d6b"
PRIMARY_GATE_COMMIT = "a3efc76cb0565ff4638c30a1c5cb747b95fa73a4"
HISTORICAL_LABEL = "HBA1C_AVAILABLE_PAIRED_COHORT_SENSITIVITY_ANALYSIS"
CORRECTED_LABEL = "UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY"
HISTORICAL_BLOBS = {
    "artifacts/stage8_exploratory/status.json": "f1b204e54700035c10bacf2559c64760d41e7c8f",
    "artifacts/stage8_exploratory/oof_predictions.parquet": (
        "0e3596938e1e0172f75f36892cb2133b927d9d23"
    ),
    "reports/stage8_underpowered_exploratory_development_report.md": (
        "2a151149b1a7f230b6009d639607273a51fbffee"
    ),
    "reports/tables/stage8_exploratory_folds.csv": (
        "b60a2c303937c119b112ae68c6a011e9b66eb147"
    ),
}
EXPECTED_VALIDATION_CYCLES = {
    "1": ["2003-2004", "2009-2010"],
    "2": ["2005-2006", "2013-2014"],
    "3": ["2001-2002", "2011-2012"],
    "4": ["1999-2000", "2007-2008"],
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


def validate_stage8_correction_addendum(
    path: str | Path,
    *,
    root: str | Path,
) -> dict[str, Any]:
    repository = Path(root).resolve()
    artifact = Path(path)
    if not artifact.is_absolute():
        artifact = repository / artifact
    payload = json.loads(artifact.read_text(encoding="utf-8"))

    if payload.get("status") != "CORRECTION_ADDENDUM_RECORDED_BEFORE_CORRECTED_FITTING":
        raise ValueError("Unexpected Stage 8 correction-addendum status")
    historical = payload["historical_analysis_preservation"]
    if historical["prefitting_addendum_commit"] != HISTORICAL_ADDENDUM_COMMIT:
        raise ValueError("Historical pre-fitting addendum commit changed")
    if historical["hba1c_available_analysis_commit"] != HISTORICAL_ANALYSIS_COMMIT:
        raise ValueError("Historical HbA1c-available analysis commit changed")
    if historical["authoritative_relabel"] != HISTORICAL_LABEL:
        raise ValueError("Historical analysis was not relabeled as the sensitivity analysis")
    if historical["relabel_method"] != (
        "interpretation_overlay_without_rewriting_historical_files"
    ):
        raise ValueError("Historical files may not be rewritten for relabeling")

    for commit in (HISTORICAL_ADDENDUM_COMMIT, HISTORICAL_ANALYSIS_COMMIT):
        if _git(repository, "cat-file", "-t", commit) != "commit":
            raise ValueError(f"Required historical commit is unavailable: {commit}")
    for file_path, expected_blob in HISTORICAL_BLOBS.items():
        actual_blob = _git(
            repository, "rev-parse", f"{HISTORICAL_ANALYSIS_COMMIT}:{file_path}"
        )
        if actual_blob != expected_blob:
            raise ValueError(f"Historical blob differs at its commit: {file_path}")
    changed = _git(
        repository,
        "diff",
        "--name-only",
        HISTORICAL_ANALYSIS_COMMIT,
        "--",
        *HISTORICAL_BLOBS,
    )
    if changed:
        raise ValueError(f"Historical analysis files were modified: {changed}")

    historical_cohort = historical["cohort"]
    correction = payload["correction"]
    if historical_cohort["hba1c_role"] != "availability_filter_only_not_a_predictor":
        raise ValueError("Historical HbA1c availability role changed")
    if correction["label"] != CORRECTED_LABEL:
        raise ValueError("Corrected analysis label changed")
    if correction["hba1c"] != "not_required_not_a_predictor":
        raise ValueError("Corrected cohort must not require HbA1c")
    if correction["required_nonmissing_source_fields"] != [
        "systolic_bp_mean",
        "total_cholesterol",
        "hdl_cholesterol",
    ]:
        raise ValueError("Corrected Section 20 complete-case fields changed")

    columns = [
        "cohort_eligible",
        "split",
        "cvd_death_5y",
        "systolic_bp_mean",
        "total_cholesterol",
        "hdl_cholesterol",
        "hba1c",
    ]
    frame = pd.read_parquet(repository / correction["source"], columns=columns)
    base = frame.loc[
        frame["cohort_eligible"].eq(correction["cohort_eligible"])
        & frame["split"].eq(correction["split"])
    ]
    corrected_mask = base[correction["required_nonmissing_source_fields"]].notna().all(axis=1)
    historical_mask = corrected_mask & base["hba1c"].notna()
    corrected_counts = (
        int(corrected_mask.sum()),
        int(base.loc[corrected_mask, "cvd_death_5y"].sum()),
    )
    historical_counts = (
        int(historical_mask.sum()),
        int(base.loc[historical_mask, "cvd_death_5y"].sum()),
    )
    if corrected_counts != (19018, 218):
        raise ValueError(f"Corrected cohort counts differ: {corrected_counts}")
    if historical_counts != (18987, 218):
        raise ValueError(f"Historical cohort counts differ: {historical_counts}")
    if corrected_counts[0] - historical_counts[0] != 31:
        raise ValueError("Correction must restore exactly 31 HbA1c-unavailable participants")

    cross_validation = payload["cross_validation"]
    if cross_validation["outer_validation_cycles"] != EXPECTED_VALIDATION_CYCLES:
        raise ValueError("Historical outer-fold cycle assignment changed")
    if cross_validation["inner_splits"] != 4 or cross_validation["seed"] != 20260714:
        raise ValueError("Nested calibration folds or seed changed")
    gate = payload["preserved_primary_export_gate"]
    if gate != {
        "commit": PRIMARY_GATE_COMMIT,
        "status": "FAIL_INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE",
        "n_eff_design": 13061.9598240511,
        "n_pmsampsize": 13732,
        "PRIMARY_EXPORT_GATE": "FAIL",
        "primary_export_gate_changed": False,
    }:
        raise ValueError("Primary/export gate failure was modified")

    corrected_dir = repository / "artifacts/stage8_exploratory_corrected"
    fitted_before_addendum: list[Path] = []
    if corrected_dir.exists():
        for pattern in ("*.parquet", "*.csv", "*.joblib", "*.pkl"):
            fitted_before_addendum.extend(corrected_dir.glob(pattern))
    if fitted_before_addendum:
        raise ValueError(
            "Corrected fitting outputs existed before the addendum: "
            f"{fitted_before_addendum}"
        )
    return {
        "status": "PASS",
        "historical_analysis_label": HISTORICAL_LABEL,
        "historical_cohort_n": historical_counts[0],
        "historical_events": historical_counts[1],
        "corrected_analysis_label": CORRECTED_LABEL,
        "corrected_cohort_n": corrected_counts[0],
        "corrected_events": corrected_counts[1],
        "restored_hba1c_unavailable_participants": 31,
        "primary_export_gate": "FAIL_INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE",
        "corrected_fitting_observed_before_addendum": False,
    }


__all__ = ["validate_stage8_correction_addendum"]
