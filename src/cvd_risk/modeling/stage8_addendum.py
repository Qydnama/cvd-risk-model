"""Validate the pre-fitting Stage 8 exploratory-continuation addendum."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

EXPECTED_GATE_COMMIT = "a3efc76cb0565ff4638c30a1c5cb747b95fa73a4"
EXPECTED_LABEL = "UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY"
EXPECTED_GATE_BLOBS = {
    "artifacts/stage8_clinical_gate/status.json": "aa84bf5d83ea9acb5f4b7f9386cfb5c2c4e9afa6",
    "reports/stage8_clinical_core_full_gate_report.md": "ca02d98b882e55e6ac5d729bb9bb2fae93a1260e",
    "reports/tables/stage8_clinical_core_full_gate.csv": "83f52ae31d5abb80aca9a75fd3107999cecd7d1e",
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


def validate_stage8_exploratory_addendum(
    path: str | Path,
    *,
    root: str | Path,
) -> dict[str, Any]:
    repository = Path(root).resolve()
    artifact = Path(path)
    if not artifact.is_absolute():
        artifact = repository / artifact
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload.get("status") != "EXPLORATORY_CONTINUATION_AUTHORIZED_BEFORE_FITTING":
        raise ValueError("Unexpected Stage 8 exploratory addendum status")
    if payload.get("label") != EXPECTED_LABEL:
        raise ValueError("Stage 8 exploratory label differs")
    timing = payload["decision_timing"]
    required_unseen = {
        "any_stage8_model_fit",
        "any_stage8_oof_prediction",
        "any_stage8_performance_metric",
        "any_stage8_predictor_outcome_association",
    }
    if set(timing["made_before_observing"]) != required_unseen:
        raise ValueError("Addendum does not preserve pre-fitting timing")
    gate = payload["preserved_primary_export_gate"]
    if gate != {
        "status": "FAIL_INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE",
        "n_eff_design": 13061.9598240511,
        "n_pmsampsize": 13732,
        "primary_export_gate_changed": False,
        "primary_export_allowed": False,
    }:
        raise ValueError("Primary/export gate failure was changed or relabeled")
    evidence = payload["immutable_gate_evidence"]
    if evidence["commit"] != EXPECTED_GATE_COMMIT:
        raise ValueError("Addendum is not bound to the Stage 8 gate commit")
    for path_text, blob in EXPECTED_GATE_BLOBS.items():
        if _git(repository, "rev-parse", f"{EXPECTED_GATE_COMMIT}:{path_text}") != blob:
            raise ValueError(f"Stage 8 gate Git blob differs: {path_text}")
    if _git(repository, "diff", "--name-only", EXPECTED_GATE_COMMIT, "--", *EXPECTED_GATE_BLOBS):
        raise ValueError("Existing Stage 8 gate evidence was modified after its commit")

    cohort_policy = payload["fixed_exploratory_cohort"]
    if (cohort_policy["n"], cohort_policy["raw_cvd_events"]) != (18987, 218):
        raise ValueError("Exploratory cohort aggregate counts differ")
    columns = [
        "cohort_eligible",
        "split",
        "cvd_death_5y",
        *cohort_policy["required_nonmissing_source_fields"],
    ]
    frame = pd.read_parquet(repository / cohort_policy["source"], columns=columns)
    frame = frame.loc[
        frame["cohort_eligible"].eq(cohort_policy["cohort_eligible"])
        & frame["split"].eq(cohort_policy["split"])
    ]
    selected = frame[cohort_policy["required_nonmissing_source_fields"]].notna().all(axis=1)
    if (int(selected.sum()), int(frame.loc[selected, "cvd_death_5y"].sum())) != (18987, 218):
        raise ValueError("Rebuilt cohort does not reproduce addendum aggregate counts")
    models = payload["paired_models"]
    if models["minimal_comparator"]["predictor_parameters"] != 15:
        raise ValueError("Minimal comparator parameter budget changed")
    if models["clinical_core"]["predictor_parameters"] != 20:
        raise ValueError("Clinical_core parameter budget changed")
    if payload["cross_validation"] != {
        "group": "survey_cycle",
        "outer_splitter": "StratifiedGroupKFold",
        "outer_splits": 4,
        "inner_splits": 4,
        "shuffle": True,
        "seed": 20260714,
        "identical_participants_and_outer_folds": True,
    }:
        raise ValueError("Exploratory grouped-CV policy changed")
    final = payload["required_final_status"]
    if final != {
        "PRIMARY_EXPORT_GATE": "FAIL",
        "EXPLORATORY_FITTING": "COMPLETE",
        "WEB_ROUTING_ALLOWED": False,
        "EXTERNAL_VALIDATION": "NOT_PERFORMED",
    }:
        raise ValueError("Required final exploratory status changed")
    exploratory_dir = repository / "artifacts/stage8_exploratory"
    forbidden = []
    for pattern in ("*.joblib", "*oof*.parquet", "*prediction*.parquet"):
        forbidden.extend(exploratory_dir.glob(pattern))
    if forbidden:
        raise ValueError(f"Addendum was not committed before fitted artifacts: {forbidden}")
    return {
        "status": "PASS",
        "gate_commit": EXPECTED_GATE_COMMIT,
        "primary_export_gate": "FAIL_INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE",
        "cohort_n": 18987,
        "events": 218,
        "fitting_observed_before_addendum": False,
    }


__all__ = ["validate_stage8_exploratory_addendum"]
