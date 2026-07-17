"""Validation for the pre-fitting Stage 8 protocol clarification."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from cvd_risk.provenance import sha256_file

EXPECTED_HISTORY_COMMIT = "a6fac5d365a7369ca8492178fcca0e568ee0c097"
EXPECTED_PLAN_BLOB = "108f27945f4a850ce8ab6346a90243f8a7d1aaee"
EXPECTED_CONFIG_BLOB = "2858b795d1ce66483f288f54648315c3d1f6fae7"
EXPECTED_COHORT_SHA256 = "ce66815ecaf77176a8c46db78901fa1b3b73ae9e70d09134ef29b80714dd5964"


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


def materialize_sbp_alias(source: pd.Series) -> pd.Series:
    """Create the canonical Stage 8 alias with exact value preservation."""

    if not pd.api.types.is_float_dtype(source.dtype):
        raise TypeError("systolic_bp_mean must have a floating dtype")
    canonical = pd.Series(
        source.to_numpy(dtype=np.float64, na_value=np.nan),
        index=source.index,
        name="sbp_mean",
        dtype=np.float64,
    )
    original = source.to_numpy(dtype=np.float64, na_value=np.nan)
    if not np.array_equal(original, canonical.to_numpy(), equal_nan=True):
        raise RuntimeError("sbp_mean alias changed a systolic_bp_mean value")
    finite = canonical[np.isfinite(canonical)]
    if finite.empty or (finite <= 0).any():
        raise ValueError("systolic_bp_mean contains no valid positive mmHg values")
    return canonical


def validate_stage8_clarification(
    artifact_path: str | Path,
    *,
    root: str | Path,
) -> dict[str, Any]:
    repository = Path(root).resolve()
    path = Path(artifact_path)
    if not path.is_absolute():
        path = repository / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Stage 8 clarification schema_version must be 1")
    if payload.get("status") != "NARROW_CLARIFICATION_RECORDED_BEFORE_FITTING":
        raise ValueError("Stage 8 clarification status is invalid")
    if payload.get("external_validation_claim") is not False:
        raise ValueError("Stage 8 clarification cannot claim external validation")

    history = payload["history_confirmation"]
    if history.get("preexisting_policy_commit") != EXPECTED_HISTORY_COMMIT:
        raise ValueError("Unexpected preexisting policy commit")
    evidence = (
        (history["implementation_plan"], EXPECTED_PLAN_BLOB),
        (history["experiment_config"], EXPECTED_CONFIG_BLOB),
    )
    for entry, expected_blob in evidence:
        revision = f"{EXPECTED_HISTORY_COMMIT}:{entry['path']}"
        if entry.get("blob_sha1") != expected_blob:
            raise ValueError(f"Clarification records the wrong blob for {entry['path']}")
        if _git(repository, "rev-parse", revision) != expected_blob:
            raise ValueError(f"Git history no longer resolves the expected blob: {revision}")
        historical_text = _git(repository, "show", revision)
        for required in entry["required_preexisting_text"]:
            if required not in historical_text:
                raise ValueError(f"Historical policy evidence is absent: {required}")

    if payload["event_floor"] != {"raw_events_minimum": 200, "changed": False}:
        raise ValueError("Raw-event floor 200 must remain unchanged")
    strict = payload["section_10_sensitivity_feasibility"]
    if strict["clinical_hba1c_paired"] != {
        "n": 18262,
        "raw_events": 198,
        "event_gate_status": "FAIL",
    }:
        raise ValueError("Section 10 N/events/FAIL must remain unchanged")
    primary = payload["section_20_primary_policy"]
    if primary["preflight_raw_events"] != 218:
        raise ValueError("Section 20 preflight event count differs")
    if primary["event_gate_alone_authorizes_fitting"] is not False:
        raise ValueError("Event count alone must not authorize fitting")
    if payload["fitting_authorization"]["authorized_now"] is not False:
        raise ValueError("Clarification must precede the full fitting gate")

    gate_path = repository / "reports/tables/stage8_clinical_route_gate.csv"
    gates = pd.read_csv(gate_path).set_index("analysis")
    expected_rows = {
        "strict_section_10_clinical_hba1c_paired": (18262, 198, "FAIL"),
        "route_policy_section_20_clinical_core": (19018, 218, "PASS"),
    }
    for name, (n, events, status) in expected_rows.items():
        row = gates.loc[name]
        if (int(row["n"]), int(row["events"]), str(row["event_gate_status"])) != (
            n,
            events,
            status,
        ):
            raise ValueError(f"Preflight gate evidence differs for {name}")

    cohort_path = repository / "data/processed/nhanes_cohort_v2_4.parquet"
    if sha256_file(cohort_path) != EXPECTED_COHORT_SHA256:
        raise ValueError("Rebuilt NHANES cohort binding changed")
    if str(pq.read_schema(cohort_path).field("systolic_bp_mean").type) != "double":
        raise TypeError("systolic_bp_mean Arrow dtype is not double")
    source = pd.read_parquet(cohort_path, columns=["systolic_bp_mean"])["systolic_bp_mean"]
    alias = materialize_sbp_alias(source)
    if alias.dtype != np.dtype("float64"):
        raise TypeError("sbp_mean pipeline alias is not float64")

    forbidden_patterns = ("*.joblib", "*oof*.parquet", "*prediction*.parquet")
    artifact_dir = repository / "artifacts/stage8_clinical_preflight"
    forbidden = [item for pattern in forbidden_patterns for item in artifact_dir.glob(pattern)]
    if forbidden:
        raise ValueError(f"Pre-fitting clarification contains fitted outputs: {forbidden}")
    return {
        "status": "PASS",
        "artifact_sha256": sha256_file(path),
        "history_commit": EXPECTED_HISTORY_COMMIT,
        "history_confirmed": True,
        "section_10_status": "FAIL_198_EVENTS",
        "section_20_event_gate_only": "PASS_218_EVENTS",
        "fitting_authorized": False,
        "sbp_alias_equality": "PASS",
    }


__all__ = ["materialize_sbp_alias", "validate_stage8_clarification"]
