"""Portable metadata amendment for the immutable external-validation lock."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ORIGINAL_LOCK_COMMIT = "a75366b7ab29d7a29d18353dc18e2d29ef0b8218"
LOCKED_SOURCE_COMMIT = "5cace77785c3e7e3398ead12efc101ae235034ec"
ORIGINAL_LOCK_PATH = Path("artifacts/minimal_v1/external_validation_lock.json")
AMENDMENT_PATH = Path(
    "artifacts/minimal_v1/external_validation_lock_amendment_001.json"
)
ACCESS_PROVENANCE_PATH = Path("artifacts/audit/access_provenance_manifest.json")
QUARANTINE_MANIFEST_PATH = Path("artifacts/audit/quarantine_manifest.json")
SEALED_AUDIT_PATH = Path("artifacts/audit/sealed_audit_manifest.json")

NHIS_TEMPORAL_YEARS = [2010, 2011, 2012, 2013, 2014]
NHANES_EXTERNAL_CYCLES = [
    "1999-2000",
    "2001-2002",
    "2003-2004",
    "2005-2006",
    "2007-2008",
    "2009-2010",
    "2011-2012",
    "2013-2014",
]

METADATA_ONLY_PATHS = [
    ACCESS_PROVENANCE_PATH.as_posix(),
    SEALED_AUDIT_PATH.as_posix(),
    AMENDMENT_PATH.as_posix(),
    "src/cvd_risk/modeling/lock_amendment.py",
    "scripts/12_create_lock_amendment.py",
    "scripts/13_validate_lock_amendment.py",
    "tests/test_lock_amendment.py",
]

TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".lock",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def _git(*args: str, binary: bool = False) -> str | bytes:
    completed = subprocess.run(
        ["git", *args], capture_output=True, text=not binary, check=False
    )
    if completed.returncode != 0:
        stderr = completed.stderr
        message = stderr.decode() if isinstance(stderr, bytes) else stderr
        raise RuntimeError(message.strip() or "git command failed")
    return completed.stdout


def repository_root() -> Path:
    return Path(str(_git("rev-parse", "--show-toplevel")).strip()).resolve()


def _portable_sha256(path: Path, *, hash_mode: str) -> str:
    payload = path.read_bytes()
    if hash_mode == "normalized_text_lf":
        payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    elif hash_mode != "raw_bytes":
        raise ValueError(f"Unknown portable hash mode: {hash_mode}")
    return hashlib.sha256(payload).hexdigest()


def portable_binding(path: str | Path, *, root: Path) -> dict[str, str]:
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Portable binding must be repository-relative: {relative}")
    target = root / relative
    if not target.is_file():
        raise FileNotFoundError(f"Portable binding target is absent: {relative}")
    hash_mode = "normalized_text_lf" if relative.suffix in TEXT_SUFFIXES else "raw_bytes"
    return {
        "repository_relative_path": relative.as_posix(),
        "sha256": _portable_sha256(target, hash_mode=hash_mode),
        "hash_mode": hash_mode,
    }


def verify_portable_bindings(bindings: dict[str, Any], *, root: Path) -> None:
    for name, entry in bindings.items():
        relative = Path(entry["repository_relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Non-portable binding path: {name}")
        target = root / relative
        if not target.is_file() or _portable_sha256(
            target, hash_mode=entry["hash_mode"]
        ) != entry["sha256"]:
            raise ValueError(f"Changed or absent portable binding: {name}")


def _commit_file_sha256(commit: str, path: Path) -> str:
    payload = _git("show", f"{commit}:{path.as_posix()}", binary=True)
    assert isinstance(payload, bytes)
    return hashlib.sha256(payload).hexdigest()


def _access_provenance_payload(*, root: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_version": "2.4",
        "status": "ACCESS_PROVENANCE_RECORDED_WITHOUT_HELDOUT_UNSEAL",
        "recorded_on": "2026-07-15",
        "pre_amendment_labelled_artifact": {
            "individual_labels_were_materialized": True,
            "known_aggregate_n": 20598,
            "known_aggregate_cvd_events": 225,
            "known_prior_outputs": ["aggregate_n", "aggregate_event_count"],
            "performance_metrics_computed_before_amendment": "UNKNOWN",
            "predictor_outcome_associations_before_amendment": "UNKNOWN",
            "subgroup_or_cycle_results_before_amendment": "UNKNOWN",
            "quarantined_artifact_sha256": (
                "3ad309b476fb66c6ef2d78cf1027855bdca439f75195f03360463e3fd73285b5"
            ),
        },
        "post_amendment_controls": {
            "heldout_outcomes_used_for_model_selection": False,
            "heldout_individual_outcomes_unsealed_after_lock": False,
            "label_free_predictions_created": False,
            "next_permitted_action": "create_label_free_predictions",
        },
        "bindings": {
            "quarantine_manifest": portable_binding(
                QUARANTINE_MANIFEST_PATH, root=root
            ),
            "original_external_validation_lock": portable_binding(
                ORIGINAL_LOCK_PATH, root=root
            ),
        },
    }


def _sealed_audit_payload(*, root: Path) -> dict[str, Any]:
    aggregate_paths = {
        "canonical_stop_gate": Path("reports/stop_gate.json"),
        "event_counts": Path("reports/tables/event_counts.csv"),
        "sample_size_gate": Path("reports/tables/sample_size_gate.csv"),
        "survey_design_precision": Path(
            "reports/tables/survey_design_precision.csv"
        ),
        "effective_sample_size": Path("reports/tables/effective_sample_size.csv"),
    }
    return {
        "schema_version": 1,
        "protocol_version": "2.4",
        "status": "SEALED_AGGREGATE_AUDIT_RECORDED",
        "individual_heldout_records_exposed_by_manifest": False,
        "individual_heldout_outcomes_unsealed": False,
        "allowed_prelock_content": [
            "aggregate_n",
            "aggregate_event_count",
            "aggregate_cohort_flow",
            "unstratified_missingness",
            "design_based_prevalence_and_precision",
            "sample_size_and_event_stop_gates",
        ],
        "content_access_mode": "existing_aggregate_artifacts_hashed_without_label_join",
        "bindings": {
            name: portable_binding(path, root=root)
            for name, path in aggregate_paths.items()
        },
    }


def _portable_locked_bindings(
    lock: dict[str, Any], *, root: Path
) -> dict[str, dict[str, str]]:
    paths: dict[str, Path] = {
        name: Path(name) for name in lock["code_and_config_bindings"]
    }
    paths.update(
        {
            "model.full_pipeline": Path(
                "artifacts/minimal_v1/full_development_pipeline.joblib"
            ),
            "model.preprocessing": Path("artifacts/minimal_v1/preprocessing.joblib"),
            "model.estimator": Path("artifacts/minimal_v1/estimator.joblib"),
            "model.calibrator": Path("artifacts/minimal_v1/calibrator.joblib"),
            "development.cohort": Path("data/processed/nhis_cohort.parquet"),
            "development.oof_predictions": Path(
                "artifacts/stage4_baselines/oof_predictions.parquet"
            ),
            "development.stage4_manifest": Path(
                "artifacts/stage4_baselines/manifest.json"
            ),
            "development.stage5_manifest": Path(
                "artifacts/stage5_ml_candidates/manifest.json"
            ),
            "development.stage5a_manifest": Path(
                "artifacts/stage5a_planning/manifest.json"
            ),
            "development.stage5b_manifest": Path(
                "artifacts/stage5b_operating_point/manifest.json"
            ),
            "development.serialized_pipeline_manifest": Path(
                "artifacts/minimal_v1/serialized_pipeline_manifest.json"
            ),
            "development.chf_lineage": Path(
                "reports/tables/stage5b_chf_lineage.csv"
            ),
        }
    )
    return {name: portable_binding(path, root=root) for name, path in paths.items()}


def create_lock_amendment(*, force: bool = False) -> dict[str, Any]:
    root = repository_root()
    outputs = [ACCESS_PROVENANCE_PATH, SEALED_AUDIT_PATH, AMENDMENT_PATH]
    if any((root / path).exists() for path in outputs) and not force:
        raise FileExistsError("Lock-amendment metadata outputs already exist")
    lock_path = root / ORIGINAL_LOCK_PATH
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["locked_source_commit"] != LOCKED_SOURCE_COMMIT:
        raise ValueError("Unexpected locked source commit")
    lock_sha = _portable_sha256(lock_path, hash_mode="normalized_text_lf")
    if lock_sha != _commit_file_sha256(ORIGINAL_LOCK_COMMIT, ORIGINAL_LOCK_PATH):
        raise ValueError("Immutable original lock differs from commit a75366b7")

    access = _access_provenance_payload(root=root)
    sealed = _sealed_audit_payload(root=root)
    for path, payload in (
        (ACCESS_PROVENANCE_PATH, access),
        (SEALED_AUDIT_PATH, sealed),
    ):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    invariant = {
        "full_pipeline_sha256": lock["model"]["artifact"]["sha256"],
        "preprocessing_sha256": lock["model"]["preprocessing"]["sha256"],
        "estimator_sha256": lock["model"]["estimator"]["sha256"],
        "calibrator_sha256": lock["model"]["calibrator"]["sha256"],
        "cohort_sha256": lock["development_evidence"]["cohort"]["sha256"],
        "oof_predictions_sha256": lock["development_evidence"][
            "oof_predictions"
        ]["sha256"],
        "cutoff": lock["output_policy"]["secondary_screening_operating_point"][
            "cutoff"
        ],
        "cutoff_selection_rule": lock["output_policy"][
            "secondary_screening_operating_point"
        ]["selection_rule"],
    }
    amendment: dict[str, Any] = {
        "schema_version": 1,
        "amendment_id": "external_validation_lock_amendment_001",
        "status": "METADATA_BLOCKERS_CLOSED_NO_MODEL_CHANGE",
        "metadata_only": True,
        "original_lock": {
            "repository_relative_path": ORIGINAL_LOCK_PATH.as_posix(),
            "sha256": lock_sha,
            "commit": ORIGINAL_LOCK_COMMIT,
            "locked_source_commit": LOCKED_SOURCE_COMMIT,
            "immutable_original_not_modified": True,
        },
        "validation_cohorts": {
            "nhis_temporal_years": NHIS_TEMPORAL_YEARS,
            "nhanes_external_cycles": NHANES_EXTERNAL_CYCLES,
            "list_semantics": "explicit_enumeration_not_range_endpoints",
        },
        "portable_binding_policy": {
            "repository_root_source": "git_rev_parse_show_toplevel",
            "absolute_paths_required": False,
            "binding_fields": ["repository_relative_path", "sha256", "hash_mode"],
            "content_verification": (
                "normalized_lf_sha256_for_text_raw_sha256_for_binary_without_data_deserialization"
            ),
        },
        "portable_locked_bindings": _portable_locked_bindings(lock, root=root),
        "audit_manifest_bindings": {
            "access_provenance": portable_binding(
                ACCESS_PROVENANCE_PATH, root=root
            ),
            "quarantine": portable_binding(QUARANTINE_MANIFEST_PATH, root=root),
            "sealed_audit": portable_binding(SEALED_AUDIT_PATH, root=root),
        },
        "locked_values_unchanged": invariant,
        "heldout_outcomes_accessed_for_amendment": False,
        "heldout_outcomes_unsealed": False,
        "next_permitted_action_after_commit_and_validation": (
            "create_label_free_predictions"
        ),
        "metadata_only_commit_paths": METADATA_ONLY_PATHS,
    }
    target = root / AMENDMENT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(amendment, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    verify_amendment_content(amendment, lock=lock, root=root)
    return amendment


def verify_amendment_content(
    amendment: dict[str, Any], *, lock: dict[str, Any], root: Path
) -> None:
    if amendment.get("status") != "METADATA_BLOCKERS_CLOSED_NO_MODEL_CHANGE":
        raise ValueError("Unexpected amendment status")
    if amendment.get("metadata_only") is not True:
        raise ValueError("Amendment is not metadata-only")
    if amendment.get("heldout_outcomes_accessed_for_amendment") is not False:
        raise ValueError("Amendment reports held-out outcome access")
    cohorts = amendment["validation_cohorts"]
    if cohorts["nhis_temporal_years"] != NHIS_TEMPORAL_YEARS:
        raise ValueError("NHIS temporal years are not explicitly enumerated")
    if cohorts["nhanes_external_cycles"] != NHANES_EXTERNAL_CYCLES:
        raise ValueError("NHANES cycles are not explicitly enumerated")
    if amendment["original_lock"]["sha256"] != _portable_sha256(
        root / ORIGINAL_LOCK_PATH, hash_mode="normalized_text_lf"
    ):
        raise ValueError("Original lock SHA differs from amendment")
    verify_portable_bindings(amendment["portable_locked_bindings"], root=root)
    verify_portable_bindings(amendment["audit_manifest_bindings"], root=root)
    expected = {
        "full_pipeline_sha256": lock["model"]["artifact"]["sha256"],
        "preprocessing_sha256": lock["model"]["preprocessing"]["sha256"],
        "estimator_sha256": lock["model"]["estimator"]["sha256"],
        "calibrator_sha256": lock["model"]["calibrator"]["sha256"],
        "cohort_sha256": lock["development_evidence"]["cohort"]["sha256"],
        "oof_predictions_sha256": lock["development_evidence"][
            "oof_predictions"
        ]["sha256"],
        "cutoff": lock["output_policy"]["secondary_screening_operating_point"][
            "cutoff"
        ],
        "cutoff_selection_rule": lock["output_policy"][
            "secondary_screening_operating_point"
        ]["selection_rule"],
    }
    if amendment["locked_values_unchanged"] != expected:
        raise ValueError("Locked model/data/cutoff values changed in amendment")


def validate_committed_lock_amendment(
    *, amendment_path: str | Path = AMENDMENT_PATH
) -> dict[str, Any]:
    root = repository_root()
    amendment_file = root / amendment_path
    amendment = json.loads(amendment_file.read_text(encoding="utf-8"))
    lock = json.loads((root / ORIGINAL_LOCK_PATH).read_text(encoding="utf-8"))
    verify_amendment_content(amendment, lock=lock, root=root)
    commit_sha = _commit_file_sha256(ORIGINAL_LOCK_COMMIT, ORIGINAL_LOCK_PATH)
    if commit_sha != amendment["original_lock"]["sha256"]:
        raise ValueError("Original lock commit binding changed")
    current_head = str(_git("rev-parse", "HEAD")).strip()
    for ancestor_commit in (LOCKED_SOURCE_COMMIT, ORIGINAL_LOCK_COMMIT):
        status = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor_commit, current_head],
            check=False,
        ).returncode
        if status != 0:
            raise RuntimeError(f"Required ancestor is absent: {ancestor_commit}")
    changed = set(
        str(
            _git("diff", "--name-only", f"{ORIGINAL_LOCK_COMMIT}..{current_head}")
        ).splitlines()
    )
    allowed = set(amendment["metadata_only_commit_paths"])
    if changed != allowed:
        raise RuntimeError(
            f"Amendment commit contains non-metadata paths: {sorted(changed ^ allowed)}"
        )
    if str(_git("status", "--porcelain", "--untracked-files=all")).strip():
        raise RuntimeError("Amendment validation requires a clean working tree")
    upstream = str(_git("rev-parse", "@{upstream}")).strip()
    if current_head != upstream:
        raise RuntimeError("Amendment HEAD is not pushed to upstream")
    return {
        "status": "LOCK_AMENDMENT_VALID",
        "current_head": current_head,
        "original_lock_commit_is_ancestor": True,
        "locked_source_commit_is_ancestor": True,
        "original_lock_sha256": amendment["original_lock"]["sha256"],
        "portable_binding_count": len(amendment["portable_locked_bindings"]),
        "audit_manifest_sha_valid": True,
        "locked_model_data_cutoff_unchanged": True,
        "heldout_outcomes_accessed": False,
    }


__all__ = [
    "ACCESS_PROVENANCE_PATH",
    "AMENDMENT_PATH",
    "NHANES_EXTERNAL_CYCLES",
    "NHIS_TEMPORAL_YEARS",
    "ORIGINAL_LOCK_COMMIT",
    "ORIGINAL_LOCK_PATH",
    "SEALED_AUDIT_PATH",
    "create_lock_amendment",
    "portable_binding",
    "validate_committed_lock_amendment",
    "verify_amendment_content",
    "verify_portable_bindings",
]
