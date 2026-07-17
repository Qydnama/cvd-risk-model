"""Validate the research-only Stage 9 metadata for locked minimal_v1."""

from __future__ import annotations

import hashlib
import json
import subprocess
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd

from cvd_risk.modeling.minimal_v1_model_card import (
    PIPELINE_HASHES,
    validate_minimal_v1_model_card,
)
from cvd_risk.modeling.preprocessing import ROUTE_FEATURES, ROUTE_PARAMETER_COUNTS
from cvd_risk.schemas import validate_fast_feature_schema

AUTHORIZATION_COMMIT = "7930f52617c504f17dbbebe6f7fedcafcb416842"
REPOSITORY_BASELINE_COMMIT = "e3b6977fe4e8165037e3468270a04245667a2f1d"
MODEL_SOURCE_COMMIT = "5cace77785c3e7e3398ead12efc101ae235034ec"
MODEL_CARD_COMMIT = "abc968a7a48f3dda739c1f22290f6d6366c76916"
LOCKED_CUTOFF = "0.007469058716065273"
PLANNED_TAG = "minimal-v1-research-v1.0.0"

AUTHORIZATION_FILES = {
    "status": {
        "path": "artifacts/stage9_research_release_authorization/status.json",
        "git_blob_sha1": "944ff0d02cacbafc4d531aa1c1e2472f173597ec",
        "normalized_lf_sha256": (
            "f1e1a648e1fdc960433dca9d65c00a740d4faa7045024380421b63f195cfde60"
        ),
    },
    "report": {
        "path": "reports/stage9_research_release_authorization.md",
        "git_blob_sha1": "8b722d93c4adde3fb5787011c878d308cef8162c",
        "normalized_lf_sha256": (
            "2f71ef31783b6e56af9ddfb5b7006fed376b88d543687c6b4fa26546f6bd639b"
        ),
    },
}

DOCUMENTATION_HASHES = {
    "artifacts/minimal_v1/serialized_pipeline_manifest.json": (
        "691ed59e738bc4a8ad50d05e5faa2718059eb303f626de77288b98d4e0967a0c"
    ),
    "artifacts/minimal_v1_model_card/status.json": (
        "6693af48fcd8b58ca96f320956381b6aaac331072eaed8c056507bd9540d40a4"
    ),
    "reports/minimal_v1_model_card.md": (
        "16db5b6b87707ef92af6d49e6f78fae56ea0b40de97d8bab7d1b7763e8841510"
    ),
    "uv.lock": "8b98259195bfa9e92f1e23679548ab0a2c873c6fb1fd293270bc7a711faed672",
}

INPUT_SCHEMA_PATH = "artifacts/minimal_v1_research_release/input_schema.json"
INPUT_SCHEMA_NORMALIZED_LF_SHA256 = (
    "81ef33867f6f2b1144d65e6f7b20a1feddea27660eb9dc45a7fba1dd2d7aefd1"
)
INPUT_SCHEMA_GIT_BLOB = "24f1dec9ec223fad4991403c78505c84de46d573"
RELEASE_MANIFEST_PATH = "artifacts/minimal_v1_research_release/release_manifest.json"

EXPECTED_RUNTIME_VERSIONS = {
    "joblib": "1.5.3",
    "numpy": "2.5.1",
    "pandas": "2.3.3",
    "scikit-learn": "1.9.0",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_text_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


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


def _require_ancestor(root: Path, commit: str) -> None:
    if _git(root, "cat-file", "-t", commit) != "commit":
        raise ValueError(f"Stage 9 evidence commit is unavailable: {commit}")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"Stage 9 evidence is not an ancestor of HEAD: {commit}")


def _synthetic_input() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [40.0, 60.0, 79.0],
            "sex": ["female", "male", "female"],
            "bmi": [21.5, 27.4, 35.2],
            "smoking_status": ["never", "former", "current"],
            "hypertension_dx": [False, True, True],
            "diabetes_dx": [False, False, True],
            "general_health": ["excellent", "good", "poor"],
        }
    )


def validate_minimal_v1_research_input(
    frame: pd.DataFrame,
    schema: dict[str, Any],
) -> None:
    """Validate complete research input without invoking fitted imputation."""

    expected = list(ROUTE_FEATURES["minimal"])
    if schema.get("ordered_features") != expected or schema.get("required") != expected:
        raise ValueError("Stage 9 input feature order differs from locked minimal_v1")
    if list(frame.columns) != expected:
        raise ValueError("Research input columns must exactly match the locked order")
    if frame.empty:
        raise ValueError("Research input must contain at least one record")
    validate_fast_feature_schema(frame, allow_missing=False)

    properties = schema["properties"]
    for column in ("age", "bmi"):
        values = pd.to_numeric(frame[column], errors="raise")
        if not values.between(
            properties[column]["minimum"],
            properties[column]["maximum"],
            inclusive="both",
        ).all():
            raise ValueError(f"{column} is outside the Stage 9 research-input range")


def _validate_authorization(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    _require_ancestor(root, AUTHORIZATION_COMMIT)
    for binding in AUTHORIZATION_FILES.values():
        relative = binding["path"]
        if _git(root, "rev-parse", f"{AUTHORIZATION_COMMIT}:{relative}") != binding[
            "git_blob_sha1"
        ]:
            raise ValueError(f"Stage 9 authorization blob changed at commit: {relative}")
        if _git(root, "diff", "--name-only", AUTHORIZATION_COMMIT, "--", relative):
            raise ValueError(f"Stage 9 authorization changed after commit: {relative}")
        if _normalized_text_sha256(root / relative) != binding["normalized_lf_sha256"]:
            raise ValueError(f"Stage 9 authorization SHA-256 changed: {relative}")

    status = json.loads(
        (root / AUTHORIZATION_FILES["status"]["path"]).read_text(encoding="utf-8")
    )
    required = {
        "status": "AUTHORIZED_RESEARCH_ONLY_METADATA_PACKAGING",
        "authorized_model_id": "minimal_v1",
        "research_use_only": True,
        "existing_locked_pipeline_only": True,
        "refit_allowed": False,
        "new_serialized_estimator_allowed": False,
        "binary_repackaging_allowed": False,
        "heldout_label_access_allowed": False,
        "clinical_deployment_allowed": False,
        "api_route_allowed": False,
        "web_routing_allowed": False,
        "routing_manifest_allowed": False,
        "clinical_core_export_allowed": False,
        "clinical_hba1c_export_allowed": False,
        "primary_export_gate": "FAIL",
        "external_validation_interpretation": "UNDERPOWERED_EXPLORATORY",
    }
    if any(status.get(key) != value for key, value in required.items()):
        raise ValueError("Stage 9 research-only authorization changed")
    if manifest.get("authorization") != {
        "commit": AUTHORIZATION_COMMIT,
        "status_path": AUTHORIZATION_FILES["status"]["path"],
        "status_git_blob_sha1": AUTHORIZATION_FILES["status"]["git_blob_sha1"],
        "status_normalized_lf_sha256": AUTHORIZATION_FILES["status"][
            "normalized_lf_sha256"
        ],
        "report_path": AUTHORIZATION_FILES["report"]["path"],
        "report_git_blob_sha1": AUTHORIZATION_FILES["report"]["git_blob_sha1"],
        "report_normalized_lf_sha256": AUTHORIZATION_FILES["report"][
            "normalized_lf_sha256"
        ],
    }:
        raise ValueError("Stage 9 manifest authorization binding changed")
    return required


def _validate_schema(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    schema_path = root / INPUT_SCHEMA_PATH
    if _normalized_text_sha256(schema_path) != INPUT_SCHEMA_NORMALIZED_LF_SHA256:
        raise ValueError("Stage 9 portable input schema SHA-256 changed")
    if _git(root, "hash-object", INPUT_SCHEMA_PATH) != INPUT_SCHEMA_GIT_BLOB:
        raise ValueError("Stage 9 portable input schema git blob changed")
    if manifest.get("portable_contract") != {
        "path": INPUT_SCHEMA_PATH,
        "normalized_lf_sha256": INPUT_SCHEMA_NORMALIZED_LF_SHA256,
        "git_blob_sha1": INPUT_SCHEMA_GIT_BLOB,
    }:
        raise ValueError("Stage 9 portable input contract binding changed")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    required = {
        "artifact_role": "RESEARCH_ONLY_PORTABLE_INPUT_CONTRACT",
        "model_id": "minimal_v1",
        "route": "minimal",
        "type": "object",
        "additionalProperties": False,
    }
    if any(schema.get(key) != value for key, value in required.items()):
        raise ValueError("Stage 9 portable input schema identity changed")
    governance = schema.get("governance", {})
    if governance != {
        "research_use_only": True,
        "clinical_deployment_allowed": False,
        "api_route_allowed": False,
        "web_routing_allowed": False,
        "external_validation_interpretation": "UNDERPOWERED_EXPLORATORY",
    }:
        raise ValueError("Stage 9 portable input governance changed")
    missingness = schema.get("missingness_policy", {})
    if missingness.get("personal_inference") != (
        "ALL_SEVEN_FIELDS_REQUIRED_NO_FALLBACK_ROUTE_AUTHORIZED"
    ):
        raise ValueError("Stage 9 missingness policy changed")
    if missingness.get("pipeline_imputation_authorized_for_silent_personal_input_substitution"):
        raise ValueError("Stage 9 schema permits silent personal-input imputation")
    validate_minimal_v1_research_input(_synthetic_input(), schema)
    return schema


def _validate_manifest_and_files(root: Path, manifest: dict[str, Any]) -> None:
    required = {
        "status": "RESEARCH_ONLY_SOURCE_SNAPSHOT_CANDIDATE",
        "model_id": "minimal_v1",
        "protocol_version": "2.4",
        "release_name": "minimal_v1 research snapshot v1.0.0",
        "planned_tag": PLANNED_TAG,
        "tag_target_policy": "MERGED_MAIN_COMMIT_WITH_SUCCESSFUL_POST_MERGE_CI",
        "repository_baseline_commit": REPOSITORY_BASELINE_COMMIT,
        "model_source_commit": MODEL_SOURCE_COMMIT,
        "model_card_commit": MODEL_CARD_COMMIT,
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValueError("Stage 9 release identity changed")
    for commit in (REPOSITORY_BASELINE_COMMIT, MODEL_SOURCE_COMMIT, MODEL_CARD_COMMIT):
        _require_ancestor(root, commit)

    locked = manifest.get("locked_model", {})
    if locked != {
        "route": "minimal",
        "model": "LogisticRegression",
        "calibration": "sigmoid",
        "predictor_parameters": 15,
        "probability_clip": 0.000001,
        "locked_cutoff": float(LOCKED_CUTOFF),
        "refit_performed": False,
        "recalibration_performed": False,
    }:
        raise ValueError("Stage 9 locked-model identity changed")

    expected_binaries = {
        "full_development_pipeline": {
            "path": "artifacts/minimal_v1/full_development_pipeline.joblib",
            "sha256": PIPELINE_HASHES["full_development_pipeline.joblib"],
        },
        "preprocessing": {
            "path": "artifacts/minimal_v1/preprocessing.joblib",
            "sha256": PIPELINE_HASHES["preprocessing.joblib"],
        },
        "estimator": {
            "path": "artifacts/minimal_v1/estimator.joblib",
            "sha256": PIPELINE_HASHES["estimator.joblib"],
        },
        "calibrator": {
            "path": "artifacts/minimal_v1/calibrator.joblib",
            "sha256": PIPELINE_HASHES["calibrator.joblib"],
        },
    }
    if manifest.get("locked_binary_files") != expected_binaries:
        raise ValueError("Stage 9 locked binary inventory changed")
    for binding in expected_binaries.values():
        if _sha256(root / binding["path"]) != binding["sha256"]:
            raise ValueError(f"Stage 9 locked binary changed: {binding['path']}")

    expected_documentation = {
        "serialized_pipeline_manifest": {
            "path": "artifacts/minimal_v1/serialized_pipeline_manifest.json",
            "normalized_lf_sha256": DOCUMENTATION_HASHES[
                "artifacts/minimal_v1/serialized_pipeline_manifest.json"
            ],
        },
        "model_card_status": {
            "path": "artifacts/minimal_v1_model_card/status.json",
            "normalized_lf_sha256": DOCUMENTATION_HASHES[
                "artifacts/minimal_v1_model_card/status.json"
            ],
        },
        "model_card_report": {
            "path": "reports/minimal_v1_model_card.md",
            "normalized_lf_sha256": DOCUMENTATION_HASHES[
                "reports/minimal_v1_model_card.md"
            ],
        },
        "python_lock": {
            "path": "uv.lock",
            "normalized_lf_sha256": DOCUMENTATION_HASHES["uv.lock"],
        },
    }
    if manifest.get("documentation_bindings") != expected_documentation:
        raise ValueError("Stage 9 documentation bindings changed")
    for binding in expected_documentation.values():
        if _normalized_text_sha256(root / binding["path"]) != binding[
            "normalized_lf_sha256"
        ]:
            raise ValueError(f"Stage 9 documentation changed: {binding['path']}")

    policy = manifest.get("release_asset_policy", {})
    if policy != {
        "release_type": "SOURCE_AND_METADATA_RESEARCH_SNAPSHOT",
        "separate_model_binary_attachment": False,
        "binary_repackaging": False,
        "production_bundle": False,
        "api_or_web_artifact": False,
        "routing_manifest": False,
    }:
        raise ValueError("Stage 9 release asset policy changed")
    governance = manifest.get("governance", {})
    if governance != {
        "research_use_only": True,
        "clinical_deployment_allowed": False,
        "primary_export_gate": "FAIL",
        "external_validation_interpretation": "UNDERPOWERED_EXPLORATORY",
        "independent_external_validation_claim_allowed": False,
        "validation_for_kazakhstan_claim_allowed": False,
    }:
        raise ValueError("Stage 9 release governance changed")
    if manifest.get("runtime_contract") != {
        "python": ">=3.12,<3.13",
        "joblib": "1.5.3",
        "numpy": "2.5.1",
        "pandas": "2.3.3",
        "scikit_learn": "1.9.0",
        "environment_source": "uv.lock",
        "smoke_test_input": "SYNTHETIC_COMPLETE_CANONICAL_RECORDS_ONLY",
        "participant_data_accessed": False,
        "heldout_labels_accessed": False,
    }:
        raise ValueError("Stage 9 runtime contract changed")


def _run_smoke_test(root: Path, schema: dict[str, Any]) -> np.ndarray:
    versions = {package: version(package) for package in EXPECTED_RUNTIME_VERSIONS}
    if versions != EXPECTED_RUNTIME_VERSIONS:
        raise ValueError(f"Stage 9 runtime differs from uv.lock: {versions}")

    # Hashes are verified before this trusted, repository-owned pickle is deserialized.
    # NumPy 2.5 deprecates joblib's internal array-shape restoration used by this frozen
    # artifact; suppress only that upstream compatibility warning during the trusted load.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated in NumPy 2.5.*",
            category=DeprecationWarning,
            module=r"joblib\.numpy_pickle",
        )
        model = joblib.load(root / "artifacts/minimal_v1/full_development_pipeline.joblib")
    if (model.name, model.route, model.probability_clip) != (
        "LogisticRegression",
        "minimal",
        0.000001,
    ):
        raise ValueError("Stage 9 deserialized pipeline identity changed")
    if type(model.estimator).__name__ != "LogisticRegression":
        raise ValueError("Stage 9 estimator class changed")
    if type(model.calibrator).__name__ != "LogisticRegression":
        raise ValueError("Stage 9 sigmoid calibrator class changed")
    if model.preprocessor.n_features_in_ != len(ROUTE_FEATURES["minimal"]):
        raise ValueError("Stage 9 preprocessor input count changed")
    if len(model.preprocessor.get_feature_names_out()) != ROUTE_PARAMETER_COUNTS["minimal"]:
        raise ValueError("Stage 9 transformed parameter count changed")

    frame = _synthetic_input()
    validate_minimal_v1_research_input(frame, schema)
    calibrated = np.asarray(model.predict_proba(frame, calibrated=True), dtype=np.float64)
    uncalibrated = np.asarray(model.predict_proba(frame, calibrated=False), dtype=np.float64)
    for label, probability in (("calibrated", calibrated), ("uncalibrated", uncalibrated)):
        if probability.shape != (len(frame),):
            raise ValueError(f"Stage 9 {label} smoke predictions have the wrong shape")
        valid_range = ((probability >= 0) & (probability <= 1)).all()
        if not np.isfinite(probability).all() or not valid_range:
            raise ValueError(f"Stage 9 {label} smoke predictions are invalid")
    return calibrated


def validate_stage9_research_release(
    manifest_path: str | Path = RELEASE_MANIFEST_PATH,
    *,
    root: str | Path,
) -> dict[str, Any]:
    """Validate research-only metadata and smoke-test the unchanged locked pipeline."""

    repository = Path(root).resolve()
    path = Path(manifest_path)
    if not path.is_absolute():
        path = repository / path
    manifest = json.loads(path.read_text(encoding="utf-8"))

    _validate_manifest_and_files(repository, manifest)
    authorization = _validate_authorization(repository, manifest)
    schema = _validate_schema(repository, manifest)
    card = validate_minimal_v1_model_card(
        "artifacts/minimal_v1_model_card/status.json",
        root=repository,
    )

    closure = json.loads(
        (repository / "artifacts/stage8_closure/status.json").read_text(encoding="utf-8")
    )
    if closure.get("PRIMARY_EXPORT_GATE") != "FAIL":
        raise ValueError("Stage 9 changed the Stage 8 primary export gate")
    if closure.get("WEB_ROUTING_ALLOWED") is not False:
        raise ValueError("Stage 9 changed the Stage 8 web-routing prohibition")

    probability = _run_smoke_test(repository, schema)
    report = (repository / "reports/stage9_minimal_v1_research_release.md").read_text(
        encoding="utf-8"
    )
    required_tokens = (
        "RESEARCH_ONLY_SOURCE_SNAPSHOT_CANDIDATE",
        "CLINICAL_DEPLOYMENT_ALLOWED = FALSE",
        "WEB_ROUTING_ALLOWED = FALSE",
        "PRIMARY_EXPORT_GATE = FAIL",
        "UNDERPOWERED_EXPLORATORY",
        PLANNED_TAG,
        LOCKED_CUTOFF,
    )
    if any(token not in report for token in required_tokens):
        raise ValueError("Stage 9 release report lacks a required governance disclosure")

    return {
        "validation_status": "PASS",
        "release_status": manifest["status"],
        "model_id": manifest["model_id"],
        "planned_tag": PLANNED_TAG,
        "research_use_only": authorization["research_use_only"],
        "refit_performed": False,
        "heldout_labels_accessed": False,
        "clinical_deployment_allowed": False,
        "api_or_web_route_created": False,
        "primary_export_gate": closure["PRIMARY_EXPORT_GATE"],
        "validation_interpretation": card["validation_interpretation"],
        "locked_pipeline_sha256": PIPELINE_HASHES["full_development_pipeline.joblib"],
        "locked_cutoff": LOCKED_CUTOFF,
        "locked_binary_artifacts_verified": len(PIPELINE_HASHES),
        "synthetic_smoke_records": int(len(probability)),
        "synthetic_predictions_finite": bool(np.isfinite(probability).all()),
    }


__all__ = ["validate_minimal_v1_research_input", "validate_stage9_research_release"]
