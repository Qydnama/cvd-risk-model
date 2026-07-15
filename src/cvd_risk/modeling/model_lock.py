"""Create an immutable model lock before held-out prediction or outcome access."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml
from joblib import dump, load  # type: ignore[import-untyped]

from cvd_risk.modeling.baselines import FittedBaselineModel
from cvd_risk.modeling.preprocessing import ROUTE_FEATURES
from cvd_risk.provenance import sha256_file


def load_model_lock_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Model-lock config schema_version must be 1")
    if payload.get("stage") != "stage6_external_validation_model_lock":
        raise ValueError("Unexpected model-lock stage")
    required = {
        "clean_working_tree": True,
        "head_equals_upstream": True,
        "heldout_outcomes_accessed": False,
        "post_chf_lineage_confirmed": True,
        "prelock_audit_status": "PRELOCK_OPERATING_POINT_AUDIT_COMPLETE",
    }
    if payload.get("requirements") != required:
        raise ValueError("Model-lock requirements differ from the locked protocol")
    if payload.get("output") != "artifacts/minimal_v1/external_validation_lock.json":
        raise ValueError("Unexpected model-lock output path")
    return payload


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


def _binding(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required lock input is absent: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def verify_binding_map(bindings: dict[str, Any]) -> None:
    for name, entry in bindings.items():
        path = Path(entry["path"])
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Changed or absent binding: {name}")


def export_serialized_pipeline(
    *, config_path: str | Path, force: bool = False
) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_model_lock_config(config_path)
    paths = {key: Path(value) for key, value in config["serialized_pipeline"].items()}
    source_path = paths.pop("source")
    manifest_path = paths.pop("manifest")
    known = [manifest_path, *paths.values()]
    if any(path.exists() for path in known) and not force:
        raise FileExistsError("Serialized final-pipeline artifacts already exist")
    pipeline = load(source_path)
    if not isinstance(pipeline, FittedBaselineModel):
        raise TypeError("Selected artifact is not a FittedBaselineModel")
    if pipeline.name != "LogisticRegression" or pipeline.route != "minimal":
        raise ValueError("Unexpected selected full-development pipeline")
    for path in known:
        path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, paths["full_pipeline"])
    dump(pipeline.preprocessor, paths["preprocessing"], compress=0)
    dump(pipeline.estimator, paths["estimator"], compress=0)
    dump(pipeline.calibrator, paths["calibrator"], compress=0)
    if sha256_file(source_path) != sha256_file(paths["full_pipeline"]):
        raise RuntimeError("Full-pipeline copy differs from the fitted Stage 4 artifact")
    input_order = list(ROUTE_FEATURES[pipeline.route])
    transformed_order = pipeline.preprocessor.get_feature_names_out().astype(str).tolist()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "stage": "final_full_development_pipeline_serialization",
        "status": "SERIALIZED_WITHOUT_REFIT",
        "refit_performed": False,
        "heldout_outcomes_accessed": False,
        "source": _binding(source_path),
        "outputs": {key: _binding(path) for key, path in paths.items()},
        "feature_schema": {
            "input_order": input_order,
            "input_types_and_levels": {
                "age": "numeric",
                "sex": ["female", "male"],
                "bmi": "numeric",
                "smoking_status": ["never", "former", "current"],
                "hypertension_dx": "boolean",
                "diabetes_dx": "boolean",
                "general_health": ["excellent", "very_good", "good", "fair", "poor"],
            },
            "transformed_order": transformed_order,
            "transformed_count": len(transformed_order),
        },
        "pipeline": {
            "name": pipeline.name,
            "route": pipeline.route,
            "probability_clip": pipeline.probability_clip,
            "has_fitted_preprocessor": hasattr(pipeline.preprocessor, "feature_names_out_"),
            "has_fitted_estimator": hasattr(pipeline.estimator, "coef_"),
            "has_fitted_calibrator": hasattr(pipeline.calibrator, "coef_"),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def _verify_stage5b_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("stage") != "stage5b_prelock_operating_point_audit":
        raise ValueError("Unexpected Stage 5b manifest")
    if manifest.get("status") != "PRELOCK_OPERATING_POINT_AUDIT_COMPLETE":
        raise ValueError("Stage 5b audit is incomplete")
    if manifest.get("heldout_outcomes_accessed") is not False:
        raise ValueError("Stage 5b did not preserve held-out outcome sealing")
    if manifest.get("external_validation_lock_created") is not False:
        raise ValueError("An external-validation lock already exists")
    if manifest.get("cohort_and_oof_post_chf_amendment") is not True:
        raise ValueError("Post-CHF cohort/OOF lineage is not confirmed")
    if manifest.get("sigmoid_comparison_blocker_closed") is not True:
        raise ValueError("Uncalibrated-versus-sigmoid blocker is not closed")
    policy = manifest.get("secondary_operating_point", {})
    if (
        policy.get("target_survey_weighted_sensitivity") != 0.80
        or policy.get("primary_model_output") != "continuous_five_year_risk"
        or policy.get("role") != "secondary_screening_operating_point"
        or policy.get("achieved_survey_weighted_sensitivity", 0.0) < 0.80
    ):
        raise ValueError("Stage 5b operating point differs from the approved policy")
    verify_binding_map(manifest["input_bindings"])
    verify_binding_map(manifest["outputs"])
    return manifest


def _require_clean_synced_commit() -> dict[str, str]:
    dirty = _git("status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise RuntimeError(
            "MODEL_LOCK_BLOCKED_DIRTY_WORKTREE: commit the pre-lock audit first"
        )
    head = _git("rev-parse", "HEAD")
    upstream = _git("rev-parse", "@{upstream}")
    if head != upstream:
        raise RuntimeError("MODEL_LOCK_BLOCKED_HEAD_NOT_PUSHED")
    return {
        "commit": head,
        "tree": _git("rev-parse", "HEAD^{tree}"),
        "branch": _git("branch", "--show-current"),
        "upstream": _git("rev-parse", "--abbrev-ref", "@{upstream}"),
        "remote": _git("remote", "get-url", "origin"),
    }


def create_external_validation_lock(
    *, config_path: str | Path
) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_model_lock_config(config_path)
    output_path = Path(config["output"])
    if output_path.exists():
        raise FileExistsError(
            "External-validation lock already exists and is immutable"
        )
    git_state = _require_clean_synced_commit()
    stage5b_path = Path(config["stage5b_manifest"])
    stage5b = _verify_stage5b_manifest(stage5b_path)
    serialized_manifest_path = Path(config["serialized_pipeline_manifest"])
    serialized = json.loads(serialized_manifest_path.read_text(encoding="utf-8"))
    if (
        serialized.get("status") != "SERIALIZED_WITHOUT_REFIT"
        or serialized.get("refit_performed") is not False
        or serialized.get("heldout_outcomes_accessed") is not False
    ):
        raise ValueError("Final serialized pipeline provenance is invalid")
    verify_binding_map(serialized["outputs"])
    selection_path = Path(config["selected_model_config"])
    selection = yaml.safe_load(selection_path.read_text(encoding="utf-8"))
    primary = selection["primary"]
    if (
        primary["model"] != "LogisticRegression"
        or primary["route"] != "minimal"
        or primary["calibration"] != "sigmoid"
    ):
        raise ValueError("Unexpected model selection at lock time")
    source_model_path = Path(primary["artifact"])
    model_path = Path(config["serialized_pipeline"]["full_pipeline"])
    stage4_path = Path(config["stage4_manifest"])
    stage4 = json.loads(stage4_path.read_text(encoding="utf-8"))
    modeling_config = yaml.safe_load(Path("configs/modeling.yaml").read_text(encoding="utf-8"))
    model_entry = next(
        entry for entry in stage4["models"] if entry["name"] == primary["model"]
    )
    if (
        sha256_file(source_model_path) != model_entry["sha256"]
        or sha256_file(model_path) != model_entry["sha256"]
    ):
        raise ValueError("Selected pipeline artifact differs from Stage 4")

    tracked_files = [
        config_path,
        selection_path,
        Path(config["operating_point_config"]),
        *[Path(item) for item in config["pipeline_sources"]],
        *[Path(item) for item in config["runtime_bindings"]],
        stage4_path,
        Path(config["stage5_manifest"]),
        Path(config["stage5a_manifest"]),
        stage5b_path,
        serialized_manifest_path,
        *[Path(value) for key, value in config["serialized_pipeline"].items() if key != "source"],
    ]
    for tracked in tracked_files:
        _git("ls-files", "--error-unmatch", tracked.as_posix())

    policy = stage5b["secondary_operating_point"]
    stage0 = json.loads(Path("artifacts/audit/stage0_status.json").read_text(encoding="utf-8"))
    experiment = yaml.safe_load(Path("configs/experiment.yaml").read_text(encoding="utf-8"))
    stage5a = json.loads(Path(config["stage5a_manifest"]).read_text(encoding="utf-8"))
    stage5 = json.loads(Path(config["stage5_manifest"]).read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "lock_type": "external_validation_model_lock",
        "status": "MODEL_LOCKED_AWAITING_LABEL_FREE_VALIDATION_PREDICTIONS",
        "immutable": True,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol_version": config["protocol_version"],
        "locked_source_commit": git_state["commit"],
        "heldout_outcomes_accessed_before_lock": False,
        "model": {
            "name": primary["model"],
            "route": primary["route"],
            "calibration": primary["calibration"],
            "artifact": _binding(model_path),
            "preprocessing": serialized["outputs"]["preprocessing"],
            "estimator": serialized["outputs"]["estimator"],
            "calibrator": serialized["outputs"]["calibrator"],
            "feature_schema": serialized["feature_schema"],
            "feature_names": model_entry["feature_names"],
            "predictor_parameters": model_entry["predictor_parameters"],
            "probability_clip": modeling_config["calibration"]["probability_clip"],
        },
        "output_policy": {
            "primary": "continuous_five_year_risk",
            "secondary_screening_operating_point": {
                "cutoff": policy["exact_cutoff"],
                "selection_rule": policy["selection_rule"],
                "classification_rule": policy["classification_rule"],
                "target_survey_weighted_sensitivity": policy[
                    "target_survey_weighted_sensitivity"
                ],
                "development_oof_sensitivity": policy[
                    "achieved_survey_weighted_sensitivity"
                ],
                "interpretation": policy["interpretation"],
                "next_higher_cutoff": policy["next_higher_cutoff"],
                "next_higher_cutoff_sensitivity": policy[
                    "next_higher_cutoff_sensitivity"
                ],
            },
        },
        "development_evidence": {
            "cohort": stage5b["input_bindings"]["cohort"],
            "oof_predictions": stage5b["input_bindings"]["oof_predictions"],
            "stage4_manifest": _binding(stage4_path),
            "stage5_manifest": _binding(Path(config["stage5_manifest"])),
            "stage5a_manifest": _binding(Path(config["stage5a_manifest"])),
            "stage5b_manifest": _binding(stage5b_path),
            "serialized_pipeline_manifest": _binding(serialized_manifest_path),
            "post_chf_lineage_confirmed": True,
            "chf_lineage": stage5b["outputs"]["lineage"],
            "sigmoid_comparison_blocker_closed": True,
            "cluster_bootstrap": stage5b["cluster_bootstrap"],
            "stability_ranges": stage5b["stability_ranges"],
        },
        "git": git_state,
        "environment": {
            "python": platform.python_version(),
            "r": stage0["checks"]["rscript"]["version"],
            "r_packages": stage0["r_packages"],
            "python_packages": {
                package: version(package)
                for package in (
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "joblib",
                    "pyarrow",
                    "xgboost",
                )
            },
        },
        "seeds": {
            "experiment": experiment["experiment"]["random_seed"],
            "stage4_baselines": stage4["seed"],
            "stage5_ml_candidates": stage5["seed"],
            "stage5a_pmvalsampsize": stage5a["pmvalsampsize_arguments"]["seed"],
            "stage5b_cluster_bootstrap": stage5b["cluster_bootstrap"]["seed"],
            "stage5b_calibration_comparison": stage5b[
                "calibration_comparison_evidence"
            ]["seed"],
        },
        "code_and_config_bindings": {
            tracked.as_posix(): _binding(tracked) for tracked in tracked_files
        },
        "validation_sequence": {
            "next": "create_label_free_predictions",
            "model_changes_after_lock_forbidden": True,
            "heldout_predictor_analysis_before_unseal_allowed": True,
            "heldout_outcome_access_before_prediction_freeze_forbidden": True,
            "nhis_temporal_years": [2010, 2014],
            "nhanes_external_cycles": [1999, 2014],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def validate_external_validation_lock(*, lock_path: str | Path) -> dict[str, Any]:
    path = Path(lock_path)
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("status") != "MODEL_LOCKED_AWAITING_LABEL_FREE_VALIDATION_PREDICTIONS":
        raise ValueError("Unexpected external-validation lock status")
    locked_source = lock["locked_source_commit"]
    current_head = _git("rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", locked_source, current_head],
        check=False,
    ).returncode
    if ancestor != 0:
        raise RuntimeError("LOCKED_SOURCE_COMMIT_IS_NOT_ANCESTOR")
    verify_binding_map(lock["code_and_config_bindings"])
    verify_binding_map(
        {
            "pipeline": lock["model"]["artifact"],
            "preprocessing": lock["model"]["preprocessing"],
            "estimator": lock["model"]["estimator"],
            "calibrator": lock["model"]["calibrator"],
            "cohort": lock["development_evidence"]["cohort"],
            "oof_predictions": lock["development_evidence"]["oof_predictions"],
            "stage5_manifest": lock["development_evidence"]["stage5_manifest"],
            "stage4_manifest": lock["development_evidence"]["stage4_manifest"],
            "stage5a_manifest": lock["development_evidence"]["stage5a_manifest"],
            "stage5b_manifest": lock["development_evidence"]["stage5b_manifest"],
            "serialized_pipeline_manifest": lock["development_evidence"][
                "serialized_pipeline_manifest"
            ],
            "chf_lineage": lock["development_evidence"]["chf_lineage"],
        }
    )
    changed = set(_git("diff", "--name-only", f"{locked_source}..{current_head}").splitlines())
    allowed = {path.as_posix()}
    if changed != allowed:
        raise RuntimeError(f"POST_LOCK_COMMIT_CHANGED_FORBIDDEN_FILES: {sorted(changed)}")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("POST_LOCK_WORKTREE_NOT_CLEAN")
    if current_head != _git("rev-parse", "@{upstream}"):
        raise RuntimeError("POST_LOCK_HEAD_NOT_PUSHED")
    return {
        "status": "LOCK_VALID",
        "locked_source_commit": locked_source,
        "current_head": current_head,
        "locked_source_is_ancestor": True,
        "changed_files_after_locked_source": sorted(changed),
        "all_locked_sha_bindings_valid": True,
    }


__all__ = [
    "create_external_validation_lock",
    "export_serialized_pipeline",
    "load_model_lock_config",
    "verify_binding_map",
    "validate_external_validation_lock",
]
