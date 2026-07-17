from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import cvd_risk.modeling.model_lock as model_lock
from cvd_risk.modeling.model_lock import (
    load_model_lock_config,
    validate_external_validation_lock,
    verify_binding_map,
)
from cvd_risk.provenance import sha256_file


def test_model_lock_config_requires_clean_synced_commit() -> None:
    config = load_model_lock_config("configs/model_lock.yaml")

    assert config["requirements"]["clean_working_tree"] is True
    assert config["requirements"]["head_equals_upstream"] is True
    assert config["requirements"]["post_chf_lineage_confirmed"] is True


def test_binding_verification_detects_changed_file(tmp_path: Path) -> None:
    bound = tmp_path / "bound.txt"
    bound.write_text("original", encoding="utf-8")
    bindings = {
        "bound": {"path": str(bound), "sha256": sha256_file(bound)}
    }

    verify_binding_map(bindings)
    bound.write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="Changed or absent binding"):
        verify_binding_map(bindings)


def test_model_lock_refuses_dirty_worktree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_lock, "_git", lambda *args: "?? audit-output.csv")

    with pytest.raises(RuntimeError, match="MODEL_LOCK_BLOCKED_DIRTY_WORKTREE"):
        model_lock._require_clean_synced_commit()


@pytest.mark.requires_local_research_artifacts
def test_serialized_pipeline_is_exact_stage4_copy_with_component_sha() -> None:
    manifest = json.loads(
        Path("artifacts/minimal_v1/serialized_pipeline_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["status"] == "SERIALIZED_WITHOUT_REFIT"
    assert manifest["refit_performed"] is False
    assert manifest["source"]["sha256"] == manifest["outputs"]["full_pipeline"][
        "sha256"
    ]
    verify_binding_map(manifest["outputs"])
    assert manifest["feature_schema"]["transformed_count"] == 15


def test_lock_validation_requires_ancestry_not_head_equality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bound = tmp_path / "bound.txt"
    bound.write_text("frozen", encoding="utf-8")
    binding = {"path": str(bound), "sha256": sha256_file(bound)}
    lock_path = tmp_path / "external_validation_lock.json"
    lock = {
        "status": "MODEL_LOCKED_AWAITING_LABEL_FREE_VALIDATION_PREDICTIONS",
        "locked_source_commit": "first-commit",
        "code_and_config_bindings": {"bound": binding},
        "model": {
            "artifact": binding,
            "preprocessing": binding,
            "estimator": binding,
            "calibrator": binding,
        },
        "development_evidence": {
            "cohort": binding,
            "oof_predictions": binding,
            "stage4_manifest": binding,
            "stage5_manifest": binding,
            "stage5a_manifest": binding,
            "stage5b_manifest": binding,
            "serialized_pipeline_manifest": binding,
            "chf_lineage": binding,
        },
    }
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    def fake_git(*args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return "second-commit"
        if args == ("rev-parse", "@{upstream}"):
            return "second-commit"
        if args[0] == "diff":
            return lock_path.as_posix()
        if args[0] == "status":
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(model_lock, "_git", fake_git)
    monkeypatch.setattr(
        model_lock.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )

    result = validate_external_validation_lock(lock_path=lock_path)

    assert result["locked_source_commit"] == "first-commit"
    assert result["current_head"] == "second-commit"
    assert result["locked_source_is_ancestor"] is True
