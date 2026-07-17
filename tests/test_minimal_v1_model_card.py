from __future__ import annotations

from pathlib import Path

from cvd_risk.modeling.minimal_v1_model_card import validate_minimal_v1_model_card


def test_minimal_v1_model_card_matches_locked_evidence() -> None:
    result = validate_minimal_v1_model_card(
        "artifacts/minimal_v1_model_card/status.json",
        root=Path.cwd(),
    )

    assert result["validation_status"] == "PASS"
    assert result["status"] == "DOCUMENTED_LOCKED_RESEARCH_MODEL_UNCHANGED"
    assert result["model_id"] == "minimal_v1"
    assert result["lock_changed"] is False
    assert result["refit_performed"] is False
    assert result["clinical_deployment_approved"] is False
    assert result["locked_binary_artifacts_verified"] == 4
    assert result["clinical_routes_exported"] == 0
