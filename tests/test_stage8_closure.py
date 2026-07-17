from __future__ import annotations

from pathlib import Path

from cvd_risk.modeling.stage8_closure import validate_stage8_closure


def test_stage8_closure_matches_immutable_evidence() -> None:
    result = validate_stage8_closure(
        "artifacts/stage8_closure/status.json",
        root=Path.cwd(),
    )

    assert result["status"] == "CLOSED_PRIMARY_GATE_FAILED_EXPLORATORY_ANALYSES_RETAINED"
    assert result["PRIMARY_EXPORT_GATE"] == "FAIL"
    assert result["CLINICAL_CORE_EXPORT_ALLOWED"] is False
    assert result["CLINICAL_HBA1C_FITTING"] == "NOT_AUTHORIZED_NOT_PERFORMED"
    assert result["WEB_ROUTING_ALLOWED"] is False
    assert result["deployable_clinical_artifacts"] == 0
