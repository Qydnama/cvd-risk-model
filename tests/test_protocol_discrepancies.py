from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cvd_risk.reporting.protocol_discrepancies import (
    NHANES_WEIGHT_DISCREPANCY_ID,
    load_protocol_discrepancies,
    protocol_discrepancy_checks,
)


def test_repository_weight_discrepancy_is_resolved_by_amendment_2_4() -> None:
    discrepancies = load_protocol_discrepancies("configs/protocol_discrepancies.yaml")
    row = discrepancies.set_index("id").loc[NHANES_WEIGHT_DISCREPANCY_ID]
    assert row["status"] == "resolved"
    assert bool(row["blocking"])
    checks = protocol_discrepancy_checks(discrepancies).set_index("check")
    assert (
        checks.loc[f"protocol_discrepancy:{NHANES_WEIGHT_DISCREPANCY_ID}", "status"]
        == "pass"
    )


def test_registry_cannot_disable_known_weight_guard(tmp_path: Path) -> None:
    path = tmp_path / "disabled.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "discrepancies": [
                    {
                        "id": NHANES_WEIGHT_DISCREPANCY_ID,
                        "title": "Known issue",
                        "status": "awaiting_user_decision",
                        "blocking": False,
                        "locked_protocol_rule": "Current rule",
                        "official_guidance": "Official rule",
                        "recommended_resolution": "Resolve it",
                        "resolution": None,
                        "official_sources": ["https://example.test/official"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must remain blocking"):
        load_protocol_discrepancies(path)


def test_resolved_discrepancy_requires_documented_resolution(tmp_path: Path) -> None:
    path = tmp_path / "undocumented.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "discrepancies": [
                    {
                        "id": NHANES_WEIGHT_DISCREPANCY_ID,
                        "title": "Known issue",
                        "status": "resolved",
                        "blocking": True,
                        "locked_protocol_rule": "Current rule",
                        "official_guidance": "Official rule",
                        "recommended_resolution": "Resolve it",
                        "resolution": None,
                        "official_sources": ["https://example.test/official"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires a resolution"):
        load_protocol_discrepancies(path)
