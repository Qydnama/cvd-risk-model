from __future__ import annotations

import json
from pathlib import Path

import pytest

from cvd_risk.provenance import reject_quarantined_input, sha256_file, verify_quarantine


def _manifest(tmp_path: Path, artifact: Path) -> Path:
    path = tmp_path / "quarantine.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_version": "2.4",
                "artifacts": [
                    {
                        "artifact_id": "old",
                        "original_path": str(tmp_path / "old-location.parquet"),
                        "quarantine_path": str(artifact),
                        "sha256": sha256_file(artifact),
                        "size_bytes": artifact.stat().st_size,
                        "status": "quarantined_pre_amendment",
                        "blocked_uses": ["model_training"],
                        "reason": "pre-amendment",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_repository_quarantine_is_intact() -> None:
    payload = verify_quarantine("artifacts/audit/quarantine_manifest.json")
    assert payload["protocol_version"] == "2.4"


def test_quarantined_digest_cannot_be_reused_from_another_path(tmp_path: Path) -> None:
    quarantined = tmp_path / "quarantined.parquet"
    quarantined.write_bytes(b"old labelled cohort")
    manifest = _manifest(tmp_path, quarantined)
    copied = tmp_path / "copied.parquet"
    copied.write_bytes(quarantined.read_bytes())

    with pytest.raises(ValueError, match="is quarantined"):
        reject_quarantined_input(copied, manifest)


def test_quarantine_verification_rejects_restored_original_path(tmp_path: Path) -> None:
    quarantined = tmp_path / "quarantined.parquet"
    quarantined.write_bytes(b"old labelled cohort")
    manifest = _manifest(tmp_path, quarantined)
    (tmp_path / "old-location.parquet").write_bytes(b"restored")

    with pytest.raises(ValueError, match="remains accessible"):
        verify_quarantine(manifest)
