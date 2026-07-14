"""Protocol provenance and quarantine guards."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest for a local file."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_quarantine_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate the immutable-artifact quarantine registry."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Quarantine manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Quarantine manifest schema_version must be 1")
    if payload.get("protocol_version") != "2.4":
        raise ValueError("Quarantine manifest must be bound to protocol 2.4")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Quarantine manifest must contain at least one artifact")
    for entry in artifacts:
        if not isinstance(entry, Mapping):
            raise ValueError("Every quarantined artifact must be a mapping")
        required = {
            "artifact_id",
            "original_path",
            "quarantine_path",
            "sha256",
            "size_bytes",
            "status",
            "blocked_uses",
            "reason",
        }
        missing = required.difference(entry)
        if missing:
            raise ValueError(f"Quarantine entry is missing fields: {sorted(missing)}")
        digest = entry["sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Quarantine entry has an invalid SHA-256 digest")
        if entry["status"] != "quarantined_pre_amendment":
            raise ValueError("Quarantine entry has an unsupported status")
        blocked_uses = entry["blocked_uses"]
        if not isinstance(blocked_uses, list) or not blocked_uses:
            raise ValueError("Quarantine entry must declare blocked uses")
    return payload


def verify_quarantine(path: str | Path) -> dict[str, Any]:
    """Verify retained binaries and ensure every old path is unavailable.

    Derived binaries are intentionally ignored by Git. A clean clone therefore
    passes when both the original and quarantined copy are absent; the manifest
    digest still blocks a matching artifact if it reappears later.
    """

    payload = load_quarantine_manifest(path)
    for entry in payload["artifacts"]:
        original = Path(entry["original_path"])
        quarantined = Path(entry["quarantine_path"])
        if original.exists():
            raise ValueError(f"Pre-amendment artifact remains accessible: {original}")
        if not quarantined.exists():
            continue
        if not quarantined.is_file():
            raise ValueError(f"Quarantine path is not a file: {quarantined}")
        if quarantined.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"Quarantined artifact size changed: {quarantined}")
        if sha256_file(quarantined) != entry["sha256"]:
            raise ValueError(f"Quarantined artifact SHA-256 changed: {quarantined}")
    return payload


def reject_quarantined_input(path: str | Path, manifest_path: str | Path) -> None:
    """Reject a path or digest registered as a pre-amendment artifact."""

    candidate = Path(path)
    payload = load_quarantine_manifest(manifest_path)
    candidate_digest = sha256_file(candidate) if candidate.is_file() else None
    for entry in payload["artifacts"]:
        registered_paths = {Path(entry["original_path"]), Path(entry["quarantine_path"])}
        if candidate in registered_paths or candidate_digest == entry["sha256"]:
            raise ValueError(
                f"Input {candidate} is quarantined under {entry['artifact_id']} "
                "and cannot be reused"
            )


__all__ = [
    "load_quarantine_manifest",
    "reject_quarantined_input",
    "sha256_file",
    "verify_quarantine",
]
