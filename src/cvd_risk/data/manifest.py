"""Manifest helpers for manually supplied, immutable research inputs."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from cvd_risk.data.load_nhis import sha256_file


def register_local_input(
    manifest_path: str | Path,
    input_path: str | Path,
    *,
    dataset: str,
    component: str,
    period: str,
    source_url: str,
) -> dict[str, Any]:
    """Atomically add or refresh a manual file's size and SHA-256 binding."""

    path = Path(input_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Manifest input does not exist: {path}")
    destination = Path(manifest_path)
    if destination.exists():
        loaded = yaml.safe_load(destination.read_text(encoding="utf-8")) or {}
    else:
        loaded = {}
    if not isinstance(loaded, dict) or loaded.get("schema_version", 1) != 1:
        raise ValueError(f"Unsupported manifest schema: {destination}")
    files = loaded.setdefault("files", [])
    if not isinstance(files, list):
        raise ValueError(f"Manifest files must be a list: {destination}")
    try:
        stored_path = path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        stored_path = str(path)
    entry = {
        "source": "IPUMS NHIS",
        "dataset": dataset,
        "period": period,
        "component": component,
        "url": source_url,
        "local_path": stored_path,
        "registered_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "status": "manual_input",
    }
    matching = next(
        (
            item
            for item in files
            if isinstance(item, dict)
            and item.get("dataset") == dataset
            and item.get("component") == component
            and item.get("period") == period
        ),
        None,
    )
    if matching is None:
        files.append(entry)
    else:
        matching.clear()
        matching.update(entry)
    loaded["schema_version"] = 1
    loaded["generated_at"] = entry["registered_at"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        yaml.safe_dump(loaded, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return entry
