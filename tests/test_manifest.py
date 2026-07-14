from __future__ import annotations

from pathlib import Path

import yaml

from cvd_risk.data.manifest import register_local_input


def test_register_local_input_is_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "extract.csv.gz"
    source.write_bytes(b"immutable input")
    manifest = tmp_path / "manifest.yaml"
    first = register_local_input(
        manifest,
        source,
        dataset="nhis",
        component="ipums_extract",
        period="1997-2014",
        source_url="https://nhis.ipums.org/nhis/",
    )
    second = register_local_input(
        manifest,
        source,
        dataset="nhis",
        component="ipums_extract",
        period="1997-2014",
        source_url="https://nhis.ipums.org/nhis/",
    )
    document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert len(document["files"]) == 1
    assert first["sha256"] == second["sha256"]
