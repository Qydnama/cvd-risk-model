from __future__ import annotations

import json
from pathlib import Path

import pytest

from cvd_risk.modeling.lock_amendment import (
    AMENDMENT_PATH,
    NHANES_EXTERNAL_CYCLES,
    NHIS_TEMPORAL_YEARS,
    ORIGINAL_LOCK_PATH,
    portable_binding,
    verify_amendment_content,
    verify_portable_bindings,
)


def test_validation_cohorts_are_explicit_enumerations() -> None:
    assert NHIS_TEMPORAL_YEARS == [2010, 2011, 2012, 2013, 2014]
    assert NHANES_EXTERNAL_CYCLES == [
        "1999-2000",
        "2001-2002",
        "2003-2004",
        "2005-2006",
        "2007-2008",
        "2009-2010",
        "2011-2012",
        "2013-2014",
    ]


def test_portable_binding_resolves_from_repository_root(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "artifact.json"
    target.parent.mkdir()
    target.write_text(json.dumps({"frozen": True}), encoding="utf-8")

    binding = portable_binding("nested/artifact.json", root=tmp_path)
    target.rename(tmp_path / "nested" / "moved.json")
    (tmp_path / "nested" / "moved.json").rename(target)

    verify_portable_bindings({"artifact": binding}, root=tmp_path)
    assert binding["repository_relative_path"] == "nested/artifact.json"


def test_portable_binding_rejects_absolute_and_parent_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repository-relative"):
        portable_binding(tmp_path / "artifact.json", root=tmp_path)
    with pytest.raises(ValueError, match="repository-relative"):
        portable_binding("../artifact.json", root=tmp_path)


def test_generated_amendment_verifies_without_absolute_windows_paths() -> None:
    root = Path.cwd().resolve()
    amendment = json.loads((root / AMENDMENT_PATH).read_text(encoding="utf-8"))
    lock = json.loads((root / ORIGINAL_LOCK_PATH).read_text(encoding="utf-8"))

    verify_amendment_content(amendment, lock=lock, root=root)

    assert amendment["portable_binding_policy"]["absolute_paths_required"] is False
    assert set(amendment["audit_manifest_bindings"]) == {
        "access_provenance",
        "quarantine",
        "sealed_audit",
    }
    assert all(
        not Path(entry["repository_relative_path"]).is_absolute()
        for entry in amendment["portable_locked_bindings"].values()
    )
