from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cvd_risk.data.load_nhanes import (
    NHANESLoadError,
    cycle_file_specs,
    load_nhanes_cycle,
    merge_component,
)


def _mortality_frame():
    return pd.DataFrame(
        {
            "SEQN": pd.Series([1, 2], dtype="Int64"),
            "ELIGSTAT": pd.Series([1, 1], dtype="Int64"),
            "MORTSTAT": pd.Series([0, 1], dtype="Int64"),
            "UCOD_LEADING": pd.Series([pd.NA, "001"], dtype="string"),
            "DIABETES": pd.Series([pd.NA, 0], dtype="Int64"),
            "HYPERTEN": pd.Series([pd.NA, 1], dtype="Int64"),
            "PERMTH_INT": pd.Series([120, 50], dtype="Int64"),
            "PERMTH_EXM": pd.Series([119, 49], dtype="Int64"),
        }
    )


def test_profile_inventory_includes_fast_files_for_extended_model():
    fast = cycle_file_specs("1999-2000", profile="fast")
    extended = cycle_file_specs("1999-2000", profile="extended")

    assert len(fast) == 8
    assert len(extended) == 11
    assert {spec.component for spec in fast} >= {"DEMO", "MCQ", "LMF"}
    assert {spec.component for spec in extended} >= {"BPX", "TCHOL_HDL", "GHB"}


def test_merge_component_enforces_one_to_one_and_preserves_demo_rows():
    demo = pd.DataFrame({"SEQN": [1, 2, 3], "RIDAGEYR": [40, 50, 60]})
    bmx = pd.DataFrame({"SEQN": [1, 3], "BMXBMI": [22.0, 31.0]})

    merged = merge_component(demo, bmx, component_name="BMX")

    assert len(merged) == 3
    assert merged["SEQN"].is_unique
    assert pd.isna(merged.loc[1, "BMXBMI"])


def test_merge_component_rejects_duplicate_participants():
    demo = pd.DataFrame({"SEQN": [1, 2]})
    duplicate = pd.DataFrame({"SEQN": [1, 1], "BMXBMI": [20.0, 21.0]})

    with pytest.raises(NHANESLoadError, match="duplicate SEQN"):
        merge_component(demo, duplicate, component_name="BMX")


def test_load_cycle_assembles_fast_components_and_lmf(tmp_path, monkeypatch):
    cycle = "1999-2000"
    cycle_dir = tmp_path / cycle
    cycle_dir.mkdir()
    specs = cycle_file_specs(cycle, profile="fast")
    for spec in specs:
        (cycle_dir / spec.filename).touch()

    def fake_read_xpt(path):
        path = Path(path)
        if path.name == "DEMO.xpt":
            return pd.DataFrame({"SEQN": [1, 2], "RIDAGEYR": [40, 70]})
        variable = f"RAW_{path.stem}"
        return pd.DataFrame({"SEQN": [1, 2], variable: [10, 20]})

    monkeypatch.setattr("cvd_risk.data.load_nhanes.read_xpt", fake_read_xpt)
    monkeypatch.setattr("cvd_risk.data.load_nhanes.read_lmf", lambda _: _mortality_frame())

    frame = load_nhanes_cycle(cycle, raw_dir=tmp_path, profile="fast")

    assert len(frame) == 2
    assert frame[["CYCLE", "SEQN"]].duplicated().sum() == 0
    assert frame["MORTSTAT"].tolist() == [0, 1]


def test_load_cycle_reports_every_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="DEMO.xpt"):
        load_nhanes_cycle("1999-2000", raw_dir=tmp_path, profile="fast")
