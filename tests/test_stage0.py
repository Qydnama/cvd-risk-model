from __future__ import annotations

import json
from pathlib import Path

from cvd_risk.stage0 import run_stage0_check


def test_stage0_reports_missing_rscript_as_a_hard_block(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("cvd_risk.stage0._find_rscript", lambda: None)
    output = tmp_path / "stage0.json"

    result = run_stage0_check(output_path=output)

    assert result["status"] == "BLOCKED_RSCRIPT_NOT_FOUND"
    assert result["ready_for_stage1"] is False
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["checks"]["protocol_lock"]["status"] == "PASS"
    assert written["checks"]["quarantine"]["status"] == "PASS"


def test_r_reference_scaffold_is_present() -> None:
    required = [
        Path("scripts/r/check_environment.R"),
        Path("scripts/r/survey_reference.R"),
        Path("scripts/r/pmsampsize_reference.R"),
        Path("scripts/r/pmvalsampsize_reference.R"),
        Path("scripts/r/run_reference_tests.R"),
        Path("scripts/r/survey_design_gate.R"),
        Path("tests/fixtures/r/survey_binary.csv"),
        Path("tests/fixtures/r/survey_binary_expected.csv"),
        Path("tests/fixtures/r/sample_size_expected.csv"),
    ]
    assert all(path.is_file() for path in required)
