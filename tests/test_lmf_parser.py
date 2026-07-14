from __future__ import annotations

import pandas as pd
import pytest

from cvd_risk.data.read_lmf import (
    LMFValidationError,
    merge_demo_lmf,
    read_lmf,
)

# Literal rows checked against the official inclusive positions:
# SEQN 1-6, ELIGSTAT 15, MORTSTAT 16, UCOD 17-19, DIABETES 20,
# HYPERTEN 21, PERMTH_INT 43-45, and PERMTH_EXM 46-48.
DECEASED_LINE = "000123        1100110                     060058"
ALIVE_LINE = "000124        10                          120119"
INELIGIBLE_LINE = "000125        3                                 "
NCHS_TRIMMED_INELIGIBLE_LINE = "00001         2.   ..                     .  ."


def _write_lmf(tmp_path, lines):
    assert all(15 <= len(line) <= 48 for line in lines)
    path = tmp_path / "mortality.dat"
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def test_read_lmf_uses_exact_fixed_width_positions(tmp_path):
    frame = read_lmf(_write_lmf(tmp_path, [DECEASED_LINE, ALIVE_LINE, INELIGIBLE_LINE]))

    assert frame["SEQN"].tolist() == [123, 124, 125]
    assert frame.loc[0, "ELIGSTAT"] == 1
    assert frame.loc[0, "MORTSTAT"] == 1
    assert frame.loc[0, "UCOD_LEADING"] == "001"
    assert frame.loc[0, "DIABETES"] == 1
    assert frame.loc[0, "HYPERTEN"] == 0
    assert frame.loc[0, "PERMTH_INT"] == 60
    assert frame.loc[0, "PERMTH_EXM"] == 58
    assert frame.loc[1, "MORTSTAT"] == 0
    assert pd.isna(frame.loc[1, "UCOD_LEADING"])
    assert pd.isna(frame.loc[2, "MORTSTAT"])


def test_read_lmf_accepts_official_dot_missing_values_and_trimmed_trailing_blanks(tmp_path):
    frame = read_lmf(_write_lmf(tmp_path, [NCHS_TRIMMED_INELIGIBLE_LINE]))

    assert frame.loc[0, "SEQN"] == 1
    assert frame.loc[0, "ELIGSTAT"] == 2
    assert pd.isna(frame.loc[0, "MORTSTAT"])
    assert pd.isna(frame.loc[0, "PERMTH_EXM"])


def test_read_lmf_rejects_duplicate_seqn(tmp_path):
    path = _write_lmf(tmp_path, [DECEASED_LINE, DECEASED_LINE])

    with pytest.raises(LMFValidationError, match="duplicate SEQN"):
        read_lmf(path)


def test_read_lmf_rejects_invalid_eligibility_code(tmp_path):
    invalid = DECEASED_LINE[:14] + "9" + DECEASED_LINE[15:]
    path = _write_lmf(tmp_path, [invalid])

    with pytest.raises(LMFValidationError, match="invalid ELIGSTAT"):
        read_lmf(path)


def test_merge_demo_lmf_is_one_to_one(tmp_path):
    mortality = read_lmf(_write_lmf(tmp_path, [DECEASED_LINE, ALIVE_LINE]))
    demo = pd.DataFrame({"SEQN": [123.0, 124.0], "RIDAGEYR": [70, 50]})

    merged = merge_demo_lmf(demo, mortality)

    assert len(merged) == 2
    assert merged["SEQN"].is_unique
    assert merged["MORTSTAT"].tolist() == [1, 0]


def test_merge_demo_lmf_rejects_participant_mismatch(tmp_path):
    mortality = read_lmf(_write_lmf(tmp_path, [DECEASED_LINE, ALIVE_LINE]))
    demo = pd.DataFrame({"SEQN": [123, 999]})

    with pytest.raises(LMFValidationError, match="participant mismatch"):
        merge_demo_lmf(demo, mortality)
