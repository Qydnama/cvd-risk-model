"""Parser and merge checks for the 2019 public-use NHANES mortality files."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

# Official R program uses inclusive 1-based positions.  pandas uses zero-based,
# half-open colspecs, hence (0, 6), (14, 15), ... below.
LMF_NAMES = (
    "SEQN",
    "ELIGSTAT",
    "MORTSTAT",
    "UCOD_LEADING",
    "DIABETES",
    "HYPERTEN",
    "PERMTH_INT",
    "PERMTH_EXM",
)
LMF_COLSPECS = (
    (0, 6),
    (14, 15),
    (15, 16),
    (16, 19),
    (19, 20),
    (20, 21),
    (42, 45),
    (45, 48),
)
INTEGER_COLUMNS = (
    "SEQN",
    "ELIGSTAT",
    "MORTSTAT",
    "DIABETES",
    "HYPERTEN",
    "PERMTH_INT",
    "PERMTH_EXM",
)
VALID_UCOD_LEADING = {f"{value:03d}" for value in range(1, 11)}


class LMFValidationError(ValueError):
    """Raised when a mortality file or merge violates the NCHS contract."""


def normalize_seqn(values: pd.Series, *, context: str = "data") -> pd.Series:
    """Return nullable integer SEQN, rejecting missing and non-integer IDs."""

    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        raise LMFValidationError(f"{context}: SEQN contains missing or non-numeric values")
    fractional = numeric.mod(1).ne(0)
    if fractional.any():
        raise LMFValidationError(f"{context}: SEQN contains non-integer values")
    normalized = numeric.astype("Int64")
    if normalized.le(0).any():
        raise LMFValidationError(f"{context}: SEQN must be positive")
    return normalized


def _invalid_values(series: pd.Series, allowed: Iterable[int]) -> set[int]:
    allowed_set = set(allowed)
    return {int(value) for value in series.dropna().unique() if int(value) not in allowed_set}


def validate_lmf(frame: pd.DataFrame, *, context: str = "LMF") -> None:
    missing_columns = set(LMF_NAMES).difference(frame.columns)
    if missing_columns:
        raise LMFValidationError(f"{context}: missing columns {sorted(missing_columns)}")
    if frame.empty:
        raise LMFValidationError(f"{context}: file contains no records")
    if frame["SEQN"].isna().any():
        raise LMFValidationError(f"{context}: SEQN contains missing values")
    duplicates = frame.loc[frame["SEQN"].duplicated(keep=False), "SEQN"].unique().tolist()
    if duplicates:
        raise LMFValidationError(f"{context}: duplicate SEQN values: {duplicates[:10]}")

    allowed_codes = {
        "ELIGSTAT": {1, 2, 3},
        "MORTSTAT": {0, 1},
        "DIABETES": {0, 1},
        "HYPERTEN": {0, 1},
    }
    for column, allowed in allowed_codes.items():
        invalid = _invalid_values(frame[column], allowed)
        if invalid:
            raise LMFValidationError(f"{context}: invalid {column} values {sorted(invalid)}")
    if frame["ELIGSTAT"].isna().any():
        raise LMFValidationError(f"{context}: ELIGSTAT contains missing values")

    ucod = frame["UCOD_LEADING"].dropna()
    invalid_ucod = set(ucod).difference(VALID_UCOD_LEADING)
    if invalid_ucod:
        raise LMFValidationError(f"{context}: invalid UCOD_LEADING values {sorted(invalid_ucod)}")
    for column in ("PERMTH_INT", "PERMTH_EXM"):
        invalid_time = frame[column].dropna().loc[lambda values: (values < 0) | (values > 374)]
        if not invalid_time.empty:
            raise LMFValidationError(
                f"{context}: {column} is outside the official 0-374 month range"
            )


def read_lmf(path: str | Path, *, validate: bool = True) -> pd.DataFrame:
    """Read an NCHS 2019 public-use fixed-width NHANES LMF.

    ``UCOD_LEADING`` deliberately remains a three-character string so leading
    zeros in codes such as ``001`` and ``005`` cannot be lost.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"NHANES linked mortality file not found: {path}")
    frame = pd.read_fwf(
        path,
        colspecs=LMF_COLSPECS,
        names=LMF_NAMES,
        header=None,
        dtype="string",
        keep_default_na=False,
        encoding="ascii",
    )
    for column in frame.columns:
        stripped = frame[column].str.strip()
        frame[column] = stripped.mask(stripped.isin(["", "."]), pd.NA)
    for column in INTEGER_COLUMNS:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype("Int64")
        except (TypeError, ValueError) as exc:
            raise LMFValidationError(f"{path}: {column} contains non-integer text") from exc
    frame["UCOD_LEADING"] = frame["UCOD_LEADING"].astype("string")
    if validate:
        validate_lmf(frame, context=str(path))
    return frame


def merge_demo_lmf(
    demo: pd.DataFrame,
    mortality: pd.DataFrame,
    *,
    require_complete_match: bool = True,
) -> pd.DataFrame:
    """Merge DEMO and LMF with explicit one-to-one and participant checks."""

    if "SEQN" not in demo.columns:
        raise LMFValidationError("DEMO: missing SEQN")
    validate_lmf(mortality)
    left = demo.copy()
    right = mortality.copy()
    left["SEQN"] = normalize_seqn(left["SEQN"], context="DEMO")
    right["SEQN"] = normalize_seqn(right["SEQN"], context="LMF")
    duplicates = left.loc[left["SEQN"].duplicated(keep=False), "SEQN"].unique().tolist()
    if duplicates:
        raise LMFValidationError(f"DEMO: duplicate SEQN values: {duplicates[:10]}")
    overlap = (set(left.columns) & set(right.columns)) - {"SEQN"}
    if overlap:
        raise LMFValidationError(f"DEMO already contains mortality columns: {sorted(overlap)}")

    if require_complete_match:
        demo_ids = set(left["SEQN"].tolist())
        lmf_ids = set(right["SEQN"].tolist())
        missing_lmf = demo_ids - lmf_ids
        extra_lmf = lmf_ids - demo_ids
        if missing_lmf or extra_lmf:
            raise LMFValidationError(
                "DEMO/LMF participant mismatch: "
                f"{len(missing_lmf)} missing from LMF, {len(extra_lmf)} missing from DEMO"
            )
    merged = left.merge(right, on="SEQN", how="left", validate="one_to_one", sort=False)
    if len(merged) != len(left) or merged["SEQN"].duplicated().any():
        raise LMFValidationError("DEMO/LMF merge duplicated participants")
    return merged
