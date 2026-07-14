"""Discovery, loading, and fail-fast validation for an IPUMS NHIS extract."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_NHIS_COLUMNS: tuple[str, ...] = (
    "YEAR",
    "NHISPID",
    "QUARTER",
    "STRATA",
    "PSU",
    "MORTWTSA",
    "AGE",
    "SEX",
    "BMI",
    "SMOKESTATUS2",
    "HYPERTENEV",
    "DIABETICEV",
    "HEALTH",
    "CHEARTDIEV",
    "ANGIPECEV",
    "HEARTATTEV",
    "STROKEV",
    "CONHFAILEV",
    "MORTELIG",
    "MORTSTAT",
    "MORTUCODLD",
    "MORTDODY",
    "MORTDODQ",
)

_DATA_SUFFIXES = (".csv", ".csv.gz", ".dta")
_CODEBOOK_SUFFIXES = {
    ".csv",
    ".doc",
    ".docx",
    ".htm",
    ".html",
    ".json",
    ".pdf",
    ".rtf",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
_CODEBOOK_NAME_MARKERS = ("codebook", "ddi", "dictionary", "metadata")


class NHISInputError(ValueError):
    """Raised when the manual IPUMS input does not satisfy the data contract."""


@dataclass(frozen=True, slots=True)
class NHISInputFiles:
    """The unambiguous data/codebook pair for one IPUMS extract."""

    data_path: Path
    codebook_path: Path


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for a raw input or codebook."""

    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _is_supported_data_file(path: Path) -> bool:
    name = path.name.lower()
    looks_like_documentation = any(marker in name for marker in _CODEBOOK_NAME_MARKERS)
    return (
        path.is_file()
        and not looks_like_documentation
        and any(name.endswith(suffix) for suffix in _DATA_SUFFIXES)
    )


def _documentation_candidates(directory: Path, data_path: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in directory.iterdir():
        if not path.is_file() or path == data_path or path.name.startswith("."):
            continue
        name = path.name.lower()
        suffix = path.suffix.lower()
        if suffix not in _CODEBOOK_SUFFIXES:
            continue
        if any(marker in name for marker in _CODEBOOK_NAME_MARKERS) or suffix == ".xml":
            candidates.append(path)

    if candidates:
        return sorted(candidates)

    # Some IPUMS deliveries use an extract identifier for both files and do not
    # include the word "codebook". Accept a sole documentation file, but never
    # guess when more than one candidate exists.
    fallback = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path != data_path
        and not path.name.startswith(".")
        and path.suffix.lower() in _CODEBOOK_SUFFIXES.difference({".csv"})
    ]
    return sorted(fallback)


def discover_nhis_input(raw_dir: str | Path) -> NHISInputFiles:
    """Discover exactly one supported extract and its accompanying codebook.

    Ambiguity is an error by design: selecting the newest or first file could
    pair data with the wrong labels and silently reverse categorical codes.
    """

    directory = Path(raw_dir)
    if not directory.exists():
        raise FileNotFoundError(f"NHIS raw directory does not exist: {directory}")
    if not directory.is_dir():
        raise NHISInputError(f"Expected an NHIS directory, got: {directory}")

    data_candidates = sorted(path for path in directory.iterdir() if _is_supported_data_file(path))
    if not data_candidates:
        raise FileNotFoundError(
            f"No NHIS extract found in {directory}; expected .csv.gz, .csv, or .dta"
        )
    if len(data_candidates) != 1:
        names = [path.name for path in data_candidates]
        raise NHISInputError(f"Ambiguous NHIS extracts in {directory}: {names}")

    data_path = data_candidates[0]
    codebooks = _documentation_candidates(directory, data_path)
    if not codebooks:
        raise FileNotFoundError(
            f"No IPUMS codebook found next to {data_path.name}; keep the extract codebook"
        )
    if len(codebooks) != 1:
        names = [path.name for path in codebooks]
        raise NHISInputError(f"Ambiguous IPUMS codebooks in {directory}: {names}")
    return NHISInputFiles(data_path=data_path, codebook_path=codebooks[0])


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    converted = pd.to_numeric(frame[column], errors="coerce")
    invalid = frame[column].notna() & converted.isna()
    if invalid.any():
        examples = frame.loc[invalid, column].astype(str).head(5).tolist()
        raise NHISInputError(f"{column} contains non-numeric values: {examples}")
    return converted


def validate_nhis_schema(
    frame: pd.DataFrame,
    *,
    allowed_years: Iterable[int] = range(1997, 2015),
    require_all_years: bool = False,
) -> None:
    """Validate names and structural invariants without recoding any values."""

    if frame.empty:
        raise NHISInputError("NHIS extract is empty")
    duplicated_columns = frame.columns[frame.columns.duplicated()].tolist()
    if duplicated_columns:
        raise NHISInputError(f"Duplicate NHIS columns: {duplicated_columns}")

    missing = sorted(set(REQUIRED_NHIS_COLUMNS).difference(frame.columns))
    if missing:
        raise NHISInputError(f"NHIS extract is missing required columns: {missing}")

    year = _numeric_series(frame, "YEAR")
    quarter = _numeric_series(frame, "QUARTER")
    if year.isna().any() or (year % 1 != 0).any():
        raise NHISInputError("YEAR must contain non-missing integer survey years")
    if quarter.isna().any() or (quarter % 1 != 0).any():
        raise NHISInputError("QUARTER must contain non-missing integers")

    allowed = {int(value) for value in allowed_years}
    observed = {int(value) for value in year.unique()}
    unexpected = sorted(observed.difference(allowed))
    if unexpected:
        raise NHISInputError(f"NHIS extract contains years outside 1997-2014: {unexpected}")
    if require_all_years:
        absent = sorted(allowed.difference(observed))
        if absent:
            raise NHISInputError(f"NHIS extract is missing requested years: {absent}")
    bad_quarters = sorted({int(value) for value in quarter.unique()}.difference({1, 2, 3, 4}))
    if bad_quarters:
        raise NHISInputError(f"QUARTER contains values outside 1-4: {bad_quarters}")

    identifiers = frame["NHISPID"]
    if identifiers.isna().any() or identifiers.astype(str).str.strip().eq("").any():
        raise NHISInputError("NHISPID contains missing or blank identifiers")
    duplicate_count = int(identifiers.duplicated().sum())
    if duplicate_count:
        raise NHISInputError(f"NHISPID is not unique ({duplicate_count} duplicate rows)")

    for column in ("STRATA", "PSU"):
        if frame[column].isna().any():
            raise NHISInputError(f"{column} contains missing survey-design values")
    weights = _numeric_series(frame, "MORTWTSA")
    if weights.isna().any() or (weights < 0).any():
        raise NHISInputError("MORTWTSA must be numeric, non-missing, and non-negative")


def read_nhis_data(data_path: str | Path) -> pd.DataFrame:
    """Read one supported IPUMS extract without applying category assumptions."""

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"NHIS extract not found: {path}")
    name = path.name.lower()
    if name.endswith((".csv", ".csv.gz")):
        return pd.read_csv(path, low_memory=False)
    if name.endswith(".dta"):
        # Raw numeric codes are intentionally retained. The mapping is applied
        # explicitly from feature_mapping.yaml and checked for unknown values.
        return pd.read_stata(path, convert_categoricals=False)
    raise NHISInputError(f"Unsupported NHIS extract format: {path}")


def load_nhis(
    path: str | Path,
    *,
    codebook_path: str | Path | None = None,
    require_all_years: bool = False,
) -> pd.DataFrame:
    """Load an NHIS extract from a directory or explicit file and validate it."""

    input_path = Path(path)
    if input_path.is_dir():
        discovered = discover_nhis_input(input_path)
        data_path = discovered.data_path
        resolved_codebook = discovered.codebook_path
        if codebook_path is not None and Path(codebook_path) != resolved_codebook:
            raise NHISInputError("Do not combine an explicit codebook with another directory pair")
    else:
        data_path = input_path
        if codebook_path is None:
            candidates = _documentation_candidates(data_path.parent, data_path)
            if len(candidates) != 1:
                raise NHISInputError(
                    "An explicit data file requires one unambiguous adjacent codebook "
                    "or codebook_path"
                )
            resolved_codebook = candidates[0]
        else:
            resolved_codebook = Path(codebook_path)

    if not resolved_codebook.exists() or not resolved_codebook.is_file():
        raise FileNotFoundError(f"IPUMS codebook not found: {resolved_codebook}")

    frame = read_nhis_data(data_path)
    validate_nhis_schema(frame, require_all_years=require_all_years)
    frame.attrs["source"] = "IPUMS NHIS"
    frame.attrs["data_path"] = str(data_path.resolve())
    frame.attrs["codebook_path"] = str(resolved_codebook.resolve())
    frame.attrs["extract_sha256"] = sha256_file(data_path)
    frame.attrs["codebook_sha256"] = sha256_file(resolved_codebook)
    return frame
