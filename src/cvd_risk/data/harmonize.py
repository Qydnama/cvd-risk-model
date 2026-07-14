"""Explicit, codebook-driven harmonization shared by cohort construction."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from cvd_risk.data.load_nhis import sha256_file
from cvd_risk.schemas import FAST_FEATURES, validate_fast_feature_schema

BASELINE_CVD_COLUMNS: tuple[str, ...] = (
    "coronary_heart_disease",
    "angina_pectoris",
    "myocardial_infarction",
    "stroke",
)

SENSITIVITY_BASELINE_CVD_COLUMNS: tuple[str, ...] = (
    *BASELINE_CVD_COLUMNS,
    "congestive_heart_failure",
)

DESIGN_COLUMNS: tuple[str, ...] = (
    "survey_year",
    "participant_id",
    "interview_quarter",
    "strata",
    "psu",
    "mortality_weight",
)

OUTCOME_COMPONENT_COLUMNS: tuple[str, ...] = (
    "mortality_eligible",
    "deceased",
    "ucod_leading",
    "death_year",
    "death_quarter",
)

DEFAULT_MAPPING_PATH = Path(__file__).resolve().parents[3] / "configs" / "feature_mapping.yaml"
VERIFIED_CODEBOOK_STATUS = "verified_against_user_extract"


class HarmonizationError(ValueError):
    """Raised when a source value cannot be safely mapped to the canonical schema."""


class CodebookVerificationError(HarmonizationError):
    """Raised when a mapping is not bound to the exact user extract/codebook."""


def load_feature_mapping(path: str | Path = DEFAULT_MAPPING_PATH) -> dict[str, Any]:
    """Load and structurally validate the explicit variable mapping."""

    mapping_path = Path(path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Feature mapping not found: {mapping_path}")
    with mapping_path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise HarmonizationError("feature_mapping.yaml must have schema_version: 1")
    variables = document.get("variables")
    if not isinstance(variables, dict) or not variables:
        raise HarmonizationError("feature_mapping.yaml must contain a non-empty variables map")

    required_canonical = set(
        (
            *DESIGN_COLUMNS,
            *FAST_FEATURES,
            *SENSITIVITY_BASELINE_CVD_COLUMNS,
            *OUTCOME_COMPONENT_COLUMNS,
        )
    )
    missing = sorted(required_canonical.difference(variables))
    if missing:
        raise HarmonizationError(f"Feature mapping misses canonical variables: {missing}")
    for canonical, specification in variables.items():
        if not isinstance(specification, dict):
            raise HarmonizationError(f"Mapping for {canonical!r} must be a dictionary")
        if not any(
            key in specification
            for key in ("source", "source_candidates", "derivation_sources")
        ):
            raise HarmonizationError(f"Mapping for {canonical!r} has no source")
        if "kind" not in specification or "role" not in specification:
            raise HarmonizationError(f"Mapping for {canonical!r} must declare kind and role")
    return document


def verify_codebook_binding(
    data_path: str | Path,
    codebook_path: str | Path,
    *,
    mapping_path: str | Path = DEFAULT_MAPPING_PATH,
) -> dict[str, str]:
    """Verify that mapping metadata names these exact immutable input files."""

    document = load_feature_mapping(mapping_path)
    verification = document.get("codebook_verification", {})
    if verification.get("status") != VERIFIED_CODEBOOK_STATUS:
        raise CodebookVerificationError(
            f"Mapping is pending codebook review; expected status {VERIFIED_CODEBOOK_STATUS!r}"
        )
    expected_extract = verification.get("extract_sha256")
    expected_codebook = verification.get("codebook_sha256")
    if not expected_extract or not expected_codebook:
        raise CodebookVerificationError("Verified mapping must contain both SHA-256 digests")

    actual_extract = sha256_file(data_path)
    actual_codebook = sha256_file(codebook_path)
    if actual_extract != expected_extract:
        raise CodebookVerificationError("NHIS extract SHA-256 does not match feature mapping")
    if actual_codebook != expected_codebook:
        raise CodebookVerificationError("IPUMS codebook SHA-256 does not match feature mapping")
    return {
        "codebook_verification_status": VERIFIED_CODEBOOK_STATUS,
        "extract_sha256": actual_extract,
        "codebook_sha256": actual_codebook,
    }


def _metadata_value(frame: pd.DataFrame, name: str) -> Any:
    value = frame.attrs.get(name)
    if value is not None:
        return value
    if name not in frame:
        return None
    unique = frame[name].dropna().astype(str).unique()
    if len(unique) > 1:
        raise CodebookVerificationError(f"Multiple {name} values occur in one frame")
    return unique[0] if len(unique) == 1 else None


def require_verified_codebook(frame: pd.DataFrame) -> None:
    """Block final cohort construction until exact codebook review is recorded."""

    status = _metadata_value(frame, "codebook_verification_status")
    extract_sha256 = _metadata_value(frame, "extract_sha256")
    codebook_sha256 = _metadata_value(frame, "codebook_sha256")
    feature_mapping_sha256 = _metadata_value(frame, "feature_mapping_sha256")
    if status != VERIFIED_CODEBOOK_STATUS:
        raise CodebookVerificationError(
            "Final NHIS cohort is blocked until feature_mapping.yaml is verified "
            "against the user's extract and codebook"
        )
    for label, digest in (
        ("extract_sha256", extract_sha256),
        ("codebook_sha256", codebook_sha256),
        ("feature_mapping_sha256", feature_mapping_sha256),
    ):
        if not isinstance(digest, str) or len(digest) != 64:
            raise CodebookVerificationError(f"Verified NHIS metadata has invalid {label}")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise CodebookVerificationError(
                f"Verified NHIS metadata has non-hexadecimal {label}"
            ) from exc


def _equivalent_keys(value: Any) -> tuple[Any, ...]:
    if pd.isna(value):
        return ()
    candidates: list[Any] = [value]
    if isinstance(value, str):
        stripped = value.strip()
        candidates.extend((stripped, stripped.casefold()))
        try:
            number = float(stripped)
        except ValueError:
            pass
        else:
            candidates.append(int(number) if number.is_integer() else number)
    elif isinstance(value, float) and value.is_integer():
        candidates.extend((int(value), str(int(value))))
    else:
        candidates.append(str(value))
    return tuple(dict.fromkeys(candidates))


def _mapping_lookup(value: Any, mapping: Mapping[Any, Any]) -> tuple[bool, Any]:
    string_mapping = {
        key.casefold(): mapped for key, mapped in mapping.items() if isinstance(key, str)
    }
    for candidate in _equivalent_keys(value):
        try:
            if candidate in mapping:
                return True, mapping[candidate]
        except TypeError:
            continue
        if isinstance(candidate, str) and candidate.casefold() in string_mapping:
            return True, string_mapping[candidate.casefold()]
    return False, None


def _missing_reason(value: Any, specification: Mapping[str, Any]) -> str | None:
    if pd.isna(value):
        return "true_missing"
    missing_codes = specification.get("missing_codes", {})
    if not isinstance(missing_codes, dict):
        raise HarmonizationError("missing_codes must map reasons to code lists")
    for reason, codes in missing_codes.items():
        if not isinstance(codes, list):
            raise HarmonizationError(f"Missing-code group {reason!r} must be a list")
        found, _ = _mapping_lookup(value, {code: True for code in codes})
        if found:
            return str(reason)
    return None


def _numeric_value(value: Any, canonical: str, specification: Mapping[str, Any]) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise HarmonizationError(f"{canonical}: expected numeric value, got {value!r}") from exc

    threshold = specification.get("apply_scale_when_abs_gte")
    scale = float(specification.get("scale", 1.0))
    if threshold is None or abs(numeric) >= float(threshold):
        numeric *= scale

    valid_range = specification.get("valid_range")
    if valid_range is not None:
        if not isinstance(valid_range, list) or len(valid_range) != 2:
            raise HarmonizationError(f"{canonical}: valid_range must be [min, max]")
        minimum, maximum = valid_range
        if minimum is not None and numeric < float(minimum):
            raise HarmonizationError(f"{canonical}: value {numeric} is below {minimum}")
        if maximum is not None and numeric > float(maximum):
            raise HarmonizationError(f"{canonical}: value {numeric} is above {maximum}")
    return numeric


def _convert_value(
    value: Any,
    *,
    canonical: str,
    specification: Mapping[str, Any],
) -> tuple[Any, str | None]:
    reason = _missing_reason(value, specification)
    if reason is not None:
        return pd.NA, reason

    kind = specification["kind"]
    if kind in {"category", "boolean"}:
        mapping = specification.get("mapping")
        if not isinstance(mapping, dict):
            raise HarmonizationError(f"{canonical}: categorical mapping is missing")
        found, mapped = _mapping_lookup(value, mapping)
        if not found:
            raise HarmonizationError(f"{canonical}: unmapped source code {value!r}")
        return mapped, None

    if kind in {"numeric", "integer"}:
        numeric = _numeric_value(value, canonical, specification)
        if kind == "integer":
            if not numeric.is_integer():
                raise HarmonizationError(f"{canonical}: expected integer, got {numeric}")
            return int(numeric), None
        return numeric, None

    if kind == "code":
        width = int(specification.get("width", 0))
        text = str(value).strip()
        try:
            number = float(text)
        except ValueError:
            normalized = text
        else:
            if not number.is_integer():
                raise HarmonizationError(f"{canonical}: non-integral code {value!r}")
            normalized = str(int(number))
        return normalized.zfill(width), None

    if kind == "identity":
        return value, None
    raise HarmonizationError(f"{canonical}: unsupported mapping kind {kind!r}")


def _resolve_source(
    raw: pd.DataFrame,
    canonical: str,
    specification: Mapping[str, Any],
) -> tuple[pd.Series, pd.Series]:
    if "source" in specification:
        source = str(specification["source"])
        if source not in raw:
            if specification.get("required", True):
                raise HarmonizationError(
                    f"{canonical}: required source column {source!r} is absent"
                )
            empty = pd.Series(pd.NA, index=raw.index, dtype="object")
            return empty, empty.copy()
        return raw[source].copy(), pd.Series(source, index=raw.index, dtype="string")

    candidates = [str(value) for value in specification.get("source_candidates", [])]
    available = [source for source in candidates if source in raw]
    if not available:
        if specification.get("required", True):
            raise HarmonizationError(
                f"{canonical}: none of the source candidates exist: {candidates}"
            )
        empty = pd.Series(pd.NA, index=raw.index, dtype="object")
        return empty, empty.copy()

    combined = pd.Series(pd.NA, index=raw.index, dtype="object")
    provenance = pd.Series(pd.NA, index=raw.index, dtype="string")
    for source in available:
        use = combined.isna() & raw[source].notna()
        combined.loc[use] = raw.loc[use, source]
        provenance.loc[use] = source
    return combined, provenance


def _cast_canonical(series: pd.Series, kind: str, canonical: str) -> pd.Series:
    if kind == "boolean":
        return series.astype("boolean")
    if kind == "integer":
        return series.astype("Int64")
    if kind == "numeric":
        return series.astype("Float64")
    if kind in {"category", "code"} or canonical in {"participant_id", "race_ethnicity"}:
        return series.astype("string")
    return series


_MEXICAN_HISPETH_CODES = frozenset({20, 21, 22, 23, 24})
_OTHER_HISPETH_CODES = frozenset({30, 40, 50, 60, 61, 62, 63, 64, 65, 66, 67, 70})
_MISSING_HISPETH_CODES = frozenset({90, 91, 92, 93, 99})
_WHITE_RACEA_CODES = frozenset({100})
_BLACK_RACEA_CODES = frozenset({200})
_OTHER_RACEA_CODES = frozenset(
    {
        300,
        310,
        320,
        330,
        340,
        350,
        400,
        410,
        411,
        412,
        413,
        414,
        415,
        416,
        420,
        421,
        422,
        423,
        430,
        431,
        432,
        433,
        434,
        500,
        510,
        520,
        530,
        540,
        550,
        560,
        570,
        580,
        600,
        610,
        611,
        612,
        613,
        614,
        615,
        616,
        617,
    }
)
_MISSING_RACEA_CODES = frozenset({900, 970, 980, 990})


def _integral_code(value: Any, *, source: str) -> int:
    if pd.isna(value):
        raise HarmonizationError(f"{source}: missing value")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise HarmonizationError(f"{source}: expected numeric code, got {value!r}") from exc
    if not numeric.is_integer():
        raise HarmonizationError(f"{source}: expected integer code, got {value!r}")
    return int(numeric)


def _derive_race_ethnicity(raw: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    missing_sources = sorted({"RACEA", "HISPETH"}.difference(raw.columns))
    if missing_sources:
        raise HarmonizationError(
            f"race_ethnicity: required derivation sources are absent: {missing_sources}"
        )
    raw_pairs = pd.Series(pd.NA, index=raw.index, dtype="string")
    values = pd.Series(pd.NA, index=raw.index, dtype="string")
    reasons = pd.Series(pd.NA, index=raw.index, dtype="string")
    for index in raw.index:
        race_value = raw.at[index, "RACEA"]
        hisp_value = raw.at[index, "HISPETH"]
        raw_pairs.at[index] = f"RACEA={race_value};HISPETH={hisp_value}"
        if pd.isna(hisp_value):
            reasons.at[index] = "hispeth_true_missing"
            continue
        hispeth = _integral_code(hisp_value, source="HISPETH")
        if hispeth in _MEXICAN_HISPETH_CODES:
            values.at[index] = "mexican_american"
            continue
        if hispeth in _OTHER_HISPETH_CODES:
            values.at[index] = "other_hispanic"
            continue
        if hispeth in _MISSING_HISPETH_CODES:
            reasons.at[index] = "hispeth_unknown_or_not_in_universe"
            continue
        if hispeth != 10:
            raise HarmonizationError(f"HISPETH: unmapped source code {hispeth!r}")
        if pd.isna(race_value):
            reasons.at[index] = "racea_true_missing"
            continue
        racea = _integral_code(race_value, source="RACEA")
        if racea in _WHITE_RACEA_CODES:
            values.at[index] = "non_hispanic_white"
        elif racea in _BLACK_RACEA_CODES:
            values.at[index] = "non_hispanic_black"
        elif racea in _OTHER_RACEA_CODES:
            values.at[index] = "other_or_multiracial"
        elif racea in _MISSING_RACEA_CODES:
            reasons.at[index] = "racea_unknown"
        else:
            raise HarmonizationError(f"RACEA: unmapped source code {racea!r}")
    return raw_pairs, values, reasons


def harmonize_nhis(
    raw: pd.DataFrame,
    *,
    mapping_path: str | Path = DEFAULT_MAPPING_PATH,
    strict: bool = True,
) -> pd.DataFrame:
    """Map an IPUMS NHIS extract to canonical values with full provenance.

    Every canonical variable produces ``<name>_raw``, ``<name>``, and
    ``<name>_missing_reason``. Unknown non-missing codes raise in strict mode;
    they are never silently converted to ``No`` or another reference category.
    """

    if raw.empty:
        raise HarmonizationError("Cannot harmonize an empty NHIS frame")
    document = load_feature_mapping(mapping_path)
    verification = document.get("codebook_verification", {})
    verification_status = verification.get("status", "unspecified")
    extract_sha256 = raw.attrs.get("extract_sha256")
    codebook_sha256 = raw.attrs.get("codebook_sha256")
    if verification_status == VERIFIED_CODEBOOK_STATUS:
        expected_extract = verification.get("extract_sha256")
        expected_codebook = verification.get("codebook_sha256")
        if not expected_extract or not expected_codebook:
            raise CodebookVerificationError(
                "Verified mapping must record extract_sha256 and codebook_sha256"
            )
        if extract_sha256 != expected_extract or codebook_sha256 != expected_codebook:
            raise CodebookVerificationError(
                "Loaded NHIS extract/codebook hashes do not match the verified mapping"
            )
    output = pd.DataFrame(index=raw.index)

    for canonical, specification in document["variables"].items():
        if canonical == "race_ethnicity":
            raw_pairs, values, reasons = _derive_race_ethnicity(raw)
            output["race_ethnicity_raw"] = raw_pairs
            output["race_ethnicity_source"] = "RACEA+HISPETH"
            output["race_ethnicity"] = values
            output["race_ethnicity_missing_reason"] = reasons
            continue
        source_values, provenance = _resolve_source(raw, canonical, specification)
        output[f"{canonical}_raw"] = source_values
        output[f"{canonical}_source"] = provenance

        canonical_values: list[Any] = []
        missing_reasons: list[str | None] = []
        source_unavailable = not provenance.notna().any()
        for index, value in source_values.items():
            if pd.isna(value) and source_unavailable:
                canonical_values.append(pd.NA)
                missing_reasons.append("source_unavailable")
                continue
            try:
                mapped, reason = _convert_value(
                    value,
                    canonical=canonical,
                    specification=specification,
                )
            except HarmonizationError as exc:
                if strict:
                    raise HarmonizationError(f"{exc} at row index {index!r}") from exc
                mapped, reason = pd.NA, "unmapped_or_invalid"
            canonical_values.append(mapped)
            missing_reasons.append(reason)

        converted = pd.Series(canonical_values, index=raw.index)
        output[canonical] = _cast_canonical(converted, specification["kind"], canonical)
        output[f"{canonical}_missing_reason"] = pd.Series(
            missing_reasons, index=raw.index, dtype="string"
        )

    output["codebook_verification_status"] = verification_status
    output["extract_sha256"] = extract_sha256 if extract_sha256 is not None else pd.NA
    output["codebook_sha256"] = codebook_sha256 if codebook_sha256 is not None else pd.NA
    mapping_sha256 = sha256_file(mapping_path)
    output["feature_mapping_sha256"] = mapping_sha256
    validate_harmonized_nhis(output)
    output = output.reset_index(drop=True)
    output.attrs.update(raw.attrs)
    output.attrs["feature_mapping_path"] = str(Path(mapping_path).resolve())
    output.attrs["codebook_verification_status"] = verification_status
    output.attrs["extract_sha256"] = extract_sha256
    output.attrs["codebook_sha256"] = codebook_sha256
    output.attrs["feature_mapping_sha256"] = mapping_sha256
    return output


def validate_harmonized_nhis(frame: pd.DataFrame) -> None:
    """Validate the canonical contract required by the cohort builder."""

    required = set(
        (
            *DESIGN_COLUMNS,
            *FAST_FEATURES,
            *SENSITIVITY_BASELINE_CVD_COLUMNS,
            *OUTCOME_COMPONENT_COLUMNS,
        )
    )
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise HarmonizationError(f"Harmonized NHIS frame misses columns: {missing}")
    if frame["participant_id"].isna().any():
        raise HarmonizationError("Harmonized NHIS has missing participant identifiers")
    if frame["participant_id"].duplicated().any():
        raise HarmonizationError("Harmonized NHIS has duplicated participant identifiers")
    validate_fast_feature_schema(frame, allow_missing=True)


def mapping_roles(path: str | Path = DEFAULT_MAPPING_PATH) -> dict[str, str]:
    """Return canonical roles, useful for enforcing audit-only variables."""

    document = load_feature_mapping(path)
    return {
        canonical: str(specification["role"])
        for canonical, specification in document["variables"].items()
    }
