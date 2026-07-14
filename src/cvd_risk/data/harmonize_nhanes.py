"""Cycle-aware, fail-fast harmonization of continuous NHANES 1999-2014."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from cvd_risk.data.harmonize import FAST_FEATURES, SENSITIVITY_BASELINE_CVD_COLUMNS
from cvd_risk.schemas import CanonicalSchemaError, validate_fast_feature_schema

NHANES_CYCLE_START_YEAR: dict[str, int] = {
    "1999-2000": 1999,
    "2001-2002": 2001,
    "2003-2004": 2003,
    "2005-2006": 2005,
    "2007-2008": 2007,
    "2009-2010": 2009,
    "2011-2012": 2011,
    "2013-2014": 2013,
}

GENERAL_HEALTH_SOURCE: dict[str, str] = {
    cycle: "HUQ010" for cycle in NHANES_CYCLE_START_YEAR
}

# Official mg/dL names differ from the simplified conceptual component names.
HDL_SOURCE: dict[str, str] = {
    "1999-2000": "LBDHDL",
    "2001-2002": "LBDHDL",
    "2003-2004": "LBXHDD",
    "2005-2006": "LBDHDD",
    "2007-2008": "LBDHDD",
    "2009-2010": "LBDHDD",
    "2011-2012": "LBDHDD",
    "2013-2014": "LBDHDD",
}

BASELINE_CVD_SOURCE: dict[str, str] = {
    "congestive_heart_failure": "MCQ160B",
    "coronary_heart_disease": "MCQ160C",
    "angina_pectoris": "MCQ160D",
    "myocardial_infarction": "MCQ160E",
    "stroke": "MCQ160F",
}

EXTENDED_FEATURES: tuple[str, ...] = (
    "systolic_bp",
    "total_cholesterol",
    "hdl_cholesterol",
    "hba1c",
)
SBP_SOURCES: tuple[str, ...] = ("BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4")
SAS_XPORT_MACHINE_ZERO_THRESHOLD = 1e-70

_AGE_MAX_BY_CYCLE = {
    cycle: (85 if start_year <= 2005 else 80)
    for cycle, start_year in NHANES_CYCLE_START_YEAR.items()
}
_STRATA_RANGE_BY_CYCLE = {
    "1999-2000": (1, 13),
    "2001-2002": (14, 28),
    "2003-2004": (29, 43),
    "2005-2006": (44, 58),
    "2007-2008": (59, 74),
    "2009-2010": (75, 89),
    "2011-2012": (90, 103),
    "2013-2014": (104, 118),
}
_PSU_MAX_BY_CYCLE = {
    "1999-2000": 3,
    "2001-2002": 2,
    "2003-2004": 2,
    "2005-2006": 2,
    "2007-2008": 2,
    "2009-2010": 3,
    "2011-2012": 3,
    "2013-2014": 2,
}

_SEX_MAPPING = {1: "male", 2: "female"}
_RACE_MAPPING = {
    1: "mexican_american",
    2: "other_hispanic",
    3: "non_hispanic_white",
    4: "non_hispanic_black",
    5: "other_or_multiracial",
}
_GENERAL_HEALTH_MAPPING = {
    1: "excellent",
    2: "very_good",
    3: "good",
    4: "fair",
    5: "poor",
}
_YES_NO_MAPPING = {1: True, 2: False}
_STANDARD_MISSING_CODES = {7: "refused", 9: "dont_know"}

_CORE_REQUIRED_SOURCES = {
    "CYCLE",
    "SEQN",
    "RIDAGEYR",
    "RIAGENDR",
    "RIDRETH1",
    "WTMEC2YR",
    "SDMVSTRA",
    "SDMVPSU",
    "BMXBMI",
    "SMQ020",
    "SMQ040",
    "BPQ020",
    "DIQ010",
    "MCQ160B",
    "MCQ160C",
    "MCQ160D",
    "MCQ160E",
    "MCQ160F",
    "ELIGSTAT",
    "MORTSTAT",
    "UCOD_LEADING",
    "PERMTH_INT",
}


class NHANESHarmonizationError(ValueError):
    """Raised when source availability, codes, or ranges violate the contract."""


def normalize_xpt_machine_zero(value: Any) -> Any:
    """Convert the legacy IBM/SAS machine-zero artefact to exact numeric zero."""

    if pd.isna(value) or isinstance(value, bool):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if numeric != 0 and abs(numeric) < SAS_XPORT_MACHINE_ZERO_THRESHOLD:
        return 0.0
    return value


def _require_columns(frame: pd.DataFrame, columns: Sequence[str] | set[str], context: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise NHANESHarmonizationError(f"{context} is missing required columns: {missing}")


def _as_number(value: Any, *, name: str) -> float:
    normalized = normalize_xpt_machine_zero(value)
    try:
        numeric = float(normalized)
    except (TypeError, ValueError) as exc:
        raise NHANESHarmonizationError(f"{name}: expected numeric value, got {value!r}") from exc
    if not math.isfinite(numeric):
        raise NHANESHarmonizationError(f"{name}: value must be finite, got {value!r}")
    return numeric


def _as_integer(value: Any, *, name: str) -> int:
    numeric = _as_number(value, name=name)
    if not numeric.is_integer():
        raise NHANESHarmonizationError(f"{name}: expected an integer code, got {value!r}")
    return int(numeric)


def _numeric_converter(
    name: str,
    *,
    minimum: float | None,
    maximum: float | None,
) -> Callable[[Any], tuple[Any, str | None]]:
    def convert(value: Any) -> tuple[Any, str | None]:
        if pd.isna(value):
            return pd.NA, "true_missing"
        numeric = _as_number(value, name=name)
        if minimum is not None and numeric < minimum:
            raise NHANESHarmonizationError(f"{name}: {numeric} is below {minimum}")
        if maximum is not None and numeric > maximum:
            raise NHANESHarmonizationError(f"{name}: {numeric} is above {maximum}")
        return numeric, None

    return convert


def _integer_converter(
    name: str,
    *,
    minimum: int | None,
    maximum: int | None,
) -> Callable[[Any], tuple[Any, str | None]]:
    def convert(value: Any) -> tuple[Any, str | None]:
        if pd.isna(value):
            return pd.NA, "true_missing"
        integer = _as_integer(value, name=name)
        if minimum is not None and integer < minimum:
            raise NHANESHarmonizationError(f"{name}: {integer} is below {minimum}")
        if maximum is not None and integer > maximum:
            raise NHANESHarmonizationError(f"{name}: {integer} is above {maximum}")
        return integer, None

    return convert


def _coded_converter(
    name: str,
    mapping: Mapping[int, Any],
    *,
    missing_codes: Mapping[int, str] | None = None,
) -> Callable[[Any], tuple[Any, str | None]]:
    missing = dict(missing_codes or {})

    def convert(value: Any) -> tuple[Any, str | None]:
        if pd.isna(value):
            return pd.NA, "true_missing"
        code = _as_integer(value, name=name)
        if code in missing:
            return pd.NA, missing[code]
        if code not in mapping:
            raise NHANESHarmonizationError(f"{name}: unknown code {code}")
        return mapping[code], None

    return convert


def _cast(series: pd.Series, dtype: str) -> pd.Series:
    if dtype == "string":
        return series.astype("string")
    if dtype == "boolean":
        return series.astype("boolean")
    if dtype == "integer":
        return series.astype("Int64")
    if dtype == "numeric":
        return series.astype("Float64")
    return series


def _assign_feature(
    output: pd.DataFrame,
    *,
    canonical: str,
    raw_values: pd.Series,
    sources: str | pd.Series,
    converter: Callable[[Any], tuple[Any, str | None]],
    dtype: str,
    strict: bool,
) -> None:
    output[f"{canonical}_raw"] = raw_values.copy()
    if isinstance(sources, str):
        source_values = pd.Series(sources, index=raw_values.index, dtype="string")
    else:
        source_values = sources.astype("string")
    output[f"{canonical}_source"] = source_values

    converted: list[Any] = []
    reasons: list[str | None] = []
    rows = zip(raw_values.items(), source_values.tolist(), strict=True)
    for (index, value), source_value in rows:
        if pd.isna(source_value):
            converted.append(pd.NA)
            reasons.append("source_unavailable")
            continue
        try:
            canonical_value, reason = converter(value)
        except NHANESHarmonizationError as exc:
            if strict:
                raise NHANESHarmonizationError(f"{exc} at row index {index!r}") from exc
            canonical_value, reason = pd.NA, "unmapped_or_invalid"
        converted.append(canonical_value)
        reasons.append(reason)
    output[canonical] = _cast(pd.Series(converted, index=raw_values.index), dtype)
    output[f"{canonical}_missing_reason"] = pd.Series(
        reasons, index=raw_values.index, dtype="string"
    )


def _cycle_aware_source(
    raw: pd.DataFrame,
    cycles: pd.Series,
    source_by_cycle: Mapping[str, str],
    *,
    canonical: str,
) -> tuple[pd.Series, pd.Series]:
    values = pd.Series(pd.NA, index=raw.index, dtype="object")
    sources = pd.Series(pd.NA, index=raw.index, dtype="string")
    for cycle in cycles.unique():
        source = source_by_cycle[str(cycle)]
        if source not in raw.columns:
            raise NHANESHarmonizationError(
                f"{canonical}: cycle {cycle} requires source column {source!r}"
            )
        mask = cycles.eq(cycle)
        values.loc[mask] = raw.loc[mask, source]
        sources.loc[mask] = source
    return values, sources


def _smoking_status(
    ever_value: Any,
    current_value: Any,
) -> tuple[Any, str | None]:
    if pd.isna(ever_value):
        return pd.NA, "true_missing"
    ever = _as_integer(ever_value, name="SMQ020")
    if ever == 7:
        return pd.NA, "refused"
    if ever == 9:
        return pd.NA, "dont_know"
    if ever == 2:
        if not pd.isna(current_value):
            current = _as_integer(current_value, name="SMQ040")
            if current not in {7, 9}:
                raise NHANESHarmonizationError(
                    "SMQ020 indicates never smoked but SMQ040 contains a smoking status"
                )
        return "never", None
    if ever != 1:
        raise NHANESHarmonizationError(f"SMQ020: unknown code {ever}")
    if pd.isna(current_value):
        return pd.NA, "current_status_missing"
    current = _as_integer(current_value, name="SMQ040")
    if current in {1, 2}:
        return "current", None
    if current == 3:
        return "former", None
    if current == 7:
        return pd.NA, "refused_current_status"
    if current == 9:
        return pd.NA, "dont_know_current_status"
    raise NHANESHarmonizationError(f"SMQ040: unknown code {current}")


def _raw_pair(first: Any, second: Any) -> str:
    def display(value: Any) -> str:
        return "<missing>" if pd.isna(value) else str(value)

    return f"SMQ020={display(first)};SMQ040={display(second)}"


def _ucod_converter(value: Any) -> tuple[Any, str | None]:
    if pd.isna(value) or str(value).strip() in {"", "."}:
        return pd.NA, "true_missing"
    text = str(value).strip()
    try:
        numeric = float(text)
    except ValueError as exc:
        raise NHANESHarmonizationError(f"UCOD_LEADING: invalid code {value!r}") from exc
    if not numeric.is_integer():
        raise NHANESHarmonizationError(f"UCOD_LEADING: invalid code {value!r}")
    normalized = f"{int(numeric):03d}"
    if normalized not in {f"{code:03d}" for code in range(1, 11)}:
        raise NHANESHarmonizationError(f"UCOD_LEADING: unknown code {normalized}")
    return normalized, None


def _assign_design(
    output: pd.DataFrame,
    raw: pd.DataFrame,
    cycles: pd.Series,
    strict: bool,
) -> None:
    output["survey_cycle_raw"] = raw["CYCLE"]
    output["survey_cycle_source"] = "CYCLE"
    output["survey_cycle"] = cycles.astype("string")
    output["survey_cycle_missing_reason"] = pd.Series(pd.NA, index=raw.index, dtype="string")
    output["survey_year_raw"] = raw["CYCLE"]
    output["survey_year_source"] = "CYCLE:start_year"
    output["survey_year"] = cycles.map(NHANES_CYCLE_START_YEAR).astype("Int64")
    output["survey_year_missing_reason"] = pd.Series(pd.NA, index=raw.index, dtype="string")

    _assign_feature(
        output,
        canonical="seqn",
        raw_values=raw["SEQN"],
        sources="SEQN",
        converter=_integer_converter("SEQN", minimum=1, maximum=999999),
        dtype="integer",
        strict=strict,
    )
    if output["seqn"].isna().any():
        raise NHANESHarmonizationError("SEQN cannot be missing")
    output["participant_id_raw"] = raw["SEQN"]
    output["participant_id_source"] = "CYCLE+SEQN"
    output["participant_id"] = (
        output["survey_cycle"].astype(str) + ":" + output["seqn"].astype(str)
    ).astype("string")
    output["participant_id_missing_reason"] = pd.Series(pd.NA, index=raw.index, dtype="string")

    for canonical, source in (("strata", "SDMVSTRA"), ("psu", "SDMVPSU")):
        _assign_feature(
            output,
            canonical=canonical,
            raw_values=raw[source],
            sources=source,
            converter=_integer_converter(source, minimum=1, maximum=None),
            dtype="integer",
            strict=strict,
        )
    _assign_feature(
        output,
        canonical="mec_weight",
        raw_values=raw["WTMEC2YR"],
        sources="WTMEC2YR",
        converter=_numeric_converter("WTMEC2YR", minimum=0, maximum=None),
        dtype="numeric",
        strict=strict,
    )
    early_cycle = cycles.isin({"1999-2000", "2001-2002"})
    four_year_raw = pd.Series(pd.NA, index=raw.index, dtype="object")
    four_year_source = pd.Series(pd.NA, index=raw.index, dtype="string")
    if "WTMEC4YR" in raw:
        four_year_raw.loc[early_cycle] = raw.loc[early_cycle, "WTMEC4YR"]
        four_year_source.loc[early_cycle] = "WTMEC4YR"
    _assign_feature(
        output,
        canonical="mec_four_year_weight",
        raw_values=four_year_raw,
        sources=four_year_source,
        converter=_numeric_converter("WTMEC4YR", minimum=0, maximum=None),
        dtype="numeric",
        strict=strict,
    )
    if early_cycle.any() and "WTMEC4YR" not in raw:
        raise NHANESHarmonizationError(
            "WTMEC4YR is required for the pooled 1999-2002 NHANES estimand"
        )
    missing_early_weight = early_cycle & output["mec_four_year_weight"].isna()
    if missing_early_weight.any():
        raise NHANESHarmonizationError(
            "WTMEC4YR is missing for "
            f"{int(missing_early_weight.sum())} records in the 1999-2002 cycles"
        )

    selected_weight = output["mec_weight"].copy()
    selected_weight.loc[early_cycle] = output.loc[early_cycle, "mec_four_year_weight"]
    divisor = pd.Series(8.0, index=raw.index, dtype="Float64")
    divisor.loc[early_cycle] = 4.0
    weight_source = pd.Series("WTMEC2YR/8", index=raw.index, dtype="string")
    weight_source.loc[early_cycle] = "WTMEC4YR/4"

    output["analysis_weight_raw"] = selected_weight.astype("Float64")
    output["analysis_weight_source"] = weight_source
    output["analysis_weight_multiplier"] = (1.0 / divisor).astype("Float64")
    output["analysis_weight"] = (selected_weight / divisor).astype("Float64")
    output["analysis_weight_missing_reason"] = pd.Series(
        pd.NA, index=raw.index, dtype="string"
    )
    output.loc[
        ~early_cycle & output["mec_weight"].isna(), "analysis_weight_missing_reason"
    ] = output.loc[~early_cycle, "mec_weight_missing_reason"]
    output["weight_protocol"] = "amendment_2_4_source_specific_pooling"
    output["weight_estimand"] = "nhanes_external_validation_1999_2014"


def _assign_fast_features(
    output: pd.DataFrame,
    raw: pd.DataFrame,
    cycles: pd.Series,
    strict: bool,
) -> None:
    _assign_feature(
        output,
        canonical="age",
        raw_values=raw["RIDAGEYR"],
        sources="RIDAGEYR",
        converter=_integer_converter("RIDAGEYR", minimum=0, maximum=85),
        dtype="integer",
        strict=strict,
    )
    _assign_feature(
        output,
        canonical="sex",
        raw_values=raw["RIAGENDR"],
        sources="RIAGENDR",
        converter=_coded_converter("RIAGENDR", _SEX_MAPPING),
        dtype="string",
        strict=strict,
    )
    _assign_feature(
        output,
        canonical="bmi",
        raw_values=raw["BMXBMI"],
        sources="BMXBMI",
        converter=_numeric_converter("BMXBMI", minimum=7, maximum=150),
        dtype="numeric",
        strict=strict,
    )

    smoking_pairs = zip(raw["SMQ020"], raw["SMQ040"], strict=True)
    smoking_raw = pd.Series(
        [_raw_pair(ever, current) for ever, current in smoking_pairs],
        index=raw.index,
        dtype="string",
    )
    smoking_source = pd.Series("SMQ020+SMQ040", index=raw.index, dtype="string")
    smoking_values: list[Any] = []
    smoking_reasons: list[str | None] = []
    for index in raw.index:
        try:
            value, reason = _smoking_status(raw.at[index, "SMQ020"], raw.at[index, "SMQ040"])
        except NHANESHarmonizationError as exc:
            if strict:
                raise NHANESHarmonizationError(f"{exc} at row index {index!r}") from exc
            value, reason = pd.NA, "unmapped_or_invalid"
        smoking_values.append(value)
        smoking_reasons.append(reason)
    output["smoking_status_raw"] = smoking_raw
    output["smoking_status_source"] = smoking_source
    output["smoking_status"] = pd.Series(smoking_values, index=raw.index, dtype="string")
    output["smoking_status_missing_reason"] = pd.Series(
        smoking_reasons, index=raw.index, dtype="string"
    )

    _assign_feature(
        output,
        canonical="hypertension_dx",
        raw_values=raw["BPQ020"],
        sources="BPQ020",
        converter=_coded_converter(
            "BPQ020", _YES_NO_MAPPING, missing_codes=_STANDARD_MISSING_CODES
        ),
        dtype="boolean",
        strict=strict,
    )
    _assign_feature(
        output,
        canonical="diabetes_dx",
        raw_values=raw["DIQ010"],
        sources="DIQ010",
        converter=_coded_converter(
            "DIQ010",
            _YES_NO_MAPPING,
            missing_codes={3: "borderline", **_STANDARD_MISSING_CODES},
        ),
        dtype="boolean",
        strict=strict,
    )
    general_raw, general_sources = _cycle_aware_source(
        raw,
        cycles,
        GENERAL_HEALTH_SOURCE,
        canonical="general_health",
    )
    _assign_feature(
        output,
        canonical="general_health",
        raw_values=general_raw,
        sources=general_sources,
        converter=_coded_converter(
            "general health",
            _GENERAL_HEALTH_MAPPING,
            missing_codes=_STANDARD_MISSING_CODES,
        ),
        dtype="string",
        strict=strict,
    )
    if "HSD010" in raw:
        hsd_source = pd.Series("HSD010", index=raw.index, dtype="string")
        expected_unavailable = cycles.eq("1999-2000")
        hsd_source.loc[expected_unavailable] = "EXPECTED_NOT_AVAILABLE"
        _assign_feature(
            output,
            canonical="general_health_hsd_sensitivity",
            raw_values=raw["HSD010"],
            sources=hsd_source,
            converter=_coded_converter(
                "HSD010",
                _GENERAL_HEALTH_MAPPING,
                missing_codes=_STANDARD_MISSING_CODES,
            ),
            dtype="string",
            strict=strict,
        )
        output.loc[
            expected_unavailable,
            "general_health_hsd_sensitivity_missing_reason",
        ] = "expected_not_available"


def _assign_race_and_baseline(output: pd.DataFrame, raw: pd.DataFrame, strict: bool) -> None:
    _assign_feature(
        output,
        canonical="race_ethnicity",
        raw_values=raw["RIDRETH1"],
        sources="RIDRETH1",
        converter=_coded_converter(
            "RIDRETH1", _RACE_MAPPING, missing_codes=_STANDARD_MISSING_CODES
        ),
        dtype="string",
        strict=strict,
    )
    for canonical, source in BASELINE_CVD_SOURCE.items():
        _assign_feature(
            output,
            canonical=canonical,
            raw_values=raw[source],
            sources=source,
            converter=_coded_converter(
                source, _YES_NO_MAPPING, missing_codes=_STANDARD_MISSING_CODES
            ),
            dtype="boolean",
            strict=strict,
        )


def _assign_outcome_components(output: pd.DataFrame, raw: pd.DataFrame, strict: bool) -> None:
    specifications = (
        (
            "mortality_eligible",
            "ELIGSTAT",
            _coded_converter("ELIGSTAT", {1: True, 2: False, 3: False}),
            "boolean",
        ),
        (
            "deceased",
            "MORTSTAT",
            _coded_converter("MORTSTAT", {0: False, 1: True}),
            "boolean",
        ),
        ("ucod_leading", "UCOD_LEADING", _ucod_converter, "string"),
        (
            "permth_int",
            "PERMTH_INT",
            _integer_converter("PERMTH_INT", minimum=0, maximum=374),
            "integer",
        ),
    )
    for canonical, source, converter, dtype in specifications:
        _assign_feature(
            output,
            canonical=canonical,
            raw_values=raw[source],
            sources=source,
            converter=converter,
            dtype=dtype,
            strict=strict,
        )
    output["ELIGSTAT"] = (
        output["mortality_eligible_raw"]
        .map(lambda value: pd.NA if pd.isna(value) else _as_integer(value, name="ELIGSTAT"))
        .astype("Int64")
    )
    output["MORTSTAT"] = (
        output["deceased_raw"]
        .map(lambda value: pd.NA if pd.isna(value) else _as_integer(value, name="MORTSTAT"))
        .astype("Int64")
    )
    output["UCOD_LEADING"] = output["ucod_leading"].astype("string")
    output["PERMTH_INT"] = output["permth_int"].astype("Int64")


def _raw_readings(row: pd.Series) -> str:
    values = ["<missing>" if pd.isna(row[source]) else str(row[source]) for source in SBP_SOURCES]
    return ";".join(f"{source}={value}" for source, value in zip(SBP_SOURCES, values, strict=True))


def _systolic_reading(value: Any, *, source: str) -> float:
    numeric = _as_number(value, name=source)
    if numeric < 40 or numeric > 300:
        raise NHANESHarmonizationError(f"{source}: {numeric} is outside 40-300 mmHg")
    if not numeric.is_integer() or int(numeric) % 2:
        raise NHANESHarmonizationError(f"{source}: expected an even integer mmHg reading")
    return numeric


def _assign_extended(
    output: pd.DataFrame,
    raw: pd.DataFrame,
    cycles: pd.Series,
    strict: bool,
) -> None:
    _require_columns(raw, {*SBP_SOURCES, "LBXTC", "LBXGH"}, "Extended NHANES input")
    bp_values: list[Any] = []
    bp_reasons: list[str | None] = []
    for index, row in raw.loc[:, SBP_SOURCES].iterrows():
        readings: list[float] = []
        for source in SBP_SOURCES:
            value = row[source]
            if pd.isna(value):
                continue
            try:
                converted = _systolic_reading(value, source=source)
            except NHANESHarmonizationError as exc:
                if strict:
                    raise NHANESHarmonizationError(f"{exc} at row index {index!r}") from exc
                continue
            readings.append(converted)
        if readings:
            bp_values.append(sum(readings) / len(readings))
            bp_reasons.append(None)
        else:
            bp_values.append(pd.NA)
            bp_reasons.append("true_missing")
    output["systolic_bp_raw"] = raw.loc[:, SBP_SOURCES].apply(_raw_readings, axis=1)
    output["systolic_bp_source"] = "+".join(SBP_SOURCES)
    output["systolic_bp"] = pd.Series(bp_values, index=raw.index, dtype="Float64")
    output["systolic_bp_missing_reason"] = pd.Series(bp_reasons, index=raw.index, dtype="string")
    output["systolic_bp_mean"] = output["systolic_bp"]

    _assign_feature(
        output,
        canonical="total_cholesterol",
        raw_values=raw["LBXTC"],
        sources="LBXTC",
        converter=_numeric_converter("LBXTC", minimum=40, maximum=850),
        dtype="numeric",
        strict=strict,
    )
    hdl_raw, hdl_sources = _cycle_aware_source(raw, cycles, HDL_SOURCE, canonical="hdl_cholesterol")
    _assign_feature(
        output,
        canonical="hdl_cholesterol",
        raw_values=hdl_raw,
        sources=hdl_sources,
        converter=_numeric_converter("HDL cholesterol", minimum=5, maximum=250),
        dtype="numeric",
        strict=strict,
    )
    _assign_feature(
        output,
        canonical="hba1c",
        raw_values=raw["LBXGH"],
        sources="LBXGH",
        converter=_numeric_converter("LBXGH", minimum=2, maximum=20),
        dtype="numeric",
        strict=strict,
    )


def harmonize_nhanes(
    raw: pd.DataFrame,
    *,
    include_extended: bool = False,
    strict: bool = True,
) -> pd.DataFrame:
    """Return canonical NHANES values with raw/source/missing provenance."""

    if raw.empty:
        raise NHANESHarmonizationError("Cannot harmonize an empty NHANES frame")
    raw = raw.reset_index(drop=True).copy()
    _require_columns(raw, _CORE_REQUIRED_SOURCES, "NHANES input")
    cycles = raw["CYCLE"].astype("string").str.strip()
    if cycles.isna().any():
        raise NHANESHarmonizationError("CYCLE cannot be missing")
    unknown_cycles = sorted(set(cycles).difference(NHANES_CYCLE_START_YEAR))
    if unknown_cycles:
        raise NHANESHarmonizationError(f"Unknown NHANES cycles: {unknown_cycles}")

    output = pd.DataFrame(index=raw.index)
    _assign_design(output, raw, cycles, strict)
    _assign_fast_features(output, raw, cycles, strict)
    _assign_race_and_baseline(output, raw, strict)
    _assign_outcome_components(output, raw, strict)
    if include_extended:
        _assign_extended(output, raw, cycles, strict)
    validate_harmonized_nhanes(output, include_extended=include_extended)
    return output.reset_index(drop=True)


def validate_harmonized_nhanes(
    frame: pd.DataFrame,
    *,
    include_extended: bool = False,
) -> None:
    """Validate the cross-source canonical schema and NHANES design fields."""

    canonical = {
        *FAST_FEATURES,
        *SENSITIVITY_BASELINE_CVD_COLUMNS,
        "race_ethnicity",
        "participant_id",
        "survey_cycle",
        "survey_year",
        "strata",
        "psu",
        "mec_weight",
        "analysis_weight",
        "mortality_eligible",
        "deceased",
        "ucod_leading",
        "permth_int",
        "ELIGSTAT",
        "MORTSTAT",
        "UCOD_LEADING",
        "PERMTH_INT",
    }
    if include_extended:
        canonical.update(EXTENDED_FEATURES)
    missing = sorted(canonical.difference(frame.columns))
    if missing:
        raise NHANESHarmonizationError(f"Harmonized NHANES misses columns: {missing}")
    provenance_variables = {
        *FAST_FEATURES,
        *SENSITIVITY_BASELINE_CVD_COLUMNS,
        "race_ethnicity",
        "participant_id",
    }
    if include_extended:
        provenance_variables.update(EXTENDED_FEATURES)
    missing_provenance = sorted(
        column
        for variable in provenance_variables
        for column in (f"{variable}_raw", f"{variable}_source", f"{variable}_missing_reason")
        if column not in frame
    )
    if missing_provenance:
        raise NHANESHarmonizationError(
            f"Harmonized NHANES misses provenance columns: {missing_provenance}"
        )
    if frame["participant_id"].isna().any() or frame["participant_id"].duplicated().any():
        raise NHANESHarmonizationError("NHANES participant_id must be present and unique")
    if not frame["survey_cycle"].isin(NHANES_CYCLE_START_YEAR).all():
        raise NHANESHarmonizationError("Harmonized NHANES contains an invalid survey_cycle")
    expected_year = frame["survey_cycle"].map(NHANES_CYCLE_START_YEAR).astype("Int64")
    if not frame["survey_year"].equals(expected_year):
        raise NHANESHarmonizationError("survey_year does not equal the cycle start year")

    for cycle, group in frame.groupby("survey_cycle", observed=True):
        cycle_name = str(cycle)
        age = pd.to_numeric(group["age"], errors="coerce").dropna()
        if age.gt(_AGE_MAX_BY_CYCLE[cycle_name]).any():
            raise NHANESHarmonizationError(
                f"RIDAGEYR exceeds the {cycle_name} topcode {_AGE_MAX_BY_CYCLE[cycle_name]}"
            )
        strata = pd.to_numeric(group["strata"], errors="coerce").dropna()
        strata_minimum, strata_maximum = _STRATA_RANGE_BY_CYCLE[cycle_name]
        if not strata.between(strata_minimum, strata_maximum, inclusive="both").all():
            raise NHANESHarmonizationError(f"SDMVSTRA is outside the {cycle_name} range")
        psu = pd.to_numeric(group["psu"], errors="coerce").dropna()
        if not psu.between(1, _PSU_MAX_BY_CYCLE[cycle_name], inclusive="both").all():
            raise NHANESHarmonizationError(f"SDMVPSU is outside the {cycle_name} range")

    allowed_categories = {
        "sex": {"male", "female"},
        "smoking_status": {"never", "former", "current"},
        "general_health": {"excellent", "very_good", "good", "fair", "poor"},
    }
    for column, allowed in allowed_categories.items():
        invalid = set(frame[column].dropna().astype(str)).difference(allowed)
        if invalid:
            raise NHANESHarmonizationError(f"{column} has invalid categories: {sorted(invalid)}")
    try:
        validate_fast_feature_schema(frame, allow_missing=True)
    except CanonicalSchemaError as exc:
        raise NHANESHarmonizationError(f"Canonical fast-feature schema violation: {exc}") from exc
    weights = pd.to_numeric(frame["mec_weight"], errors="coerce")
    if (weights.dropna() < 0).any():
        raise NHANESHarmonizationError("mec_weight cannot be negative")
    early_cycle = frame["survey_cycle"].isin({"1999-2000", "2001-2002"})
    if early_cycle.any() and "mec_four_year_weight" not in frame:
        raise NHANESHarmonizationError("mec_four_year_weight is required for 1999-2002")
    expected_weight = (weights / 8.0).astype("Float64")
    expected_weight.loc[early_cycle] = (
        pd.to_numeric(frame.loc[early_cycle, "mec_four_year_weight"], errors="coerce") / 4.0
    ).astype("Float64")
    actual_weight = frame["analysis_weight"].astype("Float64")
    comparable = expected_weight.notna() & actual_weight.notna()
    if not (expected_weight.loc[comparable] == actual_weight.loc[comparable]).all():
        raise NHANESHarmonizationError(
            "analysis_weight must equal WTMEC4YR/4 for 1999-2002 and WTMEC2YR/8 later"
        )
