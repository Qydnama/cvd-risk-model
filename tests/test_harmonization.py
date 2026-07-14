from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cvd_risk.data.harmonize import (
    FAST_FEATURES,
    HarmonizationError,
    harmonize_nhis,
    mapping_roles,
)
from cvd_risk.data.load_nhis import (
    NHISInputError,
    discover_nhis_input,
    load_nhis,
    validate_nhis_schema,
)


def _raw_nhis() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "YEAR": [2009, 2010],
            "NHISPID": ["person-1", "person-2"],
            "QUARTER": [1, 2],
            "STRATA": [10, 11],
            "PSU": [1, 2],
            "MORTWTSA": [130.0, 50.0],
            "AGE": [55, 62],
            "SEX": [1, 2],
            "BMI": [2750, 9999],
            "SMOKESTATUS2": [30, 40],
            "HYPERTENEV": [1, 7],
            "DIABETICEV": [2, 3],
            "HEALTH": [3, 9],
            "CHEARTDIEV": [1, 1],
            "ANGIPECEV": [1, 1],
            "HEARTATTEV": [1, 1],
            "STROKEV": [1, 1],
            "CONHFAILEV": [1, 1],
            "MORTELIG": [1, 1],
            "MORTSTAT": [1, 0],
            "MORTUCODLD": [1, 0],
            "MORTDODY": [2012, 0],
            "MORTDODQ": [4, 0],
        }
    )


@pytest.mark.parametrize("suffix", [".csv", ".csv.gz", ".dta"])
def test_loader_supports_protocol_formats_and_requires_codebook(
    tmp_path: Path, suffix: str
) -> None:
    raw = _raw_nhis()
    data_path = tmp_path / f"nhis_extract{suffix}"
    if suffix == ".dta":
        raw.to_stata(data_path, write_index=False)
    else:
        raw.to_csv(data_path, index=False)
    (tmp_path / "nhis_extract_codebook.xml").write_text("<codebook />", encoding="utf-8")

    loaded = load_nhis(tmp_path)

    assert len(loaded) == 2
    assert Path(loaded.attrs["codebook_path"]).name == "nhis_extract_codebook.xml"


def test_codebook_csv_is_not_mistaken_for_a_second_data_extract(tmp_path: Path) -> None:
    _raw_nhis().to_csv(tmp_path / "extract.csv", index=False)
    pd.DataFrame({"variable": ["SEX"], "label": ["Sex"]}).to_csv(
        tmp_path / "extract_codebook.csv", index=False
    )

    discovered = discover_nhis_input(tmp_path)

    assert discovered.data_path.name == "extract.csv"
    assert discovered.codebook_path.name == "extract_codebook.csv"


def test_loader_rejects_missing_codebook(tmp_path: Path) -> None:
    _raw_nhis().to_csv(tmp_path / "extract.csv", index=False)

    with pytest.raises(FileNotFoundError, match="codebook"):
        load_nhis(tmp_path)


def test_schema_validation_is_strict_about_required_columns_and_ids() -> None:
    missing = _raw_nhis().drop(columns="MORTUCODLD")
    with pytest.raises(NHISInputError, match="MORTUCODLD"):
        validate_nhis_schema(missing)

    duplicate = _raw_nhis()
    duplicate.loc[1, "NHISPID"] = duplicate.loc[0, "NHISPID"]
    with pytest.raises(NHISInputError, match="not unique"):
        validate_nhis_schema(duplicate)


def test_harmonization_keeps_raw_canonical_and_missing_reason() -> None:
    harmonized = harmonize_nhis(_raw_nhis())

    assert set(FAST_FEATURES).issubset(harmonized.columns)
    assert harmonized.loc[0, "bmi_raw"] == 2750
    assert harmonized.loc[0, "bmi"] == pytest.approx(27.5)
    assert harmonized.loc[0, "smoking_status"] == "never"
    assert bool(harmonized.loc[0, "hypertension_dx"]) is False
    assert bool(harmonized.loc[0, "diabetes_dx"]) is True
    assert harmonized.loc[0, "general_health"] == "good"
    assert harmonized.loc[0, "ucod_leading"] == "001"

    assert pd.isna(harmonized.loc[1, "bmi"])
    assert harmonized.loc[1, "bmi_missing_reason"] == "unknown"
    assert pd.isna(harmonized.loc[1, "smoking_status"])
    assert harmonized.loc[1, "smoking_status_missing_reason"] == "current_status_unknown"
    assert pd.isna(harmonized.loc[1, "hypertension_dx"])
    assert harmonized.loc[1, "hypertension_dx_missing_reason"] == "refused"
    assert pd.isna(harmonized.loc[1, "diabetes_dx"])
    assert (
        harmonized.loc[1, "diabetes_dx_missing_reason"]
        == "borderline_response_requires_protocol_decision"
    )


def test_unknown_nonmissing_code_fails_instead_of_becoming_reference_category() -> None:
    raw = _raw_nhis()
    raw.loc[0, "CHEARTDIEV"] = 6

    with pytest.raises(HarmonizationError, match="unmapped source code"):
        harmonize_nhis(raw)


def test_race_ethnicity_is_optional_and_explicitly_audit_only() -> None:
    without_race = harmonize_nhis(_raw_nhis())
    assert without_race["race_ethnicity"].isna().all()
    assert without_race["race_ethnicity_missing_reason"].eq("source_unavailable").all()

    raw = _raw_nhis().assign(RACE_ETHNICITY=["group-a", "group-b"])
    with_race = harmonize_nhis(raw)
    assert with_race["race_ethnicity"].tolist() == ["group-a", "group-b"]
    assert mapping_roles()["race_ethnicity"] == "audit_only"
