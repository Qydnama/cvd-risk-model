"""Pre-specified cross-source harmonization differences for stage 2."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HARMONIZATION_DIFFERENCES: tuple[dict[str, str], ...] = (
    {
        "feature": "age",
        "nhis_measurement": "Interview age (IPUMS AGE)",
        "nhanes_measurement": "Interview age (RIDAGEYR)",
        "comparability": "closely comparable after common 40–79 restriction",
        "required_action": "validate integer range and cycle/year coverage",
    },
    {
        "feature": "sex",
        "nhis_measurement": "SEX",
        "nhanes_measurement": "RIAGENDR (official release label: Gender)",
        "comparability": "binary male/female coding; wording is not explicitly sex at birth",
        "required_action": "document wording mismatch for the future site interface",
    },
    {
        "feature": "bmi",
        "nhis_measurement": "derived from self-reported height and weight",
        "nhanes_measurement": "measured MEC BMI (BMXBMI)",
        "comparability": "same unit, materially different measurement process",
        "required_action": "report external-distribution shift and calibration",
    },
    {
        "feature": "smoking_status",
        "nhis_measurement": "harmonized SMOKESTATUS2",
        "nhanes_measurement": "derived jointly from SMQ020 and SMQ040",
        "comparability": "common never/former/current categories",
        "required_action": "keep unknown question-universe responses missing",
    },
    {
        "feature": "hypertension_dx",
        "nhis_measurement": "self-reported prior diagnosis (HYPERTENEV)",
        "nhanes_measurement": "self-reported prior diagnosis (BPQ020)",
        "comparability": "questionnaire diagnosis; not measured blood pressure",
        "required_action": "do not substitute BP measurements in fast model",
    },
    {
        "feature": "diabetes_dx",
        "nhis_measurement": "self-reported prior diagnosis (DIABETICEV)",
        "nhanes_measurement": "self-reported prior diagnosis (DIQ010)",
        "comparability": "borderline responses are not equivalent to No",
        "required_action": "retain borderline as a distinct missing reason",
    },
    {
        "feature": "general_health",
        "nhis_measurement": "HEALTH",
        "nhanes_measurement": "1999–2000 HUQ010; 2001–2014 HSD010",
        "comparability": "common ordered excellent→poor labels after cycle check",
        "required_action": "download HUQ.xpt for 1999 rather than HSQ.xpt",
    },
    {
        "feature": "race_ethnicity",
        "nhis_measurement": "requires explicit derivation from race + ethnicity",
        "nhanes_measurement": "RIDRETH1 audit categories",
        "comparability": "audit-only; categories are not automatically equivalent",
        "required_action": "defer NHIS derivation until exact IPUMS codebook review",
    },
    {
        "feature": "outcome_timing",
        "nhis_measurement": "interview/death year and quarter",
        "nhanes_measurement": "PERMTH_INT exact integer months",
        "comparability": "same five-year estimand with different precision",
        "required_action": "NHIS quarter 20 excluded in primary, included in sensitivity",
    },
    {
        "feature": "survey_design",
        "nhis_measurement": "MORTWTSA, STRATA, PSU; annual pooled weights",
        "nhanes_measurement": "WTMEC2YR, SDMVSTRA, SDMVPSU; eight MEC cycles",
        "comparability": "source-specific design and scaling",
        "required_action": "never concatenate and apply one shared survey weight",
    },
    {
        "feature": "legacy_xpt_zero",
        "nhis_measurement": "not applicable",
        "nhanes_measurement": "legacy SAS zero may decode near 5.397605e-79",
        "comparability": "parser artifact, not a biological measurement",
        "required_action": "normalize machine-zero before age/weight eligibility checks",
    },
    {
        "feature": "extended_laboratory_files",
        "nhis_measurement": "not available",
        "nhanes_measurement": "1999–2004 LAB13/L13 and LAB10/L10 legacy files",
        "comparability": "NHANES-only extended analysis",
        "required_action": "use cycle-aware HDL names and verify laboratory weights",
    },
)


def harmonization_differences_table() -> pd.DataFrame:
    """Return a stable, publication-ready differences table."""

    return pd.DataFrame(HARMONIZATION_DIFFERENCES)


def write_harmonization_differences(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    harmonization_differences_table().to_csv(temporary, index=False)
    temporary.replace(output)
    return output
