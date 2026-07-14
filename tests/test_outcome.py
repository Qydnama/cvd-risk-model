from __future__ import annotations

import pandas as pd
import pytest

from cvd_risk.data.cohort import (
    OutcomeDefinitionError,
    derive_nhanes_cvd_outcome,
    derive_nhis_cvd_outcome,
)


def _nhis_outcome_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": ["q19", "q20", "q21", "other", "alive"],
            "survey_year": [2000] * 5,
            "interview_quarter": [1] * 5,
            "mortality_eligible": [True] * 5,
            "deceased": [True, True, True, True, False],
            "ucod_leading": ["001", "005", "001", "002", pd.NA],
            "death_year": [2004, 2005, 2005, 2001, pd.NA],
            "death_quarter": [4, 1, 2, 1, pd.NA],
        }
    )


def test_nhis_primary_excludes_quarter_20_and_sensitivity_includes_it() -> None:
    primary = derive_nhis_cvd_outcome(_nhis_outcome_rows())

    assert primary["followup_quarters"].tolist()[:4] == [19, 20, 21, 4]
    assert primary.loc[0, "cvd_death_5y"] == 1
    assert pd.isna(primary.loc[1, "cvd_death_5y"])
    assert primary.loc[1, "outcome_missing_reason"] == "ambiguous_20_quarter_boundary"
    assert bool(primary.loc[1, "cvd_death_5y_boundary_20q"])
    assert primary.loc[2, "cvd_death_5y"] == 0
    assert primary.loc[3, "cvd_death_5y"] == 0
    assert primary.loc[4, "cvd_death_5y"] == 0

    sensitivity = derive_nhis_cvd_outcome(_nhis_outcome_rows(), include_ambiguous_boundary=True)
    assert sensitivity.loc[1, "cvd_death_5y"] == 1
    assert pd.isna(sensitivity.loc[1, "outcome_missing_reason"])


def test_nhis_rejects_negative_time_instead_of_silently_repairing_it() -> None:
    row = _nhis_outcome_rows().iloc[[0]].copy()
    row["death_year"] = 1999

    with pytest.raises(OutcomeDefinitionError, match="Negative NHIS follow-up"):
        derive_nhis_cvd_outcome(row)


def test_nhis_unknown_linkage_or_cause_never_becomes_zero() -> None:
    rows = _nhis_outcome_rows().iloc[[0, 1]].copy()
    rows = rows.assign(
        mortality_eligible=pd.Series([pd.NA, True], index=rows.index, dtype="object"),
        ucod_leading=pd.Series(["001", pd.NA], index=rows.index, dtype="object"),
    )

    outcome = derive_nhis_cvd_outcome(rows)

    assert outcome["cvd_death_5y"].isna().all()
    assert outcome.iloc[0]["outcome_missing_reason"] == "mortality_eligibility_unknown"
    assert outcome.iloc[1]["outcome_missing_reason"] == "cause_unknown"


def test_nhanes_exact_month_rule_includes_month_60() -> None:
    frame = pd.DataFrame(
        {
            "ELIGSTAT": [1, 1, 1, 1],
            "MORTSTAT": [1, 1, 1, 0],
            "UCOD_LEADING": [1, "005", 2, pd.NA],
            "PERMTH_INT": [60, 61, 3, 72],
        }
    )

    outcome = derive_nhanes_cvd_outcome(frame)

    assert outcome["cvd_death_5y"].tolist() == [1, 0, 0, 0]


def test_nhanes_alive_requires_full_potential_followup() -> None:
    frame = pd.DataFrame(
        {
            "ELIGSTAT": [1, 1, 1],
            "MORTSTAT": [0, 0, 0],
            "UCOD_LEADING": [pd.NA, pd.NA, pd.NA],
            "PERMTH_INT": [60, 59, pd.NA],
        }
    )
    outcome = derive_nhanes_cvd_outcome(frame)
    assert outcome.loc[0, "cvd_death_5y"] == 0
    assert pd.isna(outcome.loc[1, "cvd_death_5y"])
    assert pd.isna(outcome.loc[2, "cvd_death_5y"])
    assert outcome.loc[1, "outcome_missing_reason"] == "insufficient_potential_followup"


def test_nhanes_rejects_negative_followup() -> None:
    frame = pd.DataFrame(
        {
            "ELIGSTAT": [1],
            "MORTSTAT": [1],
            "UCOD_LEADING": [1],
            "PERMTH_INT": [-1],
        }
    )
    with pytest.raises(OutcomeDefinitionError, match="Negative NHANES follow-up"):
        derive_nhanes_cvd_outcome(frame)


@pytest.mark.parametrize("bad_months", ["not-a-number", 12.5])
def test_nhanes_rejects_malformed_or_fractional_followup(bad_months: object) -> None:
    frame = pd.DataFrame(
        {
            "ELIGSTAT": [1],
            "MORTSTAT": [1],
            "UCOD_LEADING": [1],
            "PERMTH_INT": [bad_months],
        }
    )
    with pytest.raises(OutcomeDefinitionError, match="PERMTH_INT"):
        derive_nhanes_cvd_outcome(frame)


@pytest.mark.parametrize("bad_cause", [777, "garbage"])
def test_unknown_leading_cause_never_becomes_non_cvd_zero(bad_cause: object) -> None:
    frame = pd.DataFrame(
        {
            "ELIGSTAT": [1],
            "MORTSTAT": [1],
            "UCOD_LEADING": [bad_cause],
            "PERMTH_INT": [12],
        }
    )
    with pytest.raises(OutcomeDefinitionError, match="UCOD"):
        derive_nhanes_cvd_outcome(frame)
