from cvd_risk.reporting.harmonization import harmonization_differences_table


def test_harmonization_report_records_measured_vs_reported_bmi_and_huq() -> None:
    table = harmonization_differences_table().set_index("feature")
    assert "self-reported" in str(table.loc["bmi", "nhis_measurement"])
    assert "measured" in str(table.loc["bmi", "nhanes_measurement"])
    assert "HUQ010" in str(table.loc["general_health", "nhanes_measurement"])
    assert "audit-only" in str(table.loc["race_ethnicity", "comparability"])
