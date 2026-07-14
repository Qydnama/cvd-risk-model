"""Orchestration for the mandatory stage-3 cohort audit."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from cvd_risk.reporting.figures import (
    plot_age_distribution,
    plot_outcome_by_year,
)
from cvd_risk.reporting.harmonization import harmonization_differences_table
from cvd_risk.reporting.protocol_discrepancies import (
    load_protocol_discrepancies,
    protocol_discrepancy_checks,
)
from cvd_risk.reporting.tables import (
    FAST_FEATURES,
    StopGateResult,
    StopGateThresholds,
    evaluate_stop_gate,
    event_counts_table,
    feature_availability_by_period_table,
    missingness_reason_table,
    missingness_table,
    subgroup_event_counts_table,
)
from cvd_risk.schemas import validate_fast_feature_schema
from cvd_risk.stage0 import _find_rscript

EXPECTED_NHIS_YEARS = tuple(range(1997, 2015))
EXPECTED_NHANES_CYCLES = (
    "1999-2000",
    "2001-2002",
    "2003-2004",
    "2005-2006",
    "2007-2008",
    "2009-2010",
    "2011-2012",
    "2013-2014",
)


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a supported processed/interim table with an explicit error."""

    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(
            f"Required table not found: {table_path}. Run the cohort builder first."
        )
    suffixes = [suffix.lower() for suffix in table_path.suffixes]
    if suffixes[-1:] == [".parquet"]:
        return pd.read_parquet(table_path)
    if suffixes[-1:] == [".csv"] or suffixes[-2:] == [".csv", ".gz"]:
        return pd.read_csv(table_path)
    raise ValueError(f"Unsupported table format: {table_path}")


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _validate_cohort_contract(frame: pd.DataFrame, name: str) -> None:
    required = {
        "participant_id",
        "survey_year",
        "analysis_weight",
        "strata",
        "psu",
        "cvd_death_5y",
    }.union(FAST_FEATURES)
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Cohort {name!r} is missing canonical columns: {sorted(missing)}")
    if frame["participant_id"].isna().any():
        raise ValueError(f"Cohort {name!r} has missing participant identifiers")
    if frame[["strata", "psu"]].isna().any().any():
        raise ValueError(f"Cohort {name!r} has missing survey strata or PSU")
    duplicates = int(frame["participant_id"].duplicated().sum())
    if duplicates:
        raise ValueError(f"Cohort {name!r} has {duplicates} duplicated participants")
    if not frame["age"].between(40, 79, inclusive="both").all():
        raise ValueError(f"Cohort {name!r} contains participants outside age 40–79")
    validate_fast_feature_schema(frame, allow_missing=True)


def _validate_nhis_codebook_binding(frame: pd.DataFrame) -> dict[str, str]:
    required = {
        "codebook_verification_status",
        "extract_sha256",
        "codebook_sha256",
        "feature_mapping_sha256",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "NHIS codebook verification is not bound to the processed cohort; "
            f"missing columns: {sorted(missing)}"
        )
    null_counts = frame[list(required)].isna().sum()
    if int(null_counts.sum()) > 0:
        details = {column: int(count) for column, count in null_counts.items() if count}
        raise ValueError(f"NHIS codebook binding contains missing values: {details}")
    statuses = set(frame["codebook_verification_status"].dropna().astype(str))
    if statuses != {"verified_against_user_extract"}:
        raise ValueError(
            "NHIS codebook mappings are not verified against the user's exact "
            f"extract: {sorted(statuses)}"
        )
    binding: dict[str, str] = {}
    for column in ("extract_sha256", "codebook_sha256", "feature_mapping_sha256"):
        values = set(frame[column].dropna().astype(str).str.lower())
        if len(values) != 1:
            raise ValueError(f"NHIS cohort must have one immutable {column} binding")
        value = next(iter(values))
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"NHIS {column} is not a valid SHA-256 digest")
        binding[column] = value
    return binding


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance_entry(path: str | Path) -> dict[str, str | int]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Provenance input does not exist: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _load_flow(path: str | Path, source: str) -> pd.DataFrame:
    flow = read_table(path)
    required = {"step_order", "step", "n_participants", "removed_from_previous"}
    missing = required.difference(flow.columns)
    if missing:
        raise KeyError(f"{source} cohort flow is missing columns: {sorted(missing)}")
    ordered = flow.sort_values("step_order").reset_index(drop=True)
    if ordered["step_order"].duplicated().any():
        raise ValueError(f"{source} cohort flow has duplicate step_order values")
    if ordered["step"].astype(str).duplicated().any():
        raise ValueError(f"{source} cohort flow has duplicate step labels")
    counts = pd.to_numeric(ordered["n_participants"], errors="raise")
    removed = pd.to_numeric(ordered["removed_from_previous"], errors="raise")
    if ((counts < 0) | (counts % 1 != 0)).any():
        raise ValueError(f"{source} cohort flow participant counts must be integers")
    if ((removed < 0) | (removed % 1 != 0)).any():
        raise ValueError(f"{source} cohort flow removal counts must be integers")
    if not counts.is_monotonic_decreasing:
        raise ValueError(f"{source} cohort flow grows after a filter or merge")
    expected_removed = counts.shift(1).sub(counts).fillna(0)
    if not removed.eq(expected_removed).all():
        mismatch = ordered.loc[
            ~removed.eq(expected_removed),
            ["step_order", "step", "n_participants", "removed_from_previous"],
        ]
        raise ValueError(
            f"{source} cohort flow removed_from_previous is inconsistent; "
            f"examples: {mismatch.head(5).to_dict(orient='records')}"
        )
    return ordered


def _baseline_cvd_sensitivity_table(
    cohorts: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Summarize the pre-specified CHF-strict sensitivity subsets."""

    required = {
        "congestive_heart_failure",
        "baseline_cvd_status_with_chf",
        "chf_sensitivity_available",
        "cohort_chf_sensitivity_eligible",
    }
    rows: list[dict[str, object]] = []
    for cohort_name, frame in cohorts.items():
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(
                f"Cohort {cohort_name!r} is missing CHF sensitivity fields: {sorted(missing)}"
            )
        available = frame["chf_sensitivity_available"].astype(bool)
        strict_eligible = frame["cohort_chf_sensitivity_eligible"].astype(bool)
        chf = frame["congestive_heart_failure"].astype("boolean")
        rows.append(
            {
                "cohort": cohort_name,
                "primary_n": len(frame),
                "chf_available_n": int(available.sum()),
                "chf_present_n": int((available & chf.eq(True).fillna(False)).sum()),
                "chf_unknown_n": int((available & chf.isna()).sum()),
                "chf_strict_n": int(strict_eligible.sum()),
                "scope": (
                    "NHIS 1999/2002 only"
                    if cohort_name.startswith("nhis")
                    else "NHANES 1999-2014"
                ),
            }
        )
    return pd.DataFrame(rows)


def _run_r_survey_design_gate(
    cohorts: dict[str, pd.DataFrame],
    *,
    r2_fraction: float,
    parameters: int,
    shrinkage: float,
    max_ci_half_width: float,
    minimum_design_df: int,
) -> pd.DataFrame:
    """Run the locked survey/pmsampsize gate in the reference R backend."""

    rscript = _find_rscript()
    if rscript is None:
        raise RuntimeError("Rscript is required for the Stage-3 design-based gate")
    parts: list[pd.DataFrame] = []
    for cohort_name, frame in cohorts.items():
        required = {"cvd_death_5y", "analysis_weight", "strata", "psu"}
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"{cohort_name} is missing survey-design fields: {sorted(missing)}")
        part = frame.loc[:, ["cvd_death_5y", "analysis_weight", "strata", "psu"]].copy()
        part.columns = ["outcome", "weight", "strata", "psu"]
        part.insert(0, "cohort", cohort_name)
        parts.append(part)
    design_input = pd.concat(parts, ignore_index=True)
    with tempfile.TemporaryDirectory(prefix="cvd-risk-r-gate-") as directory:
        input_path = Path(directory) / "survey_input.csv"
        output_path = Path(directory) / "survey_output.csv"
        design_input.to_csv(input_path, index=False)
        completed = subprocess.run(
            [
                rscript,
                "scripts/r/survey_design_gate.R",
                str(input_path),
                str(output_path),
                str(r2_fraction),
                str(parameters),
                str(shrinkage),
                str(max_ci_half_width),
                str(minimum_design_df),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            output = (completed.stdout or completed.stderr).strip()
            raise RuntimeError(f"R survey-design gate failed: {output}")
        metrics = pd.read_csv(output_path)
    observed = set(metrics["cohort"].astype(str))
    if observed != set(cohorts):
        raise ValueError(f"R survey-design gate returned unexpected cohorts: {sorted(observed)}")
    return metrics


def _nhanes_design_collision_audit(nhanes: pd.DataFrame) -> pd.DataFrame:
    """Audit published NHANES strata/PSU pairs without adding cycle identifiers."""

    required = {"survey_cycle", "strata", "psu"}
    missing = required.difference(nhanes.columns)
    if missing:
        raise KeyError(f"NHANES design collision audit is missing: {sorted(missing)}")
    pair_cycles = nhanes.groupby(["strata", "psu"], observed=True)["survey_cycle"].nunique()
    repeated_pairs = int(pair_cycles.gt(1).sum())
    return pd.DataFrame(
        [
            {
                "check": "published_strata_psu_pairs_repeated_across_cycles",
                "observed": repeated_pairs,
                "minimum": 0,
                "status": "pass" if repeated_pairs == 0 else "requires_protocol_amendment",
                "detail": (
                    "Published (SDMVSTRA, SDMVPSU) pairs are used as released; "
                    "cycle was not appended."
                ),
            },
            {
                "check": "automatic_cycle_interaction_forbidden",
                "observed": False,
                "minimum": False,
                "status": "pass",
                "detail": "No cycle interaction was added to strata or PSU identifiers.",
            },
        ]
    )


def _sealed_subgroup_counts(
    cohorts: dict[str, pd.DataFrame],
    *,
    subgroup_col: str,
) -> pd.DataFrame:
    """Return label-free held-out subgroup sizes without accessing outcomes."""

    rows: list[dict[str, object]] = []
    for cohort_name, frame in cohorts.items():
        if subgroup_col not in frame:
            continue
        known = frame[subgroup_col].notna()
        missing = int((~known).sum())
        for category, count in (
            frame.loc[known, subgroup_col].astype(str).value_counts().sort_index().items()
        ):
            rows.append(
                {
                    "cohort": cohort_name,
                    "subgroup": subgroup_col,
                    "category": category,
                    "status": "sealed_until_model_lock",
                    "n_participants": int(count),
                    "n_events": pd.NA,
                    "n_non_events": pd.NA,
                    "n_missing_subgroup": missing,
                    "weighted_event_prevalence": pd.NA,
                    "eligible_for_performance_audit": False,
                }
            )
    return pd.DataFrame(
        rows,
        columns=(
            "cohort",
            "subgroup",
            "category",
            "status",
            "n_participants",
            "n_events",
            "n_non_events",
            "n_missing_subgroup",
            "weighted_event_prevalence",
            "eligible_for_performance_audit",
        ),
    )


def _huq_hsd_disagreement_table(nhanes: pd.DataFrame) -> pd.DataFrame:
    """Cross-tab primary HUQ010 against sensitivity-only HSD010 without outcomes."""

    required = {
        "survey_cycle",
        "general_health",
        "general_health_hsd_sensitivity",
        "analysis_weight",
    }
    missing = required.difference(nhanes.columns)
    if missing:
        raise KeyError(f"NHANES HUQ/HSD audit is missing fields: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for cycle, frame in nhanes.groupby("survey_cycle", observed=True):
        cycle_name = str(cycle)
        if cycle_name == "1999-2000":
            rows.append(
                {
                    "survey_cycle": cycle_name,
                    "huq_category": pd.NA,
                    "hsd_category": pd.NA,
                    "n_paired": 0,
                    "weighted_n": 0.0,
                    "disagreement": pd.NA,
                    "status": "EXPECTED_NOT_AVAILABLE",
                }
            )
            continue
        paired = frame["general_health"].notna() & frame[
            "general_health_hsd_sensitivity"
        ].notna()
        paired_frame = frame.loc[paired]
        for (huq, hsd), group in paired_frame.groupby(
            ["general_health", "general_health_hsd_sensitivity"], observed=True
        ):
            rows.append(
                {
                    "survey_cycle": cycle_name,
                    "huq_category": str(huq),
                    "hsd_category": str(hsd),
                    "n_paired": len(group),
                    "weighted_n": float(group["analysis_weight"].sum()),
                    "disagreement": str(huq) != str(hsd),
                    "status": "AVAILABLE",
                }
            )
    return pd.DataFrame(rows)


def run_stage3_audit(
    *,
    nhis_cohort_path: str | Path,
    nhanes_cohort_path: str | Path,
    nhis_flow_path: str | Path,
    nhanes_flow_path: str | Path,
    protocol_discrepancies_path: str | Path,
    reports_dir: str | Path,
    effective_degrees_of_freedom: int = 20,
    thresholds: StopGateThresholds | None = None,
    config_path: str | Path | None = None,
    protocol_path: str | Path | None = None,
    feature_mapping_path: str | Path | None = None,
    harmonization_differences_path: str | Path | None = None,
    race_ethnicity_audit_enabled: bool = True,
    race_minimum_events: int = 20,
    race_minimum_non_events: int = 20,
    r2_fraction: float = 0.15,
    pmsampsize_parameters: int = 15,
    pmsampsize_shrinkage: float = 0.90,
    risk_ci_max_half_width: float = 0.0025,
    minimum_design_df: int = 8,
    overwrite: bool = False,
) -> StopGateResult:
    """Write every mandatory stage-3 table/figure and evaluate the stop-gate."""

    reports_root = Path(reports_dir)
    table_dir = reports_root / "tables"
    figure_dir = reports_root / "figures"
    expected_outputs = (
        table_dir / "cohort_flow_nhis.csv",
        table_dir / "cohort_flow_nhanes.csv",
        table_dir / "event_counts.csv",
        table_dir / "missingness_by_source.csv",
        table_dir / "feature_availability.csv",
        table_dir / "data_discrepancies.csv",
        table_dir / "harmonization_differences.csv",
        table_dir / "race_ethnicity_event_counts.csv",
        table_dir / "protocol_discrepancies.csv",
        table_dir / "baseline_cvd_sensitivity.csv",
        table_dir / "survey_design_precision.csv",
        table_dir / "effective_sample_size.csv",
        table_dir / "sample_size_gate.csv",
        table_dir / "nhanes_design_collision_audit.csv",
        table_dir / "huq_hsd_disagreement.csv",
        table_dir / "stop_gate.csv",
        figure_dir / "outcome_by_year_nhis_development.png",
        figure_dir / "age_distribution.png",
        reports_root / "feasibility_report.md",
        reports_root / "stop_gate.json",
    )
    existing_outputs = [path for path in expected_outputs if path.exists()]
    if existing_outputs and not overwrite:
        raise FileExistsError(
            "Cohort-audit outputs already exist and overwrite is disabled: "
            f"{[str(path) for path in existing_outputs]}. Use --force only after "
            "confirming the replacement is intentional."
        )

    protocol_discrepancies = load_protocol_discrepancies(protocol_discrepancies_path)
    nhis = read_table(nhis_cohort_path)
    nhanes = read_table(nhanes_cohort_path)
    _validate_cohort_contract(nhis, "nhis")
    _validate_cohort_contract(nhanes, "nhanes")
    codebook_binding = _validate_nhis_codebook_binding(nhis)
    if feature_mapping_path is not None:
        current_mapping_hash = _sha256(Path(feature_mapping_path))
        if current_mapping_hash != codebook_binding["feature_mapping_sha256"]:
            raise ValueError("Processed NHIS cohort was built with a different feature mapping")
    if "split" not in nhis:
        raise KeyError("NHIS cohort must contain the pre-specified 'split' column")
    if nhis["split"].isna().any():
        raise ValueError("NHIS cohort contains missing split assignments")
    unexpected_splits = set(nhis["split"].dropna().unique()).difference(
        {"development", "temporal_test"}
    )
    if unexpected_splits:
        raise ValueError(f"Unexpected NHIS splits: {sorted(unexpected_splits)}")
    expected_split = nhis["survey_year"].map(
        lambda year: (
            "development"
            if 1997 <= int(year) <= 2009
            else "temporal_test"
            if 2010 <= int(year) <= 2014
            else "out_of_protocol"
        )
    )
    wrong_split = nhis["split"].astype(str).ne(expected_split)
    if wrong_split.any():
        examples = nhis.loc[wrong_split, ["survey_year", "split"]].head(5)
        raise ValueError(
            "NHIS split does not match the fixed temporal protocol; examples: "
            f"{examples.to_dict(orient='records')}"
        )
    if not nhanes["survey_year"].between(1999, 2014, inclusive="both").all():
        raise ValueError("NHANES external cohort contains years outside 1999–2014")
    observed_nhis_years = set(pd.to_numeric(nhis["survey_year"], errors="raise"))
    if observed_nhis_years != set(EXPECTED_NHIS_YEARS):
        missing_years = sorted(set(EXPECTED_NHIS_YEARS).difference(observed_nhis_years))
        unexpected_years = sorted(observed_nhis_years.difference(EXPECTED_NHIS_YEARS))
        raise ValueError(
            "NHIS cohort does not cover every protocol year; "
            f"missing={missing_years}, unexpected={unexpected_years}"
        )

    cohorts = {
        "nhis_development": nhis.loc[nhis["split"] == "development"].copy(),
        "nhis_temporal_test": nhis.loc[nhis["split"] == "temporal_test"].copy(),
        "nhanes_external": nhanes.copy(),
    }
    empty = [name for name, frame in cohorts.items() if frame.empty]
    if empty:
        raise ValueError(f"Required cohorts are empty: {empty}")

    weight_columns = {name: "analysis_weight" for name in cohorts}
    counts = event_counts_table(cohorts, weight_columns=weight_columns)
    missingness_parts = [missingness_table(cohorts), missingness_reason_table(cohorts)]
    nonempty_missingness_parts = [part for part in missingness_parts if not part.empty]
    missingness = pd.concat(nonempty_missingness_parts, ignore_index=True)
    if "survey_cycle" not in nhanes:
        raise KeyError("NHANES cohort must contain survey_cycle for availability audit")
    observed_cycles = set(nhanes["survey_cycle"].dropna().astype(str))
    if observed_cycles != set(EXPECTED_NHANES_CYCLES):
        missing_cycles = sorted(set(EXPECTED_NHANES_CYCLES).difference(observed_cycles))
        unexpected_cycles = sorted(observed_cycles.difference(EXPECTED_NHANES_CYCLES))
        raise ValueError(
            "NHANES cohort does not cover all eight protocol cycles; "
            f"missing={missing_cycles}, unexpected={unexpected_cycles}"
        )
    audited_features = (*FAST_FEATURES, "race_ethnicity")
    availability = pd.concat(
        [
            feature_availability_by_period_table(
                nhis,
                source="nhis",
                period_col="survey_year",
                features=audited_features,
                expected_periods=EXPECTED_NHIS_YEARS,
            ),
            feature_availability_by_period_table(
                nhanes,
                source="nhanes",
                period_col="survey_cycle",
                features=audited_features,
                expected_periods=EXPECTED_NHANES_CYCLES,
            ),
        ],
        ignore_index=True,
    )
    development_race_counts = subgroup_event_counts_table(
        {"nhis_development": cohorts["nhis_development"]},
        subgroup_col="race_ethnicity",
        minimum_events=race_minimum_events,
        minimum_non_events=race_minimum_non_events,
    )
    heldout_race_counts = _sealed_subgroup_counts(
        {
            "nhis_temporal_test": cohorts["nhis_temporal_test"],
            "nhanes_external": cohorts["nhanes_external"],
        },
        subgroup_col="race_ethnicity",
    )
    for table in (development_race_counts, heldout_race_counts):
        for column in ("n_events", "n_non_events"):
            table[column] = pd.to_numeric(table[column], errors="coerce").astype("Int64")
        table["weighted_event_prevalence"] = pd.to_numeric(
            table["weighted_event_prevalence"], errors="coerce"
        ).astype("Float64")
    race_counts = pd.concat(
        [development_race_counts, heldout_race_counts],
        ignore_index=True,
    )
    baseline_cvd_sensitivity = _baseline_cvd_sensitivity_table(cohorts)
    survey_design = _run_r_survey_design_gate(
        cohorts,
        r2_fraction=r2_fraction,
        parameters=pmsampsize_parameters,
        shrinkage=pmsampsize_shrinkage,
        max_ci_half_width=risk_ci_max_half_width,
        minimum_design_df=minimum_design_df,
    )
    effective_sample_size = survey_design.loc[
        :, ["cohort", "n_nominal", "deff_y", "deff_y_gate", "n_eff_design", "n_eff_kish"]
    ].copy()
    sample_size_gate = survey_design.loc[
        survey_design["cohort"].eq("nhis_development"),
        [
            "cohort",
            "expected_r2_cs",
            "pmsampsize_required_n",
            "required_n_nominal",
            "n_eff_design",
            "pass_information",
        ],
    ].copy()
    collision_audit = _nhanes_design_collision_audit(nhanes)
    huq_hsd_disagreement = _huq_hsd_disagreement_table(nhanes)
    race_ready_by_cohort = {
        cohort_name: int(group.loc[group["eligible_for_performance_audit"], "category"].nunique())
        >= 2
        for cohort_name, group in race_counts.groupby("cohort", observed=True)
    }
    discrepancy_rows: list[dict[str, object]] = []
    if race_ethnicity_audit_enabled:
        for cohort_name in ("nhis_development",):
            if race_ready_by_cohort.get(cohort_name, False):
                continue
            discrepancy_rows.append(
                {
                    "source": cohort_name,
                    "severity": "warning",
                    "code": "race_ethnicity_audit_unavailable",
                    "blocks_core_stop_gate": False,
                    "detail": (
                        "Race/ethnicity is audit-only and not ready for reliable "
                        "subgroup analysis; it remains excluded from predictors."
                    ),
                }
            )
    discrepancies = pd.DataFrame(
        discrepancy_rows,
        columns=(
            "source",
            "severity",
            "code",
            "blocks_core_stop_gate",
            "detail",
        ),
    )
    harmonization_differences = (
        read_table(harmonization_differences_path)
        if harmonization_differences_path is not None
        else harmonization_differences_table()
    )
    required_difference_columns = {
        "feature",
        "nhis_measurement",
        "nhanes_measurement",
        "comparability",
        "required_action",
    }
    missing_difference_columns = required_difference_columns.difference(
        harmonization_differences.columns
    )
    if missing_difference_columns:
        raise KeyError(
            f"Harmonization differences table is incomplete: {sorted(missing_difference_columns)}"
        )
    nhis_flow = _load_flow(nhis_flow_path, "NHIS")
    nhanes_flow = _load_flow(nhanes_flow_path, "NHANES")
    if int(nhis_flow.iloc[-1]["n_participants"]) != len(nhis):
        raise ValueError("NHIS flow final count does not match the processed cohort")
    if int(nhanes_flow.iloc[-1]["n_participants"]) != len(nhanes):
        raise ValueError("NHANES flow final count does not match the processed cohort")

    _atomic_csv(nhis_flow, table_dir / "cohort_flow_nhis.csv")
    _atomic_csv(nhanes_flow, table_dir / "cohort_flow_nhanes.csv")
    _atomic_csv(counts, table_dir / "event_counts.csv")
    _atomic_csv(missingness, table_dir / "missingness_by_source.csv")
    _atomic_csv(availability, table_dir / "feature_availability.csv")
    _atomic_csv(discrepancies, table_dir / "data_discrepancies.csv")
    _atomic_csv(
        harmonization_differences,
        table_dir / "harmonization_differences.csv",
    )
    _atomic_csv(race_counts, table_dir / "race_ethnicity_event_counts.csv")
    _atomic_csv(
        protocol_discrepancies,
        table_dir / "protocol_discrepancies.csv",
    )
    _atomic_csv(
        baseline_cvd_sensitivity,
        table_dir / "baseline_cvd_sensitivity.csv",
    )
    _atomic_csv(survey_design, table_dir / "survey_design_precision.csv")
    _atomic_csv(effective_sample_size, table_dir / "effective_sample_size.csv")
    _atomic_csv(sample_size_gate, table_dir / "sample_size_gate.csv")
    _atomic_csv(collision_audit, table_dir / "nhanes_design_collision_audit.csv")
    _atomic_csv(huq_hsd_disagreement, table_dir / "huq_hsd_disagreement.csv")

    plot_outcome_by_year(
        cohorts["nhis_development"],
        figure_dir / "outcome_by_year_nhis_development.png",
    )
    plot_age_distribution(cohorts, figure_dir / "age_distribution.png")

    statistical_result = evaluate_stop_gate(
        counts,
        effective_degrees_of_freedom=effective_degrees_of_freedom,
        thresholds=thresholds,
    )
    discrepancy_checks = protocol_discrepancy_checks(protocol_discrepancies)
    design_check_rows: list[dict[str, object]] = []
    for row in survey_design.to_dict(orient="records"):
        cohort_name = str(row["cohort"])
        design_check_rows.extend(
            [
                {
                    "check": f"risk_precision:{cohort_name}",
                    "observed": row["max_ci_half_width"],
                    "minimum": risk_ci_max_half_width,
                    "status": "pass" if row["pass_risk_precision"] else "fail",
                },
                {
                    "check": f"design_df:{cohort_name}",
                    "observed": row["design_df"],
                    "minimum": minimum_design_df,
                    "status": "pass" if row["pass_design_df"] else "fail",
                },
            ]
        )
        if cohort_name == "nhis_development":
            design_check_rows.append(
                {
                    "check": "pmsampsize_information:nhis_development",
                    "observed": row["n_eff_design"],
                    "minimum": row["pmsampsize_required_n"],
                    "status": "pass" if row["pass_information"] else "fail",
                }
            )
    design_checks = pd.DataFrame(design_check_rows)
    collision_checks = collision_audit.loc[:, ["check", "observed", "minimum", "status"]]
    stop_gate_details = pd.concat(
        [statistical_result.details, design_checks, collision_checks, discrepancy_checks],
        ignore_index=True,
    )
    ready_for_modeling = bool((stop_gate_details["status"] == "pass").all())
    if ready_for_modeling:
        summary = "PASS: all pre-specified feasibility checks passed."
    else:
        unresolved = ", ".join(
            stop_gate_details.loc[stop_gate_details["status"] != "pass", "check"]
        )
        summary = f"STOP: modeling is not authorized; unresolved checks: {unresolved}."
    result = StopGateResult(
        details=stop_gate_details,
        ready_for_modeling=ready_for_modeling,
        summary=summary,
    )
    _atomic_csv(result.details, table_dir / "stop_gate.csv")
    status: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "ready_for_modeling": result.ready_for_modeling,
        "summary": result.summary,
        "effective_degrees_of_freedom": effective_degrees_of_freedom,
        "policy": (
            "Temporal/external labels were accessed only for the pre-specified "
            "feasibility audit; they remain prohibited for model selection."
        ),
        "nhis_codebook_binding": codebook_binding,
        "race_ethnicity_audit_enabled": race_ethnicity_audit_enabled,
        "race_ethnicity_audit_ready": (
            not race_ethnicity_audit_enabled
            or race_ready_by_cohort.get("nhis_development", False)
        ),
        "heldout_race_ethnicity_labels_sealed": True,
        "protocol_discrepancies_ready": bool((discrepancy_checks["status"] == "pass").all()),
        "unresolved_blocking_protocol_discrepancies": protocol_discrepancies.loc[
            (protocol_discrepancies["blocking"]) & (protocol_discrepancies["status"] != "resolved"),
            "id",
        ].tolist(),
        "inputs": {
            "nhis_cohort": _provenance_entry(nhis_cohort_path),
            "nhanes_cohort": _provenance_entry(nhanes_cohort_path),
            "nhis_flow": _provenance_entry(nhis_flow_path),
            "nhanes_flow": _provenance_entry(nhanes_flow_path),
            "protocol_discrepancies": _provenance_entry(protocol_discrepancies_path),
        },
    }
    if config_path is not None:
        status["inputs"]["experiment_config"] = _provenance_entry(config_path)
    if protocol_path is not None:
        status["inputs"]["research_protocol"] = _provenance_entry(protocol_path)
    if feature_mapping_path is not None:
        status["inputs"]["feature_mapping"] = _provenance_entry(feature_mapping_path)
    if harmonization_differences_path is not None:
        status["inputs"]["harmonization_differences_input"] = _provenance_entry(
            harmonization_differences_path
        )
    status_path = reports_root / "stop_gate.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)

    report_path = reports_root / "feasibility_report.md"
    lines = [
        "# Stage 3 feasibility report",
        "",
        f"Generated: {status['generated_at']}",
        "",
        f"**{result.summary}**",
        "",
        "## Event counts",
        "",
        counts.to_markdown(index=False),
        "",
        "## Stop-gate checks",
        "",
        result.details.to_markdown(index=False),
        "",
        "## Data discrepancies and conditional audits",
        "",
        (
            discrepancies.to_markdown(index=False)
            if not discrepancies.empty
            else "No stage-3 discrepancies were detected."
        ),
        "",
        "## NHIS–NHANES harmonization differences",
        "",
        harmonization_differences.to_markdown(index=False),
        "",
        "## Blocking protocol discrepancies",
        "",
        protocol_discrepancies.to_markdown(index=False),
        "",
        "## Race/ethnicity audit feasibility",
        "",
        race_counts.to_markdown(index=False),
        "",
        "## CHF-strict baseline-CVD sensitivity subsets",
        "",
        baseline_cvd_sensitivity.to_markdown(index=False),
        "",
        "## R survey-design precision and effective sample size",
        "",
        survey_design.to_markdown(index=False),
        "",
        "## NHANES design collision audit",
        "",
        collision_audit.to_markdown(index=False),
        "",
        "## Label-free HUQ010/HSD010 disagreement audit",
        "",
        huq_hsd_disagreement.to_markdown(index=False),
        "",
        "Temporal and external outcomes were inspected only for the "
        "pre-specified feasibility audit. They were not used for model or "
        "threshold selection.",
        "",
        "No age range, prediction horizon, outcome definition, or train/test "
        "boundary was changed. The explicitly approved CHF sensitivity amendment "
        "is recorded in the protocol discrepancy registry.",
    ]
    report_tmp = report_path.with_suffix(".md.tmp")
    report_tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report_tmp.replace(report_path)

    output_paths = {
        "cohort_flow_nhis": table_dir / "cohort_flow_nhis.csv",
        "cohort_flow_nhanes": table_dir / "cohort_flow_nhanes.csv",
        "event_counts": table_dir / "event_counts.csv",
        "missingness_by_source": table_dir / "missingness_by_source.csv",
        "feature_availability": table_dir / "feature_availability.csv",
        "data_discrepancies": table_dir / "data_discrepancies.csv",
        "harmonization_differences": table_dir / "harmonization_differences.csv",
        "race_ethnicity_event_counts": table_dir / "race_ethnicity_event_counts.csv",
        "protocol_discrepancies_table": table_dir / "protocol_discrepancies.csv",
        "baseline_cvd_sensitivity": table_dir / "baseline_cvd_sensitivity.csv",
        "survey_design_precision": table_dir / "survey_design_precision.csv",
        "effective_sample_size": table_dir / "effective_sample_size.csv",
        "sample_size_gate": table_dir / "sample_size_gate.csv",
        "nhanes_design_collision_audit": table_dir / "nhanes_design_collision_audit.csv",
        "huq_hsd_disagreement": table_dir / "huq_hsd_disagreement.csv",
        "stop_gate_table": table_dir / "stop_gate.csv",
        "outcome_by_year_nhis_development": (
            figure_dir / "outcome_by_year_nhis_development.png"
        ),
        "age_distribution": figure_dir / "age_distribution.png",
        "feasibility_report": report_path,
    }
    status["outputs"] = {name: _provenance_entry(path) for name, path in output_paths.items()}
    temporary = status_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    temporary.replace(status_path)
    return result
