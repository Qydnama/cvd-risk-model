"""Orchestration for the mandatory stage-3 cohort audit."""

from __future__ import annotations

import hashlib
import json
import re
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
        table_dir / "stop_gate.csv",
        figure_dir / "outcome_by_year.png",
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
    race_counts = subgroup_event_counts_table(
        cohorts,
        subgroup_col="race_ethnicity",
        minimum_events=race_minimum_events,
        minimum_non_events=race_minimum_non_events,
    )
    race_ready_by_cohort = {
        cohort_name: int(group.loc[group["eligible_for_performance_audit"], "category"].nunique())
        >= 2
        for cohort_name, group in race_counts.groupby("cohort", observed=True)
    }
    discrepancy_rows: list[dict[str, object]] = []
    if race_ethnicity_audit_enabled:
        for cohort_name in cohorts:
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

    if "survey_year" in nhis and "survey_year" in nhanes:
        plot_frame = pd.concat(
            [
                nhis.assign(source="nhis"),
                nhanes.assign(source="nhanes"),
            ],
            ignore_index=True,
        )
        plot_outcome_by_year(
            plot_frame,
            figure_dir / "outcome_by_year.png",
            group_col="source",
        )
    else:
        missing_year = [
            name
            for name, frame in (("nhis", nhis), ("nhanes", nhanes))
            if "survey_year" not in frame
        ]
        raise KeyError(f"Missing survey_year in cohorts: {missing_year}")
    plot_age_distribution(cohorts, figure_dir / "age_distribution.png")

    statistical_result = evaluate_stop_gate(
        counts,
        effective_degrees_of_freedom=effective_degrees_of_freedom,
        thresholds=thresholds,
    )
    discrepancy_checks = protocol_discrepancy_checks(protocol_discrepancies)
    stop_gate_details = pd.concat(
        [statistical_result.details, discrepancy_checks], ignore_index=True
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
            not race_ethnicity_audit_enabled or all(race_ready_by_cohort.values())
        ),
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
        "Temporal and external outcomes were inspected only for the "
        "pre-specified feasibility audit. They were not used for model or "
        "threshold selection.",
        "",
        "No age range, prediction horizon, outcome definition, baseline-CVD "
        "exclusion, or train/test boundary was changed.",
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
        "stop_gate_table": table_dir / "stop_gate.csv",
        "outcome_by_year": figure_dir / "outcome_by_year.png",
        "age_distribution": figure_dir / "age_distribution.png",
        "feasibility_report": report_path,
    }
    status["outputs"] = {name: _provenance_entry(path) for name, path in output_paths.items()}
    temporary = status_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    temporary.replace(status_path)
    return result
