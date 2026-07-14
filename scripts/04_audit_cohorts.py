"""Generate the mandatory stage-3 cohort audit and stop-gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cvd_risk.reporting.audit import run_stage3_audit
from cvd_risk.reporting.tables import StopGateThresholds
from cvd_risk.runtime import DEFAULT_CONFIG_PATH, initialize_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--feature-mapping",
        type=Path,
        default=Path("configs/feature_mapping.yaml"),
    )
    parser.add_argument(
        "--nhis-cohort",
        type=Path,
    )
    parser.add_argument(
        "--nhanes-cohort",
        type=Path,
    )
    parser.add_argument(
        "--nhis-flow",
        type=Path,
    )
    parser.add_argument(
        "--nhanes-flow",
        type=Path,
    )
    parser.add_argument("--harmonization-differences", type=Path)
    parser.add_argument("--protocol-discrepancies", type=Path)
    parser.add_argument("--reports-dir", type=Path)
    parser.add_argument("--effective-df", type=int)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Intentionally replace an existing audit after validating new inputs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = initialize_runtime(args.config)
    configured_df = config.get("stop_gate", {}).get("preliminary_logistic_effective_df", 20)
    effective_df = args.effective_df if args.effective_df is not None else configured_df
    if isinstance(effective_df, bool) or not isinstance(effective_df, int):
        raise ValueError("stop_gate.preliminary_logistic_effective_df must be an integer")
    stop_gate_config = config.get("stop_gate", {})
    minimum_events = stop_gate_config.get("minimum_events", {})
    thresholds = StopGateThresholds(
        nhis_development_events=int(
            minimum_events.get("nhis_minimal_development", 500)
        ),
        nhis_temporal_test_events=int(minimum_events.get("nhis_temporal_test", 200)),
        nhanes_external_events=int(minimum_events.get("nhanes_external_validation", 100)),
        events_per_effective_df=int(stop_gate_config.get("minimum_events_per_effective_df", 20)),
    )
    paths = config.get("paths", {})
    processed_dir = Path(paths.get("processed_dir", "data/processed"))
    interim_dir = Path(paths.get("interim_dir", "data/interim"))
    configured_table_dir = Path(paths.get("report_tables_dir", "reports/tables"))
    nhis_cohort = args.nhis_cohort or processed_dir / "nhis_cohort.parquet"
    nhanes_cohort = args.nhanes_cohort or processed_dir / "nhanes_cohort_v2_4.parquet"
    nhis_flow = args.nhis_flow or interim_dir / "nhis_cohort_flow.csv"
    nhanes_flow = args.nhanes_flow or interim_dir / "nhanes_cohort_flow.csv"
    harmonization_differences = (
        args.harmonization_differences or interim_dir / "harmonization_differences.csv"
    )
    protocol_discrepancies = args.protocol_discrepancies or Path(
        paths.get(
            "protocol_discrepancies",
            "configs/protocol_discrepancies.yaml",
        )
    )
    reports_dir = args.reports_dir or configured_table_dir.parent
    race_config = config.get("audit", {}).get("race_ethnicity", {})
    try:
        result = run_stage3_audit(
            nhis_cohort_path=nhis_cohort,
            nhanes_cohort_path=nhanes_cohort,
            nhis_flow_path=nhis_flow,
            nhanes_flow_path=nhanes_flow,
            protocol_discrepancies_path=protocol_discrepancies,
            reports_dir=reports_dir,
            effective_degrees_of_freedom=effective_df,
            thresholds=thresholds,
            config_path=args.config,
            protocol_path=Path("CODEX_CVD_MODEL_IMPLEMENTATION_PLAN.md"),
            feature_mapping_path=args.feature_mapping,
            harmonization_differences_path=harmonization_differences,
            race_ethnicity_audit_enabled=bool(race_config.get("enabled", True)),
            race_minimum_events=int(race_config.get("minimum_events_per_group", 20)),
            race_minimum_non_events=int(race_config.get("minimum_non_events_per_group", 20)),
            overwrite=args.force,
        )
    except FileNotFoundError as exc:
        print(f"STOP: stage-3 audit input is missing: {exc}")
        return 2
    print(result.summary)
    return 0 if result.ready_for_modeling else 2


if __name__ == "__main__":
    sys.exit(main())
