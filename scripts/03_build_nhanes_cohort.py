"""Build the protocol-locked NHANES external-validation cohort."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import pandas as pd

from cvd_risk.data.cohort_nhanes import build_nhanes_cohort_with_flow
from cvd_risk.data.harmonize_nhanes import (
    NHANES_CYCLE_START_YEAR,
    harmonize_nhanes,
)
from cvd_risk.data.load_nhanes import load_nhanes_cycles
from cvd_risk.provenance import reject_quarantined_input
from cvd_risk.reporting.harmonization import write_harmonization_differences
from cvd_risk.runtime import DEFAULT_CONFIG_PATH, initialize_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--nhanes-config", type=Path, default=Path("configs/nhanes_files.yaml"))
    parser.add_argument("--mapping", type=Path, default=Path("configs/feature_mapping.yaml"))
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--flow-output", type=Path)
    parser.add_argument("--differences-output", type=Path)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse a complete cohort only when its configuration bindings still match",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _refuse_overwrite(paths: tuple[Path, ...], *, force: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not force:
        raise FileExistsError(f"NHANES outputs already exist and overwrite is disabled: {existing}")


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_reusable_outputs(
    *,
    output: Path,
    flow_output: Path,
    differences_output: Path,
    config: Path,
    nhanes_config: Path,
    mapping: Path,
) -> int:
    expected = (output, flow_output, differences_output)
    present = tuple(path.is_file() for path in expected)
    if not all(present):
        missing = [str(path) for path, exists in zip(expected, present, strict=True) if not exists]
        raise FileNotFoundError(f"Cannot reuse a partial NHANES build; missing outputs: {missing}")

    cohort = pd.read_parquet(output)
    required_bindings = {
        "experiment_config_sha256": _sha256(config),
        "nhanes_files_config_sha256": _sha256(nhanes_config),
        "feature_mapping_sha256": _sha256(mapping),
    }
    for column, expected_hash in required_bindings.items():
        if column not in cohort or set(cohort[column].dropna().astype(str)) != {expected_hash}:
            raise ValueError(
                f"Existing NHANES cohort is not bound to the current {column}; "
                "rerun with --force after reviewing the configuration change"
            )
    if cohort.empty or cohort["participant_id"].isna().any():
        raise ValueError("Existing NHANES cohort is empty or has missing participant IDs")
    if cohort["participant_id"].duplicated().any():
        raise ValueError("Existing NHANES cohort contains duplicate participant IDs")
    if set(cohort["survey_cycle"].astype(str)) != set(NHANES_CYCLE_START_YEAR):
        raise ValueError("Existing NHANES cohort does not contain all protocol cycles")
    if set(cohort["split"].astype(str)) != {"external_validation"}:
        raise ValueError("Existing NHANES cohort has an invalid split assignment")

    flow = pd.read_csv(flow_output)
    if flow.empty or int(flow.iloc[-1]["n_participants"]) != len(cohort):
        raise ValueError("Existing NHANES flow does not match the cohort row count")
    differences = pd.read_csv(differences_output)
    required_difference_columns = {
        "feature",
        "nhis_measurement",
        "nhanes_measurement",
        "comparability",
        "required_action",
    }
    if not required_difference_columns.issubset(differences.columns):
        raise ValueError("Existing harmonization-differences table is incomplete")
    return len(cohort)


def main() -> int:
    args = parse_args()
    config = initialize_runtime(args.config)
    paths = config.get("paths", {})
    raw_dir = args.raw_dir or Path(paths.get("nhanes_raw_dir", "data/raw/nhanes"))
    processed_dir = Path(paths.get("processed_dir", "data/processed"))
    interim_dir = Path(paths.get("interim_dir", "data/interim"))
    output = args.output or processed_dir / "nhanes_cohort_v2_4.parquet"
    flow_output = args.flow_output or interim_dir / "nhanes_cohort_flow.csv"
    differences_output = args.differences_output or interim_dir / "harmonization_differences.csv"
    required_configs = (args.config, args.nhanes_config, args.mapping)
    missing_configs = [str(path) for path in required_configs if not path.is_file()]
    if missing_configs:
        raise FileNotFoundError(f"Missing configuration files: {missing_configs}")

    output_paths = (output, flow_output, differences_output)
    if args.reuse_existing and any(path.exists() for path in output_paths) and not args.force:
        reject_quarantined_input(
            output,
            Path(paths.get("quarantine_manifest", "artifacts/audit/quarantine_manifest.json")),
        )
        participant_count = _validate_reusable_outputs(
            output=output,
            flow_output=flow_output,
            differences_output=differences_output,
            config=args.config,
            nhanes_config=args.nhanes_config,
            mapping=args.mapping,
        )
        print(f"NHANES cohort: reused {participant_count:,} validated participants -> {output}")
        return 0
    _refuse_overwrite(output_paths, force=args.force)

    cycles = tuple(NHANES_CYCLE_START_YEAR)
    raw = load_nhanes_cycles(
        cycles,
        raw_dir=raw_dir,
        config_path=args.nhanes_config,
        profile="all",
        include_mortality=True,
    )
    harmonized = harmonize_nhanes(raw, include_extended=True, strict=True)
    cohort, flow = build_nhanes_cohort_with_flow(harmonized)

    retained_cycles = set(cohort["survey_cycle"].astype(str))
    if retained_cycles != set(cycles):
        missing = sorted(set(cycles).difference(retained_cycles))
        raise ValueError(
            "Final NHANES cohort does not retain participants in every protocol cycle; "
            f"missing={missing}"
        )
    cohort["experiment_config_sha256"] = _sha256(args.config)
    cohort["nhanes_files_config_sha256"] = _sha256(args.nhanes_config)
    cohort["feature_mapping_sha256"] = _sha256(args.mapping)

    _atomic_parquet(cohort, output)
    _atomic_csv(flow, flow_output)
    write_harmonization_differences(differences_output)

    print(f"NHANES cohort: {len(cohort):,} participants -> {output}")
    print(f"NHANES flow: {flow_output}")
    print(f"Harmonization differences: {differences_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
