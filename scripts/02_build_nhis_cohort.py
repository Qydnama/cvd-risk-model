"""Build the protocol-locked NHIS development and temporal cohorts."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from cvd_risk.data.cohort import build_nhis_cohort_with_flow
from cvd_risk.data.harmonize import harmonize_nhis, verify_codebook_binding
from cvd_risk.data.load_nhis import discover_nhis_input, load_nhis
from cvd_risk.data.manifest import register_local_input
from cvd_risk.runtime import DEFAULT_CONFIG_PATH, initialize_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mapping", type=Path, default=Path("configs/feature_mapping.yaml"))
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--flow-output", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _refuse_overwrite(paths: tuple[Path, ...], *, force: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not force:
        raise FileExistsError(f"NHIS outputs already exist and overwrite is disabled: {existing}")


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


def main() -> int:
    args = parse_args()
    config = initialize_runtime(args.config)
    paths = config.get("paths", {})
    raw_dir = args.raw_dir or Path(paths.get("nhis_raw_dir", "data/raw/nhis"))
    output = args.output or Path(paths.get("processed_dir", "data/processed")) / (
        "nhis_cohort.parquet"
    )
    flow_output = (
        args.flow_output or Path(paths.get("interim_dir", "data/interim")) / "nhis_cohort_flow.csv"
    )
    _refuse_overwrite((output, flow_output), force=args.force)

    inputs = discover_nhis_input(raw_dir)
    raw = load_nhis(raw_dir, require_all_years=True)
    verify_codebook_binding(
        inputs.data_path,
        inputs.codebook_path,
        mapping_path=args.mapping,
    )
    harmonized = harmonize_nhis(raw, mapping_path=args.mapping, strict=True)
    cohort, flow = build_nhis_cohort_with_flow(harmonized)
    if set(cohort["survey_year"].astype(int)) != set(range(1997, 2015)):
        raise ValueError("Final NHIS cohort does not retain participants in every protocol year")
    _atomic_parquet(cohort, output)
    _atomic_csv(flow, flow_output)

    manifest_path = Path(paths.get("data_manifest", "configs/data_manifest.yaml"))
    register_local_input(
        manifest_path,
        inputs.data_path,
        dataset="nhis",
        component="ipums_extract",
        period="1997-2018",
        source_url="https://nhis.ipums.org/nhis/",
    )
    register_local_input(
        manifest_path,
        inputs.codebook_path,
        dataset="nhis",
        component="ipums_codebook",
        period="1997-2018",
        source_url="https://nhis.ipums.org/nhis/",
    )
    print(f"NHIS cohort: {len(cohort):,} participants -> {output}")
    print(f"NHIS flow: {flow_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
