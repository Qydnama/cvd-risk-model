"""Tune and compare Random Forest and XGBoost on NHIS development only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cvd_risk.modeling.stage5 import run_stage5_ml_candidates
from cvd_risk.runtime import DEFAULT_CONFIG_PATH, initialize_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--stage5-config", type=Path, default=Path("configs/stage5_ml.yaml"))
    parser.add_argument(
        "--stage4-manifest",
        type=Path,
        default=Path("artifacts/stage4_baselines/manifest.json"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace only the known Stage 5 candidate outputs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    initialize_runtime(args.config)
    try:
        manifest = run_stage5_ml_candidates(
            stage5_config_path=args.stage5_config,
            stage4_manifest_path=args.stage4_manifest,
            force=args.force,
        )
    except (FileNotFoundError, FileExistsError, KeyError, RuntimeError, ValueError) as exc:
        print(f"STOP: {exc}")
        return 2
    print(
        "Stage 5 ML candidates fitted on NHIS development only: "
        f"N={manifest['n']:,}, events={manifest['events']:,}. "
        "No final model lock was created."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
