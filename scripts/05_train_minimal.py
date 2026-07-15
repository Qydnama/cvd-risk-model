"""Run development-only EDA and nested-CV logistic baseline training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cvd_risk.modeling.training import run_stage4_baselines
from cvd_risk.runtime import DEFAULT_CONFIG_PATH, initialize_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--modeling-config", type=Path, default=Path("configs/modeling.yaml"))
    parser.add_argument("--stop-gate", type=Path, default=Path("reports/stop_gate.json"))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace only the known Stage 4 outputs if they already exist",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    initialize_runtime(args.config)
    try:
        manifest = run_stage4_baselines(
            experiment_config_path=args.config,
            modeling_config_path=args.modeling_config,
            stop_gate_path=args.stop_gate,
            force=args.force,
        )
    except (FileNotFoundError, FileExistsError, KeyError, RuntimeError, ValueError) as exc:
        print(f"STOP: {exc}")
        return 2
    print(
        "Stage 4 baselines fitted on NHIS development only: "
        f"N={manifest['n']:,}, events={manifest['events']:,}. "
        "No final model lock was created."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
