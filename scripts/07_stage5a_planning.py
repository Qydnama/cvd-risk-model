"""Run provisional pmvalsampsize planning from selected development OOF risk."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cvd_risk.modeling.stage5a import run_stage5a_planning
from cvd_risk.runtime import DEFAULT_CONFIG_PATH, initialize_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--planning-config",
        type=Path,
        default=Path("configs/stage5a_planning.yaml"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    initialize_runtime(args.config)
    try:
        manifest = run_stage5a_planning(config_path=args.planning_config, force=args.force)
    except (FileNotFoundError, FileExistsError, KeyError, RuntimeError, ValueError) as exc:
        print(f"STOP: {exc}")
        return 2
    required_n = manifest["stability"]["provisional_required_n"]
    print(
        "Stage 5a provisional planning complete: "
        f"IID/SRS-equivalent required N={required_n:,}. No model lock was created."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
