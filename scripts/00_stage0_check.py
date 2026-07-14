"""Run the protocol-2.4 Stage-0 hard gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cvd_risk.runtime import DEFAULT_CONFIG_PATH
from cvd_risk.stage0 import run_stage0_check


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    status = run_stage0_check(args.config, output_path=args.output)
    print(f"Stage 0: {status['status']}")
    return 0 if status["ready_for_stage1"] else 2


if __name__ == "__main__":
    sys.exit(main())
