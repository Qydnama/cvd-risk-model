"""Run the Stage 8 clinical-route feasibility preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvd_risk.modeling.stage8_preflight import run_stage8_preflight  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage8_clinical_preflight.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    status = run_stage8_preflight(args.config, root=ROOT, force=args.force)
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
