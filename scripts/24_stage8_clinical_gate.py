"""Run the full Stage 8 clinical_core pre-fitting gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvd_risk.modeling.stage8_gate import run_stage8_clinical_gate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage8_clinical_gate.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = run_stage8_clinical_gate(args.config, root=ROOT, force=args.force)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
