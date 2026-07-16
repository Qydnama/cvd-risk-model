"""Validate the frozen Stage 8 full-gate failure and absence of fitted models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvd_risk.modeling.stage8_gate import validate_stage8_gate_status  # noqa: E402


def main() -> None:
    result = validate_stage8_gate_status(
        "artifacts/stage8_clinical_gate/status.json",
        root=ROOT,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
