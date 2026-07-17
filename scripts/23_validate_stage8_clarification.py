"""Validate Stage 8 history evidence and the pre-fitting clarification artifact."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvd_risk.modeling.stage8_clarification import (  # noqa: E402
    validate_stage8_clarification,
)


def main() -> None:
    result = validate_stage8_clarification(
        "artifacts/stage8_clinical_preflight/protocol_clarification.json",
        root=ROOT,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
