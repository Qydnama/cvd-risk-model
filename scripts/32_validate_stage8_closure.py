"""Validate the Stage 8 closure decision against immutable evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvd_risk.modeling.stage8_closure import validate_stage8_closure  # noqa: E402


def main() -> None:
    result = validate_stage8_closure(
        "artifacts/stage8_closure/status.json",
        root=ROOT,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
