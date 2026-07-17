"""Validate the minimal_v1 model card against locked evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvd_risk.modeling.minimal_v1_model_card import (  # noqa: E402
    validate_minimal_v1_model_card,
)


def main() -> None:
    result = validate_minimal_v1_model_card(
        "artifacts/minimal_v1_model_card/status.json",
        root=ROOT,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
