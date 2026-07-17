"""Validate the pre-fitting Stage 8 primary-cohort correction addendum."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvd_risk.modeling.stage8_correction_addendum import (  # noqa: E402
    validate_stage8_correction_addendum,
)


def main() -> None:
    result = validate_stage8_correction_addendum(
        "artifacts/stage8_exploratory_correction/addendum.json",
        root=ROOT,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
