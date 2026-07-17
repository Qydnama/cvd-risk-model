"""Validate corrected Stage 8 paired OOF outputs and all prohibitions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvd_risk.modeling.stage8_exploratory_corrected import (  # noqa: E402
    validate_corrected_stage8_status,
)


def main() -> None:
    result = validate_corrected_stage8_status(
        "artifacts/stage8_exploratory_corrected/status.json",
        root=ROOT,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
