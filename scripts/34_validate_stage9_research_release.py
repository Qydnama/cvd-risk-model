"""Validate the Stage 9 minimal_v1 research-only release candidate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvd_risk.modeling.stage9_research_release import (  # noqa: E402
    validate_stage9_research_release,
)


def main() -> None:
    result = validate_stage9_research_release(root=ROOT)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
