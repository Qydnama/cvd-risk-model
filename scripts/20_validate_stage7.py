"""Validate Stage 7 aggregates and immutable model bindings."""

from __future__ import annotations

import argparse

from cvd_risk.modeling.stage7 import validate_stage7_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage7_secondary.yaml")
    args = parser.parse_args()
    result = validate_stage7_outputs(args.config)
    print(
        f"STAGE7_VALID: status={result['status']}; cohorts={len(result['cohorts'])}; "
        f"outputs={result['output_count']}; pipeline_sha256={result['pipeline_sha256']}"
    )


if __name__ == "__main__":
    main()
