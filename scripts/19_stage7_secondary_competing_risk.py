"""Run Stage 7 secondary censored/competing-risk analysis."""

from __future__ import annotations

import argparse

from cvd_risk.modeling.stage7 import run_stage7_secondary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage7_secondary.yaml")
    args = parser.parse_args()
    manifest = run_stage7_secondary(args.config)
    print(
        f"{manifest['status']}: primary_model_modified=False; "
        "preprocessing_modified=False; calibrator_modified=False; "
        "cutoff_modified_or_reselected=False"
    )


if __name__ == "__main__":
    main()
