from __future__ import annotations

import argparse

from cvd_risk.modeling.model_lock import export_serialized_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serialize the selected full-development pipeline without refitting."
    )
    parser.add_argument("--config", default="configs/model_lock.yaml")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_serialized_pipeline(config_path=args.config, force=args.force)
    print(
        f"Final pipeline {result['status']}; refit={result['refit_performed']}; "
        f"heldout outcomes accessed={result['heldout_outcomes_accessed']}."
    )


if __name__ == "__main__":
    main()
