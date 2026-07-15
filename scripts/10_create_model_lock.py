from __future__ import annotations

import argparse

from cvd_risk.modeling.model_lock import create_external_validation_lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the immutable external-validation model lock."
    )
    parser.add_argument("--config", default="configs/model_lock.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = create_external_validation_lock(config_path=args.config)
    print(
        f"Model lock created at commit {result['locked_source_commit']}: "
        f"{result['status']}."
    )


if __name__ == "__main__":
    main()
