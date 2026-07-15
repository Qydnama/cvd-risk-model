from __future__ import annotations

import argparse

from cvd_risk.modeling.model_lock import validate_external_validation_lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the committed model lock.")
    parser.add_argument(
        "--lock", default="artifacts/minimal_v1/external_validation_lock.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_external_validation_lock(lock_path=args.lock)
    print(
        f"{result['status']}: locked source {result['locked_source_commit']} "
        f"is an ancestor of {result['current_head']}; all SHA bindings valid."
    )


if __name__ == "__main__":
    main()
