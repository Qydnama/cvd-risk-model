from __future__ import annotations

import argparse

from cvd_risk.modeling.lock_amendment import create_lock_amendment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create metadata-only external-validation lock amendment 001."
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = create_lock_amendment(force=args.force)
    print(
        f"{result['amendment_id']}: {result['status']}; "
        "held-out outcomes accessed=False."
    )


if __name__ == "__main__":
    main()
