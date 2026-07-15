from __future__ import annotations

import argparse

from cvd_risk.modeling.operating_point import run_operating_point_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the fixed secondary screening operating point before lock."
    )
    parser.add_argument("--config", default="configs/operating_point.yaml")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_operating_point_audit(config_path=args.config, force=args.force)
    policy = result["secondary_operating_point"]
    print(
        "Stage 5b pre-lock audit complete: "
        f"cutoff={policy['exact_cutoff']:.12f}, "
        f"weighted sensitivity={policy['achieved_survey_weighted_sensitivity']:.6f}. "
        "No held-out outcome accessed; no model lock created."
    )


if __name__ == "__main__":
    main()
