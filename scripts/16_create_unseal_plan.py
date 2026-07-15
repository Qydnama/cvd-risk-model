from __future__ import annotations

import argparse

from cvd_risk.modeling.unseal_validation import create_unseal_plan_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the pre-registered Stage 6 unseal validation plan"
    )
    parser.add_argument("--config", default="configs/stage6_unseal_validation.yaml")
    parser.add_argument(
        "--output",
        default="artifacts/stage6_unseal_plan/unseal_plan_manifest.json",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = create_unseal_plan_manifest(
        config_path=args.config,
        output_path=args.output,
        force=args.force,
    )
    print(
        f"{manifest['status']}: outcomes loaded=False; outcomes unsealed=False; "
        f"next={manifest['next_action']}"
    )


if __name__ == "__main__":
    main()
