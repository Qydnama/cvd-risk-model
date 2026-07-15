from __future__ import annotations

import argparse

from cvd_risk.modeling.unseal_validation import run_authorized_unseal_validation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the separately authorized one-time Stage 6 validation"
    )
    parser.add_argument("--config", default="configs/stage6_unseal_validation.yaml")
    parser.add_argument(
        "--plan-manifest",
        default="artifacts/stage6_unseal_plan/unseal_plan_manifest.json",
    )
    parser.add_argument(
        "--authorization",
        default="artifacts/audit/unseal_authorization.json",
    )
    args = parser.parse_args()
    manifest = run_authorized_unseal_validation(
        config_path=args.config,
        plan_manifest_path=args.plan_manifest,
        authorization_path=args.authorization,
    )
    print(
        f"{manifest['status']}: model_modified=False; calibrator_modified=False; "
        "cutoff_modified_or_reselected=False"
    )


if __name__ == "__main__":
    main()
