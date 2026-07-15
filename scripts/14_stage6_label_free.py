from __future__ import annotations

import argparse

from cvd_risk.modeling.stage6 import run_stage6_label_free


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Stage 6 label-free predictions")
    parser.add_argument("--config", default="configs/stage6_label_free.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = run_stage6_label_free(config_path=args.config, force=args.force)
    print(
        f"{manifest['status']}: cohorts={len(manifest['cohorts'])}; "
        "held-out outcomes accessed=False; held-out outcomes unsealed=False"
    )


if __name__ == "__main__":
    main()
