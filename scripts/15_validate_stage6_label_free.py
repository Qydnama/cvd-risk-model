from __future__ import annotations

from cvd_risk.modeling.stage6 import validate_stage6_label_free


def main() -> None:
    result = validate_stage6_label_free()
    print(
        f"{result['status']}: cohorts={result['cohort_count']}; "
        f"pipeline_sha256={result['pipeline_sha256']}; "
        "held-out outcomes accessed=False; held-out outcomes unsealed=False"
    )


if __name__ == "__main__":
    main()
