from __future__ import annotations

from cvd_risk.modeling.unseal_validation import validate_unseal_plan


def main() -> None:
    result = validate_unseal_plan()
    print(
        f"{result['status']}: stage6_commit={result['stage6_commit']}; "
        "held-out outcomes loaded=False; outcomes unsealed=False"
    )


if __name__ == "__main__":
    main()
