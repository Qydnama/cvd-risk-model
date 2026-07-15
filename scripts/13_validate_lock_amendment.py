from __future__ import annotations

from cvd_risk.modeling.lock_amendment import validate_committed_lock_amendment


def main() -> None:
    result = validate_committed_lock_amendment()
    print(
        f"{result['status']}: portable_bindings={result['portable_binding_count']}; "
        "locked model/data/cutoff unchanged; held-out outcomes accessed=False."
    )


if __name__ == "__main__":
    main()
