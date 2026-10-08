"""Small CLI demonstrating the Assignment 2 two-agent data exchange."""

from __future__ import annotations

import argparse
import json

from .prototype import OrchestratorAgent


def main() -> int:
    parser = argparse.ArgumentParser(description="Research-only CVD agent prototype")
    parser.add_argument("--age", type=float, required=True)
    parser.add_argument("--sex", required=True)
    parser.add_argument("--bmi", type=float, required=True)
    parser.add_argument("--smoking-status", required=True)
    parser.add_argument("--hypertension", action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument("--diabetes", action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument("--general-health", required=True)
    args = parser.parse_args()
    state = OrchestratorAgent().assess(
        {
            "age": args.age,
            "sex": args.sex,
            "bmi": args.bmi,
            "smoking_status": args.smoking_status,
            "hypertension_dx": args.hypertension,
            "diabetes_dx": args.diabetes,
            "general_health": args.general_health,
        }
    )
    print(json.dumps(state.model_dump(), indent=2))
    print("Research and educational use only. Not a diagnosis or treatment recommendation.")
    return 0 if state.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
