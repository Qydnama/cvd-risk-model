"""Command-line research interface."""

from __future__ import annotations

import argparse
import json
import sys

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from .config import REPO_ROOT
from .deep import build_agent_graph, run_agent_assessment
from .tools import get_model_evidence, load_input_schema, verify_model_artifacts
from .workflow import OrchestratorAgent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Research-only minimal_v1 multi-agent system")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("health")
    assess = commands.add_parser("assess")
    assess.add_argument("--age", type=float, required=True)
    assess.add_argument("--sex", required=True)
    assess.add_argument("--bmi", type=float, required=True)
    assess.add_argument("--smoking-status", required=True)
    assess.add_argument("--hypertension", action=argparse.BooleanOptionalAction, required=True)
    assess.add_argument("--diabetes", action=argparse.BooleanOptionalAction, required=True)
    assess.add_argument("--general-health", required=True)
    assess.add_argument("--agent-mode", action="store_true", help="Also run Deep Agents with configured LLM provider")
    commands.add_parser("chat")
    commands.add_parser("evidence")
    args = parser.parse_args(argv)
    if args.command == "health":
        try:
            result = verify_model_artifacts()
            load_input_schema()
            get_model_evidence()
            build_agent_graph(FakeListChatModel(responses=["configuration check"]))
            print(json.dumps({"status": "ok", "model_id": result["model_id"], "verified_artifacts": result["locked_binary_artifacts_verified"], "research_only": True}))
            return 0
        except Exception as exc:
            print(f"Health check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    if args.command == "evidence":
        print(get_model_evidence().model_dump_json(indent=2))
        return 0
    if args.command == "assess":
        raw = {"age": args.age, "sex": args.sex, "bmi": args.bmi, "smoking_status": args.smoking_status, "hypertension_dx": args.hypertension, "diabetes_dx": args.diabetes, "general_health": args.general_health}
        state = OrchestratorAgent().assess(raw)
        result = {"run_id": state.run_id, "status": state.status, "errors": state.errors, "response": state.safety_review.final_response if state.safety_review else None}
        if args.agent_mode and state.status == "completed":
            try:
                result["deep_agents"] = run_agent_assessment(state)
            except Exception as exc:
                result["status"] = "agent_error"
                result["errors"] = [f"{type(exc).__name__}: {exc}"]
                result["response"] = None
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "completed" else 1
    if args.command == "chat":
        print("Research-only interactive mode. Enter a seven-field JSON object, or 'quit'.")
        while True:
            try:
                line = input("cvd> ").strip()
            except EOFError:
                return 0
            if line.lower() in {"quit", "exit"}:
                return 0
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("Expected a JSON object")
                state = OrchestratorAgent().assess(raw)
                print(state.safety_review.final_response if state.safety_review and state.safety_review.passed else "; ".join(state.errors))
            except (ValueError, json.JSONDecodeError) as exc:
                print(f"Invalid input: {exc}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
