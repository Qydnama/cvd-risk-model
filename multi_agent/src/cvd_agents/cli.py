"""Command line interface.

    cvd-agents assess --age 58 --sex male ...   single-shot validation + prediction (Assignment 2 path)
    cvd-agents chat [--llm env|simulated]       multi-agent dialogue
    cvd-agents scenarios [--mode offline|live]  run the scenario suite and write a summary
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import DEFAULT_LIMITS, load_dotenv, model_spec
from .prototype import OrchestratorAgent

DISCLAIMER = "Research and educational use only. Not a diagnosis or treatment recommendation."


def _utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def cmd_assess(args: argparse.Namespace) -> int:
    state = OrchestratorAgent().assess({
        "age": args.age, "sex": args.sex, "bmi": args.bmi, "smoking_status": args.smoking_status,
        "hypertension_dx": args.hypertension, "diabetes_dx": args.diabetes, "general_health": args.general_health,
    })
    print(json.dumps(state.model_dump(), indent=2))
    print(DISCLAIMER)
    return 0 if state.status == "completed" else 1


def cmd_chat(args: argparse.Namespace) -> int:
    from .graph import CVDAssistant
    from .llm import build_chat_model
    from .offline_llm import simulated_llm

    if args.llm == "simulated":
        llm, label = simulated_llm(), "SIMULATED rule-based stand-in (not a language model)"
    else:
        llm, label = build_chat_model(DEFAULT_LIMITS), f"live LLM {model_spec()}"
    assistant = CVDAssistant(llm)
    print(f"[{label}] session {assistant.session.session_id}. {DISCLAIMER} Type 'exit' to quit.")
    while True:
        try:
            message = input("\nВы: ").strip()
        except EOFError:
            break
        if message.lower() in {"exit", "quit", "выход"}:
            break
        if not message:
            continue
        result = assistant.handle(message)
        print(f"\nАссистент: {result.reply}")
        if args.debug:
            print(f"  [intent={result.intent} status={result.status} phase={result.phase} agents={result.agents_called}]")
    print(f"Logs: {assistant.recorder.path}")
    return 0


def cmd_scenarios(args: argparse.Namespace) -> int:
    from .scenario_runner import run_all

    summary = run_all(args.mode, only=args.only or None)
    for result in summary["results"]:
        mark = "PASS" if result["passed"] else "FAIL"
        print(f"{mark} {result['id']:<28} [{result['llm']}] {'; '.join(result['failures'])}")
    load = summary["typical_scenario_load"]
    print(f"\n{summary['passed']}/{summary['total']} passed; mode={summary['mode']}; model={summary['model']}")
    if load:
        print(f"Typical scenario: {load['total_llm_and_tool_calls']} LLM+tool calls, max agent share "
              f"{load['max_share']:.1%}, 40% rule {'passed' if load['passes_40_percent_rule'] else 'FAILED'}")
    return 0 if summary["passed"] == summary["total"] else 1


def main(argv: list[str] | None = None) -> int:
    _utf8()
    load_dotenv()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0].startswith("--") and argv[0] not in ("-h", "--help"):
        argv.insert(0, "assess")  # backward compatibility with the Assignment 2 command
    parser = argparse.ArgumentParser(description="Research-only multi-agent interface to minimal_v1")
    sub = parser.add_subparsers(dest="command", required=True)

    assess = sub.add_parser("assess", help="validate seven inputs and run the locked model")
    assess.add_argument("--age", type=float, required=True)
    assess.add_argument("--sex", required=True)
    assess.add_argument("--bmi", type=float, required=True)
    assess.add_argument("--smoking-status", required=True)
    assess.add_argument("--hypertension", action=argparse.BooleanOptionalAction, required=True)
    assess.add_argument("--diabetes", action=argparse.BooleanOptionalAction, required=True)
    assess.add_argument("--general-health", required=True)
    assess.set_defaults(func=cmd_assess)

    chat = sub.add_parser("chat", help="interactive multi-agent dialogue")
    chat.add_argument("--llm", choices=["env", "simulated"], default="env")
    chat.add_argument("--debug", action="store_true")
    chat.set_defaults(func=cmd_chat)

    scenarios = sub.add_parser("scenarios", help="run the scenario suite")
    scenarios.add_argument("--mode", choices=["offline", "live"], default="offline")
    scenarios.add_argument("--only", nargs="*")
    scenarios.set_defaults(func=cmd_scenarios)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
