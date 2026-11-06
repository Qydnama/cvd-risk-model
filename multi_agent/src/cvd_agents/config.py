"""Paths, model selection and execution limits. No secrets live here."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]  # multi_agent/
REPO_ROOT = PACKAGE_ROOT.parent
KNOWLEDGE_DIR = PACKAGE_ROOT / "knowledge"
SCENARIOS_PATH = PACKAGE_ROOT / "scenarios" / "scenarios.json"


def runs_dir() -> Path:
    return Path(os.getenv("CVD_RUNS_DIR", str(PACKAGE_ROOT / "runs")))


def load_dotenv(path: Path | None = None) -> list[str]:
    """Load KEY=VALUE pairs from multi_agent/.env without overriding the real environment.

    Returns the variable names that were set (never the values).
    """
    env_path = path or PACKAGE_ROOT / ".env"
    loaded: list[str] = []
    if not env_path.is_file():
        return loaded
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def model_spec() -> str | None:
    """Provider-qualified model name, e.g. ``openai:<model>`` or ``anthropic:<model>``.

    API keys are read by the provider SDK from its own environment variable
    (OPENAI_API_KEY, ANTHROPIC_API_KEY); they are never passed through code.
    """
    value = os.getenv("AGENT_MODEL", "").strip()
    return value or None


@dataclass(frozen=True)
class Limits:
    max_agent_steps_per_turn: int = 10
    # create_agent uses 2 graph steps per tool round: 3 tool rounds + final answer = 7, plus 1 spare.
    specialist_recursion_limit: int = 8
    graph_recursion_limit: int = 25
    llm_timeout_seconds: float = 30.0
    turn_timeout_seconds: float = 90.0
    llm_max_attempts: int = 3  # first attempt + 2 retries for transient failures
    retry_base_delay_seconds: float = 1.0
    safety_max_revisions: int = 1
    max_invalid_attempts_per_field: int = 3
    history_window: int = 8


DEFAULT_LIMITS = Limits()
