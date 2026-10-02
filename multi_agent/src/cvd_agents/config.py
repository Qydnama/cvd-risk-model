"""Fixed safety limits and repository location."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_DIR.parent
RUNS_DIR = PROJECT_DIR / "runs"
MAX_AGENT_STEPS = 12
MAX_REVIEW_REVISIONS = 1
TOTAL_RUN_TIMEOUT_SECONDS = 60
TOOL_TIMEOUT_SECONDS = 20
