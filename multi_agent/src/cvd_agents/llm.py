"""LLM construction, bounded retries and deadlines.

Provider SDK retries are disabled (``max_retries=0``) so that every attempt goes
through :func:`call_with_retries` and is visible in the run log.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel

from .config import Limits, model_spec
from .telemetry import RunRecorder

T = TypeVar("T")

TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}
TRANSIENT_NAMES = {
    "TimeoutError",
    "APITimeoutError",
    "APIConnectionError",
    "RateLimitError",
    "InternalServerError",
    "ServiceUnavailableError",
    "OverloadedError",
    "ConnectError",
    "ReadTimeout",
    "ConnectTimeout",
    "TransientLLMError",
}


class ConfigurationError(RuntimeError):
    pass


class LLMUnavailableError(RuntimeError):
    """Raised after the bounded retries for a transient provider failure are exhausted."""


class TurnTimeoutError(RuntimeError):
    pass


class StepLimitError(RuntimeError):
    pass


class TransientLLMError(RuntimeError):
    """Used by the offline simulator to emulate timeouts / 429 / 5xx."""


def is_transient(exc: BaseException) -> bool:
    name = type(exc).__name__
    # langchain-openai wraps SDK timeouts as OpenAITimeoutError (found in the first live run).
    if name in TRANSIENT_NAMES or name.endswith(("TimeoutError", "Timeout")):
        return True
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    return isinstance(status, int) and status in TRANSIENT_STATUS


def build_chat_model(limits: Limits, spec: str | None = None) -> BaseChatModel:
    from langchain.chat_models import init_chat_model

    selected = spec or model_spec()
    if not selected:
        raise ConfigurationError(
            "AGENT_MODEL is not set. Use e.g. AGENT_MODEL=openai:<model> with OPENAI_API_KEY, "
            "or run with --llm simulated for the offline simulator."
        )
    return init_chat_model(selected, timeout=limits.llm_timeout_seconds, max_retries=0)


def call_with_retries(
    fn: Callable[[], T],
    *,
    recorder: RunRecorder,
    agent: str,
    limits: Limits,
    deadline: float,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    for attempt in range(1, limits.llm_max_attempts + 1):
        if time.monotonic() > deadline:
            raise TurnTimeoutError("Turn time limit exceeded")
        try:
            return fn()
        except Exception as exc:
            if not is_transient(exc):
                raise
            recorder.log(agent=agent, kind="retry", name="llm_retry", status="error",
                         attempt=attempt, error_type=type(exc).__name__)
            if attempt == limits.llm_max_attempts:
                raise LLMUnavailableError(f"LLM unavailable after {attempt} attempts: {type(exc).__name__}") from exc
            sleep(limits.retry_base_delay_seconds * (2 ** (attempt - 1)))
    raise AssertionError("unreachable")
