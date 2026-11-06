"""Explicit LangGraph workflow for one dialogue turn and the session facade.

The order of agents is fixed by graph edges, not by an LLM: input validation always
precedes prediction, a confirmed input always precedes prediction, and every
explanation always passes the safety review before it reaches the user.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Callable, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from .agents import (
    EvidenceAgent,
    ExplanationAgent,
    InputValidationAgent,
    OrchestratorAgent,
    RiskPredictionAgent,
    SafetyReviewAgent,
    TurnContext,
)
from .config import DEFAULT_LIMITS, Limits, runs_dir
from .contracts import (
    ChatMessage,
    EvidenceBundle,
    ExplanationDraft,
    RoutingDecision,
    SafetyReview,
    SessionState,
    TurnResult,
)
from .llm import ConfigurationError, LLMUnavailableError, StepLimitError, TurnTimeoutError
from .telemetry import RunRecorder, TelemetryCallback, calculate_agent_load

DISCLAIMER = "Только для исследований и обучения. Не диагноз и не рекомендация лечения; модель не валидирована для Казахстана."
QUESTION_KIND = {"question_model": "model", "question_cvd_general": "medical", "question_result": "result"}


class TurnState(TypedDict, total=False):
    message: str
    decision: RoutingDecision
    question: str
    question_kind: str
    bundle: EvidenceBundle
    draft: ExplanationDraft
    review: SafetyReview
    revisions: int
    reply: str
    status: str
    error: str


class CVDAssistant:
    def __init__(
        self,
        llm: BaseChatModel,
        *,
        limits: Limits = DEFAULT_LIMITS,
        session_id: str | None = None,
        runs_path: Path | None = None,
        persist: bool = True,
    ):
        self.limits = limits
        self.session = SessionState(session_id=session_id or uuid.uuid4().hex)
        self.recorder = RunRecorder(self.session.session_id, runs_path or runs_dir(), persist=persist)
        self.callback = TelemetryCallback(self.recorder)
        self.orchestrator = OrchestratorAgent(llm)
        self.input_agent = InputValidationAgent(llm)
        self.prediction_agent = RiskPredictionAgent(None)
        self.evidence_agent = EvidenceAgent(llm)
        self.explanation_agent = ExplanationAgent(llm)
        self.safety_agent = SafetyReviewAgent(llm)
        self.agents = [self.orchestrator, self.input_agent, self.prediction_agent,
                       self.evidence_agent, self.explanation_agent, self.safety_agent]
        self.ctx: TurnContext | None = None
        self.graph = self._build_graph()

    # --- graph ----------------------------------------------------------------------------

    def _guard(self, fn: Callable[[TurnState], dict[str, Any]]) -> Callable[[TurnState], dict[str, Any]]:
        def node(state: TurnState) -> dict[str, Any]:
            try:
                return fn(state)
            except (StepLimitError, TurnTimeoutError) as exc:
                return {"status": "limit_exceeded", "error": f"{type(exc).__name__}: {exc}",
                        "reply": "Обработка остановлена: превышен лимит шагов или времени. Попробуйте переформулировать запрос."}
            except LLMUnavailableError as exc:
                return {"status": "error", "error": str(exc),
                        "reply": "Языковая модель временно недоступна после нескольких попыток. Прогноз не изменён; повторите позже."}
            except ConfigurationError as exc:
                return {"status": "error", "error": str(exc), "reply": "Система не настроена: " + str(exc)}
            except Exception as exc:  # noqa: BLE001 - every failure becomes an explicit terminal state
                return {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                        "reply": "Внутренняя ошибка обработки. Вероятность не рассчитывалась и не выдумывалась."}
        node.__name__ = fn.__name__
        return node

    def _build_graph(self):
        g = StateGraph(TurnState)
        for name, fn in {
            "route": self._route, "collect": self._collect, "predict": self._predict,
            "evidence": self._evidence, "explain": self._explain, "review": self._review,
            "refuse": self._refuse, "reset": self._reset, "finalize": self._finalize,
        }.items():
            g.add_node(name, fn if name == "finalize" else self._guard(fn))
        g.add_edge(START, "route")
        g.add_conditional_edges("route", self._after_route)
        g.add_edge("collect", "finalize")
        g.add_conditional_edges("predict", lambda s: "finalize" if s.get("status") else "evidence")
        g.add_conditional_edges("evidence", lambda s: "finalize" if s.get("status") else "explain")
        g.add_conditional_edges("explain", lambda s: "finalize" if s.get("status") else "review")
        g.add_conditional_edges("review", self._after_review)
        g.add_edge("refuse", "finalize")
        g.add_edge("reset", "finalize")
        g.add_edge("finalize", END)
        return g.compile(name="cvd_turn_graph")

    def _after_route(self, state: TurnState) -> str:
        if state.get("status"):
            return "finalize"
        intent = state["decision"].intent
        if intent in ("calculate", "provide_features", "change_features"):
            return "collect"
        if intent == "confirm":
            return "predict" if self.session.phase == "awaiting_confirmation" else "collect"
        if intent in QUESTION_KIND:
            if intent == "question_result" and not self.session.prediction_is_current():
                return "finalize"
            return "evidence"
        return {"reset": "reset", "out_of_scope": "refuse"}.get(intent, "finalize")

    def _after_review(self, state: TurnState) -> str:
        if state.get("status"):
            return "finalize"
        if state["review"].verdict == "approved":
            return "finalize"
        return "explain" if state.get("revisions", 0) < self.limits.safety_max_revisions else "finalize"

    # --- nodes (each delegates to exactly one agent) ----------------------------------------

    def _route(self, state: TurnState) -> dict[str, Any]:
        decision = self.orchestrator.route(self.session, state["message"], self.ctx)
        update: dict[str, Any] = {"decision": decision}
        if decision.intent in QUESTION_KIND:
            update.update(question=state["message"], question_kind=QUESTION_KIND[decision.intent])
        if decision.intent == "question_result" and not self.session.prediction_is_current():
            update.update(status="needs_input", reply="Актуального результата расчёта нет. Напишите «хочу оценить риск» и укажите семь признаков.")
        if decision.intent == "reject":
            self.session.phase = "collecting"
            update.update(status="needs_input", reply="Хорошо. Укажите, какое значение нужно исправить.")
        return update

    def _collect(self, state: TurnState) -> dict[str, Any]:
        outcome = self.input_agent.run(self.session, state["message"], self.ctx)
        status = "awaiting_confirmation" if outcome.ready_for_confirmation else "needs_input"
        return {"status": status, "reply": outcome.reply}

    def _predict(self, state: TurnState) -> dict[str, Any]:
        self.prediction_agent.run(self.session, self.ctx)
        return {"question": "Что означает полученный результат модели?", "question_kind": "result"}

    def _evidence(self, state: TurnState) -> dict[str, Any]:
        return {"bundle": self.evidence_agent.run(state["question"], state["question_kind"], self.session, self.ctx)}

    def _explain(self, state: TurnState) -> dict[str, Any]:
        issues = state["review"].issues if state.get("review") else None
        draft = self.explanation_agent.run(state["bundle"], self.session, self.ctx, issues)
        return {"draft": draft, "revisions": state.get("revisions", -1) + 1}

    def _review(self, state: TurnState) -> dict[str, Any]:
        return {"review": self.safety_agent.review(state["draft"], state["bundle"], self.session, self.ctx, state["revisions"])}

    def _refuse(self, state: TurnState) -> dict[str, Any]:
        return {"status": "refused", "reply": self.safety_agent.refusal(self.ctx)}

    def _reset(self, state: TurnState) -> dict[str, Any]:
        sid, turn, history = self.session.session_id, self.session.turn, self.session.history
        self.session = SessionState(session_id=sid, turn=turn, history=history)
        return {"status": "answered", "reply": "Данные и результат сброшены. Можно задать вопрос или начать новый расчёт."}

    def _finalize(self, state: TurnState) -> dict[str, Any]:
        if state.get("status"):
            return {}
        review = state.get("review")
        if review is not None and review.verdict == "approved":
            text = state["draft"].text
            if state.get("question_kind") in ("result", "medical"):
                text = f"{text}\n\n{DISCLAIMER}"
            return {"status": "answered", "reply": text}
        if review is not None:
            blocked = review.model_copy(update={"verdict": "blocked"})
            return {"status": "answered", "review": blocked,
                    "reply": SafetyReviewAgent.fallback(state["bundle"], self.session) + f"\n\n{DISCLAIMER}"}
        return {"status": "error", "reply": "Не удалось обработать запрос.", "error": "no terminal state"}

    # --- public API ---------------------------------------------------------------------------

    def handle(self, message: str) -> TurnResult:
        self.session.turn += 1
        self.recorder.turn = self.session.turn
        self.ctx = TurnContext(self.recorder, self.callback, self.limits,
                               deadline=time.monotonic() + self.limits.turn_timeout_seconds)
        started = time.perf_counter()
        try:
            final: TurnState = self.graph.invoke({"message": message},
                                                 config={"recursion_limit": self.limits.graph_recursion_limit})
        except Exception as exc:  # noqa: BLE001 - graph-level failure (e.g. recursion limit)
            final = {"status": "limit_exceeded" if "Recursion" in type(exc).__name__ else "error",
                     "error": f"{type(exc).__name__}: {exc}", "reply": "Обработка остановлена из-за внутреннего лимита."}
        decision = final.get("decision")
        bundle = final.get("bundle")
        result = TurnResult(
            session_id=self.session.session_id,
            turn=self.session.turn,
            intent=decision.intent if decision else None,
            status=final.get("status", "error"),
            reply=final.get("reply", ""),
            phase=self.session.phase,
            prediction=self.session.prediction if self.session.prediction_is_current() else None,
            safety=final.get("review"),
            evidence_fact_ids=[f.fact_id for f in bundle.facts] if bundle else [],
            agents_called=list(self.ctx.agents_called),
            error=final.get("error"),
        )
        self.recorder.log(agent="session", kind="turn", name=result.intent or "none", status=result.status,
                          duration_ms=(time.perf_counter() - started) * 1000, steps=self.ctx.steps)
        self.session.history += [ChatMessage(role="user", text=message), ChatMessage(role="assistant", text=result.reply)]
        self.session.history = self.session.history[-2 * self.limits.history_window:]
        self.recorder.save_json("state.json", self.session.sanitized())
        self.recorder.save_json("load_report.json", calculate_agent_load(self.recorder.events))
        return result
