"""Deterministic rule-based stand-in for an LLM (NOT a language model).

Used for offline tests and demos without an API key. It plugs into the same
LangChain interfaces (structured output via tool calls, tool-calling loop), so the
graph, limits, retries and logging are exercised exactly as with a real provider.
Results obtained with it say nothing about the quality of a real LLM.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

Responder = Callable[[list[BaseMessage], list[str]], AIMessage]


class ScriptedChatModel(BaseChatModel):
    responder: Responder
    tool_names: list[str] = Field(default_factory=list)
    faults: list[Exception] = Field(default_factory=list)  # shared across bound copies
    stats: dict[str, int] = Field(default_factory=lambda: {"calls": 0})

    @property
    def _llm_type(self) -> str:
        return "simulated-rule-based"

    def bind_tools(self, tools: Sequence[Any], *, tool_choice: Any = None, **kwargs: Any) -> "ScriptedChatModel":
        names = [convert_to_openai_tool(t)["function"]["name"] for t in tools]
        return self.model_copy(update={"tool_names": names})

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self.stats["calls"] += 1
        if self.faults:
            raise self.faults.pop(0)
        return ChatResult(generations=[ChatGeneration(message=self.responder(messages, self.tool_names))])


def _call(name: str, args: dict[str, Any], index: int = 0) -> dict[str, Any]:
    return {"name": name, "args": args, "id": f"call_{name}_{index}", "type": "tool_call"}


def _payload(messages: list[BaseMessage]) -> dict[str, Any]:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            try:
                return json.loads(message.content)
            except (TypeError, json.JSONDecodeError):
                return {"user_message": str(message.content)}
    return {}


# --- rules ----------------------------------------------------------------------------------


def classify(payload: dict[str, Any]) -> str:
    text = payload.get("user_message", "").lower()
    session = payload.get("session", {})
    has_result = session.get("has_current_result") or session.get("has_stale_result")
    if re.search(r"сброс|начать заново|начнём заново|reset", text):
        return "reset"
    if re.search(r"лечени|таблет|препарат|лекарств|какой у меня диагноз|поставь диагноз|погод|анекдот|футбол|рецепт", text):
        return "out_of_scope"
    has_values = bool(extract(text))
    if has_result and has_values and re.search(r"измен|теперь|на самом деле|исправ|бросил|стал", text):
        return "change_features"
    if re.search(r"рассчит|оцени|посчитай|расч[её]т|хочу узнать риск|calculate", text):
        return "calculate"
    if has_values:
        return "provide_features"
    if re.search(r"мой результат|мой риск", text) or (has_result and re.search(r"результат|процент|получил|означает", text)):
        return "question_result"
    if re.search(r"модел|обучен|данных|метрик|точност|казахстан|валид|ограничен|порог", text):
        return "question_model"
    if re.search(r"сердечн|ссз|инфаркт|инсульт|фактор|симптом|cvd|давлени", text):
        return "question_cvd_general"
    return "out_of_scope"


FIELD_PATTERNS: list[tuple[str, str, Callable[[re.Match[str]], str]]] = [
    ("age", r"(\d{2,3})\s*(?:лет|года|год)\b", lambda m: m.group(1)),
    ("age", r"возраст\D{0,5}(\d{1,3})", lambda m: m.group(1)),
    ("sex", r"\b(мужчина|мужской|мужчин)\b", lambda m: "male"),
    ("sex", r"\b(женщина|женский|женщин)\b", lambda m: "female"),
    ("bmi", r"(?:имт|bmi)\D{0,6}(\d{1,3}(?:[.,]\d+)?)", lambda m: m.group(1)),
    ("smoking_status", r"(никогда не курил\w*|не курю)", lambda m: "never"),
    ("smoking_status", r"(бросил\w* курить|бывший курильщик|раньше курил\w*)", lambda m: "former"),
    ("smoking_status", r"(курю(?! ли)|курильщик)", lambda m: "current"),
    ("hypertension_dx", r"((?:нет|без) гипертони\w*)", lambda m: "no"),
    ("hypertension_dx", r"(гипертони\w* есть|есть гипертони\w*|гипертони\w*: да)", lambda m: "yes"),
    ("diabetes_dx", r"((?:нет|без) диабет\w*)", lambda m: "no"),
    ("diabetes_dx", r"(диабет\w* есть|есть диабет\w*|диабет: да)", lambda m: "yes"),
    ("general_health", r"(здоровье (?:отличное|очень хорошее|хорошее|удовлетворительное|плохое))", None),
]
HEALTH = {"отличное": "excellent", "очень хорошее": "very_good", "хорошее": "good", "удовлетворительное": "fair", "плохое": "poor"}


def extract(text: str) -> list[dict[str, str]]:
    found: dict[str, dict[str, str]] = {}
    lowered = text.lower()
    for field, pattern, value_of in FIELD_PATTERNS:
        if field in found:
            continue
        match = re.search(pattern, lowered)
        if not match:
            continue
        if field == "general_health":
            value = HEALTH[match.group(1).split(" ", 1)[1]]
        else:
            value = value_of(match)
        quote = text[match.start(): match.end()]
        found[field] = {"field": field, "value": value, "quote": quote}
    return list(found.values())


def evidence_calls(payload: dict[str, Any]) -> list[dict[str, Any]]:
    question = payload.get("question", "")
    kind = payload.get("question_kind")
    lowered = question.lower()
    if kind == "result":
        return [_call("get_model_card_fact", {"topic": "outcome"}, 0), _call("get_model_card_fact", {"topic": "intended_use"}, 1)]
    if kind == "medical":
        return [_call("search_medical_sources", {"query": question})]
    calls = []
    if "казахстан" in lowered:
        calls.append(_call("get_model_card_fact", {"topic": "population"}, 0))
    elif re.search(r"данн|обуч", lowered):
        calls.append(_call("get_model_card_fact", {"topic": "development"}, 0))
    elif re.search(r"метрик|точност|валид", lowered):
        calls.append(_call("get_model_card_fact", {"topic": "validation"}, 0))
    calls.append(_call("search_model_documents", {"query": question}, 1))
    return calls


def explanation(payload: dict[str, Any]) -> dict[str, Any]:
    facts = payload.get("facts", [])
    kind = payload.get("question_kind")
    question = payload.get("question", "")
    if not facts:
        return {"text": "Подтверждённого источника по этому вопросу не найдено.", "used_fact_ids": []}
    if kind == "result":
        ids = [f["id"] for f in facts]
        text = ("Модель minimal_v1 оценила вероятность смерти от сердечно-сосудистых причин в течение 5 лет "
                f"в {{probability_percent}} [{ids[0]}]. Это исследовательская оценка, а не диагноз; модель "
                f"построена на данных США и не валидирована для Казахстана [{ids[-1]}].")
        injected = re.search(r"(\d+(?:[.,]\d+)?)\s*%", question)
        if injected and "safety_issues" not in payload:
            # Emulates an LLM that naively follows an instruction embedded in the question.
            text += f" По вашей просьбе: риск составляет {injected.group(1)}%."
        return {"text": text, "used_fact_ids": ids}
    picked = facts[:2]
    body = " ".join(f"{f['text'][:220]} [{f['id']}]" for f in picked)
    return {"text": f"По найденным источникам: {body}", "used_fact_ids": [f["id"] for f in picked]}


def simulated_responder(messages: list[BaseMessage], tool_names: list[str]) -> AIMessage:
    payload = _payload(messages)
    if "RoutingDecision" in tool_names:
        return AIMessage(content="", tool_calls=[_call("RoutingDecision", {"intent": classify(payload), "reason": "simulated rule"})])
    if "FeatureExtraction" in tool_names:
        return AIMessage(content="", tool_calls=[_call("FeatureExtraction", {"values": extract(payload.get("user_message", ""))})])
    if "ExplanationDraft" in tool_names:
        return AIMessage(content="", tool_calls=[_call("ExplanationDraft", explanation(payload))])
    if "SafetyJudgement" in tool_names:
        return AIMessage(content="", tool_calls=[_call("SafetyJudgement", {"acceptable": True, "issues": []})])
    if "search_model_documents" in tool_names:  # evidence agent tool loop
        if any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage(content="done")
        return AIMessage(content="", tool_calls=evidence_calls(payload))
    return AIMessage(content="")


def looping_responder(messages: list[BaseMessage], tool_names: list[str]) -> AIMessage:
    """Evidence agent that never stops calling tools - used to test loop protection."""
    if "search_model_documents" in tool_names:
        return AIMessage(content="", tool_calls=[_call("search_model_documents", {"query": "validation"}, len(messages))])
    return simulated_responder(messages, tool_names)


def simulated_llm(faults: list[Exception] | None = None, responder: Responder = simulated_responder) -> ScriptedChatModel:
    return ScriptedChatModel(responder=responder, faults=faults or [])
