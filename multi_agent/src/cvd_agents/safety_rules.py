"""Deterministic safety checks used as tools by the safety review agent."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.tools import tool

# Numbers that describe the model itself and may appear without a citation.
MODEL_CONSTANTS = (5.0, 7.0, 40.0, 79.0, 100.0)

_NUMBER = re.compile(r"(?<![A-Za-zА-Яа-яЁё_#\d])(\d+(?:[.,]\d+)?)(?![A-Za-zА-Яа-яЁё_\d])")
_CITATION = re.compile(r"\[[^\]]{1,80}\]")

PROHIBITED: dict[str, list[str]] = {
    "diagnosis": [
        r"\bу вас (есть|имеется|диагностир\w*)\b",
        r"\bваш диагноз\b",
        r"\bвы (больны|страдаете)\b",
        r"\bставлю диагноз\b",
        r"\byou have (heart|cardiovascular) disease\b",
    ],
    "treatment": [
        r"\b(принимайте|начните принимать|назначаю|рекомендую принимать|вам следует принимать|вам нужно принимать)\b",
        r"\bдозировк\w*\b",
        r"\b(примите|пейте|начните) (таблет\w*|препарат\w*|аспирин\w*|статин\w*)\b",
        r"\b(take|start taking) (aspirin|statins?|medication)\b",
    ],
}
OUTCOME_MISMATCH = [
    r"\bвероятност\w* (заболе\w*|развити\w*|инфаркт\w*|инсульт\w*)",
    r"\bриск\w* (заболеть|развития заболевания|инфаркта|инсульта)\b",
    r"\bвероятность (иметь|наличия) ссз\b",
]


def numbers_in(text: str) -> list[str]:
    return _NUMBER.findall(_CITATION.sub(" ", text))


def _matches(token: str, allowed: list[float]) -> bool:
    value = float(token.replace(",", "."))
    decimals = len(token.replace(",", ".").split(".")[1]) if "." in token.replace(",", ".") else 0
    return any(round(candidate, decimals) == value or abs(candidate - value) < 1e-9 for candidate in allowed)


def allowed_numbers(prediction_percent: float | None, feature_numbers: list[float], fact_texts: list[str]) -> list[float]:
    allowed = list(MODEL_CONSTANTS) + [float(v) for v in feature_numbers]
    if prediction_percent is not None:
        allowed += [prediction_percent, prediction_percent / 100]
    for text in fact_texts:
        allowed += [float(n.replace(",", ".")) for n in numbers_in(text)]
    return allowed


def check_numbers(
    text: str,
    prediction_percent: float | None = None,
    feature_numbers: list[float] | None = None,
    fact_texts: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Flag every number in the draft that is not the model result, a user value or a cited fact."""
    allowed = allowed_numbers(prediction_percent, feature_numbers or [], fact_texts or [])
    return [
        {"check": "numbers", "detail": f"Unsupported number in answer: {token}"}
        for token in numbers_in(text)
        if not _matches(token, allowed)
    ]


def check_citations(used_fact_ids: list[str], available_fact_ids: list[str]) -> list[dict[str, Any]]:
    """Every cited fact must exist in the evidence bundle; non-empty evidence must be cited."""
    issues = [
        {"check": "citations", "detail": f"Cited fact not in evidence: {fact_id}"}
        for fact_id in used_fact_ids
        if fact_id not in available_fact_ids
    ]
    if available_fact_ids and not used_fact_ids:
        issues.append({"check": "citations", "detail": "Answer does not cite any evidence fact"})
    return issues


def check_prohibited_claims(text: str, result_context: bool = False) -> list[dict[str, Any]]:
    """Detect diagnoses, treatment instructions, a Kazakhstan validation claim and wrong outcome wording."""
    lowered = text.lower()
    issues = []
    for check, patterns in PROHIBITED.items():
        for pattern in patterns:
            if re.search(pattern, lowered):
                issues.append({"check": check, "detail": f"Prohibited wording matched: {pattern}"})
    for sentence in re.split(r"(?<=[.!?])\s+", lowered):
        negated = re.search(r"\bне\b|\bнет\b|\bни\b|\bбез\b|запрещ|нельзя|отсутств|not\b|\bno\b", sentence)
        if "казахстан" in sentence and re.search(r"валидир|валидац|подтвержд|проверен", sentence) and not negated:
            issues.append({"check": "kazakhstan", "detail": "Claims validation for Kazakhstan"})
        # e.g. "это не оценка риска развития заболевания" is a correct disclaimer, not a mismatch
        if result_context and not negated and any(re.search(p, sentence) for p in OUTCOME_MISMATCH):
            issues.append({"check": "outcome", "detail": "Result described as disease/event risk instead of 5-year CVD mortality"})
    return issues


@tool
def check_draft_rules(
    text: str,
    used_fact_ids: list[str],
    available_fact_ids: list[str],
    fact_texts: list[str],
    prediction_percent: float | None = None,
    feature_numbers: list[float] | None = None,
    result_context: bool = False,
) -> list[dict[str, Any]]:
    """Deterministic rule engine: numbers vs model result and facts, citations, prohibited claims, placeholders."""
    issues = check_numbers(text, prediction_percent, feature_numbers, fact_texts)
    issues += check_citations(used_fact_ids, available_fact_ids)
    issues += check_prohibited_claims(text, result_context)
    if re.search(r"\{[a-z_]+\}", text):
        issues.append({"check": "placeholder", "detail": "Unfilled placeholder in answer"})
    return issues


SAFETY_TOOLS = [check_draft_rules]
