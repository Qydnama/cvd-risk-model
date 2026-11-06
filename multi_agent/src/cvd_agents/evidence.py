"""Evidence store over registered sources and the evidence agent's tools.

Every returned fact carries its source path and locator. The search is local and
deterministic (keyword TF-IDF with a small RU->EN synonym map), so results are
reproducible and no external embedding API is needed.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache
from typing import Any, Literal

from langchain_core.tools import tool

from .config import KNOWLEDGE_DIR, REPO_ROOT
from .contracts import EvidenceFact

MAX_FACT_CHARS = 700
TOP_K = 3

SYNONYMS: dict[str, str] = {
    "обуч": "development nhis training",
    "данн": "data nhis nhanes development",
    "выбор": "cohort n",
    "метрик": "roc-auc auc brier calibration metric",
    "точност": "roc-auc auc discrimination",
    "калибр": "calibration slope intercept",
    "валид": "validation external temporal underpowered exploratory",
    "провер": "validation",
    "казах": "kazakhstan population transportability",
    "огранич": "limitations prohibited underpowered exploratory",
    "порог": "cutoff operating point",
    "призна": "features predictors input",
    "исход": "outcome endpoint mortality primary output",
    "смерт": "mortality death deaths",
    "курен": "tobacco smoking",
    "курит": "tobacco smoking",
    "давлен": "blood pressure hypertension",
    "гиперт": "hypertension blood pressure",
    "диабет": "diabetes glucose",
    "сахар": "glucose diabetes",
    "фактор": "risk factors",
    "симпт": "symptoms",
    "инфарк": "heart attack",
    "инсуль": "stroke",
    "профил": "prevented prevention reduce risk",
    "снизи": "reduce risk cessation",
    "клинич": "clinical deployment routes",
    "назнач": "intended use",
    "казахстан": "kazakhstan",
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zа-яё0-9][a-zа-яё0-9\-]*", text.lower())


def _stem(token: str) -> str:
    return token[:6] if len(token) > 6 else token


def _expand(query: str) -> list[str]:
    tokens = _tokens(query)
    expanded = list(tokens)
    for token in tokens:
        for prefix, extra in SYNONYMS.items():
            if token.startswith(prefix):
                expanded.extend(_tokens(extra))
    return [_stem(t) for t in expanded if len(t) > 2 or t.isdigit()]


def _truncate(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= MAX_FACT_CHARS else text[: MAX_FACT_CHARS - 1] + "…"


def _markdown_chunks(source: dict[str, Any], text: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    heading = "(top)"
    block: list[str] = []
    counter: Counter[str] = Counter()

    def flush() -> None:
        if block:
            counter[heading] += 1
            body = "\n".join(block)
            chunks.append({"locator": f"{heading} ¶{counter[heading]}", "text": body, "search": f"{heading} {body}"})
            block.clear()

    for line in text.splitlines():
        if line.startswith("#"):
            flush()
            heading = line.lstrip("#").strip()
        elif not line.strip():
            flush()
        else:
            block.append(line)
    flush()
    return chunks


def _json_chunks(source: dict[str, Any], data: dict[str, Any]) -> list[dict[str, Any]]:
    chunks = []
    for key, value in data.items():
        body = f"{key} = {json.dumps(value, ensure_ascii=False)}"
        chunks.append({"locator": f"/{key}", "text": body, "search": body.replace("_", " ")})
    return chunks


@lru_cache(maxsize=1)
def load_index() -> tuple[dict[str, Any], ...]:
    registry = json.loads((KNOWLEDGE_DIR / "sources.json").read_text(encoding="utf-8"))
    index: list[dict[str, Any]] = []
    for source in registry["sources"]:
        path = REPO_ROOT / source["path"]
        raw = path.read_text(encoding="utf-8")
        if source["format"] == "markdown":
            chunks = _markdown_chunks(source, raw)
        elif source["format"] == "json":
            chunks = _json_chunks(source, json.loads(raw))
        else:  # curated excerpts
            data = json.loads(raw)
            chunks = [
                {
                    "locator": f"{item['id']} ({item['section']})",
                    "text": f"{item['text']} [{data['publisher']}, {data['url']}, published {data['published']}, accessed {data['accessed']}]",
                    "search": f"{item['section']} {item['text']} {item.get('keywords_ru', '')}",
                }
                for item in data["excerpts"]
            ]
        for number, chunk in enumerate(chunks, start=1):
            index.append(
                {
                    "fact_id": f"{source['source_id']}#{number}",
                    "source_id": source["source_id"],
                    "source_path": source["path"],
                    "kind": source["kind"],
                    "locator": chunk["locator"],
                    "text": _truncate(chunk["text"]),
                    "stems": Counter(_stem(t) for t in _tokens(chunk["search"])),
                }
            )
    return tuple(index)


def search(query: str, kind: Literal["model", "medical"], top_k: int = TOP_K) -> list[EvidenceFact]:
    pool = [entry for entry in load_index() if entry["kind"] == kind]
    terms = _expand(query)
    if not terms or not pool:
        return []
    df = Counter(term for entry in pool for term in set(terms) if entry["stems"][term])
    scored = []
    for entry in pool:
        score = sum(
            (1 + math.log(entry["stems"][term])) * math.log(1 + len(pool) / df[term])
            for term in set(terms)
            if entry["stems"][term]
        )
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda item: (-item[0], item[1]["fact_id"]))
    return [_to_fact(entry) for _, entry in scored[:top_k]]


def _to_fact(entry: dict[str, Any]) -> EvidenceFact:
    return EvidenceFact(**{k: entry[k] for k in ("fact_id", "source_id", "source_path", "locator", "kind", "text")})


ModelTopic = Literal[
    "identity", "outcome", "features", "intended_use", "development", "validation",
    "operating_point", "clinical_routes", "population",
]

TOPIC_KEYS: dict[str, tuple[str, ...]] = {
    "identity": ("model_id", "model", "calibration", "predictor_parameters"),
    "outcome": ("primary_output",),
    "features": ("model_id",),
    "intended_use": ("intended_use", "prohibited_uses"),
    "development": ("development",),
    "validation": ("validation",),
    "operating_point": ("operating_point",),
    "clinical_routes": ("clinical_routes", "clinical_deployment_approved"),
    "population": ("prohibited_uses",),
}


def model_card_facts(topic: str) -> list[EvidenceFact]:
    keys = TOPIC_KEYS.get(topic, ())
    by_locator = {e["locator"]: e for e in load_index() if e["source_id"] == "model_card_status"}
    facts = [_to_fact(by_locator[f"/{key}"]) for key in keys if f"/{key}" in by_locator]
    sections = {"features": "Входные признаки", "population": "Population и переносимость"}
    if topic in sections:
        facts += [
            _to_fact(e)
            for e in load_index()
            if e["source_id"] == "model_card" and e["locator"].startswith(sections[topic])
        ]
    return facts


def _render(facts: list[EvidenceFact]) -> str:
    if not facts:
        return "NO_VERIFIED_SOURCE: nothing found for this query."
    return "\n".join(f"[{f.fact_id}] ({f.source_path}, {f.locator}) {f.text}" for f in facts)


# --- Tools of the evidence agent (exactly three) ---------------------------------------


@tool(response_format="content_and_artifact")
def get_model_card_fact(topic: ModelTopic) -> tuple[str, list[dict[str, Any]]]:
    """Return verified model-card facts for a topic.

    Topics: identity (model type, calibration); outcome (what is predicted); features (7 inputs);
    intended_use (allowed and prohibited uses); development (training data and metrics);
    validation (temporal/external validation results and their status); operating_point (cutoff);
    clinical_routes (clinical/web deployment status); population (transportability to other
    countries incl. Kazakhstan, prohibited claims).
    """
    facts = model_card_facts(topic)
    return _render(facts), [f.model_dump() for f in facts]


@tool(response_format="content_and_artifact")
def search_model_documents(query: str) -> tuple[str, list[dict[str, Any]]]:
    """Search the minimal_v1 model card, manifests and validation reports. Returns cited passages."""
    facts = search(query, "model")
    return _render(facts), [f.model_dump() for f in facts]


@tool(response_format="content_and_artifact")
def search_medical_sources(query: str) -> tuple[str, list[dict[str, Any]]]:
    """Search curated verbatim excerpts from vetted medical sources (WHO). Returns cited passages."""
    facts = search(query, "medical")
    return _render(facts), [f.model_dump() for f in facts]


EVIDENCE_TOOLS = [get_model_card_fact, search_model_documents, search_medical_sources]
