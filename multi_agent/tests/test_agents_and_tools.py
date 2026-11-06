"""Agent registry, actual tool sets, safety rules, evidence and retry behaviour (offline)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from cvd_agents.config import DEFAULT_LIMITS, KNOWLEDGE_DIR, PACKAGE_ROOT, REPO_ROOT
from cvd_agents.evidence import search
from cvd_agents.graph import CVDAssistant
from cvd_agents.llm import LLMUnavailableError, TransientLLMError, call_with_retries
from cvd_agents.offline_llm import simulated_llm
from cvd_agents.safety_rules import check_citations, check_draft_rules, check_numbers, check_prohibited_claims
from cvd_agents.telemetry import RunRecorder


@pytest.fixture()
def assistant(tmp_path: Path) -> CVDAssistant:
    return CVDAssistant(simulated_llm(), runs_path=tmp_path)


def test_six_distinct_agents_with_at_most_five_tools(assistant: CVDAssistant):
    names = [agent.name for agent in assistant.agents]
    assert len(names) == len(set(names)) == 6
    for agent in assistant.agents:
        assert len(agent.tools) <= 5, agent.name
    assert assistant.orchestrator.tools == []


def test_framework_adds_no_hidden_tools_or_subagents(assistant: CVDAssistant):
    """Introspect the compiled create_agent graph: exactly the declared tools, nothing added."""
    compiled = assistant.evidence_agent.graph
    tool_node = compiled.nodes["tools"].bound
    assert sorted(tool_node.tools_by_name) == sorted(t.name for t in assistant.evidence_agent.tools)
    assert "task" not in tool_node.tools_by_name and "execute" not in tool_node.tools_by_name
    turn_graph_nodes = set(assistant.graph.nodes) - {"__start__"}
    assert turn_graph_nodes == {"route", "collect", "predict", "evidence", "explain", "review", "refuse", "reset", "finalize"}


def test_numbers_must_come_from_model_facts_or_user():
    ok = check_numbers("Оценка 0.84% за 5 лет.", 0.8374)
    bad = check_numbers("Ваш риск 50%.", 0.8374)
    assert ok == [] and bad and "50" in bad[0]["detail"]


def test_prohibited_claims_detected_and_negation_respected():
    assert check_prohibited_claims("Принимайте статины каждый день.")
    assert check_prohibited_claims("Модель валидирована для Казахстана.")
    assert check_prohibited_claims("Модель не валидирована для Казахстана.") == []
    assert check_prohibited_claims("Это вероятность развития инфаркта.", True)


def test_citations_must_exist():
    assert check_citations(["x#1"], ["y#1"])
    assert check_citations([], ["y#1"])
    assert check_citations(["y#1"], ["y#1"]) == []


def test_rule_engine_tool_accepts_ranges_from_facts():
    issues = check_draft_rules.invoke({"text": "Обучение на NHIS 1997–2009 [f#1].", "used_fact_ids": ["f#1"],
                                       "available_fact_ids": ["f#1"], "fact_texts": ['years = "1997-2009"']})
    assert issues == []


def test_evidence_facts_point_to_existing_sources():
    facts = search("Валидирована ли модель для Казахстана?", "model") + search("факторы риска ССЗ", "medical")
    assert facts
    for fact in facts:
        assert (REPO_ROOT / fact.source_path).is_file()
    assert any(f.source_id == "who_cvd_fact_sheet" for f in facts)
    who = json.loads((KNOWLEDGE_DIR / "medical" / "who_cvd_fact_sheet.json").read_text(encoding="utf-8"))
    assert who["url"].startswith("https://www.who.int/") and who["accessed"] and who["excerpts"]


def test_retries_are_bounded_and_logged(tmp_path: Path):
    recorder = RunRecorder("r", tmp_path)
    calls = {"n": 0}

    def always_fails():
        calls["n"] += 1
        raise TransientLLMError("429")

    with pytest.raises(LLMUnavailableError):
        call_with_retries(always_fails, recorder=recorder, agent="a", limits=DEFAULT_LIMITS,
                          deadline=float("inf"), sleep=lambda s: None)
    assert calls["n"] == DEFAULT_LIMITS.llm_max_attempts
    assert sum(e["kind"] == "retry" for e in recorder.events) == DEFAULT_LIMITS.llm_max_attempts


def test_non_transient_errors_are_not_retried(tmp_path: Path):
    recorder = RunRecorder("r", tmp_path)
    calls = {"n": 0}

    def bad_request():
        calls["n"] += 1
        raise ValueError("invalid schema")

    with pytest.raises(ValueError):
        call_with_retries(bad_request, recorder=recorder, agent="a", limits=DEFAULT_LIMITS,
                          deadline=float("inf"), sleep=lambda s: None)
    assert calls["n"] == 1


def test_every_llm_call_is_logged_and_attributed_without_health_values(tmp_path: Path):
    llm = simulated_llm()
    assistant = CVDAssistant(llm, runs_path=tmp_path, session_id="log_check")
    for message in ["Хочу оценить риск: мне 58 лет, мужчина, ИМТ 28.1, бросил курить, есть гипертония, нет диабета, здоровье хорошее", "да"]:
        assistant.handle(message)
    events = assistant.recorder.events
    assert sum(e["kind"] == "llm" for e in events) == llm.stats["calls"]
    assert all(e["agent"] != "unattributed" for e in events)
    raw_log = (tmp_path / "log_check" / "events.jsonl").read_text(encoding="utf-8")
    assert "28.1" not in raw_log and "мужчина" not in raw_log
    state = json.loads((tmp_path / "log_check" / "state.json").read_text(encoding="utf-8"))
    assert "features" not in state and state["known_features"]


def test_llm_never_fills_values_the_user_did_not_write(tmp_path: Path):
    from langchain_core.messages import AIMessage

    from cvd_agents.offline_llm import simulated_responder

    def hallucinating(messages, tools):
        if "FeatureExtraction" in tools:
            return AIMessage(content="", tool_calls=[{"name": "FeatureExtraction", "id": "1", "type": "tool_call",
                             "args": {"values": [{"field": "bmi", "value": "27", "quote": "ИМТ 27"}]}}])
        return simulated_responder(messages, tools)

    assistant = CVDAssistant(simulated_llm(responder=hallucinating), runs_path=tmp_path)
    result = assistant.handle("Хочу оценить риск")
    assert "bmi" not in assistant.session.features
    assert result.prediction is None


def test_no_api_keys_in_tracked_files():
    tracked = subprocess.run(["git", "ls-files", "multi_agent"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.split()
    for name in tracked:
        path = REPO_ROOT / name
        if path.suffix in {".py", ".md", ".json", ".toml", ".example"} and path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert "sk-proj-" not in text and "sk-ant-" not in text, name
    assert (PACKAGE_ROOT / ".gitignore").read_text(encoding="utf-8").count(".env") >= 1


def test_regressions_found_in_live_run():
    """Both defects were found only by the live run with gpt-4.1-mini (see VERIFICATION.md)."""
    from cvd_agents.llm import is_transient

    class OpenAITimeoutError(Exception):  # name used by langchain-openai
        pass

    assert is_transient(OpenAITimeoutError())
    assert check_prohibited_claims("Запрещено заявлять о валидации модели для Казахстана.") == []
    assert check_prohibited_claims("Модель подтверждена для Казахстана.")


def test_more_regressions_found_in_live_run():
    assert check_prohibited_claims("Это не является оценкой риска развития заболевания.", True) == []
    assert check_prohibited_claims("Это вероятность развития заболевания.", True)

    import re
    text = re.sub(r"\{probability_percent\}\s?%?", "0.95%", "составляет {probability_percent}%.")
    assert text == "составляет 0.95%."
