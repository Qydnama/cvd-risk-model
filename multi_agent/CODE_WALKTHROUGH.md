# Путь выполнения по файлам

## 1. Модель `minimal_v1` (не изменяется)

Описание модели: [reports/minimal_v1_model_card.md](../reports/minimal_v1_model_card.md).

- Исход: вероятность смерти от сердечно-сосудистых причин в течение 5 лет.
- Обучение: NHIS 1997–2009 (N = 178 120, 2 036 CVD-смертей).
- Проверки: temporal на NHIS 2010–2014 и external на NHANES 1999–2014. Обе со статусом underpowered/exploratory.

Семь входов: `age`, `sex`, `bmi`, `smoking_status`, `hypertension_dx`, `diabetes_dx`, `general_health`.

Pipeline [full_development_pipeline.joblib](../artifacts/minimal_v1/full_development_pipeline.joblib) уже содержит preprocessing, логистическую регрессию и sigmoid-калибратор. Вероятность считается так: `predict_proba(frame, calibrated=True)`.

## 2. Модули подпроекта

| Файл | Назначение |
|---|---|
| [config.py](src/cvd_agents/config.py) | Пути, лимиты (`Limits`), чтение `.env`, имя модели из `AGENT_MODEL` |
| [contracts.py](src/cvd_agents/contracts.py) | Pydantic-сообщения между агентами и `SessionState` |
| [tools.py](src/cvd_agents/tools.py) | Детерминированные функции: схема, частичная и полная проверка, `verify_locked_release` (кэш на процесс), инференс |
| [agent_tools.py](src/cvd_agents/agent_tools.py) | Обёртки LangChain `@tool` над `tools.py` для агентов проверки и прогноза |
| [evidence.py](src/cvd_agents/evidence.py) | Индекс источников, поиск, три инструмента агента доказательств |
| [safety_rules.py](src/cvd_agents/safety_rules.py) | Правила безопасности и инструмент `check_draft_rules` |
| [llm.py](src/cvd_agents/llm.py) | `build_chat_model`, классификация временных ошибок, `call_with_retries` |
| [telemetry.py](src/cvd_agents/telemetry.py) | `RunRecorder` (JSONL), `TelemetryCallback`, `calculate_agent_load` |
| [agents.py](src/cvd_agents/agents.py) | Шесть агентов |
| [graph.py](src/cvd_agents/graph.py) | `StateGraph` одного хода, `CVDAssistant` (сессия, журнал) |
| [offline_llm.py](src/cvd_agents/offline_llm.py) | Симулятор LLM на правилах для тестов без API |
| [scenario_runner.py](src/cvd_agents/scenario_runner.py) | Запуск [scenarios.json](scenarios/scenarios.json), внедрение сбоев, проверка ожиданий |
| [cli.py](src/cvd_agents/cli.py) | Команды `assess`, `chat`, `scenarios` |
| [prototype.py](src/cvd_agents/prototype.py) | Конвейер Ассайнмента 2 для команды `assess` |

## 3. Один ход: «да» в ответ на подтверждение

```text
CVDAssistant.handle("да")
  → TurnContext (дедлайн 90 с, счётчик шагов)
  → graph.invoke
     route    : OrchestratorAgent.route — правило: фаза awaiting_confirmation и «да» → confirm (без LLM)
     predict  : RiskPredictionAgent.run
                  tool verify_locked_release → validate_stage9_research_release (SHA, provenance, smoke)
                  tool predict_cvd_mortality_risk → pipeline.predict_proba(calibrated=True)
                  SessionState.prediction + хеш признаков; фаза result_ready
     evidence : EvidenceAgent.run — create_agent(LLM, 3 инструмента), recursion_limit 8
                  факты = artifact ответов инструментов + факт prediction#1 от модели
     explain  : ExplanationAgent.run — LLM пишет текст с заполнителем {probability_percent}; число подставляет код
     review   : SafetyReviewAgent.review — tool check_draft_rules; если правила пройдены — LLM SafetyJudgement
                  approved → finalize; revise → explain (не более 1 раза); повторный отказ → безопасный шаблон
     finalize : текст + дисклеймер
  → TurnResult; журнал events.jsonl, state.json (без значений признаков), load_report.json
```

## 4. Как обеспечиваются ключевые свойства

| Свойство | Где в коде |
|---|---|
| LLM не выдумывает значения | `InputValidationAgent.run`: цитата каждого значения должна дословно встречаться в сообщении. Тест `test_llm_never_fills_values_the_user_did_not_write` |
| Вероятность считает только модель | Узел `predict` — единственный путь к `PredictionResult`. Число в тексте подставляет код, правило `check_numbers` сверяет все числа ответа |
| Прогноз только после подтверждения | `_after_route`: `confirm` ведёт в `predict` только в фазе `awaiting_confirmation` |
| Устаревший прогноз не показывается | `SessionState.prediction_is_current()` сравнивает хеши признаков |
| Каждый ответ проверен | В графе после `explain` только ребро в `review` |
| Ошибки не превращаются в выдуманный ответ | `CVDAssistant._guard` переводит любое исключение в статус `error` или `limit_exceeded` с фиксированным сообщением |

## 5. Проверки

Подробности в [VERIFICATION.md](VERIFICATION.md):
- [tests/test_prototype.py](tests/test_prototype.py) — регрессия Ассайнмента 2;
- [tests/test_agents_and_tools.py](tests/test_agents_and_tools.py) — набор инструментов, правила, поиск, повторы, журнал, отсутствие ключей;
- [tests/test_scenarios.py](tests/test_scenarios.py) — 18 сценариев офлайн и живой тест с маркером `live`.
