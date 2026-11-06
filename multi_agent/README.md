# Многоагентная система для исследовательской оценки риска (Ассайнмент 4)

Тема: **многоагентная ИИ-система для интерпретируемой оценки пятилетней вероятности смерти от сердечно-сосудистых причин**. Используется зафиксированная исследовательская модель `minimal_v1`. Проект не предназначен для диагностики, выбора лечения или клинического применения. Модель не валидирована для Казахстана.

## Что сделано

Шесть агентов работают вместе в графе LangGraph:
- оркестратор;
- проверка входа;
- прогноз;
- доказательства;
- объяснение;
- проверка безопасности.

Пользователь может:
- задать вопрос о модели или о ССЗ без расчёта;
- запросить расчёт: система соберёт недостающие признаки, покажет значения на подтверждение и вызовет модель;
- обсудить полученный результат;
- изменить признак: прежний результат будет сброшен, система предложит пересчёт.

Запросы вне области системы, а также запросы диагноза и лечения система отклоняет.

Документы:
- [ARCHITECTURE.md](ARCHITECTURE.md) — роли, инструменты, состояние, лимиты, журнал;
- [VERIFICATION.md](VERIFICATION.md) — что именно проверено (офлайн и на реальном API) и какие дефекты найдены;
- [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md) — путь выполнения по файлам;
- [PLAN_ASSIGNMENT_4.md](PLAN_ASSIGNMENT_4.md) — исходный аудит и план.

## Запуск

Из `multi_agent/`:

```powershell
uv sync --locked --group dev
uv run --locked pytest -q                                   # офлайн-тесты, без API
uv run --locked cvd-agents scenarios --mode offline         # 18 сценариев на симуляторе
uv run --locked cvd-agents chat --llm simulated --debug     # диалог без API (симулятор, не LLM)
```

С реальной LLM нужен файл `multi_agent/.env` по образцу [.env.example](.env.example): `AGENT_MODEL=openai:gpt-4.1-mini` и `OPENAI_API_KEY=...`. Файл в `.gitignore`. Переменные окружения имеют приоритет над `.env`.

```powershell
uv run --locked cvd-agents chat --debug
uv run --locked cvd-agents scenarios --mode live
uv run --locked pytest -q -m live
```

Прежний одноразовый режим Ассайнмента 2 сохранён:

```powershell
uv run --locked cvd-agents assess --age 58 --sex male --bmi 28.1 --smoking-status former --hypertension --no-diabetes --general-health good
```

Журналы пишутся в `runs/` (каталог в `.gitignore`). Зачётные сводки сценариев скопированы в [evidence/](evidence/).

## Что не меняется

Корневой проект, `artifacts/**`, `reports/**` и корневой `uv.lock`. Подпроект имеет собственное окружение. Версии научных библиотек соответствуют release manifest.
