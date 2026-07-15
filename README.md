# Five-year cardiovascular mortality risk model

Воспроизводимый исследовательский пайплайн для разработки на NHIS и внешней
валидации на NHANES интерпретируемой модели пятилетнего риска смерти от
сердечно-сосудистых причин. Главный протокол проекта —
[`CODEX_CVD_MODEL_IMPLEMENTATION_PLAN.md`](CODEX_CVD_MODEL_IMPLEMENTATION_PLAN.md).

Проект исследовательский: он не является медицинским изделием, не ставит диагноз
и пока не валидирован для населения Казахстана.

## Зафиксированный протокол

- Python 3.12, управление окружением и lock-файлом через `uv`.
- NHIS 1997–2009 — development; NHIS 2010–2014 — temporal test.
- NHANES 1999–2014 — независимая внешняя проверка быстрой модели.
- Возраст 40–79 лет, отсутствие диагностированного ССЗ на старте.
- Исход — смерть от болезней сердца или цереброваскулярных заболеваний за 5 лет.
- Race/ethnicity не является предиктором и используется только для описания и
  subgroup/fairness audit, если кодировки и число событий допускают надежную оценку.
- Обучение закрыто до выпуска cohort audit и прохождения stop-gate.

Числовые параметры протокола находятся в `configs/experiment.yaml`. Их нельзя
незаметно менять после просмотра результатов temporal или external test.

## Быстрый старт (PowerShell / Windows)

Установите [`uv`](https://docs.astral.sh/uv/) и выполните из корня проекта:

```powershell
uv python install 3.12
uv sync --locked --dev
uv run pytest
```

`uv run pytest` — документированная команда проверки чистой установки. После
первого `uv sync` она использует локальное окружение `.venv` и зафиксированный
`uv.lock`.

GNU Make удобен, но не обязателен на Windows. Эквивалентные команды:

| Задача | Make | PowerShell без Make |
| --- | --- | --- |
| Тесты | `make test` | `uv run pytest` |
| Проверка IPUMS | `make check-nhis` | `uv run python scripts/00_inspect_nhis_input.py` |
| Только NHANES | `make nhanes` | `uv run python scripts/01_download_nhanes.py` |
| Данные и когорты | `make data` | последовательно запустить скрипты `00`–`03` из `scripts/` |
| Cohort audit | `make audit` | `uv run python scripts/04_audit_cohorts.py --config configs/experiment.yaml` |
| Baseline-модели | `make train` | `uv run --extra modeling python scripts/05_train_minimal.py --config configs/experiment.yaml` |

`make train` проверяет неизменность Stage 3 stop-gate и обучает только три
логистических baseline-кандидата на NHIS development 1997–2009. Imputation,
weighted spline knots и sigmoid calibration обучаются внутри grouped folds;
NHIS temporal и NHANES outcomes команда не загружает. Команда не выбирает
финальную модель и не создает external-validation lock.

## Ручной файл IPUMS NHIS

Автоматизировать вход в IPUMS или передавать проекту пароль не нужно. Создайте
extract за 1997–2018 годы согласно главному протоколу, скачайте данные (`.csv.gz`,
`.csv` или `.dta`) и codebook именно этого extract, затем положите оба файла в:

```text
data/raw/nhis/
```

Годы 1997–2014 используются для primary development/temporal cohorts, а
2015–2018 сохраняются отдельно как `recent_censored` и не добавляются в
пятилетний binary outcome.

Primary prevalent-CVD exclusion одинаков для NHIS и NHANES и включает coronary
heart disease, angina, myocardial infarction и stroke. `CONHFAILEV` в NHIS
доступна только в 1999/2002, поэтому CHF сохраняется как sensitivity-only:
NIU не считается отсутствием заболевания. Параллельный CHF-strict флаг строится
для NHIS 1999/2002 и NHANES 1999–2014.

Для race/ethnicity audit добавьте в extract согласованные IPUMS-переменные расы и
этничности и сохраните их labels/codebook. До сверки codebook неизвестные числовые
коды нельзя интерпретировать как категории.

Для основной межгодовой derivation добавьте **`RACEA` и `HISPETH`**: обе
переменные покрывают весь период 1997–2018. `RACENEW` и `RACEBR` можно добавить
дополнительно для sensitivity-аудита, но они начинаются только с 1999 года и не
заменяют поля с полным покрытием. Итоговые общие категории будут утверждены лишь
после проверки labels и frequencies именно вашего codebook.
См. официальные описания IPUMS NHIS для
[`RACEA`](https://nhis.ipums.org/nhis-action/variables/racea) и
[`HISPETH`](https://nhis.ipums.org/nhis-action/variables/HISPETH).

Сначала выполните:

```powershell
uv run python scripts/00_inspect_nhis_input.py
```

Команда проверит 25 обязательных полей и все 22 года, рассчитает SHA-256 extract и
codebook, зарегистрирует их в manifest и создаст
`reports/nhis_input_inspection.md`. Она намеренно не подтверждает семантику кодов:
после ручной сверки labels/universe/missing codes нужно зафиксировать выданные
хэши и статус `verified_against_user_extract` в `configs/feature_mapping.yaml`.
До этого `make data` остановится **до** массовой загрузки NHANES.

NHANES и public-use Linked Mortality Files загружаются автоматически. Для проверки
URL одного цикла без массовой загрузки:

```powershell
uv run python -m cvd_risk.data.download_nhanes --config configs/nhanes_files.yaml --output-dir data/raw/nhanes --manifest configs/data_manifest.yaml --cycle 1999-2000 --preflight
```

Полная загрузка и построение когорт запускаются командой `make data`. Повторный
запуск проверяет manifest/checksum и не должен повторно скачивать корректные файлы.

## Структура

```text
configs/              протокол эксперимента, mapping и manifest данных
data/raw/             исходные файлы (не коммитятся)
data/interim/         промежуточные данные (не коммитятся)
data/processed/       аналитические когорты (не коммитятся)
src/cvd_risk/         воспроизводимая библиотечная логика
scripts/              последовательные entry points пайплайна
tests/                unit/integration tests
reports/tables/       таблицы cohort audit
reports/figures/      графики cohort audit
artifacts/            будущие модели и их metadata
```

Каждый загруженный файл регистрируется в `configs/data_manifest.yaml`: URL, время
загрузки, локальный путь, размер и SHA-256. Сырые и производные данные, а также
обученные бинарные модели исключены из Git.

## Stage 0 и обязательный stop-gate

Amendment 2.4 закрепляет R как reference backend для `survey 4.5`,
`pmsampsize 1.1.3` и `pmvalsampsize 0.1.0`. Команда `make stage0` проверяет SHA
протокола, quarantine старого NHANES cohort, R 4.6.1, `renv.lock`, точные версии
пакетов и замороженные reference outputs для survey/sample-size расчетов. Для
чистой установки R-зависимостей выполните
`Rscript -e "renv::restore(prompt = FALSE)"`, затем `make stage0`. На Windows
Stage 0 также находит установленный R по `.R-version`, если `Rscript` ещё не
добавлен в `PATH`. Любое расхождение получает hard `BLOCKED_*` status; `make data`
и `make audit` дальше не продолжаются.

Pre-amendment `nhanes_cohort.parquet` зарегистрирован в
`artifacts/audit/quarantine_manifest.json` и запрещён для валидации, clinical
development, stop-gate и обучения.

## Обязательный stop-gate

До этапа моделирования `make audit` должен создать cohort flow, event counts,
missingness, feature availability, CHF sensitivity, R/Taylor survey precision,
DEFF, design/Kish ESS, `pmsampsize` gate и диагностические графики. Минимальные желательные
числа событий: 500 в NHIS development, 200 в NHIS temporal test и 100 в NHANES
external validation. Также проверяются 20 событий на каждый предварительно
зафиксированный effective degree of freedom логистической модели.

Если критерий не выполнен, пайплайн выпускает feasibility report и останавливается.
Он не расширяет возраст, горизонт, outcome и не объединяет test с train без явного
изменения протокола.

### Веса NHANES — решение amendment 2.4

Официальная
[`инструкция CDC`](https://wwwn.cdc.gov/nchs/nhanes/tutorials/weighting.aspx)
требует при объединении 1999–2000 и 2001–2002 использовать
предоставленный четырёхлетний вес из-за разных reference populations. Для
16-летнего объединения рекомендуемое масштабирование — `WTMEC4YR / 4` для
1999–2002 и `WTMEC2YR / 8` для 2003–2014.

Amendment 2.4 принимает это правило: `WTMEC4YR / 4` используется для
1999–2002, а `WTMEC2YR / 8` — для 2003–2014. Запись в
`configs/protocol_discrepancies.yaml` сохранена как `resolved`, чтобы решение
оставалось видимым в provenance и audit outputs.
