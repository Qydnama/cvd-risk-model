# Stage 9: `minimal_v1` research release candidate

## Итоговый статус

`minimal_v1` подготовлен только как
`RESEARCH_ONLY_SOURCE_SNAPSHOT_CANDIDATE`. Это metadata/documentation layer над уже
существующим locked pipeline, а не новый model export, refit или deployment package.

## Что добавлено

- portable `input_schema.json` для семи canonical model inputs;
- `release_manifest.json` с побайтовыми SHA-256 locked binaries и переносимыми
  LF-normalized SHA-256 текстовых metadata, model card, authorization и `uv.lock`;
- read-only validator, который до десериализации проверяет доверенный SHA pipeline;
- inference smoke test на полностью синтетических canonical records;
- команды воспроизведения для Windows/Linux и CI.

Pipeline, preprocessing, estimator и sigmoid calibrator не пересериализовывались. Ни один
participant record и ни одна NHIS temporal/NHANES held-out label не читались.

## Input contract

Порядок model inputs остаётся неизменным:

1. `age`;
2. `sex`;
3. `bmi`;
4. `smoking_status`;
5. `hypertension_dx`;
6. `diabetes_dx`;
7. `general_health`.

Schema требует полный набор. Хотя locked preprocessing содержит обученные на development
data imputation values, бесшумная подстановка отсутствующих персональных inputs не
разрешена. Маршрут `minimal_no_general_health` не экспортирован и fallback routing не
создаётся.

Высота и вес не являются прямыми model inputs. Внешний исследовательский интерфейс может
вычислить BMI только после собственной проверки единиц и диапазонов; API/web interface в
этом Stage 9 не создаётся.

## Smoke validation

Validator выполняет следующий порядок:

1. сверяет authorization commit/blob и неизменность Stage 8 closure;
2. сверяет SHA-256 четырёх существующих locked binary artifacts;
3. сверяет schema, model card, manifest и `uv.lock`;
4. только после успешной проверки SHA загружает full pipeline;
5. проверяет route, feature order, 15 transformed parameters, sigmoid calibrator и cutoff;
6. получает finite probabilities в `[0, 1]` для синтетических complete records.

Smoke predictions не являются performance metrics, participant results или новой
validation.

## Governance

- `RESEARCH_USE_ONLY = TRUE`;
- `CLINICAL_DEPLOYMENT_ALLOWED = FALSE`;
- `API_ROUTE_ALLOWED = FALSE`;
- `WEB_ROUTING_ALLOWED = FALSE`;
- `PRIMARY_EXPORT_GATE = FAIL`;
- external validation interpretation остаётся `UNDERPOWERED_EXPLORATORY`.

Clinical_core и clinical_hba1c не экспортируются. Locked cutoff
`0.007469058716065273` остаётся secondary screening reference, а не diagnostic boundary.

## Release procedure

Планируемый tag: `minimal-v1-research-v1.0.0`. Он может указывать только на merged `main`
commit после успешного post-merge CI и PASS Stage 9 validator. GitHub release должен быть
помечен как research snapshot; отдельный model-binary или production asset не
прикрепляется.

Machine-readable artifacts:

- `artifacts/stage9_research_release_authorization/status.json`;
- `artifacts/minimal_v1_research_release/input_schema.json`;
- `artifacts/minimal_v1_research_release/release_manifest.json`.
