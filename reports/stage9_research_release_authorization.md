# Stage 9 research-only release authorization

## Решение

До создания новых Stage 9 metadata разрешена только упаковка уже зафиксированного
`minimal_v1` как воспроизводимого исследовательского snapshot. Статус решения:
`AUTHORIZED_RESEARCH_ONLY_METADATA_PACKAGING`.

Решение принято после закрытия Stage 8 и публикации model card, но до создания Stage 9
input schema, release manifest, inference smoke test или release tag. Оно не основано на
новом fitting, новых predictions, performance metrics, predictor-outcome associations или
повторном доступе к held-out labels.

## Разрешённый scope

- переносимая machine-readable schema семи входных признаков `minimal_v1`;
- manifest, связывающий source commit, evidence и существующие locked SHA;
- побайтовая проверка уже сохранённых pipeline-компонентов;
- локальный inference smoke test на синтетических, не являющихся данными участников,
  записях;
- reproducibility documentation;
- source commit tag после merge и успешного post-merge CI.

Новый estimator не создаётся, существующие joblib-файлы не пересериализуются и не
переупаковываются в отдельный release asset.

## Явно не разрешено

- refit или recalibration;
- изменение sigmoid calibrator или cutoff `0.007469058716065273`;
- доступ к NHIS temporal или NHANES held-out labels;
- API route, web routing или routing manifest;
- clinical deployment, diagnosis, treatment или triage claims;
- export `clinical_core` или `clinical_hba1c`;
- заявление о достаточной independent external-validation precision;
- прикрепление model binary как отдельного production/deployment release asset.

Stage 8 сохраняет `PRIMARY_EXPORT_GATE = FAIL`. Stage 6 temporal/external результаты
остаются `UNDERPOWERED_EXPLORATORY`. Research snapshot не меняет эти результаты и не
является новой external validation.

## Неизменная идентичность

| Поле | Зафиксированное значение |
|---|---|
| Model ID | `minimal_v1` |
| Full pipeline SHA-256 | `758d635310a43596655bd23a0ebcebc293a0f4bde4ab5d98f3852a84e023a010` |
| Calibration | `sigmoid` |
| Predictor parameters | 15 |
| Locked cutoff | `0.007469058716065273` |

## Условия source tag

Tag и GitHub source release допустимы только после merge Stage 9 PR в `main`, успешного
post-merge GitHub Actions, PASS Stage 9 validator и подтверждения чистого worktree. Release
должен называться research snapshot и не должен содержать отдельный deployable model asset.

Machine-readable companion:
`artifacts/stage9_research_release_authorization/status.json`.
