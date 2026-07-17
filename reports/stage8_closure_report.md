# Stage 8 closure report

## Итоговый статус

Stage 8 закрыт со статусом
`CLOSED_PRIMARY_GATE_FAILED_EXPLORATORY_ANALYSES_RETAINED`.

Primary/export gate остаётся неизменным:

- `PRIMARY_EXPORT_GATE = FAIL`;
- причина: `FAIL_INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE`;
- `n_eff_design = 13061.9598240511 < N_pmsampsize = 13732`.

Следовательно, ни `clinical_core`, ни `clinical_hba1c` не получают права на export,
production use, API route или web routing. Stage 8 не изменял locked `minimal_v1`, его
sigmoid calibrator или cutoff.

## Реестр Stage 8 анализов

| Анализ | Cohort | Events | Обязательная интерпретация | Статус |
|---|---:|---:|---|---|
| Full primary/export gate | 19,018 | 218 | Primary Section 20 clinical_core feasibility | FAIL |
| Исторический paired cohort с доступным HbA1c | 18,987 | 218 | `HBA1C_AVAILABLE_PAIRED_COHORT_SENSITIVITY_ANALYSIS` | Сохранён без изменения |
| Исправленный primary Section 20 paired OOF | 19,018 | 218 | `UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY` | COMPLETE |
| clinical_hba1c predictor model | — | — | Отдельного разрешения на fitting не было | NOT PERFORMED |

В историческом N=18,987 анализе HbA1c применялся только как availability filter и не
был predictor. Исправленный N=19,018 анализ требует complete SBP, total cholesterol и
HDL, но не требует и не загружает HbA1c. Оба анализа остаются NHANES internal
development/exploratory analyses, а не independent external validation.

## Исправленное paired OOF сравнение

Minimal comparator и clinical_core использовали одинаковых участников, те же четыре
grouped folds, неизменённые спецификации 15 и 20 predictor parameters, fold-only
minimal imputation, отсутствие clinical-field imputation и ту же nested grouped sigmoid
calibration. Paired cluster bootstrap дал 2,000/2,000 валидных реплик.

Основные paired differences `clinical_core - minimal`:

| Метрика | Difference | 95% CI |
|---|---:|---:|
| ROC-AUC | -0.000168151 | [-0.006102568; 0.005533571] |
| PR-AUC | -0.001801749 | [-0.006058760; 0.002564468] |
| Brier score | 0.000006830 | [-0.000009033; 0.000022685] |
| Log loss | 0.000139979 | [-0.000077655; 0.000362931] |
| Calibration intercept | 0.059480088 | [-0.033276418; 0.151859089] |
| Calibration slope | 0.013038681 | [-0.009169334; 0.034595537] |
| O/E | 0.004665233 | [-0.002358810; 0.011197986] |

Эти underpowered exploratory результаты не показывают подтверждённого общего улучшения
clinical_core относительно minimal comparator. Они не отменяют full-gate FAIL.

## Отдельный post-hoc high-sensitivity анализ

High-sensitivity operating points рассчитаны только по NHIS development OOF predictions:

- `POST_HOC_SECONDARY_HIGH_SENSITIVITY_OPERATING_POINT`: cutoff
  `0.004049691041817001`;
- `EXPLORATORY_EXTREME_SENSITIVITY_STRESS_TEST`: cutoff
  `0.002714729600550364`.

Они не являются Stage 8 clinical thresholds, не меняют original locked cutoff
`0.007469058716065273` и не являются externally validated operating points. NHIS temporal
и NHANES held-out labels для их выбора не использовались.

## Финальное решение

- `CLINICAL_CORE_EXPORT_ALLOWED = false`;
- `CLINICAL_HBA1C_FITTING = NOT_AUTHORIZED_NOT_PERFORMED`;
- `CLINICAL_HBA1C_EXPORT_ALLOWED = false`;
- `WEB_ROUTING_ALLOWED = false`;
- `EXTERNAL_VALIDATION = NOT_PERFORMED` для Stage 8 clinical analyses;
- создание clinical estimator, serialized pipeline, API route и routing manifest запрещено.

Дальнейшая работа может включать итоговый model card и research reporting для ранее
locked `minimal_v1`, но не clinical export без нового заранее зафиксированного протокола
и отдельного прохождения обязательных gates.
