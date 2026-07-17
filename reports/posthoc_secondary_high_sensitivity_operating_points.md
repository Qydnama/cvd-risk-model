# Отдельный post-hoc анализ operating points с высокой чувствительностью

Этот отдельный анализ использует только sigmoid-calibrated LogisticRegression out-of-fold predictions из NHIS development 1997–2009. Доступ к NHIS temporal или NHANES held-out labels не выполнялся. Pipeline minimal_v1, sigmoid calibrator, locked cutoff, артефакты Stage 6/7, manifests и записанные SHA не изменялись.

Эти post-hoc пороги рассчитаны только на development OOF и не являются внешне валидированными порогами. Правило классификации: predicted risk >= cutoff. FN/FP counts — невзвешенные количества участников; confusion matrices дополнительно содержат survey-weighted totals.

## Operating points

### high_sensitivity_90

**POST_HOC_SECONDARY_HIGH_SENSITIVITY_OPERATING_POINT**

- Целевая survey-weighted sensitivity: >= 0.90
- Точный cutoff: `0.004049691041817001`
- Survey-weighted sensitivity: 0.900030845
- Survey-weighted specificity: 0.515963057
- Survey-weighted PPV: 0.017103164
- Survey-weighted NPV: 0.998190118
- Survey-weighted F1-score: 0.033568432
- False-negative count: 180
- False-positive count: 90325
- Weighted flagged fraction: 0.487893780
- Следующий более высокий наблюдаемый cutoff: `0.004049714609326391` (sensitivity 0.898466362; ниже цели)

Confusion matrix:

| Фактический исход | Predicted positive: raw / weighted | Predicted negative: raw / weighted |
|---|---:|---:|
| CVD death | 1856 / 852491.076923 | 180 / 94688.769231 |
| No CVD death | 90325 / 48991565.153846 | 85759 / 52222951.307692 |

### high_sensitivity_95

**EXPLORATORY_EXTREME_SENSITIVITY_STRESS_TEST**

- Целевая survey-weighted sensitivity: >= 0.95
- Точный cutoff: `0.002714729600550364`
- Survey-weighted sensitivity: 0.950238433
- Survey-weighted specificity: 0.372279899
- Survey-weighted PPV: 0.013968412
- Survey-weighted NPV: 0.998750687
- Survey-weighted F1-score: 0.027532104
- False-negative count: 89
- False-positive count: 114316
- Weighted flagged fraction: 0.630710290
- Следующий более высокий наблюдаемый cutoff: `0.002714892528450675` (sensitivity 0.949895633; ниже цели)

Confusion matrix:

| Фактический исход | Predicted positive: raw / weighted | Predicted negative: raw / weighted |
|---|---:|---:|
| CVD death | 1947 / 900046.692308 | 89 / 47133.153846 |
| No CVD death | 114316 / 63534386.461538 | 61768 / 37680130.000000 |

## Сравнение с original locked cutoff

| Operating point        | Cutoff               | Sensitivity   | Specificity   | PPV         | NPV         | F1          | FN count   | FP count   | Flagged fraction   |
|:-----------------------|:---------------------|:--------------|:--------------|:------------|:------------|:------------|:-----------|:-----------|:-------------------|
| original_locked_cutoff | 0.007469058716065273 | 0.800234672   | 0.697255048   | 0.024138934 | 0.997326038 | 0.046864218 | 366        | 58830      | 0.307357368        |
| high_sensitivity_90    | 0.004049691041817001 | 0.900030845   | 0.515963057   | 0.017103164 | 0.998190118 | 0.033568432 | 180        | 90325      | 0.487893780        |
| high_sensitivity_95    | 0.002714729600550364 | 0.950238433   | 0.372279899   | 0.013968412 | 0.998750687 | 0.027532104 | 89         | 114316     | 0.630710290        |

Original cutoff `0.007469058716065273` остаётся заблокированным и неизменным. Два более низких post-hoc cutoff повышают development OOF sensitivity ценой снижения specificity и PPV и увеличения weighted flagged fraction.

## Проверка максимальности cutoff

Для каждой цели predictions сортировались по убыванию и накапливались по точным совпадающим значениям probability с ненормированными NHIS survey weights. Указанный cutoff — первое наблюдаемое значение, на котором cumulative weighted event fraction достигает цели. Следующий более высокий cutoff не достигает цели, что подтверждает максимальность при правиле `risk >= cutoff`.

Вход: `artifacts/stage4_baselines/oof_predictions.parquet`; проверенный SHA-256: `cd8817d7a818c02914777f8288b4901ffbc7f19d5e2d33e331c7577def50d386`.
