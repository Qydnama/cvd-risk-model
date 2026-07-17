# Model card: `minimal_v1`

## Краткое резюме

`minimal_v1` — зафиксированный исследовательский pipeline логистической регрессии для
оценки вероятности смерти от сердечно-сосудистых причин в течение пяти лет. Он разработан
на NHIS 1997–2009 для взрослых 40–79 лет и использует только семь признаков, доступных без
лабораторного обследования.

Модель не является медицинским диагнозом, клинически одобренной шкалой, средством выбора
лечения или заменой профессиональной оценки. Несмотря на наличие serialized pipeline,
clinical deployment, API и web routing не утверждены.

## Идентификация и неизменность

| Поле | Значение |
|---|---|
| Model ID | `minimal_v1` |
| Protocol | `2.4` |
| Estimator | Survey-weighted `LogisticRegression` |
| Calibration | Nested grouped sigmoid/Platt |
| Predictor parameters | 15 |
| Locked source commit | `5cace77785c3e7e3398ead12efc101ae235034ec` |
| Full pipeline SHA-256 | `758d635310a43596655bd23a0ebcebc293a0f4bde4ab5d98f3852a84e023a010` |
| Model-card status | `DOCUMENTED_LOCKED_RESEARCH_MODEL_UNCHANGED` |

Model card создан после lock только из сохранённых aggregate evidence. Pipeline,
preprocessing, estimator, calibrator, cutoff, manifests и recorded SHA не изменялись;
refit не выполнялся.

## Целевой исход и выход модели

Primary output — непрерывная оценка вероятности **пятилетней CVD mortality**, а не общей
вероятности сердечно-сосудистого заболевания. Endpoint не включает диагностику текущего
ССЗ и не должен интерпретироваться как вероятность инфаркта, инсульта или любого CVD event.

Secondary screening reference использует правило `predicted risk >= cutoff` с неизменным
cutoff `0.007469058716065273`. Этот порог был выбран на development OOF для weighted
sensitivity не ниже 0.80 и не является клинически утверждённой high-risk boundary.

## Входные признаки

Pipeline ожидает следующий порядок:

1. `age` — numeric;
2. `sex` — `female` или `male`;
3. `bmi` — numeric;
4. `smoking_status` — `never`, `former` или `current`;
5. `hypertension_dx` — boolean;
6. `diabetes_dx` — boolean;
7. `general_health` — `excellent`, `very_good`, `good`, `fair` или `poor`.

Age и BMI представлены restricted cubic spline terms. Категориальные признаки используют
prespecified encoding. Обученные imputation values, spline knots и sigmoid calibrator
являются частью locked pipeline. Неизвестные пользовательские значения нельзя заменять
вымышленными или средними значениями вне pipeline.

## Development data и дизайн

- Источник: NHIS 1997–2009.
- N = 178,120; raw CVD deaths = 2,036.
- Five-fold `StratifiedGroupKFold`, grouped по survey year.
- Survey weights нормировались отдельно внутри каждого training fold.
- Imputation, encoding и spline knots обучались только внутри training folds.
- Sigmoid calibration выполнялась nested grouped cross-fitting.

Development survey-weighted OOF metrics для locked sigmoid model:

| Metric | Estimate |
|---|---:|
| ROC-AUC | 0.82156 |
| PR-AUC | 0.0515941 |
| Brier score | 0.00897681 |
| Log loss | 0.0455431 |
| Calibration intercept | 0.0160432 |
| Calibration slope | 1.00373 |
| O/E | 1.00129 |

## Validation evidence: обязательная интерпретация

NHIS temporal и NHANES external-by-data-source проверки являются
**underpowered/exploratory**. Обе pre-unseal precision gates были FAIL:

| Cohort | Nominal N | Required N after design effect | Gate effective N | Gate |
|---|---:|---:|---:|---|
| NHIS temporal 2010–2014 | 82,860 | 99,862 | 34,089.26 | FAIL |
| NHANES 1999–2014 | 20,911 | 144,636 | 5,939.82 | FAIL |

Поэтому следующие результаты не подтверждают достаточную validation precision и не дают
права на заявление о clinically validated model.

| Metric | NHIS temporal estimate (95% CI) | NHANES external estimate (95% CI) |
|---|---:|---:|
| ROC-AUC | 0.7950 (0.7795–0.8099) | 0.8191 (0.7943–0.8431) |
| PR-AUC | 0.0364 (0.0321–0.0423) | 0.0531 (0.0412–0.0687) |
| Brier score | 0.008276 (0.007797–0.008750) | 0.008062 (0.007075–0.008990) |
| Log loss | 0.043712 (0.041544–0.045970) | 0.041871 (0.037423–0.046121) |
| Calibration intercept | -0.5030 (-0.7163 to -0.2947) | -0.0831 (-0.4606 to 0.2745) |
| Calibration slope | 0.9140 (0.8622–0.9645) | 1.0312 (0.9362–1.1284) |
| O/E | 0.8551 (0.8053–0.9043) | 0.8217 (0.7180–0.9201) |

O/E ниже 1 указывает на aggregate overprediction относительно наблюдаемой частоты. В
NHIS temporal calibration intercept также показывает заметный calibration shift.
Recalibration-in-the-large исследовалась только как sensitivity analysis и не заменяет
locked calibrator.

## Fixed operating point

| Cohort | Sensitivity | Specificity | PPV | NPV | Weighted flagged fraction |
|---|---:|---:|---:|---:|---:|
| NHIS development OOF | 0.8002 | 0.6973 | 0.02414 | 0.99733 | 0.3074 |
| NHIS temporal | 0.7814 | 0.6664 | 0.01953 | 0.99722 | 0.3374 |
| NHANES external | 0.7897 | 0.6595 | 0.01906 | 0.99734 | 0.3442 |

Низкий PPV означает большое число false-positive flags при редком исходе. Порог подходит
только как исследовательский secondary screening reference; он не определяет диагноз,
лечение или срочность помощи.

## Subgroup и fairness limitations

Subgroup analyses были pre-registered, но остаются exploratory. Race/ethnicity является
audit-only и не входит в predictors. Категории с менее чем 20 events или 20 non-events
показывались только как counts.

Особенно важна неоднородность sensitivity fixed cutoff по возрасту:

| Cohort | Age | Raw events | Sensitivity (95% CI) |
|---|---|---:|---:|
| NHIS temporal | 40–54 | 165 | 0.350 (0.283–0.417) |
| NHIS temporal | 55–64 | 207 | 0.753 (0.694–0.808) |
| NHIS temporal | 65–79 | 462 | 0.990 (0.984–0.997) |
| NHANES external | 40–54 | 51 | 0.501 (0.397–0.611) |
| NHANES external | 55–64 | 45 | 0.724 (0.564–0.911) |
| NHANES external | 65–79 | 159 | 0.957 (0.920–0.988) |

Эта разница ожидаема для единого низкого absolute-risk cutoff, сильно зависящего от
возраста, но делает порог непригодным для самостоятельного clinical triage. Differences
in discrimination или sensitivity не являются сами по себе доказательством fairness.

## Post-hoc high-sensitivity operating points

Отдельный NHIS development OOF analysis получил:

- `POST_HOC_SECONDARY_HIGH_SENSITIVITY_OPERATING_POINT`:
  `0.004049691041817001`;
- `EXPLORATORY_EXTREME_SENSITIVITY_STRESS_TEST`:
  `0.002714729600550364`.

Они не меняют locked cutoff, не выбирались по NHIS temporal или NHANES held-out labels и
не являются externally validated thresholds.

## Clinical routes

Stage 8 closed с `PRIMARY_EXPORT_GATE = FAIL` для clinical_core. Clinical_core остаётся
`UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY`; статус `CLINICAL_HBA1C_FITTING =
NOT_AUTHORIZED_NOT_PERFORMED`, fitting не выполнялся. Clinical estimator, serialized
pipeline, API route и web routing не созданы.

## Intended use

Допустимо:

- воспроизводимое исследование пятилетней CVD mortality risk;
- сравнение методов и aggregate performance;
- exploratory secondary screening simulations с явным описанием ограничений;
- дальнейшая prospective validation по новому заранее зафиксированному протоколу.

Запрещено или не утверждено:

- медицинская диагностика;
- самостоятельное решение о лечении, triage или отказе от помощи;
- использование как clinically endorsed risk score;
- утверждение о независимой validation precision;
- выводы о населении Казахстана на основании данных США;
- скрытая замена missing inputs вымышленными значениями;
- изменение model/calibrator/cutoff под тем же ID `minimal_v1`.

## Population и переносимость

Development и validation data отражают население США и соответствующие survey periods.
Переносимость на другие страны, системы здравоохранения, периоды, способы измерения и
возраст вне 40–79 лет не установлена. Модель не должна описываться как валидированная для
Казахстана.

## Governance и обновление

Любое изменение predictor set, preprocessing, estimator, calibrator, cutoff или output
semantics требует нового model ID, нового lock и отдельной validation. Post-hoc результаты
не могут переписать prespecified primary result. Проверяемый machine-readable companion:
`artifacts/minimal_v1_model_card/status.json`.

Основные источники evidence:

- `artifacts/minimal_v1/external_validation_lock.json`;
- `artifacts/minimal_v1/external_validation_lock_amendment_001.json`;
- `reports/stage6_underpowered_exploratory_validation_final.md`;
- `reports/posthoc_secondary_high_sensitivity_operating_points.md`;
- `reports/stage8_closure_report.md`.
