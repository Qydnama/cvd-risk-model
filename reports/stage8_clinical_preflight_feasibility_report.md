# Stage 8 clinical routes: preflight feasibility report

## Решение

**BLOCKED_PROTOCOL_CLARIFICATION_REQUIRED — модели не обучались.**

Stage 6 one-time external validation уже завершён, поэтому этот preflight имел право прочитать rebuilt v2.4 NHANES labels. Это не повтор внешней валидации: анализ относится только к будущему NHANES internal grouped validation.

Исходная primary NHANES cohort: 20,911 участников, 255 событий.

## Route-specific event gates

| analysis                                      |     n |   events |   participants_retained_fraction |   weighted_population_retained_fraction |   design_df |   minimum_events | event_gate_status   |
|:----------------------------------------------|------:|---------:|---------------------------------:|----------------------------------------:|------------:|-----------------:|:--------------------|
| strict_section_10_clinical_core               | 18292 |      198 |                         0.874755 |                                0.89227  |         123 |              200 | FAIL                |
| strict_section_10_clinical_hba1c_paired       | 18262 |      198 |                         0.87332  |                                0.89108  |         123 |              200 | FAIL                |
| route_policy_section_20_clinical_core         | 19018 |      218 |                         0.909473 |                                0.923012 |         123 |              200 | PASS                |
| route_policy_section_20_clinical_hba1c_paired | 18987 |      218 |                         0.907991 |                                0.921791 |         123 |              200 | PASS                |

Раздел 10 требует paired complete-case с одновременно наблюдаемыми всеми minimal и clinical predictors. При этой строгой трактовке и clinical_core, и paired clinical_hba1c содержат 198 событий, то есть не достигают locked floor 200. Раздел 20 и experiment.yaml определяют complete-case только по обязательным clinical fields, оставляя minimal missingness для обработки внутри folds; при этой трактовке остаётся 218 событий и gate проходит. Выбирать одну трактовку post-hoc без явного protocol decision нельзя.

Дополнительное schema discrepancy: protocol и experiment.yaml называют поле `sbp_mean`, тогда как rebuilt cohort материализует `systolic_bp_mean`. Preflight использовал его только как явно задокументированный audit alias; modeling/export до фиксации canonical contract запрещён.

## Availability каждого predictor

| field             | canonical_field   |   n_total |   n_observed |   n_missing |   events_observed |   events_lost |   observed_fraction |   weighted_observed_fraction |
|:------------------|:------------------|----------:|-------------:|------------:|------------------:|--------------:|--------------------:|-----------------------------:|
| age               | age               |     20911 |        20911 |           0 |               255 |             0 |            1        |                     1        |
| sex               | sex               |     20911 |        20911 |           0 |               255 |             0 |            1        |                     1        |
| bmi               | bmi               |     20911 |        20549 |         362 |               234 |            21 |            0.982689 |                     0.984809 |
| smoking_status    | smoking_status    |     20911 |        20892 |          19 |               255 |             0 |            0.999091 |                     0.999462 |
| hypertension_dx   | hypertension_dx   |     20911 |        20847 |          64 |               252 |             3 |            0.996939 |                     0.998131 |
| diabetes_dx       | diabetes_dx       |     20911 |        20433 |         478 |               250 |             5 |            0.977141 |                     0.979006 |
| general_health    | general_health    |     20911 |        20900 |          11 |               255 |             0 |            0.999474 |                     0.999679 |
| systolic_bp_mean  | sbp_mean          |     20911 |        20037 |         874 |               240 |            15 |            0.958204 |                     0.962362 |
| total_cholesterol | total_cholesterol |     20911 |        19752 |        1159 |               230 |            25 |            0.944575 |                     0.955646 |
| hdl_cholesterol   | hdl_cholesterol   |     20911 |        19752 |        1159 |               230 |            25 |            0.944575 |                     0.955662 |
| hba1c             | hba1c             |     20911 |        20012 |         899 |               236 |            19 |            0.957008 |                     0.966292 |

## Поддержка по survey cycle

| analysis                                      | survey_cycle   |    n |   events |   non_events |   weighted_population |
|:----------------------------------------------|:---------------|-----:|---------:|-------------:|----------------------:|
| strict_section_10_clinical_core               | 1999-2000      | 1926 |       27 |         1899 |           9.92576e+06 |
| strict_section_10_clinical_core               | 2001-2002      | 2065 |       17 |         2048 |           1.18475e+07 |
| strict_section_10_clinical_core               | 2003-2004      | 1931 |       27 |         1904 |           1.13171e+07 |
| strict_section_10_clinical_core               | 2005-2006      | 1986 |       27 |         1959 |           1.22221e+07 |
| strict_section_10_clinical_core               | 2007-2008      | 2642 |       27 |         2615 |           1.24291e+07 |
| strict_section_10_clinical_core               | 2009-2010      | 2775 |       27 |         2748 |           1.27987e+07 |
| strict_section_10_clinical_core               | 2011-2012      | 2342 |       22 |         2320 |           1.33058e+07 |
| strict_section_10_clinical_core               | 2013-2014      | 2625 |       24 |         2601 |           1.3644e+07  |
| strict_section_10_clinical_hba1c_paired       | 1999-2000      | 1926 |       27 |         1899 |           9.92576e+06 |
| strict_section_10_clinical_hba1c_paired       | 2001-2002      | 2062 |       17 |         2045 |           1.18197e+07 |
| strict_section_10_clinical_hba1c_paired       | 2003-2004      | 1930 |       27 |         1903 |           1.13105e+07 |
| strict_section_10_clinical_hba1c_paired       | 2005-2006      | 1980 |       27 |         1953 |           1.21905e+07 |
| strict_section_10_clinical_hba1c_paired       | 2007-2008      | 2636 |       27 |         2609 |           1.24103e+07 |
| strict_section_10_clinical_hba1c_paired       | 2009-2010      | 2767 |       27 |         2740 |           1.27685e+07 |
| strict_section_10_clinical_hba1c_paired       | 2011-2012      | 2338 |       22 |         2316 |           1.32969e+07 |
| strict_section_10_clinical_hba1c_paired       | 2013-2014      | 2623 |       24 |         2599 |           1.3638e+07  |
| route_policy_section_20_clinical_core         | 1999-2000      | 1981 |       30 |         1951 |           1.01398e+07 |
| route_policy_section_20_clinical_core         | 2001-2002      | 2181 |       19 |         2162 |           1.24428e+07 |
| route_policy_section_20_clinical_core         | 2003-2004      | 2004 |       28 |         1976 |           1.16529e+07 |
| route_policy_section_20_clinical_core         | 2005-2006      | 2045 |       27 |         2018 |           1.25368e+07 |
| route_policy_section_20_clinical_core         | 2007-2008      | 2740 |       33 |         2707 |           1.2854e+07  |
| route_policy_section_20_clinical_core         | 2009-2010      | 2880 |       29 |         2851 |           1.32742e+07 |
| route_policy_section_20_clinical_core         | 2011-2012      | 2438 |       25 |         2413 |           1.37133e+07 |
| route_policy_section_20_clinical_core         | 2013-2014      | 2749 |       27 |         2722 |           1.42351e+07 |
| route_policy_section_20_clinical_hba1c_paired | 1999-2000      | 1981 |       30 |         1951 |           1.01398e+07 |
| route_policy_section_20_clinical_hba1c_paired | 2001-2002      | 2178 |       19 |         2159 |           1.2415e+07  |
| route_policy_section_20_clinical_hba1c_paired | 2003-2004      | 2003 |       28 |         1975 |           1.16463e+07 |
| route_policy_section_20_clinical_hba1c_paired | 2005-2006      | 2039 |       27 |         2012 |           1.25052e+07 |
| route_policy_section_20_clinical_hba1c_paired | 2007-2008      | 2733 |       33 |         2700 |           1.28318e+07 |
| route_policy_section_20_clinical_hba1c_paired | 2009-2010      | 2872 |       29 |         2843 |           1.3244e+07  |
| route_policy_section_20_clinical_hba1c_paired | 2011-2012      | 2434 |       25 |         2409 |           1.37044e+07 |
| route_policy_section_20_clinical_hba1c_paired | 2013-2014      | 2747 |       27 |         2720 |           1.42291e+07 |

## Разрешённое продолжение

Нужно заранее выбрать и письменно зафиксировать complete-case policy для minimal predictors в clinical routes и canonical имя SBP. Если утверждается строгий section-10 вариант, Stage 8 завершается feasibility report без fitting. Если утверждается section-20 вариант, можно отдельно preregister grouped folds, метрики, paired 2,000-repeat cluster bootstrap и DCA до обучения моделей.

Никакие Stage 8 model, calibrator, OOF predictions или routing/export artifacts в этом запуске не создавались. Locked minimal_v1 и Stage 6/7 не изменялись.

Machine-readable status: `artifacts/stage8_clinical_preflight/status.json`.
