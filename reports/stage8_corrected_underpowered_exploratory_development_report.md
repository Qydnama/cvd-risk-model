# Stage 8 corrected: UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY

**UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY**

Это исправленный парный OOF-анализ на primary Section 20 clinical_core cohort: N=19,018, 218 raw CVD events. Требуются complete SBP, total cholesterol и HDL; HbA1c не требуется, не загружается и не является predictor.

Предыдущий анализ N=18,987 / 218 events сохраняется без изменения и в соответствии с correction addendum интерпретируется только как `HBA1C_AVAILABLE_PAIRED_COHORT_SENSITIVITY_ANALYSIS`.

Primary/export gate остаётся неизменным: `FAIL_INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE`, n_eff_design=13061.96 < N_pmsampsize=13732. Этот результат — underpowered NHANES internal development analysis, а не independent external validation.

## Дизайн

Minimal comparator и clinical_core рассчитаны для одних и тех же 19,018 участников и на одинаковых четырёх grouped folds. Использованы неизменённые спецификации 15 и 20 predictor parameters. Imputation minimal predictors обучалась только внутри соответствующих training folds; clinical fields не импутировались. Calibration — та же nested grouped sigmoid calibration.

## Survey-weighted OOF metrics

| model         | metric                  |   estimate |    ci_lower |   ci_upper |   valid_replicates |
|:--------------|:------------------------|-----------:|------------:|-----------:|-------------------:|
| minimal       | roc_auc                 | 0.82046    |  0.794589   | 0.845831   |               2000 |
| minimal       | average_precision       | 0.0533093  |  0.0382432  | 0.0746565  |               2000 |
| minimal       | brier_score             | 0.0075172  |  0.00645574 | 0.00851694 |               2000 |
| minimal       | log_loss                | 0.0392803  |  0.0343788  | 0.0439391  |               2000 |
| minimal       | calibration_intercept   | 0.0921086  | -0.311493   | 0.474834   |               2000 |
| minimal       | calibration_slope       | 1.02503    |  0.932174   | 1.12256    |               2000 |
| minimal       | observed_expected_ratio | 0.989775   |  0.848577   | 1.12231    |               2000 |
| clinical_core | roc_auc                 | 0.820292   |  0.79427    | 0.844935   |               2000 |
| clinical_core | average_precision       | 0.0515076  |  0.0369322  | 0.0726264  |               2000 |
| clinical_core | brier_score             | 0.00752403 |  0.00645451 | 0.00852909 |               2000 |
| clinical_core | log_loss                | 0.0394203  |  0.0345341  | 0.0440795  |               2000 |
| clinical_core | calibration_intercept   | 0.151589   | -0.239289   | 0.512462   |               2000 |
| clinical_core | calibration_slope       | 1.03807    |  0.947465   | 1.13081    |               2000 |
| clinical_core | observed_expected_ratio | 0.99444    |  0.853103   | 1.12729    |               2000 |

## Paired differences: clinical_core minus minimal

| metric                  |     estimate |     ci_lower |    ci_upper |   valid_replicates |
|:------------------------|-------------:|-------------:|------------:|-------------------:|
| roc_auc                 | -0.000168151 | -0.00610257  | 0.00553357  |               2000 |
| average_precision       | -0.00180175  | -0.00605876  | 0.00256447  |               2000 |
| brier_score             |  6.83012e-06 | -9.03295e-06 | 2.26853e-05 |               2000 |
| log_loss                |  0.000139979 | -7.76546e-05 | 0.000362931 |               2000 |
| calibration_intercept   |  0.0594801   | -0.0332764   | 0.151859    |               2000 |
| calibration_slope       |  0.0130387   | -0.00916933  | 0.0345955   |               2000 |
| observed_expected_ratio |  0.00466523  | -0.00235881  | 0.011198    |               2000 |

95% percentile CI рассчитаны тем же paired stratified PSU-within-strata cluster bootstrap: 2,000/2,000 valid replicates, fixed OOF predictions, без refit.

## Decision-curve analysis

| label                                     |   threshold | quantity                    |     estimate |      ci_lower |      ci_upper |   valid_replicates |   confidence_level |
|:------------------------------------------|------------:|:----------------------------|-------------:|--------------:|--------------:|-------------------:|-------------------:|
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.005 | minimal_net_benefit         |  0.00448145  |   0.00362421  |   0.00532977  |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.005 | clinical_core_net_benefit   |  0.00456653  |   0.00368145  |   0.00547487  |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.005 | clinical_core_minus_minimal |  8.50796e-05 |  -0.000101713 |   0.000278385 |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.005 | treat_all_net_benefit       |  0.00276196  | nan           | nan           |                  0 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.005 | treat_none_net_benefit      |  0           | nan           | nan           |                  0 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.01  | minimal_net_benefit         |  0.00329642  |   0.0024539   |   0.00410092  |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.01  | clinical_core_net_benefit   |  0.00302969  |   0.00227879  |   0.00377393  |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.01  | clinical_core_minus_minimal | -0.000266728 |  -0.000568892 |   4.29688e-05 |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.01  | treat_all_net_benefit       | -0.0022746   | nan           | nan           |                  0 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.01  | treat_none_net_benefit      |  0           | nan           | nan           |                  0 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.02  | minimal_net_benefit         |  0.00170807  |   0.00107615  |   0.00233699  |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.02  | clinical_core_net_benefit   |  0.00180593  |   0.00118266  |   0.00242434  |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.02  | clinical_core_minus_minimal |  9.78586e-05 |  -0.000175289 |   0.000370944 |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.02  | treat_all_net_benefit       | -0.0125019   | nan           | nan           |                  0 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.02  | treat_none_net_benefit      |  0           | nan           | nan           |                  0 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.03  | minimal_net_benefit         |  0.00145507  |   0.00088759  |   0.00201335  |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.03  | clinical_core_net_benefit   |  0.000936856 |   0.000472826 |   0.00137305  |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.03  | clinical_core_minus_minimal | -0.000518216 |  -0.000863046 |  -0.000177136 |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.03  | treat_all_net_benefit       | -0.0229401   | nan           | nan           |                  0 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.03  | treat_none_net_benefit      |  0           | nan           | nan           |                  0 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.05  | minimal_net_benefit         |  0.000625887 |   0.000220728 |   0.00101659  |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.05  | clinical_core_net_benefit   |  0.000519439 |   0.000151937 |   0.000885871 |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.05  | clinical_core_minus_minimal | -0.000106448 |  -0.000288838 |   8.07829e-05 |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.05  | treat_all_net_benefit       | -0.0444756   | nan           | nan           |                  0 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.05  | treat_none_net_benefit      |  0           | nan           | nan           |                  0 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.075 | minimal_net_benefit         |  0.00027275  |  -2.28474e-05 |   0.000550573 |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.075 | clinical_core_net_benefit   |  0.000259018 |  -3.21301e-05 |   0.000532306 |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.075 | clinical_core_minus_minimal | -1.37325e-05 |  -8.58774e-05 |   5.65608e-05 |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.075 | treat_all_net_benefit       | -0.0727047   | nan           | nan           |                  0 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.075 | treat_none_net_benefit      |  0           | nan           | nan           |                  0 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.1   | minimal_net_benefit         |  0.000127054 |  -0.000117987 |   0.000361538 |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.1   | clinical_core_net_benefit   | -4.75155e-06 |  -0.000183963 |   0.000163751 |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.1   | clinical_core_minus_minimal | -0.000131806 |  -0.000309992 |   4.79403e-05 |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.1   | treat_all_net_benefit       | -0.102502    | nan           | nan           |                  0 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.1   | treat_none_net_benefit      |  0           | nan           | nan           |                  0 |               0.95 |

## Идентичные outer folds

| label                                     |   fold | train_cycles                                                | validation_cycles   |   n_train |   events_train |   n_validation |   events_validation | participants_identical_for_models   |
|:------------------------------------------|-------:|:------------------------------------------------------------|:--------------------|----------:|---------------:|---------------:|--------------------:|:------------------------------------|
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |      1 | 1999-2000;2001-2002;2005-2006;2007-2008;2011-2012;2013-2014 | 2003-2004;2009-2010 |     14134 |            161 |           4884 |                  57 | True                                |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |      2 | 1999-2000;2001-2002;2003-2004;2007-2008;2009-2010;2011-2012 | 2005-2006;2013-2014 |     14224 |            164 |           4794 |                  54 | True                                |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |      3 | 1999-2000;2003-2004;2005-2006;2007-2008;2009-2010;2013-2014 | 2001-2002;2011-2012 |     14399 |            174 |           4619 |                  44 | True                                |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |      4 | 2001-2002;2003-2004;2005-2006;2009-2010;2011-2012;2013-2014 | 1999-2000;2007-2008 |     14297 |            155 |           4721 |                  63 | True                                |

## Ограничения

Не создавались full-cohort estimator, serialized pipeline, deployable/production artifact, API route или web routing configuration. Анализ не меняет Stage 8 full-gate FAIL и не является независимой внешней валидацией.
