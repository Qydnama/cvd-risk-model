# Stage 8: UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY

## Обязательная интерпретация

**UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY**

Это underpowered NHANES internal development analysis. Он не является independent external validation и не разрешает export, production use, API или web routing.

Primary/export gate остаётся неизменным: `FAIL_INSUFFICIENT_EFFECTIVE_SAMPLE_SIZE`, n_eff_design=13061.96 < N_pmsampsize=13732.

## Дизайн

Обе модели получили OOF predictions для одних и тех же 18,987 участников и одинаковых четырёх folds, grouped по NHANES survey cycle. HbA1c использовался только для фиксированного availability mask и не входил ни в одну модель. Clinical fields были complete-case и не импутировались; imputation minimal predictors обучалась отдельно только внутри каждого training fold. Sigmoid calibration была nested и grouped.

## Survey-weighted OOF metrics

| model         | metric                  |   estimate |    ci_lower |   ci_upper |   valid_replicates |
|:--------------|:------------------------|-----------:|------------:|-----------:|-------------------:|
| minimal       | roc_auc                 | 0.82057    |  0.794692   | 0.84596    |               2000 |
| minimal       | average_precision       | 0.0534023  |  0.038223   | 0.0747753  |               2000 |
| minimal       | brier_score             | 0.00752648 |  0.00646448 | 0.00853006 |               2000 |
| minimal       | log_loss                | 0.0393164  |  0.0344233  | 0.0439878  |               2000 |
| minimal       | calibration_intercept   | 0.0895816  | -0.313106   | 0.469503   |               2000 |
| minimal       | calibration_slope       | 1.02448    |  0.931967   | 1.12178    |               2000 |
| minimal       | observed_expected_ratio | 0.989557   |  0.848735   | 1.12232    |               2000 |
| clinical_core | roc_auc                 | 0.820307   |  0.794318   | 0.844929   |               2000 |
| clinical_core | average_precision       | 0.0516516  |  0.0370876  | 0.0728225  |               2000 |
| clinical_core | brier_score             | 0.00753339 |  0.00646372 | 0.00853984 |               2000 |
| clinical_core | log_loss                | 0.0394596  |  0.0345689  | 0.0441264  |               2000 |
| clinical_core | calibration_intercept   | 0.149762   | -0.240678   | 0.509775   |               2000 |
| clinical_core | calibration_slope       | 1.03766    |  0.946847   | 1.1303     |               2000 |
| clinical_core | observed_expected_ratio | 0.994372   |  0.852982   | 1.12734    |               2000 |

## Paired differences: clinical_core minus minimal

| metric                  |     estimate |     ci_lower |    ci_upper |   valid_replicates |
|:------------------------|-------------:|-------------:|------------:|-------------------:|
| roc_auc                 | -0.000263679 | -0.0061794   | 0.00543098  |               2000 |
| average_precision       | -0.00175068  | -0.00599794  | 0.00259922  |               2000 |
| brier_score             |  6.90672e-06 | -8.81266e-06 | 2.27457e-05 |               2000 |
| log_loss                |  0.000143174 | -7.52314e-05 | 0.000364996 |               2000 |
| calibration_intercept   |  0.0601801   | -0.031931    | 0.151906    |               2000 |
| calibration_slope       |  0.0131768   | -0.00860899  | 0.0346812   |               2000 |
| observed_expected_ratio |  0.00481433  | -0.00220651  | 0.0113193   |               2000 |

95% percentile intervals получены из pre-specified stratified PSU-within-strata cluster bootstrap: 2,000/2,000 valid paired replicates, fixed OOF predictions, без refit.

## Decision-curve analysis

| label                                     |   threshold | quantity                    |     estimate |      ci_lower |      ci_upper |   valid_replicates |   confidence_level |
|:------------------------------------------|------------:|:----------------------------|-------------:|--------------:|--------------:|-------------------:|-------------------:|
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.005 | minimal_net_benefit         |  0.00446188  |   0.00360935  |   0.00531151  |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.005 | clinical_core_net_benefit   |  0.00457095  |   0.00368718  |   0.00548357  |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.005 | clinical_core_minus_minimal |  0.000109076 |  -8.98224e-05 |   0.000313947 |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.005 | treat_all_net_benefit       |  0.00277227  | nan           | nan           |                  0 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.005 | treat_none_net_benefit      |  0           | nan           | nan           |                  0 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.01  | minimal_net_benefit         |  0.00331442  |   0.00247725  |   0.00412025  |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.01  | clinical_core_net_benefit   |  0.00306805  |   0.00231224  |   0.0037997   |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.01  | clinical_core_minus_minimal | -0.000246364 |  -0.000552608 |   5.35132e-05 |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.01  | treat_all_net_benefit       | -0.00226424  | nan           | nan           |                  0 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.01  | treat_none_net_benefit      |  0           | nan           | nan           |                  0 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.02  | minimal_net_benefit         |  0.00167726  |   0.00103584  |   0.0023183   |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.02  | clinical_core_net_benefit   |  0.00180711  |   0.00118351  |   0.00242816  |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.02  | clinical_core_minus_minimal |  0.000129847 |  -0.000148029 |   0.000397542 |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.02  | treat_all_net_benefit       | -0.0124914   | nan           | nan           |                  0 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.02  | treat_none_net_benefit      |  0           | nan           | nan           |                  0 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.03  | minimal_net_benefit         |  0.00145253  |   0.000883889 |   0.00201188  |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.03  | clinical_core_net_benefit   |  0.000933768 |   0.000468132 |   0.00136874  |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.03  | clinical_core_minus_minimal | -0.000518758 |  -0.000864211 |  -0.000175537 |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.03  | treat_all_net_benefit       | -0.0229295   | nan           | nan           |                  0 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.03  | treat_none_net_benefit      |  0           | nan           | nan           |                  0 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.05  | minimal_net_benefit         |  0.00062794  |   0.000225323 |   0.00102031  |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.05  | clinical_core_net_benefit   |  0.000511885 |   0.000144377 |   0.000879287 |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.05  | clinical_core_minus_minimal | -0.000116055 |  -0.000302693 |   7.19905e-05 |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.05  | treat_all_net_benefit       | -0.0444648   | nan           | nan           |                  0 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.05  | treat_none_net_benefit      |  0           | nan           | nan           |                  0 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.075 | minimal_net_benefit         |  0.000271645 |  -2.45297e-05 |   0.000550551 |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.075 | clinical_core_net_benefit   |  0.000254682 |  -3.60471e-05 |   0.000528357 |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.075 | clinical_core_minus_minimal | -1.69629e-05 |  -9.03837e-05 |   5.38347e-05 |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.075 | treat_all_net_benefit       | -0.0726936   | nan           | nan           |                  0 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.075 | treat_none_net_benefit      |  0           | nan           | nan           |                  0 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.1   | minimal_net_benefit         |  0.000123621 |  -0.000121891 |   0.00035744  |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.1   | clinical_core_net_benefit   | -1.078e-05   |  -0.000189995 |   0.000158465 |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.1   | clinical_core_minus_minimal | -0.000134401 |  -0.000310674 |   4.3843e-05  |               2000 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.1   | treat_all_net_benefit       | -0.102491    | nan           | nan           |                  0 |               0.95 |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |       0.1   | treat_none_net_benefit      |  0           | nan           | nan           |                  0 |               0.95 |

## Identical outer folds

| label                                     |   fold | train_cycles                                                | validation_cycles   |   n_train |   events_train |   n_validation |   events_validation | participants_identical_for_models   |
|:------------------------------------------|-------:|:------------------------------------------------------------|:--------------------|----------:|---------------:|---------------:|--------------------:|:------------------------------------|
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |      1 | 1999-2000;2001-2002;2005-2006;2007-2008;2011-2012;2013-2014 | 2003-2004;2009-2010 |     14112 |            161 |           4875 |                  57 | True                                |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |      2 | 1999-2000;2001-2002;2003-2004;2007-2008;2009-2010;2011-2012 | 2005-2006;2013-2014 |     14201 |            164 |           4786 |                  54 | True                                |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |      3 | 1999-2000;2003-2004;2005-2006;2007-2008;2009-2010;2013-2014 | 2001-2002;2011-2012 |     14375 |            174 |           4612 |                  44 | True                                |
| UNDERPOWERED_EXPLORATORY_DEVELOPMENT_ONLY |      4 | 2001-2002;2003-2004;2005-2006;2009-2010;2011-2012;2013-2014 | 1999-2000;2007-2008 |     14273 |            155 |           4714 |                  63 | True                                |

## Ограничения

Результаты получены после aggregate gate failure и являются underpowered, development-only и exploratory. Они не переопределяют Section 10 FAIL, full-gate FAIL или ранее зафиксированные Stage 6/7 результаты. Полный-cohort estimator не обучался и сериализованный pipeline не создавался.
