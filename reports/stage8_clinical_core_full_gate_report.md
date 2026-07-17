# Stage 8 clinical_core full pre-fitting gate

**FAIL_FULL_GATE_CLINICAL_CORE_FITTING_FORBIDDEN**

Анализ относится только к NHANES internal development/exploratory clinical route и не является новой independent external validation.

## Полный gate

| check                    |       observed |   required | status   |
|:-------------------------|---------------:|-----------:|:---------|
| raw_events               |   218          |   200      | PASS     |
| n_eff_design             | 13062          | 13732      | FAIL     |
| risk_ci_half_width       |     0.00167615 |     0.0025 | PASS     |
| design_df                |   123          |     8      | PASS     |
| survey_weight_validation |     7          |     7      | PASS     |
| missingness_validation   |    10          |    10      | PASS     |
| sbp_schema_alias         |     1          |     1      | PASS     |

## Design-based information

| cohort                        |   n_nominal |   raw_events |      p_hat |          se |    ci_low |   ci_high |   max_ci_half_width |   deff_y |   deff_y_gate |   n_eff_design |   n_eff_kish |   design_df |   expected_r2_cs |   pmsampsize_required_n |   required_n_nominal | pass_information   | pass_risk_precision   | pass_design_df   |   survey_version | pmsampsize_version   |
|:------------------------------|------------:|-------------:|-----------:|------------:|----------:|----------:|--------------------:|---------:|--------------:|---------------:|-------------:|------------:|-----------------:|------------------------:|---------------------:|:-------------------|:----------------------|:-----------------|-----------------:|:---------------------|
| nhanes_internal_clinical_core |       19018 |          218 | 0.00774815 | 0.000767195 | 0.0063682 | 0.0094243 |          0.00167615 |  1.45598 |       1.45598 |          13062 |      11147.1 |         123 |        0.0130135 |                   13732 |                19994 | False              | True                  | True             |              4.5 | 1.1.3                |

Primary Section 20 cohort содержит 19,018 участников и 218 raw events. Event floor сам по себе не определял решение: fitting разрешается только при PASS всех строк полного gate.

Section 10 paired all-predictor complete-case сохраняется отдельно как sensitivity feasibility cohort: N=18,262, events=198, FAIL.

## Minimal-predictor missingness

| field             | role                            |     n |   n_missing |   missing_fraction | imputation_policy             | status   |
|:------------------|:--------------------------------|------:|------------:|-------------------:|:------------------------------|:---------|
| systolic_bp_mean  | required_clinical               | 19018 |           0 |        0           | forbidden                     | PASS     |
| total_cholesterol | required_clinical               | 19018 |           0 |        0           | forbidden                     | PASS     |
| hdl_cholesterol   | required_clinical               | 19018 |           0 |        0           | forbidden                     | PASS     |
| age               | minimal_fold_imputation_allowed | 19018 |           0 |        0           | fit_inside_training_fold_only | PASS     |
| sex               | minimal_fold_imputation_allowed | 19018 |           0 |        0           | fit_inside_training_fold_only | PASS     |
| bmi               | minimal_fold_imputation_allowed | 19018 |         228 |        0.0119886   | fit_inside_training_fold_only | PASS     |
| smoking_status    | minimal_fold_imputation_allowed | 19018 |          17 |        0.00089389  | fit_inside_training_fold_only | PASS     |
| hypertension_dx   | minimal_fold_imputation_allowed | 19018 |          52 |        0.00273425  | fit_inside_training_fold_only | PASS     |
| diabetes_dx       | minimal_fold_imputation_allowed | 19018 |         431 |        0.0226627   | fit_inside_training_fold_only | PASS     |
| general_health    | minimal_fold_imputation_allowed | 19018 |          10 |        0.000525818 | fit_inside_training_fold_only | PASS     |

Clinical fields не импутировались. Любая будущая imputation minimal predictors должна обучаться только на training partition каждого grouped fold.

Исходный NHANES cohort, locked minimal_v1, Stage 6/7 artifacts и existing reports не изменялись.
