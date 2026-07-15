# Stage 6 temporal and external validation

## Mandatory interpretation

All results are **underpowered/exploratory** because both pre-unseal survey precision gates failed. No claim of adequate precision is made. The frozen model, preprocessing, sigmoid calibrator and cutoff were not changed or reselected.

## Original frozen-model overall metrics

| cohort          | metric                      |       estimate |
|:----------------|:----------------------------|---------------:|
| nhis_temporal   | weighted_prevalence         |    0.00843398  |
| nhis_temporal   | roc_auc                     |    0.795011    |
| nhis_temporal   | average_precision           |    0.0364376   |
| nhis_temporal   | brier_score                 |    0.00827614  |
| nhis_temporal   | log_loss                    |    0.0437118   |
| nhis_temporal   | calibration_intercept       |   -0.502955    |
| nhis_temporal   | calibration_slope           |    0.913977    |
| nhis_temporal   | observed_expected_ratio     |    0.855134    |
| nhis_temporal   | log_observed_expected_ratio |   -0.156497    |
| nhis_temporal   | sensitivity                 |    0.781354    |
| nhis_temporal   | specificity                 |    0.666364    |
| nhis_temporal   | ppv                         |    0.0195308   |
| nhis_temporal   | npv                         |    0.997217    |
| nhis_temporal   | flagged_fraction            |    0.337412    |
| nhis_temporal   | tp_per_10000                |   65.8992      |
| nhis_temporal   | fn_per_10000                |   18.4405      |
| nhis_temporal   | fp_per_10000                | 3308.22        |
| nhis_temporal   | tn_per_10000                | 6607.44        |
| nhis_temporal   | net_benefit_model           |    0.0041004   |
| nhis_temporal   | net_benefit_treat_all       |    0.000972179 |
| nhis_temporal   | net_benefit_treat_none      |    0           |
| nhanes_external | weighted_prevalence         |    0.00830878  |
| nhanes_external | roc_auc                     |    0.81908     |
| nhanes_external | average_precision           |    0.0531412   |
| nhanes_external | brier_score                 |    0.00806198  |
| nhanes_external | log_loss                    |    0.0418707   |
| nhanes_external | calibration_intercept       |   -0.0830899   |
| nhanes_external | calibration_slope           |    1.03117     |
| nhanes_external | observed_expected_ratio     |    0.821677    |
| nhanes_external | log_observed_expected_ratio |   -0.196408    |
| nhanes_external | sensitivity                 |    0.789659    |
| nhanes_external | specificity                 |    0.659545    |
| nhanes_external | ppv                         |    0.0190626   |
| nhanes_external | npv                         |    0.997335    |
| nhanes_external | flagged_fraction            |    0.344187    |
| nhanes_external | tp_per_10000                |   65.6111      |
| nhanes_external | fn_per_10000                |   17.4768      |
| nhanes_external | fp_per_10000                | 3376.26        |
| nhanes_external | tn_per_10000                | 6540.65        |
| nhanes_external | net_benefit_model           |    0.00402038  |
| nhanes_external | net_benefit_treat_all       |    0.000846042 |
| nhanes_external | net_benefit_treat_none      |    0           |

## Stratified PSU cluster-bootstrap intervals

| cohort          | metric                      |       estimate |       ci_lower |      ci_upper |   confidence_level |   valid_replicates | status   |   full_ci_width |   target_full_ci_width | passes_width_target   |
|:----------------|:----------------------------|---------------:|---------------:|--------------:|-------------------:|-------------------:|:---------|----------------:|-----------------------:|:----------------------|
| nhis_temporal   | weighted_prevalence         |    0.00843398  |    0.00793557  |    0.00892798 |               0.95 |               1999 | complete |     0.000992412 |                  nan   | <NA>                  |
| nhis_temporal   | roc_auc                     |    0.795011    |    0.779547    |    0.809928   |               0.95 |               1999 | complete |     0.0303816   |                    0.1 | True                  |
| nhis_temporal   | average_precision           |    0.0364376   |    0.0320766   |    0.0423169  |               0.95 |               1999 | complete |     0.0102403   |                  nan   | <NA>                  |
| nhis_temporal   | brier_score                 |    0.00827614  |    0.00779703  |    0.00874992 |               0.95 |               1999 | complete |     0.000952894 |                  nan   | <NA>                  |
| nhis_temporal   | log_loss                    |    0.0437118   |    0.0415439   |    0.0459704  |               0.95 |               1999 | complete |     0.00442652  |                  nan   | <NA>                  |
| nhis_temporal   | calibration_intercept       |   -0.502955    |   -0.716251    |   -0.294749   |               0.95 |               1999 | complete |     0.421502    |                  nan   | <NA>                  |
| nhis_temporal   | calibration_slope           |    0.913977    |    0.862231    |    0.964541   |               0.95 |               1999 | complete |     0.10231     |                    0.4 | True                  |
| nhis_temporal   | observed_expected_ratio     |    0.855134    |    0.805261    |    0.904324   |               0.95 |               1999 | complete |     0.0990629   |                    0.2 | True                  |
| nhis_temporal   | log_observed_expected_ratio |   -0.156497    |   -0.216588    |   -0.100567   |               0.95 |               1999 | complete |     0.116021    |                  nan   | <NA>                  |
| nhis_temporal   | sensitivity                 |    0.781354    |    0.75113     |    0.810738   |               0.95 |               1999 | complete |     0.0596085   |                  nan   | <NA>                  |
| nhis_temporal   | specificity                 |    0.666364    |    0.662958    |    0.669676   |               0.95 |               1999 | complete |     0.00671837  |                  nan   | <NA>                  |
| nhis_temporal   | ppv                         |    0.0195308   |    0.0181611   |    0.0208572  |               0.95 |               1999 | complete |     0.00269616  |                  nan   | <NA>                  |
| nhis_temporal   | npv                         |    0.997217    |    0.996769    |    0.997639   |               0.95 |               1999 | complete |     0.00087002  |                  nan   | <NA>                  |
| nhis_temporal   | flagged_fraction            |    0.337412    |    0.334121    |    0.340889   |               0.95 |               1999 | complete |     0.00676783  |                  nan   | <NA>                  |
| nhis_temporal   | tp_per_10000                |   65.8992      |   61.3136      |   70.5543     |               0.95 |               1999 | complete |     9.24068     |                  nan   | <NA>                  |
| nhis_temporal   | fn_per_10000                |   18.4405      |   15.6384      |   21.4296     |               0.95 |               1999 | complete |     5.79117     |                  nan   | <NA>                  |
| nhis_temporal   | fp_per_10000                | 3308.22        | 3275.15        | 3342.13       |               0.95 |               1999 | complete |    66.9725      |                  nan   | <NA>                  |
| nhis_temporal   | tn_per_10000                | 6607.44        | 6572.86        | 6640.08       |               0.95 |               1999 | complete |    67.2251      |                  nan   | <NA>                  |
| nhis_temporal   | net_benefit_model           |    0.0041004   |    0.00363304  |    0.0045579  |               0.95 |               1999 | complete |     0.000924861 |                  nan   | <NA>                  |
| nhis_temporal   | net_benefit_treat_all       |    0.000972179 |    0.000470017 |    0.0014699  |               0.95 |               1999 | complete |     0.000999881 |                  nan   | <NA>                  |
| nhis_temporal   | net_benefit_treat_none      |    0           |    0           |    0          |               0.95 |               1999 | complete |     0           |                  nan   | <NA>                  |
| nhanes_external | weighted_prevalence         |    0.00830878  |    0.00726084  |    0.00931023 |               0.95 |               2000 | complete |     0.00204939  |                  nan   | <NA>                  |
| nhanes_external | roc_auc                     |    0.81908     |    0.794335    |    0.843061   |               0.95 |               2000 | complete |     0.0487262   |                    0.1 | True                  |
| nhanes_external | average_precision           |    0.0531412   |    0.0411891   |    0.0686904  |               0.95 |               2000 | complete |     0.0275013   |                  nan   | <NA>                  |
| nhanes_external | brier_score                 |    0.00806198  |    0.00707515  |    0.00898957 |               0.95 |               2000 | complete |     0.00191442  |                  nan   | <NA>                  |
| nhanes_external | log_loss                    |    0.0418707   |    0.037423    |    0.0461205  |               0.95 |               2000 | complete |     0.00869752  |                  nan   | <NA>                  |
| nhanes_external | calibration_intercept       |   -0.0830899   |   -0.460638    |    0.274538   |               0.95 |               2000 | complete |     0.735176    |                  nan   | <NA>                  |
| nhanes_external | calibration_slope           |    1.03117     |    0.936174    |    1.12842    |               0.95 |               2000 | complete |     0.192244    |                    0.4 | True                  |
| nhanes_external | observed_expected_ratio     |    0.821677    |    0.717998    |    0.920116   |               0.95 |               2000 | complete |     0.202118    |                    0.2 | False                 |
| nhanes_external | log_observed_expected_ratio |   -0.196408    |   -0.331289    |   -0.0832557  |               0.95 |               2000 | complete |     0.248033    |                  nan   | <NA>                  |
| nhanes_external | sensitivity                 |    0.789659    |    0.730645    |    0.84995    |               0.95 |               2000 | complete |     0.119305    |                  nan   | <NA>                  |
| nhanes_external | specificity                 |    0.659545    |    0.651823    |    0.66674    |               0.95 |               2000 | complete |     0.0149171   |                  nan   | <NA>                  |
| nhanes_external | ppv                         |    0.0190626   |    0.0168006   |    0.0213602  |               0.95 |               2000 | complete |     0.00455959  |                  nan   | <NA>                  |
| nhanes_external | npv                         |    0.997335    |    0.996365    |    0.998225   |               0.95 |               2000 | complete |     0.00185951  |                  nan   | <NA>                  |
| nhanes_external | flagged_fraction            |    0.344187    |    0.337003    |    0.351987   |               0.95 |               2000 | complete |     0.0149841   |                  nan   | <NA>                  |
| nhanes_external | tp_per_10000                |   65.6111      |   57.6268      |   73.6827     |               0.95 |               2000 | complete |    16.0559      |                  nan   | <NA>                  |
| nhanes_external | fn_per_10000                |   17.4768      |   11.6056      |   23.7662     |               0.95 |               2000 | complete |    12.1606      |                  nan   | <NA>                  |
| nhanes_external | fp_per_10000                | 3376.26        | 3305.5         | 3451.1        |               0.95 |               2000 | complete |   145.604       |                  nan   | <NA>                  |
| nhanes_external | tn_per_10000                | 6540.65        | 6464.93        | 6612.39       |               0.95 |               2000 | complete |   147.459       |                  nan   | <NA>                  |
| nhanes_external | net_benefit_model           |    0.00402038  |    0.00322009  |    0.00483527 |               0.95 |               2000 | complete |     0.00161518  |                  nan   | <NA>                  |
| nhanes_external | net_benefit_treat_all       |    0.000846042 |   -0.000209781 |    0.00185503 |               0.95 |               2000 | complete |     0.00206481  |                  nan   | <NA>                  |
| nhanes_external | net_benefit_treat_none      |    0           |    0           |    0          |               0.95 |               2000 | complete |     0           |                  nan   | <NA>                  |

## Design-based overall risk precision

| cohort          |   n_nominal |   raw_events |   weighted_prevalence |   ci_lower |   ci_upper |   full_ci_width |   target_full_ci_width | passes_target   |   design_df | method                 |
|:----------------|------------:|-------------:|----------------------:|-----------:|-----------:|----------------:|-----------------------:|:----------------|------------:|:-----------------------|
| nhis_temporal   |       82860 |          834 |            0.00843398 | 0.00772787 | 0.009204   |      0.00147612 |                  0.005 | True            |         300 | survey_svyciprop_logit |
| nhanes_external |       20911 |          255 |            0.00830878 | 0.00697743 | 0.00989163 |      0.0029142  |                  0.005 | True            |         123 | survey_svyciprop_logit |

Recalibration-in-the-large is reported separately as a sensitivity analysis and never replaces the primary predictions. The secondary cutoff remains `0.007469058716065` and is not a diagnosis or clinically endorsed boundary.
