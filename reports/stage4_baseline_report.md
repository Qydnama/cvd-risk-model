# Stage 4: development EDA and logistic baselines

## Scope and leakage policy

Only NHIS development 1997–2009 was loaded. NHIS temporal and NHANES individual outcomes were not loaded, scored, or used for model decisions.

- Participants: 178,120
- Five-year CVD deaths: 2,036
- Outer validation: StratifiedGroupKFold grouped by survey year
- Calibration: sigmoid/Platt with nested grouped cross-fitting
- Survey weights: normalized to mean 1 separately in every training fold
- Imputation and weighted spline knots: fitted separately in every training fold
- This stage does not create an external-validation lock or choose a final model

## Survey-weighted out-of-fold metrics

| model                             | route                     |   predictor_parameters | calibration   | weighting       |      n |   events |   roc_auc |   average_precision |   brier_score |   log_loss |   calibration_intercept |   calibration_slope |   observed_expected_ratio |
|:----------------------------------|:--------------------------|-----------------------:|:--------------|:----------------|-------:|---------:|----------:|--------------------:|--------------:|-----------:|------------------------:|--------------------:|--------------------------:|
| AgeSexLogistic                    | age_sex                   |                      4 | none          | survey_weighted | 178120 |     2036 |  0.77022  |           0.0344283 |    0.00905922 |  0.0478423 |             -0.0293721  |            0.993596 |                  0.997336 |
| AgeSexLogistic                    | age_sex                   |                      4 | sigmoid       | survey_weighted | 178120 |     2036 |  0.770033 |           0.0344507 |    0.00905912 |  0.0478437 |              0.00947679 |            1.00197  |                  1.00123  |
| LogisticRegressionNoGeneralHealth | minimal_no_general_health |                     11 | none          | survey_weighted | 178120 |     2036 |  0.808838 |           0.0452893 |    0.00901108 |  0.0462737 |             -0.0398366  |            0.990604 |                  0.997824 |
| LogisticRegressionNoGeneralHealth | minimal_no_general_health |                     11 | sigmoid       | survey_weighted | 178120 |     2036 |  0.808787 |           0.0453147 |    0.00900997 |  0.0462744 |              0.0135549  |            1.00311  |                  1.00104  |
| LogisticRegression                | minimal                   |                     15 | none          | survey_weighted | 178120 |     2036 |  0.821598 |           0.0515827 |    0.00897883 |  0.045543  |             -0.0509779  |            0.98783  |                  0.996922 |
| LogisticRegression                | minimal                   |                     15 | sigmoid       | survey_weighted | 178120 |     2036 |  0.82156  |           0.0515941 |    0.00897681 |  0.0455431 |              0.0160432  |            1.00373  |                  1.00129  |

## Outer folds

|   fold | train_years                                            | validation_years   |   n_train |   events_train |   n_validation |   events_validation |
|-------:|:-------------------------------------------------------|:-------------------|----------:|---------------:|---------------:|--------------------:|
|      1 | 1998;1999;2000;2001;2002;2003;2004;2005;2006;2007;2008 | 1997;2009          |    148447 |           1658 |          29673 |                 378 |
|      2 | 1997;1998;1999;2000;2002;2004;2005;2006;2007;2009      | 2001;2003;2008     |    138627 |           1575 |          39493 |                 461 |
|      3 | 1997;1999;2000;2001;2002;2003;2004;2007;2008;2009      | 1998;2005;2006     |    137588 |           1624 |          40532 |                 412 |
|      4 | 1997;1998;1999;2001;2003;2004;2005;2006;2008;2009      | 2000;2002;2007     |    138206 |           1606 |          39914 |                 430 |
|      5 | 1997;1998;2000;2001;2002;2003;2005;2006;2007;2008;2009 | 1999;2004          |    149612 |           1681 |          28508 |                 355 |

## Development missingness

| feature         |      n |   n_missing |   missing_fraction_unweighted |   missing_fraction_weighted |
|:----------------|-------:|------------:|------------------------------:|----------------------------:|
| age             | 178120 |           0 |                   0           |                 0           |
| sex             | 178120 |           0 |                   0           |                 0           |
| bmi             | 178120 |        6939 |                   0.0389569   |                 0.0391103   |
| smoking_status  | 178120 |        1334 |                   0.00748933  |                 0.00750348  |
| hypertension_dx | 178120 |         165 |                   0.000926342 |                 0.000810591 |
| diabetes_dx     | 178120 |        2383 |                   0.0133786   |                 0.0127788   |
| general_health  | 178120 |         127 |                   0.000713002 |                 0.000676983 |

## Implausible-value audit

| check                       |   n_flagged |   n_total |
|:----------------------------|------------:|----------:|
| age_outside_protocol_40_79  |           0 |    178120 |
| bmi_below_future_input_10   |          10 |    178120 |
| bmi_above_future_input_99_8 |           0 |    178120 |
| non_finite_age              |           0 |    178120 |
| non_finite_bmi              |           0 |    178120 |

## Status

Baseline candidates were fitted reproducibly. Random Forest, boosting, final model selection, threshold policy, and external-validation lock remain pending.

Machine-readable provenance: `artifacts/stage4_baselines/manifest.json`
