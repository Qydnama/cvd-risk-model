# Stage 5: Random Forest and XGBoost development comparison

## Scope

Only NHIS development 1997–2009 was used. Hyperparameters were selected inside each outer fold by three-fold grouped tuning on survey-weighted Brier score. Sigmoid calibration used inner out-of-fold predictions.

NHIS temporal and NHANES individual outcomes were not loaded. No final model was selected and no external-validation lock was created.

## Survey-weighted sigmoid OOF comparison

| model                             | route                     | predictor_parameters   | calibration   | weighting       |      n |   events |   roc_auc |   average_precision |   brier_score |   log_loss |   calibration_intercept |   calibration_slope |   observed_expected_ratio | model_family   | candidate_status                 |
|:----------------------------------|:--------------------------|:-----------------------|:--------------|:----------------|-------:|---------:|----------:|--------------------:|--------------:|-----------:|------------------------:|--------------------:|--------------------------:|:---------------|:---------------------------------|
| LogisticRegression                | minimal                   | 15                     | sigmoid       | survey_weighted | 178120 |     2036 |  0.82156  |           0.0515941 |    0.00897681 |  0.0455431 |              0.0160432  |             1.00373 |                   1.00129 | logistic       | development_candidate_not_locked |
| XGBoost                           | minimal                   | <NA>                   | sigmoid       | survey_weighted | 178120 |     2036 |  0.819088 |           0.0508778 |    0.00897913 |  0.0456774 |              0.108995   |             1.02556 |                   1.00726 | tree_ml        | development_candidate_not_locked |
| RandomForest                      | minimal                   | <NA>                   | sigmoid       | survey_weighted | 178120 |     2036 |  0.813786 |           0.0466259 |    0.00899616 |  0.0460019 |              0.127562   |             1.02971 |                   1.00866 | tree_ml        | development_candidate_not_locked |
| LogisticRegressionNoGeneralHealth | minimal_no_general_health | 11                     | sigmoid       | survey_weighted | 178120 |     2036 |  0.808787 |           0.0453147 |    0.00900997 |  0.0462744 |              0.0135549  |             1.00311 |                   1.00104 | logistic       | development_candidate_not_locked |
| AgeSexLogistic                    | age_sex                   | 4                      | sigmoid       | survey_weighted | 178120 |     2036 |  0.770033 |           0.0344507 |    0.00905912 |  0.0478437 |              0.00947679 |             1.00197 |                   1.00123 | logistic       | development_candidate_not_locked |

## Fold stability

| model                             |   brier_mean |    brier_sd |   auc_mean |     auc_sd |
|:----------------------------------|-------------:|------------:|-----------:|-----------:|
| AgeSexLogistic                    |   0.00916465 | 0.000873468 |   0.772151 | 0.0100514  |
| LogisticRegression                |   0.00907623 | 0.000827115 |   0.822661 | 0.00594336 |
| LogisticRegressionNoGeneralHealth |   0.00911178 | 0.000853999 |   0.810633 | 0.0095657  |
| RandomForest                      |   0.00909728 | 0.000855403 |   0.815708 | 0.00977506 |
| XGBoost                           |   0.0090786  | 0.000839121 |   0.820603 | 0.00795841 |

## Hyperparameters selected for full-development candidate artifacts

| model        | candidate_id       | parameters                                                                                                                                                                                                                                |
|:-------------|:-------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| RandomForest | rf_depth10_leaf100 | {"bootstrap":true,"criterion":"log_loss","max_depth":10,"max_features":"sqrt","min_samples_leaf":100,"n_estimators":300,"n_jobs":1}                                                                                                       |
| XGBoost      | xgb_depth2_child20 | {"colsample_bytree":1.0,"eval_metric":"logloss","learning_rate":0.05,"max_depth":2,"min_child_weight":20,"n_estimators":300,"n_jobs":1,"objective":"binary:logistic","reg_lambda":1.0,"subsample":1.0,"tree_method":"hist","verbosity":0} |

## Interpretation

These are development comparisons, not temporal or external validation. The lowest development Brier score does not automatically authorize choosing a final model; discrimination, calibration, stability, and simplicity must be reviewed before Stage 5a and model lock.

Machine-readable provenance: `artifacts/stage5_ml_candidates/manifest.json`
