# Stage 5b: pre-lock secondary screening operating point

## Interpretation

The model's primary output remains continuous five-year CVD mortality risk. This cutoff is a secondary screening operating point; it is not a diagnosis or a clinically endorsed high-risk boundary.

## Fixed development OOF cutoff

- Selection rule: maximum cutoff with survey-weighted sensitivity >= 0.80.
- Exact cutoff: `0.007469058716`.
- Survey-weighted sensitivity: 0.800235.
- Next higher cutoff: `0.007469092542` (sensitivity 0.799843, below target).
- Cutoff is fixed in every bootstrap, fold, year, and DCA calculation.

## Overall metrics with stratified PSU cluster-bootstrap 95% CI

| metric                    |      estimate |      ci_lower |      ci_upper |   confidence_level |
|:--------------------------|--------------:|--------------:|--------------:|-------------------:|
| sensitivity               |    0.800235   |    0.784211   |    0.816748   |               0.95 |
| specificity               |    0.697255   |    0.695052   |    0.699502   |               0.95 |
| ppv                       |    0.0241389  |    0.0231181  |    0.0252078  |               0.95 |
| npv                       |    0.997326   |    0.997088   |    0.997577   |               0.95 |
| flagged_fraction          |    0.307357   |    0.30511    |    0.309507   |               0.95 |
| prevalence                |    0.00927138 |    0.00889339 |    0.00964076 |               0.95 |
| tp_per_10000              |   74.1928     |   71.0479     |   77.5572     |               0.95 |
| fn_per_10000              |   18.521      |   16.7957     |   20.1793     |               0.95 |
| fp_per_10000              | 2999.38       | 2976.83       | 3021.32       |               0.95 |
| tn_per_10000              | 6907.91       | 6886.05       | 6930.49       |               0.95 |
| net_benefit_model         |    0.00516217 |    0.00484901 |    0.00550072 |               0.95 |
| net_benefit_treat_all     |    0.00181588 |    0.00143505 |    0.00218804 |               0.95 |
| net_benefit_treat_none    |    0          |    0          |    0          |               0.95 |
| net_benefit_vs_treat_all  |    0.00334628 |    0.00317851 |    0.00352115 |               0.95 |
| net_benefit_vs_treat_none |    0.00516217 |    0.00484901 |    0.00550072 |               0.95 |

## Survey-weighted confusion matrix per 10,000

| actual_outcome   |   screen negative |   screen positive |
|:-----------------|------------------:|------------------:|
| CVD death        |            18.521 |           74.1928 |
| no CVD death     |          6907.91  |         2999.38   |

Cell-specific intervals are available in the machine-readable confusion table.

## Decision curve at the fixed cutoff

| strategy               |     cutoff |   net_benefit |   ci_lower |   ci_upper |   confidence_level |
|:-----------------------|-----------:|--------------:|-----------:|-----------:|-------------------:|
| model                  | 0.00746906 |    0.00516217 | 0.00484901 | 0.00550072 |               0.95 |
| treat_all              | 0.00746906 |    0.00181588 | 0.00143505 | 0.00218804 |               0.95 |
| treat_none             | 0.00746906 |    0          | 0          | 0          |               0.95 |
| model_minus_treat_all  | 0.00746906 |    0.00334628 | 0.00317851 | 0.00352115 |               0.95 |
| model_minus_treat_none | 0.00746906 |    0.00516217 | 0.00484901 | 0.00550072 |               0.95 |

## Uncalibrated LogisticRegression versus sigmoid

| metric                           |   uncalibrated |    sigmoid |   delta_sigmoid_minus_uncalibrated |   delta_ci_lower |   delta_ci_upper | decision                                         | passed   |
|:---------------------------------|---------------:|-----------:|-----------------------------------:|-----------------:|-----------------:|:-------------------------------------------------|:---------|
| brier_score                      |     0.00897883 | 0.00897681 |                       -2.01683e-06 |     -3.02557e-06 |     -1.04116e-06 | improved_paired_cluster_bootstrap_ci_below_zero  | True     |
| log_loss                         |     0.045543   | 0.0455431  |                        3.67477e-08 |     -6.61226e-06 |      6.66575e-06 | numerical_tie_no_bootstrap_evidence_of_worsening | True     |
| absolute_calibration_intercept   |     0.0509779  | 0.0160432  |                       -0.0349347   |    nan           |    nan           | absolute_calibration_error_improved              | True     |
| absolute_calibration_slope_error |     0.0121695  | 0.00373052 |                       -0.00843898  |    nan           |    nan           | absolute_calibration_error_improved              | True     |
| absolute_observed_expected_error |     0.00307841 | 0.00129303 |                       -0.00178538  |    nan           |    nan           | absolute_calibration_error_improved              | True     |

The comparison already existed in the b92f921 source tree. Sigmoid improves weighted Brier with a paired cluster-bootstrap interval below zero. The log-loss difference is 3.67e-08 and its paired interval spans zero, so it is treated as a numerical tie rather than claimed as an improvement. Calibration intercept, slope error, and O/E error all improve.

## Fold stability

|   fold |     n |   events |   sensitivity |   specificity |       ppv |      npv |   flagged_fraction |   net_benefit_model |
|-------:|------:|---------:|--------------:|--------------:|----------:|---------:|-------------------:|--------------------:|
|      1 | 29673 |      378 |      0.81316  |      0.684393 | 0.025344  | 0.997252 |           0.320579 |          0.00577346 |
|      2 | 39493 |      461 |      0.781302 |      0.707122 | 0.0261017 | 0.996902 |           0.297736 |          0.00558935 |
|      3 | 40532 |      412 |      0.795078 |      0.693546 | 0.0216185 | 0.99749  |           0.310581 |          0.0044276  |
|      4 | 39914 |      430 |      0.820411 |      0.69338  | 0.0219946 | 0.997828 |           0.310903 |          0.00455003 |
|      5 | 28508 |      355 |      0.797622 |      0.706485 | 0.0272151 | 0.99706  |           0.298652 |          0.00594156 |

## Year stability

|   survey_year |     n |   events |   sensitivity |   specificity |       ppv |      npv |   flagged_fraction |   net_benefit_model |
|--------------:|------:|---------:|--------------:|--------------:|----------:|---------:|-------------------:|--------------------:|
|          1997 | 15994 |      248 |      0.846906 |      0.694864 | 0.0347295 | 0.997152 |           0.312069 |          0.00857116 |
|          1998 | 14479 |      194 |      0.835917 |      0.693531 | 0.0298915 | 0.997334 |           0.312383 |          0.00705712 |
|          1999 | 14117 |      219 |      0.805369 |      0.708055 | 0.0363038 | 0.99626  |           0.298862 |          0.00868245 |
|          2000 | 14612 |      174 |      0.790721 |      0.699595 | 0.0248021 | 0.997118 |           0.305097 |          0.00532806 |
|          2001 | 15044 |      179 |      0.842698 |      0.711266 | 0.0282557 | 0.997802 |           0.294199 |          0.00616143 |
|          2002 | 14006 |      171 |      0.831121 |      0.698824 | 0.0279351 | 0.99749  |           0.306637 |          0.00632287 |
|          2003 | 13838 |      164 |      0.769173 |      0.718506 | 0.0268702 | 0.996764 |           0.286373 |          0.00559776 |
|          2004 | 14391 |      136 |      0.784545 |      0.705074 | 0.0189804 | 0.997782 |           0.298461 |          0.00346155 |
|          2005 | 14561 |      120 |      0.737238 |      0.695389 | 0.015434  | 0.997559 |           0.307395 |          0.00246679 |
|          2006 | 11492 |       98 |      0.792599 |      0.691746 | 0.0206008 | 0.997553 |           0.312184 |          0.00413037 |
|          2007 | 11296 |       85 |      0.847415 |      0.683103 | 0.0145263 | 0.99877  |           0.319806 |          0.00227392 |
|          2008 | 10611 |      118 |      0.739739 |      0.692928 | 0.0236733 | 0.996234 |           0.311384 |          0.00508371 |
|          2009 | 13679 |      130 |      0.769329 |      0.67617  | 0.0182806 | 0.997333 |           0.327295 |          0.0035652  |

Fold/year estimates are descriptive stability checks. The cutoff was not reselected within any fold or year.

## CHF amendment lineage

| check                                            | passed   | evidence                                                                                                                                                                                                                                                                                                                                                                                 |
|:-------------------------------------------------|:---------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| chf_amendment_resolved                           | True     | Resolved by explicit user approval on 2026-07-14. The primary NHIS and NHANES cohorts use the same four consistently measured exclusions: coronary heart disease, angina, myocardial infarction, and stroke. CONHFAILEV remains unknown outside 1999/2002 and is used only for an NHIS 1999/2002 CHF-strict sensitivity subset; MCQ160B is retained for the parallel NHANES sensitivity. |
| primary_and_sensitivity_rules_locked             | True     | four-condition primary exclusion; CHF sensitivity only in 1999/2002                                                                                                                                                                                                                                                                                                                      |
| cohort_contains_chf_amendment_fields             | True     | baseline_cvd_status_with_chf, chf_sensitivity_available, cohort_chf_sensitivity_eligible, congestive_heart_failure                                                                                                                                                                                                                                                                       |
| cohort_hash_bound_after_amendment                | True     | f1d1fd3fac65353bbbac64aaa07b7d9e9c1e35f1a81e98708dfe1dafd1c1380b                                                                                                                                                                                                                                                                                                                         |
| oof_hash_matches_stage4_manifest                 | True     | cd8817d7a818c02914777f8288b4901ffbc7f19d5e2d33e331c7577def50d386                                                                                                                                                                                                                                                                                                                         |
| oof_rows_match_post_amendment_development_cohort | True     | cohort_n=178120; oof_n=178120                                                                                                                                                                                                                                                                                                                                                            |
| stage4_based_on_chf_amendment_commit             | True     | stage3/base=aeb95c093e9f2546faa6edbe2ca8a98d2a04a5fa; stage4_commit=a486317de194a99ce16fa65ba3b40c63fda458ba                                                                                                                                                                                                                                                                             |
| stage4_commit_follows_stage3_amendment           | True     | aeb95c093e9f2546faa6edbe2ca8a98d2a04a5fa -> a486317de194a99ce16fa65ba3b40c63fda458ba                                                                                                                                                                                                                                                                                                     |

All lineage checks passed. The cohort hash used by Stage 3, Stage 4 and Stage 5 is identical, the cohort contains the CHF sensitivity fields, and the selected OOF rows exactly match the post-amendment development cohort.

PPV: 0.024139; NPV: 0.997326.

No held-out outcome was loaded or used. No model lock was created by this audit.

Machine-readable provenance: `artifacts/stage5b_operating_point/manifest.json`
