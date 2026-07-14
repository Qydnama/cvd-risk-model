# Stage 3 feasibility report

Generated: 2026-07-14T16:14:19.553622Z

**PASS: all pre-specified feasibility checks passed.**

## Event counts

| cohort             |   n_participants |   n_events |   n_non_events |   event_prevalence_unweighted | weight_column   |   weight_sum |   weighted_event_sum |   event_prevalence_weighted |
|:-------------------|-----------------:|-----------:|---------------:|------------------------------:|:----------------|-------------:|---------------------:|----------------------------:|
| nhis_development   |           178120 |       2036 |         176084 |                     0.0114305 | analysis_weight |  1.02162e+08 |               947180 |                  0.00927138 |
| nhis_temporal_test |            82860 |        834 |          82026 |                     0.0100652 | analysis_weight |  1.18467e+08 |               999145 |                  0.00843398 |
| nhanes_external    |            20911 |        255 |          20656 |                     0.0121945 | analysis_weight |  1.09261e+08 |               907824 |                  0.00830878 |

## Stop-gate checks

| check                                                           | observed           | minimum   | status   |
|:----------------------------------------------------------------|:-------------------|:----------|:---------|
| events:nhis_development                                         | 2036.0             | 500       | pass     |
| events:nhis_temporal_test                                       | 834.0              | 200       | pass     |
| events:nhanes_external                                          | 255.0              | 100       | pass     |
| events_per_effective_df:nhis_development                        | 101.8              | 20        | pass     |
| risk_precision:nhis_development                                 | 0.0005351375787263 | 0.0025    | pass     |
| design_df:nhis_development                                      | 639.0              | 8.0       | pass     |
| pmsampsize_information:nhis_development                         | 130786.538519483   | 8934.0    | pass     |
| risk_precision:nhis_temporal_test                               | 0.0007700206758337 | 0.0025    | pass     |
| design_df:nhis_temporal_test                                    | 300.0              | 8.0       | pass     |
| risk_precision:nhanes_external                                  | 0.0015828529694421 | 0.0025    | pass     |
| design_df:nhanes_external                                       | 123.0              | 8.0       | pass     |
| published_strata_psu_pairs_repeated_across_cycles               | 0                  | 0         | pass     |
| automatic_cycle_interaction_forbidden                           | False              | False     | pass     |
| protocol_discrepancy:nhanes_pooled_weight_1999_2002             | resolved           | resolved  | pass     |
| protocol_discrepancy:nhis_congestive_heart_failure_availability | resolved           | resolved  | pass     |

## Data discrepancies and conditional audits

No stage-3 discrepancies were detected.

## NHIS–NHANES harmonization differences

| feature                   | nhis_measurement                                   | nhanes_measurement                                       | comparability                                                     | required_action                                                     |
|:--------------------------|:---------------------------------------------------|:---------------------------------------------------------|:------------------------------------------------------------------|:--------------------------------------------------------------------|
| age                       | Interview age (IPUMS AGE)                          | Interview age (RIDAGEYR)                                 | closely comparable after common 40–79 restriction                 | validate integer range and cycle/year coverage                      |
| sex                       | SEX                                                | RIAGENDR (official release label: Gender)                | binary male/female coding; wording is not explicitly sex at birth | document wording mismatch for the future site interface             |
| bmi                       | derived from self-reported height and weight       | measured MEC BMI (BMXBMI)                                | same unit, materially different measurement process               | report external-distribution shift and calibration                  |
| smoking_status            | harmonized SMOKESTATUS2                            | derived jointly from SMQ020 and SMQ040                   | common never/former/current categories                            | keep unknown question-universe responses missing                    |
| hypertension_dx           | self-reported prior diagnosis (HYPERTENEV)         | self-reported prior diagnosis (BPQ020)                   | questionnaire diagnosis; not measured blood pressure              | do not substitute BP measurements in fast model                     |
| diabetes_dx               | self-reported prior diagnosis (DIABETICEV)         | self-reported prior diagnosis (DIQ010)                   | borderline responses are not equivalent to No                     | retain borderline as a distinct missing reason                      |
| general_health            | HEALTH                                             | HUQ010 in every 1999–2014 cycle; HSD010 sensitivity-only | common ordered excellent→poor labels after cycle check            | report label-free HUQ/HSD disagreement; never backfill HUQ from HSD |
| race_ethnicity            | verified five-category bridge from RACEA + HISPETH | RIDRETH1 audit categories                                | audit-only collapsed bridge; historical classifications differ    | exclude from predictors and retain subgroup limitations             |
| outcome_timing            | interview/death year and quarter                   | PERMTH_INT exact integer months                          | same five-year estimand with different precision                  | NHIS quarter 20 excluded in primary, included in sensitivity        |
| survey_design             | MORTWTSA, STRATA, PSU; annual pooled weights       | WTMEC2YR, SDMVSTRA, SDMVPSU; eight MEC cycles            | source-specific design and scaling                                | never concatenate and apply one shared survey weight                |
| legacy_xpt_zero           | not applicable                                     | legacy SAS zero may decode near 5.397605e-79             | parser artifact, not a biological measurement                     | normalize machine-zero before age/weight eligibility checks         |
| extended_laboratory_files | not available                                      | 1999–2004 LAB13/L13 and LAB10/L10 legacy files           | NHANES-only extended analysis                                     | use cycle-aware HDL names and verify laboratory weights             |

## Blocking protocol discrepancies

| id                                         | title                                                                        | status   | blocking   | locked_protocol_rule                                                                                                                                                                               | official_guidance                                                                                                                                                                                                                                      | recommended_resolution                                                                                                                                                                                                                                                                                                                                                                        | resolution                                                                                                                                                                                                                                                                                                                                                                               | official_sources                                                                                                                                   |
|:-------------------------------------------|:-----------------------------------------------------------------------------|:---------|:-----------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------|
| nhanes_pooled_weight_1999_2002             | NHANES 1999-2002 pooled MEC weight scaling                                   | resolved | True       | Amendment 2.4 uses WTMEC4YR / 4 for 1999-2002 and WTMEC2YR / 8 for 2003-2014 in the pooled 1999-2014 NHANES estimand.                                                                              | CDC instructs analysts combining 1999-2000 with 2001-2002 to use the provided four-year weight because the two cycles used different census reference populations.                                                                                     | For a 16-year 1999-2014 pooled analysis, use WTMEC4YR / 4 for 1999-2002 and WTMEC2YR / 8 for 2003-2014, then update the locked protocol, implementation, and tests together before marking resolved.                                                                                                                                                                                          | Resolved by protocol amendment 2.4 on 2026-07-14. The experiment configuration, locked runtime contract, harmonization code and tests must use the CDC-compatible source-specific pooled-weight rules.                                                                                                                                                                                   | https://wwwn.cdc.gov/nchs/nhanes/tutorials/weighting.aspx | https://wwwn.cdc.gov/nchs/data/nhanes/analyticguidelines/05-06-analytic-guidelines.pdf |
| nhis_congestive_heart_failure_availability | NHIS congestive-heart-failure exclusion is unavailable in most primary years | resolved | True       | Amendment 2.4 requires exclusion of baseline congestive heart failure together with coronary heart disease, angina, myocardial infarction, and stroke for the NHIS 1997-2014 fixed-horizon cohort. | IPUMS NHIS documents CONHFAILEV as available only in 1999 and 2002. In the delivered extract it is NIU in every other year, so treating NIU as absence would silently misclassify baseline disease and treating it as unknown removes all other years. | For the cross-year primary NHIS cohort, use the four consistently measured exclusions (coronary heart disease, angina, myocardial infarction, and stroke). Retain CONHFAILEV only for a documented 1999/2002 sensitivity audit. An alternative is the broader HEARTCONEV surrogate, but it is not equivalent to congestive heart failure and would require a new IPUMS extract and amendment. | Resolved by explicit user approval on 2026-07-14. The primary NHIS and NHANES cohorts use the same four consistently measured exclusions: coronary heart disease, angina, myocardial infarction, and stroke. CONHFAILEV remains unknown outside 1999/2002 and is used only for an NHIS 1999/2002 CHF-strict sensitivity subset; MCQ160B is retained for the parallel NHANES sensitivity. | https://nhis.ipums.org/nhis-action/variables/170854                                                                                                |

## Race/ethnicity audit feasibility

| cohort             | subgroup       | category             | status                  |   n_participants | n_events   | n_non_events   |   n_missing_subgroup | weighted_event_prevalence   | eligible_for_performance_audit   |
|:-------------------|:---------------|:---------------------|:------------------------|-----------------:|:-----------|:---------------|---------------------:|:----------------------------|:---------------------------------|
| nhis_development   | race_ethnicity | mexican_american     | available               |            13318 | 132        | 13186          |                   35 | 0.009121082440791546        | True                             |
| nhis_development   | race_ethnicity | non_hispanic_black   | available               |            24491 | 416        | 24075          |                   35 | 0.01371946386407899         | True                             |
| nhis_development   | race_ethnicity | non_hispanic_white   | available               |           122791 | 1332       | 121459         |                   35 | 0.008856451973174589        | True                             |
| nhis_development   | race_ethnicity | other_hispanic       | available               |            10761 | 99         | 10662          |                   35 | 0.007322866895972334        | True                             |
| nhis_development   | race_ethnicity | other_or_multiracial | available               |             6724 | 56         | 6668           |                   35 | 0.00767620580388249         | True                             |
| nhis_temporal_test | race_ethnicity | mexican_american     | sealed_until_model_lock |             6993 | <NA>       | <NA>           |                    0 | <NA>                        | False                            |
| nhis_temporal_test | race_ethnicity | non_hispanic_black   | sealed_until_model_lock |            12430 | <NA>       | <NA>           |                    0 | <NA>                        | False                            |
| nhis_temporal_test | race_ethnicity | non_hispanic_white   | sealed_until_model_lock |            52766 | <NA>       | <NA>           |                    0 | <NA>                        | False                            |
| nhis_temporal_test | race_ethnicity | other_hispanic       | sealed_until_model_lock |             5298 | <NA>       | <NA>           |                    0 | <NA>                        | False                            |
| nhis_temporal_test | race_ethnicity | other_or_multiracial | sealed_until_model_lock |             5373 | <NA>       | <NA>           |                    0 | <NA>                        | False                            |
| nhanes_external    | race_ethnicity | mexican_american     | sealed_until_model_lock |             3768 | <NA>       | <NA>           |                    0 | <NA>                        | False                            |
| nhanes_external    | race_ethnicity | non_hispanic_black   | sealed_until_model_lock |             4590 | <NA>       | <NA>           |                    0 | <NA>                        | False                            |
| nhanes_external    | race_ethnicity | non_hispanic_white   | sealed_until_model_lock |             9470 | <NA>       | <NA>           |                    0 | <NA>                        | False                            |
| nhanes_external    | race_ethnicity | other_hispanic       | sealed_until_model_lock |             1632 | <NA>       | <NA>           |                    0 | <NA>                        | False                            |
| nhanes_external    | race_ethnicity | other_or_multiracial | sealed_until_model_lock |             1451 | <NA>       | <NA>           |                    0 | <NA>                        | False                            |

## CHF-strict baseline-CVD sensitivity subsets

| cohort             |   primary_n |   chf_available_n |   chf_present_n |   chf_unknown_n |   chf_strict_n | scope               |
|:-------------------|------------:|------------------:|----------------:|----------------:|---------------:|:--------------------|
| nhis_development   |      178120 |             28123 |             202 |              17 |          27904 | NHIS 1999/2002 only |
| nhis_temporal_test |       82860 |                 0 |               0 |               0 |              0 | NHIS 1999/2002 only |
| nhanes_external    |       20911 |             20911 |             293 |              20 |          20598 | NHANES 1999-2014    |

## R survey-design precision and effective sample size

| cohort             |   n_nominal |   raw_events |      p_hat |          se |     ci_low |    ci_high |   max_ci_half_width |   deff_y |   deff_y_gate |   n_eff_design |   n_eff_kish |   design_df |   expected_r2_cs |   pmsampsize_required_n |   required_n_nominal |   pass_information | pass_risk_precision   | pass_design_df   |
|:-------------------|------------:|-------------:|-----------:|------------:|-----------:|-----------:|--------------------:|---------:|--------------:|---------------:|-------------:|------------:|-----------------:|------------------------:|---------------------:|-------------------:|:----------------------|:-----------------|
| nhis_development   |      178120 |         2036 | 0.00927138 | 0.000265014 | 0.00876519 | 0.00980652 |         0.000535138 |  1.36191 |       1.36191 |       130787   |     124416   |         639 |        0.0149853 |                    8934 |                12168 |                  1 | True                  | True             |
| nhis_temporal_test |       82860 |          834 | 0.00843398 | 0.000374589 | 0.00772787 | 0.009204   |         0.000770021 |  1.39027 |       1.39027 |        59599.8 |      51025.8 |         300 |      nan         |                     nan |                  nan |                nan | True                  | True             |
| nhanes_external    |       20911 |          255 | 0.00830878 | 0.000732521 | 0.00697743 | 0.00989163 |         0.00158285  |  1.36176 |       1.36176 |        15355.9 |      12194.7 |         123 |      nan         |                     nan |                  nan |                nan | True                  | True             |

## NHANES design collision audit

| check                                             |   observed |   minimum | status   | detail                                                                            |
|:--------------------------------------------------|-----------:|----------:|:---------|:----------------------------------------------------------------------------------|
| published_strata_psu_pairs_repeated_across_cycles |          0 |         0 | pass     | Published (SDMVSTRA, SDMVPSU) pairs are used as released; cycle was not appended. |
| automatic_cycle_interaction_forbidden             |      False |     False | pass     | No cycle interaction was added to strata or PSU identifiers.                      |

## Label-free HUQ010/HSD010 disagreement audit

| survey_cycle   | huq_category   | hsd_category   |   n_paired |       weighted_n | disagreement   | status                 |
|:---------------|:---------------|:---------------|-----------:|-----------------:|:---------------|:-----------------------|
| 1999-2000      | <NA>           | <NA>           |          0 |      0           | <NA>           | EXPECTED_NOT_AVAILABLE |
| 2001-2002      | excellent      | excellent      |        202 |      1.51617e+06 | False          | AVAILABLE              |
| 2001-2002      | excellent      | fair           |         13 |  39995.1         | True           | AVAILABLE              |
| 2001-2002      | excellent      | good           |         47 | 185492           | True           | AVAILABLE              |
| 2001-2002      | excellent      | poor           |          2 |  10400.8         | True           | AVAILABLE              |
| 2001-2002      | excellent      | very_good      |        154 |      1.0602e+06  | True           | AVAILABLE              |
| 2001-2002      | fair           | excellent      |          4 |  11662.6         | True           | AVAILABLE              |
| 2001-2002      | fair           | fair           |        255 | 941016           | False          | AVAILABLE              |
| 2001-2002      | fair           | good           |        105 | 442479           | True           | AVAILABLE              |
| 2001-2002      | fair           | poor           |         20 |  98132           | True           | AVAILABLE              |
| 2001-2002      | fair           | very_good      |         12 |  47334.6         | True           | AVAILABLE              |
| 2001-2002      | good           | excellent      |         28 | 126295           | True           | AVAILABLE              |
| 2001-2002      | good           | fair           |         92 | 341698           | True           | AVAILABLE              |
| 2001-2002      | good           | good           |        470 |      2.53676e+06 | False          | AVAILABLE              |
| 2001-2002      | good           | poor           |          5 |  20355.2         | True           | AVAILABLE              |
| 2001-2002      | good           | very_good      |        137 | 840935           | True           | AVAILABLE              |
| 2001-2002      | poor           | excellent      |          1 |    945.612       | True           | AVAILABLE              |
| 2001-2002      | poor           | fair           |         28 | 124559           | True           | AVAILABLE              |
| 2001-2002      | poor           | good           |         13 |  42798.7         | True           | AVAILABLE              |
| 2001-2002      | poor           | poor           |         42 | 170021           | False          | AVAILABLE              |
| 2001-2002      | poor           | very_good      |          2 |   8741.73        | True           | AVAILABLE              |
| 2001-2002      | very_good      | excellent      |         56 | 411868           | True           | AVAILABLE              |
| 2001-2002      | very_good      | fair           |         21 |  79698.6         | True           | AVAILABLE              |
| 2001-2002      | very_good      | good           |        174 | 980572           | True           | AVAILABLE              |
| 2001-2002      | very_good      | poor           |          7 |  12264.1         | True           | AVAILABLE              |
| 2001-2002      | very_good      | very_good      |        381 |      2.78946e+06 | False          | AVAILABLE              |
| 2003-2004      | excellent      | excellent      |        163 |      1.16963e+06 | False          | AVAILABLE              |
| 2003-2004      | excellent      | fair           |         15 |  71752.8         | True           | AVAILABLE              |
| 2003-2004      | excellent      | good           |         58 | 283159           | True           | AVAILABLE              |
| 2003-2004      | excellent      | very_good      |        113 | 852066           | True           | AVAILABLE              |
| 2003-2004      | fair           | excellent      |          3 |   5825.32        | True           | AVAILABLE              |
| 2003-2004      | fair           | fair           |        250 | 950575           | False          | AVAILABLE              |
| 2003-2004      | fair           | good           |        112 | 487184           | True           | AVAILABLE              |
| 2003-2004      | fair           | poor           |         15 |  66798.8         | True           | AVAILABLE              |
| 2003-2004      | fair           | very_good      |         16 |  58545.7         | True           | AVAILABLE              |
| 2003-2004      | good           | excellent      |         20 |  78269.1         | True           | AVAILABLE              |
| 2003-2004      | good           | fair           |        116 | 430301           | True           | AVAILABLE              |
| 2003-2004      | good           | good           |        394 |      2.36452e+06 | False          | AVAILABLE              |
| 2003-2004      | good           | poor           |          6 |  22480.1         | True           | AVAILABLE              |
| 2003-2004      | good           | very_good      |        138 | 880259           | True           | AVAILABLE              |
| 2003-2004      | poor           | excellent      |          1 |   8705.86        | True           | AVAILABLE              |
| 2003-2004      | poor           | fair           |         43 | 210448           | True           | AVAILABLE              |
| 2003-2004      | poor           | good           |          6 |  28118.1         | True           | AVAILABLE              |
| 2003-2004      | poor           | poor           |         56 | 259430           | False          | AVAILABLE              |
| 2003-2004      | poor           | very_good      |          2 |  13605.9         | True           | AVAILABLE              |
| 2003-2004      | very_good      | excellent      |         38 | 238504           | True           | AVAILABLE              |
| 2003-2004      | very_good      | fair           |         23 |  80283.7         | True           | AVAILABLE              |
| 2003-2004      | very_good      | good           |        168 | 976276           | True           | AVAILABLE              |
| 2003-2004      | very_good      | poor           |          2 |   6526.89        | True           | AVAILABLE              |
| 2003-2004      | very_good      | very_good      |        295 |      2.30381e+06 | False          | AVAILABLE              |
| 2005-2006      | excellent      | excellent      |        141 |      1.10602e+06 | False          | AVAILABLE              |
| 2005-2006      | excellent      | fair           |          6 |  13466.8         | True           | AVAILABLE              |
| 2005-2006      | excellent      | good           |         39 | 199429           | True           | AVAILABLE              |
| 2005-2006      | excellent      | poor           |          1 |   1213.29        | True           | AVAILABLE              |
| 2005-2006      | excellent      | very_good      |        121 | 886596           | True           | AVAILABLE              |
| 2005-2006      | fair           | excellent      |          1 |   2722.21        | True           | AVAILABLE              |
| 2005-2006      | fair           | fair           |        229 | 928751           | False          | AVAILABLE              |
| 2005-2006      | fair           | good           |        112 | 505182           | True           | AVAILABLE              |
| 2005-2006      | fair           | poor           |         15 |  60239.5         | True           | AVAILABLE              |
| 2005-2006      | fair           | very_good      |         13 |  51446.3         | True           | AVAILABLE              |
| 2005-2006      | good           | excellent      |         27 | 116883           | True           | AVAILABLE              |
| 2005-2006      | good           | fair           |        110 | 567020           | True           | AVAILABLE              |
| 2005-2006      | good           | good           |        480 |      2.98138e+06 | False          | AVAILABLE              |
| 2005-2006      | good           | poor           |          5 |  31591.7         | True           | AVAILABLE              |
| 2005-2006      | good           | very_good      |        130 | 804179           | True           | AVAILABLE              |
| 2005-2006      | poor           | fair           |         20 |  82047.7         | True           | AVAILABLE              |
| 2005-2006      | poor           | good           |          5 |  21827.1         | True           | AVAILABLE              |
| 2005-2006      | poor           | poor           |         43 | 196162           | False          | AVAILABLE              |
| 2005-2006      | very_good      | excellent      |         33 | 221699           | True           | AVAILABLE              |
| 2005-2006      | very_good      | fair           |         15 |  52406.5         | True           | AVAILABLE              |
| 2005-2006      | very_good      | good           |        179 |      1.1327e+06  | True           | AVAILABLE              |
| 2005-2006      | very_good      | very_good      |        344 |      2.67202e+06 | False          | AVAILABLE              |
| 2007-2008      | excellent      | excellent      |        143 | 873175           | False          | AVAILABLE              |
| 2007-2008      | excellent      | fair           |         13 |  48156.8         | True           | AVAILABLE              |
| 2007-2008      | excellent      | good           |         54 | 215505           | True           | AVAILABLE              |
| 2007-2008      | excellent      | very_good      |        125 | 757959           | True           | AVAILABLE              |
| 2007-2008      | fair           | excellent      |          6 |   6867.46        | True           | AVAILABLE              |
| 2007-2008      | fair           | fair           |        370 |      1.13039e+06 | False          | AVAILABLE              |
| 2007-2008      | fair           | good           |        152 | 591276           | True           | AVAILABLE              |
| 2007-2008      | fair           | poor           |         23 |  94012           | True           | AVAILABLE              |
| 2007-2008      | fair           | very_good      |         27 |  80678.9         | True           | AVAILABLE              |
| 2007-2008      | good           | excellent      |         24 |  68954.9         | True           | AVAILABLE              |
| 2007-2008      | good           | fair           |        125 | 380093           | True           | AVAILABLE              |
| 2007-2008      | good           | good           |        635 |      2.93171e+06 | False          | AVAILABLE              |
| 2007-2008      | good           | poor           |          2 |   2604.77        | True           | AVAILABLE              |
| 2007-2008      | good           | very_good      |        189 |      1.03041e+06 | True           | AVAILABLE              |
| 2007-2008      | poor           | excellent      |          1 |    923.753       | True           | AVAILABLE              |
| 2007-2008      | poor           | fair           |         64 | 157504           | True           | AVAILABLE              |
| 2007-2008      | poor           | good           |         19 |  52636.3         | True           | AVAILABLE              |
| 2007-2008      | poor           | poor           |         69 | 224824           | False          | AVAILABLE              |
| 2007-2008      | poor           | very_good      |          2 |  13853.4         | True           | AVAILABLE              |
| 2007-2008      | very_good      | excellent      |         61 | 351864           | True           | AVAILABLE              |
| 2007-2008      | very_good      | fair           |         28 | 104201           | True           | AVAILABLE              |
| 2007-2008      | very_good      | good           |        239 |      1.20412e+06 | True           | AVAILABLE              |
| 2007-2008      | very_good      | poor           |          2 |   5435.82        | True           | AVAILABLE              |
| 2007-2008      | very_good      | very_good      |        408 |      2.5222e+06  | False          | AVAILABLE              |
| 2009-2010      | excellent      | excellent      |        177 |      1.11513e+06 | False          | AVAILABLE              |
| 2009-2010      | excellent      | fair           |          9 |  22751.9         | True           | AVAILABLE              |
| 2009-2010      | excellent      | good           |         50 | 209769           | True           | AVAILABLE              |
| 2009-2010      | excellent      | poor           |          1 |  13263.1         | True           | AVAILABLE              |
| 2009-2010      | excellent      | very_good      |        117 | 757259           | True           | AVAILABLE              |
| 2009-2010      | fair           | excellent      |         11 |  28171.7         | True           | AVAILABLE              |
| 2009-2010      | fair           | fair           |        370 |      1.06262e+06 | False          | AVAILABLE              |
| 2009-2010      | fair           | good           |        160 | 511724           | True           | AVAILABLE              |
| 2009-2010      | fair           | poor           |         41 | 106484           | True           | AVAILABLE              |
| 2009-2010      | fair           | very_good      |         23 |  93502.1         | True           | AVAILABLE              |
| 2009-2010      | good           | excellent      |         38 | 135926           | True           | AVAILABLE              |
| 2009-2010      | good           | fair           |        172 | 602010           | True           | AVAILABLE              |
| 2009-2010      | good           | good           |        663 |      2.83128e+06 | False          | AVAILABLE              |
| 2009-2010      | good           | poor           |          7 |  14512           | True           | AVAILABLE              |
| 2009-2010      | good           | very_good      |        158 | 853091           | True           | AVAILABLE              |
| 2009-2010      | poor           | excellent      |          2 |  20007.2         | True           | AVAILABLE              |
| 2009-2010      | poor           | fair           |         48 | 130385           | True           | AVAILABLE              |
| 2009-2010      | poor           | good           |         15 |  39539           | True           | AVAILABLE              |
| 2009-2010      | poor           | poor           |         66 | 212934           | False          | AVAILABLE              |
| 2009-2010      | very_good      | excellent      |         51 | 233508           | True           | AVAILABLE              |
| 2009-2010      | very_good      | fair           |         21 | 112769           | True           | AVAILABLE              |
| 2009-2010      | very_good      | good           |        224 |      1.11266e+06 | True           | AVAILABLE              |
| 2009-2010      | very_good      | poor           |          4 |  21438.1         | True           | AVAILABLE              |
| 2009-2010      | very_good      | very_good      |        409 |      2.76874e+06 | False          | AVAILABLE              |
| 2011-2012      | excellent      | excellent      |        132 |      1.07004e+06 | False          | AVAILABLE              |
| 2011-2012      | excellent      | fair           |          7 |  15543.1         | True           | AVAILABLE              |
| 2011-2012      | excellent      | good           |         35 | 137987           | True           | AVAILABLE              |
| 2011-2012      | excellent      | poor           |          1 |   1650.57        | True           | AVAILABLE              |
| 2011-2012      | excellent      | very_good      |        102 | 772629           | True           | AVAILABLE              |
| 2011-2012      | fair           | excellent      |         11 |  28469.2         | True           | AVAILABLE              |
| 2011-2012      | fair           | fair           |        293 |      1.16195e+06 | False          | AVAILABLE              |
| 2011-2012      | fair           | good           |        152 | 643273           | True           | AVAILABLE              |
| 2011-2012      | fair           | poor           |         19 |  85122.1         | True           | AVAILABLE              |
| 2011-2012      | fair           | very_good      |         16 |  95028.4         | True           | AVAILABLE              |
| 2011-2012      | good           | excellent      |         32 | 134301           | True           | AVAILABLE              |
| 2011-2012      | good           | fair           |        129 | 527647           | True           | AVAILABLE              |
| 2011-2012      | good           | good           |        618 |      3.38215e+06 | False          | AVAILABLE              |
| 2011-2012      | good           | poor           |         11 |  34465.8         | True           | AVAILABLE              |
| 2011-2012      | good           | very_good      |        157 | 808315           | True           | AVAILABLE              |
| 2011-2012      | poor           | excellent      |          1 |   3115.3         | True           | AVAILABLE              |
| 2011-2012      | poor           | fair           |         45 | 142537           | True           | AVAILABLE              |
| 2011-2012      | poor           | good           |          9 |  34932.6         | True           | AVAILABLE              |
| 2011-2012      | poor           | poor           |         57 | 195224           | False          | AVAILABLE              |
| 2011-2012      | poor           | very_good      |          1 |  11514.9         | True           | AVAILABLE              |
| 2011-2012      | very_good      | excellent      |         49 | 280149           | True           | AVAILABLE              |
| 2011-2012      | very_good      | fair           |         14 |  53656.3         | True           | AVAILABLE              |
| 2011-2012      | very_good      | good           |        182 |      1.08791e+06 | True           | AVAILABLE              |
| 2011-2012      | very_good      | very_good      |        371 |      2.88437e+06 | False          | AVAILABLE              |
| 2013-2014      | excellent      | excellent      |        152 | 896378           | False          | AVAILABLE              |
| 2013-2014      | excellent      | fair           |         10 |  36029.1         | True           | AVAILABLE              |
| 2013-2014      | excellent      | good           |         42 | 216080           | True           | AVAILABLE              |
| 2013-2014      | excellent      | poor           |          1 |   1863.88        | True           | AVAILABLE              |
| 2013-2014      | excellent      | very_good      |        114 | 810084           | True           | AVAILABLE              |
| 2013-2014      | fair           | excellent      |          6 |  23890.9         | True           | AVAILABLE              |
| 2013-2014      | fair           | fair           |        361 |      1.42776e+06 | False          | AVAILABLE              |
| 2013-2014      | fair           | good           |        145 | 593436           | True           | AVAILABLE              |
| 2013-2014      | fair           | poor           |         35 | 128492           | True           | AVAILABLE              |
| 2013-2014      | fair           | very_good      |         21 |  75826.5         | True           | AVAILABLE              |
| 2013-2014      | good           | excellent      |         27 | 117921           | True           | AVAILABLE              |
| 2013-2014      | good           | fair           |        158 | 619627           | True           | AVAILABLE              |
| 2013-2014      | good           | good           |        681 |      3.31386e+06 | False          | AVAILABLE              |
| 2013-2014      | good           | poor           |          5 |  12243.5         | True           | AVAILABLE              |
| 2013-2014      | good           | very_good      |        157 | 808786           | True           | AVAILABLE              |
| 2013-2014      | poor           | fair           |         39 | 182061           | True           | AVAILABLE              |
| 2013-2014      | poor           | good           |          9 |  37302.6         | True           | AVAILABLE              |
| 2013-2014      | poor           | poor           |         61 | 207881           | False          | AVAILABLE              |
| 2013-2014      | very_good      | excellent      |         50 | 349583           | True           | AVAILABLE              |
| 2013-2014      | very_good      | fair           |         22 | 103402           | True           | AVAILABLE              |
| 2013-2014      | very_good      | good           |        247 |      1.54638e+06 | True           | AVAILABLE              |
| 2013-2014      | very_good      | poor           |          4 |  22599           | True           | AVAILABLE              |
| 2013-2014      | very_good      | very_good      |        388 |      2.70525e+06 | False          | AVAILABLE              |

Temporal and external outcomes were inspected only for the pre-specified feasibility audit. They were not used for model or threshold selection.

No age range, prediction horizon, outcome definition, or train/test boundary was changed. The explicitly approved CHF sensitivity amendment is recorded in the protocol discrepancy registry.
