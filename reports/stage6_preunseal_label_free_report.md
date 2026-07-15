# Stage 6: pre-unseal label-free predictions

## Result

The immutable sigmoid-calibrated minimal LogisticRegression pipeline was applied once to NHIS 2010–2014 and all eight NHANES 1999–2014 cycles. Only explicitly allowlisted predictor, identifier, and survey-design columns were deserialized. No held-out outcome, mortality, cause-of-death, or follow-up column was loaded or joined.

The primary output remains continuous five-year risk. The fixed cutoff is a secondary screening operating point, not a diagnosis or clinically endorsed high-risk boundary.

## Label-free prediction distribution

| cohort          |   n_nominal |   weight_sum |   weighted_prediction_mean |   weighted_prediction_sd |   weighted_lp_mean |   weighted_lp_sd |   prediction_min |   prediction_max |   weighted_screen_positive_fraction |
|:----------------|------------:|-------------:|---------------------------:|-------------------------:|-------------------:|-----------------:|-----------------:|-----------------:|------------------------------------:|
| nhis_temporal   |       82860 |  1.18467e+08 |                 0.00986276 |                0.016027  |           -5.29413 |          1.12771 |      0.000475555 |         0.426326 |                            0.337412 |
| nhanes_external |       20911 |  1.09261e+08 |                 0.010112   |                0.0165939 |           -5.26198 |          1.11704 |      0.000475582 |         0.376971 |                            0.344187 |

## Survey design effects

| cohort          | variable         |   n_nominal |   weighted_mean |   weighted_population_sd |   design_se |      srs_se |   design_effect |   design_df |
|:----------------|:-----------------|------------:|----------------:|-------------------------:|------------:|------------:|----------------:|------------:|
| nhis_temporal   | linear_predictor |       82860 |     -5.29413    |                1.12771   | 0.00610787  | 0.00391766  |         2.43068 |         300 |
| nhis_temporal   | prediction       |       82860 |      0.00986276 |                0.016027  | 7.46879e-05 | 5.56774e-05 |         1.79946 |         300 |
| nhanes_external | linear_predictor |       20911 |     -5.26198    |                1.11704   | 0.0144937   | 0.00772467  |         3.52048 |         123 |
| nhanes_external | prediction       |       20911 |      0.010112   |                0.0165939 | 0.000166429 | 0.000114752 |         2.10346 |         123 |

## Precision gate

- nhis_temporal: **FAIL / underpowered**; nominal N=82,860, DEFF gate=2.4307, effective N=34,089.3, required nominal N=99,862.
- nhanes_external: **FAIL / underpowered**; nominal N=20,911, DEFF gate=3.5205, effective N=5,939.8, required nominal N=144,636.

The pmvalsampsize prevalence argument remains the frozen survey-weighted NHIS development prevalence; held-out prevalence was not read. A failed gate does not permit model changes and marks subsequent validation as underpowered/exploratory with full confidence intervals.

Held-out outcomes remain sealed. This report does not authorize or perform unsealing.

Machine-readable provenance: `artifacts/stage6_label_free/manifest.json`
