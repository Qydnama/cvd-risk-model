# Stage 5a: provisional external-validation precision planning

## Scope

Planning uses only sigmoid-calibrated LogisticRegression minimal out-of-fold predictions from NHIS development 1997–2009. No temporal or NHANES outcome was loaded or used.

The required prevalence argument is the survey-weighted NHIS development five-year CVD mortality prevalence. This pre-lock value remains fixed for provisional planning and is not estimated from held-out labels.

## OOF linear-predictor distribution

- Weighted prevalence: 0.009271379395
- Weighted LP mean: -5.385179918012
- Weighted LP population SD: 1.132691145530
- Predictions clipped low/high: 0/0

## pmvalsampsize stability

| package       | package_version   | type   |   prevalence |   lp_mean |   lp_sd |   simobs |     seed |   criterion_oe_n |   criterion_slope_n |   criterion_cstatistic_n |   required_n |   required_events |   achieved_oe_ci_width |   achieved_slope_ci_width |   achieved_cstatistic_ci_width | raw_output_path                                                                                                     | raw_output_sha256                                                |   relative_difference_from_previous |
|:--------------|:------------------|:-------|-------------:|----------:|--------:|---------:|---------:|-----------------:|--------------------:|-------------------------:|-------------:|------------------:|-----------------------:|--------------------------:|-------------------------------:|:--------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------|------------------------------------:|
| pmvalsampsize | 0.1.0             | binary |   0.00927138 |  -5.38518 | 1.13269 |    1e+06 | 20260714 |            41084 |                9668 |                     9860 |        41084 |           380.905 |               0.200253 |                       0.4 |                            0.1 | C:\Users\amang\Desktop\Sabak\Codes\Python\Machine Learning\SSR\artifacts\stage5a_planning\pmvalsampsize_1000000.csv | 580180d0a79a0b700d422cb49d391f30498860ca0f4691215365d6d658d9429d |                                 nan |
| pmvalsampsize | 0.1.0             | binary |   0.00927138 |  -5.38518 | 1.13269 |    2e+06 | 20260714 |            41084 |                9652 |                     9860 |        41084 |           380.905 |               0.200253 |                       0.4 |                            0.1 | C:\Users\amang\Desktop\Sabak\Codes\Python\Machine Learning\SSR\artifacts\stage5a_planning\pmvalsampsize_2000000.csv | bbf1d5357f26356d89f30a111e083999d89279f614d637683b0bf5c518583403 |                                   0 |

Stability relative difference: 0.000000 (tolerance 1.00%).

Provisional IID/SRS-equivalent required N: **41,084**.

Survey-design correction is intentionally deferred to Stage 6, after frozen label-free held-out predictions provide DEFF_LP and DEFF_prediction.

No external-validation lock was created in this stage. Threshold policy and remaining pre-lock metadata must be frozen first.

Machine-readable provenance: `artifacts/stage5a_planning/manifest.json`
