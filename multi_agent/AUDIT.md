# Phase 0 audit

- The root research project and `uv.lock` are immutable for this addition. The Stage 9 release manifest binds Python 3.12, joblib 1.5.3, NumPy 2.5.1, pandas 2.3.3, and scikit-learn 1.9.0.
- `validate_stage9_research_release(root=...)` verifies locked binary SHA-256 values, schema and documentation bindings, Git provenance, authorization, governance, and a real synthetic smoke prediction before use.
- The serialized `FittedBaselineModel` contains fitted preprocessing, estimator, sigmoid calibrator, and `probability_clip=1e-6`. Its `predict_proba(frame, calibrated=True)` transforms the seven ordered columns, obtains the estimator probability, clips it **for the logit supplied to the fitted calibrator**, then returns the calibrated probability. The separate calibrator must not be applied again. There is no additional output clipping in this method.
- The portable input schema requires exactly age, sex, bmi, smoking_status, hypertension_dx, diabetes_dx, general_health with no missing personal fields. Age is 40–79, BMI is 10–99.8. The root `validate_minimal_v1_research_input` checks the locked dataframe contract.
- The release is research only. Its primary export gate failed; external validation is underpowered and exploratory; clinical, API and web routing are disallowed. No Kazakhstan validation is claimed.
- `artifacts/minimal_v1/**` and `artifacts/minimal_v1_research_release/**` must remain byte unchanged.
