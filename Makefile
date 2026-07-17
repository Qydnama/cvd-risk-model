UV ?= uv
PYTHON := $(UV) run python
EXPERIMENT_CONFIG ?= configs/experiment.yaml
NHANES_CONFIG ?= configs/nhanes_files.yaml
DATA_MANIFEST ?= configs/data_manifest.yaml

.PHONY: help sync test stage0 check-nhis nhanes data audit train ml plan-validation audit-operating-point export-pipeline lock-model validate-lock label-free validate-label-free validate-research-release

help:
	@echo "Targets:"
	@echo "  sync   Install the locked Python 3.12 development environment"
	@echo "  test   Run all unit tests"
	@echo "  stage0 Verify protocol/R locks and pre-amendment artifact quarantine"
	@echo "  check-nhis  Inspect and verify the manual IPUMS input before downloads"
	@echo "  nhanes Download the configured NHANES/LMF inventory independently"
	@echo "  data   Download NHANES and build the NHIS/NHANES cohorts"
	@echo "  audit  Produce the required cohort audit and evaluate the stop-gate"
	@echo "  train  Fit Stage 4 logistic baselines on NHIS development only"
	@echo "  ml     Tune Stage 5 RF/XGBoost candidates on NHIS development only"
	@echo "  plan-validation  Run Stage 5a provisional pmvalsampsize planning"
	@echo "  audit-operating-point  Run the Stage 5b fixed-cutoff pre-lock audit"
	@echo "  export-pipeline  Serialize the final full-development pipeline without refit"
	@echo "  lock-model  Create the immutable external-validation model lock"
	@echo "  validate-lock  Verify lock ancestry, scope and all frozen SHA bindings"
	@echo "  label-free  Freeze Stage 6 held-out predictions without reading outcomes"
	@echo "  validate-label-free  Verify frozen Stage 6 prediction and provenance SHA bindings"
	@echo "  validate-research-release  Verify the research-only minimal_v1 Stage 9 snapshot"

sync:
	$(UV) sync --locked --dev

test:
	$(UV) run pytest

stage0:
	$(PYTHON) scripts/00_stage0_check.py --config $(EXPERIMENT_CONFIG)

check-nhis:
	$(PYTHON) scripts/00_inspect_nhis_input.py --config $(EXPERIMENT_CONFIG)

nhanes:
	$(PYTHON) scripts/01_download_nhanes.py --config $(NHANES_CONFIG) --manifest $(DATA_MANIFEST)

data: stage0
	$(PYTHON) scripts/00_inspect_nhis_input.py --config $(EXPERIMENT_CONFIG)
	$(PYTHON) scripts/02_build_nhis_cohort.py --config $(EXPERIMENT_CONFIG)
	$(PYTHON) scripts/01_download_nhanes.py --config $(NHANES_CONFIG) --manifest $(DATA_MANIFEST)
	$(PYTHON) scripts/03_build_nhanes_cohort.py --config $(EXPERIMENT_CONFIG)

audit: stage0
	$(PYTHON) scripts/04_audit_cohorts.py --config $(EXPERIMENT_CONFIG)

train:
	$(UV) run --extra modeling python scripts/05_train_minimal.py --config $(EXPERIMENT_CONFIG)

ml:
	$(UV) run --extra modeling python scripts/06_train_ml_candidates.py --config $(EXPERIMENT_CONFIG)

plan-validation:
	$(UV) run --extra modeling python scripts/07_stage5a_planning.py --config $(EXPERIMENT_CONFIG)

audit-operating-point:
	$(UV) run --extra modeling python scripts/08_operating_point_audit.py --config configs/operating_point.yaml

export-pipeline:
	$(UV) run --extra modeling python scripts/09_export_final_pipeline.py --config configs/model_lock.yaml

lock-model:
	$(UV) run --extra modeling python scripts/10_create_model_lock.py --config configs/model_lock.yaml

validate-lock:
	$(UV) run --extra modeling python scripts/11_validate_model_lock.py

label-free:
	$(UV) run --extra modeling python scripts/14_stage6_label_free.py --config configs/stage6_label_free.yaml

validate-label-free:
	$(UV) run --extra modeling python scripts/15_validate_stage6_label_free.py

validate-research-release:
	$(UV) run --frozen python scripts/34_validate_stage9_research_release.py
