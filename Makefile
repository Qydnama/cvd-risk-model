UV ?= uv
PYTHON := $(UV) run python
EXPERIMENT_CONFIG ?= configs/experiment.yaml
NHANES_CONFIG ?= configs/nhanes_files.yaml
DATA_MANIFEST ?= configs/data_manifest.yaml

.PHONY: help sync test stage0 check-nhis nhanes data audit train

help:
	@echo "Targets:"
	@echo "  sync   Install the locked Python 3.12 development environment"
	@echo "  test   Run all unit tests"
	@echo "  stage0 Verify protocol/R locks and pre-amendment artifact quarantine"
	@echo "  check-nhis  Inspect and verify the manual IPUMS input before downloads"
	@echo "  nhanes Download the configured NHANES/LMF inventory independently"
	@echo "  data   Download NHANES and build the NHIS/NHANES cohorts"
	@echo "  audit  Produce the required cohort audit and evaluate the stop-gate"
	@echo "  train  Run the training guard (training remains blocked before stop-gate approval)"

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
	$(PYTHON) scripts/05_train_minimal.py --config $(EXPERIMENT_CONFIG)
