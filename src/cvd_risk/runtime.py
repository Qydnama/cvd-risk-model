"""Shared configuration, logging, and deterministic-seed helpers."""

from __future__ import annotations

import logging
import os
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_CONFIG_PATH = Path("configs/experiment.yaml")

_LOCKED_PROTOCOL_VALUES: dict[tuple[str, ...], Any] = {
    ("schema_version",): 2,
    ("experiment", "protocol_version"): "2.4",
    (
        "experiment",
        "protocol_sha256",
    ): "c57ab77c2f36f30f9180c1595c30dedf1ac9a9919f7f810511c888d6348a28ce",
    ("cohort", "age_min"): 40,
    ("cohort", "age_max"): 79,
    ("cohort", "mortality_followup_end"): "2019-12-31",
    ("cohort", "require_mortality_linkage_eligibility"): True,
    ("cohort", "require_known_outcome"): True,
    ("cohort", "exclude_prevalent_cvd"): True,
    ("cohort", "prevalent_cvd_conditions"): [
        "coronary_heart_disease",
        "angina_pectoris",
        "myocardial_infarction",
        "stroke",
    ],
    ("cohort", "sensitivity_prevalent_cvd_conditions"): [
        "coronary_heart_disease",
        "angina_pectoris",
        "myocardial_infarction",
        "stroke",
        "congestive_heart_failure",
    ],
    ("cohort", "nhis_chf_sensitivity_years"): [1999, 2002],
    ("outcome", "horizon_months"): 60,
    ("outcome", "nhanes_max_followup_months_inclusive"): 60,
    ("outcome", "nhis_max_followup_quarters_inclusive"): 19,
    ("outcome", "nhis_ambiguous_boundary_quarter"): 20,
    ("outcome", "nhis_include_boundary_in_primary_analysis"): False,
    ("outcome", "reject_negative_followup"): True,
    ("splits", "nhis_development", "start_year"): 1997,
    ("splits", "nhis_development", "end_year"): 2009,
    ("splits", "nhis_temporal_test", "start_year"): 2010,
    ("splits", "nhis_temporal_test", "end_year"): 2014,
    ("splits", "nhanes_external_validation", "start_year"): 1999,
    ("splits", "nhanes_external_validation", "end_year"): 2014,
    ("survey_design", "nhis", "pooled_weight_divisor", "development"): 13,
    ("survey_design", "nhis", "pooled_weight_divisor", "temporal_test"): 5,
    ("statistical_backend", "reference"): "R",
    ("statistical_backend", "r_packages", "survey"): "4.5",
    ("statistical_backend", "r_packages", "pmsampsize"): "1.1.3",
    ("statistical_backend", "r_packages", "pmvalsampsize"): "0.1.0",
    ("statistical_backend", "missing_rscript_status"): "BLOCKED_RSCRIPT_NOT_FOUND",
    ("survey_design", "variance_estimator"): "taylor_linearization",
    ("survey_design", "lonely_psu", "primary"): "adjust",
    ("survey_design", "lonely_psu", "domain_adjust"): True,
    ("survey_design", "minimum_design_df"): 8,
    ("survey_design", "nhis", "identifiers_are_ipums_integrated"): True,
    ("survey_design", "nhis", "forbid_year_interaction"): True,
    ("survey_design", "nhanes", "identifiers_are_published"): True,
    ("survey_design", "nhanes", "collision_audit_required"): True,
    ("survey_design", "nhanes", "automatic_cycle_interaction_forbidden"): True,
    (
        "survey_design",
        "nhanes",
        "pooled_weight_rules",
        "external_1999_2002",
        "source",
    ): "WTMEC4YR",
    (
        "survey_design",
        "nhanes",
        "pooled_weight_rules",
        "external_1999_2002",
        "divisor",
    ): 4,
    (
        "survey_design",
        "nhanes",
        "pooled_weight_rules",
        "external_2003_2014",
        "source",
    ): "WTMEC2YR",
    (
        "survey_design",
        "nhanes",
        "pooled_weight_rules",
        "external_2003_2014",
        "divisor",
    ): 8,
    ("stop_gate", "minimum_events", "nhis_minimal_development"): 500,
    (
        "stop_gate",
        "minimum_events",
        "nhis_minimal_no_general_health_development",
    ): 500,
    ("stop_gate", "minimum_events", "nhanes_clinical_core_development"): 200,
    (
        "stop_gate",
        "minimum_events",
        "nhanes_clinical_hba1c_paired_development",
    ): 200,
    ("stop_gate", "minimum_events", "nhis_temporal_test"): 200,
    ("stop_gate", "minimum_events", "nhanes_external_validation"): 100,
    ("stop_gate", "minimum_events_per_effective_df"): 20,
    ("stop_gate", "risk_ci_max_half_width"): 0.0025,
    ("stop_gate", "pmsampsize", "expected_r2_fraction_of_max"): 0.15,
    ("stop_gate", "pmsampsize", "parameters"): 15,
    ("stop_gate", "pmsampsize", "shrinkage"): 0.9,
    ("missing_data", "minimal", "primary"): "within_training_fold_imputation",
    ("missing_data", "clinical_core", "primary"): "complete_case",
    ("missing_data", "clinical_hba1c", "primary"): "paired_complete_case",
    ("missing_data", "clinical_hba1c", "export_if_only_imputed_passes"): False,
    ("reproducibility", "overwrite_outputs"): False,
}


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load a YAML experiment config and require a mapping at its root."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)

    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}, got {type(loaded).__name__}")
    return loaded


def _nested_value(config: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise ValueError(f"Missing locked protocol config: {'.'.join(path)}")
        current = current[key]
    return current


def validate_locked_protocol(config: Mapping[str, Any]) -> None:
    """Fail if YAML silently diverges from constants implemented in stages 0–3."""

    mismatches: list[str] = []
    for path, expected in _LOCKED_PROTOCOL_VALUES.items():
        observed = _nested_value(config, path)
        if type(observed) is not type(expected) or observed != expected:
            mismatches.append(f"{'.'.join(path)}={observed!r} (implemented protocol: {expected!r})")
    causes = _nested_value(config, ("outcome", "cardiovascular_ucod_leading"))
    if causes != ["001", "005"]:
        mismatches.append("outcome.cardiovascular_ucod_leading must remain ['001', '005']")
    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(
            "Experiment config diverges from the locked research protocol. "
            "Update code and tests only after explicit protocol approval: "
            f"{details}"
        )


def seed_everything(seed: int) -> None:
    """Seed the random generators used in stages 0-3 and child processes."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def configure_logging(config: Mapping[str, Any] | None = None) -> None:
    """Configure process logging from the ``logging`` config section."""
    section = config.get("logging", {}) if config is not None else {}
    if not isinstance(section, Mapping):
        raise ValueError("The 'logging' config section must be a mapping")

    level_name = str(section.get("level", "INFO")).upper()
    level = logging.getLevelNamesMapping().get(level_name)
    if level is None:
        raise ValueError(f"Unknown logging level: {level_name}")

    logging.basicConfig(
        level=level,
        format=str(section.get("format", "%(levelname)s | %(name)s | %(message)s")),
        datefmt=str(section.get("date_format", "%Y-%m-%dT%H:%M:%S%z")),
    )


def initialize_runtime(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load configuration and apply the shared logging and seed policy."""
    config = load_config(path)
    validate_locked_protocol(config)
    experiment = config.get("experiment")
    if not isinstance(experiment, Mapping):
        raise ValueError("The 'experiment' config section must be a mapping")

    seed = experiment.get("random_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("experiment.random_seed must be an integer")

    configure_logging(config)
    seed_everything(seed)
    return config


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "configure_logging",
    "initialize_runtime",
    "load_config",
    "seed_everything",
    "validate_locked_protocol",
]
