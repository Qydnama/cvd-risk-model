from __future__ import annotations

from copy import deepcopy

import pytest

from cvd_risk.runtime import load_config, validate_locked_protocol


def test_repository_config_matches_locked_protocol() -> None:
    validate_locked_protocol(load_config())


def test_protocol_config_cannot_silently_change_age() -> None:
    config = deepcopy(load_config())
    config["cohort"]["age_max"] = 84
    with pytest.raises(ValueError, match="age_max"):
        validate_locked_protocol(config)


def test_protocol_config_rejects_bool_disguised_as_integer() -> None:
    config = deepcopy(load_config())
    config["outcome"]["horizon_months"] = True
    with pytest.raises(ValueError, match="horizon_months"):
        validate_locked_protocol(config)
