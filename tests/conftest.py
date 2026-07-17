"""Test-suite configuration that does not alter locked project metadata."""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_local_research_artifacts: requires gitignored data or "
        "workstation-bound locked artifacts",
    )
