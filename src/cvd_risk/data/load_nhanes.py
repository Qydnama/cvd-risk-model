"""Strict cycle-level loading and one-to-one assembly of raw NHANES files."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pandas as pd

from cvd_risk.data.download_nhanes import FileSpec, load_file_specs
from cvd_risk.data.read_lmf import LMFValidationError, merge_demo_lmf, normalize_seqn, read_lmf

Profile = Literal["fast", "extended", "all"]


class NHANESLoadError(ValueError):
    """Raised when raw NHANES components cannot be assembled safely."""


def read_xpt(path: str | Path) -> pd.DataFrame:
    """Read a SAS transport file and normalize its identifier/column names."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"NHANES XPT file not found: {path}")
    try:
        frame = pd.read_sas(path, format="xport", encoding="latin-1")
    except Exception as exc:  # pandas raises several parser-specific exceptions
        raise NHANESLoadError(f"Could not parse NHANES XPT file {path}: {exc}") from exc
    normalized_names = [str(column).upper() for column in frame.columns]
    if len(normalized_names) != len(set(normalized_names)):
        raise NHANESLoadError(f"{path}: column names collide after upper-casing")
    frame.columns = normalized_names
    if "SEQN" not in frame.columns:
        raise NHANESLoadError(f"{path}: required identifier SEQN is missing")
    try:
        frame["SEQN"] = normalize_seqn(frame["SEQN"], context=str(path))
    except LMFValidationError as exc:
        raise NHANESLoadError(str(exc)) from exc
    duplicates = frame.loc[frame["SEQN"].duplicated(keep=False), "SEQN"].unique().tolist()
    if duplicates:
        raise NHANESLoadError(f"{path}: duplicate SEQN values: {duplicates[:10]}")
    return frame


def merge_component(
    base: pd.DataFrame,
    component: pd.DataFrame,
    *,
    component_name: str,
) -> pd.DataFrame:
    """Left-merge a participant component, enforcing a one-to-one key."""

    if "SEQN" not in base.columns or "SEQN" not in component.columns:
        raise NHANESLoadError(f"{component_name}: SEQN is required on both sides of the merge")
    left = base.copy()
    right = component.copy()
    try:
        left["SEQN"] = normalize_seqn(left["SEQN"], context="assembled NHANES")
        right["SEQN"] = normalize_seqn(right["SEQN"], context=component_name)
    except LMFValidationError as exc:
        raise NHANESLoadError(str(exc)) from exc
    if left["SEQN"].duplicated().any():
        raise NHANESLoadError("assembled NHANES data contain duplicate SEQN values")
    if right["SEQN"].duplicated().any():
        raise NHANESLoadError(f"{component_name}: duplicate SEQN values violate one-to-one merge")
    overlap = (set(left.columns) & set(right.columns)) - {"SEQN"}
    if overlap:
        raise NHANESLoadError(f"{component_name}: overlapping variables {sorted(overlap)}")
    try:
        merged = left.merge(right, on="SEQN", how="left", validate="one_to_one", sort=False)
    except pd.errors.MergeError as exc:
        raise NHANESLoadError(f"{component_name}: merge is not one-to-one") from exc
    if len(merged) != len(left) or merged["SEQN"].duplicated().any():
        raise NHANESLoadError(f"{component_name}: merge duplicated participants")
    return merged


def _profile_tiers(profile: Profile) -> set[str]:
    if profile == "fast":
        return {"fast"}
    if profile in {"extended", "all"}:
        # The extended clinical analysis includes the seven fast-model files.
        return {"fast", "extended"}
    raise NHANESLoadError(f"Unknown NHANES profile {profile!r}")


def cycle_file_specs(
    cycle: str,
    *,
    config_path: str | Path = "configs/nhanes_files.yaml",
    profile: Profile = "all",
) -> list[FileSpec]:
    return load_file_specs(config_path, cycles=[cycle], tiers=_profile_tiers(profile))


def required_cycle_paths(
    cycle: str,
    *,
    raw_dir: str | Path = "data/raw/nhanes",
    config_path: str | Path = "configs/nhanes_files.yaml",
    profile: Profile = "all",
) -> dict[str, Path]:
    specs = cycle_file_specs(cycle, config_path=config_path, profile=profile)
    cycle_dir = Path(raw_dir) / cycle
    paths = {spec.component: cycle_dir / spec.filename for spec in specs}
    missing = [f"{component}: {path}" for component, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required NHANES files:\n" + "\n".join(missing))
    return paths


def load_nhanes_cycle(
    cycle: str,
    *,
    raw_dir: str | Path = "data/raw/nhanes",
    config_path: str | Path = "configs/nhanes_files.yaml",
    profile: Profile = "all",
    include_mortality: bool = True,
    require_complete_lmf_match: bool = True,
) -> pd.DataFrame:
    """Load one cycle with DEMO as the participant spine and strict 1:1 merges."""

    specs = cycle_file_specs(cycle, config_path=config_path, profile=profile)
    by_component = {spec.component: spec for spec in specs}
    if "DEMO" not in by_component:
        raise NHANESLoadError(f"{cycle}: config does not contain DEMO")
    cycle_dir = Path(raw_dir) / cycle
    required = [spec for spec in specs if include_mortality or spec.component != "LMF"]
    missing = [
        cycle_dir / spec.filename for spec in required if not (cycle_dir / spec.filename).is_file()
    ]
    if missing:
        details = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"{cycle}: missing required NHANES files:\n{details}")

    demo_spec = by_component["DEMO"]
    assembled = read_xpt(cycle_dir / demo_spec.filename)
    for spec in specs:
        if spec.component in {"DEMO", "LMF"}:
            continue
        component = read_xpt(cycle_dir / spec.filename)
        assembled = merge_component(assembled, component, component_name=spec.component)

    if include_mortality:
        mortality_spec = by_component.get("LMF")
        if mortality_spec is None:
            raise NHANESLoadError(f"{cycle}: config does not contain the linked mortality file")
        mortality = read_lmf(cycle_dir / mortality_spec.filename)
        try:
            assembled = merge_demo_lmf(
                assembled,
                mortality,
                require_complete_match=require_complete_lmf_match,
            )
        except LMFValidationError as exc:
            raise NHANESLoadError(f"{cycle}: {exc}") from exc
    assembled.insert(0, "CYCLE", cycle)
    if assembled.duplicated(["CYCLE", "SEQN"]).any():
        raise NHANESLoadError(f"{cycle}: duplicate participants after assembly")
    return assembled


def load_nhanes_cycles(
    cycles: Sequence[str],
    *,
    raw_dir: str | Path = "data/raw/nhanes",
    config_path: str | Path = "configs/nhanes_files.yaml",
    profile: Profile = "all",
    include_mortality: bool = True,
) -> pd.DataFrame:
    """Load and concatenate cycles while preserving the composite participant key."""

    if not cycles:
        raise NHANESLoadError("At least one NHANES cycle is required")
    frames = [
        load_nhanes_cycle(
            cycle,
            raw_dir=raw_dir,
            config_path=config_path,
            profile=profile,
            include_mortality=include_mortality,
        )
        for cycle in cycles
    ]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if combined.duplicated(["CYCLE", "SEQN"]).any():
        raise NHANESLoadError("Duplicate (CYCLE, SEQN) participant keys across loaded cycles")
    return combined
