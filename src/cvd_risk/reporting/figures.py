"""Mandatory cohort-audit figures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_outcome_by_year(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    year_col: str = "survey_year",
    target_col: str = "cvd_death_5y",
    group_col: str | None = None,
) -> Path:
    """Plot unweighted five-year event prevalence and event counts by year."""

    required = {year_col, target_col}
    if group_col is not None:
        required.add(group_col)
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing columns for outcome-by-year plot: {sorted(missing)}")
    groupers = [year_col] if group_col is None else [group_col, year_col]
    grouped = (
        frame.groupby(groupers, observed=True)[target_col]
        .agg(event_rate="mean", n_events="sum", n_participants="size")
        .reset_index()
        .sort_values(year_col)
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    fig, (rate_axis, count_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
    )
    group_names: list[object] = (
        ["all"] if group_col is None else grouped[group_col].drop_duplicates().tolist()
    )
    for group_name in group_names:
        group = grouped if group_col is None else grouped.loc[grouped[group_col] == group_name]
        label = str(group_name)
        rate_axis.plot(group[year_col], group["event_rate"], marker="o", label=label)
        count_axis.plot(group[year_col], group["n_events"], marker="o", label=label)
    rate_axis.set_ylabel("Unweighted 5-year\nCVD mortality prevalence")
    count_axis.set_xlabel("Survey year")
    count_axis.set_ylabel("Event count")
    for axis in (rate_axis, count_axis):
        axis.grid(alpha=0.25)
        if group_col is not None:
            axis.legend(title=group_col)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_age_distribution(
    cohorts: dict[str, pd.DataFrame],
    output_path: str | Path,
    *,
    age_col: str = "age",
) -> Path:
    """Plot comparable age histograms for the three protocol cohorts."""

    if not cohorts:
        raise ValueError("At least one cohort is required")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(10, 5))
    bins = list(range(40, 81, 2))
    for name, frame in cohorts.items():
        if age_col not in frame:
            raise KeyError(f"Missing age column {age_col!r} in cohort {name!r}")
        axis.hist(
            frame[age_col].dropna(),
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.5,
            label=name,
        )
    axis.set_xlabel("Age at baseline")
    axis.set_ylabel("Density")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
