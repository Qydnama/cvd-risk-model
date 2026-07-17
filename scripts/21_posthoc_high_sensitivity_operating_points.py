"""Post-hoc high-sensitivity operating points from NHIS development OOF only.

This analysis is intentionally separate from the locked model and validation stages.
It reads the already-frozen Stage 4 development OOF predictions and writes only new
post-hoc report files. It does not read temporal NHIS or NHANES prediction/label files.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OOF_PATH = ROOT / "artifacts/stage4_baselines/oof_predictions.parquet"
EXPECTED_OOF_SHA256 = "cd8817d7a818c02914777f8288b4901ffbc7f19d5e2d33e331c7577def50d386"
LOCKED_CUTOFF = 0.007469058716065273
MODEL = "LogisticRegression"
PREDICTION_COLUMN = "prediction_sigmoid"
METRICS_PATH = ROOT / "reports/tables/posthoc_high_sensitivity_operating_points.csv"
CONFUSION_PATH = ROOT / "reports/tables/posthoc_high_sensitivity_confusion_matrix.csv"
REPORT_PATH = ROOT / "reports/posthoc_secondary_high_sensitivity_operating_points.md"

OPERATING_POINTS = (
    (
        "high_sensitivity_90",
        "POST_HOC_SECONDARY_HIGH_SENSITIVITY_OPERATING_POINT",
        0.90,
    ),
    (
        "high_sensitivity_95",
        "EXPLORATORY_EXTREME_SENSITIVITY_STRESS_TEST",
        0.95,
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_development_oof() -> pd.DataFrame:
    if sha256_file(OOF_PATH) != EXPECTED_OOF_SHA256:
        raise RuntimeError("Stage 4 OOF SHA-256 differs from the locked input binding")
    frame = pd.read_parquet(OOF_PATH)
    selected = frame.loc[frame["model"].eq(MODEL)].reset_index(drop=True)
    required = {
        "participant_id",
        "survey_year",
        "analysis_weight",
        "cvd_death_5y",
        PREDICTION_COLUMN,
    }
    if missing := required.difference(selected.columns):
        raise ValueError(f"Development OOF columns are missing: {sorted(missing)}")
    if len(selected) != 178_120 or selected["participant_id"].duplicated().any():
        raise ValueError("Selected OOF rows do not uniquely cover the development cohort")
    years = selected["survey_year"].to_numpy(dtype=np.int64)
    if not np.logical_and(years >= 1997, years <= 2009).all():
        raise ValueError("Held-out survey year detected; refusing post-hoc analysis")
    risk = selected[PREDICTION_COLUMN].to_numpy(dtype=np.float64)
    outcome = selected["cvd_death_5y"].to_numpy(dtype=np.int8)
    weight = selected["analysis_weight"].to_numpy(dtype=np.float64)
    if not (np.isfinite(risk).all() and np.isfinite(weight).all()):
        raise ValueError("Risk and survey weights must be finite")
    if (risk < 0).any() or (risk > 1).any() or (weight <= 0).any():
        raise ValueError("Risk must be in [0, 1] and survey weights must be positive")
    if not np.isin(outcome, [0, 1]).all():
        raise ValueError("Outcome must be binary")
    return selected


def select_maximum_cutoff(
    risk: np.ndarray, outcome: np.ndarray, weight: np.ndarray, target: float
) -> dict[str, float]:
    order = np.argsort(-risk, kind="stable")
    sorted_risk = risk[order]
    positive_weight = weight[order] * outcome[order]
    group_ends = np.r_[np.flatnonzero(sorted_risk[:-1] != sorted_risk[1:]), len(sorted_risk) - 1]
    sensitivities = np.cumsum(positive_weight)[group_ends] / positive_weight.sum()
    crossing = np.flatnonzero(sensitivities >= target)
    if len(crossing) == 0:
        raise RuntimeError(f"No observed cutoff achieves sensitivity >= {target}")
    index = int(crossing[0])
    return {
        "cutoff": float(sorted_risk[group_ends[index]]),
        "sensitivity": float(sensitivities[index]),
        "next_higher_cutoff": (
            float(sorted_risk[group_ends[index - 1]]) if index > 0 else float("nan")
        ),
        "next_higher_sensitivity": float(sensitivities[index - 1]) if index > 0 else 0.0,
    }


def operating_metrics(
    frame: pd.DataFrame, cutoff: float
) -> tuple[dict[str, float | int], dict[str, float | int]]:
    risk = frame[PREDICTION_COLUMN].to_numpy(dtype=np.float64)
    outcome = frame["cvd_death_5y"].to_numpy(dtype=np.int8)
    weight = frame["analysis_weight"].to_numpy(dtype=np.float64)
    flagged = risk >= cutoff
    masks = {
        "tp": flagged & (outcome == 1),
        "fn": (~flagged) & (outcome == 1),
        "fp": flagged & (outcome == 0),
        "tn": (~flagged) & (outcome == 0),
    }
    raw = {key: int(mask.sum()) for key, mask in masks.items()}
    weighted = {key: float(weight[mask].sum()) for key, mask in masks.items()}
    tp, fn, fp, tn = (weighted[key] for key in ("tp", "fn", "fp", "tn"))
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    f1 = 2.0 * tp / (2.0 * tp + fp + fn)
    metrics: dict[str, float | int] = {
        "cutoff": cutoff,
        "survey_weighted_sensitivity": sensitivity,
        "survey_weighted_specificity": specificity,
        "survey_weighted_ppv": ppv,
        "survey_weighted_npv": npv,
        "survey_weighted_f1": f1,
        "false_negative_count": raw["fn"],
        "false_positive_count": raw["fp"],
        "weighted_flagged_fraction": (tp + fp) / (tp + fn + fp + tn),
        "n": len(frame),
        "events": int(outcome.sum()),
    }
    confusion: dict[str, float | int] = {
        "raw_tp": raw["tp"],
        "raw_fn": raw["fn"],
        "raw_fp": raw["fp"],
        "raw_tn": raw["tn"],
        "weighted_tp": tp,
        "weighted_fn": fn,
        "weighted_fp": fp,
        "weighted_tn": tn,
    }
    return metrics, confusion


def fmt_probability(value: float) -> str:
    return repr(float(value))


def fmt_metric(value: float) -> str:
    return format(value, ".9f")


def build_report(metrics: pd.DataFrame, confusion: pd.DataFrame) -> str:
    by_name = metrics.set_index("operating_point")
    confusion_by_name = confusion.set_index("operating_point")
    lines = [
        "# Отдельный post-hoc анализ operating points с высокой чувствительностью",
        "",
        "Этот отдельный анализ использует только sigmoid-calibrated LogisticRegression "
        "out-of-fold predictions из NHIS development 1997–2009. Доступ к NHIS temporal "
        "или NHANES held-out labels не выполнялся. Pipeline minimal_v1, sigmoid calibrator, "
        "locked cutoff, артефакты Stage 6/7, manifests и записанные SHA не изменялись.",
        "",
        "Эти post-hoc пороги рассчитаны только на development OOF и не являются внешне "
        "валидированными порогами. Правило классификации: predicted risk >= cutoff. FN/FP "
        "counts — невзвешенные количества участников; confusion matrices дополнительно "
        "содержат survey-weighted totals.",
        "",
        "## Operating points",
        "",
    ]
    for name, label, target in OPERATING_POINTS:
        row = by_name.loc[name]
        cm = confusion_by_name.loc[name]
        lines.extend(
            [
                f"### {name}",
                "",
                f"**{label}**",
                "",
                f"- Целевая survey-weighted sensitivity: >= {target:.2f}",
                f"- Точный cutoff: `{fmt_probability(float(row['cutoff']))}`",
                "- Survey-weighted sensitivity: "
                f"{fmt_metric(float(row['survey_weighted_sensitivity']))}",
                "- Survey-weighted specificity: "
                f"{fmt_metric(float(row['survey_weighted_specificity']))}",
                f"- Survey-weighted PPV: {fmt_metric(float(row['survey_weighted_ppv']))}",
                f"- Survey-weighted NPV: {fmt_metric(float(row['survey_weighted_npv']))}",
                f"- Survey-weighted F1-score: {fmt_metric(float(row['survey_weighted_f1']))}",
                f"- False-negative count: {int(row['false_negative_count'])}",
                f"- False-positive count: {int(row['false_positive_count'])}",
                "- Weighted flagged fraction: "
                f"{fmt_metric(float(row['weighted_flagged_fraction']))}",
                "- Следующий более высокий наблюдаемый cutoff: "
                f"`{fmt_probability(float(row['next_higher_cutoff']))}` "
                f"(sensitivity {fmt_metric(float(row['next_higher_sensitivity']))}; "
                "ниже цели)",
                "",
                "Confusion matrix:",
                "",
                "| Фактический исход | Predicted positive: raw / weighted | "
                "Predicted negative: raw / weighted |",
                "|---|---:|---:|",
                f"| CVD death | {int(cm['raw_tp'])} / {float(cm['weighted_tp']):.6f} | "
                f"{int(cm['raw_fn'])} / {float(cm['weighted_fn']):.6f} |",
                f"| No CVD death | {int(cm['raw_fp'])} / {float(cm['weighted_fp']):.6f} | "
                f"{int(cm['raw_tn'])} / {float(cm['weighted_tn']):.6f} |",
                "",
            ]
        )
    comparison_columns = [
        "operating_point",
        "cutoff",
        "survey_weighted_sensitivity",
        "survey_weighted_specificity",
        "survey_weighted_ppv",
        "survey_weighted_npv",
        "survey_weighted_f1",
        "false_negative_count",
        "false_positive_count",
        "weighted_flagged_fraction",
    ]
    display = metrics[comparison_columns].copy()
    display.columns = [
        "Operating point",
        "Cutoff",
        "Sensitivity",
        "Specificity",
        "PPV",
        "NPV",
        "F1",
        "FN count",
        "FP count",
        "Flagged fraction",
    ]
    float_columns = ["Sensitivity", "Specificity", "PPV", "NPV", "F1", "Flagged fraction"]
    for column in float_columns:
        display[column] = display[column].map(lambda value: f"{value:.9f}")
    display["Cutoff"] = display["Cutoff"].map(fmt_probability)
    lines.extend(
        [
            "## Сравнение с original locked cutoff",
            "",
            display.to_markdown(index=False, disable_numparse=True),
            "",
            "Original cutoff `0.007469058716065273` остаётся заблокированным и неизменным. "
            "Два более низких post-hoc cutoff повышают development OOF sensitivity ценой "
            "снижения specificity и PPV и увеличения weighted flagged fraction.",
            "",
            "## Проверка максимальности cutoff",
            "",
            "Для каждой цели predictions сортировались по убыванию и накапливались по "
            "точным совпадающим значениям probability с ненормированными NHIS survey "
            "weights. Указанный cutoff — первое наблюдаемое значение, на котором cumulative "
            "weighted event fraction достигает цели. Следующий более высокий cutoff не "
            "достигает цели, что подтверждает максимальность при правиле `risk >= cutoff`.",
            "",
            f"Вход: `{OOF_PATH.relative_to(ROOT).as_posix()}`; проверенный SHA-256: "
            f"`{EXPECTED_OOF_SHA256}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    frame = load_development_oof()
    risk = frame[PREDICTION_COLUMN].to_numpy(dtype=np.float64)
    outcome = frame["cvd_death_5y"].to_numpy(dtype=np.int8)
    weight = frame["analysis_weight"].to_numpy(dtype=np.float64)
    metric_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []

    locked_metrics, locked_confusion = operating_metrics(frame, LOCKED_CUTOFF)
    metric_rows.append(
        {
            "operating_point": "original_locked_cutoff",
            "label": "ORIGINAL_LOCKED_CUTOFF_REFERENCE",
            "target_survey_weighted_sensitivity": 0.80,
            **locked_metrics,
            "next_higher_cutoff": np.nan,
            "next_higher_sensitivity": np.nan,
        }
    )
    confusion_rows.append({"operating_point": "original_locked_cutoff", **locked_confusion})

    for name, label, target in OPERATING_POINTS:
        selected = select_maximum_cutoff(risk, outcome, weight, target)
        point_metrics, point_confusion = operating_metrics(frame, selected["cutoff"])
        if point_metrics["survey_weighted_sensitivity"] < target:
            raise RuntimeError(f"{name} does not achieve its sensitivity target")
        if selected["next_higher_sensitivity"] >= target:
            raise RuntimeError(f"{name} is not the maximum qualifying observed cutoff")
        metric_rows.append(
            {
                "operating_point": name,
                "label": label,
                "target_survey_weighted_sensitivity": target,
                **point_metrics,
                "next_higher_cutoff": selected["next_higher_cutoff"],
                "next_higher_sensitivity": selected["next_higher_sensitivity"],
            }
        )
        confusion_rows.append({"operating_point": name, **point_confusion})

    metrics = pd.DataFrame(metric_rows)
    confusion = pd.DataFrame(confusion_rows)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    confusion.to_csv(CONFUSION_PATH, index=False)
    REPORT_PATH.write_text(build_report(metrics, confusion), encoding="utf-8")
    print(REPORT_PATH.relative_to(ROOT))
    print(METRICS_PATH.relative_to(ROOT))
    print(CONFUSION_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
