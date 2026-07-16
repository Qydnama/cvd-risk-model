"""Stage 8 clinical-route preflight without model fitting."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from cvd_risk.provenance import reject_quarantined_input, sha256_file

EXPECTED_MINIMAL = (
    "age",
    "sex",
    "bmi",
    "smoking_status",
    "hypertension_dx",
    "diabetes_dx",
    "general_health",
)
EXPECTED_CORE_CANONICAL = ("sbp_mean", "total_cholesterol", "hdl_cholesterol")
EXPECTED_HBA1C_CANONICAL = (*EXPECTED_CORE_CANONICAL, "hba1c")
EXPECTED_MATERIALIZED_ALIASES = {"sbp_mean": "systolic_bp_mean"}


def _normalized_text_sha256(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _binding(path: Path, *, root: Path) -> dict[str, Any]:
    relative = path.resolve().relative_to(root.resolve())
    return {
        "repository_relative_path": relative.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_stage8_preflight_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Stage 8 preflight config schema_version must be 1")
    if payload.get("stage") != "stage8_clinical_preflight":
        raise ValueError("Unexpected Stage 8 preflight stage")
    scope = payload.get("scope", {})
    if scope.get("external_validation_claim") is not False:
        raise ValueError("Stage 8 clinical analysis must not claim external validation")
    features = payload.get("features", {})
    if tuple(features.get("minimal", ())) != EXPECTED_MINIMAL:
        raise ValueError("Stage 8 minimal feature contract differs from the protocol")
    if tuple(features.get("clinical_core_canonical", ())) != EXPECTED_CORE_CANONICAL:
        raise ValueError("Stage 8 clinical_core feature contract differs from the protocol")
    if tuple(features.get("clinical_hba1c_canonical", ())) != EXPECTED_HBA1C_CANONICAL:
        raise ValueError("Stage 8 clinical_hba1c feature contract differs from the protocol")
    if features.get("materialized_aliases") != EXPECTED_MATERIALIZED_ALIASES:
        raise ValueError("Stage 8 materialized alias audit differs from the known cohort schema")
    gate = payload.get("gate", {})
    if gate.get("minimum_events") != {
        "clinical_core": 200,
        "clinical_hba1c_paired": 200,
    }:
        raise ValueError("Stage 8 event floors differ from the locked protocol")
    return payload


def verify_stage6_completion(config: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    inputs = config["inputs"]
    manifest_path = root / inputs["stage6_manifest"]
    receipt_path = root / inputs["stage6_receipt"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required_manifest = {
        "stage": "stage6_authorized_unseal_validation",
        "status": "COMPLETE_UNDERPOWERED_EXPLORATORY",
        "outcomes_unsealed": True,
        "model_modified": False,
        "calibrator_modified": False,
        "cutoff_modified_or_reselected": False,
    }
    if any(manifest.get(key) != value for key, value in required_manifest.items()):
        raise ValueError("Stage 6 completion does not authorize the Stage 8 preflight")
    if receipt.get("status") != "ONE_TIME_UNSEAL_VALIDATION_COMPLETE":
        raise ValueError("Stage 6 access receipt is incomplete")
    execution = receipt.get("execution_manifest", {})
    if execution.get("repository_relative_path") != inputs["stage6_manifest"]:
        raise ValueError("Stage 6 receipt points to a different execution manifest")
    if execution.get("sha256") != _normalized_text_sha256(manifest_path):
        raise ValueError("Stage 6 execution manifest differs from its receipt")
    return {
        "manifest": _binding(manifest_path, root=root),
        "receipt": _binding(receipt_path, root=root),
    }


def load_stage8_cohort(config: Mapping[str, Any], *, root: Path) -> pd.DataFrame:
    inputs = config["inputs"]
    cohort_path = root / inputs["cohort"]
    if sha256_file(cohort_path) != inputs["cohort_sha256"]:
        raise ValueError("Rebuilt NHANES cohort differs from the preflight binding")
    reject_quarantined_input(cohort_path, root / inputs["quarantine_manifest"])
    materialized_core = [EXPECTED_MATERIALIZED_ALIASES.get(x, x) for x in EXPECTED_CORE_CANONICAL]
    columns = [
        "participant_id",
        "survey_cycle",
        "strata",
        "psu",
        "analysis_weight",
        "cohort_eligible",
        "split",
        "cvd_death_5y",
        *EXPECTED_MINIMAL,
        *materialized_core,
        "hba1c",
    ]
    frame = pd.read_parquet(cohort_path, columns=columns)
    cohort_filter = config["cohort_filter"]
    frame = frame.loc[
        frame["cohort_eligible"].eq(cohort_filter["cohort_eligible"])
        & frame["split"].eq(cohort_filter["split"])
    ].reset_index(drop=True)
    if frame.empty or frame["participant_id"].duplicated().any():
        raise ValueError("Stage 8 cohort is empty or contains duplicate participants")
    if frame["cvd_death_5y"].isna().any() or not frame["cvd_death_5y"].isin([0, 1]).all():
        raise ValueError("Stage 8 outcome must be a complete binary field")
    weights = frame["analysis_weight"].to_numpy(dtype=np.float64)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Stage 8 survey weights must be finite and positive")
    return frame


def clinical_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Return both conflicting protocol interpretations without choosing between them."""

    materialized_core = [EXPECTED_MATERIALIZED_ALIASES.get(x, x) for x in EXPECTED_CORE_CANONICAL]
    minimal_complete = frame[list(EXPECTED_MINIMAL)].notna().all(axis=1)
    core_complete = frame[materialized_core].notna().all(axis=1)
    hba1c_complete = frame["hba1c"].notna()
    return {
        "strict_section_10_clinical_core": minimal_complete & core_complete,
        "strict_section_10_clinical_hba1c_paired": (
            minimal_complete & core_complete & hba1c_complete
        ),
        "route_policy_section_20_clinical_core": core_complete,
        "route_policy_section_20_clinical_hba1c_paired": core_complete & hba1c_complete,
    }


def route_gate_table(frame: pd.DataFrame, minimum_events: int) -> pd.DataFrame:
    weights = frame["analysis_weight"].to_numpy(dtype=np.float64)
    outcome = frame["cvd_death_5y"].to_numpy(dtype=np.int8)
    rows: list[dict[str, Any]] = []
    for analysis, mask_series in clinical_masks(frame).items():
        mask = mask_series.to_numpy(dtype=bool)
        selected_weight = weights[mask]
        selected_outcome = outcome[mask]
        event_count = int(selected_outcome.sum())
        strata = frame.loc[mask, "strata"].nunique()
        psus = frame.loc[mask, ["strata", "psu"]].drop_duplicates().shape[0]
        rows.append(
            {
                "analysis": analysis,
                "n": int(mask.sum()),
                "events": event_count,
                "non_events": int(mask.sum() - event_count),
                "participants_retained_fraction": float(mask.mean()),
                "weighted_population_retained_fraction": float(
                    selected_weight.sum() / weights.sum()
                ),
                "weighted_event_prevalence": float(
                    np.sum(selected_weight * selected_outcome) / selected_weight.sum()
                ),
                "kish_effective_n": float(
                    selected_weight.sum() ** 2 / np.square(selected_weight).sum()
                ),
                "design_df": int(psus - strata),
                "minimum_events": minimum_events,
                "event_gate_status": "PASS" if event_count >= minimum_events else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def field_attrition_table(frame: pd.DataFrame) -> pd.DataFrame:
    weights = frame["analysis_weight"].to_numpy(dtype=np.float64)
    outcome = frame["cvd_death_5y"].to_numpy(dtype=np.int8)
    fields = [
        *EXPECTED_MINIMAL,
        "systolic_bp_mean",
        "total_cholesterol",
        "hdl_cholesterol",
        "hba1c",
    ]
    rows: list[dict[str, Any]] = []
    for field in fields:
        observed = frame[field].notna().to_numpy()
        rows.append(
            {
                "field": field,
                "canonical_field": "sbp_mean" if field == "systolic_bp_mean" else field,
                "n_total": len(frame),
                "n_observed": int(observed.sum()),
                "n_missing": int((~observed).sum()),
                "events_observed": int(outcome[observed].sum()),
                "events_lost": int(outcome[~observed].sum()),
                "observed_fraction": float(observed.mean()),
                "weighted_observed_fraction": float(weights[observed].sum() / weights.sum()),
            }
        )
    return pd.DataFrame(rows)


def cycle_support_table(frame: pd.DataFrame) -> pd.DataFrame:
    outcome = frame["cvd_death_5y"].to_numpy(dtype=np.int8)
    rows: list[dict[str, Any]] = []
    for analysis, mask_series in clinical_masks(frame).items():
        mask = mask_series.to_numpy(dtype=bool)
        selected = frame.loc[mask].copy()
        for cycle, group in selected.groupby("survey_cycle", observed=True, sort=True):
            rows.append(
                {
                    "analysis": analysis,
                    "survey_cycle": str(cycle),
                    "n": len(group),
                    "events": int(group["cvd_death_5y"].sum()),
                    "non_events": int(len(group) - group["cvd_death_5y"].sum()),
                    "weighted_population": float(group["analysis_weight"].sum()),
                }
            )
        if int(outcome[mask].sum()) == 0:
            raise ValueError(f"No events remain for {analysis}")
    return pd.DataFrame(rows)


def _write_report(
    path: Path,
    *,
    frame: pd.DataFrame,
    gates: pd.DataFrame,
    attrition: pd.DataFrame,
    cycles: pd.DataFrame,
    status_path: Path,
) -> None:
    compact_gates = gates[
        [
            "analysis",
            "n",
            "events",
            "participants_retained_fraction",
            "weighted_population_retained_fraction",
            "design_df",
            "minimum_events",
            "event_gate_status",
        ]
    ]
    lines = [
        "# Stage 8 clinical routes: preflight feasibility report",
        "",
        "## Решение",
        "",
        "**BLOCKED_PROTOCOL_CLARIFICATION_REQUIRED — модели не обучались.**",
        "",
        "Stage 6 one-time external validation уже завершён, поэтому этот preflight имел "
        "право прочитать rebuilt v2.4 NHANES labels. Это не повтор внешней валидации: "
        "анализ относится только к будущему NHANES internal grouped validation.",
        "",
        f"Исходная primary NHANES cohort: {len(frame):,} участников, "
        f"{int(frame['cvd_death_5y'].sum()):,} событий.",
        "",
        "## Route-specific event gates",
        "",
        compact_gates.to_markdown(index=False),
        "",
        "Раздел 10 требует paired complete-case с одновременно наблюдаемыми всеми minimal "
        "и clinical predictors. При этой строгой трактовке и clinical_core, и paired "
        "clinical_hba1c содержат 198 событий, то есть не достигают locked floor 200. "
        "Раздел 20 и experiment.yaml определяют complete-case только по обязательным "
        "clinical fields, оставляя minimal missingness для обработки внутри folds; при этой "
        "трактовке остаётся 218 событий и gate проходит. Выбирать одну трактовку post-hoc "
        "без явного protocol decision нельзя.",
        "",
        "Дополнительное schema discrepancy: protocol и experiment.yaml называют поле "
        "`sbp_mean`, тогда как rebuilt cohort материализует `systolic_bp_mean`. Preflight "
        "использовал его только как явно задокументированный audit alias; modeling/export "
        "до фиксации canonical contract запрещён.",
        "",
        "## Availability каждого predictor",
        "",
        attrition.to_markdown(index=False),
        "",
        "## Поддержка по survey cycle",
        "",
        cycles.to_markdown(index=False),
        "",
        "## Разрешённое продолжение",
        "",
        "Нужно заранее выбрать и письменно зафиксировать complete-case policy для minimal "
        "predictors в clinical routes и canonical имя SBP. Если утверждается строгий "
        "section-10 вариант, Stage 8 завершается feasibility report без fitting. Если "
        "утверждается section-20 вариант, можно отдельно preregister grouped folds, метрики, "
        "paired 2,000-repeat cluster bootstrap и DCA до обучения моделей.",
        "",
        "Никакие Stage 8 model, calibrator, OOF predictions или routing/export artifacts "
        "в этом запуске не создавались. Locked minimal_v1 и Stage 6/7 не изменялись.",
        "",
        f"Machine-readable status: `{status_path.as_posix()}`.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_stage8_preflight(
    config_path: str | Path,
    *,
    root: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    config_file = Path(config_path)
    repository = Path(root) if root is not None else config_file.resolve().parents[1]
    config = load_stage8_preflight_config(config_file)
    stage6_bindings = verify_stage6_completion(config, root=repository)
    frame = load_stage8_cohort(config, root=repository)
    minimum_events = int(config["gate"]["minimum_events"]["clinical_core"])
    gates = route_gate_table(frame, minimum_events)
    attrition = field_attrition_table(frame)
    cycles = cycle_support_table(frame)

    outputs = {key: repository / value for key, value in config["outputs"].items()}
    known = [path for key, path in outputs.items() if key != "artifact_dir"]
    if any(path.exists() for path in known) and not force:
        raise FileExistsError("Stage 8 preflight outputs exist; pass --force to replace them")
    for path in known:
        path.parent.mkdir(parents=True, exist_ok=True)
    gates.to_csv(outputs["route_gate"], index=False)
    attrition.to_csv(outputs["field_attrition"], index=False)
    cycles.to_csv(outputs["cycle_support"], index=False)
    _write_report(
        outputs["report"],
        frame=frame,
        gates=gates,
        attrition=attrition,
        cycles=cycles,
        status_path=outputs["status"].relative_to(repository),
    )

    strict = gates[gates["analysis"].str.startswith("strict_section_10")]
    route_policy = gates[gates["analysis"].str.startswith("route_policy_section_20")]
    status: dict[str, Any] = {
        "schema_version": 1,
        "stage": "stage8_clinical_preflight",
        "status": "BLOCKED_PROTOCOL_CLARIFICATION_REQUIRED",
        "generated_at": datetime.now(UTC).isoformat(),
        "interpretation": "nhanes_internal_grouped_validation_preflight_not_external_validation",
        "nhanes_labels_accessed": True,
        "stage6_external_validation_repeated": False,
        "models_fitted": False,
        "calibrators_fitted": False,
        "oof_predictions_created": False,
        "routing_or_export_artifacts_created": False,
        "strict_section_10_all_event_gates_pass": bool(
            strict["event_gate_status"].eq("PASS").all()
        ),
        "route_policy_section_20_all_event_gates_pass": bool(
            route_policy["event_gate_status"].eq("PASS").all()
        ),
        "blocking_discrepancies": [
            "complete_case_definition_conflicts_between_protocol_sections_10_and_20",
            "canonical_sbp_mean_is_materialized_as_systolic_bp_mean",
        ],
        "input_bindings": {
            "config": _binding(config_file, root=repository),
            "cohort": _binding(repository / config["inputs"]["cohort"], root=repository),
            **stage6_bindings,
        },
        "outputs": {
            key: _binding(path, root=repository)
            for key, path in outputs.items()
            if key not in {"artifact_dir", "status"}
        },
    }
    outputs["status"].write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return status


__all__ = [
    "clinical_masks",
    "field_attrition_table",
    "load_stage8_preflight_config",
    "route_gate_table",
    "run_stage8_preflight",
]
