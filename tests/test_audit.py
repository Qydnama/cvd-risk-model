from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from cvd_risk.reporting.audit import (
    _load_flow,
    _validate_cohort_contract,
    run_stage3_audit,
)
from cvd_risk.schemas import CanonicalSchemaError


def _protocol_registry(path: Path, *, resolved: bool) -> Path:
    registry = path / ("protocol_resolved.yaml" if resolved else "protocol_pending.yaml")
    registry.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "discrepancies": [
                    {
                        "id": "nhanes_pooled_weight_1999_2002",
                        "title": "NHANES pooled weights",
                        "status": "resolved" if resolved else "awaiting_user_decision",
                        "blocking": True,
                        "locked_protocol_rule": "WTMEC2YR / 8 for all cycles",
                        "official_guidance": "Use the supplied four-year early weight",
                        "recommended_resolution": "Update protocol and implementation",
                        "resolution": "Protocol and code updated together" if resolved else None,
                        "official_sources": [
                            "https://wwwn.cdc.gov/nchs/nhanes/tutorials/weighting.aspx"
                        ],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return registry


def _cohort(n: int, events: int, *, split: str | None = None) -> pd.DataFrame:
    target = [1] * events + [0] * (n - events)
    frame = pd.DataFrame(
        {
            "participant_id": [f"id-{index}" for index in range(n)],
            "survey_year": [2005] * n,
            "survey_cycle": ["2005-2006"] * n,
            "age": [60] * n,
            "sex": ["female"] * n,
            "bmi": [25.0] * n,
            "smoking_status": ["never"] * n,
            "hypertension_dx": [False] * n,
            "diabetes_dx": [False] * n,
            "general_health": ["good"] * n,
            "general_health_hsd_sensitivity": ["good"] * n,
            "cvd_death_5y": target,
            "analysis_weight": [1.0] * n,
            "strata": [1] * n,
            "psu": [1] * n,
            "codebook_verification_status": ["verified_against_user_extract"] * n,
            "extract_sha256": ["a" * 64] * n,
            "codebook_sha256": ["b" * 64] * n,
            "feature_mapping_sha256": ["c" * 64] * n,
            "congestive_heart_failure": [False] * n,
            "baseline_cvd_status_with_chf": ["absent"] * n,
            "chf_sensitivity_available": [True] * n,
            "cohort_chf_sensitivity_eligible": [True] * n,
        }
    )
    if split is not None:
        frame["split"] = split
    return frame


def test_stage3_audit_writes_protocol_outputs(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("pyarrow")

    def fake_r_gate(cohorts, **_kwargs):
        rows = []
        for name, frame in cohorts.items():
            rows.append(
                {
                    "cohort": name,
                    "n_nominal": len(frame),
                    "raw_events": int(frame["cvd_death_5y"].sum()),
                    "p_hat": float(frame["cvd_death_5y"].mean()),
                    "se": 0.001,
                    "ci_low": 0.001,
                    "ci_high": 0.003,
                    "max_ci_half_width": 0.001,
                    "deff_y": 1.0,
                    "deff_y_gate": 1.0,
                    "n_eff_design": float(len(frame)),
                    "n_eff_kish": float(len(frame)),
                    "design_df": 10,
                    "expected_r2_cs": 0.01 if name == "nhis_development" else pd.NA,
                    "pmsampsize_required_n": 100 if name == "nhis_development" else pd.NA,
                    "required_n_nominal": 100 if name == "nhis_development" else pd.NA,
                    "pass_information": name != "nhis_development" or len(frame) >= 100,
                    "pass_risk_precision": True,
                    "pass_design_df": True,
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr("cvd_risk.reporting.audit._run_r_survey_design_gate", fake_r_gate)
    nhis_development = _cohort(600, 500, split="development")
    nhis_development["survey_year"] = [1997 + (index % 13) for index in range(600)]
    nhis_temporal = _cohort(250, 200, split="temporal_test")
    nhis_temporal["survey_year"] = [2010 + (index % 5) for index in range(250)]
    nhis_temporal["participant_id"] = [f"temporal-{index}" for index in range(len(nhis_temporal))]
    nhis = pd.concat([nhis_development, nhis_temporal], ignore_index=True)
    nhanes = _cohort(120, 100)
    cycles = [
        "1999-2000",
        "2001-2002",
        "2003-2004",
        "2005-2006",
        "2007-2008",
        "2009-2010",
        "2011-2012",
        "2013-2014",
    ]
    nhanes["survey_cycle"] = [cycles[index % 8] for index in range(120)]
    nhanes["survey_year"] = nhanes["survey_cycle"].str.slice(0, 4).astype(int)
    nhanes["strata"] = nhanes["survey_cycle"].map(
        {cycle: index + 1 for index, cycle in enumerate(cycles)}
    )
    nhis_path = tmp_path / "nhis.parquet"
    nhanes_path = tmp_path / "nhanes.parquet"
    nhis.to_parquet(nhis_path, index=False)
    nhanes.to_parquet(nhanes_path, index=False)
    flow = pd.DataFrame(
        {
            "step_order": [1, 2],
            "step": ["raw", "eligible"],
            "n_participants": [1000, 850],
            "removed_from_previous": [0, 150],
        }
    )
    nhis_flow = tmp_path / "nhis_flow.csv"
    nhanes_flow = tmp_path / "nhanes_flow.csv"
    flow.assign(n_participants=[1000, len(nhis)]).to_csv(nhis_flow, index=False)
    flow.assign(
        n_participants=[200, len(nhanes)],
        removed_from_previous=[0, 200 - len(nhanes)],
    ).to_csv(nhanes_flow, index=False)

    result = run_stage3_audit(
        nhis_cohort_path=nhis_path,
        nhanes_cohort_path=nhanes_path,
        nhis_flow_path=nhis_flow,
        nhanes_flow_path=nhanes_flow,
        protocol_discrepancies_path=_protocol_registry(tmp_path, resolved=True),
        reports_dir=tmp_path / "reports",
        effective_degrees_of_freedom=20,
    )

    assert result.ready_for_modeling
    assert (tmp_path / "reports/tables/event_counts.csv").exists()
    assert (tmp_path / "reports/figures/age_distribution.png").exists()
    assert (tmp_path / "reports/stop_gate.json").exists()
    assert (tmp_path / "reports/feasibility_report.md").exists()
    assert (tmp_path / "reports/tables/protocol_discrepancies.csv").exists()
    assert (tmp_path / "reports/tables/baseline_cvd_sensitivity.csv").exists()
    assert (tmp_path / "reports/tables/survey_design_precision.csv").exists()
    assert (tmp_path / "reports/tables/huq_hsd_disagreement.csv").exists()
    status = json.loads((tmp_path / "reports/stop_gate.json").read_text(encoding="utf-8"))
    assert status["protocol_discrepancies_ready"] is True
    assert "protocol_discrepancies" in status["inputs"]
    assert "protocol_discrepancies_table" in status["outputs"]
    assert "baseline_cvd_sensitivity" in status["outputs"]
    assert "survey_design_precision" in status["outputs"]
    assert "huq_hsd_disagreement" in status["outputs"]

    blocked_result = run_stage3_audit(
        nhis_cohort_path=nhis_path,
        nhanes_cohort_path=nhanes_path,
        nhis_flow_path=nhis_flow,
        nhanes_flow_path=nhanes_flow,
        protocol_discrepancies_path=_protocol_registry(tmp_path, resolved=False),
        reports_dir=tmp_path / "reports_blocked",
        effective_degrees_of_freedom=20,
    )
    assert not blocked_result.ready_for_modeling
    protocol_check = blocked_result.details.loc[
        blocked_result.details["check"] == "protocol_discrepancy:nhanes_pooled_weight_1999_2002"
    ].iloc[0]
    assert protocol_check["status"] == "blocked_unresolved_protocol_discrepancy"
    blocked_status = json.loads(
        (tmp_path / "reports_blocked/stop_gate.json").read_text(encoding="utf-8")
    )
    assert blocked_status["protocol_discrepancies_ready"] is False
    assert blocked_status["unresolved_blocking_protocol_discrepancies"] == [
        "nhanes_pooled_weight_1999_2002"
    ]
    with pytest.raises(FileExistsError, match="overwrite is disabled"):
        run_stage3_audit(
            nhis_cohort_path=nhis_path,
            nhanes_cohort_path=nhanes_path,
            nhis_flow_path=nhis_flow,
            nhanes_flow_path=nhanes_flow,
            protocol_discrepancies_path=_protocol_registry(tmp_path, resolved=True),
            reports_dir=tmp_path / "reports",
            effective_degrees_of_freedom=20,
        )


def test_stage3_audit_rejects_temporal_split_leakage(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    nhis = _cohort(2, 1, split="development")
    nhis.loc[1, "survey_year"] = 2014
    nhanes = _cohort(2, 1)
    nhis_path = tmp_path / "nhis.parquet"
    nhanes_path = tmp_path / "nhanes.parquet"
    nhis.to_parquet(nhis_path, index=False)
    nhanes.to_parquet(nhanes_path, index=False)
    flow = pd.DataFrame(
        {
            "step_order": [1],
            "step": ["eligible"],
            "n_participants": [2],
            "removed_from_previous": [0],
        }
    )
    nhis_flow = tmp_path / "nhis_flow.csv"
    nhanes_flow = tmp_path / "nhanes_flow.csv"
    flow.to_csv(nhis_flow, index=False)
    flow.to_csv(nhanes_flow, index=False)
    with pytest.raises(ValueError, match="fixed temporal protocol"):
        run_stage3_audit(
            nhis_cohort_path=nhis_path,
            nhanes_cohort_path=nhanes_path,
            nhis_flow_path=nhis_flow,
            nhanes_flow_path=nhanes_flow,
            protocol_discrepancies_path=_protocol_registry(tmp_path, resolved=False),
            reports_dir=tmp_path / "reports",
        )


def test_stage3_audit_blocks_unverified_nhis_codebook(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    nhis = _cohort(2, 1, split="development")
    nhis["codebook_verification_status"] = "pending_user_extract"
    nhanes = _cohort(2, 1)
    nhis_path = tmp_path / "nhis.parquet"
    nhanes_path = tmp_path / "nhanes.parquet"
    nhis.to_parquet(nhis_path, index=False)
    nhanes.to_parquet(nhanes_path, index=False)
    flow = pd.DataFrame(
        {
            "step_order": [1],
            "step": ["eligible"],
            "n_participants": [2],
            "removed_from_previous": [0],
        }
    )
    nhis_flow = tmp_path / "nhis_flow.csv"
    nhanes_flow = tmp_path / "nhanes_flow.csv"
    flow.to_csv(nhis_flow, index=False)
    flow.to_csv(nhanes_flow, index=False)
    with pytest.raises(ValueError, match="not verified"):
        run_stage3_audit(
            nhis_cohort_path=nhis_path,
            nhanes_cohort_path=nhanes_path,
            nhis_flow_path=nhis_flow,
            nhanes_flow_path=nhanes_flow,
            protocol_discrepancies_path=_protocol_registry(tmp_path, resolved=False),
            reports_dir=tmp_path / "reports",
        )


def test_final_cohort_contract_rejects_noncanonical_categories() -> None:
    frame = _cohort(2, 1)
    frame["sex"] = "alien"
    with pytest.raises(CanonicalSchemaError, match="sex"):
        _validate_cohort_contract(frame, "bad")


def test_load_flow_rejects_inconsistent_removal_arithmetic(tmp_path: Path) -> None:
    path = tmp_path / "bad_flow.csv"
    pd.DataFrame(
        {
            "step_order": [1, 2],
            "step": ["raw", "eligible"],
            "n_participants": [100, 80],
            "removed_from_previous": [0, 19],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="removed_from_previous is inconsistent"):
        _load_flow(path, "test")


def test_load_flow_rejects_duplicate_steps(tmp_path: Path) -> None:
    path = tmp_path / "duplicate_flow.csv"
    pd.DataFrame(
        {
            "step_order": [1, 1],
            "step": ["raw", "raw"],
            "n_participants": [100, 100],
            "removed_from_previous": [0, 0],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate step_order"):
        _load_flow(path, "test")
