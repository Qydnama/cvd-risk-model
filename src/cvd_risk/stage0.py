"""Stage-0 environment, protocol-lock and quarantine audit."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cvd_risk.provenance import sha256_file, verify_quarantine
from cvd_risk.runtime import DEFAULT_CONFIG_PATH, initialize_runtime


def _atomic_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def run_stage0_check(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit Stage 0 and emit a machine-readable hard status."""

    config_path = Path(config_path)
    config = initialize_runtime(config_path)
    paths = config["paths"]
    protocol_path = Path("CODEX_CVD_MODEL_IMPLEMENTATION_PLAN.md")
    expected_protocol_hash = config["experiment"]["protocol_sha256"]
    observed_protocol_hash = sha256_file(protocol_path)
    if observed_protocol_hash != expected_protocol_hash:
        raise ValueError(
            "Research protocol SHA-256 differs from configs/experiment.yaml: "
            f"{observed_protocol_hash} != {expected_protocol_hash}"
        )

    quarantine_manifest = Path(paths["quarantine_manifest"])
    quarantine = verify_quarantine(quarantine_manifest)
    rscript = shutil.which("Rscript")
    checks: dict[str, dict[str, Any]] = {
        "protocol_lock": {
            "status": "PASS",
            "path": str(protocol_path),
            "sha256": observed_protocol_hash,
        },
        "quarantine": {
            "status": "PASS",
            "manifest": str(quarantine_manifest),
            "artifact_count": len(quarantine["artifacts"]),
        },
    }

    if rscript is None:
        status = config["statistical_backend"]["missing_rscript_status"]
        checks["rscript"] = {"status": status, "executable": None}
        r_version = None
    else:
        completed = subprocess.run(
            [rscript, "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        r_version = (completed.stdout or completed.stderr).strip()
        checks["rscript"] = {"status": "PASS", "executable": rscript, "version": r_version}
        required_locks = [Path(".R-version"), Path("renv.lock")]
        missing_locks = [str(path) for path in required_locks if not path.is_file()]
        if missing_locks:
            status = "BLOCKED_R_LOCKS_MISSING"
            checks["r_locks"] = {"status": status, "missing": missing_locks}
        else:
            status = "PASS"
            checks["r_locks"] = {"status": "PASS", "files": [str(p) for p in required_locks]}

    payload: dict[str, Any] = {
        "schema_version": 1,
        "stage": 0,
        "protocol_version": config["experiment"]["protocol_version"],
        "generated_at": datetime.now(UTC).isoformat(),
        "status": status,
        "ready_for_stage1": status == "PASS",
        "reference_backend": "R",
        "r_version": r_version,
        "r_packages": config["statistical_backend"]["r_packages"],
        "checks": checks,
    }
    destination = Path(output_path or paths["stage0_status"])
    _atomic_json(payload, destination)
    return payload


__all__ = ["run_stage0_check"]
