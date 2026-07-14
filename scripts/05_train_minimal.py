"""Stage-5 guard: training is intentionally unavailable before approval."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from cvd_risk.runtime import DEFAULT_CONFIG_PATH, initialize_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    config = initialize_runtime(args.config)
    table_dir = Path(config.get("paths", {}).get("report_tables_dir", "reports/tables"))
    status_path = table_dir.parent / "stop_gate.json"
    if not status_path.exists():
        print("STOP: reports/stop_gate.json is absent. Complete stages 0–3 first.")
        return 2
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("schema_version") != 1:
        print("STOP: unsupported or missing stop-gate provenance schema.")
        return 2
    expected_inputs = {
        "nhis_cohort",
        "nhanes_cohort",
        "nhis_flow",
        "nhanes_flow",
        "experiment_config",
        "research_protocol",
        "feature_mapping",
        "harmonization_differences_input",
        "protocol_discrepancies",
    }
    inputs = status.get("inputs")
    if not isinstance(inputs, dict) or not expected_inputs.issubset(inputs):
        print("STOP: stop-gate provenance is missing mandatory input bindings.")
        return 2
    audited_config = Path(inputs["experiment_config"]["path"]).resolve()
    if args.config.resolve() != audited_config:
        print(
            "STOP: training config differs from the config used for cohort audit: "
            f"{args.config.resolve()} != {audited_config}"
        )
        return 2
    expected_outputs = {
        "cohort_flow_nhis",
        "cohort_flow_nhanes",
        "event_counts",
        "missingness_by_source",
        "feature_availability",
        "data_discrepancies",
        "harmonization_differences",
        "race_ethnicity_event_counts",
        "protocol_discrepancies_table",
        "stop_gate_table",
        "outcome_by_year",
        "age_distribution",
        "feasibility_report",
    }
    outputs = status.get("outputs")
    if not isinstance(outputs, dict) or not expected_outputs.issubset(outputs):
        print("STOP: stop-gate provenance does not include all mandatory outputs.")
        return 2
    bound_files = {**inputs, **outputs}
    for name, entry in bound_files.items():
        path = Path(entry["path"])
        if not path.is_file():
            print(f"STOP: audited input {name!r} is missing: {path}")
            return 2
        digest = _sha256(path)
        if digest != entry.get("sha256"):
            print(f"STOP: audited input {name!r} changed after cohort audit: {path}")
            return 2
    if status.get("protocol_discrepancies_ready") is not True:
        unresolved = status.get("unresolved_blocking_protocol_discrepancies", [])
        print(f"STOP: blocking protocol discrepancies require an explicit resolution: {unresolved}")
        return 2
    if not status.get("ready_for_modeling", False):
        print(status.get("summary", "STOP: cohort feasibility gate did not pass."))
        return 2
    print(
        "STOP: the feasibility gate passed, but stage 5 has not been authorized "
        "in this run. Review the audit and explicitly approve model training."
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
