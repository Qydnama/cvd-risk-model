"""Run the unchanged CVD pipeline on complete synthetic research records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd

from cvd_risk.modeling.stage9_research_release import validate_minimal_v1_research_input


def verify_bundle(root: Path) -> dict[str, Any]:
    """Verify the existing manifest, runtime and hashes before trusted deserialization."""
    root = root.resolve()
    manifest = json.loads(
        (root / "artifacts/minimal_v1_research_release/release_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest["model_id"] != "minimal_v1" or not manifest["governance"]["research_use_only"]:
        raise ValueError("The demo requires the minimal_v1 research snapshot")
    if manifest["governance"]["clinical_deployment_allowed"]:
        raise ValueError("Unexpected clinical deployment policy")
    if sys.version_info[:2] != (3, 12):
        raise ValueError("The locked research runtime requires Python 3.12")
    runtime = manifest["runtime_contract"]
    for package, key in (
        ("numpy", "numpy"), ("pandas", "pandas"),
        ("joblib", "joblib"), ("scikit-learn", "scikit_learn"),
    ):
        if version(package) != runtime[key]:
            raise ValueError(f"Runtime version mismatch: {package}")
    for binding in manifest["locked_binary_files"].values():
        path = (root / binding["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Artifact path escapes the research bundle")
        if hashlib.sha256(path.read_bytes()).hexdigest() != binding["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {binding['path']}")
    contract = manifest["portable_contract"]
    schema_bytes = (root / contract["path"]).read_bytes()
    if hashlib.sha256(schema_bytes.replace(b"\r\n", b"\n")).hexdigest() != contract[
        "normalized_lf_sha256"
    ]:
        raise ValueError("Input schema checksum mismatch")
    lock = manifest["documentation_bindings"]["python_lock"]
    lock_bytes = (root / lock["path"]).read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(lock_bytes).hexdigest() != lock["normalized_lf_sha256"]:
        raise ValueError("Dependency lock checksum mismatch")
    return manifest


def predict_synthetic_records(root: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate complete inputs and perform calibrated inference without refitting."""
    manifest = verify_bundle(root)
    schema = json.loads((root / manifest["portable_contract"]["path"]).read_text("utf-8"))
    frame = pd.DataFrame(records)
    validate_minimal_v1_research_input(frame, schema)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated in NumPy 2.5.*",
            category=DeprecationWarning,
            module=r"joblib\.numpy_pickle",
        )
        model = joblib.load(root / "artifacts/minimal_v1/full_development_pipeline.joblib")
    probabilities = np.asarray(model.predict_proba(frame, calibrated=True), dtype=float)
    if probabilities.shape != (len(frame),) or not np.isfinite(probabilities).all():
        raise ValueError("The model returned invalid research probabilities")
    if not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("The model returned an out-of-range probability")
    return {
        "status": "PASS",
        "model_id": manifest["model_id"],
        "source_revision": os.environ.get("CVD_SOURCE_REVISION", "local-checkout"),
        "research_use_only": True,
        "clinical_deployment_allowed": False,
        "input_kind": "SYNTHETIC_DEMONSTRATION",
        "refit_performed": False,
        "verified_binary_artifacts": len(manifest["locked_binary_files"]),
        "predictions": [
            {"record": i + 1, "five_year_cvd_mortality_probability": float(p)}
            for i, p in enumerate(probabilities)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--input", type=Path, default=Path("devops/synthetic_inputs.json"))
    args = parser.parse_args()
    try:
        records = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
            raise ValueError("Input must be a JSON list of complete synthetic records")
        result = predict_synthetic_records(args.root.resolve(), records)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError) as error:
        print(f"Research demo failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
