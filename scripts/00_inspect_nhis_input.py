"""Inspect and fingerprint the manual IPUMS NHIS extract without approving labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cvd_risk.data.harmonize import (
    VERIFIED_CODEBOOK_STATUS,
    load_feature_mapping,
    verify_codebook_binding,
)
from cvd_risk.data.load_nhis import NHISInputError, discover_nhis_input, load_nhis
from cvd_risk.data.manifest import register_local_input
from cvd_risk.runtime import DEFAULT_CONFIG_PATH, initialize_runtime


def _missing_codes(specification: dict) -> set[object]:
    codes: set[object] = set()
    for values in specification.get("missing_codes", {}).values():
        codes.update(values)
    return codes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mapping", type=Path, default=Path("configs/feature_mapping.yaml"))
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/nhis_input_inspection.md"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = initialize_runtime(args.config)
    paths = config.get("paths", {})
    raw_dir = args.raw_dir or Path(paths.get("nhis_raw_dir", "data/raw/nhis"))
    try:
        inputs = discover_nhis_input(raw_dir)
        raw = load_nhis(raw_dir, require_all_years=True)
    except (FileNotFoundError, NHISInputError) as exc:
        print(f"STOP: manual IPUMS NHIS input is not ready: {exc}")
        print(
            "Place one .csv.gz, .csv, or .dta extract and its codebook in "
            f"{raw_dir}, then rerun this command."
        )
        return 2
    mapping = load_feature_mapping(args.mapping)
    verification = mapping.get("codebook_verification", {})
    status = verification.get("status", "unspecified")
    race_candidates = ["RACE_ETHNICITY", "RACENEW", "RACEA", "HISPETH"]
    available_race = [column for column in race_candidates if column in raw]
    expected_primary_years = set(range(1997, 2015))
    baseline_availability: dict[str, list[int]] = {}
    for name, specification in mapping.get("variables", {}).items():
        if specification.get("role") != "baseline_cvd_exclusion":
            continue
        source = specification["source"]
        substantive = raw[source].notna() & ~raw[source].isin(_missing_codes(specification))
        baseline_availability[name] = sorted(
            int(year) for year in raw.loc[substantive, "YEAR"].unique()
        )
    incomplete_baseline = {
        name: years
        for name, years in baseline_availability.items()
        if not expected_primary_years.issubset(years)
    }

    manifest_path = Path(paths.get("data_manifest", "configs/data_manifest.yaml"))
    extract_entry = register_local_input(
        manifest_path,
        inputs.data_path,
        dataset="nhis",
        component="ipums_extract",
        period="1997-2018",
        source_url="https://nhis.ipums.org/nhis/",
    )
    codebook_entry = register_local_input(
        manifest_path,
        inputs.codebook_path,
        dataset="nhis",
        component="ipums_codebook",
        period="1997-2018",
        source_url="https://nhis.ipums.org/nhis/",
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# IPUMS NHIS input inspection",
        "",
        f"- Data: `{inputs.data_path}`",
        f"- Data rows: {len(raw):,}",
        f"- Data SHA-256: `{extract_entry['sha256']}`",
        f"- Codebook: `{inputs.codebook_path}`",
        f"- Codebook SHA-256: `{codebook_entry['sha256']}`",
        f"- Years: {int(raw['YEAR'].min())}–{int(raw['YEAR'].max())}",
        f"- Mapping status: `{status}`",
        f"- Available race/ethnicity inputs: {available_race or 'none'}",
        "",
        "## Baseline CVD variable availability",
        "",
        "| Canonical variable | Years with substantive responses |",
        "|---|---|",
        *(
            f"| `{name}` | {years[0]}–{years[-1]} |"
            if years and years == list(range(years[0], years[-1] + 1))
            else f"| `{name}` | {', '.join(map(str, years)) or 'none'} |"
            for name, years in baseline_availability.items()
        ),
    ]
    if status == VERIFIED_CODEBOOK_STATUS:
        verify_codebook_binding(
            inputs.data_path,
            inputs.codebook_path,
            mapping_path=args.mapping,
        )
        lines.extend(
            [
                "",
                "## Verification result",
                "",
                "PASS: mapping was manually reviewed and is bound to these exact "
                "extract/DDI SHA-256 digests.",
            ]
        )
        if incomplete_baseline:
            lines.extend(
                [
                    "",
                    "## Cohort-contract result",
                    "",
                    "BLOCKED: at least one required baseline-CVD exclusion is not "
                    "available in every 1997–2014 primary year. See "
                    "`configs/protocol_discrepancies.yaml` before cohort construction.",
                ]
            )
    else:
        lines.extend(
            [
                "",
                "## Required manual review",
                "",
                "Compare every source code, label, missing category, question universe, "
                "and year availability against this exact codebook. In particular review "
                "yes/no direction, BMI scaling, smoking categories, mortality codes, "
                "baseline-CVD fields, and race/ethnicity comparability.",
                "",
                "Only after that review may `configs/feature_mapping.yaml` be changed to:",
                "",
                "```yaml",
                "codebook_verification:",
                f"  status: {VERIFIED_CODEBOOK_STATUS}",
                "  required_before_cohort_audit: true",
                f"  extract_sha256: {extract_entry['sha256']}",
                f"  codebook_sha256: {codebook_entry['sha256']}",
                "```",
                "",
                "This inspection command intentionally does not approve the mapping.",
            ]
        )
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.report}")

    if status != VERIFIED_CODEBOOK_STATUS:
        print(
            "STOP: mapping labels are still pending manual codebook review. "
            "NHIS cohort construction remains blocked."
        )
        return 2
    if incomplete_baseline:
        print(
            "STOP: required baseline-CVD variables are not available in every primary "
            f"year: {incomplete_baseline}"
        )
        return 3
    print("PASS: mapping is bound to this exact extract and codebook.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
