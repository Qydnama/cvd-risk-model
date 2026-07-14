"""Validated registry of protocol discrepancies that can block modeling."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

NHANES_WEIGHT_DISCREPANCY_ID = "nhanes_pooled_weight_1999_2002"
PROTOCOL_DISCREPANCY_COLUMNS = (
    "id",
    "title",
    "status",
    "blocking",
    "locked_protocol_rule",
    "official_guidance",
    "recommended_resolution",
    "resolution",
    "official_sources",
)
_ALLOWED_STATUSES = {"awaiting_user_decision", "resolved"}


def _required_text(entry: Mapping[str, Any], field: str, *, item_id: str) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Protocol discrepancy {item_id!r} requires non-empty text field {field!r}"
        )
    return value.strip()


def load_protocol_discrepancies(path: str | Path) -> pd.DataFrame:
    """Load and strictly validate the versioned discrepancy registry."""

    registry_path = Path(path)
    if not registry_path.is_file():
        raise FileNotFoundError(f"Protocol discrepancy registry not found: {registry_path}")
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Protocol discrepancy registry root must be a mapping")
    if payload.get("schema_version") != 1:
        raise ValueError("Protocol discrepancy registry schema_version must be 1")
    entries = payload.get("discrepancies")
    if not isinstance(entries, list):
        raise ValueError("Protocol discrepancy registry must contain a list")

    rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for raw_entry in entries:
        if not isinstance(raw_entry, Mapping):
            raise ValueError("Every protocol discrepancy must be a mapping")
        item_id = _required_text(raw_entry, "id", item_id="<unknown>")
        if item_id in seen_ids:
            raise ValueError(f"Duplicate protocol discrepancy id: {item_id!r}")
        seen_ids.add(item_id)

        status = _required_text(raw_entry, "status", item_id=item_id)
        if status not in _ALLOWED_STATUSES:
            raise ValueError(f"Protocol discrepancy {item_id!r} has unsupported status {status!r}")
        blocking = raw_entry.get("blocking")
        if not isinstance(blocking, bool):
            raise ValueError(f"Protocol discrepancy {item_id!r} field 'blocking' must be boolean")
        resolution = raw_entry.get("resolution")
        if resolution is not None and (not isinstance(resolution, str) or not resolution.strip()):
            raise ValueError(f"Protocol discrepancy {item_id!r} resolution must be null or text")
        if status == "resolved" and resolution is None:
            raise ValueError(f"Resolved protocol discrepancy {item_id!r} requires a resolution")
        sources = raw_entry.get("official_sources")
        if (
            not isinstance(sources, list)
            or not sources
            or not all(
                isinstance(source, str) and source.startswith("https://") for source in sources
            )
        ):
            raise ValueError(f"Protocol discrepancy {item_id!r} requires HTTPS official_sources")

        rows.append(
            {
                "id": item_id,
                "title": _required_text(raw_entry, "title", item_id=item_id),
                "status": status,
                "blocking": blocking,
                "locked_protocol_rule": _required_text(
                    raw_entry, "locked_protocol_rule", item_id=item_id
                ),
                "official_guidance": _required_text(
                    raw_entry, "official_guidance", item_id=item_id
                ),
                "recommended_resolution": _required_text(
                    raw_entry, "recommended_resolution", item_id=item_id
                ),
                "resolution": resolution.strip() if isinstance(resolution, str) else pd.NA,
                "official_sources": " | ".join(sources),
            }
        )

    if NHANES_WEIGHT_DISCREPANCY_ID not in seen_ids:
        raise ValueError(
            "Protocol discrepancy registry must retain the known NHANES pooled "
            f"weight issue {NHANES_WEIGHT_DISCREPANCY_ID!r}"
        )
    known = next(row for row in rows if row["id"] == NHANES_WEIGHT_DISCREPANCY_ID)
    if known["blocking"] is not True:
        raise ValueError(
            "The known NHANES pooled-weight discrepancy must remain blocking; "
            "resolve it with a documented decision instead of disabling the guard"
        )
    return pd.DataFrame(rows, columns=PROTOCOL_DISCREPANCY_COLUMNS)


def protocol_discrepancy_checks(discrepancies: pd.DataFrame) -> pd.DataFrame:
    """Convert discrepancy state into stop-gate checks."""

    missing = set(PROTOCOL_DISCREPANCY_COLUMNS).difference(discrepancies.columns)
    if missing:
        raise KeyError(f"Protocol discrepancy table is missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for entry in discrepancies.to_dict(orient="records"):
        resolved = entry["status"] == "resolved"
        blocking = bool(entry["blocking"])
        rows.append(
            {
                "check": f"protocol_discrepancy:{entry['id']}",
                "observed": entry["status"],
                "minimum": "resolved" if blocking else "not_blocking",
                "status": (
                    "blocked_unresolved_protocol_discrepancy"
                    if blocking and not resolved
                    else "pass"
                ),
            }
        )
    return pd.DataFrame(rows, columns=("check", "observed", "minimum", "status"))
