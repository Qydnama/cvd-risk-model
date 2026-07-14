"""Reproducible downloader for public CDC/NCHS NHANES and mortality files."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import requests
import yaml

LOGGER = logging.getLogger(__name__)
RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
VALID_TIERS = {"fast", "extended"}
VALID_FORMATS = {"xpt", "fwf"}


class DownloadError(RuntimeError):
    """Raised when a configured file cannot be acquired safely."""


class ConfigError(ValueError):
    """Raised when ``nhanes_files.yaml`` violates its schema."""


def tiers_for_profile(profile: str) -> set[str]:
    """Map a model profile to the raw tiers required to build it."""

    if profile == "fast":
        return {"fast"}
    if profile in {"extended", "all"}:
        return {"fast", "extended"}
    raise ConfigError(f"Unknown NHANES profile {profile!r}")


@dataclass(frozen=True)
class FileSpec:
    cycle: str
    component: str
    tier: str
    format: str
    filename: str
    url: str
    logical_components: tuple[str, ...] = ()

    @property
    def dataset(self) -> str:
        return "linked_mortality" if self.format == "fwf" else "nhanes"


@dataclass(frozen=True)
class DownloadResult:
    cycle: str
    component: str
    path: str
    status: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class PreflightResult:
    cycle: str
    component: str
    url: str
    status_code: int
    content_length: int | None
    content_type: str | None


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _remote_basename(url: str) -> str:
    return Path(unquote(urlparse(url).path)).name


def _validate_file_spec(spec: FileSpec, allowed_hosts: set[str] | None = None) -> None:
    parsed = urlparse(spec.url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ConfigError(f"{spec.cycle}/{spec.component}: URL must be absolute HTTPS: {spec.url}")
    if allowed_hosts and (parsed.hostname or "").lower() not in allowed_hosts:
        raise ConfigError(f"{spec.cycle}/{spec.component}: non-approved host in {spec.url}")
    remote_name = _remote_basename(spec.url)
    if remote_name != spec.filename:
        raise ConfigError(
            f"{spec.cycle}/{spec.component}: expected filename {spec.filename!r}, "
            f"but URL points to {remote_name!r}"
        )
    if spec.tier not in VALID_TIERS:
        raise ConfigError(f"{spec.cycle}/{spec.component}: unsupported tier {spec.tier!r}")
    if spec.format not in VALID_FORMATS:
        raise ConfigError(f"{spec.cycle}/{spec.component}: unsupported format {spec.format!r}")


def load_file_specs(
    config_path: str | Path,
    *,
    cycles: Iterable[str] | None = None,
    tiers: Iterable[str] | None = None,
) -> list[FileSpec]:
    """Read and strictly validate the explicit CDC file inventory."""

    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        raise ConfigError(f"{config_path}: expected schema_version: 1")
    raw_cycles = raw.get("cycles")
    if not isinstance(raw_cycles, Mapping) or not raw_cycles:
        raise ConfigError(f"{config_path}: 'cycles' must be a non-empty mapping")

    selected_cycles = set(cycles) if cycles is not None else set(raw_cycles)
    unknown_cycles = selected_cycles.difference(raw_cycles)
    if unknown_cycles:
        raise ConfigError(f"Unknown NHANES cycle(s): {sorted(unknown_cycles)}")
    selected_tiers = set(tiers) if tiers is not None else VALID_TIERS
    unknown_tiers = selected_tiers.difference(VALID_TIERS)
    if unknown_tiers:
        raise ConfigError(f"Unknown tier(s): {sorted(unknown_tiers)}")
    allowed_hosts = {str(host).lower() for host in raw.get("allowed_hosts", [])} or None

    specs: list[FileSpec] = []
    for cycle, cycle_data in raw_cycles.items():
        if cycle not in selected_cycles:
            continue
        files = cycle_data.get("files") if isinstance(cycle_data, Mapping) else None
        if not isinstance(files, list) or not files:
            raise ConfigError(f"{cycle}: 'files' must be a non-empty list")
        seen_components: set[str] = set()
        seen_filenames: set[str] = set()
        for item in files:
            if not isinstance(item, Mapping):
                raise ConfigError(f"{cycle}: every file entry must be a mapping")
            try:
                spec = FileSpec(
                    cycle=str(cycle),
                    component=str(item["component"]),
                    tier=str(item["tier"]),
                    format=str(item["format"]),
                    filename=str(item["filename"]),
                    url=str(item["url"]),
                    logical_components=tuple(
                        str(value) for value in item.get("logical_components", [])
                    ),
                )
            except KeyError as exc:
                raise ConfigError(f"{cycle}: missing required key {exc.args[0]!r}") from exc
            _validate_file_spec(spec, allowed_hosts)
            if spec.component in seen_components:
                raise ConfigError(f"{cycle}: duplicate component {spec.component!r}")
            filename_key = spec.filename.casefold()
            if filename_key in seen_filenames:
                raise ConfigError(f"{cycle}: duplicate filename {spec.filename!r}")
            seen_components.add(spec.component)
            seen_filenames.add(filename_key)
            if spec.tier in selected_tiers:
                specs.append(spec)

    return specs


class ManifestStore:
    """Small atomic YAML manifest compatible with ``configs/data_manifest.yaml``."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
        else:
            loaded = {}
        if not isinstance(loaded, dict):
            raise DownloadError(f"Manifest must contain a mapping: {self.path}")
        if loaded.get("schema_version", 1) != 1:
            raise DownloadError(f"Unsupported manifest schema in {self.path}")
        files = loaded.get("files", [])
        if not isinstance(files, list):
            raise DownloadError(f"Manifest 'files' must be a list: {self.path}")
        self.data: dict[str, Any] = {
            "schema_version": 1,
            "generated_at": loaded.get("generated_at"),
            "files": files,
        }

    def find(self, spec: FileSpec) -> dict[str, Any] | None:
        for entry in self.data["files"]:
            if (
                isinstance(entry, dict)
                and entry.get("url") == spec.url
                and entry.get("cycle") == spec.cycle
                and entry.get("component") == spec.component
            ):
                return entry
        return None

    def record(
        self,
        spec: FileSpec,
        path: Path,
        *,
        status: str,
        sha256: str,
        size_bytes: int,
    ) -> None:
        entry = {
            "source": "CDC/NCHS",
            "dataset": spec.dataset,
            "cycle": spec.cycle,
            "component": spec.component,
            "url": spec.url,
            "local_path": path.as_posix(),
            "downloaded_at": _utc_now(),
            "size_bytes": int(size_bytes),
            "sha256": sha256,
            "status": status,
        }
        current = self.find(spec)
        if current is None:
            self.data["files"].append(entry)
        else:
            current.clear()
            current.update(entry)
        self.data["generated_at"] = _utc_now()
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            yaml.safe_dump(self.data, handle, sort_keys=False, allow_unicode=True)
        os.replace(temporary, self.path)


def _content_disposition_filename(headers: Mapping[str, str]) -> str | None:
    value = headers.get("Content-Disposition") or headers.get("content-disposition")
    if not value:
        return None
    match = re.search(r"filename\*?=(?:UTF-8''|\")?([^\";]+)", value, flags=re.IGNORECASE)
    return unquote(match.group(1).strip()) if match else None


def _validate_response_metadata(spec: FileSpec, response: Any) -> None:
    disposition_name = _content_disposition_filename(response.headers)
    if disposition_name and disposition_name != spec.filename:
        raise DownloadError(
            f"{spec.cycle}/{spec.component}: CDC returned filename "
            f"{disposition_name!r}; expected {spec.filename!r}"
        )
    content_type = (response.headers.get("Content-Type") or "").lower()
    if "text/html" in content_type:
        raise DownloadError(
            f"{spec.cycle}/{spec.component}: URL returned HTML instead of a data file"
        )


def _validate_local_file(spec: FileSpec, path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise DownloadError(f"{spec.cycle}/{spec.component}: empty or missing file {path}")
    with path.open("rb") as handle:
        prefix = handle.read(80)
    if spec.format == "xpt" and not prefix.startswith(b"HEADER RECORD"):
        raise DownloadError(f"{spec.cycle}/{spec.component}: {path} is not a SAS XPORT file")
    if spec.format == "fwf":
        first_line = prefix.splitlines()[0] if prefix else b""
        # NCHS omits trailing blanks on records whose final fields are missing,
        # so a legitimate row can be shorter than the nominal 48 columns.
        valid_identifier = first_line[:6].strip().isdigit()
        valid_eligibility = len(first_line) >= 15 and first_line[14:15] in {b"1", b"2", b"3"}
        if not valid_identifier or not valid_eligibility:
            raise DownloadError(
                f"{spec.cycle}/{spec.component}: {path} is not an NCHS fixed-width LMF"
            )


def _existing_result(
    spec: FileSpec,
    destination: Path,
    manifest: ManifestStore | None,
) -> DownloadResult | None:
    if not destination.exists():
        return None
    try:
        _validate_local_file(spec, destination)
    except DownloadError:
        # Preserve the suspect file until a complete replacement has passed
        # validation; the downloader writes to a sibling .part file first.
        return None
    size = destination.stat().st_size
    digest = sha256_file(destination)
    entry = manifest.find(spec) if manifest else None
    if entry and (entry.get("size_bytes") != size or entry.get("sha256") != digest):
        return None
    if manifest and entry is None:
        manifest.record(spec, destination, status="existing", sha256=digest, size_bytes=size)
    return DownloadResult(spec.cycle, spec.component, str(destination), "cached", size, digest)


def download_file(
    spec: FileSpec,
    destination: str | Path,
    *,
    manifest: ManifestStore | None = None,
    session: Any | None = None,
    attempts: int = 3,
    timeout: tuple[float, float] = (10.0, 60.0),
    backoff_seconds: float = 1.0,
    force: bool = False,
) -> DownloadResult:
    """Download one file atomically, with retries, checksum, and idempotency."""

    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    _validate_file_spec(spec)
    destination = Path(destination)
    if not force:
        cached = _existing_result(spec, destination, manifest)
        if cached is not None:
            return cached
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.part")
    client = session or requests.Session()
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        partial.unlink(missing_ok=True)
        response = None
        try:
            response = client.get(
                spec.url,
                stream=True,
                timeout=timeout,
                allow_redirects=True,
                headers={"Accept-Encoding": "identity"},
            )
            if response.status_code in RETRYABLE_STATUS_CODES:
                raise requests.HTTPError(
                    f"retryable HTTP {response.status_code}", response=response
                )
            response.raise_for_status()
            _validate_response_metadata(spec, response)
            expected_size = response.headers.get("Content-Length")
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            if expected_size is not None and partial.stat().st_size != int(expected_size):
                raise DownloadError(
                    f"{spec.cycle}/{spec.component}: received {partial.stat().st_size} bytes, "
                    f"expected {expected_size}"
                )
            _validate_local_file(spec, partial)
            digest = sha256_file(partial)
            size = partial.stat().st_size
            os.replace(partial, destination)
            if manifest:
                manifest.record(
                    spec,
                    destination,
                    status="downloaded",
                    sha256=digest,
                    size_bytes=size,
                )
            return DownloadResult(
                spec.cycle,
                spec.component,
                str(destination),
                "downloaded",
                size,
                digest,
            )
        except (requests.RequestException, OSError, ValueError, DownloadError) as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(backoff_seconds * (2 ** (attempt - 1)))
        finally:
            if response is not None:
                response.close()
    raise DownloadError(
        f"Failed to download {spec.cycle}/{spec.component} after {attempts} attempts: {last_error}"
    ) from last_error


def preflight_file(
    spec: FileSpec,
    *,
    session: Any | None = None,
    attempts: int = 3,
    timeout: tuple[float, float] = (10.0, 30.0),
    backoff_seconds: float = 1.0,
) -> PreflightResult:
    """Validate a remote file using headers only; no response body is saved."""

    client = session or requests.Session()
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        response = None
        try:
            response = client.head(
                spec.url,
                timeout=timeout,
                allow_redirects=True,
                headers={"Accept-Encoding": "identity"},
            )
            if response.status_code == 405:
                response.close()
                response = client.get(
                    spec.url,
                    stream=True,
                    timeout=timeout,
                    allow_redirects=True,
                    headers={"Accept-Encoding": "identity"},
                )
            if response.status_code in RETRYABLE_STATUS_CODES:
                raise requests.HTTPError(
                    f"retryable HTTP {response.status_code}", response=response
                )
            response.raise_for_status()
            _validate_response_metadata(spec, response)
            length_header = response.headers.get("Content-Length")
            content_length = int(length_header) if length_header is not None else None
            if content_length is not None and content_length <= 0:
                raise DownloadError(f"{spec.cycle}/{spec.component}: CDC reported an empty file")
            return PreflightResult(
                cycle=spec.cycle,
                component=spec.component,
                url=spec.url,
                status_code=int(response.status_code),
                content_length=content_length,
                content_type=response.headers.get("Content-Type"),
            )
        except (requests.RequestException, ValueError, DownloadError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(backoff_seconds * (2 ** (attempt - 1)))
        finally:
            if response is not None:
                response.close()
    raise DownloadError(
        f"Preflight failed for {spec.cycle}/{spec.component} after "
        f"{attempts} attempts: {last_error}"
    ) from last_error


def download_specs(
    specs: Sequence[FileSpec],
    *,
    output_dir: str | Path,
    manifest_path: str | Path,
    session: Any | None = None,
    **download_kwargs: Any,
) -> list[DownloadResult]:
    manifest = ManifestStore(manifest_path)
    output_dir = Path(output_dir)
    return [
        download_file(
            spec,
            output_dir / spec.cycle / spec.filename,
            manifest=manifest,
            session=session,
            **download_kwargs,
        )
        for spec in specs
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/nhanes_files.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/nhanes"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/data_manifest.yaml"))
    parser.add_argument(
        "--cycle", action="append", help="Cycle such as 1999-2000; repeat as needed"
    )
    parser.add_argument("--profile", choices=("fast", "extended", "all"), default="all")
    parser.add_argument(
        "--preflight", action="store_true", help="Check remote headers without downloading"
    )
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--read-timeout", type=float, default=60.0)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    tiers = tiers_for_profile(args.profile)
    specs = load_file_specs(args.config, cycles=args.cycle, tiers=tiers)
    session = requests.Session()
    session.headers["User-Agent"] = "cvd-risk-research/0.1 (CDC public data downloader)"
    timeout = (args.connect_timeout, args.read_timeout)
    if args.preflight:
        for spec in specs:
            preflight_result = preflight_file(
                spec, session=session, attempts=args.attempts, timeout=timeout
            )
            LOGGER.info(
                "OK %s/%s (%s bytes)",
                preflight_result.cycle,
                preflight_result.component,
                preflight_result.content_length,
            )
        return 0
    results = download_specs(
        specs,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        session=session,
        attempts=args.attempts,
        timeout=timeout,
        force=args.force,
    )
    for download_result in results:
        LOGGER.info(
            "%s %s/%s -> %s",
            download_result.status,
            download_result.cycle,
            download_result.component,
            download_result.path,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
