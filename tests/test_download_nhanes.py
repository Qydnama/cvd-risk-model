from __future__ import annotations

import hashlib

import pytest
import requests
import yaml

from cvd_risk.data.download_nhanes import (
    ConfigError,
    FileSpec,
    ManifestStore,
    download_file,
    load_file_specs,
    preflight_file,
    tiers_for_profile,
)


class FakeResponse:
    def __init__(self, content=b"", *, status_code=200, headers=None):
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        del chunk_size
        yield self.content

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, *, gets=None, heads=None):
        self.gets = list(gets or [])
        self.heads = list(heads or [])
        self.get_calls = 0
        self.head_calls = 0

    def get(self, *args, **kwargs):
        del args, kwargs
        item = self.gets[self.get_calls]
        self.get_calls += 1
        if isinstance(item, Exception):
            raise item
        return item

    def head(self, *args, **kwargs):
        del args, kwargs
        item = self.heads[self.head_calls]
        self.head_calls += 1
        if isinstance(item, Exception):
            raise item
        return item


def _demo_spec():
    return FileSpec(
        cycle="1999-2000",
        component="DEMO",
        tier="fast",
        format="xpt",
        filename="DEMO.xpt",
        url="https://example.test/DEMO.xpt",
    )


def _lmf_spec():
    return FileSpec(
        cycle="1999-2000",
        component="LMF",
        tier="fast",
        format="fwf",
        filename="mortality.dat",
        url="https://example.test/mortality.dat",
    )


def test_official_inventory_contains_all_cycles_and_legacy_names():
    specs = load_file_specs("configs/nhanes_files.yaml")

    assert len(specs) == 100
    assert {spec.cycle for spec in specs} == {
        "1999-2000",
        "2001-2002",
        "2003-2004",
        "2005-2006",
        "2007-2008",
        "2009-2010",
        "2011-2012",
        "2013-2014",
    }
    filenames = {(spec.cycle, spec.component): spec.filename for spec in specs}
    assert filenames[("1999-2000", "TCHOL_HDL")] == "LAB13.xpt"
    assert filenames[("2001-2002", "TCHOL_HDL")] == "L13_B.xpt"
    assert filenames[("2003-2004", "GHB")] == "L10_C.xpt"
    assert filenames[("2005-2006", "TCHOL")] == "TCHOL_D.xpt"
    assert filenames[("2013-2014", "HUQ")] == "HUQ_H.xpt"


def test_extended_download_profile_includes_fast_and_extended_tiers():
    assert tiers_for_profile("extended") == {"fast", "extended"}
    specs = load_file_specs(
        "configs/nhanes_files.yaml",
        cycles=["1999-2000"],
        tiers=tiers_for_profile("extended"),
    )

    assert len(specs) == 11
    assert {spec.component for spec in specs} >= {"DEMO", "LMF", "BPX", "TCHOL_HDL", "GHB"}


def test_config_rejects_url_filename_mismatch(tmp_path):
    config = {
        "schema_version": 1,
        "cycles": {
            "1999-2000": {
                "files": [
                    {
                        "component": "DEMO",
                        "tier": "fast",
                        "format": "xpt",
                        "filename": "WRONG.xpt",
                        "url": "https://example.test/DEMO.xpt",
                    }
                ]
            }
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ConfigError, match="expected filename"):
        load_file_specs(path)


def test_download_retries_writes_sha256_manifest_and_is_idempotent(tmp_path, monkeypatch):
    payload = b"HEADER RECORD" + b" " * 100
    response = FakeResponse(payload, headers={"Content-Length": str(len(payload))})
    session = FakeSession(gets=[requests.ConnectionError("temporary"), response])
    monkeypatch.setattr("cvd_risk.data.download_nhanes.time.sleep", lambda _: None)
    manifest = ManifestStore(tmp_path / "manifest.yaml")
    destination = tmp_path / "1999-2000" / "DEMO.xpt"

    result = download_file(
        _demo_spec(),
        destination,
        manifest=manifest,
        session=session,
        attempts=2,
    )
    cached = download_file(_demo_spec(), destination, manifest=manifest, session=session)

    expected_hash = hashlib.sha256(payload).hexdigest()
    assert result.status == "downloaded"
    assert result.sha256 == expected_hash
    assert cached.status == "cached"
    assert session.get_calls == 2
    saved = yaml.safe_load((tmp_path / "manifest.yaml").read_text(encoding="utf-8"))
    assert saved["schema_version"] == 1
    assert saved["files"][0]["sha256"] == expected_hash
    assert saved["files"][0]["status"] == "downloaded"


def test_download_atomically_replaces_corrupt_existing_file(tmp_path):
    destination = tmp_path / "1999-2000" / "DEMO.xpt"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"partial HTML response")
    payload = b"HEADER RECORD" + b" " * 100
    session = FakeSession(
        gets=[FakeResponse(payload, headers={"Content-Length": str(len(payload))})]
    )

    result = download_file(_demo_spec(), destination, session=session)

    assert result.status == "downloaded"
    assert destination.read_bytes() == payload
    assert session.get_calls == 1


def test_download_accepts_lmf_records_with_trimmed_trailing_blanks(tmp_path):
    payload = b"00001         2.   ..                     .  .\r\n"
    session = FakeSession(
        gets=[FakeResponse(payload, headers={"Content-Length": str(len(payload))})]
    )
    destination = tmp_path / "mortality.dat"

    result = download_file(_lmf_spec(), destination, session=session)

    assert result.status == "downloaded"
    assert destination.read_bytes() == payload


def test_preflight_uses_head_without_downloading_body():
    response = FakeResponse(
        headers={"Content-Length": "123", "Content-Type": "application/octet-stream"}
    )
    session = FakeSession(heads=[response])

    result = preflight_file(_demo_spec(), session=session)

    assert result.status_code == 200
    assert result.content_length == 123
    assert session.head_calls == 1
    assert session.get_calls == 0
