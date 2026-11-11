"""Downloads are pinned by sha256: wrong bytes are discarded, right bytes cached."""

from __future__ import annotations

import hashlib
import json

import pytest

from bench import fetch

BODY = b"begin model\nend model\n"
SHA = hashlib.sha256(BODY).hexdigest()
SOURCE = {"repo": "o/r", "commit": "a" * 40, "path": "models/m/m.bngl", "sha256": SHA}


def test_model_file_downloads_checks_and_caches(tmp_path, monkeypatch):
    calls = []

    def fake_download(url, dest, timeout=120):
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(BODY)

    monkeypatch.setattr(fetch, "_download", fake_download)
    p = fetch.model_file(SOURCE, tmp_path)
    assert p.read_bytes() == BODY
    assert calls == [f"https://raw.githubusercontent.com/o/r/{'a' * 40}/models/m/m.bngl"]
    assert fetch.model_file(SOURCE, tmp_path) == p
    assert len(calls) == 1  # second call served from the cache


def test_model_file_with_wrong_bytes_is_discarded(tmp_path, monkeypatch):
    def fake_download(url, dest, timeout=120):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(BODY + b"tampered")

    monkeypatch.setattr(fetch, "_download", fake_download)
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        fetch.model_file(SOURCE, tmp_path)
    assert not (tmp_path / "models" / SHA / "m.bngl").exists()


def test_corrupted_cache_entry_is_downloaded_again(tmp_path, monkeypatch):
    cached = tmp_path / "models" / SHA / "m.bngl"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"bit rot")
    monkeypatch.setattr(fetch, "_download", lambda url, dest, timeout=120: dest.write_bytes(BODY))
    assert fetch.model_file(SOURCE, tmp_path).read_bytes() == BODY


def test_bionetgen_with_matching_marker_is_not_downloaded_again(tmp_path, monkeypatch):
    key = fetch.bionetgen_platform_key()
    dest = tmp_path / "bionetgen" / f"BioNetGen-{fetch.BIONETGEN_VERSION}"
    dest.mkdir(parents=True)
    (dest / fetch.FETCH_MARKER).write_text(json.dumps({"sha256": fetch.BIONETGEN_SHA256[key]}))

    def no_download(*a, **k):
        raise AssertionError("downloaded although the marker matched")

    monkeypatch.setattr(fetch, "_download", no_download)
    assert fetch.fetch_bionetgen(tmp_path) == dest


def test_bionetgen_with_wrong_tarball_is_refused(tmp_path, monkeypatch):
    def bad_download(url, dest, timeout=120):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"not the release")

    monkeypatch.setattr(fetch, "_download", bad_download)
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        fetch.fetch_bionetgen(tmp_path)
    assert not list((tmp_path / "bionetgen").glob("*.tar.gz"))
