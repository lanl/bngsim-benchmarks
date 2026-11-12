"""Downloads are pinned by sha256: wrong bytes are discarded, right bytes cached."""

from __future__ import annotations

import hashlib
import json

import pytest

from bench import env, fetch

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


# --------------------------------------------------------------------------- #
# The glibc loader swap (fetch + how env builds the launch prefix from it)
# --------------------------------------------------------------------------- #
def _fake_glibc(tmp_path, monkeypatch):
    """Install a stand-in glibc tree without touching the network or rpm2cpio.

    ``_download`` writes a dummy RPM whose bytes hash to the pinned sha256;
    ``_extract_rpm`` lays out the loader (``lib64``) and libstdc++
    (``usr/lib64``) the way the real RPMs do.
    """
    bodies = {spec["url"]: f"rpm:{spec['sha256']}".encode() for spec in fetch.GLIBC_RPMS}
    # Pin each RPM's sha256 to the bytes our fake download writes.
    for spec in fetch.GLIBC_RPMS:
        spec_sha = hashlib.sha256(bodies[spec["url"]]).hexdigest()
        monkeypatch.setitem(spec, "sha256", spec_sha)

    def fake_download(url, dest, timeout=120):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(bodies[url])

    def fake_extract(rpm, dest):
        if "glibc" in rpm.name:
            lib = dest / "lib64"
            lib.mkdir(parents=True, exist_ok=True)
            (lib / "ld-linux-x86-64.so.2").write_bytes(b"\x7fELF ld")
            (lib / "libc.so.6").write_bytes(b"GLIBC_2.34\x00GLIBC_2.17\x00")
            (lib / "libm.so.6").write_bytes(b"GLIBC_2.34\x00")
        else:
            lib = dest / "usr" / "lib64"
            lib.mkdir(parents=True, exist_ok=True)
            (lib / "libstdc++.so.6.0.29").write_bytes(b"GLIBCXX_3.4.29\x00CXXABI_1.3.9\x00")
            (lib / "libstdc++.so.6").symlink_to("libstdc++.so.6.0.29")

    monkeypatch.setattr(fetch, "_download", fake_download)
    monkeypatch.setattr(fetch, "_extract_rpm", fake_extract)


def test_fetch_glibc_installs_and_records_the_loader(tmp_path, monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    _fake_glibc(tmp_path, monkeypatch)
    root = fetch.fetch_glibc(tmp_path)
    assert root == tmp_path / "glibc" / fetch.GLIBC_TAG
    m = json.loads((root / fetch.GLIBC_MARKER).read_text())
    assert m["tag"] == fetch.GLIBC_TAG
    assert m["ld_so"] == "tree/lib64/ld-linux-x86-64.so.2"
    assert m["glibc_version"] == "2.34"
    assert m["libstdcxx_version"] == "3.4.29"
    # library_path covers both the loader's dir and libstdc++'s dir.
    assert set(m["library_path"]) == {"tree/lib64", "tree/usr/lib64"}


def test_fetch_glibc_reuses_a_matching_tree(tmp_path, monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    _fake_glibc(tmp_path, monkeypatch)
    fetch.fetch_glibc(tmp_path)

    def no_download(*a, **k):
        raise AssertionError("re-downloaded although the marker matched")

    monkeypatch.setattr(fetch, "_download", no_download)
    assert fetch.fetch_glibc(tmp_path) == tmp_path / "glibc" / fetch.GLIBC_TAG


def test_fetch_glibc_is_linux_only(tmp_path, monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    with pytest.raises(SystemExit, match="only for Linux"):
        fetch.fetch_glibc(tmp_path)


def test_run_network_launch_is_empty_without_a_tree(tmp_path, monkeypatch):
    env.glibc_identity.cache_clear()
    env.run_network_launch.cache_clear()
    monkeypatch.delenv("BENCH_GLIBC_ROOT", raising=False)
    monkeypatch.setattr(env, "bngpath", lambda: tmp_path / "bionetgen" / "BioNetGen-x")
    assert env.run_network_launch() == []
    assert env.glibc_identity() == {"swapped": False}


def test_run_network_launch_prefixes_the_loader_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    _fake_glibc(tmp_path, monkeypatch)
    root = fetch.fetch_glibc(tmp_path)
    env.glibc_identity.cache_clear()
    env.run_network_launch.cache_clear()
    monkeypatch.setenv("BENCH_GLIBC_ROOT", str(root))
    launch = env.run_network_launch()
    assert launch[0] == str(root / "tree" / "lib64" / "ld-linux-x86-64.so.2")
    assert launch[1] == "--library-path"
    assert str(root / "tree" / "usr" / "lib64") in launch[2]
    ident = env.glibc_identity()
    assert ident["swapped"] and ident["glibc_version"] == "2.34"
    env.glibc_identity.cache_clear()
    env.run_network_launch.cache_clear()
