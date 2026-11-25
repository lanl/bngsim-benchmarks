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


def test_model_url_percent_encodes_the_path_but_keeps_its_slashes():
    source = {**SOURCE, "path": "final/BIOMD0000000589/Valero et al 2015 #10a?.xml"}
    assert fetch.model_url(source) == (
        f"https://raw.githubusercontent.com/o/r/{'a' * 40}/"
        "final/BIOMD0000000589/Valero%20et%20al%202015%20%2310a%3F.xml")
    # The cached copy keeps the file's own name.
    assert fetch.model_cache_path(source, "c").name == "Valero et al 2015 #10a?.xml"


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


def test_incomplete_marker_is_native_for_both_identity_and_launch(tmp_path, monkeypatch):
    # A marker that is valid JSON but missing the keys a prefix needs (or whose
    # ld.so is gone) must read as native *consistently*: no prefix, and an
    # identity that does not claim "swapped" -- otherwise a result could be
    # labelled swapped while run_network actually ran under the host loader.
    root = tmp_path / "glibc" / fetch.GLIBC_TAG
    root.mkdir(parents=True)
    (root / fetch.GLIBC_MARKER).write_text(json.dumps({"tag": fetch.GLIBC_TAG}))  # no ld_so
    env.glibc_identity.cache_clear()
    env.run_network_launch.cache_clear()
    monkeypatch.setenv("BENCH_GLIBC_ROOT", str(root))
    assert env.run_network_launch() == []
    assert env.glibc_identity() == {"swapped": False}
    env.glibc_identity.cache_clear()
    env.run_network_launch.cache_clear()


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


def test_bionetgen_is_pinned_to_the_release_that_bundles_nfsim_2():
    from bench import env

    assert fetch.BIONETGEN_VERSION == "2.9.5"
    assert env.NFSIM_VERSION[fetch.BIONETGEN_VERSION] == "2.0.1"
    assert set(fetch.BIONETGEN_SHA256) == {"mac", "linux"}
    assert all(len(h) == 64 and int(h, 16) >= 0 for h in fetch.BIONETGEN_SHA256.values())


# --------------------------------------------------------------------------- #
# RuleMonkey: the pinned source, built with the env's compiler
# --------------------------------------------------------------------------- #
def _fake_rm_install(tmp_path, **marker):
    dest = fetch.rulemonkey_dir(tmp_path)
    (dest / "bin").mkdir(parents=True)
    exe = dest / "bin" / "rm_driver"
    exe.write_bytes(b"\x7fELF rm_driver")
    m = {"source_sha256": fetch.RULEMONKEY_SHA256, "recipe": fetch.RULEMONKEY_RECIPE,
         "binary_sha256": fetch.sha256_file(exe), "version": fetch.RULEMONKEY_VERSION,
         "commit": fetch.RULEMONKEY_COMMIT, **marker}
    (dest / fetch.RULEMONKEY_MARKER).write_text(json.dumps(m))
    return dest


def test_rulemonkey_is_pinned_to_a_release_tag_commit_and_tarball():
    assert fetch.RULEMONKEY_TAG == "v3.13.0"
    assert len(fetch.RULEMONKEY_COMMIT) == 40 and len(fetch.RULEMONKEY_SHA256) == 64
    assert fetch.RULEMONKEY_URL.endswith("/archive/refs/tags/v3.13.0.tar.gz")


def test_a_current_rulemonkey_build_is_reused_without_building(tmp_path, monkeypatch):
    dest = _fake_rm_install(tmp_path)
    monkeypatch.setattr(fetch, "rulemonkey_tarball", lambda cache: tmp_path / "t.tar.gz")

    def no_build(*a, **k):
        raise AssertionError("rebuilt although the build was current")

    monkeypatch.setattr(fetch, "_conda_cxx", no_build)
    assert fetch.fetch_rulemonkey(tmp_path) == dest


@pytest.mark.parametrize("marker", [{"recipe": 0}, {"source_sha256": "0" * 64},
                                    {"binary_sha256": "0" * 64}])
def test_a_build_from_another_recipe_source_or_binary_is_not_current(tmp_path, marker):
    dest = _fake_rm_install(tmp_path, **marker)
    assert not fetch.rulemonkey_build_current(dest)


def test_the_rulemonkey_tarball_with_wrong_bytes_is_refused(tmp_path, monkeypatch):
    def bad_download(url, dest, timeout=120):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"not the release")

    monkeypatch.setattr(fetch, "_download", bad_download)
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        fetch.rulemonkey_tarball(tmp_path)
    assert not list((tmp_path / "rulemonkey" / "downloads").glob("*.tar.gz"))


def test_a_compiler_outside_the_environment_is_refused(tmp_path, monkeypatch):
    system_cxx = tmp_path / "usr" / "bin" / "g++"
    system_cxx.parent.mkdir(parents=True)
    system_cxx.write_text("#!/bin/sh\n")
    system_cxx.chmod(0o755)
    monkeypatch.setenv("CXX", str(system_cxx))
    with pytest.raises(SystemExit, match="not this environment's conda-forge compiler"):
        fetch._conda_cxx()


def test_the_environments_own_compiler_is_accepted(tmp_path, monkeypatch):
    import sys

    prefix = tmp_path / "env"
    cxx = prefix / "bin" / "x86_64-conda-linux-gnu-g++"
    cxx.parent.mkdir(parents=True)
    cxx.write_text("#!/bin/sh\n")
    cxx.chmod(0o755)
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.delenv("CXX", raising=False)
    assert fetch._conda_cxx() == cxx.resolve()
