"""Locating a source-built libRoadRunner via ``$BENCH_ROADRUNNER_PATH``.

On linux-aarch64 there is no libRoadRunner wheel or sdist, so the suite is
pointed at a from-source build with this variable (docs/BUILD_NOTES.md). The
logic mirrors the ``BENCH_GLIBC_ROOT`` tree lookup in ``test_fetch.py``: an env
var names an external artifact, and an absent or malformed one reads as "use the
pinned wheel" without error.
"""

from __future__ import annotations

import sys

import pytest

from bench import env


def _mk_build(root, *, dist_info=True):
    """A minimal source-build layout: the package dir, optionally the dist-info."""
    (root / "roadrunner").mkdir(parents=True)
    (root / "roadrunner" / "__init__.py").write_text("", encoding="utf-8")
    if dist_info:
        d = root / "libroadrunner-2.11.0.dist-info"
        d.mkdir()
        (d / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: libroadrunner\nVersion: 2.11.0\n", encoding="utf-8"
        )
    return root


def test_source_path_is_none_without_the_var(tmp_path, monkeypatch):
    monkeypatch.delenv("BENCH_ROADRUNNER_PATH", raising=False)
    assert env.roadrunner_source_path() is None


def test_source_path_is_none_when_the_var_names_no_package(tmp_path, monkeypatch):
    # A directory that exists but holds no ``roadrunner`` package is not a build;
    # treated as unset so a typo falls back to the pinned wheel, not a hard error.
    monkeypatch.setenv("BENCH_ROADRUNNER_PATH", str(tmp_path))
    assert env.roadrunner_source_path() is None


def test_source_path_resolves_a_real_build(tmp_path, monkeypatch):
    root = _mk_build(tmp_path / "site-packages")
    monkeypatch.setenv("BENCH_ROADRUNNER_PATH", str(root))
    assert env.roadrunner_source_path() == root


def test_ensure_on_path_prepends_once(tmp_path, monkeypatch):
    root = _mk_build(tmp_path / "site-packages")
    monkeypatch.setenv("BENCH_ROADRUNNER_PATH", str(root))
    monkeypatch.setattr(sys, "path", list(sys.path))  # isolate the edit
    env._ensure_roadrunner_on_path()
    env._ensure_roadrunner_on_path()  # idempotent
    assert sys.path.count(str(root)) == 1
    assert sys.path[0] == str(root)  # prepended, so the build wins


def test_ensure_on_path_is_a_noop_without_the_var(monkeypatch):
    monkeypatch.delenv("BENCH_ROADRUNNER_PATH", raising=False)
    before = list(sys.path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    env._ensure_roadrunner_on_path()
    assert sys.path == before


def test_identity_records_the_source_when_set(tmp_path, monkeypatch):
    # The version still comes from importlib.metadata (the real libroadrunner the
    # test env has, or absent -> error); the point here is that ``source`` is
    # recorded so a source-built result is never mistaken for a wheel one.
    root = _mk_build(tmp_path / "site-packages")
    monkeypatch.setenv("BENCH_ROADRUNNER_PATH", str(root))
    monkeypatch.setattr(sys, "path", list(sys.path))
    env.roadrunner_identity.cache_clear()
    try:
        ident = env.roadrunner_identity()
    finally:
        env.roadrunner_identity.cache_clear()
    assert ident["source"] == str(root)


def test_identity_omits_source_when_unset(monkeypatch):
    monkeypatch.delenv("BENCH_ROADRUNNER_PATH", raising=False)
    env.roadrunner_identity.cache_clear()
    try:
        ident = env.roadrunner_identity()
    finally:
        env.roadrunner_identity.cache_clear()
    assert "source" not in ident


def test_copasi_identity_reports_both_versions(monkeypatch):
    # COPASI's version string is the python-copasi package (the native engine);
    # basico is the pure-Python wrapper the adapter calls. Both from metadata.
    import importlib.metadata as md

    versions = {"python-copasi": "4.48.311", "copasi-basico": "0.92"}
    monkeypatch.setattr(md, "version", lambda name: versions[name])
    env.copasi_identity.cache_clear()
    try:
        ident = env.copasi_identity()
    finally:
        env.copasi_identity.cache_clear()
    assert ident == {"version": "4.48.311", "basico": "0.92"}


def test_copasi_identity_reports_absence_as_an_error(monkeypatch):
    # A missing package is a broken environment (it is pinned); identity records
    # version=None plus the error, which the doctor turns into a FAIL.
    import importlib.metadata as md

    def _absent(name):
        raise md.PackageNotFoundError(name)

    monkeypatch.setattr(md, "version", _absent)
    env.copasi_identity.cache_clear()
    try:
        ident = env.copasi_identity()
    finally:
        env.copasi_identity.cache_clear()
    assert ident["version"] is None
    assert "PackageNotFoundError" in ident["error"]


def test_amici_identity_reports_the_version(monkeypatch):
    import importlib.metadata as md

    monkeypatch.setattr(md, "version", lambda name: "1.1.0" if name == "amici" else None)
    env.amici_identity.cache_clear()
    try:
        ident = env.amici_identity()
    finally:
        env.amici_identity.cache_clear()
    assert ident == {"version": "1.1.0"}


def test_amici_identity_reports_absence_as_an_error(monkeypatch):
    import importlib.metadata as md

    def _absent(name):
        raise md.PackageNotFoundError(name)

    monkeypatch.setattr(md, "version", _absent)
    env.amici_identity.cache_clear()
    try:
        ident = env.amici_identity()
    finally:
        env.amici_identity.cache_clear()
    assert ident["version"] is None
    assert "PackageNotFoundError" in ident["error"]


# --------------------------------------------------------------------------- #
# The network-free references: NFsim through BNG2.pl, and standalone RuleMonkey
# --------------------------------------------------------------------------- #
def _bng_dist(root):
    """A BioNetGen layout: BNG2.pl, Perl2/, a marker, and the two launched binaries."""
    (root / "bin").mkdir(parents=True)
    (root / "Perl2").mkdir()
    (root / "BNG2.pl").write_text("#!/usr/bin/perl\n")
    (root / ".bench-fetch.json").write_text('{"version": "2.9.3"}')
    for name in ("NFsim", "run_network", "sundials-config"):
        (root / "bin" / name).write_text("binary")
    return root


def test_without_a_loader_swap_bng2pl_runs_the_distribution_itself(tmp_path, monkeypatch):
    root = _bng_dist(tmp_path / "BioNetGen-2.9.3")
    monkeypatch.setenv("BNGPATH", str(root))
    monkeypatch.setattr(env, "run_network_launch", lambda: [])
    assert env.bng_launch_tree(tmp_path / "tree") == root
    assert not (tmp_path / "tree").exists()


def test_the_launch_tree_wraps_nfsim_and_run_network_in_the_loader(tmp_path, monkeypatch):
    root = _bng_dist(tmp_path / "BioNetGen-2.9.3")
    monkeypatch.setenv("BNGPATH", str(root))
    monkeypatch.setattr(env, "run_network_launch",
                        lambda: ["/g/ld.so", "--library-path", "/g/lib64:/g/usr lib64"])
    tree = env.bng_launch_tree(tmp_path / "tree")
    assert tree == tmp_path / "tree"
    # Everything else is the distribution itself, through a symlink.
    for name in ("BNG2.pl", "Perl2", ".bench-fetch.json"):
        assert (tree / name).is_symlink() and (tree / name).resolve() == (root / name).resolve()
    assert (tree / "bin" / "sundials-config").resolve() == (root / "bin" / "sundials-config")
    for name in ("NFsim", "run_network"):
        wrapper = tree / "bin" / name
        assert not wrapper.is_symlink() and wrapper.stat().st_mode & 0o111
        text = wrapper.read_text()
        assert text.startswith("#!/bin/sh\n")
        assert f"exec /g/ld.so --library-path '/g/lib64:/g/usr lib64' {root / 'bin' / name} \"$@\"" \
            in text
    # The binaries themselves are untouched.
    assert (root / "bin" / "NFsim").read_text() == "binary"
    # Rebuilt on every call, so it follows the loader.
    monkeypatch.setattr(env, "run_network_launch", lambda: ["/h/ld.so", "--library-path", "/h"])
    env.bng_launch_tree(tree)
    assert "/h/ld.so" in (tree / "bin" / "NFsim").read_text()


@pytest.mark.parametrize("bng, nfsim", [("2.9.4", "2.0.1"), ("2.9.3", "1.14.3")])
def test_nfsim_identity_is_the_binary_bioNetGen_bundles(tmp_path, monkeypatch, bng, nfsim):
    # 2.9.4 is the pinned release (NFsim 2); 2.9.3 (NFsim 1.14.3) is for comparison runs.
    root = _bng_dist(tmp_path / f"BioNetGen-{bng}")
    monkeypatch.setenv("BNGPATH", str(root))
    monkeypatch.setattr(env, "bionetgen_identity", lambda: {"version": bng})
    env.nfsim_identity.cache_clear()
    try:
        ident = env.nfsim_identity()
        assert ident["version"] == nfsim and ident["bundled_with"] == f"BioNetGen {bng}"
        assert ident["path"] == str(root / "bin" / "NFsim") and len(ident["sha256"]) == 64
    finally:
        env.nfsim_identity.cache_clear()


def _rm_install(root, marker=True):
    (root / "bin").mkdir(parents=True)
    exe = root / "bin" / "rm_driver"
    exe.write_bytes(b"driver")
    if marker:
        from bench import fetch

        (root / ".bench-build.json").write_text(
            '{"version": "3.13.0", "commit": "ad3bd75", "compiler": {"version": "gcc 15"}, '
            f'"binary_sha256": "{fetch.sha256_file(exe)}"}}')
    return root


def test_rulemonkey_is_found_from_bngpath_in_the_shared_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("BENCH_RULEMONKEY_PATH", raising=False)
    monkeypatch.setenv("BNGPATH", str(tmp_path / "bionetgen" / "BioNetGen-2.9.3"))
    assert env.rulemonkey_root() is None
    root = _rm_install(tmp_path / "rulemonkey" / "RuleMonkey-3.13.0")
    assert env.rulemonkey_root() == root
    assert env.rulemonkey_bin() == root / "bin" / "rm_driver"


def test_bench_rulemonkey_path_names_another_install(tmp_path, monkeypatch):
    root = _rm_install(tmp_path / "elsewhere", marker=False)
    monkeypatch.setenv("BENCH_RULEMONKEY_PATH", str(root))
    env.rulemonkey_identity.cache_clear()
    try:
        ident = env.rulemonkey_identity()
        # Usable, but without a matching marker it is not the pinned build.
        assert ident["path"] == str(root / "bin" / "rm_driver")
        assert ident["version"] is None and ident["built"] is False
    finally:
        env.rulemonkey_identity.cache_clear()


def test_rulemonkey_identity_reads_the_build_marker(tmp_path, monkeypatch):
    root = _rm_install(tmp_path / "rm")
    monkeypatch.setenv("BENCH_RULEMONKEY_PATH", str(root))
    env.rulemonkey_identity.cache_clear()
    try:
        ident = env.rulemonkey_identity()
        assert (ident["version"], ident["commit"], ident["compiler"], ident["built"]) == \
            ("3.13.0", "ad3bd75", "gcc 15", True)
    finally:
        env.rulemonkey_identity.cache_clear()


def test_a_missing_rulemonkey_is_reported_not_raised(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCH_RULEMONKEY_PATH", str(tmp_path / "none"))
    env.rulemonkey_identity.cache_clear()
    try:
        assert env.rulemonkey_identity()["version"] is None
        assert "fetch-rulemonkey" in env.rulemonkey_identity()["error"]
        with pytest.raises(RuntimeError, match="fetch-rulemonkey"):
            env.rulemonkey_bin()
    finally:
        env.rulemonkey_identity.cache_clear()
