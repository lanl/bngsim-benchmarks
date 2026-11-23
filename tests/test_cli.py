"""Deadline parsing for `bench run --budget/--until`."""

from __future__ import annotations

import argparse
import datetime as dt
import signal
import subprocess
import types

import pytest

from bench import cli
from bench import env as bench_env
from bench.cli import parse_duration, parse_until


@pytest.mark.parametrize(
    "text, seconds",
    [("9h30m", 9 * 3600 + 30 * 60), ("45m", 2700), ("90s", 90), ("1h", 3600),
     ("2h0m5s", 7205), ("120", 120), ("0.5", 0.5)],
)
def test_durations(text, seconds):
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["", "abc", "5d", "h", "1.5h"])
def test_bad_durations_are_refused(text):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_duration(text)


def test_clock_time_later_today():
    now = dt.datetime(2026, 11, 11, 9, 0).timestamp()
    assert parse_until("17:30", now) == dt.datetime(2026, 11, 11, 17, 30).timestamp()


def test_clock_time_already_past_means_tomorrow():
    now = dt.datetime(2026, 11, 11, 18, 0).timestamp()
    assert parse_until("06:15", now) == dt.datetime(2026, 11, 12, 6, 15).timestamp()


def test_iso_time():
    assert parse_until("2026-11-12T08:00:00") == dt.datetime(2026, 11, 12, 8, 0).timestamp()


def test_slurm_deadline_outside_an_allocation_is_refused(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(SystemExit, match="not inside a Slurm allocation"):
        parse_until("slurm")


def _doctor_env():
    """A stand-in env for the doctor load check (native loader, fixed binary)."""
    return types.SimpleNamespace(
        run_network_launch=lambda: [],
        run_network_bin=lambda: "/bng/bin/run_network",
        glibc_identity=lambda: {"swapped": False},
    )


def _fake_run(output, returncode=1):
    def run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr=output)
    return run


def test_doctor_load_check_passes_when_run_network_only_prints_its_version(monkeypatch):
    # run_network with no args prints a usage/version banner and exits 1; that is
    # a successful *load*.
    monkeypatch.setattr(cli.subprocess, "run",
                        _fake_run("run_network version 2.9.3\nUsage: run_network ..."))
    assert cli._check_run_network_loads(_doctor_env()) is True


def test_doctor_load_check_fails_on_a_non_glibc_shared_object_error(monkeypatch):
    # The regression: a libstdc++ failure carries no "GLIBC", so the old heuristic
    # (GLIBC-absent => OK) wrongly reported READY. It must FAIL.
    err = ("run_network: error while loading shared libraries: libstdc++.so.6: "
           "cannot open shared object file: No such file or directory")
    monkeypatch.setattr(cli.subprocess, "run", _fake_run(err))
    assert cli._check_run_network_loads(_doctor_env()) is False


def test_doctor_load_check_fails_on_the_glibc_version_error(monkeypatch):
    err = "run_network: /lib64/libc.so.6: version `GLIBC_2.34' not found (required by run_network)"
    monkeypatch.setattr(cli.subprocess, "run", _fake_run(err))
    assert cli._check_run_network_loads(_doctor_env()) is False


def test_core_selftest_verdict_passes_on_clean_exit():
    passed, lines = cli._selftest_verdict(0, "")
    assert passed is True
    assert any("from_net" in s for s in lines)


def test_core_selftest_verdict_names_the_abi_mismatch_on_a_segfault():
    # A native crash leaves no traceback; the child exits -SIGSEGV. The verdict
    # must FAIL and point at the libstdc++ ABI mismatch and BUILD_NOTES, since that
    # is overwhelmingly the cause and a bare "exit -11" tells a cluster user nothing.
    passed, lines = cli._selftest_verdict(-signal.SIGSEGV, "")
    assert passed is False
    blob = "\n".join(lines)
    assert "SIGSEGV" in blob
    assert "ABI" in blob and "libstdc++" in blob
    assert "BUILD_NOTES.md" in blob


def test_core_selftest_verdict_reports_a_signal_with_no_name():
    passed, lines = cli._selftest_verdict(-99, "")
    assert passed is False
    assert "signal 99" in "\n".join(lines)


def test_core_selftest_verdict_surfaces_a_python_error_verbatim():
    # A POSITIVE exit is an ordinary exception (the core rejected the net), not an
    # ABI crash -- show the message, do not mis-diagnose it as an ABI mismatch.
    passed, lines = cli._selftest_verdict(1, "Traceback ...\nModelError: bad .net")
    assert passed is False
    blob = "\n".join(lines)
    assert "ModelError: bad .net" in blob
    assert "ABI" not in blob


def test_core_selftest_check_fails_on_a_segfaulting_child(monkeypatch):
    def run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, -signal.SIGSEGV, stdout="", stderr="")
    monkeypatch.setattr(cli.subprocess, "run", run)
    assert cli._check_core_loads_net() is False


def test_core_selftest_check_passes_when_the_child_loads_the_model(monkeypatch):
    def run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(cli.subprocess, "run", run)
    assert cli._check_core_loads_net() is True


@pytest.mark.parametrize("enc", ["utf-8", "UTF-8", "utf8", "UTF8"])
def test_utf8_encodings_are_accepted(enc):
    assert cli._encoding_is_utf8(enc) is True


@pytest.mark.parametrize("enc", ["US-ASCII", "ascii", "latin-1", "ISO-8859-1", "cp1252", ""])
def test_non_utf8_encodings_are_rejected(enc):
    assert cli._encoding_is_utf8(enc) is False


def test_roadrunner_check_fails_when_the_package_is_absent():
    # The environment is pinned to install libRoadRunner; a missing package is a
    # broken environment and must FAIL, not warn.
    env = types.SimpleNamespace(
        roadrunner_identity=lambda: {"version": None, "error": "PackageNotFoundError"})
    assert cli._check_roadrunner_loads(env) is False


def test_roadrunner_check_passes_when_it_imports():
    # Skip only when roadrunner is genuinely unusable; a bare importorskip would
    # miss the preload that makes the wheel's _roadrunner.so find libpython, so it
    # would fail on a host without a system libpython even though the engine works.
    try:
        bench_env.import_roadrunner()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"roadrunner not importable: {exc}")
    env = types.SimpleNamespace(
        roadrunner_identity=lambda: {"version": "2.11.0"},
        import_roadrunner=bench_env.import_roadrunner)
    assert cli._check_roadrunner_loads(env) is True


def test_copasi_check_fails_when_the_package_is_absent():
    # The environment is pinned to install COPASI; a missing package is a broken
    # environment and must FAIL, not warn -- same contract as roadrunner.
    env = types.SimpleNamespace(
        copasi_identity=lambda: {"version": None, "error": "PackageNotFoundError"})
    assert cli._check_copasi_loads(env) is False


def test_copasi_check_passes_when_it_imports():
    # basico carries the native python-copasi extension; skip if the wheel is not
    # installed on this host rather than fail the suite.
    pytest.importorskip("basico")
    env = types.SimpleNamespace(
        copasi_identity=lambda: {"version": "4.48.311", "basico": "0.92"})
    assert cli._check_copasi_loads(env) is True


def test_amici_check_fails_when_the_package_is_absent():
    # The environment is pinned to install AMICI; a missing package is a broken
    # environment and must FAIL, not warn -- same contract as the other engines.
    env = types.SimpleNamespace(
        amici_identity=lambda: {"version": None, "error": "PackageNotFoundError"})
    assert cli._check_amici_loads(env) is False


@pytest.mark.integration
def test_amici_check_passes_when_it_compiles_and_runs():
    # Unlike the other engines' import checks, AMICI's doctor self-test actually
    # compiles a C++ model and integrates it -- the ABI-risky path, and slow -- so
    # it is an integration test, gated behind the pinned package being installed.
    pytest.importorskip("amici")
    env = types.SimpleNamespace(
        amici_identity=lambda: {"version": bench_env.amici_identity().get("version")})
    assert cli._check_amici_loads(env) is True


def test_nfsim_check_reads_the_version_off_its_banner(monkeypatch):
    env = types.SimpleNamespace(
        nfsim_identity=lambda: {"version": "1.14.3", "path": "/bng/bin/NFsim",
                                "bundled_with": "BioNetGen 2.9.3"},
        run_network_launch=lambda: [])
    monkeypatch.setattr(cli.subprocess, "run", _fake_run(
        "starting NFsim v1.14.3...\n\n\tNo parameters given, so I won't do anything.", 0))
    assert cli._check_nfsim_loads(env) is True
    monkeypatch.setattr(cli.subprocess, "run", _fake_run("starting NFsim v1.15.0...", 0))
    assert cli._check_nfsim_loads(env) is False
    monkeypatch.setattr(cli.subprocess, "run", _fake_run(
        "NFsim: /lib64/libc.so.6: version `GLIBC_2.32' not found (required by NFsim)"))
    assert cli._check_nfsim_loads(env) is False


def test_rulemonkey_check_warns_when_unbuilt_and_fails_when_broken(monkeypatch):
    missing = types.SimpleNamespace(rulemonkey_identity=lambda: {
        "version": None, "error": "rm_driver not found (run `bench fetch-rulemonkey`)"})
    assert cli._check_rulemonkey_runs(missing) is True
    built = types.SimpleNamespace(rulemonkey_identity=lambda: {
        "version": "3.13.0", "path": "/rm/bin/rm_driver", "built": True, "compiler": "gcc 15"})
    monkeypatch.setattr(cli.subprocess, "run", _fake_run("Usage: rm_driver <xml_path> ..."))
    assert cli._check_rulemonkey_runs(built) is True
    monkeypatch.setattr(cli.subprocess, "run", _fake_run("rm_driver: GLIBCXX_3.4.32 not found"))
    assert cli._check_rulemonkey_runs(built) is False
