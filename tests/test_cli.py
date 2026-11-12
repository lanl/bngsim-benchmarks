"""Deadline parsing for `bench run --budget/--until`."""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import types

import pytest

from bench import cli
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


@pytest.mark.parametrize("enc", ["utf-8", "UTF-8", "utf8", "UTF8"])
def test_utf8_encodings_are_accepted(enc):
    assert cli._encoding_is_utf8(enc) is True


@pytest.mark.parametrize("enc", ["US-ASCII", "ascii", "latin-1", "ISO-8859-1", "cp1252", ""])
def test_non_utf8_encodings_are_rejected(enc):
    assert cli._encoding_is_utf8(enc) is False
