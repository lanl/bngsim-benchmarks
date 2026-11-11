"""Deadline parsing for `bench run --budget/--until`."""

from __future__ import annotations

import argparse
import datetime as dt

import pytest

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
