"""The interleaved timing protocol, driven by a fake clock so every duration is
exact, and the per-engine absolute-time statistics."""

from __future__ import annotations

import numpy as np
import pytest

from bench.timing import absolute_time_stats, round_robin_interleaved


class FakeClock:
    """A clock that only moves when an engine 'runs'."""

    def __init__(self):
        self.now = 0.0
        self.log: list[str] = []

    def __call__(self):
        return self.now

    def engine(self, name, seconds):
        def run():
            self.log.append(name)
            self.now += seconds
            return name
        return run


def test_every_engine_is_measured_the_same_number_of_rounds():
    clk = FakeClock()
    reps = round_robin_interleaved(
        {"A": clk.engine("A", 1.0), "B": clk.engine("B", 3.0), "C": clk.engine("C", 5.0)},
        warmup_rounds=1, min_rounds=5, max_rounds=5, clock=clk)
    assert reps["rounds"] == 5
    assert reps["order"] == "A B C"
    for name, secs in (("A", 1.0), ("B", 3.0), ("C", 5.0)):
        assert reps["engines"][name]["seconds"] == [secs] * 5


def test_two_engine_median_ratio_is_exact():
    clk = FakeClock()
    reps = round_robin_interleaved(
        {"A": clk.engine("A", 1.0), "B": clk.engine("B", 3.0)},
        warmup_rounds=1, min_rounds=5, max_rounds=5, clock=clk)
    stats = absolute_time_stats({k: v["seconds"] for k, v in reps["engines"].items()})
    # No spread at all: each engine's median is its one duration, CI collapses to it.
    assert stats["A"]["median_seconds"] == pytest.approx(1.0, rel=1e-12)
    assert stats["B"]["median_seconds"] == pytest.approx(3.0, rel=1e-12)
    for name in ("A", "B"):
        assert stats[name]["ci_low"] == pytest.approx(stats[name]["median_seconds"], rel=1e-12)
        assert stats[name]["ci_high"] == pytest.approx(stats[name]["median_seconds"], rel=1e-12)


def test_engines_rotate_the_start_each_round_after_warmup():
    clk = FakeClock()
    round_robin_interleaved(
        {"A": clk.engine("A", 1.0), "B": clk.engine("B", 1.0), "C": clk.engine("C", 1.0)},
        warmup_rounds=1, min_rounds=3, max_rounds=3, clock=clk)
    # warm-up round starts at A (ABC), then measured rounds rotate the start:
    # round 0 ABC, round 1 BCA, round 2 CAB -- no engine is always first.
    assert "".join(clk.log) == "ABC" + "ABC" + "BCA" + "CAB"


def test_captures_one_output_per_kept_round_not_warmups():
    clk = FakeClock()
    a_seen: list[int] = []

    def a():
        a_seen.append(len(a_seen))
        clk.now += 1.0
        return ("a", len(a_seen))

    reps = round_robin_interleaved(
        {"A": a, "B": clk.engine("B", 1.0)},
        warmup_rounds=1, min_rounds=3, max_rounds=3, clock=clk,
        captures={"A": lambda v: v[1]})
    # One capture per kept round, aligned with the timings; the warm-up round (the
    # first call of each engine) is not captured.
    assert reps["engines"]["A"]["outputs"] == [2, 3, 4]
    assert len(reps["engines"]["A"]["outputs"]) == reps["rounds"]
    # An engine with no capture carries no outputs key.
    assert "outputs" not in reps["engines"]["B"]


def test_warmup_rounds_are_not_measured():
    clk = FakeClock()
    durations = iter([100.0, 1.0, 1.0, 1.0])  # the first A call is a slow cold start

    def a():
        clk.now += next(durations)

    reps = round_robin_interleaved(
        {"A": a, "B": clk.engine("B", 2.0)},
        warmup_rounds=1, min_rounds=3, max_rounds=3, clock=clk)
    assert reps["engines"]["A"]["seconds"] == [1.0, 1.0, 1.0]


def test_budget_stops_after_min_rounds_but_never_before():
    clk = FakeClock()
    # Each round costs 10 s; the 15 s budget is spent after 2 rounds, but 3 required.
    reps = round_robin_interleaved(
        {"A": clk.engine("A", 5.0), "B": clk.engine("B", 5.0)},
        warmup_rounds=0, min_rounds=3, max_rounds=20, budget_seconds=15.0, clock=clk)
    assert reps["rounds"] == 3


def test_max_rounds_caps_a_cheap_job():
    clk = FakeClock()
    reps = round_robin_interleaved(
        {"A": clk.engine("A", 1e-4), "B": clk.engine("B", 1e-4)},
        warmup_rounds=0, min_rounds=2, max_rounds=7, budget_seconds=1e9, clock=clk)
    assert reps["rounds"] == 7


def _noisy(seed=1, n=15, ratios=(1.0, 2.0)):
    rng = np.random.default_rng(seed)
    return {chr(ord("A") + i): (r * 0.010 * rng.lognormal(0.0, 0.1, n)).tolist()
            for i, r in enumerate(ratios)}


def test_interval_brackets_each_engine_median():
    stats = absolute_time_stats(_noisy())
    for name, s in stats.items():
        assert s["ci_low"] <= s["median_seconds"] <= s["ci_high"]
        assert s["ci_low"] < s["ci_high"]


def test_same_repetitions_give_the_same_interval():
    secs = _noisy()
    assert absolute_time_stats(secs) == absolute_time_stats(secs)


def test_scaling_time_scales_the_medians_and_bounds():
    secs = _noisy()
    s1 = absolute_time_stats(secs)
    s2 = absolute_time_stats({k: [x * 1000 for x in v] for k, v in secs.items()})
    for name in secs:
        for key in ("median_seconds", "ci_low", "ci_high"):
            assert s2[name][key] == pytest.approx(s1[name][key] * 1000, rel=1e-12)


@pytest.mark.parametrize("secs", [
    {},                                   # no engines
    {"A": [], "B": []},                   # no rounds
    {"A": [1.0, 2.0], "B": [1.0]},        # unequal round counts
    {"A": [0.0, 1.0], "B": [1.0, 1.0]},   # a non-positive time
])
def test_bad_repetitions_are_refused(secs):
    with pytest.raises(ValueError):
        absolute_time_stats(secs)


def test_no_engines_is_refused():
    with pytest.raises(ValueError):
        round_robin_interleaved({}, min_rounds=1, max_rounds=1)


def test_min_rounds_above_max_rounds_is_refused():
    with pytest.raises(ValueError):
        round_robin_interleaved({"A": lambda: None}, min_rounds=5, max_rounds=4)
