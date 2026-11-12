"""The paired timing protocol, driven by a fake clock so every duration is exact."""

from __future__ import annotations

import numpy as np
import pytest

from bench.timing import paired_interleaved, speedup


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
        return run


def test_ratio_of_known_durations_is_exact():
    clk = FakeClock()
    reps = paired_interleaved(clk.engine("A", 1.0), clk.engine("B", 3.0),
                              warmup_pairs=1, min_pairs=5, max_pairs=5, clock=clk)
    assert reps["a_seconds"] == [1.0] * 5
    assert reps["b_seconds"] == [3.0] * 5
    s = speedup(reps["a_seconds"], reps["b_seconds"])
    # B takes 3 s, A takes 1 s: A is exactly 3x faster, with no spread at all.
    assert s["speedup"] == pytest.approx(3.0, rel=1e-12)
    assert s["ci_low"] == pytest.approx(3.0, rel=1e-12)
    assert s["ci_high"] == pytest.approx(3.0, rel=1e-12)


def test_engines_alternate_abba_after_warmup():
    clk = FakeClock()
    paired_interleaved(clk.engine("A", 1.0), clk.engine("B", 1.0),
                       warmup_pairs=1, min_pairs=4, max_pairs=4, clock=clk)
    # warm-up pair AB, then measured pairs AB, BA, AB, BA
    assert "".join(clk.log) == "AB" + "AB" + "BA" + "AB" + "BA"


def test_captures_one_output_per_kept_pair_not_warmups():
    clk = FakeClock()
    a_seen: list[int] = []
    b_seen: list[int] = []

    def a():
        a_seen.append(len(a_seen))
        clk.now += 1.0
        return ("a", len(a_seen))

    def b():
        b_seen.append(len(b_seen))
        clk.now += 1.0
        return ("b", len(b_seen))

    reps = paired_interleaved(a, b, warmup_pairs=1, min_pairs=3, max_pairs=3, clock=clk,
                              capture_a=lambda v: v[1], capture_b=lambda v: v[1])
    # One capture per kept pair, aligned with the timings; the warm-up pair (the
    # first call of each engine) is not captured.
    assert reps["a_outputs"] == [2, 3, 4] and reps["b_outputs"] == [2, 3, 4]
    assert len(reps["a_outputs"]) == reps["pairs"] == len(reps["a_seconds"])


def test_no_captures_leaves_the_output_keys_out():
    clk = FakeClock()
    reps = paired_interleaved(clk.engine("A", 1.0), clk.engine("B", 1.0),
                              warmup_pairs=0, min_pairs=2, max_pairs=2, clock=clk)
    assert "a_outputs" not in reps and "b_outputs" not in reps


def test_warmup_repetitions_are_not_measured():
    clk = FakeClock()
    durations = iter([100.0, 1.0, 1.0, 1.0])  # the first A call is a slow cold start

    def a():
        clk.now += next(durations)

    reps = paired_interleaved(a, clk.engine("B", 2.0), warmup_pairs=1, min_pairs=3,
                              max_pairs=3, clock=clk)
    assert reps["a_seconds"] == [1.0, 1.0, 1.0]


def test_budget_stops_after_min_pairs_but_never_before():
    clk = FakeClock()
    # Each pair costs 10 s; the 15 s budget is spent after 2 pairs, but 3 are required.
    reps = paired_interleaved(clk.engine("A", 5.0), clk.engine("B", 5.0), warmup_pairs=0,
                              min_pairs=3, max_pairs=20, budget_seconds=15.0, clock=clk)
    assert reps["pairs"] == 3


def test_max_pairs_caps_a_cheap_job():
    clk = FakeClock()
    reps = paired_interleaved(clk.engine("A", 1e-4), clk.engine("B", 1e-4), warmup_pairs=0,
                              min_pairs=2, max_pairs=7, budget_seconds=1e9, clock=clk)
    assert reps["pairs"] == 7


def _noisy(seed=1, n=15, true_ratio=2.0):
    rng = np.random.default_rng(seed)
    a = 0.010 * rng.lognormal(0.0, 0.1, n)
    b = true_ratio * 0.010 * rng.lognormal(0.0, 0.1, n)
    return a.tolist(), b.tolist()


def test_interval_brackets_the_point_estimate():
    s = speedup(*_noisy())
    assert s["ci_low"] <= s["speedup"] <= s["ci_high"]
    assert s["ci_low"] < s["ci_high"]


def test_speedup_is_unchanged_by_the_unit_of_time():
    a, b = _noisy()
    s1 = speedup(a, b)
    s2 = speedup([x * 1000 for x in a], [x * 1000 for x in b])  # seconds -> milliseconds
    for k in ("speedup", "ci_low", "ci_high"):
        assert s2[k] == pytest.approx(s1[k], rel=1e-12)


def test_swapping_engines_inverts_the_speedup():
    a, b = _noisy()
    fwd, rev = speedup(a, b), speedup(b, a)
    assert rev["speedup"] == pytest.approx(1 / fwd["speedup"], rel=1e-12)
    # Same resampled pairs, so the interval inverts too (bounds swap).
    assert rev["ci_low"] == pytest.approx(1 / fwd["ci_high"], rel=1e-9)
    assert rev["ci_high"] == pytest.approx(1 / fwd["ci_low"], rel=1e-9)


def test_same_repetitions_give_the_same_interval():
    a, b = _noisy()
    assert speedup(a, b) == speedup(a, b)


@pytest.mark.parametrize("a, b", [([], []), ([1.0], [1.0, 2.0]), ([0.0], [1.0]), ([1.0], [-1.0])])
def test_bad_repetitions_are_refused(a, b):
    with pytest.raises(ValueError):
        speedup(a, b)


def test_min_pairs_above_max_pairs_is_refused():
    with pytest.raises(ValueError):
        paired_interleaved(lambda: None, lambda: None, min_pairs=5, max_pairs=4)
