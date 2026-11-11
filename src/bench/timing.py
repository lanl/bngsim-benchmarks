"""The paired, interleaved timing protocol (docs/DESIGN.md section 7).

Two engines, A (bngsim) and B (the reference), are timed in one process on one
core, alternating in an ABBA pattern so neither always runs first. Every
repetition is kept. The headline is the speedup of A over B, the ratio of the
median times, with a bootstrap confidence interval that resamples whole pairs.
Forming the ratio inside one job is what makes it robust to run-to-run drift and
to differences between nodes; absolute times are reported too, but they are
only comparable within one node type.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np


def paired_interleaved(
    run_a: Callable[[], Any],
    run_b: Callable[[], Any],
    *,
    warmup_pairs: int = 1,
    min_pairs: int = 5,
    max_pairs: int = 20,
    budget_seconds: float = 120.0,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Time ``run_a`` and ``run_b`` in alternation and return every repetition.

    Runs ``warmup_pairs`` discarded pairs, then pairs until ``max_pairs`` are
    measured, or until at least ``min_pairs`` are measured and the time spent
    measuring has passed ``budget_seconds``. Pair ``i`` runs A first when ``i``
    is even and B first when it is odd.
    """
    if min_pairs < 1 or max_pairs < min_pairs:
        raise ValueError("need 1 <= min_pairs <= max_pairs")

    def once(fn: Callable[[], Any]) -> float:
        t0 = clock()
        fn()
        return clock() - t0

    for i in range(warmup_pairs):
        if i % 2 == 0:
            once(run_a)
            once(run_b)
        else:
            once(run_b)
            once(run_a)

    a_times: list[float] = []
    b_times: list[float] = []
    spent = 0.0
    i = 0
    while len(a_times) < max_pairs:
        if len(a_times) >= min_pairs and spent >= budget_seconds:
            break
        if i % 2 == 0:
            ta = once(run_a)
            tb = once(run_b)
        else:
            tb = once(run_b)
            ta = once(run_a)
        a_times.append(ta)
        b_times.append(tb)
        spent += ta + tb
        i += 1
    return {
        "a_seconds": a_times,
        "b_seconds": b_times,
        "warmup_pairs": warmup_pairs,
        "pairs": len(a_times),
        "order": "ABBA",
    }


def speedup(
    a_seconds: list[float],
    b_seconds: list[float],
    *,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Speedup of A over B (median B / median A) with a bootstrap interval.

    Resampling draws whole pairs, so the interval keeps whatever the two engines
    shared in each pair (the machine's state at that moment). The seed is fixed
    so the same repetitions always give the same interval.
    """
    a = np.asarray(a_seconds, dtype=float)
    b = np.asarray(b_seconds, dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("need the same, non-zero number of A and B repetitions")
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("repetition times must be positive")
    point = float(np.median(b) / np.median(a))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    boot = np.median(b[idx], axis=1) / np.median(a[idx], axis=1)
    lo, hi = np.quantile(boot, [(1 - confidence) / 2, 1 - (1 - confidence) / 2])
    return {
        "speedup": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "confidence": confidence,
        "median_a_seconds": float(np.median(a)),
        "median_b_seconds": float(np.median(b)),
    }
