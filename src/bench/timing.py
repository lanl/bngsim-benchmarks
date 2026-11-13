"""The interleaved timing protocol (docs/DESIGN.md section 7).

Every engine compared on a problem -- BNGsim and each reference engine -- is
timed in one process on one pinned core, cycling round-robin (A B C A B C …) so
each engine's runs sit adjacent in time to the others'. The engine that goes
first rotates each round, so none is always first. Warm-up rounds are discarded
and every remaining repetition is kept.

Absolute time per engine is the artifact (section 7.5): ``absolute_time_stats``
reports each engine's median with a bootstrap confidence interval on the median,
resampling whole rounds so the interval keeps the machine state the engines
shared in each round. Forming the interleave is a measurement-hygiene win -- it
cancels the common-mode run-to-run drift of section 7.6 out of every comparison,
which is why it is kept even though the headline is now absolute times. A speedup
is only ever a view derived against a baseline the reader chooses; the suite
never stores an N×N matrix of pairwise ratios.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np


def round_robin_interleaved(
    runs: dict[str, Callable[[], Any]],
    *,
    warmup_rounds: int = 1,
    min_rounds: int = 5,
    max_rounds: int = 20,
    budget_seconds: float = 120.0,
    clock: Callable[[], float] = time.perf_counter,
    captures: dict[str, Callable[[Any], Any]] | None = None,
) -> dict[str, Any]:
    """Time every engine in ``runs`` round-robin and return every repetition.

    Each round runs every engine once. The engine that goes first rotates by one
    each round (measured round ``r`` starts at engine ``r`` modulo the engine
    count), so over the run no engine is always first or last (docs/DESIGN.md
    section 7.1). Runs ``warmup_rounds`` discarded rounds, then rounds until
    ``max_rounds`` are measured, or until at least ``min_rounds`` are measured and
    the time spent measuring has passed ``budget_seconds``.

    ``captures``, when given, maps an engine name to a function called on that
    engine's return value for every *kept* repetition (never a warm-up); the
    results are returned under that engine's ``outputs``, aligned with its
    ``seconds``. The capture runs outside the timed region, so it does not bias
    the measurement; what it extracts, and what is done with it, is the caller's
    business (this module knows nothing of trajectories).

    Returns ``{"rounds": n, "warmup_rounds": w, "order": "A B C",
    "engines": {name: {"seconds": [...], "outputs"?: [...]}}}``. Every engine has
    exactly ``rounds`` measurements, so the arrays line up round by round.
    """
    if not runs:
        raise ValueError("need at least one engine to time")
    if min_rounds < 1 or max_rounds < min_rounds:
        raise ValueError("need 1 <= min_rounds <= max_rounds")
    names = list(runs)
    n = len(names)
    captures = captures or {}

    def once(fn: Callable[[], Any]) -> tuple[float, Any]:
        t0 = clock()
        value = fn()
        return clock() - t0, value

    # Warm-up rounds, rotating the start so a warm-up is not always one engine.
    for i in range(warmup_rounds):
        for j in range(n):
            once(runs[names[(i + j) % n]])

    seconds: dict[str, list[float]] = {name: [] for name in names}
    outputs: dict[str, list[Any]] = {name: [] for name in names}
    spent = 0.0
    rounds = 0
    while rounds < max_rounds:
        if rounds >= min_rounds and spent >= budget_seconds:
            break
        start = rounds % n
        for j in range(n):
            name = names[(start + j) % n]
            dt, value = once(runs[name])
            seconds[name].append(dt)
            spent += dt
            if name in captures:
                outputs[name].append(captures[name](value))
        rounds += 1

    engines: dict[str, Any] = {}
    for name in names:
        entry: dict[str, Any] = {"seconds": seconds[name]}
        if name in captures:
            entry["outputs"] = outputs[name]
        engines[name] = entry
    return {
        "rounds": rounds,
        "warmup_rounds": warmup_rounds,
        "order": " ".join(names),
        "engines": engines,
    }


def absolute_time_stats(
    seconds_by_engine: dict[str, list[float]],
    *,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Per-engine median time with a bootstrap confidence interval on the median.

    The resampling draws whole rounds, and the *same* resampled round indices are
    applied to every engine, so the interval preserves what the engines shared in
    each round -- the machine's state at that moment. That is the
    measurement-hygiene win the interleave buys (docs/DESIGN.md section 7.5), and
    it needs every engine to have the same number of rounds, which
    ``round_robin_interleaved`` guarantees. The seed is fixed, so the same
    repetitions always give the same interval.

    Each engine's absolute median is the stored artifact; a speedup is only ever
    derived from these medians, never stored here.
    """
    names = list(seconds_by_engine)
    if not names:
        raise ValueError("need at least one engine")
    arrays = {name: np.asarray(seconds_by_engine[name], dtype=float) for name in names}
    sizes = {a.size for a in arrays.values()}
    if len(sizes) != 1 or 0 in sizes:
        raise ValueError("every engine needs the same, non-zero number of rounds")
    size = sizes.pop()
    for name, a in arrays.items():
        if np.any(a <= 0):
            raise ValueError(f"{name}: repetition times must be positive")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, size, size=(n_boot, size))
    lo_q, hi_q = (1 - confidence) / 2, 1 - (1 - confidence) / 2
    out: dict[str, dict[str, float]] = {}
    for name, a in arrays.items():
        boot = np.median(a[idx], axis=1)
        lo, hi = np.quantile(boot, [lo_q, hi_q])
        out[name] = {
            "median_seconds": float(np.median(a)),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "confidence": confidence,
            "rounds": int(size),
        }
    return out
