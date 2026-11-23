"""Partial scaling (PSA) of BNGL models: BNGsim's PSA against run_network's, and against exact SSA.

The system under test is BNGsim's partial-scaling approximation (Lin, Feng & Hlavacek
2019, J Chem Phys 150:244101), ``Simulator(method="psa", poplevel=N_c)``, at every
critical population size N_c of the protocol's grid (``poplevels``). A reaction whose
smallest participating population is N fires ``m = max(1, floor(N / N_c))`` molecules
at once at ``1/m`` of its propensity, so its mean drift is exact and its noise is that
of a population near N_c: PSA trades variance for events. On one problem it is
compared with:

- **run_network PSA** at the same N_c -- BioNetGen's own implementation of the method
  (``-p psa --poplevel N_c``, what BNG2.pl runs for ``simulate({method=>"psa",
  poplevel=>N_c})``), a fresh process per trajectory. This is the like-for-like check.
- **exact SSA**, by BNGsim (``method="ssa"``) and by run_network (``-p ssa``): the
  N_c = infinity row, and the truth PSA approximates at every N_c.

Each of the ``2 (K + 1)`` ensembles (two engines; exact SSA and K values of N_c) is
drawn as the ``ssa`` capability draws one -- the same engines, ``replicates``
trajectories on the uniform grid, fixed seeds from a block of its own -- in its own
forked child under a wall budget. The exact-SSA blocks are the ``ssa`` capability's
for these two engines, so the N_c = infinity ensembles are the ones an ``ssa`` job on
the same protocol draws.

**What is gated (decides PASS or DIFF).** Every gated comparison is the ``ssa``
capability's permutation max-T test on per-cell means and variances, false-alarm
rate ``VERDICT["alpha"]`` each, with bngsim's parity-core verdict beside it:

1. *Like for like*, at every N_c and at N_c = infinity: BNGsim's ensemble against
   run_network's. Both claim to implement one method; they must sample one law.
   (Verdict keys ``run_network@nc=<N_c>`` and, for exact SSA, ``run_network``.)
2. *Exactness where nothing scaled*: at an N_c where no BNGsim trajectory scaled any
   channel (every ``qexc_integral`` of its ``psa_diagnostics`` is exactly zero, so
   ``m = 1`` throughout), PSA *is* exact SSA, and BNGsim's PSA ensemble must sample
   the law of its exact one. A disagreement there is a defect, not an approximation.
   (Key ``exact@nc=<N_c>``.)

**What is reported (measured, never a failure).** At every N_c, each engine's PSA
ensemble against that engine's exact one: a means-only permutation max-T test (same
alpha), the largest mean shift in units of the exact standard deviation, and the
range of the variance ratio PSA / exact, each with the cell where it occurs; and
BNGsim's own diagnostics (the share of trajectories that scaled, the event-rate
ratio ``speedup``, the largest dwell-time multiplier). Variance inflation is the
method's known price and is only reported. Means are what PSA preserves to first
order, so **a flag** is raised -- in the record, beside the outcome, not instead of it
-- at every N_c of at least ``ACCURACY["flag_from_poplevel"]`` (bngsim documents
100-1000 as PSA's working range) where BNGsim's PSA scaled and its means differ
significantly from exact SSA's. The flag says whether run_network's PSA differs too
(then it is the method's inaccuracy on this model, not BNGsim's).

**Outcome.** As in ``ssa``: EXCEPTION (TIMEOUT) when a BNGsim ensemble failed,
REFERENCE_FAILED when a run_network ensemble failed (with a ``<variant>_error``;
run_network's PSA is known to abort on a zeroth-order reaction, and that is recorded,
never worked around), PASS when every gated comparison passed. A gated comparison
that failed makes the job DIFF even when some reference failed, so a wrong answer is
never timed.

**Timing.** Every ensemble of the correctness job is timed again, warm: ``timing_
replicates`` trajectories per repetition, the same seeds every round, all interleaved
round-robin (``bench.timing``), each repetition checked against the first. Each
variant's absolute time per ensemble and per trajectory is the artifact; the PSA
speed-up over exact SSA of the same engine at each N_c is derived from them in the
report, beside BNGsim's event-rate ratio. The one-time cost -- each BNGsim
simulator's load and build -- is measured once, cold; run_network loads the ``.net``
in every trajectory. A run_network variant the correctness job recorded as failed is
not timed.

Input formats: BNGsim's PSA reads ``.net``, BNGL and SBML (``sbml_psa``); there is no
independent PSA engine on SBML, so this capability takes BNGL only.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .. import netfile, stochastic
from ..jobs import CORRECTNESS, TIMING, Job
from ..specs import Problem
from ..timing import absolute_time_stats, round_robin_interleaved
from . import ssa
from ._common import (
    engine_role,
    outcome,
    prepare_network,
    progress,
    reproducibility,
    run_isolated,
    threads,
)

CAPABILITY = "psa"
# Bump when this module's computation changes; every result it wrote goes stale.
HARNESS_VERSION = "1"

REFERENCE = "run_network"
ENGINES = ("bngsim", REFERENCE)
EXACT = "exact"
DEFAULT_POPLEVELS = (10, 30, 100, 300, 1000)

# The gated comparisons' test: the ssa capability's, minus its exact-moments half.
VERDICT = {k: ssa.VERDICT[k] for k in ("two_sample", "alpha", "permutations", "stop_exceedances")}
# The reported comparison against exact SSA, and from which N_c a significant
# difference in means is flagged. In the correctness definition, like VERDICT.
ACCURACY = {"means": "permutation max-T on means/1", "flag_from_poplevel": 100}

# Seeds: variant k (0 = exact SSA, k = i + 1 for poplevels[i]) of engine e draws
# from block 2k + e, of ssa.SEED_BLOCK seeds each. k = 0 is the ssa capability's
# block for each engine.
ENGINE_BLOCKS = {"bngsim": 0, "run_network": 1}

# Planning: a PSA trajectory can cost more than an exact one (a channel that does
# not scale still pays for computing its multiplier), so it is planned at this many
# exact trajectories.
PSA_COST_FACTOR = 2.0
# One run_network trajectory is stopped after this many times its planned cost (at
# least MIN_TRAJECTORY_TIMEOUT s, at most the ensemble's budget): a process that
# hangs (run_network does, on the Rohrs 2018 network) fails its ensemble then, not
# after the whole budget. Planning never enters a freshness key.
TRAJECTORY_TIMEOUT_FACTOR = 50.0
MIN_TRAJECTORY_TIMEOUT = 60.0


# --------------------------------------------------------------------------- #
# Protocol, names and seeds
# --------------------------------------------------------------------------- #
def protocol(problem: Problem) -> dict[str, Any]:
    """The PSA protocol with defaults filled in, as it enters a job definition."""
    p = dict(problem.protocol_for(CAPABILITY))
    p.setdefault("seed", ssa.DEFAULT_SEED)
    p.setdefault("timing_replicates", ssa.DEFAULT_TIMING_REPLICATES)
    p.setdefault("poplevels", list(DEFAULT_POPLEVELS))
    return p


def nc_text(nc: float) -> str:
    """N_c as written in names and on run_network's command line: ``100``, ``2.5``."""
    v = float(nc)
    return str(int(v)) if v.is_integer() else repr(v)


def label(nc: float | None) -> str:
    """``exact`` for N_c = infinity, else ``nc=<N_c>``."""
    return EXACT if nc is None else f"nc={nc_text(nc)}"


def variant(engine: str, nc: float | None) -> str:
    """An engine at one N_c: ``bngsim`` (its exact SSA), ``bngsim@nc=100``."""
    return engine if nc is None else f"{engine}@{label(nc)}"


def base_engine(name: str) -> str:
    return name.split("@", 1)[0]


def exactness_key(nc: float) -> str:
    """The verdict of BNGsim's PSA at ``nc`` against its exact SSA (where nothing scaled)."""
    return f"{EXACT}@{label(nc)}"


def levels(proto: dict[str, Any]) -> list[float | None]:
    """Exact SSA (None) first, then the protocol's N_c in order."""
    return [None, *proto["poplevels"]]


def variants(proto: dict[str, Any]) -> list[tuple[str, str, float | None, int]]:
    """Every ensemble a job draws: (name, engine, N_c, level index)."""
    return [(variant(engine, nc), engine, nc, k)
            for k, nc in enumerate(levels(proto)) for engine in ENGINES]


def seed_for(engine: str, k: int, seed: int) -> int:
    """The first seed of ``engine``'s ensemble at level ``k`` (0 = exact SSA)."""
    return int(seed) + ssa.SEED_BLOCK * (2 * int(k) + ENGINE_BLOCKS[engine])


def engines_id() -> str:
    return "bngsim-vs-" + REFERENCE


def n_comparisons(proto: dict[str, Any]) -> int:
    """Permutation tests one job may run: like for like at K + 1 levels, exactness at
    up to K, and the means of both engines against exact at K."""
    k = len(proto["poplevels"])
    return (k + 1) + k + 2 * k


def jobs(problem: Problem) -> list[Job]:
    # BNGL only: no independent PSA engine reads SBML.
    if problem.source.get("format") != "bngl":
        return []
    proto = protocol(problem)
    engines = engines_id()
    base = {
        "harness": f"{CAPABILITY}/{engines}/{HARNESS_VERSION}",
        "source_sha256": problem.source["sha256"],
        "protocol": proto,
        "setting": {"method": "psa", "exact": "ssa", "propensities": "auto"},
    }
    est_c, est_t = planning_seconds(problem)
    backstop = (len(variants(proto)) + 1) * engine_budget_seconds(problem) \
        + ssa.VERDICT_SECONDS * n_comparisons(proto)
    correctness = Job(
        CAPABILITY, problem.id, engines, "auto", CORRECTNESS,
        definition={**base, "verdict": VERDICT, "accuracy": ACCURACY},
        est_seconds=est_c,
        timeout_seconds=max(ssa.MIN_TIMEOUT_SECONDS, ssa.TIMEOUT_FACTOR * est_c, backstop),
    )
    timing = Job(
        CAPABILITY, problem.id, engines, "auto", TIMING,
        definition={**base, "timing": ssa.TIMING_PROTOCOL,
                    "timing_check": ssa.REPRO_CHECK_VERSION},
        est_seconds=est_t,
        timeout_seconds=max(ssa.MIN_TIMEOUT_SECONDS, ssa.TIMEOUT_FACTOR * est_t),
        requires=correctness.id,
    )
    return [correctness, timing]


# --------------------------------------------------------------------------- #
# Planning (never part of a freshness key)
# --------------------------------------------------------------------------- #
def trajectory_seconds(problem: Problem) -> dict[str, float]:
    """Planned seconds per exact-SSA trajectory, per engine, from the spec's
    ``size.psa_events`` (exact-SSA events per trajectory under this protocol), else
    ``size.ssa_events``."""
    events = problem.size.get("psa_events", problem.size.get("ssa_events", ssa.DEFAULT_EVENTS))
    return {name: ssa.TRAJECTORY_SECONDS.get(name, 1e-3) + ssa.EVENT_SECONDS * events
            for name in ENGINES}


def _all_ensembles_seconds(problem: Problem, n: int) -> float:
    k = len(protocol(problem)["poplevels"])
    return int(n) * sum(trajectory_seconds(problem).values()) * (1 + PSA_COST_FACTOR * k)


def planning_seconds(problem: Problem) -> tuple[float, float]:
    """Rough wall-clock estimates (correctness, timing) for one problem."""
    proto = protocol(problem)
    correctness = 30.0 + _all_ensembles_seconds(problem, proto["replicates"]) \
        + ssa.VERDICT_SECONDS * n_comparisons(proto)
    tp = ssa.TIMING_PROTOCOL
    rnd = _all_ensembles_seconds(problem, proto["timing_replicates"])
    measured = max(tp["min_rounds"] * rnd, min(tp["budget_seconds"] + rnd, tp["max_rounds"] * rnd))
    return correctness, 60.0 + (tp["warmup_rounds"] + 1) * rnd + measured


def trajectory_timeout(problem: Problem, budget: float) -> float:
    """Seconds one run_network trajectory may take before it is stopped."""
    planned = max(trajectory_seconds(problem).values()) * PSA_COST_FACTOR
    return min(float(budget), max(MIN_TRAJECTORY_TIMEOUT, TRAJECTORY_TIMEOUT_FACTOR * planned))


def _timeouts(problem: Problem, ctx: dict[str, Any]) -> tuple[float, float]:
    """(ensemble budget, run_network trajectory timeout) for a job; ``ctx`` may set
    either (``engine_budget_seconds``, ``trajectory_timeout_seconds``)."""
    budget = float(ctx.get("engine_budget_seconds") or engine_budget_seconds(problem))
    traj = float(ctx.get("trajectory_timeout_seconds") or trajectory_timeout(problem, budget))
    return budget, min(float(ctx["subprocess_timeout"]), traj)


def engine_budget_seconds(problem: Problem) -> float:
    """Wall budget for one ensemble's cold build + sampling in the correctness job."""
    slowest = max(trajectory_seconds(problem).values()) * PSA_COST_FACTOR \
        * int(protocol(problem)["replicates"])
    return min(ssa.ENGINE_BUDGET_CAP_SECONDS,
               max(ssa.MIN_TIMEOUT_SECONDS, ssa.TIMEOUT_FACTOR * slowest))


# --------------------------------------------------------------------------- #
# The engines: the ssa capability's, at one N_c
# --------------------------------------------------------------------------- #
def psa_summary(per_run: list[dict[str, Any]]) -> dict[str, Any]:
    """An ensemble's PSA diagnostics, one dict per trajectory (``Result.psa_diagnostics``),
    in a few numbers. A trajectory *scaled* when some channel fired with ``m > 1`` while
    it had propensity, i.e. some ``qexc_integral`` is nonzero; one that did not was a
    run of exact SSA (``activation_crossed`` only says scaling became possible). The
    ensemble's ``regime`` is ``exact`` when no trajectory scaled."""
    n = len(per_run)
    if not n:
        return {"trajectories": 0, "regime": None}
    scaled = [any(float(q) > 0.0 for q in (d.get("qexc_integral") or [])) for d in per_run]
    crossed = [bool(d.get("activation_crossed")) for d in per_run]
    sp = np.array([float(d.get("speedup", math.nan)) for d in per_run], dtype=float)
    sp = sp[np.isfinite(sp)]
    best, where = 1.0, None
    reactions: set[int] = set()
    for d in per_run:
        idx = list(d.get("reaction_index") or [])
        for i, (m, q) in enumerate(zip(d.get("mbar") or [], d.get("qexc_integral") or [])):
            if float(q) > 0.0 and i < len(idx):
                reactions.add(int(idx[i]))
            if float(m) > best:
                best, where = float(m), (int(idx[i]) if i < len(idx) else None)
    return {
        "trajectories": n,
        "regime": "scaled" if any(scaled) else EXACT,
        "scaled_fraction": float(np.mean(scaled)),
        "crossed_fraction": float(np.mean(crossed)),
        # The event-rate ratio exact / scaled events of each trajectory (bngsim's
        # ``speedup``): not a wall-clock speed-up, which the timing job measures.
        "speedup": ({"mean": float(sp.mean()), "median": float(np.median(sp)),
                     "min": float(sp.min()), "max": float(sp.max())} if sp.size else None),
        "mbar_max": {"value": best, "reaction": where},
        "scaled_reactions": sorted(reactions),
        "peak_population": float(max(float(d.get("peak_population") or 0.0) for d in per_run)),
    }


class BngsimPSA(ssa.BngsimSSA):
    """BNGsim's PSA at one N_c, ``Simulator(method="psa", poplevel=N_c)``, sampled as
    ``ssa.BngsimSSA`` samples exact SSA; its record adds the PSA diagnostics."""

    def __init__(self, net: Path, proto: dict[str, Any], poplevel: float):
        self.poplevel = float(poplevel)
        super().__init__(net, proto)

    def _simulator(self, bngsim):
        return bngsim.Simulator(self.model, method="psa", poplevel=self.poplevel)

    def _details(self, results: list) -> dict[str, Any]:
        out = super()._details(results)
        out["psa"] = psa_summary([dict(getattr(r, "psa_diagnostics", None) or {})
                                  for r in results])
        return out

    def last_summary(self) -> dict[str, Any]:
        """The PSA diagnostics of the last ``ensemble`` (outside any timed region)."""
        result = self.last_result
        per = list(getattr(result, "psa_diagnostics_per_run", None) or []) if result else []
        return psa_summary([dict(d) for d in per])

    def config(self) -> dict[str, Any]:
        return {**super().config(), "method": "psa", "poplevel": self.poplevel}


class RunNetworkPSA(ssa.RunNetworkSSA):
    """run_network's PSA at one N_c (its heterogeneous adaptive scaling method), one
    fresh process per trajectory, as ``ssa.RunNetworkSSA`` runs exact SSA."""

    def __init__(self, net: Path, proto: dict[str, Any], out_prefix: Path, *, timeout: float,
                 poplevel: float):
        super().__init__(net, proto, out_prefix, timeout=timeout)
        self.poplevel = float(poplevel)
        self.METHOD_ARGS = ("-p", "psa", "--poplevel", nc_text(poplevel))

    def config(self) -> dict[str, Any]:
        return {**super().config(), "method": "psa (heterogeneous adaptive scaling)",
                "poplevel": self.poplevel}


BUILD_STAGE = {"bngsim": "load + simulator build", "run_network": "trajectories (subprocesses)"}


def _build(engine: str, nc: float | None, prep: dict[str, Any], *, out_prefix: Path,
           timeout: float):
    net, proto = prep["net"], prep["proto"]
    if engine == "bngsim":
        return ssa.BngsimSSA(net, proto) if nc is None else BngsimPSA(net, proto, nc)
    if nc is None:
        return ssa.RunNetworkSSA(net, proto, out_prefix, timeout=timeout)
    return RunNetworkPSA(net, proto, out_prefix, timeout=timeout, poplevel=nc)


def _progress(msg: str) -> None:
    progress(CAPABILITY, msg)


# --------------------------------------------------------------------------- #
# Preparation, ensembles and comparisons
# --------------------------------------------------------------------------- #
def _prepare(problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    proto = protocol(problem)
    prep = prepare_network(problem, ctx, proto["netgen"])
    prep["proto"] = proto
    # The zeroth-order reactions, read by the harness itself: run_network's PSA is
    # known to abort on one, and the record should show the network has them.
    try:
        network = netfile.read(prep["net"])
        prep["info"]["zeroth_order_reactions"] = sum(1 for r in network.reactions
                                                     if not r.reactants)
    except (netfile.NetFileError, ValueError, ArithmeticError) as exc:
        prep["info"]["netfile_error"] = f"{type(exc).__name__}: {exc}"[:400]
    return prep


def _cold(engine: str, nc: float | None, prep: dict[str, Any], n: int, seed: int,
          out_prefix: Path, timeout: float):
    def fn(stage):
        t0 = time.perf_counter()
        stage(BUILD_STAGE[engine])
        eng = _build(engine, nc, prep, out_prefix=out_prefix, timeout=timeout)
        stage(f"ensemble of {n}")
        t1 = time.perf_counter()
        stack = eng.ensemble_detailed(n, seed) if engine == "bngsim" else eng.ensemble(n, seed)
        return stack, {**eng.config(), "ensemble_seconds": time.perf_counter() - t1,
                       "cold_seconds": time.perf_counter() - t0, "first_seed": seed}
    return fn


def effect_sizes(a: np.ndarray, b: np.ndarray, *, times: np.ndarray | None = None,
                 species: list[str] | None = None) -> dict[str, Any]:
    """How far ensemble ``a`` (PSA) is from ``b`` (exact SSA), in units that do not shrink
    with the ensemble size, each at its worst cell with that cell's standard error in
    the same units (``se``, the noise the value carries):

    - ``mean_error_over_peak``: ``|mean_a - mean_b|`` over the peak of the species'
      exact mean, over the species that are ever nonzero -- the robust summary, as the
      ODE verdict measures error;
    - ``mean_shift_sd``: ``|mean_a - mean_b|`` over the exact standard deviation, over
      the cells where ``b`` varies. Where PSA has inflated a cell's variance, its own
      sampling noise is large in these units (``se`` says how large);
    - ``variance_ratio``: ``var_a / var_b`` over the cells where ``b`` varies -- its
      median, and its extremes with where they sit.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    grid = b.shape[1:]
    if a.shape[1:] != grid:
        return {"mean_error_over_peak": None, "mean_shift_sd": None, "variance_ratio": None}
    na, nb = a.shape[0], b.shape[0]
    xa, xb = a.reshape(na, -1), b.reshape(nb, -1)
    ma, mb = xa.mean(axis=0), xb.mean(axis=0)
    va, vb = xa.var(axis=0, ddof=1), xb.var(axis=0, ddof=1)
    diff = np.abs(ma - mb)
    se = np.sqrt(va / na + vb / nb)

    def at(cell: int) -> dict[str, Any]:
        return stochastic._locate(cell, grid, times, species)

    out: dict[str, Any] = {"mean_error_over_peak": None, "mean_shift_sd": None,
                           "variance_ratio": None}
    peak = np.repeat(np.abs(mb.reshape(grid)).max(axis=0)[None, :], grid[0], axis=0).ravel()
    moving = peak > 0
    if moving.any():
        rel = np.where(moving, diff / np.where(moving, peak, 1.0), -1.0)
        c = int(np.argmax(rel))
        out["mean_error_over_peak"] = {"max": float(rel[c]), "se": float(se[c] / peak[c]),
                                       "at": at(c)}
    live = vb > 0
    if live.any():
        sd = np.sqrt(np.where(live, vb, 1.0))
        shift = np.where(live, diff / sd, -1.0)
        c = int(np.argmax(shift))
        out["mean_shift_sd"] = {"max": float(shift[c]), "se": float(se[c] / sd[c]), "at": at(c)}
        cells = np.flatnonzero(live)
        ratio = va[live] / vb[live]
        hi, lo = int(np.argmax(ratio)), int(np.argmin(ratio))
        out["variance_ratio"] = {"median": float(np.median(ratio)),
                                 "max": float(ratio[hi]), "at_max": at(cells[hi]),
                                 "min": float(ratio[lo]), "at_min": at(cells[lo]),
                                 "n_cells": int(cells.size)}
    return out


_MEANS_KEEP = ("passed", "p_value", "alpha", "permutations", "stopped_early", "z_max",
               "worst", "test", "replicates", "reason")


def _accuracy(psa_stack: np.ndarray, exact_stack: np.ndarray, k: int,
              prep: dict[str, Any]) -> dict[str, Any]:
    """A PSA ensemble against exact SSA of the same engine: reported, not gated."""
    proto, t = prep["proto"], ssa.times(prep["proto"])
    v = stochastic.two_sample_verdict(
        psa_stack, exact_stack, alpha=VERDICT["alpha"], permutations=VERDICT["permutations"],
        stop_exceedances=VERDICT["stop_exceedances"],
        seed=ssa._permutation_seed(proto["seed"], k), times=t,
        species=prep["species_names"], variance=False)
    return {"means": {key: v[key] for key in _MEANS_KEEP if key in v},
            **effect_sizes(psa_stack, exact_stack, times=t, species=prep["species_names"])}


def _flags(proto: dict[str, Any], accuracy: dict[str, Any]) -> list[dict[str, Any]]:
    """The large-N_c flags: BNGsim's PSA scaled, and its means differ from exact SSA's."""
    out = []
    for nc in proto["poplevels"]:
        acc = accuracy.get(label(nc)) or {}
        means = (acc.get("bngsim") or {}).get("means") or {}
        if nc < ACCURACY["flag_from_poplevel"] or acc.get("regime") != "scaled" \
                or means.get("passed") is not False:
            continue
        rn = (acc.get(REFERENCE) or {}).get("means")
        out.append({
            "poplevel": nc,
            "flag": "bngsim's PSA means differ from exact SSA",
            "p_value": means.get("p_value"),
            "worst": means.get("worst"),
            "mean_error_over_peak": (acc["bngsim"].get("mean_error_over_peak") or {}).get("max"),
            "mean_shift_sd": (acc["bngsim"].get("mean_shift_sd") or {}).get("max"),
            # True: run_network's PSA differs too, so it is the method's error on this
            # model; False: only bngsim's does; None: run_network's could not be judged.
            "run_network_differs_too": None if not rn else (rn.get("passed") is False),
        })
    return out


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
def run_correctness(job: Job, problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    try:
        prep = _prepare(problem, ctx)
    except Exception as exc:  # noqa: BLE001
        return {"outcome": "BAD_TEST", "error": f"{type(exc).__name__}: {exc}"}
    proto, info = prep["proto"], prep["info"]
    n, seed = int(proto["replicates"]), int(proto["seed"])
    workdir = Path(ctx["workdir"])
    budget, timeout = _timeouts(problem, ctx)
    info["engine_budget_seconds"] = budget
    info["trajectory_timeout_seconds"] = timeout

    ensembles: dict[str, np.ndarray] = {}
    configs: dict[str, dict[str, Any]] = {}
    bn_errors: dict[str, str] = {}
    ref_errors: dict[str, str] = {}
    for name, engine, nc, k in variants(proto):
        _progress(f"{name}: start (budget {budget:g} s; {threads()} threads in the worker)")
        out = run_isolated(_cold(engine, nc, prep, n, seed_for(engine, k, seed),
                                 workdir / name.replace("@", "_") / "rn", timeout),
                           budget, workdir / "isolated" / name.replace("@", "_"))
        _progress(f"{name}: {out.get('error') or 'done'} after {out['seconds']:.1f} s")
        if "error" in out:
            (bn_errors if engine == "bngsim" else ref_errors)[name] = out["error"]
        else:
            ensembles[name] = out["trajectory"]
            configs[name] = out["config"]
    info["engines"] = configs

    # Gated: like for like at every level, and exactness where nothing scaled.
    verdicts: dict[str, Any] = {}
    for k, nc in enumerate(levels(proto)):
        bn, ref = variant("bngsim", nc), variant(REFERENCE, nc)
        if bn in ensembles and ref in ensembles:
            _progress(f"verdict {bn} vs {ref}")
            verdicts[ref] = ssa._two_sample(ensembles[bn], ensembles[ref], 2 * k, prep)
    regimes = {label(nc): (configs.get(variant("bngsim", nc), {}).get("psa") or {}).get("regime")
               for nc in proto["poplevels"]}
    for k, nc in enumerate(proto["poplevels"], start=1):
        bn = variant("bngsim", nc)
        if regimes[label(nc)] == EXACT and bn in ensembles and "bngsim" in ensembles:
            _progress(f"verdict {bn} vs bngsim (nothing scaled)")
            verdicts[exactness_key(nc)] = ssa._two_sample(ensembles[bn], ensembles["bngsim"],
                                                          2 * k + 1, prep)

    # Reported: each engine's PSA against its own exact SSA, per N_c.
    accuracy: dict[str, Any] = {}
    for k, nc in enumerate(proto["poplevels"], start=1):
        entry: dict[str, Any] = {"poplevel": nc, "regime": regimes[label(nc)],
                                 "bngsim_psa": configs.get(variant("bngsim", nc), {}).get("psa")}
        for e, engine in enumerate(ENGINES):
            approx, exact = variant(engine, nc), variant(engine, None)
            if approx in ensembles and exact in ensembles:
                _progress(f"accuracy of {approx} against {exact}")
                entry[engine] = _accuracy(ensembles[approx], ensembles[exact],
                                          1000 + 2 * k + e, prep)
        accuracy[label(nc)] = entry
    flags = _flags(proto, accuracy)
    info["psa"] = {"poplevels": list(proto["poplevels"]),
                   "flag_from_poplevel": ACCURACY["flag_from_poplevel"],
                   "accuracy": accuracy, "flags": flags}
    info["ensembles"] = {name: ssa._final_moments(stack, prep["species_names"])
                         for name, stack in ensembles.items()}

    n_refs = len(levels(proto))
    failed = sorted(name for name, v in verdicts.items() if not v.get("passed"))
    if bn_errors:
        bn_error = "; ".join(f"{name}: {err}" for name, err in bn_errors.items())[:2000]
        result = outcome(bn_error, ref_errors, n_refs, False)
        if result == "EXCEPTION" and all(e.startswith("TIMEOUT") for e in bn_errors.values()):
            result = "TIMEOUT"
    elif failed:
        result = "DIFF"  # a wrong answer, whatever else failed
    else:
        result = outcome(None, ref_errors, n_refs, bool(verdicts))
    rec: dict[str, Any] = {"outcome": result, "measurements": info}
    if verdicts:
        rec["verdicts"] = verdicts
    if flags:
        rec["psa_flags"] = [f"nc={nc_text(f['poplevel'])}: {f['flag']} "
                            f"(p {_g(f['p_value'], '.2g')})" for f in flags]
    if bn_errors:
        rec["bngsim_error"] = bn_error
    for name, err in ref_errors.items():
        rec[f"{name}_error"] = err
    return rec


def _correctness_errors(job: Job, ctx: dict[str, Any]) -> dict[str, str]:
    """The run_network variants this timing job's correctness job recorded as failed."""
    from ..results import result_path

    requires, results_dir = getattr(job, "requires", None), ctx.get("results_dir")
    if not requires or not results_dir:
        return {}
    try:
        rec = json.loads(result_path(results_dir, requires).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k[: -len("_error")]: str(v) for k, v in rec.items()
            if k.endswith("_error") and base_engine(k[: -len("_error")]) == REFERENCE and v}


def run_timing(job: Job, problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    try:
        prep = _prepare(problem, ctx)
    except Exception as exc:  # noqa: BLE001 - a bad test input, a labelled row
        return {"outcome": "BAD_TEST", "error": f"{type(exc).__name__}: {exc}"}
    proto, info = prep["proto"], prep["info"]
    workdir = Path(ctx["workdir"])
    _, timeout = _timeouts(problem, ctx)
    k_traj, seed = int(proto["timing_replicates"]), int(proto["seed"])
    info["timing_replicates"] = k_traj
    failed_before = _correctness_errors(job, ctx)

    engines: dict[str, Any] = {}
    seeds: dict[str, int] = {}
    configs: dict[str, dict[str, Any]] = {}
    anchors: dict[str, np.ndarray] = {}
    ref_errors: dict[str, str] = {}
    for name, engine, nc, k in variants(proto):
        s = seed_for(engine, k, seed)
        if engine == "bngsim":
            # Cold: load, simulator build and a first ensemble, with the empty
            # compile cache the worker gave this job. It anchors the warm ones.
            t0 = time.perf_counter()
            eng = _build(engine, nc, prep, out_prefix=workdir, timeout=timeout)
            first = eng.ensemble(k_traj, s)
            cfg = {**eng.config(), "cold_seconds": time.perf_counter() - t0}
            if nc is not None:
                cfg["psa"] = eng.last_summary()
            anchors[name] = ssa._summary(first)
        else:
            if name in failed_before:
                ref_errors[name] = ("failed in the correctness job, so not timed ("
                                    + failed_before[name][:300] + ")")
                _progress(f"{name}: skipped, {ref_errors[name]}")
                continue
            _progress(f"{name}: build + trial run")
            try:
                eng = _build(engine, nc, prep, out_prefix=workdir / name.replace("@", "_") / "rn",
                             timeout=timeout)
                eng.ensemble(1, s)
                cfg = eng.config()
            except Exception as exc:  # noqa: BLE001
                ref_errors[name] = f"{type(exc).__name__}: {exc}"[:400]
                continue
        engines[name], seeds[name], configs[name] = eng, s, cfg
    info["bngsim_cold_seconds"] = configs["bngsim"]["cold_seconds"]

    _progress(f"interleaved timing of {', '.join(engines)}, {k_traj} trajectories per ensemble")
    runs = {name: (lambda e=eng, s=seeds[name]: e.ensemble(k_traj, s))
            for name, eng in engines.items()}
    reps = round_robin_interleaved(runs, captures={name: ssa._summary for name in runs},
                                   **job.definition["timing"])
    outputs = {name: reps["engines"][name].pop("outputs") for name in runs}
    repro = {name: (reproducibility(anchors[name], outs, ssa.REPRO_TOL) if name in anchors
                    else reproducibility(outs[0], outs[1:], ssa.REPRO_TOL))
             for name, outs in outputs.items()}
    reproducible = all(r["passed"] for r in repro.values())

    times_ = absolute_time_stats({name: reps["engines"][name]["seconds"] for name in runs})
    per_trajectory = {name: {key: t[key] / k_traj
                             for key in ("median_seconds", "ci_low", "ci_high")}
                      for name, t in times_.items()}
    rec: dict[str, Any] = {
        "outcome": "PASS" if reproducible else "NOT_REPRODUCIBLE",
        "measurements": {
            **info,
            # The exact-SSA anchor, as the gathered tables expect it; every other
            # variant (BNGsim's PSA at each N_c, run_network) beside it.
            "bngsim": configs["bngsim"],
            "references": {name: cfg for name, cfg in configs.items() if name != "bngsim"},
            "engine_roles": {name: engine_role(base_engine(name)) for name in engines},
            "repetitions": reps,
            "reproducibility": repro,
            "times": times_,
            "times_per_trajectory": per_trajectory,
        },
    }
    for name, err in ref_errors.items():
        rec[f"{name}_error"] = err
    return rec


def run(job: Job, problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    if job.kind == CORRECTNESS:
        return run_correctness(job, problem, ctx)
    if job.kind == TIMING:
        return run_timing(job, problem, ctx)
    raise ValueError(f"unknown job kind {job.kind!r}")


# --------------------------------------------------------------------------- #
# The gathered report: one row per (problem, N_c)
# --------------------------------------------------------------------------- #
def report_fields(kind: str, m: dict[str, Any]) -> dict[str, Any]:
    """What a gathered row keeps beyond the stochastic fields every row has: the
    per-N_c accuracy and flags (correctness), and BNGsim's event-rate ratio per
    variant (timing; the wall-clock times are already in the row)."""
    if kind == CORRECTNESS:
        psa = m.get("psa") or {}
        acc = {}
        for name, a in (psa.get("accuracy") or {}).items():
            row = {"poplevel": a.get("poplevel"), "regime": a.get("regime")}
            diag = a.get("bngsim_psa") or {}
            row["scaled_fraction"] = diag.get("scaled_fraction")
            row["speedup_events"] = (diag.get("speedup") or {}).get("median")
            for engine in ENGINES:
                e = a.get(engine) or {}
                means = e.get("means") or {}
                row[engine] = {"means_passed": means.get("passed"),
                               "means_p_value": means.get("p_value"),
                               "mean_error_over_peak":
                                   (e.get("mean_error_over_peak") or {}).get("max"),
                               "mean_error_over_peak_se":
                                   (e.get("mean_error_over_peak") or {}).get("se"),
                               "mean_shift_sd": (e.get("mean_shift_sd") or {}).get("max"),
                               "variance_ratio_median":
                                   (e.get("variance_ratio") or {}).get("median"),
                               "variance_ratio_max": (e.get("variance_ratio") or {}).get("max")}
            acc[name] = row
        return {"psa": {"poplevels": psa.get("poplevels"), "accuracy": acc,
                        "flags": psa.get("flags") or []}}
    refs = m.get("references") or {}
    return {"psa_speedup_events": {name: ((cfg.get("psa") or {}).get("speedup") or {}).get("median")
                                   for name, cfg in refs.items() if base_engine(name) == "bngsim"}}


def _g(x: Any, spec: str = ".3g") -> str:
    return "" if x is None else format(x, spec)


def _why(row: dict[str, Any], name: str) -> str:
    err = str(row.get(f"{name}_error") or "")
    if not err:
        return ""
    return "timeout" if err.startswith("TIMEOUT") else "crash"


def report_lines(kind: str, rows: list[dict[str, Any]]) -> list[str]:
    """The PSA tables of the summary, one row per (problem, N_c)."""
    if kind == CORRECTNESS:
        return _accuracy_table(rows)
    if kind == TIMING:
        return _speedup_table(rows)
    return []


def _accuracy_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["", "### PSA against exact SSA, per N_c", "",
             "| problem | N_c | bngsim regime | scaled | events ratio | vs run_network PSA "
             "| means vs exact: bngsim | means vs exact: run_network | mean error / peak "
             "| variance ratio (median, max) | flag |",
             "|---|---:|---|---:|---:|---|---|---|---:|---:|---|"]
    for r in rows:
        psa = r.get("psa") or {}
        flagged = {f.get("poplevel") for f in psa.get("flags") or []}
        verdicts = r.get("verdicts") or {}
        for nc in psa.get("poplevels") or []:
            a = (psa.get("accuracy") or {}).get(label(nc)) or {}
            ref = variant(REFERENCE, nc)
            v = verdicts.get(ref)
            like = (("ok" if v.get("passed") else "DIFF") + f" (p {v['p_value']:.2g})"
                    if v and v.get("p_value") is not None else _why(r, ref))
            ex = verdicts.get(exactness_key(nc))
            regime = a.get("regime") or ""
            if ex is not None:
                regime += (" (= exact: ok)" if ex.get("passed") else " (= exact: DIFF)")

            def means(engine: str) -> str:
                e = a.get(engine) or {}
                if e.get("means_passed") is None:
                    return ""
                return (("same" if e["means_passed"] else "differ")
                        + f" (p {_g(e.get('means_p_value'), '.2g')})")

            bn = a.get("bngsim") or {}
            frac = a.get("scaled_fraction")
            lines.append("| " + " | ".join([
                r["problem"], nc_text(nc), regime,
                "" if frac is None else f"{frac:.0%}", _g(a.get("speedup_events")), like,
                means("bngsim"), means(REFERENCE),
                (f"{_g(bn.get('mean_error_over_peak'), '.2g')} ± "
                 f"{_g(bn.get('mean_error_over_peak_se'), '.1g')}"
                 if bn.get("mean_error_over_peak") is not None else ""),
                (f"{_g(bn.get('variance_ratio_median'), '.3g')}, "
                 f"{_g(bn.get('variance_ratio_max'), '.3g')}"
                 if bn.get("variance_ratio_max") is not None else ""),
                "FLAG" if nc in flagged else "",
            ]) + " |")
    lines += ["", "*regime*: *scaled* when some bngsim trajectory fired a channel with "
              "multiplier m > 1, *exact* when none did -- PSA is then exact SSA, and "
              "bngsim's PSA ensemble is gated against its exact one (*= exact*). *scaled*: "
              "the share of trajectories that scaled; *events ratio*: bngsim's median "
              "exact/scaled event count per trajectory (not a wall-clock speed-up). "
              "*vs run_network PSA* is gated (permutation max-T on means and variances). "
              "The rest is measured and never fails a job: *means vs exact* is a "
              "means-only permutation max-T test of each engine's PSA against its own "
              "exact SSA; *mean error / peak* is bngsim's largest |mean difference| as a "
              "fraction of the species' exact peak mean, ± its standard error at that cell "
              "(an error of two or three of them is noise), *variance ratio* the median and "
              "largest PSA/exact variance ratio over cells (each with its cell and standard "
              "error, and the shift in exact-SSA SDs, in the JSON row). *FLAG*: at an N_c "
              "in PSA's working range bngsim's PSA scaled and its means differ from exact "
              "SSA's (the JSON row says whether run_network's PSA differs too)."]
    return lines


def _speedup_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["", "### PSA speed-up over exact SSA of the same engine, per N_c", "",
             "| problem | N_c | bngsim exact (s/traj) | bngsim PSA (s/traj) | bngsim speed-up "
             "| events ratio | run_network exact (s/traj) | run_network PSA (s/traj) "
             "| run_network speed-up |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        per = r.get("times_per_trajectory") or {}
        trusted = r.get("outcome") == "PASS"
        events = r.get("psa_speedup_events") or {}
        ncs = sorted({float(name.split("=", 1)[1]) for name in per if "@nc=" in name}
                     | {float(name.split("=", 1)[1]) for name in events if "@nc=" in name})
        for nc in ncs:
            cells = [r["problem"], nc_text(nc)]
            for engine in ENGINES:
                ex = (per.get(variant(engine, None)) or {}).get("median_seconds")
                ps = (per.get(variant(engine, nc)) or {}).get("median_seconds")
                speed = ex / ps if trusted and ex and ps else None
                cells += [_g(ex, ".4g"), _g(ps, ".4g") or _why(r, variant(engine, nc)),
                          _g(speed, ".2f")]
                if engine == "bngsim":
                    cells.append(_g(events.get(variant("bngsim", nc))))
            lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Per-trajectory medians of the interleaved ensembles (the full table "
              "above has their CIs and pre-sim costs); *speed-up* = exact / PSA of the "
              "same engine, derived here and shown only for a PASS row. *events ratio*: "
              "bngsim's median exact/scaled event count per trajectory, what the speed-up "
              "would be if an event cost the same under both."]
    return lines
