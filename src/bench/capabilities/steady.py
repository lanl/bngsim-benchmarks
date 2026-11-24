"""Steady state of BNGL models: BNGsim's nonlinear solver against reference engines.

The system under test is BNGsim's steady-state solve, ``Simulator.steady_state(
method="newton")``: a CVODE burst carries the state from the declared initial state
into the basin of the steady state it reaches, and KINSOL polishes the root there.
On one problem it is compared with every reference in ``REFERENCE_ENGINES``, each
reading the same generated network:

- **run_network** -- BioNetGen's CVODE binary, integrating from the initial state to
  the spec's horizon ``t_end`` (reads the ``.net``). Its run must have settled: over
  the second half of the horizon no species may move by more than a tenth of the
  verdict's tolerance. It is the **integration reference**: what the dynamics reach
  from the declared initial state, which is the question every job asks.
- **roadrunner** -- libRoadRunner's steady-state solver (NLEQ2, with conserved-moiety
  analysis and its presimulation), on SBML exported from the same ``.net``.
- **copasi** -- COPASI's steady-state task (Enhanced Newton, its defaults), through
  its own API, on the same SBML.
- **amici** -- AMICI's steady state by simulation (its ``integrationOnly`` mode, the
  one its defaults amount to), on the same SBML, compiled once.

and against two **engine-free checks**, computed by the harness from its own reading
of the ``.net`` (``bench.netfile``), with no engine's code involved:

- **residual**: at the returned state, each species' net rate of change is a
  vanishing part of its gross flux, ``|f_i(x)| <= residual_tol * g_i(x)`` (``g_i``
  floored at ``residual_floor`` of the largest), where ``g_i`` is what flows in plus
  what flows out -- a test in no units, so a model in molecule counts and one in
  molar units are held to the same standard;
- **conservation**: every conservation law of the network (``L S = 0``) has the
  total it has in the initial state, ``L x = L x0``. A conservation law makes the
  Jacobian singular, and a Newton step that is not confined to the laws can walk
  off them to a root the initial state can never reach.

**Points.** A protocol solves one point -- the model's parameters, with any the spec
sets in ``parameters`` -- or a dose-response ``scan`` of one parameter over N points.
Every point starts from the network's seed species at that point's parameters (BNG
``parameter_scan`` with ``reset_conc``): an initial amount that reads a scanned total
moves with it. Each engine is given the dose its own way (a rewritten ``.net`` for
run_network, ``set_param`` for BNGsim, the parameter for the SBML engines), and the
engine-free checks are computed with the network read at that point's parameters,
so an engine that did not take a dose fails them.

**Verdict.** Two states agree when, at every point and for every species that is not
a write-only accumulator, ``|a_i - b_i| <= rtol * max(|a_i|, |b_i|) + atol_i`` with
``rtol = 1e-4`` (bngsim's parity-core relative tolerance) and ``atol_i`` the larger
of ten times the protocol's integration ``atol`` and 1e-9 of the state's largest
species. The worst ratio of the two sides is reported; it is never averaged over
species or points, so landing on another branch of a bistable model -- which moves
some species by far more than ``rtol`` -- fails at that point. A reference is used
only if it answered the question: a reference whose state fails an engine-free
check, an integration reference that did not settle, or a solver reference that
returned a state the integration reference does not reach (a saddle, the other
branch) has failed, and is recorded with a ``<name>_error`` (``REFERENCE_FAILED``),
not scored. BNGsim passes when its state passes both engine-free checks and agrees
with every reference left, at every point. A disagreement is ``DIFF`` even when some
reference failed, so a wrong answer is never timed.

BNGsim's own integration path, ``steady_state(method="integration")``, is run and
checked the same way, and recorded beside the verdict without gating it: it is a
cost comparison (the default method), not the solver under test.

**Timing.** Every engine solves every point of the protocol in one timed repetition,
interleaved round-robin (``bench.timing``), and reports its absolute time with a
bootstrap CI, and the time per point. BNGsim's Newton solve, BNGsim's integration to
steady state, run_network's integration to the horizon, and the SBML engines'
steady-state solvers are timed together; the one-time cost -- BNGsim's load and
simulator build, libRoadRunner's SBML load, AMICI's compile, COPASI's import -- is
measured once, cold, beside it. A reference the correctness job recorded as failed
is not timed. Each repetition must reproduce the first, and each engine's answer is
checked against BNGsim's with the verdict and recorded beside its time, so a time
of a different state is labelled as one.

Failures are attributed as in ``ode``: each engine builds and solves in its own
forked child under a wall budget (``run_isolated``); a reference that fails or runs
out of time is ``REFERENCE_FAILED``; a BNGsim failure is ``EXCEPTION`` (``TIMEOUT``
when it ran out of time).

Input formats: BNGsim's steady state works on any model it loads -- ``.net``, BNGL
(through BNG2.pl), SBML and Antimony. This capability builds every job from BNGL
through the generated ``.net``; ``jobs`` is where an SBML suite would be admitted,
loading the SBML into BNGsim and the SBML engines directly and reaching run_network
and the engine-free checks through a converted ``.net``.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from .. import env, netfile
from ..jobs import CORRECTNESS, TIMING, Job
from ..specs import Problem
from ..timing import absolute_time_stats, round_robin_interleaved
from ._common import (
    engine_role,
    outcome,
    prepare_network,
    progress,
    read_dat,
    recorded_command,
    reproducibility,
    run_isolated,
    threads,
)
from .ode import run_network_command as _ode_run_network_command

CAPABILITY = "steady_state"
# Bump when this module's computation changes; every result it wrote goes stale.
HARNESS_VERSION = "1"
# The job id's setting: BNGsim's Newton solve (integration burst, KINSOL polish).
SETTING = "newton"

TIMING_PROTOCOL = {"warmup_rounds": 1, "min_rounds": 5, "max_rounds": 20, "budget_seconds": 120.0}
# A timing-only part of the key, as in ode.
REPRO_CHECK_VERSION = 1
# A repetition solves the same points from the same state, so it must return the
# same states; as a fraction of each species' largest value over the points.
REPRO_TOL = 1e-6
DEFAULT_N_STEPS = 20

# The verdict (see the module docstring). In the correctness definition, so a
# change restales every correctness result and nothing else.
VERDICT = {
    "agreement": "species by species at every point, |a-b| <= rtol*max(|a|,|b|) + atol/1",
    "rtol": 1e-4,
    # atol_i = max(atol_factor * protocol atol, peak_floor * the state's largest species)
    "atol_factor": 10.0,
    "peak_floor": 1e-9,
    # The integration reference's largest move over the second half of its horizon,
    # as a fraction of the agreement tolerance.
    "settled": 0.1,
    # |f_i| <= residual_tol * max(g_i, residual_floor * max_j g_j)
    "residual_tol": 1e-4,
    "residual_floor": 1e-6,
    # |L_k (x - x0)| <= conservation_tol * |L_k| max(|x|, |x0|)
    "conservation_tol": 1e-6,
}

# How each engine is asked for its steady state. In every definition, so changing
# one restales the results it produced.
BNGSIM_SETTINGS = {"method": "newton", "max_time": "t_end", "rtol": "protocol",
                   "atol": "protocol (exact)", "mask": "write-only accumulators excluded"}
REFERENCE_SETTINGS = {
    "run_network": {"method": "cvode (dense)", "horizon": "t_end", "grid": "n_steps"},
    "roadrunner": {"solver": "nleq2", "allow_presimulation": True,
                   "conserved_moiety_analysis": True, "integrator": "cvode, protocol tolerances"},
    "amici": {"mode": "integrationOnly", "tolerances": "protocol rtol/atol",
              "steady_state_tolerances": "protocol rtol/atol",
              "steady_state_tolerance_factor": 1.0},
    "copasi": {"task": "Steady-State", "method": "Enhanced Newton (COPASI's defaults)"},
}
COMPARATORS = ("bngsim_integration",)


def protocol(problem: Problem) -> dict[str, Any]:
    """The steady-state protocol with defaults filled in, as it enters a definition
    (without the planning keys, which no result depends on)."""
    p = {k: v for k, v in problem.protocol_for(CAPABILITY).items() if k not in PLANNING_KEYS}
    p.setdefault("n_steps", DEFAULT_N_STEPS)
    p["parameters"] = dict(p.get("parameters") or {})
    if p.get("scan") is not None:
        scan = dict(p["scan"])
        if "values" not in scan:
            scan.setdefault("log", False)
        p["scan"] = scan
    return p


def points(proto: dict[str, Any]) -> list[dict[str, float]]:
    """The protocol's points: each a mapping of the parameters it sets. One point
    without a scan; with one, a point per scanned value, in order, each also
    setting the protocol's ``parameters``."""
    base = {k: float(v) for k, v in (proto.get("parameters") or {}).items()}
    scan = proto.get("scan")
    if not scan:
        return [base]
    if "values" in scan:
        values = [float(v) for v in scan["values"]]
    elif scan.get("log"):
        values = np.logspace(math.log10(float(scan["min"])), math.log10(float(scan["max"])),
                             int(scan["points"])).tolist()
    else:
        values = np.linspace(float(scan["min"]), float(scan["max"]), int(scan["points"])).tolist()
    return [{**base, str(scan["parameter"]): float(v)} for v in values]


def engines_id() -> str:
    """The ``engines`` id segment: BNGsim versus every registered reference."""
    return "bngsim-vs-" + "-".join(REFERENCE_ENGINES)


def jobs(problem: Problem) -> list[Job]:
    # BNGL only for now. An SBML source would load directly into BNGsim and the
    # SBML engines, and reach run_network and the engine-free checks through a
    # converted .net: the place an SBML steady-state suite would be admitted.
    if problem.source.get("format") != "bngl":
        return []
    proto = protocol(problem)
    engines = engines_id()
    base = {
        "harness": f"{CAPABILITY}/{engines}/{HARNESS_VERSION}",
        "source_sha256": problem.source["sha256"],
        "protocol": proto,
        "setting": {"bngsim": BNGSIM_SETTINGS, "codegen": "auto"},
        # As in ode: the SBML references' versions key only this capability.
        "reference_versions": {
            "roadrunner": env.roadrunner_identity().get("version"),
            "copasi": env.copasi_identity().get("version"),
            "amici": env.amici_identity().get("version"),
        },
        "reference_settings": REFERENCE_SETTINGS,
    }
    est_c, est_t = planning_seconds(problem)
    backstop = (len(REFERENCE_ENGINES) + len(COMPARATORS) + 2) * engine_budget_seconds(problem)
    correctness = Job(
        CAPABILITY, problem.id, engines, SETTING, CORRECTNESS,
        definition={**base, "verdict": VERDICT},
        est_seconds=est_c,
        timeout_seconds=max(MIN_TIMEOUT_SECONDS, TIMEOUT_FACTOR * est_c, backstop),
    )
    timing = Job(
        CAPABILITY, problem.id, engines, SETTING, TIMING,
        definition={**base, "timing": TIMING_PROTOCOL, "timing_check": REPRO_CHECK_VERSION},
        est_seconds=est_t,
        timeout_seconds=max(MIN_TIMEOUT_SECONDS, TIMEOUT_FACTOR * est_t),
        requires=correctness.id,
    )
    return [correctness, timing]


# --------------------------------------------------------------------------- #
# Planning (never part of a freshness key)
# --------------------------------------------------------------------------- #
# A job builds every engine once -- dominated, as in ode, by AMICI's compile, which
# grows as reactions**1.5 -- and then solves every point with every engine. On the
# two pilot problems (9 and 2 species, x86) the compile is 21-26 s and a point costs
# all six engines together about 10 ms, run_network's process launch most of it;
# correctness took 26-27 s and timing 32-46 s. The per-point term grows like the
# build. Provisional until larger models have run.
MIN_TIMEOUT_SECONDS = 1800.0
TIMEOUT_FACTOR = 4.0
ENGINE_BUDGET_CAP_SECONDS = 3600.0
# A spec may give its own per-engine budget (``engine_budget_seconds`` in its
# protocol), in place of the estimate and above the cap: for a problem known to need
# more than an hour per engine (thousands of species at many points), which then
# needs an allocation long enough for every engine to use it. Planning only: it is
# left out of every definition, like the cap.
PLANNING_KEYS = ("engine_budget_seconds",)


def n_points(problem: Problem) -> int:
    return len(points(protocol(problem)))


def planning_seconds(problem: Problem) -> tuple[float, float]:
    """Rough wall-clock estimates (correctness, timing) for one problem."""
    reactions = problem.size.get("reactions") or 10 * problem.size.get("species", 100)
    build = 30.0 + 0.0021 * reactions ** 1.5
    per_point = 0.02 + 2e-4 * reactions ** 1.5
    n = n_points(problem)
    correctness = build + n * per_point
    tp = TIMING_PROTOCOL
    rnd = n * per_point
    measured = max(tp["min_rounds"] * rnd, min(tp["budget_seconds"] + rnd, tp["max_rounds"] * rnd))
    # + the cold builds, BNGsim's cold solve, the trial solves and the warm-up round.
    timing = 60.0 + build + (tp["warmup_rounds"] + 2) * rnd + measured
    return correctness, timing


def engine_budget_seconds(problem: Problem) -> float:
    """Wall budget for one engine's cold build + every point in the correctness job:
    the spec's own, if it gives one, else the estimate within the cap."""
    own = problem.protocol_for(CAPABILITY).get("engine_budget_seconds")
    if own is not None:
        return float(own)
    est_c, _ = planning_seconds(problem)
    return min(ENGINE_BUDGET_CAP_SECONDS, max(MIN_TIMEOUT_SECONDS, TIMEOUT_FACTOR * est_c))


# --------------------------------------------------------------------------- #
# The verdict and the engine-free checks
# --------------------------------------------------------------------------- #
def _bound(a: np.ndarray, b: np.ndarray, atol: float) -> np.ndarray:
    """The agreement tolerance per species of two states at one point."""
    scale = np.maximum(np.abs(a), np.abs(b))
    peak = float(np.max(scale)) if scale.size else 0.0
    floor = max(VERDICT["atol_factor"] * atol, VERDICT["peak_floor"] * peak)
    return VERDICT["rtol"] * scale + floor


def _ratios(a: np.ndarray, b: np.ndarray, atol: float) -> np.ndarray:
    """``|a - b|`` over the agreement tolerance, species by species; ``inf`` where
    either side is not finite."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    out = np.full(a.shape, np.inf)
    aa, bb = np.where(finite, a, 0.0), np.where(finite, b, 0.0)
    out[finite] = (np.abs(aa - bb) / _bound(aa, bb, atol))[finite]
    return out


def _keep(n_species: int, sinks: list[int]) -> np.ndarray:
    keep = np.ones(n_species, dtype=bool)
    keep[list(sinks)] = False
    return keep


def agreement(a: np.ndarray, b: np.ndarray, atol: float, species: list[str],
              sinks: list[int] = ()) -> dict[str, Any]:
    """Whether states ``a`` and ``b`` (``n_points x n_species``) agree at every
    point, species by species (write-only accumulators left out). ``max_ratio`` is
    the worst species' distance over its tolerance, so ``<= 1`` passes."""
    a, b = np.atleast_2d(np.asarray(a, dtype=float)), np.atleast_2d(np.asarray(b, dtype=float))
    if a.shape != b.shape or a.size == 0:
        return {"passed": False, "reason": f"shapes differ: {list(a.shape)} vs {list(b.shape)}",
                "max_ratio": math.inf, "worst_point": 0, "worst_species": None,
                "points_agreed": 0, "points": int(max(a.shape[0], b.shape[0]))}
    keep = _keep(a.shape[1], sinks)
    per_point = []
    worst = (0.0, 0, 0)
    for p in range(a.shape[0]):
        r = _ratios(a[p, keep], b[p, keep], atol)
        m = float(np.max(r)) if r.size else 0.0
        per_point.append(m)
        if m > worst[0]:
            worst = (m, p, int(np.flatnonzero(keep)[int(np.argmax(r))]))
    m, p, i = worst
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(a - b) / np.maximum(np.abs(a), np.abs(b))
    rel = rel[:, keep]
    return {
        "passed": bool(all(x <= 1.0 for x in per_point)),
        "max_ratio": m,
        "worst_point": p,
        "worst_species": species[i] if species else i,
        "points_agreed": int(sum(1 for x in per_point if x <= 1.0)),
        "points": len(per_point),
        "max_rel": float(np.nanmax(rel)) if rel.size and np.any(np.isfinite(rel)) else None,
        "per_point": per_point,
    }


def settledness(tail: np.ndarray, atol: float, sinks: list[int] = ()) -> float:
    """The largest move over the second half of an integration's output, as a
    fraction of the agreement tolerance at its final state (``tail``'s last row)."""
    tail = np.asarray(tail, dtype=float)
    keep = _keep(tail.shape[1], sinks)
    last = tail[-1, keep]
    return float(max((np.max(_ratios(row[keep], last, atol)) for row in tail[:-1]), default=0.0))


def engine_free(states: np.ndarray, prep: dict[str, Any]) -> dict[str, Any]:
    """The harness's own checks of states ``(n_points x n_species)``: the residual
    (net flux a vanishing part of gross flux, species by species) and the
    conservation of every conservation law's initial total, at every point."""
    states = np.atleast_2d(np.asarray(states, dtype=float))
    names = prep["species_names"]
    nets = prep.get("point_networks")
    if not nets:
        why = (prep.get("info") or {}).get("netfile_error", "the .net could not be read")
        return {"passed": True, "evaluated": False, "reason": why}
    if states.shape != (len(nets), len(names)):
        bad = {"passed": False, "evaluated": True, "worst_point": 0, "max": math.inf,
               "worst_species": None}
        return {"passed": False, "evaluated": True,
                "reason": f"states of shape {list(states.shape)}, expected "
                          f"{[len(nets), len(names)]}",
                "residual": dict(bad), "conservation": dict(bad, n_laws=int(prep["laws"].shape[0]))}
    sinks = prep["sinks"]
    keep = _keep(states.shape[1], sinks)
    out: dict[str, Any] = {"evaluated": True}
    # Residual.
    residual: dict[str, Any] = {"tol": VERDICT["residual_tol"]}
    try:
        worst = (0.0, 0, 0)
        per_point = []
        for p, (x, net) in enumerate(zip(states, nets)):
            if not np.all(np.isfinite(x)):
                per_point.append(math.inf)
                worst = max(worst, (math.inf, p, 0))
                continue
            f, g = net.flux(x)
            f, g = np.abs(f[keep]), g[keep]
            top = float(np.max(g)) if g.size else 0.0
            if top == 0.0:
                per_point.append(0.0)
                continue
            rho = f / np.maximum(g, VERDICT["residual_floor"] * top)
            m = float(np.max(rho))
            per_point.append(m)
            if m > worst[0]:
                worst = (m, p, int(np.flatnonzero(keep)[int(np.argmax(rho))]))
        residual.update(evaluated=True, max=worst[0], worst_point=worst[1],
                        worst_species=names[worst[2]] if names else worst[2],
                        passed=bool(worst[0] <= VERDICT["residual_tol"]), per_point=per_point)
    except (netfile.NetFileError, ValueError, ArithmeticError) as exc:
        residual.update(evaluated=False, passed=True, reason=f"{type(exc).__name__}: {exc}"[:300])
    out["residual"] = residual
    # Conservation.
    laws = prep["laws"]
    cons: dict[str, Any] = {"n_laws": int(laws.shape[0]), "tol": VERDICT["conservation_tol"]}
    if laws.shape[0]:
        worst = (0.0, 0)
        per_point = []
        for p, x in enumerate(states):
            x0 = prep["x0"][p]
            if not np.all(np.isfinite(x)):
                per_point.append(math.inf)
                worst = max(worst, (math.inf, p))
                continue
            sub, sub0 = x[prep["law_species"]], x0[prep["law_species"]]
            d = np.abs(laws @ (sub - sub0))
            scale = np.abs(laws) @ np.maximum(np.abs(sub), np.abs(sub0))
            with np.errstate(divide="ignore", invalid="ignore"):
                defect = np.where(scale > 0, d / scale, np.where(d > 0, np.inf, 0.0))
            m = float(np.max(defect))
            per_point.append(m)
            worst = max(worst, (m, p))
        cons.update(max=worst[0], worst_point=worst[1],
                    passed=bool(worst[0] <= VERDICT["conservation_tol"]), per_point=per_point)
    else:
        cons.update(max=0.0, passed=True)
    out["conservation"] = cons
    out["passed"] = bool(residual["passed"] and cons["passed"])
    return out


# Largest network whose roots are classified (a dense Jacobian by differences of the
# harness's own right-hand side, one evaluation per species).
STABILITY_MAX_SPECIES = 200


def stability(x: np.ndarray, p: int, prep: dict[str, Any]) -> dict[str, Any]:
    """Whether the steady state ``x`` at point ``p`` is one the system can rest at:
    the eigenvalues of the harness's own Jacobian restricted to the directions the
    reactions move the state in (so a conservation law's zero eigenvalue is left
    out). ``stable`` when every real part is negative, ``unstable`` (a saddle, or a
    source) when one is positive, ``undetermined`` within the differences' noise
    or where the check does not apply. For a disagreement's record, not a verdict."""
    nets, basis = prep.get("point_networks"), prep.get("moving_basis")
    x = np.asarray(x, dtype=float)
    if (not nets or basis is None or prep["sinks"] or x.size > STABILITY_MAX_SPECIES
            or not np.all(np.isfinite(x))):
        return {"stability": "undetermined", "reason": "not classified for this network"}
    if basis.shape[1] == 0:
        return {"stability": "undetermined", "reason": "no reaction moves the state"}
    try:
        jac = nets[p].jacobian(x)
    except (netfile.NetFileError, ValueError, ArithmeticError) as exc:
        return {"stability": "undetermined", "reason": f"{type(exc).__name__}: {exc}"[:200]}
    eig = np.linalg.eigvals(basis.T @ jac @ basis)
    if not np.all(np.isfinite(eig)):
        return {"stability": "undetermined", "reason": "non-finite Jacobian"}
    top = float(np.max(eig.real))
    scale = float(np.max(np.abs(eig))) if eig.size else 0.0
    if scale == 0.0 or abs(top) <= 1e-6 * scale:
        kind = "undetermined"
    else:
        kind = "stable" if top < 0 else "unstable"
    return {"stability": kind, "max_real_eigenvalue": top, "spectral_radius": scale}


def _root_words(st: dict[str, Any]) -> str:
    return {"stable": "a stable steady state", "unstable": "an unstable steady state (a saddle "
            "or a source, where nothing can rest)"}.get(st.get("stability"), "a state")


def _engine_free_error(check: dict[str, Any], prep: dict[str, Any]) -> str | None:
    """Why a reference's states fail the engine-free checks; None when they pass."""
    if check.get("passed", True):
        return None
    if check.get("reason"):
        return f"INVALID_STATES: {check['reason']}"
    pts = prep["points"]
    res, cons = check.get("residual") or {}, check.get("conservation") or {}
    if not cons.get("passed", True):
        p = cons["worst_point"]
        return (f"OFF_MANIFOLD: at point {p} ({_point_label(pts[p])}) a conservation law's "
                f"total moved by {cons['max']:.3g} of itself (tolerance "
                f"{VERDICT['conservation_tol']:g}): a state the initial state cannot reach")
    p = res["worst_point"]
    return (f"NOT_A_STEADY_STATE: at point {p} ({_point_label(pts[p])}) {res['worst_species']} "
            f"has a net rate {res['max']:.3g} of its gross flux (tolerance "
            f"{VERDICT['residual_tol']:g})")


def _point_label(point: dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.6g}" for k, v in point.items()) or "the model's parameters"


def _summary(check: dict[str, Any]) -> dict[str, Any]:
    """An engine-free check or an agreement without its per-point lists."""
    out = {}
    for k, v in check.items():
        if k == "per_point":
            continue
        out[k] = _summary(v) if isinstance(v, dict) else v
    return out


# --------------------------------------------------------------------------- #
# The system under test and the reference engines
# --------------------------------------------------------------------------- #
class BngsimSteady:
    """BNGsim's steady state on the ``.net``: one loaded model, every point solved
    after ``set_param`` and ``reset`` -- the declared initial state at that point.

    ``method="newton"`` is the solver under test; ``"integration"`` is BNGsim's
    integration to steady state, a cost comparator. Both are held to the protocol's
    integration tolerances (``atol`` exactly, as run_network holds it) and given the
    protocol's horizon as ``max_time``. A write-only accumulator, which has no
    steady value, is left out of the convergence test with ``mask`` (BNGsim's own
    recipe); a solve BNGsim reports as not converged is a failure.
    """

    NAME = "bngsim"

    def __init__(self, net: Path, proto: dict[str, Any], pts: list[dict[str, float]],
                 method: str, keep: list[str] | None = None):
        import bngsim

        self.proto, self.points, self.method = proto, pts, method
        t0 = time.perf_counter()
        self.model = bngsim.Model.from_net(str(net))
        self.load_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        self.sim = bngsim.Simulator(self.model, method="ode")
        self.codegen_seconds = time.perf_counter() - t1
        self.atol = bngsim.ExactAtol(float(proto["atol"]))
        self.mask = keep
        self.diagnostics: list[dict[str, Any]] = []

    def solve(self, point: dict[str, float]):
        for name, value in point.items():
            self.model.set_param(name, float(value))
        self.model.reset()
        p = self.proto
        return self.sim.steady_state(method=self.method, max_time=float(p["t_end"]),
                                     rtol=float(p["rtol"]), atol=self.atol, mask=self.mask)

    def solve_all(self, detailed: bool = False) -> np.ndarray:
        out = []
        diags = []
        for i, point in enumerate(self.points):
            r = self.solve(point)
            x = np.array(r.concentrations, dtype=float)
            if not r.converged:
                raise RuntimeError(
                    f"steady_state(method={self.method!r}) did not converge at point {i} "
                    f"({_point_label(point)}): residual {float(r.residual):.3g}")
            if not np.all(np.isfinite(x)):
                raise RuntimeError(f"steady_state(method={self.method!r}) returned non-finite "
                                   f"values at point {i}")
            out.append(x)
            if detailed:
                diags.append({
                    "method_used": str(r.method_used), "residual": float(r.residual),
                    "n_steps": int(r.n_steps), "n_rhs_evals": int(r.n_rhs_evals),
                    "root_stability": str(r.root_stability),
                    "n_unstable_roots_rejected": int(r.n_unstable_roots_rejected),
                    "rhs_backend": str(r.rhs_backend),
                    "jacobian": str(r.solver_jacobian_source),
                    "linear_solver": str(r.linear_solver),
                })
        if detailed:
            self.diagnostics = diags
        return np.array(out)

    def config(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "method": self.method,
            "codegen_backend": self.sim.codegen_backend,
            "load_seconds": self.load_seconds,
            "codegen_seconds": self.codegen_seconds,
            "mask": None if self.mask is None else f"{len(self.mask)} species tested",
        }
        if self.diagnostics:
            d = self.diagnostics
            out["points"] = d
            out["methods_used"] = sorted({x["method_used"] for x in d})
            out["root_stability"] = sorted({x["root_stability"] for x in d})
        return out


class RunNetworkSteady:
    """run_network as the integration reference: a fresh process per point, each
    integrating the point's ``.net`` (its parameters rewritten) from the seed
    species to ``t_end``, on ``n_steps`` output intervals. The final state is the
    answer; the second half of the output says whether it had settled."""

    NAME = "run_network"
    INPUT = "net"

    def __init__(self, nets: list[Path], proto: dict[str, Any], out_prefix: Path, *,
                 timeout: float, sinks: list[int] = ()):
        self.nets, self.proto, self.out_prefix, self.timeout = nets, proto, out_prefix, timeout
        self.sinks = list(sinks)
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        self.settled: list[float] = []

    def _command(self, net: Path) -> list[str]:
        p = self.proto
        return _ode_run_network_command(
            net, {"t_start": 0, "t_end": p["t_end"], "n_steps": p["n_steps"], "rtol": p["rtol"],
                  "atol": p["atol"], "sparse": False}, self.out_prefix)

    def solve_point(self, i: int) -> np.ndarray:
        cmd = self._command(self.nets[i])
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout,
                              check=False)
        if proc.returncode != 0:
            tail = (proc.stderr.strip() or proc.stdout.strip()).splitlines()
            last = (tail[-1] if tail else "")[:300]
            raise RuntimeError(f"run_network failed at point {i}: {last}")
        cdat = Path(f"{self.out_prefix}.cdat")
        if not cdat.exists():
            raise RuntimeError(f"run_network produced no .cdat at point {i}")
        _, values, _ = read_dat(cdat)
        cdat.unlink()
        n = int(self.proto["n_steps"])
        if values.shape[0] != n + 1:
            raise RuntimeError(f"run_network wrote {values.shape[0]} time points at point {i}, "
                               f"expected {n + 1} (it stopped early)")
        return values

    def solve_all(self, detailed: bool = False) -> np.ndarray:
        finals, settled = [], []
        half = int(self.proto["n_steps"]) // 2
        for i in range(len(self.nets)):
            values = self.solve_point(i)
            finals.append(values[-1])
            if detailed:
                settled.append(settledness(values[half:], float(self.proto["atol"]), self.sinks))
        if detailed:
            self.settled = settled
        return np.array(finals)

    def config(self) -> dict[str, Any]:
        cmd = recorded_command(self._command(self.nets[0]))
        names = {str(n): "<point>.net" for n in self.nets} | {str(self.out_prefix): "<out>"}
        out: dict[str, Any] = {"method": "cvode (dense), integrated to t_end",
                               "command": [names.get(c, c) for c in cmd]}
        if self.settled:
            out["settled"] = {"per_point": self.settled, "max": max(self.settled),
                              "tol": VERDICT["settled"]}
        return out


def _align(species_names: list[str], name_to_id: dict[str, str]) -> list[str]:
    missing = [n for n in species_names if n not in name_to_id]
    if missing:
        raise RuntimeError(
            f"cannot align {len(missing)} BNGsim species to SBML ids "
            f"(e.g. {missing[:3]}); export/name map is wrong")
    return [name_to_id[n] for n in species_names]


class RoadRunnerSteady:
    """libRoadRunner's steady-state solver, on the SBML exported from the ``.net``.

    NLEQ2 with conserved-moiety analysis (which removes the singular directions a
    conservation law puts in the Jacobian) and presimulation allowed: libRoadRunner
    simulates on from the initial state before its Newton solve, following its
    ``presimulation_times`` program (0.1 to 10^4 by default; recorded in the
    config). Without it, NLEQ2 from the toggle switch's initial state stops on
    "possible negative concentrations" at each dose tried. Each point sets its
    parameters, ``reset()``s -- which rebuilds an initial amount that reads one --
    and calls ``steadyState()``; species are read as amounts by SBML id in BNGsim's
    column order.
    """

    NAME = "roadrunner"
    INPUT = "sbml"

    def __init__(self, sbml: Path, species_names: list[str], name_to_id: dict[str, str],
                 proto: dict[str, Any], pts: list[dict[str, float]]):
        roadrunner = env.import_roadrunner()

        self.ids = _align(species_names, name_to_id)
        self.proto, self.points = proto, pts
        t0 = time.perf_counter()
        self.rr = roadrunner.RoadRunner(str(sbml))
        self.rr.conservedMoietyAnalysis = True
        self.load_seconds = time.perf_counter() - t0
        integrator = self.rr.getIntegrator()
        integrator.setValue("relative_tolerance", float(proto["rtol"]))
        integrator.setValue("absolute_tolerance", float(proto["atol"]))
        self.solver = self.rr.getSteadyStateSolver()
        self.solver.setValue("allow_presimulation", True)

    def solve_all(self, detailed: bool = False) -> np.ndarray:
        out = []
        for i, point in enumerate(self.points):
            for name, value in point.items():
                self.rr[name] = float(value)
            self.rr.reset()
            try:
                self.rr.steadyState()
            except Exception as exc:  # noqa: BLE001 - its message says why
                raise RuntimeError(f"steadyState() failed at point {i}: {exc}") from exc
            out.append([self.rr[s] for s in self.ids])
        return np.array(out, dtype=float)

    def config(self) -> dict[str, Any]:
        settings = {k: self.solver.getValue(k) for k in
                    ("allow_presimulation", "presimulation_times", "relative_tolerance",
                     "maximum_iterations")}
        return {"solver": self.solver.getName(), "settings": json.loads(json.dumps(settings,
                                                                                    default=str)),
                "conserved_moiety_analysis": True,
                "version": env.roadrunner_identity().get("version"),
                "output": "amount", "load_seconds": self.load_seconds}


class AmiciSteady:
    """AMICI's steady state by simulation, on the SBML exported from the ``.net``.

    The model is compiled once (``codegen_seconds``), into the job's node-local
    scratch. The steady state is asked for as an output at ``t = inf`` with the
    ``integrationOnly`` computation mode: AMICI integrates until its weighted
    steady-state test passes, held here to the protocol's tolerances with a factor
    of 1 (its default factor, 100, stops a hundred times looser). That is the
    computation AMICI's defaults amount to -- its Newton solver is off until
    ``newton_max_steps`` is set -- and the one that answers which state the initial
    state reaches: AMICI 1.1.0 removes a conservation law of non-constant species
    only experimentally, so its Newton step from the initial state faces the
    singular Jacobian of the law, and can leave it. A fresh solver per solve, as in
    ``ode``. Parameters are set by id (``amici_`` prefixed where AMICI renamed one),
    and an initial amount that reads one follows it.
    """

    NAME = "amici"
    INPUT = "sbml"

    def __init__(self, sbml: Path, species_names: list[str], name_to_id: dict[str, str],
                 proto: dict[str, Any], pts: list[dict[str, float]]):
        import amici
        from amici.sim import sundials as asun

        want = _align(species_names, name_to_id)
        self.proto, self.points = proto, pts
        outdir = sbml.parent / "amici_steady"
        t0 = time.perf_counter()
        amici.SbmlImporter(str(sbml)).sbml2amici(
            "bench_steady", str(outdir), generate_sensitivity_code=False, verbose=False)
        self.codegen_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        self.model = amici.import_model_module("bench_steady", str(outdir)).get_model()
        self.model.set_timepoints([np.inf])
        self.model.set_steady_state_computation_mode(
            asun.SteadyStateComputationMode_integrationOnly)
        self.load_seconds = time.perf_counter() - t1
        pos = {sid: i for i, sid in enumerate(self.model.get_state_ids())}
        unmapped = [i for i in want if i not in pos]
        if unmapped:
            raise RuntimeError(
                f"cannot align {len(unmapped)} SBML ids to AMICI state ids "
                f"(e.g. {unmapped[:3]}); the SBML export and AMICI disagree")
        self.idx = [pos[i] for i in want]
        free = set(self.model.get_free_parameter_ids())
        fixed = set(self.model.get_fixed_parameter_ids())
        self.setters: dict[str, Any] = {}
        for name in (pts[0] if pts else {}):
            pid = next((c for c in (name, f"amici_{name}") if c in free | fixed), None)
            if pid is None:
                raise RuntimeError(f"AMICI exposes no parameter {name!r}")
            setter = (self.model.set_free_parameter_by_id if pid in free
                      else self.model.set_fixed_parameter_by_id)
            self.setters[name] = (setter, pid)

    def _solver(self):
        from amici.sim.sundials import LinearMultistepMethod, SensitivityOrder

        p = self.proto
        solver = self.model.create_solver()
        solver.set_relative_tolerance(float(p["rtol"]))
        solver.set_absolute_tolerance(float(p["atol"]))
        solver.set_relative_tolerance_steady_state(float(p["rtol"]))
        solver.set_absolute_tolerance_steady_state(float(p["atol"]))
        solver.set_steady_state_tolerance_factor(1.0)
        solver.set_sensitivity_order(SensitivityOrder.none)
        solver.set_linear_multistep_method(LinearMultistepMethod.BDF)
        return solver

    def solve_all(self, detailed: bool = False) -> np.ndarray:
        from amici.sim.sundials import run_simulation

        out = []
        for i, point in enumerate(self.points):
            for name, value in point.items():
                setter, pid = self.setters[name]
                setter(pid, float(value))
            rdata = run_simulation(self.model, self._solver())
            if rdata.status != 0:
                raise RuntimeError(f"AMICI steady state failed at point {i} with status "
                                   f"{rdata.status} (posteq_status "
                                   f"{[int(s) for s in rdata.posteq_status]})")
            x = np.asarray(rdata.x, dtype=float)[-1][self.idx]
            if not np.all(np.isfinite(x)):
                raise RuntimeError(f"AMICI returned non-finite values with a success status "
                                   f"at point {i}")
            out.append(x)
        return np.array(out)

    def config(self) -> dict[str, Any]:
        return {"mode": "integrationOnly (simulation to steady state)",
                "version": env.amici_identity().get("version"),
                "output": "amount", "load_seconds": self.load_seconds,
                "codegen_seconds": self.codegen_seconds}


class CopasiSteady:
    """COPASI's steady-state task, on the exported SBML, driven through COPASI's own
    API (BASICO only imports the model, so a solve is COPASI's work and not a pandas
    frame). Each point sets the parameters' initial values and updates the initial
    values that depend on them, then runs the task from the initial state with
    COPASI's default method (Enhanced Newton: Newton, with forward integration when
    it fails). A result COPASI does not call a steady state, or one it reports with
    negative concentrations, is a failure. Concentrations are read by SBML id; the
    exported SBML has a unit compartment, so they equal BNGsim's amounts."""

    NAME = "copasi"
    INPUT = "sbml"
    # CSteadyStateMethod's result codes: notFound, found, foundEquilibrium, foundNegative.
    RESULTS = {0: "not found", 1: "found", 2: "found (equilibrium)",
               3: "found with negative concentrations"}

    def __init__(self, sbml: Path, species_names: list[str], name_to_id: dict[str, str],
                 proto: dict[str, Any], pts: list[dict[str, float]]):
        import basico
        import COPASI

        self._copasi = COPASI
        ids = _align(species_names, name_to_id)
        self.proto, self.points = proto, pts
        t0 = time.perf_counter()
        self.dm = basico.load_model(str(sbml))
        if self.dm is None:
            raise RuntimeError("COPASI could not import the exported SBML")
        self.model = self.dm.getModel()
        metabs = self.model.getMetabolites()
        by_id = {metabs.get(i).getSBMLId(): metabs.get(i) for i in range(metabs.size())}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise RuntimeError(f"COPASI's model lacks species {missing[:3]}")
        self.metabs = [by_id[i] for i in ids]
        values = self.model.getModelValues()
        params = {values.get(i).getSBMLId(): values.get(i) for i in range(values.size())}
        self.params = {}
        for name in (pts[0] if pts else {}):
            if name not in params:
                raise RuntimeError(f"COPASI's model has no parameter {name!r}")
            self.params[name] = params[name]
        self.task = self.dm.getTask("Steady-State")
        self.settings = basico.get_task_settings("Steady-State", model=self.dm).get("method", {})
        self.load_seconds = time.perf_counter() - t0

    def solve_all(self, detailed: bool = False) -> np.ndarray:
        cp = self._copasi
        out = []
        for i, point in enumerate(self.points):
            for name, value in point.items():
                param = self.params[name]
                param.setInitialValue(float(value))
                self.model.updateInitialValues(param.getInitialValueReference())
            before = cp.CCopasiMessage.size()
            if not self.task.initializeRaw(cp.CCopasiTask.OUTPUT_UI):
                msgs = cp.CCopasiMessage.getAllMessageText() \
                    if cp.CCopasiMessage.size() > before else ""
                raise RuntimeError(f"COPASI steady-state task initialisation failed at point "
                                   f"{i}: {msgs.strip()[:300]}")
            self.task.processRaw(True)
            result = int(self.task.getResult())
            if result not in (1, 2):
                raise RuntimeError(f"COPASI steady state {self.RESULTS.get(result, result)} "
                                   f"at point {i}")
            out.append([m.getConcentration() for m in self.metabs])
        return np.array(out, dtype=float)

    def config(self) -> dict[str, Any]:
        keep = ("name", "Resolution", "Use Newton", "Use Integration", "Use Back Integration",
                "Iteration Limit", "Target Criterion")
        return {"task": "Steady-State",
                "method": {k: self.settings.get(k) for k in keep if k in self.settings},
                "version": env.copasi_identity().get("version"),
                "output": "amount", "load_seconds": self.load_seconds}


# The reference engines BNGsim is compared against, in a fixed order. The first is
# the integration reference, which decides which state the initial state reaches.
# The order is ode's, so the two capabilities' job ids read alike.
REFERENCE_ENGINES: dict[str, type] = {
    RunNetworkSteady.NAME: RunNetworkSteady,
    RoadRunnerSteady.NAME: RoadRunnerSteady,
    CopasiSteady.NAME: CopasiSteady,
    AmiciSteady.NAME: AmiciSteady,
}
INTEGRATION_REFERENCE = RunNetworkSteady.NAME

BUILD_STAGE = {
    "bngsim": "load + simulator build",
    "bngsim_integration": "load + simulator build",
    "run_network": "integration (subprocesses)",
    "roadrunner": "SBML load (LLVM JIT) + moiety analysis",
    "amici": "SBML import + C++ compile",
    "copasi": "SBML import",
}


def _build_reference(name: str, prep: dict[str, Any], *, out_prefix: Path, timeout: float):
    cls = REFERENCE_ENGINES[name]
    if cls.INPUT == "net":
        return cls(prep["point_nets"], prep["proto"], out_prefix, timeout=timeout,
                   sinks=prep["sinks"])
    return cls(prep["sbml"], prep["species_names"], prep["name_to_id"], prep["proto"],
               prep["points"])


def _build_bngsim(prep: dict[str, Any], method: str) -> BngsimSteady:
    keep = None
    if prep["sinks"]:
        keep = [n for i, n in enumerate(prep["species_names"]) if i not in set(prep["sinks"])]
    return BngsimSteady(prep["net"], prep["proto"], prep["points"], method, keep)


def _progress(msg: str) -> None:
    progress(CAPABILITY, msg)


# --------------------------------------------------------------------------- #
# Preparation
# --------------------------------------------------------------------------- #
def network_checks(text: str, pts: list[dict[str, float]],
                   species_names: list[str]) -> dict[str, Any]:
    """What the engine-free checks need from the ``.net`` text: the network read at
    each point's parameters (its right-hand side and initial state there), the
    write-only accumulators, the conservation laws over the other species, and the
    directions the reactions move the state in.

    Raises ValueError (a bad test) when a point sets a parameter the network does
    not have, or one written as an expression of others. A ``.net`` the reader
    cannot read leaves the checks out (``netfile_error``), not the job."""
    try:
        network = netfile.parse(text)
    except (netfile.NetFileError, ValueError, ArithmeticError) as exc:
        return {"point_networks": None, "sinks": [], "laws": np.zeros((0, 0)),
                "law_species": [], "moving_basis": None, "x0": None,
                "netfile_error": f"{type(exc).__name__}: {exc}"[:400]}
    for name in pts[0]:
        if name not in network.parameters:
            raise ValueError(f"the network has no parameter {name!r} to set")
        if not network.is_primary(name):
            raise ValueError(f"parameter {name!r} is an expression of others in the .net "
                             f"({network.parameter_exprs[name]!r}); set a parameter it "
                             "reads instead")
    nets = [netfile.parse(text, pt) for pt in pts]
    sinks = network.pure_sinks()
    # Laws over the species that have a steady value: a write-only accumulator grows
    # for as long as anything feeds it, so a law through it holds only if the
    # accumulator is integrated, which a masked solve need not do.
    sink_set = set(sinks)
    law_species = [i for i in range(len(network.species)) if i not in sink_set]
    at = {i: k for k, i in enumerate(law_species)}
    sub = netfile.Network(species=[network.species[i] for i in law_species],
                          fixed=[network.fixed[i] for i in law_species],
                          reactions=[netfile.Reaction(
                              r.index, tuple(at[i] for i in r.reactants if i in at),
                              tuple(at[i] for i in r.products if i in at), r.rate)
                              for r in network.reactions])
    out: dict[str, Any] = {
        "point_networks": nets,
        "x0": np.array([n.initial for n in nets], dtype=float),
        "sinks": sinks,
        "law_species": law_species,
        "laws": sub.conservation_laws(),
        "moving_basis": network.subspaces()[1],
    }
    if sinks:
        out["pure_sinks"] = [species_names[i] for i in sinks]
    return out


def _prepare(problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    """Build the network and the SBML once, read the ``.net`` (once per point, at
    that point's parameters), and write each point's ``.net`` for run_network."""
    proto = protocol(problem)
    prep = prepare_network(problem, ctx, proto["netgen"])
    prep["proto"] = proto
    pts = points(proto)
    prep["points"] = pts
    info = prep["info"]
    info["points"] = pts
    net: Path = prep["net"]
    text = net.read_text(encoding="utf-8")
    checks = network_checks(text, pts, prep["species_names"])
    for key in ("netfile_error", "pure_sinks"):
        if key in checks:
            info[key] = checks.pop(key)
    prep.update(checks)
    if prep["point_networks"] is not None:
        info["conservation_laws"] = int(prep["laws"].shape[0])
    # run_network reads a file: one per point, with the point's parameters written in.
    if pts == [{}]:
        prep["point_nets"] = [net]
    else:
        d = Path(ctx["workdir"]) / "points"
        d.mkdir(parents=True, exist_ok=True)
        prep["point_nets"] = []
        for i, pt in enumerate(pts):
            path = d / f"point_{i:04d}.net"
            try:
                path.write_text(netfile.with_parameters(text, pt), encoding="utf-8")
            except netfile.NetFileError as exc:
                raise ValueError(str(exc)) from exc
            prep["point_nets"].append(path)
    return prep


# --------------------------------------------------------------------------- #
# Correctness
# --------------------------------------------------------------------------- #
def _cold_bngsim(prep: dict[str, Any], method: str):
    def fn(stage):
        t0 = time.perf_counter()
        stage(BUILD_STAGE["bngsim"])
        engine = _build_bngsim(prep, method)
        stage(f"steady state at {len(prep['points'])} point(s)")
        states = engine.solve_all(detailed=True)
        return states, {**engine.config(), "cold_seconds": time.perf_counter() - t0}
    return fn


def _cold_reference(name: str, prep: dict[str, Any], out_prefix: Path, timeout: float):
    def fn(stage):
        t0 = time.perf_counter()
        stage(BUILD_STAGE.get(name, "build"))
        ref = _build_reference(name, prep, out_prefix=out_prefix, timeout=timeout)
        stage(f"steady state at {len(prep['points'])} point(s)")
        states = ref.solve_all(detailed=True)
        return states, {**ref.config(), "cold_seconds": time.perf_counter() - t0}
    return fn


def _states_record(states: dict[str, np.ndarray], prep: dict[str, Any]) -> dict[str, Any]:
    """Every engine's states, by species name, when they are small enough to keep
    (BNGsim's always: the dose-response itself)."""
    names = prep["species_names"]
    out = {}
    for engine, x in states.items():
        if engine != "bngsim" and x.size > 20_000:
            continue
        out[engine] = {"species": names, "values": np.asarray(x).tolist()}
    return out


def run_correctness(job: Job, problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    try:
        prep = _prepare(problem, ctx)
    except Exception as exc:  # noqa: BLE001 - a bad test input, a labelled row
        return {"outcome": "BAD_TEST", "error": f"{type(exc).__name__}: {exc}"}
    proto, info = prep["proto"], prep["info"]
    atol = float(proto["atol"])
    names, sinks = prep["species_names"], prep["sinks"]
    workdir = Path(ctx["workdir"])
    budget = float(ctx.get("engine_budget_seconds") or engine_budget_seconds(problem))
    info["engine_budget_seconds"] = budget
    timeout = min(float(ctx["subprocess_timeout"]), budget)

    def isolated(name: str, fn) -> dict[str, Any]:
        _progress(f"{name}: start (budget {budget:g} s; {threads()} threads in the worker)")
        out = run_isolated(fn, budget, workdir / "isolated" / name)
        _progress(f"{name}: {out.get('error') or 'done'} after {out['seconds']:.1f} s")
        return out

    states: dict[str, np.ndarray] = {}
    bn_error = None
    out = isolated("bngsim", _cold_bngsim(prep, "newton"))
    if "error" in out:
        bn_error = out["error"]
    else:
        info["bngsim"] = out["config"]
        states["bngsim"] = out["trajectory"]

    # BNGsim's integration path: checked and recorded, not scored.
    out = isolated("bngsim_integration", _cold_bngsim(prep, "integration"))
    comparator: dict[str, Any] = {}
    if "error" in out:
        comparator["error"] = out["error"]
    else:
        comparator["config"] = out["config"]
        states["bngsim_integration"] = out["trajectory"]

    references: dict[str, Any] = {}
    ref_errors: dict[str, str] = {}
    for name in REFERENCE_ENGINES:
        out = isolated(name, _cold_reference(name, prep, workdir / name / "rn", timeout))
        if "error" in out:
            ref_errors[name] = out["error"]
        else:
            references[name] = out["config"]
            states[name] = out["trajectory"]
    info["references"] = references

    # Every engine's states against the engine-free checks.
    checks = {engine: engine_free(x, prep) for engine, x in states.items()}

    # A reference is used only if it answered the question (see the docstring).
    valid: dict[str, np.ndarray] = {}
    reference_checks: dict[str, Any] = {}
    for name in REFERENCE_ENGINES:
        if name not in states:
            continue
        reference_checks[name] = {"engine_free": _summary(checks[name])}
        err = None
        settled = (references[name].get("settled") or {})
        if settled and settled.get("max", 0.0) > VERDICT["settled"]:
            p = int(np.argmax(settled["per_point"]))
            err = (f"NOT_SETTLED: at point {p} ({_point_label(prep['points'][p])}) the "
                   f"integration still moved by {settled['max']:.3g} of the agreement "
                   f"tolerance over the second half of t_end = {proto['t_end']:g} (at most "
                   f"{VERDICT['settled']:g}); the horizon is too short for this point")
        err = err or _engine_free_error(checks[name], prep)
        if err:
            ref_errors[name] = err
        else:
            valid[name] = states[name]
    branch = valid.get(INTEGRATION_REFERENCE)
    if branch is not None:
        for name in list(valid):
            if name == INTEGRATION_REFERENCE:
                continue
            v = agreement(valid[name], branch, atol, names, sinks)
            reference_checks[name]["vs_integration"] = _summary(v)
            if not v["passed"]:
                p = v["worst_point"]
                st = stability(valid[name][p], p, prep)
                reference_checks[name]["root"] = {"point": p, **st}
                ref_errors[name] = (
                    f"OTHER_STATE: at point {p} ({_point_label(prep['points'][p])}) it returned "
                    f"{_root_words(st)} that the integration from the initial state does not "
                    f"reach ({v['worst_species']} off by {v['max_ratio']:.3g} of the tolerance; "
                    f"{v['points_agreed']}/{v['points']} points agree)")
                del valid[name]
    info["reference_checks"] = reference_checks

    verdicts: dict[str, Any] = {}
    if "bngsim" in states:
        bn = states["bngsim"]
        for name, x in valid.items():
            verdicts[name] = agreement(bn, x, atol, names, sinks)
            if not verdicts[name]["passed"]:
                # Which side of a disagreement can the system rest at?
                p = verdicts[name]["worst_point"]
                verdicts[name]["stability"] = {"point": p, "bngsim": stability(bn[p], p, prep),
                                               name: stability(x[p], p, prep)}
        verdicts["engine_free"] = checks["bngsim"]
    if "bngsim_integration" in states:
        comparator["engine_free"] = _summary(checks["bngsim_integration"])
        if "bngsim" in states:
            comparator["vs_bngsim"] = _summary(agreement(states["bngsim_integration"],
                                                         states["bngsim"], atol, names, sinks))
        if branch is not None:
            comparator["vs_integration"] = _summary(agreement(states["bngsim_integration"],
                                                              branch, atol, names, sinks))
    info["bngsim_integration"] = comparator
    info["states"] = _states_record(states, prep)

    passed = bool(verdicts) and all(v.get("passed") for v in verdicts.values())
    if bn_error:
        result = outcome(bn_error, ref_errors, len(REFERENCE_ENGINES), False)
        if bn_error.startswith("TIMEOUT") and result == "EXCEPTION":
            result = "TIMEOUT"
    elif not passed:
        result = "DIFF"  # a wrong answer, whatever else failed
    else:
        result = outcome(None, ref_errors, len(REFERENCE_ENGINES), True)
    rec: dict[str, Any] = {"outcome": result, "measurements": info}
    if verdicts:
        rec["verdicts"] = verdicts
    if bn_error:
        rec["bngsim_error"] = bn_error
    for name, err in ref_errors.items():
        rec[f"{name}_error"] = err
    return rec


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def _correctness_errors(job: Job, ctx: dict[str, Any]) -> dict[str, str]:
    """The references this timing job's correctness job recorded as failed."""
    from ..results import result_path

    requires, results_dir = getattr(job, "requires", None), ctx.get("results_dir")
    if not requires or not results_dir:
        return {}
    try:
        rec = json.loads(result_path(results_dir, requires).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {name: str(rec[f"{name}_error"]) for name in REFERENCE_ENGINES
            if rec.get(f"{name}_error")}


def run_timing(job: Job, problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    try:
        prep = _prepare(problem, ctx)
    except Exception as exc:  # noqa: BLE001 - a bad test input, a labelled row
        return {"outcome": "BAD_TEST", "error": f"{type(exc).__name__}: {exc}"}
    proto, info = prep["proto"], prep["info"]
    names, sinks = prep["species_names"], prep["sinks"]
    atol = float(proto["atol"])
    workdir = Path(ctx["workdir"])
    timeout = ctx["subprocess_timeout"]
    n = len(prep["points"])

    # BNGsim cold: load, simulator build and a first solve of every point, with the
    # empty compile cache the worker gave this job. Its states anchor the rest.
    t0 = time.perf_counter()
    bngsim_engine = _build_bngsim(prep, "newton")
    anchor = bngsim_engine.solve_all()
    info["bngsim_cold_seconds"] = time.perf_counter() - t0
    info["bngsim"] = bngsim_engine.config()

    engines: dict[str, Any] = {"bngsim": bngsim_engine}
    errors: dict[str, str] = {}
    references: dict[str, Any] = {}
    _progress("bngsim_integration: build + trial solve")
    try:
        integ = _build_bngsim(prep, "integration")
        integ.solve_all()
        engines["bngsim_integration"] = integ
        references["bngsim_integration"] = integ.config()
    except Exception as exc:  # noqa: BLE001
        errors["bngsim_integration"] = f"{type(exc).__name__}: {exc}"[:400]
    # A reference whose answer the correctness job did not accept has no time worth
    # measuring: record why and time the rest. The others get a trial solve, so one
    # that crashes is dropped rather than taking the interleave down.
    failed = _correctness_errors(job, ctx)
    for name in REFERENCE_ENGINES:
        if name in failed:
            errors[name] = f"failed in the correctness job, so not timed ({failed[name][:300]})"
            _progress(f"{name}: skipped, {errors[name]}")
            continue
        _progress(f"{name}: build + trial solve")
        try:
            engine = _build_reference(name, prep, out_prefix=workdir / name / "rn",
                                      timeout=timeout)
            engine.solve_all()
            engines[name] = engine
            references[name] = engine.config()
        except Exception as exc:  # noqa: BLE001
            errors[name] = f"{type(exc).__name__}: {exc}"[:400]
    info["references"] = references

    _progress(f"interleaved timing of {', '.join(engines)}, {n} point(s) per repetition")
    runs = {name: eng.solve_all for name, eng in engines.items()}
    captures = {name: np.array for name in engines}
    reps = round_robin_interleaved(runs, captures=captures, **job.definition["timing"])

    outputs = {name: reps["engines"][name].pop("outputs") for name in runs}
    repro: dict[str, Any] = {"bngsim": reproducibility(anchor, outputs["bngsim"], REPRO_TOL)}
    answers: dict[str, Any] = {}
    for name in engines:
        if name == "bngsim":
            continue
        outs = outputs[name]
        repro[name] = reproducibility(outs[0], outs[1:], REPRO_TOL)
        answers[name] = _summary(agreement(outs[0], anchor, atol, names, sinks))
    reproducible = all(r["passed"] for r in repro.values())

    times = absolute_time_stats({name: reps["engines"][name]["seconds"] for name in runs})
    per_point = {name: {key: t[key] / n for key in ("median_seconds", "ci_low", "ci_high")}
                 for name, t in times.items()}
    rec: dict[str, Any] = {
        "outcome": "PASS" if reproducible else "NOT_REPRODUCIBLE",
        "measurements": {
            **info,
            "engine_roles": {name: engine_role(name) for name in engines},
            "repetitions": reps,
            "reproducibility": repro,
            "answers": answers,
            "times": times,
            "times_per_point": per_point,
            "n_points": n,
        },
    }
    for name, err in errors.items():
        rec[f"{name}_error"] = err
    return rec


def run(job: Job, problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    if job.kind == CORRECTNESS:
        return run_correctness(job, problem, ctx)
    if job.kind == TIMING:
        return run_timing(job, problem, ctx)
    raise ValueError(f"unknown job kind {job.kind!r}")
