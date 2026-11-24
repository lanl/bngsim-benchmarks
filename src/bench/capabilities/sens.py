"""Forward sensitivities of BNGL models: BNGsim's ``dy/dp`` against reference engines.

BNGsim's forward sensitivities -- CVODES integrating the state and its derivatives
with respect to parameters together (``Simulator(sensitivity_params=...)``) -- are
the system under test, in two configurations: the ``staggered`` corrector,
bngsim's default, and ``simultaneous``, AMICI's default. On one problem they are
compared with every reference in ``REFERENCE_ENGINES``:

- **amici** -- AMICI's forward sensitivities (CVODES, with generated sensitivity
  equations), compiled from the SBML exported from the generated ``.net``, on a
  linear parameter scale;
- **run_network_fd** -- central finite differences from BioNetGen's
  ``run_network``, an engine that shares no code with BNGsim: the ``.net`` with one
  parameter moved up and down by a relative step, at two steps and at tolerances
  far tighter than the protocol's (``bench.sensitivity`` explains the error bound);
- **copasi** -- COPASI's time-course sensitivities task (LSODA integrating the
  sensitivity system), loaded through BASICO from the same SBML, read at every
  output time through a data handler.

**Outputs and parameters.** Every engine returns ``S[t, i, k] = d x_i(t) / d p_k``
for every species on the protocol's grid; the observables are formed from it by the
harness, as the ``.net``'s weighted sums of species, the same way for every engine.
The parameters are the protocol's ``parameters``, a convention every problem names:
``independent``, every parameter the ``.net`` writes as a number that some reaction
rate or initial amount depends on, directly or through derived parameters and
functions; ``rate_constants``, those that reach a rate; ``fitted``, the problem's
fitted set (``fitted``); or an explicit list. The harness reads that from the ``.net``
itself (``netfile.Structure``). A problem whose fitted set is small is differentiated
over every independent parameter, and its fitted set is then a job setting of its own
(``fitted``, beside ``auto``). A parameter that sizes a compartment is left out, and
so is one whose value is zero unless the protocol gives it a scale
(``parameter_scales``): the semi-scaled sensitivity is ``scale * dx/dp``, a parameter's
scale is its value's magnitude by default, and the finite differences step by
``fd_step * scale``, so a zero-valued dose or switch gets an absolute step. Every
parameter left out is recorded with its reason.

**Protocol.** One measured phase from the network's initial state (``t_end``,
``n_steps``), or an ordered list of finite ``phases`` run back to back, t = 0 at the
start of the first measured one (the phases before it run at negative times, as
AMICI's pre-simulation does). Each phase has a ``duration``, a grid (``n_steps``) when
it is measured, and, as it starts, the parameter values it ``set``s and the species
it ``reset``s -- to a number or an expression of the parameters (``S = 2*S0``). The
state and its sensitivities are carried from one phase into the next; a reset
species starts its phase with its new value and, as its sensitivity row, that
value's derivative. BNGsim runs the phases on one Simulator, its sensitivities
carried across each boundary (``carry_sensitivities=True``); it will not carry them
across a species reset, and that refusal is BNGsim's result (UNSUPPORTED, in its own
words). AMICI runs an equilibration then a measured phase by its own pre-simulation
(``t_presim``, the set parameters compiled as its fixed parameters), and any other
protocol as a chain of simulations, each started from the last one's state and state
sensitivities; the finite differences by a fresh run_network per phase, each started
from the last one's state, as BNG2.pl continues a run, a reset written into the file
as an expression run_network evaluates itself. A phase has a duration, not a
steady-state criterion, since a model need not reach a steady state (the p38-ATF2
network has an absorbing state its authors' 20000 s equilibration is far from); how
far an unmeasured phase still moved over its second half is recorded
(``drift_last_half_of_unmeasured``). A parameter the protocol sets is the protocol's,
not the model's, and is not differentiated. COPASI's time-course sensitivities task
cannot begin from an earlier phase's sensitivities, so it is recorded as not
applicable to a multi-phase protocol (named, not run, not counted).

**Correctness.** Each engine builds and solves cold in its own forked child under a
wall budget (``run_isolated``), so an AMICI compile that would take hours is stopped
and recorded, and the job goes on. The verdict of BNGsim against a reference is the
column error of ``bench.sensitivity`` -- the largest difference of the semi-scaled
sensitivity ``p_k S`` over time, as a fraction of that column's peak, with absolute
floors -- at most ``VERDICT["tol"]`` in every judged column. A reference judges the
columns it produced: AMICI only the parameters it exposes as free, finite
differences only the columns their two-step error estimate resolves. Each further
BNGsim configuration must agree with the default one, and BNGsim's own observable
sensitivities with the ones the harness forms from its species sensitivities.

**Attribution.** A reference that crashes, runs out of time or cannot differentiate a
parameter is recorded, never patched (``<name>_error``). A reference that runs but
disagrees with BNGsim is checked against the other references: where, on every
column it fails, another reference agrees with BNGsim and disagrees with it -- two
independent engines against one -- the disagreement is the reference's own, and the
verdict records ``refuted_by``. The outcome is DIFF when any disagreement is not
refuted (or a BNGsim configuration disagrees with the default); otherwise
REFERENCE_FAILED when a reference failed, was refuted, or left a parameter that no
agreeing reference judged; otherwise PASS. A BNGsim failure is EXCEPTION (TIMEOUT
when it ran out of time).

**Timing.** The interleave times one forward-sensitivity solve of the protocol per
engine: each BNGsim configuration, AMICI and COPASI, beside a plain solve of the same
protocol by BNGsim (``bngsim_ode``) and by run_network, the unit of a finite-difference
gradient. The job reports each
engine's absolute time with its bootstrap CI, the one-time cost before solving --
BNGsim's load and compile, AMICI's code generation and C++ compile, COPASI's import
-- measured once and cold, and the finite-difference baselines derived from the
plain solves: ``N + 1`` solves for forward differences, ``2N`` for central ones. Each
kept repetition must reproduce the first run.

Input formats: the jobs are built from BNGL through the generated ``.net``; BNGsim and
the finite differences read the ``.net``, AMICI and COPASI the SBML exported from it.
An SBML suite (a PEtab problem's model, say) would load the SBML into BNGsim, AMICI
and COPASI directly and take its finite differences from an engine that reads SBML;
the engine adapters take their input by kind (``INPUT``), and ``jobs`` is where such
a suite would be admitted, with its own engines id.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from .. import env, netfile
from .. import sensitivity as sv
from ..jobs import CORRECTNESS, TIMING, Job
from ..specs import SENS_PARAMETER_CLASSES, Problem
from ..timing import absolute_time_stats, round_robin_interleaved
from ._common import (
    correctness_timeouts,
    engine_role,
    prepare_network,
    progress,
    read_dat,
    recorded_command,
    reproducibility,
    run_isolated,
    threads,
)

CAPABILITY = "sens"
# Bump when this module's computation changes; every result it wrote goes stale.
HARNESS_VERSION = "1"

TIMING_PROTOCOL = {"warmup_rounds": 1, "min_rounds": 5, "max_rounds": 20, "budget_seconds": 120.0}
# A timing-only part of the key, as in ode.
REPRO_CHECK_VERSION = 1
# Worst a kept repetition may disagree with the first run, per sensitivity column,
# as a fraction of that column's peak (as ode's REPRO_TOL for trajectories).
REPRO_TOL = 1e-6

# The verdict (bench.sensitivity). In the correctness definition, so a change
# restales every correctness result and nothing else.
# - tol: the largest column error a comparison passes with. The engines solve at the
#   protocol's tolerances (1e-8 in the pilots); their sensitivities agree to about
#   1e-5 of a column's peak, finite differences to about 1e-4, and a wrong
#   derivative (a missed chain-rule term, a dropped initial-condition seed) is off
#   by order one.
# - floor_rel: a column whose largest semi-scaled sensitivity is under this fraction
#   of its output's peak is negligible, and judged against that floor;
# - floor_atol: no column's allowed error is under this many state atols;
# - fd_resolved: an FD column is judged only where its two-step error estimate is at
#   most this fraction of the allowed error.
VERDICT = {
    "metric": "semi-scaled column error/1",
    "tol": 1e-3,
    "floor_rel": 1e-3,
    "floor_atol": 100.0,
    "fd_resolved": 0.1,
}

# The conventions a protocol may name instead of a list (specs.py validates the same).
PARAMETER_CLASSES = SENS_PARAMETER_CLASSES
# The finite differences: step h (and h/2) relative to each parameter's scale -- its
# value, or the protocol's parameter_scales for one whose value is zero -- at
# tolerances tighter than any protocol's, so the integration's error over the step
# is small against the derivative (bench.sensitivity).
FD_DEFAULTS = {"fd_step": 1e-3, "fd_rtol": 1e-10, "fd_atol": 1e-12}

# The job settings. ``auto``: the parameters the problem's convention names, every
# BNGsim choice (codegen, linear solver) left to bngsim. ``fitted``: the same, over
# the problem's fitted set only, where its convention is another (a problem whose
# fitted set is small is differentiated over every independent parameter, and the
# fitted subset is kept as a setting beside it).
DEFAULT_SETTING = "auto"
FITTED_SETTING = "fitted"

# The BNGsim configurations under test: CVODES' two correctors for the coupled
# state + sensitivity system. Codegen and the linear solver are bngsim's own choice
# (sensitivities always compile; there is no interpreted path to compare).
CONFIGURATIONS: dict[str, dict[str, str]] = {
    "bngsim": {"sensitivity_method": "staggered"},
    "bngsim_simultaneous": {"sensitivity_method": "simultaneous"},
}
DEFAULT_CONFIGURATION = "bngsim"


def protocol(problem: Problem, setting: str = DEFAULT_SETTING) -> dict[str, Any]:
    """The sensitivity protocol with defaults filled in, as it enters a job definition;
    for the ``fitted`` setting, over the fitted set."""
    p = dict(problem.protocol_for(CAPABILITY))
    for k, v in FD_DEFAULTS.items():
        p.setdefault(k, v)
    if setting == FITTED_SETTING:
        p["parameters"] = "fitted"
    return p


def settings(problem: Problem) -> list[str]:
    """The settings a problem is run at: its own convention, and its fitted set
    beside it when that is another selection."""
    p = problem.protocol_for(CAPABILITY)
    out = [DEFAULT_SETTING]
    if p.get("fitted") and p.get("parameters") != "fitted":
        out.append(FITTED_SETTING)
    return out


def engines_id() -> str:
    """The ``engines`` id segment: BNGsim versus every registered reference."""
    return "bngsim-vs-" + "-".join(REFERENCE_ENGINES)


def jobs(problem: Problem) -> list[Job]:
    # BNGL only for now; an SBML source is where a sens-sbml suite would be admitted
    # (see the module docstring), with its own engines id.
    if problem.source.get("format") != "bngl":
        return []
    engines = engines_id()
    out: list[Job] = []
    for setting in settings(problem):
        base = {
            "harness": f"{CAPABILITY}/{engines}/{HARNESS_VERSION}",
            "source_sha256": problem.source["sha256"],
            "protocol": protocol(problem, setting),
            "setting": {"configurations": CONFIGURATIONS, "codegen": "auto",
                        "linear_solver": "auto"},
            # As in ode: the SBML references' versions key only this capability;
            # run_network's rides the global fingerprint.
            "reference_versions": {
                "amici": env.amici_identity().get("version"),
                "copasi": env.copasi_identity().get("version"),
            },
        }
        est_c, est_t = planning_seconds(problem, setting)
        backstop = (len(REFERENCE_ENGINES) + len(CONFIGURATIONS) + 1) \
            * engine_budget_seconds(problem, setting)
        correctness = Job(
            CAPABILITY, problem.id, engines, setting, CORRECTNESS,
            definition={**base, "verdict": VERDICT},
            est_seconds=est_c,
            timeout_seconds=max(MIN_TIMEOUT_SECONDS, TIMEOUT_FACTOR * est_c, backstop),
        )
        timing = Job(
            CAPABILITY, problem.id, engines, setting, TIMING,
            definition={**base, "timing": TIMING_PROTOCOL, "timing_check": REPRO_CHECK_VERSION},
            est_seconds=est_t,
            timeout_seconds=max(MIN_TIMEOUT_SECONDS, TIMEOUT_FACTOR * est_t),
            requires=correctness.id,
        )
        out += [correctness, timing]
    return out


# --------------------------------------------------------------------------- #
# Planning (never part of a freshness key)
# --------------------------------------------------------------------------- #
# A sensitivity job's cost is a few plain solves' worth per parameter for every
# engine, plus AMICI's code generation and compile, which dominates on all but the
# smallest networks and grows faster than the network (the BNGsim paper's AMICI
# compiles: about 25 s at 100 reactions, 250 s at 3700, over an hour at 15000).
# The number of parameters comes from an explicit list or the fitted set, else the
# spec's ``size.sens_parameters`` (planning only), else DEFAULT_PARAMETER_COUNT.
# Calibrated on the two pilot problems; provisional until larger ones have run.
MIN_TIMEOUT_SECONDS = 1800.0
TIMEOUT_FACTOR = 4.0
ENGINE_BUDGET_CAP_SECONDS = 3600.0
DEFAULT_PARAMETER_COUNT = 20


def parameter_count(problem: Problem, setting: str = DEFAULT_SETTING) -> int:
    proto = protocol(problem, setting)
    chosen = proto["parameters"]
    if isinstance(chosen, list):
        return len(chosen)
    if chosen == "fitted":
        return len(proto["fitted"])
    return int(problem.size.get("sens_parameters", DEFAULT_PARAMETER_COUNT))


def engine_seconds(problem: Problem, setting: str = DEFAULT_SETTING) -> dict[str, float]:
    """Planned seconds for each engine's cold build and first solve."""
    reactions = problem.size.get("reactions") or 10 * problem.size.get("species", 100)
    n = parameter_count(problem, setting)
    phases = len(phase_plan(problem.protocol_for(CAPABILITY)))
    # One plain solve of the protocol (every phase) at its tolerances.
    solve = phases * (0.002 + 2e-6 * reactions ** 1.5)
    sens = 2.0 * (1 + n) * solve  # a forward-sensitivity solve
    return {
        "bngsim": 5.0 + 1e-3 * reactions + sens,
        "bngsim_simultaneous": 1.0 + sens,
        "amici": 25.0 + 3e-4 * reactions ** 1.7 + sens,
        # 4N + 1 solves, each a fresh process per phase at tolerances 100x tighter.
        "run_network_fd": (4 * n + 1) * (phases * 0.02 + 3.0 * solve),
        "copasi": 5.0 + 50.0 * sens,
    }


def planning_seconds(problem: Problem, setting: str = DEFAULT_SETTING) -> tuple[float, float]:
    """Rough wall-clock estimates (correctness, timing) for one problem."""
    per = engine_seconds(problem, setting)
    correctness = 30.0 + sum(per.values())
    # Timing rebuilds every engine but the FD reference, then repeats each solve over
    # at least min_rounds plus a warm-up; the solves are a small part of the build.
    builds = sum(v for k, v in per.items() if k != "run_network_fd")
    timing = 60.0 + builds + (TIMING_PROTOCOL["min_rounds"] + 2) * 0.2 * builds
    return correctness, timing


def engine_budget_seconds(problem: Problem, setting: str = DEFAULT_SETTING) -> float:
    """Wall budget for one engine's cold build + solve in the correctness job."""
    slowest = max(engine_seconds(problem, setting).values())
    return min(ENGINE_BUDGET_CAP_SECONDS, max(MIN_TIMEOUT_SECONDS, TIMEOUT_FACTOR * slowest))


# --------------------------------------------------------------------------- #
# Choosing the parameters
# --------------------------------------------------------------------------- #
def select_parameters(structure: netfile.Structure, chosen: str | list[str], *,
                      fitted: list[str] | None = None,
                      scales: dict[str, float] | None = None,
                      switched: set[str] | frozenset[str] = frozenset(),
                      ) -> tuple[list[str], dict[str, str]]:
    """The parameters to differentiate, in ``.net`` order, and those left out with
    the reason.

    ``chosen`` is a convention or a list: ``independent``, every parameter written
    as a number that reaches a reaction rate or an initial amount; ``rate_constants``,
    those that reach a rate; ``fitted``, the ``fitted`` list. A parameter that sizes a
    compartment is left out (bngsim folds a volume into constants at load), and so is
    one whose value is zero unless ``scales`` gives it a scale: a relative step and a
    semi-scaled sensitivity of a zero are zero; and so is one a multi-phase protocol
    sets (``switched``), which is the protocol's, not the model's. Raises ValueError
    for a listed name the network cannot honour."""
    primary = structure.primary()
    rates = structure.reaching_rates()
    initial = structure.reaching_initial()
    sizes = structure.sizing_compartments()
    scales = scales or {}
    excluded: dict[str, str] = {}
    if chosen == "fitted":
        chosen = list(fitted or [])
        if not chosen:
            raise ValueError("protocol.sens.parameters: fitted, but no fitted set is given")
    listed = isinstance(chosen, list)
    named = list(chosen) if listed else []
    bad_scales = [p for p in scales if p not in primary]
    for where, names in (("parameters", named), ("parameter_scales", bad_scales)):
        unknown = [p for p in names if p not in structure.expressions]
        derived = [p for p in names if p in structure.expressions and p not in primary]
        if unknown or derived:
            raise ValueError(
                f"protocol.sens.{where}: "
                + "; ".join(filter(None, [
                    f"not parameters of the network: {', '.join(unknown)}" if unknown else "",
                    f"derived (an expression, not a number): {', '.join(derived)}"
                    if derived else ""])))
    if chosen == "independent":
        candidates = [p for p in primary if p in rates or p in initial]
    elif chosen == "rate_constants":
        candidates = [p for p in primary if p in rates]
    else:
        candidates = [p for p in primary if p in named]
        for p in candidates:
            if p not in rates and p not in initial:
                excluded[p] = "reaches no reaction rate or initial amount"
        candidates = [p for p in candidates if p not in excluded]
    chosen_out = []
    for p in candidates:
        if p in switched:
            excluded[p] = "set by the protocol"
        elif p in sizes:
            excluded[p] = "sizes a compartment"
        elif primary[p] == 0.0 and p not in scales:
            excluded[p] = "zero value: no relative step (give it a parameter_scales entry)"
        else:
            chosen_out.append(p)
    return chosen_out, excluded


def convention(proto: dict[str, Any]) -> str:
    """The label of the protocol's parameter selection: ``fitted`` (the problem's
    fitted set), ``independent`` (a convention over the network: every independent
    parameter, or every rate constant) or ``listed`` (an explicit list)."""
    chosen = proto["parameters"]
    if chosen == "fitted" or (isinstance(chosen, list) and chosen == proto.get("fitted")):
        return "fitted"
    return "listed" if isinstance(chosen, list) else "independent"


def parameter_scales(structure: netfile.Structure, params: list[str],
                     scales: dict[str, float] | None = None) -> np.ndarray:
    """Each parameter's scale: the protocol's ``parameter_scales`` entry, else the
    magnitude of its value. The semi-scaled sensitivity is ``scale * dy/dp``, and the
    finite differences step by ``fd_step * scale``."""
    scales = scales or {}
    return np.array([float(scales[p]) if p in scales else abs(structure.literal(p))
                     for p in params])


# --------------------------------------------------------------------------- #
# The system under test and the reference engines
# --------------------------------------------------------------------------- #
def phase_plan(proto: dict[str, Any]) -> list[dict[str, Any]]:
    """The protocol's phases in order, from the protocol alone: each one's ``start``,
    ``duration``, ``n_steps`` (None for an unmeasured phase), and what it ``set``s and
    ``reset``s as it starts. The phases run back to back; t = 0 is the start of the
    first measured phase, so the phases before it run at negative times (as AMICI's
    pre-simulation does). A protocol without ``phases`` is one measured phase from
    t = 0, of ``t_end`` in ``n_steps``."""
    if "phases" not in proto:
        return [{"start": 0.0, "duration": float(proto["t_end"]),
                 "n_steps": int(proto["n_steps"]), "set": {}, "reset": {}}]
    phases = [{"duration": float(ph["duration"]),
               "n_steps": int(ph["n_steps"]) if "n_steps" in ph else None,
               "set": dict(ph.get("set") or {}), "reset": dict(ph.get("reset") or {})}
              for ph in proto["phases"]]
    first = next(k for k, ph in enumerate(phases) if ph["n_steps"] is not None)
    start = -sum(ph["duration"] for ph in phases[:first])
    for ph in phases:
        ph["start"] = start
        start += ph["duration"]
    return phases


def times(proto: dict[str, Any]) -> np.ndarray:
    """The output times: every measured phase's grid, start to end, in order (a
    measured phase reports its own start, as a BioNetGen simulate does)."""
    return np.concatenate([np.linspace(ph["start"], ph["start"] + ph["duration"],
                                       ph["n_steps"] + 1)
                           for ph in phase_plan(proto) if ph["n_steps"] is not None])


def schedule(structure: netfile.Structure, proto: dict[str, Any]) -> dict[str, Any]:
    """The protocol as the engines run it: ``phases`` (``phase_plan``), each with the
    ``values`` of every parameter the protocol sets that are in force during it,
    and its resets as ``[(species index, expression)]``; ``switched``, those
    parameters; ``plain``, whether it is a single phase from the network's own
    initial state.

    A set never moves a species: the state is carried from one phase into the next
    by every engine, so a set parameter that also seeds an inert species (one no
    reaction consumes or produces and no rate reads, as a vestigial inhibitor
    species beside the parameter its rate law reads) leaves that species where it
    was; those are recorded (``held``).

    Raises ValueError for a protocol the engines could not all run the same way: a
    set parameter that is not a number in the network, or that reaches a
    compartment size or the initial amount of a species that is not inert (whether
    a change mid-protocol moves a species is where engines differ: BioNetGen's
    setParameter does not), or that reaches no rate at all; a reset of a species the
    network does not have, or to an expression of names that are not its
    parameters; and a multi-phase protocol on a network whose rates read the clock
    (run_network starts each phase at 0)."""
    phases = phase_plan(proto)
    switched = sorted({p for ph in phases for p in ph["set"]})
    primary = structure.primary()
    rates, initial = structure.reaching_rates(), structure.reaching_initial()
    sizes = structure.sizing_compartments()
    inert = structure.inert()
    held = {p: [structure.species[i] for i in structure.initial_reading(p)]
            for p in switched if p in primary and p in initial}
    bad = [f"{p} is not a number in the network" for p in switched if p not in primary]
    bad += [f"{p} reaches a compartment size" for p in switched if p in primary and p in sizes]
    seeding = [p for p in held if not set(structure.initial_reading(p)) <= inert]
    bad += [f"{p} reaches an initial amount (of {', '.join(held[p][:3])})" for p in seeding]
    bad += [f"{p} reaches no reaction rate" for p in switched
            if p in primary and p not in rates and p not in sizes and p not in seeding]
    index = {name: i for i, name in enumerate(structure.species)}
    for ph in phases:
        for name, expr in ph["reset"].items():
            if name.lstrip("$") not in index:
                bad.append(f"a reset of {name}, which is not a species of the network")
            elif isinstance(expr, str):
                unknown = sorted(netfile.names_in(expr) - set(structure.expressions))
                if unknown:
                    bad.append(f"the reset of {name} reads {', '.join(unknown)}, which are not "
                               "parameters of the network")
    if bad:
        raise ValueError("protocol.sens.phases: " + "; ".join(bad))
    if len(phases) > 1 and structure.reads_clock():
        raise ValueError("protocol.sens.phases: a multi-phase protocol on a network whose rates "
                         "read the clock (time()): the engines would not agree on when it starts")
    values = {p: float(primary[p]) for p in switched}
    for ph in phases:
        values = {**values, **{p: float(v) for p, v in ph["set"].items()}}
        ph["values"] = dict(values)
        ph["resets"] = [(index[name.lstrip("$")],
                         expr if isinstance(expr, str) else repr(float(expr)))
                        for name, expr in ph["reset"].items()]
    plain = len(phases) == 1 and not phases[0]["set"] and not phases[0]["resets"]
    return {"phases": phases, "switched": switched, "plain": plain,
            **({"held": held} if held else {})}


def reset_values(net_text: str, sched: dict[str, Any], params: list[str],
                 scales: np.ndarray) -> None:
    """Give every reset in ``sched`` its ``value`` and its ``gradient`` (d value / d p
    for each differentiated parameter): the reset species' new state and the
    sensitivity row it starts the phase with. Computed by the harness's own reader
    from the ``.net``, through every derived parameter: the value with the phase's
    parameter values in force, the gradient by central differences of the
    expression (exact to rounding for the polynomial expressions a reset takes).
    BNGsim and AMICI are handed these; the finite differences evaluate the
    expression themselves, in run_network, at each perturbed point."""
    for ph in sched["phases"]:
        ph["reset_values"] = []
        for i, expr in ph["resets"]:
            def value(over: dict[str, float]) -> float:
                net = netfile.parse(net_text, {**ph["values"], **over})
                return netfile.evaluate(expr, net.parameters)
            base = netfile.parse(net_text, ph["values"]).parameters
            grad = []
            for p, s in zip(params, scales):
                h = 1e-6 * s
                grad.append((value({p: base[p] + h}) - value({p: base[p] - h})) / (2 * h))
            ph["reset_values"].append((i, value({}), np.array(grad)))


def phases_info(sched: dict[str, Any], species: list[str]) -> list[dict[str, Any]]:
    """The schedule as the record names it: each phase's span, grid, the parameter
    values in force and its resets (species, expression, value)."""
    out = []
    for ph in sched["phases"]:
        resets = [{"species": species[i], "expression": expr}
                  for i, expr in ph["resets"]]
        for r, (_, value, _) in zip(resets, ph.get("reset_values", [])):
            r["value"] = float(value)
        out.append({"start": ph["start"], "duration": ph["duration"],
                    "n_steps": ph["n_steps"], "values": dict(ph["values"]),
                    **({"resets": resets} if resets else {})})
    return out


# bngsim's own words when it will not carry sensitivities across a species reset.
_BNGSIM_NO_SEED = "no matching forward-sensitivity seed"


def pack(S: np.ndarray, X: np.ndarray, *extra: np.ndarray) -> np.ndarray:
    """One array a forked child can return: ``S`` ``(n_t, n_sp, n_p)``, any further
    tensors of the same shape, then ``X`` ``(n_t, n_sp)`` as the last slice."""
    return np.concatenate([np.asarray(S, float), *map(np.asarray, extra),
                           np.asarray(X, float)[:, :, None]], axis=2)


def unpack(a: np.ndarray, n_p: int) -> tuple[list[np.ndarray], np.ndarray]:
    """The tensors and the trajectory of ``pack``."""
    k = (a.shape[2] - 1) // n_p
    return [a[:, :, j * n_p:(j + 1) * n_p] for j in range(k)], a[:, :, -1]


def _drift(pre_species: np.ndarray) -> float | None:
    """How far an unmeasured phase still moved over its second half: the largest
    change from its midpoint to its end, as a fraction of the largest species at the
    end (a diagnostic of the protocol, not a verdict)."""
    x = np.asarray(pre_species, dtype=float)
    peak = float(np.max(np.abs(x[-1]))) if x.size else 0.0
    return float(np.max(np.abs(x[-1] - x[-2])) / peak) if peak > 0 and len(x) >= 2 else None


def _plain_schedule(proto: dict[str, Any]) -> dict[str, Any]:
    """The schedule of a protocol with no phases of its own (what ``schedule`` gives
    it, without a network to check it against)."""
    if "phases" in proto:
        raise ValueError("a protocol with phases is run from its schedule (sens.schedule)")
    phases = phase_plan(proto)
    for ph in phases:
        ph.update(values={}, resets=[], reset_values=[])
    return {"phases": phases, "switched": [], "plain": True}


class Measured:
    """The measured phases of one protocol run, as one bngsim Result reads: the
    species, observables and sensitivities of every measured phase, concatenated in
    order, and the likelihood gradient summed over them. A single-phase run is
    itself (``Measured([result])``)."""

    def __init__(self, results: list):
        self.results = results
        last = results[-1]
        self.sensitivity_params = list(getattr(last, "sensitivity_params", []) or [])
        self.solver_stats = getattr(last, "solver_stats", None)
        self.observable_names = list(getattr(last, "observable_names", []) or [])
        self.has_sensitivities_observables = all(
            getattr(r, "has_sensitivities_observables", False) for r in results)

    def _cat(self, attr: str) -> np.ndarray:
        return np.concatenate([np.asarray(getattr(r, attr), dtype=float) for r in self.results])

    @property
    def species(self) -> np.ndarray:
        return self._cat("species")

    @property
    def observables(self) -> np.ndarray:
        return self._cat("observables")

    @property
    def sensitivities(self) -> np.ndarray:
        return self._cat("sensitivities")

    @property
    def sensitivities_observables(self) -> np.ndarray:
        return self._cat("sensitivities_observables")

    def neg_log_likelihood_gradient(self, data, sigma, *, target: str):
        """The sum over the measured phases of each one's gradient, the data split
        by the phases' rows."""
        total, grad, row = 0.0, None, 0
        for r in self.results:
            n = len(np.asarray(r.species))
            value, g = r.neg_log_likelihood_gradient(data[row:row + n], sigma[row:row + n],
                                                     target=target)
            total += float(value)
            grad = np.asarray(g, dtype=float) if grad is None else grad + np.asarray(g, float)
            row += n
        return total, grad


class _BngsimProtocol:
    """Runs the protocol on one bngsim model and Simulator, through its public API:
    each phase in order -- the parameter values it sets (``model.set_param``), the
    species it resets (``model.set_concentration``), then ``run`` over its span, the
    first from the network's initial state after ``model.reset()``, every later one
    continuing from where the last ended, its sensitivities carried across the
    boundary (``carry_sensitivities=True``) when the Simulator has them. A reset in
    the first phase is an initial condition whose derivative bngsim is told
    (``declare_ic_sensitivity``); bngsim refuses to carry sensitivities across a
    reset in a later phase, and that refusal is BNGsim's result."""

    def _protocol(self, carry: bool) -> Measured:
        rtol = float(self.proto["rtol"])
        measured, self.drift = [], []
        for k, ph in enumerate(self.schedule["phases"]):
            for name, value in ph["values"].items():
                self.model.set_param(name, value)
            if k == 0:
                self.model.reset()
            for i, value, grad in ph.get("reset_values", []):
                name = self.species_names[i]
                self.model.set_concentration(name, value)
                if k == 0 and carry:
                    self.model.declare_ic_sensitivity(
                        {name: dict(zip(self.params, map(float, grad)))})
            n = ph["n_steps"]
            extra = {"carry_sensitivities": True} if carry and k > 0 else {}
            try:
                result = self.sim.run(t_span=(ph["start"], ph["start"] + ph["duration"]),
                                      n_points=(n + 1) if n is not None else 3, rtol=rtol,
                                      atol=self.atol, **extra)
            except Exception as exc:
                if extra and ph.get("reset_values") and _BNGSIM_NO_SEED in str(exc):
                    raise RuntimeError(
                        f"UNSUPPORTED: bngsim will not carry forward sensitivities across "
                        f"the species reset that starts phase {k + 1} of "
                        f"{len(self.schedule['phases'])}: {exc}") from exc
                raise
            if n is None:
                self.drift.append(_drift(np.asarray(result.species, dtype=float)))
            else:
                measured.append(result)
        return Measured(measured)

    def _phases_config(self) -> dict[str, Any]:
        if self.schedule["plain"]:
            return {}
        return {"phases": {"n": len(self.schedule["phases"]),
                           "drift_last_half_of_unmeasured": self.drift}}


class BngsimSens(_BngsimProtocol):
    """BNGsim's forward sensitivities on the ``.net``: one model, one Simulator built
    for the parameters (which compiles the model with its analytic sensitivity
    right-hand side), run through the protocol's phases each time, its
    sensitivities carried from one phase to the next."""

    def __init__(self, net: Path, proto: dict[str, Any], params: list[str], *,
                 sensitivity_method: str = "staggered", schedule: dict[str, Any] | None = None):
        import bngsim

        self.proto, self.params, self.method = proto, list(params), sensitivity_method
        self.schedule, self.drift = schedule or _plain_schedule(proto), []
        t0 = time.perf_counter()
        self.model = bngsim.Model.from_net(str(net))
        self.load_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        self.sim = bngsim.Simulator(self.model, method="ode", sensitivity_params=self.params,
                                    sensitivity_method=sensitivity_method)
        self.codegen_seconds = time.perf_counter() - t1
        # As in ode: the scalar atol held exactly, the tolerance AMICI takes too.
        self.atol = bngsim.ExactAtol(float(proto["atol"]))
        self.species_names = list(self.model.species_names)
        self.covered = np.ones(len(self.params), bool)
        self.stats: dict[str, Any] = {}

    def run(self):
        return self._protocol(carry=True)

    def tensors(self, result) -> tuple[np.ndarray, np.ndarray]:
        names = list(result.sensitivity_params)
        if names != self.params:
            raise RuntimeError(f"bngsim returned sensitivities for {names[:5]}..., "
                               f"not the {len(self.params)} parameters asked for")
        self.stats = dict(getattr(result, "solver_stats", None) or {})
        return np.array(result.sensitivities, dtype=float), np.array(result.species, dtype=float)

    @staticmethod
    def observable_sensitivities(result) -> tuple[list[str], np.ndarray]:
        """BNGsim's own observable sensitivities ``(n_t, n_obs, n_p)``, by name."""
        if not result.has_sensitivities_observables:
            return [], np.zeros((0, 0, 0))
        return list(result.observable_names), np.array(result.sensitivities_observables, float)

    def config(self) -> dict[str, Any]:
        from .ode import LINEAR_SOLVER_NAMES

        code = self.stats.get("linear_solver")
        return {
            "sensitivity_method": self.method,
            "codegen_backend": self.sim.codegen_backend,
            "codegen_cache_hit": self.sim.codegen_cache_hit,
            "analytic_sensitivity_rhs": bool(self.sim.has_analytic_sens_rhs),
            "sensitivity_rhs_decline": self.sim.sens_rhs_decline_reason,
            "linear_solver": LINEAR_SOLVER_NAMES.get(code, f"kind_{code}") if code is not None
            else None,
            "solver_stats": {k: self.stats.get(k) for k in
                             ("n_steps", "n_rhs_evals", "n_err_test_fails",
                              "n_sens_err_test_fails", "n_sens_nonlin_conv_fails")},
            "load_seconds": self.load_seconds,
            "codegen_seconds": self.codegen_seconds,
            **self._phases_config(),
        }


class BngsimPlain(_BngsimProtocol):
    """BNGsim's plain solve of the same protocol, no sensitivities: the unit a
    finite-difference gradient is paid in, timed beside the sensitivity solve."""

    def __init__(self, net: Path, proto: dict[str, Any], *,
                 schedule: dict[str, Any] | None = None):
        import bngsim

        self.proto, self.params = proto, []
        self.schedule, self.drift = schedule or _plain_schedule(proto), []
        t0 = time.perf_counter()
        self.model = bngsim.Model.from_net(str(net))
        self.load_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        self.sim = bngsim.Simulator(self.model, method="ode")
        self.codegen_seconds = time.perf_counter() - t1
        self.atol = bngsim.ExactAtol(float(proto["atol"]))
        self.species_names = list(self.model.species_names)

    def run(self):
        return self._protocol(carry=False)

    @staticmethod
    def trajectory(result) -> np.ndarray:
        return np.asarray(result.species, dtype=float)

    def config(self) -> dict[str, Any]:
        return {"codegen_backend": getattr(self.sim, "codegen_backend", None),
                "load_seconds": self.load_seconds, "codegen_seconds": self.codegen_seconds,
                **self._phases_config()}


def _align(species_names: list[str], name_to_id: dict[str, str]) -> list[str]:
    missing = [n for n in species_names if n not in name_to_id]
    if missing:
        raise RuntimeError(
            f"cannot align {len(missing)} BNGsim species to SBML ids "
            f"(e.g. {missing[:3]}); export/name map is wrong")
    return [name_to_id[n] for n in species_names]


def _not_covered(names: list[str], covered: np.ndarray, why: str) -> dict[str, str]:
    return {p: why for p, c in zip(names, covered) if not c}


class AmiciSens:
    """AMICI's forward sensitivities, from the SBML exported from the ``.net``.

    ``sbml2amici`` generates and compiles the model with its sensitivity equations
    once, in ``__init__`` (``codegen_seconds``: a pre-solve cost). The parameters are
    put on a linear scale, so ``sx`` is ``d x / d p`` and not ``d x / d log p``. A
    parameter is matched to AMICI's free parameters by SBML id (the BNGsim name;
    AMICI prefixes ``amici_`` to a name that collides with its own symbols); one
    AMICI does not expose as free -- an SBML parameter it folds into an expression
    -- is not differentiated, and recorded. Every run uses a fresh solver (as in
    ode), with forward sensitivities at the protocol's tolerances and AMICI's own
    defaults otherwise (its corrector is ``simultaneous``). AMICI observes the
    ``.net``'s observables, as weighted sums of species, with a Gaussian noise model
    whose scale an ``ExpData`` sets (the adjoint gradient's objective).

    The parameters a multi-phase protocol sets are compiled as AMICI's fixed
    parameters, and the phases run in one of two ways:

    - **pre-simulation**, for an equilibration then a measured phase with no reset:
      AMICI's own, an ``ExpData`` with the first phase as ``t_presim`` (from
      ``t0 - t_presim`` to ``t0 = 0``) at ``fixed_parameters_presimulation``, the
      second at ``fixed_parameters``; AMICI carries the state and its sensitivities
      across ``t0`` itself. Not AMICI's steady-state pre-equilibration: a phase has a
      duration, which a model need not reach a steady state in (p38-ATF2 has an
      absorbing state its authors' 20000 s equilibration is far from);
    - **chained**, for any other: one AMICI simulation per phase, each at its fixed
      parameters from the state and the state sensitivities the last ended in
      (``set_initial_state``, ``set_initial_state_sensitivities``), a reset species
      given its new value and, as its sensitivity row, the value's derivative (both
      from the harness, ``reset_values``). Compiled without conservation laws, whose
      totals a reset would change.
    """

    NAME = "amici"
    INPUT = "sbml"

    def __init__(self, sbml: Path, species_names: list[str], name_to_id: dict[str, str],
                 proto: dict[str, Any], params: list[str], *,
                 schedule: dict[str, Any] | None = None,
                 observables: tuple[list[str], np.ndarray] | None = None):
        import amici
        from amici.sim import sundials as ss

        self._ss = ss
        want_ids = _align(species_names, name_to_id)
        self.proto, self.params = proto, list(params)
        self.schedule = schedule or _plain_schedule(proto)
        self.mode = amici_mode(self.schedule)
        fixed = list(self.schedule["switched"])
        # The observables, as the .net's weighted sums of species (no more: AMICI's
        # default observes every species, which only lengthens its build).
        names, weights = observables if observables is not None else ([], np.zeros((0, 0)))
        channels = []
        for o, (name, row) in enumerate(zip(names, weights)):
            terms = [f"{float(w)!r}*{sid}" for sid, w in zip(want_ids, row) if w != 0]
            channels.append(amici.MeasurementChannel(
                f"bench_obs{o}", name=name, formula=" + ".join(terms) or "0",
                noise_distribution="normal", sigma=1.0))
        self.n_observables = len(channels)
        outdir = sbml.parent / "amici_sens_model"
        t0 = time.perf_counter()
        amici.SbmlImporter(str(sbml)).sbml2amici(
            "bench_sens_model", str(outdir), generate_sensitivity_code=True, verbose=False,
            fixed_parameters=fixed or None, observation_model=channels,
            compute_conservation_laws=self.mode != "chained")
        self.codegen_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        self.model = amici.import_model_module("bench_sens_model", str(outdir)).get_model()
        self.model.set_timepoints(times(proto))
        free = list(self.model.get_free_parameter_ids())
        ids = []
        for p in self.params:
            ids.append(p if p in free else (f"amici_{p}" if f"amici_{p}" in free else None))
        self.covered = np.array([i is not None for i in ids], bool)
        if not self.covered.any():
            raise RuntimeError("AMICI exposes none of the parameters as free parameters")
        self.model.set_parameter_scale(ss.ParameterScaling.none)
        self.model.set_parameter_list([free.index(i) for i in ids if i is not None])
        self._fixed = self._fixed_order(fixed)
        if fixed:
            self.model.set_fixed_parameters(self._fixed_values(self.schedule["phases"][0]))
        self.load_seconds = time.perf_counter() - t1
        pos = {sid: i for i, sid in enumerate(self.model.get_state_ids())}
        unmapped = [i for i in want_ids if i not in pos]
        if unmapped:
            raise RuntimeError(
                f"cannot align {len(unmapped)} SBML ids to AMICI state ids "
                f"(e.g. {unmapped[:3]}); the SBML export and AMICI disagree")
        self.idx = [pos[i] for i in want_ids]
        self.uncovered = _not_covered(self.params, self.covered,
                                      "not a free parameter of the AMICI model")
        if self.mode == "chained":
            # The network's own initial state and its sensitivities, which every
            # chained run starts its first phase from (a custom state persists).
            nx, n_k = len(pos), int(self.covered.sum())
            self.x0 = np.asarray(self.model.get_initial_state(), dtype=float)
            self.sx0 = np.asarray(self.model.get_initial_state_sensitivities(),
                                  dtype=float).reshape(n_k, nx)
        self.edata = self._new_edata() if self.mode == "presimulation" else None

    def _fixed_order(self, fixed: list[str]) -> list[str]:
        """The protocol's parameters in AMICI's fixed-parameter order."""
        ids = list(self.model.get_fixed_parameter_ids())
        name = {}
        for p in fixed:
            aid = p if p in ids else (f"amici_{p}" if f"amici_{p}" in ids else None)
            if aid is None:
                raise RuntimeError(f"AMICI has no fixed parameter for {p}, which the protocol sets")
            name[aid] = p
        missing = [i for i in ids if i not in name]
        if missing:
            raise RuntimeError(f"AMICI has fixed parameters the protocol does not set: {missing}")
        return [name[i] for i in ids]

    def _fixed_values(self, phase: dict[str, Any]) -> list[float]:
        return [float(phase["values"][p]) for p in self._fixed]

    def _new_edata(self):
        """A fresh ``ExpData``: for a pre-simulated protocol, its two phases."""
        edata = self._ss.ExpData(self.model)
        if self.mode == "presimulation":
            pre, measured = self.schedule["phases"]
            edata.t_presim = float(pre["duration"])
            edata.fixed_parameters_presimulation = self._fixed_values(pre)
            edata.fixed_parameters = self._fixed_values(measured)
        return edata

    def _solver(self, method):
        ss = self._ss
        solver = self.model.create_solver()
        solver.set_relative_tolerance(float(self.proto["rtol"]))
        solver.set_absolute_tolerance(float(self.proto["atol"]))
        solver.set_sensitivity_order(ss.SensitivityOrder.first)
        solver.set_sensitivity_method(method)
        return solver

    def adjoint_problem(self, data: np.ndarray, sigma: np.ndarray):
        """The ``ExpData`` of a Gaussian log-likelihood: ``data`` ``(n_t, n_obs)``
        measured with noise scales ``sigma`` (the same shape). Not for a chained
        protocol: AMICI's adjoint runs backward through one simulation, and a chain
        is several."""
        if self.mode == "chained":
            raise ValueError("UNSUPPORTED: AMICI's adjoint cannot run back through a "
                             "protocol the harness chains from several simulations")
        edata = self._new_edata()
        edata.set_measurements(np.asarray(data, float).ravel().tolist())
        edata.set_noise_scales(np.asarray(sigma, float).ravel().tolist())
        return edata

    def adjoint_gradient(self, edata) -> np.ndarray:
        """The gradient of the negative log-likelihood ``edata`` defines, by AMICI's
        adjoint sensitivities (one backward solve, whatever the number of
        parameters): ``(n_p,)``, NaN for a parameter AMICI does not differentiate."""
        rdata = self._ss.run_simulation(self.model,
                                        self._solver(self._ss.SensitivityMethod.adjoint), edata)
        if rdata.status != 0:
            raise RuntimeError(f"AMICI adjoint simulation failed with status {rdata.status}")
        grad = np.full(len(self.params), np.nan)
        grad[self.covered] = -np.asarray(rdata.sllh, dtype=float)
        return grad

    def _simulate(self, edata=None):
        rdata = self._ss.run_simulation(self.model,
                                        self._solver(self._ss.SensitivityMethod.forward), edata)
        if rdata.status != 0:
            raise RuntimeError(f"AMICI simulation failed with status {rdata.status}")
        return rdata

    def run(self):
        if self.mode != "chained":
            return [self._simulate(self.edata)]
        measured = []
        x, sx = self.x0.copy(), self.sx0.copy()
        for k, ph in enumerate(self.schedule["phases"]):
            if self._fixed:
                self.model.set_fixed_parameters(self._fixed_values(ph))
            for i, value, grad in ph.get("reset_values", []):
                x[self.idx[i]] = value
                sx[:, self.idx[i]] = np.asarray(grad, dtype=float)[self.covered]
            end = ph["start"] + ph["duration"]
            n = ph["n_steps"]
            self.model.set_t0(ph["start"])
            self.model.set_timepoints(np.linspace(ph["start"], end, n + 1).tolist()
                                      if n is not None else [end])
            self.model.set_initial_state(x.tolist())
            self.model.set_initial_state_sensitivities(sx.ravel().tolist())
            rdata = self._simulate()
            x = np.asarray(rdata.x, dtype=float)[-1].copy()
            sx = np.asarray(rdata.sx, dtype=float)[-1].copy()
            if n is not None:
                measured.append(rdata)
        return measured

    def tensors(self, rdatas) -> tuple[np.ndarray, np.ndarray]:
        x = np.concatenate([np.asarray(r.x, dtype=float)[:, self.idx] for r in rdatas])
        sx = np.concatenate([np.transpose(np.asarray(r.sx, dtype=float), (0, 2, 1))[:, self.idx, :]
                             for r in rdatas])
        # A success status does not guarantee finite numbers (as in ode).
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(sx))):
            raise RuntimeError("AMICI returned non-finite values with a success status")
        S = np.full((x.shape[0], x.shape[1], len(self.params)), np.nan)
        S[:, :, self.covered] = sx
        return S, x

    def config(self) -> dict[str, Any]:
        return {
            "integrator": "CVODES BDF, forward sensitivities",
            "version": env.amici_identity().get("version"),
            "parameter_scale": "linear",
            "internal_sensitivity_method": "AMICI default (simultaneous)",
            "phases": self.mode,
            "conservation_laws": self.mode != "chained",
            "not_differentiated": self.uncovered,
            "load_seconds": self.load_seconds,
            "codegen_seconds": self.codegen_seconds,
        }


def amici_mode(sched: dict[str, Any]) -> str:
    """How AMICI runs a protocol (``AmiciSens``): ``single``, ``presimulation`` (an
    unmeasured phase then a measured one, no resets) or ``chained``."""
    phases = sched["phases"]
    if len(phases) == 1 and not phases[0]["resets"]:
        return "single"
    if (len(phases) == 2 and phases[0]["n_steps"] is None and phases[1]["n_steps"] is not None
            and not phases[0]["resets"] and not phases[1]["resets"]):
        return "presimulation"
    return "chained"


class CopasiSens:
    """COPASI's time-course sensitivities task, on the SBML exported from the ``.net``.

    Loaded through BASICO, then driven through COPASI's own API: the task's
    parameters are the model values' initial values (by SBML id), its targets the
    species' concentrations (an amount, for the unit compartment the export writes).
    The task's result array holds the sensitivities at the current time only, so a
    data handler records every element of it, and every species, at each output
    time. A parameter COPASI has no model value for is not differentiated, and
    recorded.
    """

    NAME = "copasi"
    INPUT = "sbml"

    @staticmethod
    def not_applicable(prep: dict[str, Any]) -> str | None:
        """Why COPASI cannot run this protocol, or None. Its time-course
        sensitivities task starts every sensitivity from the initial state's own
        derivative; it has no way to begin from an earlier phase's."""
        sched = prep.get("schedule")
        if sched and not sched["plain"]:
            return ("a multi-phase protocol: COPASI's time-course sensitivities task cannot "
                    "carry one phase's sensitivities into the next")
        return None

    def __init__(self, sbml: Path, species_names: list[str], name_to_id: dict[str, str],
                 proto: dict[str, Any], params: list[str], *,
                 schedule: dict[str, Any] | None = None):
        import basico

        if schedule and not schedule["plain"]:
            raise ValueError("COPASI's time-course sensitivities run a single phase")
        import COPASI

        self._copasi = COPASI
        ids = _align(species_names, name_to_id)
        self.proto, self.params = proto, list(params)
        t_end, n_steps = _first_grid(proto)
        self.n_t = n_steps + 1
        t0 = time.perf_counter()
        self.dm = basico.load_model(str(sbml))
        if self.dm is None:
            raise RuntimeError("COPASI could not import the exported SBML")
        model = self.dm.getModel()
        self.task = self.dm.getTask("Time-Course Sensitivities")
        if self.task is None:
            raise RuntimeError("this COPASI has no time-course sensitivities task")
        problem = self.task.getProblem()
        problem.clearParameterCNs()
        problem.clearTargetCNs()
        values = {model.getModelValue(j).getSBMLId(): model.getModelValue(j)
                  for j in range(model.getNumModelValues())}
        self.covered = np.array([p in values for p in self.params], bool)
        if not self.covered.any():
            raise RuntimeError("COPASI has a model value for none of the parameters")
        self.uncovered = _not_covered(self.params, self.covered, "no COPASI model value")
        for p, c in zip(self.params, self.covered):
            if c:
                problem.addParameterCN(self._cn(values[p].getInitialValueReference()))
        metabs = {model.getMetabolite(j).getSBMLId(): model.getMetabolite(j)
                  for j in range(model.getNumMetabs())}
        missing = [i for i in ids if i not in metabs]
        if missing:
            raise RuntimeError(f"COPASI's model lacks species {missing[:3]}")
        for i in ids:
            problem.addTargetCN(self._cn(metabs[i].getConcentrationReference()))
        problem.setOutputStartTime(0.0)
        problem.setDuration(t_end)
        problem.setStepNumber(n_steps)
        problem.setAutomaticStepSize(False)
        problem.setTimeSeriesRequested(False)
        method = self.task.getMethod()
        method.getParameter("Relative Tolerance").setDblValue(float(proto["rtol"]))
        method.getParameter("Absolute Tolerance").setDblValue(float(proto["atol"]))
        self.method_name = method.getObjectName()
        # What the handler records at each output time: the time, every element of
        # the targets x parameters array, then every species.
        array = problem.getTargetsResultAnnotated()
        n_sp, n_k = len(ids), int(self.covered.sum())
        names = [self._cn(model.getValueReference())]
        for i in range(n_sp):
            for k in range(n_k):
                index = COPASI.SizeTStdVector()
                index.push_back(i)
                index.push_back(k)
                names.append(self._cn(array.addElementReference(index)))
        names += [self._cn(metabs[i].getConcentrationReference()) for i in ids]
        self._names, self._shape = names, (n_sp, n_k)
        self.load_seconds = time.perf_counter() - t0

    def _cn(self, obj):
        return self._copasi.CRegisteredCommonName(obj.getCN().getString())

    def _fail(self, what: str, before: int):
        cm = self._copasi.CCopasiMessage
        msgs = cm.getAllMessageText() if cm.size() > before else ""
        raise RuntimeError(f"COPASI {what} failed: {msgs.strip()[:300]}")

    def run(self):
        cp = self._copasi
        handler = cp.CDataHandler()
        for name in self._names:
            handler.addDuringName(name)
        before = cp.CCopasiMessage.size()
        if not self.task.initializeRawWithOutputHandler(cp.CCopasiTask.OUTPUT_UI, handler):
            self._fail("time-course sensitivities initialisation", before)
        if not self.task.processRaw(True):
            self._fail("time-course sensitivities run", before)
        rows = handler.getNumRowsDuring()
        return np.array([list(handler.getNthRow(i)) for i in range(rows)], dtype=float)

    def tensors(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_sp, n_k = self._shape
        n_t = self.n_t
        if data.shape != (n_t, 1 + n_sp * n_k + n_sp):
            raise RuntimeError(f"COPASI recorded {data.shape[0]} time points of width "
                               f"{data.shape[1] if data.ndim == 2 else '?'}, expected {n_t}")
        S = np.full((n_t, n_sp, len(self.params)), np.nan)
        S[:, :, self.covered] = data[:, 1:1 + n_sp * n_k].reshape(n_t, n_sp, n_k)
        return S, data[:, 1 + n_sp * n_k:]

    def config(self) -> dict[str, Any]:
        return {
            "method": f"time-course sensitivities ({self.method_name})",
            "version": env.copasi_identity().get("version"),
            "output": "amount",
            "not_differentiated": self.uncovered,
            "load_seconds": self.load_seconds,
        }


def _first_grid(proto: dict[str, Any]) -> tuple[float, int]:
    """The first measured phase's ``(duration, n_steps)``."""
    ph = next(ph for ph in phase_plan(proto) if ph["n_steps"] is not None)
    return ph["duration"], ph["n_steps"]


def run_network_command(net: Path, proto: dict[str, Any], out_prefix: Path, *,
                        rtol: float, atol: float,
                        grid: tuple[float, int] | None = None) -> list[str]:
    """The run_network call BNG2.pl makes for an ODE simulate, at the given tolerances,
    over ``grid`` -- ``(t_end, n_steps)`` from 0 -- or the first measured phase's."""
    t_end, n_steps = grid or _first_grid(proto)
    return [
        *env.run_network_launch(),
        str(env.run_network_bin()), "-o", str(out_prefix), "-p", "cvode",
        "-a", repr(float(atol)), "-r", repr(float(rtol)),
        "--cdat", "1", "--fdat", "0", "-g", str(net), str(net),
        repr(t_end / n_steps), str(n_steps),
    ]


class _RunNetworkProtocol:
    """run_network through the protocol, one fresh process per phase, as BNG2.pl runs
    a multi-phase protocol: each phase's ``.net`` has the parameter values in force
    during it, and, from the second on, the state the last phase ended in as its
    initial amounts (13 significant digits, as run_network writes them). A reset
    species' amount is its expression, added to the parameters block as a derived
    parameter (``_bench_reset_<n>``), so run_network evaluates it from the very
    parameters it was handed -- at each perturbed point, for the finite
    differences: the derivative of a reset follows from the file, not from the
    harness. run_network starts each phase's clock at 0, which is why a multi-phase
    protocol refuses a network that reads it."""

    def _setup(self, net: Path, proto: dict[str, Any], workdir: Path, *, timeout: float,
               rtol: float, atol: float, schedule: dict[str, Any] | None):
        self.text = net.read_text(encoding="utf-8")
        self.initial = netfile.structure(self.text).initial
        self.proto, self.workdir, self.timeout = proto, workdir, timeout
        self.rtol, self.atol = float(rtol), float(atol)
        self.schedule = schedule or _plain_schedule(proto)
        workdir.mkdir(parents=True, exist_ok=True)
        self.solves, self.solve_seconds, self.drift = 0, 0.0, []

    def _process(self, text: str, t_end: float, n_steps: int) -> np.ndarray:
        net = self.workdir / "fd.net"
        net.write_text(text, encoding="utf-8")
        cmd = run_network_command(net, self.proto, self.workdir / "fd", rtol=self.rtol,
                                  atol=self.atol, grid=(t_end, n_steps))
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout,
                              check=False)
        self.solve_seconds += time.perf_counter() - t0
        self.solves += 1
        if proc.returncode != 0:
            tail = (proc.stderr.strip() or proc.stdout.strip()).splitlines()
            raise RuntimeError(f"run_network failed: {(tail[-1] if tail else '')[:300]}")
        cdat = self.workdir / "fd.cdat"
        _, values, _ = read_dat(cdat)
        cdat.unlink()
        if values.shape[0] != n_steps + 1:
            raise RuntimeError(f"run_network wrote {values.shape[0]} time points, "
                               f"expected {n_steps + 1}")
        return values

    def _solve(self, text: str) -> np.ndarray:
        """The protocol's measured trajectory ``(n_t, n_species)`` for this ``.net``:
        every measured phase's rows, in order."""
        measured, state, self.drift = [], None, []
        for k, ph in enumerate(self.schedule["phases"]):
            phase_text = netfile.with_parameters(text, ph["values"]) if ph["values"] else text
            if k > 0 or ph["resets"]:
                amounts: list[float | str] = list(self.initial) if state is None else list(state)
                added = {}
                for j, (i, expr) in enumerate(ph["resets"]):
                    name = f"_bench_reset_{j}"
                    added[name] = expr
                    amounts[i] = name
                if added:
                    phase_text = netfile.with_added_parameters(phase_text, added)
                phase_text = netfile.with_initial_amounts(phase_text, amounts)
            n = ph["n_steps"]
            values = self._process(phase_text, ph["duration"], n if n is not None else 2)
            if n is None:
                self.drift.append(_drift(values))
            else:
                measured.append(values)
            state = values[-1]
        return np.concatenate(measured)

    def _command(self) -> list[str]:
        cmd = recorded_command(run_network_command(
            self.workdir / "fd.net", self.proto, self.workdir / "fd", rtol=self.rtol,
            atol=self.atol))
        names = {str(self.workdir / "fd.net"): "<perturbed>.net", str(self.workdir / "fd"): "<out>"}
        return [names.get(c, c) for c in cmd]

    def _phases_config(self) -> dict[str, Any]:
        if self.schedule["plain"]:
            return {}
        return {"phases": {"n": len(self.schedule["phases"]),
                           "processes_per_solve": len(self.schedule["phases"]),
                           "drift_last_half_of_unmeasured": self.drift}}


class FdRunNetwork(_RunNetworkProtocol):
    """Central finite differences from run_network, an engine independent of BNGsim.

    For each parameter, the ``.net`` is written with the parameter at ``p + h s``
    and ``p - h s`` (``s`` its scale: its value's magnitude, so a relative step, or
    the protocol's ``parameter_scales`` entry for a parameter at zero, an absolute
    one) and solved by fresh run_network processes, which re-derive every derived
    parameter, rate and initial amount from it -- so the derivative is the total
    one, through every chain the network has, and through every phase of the
    protocol (a reset's expression included). For a parameter at zero the lower point is negative:
    outside a dose's or a switch's physical range, but not the ODE's, whose
    right-hand side is as smooth there. Two steps, ``h`` and ``h/2``, at the
    protocol's ``fd_rtol``/``fd_atol``: ``4N + 1`` protocol solves in all.
    ``D(h/2)`` is the reference; ``|D(h) - D(h/2)|`` bounds its error
    (``bench.sensitivity``).
    """

    NAME = "run_network_fd"
    INPUT = "net"

    def __init__(self, net: Path, proto: dict[str, Any], params: list[str], values: np.ndarray,
                 scales: np.ndarray, workdir: Path, *, timeout: float,
                 schedule: dict[str, Any] | None = None):
        self._setup(net, proto, workdir, timeout=timeout, rtol=proto["fd_rtol"],
                    atol=proto["fd_atol"], schedule=schedule)
        self.params = list(params)
        self.values, self.scales = np.asarray(values, float), np.asarray(scales, float)
        self.covered = np.ones(len(self.params), bool)

    def run(self) -> np.ndarray:
        """``pack(D(h/2), D(h), x)``: the finer quotient first."""
        base = self._solve(self.text)
        h = float(self.proto["fd_step"])
        quotients = []
        for step in (h / 2, h):
            cols = []
            for name, p, s in zip(self.params, self.values, self.scales):
                up = self._solve(netfile.with_parameters(self.text, {name: p + step * s}))
                down = self._solve(netfile.with_parameters(self.text, {name: p - step * s}))
                cols.append((up - down) / (2 * step * s))
            quotients.append(np.stack(cols, axis=2))
        return pack(quotients[0], base, quotients[1])

    def config(self) -> dict[str, Any]:
        return {
            "method": "central differences at steps h and h/2 (relative to each "
                      "parameter's scale)",
            "engine": "run_network (fresh process per solve)",
            "step": float(self.proto["fd_step"]),
            "rtol": self.rtol,
            "atol": self.atol,
            "command": self._command(),
            "solves": self.solves,
            "solve_seconds": self.solve_seconds,
            **self._phases_config(),
        }


class RunNetworkPlain(_RunNetworkProtocol):
    """run_network's plain solve of the protocol at its own tolerances, as BNG2.pl
    runs it: the unit of a finite-difference gradient, timed (a fresh process per
    phase, its load inside every solve)."""

    def __init__(self, net: Path, proto: dict[str, Any], workdir: Path, *, timeout: float,
                 schedule: dict[str, Any] | None = None):
        self._setup(net, proto, workdir, timeout=timeout, rtol=proto["rtol"], atol=proto["atol"],
                    schedule=schedule)

    def run(self) -> np.ndarray:
        return self._solve(self.text)

    warm = run

    @staticmethod
    def trajectory(values: np.ndarray) -> np.ndarray:
        return np.asarray(values)

    def config(self) -> dict[str, Any]:
        return {"command": self._command(), **self._phases_config()}


# The reference engines BNGsim is compared against, in a fixed order.
REFERENCE_ENGINES: dict[str, type] = {
    AmiciSens.NAME: AmiciSens,
    FdRunNetwork.NAME: FdRunNetwork,
    CopasiSens.NAME: CopasiSens,
}
FD = FdRunNetwork.NAME

BUILD_STAGE = {
    "bngsim": "load + codegen (sensitivity RHS)",
    "bngsim_simultaneous": "load + codegen (sensitivity RHS)",
    "amici": "SBML import + sensitivity code generation + C++ compile",
    "run_network_fd": "perturbed solves (subprocesses)",
    "copasi": "SBML import",
}


def _build_reference(name: str, prep: dict[str, Any], workdir: Path, timeout: float):
    cls = REFERENCE_ENGINES[name]
    if cls.INPUT == "net":
        return cls(prep["net"], prep["proto"], prep["params"], prep["values"], prep["scales"],
                   workdir, timeout=timeout, schedule=prep["schedule"])
    # AMICI observes the .net's observables (for the adjoint gradient, D7).
    extra = ({"observables": (prep["observables"], prep["weights"])}
             if name == AmiciSens.NAME else {})
    return cls(prep["sbml"], prep["species_names"], prep["name_to_id"], prep["proto"],
               prep["params"], schedule=prep["schedule"], **extra)


def not_applicable(prep: dict[str, Any]) -> dict[str, str]:
    """The references that do not apply to this problem's protocol, with the reason:
    named in the record, not run and not counted (``not_applicable``)."""
    out = {}
    for name, cls in REFERENCE_ENGINES.items():
        check = getattr(cls, "not_applicable", None)
        reason = check(prep) if check else None
        if reason:
            out[name] = reason
    return out


def _progress(msg: str) -> None:
    progress(CAPABILITY, msg)


# --------------------------------------------------------------------------- #
# Preparation and the cold runs of the correctness job
# --------------------------------------------------------------------------- #
def _prepare(problem: Problem, ctx: dict[str, Any], setting: str = DEFAULT_SETTING
             ) -> dict[str, Any]:
    proto = protocol(problem, setting)
    prep = prepare_network(problem, ctx, proto["netgen"])
    structure = netfile.read_structure(prep["net"])
    if structure.species != [n.lstrip("$") for n in prep["species_names"]]:
        raise RuntimeError("the .net's species and BNGsim's species_names disagree")
    scales = proto.get("parameter_scales") or {}
    sched = schedule(structure, proto)
    params, excluded = select_parameters(structure, proto["parameters"],
                                         fitted=proto.get("fitted"), scales=scales,
                                         switched=set(sched["switched"]))
    if not params:
        raise ValueError(f"no parameter to differentiate ({proto['parameters']!r}; "
                         f"left out: {excluded})")
    observables, weights = structure.observable_weights()
    prep.update(proto=proto, structure=structure, params=params,
                values=np.array([structure.literal(p) for p in params]),
                scales=parameter_scales(structure, params, scales),
                observables=observables, weights=weights, schedule=sched)
    reset_values(Path(prep["net"]).read_text(encoding="utf-8"), sched, params, prep["scales"])
    prep["info"]["parameters"] = {
        "selection": proto["parameters"], "convention": convention(proto), "n": len(params),
        "names": params, "excluded": excluded,
        "scales": {p: float(scales[p]) for p in params if p in scales},
    }
    if not sched["plain"]:
        prep["info"]["protocol_phases"] = phases_info(sched, structure.species)
    prep["info"]["network"]["observables"] = len(observables)
    return prep


def _cold_bngsim(prep: dict[str, Any], name: str):
    def fn(stage):
        t0 = time.perf_counter()
        stage(BUILD_STAGE.get(name, "build"))
        engine = BngsimSens(prep["net"], prep["proto"], prep["params"],
                            schedule=prep["schedule"], **CONFIGURATIONS[name])
        stage("first solve")
        t1 = time.perf_counter()
        result = engine.run()
        solve_seconds = time.perf_counter() - t1
        S, X = engine.tensors(result)
        config = {**engine.config(), "solve_seconds": solve_seconds,
                  "cold_seconds": time.perf_counter() - t0,
                  "observables_check": _observables_check(prep, engine, result, S, X)}
        return pack(S, X), config
    return fn


def _observables_check(prep, engine, result, S, X) -> dict[str, Any]:
    """BNGsim's own observable sensitivities against the ones the harness forms from
    its species sensitivities, by the same verdict (``passed`` gates the job)."""
    names, own = engine.observable_sensitivities(result)
    want = prep["observables"]
    if not want:
        return {"passed": True, "n_observables": 0}
    if names != want:
        return {"passed": False, "reason": f"bngsim's observables {names[:5]} are not the "
                                           f"network's {want[:5]}"}
    derived = np.einsum("oi,tik->tok", prep["weights"], S)
    y = X @ prep["weights"].T
    c = sv.compare(sv.scaled(own, prep["scales"]), sv.scaled(derived, prep["scales"]),
                   np.abs(y).max(axis=0), atol=float(prep["proto"]["atol"]), verdict=VERDICT)
    s = c.summary([f"observable:{n}" for n in want], prep["params"], times(prep["proto"]),
                  n_species=0)
    return {k: s[k] for k in ("passed", "max_error", "n_judged", "n_failed") if k in s}


def _cold_reference(name: str, prep: dict[str, Any], workdir: Path, timeout: float):
    def fn(stage):
        t0 = time.perf_counter()
        stage(BUILD_STAGE.get(name, "build"))
        ref = _build_reference(name, prep, workdir, timeout)
        stage("first solve")
        t1 = time.perf_counter()
        raw = ref.run()
        solve_seconds = time.perf_counter() - t1
        packed = raw if name == FD else pack(*ref.tensors(raw))
        return packed, {**ref.config(), "covered": [bool(c) for c in ref.covered],
                        "solve_seconds": solve_seconds,
                        "cold_seconds": time.perf_counter() - t0}
    return fn


# --------------------------------------------------------------------------- #
# Verdicts
# --------------------------------------------------------------------------- #
class _Run:
    """One engine's result on the outputs (species then observables)."""

    def __init__(self, prep: dict[str, Any], S: np.ndarray, X: np.ndarray,
                 covered: np.ndarray, estimate: np.ndarray | None = None):
        S_out, self.y = sv.outputs(S, X, prep["weights"])
        self.sig = sv.scaled(S_out, prep["scales"])
        self.y_peak = np.max(np.where(np.isfinite(self.y), np.abs(self.y), 0.0), axis=0)
        self.covered = np.asarray(covered, bool)
        self.estimate = estimate  # FD only: the two-step error estimate, semi-scaled


def _run_from(prep: dict[str, Any], name: str, packed: np.ndarray, config: dict[str, Any]) -> _Run:
    n_p = len(prep["params"])
    tensors, X = unpack(packed, n_p)
    covered = np.asarray(config.get("covered", [True] * n_p), bool)
    if name == FD:
        fine, coarse = tensors
        fine_out, _ = sv.outputs(fine, X, prep["weights"])
        coarse_out, _ = sv.outputs(coarse, X, prep["weights"])
        est = sv.fd_estimate(sv.scaled(coarse_out, prep["scales"]),
                             sv.scaled(fine_out, prep["scales"]))
        return _Run(prep, fine, X, covered, est)
    return _Run(prep, tensors[0], X, covered)


def _judge(a: _Run, b: _Run, atol: float) -> sv.Comparison:
    """``a`` against ``b``, on the columns both produced (and FD resolves)."""
    y_peak = np.fmax(a.y_peak, b.y_peak)
    judged = np.broadcast_to(a.covered & b.covered, a.sig.shape[1:]).copy()
    for this, other in ((a, b), (b, a)):
        if this.estimate is not None:
            judged &= sv.fd_resolved(this.estimate, other.sig, this.sig, y_peak,
                                     atol=atol, verdict=VERDICT)
    return sv.compare(a.sig, b.sig, y_peak, atol=atol, verdict=VERDICT, judged=judged)


def _output_names(prep: dict[str, Any]) -> list[str]:
    return [*prep["species_names"], *(f"observable:{n}" for n in prep["observables"])]


def trajectory_error(a: np.ndarray, b: np.ndarray, atol: float) -> float | None:
    """The two engines' species trajectories' worst disagreement, as a fraction of
    each species' peak in either, floored at 100 state atols: a species one engine
    holds at 0 and the other at rounding noise (1e-21) is not a disagreement of
    order one. A diagnostic beside the verdict, not part of it."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not a.size:
        return None
    scale = np.maximum(np.maximum(np.nanmax(np.abs(a), axis=0), np.nanmax(np.abs(b), axis=0)),
                       100.0 * atol)
    return float(np.nanmax(np.nanmax(np.abs(a - b), axis=0) / scale))


def _summary(c: sv.Comparison, prep: dict[str, Any], a: _Run, b: _Run) -> dict[str, Any]:
    out = c.summary(_output_names(prep), prep["params"], times(prep["proto"]),
                    n_species=len(prep["species_names"]))
    n_sp = len(prep["species_names"])
    out["trajectory_max_err_over_peak"] = trajectory_error(
        a.y[:, :n_sp], b.y[:, :n_sp], float(prep["proto"]["atol"]))
    return out


def verdicts(prep: dict[str, Any], bn: _Run, refs: dict[str, _Run]
             ) -> tuple[dict[str, Any], dict[str, sv.Comparison]]:
    """BNGsim against each reference, and which references a disagreement refutes.

    Returns the JSON verdicts and the comparisons themselves (for coverage)."""
    atol = float(prep["proto"]["atol"])
    vs_bn = {name: _judge(bn, run, atol) for name, run in refs.items()}
    out = {name: _summary(c, prep, bn, refs[name]) for name, c in vs_bn.items()}
    for name, c in vs_bn.items():
        if c.passed or not c.judged.any():
            continue
        vouchers, by = [], []
        for other, c_other in vs_bn.items():
            if other == name:
                continue
            between = _judge(refs[other], refs[name], atol)
            vouchers.append((c_other.judged & ~c_other.failing, between.failing))
            if (c_other.judged & ~c_other.failing & between.failing & c.failing).any():
                by.append(other)
        if sv.refuted(c.failing, vouchers):
            out[name]["refuted_by"] = sorted(by)
    return out, vs_bn


def coverage(prep: dict[str, Any], bn: _Run, vs_bn: dict[str, sv.Comparison],
             verdict_json: dict[str, Any]) -> dict[str, Any]:
    """Which of BNGsim's columns some unrefuted, passing reference judged.

    A column that is negligible in BNGsim's own result (its peak under the verdict's
    floor) needs no reference: there is nothing to differentiate. Every other column
    must be judged by at least one reference that agreed with BNGsim, or the
    parameter it belongs to is ``unjudged``."""
    judged = np.zeros(bn.sig.shape[1:], bool)
    for name, c in vs_bn.items():
        if c.passed and "refuted_by" not in verdict_json[name]:
            judged |= c.judged
    floor = sv.column_scale(np.zeros_like(bn.sig), np.zeros_like(bn.sig), bn.y_peak,
                            atol=float(prep["proto"]["atol"]), verdict=VERDICT)
    significant = sv.peak(bn.sig) >= floor
    missing = significant & ~judged
    params = prep["params"]
    return {"n_columns": int(judged.size), "n_significant": int(significant.sum()),
            "n_judged": int((significant & judged).sum()),
            "unjudged": [p for k, p in enumerate(params) if missing[:, k].any()]}


def sens_outcome(bn_error: str | None, ref_errors: dict[str, str], n_refs: int,
                 verdict_json: dict[str, Any], bngsim_consistent: bool,
                 unjudged: list[str]) -> str:
    """The outcome, attributing a failure to BNGsim or the references.

    BNGsim failing is EXCEPTION (BAD_TEST when every reference failed too). A
    disagreement no other reference refutes, or a BNGsim configuration that
    disagrees with the default, is DIFF -- it takes precedence, so a BNGsim that
    disagrees with a reference is never timed. Otherwise REFERENCE_FAILED when a
    reference failed, was refuted, or a parameter went unjudged; else PASS.
    """
    if bn_error:
        return "BAD_TEST" if len(ref_errors) == n_refs else "EXCEPTION"
    diffs = [n for n, v in verdict_json.items()
             if v.get("n_judged") and not v.get("passed") and "refuted_by" not in v]
    if diffs or not bngsim_consistent:
        return "DIFF"
    refuted_any = any("refuted_by" in v for v in verdict_json.values())
    if ref_errors or refuted_any or unjudged or not verdict_json:
        return "REFERENCE_FAILED"
    return "PASS"


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
def run_correctness(job: Job, problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    try:
        prep = _prepare(problem, ctx, job.setting)
    except Exception as exc:  # noqa: BLE001 - a bad test input, a labelled row
        return {"outcome": "BAD_TEST", "error": f"{type(exc).__name__}: {exc}"}
    info = prep["info"]
    workdir = Path(ctx["workdir"])
    budget = float(ctx.get("engine_budget_seconds")
                   or engine_budget_seconds(problem, job.setting))
    info["engine_budget_seconds"] = budget
    timeout = min(float(ctx["subprocess_timeout"]), budget)

    def isolated(name: str, fn) -> dict[str, Any]:
        _progress(f"{name}: start (budget {budget:g} s; {threads()} threads in the worker)")
        out = run_isolated(fn, budget, workdir / "isolated" / name)
        _progress(f"{name}: {out.get('error') or 'done'} after {out['seconds']:.1f} s")
        return out

    # The BNGsim configurations, the default first.
    bn_runs: dict[str, _Run] = {}
    bn_errors: dict[str, str] = {}
    configs: dict[str, Any] = {}
    for name in CONFIGURATIONS:
        out = isolated(name, _cold_bngsim(prep, name))
        if "error" in out:
            bn_errors[name] = out["error"]
        else:
            configs[name] = out["config"]
            bn_runs[name] = _run_from(prep, name, out["trajectory"], out["config"])
    info["bngsim"] = configs.get(DEFAULT_CONFIGURATION, {})
    bn_error = bn_errors.get(DEFAULT_CONFIGURATION)

    references: dict[str, Any] = {}
    ref_runs: dict[str, _Run] = {}
    ref_errors: dict[str, str] = {}
    skipped = not_applicable(prep)
    for name in REFERENCE_ENGINES:
        if name in skipped:
            _progress(f"{name}: not applicable, {skipped[name]}")
            continue
        out = isolated(name, _cold_reference(name, prep, workdir / name, timeout))
        if "error" in out:
            ref_errors[name] = out["error"]
            continue
        references[name] = out["config"]
        try:
            ref_runs[name] = _run_from(prep, name, out["trajectory"], out["config"])
        except Exception as exc:  # noqa: BLE001 - an unreadable result is a failed reference
            ref_errors[name] = f"{type(exc).__name__}: {exc}"[:400]
    info["references"] = references

    verdict_json: dict[str, Any] = {}
    vs_bn: dict[str, sv.Comparison] = {}
    consistent = True
    if bn_error is None:
        bn = bn_runs[DEFAULT_CONFIGURATION]
        verdict_json, vs_bn = verdicts(prep, bn, ref_runs)
        atol = float(prep["proto"]["atol"])
        # Every other configuration against the default one; and BNGsim's own
        # observable sensitivities against the harness's.
        others: dict[str, Any] = {}
        for name, run in bn_runs.items():
            if name == DEFAULT_CONFIGURATION:
                continue
            others[name] = _summary(_judge(run, bn, atol), prep, run, bn)
            consistent &= bool(others[name]["passed"])
        info["configurations"] = others
        for name, cfg in configs.items():
            consistent &= bool(cfg.get("observables_check", {}).get("passed", True))
        info["coverage"] = coverage(prep, bn, vs_bn, verdict_json)
    elif len(ref_runs) > 1:
        # BNGsim gave nothing to judge: whether the references agree with each other
        # still says the protocol is well posed (evidence beside the outcome).
        names = list(ref_runs)
        info["references_agreement"] = {
            f"{a}_vs_{b}": _summary(_judge(ref_runs[a], ref_runs[b], float(prep["proto"]["atol"])),
                                    prep, ref_runs[a], ref_runs[b])
            for i, a in enumerate(names) for b in names[i + 1:]}

    # A further configuration that fails is BNGsim failing, as the default would be.
    if bn_error is None:
        bn_error = next((f"{name}: {err}" for name, err in bn_errors.items()), None)
    unjudged = info.get("coverage", {}).get("unjudged", [])
    result = sens_outcome(bn_error, ref_errors, len(REFERENCE_ENGINES) - len(skipped),
                          verdict_json, consistent, unjudged)
    if bn_error and result == "EXCEPTION" and \
            any(part.strip() == "TIMEOUT" for part in bn_error.split(":")[:2]):
        result = "TIMEOUT"
    if bn_error and result in ("EXCEPTION", "BAD_TEST") and \
            any(part.strip() == "UNSUPPORTED" for part in bn_error.split(":")[:2]):
        # bngsim refused the protocol (a feature it does not have), in its own words.
        result = "UNSUPPORTED"
    rec: dict[str, Any] = {"outcome": result, "measurements": info}
    if verdict_json:
        rec["verdicts"] = verdict_json
    if bn_error:
        rec["bngsim_error"] = bn_error
    for name, err in ref_errors.items():
        rec[f"{name}_error"] = err
    if skipped:
        rec["not_applicable"] = skipped
    return rec


def _role(name: str) -> str:
    """How an engine is timed, for the timing record (``gather`` reads it)."""
    if name in CONFIGURATIONS:
        return (f"warm, in-process (system under test, "
                f"{CONFIGURATIONS[name]['sensitivity_method']} corrector)")
    if name == "bngsim_ode":
        return "warm, in-process (plain solve: the unit of a finite-difference gradient)"
    if name == "bngsim_gradient":
        return ("warm, in-process (forward-sensitivity gradient of a synthetic "
                "log-likelihood)")
    if name == "amici_adjoint":
        return "warm, in-process (adjoint gradient of the same log-likelihood; timing only)"
    return engine_role(name)


# The plain solves timed beside the sensitivity solves: the unit of an FD gradient.
PLAIN = ("bngsim_ode", "run_network")

# --------------------------------------------------------------------------- #
# The gradient of a log-likelihood: AMICI's adjoint beside BNGsim's forward
# sensitivities (decision D7: timing only)
# --------------------------------------------------------------------------- #
# bngsim has no adjoint. What a modeller fitting N_p ~ 100 parameters would otherwise
# pay is AMICI's adjoint gradient: one forward and one backward solve, whatever N_p.
# The objective is a synthetic Gaussian negative log-likelihood of the .net's
# observables -- the cost of a gradient does not depend on what the data are, only on
# the residuals not being zero: data at GRADIENT_DATA_FACTOR times BNGsim's nominal
# observables, each observable's noise scale GRADIENT_SIGMA_FRACTION of its peak.
GRADIENT = ("bngsim_gradient", "amici_adjoint")
GRADIENT_DATA_FACTOR = 0.9
GRADIENT_SIGMA_FRACTION = 0.1


def synthetic_likelihood(y: np.ndarray, atol: float) -> tuple[np.ndarray, np.ndarray]:
    """The data and noise scales ``(n_t, n_obs)`` of the synthetic log-likelihood, from
    the nominal observables ``y``: a constant scale per observable, floored at 100
    atols so an observable that never moves still has one."""
    y = np.asarray(y, dtype=float)
    sigma = np.maximum(GRADIENT_SIGMA_FRACTION * np.max(np.abs(y), axis=0), 100.0 * atol)
    return GRADIENT_DATA_FACTOR * y, np.broadcast_to(sigma, y.shape).copy()


class BngsimGradient:
    """BNGsim's gradient of the synthetic negative log-likelihood: a forward-sensitivity
    solve of the protocol, then ``Result.neg_log_likelihood_gradient`` on the
    observables (both inside the timed call, as a fit pays both)."""

    def __init__(self, engine: BngsimSens, data: np.ndarray, sigma: np.ndarray):
        self.engine, self.data, self.sigma = engine, data, sigma

    def run(self) -> np.ndarray:
        _, grad = self.engine.run().neg_log_likelihood_gradient(self.data, self.sigma,
                                                                target="observables")
        return np.asarray(grad, dtype=float)[None, :]


class AmiciAdjoint:
    """AMICI's adjoint gradient of the same negative log-likelihood, on the model AMICI
    built for its forward sensitivities (one build, so no pre-sim cost of its own)."""

    def __init__(self, engine: AmiciSens, data: np.ndarray, sigma: np.ndarray):
        self.engine = engine
        self.edata = engine.adjoint_problem(data, sigma)

    def run(self) -> np.ndarray:
        return self.engine.adjoint_gradient(self.edata)[None, :]


def gradient_agreement(bn: np.ndarray, amici: np.ndarray, scales: np.ndarray) -> dict[str, Any]:
    """How far apart the two gradients are, recorded beside their times (not a verdict):
    the largest difference of the semi-scaled gradient ``scale * dL/dp`` over the
    parameters AMICI differentiates, as a fraction of the largest BNGsim component."""
    a, b = np.asarray(amici, float).ravel() * scales, np.asarray(bn, float).ravel() * scales
    ok = np.isfinite(a)
    peak = float(np.max(np.abs(b[ok]))) if ok.any() else 0.0
    err = float(np.max(np.abs(a[ok] - b[ok]))) / peak if peak > 0 else None
    return {"max_rel_difference": err, "n_parameters": int(ok.sum()),
            "note": "timing only: the gradients are compared to show the two engines "
                    "differentiate the same objective, not judged"}


def _capture(engine, scales: np.ndarray):
    """What a timed repetition is checked on: its semi-scaled sensitivities, one
    column per (species, parameter) -- extracted outside the timed region. A
    parameter the engine does not differentiate is a column of zeros."""
    def capture(raw):
        sig = sv.scaled(engine.tensors(raw)[0], scales)
        return np.nan_to_num(sig.reshape(sig.shape[0], -1), nan=0.0)
    return capture


def run_timing(job: Job, problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    try:
        prep = _prepare(problem, ctx, job.setting)
    except Exception as exc:  # noqa: BLE001 - a bad test input, a labelled row
        return {"outcome": "BAD_TEST", "error": f"{type(exc).__name__}: {exc}"}
    proto, info = prep["proto"], prep["info"]
    workdir = Path(ctx["workdir"])
    timeout = ctx["subprocess_timeout"]
    n = len(prep["params"])

    # BNGsim cold: load, compile and a first solve of the default configuration,
    # with the empty compile cache the worker gave this job. Its result anchors the
    # warm repetitions.
    phases = prep["schedule"]
    t0 = time.perf_counter()
    bngsim_engine = BngsimSens(prep["net"], proto, prep["params"], schedule=phases,
                               **CONFIGURATIONS[DEFAULT_CONFIGURATION])
    first = bngsim_engine.run()
    info["bngsim_cold_seconds"] = time.perf_counter() - t0
    anchor = _capture(bngsim_engine, prep["scales"])(first)
    info["bngsim"] = bngsim_engine.config()
    engines: dict[str, Any] = {DEFAULT_CONFIGURATION: bngsim_engine}
    configs: dict[str, Any] = {}
    for name, cfg in CONFIGURATIONS.items():
        if name == DEFAULT_CONFIGURATION:
            continue
        eng = BngsimSens(prep["net"], proto, prep["params"], schedule=phases, **cfg)
        eng.tensors(eng.run())  # trial run
        engines[name] = eng
        configs[name] = eng.config()
    # The unit of a finite-difference gradient: one plain solve of the protocol, by
    # BNGsim and by run_network (a fresh process per phase, as BNG2.pl and a
    # finite-difference fit run it), at the protocol's tolerances.
    plain = BngsimPlain(prep["net"], proto, schedule=phases)
    plain.run()  # trial run
    engines["bngsim_ode"] = plain
    configs["bngsim_ode"] = plain.config()
    info["configurations"] = configs

    # Each reference: build and give it one trial run, so one that crashes is
    # recorded and dropped rather than taking the interleave down. The FD reference
    # is not timed; its cost is the derived baseline below.
    references: dict[str, Any] = {}
    ref_errors: dict[str, str] = {}
    timed_out = correctness_timeouts(job, ctx, REFERENCE_ENGINES)
    skipped = not_applicable(prep)
    try:
        rn = RunNetworkPlain(prep["net"], proto, workdir / "rn_plain", timeout=timeout,
                             schedule=phases)
        rn.run()
        engines["run_network"] = rn
        references["run_network"] = rn.config()
    except Exception as exc:  # noqa: BLE001
        ref_errors["run_network"] = f"{type(exc).__name__}: {exc}"[:400]
    for name in REFERENCE_ENGINES:
        if name == FD or name in skipped:
            continue
        if name in timed_out:
            ref_errors[name] = f"TIMEOUT in the correctness job, so not timed ({timed_out[name]})"
            _progress(f"{name}: skipped, {ref_errors[name]}")
            continue
        _progress(f"{name}: build + trial run")
        try:
            engine = _build_reference(name, prep, workdir / name, timeout)
            engine.tensors(engine.run())
            engines[name] = engine
            references[name] = engine.config()
        except Exception as exc:  # noqa: BLE001
            ref_errors[name] = f"{type(exc).__name__}: {exc}"[:400]
    # D7: the gradient of a log-likelihood, AMICI's adjoint beside BNGsim's forward
    # sensitivities, when the network has observables and AMICI survived its trial.
    adjoint: dict[str, Any] = {}
    if prep["observables"] and "amici" in engines:
        _progress("amici_adjoint: trial run")
        try:
            y = np.asarray(first.observables, dtype=float)
            data, sigma = synthetic_likelihood(y, float(proto["atol"]))
            grad_bn = BngsimGradient(bngsim_engine, data, sigma)
            grad_am = AmiciAdjoint(engines["amici"], data, sigma)
            g_bn, g_am = grad_bn.run(), grad_am.run()
            engines["bngsim_gradient"], engines["amici_adjoint"] = grad_bn, grad_am
            references["amici_adjoint"] = {"shared_build": "amici"}
            adjoint = {"objective": "Gaussian negative log-likelihood of the observables",
                       "data_factor": GRADIENT_DATA_FACTOR,
                       "sigma_fraction_of_peak": GRADIENT_SIGMA_FRACTION,
                       "observables": len(prep["observables"]),
                       "agreement": gradient_agreement(g_bn, g_am, prep["scales"])}
        except Exception as exc:  # noqa: BLE001 - a reference failure, recorded
            ref_errors["amici_adjoint"] = f"{type(exc).__name__}: {exc}"[:400]
    elif "amici" in engines:
        adjoint = {"skipped": "the network has no observables, so no likelihood"}
    info["references"] = references

    _progress(f"interleaved timing of {', '.join(engines)}")
    runs = {name: eng.run for name, eng in engines.items()}
    captures = {name: (type(eng).trajectory if name in PLAIN
                       else (lambda g: g) if name in GRADIENT
                       else _capture(eng, prep["scales"]))
                for name, eng in engines.items()}
    reps = round_robin_interleaved(runs, captures=captures, **job.definition["timing"])

    outputs = {name: reps["engines"][name].pop("outputs") for name in runs}
    repro: dict[str, Any] = {DEFAULT_CONFIGURATION: reproducibility(
        anchor, outputs[DEFAULT_CONFIGURATION], REPRO_TOL)}
    for name in engines:
        if name != DEFAULT_CONFIGURATION:
            outs = outputs[name]
            repro[name] = reproducibility(outs[0], outs[1:], REPRO_TOL)
    reproducible = all(r["passed"] for r in repro.values())

    times_ = absolute_time_stats({name: reps["engines"][name]["seconds"] for name in runs})
    # A finite-difference gradient costs N + 1 plain solves (forward differences) or
    # 2N (central): derived from the measured plain solves, not measured as such.
    baselines = {name: {"forward_seconds": (n + 1) * times_[name]["median_seconds"],
                        "central_seconds": 2 * n * times_[name]["median_seconds"],
                        "parameters": n}
                 for name in PLAIN if name in times_}
    rec: dict[str, Any] = {
        "outcome": "PASS" if reproducible else "NOT_REPRODUCIBLE",
        "measurements": {
            **info,
            "engine_roles": {name: _role(name) for name in engines},
            "repetitions": reps,
            "reproducibility": repro,
            "times": times_,
            "fd_baselines": baselines,
            **({"adjoint": adjoint} if adjoint else {}),
        },
    }
    for name, err in ref_errors.items():
        rec[f"{name}_error"] = err
    if skipped:
        rec["not_applicable"] = skipped
    return rec


def run(job: Job, problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    if job.kind == CORRECTNESS:
        return run_correctness(job, problem, ctx)
    if job.kind == TIMING:
        return run_timing(job, problem, ctx)
    raise ValueError(f"unknown job kind {job.kind!r}")
