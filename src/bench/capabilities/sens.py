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
The parameters are the protocol's ``parameters``: ``rate_constants`` (the default),
every parameter the ``.net`` writes as a number that some reaction rate depends on,
directly or through derived parameters and functions; ``all``, those that reach a
rate or an initial amount; or an explicit list. The harness reads that from the
``.net`` itself (``netfile.Structure``). A parameter whose value is zero (no relative
step) or that sizes a compartment is left out, and recorded with its reason.

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

**Timing.** The interleave times one forward-sensitivity solve per engine: each BNGsim
configuration, AMICI and COPASI, beside a plain solve of BNGsim (``bngsim_ode``) and
of run_network, the unit of a finite-difference gradient. The job reports each
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
    max_err_over_peak,
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

# The classes a protocol may name instead of a list (specs.py validates the same).
PARAMETER_CLASSES = SENS_PARAMETER_CLASSES
DEFAULT_PARAMETERS = "rate_constants"
# The finite differences: relative step h (and h/2), at tolerances tighter than any
# protocol's, so the integration's error over the step is small against the
# derivative (bench.sensitivity).
FD_DEFAULTS = {"fd_step": 1e-3, "fd_rtol": 1e-10, "fd_atol": 1e-12}

# The BNGsim configurations under test: CVODES' two correctors for the coupled
# state + sensitivity system. Codegen and the linear solver are bngsim's own choice
# (sensitivities always compile; there is no interpreted path to compare).
CONFIGURATIONS: dict[str, dict[str, str]] = {
    "bngsim": {"sensitivity_method": "staggered"},
    "bngsim_simultaneous": {"sensitivity_method": "simultaneous"},
}
DEFAULT_CONFIGURATION = "bngsim"


def protocol(problem: Problem) -> dict[str, Any]:
    """The sensitivity protocol with defaults filled in, as it enters a job definition."""
    p = dict(problem.protocol_for(CAPABILITY))
    p.setdefault("parameters", DEFAULT_PARAMETERS)
    for k, v in FD_DEFAULTS.items():
        p.setdefault(k, v)
    return p


def engines_id() -> str:
    """The ``engines`` id segment: BNGsim versus every registered reference."""
    return "bngsim-vs-" + "-".join(REFERENCE_ENGINES)


def jobs(problem: Problem) -> list[Job]:
    # BNGL only for now; an SBML source is where a sens-sbml suite would be admitted
    # (see the module docstring), with its own engines id.
    if problem.source.get("format") != "bngl":
        return []
    proto = protocol(problem)
    engines = engines_id()
    base = {
        "harness": f"{CAPABILITY}/{engines}/{HARNESS_VERSION}",
        "source_sha256": problem.source["sha256"],
        "protocol": proto,
        "setting": {"configurations": CONFIGURATIONS, "codegen": "auto", "linear_solver": "auto"},
        # As in ode: the SBML references' versions key only this capability;
        # run_network's rides the global fingerprint.
        "reference_versions": {
            "amici": env.amici_identity().get("version"),
            "copasi": env.copasi_identity().get("version"),
        },
    }
    est_c, est_t = planning_seconds(problem)
    backstop = (len(REFERENCE_ENGINES) + len(CONFIGURATIONS) + 1) * engine_budget_seconds(problem)
    correctness = Job(
        CAPABILITY, problem.id, engines, "auto", CORRECTNESS,
        definition={**base, "verdict": VERDICT},
        est_seconds=est_c,
        timeout_seconds=max(MIN_TIMEOUT_SECONDS, TIMEOUT_FACTOR * est_c, backstop),
    )
    timing = Job(
        CAPABILITY, problem.id, engines, "auto", TIMING,
        definition={**base, "timing": TIMING_PROTOCOL, "timing_check": REPRO_CHECK_VERSION},
        est_seconds=est_t,
        timeout_seconds=max(MIN_TIMEOUT_SECONDS, TIMEOUT_FACTOR * est_t),
        requires=correctness.id,
    )
    return [correctness, timing]


# --------------------------------------------------------------------------- #
# Planning (never part of a freshness key)
# --------------------------------------------------------------------------- #
# A sensitivity job's cost is a few plain solves' worth per parameter for every
# engine, plus AMICI's code generation and compile, which dominates on all but the
# smallest networks and grows faster than the network (the BNGsim paper's AMICI
# compiles: about 25 s at 100 reactions, 250 s at 3700, over an hour at 15000).
# The number of parameters comes from an explicit list, else the spec's
# ``size.sens_parameters`` (planning only), else DEFAULT_PARAMETER_COUNT.
# Calibrated on the two pilot problems; provisional until larger ones have run.
MIN_TIMEOUT_SECONDS = 1800.0
TIMEOUT_FACTOR = 4.0
ENGINE_BUDGET_CAP_SECONDS = 3600.0
DEFAULT_PARAMETER_COUNT = 20


def parameter_count(problem: Problem) -> int:
    chosen = protocol(problem)["parameters"]
    if isinstance(chosen, list):
        return len(chosen)
    return int(problem.size.get("sens_parameters", DEFAULT_PARAMETER_COUNT))


def engine_seconds(problem: Problem) -> dict[str, float]:
    """Planned seconds for each engine's cold build and first solve."""
    reactions = problem.size.get("reactions") or 10 * problem.size.get("species", 100)
    n = parameter_count(problem)
    solve = 0.002 + 2e-6 * reactions ** 1.5  # one plain solve at the protocol's tolerances
    sens = 2.0 * (1 + n) * solve  # a forward-sensitivity solve
    return {
        "bngsim": 5.0 + 1e-3 * reactions + sens,
        "bngsim_simultaneous": 1.0 + sens,
        "amici": 25.0 + 3e-4 * reactions ** 1.7 + sens,
        # 4N + 1 solves, each a fresh process at tolerances 100x tighter.
        "run_network_fd": (4 * n + 1) * (0.02 + 3.0 * solve),
        "copasi": 5.0 + 50.0 * sens,
    }


def planning_seconds(problem: Problem) -> tuple[float, float]:
    """Rough wall-clock estimates (correctness, timing) for one problem."""
    per = engine_seconds(problem)
    correctness = 30.0 + sum(per.values())
    # Timing rebuilds every engine but the FD reference, then repeats each solve over
    # at least min_rounds plus a warm-up; the solves are a small part of the build.
    builds = sum(v for k, v in per.items() if k != "run_network_fd")
    timing = 60.0 + builds + (TIMING_PROTOCOL["min_rounds"] + 2) * 0.2 * builds
    return correctness, timing


def engine_budget_seconds(problem: Problem) -> float:
    """Wall budget for one engine's cold build + solve in the correctness job."""
    slowest = max(engine_seconds(problem).values())
    return min(ENGINE_BUDGET_CAP_SECONDS, max(MIN_TIMEOUT_SECONDS, TIMEOUT_FACTOR * slowest))


# --------------------------------------------------------------------------- #
# Choosing the parameters
# --------------------------------------------------------------------------- #
def select_parameters(structure: netfile.Structure, chosen: str | list[str]
                      ) -> tuple[list[str], dict[str, str]]:
    """The parameters to differentiate, in ``.net`` order, and those left out with
    the reason. Raises ValueError for an explicit name the network cannot honour."""
    primary = structure.primary()
    rates = structure.reaching_rates()
    initial = structure.reaching_initial()
    sizes = structure.sizing_compartments()
    excluded: dict[str, str] = {}
    if chosen == "rate_constants":
        candidates = [p for p in primary if p in rates]
    elif chosen == "all":
        candidates = [p for p in primary if p in rates or p in initial]
    else:
        unknown = [p for p in chosen if p not in structure.expressions]
        derived = [p for p in chosen if p in structure.expressions and p not in primary]
        if unknown or derived:
            raise ValueError(
                "protocol.sens.parameters: "
                + "; ".join(filter(None, [
                    f"not parameters of the network: {', '.join(unknown)}" if unknown else "",
                    f"derived (an expression, not a number): {', '.join(derived)}"
                    if derived else ""])))
        candidates = [p for p in primary if p in chosen]
        for p in candidates:
            if p not in rates and p not in initial:
                excluded[p] = "reaches no reaction rate or initial amount"
        candidates = [p for p in candidates if p not in excluded]
    chosen_out = []
    for p in candidates:
        if p in sizes:
            excluded[p] = "sizes a compartment"
        elif primary[p] == 0.0:
            excluded[p] = "zero value: no relative step"
        else:
            chosen_out.append(p)
    return chosen_out, excluded


# --------------------------------------------------------------------------- #
# The system under test and the reference engines
# --------------------------------------------------------------------------- #
def _grid(proto: dict[str, Any]) -> tuple[float, int]:
    return float(proto["t_end"]), int(proto["n_steps"])


def times(proto: dict[str, Any]) -> np.ndarray:
    t_end, n_steps = _grid(proto)
    return np.linspace(0.0, t_end, n_steps + 1)


def pack(S: np.ndarray, X: np.ndarray, *extra: np.ndarray) -> np.ndarray:
    """One array a forked child can return: ``S`` ``(n_t, n_sp, n_p)``, any further
    tensors of the same shape, then ``X`` ``(n_t, n_sp)`` as the last slice."""
    return np.concatenate([np.asarray(S, float), *map(np.asarray, extra),
                           np.asarray(X, float)[:, :, None]], axis=2)


def unpack(a: np.ndarray, n_p: int) -> tuple[list[np.ndarray], np.ndarray]:
    """The tensors and the trajectory of ``pack``."""
    k = (a.shape[2] - 1) // n_p
    return [a[:, :, j * n_p:(j + 1) * n_p] for j in range(k)], a[:, :, -1]


class BngsimSens:
    """BNGsim's forward sensitivities on the ``.net``: one model, one Simulator built
    for the parameters (which compiles the model with its analytic sensitivity
    right-hand side), solved after ``model.reset()`` each time."""

    def __init__(self, net: Path, proto: dict[str, Any], params: list[str], *,
                 sensitivity_method: str = "staggered"):
        import bngsim

        self.proto, self.params, self.method = proto, list(params), sensitivity_method
        t0 = time.perf_counter()
        self.model = bngsim.Model.from_net(str(net))
        self.load_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        self.sim = bngsim.Simulator(self.model, method="ode", sensitivity_params=self.params,
                                    sensitivity_method=sensitivity_method)
        self.codegen_seconds = time.perf_counter() - t1
        # As in ode: the scalar atol held exactly, the tolerance AMICI takes too.
        self.atol = bngsim.ExactAtol(float(proto["atol"]))
        self.covered = np.ones(len(self.params), bool)
        self.stats: dict[str, Any] = {}

    def run(self):
        t_end, n_steps = _grid(self.proto)
        self.model.reset()
        return self.sim.run(t_span=(0.0, t_end), n_points=n_steps + 1,
                            rtol=float(self.proto["rtol"]), atol=self.atol)

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
        }


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
    defaults otherwise (its corrector is ``simultaneous``).
    """

    NAME = "amici"
    INPUT = "sbml"

    def __init__(self, sbml: Path, species_names: list[str], name_to_id: dict[str, str],
                 proto: dict[str, Any], params: list[str]):
        import amici
        from amici.sim import sundials as ss

        self._ss = ss
        want_ids = _align(species_names, name_to_id)
        self.proto, self.params = proto, list(params)
        outdir = sbml.parent / "amici_sens_model"
        t0 = time.perf_counter()
        amici.SbmlImporter(str(sbml)).sbml2amici(
            "bench_sens_model", str(outdir), generate_sensitivity_code=True, verbose=False)
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

    def run(self):
        ss = self._ss
        solver = self.model.create_solver()
        solver.set_relative_tolerance(float(self.proto["rtol"]))
        solver.set_absolute_tolerance(float(self.proto["atol"]))
        solver.set_sensitivity_order(ss.SensitivityOrder.first)
        solver.set_sensitivity_method(ss.SensitivityMethod.forward)
        rdata = ss.run_simulation(self.model, solver)
        if rdata.status != 0:
            raise RuntimeError(f"AMICI simulation failed with status {rdata.status}")
        return rdata

    def tensors(self, rdata) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(rdata.x, dtype=float)[:, self.idx]
        sx = np.transpose(np.asarray(rdata.sx, dtype=float), (0, 2, 1))[:, self.idx, :]
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
            "not_differentiated": self.uncovered,
            "load_seconds": self.load_seconds,
            "codegen_seconds": self.codegen_seconds,
        }


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

    def __init__(self, sbml: Path, species_names: list[str], name_to_id: dict[str, str],
                 proto: dict[str, Any], params: list[str]):
        import basico
        import COPASI

        self._copasi = COPASI
        ids = _align(species_names, name_to_id)
        self.proto, self.params = proto, list(params)
        t_end, n_steps = _grid(proto)
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
        n_t = _grid(self.proto)[1] + 1
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


def run_network_command(net: Path, proto: dict[str, Any], out_prefix: Path, *,
                        rtol: float, atol: float) -> list[str]:
    """The run_network call BNG2.pl makes for an ODE simulate, at the given tolerances."""
    t_end, n_steps = _grid(proto)
    return [
        *env.run_network_launch(),
        str(env.run_network_bin()), "-o", str(out_prefix), "-p", "cvode",
        "-a", repr(float(atol)), "-r", repr(float(rtol)),
        "--cdat", "1", "--fdat", "0", "-g", str(net), str(net),
        repr(t_end / n_steps), str(n_steps),
    ]


class FdRunNetwork:
    """Central finite differences from run_network, an engine independent of BNGsim.

    For each parameter, the ``.net`` is written with the parameter at ``p (1 + h)``
    and ``p (1 - h)`` and solved by a fresh run_network process, which re-derives
    every derived parameter, rate and initial amount from it -- so the derivative is
    the total one, through every chain the network has. Two steps, ``h`` and ``h/2``,
    at the protocol's ``fd_rtol``/``fd_atol``: ``4N + 1`` solves in all.
    ``D(h/2)`` is the reference; ``|D(h) - D(h/2)|`` bounds its error
    (``bench.sensitivity``).
    """

    NAME = "run_network_fd"
    INPUT = "net"

    def __init__(self, net: Path, proto: dict[str, Any], params: list[str], values: np.ndarray,
                 workdir: Path, *, timeout: float):
        self.text = net.read_text(encoding="utf-8")
        self.proto, self.params, self.values = proto, list(params), np.asarray(values, float)
        self.workdir, self.timeout = workdir, timeout
        workdir.mkdir(parents=True, exist_ok=True)
        self.covered = np.ones(len(self.params), bool)
        self.solves = 0
        self.solve_seconds = 0.0

    def _solve(self, text: str) -> np.ndarray:
        net = self.workdir / "fd.net"
        net.write_text(text, encoding="utf-8")
        cmd = run_network_command(net, self.proto, self.workdir / "fd",
                                  rtol=float(self.proto["fd_rtol"]),
                                  atol=float(self.proto["fd_atol"]))
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
        n_t = _grid(self.proto)[1] + 1
        if values.shape[0] != n_t:
            raise RuntimeError(f"run_network wrote {values.shape[0]} time points, expected {n_t}")
        return values

    def run(self) -> np.ndarray:
        """``pack(D(h/2), D(h), x)``: the finer quotient first."""
        base = self._solve(self.text)
        h = float(self.proto["fd_step"])
        quotients = []
        for step in (h / 2, h):
            cols = []
            for name, p in zip(self.params, self.values):
                up = self._solve(netfile.with_parameters(self.text, {name: p * (1 + step)}))
                down = self._solve(netfile.with_parameters(self.text, {name: p * (1 - step)}))
                cols.append((up - down) / (2 * step * p))
            quotients.append(np.stack(cols, axis=2))
        return pack(quotients[0], base, quotients[1])

    def config(self) -> dict[str, Any]:
        cmd = recorded_command(run_network_command(
            self.workdir / "fd.net", self.proto, self.workdir / "fd",
            rtol=float(self.proto["fd_rtol"]), atol=float(self.proto["fd_atol"])))
        names = {str(self.workdir / "fd.net"): "<perturbed>.net", str(self.workdir / "fd"): "<out>"}
        return {
            "method": "central differences at steps h and h/2 (relative)",
            "engine": "run_network (fresh process per solve)",
            "step": float(self.proto["fd_step"]),
            "rtol": float(self.proto["fd_rtol"]),
            "atol": float(self.proto["fd_atol"]),
            "command": [names.get(c, c) for c in cmd],
            "solves": self.solves,
            "solve_seconds": self.solve_seconds,
        }


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
        return cls(prep["net"], prep["proto"], prep["params"], prep["values"], workdir,
                   timeout=timeout)
    return cls(prep["sbml"], prep["species_names"], prep["name_to_id"], prep["proto"],
               prep["params"])


def _progress(msg: str) -> None:
    progress(CAPABILITY, msg)


# --------------------------------------------------------------------------- #
# Preparation and the cold runs of the correctness job
# --------------------------------------------------------------------------- #
def _prepare(problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    proto = protocol(problem)
    prep = prepare_network(problem, ctx, proto["netgen"])
    structure = netfile.read_structure(prep["net"])
    if structure.species != [n.lstrip("$") for n in prep["species_names"]]:
        raise RuntimeError("the .net's species and BNGsim's species_names disagree")
    params, excluded = select_parameters(structure, proto["parameters"])
    if not params:
        raise ValueError(f"no parameter to differentiate ({proto['parameters']!r}; "
                         f"left out: {excluded})")
    observables, weights = structure.observable_weights()
    prep.update(proto=proto, structure=structure, params=params,
                values=np.array([structure.literal(p) for p in params]),
                observables=observables, weights=weights)
    prep["info"]["parameters"] = {"selection": proto["parameters"], "n": len(params),
                                  "names": params, "excluded": excluded}
    prep["info"]["network"]["observables"] = len(observables)
    return prep


def _cold_bngsim(prep: dict[str, Any], name: str):
    def fn(stage):
        t0 = time.perf_counter()
        stage(BUILD_STAGE.get(name, "build"))
        engine = BngsimSens(prep["net"], prep["proto"], prep["params"], **CONFIGURATIONS[name])
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
    c = sv.compare(sv.scaled(own, prep["values"]), sv.scaled(derived, prep["values"]),
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
        self.sig = sv.scaled(S_out, prep["values"])
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
        est = sv.fd_estimate(sv.scaled(coarse_out, prep["values"]),
                             sv.scaled(fine_out, prep["values"]))
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


def _summary(c: sv.Comparison, prep: dict[str, Any], a: _Run, b: _Run) -> dict[str, Any]:
    out = c.summary(_output_names(prep), prep["params"], times(prep["proto"]),
                    n_species=len(prep["species_names"]))
    n_sp = len(prep["species_names"])
    out["trajectory_max_err_over_peak"] = max_err_over_peak(a.y[:, :n_sp], b.y[:, :n_sp])
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
        prep = _prepare(problem, ctx)
    except Exception as exc:  # noqa: BLE001 - a bad test input, a labelled row
        return {"outcome": "BAD_TEST", "error": f"{type(exc).__name__}: {exc}"}
    info = prep["info"]
    workdir = Path(ctx["workdir"])
    budget = float(ctx.get("engine_budget_seconds") or engine_budget_seconds(problem))
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
    for name in REFERENCE_ENGINES:
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

    # A further configuration that fails is BNGsim failing, as the default would be.
    if bn_error is None:
        bn_error = next((f"{name}: {err}" for name, err in bn_errors.items()), None)
    unjudged = info.get("coverage", {}).get("unjudged", [])
    result = sens_outcome(bn_error, ref_errors, len(REFERENCE_ENGINES), verdict_json,
                          consistent, unjudged)
    if bn_error and result == "EXCEPTION" and \
            any(part.strip() == "TIMEOUT" for part in bn_error.split(":")[:2]):
        result = "TIMEOUT"
    rec: dict[str, Any] = {"outcome": result, "measurements": info}
    if verdict_json:
        rec["verdicts"] = verdict_json
    if bn_error:
        rec["bngsim_error"] = bn_error
    for name, err in ref_errors.items():
        rec[f"{name}_error"] = err
    return rec


def _role(name: str) -> str:
    """How an engine is timed, for the timing record (``gather`` reads it)."""
    if name in CONFIGURATIONS:
        return (f"warm, in-process (system under test, "
                f"{CONFIGURATIONS[name]['sensitivity_method']} corrector)")
    if name == "bngsim_ode":
        return "warm, in-process (plain solve: the unit of a finite-difference gradient)"
    return engine_role(name)


def _capture(engine, values: np.ndarray):
    """What a timed repetition is checked on: its semi-scaled sensitivities, one
    column per (species, parameter) -- extracted outside the timed region. A
    parameter the engine does not differentiate is a column of zeros."""
    def capture(raw):
        sig = sv.scaled(engine.tensors(raw)[0], values)
        return np.nan_to_num(sig.reshape(sig.shape[0], -1), nan=0.0)
    return capture


def run_timing(job: Job, problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    from .ode import BngsimODE, RunNetworkODE

    try:
        prep = _prepare(problem, ctx)
    except Exception as exc:  # noqa: BLE001 - a bad test input, a labelled row
        return {"outcome": "BAD_TEST", "error": f"{type(exc).__name__}: {exc}"}
    proto, info = prep["proto"], prep["info"]
    workdir = Path(ctx["workdir"])
    timeout = ctx["subprocess_timeout"]
    n = len(prep["params"])

    # BNGsim cold: load, compile and a first solve of the default configuration,
    # with the empty compile cache the worker gave this job. Its result anchors the
    # warm repetitions.
    t0 = time.perf_counter()
    bngsim_engine = BngsimSens(prep["net"], proto, prep["params"],
                               **CONFIGURATIONS[DEFAULT_CONFIGURATION])
    first = bngsim_engine.run()
    info["bngsim_cold_seconds"] = time.perf_counter() - t0
    anchor = _capture(bngsim_engine, prep["values"])(first)
    info["bngsim"] = bngsim_engine.config()
    engines: dict[str, Any] = {DEFAULT_CONFIGURATION: bngsim_engine}
    configs: dict[str, Any] = {}
    for name, cfg in CONFIGURATIONS.items():
        if name == DEFAULT_CONFIGURATION:
            continue
        eng = BngsimSens(prep["net"], proto, prep["params"], **cfg)
        eng.tensors(eng.run())  # trial run
        engines[name] = eng
        configs[name] = eng.config()
    # The unit of a finite-difference gradient: one plain solve, by BNGsim and by
    # run_network (a fresh process, as BNG2.pl and a finite-difference fit run it).
    plain_proto = {**proto, "t_start": 0, "sparse": False}
    plain = BngsimODE(prep["net"], plain_proto)
    plain.warm()
    engines["bngsim_ode"] = plain
    configs["bngsim_ode"] = {"load_seconds": plain.load_seconds}
    info["configurations"] = configs

    # Each reference: build and give it one trial run, so one that crashes is
    # recorded and dropped rather than taking the interleave down. The FD reference
    # is not timed; its cost is the derived baseline below.
    references: dict[str, Any] = {}
    ref_errors: dict[str, str] = {}
    timed_out = correctness_timeouts(job, ctx, REFERENCE_ENGINES)
    try:
        rn = RunNetworkODE(prep["net"], plain_proto, workdir / "rn_plain", sparse=False,
                           timeout=timeout)
        rn.run()
        engines["run_network"] = rn
        references["run_network"] = rn.config()
    except Exception as exc:  # noqa: BLE001
        ref_errors["run_network"] = f"{type(exc).__name__}: {exc}"[:400]
    for name in REFERENCE_ENGINES:
        if name == FD:
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
    info["references"] = references

    _progress(f"interleaved timing of {', '.join(engines)}")
    runs = {name: (eng.warm if name in ("bngsim_ode", "run_network") else eng.run)
            for name, eng in engines.items()}
    captures = {name: (type(eng).trajectory if name in ("bngsim_ode", "run_network")
                       else _capture(eng, prep["values"]))
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
                 for name in ("bngsim_ode", "run_network") if name in times_}
    rec: dict[str, Any] = {
        "outcome": "PASS" if reproducible else "NOT_REPRODUCIBLE",
        "measurements": {
            **info,
            "engine_roles": {name: _role(name) for name in engines},
            "repetitions": reps,
            "reproducibility": repro,
            "times": times_,
            "fd_baselines": baselines,
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
