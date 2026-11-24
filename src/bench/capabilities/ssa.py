"""Exact stochastic simulation (SSA) of BNGL models: BNGsim against reference engines.

BNGsim's exact SSA (``Simulator(method="ssa")``) is the system under test. On one
problem it is compared with every reference in ``REFERENCE_ENGINES``, each running
Gillespie's direct method on the same generated network:

- **run_network** -- BioNetGen's binary, ``-p ssa``, a fresh process per trajectory
  the way BNG2.pl runs a ``simulate`` (reads the ``.net``);
- **roadrunner** -- libRoadRunner's ``gillespie`` integrator, in process, on SBML
  exported from the same ``.net``;
- **copasi** -- COPASI's direct method, in process through its own API, on the same
  SBML, read with molecule-count semantics.

and, where the spec declares it, against the network's **exact moments**
(``moments: linear``): a network whose every propensity is at most linear in the
state has means and variances that solve closed linear ODEs, computed by the
harness from the ``.net`` with no simulator involved (``bench.stochastic``). That is
this suite's DSMTS-style analytic reference for BNGL models.

**Correctness.** Every engine draws ``replicates`` independent trajectories, from
fixed seeds in a block of its own (``seed_for``), on the uniform grid of
``n_steps + 1`` points over ``[0, t_end]``. The verdict of BNGsim against a
reference is a permutation max-T test on the per-cell means and variances of the
two ensembles; against exact moments, per-cell z-tests with a Bonferroni bound. Both
have a stated false-alarm rate, ``VERDICT["alpha"]`` per comparison, and both are
reproducible to the bit: the ensembles come from recorded seeds, and the
relabellings from a seed derived from the protocol's. See ``bench.stochastic`` and
docs/DESIGN.md §5.3. The job passes only when BNGsim agrees with every reference.

Failures are attributed exactly as in ``ode``: each engine builds and samples in its
own forked child under a wall budget (``run_isolated``); a reference that fails or
runs out of time is ``REFERENCE_FAILED`` with a ``<name>_error`` and BNGsim is still
judged against the others; a BNGsim failure is ``EXCEPTION`` (``TIMEOUT`` when it ran
out of time). A reference that cannot simulate this model's exact SSA at all is
recorded the same way, with an error starting ``UNSUPPORTED`` and without being run:
libRoadRunner's gillespie fires the SBML rate law as the propensity, so on a
reaction with a repeated reactant (``A + A``) it uses ``k*A*A`` where the exact
propensity is ``k*A*(A-1)`` (and since 2.9.3 can fire without end once ``A`` is 0).

**Timing.** Each engine's timed unit is one ensemble of ``timing_replicates``
trajectories with the same seeds every round, interleaved round-robin with the
other engines (``bench.timing``); the job reports each engine's absolute time per
ensemble with a bootstrap CI, and per trajectory as that divided by the ensemble
size. Because the seeds repeat, every kept repetition must reproduce the first one
exactly, which also shows that an engine's reset between ensembles is complete. The
one-time cost before sampling -- BNGsim's load and propensity compile, libRoadRunner's
SBML load and JIT, COPASI's import -- is measured once, cold, and reported beside it;
run_network loads the ``.net`` inside every trajectory, so its load is in its time.

Input formats: a BNGL model runs through the generated ``.net`` as above. An SBML
model (the ssa-sbml suite) is read as it is -- BNGsim, roadrunner and COPASI load
the pinned SBML, its amounts as molecule counts, and run_network reads a network
derived from it where that is faithful for the jump process -- by
``bench.capabilities.ssa_sbml``, through the same jobs, verdicts and timing.

**Parts.** An engine's ensemble is independent of the others', so a correctness
job's engines are its parts (``Job.parts``): a job too long for one allocation
runs each engine as its own Slurm array task (``bench submit``), each storing
its ensemble beside the job's result, and the job then scores the stored
ensembles instead of sampling them again. The result is the same either way.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from .. import env, netfile, stochastic
from ..jobs import CORRECTNESS, TIMING, Job
from ..specs import Problem
from ..timing import absolute_time_stats, round_robin_interleaved
from ._common import (
    correctness_timeouts,
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

CAPABILITY = "ssa"
# Bump when this module's computation changes; every result it wrote goes stale.
HARNESS_VERSION = "1"

TIMING_PROTOCOL = {"warmup_rounds": 1, "min_rounds": 5, "max_rounds": 20, "budget_seconds": 120.0}
# A timing-only part of the key, as in ode.
REPRO_CHECK_VERSION = 1
# A repetition reruns the same seeds, so it must reproduce the first run exactly;
# the tolerance only absorbs summation order in the summary it is compared on.
REPRO_TOL = 1e-9

# The verdict's statistics (bench.stochastic). In the correctness definition, so a
# change restales every correctness result and nothing else.
# alpha is the false-alarm rate of one comparison: with three references and exact
# moments a job compares four times, so a correct BNGsim fails a job with
# probability at most 0.004, and a ten-problem suite at most 0.04.
VERDICT = {
    "two_sample": "permutation max-T on means and variances/1",
    "analytic": "Bonferroni z on means and log-variances/1",
    "alpha": 1e-3,
    "permutations": 1999,
    "stop_exceedances": 20,
}

DEFAULT_SEED = 1
DEFAULT_TIMING_REPLICATES = 100
# Each engine's trajectories take seeds from a block of its own, so no two engines
# ever share one (and identical generators could never correlate the ensembles).
SEED_BLOCK = 10_000_000
# Fixed, not derived from the registry's order, so adding an engine never moves
# another engine's seeds (and with them its recorded results). A new reference
# takes the next free block here.
ENGINE_SEED_BLOCKS = {"bngsim": 0, "run_network": 1, "roadrunner": 2, "copasi": 3}
ANALYTIC = "analytic"
# The kinds of exact moments a spec may declare (specs.py validates the same set):
# ``linear`` (computed from a first-order .net) or, for an SBML model, a pinned
# ``table`` of means and SDs, as DSMTS publishes them.
MOMENT_KINDS = ("linear", "table")


def moments_kind(proto: dict[str, Any]) -> str | None:
    m = proto.get("moments")
    return m.get("kind") if isinstance(m, dict) else m


def protocol(problem: Problem) -> dict[str, Any]:
    """The SSA protocol with defaults filled in, as it enters a job definition."""
    p = dict(problem.protocol_for(CAPABILITY))
    p.setdefault("seed", DEFAULT_SEED)
    p.setdefault("timing_replicates", DEFAULT_TIMING_REPLICATES)
    return p


def seed_for(engine: str, seed: int) -> int:
    """The first seed of ``engine``'s block. A reference's trajectory ``i`` uses
    ``seed_for(...) + i``; BNGsim's ensemble is ``run_replicates(seed=seed_for(...))``,
    whose replicate seeds bngsim derives from that base."""
    return int(seed) + SEED_BLOCK * ENGINE_SEED_BLOCKS[engine]


def engines_id() -> str:
    """The ``engines`` id segment: BNGsim versus every registered reference."""
    return "bngsim-vs-" + "-".join(REFERENCE_ENGINES)


def jobs(problem: Problem) -> list[Job]:
    fmt = problem.source.get("format")
    if fmt not in ("bngl", "sbml"):
        return []
    proto = protocol(problem)
    engines = engines_id()
    base = {
        "harness": f"{CAPABILITY}/{engines}/{HARNESS_VERSION}",
        "source_sha256": problem.source["sha256"],
        "protocol": proto,
        "setting": {"method": "ssa", "propensities": "auto"},
        # As in ode: the SBML references' versions key only this capability.
        "reference_versions": {
            "roadrunner": env.roadrunner_identity().get("version"),
            "copasi": env.copasi_identity().get("version"),
        },
    }
    if fmt == "sbml":
        from .ssa_sbml import SBML_HARNESS_VERSION

        # Only an SBML job carries this, so a .net job's definition is unchanged.
        base["input"] = {"format": "sbml", "harness": SBML_HARNESS_VERSION}
    est_c, est_t = planning_seconds(problem)
    comparisons = len(REFERENCE_ENGINES) + 1
    backstop = (len(REFERENCE_ENGINES) + 2) * engine_budget_seconds(problem) \
        + VERDICT_SECONDS * comparisons
    correctness = Job(
        CAPABILITY, problem.id, engines, "auto", CORRECTNESS,
        definition={**base, "verdict": VERDICT},
        est_seconds=est_c,
        timeout_seconds=max(MIN_TIMEOUT_SECONDS, TIMEOUT_FACTOR * est_c, backstop),
        parts=ensemble_seconds(problem),
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
# A trajectory costs a fixed overhead per engine plus a cost per reaction event.
# run_network's overhead is a process launch and a .net read (about 7 ms on x86 with
# the glibc loader swap, for a few-species network); an in-process engine's is under
# a millisecond (COPASI's is the largest seen, about 0.5 ms). The per-event cost is a
# conservative bound for every engine on small networks (BNGsim measures about
# 0.5 us). The events per trajectory come from the spec's ``size.ssa_events``
# (planning only), else DEFAULT_EVENTS. Calibrated on the two pilot problems. A spec
# may instead give an engine's measured cost per trajectory, ``size.ssa_ms_<engine>``
# (milliseconds, planning only), which then takes the place of that estimate.
MIN_TIMEOUT_SECONDS = 1800.0
TIMEOUT_FACTOR = 4.0
# The most wall time one engine gets for its cold build and ensemble in the
# correctness job: room for an ensemble of a few hours (a model whose every
# trajectory takes minutes), inside one allocation when the engine runs as a part.
ENSEMBLE_BUDGET_CAP_SECONDS = 28800.0
# The cap the psa capability applies to its own engines (bench.capabilities.psa).
ENGINE_BUDGET_CAP_SECONDS = 3600.0
DEFAULT_EVENTS = 1000
EVENT_SECONDS = 2e-6
TRAJECTORY_SECONDS = {"bngsim": 1e-3, "run_network": 0.01, "roadrunner": 1e-3, "copasi": 1e-3}
# A comparison that fails runs every relabelling: about 5 s, single-threaded, at
# 10^4 replicates per engine on a 201-point, 4-species grid.
VERDICT_SECONDS = 30.0


def trajectory_seconds(problem: Problem) -> dict[str, float]:
    """Planned seconds per trajectory, per engine: the spec's measured cost
    (``size.ssa_ms_<engine>``) where it gives one, else the event-count estimate."""
    events = problem.size.get("ssa_events", DEFAULT_EVENTS)
    out = {}
    for name in ("bngsim", *REFERENCE_ENGINES):
        ms = problem.size.get(f"ssa_ms_{name}")
        out[name] = ms / 1000.0 if ms is not None else \
            TRAJECTORY_SECONDS.get(name, 1e-3) + EVENT_SECONDS * events
    return out


def ensemble_seconds(problem: Problem) -> dict[str, float]:
    """Planned seconds of each engine's correctness ensemble: the job's parts."""
    n = int(protocol(problem)["replicates"])
    return {name: 10.0 + n * t for name, t in trajectory_seconds(problem).items()}


def planning_seconds(problem: Problem) -> tuple[float, float]:
    """Rough wall-clock estimates (correctness, timing) for one problem."""
    proto = protocol(problem)
    per = trajectory_seconds(problem)
    all_engines = sum(per.values())
    correctness = 30.0 + sum(ensemble_seconds(problem).values()) \
        + VERDICT_SECONDS * (len(REFERENCE_ENGINES) + 1)
    tp = TIMING_PROTOCOL
    rnd = int(proto["timing_replicates"]) * all_engines
    measured = max(tp["min_rounds"] * rnd, min(tp["budget_seconds"] + rnd, tp["max_rounds"] * rnd))
    # + BNGsim's cold ensemble and the warm-up rounds; the trial runs are one
    # trajectory each.
    timing = 60.0 + (tp["warmup_rounds"] + 1) * rnd + measured
    return correctness, timing


def engine_budget_seconds(problem: Problem) -> float:
    """Wall budget for one engine's cold build + ensemble in the correctness job."""
    slowest = max(ensemble_seconds(problem).values())
    return min(ENSEMBLE_BUDGET_CAP_SECONDS, max(MIN_TIMEOUT_SECONDS, TIMEOUT_FACTOR * slowest))


# --------------------------------------------------------------------------- #
# The system under test and the reference engines
# --------------------------------------------------------------------------- #
def _grid(proto: dict[str, Any]) -> tuple[float, int]:
    return float(proto["t_end"]), int(proto["n_steps"])


def times(proto: dict[str, Any]) -> np.ndarray:
    t_end, n_steps = _grid(proto)
    return np.linspace(0.0, t_end, n_steps + 1)


class BngsimSSA:
    """BNGsim's exact SSA on the ``.net``: one loaded model, sampled repeatedly.

    An ensemble is ``run_replicates`` after ``model.reset()``, the way a user samples
    one -- one simulator, its dependency graph built once. Building the simulator
    compiles the propensities where bngsim can (``propensity_backend`` ``cc``); that
    is its cold cost, recorded as ``codegen_seconds`` beside the load.
    """

    NAME = "bngsim"

    def __init__(self, net: Path, proto: dict[str, Any]):
        import bngsim

        self.proto = proto
        t0 = time.perf_counter()
        self.model = bngsim.Model.from_net(str(net))
        self.load_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        self.sim = self._simulator(bngsim)
        self.codegen_seconds = time.perf_counter() - t1
        self.details: dict[str, Any] = {}
        self.backend: str | None = None
        self.last_result = None

    def _simulator(self, bngsim):
        """The simulator an ensemble samples (a subclass may choose another method)."""
        return bngsim.Simulator(self.model, method="ssa")

    def _span(self) -> dict[str, Any]:
        t_end, n_steps = _grid(self.proto)
        return {"t_span": (0.0, t_end), "n_points": n_steps + 1}

    def ensemble(self, n: int, seed: int) -> np.ndarray:
        """``(n, n_times, n_species)``, copied out of the result."""
        self.model.reset()
        result = self.sim.run_replicates(int(n), seed=int(seed), squeeze=True, **self._span())
        self.backend = (getattr(result, "ssa_diagnostics", None) or {}).get("propensity_backend")
        self.last_result = result  # read outside the timed region, if at all
        return np.array(result.species, dtype=float)

    def ensemble_detailed(self, n: int, seed: int) -> np.ndarray:
        """The same ensemble one Result per replicate, to record each one's event
        count and SSA diagnostics (the correctness job; timing uses ``ensemble``)."""
        self.model.reset()
        results = self.sim.run_replicates(int(n), seed=int(seed), **self._span())
        stack = np.stack([np.array(r.species, dtype=float) for r in results])
        self.backend = (getattr(results[0], "ssa_diagnostics", None) or {}).get(
            "propensity_backend")
        self.details = self._details(results)
        return stack

    def _details(self, results: list) -> dict[str, Any]:
        """What the record keeps of an ensemble's per-replicate Results: the event
        counts per trajectory and the SSA diagnostics, summed over replicates."""
        events = np.array([float((r.solver_stats or {}).get("n_steps", 0)) for r in results])
        diag: dict[str, Any] = {}
        for r in results:
            for k, v in (getattr(r, "ssa_diagnostics", None) or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    diag[k] = diag.get(k, 0) + v
                elif v and k not in diag:
                    diag[k] = v
        return {
            "events_per_trajectory": {"mean": float(events.mean()), "median": float(np.median(events)),
                                      "max": float(events.max())},
            "diagnostics": diag,
        }

    def config(self) -> dict[str, Any]:
        return {
            "method": "ssa",
            "propensity_backend": self.backend,
            "load_seconds": self.load_seconds,
            "codegen_seconds": self.codegen_seconds,
            **self.details,
        }


def run_network_command(net: Path, proto: dict[str, Any], out_prefix: Path, seed: int,
                        method: tuple[str, ...] = ("-p", "ssa")) -> list[str]:
    """The run_network call BNG2.pl makes for ``simulate({method=>"ssa"})``; another
    stochastic method passes its own flags, which BNG2.pl puts in the same place
    (``("-p", "psa", "--poplevel", "100")`` for ``simulate({method=>"psa",
    poplevel=>100})``)."""
    t_end, n_steps = _grid(proto)
    return [
        *env.run_network_launch(),
        str(env.run_network_bin()), "-o", str(out_prefix), *method, "-h", str(int(seed)),
        "--cdat", "1", "--fdat", "0", "-g", str(net), str(net),
        repr(t_end / n_steps), str(n_steps),
    ]


class RunNetworkSSA:
    """run_network's direct method as a reference: one fresh process per trajectory.

    There is no warm run_network -- BNG runs every ``simulate`` as a process that
    reads the ``.net`` -- so an ensemble of ``n`` is ``n`` processes, each with its
    own seed, each read back from its ``.cdat``. Its columns are in the ``.net``
    species order, which is BNGsim's.
    """

    NAME = "run_network"
    INPUT = "net"
    # The method flags of every trajectory (run_network_command).
    METHOD_ARGS: tuple[str, ...] = ("-p", "ssa")

    def __init__(self, net: Path, proto: dict[str, Any], out_prefix: Path, *, timeout: float):
        # Absolute, since every trajectory runs in the output directory.
        self.net, self.out_prefix = Path(net).absolute(), Path(out_prefix).absolute()
        self.proto, self.timeout = proto, timeout
        self.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    def trajectory(self, seed: int) -> np.ndarray:
        cmd = run_network_command(self.net, self.proto, self.out_prefix, seed, self.METHOD_ARGS)
        # Run in the job's scratch directory, so a crash's core file stays there.
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout,
                                  check=False, cwd=self.out_prefix.parent)
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) \
                else (exc.stdout or "")
            tail = out.strip().splitlines()
            raise RuntimeError(f"run_network failed at seed {int(seed)} (no result within "
                               f"{self.timeout:g} s, so stopped): "
                               f"{(tail[-1] if tail else '')[:300]}") from None
        if proc.returncode != 0:
            tail = (proc.stderr.strip() or proc.stdout.strip()).splitlines()
            how = (f"killed by signal {-proc.returncode}" if proc.returncode < 0
                   else f"exit {proc.returncode}")
            raise RuntimeError(f"run_network failed at seed {int(seed)} ({how}): "
                               f"{(tail[-1] if tail else '')[:300]}")
        cdat = Path(f"{self.out_prefix}.cdat")
        if not cdat.exists():
            raise RuntimeError("run_network produced no .cdat")
        _, values, _ = read_dat(cdat)
        cdat.unlink()
        if values.shape[0] != _grid(self.proto)[1] + 1:
            raise RuntimeError(f"run_network wrote {values.shape[0]} time points, "
                               f"expected {_grid(self.proto)[1] + 1}")
        return values

    def ensemble(self, n: int, seed: int) -> np.ndarray:
        return np.stack([self.trajectory(int(seed) + i) for i in range(int(n))])

    def config(self) -> dict[str, Any]:
        # The flags, with the job's scratch paths and the per-trajectory seed as
        # placeholders: they differ every run and say nothing about the experiment.
        cmd = recorded_command(run_network_command(self.net, self.proto, self.out_prefix, 0,
                                                   self.METHOD_ARGS))
        names = {str(self.net): "model.net", str(self.out_prefix): "<out>"}
        cmd = [names.get(c, c) for c in cmd]
        cmd[cmd.index("-h") + 1] = "<seed>"
        return {"method": "ssa (direct method)", "command": cmd,
                "seeds": "first_seed + i for trajectory i"}


def _align(species_names: list[str], name_to_id: dict[str, str]) -> list[str]:
    missing = [n for n in species_names if n not in name_to_id]
    if missing:
        raise RuntimeError(
            f"cannot align {len(missing)} BNGsim species to SBML ids "
            f"(e.g. {missing[:3]}); export/name map is wrong")
    return [name_to_id[n] for n in species_names]


class RoadRunnerSSA:
    """libRoadRunner's ``gillespie`` integrator, on the SBML exported from the ``.net``.

    Fixed output steps (``variable_step_size`` off), species selected as amounts by
    SBML id in BNGsim's column order; each trajectory is ``reset()``, a new seed, and
    ``simulate``. Not applicable to a network with a repeated reactant (``unsupported``).
    """

    NAME = "roadrunner"
    INPUT = "sbml"

    @staticmethod
    def unsupported(network: netfile.Network | None) -> str | None:
        if network is None:
            return None
        reps = network.repeated_reactants()
        if not reps:
            return None
        r = reps[0]
        return (f"UNSUPPORTED: {len(reps)} reaction(s) consume one species twice (first: "
                f"reaction {r.index}{' ' + r.label if r.label else ''}); libRoadRunner's "
                "gillespie fires the SBML rate law k*A*A as the propensity, where exact SSA "
                "uses k*A*(A-1), and can fire without end once A reaches 0")

    def __init__(self, sbml: Path, species_names: list[str], name_to_id: dict[str, str],
                 proto: dict[str, Any]):
        roadrunner = env.import_roadrunner()

        self.selections = _align(species_names, name_to_id)
        self.proto = proto
        t0 = time.perf_counter()
        self.rr = roadrunner.RoadRunner(str(sbml))
        self.load_seconds = time.perf_counter() - t0
        self.rr.setIntegrator("gillespie")
        self.integrator = self.rr.getIntegrator()
        self.integrator.setValue("variable_step_size", False)
        self.rr.timeCourseSelections = ["time"] + self.selections

    def trajectory(self, seed: int) -> np.ndarray:
        t_end, n_steps = _grid(self.proto)
        self.rr.reset()
        self.integrator.setValue("seed", int(seed))
        out = np.array(self.rr.simulate(0.0, t_end, n_steps + 1), dtype=float)[:, 1:]
        if out.shape[0] != n_steps + 1:
            raise RuntimeError(f"roadrunner returned {out.shape[0]} time points, "
                               f"expected {n_steps + 1}")
        return out

    def ensemble(self, n: int, seed: int) -> np.ndarray:
        return np.stack([self.trajectory(int(seed) + i) for i in range(int(n))])

    def config(self) -> dict[str, Any]:
        return {"integrator": "gillespie (fixed output steps)",
                "version": env.roadrunner_identity().get("version"),
                "output": "amount", "load_seconds": self.load_seconds}


class CopasiSSA:
    """COPASI's direct method (``Method_directMethod``), on the exported SBML.

    Loaded through BASICO and then driven through COPASI's own task API, so a
    trajectory costs COPASI's work and a read of its time series, not a pandas
    frame per call. Two settings make it the same experiment as the others:

    - **Molecule counts.** The exported SBML declares no units, so COPASI imports
      its amounts in moles and would hold ~6e23 particles per unit; setting the
      quantity unit to ``#`` makes an amount of 1 one particle, as in the ``.net``.
    - **No step ceiling.** ``Max Internal Steps`` is raised to its maximum, so a
      long trajectory is never cut short by COPASI's default cap on events per
      output interval; the per-engine wall budget bounds it instead.

    The seed is read when the task initialises, so each trajectory sets its seed
    and initialises before it runs (about 30 us).
    """

    NAME = "copasi"
    INPUT = "sbml"

    def __init__(self, sbml: Path, species_names: list[str], name_to_id: dict[str, str],
                 proto: dict[str, Any]):
        import basico
        import COPASI

        self._copasi = COPASI
        self.ids = _align(species_names, name_to_id)
        self.proto = proto
        t_end, n_steps = _grid(proto)
        t0 = time.perf_counter()
        self.dm = basico.load_model(str(sbml))
        if self.dm is None:
            raise RuntimeError("COPASI could not import the exported SBML")
        model = self.dm.getModel()
        self.imported_quantity_unit = model.getQuantityUnit()
        model.setQuantityUnit("#")
        model.compileIfNecessary()
        self.task = self.dm.getTask("Time-Course")
        self.task.setMethodType(COPASI.CTaskEnum.Method_directMethod)
        problem = self.task.getProblem()
        problem.setOutputStartTime(0.0)
        problem.setDuration(t_end)
        problem.setStepNumber(n_steps)
        problem.setAutomaticStepSize(False)
        problem.setTimeSeriesRequested(True)
        self.method = self.task.getMethod()
        self.method.getParameter("Max Internal Steps").setIntValue(2**31 - 1)
        self.method.getParameter("Use Random Seed").setBoolValue(True)
        self.load_seconds = time.perf_counter() - t0
        self._columns: list[int] | None = None

    def _fail(self, what: str, before: int):
        msgs = self._copasi.CCopasiMessage.getAllMessageText() \
            if self._copasi.CCopasiMessage.size() > before else ""
        raise RuntimeError(f"COPASI {what} failed: {msgs.strip()[:300]}")

    def trajectory(self, seed: int) -> np.ndarray:
        cp = self._copasi
        self.method.getParameter("Random Seed").setIntValue(int(seed))
        before = cp.CCopasiMessage.size()
        if not self.task.initializeRaw(cp.CCopasiTask.OUTPUT_UI):
            self._fail("task initialisation", before)
        if not self.task.processRaw(True):
            self._fail("direct-method run", before)
        ts = self.task.getTimeSeries()
        if self._columns is None:
            col = {ts.getSBMLId(j, self.dm): j for j in range(ts.getNumVariables())}
            missing = [i for i in self.ids if i not in col]
            if missing:
                raise RuntimeError(f"COPASI's time series lacks species {missing[:3]}")
            self._columns = [col[i] for i in self.ids]
        rows = ts.getRecordedSteps()
        if rows != _grid(self.proto)[1] + 1:
            raise RuntimeError(f"COPASI recorded {rows} time points, "
                               f"expected {_grid(self.proto)[1] + 1}")
        return np.array([[ts.getData(i, j) for j in self._columns] for i in range(rows)],
                        dtype=float)

    def ensemble(self, n: int, seed: int) -> np.ndarray:
        return np.stack([self.trajectory(int(seed) + i) for i in range(int(n))])

    def config(self) -> dict[str, Any]:
        return {"method": "direct method (Gillespie)",
                "version": env.copasi_identity().get("version"),
                "output": "particle number",
                "quantity_unit": f"# (imported as {self.imported_quantity_unit})",
                "load_seconds": self.load_seconds}


# The reference engines BNGsim is compared against, in a fixed order.
REFERENCE_ENGINES: dict[str, type] = {
    RunNetworkSSA.NAME: RunNetworkSSA,
    RoadRunnerSSA.NAME: RoadRunnerSSA,
    CopasiSSA.NAME: CopasiSSA,
}

BUILD_STAGE = {
    "bngsim": "load + propensity compile",
    "run_network": "trajectories (subprocesses)",
    "roadrunner": "SBML load (LLVM JIT)",
    "copasi": "SBML import",
}


def _build_reference(name: str, prep: dict[str, Any], *, out_prefix: Path, timeout: float):
    if prep.get("format") == "sbml":
        from .ssa_sbml import build_reference

        return build_reference(name, prep, out_prefix=out_prefix, timeout=timeout)
    cls = REFERENCE_ENGINES[name]
    if cls.INPUT == "net":
        return cls(prep["net"], prep["proto"], out_prefix, timeout=timeout)
    return cls(prep["sbml"], prep["species_names"], prep["name_to_id"], prep["proto"])


def _unsupported(name: str, prep: dict[str, Any]) -> str | None:
    check = getattr(REFERENCE_ENGINES[name], "unsupported", None)
    return check(prep.get("network")) if check else None


def _progress(msg: str) -> None:
    progress(CAPABILITY, msg)


# --------------------------------------------------------------------------- #
# Preparation, ensembles and the verdict
# --------------------------------------------------------------------------- #
def _build_bngsim(prep: dict[str, Any]) -> BngsimSSA:
    if prep.get("format") == "sbml":
        from .ssa_sbml import BngsimSSASBML

        return BngsimSSASBML(prep["sbml"], prep["species_names"], prep["proto"])
    return BngsimSSA(prep["net"], prep["proto"])


def _prepare(problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    proto = protocol(problem)
    if problem.source.get("format") == "sbml":
        from .ssa_sbml import prepare

        return prepare(problem, ctx, proto)
    prep = prepare_network(problem, ctx, proto["netgen"])
    prep["proto"] = proto
    # The harness's own reading of the network: for the exact moments and the
    # repeated-reactant screen. A .net it cannot read costs only those two.
    try:
        prep["network"] = netfile.read(prep["net"])
    except (netfile.NetFileError, ValueError, ArithmeticError) as exc:
        prep["network"] = None
        prep["info"]["netfile_error"] = f"{type(exc).__name__}: {exc}"[:400]
    return prep


def _cold_bngsim(prep: dict[str, Any], n: int, seed: int):
    def fn(stage):
        t0 = time.perf_counter()
        stage(BUILD_STAGE["bngsim"])
        engine = _build_bngsim(prep)
        stage(f"ensemble of {n}")
        t1 = time.perf_counter()
        stack = engine.ensemble_detailed(n, seed)
        return stack, {**engine.config(), "ensemble_seconds": time.perf_counter() - t1,
                       "cold_seconds": time.perf_counter() - t0, "seed": seed}
    return fn


def _cold_reference(name: str, prep: dict[str, Any], n: int, seed: int, out_prefix: Path,
                    timeout: float):
    def fn(stage):
        t0 = time.perf_counter()
        stage(BUILD_STAGE.get(name, "build"))
        ref = _build_reference(name, prep, out_prefix=out_prefix, timeout=timeout)
        stage(f"ensemble of {n}")
        t1 = time.perf_counter()
        stack = ref.ensemble(n, seed)
        return stack, {**ref.config(), "ensemble_seconds": time.perf_counter() - t1,
                       "cold_seconds": time.perf_counter() - t0, "first_seed": seed}
    return fn


def _final_moments(stack: np.ndarray, names: list[str]) -> dict[str, Any]:
    """The ensemble's mean and variance per species at the last time point: a small,
    readable summary for the record (the verdict uses every cell)."""
    last = stack[:, -1, :]
    return {"replicates": int(stack.shape[0]),
            "final_mean": dict(zip(names, map(float, last.mean(axis=0)))),
            "final_variance": dict(zip(names, map(float, last.var(axis=0, ddof=1))))}


def _permutation_seed(seed: int, k: int) -> int:
    """The relabelling seed of comparison ``k``, derived from the protocol's seed."""
    return int(np.random.SeedSequence([int(seed), int(k)]).generate_state(1)[0])


def _kept(prep: dict[str, Any]) -> np.ndarray:
    """The output rows every comparison uses: all of them, except (for an SBML
    model) the instants a time-triggered event fires (ssa_sbml.excluded_rows)."""
    keep = prep.get("keep")
    return np.ones(times(prep["proto"]).size, dtype=bool) if keep is None else keep


def _two_sample(bn: np.ndarray, ref: np.ndarray, k: int, prep: dict[str, Any]) -> dict[str, Any]:
    keep = _kept(prep)
    bn, ref = bn[:, keep], ref[:, keep]
    v = stochastic.two_sample_verdict(
        bn, ref, alpha=VERDICT["alpha"], permutations=VERDICT["permutations"],
        stop_exceedances=VERDICT["stop_exceedances"],
        seed=_permutation_seed(prep["proto"]["seed"], k),
        times=times(prep["proto"])[keep], species=prep["species_names"])
    if "reason" not in v:
        v["parity_core"] = stochastic.parity_core_verdict(bn, ref)
    return v


def _analytic(stack: np.ndarray, moments: tuple[np.ndarray, np.ndarray],
              prep: dict[str, Any]) -> dict[str, Any]:
    mean, var = moments
    keep = _kept(prep)
    return stochastic.analytic_verdict(stack[:, keep], mean[keep], var[keep],
                                       alpha=VERDICT["alpha"],
                                       times=times(prep["proto"])[keep],
                                       species=prep["species_names"])


def exact_moments(prep: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """The spec's exact moments on the protocol grid: computed from the ``.net``
    (``moments: linear``) or read from a pinned table (``kind: table``)."""
    kind = moments_kind(prep["proto"])
    if kind not in MOMENT_KINDS:
        raise ValueError(f"unknown moments kind {kind!r}")
    if kind == "table":
        from .ssa_sbml import table_moments

        return table_moments(prep["proto"]["moments"], prep["cache_dir"], prep["proto"],
                             prep["species_names"])
    if prep.get("network") is None:
        raise RuntimeError("the .net could not be read: "
                           + str(prep["info"].get("netfile_error", "unknown error")))
    t_end, n_steps = _grid(prep["proto"])
    return stochastic.linear_moments(prep["network"], t_end, n_steps)


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
def _derive_isolated(prep: dict[str, Any], workdir: Path, budget: float,
                     timeout: float) -> dict[str, Any]:
    """For an SBML model, the network run_network samples (``ssa_sbml.derive_network``),
    derived in a forked child under ``budget``; run_network not applicable, with the
    reason, when there is no faithful one or the derivation failed (it is model
    preparation, never run_network's failure)."""
    from .ssa_sbml import derive_network

    def fn(stage):
        derived = derive_network(prep["sbml"], prep["species_names"], prep["proto"],
                                 workdir / "derived", timeout=timeout, stage=stage)
        return np.zeros(0), derived

    _progress(f"run_network: deriving a network from the SBML (budget {budget:g} s)")
    out = run_isolated(fn, budget, workdir / "isolated" / "derive")
    if "error" in out:
        derived = {"applicable": False, "info": {},
                   "reason": f"the network derivation failed: {out['error']}"[:400]}
    else:
        derived = out["config"]
    derived.setdefault("info", {})["seconds"] = out["seconds"]
    _progress("run_network: " + ("derived network ready" if derived.get("applicable")
                                 else f"not applicable ({derived.get('reason')})"))
    return derived


def _engine_run(name: str, prep: dict[str, Any], n: int, seed: int, workdir: Path,
                budget: float, timeout: float) -> dict[str, Any]:
    """One engine's correctness ensemble, cold, in its own forked child: what
    ``run_isolated`` returns; for a reference that cannot simulate the model,
    ``{"error": "UNSUPPORTED ..."}`` without running it; for one that does not
    apply to it (run_network on an SBML model with no faithful network),
    ``{"not_applicable": reason}``."""
    def isolated(fn) -> dict[str, Any]:
        _progress(f"{name}: start (budget {budget:g} s; {threads()} threads in the worker)")
        out = run_isolated(fn, budget, workdir / "isolated" / name)
        _progress(f"{name}: {out.get('error') or 'done'} after {out['seconds']:.1f} s")
        return out

    if name == "bngsim":
        return isolated(_cold_bngsim(prep, n, seed_for("bngsim", seed)))
    reason = _unsupported(name, prep)
    if reason:
        _progress(f"{name}: not run, {reason}")
        return {"error": reason}
    extra: dict[str, Any] = {}
    if prep.get("format") == "sbml" and name == "run_network":
        derived = _derive_isolated(prep, workdir, budget, timeout)
        extra["derived_network"] = derived.get("info", {})
        if not derived.get("applicable"):
            return {"not_applicable": str(derived.get("reason")), **extra}
        prep["derived"] = derived
    out = isolated(_cold_reference(name, prep, n, seed_for(name, seed),
                                   workdir / name / "rn", timeout))
    return {**out, **extra}


# What a stored part keeps of ``_engine_run``'s output besides the ensemble.
_PART_FIELDS = ("config", "error", "seconds", "not_applicable", "derived_network")


def _store_part(job: Job, name: str, out: dict[str, Any], ctx: dict[str, Any]) -> None:
    from ..results import write_part

    record = {k: out[k] for k in _PART_FIELDS if k in out}
    arrays = {"trajectory": out["trajectory"]} if "trajectory" in out else None
    write_part(ctx["results_dir"], job, name, ctx["key"], record, arrays)


def _stored_part(job: Job, name: str, ctx: dict[str, Any]) -> dict[str, Any] | None:
    """An engine's ensemble as a part of this job stored it, under the job's
    current key (``Job.parts``); None when there is none."""
    if (job is None or name not in job.parts or not ctx.get("key")
            or not ctx.get("results_dir")):
        return None
    from ..results import read_part

    got = read_part(ctx["results_dir"], job, name, ctx["key"])
    if got is None:
        return None
    rec, arrays = got
    out = {k: rec[k] for k in _PART_FIELDS if k in rec}
    if "trajectory" in arrays:
        out["trajectory"] = arrays["trajectory"]
    elif "error" not in out and "not_applicable" not in out:
        return None
    out["from_part"] = True
    _progress(f"{name}: the ensemble its own task stored ({out.get('seconds', 0):.1f} s there)")
    return out


def run_correctness(job: Job, problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    try:
        prep = _prepare(problem, ctx)
    except Exception as exc:  # noqa: BLE001
        return {"outcome": "BAD_TEST", "error": f"{type(exc).__name__}: {exc}"}
    proto, info = prep["proto"], prep["info"]
    n, seed = int(proto["replicates"]), int(proto["seed"])
    workdir = Path(ctx["workdir"])
    budget = float(ctx.get("engine_budget_seconds") or engine_budget_seconds(problem))
    info["engine_budget_seconds"] = budget
    timeout = min(float(ctx["subprocess_timeout"]), budget)

    part = ctx.get("part")
    if part:
        # One engine only, stored for the run that scores the job (Job.parts).
        _store_part(job, part, _engine_run(part, prep, n, seed, workdir, budget, timeout), ctx)
        return {"outcome": "PART_STORED"}
    outs = {}
    for name in ("bngsim", *REFERENCE_ENGINES):
        outs[name] = _stored_part(job, name, ctx) or _engine_run(
            name, prep, n, seed, workdir, budget, timeout)
    from_parts = [name for name, o in outs.items() if o.get("from_part")]
    if from_parts:
        info["from_parts"] = from_parts

    bn = None
    bn_error = outs["bngsim"].get("error")
    if not bn_error:
        info["bngsim"] = outs["bngsim"]["config"]
        bn = outs["bngsim"]["trajectory"]

    references: dict[str, Any] = {}
    ensembles: dict[str, np.ndarray] = {}
    ref_errors: dict[str, str] = {}
    not_applicable: dict[str, str] = {}
    for name in REFERENCE_ENGINES:
        out = outs[name]
        if "derived_network" in out:
            info["derived_network"] = out["derived_network"]
        if "not_applicable" in out:
            not_applicable[name] = out["not_applicable"]
        elif "error" in out:
            ref_errors[name] = out["error"]
        else:
            ensembles[name] = out["trajectory"]
            references[name] = out["config"]
    info["references"] = references

    moments = None
    n_refs = len(REFERENCE_ENGINES) - len(not_applicable)
    if proto.get("moments"):
        n_refs += 1
        t0 = time.perf_counter()
        try:
            moments = exact_moments(prep)
            info[ANALYTIC] = {"kind": moments_kind(proto), "seconds": time.perf_counter() - t0}
        except Exception as exc:  # noqa: BLE001 - a reference that could not be built
            ref_errors[ANALYTIC] = f"{type(exc).__name__}: {exc}"[:400]

    verdicts: dict[str, Any] = {}
    if bn is not None:
        for name, stack in ensembles.items():
            _progress(f"verdict vs {name}")
            # The relabelling seed follows the engine's registry position, so a
            # verdict does not depend on which other references happened to run.
            verdicts[name] = _two_sample(bn, stack, list(REFERENCE_ENGINES).index(name), prep)
        if moments is not None:
            verdicts[ANALYTIC] = _analytic(bn, moments, prep)
    passed = bool(verdicts) and all(v.get("passed") for v in verdicts.values())

    # Every engine's ensemble against the exact moments too: not scored, but it
    # says which side of a disagreement is wrong.
    if moments is not None and ensembles:
        info["reference_vs_analytic"] = {
            name: {k: v for k, v in _analytic(stack, moments, prep).items()
                   if k in ("passed", "p_value", "p_min", "z_max", "worst", "n_rejected")}
            for name, stack in ensembles.items()}
    names = prep["species_names"]
    info["ensembles"] = {name: _final_moments(stack, names)
                         for name, stack in (("bngsim", bn), *ensembles.items())
                         if stack is not None}
    if moments is not None:
        info["ensembles"][ANALYTIC] = {
            "final_mean": dict(zip(names, map(float, moments[0][-1]))),
            "final_variance": dict(zip(names, map(float, moments[1][-1])))}

    result = outcome(bn_error, ref_errors, n_refs, passed)
    if bn_error and bn_error.startswith("TIMEOUT") and result == "EXCEPTION":
        result = "TIMEOUT"
    rec: dict[str, Any] = {"outcome": result, "measurements": info}
    if verdicts:
        rec["verdicts"] = verdicts
    if bn_error:
        rec["bngsim_error"] = bn_error
    for name, err in ref_errors.items():
        rec[f"{name}_error"] = err
    if not_applicable:
        rec["not_applicable"] = not_applicable
    return rec


def _summary(stack: np.ndarray) -> np.ndarray:
    """What a timed ensemble is checked on: per cell, its mean and mean square.
    The same seeds give the same ensemble, so these must match the first run's."""
    stack = np.asarray(stack, dtype=float)
    return np.concatenate([stack.mean(axis=0), (stack**2).mean(axis=0)], axis=1)


def run_timing(job: Job, problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    try:
        prep = _prepare(problem, ctx)
    except Exception as exc:  # noqa: BLE001 - a bad test input, a labelled row
        return {"outcome": "BAD_TEST", "error": f"{type(exc).__name__}: {exc}"}
    proto, info = prep["proto"], prep["info"]
    workdir = Path(ctx["workdir"])
    timeout = ctx["subprocess_timeout"]
    k, seed = int(proto["timing_replicates"]), int(proto["seed"])
    info["timing_replicates"] = k

    # BNGsim cold: load, propensity compile and a first ensemble, with the empty
    # compile cache the worker gave this job. That ensemble anchors the warm ones.
    t0 = time.perf_counter()
    bngsim_engine = _build_bngsim(prep)
    first = bngsim_engine.ensemble(k, seed_for("bngsim", seed))
    info["bngsim_cold_seconds"] = time.perf_counter() - t0
    info["bngsim"] = bngsim_engine.config()
    anchor = _summary(first)

    # Build each reference and give it a one-trajectory trial run, so one that
    # crashes is recorded and dropped rather than taking the interleave down.
    references: dict[str, Any] = {}
    engines: dict[str, Any] = {"bngsim": bngsim_engine}
    ref_errors: dict[str, str] = {}
    not_applicable: dict[str, str] = {}
    timed_out = correctness_timeouts(job, ctx, REFERENCE_ENGINES)
    for name in REFERENCE_ENGINES:
        reason = _unsupported(name, prep)
        if reason:
            ref_errors[name] = reason
            continue
        if name in timed_out:
            ref_errors[name] = f"TIMEOUT in the correctness job, so not timed ({timed_out[name]})"
            _progress(f"{name}: skipped, {ref_errors[name]}")
            continue
        if prep.get("format") == "sbml" and name == "run_network":
            # Derived as the correctness job derives it, and judged the same way.
            derived = _derive_isolated(prep, workdir, float(timeout), float(timeout))
            if not derived.get("applicable"):
                not_applicable[name] = str(derived.get("reason"))
                continue
            prep["derived"] = derived
        _progress(f"{name}: build + trial run")
        try:
            engine = _build_reference(name, prep, out_prefix=workdir / name / "rn",
                                      timeout=timeout)
            engine.ensemble(1, seed_for(name, seed))
            engines[name] = engine
            references[name] = engine.config()
        except Exception as exc:  # noqa: BLE001
            ref_errors[name] = f"{type(exc).__name__}: {exc}"[:400]
    info["references"] = references

    _progress(f"interleaved timing of {', '.join(engines)}, {k} trajectories per ensemble")
    runs = {name: (lambda e=eng, s=seed_for(name, seed): e.ensemble(k, s))
            for name, eng in engines.items()}
    captures = {name: _summary for name in engines}
    reps = round_robin_interleaved(runs, captures=captures, **job.definition["timing"])

    outputs = {name: reps["engines"][name].pop("outputs") for name in runs}
    repro: dict[str, Any] = {"bngsim": reproducibility(anchor, outputs["bngsim"], REPRO_TOL)}
    for name in engines:
        if name != "bngsim":
            outs = outputs[name]
            repro[name] = reproducibility(outs[0], outs[1:], REPRO_TOL)
    reproducible = all(r["passed"] for r in repro.values())

    times_ = absolute_time_stats({name: reps["engines"][name]["seconds"] for name in runs})
    per_trajectory = {name: {key: t[key] / k for key in ("median_seconds", "ci_low", "ci_high")}
                      for name, t in times_.items()}
    rec: dict[str, Any] = {
        "outcome": "PASS" if reproducible else "NOT_REPRODUCIBLE",
        "measurements": {
            **info,
            "engine_roles": {name: engine_role(name) for name in engines},
            "repetitions": reps,
            "reproducibility": repro,
            "times": times_,
            "times_per_trajectory": per_trajectory,
        },
    }
    for name, err in ref_errors.items():
        rec[f"{name}_error"] = err
    if not_applicable:
        rec["not_applicable"] = not_applicable
    return rec


def run(job: Job, problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    if job.kind == CORRECTNESS:
        return run_correctness(job, problem, ctx)
    if job.kind == TIMING:
        return run_timing(job, problem, ctx)
    raise ValueError(f"unknown job kind {job.kind!r}")
