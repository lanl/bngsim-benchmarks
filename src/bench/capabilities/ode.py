"""ODE simulation of BNGL and SBML models: BNGsim against a registry of reference engines.

BNGsim is the system under test; it is compared, on one problem, against every
reference ODE engine in ``REFERENCE_ENGINES``:

- **run_network** -- BioNetGen's CVODE binary, the trusted dense oracle, run as a
  fresh subprocess the way BNG2.pl runs it (reads the ``.net``).
- **roadrunner** -- libRoadRunner's CVODE, an in-process library that reads SBML
  exported from the same ``.net``, so its verdict also validates the export.
- **copasi** -- COPASI's LSODA (the ``deterministic`` method), an in-process
  library driven through BASICO, reading the same exported SBML.
- **amici** -- AMICI's CVODES, which compiles the exported SBML to a C++/SWIG
  extension once and then integrates the forward problem.

Further SBML-reading references drop in the same way, with the same
warm/run/config contract; nothing else changes when one is added. Adding one
re-ids every ODE job (``engines_id``) and re-runs the corpus.

Pipeline, per problem:

1. **Network generation.** BNG2.pl runs on a copy of the model with every action
   removed and the spec's ``generate_network(...)`` call appended, producing a
   ``.net``. The same ``.net`` feeds BNGsim and run_network.
2. **SBML export.** The ``.net`` is loaded once and written to SBML
   (``bngsim.convert.write_sbml``); the SBML-reading engines read that. Both the
   ``.net`` and the SBML are built once in ``_prepare`` and charged to no engine.
3. Every engine integrates the protocol and returns a species trajectory in
   **BNGsim's column order**.

The correctness job records a verdict of BNGsim against *each* reference and
passes only when BNGsim agrees with every one -- a genuine cross-engine check. A
reference agrees only when bngsim's own deterministic verdict (``parity_core.differ``,
vendored unmodified) passes AND a per-species check does, which judges every compared
species against its own peak, with a floor tied to the protocol's atol (see
``species_check`` and the constants beside ``VERDICT_VERSION``). The job stores the
trajectories it compared, so a verdict change is re-scored (``rescore``) rather than
re-simulated. The timing job
interleaves all the engines round-robin in one drift-cancelling run (``bench.timing``)
and reports each engine's absolute time with a bootstrap CI; a speedup is only a
derived view (``gather``), never a stored headline.

A reference engine that crashes is a result to attribute, not a bug to patch: the
correctness job records it as ``REFERENCE_FAILED`` with a ``<name>_error`` and
still scores BNGsim against the references that ran, and the timing job gives each
reference a trial run, records any crash the same way, and drops it from the timed
set -- BNGsim and the surviving references are still timed. A disagreement takes
precedence: when a reference that ran disagrees with BNGsim the outcome is
``DIFF``, whatever else failed (the failures are still recorded), so a wrong
answer is never timed.

**Per-engine wall budget.** In the correctness job every engine (BNGsim and each
reference) builds and solves cold in its own forked child process, with its own
wall budget (``engine_budget_seconds``, at most ``ENGINE_BUDGET_CAP_SECONDS``). An
engine that does not finish is killed with its whole process tree (AMICI's
compiler included) and recorded as ``<name>_error = "TIMEOUT: ..."``, naming the
stage it was in; a reference timeout is ``REFERENCE_FAILED``, a BNGsim timeout
``TIMEOUT`` -- the paper's per-engine attribution. So one slow reference (AMICI's
C++ compile, libRoadRunner's LLVM build on a 10^4-reaction network) costs at most
its budget and never the other engines' results. A reference that crashes the
process outright (a segfault) is attributed the same way. The timing job does
not re-try a reference that timed out in the correctness job: it records it and
times the rest.

**Column alignment** is the one real correctness risk, and it is self-checking:
run_network reads the same ``.net`` as BNGsim, so its columns share that order by
construction; an SBML engine's columns are aligned to BNGsim's ``species_names``
by a species-name -> SBML-id map (not by position), guarding on an equal count.
If an alignment were wrong, the verdict would fail.

Setting ``auto`` leaves every BNGsim choice (linear solver, compiled or
interpreted right-hand side) to bngsim, as a user gets it.

**SBML models** (the ode-sbml suite) take a second path through the same
registry, with no network generation and no export: BNGsim reads the pinned
SBML with ``Model.from_sbml``, and roadrunner, COPASI and AMICI read the same
file. Their protocol is the model's SED-ML time course, which may start its
output (``output_start``) after the integration starts (``t_start``). What is
compared is each **state species** of the SBML, by its SBML id: every species
the SBML neither declares ``constant`` nor sets by an assignment rule (those are
functions of the parameters and the state, and not every engine reports them).
Every engine must report every state species, at the protocol's output times,
or it has failed; nothing is aligned by position. BNGsim takes ``atol`` as a
plain float, its own reading of an absolute tolerance, as an SBML user gets it
(the ``.net`` path holds it exactly instead, to match run_network).

run_network reads only ``.net``, so for an SBML model it runs on a network
derived from the SBML (``derive_network``): ``bngsim.convert.sbml_to_bngl``
writes compartmental BNGL, whose fixed-time events become phases of the
protocol; BNG2.pl generates the ``.net``; and the phases run as BNG2.pl runs
them, a fresh run_network per phase from the state the last one left. The
derivation is model preparation, charged to no engine, and it gates
run_network: if the SBML has no faithful BNGL form, or BNGsim on the derived
network does not reproduce BNGsim on the SBML, run_network is *not applicable*
to the model. That is recorded with its reason, and it is left out of the count
of references, not counted as a failure. A run_network failure after a faithful
derivation is a reference failure like any other.

The reference for correctness is run_network with its default (dense) linear
solver, as in bngsim's own parity checks. A model's ``sparse`` option (BNG2.pl's
``sparse=>1``, run_network ``-b``, an iterative Krylov solver) is not trusted as
an oracle: on some models it leaves the solution by more than the comparison
tolerance even at tight tolerances. When a model asks for ``sparse``, the
correctness job also records whether the sparse run agrees with the dense one,
and the timing job times run_network with the authors' setting only when it does;
otherwise it times the dense run.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .. import env, fetch
from ..jobs import CORRECTNESS, TIMING, Job
from ..specs import Problem
from ..timing import absolute_time_stats, round_robin_interleaved
# The method-independent parts of the pipeline live in _common, shared with the
# other capabilities; they keep their names here, where they were first written.
from ._common import (  # noqa: F401 - re-exported under their ode names
    all_passed,
    correctness_timeouts,
    export_sbml,
    generate_network,
    max_err_over_peak,
    netgen_bngl,
    network_size,
    prepare_network,
    progress,
    read_dat,
    recorded_command,
    run_isolated,
)
from ._common import engine_role as _engine_role
from ._common import outcome as _outcome
from ._common import reproducibility as _reproducibility
from ._common import threads as _threads

CAPABILITY = "ode"
# Bump when this module's computation changes; every result it wrote goes stale.
# 2: the correctness reference is run_network's dense solver, not the model's
#    sparse option; timing uses sparse only where it agrees with dense.
# 3: correctness records max_err_over_peak.
# 4: BNGsim is compared against a registry of reference engines (run_network and
#    libRoadRunner), not run_network alone; timing interleaves all engines
#    round-robin and reports each engine's absolute time.
# 5: the registry carries COPASI and AMICI as well; the timing job gives each
#    reference a trial run and drops any that crash (recording a <name>_error),
#    so one crashing reference no longer loses BNGsim and the survivors' timings.
HARNESS_VERSION = "5"
# The SBML path's own version, a key only SBML jobs carry, so it moves without
# restaling a .net result (and HARNESS_VERSION moves both).
SBML_HARNESS_VERSION = "1"

TIMING_PROTOCOL = {"warmup_rounds": 1, "min_rounds": 5, "max_rounds": 20, "budget_seconds": 120.0}

# A timing-only part of the key (sibling of the protocol, not inside it): bump to
# restale every timing result without disturbing correctness. 1 adds the
# per-repetition reproducibility check (issue #8).
REPRO_CHECK_VERSION = 1
# Worst a kept repetition may disagree with the first run, as a fraction of each
# species' peak, before the timing is not trusted. bngsim reproduces its first run
# to ~3e-9; a reset that solves another trajectory is off by >=10% of the peak, so
# this sits far from both. Loosen to 1e-4 if a stiff model ever drifts legitimately
# near it (still a 1000x margin over the bug).
REPRO_TOL = 1e-6
LINEAR_SOLVER_NAMES = {0: "dense", 1: "KLU", 2: "LAPACK-dense"}

# The correctness gate: a reference agrees with BNGsim only when bngsim's vendored
# verdict (parity_core.differ, unmodified) passes AND so does a per-species check
# (``species_check``). The vendored verdict can hard-fail a species only when its
# peak is past SIGNIF_FLOOR (1e-3) of the file-wide peak or its range past 1e-3 of
# the largest range, so on a model whose species span many decades most species are
# never judged: of the ode-sbml suite it judged 1 of Nishio 2008's 59 compared
# species and 16 of Smith 2013's 105, and passed AMICI on Smith although AMICI never
# gave the first of its two insulin doses. The per-species check judges every species
# on its own scale: its worst disagreement over the run against the larger of
# SPECIES_TOL of its own peak (over both engines) and SPECIES_ATOL_FACTOR times the
# protocol's atol.
#
# Calibrated on the ode-net and ode-sbml suites (docs/DESIGN.md §6.4):
# - SPECIES_TOL = 1e-2, the vendored movement gate's own ceiling, now on every
#   species. Every comparison that should pass is inside it: the worst is
#   libRoadRunner on Koo 2013 at 4.6e-3 of a species' peak (its own integration
#   error at the SED-ML's rtol of 1e-6, 4.4e-3 from the converged solution), the
#   next Koo 2013 at 5.8e-4 and Verma 2016 at 2.2e-4, and every ode-net comparison
#   is under 1e-4. Every known-bad one is past it by 10x or more: AMICI on Smith
#   2013 (the missing insulin dose, 1.0), COPASI (1.0) and libRoadRunner (0.13) on
#   Winter 2017.
# - SPECIES_ATOL_FACTOR = 10, the vendored MOVE_ATOL_FACTOR: two solvers holding
#   their local error to atol differ by a few atol on a species near zero (bngsim's
#   rr_ode census: up to 3.8 atol on comparisons that pass). Every species of a
#   passing comparison past 1e-2 of its peak is under 2e-8 atol (one engine reports
#   round-off where the other holds exact zero, e.g. TP53 Ser15~p on Lang 2024 at
#   1e-26 against atol 1e-12), so the floor costs none of them.
# Bump VERDICT_VERSION whenever the check, its constants, or how the two halves are
# combined change. It is a correctness-only part of the job definition (as
# timing_check is timing-only), so every ODE correctness result goes stale -- and
# ``bench rescore`` re-scores it from its stored trajectories -- while no timing
# job's definition, id or key changes. The outcome rule (``_common.OUTCOME_RULE``)
# is part of the scoring too, so it moves this as well.
# 1: the vendored verdict AND the per-species check.
# 2: outcome rule 2 -- a disagreement is DIFF even when another reference failed.
VERDICT_VERSION = 2
SPECIES_TOL = 1e-2
SPECIES_ATOL_FACTOR = 10.0


def protocol(problem: Problem) -> dict[str, Any]:
    """The ODE protocol with defaults filled in, as it enters a job definition.

    An SBML protocol's output starts where its integration does unless it says
    otherwise; a BNGL protocol has run_network's ``sparse`` option instead."""
    p = dict(problem.protocol_for(CAPABILITY))
    p.setdefault("t_start", 0)
    if problem.source.get("format") == "sbml":
        p.setdefault("output_start", p["t_start"])
    else:
        p.setdefault("sparse", False)
    return p


def output_times(proto: dict[str, Any]) -> np.ndarray:
    """The protocol's output times: ``n_steps`` equal steps from the output start
    (the integration start unless the protocol names another) to ``t_end``."""
    start = float(proto.get("output_start", proto["t_start"]))
    return np.linspace(start, float(proto["t_end"]), int(proto["n_steps"]) + 1)


def engines_id() -> str:
    """The ``engines`` id segment: BNGsim versus every registered reference.

    Derived from the populated registry, so adding a reference engine re-ids the
    jobs and re-runs them, which is the intended behaviour.
    """
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
        "setting": {"solver": "auto", "codegen": "auto"},
        # The reference-engine versions key only this capability's freshness, not
        # the global per-kind fingerprint: a roadrunner/copasi/amici change restales
        # ode results, never a future capability that does not use it (see
        # env.fingerprint). run_network's version rides the global fingerprint.
        "reference_versions": {
            "roadrunner": env.roadrunner_identity().get("version"),
            "copasi": env.copasi_identity().get("version"),
            "amici": env.amici_identity().get("version"),
        },
    }
    if fmt == "sbml":
        # Only an SBML job carries this, so a .net job's definition is unchanged.
        base["input"] = {"format": "sbml", "harness": SBML_HARNESS_VERSION}
    est_c, est_t = planning_seconds(problem)
    # The per-engine budgets bound the engines; the job timeout is the backstop for
    # what they do not cover (netgen, export, the sparse check), so it leaves room
    # for every engine to use its whole budget.
    backstop = (len(REFERENCE_ENGINES) + 2) * engine_budget_seconds(problem)
    correctness = Job(
        CAPABILITY, problem.id, engines, "auto", CORRECTNESS,
        # A correctness-only part of the key (as timing_check is timing-only): a
        # verdict change restales every correctness result, which `bench rescore`
        # re-scores from its stored trajectories, and no timing result.
        definition={**base, "verdict": VERDICT_VERSION},
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


# Planning only: never part of a freshness key. A job's cost is dominated by work
# that grows faster than the network -- AMICI's SBML-to-C++ compile, libRoadRunner's
# SBML read, run_network's solve -- so the estimate grows as reactions**1.5. The
# constant is calibrated on beta_catenin_destruction_complex_barua2013 (2737
# reactions, about 5 min for correctness on x86, AMICI's compile 220 s of it) and is
# provisional until the larger core models have run. The timeout leaves room for a
# slower node; a deadline still stops a job that would not finish in its allocation.
MIN_TIMEOUT_SECONDS = 1800.0
TIMEOUT_FACTOR = 4.0
# The most wall time any one engine gets to build and solve cold in the correctness
# job. The bngsim paper uses a 20-min per-engine wall limit; this is looser so a
# reference that is merely slow (run_network on a 2.7e4-reaction network, ~20 min)
# still yields a verdict, and tight enough that netgen + BNGsim + four references
# fit a 10 h allocation. Like a job timeout it is planning, not part of the
# freshness key: an engine that finishes inside it gives the same result either
# way; a result records the budget it ran under.
ENGINE_BUDGET_CAP_SECONDS = 3600.0


def engine_budget_seconds(problem: Problem) -> float:
    """Wall budget for one engine's cold build + solve in the correctness job."""
    est_c, _ = planning_seconds(problem)
    return min(ENGINE_BUDGET_CAP_SECONDS, max(MIN_TIMEOUT_SECONDS, TIMEOUT_FACTOR * est_c))


def planning_seconds(problem: Problem) -> tuple[float, float]:
    """Rough wall-clock estimates (correctness, timing) for one problem."""
    reactions = problem.size.get("reactions") or 10 * problem.size.get("species", 100)
    correctness = 30.0 + 0.0021 * reactions ** 1.5
    # Timing repeats every engine's solve over at least min_rounds plus warmup.
    timing = 60.0 + 1.5 * correctness
    return correctness, timing


# --------------------------------------------------------------------------- #
# The system under test and the reference engines
# --------------------------------------------------------------------------- #
class BngsimODE:
    """One loaded BNGsim model, ready to integrate the protocol repeatedly.

    The system under test, timed warm (model loaded, state reset between runs).
    """

    def __init__(self, net: Path, proto: dict[str, Any]):
        import bngsim

        self.proto = proto
        t0 = time.perf_counter()
        self.model = bngsim.Model.from_net(str(net))
        self.load_seconds = time.perf_counter() - t0
        self.sim = bngsim.Simulator(self.model, method="ode")
        # run_network holds -a atol exactly for every species; bngsim reads a plain
        # float as a per-species ceiling, so the comparison passes it exactly.
        self.atol = bngsim.ExactAtol(float(proto["atol"]))

    def run(self):
        p = self.proto
        return self.sim.run(
            t_span=(float(p["t_start"]), float(p["t_end"])),
            n_points=int(p["n_steps"]) + 1,
            rtol=float(p["rtol"]),
            atol=self.atol,
        )

    def warm(self):
        self.model.reset()
        return self.run()

    @staticmethod
    def trajectory(result) -> np.ndarray:
        """The species trajectory, copied: ``.species`` may be a buffer the next
        run overwrites, which would make an anchor silently track a later run."""
        return np.array(result.species)

    def config(self, result) -> dict[str, Any]:
        stats = getattr(result, "solver_stats", None) or {}
        code = stats.get("linear_solver")
        return {
            "codegen_backend": self.sim.codegen_backend,
            "codegen_cache_hit": self.sim.codegen_cache_hit,
            "jacobian": self.sim.jacobian_strategy,
            "linear_solver": LINEAR_SOLVER_NAMES.get(code, f"kind_{code}"),
            "load_seconds": self.load_seconds,
            "jacobian_seconds": float(self.sim.last_jacobian_sec),
            "codegen_seconds": float(self.sim.last_codegen_sec),
        }


def run_network_command(net: Path, proto: dict[str, Any], out_prefix: Path) -> list[str]:
    """The run_network call BNG2.pl makes for this simulate."""
    t_start, t_end, n_steps = float(proto["t_start"]), float(proto["t_end"]), int(proto["n_steps"])
    cmd = [
        # On a host whose glibc is too old to load run_network, this prefixes the
        # command with a newer loader (env.run_network_launch); [] otherwise.
        *env.run_network_launch(),
        str(env.run_network_bin()), "-o", str(out_prefix), "-p", "cvode",
        "-a", repr(float(proto["atol"])), "-r", repr(float(proto["rtol"])),
    ]
    if proto.get("sparse"):
        cmd.append("-b")
    if t_start != 0.0:
        cmd += ["-i", repr(t_start)]
    cmd += [
        "--cdat", "1", "--fdat", "0", "-g", str(net), str(net),
        repr((t_end - t_start) / n_steps), str(n_steps),
    ]
    return cmd


def _run_network(cmd: list[str], out_prefix: Path, *, timeout: float):
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if proc.returncode != 0:
        tail = (proc.stderr.strip() or proc.stdout.strip()).splitlines()
        raise RuntimeError(f"run_network failed: {(tail[-1] if tail else '')[:300]}")
    cdat = Path(f"{out_prefix}.cdat")
    if not cdat.exists():
        raise RuntimeError("run_network produced no .cdat")
    return read_dat(cdat)


class RunNetworkODE:
    """run_network as a reference engine: a fresh subprocess per run.

    There is no warm run_network -- every BNG simulation is a fresh process that
    reads the ``.net`` -- so its timed unit is a full invocation. Its columns are
    in the ``.net`` species order, which is BNGsim's, so no name map is needed.
    """

    NAME = "run_network"
    INPUT = "net"

    def __init__(self, net: Path, proto: dict[str, Any], out_prefix: Path, *,
                 sparse: bool, timeout: float):
        self.cmd = run_network_command(net, dict(proto, sparse=sparse), out_prefix)
        self.out_prefix = out_prefix
        self.timeout = timeout
        self.sparse = sparse

    def run(self) -> np.ndarray:
        _, values, _ = _run_network(self.cmd, self.out_prefix, timeout=self.timeout)
        return values

    # A fresh process each time; there is no cheaper warm run to distinguish.
    warm = run

    @staticmethod
    def trajectory(values: np.ndarray) -> np.ndarray:
        return np.array(values)

    def config(self) -> dict[str, Any]:
        return {"solver": "sparse (-b)" if self.sparse else "dense",
                "command": recorded_command(self.cmd)}


class RoadRunnerODE:
    """libRoadRunner as a reference engine, reading SBML exported from the ``.net``.

    An in-process library, so it is timed warm like BNGsim: the model is loaded
    once (its cost a separate cold measurement) and each run is ``reset()`` +
    ``simulate()``. Species are selected as **amounts** by SBML id, in BNGsim's
    column order, via the name map -- BNG-derived SBML has a unit compartment, so
    an amount equals a concentration numerically, and an amount is what BNGsim's
    ``.species`` reports. If ``reset()`` ever proved too weak (an event changing a
    parameter), the per-run reproducibility check would trip and the adapter would
    switch to ``resetToOrigin()``.
    """

    NAME = "roadrunner"
    INPUT = "sbml"

    def __init__(self, sbml: Path, species_names: list[str], name_to_id: dict[str, str],
                 proto: dict[str, Any]):
        roadrunner = env.import_roadrunner()

        missing = [n for n in species_names if n not in name_to_id]
        if missing:
            raise RuntimeError(
                f"cannot align {len(missing)} BNGsim species to SBML ids "
                f"(e.g. {missing[:3]}); export/name map is wrong")
        self.selections = [name_to_id[n] for n in species_names]  # bare ids = amounts
        self.proto = proto
        t0 = time.perf_counter()
        self.rr = roadrunner.RoadRunner(str(sbml))
        self.load_seconds = time.perf_counter() - t0
        self.rr.setIntegrator("cvode")
        integrator = self.rr.getIntegrator()
        integrator.setValue("relative_tolerance", float(proto["rtol"]))
        integrator.setValue("absolute_tolerance", float(proto["atol"]))
        self.rr.timeCourseSelections = ["time"] + self.selections

    def run(self):
        p = self.proto
        return self.rr.simulate(float(p["t_start"]), float(p["t_end"]), int(p["n_steps"]) + 1)

    def warm(self):
        self.rr.reset()
        return self.run()

    @staticmethod
    def trajectory(result) -> np.ndarray:
        # Columns are [time] + the selected ids in BNGsim order; drop time.
        return np.asarray(result)[:, 1:]

    def config(self) -> dict[str, Any]:
        return {
            "integrator": "cvode",
            "version": env.roadrunner_identity().get("version"),
            "output": "amount",
            "load_seconds": self.load_seconds,
        }


class CopasiODE:
    """COPASI as a reference engine, reading SBML exported from the ``.net``.

    Driven through BASICO (``copasi-basico``, imported as ``basico``), a
    pure-Python wrapper over COPASI's SWIG bindings. Like roadrunner it is an
    in-process library timed warm: the model is loaded once (its cost a separate
    cold measurement) and each run is one ``run_time_course`` call, which with
    BASICO's default ``use_initial_values=True`` resets to initial conditions
    first -- the same reset-and-rerun contract as roadrunner (verified: repeated
    runs reproduce to the bit).

    Output is read as **concentration**, not particle number. BNG-derived SBML has
    a unit compartment, so a species' concentration equals its amount numerically,
    which is what BNGsim's ``.species`` reports; COPASI's particle *number* is that
    amount times Avogadro, so ``use_numbers`` would be off by ~6e23. Columns are
    keyed by SBML id (``use_sbml_id=True``) and reordered to BNGsim's column order
    by the name map, never by position.
    """

    NAME = "copasi"
    INPUT = "sbml"

    def __init__(self, sbml: Path, species_names: list[str], name_to_id: dict[str, str],
                 proto: dict[str, Any]):
        import basico

        missing = [n for n in species_names if n not in name_to_id]
        if missing:
            raise RuntimeError(
                f"cannot align {len(missing)} BNGsim species to SBML ids "
                f"(e.g. {missing[:3]}); export/name map is wrong")
        self.selections = [name_to_id[n] for n in species_names]
        self.proto = proto
        t0 = time.perf_counter()
        # Keep the data-model handle and pass it to every call: BASICO otherwise
        # operates on a single global "current model", which concurrent engines
        # in one process would race on. (The suite times serially, but this keeps
        # the adapter correct regardless.)
        self.dm = basico.load_model(str(sbml))
        self.load_seconds = time.perf_counter() - t0

    def run(self):
        import basico

        p = self.proto
        t_start, t_end, n_steps = float(p["t_start"]), float(p["t_end"]), int(p["n_steps"])
        # 3-arg form is (start, duration, intervals); the middle arg is a duration,
        # not an end time, so pass t_end - t_start.
        df = basico.run_time_course(
            t_start, t_end - t_start, n_steps,
            method="deterministic",  # COPASI's LSODA
            r_tol=float(p["rtol"]), a_tol=float(p["atol"]),
            automatic=False,  # honour the fixed interval count, not adaptive output
            use_concentrations=True,  # amounts for a unit compartment; NOT use_numbers
            use_sbml_id=True,  # columns keyed by SBML id, for alignment by name map
            model=self.dm,
        )
        return df[self.selections].to_numpy()

    # BASICO resets to initial conditions each call (use_initial_values defaults
    # True); there is no cheaper warm path to distinguish.
    warm = run

    @staticmethod
    def trajectory(values: np.ndarray) -> np.ndarray:
        # run() already selected the species columns in BNGsim order.
        return np.asarray(values)

    def config(self) -> dict[str, Any]:
        return {
            "method": "deterministic (LSODA)",
            "version": env.copasi_identity().get("version"),
            "output": "amount",
            "load_seconds": self.load_seconds,
        }


class AmiciODE:
    """AMICI as a reference engine, reading SBML exported from the ``.net``.

    AMICI compiles the SBML model to a C++/SWIG Python extension and integrates it
    with CVODES. The compile is done once in ``__init__`` -- a cold cost, like
    roadrunner's model load, recorded as ``codegen_seconds`` -- into a directory
    under the job's node-local scratch (``sbml.parent``), which the worker tears
    down afterwards. Each run then uses a **fresh solver** (``create_solver`` makes
    a new one each call), so there is no carried solver state between runs; that
    sidesteps the reused-solver root-finding leak in AMICI 1.2.0 (issue #3512) and
    gives the same independent-integration-from-initial-state contract roadrunner's
    ``reset()`` does.

    Columns are read as **amounts** (``rdata.x``, the model state) and aligned to
    BNGsim's order by SBML id via the name map, never by position; BNG-derived SBML
    has a unit compartment, so an amount equals a concentration numerically and is
    what BNGsim's ``.species`` reports. The pinned version is 1.1.0; see
    ``pixi.toml`` for why 1.2.0 is held back. The 1.x API is snake_case throughout,
    with the solver enums living in ``amici.sim.sundials`` (not top-level
    ``amici``).
    """

    NAME = "amici"
    INPUT = "sbml"

    def __init__(self, sbml: Path, species_names: list[str], name_to_id: dict[str, str],
                 proto: dict[str, Any]):
        import amici

        missing = [n for n in species_names if n not in name_to_id]
        if missing:
            raise RuntimeError(
                f"cannot align {len(missing)} BNGsim species to SBML ids "
                f"(e.g. {missing[:3]}); export/name map is wrong")
        want_ids = [name_to_id[n] for n in species_names]
        self.proto = proto

        # Compile the model once into the job's node-local scratch. sbml2amici
        # shells out to $CC/$CXX (the conda compiler, per pixi.toml) to build a
        # fresh extension; generate_sensitivity_code=False keeps it to the forward
        # problem, which is all a trajectory comparison needs.
        outdir = sbml.parent / "amici_model"
        t0 = time.perf_counter()
        amici.SbmlImporter(str(sbml)).sbml2amici(
            "bench_model", str(outdir), generate_sensitivity_code=False, verbose=False)
        self.codegen_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        self.model = amici.import_model_module("bench_model", str(outdir)).get_model()
        p = proto
        self.model.set_timepoints(
            np.linspace(float(p["t_start"]), float(p["t_end"]), int(p["n_steps"]) + 1))
        self.load_seconds = time.perf_counter() - t1

        # rdata.x columns follow model.get_state_ids() (the SBML ids), NOT
        # get_state_ids_solver() (the conservation-law-reduced solver state, a
        # different length). Map BNGsim's ids onto those columns, guarding on any
        # that AMICI does not expose.
        pos = {sid: i for i, sid in enumerate(self.model.get_state_ids())}
        unmapped = [i for i in want_ids if i not in pos]
        if unmapped:
            raise RuntimeError(
                f"cannot align {len(unmapped)} SBML ids to AMICI state ids "
                f"(e.g. {unmapped[:3]}); the SBML export and AMICI disagree")
        self.idx = [pos[i] for i in want_ids]

    def _simulate(self):
        """One forward run from the initial state: the run data and its state."""
        from amici.sim.sundials import (
            LinearMultistepMethod, SensitivityOrder, run_simulation)

        p = self.proto
        # Fresh solver per run: cheap, and it carries no state between runs.
        solver = self.model.create_solver()
        solver.set_relative_tolerance(float(p["rtol"]))
        solver.set_absolute_tolerance(float(p["atol"]))
        solver.set_sensitivity_order(SensitivityOrder.none)
        solver.set_linear_multistep_method(LinearMultistepMethod.BDF)
        rdata = run_simulation(self.model, solver)
        if rdata.status != 0:
            raise RuntimeError(f"AMICI simulation failed with status {rdata.status}")
        x = np.asarray(rdata.x)
        # A success status does not by itself guarantee finite output (AMICI issue
        # #3513 returns NaN with status 0); and a silent NaN would slip past the
        # verdict's nan-skipping error metric, so reject it here.
        if not np.all(np.isfinite(x)):
            raise RuntimeError("AMICI returned non-finite values with a success status")
        return rdata, x

    def run(self):
        _, x = self._simulate()
        return x[:, self.idx]

    # A fresh solver each run already integrates from the initial state, so warm
    # and cold do the same work (the cost that differs, the compile, is in __init__).
    warm = run

    @staticmethod
    def trajectory(values: np.ndarray) -> np.ndarray:
        # run() already selected the state columns in BNGsim order.
        return np.asarray(values)

    def config(self) -> dict[str, Any]:
        return {
            "integrator": "CVODES BDF",
            "version": env.amici_identity().get("version"),
            "sensitivities": "none",
            "output": "amount",
            "load_seconds": self.load_seconds,
            "codegen_seconds": self.codegen_seconds,
        }


# The reference engines BNGsim is compared against, in a fixed order. run_network
# builds from the ``.net``; roadrunner, copasi and amici from the SBML.
REFERENCE_ENGINES: dict[str, type] = {
    RunNetworkODE.NAME: RunNetworkODE,
    RoadRunnerODE.NAME: RoadRunnerODE,
    CopasiODE.NAME: CopasiODE,
    AmiciODE.NAME: AmiciODE,
}


# --------------------------------------------------------------------------- #
# SBML models: the same engines, reading the pinned SBML
# --------------------------------------------------------------------------- #
def _time_scale(proto: dict[str, Any]) -> float:
    """The size of the protocol's times, for telling two of them apart."""
    return max(1.0, abs(float(proto["t_start"])), abs(float(proto["t_end"])))


def _close(a: float, b: float, scale: float) -> bool:
    return abs(float(a) - float(b)) <= 1e-9 * scale


def check_times(times, proto: dict[str, Any]) -> None:
    """Refuse an engine's output unless it is at exactly the protocol's output
    times. The verdict compares rows, never times, so an engine that ran another
    horizon or grid would otherwise be scored row by row against BNGsim."""
    grid = output_times(proto)
    t = np.asarray(times, dtype=float).ravel()
    if t.shape != grid.shape or not np.all(np.abs(t - grid) <= 1e-9 * _time_scale(proto)):
        got = f"{t.size} times from {t[0]:g} to {t[-1]:g}" if t.size else "no times"
        raise RuntimeError(
            f"output is not at the protocol's times: {got}, where the protocol has "
            f"{grid.size} from {grid[0]:g} to {grid[-1]:g}")


def sbml_species(sbml: Path) -> tuple[list[str], dict[str, Any]]:
    """The SBML's state species, in document order, and a census of the file.

    A state species is one the SBML neither declares ``constant`` nor sets by an
    assignment rule: what an ODE solver integrates, and what every engine
    compares. The census records the others by id, so a reader sees what is left
    out and why.
    """
    import libsbml

    doc = libsbml.readSBMLFromFile(str(sbml))
    model = doc.getModel()
    if model is None:
        errors = [doc.getError(i).getMessage().strip() for i in range(doc.getNumErrors())]
        raise RuntimeError(f"libsbml cannot read {sbml.name}: {'; '.join(errors)[:300]}")
    rule_set = {model.getRule(i).getVariable() for i in range(model.getNumRules())
                if model.getRule(i).isAssignment()}
    state, constant, assigned = [], [], []
    for i in range(model.getNumSpecies()):
        sp = model.getSpecies(i)
        sid = sp.getId()
        if sp.getConstant():
            constant.append(sid)
        elif sid in rule_set:
            assigned.append(sid)
        else:
            state.append(sid)
    if not state:
        raise RuntimeError(f"{sbml.name} has no state species to compare")
    census = {
        "species": model.getNumSpecies(),
        "reactions": model.getNumReactions(),
        "events": model.getNumEvents(),
        "compared_species": len(state),
        "not_compared": {"constant": constant, "assignment_rule": assigned},
    }
    return state, census


def _bngsim_sbml_run(sim, proto: dict[str, Any], *, rtol: float, atol):
    """One BNGsim solve of an SBML protocol. An output start after the integration
    start is a first sample time at the integration start, dropped afterwards."""
    t0, t1 = float(proto["t_start"]), float(proto["t_end"])
    grid = output_times(proto)
    if _close(grid[0], t0, _time_scale(proto)):
        return sim.run(t_span=(t0, t1), n_points=grid.size, rtol=rtol, atol=atol)
    return sim.run(sample_times=[t0, *grid.tolist()], rtol=rtol, atol=atol)


def _leading_rows(proto: dict[str, Any]) -> int:
    """Rows a solve reports before the output starts: the integration start, when
    the output starts later."""
    grid = output_times(proto)
    return 0 if _close(grid[0], float(proto["t_start"]), _time_scale(proto)) else 1


def _bngsim_columns(result, species_ids: list[str]) -> list[int]:
    """BNGsim's column of each compared SBML species, by SBML id (never by
    position). A result's names, not the model's: an SBML event that assigns a
    parameter makes it a model state entry that the reported species omit."""
    pos = {name: i for i, name in enumerate(result.species_names)}
    missing = [sid for sid in species_ids if sid not in pos]
    if missing:
        raise RuntimeError(
            f"BNGsim reports no column for {len(missing)} compared SBML species "
            f"(e.g. {missing[:3]})")
    return [pos[sid] for sid in species_ids]


class BngsimSBML(BngsimODE):
    """BNGsim reading the pinned SBML itself (``Model.from_sbml``).

    It reports concentrations, as every SBML engine here is asked to. ``atol`` is
    the plain float an SBML user passes, which BNGsim reads as a per-species
    ceiling; the ``.net`` path holds it exactly, to match run_network's reading,
    but no SBML reference reads it that way (libRoadRunner scales it per species),
    and on a model whose species are nanomolar an exact 1e-9 is no tolerance.
    """

    def __init__(self, sbml: Path, species_ids: list[str], proto: dict[str, Any]):
        import bngsim

        self.proto = proto
        self.species_ids = list(species_ids)
        self.lead = _leading_rows(proto)
        self.columns: list[int] | None = None
        t0 = time.perf_counter()
        self.model = bngsim.Model.from_sbml(str(sbml))
        self.load_seconds = time.perf_counter() - t0
        self.sim = bngsim.Simulator(self.model, method="ode")
        self.atol = float(proto["atol"])

    def run(self):
        return _bngsim_sbml_run(self.sim, self.proto, rtol=float(self.proto["rtol"]),
                                atol=self.atol)

    def trajectory(self, result) -> np.ndarray:
        if self.columns is None:
            self.columns = _bngsim_columns(result, self.species_ids)
        return np.array(result.species)[self.lead:, self.columns]

    def times(self, result) -> np.ndarray:
        return np.asarray(result.time)[self.lead:]

    def config(self, result) -> dict[str, Any]:
        return {**super().config(result), "input": "sbml (Model.from_sbml)",
                "atol": "float (per-species ceiling)"}


class RoadRunnerSBML(RoadRunnerODE):
    """libRoadRunner reading the pinned SBML.

    Each compared species is selected as a **concentration** (``[id]``), which is
    what BNGsim reports for an SBML model in any compartment. A warm run follows
    ``resetAll()``, not ``reset()``: an SBML event may assign a global parameter,
    which ``reset()`` would leave at its assigned value (issue #8).
    """

    def __init__(self, sbml: Path, species_ids: list[str], proto: dict[str, Any]):
        roadrunner = env.import_roadrunner()

        self.proto = proto
        self.lead = _leading_rows(proto)
        t0 = time.perf_counter()
        self.rr = roadrunner.RoadRunner(str(sbml))
        self.load_seconds = time.perf_counter() - t0
        self.rr.setIntegrator("cvode")
        integrator = self.rr.getIntegrator()
        integrator.setValue("relative_tolerance", float(proto["rtol"]))
        integrator.setValue("absolute_tolerance", float(proto["atol"]))
        self.rr.timeCourseSelections = ["time"] + [f"[{sid}]" for sid in species_ids]

    def run(self):
        p = self.proto
        grid = output_times(p)
        if self.lead:
            return self.rr.simulate(times=[float(p["t_start"]), *grid.tolist()])
        return self.rr.simulate(float(p["t_start"]), float(p["t_end"]), grid.size)

    def warm(self):
        self.rr.resetAll()
        return self.run()

    def trajectory(self, result) -> np.ndarray:
        return np.asarray(result)[self.lead:, 1:]

    def times(self, result) -> np.ndarray:
        return np.asarray(result)[self.lead:, 0]

    def config(self) -> dict[str, Any]:
        return {**super().config(), "output": "concentration", "reset": "resetAll"}


class CopasiSBML(CopasiODE):
    """COPASI reading the pinned SBML: columns by SBML id, as concentrations.

    An output start after the integration start is asked for as explicit output
    times (BASICO ``values``) over the whole duration; otherwise the call is the
    ``.net`` path's. A compared species COPASI leaves out of its output is a
    COPASI failure, as on the ``.net`` path (never filled in here).
    """

    def __init__(self, sbml: Path, species_ids: list[str], proto: dict[str, Any]):
        import basico

        self.selections = list(species_ids)
        self.proto = proto
        t0 = time.perf_counter()
        self.dm = basico.load_model(str(sbml))
        self.load_seconds = time.perf_counter() - t0

    def run(self):
        import basico

        p = self.proto
        t_start, t_end, n_steps = float(p["t_start"]), float(p["t_end"]), int(p["n_steps"])
        kw = dict(method="deterministic", r_tol=float(p["rtol"]), a_tol=float(p["atol"]),
                  automatic=False, use_concentrations=True, use_sbml_id=True, model=self.dm)
        if _leading_rows(p):
            return basico.run_time_course(duration=t_end - t_start,
                                          values=output_times(p).tolist(), **kw)
        return basico.run_time_course(t_start, t_end - t_start, n_steps, **kw)

    warm = run

    def trajectory(self, df) -> np.ndarray:
        return df[self.selections].to_numpy()

    @staticmethod
    def times(df) -> np.ndarray:
        return np.asarray(df.index, dtype=float)


class AmiciSBML(AmiciODE):
    """AMICI reading the pinned SBML, from the protocol's start time. A compared
    species must be an AMICI state, read by SBML id."""

    def __init__(self, sbml: Path, species_ids: list[str], proto: dict[str, Any]):
        import amici

        self.proto = proto
        outdir = sbml.parent / "amici_model"
        t0 = time.perf_counter()
        amici.SbmlImporter(str(sbml)).sbml2amici(
            "bench_model", str(outdir), generate_sensitivity_code=False, verbose=False)
        self.codegen_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        self.model = amici.import_model_module("bench_model", str(outdir)).get_model()
        self.model.set_t0(float(proto["t_start"]))
        self.model.set_timepoints(output_times(proto))
        self.load_seconds = time.perf_counter() - t1
        pos = {sid: i for i, sid in enumerate(self.model.get_state_ids())}
        unmapped = [sid for sid in species_ids if sid not in pos]
        if unmapped:
            raise RuntimeError(
                f"AMICI has no state for {len(unmapped)} compared SBML species "
                f"(e.g. {unmapped[:3]})")
        self.idx = [pos[sid] for sid in species_ids]

    def run(self):
        rdata, x = self._simulate()
        return np.asarray(rdata.ts, dtype=float), x[:, self.idx]

    warm = run

    @staticmethod
    def trajectory(values) -> np.ndarray:
        return np.asarray(values[1])

    @staticmethod
    def times(values) -> np.ndarray:
        return np.asarray(values[0])


# -- run_network on a network derived from the SBML --------------------------
# The derived network is checked with BNGsim on both sides at this tolerance
# (float atol on both), tight enough that what remains is the model, not the
# integration.
FAITHFULNESS_RTOL = 1e-10
FAITHFULNESS_ATOL = 1e-18
_NET_SPECIES_RE = re.compile(r"^(\s*\d+\s+)(\S+)(\s+)(\S+)(.*)$", re.DOTALL)


def net_species(lines: list[str]) -> list[tuple[int, str]]:
    """``(line index, name)`` of each species of a ``.net``, in order."""
    out: list[tuple[int, str]] = []
    inside = False
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("begin species"):
            inside = True
        elif s.startswith("end species"):
            break
        elif inside and s and not s.startswith("#"):
            m = _NET_SPECIES_RE.match(line)
            if m:
                out.append((i, m.group(2)))
    return out


def net_groups(lines: list[str]) -> dict[str, list[tuple[int, float]]]:
    """Each group (observable) of a ``.net``: its name and its ``(species index,
    weight)`` members, the index 0-based, as BNG2.pl resolved the pattern."""
    out: dict[str, list[tuple[int, float]]] = {}
    inside = False
    for line in lines:
        s = line.strip()
        if s.startswith("begin groups"):
            inside = True
        elif s.startswith("end groups"):
            break
        elif inside and s and not s.startswith("#"):
            fields = s.split()
            if len(fields) < 2:
                continue
            members = []
            for item in (fields[2].split(",") if len(fields) > 2 else []):
                weight, _, index = item.rpartition("*")
                members.append((int(index) - 1, float(weight) if weight else 1.0))
            out[fields[1]] = members
    return out


def species_key(name: str) -> tuple[str | None, str]:
    """``(compartment, molecule)`` of a one-molecule BNG species name, in either
    form BNG writes it (``@c::$M()`` in a ``.net``, ``@c:M()`` in an action);
    a fixed species' ``$`` is left out, as BNG2.pl leaves it out."""
    s = name.lstrip("$")
    comp = None
    if s.startswith("@"):
        comp, _, s = s[1:].partition(":")
        s = s.lstrip(":")
    s = s.lstrip("$")
    return comp, s[:-2] if s.endswith("()") else s


def align_derived(species_ids: list[str], lines: list[str]) -> tuple[list[int], list[str]]:
    """The derived network's species index of each compared SBML species, and the
    ids it cannot place. The converter writes an observable named after each SBML
    species, which BNG2.pl resolves to that species' index; failing that, the
    molecule is named after the id. Never by position."""
    rows = net_species(lines)
    groups = net_groups(lines)
    by_molecule: dict[str, list[int]] = {}
    for k, (_, name) in enumerate(rows):
        by_molecule.setdefault(species_key(name)[1], []).append(k)
    columns, missing = [], []
    for sid in species_ids:
        members = groups.get(sid)
        if members is not None and len(members) == 1 and members[0][1] == 1.0:
            columns.append(members[0][0])
        elif members is None and len(by_molecule.get(sid, [])) == 1:
            columns.append(by_molecule[sid][0])
        else:
            missing.append(sid)
    return columns, missing


def sample_phase(t0: float, t1: float, grid: np.ndarray, scale: float) -> dict[str, Any]:
    """How run_network samples one phase ``[t0, t1]``: the protocol's output times
    inside it and its end. ``{"dt", "n"}`` (run_network's ``sample_time n``) when
    those are equal steps from ``t0``, as for a single phase on the protocol's own
    grid; else ``{"times"}``, which run_network needs three of, so a phase with
    fewer gets midpoints (rows no output time reads)."""
    pts = [float(g) for g in grid if g > t0 + 1e-9 * scale and g <= t1 + 1e-9 * scale]
    if not pts or not _close(pts[-1], t1, scale):
        pts.append(float(t1))
    h = (t1 - t0) / len(pts)
    if all(_close(p, t0 + (k + 1) * h, scale) for k, p in enumerate(pts)):
        return {"dt": h, "n": len(pts)}
    while len(pts) < 3:
        knots = [t0, *pts]
        k = max(range(len(pts)), key=lambda i: knots[i + 1] - knots[i])
        pts.insert(k, (knots[k] + knots[k + 1]) / 2)
    return {"times": pts}


def stitch(outputs: list[tuple[np.ndarray, np.ndarray]], phases: list[dict[str, Any]],
           grid: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """The protocol's output rows from the phases' outputs. Each phase's rows are
    its start and its samples; an output time at a boundary between phases is the
    earlier phase's last row, the state before the next phase's changes."""
    times, rows = [], []
    for g in grid:
        if _close(g, phases[0]["t_start"], scale):
            j = 0
        else:
            j = next((j for j, ph in enumerate(phases)
                      if g > ph["t_start"] + 1e-9 * scale and g <= ph["t_end"] + 1e-9 * scale),
                     None)
            if j is None:
                raise RuntimeError(f"no phase of the derived protocol covers t={g:g}")
        t, v = outputs[j]
        k = int(np.argmin(np.abs(np.asarray(t) - g)))
        if not _close(t[k], g, scale):
            raise RuntimeError(f"phase {j} has no output row at t={g:g}")
        times.append(float(t[k]))
        rows.append(np.asarray(v)[k])
    return np.asarray(times), np.vstack(rows)


def boundary_times(phases: list[dict[str, Any]], grid: np.ndarray, scale: float) -> list[float]:
    """Output times at which the derived protocol changes the state between
    phases, where an SBML event fired. What is reported at that instant, before
    or after the change, is the conversion's convention, not the network's."""
    marks = [ph["t_start"] for ph in phases if ph["changes"]]
    return [float(g) for g in grid if any(_close(g, m, scale) for m in marks)]


def _set_net_values(lines: list[str], rows: list[tuple[int, str]],
                    values: dict[int, float]) -> list[str]:
    """A copy of a ``.net``'s lines with the given species' values replaced."""
    out = list(lines)
    for k, value in values.items():
        i = rows[k][0]
        m = _NET_SPECIES_RE.match(out[i])
        out[i] = f"{m.group(1)}{m.group(2)}{m.group(3)}{value!r}{m.group(5)}"
    return out


def _summarized_command(cmd: list[str], phase: dict[str, Any]) -> list[str]:
    """A phase's recorded command, with a list of sample times summarized."""
    if "times" not in phase:
        return cmd
    n = len(phase["times"])
    return [*cmd[:-n], f"<{n} sample times, {phase['times'][0]:g} to {phase['times'][-1]:g}>"]


def _phase_command(net: Path, phase: dict[str, Any], rtol: float, atol: float,
                   out_prefix: Path) -> list[str]:
    """The run_network call for one phase, as BNG2.pl makes it."""
    cmd = [
        *env.run_network_launch(),
        str(env.run_network_bin()), "-o", str(out_prefix), "-p", "cvode",
        "-a", repr(float(atol)), "-r", repr(float(rtol)),
    ]
    if float(phase["t_start"]) != 0.0:
        cmd += ["-i", repr(float(phase["t_start"]))]
    cmd += ["--cdat", "1", "--fdat", "0", "-g", str(net), str(net)]
    if "dt" in phase:
        cmd += [repr(float(phase["dt"])), str(int(phase["n"]))]
    else:
        cmd += [repr(float(t)) for t in phase["times"]]
    return cmd


class RunNetworkSBML:
    """run_network on the network derived from the SBML (``derive_network``).

    The protocol runs as BNG2.pl runs a sequence of ``simulate`` actions: a fresh
    run_network per phase, each from the ``.net`` with the species set to the last
    row the phase before wrote, and to any value the protocol sets in between (an
    SBML event's assignment). An event-free model on its own output grid is one
    run_network call, exactly as on the ``.net`` path. Columns are the compared
    species, found in the derived network by name.
    """

    NAME = "run_network"
    INPUT = "derived"

    def __init__(self, derived: dict[str, Any], proto: dict[str, Any], out_prefix: Path, *,
                 timeout: float):
        self.net = Path(derived["net"])
        self.phases = derived["phases"]
        self.columns = derived["columns"]
        self.boundary = derived.get("boundary_times", [])
        self.proto = proto
        self.out_prefix = Path(out_prefix)
        self.out_prefix.parent.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.lines = self.net.read_text(encoding="utf-8").splitlines(keepends=True)
        self.rows = net_species(self.lines)
        self.commands: list[list[str]] = []

    def _phase_net(self, j: int, state: np.ndarray | None, changes) -> Path:
        if j == 0 and not changes:
            return self.net
        values: dict[int, float] = {}
        if state is not None:
            values.update({k: float(v) for k, v in enumerate(state)})
        values.update({int(k): float(v) for k, v in changes})
        path = self.out_prefix.parent / f"{self.out_prefix.name}_phase{j}.net"
        path.write_text("".join(_set_net_values(self.lines, self.rows, values)), encoding="utf-8")
        return path

    def run(self):
        p = self.proto
        outputs = []
        state = None
        commands = []
        for j, phase in enumerate(self.phases):
            net = self._phase_net(j, state, phase["changes"])
            prefix = Path(f"{self.out_prefix}_p{j}")
            cmd = _phase_command(net, phase, float(p["rtol"]), float(p["atol"]), prefix)
            t, values, _ = _run_network(cmd, prefix, timeout=self.timeout)
            if values.shape[1] != len(self.rows):
                raise RuntimeError(f"run_network wrote {values.shape[1]} species columns, "
                                   f"the derived network has {len(self.rows)}")
            outputs.append((t, values))
            state = values[-1]
            commands.append(_summarized_command(recorded_command(cmd), phase))
        self.commands = commands
        times, rows = stitch(outputs, self.phases, output_times(p), _time_scale(p))
        return times, rows[:, self.columns]

    warm = run

    @staticmethod
    def trajectory(values) -> np.ndarray:
        return np.asarray(values[1])

    @staticmethod
    def times(values) -> np.ndarray:
        return np.asarray(values[0])

    def config(self) -> dict[str, Any]:
        return {"solver": "dense", "input": "network derived from the SBML",
                "phases": len(self.phases), "commands": self.commands,
                "boundary_times": self.boundary}


def _bngsim_net_phases(net: Path, phases: list[dict[str, Any]], proto: dict[str, Any],
                       rtol: float, atol: float) -> tuple[np.ndarray, np.ndarray]:
    """BNGsim on the derived network, through the same phases run_network runs:
    the faithfulness check of the derivation and of the phase driver at once."""
    import bngsim

    model = bngsim.Model.from_net(str(net))
    sim = bngsim.Simulator(model, method="ode")
    outputs = []
    state = None
    for j, phase in enumerate(phases):
        if state is None:
            model.reset()
            values = np.asarray(model.get_state(), dtype=float)
        else:
            values = np.array(state, dtype=float)
        for k, v in phase["changes"]:
            values[int(k)] = float(v)
        model.set_state(values)
        t0, t1 = float(phase["t_start"]), float(phase["t_end"])
        if "dt" in phase:
            r = sim.run(t_span=(t0, t1), n_points=int(phase["n"]) + 1, rtol=rtol, atol=atol)
        else:
            r = sim.run(sample_times=[t0, *phase["times"]], rtol=rtol, atol=atol)
        out = np.array(r.species)
        outputs.append((np.asarray(r.time), out))
        state = out[-1]
    return stitch(outputs, phases, output_times(proto), _time_scale(proto))


def derive_network(sbml: Path, species_ids: list[str], proto: dict[str, Any], workdir: Path, *,
                   timeout: float, stage=lambda text: None) -> dict[str, Any]:
    """The network run_network reads for an SBML model, or why there is none.

    1. ``bngsim.convert.sbml_to_bngl`` (strict) writes compartmental BNGL; a
       fixed-time SBML event becomes a ``setConcentration`` between ``simulate``
       phases. A model it refuses has no faithful BNGL form.
    2. BNG2.pl generates the ``.net`` from it.
    3. Each compared species is found in the network by name, and each phase is
       sampled at the protocol's output times.
    4. BNGsim on the derived network, run through those phases, must reproduce
       BNGsim on the SBML at a tight tolerance, away from the instants the state
       is changed (``boundary_times``). A converter that dropped an event, say,
       fails here.

    Returns ``{"applicable": True, "net", "columns", "phases", "boundary_times",
    "info"}``, or ``{"applicable": False, "reason", "info"}``. Model preparation:
    charged to no engine.
    """
    import bngsim
    from bngsim.convert import parse_bngl_protocol, sbml_to_bngl

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    info: dict[str, Any] = {}

    def not_applicable(reason: str) -> dict[str, Any]:
        return {"applicable": False, "reason": reason[:400], "info": info}

    t_start, t_end = float(proto["t_start"]), float(proto["t_end"])
    grid = output_times(proto)
    scale = _time_scale(proto)

    stage("sbml_to_bngl")
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = sbml_to_bngl(str(sbml), str(workdir / f"{sbml.stem}.bngl"), strict=True,
                                  t_span=(t_start, t_end), n_points=grid.size)
    except bngsim.ConversionError as exc:
        return not_applicable(f"no faithful BNGL form: bngsim.convert.sbml_to_bngl refuses it "
                              f"({exc})")
    except Exception as exc:  # noqa: BLE001 - the converter's failure, not run_network's
        return not_applicable(f"bngsim.convert.sbml_to_bngl failed: {type(exc).__name__}: {exc}")
    info["conversion_seconds"] = time.perf_counter() - t0
    info["converter_notes"] = sorted({str(w.message)[:300] for w in caught})
    info["bngsim_only"] = [str(x)[:300] for x in (report.bngsim_only or [])]
    text = report.output_text

    # The phases: the converter's simulate steps and the state changes between them.
    phases: list[dict[str, Any]] = []
    pending: list[tuple[str, float]] = []
    steps = parse_bngl_protocol(text).steps if "begin actions" in text else ()
    for step in steps:
        kind = getattr(step, "kind", None)
        if type(step).__name__ == "StateChange":
            if kind != "set_concentration" or step.value is None or step.value_expr:
                return not_applicable(f"the derived protocol has a {kind} step ({step.target}), "
                                      f"which this run_network driver does not run")
            pending.append((step.target, float(step.value)))
        elif kind == "simulate" and getattr(step, "method", "ode") == "ode":
            a, b = (float(x) for x in step.t_span)
            phases.append({"t_start": a, "t_end": b, "changes": pending})
            pending = []
        else:
            return not_applicable(f"the derived protocol has a {kind} step, which this "
                                  f"run_network driver does not run")
    if not phases:
        phases = [{"t_start": t_start, "t_end": t_end, "changes": pending}]
        pending = []
    if pending:
        return not_applicable("the derived protocol changes the state after its last phase")
    ends = [(ph["t_start"], ph["t_end"]) for ph in phases]
    if (not _close(ends[0][0], t_start, scale) or not _close(ends[-1][1], t_end, scale)
            or any(not _close(ends[j][0], ends[j - 1][1], scale) for j in range(1, len(ends)))):
        return not_applicable(f"the derived protocol's phases {ends} do not run "
                              f"[{t_start:g}, {t_end:g}] in order")

    stage("BNG2.pl network generation")
    try:
        net, info["netgen_seconds"] = generate_network(
            text, "generate_network({overwrite=>1})", workdir / "netgen", timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return not_applicable(f"BNG2.pl cannot build the derived model: {exc}")
    info["network"] = network_size(net)

    net_lines = net.read_text(encoding="utf-8").splitlines(keepends=True)
    rows = net_species(net_lines)
    columns, unaligned = align_derived(species_ids, net_lines)
    if unaligned:
        return not_applicable(f"cannot find {len(unaligned)} compared species in the derived "
                              f"network by name (e.g. {unaligned[:3]})")
    keys = {species_key(name): k for k, (_, name) in enumerate(rows)}
    for ph in phases:
        changes = []
        for target, value in ph["changes"]:
            k = keys.get(species_key(target))
            if k is None:
                return not_applicable(f"the derived protocol sets {target}, which is not a "
                                      f"species of the derived network")
            changes.append([k, value])
        ph["change_targets"] = [target for target, _ in ph["changes"]]
        ph["changes"] = changes
        ph.update(sample_phase(ph["t_start"], ph["t_end"], grid, scale))
    boundary = boundary_times(phases, grid, scale)

    stage("faithfulness check (BNGsim on the derived network vs on the SBML)")
    try:
        _, derived = _bngsim_net_phases(net, phases, proto, FAITHFULNESS_RTOL, FAITHFULNESS_ATOL)
        model = bngsim.Model.from_sbml(str(sbml))
        result = _bngsim_sbml_run(bngsim.Simulator(model, method="ode"), proto,
                                  rtol=FAITHFULNESS_RTOL, atol=FAITHFULNESS_ATOL)
        source = np.array(result.species)[_leading_rows(proto):,
                                          _bngsim_columns(result, species_ids)]
    except Exception as exc:  # noqa: BLE001
        return not_applicable(f"the faithfulness check could not run: {type(exc).__name__}: {exc}")
    keep = np.array([not any(_close(g, b, scale) for b in boundary) for g in grid])
    # The vendored verdict alone decides (the timing job derives the network too,
    # and decides the same way); the per-species check is recorded beside it.
    verdict = _vendored_verdict(derived[:, columns][keep], source[keep], FAITHFULNESS_ATOL)
    check = species_check(derived[:, columns][keep], source[keep], FAITHFULNESS_ATOL,
                          species=species_ids, times=grid[keep])
    info["faithfulness"] = {
        "passed": bool(verdict.get("passed")),
        "max_err_over_peak": verdict.get("max_err_over_peak"),
        "species_check": _check_summary(check),
        "rtol": FAITHFULNESS_RTOL, "atol": FAITHFULNESS_ATOL,
        "rows_not_compared": boundary,
    }
    if not verdict.get("passed"):
        return not_applicable(
            f"the derived network is not faithful: BNGsim on it misses BNGsim on the SBML "
            f"by {verdict.get('max_err_over_peak'):.3g} of a species' peak")
    info["phases"] = len(phases)
    return {"applicable": True, "net": str(net), "columns": columns, "phases": phases,
            "boundary_times": boundary, "info": info}


SBML_REFERENCE_ENGINES: dict[str, type] = {
    RunNetworkSBML.NAME: RunNetworkSBML,
    RoadRunnerSBML.NAME: RoadRunnerSBML,
    CopasiSBML.NAME: CopasiSBML,
    AmiciSBML.NAME: AmiciSBML,
}


def _build_reference(name: str, prep: dict[str, Any], *, sparse: bool,
                     out_prefix: Path, timeout: float):
    """Construct one reference engine from the prepared artifacts."""
    if prep.get("format") == "sbml":
        cls = SBML_REFERENCE_ENGINES[name]
        if cls.INPUT == "derived":
            return cls(prep["derived"], prep["proto"], out_prefix, timeout=timeout)
        return cls(prep["sbml"], prep["species_ids"], prep["proto"])
    cls = REFERENCE_ENGINES[name]
    if cls.INPUT == "net":
        return cls(prep["net"], prep["proto"], out_prefix, sparse=sparse, timeout=timeout)
    return cls(prep["sbml"], prep["species_names"], prep["name_to_id"], prep["proto"])


# --------------------------------------------------------------------------- #
# Per-engine isolation (correctness job)
# --------------------------------------------------------------------------- #
# What each engine is doing while it builds, for a timeout's error message.
BUILD_STAGE = {
    "bngsim": "load + codegen",
    "run_network": "solve (subprocess)",
    "roadrunner": "SBML load (LLVM JIT)",
    "copasi": "SBML load",
    "amici": "SBML import + C++ compile",
}


def _progress(msg: str) -> None:
    progress(CAPABILITY, msg)


def _build_bngsim(prep: dict[str, Any]):
    """The system under test, from the ``.net`` or, for an SBML model, the SBML."""
    if prep.get("format") == "sbml":
        return BngsimSBML(prep["sbml"], prep["species_ids"], prep["proto"])
    return BngsimODE(prep["net"], prep["proto"])


def _checked_trajectory(engine, result, prep: dict[str, Any]) -> np.ndarray:
    """An engine's trajectory; for an SBML model, only once its output is at the
    protocol's times (a ``.net`` engine's grid is the one it was asked for)."""
    if prep.get("format") == "sbml":
        check_times(engine.times(result), prep["proto"])
    return engine.trajectory(result)


def _cold_bngsim(prep: dict[str, Any]):
    def fn(stage):
        t0 = time.perf_counter()
        stage(BUILD_STAGE["bngsim"])
        engine = _build_bngsim(prep)
        stage("first solve")
        result = engine.run()
        return (_checked_trajectory(engine, result, prep),
                {**engine.config(result), "cold_seconds": time.perf_counter() - t0})
    return fn


def _cold_reference(name: str, prep: dict[str, Any], out_prefix: Path, timeout: float):
    def fn(stage):
        t0 = time.perf_counter()
        stage(BUILD_STAGE.get(name, "build"))
        ref = _build_reference(name, prep, sparse=False, out_prefix=out_prefix, timeout=timeout)
        stage("first solve")
        values = ref.run()
        return (_checked_trajectory(ref, values, prep),
                {**ref.config(), "cold_seconds": time.perf_counter() - t0})
    return fn


def _derive_isolated(prep: dict[str, Any], workdir: Path, budget: float, timeout: float):
    """``derive_network`` in a forked child under ``budget``: what it returns, or
    run_network not applicable, with the reason, when the derivation itself
    failed or ran out of time (it is preparation, never run_network's failure)."""
    def fn(stage):
        derived = derive_network(prep["sbml"], prep["species_ids"], prep["proto"],
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


# --------------------------------------------------------------------------- #
# Verdict and reproducibility
# --------------------------------------------------------------------------- #
def _prepare(problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    if problem.source.get("format") == "sbml":
        return _prepare_sbml(problem, ctx)
    proto = protocol(problem)
    return {**prepare_network(problem, ctx, proto["netgen"]), "proto": proto}


def _prepare_sbml(problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    """An SBML model needs no network: the pinned file, copied to the job's own
    scratch (AMICI compiles beside it), and its state species."""
    proto = protocol(problem)
    pinned = fetch.model_file(problem.source, ctx["cache_dir"])
    workdir = Path(ctx["workdir"])
    (workdir / "sbml").mkdir(parents=True, exist_ok=True)
    sbml = workdir / "sbml" / pinned.name
    shutil.copyfile(pinned, sbml)
    species_ids, census = sbml_species(sbml)
    size = {"species": census["species"], "reactions": census["reactions"]}
    info: dict[str, Any] = {"network": size, "sbml": census}
    if problem.size.get("reactions") not in (None, size["reactions"]):
        info["size_mismatch"] = {"spec": problem.size, "sbml": size}
    return {"format": "sbml", "sbml": sbml, "species_ids": species_ids, "proto": proto,
            "info": info}


def _vendored_verdict(bn: np.ndarray, ref: np.ndarray, atol: float) -> dict[str, Any]:
    """bngsim's own deterministic verdict (``parity_core.differ``), unmodified,
    on the rows and columns the two trajectories share."""
    from .._vendor.parity_core import differ

    n_t = min(bn.shape[0], ref.shape[0])
    n_sp = min(bn.shape[1], ref.shape[1])
    if n_t == 0 or n_sp == 0:
        return {"passed": False, "reason": "nothing to compare"}
    v = differ.deterministic_verdict(bn[:n_t, :n_sp], ref[:n_t, :n_sp], atol=atol)
    keep = (
        "passed", "max_rel", "max_abs", "n_cells", "n_fail", "n_soft_fail",
        "n_hard_fail", "budget_forgiven", "no_signal", "max_err_over_movement",
    )
    out = {k: v[k] for k in keep if k in v}
    out["max_err_over_peak"] = max_err_over_peak(bn[:n_t, :n_sp], ref[:n_t, :n_sp])
    out["shapes"] = {"bngsim": list(bn.shape), "reference": list(ref.shape)}
    return out


def _verdict(bn: np.ndarray, ref: np.ndarray, atol: float, *,
             species: list[str] | None = None, times: np.ndarray | None = None
             ) -> dict[str, Any]:
    """BNGsim against one reference: the vendored verdict AND the per-species
    check, both recorded, ``passed`` only when both pass.

    The vendored verdict's own result keeps its flat keys (``max_rel``,
    ``n_hard_fail``, ...) with its pass as ``vendored_passed``; the per-species
    check is ``species_check``."""
    out = _vendored_verdict(bn, ref, atol)
    if "reason" in out:
        return out
    n_t = min(bn.shape[0], ref.shape[0])
    n_sp = min(bn.shape[1], ref.shape[1])
    check = species_check(bn[:n_t, :n_sp], ref[:n_t, :n_sp], atol,
                          species=None if species is None else list(species)[:n_sp],
                          times=None if times is None else np.asarray(times)[:n_t])
    out["vendored_passed"] = bool(out["passed"])
    out["species_check"] = check
    out["passed"] = bool(out["vendored_passed"] and check["passed"])
    return out


# --------------------------------------------------------------------------- #
# The per-species check, beside the vendored verdict (constants at the top)
# --------------------------------------------------------------------------- #
def _finite_abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|a - b| cell by cell, where two equal values (two NaNs, or the same
    infinity) differ by 0 and a finite value against a non-finite one, or two
    different non-finite values, by infinity."""
    with np.errstate(invalid="ignore", over="ignore"):
        d = np.abs(a - b)
    fa, fb = np.isfinite(a), np.isfinite(b)
    same = (np.isnan(a) & np.isnan(b)) | (~fa & ~fb & (a == b))
    d = np.where(fa & fb, d, np.inf)
    return np.where(same, 0.0, d)


def species_check(a: np.ndarray, b: np.ndarray, atol: float, *,
                  species: list[str] | None = None, times: np.ndarray | None = None,
                  tol: float | None = None, atol_factor: float | None = None
                  ) -> dict[str, Any]:
    """Each species' worst disagreement, judged against that species' own scale.

    ``a`` (BNGsim) and ``b`` (the reference) are aligned ``(n_time, n_species)``
    arrays. A species fails when ``max_t |a - b|`` is past ``allowed = max(tol *
    peak, floor)``, where ``peak`` is its largest magnitude over time on either
    side (so an engine that moves a species the other holds at zero is judged
    against what it moved it to) and ``floor = atol_factor * atol``. A
    non-finite value against a finite one always fails. Reports the worst species
    by ``err / allowed`` (its name, error, peak, the floor, and the time of its
    worst cell), how many species fail and which, and how many sit at the floor
    (their peak is so small that the floor, not their peak, sets what they may
    be off by)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    tol = SPECIES_TOL if tol is None else float(tol)
    factor = SPECIES_ATOL_FACTOR if atol_factor is None else float(atol_factor)
    floor = factor * float(atol)
    out: dict[str, Any] = {"tol": tol, "atol": float(atol), "atol_factor": factor,
                           "floor": floor, "n_species": int(a.shape[1])}
    if a.size == 0:
        return {**out, "passed": False, "reason": "nothing to compare"}
    diff = _finite_abs_diff(a, b)
    err = diff.max(axis=0)
    mag = np.maximum(np.where(np.isfinite(a), np.abs(a), 0.0),
                     np.where(np.isfinite(b), np.abs(b), 0.0))
    peak = mag.max(axis=0)
    allowed = np.maximum(tol * peak, floor)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(err > 0, err / np.where(allowed > 0, allowed, np.inf), 0.0)
    score = np.where(np.isnan(score), np.inf, score)
    fails = score > 1.0
    j = int(np.argmax(score))
    names = list(species) if species is not None else [f"#{k}" for k in range(a.shape[1])]

    def name(k: int) -> str:
        return str(names[k]) if k < len(names) else f"#{k}"

    worst: dict[str, Any] = {
        "species": name(j), "index": j, "err": float(err[j]), "peak": float(peak[j]),
        "err_over_peak": float(err[j] / peak[j]) if peak[j] > 0 else None,
        "allowed": float(allowed[j]), "score": float(score[j]),
    }
    if times is not None and len(times) == a.shape[0]:
        worst["t"] = float(np.asarray(times)[int(np.argmax(diff[:, j]))])
    failing = [name(int(k)) for k in np.argsort(-score) if fails[k]]
    return {**out, "passed": not bool(fails.any()), "n_fail": int(fails.sum()),
            "failing": failing[:10], "n_at_floor": int(np.sum(tol * peak < floor)),
            "worst": worst}


def sparse_check(net: Path, proto: dict[str, Any], workdir: Path, dense: np.ndarray,
                 timeout: float, *, keep: dict[str, np.ndarray] | None = None
                 ) -> dict[str, Any] | None:
    """Whether run_network with the authors' ``sparse`` option agrees with the
    dense reference; None when the protocol does not ask for sparse. ``keep``,
    when given, receives the sparse trajectory as ``run_network_sparse``."""
    if not proto.get("sparse"):
        return None
    try:
        engine = RunNetworkODE(net, proto, workdir / "rn_sparse", sparse=True, timeout=timeout)
        t0 = time.perf_counter()
        values = engine.run()
        seconds = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        return {"agrees_with_dense": False, "error": f"{type(exc).__name__}: {exc}"[:400]}
    if keep is not None:
        keep["run_network_sparse"] = np.asarray(values, dtype=np.float64)
    # The vendored verdict alone decides whether the timing job may time the
    # authors' sparse setting; the correctness job records the per-species check
    # of the same two runs beside it (``_score``).
    v = _vendored_verdict(values, dense, float(proto["atol"]))
    return {"agrees_with_dense": bool(v.get("passed")), "max_rel": v.get("max_rel"),
            "n_hard_fail": v.get("n_hard_fail"), "seconds": seconds}


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
def run_correctness(job: Job, problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    try:
        prep = _prepare(problem, ctx)
    except Exception as exc:  # noqa: BLE001
        return {"outcome": "BAD_TEST", "error": f"{type(exc).__name__}: {exc}"}
    proto, info = prep["proto"], prep["info"]
    atol = float(proto["atol"])
    workdir = Path(ctx["workdir"])
    budget = float(ctx.get("engine_budget_seconds") or engine_budget_seconds(problem))
    info["engine_budget_seconds"] = budget
    # A subprocess an engine starts (run_network) never outlives the engine's budget.
    timeout = min(float(ctx["subprocess_timeout"]), budget)

    def isolated(name: str, fn) -> dict[str, Any]:
        _progress(f"{name}: start (budget {budget:g} s; {_threads()} threads in the worker)")
        out = run_isolated(fn, budget, workdir / "isolated" / name)
        _progress(f"{name}: {out.get('error') or 'done'} after {out['seconds']:.1f} s")
        return out

    not_applicable = _not_applicable(prep, workdir, budget, timeout)

    bn = None
    bn_error = None
    out = isolated("bngsim", _cold_bngsim(prep))
    if "error" in out:
        bn_error = out["error"]
    else:
        info["bngsim"] = out["config"]
        bn = out["trajectory"]

    references: dict[str, Any] = {}
    ref_trajectories: dict[str, np.ndarray] = {}
    ref_errors: dict[str, str] = {}
    for name in REFERENCE_ENGINES:
        if name in not_applicable:
            continue
        out = isolated(name, _cold_reference(name, prep, workdir / name, timeout))
        if "error" in out:
            ref_errors[name] = out["error"]
        else:
            ref_trajectories[name] = out["trajectory"]
            references[name] = out["config"]
    info["references"] = references

    # run_network's authors' sparse setting, judged against its own dense run.
    sparse_traj: dict[str, np.ndarray] = {}
    if "run_network" in ref_trajectories and prep.get("format") != "sbml":
        check = sparse_check(prep["net"], proto, workdir, ref_trajectories["run_network"], timeout,
                             keep=sparse_traj)
        if check is not None:
            info["sparse_reference"] = check

    species = prep.get("species_ids") if prep.get("format") == "sbml" else prep.get("species_names")
    verdicts, outcome = _score(bn, ref_trajectories, references, proto, species=species,
                               bn_error=bn_error, ref_errors=ref_errors,
                               not_applicable=not_applicable)
    if "run_network_sparse" in sparse_traj and "sparse_reference" in info:
        info["sparse_reference"]["species_check"] = _sparse_species_check(
            sparse_traj["run_network_sparse"], ref_trajectories["run_network"], proto, species)
    rec: dict[str, Any] = {"outcome": outcome, "measurements": info}
    if verdicts:
        rec["verdicts"] = verdicts
    if bn_error:
        rec["bngsim_error"] = bn_error
    for name, err in ref_errors.items():
        rec[f"{name}_error"] = err
    if not_applicable:
        rec["not_applicable"] = not_applicable
    stored = stored_trajectories(prep, bn, ref_trajectories, sparse_traj)
    if stored:
        rec["_trajectories"] = stored
    return rec


def stored_trajectories(prep: dict[str, Any], bn: np.ndarray | None,
                        refs: dict[str, np.ndarray],
                        sparse: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """What a correctness job stores beside its record (the worker writes it as
    ``<job id>.npz``): every engine's compared trajectory as it was scored, each
    under its engine's name and in the same columns, with the output times
    (``time``) and the compared species (``species``: BNGsim's species names on
    the ``.net`` path, SBML ids on the SBML path). run_network's sparse run, when
    the model asks for one, is ``run_network_sparse``. Enough to re-score every
    verdict (``bench rescore``) without re-simulating; empty when no engine ran."""
    engines = {**({"bngsim": bn} if bn is not None else {}), **refs, **sparse}
    if not engines:
        return {}
    species = prep.get("species_ids") if prep.get("format") == "sbml" else prep.get("species_names")
    out: dict[str, np.ndarray] = {
        "time": output_times(prep["proto"]),
        "species": np.asarray([str(s) for s in species or []], dtype=str),
    }
    for name, traj in engines.items():
        out[name] = np.asarray(traj, dtype=np.float64)
    return out


def _not_applicable(prep: dict[str, Any], workdir: Path, budget: float,
                    timeout: float) -> dict[str, str]:
    """References that do not apply to this model, each with its reason; for an
    SBML model, run_network when no faithful network can be derived. Records the
    derivation in the measurements either way, and on success leaves the derived
    network in ``prep`` for run_network to read."""
    if prep.get("format") != "sbml":
        return {}
    derived = _derive_isolated(prep, workdir, budget, timeout)
    prep["info"]["derived_network"] = derived.get("info", {})
    if derived.get("applicable"):
        prep["derived"] = derived
        return {}
    return {"run_network": str(derived.get("reason"))}


def _reference_verdict(bn: np.ndarray, ref: np.ndarray, atol: float, config: dict[str, Any],
                       proto: dict[str, Any], species: list[str] | None = None) -> dict[str, Any]:
    """BNGsim against one reference. Rows at the instants a derived network's
    protocol changes the state (``boundary_times``: an SBML event, translated to
    an action between run_network phases) are left out, and named: what is
    reported at that instant is the conversion's convention, not run_network's."""
    grid = output_times(proto)
    boundary = config.get("boundary_times") or []
    if not boundary:
        return _verdict(bn, ref, atol, species=species, times=grid)
    scale = _time_scale(proto)
    keep = np.array([not any(_close(g, b, scale) for b in boundary) for g in grid])
    n = min(bn.shape[0], ref.shape[0], keep.size)
    v = _verdict(bn[:n][keep[:n]], ref[:n][keep[:n]], atol, species=species,
                 times=grid[:n][keep[:n]])
    v["rows_not_compared"] = list(boundary)
    return v


def _score(bn: np.ndarray | None, refs: dict[str, np.ndarray], configs: dict[str, Any],
           proto: dict[str, Any], *, species: list[str] | None, bn_error: str | None,
           ref_errors: dict[str, str], not_applicable: dict[str, str]
           ) -> tuple[dict[str, Any], str]:
    """The verdict of BNGsim against each reference that ran, and the job's
    outcome. The one place both are computed, from a run (``run_correctness``)
    or from its stored trajectories (``rescore``), so the two cannot drift."""
    atol = float(proto["atol"])
    verdicts: dict[str, Any] = {}
    if bn is not None:
        for name, traj in refs.items():
            verdicts[name] = _reference_verdict(bn, traj, atol, configs.get(name) or {}, proto,
                                                species)
    outcome = _outcome(bn_error, ref_errors, len(REFERENCE_ENGINES) - len(not_applicable),
                       all_passed(verdicts))
    if bn_error and bn_error.startswith("TIMEOUT") and outcome == "EXCEPTION":
        outcome = "TIMEOUT"  # BNGsim, not a reference, ran out of time
    return verdicts, outcome


def _check_summary(check: dict[str, Any]) -> dict[str, Any]:
    """A per-species check as an informational field beside a decision it does
    not take part in (the faithfulness and sparse checks)."""
    return {k: check.get(k) for k in ("passed", "n_fail", "floor", "tol", "worst")}


def _sparse_species_check(sparse: np.ndarray, dense: np.ndarray, proto: dict[str, Any],
                          species: list[str] | None) -> dict[str, Any]:
    n_t = min(sparse.shape[0], dense.shape[0])
    n_sp = min(sparse.shape[1], dense.shape[1])
    return _check_summary(species_check(
        sparse[:n_t, :n_sp], dense[:n_t, :n_sp], float(proto["atol"]),
        species=None if species is None else list(species)[:n_sp],
        times=output_times(proto)[:n_t]))


def rescore(job: Job, record: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """A correctness record re-scored from its stored trajectories with the
    current verdict: the record's new ``verdicts`` and ``outcome`` (and, where it
    stored a sparse run, its sparse check's per-species result). Engine failures,
    configurations and not-applicable references are the record's own."""
    proto = job.definition["protocol"]
    species = [str(s) for s in arrays["species"]] if "species" in arrays else None
    refs = {name: arrays[name] for name in REFERENCE_ENGINES if name in arrays}
    ref_errors = {name: str(record[f"{name}_error"]) for name in REFERENCE_ENGINES
                  if record.get(f"{name}_error")}
    m = record.get("measurements") or {}
    verdicts, outcome = _score(arrays.get("bngsim"), refs, m.get("references") or {}, proto,
                               species=species, bn_error=record.get("bngsim_error"),
                               ref_errors=ref_errors,
                               not_applicable=record.get("not_applicable") or {})
    update: dict[str, Any] = {"outcome": outcome}
    if verdicts:
        update["verdicts"] = verdicts
    sparse = m.get("sparse_reference")
    if sparse is not None and "run_network_sparse" in arrays and "run_network" in arrays:
        sparse = dict(sparse)
        sparse["species_check"] = _sparse_species_check(
            arrays["run_network_sparse"], arrays["run_network"], proto, species)
        update["measurements"] = {**m, "sparse_reference": sparse}
    return update


def _correctness_timeouts(job: Job, ctx: dict[str, Any]) -> dict[str, str]:
    """References recorded as timed out by this timing job's correctness job."""
    return correctness_timeouts(job, ctx, REFERENCE_ENGINES)


def run_timing(job: Job, problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    try:
        prep = _prepare(problem, ctx)
    except Exception as exc:  # noqa: BLE001
        # A model that cannot be read or whose network cannot be built is a bad
        # test input, not an engine failure. Report it as a labelled failed row and
        # not the worker's generic EXCEPTION (and not a missing row in gather).
        return {"outcome": "BAD_TEST", "error": f"{type(exc).__name__}: {exc}"}
    proto, info = prep["proto"], prep["info"]
    workdir = Path(ctx["workdir"])
    timeout = ctx["subprocess_timeout"]
    # Derived before any engine loads, so the derivation's fork is from a process
    # that has started no engine threads.
    not_applicable = (_not_applicable(prep, workdir, engine_budget_seconds(problem),
                                      float(timeout))
                      if prep.get("format") == "sbml" else {})

    # BNGsim cold: load and first integration, with the empty compile cache the
    # worker gave this job. Its trajectory anchors the warm runs' reproducibility
    # (a reset bug shows on the first warm run, which warm-vs-warm would miss).
    t0 = time.perf_counter()
    bngsim_engine = _build_bngsim(prep)
    first = bngsim_engine.run()
    info["bngsim_cold_seconds"] = time.perf_counter() - t0
    info["bngsim"] = bngsim_engine.config(first)
    bngsim_cold = _checked_trajectory(bngsim_engine, first, prep)

    # Build each reference and give it one trial run, isolating any that fail. A
    # reference can crash inside run() rather than __init__ (COPASI drops a constant
    # source species and then raises on the missing column; AMICI can fail at
    # integration), and the interleave's warm-up round is not fault-isolated -- so a
    # trial run here is what keeps one crasher from taking down the whole timing job.
    # A failed reference is recorded (as <name>_error) and dropped from the timed
    # set, exactly as the correctness job records REFERENCE_FAILED; BNGsim and the
    # surviving references are still timed. The trial result is discarded (the real
    # timing is the interleave below), at the cost of one extra cold run per engine.
    # BNGsim is never dropped -- it ran cold above, and a BNGsim failure there is an
    # EXCEPTION the worker records, not an engine we quietly skip.
    #
    # run_network is timed as the authors ran it only if that run is right: time
    # sparse only where it agrees with the dense run. Its dense run and sparse check
    # are themselves the trial, so no extra warm() is needed for it.
    references: dict[str, Any] = {}
    engines: dict[str, Any] = {"bngsim": bngsim_engine}
    ref_errors: dict[str, str] = {}
    # A reference that ran out of its budget in the correctness job would only do
    # so again here, in-process and unbounded: record it and time the rest.
    timed_out = _correctness_timeouts(job, ctx)
    for name in REFERENCE_ENGINES:
        if name in not_applicable:
            continue
        if name in timed_out:
            ref_errors[name] = f"TIMEOUT in the correctness job, so not timed ({timed_out[name]})"
            _progress(f"{name}: skipped, {ref_errors[name]}")
            continue
        _progress(f"{name}: build + trial run")
        try:
            if name == "run_network" and prep.get("format") != "sbml":
                dense_engine = RunNetworkODE(prep["net"], proto, workdir / "rn_dense",
                                             sparse=False, timeout=timeout)
                dense = RunNetworkODE.trajectory(dense_engine.run())
                check = sparse_check(prep["net"], proto, workdir, dense, timeout)
                use_sparse = bool(check and check["agrees_with_dense"])
                if check is not None:
                    info["sparse_reference"] = check
                engine = RunNetworkODE(prep["net"], proto, workdir / "rn", sparse=use_sparse,
                                       timeout=timeout)
            else:
                engine = _build_reference(name, prep, sparse=False, out_prefix=workdir / name,
                                          timeout=timeout)
                # Trial run: COPASI/AMICI surface failures here, not in __init__ --
                # and the trajectory too, since an SBML engine selects its columns
                # there, outside the timed region.
                _checked_trajectory(engine, engine.warm(), prep)
            engines[name] = engine
            references[name] = engine.config()
        except Exception as exc:  # noqa: BLE001
            ref_errors[name] = f"{type(exc).__name__}: {exc}"[:400]
    info["references"] = references

    # Interleave every engine round-robin in one drift-cancelling run. The timed
    # callable returns each engine's raw result; the trajectory is extracted
    # outside the timed region (the capture), so it does not bias the measurement.
    _progress(f"interleaved timing of {', '.join(engines)}")
    runs = {name: eng.warm for name, eng in engines.items()}
    captures = {name: eng.trajectory for name, eng in engines.items()}
    reps = round_robin_interleaved(runs, captures=captures, **job.definition["timing"])

    # Every kept repetition must reproduce the first run (issue #8). BNGsim's warm
    # re-runs are measured against its cold first run; each reference's are measured
    # against its own first kept repetition (a legitimately-sparse run_network run
    # differs from dense but is still reproducible).
    outputs = {name: reps["engines"][name].pop("outputs") for name in runs}
    repro: dict[str, Any] = {"bngsim": _reproducibility(bngsim_cold, outputs["bngsim"], REPRO_TOL)}
    for name in engines:  # the surviving references; a dropped one is not in `engines`
        if name == "bngsim":
            continue
        outs = outputs[name]
        repro[name] = _reproducibility(outs[0], outs[1:], REPRO_TOL) if outs else \
            {"passed": True, "n_reps": 0, "max_err_over_peak": 0.0, "worst_rep": None,
             "per_rep": [], "tol": REPRO_TOL}
    reproducible = all(r["passed"] for r in repro.values())

    seconds_by_engine = {name: reps["engines"][name]["seconds"] for name in runs}
    times = absolute_time_stats(seconds_by_engine)
    rec: dict[str, Any] = {
        # A dropped reference does not fail the row: it is recorded, not scored
        # (parity_core.taxonomy). The outcome rests only on whether BNGsim and the
        # surviving references each reproduced their first run.
        "outcome": "PASS" if reproducible else "NOT_REPRODUCIBLE",
        "measurements": {
            **info,
            "engine_roles": {name: _engine_role(name) for name in engines},
            "repetitions": reps,
            "reproducibility": repro,
            "times": times,
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
