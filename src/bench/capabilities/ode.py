"""ODE simulation of BNGL models: BNGsim against a registry of reference engines.

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

The correctness job records a verdict of BNGsim against *each* reference
(bngsim's own deterministic verdict, ``parity_core.differ``) and passes only when
BNGsim agrees with every one -- a genuine cross-engine check. The timing job
interleaves all the engines round-robin in one drift-cancelling run (``bench.timing``)
and reports each engine's absolute time with a bootstrap CI; a speedup is only a
derived view (``gather``), never a stored headline.

A reference engine that crashes is a result to attribute, not a bug to patch: the
correctness job records it as ``REFERENCE_FAILED`` with a ``<name>_error`` and
still scores BNGsim against the references that ran, and the timing job gives each
reference a trial run, records any crash the same way, and drops it from the timed
set -- BNGsim and the surviving references are still timed.

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

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .. import env, fetch
from ..jobs import CORRECTNESS, TIMING, Job
from ..specs import Problem
from ..timing import absolute_time_stats, round_robin_interleaved

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

# Actions BNG2.pl would execute; none may run before the appended netgen.
_ACTION_LINE_RE = re.compile(
    r"^(generate_network|simulate\w*|parameter_scan|bifurcate|readFile|setOption|"
    r"writeSBML|writeMfile|writeModel|writeXML|writeFile|writeNET|writeNetwork|"
    r"writeMexfile|visualize|setConcentration|addConcentration|saveConcentrations|"
    r"saveParameters|resetConcentrations|resetParameters|setParameter|quit)\b"
)
# A setOption that changes the generated network (the count-per-concentration
# unit) must survive, or BNG2.pl builds a different network.
_KEEP_SETOPTION_RE = re.compile(r'^\s*setOption\(\s*"NumberPerQuantityUnit"')
_ACTIONS_BLOCK_RE = re.compile(r"^\s*begin\s+actions\b.*?^\s*end\s+actions\b", re.DOTALL | re.MULTILINE | re.IGNORECASE)
_END_MODEL_RE = re.compile(r"^\s*end\s+model\b.*$", re.MULTILINE | re.IGNORECASE)


def protocol(problem: Problem) -> dict[str, Any]:
    """The ODE protocol with defaults filled in, as it enters a job definition."""
    p = dict(problem.protocol_for(CAPABILITY))
    p.setdefault("t_start", 0)
    p.setdefault("sparse", False)
    return p


def engines_id() -> str:
    """The ``engines`` id segment: BNGsim versus every registered reference.

    Derived from the populated registry, so adding a reference engine re-ids the
    jobs and re-runs them, which is the intended behaviour.
    """
    return "bngsim-vs-" + "-".join(REFERENCE_ENGINES)


def jobs(problem: Problem) -> list[Job]:
    if problem.source.get("format") != "bngl":
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
    est_c, est_t = planning_seconds(problem)
    # The per-engine budgets bound the engines; the job timeout is the backstop for
    # what they do not cover (netgen, export, the sparse check), so it leaves room
    # for every engine to use its whole budget.
    backstop = (len(REFERENCE_ENGINES) + 2) * engine_budget_seconds(problem)
    correctness = Job(
        CAPABILITY, problem.id, engines, "auto", CORRECTNESS,
        definition=base,
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
# Network generation
# --------------------------------------------------------------------------- #
def netgen_bngl(text: str, netgen_call: str) -> str:
    """The model with every action removed and ``netgen_call`` appended."""
    text = _ACTIONS_BLOCK_RE.sub("", text)
    ends = list(_END_MODEL_RE.finditer(text))
    if ends:
        text = text[: ends[-1].end()]
    # Join backslash continuations so a multi-line action is removed whole.
    text = re.sub(r"\\\s*\n", " ", text)
    kept = [
        ln for ln in text.splitlines()
        if not _ACTION_LINE_RE.match(ln.strip()) or _KEEP_SETOPTION_RE.match(ln)
    ]
    return "\n".join(kept).rstrip() + "\n" + netgen_call.strip() + "\n"


def generate_network(
    bngl_text: str, netgen_call: str, workdir: Path, *, timeout: float
) -> tuple[Path, float]:
    workdir.mkdir(parents=True, exist_ok=True)
    bngl = workdir / "model.bngl"
    net = workdir / "model.net"
    net.unlink(missing_ok=True)
    bngl.write_text(netgen_bngl(bngl_text, netgen_call), encoding="utf-8")
    t0 = time.perf_counter()
    proc = subprocess.run(
        ["perl", str(env.bng2_pl()), str(bngl)],
        capture_output=True, text=True, timeout=timeout, cwd=workdir, check=False,
    )
    seconds = time.perf_counter() - t0
    if proc.returncode != 0 or not net.exists():
        tail = (proc.stderr.strip() or proc.stdout.strip()).splitlines()
        raise RuntimeError(f"BNG2.pl network generation failed: {(tail[-1] if tail else '')[:300]}")
    return net, seconds


def network_size(net: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    block = None
    for line in net.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("begin "):
            block = s.split()[1]
            counts.setdefault(block, 0)
        elif s.startswith("end "):
            block = None
        elif block and s and not s.startswith("#"):
            counts[block] += 1
    return {"species": counts.get("species", 0), "reactions": counts.get("reactions", 0)}


def export_sbml(net: Path, out: Path) -> tuple[Path, list[str], dict[str, str]]:
    """Write ``net`` to SBML and return the path, BNGsim's species order, and a
    species-name -> SBML-id map for the SBML-reading engines to align on.

    ``model.species_names`` is BNGsim's canonical column order (what ``.species``
    returns); the SBML carries each species' BNGsim name in its ``name`` attribute
    and a sanitised ``id``. The map lets an SBML engine request its columns in
    BNGsim's order by name, never by position.
    """
    import bngsim
    import libsbml
    from bngsim.convert import write_sbml

    model = bngsim.Model.from_net(str(net))
    species_names = list(model.species_names)
    sbml_text = write_sbml(model)
    out.write_text(sbml_text, encoding="utf-8")
    doc = libsbml.readSBMLFromString(sbml_text)
    sbml_model = doc.getModel()
    if sbml_model is None:
        raise RuntimeError("write_sbml produced a document libsbml cannot read")
    name_to_id = {
        sbml_model.getSpecies(i).getName(): sbml_model.getSpecies(i).getId()
        for i in range(sbml_model.getNumSpecies())
    }
    return out, species_names, name_to_id


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


def recorded_command(cmd: list[str]) -> list[str]:
    """The run_network flags to record: from the binary on, minus any loader
    prefix prepended by ``env.run_network_launch`` and the binary path itself."""
    rn = str(env.run_network_bin())
    i = cmd.index(rn) if rn in cmd else 0
    return cmd[i + 1:]


def read_dat(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """A BNG ``.cdat``/``.gdat``: a ``#`` header of column names, time first."""
    names: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                if s.startswith("#"):
                    names = s.lstrip("#").split()[1:]
                break
    data = np.loadtxt(path, comments="#", ndmin=2, encoding="utf-8")
    if not names:
        names = [f"S{i + 1}" for i in range(data.shape[1] - 1)]
    return data[:, 0].copy(), data[:, 1:], names


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

    def run(self):
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


def _build_reference(name: str, prep: dict[str, Any], *, sparse: bool,
                     out_prefix: Path, timeout: float):
    """Construct one reference engine from the prepared artifacts."""
    cls = REFERENCE_ENGINES[name]
    if cls.INPUT == "net":
        return cls(prep["net"], prep["proto"], out_prefix, sparse=sparse, timeout=timeout)
    return cls(prep["sbml"], prep["species_names"], prep["name_to_id"], prep["proto"])


def _engine_role(name: str) -> str:
    """How an engine is timed, for the human-readable timing record. Derived from
    the engine name so it never drifts from the registry: run_network is a fresh
    subprocess per run; every other reference (and BNGsim) is an in-process library
    timed warm."""
    if name == "bngsim":
        return "warm, in-process (system under test)"
    if name == "run_network":
        return "fresh subprocess"
    return "warm, in-process"


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
    """One flushed line on stderr: the runner keeps the tail of a job's output in
    a TIMEOUT record, so this says where a stopped job was."""
    print(f"[ode {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _threads() -> int | str:
    """OS threads in this process (fork is only safe from a single-threaded one)."""
    try:
        return len(os.listdir("/proc/self/task"))
    except OSError:
        return "?"


def _descendants(pid: int) -> list[int]:
    """Every live descendant of ``pid``, from ``/proc``; empty where there is none."""
    children: dict[int, list[int]] = {}
    try:
        entries = [d for d in os.listdir("/proc") if d.isdigit()]
    except OSError:
        return []
    for d in entries:
        try:
            with open(f"/proc/{d}/stat", encoding="utf-8", errors="replace") as fh:
                ppid = int(fh.read().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(d))
    out: list[int] = []
    stack = [pid]
    while stack:
        for child in children.get(stack.pop(), []):
            out.append(child)
            stack.append(child)
    return out


def _kill_tree(pid: int) -> None:
    """SIGKILL ``pid`` and everything under it (an AMICI build's compilers too).
    Stop the tree first, so nothing forks between collecting and killing it."""
    tree = [pid, *_descendants(pid)]
    for sig in (signal.SIGSTOP, signal.SIGKILL):
        if sig == signal.SIGKILL:
            tree = list(dict.fromkeys([*tree, *_descendants(pid)]))
        for p in tree:
            try:
                os.kill(p, sig)
            except OSError:
                pass


def run_isolated(fn, budget: float, workdir: Path, *, poll: float = 0.2) -> dict[str, Any]:
    """Run ``fn(stage)`` in a forked child under a wall budget.

    ``fn`` returns ``(trajectory, config)`` and calls ``stage(text)`` as it moves
    through its work. Returns ``{"trajectory", "config"}`` on success, or
    ``{"error"}`` -- ``"TIMEOUT: ..."`` naming the stage when the budget ran out
    (the child's whole process tree is killed), the child's exception when it
    raised, or the signal when it died outright. The child shares nothing back but
    files in ``workdir``, so a crash or a leak in it cannot touch the caller.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    stage_file, traj_file, rec_file = (workdir / "stage", workdir / "trajectory.npy",
                                       workdir / "result.json")
    for f in (stage_file, traj_file, rec_file):
        f.unlink(missing_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    pid = os.fork()
    if pid == 0:  # child: never return into the caller's stack
        code = 0
        try:
            def stage(text: str) -> None:
                stage_file.write_text(text, encoding="utf-8")

            trajectory, config = fn(stage)
            np.save(traj_file, np.asarray(trajectory))
            rec_file.write_text(json.dumps({"config": config}, default=str), encoding="utf-8")
        except BaseException as exc:  # noqa: BLE001 - every failure is reported
            try:
                rec_file.write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}"[:400]}),
                                    encoding="utf-8")
            except BaseException:  # noqa: BLE001
                code = 1
        finally:
            os._exit(code)

    deadline = time.monotonic() + budget
    t0 = time.perf_counter()
    status = None
    while status is None:
        done, st = os.waitpid(pid, os.WNOHANG)
        if done:
            status = st
            break
        if time.monotonic() >= deadline:
            where = stage_file.read_text(encoding="utf-8") if stage_file.exists() else "start"
            _kill_tree(pid)
            os.waitpid(pid, 0)
            return {"error": f"TIMEOUT: no result within the {budget:g} s per-engine budget "
                             f"(stopped in: {where})",
                    "seconds": time.perf_counter() - t0}
        time.sleep(poll)
    seconds = time.perf_counter() - t0
    if os.WIFSIGNALED(status):
        return {"error": f"crashed: killed by signal {os.WTERMSIG(status)}", "seconds": seconds}
    if not rec_file.exists():
        return {"error": f"crashed: exit status {os.WEXITSTATUS(status)}, no result",
                "seconds": seconds}
    rec = json.loads(rec_file.read_text(encoding="utf-8"))
    if "error" in rec:
        return {"error": rec["error"], "seconds": seconds}
    return {"trajectory": np.load(traj_file), "config": rec["config"], "seconds": seconds}


def _cold_bngsim(prep: dict[str, Any]):
    def fn(stage):
        t0 = time.perf_counter()
        stage(BUILD_STAGE["bngsim"])
        engine = BngsimODE(prep["net"], prep["proto"])
        stage("first solve")
        result = engine.run()
        return (BngsimODE.trajectory(result),
                {**engine.config(result), "cold_seconds": time.perf_counter() - t0})
    return fn


def _cold_reference(name: str, prep: dict[str, Any], out_prefix: Path, timeout: float):
    def fn(stage):
        cls = REFERENCE_ENGINES[name]
        t0 = time.perf_counter()
        stage(BUILD_STAGE.get(name, "build"))
        ref = _build_reference(name, prep, sparse=False, out_prefix=out_prefix, timeout=timeout)
        stage("first solve")
        values = ref.run()
        return cls.trajectory(values), {**ref.config(), "cold_seconds": time.perf_counter() - t0}
    return fn


# --------------------------------------------------------------------------- #
# Verdict and reproducibility
# --------------------------------------------------------------------------- #
def _prepare(problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    proto = protocol(problem)
    model = fetch.model_file(problem.source, ctx["cache_dir"])
    workdir = Path(ctx["workdir"])
    net, netgen_seconds = generate_network(
        model.read_text(encoding="utf-8"), proto["netgen"], workdir / "netgen",
        timeout=ctx["subprocess_timeout"],
    )
    size = network_size(net)
    sbml, species_names, name_to_id = export_sbml(net, workdir / "model.xml")
    info: dict[str, Any] = {"netgen_seconds": netgen_seconds, "network": size}
    if problem.size and any(problem.size.get(k) not in (None, v) for k, v in size.items()):
        info["size_mismatch"] = {"spec": problem.size, "generated": size}
    return {"net": net, "sbml": sbml, "species_names": species_names,
            "name_to_id": name_to_id, "proto": proto, "info": info}


def _verdict(bn: np.ndarray, ref: np.ndarray, atol: float) -> dict[str, Any]:
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


def max_err_over_peak(a: np.ndarray, b: np.ndarray) -> float | None:
    """The worst disagreement, as a fraction of each species' peak in ``b``.

    Reported beside the verdict for every comparison, passing or not: the
    verdict's own ``max_rel`` counts only failing cells and so reads 0 on a pass,
    which says nothing about how close two passing runs are. Species that are
    zero throughout ``b`` are left out (no scale to measure against); None when
    every species is.
    """
    peak = np.nanmax(np.abs(b), axis=0)
    moving = peak > 0
    if not np.any(moving):
        return None
    err = np.nanmax(np.abs(a[:, moving] - b[:, moving]), axis=0)
    return float(np.max(err / peak[moving]))


def _reproducibility(anchor: np.ndarray, others: list[np.ndarray], tol: float) -> dict[str, Any]:
    """Whether every repetition in ``others`` reproduces ``anchor``.

    Each timed repetition must solve the same problem the first run did; one that
    does not makes the ratio compare an engine with a run of something else (issue
    #8). The disagreement of each repetition is ``max_err_over_peak`` against the
    anchor (full trajectory, so a wrong transient that reconverges to the same
    final state is still caught); a species that never moves off zero contributes
    nothing (None reads as 0.0). ``passed`` is true when every repetition is within
    ``tol``; with nothing to compare (a single repetition) it is vacuously true.
    """
    per_rep: list[float] = []
    for other in others:
        n_t = min(anchor.shape[0], other.shape[0])
        n_sp = min(anchor.shape[1], other.shape[1])
        err = max_err_over_peak(other[:n_t, :n_sp], anchor[:n_t, :n_sp])
        per_rep.append(0.0 if err is None else float(err))
    worst = max(per_rep) if per_rep else 0.0
    return {
        "passed": worst <= tol,
        "tol": tol,
        "n_reps": len(others),
        "max_err_over_peak": worst,
        "worst_rep": int(np.argmax(per_rep)) if per_rep else None,
        "per_rep": per_rep,
    }


def sparse_check(net: Path, proto: dict[str, Any], workdir: Path, dense: np.ndarray,
                 timeout: float) -> dict[str, Any] | None:
    """Whether run_network with the authors' ``sparse`` option agrees with the
    dense reference; None when the protocol does not ask for sparse."""
    if not proto.get("sparse"):
        return None
    try:
        engine = RunNetworkODE(net, proto, workdir / "rn_sparse", sparse=True, timeout=timeout)
        t0 = time.perf_counter()
        values = engine.run()
        seconds = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        return {"agrees_with_dense": False, "error": f"{type(exc).__name__}: {exc}"[:400]}
    v = _verdict(values, dense, float(proto["atol"]))
    return {"agrees_with_dense": bool(v.get("passed")), "max_rel": v.get("max_rel"),
            "n_hard_fail": v.get("n_hard_fail"), "seconds": seconds}


def _outcome(bn_error: str | None, ref_errors: dict[str, str], n_refs: int,
             passed: bool | None) -> str:
    """The outcome, attributing a failure to BNGsim or the references.

    BAD_TEST when BNGsim failed and so did every reference (the model, not an
    engine, is the problem); EXCEPTION when only BNGsim failed; REFERENCE_FAILED
    when BNGsim ran but some reference failed; otherwise PASS when BNGsim agrees
    with every reference, else DIFF.
    """
    if bn_error and len(ref_errors) == n_refs:
        return "BAD_TEST"
    if bn_error:
        return "EXCEPTION"
    if ref_errors:
        return "REFERENCE_FAILED"
    return "PASS" if passed else "DIFF"


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
        out = isolated(name, _cold_reference(name, prep, workdir / name, timeout))
        if "error" in out:
            ref_errors[name] = out["error"]
        else:
            ref_trajectories[name] = out["trajectory"]
            references[name] = out["config"]
    info["references"] = references

    # run_network's authors' sparse setting, judged against its own dense run.
    if "run_network" in ref_trajectories:
        check = sparse_check(prep["net"], proto, workdir, ref_trajectories["run_network"], timeout)
        if check is not None:
            info["sparse_reference"] = check

    verdicts: dict[str, Any] = {}
    if bn is not None:
        for name, traj in ref_trajectories.items():
            verdicts[name] = _verdict(bn, traj, atol)
    passed = bool(verdicts) and all(v.get("passed") for v in verdicts.values())

    outcome = _outcome(bn_error, ref_errors, len(REFERENCE_ENGINES), passed)
    if bn_error and bn_error.startswith("TIMEOUT") and outcome == "EXCEPTION":
        outcome = "TIMEOUT"  # BNGsim, not a reference, ran out of time
    rec: dict[str, Any] = {"outcome": outcome, "measurements": info}
    if verdicts:
        rec["verdicts"] = verdicts
    if bn_error:
        rec["bngsim_error"] = bn_error
    for name, err in ref_errors.items():
        rec[f"{name}_error"] = err
    return rec


def _correctness_timeouts(job: Job, ctx: dict[str, Any]) -> dict[str, str]:
    """References recorded as timed out by this timing job's correctness job."""
    from ..results import result_path

    requires, results_dir = getattr(job, "requires", None), ctx.get("results_dir")
    if not requires or not results_dir:
        return {}
    try:
        rec = json.loads(result_path(results_dir, requires).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {name: str(rec[f"{name}_error"]) for name in REFERENCE_ENGINES
            if str(rec.get(f"{name}_error") or "").startswith("TIMEOUT")}


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

    # BNGsim cold: load and first integration, with the empty compile cache the
    # worker gave this job. Its trajectory anchors the warm runs' reproducibility
    # (a reset bug shows on the first warm run, which warm-vs-warm would miss).
    t0 = time.perf_counter()
    bngsim_engine = BngsimODE(prep["net"], proto)
    first = bngsim_engine.run()
    info["bngsim_cold_seconds"] = time.perf_counter() - t0
    info["bngsim"] = bngsim_engine.config(first)
    bngsim_cold = BngsimODE.trajectory(first)

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
        if name in timed_out:
            ref_errors[name] = f"TIMEOUT in the correctness job, so not timed ({timed_out[name]})"
            _progress(f"{name}: skipped, {ref_errors[name]}")
            continue
        _progress(f"{name}: build + trial run")
        try:
            if name == "run_network":
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
                engine.warm()  # trial run: COPASI/AMICI surface failures here, not in __init__
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
    captures = {name: type(eng).trajectory for name, eng in engines.items()}
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
    return rec


def run(job: Job, problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    if job.kind == CORRECTNESS:
        return run_correctness(job, problem, ctx)
    if job.kind == TIMING:
        return run_timing(job, problem, ctx)
    raise ValueError(f"unknown job kind {job.kind!r}")
