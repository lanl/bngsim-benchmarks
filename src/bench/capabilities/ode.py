"""ODE simulation of BNGL models: BNGsim against a registry of reference engines.

BNGsim is the system under test; it is compared, on one problem, against every
reference ODE engine in ``REFERENCE_ENGINES``. Seeded now with:

- **run_network** -- BioNetGen's CVODE binary, the trusted dense oracle, run as a
  fresh subprocess the way BNG2.pl runs it (reads the ``.net``).
- **roadrunner** -- libRoadRunner's CVODE, an in-process library that reads SBML
  exported from the same ``.net``, so its verdict also validates the export.

AMICI and COPASI are designed to drop in as further ``REFERENCE_ENGINES`` entries
(both SBML, both the same warm/run/config contract); nothing else changes when
one is added.

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

import re
import subprocess
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
HARNESS_VERSION = "4"

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
        # The libRoadRunner version keys only this capability's freshness, not the
        # global per-kind fingerprint: a roadrunner change restales ode results,
        # never a future capability that does not use it (see env.fingerprint).
        "reference_versions": {"roadrunner": env.roadrunner_identity().get("version")},
    }
    species = problem.size.get("species", 100)
    correctness = Job(
        CAPABILITY, problem.id, engines, "auto", CORRECTNESS,
        definition=base,
        est_seconds=20.0 + 0.1 * species,
    )
    timing = Job(
        CAPABILITY, problem.id, engines, "auto", TIMING,
        definition={**base, "timing": TIMING_PROTOCOL, "timing_check": REPRO_CHECK_VERSION},
        est_seconds=60.0 + 0.5 * species,
        requires=correctness.id,
    )
    return [correctness, timing]


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
        import roadrunner

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


# The reference engines BNGsim is compared against, in a fixed order. run_network
# builds from the ``.net``; roadrunner (and, later, amici/copasi) from the SBML.
REFERENCE_ENGINES: dict[str, type] = {
    RunNetworkODE.NAME: RunNetworkODE,
    RoadRunnerODE.NAME: RoadRunnerODE,
}


def _build_reference(name: str, prep: dict[str, Any], *, sparse: bool,
                     out_prefix: Path, timeout: float):
    """Construct one reference engine from the prepared artifacts."""
    cls = REFERENCE_ENGINES[name]
    if cls.INPUT == "net":
        return cls(prep["net"], prep["proto"], out_prefix, sparse=sparse, timeout=timeout)
    return cls(prep["sbml"], prep["species_names"], prep["name_to_id"], prep["proto"])


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
    timeout = ctx["subprocess_timeout"]

    bn = None
    bn_error = None
    try:
        t0 = time.perf_counter()
        engine = BngsimODE(prep["net"], proto)
        result = engine.run()
        info["bngsim"] = {**engine.config(result), "cold_seconds": time.perf_counter() - t0}
        bn = BngsimODE.trajectory(result)
    except Exception as exc:  # noqa: BLE001
        bn_error = f"{type(exc).__name__}: {exc}"[:400]

    references: dict[str, Any] = {}
    ref_trajectories: dict[str, np.ndarray] = {}
    ref_errors: dict[str, str] = {}
    for name, cls in REFERENCE_ENGINES.items():
        try:
            t0 = time.perf_counter()
            ref = _build_reference(name, prep, sparse=False, out_prefix=workdir / name,
                                   timeout=timeout)
            values = ref.run()
            ref_trajectories[name] = cls.trajectory(values)
            references[name] = {**ref.config(), "cold_seconds": time.perf_counter() - t0}
        except Exception as exc:  # noqa: BLE001
            ref_errors[name] = f"{type(exc).__name__}: {exc}"[:400]
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

    rec: dict[str, Any] = {
        "outcome": _outcome(bn_error, ref_errors, len(REFERENCE_ENGINES), passed),
        "measurements": info,
    }
    if verdicts:
        rec["verdicts"] = verdicts
    if bn_error:
        rec["bngsim_error"] = bn_error
    for name, err in ref_errors.items():
        rec[f"{name}_error"] = err
    return rec


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

    # Build each reference. run_network is timed as the authors ran it only if that
    # run is right: time sparse only where it agrees with the dense run.
    references: dict[str, Any] = {}
    engines: dict[str, Any] = {"bngsim": bngsim_engine}
    for name in REFERENCE_ENGINES:
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
        engines[name] = engine
        references[name] = engine.config()
    info["references"] = references

    # Interleave every engine round-robin in one drift-cancelling run. The timed
    # callable returns each engine's raw result; the trajectory is extracted
    # outside the timed region (the capture), so it does not bias the measurement.
    runs = {name: eng.warm for name, eng in engines.items()}
    captures = {name: type(eng).trajectory for name, eng in engines.items()}
    reps = round_robin_interleaved(runs, captures=captures, **job.definition["timing"])

    # Every kept repetition must reproduce the first run (issue #8). BNGsim's warm
    # re-runs are measured against its cold first run; each reference's are measured
    # against its own first kept repetition (a legitimately-sparse run_network run
    # differs from dense but is still reproducible).
    outputs = {name: reps["engines"][name].pop("outputs") for name in runs}
    repro: dict[str, Any] = {"bngsim": _reproducibility(bngsim_cold, outputs["bngsim"], REPRO_TOL)}
    for name in REFERENCE_ENGINES:
        outs = outputs[name]
        repro[name] = _reproducibility(outs[0], outs[1:], REPRO_TOL) if outs else \
            {"passed": True, "n_reps": 0, "max_err_over_peak": 0.0, "worst_rep": None,
             "per_rep": [], "tol": REPRO_TOL}
    reproducible = all(r["passed"] for r in repro.values())

    seconds_by_engine = {name: reps["engines"][name]["seconds"] for name in runs}
    times = absolute_time_stats(seconds_by_engine)
    return {
        "outcome": "PASS" if reproducible else "NOT_REPRODUCIBLE",
        "measurements": {
            **info,
            "engine_roles": {"bngsim": "warm, in-process (system under test)",
                             "run_network": "fresh subprocess", "roadrunner": "warm, in-process"},
            "repetitions": reps,
            "reproducibility": repro,
            "times": times,
        },
    }


def run(job: Job, problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    if job.kind == CORRECTNESS:
        return run_correctness(job, problem, ctx)
    if job.kind == TIMING:
        return run_timing(job, problem, ctx)
    raise ValueError(f"unknown job kind {job.kind!r}")
