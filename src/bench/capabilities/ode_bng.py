"""ODE simulation of BNGL models: BNGsim against BioNetGen's run_network.

Pipeline, per problem:

1. **Network generation.** BNG2.pl runs on a copy of the model with every action
   removed and the spec's ``generate_network(...)`` call appended, producing a
   ``.net``. The same ``.net`` feeds both engines, so its cost is charged to
   neither.
2. **BNGsim** loads the ``.net`` in-process and integrates it.
3. **run_network** (BioNetGen's CVODE binary) integrates the same ``.net`` as a
   subprocess, the way BNG2.pl would run it.

The correctness job compares the two species trajectories with bngsim's own
deterministic verdict (``parity_core.differ``). The timing job pairs a warm
BNGsim run (model loaded, state reset) with a full run_network call (a fresh
process that reads the ``.net``, as every legacy simulation does) under the
paired protocol of ``bench.timing``; it also records BNGsim's cold cost.

Setting ``auto`` leaves every BNGsim choice (linear solver, compiled or
interpreted right-hand side) to bngsim, as a user gets it.

The reference for correctness is run_network with its default (dense) linear
solver, as in bngsim's own parity checks. A model's ``sparse`` option (BNG2.pl's
``sparse=>1``, run_network ``-b``, an iterative Krylov solver) is not trusted as
an oracle: on some models it leaves the solution by more than the comparison
tolerance even at tight tolerances. When a model asks for ``sparse``, the
correctness job also records whether the sparse run agrees with the dense one,
and the timing job times run_network with the authors' setting only when it
does; otherwise it times the dense run. A speedup is always measured against a
reference run that produced the right answer.
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
from ..timing import paired_interleaved, speedup

CAPABILITY = "ode"
ENGINES = "bngsim-vs-run_network"
# Bump when this module's computation changes; every result it wrote goes stale.
# 2: the correctness reference is run_network's dense solver, not the model's
#    sparse option; timing uses sparse only where it agrees with dense.
# 3: correctness records max_err_over_peak.
HARNESS_VERSION = "3"

TIMING_PROTOCOL = {"warmup_pairs": 1, "min_pairs": 5, "max_pairs": 20, "budget_seconds": 120.0}

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


def jobs(problem: Problem) -> list[Job]:
    if problem.source.get("format") != "bngl":
        return []
    proto = protocol(problem)
    base = {
        "harness": f"{CAPABILITY}/{ENGINES}/{HARNESS_VERSION}",
        "source_sha256": problem.source["sha256"],
        "protocol": proto,
        "setting": {"solver": "auto", "codegen": "auto"},
    }
    species = problem.size.get("species", 100)
    correctness = Job(
        CAPABILITY, problem.id, ENGINES, "auto", CORRECTNESS,
        definition=base,
        est_seconds=20.0 + 0.1 * species,
    )
    timing = Job(
        CAPABILITY, problem.id, ENGINES, "auto", TIMING,
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
    bngl.write_text(netgen_bngl(bngl_text, netgen_call))
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
    for line in net.read_text().splitlines():
        s = line.strip()
        if s.startswith("begin "):
            block = s.split()[1]
            counts.setdefault(block, 0)
        elif s.startswith("end "):
            block = None
        elif block and s and not s.startswith("#"):
            counts[block] += 1
    return {"species": counts.get("species", 0), "reactions": counts.get("reactions", 0)}


# --------------------------------------------------------------------------- #
# The two engines
# --------------------------------------------------------------------------- #
class BngsimODE:
    """One loaded BNGsim model, ready to integrate the protocol repeatedly."""

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
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s:
                if s.startswith("#"):
                    names = s.lstrip("#").split()[1:]
                break
    data = np.loadtxt(path, comments="#", ndmin=2)
    if not names:
        names = [f"S{i + 1}" for i in range(data.shape[1] - 1)]
    return data[:, 0].copy(), data[:, 1:], names


def run_network(cmd: list[str], out_prefix: Path, *, timeout: float):
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if proc.returncode != 0:
        tail = (proc.stderr.strip() or proc.stdout.strip()).splitlines()
        raise RuntimeError(f"run_network failed: {(tail[-1] if tail else '')[:300]}")
    cdat = Path(f"{out_prefix}.cdat")
    if not cdat.exists():
        raise RuntimeError("run_network produced no .cdat")
    return read_dat(cdat)


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
def _prepare(problem: Problem, ctx: dict[str, Any]) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    proto = protocol(problem)
    model = fetch.model_file(problem.source, ctx["cache_dir"])
    net, netgen_seconds = generate_network(
        model.read_text(), proto["netgen"], Path(ctx["workdir"]) / "netgen",
        timeout=ctx["subprocess_timeout"],
    )
    size = network_size(net)
    info = {"netgen_seconds": netgen_seconds, "network": size}
    if problem.size and any(problem.size.get(k) not in (None, v) for k, v in size.items()):
        info["size_mismatch"] = {"spec": problem.size, "generated": size}
    return net, proto, info


def _verdict(bn: np.ndarray, rn: np.ndarray, atol: float) -> dict[str, Any]:
    from .._vendor.parity_core import differ

    n_t = min(bn.shape[0], rn.shape[0])
    n_sp = min(bn.shape[1], rn.shape[1])
    if n_t == 0 or n_sp == 0:
        return {"passed": False, "reason": "nothing to compare"}
    v = differ.deterministic_verdict(bn[:n_t, :n_sp], rn[:n_t, :n_sp], atol=atol)
    keep = (
        "passed", "max_rel", "max_abs", "n_cells", "n_fail", "n_soft_fail",
        "n_hard_fail", "budget_forgiven", "no_signal", "max_err_over_movement",
    )
    out = {k: v[k] for k in keep if k in v}
    out["max_err_over_peak"] = max_err_over_peak(bn[:n_t, :n_sp], rn[:n_t, :n_sp])
    out["shapes"] = {"bngsim": list(bn.shape), "run_network": list(rn.shape)}
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
    does not makes the ratio compare BNGsim with a run of something else (issue
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


def _reference_run(net: Path, proto: dict[str, Any], workdir: Path, name: str, *,
                   sparse: bool, timeout: float) -> tuple[np.ndarray, list[str], float]:
    cmd = run_network_command(net, dict(proto, sparse=sparse), workdir / name)
    t0 = time.perf_counter()
    _, values, _ = run_network(cmd, workdir / name, timeout=timeout)
    return values, cmd, time.perf_counter() - t0


def sparse_check(net: Path, proto: dict[str, Any], workdir: Path, dense: np.ndarray,
                 timeout: float) -> dict[str, Any] | None:
    """Whether run_network with the authors' ``sparse`` option agrees with the
    dense reference; None when the protocol does not ask for sparse."""
    if not proto.get("sparse"):
        return None
    try:
        values, _, seconds = _reference_run(net, proto, workdir, "rn_sparse", sparse=True,
                                            timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return {"agrees_with_dense": False, "error": f"{type(exc).__name__}: {exc}"[:400]}
    v = _verdict(values, dense, float(proto["atol"]))
    return {"agrees_with_dense": bool(v.get("passed")), "max_rel": v.get("max_rel"),
            "n_hard_fail": v.get("n_hard_fail"), "seconds": seconds}


def _outcome(bn_error: str | None, rn_error: str | None, passed: bool | None) -> str:
    if bn_error and rn_error:
        return "BAD_TEST"
    if bn_error:
        return "EXCEPTION"
    if rn_error:
        return "REFERENCE_FAILED"
    return "PASS" if passed else "DIFF"


def run_correctness(job: Job, problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    try:
        net, proto, info = _prepare(problem, ctx)
    except Exception as exc:  # noqa: BLE001
        return {"outcome": "BAD_TEST", "error": f"{type(exc).__name__}: {exc}"}
    bn = rn = None
    bn_error = rn_error = None
    try:
        t0 = time.perf_counter()
        engine = BngsimODE(net, proto)
        result = engine.run()
        info["bngsim"] = {**engine.config(result), "cold_seconds": time.perf_counter() - t0}
        bn = np.asarray(result.species)
    except Exception as exc:  # noqa: BLE001
        bn_error = f"{type(exc).__name__}: {exc}"[:400]
    workdir = Path(ctx["workdir"])
    try:
        rn, cmd, seconds = _reference_run(net, proto, workdir, "rn", sparse=False,
                                          timeout=ctx["subprocess_timeout"])
        info["run_network"] = {"seconds": seconds, "solver": "dense",
                               "command": recorded_command(cmd)}
    except Exception as exc:  # noqa: BLE001
        rn_error = f"{type(exc).__name__}: {exc}"[:400]
    if rn is not None:
        check = sparse_check(net, proto, workdir, rn, ctx["subprocess_timeout"])
        if check is not None:
            info["sparse_reference"] = check
    verdict = _verdict(bn, rn, float(proto["atol"])) if bn is not None and rn is not None else None
    rec: dict[str, Any] = {
        "outcome": _outcome(bn_error, rn_error, verdict and verdict.get("passed")),
        "measurements": info,
    }
    if verdict is not None:
        rec["verdict"] = verdict
    if bn_error:
        rec["bngsim_error"] = bn_error
    if rn_error:
        rec["run_network_error"] = rn_error
    return rec


def run_timing(job: Job, problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    net, proto, info = _prepare(problem, ctx)
    # Cold: load and first integration of a fresh model, with the empty compile
    # cache the worker gave this job.
    t0 = time.perf_counter()
    engine = BngsimODE(net, proto)
    first = engine.run()
    info["bngsim_cold_seconds"] = time.perf_counter() - t0
    info["bngsim"] = engine.config(first)
    # Copy now: .species may be a buffer the next run overwrites, which would make
    # the reproducibility anchor silently track the last warm run.
    cold_a = np.array(first.species)
    workdir = Path(ctx["workdir"])
    timeout = ctx["subprocess_timeout"]
    # Time run_network as the authors ran it only if that run is right.
    dense, _, _ = _reference_run(net, proto, workdir, "rn_dense", sparse=False, timeout=timeout)
    check = sparse_check(net, proto, workdir, dense, timeout)
    use_sparse = bool(check and check["agrees_with_dense"])
    info["reference_config"] = {
        "solver": "sparse (-b)" if use_sparse else "dense",
        "authors_setting": "sparse" if proto.get("sparse") else "dense",
    }
    if check is not None:
        info["sparse_reference"] = check
    out = workdir / "rn"
    cmd = run_network_command(net, dict(proto, sparse=use_sparse), out)
    reps = paired_interleaved(
        engine.warm,
        lambda: run_network(cmd, out, timeout=timeout),
        capture_a=lambda result: np.array(result.species),
        capture_b=lambda res: np.array(res[1]),
        **job.definition["timing"],
    )
    # Every kept repetition must reproduce the first run, or the ratio compares
    # BNGsim with a run of something else (issue #8). BNGsim's warm re-runs are
    # measured against its cold first run (a reset bug shows on the first warm run,
    # which warm-vs-warm would miss); run_network's are measured against its own
    # first repetition, not the dense reference -- a legitimately-sparse timed run
    # differs from dense but is still reproducible.
    a_outputs = reps.pop("a_outputs")
    b_outputs = reps.pop("b_outputs")
    repro = {
        "bngsim": _reproducibility(cold_a, a_outputs, REPRO_TOL),
        "run_network": _reproducibility(b_outputs[0], b_outputs[1:], REPRO_TOL),
    }
    reproducible = repro["bngsim"]["passed"] and repro["run_network"]["passed"]
    stats = speedup(reps["a_seconds"], reps["b_seconds"])
    return {
        "outcome": "PASS" if reproducible else "NOT_REPRODUCIBLE",
        "measurements": {
            **info,
            "engines": {"a": "bngsim (warm, in-process)", "b": "run_network (subprocess)"},
            "repetitions": reps,
            "reproducibility": repro,
            "summary": stats,
        },
    }


def run(job: Job, problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    if job.kind == CORRECTNESS:
        return run_correctness(job, problem, ctx)
    if job.kind == TIMING:
        return run_timing(job, problem, ctx)
    raise ValueError(f"unknown job kind {job.kind!r}")
