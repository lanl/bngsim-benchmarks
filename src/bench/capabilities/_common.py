"""What the capabilities share: building a network from BNGL, exporting it to SBML,
reading BioNetGen output, running one engine in a fork under a wall budget, and
turning per-engine results into an outcome.

A capability (``ode``, ``ssa``, ...) compares BNGsim against reference engines on
the same generated network. These helpers are the parts of that pipeline that do
not depend on the method: they were written for, and first lived in, ``ode``,
which still exposes them under their old names. Nothing here enters a job
definition or a freshness key, so moving code between this module and a
capability never restales a result.
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
from ..jobs import Job
from ..specs import Problem

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


def progress(tag: str, msg: str) -> None:
    """One flushed line on stderr: the runner keeps the tail of a job's output in
    a TIMEOUT record, so this says where a stopped job was."""
    print(f"[{tag} {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Network generation and SBML export
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


def prepare_network(problem: Problem, ctx: dict[str, Any], netgen_call: str) -> dict[str, Any]:
    """Fetch the problem's BNGL, build its network with ``netgen_call`` and export
    the network to SBML, once per job and charged to no engine.

    Returns the ``.net`` and SBML paths, BNGsim's species order, the species-name
    -> SBML-id map the SBML engines align on, and ``info`` (netgen time, the
    generated size, and any disagreement with the size the spec records).
    """
    model = fetch.model_file(problem.source, ctx["cache_dir"])
    workdir = Path(ctx["workdir"])
    net, netgen_seconds = generate_network(
        model.read_text(encoding="utf-8"), netgen_call, workdir / "netgen",
        timeout=ctx["subprocess_timeout"],
    )
    size = network_size(net)
    sbml, species_names, name_to_id = export_sbml(net, workdir / "model.xml")
    info: dict[str, Any] = {"netgen_seconds": netgen_seconds, "network": size}
    if problem.size and any(problem.size.get(k) not in (None, v) for k, v in size.items()):
        info["size_mismatch"] = {"spec": problem.size, "generated": size}
    return {"net": net, "sbml": sbml, "species_names": species_names,
            "name_to_id": name_to_id, "info": info}


# --------------------------------------------------------------------------- #
# BioNetGen output
# --------------------------------------------------------------------------- #
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


def engine_role(name: str) -> str:
    """How an engine is timed, for the human-readable timing record (``gather``
    reads it). Derived from the engine name so it never drifts from a registry:
    run_network is a fresh subprocess per run; every other reference (and BNGsim)
    is an in-process library timed warm."""
    if name == "bngsim":
        return "warm, in-process (system under test)"
    if name == "run_network":
        return "fresh subprocess"
    return "warm, in-process"


# --------------------------------------------------------------------------- #
# Per-engine isolation
# --------------------------------------------------------------------------- #
def threads() -> int | str:
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


# --------------------------------------------------------------------------- #
# Closeness, reproducibility and outcome
# --------------------------------------------------------------------------- #
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


def reproducibility(anchor: np.ndarray, others: list[np.ndarray], tol: float) -> dict[str, Any]:
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


def outcome(bn_error: str | None, ref_errors: dict[str, str], n_refs: int,
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


def correctness_timeouts(job: Job, ctx: dict[str, Any], names) -> dict[str, str]:
    """The references in ``names`` recorded as timed out by this timing job's
    correctness job."""
    from ..results import result_path

    requires, results_dir = getattr(job, "requires", None), ctx.get("results_dir")
    if not requires or not results_dir:
        return {}
    try:
        rec = json.loads(result_path(results_dir, requires).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {name: str(rec[f"{name}_error"]) for name in names
            if str(rec.get(f"{name}_error") or "").startswith("TIMEOUT")}
