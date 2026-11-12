"""Slurm job arrays for unattended runs: ``bench submit``.

``bench submit`` picks the jobs that need running by the same rule as
``bench run``, packs them into array tasks that fit the allocation's wall-time
limit, writes a submission directory, and hands its batch script to sbatch:

    slurm/<UTC time>/
        submit.sbatch     the array job script
        tasks/0000.txt    the job ids array task 0 runs, one per line
        logs/             one log per array task (%A_%a.out)
        plan.json         the settings, each task's jobs and estimated cost,
                          and the Slurm job id once submitted

Each array task runs ``bench run --jobs-from tasks/NNNN.txt`` on its node with a
deadline at the end of its allocation: correctness jobs first, then the timing
jobs whose correctness job passed, as on a checked-out node. A problem's
correctness and timing jobs always go to the same task, so a timing job never
waits on another task. A job that does not finish by the deadline is stopped
without a result; ``bench plan`` shows it and the next submit picks it up.

The cluster settings come from a YAML file that is not committed, because it
names local infrastructure. ``slurm.example.yaml`` shows every key.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import re
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .jobs import TIMING, Job
from .specs import Problem


class ConfigError(ValueError):
    def __init__(self, path: Path | str, problems: list[str]):
        self.path = str(path)
        self.problems = problems
        super().__init__(f"{path}:\n" + "\n".join(f"  - {p}" for p in problems))


class SubmitError(RuntimeError):
    pass


# A value written after "#SBATCH --key=". Slurm reads these lines itself, not a
# shell, so nothing can be quoted: refuse whitespace, quotes and shell syntax.
_SBATCH_VALUE = re.compile(r"^[A-Za-z0-9_.,:|&!*()\[\]+=@/%-]+$")
_SBATCH_OPTION = re.compile(r"^--[A-Za-z][A-Za-z0-9-]*(=[A-Za-z0-9_.,:|&!*()\[\]+=@/%-]+)?$")
# Paths written into #SBATCH lines and the script: plain characters only.
_PLAIN_PATH = re.compile(r"^[A-Za-z0-9_.,:+=@/~-]+$")


@dataclass(frozen=True)
class SlurmConfig:
    time_limit: str
    cpus_per_task: int
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    constraint: str | None = None
    exclusive: bool = False
    timing_workers: int = 1
    max_concurrent: int | None = None
    max_array_size: int = 1001
    margin_seconds: float = 300.0
    startup_seconds: float = 300.0
    node_type: str | None = None
    job_name: str = "bngsim-bench"
    bench: str = "pixi run --frozen bench"
    setup: tuple[str, ...] = ()
    extra_sbatch: tuple[str, ...] = ()

    @property
    def time_limit_seconds(self) -> int:
        return parse_time_limit(self.time_limit)

    @property
    def capacity_seconds(self) -> float:
        """How long the jobs of one array task may take, by their estimates."""
        return self.time_limit_seconds - self.margin_seconds - self.startup_seconds


_REQUIRED = ("time_limit", "cpus_per_task")
_KEYS = {f.name for f in dataclasses.fields(SlurmConfig)}


def parse_time_limit(text: str) -> int:
    """Seconds in a Slurm time limit: ``M``, ``M:S``, ``H:M:S``, ``D-H``,
    ``D-H:M`` or ``D-H:M:S``."""
    m = re.fullmatch(r"(?:(\d+)-)?(\d+)(?::(\d+))?(?::(\d+))?", text.strip())
    if not m:
        raise ValueError(f"not a Slurm time limit: {text!r}")
    days, a, b, c = m.groups()
    a, b, c = int(a), (int(b) if b is not None else None), (int(c) if c is not None else None)
    if days is not None:  # D-H, D-H:M, D-H:M:S
        h, mi, s = a, b or 0, c or 0
        if any(x >= 60 for x in (mi, s)):
            raise ValueError(f"not a Slurm time limit: {text!r}")
        total = int(days) * 86400 + h * 3600 + mi * 60 + s
    elif c is not None:  # H:M:S
        if b >= 60 or c >= 60:
            raise ValueError(f"not a Slurm time limit: {text!r}")
        total = a * 3600 + b * 60 + c
    elif b is not None:  # M:S
        if b >= 60:
            raise ValueError(f"not a Slurm time limit: {text!r}")
        total = a * 60 + b
    else:  # M
        total = a * 60
    if total <= 0:
        raise ValueError(f"a Slurm time limit must be positive: {text!r}")
    return total


def _positive_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def _number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0


def validate_config(data: Any, path: Path | str = "<config>") -> SlurmConfig:
    """A SlurmConfig, or a ConfigError naming every bad field at once."""
    errs: list[str] = []
    if not isinstance(data, dict):
        raise ConfigError(path, ["the file must hold a mapping of settings"])
    for k in sorted(set(data) - _KEYS):
        errs.append(f"unknown key {k!r}")
    for k in _REQUIRED:
        if data.get(k) is None:
            errs.append(f"{k} is required")

    tl = data.get("time_limit")
    if tl is not None:
        if not isinstance(tl, str):
            # YAML reads an unquoted 10:00:00 as the integer 36000 (base 60).
            errs.append("time_limit must be a quoted string, e.g. \"10:00:00\"")
        else:
            try:
                parse_time_limit(tl)
            except ValueError as exc:
                errs.append(f"time_limit: {exc}")
    for k in ("cpus_per_task", "timing_workers", "max_array_size", "max_concurrent"):
        v = data.get(k)
        if v is not None and not _positive_int(v):
            errs.append(f"{k} must be a positive integer, not {v!r}")
    for k in ("margin_seconds", "startup_seconds"):
        v = data.get(k)
        if v is not None and not _number(v):
            errs.append(f"{k} must be a number of seconds >= 0, not {v!r}")
    for k in ("partition", "account", "qos", "constraint", "job_name"):
        v = data.get(k)
        if v is not None and not (isinstance(v, str) and _SBATCH_VALUE.match(v)):
            errs.append(f"{k} {v!r} is not a plain Slurm value (no spaces, quotes or shell syntax)")
    if data.get("exclusive") is not None and not isinstance(data["exclusive"], bool):
        errs.append("exclusive must be true or false")
    nt = data.get("node_type")
    if nt is not None and not (isinstance(nt, str) and nt.strip() and "\n" not in nt):
        errs.append("node_type must be the one-line node type `bench doctor` prints on a compute node")
    b = data.get("bench")
    if b is not None and not (isinstance(b, str) and b.strip() and "\n" not in b):
        errs.append("bench must be a one-line command")
    for k in ("setup", "extra_sbatch"):
        v = data.get(k)
        if v is None:
            continue
        if not (isinstance(v, list) and all(isinstance(x, str) for x in v)):
            errs.append(f"{k} must be a list of strings")
            continue
        for x in v:
            if "\n" in x:
                errs.append(f"{k}: {x!r} spans more than one line")
            elif k == "extra_sbatch" and not _SBATCH_OPTION.match(x):
                errs.append(f"extra_sbatch: {x!r} is not one --option or --option=value")
    if not errs:
        kw = {k: v for k, v in data.items() if v is not None}
        for k in ("setup", "extra_sbatch"):
            if k in kw:
                kw[k] = tuple(kw[k])
        for k in ("margin_seconds", "startup_seconds"):
            if k in kw:
                kw[k] = float(kw[k])
        cfg = SlurmConfig(**kw)
        if cfg.capacity_seconds <= 0:
            errs.append(
                f"time_limit {cfg.time_limit} leaves no time for jobs after margin_seconds "
                f"{cfg.margin_seconds:g} and startup_seconds {cfg.startup_seconds:g}")
        else:
            return cfg
    raise ConfigError(path, errs)


def load_config(path: Path | str) -> SlurmConfig:
    p = Path(path)
    if not p.exists():
        raise ConfigError(p, [(
            "no such file. Copy slurm.example.yaml to slurm.yaml and fill it in "
            "(slurm.yaml is git-ignored: it names local infrastructure)")])
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(p, [f"not valid YAML: {exc}"]) from exc
    return validate_config(data, p)


# --- Packing ---------------------------------------------------------------


def phase_bound(costs: list[float], workers: int) -> float:
    """An upper bound on how long ``workers`` take to run ``costs`` when each
    free worker takes the next job in list order, as the runner does (Graham's
    list-scheduling bound: sum/w + (1 - 1/w) * max)."""
    if not costs:
        return 0.0
    return sum(costs) / workers + (1.0 - 1.0 / workers) * max(costs)


@dataclass
class Task:
    jobs: list[Job] = field(default_factory=list)
    # running sums and maxima of the two phases' estimated costs
    c_sum: float = 0.0
    c_max: float = 0.0
    t_sum: float = 0.0
    t_max: float = 0.0

    def bound(self, cfg: SlurmConfig) -> float:
        return _bound(self.c_sum, self.c_max, self.t_sum, self.t_max, cfg)

    def add(self, unit: list[Job]) -> None:
        self.jobs.extend(unit)
        for j in unit:
            if j.kind == TIMING:
                self.t_sum += j.est_seconds
                self.t_max = max(self.t_max, j.est_seconds)
            else:
                self.c_sum += j.est_seconds
                self.c_max = max(self.c_max, j.est_seconds)

    def bound_with(self, unit: list[Job], cfg: SlurmConfig) -> float:
        c = [j.est_seconds for j in unit if j.kind != TIMING]
        t = [j.est_seconds for j in unit if j.kind == TIMING]
        return _bound(self.c_sum + sum(c), max([self.c_max, *c]),
                      self.t_sum + sum(t), max([self.t_max, *t]), cfg)


def _bound(c_sum: float, c_max: float, t_sum: float, t_max: float, cfg: SlurmConfig) -> float:
    w, tw = cfg.cpus_per_task, cfg.timing_workers
    c = c_sum / w + (1.0 - 1.0 / w) * c_max if c_sum else 0.0
    t = t_sum / tw + (1.0 - 1.0 / tw) * t_max if t_sum else 0.0
    return c + t


@dataclass
class Packing:
    tasks: list[Task]
    long: list[list[Job]]  # units too long for one allocation, not packed


def to_run(jobs: list[Job], results_dir: Path | str,
           fingerprints: dict[str, dict[str, Any]]) -> list[Job]:
    """The jobs ``bench run`` would launch: correctness jobs without a current
    result, and timing jobs without one whose correctness job has passed or will
    run first. Blocked timing jobs (their correctness job did not pass) are left
    out."""
    from .results import status

    by_id = {j.id: j for j in jobs}
    out = []
    for j in jobs:
        st = status(j, results_dir, fingerprints, by_id).state
        if st in ("missing", "stale") or (j.kind == TIMING and st == "waiting"):
            out.append(j)
    return out


def pack(jobs: list[Job], cfg: SlurmConfig) -> Packing:
    """Pack ``jobs`` (all to be run) into array tasks by first-fit decreasing.

    A unit is a correctness job and the timing jobs that require it (or such a
    timing job alone, when its correctness result is already current). Each
    task's bound on its run time stays within ``cfg.capacity_seconds``. A unit
    whose own bound exceeds it cannot be run in one allocation: it is returned in
    ``long`` and not packed.
    """
    units: dict[str, list[Job]] = {}
    for j in jobs:
        units.setdefault(j.requires or j.id, []).append(j)
    cap = cfg.capacity_seconds
    ordered = sorted(units.values(),
                     key=lambda u: (-Task().bound_with(u, cfg), u[0].id))
    tasks: list[Task] = []
    long: list[list[Job]] = []
    for unit in ordered:
        if Task().bound_with(unit, cfg) > cap:
            long.append(unit)
            continue
        for task in tasks:
            if task.bound_with(unit, cfg) <= cap:
                task.add(unit)
                break
        else:
            t = Task()
            t.add(unit)
            tasks.append(t)
    order = {j.id: i for i, j in enumerate(jobs)}
    for t in tasks:
        t.jobs.sort(key=lambda j: order[j.id])
    return Packing(tasks, long)


# --- The submission directory ---------------------------------------------


def batch_script(cfg: SlurmConfig, *, n_tasks: int, sub_dir: Path, root: Path,
                 problems_dir: Path, results_dir: Path, cache_dir: Path,
                 created: str) -> str:
    for name, p in (("submission directory", sub_dir), ("repository", root)):
        if not _PLAIN_PATH.match(str(p)):
            raise SubmitError(f"the {name} path {str(p)!r} has characters a #SBATCH line "
                              "cannot hold; use a path without spaces or quotes")
    array = f"0-{n_tasks - 1}" + (f"%{cfg.max_concurrent}" if cfg.max_concurrent else "")
    sb = [
        f"--job-name={cfg.job_name}",
        f"--array={array}",
        f"--time={cfg.time_limit}",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={cfg.cpus_per_task}",
    ]
    for key in ("partition", "account", "qos", "constraint"):
        v = getattr(cfg, key)
        if v:
            sb.append(f"--{key}={v}")
    if cfg.exclusive:
        sb.append("--exclusive")
    sb.append(f"--output={sub_dir}/logs/%A_%a.out")
    sb.extend(cfg.extra_sbatch)
    q = shlex.quote
    lines = ["#!/bin/bash", *(f"#SBATCH {x}" for x in sb),
             f"# Written by `bench submit` at {created}. plan.json in this directory",
             "# lists each task's jobs and estimated cost.",
             "set -euo pipefail",
             "START=$(date +%s)",
             f"cd {q(str(root))}"]
    if cfg.setup:
        lines += ["# setup (from the config file)", *cfg.setup]
    lines += [
        "TASK=$(printf '%04d' \"$SLURM_ARRAY_TASK_ID\")",
        "# The deadline is the end of this allocation, less what has run already.",
        f"LEFT=$(( {cfg.time_limit_seconds} - ( $(date +%s) - START ) ))",
        f"{cfg.bench} run --jobs-from {q(str(sub_dir / 'tasks'))}/\"$TASK\".txt \\",
        (f"    --problems {q(str(problems_dir))} --results {q(str(results_dir))} "
         f"--cache {q(str(cache_dir))} \\"),
        (f"    --workers {cfg.cpus_per_task} --timing-workers {cfg.timing_workers} "
         f"--budget \"$LEFT\" --margin {cfg.margin_seconds:g}"),
        "",
    ]
    return "\n".join(lines)


def preflight(problems: list[Problem], cache_dir: Path) -> None:
    """Fail here, on the login node, rather than in every array task: bngsim and
    BioNetGen must be installed, and every model is downloaded and checked now,
    so compute nodes need no network."""
    from . import env, fetch

    b, g = env.bngsim_identity(), env.bionetgen_identity()
    if not b.get("build_commit"):
        raise SubmitError(f"bngsim is not usable here: {b.get('error') or 'no build commit'}")
    if g.get("error"):
        raise SubmitError(f"BioNetGen: {g['error']} (run `pixi run fetch-bionetgen`)")
    for p in problems:
        try:
            fetch.model_file(p.source, cache_dir)
        except Exception as exc:
            raise SubmitError(f"could not fetch the model for {p.id}: {exc}") from exc


def run_sbatch(script: Path) -> str:
    """Submit ``script``; the Slurm job id."""
    try:
        out = subprocess.run(["sbatch", "--parsable", str(script)], capture_output=True,
                             text=True, timeout=120, check=False)
    except FileNotFoundError as exc:
        raise SubmitError("sbatch not found: run `bench submit` on a Slurm login node, "
                          "or use --dry-run") from exc
    if out.returncode != 0:
        raise SubmitError(f"sbatch failed ({out.returncode}): {out.stderr.strip()}")
    m = re.match(r"^(\d+)(;\S+)?$", out.stdout.strip())
    if not m:
        raise SubmitError(f"could not read a job id from sbatch's output: {out.stdout!r}")
    return m.group(1)


@dataclass
class Submission:
    directory: Path | None
    packing: Packing
    job_id: str | None = None


def submit(
    jobs: list[Job],
    problems: list[Problem],
    cfg: SlurmConfig,
    *,
    root: Path,
    problems_dir: Path,
    results_dir: Path,
    cache_dir: Path,
    out_dir: Path,
    fingerprints: dict[str, dict[str, Any]],
    dry_run: bool = False,
    skip_long: bool = False,
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
    sbatch: Callable[[Path], str] | None = None,
    check: Callable[[list[Problem], Path], None] | None = None,
) -> Submission:
    """Pack the jobs that need running, write a submission directory, and submit
    it (unless ``dry_run``). Nothing is written when nothing needs running."""
    if cfg.node_type:
        # Judge timing results against the compute nodes' type, not this node's.
        fingerprints = {**fingerprints,
                        TIMING: {**fingerprints[TIMING], "node_type": cfg.node_type}}
    todo = to_run(jobs, results_dir, fingerprints)
    packing = pack(todo, cfg)
    if packing.long and not skip_long:
        names = "\n".join(f"  {u[0].requires or u[0].id}" for u in packing.long)
        raise SubmitError(
            f"{len(packing.long)} problems are estimated to take longer than one "
            f"allocation ({cfg.capacity_seconds:g} s of {cfg.time_limit}):\n{names}\n"
            "Run them on a checked-out node with `bench run`, or pass --skip-long "
            "to submit the rest.")
    if not packing.tasks:
        return Submission(None, packing)
    if len(packing.tasks) > cfg.max_array_size:
        raise SubmitError(
            f"{len(packing.tasks)} array tasks exceed max_array_size {cfg.max_array_size}; "
            "select fewer jobs, or raise max_array_size if the cluster allows it")
    needed = {j.problem for t in packing.tasks for j in t.jobs}
    (check or preflight)([p for p in problems if p.id in needed], cache_dir)

    created = now()
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    sub_dir = out_dir.resolve() / stamp
    if sub_dir.exists():
        raise SubmitError(f"{sub_dir} already exists")
    (sub_dir / "tasks").mkdir(parents=True)
    (sub_dir / "logs").mkdir()
    for i, t in enumerate(packing.tasks):
        (sub_dir / "tasks" / f"{i:04d}.txt").write_text(
            "".join(f"{j.id}\n" for j in t.jobs), encoding="utf-8")
    script = sub_dir / "submit.sbatch"
    script.write_text(batch_script(
        cfg, n_tasks=len(packing.tasks), sub_dir=sub_dir, root=root.resolve(),
        problems_dir=problems_dir.resolve(), results_dir=results_dir.resolve(),
        cache_dir=cache_dir.resolve(), created=created.isoformat(timespec="seconds")),
        encoding="utf-8")
    plan = {
        "created": created.isoformat(timespec="seconds"),
        "config": dataclasses.asdict(cfg),
        "capacity_seconds": cfg.capacity_seconds,
        "tasks": [{"index": i, "est_seconds": round(t.bound(cfg), 1),
                   "jobs": [j.id for j in t.jobs]} for i, t in enumerate(packing.tasks)],
        "long": [[j.id for j in u] for u in packing.long],
        "slurm_job_id": None,
    }
    plan_path = sub_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    sub = Submission(sub_dir, packing)
    if dry_run:
        return sub
    sub.job_id = (sbatch or run_sbatch)(script)
    plan["slurm_job_id"] = sub.job_id
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return sub
