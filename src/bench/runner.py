"""Run jobs: serially, or as a pool of worker processes on one node.

Correctness jobs run first, at ``workers`` concurrency. Timing jobs run after,
at ``timing_workers`` concurrency (1 unless a node-sharing check has shown that
more do not disturb the ratios), and only for problems whose correctness job
passed on this build.

A deadline (``until``) bounds the run, for allocations with a wall-time limit:
a job is not started unless its estimated cost fits before the deadline less a
safety margin, and a job still running when the deadline arrives is stopped
without a result, so the next run picks it up. Finished jobs are never lost:
each one wrote its own result file.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .jobs import CORRECTNESS, TIMING, Job
from .results import freshness_key, key_parts, status, write_result


@dataclass(frozen=True)
class Launch:
    returncode: int | None
    timed_out: bool
    seconds: float
    output: str = ""


@dataclass
class RunConfig:
    problems_dir: Path
    results_dir: Path
    cache_dir: Path
    workers: int = 1
    timing_workers: int = 1
    until: float | None = None  # epoch seconds
    margin_seconds: float = 60.0
    limit: int | None = None  # launch at most this many jobs
    force: bool = False
    python: str = sys.executable
    quiet: bool = False


@dataclass
class RunSummary:
    launched: list[str] = field(default_factory=list)
    outcomes: dict[str, str] = field(default_factory=dict)
    deferred: list[str] = field(default_factory=list)  # did not fit before the deadline
    cut: list[str] = field(default_factory=list)  # stopped at the deadline, no result
    skipped: dict[str, str] = field(default_factory=dict)  # fresh / blocked / waiting
    failed: dict[str, str] = field(default_factory=dict)  # worker crashed, no result
    done: int = 0  # jobs finished, for progress


Launcher = Callable[[Job, float, "int | None", RunConfig], Launch]


def subprocess_launcher(job: Job, timeout: float, cpu: int | None, cfg: RunConfig) -> Launch:
    """Run ``job`` in a fresh worker process; kill its whole process group on timeout."""
    cmd = [
        cfg.python, "-m", "bench.worker", "--job", job.id,
        "--problems", str(cfg.problems_dir), "--results", str(cfg.results_dir),
        "--cache", str(cfg.cache_dir), "--subprocess-timeout", str(job.timeout_seconds),
    ]
    env = dict(os.environ)
    if cpu is not None:
        env["BENCH_CPU"] = str(cpu)
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=max(timeout, 0.0))
        return Launch(proc.returncode, False, time.perf_counter() - t0, out or "")
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        out, _ = proc.communicate()
        return Launch(None, True, time.perf_counter() - t0, out or "")


def _cpu_slots(n: int) -> list[int | None]:
    if hasattr(os, "sched_getaffinity"):
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) >= n:
            return cpus[:n]
    return [None] * n


def _timeout_record(job: Job, fp: dict[str, Any], launch: Launch) -> dict[str, Any]:
    from . import env

    return {
        "schema": 1,
        "job_id": job.id,
        "capability": job.capability,
        "problem": job.problem,
        "engines": job.engines,
        "setting": job.setting,
        "kind": job.kind,
        "outcome": "TIMEOUT",
        "error": f"stopped after {job.timeout_seconds:g} s",
        "key": freshness_key(job, fp),
        "key_parts": key_parts(job, fp),
        "environment": env.environment(),
        "wall_seconds": launch.seconds,
    }


def _run_phase(
    jobs: list[Job],
    workers: int,
    cfg: RunConfig,
    fingerprints: dict[str, dict[str, Any]],
    summary: RunSummary,
    launcher: Launcher,
    clock: Callable[[], float],
    total: int,
) -> None:
    pending = list(jobs)
    slots = _cpu_slots(workers)
    free = list(slots)
    running: dict[Future, tuple[Job, float, int | None]] = {}

    def log(msg: str) -> None:
        if not cfg.quiet:
            print(msg, flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending or running:
            while pending and free:
                if cfg.limit is not None and len(summary.launched) >= cfg.limit:
                    pending.clear()
                    break
                job = pending.pop(0)
                timeout = job.timeout_seconds
                if cfg.until is not None:
                    remaining = cfg.until - clock() - cfg.margin_seconds
                    if remaining < job.est_seconds:
                        summary.deferred.append(job.id)
                        continue
                    timeout = min(timeout, remaining)
                cpu = free.pop(0)
                summary.launched.append(job.id)
                fut = pool.submit(launcher, job, timeout, cpu, cfg)
                running[fut] = (job, timeout, cpu)
            if not running:
                break
            done, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for fut in done:
                job, timeout, cpu = running.pop(fut)
                free.append(cpu)
                summary.done += 1
                n = summary.done
                try:
                    launch = fut.result()
                except Exception as exc:  # noqa: BLE001
                    summary.failed[job.id] = f"launcher error: {exc}"
                    log(f"[{n}/{total}] LAUNCH-ERROR {job.id}: {exc}")
                    continue
                if launch.timed_out:
                    if timeout < job.timeout_seconds:
                        summary.cut.append(job.id)
                        log(f"[{n}/{total}] CUT (deadline) {job.id}")
                    else:
                        write_result(
                            cfg.results_dir, job,
                            _timeout_record(job, fingerprints[job.kind], launch),
                        )
                        summary.outcomes[job.id] = "TIMEOUT"
                        log(f"[{n}/{total}] TIMEOUT {job.id} ({launch.seconds:.1f}s)")
                    continue
                from .results import read_result

                rec = read_result(cfg.results_dir, job)
                if launch.returncode != 0 or rec is None:
                    tail = launch.output.strip().splitlines()[-5:]
                    summary.failed[job.id] = f"worker exit {launch.returncode}: " + " | ".join(tail)
                    log(f"[{n}/{total}] WORKER-FAILED {job.id}: {summary.failed[job.id]}")
                    continue
                summary.outcomes[job.id] = rec.get("outcome", "?")
                log(f"[{n}/{total}] {rec.get('outcome', '?'):<16} {job.id} ({launch.seconds:.1f}s)")


def run(
    jobs: list[Job],
    cfg: RunConfig,
    *,
    fingerprints: dict[str, dict[str, Any]] | None = None,
    launcher: Launcher = subprocess_launcher,
    clock: Callable[[], float] = time.time,
) -> RunSummary:
    if fingerprints is None:
        from .results import fingerprints_now

        fingerprints = fingerprints_now()
    by_id = {j.id: j for j in jobs}
    summary = RunSummary()

    def todo(kind: str) -> list[Job]:
        out = []
        for j in jobs:
            if j.kind != kind:
                continue
            st = status(j, cfg.results_dir, fingerprints, by_id)
            if st.state in ("missing", "stale") or (cfg.force and st.state == "fresh"):
                out.append(j)
            else:
                summary.skipped[j.id] = st.state
        return out

    first = todo(CORRECTNESS)
    total = len(first) + sum(1 for j in jobs if j.kind == TIMING)
    _run_phase(first, cfg.workers, cfg, fingerprints, summary, launcher, clock, total)
    # Timing jobs are judged after the correctness phase: a correctness job that
    # just passed unblocks its timing job in the same run.
    for jid in [j.id for j in jobs if j.kind == TIMING]:
        summary.skipped.pop(jid, None)
    second = todo(TIMING)
    _run_phase(second, cfg.timing_workers, cfg, fingerprints, summary, launcher, clock, total)
    return summary
