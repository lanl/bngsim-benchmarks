"""Run one job in this process and write its result.

    python -m bench.worker --job <id> --problems DIR --results DIR --cache DIR

The runners start one worker process per job, so a crash, a hang or a leak in
one job cannot touch another, and a timeout can always be enforced by killing
the process. A worker that is killed writes nothing; the job simply has no
current result and runs again next time.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

from .env import HARNESS_ENV, SINGLE_THREAD_ENV


def _pin_to_cpu() -> int | None:
    cpu = os.environ.get("BENCH_CPU")
    if cpu is None or not hasattr(os, "sched_setaffinity"):
        return None
    try:
        os.sched_setaffinity(0, {int(cpu)})
        return int(cpu)
    except (OSError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bench.worker")
    ap.add_argument("--job", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--subprocess-timeout", type=float, default=1800.0)
    args = ap.parse_args(argv)

    # Before anything imports numpy or bngsim: one thread per job.
    for k, v in SINGLE_THREAD_ENV.items():
        os.environ.setdefault(k, v)
    os.environ.update(HARNESS_ENV)
    pinned = _pin_to_cpu()
    scratch = Path(tempfile.mkdtemp(prefix="bench-job-"))
    # An empty, node-local compile cache per job, so a cold measurement is cold
    # and no two jobs share compiled code through the filesystem.
    os.environ["BNGSIM_CODEGEN_CACHE_DIR"] = str(scratch / "codegen")

    from . import env, jobs, specs
    from .capabilities import CAPABILITIES
    from .results import freshness_key, key_parts, write_result

    problems = specs.load_all(args.problems)
    all_jobs = {j.id: j for j in jobs.jobs_for(problems)}
    job = all_jobs.get(args.job)
    if job is None:
        print(f"no job {args.job!r} under {args.problems}", file=sys.stderr)
        return 2
    problem = next(p for p in problems if p.id == job.problem)
    fp = env.fingerprint(timing=job.kind == jobs.TIMING)
    ctx = {
        "cache_dir": args.cache,
        "workdir": str(scratch),
        "subprocess_timeout": args.subprocess_timeout,
    }
    started = dt.datetime.now(dt.UTC)
    t0 = time.perf_counter()
    try:
        body = CAPABILITIES[job.capability].run(job, problem, ctx)
    except Exception as exc:  # noqa: BLE001 - every failure becomes a result
        body = {
            "outcome": "EXCEPTION",
            "error": f"{type(exc).__name__}: {exc}"[:1000],
            "traceback": traceback.format_exc()[-4000:],
        }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    record = {
        "schema": 1,
        "job_id": job.id,
        "capability": job.capability,
        "problem": job.problem,
        "engines": job.engines,
        "setting": job.setting,
        "kind": job.kind,
        **body,
        "key": freshness_key(job, fp),
        "key_parts": key_parts(job, fp),
        "environment": env.environment(),
        "pinned_cpu": pinned,
        "started": started.isoformat(),
        "wall_seconds": time.perf_counter() - t0,
    }
    write_result(args.results, job, record)
    print(f"{record['outcome']:<16} {job.id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
