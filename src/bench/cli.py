"""The ``bench`` command.

    bench list [PATTERN...]          problems and the jobs they define
    bench plan [PATTERN...]          where each job stands: fresh, stale, missing, ...
    bench run  [PATTERN...]          run every job without a current result
    bench gather                     assemble results into reports/
    bench fetch-bionetgen            download and check BioNetGen
    bench doctor                     the environment a job would record

PATTERN selects jobs whose id contains it (a problem id, a capability, "timing").
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from . import jobs as jobs_mod
from . import specs

ROOT = Path.cwd()


def _common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--problems", type=Path, default=ROOT / "problems",
                    help="problem specs (default: ./problems)")
    ap.add_argument("--results", type=Path, default=ROOT / "results",
                    help="per-job results (default: ./results)")
    ap.add_argument("--cache", type=Path, default=ROOT / ".cache",
                    help="downloads: models, BioNetGen (default: ./.cache)")


def _load_jobs(args) -> list[jobs_mod.Job]:
    problems = specs.load_all(args.problems)
    return jobs_mod.select(jobs_mod.jobs_for(problems), getattr(args, "patterns", None))


_DURATION_RE = re.compile(r"^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$")


def parse_duration(text: str) -> float:
    """``9h30m``, ``45m``, ``90s``, or a plain number of seconds."""
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        pass
    m = _DURATION_RE.match(text)
    if not m or not any(m.groups()):
        raise argparse.ArgumentTypeError(f"not a duration: {text!r} (e.g. 9h30m, 45m, 90s)")
    h, mi, s = (int(g or 0) for g in m.groups())
    return float(h * 3600 + mi * 60 + s)


def slurm_end_time() -> float:
    """The current Slurm allocation's end time, from squeue."""
    import os

    jid = os.environ.get("SLURM_JOB_ID")
    if not jid:
        raise SystemExit("--until slurm: not inside a Slurm allocation (SLURM_JOB_ID is unset)")
    out = subprocess.run(["squeue", "-h", "-j", jid, "-o", "%e"],
                         capture_output=True, text=True, timeout=30, check=False)
    text = out.stdout.strip()
    try:
        return dt.datetime.fromisoformat(text).timestamp()
    except ValueError:
        raise SystemExit(f"--until slurm: could not read the end time from squeue: {text!r}")


def parse_until(text: str, now: float | None = None) -> float:
    """An ISO 8601 time, ``HH:MM`` (today, or tomorrow if already past), or ``slurm``."""
    now = time.time() if now is None else now
    if text == "slurm":
        return slurm_end_time()
    m = re.match(r"^(\d{1,2}):(\d{2})$", text)
    if m:
        base = dt.datetime.fromtimestamp(now)
        t = base.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0)
        if t.timestamp() <= now:
            t += dt.timedelta(days=1)
        return t.timestamp()
    try:
        return dt.datetime.fromisoformat(text).timestamp()
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a time: {text!r} (ISO 8601, HH:MM, or slurm)")


def cmd_list(args) -> int:
    problems = specs.load_all(args.problems)
    for p in problems:
        size = ", ".join(f"{k} {v}" for k, v in p.size.items())
        print(f"{p.id}  [{p.layer}]  {', '.join(p.capabilities)}" + (f"  ({size})" if size else ""))
    jobs = _load_jobs(args)
    print(f"\n{len(jobs)} jobs:")
    for j in jobs:
        print(f"  {j.id}")
    return 0


def cmd_plan(args) -> int:
    from .results import fingerprints_now, status

    jobs = _load_jobs(args)
    fps = fingerprints_now()
    by_id = {j.id: j for j in jobs}
    sts = [status(j, args.results, fps, by_id) for j in jobs]
    for st in sts:
        extra = f"  {st.outcome}" if st.outcome else ""
        reason = f"  ({st.reason})" if st.reason else ""
        print(f"{st.state:<8} {st.job.id}{extra}{reason}")
    counts = Counter(st.state for st in sts)
    todo = [st for st in sts if st.state in ("missing", "stale", "waiting")]
    est = sum(st.job.est_seconds for st in todo)
    print("\n" + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
          + f"; to run: {len(todo)} jobs, roughly {est / 60:.0f} min serially")
    return 0


def cmd_run(args) -> int:
    from .runner import RunConfig, run

    until = None
    if args.until and args.budget:
        raise SystemExit("give --until or --budget, not both")
    if args.until:
        until = parse_until(args.until)
    elif args.budget:
        until = time.time() + parse_duration(args.budget)
    cfg = RunConfig(
        problems_dir=args.problems.resolve(), results_dir=args.results.resolve(),
        cache_dir=args.cache.resolve(), workers=args.workers,
        timing_workers=args.timing_workers, until=until, margin_seconds=args.margin,
        limit=args.next, force=args.force,
    )
    jobs = _load_jobs(args)
    if until is not None:
        print(f"deadline {dt.datetime.fromtimestamp(until).isoformat(timespec='seconds')}"
              f" (margin {args.margin:g} s)")
    s = run(jobs, cfg)
    counts = Counter(s.outcomes.values())
    print("\nran {} jobs: {}".format(len(s.outcomes), ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "none"))
    if s.deferred:
        print(f"deferred {len(s.deferred)} jobs that would not finish before the deadline")
    if s.cut:
        print(f"stopped {len(s.cut)} jobs at the deadline; they run again next time")
    if s.failed:
        print(f"{len(s.failed)} workers failed without a result:")
        for jid, why in s.failed.items():
            print(f"  {jid}: {why}")
    skipped = Counter(s.skipped.values())
    if skipped:
        print("not run: " + ", ".join(f"{k} {v}" for k, v in sorted(skipped.items())))
    return 1 if s.failed else 0


def cmd_gather(args) -> int:
    from .gather import gather
    from .results import fingerprints_now

    reports = gather(_load_jobs(args), args.results, args.reports, fingerprints_now())
    for name, rows in sorted(reports.items()):
        print(f"{name}: {len(rows)} rows -> {args.reports / (name + '.json')}")
    print(f"summary -> {args.reports / 'summary.md'}")
    return 0


def cmd_fetch_bionetgen(args) -> int:
    from .fetch import fetch_bionetgen

    print(fetch_bionetgen(args.cache))
    return 0


def cmd_doctor(args) -> int:
    from . import env
    from ._vendor import VENDOR_JSON, parity_core_record
    from .fetch import sha256_file

    ok = True
    info = env.environment()
    print(json.dumps(info, indent=2, default=str))
    b, g = info["bngsim"], info["bionetgen"]
    if not b.get("build_commit"):
        ok = False
        print(f"FAIL bngsim: {b.get('error') or 'no build commit'}")
    if g.get("error"):
        ok = False
        print(f"FAIL BioNetGen: {g['error']} (run `pixi run fetch-bionetgen`)")
    elif not g.get("fetched"):
        print("WARN BioNetGen: not the fetched, sha256-checked copy")
    rec = parity_core_record()
    for name, want in rec["files"].items():
        if sha256_file(VENDOR_JSON.parent / "parity_core" / name) != want:
            ok = False
            print(f"FAIL vendored parity_core/{name} differs from VENDOR.json")
    try:
        n = len(specs.load_all(args.problems))
        print(f"OK   {n} problem specs load")
    except specs.SpecError as exc:
        ok = False
        print(f"FAIL problem specs:\n{exc}")
    print("READY" if ok else "NOT READY")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="problems and their jobs")
    _common(p)
    p.add_argument("patterns", nargs="*")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("plan", help="where each job stands")
    _common(p)
    p.add_argument("patterns", nargs="*")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("run", help="run jobs without a current result")
    _common(p)
    p.add_argument("patterns", nargs="*")
    p.add_argument("--workers", type=int, default=1, help="concurrent correctness jobs")
    p.add_argument("--timing-workers", type=int, default=1, help="concurrent timing jobs")
    p.add_argument("--until", help="deadline: ISO 8601 time, HH:MM, or 'slurm'")
    p.add_argument("--budget", help="deadline as a duration from now, e.g. 9h30m")
    p.add_argument("--margin", type=float, default=300.0,
                   help="seconds kept free before the deadline (default 300)")
    p.add_argument("--next", type=int, help="launch at most this many jobs")
    p.add_argument("--force", action="store_true", help="re-run jobs with a current result")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("gather", help="assemble results into reports")
    _common(p)
    p.add_argument("--reports", type=Path, default=ROOT / "reports")
    p.set_defaults(fn=cmd_gather, patterns=None)

    p = sub.add_parser("fetch-bionetgen", help="download and check BioNetGen")
    _common(p)
    p.set_defaults(fn=cmd_fetch_bionetgen)

    p = sub.add_parser("doctor", help="the environment a job would record")
    _common(p)
    p.set_defaults(fn=cmd_doctor)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except specs.SpecError as exc:
        print(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
