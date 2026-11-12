"""The ``bench`` command.

    bench list [PATTERN...]          problems and the jobs they define
    bench plan [PATTERN...]          where each job stands: fresh, stale, missing, ...
    bench run  [PATTERN...]          run every job without a current result
    bench submit [PATTERN...]        the same, as a Slurm job array (see slurm.example.yaml)
    bench gather                     assemble results into reports/
    bench fetch-bionetgen            download and check BioNetGen
    bench fetch-glibc                install a newer glibc loader (old-glibc Linux)
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
    all_jobs = jobs_mod.jobs_for(problems)
    jobs_from = getattr(args, "jobs_from", None)
    if jobs_from is not None:
        if args.patterns:
            raise SystemExit("give job patterns or --jobs-from, not both")
        ids = [ln.strip() for ln in jobs_from.read_text(encoding="utf-8").splitlines()]
        ids = [i for i in ids if i and not i.startswith("#")]
        try:
            return jobs_mod.select_exact(all_jobs, ids)
        except ValueError as exc:
            raise SystemExit(f"{jobs_from}: {exc}") from exc
    return jobs_mod.select(all_jobs, getattr(args, "patterns", None))


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


def cmd_submit(args) -> int:
    from . import slurm
    from .results import fingerprints_now

    try:
        cfg = slurm.load_config(args.config)
    except slurm.ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    problems = specs.load_all(args.problems)
    jobs = _load_jobs(args)
    try:
        sub = slurm.submit(
            jobs, problems, cfg, root=ROOT, problems_dir=args.problems,
            results_dir=args.results, cache_dir=args.cache, out_dir=args.out,
            fingerprints=fingerprints_now(), dry_run=args.dry_run, skip_long=args.skip_long,
        )
    except slurm.SubmitError as exc:
        print(f"bench submit: {exc}", file=sys.stderr)
        return 2
    p = sub.packing
    if not p.tasks:
        print("nothing to run: every selected job has a current result"
              + (f" ({len(p.long)} too long for one allocation, skipped)" if p.long else ""))
        return 0
    n_jobs = sum(len(t.jobs) for t in p.tasks)
    longest = max(t.bound(cfg) for t in p.tasks)
    n = len(p.tasks)
    print(f"{n_jobs} jobs in {n} array task{'s' if n != 1 else ''} of {cfg.cpus_per_task} CPUs; "
          f"longest estimate {longest / 3600:.2f} h of a {cfg.time_limit_seconds / 3600:.2f} h "
          "time limit")
    if p.long:
        print(f"skipped {len(p.long)} problems too long for one allocation (listed in plan.json)")
    print(f"wrote {sub.directory}")
    if sub.job_id is None:
        print(f"dry run: submit it with  sbatch {sub.directory / 'submit.sbatch'}")
    else:
        print(f"submitted Slurm job {sub.job_id}; logs in {sub.directory / 'logs'}")
    return 0


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


def cmd_fetch_glibc(args) -> int:
    from .fetch import fetch_glibc

    print(fetch_glibc(args.cache))
    return 0


def _check_run_network_loads(env) -> bool:
    """Whether run_network actually launches here, and under which loader.

    run_network with no args prints its version and exits 1; the point is only
    that it *loads*. A machine whose glibc is too old fails with a GLIBC error
    unless a newer loader was fetched (`fetch-glibc`).
    """
    launch = env.run_network_launch()
    loader = env.glibc_identity()
    try:
        proc = subprocess.run([*launch, str(env.run_network_bin())],
                              capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"FAIL run_network: cannot launch ({type(exc).__name__}: {exc})")
        return False
    out = (proc.stdout + proc.stderr)
    # The dynamic loader's failure phrasings: a shared object it cannot open, a
    # symbol version the host libc lacks (the GLIBC_2.34 case this swap fixes), or
    # a missing symbol. Any of them means run_network did not actually load -- key
    # off the failure, not off the absence of "GLIBC" (a libstdc++ "cannot open
    # shared object" failure carries no "GLIBC" and must not read as OK).
    load_failed = any(s in out for s in (
        "error while loading shared libraries",
        "not found (required by",
        "symbol lookup error",
        "undefined symbol",
    ))
    if "run_network" in out and not load_failed:
        where = (f"under glibc {loader.get('glibc_version')} ({loader.get('tag')})"
                 if loader.get("swapped") else "with the host loader")
        print(f"OK   run_network loads {where}")
        return True
    detail = out.strip().splitlines()
    msg = detail[-1][:200] if detail else f"exit {proc.returncode}"
    hint = "" if loader.get("swapped") else " (run `pixi run fetch-glibc`)"
    print(f"FAIL run_network does not load: {msg}{hint}")
    return False


def _encoding_is_utf8(enc: str) -> bool:
    """Whether ``enc`` names UTF-8, however it is spelled (``UTF-8``, ``utf8``)."""
    try:
        import codecs

        return codecs.lookup(enc).name == "utf-8"
    except (LookupError, TypeError):
        return False


def cmd_doctor(args) -> int:
    import locale

    from . import env
    from ._vendor import VENDOR_JSON, parity_core_record
    from .fetch import sha256_file

    ok = True
    info = env.environment()
    print(json.dumps(info, indent=2, default=str))
    enc = locale.getpreferredencoding(False)
    if _encoding_is_utf8(enc):
        print(f"OK   text files are read as {enc}")
    else:
        ok = False
        print(f"FAIL text files are read as {enc}, not UTF-8 "
              "(set PYTHONUTF8=1; a model with a non-ASCII byte would fail to load)")
    b, g = info["bngsim"], info["bionetgen"]
    if not b.get("build_commit"):
        ok = False
        print(f"FAIL bngsim: {b.get('error') or 'no build commit'}")
    if g.get("error"):
        ok = False
        print(f"FAIL BioNetGen: {g['error']} (run `pixi run fetch-bionetgen`)")
    elif not g.get("fetched"):
        print("WARN BioNetGen: not the fetched, sha256-checked copy")
    if not g.get("error"):
        ok = _check_run_network_loads(env) and ok
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
    p.add_argument("--jobs-from", type=Path,
                   help="run exactly the job ids in this file, one per line (as `bench submit` writes)")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("submit", help="submit jobs as a Slurm job array")
    _common(p)
    p.add_argument("patterns", nargs="*")
    p.add_argument("--config", type=Path, default=ROOT / "slurm.yaml",
                   help="cluster settings (default: ./slurm.yaml; see slurm.example.yaml)")
    p.add_argument("--out", type=Path, default=ROOT / "slurm",
                   help="where submission directories go (default: ./slurm)")
    p.add_argument("--dry-run", action="store_true",
                   help="write the submission directory but do not call sbatch")
    p.add_argument("--skip-long", action="store_true",
                   help="submit the rest when some problems would not fit one allocation")
    p.set_defaults(fn=cmd_submit)

    p = sub.add_parser("gather", help="assemble results into reports")
    _common(p)
    p.add_argument("--reports", type=Path, default=ROOT / "reports")
    p.set_defaults(fn=cmd_gather, patterns=None)

    p = sub.add_parser("fetch-bionetgen", help="download and check BioNetGen")
    _common(p)
    p.set_defaults(fn=cmd_fetch_bionetgen)

    p = sub.add_parser("fetch-glibc",
                       help="install a newer glibc loader (old-glibc Linux hosts only)")
    _common(p)
    p.set_defaults(fn=cmd_fetch_glibc)

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
