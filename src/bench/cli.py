"""The ``bench`` command.

    bench list [PATTERN...]          problems and the jobs they define
    bench plan [PATTERN...]          where each job stands: fresh, stale, missing, ...
    bench run  [PATTERN...]          run every job without a current result
    bench submit [PATTERN...]        the same, as a Slurm job array (see slurm.example.yaml)
    bench rescore [PATTERN...]       re-score stored correctness results with the current verdict
    bench gather                     assemble results into reports/
    bench publish SUITE              a suite's results, scrubbed, into published/ (to commit)
    bench fetch-bionetgen            download and check BioNetGen
    bench fetch-models [PATTERN...]  download and check the pinned model files
    bench fetch-glibc                install a newer glibc loader (old-glibc Linux)
    bench fetch-rulemonkey           download RuleMonkey and build its rm_driver
    bench doctor                     the environment a job would record

PATTERN selects jobs whose id contains it (a problem id, a capability, "timing"),
or names a core suite ("ode-net"), selecting every job of its members.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import signal
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


def _expand_suites(patterns: list[str] | None, problems: list[specs.Problem]) -> list[str]:
    """Patterns with each suite name (``ode-net``) replaced by its members' problems."""
    members = specs.suite_members(problems)
    out: list[str] = []
    for pat in patterns or []:
        if pat in members:
            out.extend(f"/{p.id}/" for p in members[pat])
        else:
            out.append(pat)
    return out


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
    return jobs_mod.select(all_jobs, _expand_suites(getattr(args, "patterns", None), problems))


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
    for name, members in specs.suite_members(problems).items():
        cap, formats = specs.SUITES[name]
        print(f"\nsuite {name} ({cap} on {'/'.join(formats)}):")
        for m in members:
            print(f"  {m.suites[name]:>2}  {m.id}")
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


def cmd_rescore(args) -> int:
    from .jobs import CORRECTNESS
    from .rescore import rescore
    from .results import fingerprints_now

    jobs = [j for j in _load_jobs(args) if j.kind == CORRECTNESS]
    done = rescore(jobs, args.results, fingerprints_now()[CORRECTNESS], force=args.force,
                   dry_run=args.dry_run)
    for r in done:
        if r.state == "rescored":
            change = r.before if r.before == r.after else f"{r.before} -> {r.after}"
            print(f"{'rescored':<16} {r.job.id}  {change}")
        elif r.state != "current" or args.verbose:
            extra = f"  {r.before}" if r.before else ""
            print(f"{r.state:<16} {r.job.id}{extra}" + (f"  ({r.reason})" if r.reason else ""))
    counts = Counter(r.state for r in done)
    changed = sum(1 for r in done if r.state == "rescored" and r.before != r.after)
    tally = ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "nothing selected"
    dry = " (dry run: nothing written)" if args.dry_run else ""
    print(f"\n{tally}; {changed} outcomes changed{dry}")
    return 0


def cmd_gather(args) -> int:
    from .gather import gather
    from .results import fingerprints_now

    reports = gather(_load_jobs(args), args.results, args.reports, fingerprints_now())
    for name, rows in sorted(reports.items()):
        print(f"{name}: {len(rows)} rows -> {args.reports / (name + '.json')}")
    print(f"summary -> {args.reports / 'summary.md'}")
    return 0


def cmd_publish(args) -> int:
    from .publish import PublishError, default_slurm_config, publish
    from .results import fingerprints_now

    problems = specs.load_all(args.problems)
    try:
        dest = publish(args.suite, problems, jobs_mod.jobs_for(problems), args.results,
                       args.out, fingerprints_now(), root=ROOT,
                       slurm_config=default_slurm_config(ROOT))
    except PublishError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"published {args.suite} -> {dest}")
    return 0


def cmd_fetch_bionetgen(args) -> int:
    from .fetch import fetch_bionetgen

    print(fetch_bionetgen(args.cache))
    return 0


def cmd_fetch_glibc(args) -> int:
    from .fetch import fetch_glibc

    print(fetch_glibc(args.cache))
    return 0


def cmd_fetch_rulemonkey(args) -> int:
    from .fetch import fetch_rulemonkey

    print(fetch_rulemonkey(args.cache, download_only=args.download_only, jobs=args.jobs))
    return 0


def _model_sources(problems: list[specs.Problem]) -> dict[tuple[str, str], tuple[dict, list[str]]]:
    """Each distinct cached model file the problems read, keyed as the cache is
    (sha256, file name), with the problems that read it. Same bytes under the
    same name are one entry; under another name they are a second entry, since
    BioNetGen names its outputs after the model file and a job reads that path."""
    from .fetch import model_cache_path

    out: dict[tuple[str, str], tuple[dict, list[str]]] = {}
    for p in problems:
        key = (p.source["sha256"], model_cache_path(p.source, "").name)
        out.setdefault(key, (p.source, []))[1].append(p.id)
    return out


def cmd_fetch_models(args) -> int:
    from .fetch import model_cache_state, model_file

    problems = specs.load_all(args.problems)
    if args.patterns:
        wanted = {j.problem for j in _load_jobs(args)}
        problems = [p for p in problems if p.id in wanted]
    counts: Counter[str] = Counter()
    for (sha, _), (source, ids) in _model_sources(problems).items():
        where = f"{source['repo']}@{source['commit'][:12]}:{source['path']}"
        state = model_cache_state(source, args.cache)
        if state == "cached":
            counts["cached"] += 1
            print(f"{'cached':<9} {sha[:12]}  {where}")
            continue
        try:
            model_file(source, args.cache)
        except Exception as exc:  # report every file, then fail
            counts["failed"] += 1
            print(f"{'FAIL':<9} {sha[:12]}  {where}\n{'':<10}{exc}\n"
                  f"{'':<10}(named by {', '.join(ids)})")
            continue
        counts["fetched"] += 1
        print(f"{'refetched' if state == 'corrupt' else 'fetched':<9} {sha[:12]}  {where}")
    total = sum(counts.values())
    print(f"\n{total} model files for {len(problems)} problems: "
          f"{counts['cached']} already cached, {counts['fetched']} fetched, "
          f"{counts['failed']} failed -> {args.cache / 'models'}")
    return 1 if counts["failed"] else 0


def _report_model_cache(problems: list[specs.Problem], cache_dir: Path) -> None:
    """One doctor line on the model cache. A gap is a warning, not a failure:
    a job downloads its own model on first use (and `bench submit` fetches all
    of them up front), so only a host with no network would be stuck."""
    from .fetch import model_cache_state

    states = Counter(model_cache_state(src, cache_dir)
                     for src, _ in _model_sources(problems).values())
    total = sum(states.values())
    if states["cached"] == total:
        print(f"OK   {total} model files cached and sha256-checked ({cache_dir / 'models'})")
        return
    gaps = [f"{n} {what}" for what, n in (("not cached", states["missing"]),
                                          ("with the wrong sha256", states["corrupt"])) if n]
    print(f"WARN {total} model files: {', '.join(gaps)} "
          "(run `pixi run fetch-models`; otherwise each job downloads its own)")


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


# A trivial BNG .net -- one species, one degradation reaction -- used only to
# exercise bngsim's native core end to end (parse + load) in a child process.
# from_net reads a .net directly, so this needs no BNGPATH, no BNG2.pl and no
# network: it isolates the compiled core itself.
_SELFTEST_NET = """\
# bngsim-benchmarks doctor core self-test network
begin parameters
    1 _rateLaw1  1.0
end parameters
begin species
    1 A() 100
end species
begin reactions
    1 1 0 _rateLaw1 #_R1
end reactions
"""


def _selftest_verdict(returncode: int, stderr: str) -> tuple[bool, list[str]]:
    """Turn a core self-test child's exit into (passed, lines-to-print).

    returncode 0 means from_net loaded the model. A NEGATIVE returncode means the
    child died on a signal N (-N): a native crash with no Python traceback -- the
    failure mode this check exists for. Classically an ABI mismatch: a bngsim core
    built against one libstdc++ run against another (a pre-GCC-9 build of
    std::filesystem::path run against a GCC>=9 libstdc++ corrupts the heap inside
    from_net). A POSITIVE returncode is an ordinary Python exception (e.g. the core
    rejected the net) -- surfaced verbatim, not as an ABI diagnosis.
    """
    if returncode == 0:
        return True, ["OK   bngsim core loads a model (from_net)"]
    if returncode < 0:
        sig = -returncode
        try:
            name = signal.Signals(sig).name
        except ValueError:
            name = f"signal {sig}"
        return False, [
            f"FAIL bngsim core crashed loading a trivial model ({name}) -- almost always a C++",
            "     libstdc++ ABI mismatch between the compiler that built bngsim's core and this",
            "     environment's libstdc++ (e.g. a system GCC<9 build of std::filesystem run",
            "     against a conda-forge GCC>=9 libstdc++). Rebuild bngsim's core with the SAME",
            "     toolchain that provides your runtime libstdc++; see docs/BUILD_NOTES.md.",
        ]
    tail = (stderr.strip().splitlines() or ["(no output)"])[-1][:200]
    return False, [f"FAIL bngsim core could not load the self-test model: {tail}"]


def _check_core_loads_net() -> bool:
    """Whether bngsim's native core actually LOADS a model here, not just imports.

    doctor's build-commit check confirms only that ``import bngsim`` works. But the
    compiled core can import cleanly and then segfault on first real use, leaving no
    traceback and no result -- just ``exit -11`` hours into a timing run on an
    otherwise-wasted exclusive node. We catch it before that: load a trivial .net in
    a CHILD process (so a fatal signal cannot take doctor down) with core dumps
    disabled, and translate the exit into an actionable message.
    """
    import tempfile

    def _no_core() -> None:  # pragma: no cover - POSIX child setup
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    code = "import sys, bngsim; bngsim.Model.from_net(sys.argv[1])"
    with tempfile.TemporaryDirectory() as d:
        net = Path(d) / "selftest.net"
        net.write_text(_SELFTEST_NET, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code, str(net)],
                capture_output=True, text=True, timeout=120, check=False,
                preexec_fn=_no_core,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"FAIL bngsim core: cannot run the load self-test ({type(exc).__name__}: {exc})")
            return False
    passed, lines = _selftest_verdict(proc.returncode, proc.stderr)
    for line in lines:
        print(line)
    return passed


def _check_roadrunner_loads(env) -> bool:
    """Whether libRoadRunner, the SBML reference ODE engine, imports here.

    The environment is pinned to install it (``libroadrunner`` on PyPI, imported
    as ``roadrunner``); a missing or unimportable package is a broken environment,
    so this FAILs rather than warns. The version comes from package metadata
    without the import; the import confirms the wheel actually loads on this host.
    """
    ident = env.roadrunner_identity()
    if not ident.get("version"):
        print(f"FAIL libRoadRunner: {ident.get('error') or 'not installed'} "
              "(pinned in pixi.toml as `libroadrunner`)")
        return False
    try:
        env.import_roadrunner()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL libRoadRunner {ident['version']} does not import: {type(exc).__name__}: {exc}")
        return False
    print(f"OK   libRoadRunner {ident['version']} imports")
    return True


def _check_copasi_loads(env) -> bool:
    """Whether COPASI, the SBML reference ODE engine driven through BASICO, works here.

    The environment is pinned to install it (``copasi-basico`` + ``python-copasi``
    on PyPI, imported as ``basico`` / ``COPASI``); a missing or unimportable
    package is a broken environment, so this FAILs rather than warns. The version
    comes from metadata without the import; the import then confirms the native
    ``python-copasi`` extension actually loads on this host (e.g. the aarch64 wheel
    against the node's glibc).
    """
    ident = env.copasi_identity()
    if not ident.get("version"):
        print(f"FAIL COPASI: {ident.get('error') or 'not installed'} "
              "(pinned in pixi.toml as `copasi-basico` + `python-copasi`)")
        return False
    try:
        import basico  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL COPASI {ident['version']} does not import: {type(exc).__name__}: {exc}")
        return False
    print(f"OK   COPASI {ident['version']} imports (basico {ident.get('basico')})")
    return True


_AMICI_SELFTEST = r'''
import sys, tempfile, numpy as np
import amici
from amici.sim.sundials import run_simulation, SensitivityOrder, LinearMultistepMethod

# A trivial irreversible conversion A -> B, k=1, as SBML. Compiling and running
# this exercises exactly the runtime path AmiciODE uses: sbml2amici shells out to
# $CC/$CXX to build a fresh extension, then CVODES integrates it.
SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model>
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialAmount="1" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialAmount="0" hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters><parameter id="k" value="1" constant="true"/></listOfParameters>
    <listOfReactions>
      <reaction id="r" reversible="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>k</ci><ci>A</ci></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

with tempfile.TemporaryDirectory() as d:
    xml = d + "/m.xml"
    open(xml, "w").write(SBML)
    amici.SbmlImporter(xml).sbml2amici("st", d + "/st", generate_sensitivity_code=False, verbose=False)
    model = amici.import_model_module("st", d + "/st").get_model()
    model.set_timepoints(np.linspace(0.0, 1.0, 3))
    solver = model.create_solver()
    solver.set_sensitivity_order(SensitivityOrder.none)
    solver.set_linear_multistep_method(LinearMultistepMethod.BDF)
    rdata = run_simulation(model, solver)
    assert rdata.status == 0, f"status {rdata.status}"
    x = np.asarray(rdata.x)
    assert x.shape == (3, 2), f"shape {x.shape}"
    assert np.all(np.isfinite(x)), "non-finite output"
    # A decays, B grows; a smoke that the integration actually ran.
    assert x[-1, 0] < x[0, 0] and x[-1, 1] > x[0, 1], "trajectory did not evolve"
sys.exit(0)
'''


def _check_amici_loads(env) -> bool:
    """Whether AMICI, the CVODES reference engine, compiles AND runs a model here.

    AMICI ships an sdist only, so it is built from source on install, and every
    model it runs compiles a fresh C++ extension via ``sbml2amici`` using
    ``$CC``/``$CXX``. That per-model compile is the ABI-risky step -- a mismatch
    between the compiler and this env's libstdc++ is what segfaulted bngsim's core
    on ARM -- so a bare ``import amici`` is not enough: doctor must exercise the
    compile+run, as ``_check_core_loads_net`` does for bngsim. It runs in a CHILD
    process (a fatal signal cannot take doctor down) and translates a crash into an
    actionable ABI message. FAILs (not warns): the environment is pinned to install
    AMICI, so an absent or broken one is a broken environment.
    """
    ident = env.amici_identity()
    if not ident.get("version"):
        print(f"FAIL AMICI: {ident.get('error') or 'not installed'} "
              "(pinned in pixi.toml as `amici`)")
        return False

    def _no_core() -> None:  # pragma: no cover - POSIX child setup
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    try:
        proc = subprocess.run(
            [sys.executable, "-c", _AMICI_SELFTEST],
            capture_output=True, text=True, timeout=300, check=False,
            preexec_fn=_no_core,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"FAIL AMICI {ident['version']}: cannot run the compile self-test "
              f"({type(exc).__name__}: {exc})")
        return False
    if proc.returncode == 0:
        print(f"OK   AMICI {ident['version']} compiles and runs a model")
        return True
    if proc.returncode < 0:
        sig = -proc.returncode
        try:
            name = signal.Signals(sig).name
        except ValueError:
            name = f"signal {sig}"
        print(f"FAIL AMICI {ident['version']} crashed compiling/running a model ({name}) -- almost")
        print("     always a C++ libstdc++ ABI mismatch between $CC/$CXX and this environment's")
        print("     libstdc++. Set CC/CXX to the conda-forge compiler; see docs/BUILD_NOTES.md.")
        return False
    tail = (proc.stderr.strip().splitlines() or ["(no output)"])[-1][:200]
    print(f"FAIL AMICI {ident['version']} could not compile/run the self-test model: {tail}")
    return False


_LOAD_FAILURES = (
    "error while loading shared libraries",
    "not found (required by",
    "symbol lookup error",
    "undefined symbol",
)


def _check_nfsim_loads(env) -> bool:
    """Whether the standalone NFsim (bundled with BioNetGen) launches here, under
    the same loader as run_network, and which version it reports.

    NFsim with no arguments prints its banner (``starting NFsim v1.14.3...``) and
    exits. The network-free capability runs it through BNG2.pl, whose ``bin/NFsim``
    the harness wraps with this launch prefix (``env.bng_launch_tree``).
    """
    ident = env.nfsim_identity()
    if not ident.get("path"):
        print(f"FAIL NFsim: {ident.get('error')}")
        return False
    try:
        proc = subprocess.run([*env.run_network_launch(), ident["path"]], capture_output=True,
                              text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"FAIL NFsim: cannot launch ({type(exc).__name__}: {exc})")
        return False
    out = proc.stdout + proc.stderr
    m = re.search(r"starting NFsim v([0-9][0-9.]*[0-9])", out)
    if m and not any(s in out for s in _LOAD_FAILURES):
        want = ident.get("version")
        if want and m.group(1) != want:
            print(f"FAIL NFsim reports v{m.group(1)}, but {ident.get('bundled_with')} "
                  f"should bundle v{want}")
            return False
        print(f"OK   NFsim {m.group(1)} loads ({ident['path']})")
        return True
    detail = out.strip().splitlines()
    print(f"FAIL NFsim does not load: {detail[-1][:200] if detail else proc.returncode}")
    return False


def _check_rulemonkey_runs(env) -> bool:
    """Whether the standalone RuleMonkey driver is built and runs here.

    It is not part of the pinned environment (it is built from source by `bench
    fetch-rulemonkey`), so a missing driver is a warning -- only the network-free
    capability needs it, and records its absence as a reference failure. A driver
    that is present but does not run is a failure.
    """
    ident = env.rulemonkey_identity()
    if not ident.get("path"):
        print(f"WARN RuleMonkey: {ident.get('error')}")
        return True
    try:
        proc = subprocess.run([ident["path"]], capture_output=True, text=True, timeout=30,
                              check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"FAIL RuleMonkey: cannot launch {ident['path']} ({type(exc).__name__}: {exc})")
        return False
    if "Usage: rm_driver" not in proc.stdout + proc.stderr:
        detail = (proc.stdout + proc.stderr).strip().splitlines()
        print(f"FAIL RuleMonkey rm_driver does not run: {detail[-1][:200] if detail else ''}")
        return False
    if not ident.get("built"):
        print(f"WARN RuleMonkey: {ident['path']} runs but has no matching build marker "
              "(not the pinned build of `bench fetch-rulemonkey`)")
        return True
    print(f"OK   RuleMonkey {ident.get('version')} rm_driver runs ({ident.get('compiler')}; "
          f"{ident['path']})")
    return True


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
    else:
        # The core imported (it stamped a commit); confirm it also LOADS a model.
        # A core that imports but segfaults in from_net otherwise sails past doctor
        # and dies only mid-run (see docs/BUILD_NOTES.md, ABI mismatch).
        ok = _check_core_loads_net() and ok
    if g.get("error"):
        ok = False
        print(f"FAIL BioNetGen: {g['error']} (run `pixi run fetch-bionetgen`)")
    elif not g.get("fetched"):
        print("WARN BioNetGen: not the fetched, sha256-checked copy")
    if not g.get("error"):
        ok = _check_run_network_loads(env) and ok
    ok = _check_roadrunner_loads(env) and ok
    ok = _check_copasi_loads(env) and ok
    ok = _check_amici_loads(env) and ok
    if not g.get("error"):
        ok = _check_nfsim_loads(env) and ok
    ok = _check_rulemonkey_runs(env) and ok
    rec = parity_core_record()
    for name, want in rec["files"].items():
        if sha256_file(VENDOR_JSON.parent / "parity_core" / name) != want:
            ok = False
            print(f"FAIL vendored parity_core/{name} differs from VENDOR.json")
    try:
        problems = specs.load_all(args.problems)
        print(f"OK   {len(problems)} problem specs load")
        _report_model_cache(problems, args.cache)
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

    p = sub.add_parser("rescore",
                       help="re-score stored correctness results with the current verdict")
    _common(p)
    p.add_argument("patterns", nargs="*")
    p.add_argument("--force", action="store_true",
                   help="re-score current results too (they are rewritten as they are)")
    p.add_argument("--dry-run", action="store_true", help="report, but write nothing")
    p.add_argument("--verbose", action="store_true", help="also list the current results")
    p.set_defaults(fn=cmd_rescore)

    p = sub.add_parser("gather", help="assemble results into reports")
    _common(p)
    p.add_argument("--reports", type=Path, default=ROOT / "reports")
    p.set_defaults(fn=cmd_gather, patterns=None)

    p = sub.add_parser("publish",
                       help="write a suite's results, scrubbed, to published/ for committing")
    _common(p)
    p.add_argument("suite", help="a suite name, e.g. ode-net")
    p.add_argument("--out", type=Path, default=ROOT / "published")
    p.set_defaults(fn=cmd_publish, patterns=None)

    p = sub.add_parser("fetch-bionetgen", help="download and check BioNetGen")
    _common(p)
    p.set_defaults(fn=cmd_fetch_bionetgen)

    p = sub.add_parser("fetch-models",
                       help="download and check every model file the specs pin")
    _common(p)
    p.add_argument("patterns", nargs="*",
                   help="only the problems with a job matching one of these")
    p.set_defaults(fn=cmd_fetch_models)

    p = sub.add_parser("fetch-glibc",
                       help="install a newer glibc loader (old-glibc Linux hosts only)")
    _common(p)
    p.set_defaults(fn=cmd_fetch_glibc)

    p = sub.add_parser("fetch-rulemonkey",
                       help="download RuleMonkey's pinned release and build rm_driver")
    _common(p)
    p.add_argument("--download-only", action="store_true",
                   help="only download and check the source (on a node with the network)")
    p.add_argument("--jobs", type=int, help="parallel compile jobs (default: every CPU)")
    p.set_defaults(fn=cmd_fetch_rulemonkey)

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
