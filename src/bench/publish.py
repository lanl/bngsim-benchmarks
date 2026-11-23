"""Publish one suite's results as a machine-readable snapshot, to be committed.

    bench publish SUITE [--out published]

writes ``published/<suite>/<label>/``, where ``label`` names the BNGsim build the
results were measured on (``bngsim-<version>-<build commit>``):

    manifest.json        what was measured, on what, with which engines and harness,
                         and the suite's members with their provenance
    correctness.csv      one row per (problem, reference engine)
    timing.csv           one row per (problem, engine)
    jobs/<job id>.json   every job's full result record

A snapshot is all or nothing. Every job of the suite must have a current result
(a timing judged on the node type it was measured on), so a published set never
mixes BNGsim builds or misses a model. Records are scrubbed of where they ran --
host names, scheduler job ids, local paths -- and publishing refuses outright if
any of that survives the scrub, rather than leak it. The column meanings are in
docs/published_format.md.
"""

from __future__ import annotations

import csv
import datetime as dt
import getpass
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from . import specs
from .gather import _errored_engines, _presim, _presim_total, _why, reference_agreement
from .jobs import CORRECTNESS, TIMING, Job
from .results import freshness_key, read_result, result_path

SCHEMA = 1

# Fields that say where a job ran, not what it measured: dropped before publishing.
_DROP = (
    ("environment", "host", "hostname"),
    ("environment", "host", "slurm_job_id"),
    ("environment", "bionetgen", "bngpath"),
)

CORRECTNESS_COLUMNS = [
    "suite", "position", "problem", "species", "reactions", "outcome", "agreement",
    "engine", "status", "passed", "max_err_over_peak", "max_rel", "n_cells",
    "cold_seconds", "error",
]
TIMING_COLUMNS = [
    "suite", "position", "problem", "species", "reactions", "outcome", "engine", "role",
    "status", "presim_seconds", "netgen_seconds", "codegen_seconds", "load_seconds",
    "backend", "sim_median_seconds", "sim_ci_low_seconds", "sim_ci_high_seconds",
    "rounds", "reproducible", "error",
]


# The status of a reference that does not apply to the model: it was not run, and
# its row says why (the record's ``not_applicable``).
NOT_APPLICABLE = "not_applicable"


class PublishError(RuntimeError):
    """The suite cannot be published as asked; the message says why."""


# --------------------------------------------------------------------------- #
# Scrubbing
# --------------------------------------------------------------------------- #
def _replacements(root: Path) -> list[tuple[re.Pattern[str], str]]:
    """Local path prefixes and what each becomes, longest first."""
    user = re.escape(getpass.getuser())
    subs = [
        (re.compile(re.escape(str(root.resolve()))), "."),
        (re.compile(re.escape(str(Path.home().resolve()))), "~"),
        (re.compile(re.escape(str(Path.home()))), "~"),
        (re.compile(rf"(?:/ram)?/tmp/{user}(?=/|$)"), "$TMPDIR"),
        (re.compile(r"(?:/ram)?/tmp/bench-job-[A-Za-z0-9_]+"), "$TMPDIR/bench-job"),
    ]
    return subs


def _sweep(obj: Any, subs: list[tuple[re.Pattern[str], str]]) -> Any:
    if isinstance(obj, str):
        for pat, rep in subs:
            obj = pat.sub(rep, obj)
        return obj
    if isinstance(obj, dict):
        return {k: _sweep(v, subs) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sweep(v, subs) for v in obj]
    return obj


def scrub(rec: dict[str, Any], root: Path) -> dict[str, Any]:
    """A copy of ``rec`` without the fields that say where it ran, and with local
    paths rewritten relative to the repository (``.``), home (``~``) or the job's
    scratch directory (``$TMPDIR``)."""
    rec = json.loads(json.dumps(rec, default=str))
    for path in _DROP:
        d: Any = rec
        for k in path[:-1]:
            d = d.get(k) if isinstance(d, dict) else None
        if isinstance(d, dict):
            d.pop(path[-1], None)
    return _sweep(rec, _replacements(root))


def forbidden_tokens(raw_records: list[dict[str, Any]], slurm_config: Path | None) -> list[str]:
    """What must never appear in a published file: the hosts and scheduler job ids
    the raw records name, this user, and the local scheduler settings."""
    tokens: set[str] = {getpass.getuser(), str(Path.home())}
    for rec in raw_records:
        host = (rec.get("environment") or {}).get("host") or {}
        for k in ("hostname", "slurm_job_id"):
            if host.get(k):
                tokens.add(str(host[k]))
    if slurm_config is not None and slurm_config.exists():
        import yaml

        data = yaml.safe_load(slurm_config.read_text(encoding="utf-8")) or {}
        for k in ("account", "partition", "qos", "constraint"):
            if data.get(k):
                tokens.add(str(data[k]))
    return sorted(t for t in tokens if t)


def leaks(obj: Any, tokens: list[str], where: str = "") -> list[str]:
    """Every place in ``obj`` (keys and string values) where a token appears as a
    whole word; numbers are not searched."""
    found: list[str] = []
    pats = [(t, re.compile(rf"(?<![A-Za-z0-9_]){re.escape(t)}(?![A-Za-z0-9_])")) for t in tokens]
    if isinstance(obj, str):
        found += [f"{where}: {t!r}" for t, p in pats if p.search(obj)]
    elif isinstance(obj, dict):
        for k, v in obj.items():
            found += leaks(k, tokens, f"{where}.{k} (key)")
            found += leaks(v, tokens, f"{where}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found += leaks(v, tokens, f"{where}[{i}]")
    return found


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #
def _size(rec: dict[str, Any], problem: specs.Problem) -> tuple[Any, Any]:
    net = (rec.get("measurements") or {}).get("network") or problem.size or {}
    return net.get("species"), net.get("reactions")


def _not_applicable(rec: dict[str, Any], *, besides: set[str]) -> dict[str, str]:
    """References that do not apply to the model (run_network on an SBML model with
    no faithful network), with the reason, that have no other row."""
    return {name: str(reason) for name, reason in sorted((rec.get("not_applicable") or {}).items())
            if name not in besides}


def correctness_rows(suite: str, problem: specs.Problem, rec: dict[str, Any]) -> list[dict]:
    """One row per reference engine: its verdict, or why it has none -- including a
    reference that does not apply to the model, so no engine is left out."""
    verdicts = rec.get("verdicts") or {}
    refs = (rec.get("measurements") or {}).get("references") or {}
    species, reactions = _size(rec, problem)
    base = {"suite": suite, "position": problem.suites[suite], "problem": problem.id,
            "species": species, "reactions": reactions, "outcome": rec.get("outcome"),
            "agreement": reference_agreement(rec)}
    rows = []
    for engine in sorted(set(verdicts) | _errored_engines(rec)):
        v = verdicts.get(engine)
        if v is not None:
            status = "ok" if v.get("passed") else "DIFF"
        else:
            status = _why(rec, engine)
        rows.append({**base, "engine": engine, "status": status,
                     "passed": None if v is None else bool(v.get("passed")),
                     "max_err_over_peak": (v or {}).get("max_err_over_peak"),
                     "max_rel": (v or {}).get("max_rel"),
                     "n_cells": (v or {}).get("n_cells"),
                     "cold_seconds": (refs.get(engine) or {}).get("cold_seconds"),
                     "error": rec.get(f"{engine}_error")})
    for engine, reason in _not_applicable(rec, besides={r["engine"] for r in rows}).items():
        rows.append({**base, "engine": engine, "status": NOT_APPLICABLE, "error": reason})
    if not rows:  # the job failed before any engine ran
        rows.append({**base, "engine": None, "status": rec.get("outcome"),
                     "error": rec.get("error") or rec.get("bngsim_error")})
    return rows


def timing_rows(suite: str, problem: specs.Problem, rec: dict[str, Any]) -> list[dict]:
    """One row per engine: its one-time pre-sim cost and per-solve time, or why it
    was not timed."""
    m = rec.get("measurements") or {}
    times = m.get("times") or {}
    presim = _presim(m)
    repro = m.get("reproducibility") or {}
    rounds = (m.get("repetitions") or {}).get("rounds")
    species, reactions = _size(rec, problem)
    base = {"suite": suite, "position": problem.suites[suite], "problem": problem.id,
            "species": species, "reactions": reactions, "outcome": rec.get("outcome")}
    rows = []
    for engine, t in times.items():
        p = presim.get(engine) or {}
        rows.append({**base, "engine": engine, "role": p.get("role"), "status": "timed",
                     "presim_seconds": _presim_total(p),
                     "netgen_seconds": p.get("netgen_seconds"),
                     "codegen_seconds": p.get("codegen_seconds"),
                     "load_seconds": p.get("load_seconds"), "backend": p.get("backend"),
                     "sim_median_seconds": t.get("median_seconds"),
                     "sim_ci_low_seconds": t.get("ci_low"),
                     "sim_ci_high_seconds": t.get("ci_high"), "rounds": rounds,
                     "reproducible": (repro.get(engine) or {}).get("passed"),
                     "error": None})
    for engine in sorted(_errored_engines(rec) - set(times)):
        rows.append({**base, "engine": engine, "status": _why(rec, engine),
                     "error": rec.get(f"{engine}_error")})
    for engine, reason in _not_applicable(rec, besides={r["engine"] for r in rows}).items():
        rows.append({**base, "engine": engine, "status": NOT_APPLICABLE, "error": reason})
    if not rows:
        rows.append({**base, "engine": None, "status": rec.get("outcome"),
                     "error": rec.get("error") or rec.get("bngsim_error")})
    return rows


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({c: ("" if row.get(c) is None else row.get(c)) for c in columns})


# --------------------------------------------------------------------------- #
# Publish
# --------------------------------------------------------------------------- #
def _distinct(records: list[dict[str, Any]], *path: str) -> list[Any]:
    out: list[Any] = []
    for rec in records:
        d: Any = rec
        for k in path:
            d = d.get(k) if isinstance(d, dict) else None
        if d is not None and d not in out:
            out.append(d)
    return out


def publish(
    suite: str,
    problems: list[specs.Problem],
    jobs: list[Job],
    results_dir: Path,
    out_dir: Path,
    fingerprints: dict[str, dict[str, Any]],
    *,
    root: Path,
    slurm_config: Path | None = None,
    now: dt.datetime | None = None,
) -> Path:
    """Write the snapshot for ``suite`` and return its directory."""
    members = specs.suite_members(problems).get(suite)
    if not members:
        raise PublishError(f"no suite {suite!r}; known: {', '.join(specs.suite_members(problems))}")
    member_ids = {p.id for p in members}
    suite_jobs = [j for j in jobs if j.problem in member_ids]
    by_problem = {p.id: p for p in members}

    raw: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for job in suite_jobs:
        rec = read_result(results_dir, job)
        if rec is None:
            missing.append(f"{job.id}: no result")
        else:
            raw[job.id] = rec
    # A timing is current on the node type it was measured on: judge it there.
    node_types = sorted({r.get("environment", {}).get("node_type") for j, r in raw.items()
                         if r.get("kind") == TIMING} - {None})
    if len(node_types) > 1:
        raise PublishError("timing results come from more than one node type: "
                           + "; ".join(node_types))
    fps = dict(fingerprints)
    if node_types:
        fps[TIMING] = {**fps[TIMING], "node_type": node_types[0]}
    for job in suite_jobs:
        rec = raw.get(job.id)
        if rec is not None and rec.get("key") != freshness_key(job, fps[job.kind]):
            missing.append(f"{job.id}: result is stale")
    if missing:
        raise PublishError(f"suite {suite!r} is not complete and current:\n  "
                           + "\n  ".join(missing))

    records = list(raw.values())
    builds = _distinct(records, "environment", "bngsim")
    bngsim = {"version": builds[0].get("version"), "build_commit": builds[0].get("build_commit")}
    label = f"bngsim-{bngsim['version']}-{(bngsim['build_commit'] or 'unknown')[:12]}"
    dest = out_dir / suite / label

    tokens = forbidden_tokens(records, slurm_config)
    scrubbed = {jid: scrub(rec, root) for jid, rec in raw.items()}
    found: list[str] = []
    for jid, rec in scrubbed.items():
        found += leaks(rec, tokens, jid)

    corr_rows: list[dict] = []
    time_rows: list[dict] = []
    job_entries = []
    for job in sorted(suite_jobs, key=lambda j: (by_problem[j.problem].suites[suite], j.kind)):
        rec = scrubbed[job.id]
        p = by_problem[job.problem]
        if job.kind == CORRECTNESS:
            corr_rows += correctness_rows(suite, p, rec)
        else:
            time_rows += timing_rows(suite, p, rec)
        job_entries.append({"job_id": job.id, "problem": job.problem, "kind": job.kind,
                            "outcome": rec.get("outcome"), "file": f"jobs/{job.id}.json"})

    def engine_versions() -> dict[str, list[Any]]:
        return {name: _distinct(records, "environment", name, "version")
                for name in ("bionetgen", "roadrunner", "copasi", "amici")}

    manifest = {
        "schema": SCHEMA,
        "suite": suite,
        "capabilities": sorted({j.capability for j in suite_jobs}),
        "published": (now or dt.datetime.now(dt.UTC)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bngsim": bngsim,
        "engines": engine_versions(),
        "node_type": node_types[0] if node_types else None,
        "harness": {
            "bench": _distinct(records, "environment", "harness", "bench"),
            "parity_core": _distinct(records, "environment", "harness", "parity_core"),
            "source_commits": _distinct(records, "environment", "harness_source", "commit"),
        },
        "members": [{
            "position": p.suites[suite], "problem": p.id, "title": p.title,
            "source": p.source, "publications": list(p.publications),
            "verification": p.verification, "size": p.size,
        } for p in members],
        "jobs": job_entries,
        "files": {"correctness": "correctness.csv", "timing": "timing.csv",
                  "columns": "docs/published_format.md"},
    }
    found += leaks(_sweep(manifest, _replacements(root)), tokens, "manifest")
    if found:
        raise PublishError("refusing to publish: local details survived the scrub:\n  "
                           + "\n  ".join(found[:40]))

    if dest.exists():
        shutil.rmtree(dest)
    (dest / "jobs").mkdir(parents=True)
    for job in suite_jobs:
        path = result_path(dest / "jobs", job)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(scrubbed[job.id], indent=1, sort_keys=True) + "\n",
                        encoding="utf-8")
    _write_csv(dest / "correctness.csv", CORRECTNESS_COLUMNS, corr_rows)
    _write_csv(dest / "timing.csv", TIMING_COLUMNS, time_rows)
    (dest / "manifest.json").write_text(
        json.dumps(_sweep(manifest, _replacements(root)), indent=1) + "\n", encoding="utf-8")
    return dest


def default_slurm_config(root: Path) -> Path | None:
    p = Path(os.environ.get("BENCH_SLURM_CONFIG", root / "slurm.yaml"))
    return p if p.exists() else None
