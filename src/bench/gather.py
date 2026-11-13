"""Assemble per-job results into one report per capability and kind.

    reports/<capability>-<kind>.json   every row, with what it was measured on
    reports/summary.md                 a readable table of the same

A row records whether its result is current (its key matches this environment),
so a report never silently mixes builds: a stale row is marked as such.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .jobs import CORRECTNESS, TIMING, Job
from .results import freshness_key, read_result


def _row(job: Job, rec: dict[str, Any], current: bool) -> dict[str, Any]:
    m = rec.get("measurements", {})
    env = rec.get("environment", {})
    row: dict[str, Any] = {
        "job_id": job.id,
        "problem": job.problem,
        "setting": job.setting,
        "outcome": rec.get("outcome"),
        "current": current,
        "bngsim": (env.get("bngsim") or {}).get("build_commit"),
        "node_type": env.get("node_type"),
        "network": m.get("network"),
    }
    if job.kind == CORRECTNESS:
        # One verdict of BNGsim against each reference engine. The table shows the
        # worst agreement across references (the hardest check the row passed) and
        # keeps the per-engine numbers for the JSON.
        verdicts = rec.get("verdicts") or {}
        row["verdicts"] = {name: {"max_err_over_peak": v.get("max_err_over_peak"),
                                  "max_rel": v.get("max_rel"), "n_cells": v.get("n_cells"),
                                  "passed": v.get("passed")}
                           for name, v in verdicts.items()}
        peaks = [v.get("max_err_over_peak") for v in verdicts.values()
                 if v.get("max_err_over_peak") is not None]
        row["max_err_over_peak"] = max(peaks) if peaks else None
    else:
        # Absolute time per engine (bngsim + each reference), with its bootstrap CI
        # and whether that engine reproduced its first run. A speedup is derived in
        # rendering against a baseline, never stored here (docs/DESIGN.md 6.3).
        repro = m.get("reproducibility") or {}
        row.update(
            times=m.get("times") or {},
            reproducible={name: (r or {}).get("passed") for name, r in repro.items()},
            rounds=(m.get("repetitions") or {}).get("rounds"),
            bngsim_cold_seconds=m.get("bngsim_cold_seconds"),
            reference_solver=((m.get("references") or {}).get("run_network") or {}).get("solver"),
            bngsim_config={k: (m.get("bngsim") or {}).get(k)
                           for k in ("codegen_backend", "linear_solver", "jacobian")},
        )
    # Any engine's failure is recorded under "<name>_error"; pass every one through,
    # plus the generic "error" of a job that failed before any engine ran.
    for k, val in rec.items():
        if k == "error" or k.endswith("_error"):
            if val:
                row[k] = val
    return row


def gather(
    jobs: list[Job],
    results_dir: str | Path,
    reports_dir: str | Path,
    fingerprints: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    reports: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        rec = read_result(results_dir, job)
        if rec is None:
            continue
        current = rec.get("key") == freshness_key(job, fingerprints[job.kind])
        reports[f"{job.capability}-{job.kind}"].append(_row(job, rec, current))
    out = Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in reports.items():
        (out / f"{name}.json").write_text(json.dumps(rows, indent=2, default=str) + "\n",
                                          encoding="utf-8")
    (out / "summary.md").write_text(render(reports), encoding="utf-8")
    return dict(reports)


def _fmt(x: Any, spec: str) -> str:
    return "" if x is None else format(x, spec)


# The derived-speedup baseline: the reference every other engine's time is divided
# against. run_network is the trusted dense oracle and the historical baseline, so
# a reader sees "how much faster than run_network" without a stored ratio (6.3).
SPEEDUP_BASELINE = "run_network"


def _engine_order(rows: list[dict[str, Any]]) -> list[str]:
    """Every engine seen across the timing rows, BNGsim first then references in
    first-seen order, so the per-engine table has a stable column set."""
    order: list[str] = ["bngsim"]
    for r in rows:
        for name in (r.get("times") or {}):
            if name not in order:
                order.append(name)
    return order


def render(reports: dict[str, list[dict[str, Any]]]) -> str:
    lines = ["# Results summary", ""]
    for name in sorted(reports):
        rows = reports[name]
        counts = Counter(r["outcome"] for r in rows)
        stale = sum(1 for r in rows if not r["current"])
        tally = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        lines += [f"## {name}", "", tally + (f" ({stale} not current)" if stale else ""), ""]
        if name.endswith(CORRECTNESS):
            lines += _correctness_table(rows)
        elif name.endswith(TIMING):
            lines += _timing_table(rows)
        lines.append("")
    return "\n".join(lines)


def _correctness_table(rows: list[dict[str, Any]]) -> list[str]:
    """One row per problem: BNGsim's worst agreement across every reference, and
    whether it agreed with each reference individually."""
    refs = sorted({name for r in rows for name in (r.get("verdicts") or {})})
    header = ["problem", "species", "outcome", "max error / peak",
              *(f"vs {name}" for name in refs), "current"]
    lines = ["| " + " | ".join(header) + " |",
             "|---|---:|---|---:|" + "---|" * len(refs) + "--|"]
    for r in rows:
        sp = (r.get("network") or {}).get("species")
        verdicts = r.get("verdicts") or {}
        agree = []
        for name in refs:
            v = verdicts.get(name)
            agree.append("" if not v else ("ok" if v.get("passed") else "DIFF"))
        lines.append("| " + " | ".join([
            r["problem"], _fmt(sp, "d"), r["outcome"],
            _fmt(r.get("max_err_over_peak"), ".2e"), *agree,
            "yes" if r["current"] else "no",
        ]) + " |")
    return lines


def _timing_table(rows: list[dict[str, Any]]) -> list[str]:
    """One row per (problem, engine): the engine's absolute median time with its
    bootstrap CI, whether it reproduced its first run, and a derived speedup versus
    the baseline reference. Absolute times are the artifact; the speedup is a view
    (docs/DESIGN.md 6.3)."""
    engines = _engine_order(rows)
    header = ["problem", "species", "engine", "outcome", "median (s)", "95% CI",
              f"x {SPEEDUP_BASELINE}", "reproducible", "rounds", "current"]
    lines = ["| " + " | ".join(header) + " |",
             "|---|---:|---|---|---:|---|---:|---|---:|---|"]
    for r in rows:
        sp = (r.get("network") or {}).get("species")
        times = r.get("times") or {}
        repro = r.get("reproducible") or {}
        # A derived number is only trusted when the job passed; a NOT_REPRODUCIBLE
        # (or failed) row would otherwise publish a speedup against times it distrusts.
        trusted = r.get("outcome") == "PASS"
        base = (times.get(SPEEDUP_BASELINE) or {}).get("median_seconds") if trusted else None
        for engine in engines:
            t = times.get(engine)
            if not t:
                continue
            median = t.get("median_seconds")
            ci = (f"{_fmt(t.get('ci_low'), '.4f')}–{_fmt(t.get('ci_high'), '.4f')}"
                  if trusted and t.get("ci_low") is not None else "")
            speed = (base / median if trusted and base and median else None)
            rep = repro.get(engine)
            lines.append("| " + " | ".join([
                r["problem"], _fmt(sp, "d"), engine, r["outcome"],
                _fmt(median, ".4f"), ci, _fmt(speed, ".2f"),
                "" if rep is None else ("yes" if rep else "NO"),
                _fmt(r.get("rounds"), "d"), "yes" if r["current"] else "no",
            ]) + " |")
    return lines
