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
        v = rec.get("verdict") or {}
        row.update(max_err_over_peak=v.get("max_err_over_peak"), max_rel=v.get("max_rel"),
                   n_cells=v.get("n_cells"))
    else:
        s = m.get("summary") or {}
        row.update(
            speedup=s.get("speedup"), ci_low=s.get("ci_low"), ci_high=s.get("ci_high"),
            median_bngsim_seconds=s.get("median_a_seconds"),
            median_reference_seconds=s.get("median_b_seconds"),
            pairs=(m.get("repetitions") or {}).get("pairs"),
            bngsim_cold_seconds=m.get("bngsim_cold_seconds"),
            reference_solver=(m.get("reference_config") or {}).get("solver"),
            bngsim_config={k: (m.get("bngsim") or {}).get(k)
                           for k in ("codegen_backend", "linear_solver", "jacobian")},
        )
    for k in ("error", "bngsim_error", "run_network_error"):
        if rec.get(k):
            row[k] = rec[k]
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


def render(reports: dict[str, list[dict[str, Any]]]) -> str:
    lines = ["# Results summary", ""]
    for name in sorted(reports):
        rows = reports[name]
        counts = Counter(r["outcome"] for r in rows)
        stale = sum(1 for r in rows if not r["current"])
        tally = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        lines += [f"## {name}", "", tally + (f" ({stale} not current)" if stale else ""), ""]
        if name.endswith(CORRECTNESS):
            lines += ["| problem | species | outcome | max error / peak | current |",
                      "|---|---:|---|---:|---|"]
            for r in rows:
                sp = (r.get("network") or {}).get("species")
                lines.append(f"| {r['problem']} | {_fmt(sp, 'd')} | {r['outcome']} | "
                             f"{_fmt(r.get('max_err_over_peak'), '.2e')} | {'yes' if r['current'] else 'no'} |")
        elif name.endswith(TIMING):
            lines += ["| problem | species | speedup | 95% CI | BNGsim warm (s) | reference (s) "
                      "| reference solver | pairs | current |",
                      "|---|---:|---:|---|---:|---:|---|---:|---|"]
            for r in rows:
                sp = (r.get("network") or {}).get("species")
                ci = (f"{_fmt(r.get('ci_low'), '.2f')}–{_fmt(r.get('ci_high'), '.2f')}"
                      if r.get("ci_low") is not None else "")
                lines.append(
                    f"| {r['problem']} | {_fmt(sp, 'd')} | {_fmt(r.get('speedup'), '.2f')} | {ci} | "
                    f"{_fmt(r.get('median_bngsim_seconds'), '.4f')} | "
                    f"{_fmt(r.get('median_reference_seconds'), '.4f')} | {r.get('reference_solver') or ''} | "
                    f"{_fmt(r.get('pairs'), 'd')} | "
                    f"{'yes' if r['current'] else 'no'} |")
        lines.append("")
    return "\n".join(lines)
