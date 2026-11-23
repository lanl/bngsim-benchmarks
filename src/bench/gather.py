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
    if job.kind == CORRECTNESS and job.capability in STOCHASTIC:
        # A stochastic verdict is a test: one p-value per comparison (BNGsim against
        # each reference ensemble, and against exact moments where there are any).
        # The table shows the smallest -- the comparison closest to failing.
        verdicts = rec.get("verdicts") or {}
        row["verdicts"] = {name: {"passed": v.get("passed"), "p_value": _p_value(v),
                                  "z_max": v.get("z_max"), "alpha": v.get("alpha"),
                                  "replicates": v.get("replicates"), "test": v.get("test")}
                           for name, v in verdicts.items()}
        ps = [_p_value(v) for v in verdicts.values() if _p_value(v) is not None]
        row["min_p_value"] = min(ps) if ps else None
        row["replicates"] = (m.get("ensembles") or {}).get("bngsim", {}).get("replicates")
    elif job.kind == CORRECTNESS:
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
        reps = m.get("repetitions") or {}
        row.update(
            times=m.get("times") or {},
            reproducible={name: (r or {}).get("passed") for name, r in repro.items()},
            rounds=reps.get("rounds"),
            warmup_rounds=reps.get("warmup_rounds"),
            presim=_presim(m),
            bngsim_cold_seconds=m.get("bngsim_cold_seconds"),
            reference_solver=((m.get("references") or {}).get("run_network") or {}).get("solver"),
            bngsim_config={k: (m.get("bngsim") or {}).get(k)
                           for k in ("codegen_backend", "linear_solver", "jacobian")},
        )
        if job.capability in STOCHASTIC:
            # A timed repetition is an ensemble; the per-trajectory time is derived.
            row.update(timing_replicates=m.get("timing_replicates"),
                       times_per_trajectory=m.get("times_per_trajectory") or {},
                       bngsim_config={"propensity_backend":
                                      (m.get("bngsim") or {}).get("propensity_backend")})
    # Any engine's failure is recorded under "<name>_error"; pass every one through,
    # plus the generic "error" of a job that failed before any engine ran.
    for k, val in rec.items():
        if k == "error" or k.endswith("_error"):
            if val:
                row[k] = val
    return row


# Capabilities whose correctness verdict is a statistical test on ensembles.
STOCHASTIC = frozenset({"ssa"})


def _p_value(v: dict[str, Any]) -> float | None:
    """A stochastic verdict's p-value: the permutation p-value of a two-ensemble
    comparison, or the smallest per-cell p-value against exact moments."""
    p = v.get("p_value", v.get("p_min"))
    return p if isinstance(p, (int, float)) else None


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


def _sec(x: Any) -> str:
    """Seconds at four significant figures, so a millisecond solve reads as
    ``0.001108`` and not ``0.00``. The run-to-run spread (the CI) bounds the real
    precision; this just stops the renderer from throwing digits away."""
    return "" if x is None else f"{x:.4g}"


# Pre-simulation cost: the one-time load/build an engine pays before the timed solve.
# BNGsim's is network generation + codegen + load; a reference's is its own codegen
# (AMICI compiles SBML->C++) + load. It is reported beside the solve time because a
# small solve can hide a large build -- the re-simulation claim (docs/DESIGN.md 1, 7.3).
_PRESIM_PARTS = ("netgen_seconds", "codegen_seconds", "load_seconds")


def _presim(m: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-engine pre-simulation breakdown, pulled from the timing measurements.

    Each entry carries the engine's recorded ``role`` (``engine_roles``), so the
    renderer can tell an engine whose load is genuinely folded into every solve
    (``fresh subprocess``: run_network) from one whose pre-sim number is missing."""
    roles = m.get("engine_roles") or {}
    bng = m.get("bngsim") or {}
    out: dict[str, dict[str, Any]] = {
        "bngsim": {
            "netgen_seconds": m.get("netgen_seconds"),
            "codegen_seconds": bng.get("codegen_seconds"),
            "load_seconds": bng.get("load_seconds"),
            "backend": bng.get("codegen_backend") or bng.get("propensity_backend"),
            "role": roles.get("bngsim"),
        }
    }
    for name, rv in (m.get("references") or {}).items():
        rv = rv or {}
        out[name] = {
            "codegen_seconds": rv.get("codegen_seconds"),
            "load_seconds": rv.get("load_seconds"),
            "role": roles.get(name),
        }
    return out


# The role ``run_timing`` records for an engine run as a fresh process per solve: its
# load is inside every timed rep, so there is no separate pre-sim number to report.
_FRESH_SUBPROCESS = "fresh subprocess"
_IN_SIM = "in sim"


def _presim_total(d: dict[str, Any] | None) -> float | None:
    if not d:
        return None
    vals = [d.get(k) for k in _PRESIM_PARTS]
    vals = [v for v in vals if isinstance(v, (int, float))]
    return sum(vals) if vals else None


def _presim_cells(d: dict[str, Any] | None) -> tuple[str, str]:
    """The (total, breakdown) pre-sim cells for one engine. A fresh-subprocess engine
    is labelled, not given an invented split; a missing number stays blank."""
    if d and d.get("role") == _FRESH_SUBPROCESS:
        return _IN_SIM, "fresh subprocess per rep"
    parts = []
    for key in _PRESIM_PARTS:
        v = (d or {}).get(key)
        if isinstance(v, (int, float)):
            label = key[: -len("_seconds")]
            if key == "codegen_seconds" and (d or {}).get("backend"):
                label += f" [{d['backend']}]"
            parts.append(f"{label} {_sec(v)}")
    return _sec(_presim_total(d)), " + ".join(parts)


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
        stochastic = name.rsplit("-", 1)[0] in STOCHASTIC
        if name.endswith(CORRECTNESS):
            lines += _stochastic_correctness_table(rows) if stochastic else _correctness_table(rows)
        elif name.endswith(TIMING):
            lines += _timing_table(rows, per_trajectory=stochastic)
        lines.append("")
    return "\n".join(lines)


def _errored_engines(r: dict[str, Any]) -> set[str]:
    """References that failed to run, recorded as ``<name>_error`` (not the generic
    job-level ``error``)."""
    return {k[: -len("_error")] for k in r
            if k.endswith("_error") and k != "error" and r.get(k)}


def _why(r: dict[str, Any], name: str) -> str:
    """How a reference failed to run: ``timeout`` (its per-engine budget ran out),
    ``n/a`` (it cannot simulate this model faithfully, so it was not run) or
    ``crash`` (anything else)."""
    err = str(r.get(f"{name}_error") or "")
    if err.startswith("TIMEOUT"):
        return "timeout"
    return "n/a" if err.startswith("UNSUPPORTED") else "crash"


def reference_agreement(r: dict[str, Any]) -> str:
    """BNGsim's agreement with its references as one derived label, beside (never
    instead of) the outcome enum: ``FULL 4/4`` when every reference ran and agreed;
    ``PARTIAL 3/4 (copasi: crash)`` when the only shortfall is a reference that failed
    to run; ``DIFF 3/4 (roadrunner: diff)`` when a reference ran and disagreed. ``n/m``
    is agreed / attempted, where a crashed reference counts as attempted. Blank when no
    reference was attempted (e.g. the job failed before any engine ran)."""
    verdicts = r.get("verdicts") or {}
    crashed = _errored_engines(r) - set(verdicts)
    attempted = len(verdicts) + len(crashed)
    if not attempted:
        return ""
    passed = sum(1 for v in verdicts.values() if v.get("passed"))
    diffs = sorted(name for name, v in verdicts.items() if not v.get("passed"))
    misses = sorted([(name, _why(r, name)) for name in crashed]
                    + [(name, "diff") for name in diffs])
    label = "DIFF" if diffs else ("PARTIAL" if crashed else "FULL")
    named = f" ({', '.join(f'{n}: {why}' for n, why in misses)})" if misses else ""
    return f"{label} {passed}/{attempted}{named}"


def _correctness_table(rows: list[dict[str, Any]]) -> list[str]:
    """One row per problem: the outcome enum, BNGsim's agreement with its references
    (``FULL 4/4`` / ``PARTIAL 3/4 (copasi: crash)``, see ``reference_agreement``), the
    worst agreement, and each reference individually -- ``ok``, ``DIFF`` (ran but
    disagreed) or ``crash`` (failed to run). A single broken reference no longer hides
    the others behind one ``REFERENCE_FAILED``."""
    refs = sorted({name for r in rows for name in (r.get("verdicts") or {})}
                  | {name for r in rows for name in _errored_engines(r)})
    header = ["problem", "species", "outcome", "references", "max error / peak",
              *(f"vs {name}" for name in refs), "current"]
    lines = ["| " + " | ".join(header) + " |",
             "|---|---:|---|---|---:|" + "---|" * len(refs) + "--|"]
    for r in rows:
        sp = (r.get("network") or {}).get("species")
        verdicts = r.get("verdicts") or {}
        errs = _errored_engines(r)
        cells = []
        for name in refs:
            v = verdicts.get(name)
            if v is not None:
                cells.append("ok" if v.get("passed") else "DIFF")
            elif name in errs:
                cells.append(_why(r, name))
            else:
                cells.append("")
        lines.append("| " + " | ".join([
            r["problem"], _fmt(sp, "d"), r["outcome"], reference_agreement(r),
            _fmt(r.get("max_err_over_peak"), ".2e"), *cells,
            "yes" if r["current"] else "no",
        ]) + " |")
    return lines


def _stochastic_correctness_table(rows: list[dict[str, Any]]) -> list[str]:
    """One row per problem, as ``_correctness_table`` but for a statistical verdict:
    the smallest p-value over the comparisons beside the outcome and the agreement
    label, the ensemble size, and each comparison's result with its p-value --
    ``ok (p 0.41)``, ``DIFF (p 0.0005)``, or why a reference did not run. A
    comparison fails when its p-value is at most alpha (in the JSON row)."""
    refs = sorted({name for r in rows for name in (r.get("verdicts") or {})}
                  | {name for r in rows for name in _errored_engines(r)})
    header = ["problem", "species", "outcome", "references", "min p", "replicates",
              *(f"vs {name}" for name in refs), "current"]
    lines = ["| " + " | ".join(header) + " |",
             "|---|---:|---|---|---:|---:|" + "---|" * len(refs) + "--|"]
    for r in rows:
        sp = (r.get("network") or {}).get("species")
        verdicts = r.get("verdicts") or {}
        errs = _errored_engines(r)
        cells = []
        for name in refs:
            v = verdicts.get(name)
            if v is not None:
                p = v.get("p_value")
                cells.append(("ok" if v.get("passed") else "DIFF")
                             + (f" (p {p:.2g})" if p is not None else ""))
            elif name in errs:
                cells.append(_why(r, name))
            else:
                cells.append("")
        lines.append("| " + " | ".join([
            r["problem"], _fmt(sp, "d"), r["outcome"], reference_agreement(r),
            _fmt(r.get("min_p_value"), ".2g"), _fmt(r.get("replicates"), "d"), *cells,
            "yes" if r["current"] else "no",
        ]) + " |")
    return lines


def _timing_table(rows: list[dict[str, Any]], *, per_trajectory: bool = False) -> list[str]:
    """One row per (problem, engine): the one-time pre-sim cost (load/compile) and its
    breakdown beside the per-solve median time with its bootstrap CI, whether the
    engine reproduced its first run, and a derived speedup versus the baseline
    reference. Two numbers, not one: a fast solve can sit behind a slow build (AMICI
    compiles SBML->C++), and the fitting-loop amortization argument needs both. A
    reference that crashed during timing gets a row saying so rather than vanishing.
    Absolute times are the artifact; the speedup is a view (docs/DESIGN.md 6.3, 7.3)."""
    engines = _engine_order(rows)
    # A stochastic timing's repetition is an ensemble: label it so, and add the
    # derived time per trajectory beside it.
    unit = "ensemble" if per_trajectory else "sim"
    header = ["problem", "species", "engine", "outcome", "pre-sim (s)",
              "pre-sim parts (s)", f"{unit} median (s)", f"{unit} 95% CI (s)",
              *(["per trajectory (s)"] if per_trajectory else []),
              f"x {SPEEDUP_BASELINE}", "reproducible", "rounds", "current"]
    lines = ["| " + " | ".join(header) + " |",
             "|---|---:|---|---|---:|---|---:|---|" + ("---:|" if per_trajectory else "")
             + "---:|---|---:|---|"]
    for r in rows:
        sp = (r.get("network") or {}).get("species")
        times = r.get("times") or {}
        per_traj = r.get("times_per_trajectory") or {}
        presim = r.get("presim") or {}
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
            ci = (f"{_sec(t.get('ci_low'))}–{_sec(t.get('ci_high'))}"
                  if trusted and t.get("ci_low") is not None else "")
            speed = (base / median if trusted and base and median else None)
            rep = repro.get(engine)
            total, parts = _presim_cells(presim.get(engine))
            lines.append("| " + " | ".join([
                r["problem"], _fmt(sp, "d"), engine, r["outcome"],
                total, parts, _sec(median), ci,
                *([_sec((per_traj.get(engine) or {}).get("median_seconds"))]
                  if per_trajectory else []),
                _fmt(speed, ".2f"),
                "" if rep is None else ("yes" if rep else "NO"),
                _fmt(r.get("rounds"), "d"), "yes" if r["current"] else "no",
            ]) + " |")
        for engine in sorted(_errored_engines(r) - set(times)):
            lines.append("| " + " | ".join([
                r["problem"], _fmt(sp, "d"), engine, r["outcome"], "", "", _why(r, engine),
                "", *([""] if per_trajectory else []), "", "", "",
                "yes" if r["current"] else "no",
            ]) + " |")
    lines += ["", "Times in seconds. *pre-sim* is the one-time load/build before the "
              "timed solve, measured once (cold), so it has no CI: bngsim = netgen + "
              "codegen [backend] + load; a reference = its own codegen + load. *in sim*: "
              "the engine runs as a fresh subprocess per solve, so its load is inside "
              "every sim rep. *sim* is the warm-solve median over the rounds, with a "
              "bootstrap 95% CI; *crash*: the reference failed to run and was dropped "
              "from the timed set (its error is in the JSON row); *timeout*: it ran "
              "past its per-engine wall budget in the correctness job, so it was not "
              "timed."]
    if per_trajectory:
        n = sorted({r.get("timing_replicates") for r in rows if r.get("timing_replicates")})
        lines += ["", "A timed repetition is an *ensemble* of "
                  + (" / ".join(map(str, n)) if n else "N")
                  + " trajectories, from the same seeds every round, so every round must "
                  "reproduce the first; *per trajectory* is the ensemble median divided by "
                  "the ensemble size. For bngsim, *codegen* is building the SSA simulator, "
                  "which compiles the propensities where it can [cc]. *n/a*: the reference "
                  "cannot simulate this model's exact SSA, so it was not run."]
    return lines
