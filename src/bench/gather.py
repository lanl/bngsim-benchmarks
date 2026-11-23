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
        # Network-free: BNGsim's other engine drawn and compared beside the verdicts,
        # and the same-seed check of the embedding (recorded, never scored).
        cross = m.get("cross_setting") or {}
        if cross.get("verdict"):
            row["cross_setting"] = {"setting": cross.get("setting"),
                                    "passed": cross["verdict"].get("passed"),
                                    "p_value": _p_value(cross["verdict"])}
        if m.get("same_seed"):
            row["same_seed"] = m["same_seed"]
    elif job.kind == CORRECTNESS and job.capability in STEADY:
        # One agreement per reference, at every point of the protocol: the worst
        # species' distance from the reference's state over what the verdict allows
        # (at most 1 agrees). BNGsim's own state against the harness's engine-free
        # checks is kept apart from them: it is not a reference.
        verdicts = dict(rec.get("verdicts") or {})
        checks = verdicts.pop("engine_free", None)
        row["verdicts"] = {name: {k: v.get(k) for k in ("passed", "max_ratio", "worst_point",
                                                       "worst_species", "points_agreed",
                                                       "points", "max_rel")}
                           for name, v in verdicts.items()}
        ratios = [v.get("max_ratio") for v in verdicts.values()
                  if isinstance(v.get("max_ratio"), (int, float))]
        row["max_ratio"] = max(ratios) if ratios else None
        row["points"] = len(m.get("points") or []) or None
        row["engine_free"] = _engine_free_summary(checks)
    elif job.kind == CORRECTNESS:
        # One verdict of BNGsim against each reference engine. The table shows the
        # worst agreement across references (the hardest check the row passed) and
        # keeps the per-engine numbers for the JSON.
        verdicts = rec.get("verdicts") or {}
        row["verdicts"] = {name: {"max_err_over_peak": v.get("max_err_over_peak"),
                                  "max_rel": v.get("max_rel"), "n_cells": v.get("n_cells"),
                                  "passed": v.get("passed"),
                                  # Both halves of the gate, where the record has them
                                  # (one scored before the per-species check has neither).
                                  "vendored_passed": v.get("vendored_passed"),
                                  "species_check": _species_summary(v.get("species_check"))}
                           for name, v in verdicts.items()}
        peaks = [v.get("max_err_over_peak") for v in verdicts.values()
                 if v.get("max_err_over_peak") is not None]
        row["max_err_over_peak"] = max(peaks) if peaks else None
        row["worst_species"] = worst_species(verdicts)
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
            if m.get("solve_split"):
                # Each engine's own account of its solve, apart from its start-up.
                row["solve_split"] = m["solve_split"]
    # Any engine's failure is recorded under "<name>_error"; pass every one through,
    # plus the generic "error" of a job that failed before any engine ran.
    for k, val in rec.items():
        if k == "error" or k.endswith("_error"):
            if val:
                row[k] = val
    # A reference that does not apply to the model, with its reason: shown, never
    # counted as attempted.
    if rec.get("not_applicable"):
        row["not_applicable"] = dict(rec["not_applicable"])
    return row


# Capabilities whose correctness verdict is a statistical test on ensembles.
STOCHASTIC = frozenset({"ssa", "nf"})
# Capabilities whose correctness verdict is a steady state's agreement, point by point.
STEADY = frozenset({"steady_state"})


def _p_value(v: dict[str, Any]) -> float | None:
    """A stochastic verdict's p-value, comparable to its alpha: the permutation
    p-value of a two-ensemble comparison, or the Bonferroni-adjusted one against
    exact moments (the smallest per-cell p-value, ``p_min``, for a record without it)."""
    p = v.get("p_value", v.get("p_min"))
    return p if isinstance(p, (int, float)) else None


def _species_summary(check: dict[str, Any] | None) -> dict[str, Any] | None:
    if not check:
        return None
    return {k: check.get(k) for k in ("passed", "n_fail", "n_species", "n_at_floor", "tol",
                                      "floor", "failing", "worst")}


def _engine_free_summary(check: dict[str, Any] | None) -> dict[str, Any] | None:
    """A steady state's engine-free checks (residual, conservation) without their
    per-point lists."""
    if not check:
        return None
    out: dict[str, Any] = {"passed": check.get("passed"), "evaluated": check.get("evaluated")}
    for part in ("residual", "conservation"):
        c = check.get(part)
        if isinstance(c, dict):
            out[part] = {k: c[k] for k in ("passed", "max", "tol", "worst_point",
                                           "worst_species", "n_laws", "reason") if k in c}
    if check.get("reason"):
        out["reason"] = check["reason"]
    return out


def worst_species(verdicts: dict[str, Any]) -> dict[str, Any] | None:
    """The species closest to (or furthest past) what the per-species check allows,
    over every reference: its name, the reference, its error, its peak, and what it
    was allowed (``allowed``: the larger of the check's tolerance times its peak and
    the atol floor). None when no verdict carries a per-species check."""
    best = None
    for name, v in verdicts.items():
        w = ((v or {}).get("species_check") or {}).get("worst")
        if w and (best is None or (w.get("score") or 0) > (best.get("score") or 0)):
            best = {**w, "reference": name,
                    "floor": (v.get("species_check") or {}).get("floor")}
    return best


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
# A network-free BNGsim has no network: its BNG2.pl step writes the XML instead.
_PRESIM_PARTS = ("netgen_seconds", "xml_seconds", "codegen_seconds", "load_seconds")


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
            "xml_seconds": m.get("xml_seconds"),
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
# A capability whose references do not include run_network names its own: the
# network-free one divides by standalone NFsim, the network-free engine of record.
SPEEDUP_BASELINES = {"nf": "nfsim"}


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
        capability = name.rsplit("-", 1)[0]
        stochastic = capability in STOCHASTIC
        if name.endswith(CORRECTNESS):
            if capability in STEADY:
                lines += _steady_correctness_table(rows)
            else:
                lines += (_stochastic_correctness_table(rows) if stochastic
                          else _correctness_table(rows))
        elif name.endswith(TIMING):
            lines += _timing_table(rows, per_trajectory=stochastic,
                                   baseline=SPEEDUP_BASELINES.get(capability, SPEEDUP_BASELINE))
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
    is agreed / attempted, where a crashed reference counts as attempted. A reference
    that does not apply to the model is named (``run_network: n/a``) but not counted.
    Blank when no reference was attempted (e.g. the job failed before any engine
    ran)."""
    verdicts = r.get("verdicts") or {}
    crashed = _errored_engines(r) - set(verdicts)
    attempted = len(verdicts) + len(crashed)
    if not attempted:
        return ""
    passed = sum(1 for v in verdicts.values() if v.get("passed"))
    diffs = sorted(name for name, v in verdicts.items() if not v.get("passed"))
    misses = sorted([(name, _why(r, name)) for name in crashed]
                    + [(name, "diff") for name in diffs]
                    + [(name, "n/a") for name in (r.get("not_applicable") or {})])
    label = "DIFF" if diffs else ("PARTIAL" if crashed else "FULL")
    named = f" ({', '.join(f'{n}: {why}' for n, why in misses)})" if misses else ""
    return f"{label} {passed}/{attempted}{named}"


def _correctness_table(rows: list[dict[str, Any]]) -> list[str]:
    """One row per problem: the outcome enum, BNGsim's agreement with its references
    (``FULL 4/4`` / ``PARTIAL 3/4 (copasi: crash)``, see ``reference_agreement``), the
    worst agreement, and each reference individually -- ``ok``, ``DIFF`` (ran but
    disagreed), ``crash`` (failed to run) or ``n/a`` (does not apply to the model). A
    single broken reference no longer hides the others behind one
    ``REFERENCE_FAILED``."""
    refs = sorted({name for r in rows for name in (r.get("verdicts") or {})}
                  | {name for r in rows for name in _errored_engines(r)}
                  | {name for r in rows for name in (r.get("not_applicable") or {})})
    header = ["problem", "species", "outcome", "references", "max error / peak",
              "worst species", *(f"vs {name}" for name in refs), "current"]
    lines = ["| " + " | ".join(header) + " |",
             "|---|---:|---|---|---:|---|" + "---|" * len(refs) + "--|"]
    for r in rows:
        sp = (r.get("network") or {}).get("species")
        verdicts = r.get("verdicts") or {}
        errs = _errored_engines(r)
        cells = []
        for name in refs:
            v = verdicts.get(name)
            if v is not None:
                cells.append(_verdict_cell(v))
            elif name in errs:
                cells.append(_why(r, name))
            elif name in (r.get("not_applicable") or {}):
                cells.append("n/a")
            else:
                cells.append("")
        lines.append("| " + " | ".join([
            r["problem"], _fmt(sp, "d"), r["outcome"], reference_agreement(r),
            _fmt(r.get("max_err_over_peak"), ".2e"), _worst_cell(r.get("worst_species")),
            *cells, "yes" if r["current"] else "no",
        ]) + " |")
    lines += ["", "*max error / peak*: the worst disagreement over every reference, as a "
              "fraction of the reference's own peak of that species (1.0 where one engine "
              "holds a species at zero and the other reports round-off). *worst species*: "
              "the species nearest to failing the per-species check, over every reference: "
              "its worst error as a fraction of its own peak over both engines or, where "
              "the species is so small that the check's atol floor sets what it may be "
              "off by, as a percentage of that floor. A reference passes only "
              "when bngsim's vendored verdict AND the per-species check both pass; *DIFF "
              "(species)*, *DIFF (vendored)* or *DIFF (both)* says which failed."]
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


def _steady_correctness_table(rows: list[dict[str, Any]]) -> list[str]:
    """One row per problem, as ``_correctness_table`` but for a steady state: the
    number of points, the worst agreement over the references, each reference's
    result -- ``ok``, ``DIFF (k/n points)``, or why it was not scored -- and BNGsim's
    own state against the engine-free checks."""
    refs = sorted(({name for r in rows for name in (r.get("verdicts") or {})}
                   | {name for r in rows for name in _errored_engines(r)}) - {"bngsim"})
    header = ["problem", "species", "points", "outcome", "references", "worst ratio",
              *(f"vs {name}" for name in refs), "engine-free", "current"]
    lines = ["| " + " | ".join(header) + " |",
             "|---|---:|---:|---|---|---:|" + "---|" * len(refs) + "---|--|"]
    for r in rows:
        sp = (r.get("network") or {}).get("species")
        verdicts = r.get("verdicts") or {}
        errs = _errored_engines(r)
        cells = []
        for name in refs:
            v = verdicts.get(name)
            if v is not None:
                agreed, n = v.get("points_agreed"), v.get("points")
                cells.append("ok" if v.get("passed") else
                             f"DIFF ({agreed}/{n} points)" if isinstance(agreed, int)
                             and isinstance(n, int) else "DIFF")
            elif name in errs:
                cells.append(_why(r, name))
            else:
                cells.append("")
        lines.append("| " + " | ".join([
            r["problem"], _fmt(sp, "d"), _fmt(r.get("points"), "d"), r["outcome"],
            reference_agreement(r), _fmt(r.get("max_ratio"), ".3g"), *cells,
            _engine_free_cell(r.get("engine_free")), "yes" if r["current"] else "no",
        ]) + " |")
    lines += ["", "*worst ratio*: over every reference and every point, the worst species' "
              "distance from the reference's state as a fraction of what the verdict allows "
              "it (bench.capabilities.steady); a reference agrees when it is at most 1 at "
              "every point, and *DIFF (k/n points)* says at how many it was. *engine-free*: "
              "BNGsim's state against the harness's own checks of the .net, which involve no "
              "engine: each species' net rate a vanishing part of its gross flux "
              "(*residual*), and every conservation law's initial total kept "
              "(*conservation*). *crash*: the reference failed, or did not answer the "
              "question -- it did not settle, its state fails an engine-free check, or it "
              "returned a root the integration from the initial state does not reach -- so "
              "it was not scored (its reason is in the JSON row); *timeout*: it ran past its "
              "per-engine wall budget."]
    return lines


def _engine_free_cell(c: dict[str, Any] | None) -> str:
    if not c:
        return ""
    if c.get("evaluated") is False:
        return "not evaluated"
    if c.get("passed"):
        return "ok"
    failed = [part for part in ("residual", "conservation")
              if (c.get(part) or {}).get("passed") is False]
    return f"FAIL ({', '.join(failed)})" if failed else "FAIL"


def failed_check(v: dict[str, Any]) -> str | None:
    """Which half of the ODE correctness gate a failed verdict failed: ``vendored``
    (bngsim's deterministic verdict), ``species`` (the per-species check) or
    ``both``. None when it passed, or when its record does not carry both halves."""
    if v.get("passed"):
        return None
    vendored = v.get("vendored_passed")
    species = (v.get("species_check") or {}).get("passed")
    if vendored is None or species is None:
        return None
    if not vendored and not species:
        return "both"
    return "species" if vendored else "vendored"


def _verdict_cell(v: dict[str, Any]) -> str:
    """``ok``, or ``DIFF`` with which half of the gate failed, where the record
    has both halves."""
    if v.get("passed"):
        return "ok"
    half = failed_check(v)
    return f"DIFF ({half})" if half else "DIFF"


def _short(name: str, n: int = 40) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


def _worst_cell(w: dict[str, Any] | None) -> str:
    """The worst species as ``<name> (<reference>) 3.1e-02 of peak``, or, where
    the atol floor set what it was allowed, its error as a percentage of that floor."""
    if not w:
        return ""
    where = f"`{_short(str(w.get('species')))}` ({w.get('reference')})"
    floor, peak, err = w.get("floor"), w.get("peak"), w.get("err")
    if floor and w.get("allowed") == floor and err is not None:
        return f"{where} {err / floor * 100:.2g}% of floor"
    if peak and err is not None:
        return f"{where} {err / peak:.1e} of peak"
    return where


def _timing_table(rows: list[dict[str, Any]], *, per_trajectory: bool = False,
                  baseline: str = SPEEDUP_BASELINE) -> list[str]:
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
    # Where the engines report their own solve time (network-free), it is shown per
    # trajectory beside the wall time, which for a subprocess holds its start-up.
    solve = per_trajectory and any(r.get("solve_split") for r in rows)
    header = ["problem", "species", "engine", "outcome", "pre-sim (s)",
              "pre-sim parts (s)", f"{unit} median (s)", f"{unit} 95% CI (s)",
              *(["per trajectory (s)"] if per_trajectory else []),
              *(["solve per trajectory (s)"] if solve else []),
              f"x {baseline}", "reproducible", "rounds", "current"]
    lines = ["| " + " | ".join(header) + " |",
             "|---|---:|---|---|---:|---|---:|---|" + ("---:|" if per_trajectory else "")
             + ("---:|" if solve else "") + "---:|---|---:|---|"]
    for r in rows:
        sp = (r.get("network") or {}).get("species")
        times = r.get("times") or {}
        per_traj = r.get("times_per_trajectory") or {}
        presim = r.get("presim") or {}
        repro = r.get("reproducible") or {}
        # A derived number is only trusted when the job passed; a NOT_REPRODUCIBLE
        # (or failed) row would otherwise publish a speedup against times it distrusts.
        trusted = r.get("outcome") == "PASS"
        base = (times.get(baseline) or {}).get("median_seconds") if trusted else None
        split = r.get("solve_split") or {}
        k = r.get("timing_replicates")
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
                *([_sec((split.get(engine) or {}).get("solve_seconds") / k
                        if (split.get(engine) or {}).get("solve_seconds") is not None and k
                        else None)] if solve else []),
                _fmt(speed, ".2f"),
                "" if rep is None else ("yes" if rep else "NO"),
                _fmt(r.get("rounds"), "d"), "yes" if r["current"] else "no",
            ]) + " |")
        for engine in sorted(_errored_engines(r) - set(times)):
            lines.append("| " + " | ".join([
                r["problem"], _fmt(sp, "d"), engine, r["outcome"], "", "", _why(r, engine),
                "", *([""] if per_trajectory else []), *([""] if solve else []), "", "", "",
                "yes" if r["current"] else "no",
            ]) + " |")
        for engine in sorted(set(r.get("not_applicable") or {}) - set(times)):
            lines.append("| " + " | ".join([
                r["problem"], _fmt(sp, "d"), engine, r["outcome"], "", "", "n/a",
                "", *([""] if per_trajectory else []), *([""] if solve else []), "", "", "",
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
              "timed; *n/a*: the engine does not apply to the model (its reason is in "
              "the JSON row)."]
    if per_trajectory:
        n = sorted({r.get("timing_replicates") for r in rows if r.get("timing_replicates")})
        lines += ["", "A timed repetition is an *ensemble* of "
                  + (" / ".join(map(str, n)) if n else "N")
                  + " trajectories, from the same seeds every round, so every round must "
                  "reproduce the first; *per trajectory* is the ensemble median divided by "
                  "the ensemble size. For bngsim, *codegen* is building the SSA simulator, "
                  "which compiles the propensities where it can [cc]. *n/a*: the reference "
                  "cannot simulate this model's exact SSA, so it was not run."]
    if solve:
        lines += ["", "*solve per trajectory* is each engine's own account of its solve, apart "
                  "from its start-up: bngsim's session simulate calls (wall), NFsim's "
                  "reported simulation CPU time, rm_driver's reported simulation time. For a "
                  "network-free bngsim, *xml* is BNG2.pl writing the BNG-XML and *load* "
                  "building the session from it; NFsim's runs are BNG2.pl processes, which "
                  "parse the model and write the XML for every trajectory."]
    return lines
