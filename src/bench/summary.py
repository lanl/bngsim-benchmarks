"""Readable summaries of the published results, generated from the published data.

    published/<suite>/<label>/README.md   one snapshot: what it measured on what, every
                                          member's correctness, the timings
    published/README.md                   every suite: its latest snapshot, its tally,
                                          and whether it is CURRENT or OUT OF DATE
    published/index.json                  the same index, machine-readable

A snapshot's README is a function of that snapshot's manifest.json, correctness.csv
and timing.csv alone. The index is a function of the snapshots, problems/ and the
code (``bench.staleness``), and of the newest bngsim release seen on PyPI, which
index.json records with the day it was looked up. So every one of these files can be
regenerated from what is committed: tests/test_summary.py does so and fails on any
difference, and ``bench publish --index`` rewrites them.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import specs, staleness

INDEX_SCHEMA = 1
README = "README.md"
INDEX_JSON = "index.json"

# One line per suite, for people (the suites themselves are specs.SUITES).
DESCRIPTIONS = {
    "ode-net": "ODE on the `.net` generated from published BNGL",
    "ode-sbml": "ODE on published SBML, read as it is",
    "ssa-sbml": "exact SSA on published SBML, amounts as molecule counts",
    "ss-net": "steady state on the `.net` generated from published BNGL",
    "ss-sbml": "steady state on published SBML, read as it is",
    "ssa-net": "exact SSA on the `.net` generated from published BNGL",
    "psa-net": "partial scaling (PSA) on the `.net` generated from published BNGL",
    "sens-net": "forward sensitivities on the `.net` generated from published BNGL",
    "ode-antimony": "ODE on Antimony converted from ode-net and ode-sbml members",
    "sens-sbml": "forward sensitivities on SBML, from PEtab benchmark problems",
    "nf-xml-aggregation": "network-free on BNG-XML from published BNGL: multivalent aggregation",
    "nf-xml-signaling": "network-free on BNG-XML from published BNGL: signaling and gene circuits",
}

# Job outcomes in the order a tally lists them; any other comes after, by name.
OUTCOMES = ("PASS", "REFERENCE_FAILED", "DIFF", "UNSUPPORTED", "EXCEPTION", "TIMEOUT",
            "BAD_TEST")
OUTCOME_MEANINGS = {
    "PASS": "BNGsim agrees with every reference",
    "REFERENCE_FAILED": "BNGsim ran and agrees with every reference that ran; another "
                        "reference failed on its own (a result about that reference)",
    "DIFF": "BNGsim disagrees with a reference that ran, whatever else failed",
    "UNSUPPORTED": "BNGsim declines the model or protocol as unsupported, saying why",
    "EXCEPTION": "BNGsim failed on a model a reference ran",
    "TIMEOUT": "BNGsim ran out of time",
    "BAD_TEST": "the model could not be prepared, or every engine failed",
}
ENGINE_NAMES = {"bionetgen": "BioNetGen", "roadrunner": "libRoadRunner", "copasi": "COPASI",
                "amici": "AMICI", "nfsim": "NFsim", "rulemonkey": "RuleMonkey",
                "antimony": "Antimony", "petab": "PEtab"}
# A per-reference status as a table cell (correctness.csv's ``status``).
STATUS_CELLS = {"ok": "ok", "crash": "crash", "timeout": "timeout", "n/a": "unsupported",
                "not_applicable": "–"}
# Which half of the ODE gate a DIFF failed (correctness.csv's ``failed_check``).
FAILED_CHECKS = {"vendored": "failed bngsim's vendored verdict",
                 "species": "failed the per-species check",
                 "both": "failed both the vendored verdict and the per-species check"}
NOT_TIMED = {"crash": "crash", "timeout": "timeout", "n/a": "unsupported",
             "not_applicable": "not applicable"}
FORMAT_DOC = "docs/published_format.md"


# --------------------------------------------------------------------------- #
# Reading a snapshot
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Snapshot:
    path: Path
    manifest: dict[str, Any]
    correctness: list[dict[str, str]]
    timing: list[dict[str, str]]

    @property
    def suite(self) -> str:
        return self.manifest["suite"]

    @property
    def label(self) -> str:
        return self.path.name


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_snapshot(path: Path) -> Snapshot:
    path = Path(path)
    return Snapshot(path, json.loads((path / "manifest.json").read_text(encoding="utf-8")),
                    _read_csv(path / "correctness.csv"), _read_csv(path / "timing.csv"))


def snapshots(published: Path) -> dict[str, list[Path]]:
    """Every snapshot directory under ``published``, by suite, oldest first (by the
    manifest's ``published`` time, then the label)."""
    out: dict[str, list[tuple[str, str, Path]]] = {}
    for man in sorted(Path(published).glob("*/*/manifest.json")):
        m = json.loads(man.read_text(encoding="utf-8"))
        out.setdefault(m.get("suite") or man.parent.parent.name, []).append(
            (m.get("published") or "", man.parent.name, man.parent))
    return {s: [p for _, _, p in sorted(v)] for s, v in sorted(out.items())}


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def _natural(name: str) -> list[Any]:
    """Sort key putting ``nc=30`` before ``nc=100``."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def _sig(x: float, digits: int) -> str:
    if x >= 10 ** digits:
        return f"{x:.0f}"
    if x >= 1e-4:
        return f"{x:#.{digits}g}".rstrip(".")
    return f"{x:.{digits - 1}e}"


def seconds(text: str | None, digits: int = 3) -> str:
    """A time in seconds as people read it: three significant figures (no exponent
    from 0.0001 s up)."""
    return "" if text in (None, "") else _sig(float(text), digits)


def solve_cells(median: str | None, low: str | None, high: str | None) -> tuple[str, str]:
    """A median and its confidence interval, to as many significant figures (3 to 6)
    as it takes for the interval's ends to differ."""
    if median in (None, ""):
        return "", ""
    if low in (None, "") or high in (None, ""):
        return seconds(median), ""
    for digits in range(3, 7):
        lo, hi = seconds(low, digits), seconds(high, digits)
        if lo != hi:
            break
    return seconds(median, digits), f"{lo}–{hi}"


def _ratio(x: float) -> str:
    return f"{x:.0f}" if x >= 10 else f"{x:#.2g}".rstrip(".")


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def one_line(text: str | None, limit: int = 200) -> str:
    """The first line of an error, cut at ``limit`` characters, as a code span."""
    first = (text or "").strip().splitlines()[0].strip() if (text or "").strip() else ""
    if len(first) > limit:
        first = first[: limit - 1].rstrip() + "…"
    tick = "``" if "`" in first else "`"
    pad = " " if tick == "``" else ""
    return f"{tick}{pad}{first}{pad}{tick}" if first else ""


def _table(header: list[str], align: str, rows: list[list[str]]) -> list[str]:
    seps = {"l": "---", "r": "--:"}
    return (["| " + " | ".join(header) + " |",
             "|" + "|".join(seps[a] for a in align) + "|"]
            + ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows])


def tally(outcomes: list[str]) -> str:
    """``4 PASS, 6 REFERENCE_FAILED``, in the order of OUTCOMES."""
    counts = Counter(outcomes)
    order = [o for o in OUTCOMES if o in counts] + sorted(set(counts) - set(OUTCOMES))
    return ", ".join(f"{counts[o]} {o}" for o in order)


def _positions(ps: list[int]) -> str:
    """``1-4, 7``."""
    ps = sorted(set(ps))
    runs: list[list[int]] = []
    for p in ps:
        if runs and p == runs[-1][-1] + 1:
            runs[-1].append(p)
        else:
            runs.append([p])
    return ", ".join(f"#{r[0]}" if len(r) == 1 else f"#{r[0]}-{r[-1]}" for r in runs)


def _when(stamp: str | None) -> str:
    m = re.match(r"^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})", stamp or "")
    return f"{m.group(1)} {m.group(2)} UTC" if m else (stamp or "")


# --------------------------------------------------------------------------- #
# One snapshot's README
# --------------------------------------------------------------------------- #
def _groups(rows: list[dict[str, str]]) -> list[tuple[tuple[int, str], list[dict[str, str]]]]:
    """Rows grouped by (position, setting), in the order the CSV lists them."""
    out: dict[tuple[int, str], list[dict[str, str]]] = {}
    for r in rows:
        out.setdefault((int(r["position"]), r.get("setting") or ""), []).append(r)
    return list(out.items())


def _engine_order(names: set[str], manifest: dict[str, Any]) -> list[str]:
    """BNGsim's own engines first, then the references in the order the job ids name
    them (``bngsim-vs-run_network-roadrunner-copasi-amici``), then the rest."""
    named: list[str] = []
    for job in manifest.get("jobs") or []:
        parts = job["job_id"].split("/")
        if len(parts) > 2:
            for n in parts[2].split("-"):
                if n not in named:
                    named.append(n)
    own = sorted((n for n in names if n.startswith("bngsim")), key=_natural)
    refs = [n for n in named if n in names and n not in own]
    rest = sorted(names - set(own) - set(refs), key=_natural)
    return own + refs + rest


def _member_cell(snap: Snapshot, problem: str) -> str:
    for m in snap.manifest.get("members") or []:
        src = m.get("source") or {}
        if m.get("problem") == problem and src.get("repo") and src.get("commit") \
                and src.get("path"):
            return (f"[{problem}](https://github.com/{src['repo']}/blob/{src['commit']}/"
                    f"{src['path']})")
    return problem


def _size(row: dict[str, str]) -> str:
    sp, rx = row.get("species") or "", row.get("reactions") or ""
    return f"{sp} / {rx}" if sp or rx else ""


def _verdict(row: dict[str, str]) -> str:
    status = row.get("status") or ""
    if status == "DIFF":
        if f"{row.get('engine')}: refuted" in (row.get("agreement") or ""):
            return "refuted"
        fc = row.get("failed_check")
        return f"DIFF ({fc})" if fc else "DIFF"
    return STATUS_CELLS.get(status, status)


def _failure_line(row: dict[str, str], where: str) -> str:
    engine, status = row.get("engine") or "", row.get("status") or ""
    if not engine:
        err = one_line(row.get("error"))
        return f"- {where}: the job failed before any engine ran ({status})" \
            + (f", {err}" if err else "")
    word = _verdict(row)
    if status == "DIFF" and word == "refuted":
        detail = ("it disagrees with BNGsim, and the references that agree with BNGsim "
                  "refute it")
    elif status == "DIFF":
        word = "DIFF"
        bits = ["BNGsim disagrees with it"]
        if row.get("max_err_over_peak"):
            bits.append(f"max error / peak {float(row['max_err_over_peak']):.2g}")
        if row.get("failed_check") in FAILED_CHECKS:
            bits.append(FAILED_CHECKS[row["failed_check"]])
        detail = "; ".join(bits)
    else:
        detail = one_line(row.get("error"))
    return f"- {where}, {engine}: {word}" + (f", {detail}" if detail else "")


def _header(snap: Snapshot) -> list[str]:
    m = snap.manifest
    b = m.get("bngsim") or {}
    engines = ", ".join(f"{ENGINE_NAMES.get(k, k)} {', '.join(map(str, v))}"
                        for k, v in (m.get("engines") or {}).items() if v)
    h = m.get("harness") or {}
    cores = "; ".join(f"{c.get('upstream')} {c.get('ref')} ({str(c.get('commit'))[:12]})"
                      for c in h.get("parity_core") or [])
    harness = f"bench {', '.join(h.get('bench') or ['?'])}"
    commits = ", ".join(str(c)[:12] for c in h.get("source_commits") or [])
    if commits:
        harness += f" at commit{'s' if len(h['source_commits']) > 1 else ''} {commits}"
    if cores:
        harness += f"; verdict code {cores}"
    node = (m.get("node_type") or "none (no timing)").replace(" | ", ", ")
    n = len(m.get("members") or [])
    what = DESCRIPTIONS.get(snap.suite, snap.suite)
    lines = [
        f"# {snap.suite}: bngsim {b.get('version')} ({b.get('build_commit')})",
        "",
        f"{what[:1].upper()}{what[1:]}. {n} models, each run by BNGsim and by reference "
        "engines that check it.",
        "Generated by `bench publish` from this directory's `manifest.json`, "
        "`correctness.csv` and `timing.csv`; do not edit. Column meanings: "
        f"[{FORMAT_DOC}](../../../{FORMAT_DOC}).",
        "",
        f"- BNGsim: {b.get('version')}, build {b.get('build_commit')}",
        f"- Environment: {engines}" if engines else "- Environment: (not recorded)",
        f"- Harness: {harness}",
        f"- Timing node: {node}",
        f"- Published: {_when(m.get('published'))}",
    ]
    return lines


def _correctness(snap: Snapshot) -> list[str]:
    groups = _groups(snap.correctness)
    multi = len({s for (_, s), _ in groups}) > 1
    engines = _engine_order({r["engine"] for r in snap.correctness if r.get("engine")},
                            snap.manifest)
    jobs = [j for j in snap.manifest.get("jobs") or [] if j.get("kind") == "correctness"]
    n_members = len(snap.manifest.get("members") or [])
    count = f" of {len(jobs)} correctness jobs" if len(jobs) != n_members else ""
    lines = ["", "## Correctness", "",
             f"{tally([g[0]['outcome'] for _, g in groups])}{count}. Each reference "
             "column is that engine's check of BNGsim: ok, DIFF (it disagrees; for an ODE, "
             "which half of the gate failed), refuted (it disagrees, and the references "
             "that agree with BNGsim refute it), crash, timeout, unsupported (it cannot "
             "run the model faithfully), – (the comparison does not apply).", ""]
    header = ["#", "model", *(["setting"] if multi else []), "species / reactions",
              "outcome", *engines]
    align = "rl" + ("l" if multi else "") + "rl" + "l" * len(engines)
    rows, fails, na = [], [], {}
    last = None
    for (pos, setting), g in groups:
        by = {r.get("engine") or "": r for r in g}
        problem = g[0]["problem"]
        again, last = pos == last, pos
        rows.append(["" if again else str(pos), "" if again else _member_cell(snap, problem),
                     *([setting] if multi else []), "" if again else _size(g[0]),
                     g[0]["outcome"],
                     *(_verdict(by[e]) if e in by else "" for e in engines)])
        where = f"#{pos} {problem}" + (f" [{setting}]" if multi else "")
        for r in g:
            if r.get("status") == "not_applicable":
                na.setdefault((r["engine"], r.get("error") or ""), []).append(pos)
            elif r.get("status") != "ok":
                fails.append(_failure_line(r, where))
    lines += _table(header, align, rows)
    if fails:
        lines += ["", "Every check that did not pass, and why:", "", *fails]
    if na:
        lines += ["", "Not applicable by design:", ""]
        lines += [f"- {engine}, {_positions(ps)}: {one_line(reason)}"
                  for (engine, reason), ps in na.items()]
    return lines


def _timing(snap: Snapshot) -> list[str]:
    groups = _groups(snap.timing)
    if not groups:
        return ["", "## Timing", "", "No timing in this snapshot."]
    multi = len({s for (_, s), _ in groups}) > 1
    engines = _engine_order({r["engine"] for r in snap.timing if r.get("engine")},
                            snap.manifest)
    rank = {e: i for i, e in enumerate(engines)}
    lines = ["", "## Timing", "",
             "Absolute seconds on the timing node. *pre-sim*: the one-time cost before the "
             "timed solves, measured cold (BNGsim: network generation, code generation and "
             "load; a reference: its own code generation and load); blank for an engine "
             "started fresh for every solve, whose load is inside every solve. *solve*: "
             "the median warm solve over the timed rounds, with its bootstrap 95% "
             "confidence interval. An engine that did not run in the correctness job is "
             "not timed (its reason is under Correctness); a timing whose correctness "
             "outcome does not let it run is withheld.", ""]
    roles: dict[str, list[str]] = {}
    for r in sorted(snap.timing, key=lambda r: rank.get(r.get("engine") or "", len(rank))):
        if r.get("status") == "timed" and r.get("role") \
                and r["role"] not in roles.setdefault(r["engine"], []):
            roles[r["engine"]].append(r["role"])
    by_role: dict[str, list[str]] = {}
    for e, rs in roles.items():
        for role in rs:
            by_role.setdefault(role, []).append(e)
    if by_role:
        lines += ["How each engine is timed:", "",
                  *(f"- {', '.join(es)}: {role}" for role, es in by_role.items()), ""]
    header = ["#", "model", *(["setting"] if multi else []), "rounds", "engine",
              "pre-sim", "solve", "95% CI"]
    align = "rl" + ("l" if multi else "") + "rlrrl"
    rows, failed = [], []
    solve: dict[tuple[int, str], dict[str, float]] = {}
    last = None
    for (pos, setting), g in groups:
        g = sorted(g, key=lambda r: rank.get(r.get("engine") or "", len(rank)))
        problem = g[0]["problem"]
        rounds = next((r.get("rounds") for r in g if r.get("rounds")), "") or ""
        again, last = pos == last, pos
        for i, r in enumerate(g):
            first = i == 0
            lead = [str(pos) if first and not again else "",
                    problem if first and not again else "",
                    *([setting if first else ""] if multi else []), rounds if first else ""]
            status = r.get("status") or ""
            if status == "timed":
                med, ci = solve_cells(r.get("sim_median_seconds"), r.get("sim_ci_low_seconds"),
                                      r.get("sim_ci_high_seconds"))
                if r.get("reproducible") == "False":
                    med += " (not reproducible)"
                rows.append([*lead, r["engine"], seconds(r.get("presim_seconds")), med, ci])
                if r.get("sim_median_seconds"):
                    solve.setdefault((pos, setting), {})[r["engine"]] = \
                        float(r["sim_median_seconds"])
            elif not r.get("engine"):
                what = (f"withheld: {r.get('error')}" if status == "withheld"
                        else f"timing failed: {status}")
                rows.append([*lead, "–", "", what, ""])
                if status != "withheld":
                    failed.append(f"- #{pos} {problem}" + (f" [{setting}]" if multi else "")
                                  + f": {status}, {one_line(r.get('error'))}")
            else:
                rows.append([*lead, r["engine"], "",
                             f"not timed: {NOT_TIMED.get(status, status)}", ""])
    lines += _table(header, align, rows)
    if failed:
        lines += ["", "Timing jobs that failed:", "", *failed]
    lines += _speedups(snap, groups, solve, engines, multi)
    return lines


def _speedups(snap: Snapshot, groups: list, solve: dict[tuple[int, str], dict[str, float]],
              engines: list[str], multi: bool) -> list[str]:
    """Each reference's solve median over BNGsim's (``bngsim``, or ``bngsim@nc=N`` for
    a reference at the same ``@nc=N``): derived from the table above, never stored."""
    refs = [e for e in engines if not e.startswith("bngsim")]

    def ratio(times: dict[str, float], ref: str) -> str:
        own = "bngsim" + (ref[ref.index("@"):] if "@" in ref else "")
        if ref in times and times.get(own):
            return _ratio(times[ref] / times[own])
        return ""

    rows = []
    for (pos, setting), g in groups:
        times = solve.get((pos, setting)) or {}
        cells = [ratio(times, r) for r in refs]
        if any(cells):
            rows.append([str(pos), g[0]["problem"], *([setting] if multi else []), *cells])
    if not rows:
        return []
    used = [i for i in range(len(refs)) if any(r[-len(refs) + i] for r in rows)]
    lead = 3 if multi else 2
    rows = [r[:lead] + [r[lead + i] for i in used] for r in rows]
    header = ["#", "model", *(["setting"] if multi else []), *(refs[i] for i in used)]
    return (["", "### Speedup (derived)", "",
             "Each reference's solve median divided by BNGsim's (`bngsim`; for a "
             "reference at `@nc=N`, `bngsim@nc=N`), from the table above: above 1, "
             "BNGsim solves faster. Pre-sim costs are not in it.", ""]
            + _table(header, "rl" + ("l" if multi else "") + "r" * len(used), rows))


def snapshot_readme(snap: Snapshot) -> str:
    """The README of one snapshot, from its manifest and CSVs only."""
    return "\n".join(_header(snap) + _correctness(snap) + _timing(snap)) + "\n"


# --------------------------------------------------------------------------- #
# The index
# --------------------------------------------------------------------------- #
def index(published: Path, problems: list[specs.Problem], jobs: list, *,
          release: dict[str, str] | None, root: Path) -> dict[str, Any]:
    """Every suite of specs.SUITES: its latest snapshot, the tally of its correctness
    outcomes, and its status (``bench.staleness``)."""
    published = Path(published)
    found = snapshots(published)
    members = specs.suite_members(problems)
    pins = staleness.pins(root)
    skipped = staleness.skipped_releases(published)
    suites = []
    for suite, (cap, formats) in specs.SUITES.items():
        entry: dict[str, Any] = {
            "suite": suite, "description": DESCRIPTIONS.get(suite, ""), "capability": cap,
            "formats": list(formats), "members": len(members.get(suite) or []),
        }
        paths = found.get(suite) or []
        if not paths:
            entry.update(snapshot=None, status=staleness.NOT_PUBLISHED, reasons=[], notes=[])
            suites.append(entry)
            continue
        snap = read_snapshot(paths[-1])
        m = snap.manifest
        outcomes = [j.get("outcome") for j in m.get("jobs") or []
                    if j.get("kind") == "correctness"]
        reasons, notes = staleness.assess(snap.path, problems, jobs, pins=pins,
                                          release=release, skipped=skipped)
        entry.update(
            snapshot=snap.path.relative_to(published).as_posix(),
            older=[p.relative_to(published).as_posix() for p in paths[:-1]],
            published=m.get("published"), bngsim=m.get("bngsim"),
            correctness={"jobs": len(outcomes),
                         "outcomes": dict(sorted(Counter(outcomes).items()))},
            status=staleness.OUT_OF_DATE if reasons else staleness.CURRENT, reasons=reasons,
            notes=notes)
        suites.append(entry)
    return {"schema": INDEX_SCHEMA, "bngsim_release": release,
            "skipped_releases": skipped, "suites": suites,
            "generated_by": "bench publish --index", "formats": FORMAT_DOC}


def index_readme(idx: dict[str, Any]) -> str:
    rel = idx.get("bngsim_release") or {}
    seen = (f"Newest bngsim release on PyPI: {rel['version']} (looked up {rel['checked']})."
            if rel.get("version") else "Newest bngsim release on PyPI: not looked up.")
    if idx.get("skipped_releases"):
        seen += (f" Skipped on purpose ([{staleness.SKIPPED_FILE}]({staleness.SKIPPED_FILE}), "
                 "no snapshot is out of date for these): "
                 + ", ".join(idx["skipped_releases"]) + ".")
    rows = []
    for e in idx["suites"]:
        if e.get("snapshot") is None:
            rows.append([e["suite"], e["description"], str(e["members"]), "–", "–",
                         "not published"])
            continue
        b = e.get("bngsim") or {}
        c = e["correctness"]
        outcomes = [o for o, n in c["outcomes"].items() for _ in range(n)]
        count = f" ({c['jobs']} jobs)" if c["jobs"] != e["members"] else ""
        older = f" (+{len(e['older'])} older)" if e.get("older") else ""
        if e["status"] == staleness.CURRENT:
            status = "CURRENT" + (f" ({'; '.join(e['notes'])})" if e.get("notes") else "")
        else:
            status = f"**{e['status']}**: " + "; ".join(e["reasons"])
        rows.append([f"[{e['suite']}]({e['snapshot']}/{README})", e["description"],
                     str(e["members"]),
                     f"bngsim {b.get('version')}, {str(e.get('published') or '')[:10]}{older}",
                     tally(outcomes) + count, status])
    legend = [f"- {o}: {OUTCOME_MEANINGS[o]}" for o in OUTCOMES]
    lines = [
        "# Published results",
        "",
        "Our results, one snapshot per suite and BNGsim build, each with its own summary "
        "(follow the suite name). Generated by `bench publish --index` from the snapshots, "
        "`problems/` and the code; do not edit. The same index, machine-readable: "
        f"[{INDEX_JSON}]({INDEX_JSON}); the files: [{FORMAT_DOC}](../{FORMAT_DOC}).",
        "",
        seen,
        "",
        *_table(["suite", "what", "models", "latest snapshot", "correctness", "status"],
                "llrlll", rows),
        "",
        "A snapshot is OUT OF DATE when a newer bngsim release is on PyPI, or when what "
        "it measured differs from what the repository would measure now: the suite's "
        "members, a model file, a protocol, a verdict or the harness (problems/ and the "
        "code), or the engine versions the environment pins. `bench staleness` says "
        "why; a weekly workflow runs it and keeps one issue open while a suite is out "
        "of date.",
        "",
        "Outcomes (one per correctness job):",
        "",
        *legend,
    ]
    return "\n".join(lines) + "\n"


def status_of(idx: dict[str, Any] | None) -> list[tuple[Any, ...]] | None:
    """What an index says about each suite -- snapshot, status, reasons -- without
    the day the bngsim release was looked up."""
    if not idx:
        return None
    return [(e.get("suite"), e.get("snapshot"), e.get("status"), tuple(e.get("reasons") or ()),
             tuple(e.get("notes") or ())) for e in idx.get("suites") or []]


def committed_status(published: Path) -> list[tuple[Any, ...]] | None:
    """``status_of`` the committed published/index.json."""
    try:
        return status_of(json.loads((Path(published) / INDEX_JSON).read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None


def staleness_markdown(idx: dict[str, Any], *, shown: bool, link_base: str | None) -> str:
    """The out-of-date suites as markdown, for the issue the staleness workflow keeps:
    each with its latest snapshot and every reason. ``shown`` says whether the
    committed index already says so; ``link_base`` is the URL published/ paths are
    relative to (plain paths without it)."""
    rel = idx.get("bngsim_release") or {}
    stale = [e for e in idx["suites"] if e["status"] == staleness.OUT_OF_DATE]
    unpublished = [e["suite"] for e in idx["suites"] if e["status"] == staleness.NOT_PUBLISHED]

    def where(e: dict[str, Any]) -> str:
        label = e["snapshot"].split("/", 1)[-1]
        if link_base:
            return f"[{label}]({link_base.rstrip('/')}/{e['snapshot']}/{README})"
        return f"`published/{e['snapshot']}`"

    seen = (f"newest bngsim release on PyPI: {rel['version']}, looked up {rel['checked']}"
            if rel.get("version") else "bngsim release not looked up")
    if idx.get("skipped_releases"):
        seen += "; skipped on purpose: " + ", ".join(idx["skipped_releases"])
    lines = [f"{len(stale)} published suite{'s are' if len(stale) != 1 else ' is'} out of "
             f"date ({seen}).", ""]
    if stale:
        lines += _table(["suite", "snapshot", "why"], "lll",
                        [[e["suite"], where(e), "; ".join(e["reasons"])] for e in stale])
        lines.append("")
    if unpublished:
        lines += [f"Not published yet: {', '.join(unpublished)}.", ""]
    if not shown:
        lines += ["The committed `published/README.md` does not show this yet: run "
                  "`pixi run bench publish --index` and commit it.", ""]
    lines += ["To bring a suite up to date, re-run it (`bench submit SUITE`), publish it "
              "(`bench publish SUITE`) and commit `published/`. The staleness workflow "
              "(`.github/workflows/staleness.yml`) updates this issue every week and on "
              "every push to main, and closes it when every published suite is current."]
    return "\n".join(lines) + "\n"


def recorded_release(published: Path) -> dict[str, str] | None:
    """The bngsim release the committed index records, if any."""
    try:
        idx = json.loads((Path(published) / INDEX_JSON).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    rel = idx.get("bngsim_release")
    return rel if isinstance(rel, dict) and rel.get("version") else None


def generated(published: Path, problems: list[specs.Problem], jobs: list, *,
              release: dict[str, str] | None, root: Path) -> dict[Path, str]:
    """Every generated file under ``published`` and what it should hold."""
    published = Path(published)
    files = {}
    for paths in snapshots(published).values():
        for p in paths:
            files[p / README] = snapshot_readme(read_snapshot(p))
    idx = index(published, problems, jobs, release=release, root=root)
    files[published / INDEX_JSON] = json.dumps(idx, indent=1) + "\n"
    files[published / README] = index_readme(idx)
    return files


def differing(files: dict[Path, str]) -> list[Path]:
    """The generated files whose committed text is not what they should hold."""
    return [p for p, text in files.items()
            if not p.exists() or p.read_text(encoding="utf-8") != text]


def write(files: dict[Path, str]) -> list[Path]:
    """Write every file that differs; return those written."""
    changed = differing(files)
    for p in changed:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(files[p], encoding="utf-8")
    return changed
