"""Whether a published snapshot still says what the repository would measure now.

A snapshot is OUT OF DATE, with a reason for each of these, when

- a newer bngsim release is on PyPI than the one it measured, unless it is one we
  skip on purpose (published/skipped-releases.json, each with the reason), or the
  environment (pixi.toml) pins another bngsim;
- its suite's members, or their positions, differ from problems/;
- a member's jobs differ: one the suite defines now is not in the snapshot, or one in
  it is no longer defined (an engine added to a capability, a setting changed);
- a job's definition differs from the one its record was keyed on: the model file's
  sha256, the protocol, the verdict, the harness version, the timing protocol, ...
  (everything ``Job.definition`` holds; see bench.jobs);
- the harness its records were keyed on differs from this code's: the bench version,
  the vendored verdict code (parity_core), the environment the harness forces;
- the reference engines its records were keyed on differ from the versions the
  environment pins (pixi.toml; bench.fetch for BioNetGen and RuleMonkey).

A skipped release newer than the snapshot does not make it out of date; it is
noted beside its CURRENT status instead.

Everything but the PyPI releases is read from committed files -- problems/, the code,
pixi.toml, the snapshot's records -- never from the installed environment, so the
answer is the same on every machine. A job reads its reference engines' versions
from the environment it runs in (``reference_versions``, the Antimony converter's
``input.antimony``); here they are taken from what the environment pins instead.
A withheld timing has no record, so only its correctness job's definition is
compared (they share the model, protocol and harness).
"""

from __future__ import annotations

import json
import re
import tomllib
import urllib.request
from pathlib import Path
from typing import Any, Callable

from . import specs

CURRENT = "CURRENT"
OUT_OF_DATE = "OUT OF DATE"
NOT_PUBLISHED = "NOT PUBLISHED"

RELEASES_URL = "https://pypi.org/pypi/bngsim/json"
# bngsim releases we skip on purpose, each with why: {"0.17.3": "<reason and date>"}.
SKIPPED_FILE = "skipped-releases.json"

# A job's environment-read version -> the pixi.toml [pypi-dependencies] package pinning it.
PIXI_PACKAGES = {
    "bngsim": "bngsim",
    "roadrunner": "libroadrunner",
    "copasi": "python-copasi",
    "amici": "amici",
    "petab": "petab",
    "antimony": "antimony",
}
# What a definition's top-level key is called in a reason.
LABELS = {
    "source_sha256": "model file",
    "petab_sha256": "PEtab tables",
    "protocol": "protocol",
    "verdict": "verdict",
    "harness": "harness",
    "setting": "setting",
    "timing": "timing protocol",
    "timing_check": "timing check",
    "input": "input conversion",
    "accuracy": "accuracy check",
}
# Members named in one reason before the rest are counted.
NAMED = 4


# --------------------------------------------------------------------------- #
# What the environment pins, and the newest release
# --------------------------------------------------------------------------- #
def pins(root: Path) -> dict[str, Any]:
    """The versions the environment pins: exact ``==`` pins in pixi.toml's
    ``[pypi-dependencies]`` (None for one that is not an exact pin), and BioNetGen and
    RuleMonkey from bench.fetch."""
    from . import fetch

    data = tomllib.loads((Path(root) / "pixi.toml").read_text(encoding="utf-8"))
    pypi = data.get("pypi-dependencies") or {}
    out: dict[str, Any] = {}
    for name, pkg in PIXI_PACKAGES.items():
        spec = pypi.get(pkg)
        if isinstance(spec, dict):
            spec = spec.get("version")
        m = re.fullmatch(r"\s*==\s*([^\s,;]+)\s*", spec or "")
        out[name] = m.group(1) if m else None
    out["rulemonkey"] = {"version": fetch.RULEMONKEY_VERSION, "commit": fetch.RULEMONKEY_COMMIT}
    out["bionetgen"] = {"version": fetch.BIONETGEN_VERSION,
                        "sha256": sorted(fetch.BIONETGEN_SHA256.values())}
    return out


def release_key(version: str | None) -> tuple[int, ...] | None:
    """A final release's version as numbers (``0.17.3`` -> (0, 17, 3)); None for a
    pre-, post- or dev release, or anything else."""
    if not isinstance(version, str) or not re.fullmatch(r"\d+(\.\d+)*", version):
        return None
    return tuple(int(p) for p in version.split("."))


def final_releases(data: dict[str, Any]) -> list[str]:
    """Every final release in PyPI's JSON for a project that has a file not yanked,
    oldest first."""
    found = [v for v, files in (data.get("releases") or {}).items()
             if release_key(v) is not None and any(not f.get("yanked") for f in files or [])]
    return sorted(found, key=release_key)


def newest_release(data: dict[str, Any]) -> str | None:
    """The newest of ``final_releases``."""
    found = final_releases(data)
    return found[-1] if found else None


def look_up_releases(url: str = RELEASES_URL, timeout: float = 30.0,
                     opener: Callable[..., Any] | None = None) -> list[str]:
    """bngsim's final releases on PyPI, oldest first. Raises OSError (or ValueError)
    when they cannot be read."""
    from .fetch import USER_AGENT

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               "Accept": "application/json"})
    with (opener or urllib.request.urlopen)(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    found = final_releases(data)
    if not found:
        raise ValueError(f"{url} lists no final release")
    return found


def skipped_releases(published: Path) -> dict[str, str]:
    """The bngsim releases skipped on purpose (published/skipped-releases.json), each
    with its reason; none when there is no file. Raises ValueError for a file that is
    not a mapping of final release versions to reasons."""
    path = Path(published) / SKIPPED_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(data, dict):
        bad = ["not a mapping"]
    else:
        bad = [f"{k!r}: {'not a final release' if release_key(k) is None else 'no reason'}"
               for k, v in data.items()
               if release_key(k) is None or not (isinstance(v, str) and v.strip())]
    if bad:
        raise ValueError(f"{path}: a mapping of final bngsim releases to the reason each is "
                         "skipped; " + "; ".join(bad))
    return dict(data)


# --------------------------------------------------------------------------- #
# Comparing
# --------------------------------------------------------------------------- #
def _plain(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str))


def pinned_definition(definition: dict[str, Any], pinned: dict[str, Any]) -> dict[str, Any]:
    """``definition`` with what a job reads from the installed environment replaced
    by what the environment pins. Raises KeyError for a reference engine with no pin,
    so a new one cannot slip past the check."""
    d = _plain(definition)
    rv = d.get("reference_versions")
    if isinstance(rv, dict):
        for name in rv:
            if name not in pinned:
                raise KeyError(f"no pin for reference engine {name!r}: add it to "
                               "bench.staleness.PIXI_PACKAGES or pins()")
            rv[name] = _plain(pinned[name])
    inp = d.get("input")
    if isinstance(inp, dict) and "antimony" in inp:
        inp["antimony"] = pinned.get("antimony")
    return d


def _version(v: Any) -> str:
    if isinstance(v, dict):
        bits = [str(v.get("version"))] + ([str(v["commit"])[:12]] if v.get("commit") else [])
        return " ".join(bits)
    return str(v)


def _members(ids: list[str], total: int) -> str:
    ids = sorted(set(ids))
    if total and len(ids) == total:
        return f"all {total} members"
    shown = ", ".join(ids[:NAMED])
    return shown + (f" and {len(ids) - NAMED} more" if len(ids) > NAMED else "")


def _read_records(snapshot: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for entry in manifest.get("jobs") or []:
        if entry.get("file"):
            try:
                out[entry["job_id"]] = json.loads(
                    (snapshot / entry["file"]).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                out[entry["job_id"]] = {}
    return out


def release_check(measured: str | None, release: dict[str, Any] | None,
                  skipped: dict[str, str]) -> tuple[list[str], list[str]]:
    """``(reasons, notes)`` for a snapshot of bngsim ``measured``: the newest release on
    PyPI newer than it and not skipped is a reason; each skipped one newer than it, a
    note. ``release`` is what was seen on PyPI: ``version`` (the newest), ``releases``
    (every final release; just ``version`` when absent) and the day it was ``checked``."""
    if not release or not release.get("version"):
        return [], []
    old = release_key(measured)
    newer = [v for v in release.get("releases") or [release["version"]]
             if release_key(v) is not None and (old is None or release_key(v) > old)]
    newer.sort(key=release_key)
    kept = [v for v in newer if v not in skipped]
    reasons = [f"bngsim {kept[-1]} is on PyPI (snapshot: {measured})"] if kept else []
    notes = [f"bngsim {v} skipped: {skipped[v]}" for v in newer if v in skipped]
    return reasons, notes


def reasons(snapshot: Path, problems: list[specs.Problem], jobs: list, *,
            pins: dict[str, Any], release: dict[str, Any] | None,
            skipped: dict[str, str] | None = None) -> list[str]:
    """Why the snapshot at ``snapshot`` is out of date; empty when it is current."""
    return assess(snapshot, problems, jobs, pins=pins, release=release, skipped=skipped)[0]


def assess(snapshot: Path, problems: list[specs.Problem], jobs: list, *,
           pins: dict[str, Any], release: dict[str, Any] | None,
           skipped: dict[str, str] | None = None) -> tuple[list[str], list[str]]:
    """``(reasons, notes)`` for the snapshot at ``snapshot``: why it is out of date
    (empty when it is current), and what is worth saying beside a CURRENT status (a
    newer release skipped on purpose). ``jobs`` is every job the specs define
    (bench.jobs.jobs_for); ``release`` what was seen on PyPI (``release_check``), or
    None when it was not looked up; ``skipped`` the releases skipped on purpose."""
    from . import env

    snapshot = Path(snapshot)
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    suite = manifest["suite"]
    measured = (manifest.get("bngsim") or {}).get("version")

    # bngsim: the newest release not skipped, and the environment's pin.
    out, notes = release_check(measured, release, skipped or {})
    if pins.get("bngsim") and pins["bngsim"] != measured:
        out.append(f"the environment pins bngsim {pins['bngsim']} (snapshot: {measured})")

    # Members and their positions.
    if suite not in specs.SUITES:
        return out + [f"no suite {suite!r} is defined any more"], notes
    now = {p.id: p.suites[suite] for p in specs.suite_members(problems).get(suite) or []}
    then = {m["problem"]: m["position"] for m in manifest.get("members") or []}
    gone = sorted(set(then) - set(now), key=lambda i: then[i])
    new_ids = sorted(set(now) - set(then), key=lambda i: now[i])
    if gone or new_ids:
        bits = []
        if gone:
            bits.append("out " + ", ".join(f"{i} (#{then[i]})" for i in gone))
        if new_ids:
            bits.append("in " + ", ".join(f"{i} (#{now[i]})" for i in new_ids))
        out.append("members changed: " + "; ".join(bits))
    moved = sorted(i for i in set(now) & set(then) if now[i] != then[i])
    if moved:
        out.append("positions changed: " + ", ".join(f"{i} #{then[i]} -> #{now[i]}"
                                                     for i in moved))

    # Jobs of the members in both.
    kept = set(now) & set(then)
    cap = specs.SUITES[suite][0]
    current = {j.id: j for j in jobs if j.problem in kept and j.capability == cap}
    entries = {e["job_id"]: e for e in manifest.get("jobs") or [] if e.get("problem") in kept}
    added = sorted({current[i].problem for i in set(current) - set(entries)})
    dropped = sorted({entries[i]["problem"] for i in set(entries) - set(current)})
    if added or dropped:
        bits = ([f"new jobs for {_members(added, len(kept))}"] if added else []) + \
               ([f"jobs no longer defined for {_members(dropped, len(kept))}"]
                if dropped else [])
        out.append("jobs changed (engines or settings): " + "; ".join(bits))

    # Each recorded job's definition and environment, against now.
    records = _read_records(snapshot, manifest)
    changed: dict[str, set[str]] = {}
    engines: dict[str, tuple[str, str]] = {}
    unkeyed: set[str] = set()
    environment: dict[str, tuple[str, str]] = {}
    harness = _plain(env.harness_identity())
    for jid, rec in records.items():
        job = current.get(jid)
        if job is None:
            continue
        parts = rec.get("key_parts") or {}
        then_def = (parts.get("job") or {}).get("definition")
        if not isinstance(then_def, dict):
            unkeyed.add(job.problem)
            continue
        now_def = pinned_definition(job.definition, pins)
        then_def = _plain(then_def)
        for key in sorted(set(then_def) | set(now_def)):
            if then_def.get(key) == now_def.get(key):
                continue
            if key == "reference_versions":
                a, b = then_def.get(key) or {}, now_def.get(key) or {}
                for name in sorted(set(a) | set(b)):
                    if a.get(name) != b.get(name):
                        engines[name] = (_version(a.get(name)), _version(b.get(name)))
            else:
                changed.setdefault(LABELS.get(key, key), set()).add(job.problem)
        e = parts.get("environment") or {}
        bng = e.get("bionetgen") or {}
        pin = pins.get("bionetgen") or {}
        if bng.get("version") != pin.get("version"):
            environment["BioNetGen"] = (str(bng.get("version")), str(pin.get("version")))
        elif pin.get("sha256") and bng.get("sha256") not in pin["sha256"]:
            environment["BioNetGen"] = (f"{bng.get('version')} (another build)",
                                        str(pin.get("version")))
        h = _plain(e.get("harness") or {})
        if h.get("bench") != harness.get("bench"):
            environment["harness version"] = (f"bench {h.get('bench')}",
                                              f"bench {harness.get('bench')}")
        pc_then, pc_now = h.get("parity_core") or {}, harness.get("parity_core") or {}
        if pc_then != pc_now:
            environment["vendored verdict code"] = (
                f"{pc_then.get('ref')} ({str(pc_then.get('commit'))[:12]})",
                f"{pc_now.get('ref')} ({str(pc_now.get('commit'))[:12]})")
        if h.get("forced_env") != harness.get("forced_env"):
            environment["forced harness environment"] = (
                json.dumps(h.get("forced_env"), sort_keys=True),
                json.dumps(harness.get("forced_env"), sort_keys=True))
    for label in sorted(changed):
        out.append(f"{label} changed: {_members(sorted(changed[label]), len(kept))}")
    for name in sorted(engines):
        old, new = engines[name]
        out.append(f"the environment pins {name} {new} (snapshot: {old})")
    for name in sorted(environment):
        old, new = environment[name]
        out.append(f"{name} is now {new} (snapshot: {old})")
    if unkeyed:
        out.append(f"records without a key, so not comparable: {_members(sorted(unkeyed), 0)}")
    return out, notes
