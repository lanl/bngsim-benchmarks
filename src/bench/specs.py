"""Problem specifications: one YAML file per problem under ``problems/``.

A problem is a model file (pinned by repository, commit, path and sha256), the
publication it comes from, the evidence that the file reproduces that
publication, the capabilities it exercises, and a fixed protocol per capability.
See docs/DESIGN.md sections 4 and 5.1.

Loading a spec validates it completely and reports every problem at once, so a
broken spec is found when the job list is built, not hours into a run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

LAYERS = ("pilot", "standards", "core", "series", "breadth", "application")

# docs/DESIGN.md section 4.1. "pending" is allowed only in the pilot layer, whose
# models are provisional until the core suites are chosen and audited.
VERIFICATION_KINDS = ("author_file", "equations", "data", "theory", "structure")
PENDING = "pending"

# docs/DESIGN.md section 3.2. A core suite is one feature on one input format,
# holding exactly SUITE_SIZE problems in a fixed order (positions 1..SUITE_SIZE).
# Each entry: the capability its members exercise, and the source formats it takes.
# A suite is registered when its members are committed, so every registered suite
# that appears in a set of specs must be complete.
SUITE_SIZE = 10
SUITES: dict[str, tuple[str, tuple[str, ...]]] = {
    "ode-net": ("ode", ("bngl",)),  # ODE on the .net generated from published BNGL
    "ode-sbml": ("ode", ("sbml",)),  # ODE on published SBML, read as it is
}

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.=@-]*$")
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class SpecError(ValueError):
    """A spec that does not validate; ``problems`` lists every reason."""

    def __init__(self, path: Path | str, problems: list[str]):
        self.path = str(path)
        self.problems = problems
        super().__init__(f"{path}:\n  - " + "\n  - ".join(problems))


@dataclass(frozen=True)
class Problem:
    id: str
    layer: str
    title: str
    source: dict[str, str]
    publications: tuple[dict[str, str], ...]
    verification: dict[str, str]
    capabilities: tuple[str, ...]
    protocol: dict[str, dict[str, Any]]
    size: dict[str, int] = field(default_factory=dict)
    suites: dict[str, int] = field(default_factory=dict)
    path: str = ""

    def protocol_for(self, capability: str) -> dict[str, Any]:
        return self.protocol[capability]


def _validate_ode_protocol(p: Any, where: str, errs: list[str],
                           source_format: str | None = None) -> None:
    """The ODE protocol. A BNGL model names its network generation (``netgen``)
    and may ask for run_network's ``sparse`` solver. An SBML model has neither:
    it is read as it is, and its protocol is its SED-ML time course, whose output
    may start (``output_start``) after the integration does (``t_start``)."""
    if not isinstance(p, dict):
        errs.append(f"{where}: must be a mapping")
        return
    if source_format == "sbml":
        allowed = {"t_start", "output_start", "t_end", "n_steps", "rtol", "atol"}
        required: tuple[str, ...] = ("t_end", "n_steps", "rtol", "atol")
    else:
        allowed = {"netgen", "t_start", "t_end", "n_steps", "rtol", "atol", "sparse"}
        required = ("netgen", "t_end", "n_steps", "rtol", "atol")
    for k in sorted(set(p) - allowed):
        errs.append(f"{where}: unknown key {k!r}"
                    + (" (an SBML model is read as it is)" if source_format == "sbml"
                       and k in ("netgen", "sparse") else ""))
    for k in required:
        if k not in p:
            errs.append(f"{where}: missing {k!r}")
    netgen = p.get("netgen")
    if netgen is not None and not (
        isinstance(netgen, str) and netgen.strip().startswith("generate_network(")
    ):
        errs.append(f"{where}.netgen: must be a generate_network(...) call")
    t_start = p.get("t_start", 0)
    t_end = p.get("t_end")
    if not isinstance(t_start, (int, float)) or isinstance(t_start, bool):
        errs.append(f"{where}.t_start: must be a number")
    elif isinstance(t_end, (int, float)) and not isinstance(t_end, bool):
        if not t_end > t_start:
            errs.append(f"{where}: t_end must be greater than t_start")
    elif t_end is not None:
        errs.append(f"{where}.t_end: must be a number")
    n_steps = p.get("n_steps")
    if n_steps is not None and (
        not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 1
    ):
        errs.append(f"{where}.n_steps: must be a positive integer")
    for k in ("rtol", "atol"):
        v = p.get(k)
        if v is not None and (
            not isinstance(v, (int, float)) or isinstance(v, bool) or not v > 0
        ):
            errs.append(f"{where}.{k}: must be a positive number")
    if "sparse" in p and not isinstance(p["sparse"], bool):
        errs.append(f"{where}.sparse: must be true or false")
    if "output_start" in p:
        out = p["output_start"]
        if not isinstance(out, (int, float)) or isinstance(out, bool):
            errs.append(f"{where}.output_start: must be a number")
        elif (isinstance(t_start, (int, float)) and not isinstance(t_start, bool)
              and isinstance(t_end, (int, float)) and not isinstance(t_end, bool)
              and not t_start <= out < t_end):
            errs.append(f"{where}.output_start: must be at least t_start and less than t_end")


def _int_in(v: Any, lo: int, hi: int) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi


# DSMTS's advice for a test against exact moments, which rests on the central limit
# theorem (bench.stochastic.analytic_verdict).
SSA_MOMENTS_MIN_REPLICATES = 10_000


def _validate_ssa_protocol(p: Any, where: str, errs: list[str],
                           source_format: str | None = None) -> None:
    """Exact SSA on the generated network: the grid ``0..t_end`` in ``n_steps``, the
    ensemble size per engine for correctness (``replicates``) and per timed
    repetition (``timing_replicates``), the base ``seed`` every engine's seeds derive
    from, and optionally ``moments: linear`` -- exact moments of a first-order network
    as a further reference (bench.capabilities.ssa)."""
    if not isinstance(p, dict):
        errs.append(f"{where}: must be a mapping")
        return
    allowed = {"netgen", "t_end", "n_steps", "replicates", "timing_replicates", "seed", "moments"}
    for k in sorted(set(p) - allowed):
        errs.append(f"{where}: unknown key {k!r}")
    for k in ("netgen", "t_end", "n_steps", "replicates"):
        if k not in p:
            errs.append(f"{where}: missing {k!r}")
    netgen = p.get("netgen")
    if netgen is not None and not (
        isinstance(netgen, str) and netgen.strip().startswith("generate_network(")
    ):
        errs.append(f"{where}.netgen: must be a generate_network(...) call")
    t_end = p.get("t_end")
    if t_end is not None and (
        not isinstance(t_end, (int, float)) or isinstance(t_end, bool) or not t_end > 0
    ):
        errs.append(f"{where}.t_end: must be a positive number (SSA runs from t = 0)")
    if "n_steps" in p and not _int_in(p["n_steps"], 1, 10**6):
        errs.append(f"{where}.n_steps: must be a positive integer")
    # Each engine's seeds come from a block of 10**7 (ssa.SEED_BLOCK), and COPASI's
    # seed is a C int: these bounds keep every block disjoint and in range.
    if "replicates" in p and not _int_in(p["replicates"], 100, 10**6):
        errs.append(f"{where}.replicates: an integer from 100 to 1000000")
    if "timing_replicates" in p and not _int_in(p["timing_replicates"], 1, 10**6):
        errs.append(f"{where}.timing_replicates: an integer from 1 to 1000000")
    if "seed" in p and not _int_in(p["seed"], 0, 2**30):
        errs.append(f"{where}.seed: an integer from 0 to 2**30")
    if "moments" in p:
        if p["moments"] != "linear":
            errs.append(f"{where}.moments: 'linear' (exact moments of a first-order network)")
        elif _int_in(p.get("replicates"), 0, SSA_MOMENTS_MIN_REPLICATES - 1):
            errs.append(f"{where}: a test against exact moments needs replicates >= "
                        f"{SSA_MOMENTS_MIN_REPLICATES}")


def _number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# The most points a steady-state scan may ask for: each is a solve per engine.
STEADY_MAX_POINTS = 1000
_PARAM_NAME_RE = re.compile(r"^[A-Za-z_]\w*$")


def _validate_steady_protocol(p: Any, where: str, errs: list[str],
                              source_format: str | None = None) -> None:
    """Steady state of the generated network, from its seed species
    (bench.capabilities.steady): the integration horizon ``t_end`` and its output
    grid ``n_steps`` for the integration references, the integrators' ``rtol`` and
    ``atol``, parameters set for every point (``parameters``), and optionally a
    dose-response ``scan`` of one parameter -- explicit ``values``, or ``min``,
    ``max`` and ``points``, spaced on a ``log`` scale or linearly."""
    if not isinstance(p, dict):
        errs.append(f"{where}: must be a mapping")
        return
    allowed = {"netgen", "t_end", "n_steps", "rtol", "atol", "parameters", "scan"}
    for k in sorted(set(p) - allowed):
        errs.append(f"{where}: unknown key {k!r}")
    for k in ("netgen", "t_end", "rtol", "atol"):
        if k not in p:
            errs.append(f"{where}: missing {k!r}")
    netgen = p.get("netgen")
    if netgen is not None and not (
        isinstance(netgen, str) and netgen.strip().startswith("generate_network(")
    ):
        errs.append(f"{where}.netgen: must be a generate_network(...) call")
    if "t_end" in p and not (_number(p["t_end"]) and p["t_end"] > 0):
        errs.append(f"{where}.t_end: must be a positive number (integration runs from t = 0)")
    # The integration references are judged settled on the second half of their
    # output, so the grid needs a point inside it besides the last.
    if "n_steps" in p and not _int_in(p["n_steps"], 2, 10**5):
        errs.append(f"{where}.n_steps: an integer from 2 to 100000")
    for k in ("rtol", "atol"):
        if k in p and not (_number(p[k]) and p[k] > 0):
            errs.append(f"{where}.{k}: must be a positive number")
    params = p.get("parameters", {})
    if not isinstance(params, dict):
        errs.append(f"{where}.parameters: a mapping of parameter name to value")
        params = {}
    for name, v in params.items():
        if not (isinstance(name, str) and _PARAM_NAME_RE.match(name)):
            errs.append(f"{where}.parameters: {name!r} is not a parameter name")
        if not _number(v):
            errs.append(f"{where}.parameters.{name}: must be a number")
    scan = p.get("scan")
    if scan is None:
        return
    if not isinstance(scan, dict):
        errs.append(f"{where}.scan: a mapping (parameter, and values or min/max/points)")
        return
    for k in sorted(set(scan) - {"parameter", "values", "min", "max", "points", "log"}):
        errs.append(f"{where}.scan: unknown key {k!r}")
    name = scan.get("parameter")
    if not (isinstance(name, str) and _PARAM_NAME_RE.match(name)):
        errs.append(f"{where}.scan.parameter: the name of the parameter scanned")
    elif name in params:
        errs.append(f"{where}.scan.parameter: {name!r} is also set in parameters")
    if "values" in scan:
        if any(k in scan for k in ("min", "max", "points", "log")):
            errs.append(f"{where}.scan: values, or min/max/points/log, not both")
        vals = scan["values"]
        if not (isinstance(vals, list) and 1 <= len(vals) <= STEADY_MAX_POINTS
                and all(_number(v) for v in vals)):
            errs.append(f"{where}.scan.values: a list of 1 to {STEADY_MAX_POINTS} numbers")
        elif len(set(vals)) != len(vals):
            errs.append(f"{where}.scan.values: each value once")
        return
    for k in ("min", "max", "points"):
        if k not in scan:
            errs.append(f"{where}.scan: missing {k!r} (or give values)")
    lo, hi = scan.get("min"), scan.get("max")
    if ("min" in scan and not _number(lo)) or ("max" in scan and not _number(hi)):
        errs.append(f"{where}.scan: min and max must be numbers")
    elif _number(lo) and _number(hi) and not hi > lo:
        errs.append(f"{where}.scan: max must be greater than min")
    if "points" in scan and not _int_in(scan["points"], 2, STEADY_MAX_POINTS):
        errs.append(f"{where}.scan.points: an integer from 2 to {STEADY_MAX_POINTS}")
    if "log" in scan and not isinstance(scan["log"], bool):
        errs.append(f"{where}.scan.log: must be true or false")
    elif scan.get("log") and _number(lo) and not lo > 0:
        errs.append(f"{where}.scan: a log scale needs min > 0")


# Network-free ensembles: the most replicates (each engine's seeds come from a block
# of 10**7, with 10**6 per later network-free step; bench.capabilities.nf).
NF_MAX_REPLICATES = 10**6


def _validate_nf_protocol(p: Any, where: str, errs: list[str],
                          source_format: str | None = None) -> None:
    """Network-free simulation (bench.capabilities.nf): the protocol BNG2.pl runs
    for the NFsim reference and every other engine replays -- ``actions``, the
    file's own action block (``own``) or a list of BNGL actions, holding
    network-free ``simulate`` steps and ``setParameter`` (bench.nfactions); the
    ensemble size per engine for correctness (``replicates``) and per timed
    repetition (``timing_replicates``); the base ``seed``; and, for a model whose
    network is finite, the ``netgen`` call that makes exact SSA on it a further
    reference. The model must be BNGL: BNG2.pl writes the BNG-XML every engine reads."""
    from . import nfactions

    if not isinstance(p, dict):
        errs.append(f"{where}: must be a mapping")
        return
    if source_format not in (None, "bngl"):
        errs.append(f"{where}: network-free simulation needs a BNGL source "
                    "(BNG2.pl writes the BNG-XML every engine reads)")
    allowed = {"actions", "replicates", "timing_replicates", "seed", "netgen"}
    for k in sorted(set(p) - allowed):
        errs.append(f"{where}: unknown key {k!r}")
    for k in ("actions", "replicates"):
        if k not in p:
            errs.append(f"{where}: missing {k!r}")
    actions = p.get("actions")
    if actions is not None and actions != "own":
        if not (isinstance(actions, str) and actions.strip()):
            errs.append(f"{where}.actions: 'own' (the file's action block) or BNGL actions")
        else:
            try:
                nfactions.protocol(actions)
            except nfactions.ActionError as exc:
                errs.append(f"{where}.actions: {exc}")
    netgen = p.get("netgen")
    if netgen is not None and not (
        isinstance(netgen, str) and netgen.strip().startswith("generate_network(")
    ):
        errs.append(f"{where}.netgen: must be a generate_network(...) call")
    if "replicates" in p and not _int_in(p["replicates"], 100, NF_MAX_REPLICATES):
        errs.append(f"{where}.replicates: an integer from 100 to {NF_MAX_REPLICATES}")
    if "timing_replicates" in p and not _int_in(p["timing_replicates"], 1, NF_MAX_REPLICATES):
        errs.append(f"{where}.timing_replicates: an integer from 1 to {NF_MAX_REPLICATES}")
    if "seed" in p and not _int_in(p["seed"], 0, 2**30):
        errs.append(f"{where}.seed: an integer from 0 to 2**30")


# One validator per capability's protocol. A capability without one cannot be
# named in a spec. Each is called with the source's format (``source_format``);
# one whose capability runs on a single input format may ignore it.
PROTOCOL_VALIDATORS = {
    "ode": _validate_ode_protocol,
    "ssa": _validate_ssa_protocol,
    "steady_state": _validate_steady_protocol,
    "nf": _validate_nf_protocol,
}


def validate(data: Any, path: Path | str = "<spec>") -> Problem:
    errs: list[str] = []
    if not isinstance(data, dict):
        raise SpecError(path, ["top level must be a mapping"])

    allowed = {
        "id", "layer", "title", "source", "publications", "verification",
        "capabilities", "protocol", "size", "suites",
    }
    for k in sorted(set(data) - allowed):
        errs.append(f"unknown key {k!r}")

    pid = data.get("id")
    if not isinstance(pid, str) or not _ID_RE.match(pid):
        errs.append("id: lowercase letters, digits and _.=@- only, starting with a letter or digit")
    elif Path(path).suffix in (".yaml", ".yml") and Path(path).stem != pid:
        errs.append(f"id {pid!r} must match the file name {Path(path).name!r}")

    layer = data.get("layer")
    if layer not in LAYERS:
        errs.append(f"layer: one of {', '.join(LAYERS)}")

    title = data.get("title")
    if not isinstance(title, str) or not title.strip():
        errs.append("title: required")

    source = data.get("source")
    if not isinstance(source, dict):
        errs.append("source: required mapping (repo, commit, path, sha256)")
        source = {}
    else:
        if not _REPO_RE.match(str(source.get("repo", ""))):
            errs.append("source.repo: GitHub owner/name")
        if not _SHA1_RE.match(str(source.get("commit", ""))):
            errs.append("source.commit: a full 40-character commit hash")
        spath = source.get("path")
        if not isinstance(spath, str) or not spath or spath.startswith("/") or ".." in spath:
            errs.append("source.path: a relative path inside the repository")
        if not _SHA256_RE.match(str(source.get("sha256", ""))):
            errs.append("source.sha256: 64 lowercase hex characters")
        if source.get("format") not in ("bngl", "sbml", "net"):
            errs.append("source.format: bngl, sbml or net")

    pubs = data.get("publications")
    if not isinstance(pubs, list) or not pubs:
        errs.append("publications: at least one, each with a citation")
        pubs = []
    else:
        for i, pub in enumerate(pubs):
            if not isinstance(pub, dict) or not str(pub.get("citation", "")).strip():
                errs.append(f"publications[{i}]: needs a citation")

    ver = data.get("verification")
    if not isinstance(ver, dict):
        errs.append("verification: required mapping (kind, evidence)")
        ver = {}
    else:
        kind = ver.get("kind")
        if kind == PENDING:
            if layer != "pilot":
                errs.append("verification.kind: 'pending' is allowed only in the pilot layer")
        elif kind not in VERIFICATION_KINDS:
            errs.append(f"verification.kind: one of {', '.join(VERIFICATION_KINDS)}")
        if kind == "structure" and not str(ver.get("reason", "")).strip():
            errs.append("verification: kind 'structure' needs a written reason")
        if not str(ver.get("evidence", "")).strip():
            errs.append("verification.evidence: where the evidence lives")

    caps = data.get("capabilities")
    if not isinstance(caps, list) or not caps:
        errs.append("capabilities: a non-empty list")
        caps = []
    protocol = data.get("protocol")
    if not isinstance(protocol, dict):
        errs.append("protocol: a mapping with one entry per capability")
        protocol = {}
    for cap in caps:
        if cap not in PROTOCOL_VALIDATORS:
            errs.append(f"capabilities: unknown capability {cap!r}")
        elif cap not in protocol:
            errs.append(f"protocol: missing an entry for capability {cap!r}")
        else:
            PROTOCOL_VALIDATORS[cap](protocol[cap], f"protocol.{cap}", errs,
                                     source_format=source.get("format"))
    for cap in sorted(set(protocol) - set(caps)):
        errs.append(f"protocol.{cap}: not a listed capability")

    size = data.get("size", {})
    if not isinstance(size, dict) or not all(
        isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in size.values()
    ):
        errs.append("size: a mapping of non-negative integers")
        size = {}

    suites = data.get("suites", {})
    if not isinstance(suites, dict):
        errs.append("suites: a mapping of suite name to position")
        suites = {}
    for name, pos in suites.items():
        if name not in SUITES:
            errs.append(f"suites: unknown suite {name!r} (one of {', '.join(SUITES)})")
            continue
        cap, formats = SUITES[name]
        if layer != "core":
            errs.append(f"suites.{name}: only a core problem may belong to a suite")
        if cap not in caps:
            errs.append(f"suites.{name}: a member must list capability {cap!r}")
        if isinstance(source, dict) and source.get("format") not in formats:
            errs.append(f"suites.{name}: source.format must be one of {', '.join(formats)}")
        if not isinstance(pos, int) or isinstance(pos, bool) or not 1 <= pos <= SUITE_SIZE:
            errs.append(f"suites.{name}: position must be an integer from 1 to {SUITE_SIZE}")

    if errs:
        raise SpecError(path, errs)
    return Problem(
        id=pid,
        layer=layer,
        title=title.strip(),
        source={k: str(v) for k, v in source.items()},
        publications=tuple({k: str(v) for k, v in p.items()} for p in pubs),
        verification={k: str(v) for k, v in ver.items()},
        capabilities=tuple(caps),
        protocol={k: dict(v) for k, v in protocol.items()},
        size=dict(size),
        suites=dict(suites),
        path=str(path),
    )


def load(path: str | Path) -> Problem:
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SpecError(path, [f"not valid YAML: {exc}"]) from exc
    return validate(data, path)


def load_all(root: str | Path) -> list[Problem]:
    """Every spec under ``root``, sorted by id. Raises on the first invalid file
    only after collecting all of them, so one run reports every broken spec."""
    root = Path(root)
    problems: list[Problem] = []
    errors: list[SpecError] = []
    for path in sorted(root.rglob("*.yaml")):
        try:
            problems.append(load(path))
        except SpecError as exc:
            errors.append(exc)
    if errors:
        raise SpecError(root, [str(e) for e in errors])
    seen: dict[str, str] = {}
    dups = []
    for p in problems:
        if p.id in seen:
            dups.append(f"duplicate id {p.id!r}: {seen[p.id]} and {p.path}")
        seen[p.id] = p.path
    if dups:
        raise SpecError(root, dups)
    suite_errs = check_suites(problems)
    if suite_errs:
        raise SpecError(root, suite_errs)
    return sorted(problems, key=lambda p: p.id)


def suite_members(problems: list[Problem]) -> dict[str, list[Problem]]:
    """Each suite named by ``problems``, its members in position order."""
    out: dict[str, list[Problem]] = {}
    for p in problems:
        for name in p.suites:
            out.setdefault(name, []).append(p)
    return {name: sorted(ms, key=lambda p: p.suites[name]) for name, ms in sorted(out.items())}


def check_suites(problems: list[Problem]) -> list[str]:
    """Every suite the problems name holds exactly SUITE_SIZE of them, one at
    each position 1..SUITE_SIZE (docs/DESIGN.md section 3.2)."""
    errs: list[str] = []
    for name, members in suite_members(problems).items():
        by_pos: dict[int, list[str]] = {}
        for p in members:
            by_pos.setdefault(p.suites[name], []).append(p.id)
        for pos, ids in sorted(by_pos.items()):
            if len(ids) > 1:
                errs.append(f"suite {name!r}: position {pos} is held by {', '.join(ids)}")
        missing = [i for i in range(1, SUITE_SIZE + 1) if i not in by_pos]
        if missing:
            errs.append(f"suite {name!r}: {len(members)} of {SUITE_SIZE} members; "
                        f"no member at position {', '.join(map(str, missing))}")
    return errs
