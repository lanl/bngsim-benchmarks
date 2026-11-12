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
# models are provisional until the core is chosen and audited.
VERIFICATION_KINDS = ("author_file", "equations", "data", "theory", "structure")
PENDING = "pending"

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
    path: str = ""

    def protocol_for(self, capability: str) -> dict[str, Any]:
        return self.protocol[capability]


def _validate_ode_protocol(p: Any, where: str, errs: list[str]) -> None:
    if not isinstance(p, dict):
        errs.append(f"{where}: must be a mapping")
        return
    allowed = {"netgen", "t_start", "t_end", "n_steps", "rtol", "atol", "sparse"}
    for k in sorted(set(p) - allowed):
        errs.append(f"{where}: unknown key {k!r}")
    for k in ("netgen", "t_end", "n_steps", "rtol", "atol"):
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


# One validator per capability's protocol. A capability without one cannot be
# named in a spec.
PROTOCOL_VALIDATORS = {"ode": _validate_ode_protocol}


def validate(data: Any, path: Path | str = "<spec>") -> Problem:
    errs: list[str] = []
    if not isinstance(data, dict):
        raise SpecError(path, ["top level must be a mapping"])

    allowed = {
        "id", "layer", "title", "source", "publications", "verification",
        "capabilities", "protocol", "size",
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
            PROTOCOL_VALIDATORS[cap](protocol[cap], f"protocol.{cap}", errs)
    for cap in sorted(set(protocol) - set(caps)):
        errs.append(f"protocol.{cap}: not a listed capability")

    size = data.get("size", {})
    if not isinstance(size, dict) or not all(
        isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in size.values()
    ):
        errs.append("size: a mapping of non-negative integers")
        size = {}

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
    return sorted(problems, key=lambda p: p.id)
