"""Jobs: the unit of work.

A job is one problem x one capability x one set of engines x one setting x one
kind (correctness or timing). Its id is deterministic and readable,

    <capability>/<problem>/<engines>/<setting>/<kind>
    ode/kinetic_proofreading_hlavacek2001/bngsim-vs-run_network-roadrunner/auto/correctness

and its ``definition`` is everything that decides what it computes. The job list
is generated from the problem specs, so it never drifts from them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .specs import Problem

CORRECTNESS = "correctness"
TIMING = "timing"
KINDS = (CORRECTNESS, TIMING)


@dataclass(frozen=True)
class Job:
    capability: str
    problem: str
    engines: str
    setting: str
    kind: str
    # Everything that determines the computation: the model's sha256, the
    # protocol, the harness version, the setting's parameters. Part of the
    # freshness key, so changing any of it makes the old result stale.
    definition: dict[str, Any] = field(default_factory=dict)
    # Planning only (deadline packing, Slurm array sizing); never part of a key.
    est_seconds: float = 60.0
    timeout_seconds: float = 1800.0
    # A timing job runs only after this correctness job passed on the same build.
    requires: str | None = None

    @property
    def id(self) -> str:
        return f"{self.capability}/{self.problem}/{self.engines}/{self.setting}/{self.kind}"

    def canonical(self) -> str:
        return canonical_json({"id": self.id, "definition": self.definition})


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def jobs_for(problems: list[Problem]) -> list[Job]:
    """Every job the specs define, correctness before timing, in a stable order."""
    from .capabilities import CAPABILITIES

    out: list[Job] = []
    for p in problems:
        for cap in p.capabilities:
            out.extend(CAPABILITIES[cap].jobs(p))
    ids = [j.id for j in out]
    dups = sorted({i for i in ids if ids.count(i) > 1})
    if dups:
        raise ValueError(f"duplicate job ids: {dups}")
    by_id = {j.id: j for j in out}
    for j in out:
        if j.requires is not None and j.requires not in by_id:
            raise ValueError(f"{j.id} requires {j.requires}, which no spec defines")
    return sorted(out, key=lambda j: (KINDS.index(j.kind), j.id))


def select(jobs: list[Job], patterns: list[str] | None) -> list[Job]:
    """Jobs whose id contains any of ``patterns`` (all jobs when none are given).

    Selecting a timing job also selects the correctness job it requires.
    """
    if not patterns:
        return list(jobs)
    chosen = {j.id for j in jobs if any(p in j.id for p in patterns)}
    by_id = {j.id: j for j in jobs}
    for jid in list(chosen):
        req = by_id[jid].requires
        if req:
            chosen.add(req)
    return [j for j in jobs if j.id in chosen]


def select_exact(jobs: list[Job], ids: list[str]) -> list[Job]:
    """The jobs named in ``ids`` exactly, plus the correctness jobs they require
    (a runner needs those to judge a timing job; a current one is not re-run).

    Raises ValueError naming every id no spec defines.
    """
    by_id = {j.id: j for j in jobs}
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        raise ValueError("no such job: " + ", ".join(unknown))
    chosen = set(ids)
    for jid in ids:
        req = by_id[jid].requires
        if req:
            chosen.add(req)
    return [j for j in jobs if j.id in chosen]
