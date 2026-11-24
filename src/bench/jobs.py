"""Jobs: the unit of work.

A job is one problem x one capability x one set of engines x one setting x one
kind (correctness or timing). Its id is deterministic and readable,

    <capability>/<problem>/<engines>/<setting>/<kind>
    ode/kinetic_proofreading_hlavacek2001/bngsim-vs-run_network-roadrunner/auto/correctness

and its ``definition`` is everything that decides what it computes. The job list
is generated from the problem specs, so it never drifts from them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .specs import Problem

CORRECTNESS = "correctness"
TIMING = "timing"
KINDS = (CORRECTNESS, TIMING)
# Between a job id and a part name in a task file (``Job.task_id``).
PART_SEP = "#"


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
    # Planning only, like est_seconds: the independent pieces of the job a runner
    # may run as processes of their own -- in separate Slurm array tasks, when the
    # whole job does not fit one allocation -- each with its estimated seconds.
    # A part stores what it computed beside the job's result (results.write_part);
    # the job itself then uses every current part instead of recomputing it, so
    # its result is the same whether or not it was split (an SSA correctness job's
    # parts are its engines' ensembles).
    parts: dict[str, float] = field(default_factory=dict)
    # Set on the copy of a job that runs one part only (``as_part``).
    part: str | None = None

    @property
    def id(self) -> str:
        return f"{self.capability}/{self.problem}/{self.engines}/{self.setting}/{self.kind}"

    @property
    def task_id(self) -> str:
        """The id a task file lists: the job's, or ``<id>#<part>`` for one part."""
        return self.id if self.part is None else f"{self.id}{PART_SEP}{self.part}"

    @property
    def remainder_seconds(self) -> float:
        """The estimated cost of the job once every part is done (its scoring)."""
        return max(1.0, self.est_seconds - sum(self.parts.values()))

    def as_part(self, name: str) -> "Job":
        """The copy of this job that runs part ``name`` only. Its estimate is the
        part's plus the remainder, since the run that finishes the last part also
        runs the job itself."""
        if name not in self.parts:
            raise ValueError(f"{self.id} has no part {name!r} (parts: {', '.join(self.parts)})")
        return dataclasses.replace(self, part=name,
                                   est_seconds=self.parts[name] + self.remainder_seconds)

    def whole(self) -> "Job":
        """The job itself, for a copy that runs one part."""
        if self.part is None:
            return self
        remainder = self.est_seconds - self.parts[self.part]
        return dataclasses.replace(self, part=None,
                                   est_seconds=sum(self.parts.values()) + remainder)

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

    An id may name one part of a job (``<id>#<part>``, as ``bench submit`` writes
    it for a job split across array tasks): the result then holds that part's copy
    of the job (``Job.as_part``) in the job's place, one copy per part named. A
    timing job listed beside parts of its correctness job does not bring in the
    whole correctness job: the runner runs that once its parts are done.

    Raises ValueError naming every id no spec defines.
    """
    by_id = {j.id: j for j in jobs}
    unknown = [i for i in ids if i.split(PART_SEP, 1)[0] not in by_id]
    if unknown:
        raise ValueError("no such job: " + ", ".join(unknown))
    chosen: set[str] = set()
    parts: dict[str, list[str]] = {}
    for tid in ids:
        jid, _, part = tid.partition(PART_SEP)
        if part:
            if part not in by_id[jid].parts:
                raise ValueError(f"no such part: {tid}")
            if part not in parts.setdefault(jid, []):
                parts[jid].append(part)
        else:
            chosen.add(jid)
    for jid in list(chosen) + list(parts):
        req = by_id[jid].requires
        if req and req not in parts:
            chosen.add(req)
    out: list[Job] = []
    for j in jobs:
        if j.id in parts and j.id not in chosen:
            out.extend(j.as_part(p) for p in parts[j.id])
        elif j.id in chosen:
            out.append(j)
    return out
