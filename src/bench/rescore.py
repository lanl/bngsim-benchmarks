"""Re-score stored correctness results with the current verdict, without re-simulating.

A correctness job stores the trajectories its verdict was computed from beside its
record (``results/<job id>.npz``). When only the verdict has changed since -- the
capability's verdict version, a correctness-only part of the job definition, or
the vendored comparison core -- ``rescore`` recomputes the verdicts and the outcome
from those trajectories and rewrites the record with the current freshness key, so
it is current again exactly as a re-run would be. A record that is stale for any
other reason (another bngsim build, protocol, model, engine version) is not
touched: its trajectories are not what a run would give now, so it must re-run.
"""

from __future__ import annotations

import copy
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .jobs import CORRECTNESS, Job
from .results import (freshness_key, key_parts, read_result, read_trajectories,
                      write_result)

# The parts of a correctness key that only decide how stored trajectories are
# scored: the verdict version in the job definition, and the vendored comparison
# core in the harness identity. A record whose key differs from the current one
# only here can be re-scored; any other difference means its trajectories are
# not what a run would produce now.
SCORING_PARTS = (("job", "definition", "verdict"), ("environment", "harness", "parity_core"))


@dataclass(frozen=True)
class Rescored:
    job: Job
    state: str  # rescored | current | missing | not-rescorable | no-trajectories
    reason: str = ""
    before: str | None = None
    after: str | None = None


def _without_scoring(parts: dict[str, Any] | None) -> dict[str, Any]:
    out = copy.deepcopy(parts or {})
    for path in SCORING_PARTS:
        node = out
        for k in path[:-1]:
            node = node.get(k) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, dict):
            node.pop(path[-1], None)
    return out


def rescore_one(job: Job, results_dir: str | Path, fingerprint: dict[str, Any], *,
                force: bool = False, dry_run: bool = False) -> Rescored:
    """Re-score ``job``'s record, if only its scoring is out of date (or ``force``,
    if it is current). ``fingerprint`` is the current correctness fingerprint."""
    from .capabilities import CAPABILITIES

    cap = CAPABILITIES.get(job.capability)
    if job.kind != CORRECTNESS or not hasattr(cap, "rescore"):
        return Rescored(job, "not-rescorable", f"{job.kind} results are not re-scored")
    rec = read_result(results_dir, job)
    if rec is None:
        return Rescored(job, "missing")
    want = freshness_key(job, fingerprint)
    if rec.get("key") == want and not force:
        return Rescored(job, "current", before=rec.get("outcome"))
    new_parts = key_parts(job, fingerprint)
    if _without_scoring(rec.get("key_parts")) != _without_scoring(new_parts):
        from .results import _changed

        why = ", ".join(_changed(_without_scoring(rec.get("key_parts")),
                                 _without_scoring(new_parts)))
        return Rescored(job, "not-rescorable",
                        f"stale for more than its verdict ({why}): re-run it",
                        before=rec.get("outcome"))
    if not rec.get("verdicts") and not rec.get("trajectories"):
        # Nothing was scored (the job failed before any comparison: a model that
        # could not be prepared, a job stopped at its timeout), so no verdict took
        # part in its outcome, and the current one would give the same.
        update: dict[str, Any] = {}
    else:
        try:
            arrays = read_trajectories(results_dir, job, rec)
        except (LookupError, ValueError) as exc:
            return Rescored(job, "no-trajectories", f"{exc}: re-run it",
                            before=rec.get("outcome"))
        update = cap.rescore(job, rec, arrays)
    new = {**rec, **update, "key": want, "key_parts": new_parts}
    history = list(rec.get("rescored") or [])
    history.append({
        "at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "from_key": rec.get("key"),
        "from_outcome": rec.get("outcome"),
        "verdict": (job.definition or {}).get("verdict"),
        "parity_core": ((fingerprint.get("harness") or {}).get("parity_core")),
    })
    new["rescored"] = history
    if not dry_run:
        write_result(results_dir, job, new)
    return Rescored(job, "rescored", before=rec.get("outcome"), after=new.get("outcome"))


def rescore(jobs: list[Job], results_dir: str | Path, fingerprint: dict[str, Any], *,
            force: bool = False, dry_run: bool = False) -> list[Rescored]:
    """``rescore_one`` for every correctness job in ``jobs``."""
    return [rescore_one(j, results_dir, fingerprint, force=force, dry_run=dry_run)
            for j in jobs if j.kind == CORRECTNESS]
