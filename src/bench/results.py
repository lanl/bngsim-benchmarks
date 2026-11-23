"""One result file per job, written atomically, and whether it is still current.

A result is current ("fresh") when its freshness key equals the key the job
would have now: the job's definition plus the environment fingerprint (bngsim
build, reference engines, harness, and for timing jobs the node type). Only jobs
without a current result run, so an interrupted run loses only the jobs that
were in flight, and a new bngsim build makes exactly its results stale.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .jobs import CORRECTNESS, TIMING, Job, digest

SCHEMA = 1

# A dependent job (timing) runs when its requirement (correctness) reached one of
# these outcomes. PASS is a clean cross-engine agreement; REFERENCE_FAILED means
# BNGsim ran and was anchored by the surviving references while another reference
# crashed -- that failure is recorded, not scored (parity_core.taxonomy), and
# there is still a BNGsim-and-survivors comparison worth timing. Any other outcome
# (DIFF/EXCEPTION/BAD_TEST/...) blocks the dependent job.
_REQUIREMENT_SATISFIED = frozenset({"PASS", "REFERENCE_FAILED"})


def result_path(results_dir: str | Path, job: Job | str) -> Path:
    jid = job.id if isinstance(job, Job) else job
    return Path(results_dir) / f"{jid}.json"


def trajectories_path(results_dir: str | Path, job: Job | str) -> Path:
    """Where a job's stored trajectories go: beside its JSON record, as ``.npz``."""
    return result_path(results_dir, job).with_suffix(".npz")


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_trajectories(results_dir: str | Path, job: Job, arrays: dict[str, Any]) -> dict[str, Any]:
    """Store ``arrays`` (name -> array) for ``job`` as one compressed ``.npz``,
    atomically, and return what its JSON record keeps about the file: its name
    (relative to the record), size and sha256. The record is the authority: a
    reader checks the sha256 before trusting the file, so an ``.npz`` left from
    another run of the job is never read as this one's."""
    import numpy as np

    path = trajectories_path(results_dir, job)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as f:
            # No object arrays, so the file reads back without pickle.
            np.savez_compressed(f, **{k: np.asarray(v) for k, v in arrays.items()})
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path),
            "arrays": sorted(arrays)}


def read_trajectories(results_dir: str | Path, job: Job | str,
                      record: dict[str, Any]) -> dict[str, Any]:
    """The arrays ``record`` stored, once the file is checked against it.

    Raises ``LookupError`` when the record stored none or the file is missing,
    and ``ValueError`` when the file on disk is not the one the record names."""
    import numpy as np

    meta = record.get("trajectories")
    if not meta:
        raise LookupError("the record stored no trajectories")
    path = trajectories_path(results_dir, job)
    if not path.exists():
        raise LookupError(f"{path.name} is missing")
    if _sha256(path) != meta.get("sha256"):
        raise ValueError(f"{path.name} is not the file its record names (sha256 differs)")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def drop_trajectories(results_dir: str | Path, job: Job) -> None:
    """Remove a job's stored trajectories, when its new result stores none."""
    trajectories_path(results_dir, job).unlink(missing_ok=True)


def key_parts(job: Job, fingerprint: dict[str, Any]) -> dict[str, Any]:
    return {"job": json.loads(job.canonical()), "environment": fingerprint}


def freshness_key(job: Job, fingerprint: dict[str, Any]) -> str:
    return digest(key_parts(job, fingerprint))


def write_result(results_dir: str | Path, job: Job, record: dict[str, Any]) -> Path:
    """Write ``record`` for ``job`` atomically: readers see the old file or the
    new one, never a partial write, even if the process is killed mid-write."""
    path = result_path(results_dir, job)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, sort_keys=True, default=str)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def read_result(results_dir: str | Path, job: Job | str) -> dict[str, Any] | None:
    path = result_path(results_dir, job)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _changed(old: dict[str, Any] | None, new: dict[str, Any]) -> list[str]:
    """Which top-level parts of the key differ, for a readable 'stale' reason."""
    if not old:
        return ["no recorded key"]
    out = []
    if old.get("job") != new["job"]:
        out.append("job definition")
    oe, ne = old.get("environment", {}), new["environment"]
    for k in sorted(set(oe) | set(ne)):
        if oe.get(k) != ne.get(k):
            out.append(k)
    return out or ["key"]


@dataclass(frozen=True)
class Status:
    job: Job
    state: str  # fresh | stale | missing | waiting | blocked
    reason: str = ""
    outcome: str | None = None


def status(
    job: Job,
    results_dir: str | Path,
    fingerprints: dict[str, dict[str, Any]],
    by_id: dict[str, Job],
) -> Status:
    """Where ``job`` stands. ``fingerprints`` maps CORRECTNESS/TIMING to the
    environment fingerprint for that kind of job."""
    fp = fingerprints[job.kind]
    rec = read_result(results_dir, job)
    want = freshness_key(job, fp)
    if rec is not None and rec.get("key") == want:
        return Status(job, "fresh", outcome=rec.get("outcome"))
    if job.requires is not None:
        req = by_id[job.requires]
        req_rec = read_result(results_dir, req)
        req_fresh = req_rec is not None and req_rec.get("key") == freshness_key(
            req, fingerprints[req.kind]
        )
        if not req_fresh:
            return Status(job, "waiting", f"needs a current {req.kind} result")
        if req_rec.get("outcome") not in _REQUIREMENT_SATISFIED:
            return Status(job, "blocked", f"its {req.kind} job is {req_rec.get('outcome')}")
    if rec is None:
        return Status(job, "missing")
    reason = ", ".join(_changed(rec.get("key_parts"), key_parts(job, fp)))
    return Status(job, "stale", f"changed: {reason}", outcome=rec.get("outcome"))


def fingerprints_now() -> dict[str, dict[str, Any]]:
    from . import env

    return {CORRECTNESS: env.fingerprint(timing=False), TIMING: env.fingerprint(timing=True)}
