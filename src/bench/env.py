"""What a job ran on: the bngsim build, the reference engines, and the machine.

Every result records ``environment()``. The parts that decide whether an old
result still stands (``fingerprint``) are a subset: the bngsim build, the
reference engines, the harness, and for timing jobs the node type.
"""

from __future__ import annotations

import json
import os
import platform
import re
import socket
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import __version__
from .fetch import FETCH_MARKER

# Set in every job's environment: a timing job measures one thread unless it
# measures parallelism on purpose (docs/DESIGN.md section 7).
SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


@lru_cache(maxsize=1)
def bngsim_identity() -> dict[str, Any]:
    """Version and compiled build commit of the installed bngsim.

    The version string alone cannot identify a build: a development build
    reports the version of the last release. The build commit can.
    """
    try:
        import bngsim
    except Exception as exc:  # noqa: BLE001 - any import failure means "absent"
        return {"version": None, "build_commit": None, "error": f"{type(exc).__name__}: {exc}"}
    ident: dict[str, Any] = {"version": getattr(bngsim, "__version__", None), "build_commit": None}
    try:
        from bngsim import _build_provenance

        summary = _build_provenance.summary()
        ident["build_commit"] = summary.get("commit")
        ident["stale"] = summary.get("stale")
    except Exception as exc:  # noqa: BLE001
        ident["error"] = f"build provenance unavailable: {type(exc).__name__}: {exc}"
    return ident


def bngpath() -> Path | None:
    value = os.environ.get("BNGPATH")
    return Path(value) if value else None


@lru_cache(maxsize=1)
def bionetgen_identity() -> dict[str, Any]:
    """Which BioNetGen distribution BNGPATH names, and whether it is the fetched one."""
    root = bngpath()
    if root is None:
        return {"bngpath": None, "error": "BNGPATH is not set"}
    ident: dict[str, Any] = {"bngpath": str(root)}
    if not (root / "BNG2.pl").exists():
        ident["error"] = f"no BNG2.pl under {root}"
        return ident
    marker = root / FETCH_MARKER
    if marker.exists():
        try:
            m = json.loads(marker.read_text())
            ident.update(version=m.get("version"), sha256=m.get("sha256"), fetched=True)
        except (OSError, ValueError):
            ident["fetched"] = False
    else:
        # Found, but not installed by `bench fetch-bionetgen`: usable, recorded as
        # unverified, and never the same fingerprint as the pinned copy.
        ident["fetched"] = False
        ident["version"] = _read(str(root / "VERSION"))
    return ident


def bng2_pl() -> Path:
    root = bngpath()
    if root is None:
        raise RuntimeError("BNGPATH is not set; run `pixi run fetch-bionetgen` and use `pixi run`")
    return root / "BNG2.pl"


def run_network_bin() -> Path:
    root = bngpath()
    if root is None:
        raise RuntimeError("BNGPATH is not set; run `pixi run fetch-bionetgen` and use `pixi run`")
    return root / "bin" / "run_network"


def _cpu_model() -> str | None:
    if platform.system() == "Darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    text = _read("/proc/cpuinfo") or ""
    m = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else None


def _microcode() -> str | None:
    text = _read("/proc/cpuinfo") or ""
    m = re.search(r"^microcode\s*:\s*(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else None


@lru_cache(maxsize=1)
def host_identity() -> dict[str, Any]:
    """The machine, in enough detail to tell node types apart and to judge a timing."""
    cpu = _cpu_model()
    ident: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu_model": cpu,
        "logical_cpus": os.cpu_count(),
        "microcode": _microcode(),
        "governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "no_turbo": _read("/sys/devices/system/cpu/intel_pstate/no_turbo"),
        "boost": _read("/sys/devices/system/cpu/cpufreq/boost"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    if hasattr(os, "sched_getaffinity"):
        ident["usable_cpus"] = len(os.sched_getaffinity(0))
    return ident


def node_type() -> str:
    """The node type a timing result is tied to: CPU model, architecture, OS.

    Two nodes of one type give comparable timings; anything else does not.
    """
    h = host_identity()
    return f"{h['cpu_model'] or 'unknown-cpu'} | {h['machine']} | {h['system']}"


def harness_identity() -> dict[str, Any]:
    from ._vendor import parity_core_identity

    return {"bench": __version__, "parity_core": parity_core_identity()}


def environment() -> dict[str, Any]:
    """Everything a result records about where it ran."""
    return {
        "bngsim": bngsim_identity(),
        "bionetgen": bionetgen_identity(),
        "harness": harness_identity(),
        "host": host_identity(),
        "node_type": node_type(),
        "python": platform.python_version(),
    }


def fingerprint(*, timing: bool) -> dict[str, Any]:
    """The parts of the environment a result's freshness depends on.

    A correctness verdict holds on any machine; a timing holds only on the node
    type it was measured on.
    """
    b = bngsim_identity()
    g = bionetgen_identity()
    fp: dict[str, Any] = {
        "bngsim": {"version": b.get("version"), "build_commit": b.get("build_commit")},
        "bionetgen": {"version": g.get("version"), "sha256": g.get("sha256")},
        "harness": harness_identity(),
    }
    if timing:
        fp["node_type"] = node_type()
    return fp
