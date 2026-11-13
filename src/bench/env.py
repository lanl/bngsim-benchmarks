"""What a job ran on: the bngsim build, the reference engines, and the machine.

Every result records ``environment()``. The parts that decide whether an old
result still stands (``fingerprint``) are a subset: the bngsim build, BioNetGen,
the harness, and for timing jobs the node type. A reference engine used by only
some capabilities (libRoadRunner, for the SBML engines) is recorded here but
kept out of the global fingerprint and keyed instead inside the job definition
of the capabilities that use it, so changing it cannot stale a result that never
touched it.
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
from .fetch import FETCH_MARKER, GLIBC_MARKER, GLIBC_TAG

# Set in every job's environment: a timing job measures one thread unless it
# measures parallelism on purpose (docs/DESIGN.md section 7).
SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

# Forced in every job's environment, whatever the caller set. Without
# BNGSIM_CODEGEN_STRICT, a compiled-code build that fails falls back to the
# interpreter with only a warning, so a job could time or check the interpreter
# without knowing; with it, the job fails and says why (bngsim >= 0.17.1).
HARNESS_ENV = {
    "BNGSIM_CODEGEN_STRICT": "1",
}


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
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


@lru_cache(maxsize=1)
def roadrunner_identity() -> dict[str, Any]:
    """Version of the installed libRoadRunner, a reference ODE engine.

    The version is read from package metadata, so it does not pay the import.
    ``{version: None, error}`` when the package is absent -- the environment is
    pinned to install it, so that is a broken environment, reported as such.
    """
    try:
        from importlib.metadata import version

        return {"version": version("libroadrunner")}
    except Exception as exc:  # noqa: BLE001 - any failure means "absent/unreadable"
        return {"version": None, "error": f"{type(exc).__name__}: {exc}"}


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
            m = json.loads(marker.read_text(encoding="utf-8"))
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


def glibc_root() -> Path | None:
    """The fetched newer-glibc tree, if one is configured and present.

    ``$BENCH_GLIBC_ROOT`` overrides; otherwise it is derived from BNGPATH, whose
    grandparent is the cache dir (``.cache/bionetgen/BioNetGen-x`` ->
    ``.cache/glibc/<tag>``). ``None`` when no such tree exists, which is the
    normal case on a machine whose own glibc loads run_network.
    """
    override = os.environ.get("BENCH_GLIBC_ROOT")
    if override:
        root = Path(override)
        return root if (root / GLIBC_MARKER).exists() else None
    bng = bngpath()
    if bng is None:
        return None
    root = bng.parent.parent / "glibc" / GLIBC_TAG
    return root if (root / GLIBC_MARKER).exists() else None


def _glibc_launch() -> tuple[list[str], dict[str, Any]] | None:
    """The swapped-loader launch prefix and the marker it came from, or ``None``.

    The single source of truth for "is a usable loader swap configured": both the
    launch prefix (``run_network_launch``) and the recorded identity
    (``glibc_identity``) derive from it, so a result can never be labelled
    ``swapped`` while the binary in fact ran natively. ``None`` means no tree, an
    unreadable/incomplete marker, or a loader file that no longer exists -- every
    case in which no prefix can be built and the native loader is used.
    """
    root = glibc_root()
    if root is None:
        return None
    try:
        m = json.loads((root / GLIBC_MARKER).read_text(encoding="utf-8"))
        ld_so = root / m["ld_so"]
        libpath = os.pathsep.join(str(root / d) for d in m["library_path"])
    except (OSError, ValueError, KeyError):
        return None
    if not ld_so.exists():
        return None
    return [str(ld_so), "--library-path", libpath], m


@lru_cache(maxsize=1)
def glibc_identity() -> dict[str, Any]:
    """Whether run_network is launched under a swapped loader, and which one.

    ``swapped`` is true only when a prefix can actually be built, so it agrees
    with ``run_network_launch`` by construction.
    """
    launch = _glibc_launch()
    if launch is None:
        return {"swapped": False}
    m = launch[1]
    return {
        "swapped": True,
        "tag": m.get("tag"),
        "glibc_version": m.get("glibc_version"),
        "libstdcxx_version": m.get("libstdcxx_version"),
        "rpms": [r.get("sha256") for r in m.get("rpms", [])],
    }


@lru_cache(maxsize=1)
def run_network_launch() -> list[str]:
    """Argv prefix that runs run_network under the fetched loader, or ``[]``.

    ``[]`` means run the binary directly -- the normal case, and always so off
    Linux. When a glibc tree is present the prefix is
    ``[<ld.so>, "--library-path", <dirs>]``, which is identical and constant for
    every call, so it does not bias the paired timing (it is a real part of
    launching a fresh run_network process, which is what the reference measures).
    """
    launch = _glibc_launch()
    return launch[0] if launch is not None else []


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

    return {"bench": __version__, "parity_core": parity_core_identity(), "forced_env": HARNESS_ENV}


def environment() -> dict[str, Any]:
    """Everything a result records about where it ran."""
    import locale

    return {
        "bngsim": bngsim_identity(),
        "bionetgen": bionetgen_identity(),
        "roadrunner": roadrunner_identity(),
        "loader": glibc_identity(),
        "harness": harness_identity(),
        "host": host_identity(),
        "node_type": node_type(),
        "python": platform.python_version(),
        # The locale's text encoding, for provenance: the harness names UTF-8 at
        # each read, but a model read by BNGsim rides on this. Not in the
        # fingerprint -- PYTHONUTF8 makes it UTF-8 regardless, and it is recorded,
        # not keyed on.
        "text_encoding": locale.getpreferredencoding(False),
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
        # The reference's absolute speed (and the launch cost folded into its
        # measurement) rides on the libc/libstdc++ it runs under; node_type pins
        # the hardware, this pins the loader. A correctness verdict does not
        # depend on it, so it is a timing-only part of the key.
        loader = glibc_identity()
        if loader.get("swapped"):
            fp["loader"] = {
                "tag": loader.get("tag"),
                "glibc_version": loader.get("glibc_version"),
                "libstdcxx_version": loader.get("libstdcxx_version"),
            }
    return fp
