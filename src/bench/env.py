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
import shutil
import socket
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import __version__
from .fetch import (
    FETCH_MARKER,
    GLIBC_MARKER,
    GLIBC_TAG,
    RULEMONKEY_MARKER,
    RULEMONKEY_VERSION,
    sha256_file,
)

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


def roadrunner_source_path() -> Path | None:
    """A source-built libRoadRunner to use instead of an installed wheel, or ``None``.

    ``$BENCH_ROADRUNNER_PATH`` names the ``site-packages`` directory of a
    from-source build (the ``roadrunner`` package plus its synthesized
    ``libroadrunner-*.dist-info``). On a platform PyPI covers (linux-64,
    osx-arm64) the wheel is pinned in ``pixi.toml`` and this is unset; on
    linux-aarch64, where there is no wheel or sdist, the suite is pointed at a
    build (see ``docs/BUILD_NOTES.md``). This mirrors how ``BNGPATH`` names the
    external run_network build and ``BENCH_GLIBC_ROOT`` the loader tree: the
    tracked environment stays portable, and the one unprovided artifact is
    located by an env var. ``None`` when the var is unset or names nothing
    importable.
    """
    value = os.environ.get("BENCH_ROADRUNNER_PATH")
    if not value:
        return None
    root = Path(value)
    return root if (root / "roadrunner").is_dir() else None


def _ensure_roadrunner_on_path() -> None:
    """Put the ``$BENCH_ROADRUNNER_PATH`` build at the front of ``sys.path``.

    Idempotent, and in-process only: it edits this interpreter's ``sys.path``,
    never the environment the run_network subprocess inherits. Using ``sys.path``
    rather than ``$PYTHONPATH`` is deliberate -- on the ARM fleet the build notes
    require ``$PYTHONPATH`` to stay empty (x86 user-site numpy leaks onto the ARM
    interpreter through it), so the injection cannot ride on that variable.
    Prepended, not appended, so the build wins over any same-named package.
    """
    root = roadrunner_source_path()
    if root is None:
        return
    import sys

    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


@lru_cache(maxsize=1)
def roadrunner_identity() -> dict[str, Any]:
    """Version of the installed libRoadRunner, a reference ODE engine.

    The version is read from package metadata, so it does not pay the import.
    ``{version: None, error}`` when the package is absent -- the environment is
    pinned to install it, so that is a broken environment, reported as such.
    ``source`` is recorded when a ``$BENCH_ROADRUNNER_PATH`` build supplies it, so
    a result from a source build is never mistaken for one from the wheel.
    """
    _ensure_roadrunner_on_path()
    source = roadrunner_source_path()
    try:
        from importlib.metadata import version

        ident: dict[str, Any] = {"version": version("libroadrunner")}
    except Exception as exc:  # noqa: BLE001 - any failure means "absent/unreadable"
        ident = {"version": None, "error": f"{type(exc).__name__}: {exc}"}
    if source is not None:
        ident["source"] = str(source)
    return ident


@lru_cache(maxsize=1)
def copasi_identity() -> dict[str, Any]:
    """Versions of COPASI, a reference ODE engine driven through BASICO.

    ``version`` is COPASI itself (the ``python-copasi`` SWIG package that carries
    the native engine); ``basico`` is the pure-Python wrapper the adapter calls.
    Both read from package metadata without paying an import. ``{version: None,
    error}`` when absent -- the environment is pinned to install it, so that is a
    broken environment, reported as such.
    """
    try:
        from importlib.metadata import version

        ident: dict[str, Any] = {"version": version("python-copasi"),
                                 "basico": version("copasi-basico")}
    except Exception as exc:  # noqa: BLE001 - any failure means "absent/unreadable"
        ident = {"version": None, "error": f"{type(exc).__name__}: {exc}"}
    return ident


@lru_cache(maxsize=1)
def amici_identity() -> dict[str, Any]:
    """Version of the installed AMICI, a reference ODE engine (CVODES).

    Read from package metadata without paying the import (which pulls in the
    compiled SWIG core). ``{version: None, error}`` when absent -- the environment
    is pinned to install it, so that is a broken environment, reported as such.
    The pin is 1.1.0, held below the latest 1.2.0 on purpose (see ``pixi.toml``).
    """
    try:
        from importlib.metadata import version

        ident: dict[str, Any] = {"version": version("amici")}
    except Exception as exc:  # noqa: BLE001 - any failure means "absent/unreadable"
        ident = {"version": None, "error": f"{type(exc).__name__}: {exc}"}
    return ident


@lru_cache(maxsize=1)
def antimony_identity() -> dict[str, Any]:
    """Version of libantimony (the ``antimony`` package), which converts an
    ode-antimony member to Antimony at job time and back to SBML for the engines
    that read only SBML. Read from package metadata, without the import.
    ``{version: None, error}`` when absent."""
    try:
        from importlib.metadata import version

        ident: dict[str, Any] = {"version": version("antimony")}
    except Exception as exc:  # noqa: BLE001 - any failure means "absent/unreadable"
        ident = {"version": None, "error": f"{type(exc).__name__}: {exc}"}
    return ident


def _preload_libpython() -> None:
    """Make this interpreter's ``libpython`` findable by extensions that need it.

    libRoadRunner's wheel ships ``_roadrunner.so`` with ``NEEDED
    libpython3.x.so.1.0`` and a RUNPATH (``$ORIGIN:$ORIGIN/../lib64``) that does
    not point at the conda env's ``lib/``. conda's ``python`` is statically
    linked, so libpython is not already in the process, and the env's copy is not
    on the loader's search path -- the import then succeeds only on a host that
    happens to have a *system* libpython3.x (e.g. a developer box), and fails on a
    clean runner. Loading the env's own libpython with ``RTLD_GLOBAL`` puts its
    symbols in the global namespace and satisfies the dependency by soname,
    independent of the host. In-process, so it never touches the environment the
    ``run_network`` subprocess inherits. A no-op if the library is not found.
    """
    import ctypes
    import glob
    import sys
    import sysconfig

    libdir = sysconfig.get_config_var("LIBDIR")
    if not libdir:
        return
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    # Prefer the versioned soname (``libpython3.12.so.1.0``) over the dev symlink.
    cands = sorted(glob.glob(os.path.join(libdir, f"libpython{ver}.so*")),
                   key=lambda p: (".so." not in p, p))
    for cand in cands:
        try:
            ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue


def import_roadrunner():
    """Import and return the ``roadrunner`` module, preloading libpython first.

    The single entry point every caller uses, so the libpython workaround and
    the ``$BENCH_ROADRUNNER_PATH`` injection both live in one place. See
    :func:`_preload_libpython` and :func:`_ensure_roadrunner_on_path`.
    """
    _ensure_roadrunner_on_path()
    _preload_libpython()
    import roadrunner

    return roadrunner


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


# --------------------------------------------------------------------------- #
# The network-free reference engines: NFsim (bundled with BioNetGen) and RuleMonkey
# --------------------------------------------------------------------------- #
# The NFsim a BioNetGen release bundles (2.9.4's banner: "starting NFsim v2.0.1";
# `bench doctor` runs it and checks). A BioNetGen release pins its NFsim, so the
# version follows from the distribution, whose sha256 is in every fingerprint.
# 2.9.3 (NFsim 1.14.3) is kept for comparison runs only.
NFSIM_VERSION = {"2.9.4": "2.0.1", "2.9.3": "1.14.3"}
# The binaries BNG2.pl execs from ``$BNGPATH/bin``, which a loader swap must wrap.
LAUNCHED_BINARIES = ("NFsim", "run_network")


def nfsim_bin() -> Path:
    root = bngpath()
    if root is None:
        raise RuntimeError("BNGPATH is not set; run `pixi run fetch-bionetgen` and use `pixi run`")
    return root / "bin" / "NFsim"


@lru_cache(maxsize=1)
def nfsim_identity() -> dict[str, Any]:
    """The standalone NFsim: the binary BioNetGen bundles, its version and sha256."""
    root = bngpath()
    if root is None:
        return {"version": None, "error": "BNGPATH is not set"}
    exe = root / "bin" / "NFsim"
    if not exe.is_file():
        return {"version": None, "error": f"no NFsim under {root / 'bin'}"}
    bng = bionetgen_identity().get("version")
    return {"version": NFSIM_VERSION.get(str(bng)), "bundled_with": f"BioNetGen {bng}",
            "path": str(exe), "sha256": sha256_file(exe)}


def bng_launch_tree(dest: Path) -> Path:
    """A BNGPATH under which BNG2.pl runs its binaries the way the harness does.

    BNG2.pl execs ``$BNGPATH/bin/NFsim`` (and ``run_network``) itself, so the
    loader prefix of ``run_network_launch`` cannot be spliced into its command.
    Where a loader swap is configured, this builds at ``dest`` a mirror of the
    fetched distribution -- every entry a symlink to the original, except those
    two binaries, which are shell scripts that exec the unchanged binary under the
    swapped loader. Run BNG2.pl with ``BNGPATH=<the returned path>``. Where no swap
    is configured it returns the distribution itself and writes nothing. The
    mirror is rebuilt on every call, so it always follows the current loader tree.
    """
    import shlex

    root = bngpath()
    if root is None:
        raise RuntimeError("BNGPATH is not set; run `pixi run fetch-bionetgen` and use `pixi run`")
    prefix = run_network_launch()
    if not prefix:
        return root
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "bin").mkdir(parents=True)
    for entry in sorted(root.iterdir()):
        if entry.name != "bin":
            (dest / entry.name).symlink_to(entry)
    for exe in sorted((root / "bin").iterdir()):
        target = dest / "bin" / exe.name
        if exe.name in LAUNCHED_BINARIES:
            cmd = " ".join(shlex.quote(a) for a in [*prefix, str(exe)])
            target.write_text(
                "#!/bin/sh\n"
                f"# {exe.name} from {root}, unchanged, under the loader swap (bench.env).\n"
                f'exec {cmd} "$@"\n', encoding="utf-8")
            target.chmod(0o755)
        else:
            target.symlink_to(exe)
    return dest


def rulemonkey_root() -> Path | None:
    """The standalone RuleMonkey install, a directory holding ``bin/rm_driver``.

    ``$BENCH_RULEMONKEY_PATH`` names it when set; otherwise it is the build `bench
    fetch-rulemonkey` makes in the cache, found from BNGPATH the way the glibc tree
    is (``.cache/bionetgen/BioNetGen-x`` -> ``.cache/rulemonkey/RuleMonkey-<v>``).
    ``None`` when neither holds a driver.
    """
    value = os.environ.get("BENCH_RULEMONKEY_PATH")
    if value:
        root = Path(value)
    else:
        bng = bngpath()
        if bng is None:
            return None
        root = bng.parent.parent / "rulemonkey" / f"RuleMonkey-{RULEMONKEY_VERSION}"
    return root if (root / "bin" / "rm_driver").is_file() else None


def rulemonkey_bin() -> Path:
    root = rulemonkey_root()
    if root is None:
        raise RuntimeError("no rm_driver: run `bench fetch-rulemonkey` (or set "
                           "BENCH_RULEMONKEY_PATH to an install holding bin/rm_driver)")
    return root / "bin" / "rm_driver"


@lru_cache(maxsize=1)
def rulemonkey_identity() -> dict[str, Any]:
    """The standalone RuleMonkey: version, commit and toolchain from its build
    marker. ``built`` is false for a driver found without one (usable, recorded as
    unverified, and never keyed as the pinned build)."""
    root = rulemonkey_root()
    if root is None:
        return {"version": None,
                "error": "rm_driver not found (run `bench fetch-rulemonkey`, or set "
                         "BENCH_RULEMONKEY_PATH)"}
    exe = root / "bin" / "rm_driver"
    ident: dict[str, Any] = {"path": str(exe), "sha256": sha256_file(exe)}
    try:
        m = json.loads((root / RULEMONKEY_MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {**ident, "version": None, "built": False}
    ident.update(version=m.get("version"), commit=m.get("commit"),
                 compiler=(m.get("compiler") or {}).get("version"),
                 built=m.get("binary_sha256") == ident["sha256"])
    return ident


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


def read_harness_source() -> dict[str, Any]:
    """The harness checkout as it is now: its git commit, whether its tracked
    files have local changes, and the sha256 of its Python source
    (``source_sha256``, see ``source_digest``), which names the code exactly even
    when they do. Empty outside a git checkout."""
    root = Path(__file__).resolve().parents[2]

    def git(*argv: str) -> str | None:
        try:
            out = subprocess.run(["git", "-C", str(root), *argv], capture_output=True,
                                 text=True, timeout=30, check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if not commit:
        return {}
    status = git("status", "--porcelain", "--untracked-files=no")
    out: dict[str, Any] = {"commit": commit, "dirty": bool(status) if status is not None else None}
    try:
        out["source_sha256"] = source_digest()
    except OSError:
        pass
    return out


def source_digest() -> str:
    """sha256 of the harness's Python source: every ``.py`` file of the ``bench``
    package, by its path in the package and its bytes, in path order."""
    import hashlib

    pkg = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for path in sorted(pkg.rglob("*.py")):
        data = path.read_bytes()
        h.update(f"{path.relative_to(pkg).as_posix()}\0{len(data)}\0".encode())
        h.update(data)
    return h.hexdigest()


# The harness source this process runs, once read (capture_harness_source).
_HARNESS_SOURCE: dict[str, Any] | None = None


def capture_harness_source() -> dict[str, Any]:
    """Read the harness checkout now and keep it as the code this process runs.

    A worker calls this as its job starts, before it loads the job's code
    (``bench.worker``), so its result names the code the job ran: a long job's
    result is written hours later, by which time the checkout may have moved on."""
    global _HARNESS_SOURCE
    _HARNESS_SOURCE = read_harness_source()
    return _HARNESS_SOURCE


def harness_source() -> dict[str, Any]:
    """The harness source this process runs: as captured when its job started
    (``capture_harness_source``), else read now, once, and kept. Provenance only:
    it is recorded, never part of a freshness key (``harness_identity`` is)."""
    if _HARNESS_SOURCE is None:
        capture_harness_source()
    return _HARNESS_SOURCE


def environment() -> dict[str, Any]:
    """Everything a result records about where it ran."""
    import locale

    return {
        "bngsim": bngsim_identity(),
        "bionetgen": bionetgen_identity(),
        "roadrunner": roadrunner_identity(),
        "copasi": copasi_identity(),
        "amici": amici_identity(),
        "nfsim": nfsim_identity(),
        "rulemonkey": rulemonkey_identity(),
        "loader": glibc_identity(),
        "harness": harness_identity(),
        "harness_source": harness_source(),
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
