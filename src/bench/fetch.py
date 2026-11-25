"""Downloads, each pinned by sha256: the BioNetGen distribution, a glibc loader
for old-glibc Linux hosts, and model files.

Nothing a job reads is taken on trust from the network or from a local checkout:
every file is identified by its sha256, and a download that does not match is
discarded rather than used.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

BIONETGEN_VERSION = "2.9.4"
_BIONETGEN_BASE = (
    "https://github.com/RuleWorld/bionetgen/releases/download/"
    f"BioNetGen-{BIONETGEN_VERSION}/BioNetGen-{BIONETGEN_VERSION}"
)
# sha256 of the published release assets (GitHub's own digests of them agree).
# 2.9.4 is the first release to bundle NFsim 2 (2.0.1); 2.9.3, with NFsim 1.14.3,
# was: mac 5ddf0ebf76b9573bbfef794bc8335438b46962f704c3fffffff7db425eb8ebf9, linux
# b85743d422aa3fd112aceb3725dd1b1f2ff1ad829d20d71682680426582bcb5d (comparison runs
# only: unpack it and point BNGPATH at it).
BIONETGEN_SHA256 = {
    "mac": "712df23d1c64db0b85a4c3f7ae1027f480ddad06e1565828aa88c911c843f00c",
    "linux": "8af3bbf86a157b9474664c2bc7a647f0aa624219244bccc2ce5fd8f3c43d4ff1",
}
# Written into the extracted distribution; what jobs read to record which
# BioNetGen they ran, and to tell a fetched copy from one found elsewhere.
FETCH_MARKER = ".bench-fetch.json"

USER_AGENT = "bngsim-benchmarks"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, timeout: float = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".download-")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r, os.fdopen(fd, "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def bionetgen_platform_key() -> str:
    system = platform.system()
    if system == "Darwin":
        return "mac"
    if system == "Linux":
        return "linux"
    raise SystemExit(f"unsupported platform {system!r}: BioNetGen is fetched for macOS and Linux")


def fetch_bionetgen(cache_dir: str | Path) -> Path:
    """Install BioNetGen into ``<cache_dir>/bionetgen/BioNetGen-<version>``.

    Returns the distribution root (the directory BNGPATH should name). A copy
    already present with a matching marker is reused.
    """
    key = bionetgen_platform_key()
    want = BIONETGEN_SHA256[key]
    root = Path(cache_dir) / "bionetgen"
    dest = root / f"BioNetGen-{BIONETGEN_VERSION}"
    marker = dest / FETCH_MARKER
    if marker.exists():
        try:
            if json.loads(marker.read_text(encoding="utf-8")).get("sha256") == want:
                return dest
        except (OSError, ValueError):
            pass
    url = f"{_BIONETGEN_BASE}-{key}.tar.gz"
    tarball = root / f"BioNetGen-{BIONETGEN_VERSION}-{key}.tar.gz"
    _download(url, tarball, timeout=600)
    got = sha256_file(tarball)
    if got != want:
        tarball.unlink(missing_ok=True)
        raise SystemExit(f"sha256 mismatch for {url}\n  expected {want}\n  got      {got}")
    if dest.exists():
        shutil.rmtree(dest)
    with tarfile.open(tarball) as tf:
        # The archive roots at BioNetGen-<version>/. The "data" filter refuses
        # absolute paths and links that leave the destination.
        tf.extractall(root, filter="data")
    tarball.unlink(missing_ok=True)
    if not (dest / "BNG2.pl").exists():
        raise SystemExit(f"{dest} has no BNG2.pl after extracting {url}")
    # The "data" filter drops execute bits it does not trust; restore them on the
    # binaries BioNetGen runs.
    for exe in [dest / "BNG2.pl", *sorted((dest / "bin").glob("*"))]:
        if exe.is_file():
            exe.chmod(exe.stat().st_mode | 0o111)
    marker.write_text(
        json.dumps({"version": BIONETGEN_VERSION, "url": url, "sha256": want}, indent=2) + "\n",
        encoding="utf-8",
    )
    return dest


# --------------------------------------------------------------------------- #
# A newer glibc, for machines whose own is too old to load run_network
# --------------------------------------------------------------------------- #
# BioNetGen's prebuilt Linux run_network and NFsim (2.9.3 and 2.9.4 alike) need
# glibc >= 2.34 and libstdc++ GLIBCXX_3.4.29; a cluster on an older OS (e.g. RHEL
# 8 / glibc 2.28) cannot load them. Rather than rebuild them or pin an older
# BioNetGen, we run the unchanged binaries under a newer loader, extracted (no
# root) from public Rocky Linux 9 RPMs. The binaries' bytes -- and so BioNetGen's
# sha256 fingerprint -- are untouched; only how they are launched changes (see
# env.run_network_launch).
#
# Pinned to the Rocky *vault* (archived point releases never garbage-collected,
# unlike the rolling mirror). glibc provides the loader + libc/libm; libstdc++
# provides GLIBCXX_3.4.29 / CXXABI_1.3.9. libgcc_s only needs GCC_3.0, which old
# hosts already satisfy, so it is not fetched.
GLIBC_TAG = "rocky-9.5"
_ROCKY_VAULT = "https://dl.rockylinux.org/vault/rocky/9.5/BaseOS/x86_64/os/Packages"
GLIBC_RPMS = [
    {
        "url": f"{_ROCKY_VAULT}/g/glibc-2.34-125.el9_5.1.x86_64.rpm",
        "sha256": "9049801a0f9865ffbdb16835c2345c46c00b671582377ae7f7cf1a9a8798bfb5",
    },
    {
        "url": f"{_ROCKY_VAULT}/l/libstdc++-11.5.0-5.el9_5.x86_64.rpm",
        "sha256": "c2396f89268ca7d21f0571bfd38ad33657b4a845e446d8662d41c432f1c194de",
    },
]
GLIBC_MARKER = ".bench-glibc.json"
_LOADER_NAME = "ld-linux-x86-64.so.2"


def _extract_rpm(rpm: Path, dest: Path) -> None:
    """Unpack an RPM payload into ``dest`` with ``rpm2cpio | cpio`` (no root)."""
    for tool in ("rpm2cpio", "cpio"):
        if shutil.which(tool) is None:
            raise SystemExit(
                f"{tool} is needed to install the glibc loader but was not found; "
                "it ships with rpm/cpio on the Linux hosts that need this"
            )
    dest.mkdir(parents=True, exist_ok=True)
    r2c = subprocess.Popen(["rpm2cpio", str(rpm)], stdout=subprocess.PIPE)
    cpio = subprocess.Popen(
        ["cpio", "-idmu", "--quiet"], stdin=r2c.stdout, cwd=dest,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    r2c.stdout.close()  # let rpm2cpio get SIGPIPE if cpio exits
    cpio.communicate()
    if r2c.wait() != 0 or cpio.returncode != 0:
        raise SystemExit(f"failed to extract {rpm.name} (rpm2cpio|cpio)")


def _soname_version(path: Path, prefix: str) -> str | None:
    """The highest ``<prefix>_<ver>`` symbol version in a shared object."""
    try:
        blob = path.read_bytes()
    except OSError:
        return None
    import re

    vers = re.findall(rb"%b_([0-9][0-9.]*)" % prefix.encode(), blob)
    if not vers:
        return None
    return max((v.decode() for v in vers), key=lambda s: [int(n) for n in s.split(".")])


def fetch_glibc(cache_dir: str | Path) -> Path:
    """Install a newer glibc + libstdc++ into ``<cache_dir>/glibc/<GLIBC_TAG>``.

    Returns the tree root. A copy already present with a matching marker (same
    set of RPM sha256s) is reused. Linux only; there is nothing to do elsewhere.
    """
    if platform.system() != "Linux":
        raise SystemExit("the glibc loader swap is only for Linux hosts")
    root = Path(cache_dir) / "glibc" / GLIBC_TAG
    marker = root / GLIBC_MARKER
    want = [r["sha256"] for r in GLIBC_RPMS]
    if marker.exists():
        try:
            m = json.loads(marker.read_text(encoding="utf-8"))
            if [r.get("sha256") for r in m.get("rpms", [])] == want:
                return root
        except (OSError, ValueError):
            pass
    if root.exists():
        shutil.rmtree(root)
    tree = root / "tree"
    for spec in GLIBC_RPMS:
        rpm = root / Path(spec["url"]).name
        _download(spec["url"], rpm, timeout=600)
        got = sha256_file(rpm)
        if got != spec["sha256"]:
            rpm.unlink(missing_ok=True)
            shutil.rmtree(root, ignore_errors=True)
            raise SystemExit(
                f"sha256 mismatch for {spec['url']}\n  expected {spec['sha256']}\n  got      {got}"
            )
        _extract_rpm(rpm, tree)
        rpm.unlink(missing_ok=True)
    loaders = sorted(tree.rglob(_LOADER_NAME))
    if not loaders:
        shutil.rmtree(root, ignore_errors=True)
        raise SystemExit(f"no {_LOADER_NAME} found after extracting the glibc RPMs")
    ld_so = loaders[0]  # sorted, so the recorded loader is stable across reinstalls
    libcxx = next(iter(tree.rglob("libstdc++.so.6")), None)
    # The loader's own dir (libc/libm) plus libstdc++'s dir, as --library-path.
    libdirs: list[Path] = [ld_so.parent]
    if libcxx is not None and libcxx.parent not in libdirs:
        libdirs.append(libcxx.parent)
    libc = next((p for p in [ld_so.parent / "libc.so.6"] if p.exists()), None)
    marker.write_text(json.dumps({
        "tag": GLIBC_TAG,
        "rpms": [{"url": r["url"], "sha256": r["sha256"]} for r in GLIBC_RPMS],
        "ld_so": str(ld_so.relative_to(root)),
        "library_path": [str(d.relative_to(root)) for d in libdirs],
        "glibc_version": _soname_version(libc, "GLIBC") if libc else None,
        "libstdcxx_version": _soname_version(libcxx, "GLIBCXX") if libcxx else None,
    }, indent=2) + "\n", encoding="utf-8")
    return root


def model_url(source: dict) -> str:
    """Where a spec's ``source`` is downloaded from: the file at its pinned commit.

    The path is percent-encoded (``/`` kept), since some pinned files have spaces
    or other URL-reserved characters in their names (e.g. in sys-bio/temp-biomodels).
    """
    path = urllib.parse.quote(source["path"])
    return f"https://raw.githubusercontent.com/{source['repo']}/{source['commit']}/{path}"


def model_cache_path(source: dict, cache_dir: str | Path) -> Path:
    """Where the file a spec's ``source`` names is cached, whether or not it is yet."""
    return Path(cache_dir) / "models" / source["sha256"] / Path(source["path"]).name


def model_cache_state(source: dict, cache_dir: str | Path) -> str:
    """``cached``, ``missing`` or ``corrupt`` (present, wrong bytes). No network."""
    dest = model_cache_path(source, cache_dir)
    if not dest.is_file():
        return "missing"
    return "cached" if sha256_file(dest) == source["sha256"] else "corrupt"


def model_file(source: dict, cache_dir: str | Path) -> Path:
    """The model file a spec's ``source`` names, downloaded once and sha256-checked.

    ``source`` carries ``repo`` (GitHub owner/name), ``commit``, ``path`` and
    ``sha256``. Files are cached by sha256 and file name, so two specs naming the
    same bytes under the same name share one copy, and a changed upstream file can
    never be read in place of the pinned one. The name is kept because BioNetGen
    names its outputs after the model file. A cached copy whose bytes no longer
    match is downloaded again.
    """
    want = source["sha256"]
    dest = model_cache_path(source, cache_dir)
    if dest.exists() and sha256_file(dest) == want:
        return dest
    url = model_url(source)
    _download(url, dest)
    got = sha256_file(dest)
    if got != want:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"sha256 mismatch for {url}: expected {want}, got {got}")
    return dest


# --------------------------------------------------------------------------- #
# RuleMonkey, a standalone network-free reference engine, built from source
# --------------------------------------------------------------------------- #
# The network-free capability runs RuleMonkey's own command-line driver,
# ``rm_driver``, beside the copy bngsim embeds. It has no binary release, so the
# pinned source release is downloaded (sha256-checked, like BioNetGen) and built
# with this environment's conda-forge C++ compiler -- never the host's system
# compiler, whose libstdc++ ABI may not match (the std::filesystem break that
# segfaulted the bngsim core on ARM; docs/BUILD_NOTES.md). libstdc++ and libgcc
# are linked statically, so the built driver depends on nothing of the
# environment it was built in but the host's libc.
RULEMONKEY_VERSION = "3.13.0"
RULEMONKEY_TAG = f"v{RULEMONKEY_VERSION}"
# The commit the tag names (the tag object itself is 7c3e1abf...).
RULEMONKEY_COMMIT = "ad3bd755cce7839e09061b747facd7025cacb861"
RULEMONKEY_URL = (
    f"https://github.com/richardposner/RuleMonkey/archive/refs/tags/{RULEMONKEY_TAG}.tar.gz"
)
RULEMONKEY_SHA256 = "5f5a7c599177aaf92561d3fbed339fc93fc1b3224ef07025d337f6dd2f89d160"
RULEMONKEY_MARKER = ".bench-build.json"
# The build recipe's version: bump it when the flags or what is installed change,
# and a driver an older recipe built is rebuilt.
RULEMONKEY_RECIPE = 1
# Release build of the driver only. Upstream turns -Werror on for a top-level
# build; a newer compiler than its CI's may warn where it did not, and a warning
# is not a reason to refuse a build whose sources are pinned.
RULEMONKEY_CMAKE_ARGS = [
    "-DCMAKE_BUILD_TYPE=Release",
    "-DRULEMONKEY_BUILD_CLI=ON",
    "-DRULEMONKEY_BUILD_TESTS=OFF",
    "-DRULEMONKEY_BUILD_EXAMPLES=OFF",
    "-DRULEMONKEY_WARNINGS_AS_ERRORS=OFF",
    "-DRULEMONKEY_INSTALL=OFF",
]
_STATIC_RUNTIME = "-static-libstdc++ -static-libgcc"


def rulemonkey_dir(cache_dir: str | Path) -> Path:
    """Where ``fetch_rulemonkey`` installs: ``<cache>/rulemonkey/RuleMonkey-<v>``,
    holding ``bin/rm_driver`` and the build marker."""
    return Path(cache_dir) / "rulemonkey" / f"RuleMonkey-{RULEMONKEY_VERSION}"


def _conda_cxx() -> Path:
    """This environment's conda-forge C++ compiler.

    ``$CXX`` when it is set (pixi's activation sets it on linux-64), otherwise
    the compiler the env's ``cxx-compiler`` package installs beside its Python.
    A compiler outside the environment (``/usr/bin/g++``) is refused: its
    libstdc++ is not the one the env's other native engines were built against.
    """
    import sys

    prefix = Path(sys.prefix).resolve()
    value = os.environ.get("CXX")
    if value:
        cxx = Path(shutil.which(value) or value).resolve()
    else:
        found = sorted((prefix / "bin").glob("*-conda-linux-gnu-g++")) + sorted(
            (prefix / "bin").glob("*-apple-darwin*-clang++"))
        if not found:
            raise SystemExit(
                "no conda-forge C++ compiler in this environment (the cxx-compiler "
                "package); run inside `pixi run`, or set CXX to one")
        cxx = found[0].resolve()
    if prefix not in cxx.parents:
        raise SystemExit(
            f"refusing to build RuleMonkey with {cxx}: it is not this environment's "
            f"conda-forge compiler (under {prefix}). A system compiler's libstdc++ "
            "ABI need not match the env's (docs/BUILD_NOTES.md)")
    return cxx


def _tool(name: str) -> str:
    import sys

    found = shutil.which(name) or shutil.which(str(Path(sys.prefix) / "bin" / name))
    if not found:
        raise SystemExit(f"{name} is needed to build RuleMonkey but was not found "
                         "(it is in the bench environment: run inside `pixi run`)")
    return found


def _first_line(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = (out.stdout or out.stderr).strip().splitlines()
    return lines[0].strip() if lines else None


def rulemonkey_tarball(cache_dir: str | Path) -> Path:
    """The pinned RuleMonkey source release, downloaded once into a directory of
    its own and sha256-checked. A cached copy with other bytes is fetched again."""
    dest = Path(cache_dir) / "rulemonkey" / "downloads" / f"RuleMonkey-{RULEMONKEY_TAG}.tar.gz"
    if dest.exists() and sha256_file(dest) == RULEMONKEY_SHA256:
        return dest
    _download(RULEMONKEY_URL, dest, timeout=600)
    got = sha256_file(dest)
    if got != RULEMONKEY_SHA256:
        dest.unlink(missing_ok=True)
        raise SystemExit(f"sha256 mismatch for {RULEMONKEY_URL}\n"
                         f"  expected {RULEMONKEY_SHA256}\n  got      {got}")
    return dest


def rulemonkey_build_current(dest: Path) -> bool:
    """Whether ``dest`` holds a driver built by this recipe from the pinned source."""
    marker, exe = dest / RULEMONKEY_MARKER, dest / "bin" / "rm_driver"
    try:
        m = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (exe.is_file() and m.get("source_sha256") == RULEMONKEY_SHA256
            and m.get("recipe") == RULEMONKEY_RECIPE
            and m.get("binary_sha256") == sha256_file(exe))


def fetch_rulemonkey(cache_dir: str | Path, *, download_only: bool = False,
                     jobs: int | None = None) -> Path:
    """Download the pinned RuleMonkey release and build its ``rm_driver``.

    Returns the install directory (``rulemonkey_dir``), or with ``download_only``
    the checked tarball (the download needs the network, the build a compute
    node: they can run apart). A build already current is reused. The source is
    unpacked into a new, empty directory of its own and built out of tree with
    this environment's CMake, Ninja and conda-forge compiler; only the driver is
    installed, with a marker recording the source, the toolchain and the
    binary's sha256.
    """
    dest = rulemonkey_dir(cache_dir)
    tarball = rulemonkey_tarball(cache_dir)
    if download_only:
        return tarball
    if rulemonkey_build_current(dest):
        return dest
    cxx = _conda_cxx()
    cmake, ninja = _tool("cmake"), _tool("ninja")
    work = Path(tempfile.mkdtemp(dir=dest.parent, prefix=".build-"))
    try:
        src_root = work / "src"
        src_root.mkdir()
        with tarfile.open(tarball) as tf:
            tf.extractall(src_root, filter="data")
        src = src_root / f"RuleMonkey-{RULEMONKEY_VERSION}"
        if not (src / "CMakeLists.txt").is_file():
            raise SystemExit(f"{tarball} does not unpack to RuleMonkey-{RULEMONKEY_VERSION}/")
        build = work / "build"
        args = [*RULEMONKEY_CMAKE_ARGS, f"-DCMAKE_CXX_COMPILER={cxx}",
                f"-DCMAKE_MAKE_PROGRAM={ninja}"]
        if platform.system() == "Linux":
            args.append(f"-DCMAKE_EXE_LINKER_FLAGS={_STATIC_RUNTIME}")
        steps = [
            [cmake, "-S", str(src), "-B", str(build), "-G", "Ninja", *args],
            [cmake, "--build", str(build), "--target", "rm_driver",
             "--parallel", str(jobs or os.cpu_count() or 1)],
        ]
        log = work / "build.log"
        with open(log, "w", encoding="utf-8") as fh:
            for step in steps:
                fh.write("$ " + " ".join(step) + "\n")
                fh.flush()
                proc = subprocess.run(step, stdout=fh, stderr=subprocess.STDOUT, cwd=work,
                                      check=False)
                if proc.returncode != 0:
                    tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
                    raise SystemExit("RuleMonkey build failed:\n  " + "\n  ".join(tail))
        built = build / "rm_driver"
        if not built.is_file():
            raise SystemExit(f"the build produced no rm_driver under {build}")
        usage = subprocess.run([str(built)], capture_output=True, text=True, timeout=60,
                               check=False)
        if "Usage: rm_driver" not in usage.stdout + usage.stderr:
            raise SystemExit("the built rm_driver does not run: "
                             + (usage.stderr or usage.stdout).strip()[:300])
        stage = work / "install"
        (stage / "bin").mkdir(parents=True)
        shutil.copy2(built, stage / "bin" / "rm_driver")
        shutil.copy2(log, stage / "build.log")
        (stage / RULEMONKEY_MARKER).write_text(json.dumps({
            "version": RULEMONKEY_VERSION,
            "tag": RULEMONKEY_TAG,
            "commit": RULEMONKEY_COMMIT,
            "url": RULEMONKEY_URL,
            "source_sha256": RULEMONKEY_SHA256,
            "recipe": RULEMONKEY_RECIPE,
            "compiler": {"path": str(cxx), "version": _first_line([str(cxx), "--version"])},
            "cmake": _first_line([cmake, "--version"]),
            "ninja": _first_line([ninja, "--version"]),
            "cmake_args": args,
            "binary_sha256": sha256_file(stage / "bin" / "rm_driver"),
            "built_on": platform.node(),
            "machine": platform.machine(),
        }, indent=2) + "\n", encoding="utf-8")
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(stage, dest)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return dest
