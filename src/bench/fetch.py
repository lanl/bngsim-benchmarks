"""Downloads, each pinned by sha256: the BioNetGen distribution and model files.

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
import urllib.request
from pathlib import Path

BIONETGEN_VERSION = "2.9.3"
_BIONETGEN_BASE = (
    "https://github.com/RuleWorld/bionetgen/releases/download/"
    f"BioNetGen-{BIONETGEN_VERSION}/BioNetGen-{BIONETGEN_VERSION}"
)
# sha256 of the published release assets.
BIONETGEN_SHA256 = {
    "mac": "5ddf0ebf76b9573bbfef794bc8335438b46962f704c3fffffff7db425eb8ebf9",
    "linux": "b85743d422aa3fd112aceb3725dd1b1f2ff1ad829d20d71682680426582bcb5d",
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
# BioNetGen's prebuilt Linux run_network (2.9.3) needs glibc >= 2.34 and
# libstdc++ GLIBCXX_3.4.29; a cluster on an older OS (e.g. RHEL 8 / glibc 2.28)
# cannot load it. Rather than rebuild it or pin an older BioNetGen, we run the
# unchanged binary under a newer loader, extracted (no root) from public Rocky
# Linux 9 RPMs. The binary's bytes -- and so its sha256 fingerprint -- are
# untouched; only how it is launched changes (see env.run_network_launch).
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


def model_file(source: dict, cache_dir: str | Path) -> Path:
    """The model file a spec's ``source`` names, downloaded once and sha256-checked.

    ``source`` carries ``repo`` (GitHub owner/name), ``commit``, ``path`` and
    ``sha256``. Files are cached by sha256, so two specs naming the same bytes
    share one copy and a changed upstream file can never be read in place of the
    pinned one.
    """
    want = source["sha256"]
    dest = Path(cache_dir) / "models" / want / Path(source["path"]).name
    if dest.exists() and sha256_file(dest) == want:
        return dest
    url = f"https://raw.githubusercontent.com/{source['repo']}/{source['commit']}/{source['path']}"
    _download(url, dest)
    got = sha256_file(dest)
    if got != want:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"sha256 mismatch for {url}: expected {want}, got {got}")
    return dest
