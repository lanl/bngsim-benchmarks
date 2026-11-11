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
            if json.loads(marker.read_text()).get("sha256") == want:
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
        json.dumps({"version": BIONETGEN_VERSION, "url": url, "sha256": want}, indent=2) + "\n"
    )
    return dest


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
