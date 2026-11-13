#!/usr/bin/env python3
"""Copy bngsim's comparison core (parity_checks/_core) into src/bench/_vendor.

The suite judges agreement with the same protocol bngsim's own parity checks use
(tolerances, ensemble verdicts, outcome vocabulary), so a verdict means the same
thing in both places. bngsim does not ship that code in its wheel, so it is
copied here verbatim from one pinned commit of lanl/bngsim, with the sha256 of
every file recorded in VENDOR.json. tests/test_vendor.py fails if a copied file
is edited, or (with network access) if it no longer matches the pinned commit.

    python tools/vendor_parity_core.py --ref v0.17.2

Re-run it when the pinned bngsim release changes, and commit the result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

REPO = "lanl/bngsim"
UPSTREAM_DIR = "parity_checks/_core"
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "src" / "bench" / "_vendor" / "parity_core"
USER_AGENT = "bngsim-benchmarks"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def resolve_commit(ref: str) -> str:
    data = json.loads(_get(f"https://api.github.com/repos/{REPO}/commits/{ref}"))
    return data["sha"]


def list_files(commit: str) -> list[str]:
    url = f"https://api.github.com/repos/{REPO}/contents/{UPSTREAM_DIR}?ref={commit}"
    entries = json.loads(_get(url))
    return sorted(e["name"] for e in entries if e["type"] == "file" and e["name"].endswith(".py"))


def raw_url(commit: str, name: str) -> str:
    return f"https://raw.githubusercontent.com/{REPO}/{commit}/{UPSTREAM_DIR}/{name}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref", required=True, help="tag, branch or commit of lanl/bngsim")
    args = ap.parse_args()

    commit = resolve_commit(args.ref)
    names = list_files(commit)
    if "__init__.py" not in names:
        print(f"{UPSTREAM_DIR} at {commit} has no __init__.py", file=sys.stderr)
        return 1
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    files = {}
    for name in names:
        body = _get(raw_url(commit, name))
        (DEST / name).write_bytes(body)
        files[name] = hashlib.sha256(body).hexdigest()
    record = {
        "upstream": REPO,
        "path": UPSTREAM_DIR,
        "ref": args.ref,
        "commit": commit,
        "license": "MIT (same terms and copyright as this repository; see LICENSE)",
        "note": "Copied verbatim. Do not edit: re-run tools/vendor_parity_core.py instead.",
        "files": files,
    }
    (DEST.parent / "VENDOR.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"vendored {len(files)} files from {REPO}@{commit[:12]} ({args.ref}) into {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
