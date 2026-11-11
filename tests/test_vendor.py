"""The vendored comparison core is bngsim's, byte for byte, at the recorded commit."""

from __future__ import annotations

import re
import urllib.request

import pytest

from bench._vendor import VENDOR_JSON, parity_core_record
from bench.fetch import sha256_file

CORE = VENDOR_JSON.parent / "parity_core"


def test_record_names_a_full_commit():
    rec = parity_core_record()
    assert re.fullmatch(r"[0-9a-f]{40}", rec["commit"])
    assert rec["upstream"] == "lanl/bngsim" and rec["path"] == "parity_checks/_core"


def test_every_vendored_file_matches_its_recorded_sha256():
    rec = parity_core_record()
    on_disk = sorted(p.name for p in CORE.glob("*.py"))
    assert on_disk == sorted(rec["files"]), "a file was added to or removed from the copy"
    for name, want in rec["files"].items():
        assert sha256_file(CORE / name) == want, f"parity_core/{name} was edited"


def test_vendored_core_imports_and_exposes_the_verdict():
    from bench._vendor.parity_core import Outcome, differ

    assert Outcome.PASS.value == "PASS"
    assert callable(differ.deterministic_verdict)


@pytest.mark.network
def test_vendored_files_match_upstream_at_the_recorded_commit():
    rec = parity_core_record()
    for name in rec["files"]:
        url = (f"https://raw.githubusercontent.com/{rec['upstream']}/{rec['commit']}/"
               f"{rec['path']}/{name}")
        req = urllib.request.Request(url, headers={"User-Agent": "bngsim-benchmarks"})
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read()
        assert (CORE / name).read_bytes() == body, f"{name} differs from {url}"
