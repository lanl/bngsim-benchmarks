"""End to end on the smallest pilot model: BioNetGen builds the network, BNGsim
and run_network integrate it, the verdict and the paired timing are recorded.

run_network is the independent implementation that serves as the oracle; the
cell count and network size are exact known answers. Needs bngsim, BioNetGen
(`pixi run fetch-bionetgen`) and network access to download the model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from conftest import REPO

from bench import cli

pytestmark = pytest.mark.integration

PROBLEM = "kinetic_proofreading_hlavacek2001"
JOB = f"ode/{PROBLEM}/bngsim-vs-run_network/auto"


def _bionetgen_present() -> bool:
    root = os.environ.get("BNGPATH")
    return bool(root) and (Path(root) / "BNG2.pl").exists()


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    if not _bionetgen_present():
        pytest.skip("BioNetGen not installed; run `pixi run fetch-bionetgen`")
    pytest.importorskip("bngsim")
    out = tmp_path_factory.mktemp("integration")
    args = ["--problems", str(REPO / "problems"), "--results", str(out / "results"),
            "--cache", str(REPO / ".cache")]
    assert cli.main(["run", PROBLEM, *args]) == 0
    return out, args


def _result(out: Path, kind: str) -> dict:
    return json.loads((out / "results" / JOB / f"{kind}.json").read_text())


def test_bngsim_agrees_with_run_network(run_dir):
    out, _ = run_dir
    r = _result(out, "correctness")
    assert r["outcome"] == "PASS", r.get("verdict") or r.get("error")
    # 9 species x (300 steps + the initial point) = 2709 compared cells.
    assert r["verdict"]["n_cells"] == 9 * 301
    assert r["measurements"]["network"] == {"species": 9, "reactions": 14}
    assert "size_mismatch" not in r["measurements"]


def test_timing_records_paired_repetitions(run_dir):
    out, _ = run_dir
    r = _result(out, "timing")
    assert r["outcome"] == "PASS"
    reps = r["measurements"]["repetitions"]
    assert 5 <= reps["pairs"] <= 20 and len(reps["a_seconds"]) == reps["pairs"]
    s = r["measurements"]["summary"]
    assert s["ci_low"] <= s["speedup"] <= s["ci_high"]
    assert all(x > 0 for x in reps["a_seconds"] + reps["b_seconds"])


def test_results_record_the_build_they_measured(run_dir):
    out, _ = run_dir
    env = _result(out, "timing")["environment"]
    assert env["bngsim"]["build_commit"]
    assert env["bionetgen"]["fetched"] is True
    assert env["node_type"]


def test_second_run_has_nothing_to_do(run_dir, capsys):
    out, args = run_dir
    before = {p: p.stat().st_mtime_ns for p in (out / "results").rglob("*.json")}
    assert cli.main(["run", PROBLEM, *args]) == 0
    after = {p: p.stat().st_mtime_ns for p in (out / "results").rglob("*.json")}
    assert before == after
    assert "fresh 2" in capsys.readouterr().out
