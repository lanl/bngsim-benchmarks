"""End to end on the smallest pilot model: BioNetGen builds the network, BNGsim,
run_network and libRoadRunner integrate it, the cross-engine verdict and the
interleaved timing are recorded.

run_network and libRoadRunner are the independent implementations that serve as
oracles; the cell count and network size are exact known answers. Needs bngsim,
BioNetGen (`pixi run fetch-bionetgen`), libRoadRunner and network access to
download the model.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import REPO

from bench import cli
from bench.capabilities import ode

pytestmark = pytest.mark.integration

PROBLEM = "kinetic_proofreading_hlavacek2001"
JOB = f"ode/{PROBLEM}/{ode.engines_id()}/auto"


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


def test_bngsim_agrees_with_every_reference(run_dir):
    out, _ = run_dir
    r = _result(out, "correctness")
    assert r["outcome"] == "PASS", r.get("verdicts") or r.get("error")
    # One verdict per reference engine; BNGsim must agree with all of them.
    assert set(r["verdicts"]) == set(ode.REFERENCE_ENGINES)
    for name, v in r["verdicts"].items():
        assert v["passed"] is True, (name, v)
        # 9 species x (300 steps + the initial point) = 2709 compared cells.
        assert v["n_cells"] == 9 * 301
    assert r["measurements"]["network"] == {"species": 9, "reactions": 14}
    assert "size_mismatch" not in r["measurements"]


def test_timing_records_interleaved_repetitions_for_every_engine(run_dir):
    out, _ = run_dir
    r = _result(out, "timing")
    assert r["outcome"] == "PASS"
    engines = ["bngsim", *ode.REFERENCE_ENGINES]
    reps = r["measurements"]["repetitions"]
    assert 5 <= reps["rounds"] <= 20
    # Every engine is measured the same number of rounds, all positive.
    for name in engines:
        secs = reps["engines"][name]["seconds"]
        assert len(secs) == reps["rounds"]
        assert all(x > 0 for x in secs)
        # Raw per-repetition trajectories are consumed into the reproducibility
        # block, never serialized (issue #8).
        assert "outputs" not in reps["engines"][name]
    # Each engine reports an absolute median bracketed by its bootstrap CI.
    times = r["measurements"]["times"]
    for name in engines:
        s = times[name]
        assert s["ci_low"] <= s["median_seconds"] <= s["ci_high"]
    # Every engine reproduced its own first run.
    repro = r["measurements"]["reproducibility"]
    for name in engines:
        assert repro[name]["passed"] is True, (name, repro[name])
    assert repro["bngsim"]["n_reps"] == reps["rounds"]


def test_results_record_the_build_they_measured(run_dir):
    out, _ = run_dir
    env = _result(out, "timing")["environment"]
    assert env["bngsim"]["build_commit"]
    assert env["bionetgen"]["fetched"] is True
    assert env["node_type"]


def test_failed_codegen_build_fails_the_job_instead_of_falling_back(tmp_path):
    # A compiler that always fails, and a threshold that asks for compiled code on
    # this 9-species model. bngsim then falls back to the interpreter unless
    # BNGSIM_CODEGEN_STRICT is in effect, and only then names it in the error. The
    # caller's "0" shows the worker overrides it.
    if not _bionetgen_present():
        pytest.skip("BioNetGen not installed; run `pixi run fetch-bionetgen`")
    pytest.importorskip("bngsim")
    broken_cc = shutil.which("false")
    assert broken_cc, "no `false` command to stand in for a broken compiler"
    env = {**os.environ, "CC": broken_cc, "BNGSIM_CODEGEN_THRESHOLD": "1",
           "BNGSIM_CODEGEN_STRICT": "0"}
    job = f"{JOB}/correctness"
    proc = subprocess.run(
        [sys.executable, "-m", "bench.worker", "--job", job,
         "--problems", str(REPO / "problems"), "--results", str(tmp_path),
         "--cache", str(REPO / ".cache")],
        env=env, capture_output=True, text=True, timeout=600, check=False)
    assert proc.returncode == 0, proc.stderr[-2000:]
    r = json.loads((tmp_path / f"{job}.json").read_text())
    assert r["outcome"] == "EXCEPTION", r.get("verdict")
    assert "BNGSIM_CODEGEN_STRICT is set" in r["bngsim_error"]


def test_slurm_array_task_runs_its_jobs_for_real(tmp_path, capsys):
    # `bench submit --dry-run` on the smallest pilot model, then its batch script
    # run by bash as Slurm would run array task 0, with the real bench.
    if not _bionetgen_present():
        pytest.skip("BioNetGen not installed; run `pixi run fetch-bionetgen`")
    pytest.importorskip("bngsim")
    config = tmp_path / "slurm.yaml"
    config.write_text(
        'time_limit: "30"\ncpus_per_task: 2\nmargin_seconds: 30\nstartup_seconds: 30\n'
        f'bench: "{sys.executable} -m bench.cli"\n')
    out, results = tmp_path / "slurm", tmp_path / "results"
    common = ["--problems", str(REPO / "problems"), "--results", str(results),
              "--cache", str(REPO / ".cache")]
    assert cli.main(["submit", PROBLEM, *common, "--config", str(config),
                     "--out", str(out), "--dry-run"]) == 0
    assert "2 jobs in 1 array task" in capsys.readouterr().out
    (sub_dir,) = out.iterdir()
    env = {**os.environ, "SLURM_ARRAY_TASK_ID": "0"}
    proc = subprocess.run(["bash", str(sub_dir / "submit.sbatch")], env=env, cwd=tmp_path,
                          capture_output=True, text=True, timeout=600, check=False)
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-3000:]
    for kind in ("correctness", "timing"):
        r = json.loads((results / JOB / f"{kind}.json").read_text())
        assert r["outcome"] == "PASS", r.get("verdict") or r.get("error")
    # Everything it was given now has a current result: nothing left to submit.
    assert cli.main(["submit", PROBLEM, *common, "--config", str(config),
                     "--out", str(out), "--dry-run"]) == 0
    assert "nothing to run" in capsys.readouterr().out


def test_second_run_has_nothing_to_do(run_dir, capsys):
    out, args = run_dir
    before = {p: p.stat().st_mtime_ns for p in (out / "results").rglob("*.json")}
    assert cli.main(["run", PROBLEM, *args]) == 0
    after = {p: p.stat().st_mtime_ns for p in (out / "results").rglob("*.json")}
    assert before == after
    assert "fresh 2" in capsys.readouterr().out
