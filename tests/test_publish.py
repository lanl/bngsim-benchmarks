"""bench publish: an all-or-nothing, scrubbed, machine-readable snapshot of a suite."""

from __future__ import annotations

import csv
import datetime as dt
import getpass
import json
from pathlib import Path

import pytest

from bench import jobs as jobs_mod
from bench import specs
from bench.jobs import CORRECTNESS, TIMING
from bench.publish import PublishError, leaks, publish, scrub
from bench.results import freshness_key, key_parts, write_result

FPS = {CORRECTNESS: {"bngsim": {"version": "0.17.2", "build_commit": "932ea5e13396"}},
       TIMING: {"bngsim": {"version": "0.17.2", "build_commit": "932ea5e13396"},
                "node_type": "LOGIN NODE"}}
NODE = "Some CPU @ 2.10GHz | x86_64 | Linux"
NOW = dt.datetime(2026, 11, 23, 12, 0, tzinfo=dt.UTC)


def _member(spec_data, i):
    d = dict(spec_data, id=f"model_{i:02d}", layer="core",
             verification={"kind": "equations", "evidence": f"models/m{i}/verify.ipynb"},
             suites={"ode-net": i})
    d["source"] = dict(spec_data["source"], path=f"models/m{i}/m{i}.bngl", sha256=f"{i:064x}")
    return d


@pytest.fixture
def suite(spec_data, write_spec, tmp_path):
    for i in range(1, specs.SUITE_SIZE + 1):
        write_spec(_member(spec_data, i))
    problems = specs.load_all(tmp_path / "problems")
    return problems, jobs_mod.jobs_for(problems)


def _env(host="cn123", job="4242"):
    return {
        "bngsim": {"version": "0.17.2", "build_commit": "932ea5e13396"},
        "bionetgen": {"version": "2.9.3", "bngpath": str(Path.home() / "x/BioNetGen-2.9.3")},
        "roadrunner": {"version": "2.11.0"}, "copasi": {"version": "4.48.311"},
        "amici": {"version": "1.1.0"},
        "harness": {"bench": "0.1.0"}, "harness_source": {"commit": "c" * 40, "dirty": False},
        "host": {"hostname": host, "slurm_job_id": job, "cpu_model": "Some CPU"},
        "node_type": NODE,
    }


def _write_all(jobs, results, *, skip=None, host="cn123"):
    for job in jobs:
        if job.id == skip:
            continue
        fp = {**FPS[job.kind], "node_type": NODE} if job.kind == TIMING else FPS[job.kind]
        rec = {"job_id": job.id, "problem": job.problem, "kind": job.kind,
               "outcome": "PASS", "key": freshness_key(job, fp),
               "key_parts": key_parts(job, fp), "environment": _env(host)}
        if job.kind == CORRECTNESS:
            rec["verdicts"] = {"roadrunner": {"passed": True, "max_err_over_peak": 1e-8,
                                              "max_rel": 0.0, "n_cells": 10}}
            rec["copasi_error"] = "TIMEOUT: no result within the 3600 s per-engine budget"
            rec["measurements"] = {"network": {"species": 9, "reactions": 14},
                                   "references": {"roadrunner": {"cold_seconds": 1.5}}}
        else:
            rec["measurements"] = {
                "network": {"species": 9, "reactions": 14},
                "times": {"bngsim": {"median_seconds": 0.5, "ci_low": 0.4, "ci_high": 0.6},
                          "roadrunner": {"median_seconds": 0.8, "ci_low": 0.7,
                                         "ci_high": 0.9}},
                "reproducibility": {"bngsim": {"passed": True}, "roadrunner": {"passed": True}},
                "repetitions": {"rounds": 5},
                "bngsim": {"codegen_seconds": 2.0, "load_seconds": 0.1},
                "netgen_seconds": 1.0,
                "references": {"roadrunner": {"load_seconds": 3.0,
                                              "command": f"/tmp/{getpass.getuser()}/bench-job-ab12/model.net"}},
                "engine_roles": {"bngsim": "warm, in-process (system under test)",
                                 "roadrunner": "warm, in-process"},
            }
        write_result(results, job, rec)


def test_a_complete_suite_publishes_manifest_csvs_and_scrubbed_jobs(suite, tmp_path):
    problems, jobs = suite
    results = tmp_path / "results"
    _write_all(jobs, results)
    dest = publish("ode-net", problems, jobs, results, tmp_path / "published", FPS,
                   root=tmp_path, now=NOW)
    assert dest == tmp_path / "published" / "ode-net" / "bngsim-0.17.2-932ea5e13396"
    manifest = json.loads((dest / "manifest.json").read_text())
    assert manifest["suite"] == "ode-net" and manifest["published"] == "2026-11-23T12:00:00Z"
    assert [m["position"] for m in manifest["members"]] == list(range(1, 11))
    assert manifest["node_type"] == NODE
    assert manifest["harness"]["source_commits"] == ["c" * 40]
    assert manifest["engines"]["roadrunner"] == ["2.11.0"]
    assert len(manifest["jobs"]) == 20

    with open(dest / "correctness.csv", encoding="utf-8") as fh:
        corr = list(csv.DictReader(fh))
    assert len(corr) == 20  # 10 problems x (roadrunner ok + copasi timeout)
    first = [r for r in corr if r["problem"] == "model_01"]
    assert {r["engine"]: r["status"] for r in first} == {"roadrunner": "ok", "copasi": "timeout"}
    assert first[0]["agreement"] == "PARTIAL 1/2 (copasi: timeout)"

    with open(dest / "timing.csv", encoding="utf-8") as fh:
        timing = list(csv.DictReader(fh))
    row = next(r for r in timing if r["problem"] == "model_01" and r["engine"] == "bngsim")
    assert float(row["sim_median_seconds"]) == 0.5
    assert float(row["presim_seconds"]) == pytest.approx(3.1)  # netgen + codegen + load

    # Each job's record is there in full, minus where it ran.
    job = next(j for j in jobs if j.kind == TIMING)
    rec = json.loads((dest / "jobs" / f"{job.id}.json").read_text())
    host = rec["environment"]["host"]
    assert "hostname" not in host and "slurm_job_id" not in host
    assert "bngpath" not in rec["environment"]["bionetgen"]
    cmd = rec["measurements"]["references"]["roadrunner"]["command"]
    assert getpass.getuser() not in cmd and cmd.startswith("$TMPDIR/bench-job")


def test_an_incomplete_suite_is_refused_naming_the_gap(suite, tmp_path):
    problems, jobs = suite
    results = tmp_path / "results"
    gap = jobs[3].id
    _write_all(jobs, results, skip=gap)
    with pytest.raises(PublishError, match="no result") as err:
        publish("ode-net", problems, jobs, results, tmp_path / "published", FPS, root=tmp_path)
    assert gap in str(err.value)
    assert not (tmp_path / "published").exists()


def test_a_stale_result_is_refused(suite, tmp_path):
    problems, jobs = suite
    results = tmp_path / "results"
    _write_all(jobs, results)
    newer = {k: {**v, "bngsim": {"version": "0.18.0", "build_commit": "f" * 12}}
             for k, v in FPS.items()}
    with pytest.raises(PublishError, match="stale"):
        publish("ode-net", problems, jobs, results, tmp_path / "published", newer, root=tmp_path)


def test_a_host_name_that_survives_the_scrub_refuses_publishing(suite, tmp_path):
    problems, jobs = suite
    results = tmp_path / "results"
    _write_all(jobs, results)
    # An error message that names the node it ran on is not something we can rewrite.
    job = next(j for j in jobs if j.kind == CORRECTNESS)
    path = results / f"{job.id}.json"
    rec = json.loads(path.read_text())
    rec["copasi_error"] = "RuntimeError: license server unreachable from cn123"
    path.write_text(json.dumps(rec))
    with pytest.raises(PublishError, match="cn123"):
        publish("ode-net", problems, jobs, results, tmp_path / "published", FPS, root=tmp_path)


def test_scrub_rewrites_local_paths_and_leaks_finds_whole_words_only(tmp_path):
    rec = {"environment": _env(), "note": f"read {tmp_path}/problems/x.yaml",
           "home": str(Path.home() / "wt/scratch")}
    out = scrub(rec, tmp_path)
    assert out["note"] == "read ./problems/x.yaml" and out["home"] == "~/wt/scratch"
    assert leaks({"a": "ran on cn123."}, ["cn123"]) == [".a: 'cn123'"]
    assert leaks({"a": "cn1234 and xcn123"}, ["cn123"]) == []


def _edit(results, job, **fields):
    path = results / f"{job.id}.json"
    rec = json.loads(path.read_text())
    rec.update(fields)
    path.write_text(json.dumps(rec))


def test_a_reference_that_does_not_apply_gets_a_row_saying_why(suite, tmp_path):
    problems, jobs = suite
    results = tmp_path / "results"
    _write_all(jobs, results)
    why = "no faithful BNGL form: bngsim.convert.sbml_to_bngl refuses it (an assignment rule)"
    for job in jobs:
        if job.problem == "model_01":
            _edit(results, job, not_applicable={"run_network": why})
    dest = publish("ode-net", problems, jobs, results, tmp_path / "published", FPS,
                   root=tmp_path, now=NOW)
    for name in ("correctness.csv", "timing.csv"):
        with open(dest / name, encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r["problem"] == "model_01"]
        [rn] = [r for r in rows if r["engine"] == "run_network"]
        assert rn["status"] == "not_applicable" and rn["error"] == why
        assert all(r["engine"] != "run_network" for r in rows if r is not rn)
    with open(dest / "correctness.csv", encoding="utf-8") as fh:
        corr = [r for r in csv.DictReader(fh) if r["problem"] == "model_01"]
    assert {r["engine"]: r["status"] for r in corr} == {
        "roadrunner": "ok", "copasi": "timeout", "run_network": "not_applicable"}
    assert corr[0]["agreement"] == "PARTIAL 1/2 (copasi: timeout, run_network: n/a)"
    assert corr[2]["passed"] == ""


def _store_trajectories(jobs, results):
    """Give every correctness job stored trajectories, as the worker does."""
    import numpy as np

    from bench.results import write_trajectories

    for job in jobs:
        if job.kind == CORRECTNESS:
            t = np.linspace(0.0, 1.0, 11)
            meta = write_trajectories(results, job, {
                "time": t, "species": np.array(["A", "B"]),
                "bngsim": np.column_stack([np.exp(-t), 1 - np.exp(-t)]),
                "roadrunner": np.column_stack([np.exp(-t), 1 - np.exp(-t)]) * (1 + 1e-9)})
            _edit(results, job, trajectories=meta)


def test_small_trajectories_are_in_the_snapshot_beside_their_records(suite, tmp_path):
    from bench.results import _sha256

    problems, jobs = suite
    results = tmp_path / "results"
    _write_all(jobs, results)
    _store_trajectories(jobs, results)
    dest = publish("ode-net", problems, jobs, results, tmp_path / "published", FPS,
                   root=tmp_path, now=NOW)
    manifest = json.loads((dest / "manifest.json").read_text())
    assert manifest["trajectories"]["in_snapshot"] is True
    assert manifest["trajectories"]["files"] == 10
    corr = [j for j in jobs if j.kind == CORRECTNESS]
    for job in corr:
        rec = json.loads((dest / "jobs" / f"{job.id}.json").read_text())
        meta = rec["trajectories"]
        assert meta["in_snapshot"] is True and "note" not in meta
        npz = (dest / "jobs" / f"{job.id}.json").with_name(meta["file"])
        assert npz.exists() and _sha256(npz) == meta["sha256"]
    entry = next(e for e in manifest["jobs"] if e["job_id"] == corr[0].id)
    assert entry["trajectories"]["file"] == f"jobs/{corr[0].id}.npz"
    assert manifest["trajectories"]["total_bytes"] == sum(
        json.loads((dest / "jobs" / f"{j.id}.json").read_text())["trajectories"]["bytes"]
        for j in corr)


def test_trajectories_over_the_limit_are_named_and_marked_not_in_the_snapshot(
        suite, tmp_path, monkeypatch):
    from bench import publish as pub

    problems, jobs = suite
    results = tmp_path / "results"
    _write_all(jobs, results)
    _store_trajectories(jobs, results)
    monkeypatch.setattr(pub, "TRAJECTORIES_LIMIT_BYTES", 1000)
    dest = publish("ode-net", problems, jobs, results, tmp_path / "published", FPS,
                   root=tmp_path, now=NOW)
    assert not list(dest.rglob("*.npz"))
    manifest = json.loads((dest / "manifest.json").read_text())
    assert manifest["trajectories"]["in_snapshot"] is False
    assert manifest["trajectories"]["limit_bytes"] == 1000
    job = next(j for j in jobs if j.kind == CORRECTNESS)
    meta = json.loads((dest / "jobs" / f"{job.id}.json").read_text())["trajectories"]
    assert meta["in_snapshot"] is False and meta["note"].startswith("not in this snapshot")
    assert len(meta["sha256"]) == 64 and meta["bytes"] > 0
    entry = next(e for e in manifest["jobs"] if e["job_id"] == job.id)
    assert entry["trajectories"]["in_snapshot"] is False
    assert entry["trajectories"]["sha256"] == meta["sha256"]


def test_trajectories_that_do_not_match_their_record_refuse_publishing(suite, tmp_path):
    from bench.results import trajectories_path

    problems, jobs = suite
    results = tmp_path / "results"
    _write_all(jobs, results)
    _store_trajectories(jobs, results)
    job = next(j for j in jobs if j.kind == CORRECTNESS)
    trajectories_path(results, job).write_bytes(b"not the stored file")
    with pytest.raises(PublishError, match="not the file its record names") as err:
        publish("ode-net", problems, jobs, results, tmp_path / "published", FPS, root=tmp_path)
    assert job.id in str(err.value)
    assert not (tmp_path / "published").exists()


@pytest.mark.parametrize("vendored, species, half", [
    (True, False, "species"), (False, True, "vendored"), (False, False, "both"),
    (True, True, ""), (None, None, ""),
])
def test_a_diff_names_the_half_of_the_ode_gate_that_failed(suite, tmp_path,
                                                           vendored, species, half):
    problems, jobs = suite
    results = tmp_path / "results"
    _write_all(jobs, results)
    verdict = {"passed": bool(vendored and species), "max_err_over_peak": 0.3,
               "max_rel": 0.0, "n_cells": 10}
    if vendored is not None:
        verdict.update(vendored_passed=vendored, species_check={"passed": species})
    job = next(j for j in jobs if j.kind == CORRECTNESS and j.problem == "model_01")
    _edit(results, job, outcome="PASS" if verdict["passed"] else "DIFF",
          verdicts={"roadrunner": verdict})
    dest = publish("ode-net", problems, jobs, results, tmp_path / "published", FPS,
                   root=tmp_path, now=NOW)
    with open(dest / "correctness.csv", encoding="utf-8") as fh:
        rows = {(r["problem"], r["engine"]): r for r in csv.DictReader(fh)}
    row = rows[("model_01", "roadrunner")]
    assert row["status"] == ("ok" if verdict["passed"] else "DIFF")
    assert row["failed_check"] == half
    assert rows[("model_01", "copasi")]["failed_check"] == ""  # timed out: no verdict


# --------------------------------------------------------------------------- #
# A timing its member's correctness does not let run is withheld, not demanded
# --------------------------------------------------------------------------- #
def _member_jobs(jobs, problem="model_01"):
    corr = next(j for j in jobs if j.kind == CORRECTNESS and j.problem == problem)
    tim = next(j for j in jobs if j.kind == TIMING and j.problem == problem)
    return corr, tim


@pytest.mark.parametrize("outcome", ["DIFF", "EXCEPTION", "BAD_TEST", "TIMEOUT"])
def test_a_timing_beside_a_blocking_correctness_is_published_as_withheld(suite, tmp_path,
                                                                         outcome):
    problems, jobs = suite
    results = tmp_path / "results"
    _write_all(jobs, results)
    corr, tim = _member_jobs(jobs)
    _edit(results, corr, outcome=outcome)
    # A timing left from before the correctness became DIFF -- even one measured on
    # another node -- is neither shipped nor allowed to block the snapshot.
    _edit(results, tim, environment={**_env(), "node_type": "Another CPU | x86_64 | Linux"})
    dest = publish("ode-net", problems, jobs, results, tmp_path / "published", FPS,
                   root=tmp_path, now=NOW)
    with open(dest / "timing.csv", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["problem"] == "model_01"]
    (row,) = rows
    assert (row["outcome"], row["status"], row["error"]) == (
        "WITHHELD", "withheld", f"correctness {outcome}")
    assert row["engine"] == "" and row["sim_median_seconds"] == "" and row["species"] == "9"
    assert not (dest / "jobs" / f"{tim.id}.json").exists()
    assert (dest / "jobs" / f"{corr.id}.json").exists()
    manifest = json.loads((dest / "manifest.json").read_text())
    entry = next(e for e in manifest["jobs"] if e["job_id"] == tim.id)
    assert entry == {"job_id": tim.id, "problem": "model_01", "kind": TIMING,
                     "outcome": "WITHHELD", "withheld": f"correctness {outcome}"}
    assert manifest["node_type"] == NODE
    with open(dest / "correctness.csv", encoding="utf-8") as fh:
        assert {r["outcome"] for r in csv.DictReader(fh) if r["problem"] == "model_01"} \
            == {outcome}


def test_a_diff_member_needs_no_timing_result_but_a_passing_one_does(suite, tmp_path):
    problems, jobs = suite
    results = tmp_path / "results"
    corr, tim = _member_jobs(jobs)
    _write_all(jobs, results, skip=tim.id)
    with pytest.raises(PublishError, match="no result"):
        publish("ode-net", problems, jobs, results, tmp_path / "published", FPS, root=tmp_path)
    _edit(results, corr, outcome="DIFF")
    dest = publish("ode-net", problems, jobs, results, tmp_path / "published", FPS,
                   root=tmp_path, now=NOW)
    with open(dest / "timing.csv", encoding="utf-8") as fh:
        (row,) = [r for r in csv.DictReader(fh) if r["problem"] == "model_01"]
    assert row["status"] == "withheld" and row["error"] == "correctness DIFF"
