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
