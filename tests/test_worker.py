"""The worker runs one job in the environment the harness promises: the forced
settings are in place before the capability runs, whatever the caller set, and
the result records them."""

from __future__ import annotations

import json
import os

import pytest

from bench import env, jobs, specs, worker
from bench.capabilities import CAPABILITIES
from bench.results import result_path

FP = {
    "bngsim": {"version": "0.17.1", "build_commit": "e6c2fc65ebe3"},
    "bionetgen": {"version": "2.9.3", "sha256": "c" * 64},
    "harness": env.harness_identity(),
}


@pytest.fixture
def one_job(tmp_path, spec_data, write_spec, monkeypatch):
    """A problem on disk, its correctness job, and a fake capability that
    reports the environment it ran in."""
    problems = tmp_path / "problems"
    problem = specs.load(write_spec(spec_data, directory=problems))
    job = next(j for j in jobs.jobs_for([problem]) if j.kind == jobs.CORRECTNESS)
    seen: dict[str, str | None] = {}

    def run(job, problem, ctx):
        seen.update({k: os.environ.get(k) for k in env.HARNESS_ENV})
        return {"outcome": "PASS"}

    monkeypatch.setattr(CAPABILITIES[job.capability], "run", run)
    monkeypatch.setattr(env, "fingerprint", lambda *, timing: FP)
    monkeypatch.setattr(env, "environment", lambda: {"harness": env.harness_identity()})
    # Every variable the worker writes is registered here, so it is restored.
    for k, v in env.SINGLE_THREAD_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("BNGSIM_CODEGEN_CACHE_DIR", str(tmp_path / "unused"))
    monkeypatch.delenv("BENCH_CPU", raising=False)
    args = ["--job", job.id, "--problems", str(problems),
            "--results", str(tmp_path / "results"), "--cache", str(tmp_path / "cache")]
    return job, args, seen, tmp_path / "results"


@pytest.mark.parametrize("caller_value", [None, "0", "off"])
def test_strict_codegen_is_on_when_the_capability_runs(one_job, monkeypatch, caller_value):
    _, args, seen, _ = one_job
    if caller_value is None:
        monkeypatch.delenv("BNGSIM_CODEGEN_STRICT", raising=False)
    else:
        # A caller that turned it off must not get a job that falls back silently.
        monkeypatch.setenv("BNGSIM_CODEGEN_STRICT", caller_value)
    assert worker.main(args) == 0
    assert seen == {"BNGSIM_CODEGEN_STRICT": "1"}


def test_result_records_the_forced_settings(one_job, monkeypatch):
    job, args, _, results = one_job
    monkeypatch.delenv("BNGSIM_CODEGEN_STRICT", raising=False)
    assert worker.main(args) == 0
    record = json.loads(result_path(results, job).read_text())
    assert record["outcome"] == "PASS"
    assert record["environment"]["harness"]["forced_env"] == {"BNGSIM_CODEGEN_STRICT": "1"}


@pytest.mark.parametrize("timing", [False, True])
def test_forced_settings_are_part_of_the_freshness_key(timing):
    # Changing a forced setting must make every old result stale.
    assert env.fingerprint(timing=timing)["harness"]["forced_env"] == {
        "BNGSIM_CODEGEN_STRICT": "1"}
