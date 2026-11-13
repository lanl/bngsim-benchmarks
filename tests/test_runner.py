"""The runner's scheduling: phase order, deadlines, timeouts, limits, freshness.

A fake launcher stands in for the worker process: it writes the result a worker
would write (or none, or reports a timeout), and records what it was asked to
run. A fake clock makes the deadline arithmetic exact.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from bench import jobs, results, specs
from bench.jobs import CORRECTNESS, TIMING
from bench.runner import Launch, RunConfig, run

FPS = {
    CORRECTNESS: {"bngsim": {"build_commit": "abc"}},
    TIMING: {"bngsim": {"build_commit": "abc"}, "node_type": "n"},
}


class FakeLauncher:
    """Records calls; ``outcomes`` maps job id -> outcome, "timeout" or "crash"."""

    def __init__(self, outcomes=None, seconds=1.0):
        self.outcomes = outcomes or {}
        self.seconds = seconds
        self.calls: list[tuple[str, float]] = []
        self.lock = threading.Lock()

    def __call__(self, job, timeout, cpu, cfg):
        with self.lock:
            self.calls.append((job.id, timeout))
        what = self.outcomes.get(job.id, "PASS")
        if what == "timeout":
            return Launch(None, True, timeout)
        if what == "crash":
            return Launch(1, False, self.seconds, "Traceback ... boom")
        f = FPS[job.kind]
        results.write_result(cfg.results_dir, job, {
            "outcome": what, "key": results.freshness_key(job, f),
            "key_parts": results.key_parts(job, f),
        })
        return Launch(0, False, self.seconds)


@pytest.fixture
def two_problems(spec_data, write_spec, tmp_path):
    write_spec(spec_data)
    write_spec(dict(spec_data, id="second_model"))
    return jobs.jobs_for(specs.load_all(tmp_path / "problems"))


def cfg_for(tmp_path, **kw):
    return RunConfig(problems_dir=tmp_path / "problems", results_dir=tmp_path / "results",
                     cache_dir=tmp_path / "cache", quiet=True, **kw)


def ids(calls):
    return [c[0] for c in calls]


def test_correctness_phase_runs_before_timing(tmp_path, two_problems):
    fake = FakeLauncher()
    s = run(two_problems, cfg_for(tmp_path), fingerprints=FPS, launcher=fake)
    kinds = [jid.rsplit("/", 1)[1] for jid in ids(fake.calls)]
    assert kinds == ["correctness", "correctness", "timing", "timing"]
    assert set(s.outcomes.values()) == {"PASS"}


def test_timing_not_run_when_correctness_fails(tmp_path, two_problems):
    from bench.capabilities import ode

    base = f"ode/second_model/{ode.engines_id()}/auto"
    fake = FakeLauncher({f"{base}/correctness": "DIFF"})
    s = run(two_problems, cfg_for(tmp_path), fingerprints=FPS, launcher=fake)
    assert f"{base}/timing" not in ids(fake.calls)
    assert s.skipped[f"{base}/timing"] == "blocked"


def test_second_run_finds_everything_fresh(tmp_path, two_problems):
    run(two_problems, cfg_for(tmp_path), fingerprints=FPS, launcher=FakeLauncher())
    fake = FakeLauncher()
    s = run(two_problems, cfg_for(tmp_path), fingerprints=FPS, launcher=fake)
    assert fake.calls == []
    assert set(s.skipped.values()) == {"fresh"}
    forced = FakeLauncher()
    run(two_problems, cfg_for(tmp_path, force=True), fingerprints=FPS, launcher=forced)
    assert len(forced.calls) == 4


def test_job_that_would_overrun_the_deadline_is_not_started(tmp_path, two_problems):
    t = [1_000_000.0]
    est = two_problems[0].est_seconds
    # Room for exactly one estimated job after the margin: (100 + est) - 100 = est.
    cfg = cfg_for(tmp_path, until=t[0] + 100 + est, margin_seconds=100)
    inner = FakeLauncher()

    def takes_its_estimate(job, timeout, cpu, c):
        t[0] += job.est_seconds  # the job uses exactly its estimate
        return inner(job, timeout, cpu, c)

    s = run(two_problems, cfg, fingerprints=FPS, launcher=takes_its_estimate,
            clock=lambda: t[0])
    assert len(inner.calls) == 1
    first, second = two_problems[0], two_problems[1]
    # The other correctness job and the started job's timing job did not fit...
    assert sorted(s.deferred) == sorted([second.id, f"{first.id.rsplit('/', 1)[0]}/timing"])
    # ...and the deferred correctness job's timing job waits for it.
    assert s.skipped[f"{second.id.rsplit('/', 1)[0]}/timing"] == "waiting"
    # The started job may use only the time left before the margin.
    assert inner.calls[0][1] == pytest.approx(min(two_problems[0].timeout_seconds, est))


def test_job_cut_at_the_deadline_leaves_no_result(tmp_path, two_problems):
    now = 0.0
    first = two_problems[0]
    cfg = cfg_for(tmp_path, until=now + 10 + first.est_seconds + 5, margin_seconds=10)
    fake = FakeLauncher({first.id: "timeout"})
    s = run(two_problems[:1], cfg, fingerprints=FPS, launcher=fake, clock=lambda: now)
    assert s.cut == [first.id]
    assert results.read_result(cfg.results_dir, first) is None


def test_job_over_its_own_timeout_records_timeout(tmp_path, two_problems):
    first = two_problems[0]
    fake = FakeLauncher({first.id: "timeout"})
    cfg = cfg_for(tmp_path)
    s = run(two_problems[:1], cfg, fingerprints=FPS, launcher=fake)
    assert s.outcomes[first.id] == "TIMEOUT"
    # A timeout is a result: it is current, and the job does not run again.
    st = results.status(first, cfg.results_dir, FPS, {first.id: first})
    assert st.state == "fresh" and st.outcome == "TIMEOUT"


def test_worker_crash_is_reported_without_a_result(tmp_path, two_problems):
    first = two_problems[0]
    s = run(two_problems[:1], cfg_for(tmp_path), fingerprints=FPS,
            launcher=FakeLauncher({first.id: "crash"}))
    assert "boom" in s.failed[first.id]
    assert first.id not in s.outcomes


def test_limit_caps_launches(tmp_path, two_problems):
    fake = FakeLauncher()
    run(two_problems, cfg_for(tmp_path, limit=1), fingerprints=FPS, launcher=fake)
    assert len(fake.calls) == 1


def test_pool_never_exceeds_its_worker_count(tmp_path, spec_data, write_spec):
    for i in range(6):
        write_spec(dict(spec_data, id=f"model_{i}"))
    all_jobs = [j for j in jobs.jobs_for(specs.load_all(tmp_path / "problems"))
                if j.kind == CORRECTNESS]
    live = 0
    peak = 0
    lock = threading.Lock()
    inner = FakeLauncher()

    def slow(job, timeout, cpu, cfg):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        time.sleep(0.05)
        try:
            return inner(job, timeout, cpu, cfg)
        finally:
            with lock:
                live -= 1

    run(all_jobs, cfg_for(tmp_path, workers=3), fingerprints=FPS, launcher=slow)
    assert len(inner.calls) == 6
    assert 1 < peak <= 3


def test_results_land_where_the_job_id_says(tmp_path, two_problems):
    from bench.capabilities import ode

    run(two_problems, cfg_for(tmp_path), fingerprints=FPS, launcher=FakeLauncher())
    expected = tmp_path / "results" / "ode" / "toy_model" / ode.engines_id() / "auto"
    assert sorted(p.name for p in Path(expected).iterdir()) == ["correctness.json", "timing.json"]
