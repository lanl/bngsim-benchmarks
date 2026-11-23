"""The trajectories a correctness job stores beside its result, and re-scoring a
stored result, without re-simulating, when only its scoring is out of date
(``bench rescore``)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bench import cli, jobs, results, specs, worker
from bench.capabilities import CAPABILITIES, ode
from bench.jobs import CORRECTNESS, TIMING
from bench.rescore import rescore_one

# The current correctness fingerprint, and one that differs from it only in the
# vendored comparison core (a scoring part of the key).
FP = {"bngsim": {"version": "0.17.2", "build_commit": "abc"},
      "harness": {"bench": "0.1.0", "parity_core": {"commit": "new"}}}
OLD_CORE = {**FP, "harness": {"bench": "0.1.0", "parity_core": {"commit": "old"}}}


def wide(n_t=101, small=1.0):
    """A wide-dynamic-range pair of species: one at 1e6, one at ``small``."""
    t = np.linspace(0, 10, n_t)[:, None]
    return np.hstack([1e6 * (1 - np.exp(-t)), small * np.exp(-t / 3)])


def pair(n_t, err=0.0):
    """A reference and a BNGsim run ``err`` off on the small species."""
    ref = wide(n_t)
    bn = ref.copy()
    bn[:, 1] *= 1 + err
    return bn, ref


def stored(results_dir, job, *, bn, ref, outcome="PASS", fp=None, definition=None,
           scored=True):
    """A correctness record (and its trajectories) as an earlier run wrote it: its
    key computed for ``definition`` (the job's own by default) and ``fp``."""
    old_job = jobs.Job(job.capability, job.problem, job.engines, job.setting, job.kind,
                       definition=job.definition if definition is None else definition)
    fp = fp or FP
    rec = {"outcome": outcome, "key": results.freshness_key(old_job, fp),
           "key_parts": results.key_parts(old_job, fp),
           "measurements": {"references": {"run_network": {"solver": "dense"}}}}
    if scored:
        rec["verdicts"] = {"run_network": {"passed": True}}
        rec["trajectories"] = results.write_trajectories(results_dir, job, {
            "time": ode.output_times(job.definition["protocol"]),
            "species": np.array(["big", "small"]), "bngsim": bn, "run_network": ref})
    results.write_result(results_dir, job, rec)
    return rec


@pytest.fixture
def corr(spec_data, write_spec):
    problem = specs.load(write_spec(spec_data))
    return next(j for j in ode.jobs(problem) if j.kind == CORRECTNESS)


# --------------------------------------------------------------------------- #
# Stored trajectories
# --------------------------------------------------------------------------- #
def test_trajectories_round_trip_beside_the_record(tmp_path, corr):
    arrays = {"time": np.linspace(0, 1, 3), "species": np.array(["A()", "B()"]),
              "bngsim": np.ones((3, 2))}
    meta = results.write_trajectories(tmp_path, corr, arrays)
    path = results.trajectories_path(tmp_path, corr)
    assert path == results.result_path(tmp_path, corr).with_suffix(".npz") and path.exists()
    assert meta["file"] == path.name and meta["bytes"] == path.stat().st_size
    back = results.read_trajectories(tmp_path, corr, {"trajectories": meta})
    assert list(back["species"]) == ["A()", "B()"]
    np.testing.assert_array_equal(back["bngsim"], arrays["bngsim"])


def test_trajectories_another_run_left_are_never_read_as_this_ones(tmp_path, corr):
    meta = results.write_trajectories(tmp_path, corr, {"bngsim": np.ones((3, 2))})
    results.write_trajectories(tmp_path, corr, {"bngsim": np.zeros((3, 2))})
    with pytest.raises(ValueError, match="sha256"):
        results.read_trajectories(tmp_path, corr, {"trajectories": meta})
    with pytest.raises(LookupError):
        results.read_trajectories(tmp_path, corr, {})
    results.drop_trajectories(tmp_path, corr)
    with pytest.raises(LookupError, match="missing"):
        results.read_trajectories(tmp_path, corr, {"trajectories": meta})


@pytest.fixture
def quiet_env(monkeypatch, tmp_path):
    from bench import env

    monkeypatch.setattr(env, "fingerprint", lambda *, timing: {"bngsim": {"version": "x"}})
    monkeypatch.setattr(env, "environment", lambda: {})
    for k, v in {**env.SINGLE_THREAD_ENV, **env.HARNESS_ENV}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("BNGSIM_CODEGEN_CACHE_DIR", str(tmp_path / "unused"))
    monkeypatch.delenv("BENCH_CPU", raising=False)


def test_the_worker_stores_a_jobs_trajectories_and_the_record_points_at_them(
        tmp_path, spec_data, write_spec, monkeypatch, quiet_env):
    problems = tmp_path / "problems"
    problem = specs.load(write_spec(spec_data, directory=problems))
    job = next(j for j in jobs.jobs_for([problem]) if j.kind == CORRECTNESS)
    args = ["--job", job.id, "--problems", str(problems), "--results",
            str(tmp_path / "results"), "--cache", str(tmp_path / "cache")]
    traj = {"time": np.arange(3.0), "species": np.array(["A()"]), "bngsim": np.ones((3, 1))}
    monkeypatch.setattr(CAPABILITIES["ode"], "run",
                        lambda job, problem, ctx: {"outcome": "PASS", "_trajectories": traj})
    assert worker.main(args) == 0
    rec = json.loads(results.result_path(tmp_path / "results", job).read_text())
    assert "_trajectories" not in rec
    assert rec["trajectories"]["file"] == "correctness.npz"
    back = results.read_trajectories(tmp_path / "results", job, rec)
    np.testing.assert_array_equal(back["bngsim"], np.ones((3, 1)))

    # A later result with nothing to store removes the stale file.
    monkeypatch.setattr(CAPABILITIES["ode"], "run",
                        lambda job, problem, ctx: {"outcome": "BAD_TEST"})
    assert worker.main(args) == 0
    assert not results.trajectories_path(tmp_path / "results", job).exists()


def test_a_correctness_run_hands_back_every_engines_compared_trajectory(monkeypatch, tmp_path):
    from test_ode import _correctness_prep, _decay, _fake_bngsim, _fake_reference

    monkeypatch.setattr(ode, "_prepare", lambda problem, ctx: _correctness_prep())
    monkeypatch.setattr(ode, "BngsimODE", _fake_bngsim())
    monkeypatch.setattr(ode, "REFERENCE_ENGINES", {"good": _fake_reference(),
                                                   "bad": _fake_reference(error="boom")})
    rec = ode.run_correctness(job=None, problem=None, ctx={
        "workdir": str(tmp_path), "subprocess_timeout": 60, "engine_budget_seconds": 30.0})
    stored_arrays = rec["_trajectories"]
    # Every engine that ran (the crasher did not), the output times and the species.
    assert set(stored_arrays) == {"time", "species", "bngsim", "good"}
    assert list(stored_arrays["species"]) == ["A()", "B()"]
    np.testing.assert_allclose(stored_arrays["time"], np.linspace(0, 10, 101))
    np.testing.assert_allclose(stored_arrays["good"], _decay())


# --------------------------------------------------------------------------- #
# Re-scoring
# --------------------------------------------------------------------------- #
def test_a_result_stale_only_in_its_scoring_is_rescored_and_current_again(tmp_path, corr):
    bn, ref = pair(301)
    stored(tmp_path, corr, bn=bn, ref=ref, fp=OLD_CORE)
    by_id, fps = {corr.id: corr}, {CORRECTNESS: FP, TIMING: FP}
    assert results.status(corr, tmp_path, fps, by_id).state == "stale"
    r = rescore_one(corr, tmp_path, FP)
    assert (r.state, r.before, r.after) == ("rescored", "PASS", "PASS")
    rec = results.read_result(tmp_path, corr)
    assert rec["key"] == results.freshness_key(corr, FP)
    assert rec["rescored"][-1]["from_outcome"] == "PASS"
    assert rec["rescored"][-1]["parity_core"] == {"commit": "new"}
    assert rec["verdicts"]["run_network"]["n_cells"] == 301 * 2  # recomputed, not kept
    assert results.status(corr, tmp_path, fps, by_id).state == "fresh"
    # A second pass finds it current and leaves it alone.
    assert rescore_one(corr, tmp_path, FP).state == "current"


def test_a_result_stale_for_more_than_its_scoring_is_not_rescored(tmp_path, corr):
    bn, ref = pair(301)
    stored(tmp_path, corr, bn=bn, ref=ref,
           fp={**OLD_CORE, "bngsim": {"version": "0.17.2", "build_commit": "other"}})
    r = rescore_one(corr, tmp_path, FP)
    assert r.state == "not-rescorable" and "bngsim" in r.reason


def test_a_result_without_its_trajectories_must_re_run(tmp_path, corr):
    bn, ref = pair(301)
    stored(tmp_path, corr, bn=bn, ref=ref, fp=OLD_CORE)
    results.drop_trajectories(tmp_path, corr)
    assert rescore_one(corr, tmp_path, FP).state == "no-trajectories"


def test_a_result_that_scored_nothing_keeps_its_outcome(tmp_path, corr):
    stored(tmp_path, corr, bn=None, ref=None, outcome="BAD_TEST", fp=OLD_CORE, scored=False)
    r = rescore_one(corr, tmp_path, FP)
    assert (r.state, r.before, r.after) == ("rescored", "BAD_TEST", "BAD_TEST")


def test_rescore_keeps_reference_failures_and_their_attribution(tmp_path, corr):
    bn, ref = pair(301)
    rec = stored(tmp_path, corr, bn=bn, ref=ref, outcome="REFERENCE_FAILED", fp=OLD_CORE)
    rec["copasi_error"] = "KeyError: ['I']"
    results.write_result(tmp_path, corr, rec)
    assert rescore_one(corr, tmp_path, FP).after == "REFERENCE_FAILED"
    assert results.read_result(tmp_path, corr)["copasi_error"] == "KeyError: ['I']"


def test_bench_rescore_rescores_the_selected_results(tmp_path, spec_data, write_spec,
                                                     monkeypatch, capsys):
    problems = tmp_path / "problems"
    problem = specs.load(write_spec(spec_data, directory=problems))
    job = next(j for j in ode.jobs(problem) if j.kind == CORRECTNESS)
    bn, ref = pair(301)
    stored(tmp_path / "results", job, bn=bn, ref=ref, fp=OLD_CORE)
    monkeypatch.setattr("bench.results.fingerprints_now", lambda: {CORRECTNESS: FP, TIMING: FP})
    argv = ["rescore", "--problems", str(problems), "--results", str(tmp_path / "results")]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert f"rescored         {job.id}  PASS" in out
    assert "rescored 1; 0 outcomes changed" in out
    assert cli.main(argv) == 0
    assert "current 1; 0 outcomes changed" in capsys.readouterr().out
