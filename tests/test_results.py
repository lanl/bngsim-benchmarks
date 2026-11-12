"""Result files and freshness: a result stands only while nothing that decides it
has changed, and a write is never seen half-done."""

from __future__ import annotations

import json
import os

import pytest

from bench import env, jobs, results, specs
from bench.jobs import CORRECTNESS, TIMING


def fp(build="0b7b3424410a", node="CPU X | arm64 | Darwin"):
    base = {
        "bngsim": {"version": "0.17.0", "build_commit": build},
        "bionetgen": {"version": "2.9.3", "sha256": "c" * 64},
        "harness": {"bench": "0.1.0"},
    }
    return {CORRECTNESS: base, TIMING: {**base, "node_type": node}}


@pytest.fixture
def problem(spec_data, write_spec):
    return specs.load(write_spec(spec_data))


@pytest.fixture
def pair(problem):
    correctness, timing = jobs.jobs_for([problem])
    return correctness, timing


def put(results_dir, job, fps, outcome="PASS"):
    f = fps[job.kind]
    results.write_result(results_dir, job, {
        "outcome": outcome, "key": results.freshness_key(job, f),
        "key_parts": results.key_parts(job, f),
    })


def test_key_is_stable_for_the_same_job_and_environment(pair):
    c, _ = pair
    assert results.freshness_key(c, fp()[CORRECTNESS]) == results.freshness_key(c, fp()[CORRECTNESS])


def test_key_changes_with_protocol_build_and_bionetgen(spec_data, write_spec, pair):
    c, _ = pair
    k = results.freshness_key(c, fp()[CORRECTNESS])
    spec_data["protocol"]["ode"]["rtol"] = 1.0e-9
    c2 = jobs.jobs_for([specs.load(write_spec(spec_data))])[0]
    assert results.freshness_key(c2, fp()[CORRECTNESS]) != k
    assert results.freshness_key(c, fp(build="deadbeef0000")[CORRECTNESS]) != k
    other_bng = fp()[CORRECTNESS] | {"bionetgen": {"version": "2.9.3", "sha256": "d" * 64}}
    assert results.freshness_key(c, other_bng) != k


def test_node_type_decides_timing_freshness_only(monkeypatch):
    # The real fingerprint function: a correctness verdict holds on any machine.
    monkeypatch.setattr(env, "node_type", lambda: "CPU A | x86_64 | Linux")
    a_c, a_t = env.fingerprint(timing=False), env.fingerprint(timing=True)
    monkeypatch.setattr(env, "node_type", lambda: "CPU B | x86_64 | Linux")
    b_c, b_t = env.fingerprint(timing=False), env.fingerprint(timing=True)
    assert a_c == b_c
    assert a_t != b_t


def test_swapped_loader_decides_timing_freshness_only(monkeypatch):
    # Running run_network under a different glibc changes its absolute speed, so
    # it stales timing results -- but a correctness verdict is unaffected.
    monkeypatch.setattr(env, "node_type", lambda: "CPU A | x86_64 | Linux")
    monkeypatch.setattr(env, "glibc_identity",
                        lambda: {"swapped": True, "tag": "rocky-9.5", "glibc_version": "2.34"})
    a_c, a_t = env.fingerprint(timing=False), env.fingerprint(timing=True)
    monkeypatch.setattr(env, "glibc_identity",
                        lambda: {"swapped": True, "tag": "rocky-9.6", "glibc_version": "2.34"})
    b_c, b_t = env.fingerprint(timing=False), env.fingerprint(timing=True)
    assert a_c == b_c
    assert "loader" not in a_c
    assert a_t != b_t
    assert a_t["loader"] == {"tag": "rocky-9.5", "glibc_version": "2.34"}


def test_native_loader_is_absent_from_the_fingerprint(monkeypatch):
    monkeypatch.setattr(env, "glibc_identity", lambda: {"swapped": False})
    assert "loader" not in env.fingerprint(timing=True)


def test_write_then_read_round_trips(tmp_path, pair):
    c, _ = pair
    path = results.write_result(tmp_path, c, {"outcome": "PASS", "x": 1.5})
    assert path == tmp_path / f"{c.id}.json"
    assert results.read_result(tmp_path, c) == {"outcome": "PASS", "x": 1.5}


def test_interrupted_write_leaves_old_result_and_no_temp_file(tmp_path, pair, monkeypatch):
    c, _ = pair
    results.write_result(tmp_path, c, {"outcome": "PASS"})

    def boom(*a, **k):
        raise KeyboardInterrupt  # the process is stopped mid-write

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(KeyboardInterrupt):
        results.write_result(tmp_path, c, {"outcome": "DIFF"})
    assert results.read_result(tmp_path, c) == {"outcome": "PASS"}
    leftovers = [p.name for p in results.result_path(tmp_path, c).parent.iterdir()]
    assert leftovers == ["correctness.json"]


def test_unreadable_result_counts_as_missing(tmp_path, pair):
    c, t = pair
    path = results.result_path(tmp_path, c)
    path.parent.mkdir(parents=True)
    path.write_text("{ truncated")
    by_id = {c.id: c, t.id: t}
    assert results.status(c, tmp_path, fp(), by_id).state == "missing"


def test_status_through_a_jobs_life(tmp_path, pair):
    c, t = pair
    by_id = {c.id: c, t.id: t}
    fps = fp()
    assert results.status(c, tmp_path, fps, by_id).state == "missing"
    assert results.status(t, tmp_path, fps, by_id).state == "waiting"

    put(tmp_path, c, fps, outcome="PASS")
    assert results.status(c, tmp_path, fps, by_id).state == "fresh"
    assert results.status(t, tmp_path, fps, by_id).state == "missing"

    put(tmp_path, t, fps)
    assert results.status(t, tmp_path, fps, by_id).state == "fresh"

    # A new bngsim build: both go stale, and the reason names what changed.
    newer = fp(build="1111111aaaaa")
    st = results.status(c, tmp_path, newer, by_id)
    assert st.state == "stale" and "bngsim" in st.reason
    assert results.status(t, tmp_path, newer, by_id).state == "waiting"


def test_timing_blocked_when_correctness_did_not_pass(tmp_path, pair):
    c, t = pair
    by_id = {c.id: c, t.id: t}
    put(tmp_path, c, fp(), outcome="DIFF")
    st = results.status(t, tmp_path, fp(), by_id)
    assert st.state == "blocked" and "DIFF" in st.reason


def test_moving_to_another_node_type_stales_timing_not_correctness(tmp_path, pair):
    c, t = pair
    by_id = {c.id: c, t.id: t}
    put(tmp_path, c, fp())
    put(tmp_path, t, fp())
    moved = fp(node="CPU Y | x86_64 | Linux")
    assert results.status(c, tmp_path, moved, by_id).state == "fresh"
    st = results.status(t, tmp_path, moved, by_id)
    assert st.state == "stale" and "node_type" in st.reason


def test_result_files_are_plain_json(tmp_path, pair):
    c, _ = pair
    path = results.write_result(tmp_path, c, {"outcome": "PASS", "values": [1, 2]})
    assert json.loads(path.read_text())["values"] == [1, 2]
