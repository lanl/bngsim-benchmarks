"""Job generation: ids are deterministic and readable, timing jobs depend on
their correctness job, and the definition carries everything that decides the
computation."""

from __future__ import annotations

import pytest

from bench import jobs, specs


@pytest.fixture
def problem(spec_data, write_spec):
    return specs.load(write_spec(spec_data))


def test_job_ids_follow_the_documented_scheme(problem):
    ids = [j.id for j in jobs.jobs_for([problem])]
    # capability / problem / engines / setting / kind (docs/DESIGN.md 5.2)
    assert ids == [
        "ode/toy_model/bngsim-vs-run_network/auto/correctness",
        "ode/toy_model/bngsim-vs-run_network/auto/timing",
    ]


def test_timing_job_requires_its_correctness_job(problem):
    correctness, timing = jobs.jobs_for([problem])
    assert correctness.requires is None
    assert timing.requires == correctness.id


def test_definition_pins_model_bytes_and_filled_protocol(problem):
    correctness, timing = jobs.jobs_for([problem])
    d = correctness.definition
    assert d["source_sha256"] == "b" * 64
    # Defaults are made explicit, so an omitted key and its default give one key.
    assert d["protocol"]["t_start"] == 0
    assert d["protocol"]["sparse"] is False
    assert "timing" not in d and "timing" in timing.definition


def test_generation_is_deterministic(problem):
    a = [(j.id, j.canonical()) for j in jobs.jobs_for([problem])]
    b = [(j.id, j.canonical()) for j in jobs.jobs_for([problem])]
    assert a == b


def test_correctness_jobs_come_before_timing_jobs(spec_data, write_spec):
    second = dict(spec_data, id="another_model")
    probs = [specs.load(write_spec(spec_data)), specs.load(write_spec(second))]
    kinds = [j.kind for j in jobs.jobs_for(probs)]
    assert kinds == ["correctness", "correctness", "timing", "timing"]


def test_selecting_a_timing_job_brings_its_requirement(problem):
    all_jobs = jobs.jobs_for([problem])
    chosen = jobs.select(all_jobs, ["timing"])
    assert {j.kind for j in chosen} == {"correctness", "timing"}
    assert jobs.select(all_jobs, None) == all_jobs
    assert jobs.select(all_jobs, ["no-such-problem"]) == []


def test_non_bngl_source_gets_no_bngl_ode_jobs(spec_data, write_spec):
    spec_data["source"]["format"] = "sbml"
    assert jobs.jobs_for([specs.load(write_spec(spec_data))]) == []
