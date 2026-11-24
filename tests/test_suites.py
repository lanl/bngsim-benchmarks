"""Core suites (docs/DESIGN.md section 3.2): one feature on one input format,
exactly SUITE_SIZE problems at positions 1..SUITE_SIZE."""

from __future__ import annotations

import pytest
from conftest import REPO

from bench import cli, specs


def _member(spec_data, i, suite="ode-net", pos=None):
    d = dict(spec_data, id=f"model_{i:02d}", layer="core",
             verification={"kind": "equations", "evidence": f"models/m{i}/verify.ipynb"},
             suites={suite: i if pos is None else pos})
    d["source"] = dict(spec_data["source"], path=f"models/m{i}/m{i}.bngl",
                       sha256=f"{i:064x}")
    return d


def _write_suite(spec_data, write_spec, n=specs.SUITE_SIZE, **kw):
    for i in range(1, n + 1):
        write_spec(_member(spec_data, i, **kw))


def test_a_complete_suite_loads_in_position_order(spec_data, write_spec, tmp_path):
    _write_suite(spec_data, write_spec)
    problems = specs.load_all(tmp_path / "problems")
    members = specs.suite_members(problems)["ode-net"]
    assert [m.suites["ode-net"] for m in members] == list(range(1, specs.SUITE_SIZE + 1))


def test_an_incomplete_suite_is_refused_naming_the_gap(spec_data, write_spec, tmp_path):
    _write_suite(spec_data, write_spec, n=specs.SUITE_SIZE - 1)
    with pytest.raises(specs.SpecError, match=r"9 of 10 members; no member at position 10"):
        specs.load_all(tmp_path / "problems")


def test_a_shared_position_is_refused(spec_data, write_spec, tmp_path):
    _write_suite(spec_data, write_spec)
    write_spec(_member(spec_data, 11, pos=3))
    with pytest.raises(specs.SpecError, match=r"position 3 is held by model_03, model_11"):
        specs.load_all(tmp_path / "problems")


@pytest.mark.parametrize("change, message", [
    ({"suites": {"ode-nope": 1}}, "unknown suite 'ode-nope'"),
    ({"suites": {"ode-net": 0}}, "position must be an integer from 1 to 10"),
    ({"suites": {"ode-net": 11}}, "position must be an integer from 1 to 10"),
    ({"suites": {"ode-net": True}}, "position must be an integer from 1 to 10"),
    ({"suites": ["ode-net"]}, "suites: a mapping"),
    ({"layer": "pilot", "verification": {"kind": "pending", "evidence": "x"}},
     "only a core problem may belong to a suite"),
])
def test_bad_membership_is_refused(spec_data, write_spec, change, message):
    d = {**_member(spec_data, 1), **change}
    with pytest.raises(specs.SpecError, match=message):
        specs.load(write_spec(d))


def test_a_member_must_be_in_the_suites_format(spec_data, write_spec):
    d = _member(spec_data, 1)
    d["source"] = dict(d["source"], format="sbml")
    with pytest.raises(specs.SpecError, match="source.format must be one of bngl"):
        specs.load(write_spec(d))


def test_a_suite_name_selects_exactly_its_members_jobs(spec_data, write_spec, tmp_path):
    _write_suite(spec_data, write_spec)
    outsider = dict(spec_data, id="model_01_extra")  # shares a prefix with a member
    outsider["source"] = dict(spec_data["source"], path="models/x/x.bngl", sha256="f" * 64)
    write_spec(outsider)
    problems = specs.load_all(tmp_path / "problems")
    pats = cli._expand_suites(["ode-net"], problems)
    assert pats == [f"ode/model_{i:02d}/" for i in range(1, 11)]
    from bench import jobs as jobs_mod

    chosen = jobs_mod.select(jobs_mod.jobs_for(problems), pats)
    assert {j.problem for j in chosen} == {f"model_{i:02d}" for i in range(1, 11)}


def test_a_member_of_two_suites_gives_each_suite_only_its_capabilitys_jobs(
        spec_data, write_spec, tmp_path):
    from bench import jobs as jobs_mod

    steady = {"netgen": "generate_network({overwrite=>1})", "t_end": 100, "rtol": 1.0e-8,
              "atol": 1.0e-8}
    for i in range(1, specs.SUITE_SIZE + 1):
        d = _member(spec_data, i)
        if i == 1:  # also the steady-state suite's first member
            d["capabilities"] = ["ode", "steady_state"]
            d["protocol"] = {**d["protocol"], "steady_state": dict(steady)}
            d["suites"] = {"ode-net": 1, "ss-net": 1}
        write_spec(d)
    for i in range(2, specs.SUITE_SIZE + 1):
        d = _member(spec_data, 20 + i, suite="ss-net", pos=i)
        d["capabilities"] = ["steady_state"]
        d["protocol"] = {"steady_state": dict(steady)}
        write_spec(d)
    problems = specs.load_all(tmp_path / "problems")
    every = jobs_mod.jobs_for(problems)
    ode_jobs = jobs_mod.select(every, cli._expand_suites(["ode-net"], problems))
    ss_jobs = jobs_mod.select(every, cli._expand_suites(["ss-net"], problems))
    assert {j.capability for j in ode_jobs} == {"ode"} and len(ode_jobs) == 2 * 10
    assert {j.capability for j in ss_jobs} == {"steady_state"} and len(ss_jobs) == 2 * 10
    assert "model_01" in {j.problem for j in ode_jobs} & {j.problem for j in ss_jobs}


def test_other_patterns_pass_through(spec_data, write_spec, tmp_path):
    _write_suite(spec_data, write_spec)
    problems = specs.load_all(tmp_path / "problems")
    assert cli._expand_suites(["timing", "model_03"], problems) == ["timing", "model_03"]
    assert cli._expand_suites(None, problems) == []


def test_every_registered_suite_is_committed_whole():
    """A suite is registered when its members land; each must then be complete
    (load_all enforces the size) and drawn from the core layer."""
    members = specs.suite_members(specs.load_all(REPO / "problems"))
    assert set(members) == set(specs.SUITES)
    for name, ms in members.items():
        assert len(ms) == specs.SUITE_SIZE, name
        assert {m.layer for m in ms} == {"core"}, name
