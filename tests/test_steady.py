"""The steady-state capability's own logic, with fake engines (no simulator): the
protocol and its points, job generation and planning, the tolerance-scaled
agreement and the engine-free checks, the classification of a root, and how
per-engine states become an outcome in the correctness and timing jobs."""

from __future__ import annotations

import json
import math
import types
from pathlib import Path

import numpy as np
import pytest
from test_netfile import BINDING, TOGGLE

from bench import specs
from bench.capabilities import ode, steady

PROTOCOL = {"netgen": "generate_network({overwrite=>1})", "t_end": 100, "rtol": 1.0e-8,
            "atol": 1.0e-8}


@pytest.fixture
def ss_spec(spec_data):
    spec_data["capabilities"] = ["steady_state"]
    spec_data["protocol"] = {"steady_state": dict(PROTOCOL)}
    spec_data["size"] = {"species": 3, "reactions": 2}
    return spec_data


# --------------------------------------------------------------------------- #
# Specs, points, jobs and planning
# --------------------------------------------------------------------------- #
def test_engines_id_names_bngsim_against_every_reference_in_odes_order():
    assert steady.engines_id() == "bngsim-vs-run_network-roadrunner-copasi-amici"
    assert list(steady.REFERENCE_ENGINES)[0] == steady.INTEGRATION_REFERENCE


def test_registry_emits_one_correctness_and_one_timing_job(ss_spec, write_spec):
    problem = specs.load(write_spec(ss_spec))
    correctness, timing = steady.jobs(problem)
    assert correctness.id == f"steady_state/toy_model/{steady.engines_id()}/newton/correctness"
    assert timing.requires == correctness.id
    # Defaults are part of the definition, so changing one restales the results.
    proto = correctness.definition["protocol"]
    assert proto["n_steps"] == steady.DEFAULT_N_STEPS and proto["parameters"] == {}
    # The verdict keys correctness only; the timing protocol, timing only; how each
    # engine is asked for its steady state keys both.
    assert correctness.definition["verdict"] == steady.VERDICT
    assert "verdict" not in timing.definition
    assert timing.definition["timing"] == steady.TIMING_PROTOCOL
    for job in (correctness, timing):
        assert job.definition["reference_settings"] == steady.REFERENCE_SETTINGS
        assert job.definition["setting"]["bngsim"] == steady.BNGSIM_SETTINGS
    assert correctness.timeout_seconds >= (len(steady.REFERENCE_ENGINES) + 3) \
        * steady.engine_budget_seconds(problem)


def test_reference_versions_key_only_this_capability(ss_spec, write_spec, monkeypatch):
    monkeypatch.setattr(steady.env, "roadrunner_identity", lambda: {"version": "9.9.9"})
    monkeypatch.setattr(steady.env, "copasi_identity", lambda: {"version": "4.48.311"})
    monkeypatch.setattr(steady.env, "amici_identity", lambda: {"version": "1.1.0"})
    correctness, _ = steady.jobs(specs.load(write_spec(ss_spec)))
    assert correctness.definition["reference_versions"] == {
        "roadrunner": "9.9.9", "copasi": "4.48.311", "amici": "1.1.0"}


def test_non_bngl_source_gets_no_jobs(ss_spec, write_spec):
    ss_spec["source"]["format"] = "sbml"
    assert steady.jobs(specs.load(write_spec(ss_spec))) == []


def test_a_spec_can_carry_ode_and_steady_state_and_the_ode_jobs_do_not_change(
        spec_data, write_spec):
    ode_only = specs.load(write_spec(spec_data))
    spec_data["capabilities"] = ["ode", "steady_state"]
    spec_data["protocol"]["steady_state"] = dict(PROTOCOL)
    both = specs.load(write_spec(spec_data))
    assert [j.canonical() for j in ode.jobs(ode_only)] == [j.canonical() for j in ode.jobs(both)]
    assert len(steady.jobs(both)) == 2


@pytest.mark.parametrize("change, message", [
    ({"t_end": 0}, "t_end: must be a positive number"),
    ({"t_end": "1e5"}, "t_end: must be a positive number"),
    ({"n_steps": 1}, "n_steps: an integer from 2"),
    ({"rtol": 0}, "rtol: must be a positive number"),
    ({"atol": -1.0}, "atol: must be a positive number"),
    ({"netgen": "simulate({})"}, "netgen"),
    ({"t_start": 5}, "unknown key 't_start'"),
    ({"parameters": {"k": "big"}}, "parameters.k: must be a number"),
    ({"parameters": ["k"]}, "parameters: a mapping"),
    ({"scan": {"parameter": "k", "min": 1, "max": 10}}, "missing 'points'"),
    ({"scan": {"parameter": "k", "min": 10, "max": 1, "points": 5}}, "max must be greater"),
    ({"scan": {"parameter": "k", "min": 0, "max": 1, "points": 5, "log": True}}, "needs min > 0"),
    ({"scan": {"parameter": "k", "min": 1, "max": 2, "points": 1}}, "points: an integer from 2"),
    ({"scan": {"parameter": "k", "values": [1, 2], "min": 1}}, "not both"),
    ({"scan": {"parameter": "k", "values": []}}, "values: a list of 1"),
    ({"scan": {"parameter": "k", "values": [1, 1]}}, "each value once"),
    ({"scan": {"values": [1, 2]}}, "scan.parameter"),
    ({"scan": {"parameter": "k", "values": [1], "step": 2}}, "unknown key 'step'"),
    ({"scan": {"parameter": "k", "values": [1]}, "parameters": {"k": 2}}, "also set in parameters"),
])
def test_invalid_steady_state_protocol_is_refused_with_its_name(ss_spec, write_spec, change,
                                                                message):
    ss_spec["protocol"]["steady_state"].update(change)
    with pytest.raises(specs.SpecError, match=message.replace("(", r"\(")):
        specs.load(write_spec(ss_spec))


def test_points_are_one_without_a_scan_and_one_per_value_with_one():
    assert steady.points(dict(PROTOCOL, parameters={})) == [{}]
    assert steady.points(dict(PROTOCOL, parameters={"A0": 3})) == [{"A0": 3.0}]
    pts = steady.points(dict(PROTOCOL, parameters={"A0": 3},
                             scan={"parameter": "kr", "values": [2, 0.5]}))
    assert pts == [{"A0": 3.0, "kr": 2.0}, {"A0": 3.0, "kr": 0.5}]
    log = steady.points(dict(PROTOCOL, scan={"parameter": "kr", "min": 1e-3, "max": 1.0,
                                             "points": 4, "log": True}))
    assert [p["kr"] for p in log] == pytest.approx([1e-3, 1e-2, 1e-1, 1.0])
    lin = steady.points(dict(PROTOCOL, scan={"parameter": "kr", "min": 1, "max": 3,
                                             "points": 3, "log": False}))
    assert [p["kr"] for p in lin] == [1.0, 2.0, 3.0]


def test_planning_grows_with_points_and_size_within_its_budget(ss_spec, write_spec):
    one = specs.load(write_spec(ss_spec))
    ss_spec["protocol"]["steady_state"]["scan"] = {"parameter": "k", "min": 1, "max": 2,
                                                   "points": 100}
    many = specs.load(write_spec(ss_spec))
    assert steady.planning_seconds(many)[0] > steady.planning_seconds(one)[0]
    assert steady.planning_seconds(many)[1] > steady.planning_seconds(one)[1] > 0
    assert steady.engine_budget_seconds(one) == steady.MIN_TIMEOUT_SECONDS
    huge = types.SimpleNamespace(size={"reactions": 10**6},
                                 protocol_for=lambda c: dict(PROTOCOL))
    assert steady.engine_budget_seconds(huge) == steady.ENGINE_BUDGET_CAP_SECONDS


# --------------------------------------------------------------------------- #
# The agreement and the engine-free checks
# --------------------------------------------------------------------------- #
NAMES = ["A()", "B()", "C()"]


def test_identical_states_agree():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    v = steady.agreement(x, x.copy(), 1e-8, NAMES)
    assert v["passed"] and v["max_ratio"] == 0.0 and v["points_agreed"] == 2


def test_a_relative_difference_past_rtol_fails_and_names_the_species():
    a = np.array([[1.0, 2.0, 3.0]])
    b = a.copy()
    b[0, 1] *= 1 + 1e-3  # ten times the 1e-4 tolerance
    v = steady.agreement(a, b, 1e-8, NAMES)
    assert not v["passed"] and v["worst_species"] == "B()"
    assert v["max_ratio"] == pytest.approx(10, rel=2e-3)  # 1e-3 over rtol*|x| + the floor
    b[0, 1] = a[0, 1] * (1 + 5e-5)
    assert steady.agreement(a, b, 1e-8, NAMES)["passed"]


def test_a_species_far_below_the_peak_is_held_to_the_floor():
    a = np.array([[1e6, 1e-6, 1.0]])
    b = a.copy()
    b[0, 1] = 3e-6  # a 200% miss, far below 1e-9 of the peak (1e-3)
    assert steady.agreement(a, b, 1e-8, NAMES)["passed"]


def test_another_branch_at_one_point_fails_that_point_and_is_not_averaged_away():
    a = np.tile([[155.8, 0.1, 1.0]], (100, 1))
    b = a.copy()
    b[42] = [0.33, 11.7, 1.0]  # the other stable state, at one dose of a hundred
    v = steady.agreement(a, b, 1e-8, NAMES)
    assert not v["passed"] and v["points_agreed"] == 99 and v["worst_point"] == 42


def test_non_finite_states_and_mismatched_shapes_fail():
    a = np.array([[1.0, 2.0, 3.0]])
    b = a.copy()
    b[0, 2] = np.nan
    assert math.isinf(steady.agreement(a, b, 1e-8, NAMES)["max_ratio"])
    assert not steady.agreement(a, np.ones((2, 3)), 1e-8, NAMES)["passed"]


def test_a_write_only_accumulator_is_left_out_of_the_agreement():
    a = np.array([[1.0, 2.0, 3.0]])
    b = a.copy()
    b[0, 2] = 300.0
    assert not steady.agreement(a, b, 1e-8, NAMES)["passed"]
    assert steady.agreement(a, b, 1e-8, NAMES, sinks=[2])["passed"]


def test_settledness_is_the_tails_largest_move_over_the_tolerance():
    flat = np.tile([[1.0, 2.0, 3.0]], (5, 1))
    assert steady.settledness(flat, 1e-8) == 0.0
    drifting = flat.copy()
    drifting[:, 0] = [0.99, 0.995, 0.999, 0.9999, 1.0]
    assert steady.settledness(drifting, 1e-8) == pytest.approx(0.01 / 1e-4, rel=1e-3)


# A + B <-> C with kf = 1 from A0, B0: kf (A0 - C)(B0 - C) = kr C, so the steady C is
# the smaller root of C^2 - (A0 + B0 + kr) C + A0 B0 = 0. The larger root solves
# f = 0 on the same conservation laws, but with A and B negative.
def binding_state(kr: float, a0: float = 2.0, b0: float = 1.0, root: int = -1) -> list[float]:
    s = a0 + b0 + kr
    c = (s + root * math.sqrt(s * s - 4 * a0 * b0)) / 2
    return [a0 - c, b0 - c, c]


SCAN = [{"kr": 0.5}, {"kr": 1.0}, {"kr": 2.0}]


def binding_prep(pts=SCAN, text=BINDING, names=NAMES):
    return {"species_names": names, "points": pts, "info": {},
            **steady.network_checks(text, pts, names)}


def test_network_checks_read_each_point_and_find_the_laws():
    prep = binding_prep([{"A0": 4.0}, {"A0": 1.0}])
    assert prep["x0"].tolist() == [[4.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    assert prep["laws"].shape == (2, 3) and prep["sinks"] == []
    with pytest.raises(ValueError, match="no parameter 'nope'"):
        steady.network_checks(BINDING, [{"nope": 1.0}], NAMES)
    with pytest.raises(ValueError, match="expression of others"):
        steady.network_checks(BINDING, [{"half_B0": 1.0}], NAMES)


def test_the_steady_state_passes_the_engine_free_checks():
    prep = binding_prep()
    states = np.array([binding_state(p["kr"]) for p in SCAN])
    check = steady.engine_free(states, prep)
    assert check["passed"]
    assert check["residual"]["evaluated"] and check["residual"]["max"] < 1e-12
    assert check["conservation"]["n_laws"] == 2 and check["conservation"]["max"] < 1e-12


def test_a_state_that_is_not_a_steady_state_fails_the_residual():
    prep = binding_prep()
    states = np.array([binding_state(p["kr"]) for p in SCAN])
    states[1] = [1.9, 0.9, 0.1]  # on the laws, but not at rest
    check = steady.engine_free(states, prep)
    assert not check["passed"] and check["conservation"]["passed"]
    assert check["residual"]["worst_point"] == 1
    assert steady._engine_free_error(check, prep).startswith("NOT_A_STEADY_STATE: at point 1")


def test_a_root_off_the_initial_states_laws_fails_conservation():
    # A dose of A0 the engine did not take into its initial state: a steady state
    # of the network, but of another total.
    pts = [{"A0": 4.0}, {"A0": 2.0}]
    prep = binding_prep(pts)
    states = np.array([binding_state(1.0, a0=2.0), binding_state(1.0, a0=2.0)])
    check = steady.engine_free(states, prep)
    assert check["residual"]["passed"] and not check["conservation"]["passed"]
    assert check["conservation"]["worst_point"] == 0
    assert steady._engine_free_error(check, prep).startswith("OFF_MANIFOLD: at point 0 (A0=4)")


def test_a_rate_law_the_reader_cannot_evaluate_leaves_only_the_residual_out():
    text = BINDING.replace("1 1,2 3 kf #bind", "1 1,2 3 MM kf kr #bind")
    prep = binding_prep(text=text)
    check = steady.engine_free(np.array([binding_state(p["kr"]) for p in SCAN]), prep)
    assert check["passed"] and not check["residual"]["evaluated"]
    assert check["conservation"]["passed"] and check["conservation"]["n_laws"] == 2


def test_the_toggle_switchs_saddle_is_unstable_and_its_branches_stable():
    prep = {"species_names": ["R1()", "R2()"], "points": [{}], "info": {},
            **steady.network_checks(TOGGLE, [{}], ["R1()", "R2()"])}
    saddle = np.array([1.3165460714061394, 6.7341634999431825])
    u_branch = np.array([155.76340773977222, 0.09951301917279096])
    assert steady.engine_free(np.array([saddle, u_branch]), {**prep, "points": [{}, {}],
                              "point_networks": prep["point_networks"] * 2,
                              "x0": np.zeros((2, 2))})["passed"]
    assert steady.stability(saddle, 0, prep)["stability"] == "unstable"
    assert steady.stability(u_branch, 0, prep)["stability"] == "stable"
    assert steady.stability(np.full(2, np.nan), 0, prep)["stability"] == "undetermined"


# --------------------------------------------------------------------------- #
# Correctness and timing, with fake engines
# --------------------------------------------------------------------------- #
def exact_states(pts=SCAN, **kw) -> np.ndarray:
    return np.array([binding_state(p["kr"], **kw) for p in pts])


def _prep(pts=SCAN):
    proto = dict(PROTOCOL, n_steps=20, parameters={})
    return {"net": Path("m.net"), "sbml": Path("m.xml"), "proto": proto,
            "info": {"network": {"species": 3, "reactions": 2}, "points": pts},
            "species_names": NAMES, "name_to_id": {n: n[0] for n in NAMES},
            "points": pts, "point_nets": [Path(f"p{i}.net") for i in range(len(pts))],
            **steady.network_checks(BINDING, pts, NAMES)}


def _fake_bngsim(states=None, boom=None, sleep=0.0, integration=None):
    class FakeBngsim:
        def __init__(self, net, proto, pts, method, keep=None):
            self.method = method
            if boom and method == "newton":
                raise RuntimeError(boom)

        def solve_all(self, detailed=False):
            if sleep and self.method == "newton":
                import time
                time.sleep(sleep)
            if self.method == "integration" and integration is not None:
                return np.array(integration)
            return np.array(exact_states() if states is None else states)

        def config(self):
            return {"method": self.method, "codegen_backend": "exprtk", "load_seconds": 0.01,
                    "codegen_seconds": 0.001}

    return FakeBngsim


def _fake_reference(states=None, boom=None, sleep=0.0, settled=None, input_="sbml"):
    class FakeRef:
        INPUT = input_

        def __init__(self, *args, **kwargs):
            self.calls = 0

        def solve_all(self, detailed=False):
            if sleep:
                import time
                time.sleep(sleep)
            if boom:
                raise RuntimeError(boom)
            return np.array(exact_states() if states is None else states)

        def config(self):
            out = {"load_seconds": 0.02}
            if settled is not None:
                out["settled"] = {"per_point": [settled] * len(SCAN), "max": settled,
                                  "tol": steady.VERDICT["settled"]}
            return out

    return FakeRef


def _refs(**overrides):
    refs = {"run_network": _fake_reference(input_="net", settled=1e-6),
            "roadrunner": _fake_reference(), "copasi": _fake_reference(),
            "amici": _fake_reference()}
    refs.update(overrides)
    return refs


def _install(monkeypatch, refs, bngsim=None, pts=SCAN):
    monkeypatch.setattr(steady, "_prepare", lambda problem, ctx: _prep(pts))
    monkeypatch.setattr(steady, "BngsimSteady", bngsim or _fake_bngsim())
    monkeypatch.setattr(steady, "REFERENCE_ENGINES", refs)


def _ctx(tmp_path, **kw):
    return {"workdir": str(tmp_path), "subprocess_timeout": 60, "engine_budget_seconds": 60, **kw}


def test_correctness_passes_when_every_engine_reaches_the_steady_state(monkeypatch, tmp_path):
    _install(monkeypatch, _refs())
    rec = steady.run_correctness(None, None, _ctx(tmp_path))
    assert rec["outcome"] == "PASS", rec
    assert set(rec["verdicts"]) == {"run_network", "roadrunner", "copasi", "amici", "engine_free"}
    for name, v in rec["verdicts"].items():
        assert v["passed"], name
    m = rec["measurements"]
    assert m["bngsim_integration"]["vs_bngsim"]["passed"]
    np.testing.assert_allclose(m["states"]["bngsim"]["values"], exact_states())
    assert set(m["reference_checks"]) == {"run_network", "roadrunner", "copasi", "amici"}
    json.dumps(rec)  # every number is a plain number


def test_bngsim_on_another_root_at_one_point_is_a_diff_at_that_point(monkeypatch, tmp_path):
    wrong = exact_states()
    wrong[2] = binding_state(2.0, root=+1)  # a root of f, on the laws, not the one reached
    _install(monkeypatch, _refs(), bngsim=_fake_bngsim(states=wrong))
    rec = steady.run_correctness(None, None, _ctx(tmp_path))
    assert rec["outcome"] == "DIFF"
    v = rec["verdicts"]["run_network"]
    assert not v["passed"] and v["points_agreed"] == 2 and v["worst_point"] == 2
    assert rec["verdicts"]["engine_free"]["passed"]  # it is a root, just the wrong one
    assert set(v["stability"]) == {"point", "bngsim", "run_network"}


def test_a_reference_that_did_not_take_the_dose_is_off_the_laws_and_not_scored(
        monkeypatch, tmp_path):
    pts = [{"A0": 4.0}, {"A0": 2.0}]
    right = np.array([binding_state(1.0, a0=4.0), binding_state(1.0, a0=2.0)])
    stale = np.array([binding_state(1.0, a0=2.0)] * 2)
    refs = {name: _fake_reference(states=right, input_=cls.INPUT,
                                  settled=1e-6 if name == "run_network" else None)
            for name, cls in _refs().items()}
    refs["copasi"] = _fake_reference(states=stale)
    _install(monkeypatch, refs, bngsim=_fake_bngsim(states=right, integration=right), pts=pts)
    rec = steady.run_correctness(None, None, _ctx(tmp_path))
    assert rec["outcome"] == "REFERENCE_FAILED"
    assert rec["copasi_error"].startswith("OFF_MANIFOLD: at point 0 (A0=4)")
    assert "copasi" not in rec["verdicts"] and rec["verdicts"]["amici"]["passed"]


def test_a_reference_that_is_not_at_rest_or_did_not_settle_is_not_scored(monkeypatch, tmp_path):
    restless = exact_states()
    restless[1] = [1.9, 0.9, 0.1]
    _install(monkeypatch, _refs(run_network=_fake_reference(input_="net", settled=0.5),
                                amici=_fake_reference(states=restless)))
    rec = steady.run_correctness(None, None, _ctx(tmp_path))
    assert rec["outcome"] == "REFERENCE_FAILED"
    assert rec["run_network_error"].startswith("NOT_SETTLED: at point 0")
    assert rec["amici_error"].startswith("NOT_A_STEADY_STATE: at point 1")
    assert set(rec["verdicts"]) == {"roadrunner", "copasi", "engine_free"}


def test_a_solver_reference_on_a_root_the_integration_does_not_reach_is_not_scored(
        monkeypatch, tmp_path):
    other = exact_states()
    other[0] = binding_state(0.5, root=+1)
    _install(monkeypatch, _refs(roadrunner=_fake_reference(states=other)))
    rec = steady.run_correctness(None, None, _ctx(tmp_path))
    assert rec["outcome"] == "REFERENCE_FAILED"
    assert rec["roadrunner_error"].startswith("OTHER_STATE: at point 0 (kr=0.5)")
    assert "2/3 points agree" in rec["roadrunner_error"]
    checks = rec["measurements"]["reference_checks"]["roadrunner"]
    assert checks["root"]["point"] == 0 and not checks["vs_integration"]["passed"]
    assert "roadrunner" not in rec["verdicts"] and rec["verdicts"]["run_network"]["passed"]


def test_the_toggles_saddle_is_named_as_such(monkeypatch, tmp_path):
    names = ["R1()", "R2()"]
    u = [155.76340773977222, 0.09951301917279096]
    saddle = [1.3165460714061394, 6.7341634999431825]

    def prep(problem, ctx):
        return {"net": Path("m.net"), "sbml": Path("m.xml"),
                "proto": dict(PROTOCOL, n_steps=20, parameters={}),
                "info": {"points": [{}]}, "species_names": names,
                "name_to_id": {"R1()": "R1", "R2()": "R2"}, "points": [{}],
                "point_nets": [Path("m.net")], **steady.network_checks(TOGGLE, [{}], names)}

    monkeypatch.setattr(steady, "_prepare", prep)
    monkeypatch.setattr(steady, "BngsimSteady", _fake_bngsim(states=[u], integration=[u]))
    monkeypatch.setattr(steady, "REFERENCE_ENGINES", {
        "run_network": _fake_reference(states=[u], input_="net", settled=1e-6),
        "roadrunner": _fake_reference(states=[saddle])})
    rec = steady.run_correctness(None, None, _ctx(tmp_path))
    assert rec["outcome"] == "REFERENCE_FAILED"
    assert "an unstable steady state (a saddle" in rec["roadrunner_error"]
    assert rec["measurements"]["reference_checks"]["roadrunner"]["root"]["stability"] == "unstable"


def test_a_crashing_or_timed_out_reference_is_recorded_and_the_rest_still_judge(
        monkeypatch, tmp_path):
    _install(monkeypatch, _refs(copasi=_fake_reference(boom="edgepop"),
                                amici=_fake_reference(sleep=300)))
    rec = steady.run_correctness(None, None, _ctx(tmp_path, engine_budget_seconds=1.0))
    assert rec["outcome"] == "REFERENCE_FAILED"
    assert "edgepop" in rec["copasi_error"]
    assert rec["amici_error"].startswith("TIMEOUT")
    assert "steady state at 3 point" in rec["amici_error"]
    assert rec["verdicts"]["run_network"]["passed"]


def test_a_diff_is_a_diff_even_when_a_reference_also_failed(monkeypatch, tmp_path):
    wrong = exact_states() * 1.01
    _install(monkeypatch, _refs(copasi=_fake_reference(boom="x")),
             bngsim=_fake_bngsim(states=wrong))
    assert steady.run_correctness(None, None, _ctx(tmp_path))["outcome"] == "DIFF"


def test_bngsim_off_the_laws_fails_the_engine_free_verdict(monkeypatch, tmp_path):
    # Every reference agrees with it (they share its mistake); the harness's own
    # check of the conserved totals still catches it.
    stale = np.array([binding_state(1.0, a0=2.0)] * 2)
    pts = [{"A0": 4.0}, {"A0": 2.0}]
    refs = {name: _fake_reference(states=stale, input_=cls.INPUT) for name, cls in _refs().items()}
    _install(monkeypatch, refs, bngsim=_fake_bngsim(states=stale, integration=stale), pts=pts)
    rec = steady.run_correctness(None, None, _ctx(tmp_path))
    assert rec["outcome"] == "DIFF"
    assert not rec["verdicts"]["engine_free"]["passed"]


def test_bngsim_failing_is_an_exception_a_timeout_or_with_every_reference_a_bad_test(
        monkeypatch, tmp_path):
    _install(monkeypatch, _refs(), bngsim=_fake_bngsim(boom="no root"))
    rec = steady.run_correctness(None, None, _ctx(tmp_path))
    assert rec["outcome"] == "EXCEPTION" and "no root" in rec["bngsim_error"]
    _install(monkeypatch, _refs(), bngsim=_fake_bngsim(sleep=300))
    rec = steady.run_correctness(None, None, _ctx(tmp_path, engine_budget_seconds=1.0))
    assert rec["outcome"] == "TIMEOUT"
    _install(monkeypatch, {n: _fake_reference(boom="x") for n in _refs()},
             bngsim=_fake_bngsim(boom="no"))
    assert steady.run_correctness(None, None, _ctx(tmp_path))["outcome"] == "BAD_TEST"


def test_preparation_failure_is_a_bad_test(monkeypatch, tmp_path):
    def boom(problem, ctx):
        raise ValueError("the network has no parameter 'nope' to set")

    monkeypatch.setattr(steady, "_prepare", boom)
    for run in (steady.run_correctness, steady.run_timing):
        rec = run(None, None, _ctx(tmp_path))
        assert rec["outcome"] == "BAD_TEST" and "no parameter" in rec["error"]


def _timing_job(requires=None):
    return types.SimpleNamespace(
        definition={"timing": {"warmup_rounds": 1, "min_rounds": 3, "max_rounds": 3,
                               "budget_seconds": 0.0}},
        requires=requires)


def test_timing_interleaves_every_engine_and_reports_per_point_times(monkeypatch, tmp_path):
    _install(monkeypatch, _refs())
    rec = steady.run_timing(_timing_job(), None, _ctx(tmp_path))
    m = rec["measurements"]
    assert rec["outcome"] == "PASS", rec
    assert set(m["times"]) == {"bngsim", "bngsim_integration", "run_network", "roadrunner",
                               "copasi", "amici"}
    assert m["n_points"] == 3
    for name, t in m["times"].items():
        assert m["times_per_point"][name]["median_seconds"] == pytest.approx(
            t["median_seconds"] / 3)
    assert all(r["passed"] for r in m["reproducibility"].values())
    assert all(a["passed"] for a in m["answers"].values())
    assert m["engine_roles"]["run_network"] == "fresh subprocess"
    json.dumps(rec)


def test_timing_flags_an_engine_whose_rerun_is_not_its_first_run(monkeypatch, tmp_path):
    class Drifting(_fake_reference()):
        n = 0

        def solve_all(self, detailed=False):
            Drifting.n += 1
            return exact_states() * (1 + 1e-3 * Drifting.n)

    _install(monkeypatch, _refs(copasi=Drifting))
    rec = steady.run_timing(_timing_job(), None, _ctx(tmp_path))
    assert rec["outcome"] == "NOT_REPRODUCIBLE"
    assert not rec["measurements"]["reproducibility"]["copasi"]["passed"]
    assert not rec["measurements"]["answers"]["copasi"]["passed"]


def test_timing_skips_a_reference_correctness_rejected_and_drops_one_that_crashes(
        monkeypatch, tmp_path):
    from bench.results import result_path

    corr = "steady_state/m/x/newton/correctness"
    path = result_path(tmp_path, corr)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"outcome": "REFERENCE_FAILED",
                                "roadrunner_error": "OTHER_STATE: at point 0 ..."}))
    _install(monkeypatch, _refs(roadrunner=_fake_reference(sleep=300),
                                amici=_fake_reference(boom="compile")))
    rec = steady.run_timing(_timing_job(requires=corr), None,
                            _ctx(tmp_path, results_dir=str(tmp_path)))
    assert set(rec["measurements"]["times"]) == {"bngsim", "bngsim_integration", "run_network",
                                                 "copasi"}
    assert rec["roadrunner_error"].startswith("failed in the correctness job, so not timed")
    assert "compile" in rec["amici_error"]
    assert rec["outcome"] == "PASS"


def test_states_of_the_wrong_shape_fail_every_check_without_raising(monkeypatch, tmp_path):
    prep = binding_prep()
    assert not steady.engine_free(np.ones((3, 2)), prep)["passed"]
    v = steady.agreement(np.ones((3, 3)), np.ones((3, 2)), 1e-8, NAMES)
    assert not v["passed"] and "shapes differ" in v["reason"]
    _install(monkeypatch, _refs(amici=_fake_reference(states=np.ones((3, 2)))))
    rec = steady.run_correctness(None, None, _ctx(tmp_path))
    assert rec["outcome"] == "REFERENCE_FAILED"
    assert rec["amici_error"].startswith("INVALID_STATES: states of shape [3, 2]")
