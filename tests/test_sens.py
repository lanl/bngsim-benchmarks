"""The sens capability's own logic, with fake engines (no simulator): job generation
and planning, protocol validation, choosing the parameters, the finite-difference
reference, and how per-engine results become verdicts and an outcome in the
correctness and timing jobs -- including the attribution of a disagreement to the
reference that two other engines refute."""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest

from bench import netfile, specs
from bench.capabilities import ode, sens

PROTOCOL = {"netgen": "generate_network({overwrite=>1})", "t_end": 10, "n_steps": 20,
            "rtol": 1e-8, "atol": 1e-8, "parameters": "rate_constants"}


@pytest.fixture
def sens_spec(spec_data):
    spec_data["capabilities"] = ["sens"]
    spec_data["protocol"] = {"sens": dict(PROTOCOL)}
    return spec_data


# --------------------------------------------------------------------------- #
# Specs, jobs and planning
# --------------------------------------------------------------------------- #
def test_engines_id_names_bngsim_against_every_reference():
    assert sens.engines_id() == "bngsim-vs-amici-run_network_fd-copasi"


def test_registry_emits_one_correctness_and_one_timing_job(sens_spec, write_spec):
    problem = specs.load(write_spec(sens_spec))
    correctness, timing = sens.jobs(problem)
    assert (correctness.kind, timing.kind) == ("correctness", "timing")
    assert timing.requires == correctness.id
    assert correctness.id == f"sens/toy_model/{sens.engines_id()}/auto/correctness"
    proto = correctness.definition["protocol"]
    # Defaults are part of the definition, so changing one restales the results.
    assert proto["parameters"] == "rate_constants"
    assert sens.settings(problem) == ["auto"]
    assert (proto["fd_step"], proto["fd_rtol"], proto["fd_atol"]) == (1e-3, 1e-10, 1e-12)
    assert correctness.definition["verdict"] == sens.VERDICT and "verdict" not in timing.definition
    assert timing.definition["timing"] == sens.TIMING_PROTOCOL
    assert correctness.definition["setting"]["configurations"] == sens.CONFIGURATIONS
    budget = sens.engine_budget_seconds(problem)
    assert correctness.timeout_seconds >= (len(sens.REFERENCE_ENGINES)
                                           + len(sens.CONFIGURATIONS)) * budget


def test_reference_versions_key_only_this_capability(sens_spec, write_spec, monkeypatch):
    monkeypatch.setattr(sens.env, "amici_identity", lambda: {"version": "1.1.0"})
    monkeypatch.setattr(sens.env, "copasi_identity", lambda: {"version": "4.48.311"})
    correctness, _ = sens.jobs(specs.load(write_spec(sens_spec)))
    assert correctness.definition["reference_versions"] == {"amici": "1.1.0", "copasi": "4.48.311"}


def test_non_bngl_source_gets_no_jobs(sens_spec, write_spec):
    sens_spec["source"]["format"] = "sbml"
    assert sens.jobs(specs.load(write_spec(sens_spec))) == []


def test_a_spec_can_carry_ode_and_sens_and_the_ode_jobs_do_not_change(spec_data, write_spec):
    ode_only = specs.load(write_spec(spec_data))
    spec_data["capabilities"] = ["ode", "sens"]
    spec_data["protocol"]["sens"] = dict(PROTOCOL)
    both = specs.load(write_spec(spec_data))
    assert [j.canonical() for j in ode.jobs(ode_only)] == [j.canonical() for j in ode.jobs(both)]
    assert len(sens.jobs(both)) == 2


@pytest.mark.parametrize("change, message", [
    ({"t_end": 0}, "t_end: must be a positive number"),
    ({"t_start": 5}, "unknown key 't_start'"),
    ({"n_steps": 0}, "n_steps"),
    ({"rtol": -1}, "rtol: must be a positive number"),
    ({"parameters": "all"}, "parameters: one of independent, rate_constants, fitted"),
    ({"parameters": "fitted"}, "fitted: required with parameters: fitted"),
    ({"fitted": []}, "fitted: a non-empty list"),
    ({"parameter_scales": {"Stim": 0}}, "parameter_scales.Stim: must be a positive number"),
    ({"parameters": []}, "non-empty list"),
    ({"parameters": ["k1", "k1"]}, "listed twice"),
    ({"parameters": ["k-1"]}, "not parameter names"),
    ({"fd_step": 0.5}, "fd_step: a relative step"),
    ({"fd_rtol": 1e-6}, "fd_rtol: must not be looser than rtol"),
    ({"fd_atol": 1e-6}, "fd_atol: must not be looser than atol"),
    ({"stimulus": {"Stim": 1}}, "preequilibration and stimulus come together"),
    ({"preequilibration": {"duration": 100}}, "preequilibration and stimulus come together"),
    ({"preequilibration": {"duration": 0}, "stimulus": {"Stim": 1}},
     "preequilibration.duration: a positive time"),
    ({"preequilibration": {"duration": 9, "t_end": 9}, "stimulus": {"Stim": 1}},
     "preequilibration: unknown key 't_end'"),
    ({"preequilibration": {"duration": 9, "set": {"Stim": "off"}}, "stimulus": {"Stim": 1}},
     "preequilibration.set.Stim: must be a finite number"),
    ({"preequilibration": {"duration": 9}, "stimulus": {}}, "stimulus: a non-empty mapping"),
    ({"netgen": "simulate({})"}, "netgen"),
])
def test_invalid_sens_protocol_is_refused_with_its_name(sens_spec, write_spec, change, message):
    sens_spec["protocol"]["sens"].update(change)
    with pytest.raises(specs.SpecError, match=message.replace("(", r"\(")):
        specs.load(write_spec(sens_spec))


def test_the_parameters_must_be_named_a_convention_per_problem(sens_spec, write_spec):
    del sens_spec["protocol"]["sens"]["parameters"]
    with pytest.raises(specs.SpecError, match="missing 'parameters'"):
        specs.load(write_spec(sens_spec))


def test_a_fitted_set_beside_another_convention_is_a_setting_of_its_own(sens_spec, write_spec):
    sens_spec["protocol"]["sens"].update(parameters="independent", fitted=["k1", "k2"])
    problem = specs.load(write_spec(sens_spec))
    assert sens.settings(problem) == ["auto", "fitted"]
    jobs = sens.jobs(problem)
    assert [(j.setting, j.kind) for j in jobs] == [
        ("auto", "correctness"), ("auto", "timing"),
        ("fitted", "correctness"), ("fitted", "timing")]
    auto, fitted = jobs[0], jobs[2]
    assert auto.definition["protocol"]["parameters"] == "independent"
    assert fitted.definition["protocol"]["parameters"] == "fitted"
    assert fitted.id.endswith("/fitted/correctness") and jobs[3].requires == fitted.id
    assert sens.parameter_count(problem, "fitted") == 2
    assert sens.convention(fitted.definition["protocol"]) == "fitted"
    assert sens.convention(auto.definition["protocol"]) == "independent"
    # A problem whose convention is its fitted set has that one setting only.
    sens_spec["protocol"]["sens"]["parameters"] = "fitted"
    assert sens.settings(specs.load(write_spec(sens_spec))) == ["auto"]


def test_an_explicit_parameter_list_is_accepted_and_sizes_the_plan(sens_spec, write_spec):
    sens_spec["protocol"]["sens"]["parameters"] = ["k1", "k2", "k3"]
    problem = specs.load(write_spec(sens_spec))
    assert problem.protocol_for("sens")["parameters"] == ["k1", "k2", "k3"]
    assert sens.parameter_count(problem) == 3


def test_planning_grows_with_the_network_and_the_parameters(sens_spec, write_spec):
    small = specs.load(write_spec(sens_spec))
    sens_spec["size"] = {"species": 1000, "reactions": 15000, "sens_parameters": 40}
    big = specs.load(write_spec(sens_spec))
    (c_small, t_small), (c_big, t_big) = sens.planning_seconds(small), sens.planning_seconds(big)
    assert c_big > 10 * c_small and t_big > t_small > 0
    # AMICI's compile is what grows: on a 15000-reaction network, past the budget's cap.
    assert sens.engine_seconds(big)["amici"] > sens.ENGINE_BUDGET_CAP_SECONDS
    assert sens.engine_budget_seconds(small) == sens.MIN_TIMEOUT_SECONDS
    assert sens.engine_budget_seconds(big) == sens.ENGINE_BUDGET_CAP_SECONDS


def test_the_pilot_specs_load_and_declare_what_they_test():
    from conftest import REPO

    problems = {p.id: p for p in specs.load_all(REPO / "problems")}
    igf1r = problems["igf1r_ligand_binding_erickson2019"]
    atf2 = problems["atf2_phosphoswitch_kirsch2020"]
    assert igf1r.layer == atf2.layer == "pilot" and not igf1r.suites and not atf2.suites
    assert igf1r.capabilities == atf2.capabilities == ("sens",)
    # Fitted sets below 10: every independent parameter, the fitted set as a setting.
    for p in (igf1r, atf2):
        assert p.protocol_for("sens")["parameters"] == "independent"
        assert sens.settings(p) == ["auto", "fitted"] and len(sens.jobs(p)) == 4
    assert igf1r.protocol_for("sens")["parameter_scales"] == {"IGF1_cold_conc": 7e-12}
    # atf2 pre-equilibrates at basal activity, then stimulates (D3b).
    assert atf2.protocol_for("sens")["preequilibration"] == {"duration": 20000,
                                                             "set": {"Stim": 0}}
    assert atf2.protocol_for("sens")["stimulus"] == {"Stim": 1}


# --------------------------------------------------------------------------- #
# Choosing the parameters
# --------------------------------------------------------------------------- #
NET = """begin parameters
    1 NA        6.022e23  # Constant
    2 V         1e-12  # Constant
    3 kon_M     1e6  # Constant
    4 kon       kon_M/(NA*V)  # ConstantExpression
    5 koff      0.1  # Constant
    6 L_conc    1e-9  # Constant
    7 L0        L_conc*NA*V  # ConstantExpression
    8 Stim      0  # Constant
    9 kstim     2  # Constant
   10 vol       2.5  # Constant
   11 kvol      vol*koff  # ConstantExpression
end parameters
begin compartments
    1 C  3  vol
end compartments
begin species
    1 L() L0
    2 R() 1000
    3 L.R() 0
end species
begin reactions
    1 1,2 3 kon #bind
    2 3 1,2 koff #unbind
    3 3 2 f1 #f
    4 3 0 kvol #v
end reactions
begin functions
    1 f1() Stim*kstim
end functions
"""


ZERO = "zero value: no relative step (give it a parameter_scales entry)"


def test_rate_constants_are_the_literals_some_rate_reads_less_zeros_and_sizes():
    params, excluded = sens.select_parameters(netfile.structure(NET), "rate_constants")
    assert params == ["NA", "V", "kon_M", "koff", "kstim"]
    assert excluded == {"Stim": ZERO, "vol": "sizes a compartment"}


def test_independent_adds_the_parameters_that_reach_only_an_initial_amount():
    params, _ = sens.select_parameters(netfile.structure(NET), "independent")
    assert params == ["NA", "V", "kon_M", "koff", "L_conc", "kstim"]


def test_the_fitted_convention_differentiates_the_fitted_set():
    st = netfile.structure(NET)
    params, _ = sens.select_parameters(st, "fitted", fitted=["kstim", "koff"])
    assert params == ["koff", "kstim"]
    with pytest.raises(ValueError, match="no fitted set"):
        sens.select_parameters(st, "fitted")


def test_a_zero_valued_parameter_with_a_scale_is_differentiated_on_that_scale():
    st = netfile.structure(NET)
    params, excluded = sens.select_parameters(st, "rate_constants", scales={"Stim": 1.0})
    assert "Stim" in params and "Stim" not in excluded
    np.testing.assert_array_equal(sens.parameter_scales(st, ["koff", "Stim"], {"Stim": 1.0}),
                                  [0.1, 1.0])
    with pytest.raises(ValueError, match="parameter_scales: derived"):
        sens.select_parameters(st, "rate_constants", scales={"kon": 1.0})


def test_an_explicit_list_is_kept_in_network_order_and_checked():
    st = netfile.structure(NET)
    params, excluded = sens.select_parameters(st, ["koff", "L_conc", "Stim"])
    assert params == ["koff", "L_conc"] and "Stim" in excluded
    with pytest.raises(ValueError, match="not parameters of the network: nope"):
        sens.select_parameters(st, ["koff", "nope"])
    with pytest.raises(ValueError, match="derived .*: kon"):
        sens.select_parameters(st, ["kon"])


# --------------------------------------------------------------------------- #
# The finite-difference reference
# --------------------------------------------------------------------------- #
def test_run_network_command_carries_the_fd_tolerances(monkeypatch):
    monkeypatch.setattr(sens.env, "run_network_bin", lambda: Path("/bng/bin/run_network"))
    monkeypatch.setattr(sens.env, "run_network_launch", lambda: [])
    cmd = sens.run_network_command(Path("m.net"), {"t_end": 600, "n_steps": 300}, Path("o/fd"),
                                   rtol=1e-10, atol=1e-12)
    assert cmd == ["/bng/bin/run_network", "-o", "o/fd", "-p", "cvode", "-a", "1e-12", "-r",
                   "1e-10", "--cdat", "1", "--fdat", "0", "-g", "m.net", "m.net", "2.0", "300"]


DECAY_NET = """begin parameters
    1 k   0.3  # Constant
    2 A0  100  # Constant
end parameters
begin species
    1 A() A0
    2 B() 0
end species
begin reactions
    1 1 2 k #decay
end reactions
"""


def test_fd_reference_differentiates_the_network_it_is_given(monkeypatch, tmp_path):
    """run_network stood in for by the closed-form solution of the network text it is
    handed: the quotients must be the derivative, and their two-step estimate small."""
    proto = {**PROTOCOL, "fd_step": 1e-3, "fd_rtol": 1e-10, "fd_atol": 1e-12}
    t = sens.times(proto)
    seen = []

    def solve(self, text):
        st = netfile.structure(text)
        k, a0 = st.literal("k"), st.literal("A0")
        seen.append((k, a0))
        e = np.exp(-k * t)
        return np.stack([a0 * e, a0 * (1 - e)], axis=1)

    monkeypatch.setattr(sens.FdRunNetwork, "_solve", solve)
    net = tmp_path / "m.net"
    net.write_text(DECAY_NET)
    fd = sens.FdRunNetwork(net, proto, ["k", "A0"], np.array([0.3, 100.0]),
                           np.array([0.3, 100.0]), tmp_path / "fd", timeout=10)
    (fine, coarse), X = sens.unpack(fd.run(), 2)
    assert len(seen) == 4 * 2 + 1 and seen[0] == (0.3, 100.0)
    assert (0.3 * (1 + 5e-4), 100.0) in seen and (0.3, 100.0 * (1 - 1e-3)) in seen
    e = np.exp(-0.3 * t)
    np.testing.assert_allclose(fine[:, 0, 0], -t * 100 * e, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(fine[:, 1, 1], 1 - e, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(X[:, 0], 100 * e)
    assert np.max(np.abs(fine - coarse)) < 1e-4


# --------------------------------------------------------------------------- #
# Correctness and timing, with fake engines
# --------------------------------------------------------------------------- #
# The fake network: A -> B at rate k, A(0) = A0, observable total = A + B; the
# parameters are k and A0. Every fake engine returns the exact sensitivities, unless
# told to be off.
K, A0 = 0.3, 100.0
SPECIES = ["A()", "B()"]
PARAMS = ["k", "A0"]


def _exact(t, shift=None):
    e = np.exp(-K * t)
    X = np.stack([A0 * e, A0 * (1 - e)], axis=1)
    S = np.zeros((t.size, 2, 2))
    S[:, 0, 0], S[:, 1, 0] = -t * A0 * e, t * A0 * e
    S[:, 0, 1], S[:, 1, 1] = e, 1 - e
    if shift:  # (species, parameter, relative error)
        i, k, rel = shift
        S[:, i, k] *= 1 + rel
    return S, X


# What the jobs read of a job in these tests: its setting (and, for timing, below).
JOB = types.SimpleNamespace(setting="auto", requires=None)


def _prep():
    proto = sens.protocol(types.SimpleNamespace(protocol_for=lambda c: dict(PROTOCOL)))
    return {"net": Path("m.net"), "sbml": Path("m.xml"), "proto": proto, "info": {"network": {}},
            "species_names": SPECIES, "name_to_id": {"A()": "A", "B()": "B"},
            "params": PARAMS, "values": np.array([K, A0]), "scales": np.array([K, A0]),
            "observables": ["total"], "weights": np.array([[1.0, 1.0]])}


class _FakeResult(tuple):
    """What a fake engine's run returns: ``(S, X)``, with what a bngsim Result also
    has -- the observables (here one, the total) and the likelihood gradient."""

    @property
    def observables(self):
        return self[1].sum(axis=1, keepdims=True)

    def neg_log_likelihood_gradient(self, data, sigma, *, target):
        assert target == "observables"
        return 0.0, _nllh_gradient(*self, data, sigma)


def _nllh_gradient(S, X, data, sigma):
    """The gradient of a Gaussian negative log-likelihood of the total, A + B."""
    y, Sy = X.sum(axis=1, keepdims=True), S.sum(axis=1, keepdims=True)
    return np.einsum("to,top->p", (y - data) / sigma**2, Sy)


def _fake_bngsim(shift=None, boom=None, simultaneous_shift=None, own_obs_off=False):
    class FakeBngsim:
        def __init__(self, net, proto, params, *, sensitivity_method="staggered",
                     two_phase=None):
            if boom and (boom == "all" or sensitivity_method == boom):
                raise RuntimeError(f"bngsim {sensitivity_method} broke")
            self.method, self.t = sensitivity_method, sens.times(proto)
            self.load_seconds = self.codegen_seconds = 0.01

        def run(self):
            s = simultaneous_shift if self.method == "simultaneous" else shift
            return _FakeResult(_exact(self.t, s))

        def tensors(self, raw):
            return raw

        def observable_sensitivities(self, raw):
            S = raw[0].sum(axis=1, keepdims=True)
            return ["total"], S * (1.01 if own_obs_off else 1.0)

        def config(self):
            return {"sensitivity_method": self.method, "codegen_backend": "cc",
                    "load_seconds": 0.01, "codegen_seconds": 0.1}

    return FakeBngsim


def _fake_reference(shift=None, boom=None, sleep=0.0, covered=(True, True), name="ref"):
    class FakeRef:
        INPUT = "sbml"
        NAME = name

        def __init__(self, *args, **kwargs):
            self.t = sens.times(_prep()["proto"])
            self.covered = np.array(covered)

        def run(self):
            if sleep:
                import time
                time.sleep(sleep)
            if boom:
                raise RuntimeError(boom)
            return _exact(self.t, shift)

        def tensors(self, raw):
            S, X = raw
            S = S.copy()
            S[:, :, ~self.covered] = np.nan
            return S, X

        def adjoint_problem(self, data, sigma):
            return data, sigma

        def adjoint_gradient(self, problem):
            return _nllh_gradient(*_exact(self.t, shift), *problem)

        def config(self):
            return {"load_seconds": 0.02, "codegen_seconds": 1.0}

    return FakeRef


def _fake_fd(shift=None, noise=0.0):
    """The FD reference packs D(h/2), D(h) and the state; ``noise`` puts the two
    quotients apart in k's columns (an unresolved finite difference)."""
    class FakeFD:
        INPUT = "net"
        NAME = sens.FD

        def __init__(self, net, proto, params, values, scales, workdir, *, timeout,
                     two_phase=None):
            self.t = sens.times(proto)
            self.covered = np.ones(2, bool)

        def run(self):
            S, X = _exact(self.t, shift)
            coarse = S.copy()
            coarse[:, :, 0] *= 1 + noise
            return sens.pack(S, X, coarse)

        def config(self):
            return {"method": "fake central differences"}

    return FakeFD


def _install(monkeypatch, refs, bngsim=None):
    monkeypatch.setattr(sens, "_prepare", lambda problem, ctx, setting: _prep())
    monkeypatch.setattr(sens, "BngsimSens", bngsim or _fake_bngsim())
    monkeypatch.setattr(sens, "REFERENCE_ENGINES", refs)


def _refs(**over):
    refs = {"amici": _fake_reference(name="amici"), sens.FD: _fake_fd(),
            "copasi": _fake_reference(name="copasi")}
    refs.update(over)
    return refs


def _ctx(tmp_path, **kw):
    return {"workdir": str(tmp_path), "subprocess_timeout": 60, "engine_budget_seconds": 60, **kw}


def test_correctness_passes_when_every_engine_agrees(monkeypatch, tmp_path):
    _install(monkeypatch, _refs())
    rec = sens.run_correctness(JOB, None, _ctx(tmp_path))
    assert rec["outcome"] == "PASS", rec
    assert set(rec["verdicts"]) == {"amici", sens.FD, "copasi"}
    for v in rec["verdicts"].values():
        assert v["passed"] and v["max_error"] == 0.0 and v["n_judged"] == v["n_columns"] == 6
    m = rec["measurements"]
    assert m["configurations"]["bngsim_simultaneous"]["passed"]
    assert m["bngsim"]["observables_check"]["passed"]
    assert m["coverage"]["unjudged"] == [] and m["coverage"]["n_columns"] == 6
    # The total does not move with k: a column with nothing to judge.
    assert m["coverage"]["n_significant"] == m["coverage"]["n_judged"] == 5


def test_bngsim_off_in_one_column_is_a_diff_naming_it(monkeypatch, tmp_path):
    _install(monkeypatch, _refs(), bngsim=_fake_bngsim(shift=(1, 0, 0.05)))
    rec = sens.run_correctness(JOB, None, _ctx(tmp_path))
    assert rec["outcome"] == "DIFF"
    for v in rec["verdicts"].values():
        assert not v["passed"] and "refuted_by" not in v
        # B's k column, and the total's, which BNGsim's error moves off zero.
        assert {tuple(f[:2]) for f in v["failed"]} == {("B()", "k"), ("observable:total", "k")}
        assert v["params_failed"] == ["k"]


def test_a_reference_two_others_contradict_is_refuted_not_a_diff(monkeypatch, tmp_path):
    # COPASI drops a chain-rule term: its k column is zero where the others agree.
    _install(monkeypatch, _refs(copasi=_fake_reference(shift=(0, 0, -1.0), name="copasi")))
    rec = sens.run_correctness(JOB, None, _ctx(tmp_path))
    assert rec["outcome"] == "REFERENCE_FAILED"
    v = rec["verdicts"]["copasi"]
    assert not v["passed"] and v["refuted_by"] == ["amici", sens.FD]
    assert rec["verdicts"]["amici"]["passed"] and rec["verdicts"][sens.FD]["passed"]


def test_one_reference_against_bngsim_alone_is_not_enough_to_refute(monkeypatch, tmp_path):
    # With the other two references failed, a disagreement stands as a DIFF.
    _install(monkeypatch, _refs(amici=_fake_reference(boom="compile", name="amici"),
                                **{sens.FD: _fake_fd(shift=(0, 0, 0.1))},
                                copasi=_fake_reference(boom="x", name="copasi")))
    rec = sens.run_correctness(JOB, None, _ctx(tmp_path))
    assert rec["outcome"] == "DIFF" and "refuted_by" not in rec["verdicts"][sens.FD]


def test_a_crashing_or_timed_out_reference_is_recorded_and_the_rest_still_judge(
        monkeypatch, tmp_path):
    _install(monkeypatch, _refs(amici=_fake_reference(sleep=300, name="amici"),
                                copasi=_fake_reference(boom="KeyError: 'I'", name="copasi")))
    rec = sens.run_correctness(JOB, None, _ctx(tmp_path, engine_budget_seconds=1.0))
    assert rec["outcome"] == "REFERENCE_FAILED"
    assert rec["amici_error"].startswith("TIMEOUT") and "first solve" in rec["amici_error"]
    assert "KeyError" in rec["copasi_error"]
    assert rec["verdicts"][sens.FD]["passed"]


def test_bngsim_failing_is_an_exception_and_with_every_reference_a_bad_test(
        monkeypatch, tmp_path):
    _install(monkeypatch, _refs(), bngsim=_fake_bngsim(boom="all"))
    assert sens.run_correctness(JOB, None, _ctx(tmp_path))["outcome"] == "EXCEPTION"
    broken = {name: _fake_reference(boom="no", name=name) for name in ("amici", "copasi")}
    _install(monkeypatch, broken, bngsim=_fake_bngsim(boom="all"))
    assert sens.run_correctness(JOB, None, _ctx(tmp_path))["outcome"] == "BAD_TEST"


def test_a_further_configuration_failing_is_bngsim_failing(monkeypatch, tmp_path):
    _install(monkeypatch, _refs(), bngsim=_fake_bngsim(boom="simultaneous"))
    rec = sens.run_correctness(JOB, None, _ctx(tmp_path))
    assert rec["outcome"] == "EXCEPTION"
    assert rec["bngsim_error"].startswith("bngsim_simultaneous: RuntimeError")
    assert rec["verdicts"]["amici"]["passed"]  # the default was still judged


def test_configurations_that_disagree_are_a_diff(monkeypatch, tmp_path):
    _install(monkeypatch, _refs(), bngsim=_fake_bngsim(simultaneous_shift=(0, 1, 0.01)))
    rec = sens.run_correctness(JOB, None, _ctx(tmp_path))
    assert rec["outcome"] == "DIFF"
    assert not rec["measurements"]["configurations"]["bngsim_simultaneous"]["passed"]
    assert all(v["passed"] for v in rec["verdicts"].values())


def test_bngsim_observable_sensitivities_must_match_its_species_ones(monkeypatch, tmp_path):
    _install(monkeypatch, _refs(), bngsim=_fake_bngsim(own_obs_off=True))
    rec = sens.run_correctness(JOB, None, _ctx(tmp_path))
    assert rec["outcome"] == "DIFF"
    assert not rec["measurements"]["bngsim"]["observables_check"]["passed"]


def test_a_reference_judges_only_the_parameters_it_differentiates(monkeypatch, tmp_path):
    # AMICI cannot differentiate A0 and is wrong in it -- that column is not its to judge.
    _install(monkeypatch, _refs(amici=_fake_reference(covered=(True, False), name="amici",
                                                      shift=(0, 1, 0.5))))
    rec = sens.run_correctness(JOB, None, _ctx(tmp_path))
    assert rec["outcome"] == "PASS"
    assert rec["verdicts"]["amici"]["params_judged"] == ["k"]


def test_a_parameter_no_agreeing_reference_judged_is_a_reference_failure(monkeypatch, tmp_path):
    refs = {"amici": _fake_reference(covered=(True, False), name="amici")}
    _install(monkeypatch, refs)
    rec = sens.run_correctness(JOB, None, _ctx(tmp_path))
    assert rec["outcome"] == "REFERENCE_FAILED"
    assert rec["measurements"]["coverage"]["unjudged"] == ["A0"]


def test_finite_differences_judge_only_the_columns_they_resolve(monkeypatch, tmp_path):
    # The two steps disagree by 1% in k's columns: too noisy to judge at 1e-3, so those
    # columns are left to the other references, and a wrong BNGsim there is still caught.
    _install(monkeypatch, {sens.FD: _fake_fd(noise=0.01)})
    rec = sens.run_correctness(JOB, None, _ctx(tmp_path))
    v = rec["verdicts"][sens.FD]
    assert rec["outcome"] == "REFERENCE_FAILED" and v["passed"]
    # A0's three columns, and the total's k column (zero, so its quotients agree).
    assert v["n_judged"] == 4 and rec["measurements"]["coverage"]["unjudged"] == ["k"]


def test_preparation_failure_is_a_bad_test(monkeypatch, tmp_path):
    def boom(problem, ctx, setting):
        raise ValueError("no parameter to differentiate")

    monkeypatch.setattr(sens, "_prepare", boom)
    for run in (sens.run_correctness, sens.run_timing):
        rec = run(JOB, None, _ctx(tmp_path))
        assert rec["outcome"] == "BAD_TEST" and "no parameter" in rec["error"]


@pytest.mark.parametrize("bn_error, ref_errors, verdicts, consistent, unjudged, outcome", [
    (None, {}, {"a": {"passed": True, "n_judged": 4}}, True, [], "PASS"),
    (None, {}, {"a": {"passed": False, "n_judged": 4}}, True, [], "DIFF"),
    # A DIFF is not hidden behind a crashed reference.
    (None, {"b": "x"}, {"a": {"passed": False, "n_judged": 4}}, True, [], "DIFF"),
    (None, {"b": "x"}, {"a": {"passed": True, "n_judged": 4}}, True, [], "REFERENCE_FAILED"),
    (None, {}, {"a": {"passed": False, "n_judged": 4, "refuted_by": ["b"]},
                "b": {"passed": True, "n_judged": 4}}, True, [], "REFERENCE_FAILED"),
    (None, {}, {"a": {"passed": True, "n_judged": 4}}, False, [], "DIFF"),
    (None, {}, {"a": {"passed": True, "n_judged": 4}}, True, ["k"], "REFERENCE_FAILED"),
    (None, {"a": "x", "b": "y"}, {}, True, [], "REFERENCE_FAILED"),
    ("boom", {"a": "x"}, {}, True, [], "EXCEPTION"),
    ("boom", {"a": "x", "b": "y"}, {}, True, [], "BAD_TEST"),
])
def test_outcome_attribution(bn_error, ref_errors, verdicts, consistent, unjudged, outcome):
    assert sens.sens_outcome(bn_error, ref_errors, 2, verdicts, consistent, unjudged) == outcome


def _timing_job(requires=None):
    return types.SimpleNamespace(
        definition={"timing": {"warmup_rounds": 1, "min_rounds": 3, "max_rounds": 3,
                               "budget_seconds": 0.0}},
        requires=requires, setting="auto")


class _Plain:
    """Stands in for sens.BngsimPlain / sens.RunNetworkPlain: a plain solve."""

    def __init__(self, *args, **kwargs):
        self.load_seconds = 0.001

    def run(self):
        return _exact(sens.times(_prep()["proto"]))[1]

    warm = run

    @staticmethod
    def trajectory(values):
        return np.asarray(values)

    def config(self):
        return {"solver": "dense"}


def test_timing_interleaves_every_engine_and_derives_the_fd_baselines(monkeypatch, tmp_path):
    _install(monkeypatch, _refs())
    monkeypatch.setattr(sens, "BngsimPlain", _Plain)
    monkeypatch.setattr(sens, "RunNetworkPlain", _Plain)
    rec = sens.run_timing(_timing_job(), None, _ctx(tmp_path))
    m = rec["measurements"]
    assert rec["outcome"] == "PASS"
    # The FD reference is not timed; its cost is derived from the plain solves.
    assert set(m["times"]) == {"bngsim", "bngsim_simultaneous", "bngsim_ode", "run_network",
                               "amici", "copasi", "bngsim_gradient", "amici_adjoint"}
    for name in ("bngsim_ode", "run_network"):
        b, t = m["fd_baselines"][name], m["times"][name]["median_seconds"]
        assert b["forward_seconds"] == pytest.approx(3 * t)
        assert b["central_seconds"] == pytest.approx(4 * t) and b["parameters"] == 2
    assert all(r["passed"] for r in m["reproducibility"].values())
    assert m["engine_roles"]["run_network"] == "fresh subprocess"
    assert "simultaneous corrector" in m["engine_roles"]["bngsim_simultaneous"]
    assert m["configurations"]["bngsim_simultaneous"]["sensitivity_method"] == "simultaneous"


def test_timing_flags_an_engine_whose_rerun_is_not_its_first_run(monkeypatch, tmp_path):
    class Drifting(_fake_reference(name="amici")):
        calls = 0

        def run(self):
            Drifting.calls += 1
            return _exact(self.t, (0, 0, 1e-3 * Drifting.calls))

    _install(monkeypatch, {"amici": Drifting})
    monkeypatch.setattr(sens, "BngsimPlain", _Plain)
    monkeypatch.setattr(sens, "RunNetworkPlain", _Plain)
    rec = sens.run_timing(_timing_job(), None, _ctx(tmp_path))
    assert rec["outcome"] == "NOT_REPRODUCIBLE"
    assert not rec["measurements"]["reproducibility"]["amici"]["passed"]


def test_timing_drops_a_crashing_or_timed_out_reference_and_times_the_rest(
        monkeypatch, tmp_path):
    from bench.results import result_path

    corr = "sens/m/x/auto/correctness"
    path = result_path(tmp_path, corr)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"outcome": "REFERENCE_FAILED",
                                "amici_error": "TIMEOUT: no result within the 3600 s budget"}))
    _install(monkeypatch, _refs(amici=_fake_reference(sleep=300, name="amici"),
                                copasi=_fake_reference(boom="x", name="copasi")))
    monkeypatch.setattr(sens, "BngsimPlain", _Plain)
    monkeypatch.setattr(sens, "RunNetworkPlain", _Plain)
    rec = sens.run_timing(_timing_job(requires=corr), None,
                          _ctx(tmp_path, results_dir=str(tmp_path)))
    assert "amici" not in rec["measurements"]["times"]
    assert "copasi" not in rec["measurements"]["times"]
    assert rec["amici_error"].startswith("TIMEOUT in the correctness job")
    assert rec["copasi_error"].startswith("RuntimeError")
    assert rec["outcome"] == "PASS"


# --------------------------------------------------------------------------- #
# A two-phase protocol: pre-equilibrate, then stimulate
# --------------------------------------------------------------------------- #
GATED_NET = """begin parameters
    1 k0    0.2  # Constant
    2 k1    0.5  # Constant
    3 A0    10  # Constant
    4 Stim  0  # Constant
end parameters
begin species
    1 A() A0
    2 B() 0
end species
begin reactions
    1 1 2 kA #decay
end reactions
begin functions
    1 kA() k0+Stim*k1
end functions
"""
TP_PROTO = {**PROTOCOL, "parameters": "independent", "t_end": 4.0, "n_steps": 8,
            "preequilibration": {"duration": 3.0, "set": {"Stim": 0}}, "stimulus": {"Stim": 1}}


def test_a_two_phase_protocol_names_each_phases_values_and_leaves_its_switch_out():
    st = netfile.structure(GATED_NET)
    tp = sens.two_phase(st, TP_PROTO)
    assert tp == {"duration": 3.0, "phase1": {"Stim": 0.0}, "phase2": {"Stim": 1.0}}
    params, excluded = sens.select_parameters(st, "independent", switched=set(tp["phase1"]),
                                              scales={"Stim": 1.0})
    assert params == ["k0", "k1", "A0"]
    assert excluded == {"Stim": "set by the two-phase protocol"}
    assert sens.two_phase(st, {**PROTOCOL}) is None


@pytest.mark.parametrize("stimulus, message", [
    ({"A0": 20}, "A0 reaches an initial amount"),
    ({"kA": 1}, "kA is not a number in the network"),
    ({"nope": 1}, "nope is not a number in the network"),
])
def test_a_two_phase_protocol_sets_rate_parameters_only(stimulus, message):
    with pytest.raises(ValueError, match=message):
        sens.two_phase(netfile.structure(GATED_NET), {**TP_PROTO, "stimulus": stimulus})


def test_a_two_phase_protocol_refuses_a_network_that_reads_the_clock():
    clocked = GATED_NET.replace("k0+Stim*k1", "k0+Stim*k1*time()")
    with pytest.raises(ValueError, match="read the clock"):
        sens.two_phase(netfile.structure(clocked), TP_PROTO)


def test_bngsim_runs_the_equilibration_then_the_stimulus_carrying_its_sensitivities():
    calls = []
    model = types.SimpleNamespace(set_param=lambda k, v: calls.append(("set", k, v)),
                                  reset=lambda: calls.append(("reset",)))

    def run(**kw):
        calls.append(("run", kw["t_span"], kw.get("carry_sensitivities", False)))
        return types.SimpleNamespace(species=np.array([[1.0, 0.0], [0.6, 0.4], [0.5, 0.5]]))

    engine = sens._BngsimProtocol()
    engine.proto, engine.atol = TP_PROTO, 1e-8
    engine.model, engine.sim = model, types.SimpleNamespace(run=run)
    engine.two_phase = {"duration": 3.0, "phase1": {"Stim": 0.0}, "phase2": {"Stim": 1.0}}
    engine._protocol(carry=True)
    assert calls == [("set", "Stim", 0.0), ("reset",), ("run", (-3.0, 0.0), False),
                     ("set", "Stim", 1.0), ("run", (0.0, 4.0), True)]
    assert engine.pre_drift == pytest.approx(0.2)  # 0.1 moved over the second half, of 0.5
    calls.clear()
    engine._protocol(carry=False)  # a plain solve: the same phases, nothing carried
    assert calls[-1] == ("run", (0.0, 4.0), False)


def _gated_exact(text, t_end, n_steps):
    """The closed-form solution of GATED_NET as written in ``text`` (its parameters
    and initial amounts), over ``0..t_end``: what run_network would write."""
    net = netfile.parse(text)
    k = net.parameters["k0"] + net.parameters["Stim"] * net.parameters["k1"]
    a0, b0 = net.initial
    t = np.linspace(0.0, t_end, n_steps + 1)
    a = a0 * np.exp(-k * t)
    return np.stack([a, b0 + a0 - a], axis=1)


def test_fd_differentiates_through_both_phases(monkeypatch, tmp_path):
    proto = {**TP_PROTO, **sens.FD_DEFAULTS}
    seen = []

    def process(self, text, t_end, n_steps):
        seen.append((netfile.parse(text).parameters["Stim"], t_end, n_steps))
        return _gated_exact(text, t_end, n_steps)

    monkeypatch.setattr(sens.FdRunNetwork, "_process", process)
    net = tmp_path / "m.net"
    net.write_text(GATED_NET)
    tp = sens.two_phase(netfile.structure(GATED_NET), proto)
    fd = sens.FdRunNetwork(net, proto, ["k0", "k1", "A0"], np.array([0.2, 0.5, 10.0]),
                           np.array([0.2, 0.5, 10.0]), tmp_path / "fd", timeout=10,
                           two_phase=tp)
    (fine, coarse), X = sens.unpack(fd.run(), 3)
    # Every protocol solve is two processes: the equilibration (Stim 0, over its
    # duration), then the measured phase (Stim 1) from the state it ended in.
    assert len(seen) == 2 * (4 * 3 + 1)
    assert seen[:2] == [(0.0, 3.0, 2), (1.0, 4.0, 8)]
    t = sens.times(proto)
    a = 10.0 * np.exp(-0.2 * 3.0 - 0.7 * t)
    np.testing.assert_allclose(X[:, 0], a, rtol=1e-12)
    np.testing.assert_allclose(fine[:, 0, 0], -(3.0 + t) * a, rtol=1e-6)   # d/dk0
    np.testing.assert_allclose(fine[:, 0, 1], -t * a, rtol=1e-6, atol=1e-12)  # d/dk1
    np.testing.assert_allclose(fine[:, 0, 2], a / 10.0, rtol=1e-6)          # d/dA0
    assert fd.config()["preequilibration"]["processes_per_solve"] == 2


def test_copasi_does_not_apply_to_a_two_phase_protocol_and_the_rest_still_judge(
        monkeypatch, tmp_path):
    class Copasi(_fake_reference(name="copasi")):
        not_applicable = staticmethod(sens.CopasiSens.not_applicable)

    def prep(problem, ctx, setting):
        out = _prep()
        out["two_phase"] = {"duration": 3.0, "phase1": {"Stim": 0.0}, "phase2": {"Stim": 1.0}}
        return out

    _install(monkeypatch, _refs(copasi=Copasi))
    monkeypatch.setattr(sens, "_prepare", prep)
    rec = sens.run_correctness(JOB, None, _ctx(tmp_path))
    assert rec["outcome"] == "PASS", rec
    assert set(rec["verdicts"]) == {"amici", sens.FD}
    assert "copasi_error" not in rec and "two-phase" in rec["not_applicable"]["copasi"]
    monkeypatch.setattr(sens, "BngsimPlain", _Plain)
    monkeypatch.setattr(sens, "RunNetworkPlain", _Plain)
    timing = sens.run_timing(_timing_job(), None, _ctx(tmp_path))
    assert "copasi" not in timing["measurements"]["times"]
    assert timing["not_applicable"] == rec["not_applicable"]


# --------------------------------------------------------------------------- #
# The adjoint gradient of a log-likelihood, timed beside BNGsim's (D7)
# --------------------------------------------------------------------------- #
def test_the_synthetic_likelihood_has_nonzero_residuals_and_a_noise_scale_everywhere():
    y = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    data, sigma = sens.synthetic_likelihood(y, atol=1e-8)
    np.testing.assert_allclose(data, 0.9 * y)
    np.testing.assert_allclose(sigma[:, 0], 0.4)
    np.testing.assert_allclose(sigma[:, 1], 1e-6)  # never moves: 100 atols


def test_gradient_agreement_is_semi_scaled_over_what_amici_differentiates():
    out = sens.gradient_agreement(np.array([1.0, 2.0, 5.0]), np.array([1.0, 2.1, np.nan]),
                                  np.array([1.0, 1.0, 1.0]))
    assert out["n_parameters"] == 2 and out["max_rel_difference"] == pytest.approx(0.1 / 2.0)


def test_timing_times_amicis_adjoint_beside_bngsims_forward_gradient(monkeypatch, tmp_path):
    _install(monkeypatch, _refs())
    monkeypatch.setattr(sens, "BngsimPlain", _Plain)
    monkeypatch.setattr(sens, "RunNetworkPlain", _Plain)
    rec = sens.run_timing(_timing_job(), None, _ctx(tmp_path))
    m = rec["measurements"]
    assert rec["outcome"] == "PASS"
    assert {"bngsim_gradient", "amici_adjoint"} <= set(m["times"])
    assert m["reproducibility"]["amici_adjoint"]["passed"]
    assert m["adjoint"]["agreement"]["max_rel_difference"] == pytest.approx(0.0, abs=1e-12)
    assert m["adjoint"]["observables"] == 1
    assert "timing only" in m["engine_roles"]["amici_adjoint"]
    assert m["references"]["amici_adjoint"] == {"shared_build": "amici"}


def test_a_failing_adjoint_is_recorded_and_the_rest_still_timed(monkeypatch, tmp_path):
    class NoAdjoint(_fake_reference(name="amici")):
        def adjoint_gradient(self, problem):
            raise RuntimeError("AMICI adjoint simulation failed with status -1")

    _install(monkeypatch, _refs(amici=NoAdjoint))
    monkeypatch.setattr(sens, "BngsimPlain", _Plain)
    monkeypatch.setattr(sens, "RunNetworkPlain", _Plain)
    rec = sens.run_timing(_timing_job(), None, _ctx(tmp_path))
    assert rec["outcome"] == "PASS" and "amici" in rec["measurements"]["times"]
    assert "amici_adjoint" not in rec["measurements"]["times"]
    assert rec["amici_adjoint_error"].startswith("RuntimeError: AMICI adjoint")


def test_the_trajectory_diagnostic_does_not_read_rounding_noise_as_a_disagreement():
    a = np.array([[0.0, 1.0], [0.0, 2.0]])
    b = np.array([[3e-21, 1.0], [-2e-21, 2.002]])
    assert sens.trajectory_error(a, b, atol=1e-8) == pytest.approx(0.002 / 2.002, rel=1e-9)
