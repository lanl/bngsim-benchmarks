"""The PSA capability's own logic, with fake engines (no simulator): job generation and
planning, the protocol's validation, the per-variant seeds, the run_network PSA
command, how BNGsim's PSA diagnostics become a regime, and how the ensembles become
gated verdicts, reported accuracy, flags and an outcome -- and the timing job."""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest

from bench import gather, specs
from bench.capabilities import psa, ssa

PROTOCOL = {"netgen": "generate_network({overwrite=>1})", "t_end": 100, "n_steps": 10,
            "replicates": 200}


@pytest.fixture
def psa_spec(spec_data):
    spec_data["capabilities"] = ["psa"]
    spec_data["protocol"] = {"psa": dict(PROTOCOL)}
    spec_data["size"] = {"species": 2, "reactions": 4, "psa_events": 100}
    return spec_data


# --------------------------------------------------------------------------- #
# Specs, jobs and planning
# --------------------------------------------------------------------------- #
def test_registry_emits_one_correctness_and_one_timing_job(psa_spec, write_spec):
    problem = specs.load(write_spec(psa_spec))
    correctness, timing = psa.jobs(problem)
    assert correctness.id == "psa/toy_model/bngsim-vs-run_network/auto/correctness"
    assert timing.requires == correctness.id and timing.kind == "timing"
    proto = correctness.definition["protocol"]
    # The defaults are part of the definition, so changing one restales the results.
    assert proto["poplevels"] == list(psa.DEFAULT_POPLEVELS) == [10, 30, 100, 300, 1000]
    assert proto["seed"] == ssa.DEFAULT_SEED
    assert correctness.definition["verdict"] == psa.VERDICT
    assert correctness.definition["accuracy"] == psa.ACCURACY
    assert "verdict" not in timing.definition and "accuracy" not in timing.definition
    assert timing.definition["timing"] == ssa.TIMING_PROTOCOL
    # One budget per ensemble, two engines at every level, plus the verdicts.
    assert correctness.timeout_seconds >= 13 * psa.engine_budget_seconds(problem)


def test_the_verdict_is_the_ssa_capabilitys_permutation_test():
    for k in ("two_sample", "alpha", "permutations", "stop_exceedances"):
        assert psa.VERDICT[k] == ssa.VERDICT[k]
    assert psa.VERDICT["alpha"] == 1e-3


def test_non_bngl_source_gets_no_jobs(psa_spec, write_spec):
    psa_spec["source"]["format"] = "sbml"
    assert psa.jobs(specs.load(write_spec(psa_spec))) == []


def test_adding_psa_to_a_spec_leaves_its_ssa_jobs_unchanged(psa_spec, write_spec):
    psa_spec["capabilities"] = ["ssa"]
    psa_spec["protocol"] = {"ssa": dict(PROTOCOL)}
    ssa_only = specs.load(write_spec(psa_spec))
    psa_spec["capabilities"] = ["ssa", "psa"]
    psa_spec["protocol"]["psa"] = dict(PROTOCOL, t_end=1000, poplevels=[100])
    both = specs.load(write_spec(psa_spec))
    assert [j.canonical() for j in ssa.jobs(ssa_only)] == [j.canonical() for j in ssa.jobs(both)]
    assert len(psa.jobs(both)) == 2


@pytest.mark.parametrize("change, message", [
    ({"poplevels": 100}, "poplevels: a list"),
    ({"poplevels": []}, "poplevels: a list"),
    ({"poplevels": [1, 10]}, "each greater than 1"),
    ({"poplevels": [10, True]}, "poplevels: a list"),
    ({"poplevels": [10, float("inf")]}, "poplevels: a list"),
    ({"poplevels": list(range(2, 14))}, "poplevels: a list of 1 to 10"),
    ({"poplevels": [100, 30]}, "strictly increasing"),
    ({"poplevels": [30, 30]}, "strictly increasing"),
    ({"moments": "linear", "replicates": 10_000}, "unknown key 'moments'"),
    ({"replicates": 50}, "replicates: an integer from 100"),
    ({"seed": 2**31}, "seed"),
    ({"t_start": 5}, "unknown key 't_start'"),
])
def test_invalid_psa_protocol_is_refused_with_its_name(psa_spec, write_spec, change, message):
    psa_spec["protocol"]["psa"].update(change)
    with pytest.raises(specs.SpecError, match=message.replace("(", r"\(")):
        specs.load(write_spec(psa_spec))


def test_a_grid_of_any_increasing_numbers_above_one_is_accepted(psa_spec, write_spec):
    psa_spec["protocol"]["psa"]["poplevels"] = [2.5, 50, 1e4]
    assert specs.load(write_spec(psa_spec)).protocol_for("psa")["poplevels"] == [2.5, 50, 1e4]


def test_planning_grows_with_replicates_events_and_the_grid(psa_spec, write_spec):
    small = specs.load(write_spec(psa_spec))
    psa_spec["protocol"]["psa"]["replicates"] = 10_000
    psa_spec["size"]["psa_events"] = 10_000
    big = specs.load(write_spec(psa_spec))
    (c_small, t_small), (c_big, t_big) = psa.planning_seconds(small), psa.planning_seconds(big)
    assert c_big > 5 * c_small and t_big >= t_small > 0
    psa_spec["protocol"]["psa"]["poplevels"] = [100]
    one = specs.load(write_spec(psa_spec))
    assert psa.planning_seconds(one)[0] < c_big
    assert psa.engine_budget_seconds(small) == ssa.MIN_TIMEOUT_SECONDS
    # size.ssa_events stands in for psa_events (the SSA capability's spec field).
    del psa_spec["size"]["psa_events"]
    psa_spec["size"]["ssa_events"] = 10_000
    assert psa.trajectory_seconds(specs.load(write_spec(psa_spec))) == \
        psa.trajectory_seconds(big)


def test_a_run_network_trajectory_is_stopped_long_before_its_ensembles_budget(
        psa_spec, write_spec):
    problem = specs.load(write_spec(psa_spec))
    budget = psa.engine_budget_seconds(problem)
    assert psa.trajectory_timeout(problem, budget) == psa.MIN_TRAJECTORY_TIMEOUT < budget
    psa_spec["size"]["psa_events"] = 10**8  # a long trajectory gets its time
    big = specs.load(write_spec(psa_spec))
    assert psa.trajectory_timeout(big, 1e9) > 1e4
    assert psa.trajectory_timeout(big, 600.0) == 600.0  # never past the budget


def test_the_psa_net_suite_loads_and_declares_what_it_tests():
    from conftest import REPO

    problems = specs.load_all(REPO / "problems")
    members = specs.suite_members(problems)["psa-net"]
    assert len(members) == specs.SUITE_SIZE
    by_id = {p.id: p for p in members}
    lin = by_id["bursty_autoregulated_gene_expression_lin2016"]
    rohrs = by_id["car_cd3zeta_phosphorylation_rohrs2018"]
    assert lin.suites["psa-net"] == 1 and rohrs.suites["psa-net"] == 6
    assert set(lin.capabilities) == {"ssa", "psa"} and rohrs.capabilities == ("psa",)
    for p in members:
        proto = psa.protocol(p)
        assert p.layer == "core", p.id
        assert proto["poplevels"] == [10, 30, 100, 300, 1000], p.id
        assert len(psa.jobs(p)) == 2, p.id
        assert psa.ensemble_size(proto, None) <= psa.ensemble_size(proto, 10), p.id
    # Only the members whose exact trajectory costs tens of seconds draw fewer of them.
    assert {p.id for p in members if "exact_replicates" in p.protocol_for("psa")} == {
        "mapk_relaxation_oscillations_kochanczyk2017", "tcr_signaling_bistability_lipniacki2008"}


# --------------------------------------------------------------------------- #
# Names, seeds and the run_network command
# --------------------------------------------------------------------------- #
def test_variant_names():
    assert psa.variant("bngsim", None) == "bngsim"
    assert psa.variant("run_network", 100) == "run_network@nc=100"
    assert psa.variant("bngsim", 100.0) == "bngsim@nc=100"
    assert psa.variant("bngsim", 2.5) == "bngsim@nc=2.5"
    assert psa.exactness_key(1000) == "exact@nc=1000"
    assert psa.base_engine("run_network@nc=30") == "run_network"
    proto = {"poplevels": [10, 1000]}
    assert [v[0] for v in psa.variants(proto)] == [
        "bngsim", "run_network", "bngsim@nc=10", "run_network@nc=10",
        "bngsim@nc=1000", "run_network@nc=1000"]


def test_every_ensemble_draws_from_its_own_seed_block():
    proto = {"poplevels": list(range(2, 12))}  # the most a spec may ask for
    firsts = [psa.seed_for(engine, k, 1) for _, engine, _, k in psa.variants(proto)]
    assert len(set(firsts)) == len(firsts)
    # Exact SSA draws the ssa capability's ensembles for the two engines.
    for engine in psa.ENGINES:
        assert psa.seed_for(engine, 0, 7) == ssa.seed_for(engine, 7)
    # No ensemble reaches the next block, and every seed is a C int for run_network.
    assert 10**6 < ssa.SEED_BLOCK
    assert max(psa.seed_for(e, k, 2**30) for _, e, _, k in psa.variants(proto)) + 10**6 < 2**31


def test_run_network_psa_command_is_what_bng2pl_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(ssa.env, "run_network_bin", lambda: Path("/bng/bin/run_network"))
    monkeypatch.setattr(ssa.env, "run_network_launch", lambda: [])
    eng = psa.RunNetworkPSA(tmp_path / "m.net", {"t_end": 600, "n_steps": 300},
                            tmp_path / "o" / "rn", timeout=10, poplevel=100)
    cmd = ssa.run_network_command(eng.net, eng.proto, eng.out_prefix, 7, eng.METHOD_ARGS)
    # BNG2.pl: -o, -p psa, --poplevel, then the method's options (-h seed), then the rest.
    assert cmd[1:9] == ["-o", str(eng.out_prefix), "-p", "psa", "--poplevel", "100", "-h", "7"]
    cfg = eng.config()
    assert cfg["poplevel"] == 100.0 and "psa" in cfg["method"]
    assert cfg["command"][:8] == ["-o", "<out>", "-p", "psa", "--poplevel", "100", "-h", "<seed>"]
    # The exact-SSA engine still runs -p ssa.
    assert ssa.RunNetworkSSA.METHOD_ARGS == ("-p", "ssa")


def _diag(qexc, crossed=True, speedup=2.0, mbar=None):
    return {"active": True, "reaction_index": list(range(1, len(qexc) + 1)), "qexc_integral": qexc,
            "mbar": mbar or [1.0 + q for q in qexc], "activation_crossed": crossed,
            "speedup": speedup, "peak_population": 500.0}


def test_a_trajectory_scaled_only_if_some_channel_fired_with_a_multiplier():
    scaled = psa.psa_summary([_diag([0.0, 3.0]), _diag([0.0, 0.0], speedup=1.0)])
    assert scaled["regime"] == "scaled" and scaled["scaled_fraction"] == 0.5
    assert scaled["scaled_reactions"] == [2] and scaled["mbar_max"] == {"value": 4.0, "reaction": 2}
    assert scaled["speedup"]["median"] == 1.5
    # Activation crossed (populations past 2 N_c) is not the same as scaling.
    exact = psa.psa_summary([_diag([0.0, 0.0], speedup=1.0) for _ in range(3)])
    assert exact["regime"] == "exact" and exact["crossed_fraction"] == 1.0
    assert exact["scaled_reactions"] == []


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
N_T, N_SP = 11, 2
LAM = 5 * np.exp(-np.linspace(0, 1, N_T))[:, None] * np.ones(N_SP)


def _stack(n, seed, shift=0.0, inflate=1.0):
    """Counts around a decay, deterministic in the seed; ``inflate`` widens the law
    without moving its mean (what scaling does), ``shift`` moves the mean."""
    rng = np.random.default_rng(seed)
    x = rng.poisson(LAM, (n, N_T, N_SP)).astype(float)
    x = LAM + inflate * (x - LAM) + shift
    x[:, 0, :] = 5.0  # the initial condition
    return x


def test_effect_sizes_measure_the_error_and_the_noise_it_carries():
    exact = _stack(4000, 1)
    out = psa.effect_sizes(_stack(4000, 2, inflate=2.0), exact, species=["A()", "B()"])
    vr = out["variance_ratio"]
    assert 3.0 < vr["median"] < 5.0 and vr["min"] <= vr["median"] <= vr["max"]
    assert vr["n_cells"] == (N_T - 1) * N_SP  # the initial condition does not vary
    # A wider law at the same mean: errors within a few of their own standard errors.
    for key in ("mean_error_over_peak", "mean_shift_sd"):
        assert out[key]["max"] < 4 * out[key]["se"]
    assert out["mean_error_over_peak"]["max"] < 0.06
    assert out["mean_error_over_peak"]["at"]["species"] in ("A()", "B()")
    # A shifted law: far past its noise, in both units.
    shifted = psa.effect_sizes(_stack(4000, 3, shift=2.0), exact)
    assert shifted["mean_shift_sd"]["max"] > 0.5
    assert shifted["mean_error_over_peak"]["max"] == pytest.approx(2.0 / 5.0, abs=0.03)
    assert shifted["mean_error_over_peak"]["max"] > 10 * shifted["mean_error_over_peak"]["se"]


# --------------------------------------------------------------------------- #
# Correctness and timing, with fake engines
# --------------------------------------------------------------------------- #
GRID = [10, 100, 300]


def _prep(grid=GRID):
    proto = dict(PROTOCOL, seed=1, timing_replicates=8, poplevels=list(grid))
    return {"net": Path("m.net"), "sbml": Path("m.xml"), "proto": proto, "info": {},
            "species_names": ["A()", "B()"], "name_to_id": {}}


def _factory(laws=None, boom=None, regimes=None, built=None):
    """A stand-in for psa._build: each variant samples ``laws[name]`` = (shift, inflate)
    (default: the exact law), raises ``boom[name]``, and BNGsim's PSA reports
    ``regimes[nc]`` (default: scaled)."""
    laws, boom, regimes = laws or {}, boom or {}, regimes or {}

    class Fake:
        def __init__(self, name, nc):
            self.name, self.nc = name, nc
            if built is not None:
                built.append(name)

        def ensemble(self, n, seed):
            if self.name in boom:
                raise RuntimeError(boom[self.name])
            return _stack(n, seed, *laws.get(self.name, (0.0, 1.0)))

        ensemble_detailed = ensemble

        def last_summary(self):
            return self.config().get("psa")

        def config(self):
            cfg = {"load_seconds": 0.01, "codegen_seconds": 0.001}
            if self.name.startswith("bngsim@"):
                regime = regimes.get(self.nc, "scaled")
                cfg["psa"] = {"regime": regime,
                              "scaled_fraction": 1.0 if regime == "scaled" else 0.0,
                              "speedup": {"median": 3.0 if regime == "scaled" else 1.0}}
            return cfg

    def build(engine, nc, prep, *, out_prefix, timeout):
        return Fake(psa.variant(engine, nc), nc)

    return build


def _install(monkeypatch, grid=GRID, **kw):
    monkeypatch.setattr(psa, "_prepare", lambda problem, ctx: _prep(grid))
    monkeypatch.setattr(psa, "_build", _factory(**kw))


def _ctx(tmp_path, **kw):
    return {"workdir": str(tmp_path), "subprocess_timeout": 60, "engine_budget_seconds": 60,
            "trajectory_timeout_seconds": 30, **kw}


def _scaled_everywhere(inflate=2.0):
    """Both engines' PSA widen the law the same way at every N_c (the method's price)."""
    return {psa.variant(e, nc): (0.0, inflate) for e in psa.ENGINES for nc in GRID}


def test_correctness_passes_when_both_psas_agree_and_reports_their_error(monkeypatch, tmp_path):
    _install(monkeypatch, laws=_scaled_everywhere())
    rec = psa.run_correctness(None, None, _ctx(tmp_path))
    assert rec["outcome"] == "PASS", rec.get("verdicts")
    # Gated: like for like at every N_c and at N_c = infinity; nothing ran unscaled.
    assert set(rec["verdicts"]) == {"run_network", "run_network@nc=10", "run_network@nc=100",
                                    "run_network@nc=300"}
    for v in rec["verdicts"].values():
        assert v["passed"] and v["replicates"] == [200, 200] and "parity_core" in v
    # Reported: PSA is wider than exact, which is measured, not failed.
    acc = rec["measurements"]["psa"]["accuracy"]
    assert set(acc) == {"nc=10", "nc=100", "nc=300"}
    for a in acc.values():
        assert a["regime"] == "scaled"
        for engine in psa.ENGINES:
            assert a[engine]["means"]["passed"]
            assert a[engine]["variance_ratio"]["median"] > 2.0
    assert rec["measurements"]["psa"]["flags"] == [] and "psa_flags" not in rec
    ens = rec["measurements"]["ensembles"]
    assert set(ens) == {name for name, *_ in psa.variants(_prep()["proto"])}
    assert ens["bngsim"]["replicates"] == 200


def test_bngsims_psa_sampling_another_law_than_run_networks_is_a_diff(monkeypatch, tmp_path):
    laws = _scaled_everywhere()
    laws["bngsim@nc=100"] = (1.5, 2.0)
    _install(monkeypatch, laws=laws)
    rec = psa.run_correctness(None, None, _ctx(tmp_path))
    assert rec["outcome"] == "DIFF"
    assert not rec["verdicts"]["run_network@nc=100"]["passed"]
    assert rec["verdicts"]["run_network@nc=10"]["passed"]


def test_run_network_psa_aborting_is_a_reference_failure_and_the_rest_is_judged(
        monkeypatch, tmp_path):
    abort = "run_network failed at seed 3 (exit 1): ERROR: scaling factor of 0 th reaction < 1"
    boom = {psa.variant("run_network", nc): abort for nc in GRID}
    _install(monkeypatch, laws=_scaled_everywhere(), boom=boom)
    rec = psa.run_correctness(None, None, _ctx(tmp_path))
    assert rec["outcome"] == "REFERENCE_FAILED"
    for nc in GRID:
        assert "scaling factor" in rec[f"run_network@nc={nc}_error"]
    assert set(rec["verdicts"]) == {"run_network"} and rec["verdicts"]["run_network"]["passed"]
    acc = rec["measurements"]["psa"]["accuracy"]["nc=100"]
    assert acc["bngsim"]["means"]["passed"] and "run_network" not in acc


def test_where_nothing_scaled_bngsims_psa_must_be_exact_ssa(monkeypatch, tmp_path):
    regimes = {300: "exact"}
    _install(monkeypatch, laws=_scaled_everywhere() | {"bngsim@nc=300": (0.0, 1.0),
                                                       "run_network@nc=300": (0.0, 1.0)},
             regimes=regimes)
    rec = psa.run_correctness(None, None, _ctx(tmp_path))
    assert rec["outcome"] == "PASS"
    assert rec["verdicts"]["exact@nc=300"]["passed"]
    assert "exact@nc=100" not in rec["verdicts"]

    # The same N_c sampling another law is a defect, gated -- and a DIFF even while
    # every run_network PSA failed, so a wrong answer is never hidden or timed.
    boom = {psa.variant("run_network", nc): "aborted" for nc in GRID}
    _install(monkeypatch, laws={"bngsim@nc=300": (0.0, 2.0)}, regimes=regimes, boom=boom)
    rec = psa.run_correctness(None, None, _ctx(tmp_path))
    assert rec["outcome"] == "DIFF"
    assert not rec["verdicts"]["exact@nc=300"]["passed"]


def test_a_mean_shift_at_large_nc_is_flagged_not_failed(monkeypatch, tmp_path):
    # Both PSAs move the mean at every N_c, alike: like for like passes, and the
    # method's error is flagged where PSA is meant to be accurate (N_c >= 100).
    laws = {psa.variant(e, nc): (1.5, 2.0) for e in psa.ENGINES for nc in GRID}
    _install(monkeypatch, laws=laws)
    rec = psa.run_correctness(None, None, _ctx(tmp_path))
    assert rec["outcome"] == "PASS"
    flags = rec["measurements"]["psa"]["flags"]
    assert [f["poplevel"] for f in flags] == [100, 300]
    assert all(f["run_network_differs_too"] for f in flags)
    assert all(f["mean_shift_sd"] > 0.3 and f["p_value"] <= 1e-3 for f in flags)
    assert all(f["mean_error_over_peak"] > 0.15 for f in flags)  # 1.5 of a peak of 5
    assert rec["psa_flags"][0].startswith("nc=100: ")
    # Below N_c = 100 the same error is reported only.
    assert not rec["measurements"]["psa"]["accuracy"]["nc=10"]["bngsim"]["means"]["passed"]


def test_bngsim_failing_is_an_exception_or_a_timeout(monkeypatch, tmp_path):
    _install(monkeypatch, boom={"bngsim@nc=10": "no"})
    rec = psa.run_correctness(None, None, _ctx(tmp_path))
    assert rec["outcome"] == "EXCEPTION" and rec["bngsim_error"].startswith("bngsim@nc=10: ")
    boom = {name: "no" for name, *_ in psa.variants(_prep()["proto"])}
    _install(monkeypatch, boom=boom)
    assert psa.run_correctness(None, None, _ctx(tmp_path))["outcome"] == "BAD_TEST"

    class Slow:
        def __init__(self, *a, **k):
            pass

        def ensemble_detailed(self, n, seed):
            import time
            time.sleep(300)

    def build(engine, nc, prep, *, out_prefix, timeout):
        return Slow() if (engine, nc) == ("bngsim", 10) else _factory()(
            engine, nc, prep, out_prefix=out_prefix, timeout=timeout)

    monkeypatch.setattr(psa, "_build", build)
    rec = psa.run_correctness(None, None, _ctx(tmp_path, engine_budget_seconds=1.0))
    assert rec["outcome"] == "TIMEOUT" and "ensemble of 200" in rec["bngsim_error"]


def test_preparation_failure_is_a_bad_test(monkeypatch, tmp_path):
    def boom(problem, ctx):
        raise RuntimeError("BNG2.pl network generation failed")

    monkeypatch.setattr(psa, "_prepare", boom)
    for run in (psa.run_correctness, psa.run_timing):
        rec = run(None, None, _ctx(tmp_path))
        assert rec["outcome"] == "BAD_TEST" and "network generation" in rec["error"]


def _timing_job(requires=None):
    return types.SimpleNamespace(
        definition={"timing": {"warmup_rounds": 1, "min_rounds": 3, "max_rounds": 3,
                               "budget_seconds": 0.0}},
        requires=requires)


def test_timing_interleaves_every_variant_and_reports_per_trajectory_times(
        monkeypatch, tmp_path):
    _install(monkeypatch, grid=[10, 100])
    rec = psa.run_timing(_timing_job(), None, _ctx(tmp_path))
    m = rec["measurements"]
    assert rec["outcome"] == "PASS"
    names = {"bngsim", "run_network", "bngsim@nc=10", "run_network@nc=10", "bngsim@nc=100",
             "run_network@nc=100"}
    assert set(m["times"]) == names
    for name, t in m["times"].items():
        assert m["times_per_trajectory"][name]["median_seconds"] == pytest.approx(
            t["median_seconds"] / 8)
    assert all(r["passed"] and r["max_err_over_peak"] == 0.0 for r in m["reproducibility"].values())
    # The exact anchor where the tables expect BNGsim; every other variant beside it.
    assert set(m["references"]) == names - {"bngsim"}
    assert m["references"]["bngsim@nc=10"]["psa"]["regime"] == "scaled"
    assert m["engine_roles"]["run_network@nc=10"] == "fresh subprocess"
    assert m["engine_roles"]["bngsim@nc=10"].startswith("warm, in-process")


def test_timing_skips_what_failed_in_correctness_and_drops_a_failing_trial(
        monkeypatch, tmp_path):
    from bench.results import result_path

    corr = "psa/m/x/auto/correctness"
    path = result_path(tmp_path, corr)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"outcome": "REFERENCE_FAILED",
                                "run_network@nc=10_error": "run_network failed: aborted"}))
    built: list[str] = []
    monkeypatch.setattr(psa, "_prepare", lambda problem, ctx: _prep([10, 100]))
    monkeypatch.setattr(psa, "_build", _factory(boom={"run_network@nc=100": "crash"},
                                                built=built))
    rec = psa.run_timing(_timing_job(requires=corr), None,
                         _ctx(tmp_path, results_dir=str(tmp_path)))
    assert rec["outcome"] == "PASS"
    assert "run_network@nc=10" not in built  # not even built
    assert rec["run_network@nc=10_error"].startswith("failed in the correctness job")
    assert rec["run_network@nc=100_error"].startswith("RuntimeError: crash")
    assert set(rec["measurements"]["times"]) == {"bngsim", "run_network", "bngsim@nc=10",
                                                 "bngsim@nc=100"}
def test_the_gathered_report_has_a_row_per_nc(monkeypatch, tmp_path, psa_spec, write_spec):
    psa_spec["protocol"]["psa"]["poplevels"] = [10, 100]
    correctness, timing = psa.jobs(specs.load(write_spec(psa_spec)))
    laws = {psa.variant(e, nc): (1.5, 2.0) for e in psa.ENGINES for nc in (10, 100)}
    _install(monkeypatch, grid=[10, 100], laws=laws,
             boom={"run_network@nc=10": "run_network failed: aborted"})
    c_rec = psa.run_correctness(correctness, None, _ctx(tmp_path))
    t_rec = psa.run_timing(_timing_job(), None, _ctx(tmp_path))
    rows = {"psa-correctness": [gather._row(correctness, c_rec, True)],
            "psa-timing": [gather._row(timing, t_rec, True)]}
    text = gather.render(rows)
    assert "### PSA against exact SSA, per N_c" in text
    assert "### PSA speed-up over exact SSA of the same engine, per N_c" in text
    lines = [ln for ln in text.splitlines() if ln.startswith("| toy_model | 100 |")]
    assert len(lines) == 2  # one in each PSA table
    assert "FLAG" in lines[0] and "ok (p" in lines[0] and " ± " in lines[0]
    nc10 = [ln for ln in text.splitlines() if ln.startswith("| toy_model | 10 |")]
    assert "crash" in nc10[0] and "FLAG" not in nc10[0]
    assert rows["psa-timing"][0]["psa_speedup_events"]["bngsim@nc=10"] == 3.0


def test_exact_replicates_sizes_only_the_two_exact_ssa_ensembles(monkeypatch, tmp_path):
    prep = _prep()
    prep["proto"]["exact_replicates"] = 40
    monkeypatch.setattr(psa, "_prepare", lambda problem, ctx: prep)
    monkeypatch.setattr(psa, "_build", _factory())
    rec = psa.run_correctness(None, None, _ctx(tmp_path))
    assert rec["outcome"] == "PASS", rec.get("verdicts")
    sizes = {name: e["replicates"] for name, e in rec["measurements"]["ensembles"].items()}
    assert sizes["bngsim"] == sizes["run_network"] == 40
    assert {n for name, n in sizes.items() if "@nc=" in name} == {200}
    assert rec["verdicts"]["run_network"]["replicates"] == [40, 40]
    assert rec["verdicts"]["run_network@nc=10"]["replicates"] == [200, 200]
    # PSA where nothing scaled is still judged against the smaller exact ensemble.
    assert rec["measurements"]["psa"]["accuracy"]["nc=10"]["bngsim"]["means"]["replicates"] \
        == [200, 40]


@pytest.mark.parametrize("value, ok", [(20, True), (200, True), (19, False), (201, False),
                                       (40.0, False)])
def test_exact_replicates_is_validated(psa_spec, write_spec, value, ok):
    psa_spec["protocol"]["psa"]["exact_replicates"] = value
    if ok:
        p = specs.load(write_spec(psa_spec))
        assert psa.ensemble_size(psa.protocol(p), None) == value
        assert psa.ensemble_size(psa.protocol(p), 100) == 200
    else:
        with pytest.raises(specs.SpecError, match="exact_replicates: an integer from 20"):
            specs.load(write_spec(psa_spec))


def test_exact_replicates_lowers_the_planned_correctness_cost(psa_spec, write_spec):
    base = psa.planning_seconds(specs.load(write_spec(psa_spec)))[0]
    psa_spec["protocol"]["psa"]["exact_replicates"] = 20
    fewer = psa.planning_seconds(specs.load(write_spec(psa_spec)))[0]
    assert fewer < base
