"""The SSA capability's own logic, with fake engines (no simulator): job generation
and planning, the run_network command, the per-engine seeds, the screen that keeps
libRoadRunner off a repeated reactant, and how per-engine results become an
outcome in the correctness and timing jobs."""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest

from bench import netfile, specs
from bench.capabilities import ode, ssa

PROTOCOL = {"netgen": "generate_network({overwrite=>1})", "t_end": 100, "n_steps": 10,
            "replicates": 200}


@pytest.fixture
def ssa_spec(spec_data):
    spec_data["capabilities"] = ["ssa"]
    spec_data["protocol"] = {"ssa": dict(PROTOCOL)}
    spec_data["size"] = {"species": 2, "reactions": 4, "ssa_events": 100}
    return spec_data


# --------------------------------------------------------------------------- #
# Specs, jobs and planning
# --------------------------------------------------------------------------- #
def test_engines_id_names_bngsim_against_every_reference():
    assert ssa.engines_id() == "bngsim-vs-run_network-roadrunner-copasi"


def test_registry_emits_one_correctness_and_one_timing_job(ssa_spec, write_spec):
    problem = specs.load(write_spec(ssa_spec))
    correctness, timing = ssa.jobs(problem)
    assert (correctness.kind, timing.kind) == ("correctness", "timing")
    assert timing.requires == correctness.id
    assert correctness.id == f"ssa/toy_model/{ssa.engines_id()}/auto/correctness"
    # Defaults are part of the definition, so changing one restales the results.
    proto = correctness.definition["protocol"]
    assert proto["seed"] == ssa.DEFAULT_SEED
    assert proto["timing_replicates"] == ssa.DEFAULT_TIMING_REPLICATES
    # The verdict's statistics key correctness only; the timing protocol, timing only.
    assert correctness.definition["verdict"] == ssa.VERDICT
    assert "verdict" not in timing.definition
    assert timing.definition["timing"] == ssa.TIMING_PROTOCOL
    assert correctness.timeout_seconds >= (len(ssa.REFERENCE_ENGINES) + 2) \
        * ssa.engine_budget_seconds(problem)


def test_reference_versions_key_only_this_capability(ssa_spec, write_spec, monkeypatch):
    monkeypatch.setattr(ssa.env, "roadrunner_identity", lambda: {"version": "9.9.9"})
    monkeypatch.setattr(ssa.env, "copasi_identity", lambda: {"version": "4.48.311"})
    correctness, _ = ssa.jobs(specs.load(write_spec(ssa_spec)))
    assert correctness.definition["reference_versions"] == {
        "roadrunner": "9.9.9", "copasi": "4.48.311"}


def test_non_bngl_source_gets_no_jobs(ssa_spec, write_spec):
    ssa_spec["source"]["format"] = "sbml"
    assert ssa.jobs(specs.load(write_spec(ssa_spec))) == []


def test_a_spec_can_carry_ode_and_ssa_and_the_ode_jobs_do_not_change(spec_data, write_spec):
    ode_only = specs.load(write_spec(spec_data))
    spec_data["capabilities"] = ["ode", "ssa"]
    spec_data["protocol"]["ssa"] = dict(PROTOCOL)
    both = specs.load(write_spec(spec_data))
    assert [j.canonical() for j in ode.jobs(ode_only)] == [j.canonical() for j in ode.jobs(both)]
    assert len(ssa.jobs(both)) == 2


@pytest.mark.parametrize("change, message", [
    ({"replicates": 50}, "replicates: an integer from 100"),
    ({"replicates": 100.0}, "replicates: an integer from 100"),
    ({"t_end": 0}, "t_end: must be a positive number"),
    ({"t_end": "1e3"}, "t_end: must be a positive number"),
    ({"n_steps": 0}, "n_steps"),
    ({"seed": -1}, "seed"),
    ({"seed": 2**31}, "seed"),
    ({"timing_replicates": 0}, "timing_replicates"),
    ({"moments": "gaussian"}, "moments: 'linear'"),
    ({"moments": "linear"}, "needs replicates >= 10000"),
    ({"t_start": 5}, "unknown key 't_start'"),
    ({"netgen": "simulate({})"}, "netgen"),
])
def test_invalid_ssa_protocol_is_refused_with_its_name(ssa_spec, write_spec, change, message):
    ssa_spec["protocol"]["ssa"].update(change)
    with pytest.raises(specs.SpecError, match=message.replace("(", r"\(")):
        specs.load(write_spec(ssa_spec))


def test_exact_moments_at_dsmts_size_are_accepted(ssa_spec, write_spec):
    ssa_spec["protocol"]["ssa"].update(moments="linear", replicates=10_000)
    assert specs.load(write_spec(ssa_spec)).protocol_for("ssa")["moments"] == "linear"


def test_planning_grows_with_replicates_and_events(ssa_spec, write_spec):
    small = specs.load(write_spec(ssa_spec))
    ssa_spec["protocol"]["ssa"]["replicates"] = 10_000
    ssa_spec["size"]["ssa_events"] = 10_000
    big = specs.load(write_spec(ssa_spec))
    (c_small, t_small), (c_big, t_big) = ssa.planning_seconds(small), ssa.planning_seconds(big)
    assert c_big > 5 * c_small and t_big >= t_small > 0
    # The budget never drops below the floor nor exceeds the cap.
    assert ssa.engine_budget_seconds(small) == ssa.MIN_TIMEOUT_SECONDS
    huge = types.SimpleNamespace(size={"ssa_events": 10**8},
                                 protocol_for=lambda c: dict(PROTOCOL, replicates=10**6))
    assert ssa.engine_budget_seconds(huge) == ssa.ENGINE_BUDGET_CAP_SECONDS


def test_the_pilot_specs_load_and_declare_what_they_test():
    from conftest import REPO

    problems = {p.id: p for p in specs.load_all(REPO / "problems")}
    shz = problems["three_stage_stochastic_gene_expression_shahrezaei2008"]
    lin = problems["bursty_autoregulated_gene_expression_lin2016"]
    assert shz.layer == lin.layer == "pilot" and not shz.suites and not lin.suites
    assert shz.protocol_for("ssa")["moments"] == "linear"
    assert "moments" not in lin.protocol_for("ssa")


# --------------------------------------------------------------------------- #
# Seeds and the run_network command
# --------------------------------------------------------------------------- #
def test_every_engine_draws_from_its_own_seed_block():
    firsts = [ssa.seed_for(name, 1) for name in ("bngsim", *ssa.REFERENCE_ENGINES)]
    assert firsts == [1, 10_000_001, 20_000_001, 30_000_001]
    # The largest ensemble a spec may ask for never reaches the next block, and
    # the largest seed fits COPASI's C int.
    assert 10**6 < ssa.SEED_BLOCK
    assert ssa.seed_for("copasi", 2**30) + 10**6 < 2**31


def test_run_network_command_is_what_bng2pl_runs_for_ssa(monkeypatch):
    monkeypatch.setattr(ssa.env, "run_network_bin", lambda: Path("/bng/bin/run_network"))
    monkeypatch.setattr(ssa.env, "run_network_launch", lambda: [])
    cmd = ssa.run_network_command(Path("m.net"), {"t_end": 600, "n_steps": 300}, Path("o/rn"), 7)
    assert cmd == ["/bng/bin/run_network", "-o", "o/rn", "-p", "ssa", "-h", "7",
                   "--cdat", "1", "--fdat", "0", "-g", "m.net", "m.net", "2.0", "300"]


def test_run_network_config_records_no_scratch_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(ssa.env, "run_network_bin", lambda: Path("/bng/bin/run_network"))
    monkeypatch.setattr(ssa.env, "run_network_launch", lambda: ["/glibc/ld.so"])
    eng = ssa.RunNetworkSSA(tmp_path / "netgen" / "model.net", {"t_end": 10, "n_steps": 5},
                            tmp_path / "rn" / "rn", timeout=10)
    cmd = eng.config()["command"]
    assert str(tmp_path) not in " ".join(cmd)
    assert cmd[:6] == ["-o", "<out>", "-p", "ssa", "-h", "<seed>"]


@pytest.mark.parametrize("body, how", [
    ("echo 'edgepop: X'\nkill -SEGV $$", r"killed by signal 11\): edgepop: X"),
    ("echo 'ERROR: scaling factor of 0 th reaction < 1' >&2\nexit 1",
     r"exit 1\): ERROR: scaling factor"),
])
def test_run_network_runs_in_its_output_directory_and_says_how_it_failed(
        monkeypatch, tmp_path, body, how):
    script = tmp_path / "bin" / "run_network"
    script.parent.mkdir()
    # A stand-in binary: where it ran, then a crash or an error exit (no core file).
    script.write_text(f"#!/bin/sh\nulimit -c 0\npwd > ran_in.txt\n{body}\n")
    script.chmod(0o755)
    monkeypatch.setattr(ssa.env, "run_network_bin", lambda: script)
    monkeypatch.setattr(ssa.env, "run_network_launch", lambda: [])
    eng = ssa.RunNetworkSSA(tmp_path / "m.net", {"t_end": 10, "n_steps": 5},
                            tmp_path / "out" / "rn", timeout=10)
    with pytest.raises(RuntimeError, match=r"run_network failed at seed 7 \(" + how):
        eng.trajectory(7)
    ran_in = Path((tmp_path / "out" / "ran_in.txt").read_text().strip())
    assert ran_in.resolve() == (tmp_path / "out").resolve()


# --------------------------------------------------------------------------- #
# The repeated-reactant screen
# --------------------------------------------------------------------------- #
DIMER_NET = netfile.parse("""begin parameters
 1 k 1
end parameters
begin species
 1 A() 10
 2 B() 0
end species
begin reactions
 1 1,1 2 0.5*k #R_dimerize
end reactions
""")


def test_roadrunner_is_not_run_on_a_repeated_reactant():
    reason = ssa.RoadRunnerSSA.unsupported(DIMER_NET)
    assert reason.startswith("UNSUPPORTED") and "R_dimerize" in reason
    assert ssa.RoadRunnerSSA.unsupported(netfile.parse("begin species\n 1 A() 1\nend species\n")) \
        is None
    assert ssa.RoadRunnerSSA.unsupported(None) is None


# --------------------------------------------------------------------------- #
# Correctness and timing, with fake engines
# --------------------------------------------------------------------------- #
N_T, N_SP = 11, 2


def _fake_stack(n, seed, shift=0.0):
    """Counts from the same law for every engine (Poisson around a decay), unless
    shifted; deterministic in the seed, like a real engine."""
    rng = np.random.default_rng(seed)
    lam = 5 * np.exp(-np.linspace(0, 1, N_T))[:, None] * np.ones(N_SP) + shift
    out = rng.poisson(lam, (n, N_T, N_SP)).astype(float)
    out[:, 0, :] = 5.0  # the initial condition
    return out


def _prep(moments=None):
    proto = dict(PROTOCOL, seed=1, timing_replicates=8)
    if moments:
        proto["moments"] = moments
    return {"net": Path("m.net"), "sbml": Path("m.xml"), "proto": proto, "info": {},
            "species_names": ["A()", "B()"], "name_to_id": {"A()": "A", "B()": "B"},
            "network": None}


def _fake_bngsim(shift=0.0, boom=None):
    class FakeBngsim:
        NAME = "bngsim"

        def __init__(self, net, proto):
            if boom:
                raise RuntimeError(boom)

        def ensemble(self, n, seed):
            return _fake_stack(n, seed, shift)

        ensemble_detailed = ensemble

        def config(self):
            return {"load_seconds": 0.01, "codegen_seconds": 0.1, "propensity_backend": "cc"}

    return FakeBngsim


def _fake_reference(shift=0.0, boom=None, sleep=0.0):
    class FakeRef:
        INPUT = "sbml"

        def __init__(self, *args, **kwargs):
            pass

        def ensemble(self, n, seed):
            if sleep:
                import time
                time.sleep(sleep)
            if boom:
                raise RuntimeError(boom)
            return _fake_stack(n, seed + 12345, shift)

        def config(self):
            return {"load_seconds": 0.02}

    return FakeRef


def _install(monkeypatch, refs, bngsim=None, moments=None):
    monkeypatch.setattr(ssa, "_prepare", lambda problem, ctx: _prep(moments))
    monkeypatch.setattr(ssa, "BngsimSSA", bngsim or _fake_bngsim())
    monkeypatch.setattr(ssa, "REFERENCE_ENGINES", refs)


def _ctx(tmp_path, **kw):
    return {"workdir": str(tmp_path), "subprocess_timeout": 60, "engine_budget_seconds": 60,
            **kw}


def test_correctness_passes_when_bngsim_samples_the_references_law(monkeypatch, tmp_path):
    _install(monkeypatch, {"run_network": _fake_reference(), "copasi": _fake_reference()})
    rec = ssa.run_correctness(None, None, _ctx(tmp_path))
    assert rec["outcome"] == "PASS", rec
    assert set(rec["verdicts"]) == {"run_network", "copasi"}
    for v in rec["verdicts"].values():
        assert v["passed"] and v["replicates"] == [200, 200] and "parity_core" in v
    ens = rec["measurements"]["ensembles"]
    assert set(ens) == {"bngsim", "run_network", "copasi"}
    assert ens["bngsim"]["replicates"] == 200 and "A()" in ens["bngsim"]["final_mean"]


def test_correctness_is_a_diff_when_bngsim_samples_another_law(monkeypatch, tmp_path):
    _install(monkeypatch, {"run_network": _fake_reference(), "copasi": _fake_reference()},
             bngsim=_fake_bngsim(shift=1.5))
    rec = ssa.run_correctness(None, None, _ctx(tmp_path))
    assert rec["outcome"] == "DIFF"
    assert not any(v["passed"] for v in rec["verdicts"].values())


def test_a_crashing_reference_is_recorded_and_the_rest_still_judge(monkeypatch, tmp_path):
    _install(monkeypatch, {"run_network": _fake_reference(boom="edgepop"),
                           "copasi": _fake_reference()})
    rec = ssa.run_correctness(None, None, _ctx(tmp_path))
    assert rec["outcome"] == "REFERENCE_FAILED"
    assert "edgepop" in rec["run_network_error"]
    assert set(rec["verdicts"]) == {"copasi"} and rec["verdicts"]["copasi"]["passed"]


def test_a_reference_out_of_its_budget_is_a_reference_failure(monkeypatch, tmp_path):
    _install(monkeypatch, {"run_network": _fake_reference(sleep=300), "copasi": _fake_reference()})
    rec = ssa.run_correctness(None, None, _ctx(tmp_path, engine_budget_seconds=1.0))
    assert rec["outcome"] == "REFERENCE_FAILED"
    err = rec["run_network_error"]
    assert err.startswith("TIMEOUT") and "ensemble of 200" in err
    assert rec["verdicts"]["copasi"]["passed"]


def test_bngsim_failing_is_an_exception_and_failing_with_every_reference_a_bad_test(
        monkeypatch, tmp_path):
    _install(monkeypatch, {"copasi": _fake_reference()}, bngsim=_fake_bngsim(boom="no"))
    assert ssa.run_correctness(None, None, _ctx(tmp_path))["outcome"] == "EXCEPTION"
    _install(monkeypatch, {"copasi": _fake_reference(boom="no")}, bngsim=_fake_bngsim(boom="no"))
    assert ssa.run_correctness(None, None, _ctx(tmp_path))["outcome"] == "BAD_TEST"


def test_an_unsupported_reference_is_not_run_and_reads_as_a_reference_failure(
        monkeypatch, tmp_path):
    ran = []

    class Picky(_fake_reference()):
        @staticmethod
        def unsupported(network):
            return "UNSUPPORTED: not this model"

        def ensemble(self, n, seed):
            ran.append(n)
            return super().ensemble(n, seed)

    _install(monkeypatch, {"roadrunner": Picky, "copasi": _fake_reference()})
    rec = ssa.run_correctness(None, None, _ctx(tmp_path))
    assert rec["outcome"] == "REFERENCE_FAILED" and not ran
    assert rec["roadrunner_error"] == "UNSUPPORTED: not this model"


def test_exact_moments_are_a_reference_and_judge_every_engine(monkeypatch, tmp_path):
    _install(monkeypatch, {"copasi": _fake_reference()}, moments="linear")
    t = np.linspace(0, 1, N_T)
    lam = 5 * np.exp(-t)[:, None] * np.ones(N_SP)
    lam_var = lam.copy()
    lam_var[0] = 0.0  # the initial condition is exact
    lam[0] = 5.0
    monkeypatch.setattr(ssa, "exact_moments", lambda prep: (lam, lam_var))
    rec = ssa.run_correctness(None, None, _ctx(tmp_path))
    assert rec["outcome"] == "PASS", rec["verdicts"]
    assert rec["verdicts"]["analytic"]["passed"] and "p_min" in rec["verdicts"]["analytic"]
    assert rec["measurements"]["reference_vs_analytic"]["copasi"]["passed"]
    assert "analytic" in rec["measurements"]["ensembles"]


def test_moments_that_cannot_be_computed_are_a_failed_reference(monkeypatch, tmp_path):
    _install(monkeypatch, {"copasi": _fake_reference()}, moments="linear")
    rec = ssa.run_correctness(None, None, _ctx(tmp_path))  # network is None: unreadable
    assert rec["outcome"] == "REFERENCE_FAILED" and "analytic_error" in rec


def test_preparation_failure_is_a_bad_test(monkeypatch, tmp_path):
    def boom(problem, ctx):
        raise RuntimeError("BNG2.pl network generation failed")

    monkeypatch.setattr(ssa, "_prepare", boom)
    for run in (ssa.run_correctness, ssa.run_timing):
        rec = run(None, None, _ctx(tmp_path))
        assert rec["outcome"] == "BAD_TEST" and "network generation" in rec["error"]


def _timing_job(requires=None):
    return types.SimpleNamespace(
        definition={"timing": {"warmup_rounds": 1, "min_rounds": 3, "max_rounds": 3,
                               "budget_seconds": 0.0}},
        requires=requires)


def test_timing_interleaves_ensembles_and_reports_per_trajectory_times(monkeypatch, tmp_path):
    _install(monkeypatch, {"run_network": _fake_reference(), "copasi": _fake_reference()})
    rec = ssa.run_timing(_timing_job(), None, _ctx(tmp_path))
    m = rec["measurements"]
    assert rec["outcome"] == "PASS"
    assert set(m["times"]) == {"bngsim", "run_network", "copasi"}
    assert m["timing_replicates"] == 8
    for name, t in m["times"].items():
        assert m["times_per_trajectory"][name]["median_seconds"] == pytest.approx(
            t["median_seconds"] / 8)
    # The same seeds every round: each engine reproduced its first ensemble exactly.
    assert all(r["passed"] and r["max_err_over_peak"] == 0.0
               for r in m["reproducibility"].values())
    assert m["engine_roles"]["run_network"] == "fresh subprocess"
    assert m["bngsim"]["propensity_backend"] == "cc"


def test_timing_flags_an_engine_whose_rerun_is_not_its_first_run(monkeypatch, tmp_path):
    class Drifting(_fake_reference()):
        calls = 0

        def ensemble(self, n, seed):
            Drifting.calls += 1
            return super().ensemble(n, seed + Drifting.calls)  # a reseed that does not take

    _install(monkeypatch, {"copasi": Drifting})
    rec = ssa.run_timing(_timing_job(), None, _ctx(tmp_path))
    assert rec["outcome"] == "NOT_REPRODUCIBLE"
    assert not rec["measurements"]["reproducibility"]["copasi"]["passed"]


def test_timing_drops_a_crashing_or_timed_out_reference_and_times_the_rest(
        monkeypatch, tmp_path):
    from bench.results import result_path

    corr = "ssa/m/x/auto/correctness"
    path = result_path(tmp_path, corr)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"outcome": "REFERENCE_FAILED",
                                "run_network_error": "TIMEOUT: no result within the 1800 s budget"}))
    _install(monkeypatch, {"run_network": _fake_reference(sleep=300),
                           "roadrunner": _fake_reference(boom="x"), "copasi": _fake_reference()})
    rec = ssa.run_timing(_timing_job(requires=corr), None,
                         _ctx(tmp_path, results_dir=str(tmp_path)))
    assert set(rec["measurements"]["times"]) == {"bngsim", "copasi"}
    assert rec["run_network_error"].startswith("TIMEOUT in the correctness job")
    assert rec["roadrunner_error"].startswith("RuntimeError")
    assert rec["outcome"] == "PASS"
