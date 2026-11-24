"""The gathered summary: correctness as agreed/attempted references with any crash
named, and timing as two numbers per engine -- the one-time pre-sim build and the
warm solve with its CI. Rendering only; the per-job result schema is untouched.

The sample is a real x86 result (beta_catenin, 5 engines, E5-2699 v4), trimmed to
the fields ``gather`` reads: COPASI crashed on it while the other three references
agreed, and AMICI's small solve sits behind a large compile."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from bench import gather as g
from bench.jobs import CORRECTNESS, TIMING, Job

DATA = Path(__file__).parent / "data" / "gather"
PROBLEM = "beta_catenin_destruction_complex_barua2013"
ENGINES = "bngsim-vs-run_network-roadrunner-copasi-amici"


def _job(kind: str) -> Job:
    return Job(capability="ode", problem=PROBLEM, engines=ENGINES, setting="auto", kind=kind)


@pytest.fixture
def gathered(tmp_path):
    results = tmp_path / "results"
    jobs = [_job(CORRECTNESS), _job(TIMING)]
    for job in jobs:
        dest = results / f"{job.id}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(DATA / f"beta_catenin_{job.kind}.json", dest)
    reports = tmp_path / "reports"
    out = g.gather(jobs, results, reports, {CORRECTNESS: {}, TIMING: {}})
    return out, (reports / "summary.md").read_text(encoding="utf-8"), reports


def _row(summary: str, *cells: str) -> list[str]:
    """The summary table row whose leading cells are ``cells``, split into cells."""
    for line in summary.splitlines():
        parts = [c.strip() for c in line.strip().strip("|").split("|")]
        if parts[: len(cells)] == list(cells):
            return parts
    raise AssertionError(f"no row starting {cells!r} in:\n{summary}")


def test_sample_job_ids_match_the_committed_records():
    for kind in (CORRECTNESS, TIMING):
        rec = json.loads((DATA / f"beta_catenin_{kind}.json").read_text())
        assert rec["job_id"] == _job(kind).id


def test_a_crashed_reference_reads_partial_with_the_crash_named(gathered):
    _, summary, _ = gathered
    row = _row(summary, PROBLEM, "409")
    # The outcome enum is kept; the agreement label is added beside it.
    assert row[2] == "REFERENCE_FAILED"
    assert row[3] == "PARTIAL 3/4 (copasi: crash)"
    header = _row(summary, "problem", "species", "outcome")
    assert dict(zip(header, row))["vs copasi"] == "crash"
    assert {dict(zip(header, row))[f"vs {n}"] for n in ("amici", "roadrunner", "run_network")} == {"ok"}


def test_timing_shows_presim_beside_sim_for_each_engine(gathered):
    _, summary, _ = gathered
    header = _row(summary, "problem", "species", "engine")
    cols = lambda engine: dict(zip(header, _row(summary, PROBLEM, "409", engine)))  # noqa: E731

    amici = cols("amici")
    assert amici["pre-sim (s)"] == "220.4"          # the SBML->C++ compile...
    assert amici["pre-sim parts (s)"].startswith("codegen 220.4 + load ")
    assert amici["sim median (s)"] == "0.7154"      # ...behind a sub-second solve
    assert amici["sim 95% CI (s)"] == "0.713–0.7256"

    bngsim = cols("bngsim")
    assert bngsim["pre-sim (s)"] == "32.37"
    assert bngsim["pre-sim parts (s)"] == "netgen 21.85 + codegen [cc] 10.51 + load 0.01143"
    assert bngsim["sim median (s)"] == "0.5237"

    assert cols("roadrunner")["pre-sim parts (s)"] == "load 11.51"

    # run_network's load is inside every rep: labelled, not given an invented split.
    rn = cols("run_network")
    assert rn["pre-sim (s)"] == "in sim"
    assert rn["sim median (s)"] == "14.86"
    assert rn["x run_network"] == "1.00"

    # The crashed reference stays visible in the timing table rather than vanishing.
    assert cols("copasi")["sim median (s)"] == "crash"
    assert "Times in seconds" in summary


def test_report_json_carries_the_presim_breakdown(gathered):
    _, _, reports = gathered
    [row] = json.loads((reports / "ode-timing.json").read_text())
    assert row["presim"]["amici"]["codegen_seconds"] == pytest.approx(220.42, abs=0.01)
    assert row["presim"]["bngsim"]["backend"] == "cc"
    assert row["presim"]["run_network"]["role"] == "fresh subprocess"
    assert row["copasi_error"].startswith("KeyError")
    assert "copasi" not in row["times"]


@pytest.mark.parametrize("row, label", [
    ({"verdicts": {"a": {"passed": True}, "b": {"passed": True}}}, "FULL 2/2"),
    ({"verdicts": {"a": {"passed": True}, "b": {"passed": False}}}, "DIFF 1/2 (b: diff)"),
    ({"verdicts": {"a": {"passed": True}, "b": {"passed": False}}, "c_error": "boom"},
     "DIFF 1/3 (b: diff, c: crash)"),
    ({"verdicts": {"a": {"passed": True}}, "c_error": "boom"}, "PARTIAL 1/2 (c: crash)"),
    # A reference that ran out of its per-engine budget is named as a timeout.
    ({"verdicts": {"a": {"passed": True}}, "c_error": "TIMEOUT: no result within 3600 s"},
     "PARTIAL 1/2 (c: timeout)"),
    # The job-level "error" is not a reference crash; nothing attempted reads blank.
    ({"verdicts": {}, "error": "netgen failed"}, ""),
])
def test_reference_agreement_label(row, label):
    assert g.reference_agreement(row) == label


def test_sub_second_times_keep_their_digits():
    assert g._sec(0.001108) == "0.001108"
    assert g._sec(220.42275) == "220.4"
    assert g._sec(None) == ""


# --------------------------------------------------------------------------- #
# SSA: a statistical verdict per comparison, and ensembles as the timed unit
# --------------------------------------------------------------------------- #
SSA_ENGINES = "bngsim-vs-run_network-roadrunner-copasi"


def _ssa_records():
    network = {"species": 4, "reactions": 6}
    corr = {
        "outcome": "REFERENCE_FAILED",
        "measurements": {"network": network, "ensembles": {"bngsim": {"replicates": 10000}}},
        "verdicts": {
            "run_network": {"passed": True, "p_value": 0.41, "z_max": 3.1, "alpha": 1e-3,
                            "replicates": [10000, 10000], "test": "permutation"},
            "copasi": {"passed": True, "p_value": 0.052, "z_max": 3.9, "alpha": 1e-3,
                       "replicates": [10000, 10000], "test": "permutation"},
            "analytic": {"passed": True, "p_value": 0.0031, "p_min": 2.6e-6, "z_max": 2.9,
                         "alpha": 1e-3, "replicates": 10000, "test": "bonferroni"},
        },
        "roadrunner_error": "UNSUPPORTED: 1 reaction(s) consume one species twice",
    }
    timing = {
        "outcome": "PASS",
        "measurements": {
            "network": network, "timing_replicates": 100,
            "netgen_seconds": 0.8,
            "bngsim": {"load_seconds": 0.0013, "codegen_seconds": 0.097,
                       "propensity_backend": "cc"},
            "references": {"copasi": {"load_seconds": 0.02}, "run_network": {}},
            "engine_roles": {"bngsim": "warm, in-process (system under test)",
                             "copasi": "warm, in-process", "run_network": "fresh subprocess"},
            "times": {name: {"median_seconds": s, "ci_low": s * 0.99, "ci_high": s * 1.01}
                      for name, s in (("bngsim", 0.0264), ("copasi", 0.0544),
                                      ("run_network", 0.6879))},
            "times_per_trajectory": {name: {"median_seconds": s}
                                     for name, s in (("bngsim", 0.000264), ("copasi", 0.000544),
                                                     ("run_network", 0.006879))},
            "reproducibility": {n: {"passed": True} for n in ("bngsim", "copasi", "run_network")},
            "repetitions": {"rounds": 20, "warmup_rounds": 1},
        },
        "roadrunner_error": "UNSUPPORTED: 1 reaction(s) consume one species twice",
    }
    return corr, timing


@pytest.fixture
def ssa_gathered(tmp_path):
    results = tmp_path / "results"
    jobs = [Job("ssa", "toy", SSA_ENGINES, "auto", kind) for kind in (CORRECTNESS, TIMING)]
    for job, rec in zip(jobs, _ssa_records()):
        dest = results / f"{job.id}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(rec))
    reports = tmp_path / "reports"
    out = g.gather(jobs, results, reports, {CORRECTNESS: {}, TIMING: {}})
    return out, (reports / "summary.md").read_text(encoding="utf-8"), reports


def test_ssa_correctness_reads_a_p_value_per_comparison(ssa_gathered):
    _, summary, reports = ssa_gathered
    header = _row(summary, "problem", "species", "outcome", "references", "min p")
    row = dict(zip(header, _row(summary, "toy", "4")))
    assert row["outcome"] == "REFERENCE_FAILED"
    assert row["references"] == "PARTIAL 3/4 (roadrunner: n/a)"
    assert row["min p"] == "0.0031" and row["replicates"] == "10000"
    assert row["vs analytic"] == "ok (p 0.0031)"
    assert row["vs run_network"] == "ok (p 0.41)"
    assert row["vs roadrunner"] == "n/a"
    [jrow] = json.loads((reports / "ssa-correctness.json").read_text())
    assert jrow["verdicts"]["copasi"]["p_value"] == 0.052 and jrow["min_p_value"] == 0.0031


def test_ssa_timing_shows_ensemble_and_per_trajectory_times(ssa_gathered):
    _, summary, _ = ssa_gathered
    header = _row(summary, "problem", "species", "engine", "outcome", "pre-sim (s)")
    cols = lambda engine: dict(zip(header, _row(summary, "toy", "4", engine)))  # noqa: E731
    bn = cols("bngsim")
    assert bn["ensemble median (s)"] == "0.0264" and bn["per trajectory (s)"] == "0.000264"
    assert bn["pre-sim parts (s)"] == "netgen 0.8 + codegen [cc] 0.097 + load 0.0013"
    assert cols("run_network")["pre-sim (s)"] == "in sim"
    assert cols("run_network")["x run_network"] == "1.00"
    assert cols("roadrunner")["ensemble median (s)"] == "n/a"
    assert "ensemble* of 100 trajectories" in summary


def test_ode_tables_keep_their_columns(gathered):
    _, summary, _ = gathered
    assert "| sim median (s) | sim 95% CI (s) | x run_network |" in summary
    assert "per trajectory" not in summary and "min p" not in summary


def test_correctness_shows_which_half_of_the_gate_failed_and_the_worst_species(tmp_path):
    # A reference the vendored verdict passed and the per-species check failed
    # (AMICI on Smith 2013, which never gives the first insulin dose), beside one
    # that passed both. Each reference names the half that failed; the row names
    # its worst species, as a fraction of its own peak.
    job = Job(capability="ode", problem="smith", engines="bngsim-vs-a-b", setting="auto",
              kind=CORRECTNESS)
    check = {"passed": False, "n_fail": 10, "n_species": 105, "n_at_floor": 6, "tol": 1e-2,
             "floor": 1e-8, "failing": ["Ins"],
             "worst": {"species": "Ins", "err": 5e5, "peak": 5e5, "allowed": 5e3,
                       "score": 100.0, "err_over_peak": 1.0, "t": 2.805}}
    ok = {**check, "passed": True, "n_fail": 0, "failing": [],
          "worst": {"species": "ROS", "err": 1.2e6, "peak": 1.1e11, "allowed": 1.1e9,
                    "score": 1.1e-3, "err_over_peak": 1.1e-5}}
    rec = {"outcome": "DIFF", "measurements": {"network": {"species": 133}},
           "verdicts": {"a": {"passed": False, "vendored_passed": True, "species_check": check,
                              "max_err_over_peak": 5.2e219},
                        "b": {"passed": True, "vendored_passed": True, "species_check": ok,
                              "max_err_over_peak": 1.1e-5}}}
    path = tmp_path / "results" / f"{job.id}.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(rec))
    out = g.gather([job], tmp_path / "results", tmp_path / "reports", {CORRECTNESS: {}, TIMING: {}})
    [row] = out["ode-correctness"]
    assert row["worst_species"]["species"] == "Ins" and row["worst_species"]["reference"] == "a"
    assert row["verdicts"]["a"]["species_check"]["failing"] == ["Ins"]
    summary = (tmp_path / "reports" / "summary.md").read_text(encoding="utf-8")
    header = _row(summary, "problem", "species", "outcome")
    cols = dict(zip(header, _row(summary, "smith", "133")))
    assert cols["vs a"] == "DIFF (species)" and cols["vs b"] == "ok"
    assert cols["worst species"] == "`Ins` (a) 1.0e+00 of peak"
    assert cols["references"] == "DIFF 1/2 (a: diff)"


# --------------------------------------------------------------------------- #
# Steady state: an agreement per reference, point by point, and the engine-free
# checks of BNGsim's own state beside them
# --------------------------------------------------------------------------- #
def test_steady_state_has_its_own_correctness_table(tmp_path):
    job = Job(capability="steady_state", problem="toggle",
              engines="bngsim-vs-run_network-roadrunner-copasi-amici", setting="newton",
              kind=CORRECTNESS)
    agree = {"passed": True, "max_ratio": 0.031, "worst_point": 2, "worst_species": "u",
             "points_agreed": 3, "points": 3, "max_rel": 2e-6, "per_point": [0.01, 0.02, 0.031]}
    rec = {
        "outcome": "DIFF",
        "measurements": {"network": {"species": 2, "reactions": 4},
                         "points": [{"alpha_2": 1.0}, {"alpha_2": 50.0}, {"alpha_2": 500.0}]},
        "verdicts": {
            "run_network": agree,
            "roadrunner": {**agree, "passed": False, "max_ratio": 12.0, "points_agreed": 2},
            "engine_free": {"passed": True, "evaluated": True,
                            "residual": {"passed": True, "max": 1e-12, "tol": 1e-6,
                                         "per_point": [1e-12] * 3},
                            "conservation": {"passed": True, "max": 0.0, "n_laws": 0}},
        },
        "copasi_error": "OTHER_STATE: at point 1 (alpha_2 = 50) it returned an unstable "
                        "steady state",
        "amici_error": "TIMEOUT: no result within the 600 s per-engine budget",
    }
    path = tmp_path / "results" / f"{job.id}.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(rec))
    out = g.gather([job], tmp_path / "results", tmp_path / "reports", {CORRECTNESS: {}, TIMING: {}})
    [row] = out["steady_state-correctness"]
    assert set(row["verdicts"]) == {"run_network", "roadrunner"}
    assert row["engine_free"]["residual"] == {"passed": True, "max": 1e-12, "tol": 1e-6}
    assert row["max_ratio"] == 12.0 and row["points"] == 3
    summary = (tmp_path / "reports" / "summary.md").read_text(encoding="utf-8")
    header = _row(summary, "problem", "species", "points")
    cols = dict(zip(header, _row(summary, "toggle", "2")))
    assert cols["points"] == "3" and cols["worst ratio"] == "12"
    assert cols["vs run_network"] == "ok" and cols["vs roadrunner"] == "DIFF (2/3 points)"
    assert cols["vs copasi"] == "crash" and cols["vs amici"] == "timeout"
    assert cols["engine-free"] == "ok"
    assert cols["references"] == "DIFF 1/4 (amici: timeout, copasi: crash, roadrunner: diff)"
    # Nothing of the ODE gate: no worst-species column, no vendored/per-species footnote.
    assert "worst species" not in header and "vs engine_free" not in header
    assert "vendored" not in summary and "per-species" not in summary


def test_steady_state_engine_free_cell_names_the_failed_check():
    assert g._engine_free_cell(None) == ""
    assert g._engine_free_cell({"passed": True, "evaluated": False}) == "not evaluated"
    assert g._engine_free_cell({"passed": False, "evaluated": True,
                                "residual": {"passed": True},
                                "conservation": {"passed": False}}) == "FAIL (conservation)"


# --------------------------------------------------------------------------- #
# Network-free: the other setting and the same-seed check beside the verdicts,
# NFsim as the speedup baseline, and each engine's own solve time
# --------------------------------------------------------------------------- #
NF_ENGINES = "bngsim-vs-nfsim-rulemonkey"


def _nf_records():
    corr = {
        "outcome": "PASS",
        "measurements": {
            "ensembles": {"bngsim": {"replicates": 1000}},
            "cross_setting": {"setting": "nf_reject",
                              "verdict": {"passed": True, "p_value": 0.09}},
            "same_seed": {"engine": "rulemonkey", "trajectories": 5, "identical": 5},
        },
        "verdicts": {name: {"passed": True, "p_value": p, "z_max": 2.5, "alpha": 1e-3,
                            "replicates": [1000, 1000], "test": "permutation"}
                     for name, p in (("nfsim", 0.77), ("rulemonkey", 0.61))},
    }
    timing = {
        "outcome": "PASS",
        "measurements": {
            "timing_replicates": 20, "xml_seconds": 0.69,
            "bngsim": {"load_seconds": 0.012, "initialize_seconds": 1.0,
                       "simulate_seconds": 2.0},
            "references": {"nfsim": {}, "rulemonkey": {}},
            "engine_roles": {"bngsim": "warm, in-process (system under test)",
                             "nfsim": "fresh subprocess", "rulemonkey": "fresh subprocess"},
            "times": {name: {"median_seconds": s, "ci_low": s * 0.99, "ci_high": s * 1.01}
                      for name, s in (("bngsim", 3.2), ("nfsim", 15.0), ("rulemonkey", 5.6))},
            "times_per_trajectory": {name: {"median_seconds": s / 20}
                                     for name, s in (("bngsim", 3.2), ("nfsim", 15.0),
                                                     ("rulemonkey", 5.6))},
            "solve_split": {"bngsim": {"solve_seconds": 2.0},
                            "nfsim": {"solve_seconds": 1.6},
                            "rulemonkey": {"solve_seconds": 5.4}},
            "reproducibility": {n: {"passed": True} for n in ("bngsim", "nfsim", "rulemonkey")},
            "repetitions": {"rounds": 5, "warmup_rounds": 1},
        },
    }
    return corr, timing


def test_nf_rows_carry_the_cross_setting_check_and_divide_by_nfsim(tmp_path):
    results = tmp_path / "results"
    jobs = [Job("nf", "toy", NF_ENGINES, "nf_exact", kind) for kind in (CORRECTNESS, TIMING)]
    for job, rec in zip(jobs, _nf_records()):
        dest = results / f"{job.id}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(rec))
    reports = tmp_path / "reports"
    g.gather(jobs, results, reports, {CORRECTNESS: {}, TIMING: {}})
    summary = (reports / "summary.md").read_text(encoding="utf-8")
    [corr] = json.loads((reports / "nf-correctness.json").read_text())
    assert corr["min_p_value"] == 0.61 and corr["setting"] == "nf_exact"
    assert corr["cross_setting"] == {"setting": "nf_reject", "passed": True, "p_value": 0.09}
    assert corr["same_seed"]["identical"] == 5
    header = _row(summary, "problem", "species", "engine", "outcome", "pre-sim (s)")
    cols = lambda engine: dict(zip(header, _row(summary, "toy [nf_exact]", "", engine)))  # noqa: E501,E731
    bn = cols("bngsim")
    assert bn["pre-sim parts (s)"] == "xml 0.69 + load 0.012"
    assert bn["solve per trajectory (s)"] == "0.1" and bn["x nfsim"] == "4.69"
    assert cols("nfsim")["pre-sim (s)"] == "in sim"
    assert cols("rulemonkey")["solve per trajectory (s)"] == "0.27"
    assert "x run_network" not in summary


# --------------------------------------------------------------------------- #
# Forward sensitivities: a column error per reference, a refuted reference named
# as such, and the finite-difference baselines beside the timed solves
# --------------------------------------------------------------------------- #
SENS_ENGINES = "bngsim-vs-amici-run_network_fd-copasi"


def _sens_records():
    network = {"species": 27, "reactions": 96, "observables": 1}
    corr = {
        "outcome": "REFERENCE_FAILED",
        "measurements": {
            "network": network,
            "parameters": {"n": 11, "convention": "independent",
                           "excluded": {"IGF1_cold_conc": "zero value: no relative step"}},
            "configurations": {"bngsim_simultaneous": {"passed": True, "max_error": 1.8e-7}},
            "coverage": {"n_columns": 308, "n_significant": 120, "n_judged": 120, "unjudged": []},
        },
        "verdicts": {
            "amici": {"passed": True, "max_error": 3.5e-7, "n_judged": 308, "n_columns": 308},
            "run_network_fd": {"passed": True, "max_error": 5.1e-7, "n_judged": 300,
                               "n_columns": 308},
            "copasi": {"passed": False, "max_error": 1.06, "n_judged": 308, "n_columns": 308,
                       "n_failed": 108, "refuted_by": ["amici", "run_network_fd"]},
        },
    }
    names = ("bngsim", "bngsim_simultaneous", "bngsim_ode", "run_network", "amici", "copasi")
    secs = (0.0255, 0.0253, 0.0014, 0.0097, 0.0413, 5.75)
    timing = {
        "outcome": "PASS",
        "measurements": {
            "network": network, "netgen_seconds": 0.9,
            "parameters": {"n": 11},
            "bngsim": {"load_seconds": 0.002, "codegen_seconds": 1.9, "codegen_backend": "cc"},
            "configurations": {
                "bngsim_simultaneous": {"load_seconds": 0.001, "codegen_seconds": 0.001,
                                        "codegen_backend": "cc"},
                "bngsim_ode": {"load_seconds": 0.001}},
            "references": {"amici": {"codegen_seconds": 14.2, "load_seconds": 0.05},
                           "copasi": {"load_seconds": 0.3}, "run_network": {}},
            "engine_roles": {"bngsim": "warm, in-process (system under test, staggered corrector)",
                             "run_network": "fresh subprocess"},
            "times": {n: {"median_seconds": s, "ci_low": s * 0.99, "ci_high": s * 1.01}
                      for n, s in zip(names, secs)},
            "fd_baselines": {"bngsim_ode": {"forward_seconds": 12 * 0.0014,
                                            "central_seconds": 22 * 0.0014, "parameters": 11},
                             "run_network": {"forward_seconds": 12 * 0.0097,
                                             "central_seconds": 22 * 0.0097, "parameters": 11}},
            "reproducibility": {n: {"passed": True} for n in names},
            "repetitions": {"rounds": 20, "warmup_rounds": 1},
        },
    }
    return corr, timing


@pytest.fixture
def sens_gathered(tmp_path):
    results = tmp_path / "results"
    jobs = [Job("sens", "igf1r", SENS_ENGINES, "auto", kind) for kind in (CORRECTNESS, TIMING)]
    for job, rec in zip(jobs, _sens_records()):
        dest = results / f"{job.id}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(rec))
    reports = tmp_path / "reports"
    out = g.gather(jobs, results, reports, {CORRECTNESS: {}, TIMING: {}})
    return out, (reports / "summary.md").read_text(encoding="utf-8"), reports


def test_sens_correctness_names_a_refuted_reference_and_the_worst_column_error(sens_gathered):
    _, summary, reports = sens_gathered
    header = _row(summary, "problem", "species", "parameters", "outcome")
    row = dict(zip(header, _row(summary, "igf1r", "27")))
    assert row["parameters"] == "11 independent" and row["outcome"] == "REFERENCE_FAILED"
    assert row["references"] == "PARTIAL 2/3 (copasi: refuted)"
    # The worst error over the references that were not refuted.
    assert row["max column error"] == "5.10e-07"
    assert row["vs amici"] == "ok (3.5e-07)" and row["vs copasi"] == "refuted (1.1)"
    assert row["configurations"] == "agree"
    [jrow] = json.loads((reports / "sens-correctness.json").read_text())
    assert jrow["verdicts"]["copasi"]["refuted_by"] == ["amici", "run_network_fd"]
    assert jrow["parameters_excluded"] == {"IGF1_cold_conc": "zero value: no relative step"}


def test_sens_timing_shows_every_configuration_and_the_fd_baselines(sens_gathered):
    _, summary, _ = sens_gathered
    header = _row(summary, "problem", "species", "engine", "outcome", "pre-sim (s)")
    cols = lambda engine: dict(zip(header, _row(summary, "igf1r", "27", engine)))  # noqa: E731
    amici = cols("amici")
    assert amici["pre-sim parts (s)"] == "codegen 14.2 + load 0.05"
    assert amici["x amici"] == "1.00" and cols("bngsim")["x amici"] == "1.62"
    assert cols("bngsim_simultaneous")["pre-sim parts (s)"] == "codegen [cc] 0.001 + load 0.001"
    assert cols("run_network")["pre-sim (s)"] == "in sim"
    assert "| igf1r | 11 | run_network | 0.0097 | 0.1164 | 0.2134 | 0.0255 |" in summary


def test_a_sensitivity_disagreement_nobody_refutes_reads_diff():
    row = {"verdicts": {"a": {"passed": True}, "b": {"passed": False},
                        "c": {"passed": False, "refuted_by": ["a"]}}}
    assert g.reference_agreement(row) == "DIFF 1/3 (b: diff, c: refuted)"


def test_a_sens_setting_and_a_reference_that_does_not_apply_are_named(tmp_path):
    """A problem's fitted-set setting is a row of its own, and a reference that does
    not apply to a two-phase protocol reads n/a, not a failure."""
    job = Job("sens", "atf2", SENS_ENGINES, "fitted", CORRECTNESS)
    rec = {"outcome": "PASS",
           "measurements": {"network": {"species": 48}, "parameters": {"n": 4,
                                                                         "convention": "fitted"}},
           "verdicts": {"amici": {"passed": True, "max_error": 4e-6, "n_judged": 200},
                        "run_network_fd": {"passed": True, "max_error": 1e-5, "n_judged": 200}},
           "not_applicable": {"copasi": "a two-phase protocol"}}
    dest = tmp_path / "results" / f"{job.id}.json"
    dest.parent.mkdir(parents=True)
    dest.write_text(json.dumps(rec))
    g.gather([job], tmp_path / "results", tmp_path / "reports", {CORRECTNESS: {}, TIMING: {}})
    summary = (tmp_path / "reports" / "summary.md").read_text(encoding="utf-8")
    header = _row(summary, "problem", "species", "parameters", "outcome")
    row = dict(zip(header, _row(summary, "atf2 [fitted]", "48")))
    assert row["parameters"] == "4 fitted" and row["vs copasi"] == "n/a"
    assert row["references"] == "FULL 2/2 (copasi: n/a)"


def test_the_adjoint_gradient_is_timed_beside_bngsims_and_shares_amicis_build(tmp_path):
    """D7: AMICI's adjoint gradient of a log-likelihood beside BNGsim's forward one,
    timing only; each gradient engine's pre-sim cost is the build it runs on."""
    job = Job("sens", "igf1r", SENS_ENGINES, "auto", TIMING)
    _, timing = _sens_records()
    m = timing["measurements"]
    m["times"]["bngsim_gradient"] = {"median_seconds": 0.0261, "ci_low": 0.026, "ci_high": 0.027}
    m["times"]["amici_adjoint"] = {"median_seconds": 0.0302, "ci_low": 0.030, "ci_high": 0.031}
    m["engine_roles"].update(amici_adjoint="warm, in-process (adjoint gradient; timing only)",
                             bngsim_gradient="warm, in-process (forward-sensitivity gradient)")
    m["references"]["amici_adjoint"] = {"shared_build": "amici"}
    m["adjoint"] = {"agreement": {"max_rel_difference": 2.5e-7, "n_parameters": 11}}
    m["reproducibility"].update(bngsim_gradient={"passed": True}, amici_adjoint={"passed": True})
    dest = tmp_path / "results" / f"{job.id}.json"
    dest.parent.mkdir(parents=True)
    dest.write_text(json.dumps(timing))
    g.gather([job], tmp_path / "results", tmp_path / "reports", {CORRECTNESS: {}, TIMING: {}})
    summary = (tmp_path / "reports" / "summary.md").read_text(encoding="utf-8")
    assert "| igf1r | 11 | 0.0261 | 0.0302 | 0.0413 | 2.5e-07 |" in summary
    header = _row(summary, "problem", "species", "engine", "outcome", "pre-sim (s)")
    adj = dict(zip(header, _row(summary, "igf1r", "27", "amici_adjoint")))
    assert adj["pre-sim parts (s)"] == "codegen 14.2 + load 0.05"
