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
