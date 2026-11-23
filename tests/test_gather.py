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
