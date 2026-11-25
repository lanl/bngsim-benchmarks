"""The ODE correctness gate: bngsim's vendored verdict AND a per-species check
that judges every compared species against its own peak, with an atol floor."""

from __future__ import annotations

import numpy as np
import pytest

from bench import cli, results, specs
from bench.capabilities import ode
from bench.jobs import CORRECTNESS, TIMING
from bench.rescore import rescore_one
from test_trajectories import FP, pair, stored, wide


# --------------------------------------------------------------------------- #
# The per-species check
# --------------------------------------------------------------------------- #
def test_identical_trajectories_pass_the_species_check():
    c = ode.species_check(wide(), wide(), 1e-9)
    assert c["passed"] is True and c["n_fail"] == 0
    assert c["worst"]["score"] == 0.0


def test_a_small_species_six_percent_off_fails_where_the_vendored_verdict_cannot_see_it():
    # The hole the check closes (Nishio 2008: 6% errors on species the vendored
    # verdict never judged): the vendored verdict passes, the species check does not.
    ref = wide()
    bn = ref.copy()
    bn[:, 1] *= 1.06
    assert ode._vendored_verdict(bn, ref, 1e-9)["passed"] is True
    v = ode._verdict(bn, ref, 1e-9, species=["big", "small"], times=np.linspace(0, 10, 101))
    assert v["vendored_passed"] is True
    assert v["species_check"]["passed"] is False
    assert v["passed"] is False
    worst = v["species_check"]["worst"]
    assert worst["species"] == "small"
    assert worst["err_over_peak"] == pytest.approx(0.06 / 1.06, rel=1e-6)
    assert worst["t"] == 0.0  # where the decaying species is largest
    assert v["species_check"]["failing"] == ["small"]


def test_an_engine_that_never_gives_a_dose_on_a_small_species_fails():
    # Smith 2013 under AMICI: the reference holds a species at zero that BNGsim
    # moves (the insulin it was given). Judged against the larger side's peak.
    ref = wide()
    ref[:, 1] = 0.0
    v = ode._verdict(wide(), ref, 1e-9, species=["big", "Ins"])
    assert v["vendored_passed"] is True and v["passed"] is False
    assert v["species_check"]["worst"]["species"] == "Ins"
    assert v["species_check"]["worst"]["err_over_peak"] == pytest.approx(1.0)


def test_round_off_on_a_species_one_engine_holds_at_zero_cannot_fail():
    # Lang 2024: BNGsim holds a species at exactly 0, the reference reports 1e-26.
    # That is 100% of the reference's own peak (max_err_over_peak reads 1.0), and
    # 14 decades below atol: the floor forgives it.
    ref, bn = wide(), wide()
    ref[:, 1] = 1e-26
    bn[:, 1] = 0.0
    v = ode._verdict(bn, ref, 1e-12)
    assert v["max_err_over_peak"] == pytest.approx(1.0)
    assert v["species_check"]["passed"] is True and v["passed"] is True
    assert v["species_check"]["n_at_floor"] == 1


def test_the_floor_is_a_multiple_of_atol():
    atol = 1e-9
    floor = ode.SPECIES_ATOL_FACTOR * atol
    ref = np.zeros((5, 1))
    assert ode.species_check(ref + 0.99 * floor, ref, atol)["passed"] is True
    c = ode.species_check(ref + 1.01 * floor, ref, atol)
    assert c["passed"] is False and c["floor"] == pytest.approx(floor)
    assert c["worst"]["allowed"] == pytest.approx(floor)


def test_the_tolerance_is_a_fraction_of_each_species_own_peak():
    ref = wide(small=1.0)
    for factor, passed in ((1 + 0.9 * ode.SPECIES_TOL, True), (1 + 1.2 * ode.SPECIES_TOL, False)):
        bn = ref.copy()
        bn[:, 1] *= factor
        assert ode.species_check(bn, ref, 1e-12)["passed"] is passed


def test_a_non_finite_value_against_a_finite_one_fails_and_two_nans_agree():
    ref = wide()
    bn = ref.copy()
    bn[3, 1] = np.nan
    assert ode.species_check(bn, ref, 1e-9)["passed"] is False
    both = ref.copy()
    both[3, 1] = np.nan
    assert ode.species_check(both, both.copy(), 1e-9)["passed"] is True
    inf = ref.copy()
    inf[3, 0] = np.inf
    c = ode.species_check(inf, ref, 1e-9)
    assert c["passed"] is False and c["worst"]["index"] == 0


def test_a_vendored_failure_still_fails_the_gate():
    ref = wide()
    bn = ref.copy()
    bn[:, 0] *= 1.10  # 10% on the dominant species: both halves fail
    v = ode._verdict(bn, ref, 1e-9)
    assert v["vendored_passed"] is False and v["passed"] is False


def test_rows_left_out_at_a_boundary_stay_out_of_the_species_check():
    t = np.linspace(0, 10, 11)[:, None]
    bn = np.hstack([100 * np.exp(-t / 5), 100 - 100 * np.exp(-t / 5)])
    ref = bn.copy()
    ref[5] *= 1.5
    proto = {"t_start": 0, "t_end": 10, "n_steps": 10, "rtol": 1e-6, "atol": 1e-9,
             "output_start": 0}
    v = ode._reference_verdict(bn, ref, 1e-9, {"boundary_times": [5.0]}, proto, ["A", "B"])
    assert v["passed"] is True and v["species_check"]["passed"] is True


def test_a_correctness_run_records_both_halves_of_the_gate(monkeypatch, tmp_path):
    from test_ode import _correctness_prep, _fake_bngsim, _fake_reference

    monkeypatch.setattr(ode, "_prepare", lambda problem, ctx: _correctness_prep())
    monkeypatch.setattr(ode, "BngsimODE", _fake_bngsim())
    monkeypatch.setattr(ode, "REFERENCE_ENGINES", {"good": _fake_reference()})
    rec = ode.run_correctness(job=None, problem=None, ctx={
        "workdir": str(tmp_path), "subprocess_timeout": 60, "engine_budget_seconds": 30.0})
    v = rec["verdicts"]["good"]
    assert v["passed"] is True and v["vendored_passed"] is True
    assert v["species_check"]["passed"] is True
    assert v["species_check"]["worst"]["species"] in ("A()", "B()")


# --------------------------------------------------------------------------- #
# The verdict version: a correctness-only part of the key
# --------------------------------------------------------------------------- #
def test_the_verdict_version_keys_correctness_and_not_timing(spec_data, write_spec, monkeypatch):
    problem = specs.load(write_spec(spec_data))
    correctness, timing = ode.jobs(problem)
    assert correctness.definition["verdict"] == ode.VERDICT_VERSION
    assert "verdict" not in timing.definition
    fp = {"bngsim": {"version": "x"}}
    before = {j.kind: (j.canonical(), results.freshness_key(j, fp)) for j in (correctness, timing)}
    monkeypatch.setattr(ode, "VERDICT_VERSION", ode.VERDICT_VERSION + 1)
    c2, t2 = ode.jobs(problem)
    assert results.freshness_key(c2, fp) != before[CORRECTNESS][1]
    assert (t2.canonical(), results.freshness_key(t2, fp)) == before[TIMING]
    assert c2.id == correctness.id and t2.requires == c2.id


# --------------------------------------------------------------------------- #
# Re-scoring a result the vendored verdict alone passed
# --------------------------------------------------------------------------- #
@pytest.fixture
def corr(spec_data, write_spec):
    problem = specs.load(write_spec(spec_data))
    return next(j for j in ode.jobs(problem) if j.kind == CORRECTNESS)


def _before_the_check(job):
    """The job's definition before the per-species check existed."""
    return {k: v for k, v in job.definition.items() if k != "verdict"}


def test_a_result_scored_before_the_per_species_check_is_rescored(tmp_path, corr):
    bn, ref = pair(301, err=0.06)  # the hidden 6% the vendored verdict passes
    stored(tmp_path, corr, bn=bn, ref=ref, definition=_before_the_check(corr))
    by_id, fps = {corr.id: corr}, {CORRECTNESS: FP, TIMING: FP}
    assert results.status(corr, tmp_path, fps, by_id).state == "stale"
    r = rescore_one(corr, tmp_path, FP)
    assert (r.state, r.before, r.after) == ("rescored", "PASS", "DIFF")
    rec = results.read_result(tmp_path, corr)
    v = rec["verdicts"]["run_network"]
    assert v["vendored_passed"] is True and v["species_check"]["worst"]["species"] == "small"
    assert rec["rescored"][-1]["verdict"] == ode.VERDICT_VERSION
    st = results.status(corr, tmp_path, fps, by_id)
    assert (st.state, st.outcome) == ("fresh", "DIFF")


def test_a_reference_failed_record_hiding_a_disagreement_is_rescored_as_a_diff(tmp_path, corr):
    # Verdict 1 scored a job REFERENCE_FAILED when any reference failed, even with a
    # survivor disagreeing (Winter 2017); verdict 2 makes that DIFF, by re-scoring.
    bn, ref = pair(301, err=0.06)
    stored(tmp_path, corr, bn=bn, ref=ref, outcome="REFERENCE_FAILED",
           definition={**corr.definition, "verdict": 1})
    rec = results.read_result(tmp_path, corr)
    rec["amici_error"] = "AttributeError: 'Compartment' object has no attribute 'getValue'"
    results.write_result(tmp_path, corr, rec)
    r = rescore_one(corr, tmp_path, FP)
    assert (r.state, r.before, r.after) == ("rescored", "REFERENCE_FAILED", "DIFF")
    after = results.read_result(tmp_path, corr)
    assert after["amici_error"].startswith("AttributeError")
    assert after["rescored"][-1]["verdict"] == ode.VERDICT_VERSION >= 2


def test_a_dry_run_reports_the_new_outcome_and_writes_nothing(tmp_path, corr):
    bn, ref = pair(301, err=0.06)
    before = stored(tmp_path, corr, bn=bn, ref=ref, definition=_before_the_check(corr))
    r = rescore_one(corr, tmp_path, FP, dry_run=True)
    assert (r.state, r.after) == ("rescored", "DIFF")
    assert results.read_result(tmp_path, corr)["key"] == before["key"]


def test_bench_rescore_names_an_outcome_the_check_changed(tmp_path, spec_data, write_spec,
                                                          monkeypatch, capsys):
    problems = tmp_path / "problems"
    problem = specs.load(write_spec(spec_data, directory=problems))
    job = next(j for j in ode.jobs(problem) if j.kind == CORRECTNESS)
    bn, ref = pair(301, err=0.06)
    stored(tmp_path / "results", job, bn=bn, ref=ref, definition=_before_the_check(job))
    monkeypatch.setattr("bench.results.fingerprints_now", lambda: {CORRECTNESS: FP, TIMING: FP})
    assert cli.main(["rescore", "--problems", str(problems), "--results",
                     str(tmp_path / "results")]) == 0
    out = capsys.readouterr().out
    assert f"rescored         {job.id}  PASS -> DIFF" in out
    assert "1 outcomes changed" in out
