"""The ODE capability's own logic: preparing the netgen model, the run_network
command, reading BioNetGen output, and turning engine results into an outcome."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bench.capabilities import ode_bng

MODEL = r"""setOption("NumberPerQuantityUnit",6.0221e23)
begin model
begin parameters
  k1 1.0
end parameters
begin species
  A() 100
  B() 0
end species
begin reaction rules
  A() -> B() k1
end reaction rules
end model

begin actions
  generate_network({overwrite=>1})
  saveConcentrations()
  setParameter("k1",2.0)
  simulate({method=>"ode",t_end=>10,\
    n_steps=>100})
end actions
simulate_ode({t_end=>5,n_steps=>10})
"""

OLD_STYLE = r"""begin parameters
  k1 1.0
end parameters
begin species
  A() 100
end species
begin reaction rules
  A() -> 0 k1
end reaction rules
generate_network({overwrite=>1,\
  max_iter=>3})
simulate_ode({t_end=>10,n_steps=>100})
writeSBML()
"""

NETGEN = "generate_network({overwrite=>1,max_stoich=>{A=>2}})"


def test_netgen_model_keeps_the_model_and_drops_every_action():
    out = ode_bng.netgen_bngl(MODEL, NETGEN)
    for kept in ("begin model", "A() -> B() k1", "end model",
                 'setOption("NumberPerQuantityUnit",6.0221e23)'):
        assert kept in out
    for dropped in ("simulate", "setParameter", "saveConcentrations", "begin actions",
                    "n_steps", "overwrite=>1})\n"):
        assert dropped not in out
    # The spec's call is the only netgen, and it comes last.
    assert out.count("generate_network") == 1
    assert out.rstrip().endswith(NETGEN)


def test_netgen_model_handles_top_level_actions_with_continuations():
    out = ode_bng.netgen_bngl(OLD_STYLE, NETGEN)
    assert "max_iter" not in out  # the continuation line went with its action
    assert "simulate" not in out and "writeSBML" not in out
    assert "A() -> 0 k1" in out
    assert out.count("generate_network") == 1


def test_other_setoptions_are_dropped():
    out = ode_bng.netgen_bngl('setOption("SpeciesLabel","HNauty")\n' + MODEL, NETGEN)
    assert "SpeciesLabel" not in out


@pytest.fixture
def fake_bin(monkeypatch):
    monkeypatch.setattr(ode_bng.env, "run_network_bin", lambda: Path("/bng/bin/run_network"))
    # These tests pin the native flags BNG2.pl runs; isolate them from any glibc
    # tree in the real cache that would otherwise prepend a loader prefix.
    monkeypatch.setattr(ode_bng.env, "run_network_launch", lambda: [])


def test_run_network_command_matches_what_bng2pl_runs(fake_bin):
    proto = {"t_start": 0, "t_end": 600, "n_steps": 300, "rtol": 1e-8, "atol": 1e-6,
             "sparse": False}
    cmd = ode_bng.run_network_command(Path("m.net"), proto, Path("out/rn"))
    # step size = (600 - 0) / 300 = 2.0
    assert cmd == ["/bng/bin/run_network", "-o", "out/rn", "-p", "cvode", "-a", "1e-06",
                   "-r", "1e-08", "--cdat", "1", "--fdat", "0", "-g", "m.net", "m.net",
                   "2.0", "300"]


def test_sparse_and_start_time_flags(fake_bin):
    proto = {"t_start": 10, "t_end": 30, "n_steps": 4, "rtol": 1e-8, "atol": 1e-8,
             "sparse": True}
    cmd = ode_bng.run_network_command(Path("m.net"), proto, Path("rn"))
    assert "-b" in cmd  # BNG2.pl passes -b for sparse=>1
    assert cmd[cmd.index("-i") + 1] == "10.0"
    assert cmd[-2:] == ["5.0", "4"]  # (30 - 10) / 4 = 5.0


def test_read_dat_parses_header_time_and_values(tmp_path):
    p = tmp_path / "x.cdat"
    p.write_text("#        time    S1    S2\n 0.0  1.0  2.0\n 0.5  1.5  2.5\n")
    t, v, names = ode_bng.read_dat(p)
    np.testing.assert_array_equal(t, [0.0, 0.5])  # parsed decimal literals, exact
    np.testing.assert_array_equal(v, [[1.0, 2.0], [1.5, 2.5]])
    assert names == ["S1", "S2"]


def test_read_dat_reads_a_non_ascii_header_as_utf8(tmp_path):
    # A column name with a non-ASCII byte (an em dash). Read as UTF-8 whatever the
    # locale, so it does not raise under an ASCII locale (issue #9).
    p = tmp_path / "x.cdat"
    p.write_bytes("#  time  S—1\n 0.0  1.0\n".encode("utf-8"))
    _, v, names = ode_bng.read_dat(p)
    np.testing.assert_array_equal(v, [[1.0]])
    assert names == ["S—1"]


def test_netgen_model_keeps_a_non_ascii_comment(tmp_path):
    # An em dash in a comment must survive the netgen rewrite and write out as UTF-8.
    text = "# rate — see paper\n" + MODEL
    out = ode_bng.netgen_bngl(text, NETGEN)
    assert "—" in out
    p = tmp_path / "m.bngl"
    p.write_text(out, encoding="utf-8")
    assert p.read_text(encoding="utf-8") == out


def test_read_dat_without_header_names_columns(tmp_path):
    p = tmp_path / "x.cdat"
    p.write_text("0 1 2 3\n")
    _, v, names = ode_bng.read_dat(p)
    assert v.shape == (1, 3) and names == ["S1", "S2", "S3"]


def test_network_size_counts_entries_not_comments(tmp_path):
    p = tmp_path / "m.net"
    p.write_text(
        "begin parameters\n 1 k1 1.0\nend parameters\n"
        "begin species\n 1 A() 100\n # a comment\n 2 B() 0\nend species\n"
        "begin reactions\n 1 1 2 k1\nend reactions\n"
    )
    assert ode_bng.network_size(p) == {"species": 2, "reactions": 1}


@pytest.mark.parametrize(
    "bn_error, rn_error, passed, outcome",
    [
        (None, None, True, "PASS"),
        (None, None, False, "DIFF"),
        ("boom", None, None, "EXCEPTION"),
        (None, "boom", None, "REFERENCE_FAILED"),
        ("boom", "boom", None, "BAD_TEST"),
    ],
)
def test_outcome_attributes_failure_to_the_right_engine(bn_error, rn_error, passed, outcome):
    assert ode_bng._outcome(bn_error, rn_error, passed) == outcome


def _decay(n_t=101, scale=1.0):
    t = np.linspace(0, 10, n_t)[:, None]
    return np.hstack([100 * np.exp(-t), 100 * (1 - np.exp(-t))]) * scale


def test_identical_trajectories_pass():
    v = ode_bng._verdict(_decay(), _decay(), atol=1e-8)
    assert v["passed"] is True
    assert v["n_cells"] == 101 * 2


def test_a_ten_percent_error_fails():
    # The verdict measures error against each species' peak. A 10% error over the
    # whole trajectory reaches 10% of the peak, past the relative tolerance (1e-4)
    # and the hard ceiling (5% of peak), so no budget can forgive it.
    bad = _decay()
    bad[:, 0] *= 1.10
    assert ode_bng._verdict(bad, _decay(), atol=1e-8)["passed"] is False


def test_error_where_a_species_has_decayed_is_judged_against_its_peak():
    # 10% of a value below 1% of the peak is under 0.1% of the peak: the verdict
    # forgives it by design (bngsim's differ), and the wrapper must not undo that.
    small = _decay()
    small[60:, 0] *= 1.10  # t >= 6: 100 e^-6 = 0.25, i.e. 0.25% of the peak of 100
    assert ode_bng._verdict(small, _decay(), atol=1e-8)["passed"] is True


def test_extra_trailing_row_is_ignored_not_misaligned():
    longer = np.vstack([_decay(), _decay()[-1:]])
    v = ode_bng._verdict(longer, _decay(), atol=1e-8)
    assert v["passed"] is True
    assert v["shapes"] == {"bngsim": [102, 2], "run_network": [101, 2]}


def _fake_reference(monkeypatch, values=None, error=None):
    calls = []

    def fake(net, proto, workdir, name, *, sparse, timeout):
        calls.append(sparse)
        if error:
            raise RuntimeError(error)
        return values, ["run_network"], 0.01

    monkeypatch.setattr(ode_bng, "_reference_run", fake)
    return calls


def test_timing_prep_failure_is_a_bad_test_row_not_an_exception(monkeypatch):
    # A model that cannot be read/built must be a labelled BAD_TEST row (as in the
    # correctness job), not the worker's generic EXCEPTION or a missing row (issue
    # #9). The failure happens in _prepare, before any engine runs.
    def boom(problem, ctx):
        raise UnicodeDecodeError("ascii", b"\xe2", 0, 1, "ordinal not in range(128)")

    monkeypatch.setattr(ode_bng, "_prepare", boom)
    rec = ode_bng.run_timing(job=None, problem=None, ctx={})
    assert rec["outcome"] == "BAD_TEST"
    assert "UnicodeDecodeError" in rec["error"]


def test_sparse_check_skipped_when_the_model_does_not_ask_for_sparse(monkeypatch):
    calls = _fake_reference(monkeypatch, values=_decay())
    assert ode_bng.sparse_check(Path("m.net"), {"sparse": False, "atol": 1e-8}, Path("."),
                                _decay(), 60) is None
    assert calls == []


def test_sparse_run_that_matches_dense_is_trusted(monkeypatch):
    calls = _fake_reference(monkeypatch, values=_decay())
    check = ode_bng.sparse_check(Path("m.net"), {"sparse": True, "atol": 1e-8}, Path("."),
                                 _decay(), 60)
    assert calls == [True]  # the authors' setting, -b
    assert check["agrees_with_dense"] is True


def test_sparse_run_that_leaves_the_dense_solution_is_not_trusted(monkeypatch):
    wrong = _decay()
    wrong[:, 1] *= 1.10  # 10% of peak on a moving species: well past the 5% ceiling
    _fake_reference(monkeypatch, values=wrong)
    check = ode_bng.sparse_check(Path("m.net"), {"sparse": True, "atol": 1e-8}, Path("."),
                                 _decay(), 60)
    assert check["agrees_with_dense"] is False


def test_sparse_run_that_fails_is_not_trusted(monkeypatch):
    _fake_reference(monkeypatch, error="CVSpgmr failed")
    check = ode_bng.sparse_check(Path("m.net"), {"sparse": True, "atol": 1e-8}, Path("."),
                                 _decay(), 60)
    assert check["agrees_with_dense"] is False and "CVSpgmr" in check["error"]


def test_max_error_over_peak_is_the_worst_species():
    ref = _decay()
    bad = ref.copy()
    bad[:, 0] *= 1.10  # |a - b| = 0.1 |b| everywhere, so the worst is 0.1 x peak
    bad[:, 1] += 2.0   # a constant offset of 2 on a species whose peak is ~100
    peak1 = np.max(np.abs(ref[:, 1]))
    assert ode_bng.max_err_over_peak(bad, ref) == pytest.approx(max(0.10, 2.0 / peak1), rel=1e-12)


def test_max_error_over_peak_ignores_species_that_never_move_off_zero():
    ref = np.hstack([_decay(), np.zeros((101, 1))])
    assert ode_bng.max_err_over_peak(ref, ref) == 0.0
    assert ode_bng.max_err_over_peak(np.zeros((3, 2)), np.zeros((3, 2))) is None


def test_passing_verdict_still_reports_how_close_the_runs_are():
    near = _decay() * (1 + 1e-6)
    v = ode_bng._verdict(near, _decay(), atol=1e-8)
    assert v["passed"] is True
    assert v["max_err_over_peak"] == pytest.approx(1e-6, rel=1e-6)


def test_reproducibility_passes_when_every_repetition_is_the_first_run():
    anchor = _decay()
    r = ode_bng._reproducibility(anchor, [_decay(), _decay() * (1 + 1e-9)], ode_bng.REPRO_TOL)
    assert r["passed"] is True
    assert r["n_reps"] == 2
    assert r["max_err_over_peak"] < ode_bng.REPRO_TOL


def test_reproducibility_fails_the_repetition_that_solves_another_trajectory():
    # One repetition is off by 10% of a species' peak -- a reset that solved a
    # different problem (issue #8). It is the worst, and it trips the check.
    anchor = _decay()
    good = _decay()
    bad = _decay()
    bad[:, 0] *= 1.10
    r = ode_bng._reproducibility(anchor, [good, bad], ode_bng.REPRO_TOL)
    assert r["passed"] is False
    assert r["worst_rep"] == 1
    assert r["max_err_over_peak"] == pytest.approx(0.10, rel=1e-9)


def test_reproducibility_is_vacuously_true_with_nothing_to_compare():
    # A subprocess comparator with a single kept repetition has no second run to
    # disagree with; the check does not invent a failure.
    r = ode_bng._reproducibility(_decay(), [], ode_bng.REPRO_TOL)
    assert r["passed"] is True and r["n_reps"] == 0 and r["worst_rep"] is None


def test_reproducibility_forgives_a_species_that_never_moves_off_zero():
    anchor = np.hstack([_decay(), np.zeros((101, 1))])
    r = ode_bng._reproducibility(anchor, [anchor.copy()], ode_bng.REPRO_TOL)
    assert r["passed"] is True and r["per_rep"] == [0.0]
