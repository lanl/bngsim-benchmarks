"""The ODE capability's own logic: the engine registry and job generation,
preparing the netgen model, SBML export and column alignment, the run_network
command, reading BioNetGen output, and turning engine results into an outcome."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from bench import jobs, specs
from bench.capabilities import ode

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


# --------------------------------------------------------------------------- #
# Registry and job generation
# --------------------------------------------------------------------------- #
def test_engines_id_names_bngsim_against_every_reference():
    # The id segment is derived from the populated registry, so adding a reference
    # engine re-ids and re-runs the jobs.
    assert ode.engines_id() == "bngsim-vs-" + "-".join(ode.REFERENCE_ENGINES)
    assert "run_network" in ode.REFERENCE_ENGINES and "roadrunner" in ode.REFERENCE_ENGINES


def test_registry_emits_one_correctness_and_one_timing_job(spec_data, write_spec):
    problem = specs.load(write_spec(spec_data))
    emitted = ode.jobs(problem)
    assert [j.kind for j in emitted] == ["correctness", "timing"]
    correctness, timing = emitted
    assert timing.requires == correctness.id
    assert all(ode.engines_id() in j.id for j in emitted)
    # The job timeout is a backstop behind the per-engine budgets, never tighter.
    budget = ode.engine_budget_seconds(problem)
    assert correctness.timeout_seconds >= (len(ode.REFERENCE_ENGINES) + 2) * budget


def test_reference_versions_are_part_of_the_ode_freshness_key(spec_data, write_spec, monkeypatch):
    # Each SBML-reading reference engine's version keys only this capability's
    # freshness (it is in the job definition, hence canonical()), never the global
    # per-kind fingerprint, so a roadrunner/copasi/amici change restales ode
    # results alone. run_network's version rides the global fingerprint, not here.
    monkeypatch.setattr(ode.env, "roadrunner_identity", lambda: {"version": "9.9.9"})
    monkeypatch.setattr(ode.env, "copasi_identity", lambda: {"version": "4.48.311"})
    monkeypatch.setattr(ode.env, "amici_identity", lambda: {"version": "1.1.0"})
    problem = specs.load(write_spec(spec_data))
    correctness, _ = ode.jobs(problem)
    assert correctness.definition["reference_versions"] == {
        "roadrunner": "9.9.9", "copasi": "4.48.311", "amici": "1.1.0",
    }
    assert "reference_versions" in correctness.canonical()


def test_a_net_source_gets_no_jobs(spec_data, write_spec):
    spec_data["source"]["format"] = "net"
    assert ode.jobs(specs.load(write_spec(spec_data))) == []


# --------------------------------------------------------------------------- #
# Netgen rewrite
# --------------------------------------------------------------------------- #
def test_netgen_model_keeps_the_model_and_drops_every_action():
    out = ode.netgen_bngl(MODEL, NETGEN)
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
    out = ode.netgen_bngl(OLD_STYLE, NETGEN)
    assert "max_iter" not in out  # the continuation line went with its action
    assert "simulate" not in out and "writeSBML" not in out
    assert "A() -> 0 k1" in out
    assert out.count("generate_network") == 1


def test_other_setoptions_are_dropped():
    out = ode.netgen_bngl('setOption("SpeciesLabel","HNauty")\n' + MODEL, NETGEN)
    assert "SpeciesLabel" not in out


# --------------------------------------------------------------------------- #
# run_network command and output
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_bin(monkeypatch):
    monkeypatch.setattr(ode.env, "run_network_bin", lambda: Path("/bng/bin/run_network"))
    # These tests pin the native flags BNG2.pl runs; isolate them from any glibc
    # tree in the real cache that would otherwise prepend a loader prefix.
    monkeypatch.setattr(ode.env, "run_network_launch", lambda: [])


def test_run_network_command_matches_what_bng2pl_runs(fake_bin):
    proto = {"t_start": 0, "t_end": 600, "n_steps": 300, "rtol": 1e-8, "atol": 1e-6,
             "sparse": False}
    cmd = ode.run_network_command(Path("m.net"), proto, Path("out/rn"))
    # step size = (600 - 0) / 300 = 2.0
    assert cmd == ["/bng/bin/run_network", "-o", "out/rn", "-p", "cvode", "-a", "1e-06",
                   "-r", "1e-08", "--cdat", "1", "--fdat", "0", "-g", "m.net", "m.net",
                   "2.0", "300"]


def test_sparse_and_start_time_flags(fake_bin):
    proto = {"t_start": 10, "t_end": 30, "n_steps": 4, "rtol": 1e-8, "atol": 1e-8,
             "sparse": True}
    cmd = ode.run_network_command(Path("m.net"), proto, Path("rn"))
    assert "-b" in cmd  # BNG2.pl passes -b for sparse=>1
    assert cmd[cmd.index("-i") + 1] == "10.0"
    assert cmd[-2:] == ["5.0", "4"]  # (30 - 10) / 4 = 5.0


def test_read_dat_parses_header_time_and_values(tmp_path):
    p = tmp_path / "x.cdat"
    p.write_text("#        time    S1    S2\n 0.0  1.0  2.0\n 0.5  1.5  2.5\n")
    t, v, names = ode.read_dat(p)
    np.testing.assert_array_equal(t, [0.0, 0.5])  # parsed decimal literals, exact
    np.testing.assert_array_equal(v, [[1.0, 2.0], [1.5, 2.5]])
    assert names == ["S1", "S2"]


def test_read_dat_reads_a_non_ascii_header_as_utf8(tmp_path):
    # A column name with a non-ASCII byte (an em dash). Read as UTF-8 whatever the
    # locale, so it does not raise under an ASCII locale (issue #9).
    p = tmp_path / "x.cdat"
    p.write_bytes("#  time  S—1\n 0.0  1.0\n".encode("utf-8"))
    _, v, names = ode.read_dat(p)
    np.testing.assert_array_equal(v, [[1.0]])
    assert names == ["S—1"]


def test_netgen_model_keeps_a_non_ascii_comment(tmp_path):
    # An em dash in a comment must survive the netgen rewrite and write out as UTF-8.
    text = "# rate — see paper\n" + MODEL
    out = ode.netgen_bngl(text, NETGEN)
    assert "—" in out
    p = tmp_path / "m.bngl"
    p.write_text(out, encoding="utf-8")
    assert p.read_text(encoding="utf-8") == out


def test_read_dat_without_header_names_columns(tmp_path):
    p = tmp_path / "x.cdat"
    p.write_text("0 1 2 3\n")
    _, v, names = ode.read_dat(p)
    assert v.shape == (1, 3) and names == ["S1", "S2", "S3"]


def test_network_size_counts_entries_not_comments(tmp_path):
    p = tmp_path / "m.net"
    p.write_text(
        "begin parameters\n 1 k1 1.0\nend parameters\n"
        "begin species\n 1 A() 100\n # a comment\n 2 B() 0\nend species\n"
        "begin reactions\n 1 1 2 k1\nend reactions\n"
    )
    assert ode.network_size(p) == {"species": 2, "reactions": 1}


# --------------------------------------------------------------------------- #
# Outcome
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bn_error, ref_errors, passed, outcome",
    [
        (None, {}, True, "PASS"),
        (None, {}, False, "DIFF"),
        ("boom", {}, None, "EXCEPTION"),
        (None, {"run_network": "boom"}, None, "REFERENCE_FAILED"),
        # BNGsim failed and so did every reference -> the model, not an engine.
        ("boom", {"run_network": "boom", "roadrunner": "boom"}, None, "BAD_TEST"),
        # BNGsim failed but only one of two references did -> still BNGsim's fault.
        ("boom", {"run_network": "boom"}, None, "EXCEPTION"),
    ],
)
def test_outcome_attributes_failure_to_the_right_engine(bn_error, ref_errors, passed, outcome):
    assert ode._outcome(bn_error, ref_errors, n_refs=2, passed=passed) == outcome


# --------------------------------------------------------------------------- #
# Verdict and closeness
# --------------------------------------------------------------------------- #
def _decay(n_t=101, scale=1.0):
    t = np.linspace(0, 10, n_t)[:, None]
    return np.hstack([100 * np.exp(-t), 100 * (1 - np.exp(-t))]) * scale


def test_identical_trajectories_pass():
    v = ode._verdict(_decay(), _decay(), atol=1e-8)
    assert v["passed"] is True
    assert v["n_cells"] == 101 * 2


def test_a_ten_percent_error_fails():
    # The verdict measures error against each species' peak. A 10% error over the
    # whole trajectory reaches 10% of the peak, past the relative tolerance (1e-4)
    # and the hard ceiling (5% of peak), so no budget can forgive it.
    bad = _decay()
    bad[:, 0] *= 1.10
    assert ode._verdict(bad, _decay(), atol=1e-8)["passed"] is False


def test_error_where_a_species_has_decayed_is_judged_against_its_peak():
    # 10% of a value below 1% of the peak is under 0.1% of the peak: the verdict
    # forgives it by design (bngsim's differ), and the wrapper must not undo that.
    small = _decay()
    small[60:, 0] *= 1.10  # t >= 6: 100 e^-6 = 0.25, i.e. 0.25% of the peak of 100
    assert ode._verdict(small, _decay(), atol=1e-8)["passed"] is True


def test_extra_trailing_row_is_ignored_not_misaligned():
    longer = np.vstack([_decay(), _decay()[-1:]])
    v = ode._verdict(longer, _decay(), atol=1e-8)
    assert v["passed"] is True
    assert v["shapes"] == {"bngsim": [102, 2], "reference": [101, 2]}


# --------------------------------------------------------------------------- #
# SBML export and column alignment
# --------------------------------------------------------------------------- #
def test_roadrunner_engine_refuses_an_unalignable_species():
    # A BNGsim species with no SBML id is an alignment failure, caught loudly
    # before any simulation -- the one real correctness risk.
    with pytest.raises(RuntimeError, match="cannot align"):
        ode.RoadRunnerODE(Path("unused.xml"), ["A()", "B()"], {"A()": "A"},
                          {"t_start": 0, "t_end": 1, "n_steps": 1, "rtol": 1e-8, "atol": 1e-8})


def test_copasi_engine_refuses_an_unalignable_species():
    # Same alignment guard as roadrunner: no basico import happens before it.
    with pytest.raises(RuntimeError, match="cannot align"):
        ode.CopasiODE(Path("unused.xml"), ["A()", "B()"], {"A()": "A"},
                      {"t_start": 0, "t_end": 1, "n_steps": 1, "rtol": 1e-8, "atol": 1e-8})


class _FakeBasico:
    """Stands in for the ``basico`` module: records the run_time_course call and
    returns a DataFrame whose columns are SBML ids in an arbitrary order, so the
    adapter's reorder-by-id is what makes the output BNGsim-ordered."""

    def __init__(self):
        self.load_calls = []
        self.run_calls = []

    def load_model(self, path):
        self.load_calls.append(path)
        return f"datamodel:{path}"

    def run_time_course(self, *args, **kwargs):
        import pandas as pd

        self.run_calls.append((args, kwargs))
        # Columns deliberately NOT in BNGsim order (B before A); the adapter must
        # reindex to its `selections`.
        df = pd.DataFrame({"idB": [0.0, 1.0, 2.0], "idA": [9.0, 8.0, 7.0]})
        df.index.name = "Time"
        return df


@pytest.fixture
def fake_basico(monkeypatch):
    fake = _FakeBasico()
    monkeypatch.setitem(sys.modules, "basico", fake)
    return fake


def test_copasi_run_selects_amount_columns_in_bngsim_order(fake_basico):
    # Species A() then B(); their SBML ids are idA, idB. The engine must return the
    # idA column first (BNGsim order), not the DataFrame's native order.
    eng = ode.CopasiODE(Path("m.xml"), ["A()", "B()"], {"A()": "idA", "B()": "idB"},
                        {"t_start": 0, "t_end": 10, "n_steps": 2, "rtol": 1e-8, "atol": 1e-8})
    out = eng.run()
    assert out.shape == (3, 2)
    # Column 0 is idA (9,8,7), column 1 is idB (0,1,2) -- reordered to BNGsim order.
    assert list(out[:, 0]) == [9.0, 8.0, 7.0]
    assert list(out[:, 1]) == [0.0, 1.0, 2.0]


def test_copasi_run_asks_for_concentrations_not_particle_numbers(fake_basico):
    # The correctness-critical choice: concentration matches a unit-compartment
    # BNG SBML; use_numbers would be off by Avogadro. Also deterministic LSODA, the
    # fixed interval count, SBML-id columns, and duration = t_end - t_start.
    eng = ode.CopasiODE(Path("m.xml"), ["A()"], {"A()": "idA"},
                        {"t_start": 2, "t_end": 12, "n_steps": 5, "rtol": 1e-7, "atol": 1e-9})
    eng.run()
    (args, kwargs) = fake_basico.run_calls[0]
    assert args == (2.0, 10.0, 5)  # (start, duration, intervals); duration = 12 - 2
    assert kwargs["method"] == "deterministic"
    assert kwargs["use_concentrations"] is True
    assert "use_numbers" not in kwargs
    assert kwargs["use_sbml_id"] is True
    assert kwargs["automatic"] is False
    assert kwargs["r_tol"] == 1e-7 and kwargs["a_tol"] == 1e-9


def test_copasi_warm_is_a_fresh_run(fake_basico):
    # BASICO resets to initial state each call, so warm == run (no reset step of
    # our own); both issue a run_time_course.
    eng = ode.CopasiODE(Path("m.xml"), ["A()"], {"A()": "idA"},
                        {"t_start": 0, "t_end": 1, "n_steps": 1, "rtol": 1e-8, "atol": 1e-8})
    eng.warm()
    eng.warm()
    assert len(fake_basico.run_calls) == 2


def test_copasi_config_reports_amount_output(fake_basico, monkeypatch):
    monkeypatch.setattr(ode.env, "copasi_identity", lambda: {"version": "4.48.311"})
    eng = ode.CopasiODE(Path("m.xml"), ["A()"], {"A()": "idA"},
                        {"t_start": 0, "t_end": 1, "n_steps": 1, "rtol": 1e-8, "atol": 1e-8})
    cfg = eng.config()
    assert cfg["output"] == "amount"
    assert cfg["version"] == "4.48.311"
    assert "load_seconds" in cfg


# --------------------------------------------------------------------------- #
# AmiciODE (fake amici; the real engine compiles a C++ model, exercised by the
# doctor self-test and the integration smoke, not here)
# --------------------------------------------------------------------------- #
class _FakeSolver:
    def __init__(self):
        self.rtol = self.atol = None
        self.sensi = self.lmm = None

    def set_relative_tolerance(self, x):
        self.rtol = x

    def set_absolute_tolerance(self, x):
        self.atol = x

    def set_sensitivity_order(self, s):
        self.sensi = s

    def set_linear_multistep_method(self, m):
        self.lmm = m


class _FakeModel:
    def __init__(self, state_ids, x):
        self._state_ids = state_ids
        self._x = x
        self.timepoints = None
        self.solvers = []

    def set_timepoints(self, ts):
        self.timepoints = list(ts)

    def get_state_ids(self):
        return list(self._state_ids)

    def get_state_ids_solver(self):  # must NOT be used for rdata.x columns
        raise AssertionError("adapter must use get_state_ids(), not get_state_ids_solver()")

    def create_solver(self):
        s = _FakeSolver()
        self.solvers.append(s)
        return s


class _FakeRData:
    def __init__(self, x, status=0):
        self.x = x
        self.status = status


class _FakeAmici(types.ModuleType):
    """A fake top-level ``amici`` module plus its ``amici.sim.sundials`` submodule,
    wired into ``sys.modules`` so the adapter's imports resolve to it. ``rdata.x``
    columns follow ``get_state_ids()`` in a deliberately non-BNGsim order, so the
    adapter's reorder-by-id is what aligns the output."""

    def __init__(self, *, state_ids, x, status=0):
        super().__init__("amici")
        self._state_ids = state_ids
        self._x = np.asarray(x, dtype=float)
        self._status = status
        self.sbml2amici_calls = []
        self.run_calls = []
        self._model = _FakeModel(state_ids, self._x)

        outer = self

        class _Importer:
            def __init__(self, path):
                self.path = path

            def sbml2amici(self, name, outdir, **kwargs):
                outer.sbml2amici_calls.append((name, outdir, kwargs))

        self.SbmlImporter = _Importer

        class _Module:
            def get_model(self):
                return outer._model

        self.import_model_module = lambda name, outdir: _Module()

        # The amici.sim.sundials submodule the adapter's run() imports from.
        sundials = types.ModuleType("amici.sim.sundials")
        sundials.SensitivityOrder = types.SimpleNamespace(none="none")
        sundials.LinearMultistepMethod = types.SimpleNamespace(BDF="BDF")

        def _run_simulation(model, solver):
            outer.run_calls.append((model, solver))
            return _FakeRData(outer._x, status=outer._status)

        sundials.run_simulation = _run_simulation
        self._sundials = sundials


@pytest.fixture
def fake_amici(monkeypatch):
    # state ids B,A (NOT BNGsim order A,B); x columns follow that: col0=B, col1=A.
    x = np.array([[0.0, 9.0], [1.0, 8.0], [2.0, 7.0]])
    fake = _FakeAmici(state_ids=["B", "A"], x=x)
    monkeypatch.setitem(sys.modules, "amici", fake)
    monkeypatch.setitem(sys.modules, "amici.sim", types.ModuleType("amici.sim"))
    monkeypatch.setitem(sys.modules, "amici.sim.sundials", fake._sundials)
    return fake


def test_amici_engine_refuses_an_unalignable_species(fake_amici):
    # A BNGsim species with no SBML id fails before any compile, like roadrunner.
    with pytest.raises(RuntimeError, match="cannot align"):
        ode.AmiciODE(Path("m.xml"), ["A()", "B()"], {"A()": "A"},
                     {"t_start": 0, "t_end": 1, "n_steps": 1, "rtol": 1e-8, "atol": 1e-8})


def test_amici_run_selects_state_columns_in_bngsim_order(fake_amici):
    # BNGsim order A() then B() -> ids A, B. rdata.x columns are [B, A], so the
    # adapter must return column 1 (A: 9,8,7) first, then column 0 (B: 0,1,2).
    eng = ode.AmiciODE(Path("m.xml"), ["A()", "B()"], {"A()": "A", "B()": "B"},
                       {"t_start": 0, "t_end": 10, "n_steps": 2, "rtol": 1e-8, "atol": 1e-8})
    out = eng.run()
    assert out.shape == (3, 2)
    assert list(out[:, 0]) == [9.0, 8.0, 7.0]
    assert list(out[:, 1]) == [0.0, 1.0, 2.0]


def test_amici_compiles_without_sensitivity_code_and_sets_timepoints(fake_amici):
    ode.AmiciODE(Path("m.xml"), ["A()"], {"A()": "A"},
                 {"t_start": 0, "t_end": 10, "n_steps": 2, "rtol": 1e-8, "atol": 1e-8})
    (name, outdir, kwargs) = fake_amici.sbml2amici_calls[0]
    assert kwargs["generate_sensitivity_code"] is False
    # n_steps+1 output points, including both ends.
    assert fake_amici._model.timepoints == [0.0, 5.0, 10.0]


def test_amici_run_uses_a_fresh_solver_configured_for_deterministic_bdf(fake_amici):
    eng = ode.AmiciODE(Path("m.xml"), ["A()"], {"A()": "A"},
                       {"t_start": 0, "t_end": 1, "n_steps": 1, "rtol": 1e-7, "atol": 1e-9})
    eng.run()
    eng.run()
    # A new solver per run (no carried CVODES state -- the 1.2.0 #3512 hazard).
    assert len(fake_amici._model.solvers) == 2
    s = fake_amici._model.solvers[0]
    assert s.rtol == 1e-7 and s.atol == 1e-9
    assert s.sensi == "none"
    assert s.lmm == "BDF"


def test_amici_run_rejects_a_nonzero_status(monkeypatch):
    x = np.array([[1.0, 0.0], [0.5, 0.5]])
    fake = _FakeAmici(state_ids=["A", "B"], x=x, status=1)
    monkeypatch.setitem(sys.modules, "amici", fake)
    monkeypatch.setitem(sys.modules, "amici.sim", types.ModuleType("amici.sim"))
    monkeypatch.setitem(sys.modules, "amici.sim.sundials", fake._sundials)
    eng = ode.AmiciODE(Path("m.xml"), ["A()"], {"A()": "A"},
                       {"t_start": 0, "t_end": 1, "n_steps": 1, "rtol": 1e-8, "atol": 1e-8})
    with pytest.raises(RuntimeError, match="status 1"):
        eng.run()


def test_amici_run_rejects_nan_with_success_status(monkeypatch):
    # AMICI issue #3513: status 0 but a NaN row. A silent NaN would slip past the
    # verdict's nan-skipping metric, so the adapter must reject it.
    x = np.array([[1.0, 0.0], [np.nan, 0.5]])
    fake = _FakeAmici(state_ids=["A", "B"], x=x, status=0)
    monkeypatch.setitem(sys.modules, "amici", fake)
    monkeypatch.setitem(sys.modules, "amici.sim", types.ModuleType("amici.sim"))
    monkeypatch.setitem(sys.modules, "amici.sim.sundials", fake._sundials)
    eng = ode.AmiciODE(Path("m.xml"), ["A()"], {"A()": "A"},
                       {"t_start": 0, "t_end": 1, "n_steps": 1, "rtol": 1e-8, "atol": 1e-8})
    with pytest.raises(RuntimeError, match="non-finite"):
        eng.run()


def test_amici_config_reports_cvodes_bdf_amounts(fake_amici, monkeypatch):
    monkeypatch.setattr(ode.env, "amici_identity", lambda: {"version": "1.1.0"})
    eng = ode.AmiciODE(Path("m.xml"), ["A()"], {"A()": "A"},
                       {"t_start": 0, "t_end": 1, "n_steps": 1, "rtol": 1e-8, "atol": 1e-8})
    cfg = eng.config()
    assert cfg["integrator"] == "CVODES BDF"
    assert cfg["output"] == "amount"
    assert cfg["sensitivities"] == "none"
    assert cfg["version"] == "1.1.0"
    assert "load_seconds" in cfg and "codegen_seconds" in cfg


@pytest.mark.integration
def test_export_sbml_round_trips_bngsim_species_order(tmp_path):
    # BNGsim's column order, the SBML name attribute, and the SBML id line up, so a
    # name map lets a reader request columns in BNGsim order (needs bngsim/libsbml).
    pytest.importorskip("bngsim")
    pytest.importorskip("libsbml")
    net = tmp_path / "m.net"
    net.write_text(
        "begin parameters\n 1 k1 1.0\nend parameters\n"
        "begin species\n 1 Zed() 50\n 2 Apple() 100\n 3 Mid() 0\nend species\n"
        "begin reactions\n 1 2 3 k1\nend reactions\nbegin groups\nend groups\n"
    )
    sbml, species_names, name_to_id = ode.export_sbml(net, tmp_path / "m.xml")
    assert sbml.exists()
    assert species_names == ["Zed()", "Apple()", "Mid()"]
    # Every BNGsim species maps to an SBML id, so the engine can align by name.
    assert all(n in name_to_id for n in species_names)
    assert len(set(name_to_id[n] for n in species_names)) == len(species_names)


# --------------------------------------------------------------------------- #
# sparse_check: run_network's authors' setting, judged against its dense run
# --------------------------------------------------------------------------- #
class _FakeRunNetwork:
    """Stands in for ``RunNetworkODE`` in sparse_check: records the sparse flag it
    was built with and returns fixed values (or raises)."""

    calls: list[bool] = []
    values = None
    error = None

    def __init__(self, net, proto, out_prefix, *, sparse, timeout):
        type(self).calls.append(sparse)
        self._sparse = sparse

    def run(self):
        if type(self).error:
            raise RuntimeError(type(self).error)
        return type(self).values

    @staticmethod
    def trajectory(values):
        return np.array(values)


@pytest.fixture
def fake_run_network(monkeypatch):
    _FakeRunNetwork.calls = []
    _FakeRunNetwork.values = _decay()
    _FakeRunNetwork.error = None
    monkeypatch.setattr(ode, "RunNetworkODE", _FakeRunNetwork)
    return _FakeRunNetwork


def test_sparse_check_skipped_when_the_model_does_not_ask_for_sparse(fake_run_network):
    assert ode.sparse_check(Path("m.net"), {"sparse": False, "atol": 1e-8}, Path("."),
                            _decay(), 60) is None
    assert fake_run_network.calls == []


def test_sparse_run_that_matches_dense_is_trusted(fake_run_network):
    check = ode.sparse_check(Path("m.net"), {"sparse": True, "atol": 1e-8}, Path("."),
                             _decay(), 60)
    assert fake_run_network.calls == [True]  # the authors' setting, -b
    assert check["agrees_with_dense"] is True


def test_sparse_run_that_leaves_the_dense_solution_is_not_trusted(fake_run_network):
    wrong = _decay()
    wrong[:, 1] *= 1.10  # 10% of peak on a moving species: well past the 5% ceiling
    fake_run_network.values = wrong
    check = ode.sparse_check(Path("m.net"), {"sparse": True, "atol": 1e-8}, Path("."),
                             _decay(), 60)
    assert check["agrees_with_dense"] is False


def test_sparse_run_that_fails_is_not_trusted(fake_run_network):
    fake_run_network.error = "CVSpgmr failed"
    check = ode.sparse_check(Path("m.net"), {"sparse": True, "atol": 1e-8}, Path("."),
                             _decay(), 60)
    assert check["agrees_with_dense"] is False and "CVSpgmr" in check["error"]


# --------------------------------------------------------------------------- #
# Timing prep failure
# --------------------------------------------------------------------------- #
def test_timing_prep_failure_is_a_bad_test_row_not_an_exception(monkeypatch):
    # A model that cannot be read/built must be a labelled BAD_TEST row (as in the
    # correctness job), not the worker's generic EXCEPTION or a missing row (issue
    # #9). The failure happens in _prepare, before any engine runs.
    def boom(problem, ctx):
        raise UnicodeDecodeError("ascii", b"\xe2", 0, 1, "ordinal not in range(128)")

    monkeypatch.setattr(ode, "_prepare", boom)
    rec = ode.run_timing(job=None, problem=None, ctx={})
    assert rec["outcome"] == "BAD_TEST"
    assert "UnicodeDecodeError" in rec["error"]


def _timing_job(min_rounds=2, max_rounds=2):
    # A stand-in Job carrying only what run_timing reads: the timing protocol.
    return types.SimpleNamespace(definition={"timing": {
        "warmup_rounds": 1, "min_rounds": min_rounds, "max_rounds": max_rounds,
        "budget_seconds": 0.0}})


def _fake_bngsim():
    class FakeBngsim:
        def __init__(self, net, proto):
            pass

        def run(self):
            return _decay()

        warm = run

        @staticmethod
        def trajectory(values):
            return np.asarray(values)

        def config(self, result=None):
            return {"codegen_backend": "fake", "linear_solver": "dense"}

    return FakeBngsim


def _fake_reference(error=None):
    # An SBML-reading reference like roadrunner/copasi: constructs from prep and
    # runs to a fixed trajectory, or raises in run()/warm() to model a crasher.
    class FakeRef:
        INPUT = "sbml"

        def __init__(self, sbml, species_names, name_to_id, proto):
            pass

        def run(self):
            if error:
                raise RuntimeError(error)
            return _decay()

        warm = run

        @staticmethod
        def trajectory(values):
            return np.asarray(values)

        def config(self):
            return {"engine": "fake"}

    return FakeRef


def test_timing_drops_a_crashing_reference_and_times_the_rest(monkeypatch):
    # The 5-engine shakedown case: one reference (like COPASI on beta_catenin)
    # raises inside run(), not __init__. It must be recorded and dropped from the
    # timed set while BNGsim and the surviving references are still timed -- never
    # crash the whole timing job (as it would without per-engine fault isolation).
    proto = {"t_start": 0, "t_end": 10, "n_steps": 100, "rtol": 1e-8, "atol": 1e-6,
             "sparse": False}
    prep = {"net": Path("m.net"), "sbml": Path("m.xml"), "proto": proto, "info": {},
            "species_names": ["A()", "B()"], "name_to_id": {"A()": "idA", "B()": "idB"}}
    monkeypatch.setattr(ode, "_prepare", lambda problem, ctx: prep)
    monkeypatch.setattr(ode, "BngsimODE", _fake_bngsim())
    monkeypatch.setattr(ode, "REFERENCE_ENGINES", {
        "good": _fake_reference(),
        "bad": _fake_reference(error="KeyError: \"['I'] not in index\""),
    })

    rec = ode.run_timing(job=_timing_job(), problem=None,
                         ctx={"workdir": ".", "subprocess_timeout": 60})

    m = rec["measurements"]
    # BNGsim and the surviving reference are timed; the crasher is not.
    assert set(m["times"]) == {"bngsim", "good"}
    assert "bad" not in m["references"] and "good" in m["references"]
    # The failure is recorded and attributed, exactly as the correctness job does.
    assert rec["bad_error"].startswith("RuntimeError") and "['I']" in rec["bad_error"]
    assert "bad_error" not in m and "good_error" not in rec
    # engine_roles is derived from the surviving engines, so it never lists a
    # dropped engine (nor drifts from the registry).
    assert set(m["engine_roles"]) == {"bngsim", "good"}
    # A dropped reference does not fail the row; the survivors reproduced.
    assert rec["outcome"] == "PASS"
    assert m["reproducibility"]["good"]["passed"] is True
    assert "bad" not in m["reproducibility"]


def test_correctness_prep_failure_is_a_bad_test_row(monkeypatch):
    def boom(problem, ctx):
        raise RuntimeError("BNG2.pl network generation failed")

    monkeypatch.setattr(ode, "_prepare", boom)
    rec = ode.run_correctness(job=None, problem=None, ctx={})
    assert rec["outcome"] == "BAD_TEST"
    assert "network generation failed" in rec["error"]


# --------------------------------------------------------------------------- #
# Closeness and reproducibility
# --------------------------------------------------------------------------- #
def test_max_error_over_peak_is_the_worst_species():
    ref = _decay()
    bad = ref.copy()
    bad[:, 0] *= 1.10  # |a - b| = 0.1 |b| everywhere, so the worst is 0.1 x peak
    bad[:, 1] += 2.0   # a constant offset of 2 on a species whose peak is ~100
    peak1 = np.max(np.abs(ref[:, 1]))
    assert ode.max_err_over_peak(bad, ref) == pytest.approx(max(0.10, 2.0 / peak1), rel=1e-12)


def test_max_error_over_peak_ignores_species_that_never_move_off_zero():
    ref = np.hstack([_decay(), np.zeros((101, 1))])
    assert ode.max_err_over_peak(ref, ref) == 0.0
    assert ode.max_err_over_peak(np.zeros((3, 2)), np.zeros((3, 2))) is None


def test_passing_verdict_still_reports_how_close_the_runs_are():
    near = _decay() * (1 + 1e-6)
    v = ode._verdict(near, _decay(), atol=1e-8)
    assert v["passed"] is True
    assert v["max_err_over_peak"] == pytest.approx(1e-6, rel=1e-6)


def test_reproducibility_passes_when_every_repetition_is_the_first_run():
    anchor = _decay()
    r = ode._reproducibility(anchor, [_decay(), _decay() * (1 + 1e-9)], ode.REPRO_TOL)
    assert r["passed"] is True
    assert r["n_reps"] == 2
    assert r["max_err_over_peak"] < ode.REPRO_TOL


def test_reproducibility_fails_the_repetition_that_solves_another_trajectory():
    # One repetition is off by 10% of a species' peak -- a reset that solved a
    # different problem (issue #8). It is the worst, and it trips the check.
    anchor = _decay()
    good = _decay()
    bad = _decay()
    bad[:, 0] *= 1.10
    r = ode._reproducibility(anchor, [good, bad], ode.REPRO_TOL)
    assert r["passed"] is False
    assert r["worst_rep"] == 1
    assert r["max_err_over_peak"] == pytest.approx(0.10, rel=1e-9)


def test_reproducibility_is_vacuously_true_with_nothing_to_compare():
    # A subprocess comparator with a single kept repetition has no second run to
    # disagree with; the check does not invent a failure.
    r = ode._reproducibility(_decay(), [], ode.REPRO_TOL)
    assert r["passed"] is True and r["n_reps"] == 0 and r["worst_rep"] is None


def test_reproducibility_forgives_a_species_that_never_moves_off_zero():
    anchor = np.hstack([_decay(), np.zeros((101, 1))])
    r = ode._reproducibility(anchor, [anchor.copy()], ode.REPRO_TOL)
    assert r["passed"] is True and r["per_rep"] == [0.0]


# --------------------------------------------------------------------------- #
# Per-engine isolation and wall budget (correctness job)
# --------------------------------------------------------------------------- #
needs_proc = pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs /proc")


def test_run_isolated_returns_the_childs_trajectory_and_config(tmp_path):
    def fn(stage):
        stage("solve")
        return _decay(), {"engine": "x", "cold_seconds": 0.1}

    out = ode.run_isolated(fn, 30, tmp_path / "x", poll=0.01)
    assert "error" not in out
    np.testing.assert_array_equal(out["trajectory"], _decay())
    assert out["config"] == {"engine": "x", "cold_seconds": 0.1}


def test_run_isolated_reports_the_childs_exception(tmp_path):
    def fn(stage):
        raise KeyError("['I'] not in index")

    out = ode.run_isolated(fn, 30, tmp_path / "x", poll=0.01)
    assert out["error"].startswith("KeyError") and "['I']" in out["error"]


def test_run_isolated_attributes_a_hard_crash(tmp_path):
    import os
    import signal

    def fn(stage):
        import faulthandler
        import resource
        faulthandler.disable()  # pytest's handler would dump a traceback first
        # No core file: where the limit allows one, it is the whole process image
        # (over a gigabyte with the engines imported), written to the cwd.
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.kill(os.getpid(), signal.SIGSEGV)  # a segfault in a C++ engine

    out = ode.run_isolated(fn, 30, tmp_path / "x", poll=0.01)
    assert out["error"] == f"crashed: killed by signal {int(signal.SIGSEGV)}"


@needs_proc
def test_run_isolated_times_out_naming_the_stage_and_kills_the_whole_tree(tmp_path):
    import os
    import subprocess
    import time

    pidfile = tmp_path / "grandchild.pid"

    def fn(stage):
        stage("SBML import + C++ compile")
        # A compiler the engine started, like AMICI's build.
        proc = subprocess.Popen(["sleep", "300"])
        pidfile.write_text(str(proc.pid))
        time.sleep(300)

    out = ode.run_isolated(fn, 1.0, tmp_path / "x", poll=0.05)
    assert out["error"].startswith("TIMEOUT")
    assert "1 s per-engine budget" in out["error"]
    assert "SBML import + C++ compile" in out["error"]
    assert out["seconds"] < 30
    grandchild = int(pidfile.read_text())
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            state = Path(f"/proc/{grandchild}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except OSError:
            break  # gone
        if state == "Z":
            break  # killed, awaiting its reaper
        time.sleep(0.05)
    else:
        os.kill(grandchild, 9)
        pytest.fail("the engine's subprocess outlived the budget")


def _correctness_prep():
    proto = {"t_start": 0, "t_end": 10, "n_steps": 100, "rtol": 1e-8, "atol": 1e-6,
             "sparse": False}
    return {"net": Path("m.net"), "sbml": Path("m.xml"), "proto": proto, "info": {},
            "species_names": ["A()", "B()"], "name_to_id": {"A()": "idA", "B()": "idB"}}


def _slow_reference(seconds=300):
    base = _fake_reference()

    class SlowRef(base):
        def run(self):
            import time
            time.sleep(seconds)
            return _decay()

        warm = run

    return SlowRef


def test_correctness_records_a_reference_timeout_and_scores_the_rest(monkeypatch, tmp_path):
    monkeypatch.setattr(ode, "_prepare", lambda problem, ctx: _correctness_prep())
    monkeypatch.setattr(ode, "BngsimODE", _fake_bngsim())
    monkeypatch.setattr(ode, "REFERENCE_ENGINES", {
        "good": _fake_reference(), "slow": _slow_reference()})
    rec = ode.run_correctness(job=None, problem=None, ctx={
        "workdir": str(tmp_path), "subprocess_timeout": 60, "engine_budget_seconds": 1.0})
    # A reference out of time is a reference failure, never a BNGsim one ...
    assert rec["outcome"] == "REFERENCE_FAILED"
    assert rec["slow_error"].startswith("TIMEOUT") and "first solve" in rec["slow_error"]
    # ... and costs nothing else: BNGsim is still scored against the survivor.
    assert rec["verdicts"]["good"]["passed"] is True
    assert "slow" not in rec["verdicts"]
    assert rec["measurements"]["engine_budget_seconds"] == 1.0
    assert "cold_seconds" in rec["measurements"]["references"]["good"]


def test_correctness_bngsim_out_of_time_is_a_timeout(monkeypatch, tmp_path):
    fake = _fake_bngsim()

    class SlowBngsim(fake):
        def run(self):
            import time
            time.sleep(300)

    monkeypatch.setattr(ode, "_prepare", lambda problem, ctx: _correctness_prep())
    monkeypatch.setattr(ode, "BngsimODE", SlowBngsim)
    monkeypatch.setattr(ode, "REFERENCE_ENGINES", {"good": _fake_reference()})
    rec = ode.run_correctness(job=None, problem=None, ctx={
        "workdir": str(tmp_path), "subprocess_timeout": 60, "engine_budget_seconds": 1.0})
    assert rec["outcome"] == "TIMEOUT"
    assert rec["bngsim_error"].startswith("TIMEOUT")


def test_timing_does_not_retry_a_reference_that_timed_out_in_correctness(monkeypatch, tmp_path):
    import json

    from bench.results import result_path

    corr_id = "ode/m/bngsim-vs-good-slow/auto/correctness"
    path = result_path(tmp_path, corr_id)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"outcome": "REFERENCE_FAILED",
                                "slow_error": "TIMEOUT: no result within the 3600 s budget"}))
    monkeypatch.setattr(ode, "_prepare", lambda problem, ctx: _correctness_prep())
    monkeypatch.setattr(ode, "BngsimODE", _fake_bngsim())
    monkeypatch.setattr(ode, "REFERENCE_ENGINES", {
        "good": _fake_reference(), "slow": _slow_reference()})
    job = _timing_job()
    job.requires = corr_id
    rec = ode.run_timing(job=job, problem=None, ctx={
        "workdir": str(tmp_path), "subprocess_timeout": 60, "results_dir": str(tmp_path)})
    assert set(rec["measurements"]["times"]) == {"bngsim", "good"}
    assert rec["slow_error"].startswith("TIMEOUT in the correctness job")
    assert rec["outcome"] == "PASS"


def test_engine_budget_is_capped_and_the_job_timeout_leaves_room_for_every_engine():
    huge = types.SimpleNamespace(size={"reactions": 27457})
    small = types.SimpleNamespace(size={"reactions": 50})
    assert ode.engine_budget_seconds(huge) == ode.ENGINE_BUDGET_CAP_SECONDS
    assert ode.engine_budget_seconds(small) == ode.MIN_TIMEOUT_SECONDS
