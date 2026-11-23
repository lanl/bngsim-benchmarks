"""The ODE capability's SBML path (the ode-sbml suite): the protocol and its
output times, the state species an SBML model is compared on, each engine
reading the pinned SBML, the run_network phases on a network derived from it,
the derivation's gate, and how a reference that does not apply is recorded.

Engines are faked, as in test_ode.py; the real ones run in the suite itself."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from bench import gather, jobs, specs
from bench.capabilities import ode

SBML_PROTO = {"t_start": 0, "t_end": 10, "n_steps": 100, "rtol": 1e-6, "atol": 1e-9}


@pytest.fixture
def sbml_spec(spec_data):
    spec_data["source"] = dict(spec_data["source"], path="final/BIOMD1/model.xml", format="sbml")
    spec_data["protocol"] = {"ode": dict(SBML_PROTO)}
    return spec_data


def _proto(**kw):
    return {**SBML_PROTO, "output_start": SBML_PROTO["t_start"], **kw}


# --------------------------------------------------------------------------- #
# Specs and jobs
# --------------------------------------------------------------------------- #
def test_an_sbml_protocol_loads_without_netgen(sbml_spec, write_spec):
    p = specs.load(write_spec(sbml_spec))
    assert p.source["format"] == "sbml"


@pytest.mark.parametrize("change, message", [
    ({"netgen": "generate_network({overwrite=>1})"}, "unknown key 'netgen'"),
    ({"sparse": True}, "unknown key 'sparse'"),
    ({"output_start": 10}, "output_start: must be at least t_start and less than t_end"),
    ({"output_start": -1}, "output_start: must be at least t_start and less than t_end"),
    ({"output_start": "4"}, "output_start: must be a number"),
])
def test_a_bad_sbml_protocol_is_refused(sbml_spec, write_spec, change, message):
    sbml_spec["protocol"]["ode"].update(change)
    with pytest.raises(specs.SpecError, match=message):
        specs.load(write_spec(sbml_spec))


def test_a_bngl_protocol_still_needs_netgen_and_has_no_output_start(spec_data, write_spec):
    spec_data["protocol"]["ode"]["output_start"] = 1
    with pytest.raises(specs.SpecError, match="unknown key 'output_start'"):
        specs.load(write_spec(spec_data))
    spec_data["protocol"]["ode"].pop("output_start")
    spec_data["protocol"]["ode"].pop("netgen")
    with pytest.raises(specs.SpecError, match="missing 'netgen'"):
        specs.load(write_spec(spec_data))


def test_ode_sbml_is_a_registered_suite_of_sbml_models():
    assert specs.SUITES["ode-sbml"] == ("ode", ("sbml",))


def test_an_sbml_problem_gets_correctness_and_timing_jobs(sbml_spec, write_spec):
    problem = specs.load(write_spec(sbml_spec))
    correctness, timing = ode.jobs(problem)
    assert timing.requires == correctness.id
    assert correctness.id == f"ode/toy_model/{ode.engines_id()}/auto/correctness"
    d = correctness.definition
    # The output start is made explicit; run_network's sparse option does not apply.
    assert d["protocol"]["output_start"] == 0 and "sparse" not in d["protocol"]
    assert d["input"] == {"format": "sbml", "harness": ode.SBML_HARNESS_VERSION}


def test_a_bngl_job_definition_is_unchanged_by_the_sbml_path(spec_data, write_spec):
    # The ode-net results stay current only if a .net job's definition is exactly
    # what it was: the same keys, and a protocol with no SBML key in it.
    correctness, timing = ode.jobs(specs.load(write_spec(spec_data)))
    assert set(correctness.definition) == {
        "harness", "source_sha256", "protocol", "setting", "reference_versions"}
    assert set(correctness.definition["protocol"]) == {
        "netgen", "t_start", "t_end", "n_steps", "rtol", "atol", "sparse"}
    assert correctness.definition["harness"] == f"ode/{ode.engines_id()}/{ode.HARNESS_VERSION}"


def test_output_times_start_at_the_output_start():
    np.testing.assert_allclose(ode.output_times(_proto()), np.linspace(0, 10, 101))
    np.testing.assert_allclose(ode.output_times(_proto(output_start=4, n_steps=3)),
                               [4.0, 6.0, 8.0, 10.0])


def test_check_times_refuses_another_horizon_or_grid():
    proto = _proto(n_steps=4)
    ode.check_times([0, 2.5, 5, 7.5, 10.000000000001], proto)
    with pytest.raises(RuntimeError, match="not at the protocol's times: 5 times from 0 to 4"):
        ode.check_times([0, 1, 2, 3, 4], proto)  # COPASI's 0..1000 on a 0..12000 protocol
    with pytest.raises(RuntimeError, match="3 times"):
        ode.check_times([0, 5, 10], proto)


# --------------------------------------------------------------------------- #
# State species
# --------------------------------------------------------------------------- #
SBML_DOC = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="m">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="1" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
      <species id="K" compartment="c" initialConcentration="2" hasOnlySubstanceUnits="false"
               boundaryCondition="true" constant="true"/>
      <species id="R" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false"
               boundaryCondition="true" constant="false"/>
      <species id="B" compartment="c" initialConcentration="0" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfRules>
      <assignmentRule variable="R"><math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply><times/><cn>2</cn><ci>A</ci></apply></math></assignmentRule>
    </listOfRules>
  </model>
</sbml>
"""


def test_the_compared_species_are_the_state_species(tmp_path):
    pytest.importorskip("libsbml")
    path = tmp_path / "m.xml"
    path.write_text(SBML_DOC)
    ids, census = ode.sbml_species(path)
    # A constant species and an assignment-rule target are left out, and named.
    assert ids == ["A", "B"]
    assert census["not_compared"] == {"constant": ["K"], "assignment_rule": ["R"]}
    assert census["species"] == 4 and census["compared_species"] == 2


# --------------------------------------------------------------------------- #
# Engines reading the SBML (faked)
# --------------------------------------------------------------------------- #
class _FakeResult:
    def __init__(self, times, names, species):
        self.time = np.asarray(times, dtype=float)
        self.species_names = list(names)
        self.species = np.asarray(species, dtype=float)
        self.solver_stats = {"linear_solver": 0}


class _FakeSim:
    calls: list = []

    def __init__(self, model, method):
        self.model = model
        self.codegen_backend, self.codegen_cache_hit, self.jacobian_strategy = "fake", False, "fd"
        self.last_jacobian_sec = self.last_codegen_sec = 0.0

    def run(self, t_span=None, n_points=None, *, sample_times=None, rtol=None, atol=None):
        type(self).calls.append({"t_span": t_span, "n_points": n_points,
                                 "sample_times": sample_times, "rtol": rtol, "atol": atol})
        times = (np.asarray(sample_times) if sample_times is not None
                 else np.linspace(t_span[0], t_span[1], n_points))
        # Columns in an order of the engine's own, plus one that is not a species.
        names = ["B", "H", "A"]
        return _FakeResult(times, names, np.column_stack([times * 2, times * 0, times]))


@pytest.fixture
def fake_bngsim(monkeypatch):
    _FakeSim.calls = []
    mod = types.ModuleType("bngsim")
    mod.Model = types.SimpleNamespace(from_sbml=lambda path: types.SimpleNamespace(path=path))
    mod.Simulator = _FakeSim
    monkeypatch.setitem(sys.modules, "bngsim", mod)
    return mod


def test_bngsim_reads_the_sbml_with_a_float_atol_and_selects_by_id(fake_bngsim):
    eng = ode.BngsimSBML(Path("m.xml"), ["A", "B"], _proto())
    result = eng.run()
    call = _FakeSim.calls[0]
    assert call["t_span"] == (0.0, 10.0) and call["n_points"] == 101
    assert call["atol"] == 1e-9 and type(call["atol"]) is float  # not ExactAtol
    traj = eng.trajectory(result)
    np.testing.assert_allclose(traj[:, 0], result.time)       # A
    np.testing.assert_allclose(traj[:, 1], 2 * result.time)   # B
    np.testing.assert_allclose(eng.times(result), np.linspace(0, 10, 101))


def test_bngsim_integrates_from_t_start_when_the_output_starts_later(fake_bngsim):
    eng = ode.BngsimSBML(Path("m.xml"), ["A"], _proto(output_start=4, n_steps=3))
    result = eng.run()
    assert _FakeSim.calls[0]["sample_times"] == [0.0, 4.0, 6.0, 8.0, 10.0]
    np.testing.assert_allclose(eng.times(result), [4, 6, 8, 10])  # the t_start row dropped
    assert eng.trajectory(result).shape == (4, 1)


def test_bngsim_without_a_compared_species_fails_loudly(fake_bngsim):
    eng = ode.BngsimSBML(Path("m.xml"), ["A", "Z"], _proto())
    with pytest.raises(RuntimeError, match="no column for 1 compared SBML species"):
        eng.trajectory(eng.run())


class _FakeRR:
    def __init__(self, path):
        self.path = path
        self.timeCourseSelections = []
        self.calls = []
        self.resets = []
        self._integrator = types.SimpleNamespace(values={}, setValue=lambda k, v: None)

    def setIntegrator(self, name):
        pass

    def getIntegrator(self):
        return self._integrator

    def simulate(self, *args, times=None):
        self.calls.append((args, times))
        t = np.asarray(times) if times is not None else np.linspace(args[0], args[1], args[2])
        n = len(self.timeCourseSelections) - 1
        return np.column_stack([t] + [t] * n)

    def reset(self):
        self.resets.append("reset")

    def resetAll(self):
        self.resets.append("resetAll")


@pytest.fixture
def fake_rr(monkeypatch):
    monkeypatch.setattr(ode.env, "import_roadrunner",
                        lambda: types.SimpleNamespace(RoadRunner=_FakeRR))
    monkeypatch.setattr(ode.env, "roadrunner_identity", lambda: {"version": "2.11.0"})


def test_roadrunner_selects_concentrations_by_id_and_resets_everything(fake_rr):
    eng = ode.RoadRunnerSBML(Path("m.xml"), ["A", "B"], _proto())
    assert eng.rr.timeCourseSelections == ["time", "[A]", "[B]"]
    eng.warm()
    # resetAll, not reset: an SBML event may have assigned a global parameter.
    assert eng.rr.resets == ["resetAll"]
    assert eng.rr.calls[0] == ((0.0, 10.0, 101), None)
    assert eng.config()["output"] == "concentration"


def test_roadrunner_integrates_from_t_start_when_the_output_starts_later(fake_rr):
    eng = ode.RoadRunnerSBML(Path("m.xml"), ["A"], _proto(output_start=4, n_steps=3))
    out = eng.run()
    assert eng.rr.calls[0][1] == [0.0, 4.0, 6.0, 8.0, 10.0]
    np.testing.assert_allclose(eng.times(out), [4, 6, 8, 10])
    assert eng.trajectory(out).shape == (4, 1)


class _FakeBasico:
    def __init__(self):
        self.calls = []

    def load_model(self, path):
        return f"dm:{path}"

    def run_time_course(self, *args, **kwargs):
        import pandas as pd

        self.calls.append((args, kwargs))
        t = (kwargs["values"] if "values" in kwargs
             else np.linspace(args[0], args[0] + args[1], args[2] + 1))
        # COPASI leaves a species it holds fixed (here "K") out of its output.
        return pd.DataFrame({"B": np.asarray(t) * 2, "A": t}, index=pd.Index(t, name="Time"))


@pytest.fixture
def fake_basico(monkeypatch):
    fake = _FakeBasico()
    monkeypatch.setitem(sys.modules, "basico", fake)
    return fake


def test_copasi_asks_for_output_times_when_the_output_starts_later(fake_basico):
    eng = ode.CopasiSBML(Path("m.xml"), ["A", "B"], _proto(output_start=4, n_steps=3))
    df = eng.run()
    args, kw = fake_basico.calls[0]
    assert args == () and kw["duration"] == 10.0 and kw["values"] == [4.0, 6.0, 8.0, 10.0]
    assert kw["use_concentrations"] is True and kw["use_sbml_id"] is True
    np.testing.assert_allclose(eng.times(df), [4, 6, 8, 10])
    np.testing.assert_allclose(eng.trajectory(df)[:, 1], 2 * eng.times(df))


def test_copasi_uses_the_net_paths_call_on_its_own_grid(fake_basico):
    eng = ode.CopasiSBML(Path("m.xml"), ["A"], _proto())
    eng.run()
    assert fake_basico.calls[0][0] == (0.0, 10.0, 100)


def test_copasi_leaving_out_a_compared_species_is_its_failure(fake_basico):
    eng = ode.CopasiSBML(Path("m.xml"), ["A", "K"], _proto())
    with pytest.raises(KeyError):
        eng.trajectory(eng.run())


class _FakeAmiciModel:
    def __init__(self, state_ids):
        self._ids = state_ids
        self.t0 = None
        self.timepoints = None

    def set_t0(self, t):
        self.t0 = t

    def set_timepoints(self, ts):
        self.timepoints = list(ts)

    def get_state_ids(self):
        return list(self._ids)


@pytest.fixture
def fake_amici_sbml(monkeypatch):
    model = _FakeAmiciModel(["B", "A"])
    mod = types.ModuleType("amici")
    mod.SbmlImporter = lambda path: types.SimpleNamespace(sbml2amici=lambda *a, **k: None)
    mod.import_model_module = lambda name, outdir: types.SimpleNamespace(get_model=lambda: model)
    monkeypatch.setitem(sys.modules, "amici", mod)
    return model


def test_amici_starts_at_t_start_and_outputs_on_the_grid(fake_amici_sbml, monkeypatch):
    eng = ode.AmiciSBML(Path("w/m.xml"), ["A", "B"], _proto(output_start=4, n_steps=3))
    assert fake_amici_sbml.t0 == 0.0
    assert fake_amici_sbml.timepoints == [4.0, 6.0, 8.0, 10.0]
    assert eng.idx == [1, 0]
    x = np.array([[2.0, 1.0]] * 4)
    monkeypatch.setattr(eng, "_simulate",
                        lambda: (types.SimpleNamespace(ts=[4.0, 6.0, 8.0, 10.0]), x))
    out = eng.run()
    np.testing.assert_allclose(eng.times(out), [4, 6, 8, 10])
    np.testing.assert_allclose(eng.trajectory(out)[0], [1.0, 2.0])


def test_amici_without_a_state_for_a_compared_species_fails(fake_amici_sbml):
    with pytest.raises(RuntimeError, match="AMICI has no state for 1 compared SBML species"):
        ode.AmiciSBML(Path("w/m.xml"), ["A", "R"], _proto())


def test_every_reference_has_an_sbml_engine():
    assert list(ode.SBML_REFERENCE_ENGINES) == list(ode.REFERENCE_ENGINES)


# --------------------------------------------------------------------------- #
# run_network phases
# --------------------------------------------------------------------------- #
def test_one_phase_on_the_protocols_grid_is_run_networks_own_step_and_count():
    grid = np.linspace(0, 10, 101)
    assert ode.sample_phase(0, 10, grid, 10) == pytest.approx({"dt": 0.1, "n": 100})


def test_a_phase_off_the_grid_names_its_sample_times():
    grid = np.linspace(0, 2805, 1001)  # Smith 2013: the first event ends at t=15
    s = ode.sample_phase(0, 15, grid, 2805)
    assert s["times"][:2] == pytest.approx([2.805, 5.61]) and s["times"][-1] == 15.0


def test_a_phase_with_too_few_times_gets_midpoints():
    grid = np.linspace(0, 2805, 1001)
    s = ode.sample_phase(2800, 2805, grid, 2805)
    # The output time inside it, its end, and a midpoint no output time reads.
    assert len(s["times"]) == 3 and s["times"][-1] == 2805.0
    assert s["times"][0] == pytest.approx(2802.195)


def test_a_phase_with_no_output_time_is_one_step():
    grid = np.linspace(400, 700, 44)  # Verma 2016: the event at t=200 is before the output
    assert ode.sample_phase(0, 200, grid, 700) == {"dt": 200.0, "n": 1}


def _two_phases():
    return [{"t_start": 0.0, "t_end": 5.0, "changes": [], "dt": 1.0, "n": 5},
            {"t_start": 5.0, "t_end": 10.0, "changes": [[1, 7.0]], "dt": 1.0, "n": 5}]


def test_stitch_takes_a_boundary_row_from_the_earlier_phase():
    grid = np.linspace(0, 10, 11)
    t1, t2 = np.linspace(0, 5, 6), np.linspace(5, 10, 6)
    out1 = np.column_stack([t1, -t1])
    out2 = np.column_stack([t2, t2 * 0 + 7])
    times, rows = ode.stitch([(t1, out1), (t2, out2)], _two_phases(), grid, 10)
    np.testing.assert_allclose(times, grid)
    assert rows[5].tolist() == [5.0, -5.0]  # pre-change: the end of phase 0
    assert rows[6].tolist() == [6.0, 7.0]
    assert ode.boundary_times(_two_phases(), grid, 10) == [5.0]


def test_stitch_refuses_a_phase_without_the_row():
    grid = np.linspace(0, 10, 11)
    t = np.array([0.0, 5.0])
    with pytest.raises(RuntimeError, match="no output row at t=1"):
        ode.stitch([(t, np.zeros((2, 1))), (t + 5, np.zeros((2, 1)))], _two_phases(), grid, 10)


def test_species_names_read_the_same_in_a_net_and_in_an_action():
    assert ode.species_key("@comp1::Glucose()") == ode.species_key("@comp1:Glucose()")
    assert ode.species_key("$@c::X()") == ("c", "X")
    assert ode.species_key("@comp1::$Glucose()") == ode.species_key("@comp1:Glucose()")
    assert ode.species_key("A()") == (None, "A")


NET = """begin parameters
    1 k 1.0
end parameters
begin species
    1 @c::A() 1.5
    2 @c::B() k  # a value given as a parameter
    3 @c::$EGF_EGFR_2() 2.0
end species
begin reactions
    1 1 2 k
end reactions
begin groups
    1 __sp1                1
    2 __sp2                2
    3 __sp3                3
    4 A                    1
    5 B                    2
    6 _EGF_EGFR_2          3
end groups
"""


def test_the_derived_network_is_aligned_by_the_observable_named_after_each_id():
    lines = NET.splitlines(keepends=True)
    # "_EGF_EGFR_2" is not a BNGL molecule name, so the converter renamed the
    # molecule; the observable it named after the SBML id still finds it.
    assert ode.align_derived(["_EGF_EGFR_2", "A", "B"], lines) == ([2, 0, 1], [])
    assert ode.align_derived(["A", "nope"], lines) == ([0], ["nope"])
    assert ode.net_groups(lines)["B"] == [(1, 1.0)]


def test_a_phase_net_carries_the_state_and_the_change(tmp_path, monkeypatch):
    net = tmp_path / "m.net"
    net.write_text(NET)
    monkeypatch.setattr(ode.env, "run_network_bin", lambda: Path("/bng/bin/run_network"))
    monkeypatch.setattr(ode.env, "run_network_launch", lambda: [])
    seen = []

    def fake_run(cmd, out_prefix, *, timeout):
        seen.append(cmd)
        net_text = Path(cmd[cmd.index("-g") + 1]).read_text()
        start = float(cmd[cmd.index("-i") + 1]) if "-i" in cmd else 0.0
        t = start + np.arange(6.0)
        a = float(ode._NET_SPECIES_RE.match(net_text.splitlines()[4]).group(4))
        values = np.column_stack([a - np.arange(6.0) * 0.1, np.arange(6.0), np.full(6, 2.0)])
        return t, values, ["S1", "S2", "S3"]

    monkeypatch.setattr(ode, "_run_network", fake_run)
    derived = {"net": str(net), "phases": _two_phases(), "columns": [1, 0],
               "boundary_times": [5.0]}
    eng = ode.RunNetworkSBML(derived, _proto(n_steps=10), tmp_path / "rn", timeout=60)
    times, values = eng.run()
    np.testing.assert_allclose(times, np.linspace(0, 10, 11))
    # Phase 0 reads the derived .net as it is; phase 1 a copy set to phase 0's last
    # row, with the change applied over it (B = 7).
    assert seen[0][seen[0].index("-g") + 1] == str(net)
    phase1 = Path(seen[1][seen[1].index("-g") + 1]).read_text()
    assert "@c::A() 1.0" in phase1 and "@c::B() 7.0" in phase1
    assert "-i" in seen[1] and seen[1][seen[1].index("-i") + 1] == "5.0"
    assert values.shape == (11, 2)  # columns in the compared order, B then A
    np.testing.assert_allclose(values[:6, 0], np.arange(6.0))
    assert eng.config()["boundary_times"] == [5.0] and eng.config()["phases"] == 2


def test_a_reference_differing_only_at_a_boundary_row_passes_with_it_named():
    t = np.linspace(0, 10, 11)[:, None]
    bn = np.hstack([100 * np.exp(-t / 5), 100 - 100 * np.exp(-t / 5)])
    ref = bn.copy()
    ref[5] *= 1.5  # the row at the event instant: the other side of the change
    proto = _proto(n_steps=10)
    assert ode._reference_verdict(bn, ref, 1e-9, {}, proto)["passed"] is False
    v = ode._reference_verdict(bn, ref, 1e-9, {"boundary_times": [5.0]}, proto)
    assert v["passed"] is True and v["rows_not_compared"] == [5.0]


# --------------------------------------------------------------------------- #
# The derivation and its gate
# --------------------------------------------------------------------------- #
class _ConversionError(Exception):
    pass


def _fake_convert(monkeypatch, *, text=None, error=None, steps=()):
    mod = types.ModuleType("bngsim")
    mod.ConversionError = _ConversionError
    conv = types.ModuleType("bngsim.convert")

    def sbml_to_bngl(path, out, **kw):
        if error:
            raise _ConversionError(error)
        return types.SimpleNamespace(output_text=text, bngsim_only=[])

    conv.sbml_to_bngl = sbml_to_bngl
    conv.parse_bngl_protocol = lambda t: types.SimpleNamespace(steps=steps)
    mod.convert = conv
    monkeypatch.setitem(sys.modules, "bngsim", mod)
    monkeypatch.setitem(sys.modules, "bngsim.convert", conv)
    return mod


def StateChange(**kw):  # noqa: N802 - named as bngsim names it; the driver reads the type name
    return type("StateChange", (), {"value_expr": None, **kw})()


def test_a_model_the_converter_refuses_is_not_applicable(monkeypatch, tmp_path):
    _fake_convert(monkeypatch, error="65 amount-valued species in a non-unit compartment")
    d = ode.derive_network(Path("m.xml"), ["A"], _proto(), tmp_path, timeout=60)
    assert d["applicable"] is False
    assert "no faithful BNGL form" in d["reason"] and "amount-valued" in d["reason"]


def test_a_derived_protocol_step_the_driver_does_not_run_is_not_applicable(monkeypatch, tmp_path):
    _fake_convert(monkeypatch, text="begin model\nend model\nbegin actions\nend actions\n",
                  steps=[StateChange(kind="set_parameter", target="k", value=2.0)])
    d = ode.derive_network(Path("m.xml"), ["A"], _proto(), tmp_path, timeout=60)
    assert d["applicable"] is False and "set_parameter" in d["reason"]


@pytest.fixture
def derivable(monkeypatch, tmp_path):
    """A conversion whose network BNG2.pl 'builds' (faked), with one event at t=5."""
    steps = [types.SimpleNamespace(kind="simulate", method="ode", t_span=(0.0, 5.0)),
             StateChange(kind="set_concentration", target="@c:B()", value=7.0),
             types.SimpleNamespace(kind="simulate", method="ode", t_span=(5.0, 10.0))]
    mod = _fake_convert(monkeypatch, text="begin model\nend model\nbegin actions\nend actions\n",
                        steps=steps)
    net = tmp_path / "derived.net"
    net.write_text(NET)
    monkeypatch.setattr(ode, "generate_network", lambda *a, **k: (net, 0.5))
    grid = np.linspace(0, 10, 11)
    truth = np.column_stack([np.exp(-grid / 5), 1 - np.exp(-grid / 5)])

    class Sim:
        def __init__(self, model, method):
            pass

        def run(self, **kw):
            return _FakeResult(grid, ["A", "B"], truth)

    mod.Model = types.SimpleNamespace(from_sbml=lambda path: None)
    mod.Simulator = Sim
    return {"truth": truth, "grid": grid}


def test_a_faithful_derivation_gives_run_network_its_network(derivable, monkeypatch, tmp_path):
    truth = (derivable["grid"], derivable["truth"])
    monkeypatch.setattr(ode, "_bngsim_net_phases", lambda net, phases, proto, rtol, atol: truth)
    d = ode.derive_network(Path("m.xml"), ["A", "B"], _proto(n_steps=10), tmp_path, timeout=60)
    assert d["applicable"] is True, d.get("reason")
    assert d["columns"] == [0, 1]
    assert [p["t_start"] for p in d["phases"]] == [0.0, 5.0]
    assert d["phases"][1]["changes"] == [[1, 7.0]] and d["boundary_times"] == [5.0]
    assert d["info"]["faithfulness"]["passed"] is True


def test_a_derived_network_that_drops_an_event_is_unfaithful(derivable, monkeypatch, tmp_path):
    wrong = derivable["truth"].copy()
    wrong[6:, 1] = 0.0  # as if the change at t=5 never happened, after it
    monkeypatch.setattr(ode, "_bngsim_net_phases",
                        lambda net, phases, proto, rtol, atol: (derivable["grid"], wrong))
    d = ode.derive_network(Path("m.xml"), ["A", "B"], _proto(n_steps=10), tmp_path, timeout=60)
    assert d["applicable"] is False and "not faithful" in d["reason"]
    assert d["info"]["faithfulness"]["passed"] is False


def test_a_compared_species_missing_from_the_derived_network_is_not_applicable(
        derivable, tmp_path):
    d = ode.derive_network(Path("m.xml"), ["A", "Q"], _proto(n_steps=10), tmp_path, timeout=60)
    assert d["applicable"] is False and "cannot find 1 compared species" in d["reason"]


# --------------------------------------------------------------------------- #
# Correctness and timing on an SBML model, with run_network not applicable
# --------------------------------------------------------------------------- #
def _decay(n_t=101):
    t = np.linspace(0, 10, n_t)[:, None]
    return np.hstack([100 * np.exp(-t), 100 * (1 - np.exp(-t))])


def _sbml_prep():
    return {"format": "sbml", "sbml": Path("m.xml"), "species_ids": ["A", "B"],
            "proto": _proto(), "info": {}}


class _FakeSbmlEngine:
    INPUT = "sbml"
    shift = None

    def __init__(self, sbml, species_ids, proto):
        self.proto = proto

    def run(self):
        grid = ode.output_times(self.proto)
        if self.shift:
            grid = grid * self.shift
        return grid, _decay()

    warm = run

    @staticmethod
    def trajectory(values):
        return np.asarray(values[1])

    @staticmethod
    def times(values):
        return np.asarray(values[0])

    def config(self, result=None):
        return {"engine": "fake"}


def _fake_sbml_engine(shift=None):
    return type("Fake", (_FakeSbmlEngine,), {"shift": shift})


@pytest.fixture
def sbml_job(monkeypatch):
    monkeypatch.setattr(ode, "_prepare", lambda problem, ctx: _sbml_prep())
    monkeypatch.setattr(ode, "_derive_isolated", lambda prep, workdir, budget, timeout: {
        "applicable": False, "reason": "no faithful BNGL form: refused", "info": {"seconds": 1.0}})
    monkeypatch.setattr(ode, "BngsimSBML", _fake_sbml_engine())
    engines = {"run_network": _fake_sbml_engine(), "good": _fake_sbml_engine(),
               "late": _fake_sbml_engine(shift=1.2)}
    monkeypatch.setattr(ode, "REFERENCE_ENGINES", engines)
    monkeypatch.setattr(ode, "SBML_REFERENCE_ENGINES", engines)


def test_correctness_records_a_reference_that_does_not_apply(sbml_job, tmp_path):
    rec = ode.run_correctness(job=None, problem=None, ctx={
        "workdir": str(tmp_path), "subprocess_timeout": 60, "engine_budget_seconds": 30})
    assert rec["not_applicable"] == {"run_network": "no faithful BNGL form: refused"}
    assert "run_network" not in rec["verdicts"] and "run_network_error" not in rec
    assert rec["measurements"]["derived_network"] == {"seconds": 1.0}
    # A reference whose output is at other times than the protocol's has failed:
    # rows are never scored against BNGsim's when the times differ.
    assert "not at the protocol's times" in rec["late_error"]
    assert rec["verdicts"]["good"]["passed"] is True
    assert rec["outcome"] == "REFERENCE_FAILED"


def test_bad_test_counts_only_the_references_that_apply():
    # BNGsim failed and so did every reference that applies (run_network did not).
    assert ode._outcome("boom", {"roadrunner": "x", "copasi": "x", "amici": "x"}, 3, None) == \
        "BAD_TEST"


def test_timing_skips_a_reference_that_does_not_apply(sbml_job, tmp_path):
    job = types.SimpleNamespace(definition={"timing": {
        "warmup_rounds": 1, "min_rounds": 2, "max_rounds": 2, "budget_seconds": 0.0}})
    rec = ode.run_timing(job=job, problem=types.SimpleNamespace(size={"reactions": 10}),
                         ctx={"workdir": str(tmp_path), "subprocess_timeout": 60})
    assert set(rec["measurements"]["times"]) == {"bngsim", "good"}
    assert rec["not_applicable"] == {"run_network": "no faithful BNGL form: refused"}
    assert "not at the protocol's times" in rec["late_error"]
    assert rec["outcome"] == "PASS"


# --------------------------------------------------------------------------- #
# Gather: a reference that does not apply is shown, not counted
# --------------------------------------------------------------------------- #
def test_agreement_names_a_reference_that_does_not_apply_without_counting_it():
    row = {"verdicts": {"roadrunner": {"passed": True}, "amici": {"passed": True}},
           "copasi_error": "KeyError: ['EGF']",
           "not_applicable": {"run_network": "no faithful BNGL form"}}
    assert gather.reference_agreement(row) == "PARTIAL 2/3 (copasi: crash, run_network: n/a)"


def test_tables_show_a_reference_that_does_not_apply_as_n_a():
    row = {"problem": "m", "outcome": "PASS", "current": True, "network": {"species": 5},
           "verdicts": {"roadrunner": {"passed": True, "max_err_over_peak": 1e-6}},
           "max_err_over_peak": 1e-6, "not_applicable": {"run_network": "refused"}}
    table = "\n".join(gather._correctness_table([row]))
    assert "| n/a |" in table and "FULL 1/1 (run_network: n/a)" in table
    timing = {"problem": "m", "outcome": "PASS", "current": True, "network": {"species": 5},
              "times": {"bngsim": {"median_seconds": 0.1}}, "presim": {},
              "reproducible": {"bngsim": True}, "rounds": 5,
              "not_applicable": {"run_network": "refused"}}
    assert any("| run_network | PASS |  |  | n/a |" in ln
               for ln in gather._timing_table([timing]))


def test_jobs_for_the_committed_ode_sbml_suite_are_all_sbml():
    from conftest import REPO

    problems = specs.load_all(REPO / "problems")
    members = specs.suite_members(problems)["ode-sbml"]
    assert [m.suites["ode-sbml"] for m in members] == list(range(1, 11))
    for m in members:
        assert m.source["repo"] == "sys-bio/temp-biomodels"
        assert m.verification["kind"] == "author_file"
        for job in jobs.jobs_for([m]):
            assert job.definition["input"]["format"] == "sbml"
