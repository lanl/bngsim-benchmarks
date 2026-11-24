"""The steady-state capability's SBML path (the ss-sbml suite): the protocol's SBML
keys and points, the jobs, what the harness reads from the SBML itself (doses,
stoichiometry, conservation laws), each point's derived network, the SBML
verdict's refinements of the engine-free checks, and how a reference that does
not apply is recorded and which reference then decides the branch.

Engines are faked, as in test_steady.py; the real ones run in the suite itself."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from bench import specs
from bench.capabilities import steady, steady_sbml

SBML_PROTO = {"t_end": 100, "rtol": 1.0e-8, "atol": 1.0e-10}


@pytest.fixture
def sbml_spec(spec_data):
    spec_data["source"] = dict(spec_data["source"], path="final/BIOMD1/model.xml", format="sbml")
    spec_data["capabilities"] = ["steady_state"]
    spec_data["protocol"] = {"steady_state": dict(SBML_PROTO)}
    spec_data["size"] = {"species": 4, "reactions": 2}
    return spec_data


# A + B <-> C in a compartment of size ``size``, with a constant species E that no
# reaction changes, and kr2 = 2 kr set by an assignment rule.
def binding_sbml(size: float = 1.0) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
  <model id="binding">
    <listOfCompartments><compartment id="c" size="{size}"/></listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialConcentration="2"/>
      <species id="B" compartment="c" initialConcentration="1"/>
      <species id="C" compartment="c" initialConcentration="0"/>
      <species id="E" compartment="c" initialConcentration="1" constant="true"
               boundaryCondition="true"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="kf" value="1"/>
      <parameter id="kr" value="1"/>
      <parameter id="kr2" value="2" constant="false"/>
    </listOfParameters>
    <listOfRules>
      <assignmentRule variable="kr2">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><cn>2</cn><ci>kr</ci></apply></math>
      </assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="bind" reversible="false">
        <listOfReactants><speciesReference species="A"/><speciesReference species="B"/>
        </listOfReactants>
        <listOfProducts><speciesReference species="C"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>c</ci><ci>kf</ci><ci>A</ci><ci>B</ci></apply></math>
        </kineticLaw>
      </reaction>
      <reaction id="unbind" reversible="false">
        <listOfReactants><speciesReference species="C"/></listOfReactants>
        <listOfProducts><speciesReference species="A"/><speciesReference species="B"/>
        </listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>c</ci><ci>kr</ci><ci>C</ci></apply></math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""


# The network bngsim.convert.sbml_to_bngl and BNG2.pl derive from it: each reaction
# a signed flux into each species it changes, from nothing.
DERIVED = """# Created by BioNetGen 2.9.3
begin parameters
    1 c               1.0  # Constant
    2 kf              1.0  # Constant
    3 kr              1.0  # Constant
end parameters
begin functions
    1 v1() kf*A*B
    2 v2() kr*C
    3 __flux0_0() -1.0*v1
    4 __flux0_1() -1.0*v1
    5 __flux0_2() 1.0*v1
    6 __flux1_0() 1.0*v2
    7 __flux1_1() 1.0*v2
    8 __flux1_2() -1.0*v2
end functions
begin species
    1 @comp1::A() 2.0
    2 @comp1::B() 1.0
    3 @comp1::C() 0.0
    4 @comp1::$E() 1.0
end species
begin reactions
    1 0 1 1*__flux0_0 #_R1
    2 0 2 1*__flux0_1 #_R2
    3 0 3 1*__flux0_2 #_R3
    4 0 1 1*__flux1_0 #_R4
    5 0 2 1*__flux1_1 #_R5
    6 0 3 1*__flux1_2 #_R6
end reactions
begin groups
    1 A                    1
    2 B                    2
    3 C                    3
    4 E                    4
end groups
"""
NET_NAMES = ["@comp1::A()", "@comp1::B()", "@comp1::C()", "@comp1::$E()"]
IDS = ["A", "B", "C"]  # the state species: E is constant


def binding_state(kr: float, a0: float = 2.0, b0: float = 1.0) -> list[float]:
    s = a0 + b0 + kr
    c = (s - math.sqrt(s * s - 4 * a0 * b0)) / 2
    return [a0 - c, b0 - c, c]


@pytest.fixture
def sbml_file(tmp_path):
    path = tmp_path / "binding.xml"
    path.write_text(binding_sbml(), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Specs, points and jobs
# --------------------------------------------------------------------------- #
def test_an_sbml_steady_state_protocol_has_no_netgen(sbml_spec, write_spec):
    assert specs.load(write_spec(sbml_spec)).source["format"] == "sbml"
    sbml_spec["protocol"]["steady_state"]["netgen"] = "generate_network({overwrite=>1})"
    with pytest.raises(specs.SpecError, match="unknown key 'netgen'"):
        specs.load(write_spec(sbml_spec))


@pytest.mark.parametrize("change, message", [
    ({"species": {"A": -1}}, "species.A: must be a non-negative number"),
    ({"species": ["A"]}, "species: a mapping"),
    ({"species": {"kf": 1}, "parameters": {"kf": 2}}, "'kf' is also set in parameters"),
    ({"scan": {"species": "A", "parameter": "kf", "values": [1]}}, "not both"),
    ({"scan": {"species": "A", "values": [1]}, "species": {"A": 2}}, "also set in parameters"),
])
def test_a_bad_sbml_steady_state_protocol_is_refused(sbml_spec, write_spec, change, message):
    sbml_spec["protocol"]["steady_state"].update(change)
    with pytest.raises(specs.SpecError, match=message):
        specs.load(write_spec(sbml_spec))


def test_a_bngl_protocol_takes_no_species_doses(spec_data, write_spec):
    spec_data["capabilities"] = ["steady_state"]
    spec_data["protocol"] = {"steady_state": {
        "netgen": "generate_network({overwrite=>1})", **SBML_PROTO,
        "scan": {"species": "A", "values": [1, 2]}}}
    with pytest.raises(specs.SpecError, match="unknown key 'species'"):
        specs.load(write_spec(spec_data))
    spec_data["protocol"]["steady_state"].pop("scan")
    spec_data["protocol"]["steady_state"]["species"] = {"A": 1}
    with pytest.raises(specs.SpecError, match="unknown key 'species'"):
        specs.load(write_spec(spec_data))


def test_points_set_the_species_and_the_scanned_species():
    proto = dict(SBML_PROTO, parameters={"kf": 2}, species={"E": 3},
                 scan={"species": "A", "values": [1, 4]})
    assert steady.points(proto) == [{"kf": 2.0, "E": 3.0, "A": 1.0},
                                    {"kf": 2.0, "E": 3.0, "A": 4.0}]
    assert steady.scan_target({"species": "A"}) == "A"
    assert steady.scan_target({"parameter": "kf"}) == "kf"


def test_ss_sbml_is_a_registered_suite_of_sbml_models():
    assert specs.SUITES["ss-sbml"] == ("steady_state", ("sbml",))


def test_an_sbml_problem_gets_jobs_with_the_sbml_definition(sbml_spec, write_spec):
    correctness, timing = steady.jobs(specs.load(write_spec(sbml_spec)))
    assert correctness.id == f"steady_state/toy_model/{steady.engines_id()}/newton/correctness"
    assert timing.requires == correctness.id
    for job in (correctness, timing):
        d = job.definition
        assert d["input"] == {"format": "sbml", "harness": steady_sbml.SBML_HARNESS_VERSION}
        assert d["setting"]["bngsim"] == steady_sbml.BNGSIM_SETTINGS
        assert d["reference_settings"] == steady_sbml.REFERENCE_SETTINGS
    assert correctness.definition["verdict"] == steady_sbml.VERDICT
    assert "verdict" not in timing.definition


def test_a_bngl_job_definition_is_unchanged_by_the_sbml_path(spec_data, write_spec):
    spec_data["capabilities"] = ["steady_state"]
    spec_data["protocol"] = {"steady_state": {"netgen": "generate_network({overwrite=>1})",
                                              **SBML_PROTO}}
    correctness, _ = steady.jobs(specs.load(write_spec(spec_data)))
    d = correctness.definition
    assert "input" not in d and d["verdict"] == steady.VERDICT
    assert d["setting"]["bngsim"] == steady.BNGSIM_SETTINGS
    assert d["reference_settings"] == steady.REFERENCE_SETTINGS
    assert "species" not in d["protocol"]


def test_the_committed_ss_sbml_suite_has_ten_sbml_members():
    from conftest import REPO

    members = specs.suite_members(specs.load_all(REPO / "problems"))["ss-sbml"]
    assert len(members) == 10
    for p in members:
        assert p.source["format"] == "sbml" and "steady_state" in p.capabilities
        assert len(steady.jobs(p)) == 2


# --------------------------------------------------------------------------- #
# What the harness reads from the SBML
# --------------------------------------------------------------------------- #
def test_the_structure_reads_species_parameters_and_stoichiometry(sbml_file):
    st = steady_sbml.read_structure(sbml_file)
    assert st["species"]["E"]["constant"] and not st["species"]["A"]["boundary"]
    assert st["parameters"]["kr2"] == {"constant": False, "rule": "assignment"}
    assert st["reactions"][0]["terms"] == {"A": -1.0, "B": -1.0, "C": 1.0}
    assert st["rate_rules"] == []


def test_doses_are_parameters_or_species_and_nothing_else(sbml_file):
    st = steady_sbml.read_structure(sbml_file)
    proto = dict(SBML_PROTO, parameters={"kf": 1}, species={"E": 2},
                 scan={"species": "A", "values": [1]})
    assert steady_sbml.dose_kinds(proto, st) == {"kf": "parameter", "E": "species",
                                                 "A": "species"}
    with pytest.raises(ValueError, match="no global parameter 'A'.*scan.species"):
        steady_sbml.dose_kinds(dict(SBML_PROTO, parameters={"A": 1}), st)
    with pytest.raises(ValueError, match="rule sets it"):
        steady_sbml.dose_kinds(dict(SBML_PROTO, parameters={"kr2": 1}), st)
    with pytest.raises(ValueError, match="no species 'Z'"):
        steady_sbml.dose_kinds(dict(SBML_PROTO, species={"Z": 1}), st)


def test_the_laws_come_from_the_sbml_weighed_by_compartment_size(tmp_path):
    for size in (1.0, 2.0):
        path = tmp_path / f"b{size}.xml"
        path.write_text(binding_sbml(size), encoding="utf-8")
        laws, moving, unplaced = steady_sbml.structure_subspaces(
            NET_NAMES, steady_sbml.read_structure(path))
        assert unplaced == [] and laws.shape == (3, 4) and moving.shape == (4, 1)
        # Every law holds along the one direction the reaction moves the state in,
        # A + B -> C, and E is a law of its own.
        assert np.allclose(laws @ np.array([-1.0, -1.0, 1.0, 0.0]), 0.0)
        assert np.linalg.matrix_rank(laws[:, :3]) == 2
        assert np.allclose(np.abs(moving[:, 0]), np.array([1, 1, 1, 0]) / math.sqrt(3))
        # In amounts: a law's coefficients carry the compartment's size.
        assert np.max(np.abs(laws)) == pytest.approx(size, rel=1e-9) or size == 1.0


def test_each_point_network_carries_its_parameter_and_species_doses(tmp_path):
    pts = [{"kr": 0.5, "A": 4.0}, {"kr": 2.0, "A": 1.0}]
    doses = {"kr": "parameter", "A": "species"}
    paths, nets = steady_sbml.point_networks(DERIVED, pts, doses, tmp_path)
    assert [n.initial[0] for n in nets] == [4.0, 1.0]
    assert [n.parameters["kr"] for n in nets] == [0.5, 2.0]
    assert "@comp1::A() 4.0" in paths[0].read_text()
    with pytest.raises(ValueError, match="no species 'Z'"):
        steady_sbml.point_networks(DERIVED, [{"Z": 1.0}], {"Z": "species"}, tmp_path)
    with pytest.raises(ValueError, match="no parameter 'kz'"):
        steady_sbml.point_networks(DERIVED, [{"kz": 1.0}], {"kz": "parameter"}, tmp_path)


# --------------------------------------------------------------------------- #
# The engine-free checks on an SBML model
# --------------------------------------------------------------------------- #
SCAN = [{"kr": 0.5}, {"kr": 1.0}, {"kr": 2.0}]


def sbml_prep(tmp_path, sbml_file, pts=SCAN, text=DERIVED, **extra):
    doses = {k: "parameter" for k in pts[0]}
    paths, nets = steady_sbml.point_networks(text, pts, doses, tmp_path / "points")
    laws, moving, _ = steady_sbml.structure_subspaces(
        NET_NAMES, steady_sbml.read_structure(sbml_file))
    prep = {"format": "sbml", "sbml": sbml_file, "species_names": IDS, "species_ids": IDS,
            "points": pts,
            "info": {}, "sinks": [], "point_networks": nets, "point_nets": paths,
            "x0": np.array([n.initial for n in nets]), "net_columns": [0, 1, 2],
            "net_species_names": NET_NAMES, "laws": laws, "law_species": [0, 1, 2, 3],
            "moving_basis": moving, "residual_floor_initial": True,
            "conservation_atol": 1e-9, "doses": doses,
            "proto": dict(SBML_PROTO, n_steps=20, parameters={})}
    prep.update(extra)
    return prep


def test_the_steady_state_of_an_sbml_model_passes_with_its_constant_species_filled_in(
        tmp_path, sbml_file):
    prep = sbml_prep(tmp_path, sbml_file)
    states = np.array([binding_state(p["kr"]) for p in SCAN])
    check = steady.engine_free(states, prep)
    assert check["passed"], check
    assert check["residual"]["max"] < 1e-12 and check["conservation"]["max"] < 1e-12
    # Off its laws (a dose of A0 not taken), it fails conservation.
    off = states.copy()
    off[1] = binding_state(1.0, a0=3.0)
    bad = steady.engine_free(off, prep)
    assert bad["residual"]["passed"] and not bad["conservation"]["passed"]
    assert bad["conservation"]["worst_point"] == 1


def test_the_residual_floor_follows_the_initial_flux_so_the_origin_is_at_rest(
        tmp_path, sbml_file):
    # Every species at round-off from zero: a rest point of the network. Its own
    # largest flux is round-off too, so without the initial state's flux as a
    # floor its net rate is "all" of its gross flux.
    pts = [{"kf": 1.0}]
    text = DERIVED.replace("@comp1::A() 2.0", "@comp1::A() 0.0") \
                  .replace("@comp1::B() 1.0", "@comp1::B() 0.0") \
                  .replace("@comp1::C() 0.0", "@comp1::C() 1.0")
    prep = sbml_prep(tmp_path, sbml_file, pts=pts, text=text)
    x = np.array([[1e-17, 1e-17, 2e-17]])  # not on C's law, but the residual is the point
    with_floor = steady.engine_free(x, prep)["residual"]
    without = steady.engine_free(x, {**prep, "residual_floor_initial": False})["residual"]
    assert with_floor["passed"] and not without["passed"]


def test_a_rate_ruled_species_is_left_out_of_the_residual(tmp_path, sbml_file):
    prep = sbml_prep(tmp_path, sbml_file, residual_skip=[0, 1, 2, 3])
    check = steady.engine_free(np.array([binding_state(p["kr"]) for p in SCAN]), prep)
    assert check["passed"] and not check["residual"]["evaluated"]
    assert "no species has a gross flux" in check["residual"]["reason"]


def test_a_law_whose_total_is_zero_passes_within_the_absolute_tolerance(tmp_path, sbml_file):
    text = DERIVED.replace("@comp1::$E() 1.0", "@comp1::$E() 0.0")
    prep = sbml_prep(tmp_path, sbml_file, text=text, net_columns=[0, 1, 2, 3],
                     species_names=IDS + ["E"])
    states = np.array([binding_state(p["kr"]) + [-5e-24] for p in SCAN])
    assert steady.engine_free(states, prep)["conservation"]["passed"]
    loose = steady.engine_free(states, {**prep, "conservation_atol": None})
    assert not loose["conservation"]["passed"]


def test_the_root_classification_reads_an_sbml_state_in_the_network(tmp_path, sbml_file):
    prep = sbml_prep(tmp_path, sbml_file)
    st = steady.stability(np.array(binding_state(0.5)), 0, prep)
    assert st["stability"] == "stable"


# --------------------------------------------------------------------------- #
# Correctness and timing, with fake engines
# --------------------------------------------------------------------------- #
def _states():
    return np.array([binding_state(p["kr"]) for p in SCAN])


def _fake_bngsim(states=None):
    class FakeBngsim:
        def __init__(self, *args, **kwargs):
            self.method = args[-1]

        def solve_all(self, detailed=False):
            return np.array(_states() if states is None else states)

        def config(self):
            return {"method": self.method, "load_seconds": 0.01}

    return FakeBngsim


def _fake_reference(states=None, built=None):
    class FakeRef:
        def __init__(self, *args, **kwargs):
            if built is not None:
                built.append(type(self).__name__)

        def solve_all(self, detailed=False):
            return np.array(_states() if states is None else states)

        def config(self):
            return {"load_seconds": 0.02}

    return FakeRef


class RunNetworkMustNotRun:
    def __init__(self, *args, **kwargs):
        raise AssertionError("run_network was built although it does not apply")


@pytest.fixture
def not_applicable(monkeypatch, tmp_path, sbml_file):
    """An SBML model whose derived network is missing: run_network does not apply,
    and there are no engine-free checks."""
    def install(refs=None, bngsim=None):
        prep = sbml_prep(tmp_path, sbml_file)
        prep.update(point_networks=None, point_nets=None, net_columns=None,
                    not_applicable={"run_network": "no faithful BNGL form"},
                    integration_reference=steady_sbml.FALLBACK_INTEGRATION_REFERENCE)
        prep["info"]["netfile_error"] = "no derived network to check against"
        monkeypatch.setattr(steady, "_prepare", lambda problem, ctx: prep)
        monkeypatch.setattr(steady_sbml, "build_bngsim",
                            lambda prep, method: (bngsim or _fake_bngsim())(method))
        engines = {"run_network": RunNetworkMustNotRun, "roadrunner": _fake_reference(),
                   "copasi": _fake_reference(), "amici": _fake_reference()}
        engines.update(refs or {})
        monkeypatch.setattr(steady_sbml, "SBML_REFERENCE_ENGINES", engines)
        return prep
    return install


def _ctx(tmp_path, **kw):
    return {"workdir": str(tmp_path), "subprocess_timeout": 60, "engine_budget_seconds": 60, **kw}


def test_a_reference_that_does_not_apply_is_recorded_and_not_counted(not_applicable, tmp_path):
    not_applicable()
    rec = steady.run_correctness(None, None, _ctx(tmp_path))
    assert rec["outcome"] == "PASS", rec
    assert rec["not_applicable"] == {"run_network": "no faithful BNGL form"}
    assert "run_network_error" not in rec and "run_network" not in rec["verdicts"]
    assert rec["measurements"]["integration_reference"] == "amici"
    assert not rec["verdicts"]["engine_free"]["evaluated"]


def test_without_run_network_amici_decides_which_state_the_initial_state_reaches(
        not_applicable, tmp_path):
    other = _states()
    other[2] = [5.0, 4.0, -3.0]  # the root with negative amounts, at the last point
    not_applicable(refs={"roadrunner": _fake_reference(states=other)})
    rec = steady.run_correctness(None, None, _ctx(tmp_path))
    assert rec["outcome"] == "REFERENCE_FAILED"
    assert rec["roadrunner_error"].startswith("OTHER_STATE: at point 2")
    assert rec["verdicts"]["amici"]["passed"] and rec["verdicts"]["copasi"]["passed"]


def test_bngsim_failing_with_every_applicable_reference_is_a_bad_test(not_applicable, tmp_path):
    class Boom:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("no")

    class BoomBngsim(Boom):
        pass

    not_applicable(refs={"roadrunner": Boom, "copasi": Boom, "amici": Boom},
                   bngsim=BoomBngsim)
    rec = steady.run_correctness(None, None, _ctx(tmp_path))
    assert rec["outcome"] == "BAD_TEST", rec


def test_timing_skips_a_reference_that_does_not_apply(not_applicable, tmp_path):
    not_applicable()
    job = type("J", (), {"definition": {"timing": {"warmup_rounds": 0, "min_rounds": 2,
                                                   "max_rounds": 2, "budget_seconds": 1.0}},
                         "requires": None})()
    rec = steady.run_timing(job, None, _ctx(tmp_path))
    assert rec["outcome"] == "PASS", rec
    assert rec["not_applicable"] == {"run_network": "no faithful BNGL form"}
    assert "run_network" not in rec["measurements"]["times"]
    assert set(rec["measurements"]["times"]) == {"bngsim", "bngsim_integration", "roadrunner",
                                                 "copasi", "amici"}


def test_every_reference_has_an_sbml_engine_in_the_same_order():
    assert list(steady_sbml.SBML_REFERENCE_ENGINES) == list(steady.REFERENCE_ENGINES)


def test_run_network_on_the_derived_network_reads_only_the_compared_columns(tmp_path,
                                                                            monkeypatch):
    values = np.tile(np.array([[1.0, 2.0, 3.0, 9.0]]), (21, 1))
    monkeypatch.setattr(steady.RunNetworkSteady, "solve_point", lambda self, i: values)
    rn = steady_sbml.RunNetworkSteadySBML([Path("p0.net")], dict(SBML_PROTO, n_steps=20),
                                          tmp_path / "rn", timeout=10, columns=[2, 0])
    assert rn.solve_point(0)[-1].tolist() == [3.0, 1.0]
