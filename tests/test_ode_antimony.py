"""The ODE capability's Antimony path (the ode-antimony suite): derived specs and
their consistency with the member they come from, the three jobs of a member, the
job-time conversion and its record, the references that read the .ant through
libantimony, and the cross-format check (BNGsim on the .ant against BNGsim on the
source), run for real on a two-species model."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from bench import jobs, specs
from bench.capabilities import ode

SBML_PROTO = {"t_start": 0, "t_end": 10, "n_steps": 20, "rtol": 1e-8, "atol": 1e-12}

# A -> B at rate k*A, with a constant species C that no reaction changes: C is not
# a state species, so it is not compared.
TOY_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="toy">
    <listOfCompartments>
      <compartment id="cell" size="1" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="cell" initialConcentration="10" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
      <species id="B" compartment="cell" initialConcentration="0" hasOnlySubstanceUnits="false"
               boundaryCondition="false" constant="false"/>
      <species id="C" compartment="cell" initialConcentration="3" hasOnlySubstanceUnits="false"
               boundaryCondition="true" constant="true"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.3" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="r1" reversible="false" fast="false">
        <listOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></listOfReactants>
        <listOfProducts><speciesReference species="B" stoichiometry="1" constant="true"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>cell</ci><ci>k</ci><ci>A</ci></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""


def _sbml_member(spec_data, pid="toy_sbml", suites=None):
    d = dict(spec_data, id=pid, layer="core",
             verification={"kind": "author_file", "evidence": "curation"})
    d["source"] = dict(spec_data["source"], path="final/BIOMD1/toy.xml", format="sbml")
    d["protocol"] = {"ode": dict(SBML_PROTO)}
    d["suites"] = suites if suites is not None else {}
    return d


def _derived(src, pos=1, suite="ode-sbml", src_pos=2, **kw):
    d = dict(src, id=f"{src['id']}_antimony", suites={"ode-antimony": pos},
             derived={"from": src["id"], "suite": suite, "position": src_pos,
                      "format": "antimony"})
    d.update(kw)
    return d


# --------------------------------------------------------------------------- #
# Specs
# --------------------------------------------------------------------------- #
def test_ode_antimony_is_a_registered_suite_of_derived_antimony_members():
    assert specs.SUITES["ode-antimony"] == ("ode", ("antimony",))
    assert specs.DERIVED_FORMATS["antimony"] == ("ode",)


def test_a_derived_member_loads_and_is_read_as_antimony(spec_data, write_spec):
    p = specs.load(write_spec(_derived(_sbml_member(spec_data))))
    assert p.input_format == "antimony"
    assert p.source["format"] == "sbml"
    assert p.derived["from"] == "toy_sbml"


@pytest.mark.parametrize("change, message", [
    ({"from": "Not An Id"}, "derived.from"),
    ({"suite": "ode-nope"}, "derived.suite"),
    ({"position": 11}, "derived.position"),
    ({"format": "cellml"}, "derived.format: one of antimony"),
    ({"extra": 1}, "derived: unknown key 'extra'"),
    ({"conversion": ""}, "derived.conversion"),
])
def test_a_bad_derived_block_is_refused(spec_data, write_spec, change, message):
    d = _derived(_sbml_member(spec_data))
    d["derived"] = {**d["derived"], **change}
    with pytest.raises(specs.SpecError, match=message):
        specs.load(write_spec(d))


def test_a_derived_member_lists_only_capabilities_that_read_its_format(spec_data, write_spec):
    d = _derived(_sbml_member(spec_data), capabilities=["ode", "steady_state"])
    d["protocol"] = dict(d["protocol"], steady_state={})
    with pytest.raises(specs.SpecError, match="capability 'steady_state' does not read antimony"):
        specs.load(write_spec(d))


def test_only_a_derived_member_may_be_in_ode_antimony(spec_data, write_spec):
    d = _sbml_member(spec_data, suites={"ode-antimony": 1})
    with pytest.raises(specs.SpecError, match="source.format must be one of antimony"):
        specs.load(write_spec(d))
    d = _derived(_sbml_member(spec_data))
    d["suites"] = {"ode-sbml": 1}
    with pytest.raises(specs.SpecError, match="derived member's format must be one of sbml"):
        specs.load(write_spec(d))


def _load_pair(spec_data, write_spec, tmp_path, **derived_kw):
    src = _sbml_member(spec_data, suites={})
    write_spec(src)
    d = _derived(src, **derived_kw)
    write_spec(d)
    problems = specs.load_all(tmp_path / "problems")
    return {p.id: p for p in problems}


def test_check_derived_holds_a_member_to_its_source(spec_data, write_spec, tmp_path):
    src = _sbml_member(spec_data)
    by_id = {p.id: p for p in (specs.validate(src, "<spec>"),
                               specs.validate(_derived(src), "<spec>"))}
    errs = specs.check_derived(list(by_id.values()))
    assert errs == ["toy_sbml_antimony: toy_sbml is not member 2 of ode-sbml"]

    moved = _derived(src)
    moved["protocol"] = {"ode": dict(SBML_PROTO, rtol=1e-6)}
    moved["source"] = dict(src["source"], sha256="c" * 64)
    errs = specs.check_derived([specs.validate(src, "<spec>"), specs.validate(moved, "<spec>")])
    assert "toy_sbml_antimony: its source is not toy_sbml's" in errs
    assert "toy_sbml_antimony: its ode protocol is not toy_sbml's" in errs

    orphan = _derived(src, derived={"from": "nobody", "suite": "ode-sbml", "position": 2,
                                    "format": "antimony"})
    assert specs.check_derived([specs.validate(orphan, "<spec>")]) == [
        "toy_sbml_antimony: derived from 'nobody', which no spec defines"]


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fmt, xf", [("sbml", "bngsim_sbml"), ("bngl", "bngsim_net")])
def test_a_member_has_correctness_timing_and_a_cross_format_check(spec_data, fmt, xf):
    src = _sbml_member(spec_data) if fmt == "sbml" else dict(
        spec_data, layer="core", verification={"kind": "equations", "evidence": "nb"})
    p = specs.validate(_derived(src), "<spec>")
    corr, timing, cross = ode.jobs(p)
    assert corr.id == f"ode/{p.id}/bngsim-vs-roadrunner-copasi-amici/auto/correctness"
    assert timing.kind == jobs.TIMING and timing.requires == corr.id
    assert cross.id == f"ode/{p.id}/bngsim-vs-{xf}/auto/correctness"
    for j in (corr, timing, cross):
        assert j.definition["input"]["format"] == "antimony"
        assert j.definition["input"]["from"] == fmt
        assert "antimony" in j.definition["input"]
        assert "run_network" not in j.engines
    assert ode.is_cross_format(cross) and not ode.is_cross_format(corr)
    assert ode.job_references(cross) == (xf,)
    assert ode.job_references(corr) == ("roadrunner", "copasi", "amici")
    assert "reference_versions" not in cross.definition


def test_a_source_members_jobs_are_unchanged_by_its_derived_twin(spec_data):
    src = _sbml_member(spec_data, suites={})
    alone = ode.jobs(specs.validate(src, "<spec>"))
    both = jobs.jobs_for([specs.validate(src, "<spec>"), specs.validate(_derived(src), "<spec>")])
    mine = [j for j in both if j.problem == "toy_sbml"]
    assert [(j.id, j.definition) for j in mine] == [(j.id, j.definition) for j in
                                                    sorted(alone, key=lambda j: j.kind)]


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #
@pytest.fixture
def toy_sbml(tmp_path):
    path = tmp_path / "toy.xml"
    path.write_text(TOY_SBML, encoding="utf-8")
    return path


def test_libantimony_converts_both_ways_keeping_the_ids(toy_sbml, tmp_path):
    text = ode.sbml_to_antimony(toy_sbml)
    assert "A -> B" in text.replace("  ", " ") or "A => B" in text
    ant = tmp_path / "toy.ant"
    ant.write_text(text, encoding="utf-8")
    import libsbml

    back = libsbml.readSBMLFromString(ode.antimony_to_sbml(ant)).getModel()
    assert {back.getSpecies(i).getId() for i in range(back.getNumSpecies())} == {"A", "B", "C"}
    assert ode.sbml_to_antimony(toy_sbml) == text  # deterministic


def _ctx(tmp_path):
    return {"workdir": str(tmp_path / "work"), "cache_dir": str(tmp_path / "cache"),
            "subprocess_timeout": 60}


@pytest.fixture
def toy_member(spec_data, toy_sbml, monkeypatch):
    monkeypatch.setattr(ode.fetch, "model_file", lambda source, cache: toy_sbml)
    return specs.validate(_derived(_sbml_member(spec_data)), "<spec>")


def test_prepare_writes_the_ant_and_records_its_sha256(toy_member, tmp_path):
    prep = ode._prepare(toy_member, _ctx(tmp_path))
    assert prep["format"] == "antimony" and prep["source_format"] == "sbml"
    text = Path(prep["ant"]).read_text(encoding="utf-8")
    conv = prep["info"]["antimony"]
    assert conv["ant_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert conv["sbml_sha256"] == hashlib.sha256(Path(prep["source_sbml"]).read_bytes()).hexdigest()
    assert conv["antimony"] and conv["from"] == "sbml"
    # The source's state species: the constant C is not compared.
    assert prep["species_ids"] == ["A", "B"]
    assert conv["compared_species"] == 2


def test_a_conversion_that_loses_a_compared_species_is_a_bad_test(toy_member, tmp_path,
                                                                   monkeypatch):
    real = ode.antimony_to_sbml
    monkeypatch.setattr(ode, "antimony_to_sbml",
                        lambda ant: real(ant).replace('id="B"', 'id="B2"'))
    with pytest.raises(RuntimeError, match="loses 1 compared species"):
        ode._prepare(toy_member, _ctx(tmp_path))
    job = ode.jobs(toy_member)[0]
    rec = ode.run_correctness(job, toy_member, _ctx(tmp_path))
    assert rec["outcome"] == "BAD_TEST" and "loses 1 compared species" in rec["error"]


def test_a_reference_reads_the_ant_through_libantimony_in_its_own_build(toy_sbml, tmp_path):
    ant = tmp_path / "toy.ant"
    ant.write_text(ode.sbml_to_antimony(toy_sbml), encoding="utf-8")

    class Base:
        NAME = "base"

        def __init__(self, sbml, species_ids, proto):
            self.read = Path(sbml)
            self.load_seconds = 0.5

        def config(self):
            return {"load_seconds": self.load_seconds}

    class Reads(ode._ReadsAntimony, Base):
        pass

    ref = Reads(ant, ["A", "B"], SBML_PROTO, tmp_path / "ref")
    assert ref.read == tmp_path / "ref" / "model.xml"
    text = ref.read.read_text(encoding="utf-8")
    assert text == ode.antimony_to_sbml(ant)
    cfg = ref.config()
    assert cfg["load_seconds"] == pytest.approx(0.5 + cfg["antimony_seconds"])
    assert cfg["sbml_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert cfg["input"].startswith("antimony")


def test_every_antimony_reference_reads_the_ant():
    for name, cls in ode.ANTIMONY_REFERENCE_ENGINES.items():
        assert cls.INPUT == "antimony" and cls.NAME == name
        assert issubclass(cls, ode._ReadsAntimony)
    assert "run_network" not in ode.ANTIMONY_REFERENCE_ENGINES


def test_the_net_source_is_compared_by_sbml_id_in_the_compared_order(monkeypatch):
    monkeypatch.setattr(ode.BngsimODE, "__init__", lambda self, net, proto: None)
    eng = ode.BngsimSourceNet(Path("m.net"), ["X()", "Y()", "Z()"],
                              {"X()": "X", "Y()": "Y", "Z()": "Z"}, ["Z", "X", "Y"],
                              SBML_PROTO)
    assert eng.columns == [2, 0, 1]
    assert eng.atol == SBML_PROTO["atol"]  # a float on both sides, not ExactAtol
    with pytest.raises(RuntimeError, match="cannot place 1 compared species"):
        ode.BngsimSourceNet(Path("m.net"), ["X()"], {"X()": "X"}, ["X", "Q"], SBML_PROTO)


# --------------------------------------------------------------------------- #
# The cross-format check, for real (bngsim on the .ant and on the SBML)
# --------------------------------------------------------------------------- #
def test_the_cross_format_check_passes_and_rescores(toy_member, tmp_path):
    cross = ode.jobs(toy_member)[2]
    rec = ode.run(cross, toy_member, _ctx(tmp_path))
    assert rec["outcome"] == "PASS", rec
    assert set(rec["verdicts"]) == {"bngsim_sbml"}
    assert rec["cross_format"] == {"source_format": "sbml", "reference": "bngsim_sbml",
                                   "passed": True}
    assert rec["measurements"]["antimony"]["ant_sha256"]
    assert rec["measurements"]["bngsim"]["input"].startswith("antimony")
    arrays = rec["_trajectories"]
    assert set(arrays) >= {"time", "species", "bngsim", "bngsim_sbml"}
    assert list(arrays["species"]) == ["A", "B"]
    # A -> B at k = 0.3, from A = 10: both reads give the analytic solution.
    t = arrays["time"]
    np.testing.assert_allclose(arrays["bngsim"][:, 0], 10 * np.exp(-0.3 * t), rtol=1e-6)
    again = ode.rescore(cross, rec, arrays)
    assert again["outcome"] == "PASS" and set(again["verdicts"]) == {"bngsim_sbml"}


def test_a_cross_format_disagreement_is_a_diff(toy_member, tmp_path):
    cross = ode.jobs(toy_member)[2]
    rec = ode.run(cross, toy_member, _ctx(tmp_path))
    arrays = dict(rec["_trajectories"])
    arrays["bngsim_sbml"] = arrays["bngsim_sbml"] * 1.1
    assert ode.rescore(cross, rec, arrays)["outcome"] == "DIFF"


# --------------------------------------------------------------------------- #
# The committed suite
# --------------------------------------------------------------------------- #
def test_the_committed_suite_interleaves_ode_net_and_ode_sbml():
    from conftest import REPO

    problems = specs.load_all(REPO / "problems")
    by_id = {p.id: p for p in problems}
    members = specs.suite_members(problems)["ode-antimony"]
    assert [m.suites["ode-antimony"] for m in members] == list(range(1, 11))
    for m in members:
        pos = m.suites["ode-antimony"]
        want = "ode-net" if pos % 2 else "ode-sbml"
        assert m.derived["suite"] == want and m.derived["position"] == pos
        src = by_id[m.derived["from"]]
        assert src.suites[want] == pos
        assert m.capabilities == ("ode",)
        assert m.id == f"{src.id}_antimony"


# --------------------------------------------------------------------------- #
# The correctness and timing paths, for real with BNGsim and libRoadRunner only
# --------------------------------------------------------------------------- #
@pytest.fixture
def rr_only(monkeypatch):
    monkeypatch.setattr(ode, "ANTIMONY_REFERENCE_ENGINES",
                        {"roadrunner": ode.RoadRunnerAntimony})


def test_correctness_reads_the_ant_in_bngsim_and_roadrunner(toy_member, tmp_path, rr_only):
    corr = ode.jobs(toy_member)[0]
    rec = ode.run(corr, toy_member, _ctx(tmp_path))
    assert rec["outcome"] == "PASS", rec
    assert set(rec["verdicts"]) == {"roadrunner"}
    ref = rec["measurements"]["references"]["roadrunner"]
    assert ref["input"].startswith("antimony") and ref["antimony_seconds"] >= 0
    assert "not_applicable" not in rec and "run_network_error" not in rec
    assert set(rec["_trajectories"]) == {"time", "species", "bngsim", "roadrunner"}


def test_timing_times_bngsim_and_the_references_on_the_ant(toy_member, tmp_path, rr_only):
    import dataclasses

    timing = ode.jobs(toy_member)[1]
    timing = dataclasses.replace(timing, definition={
        **timing.definition,
        "timing": {"warmup_rounds": 1, "min_rounds": 2, "max_rounds": 2, "budget_seconds": 0.0}})
    rec = ode.run(timing, toy_member, _ctx(tmp_path))
    m = rec["measurements"]
    assert rec["outcome"] == "PASS", rec
    assert set(m["times"]) == {"bngsim", "roadrunner"}
    assert m["bngsim"]["input"].startswith("antimony")
    assert m["antimony"]["ant_sha256"]
    assert all(r["passed"] for r in m["reproducibility"].values())
