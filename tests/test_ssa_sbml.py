"""Exact SSA on SBML models (bench.capabilities.ssa_sbml) and the parts a correctness
job is split into: specs, jobs, what is compared, the rows left out, the jump-process
gate on a derived network, the moments table, and stored parts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from bench import netfile, results, specs
from bench.capabilities import ssa, ssa_sbml

PROTOCOL = {"t_end": 10, "n_steps": 10, "replicates": 200}
TABLE = {
    "kind": "table", "repo": "sbmlteam/sbml-test-suite", "commit": "a" * 40,
    "mean": {"path": "cases/stochastic/00001/m-mean.csv", "sha256": "b" * 64},
    "sd": {"path": "cases/stochastic/00001/m-sd.csv", "sha256": "c" * 64},
}


@pytest.fixture
def sbml_spec(spec_data):
    spec_data["source"]["format"] = "sbml"
    spec_data["source"]["path"] = "final/BIOMD0000000001/model.xml"
    spec_data["capabilities"] = ["ssa"]
    spec_data["protocol"] = {"ssa": dict(PROTOCOL)}
    spec_data["size"] = {"species": 3, "reactions": 2, "ssa_ms_bngsim": 5}
    return spec_data


# --------------------------------------------------------------------------- #
# Specs and jobs
# --------------------------------------------------------------------------- #
def test_an_sbml_spec_gets_jobs_that_say_their_input(sbml_spec, write_spec):
    correctness, timing = ssa.jobs(specs.load(write_spec(sbml_spec)))
    for j in (correctness, timing):
        assert j.definition["input"] == {"format": "sbml",
                                         "harness": ssa_sbml.SBML_HARNESS_VERSION}
    assert "netgen" not in correctness.definition["protocol"]
    # Every engine is a part of the correctness job, planned from its cost.
    assert set(correctness.parts) == {"bngsim", *ssa.REFERENCE_ENGINES}
    assert correctness.parts["bngsim"] == pytest.approx(10.0 + 200 * 0.005)
    assert not timing.parts


def test_a_bngl_jobs_definition_has_no_input_key(spec_data, write_spec):
    spec_data["capabilities"] = ["ssa"]
    spec_data["protocol"] = {"ssa": {"netgen": "generate_network({overwrite=>1})", **PROTOCOL}}
    correctness, _ = ssa.jobs(specs.load(write_spec(spec_data)))
    assert "input" not in correctness.definition


@pytest.mark.parametrize("change, message", [
    ({"netgen": "generate_network({overwrite=>1})"}, "an SBML model is read as it is"),
    ({"moments": "linear"}, "pinned table"),
    ({"moments": {**TABLE, "kind": "linear"}}, "pinned table"),
    ({"moments": {**TABLE, "mean": {"path": "/abs.csv", "sha256": "b" * 64}}},
     "moments.mean.path"),
    ({"moments": {**TABLE, "sd": {"path": "x.csv", "sha256": "nothex"}}}, "moments.sd.sha256"),
    ({"moments": {**TABLE, "extra": 1}}, "unknown key 'extra'"),
    ({"moments": TABLE}, "replicates >= 10000"),
])
def test_bad_sbml_ssa_protocols_are_refused(sbml_spec, write_spec, change, message):
    sbml_spec["protocol"]["ssa"].update(change)
    with pytest.raises(specs.SpecError, match=message):
        specs.load(write_spec(sbml_spec))


def test_a_moments_table_at_dsmts_size_is_accepted(sbml_spec, write_spec):
    sbml_spec["protocol"]["ssa"].update(moments=TABLE, replicates=10_000)
    p = specs.load(write_spec(sbml_spec))
    assert ssa.moments_kind(ssa.protocol(p)) == "table"


def test_the_ssa_sbml_suite_takes_sbml_members_only(spec_data, write_spec):
    spec_data["capabilities"] = ["ssa"]
    spec_data["protocol"] = {"ssa": {"netgen": "generate_network({overwrite=>1})", **PROTOCOL}}
    spec_data["suites"] = {"ssa-sbml": 1}
    with pytest.raises(specs.SpecError, match="source.format must be one of sbml"):
        specs.load(write_spec(spec_data))


def test_a_measured_cost_per_trajectory_replaces_the_event_estimate(sbml_spec, write_spec):
    sbml_spec["size"].update(ssa_ms_roadrunner=2000, ssa_ms_run_network=0, ssa_events=10**6)
    p = specs.load(write_spec(sbml_spec))
    per = ssa.trajectory_seconds(p)
    assert per["bngsim"] == 0.005 and per["roadrunner"] == 2.0 and per["run_network"] == 0.0
    assert per["copasi"] == pytest.approx(ssa.TRAJECTORY_SECONDS["copasi"] + 2.0)
    # the budget follows the slowest ensemble, under its cap
    sbml_spec["size"]["ssa_ms_roadrunner"] = 20_000
    p = specs.load(write_spec(sbml_spec))
    assert ssa.engine_budget_seconds(p) == pytest.approx(
        min(ssa.ENSEMBLE_BUDGET_CAP_SECONDS, ssa.TIMEOUT_FACTOR * (10 + 200 * 20.0)))


def test_the_committed_ssa_sbml_suite_is_complete():
    root = Path(__file__).resolve().parents[1] / "problems"
    members = specs.suite_members(specs.load_all(root))["ssa-sbml"]
    assert len(members) == specs.SUITE_SIZE
    for p in members:
        assert p.source["format"] == "sbml" and p.capabilities == ("ssa",)
        assert len(ssa.jobs(p)) == 2


# --------------------------------------------------------------------------- #
# The SBML: what is compared, rows left out, jumps
# --------------------------------------------------------------------------- #
def _species(sid, amount, *, boundary=False, constant=False):
    return (f'<species id="{sid}" compartment="c" initialAmount="{amount}" '
            f'hasOnlySubstanceUnits="true" boundaryCondition="{str(boundary).lower()}" '
            f'constant="{str(constant).lower()}"/>')


def _ref(tag, sid, st=1):
    return f'<{tag} species="{sid}" stoichiometry="{st}" constant="true"/>'


MATH = 'xmlns="http://www.w3.org/1998/Math/MathML"'


def write_sbml(path: Path, *, size: str = "1", delay: str | None = None) -> Path:
    delay_xml = (f'<delay><math {MATH}><cn>{delay}</cn></math></delay>' if delay else "")
    path.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
 <model id="toy">
  <listOfCompartments><compartment id="c" size="{size}" constant="true"/></listOfCompartments>
  <listOfSpecies>
   {_species("A", 10)} {_species("B", 0)} {_species("S", 1, boundary=True)}
   {_species("K", 5, boundary=True, constant=True)} {_species("E", 0)} {_species("Z", 3)}
  </listOfSpecies>
  <listOfParameters>
   <parameter id="k" value="0.1" constant="true"/><parameter id="t1" value="5" constant="true"/>
  </listOfParameters>
  <listOfReactions>
   <reaction id="dimer" reversible="false" fast="false">
    <listOfReactants>{_ref("speciesReference", "A", 2)}</listOfReactants>
    <listOfProducts>{_ref("speciesReference", "B")}</listOfProducts>
    <listOfModifiers><modifierSpeciesReference species="Z"/></listOfModifiers>
    <kineticLaw><math {MATH}><apply><times/><ci>k</ci><ci>A</ci><ci>Z</ci></apply></math></kineticLaw>
   </reaction>
   <reaction id="source" reversible="false" fast="false">
    <listOfReactants>{_ref("speciesReference", "S")}</listOfReactants>
    <listOfProducts>{_ref("speciesReference", "A")}</listOfProducts>
    <kineticLaw><math {MATH}><apply><times/><ci>k</ci><ci>S</ci></apply></math></kineticLaw>
   </reaction>
  </listOfReactions>
  <listOfEvents>
   <event id="at_t1" useValuesFromTriggerTime="true">
    <trigger initialValue="true" persistent="true"><math {MATH}><apply><geq/>
     <csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>
     <ci>t1</ci></apply></math></trigger>{delay_xml}
    <listOfEventAssignments><eventAssignment variable="E"><math {MATH}><cn>1</cn></math>
    </eventAssignment></listOfEventAssignments>
   </event>
   <event id="on_count" useValuesFromTriggerTime="true">
    <trigger initialValue="true" persistent="true"><math {MATH}><apply><gt/><ci>A</ci>
     <cn>20</cn></apply></math></trigger>
    <listOfEventAssignments><eventAssignment variable="E"><math {MATH}><cn>2</cn></math>
    </eventAssignment></listOfEventAssignments>
   </event>
  </listOfEvents>
 </model>
</sbml>
""", encoding="utf-8")
    return path


def test_the_census_compares_what_a_reaction_rule_or_event_changes(tmp_path):
    species, census = ssa_sbml.sbml_census(write_sbml(tmp_path / "m.xml"))
    assert species == ["A", "B", "E"]
    assert census["not_compared"] == {"constant": ["K"], "assignment_rule": [],
                                      "unchanged": ["S", "Z"]}
    assert census["time_events"] == [{"id": "at_t1", "time": 5.0, "trigger": "time >= t1"}]
    assert census["events"] == 2 and census["compared_species"] == 3


def test_a_delayed_time_event_fires_at_its_trigger_plus_its_delay(tmp_path):
    _, census = ssa_sbml.sbml_census(write_sbml(tmp_path / "m.xml", delay="2"))
    assert [e["time"] for e in census["time_events"]] == [7.0]


def test_a_compartment_whose_size_is_not_one_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="size 2"):
        ssa_sbml.sbml_census(write_sbml(tmp_path / "m.xml", size="2"))


def test_rows_at_an_instant_an_event_fires_are_left_out(tmp_path):
    _, census = ssa_sbml.sbml_census(write_sbml(tmp_path / "m.xml"))
    grid = np.linspace(0, 10, 11)
    assert ssa_sbml.excluded_rows(grid, census) == [5.0]
    assert ssa_sbml.excluded_rows(np.linspace(0, 10, 4), census) == []


NET = """begin parameters
    1 k 0.1  # Constant
end parameters
begin species
    1 A() 10
    2 B() 0
    3 $S() 1
end species
begin reactions
{reactions}
end reactions
"""


def _net(tmp_path, reactions: str) -> netfile.Network:
    path = tmp_path / "d.net"
    path.write_text(NET.format(reactions=reactions), encoding="utf-8")
    return netfile.read(path)


def test_jumps_of_the_sbml_and_of_a_derived_network(tmp_path):
    sbml = write_sbml(tmp_path / "m.xml")
    want = ssa_sbml.sbml_jumps(sbml, ["A", "B", "E"])
    assert want == Counter({(("A", -2.0), ("B", 1.0)): 1, (("A", 1.0),): 1})
    same = _net(tmp_path, "    1 1,1 2 k\n    2 3 1,3 k")
    assert ssa_sbml.net_jumps(same, [0, 1], ["A", "B"]) == want
    # flux-split: the same ODE, other jumps
    split = _net(tmp_path, "    1 0 2 k\n    2 1 0 k\n    3 1 0 k\n    4 3 1,3 k")
    assert ssa_sbml.net_jumps(split, [0, 1], ["A", "B"]) != want


def _derived(tmp_path, reactions, phases=None):
    path = tmp_path / "d.net"
    path.write_text(NET.format(reactions=reactions), encoding="utf-8")
    return {"applicable": True, "net": str(path), "columns": [0, 1],
            "phases": phases or [{"changes": [], "change_targets": []}], "info": {}}


@pytest.mark.parametrize("reactions, phases, reason", [
    ("    1 0 2 k\n    2 1 0 k\n    3 1 0 k\n    4 3 1,3 k", None, "SBML's jumps"),
    ("    1 1,1 2 k\n    2 3 1,3 k",
     [{"changes": [[2, 0.6]], "change_targets": ["S"]}, {"changes": []}], "rounds"),
    ("    1 1,1 2 k\n    2 3 1,3 k",
     [{"changes": [], "change_targets": []}, {"changes": [[2, 3.0]], "change_targets": ["S"]}],
     "2 phases"),
])
def test_a_derived_network_that_is_not_the_sbmls_jump_process_does_not_apply(
        tmp_path, monkeypatch, reactions, phases, reason):
    sbml = write_sbml(tmp_path / "m.xml")
    monkeypatch.setattr(ssa_sbml.ode, "derive_network",
                        lambda *a, **k: _derived(tmp_path, reactions, phases))
    # (the toy's dimerisation A + A would be refused first; these cases are about phases)
    monkeypatch.setattr(netfile.Network, "repeated_reactants", lambda self: [])
    out = ssa_sbml.derive_network(sbml, ["A", "B"], dict(PROTOCOL), tmp_path, timeout=10)
    assert out["applicable"] is False and reason in out["reason"]


def test_a_derived_network_with_the_sbmls_jumps_applies_unless_a_reactant_repeats(
        tmp_path, monkeypatch):
    sbml = write_sbml(tmp_path / "m.xml")
    monkeypatch.setattr(ssa_sbml.ode, "derive_network",
                        lambda *a, **k: _derived(tmp_path, "    1 1,1 2 k\n    2 3 1,3 k"))
    # the repeated reactant A + A is refused: its ODE rate does not fix the propensity
    out = ssa_sbml.derive_network(sbml, ["A", "B"], dict(PROTOCOL), tmp_path, timeout=10)
    assert out["applicable"] is False and "consume one species twice" in out["reason"]
    monkeypatch.setattr(netfile.Network, "repeated_reactants", lambda self: [])
    out = ssa_sbml.derive_network(sbml, ["A", "B"], dict(PROTOCOL), tmp_path, timeout=10)
    assert out["applicable"] is True and out["info"]["jumps"]["derived_reactions"] == 2


def test_a_derivation_the_ode_gate_refuses_is_passed_through(tmp_path, monkeypatch):
    monkeypatch.setattr(ssa_sbml.ode, "derive_network",
                        lambda *a, **k: {"applicable": False, "reason": "state events"})
    out = ssa_sbml.derive_network(tmp_path / "m.xml", ["A"], dict(PROTOCOL), tmp_path, timeout=1)
    assert out == {"applicable": False, "reason": "state events"}


# --------------------------------------------------------------------------- #
# The moments table
# --------------------------------------------------------------------------- #
def test_a_moments_table_is_read_on_the_grid_in_the_compared_order(tmp_path, monkeypatch):
    (tmp_path / "mean.csv").write_text("Time,P2,P\n" + "".join(
        f"{t},{t * 2},{100 - t}\n" for t in range(11)))
    (tmp_path / "sd.csv").write_text("Time,P2,P\n" + "".join(f"{t},1,{t}\n" for t in range(11)))
    monkeypatch.setattr(ssa_sbml.fetch, "model_file",
                        lambda src, cache: tmp_path / Path(src["path"]).name)
    spec = {**TABLE, "mean": {"path": "mean.csv"}, "sd": {"path": "sd.csv"}}
    mean, var = ssa_sbml.table_moments(spec, tmp_path, dict(PROTOCOL), ["P", "P2"])
    assert mean.shape == (11, 2) and mean[3].tolist() == [97.0, 6.0]
    assert var[3].tolist() == [9.0, 1.0]
    with pytest.raises(RuntimeError, match="not the compared species"):
        ssa_sbml.table_moments(spec, tmp_path, dict(PROTOCOL), ["P"])
    with pytest.raises(RuntimeError, match="not the protocol's grid"):
        ssa_sbml.table_moments(spec, tmp_path, dict(PROTOCOL, n_steps=5), ["P", "P2"])


# --------------------------------------------------------------------------- #
# Correctness on the SBML path, with fake engines
# --------------------------------------------------------------------------- #
from test_ssa import _ctx, _fake_bngsim, _fake_reference, _fake_stack, _prep  # noqa: E402


def _sbml_prep(keep=None):
    prep = _prep()
    prep.update(format="sbml", keep=keep)
    return prep


def _install_sbml(monkeypatch, refs, bngsim=None, keep=None, derived=None):
    monkeypatch.setattr(ssa, "_prepare", lambda problem, ctx: _sbml_prep(keep))
    monkeypatch.setattr(ssa, "_build_bngsim", lambda prep: (bngsim or _fake_bngsim())(None, None))
    monkeypatch.setattr(ssa, "REFERENCE_ENGINES", refs)
    monkeypatch.setattr(ssa, "_build_reference",
                        lambda name, prep, **kw: refs[name](None))
    monkeypatch.setattr(ssa, "_derive_isolated", lambda *a, **k: derived or {
        "applicable": False, "reason": "no faithful network", "info": {"jumps": {}}})


def test_run_network_without_a_faithful_network_is_not_applicable(monkeypatch, tmp_path):
    _install_sbml(monkeypatch, {"run_network": _fake_reference(), "copasi": _fake_reference()})
    rec = ssa.run_correctness(None, None, _ctx(tmp_path))
    assert rec["outcome"] == "PASS", rec
    assert rec["not_applicable"] == {"run_network": "no faithful network"}
    assert set(rec["verdicts"]) == {"copasi"}
    assert rec["measurements"]["derived_network"] == {"jumps": {}}


def test_rows_left_out_are_left_out_of_every_verdict(monkeypatch, tmp_path):
    keep = np.ones(11, dtype=bool)
    keep[5] = False
    # an engine that differs only at the excluded instant still agrees
    class Off(_fake_reference()):
        def ensemble(self, n, seed):
            out = _fake_stack(n, seed + 12345)
            out[:, 5, :] += 50.0
            return out
    _install_sbml(monkeypatch, {"copasi": Off}, keep=keep)
    rec = ssa.run_correctness(None, None, _ctx(tmp_path))
    assert rec["outcome"] == "PASS", rec
    keep[:] = True
    rec = ssa.run_correctness(None, None, _ctx(tmp_path))
    assert rec["outcome"] == "DIFF"


# --------------------------------------------------------------------------- #
# Parts: one engine per run, stored, then scored
# --------------------------------------------------------------------------- #
def _part_job(tmp_path):
    from bench.jobs import Job

    return Job("ssa", "toy", "e", "auto", "correctness", definition={"x": 1},
               parts={"bngsim": 10.0, "copasi": 10.0, "roadrunner": 10.0})


def test_parts_are_stored_and_the_job_scores_them_without_sampling_again(monkeypatch, tmp_path):
    refs = {"copasi": _fake_reference(), "roadrunner": _fake_reference()}
    monkeypatch.setattr(ssa, "_prepare", lambda problem, ctx: _prep())
    monkeypatch.setattr(ssa, "BngsimSSA", _fake_bngsim())
    monkeypatch.setattr(ssa, "REFERENCE_ENGINES", refs)
    job = _part_job(tmp_path)
    ctx = _ctx(tmp_path / "w", results_dir=str(tmp_path / "res"), key="K1")
    # Each part's task ran its own checkout of the harness, and says which.
    monkeypatch.setattr(ssa.env, "harness_source", lambda: {"commit": "p" * 40})
    for name in ("bngsim", "copasi", "roadrunner"):
        assert ssa.run_correctness(job.as_part(name), None, {**ctx, "part": name}) == {
            "outcome": "PART_STORED"}
    # the parts ran in forked children: the stored parts are on disk
    assert all(results.read_part(tmp_path / "res", job, n, "K1") for n in job.parts)
    monkeypatch.setattr(ssa, "_engine_run", lambda *a, **k: pytest.fail("sampled again"))
    rec = ssa.run_correctness(job, None, ctx)
    assert rec["outcome"] == "PASS", rec
    assert sorted(rec["measurements"]["from_parts"]) == ["bngsim", "copasi", "roadrunner"]
    assert rec["measurements"]["part_sources"] == {
        name: {"commit": "p" * 40} for name in ("bngsim", "copasi", "roadrunner")}
    # a part stored under another key is not used
    assert results.read_part(tmp_path / "res", job, "bngsim", "K2") is None


def test_a_failed_part_is_stored_as_its_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(ssa, "_prepare", lambda problem, ctx: _prep())
    monkeypatch.setattr(ssa, "BngsimSSA", _fake_bngsim())
    monkeypatch.setattr(ssa, "REFERENCE_ENGINES",
                        {"copasi": _fake_reference(boom="copasi broke"),
                         "roadrunner": _fake_reference()})
    job = _part_job(tmp_path)
    ctx = _ctx(tmp_path / "w", results_dir=str(tmp_path / "res"), key="K1")
    ssa.run_correctness(job.as_part("copasi"), None, {**ctx, "part": "copasi"})
    rec, arrays = results.read_part(tmp_path / "res", job, "copasi", "K1")
    assert "copasi broke" in rec["error"] and arrays == {}
    out = ssa.run_correctness(job, None, ctx)
    assert out["outcome"] == "REFERENCE_FAILED" and "copasi broke" in out["copasi_error"]


def test_a_job_with_parts_runs_whole_when_none_is_stored(monkeypatch, tmp_path):
    monkeypatch.setattr(ssa, "_prepare", lambda problem, ctx: _prep())
    monkeypatch.setattr(ssa, "BngsimSSA", _fake_bngsim())
    monkeypatch.setattr(ssa, "REFERENCE_ENGINES", {"copasi": _fake_reference(),
                                                   "roadrunner": _fake_reference()})
    job = _part_job(tmp_path)
    rec = ssa.run_correctness(job, None, _ctx(tmp_path / "w", results_dir=str(tmp_path / "r"),
                                              key="K"))
    assert rec["outcome"] == "PASS" and "from_parts" not in rec["measurements"]
