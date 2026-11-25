"""The sens capability's SBML path (the sens-sbml suite): a PEtab problem read and
applied by the harness (bench.petab_problem), its spec, its jobs, and the committed
members' sensitivity sets re-derived from their pinned tables."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from bench import fetch, jobs as jobs_mod, specs
from bench.capabilities import sens, sens_sbml
from bench.petab_problem import PetabError, PetabProblem, state_species

REPO = Path(__file__).resolve().parents[1]

SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
  <model id="toy">
    <listOfCompartments>
      <compartment id="cell" size="2"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="cell" initialConcentration="1"/>
      <species id="B" compartment="cell" initialConcentration="0"/>
      <species id="Btot" compartment="cell" initialConcentration="0"/>
      <species id="N" compartment="cell" initialAmount="4" hasOnlySubstanceUnits="true"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k1" value="1"/>
      <parameter id="k2" value="1"/>
      <parameter id="kz" value="1"/>
      <parameter id="A0" value="1"/>
      <parameter id="f" value="1"/>
    </listOfParameters>
    <listOfInitialAssignments>
      <initialAssignment symbol="A">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>A0</ci><ci>f</ci></apply>
        </math>
      </initialAssignment>
    </listOfInitialAssignments>
    <listOfRules>
      <assignmentRule variable="Btot">
        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><plus/><ci>B</ci><ci>A</ci></apply>
        </math>
      </assignmentRule>
    </listOfRules>
    <listOfReactions>
      <reaction id="r1" reversible="false">
        <listOfReactants><speciesReference species="A"/></listOfReactants>
        <listOfProducts><speciesReference species="B"/></listOfProducts>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>cell</ci><ci>k1</ci><ci>A</ci></apply>
        </math></kineticLaw>
      </reaction>
      <reaction id="r2" reversible="false">
        <listOfReactants><speciesReference species="B"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply><times/><ci>cell</ci><ci>k2</ci><ci>kz</ci><ci>B</ci></apply>
        </math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""

PARAMETERS = """parameterId\tparameterScale\tlowerBound\tupperBound\tnominalValue\testimate
k1\tlog10\t0.001\t10\t0.5\t1
k2\tlin\t0.001\t10\t0.2\t1
kz\tlin\t0\t10\t0\t1
A0\tlin\t0\t10\t3\t0
scale_obs\tlog10\t0.1\t10\t2\t1
sd\tlog10\t0.01\t10\t0.1\t1
"""
CONDITIONS = """conditionId\tconditionName\tf\tB
c1\t\t1\t
c2\t\tA0\t0.5
"""
MEASUREMENTS = """observableId\tpreequilibrationConditionId\tsimulationConditionId\tmeasurement\ttime\tobservableParameters\tnoiseParameters
obs_B\t\tc1\t0.3\t2\tscale_obs\tsd
obs_B\t\tc1\t0.2\t1\tscale_obs\tsd
obs_B\t\tc1\t0.25\t1\tscale_obs\tsd
obs_A\t\tc2\t1.5\t2\t2.0\t0.2
obs_tot\tc1\tc2\t2.0\t1\t\t0.5
"""
OBSERVABLES = """observableId\tobservableFormula\tnoiseFormula\tobservableTransformation\tnoiseDistribution
obs_B\tobservableParameter1_obs_B * B\tnoiseParameter1_obs_B\tlin\tnormal
obs_A\tA * observableParameter1_obs_A\tnoiseParameter1_obs_A\tlog10\tnormal
obs_tot\tBtot^2 + k2\tnoiseParameter1_obs_tot\tlin\tlaplace
"""


@pytest.fixture
def toy(tmp_path):
    files = {}
    for name, text in (("model.xml", SBML), ("parameters.tsv", PARAMETERS),
                       ("conditions.tsv", CONDITIONS), ("measurements.tsv", MEASUREMENTS),
                       ("observables.tsv", OBSERVABLES)):
        files[name] = tmp_path / name
        files[name].write_text(text, encoding="utf-8")
    return PetabProblem(files["model.xml"], files["conditions.tsv"], files["measurements.tsv"],
                        files["observables.tsv"], files["parameters.tsv"])


def test_the_sensitivity_set_is_the_estimated_sbml_parameters_that_reach_the_state(toy):
    params, excluded = toy.sensitivity_parameters()
    assert params == ["k1", "k2"]
    assert set(excluded) == {"kz", "scale_obs", "sd"}
    assert "zero" in excluded["kz"] and "not an SBML parameter" in excluded["sd"]
    assert state_species(toy.model) == ["A", "B", "N"]


def test_conditions_are_the_measurement_tables_pairs_with_their_overrides(toy):
    ids = [c.id for c in toy.conditions]
    assert ids == ["c1", "c2", "c1:c2"]
    c1, c2, pre = toy.conditions
    # The table's nominal values override the SBML's; a condition cell may name a
    # parameter-table id, which stands for its nominal value.
    assert c1.parameters == {"k1": 0.5, "k2": 0.2, "kz": 0.0, "A0": 3.0, "f": 1.0}
    assert c2.parameters["f"] == 3.0
    assert c1.species == {} and c2.species == {"B": 0.5}
    assert c1.times.tolist() == [1.0, 2.0]
    assert pre.preeq == "c1" and pre.preeq_parameters["f"] == 1.0 and pre.parameters["f"] == 3.0
    assert [r.time_index for r in toy.rows[:3]] == [1, 0, 0]


def test_observables_and_the_likelihood_follow_petab(toy):
    params = ["k1", "k2"]
    om = toy.output_model(["A", "B", "N"], params)
    rng = np.random.default_rng(1)
    X = [rng.uniform(0.5, 2.0, (c.times.size, 3)) for c in toy.conditions]
    S = [rng.normal(size=(c.times.size, 3, 2)) for c in toy.conditions]
    out = om.evaluate(X, S)
    # obs_B = scale * B, scale from the parameter table (nominal 2).
    assert out["y"][0] == pytest.approx(2.0 * X[0][1, 1])
    assert out["sy"][0] == pytest.approx(2.0 * S[0][1, 1])
    assert out["sigma"][0] == pytest.approx(0.1) and np.all(out["ssigma"][0] == 0)
    # obs_A reads a numeric override; obs_tot expands the rule Btot = B + A and reads
    # k2 itself, so its sensitivity has an explicit term in k2.
    assert out["y"][3] == pytest.approx(2.0 * X[1][0, 0])
    btot = X[2][0, 0] + X[2][0, 1]
    assert out["y"][4] == pytest.approx(btot ** 2 + 0.2)
    dbtot = S[2][0, 0] + S[2][0, 1]
    assert out["sy"][4] == pytest.approx(2 * btot * dbtot + np.array([0.0, 1.0]))
    lik = om.likelihood(out)
    want, want_amici = 0.0, 0.0
    for j, (m, sg) in enumerate(((0.3, 0.1), (0.2, 0.1), (0.25, 0.1))):
        r = (out["y"][j] - m) / sg
        want += -0.5 * math.log(2 * math.pi * sg ** 2) - 0.5 * r ** 2
    want_amici = want
    m, sg, s = 1.5, 0.2, out["y"][3]
    r = (math.log10(s) - math.log10(m)) / sg
    base = -0.5 * math.log(2 * math.pi * sg ** 2) - 0.5 * r ** 2
    want += base - math.log(m * math.log(10))
    want_amici += base
    m, sg, s = 2.0, 0.5, out["y"][4]
    lap = -math.log(2 * sg) - abs(s - m) / sg
    want += lap
    want_amici += lap
    assert lik["llh"] == pytest.approx(want)
    assert lik["llh_without_log_jacobian"] == pytest.approx(want_amici)
    # The gradient is the chain rule through each row's value (sigma is constant here).
    grad = sum(-(out["y"][j] - m) / sg ** 2 * out["sy"][j]
               for j, (m, sg) in enumerate(((0.3, 0.1), (0.2, 0.1), (0.25, 0.1))))
    r = (math.log10(out["y"][3]) - math.log10(1.5)) / 0.2
    grad = grad - r / 0.2 / (out["y"][3] * math.log(10)) * out["sy"][3]
    grad = grad - math.copysign(1.0, out["y"][4] - 2.0) / 0.5 * out["sy"][4]
    assert lik["gradient"] == pytest.approx(grad)
    assert np.all(lik["gradient_magnitude"] >= np.abs(lik["gradient"]) - 1e-12)


def test_a_condition_table_column_that_is_no_model_entity_is_refused(tmp_path, toy):
    bad = tmp_path / "conditions_bad.tsv"
    bad.write_text("conditionId\tnosuch\nc1\t1\nc2\t2\n", encoding="utf-8")
    with pytest.raises(PetabError, match="nosuch"):
        PetabProblem(toy.sbml, bad, tmp_path / "measurements.tsv", tmp_path / "observables.tsv",
                     tmp_path / "parameters.tsv")


def test_an_estimated_condition_column_is_not_differentiated(tmp_path, toy):
    p = tmp_path / "parameters2.tsv"
    p.write_text(PARAMETERS + "f\tlin\t0\t10\t1\t1\n", encoding="utf-8")
    pp = PetabProblem(toy.sbml, tmp_path / "conditions.tsv", tmp_path / "measurements.tsv",
                      tmp_path / "observables.tsv", p)
    params, excluded = pp.sensitivity_parameters()
    assert params == ["k1", "k2"] and "condition-table column" in excluded["f"]


# --------------------------------------------------------------------------- #
# Specs and jobs
# --------------------------------------------------------------------------- #
@pytest.fixture
def petab_spec(spec_data):
    spec_data["layer"] = "core"
    spec_data["source"] = dict(spec_data["source"], path="Benchmark-Models/T/model.xml",
                               format="sbml")
    spec_data["petab"] = {role: {"path": f"Benchmark-Models/T/{role}.tsv", "sha256": "c" * 64}
                          for role in ("conditions", "measurements", "observables",
                                       "parameters")}
    spec_data["verification"] = {"kind": "author_file", "evidence": "the PEtab collection"}
    spec_data["capabilities"] = ["sens"]
    spec_data["protocol"] = {"sens": {"rtol": 1e-8, "atol": 1e-10, "parameters": ["k1", "k2"],
                                      "llh_reference": -1.5}}
    spec_data["size"] = {"species": 3, "reactions": 2, "conditions": 3}
    return spec_data


def test_a_petab_sens_spec_loads(petab_spec, write_spec):
    p = specs.load(write_spec(petab_spec))
    assert p.petab["conditions"]["path"].endswith("conditions.tsv")
    assert p.protocol_for("sens")["parameters"] == ["k1", "k2"]


@pytest.mark.parametrize("change, message", [
    (lambda d: d.pop("petab"), "PEtab problem"),
    (lambda d: d["petab"].pop("measurements"), "missing the measurements table"),
    (lambda d: d["petab"].update(extra={"path": "x.tsv", "sha256": "c" * 64}), "unknown table"),
    (lambda d: d["petab"]["conditions"].update(sha256="nothex"), "sha256"),
    (lambda d: d["protocol"]["sens"].update(t_end=10), "unknown key 't_end'"),
    (lambda d: d["protocol"]["sens"].pop("parameters"), "missing 'parameters'"),
    (lambda d: d["protocol"]["sens"].update(fd_rtol=1e-6), "fd_rtol"),
])
def test_a_bad_petab_sens_spec_is_refused(petab_spec, write_spec, change, message):
    change(petab_spec)
    with pytest.raises(specs.SpecError, match=message):
        specs.load(write_spec(petab_spec))


def test_a_petab_block_needs_an_sbml_source(petab_spec, write_spec):
    petab_spec["source"]["format"] = "bngl"
    petab_spec["protocol"] = {"sens": {"netgen": "generate_network({overwrite=>1})",
                                       "t_end": 1, "n_steps": 2, "rtol": 1e-8, "atol": 1e-8,
                                       "parameters": "independent"}}
    with pytest.raises(specs.SpecError, match="only an SBML source"):
        specs.load(write_spec(petab_spec))


def test_sens_sbml_is_a_registered_suite_of_sbml_models():
    assert specs.SUITES["sens-sbml"] == ("sens", ("sbml",))


def test_a_petab_problem_gets_the_sbml_paths_jobs(petab_spec, write_spec):
    p = specs.load(write_spec(petab_spec))
    js = sens.jobs(p)
    assert [j.kind for j in js] == ["correctness", "timing"]
    assert {j.engines for j in js} == {"bngsim-vs-amici-fd_roadrunner"}
    c, t = js
    assert t.requires == c.id
    assert c.definition["input"]["format"] == "sbml"
    assert set(c.definition["petab_sha256"]) == {"conditions", "measurements", "observables",
                                                  "parameters"}
    assert c.definition["protocol"]["fd_atol"] == pytest.approx(1e-12)
    assert "verdict" in c.definition and "timing" in t.definition
    assert c.definition["verdict"]["gradient_tol"] == sens_sbml.VERDICT["gradient_tol"]


def test_the_gradient_is_judged_against_its_own_size_not_a_cancelled_sum():
    scales = np.ones(2)
    # Near the fitted point the gradient is a small sum of large contributions; an
    # integration error far below the contributions is not a disagreement.
    bn, ref = np.array([1e-5, 2e-5]), np.array([1.2e-5, 2.1e-5])
    assert sens_sbml.gradient_check(bn, ref, scales, ["a", "b"])["passed"] is False
    ok = sens_sbml.gradient_check(bn, ref, scales, ["a", "b"], np.array([1.0, 0.5]))
    assert ok["passed"] is True and ok["semi_scaled_size"] == 1.0


# --------------------------------------------------------------------------- #
# The committed suite
# --------------------------------------------------------------------------- #
def _members():
    return specs.suite_members(specs.load_all(REPO / "problems")).get("sens-sbml", [])


def test_the_suite_has_ten_petab_members_at_the_pinned_commit():
    ms = _members()
    assert len(ms) == specs.SUITE_SIZE
    for m in ms:
        assert m.source["repo"] == "Benchmarking-Initiative/Benchmark-Models-PEtab"
        assert m.source["commit"] == "4d2085084b289f6215a95475b1ee639fd7d42283"
        assert set(m.petab) >= set(specs.PETAB_REQUIRED)
        assert all(f["path"].startswith(str(Path(m.source["path"]).parent))
                   for f in m.petab.values())


@pytest.mark.parametrize("member", [m.id for m in _members()])
def test_each_members_sensitivity_set_is_rederived_from_its_pinned_tables(member):
    """D12: the spec writes the sensitivity set down; the pinned tables decide it."""
    m = next(p for p in _members() if p.id == member)
    cache = REPO / ".cache"
    files = {"sbml": fetch.model_cache_path(m.source, cache)}
    for role, f in m.petab.items():
        files[role] = fetch.model_cache_path({**m.source, "path": f["path"],
                                              "sha256": f["sha256"]}, cache)
    if not all(p.exists() for p in files.values()):
        pytest.skip("the pinned PEtab files are not in the cache (bench fetch-models)")
    for role, p in files.items():
        want = m.source["sha256"] if role == "sbml" else m.petab[role]["sha256"]
        assert fetch.sha256_file(p) == want, role
    pp = PetabProblem(files["sbml"], files["conditions"], files["measurements"],
                      files["observables"], files["parameters"])
    params, _ = pp.sensitivity_parameters()
    assert m.protocol_for("sens")["parameters"] == params
    assert m.size["conditions"] == len(pp.conditions)
    assert m.size["measurements"] == len(pp.rows)


def test_every_sens_sbml_job_selects_by_suite_name():
    problems = specs.load_all(REPO / "problems")
    from bench import cli

    pats = cli._expand_suites(["sens-sbml"], problems)
    chosen = jobs_mod.select(jobs_mod.jobs_for(problems), pats)
    assert len([j for j in chosen if j.kind == "correctness"]) == specs.SUITE_SIZE
    assert all(j.engines == "bngsim-vs-amici-fd_roadrunner" for j in chosen)
