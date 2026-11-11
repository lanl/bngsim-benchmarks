"""Problem-spec validation: a broken spec must be refused with a reason that
names the field, before any job runs."""

from __future__ import annotations

import pytest
from conftest import REPO

from bench import specs


def test_valid_spec_loads_with_its_fields(spec_data, write_spec):
    p = specs.load(write_spec(spec_data))
    assert p.id == "toy_model"
    assert p.capabilities == ("ode",)
    assert p.protocol_for("ode")["n_steps"] == 300
    assert p.size == {"species": 9, "reactions": 14}


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d["source"].update(sha256="b" * 63), "source.sha256"),
        (lambda d: d["source"].update(sha256="B" * 64), "source.sha256"),
        (lambda d: d["source"].update(commit="abc1234"), "source.commit"),
        (lambda d: d["source"].update(path="/etc/passwd"), "source.path"),
        (lambda d: d["source"].update(path="models/../../x.bngl"), "source.path"),
        (lambda d: d["source"].update(repo="not a repo"), "source.repo"),
        (lambda d: d["source"].update(format="sbml2"), "source.format"),
        (lambda d: d.update(layer="gold"), "layer"),
        (lambda d: d.update(publications=[]), "publications"),
        (lambda d: d.update(capabilities=["ode", "teleport"]), "unknown capability 'teleport'"),
        (lambda d: d["protocol"].pop("ode"), "missing an entry for capability 'ode'"),
        (lambda d: d["protocol"]["ode"].update(t_start=600), "t_end must be greater than t_start"),
        (lambda d: d["protocol"]["ode"].update(n_steps=0), "n_steps"),
        (lambda d: d["protocol"]["ode"].update(n_steps=True), "n_steps"),
        (lambda d: d["protocol"]["ode"].update(n_steps=300.0), "n_steps"),
        (lambda d: d["protocol"]["ode"].update(rtol=-1e-8), "rtol"),
        # PyYAML reads 1e-8 (no decimal point) as a string; it must not slip through.
        (lambda d: d["protocol"]["ode"].update(atol="1e-8"), "atol"),
        (lambda d: d["protocol"]["ode"].update(netgen="simulate({})"), "netgen"),
        (lambda d: d["protocol"]["ode"].update(stiff=True), "unknown key 'stiff'"),
        (lambda d: d.update(colour="red"), "unknown key 'colour'"),
    ],
)
def test_invalid_field_is_refused_with_its_name(spec_data, write_spec, mutate, message):
    mutate(spec_data)
    with pytest.raises(specs.SpecError, match=message.replace("(", r"\(").replace(")", r"\)")):
        specs.load(write_spec(spec_data))


def test_yaml_float_without_decimal_point_is_refused(tmp_path):
    # The literal text a hand-written spec would contain.
    text = (REPO / "problems" / "pilot" / "kinetic_proofreading_hlavacek2001.yaml").read_text()
    path = tmp_path / "kinetic_proofreading_hlavacek2001.yaml"
    path.write_text(text.replace("rtol: 1.0e-8", "rtol: 1e-8"))
    with pytest.raises(specs.SpecError, match="rtol"):
        specs.load(path)


def test_pending_verification_only_in_pilot_layer(spec_data, write_spec):
    spec_data["layer"] = "core"
    with pytest.raises(specs.SpecError, match="'pending' is allowed only in the pilot layer"):
        specs.load(write_spec(spec_data))
    spec_data["verification"]["kind"] = "equations"
    assert specs.load(write_spec(spec_data)).layer == "core"


def test_structure_evidence_needs_a_reason(spec_data, write_spec):
    spec_data["layer"] = "core"
    spec_data["verification"]["kind"] = "structure"
    with pytest.raises(specs.SpecError, match="needs a written reason"):
        specs.load(write_spec(spec_data))
    spec_data["verification"]["reason"] = "the paper reports no dynamics"
    assert specs.load(write_spec(spec_data)).verification["kind"] == "structure"


def test_id_must_match_file_name(spec_data, write_spec):
    with pytest.raises(specs.SpecError, match="must match the file name"):
        specs.load(write_spec(spec_data, name="other_name"))


def test_every_problem_reported_at_once(spec_data, write_spec):
    spec_data["source"]["sha256"] = "x"
    spec_data["protocol"]["ode"]["n_steps"] = -1
    with pytest.raises(specs.SpecError) as err:
        specs.load(write_spec(spec_data))
    assert len(err.value.problems) == 2


def test_load_all_refuses_duplicate_ids(spec_data, write_spec, tmp_path):
    write_spec(spec_data, directory=tmp_path / "problems" / "a")
    write_spec(spec_data, directory=tmp_path / "problems" / "b")
    with pytest.raises(specs.SpecError, match="duplicate id 'toy_model'"):
        specs.load_all(tmp_path / "problems")


def test_committed_problem_specs_all_validate():
    problems = specs.load_all(REPO / "problems")
    assert len(problems) >= 1
    # Sorted by id, so job order and reports are stable.
    assert [p.id for p in problems] == sorted(p.id for p in problems)
