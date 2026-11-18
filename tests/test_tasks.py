"""Agent-task validation: a broken task must be refused with a reason that names
the field, before any agent job is planned."""

from __future__ import annotations

import copy

import pytest
from conftest import REPO, VALID_TASK

from bench import specs, tasks


def test_valid_task_loads(task_data, write_task):
    task_data["request"] = "  " + task_data["request"] + "\n"
    t = tasks.load(write_task(task_data))
    assert t.request == VALID_TASK["request"]
    assert t.id == "agent/timecourse/toy_peak"
    assert t.name == "toy_peak"
    assert t.category == "timecourse"
    assert t.problem == "toy_model"
    assert t.forms == ("sbml", "bngl")
    assert [v.name for v in t.answer] == ["peak_value", "peak_time"]
    assert t.value("peak_value") == tasks.Value("peak_value", 0.0412, rel_tol=1e-3)
    assert t.value("peak_time") == tasks.Value("peak_time", 17.5, abs_tol=0.5)
    assert t.trap == "none" and t.split == "public"


def test_a_value_may_have_both_tolerances(task_data, write_task):
    task_data["answer"][0]["abs_tol"] = 1.0e-9
    v = tasks.load(write_task(task_data)).value("peak_value")
    assert (v.rel_tol, v.abs_tol) == (1e-3, 1e-9)


def test_the_expected_outcome_is_the_default_unless_an_engine_says_otherwise(task_data, write_task):
    task_data.update(id="agent/should_refuse/toy_delay", category="should_refuse")
    task_data["expected_outcome"] = {"default": "refusal", "roadrunner": "answer"}
    t = tasks.load(write_task(task_data))
    assert t.expected_for("bngsim") == "refusal"
    assert t.expected_for("amici") == "refusal"
    assert t.expected_for("roadrunner") == "answer"


def test_a_task_every_engine_refuses_needs_no_answer(task_data, write_task):
    task_data.update(id="agent/should_refuse/toy_delay", category="should_refuse")
    task_data["expected_outcome"] = {"default": "refusal"}
    del task_data["answer"]
    assert tasks.load(write_task(task_data)).answer == ()


def test_a_trap_names_its_source(task_data, write_task):
    task_data.update(trap="output_grid", trap_source="BNG2.pl's n_steps against n_points")
    t = tasks.load(write_task(task_data))
    assert (t.trap, t.trap_source) == ("output_grid", "BNG2.pl's n_steps against n_points")


def _set(path, value):
    def mutate(d):
        *parents, last = path
        for key in parents:
            d = d[key]
        d[last] = value

    return mutate


def _drop(*path):
    def mutate(d):
        *parents, last = path
        for key in parents:
            d = d[key]
        del d[last]

    return mutate


@pytest.mark.parametrize(
    "mutate, message",
    [
        (_set(["engine"], "bngsim"), "unknown key 'engine'"),
        (_drop("request"), "missing 'request'"),
        (_drop("split"), "missing 'split'"),
        (_set(["category"], "fitting"), "category: one of timecourse, scan"),
        (_set(["id"], "timecourse/toy_peak"), "id: 'agent/<category>/<name>'"),
        (_set(["id"], "agent/timecourse/Toy-Peak"), "id: 'agent/<category>/<name>'"),
        (_set(["id"], "agent/scan/toy_peak"), "must name the category 'timecourse'"),
        (_set(["id"], "agent/timecourse/other"), "must match the file"),
        (_set(["model"], "toy_model"), "model: a mapping with one key, 'problem'"),
        (_set(["model"], {"problem": "toy_model", "file": "x.xml"}), "model: a mapping with one"),
        (_set(["model", "problem"], ""), "model.problem: the id of a problem spec"),
        (_set(["forms"], []), "forms: a non-empty list of sbml, bngl"),
        (_set(["forms"], ["sbml", "net"]), "forms: a non-empty list of sbml, bngl"),
        (_set(["forms"], ["sbml", "sbml"]), "forms: a non-empty list of sbml, bngl, each once"),
        (_set(["request"], "  "), "request: the request in plain language"),
        (_set(["answer"], {"peak": 1.0}), "answer: a list of values"),
        (_set(["answer"], []), "answer: at least one value"),
        (_drop("answer"), "missing 'answer': an engine is expected to answer"),
        (_set(["answer", 0, "name"], "peak value"), "answer[0].name: letters, digits and '_'"),
        (_set(["answer", 1, "name"], "peak_value"), "answer[1].name: 'peak_value' is named twice"),
        (_set(["answer", 0, "reference"], "0.0412"), "answer[0].reference: must be a finite"),
        (_set(["answer", 0, "reference"], float("nan")), "answer[0].reference: must be a finite"),
        (_set(["answer", 0, "reference"], True), "answer[0].reference: must be a finite"),
        (_set(["answer", 0, "rel_tol"], 0), "answer[0].rel_tol: must be a positive number"),
        (
            _set(["answer", 0, "rel_tol"], "1e-3"),
            "rel_tol: must be a positive number (YAML reads 1e-3 as text: write 1.0e-3",
        ),
        (_set(["answer", 0, "rel_tol"], "-1e-3"), "answer[0].rel_tol: must be a positive number$"),
        (_set(["answer", 0, "rel_tol"], "0.5e-3"), "answer[0].rel_tol: must be a positive number$"),
        (_set(["answer", 0, "rel_tol"], "inf"), "answer[0].rel_tol: must be a positive number$"),
        (_set(["answer", 0, "rel_tol"], "small"), "answer[0].rel_tol: must be a positive number$"),
        (
            _set(["answer", 0, "reference"], "4e-2"),
            "reference: must be a finite number (YAML reads 4e-2",
        ),
        (
            _set(["answer", 0, "reference"], "about 0.04"),
            "answer[0].reference: must be a finite number$",
        ),
        (_set(["answer", 1, "abs_tol"], -0.5), "answer[1].abs_tol: must be a positive number"),
        (_drop("answer", 0, "rel_tol"), "answer[0]: needs 'rel_tol', 'abs_tol' or both"),
        (
            _set(["answer", 0, "reference"], 0),
            "answer[0]: the tolerance at this reference is below",
        ),
        (_set(["answer", 0, "rel_tol"], 1e-17), "below the spacing of doubles there"),
        (_set(["answer", 1, "abs_tol"], 5e-324), "answer[1]: the tolerance at this reference"),
        (
            _set(
                ["answer", 0],
                {"name": "peak_value", "reference": 1e308, "rel_tol": 0.9, "abs_tol": 1e308},
            ),
            "or is not finite, so every value would",
        ),
        (_set(["model", "problem"], "  "), "model.problem: the id of a problem spec"),
        (
            _set(["answer", 0, "reference"], 1e-323),
            "answer[0]: the tolerance at this reference is below",
        ),
        (_set(["answer", 0, "reference"], 10**400), "answer[0].reference: must be a finite"),
        (_set(["answer", 0, "rel_tol"], 10**400), "answer[0].rel_tol: must be a positive number"),
        (_set(["answer", 0, "rel_tol"], 1), "answer[0].rel_tol: must be below 1"),
        (_set(["answer", 0, "name"], "peak_value\n"), "answer[0].name: letters, digits and '_'"),
        (_set(["id"], "agent/timecourse/toy_peak\n"), "id: 'agent/<category>/<name>'"),
        (_set(["answer", 0, "units"], "nM"), "answer[0]: unknown key 'units'"),
        (_set(["expected_outcome"], "answer"), "expected_outcome: a mapping with 'default'"),
        (_set(["expected_outcome"], {"bngsim": "answer"}), "expected_outcome: missing 'default'"),
        (_set(["expected_outcome", "tellurium"], "answer"), "unknown engine 'tellurium'"),
        (_set(["expected_outcome", "default"], "pass"), "expected_outcome.default: one of answer"),
        (
            _set(["expected_outcome", "amici"], "refusal"),
            "a refusal is expected only in category should_refuse",
        ),
        (_set(["trap"], "off_by_one"), "trap: 'none' or one of state_carried_between_runs"),
        (_set(["trap"], "output_grid"), "trap_source: a trap names where it is recorded"),
        (_set(["trap_source"], "an issue"), "trap_source: given without a trap"),
        (_set(["split"], "private"), "split: one of public, held_out"),
    ],
)
def test_invalid_field_is_refused_with_its_name(task_data, write_task, mutate, message):
    mutate(task_data)
    path = write_task(task_data, name="toy_peak", category="timecourse")
    pattern = message.replace("(", r"\(").replace("[", r"\[")
    with pytest.raises(tasks.TaskError) as err:
        tasks.load(path)
    import re

    assert any(re.search(pattern, problem) for problem in err.value.problems), err.value.problems


def test_a_should_refuse_task_expects_a_refusal_from_some_engine(task_data, write_task):
    task_data.update(id="agent/should_refuse/toy_delay", category="should_refuse")
    with pytest.raises(tasks.TaskError, match="a should_refuse task expects a refusal"):
        tasks.load(write_task(task_data))


def test_every_problem_reported_at_once(task_data, write_task):
    task_data["category"] = "fitting"
    task_data["split"] = "private"
    task_data["answer"][0]["rel_tol"] = 0
    with pytest.raises(tasks.TaskError) as err:
        tasks.load(write_task(task_data, name="toy_peak", category="timecourse"))
    assert len(err.value.problems) == 3


def test_the_example_of_the_task_issue_loads_once_its_tolerance_has_a_decimal_point(tmp_path):
    # Issue #13 writes `rel_tol: 1e-3`, which YAML reads as text.
    example = """\
id: agent/timecourse/kinetic_proofreading_peak
category: timecourse
model: {problem: kinetic_proofreading_hlavacek2001}
forms: [sbml, bngl]
request: >
  Simulate the model from its initial state for 100 seconds.
  Report the peak value of Obs_Tot_Dimers and the time at which it occurs.
answer:
  - {name: peak_value, reference: 0.0412, rel_tol: 1e-3}
  - {name: peak_time,  reference: 17.5, abs_tol: 0.5}
expected_outcome: {default: answer}
trap: none
split: public
"""
    d = tmp_path / "tasks" / "timecourse"
    d.mkdir(parents=True)
    path = d / "kinetic_proofreading_peak.yaml"
    path.write_text(example)
    with pytest.raises(tasks.TaskError, match=r"YAML reads 1e-3 as text: write 1\.0e-3"):
        tasks.load(path)
    path.write_text(example.replace("1e-3", "1.0e-3"))
    problems = [p.id for p in specs.load_all(REPO / "problems")]
    (loaded,) = tasks.load_all(tmp_path / "tasks", problems=problems)
    assert loaded.value("peak_value").rel_tol == 1e-3
    assert loaded.request.startswith("Simulate") and loaded.request.endswith("occurs.")


def test_a_file_that_cannot_be_read_is_reported_with_the_rest(task_data, write_task, tmp_path):
    task_data["split"] = "private"
    write_task(task_data)
    d = tmp_path / "tasks" / "scan"
    d.mkdir()
    (d / "bytes.yaml").write_bytes(b"id: \xff\xfe")
    (d / "folder.yaml").mkdir()
    with pytest.raises(tasks.TaskError) as err:
        tasks.load_all(tmp_path / "tasks")
    assert len(err.value.problems) == 3
    assert sum("cannot be read as UTF-8 text" in p for p in err.value.problems) == 2


@pytest.mark.parametrize("suffix", [".yml", ".YAML", ".Yml", ".YML"])
def test_a_task_under_another_spelling_of_the_suffix_is_not_skipped(
    task_data, write_task, tmp_path, suffix
):
    # Skipped, a held-out task saved as .yml would pass the check that keeps
    # those out of the repository.
    task_data["split"] = "held_out"
    path = write_task(task_data)
    path.rename(path.with_suffix(suffix))
    with pytest.raises(tasks.TaskError, match="a task file ends in .yaml"):
        tasks.load_all(tmp_path / "tasks")


def test_a_file_named_only_by_its_suffix_is_refused(task_data, write_task, tmp_path):
    path = write_task(task_data)
    (path.parent / ".yaml").write_text("")
    with pytest.raises(tasks.TaskError, match="a task file ends in .yaml"):
        tasks.load_all(tmp_path / "tasks")


def test_a_key_given_twice_in_a_task_is_refused(tmp_path, task_data, write_task):
    # PyYAML keeps the last: held_out, then public, would load as public.
    path = write_task(task_data)
    path.write_text(path.read_text().replace("split: public", "split: held_out\nsplit: public"))
    with pytest.raises(tasks.TaskError, match="key 'split' given more than once"):
        tasks.load(path)


@pytest.mark.parametrize(
    "reference, abs_tol, accepted",
    [
        (1.0e6, 1.0e-10, False),  # the doubles near 1e6 are 1.2e-10 apart
        (1.0e6, 2.0e-10, True),
        (1.0, 1.2e-16, True),  # below 1.0 the doubles are 1.1e-16 apart
        (1.0, 2.0**-53, True),  # exactly the gap to the next double below
        (0.0, 1.0e-300, True),  # a reference of 0 takes any absolute tolerance
        (-1.0e6, 1.0e-10, False),
    ],
)
def test_a_tolerance_has_to_reach_the_next_double(
    task_data, write_task, reference, abs_tol, accepted
):
    task_data["answer"][1] = {"name": "peak_time", "reference": reference, "abs_tol": abs_tol}
    path = write_task(task_data)
    if accepted:
        assert tasks.load(path).value("peak_time").abs_tol == abs_tol
    else:
        with pytest.raises(tasks.TaskError, match="below the spacing of doubles"):
            tasks.load(path)


def test_a_relative_tolerance_reaches_the_next_double_at_a_negative_reference(
    task_data, write_task
):
    task_data["answer"][0] = {"name": "peak_value", "reference": -1.0e6, "rel_tol": 1.0e-17}
    with pytest.raises(tasks.TaskError, match="below the spacing of doubles"):
        tasks.load(write_task(task_data))


def test_other_files_beside_the_tasks_are_left_alone(task_data, write_task, tmp_path):
    path = write_task(task_data)
    (path.parent / "README.md").write_text("notes")
    (path.parent / "toy_peak.yaml.bak").write_text("id: [")
    assert len(tasks.load_all(tmp_path / "tasks")) == 1


def test_a_directory_of_tasks_that_is_not_there_is_refused(tmp_path):
    # An empty list would let the check of the committed tasks pass on nothing.
    with pytest.raises(tasks.TaskError, match="no such directory of tasks"):
        tasks.load_all(tmp_path / "moved")
    (tmp_path / "a_file").write_text("")
    with pytest.raises(tasks.TaskError, match="no such directory of tasks"):
        tasks.load_all(tmp_path / "a_file")


def test_a_negative_reference_is_measured_by_its_size(task_data, write_task):
    task_data["answer"][0].update(reference=-0.0412)
    assert tasks.load(write_task(task_data)).value("peak_value").reference == -0.0412


def test_an_absolute_tolerance_may_be_as_wide_as_the_task_needs(task_data, write_task):
    task_data["answer"][1]["abs_tol"] = 5.0
    assert tasks.load(write_task(task_data)).value("peak_time").abs_tol == 5.0


def test_not_yaml_and_not_a_mapping_are_refused(tmp_path):
    d = tmp_path / "tasks" / "timecourse"
    d.mkdir(parents=True)
    (d / "broken.yaml").write_text("id: [unclosed")
    with pytest.raises(tasks.TaskError, match="not valid YAML"):
        tasks.load(d / "broken.yaml")
    (d / "list.yaml").write_text("- a\n- b\n")
    with pytest.raises(tasks.TaskError, match="must be a YAML mapping"):
        tasks.load(d / "list.yaml")


def test_load_all_sorts_by_id_and_checks_the_problem_exists(task_data, write_task, tmp_path):
    write_task(task_data)
    other = copy.deepcopy(VALID_TASK)
    other.update(id="agent/scan/toy_ec50", category="scan")
    write_task(other)
    root = tmp_path / "tasks"
    # Under tasks/z/, so that the order of the paths is not the order of the ids.
    third = copy.deepcopy(VALID_TASK)
    third.update(id="agent/perturbation/toy_dose", category="perturbation")
    write_task(third, directory=tmp_path / "tasks" / "z")
    loaded = tasks.load_all(root, problems=["toy_model"])
    assert [t.id for t in loaded] == [
        "agent/perturbation/toy_dose",
        "agent/scan/toy_ec50",
        "agent/timecourse/toy_peak",
    ]
    with pytest.raises(tasks.TaskError, match="model.problem 'toy_model' is not a problem spec"):
        tasks.load_all(root, problems=["another_model"])


def test_load_all_reports_every_broken_task(task_data, write_task, tmp_path):
    task_data["split"] = "private"
    write_task(task_data)
    other = copy.deepcopy(VALID_TASK)
    other.update(id="agent/scan/toy_ec50", category="fitting")
    write_task(other, category="scan")
    with pytest.raises(tasks.TaskError) as err:
        tasks.load_all(tmp_path / "tasks")
    assert len(err.value.problems) == 2


def test_a_held_out_task_is_refused_in_the_repository(task_data, write_task, tmp_path):
    task_data["split"] = "held_out"
    write_task(task_data)
    with pytest.raises(tasks.TaskError, match="a held-out task is kept out of this repository"):
        tasks.load_all(tmp_path / "tasks")
    assert len(tasks.load_all(tmp_path / "tasks", allow_held_out=True)) == 1


def test_two_files_with_one_id_are_refused(task_data, write_task, tmp_path):
    write_task(task_data, directory=tmp_path / "tasks" / "a")
    write_task(task_data, directory=tmp_path / "tasks" / "b")
    with pytest.raises(tasks.TaskError, match="duplicate id 'agent/timecourse/toy_peak'"):
        tasks.load_all(tmp_path / "tasks")


def test_committed_tasks_all_validate_against_the_committed_problems():
    problems = [p.id for p in specs.load_all(REPO / "problems")]
    loaded = tasks.load_all(REPO / "tasks", problems=problems)
    assert [t.id for t in loaded] == sorted(t.id for t in loaded)


def test_a_task_file_under_problems_would_be_read_as_a_problem_spec(task_data, tmp_path):
    # Why tasks have a directory of their own: every YAML file under problems/
    # is a problem spec, and a task there stops the job list being built.
    import yaml

    d = tmp_path / "problems"
    d.mkdir()
    (d / "toy_peak.yaml").write_text(yaml.safe_dump(task_data))
    with pytest.raises(specs.SpecError, match="unknown key 'request'"):
        specs.load_all(d)
    assert not (REPO / "tasks").is_relative_to(REPO / "problems")


# --------------------------------------------------------------------------- #
# Found in a fourth review
# --------------------------------------------------------------------------- #


def test_a_merge_key_loads_and_may_be_overridden(task_data, write_task):
    path = write_task(task_data)
    text = path.read_text().replace(
        "expected_outcome:\n  default: answer\n",
        "expected_outcome:\n  <<: &base {default: refusal}\n  default: answer\n",
    )
    assert "<<:" in text
    path.write_text(text)
    assert tasks.load(path).expected_for("bngsim") == "answer"


def test_a_key_that_cannot_be_hashed_is_a_task_error(tmp_path, task_data, write_task):
    path = write_task(task_data)
    path.write_text(path.read_text() + "? [a, b]\n: 1\n")
    with pytest.raises(tasks.TaskError, match="unhashable key"):
        tasks.load(path)
    with pytest.raises(tasks.TaskError):
        tasks.load_all(tmp_path / "tasks")


def test_a_key_given_twice_in_a_nested_mapping_is_refused(task_data, write_task):
    path = write_task(task_data)
    path.write_text(
        path.read_text().replace(
            "  problem: toy_model\n", "  problem: toy_model\n  problem: other\n"
        )
    )
    with pytest.raises(tasks.TaskError, match="key 'problem' given more than once"):
        tasks.load(path)


@pytest.mark.parametrize("name", [".yml", ".YAML", ".Yml"])
def test_a_file_named_only_by_any_spelling_of_the_suffix_is_refused(
    task_data, write_task, tmp_path, name
):
    path = write_task(task_data)
    (path.parent / name).write_text("")
    with pytest.raises(tasks.TaskError, match="a task file ends in .yaml"):
        tasks.load_all(tmp_path / "tasks")


@pytest.mark.parametrize(
    "field, written, hint",
    [
        ("reference", "1e3", "write 1.0e+3"),  # PyYAML reads 1.0e3 as text too
        ("reference", "1.5e3", "write 1.5e+3"),
        ("reference", " 4e-2 ", "write 4.0e-2"),
        ("rel_tol", "1e-3", "write 1.0e-3"),
        ("rel_tol", "1e0", None),  # 1.0 is no relative tolerance
        ("rel_tol", "-1e-3", None),
        ("reference", "1e400", None),  # not finite either way
        ("abs_tol", "0e0", None),
        ("abs_tol", "1e400", None),  # no width is infinite
    ],
)
def test_the_yaml_hint_names_a_form_that_reads_as_a_number_and_passes(
    task_data, write_task, field, written, hint
):
    entry = task_data["answer"][0] if field != "abs_tol" else task_data["answer"][1]
    entry[field] = written
    with pytest.raises(tasks.TaskError) as err:
        tasks.load(write_task(task_data))
    text = " ".join(err.value.problems)
    if hint is None:
        assert "YAML reads" not in text
    else:
        assert hint in text
        import yaml

        fixed = hint.removeprefix("write ")
        assert isinstance(yaml.safe_load(f"x: {fixed}")["x"], float)
