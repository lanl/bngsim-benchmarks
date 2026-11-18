"""Scoring agent jobs: one outcome per job from a fixed vocabulary, and rates
whose unit is the task. A wrong number reported as a result must never be
counted as anything else."""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest
from conftest import VALID_TASK

from bench import scoring, tasks
from bench.scoring import Cause, Outcome, Scored

# --------------------------------------------------------------------------- #
# The vocabulary
# --------------------------------------------------------------------------- #


def test_the_vocabulary_is_the_seven_outcomes_in_report_order():
    assert [str(o) for o in scoring.ALL] == [
        "correct",
        "correct_refusal",
        "silently_wrong",
        "unfaithful_answer",
        "wrong_refusal",
        "failure",
        "invalid",
    ]


def test_the_headline_and_the_completion_rate_share_no_outcome():
    assert scoring.SILENT == {Outcome.SILENTLY_WRONG, Outcome.UNFAITHFUL_ANSWER}
    assert scoring.COMPLETE == {Outcome.CORRECT, Outcome.CORRECT_REFUSAL}
    assert not scoring.SILENT & scoring.COMPLETE
    # Every outcome but the two that complete and `invalid` gets a cause.
    assert scoring.NEEDS_CAUSE == set(scoring.ALL) - scoring.COMPLETE - {Outcome.INVALID}


def test_a_tally_lists_every_outcome_zeros_included():
    counts = scoring.tally(["correct", Outcome.CORRECT, "silently_wrong"])
    assert list(counts) == [str(o) for o in scoring.ALL]
    assert counts["correct"] == 2 and counts["silently_wrong"] == 1 and counts["failure"] == 0
    with pytest.raises(ValueError):
        scoring.tally(["PASS"])


# --------------------------------------------------------------------------- #
# One value against its reference
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value, reference, rel_tol, abs_tol, expected",
    [
        (1.0005, 1.0, 1e-3, 0.0, 0.5),  # half the tolerance
        (1.5, 1.0, 0.5, 0.0, 1.0),  # on it, in numbers a double holds exactly
        (1.002, 1.0, 1e-3, 0.0, 2.0),
        (17.0, 17.5, 0.0, 0.5, 1.0),
        (-2.0, -1.0, 0.5, 0.0, 2.0),  # relative to the reference's size
        (0.0, 1.0, 1e-3, 1e-3, 500.0),  # both: |x - y| / (abs + rel·|y|)
        (20.0, 1.0, 1e-3, 0.0, 19000.0),  # wrong by a factor of 20
        (0.0, 0.0, 0.0, 1e-9, 0.0),
    ],
)
def test_a_miss_is_the_error_in_units_of_the_tolerance(
    value, reference, rel_tol, abs_tol, expected
):
    got = scoring.miss(value, reference, rel_tol=rel_tol, abs_tol=abs_tol)
    assert got == pytest.approx(expected, rel=1e-9)
    assert scoring.within(value, reference, rel_tol=rel_tol, abs_tol=abs_tol) == (got <= 1.0)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_a_value_that_is_not_finite_is_outside_any_tolerance(value):
    assert scoring.miss(value, 1.0, rel_tol=1e-3) == math.inf
    assert not scoring.within(value, 1.0, abs_tol=1e9)


def test_a_tolerance_that_allows_nothing_is_refused():
    with pytest.raises(ValueError, match="tolerance that is positive and finite at its reference"):
        scoring.miss(1.0, 1.0)
    with pytest.raises(ValueError, match="tolerance that is positive and finite at its reference"):
        scoring.miss(0.0, 0.0, rel_tol=1e-3)  # relative to a reference of 0
    with pytest.raises(ValueError, match="tolerance that is positive and finite at its reference"):
        scoring.miss(1.0, 1.0, rel_tol=-1e-3, abs_tol=1.0)
    with pytest.raises(ValueError, match="is not finite"):
        scoring.miss(1.0, math.nan, rel_tol=1e-3)


def test_within_tolerance_means_what_the_parity_core_means():
    # The correctness jobs' mixed tolerance, with the reference on the right:
    # the vendored function is what is called, and nothing is added to it.
    from bench._vendor.parity_core.fingerprint import fingerprint_distance

    def one(x):
        return {
            "n_time": 1,
            "t_last": 1.0,
            "vars": {"v": dict.fromkeys(("last", "min", "max", "mean"), x)},
        }

    rng = np.random.default_rng(0)
    for x, y in rng.normal(size=(50, 2)):
        assert scoring.miss(x, y, rel_tol=1e-2, abs_tol=1e-3) == fingerprint_distance(
            one(float(x)), one(float(y)), 1e-2, 1e-3
        )


# --------------------------------------------------------------------------- #
# The answer file
# --------------------------------------------------------------------------- #

GOOD = {
    "outcome": "answer",
    "values": {"peak_value": 0.0412, "peak_time": 17.5},
    "checked": True,
    "check": "re-ran at rtol 1e-10; values moved by less than 1e-6 relative",
    "notes": "",
}


def test_the_answer_file_of_the_harness_issue_parses():
    answer, problems = scoring.parse_answer(json.dumps(GOOD))
    assert problems == []
    assert answer.outcome == "answer" and answer.checked is True
    assert dict(answer.values) == {"peak_value": 0.0412, "peak_time": 17.5}
    assert scoring.parse_answer(GOOD)[0] == answer  # parsed already, or as text


@pytest.mark.parametrize(
    "doc, message",
    [
        (None, "no answer file"),
        ("{not json", "not valid JSON"),
        (b"\xff\xfe", "not valid JSON"),
        ("[" * 100_000 + "]" * 100_000, "not valid JSON: nested too deeply"),
        ("[1, 2]", "must be a JSON object"),
        ({**GOOD, "outcome": "done"}, "outcome: one of answer, refusal, failure"),
        ({**GOOD, "outcome": ["answer"]}, "outcome: one of answer, refusal, failure"),
        ({"values": {}}, "outcome: one of answer, refusal, failure"),
    ],
)
def test_a_file_that_cannot_be_read_as_an_answer_says_why(doc, message):
    answer, problems = scoring.parse_answer(doc)
    assert answer is None
    assert len(problems) == 1 and message in problems[0]


@pytest.mark.parametrize(
    "change, slip",
    [
        ({"engine": "bngsim"}, "unknown key 'engine'"),
        ({"checked": "true"}, "checked: not true or false"),
        ({"notes": 3}, "notes: not text"),
        ({"values": {**GOOD["values"], "units": "nM"}}, "values.units: not a number"),
        ({"values": {**GOOD["values"], "peak_time": "about 17"}}, "values.peak_time: not a number"),
        (
            {"values": {**GOOD["values"], "peak_time": {"value": 17.5}}},
            "values.peak_time: not a number",
        ),
        ({"values": {**GOOD["values"], "peak_time": [17.5, 18]}}, "values.peak_time: not a number"),
        ({"check": 0}, "check: not text"),
        ({"notes": False}, "notes: not text"),
        ({"notes": []}, "notes: not text"),
        ({"values": {**GOOD["values"], "flag": True}}, "values.flag: not a number"),
    ],
)
def test_a_slip_is_kept_and_does_not_stop_the_file_being_read(change, slip):
    answer, problems = scoring.parse_answer({**GOOD, **change})
    assert problems == [] and answer.outcome == "answer"
    assert answer.slips == (slip,)
    assert dict(answer.values)["peak_value"] == 0.0412


@pytest.mark.parametrize(
    "written, slip",
    [
        ("17.5", "a number written as text"),
        (" 17.5\n", "a number written as text"),
        ([17.5], "a number written as a list"),
    ],
)
def test_a_number_in_quotes_or_in_a_list_is_read_as_the_number(written, slip):
    doc = {**GOOD, "values": {"peak_value": 0.0412, "peak_time": written}}
    answer, problems = scoring.parse_answer(doc)
    assert problems == []
    assert dict(answer.values) == {"peak_value": 0.0412, "peak_time": 17.5}
    assert answer.slips == (f"values.peak_time: {slip}",)


def test_a_missing_or_null_note_is_no_slip():
    for doc in ({"outcome": "refusal"}, {"outcome": "refusal", "check": None, "notes": None}):
        answer, problems = scoring.parse_answer(doc)
        assert problems == [] and answer.slips == () and answer.notes == ""


def test_what_an_agent_wrote_is_quoted_short():
    for long in ("k" * 100_000, "\U000e0001" * 100_000, "\x00" * 100_000):
        answer, _ = scoring.parse_answer({**GOOD, long: 1})
        assert len(answer.slips[0]) < 110 and answer.slips[0].endswith("...")
        answer, _ = scoring.parse_answer({**GOOD, "values": {long: "nM"}})
        assert len(answer.slips[0]) < 110 and answer.slips[0].startswith("values.")


def test_a_file_of_many_slips_lists_twenty_and_counts_the_rest(task):
    doc = {**GOOD, **{f"key{i}": i for i in range(1000)}}
    answer, _ = scoring.parse_answer(doc)
    assert len(answer.slips) == 21 and answer.slips[-1] == "and 980 more"
    values = {"peak_value": 0.5, "peak_time": 17.5, **{f"v{i}": float(i) for i in range(1000)}}
    g = scoring.grade(task, "bngsim", {"outcome": "answer", "values": values})
    assert g.outcome is Outcome.SILENTLY_WRONG and len(g.problems) <= 41


def test_the_claim_of_a_check_is_true_only_where_the_file_says_true():
    for claimed in ("true", 1, "yes", [True]):
        answer, _ = scoring.parse_answer({**GOOD, "checked": claimed})
        assert answer.checked is False and "checked: not true or false" in answer.slips
    answer, _ = scoring.parse_answer({**GOOD, "check": "re-ran", "notes": "none"})
    assert (answer.check, answer.notes) == ("re-ran", "none")
    # What is not text is dropped, not kept under the name of a note.
    answer, _ = scoring.parse_answer({**GOOD, "check": ["re-ran"], "notes": 3})
    assert (answer.check, answer.notes) == ("", "")


def test_a_number_longer_than_python_converts_is_infinite_and_the_file_is_still_read(task):
    # Python refuses an integer of more than 4300 digits. The file is valid
    # JSON, and as text or as parsed content it grades the same.
    digits = "1" + "0" * 5000
    text = f'{{"outcome": "answer", "values": {{"peak_value": {digits}, "peak_time": -{digits}}}}}'
    answer, problems = scoring.parse_answer(text)
    assert problems == []
    assert dict(answer.values) == {"peak_value": math.inf, "peak_time": -math.inf}
    assert scoring.grade(task, "bngsim", text).outcome is Outcome.SILENTLY_WRONG
    parsed = {"outcome": "answer", "values": {"peak_value": 10**5000, "peak_time": 17.5}}
    assert scoring.grade(task, "bngsim", parsed).outcome is Outcome.SILENTLY_WRONG
    # Under a stray key it does not spoil a right answer.
    right = '"values": {"peak_value": 0.0412, "peak_time": 17.5}'
    stray = f'{{"outcome": "answer", {right}, "n": {digits}}}'
    assert scoring.grade(task, "bngsim", stray).outcome is Outcome.CORRECT


def test_text_with_a_byte_order_mark_grades_as_its_bytes_do(task):
    wrong = {"outcome": "answer", "values": {"peak_value": 0.5, "peak_time": 17.5}}
    text = "\ufeff" + json.dumps(wrong)
    for doc in (text, text.encode("utf-8"), text[1:].encode("utf-16")):
        assert scoring.grade(task, "bngsim", doc).outcome is Outcome.SILENTLY_WRONG


def test_a_number_no_float_holds_is_read_as_infinite():
    zeros = "0" * 400
    text = f'{{"outcome": "answer", "values": {{"up": 1{zeros}, "down": -1{zeros}, "big": 1e400}}}}'
    answer, problems = scoring.parse_answer(text)
    assert problems == []
    assert dict(answer.values) == {"up": math.inf, "down": -math.inf, "big": math.inf}


def test_nan_in_an_answer_file_is_read_and_then_graded_as_wrong(task):
    text = '{"outcome": "answer", "values": {"peak_value": NaN, "peak_time": 17.5}}'
    answer, problems = scoring.parse_answer(text)
    assert problems == [] and math.isnan(dict(answer.values)["peak_value"])
    assert scoring.grade(task, "bngsim", text).outcome is Outcome.SILENTLY_WRONG


# --------------------------------------------------------------------------- #
# Grading one job
# --------------------------------------------------------------------------- #


@pytest.fixture
def task(task_data, write_task):
    return tasks.load(write_task(task_data))


@pytest.fixture
def refusal_task(task_data, write_task):
    """A model with a construct that only roadrunner simulates faithfully."""
    task_data.update(id="agent/should_refuse/toy_delay", category="should_refuse")
    task_data["expected_outcome"] = {"default": "refusal", "roadrunner": "answer"}
    return tasks.load(write_task(task_data))


def _answer(**values):
    return {"outcome": "answer", "values": values}


@pytest.mark.parametrize(
    "engine, answer, outcome",
    [
        # The engine can do the task.
        ("roadrunner", _answer(peak_value=0.0412, peak_time=17.5), Outcome.CORRECT),
        ("roadrunner", _answer(peak_value=0.5, peak_time=17.5), Outcome.SILENTLY_WRONG),
        ("roadrunner", {"outcome": "refusal", "notes": "delays"}, Outcome.WRONG_REFUSAL),
        ("roadrunner", {"outcome": "failure"}, Outcome.FAILURE),
        ("roadrunner", None, Outcome.INVALID),
        # It cannot, and the right outcome is a refusal.
        ("bngsim", {"outcome": "refusal", "notes": "delays"}, Outcome.CORRECT_REFUSAL),
        ("bngsim", _answer(peak_value=0.0412, peak_time=17.5), Outcome.UNFAITHFUL_ANSWER),
        ("bngsim", _answer(peak_value=0.5), Outcome.UNFAITHFUL_ANSWER),
        ("bngsim", {"outcome": "failure"}, Outcome.FAILURE),
        ("bngsim", "{not json", Outcome.INVALID),
    ],
)
def test_each_job_gets_the_outcome_of_what_was_expected_and_what_was_done(
    refusal_task, engine, answer, outcome
):
    assert scoring.grade(refusal_task, engine, answer).outcome is outcome


def test_an_answer_within_every_tolerance_is_correct_and_keeps_its_misses(task):
    g = scoring.grade(task, "bngsim", _answer(peak_value=0.0412 * 1.0005, peak_time=17.25))
    assert g.outcome is Outcome.CORRECT
    assert dict(g.misses) == pytest.approx({"peak_value": 0.5, "peak_time": 0.5})
    assert g.worst_miss == pytest.approx(0.5)


def test_one_value_outside_its_tolerance_makes_the_job_silently_wrong(task):
    g = scoring.grade(task, "bngsim", _answer(peak_value=0.0412, peak_time=18.5))
    assert g.outcome is Outcome.SILENTLY_WRONG
    assert dict(g.misses) == pytest.approx({"peak_value": 0.0, "peak_time": 2.0})


def test_a_value_on_its_tolerance_is_within_it(task):
    assert scoring.grade(task, "bngsim", _answer(peak_value=0.0412, peak_time=18.0)).outcome is (
        Outcome.CORRECT
    )


def test_a_value_that_is_not_finite_is_silently_wrong(task):
    g = scoring.grade(
        task, "bngsim", '{"outcome":"answer","values":{"peak_value":NaN,"peak_time":17.5}}'
    )
    assert g.outcome is Outcome.SILENTLY_WRONG and g.worst_miss == math.inf


def test_an_answer_that_leaves_out_a_value_and_is_right_in_the_rest_is_invalid(task):
    g = scoring.grade(task, "bngsim", _answer(peak_value=0.0412))
    assert g.outcome is Outcome.INVALID
    assert g.problems == ("values: missing 'peak_time'",)
    assert dict(g.misses) == {"peak_value": 0.0}
    # A value that cannot be read as a number is a value left out.
    words = {"outcome": "answer", "values": {"peak_value": 0.0412, "peak_time": "soon"}}
    g = scoring.grade(task, "bngsim", words)
    assert g.outcome is Outcome.INVALID
    assert g.problems == ("values: missing 'peak_time'", "values.peak_time: not a number")
    assert scoring.grade(task, "bngsim", {"outcome": "answer"}).outcome is Outcome.INVALID
    # A value under a name the task does not ask for is one too, and is named.
    g = scoring.grade(task, "bngsim", _answer(peak_value=0.0412, Peak_Time=17.5))
    assert g.outcome is Outcome.INVALID
    assert g.problems == (
        "values: missing 'peak_time'",
        "values.Peak_Time: not a value the task asks for",
    )


def test_a_wrong_answer_says_which_values_it_left_out(task):
    g = scoring.grade(task, "bngsim", _answer(peak_value=0.5))
    assert g.outcome is Outcome.SILENTLY_WRONG
    assert g.problems == ("values: missing 'peak_time'",)


@pytest.mark.parametrize(
    "change",
    [
        {"values": {"peak_value": 0.5}},  # the other value left out
        {"engine": "bngsim"},
        {"check": None, "notes": None},
        {"checked": "true"},
        {"values": {"peak_value": 0.5, "peak_time": 17.5, "units": "nM"}},
        {"values": {"peak_value": 0.5, "peak_time": "17.5"}},
        {"values": {"peak_value": "0.5", "peak_time": 17.5}},  # the wrong number in quotes
        {"values": {"peak_value": [0.5], "peak_time": 17.5}},
        {"values": {"peak_value": " 5e-1 "}},
    ],
    ids=[
        "a-value-left-out",
        "a-stray-key",
        "null-notes",
        "checked-as-text",
        "a-stray-value",
        "a-value-as-text",
        "the-wrong-number-in-quotes",
        "the-wrong-number-in-a-list",
        "in-quotes-and-another-left-out",
    ],
)
def test_a_wrong_number_is_silently_wrong_whatever_else_is_the_matter_with_the_file(task, change):
    # Graded invalid, it would leave the headline, and an agent unsure of one
    # value could leave by omitting another.
    wrong = {"outcome": "answer", "values": {"peak_value": 0.5, "peak_time": 17.5}}
    g = scoring.grade(task, "bngsim", {**wrong, **change})
    assert g.outcome is Outcome.SILENTLY_WRONG
    assert dict(g.misses)["peak_value"] > 1e4


_WRONG = '"values":{"peak_value":0.5,"peak_time":17.5}'
WRONG_THEN_RIGHT = [
    # JSON parsers keep the last of a key given twice.
    '{"outcome":"answer",' + _WRONG + ',"values":{"peak_value":0.0412,"peak_time":17.5}}',
    '{"outcome":"answer","values":{"peak_value":0.5,"peak_value":0.0412,"peak_time":17.5}}',
    '{"outcome":"answer",' + _WRONG + ',"outcome":"failure"}',
    '{"outcome":"refusal",' + _WRONG + ',"outcome":"answer"}',
]


@pytest.mark.parametrize("text", WRONG_THEN_RIGHT)
def test_a_key_given_twice_is_read_both_ways(task, text):
    g = scoring.grade(task, "bngsim", text)
    assert g.outcome is Outcome.SILENTLY_WRONG
    assert any("given more than once" in p for p in g.problems)
    assert dict(g.misses)["peak_value"] > 1e4


def test_an_answer_given_beside_a_refusal_stands_where_a_refusal_was_expected(refusal_task):
    text = '{"outcome":"answer","values":{"peak_value":0.5},"outcome":"refusal"}'
    g = scoring.grade(refusal_task, "bngsim", text)
    assert g.outcome is Outcome.UNFAITHFUL_ANSWER and "outcome: given more than once" in g.problems
    twice = '{"outcome":"refusal","outcome":"refusal","notes":"a delay"}'
    assert scoring.grade(refusal_task, "bngsim", twice).outcome is Outcome.CORRECT_REFUSAL


@pytest.mark.parametrize("values", [None, [], "n/a", 0])
def test_a_refusal_or_a_failure_need_not_give_values(task, refusal_task, values):
    # An agent with nothing to report writes null. Graded invalid, a correct
    # refusal would leave the completion rate, for the engines that refuse.
    refusal = {"outcome": "refusal", "values": values, "notes": "a delay"}
    g = scoring.grade(refusal_task, "bngsim", refusal)
    assert g.outcome is Outcome.CORRECT_REFUSAL
    assert g.problems == (() if values is None else ("values: not an object",))
    failure = {"outcome": "failure", "values": values}
    assert scoring.grade(task, "bngsim", failure).outcome is Outcome.FAILURE
    # An answer has to give them where one was expected; where a refusal was,
    # any answer is unfaithful, whatever it gives.
    g = scoring.grade(task, "bngsim", {"outcome": "answer", "values": values})
    assert g.outcome is Outcome.INVALID
    g = scoring.grade(refusal_task, "bngsim", {"outcome": "answer", "values": values})
    assert g.outcome is Outcome.UNFAITHFUL_ANSWER


def test_numbers_beside_a_refusal_or_a_failure_are_not_read(task, refusal_task):
    wrong = {"peak_value": 0.5, "peak_time": 17.5}
    g = scoring.grade(task, "bngsim", {"outcome": "failure", "values": wrong})
    assert (g.outcome, g.misses) == (Outcome.FAILURE, ())
    g = scoring.grade(task, "bngsim", {"outcome": "refusal", "values": wrong})
    assert (g.outcome, g.misses) == (Outcome.WRONG_REFUSAL, ())
    g = scoring.grade(refusal_task, "bngsim", {"outcome": "refusal", "values": wrong})
    assert (g.outcome, g.misses) == (Outcome.CORRECT_REFUSAL, ())


def test_slips_are_kept_on_every_grade(task, refusal_task):
    stray = {"engine": "bngsim"}
    slip = ("unknown key 'engine'",)
    assert scoring.grade(task, "bngsim", {"outcome": "failure", **stray}).problems == slip
    assert scoring.grade(task, "bngsim", {"outcome": "refusal", **stray}).problems == slip
    assert scoring.grade(refusal_task, "bngsim", {"outcome": "refusal", **stray}).problems == slip
    g = scoring.grade(refusal_task, "bngsim", {**_answer(peak_value=0.5), **stray})
    assert (g.outcome, g.problems) == (Outcome.UNFAITHFUL_ANSWER, slip)
    wrong = {**_answer(peak_value=0.5, peak_time=17.5), **stray}
    g = scoring.grade(task, "bngsim", wrong, engines_used=["bngsim", "roadrunner"])
    assert g.problems == ("the job was to use bngsim, and also used: roadrunner", *slip)


def test_a_slip_does_not_spoil_a_right_answer_or_a_refusal(task, refusal_task):
    right = {
        "outcome": "answer",
        "values": {"peak_value": 0.0412, "peak_time": 17.5, "units": "nM"},
    }
    g = scoring.grade(task, "bngsim", {**right, "engine": "bngsim"})
    assert g.outcome is Outcome.CORRECT
    assert g.problems == ("unknown key 'engine'", "values.units: not a number")
    refusal = {"outcome": "refusal", "check": None, "notes": "a delay"}
    assert scoring.grade(refusal_task, "bngsim", refusal).outcome is Outcome.CORRECT_REFUSAL


def test_a_number_too_large_or_nested_too_deeply_gets_a_verdict_not_a_crash(task):
    zeros = "0" * 400
    big = f'{{"outcome": "answer", "values": {{"peak_value": 1{zeros}, "peak_time": 17.5}}}}'
    assert scoring.grade(task, "bngsim", big).outcome is Outcome.SILENTLY_WRONG
    deep = "[" * 100_000 + "]" * 100_000
    assert scoring.grade(task, "bngsim", deep).outcome is Outcome.INVALID


def test_a_value_the_task_did_not_ask_for_is_ignored(task):
    g = scoring.grade(task, "bngsim", _answer(peak_value=0.0412, peak_time=17.5, extra=1e9))
    assert g.outcome is Outcome.CORRECT and "extra" not in dict(g.misses)


def test_a_job_that_used_another_engine_is_invalid_whatever_it_answered(task):
    right = _answer(peak_value=0.0412, peak_time=17.5)
    assert scoring.grade(task, "bngsim", right, engines_used=["bngsim"]).outcome is Outcome.CORRECT
    assert scoring.grade(task, "bngsim", right, engines_used={"bngsim"}).outcome is Outcome.CORRECT
    assert scoring.grade(task, "bngsim", right, engines_used=iter(["bngsim"])).outcome is (
        Outcome.CORRECT
    )
    for used in (["roadrunner"], ["bngsim", "roadrunner"]):
        g = scoring.grade(task, "bngsim", right, engines_used=used)
        assert g.outcome is Outcome.INVALID
        assert g.problems == ("the job was to use bngsim, and also used: roadrunner",)
    g = scoring.grade(task, "bngsim", right, engines_used=["copasi", "bngsim", "amici"])
    assert g.problems == ("the job was to use bngsim, and also used: amici, copasi",)
    # Before anything else: a failure reported after running another engine.
    g = scoring.grade(task, "bngsim", {"outcome": "failure"}, engines_used=["roadrunner"])
    assert g.outcome is Outcome.INVALID


def test_a_wrong_number_from_a_job_that_used_another_engine_is_invalid_and_keeps_its_miss(task):
    # Issue #15: the grader marks such a job invalid. This is the one way a
    # wrong number leaves the headline, and the grade still says how wrong.
    wrong = _answer(peak_value=0.5, peak_time=17.5)
    g = scoring.grade(task, "bngsim", wrong, engines_used=["bngsim", "roadrunner"])
    assert g.outcome is Outcome.INVALID and g.worst_miss > 1e4


def test_an_answer_from_a_job_that_did_not_import_its_engine_is_invalid(task):
    right = _answer(peak_value=0.0412, peak_time=17.5)
    g = scoring.grade(task, "bngsim", right, engines_used=[])
    assert g.outcome is Outcome.INVALID
    assert g.problems == ("the job answered without importing bngsim",)


@pytest.mark.parametrize(
    "used, error, message",
    [
        ("bngsim", TypeError, "a collection of engine names"),
        (b"bngsim", TypeError, "a collection of engine names"),
        (["numpy"], ValueError, "engines_used: 'numpy' is not one of bngsim"),
        (["BNGSim"], ValueError, "engines_used: 'BNGSim' is not one of"),
        (["tellurium"], ValueError, "engines_used: 'tellurium' is not one of"),
        ([None], ValueError, "engines_used: 'None' is not one of"),
    ],
)
def test_an_audit_that_names_no_engine_of_the_study_is_the_harnesss_mistake(
    task, used, error, message
):
    right = _answer(peak_value=0.0412, peak_time=17.5)
    with pytest.raises(error, match=message):
        scoring.grade(task, "bngsim", right, engines_used=used)


def test_a_job_that_imported_no_engine_is_graded_on_what_it_wrote(task, refusal_task):
    # It stopped at a missing compiler, or read the model and refused.
    assert scoring.grade(task, "bngsim", {"outcome": "failure"}, engines_used=[]).outcome is (
        Outcome.FAILURE
    )
    refusal = {"outcome": "refusal", "notes": "a delay"}
    assert scoring.grade(refusal_task, "bngsim", refusal, engines_used=[]).outcome is (
        Outcome.CORRECT_REFUSAL
    )


def test_the_agents_own_claim_of_a_check_is_kept_whatever_the_outcome(task, refusal_task):
    answer = {**_answer(peak_value=0.5, peak_time=17.5), "checked": True}
    assert scoring.grade(task, "bngsim", answer).checked is True
    assert scoring.grade(task, "bngsim", _answer(peak_value=0.5, peak_time=17.5)).checked is False
    claimed = {"checked": True}
    cases = [
        (task, {"outcome": "answer", "values": {"peak_value": 0.0412}, **claimed}, None),  # invalid
        (task, {**answer}, []),  # invalid: answered without its engine
        (task, {**answer}, ["roadrunner"]),  # invalid: another engine
        (task, {"outcome": "refusal", **claimed}, None),
        (task, {"outcome": "failure", **claimed}, None),
        (refusal_task, {"outcome": "refusal", **claimed}, None),
        (refusal_task, {**answer}, None),  # unfaithful
    ]
    for t, doc, used in cases:
        assert scoring.grade(t, "bngsim", doc, engines_used=used).checked is True, doc


def test_an_engine_the_study_does_not_have_is_refused(task):
    with pytest.raises(ValueError, match="unknown engine 'tellurium'"):
        scoring.grade(task, "tellurium", None)


# --------------------------------------------------------------------------- #
# Rates, with the task as the unit
# --------------------------------------------------------------------------- #


def _jobs(engine, outcomes_by_task, *, arm="prior", category="timecourse"):
    """``{task: [outcome, ...]}`` as graded jobs, one per repetition."""
    return [
        Scored(task, category, engine, arm, "agent-a", rep, Outcome(outcome))
        for task, outcomes in outcomes_by_task.items()
        for rep, outcome in enumerate(outcomes)
    ]


def test_a_decimal_on_its_tolerance_can_round_to_either_side():
    # 0.4 against 0.3 with abs_tol 0.1 is on the tolerance in decimal and just
    # past it in doubles. A task's tolerance is not set at a value an agent may
    # land on exactly.
    assert scoring.miss(0.4, 0.3, abs_tol=0.1) == pytest.approx(1.0)
    assert scoring.miss(0.4, 0.3, abs_tol=0.1) > 1.0


def test_a_tasks_rate_is_the_fraction_of_its_own_repetitions():
    jobs = _jobs(
        "bngsim", {"t1": ["correct", "silently_wrong", "silently_wrong"], "t2": ["correct"]}
    )
    assert scoring.task_rates(jobs, scoring.SILENT) == pytest.approx({"t1": 2 / 3, "t2": 0.0})
    assert scoring.task_rates(jobs, scoring.COMPLETE) == pytest.approx({"t1": 1 / 3, "t2": 1.0})


def test_the_task_is_the_unit_not_the_job():
    # Nine wrong answers to one task and one right answer to another: half the
    # tasks were done badly, not nine jobs in ten.
    jobs = _jobs("bngsim", {"t1": ["silently_wrong"] * 9, "t2": ["correct"]})
    r = scoring.rate(jobs, scoring.SILENT)
    assert r.value == pytest.approx(0.5)
    assert (r.tasks, r.jobs) == (2, 10)


def test_an_unfaithful_answer_counts_in_the_headline_and_a_refusal_in_completion():
    jobs = _jobs(
        "bngsim", {"t1": ["unfaithful_answer"], "t2": ["correct_refusal"], "t3": ["failure"]}
    )
    assert scoring.rate(jobs, scoring.SILENT).value == pytest.approx(1 / 3)
    assert scoring.rate(jobs, scoring.COMPLETE).value == pytest.approx(1 / 3)


def test_the_interval_resamples_whole_tasks_and_brackets_the_rate():
    per_task = {f"t{i}": ["silently_wrong"] if i < 3 else ["correct"] for i in range(10)}
    r = scoring.rate(_jobs("bngsim", per_task), scoring.SILENT)
    assert r.value == pytest.approx(0.3)
    assert 0.0 <= r.low < r.value < r.high <= 1.0
    # The bootstrap by hand, from the same seed.
    values = np.array([1.0] * 3 + [0.0] * 7)
    idx = np.random.default_rng(0).integers(0, 10, size=(2000, 10))
    low, high = np.quantile(values[idx].mean(axis=1), [0.025, 0.975])
    assert (r.low, r.high) == (low, high)


def test_the_same_jobs_give_the_same_interval():
    per_task = {
        f"t{i}": ["silently_wrong", "correct"] if i % 3 else ["correct"] * 2 for i in range(12)
    }
    jobs = _jobs("bngsim", per_task)
    assert scoring.rate(jobs, scoring.SILENT) == scoring.rate(jobs, scoring.SILENT)
    # And another seed resamples other tasks: on rates fine enough to tell.
    fine = _jobs(
        "bngsim", {f"t{i}": ["silently_wrong"] * i + ["correct"] * (40 - i) for i in range(40)}
    )
    assert scoring.rate(fine, scoring.SILENT, seed=1) != scoring.rate(fine, scoring.SILENT)


def test_tasks_that_all_agree_have_an_interval_of_no_width():
    # What a bootstrap over tasks gives, and a limit of it: no silent error in
    # 30 tasks does not show that the rate is 0. `rate` says so.
    r = scoring.rate(_jobs("bngsim", {f"t{i}": ["correct"] * 5 for i in range(30)}), scoring.SILENT)
    assert (r.value, r.low, r.high) == (0.0, 0.0, 0.0)


def test_the_order_of_the_jobs_does_not_change_a_rate_or_its_interval():
    per_task = {f"t{i}": ["silently_wrong"] * (i % 4) + ["correct"] * 3 for i in range(17)}
    jobs = _jobs("bngsim", per_task)
    shuffled = [jobs[i] for i in np.random.default_rng(3).permutation(len(jobs))]
    assert scoring.rate(shuffled, scoring.SILENT) == scoring.rate(jobs, scoring.SILENT)
    assert scoring.rate(iter(jobs), scoring.SILENT) == scoring.rate(jobs, scoring.SILENT)
    other = _jobs("roadrunner", {t: list(reversed(o)) for t, o in per_task.items()})
    assert scoring.paired_difference(shuffled, other[::-1], scoring.SILENT) == (
        scoring.paired_difference(jobs, other, scoring.SILENT)
    )


@pytest.mark.parametrize(
    "kw, message",
    [
        ({"seed": None}, "seed: a whole number from 0"),
        ({"seed": -1}, "seed: a whole number from 0"),
        ({"seed": True}, "seed: a whole number from 0"),
        ({"seed": 0.5}, "seed: a whole number from 0"),
        ({"n_boot": 0}, "n_boot: a positive integer"),
        ({"n_boot": True}, "n_boot: a positive integer"),
        ({"confidence": 1.5}, "confidence: between 0 and 1"),
        ({"confidence": 0}, "confidence: between 0 and 1"),
        ({"confidence": 1}, "confidence: between 0 and 1"),
        ({"confidence": "0.95"}, "confidence: between 0 and 1"),
        ({"confidence": None}, "confidence: between 0 and 1"),
    ],
)
def test_the_bootstrap_refuses_settings_that_make_no_interval(kw, message):
    jobs = _jobs("bngsim", {"t1": ["correct"], "t2": ["failure"]})
    with pytest.raises(ValueError, match=message):
        scoring.rate(jobs, scoring.SILENT, **kw)
    with pytest.raises(ValueError, match=message):
        scoring.paired_difference(jobs, jobs, scoring.SILENT, **kw)


def test_one_task_has_a_rate_and_no_interval():
    r = scoring.rate(_jobs("bngsim", {"t1": ["correct", "failure"]}), scoring.COMPLETE)
    assert (r.value, r.low, r.high, r.tasks) == (0.5, None, None, 1)


def test_no_jobs_is_refused():
    with pytest.raises(ValueError, match="no jobs"):
        scoring.rate([], scoring.SILENT)


# --------------------------------------------------------------------------- #
# Comparing engines on the same tasks
# --------------------------------------------------------------------------- #


def test_engines_are_compared_task_by_task():
    a = _jobs("bngsim", {"t1": ["correct"], "t2": ["correct"], "t3": ["silently_wrong"]})
    b = _jobs("roadrunner", {"t1": ["silently_wrong"], "t2": ["correct"], "t3": ["silently_wrong"]})
    d = scoring.paired_difference(a + a, b, scoring.SILENT)  # unequal repetitions
    assert d.value == pytest.approx((0 - 1 + 0 + 0) / 3)
    assert (d.tasks, d.jobs) == (3, 9)  # the jobs of both sides


def test_a_numpy_integer_is_an_integer():
    per_task = {f"t{i}": ["silently_wrong"] * (i % 3) + ["correct"] for i in range(9)}
    jobs = _jobs("bngsim", per_task)
    plain = scoring.rate(jobs, scoring.SILENT, seed=3, n_boot=100)
    assert scoring.rate(jobs, scoring.SILENT, seed=np.int64(3), n_boot=np.int32(100)) == plain


def test_two_tasks_are_enough_for_an_interval():
    r = scoring.rate(_jobs("bngsim", {"t1": ["silently_wrong"], "t2": ["correct"]}), scoring.SILENT)
    assert (r.value, r.low, r.high) == (0.5, 0.0, 1.0)


def test_the_outcomes_counted_are_taken_once_whatever_they_come_in():
    per_task = {"t1": ["silently_wrong"], "t2": ["unfaithful_answer"], "t3": ["correct"]}
    jobs = _jobs("bngsim", per_task)
    want = scoring.rate(jobs, scoring.SILENT)
    assert want.value == pytest.approx(2 / 3)
    # A generator would be spent on the first job, and the rest counted as 0.
    assert scoring.rate(jobs, (o for o in scoring.SILENT)) == want
    assert scoring.rate(jobs, ["silently_wrong", "unfaithful_answer"]) == want
    both = scoring.paired_difference(iter(jobs), iter(jobs), (o for o in scoring.SILENT))
    assert (both.value, both.tasks, both.jobs) == (0.0, 3, 6)
    with pytest.raises(TypeError, match="a collection of outcomes, not one name"):
        scoring.rate(jobs, "silently_wrong")
    with pytest.raises(ValueError):
        scoring.rate(jobs, ["wrong"])


def test_the_table_takes_its_categories_once_and_passes_every_setting_on():
    study = _study()
    kept = scoring.summarize(study, exclude_categories=(c for c in ["should_refuse"]))
    assert kept == scoring.summarize(study, exclude_categories=["should_refuse"])
    assert all(c.counts["correct_refusal"] == 0 for c in kept)
    with pytest.raises(TypeError, match="a collection of categories, not one name"):
        scoring.summarize(study, exclude_categories="should_refuse")
    per_task = {f"t{i}": ["silently_wrong"] * i + ["correct"] * (40 - i) for i in range(40)}
    fine = _jobs("bngsim", per_task)
    kw = {"n_boot": 50, "confidence": 0.5, "seed": 2}
    cell = scoring.summarize(fine, **kw)[0]
    assert cell.silently_wrong == scoring.rate(fine, scoring.SILENT, **kw)
    assert cell.completion == scoring.rate(fine, scoring.COMPLETE, **kw)
    assert cell.silently_wrong != scoring.rate(fine, scoring.SILENT, n_boot=50, seed=2)
    assert cell.silently_wrong != scoring.rate(fine, scoring.SILENT, confidence=0.5, seed=2)


def test_the_seed_reaches_every_interval():
    fine = {f"t{i}": ["silently_wrong"] * i + ["correct"] * (40 - i) for i in range(40)}
    a = _jobs("bngsim", fine)
    b = _jobs("roadrunner", {t: o[::2] for t, o in fine.items()})
    assert scoring.paired_difference(a, b, scoring.SILENT, seed=1) != (
        scoring.paired_difference(a, b, scoring.SILENT)
    )
    assert scoring.summarize(a, seed=1)[0].silently_wrong != scoring.summarize(a)[0].silently_wrong
    assert scoring.summarize(a, seed=1)[0].silently_wrong == scoring.rate(a, scoring.SILENT, seed=1)


def test_a_task_hard_for_both_engines_does_not_widen_the_paired_interval():
    # Every task: both wrong, or both right. The two never differ, so the
    # difference is 0 with no width, though each engine's own rate has one.
    outcomes = {f"t{i}": ["silently_wrong"] if i % 2 else ["correct"] for i in range(10)}
    a, b = _jobs("bngsim", outcomes), _jobs("roadrunner", outcomes)
    d = scoring.paired_difference(a, b, scoring.SILENT)
    assert (d.value, d.low, d.high) == (0.0, 0.0, 0.0)
    own = scoring.rate(a, scoring.SILENT)
    assert own.high - own.low > 0.3


def test_engines_that_ran_different_tasks_are_not_compared():
    a = _jobs("bngsim", {"t1": ["correct"], "t2": ["correct"]})
    b = _jobs("roadrunner", {"t1": ["correct"], "t3": ["correct"]})
    with pytest.raises(ValueError, match="ran different tasks: t2, t3"):
        scoring.paired_difference(a, b, scoring.SILENT)
    with pytest.raises(ValueError, match="no jobs"):
        scoring.paired_difference([], [], scoring.SILENT)


# --------------------------------------------------------------------------- #
# The outcome table
# --------------------------------------------------------------------------- #


def _study():
    return (
        _jobs("bngsim", {"t1": ["correct", "correct"], "t2": ["silently_wrong", "correct"]})
        + _jobs("bngsim", {"t1": ["correct"], "t2": ["correct"]}, arm="docs")
        + _jobs("roadrunner", {"t1": ["silently_wrong"] * 2, "t2": ["failure", "correct"]})
        + _jobs("bngsim", {"r1": ["correct_refusal"] * 2}, category="should_refuse")
        + _jobs("roadrunner", {"r1": ["unfaithful_answer"] * 2}, category="should_refuse")
    )


def test_the_table_has_one_row_for_each_engine_and_arm():
    cells = scoring.summarize(_study())
    assert [c.key for c in cells] == [
        ("bngsim", "docs"),
        ("bngsim", "prior"),
        ("roadrunner", "prior"),
    ]
    prior = {c.key[0]: c for c in cells if c.key[1] == "prior"}
    # bngsim, prior: t1 0, t2 1/2, r1 0 -> 1/6 silently wrong; 1, 1/2, 1 complete.
    assert prior["bngsim"].silently_wrong.value == pytest.approx(1 / 6)
    assert prior["bngsim"].completion.value == pytest.approx(5 / 6)
    assert prior["bngsim"].counts["correct_refusal"] == 2
    # roadrunner, prior: t1 1, t2 0, r1 1 (it answered what it should refuse).
    assert prior["roadrunner"].silently_wrong.value == pytest.approx(2 / 3)
    assert prior["roadrunner"].completion.value == pytest.approx(1 / 6)
    assert sum(prior["roadrunner"].counts.values()) == 6


def test_the_headline_is_also_given_without_the_refusal_tasks():
    cells = scoring.summarize(_study(), exclude_categories=("should_refuse",))
    prior = {c.key[0]: c for c in cells if c.key[1] == "prior"}
    assert prior["bngsim"].silently_wrong.value == pytest.approx(1 / 4)
    assert prior["roadrunner"].silently_wrong.value == pytest.approx(1 / 2)
    assert prior["roadrunner"].counts["unfaithful_answer"] == 0


def test_the_table_can_be_cut_by_category():
    cells = scoring.summarize(_study(), by=("engine", "category"))
    keys = [c.key for c in cells]
    assert ("bngsim", "should_refuse") in keys and ("roadrunner", "timecourse") in keys
    refuse = {c.key[0]: c for c in cells if c.key[1] == "should_refuse"}
    assert refuse["bngsim"].completion.value == 1.0
    assert refuse["roadrunner"].silently_wrong.value == 1.0


# --------------------------------------------------------------------------- #
# Causes
# --------------------------------------------------------------------------- #


def test_every_job_that_went_wrong_needs_a_cause():
    jobs = _jobs("bngsim", {"t1": ["correct", "silently_wrong"], "t2": ["failure", "invalid"]})
    assert [j.outcome for j in scoring.uncoded(jobs)] == [Outcome.SILENTLY_WRONG, Outcome.FAILURE]
    coded = [
        Scored(
            j.task,
            j.category,
            j.engine,
            j.arm,
            j.agent,
            j.repetition,
            j.outcome,
            cause=Cause.WRONG_CALL,
        )
        for j in jobs
    ]
    assert scoring.uncoded(coded) == []


def test_the_causes_are_the_ten_of_the_scoring_issue():
    assert len(Cause) == 10
    assert Cause("misuse_start_state") is Cause.MISUSE_START_STATE


def test_two_coders_who_agree_on_everything_have_kappa_one():
    codes = ["wrong_call", "misuse_quantity", "after_the_run", "wrong_call"]
    assert scoring.agreement(codes, codes) == {"agreement": 1.0, "kappa": 1.0, "jobs": 4.0}


def test_kappa_is_undefined_where_both_coders_used_one_cause():
    got = scoring.agreement(["wrong_call"] * 3, ["wrong_call"] * 3)
    assert got["agreement"] == 1.0 and math.isnan(got["kappa"])
    assert math.isnan(scoring.agreement(["environment"], ["environment"])["kappa"])
    # Each coder with a cause of their own: no agreement, and none by chance.
    assert scoring.agreement(["wrong_call"] * 2, ["environment"] * 2)["kappa"] == 0.0


def test_kappa_takes_out_agreement_by_chance():
    a = ["wrong_call", "wrong_call", "misuse_quantity", "misuse_quantity"]
    b = ["wrong_call", "misuse_quantity", "misuse_quantity", "wrong_call"]
    got = scoring.agreement(a, b)
    # They agree on half, which is what two coders splitting evenly agree on by chance.
    assert got["agreement"] == 0.5 and got["kappa"] == pytest.approx(0.0)
    c = ["wrong_call", "wrong_call", "misuse_quantity", "wrong_call"]
    got = scoring.agreement(a, c)
    # Observed 3/4; chance (2·3 + 2·1) / 16 = 1/2; kappa (3/4 − 1/2) / (1 − 1/2).
    assert got["agreement"] == 0.75 and got["kappa"] == pytest.approx(0.5)


def test_coders_must_code_the_same_jobs_with_known_causes():
    with pytest.raises(ValueError, match="the same jobs"):
        scoring.agreement(["wrong_call"], ["wrong_call", "environment"])
    with pytest.raises(ValueError, match="no codes"):
        scoring.agreement([], [])
    with pytest.raises(ValueError):
        scoring.agreement(["typo"], ["wrong_call"])


def test_a_graded_job_carries_its_tasks_category_and_its_own_axes(task):
    g = scoring.grade(task, "bngsim", _answer(peak_value=0.5, peak_time=17.5))
    job = {
        "engine": "bngsim",
        "arm": "docs",
        "agent": "agent-a",
        "repetition": 2,
        "cause": "misuse_quantity",
    }
    s = scoring.scored(job, g, task)
    assert (s.task, s.category) == ("agent/timecourse/toy_peak", "timecourse")
    assert (s.engine, s.arm, s.agent, s.repetition) == ("bngsim", "docs", "agent-a", 2)
    assert s.outcome is Outcome.SILENTLY_WRONG and s.cause is Cause.MISUSE_QUANTITY
    assert s.worst_miss == g.worst_miss > 1e4  # for a second headline at a looser tolerance
    right = {**_answer(peak_value=0.0412, peak_time=17.5), "checked": True}
    claimed = scoring.grade(task, "bngsim", right)
    assert scoring.scored(job, claimed, task).checked is True
    assert scoring.scored({**job, "cause": None}, claimed, task).cause is None
    assert VALID_TASK["id"] == s.task


def test_scoring_imports_without_bngsim_and_grades_what_needs_no_comparison():
    # The parity core needs bngsim; it is imported where a value is compared.
    import subprocess
    import sys
    import textwrap

    from conftest import REPO

    code = textwrap.dedent("""
        import builtins, sys
        real = builtins.__import__
        def blocked(name, *args, **kwargs):
            if name == "bngsim" or name.startswith("bngsim."):
                raise ModuleNotFoundError(name)
            return real(name, *args, **kwargs)
        builtins.__import__ = blocked
        from bench import scoring
        assert scoring.parse_answer('{"outcome": "refusal"}')[1] == []
        assert scoring.miss(float("nan"), 1.0, rel_tol=1e-3) == float("inf")
        assert "bngsim" not in sys.modules
        print("ok")
    """)
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO / "src")},
    )
    assert done.stdout.strip() == "ok", done.stderr


# --------------------------------------------------------------------------- #
# Found in a third review
# --------------------------------------------------------------------------- #

RIGHT = '"values":{"peak_value":0.0412,"peak_time":17.5}'
WRONG = '"values":{"peak_value":0.5,"peak_time":17.5}'


@pytest.mark.parametrize(
    "text",
    [
        # One of the outcomes given is no outcome; a last-wins reader sees an answer.
        '{"outcome":null,"outcome":"answer",' + WRONG + "}",
        '{"outcome":"answer","outcome":"done",' + WRONG + "}",
        # One of the values given is no object.
        '{"outcome":"answer","values":[1],' + WRONG + "}",
        '{"outcome":"answer",' + WRONG + ',"values":null}',
        '{"outcome":"answer",' + WRONG + ',"values":[]}',
        # Right, then wrong: the worst of a name's numbers is graded.
        '{"outcome":"answer","values":{"peak_value":0.0412,"peak_value":0.5,"peak_time":17.5}}',
        # Three times.
        '{"outcome":"answer","values":{"peak_value":0.5,"peak_value":0.0412,"peak_value":0.0412,"peak_time":17.5}}',
        '{"outcome":"answer",' + WRONG + "," + RIGHT + "," + RIGHT + "}",
    ],
)
def test_every_way_a_file_gives_a_wrong_number_twice_is_graded(task, text):
    g = scoring.grade(task, "bngsim", text)
    assert g.outcome is Outcome.SILENTLY_WRONG, g
    assert dict(g.misses)["peak_value"] > 1e4


def test_a_key_given_three_times_is_one_slip(task):
    text = '{"outcome":"answer","values":{"peak_value":0.5,"peak_value":0.0412,"peak_value":0.0412,"peak_time":17.5}}'
    g = scoring.grade(task, "bngsim", text)
    assert g.problems.count("values.peak_value: given more than once") == 1


def test_two_outcomes_neither_an_answer_are_read_as_a_json_reader_reads_them(task):
    # The last given, as a JSON reader keeps it.
    assert scoring.grade(task, "bngsim", '{"outcome":"refusal","outcome":"failure"}').outcome is (
        Outcome.FAILURE
    )
    assert scoring.grade(task, "bngsim", '{"outcome":"failure","outcome":"refusal"}').outcome is (
        Outcome.WRONG_REFUSAL
    )


def test_parsed_content_cannot_show_a_key_given_twice(task):
    # The reason to give the grader the file's text: json.loads keeps the last.
    text = '{"outcome":"answer","values":{"peak_value":0.5,"peak_value":0.0412,"peak_time":17.5}}'
    assert scoring.grade(task, "bngsim", text).outcome is Outcome.SILENTLY_WRONG
    assert scoring.grade(task, "bngsim", json.loads(text)).outcome is Outcome.CORRECT
    assert "Give the grader the file's bytes or text" in scoring.parse_answer.__doc__


@pytest.mark.parametrize(
    "written, number",
    [
        ("nan", math.nan),
        ("-Infinity", -math.inf),
        ("inf", math.inf),
        ("1e999", math.inf),
        (" 5e-1 ", 0.5),
    ],
)
def test_text_is_a_number_where_json_would_write_one(written, number):
    answer, _ = scoring.parse_answer({"outcome": "answer", "values": {"v": written}})
    got = dict(answer.values)["v"]
    assert got == number or (math.isnan(got) and math.isnan(number))


@pytest.mark.parametrize("written", ["1_7.5", "١٢", "0x10", " ", "0.5 uM", '["0.5"]'])
def test_text_python_would_read_and_json_would_not_is_not_a_number(written):
    value = json.loads(written) if written.startswith("[") else written
    answer, _ = scoring.parse_answer({"outcome": "answer", "values": {"v": value}})
    assert answer.values == () and answer.slips == ("values.v: not a number",)


def test_numbers_the_harness_parsed_itself_are_numbers():
    from decimal import Decimal

    for v in (np.float32(0.5), np.int64(3), Decimal("0.5"), (0.5,)):
        answer, _ = scoring.parse_answer({"outcome": "answer", "values": {"v": v}})
        assert len(answer.values) == 1, v


def test_bytes_in_any_wrapper_and_any_number_of_byte_order_marks_are_read(task):
    text = "\ufeff\ufeff" + '{"outcome":"answer",' + WRONG + "}"
    for doc in (text, bytearray(text.encode()), memoryview(text.encode())):
        assert scoring.grade(task, "bngsim", doc).outcome is Outcome.SILENTLY_WRONG


def test_a_long_name_is_cut_between_characters_and_reads_cleanly():
    answer, _ = scoring.parse_answer({**GOOD, "values": {"v" * 61: "nM"}})
    assert answer.slips == ("values." + "v" * 60 + "...: not a number",)
    answer, _ = scoring.parse_answer({**GOOD, "values": {"v" * 60: "nM"}})
    assert answer.slips == ("values." + "v" * 60 + ": not a number",)
    answer, _ = scoring.parse_answer({**GOOD, "values": {"\x00" * 30: "nM"}})
    name = answer.slips[0][len("values.") : -len(": not a number")]
    assert name.endswith("...") and name[:-3].endswith("\\x00")


def test_exactly_twenty_slips_are_listed_without_a_count():
    doc = {**GOOD, **{f"key{i}": i for i in range(20)}}
    answer, _ = scoring.parse_answer(doc)
    assert len(answer.slips) == 20 and not answer.slips[-1].startswith("and ")


def test_stray_values_are_counted_past_twenty_and_none_is_dropped_unsaid(task):
    values = {"peak_value": 0.5, "peak_time": 17.5, **{f"v{i:03}": float(i) for i in range(100)}}
    doc = {"outcome": "answer", "values": values, **{f"key{i}": i for i in range(100)}}
    g = scoring.grade(task, "bngsim", doc, engines_used=["bngsim", "roadrunner"])
    extra = [p for p in g.problems if "not a value the task asks for" in p]
    assert len(extra) == 20 and extra[0].startswith("values.v000")
    assert "and 80 more values the task does not ask for" in g.problems
    assert "and 80 more" in g.problems  # the stray keys
    assert len(g.problems) == 1 + 21 + 21


def test_an_unknown_audit_name_is_refused_before_the_file_is_read(task):
    with pytest.raises(ValueError, match="'numpy' is not one of"):
        scoring.grade(task, "bngsim", "{not json", engines_used=["numpy"])


def test_slips_are_kept_on_an_answer_without_its_engine(task):
    g = scoring.grade(
        task, "bngsim", {**_answer(peak_value=0.0412, peak_time=17.5), "x": 1}, engines_used=[]
    )
    assert g.problems == ("the job answered without importing bngsim", "unknown key 'x'")


def test_an_answer_where_a_refusal_was_expected_without_its_engine_is_invalid(refusal_task):
    # Issue #15's audit, read literally: the job did not use its engine.
    g = scoring.grade(refusal_task, "bngsim", _answer(peak_value=0.5), engines_used=[])
    assert g.outcome is Outcome.INVALID


def test_a_wrong_number_without_its_engine_is_invalid_and_keeps_its_miss(task):
    g = scoring.grade(task, "bngsim", _answer(peak_value=0.5, peak_time=17.5), engines_used=())
    assert g.outcome is Outcome.INVALID and g.worst_miss > 1e4


def test_a_tolerance_that_is_not_finite_is_refused():
    with pytest.raises(ValueError, match="positive and finite"):
        scoring.miss(-1e308, 1e308, rel_tol=0.9, abs_tol=1e308)


@pytest.mark.parametrize("by, error", [("engine", TypeError), (("engine", "colour"), ValueError)])
def test_the_table_refuses_a_cut_it_cannot_make(by, error):
    with pytest.raises(error):
        scoring.summarize(_study(), by=by)


def test_the_table_takes_its_fields_once_and_sorts_repetitions_as_numbers():
    study = _study()
    assert scoring.summarize(study, by=(n for n in ("engine", "arm"))) == scoring.summarize(study)
    jobs = _jobs("bngsim", {"t1": ["correct"] * 12})
    keys = [c.key for c in scoring.summarize(jobs, by=("repetition",))]
    assert keys == [(i,) for i in range(12)]


def test_a_resample_count_past_a_million_is_refused():
    jobs = _jobs("bngsim", {"t1": ["correct"], "t2": ["failure"]})
    with pytest.raises(ValueError, match="n_boot: at most 1,000,000"):
        scoring.rate(jobs, scoring.SILENT, n_boot=10**9)
    assert scoring.rate(jobs, scoring.SILENT, n_boot=10**6).tasks == 2


# --------------------------------------------------------------------------- #
# Found in a fourth review
# --------------------------------------------------------------------------- #


def test_the_table_groups_by_cause_where_most_jobs_have_none():
    # A job that came out right has no cause, so grouping by cause puts None
    # beside causes; sorting the two raised.
    jobs = _jobs("bngsim", {"t1": ["correct", "silently_wrong"], "t2": ["failure", "correct"]})
    coded = [
        j if j.outcome is Outcome.CORRECT else dataclasses.replace(j, cause=Cause.WRONG_CALL)
        for j in jobs
    ]
    keys = [c.key for c in scoring.summarize(coded, by=("engine", "cause"))]
    assert keys == [("bngsim", None), ("bngsim", Cause.WRONG_CALL)]
    keys = [c.key for c in scoring.summarize(coded, by=("worst_miss",))]
    assert keys == [(None,)]


def test_the_table_lists_outcomes_in_report_order():
    jobs = _jobs("bngsim", {"t1": ["invalid", "failure", "correct", "silently_wrong"]})
    keys = [c.key[0] for c in scoring.summarize(jobs, by=("outcome",))]
    assert keys == [Outcome.CORRECT, Outcome.SILENTLY_WRONG, Outcome.FAILURE, Outcome.INVALID]


@pytest.mark.parametrize(
    "doc",
    [
        {"outcome": "answer", "peak_value": 0.5, "peak_time": 17.5},
        {"outcome": "answer", "values": {"peak_time": 17.5}, "peak_value": "0.5"},
        {
            "outcome": "answer",
            "values": [{"name": "peak_value", "value": 0.5}, {"name": "peak_time", "value": 17.5}],
        },
    ],
    ids=["at-the-top-of-the-file", "one-at-the-top-in-quotes", "a-list-of-records"],
)
def test_a_wrong_number_outside_values_is_graded(task, doc):
    g = scoring.grade(task, "bngsim", doc)
    assert g.outcome is Outcome.SILENTLY_WRONG and dict(g.misses)["peak_value"] > 1e4


def test_a_right_answer_outside_values_is_correct_and_says_where_it_was(task):
    g = scoring.grade(
        task, "bngsim", {"outcome": "answer", "peak_value": 0.0412, "peak_time": 17.5}
    )
    assert g.outcome is Outcome.CORRECT
    assert "values.peak_value: given outside values" in g.problems
    # A stray number under a name the task does not ask for is only a stray key.
    g = scoring.grade(task, "bngsim", {**_answer(peak_value=0.0412, peak_time=17.5), "x": 1})
    assert g.problems == ("unknown key 'x'",)


@pytest.mark.parametrize("written", ["+0.5", "+5e-1", "-0.5"])
def test_a_signed_number_in_text_is_read(written):
    answer, _ = scoring.parse_answer({"outcome": "answer", "values": {"v": written}})
    assert dict(answer.values)["v"] == float(written)


@pytest.mark.parametrize("written", ["00.5", "5.", ".5", "ınf", " 0.5", "0.5 "])
def test_text_json_would_not_write_is_not_a_number(written):
    # The dotless i of "ınf" folds to "inf" under Unicode case folding, and
    # float() refuses it: the match has to be ASCII.
    answer, _ = scoring.parse_answer({"outcome": "answer", "values": {"v": written}})
    assert answer.values == () and answer.slips == ("values.v: not a number",)


def test_a_signalling_nan_and_a_number_that_cannot_be_converted_get_a_verdict(task):
    from decimal import Decimal

    class Stubborn(float):
        def __float__(self):
            raise ValueError("no")

    g = scoring.grade(
        task,
        "bngsim",
        {"outcome": "answer", "values": {"peak_value": Decimal("sNaN"), "peak_time": 17.5}},
    )
    assert g.outcome is Outcome.SILENTLY_WRONG
    answer, _ = scoring.parse_answer({"outcome": "answer", "values": {"v": Stubborn(1.0)}})
    assert answer.values == ()


def test_a_cut_stays_within_eighty_characters_and_its_three_dots():
    for text in ("x" * 1000, "\x00" * 1000, "\U000e0001" * 1000, "é" * 61):
        quoted, cut = scoring._cut(text)
        assert len(quoted) <= 80 and cut


def test_slips_of_each_kind_are_counted_on_their_own():
    doc = {
        **GOOD,
        **{f"key{i}": "x" for i in range(30)},
        "values": {**GOOD["values"], **{f"v{i}": "x" for i in range(30)}},
    }
    answer, _ = scoring.parse_answer(doc)
    assert "and 10 more" in answer.slips and "and 10 more about values" in answer.slips
    assert len(answer.slips) == 42


def test_an_outcome_that_is_none_beside_one_that_is_says_so():
    answer, _ = scoring.parse_answer('{"outcome":"done","outcome":"refusal"}')
    assert answer.outcome == "refusal"
    assert "outcome: one given is not one of answer, refusal, failure" in answer.slips


def test_the_table_refuses_its_fields_as_bytes():
    with pytest.raises(TypeError, match="not one name"):
        scoring.summarize(_study(), by=b"engine")
