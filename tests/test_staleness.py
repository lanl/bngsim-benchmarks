"""bench staleness: a published snapshot is OUT OF DATE, and says why, when a newer
bngsim is released or what it measured differs from what the repository defines now."""

from __future__ import annotations

import io
import json

import pytest
from conftest import REPO
from test_publish import FPS, NOW, _member, _write_all

from bench import env, fetch, specs, staleness
from bench import jobs as jobs_mod
from bench.publish import publish

PINS = staleness.pins(REPO)


def _environment():
    """The environment part of a record's key, as a job run with the pinned engines
    and this harness records it."""
    return {"bngsim": {"version": "0.17.2", "build_commit": "932ea5e13396"},
            "bionetgen": {"version": fetch.BIONETGEN_VERSION,
                          "sha256": fetch.BIONETGEN_SHA256["linux"]},
            "harness": json.loads(json.dumps(env.harness_identity()))}


def published(tmp_path, spec_data, write_spec, edit=None):
    """A published ode-net snapshot whose records are keyed as a real run's are:
    ``(problems, jobs, snapshot directory)``. ``edit(jobs, results)`` may change the
    results before they are published."""
    for i in range(1, specs.SUITE_SIZE + 1):
        write_spec(_member(spec_data, i))
    problems = specs.load_all(tmp_path / "problems")
    jobs = jobs_mod.jobs_for(problems)
    results = tmp_path / "results"
    _write_all(jobs, results)
    if edit:
        edit(jobs, results)
    dest = publish("ode-net", problems, jobs, results, tmp_path / "published", FPS,
                   root=tmp_path, now=NOW)
    by_id = {j.id: j for j in jobs}
    for path in (dest / "jobs").rglob("*.json"):
        rec = json.loads(path.read_text())
        job = by_id[rec["job_id"]]
        rec["key_parts"] = {"job": {"id": job.id, "definition": staleness.pinned_definition(
            job.definition, PINS)}, "environment": _environment()}
        path.write_text(json.dumps(rec))
    return problems, jobs, dest


def _publish(tmp_path, spec_data, write_spec):
    return published(tmp_path, spec_data, write_spec)[2]


def _reasons(dest, tmp_path, *, release=None, pins=PINS):
    problems = specs.load_all(tmp_path / "problems")
    return staleness.reasons(dest, problems, jobs_mod.jobs_for(problems), pins=pins,
                             release=release)


def test_a_snapshot_of_what_the_repository_defines_now_is_current(spec_data, write_spec,
                                                                  tmp_path):
    dest = _publish(tmp_path, spec_data, write_spec)
    assert _reasons(dest, tmp_path) == []
    assert _reasons(dest, tmp_path, release={"version": "0.17.2", "checked": "x"}) == []


def test_a_newer_bngsim_release_makes_it_out_of_date(spec_data, write_spec, tmp_path):
    dest = _publish(tmp_path, spec_data, write_spec)
    assert _reasons(dest, tmp_path, release={"version": "0.17.10", "checked": "x"}) == [
        "bngsim 0.17.10 is on PyPI (snapshot: 0.17.2)"]
    # An older one, or one that is not a final release, does not.
    assert _reasons(dest, tmp_path, release={"version": "0.17.1", "checked": "x"}) == []
    assert _reasons(dest, tmp_path, release={"version": "0.18.0rc1", "checked": "x"}) == []


SKIP = {"0.17.3": "skipped by decision 2026-11-25; the next measured release is 0.17.4"}


def test_a_release_skipped_on_purpose_is_a_note_not_a_reason(spec_data, write_spec,
                                                             tmp_path):
    problems, jobs, dest = published(tmp_path, spec_data, write_spec)

    def assess(*releases):
        release = {"version": releases[-1], "releases": list(releases), "checked": "x"}
        return staleness.assess(dest, problems, jobs, pins=PINS, release=release,
                                skipped=SKIP)

    note = f"bngsim 0.17.3 skipped: {SKIP['0.17.3']}"
    assert assess("0.17.2", "0.17.3") == ([], [note])
    # The next release that is not skipped makes it out of date again.
    assert assess("0.17.2", "0.17.3", "0.17.4") == (
        ["bngsim 0.17.4 is on PyPI (snapshot: 0.17.2)"], [note])
    # So does one between the snapshot and a skipped release.
    skip = {"0.17.4": "broken"}
    release = {"version": "0.17.4", "releases": ["0.17.2", "0.17.3", "0.17.4"], "checked": "x"}
    assert staleness.release_check("0.17.2", release, skip) == (
        ["bngsim 0.17.3 is on PyPI (snapshot: 0.17.2)"], ["bngsim 0.17.4 skipped: broken"])
    # A release seen before the list was recorded: just the newest.
    assert staleness.release_check("0.17.2", {"version": "0.17.3", "checked": "x"}, SKIP) \
        == ([], [note])


def test_the_skipped_releases_file_is_checked(tmp_path):
    assert staleness.skipped_releases(tmp_path) == {}
    path = tmp_path / staleness.SKIPPED_FILE
    path.write_text(json.dumps(SKIP))
    assert staleness.skipped_releases(tmp_path) == SKIP
    path.write_text(json.dumps({"0.18.0rc1": "x", "0.17.5": " "}))
    with pytest.raises(ValueError, match="'0.18.0rc1': not a final release; '0.17.5': no reason"):
        staleness.skipped_releases(tmp_path)
    assert staleness.skipped_releases(REPO / "published")  # the committed one reads


def test_a_protocol_or_model_file_change_names_the_members(spec_data, write_spec, tmp_path):
    dest = _publish(tmp_path, spec_data, write_spec)
    d = _member(spec_data, 3)
    d["protocol"] = {"ode": {**d["protocol"]["ode"], "t_end": 1200}}
    write_spec(d)
    d = _member(spec_data, 5)
    d["source"] = {**d["source"], "sha256": "e" * 64}
    write_spec(d)
    assert _reasons(dest, tmp_path) == ["model file changed: model_05",
                                        "protocol changed: model_03"]


def test_a_verdict_change_says_every_member(spec_data, write_spec, tmp_path, monkeypatch):
    from bench.capabilities import ode

    dest = _publish(tmp_path, spec_data, write_spec)
    monkeypatch.setattr(ode, "VERDICT_VERSION", ode.VERDICT_VERSION + 1)
    assert _reasons(dest, tmp_path) == ["verdict changed: all 10 members"]


def test_a_swapped_or_moved_member_is_named_with_its_position(spec_data, write_spec,
                                                              tmp_path):
    dest = _publish(tmp_path, spec_data, write_spec)
    (tmp_path / "problems" / "model_02.yaml").unlink()
    write_spec(_member(spec_data, 11) | {"suites": {"ode-net": 2}})
    assert _reasons(dest, tmp_path) == ["members changed: out model_02 (#2); in model_11 (#2)"]
    a, b = _member(spec_data, 3), _member(spec_data, 4)
    write_spec(a | {"suites": {"ode-net": 4}})
    write_spec(b | {"suites": {"ode-net": 3}})
    assert _reasons(dest, tmp_path)[1] == "positions changed: model_03 #3 -> #4, model_04 #4 -> #3"


def test_a_changed_engine_set_is_a_job_change(spec_data, write_spec, tmp_path, monkeypatch):
    from bench.capabilities import ode

    dest = _publish(tmp_path, spec_data, write_spec)
    monkeypatch.setattr(ode, "engines_id", lambda: "bngsim-vs-run_network")
    assert _reasons(dest, tmp_path) == [
        "jobs changed (engines or settings): new jobs for all 10 members; "
        "jobs no longer defined for all 10 members"]


def test_an_engine_pin_bngsim_pin_or_harness_change_is_out_of_date(spec_data, write_spec,
                                                                   tmp_path, monkeypatch):
    dest = _publish(tmp_path, spec_data, write_spec)
    pins = {**PINS, "roadrunner": "2.12.0", "bngsim": "0.17.4"}
    assert _reasons(dest, tmp_path, pins=pins) == [
        "the environment pins bngsim 0.17.4 (snapshot: 0.17.2)",
        f"the environment pins roadrunner 2.12.0 (snapshot: {PINS['roadrunner']})"]
    was = env.harness_identity()["bench"]
    monkeypatch.setattr(env, "__version__", "9.9.9")
    assert _reasons(dest, tmp_path) == [
        f"harness version is now bench 9.9.9 (snapshot: bench {was})"]


def test_a_new_bionetgen_is_an_engine_change(spec_data, write_spec, tmp_path):
    # BioNetGen 2.9.4 (which bundles NFsim 2.0.1) once its fetch pin moves.
    dest = _publish(tmp_path, spec_data, write_spec)
    pins = {**PINS, "bionetgen": {"version": "2.9.4", "sha256": ["f" * 64]}}
    assert _reasons(dest, tmp_path, pins=pins) == [
        f"BioNetGen is now 2.9.4 (snapshot: {fetch.BIONETGEN_VERSION})"]


def test_every_real_jobs_engine_versions_have_a_pin():
    problems = specs.load_all(REPO / "problems")
    names = set()
    for job in jobs_mod.jobs_for(problems):
        d = staleness.pinned_definition(job.definition, PINS)  # raises on a missing pin
        names |= set(d.get("reference_versions") or {})
    assert names and all(PINS[n] for n in names), {n: PINS.get(n) for n in names}
    assert PINS["bngsim"] and PINS["bngsim"] == staleness.pins(REPO)["bngsim"]


def test_a_reference_engine_without_a_pin_is_refused():
    with pytest.raises(KeyError, match="no pin for reference engine 'tellurium'"):
        staleness.pinned_definition({"reference_versions": {"tellurium": "2.2"}}, PINS)


def test_the_newest_release_skips_pre_releases_and_yanked_ones():
    data = {"releases": {"0.17.2": [{"yanked": False}], "0.17.3": [{"yanked": False}],
                         "0.17.10": [{"yanked": True}], "0.18.0rc1": [{"yanked": False}],
                         "0.16.0": []}}
    assert staleness.newest_release(data) == "0.17.3"
    assert staleness.final_releases(data) == ["0.17.2", "0.17.3"]
    assert staleness.release_key("0.17.10") > staleness.release_key("0.17.9")
    assert staleness.release_key("1.0.0.dev1") is None


def test_the_release_is_read_from_pypi_json():
    body = json.dumps({"releases": {"0.17.2": [{"yanked": False}],
                                    "0.17.3": [{"yanked": False}]}}).encode()
    seen = {}

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def opener(req, timeout):
        seen["url"], seen["agent"] = req.full_url, req.get_header("User-agent")
        return Resp(body)

    assert staleness.look_up_releases(opener=opener) == ["0.17.2", "0.17.3"]
    assert seen == {"url": staleness.RELEASES_URL, "agent": fetch.USER_AGENT}


@pytest.mark.network
def test_pypi_names_a_final_bngsim_release():
    assert staleness.release_key(staleness.look_up_releases()[-1]) is not None
