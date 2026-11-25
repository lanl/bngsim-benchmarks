"""The generated summaries of published/: every snapshot's README.md, the index
(published/README.md, index.json), and the guard that they match the committed data."""

from __future__ import annotations

import json

from conftest import REPO
from test_publish import _edit, _member_jobs
from test_staleness import published

from bench import cli, specs, staleness, summary
from bench import jobs as jobs_mod

PUBLISHED = REPO / "published"


def test_every_committed_summary_is_what_the_committed_data_generates():
    """The guard: each snapshot's README.md from its manifest and CSVs, and the index
    from the snapshots, problems/, the code and the bngsim release index.json records.
    A spec, a snapshot or the code that changes without its summaries fails here."""
    problems = specs.load_all(REPO / "problems")
    files = summary.generated(PUBLISHED, problems, jobs_mod.jobs_for(problems),
                              release=summary.recorded_release(PUBLISHED), root=REPO)
    assert len(files) == sum(len(v) for v in summary.snapshots(PUBLISHED).values()) + 2
    stale = [str(p.relative_to(REPO)) for p in summary.differing(files)]
    assert not stale, ("these generated files do not match the committed data; run "
                       "`pixi run bench publish --index --offline` and commit them: "
                       + ", ".join(stale))


def test_the_index_lists_every_suite_with_a_description():
    assert set(summary.DESCRIPTIONS) == set(specs.SUITES)
    idx = json.loads((PUBLISHED / summary.INDEX_JSON).read_text())
    assert [e["suite"] for e in idx["suites"]] == list(specs.SUITES)
    for e in idx["suites"]:
        assert (e["snapshot"] is None) == (e["status"] == staleness.NOT_PUBLISHED)


def test_every_committed_csv_row_names_its_setting():
    for paths in summary.snapshots(PUBLISHED).values():
        for p in paths:
            snap = summary.read_snapshot(p)
            jobs = {(j["problem"], j["job_id"].split("/")[3]) for j in snap.manifest["jobs"]}
            for row in snap.correctness + snap.timing:
                assert (row["problem"], row["setting"]) in jobs, (p, row["problem"])


# --------------------------------------------------------------------------- #
# A snapshot's README
# --------------------------------------------------------------------------- #
def _snapshot(spec_data, write_spec, tmp_path, edit=None):
    return published(tmp_path, spec_data, write_spec, edit)


def test_publish_writes_the_snapshot_readme_from_its_own_files(spec_data, write_spec,
                                                               tmp_path):
    _, _, dest = _snapshot(spec_data, write_spec, tmp_path)
    text = (dest / "README.md").read_text()
    assert text == summary.snapshot_readme(summary.read_snapshot(dest))
    assert text.startswith("# ode-net: bngsim 0.17.2 (932ea5e13396)\n")
    for line in ("- BNGsim: 0.17.2, build 932ea5e13396",
                 "- Environment: BioNetGen 2.9.3, libRoadRunner 2.11.0, COPASI 4.48.311, "
                 "AMICI 1.1.0",
                 "- Harness: bench 0.1.0 at commit cccccccccccc",
                 "- Timing node: Some CPU @ 2.10GHz, x86_64, Linux",
                 "- Published: 2026-11-23 12:00 UTC"):
        assert line in text.splitlines()
    assert "10 PASS. Each reference column" in text
    # A reference that timed out is named, with why, once per member.
    assert ("- #1 model_01, copasi: timeout, `TIMEOUT: no result within the 3600 s "
            "per-engine budget`") in text
    assert text.count("copasi: timeout") == 10
    # Absolute times; the speedup only derived from them.
    assert "| 1 | model_01 | 5 | bngsim | 3.10 | 0.500 | 0.400–0.600 |" in text
    assert "|  |  |  | roadrunner | 3.00 | 0.800 | 0.700–0.900 |" in text
    assert "| 1 | model_01 | 1.6 |" in text  # roadrunner / bngsim = 0.8 / 0.5


def test_a_diff_a_withheld_timing_and_a_reference_that_does_not_apply(spec_data, write_spec,
                                                                      tmp_path):
    why = "no faithful BNGL form"

    def edit(jobs, results):
        corr, _ = _member_jobs(jobs, "model_02")
        _edit(results, corr, outcome="DIFF", verdicts={"roadrunner": {
            "passed": False, "max_err_over_peak": 0.25, "max_rel": 0.1, "n_cells": 10,
            "vendored_passed": True, "species_check": {"passed": False}}})
        for job in jobs:
            if job.problem == "model_03":
                _edit(results, job, not_applicable={"run_network": why})

    _, _, dest = _snapshot(spec_data, write_spec, tmp_path, edit)
    text = (dest / "README.md").read_text()
    assert "9 PASS, 1 DIFF." in text
    assert ("| 2 | [model_02](https://github.com/owner/models/blob/" + "a" * 40
            + "/models/m2/m2.bngl) | 9 / 14 | DIFF |  | DIFF (species) | timeout |") in text
    assert ("- #2 model_02, roadrunner: DIFF, BNGsim disagrees with it; "
            "max error / peak 0.25; failed the per-species check") in text
    assert "| 2 | model_02 |  | – |  | withheld: correctness DIFF |  |" in text
    assert f"- run_network, #3: `{why}`" in text


def test_a_long_or_multiline_error_is_one_line_and_cut():
    assert summary.one_line("A\nB") == "`A`"
    cut = summary.one_line("x" * 300)
    assert len(cut) == 202 and cut.endswith("…`")
    assert summary.one_line("uses `bngsim`") == "`` uses `bngsim` ``"


def test_times_keep_the_figures_their_interval_needs():
    assert summary.seconds("0.04797851") == "0.0480"
    assert summary.seconds("269.88") == "270" and summary.seconds("1586.85") == "1587"
    assert summary.solve_cells("269.882", "269.878", "269.919") == ("269.88", "269.88–269.92")
    assert summary.solve_cells("0.5", "0.4", "0.6") == ("0.500", "0.400–0.600")


# --------------------------------------------------------------------------- #
# The index, and the commands
# --------------------------------------------------------------------------- #
def test_the_index_says_why_a_snapshot_is_out_of_date(spec_data, write_spec, tmp_path):
    problems, jobs, dest = _snapshot(spec_data, write_spec, tmp_path)
    release = {"version": "0.17.3", "checked": "2026-11-25"}
    idx = summary.index(tmp_path / "published", problems, jobs, release=release, root=REPO)
    entry = next(e for e in idx["suites"] if e["suite"] == "ode-net")
    assert entry["snapshot"] == "ode-net/bngsim-0.17.2-932ea5e13396"
    assert entry["status"] == staleness.OUT_OF_DATE
    assert "bngsim 0.17.3 is on PyPI (snapshot: 0.17.2)" in entry["reasons"]
    assert entry["correctness"] == {"jobs": 10, "outcomes": {"PASS": 10}}
    others = [e for e in idx["suites"] if e["suite"] != "ode-net"]
    assert {e["status"] for e in others} == {staleness.NOT_PUBLISHED}
    text = summary.index_readme(idx)
    assert "Newest bngsim release on PyPI: 0.17.3 (looked up 2026-11-25)." in text
    assert ("| [ode-net](ode-net/bngsim-0.17.2-932ea5e13396/README.md) | "
            + summary.DESCRIPTIONS["ode-net"] + " | 10 | bngsim 0.17.2, 2026-11-23 | 10 PASS | "
            "**OUT OF DATE**: bngsim 0.17.3 is on PyPI (snapshot: 0.17.2)") in text
    assert "| ss-net | " + summary.DESCRIPTIONS["ss-net"] + " | 0 | – | – | not published |" \
        in text
    md = summary.staleness_markdown(idx, shown=False, link_base="https://x/published")
    assert "[bngsim-0.17.2-932ea5e13396](https://x/published/ode-net/" in md
    assert "does not show this yet" in md


def test_a_skipped_release_leaves_the_snapshot_current_and_says_so(spec_data, write_spec,
                                                                   tmp_path):
    problems, jobs, _ = _snapshot(spec_data, write_spec, tmp_path)
    why = "skipped by decision 2026-11-25; the next measured release is 0.17.4"
    (tmp_path / "published" / staleness.SKIPPED_FILE).write_text(json.dumps({"0.17.3": why}))
    release = {"version": "0.17.3", "releases": ["0.17.2", "0.17.3"], "checked": "2026-11-25"}
    idx = summary.index(tmp_path / "published", problems, jobs, release=release, root=REPO)
    entry = next(e for e in idx["suites"] if e["suite"] == "ode-net")
    assert (entry["status"], entry["reasons"]) == (staleness.CURRENT, [])
    assert entry["notes"] == [f"bngsim 0.17.3 skipped: {why}"]
    assert idx["skipped_releases"] == {"0.17.3": why}
    assert f"| 10 PASS | CURRENT (bngsim 0.17.3 skipped: {why}) |" in summary.index_readme(idx)


def test_no_committed_snapshot_is_out_of_date_for_a_skipped_release():
    """A release skipped on purpose (skipped-releases.json) never makes a snapshot out
    of date; other reasons (an engine or harness change awaiting the next run) may."""
    idx = json.loads((PUBLISHED / summary.INDEX_JSON).read_text())
    published = [e for e in idx["suites"] if e["snapshot"]]
    skipped = json.loads((PUBLISHED / "skipped-releases.json").read_text())
    assert published
    for e in published:
        text = json.dumps(e)
        for version in skipped:
            assert f"bngsim {version} is on PyPI" not in text, e["suite"]


def test_publish_index_and_staleness_commands(spec_data, write_spec, tmp_path, monkeypatch):
    _snapshot(spec_data, write_spec, tmp_path)
    monkeypatch.setattr(cli, "ROOT", REPO)  # pixi.toml's pins
    out = tmp_path / "published"
    common = ["--problems", str(tmp_path / "problems"), "--out", str(out)]
    assert cli.main(["publish", "--problems", str(tmp_path / "problems")]) == 2
    # Offline with no release recorded, the snapshot is judged on the rest: current.
    assert cli.main(["publish", "--index", "--offline", *common]) == 0
    assert (out / "README.md").exists() and (out / "index.json").exists()
    assert cli.main(["staleness", "--offline", *common]) == 0
    # A newer release recorded in the index makes the snapshot out of date (exit 1).
    idx = json.loads((out / "index.json").read_text())
    idx["bngsim_release"] = {"version": "0.18.0", "checked": "2026-12-01"}
    (out / "index.json").write_text(json.dumps(idx))
    md = tmp_path / "issue.md"
    assert cli.main(["staleness", "--offline", *common, "--markdown", str(md)]) == 1
    assert "bngsim 0.18.0 is on PyPI" in md.read_text()
    assert "does not show this yet" in md.read_text()
    assert cli.main(["publish", "--index", "--offline", *common]) == 0
    assert "**OUT OF DATE**: bngsim 0.18.0" in (out / "README.md").read_text()
    assert cli.main(["staleness", "--offline", *common, "--markdown", str(md)]) == 1
    assert "does not show this yet" not in md.read_text()


def test_publish_looks_up_the_release_and_keeps_the_recorded_one_when_it_cannot(
        spec_data, write_spec, tmp_path, monkeypatch, capsys):
    _snapshot(spec_data, write_spec, tmp_path)
    monkeypatch.setattr(cli, "ROOT", REPO)
    out = tmp_path / "published"
    common = ["--problems", str(tmp_path / "problems"), "--out", str(out)]
    monkeypatch.setattr(staleness, "look_up_releases", lambda: ["0.17.1", "0.17.2"])
    assert cli.main(["publish", "--index", *common]) == 0
    rel = json.loads((out / "index.json").read_text())["bngsim_release"]
    assert rel["version"] == "0.17.2" and rel["releases"] == ["0.17.1", "0.17.2"]
    assert len(rel["checked"]) == 10

    def down():
        raise OSError("no network")

    monkeypatch.setattr(staleness, "look_up_releases", down)
    assert cli.main(["publish", "--index", *common]) == 0
    assert "keeping the recorded 0.17.2" in capsys.readouterr().err
    assert cli.main(["staleness", *common]) == 2
