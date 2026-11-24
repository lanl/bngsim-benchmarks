"""Jobs split into parts (Job.parts): task ids, selection, packing across array
tasks, and the runner running the parts and then the job that scores them."""

from __future__ import annotations

from bench import jobs, results, slurm
from bench.jobs import CORRECTNESS, TIMING, Job
from bench.runner import Launch, RunConfig, run

FPS = {CORRECTNESS: {"bngsim": {"build_commit": "abc"}},
       TIMING: {"bngsim": {"build_commit": "abc"}, "node_type": "n"}}


def split_pair(parts=None, est=None):
    parts = parts or {"bngsim": 600.0, "copasi": 300.0}
    c = Job("ssa", "big", "e", "auto", CORRECTNESS, definition={"d": 1},
            est_seconds=est or sum(parts.values()) + 50.0, parts=parts)
    t = Job("ssa", "big", "e", "auto", TIMING, definition={"d": 1}, est_seconds=100.0,
            requires=c.id)
    return c, t


def test_a_part_keeps_its_jobs_id_and_key_and_names_itself_in_a_task():
    c, _ = split_pair()
    p = c.as_part("copasi")
    assert p.id == c.id and p.canonical() == c.canonical()
    assert p.task_id == c.id + "#copasi"
    assert p.est_seconds == 300.0 + 50.0 and c.remainder_seconds == 50.0
    assert p.whole() == c


def test_selecting_parts_brings_no_whole_job_even_beside_its_timing_job():
    c, t = split_pair()
    got = jobs.select_exact([c, t], [c.id + "#bngsim", c.id + "#copasi", t.id])
    assert [(j.task_id, j.kind) for j in got] == [
        (c.id + "#bngsim", CORRECTNESS), (c.id + "#copasi", CORRECTNESS), (t.id, TIMING)]
    assert jobs.select_exact([c, t], [t.id]) == [c, t]


def cfg(limit="20"):
    return slurm.validate_config({"time_limit": limit, "cpus_per_task": 1,
                                  "margin_seconds": 100, "startup_seconds": 100})


def test_a_unit_over_half_an_allocation_is_split_and_its_parts_share_a_task():
    c, t = split_pair(parts={"bngsim": 600.0, "copasi": 300.0})
    wide = slurm.validate_config({"time_limit": "35", "cpus_per_task": 4,
                                  "margin_seconds": 100, "startup_seconds": 100})
    (task,) = slurm.pack([c, t], wide).tasks  # 1050 s > half of 1900 s
    assert sorted(j.task_id for j in task.jobs) == sorted(
        [c.id + "#bngsim", c.id + "#copasi", t.id, t.id])


def test_a_unit_too_long_for_one_task_is_split_into_its_parts():
    c, t = split_pair()  # 950 s + 100 s of timing > 1000 s of capacity
    packing = slurm.pack([c, t], cfg())
    assert not packing.long
    assert sorted(j.task_id for task in packing.tasks for j in task.jobs) == sorted(
        [c.id + "#bngsim", c.id + "#copasi", t.id, t.id])
    for task in packing.tasks:
        assert task.bound(cfg()) <= cfg().capacity_seconds


def test_a_unit_that_fits_is_not_split_and_stored_parts_are_not_planned_again():
    c, t = split_pair()
    # 1050 s of 3400 s: under half an allocation, so not split
    assert [j.task_id for j in slurm.pack([c, t], cfg("60")).tasks[0].jobs] == [c.id, t.id]
    packing = slurm.pack([c, t], cfg(), done_parts={c.id: {"bngsim"}})
    assert sorted(j.task_id for task in packing.tasks for j in task.jobs) == [
        c.id + "#copasi", t.id]
    packing = slurm.pack([c, t], cfg(), done_parts={c.id: {"bngsim", "copasi"}})
    (task,) = packing.tasks
    assert [j.task_id for j in task.jobs] == [c.id, t.id]
    assert task.jobs[0].est_seconds == c.remainder_seconds


def test_a_part_too_long_for_one_task_is_still_set_aside():
    c, t = split_pair(parts={"bngsim": 5000.0, "copasi": 10.0})
    packing = slurm.pack([c, t], cfg())
    assert [[j.task_id for j in u] for u in packing.long] == [[c.id + "#bngsim", t.id]]


class PartLauncher:
    """Stores a part as a worker would, or writes the job's result."""

    def __init__(self):
        self.calls = []

    def __call__(self, job, timeout, cpu, cfg):
        self.calls.append(job.task_id)
        key = results.freshness_key(job, FPS[job.kind])
        if job.part is not None:
            results.write_part(cfg.results_dir, job, job.part, key, {"seconds": 1.0})
        else:
            results.write_result(cfg.results_dir, job, {
                "outcome": "PASS", "key": key, "key_parts": results.key_parts(job, FPS[job.kind])})
        return Launch(0, False, 1.0)


def test_the_run_that_stores_the_last_part_scores_the_job_and_times_it(tmp_path):
    c, t = split_pair()
    rc = RunConfig(problems_dir=tmp_path, results_dir=tmp_path / "res", cache_dir=tmp_path,
                   quiet=True)
    first, second = PartLauncher(), PartLauncher()
    s1 = run([c.as_part("bngsim"), t], rc, fingerprints=FPS, launcher=first)
    assert first.calls == [c.id + "#bngsim"]
    assert s1.skipped[c.id] == "waiting for its parts" and s1.skipped[t.id] == "waiting"
    s2 = run([c.as_part("copasi"), t], rc, fingerprints=FPS, launcher=second)
    assert second.calls == [c.id + "#copasi", c.id, t.id]
    assert s2.outcomes[c.id] == "PASS"
    # everything is current now: nothing runs again
    third = PartLauncher()
    run([c.as_part("bngsim"), c.as_part("copasi"), t], rc, fingerprints=FPS, launcher=third)
    assert third.calls == []
