"""`bench submit`: Slurm settings, packing jobs into array tasks, and the batch
script.

The packing oracle is a simulation of the runner itself: each free worker takes
the next job in list order, correctness jobs first, then timing jobs. Every
packed task must finish inside the allocation by that simulation, which is
independent of the bound the packer uses. The batch script is checked by
running it with bash, with a stand-in for bench that records what it was given.
"""

from __future__ import annotations

import heapq
import json
import os
import random
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml
from conftest import REPO

from bench import cli, jobs, results, slurm, specs
from bench.jobs import CORRECTNESS, TIMING, Job

# --- time limits ------------------------------------------------------------


@pytest.mark.parametrize(
    "text, seconds",
    [("30", 1800), ("600", 36000), ("5:30", 330), ("10:00:00", 36000),
     ("1-0", 86400), ("1-12", 129600), ("2-03:04", 2 * 86400 + 3 * 3600 + 4 * 60),
     ("0-00:00:45", 45), ("36:00:00", 129600)],
)
def test_slurm_time_limit_formats(text, seconds):
    assert slurm.parse_time_limit(text) == seconds


@pytest.mark.parametrize("text", ["", "1h", "10:60:00", "1:2:3:4", "-5", "0", "00:00",
                                  "1-2:60", "5:60", "UNLIMITED"])
def test_bad_time_limits_are_refused(text):
    with pytest.raises(ValueError):
        slurm.parse_time_limit(text)


# --- settings ----------------------------------------------------------------

MINIMAL = {"time_limit": "1:00:00", "cpus_per_task": 8}


def test_committed_example_config_is_valid():
    cfg = slurm.load_config(REPO / "slurm.example.yaml")
    assert cfg.time_limit_seconds == 36000
    assert cfg.capacity_seconds == 36000 - cfg.margin_seconds - cfg.startup_seconds
    # The example must not name any real partition or account.
    assert cfg.partition is cfg.account is cfg.qos is cfg.constraint is None


def test_minimal_config_takes_the_defaults():
    cfg = slurm.validate_config(dict(MINIMAL))
    assert (cfg.timing_workers, cfg.bench, cfg.setup) == (1, "pixi run --frozen bench", ())


def test_unquoted_yaml_time_limit_is_refused(tmp_path):
    # YAML 1.1 reads 10:00:00 as the base-60 integer 36000, which Slurm would
    # take as 36000 minutes.
    p = tmp_path / "slurm.yaml"
    p.write_text("time_limit: 10:00:00\ncpus_per_task: 8\n")
    assert yaml.safe_load(p.read_text())["time_limit"] == 36000
    with pytest.raises(slurm.ConfigError, match="quoted string"):
        slurm.load_config(p)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"cpus_per_task": 0}, "cpus_per_task must be a positive integer"),
        ({"cpus_per_task": True}, "cpus_per_task must be a positive integer"),
        ({"timing_workers": 1.5}, "timing_workers must be a positive integer"),
        ({"partition": "a b"}, "partition 'a b' is not a plain Slurm value"),
        ({"partition": "x\n#SBATCH --account=y"}, "partition"),
        ({"account": "$(touch /tmp/x)"}, "account"),
        ({"constraint": "'cpu'"}, "constraint"),
        ({"extra_sbatch": ["--mail-type=END; rm -rf ~"]}, "extra_sbatch"),
        ({"extra_sbatch": ["-x"]}, "extra_sbatch"),
        ({"extra_sbatch": "--exclusive"}, "extra_sbatch must be a list"),
        ({"setup": ["module load a\nmodule load b"]}, "spans more than one line"),
        ({"bench": ""}, "bench must be a one-line command"),
        ({"exclusive": "yes"}, "exclusive must be true or false"),
        ({"margin_seconds": -1}, "margin_seconds"),
        ({"time_limit": "5"}, "leaves no time for jobs"),
        ({"colour": "red"}, "unknown key 'colour'"),
        ({"cpus_per_task": None}, "cpus_per_task is required"),
    ],
)
def test_bad_settings_are_refused_with_their_name(change, message):
    with pytest.raises(slurm.ConfigError, match=message.replace("$", r"\$").replace("(", r"\(")):
        slurm.validate_config({**MINIMAL, **change})


def test_every_bad_setting_is_reported_at_once():
    with pytest.raises(slurm.ConfigError) as err:
        slurm.validate_config({**MINIMAL, "cpus_per_task": 0, "partition": "a b"})
    assert len(err.value.problems) == 2


def test_missing_config_points_at_the_example(tmp_path):
    with pytest.raises(slurm.ConfigError, match="slurm.example.yaml"):
        slurm.load_config(tmp_path / "slurm.yaml")


# --- packing -----------------------------------------------------------------


def _greedy(costs: list[float], workers: int) -> float:
    """Makespan when each free worker takes the next cost in list order."""
    if not costs:
        return 0.0
    free = [0.0] * workers
    end = 0.0
    for c in costs:
        start = heapq.heappop(free)
        heapq.heappush(free, start + c)
        end = max(end, start + c)
    return end


def simulate(task_jobs: list[Job], cfg: slurm.SlurmConfig) -> float:
    """How long the runner takes on one task: all correctness jobs, then all
    timing jobs, each phase in list order."""
    c = [j.est_seconds for j in task_jobs if j.kind == CORRECTNESS]
    t = [j.est_seconds for j in task_jobs if j.kind == TIMING]
    return _greedy(c, cfg.cpus_per_task) + _greedy(t, cfg.timing_workers)


@pytest.mark.parametrize("seed", range(20))
def test_phase_bound_is_never_below_the_runners_makespan(seed):
    rng = random.Random(seed)
    costs = [rng.expovariate(1 / 100) for _ in range(rng.randint(1, 60))]
    w = rng.randint(1, 16)
    assert slurm.phase_bound(costs, w) >= _greedy(costs, w) - 1e-9


def test_phase_bound_on_one_worker_is_the_sum():
    assert slurm.phase_bound([3.0, 4.0, 5.0], 1) == 12.0


def make_jobs(costs: list[tuple[float, float | None]]) -> list[Job]:
    """One problem per (correctness cost, timing cost or None), in runner order."""
    corr, timing = [], []
    for i, (c, t) in enumerate(costs):
        cj = Job("ode", f"p{i:03d}", "a-vs-b", "auto", CORRECTNESS, est_seconds=c)
        corr.append(cj)
        if t is not None:
            timing.append(Job("ode", f"p{i:03d}", "a-vs-b", "auto", TIMING,
                              est_seconds=t, requires=cj.id))
    return corr + timing


def cfg_with(**kw) -> slurm.SlurmConfig:
    base = {"time_limit": "20", "cpus_per_task": 1, "margin_seconds": 100,
            "startup_seconds": 100}  # 1000 s for jobs
    return slurm.validate_config({**base, **kw})


@pytest.mark.parametrize("seed", range(15))
def test_every_packed_task_finishes_inside_the_allocation(seed):
    rng = random.Random(seed)
    cfg = cfg_with(time_limit="2:00:00", cpus_per_task=rng.choice([1, 4, 16, 64]),
                   timing_workers=rng.choice([1, 2]))
    costs = [(rng.uniform(5, 300), rng.choice([None, rng.uniform(10, 600)]))
             for _ in range(rng.randint(1, 200))]
    js = make_jobs(costs)
    packing = slurm.pack(js, cfg)
    packed = [j.id for t in packing.tasks for j in t.jobs]
    assert sorted(packed) == sorted(j.id for j in js)  # each job exactly once
    assert not packing.long
    for t in packing.tasks:
        assert simulate(t.jobs, cfg) <= cfg.capacity_seconds + 1e-9
        assert t.bound(cfg) <= cfg.capacity_seconds + 1e-9
        # a timing job is in the same task as the correctness job it requires
        ids = {j.id for j in t.jobs}
        assert all(j.requires in ids for j in t.jobs if j.requires)
        # correctness jobs come first, as the runner runs them
        kinds = [j.kind for j in t.jobs]
        assert kinds == sorted(kinds, key=jobs.KINDS.index)


def test_packing_known_answer():
    # 10 correctness jobs of 400 s, one worker, 1000 s per task: two per task.
    packing = slurm.pack(make_jobs([(400.0, None)] * 10), cfg_with())
    assert [len(t.jobs) for t in packing.tasks] == [2] * 5


def test_parallel_workers_share_a_task():
    # 64 jobs of 100 s on 64 workers take 100 s: one task.
    packing = slurm.pack(make_jobs([(100.0, None)] * 64), cfg_with(cpus_per_task=64))
    assert len(packing.tasks) == 1


def test_a_problem_too_long_for_one_allocation_is_set_aside():
    js = make_jobs([(100.0, 950.0), (100.0, 100.0)])
    packing = slurm.pack(js, cfg_with())
    assert [[j.problem for j in u] for u in packing.long] == [["p000", "p000"]]
    assert [j.problem for t in packing.tasks for j in t.jobs] == ["p001", "p001"]


# --- which jobs are submitted -------------------------------------------------

FPS = {
    CORRECTNESS: {"bngsim": {"build_commit": "abc"}},
    TIMING: {"bngsim": {"build_commit": "abc"}, "node_type": "login node"},
}


def record(res_dir, job, outcome, fp):
    results.write_result(res_dir, job, {"outcome": outcome, "key": results.freshness_key(job, fp),
                                        "key_parts": results.key_parts(job, fp)})


@pytest.fixture
def pair():
    js = make_jobs([(10.0, 20.0)])
    return js[0], js[1]


def test_only_jobs_without_a_current_result_are_submitted(tmp_path, pair):
    c, t = pair
    assert slurm.to_run([c, t], tmp_path, FPS) == [c, t]  # nothing yet: both
    record(tmp_path, c, "PASS", FPS[CORRECTNESS])
    assert slurm.to_run([c, t], tmp_path, FPS) == [t]  # correctness current
    record(tmp_path, t, "PASS", FPS[TIMING])
    assert slurm.to_run([c, t], tmp_path, FPS) == []


def test_timing_job_behind_a_failed_correctness_job_is_not_submitted(tmp_path, pair):
    c, t = pair
    record(tmp_path, c, "DIFF", FPS[CORRECTNESS])
    assert slurm.to_run([c, t], tmp_path, FPS) == []


def test_stale_correctness_resubmits_its_timing_job_too(tmp_path, pair):
    c, t = pair
    old = {CORRECTNESS: {"bngsim": {"build_commit": "old"}},
           TIMING: {"bngsim": {"build_commit": "old"}, "node_type": "login node"}}
    record(tmp_path, c, "PASS", old[CORRECTNESS])
    record(tmp_path, t, "PASS", old[TIMING])
    assert slurm.to_run([c, t], tmp_path, FPS) == [c, t]


# --- submit --------------------------------------------------------------------


@pytest.fixture
def setup(tmp_path, spec_data, write_spec):
    """Two problems on disk; their jobs; a config; paths for a submission."""
    problems_dir = tmp_path / "problems"
    write_spec(spec_data, directory=problems_dir)
    write_spec(dict(spec_data, id="second_model"), directory=problems_dir)
    problems = specs.load_all(problems_dir)
    all_jobs = jobs.jobs_for(problems)
    paths = {"root": tmp_path, "problems_dir": problems_dir,
             "results_dir": tmp_path / "results", "cache_dir": tmp_path / "cache",
             "out_dir": tmp_path / "slurm"}
    return problems, all_jobs, paths


class Recorder:
    def __init__(self, job_id="4242", error=None):
        self.calls: list = []
        self.job_id, self.error = job_id, error

    def __call__(self, *args):
        self.calls.append(args)
        if self.error:
            raise self.error
        return self.job_id


def do_submit(setup, cfg, **kw):
    problems, all_jobs, paths = setup
    kw.setdefault("check", Recorder())
    kw.setdefault("sbatch", Recorder())
    return slurm.submit(all_jobs, problems, cfg, fingerprints=FPS, **paths, **kw)


def test_dry_run_writes_the_submission_without_submitting(setup):
    sbatch = Recorder()
    sub = do_submit(setup, cfg_with(cpus_per_task=4), dry_run=True, sbatch=sbatch)
    assert sbatch.calls == []
    listed = [ln for f in sorted((sub.directory / "tasks").glob("*.txt"))
              for ln in f.read_text().splitlines()]
    assert sorted(listed) == sorted(j.id for j in setup[1])
    plan = json.loads((sub.directory / "plan.json").read_text())
    assert plan["slurm_job_id"] is None
    assert [t["jobs"] for t in plan["tasks"]] == [
        (sub.directory / "tasks" / f"{i:04d}.txt").read_text().split() for i in range(len(plan["tasks"]))]
    assert (sub.directory / "logs").is_dir()  # Slurm does not create it
    syntax = subprocess.run(["bash", "-n", str(sub.directory / "submit.sbatch")],
                            capture_output=True, text=True, check=False)
    assert syntax.returncode == 0, syntax.stderr


def test_submit_calls_sbatch_once_and_records_the_job_id(setup):
    sbatch = Recorder("98765")
    sub = do_submit(setup, cfg_with(cpus_per_task=4), sbatch=sbatch)
    assert sbatch.calls == [(sub.directory / "submit.sbatch",)]
    assert sub.job_id == "98765"
    assert json.loads((sub.directory / "plan.json").read_text())["slurm_job_id"] == "98765"


def test_nothing_to_run_writes_nothing(setup):
    _, all_jobs, paths = setup
    for j in all_jobs:
        record(paths["results_dir"], j, "PASS", FPS[j.kind])
    sbatch = Recorder()
    sub = do_submit(setup, cfg_with(), sbatch=sbatch)
    assert sub.directory is None and sbatch.calls == []
    assert not paths["out_dir"].exists()


def test_preflight_sees_only_the_needed_problems_and_runs_before_writing(setup):
    _, all_jobs, paths = setup
    for j in all_jobs:
        if j.problem == "second_model":
            record(paths["results_dir"], j, "PASS", FPS[j.kind])
    check = Recorder()
    do_submit(setup, cfg_with(), check=check, dry_run=True)
    assert [[p.id for p in c[0]] for c in check.calls] == [["toy_model"]]
    with pytest.raises(slurm.SubmitError, match="offline"):
        do_submit(setup, cfg_with(), check=Recorder(error=slurm.SubmitError("offline")))
    assert len(list(paths["out_dir"].iterdir())) == 1  # only the dry run's directory


def test_too_long_problems_stop_the_submission_unless_skipped(setup):
    # The toy jobs are estimated at about 21 s and 64.5 s; 30 s leaves no room.
    cfg = cfg_with(time_limit="0:40", margin_seconds=5, startup_seconds=5)
    with pytest.raises(slurm.SubmitError, match="longer than one allocation"):
        do_submit(setup, cfg)
    sub = do_submit(setup, cfg, skip_long=True)
    assert sub.directory is None  # nothing left that fits
    assert len(sub.packing.long) == 2


def test_more_tasks_than_the_array_allows_are_refused(setup):
    cfg = cfg_with(max_array_size=1, time_limit="2", margin_seconds=10, startup_seconds=10)
    with pytest.raises(slurm.SubmitError, match="exceed max_array_size 1"):
        do_submit(setup, cfg)


def test_node_type_judges_timing_results_for_the_compute_nodes(setup):
    _, all_jobs, paths = setup
    compute = {**FPS[TIMING], "node_type": "compute node"}
    for j in all_jobs:
        record(paths["results_dir"], j, "PASS", FPS[CORRECTNESS] if j.kind == CORRECTNESS else compute)
    # From the login node, the compute nodes' timing results look stale ...
    sub = do_submit(setup, cfg_with(), dry_run=True)
    assert {j.kind for t in sub.packing.tasks for j in t.jobs} == {TIMING}
    # ... unless the config says which node type the tasks run on.
    sub = do_submit(setup, cfg_with(node_type="compute node"), dry_run=True)
    assert sub.directory is None


def test_a_path_a_sbatch_line_cannot_hold_is_refused(setup):
    problems, all_jobs, paths = setup
    paths = {**paths, "out_dir": paths["root"] / "my runs"}
    with pytest.raises(slurm.SubmitError, match="without spaces"):
        slurm.submit(all_jobs, problems, cfg_with(), fingerprints=FPS, check=Recorder(),
                     sbatch=Recorder(), **paths)


# --- the batch script, run by bash ----------------------------------------------

FAKE_BENCH = textwrap.dedent("""\
    import json, os, sys
    with open(os.environ["FAKE_BENCH_OUT"], "w") as f:
        json.dump({"argv": sys.argv[1:], "cwd": os.getcwd(),
                   "from_setup": os.environ.get("FROM_SETUP")}, f)
""")


def test_batch_script_runs_bench_on_its_own_task_file(setup, tmp_path):
    fake = tmp_path / "fake_bench.py"
    fake.write_text(FAKE_BENCH)
    cfg = cfg_with(time_limit="1:00:00", cpus_per_task=4, timing_workers=2, margin_seconds=120,
                   bench=f"{sys.executable} {fake}", setup=["export FROM_SETUP=yes"],
                   partition="debug", account="proj-1", constraint="cpu&fast",
                   extra_sbatch=["--mail-type=END"], max_concurrent=3)
    sub = do_submit(setup, cfg, dry_run=True)
    script = (sub.directory / "submit.sbatch").read_text()
    for line in ("#SBATCH --time=1:00:00", "#SBATCH --cpus-per-task=4",
                 "#SBATCH --partition=debug", "#SBATCH --account=proj-1",
                 "#SBATCH --constraint=cpu&fast", "#SBATCH --mail-type=END",
                 f"#SBATCH --array=0-{len(sub.packing.tasks) - 1}%3",
                 f"#SBATCH --output={sub.directory}/logs/%A_%a.out"):
        assert line in script.splitlines()
    out = tmp_path / "bench-args.json"
    env = {**os.environ, "SLURM_ARRAY_TASK_ID": "0", "FAKE_BENCH_OUT": str(out)}
    proc = subprocess.run(["bash", str(sub.directory / "submit.sbatch")], env=env,
                          capture_output=True, text=True, timeout=60, check=False)
    assert proc.returncode == 0, proc.stderr
    got = json.loads(out.read_text())
    argv = got["argv"]
    assert argv[:3] == ["run", "--jobs-from", str(sub.directory / "tasks" / "0000.txt")]
    opts = dict(zip(argv[3::2], argv[4::2], strict=True))
    paths = setup[2]
    assert opts["--problems"] == str(paths["problems_dir"].resolve())
    assert opts["--results"] == str(paths["results_dir"].resolve())
    assert opts["--cache"] == str(paths["cache_dir"].resolve())
    assert (opts["--workers"], opts["--timing-workers"], opts["--margin"]) == ("4", "2", "120")
    assert 3600 - 10 <= int(opts["--budget"]) <= 3600  # the rest of the allocation
    assert got["from_setup"] == "yes"
    assert Path(got["cwd"]).resolve() == paths["root"].resolve()


# --- sbatch itself -------------------------------------------------------------


@pytest.fixture
def fake_sbatch(tmp_path, monkeypatch):
    def install(body: str) -> Path:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir(exist_ok=True)
        exe = bin_dir / "sbatch"
        exe.write_text("#!/bin/sh\n" + body)
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
        return tmp_path / "sbatch-args"

    return install


def test_run_sbatch_reads_the_parsable_job_id(fake_sbatch, tmp_path):
    args = fake_sbatch(f'echo "$@" > {tmp_path}/sbatch-args\necho "4242;cluster1"\n')
    assert slurm.run_sbatch(tmp_path / "submit.sbatch") == "4242"
    assert args.read_text().split() == ["--parsable", str(tmp_path / "submit.sbatch")]


@pytest.mark.parametrize("body, message", [
    ('echo "sbatch: error: invalid partition" >&2\nexit 1\n', "invalid partition"),
    ('echo "Submitted batch job 7"\n', "could not read a job id"),
])
def test_run_sbatch_failures_are_reported(fake_sbatch, tmp_path, body, message):
    fake_sbatch(body)
    with pytest.raises(slurm.SubmitError, match=message):
        slurm.run_sbatch(tmp_path / "submit.sbatch")


def test_missing_sbatch_suggests_a_dry_run(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(slurm.SubmitError, match="--dry-run"):
        slurm.run_sbatch(tmp_path / "submit.sbatch")


# --- the command line ----------------------------------------------------------


def test_cli_submit_end_to_end_with_a_fake_sbatch(setup, fake_sbatch, monkeypatch, capsys):
    _, _, paths = setup
    config = paths["root"] / "slurm.yaml"
    config.write_text(yaml.safe_dump({"time_limit": "1:00:00", "cpus_per_task": 8}))
    fake_sbatch('echo 555\n')
    monkeypatch.setattr(slurm, "preflight", lambda problems, cache: None)
    monkeypatch.setattr(results, "fingerprints_now", lambda: FPS)
    args = ["--problems", str(paths["problems_dir"]), "--results", str(paths["results_dir"]),
            "--cache", str(paths["cache_dir"]), "--config", str(config),
            "--out", str(paths["out_dir"])]
    assert cli.main(["submit", *args]) == 0
    out = capsys.readouterr().out
    assert "4 jobs in 1 array task of 8 CPUs" in out and "submitted Slurm job 555" in out
    # A bad config is refused before anything is written.
    config.write_text("time_limit: 1\n")
    assert cli.main(["submit", *args]) == 2
    assert len(list(paths["out_dir"].iterdir())) == 1


def test_run_jobs_from_refuses_unknown_ids_and_patterns(setup, tmp_path):
    _, all_jobs, paths = setup
    listed = tmp_path / "task.txt"
    listed.write_text(f"# a comment\n{all_jobs[0].id}\nnot/a/job\n")
    base = ["run", "--problems", str(paths["problems_dir"]), "--results", str(paths["results_dir"]),
            "--jobs-from", str(listed)]
    with pytest.raises(SystemExit, match="no such job: not/a/job"):
        cli.main(base)
    listed.write_text(f"{all_jobs[0].id}\n")
    with pytest.raises(SystemExit, match="not both"):
        cli.main([*base, "toy"])


def test_select_exact_adds_the_required_correctness_job():
    js = make_jobs([(1.0, 2.0), (1.0, 2.0)])
    timing = [j for j in js if j.kind == TIMING]
    assert jobs.select_exact(js, [timing[1].id]) == [js[1], timing[1]]
    with pytest.raises(ValueError, match="no such job: x, y"):
        jobs.select_exact(js, ["x", timing[0].id, "y"])
