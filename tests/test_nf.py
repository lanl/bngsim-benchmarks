"""The network-free capability's own logic, with fake engines (no simulator): specs and
jobs, seeds, the references that apply, how per-engine ensembles become an outcome,
the cross-setting and same-seed records, re-scoring from stored ensembles, timing, and
the two standalone references' adapters driven by stand-in executables."""

from __future__ import annotations

import json
import os
import stat
import types
from pathlib import Path

import numpy as np
import pytest

from bench import nfactions, specs
from bench.capabilities import nf

ACTIONS = 'simulate({method=>"nf",t_end=>10,n_steps=>10,seed=>4})'
PROTOCOL = {"actions": ACTIONS, "replicates": 200}


@pytest.fixture
def nf_spec(spec_data):
    spec_data["capabilities"] = ["nf"]
    spec_data["protocol"] = {"nf": dict(PROTOCOL)}
    spec_data["size"] = {"nf_events": 1000}
    return spec_data


# --------------------------------------------------------------------------- #
# Specs, jobs and planning
# --------------------------------------------------------------------------- #
def test_two_settings_each_with_a_correctness_and_a_timing_job(nf_spec, write_spec):
    problem = specs.load(write_spec(nf_spec))
    jobs = nf.jobs(problem)
    assert [(j.setting, j.kind) for j in jobs] == [
        ("nf_reject", "correctness"), ("nf_reject", "timing"),
        ("nf_exact", "correctness"), ("nf_exact", "timing")]
    c = jobs[0]
    assert c.id == "nf/toy_model/bngsim-vs-nfsim-rulemonkey/nf_reject/correctness"
    assert jobs[1].requires == c.id and jobs[3].requires == jobs[2].id
    assert c.definition["setting"] == {"method": "nf_reject", "session": "NfsimSession"}
    assert c.definition["verdict"] == nf.VERDICT and "verdict" not in jobs[1].definition
    assert c.definition["protocol"]["seed"] == nf.DEFAULT_SEED
    assert c.definition["protocol"]["timing_replicates"] == nf.DEFAULT_TIMING_REPLICATES


def test_a_finite_network_adds_exact_ssa_references(nf_spec, write_spec):
    nf_spec["protocol"]["nf"]["netgen"] = "generate_network({overwrite=>1})"
    problem = specs.load(write_spec(nf_spec))
    assert nf.references(problem) == ["nfsim", "rulemonkey", "run_network", "bngsim_ssa"]
    assert nf.jobs(problem)[0].engines == "bngsim-vs-nfsim-rulemonkey-run_network-bngsim_ssa"


def test_the_rulemonkey_build_keys_only_this_capability(nf_spec, write_spec, monkeypatch):
    monkeypatch.setattr(nf.env, "rulemonkey_identity",
                        lambda: {"version": "3.13.0", "commit": "ad3bd75"})
    c = nf.jobs(specs.load(write_spec(nf_spec)))[0]
    assert c.definition["reference_versions"] == {
        "rulemonkey": {"version": "3.13.0", "commit": "ad3bd75"}}


def test_a_non_bngl_source_gets_no_jobs_and_is_refused(nf_spec, write_spec):
    nf_spec["source"]["format"] = "sbml"
    with pytest.raises(specs.SpecError, match="needs a BNGL source"):
        specs.load(write_spec(nf_spec))


@pytest.mark.parametrize("change, message", [
    ({"replicates": 50}, "replicates: an integer from 100"),
    ({"timing_replicates": 0}, "timing_replicates"),
    ({"seed": -1}, "seed"),
    ({"actions": ""}, "actions: 'own'"),
    ({"actions": 'simulate({method=>"ssa",t_end=>1,n_steps=>1})'}, "not network-free"),
    ({"netgen": "simulate({})"}, "netgen"),
    ({"t_end": 5}, "unknown key 't_end'"),
])
def test_an_invalid_nf_protocol_is_refused_with_its_name(nf_spec, write_spec, change, message):
    nf_spec["protocol"]["nf"].update(change)
    with pytest.raises(specs.SpecError, match=message.replace("(", r"\(")):
        specs.load(write_spec(nf_spec))


def test_own_actions_are_accepted_by_the_spec_and_read_at_run_time(nf_spec, write_spec):
    nf_spec["protocol"]["nf"]["actions"] = "own"
    assert specs.load(write_spec(nf_spec)).protocol_for("nf")["actions"] == "own"


def test_planning_grows_with_replicates_and_events_and_the_budget_is_bounded(nf_spec,
                                                                              write_spec):
    small = specs.load(write_spec(nf_spec))
    nf_spec["protocol"]["nf"]["replicates"] = 5000
    nf_spec["size"]["nf_events"] = 10**6
    big = specs.load(write_spec(nf_spec))
    (c_small, t_small), (c_big, t_big) = nf.planning_seconds(small), nf.planning_seconds(big)
    assert c_big > 10 * c_small and t_big > t_small > 0
    assert nf.engine_budget_seconds(small) == nf.MIN_TIMEOUT_SECONDS
    assert nf.engine_budget_seconds(big) == nf.ENGINE_BUDGET_CAP_SECONDS


def test_every_engine_draws_from_its_own_seed_block_and_steps_never_meet():
    firsts = [nf.seed_for(name, 1) for name in nf.ENGINE_SEED_BLOCKS]
    assert len(set(firsts)) == len(firsts)
    gaps = sorted(firsts)
    assert min(b - a for a, b in zip(gaps, gaps[1:])) == nf.SEED_BLOCK
    last = nf.phase_seed(nf.seed_for("bngsim_ssa", 2**30) + specs.NF_MAX_REPLICATES,
                         nf.MAX_PHASES - 1)
    assert last < 2**31  # NFsim and run_network read a signed 32-bit seed
    assert nf.PHASE_SEED_STRIDE * nf.MAX_PHASES <= nf.SEED_BLOCK
    assert nf.PHASE_SEED_STRIDE >= specs.NF_MAX_REPLICATES


# --------------------------------------------------------------------------- #
# Which references apply
# --------------------------------------------------------------------------- #
TWO_STEP = nfactions.protocol(
    'simulate_nf({t_end=>5,n_steps=>5})\nsetParameter("k",2)\nsimulate_nf({t_end=>10,n_steps=>5})')


def test_rulemonkey_and_exact_ssa_cannot_replay_a_multi_step_protocol():
    prep = {"nf": TWO_STEP, "net": Path("m.net")}
    assert nf.RuleMonkeyCLI.unsupported(prep).startswith("UNSUPPORTED")
    assert nf.RunNetworkSSA.unsupported(prep).startswith("UNSUPPORTED")
    assert nf.BngsimSSA.unsupported({"nf": nfactions.protocol(ACTIONS), "net": None}) \
        .startswith("UNSUPPORTED")
    assert nf.NfsimBNG2.unsupported(prep) is None
    assert nf.RuleMonkeyCLI.unsupported({"nf": nfactions.protocol(ACTIONS)}) is None


def test_the_grid_is_every_steps_points_from_its_own_start():
    assert nf.times(TWO_STEP).tolist() == [0, 1, 2, 3, 4, 5] * 2


# --------------------------------------------------------------------------- #
# Correctness and timing, with fake engines
# --------------------------------------------------------------------------- #
N_T, N_OBS = 11, 2
NAMES = ["A", "B"]


def _fake_stack(n, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    lam = 5 * np.exp(-np.linspace(0, 1, N_T))[:, None] * np.ones(N_OBS) + shift
    out = rng.poisson(lam, (n, N_T, N_OBS)).astype(float)
    out[:, 0, :] = 5.0
    return out


def _prep():
    proto = dict(PROTOCOL, seed=1, timing_replicates=8)
    return {"proto": proto, "nf": nfactions.protocol(ACTIONS), "names": NAMES,
            "xml": Path("m.xml"), "net": None, "text": "", "info": {"xml_sha256": "x"}}


def _fake_bngsim(shift=None, boom=None):
    shift = shift or {}

    class FakeBngsim:
        NAME = "bngsim"

        def __init__(self, setting, xml, proto, names, v1143_compat=False):
            if boom and setting in boom:
                raise RuntimeError(boom[setting])
            self.setting = setting
            self.split = {"initialize_seconds": 0.0, "simulate_seconds": 0.0}

        def ensemble(self, n, seed):
            self.split = {"initialize_seconds": 0.01 * n, "simulate_seconds": 0.02 * n}
            return _fake_stack(n, seed, shift.get(self.setting, 0.0))

        def config(self):
            return {"method": self.setting, "load_seconds": 0.01, **self.split}

    return FakeBngsim


def _fake_reference(shift=0.0, boom=None, unsupported=None):
    class FakeRef:
        def __init__(self, *args, **kwargs):
            self.split = {"solve_seconds": 0.0}

        @staticmethod
        def unsupported_(prep):
            return unsupported

        def ensemble(self, n, seed):
            if boom:
                raise RuntimeError(boom)
            self.split = {"solve_seconds": 0.5 * n}
            # Same-seed twin of bngsim: the same law, and the same draws per seed.
            return _fake_stack(n, seed, shift)

        def config(self):
            return {"fake": True, **self.split}

    FakeRef.unsupported = staticmethod(lambda prep: unsupported)
    return FakeRef


def _install(monkeypatch, refs, bngsim=None, applicable=None):
    monkeypatch.setattr(nf, "_prepare", lambda problem, ctx: _prep())
    monkeypatch.setattr(nf, "BngsimNF", bngsim or _fake_bngsim())
    monkeypatch.setattr(nf, "REFERENCE_ENGINES", refs)
    monkeypatch.setattr(nf, "references", lambda problem: list(applicable or refs))
    monkeypatch.setattr(nf, "_build_reference",
                        lambda name, prep, workdir, timeout: refs[name]())


def _job(setting="nf_exact", kind="correctness", engines="bngsim-vs-nfsim-rulemonkey",
         requires=None):
    return types.SimpleNamespace(
        setting=setting, kind=kind, engines=engines, requires=requires,
        definition={"protocol": dict(PROTOCOL, seed=1),
                    "timing": {"warmup_rounds": 1, "min_rounds": 3, "max_rounds": 3,
                               "budget_seconds": 0.0}})


def _ctx(tmp_path, **kw):
    return {"workdir": str(tmp_path), "subprocess_timeout": 60, "engine_budget_seconds": 60,
            **kw}


def test_correctness_passes_and_records_cross_setting_and_same_seed(monkeypatch, tmp_path):
    _install(monkeypatch, {"nfsim": _fake_reference(), "rulemonkey": _fake_reference()})
    rec = nf.run_correctness(_job("nf_exact"), None, _ctx(tmp_path))
    assert rec["outcome"] == "PASS", rec.get("verdicts")
    assert set(rec["verdicts"]) == {"nfsim", "rulemonkey"}
    for v in rec["verdicts"].values():
        assert v["passed"] and v["replicates"] == [200, 200] and "parity_core" in v
    m = rec["measurements"]
    # The other setting is drawn and compared, never scored.
    assert m["cross_setting"]["setting"] == "nf_reject"
    assert m["cross_setting"]["verdict"]["passed"]
    # nf_exact against rm_driver, seed for seed: the fakes draw the same per seed.
    assert m["same_seed"] == {"engine": "rulemonkey", "trajectories": 5, "identical": 5,
                              "max_abs_difference": 0.0}
    assert m["rulemonkey_lineage"].startswith("same code lineage")
    arrays = rec["_trajectories"]
    assert {"bngsim", "nfsim", "rulemonkey", "cross_setting", "observables", "times"} \
        <= set(arrays)
    assert arrays["bngsim"].shape == (200, N_T, N_OBS)


def test_the_reject_setting_checks_its_embedding_against_nfsim(monkeypatch, tmp_path):
    _install(monkeypatch, {"nfsim": _fake_reference(), "rulemonkey": _fake_reference()})
    rec = nf.run_correctness(_job("nf_reject"), None, _ctx(tmp_path))
    assert rec["measurements"]["same_seed"]["engine"] == "nfsim"
    assert rec["measurements"]["cross_setting"]["setting"] == "nf_exact"


def test_bngsim_sampling_another_law_is_a_diff(monkeypatch, tmp_path):
    _install(monkeypatch, {"nfsim": _fake_reference(), "rulemonkey": _fake_reference()},
             bngsim=_fake_bngsim(shift={"nf_exact": 1.5}))
    rec = nf.run_correctness(_job("nf_exact"), None, _ctx(tmp_path))
    assert rec["outcome"] == "DIFF"
    assert not rec["measurements"]["cross_setting"]["verdict"]["passed"]


def test_the_other_setting_disagreeing_does_not_fail_this_one(monkeypatch, tmp_path):
    _install(monkeypatch, {"nfsim": _fake_reference(), "rulemonkey": _fake_reference()},
             bngsim=_fake_bngsim(shift={"nf_reject": 1.5}))
    rec = nf.run_correctness(_job("nf_exact"), None, _ctx(tmp_path))
    assert rec["outcome"] == "PASS"
    assert not rec["measurements"]["cross_setting"]["verdict"]["passed"]


def test_a_failing_or_unsupported_reference_is_a_reference_failure(monkeypatch, tmp_path):
    _install(monkeypatch, {"nfsim": _fake_reference(boom="NaN stop"),
                           "rulemonkey": _fake_reference(unsupported="UNSUPPORTED: two steps")})
    rec = nf.run_correctness(_job("nf_exact"), None, _ctx(tmp_path))
    assert rec["outcome"] == "REFERENCE_FAILED"
    assert "NaN stop" in rec["nfsim_error"]
    assert rec["rulemonkey_error"] == "UNSUPPORTED: two steps"
    assert "same_seed" not in rec["measurements"]


def test_bngsim_failing_is_an_exception_and_with_every_reference_a_bad_test(monkeypatch,
                                                                             tmp_path):
    _install(monkeypatch, {"nfsim": _fake_reference(), "rulemonkey": _fake_reference()},
             bngsim=_fake_bngsim(boom={"nf_exact": "no"}))
    rec = nf.run_correctness(_job("nf_exact"), None, _ctx(tmp_path))
    assert rec["outcome"] == "EXCEPTION" and "no" in rec["bngsim_error"]
    _install(monkeypatch, {"nfsim": _fake_reference(boom="x"),
                           "rulemonkey": _fake_reference(boom="y")},
             bngsim=_fake_bngsim(boom={"nf_exact": "no"}))
    assert nf.run_correctness(_job("nf_exact"), None, _ctx(tmp_path))["outcome"] == "BAD_TEST"


def test_preparation_failure_is_a_bad_test(monkeypatch, tmp_path):
    def boom(problem, ctx):
        raise nfactions.ActionError("the action parameter_scan cannot be replayed")

    monkeypatch.setattr(nf, "_prepare", boom)
    for run in (nf.run_correctness, nf.run_timing):
        rec = run(_job(), None, _ctx(tmp_path))
        assert rec["outcome"] == "BAD_TEST" and "parameter_scan" in rec["error"]


def test_rescoring_the_stored_ensembles_reproduces_the_verdicts(monkeypatch, tmp_path):
    _install(monkeypatch, {"nfsim": _fake_reference(), "rulemonkey": _fake_reference(boom="x")})
    rec = nf.run_correctness(_job("nf_exact"), None, _ctx(tmp_path))
    arrays = rec.pop("_trajectories")
    update = nf.rescore(_job("nf_exact"), json.loads(json.dumps(rec, default=str)), arrays)
    assert update["outcome"] == rec["outcome"] == "REFERENCE_FAILED"
    assert json.dumps(update["verdicts"], default=str) == json.dumps(rec["verdicts"], default=str)
    assert update["measurements"]["cross_setting"]["verdict"]["p_value"] == \
        rec["measurements"]["cross_setting"]["verdict"]["p_value"]


def test_timing_interleaves_ensembles_and_splits_each_engines_time(monkeypatch, tmp_path):
    _install(monkeypatch, {"nfsim": _fake_reference(), "rulemonkey": _fake_reference()})
    rec = nf.run_timing(_job("nf_reject", "timing"), None, _ctx(tmp_path))
    m = rec["measurements"]
    assert rec["outcome"] == "PASS"
    assert set(m["times"]) == {"bngsim", "nfsim", "rulemonkey"}
    for name, t in m["times"].items():
        assert m["times_per_trajectory"][name]["median_seconds"] == pytest.approx(
            t["median_seconds"] / 8)
    assert all(r["passed"] for r in m["reproducibility"].values())
    assert m["engine_roles"] == {"bngsim": "warm, in-process (system under test)",
                                 "nfsim": "fresh subprocess", "rulemonkey": "fresh subprocess"}
    assert m["solve_split"]["bngsim"] == {"initialize_seconds": 0.08, "simulate_seconds": 0.16,
                                          "solve_seconds": 0.16}
    assert m["solve_split"]["nfsim"] == {"solve_seconds": 4.0}


def test_timing_skips_a_reference_that_timed_out_or_cannot_run(monkeypatch, tmp_path):
    from bench.results import result_path

    corr = "nf/m/x/nf_exact/correctness"
    path = result_path(tmp_path, corr)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"outcome": "REFERENCE_FAILED",
                                "nfsim_error": "TIMEOUT: no result within the 60 s budget"}))
    _install(monkeypatch, {"nfsim": _fake_reference(),
                           "rulemonkey": _fake_reference(unsupported="UNSUPPORTED: two steps")})
    rec = nf.run_timing(_job("nf_exact", "timing", requires=corr), None,
                        _ctx(tmp_path, results_dir=str(tmp_path)))
    assert set(rec["measurements"]["times"]) == {"bngsim"}
    assert rec["nfsim_error"].startswith("TIMEOUT in the correctness job")
    assert rec["rulemonkey_error"] == "UNSUPPORTED: two steps"


# --------------------------------------------------------------------------- #
# The standalone references' adapters, driven by stand-in executables
# --------------------------------------------------------------------------- #
def _exe(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _adapter_prep(tmp_path, actions=ACTIONS):
    xml = tmp_path / "m.xml"
    xml.write_text("<sbml/>")
    from bench import fetch

    return {"nf": nfactions.protocol(actions), "names": ["B", "A"], "xml": xml,
            "text": "begin model\nend model\n", "info": {"xml_sha256": fetch.sha256_file(xml)}}


def test_rulemonkey_adapter_reads_rm_drivers_gdat_by_name_and_its_time(tmp_path, monkeypatch):
    rows = "\n".join(f"{t}\t{t + 1}\t{2 * t}" for t in range(11))
    exe = _exe(tmp_path / "rm" / "bin" / "rm_driver",
               "#!/bin/sh\nprintf '# time\\tA\\tB\\n%s\\n' \"$(cat <<'EOF'\n"
               f"{rows}\nEOF\n)\"\necho 0.25 >&2\n")
    monkeypatch.setattr(nf.env, "rulemonkey_bin", lambda: exe)
    monkeypatch.setattr(nf.env, "rulemonkey_identity", lambda: {"version": "3.13.0"})
    ref = nf.RuleMonkeyCLI(_adapter_prep(tmp_path), timeout=30)
    stack = ref.ensemble(2, 7)
    assert stack.shape == (2, 11, 2)
    assert stack[0, 3].tolist() == [6.0, 4.0]  # columns in the job's order: B, A
    assert ref.split == {"rm_simulate_seconds": 0.5}
    cfg = ref.config()
    assert cfg["command"] == ["rm_driver", "model.xml", "10.0", "10", "<seed>"]
    assert cfg["lineage"] == nf.RULEMONKEY_LINEAGE


FAKE_BNG2 = r"""#!/usr/bin/env perl
# A stand-in for BNG2.pl: reads the model's appended actions and writes, for each
# network-free step, the .gdat and XML BNG2.pl would, and NFsim's report lines.
use strict;
my $file = $ARGV[0];
open(my $in, '<', $file) or die;
my $text = do { local $/; <$in> };
my $k = 0;
while ($text =~ /simulate\w*\(\{(.*?)\}\)/g) {
    my $opts = $1;
    my ($suffix) = $opts =~ /suffix=>"([^"]+)"/;
    my ($seed) = $opts =~ /seed=>(\d+)/;
    my ($n) = $opts =~ /n_steps=>(\d+)/;
    open(my $g, '>', "model_$suffix.gdat") or die;
    print $g "# time B A\n";
    for my $i (0 .. $n) { print $g "$i ", $seed % 1000, " ", $i * 2, "\n"; }
    close $g;
    open(my $x, '>', "model_$suffix.xml") or die;
    print $x "<sbml/>";
    close $x;
    print "full command: $ENV{BNGPATH}/bin/NFsim -xml ./model_$suffix.xml -o ./model_$suffix.gdat -seed $seed -cb\n";
    print "   You just simulated 100 reactions in 0.5s\n";
    print "done.  Total CPU time: 0.75s\n";
    $k++;
}
exit 0;
"""


def test_nfsim_adapter_runs_bng2pl_on_the_protocol_and_reads_each_step(tmp_path, monkeypatch):
    bng = tmp_path / "bng"
    _exe(bng / "BNG2.pl", FAKE_BNG2)
    _exe(bng / "bin" / "NFsim", "#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("BNGPATH", str(bng))
    monkeypatch.setattr(nf.env, "run_network_launch", lambda: [])
    monkeypatch.setattr(nf.env, "nfsim_identity", lambda: {"version": "1.14.3"})
    prep = _adapter_prep(tmp_path, 'simulate_nf({t_end=>4,n_steps=>4,get_final_state=>1})\n'
                                   'setParameter("k",2)\nsimulate_nf({t_end=>8,n_steps=>2})')
    ref = nf.NfsimBNG2(prep, tmp_path / "work", timeout=60)
    stack = ref.ensemble(2, 20_000_001)
    assert stack.shape == (2, 5 + 3, 2)
    # Each step's own seed (the later one 10**6 on), read back by name: B then A.
    assert stack[0, 0].tolist() == [1.0, 0.0] and stack[0, 5].tolist() == [1.0, 0.0]
    assert stack[1, 1].tolist() == [2.0, 2.0]
    bngl = (tmp_path / "work" / "run" / "model.bngl").read_text()
    assert 'setParameter("k",2)' in bngl and 'seed=>21000002' in bngl
    assert ref.split == {"nfsim_simulate_cpu_seconds": 2.0, "nfsim_total_cpu_seconds": 3.0,
                         "reactions": 400}
    cfg = ref.config()
    assert cfg["xml_matches_harness"] is True
    assert cfg["nfsim_command"] == "NFsim -xml model_nfp0.xml -o model_nfp0.gdat -seed <seed> -cb"
    assert str(tmp_path) not in json.dumps(cfg)


def test_nfsim_adapter_reports_a_nan_stop_as_the_known_gap(tmp_path, monkeypatch):
    bng = tmp_path / "bng"
    _exe(bng / "BNG2.pl", FAKE_BNG2.replace('" ", $i * 2,', '" ", ($i > 1 ? "nan" : $i),'))
    monkeypatch.setenv("BNGPATH", str(bng))
    monkeypatch.setattr(nf.env, "run_network_launch", lambda: [])
    ref = nf.NfsimBNG2(_adapter_prep(tmp_path), tmp_path / "work", timeout=60)
    with pytest.raises(RuntimeError, match="nfsim#115"):
        ref.trajectory(5)


def test_nfsim_adapter_names_bng2pls_error(tmp_path, monkeypatch):
    bng = tmp_path / "bng"
    _exe(bng / "BNG2.pl", "#!/usr/bin/env perl\nprint \"ABORT: no such parameter\\n\"; exit 1;\n")
    monkeypatch.setenv("BNGPATH", str(bng))
    monkeypatch.setattr(nf.env, "run_network_launch", lambda: [])
    ref = nf.NfsimBNG2(_adapter_prep(tmp_path), tmp_path / "work", timeout=60)
    with pytest.raises(RuntimeError, match="ABORT: no such parameter"):
        ref.trajectory(5)


def test_observable_names_come_from_the_xml_in_order(tmp_path):
    xml = tmp_path / "m.xml"
    xml.write_text('<sbml xmlns="http://www.sbml.org/sbml/level3"><model><ListOfObservables>'
                   '<Observable id="O1" name="Obs_U" type="Molecules"/>'
                   '<Observable id="O2" name="Obs_P" type="Species"/>'
                   '</ListOfObservables></model></sbml>')
    assert nf.observable_names(xml) == ["Obs_U", "Obs_P"]
    assert os.path.exists(xml)
