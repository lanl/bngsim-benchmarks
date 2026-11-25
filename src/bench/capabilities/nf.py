"""Network-free simulation of BNGL models: BNGsim's two network-free engines against references.

BNGsim simulates a rule-based model without generating its network in two ways,
each the system under test in a **setting** of its own (one job per setting):

- ``nf_reject`` -- the embedded NFsim (rejection of null events, Yang et al. 2008),
  through ``bngsim.NfsimSession``;
- ``nf_exact`` -- the embedded RuleMonkey (exact event times, no rejection), through
  ``bngsim.RuleMonkeySession``.

Both read the BNG-XML that BNG2.pl's ``writeXML()`` writes from the published
BNGL, and both are compared with every reference in ``REFERENCE_ENGINES`` on the
same protocol:

- **nfsim** -- standalone NFsim 2.0.1 (bundled with BioNetGen 2.9.4) as BNG2.pl
  drives it: BNG2.pl runs the protocol's own actions (``bench.nfactions``), writing
  the XML and starting NFsim for each network-free step, with any ``setParameter``
  between steps applied by BNG2.pl, as NFsim's command line cannot change a
  parameter mid-run. A fresh BNG2.pl process per trajectory, its seed replaced.
  BNG2.pl execs ``$BNGPATH/bin/NFsim`` itself, so on an old-glibc host the job runs
  it under a mirror of the distribution whose ``NFsim`` wraps the loader swap
  (``env.bng_launch_tree``); the binary's bytes are untouched. NFsim 1.14.3
  (BioNetGen 2.9.3) is for comparison runs only (``BNGPATH`` naming that
  distribution); which ran is in the key, through BioNetGen's sha256.
- **rulemonkey** -- standalone RuleMonkey's ``rm_driver`` (v3.13.0, built from its
  pinned source by ``bench fetch-rulemonkey``), a fresh process per trajectory, on
  the same XML. It is the **same code lineage** as BNGsim's embedded ``nf_exact``
  engine, so it checks the embedding and is a cost comparator, but it is not an
  algorithmically independent answer. It cannot change a parameter between runs,
  so a multi-step protocol is ``UNSUPPORTED`` for it.
- **run_network** and **bngsim_ssa** -- where the spec names a ``netgen`` (a model
  whose network is finite), exact SSA on the generated network: BioNetGen's
  ``run_network -p ssa`` and BNGsim's own exact SSA, compared on the network's
  observables. Network-free simulation of a finite network samples the same law as
  exact SSA on it, so these are the strictest references there are; one run from
  the seed species only.

**Correctness.** Every engine draws ``replicates`` trajectories, from seeds in a
block of its own, on the protocol's grid (each network-free step's ``n_steps + 1``
points, the steps one after another); the comparison is on the model's
observables, by name. The verdict of BNGsim against each reference is the SSA
capability's: a permutation max-T test on per-cell means and variances, with an
exact false-alarm rate of ``VERDICT["alpha"]`` per comparison, and bngsim's own
``parity_core`` ensemble verdict recorded beside it (docs/DESIGN.md §5.3, §5.5). The
job passes only when BNGsim agrees with every reference. Failures are attributed
as in ``ssa``, with one rule more: ``rm_driver`` is of BNGsim's own code lineage, so
alone it can neither pass nor refute BNGsim (``verdict_outcome``). A PASS needs at
least one independent reference that applies (NFsim always does) to have agreed,
and a disagreement only ``rm_driver`` finds is attributed to it (REFERENCE_FAILED),
not a DIFF. Recorded beside the verdicts, and not gating:

- the other network-free setting of BNGsim, drawn from its own seeds and compared
  the same way (``cross_setting``): the two embedded engines against each other;
- a same-seed check of the embedding (``same_seed``): BNGsim's ``nf_exact`` against
  ``rm_driver``, and BNGsim's ``nf_reject`` against BNG2.pl's NFsim (in BNGsim's NFsim
  v1.14.3 compatibility mode where that NFsim is 1.14.3, a comparison run), on the
  first trajectories of the reference's own seeds. Two copies of one code fed one
  seed should give one trajectory.

Known reference gaps are recorded, never worked around: NFsim stops with NaN on a
rate law that divides by a single-copy count (RuleWorld/nfsim#115), which the
``nfsim`` adapter reports as such.

**Timing.** As in ``ssa``: one timed unit is an ensemble of ``timing_replicates``
trajectories from the same seeds every round, interleaved round-robin with the
references, every kept round reproducing the first. BNGsim's session is built once,
cold (its XML read, ``load_seconds``); each of its trajectories re-initialises the
session (the XML's agents) and simulates, and the two are reported apart. The
references run as fresh processes, so their start-up is inside their time; each
also reports its own solve time (NFsim's "simulated ... in" and total CPU time,
rm_driver's simulation time), recorded per ensemble beside the wall time, so the
pre-simulation and solve costs are told apart for every engine.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from .. import env, fetch, nfactions, stochastic
from ..jobs import CORRECTNESS, TIMING, Job
from ..specs import Problem
from ..timing import absolute_time_stats, round_robin_interleaved
from ._common import (
    all_passed,
    correctness_timeouts,
    engine_role,
    generate_network,
    netgen_bngl,
    network_size,
    outcome,
    progress,
    read_dat,
    recorded_command,
    reproducibility,
    run_isolated,
    threads,
)
from .ssa import VERDICT as SSA_VERDICT

CAPABILITY = "nf"
# Bump when this module's computation changes; every result it wrote goes stale.
# 3: run_network's edgepop questions answered as BNG2.pl does (EDGEPOP_REPLY).
HARNESS_VERSION = "3"
# BNGsim's two network-free engines: each is the system under test of one setting.
SETTINGS = ("nf_reject", "nf_exact")
SESSION = {"nf_reject": "NfsimSession", "nf_exact": "RuleMonkeySession"}

TIMING_PROTOCOL = {"warmup_rounds": 1, "min_rounds": 5, "max_rounds": 20, "budget_seconds": 120.0}
REPRO_CHECK_VERSION = 1
REPRO_TOL = 1e-9

# The SSA capability's ensemble verdict (bench.stochastic), and how the per-reference
# verdicts combine into the outcome (``verdict_outcome``): in the correctness
# definition, so a change restales every correctness result (re-scored by
# ``bench rescore`` from its stored ensembles).
VERDICT = {**SSA_VERDICT, "independent_reference_rule": 1}

DEFAULT_SEED = 1
DEFAULT_TIMING_REPLICATES = 20
# Each engine's trajectories take seeds from a block of its own. Fixed, never
# derived from a registry's order, so adding an engine moves no other's seeds.
SEED_BLOCK = 10_000_000
ENGINE_SEED_BLOCKS = {"nf_reject": 0, "nf_exact": 1, "nfsim": 2, "rulemonkey": 3,
                      "run_network": 4, "bngsim_ssa": 5}
# A later network-free step that starts from the seed species again draws a seed
# of its own, this far into the block per step (replicates <= 10**6, so the steps'
# seeds never meet; specs.py bounds both).
PHASE_SEED_STRIDE = 1_000_000
MAX_PHASES = 9
# Trajectories in the same-seed check of the embedding.
SAME_SEED_TRAJECTORIES = 5
# Fixed relabelling indices (the comparison's position), never the run order.
COMPARISON_INDEX = {"nfsim": 0, "rulemonkey": 1, "run_network": 2, "bngsim_ssa": 3,
                    "cross_setting": 4}
RULEMONKEY_LINEAGE = ("same code lineage as bngsim's embedded nf_exact engine (RuleMonkey "
                      "v3.13.0; bngsim 0.17.2 embeds v3.13.0-5-gdb7d1df): a cost comparator "
                      "and secondary reference, not an algorithmically independent answer")
# References of BNGsim's own network-free code lineage. They are run, compared and
# stored like any other, and a PASS needs their agreement, but alone they can
# neither pass nor refute BNGsim: a PASS needs at least one independent reference
# that applies to have agreed, and a disagreement that only a same-lineage
# reference finds is attributed to it (REFERENCE_FAILED), never a DIFF of BNGsim.
SAME_LINEAGE_REFERENCES = frozenset({"rulemonkey"})


def protocol(problem: Problem) -> dict[str, Any]:
    """The network-free protocol with defaults filled in, as it enters a job."""
    p = dict(problem.protocol_for(CAPABILITY))
    p.setdefault("seed", DEFAULT_SEED)
    p.setdefault("timing_replicates", DEFAULT_TIMING_REPLICATES)
    return p


def seed_for(engine: str, seed: int) -> int:
    """The first seed of ``engine``'s block: trajectory ``i`` uses this plus ``i``."""
    return int(seed) + SEED_BLOCK * ENGINE_SEED_BLOCKS[engine]


def phase_seed(seed: int, phase: int) -> int:
    """The seed of a trajectory's network-free step ``phase`` (step 0: ``seed``)."""
    return int(seed) + PHASE_SEED_STRIDE * int(phase)


def references(problem: Problem) -> list[str]:
    """The references that apply: NFsim and RuleMonkey always; exact SSA on the
    generated network where the spec names one (a finite network)."""
    refs = ["nfsim", "rulemonkey"]
    if protocol(problem).get("netgen"):
        refs += ["run_network", "bngsim_ssa"]
    return refs


def engines_id(problem: Problem) -> str:
    return "bngsim-vs-" + "-".join(references(problem))


def jobs(problem: Problem) -> list[Job]:
    # A BNGL source: BNG2.pl writes the BNG-XML both BNGsim engines and the
    # references read, and runs the protocol for the NFsim reference.
    if problem.source.get("format") != "bngl":
        return []
    proto = protocol(problem)
    engines = engines_id(problem)
    rm = env.rulemonkey_identity()
    out: list[Job] = []
    for setting in SETTINGS:
        base = {
            "harness": f"{CAPABILITY}/{engines}/{HARNESS_VERSION}",
            "source_sha256": problem.source["sha256"],
            "protocol": proto,
            "setting": {"method": setting, "session": SESSION[setting]},
            # The one reference outside the global fingerprint (NFsim rides on
            # BioNetGen's sha256): it keys only this capability.
            "reference_versions": {"rulemonkey": {"version": rm.get("version"),
                                                  "commit": rm.get("commit")}},
        }
        est_c, est_t = planning_seconds(problem)
        n_engines = len(references(problem)) + 2  # + this setting and the other
        backstop = (n_engines + 1) * engine_budget_seconds(problem) \
            + VERDICT_SECONDS * (n_engines + 1)
        correctness = Job(
            CAPABILITY, problem.id, engines, setting, CORRECTNESS,
            definition={**base, "verdict": VERDICT},
            est_seconds=est_c,
            timeout_seconds=max(MIN_TIMEOUT_SECONDS, TIMEOUT_FACTOR * est_c, backstop),
        )
        timing = Job(
            CAPABILITY, problem.id, engines, setting, TIMING,
            definition={**base, "timing": TIMING_PROTOCOL, "timing_check": REPRO_CHECK_VERSION},
            est_seconds=est_t,
            timeout_seconds=max(MIN_TIMEOUT_SECONDS, TIMEOUT_FACTOR * est_t),
            requires=correctness.id,
        )
        out += [correctness, timing]
    return out


# --------------------------------------------------------------------------- #
# Planning (never part of a freshness key)
# --------------------------------------------------------------------------- #
# A trajectory costs a fixed overhead per engine plus a cost per reaction event,
# from the spec's ``size.nf_events`` (planning only). Measured on the pilots
# (Darwin general node): BNG2.pl's own start, parse and writeXML take about 0.7 s
# before NFsim starts; a BNGsim session re-initialises in under 0.1 s; NFsim and
# RuleMonkey spend 2-10 us per event; run_network's launch is about 10 ms.
MIN_TIMEOUT_SECONDS = 1800.0
TIMEOUT_FACTOR = 4.0
ENGINE_BUDGET_CAP_SECONDS = 3600.0
DEFAULT_EVENTS = 100_000
EVENT_SECONDS = 1e-5
TRAJECTORY_SECONDS = {"bngsim": 0.1, "nfsim": 0.8, "rulemonkey": 0.05, "run_network": 0.01,
                      "bngsim_ssa": 1e-3}
VERDICT_SECONDS = 30.0


def trajectory_seconds(problem: Problem) -> dict[str, float]:
    events = problem.size.get("nf_events", DEFAULT_EVENTS)
    per = {"bngsim": TRAJECTORY_SECONDS["bngsim"] + EVENT_SECONDS * events}
    for name in references(problem):
        scale = 0.25 if name in ("run_network", "bngsim_ssa") else 1.0
        per[name] = TRAJECTORY_SECONDS[name] + scale * EVENT_SECONDS * events
    return per


def planning_seconds(problem: Problem) -> tuple[float, float]:
    """Rough wall-clock estimates (correctness, timing) for one job of one setting."""
    proto = protocol(problem)
    per = trajectory_seconds(problem)
    # This setting, the other (cross_setting), and every reference.
    all_engines = sum(per.values()) + per["bngsim"]
    correctness = 60.0 + int(proto["replicates"]) * all_engines \
        + VERDICT_SECONDS * (len(per) + 1)
    timed = per["bngsim"] + per["nfsim"] + per["rulemonkey"]
    tp = TIMING_PROTOCOL
    rnd = int(proto["timing_replicates"]) * timed
    measured = max(tp["min_rounds"] * rnd, min(tp["budget_seconds"] + rnd, tp["max_rounds"] * rnd))
    timing = 60.0 + (tp["warmup_rounds"] + 1) * rnd + measured
    return correctness, timing


def engine_budget_seconds(problem: Problem) -> float:
    """Wall budget for one engine's cold build + ensemble in the correctness job."""
    slowest = max(trajectory_seconds(problem).values()) * int(protocol(problem)["replicates"])
    return min(ENGINE_BUDGET_CAP_SECONDS, max(MIN_TIMEOUT_SECONDS, TIMEOUT_FACTOR * slowest))


# --------------------------------------------------------------------------- #
# Model preparation: the XML, the protocol, and (finite networks) the .net
# --------------------------------------------------------------------------- #
def _bng2(text: str, workdir: Path, *, timeout: float, bngpath: Path | None = None,
          name: str = "model") -> subprocess.CompletedProcess:
    """BNG2.pl on ``text`` (a whole BNGL file) in ``workdir``."""
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / f"{name}.bngl").write_text(text, encoding="utf-8")
    run_env = dict(os.environ)
    root = bngpath or env.bngpath()
    if root is None:
        raise RuntimeError("BNGPATH is not set; run `pixi run fetch-bionetgen` and use `pixi run`")
    run_env["BNGPATH"] = str(root)
    return subprocess.run(["perl", str(root / "BNG2.pl"), f"{name}.bngl"], capture_output=True,
                          text=True, timeout=timeout, cwd=workdir, env=run_env, check=False)


def _tail(proc: subprocess.CompletedProcess) -> str:
    lines = [ln for ln in (proc.stdout + "\n" + proc.stderr).splitlines()
             if ln.strip() and "Possible precedence problem" not in ln]
    errs = [ln for ln in lines if re.search(r"ERROR|ABORT|Error|error", ln)]
    return ((errs or lines or [""])[-1]).strip()[:300]


def observable_names(xml: Path) -> list[str]:
    """The model's observables, in the XML's order (the compared columns)."""
    root = ET.parse(xml).getroot()
    return [el.get("name") or el.get("id") for el in root.iter()
            if el.tag.split("}")[-1] == "Observable"]


def write_xml(bngl_text: str, nf: nfactions.Protocol, workdir: Path, *,
              timeout: float) -> tuple[Path, float]:
    """The BNG-XML every engine but BNG2.pl's NFsim reads: BNG2.pl's ``writeXML()``
    after the protocol's first ``setParameter`` steps, exactly what BNG2.pl writes
    for the first network-free step."""
    pre = "".join(s.render() + "\n" for s in nf.phases[0].sets)
    t0 = time.perf_counter()
    proc = _bng2(netgen_bngl(bngl_text, pre + "writeXML()"), workdir, timeout=timeout)
    seconds = time.perf_counter() - t0
    xml = workdir / "model.xml"
    if proc.returncode != 0 or not xml.exists():
        raise RuntimeError(f"BNG2.pl writeXML failed: {_tail(proc)}")
    return xml, seconds


def _prepare(problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    proto = protocol(problem)
    model = fetch.model_file(problem.source, ctx["cache_dir"])
    text = model.read_text(encoding="utf-8")
    actions = nfactions.own_actions(text) if proto["actions"] == "own" else proto["actions"]
    nf = nfactions.protocol(actions)
    if len(nf.phases) > MAX_PHASES:
        raise nfactions.ActionError(f"{len(nf.phases)} network-free steps (at most {MAX_PHASES})")
    workdir = Path(ctx["workdir"])
    timeout = float(ctx["subprocess_timeout"])
    xml, xml_seconds = write_xml(text, nf, workdir / "xml", timeout=timeout)
    names = observable_names(xml)
    if not names:
        raise RuntimeError("the model declares no observables to compare")
    info: dict[str, Any] = {"xml_seconds": xml_seconds, "protocol": nf.describe(),
                            "observables": names, "xml_sha256": fetch.sha256_file(xml)}
    prep: dict[str, Any] = {"text": text, "nf": nf, "xml": xml, "names": names, "info": info,
                            "proto": proto, "net": None}
    if proto.get("netgen"):
        pre = "".join(s.render() + "\n" for s in nf.phases[0].sets)
        net, netgen_seconds = generate_network(text, pre + proto["netgen"], workdir / "netgen",
                                               timeout=timeout)
        prep["net"] = net
        # Only the exact-SSA references read it: not BNGsim's pre-simulation cost.
        info["ssa_netgen_seconds"] = netgen_seconds
        info["network"] = network_size(net)
    return prep


def times(nf: nfactions.Protocol) -> np.ndarray:
    """The output times of every step, one after another, each step's from 0 (as
    BNG2.pl reports a network-free step: time elapsed since its start)."""
    return np.concatenate([np.linspace(0.0, p.duration, p.n_points) for p in nf.phases])


def _columns(values: np.ndarray, have: list[str], names: list[str], who: str) -> np.ndarray:
    """``values``' columns reordered to ``names`` (an engine's observables by name)."""
    index = {n: i for i, n in enumerate(have)}
    missing = [n for n in names if n not in index]
    if missing:
        raise RuntimeError(f"{who} reports no observable {missing[:3]} (it has {have[:6]})")
    return np.asarray(values, dtype=float)[:, [index[n] for n in names]]


# --------------------------------------------------------------------------- #
# The system under test
# --------------------------------------------------------------------------- #
class BngsimNF:
    """One of BNGsim's network-free engines, through its session API.

    The session is built once (the XML read and checked; ``load_seconds``). A
    trajectory initialises the session from the XML with its seed (after clearing
    any parameter the last one set) and simulates each network-free step in turn:
    a ``setParameter`` before a later step is set on the live session, and a step
    after one that did not keep its final state (``get_final_state=>0``) starts
    from the seed species again, with a seed of its own, as BNG2.pl's would.
    """

    NAME = "bngsim"

    def __init__(self, setting: str, xml: Path, nf: nfactions.Protocol, names: list[str], *,
                 v1143_compat: bool = False):
        import bngsim

        self.setting, self.nf, self.names = setting, nf, names
        kwargs: dict[str, Any] = {"block_same_complex_binding": nf.complex}
        if nf.gml is not None:
            kwargs["molecule_limit"] = nf.gml
        if setting == "nf_reject":
            if nf.utl is not None:
                kwargs["traversal_limit"] = nf.utl
            if v1143_compat:
                kwargs["v1143_compat"] = True
        elif setting != "nf_exact":
            raise ValueError(f"unknown network-free setting {setting!r}")
        self.options = dict(kwargs)
        t0 = time.perf_counter()
        cls = bngsim.NfsimSession if setting == "nf_reject" else bngsim.RuleMonkeySession
        self.session = cls(str(xml), **kwargs)
        self.load_seconds = time.perf_counter() - t0
        self.split = {"initialize_seconds": 0.0, "simulate_seconds": 0.0}
        # Whether a step after the first sets a parameter (on the live session).
        self.sets_live = any(p.sets for p in nf.phases[1:])

    def _value(self, text: str) -> float:
        try:
            return float(text)
        except ValueError:
            return float(self.session.evaluate(text))

    def trajectory(self, seed: int) -> np.ndarray:
        s = self.session
        if self.sets_live:
            # The last trajectory's later steps changed parameters: back to the XML's.
            s.clear_param_overrides()
        rows = []
        clock = 0.0
        for p in self.nf.phases:
            if p.index > 0:
                for change in p.sets:
                    s.set_param(change.name, self._value(change.value))
            if p.index == 0 or not self.nf.phases[p.index - 1].get_final_state:
                t0 = time.perf_counter()
                s.initialize(seed=phase_seed(seed, p.index))
                self.split["initialize_seconds"] += time.perf_counter() - t0
                clock = 0.0
            t0 = time.perf_counter()
            r = s.simulate(clock, clock + p.duration, p.n_points)
            self.split["simulate_seconds"] += time.perf_counter() - t0
            clock += p.duration
            rows.append(_columns(r.observables, list(r.observable_names), self.names,
                                 f"bngsim {self.setting}"))
        return np.concatenate(rows)

    def ensemble(self, n: int, seed: int) -> np.ndarray:
        self.split = {"initialize_seconds": 0.0, "simulate_seconds": 0.0}
        return np.stack([self.trajectory(int(seed) + i) for i in range(int(n))])

    def config(self) -> dict[str, Any]:
        return {"method": self.setting, "session": SESSION[self.setting],
                "options": self.options, "load_seconds": self.load_seconds, **self.split}


# --------------------------------------------------------------------------- #
# The reference engines
# --------------------------------------------------------------------------- #
_NF_SOLVE_RE = re.compile(r"You just simulated (\d+) reactions in ([0-9.eE+-]+)s")
_NF_TOTAL_RE = re.compile(r"Total CPU time: ([0-9.eE+-]+)s")
_NF_COMMAND_RE = re.compile(r"^full command: (.+)$", re.MULTILINE)


class NfsimBNG2:
    """Standalone NFsim as BNG2.pl drives it: the protocol's actions, run by BNG2.pl.

    Each trajectory is one BNG2.pl process on the model with the protocol's steps
    appended, every network-free step given its seed (``phase_seed``) and its own
    output suffix; each step's ``.gdat`` is read back. The last step runs with
    ``get_final_state=>0``: nothing follows it, and BNG2.pl reading NFsim's final
    species back (minutes per trajectory on blbr_dembo1978's chains) is no part of
    the simulation. BNG2.pl writes the XML and
    starts NFsim with its own flags (``-cb`` by default), so the run is what a user
    of BioNetGen gets. On an old-glibc host BNG2.pl is pointed at a mirror of the
    distribution whose ``bin/NFsim`` launches the unchanged binary under the
    loader swap (``env.bng_launch_tree``).
    """

    NAME = "nfsim"

    @staticmethod
    def unsupported(prep: dict[str, Any]) -> str | None:
        return None

    def __init__(self, prep: dict[str, Any], workdir: Path, *, timeout: float):
        self.prep, self.nf, self.names, self.timeout = prep, prep["nf"], prep["names"], timeout
        self.tree = env.bng_launch_tree(workdir / "bngpath")
        self.dir = workdir / "run"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.command: str | None = None
        self.xml_matches: bool | None = None
        self.split = {"nfsim_simulate_cpu_seconds": 0.0, "nfsim_total_cpu_seconds": 0.0,
                      "reactions": 0}

    def trajectory(self, seed: int) -> np.ndarray:
        seeds = [phase_seed(seed, k) for k in range(len(self.nf.phases))]
        bngl = netgen_bngl(self.prep["text"], nfactions.render(self.nf, seeds, suffix="nfp",
                                                               read_back_last=False))
        for old in self.dir.glob("model_nfp*"):
            old.unlink()
        proc = _bng2(bngl, self.dir, timeout=self.timeout, bngpath=self.tree)
        if proc.returncode != 0:
            raise RuntimeError(f"BNG2.pl/NFsim failed: {_tail(proc)}")
        for m in _NF_SOLVE_RE.finditer(proc.stdout):
            self.split["reactions"] += int(m.group(1))
            self.split["nfsim_simulate_cpu_seconds"] += float(m.group(2))
        for m in _NF_TOTAL_RE.finditer(proc.stdout):
            self.split["nfsim_total_cpu_seconds"] += float(m.group(1))
        if self.command is None:
            m = _NF_COMMAND_RE.search(proc.stdout)
            self.command = m.group(1).strip() if m else None
            xml = self.dir / "model_nfp0.xml"
            self.xml_matches = xml.exists() and \
                fetch.sha256_file(xml) == self.prep["info"]["xml_sha256"]
        rows = []
        for p in self.nf.phases:
            gdat = self.dir / f"model_nfp{p.index}.gdat"
            if not gdat.exists():
                raise RuntimeError(f"BNG2.pl/NFsim wrote no .gdat for network-free step "
                                   f"{p.index + 1}: {_tail(proc)}")
            _, values, have = read_dat(gdat)
            if values.shape[0] != p.n_points:
                raise RuntimeError(
                    f"NFsim wrote {values.shape[0]} time points for step {p.index + 1}, expected "
                    f"{p.n_points}" + (" (it stops early once a count turns NaN, e.g. a rate "
                                       "law dividing by a single-copy count: RuleWorld/nfsim#115)"
                                       if values.shape[0] < p.n_points else ""))
            cols = _columns(values, have, self.names, "NFsim")
            if not np.all(np.isfinite(cols)):
                raise RuntimeError("NFsim wrote NaN or infinite counts (a rate law dividing by "
                                   "a single-copy count stops it so: RuleWorld/nfsim#115)")
            rows.append(cols)
        return np.concatenate(rows)

    def ensemble(self, n: int, seed: int) -> np.ndarray:
        self.split = {k: 0 * v for k, v in self.split.items()}
        return np.stack([self.trajectory(int(seed) + i) for i in range(int(n))])

    def config(self) -> dict[str, Any]:
        cmd = self.command
        if cmd:
            # Scratch paths and the per-trajectory seed say nothing about the experiment.
            cmd = re.sub(r"\S*/bin/NFsim", "NFsim", cmd)
            cmd = re.sub(r"\S*model_nfp", "model_nfp", cmd)
            cmd = re.sub(r"-seed \d+", "-seed <seed>", cmd)
        return {"driver": "BNG2.pl, running the protocol's actions",
                "version": env.nfsim_identity().get("version"),
                "nfsim_command": cmd, "xml_matches_harness": self.xml_matches,
                "seeds": "first_seed + i for trajectory i (+ 10**6 per later step)",
                **self.split}


class RuleMonkeyCLI:
    """Standalone RuleMonkey's ``rm_driver``, a fresh process per trajectory, on
    the harness's XML. It reads the gdat ``rm_driver`` prints and the simulation
    time it reports on stderr."""

    NAME = "rulemonkey"

    @staticmethod
    def unsupported(prep: dict[str, Any]) -> str | None:
        nf = prep["nf"]
        if not nf.single:
            return (f"UNSUPPORTED: the protocol has {len(nf.phases)} network-free steps; "
                    "rm_driver cannot change a parameter between runs, so it cannot replay it")
        return None

    def __init__(self, prep: dict[str, Any], *, timeout: float):
        self.exe = env.rulemonkey_bin()
        self.prep, self.nf, self.names, self.timeout = prep, prep["nf"], prep["names"], timeout
        self.split = {"rm_simulate_seconds": 0.0}

    def command(self, seed: int) -> list[str]:
        p = self.nf.phases[0]
        cmd = [str(self.exe), str(self.prep["xml"]), repr(p.duration), str(p.n_steps),
               str(int(seed))]
        if not self.nf.complex:
            cmd.append("-no-bscb")
        return cmd

    def trajectory(self, seed: int) -> np.ndarray:
        proc = subprocess.run(self.command(seed), capture_output=True, text=True,
                              timeout=self.timeout, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"rm_driver failed (exit {proc.returncode}): {_tail(proc)}")
        header: list[str] = []
        rows = []
        for line in proc.stdout.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                header = s.lstrip("#").split()[1:]
                continue
            rows.append([float(x) for x in s.split()])
        values = np.asarray(rows, dtype=float)
        if values.ndim != 2 or values.shape[0] != self.nf.phases[0].n_points:
            raise RuntimeError(f"rm_driver printed {len(rows)} time points, expected "
                               f"{self.nf.phases[0].n_points}")
        try:
            self.split["rm_simulate_seconds"] += float(proc.stderr.strip().splitlines()[-1])
        except (ValueError, IndexError):
            pass
        return _columns(values[:, 1:], header, self.names, "rm_driver")

    def ensemble(self, n: int, seed: int) -> np.ndarray:
        self.split = {"rm_simulate_seconds": 0.0}
        return np.stack([self.trajectory(int(seed) + i) for i in range(int(n))])

    def config(self) -> dict[str, Any]:
        ident = env.rulemonkey_identity()
        cmd = self.command(0)
        cmd = ["rm_driver", "model.xml", *cmd[2:4], "<seed>", *cmd[5:]]
        return {"version": ident.get("version"), "commit": ident.get("commit"),
                "compiler": ident.get("compiler"), "command": cmd,
                "lineage": RULEMONKEY_LINEAGE, **self.split}


def _single_step_ssa(prep: dict[str, Any]) -> str | None:
    if prep.get("net") is None:
        return "UNSUPPORTED: no generated network (the spec names no netgen)"
    nf = prep["nf"]
    if not nf.single:
        return (f"UNSUPPORTED: exact SSA on the generated network is run from the seed "
                f"species only; this protocol has {len(nf.phases)} network-free steps")
    return None


# BNG2.pl's answer to run_network's ``edgepop: <species>`` (as in the ssa
# capability). run_network's stochastic methods write that line when a species no
# reaction consumes (the network's edge, as on-the-fly generation sees it) first
# becomes populated, and then read their stdin for BNG2.pl's reply: ``read`` and new
# network blocks, or ``continue``. BNG2.pl, running a simulation from a finished
# network, replies ``continue`` (BNGAction.pm, simulate). A run_network given no
# reply reads end of file and crashes (signal 11; Rohrs 2018's fully phosphorylated
# CD3z). So every trajectory gets BNG2.pl's reply on stdin once per species, more
# than it can ask for.
EDGEPOP_REPLY = "continue\n"


class RunNetworkSSA:
    """BioNetGen's ``run_network -p ssa`` on the generated network, a fresh
    process per trajectory, read back from its observables (``.gdat``). Each
    process is driven as BNG2.pl drives it, its ``edgepop`` questions answered on
    stdin (``EDGEPOP_REPLY``), in the job's scratch directory (a crash's core file
    stays there)."""

    NAME = "run_network"
    unsupported = staticmethod(_single_step_ssa)

    def __init__(self, prep: dict[str, Any], out_prefix: Path, *, timeout: float):
        self.nf, self.names, self.timeout = prep["nf"], prep["names"], timeout
        # Absolute: each trajectory runs in the scratch directory, not the job's.
        self.net = Path(prep["net"]).absolute() if prep.get("net") is not None else None
        self.out_prefix = Path(out_prefix).absolute()
        self.out_prefix.parent.mkdir(parents=True, exist_ok=True)
        self._replies: str | None = None

    def replies(self) -> str:
        """What every trajectory reads on stdin: BNG2.pl's ``EDGEPOP_REPLY`` once per
        species and once more, since each species can reach the edge only once."""
        if self._replies is None:
            try:
                n = network_size(self.net).get("species", 0)
            except OSError:
                n = 0  # no network to read: run_network fails on it first
            self._replies = EDGEPOP_REPLY * (n + 1)
        return self._replies

    def command(self, seed: int) -> list[str]:
        p = self.nf.phases[0]
        return [*env.run_network_launch(), str(env.run_network_bin()), "-o", str(self.out_prefix),
                "-p", "ssa", "-h", str(int(seed)), "--cdat", "0", "--fdat", "0",
                "-g", str(self.net), str(self.net), repr(p.duration / p.n_steps), str(p.n_steps)]

    def trajectory(self, seed: int) -> np.ndarray:
        proc = subprocess.run(self.command(seed), input=self.replies(), capture_output=True,
                              text=True, timeout=self.timeout, check=False,
                              cwd=self.out_prefix.parent)
        if proc.returncode != 0:
            raise RuntimeError(f"run_network failed: {_tail(proc)}")
        gdat = Path(f"{self.out_prefix}.gdat")
        if not gdat.exists():
            raise RuntimeError("run_network produced no .gdat")
        _, values, have = read_dat(gdat)
        gdat.unlink()
        if values.shape[0] != self.nf.phases[0].n_points:
            raise RuntimeError(f"run_network wrote {values.shape[0]} time points, expected "
                               f"{self.nf.phases[0].n_points}")
        return _columns(values, have, self.names, "run_network")

    def ensemble(self, n: int, seed: int) -> np.ndarray:
        return np.stack([self.trajectory(int(seed) + i) for i in range(int(n))])

    def config(self) -> dict[str, Any]:
        cmd = recorded_command(self.command(0))
        names = {str(self.net): "model.net", str(self.out_prefix): "<out>"}
        cmd = [names.get(c, c) for c in cmd]
        cmd[cmd.index("-h") + 1] = "<seed>"
        return {"method": "ssa (direct method), on the generated network", "command": cmd}


class BngsimSSA:
    """BNGsim's own exact SSA on the generated network (``run_replicates``),
    compared on its observables: BNGsim's code, but another algorithm on another
    representation of the same model."""

    NAME = "bngsim_ssa"
    unsupported = staticmethod(_single_step_ssa)

    def __init__(self, prep: dict[str, Any]):
        import bngsim

        self.nf, self.names = prep["nf"], prep["names"]
        t0 = time.perf_counter()
        self.model = bngsim.Model.from_net(str(prep["net"]))
        self.sim = bngsim.Simulator(self.model, method="ssa")
        self.load_seconds = time.perf_counter() - t0
        self.backend: str | None = None

    def ensemble(self, n: int, seed: int) -> np.ndarray:
        p = self.nf.phases[0]
        self.model.reset()
        r = self.sim.run_replicates(int(n), seed=int(seed), squeeze=True,
                                    t_span=(0.0, p.duration), n_points=p.n_points)
        self.backend = (getattr(r, "ssa_diagnostics", None) or {}).get("propensity_backend")
        obs = np.asarray(r.observables, dtype=float)
        have = list(r.observable_names)
        return np.stack([_columns(o, have, self.names, "bngsim ssa") for o in obs])

    def config(self) -> dict[str, Any]:
        return {"method": "ssa, on the generated network", "propensity_backend": self.backend,
                "load_seconds": self.load_seconds}


# The references BNGsim is compared against, in a fixed order; ``references``
# says which apply to a problem.
REFERENCE_ENGINES: dict[str, type] = {
    NfsimBNG2.NAME: NfsimBNG2,
    RuleMonkeyCLI.NAME: RuleMonkeyCLI,
    RunNetworkSSA.NAME: RunNetworkSSA,
    BngsimSSA.NAME: BngsimSSA,
}

BUILD_STAGE = {
    "bngsim": "session (XML read)",
    "nfsim": "BNG2.pl + NFsim trajectories (subprocesses)",
    "rulemonkey": "rm_driver trajectories (subprocesses)",
    "run_network": "run_network trajectories (subprocesses)",
    "bngsim_ssa": "load + propensity compile",
}
ROLES = {"nfsim": "fresh subprocess", "rulemonkey": "fresh subprocess",
         "run_network": "fresh subprocess"}


def _build_reference(name: str, prep: dict[str, Any], workdir: Path, *, timeout: float):
    if name == "nfsim":
        return NfsimBNG2(prep, workdir / "nfsim", timeout=timeout)
    if name == "rulemonkey":
        return RuleMonkeyCLI(prep, timeout=timeout)
    if name == "run_network":
        return RunNetworkSSA(prep, workdir / "run_network" / "rn", timeout=timeout)
    if name == "bngsim_ssa":
        return BngsimSSA(prep)
    raise ValueError(f"unknown reference {name!r}")


def _unsupported(name: str, prep: dict[str, Any]) -> str | None:
    return REFERENCE_ENGINES[name].unsupported(prep)


def _progress(msg: str) -> None:
    progress(CAPABILITY, msg)


def _role(name: str) -> str:
    return ROLES.get(name) or engine_role(name)


# --------------------------------------------------------------------------- #
# Ensembles and the verdict
# --------------------------------------------------------------------------- #
def _cold_bngsim(setting: str, prep: dict[str, Any], n: int, seed: int, **kw):
    def fn(stage):
        t0 = time.perf_counter()
        stage(BUILD_STAGE["bngsim"])
        engine = BngsimNF(setting, prep["xml"], prep["nf"], prep["names"], **kw)
        stage(f"ensemble of {n} ({setting})")
        t1 = time.perf_counter()
        stack = engine.ensemble(n, seed)
        return stack, {**engine.config(), "ensemble_seconds": time.perf_counter() - t1,
                       "cold_seconds": time.perf_counter() - t0, "first_seed": seed}
    return fn


def _cold_reference(name: str, prep: dict[str, Any], n: int, seed: int, workdir: Path,
                    timeout: float):
    def fn(stage):
        t0 = time.perf_counter()
        stage(BUILD_STAGE.get(name, "build"))
        ref = _build_reference(name, prep, workdir, timeout=timeout)
        stage(f"ensemble of {n}")
        t1 = time.perf_counter()
        stack = ref.ensemble(n, seed)
        return stack, {**ref.config(), "ensemble_seconds": time.perf_counter() - t1,
                       "cold_seconds": time.perf_counter() - t0, "first_seed": seed}
    return fn


def _permutation_seed(seed: int, k: int) -> int:
    return int(np.random.SeedSequence([int(seed), int(k), 7]).generate_state(1)[0])


def two_sample(bn: np.ndarray, ref: np.ndarray, name: str, seed: int, grid: np.ndarray,
               names: list[str]) -> dict[str, Any]:
    """BNGsim's ensemble against one other: the SSA capability's permutation max-T
    verdict, with bngsim's parity_core verdict recorded beside it."""
    v = stochastic.two_sample_verdict(
        bn, ref, alpha=VERDICT["alpha"], permutations=VERDICT["permutations"],
        stop_exceedances=VERDICT["stop_exceedances"],
        seed=_permutation_seed(seed, COMPARISON_INDEX[name]), times=grid, species=names)
    if "reason" not in v:
        v["parity_core"] = stochastic.parity_core_verdict(bn, ref)
    return v


def _final_moments(stack: np.ndarray, names: list[str]) -> dict[str, Any]:
    last = stack[:, -1, :]
    return {"replicates": int(stack.shape[0]),
            "final_mean": dict(zip(names, map(float, last.mean(axis=0)))),
            "final_variance": dict(zip(names, map(float, last.var(axis=0, ddof=1))))}


def _same_seed(bn: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
    k = min(len(bn), len(ref))
    diffs = [float(np.max(np.abs(bn[i] - ref[i]))) for i in range(k)]
    return {"trajectories": k, "identical": sum(d == 0.0 for d in diffs),
            "max_abs_difference": max(diffs) if diffs else None}


def independent(applicable: list[str]) -> list[str]:
    """The references in ``applicable`` that are not of BNGsim's own network-free
    code lineage (``SAME_LINEAGE_REFERENCES``)."""
    return [name for name in applicable if name not in SAME_LINEAGE_REFERENCES]


def verdict_outcome(bn_error: str | None, ref_errors: dict[str, str], applicable: list[str],
                    verdicts: dict[str, Any]) -> str:
    """The outcome, with the same-lineage rule on top of ``_common.outcome``.

    A BNGsim failure is EXCEPTION (TIMEOUT when it ran out of time), BAD_TEST when
    every reference failed too. An independent reference that ran and disagrees
    makes it DIFF, even when another reference failed to run (``OUTCOME_RULE``: a
    disagreement is never masked). Otherwise any reference that failed to run makes
    it REFERENCE_FAILED. With every reference compared: PASS needs BNGsim to agree
    with all of them *and* at least one independent reference to apply. A
    same-lineage reference alone (``rm_driver``, the upstream of the embedded
    ``nf_exact``) decides neither: with no independent reference, or when it is the
    only one to disagree, the outcome is REFERENCE_FAILED.
    """
    if bn_error:
        result = outcome(bn_error, ref_errors, len(applicable), False)
        return "TIMEOUT" if bn_error.startswith("TIMEOUT") and result == "EXCEPTION" else result
    judges = [name for name in independent(applicable) if name in verdicts]
    if any(not verdicts[name].get("passed") for name in judges):
        return "DIFF"
    if ref_errors or not verdicts or not judges:
        return "REFERENCE_FAILED"
    if all(verdicts[name].get("passed") for name in applicable if name in verdicts):
        return "PASS"
    return "REFERENCE_FAILED"


def verdict_basis(applicable: list[str], verdicts: dict[str, Any], result: str) -> dict[str, Any]:
    """Which references could decide the outcome, recorded with it: the
    independent ones, the same-lineage ones, and, where it happened, the
    disagreement that only a same-lineage reference found."""
    basis: dict[str, Any] = {"independent": independent(applicable),
                             "same_lineage": [n for n in applicable
                                              if n in SAME_LINEAGE_REFERENCES]}
    alone = [n for n in basis["same_lineage"]
             if n in verdicts and not verdicts[n].get("passed")]
    if alone and result == "REFERENCE_FAILED" and all(
            verdicts[n].get("passed") for n in basis["independent"] if n in verdicts):
        basis["same_lineage_disagreement"] = (
            f"only {', '.join(alone)} disagrees with BNGsim; a reference of BNGsim's own "
            "code lineage cannot refute it alone, so the disagreement is the reference's")
    if not basis["independent"]:
        basis["no_independent_reference"] = ("no independent reference applies, so BNGsim "
                                              "can neither pass nor be refuted")
    return basis


def _score(bn: np.ndarray | None, ensembles: dict[str, np.ndarray], seed: int,
           grid: np.ndarray, names: list[str], bn_error: str | None,
           ref_errors: dict[str, str], applicable: list[str]
           ) -> tuple[dict[str, Any], str, dict[str, Any]]:
    verdicts: dict[str, Any] = {}
    if bn is not None:
        for name, stack in ensembles.items():
            _progress(f"verdict vs {name}")
            verdicts[name] = two_sample(bn, stack, name, seed, grid, names)
            if name in SAME_LINEAGE_REFERENCES:
                verdicts[name]["same_lineage"] = True
    result = verdict_outcome(bn_error, ref_errors, applicable, verdicts)
    return verdicts, result, verdict_basis(applicable, verdicts, result)


def v1143_reference() -> bool:
    """Whether the standalone NFsim is a 1.14 release (BioNetGen 2.9.3's, for a
    comparison run), which BNGsim's ``v1143_compat`` mode reproduces seed for seed."""
    return str(env.nfsim_identity().get("version") or "").startswith("1.14.")


def other_setting(setting: str) -> str:
    return SETTINGS[1 - SETTINGS.index(setting)]


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
def run_correctness(job: Job, problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    setting = job.setting
    try:
        prep = _prepare(problem, ctx)
    except Exception as exc:  # noqa: BLE001 - a model or protocol that cannot be run
        return {"outcome": "BAD_TEST", "error": f"{type(exc).__name__}: {exc}"}
    proto, info, nf, names = prep["proto"], prep["info"], prep["nf"], prep["names"]
    n, seed = int(proto["replicates"]), int(proto["seed"])
    grid = times(nf)
    workdir = Path(ctx["workdir"])
    budget = float(ctx.get("engine_budget_seconds") or engine_budget_seconds(problem))
    info["engine_budget_seconds"] = budget
    timeout = min(float(ctx["subprocess_timeout"]), budget)

    def isolated(name: str, fn) -> dict[str, Any]:
        _progress(f"{name}: start (budget {budget:g} s; {threads()} threads in the worker)")
        out = run_isolated(fn, budget, workdir / "isolated" / name)
        _progress(f"{name}: {out.get('error') or 'done'} after {out['seconds']:.1f} s")
        return out

    bn, bn_error = None, None
    out = isolated(setting, _cold_bngsim(setting, prep, n, seed_for(setting, seed)))
    if "error" in out:
        bn_error = out["error"]
    else:
        info["bngsim"] = out["config"]
        bn = out["trajectory"]

    applicable = references(problem)
    references_: dict[str, Any] = {}
    ensembles: dict[str, np.ndarray] = {}
    ref_errors: dict[str, str] = {}
    for name in applicable:
        reason = _unsupported(name, prep)
        if reason:
            ref_errors[name] = reason
            _progress(f"{name}: not run, {reason}")
            continue
        out = isolated(name, _cold_reference(name, prep, n, seed_for(name, seed), workdir, timeout))
        if "error" in out:
            ref_errors[name] = out["error"]
        else:
            ensembles[name] = out["trajectory"]
            references_[name] = out["config"]
    info["references"] = references_
    if "rulemonkey" in references_ or "rulemonkey" in ref_errors:
        info["rulemonkey_lineage"] = RULEMONKEY_LINEAGE

    # BNGsim's other network-free engine, from its own seeds: not scored.
    other = other_setting(setting)
    cross = isolated(other, _cold_bngsim(other, prep, n, seed_for(other, seed)))
    cross_stack = None
    if "error" in cross:
        info["cross_setting"] = {"setting": other, "error": cross["error"]}
    else:
        cross_stack = cross["trajectory"]
        info["cross_setting"] = {"setting": other, "config": cross["config"]}

    # The embedding, same seed for same seed: bngsim nf_exact against rm_driver;
    # bngsim nf_reject against BNG2.pl's NFsim, in its v1.14.3 compatibility mode
    # where that NFsim is 1.14.3.
    twin = "rulemonkey" if setting == "nf_exact" else "nfsim"
    if twin in ensembles and nf.single:
        k = min(SAME_SEED_TRAJECTORIES, n)
        compat = setting == "nf_reject" and v1143_reference()
        out = isolated(f"{setting} (same seeds as {twin})",
                       _cold_bngsim(setting, prep, k, seed_for(twin, seed),
                                    v1143_compat=compat))
        if "error" in out:
            info["same_seed"] = {"engine": twin, "error": out["error"]}
        else:
            info["same_seed"] = {"engine": twin, **_same_seed(out["trajectory"],
                                                              ensembles[twin][:k])}
        if setting == "nf_reject":
            info["same_seed"]["v1143_compat"] = compat

    verdicts, result, basis = _score(bn, ensembles, seed, grid, names, bn_error, ref_errors,
                                     applicable)
    if bn is not None and cross_stack is not None:
        info["cross_setting"]["verdict"] = two_sample(bn, cross_stack, "cross_setting", seed,
                                                      grid, names)
    info["ensembles"] = {name: _final_moments(stack, names)
                         for name, stack in (("bngsim", bn), (other, cross_stack),
                                             *ensembles.items())
                         if stack is not None}

    rec: dict[str, Any] = {"outcome": result, "measurements": info, "verdict_basis": basis}
    if verdicts:
        rec["verdicts"] = verdicts
    if bn_error:
        rec["bngsim_error"] = bn_error
    for name, err in ref_errors.items():
        rec[f"{name}_error"] = err
    arrays = {name: stack for name, stack in (("bngsim", bn), *ensembles.items())
              if stack is not None}
    if cross_stack is not None:
        arrays["cross_setting"] = cross_stack
    if arrays:
        arrays["observables"] = np.asarray(names)
        arrays["times"] = grid
        rec["_trajectories"] = arrays
    return rec


def rescore(job: Job, record: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """A correctness record re-scored from its stored ensembles with the current
    verdict: new ``verdicts``, ``outcome`` and cross-setting verdict. Engine
    failures and configurations are the record's own."""
    proto = job.definition["protocol"]
    seed = int(proto.get("seed", DEFAULT_SEED))
    names = [str(s) for s in arrays["observables"]]
    grid = np.asarray(arrays["times"], dtype=float)
    applicable = job.engines.removeprefix("bngsim-vs-").split("-")
    refs = {name: arrays[name] for name in applicable if name in arrays}
    ref_errors = {name: str(record[f"{name}_error"]) for name in applicable
                  if record.get(f"{name}_error")}
    verdicts, result, basis = _score(arrays.get("bngsim"), refs, seed, grid, names,
                                     record.get("bngsim_error"), ref_errors, applicable)
    update: dict[str, Any] = {"outcome": result, "verdict_basis": basis}
    if verdicts:
        update["verdicts"] = verdicts
    m = dict(record.get("measurements") or {})
    if "cross_setting" in arrays and "bngsim" in arrays and m.get("cross_setting"):
        m["cross_setting"] = {**m["cross_setting"], "verdict": two_sample(
            arrays["bngsim"], arrays["cross_setting"], "cross_setting", seed, grid, names)}
        update["measurements"] = m
    return update


def _summary(stack: np.ndarray) -> np.ndarray:
    stack = np.asarray(stack, dtype=float)
    return np.concatenate([stack.mean(axis=0), (stack**2).mean(axis=0)], axis=1)


TIMED_REFERENCES = ("nfsim", "rulemonkey")
# Each timed engine's own measure of its solve, apart from its start-up: BNGsim's
# session simulate calls (wall), NFsim's reported simulation CPU time, rm_driver's
# reported simulation time (wall, around its simulate).
SOLVE_KEY = {"bngsim": "simulate_seconds", "nfsim": "nfsim_simulate_cpu_seconds",
             "rulemonkey": "rm_simulate_seconds"}


def run_timing(job: Job, problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    setting = job.setting
    try:
        prep = _prepare(problem, ctx)
    except Exception as exc:  # noqa: BLE001
        return {"outcome": "BAD_TEST", "error": f"{type(exc).__name__}: {exc}"}
    proto, info = prep["proto"], prep["info"]
    workdir = Path(ctx["workdir"])
    timeout = ctx["subprocess_timeout"]
    k, seed = int(proto["timing_replicates"]), int(proto["seed"])
    info["timing_replicates"] = k

    # BNGsim cold: the session built (its XML read) and a first ensemble, which
    # anchors the warm ones.
    t0 = time.perf_counter()
    bngsim_engine = BngsimNF(setting, prep["xml"], prep["nf"], prep["names"])
    first = bngsim_engine.ensemble(k, seed_for(setting, seed))
    info["bngsim_cold_seconds"] = time.perf_counter() - t0
    info["bngsim"] = bngsim_engine.config()
    anchor = _summary(first)

    references_: dict[str, Any] = {}
    engines: dict[str, Any] = {"bngsim": bngsim_engine}
    ref_errors: dict[str, str] = {}
    timed_out = correctness_timeouts(job, ctx, TIMED_REFERENCES)
    for name in TIMED_REFERENCES:
        reason = _unsupported(name, prep)
        if reason:
            ref_errors[name] = reason
            continue
        if name in timed_out:
            ref_errors[name] = f"TIMEOUT in the correctness job, so not timed ({timed_out[name]})"
            _progress(f"{name}: skipped, {ref_errors[name]}")
            continue
        _progress(f"{name}: build + trial run")
        try:
            engine = _build_reference(name, prep, workdir, timeout=timeout)
            engine.ensemble(1, seed_for(name, seed))
            engines[name] = engine
        except Exception as exc:  # noqa: BLE001
            ref_errors[name] = f"{type(exc).__name__}: {exc}"[:400]

    _progress(f"interleaved timing of {', '.join(engines)}, {k} trajectories per ensemble")
    runs = {name: (lambda e=eng, s=seed_for(setting if name == "bngsim" else name, seed):
                   e.ensemble(k, s))
            for name, eng in engines.items()}
    captures = {name: (lambda v, e=eng: (_summary(v), dict(e.split)))
                for name, eng in engines.items()}
    reps = round_robin_interleaved(runs, captures=captures, **job.definition["timing"])

    outputs = {name: reps["engines"][name].pop("outputs") for name in runs}
    summaries = {name: [o[0] for o in outs] for name, outs in outputs.items()}
    splits = {name: [o[1] for o in outs] for name, outs in outputs.items()}
    repro: dict[str, Any] = {"bngsim": reproducibility(anchor, summaries["bngsim"], REPRO_TOL)}
    for name in engines:
        if name != "bngsim":
            repro[name] = reproducibility(summaries[name][0], summaries[name][1:], REPRO_TOL)
    reproducible = all(r["passed"] for r in repro.values())
    for name, eng in engines.items():
        if name != "bngsim":
            references_[name] = eng.config()
    info["references"] = references_

    times_ = absolute_time_stats({name: reps["engines"][name]["seconds"] for name in runs})
    per_trajectory = {name: {key: t[key] / k for key in ("median_seconds", "ci_low", "ci_high")}
                      for name, t in times_.items()}
    # Each engine's own account of where an ensemble's time went, median over the
    # kept rounds (an ensemble's sum over its trajectories).
    solve_split = {}
    for name, rows in splits.items():
        if rows and rows[0]:
            med = {key: float(np.median([s[key] for s in rows])) for key in rows[0]}
            if SOLVE_KEY.get(name) in med:
                med["solve_seconds"] = med[SOLVE_KEY[name]]
            solve_split[name] = med
    rec: dict[str, Any] = {
        "outcome": "PASS" if reproducible else "NOT_REPRODUCIBLE",
        "measurements": {
            **info,
            "engine_roles": {name: _role(name) for name in engines},
            "repetitions": reps,
            "reproducibility": repro,
            "times": times_,
            "times_per_trajectory": per_trajectory,
            "solve_split": solve_split,
        },
    }
    for name, err in ref_errors.items():
        rec[f"{name}_error"] = err
    return rec


def run(job: Job, problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    if job.kind == CORRECTNESS:
        return run_correctness(job, problem, ctx)
    if job.kind == TIMING:
        return run_timing(job, problem, ctx)
    raise ValueError(f"unknown job kind {job.kind!r}")
