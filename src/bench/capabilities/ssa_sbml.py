"""Exact SSA on SBML models (the ssa-sbml suite): the SBML side of ``bench.capabilities.ssa``.

An SBML model is read as it is, with no network generation and no export:
BNGsim reads the pinned SBML with ``Model.from_sbml``, libRoadRunner and COPASI
read the same file, and run_network reads a network derived from it where that
derivation is faithful for the jump process (below). Every engine samples the
same protocol as on the ``.net`` path (``ssa.protocol``): ``replicates``
trajectories from seeds in its own block, on the grid ``0..t_end``.

**Molecule counts.** An amount is a number of molecules: an SBML species whose
amount carries no substance unit, in a compartment of size 1 (or of no stated
size), is read as a count by BNGsim and libRoadRunner; COPASI imports such an
amount in moles unless told otherwise, so its quantity unit is set to ``#`` (one
particle per unit amount; ``ssa.CopasiSSA``), recorded in its configuration
with the unit it was imported in. A compartment with a size other than 1 makes
amount and concentration differ; such a model is refused (``BAD_TEST``).

**What is compared** is every *changing* species, by SBML id: one that a
reaction changes (a non-boundary species whose net stoichiometry in some
reaction is not zero), a rate rule integrates, or an event assigns. The census
(``sbml_census``) records the others -- constant, set by an assignment rule, or
never changed -- since not every engine reports a species that never moves
(COPASI leaves them out of its stochastic time series).

**Time-triggered events.** At an output time that is exactly an instant a
time-triggered event fires, whether the row is before or after the event is a
convention the engines need not share; such rows are left out of every
comparison and named (``rows_not_compared``). libRoadRunner's gillespie
integrator fires a time-triggered event only at its first reaction event after
the trigger time, and says so in its log; it is run, and the warning it gave is
recorded in its configuration.

**run_network** reads only ``.net``. The ODE suite's derivation
(``ode.derive_network``: ``bngsim.convert.sbml_to_bngl`` -> BNG2.pl, checked
against BNGsim on the SBML) preserves the ODE right-hand side, which is not
enough for a jump process: the converter writes a law that is not mass action
as one flux reaction per species it changes (``0 -> P`` at ``-2*f``), the same
ODE but other jumps. So for SSA the derived network must also have the SBML's
jumps: every derived reaction's net change of the compared species must be an
SBML reaction's, with the same multiplicity, and no derived reaction may consume
a species twice (its ODE rate does not fix its SSA propensity). A derivation
that changes the state between phases is run only when every value it sets is a
whole number of molecules (run_network's SSA rounds a species value to an
integer), and this driver runs a single phase. Otherwise run_network is not
applicable to the model, with the reason.

**Exact moments** for an SBML model are a pinned table (``moments: {kind:
table}``): the DSMTS files of means and standard deviations, fetched and
sha256-checked like a model file, at the protocol's output times.
"""

from __future__ import annotations

import csv
import math
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .. import env, fetch, netfile
from ..specs import Problem
from . import ode
from .ssa import BngsimSSA, CopasiSSA, RoadRunnerSSA, RunNetworkSSA, _grid, times

# Bump when the SBML path's computation changes (in every SBML ssa job's
# definition, so a .net job's key is unchanged).
SBML_HARNESS_VERSION = "1"
# Two output times closer than this fraction of the horizon are the same instant.
TIME_TOL = 1e-9


# --------------------------------------------------------------------------- #
# The SBML: what is compared, and when events fire
# --------------------------------------------------------------------------- #
def _stoich(ref) -> float:
    return float(ref.getStoichiometry()) if ref.isSetStoichiometry() else 1.0


def _number_value(model, node) -> float | None:
    """The value of a constant AST node: a number, or a constant parameter."""
    import libsbml

    if node is None:
        return None
    if node.isNumber():
        return float(node.getValue())
    if node.getType() == libsbml.AST_NAME:
        p = model.getParameter(node.getName())
        if p is not None and p.getConstant() and p.isSetValue():
            return float(p.getValue())
    if node.getType() == libsbml.AST_MINUS and node.getNumChildren() == 1:
        v = _number_value(model, node.getChild(0))
        return None if v is None else -v
    return None


def _trigger_time(model, math) -> float | None:
    """The instant a trigger ``time > c`` (or ``>=``, ``<``, ``<=``, either way
    round, with a constant ``c``) first holds; None for any other trigger."""
    import libsbml

    rel = {libsbml.AST_RELATIONAL_GT, libsbml.AST_RELATIONAL_GEQ,
           libsbml.AST_RELATIONAL_LT, libsbml.AST_RELATIONAL_LEQ}
    if math is None or math.getType() not in rel or math.getNumChildren() != 2:
        return None
    a, b = math.getChild(0), math.getChild(1)
    if a.getType() == libsbml.AST_NAME_TIME:
        return _number_value(model, b)
    if b.getType() == libsbml.AST_NAME_TIME:
        return _number_value(model, a)
    return None


def _mentions_time(math) -> bool:
    import libsbml

    if math is None:
        return False
    if math.getType() == libsbml.AST_NAME_TIME:
        return True
    return any(_mentions_time(math.getChild(i)) for i in range(math.getNumChildren()))


def sbml_census(sbml: Path) -> tuple[list[str], dict[str, Any]]:
    """The SBML's changing species, in document order, and a census of the file:
    what is not compared and why, its events (``time_events``: the instant each
    time-triggered one fires), and its compartments. Raises on a compartment
    whose size is set to anything but 1, where an amount is not a concentration."""
    import libsbml

    doc = libsbml.readSBMLFromFile(str(sbml))
    model = doc.getModel()
    if model is None:
        errors = [doc.getError(i).getMessage().strip() for i in range(doc.getNumErrors())]
        raise RuntimeError(f"libsbml cannot read {sbml.name}: {'; '.join(errors)[:300]}")
    for c in model.getListOfCompartments():
        if c.isSetSize() and not math.isclose(c.getSize(), 1.0, rel_tol=1e-12):
            raise RuntimeError(
                f"compartment {c.getId()} has size {c.getSize():g}: an amount is a molecule "
                "count and a concentration only in a compartment of size 1")
    assigned = {r.getVariable() for r in model.getListOfRules() if r.isAssignment()}
    rate_ruled = {r.getVariable() for r in model.getListOfRules() if r.isRate()}
    event_set = {a.getVariable() for e in model.getListOfEvents()
                 for a in e.getListOfEventAssignments()}
    moved: set[str] = set()
    for r in model.getListOfReactions():
        net: Counter = Counter()
        for ref in r.getListOfReactants():
            net[ref.getSpecies()] -= _stoich(ref)
        for ref in r.getListOfProducts():
            net[ref.getSpecies()] += _stoich(ref)
        moved.update(s for s, v in net.items() if v != 0)
    changing, constant, by_rule, unchanged = [], [], [], []
    for sp in model.getListOfSpecies():
        sid = sp.getId()
        if sp.getConstant():
            constant.append(sid)
        elif sid in assigned:
            by_rule.append(sid)
        elif sid in rate_ruled or sid in event_set or (sid in moved and not sp.getBoundaryCondition()):
            changing.append(sid)
        else:
            unchanged.append(sid)
    if not changing:
        raise RuntimeError(f"{sbml.name} has no changing species to compare")
    time_events, other_time = [], []
    for e in model.getListOfEvents():
        trig = e.getTrigger().getMath() if e.isSetTrigger() else None
        at = _trigger_time(model, trig)
        delay = 0.0
        if e.isSetDelay():
            delay = _number_value(model, e.getDelay().getMath())
        if at is not None and delay is not None:
            time_events.append({"id": e.getId(), "time": at + delay,
                                "trigger": libsbml.formulaToL3String(trig)})
        elif _mentions_time(trig) or e.isSetDelay():
            other_time.append(e.getId())
    census = {
        "species": model.getNumSpecies(),
        "reactions": model.getNumReactions(),
        "events": model.getNumEvents(),
        "compared_species": len(changing),
        "not_compared": {"constant": constant, "assignment_rule": by_rule,
                         "unchanged": unchanged},
        "time_events": time_events,
        "compartments": {c.getId(): (c.getSize() if c.isSetSize() else None)
                         for c in model.getListOfCompartments()},
    }
    if other_time:
        census["other_time_events"] = other_time
    return changing, census


def excluded_rows(grid: np.ndarray, census: dict[str, Any]) -> list[float]:
    """The output times at which a time-triggered event fires."""
    scale = max(1.0, float(abs(grid[-1])))
    out = []
    for ev in census.get("time_events", []):
        hit = [float(g) for g in grid if abs(float(g) - ev["time"]) <= TIME_TOL * scale]
        out.extend(h for h in hit if h not in out)
    return sorted(out)


def sbml_jumps(sbml: Path, species: list[str]) -> Counter:
    """The SBML's reactions as a multiset of net changes of the compared species."""
    import libsbml

    model = libsbml.readSBMLFromFile(str(sbml)).getModel()
    keep = set(species)
    out: Counter = Counter()
    for r in model.getListOfReactions():
        net: Counter = Counter()
        for ref in r.getListOfReactants():
            net[ref.getSpecies()] -= _stoich(ref)
        for ref in r.getListOfProducts():
            net[ref.getSpecies()] += _stoich(ref)
        vec = tuple(sorted((s, float(v)) for s, v in net.items() if v and s in keep))
        if vec:
            out[vec] += 1
    return out


def net_jumps(net: netfile.Network, columns: list[int], species: list[str]) -> Counter:
    """A derived network's reactions as a multiset of net changes of the compared
    species (``columns[i]`` is species ``i``'s index in the network)."""
    name = dict(zip(columns, species))
    out: Counter = Counter()
    for r in net.reactions:
        c: Counter = Counter()
        for i in r.reactants:
            if i in name and not net.fixed[i]:
                c[name[i]] -= 1
        for i in r.products:
            if i in name and not net.fixed[i]:
                c[name[i]] += 1
        vec = tuple(sorted((s, float(v)) for s, v in c.items() if v))
        if vec:
            out[vec] += 1
    return out


def _jump_text(vec: tuple) -> str:
    return " ".join(f"{'+' if v > 0 else ''}{v:g} {s}" for s, v in vec)


def derive_network(sbml: Path, species: list[str], proto: dict[str, Any], workdir: Path, *,
                   timeout: float, stage=lambda text: None) -> dict[str, Any]:
    """The network run_network samples for an SBML model, or why there is none:
    ``ode.derive_network``, then the jump-process checks of the module docstring.
    Model preparation, charged to no engine."""
    t_end, n_steps = _grid(proto)
    oproto = {"t_start": 0.0, "output_start": 0.0, "t_end": t_end, "n_steps": n_steps,
              "rtol": ode.FAITHFULNESS_RTOL, "atol": ode.FAITHFULNESS_ATOL}
    derived = ode.derive_network(sbml, species, oproto, workdir, timeout=timeout, stage=stage)
    if not derived.get("applicable"):
        return derived
    info = derived.setdefault("info", {})

    def not_applicable(reason: str) -> dict[str, Any]:
        return {"applicable": False, "reason": reason[:400], "info": info}

    stage("jump-process check of the derived network")
    net = netfile.read(Path(derived["net"]))
    want, got = sbml_jumps(sbml, species), net_jumps(net, derived["columns"], species)
    extra, lack = got - want, want - got
    info["jumps"] = {"sbml_reactions": sum(want.values()), "derived_reactions": sum(got.values()),
                     "derived_only": len(extra), "sbml_only": len(lack)}
    if extra or lack:
        first = next(iter(extra or lack))
        side = "the derived network has" if extra else "the SBML has"
        return not_applicable(
            f"the derived network does not have the SBML's jumps (the same ODE, another "
            f"jump process): {sum(extra.values())} derived and {sum(lack.values())} SBML "
            f"reaction(s) have no counterpart; e.g. {side} a reaction changing "
            f"{_jump_text(first)} that the other lacks")
    reps = net.repeated_reactants()
    if reps:
        return not_applicable(
            f"{len(reps)} derived reaction(s) consume one species twice (first: reaction "
            f"{reps[0].index}); the ODE rate the derivation preserves does not fix the SSA "
            "propensity of such a reaction")
    for ph in derived["phases"]:
        for (_, value), target in zip(ph["changes"], ph.get("change_targets", [])):
            if float(value) != round(float(value)):
                return not_applicable(
                    f"the derived protocol sets {target} to {float(value):g} between phases, "
                    "and run_network's SSA rounds every species value to a whole number of "
                    "molecules (so it would not hold that value)")
    if len(derived["phases"]) > 1:
        return not_applicable(
            f"the derived protocol has {len(derived['phases'])} phases; this run_network "
            "SSA driver runs one")
    return derived


# --------------------------------------------------------------------------- #
# The engines on the SBML
# --------------------------------------------------------------------------- #
class BngsimSSASBML(BngsimSSA):
    """BNGsim's exact SSA reading the pinned SBML (``Model.from_sbml``): columns by
    SBML id from each result's species names, never by position."""

    def __init__(self, sbml: Path, species: list[str], proto: dict[str, Any]):
        import bngsim

        self.proto = proto
        self.species = list(species)
        t0 = time.perf_counter()
        self.model = bngsim.Model.from_sbml(str(sbml))
        self.load_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        self.sim = self._simulator(bngsim)
        self.codegen_seconds = time.perf_counter() - t1
        self.details: dict[str, Any] = {}
        self.backend: str | None = None
        self.last_result = None
        self._columns: list[int] | None = None

    def _cols(self, result) -> list[int]:
        if self._columns is None:
            self._columns = ode._bngsim_columns(result, self.species)
        return self._columns

    def _check_times(self, result) -> None:
        got = np.asarray(result.time, dtype=float).ravel()
        grid = times(self.proto)
        if got.shape != grid.shape or not np.allclose(got, grid, rtol=0, atol=TIME_TOL * grid[-1]):
            raise RuntimeError(f"BNGsim reported {got.size} output times, not the protocol's grid")

    def ensemble(self, n: int, seed: int) -> np.ndarray:
        self.model.reset()
        result = self.sim.run_replicates(int(n), seed=int(seed), squeeze=True, **self._span())
        self.backend = (getattr(result, "ssa_diagnostics", None) or {}).get("propensity_backend")
        self.last_result = result
        stack = np.array(result.species, dtype=float)
        if stack.ndim == 2:  # one replicate
            stack = stack[None]
        return stack[:, :, self._cols(result)]

    def ensemble_detailed(self, n: int, seed: int) -> np.ndarray:
        self.model.reset()
        results = self.sim.run_replicates(int(n), seed=int(seed), **self._span())
        self._check_times(results[0])
        cols = self._cols(results[0])
        stack = np.stack([np.array(r.species, dtype=float)[:, cols] for r in results])
        self.backend = (getattr(results[0], "ssa_diagnostics", None) or {}).get(
            "propensity_backend")
        self.details = self._details(results)
        return stack

    def config(self) -> dict[str, Any]:
        return {**super().config(), "input": "sbml (Model.from_sbml)"}


class RoadRunnerSSASBML(RoadRunnerSSA):
    """libRoadRunner's ``gillespie`` integrator reading the pinned SBML: each
    compared species selected as an amount by SBML id, each trajectory after
    ``resetAll()`` (an event may assign a parameter, which ``reset()`` keeps) and a
    new seed. What libRoadRunner logs while it loads the model and runs its first
    trajectory -- its notice that time-triggered events and delays are not
    precise under gillespie, say -- is kept in its configuration."""

    def __init__(self, sbml: Path, species: list[str], proto: dict[str, Any],
                 log_path: Path | None = None):
        roadrunner = env.import_roadrunner()

        self.selections = list(species)
        self.proto = proto
        self.log_path = Path(log_path) if log_path else Path(sbml).parent / "roadrunner.log"
        self.log: list[str] = []
        self._logging = self._start_log(roadrunner)
        t0 = time.perf_counter()
        self.rr = roadrunner.RoadRunner(str(sbml))
        self.load_seconds = time.perf_counter() - t0
        self.rr.setIntegrator("gillespie")
        self.integrator = self.rr.getIntegrator()
        self.integrator.setValue("variable_step_size", False)
        self.rr.timeCourseSelections = ["time"] + self.selections
        self._first = True

    def _start_log(self, roadrunner) -> bool:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            roadrunner.Logger.enableFileLogging(str(self.log_path), roadrunner.Logger.LOG_WARNING)
            return True
        except Exception:  # noqa: BLE001 - the log is a record, not a requirement
            return False

    def _stop_log(self) -> None:
        if not self._logging:
            return
        try:
            env.import_roadrunner().Logger.disableFileLogging()
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            seen: list[str] = []
            for ln in lines:
                ln = ln.strip()[:300]
                if ln and ln not in seen and "DEPRECATED" not in ln and "commented out" not in ln:
                    seen.append(ln)
            self.log = seen[:10]
        except Exception:  # noqa: BLE001
            pass
        self._logging = False

    def trajectory(self, seed: int) -> np.ndarray:
        t_end, n_steps = _grid(self.proto)
        self.rr.resetAll()
        self.integrator.setValue("seed", int(seed))
        out = np.array(self.rr.simulate(0.0, t_end, n_steps + 1), dtype=float)
        if self._first:
            self._first = False
            self._stop_log()
            grid = times(self.proto)
            if out.shape[0] != grid.size or not np.allclose(out[:, 0], grid, rtol=0,
                                                             atol=TIME_TOL * grid[-1]):
                raise RuntimeError(f"roadrunner returned {out.shape[0]} output times, not the "
                                   "protocol's grid")
        if out.shape[0] != n_steps + 1:
            raise RuntimeError(f"roadrunner returned {out.shape[0]} time points, "
                               f"expected {n_steps + 1}")
        return out[:, 1:]

    def config(self) -> dict[str, Any]:
        cfg = {**super().config(), "input": "sbml", "reset": "resetAll"}
        if self.log:
            cfg["log"] = self.log
        return cfg


class RunNetworkSSASBML(RunNetworkSSA):
    """run_network's direct method on the network derived from the SBML
    (``derive_network``): one process per trajectory, its columns found by name."""

    def __init__(self, derived: dict[str, Any], proto: dict[str, Any], out_prefix: Path, *,
                 timeout: float):
        super().__init__(Path(derived["net"]), proto, out_prefix, timeout=timeout)
        self.columns = list(derived["columns"])

    def trajectory(self, seed: int) -> np.ndarray:
        return super().trajectory(seed)[:, self.columns]

    def config(self) -> dict[str, Any]:
        return {**super().config(),
                "input": "network derived from the SBML (bngsim.convert.sbml_to_bngl, BNG2.pl)"}


def build_reference(name: str, prep: dict[str, Any], *, out_prefix: Path, timeout: float):
    if name == "run_network":
        return RunNetworkSSASBML(prep["derived"], prep["proto"], out_prefix, timeout=timeout)
    if name == "roadrunner":
        return RoadRunnerSSASBML(prep["sbml"], prep["species_names"], prep["proto"],
                                 log_path=Path(out_prefix).parent / "roadrunner.log")
    if name == "copasi":
        return CopasiSSA(prep["sbml"], prep["species_names"], prep["name_to_id"], prep["proto"])
    raise KeyError(f"no SBML adapter for reference {name!r}")


# --------------------------------------------------------------------------- #
# Preparation and exact moments
# --------------------------------------------------------------------------- #
def prepare(problem: Problem, ctx: dict[str, Any], proto: dict[str, Any]) -> dict[str, Any]:
    """The pinned SBML, copied to the job's own scratch, its changing species and
    census, and the output rows left out of the comparison."""
    pinned = fetch.model_file(problem.source, ctx["cache_dir"])
    workdir = Path(ctx["workdir"])
    (workdir / "sbml").mkdir(parents=True, exist_ok=True)
    sbml = workdir / "sbml" / pinned.name
    shutil.copyfile(pinned, sbml)
    species, census = sbml_census(sbml)
    grid = times(proto)
    skip = excluded_rows(grid, census)
    keep = np.array([float(g) not in skip for g in grid])
    size = {"species": census["species"], "reactions": census["reactions"]}
    info: dict[str, Any] = {"network": size, "sbml": census}
    if skip:
        info["rows_not_compared"] = {"times": skip, "reason": "a time-triggered event fires "
                                     "at this output time (before or after it is a convention)"}
    if problem.size.get("reactions") not in (None, size["reactions"]):
        info["size_mismatch"] = {"spec": problem.size, "sbml": size}
    return {"format": "sbml", "sbml": sbml, "species_names": species,
            "name_to_id": {s: s for s in species}, "proto": proto, "info": info,
            "network": None, "keep": keep, "cache_dir": ctx["cache_dir"]}


def _read_table(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]
    head = [c.strip() for c in rows[0]]
    data = np.array([[float(c) for c in r] for r in rows[1:]], dtype=float)
    return data[:, 0], head[1:], data[:, 1:]


def table_moments(spec: dict[str, Any], cache_dir: str | Path, proto: dict[str, Any],
                  species: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """The pinned table's exact means and variances on the protocol grid, in the
    compared species' order: ``(n_times, n_species)`` each. The table's times must
    be the grid and its species exactly the compared ones."""
    tables = {}
    for k in ("mean", "sd"):
        src = {"repo": spec["repo"], "commit": spec["commit"], **spec[k]}
        tables[k] = _read_table(fetch.model_file(src, cache_dir))
    grid = times(proto)
    out = []
    for k in ("mean", "sd"):
        t, names, values = tables[k]
        if t.shape != grid.shape or not np.allclose(t, grid, rtol=0, atol=TIME_TOL * grid[-1]):
            raise RuntimeError(f"the {k} table's times are not the protocol's grid "
                               f"({t.size} times, the grid has {grid.size})")
        if sorted(names) != sorted(species):
            raise RuntimeError(f"the {k} table's species {names} are not the compared "
                               f"species {species}")
        out.append(values[:, [names.index(s) for s in species]])
    mean, sd = out
    return mean, sd**2
