"""Steady state of SBML models (the ss-sbml suite): the steady_state capability's
second input path.

The model is the pinned SBML, read as it is. BNGsim loads it with
``Model.from_sbml`` and solves for its steady state as on the ``.net`` path
(``steady_state(method="newton")``, the integration path beside it); libRoadRunner,
COPASI and AMICI read the same file, each through its own SBML importer, with the
solver settings of ``steady`` (``REFERENCE_SETTINGS``). What is compared is every
**state species** of the SBML (``ode.sbml_species``: neither ``constant`` nor set by
an assignment rule), by SBML id, as a **concentration** -- what BNGsim reports for an
SBML model -- never by position.

**Doses.** A point sets SBML global parameters (``parameters``, ``scan.parameter``)
or, for a scan, a species' initial concentration (``scan.species``). Every point
starts from the model's initial state at that point's values, as on the ``.net``
path: BNGsim ``set_param`` then ``reset`` (an initial value that reads the
parameter follows it), then ``set_concentration`` for a species dose;
libRoadRunner sets the parameter, or ``init([id])``, and ``reset()``s; COPASI sets
the initial value and updates the initial values that depend on it; AMICI sets the
parameter, and for a species dose its initial state, the model's own at the point's
parameters with the dosed entry replaced.

**run_network and the engine-free checks** reach the model through the network the
ODE capability derives from it (``ode.derive_network``: ``bngsim.convert.
sbml_to_bngl``, BNG2.pl, and a check that BNGsim on the derived network reproduces
BNGsim on the SBML). Each point's network is the derived ``.net`` with the dose
written in -- the parameter's value, or the dosed species' seed value -- which
run_network integrates to ``t_end`` and the harness reads (``bench.netfile``) for
the **residual** at each engine's state, with the species the SBML does not
compare held at their initial values there. The converter writes each SBML
reaction as signed per-species fluxes, so the derived network's own stoichiometry
says nothing about conservation: the **conservation laws** are taken from the
SBML's reactions instead (libsbml's reading of each reactant's and product's
stoichiometry, in amounts, so each law weighs a species by its compartment's
size), and so are the directions the reactions move the state in, which the root
classification uses. A model with no faithful derived network has run_network
*not applicable* (recorded with the reason, counted neither way) and no
engine-free checks; AMICI's integration from the initial state then decides which
state the initial state reaches.

No species is left out of an SBML model's verdict as a write-only accumulator, and
BNGsim's solve is not masked: every member of the suite converges unmasked.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from .. import env, fetch, netfile
from ..specs import Problem
from . import ode, steady
from ._common import run_isolated

# The SBML path's own version, a key only SBML jobs carry.
SBML_HARNESS_VERSION = "1"

BNGSIM_SETTINGS = {**steady.BNGSIM_SETTINGS, "input": "sbml (Model.from_sbml)",
                   "atol": "protocol (float, per-species ceiling)", "mask": None,
                   "output": "concentration"}
REFERENCE_SETTINGS = {
    **steady.REFERENCE_SETTINGS,
    "run_network": {**steady.REFERENCE_SETTINGS["run_network"],
                    "input": "network derived from the SBML (ode.derive_network)"},
    "roadrunner": {**steady.REFERENCE_SETTINGS["roadrunner"], "output": "concentration"},
    "copasi": {**steady.REFERENCE_SETTINGS["copasi"], "output": "concentration"},
    "amici": {**steady.REFERENCE_SETTINGS["amici"], "output": "state (concentration)"},
}
# The SBML path's verdict: the .net path's, with the residual's floor also scaled
# by the gross flux at each point's initial state (``residual_floor_initial``): a
# bistable SBML model's off state is often the origin, where every flux vanishes
# and the floor of the state's own largest flux is round-off.
VERDICT = {**steady.VERDICT,
           "residual_floor_of": "max_j g_j at the state and at the point's initial state",
           "residual_leaves_out": "species a rate rule moves",
           "conservation_atol": "atol_factor * protocol atol, added to the law's tolerance",
           "law_coeff_floor": 1e-12}
# A law's coefficient below this is round-off of the SVD, not part of the law.
LAW_COEFF_FLOOR = 1e-12
# When run_network does not apply, the reference whose integration from the
# initial state decides which state that reaches.
FALLBACK_INTEGRATION_REFERENCE = "amici"


def _progress(msg: str) -> None:
    steady.progress(steady.CAPABILITY, msg)


# --------------------------------------------------------------------------- #
# The SBML, read by libsbml (no engine involved)
# --------------------------------------------------------------------------- #
def read_structure(sbml: Path) -> dict[str, Any]:
    """What the harness reads from the SBML itself: each species' compartment size
    and whether reactions change it, the global parameters, the reactions'
    stoichiometry (in amounts), and the species a rate rule moves."""
    import libsbml

    doc = libsbml.readSBMLFromFile(str(sbml))
    model = doc.getModel()
    if model is None:
        raise RuntimeError(f"libsbml cannot read {sbml.name}")
    sizes = {c.getId(): (c.getSize() if c.isSetSize() else 1.0)
             for c in model.getListOfCompartments()}
    rules = {}
    for r in model.getListOfRules():
        if r.isAssignment():
            rules[r.getVariable()] = "assignment"
        elif r.isRate():
            rules[r.getVariable()] = "rate"
    species = {}
    for sp in model.getListOfSpecies():
        sid = sp.getId()
        species[sid] = {"compartment": sp.getCompartment(),
                        "size": float(sizes.get(sp.getCompartment(), 1.0)),
                        "constant": bool(sp.getConstant()),
                        "boundary": bool(sp.getBoundaryCondition()),
                        "rule": rules.get(sid)}
    parameters = {q.getId(): {"constant": bool(q.getConstant()), "rule": rules.get(q.getId())}
                  for q in model.getListOfParameters()}
    reactions = []
    for rx in model.getListOfReactions():
        terms: dict[str, float] = {}
        for sign, refs in ((-1.0, rx.getListOfReactants()), (1.0, rx.getListOfProducts())):
            for ref in refs:
                if hasattr(ref, "isSetStoichiometryMath") and ref.isSetStoichiometryMath():
                    raise ValueError(f"reaction {rx.getId()} has a stoichiometry formula")
                st = float(ref.getStoichiometry())
                st = 1.0 if np.isnan(st) else st  # SBML L3 leaves an unset one NaN
                terms[ref.getSpecies()] = terms.get(ref.getSpecies(), 0.0) + sign * st
        reactions.append({"id": rx.getId(), "terms": terms})
    return {"species": species, "parameters": parameters, "reactions": reactions,
            "rate_rules": sorted(v for v, kind in rules.items() if kind == "rate"),
            "compartments": sizes}


def dose_kinds(proto: dict[str, Any], structure: dict[str, Any]) -> dict[str, str]:
    """Each name a point sets: ``"parameter"`` (a constant SBML global parameter no
    rule sets) or ``"species"`` (one of ``species``, or the scan's, whose initial
    concentration it sets). Raises ValueError -- a bad test -- for anything else."""
    kinds: dict[str, str] = {}
    scan = proto.get("scan") or {}
    names = dict.fromkeys(proto.get("parameters") or {}, "parameter")
    names.update(dict.fromkeys(proto.get("species") or {}, "species"))
    if scan:
        names[steady.scan_target(scan)] = "species" if "species" in scan else "parameter"
    for name, kind in names.items():
        if kind == "species":
            sp = structure["species"].get(name)
            if sp is None:
                raise ValueError(f"the SBML has no species {name!r} to dose")
            if sp["rule"] is not None:
                raise ValueError(f"species {name!r} is set by a {sp['rule']} rule; its initial "
                                 "value is not a dose")
        else:
            par = structure["parameters"].get(name)
            if par is None:
                where = " (it is a species: scan it with scan.species)" \
                    if name in structure["species"] else ""
                raise ValueError(f"the SBML has no global parameter {name!r}{where}")
            if not par["constant"] or par["rule"] is not None:
                raise ValueError(f"parameter {name!r} is not a constant the model reads (a rule "
                                 "sets it); set a parameter it reads instead")
        kinds[name] = kind
    return kinds


def _split(point: dict[str, float], doses: dict[str, str]):
    params = {k: float(v) for k, v in point.items() if doses[k] == "parameter"}
    species = {k: float(v) for k, v in point.items() if doses[k] == "species"}
    return params, species


def structure_subspaces(net_names: list[str], structure: dict[str, Any]
                        ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """The conservation laws and the moving directions, over the derived network's
    species, from the SBML's own reactions.

    Returns ``(laws, moving, unplaced)``: ``laws`` one row per law, applied to
    concentrations (each column weighed by its compartment's size, so a row is a
    law of amounts); ``moving`` the columns of an orthonormal basis of the
    directions the reactions (and any rate rule) move the concentrations in;
    ``unplaced`` the SBML species a reaction or rate rule changes that the derived
    network does not have (the laws are then not taken)."""
    index = {ode.species_key(name)[1]: k for k, name in enumerate(net_names)}
    n = len(net_names)
    sp = structure["species"]
    cols, unplaced = [], set()
    for rx in structure["reactions"]:
        col = np.zeros(n)
        for sid, st in rx["terms"].items():
            info = sp.get(sid)
            if info is None or info["boundary"] or info["constant"]:
                continue  # a reaction does not change a boundary or constant species
            k = index.get(sid)
            if k is None:
                unplaced.add(sid)
                continue
            col[k] += st
        cols.append(col)
    for sid in structure["rate_rules"]:
        if sid not in sp:
            continue  # a rate rule on a parameter or compartment
        k = index.get(sid)
        if k is None:
            unplaced.add(sid)
            continue
        col = np.zeros(n)
        col[k] = 1.0
        cols.append(col)
    s = np.column_stack(cols) if cols else np.zeros((n, 0))
    size = np.ones(n)
    for name, k in index.items():
        if name in sp:
            size[k] = sp[name]["size"]
    laws, _ = netfile.subspaces_of(s)
    _, moving = netfile.subspaces_of(s / size[:, None])
    # Round-off in the basis: a law over species that are all zero (a gene held at
    # 0, say) would otherwise be judged on 1e-16 coefficients of species that move.
    laws = np.where(np.abs(laws) > LAW_COEFF_FLOOR, laws, 0.0)
    return laws * size[None, :], moving, sorted(unplaced)


def rate_rule_columns(net_names: list[str], structure: dict[str, Any]) -> list[int]:
    """The derived network's species that a rate rule moves. The converter writes a
    rate rule as one signed flux, so the network has no gross flux to hold its net
    rate against: the residual leaves these species out (the agreement and the
    integration reference's settling still hold them)."""
    index = {ode.species_key(name)[1]: k for k, name in enumerate(net_names)}
    return sorted(index[sid] for sid in structure["rate_rules"] if sid in index)


# --------------------------------------------------------------------------- #
# Preparation
# --------------------------------------------------------------------------- #
def derive_proto(proto: dict[str, Any]) -> dict[str, Any]:
    """The time course ``ode.derive_network`` checks the derived network on: the
    steady-state protocol's horizon and grid, from t = 0."""
    return {"t_start": 0.0, "t_end": float(proto["t_end"]), "n_steps": int(proto["n_steps"]),
            "rtol": float(proto["rtol"]), "atol": float(proto["atol"]), "output_start": 0.0}


def _derive_isolated(sbml: Path, species_ids: list[str], proto: dict[str, Any], workdir: Path,
                     budget: float, timeout: float) -> dict[str, Any]:
    """``ode.derive_network`` in a forked child under ``budget``; run_network not
    applicable, with the reason, when the derivation itself failed."""
    def fn(stage):
        return np.zeros(0), ode.derive_network(sbml, species_ids, derive_proto(proto),
                                               workdir / "derived", timeout=timeout,
                                               stage=stage)

    _progress(f"run_network: deriving a network from the SBML (budget {budget:g} s)")
    out = run_isolated(fn, budget, workdir / "isolated" / "derive")
    if "error" in out:
        derived = {"applicable": False, "info": {},
                   "reason": f"the network derivation failed: {out['error']}"[:400]}
    else:
        derived = out["config"]
    derived.setdefault("info", {})["seconds"] = out["seconds"]
    _progress("run_network: " + ("derived network ready" if derived.get("applicable")
                                 else f"not applicable ({derived.get('reason')})"))
    return derived


def point_networks(text: str, pts: list[dict[str, float]], doses: dict[str, str],
                   workdir: Path) -> tuple[list[Path], list[netfile.Network]]:
    """Each point's network: the derived ``.net`` with the point's parameters and
    species doses written in, as a file (for run_network) and as the harness reads
    it. Raises ValueError when the derived network has no place for a dose."""
    lines = text.splitlines(keepends=True)
    rows = ode.net_species(lines)
    index = {ode.species_key(name)[1]: k for k, (_, name) in enumerate(rows)}
    network = netfile.parse(text)
    paths, nets = [], []
    workdir.mkdir(parents=True, exist_ok=True)
    for i, pt in enumerate(pts):
        params, species = _split(pt, doses)
        for name in params:
            if name not in network.parameters:
                raise ValueError(f"the derived network has no parameter {name!r}")
            if not network.is_primary(name):
                raise ValueError(f"parameter {name!r} is an expression in the derived network")
        values = {}
        for name, v in species.items():
            if name not in index:
                raise ValueError(f"the derived network has no species {name!r}")
            values[index[name]] = v
        t = "".join(ode._set_net_values(lines, rows, values)) if values else text
        if params:
            t = netfile.with_parameters(t, params)
        path = workdir / f"point_{i:04d}.net"
        path.write_text(t, encoding="utf-8")
        paths.append(path)
        nets.append(netfile.parse(t))
    return paths, nets


def prepare(problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    """The pinned SBML (copied into the job's scratch, where AMICI compiles beside
    it), its state species and structure, the points and their doses, and the
    derived network with each point's ``.net`` -- or why run_network does not
    apply."""
    proto = steady.protocol(problem)
    pinned = fetch.model_file(problem.source, ctx["cache_dir"])
    workdir = Path(ctx["workdir"])
    (workdir / "sbml").mkdir(parents=True, exist_ok=True)
    sbml = workdir / "sbml" / pinned.name
    shutil.copyfile(pinned, sbml)
    species_ids, census = ode.sbml_species(sbml)
    structure = read_structure(sbml)
    pts = steady.points(proto)
    doses = dose_kinds(proto, structure)
    size = {"species": census["species"], "reactions": census["reactions"]}
    info: dict[str, Any] = {"network": size, "sbml": census, "points": pts, "doses": doses}
    if problem.size.get("reactions") not in (None, size["reactions"]):
        info["size_mismatch"] = {"spec": problem.size, "sbml": size}
    prep: dict[str, Any] = {
        "format": "sbml", "sbml": sbml, "proto": proto, "points": pts, "info": info,
        "species_names": species_ids, "species_ids": species_ids, "doses": doses,
        "structure": structure, "sinks": [], "point_networks": None, "point_nets": None,
        "laws": np.zeros((0, 0)), "law_species": [], "x0": None, "moving_basis": None,
        "net_columns": None, "not_applicable": {}, "residual_floor_initial": True,
        "conservation_atol": steady.VERDICT["atol_factor"] * float(proto["atol"]),
        "integration_reference": steady.INTEGRATION_REFERENCE,
    }
    budget = float(ctx.get("engine_budget_seconds") or steady.engine_budget_seconds(problem))
    timeout = min(float(ctx.get("subprocess_timeout") or budget), budget)
    derived = _derive_isolated(sbml, species_ids, proto, workdir, budget, timeout)
    info["derived_network"] = derived.get("info", {})
    reason = None if derived.get("applicable") else str(derived.get("reason"))
    if reason is None:
        text = Path(derived["net"]).read_text(encoding="utf-8")
        try:
            paths, nets = point_networks(text, pts, doses, workdir / "points")
        except (ValueError, netfile.NetFileError, ArithmeticError) as exc:
            reason = f"the derived network cannot take the dose: {exc}"[:400]
    if reason is not None:
        prep["not_applicable"] = {"run_network": reason}
        prep["integration_reference"] = FALLBACK_INTEGRATION_REFERENCE
        info["netfile_error"] = f"no derived network to check against: {reason}"[:400]
        return prep
    net_names = [name for _, name in ode.net_species(text.splitlines(keepends=True))]
    laws, moving, unplaced = structure_subspaces(net_names, structure)
    prep.update({
        "derived": derived, "point_nets": paths, "point_networks": nets,
        "net_columns": list(derived["columns"]), "net_species_names": net_names,
        "x0": np.array([n.initial for n in nets], dtype=float),
        "law_species": list(range(len(net_names))),
        "laws": laws if not unplaced else np.zeros((0, len(net_names))),
        "moving_basis": moving if not unplaced else None,
        "residual_skip": rate_rule_columns(net_names, structure),
    })
    if prep["residual_skip"]:
        info["residual_leaves_out"] = [net_names[k] for k in prep["residual_skip"]]
    if unplaced:
        info["conservation_not_checked"] = (f"the derived network lacks species the SBML's "
                                            f"reactions change: {unplaced[:5]}")
    info["conservation_laws"] = int(prep["laws"].shape[0])
    return prep


# --------------------------------------------------------------------------- #
# The system under test and the reference engines, reading the SBML
# --------------------------------------------------------------------------- #
class BngsimSteadySBML(steady.BngsimSteady):
    """BNGsim's steady state on the SBML itself (``Model.from_sbml``), each point
    from the model's initial state at its doses. ``atol`` is the plain float an
    SBML user passes (as in the ODE capability's SBML path), and the compared
    species are selected by SBML id from what the solve reports."""

    def __init__(self, sbml: Path, species_ids: list[str], proto: dict[str, Any],
                 pts: list[dict[str, float]], doses: dict[str, str], method: str):
        import bngsim

        self.proto, self.points, self.method = proto, pts, method
        self.species_ids, self.doses = list(species_ids), doses
        t0 = time.perf_counter()
        self.model = bngsim.Model.from_sbml(str(sbml))
        self.load_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        self.sim = bngsim.Simulator(self.model, method="ode")
        self.codegen_seconds = time.perf_counter() - t1
        self.atol = float(proto["atol"])
        self.mask = None
        self.columns: list[int] | None = None
        self.diagnostics: list[dict[str, Any]] = []

    def solve(self, point: dict[str, float]):
        params, species = _split(point, self.doses)
        for name, value in params.items():
            self.model.set_param(name, value)
        self.model.reset()
        for name, value in species.items():
            self.model.set_concentration(name, value)
        p = self.proto
        return self.sim.steady_state(method=self.method, max_time=float(p["t_end"]),
                                     rtol=float(p["rtol"]), atol=self.atol)

    def state(self, result) -> np.ndarray:
        if self.columns is None:
            self.columns = ode._bngsim_columns(result, self.species_ids)
        return np.array(result.concentrations, dtype=float)[self.columns]

    def config(self) -> dict[str, Any]:
        return {**super().config(), "input": "sbml (Model.from_sbml)",
                "atol": "float (per-species ceiling)"}


class RunNetworkSteadySBML(steady.RunNetworkSteady):
    """run_network on each point's derived network (``prepare``), integrating to
    ``t_end`` as on the ``.net`` path; its columns are the compared species, found
    in the derived network by name."""

    def __init__(self, nets: list[Path], proto: dict[str, Any], out_prefix: Path, *,
                 timeout: float, columns: list[int]):
        super().__init__(nets, proto, out_prefix, timeout=timeout, sinks=[])
        self.columns = list(columns)

    def solve_point(self, i: int) -> np.ndarray:
        return super().solve_point(i)[:, self.columns]

    def config(self) -> dict[str, Any]:
        return {**super().config(), "input": "network derived from the SBML"}


class RoadRunnerSteadySBML(steady.RoadRunnerSteady):
    """libRoadRunner's steady-state solver on the SBML, with the ``.net`` path's
    settings (NLEQ2, conserved-moiety analysis, presimulation). Each point sets its
    parameters and species' initial concentrations (``init([id])``), ``reset()``s
    and solves; species are read as concentrations (``[id]``)."""

    def __init__(self, sbml: Path, species_ids: list[str], proto: dict[str, Any],
                 pts: list[dict[str, float]], doses: dict[str, str]):
        roadrunner = env.import_roadrunner()

        self.ids = list(species_ids)
        self.proto, self.points, self.doses = proto, pts, doses
        t0 = time.perf_counter()
        self.rr = roadrunner.RoadRunner(str(sbml))
        self.rr.conservedMoietyAnalysis = True
        self.load_seconds = time.perf_counter() - t0
        integrator = self.rr.getIntegrator()
        integrator.setValue("relative_tolerance", float(proto["rtol"]))
        integrator.setValue("absolute_tolerance", float(proto["atol"]))
        self.solver = self.rr.getSteadyStateSolver()
        self.solver.setValue("allow_presimulation", True)

    def solve_all(self, detailed: bool = False) -> np.ndarray:
        out = []
        for i, point in enumerate(self.points):
            params, species = _split(point, self.doses)
            for name, value in params.items():
                self.rr[name] = value
            for name, value in species.items():
                self.rr.setValue(f"init([{name}])", value)
            self.rr.reset()
            try:
                self.rr.steadyState()
            except Exception as exc:  # noqa: BLE001 - its message says why
                raise RuntimeError(f"steadyState() failed at point {i}: {exc}") from exc
            out.append([self.rr[f"[{s}]"] for s in self.ids])
        return np.array(out, dtype=float)

    def config(self) -> dict[str, Any]:
        return {**super().config(), "output": "concentration"}


class CopasiSteadySBML(steady.CopasiSteady):
    """COPASI's steady-state task on the SBML, as on the ``.net`` path; a species
    dose sets the metabolite's initial concentration (and the initial values that
    depend on it). Concentrations are read by SBML id."""

    def __init__(self, sbml: Path, species_ids: list[str], proto: dict[str, Any],
                 pts: list[dict[str, float]], doses: dict[str, str]):
        import basico
        import COPASI

        self._copasi = COPASI
        self.proto, self.points, self.doses = proto, pts, doses
        t0 = time.perf_counter()
        self.dm = basico.load_model(str(sbml))
        if self.dm is None:
            raise RuntimeError("COPASI could not import the SBML")
        self.model = self.dm.getModel()
        metabs = self.model.getMetabolites()
        by_id = {metabs.get(i).getSBMLId(): metabs.get(i) for i in range(metabs.size())}
        missing = [i for i in species_ids if i not in by_id]
        if missing:
            raise RuntimeError(f"COPASI's model lacks species {missing[:3]}")
        self.metabs = [by_id[i] for i in species_ids]
        values = self.model.getModelValues()
        params = {values.get(i).getSBMLId(): values.get(i) for i in range(values.size())}
        self.params, self.dosed = {}, {}
        for name, kind in doses.items():
            table, where = (params, "parameter") if kind == "parameter" else (by_id, "species")
            if name not in table:
                raise RuntimeError(f"COPASI's model has no {where} {name!r}")
            (self.params if kind == "parameter" else self.dosed)[name] = table[name]
        self.task = self.dm.getTask("Steady-State")
        self.settings = basico.get_task_settings("Steady-State", model=self.dm).get("method", {})
        self.load_seconds = time.perf_counter() - t0

    def solve_all(self, detailed: bool = False) -> np.ndarray:
        cp = self._copasi
        out = []
        for i, point in enumerate(self.points):
            params, species = _split(point, self.doses)
            for name, value in params.items():
                param = self.params[name]
                param.setInitialValue(value)
                self.model.updateInitialValues(param.getInitialValueReference())
            for name, value in species.items():
                metab = self.dosed[name]
                metab.setInitialConcentration(value)
                self.model.updateInitialValues(metab.getInitialConcentrationReference())
            before = cp.CCopasiMessage.size()
            if not self.task.initializeRaw(cp.CCopasiTask.OUTPUT_UI):
                msgs = cp.CCopasiMessage.getAllMessageText() \
                    if cp.CCopasiMessage.size() > before else ""
                raise RuntimeError(f"COPASI steady-state task initialisation failed at point "
                                   f"{i}: {msgs.strip()[:300]}")
            self.task.processRaw(True)
            result = int(self.task.getResult())
            if result not in (1, 2):
                raise RuntimeError(f"COPASI steady state {self.RESULTS.get(result, result)} "
                                   f"at point {i}")
            out.append([m.getConcentration() for m in self.metabs])
        return np.array(out, dtype=float)

    def config(self) -> dict[str, Any]:
        return {**super().config(), "output": "concentration"}


class AmiciSteadySBML(steady.AmiciSteady):
    """AMICI's steady state by simulation on the SBML (``integrationOnly``, the
    ``.net`` path's settings), compiled once. A parameter dose sets the parameter by
    id; a species dose sets the initial state: AMICI's own at the point's
    parameters, with the dosed species' entry replaced."""

    def __init__(self, sbml: Path, species_ids: list[str], proto: dict[str, Any],
                 pts: list[dict[str, float]], doses: dict[str, str]):
        import amici
        from amici.sim import sundials as asun

        self.proto, self.points, self.doses = proto, pts, doses
        outdir = sbml.parent / "amici_steady"
        t0 = time.perf_counter()
        amici.SbmlImporter(str(sbml)).sbml2amici(
            "bench_steady", str(outdir), generate_sensitivity_code=False, verbose=False)
        self.codegen_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        self.model = amici.import_model_module("bench_steady", str(outdir)).get_model()
        self.model.set_timepoints([np.inf])
        self.model.set_steady_state_computation_mode(
            asun.SteadyStateComputationMode_integrationOnly)
        self.load_seconds = time.perf_counter() - t1
        pos = {sid: i for i, sid in enumerate(self.model.get_state_ids())}
        unmapped = [i for i in species_ids if i not in pos]
        if unmapped:
            raise RuntimeError(f"AMICI has no state for {len(unmapped)} compared SBML species "
                               f"(e.g. {unmapped[:3]})")
        self.idx = [pos[i] for i in species_ids]
        free = set(self.model.get_free_parameter_ids())
        fixed = set(self.model.get_fixed_parameter_ids())
        self.setters: dict[str, Any] = {}
        self.state_idx: dict[str, int] = {}
        for name, kind in doses.items():
            if kind == "species":
                if name not in pos:
                    raise RuntimeError(f"AMICI has no state for the dosed species {name!r}")
                self.state_idx[name] = pos[name]
                continue
            pid = next((c for c in (name, f"amici_{name}") if c in free | fixed), None)
            if pid is None:
                raise RuntimeError(f"AMICI exposes no parameter {name!r}")
            setter = (self.model.set_free_parameter_by_id if pid in free
                      else self.model.set_fixed_parameter_by_id)
            self.setters[name] = (setter, pid)

    def solve_all(self, detailed: bool = False) -> np.ndarray:
        from amici.sim.sundials import run_simulation

        out = []
        for i, point in enumerate(self.points):
            params, species = _split(point, self.doses)
            for name, value in params.items():
                setter, pid = self.setters[name]
                setter(pid, value)
            if self.state_idx:
                self.model.set_initial_state([])  # the model's own, at these parameters
                x0 = np.array(self.model.get_initial_state(), dtype=float)
                for name, value in species.items():
                    x0[self.state_idx[name]] = value
                self.model.set_initial_state(x0.tolist())
            rdata = run_simulation(self.model, self._solver())
            if rdata.status != 0:
                raise RuntimeError(f"AMICI steady state failed at point {i} with status "
                                   f"{rdata.status} (posteq_status "
                                   f"{[int(s) for s in rdata.posteq_status]})")
            x = np.asarray(rdata.x, dtype=float)[-1][self.idx]
            if not np.all(np.isfinite(x)):
                raise RuntimeError(f"AMICI returned non-finite values with a success status "
                                   f"at point {i}")
            out.append(x)
        return np.array(out)

    def config(self) -> dict[str, Any]:
        return {**super().config(), "output": "state (concentration)"}


# The SBML engine for each of ``steady.REFERENCE_ENGINES`` (same names, same order,
# so an SBML job's id reads like a .net job's).
SBML_REFERENCE_ENGINES: dict[str, type] = {
    "run_network": RunNetworkSteadySBML,
    "roadrunner": RoadRunnerSteadySBML,
    "copasi": CopasiSteadySBML,
    "amici": AmiciSteadySBML,
}


def build_reference(name: str, prep: dict[str, Any], *, out_prefix: Path, timeout: float):
    cls = SBML_REFERENCE_ENGINES[name]
    if cls is RunNetworkSteadySBML or issubclass(cls, steady.RunNetworkSteady):
        if not prep.get("point_nets"):
            raise RuntimeError("run_network has no derived network for this model: "
                               + json.dumps(prep.get("not_applicable", {}))[:300])
        return cls(prep["point_nets"], prep["proto"], out_prefix, timeout=timeout,
                   columns=prep["net_columns"])
    return cls(prep["sbml"], prep["species_ids"], prep["proto"], prep["points"], prep["doses"])


def build_bngsim(prep: dict[str, Any], method: str) -> BngsimSteadySBML:
    return BngsimSteadySBML(prep["sbml"], prep["species_ids"], prep["proto"], prep["points"],
                            prep["doses"], method)
