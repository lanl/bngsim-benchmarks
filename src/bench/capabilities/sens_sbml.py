"""Forward sensitivities on SBML, from PEtab benchmark problems (the sens-sbml suite):
the sens capability's second input path.

The input is a PEtab problem (``bench.petab_problem``): the pinned SBML (the spec's
``source``) and its four tables (the spec's ``petab:`` block, each pinned by sha256
at the same commit). The problem is applied as PEtab defines it -- every simulation
condition of the measurement table with its condition-table overrides, the
parameter table's nominal values, a pre-equilibration to steady state where the
measurement table names one, the observables and noise model of every measurement
row -- and the sensitivity set is the estimated parameters that are the model's own
SBML parameters (the PEtab parameters that enter the ODE; noise, scaling and
condition-table parameters never reach the state).

**Engines.** The system under test is BNGsim on the SBML (``Model.from_sbml``), one
model and one Simulator for the sensitivity set, run through every condition: the
condition's parameters (``set_param``), ``reset()`` (so an initial assignment
follows them), its species' initial values (``set_concentration``), then a run
from t = 0 to the condition's measurement times. A pre-equilibration is BNGsim's
own recipe (docs/user-guide/steady-state.md, "Pre-equilibration"): the
pre-equilibration condition run with ``steady_state=True`` on the same Simulator,
then the simulation condition's parameters, then the measured run with
``carry_sensitivities=True``. A pre-equilibration BNGsim does not report steady
(``steady_state_reached`` false: for a model that reads the clock it never stops
early, issue #710) is BNGsim's refusal, recorded as UNSUPPORTED in its own words,
and so is a species reset after one (bngsim does not carry sensitivities across a
reset). Both of CVODES' correctors are run, as on the ``.net`` path.

The references:

- **amici** -- AMICI's own PEtab importer (``amici.importers.petab.v1``), which reads
  the problem through the ``petab`` library: an application of the problem
  independent of the harness's. Its forward sensitivities of the states and of
  the observables, its log-likelihood and its gradient, per condition, with its
  steady-state pre-equilibration. Its sensitivities are on the PEtab parameter
  scale and are put back on a linear one.
- **fd_roadrunner** -- central finite differences from libRoadRunner, which shares
  no code with BNGsim: the harness's application of each condition, every
  parameter of the sensitivity set moved up and down by a relative step ``h`` and
  ``h/2`` at tolerances far tighter than the protocol's, a pre-equilibration by
  integration to ``preeq_max_time`` (its residual recorded). Observables follow from
  its state sensitivities through the same formulas as BNGsim's.

COPASI is not wired for PEtab conditions on this path (recorded as not applicable).

**What is compared.** The outputs are the state species (by SBML semantics: an
amount for a species with ``hasOnlySubstanceUnits``, a concentration otherwise;
BNGsim reports concentrations, which the harness multiplies by the constant
compartment size where SBML means an amount) and the observables, each at the
measurement times of every condition, the conditions laid end to end on one
time axis; an observable is read only at the rows that measure it. The verdict is
the sens capability's column error (``bench.sensitivity``) on the semi-scaled
sensitivity, with its refutation and coverage rules. Beside it, two checks of the
problem as a whole: BNGsim's log-likelihood at nominal against AMICI's (and the
reference value AMICI's v1.1.0 benchmark test holds, where the spec gives one),
and its gradient over the sensitivity set against AMICI's forward gradient. The
outcome is DIFF when the column verdict, the likelihood or the gradient disagrees
with an unrefuted reference.

**Timing** interleaves, per engine, one evaluation of the whole problem: every
condition's forward-sensitivity solve (BNGsim's two correctors, AMICI's forward
mode through its PEtab simulation), a plain solve of every condition (BNGsim and
libRoadRunner: the unit of a finite-difference gradient) and AMICI's adjoint
gradient of the PEtab log-likelihood (decision D7, timing only).
"""

from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from .. import env, fetch
from .. import sensitivity as sv
from ..jobs import CORRECTNESS, TIMING, Job
from ..petab_problem import PetabError, PetabProblem, state_species
from ..specs import Problem
from ..timing import absolute_time_stats, round_robin_interleaved
from . import sens
from ._common import correctness_timeouts, engine_role, reproducibility, run_isolated, threads

CAPABILITY = sens.CAPABILITY
# The SBML path's own version, a key only SBML jobs carry.
SBML_HARNESS_VERSION = "1"
REFERENCES = ("amici", "fd_roadrunner")
NOT_WIRED = {"copasi": "COPASI is not wired for PEtab conditions on the SBML path"}
CONFIGURATIONS = sens.CONFIGURATIONS
DEFAULT_CONFIGURATION = sens.DEFAULT_CONFIGURATION
# The column verdict is the .net path's; the problem-level checks are this path's:
# - llh_atol / llh_rtol: BNGsim's log-likelihood against AMICI's, |a - b| <=
#   llh_atol + llh_rtol |b| (AMICI's own benchmark test compares with atol 1e-3);
# - gradient_tol: the largest difference of the semi-scaled gradient p dL/dp over
#   the sensitivity set, as a fraction of the gradient's size (``gradient_check``).
VERDICT = {**sens.VERDICT, "llh_atol": 1e-3, "llh_rtol": 1e-6, "gradient_tol": 1e-3}
FD_DEFAULTS = {"fd_step": 1e-3, "fd_rtol": 1e-10}
PREEQ_DEFAULTS = {"preeq_max_time": 1.0e6}
TIMING_PROTOCOL = sens.TIMING_PROTOCOL
REPRO_TOL = sens.REPRO_TOL
SETTING = sens.DEFAULT_SETTING


def engines_id() -> str:
    return "bngsim-vs-" + "-".join(REFERENCES)


def protocol(problem: Problem) -> dict[str, Any]:
    p = dict(problem.protocol_for(CAPABILITY))
    for k, v in {**FD_DEFAULTS, **PREEQ_DEFAULTS}.items():
        p.setdefault(k, v)
    p.setdefault("fd_atol", float(p["atol"]) / 100.0)
    return p


def _petab_identity() -> str | None:
    try:
        import petab

        return getattr(petab, "__version__", None)
    except Exception:  # noqa: BLE001 - recorded as missing
        return None


def jobs(problem: Problem) -> list[Job]:
    engines = engines_id()
    proto = protocol(problem)
    base = {
        "harness": f"{CAPABILITY}/{engines}/{sens.HARNESS_VERSION}",
        "input": {"format": "sbml", "harness": SBML_HARNESS_VERSION},
        "source_sha256": problem.source["sha256"],
        "petab_sha256": {k: v["sha256"] for k, v in sorted(problem.petab.items())},
        "protocol": proto,
        "setting": {"configurations": CONFIGURATIONS, "codegen": "auto",
                    "linear_solver": "auto"},
        "reference_versions": {
            "amici": env.amici_identity().get("version"),
            "roadrunner": env.roadrunner_identity().get("version"),
            "petab": _petab_identity(),
        },
    }
    est_c, est_t = planning_seconds(problem)
    correctness = Job(CAPABILITY, problem.id, engines, SETTING, CORRECTNESS,
                      definition={**base, "verdict": VERDICT}, est_seconds=est_c,
                      timeout_seconds=max(sens.MIN_TIMEOUT_SECONDS, sens.TIMEOUT_FACTOR * est_c,
                                          4 * engine_budget_seconds(problem)))
    timing = Job(CAPABILITY, problem.id, engines, SETTING, TIMING,
                 definition={**base, "timing": TIMING_PROTOCOL,
                             "timing_check": sens.REPRO_CHECK_VERSION},
                 est_seconds=est_t,
                 timeout_seconds=max(sens.MIN_TIMEOUT_SECONDS, sens.TIMEOUT_FACTOR * est_t),
                 requires=correctness.id)
    return [correctness, timing]


# --------------------------------------------------------------------------- #
# Planning (never part of a freshness key)
# --------------------------------------------------------------------------- #
def engine_seconds(problem: Problem) -> dict[str, float]:
    """Planned seconds for each engine's cold build and first evaluation."""
    size = problem.size
    rxn = size.get("reactions") or 10 * size.get("species", 10)
    n_p = len(problem.protocol_for(CAPABILITY).get("parameters") or []) or 20
    conds = size.get("conditions", 1)
    solve = conds * (0.003 + 2e-5 * rxn ** 1.5)
    sensitivity = 2.0 * (1 + n_p) * solve
    return {
        "bngsim": 5.0 + 2e-2 * rxn + sensitivity,
        "bngsim_simultaneous": 5.0 + 2e-2 * rxn + sensitivity,
        "amici": 30.0 + 0.5 * rxn + 0.3 * sensitivity,
        "fd_roadrunner": 2.0 + (4 * n_p + 1) * 3.0 * solve,
    }


def planning_seconds(problem: Problem) -> tuple[float, float]:
    per = engine_seconds(problem)
    correctness = 30.0 + sum(per.values())
    builds = per["bngsim"] + per["bngsim_simultaneous"] + per["amici"]
    timing = 60.0 + builds + (TIMING_PROTOCOL["min_rounds"] + 2) * 0.5 * builds
    return correctness, timing


def engine_budget_seconds(problem: Problem) -> float:
    slowest = max(engine_seconds(problem).values())
    return min(sens.ENGINE_BUDGET_CAP_SECONDS,
               max(sens.MIN_TIMEOUT_SECONDS, sens.TIMEOUT_FACTOR * slowest))


# --------------------------------------------------------------------------- #
# Preparation
# --------------------------------------------------------------------------- #
def _fetch_petab(problem: Problem, ctx: dict[str, Any], dest: Path) -> dict[str, Path]:
    """The SBML and the PEtab tables, sha256-checked, copied side by side into the
    job's scratch (where AMICI compiles beside them)."""
    dest.mkdir(parents=True, exist_ok=True)
    files = {"sbml": fetch.model_file(problem.source, ctx["cache_dir"])}
    for role, f in problem.petab.items():
        files[role] = fetch.model_file({"repo": problem.source["repo"],
                                        "commit": problem.source["commit"],
                                        "path": f["path"], "sha256": f["sha256"]},
                                       ctx["cache_dir"])
    out = {}
    for role, path in files.items():
        out[role] = dest / path.name
        shutil.copyfile(path, out[role])
    return out


def prepare(problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    proto = protocol(problem)
    workdir = Path(ctx["workdir"])
    files = _fetch_petab(problem, ctx, workdir / "petab")
    pp = PetabProblem(files["sbml"], files["conditions"], files["measurements"],
                      files["observables"], files["parameters"])
    params, excluded = pp.sensitivity_parameters()
    listed = proto.get("parameters")
    if listed is not None and list(listed) != params:
        raise ValueError(f"protocol.sens.parameters is not the problem's sensitivity set: "
                         f"spec {list(listed)[:5]}... vs PEtab {params[:5]}... "
                         f"({len(listed)} vs {len(params)})")
    if not params:
        raise ValueError(f"no parameter to differentiate (left out: {excluded})")
    states = state_species(pp.model)
    info_m = pp.model
    hosu = [s for s in states if info_m["species"][s]["hosu"]]
    variable = [c for c, d in info_m["compartments"].items() if not d["constant"]]
    sizes = np.array([info_m["compartments"][info_m["species"][s]["compartment"]]["size"]
                      if s in hosu else 1.0 for s in states])
    times = [c.times for c in pp.conditions]
    offsets = np.cumsum([0] + [t.size for t in times])
    output_model = pp.output_model(states, params)
    observables = output_model.observables
    cells = _observable_cells(pp, offsets, observables)
    values = np.array([pp.nominal[p] for p in params])
    info = {
        "network": {"species": len(info_m["species"]), "state_species": len(states),
                    "events": info_m["events"]},
        "petab": pp.summary(),
        "parameters": {"selection": "estimated SBML parameters", "n": len(params),
                       "names": params, "excluded": excluded,
                       "scales_in_petab": {p: pp.scales.get(p) for p in params}},
        "species_in_amount": hosu,
    }
    if variable:
        info["variable_compartments"] = variable
    if hosu and variable and any(info_m["species"][s]["compartment"] in variable for s in hosu):
        raise ValueError("an amount-only species in a compartment whose size changes: the "
                         "harness's concentration-to-amount factor would not be constant")
    return {"format": "sbml", "proto": proto, "petab": pp, "files": files,
            "params": params, "values": values, "scales": np.abs(values),
            "states": states, "amount_factor": sizes, "times": times, "offsets": offsets,
            "observables": observables, "cells": cells, "output_model": output_model,
            "info": info}


def _observable_cells(pp: PetabProblem, offsets: np.ndarray, observables: list[str]):
    """Each measurement row's cell on the outputs' time axis, ``(global time index,
    observable index)``; the first row of a cell is the one read (replicates)."""
    index = {o: k for k, o in enumerate(observables)}
    seen: dict[tuple[int, int], int] = {}
    cell_rows = []
    for j, row in enumerate(pp.rows):
        cell = (int(offsets[row.condition] + row.time_index), index[row.observable])
        if cell not in seen:
            seen[cell] = j
            cell_rows.append((j, *cell))
    return cell_rows


# --------------------------------------------------------------------------- #
# Engines
# --------------------------------------------------------------------------- #
def _sample_times(times: np.ndarray) -> tuple[list[float], slice]:
    """The run's sample times from t = 0, and the rows of its output that are the
    measurement times (a run needs two sample times; a condition measured at t = 0
    alone is run to a moment after it, and that row dropped)."""
    t = [float(x) for x in times]
    if t[0] > 0:
        return [0.0, *t], slice(1, 1 + len(t))
    if len(t) == 1:
        return [0.0, 1e-9], slice(0, 1)
    return t, slice(0, len(t))


class BngsimPetab:
    """BNGsim on the PEtab problem's SBML (see the module docstring)."""

    def __init__(self, prep: dict[str, Any], *, sensitivity_method: str | None = "staggered"):
        import bngsim

        self.prep, self.proto = prep, prep["proto"]
        self.method = sensitivity_method
        self.params = list(prep["params"]) if sensitivity_method else []
        t0 = time.perf_counter()
        self.model = bngsim.Model.from_sbml(str(prep["files"]["sbml"]))
        self.load_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        kw = ({"sensitivity_params": self.params, "sensitivity_method": sensitivity_method}
              if sensitivity_method else {})
        self.sim = bngsim.Simulator(self.model, method="ode", **kw)
        self.codegen_seconds = time.perf_counter() - t1
        self.atol = bngsim.ExactAtol(float(self.proto["atol"]))
        self.rtol = float(self.proto["rtol"])
        names = list(self.model.species_names)
        pos = {n: i for i, n in enumerate(names)}
        missing = [s for s in prep["states"] if s not in pos]
        if missing:
            raise RuntimeError(f"BNGsim reports no species for {len(missing)} state species "
                               f"(e.g. {missing[:3]})")
        self.columns = [pos[s] for s in prep["states"]]
        self.factor = prep["amount_factor"]
        self.species_index = pos
        self.stats: dict[str, Any] = {}
        self.preeq: list[dict[str, Any]] = []

    def _set_species(self, values: dict[str, float]) -> None:
        info = self.prep["petab"].model
        for s, v in values.items():
            d = info["species"][s]
            conc = v / info["compartments"][d["compartment"]]["size"] if d["hosu"] else v
            self.model.set_concentration(s, float(conc))

    def _condition(self, cond) -> Any:
        sens_on = bool(self.params)
        sample, lead = _sample_times(cond.times)
        if cond.preeq:
            for p, v in cond.preeq_parameters.items():
                self.model.set_param(p, v)
            self.model.reset()
            self._set_species(cond.preeq_species)
            r = self.sim.run(t_span=(0.0, float(self.proto["preeq_max_time"])), n_points=2,
                             steady_state=True, rtol=self.rtol, atol=self.atol)
            stats = dict(getattr(r, "solver_stats", None) or {})
            reached = bool(stats.get("steady_state_reached"))
            self.preeq.append({"condition": cond.id, "steady_state_reached": reached,
                               "t_end": float(np.asarray(r.time)[-1])})
            if not reached:
                raise RuntimeError(
                    f"UNSUPPORTED: bngsim did not reach a steady state in the pre-equilibration "
                    f"{cond.preeq!r} of {cond.sim!r} within {self.proto['preeq_max_time']:g} "
                    f"(solver_stats steady_state_reached false; a model that reads the clock or "
                    f"has an event is never marked steady, bngsim issue #710)")
            for p, v in cond.parameters.items():
                self.model.set_param(p, v)
            if cond.species:
                raise RuntimeError(
                    f"UNSUPPORTED: the simulation condition {cond.sim!r} resets species "
                    f"{sorted(cond.species)[:3]} after its pre-equilibration; bngsim does not "
                    f"carry forward sensitivities across a species reset")
            extra = {"carry_sensitivities": True} if sens_on else {}
            result = self.sim.run(sample_times=sample, rtol=self.rtol, atol=self.atol, **extra)
        else:
            for p, v in cond.parameters.items():
                self.model.set_param(p, v)
            self.model.reset()
            self._set_species(cond.species)
            result = self.sim.run(sample_times=sample, rtol=self.rtol, atol=self.atol)
        return result, lead

    def run(self):
        self.preeq = []
        return [self._condition(c) for c in self.prep["petab"].conditions]

    def tensors(self, results) -> tuple[list[np.ndarray], list[np.ndarray]]:
        X, S = [], []
        for result, lead in results:
            x = np.asarray(result.species, dtype=float)[lead, self.columns] * self.factor
            X.append(x)
            if self.params:
                names = list(result.sensitivity_params)
                if names != self.params:
                    raise RuntimeError(f"bngsim returned sensitivities for {names[:5]}..., not "
                                       f"the {len(self.params)} parameters asked for")
                s = np.asarray(result.sensitivities, dtype=float)[lead, self.columns, :]
                S.append(s * self.factor[None, :, None])
                self.stats = dict(getattr(result, "solver_stats", None) or {})
            else:
                S.append(np.zeros((x.shape[0], x.shape[1], 0)))
        return X, S

    def config(self) -> dict[str, Any]:
        from .ode import LINEAR_SOLVER_NAMES

        code = self.stats.get("linear_solver")
        out = {
            "input": "sbml (Model.from_sbml)",
            "sensitivity_method": self.method,
            "codegen_backend": getattr(self.sim, "codegen_backend", None),
            "load_seconds": self.load_seconds,
            "codegen_seconds": self.codegen_seconds,
            "atol": "ExactAtol (scalar, held exactly)",
        }
        if self.params:
            out.update({
                "analytic_sensitivity_rhs": bool(self.sim.has_analytic_sens_rhs),
                "sensitivity_rhs_decline": self.sim.sens_rhs_decline_reason,
                "linear_solver": LINEAR_SOLVER_NAMES.get(code, f"kind_{code}")
                if code is not None else None,
            })
        if self.preeq:
            out["preequilibration"] = self.preeq
        return out


def _unscale(sx: np.ndarray, values: np.ndarray, scales: list[str]) -> np.ndarray:
    """AMICI's sensitivities on each parameter's PEtab scale, back on a linear one
    (axis -1 is the parameter axis)."""
    factor = np.ones(len(values))
    for k, (v, s) in enumerate(zip(values, scales)):
        if s == "log10":
            factor[k] = 1.0 / (v * np.log(10.0))
        elif s == "log":
            factor[k] = 1.0 / v
    return sx * factor


class AmiciPetab:
    """AMICI's PEtab importer on the problem (see the module docstring). Built once
    (code generation and the C++ compile: ``codegen_seconds``), then each run is one
    ``simulate_petab`` of every condition at the nominal parameters, with forward
    sensitivities at the protocol's tolerances."""

    NAME = "amici"

    def __init__(self, prep: dict[str, Any], workdir: Path, *, adjoint: bool = False):
        import amici.sim.sundials as ss
        import petab.v1 as petab
        from amici.importers.petab.v1 import import_petab_problem
        from amici.sim.sundials.petab.v1 import create_edatas
        from amici.sim.sundials.petab.v1._parameter_mapping import create_parameter_mapping

        self._ss = ss
        self.prep, self.proto = prep, prep["proto"]
        f = prep["files"]
        self.problem = petab.Problem.from_files(
            sbml_file=str(f["sbml"]), condition_file=str(f["conditions"]),
            measurement_file=str(f["measurements"]), parameter_file=str(f["parameters"]),
            observable_files=[str(f["observables"])])
        outdir = Path(workdir) / "amici_petab_model"
        module = "bench_petab_" + hashlib.sha1(str(outdir.resolve()).encode()).hexdigest()[:10]
        t0 = time.perf_counter()
        self.model = import_petab_problem(self.problem, output_dir=str(outdir),
                                          model_name=module, compile_=True,
                                          generate_sensitivity_code=True)
        self.codegen_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        self.conditions = self.problem.get_simulation_conditions_from_measurement_df()
        self.mapping = create_parameter_mapping(petab_problem=self.problem,
                                                simulation_conditions=self.conditions,
                                                scaled_parameters=True,
                                                amici_model=self.model)
        self.edatas = create_edatas(amici_model=self.model, petab_problem=self.problem,
                                    simulation_conditions=self.conditions)
        self.load_seconds = time.perf_counter() - t1
        self.adjoint = adjoint
        # AMICI's conditions in the harness's order, by (pre-equilibration, simulation).
        order = {}
        cols = list(self.conditions.columns)
        for i, r in self.conditions.reset_index(drop=True).iterrows():
            pre = r["preequilibrationConditionId"] if "preequilibrationConditionId" in cols \
                else None
            order[(pre or None, r["simulationConditionId"])] = i
        self.order = [order[(c.preeq, c.sim)] for c in prep["petab"].conditions]
        state_ids = list(self.model.get_state_ids())
        pos = {s: i for i, s in enumerate(state_ids)}
        missing = [s for s in prep["states"] if s not in pos]
        if missing:
            raise RuntimeError(f"AMICI has no state for {missing[:3]}")
        self.idx = [pos[s] for s in prep["states"]]
        free = list(self.model.get_free_parameter_ids())
        self.free = free
        self.covered = np.array([p in free for p in prep["params"]], bool)
        self.pidx = [free.index(p) for p, c in zip(prep["params"], self.covered) if c]
        self.uncovered = {p: "not a free parameter of AMICI's PEtab model"
                          for p, c in zip(prep["params"], self.covered) if not c}
        self.obs_ids = list(self.model.get_observable_ids())

    def _solver(self, method):
        ss = self._ss
        solver = self.model.create_solver()
        solver.set_relative_tolerance(float(self.proto["rtol"]))
        solver.set_absolute_tolerance(float(self.proto["atol"]))
        solver.set_max_steps(int(self.proto.get("max_steps", 100000)))
        solver.set_sensitivity_order(ss.SensitivityOrder.first)
        solver.set_sensitivity_method(method)
        return solver

    def run(self):
        from amici.sim.sundials.petab.v1 import simulate_petab

        ss = self._ss
        method = ss.SensitivityMethod.adjoint if self.adjoint else ss.SensitivityMethod.forward
        res = simulate_petab(self.problem, amici_model=self.model, solver=self._solver(method),
                             simulation_conditions=self.conditions, edatas=self.edatas,
                             parameter_mapping=self.mapping, scaled_parameters=False)
        bad = [(i, int(r.status)) for i, r in enumerate(res["rdatas"]) if r.status != 0]
        if bad:
            raise RuntimeError(f"AMICI simulation failed (condition, status): {bad[:5]}")
        return res

    def tensors(self, res) -> tuple[list[np.ndarray], list[np.ndarray]]:
        X, S = [], []
        params = self.prep["params"]
        scales = [self.prep["petab"].scales.get(p, "lin") for p in params]
        values = self.prep["values"]
        for k, c in zip(self.order, self.prep["petab"].conditions):
            rd = res["rdatas"][k]
            ts = np.asarray(rd.ts, dtype=float)
            it = np.searchsorted(ts, c.times)
            x = np.asarray(rd.x, dtype=float)[it][:, self.idx]
            sx = np.asarray(rd.sx, dtype=float)  # (nt, len(plist), nx)
            cols = self._plist_columns(k, sx.shape[1])
            s = np.full((len(it), len(self.idx), len(params)), np.nan)
            s_cov = np.transpose(sx[it][:, cols, :], (0, 2, 1))[:, self.idx, :]
            s[:, :, self.covered] = _unscale(
                s_cov, values[self.covered], [sc for sc, cv in zip(scales, self.covered) if cv])
            if not (np.all(np.isfinite(x)) and np.all(np.isfinite(s[:, :, self.covered]))):
                raise RuntimeError("AMICI returned non-finite values with a success status")
            X.append(x)
            S.append(s)
        return X, S

    def _plist_columns(self, k: int, width: int) -> list[int]:
        """The columns of condition ``k``'s ``sx``/``sy`` that hold the covered
        parameters: AMICI's PEtab simulation differentiates, per condition, only the
        model parameters its ``ExpData.plist`` names (those mapped to estimated
        PEtab parameters there)."""
        plist = [int(i) for i in self.edatas[k].plist]
        if len(plist) != width:
            raise RuntimeError(f"AMICI's sensitivities have {width} parameters, its plist "
                               f"{len(plist)}")
        out = []
        for f in self.pidx:
            if f not in plist:
                raise RuntimeError(f"AMICI does not differentiate {self.free[f]} in its "
                                   f"condition {k}")
            out.append(plist.index(f))
        return out

    def observables(self, res) -> tuple[np.ndarray, np.ndarray]:
        """AMICI's own observable value and sensitivities at every measurement row."""
        prep = self.prep
        rows = prep["petab"].rows
        y = np.zeros(len(rows))
        sy = np.full((len(rows), len(prep["params"])), np.nan)
        scales = [prep["petab"].scales.get(p, "lin") for p, c in
                  zip(prep["params"], self.covered) if c]
        for j, row in enumerate(rows):
            rd = res["rdatas"][self.order[row.condition]]
            ts = np.asarray(rd.ts, dtype=float)
            it = int(np.searchsorted(ts, row.time))
            io = self.obs_ids.index(row.observable)
            y[j] = float(np.asarray(rd.y)[it, io])
            sy_all = np.asarray(rd.sy, dtype=float)
            s = sy_all[it, self._plist_columns(self.order[row.condition], sy_all.shape[1]), io]
            sy[j, self.covered] = _unscale(s, prep["values"][self.covered], scales)
        return y, sy

    def likelihood(self, res) -> dict[str, Any]:
        sllh = res.get("sllh") or {}
        grad = np.array([float(sllh.get(p, np.nan)) for p in self.prep["params"]])
        return {"llh": float(res["llh"]), "gradient": grad}

    def config(self) -> dict[str, Any]:
        return {
            "importer": "amici.importers.petab.v1 (petab " + str(_petab_identity()) + ")",
            "version": env.amici_identity().get("version"),
            "integrator": "CVODES BDF, forward sensitivities" if not self.adjoint
            else "CVODES BDF, adjoint sensitivities",
            "sensitivities": "on the PEtab parameter scale, put back on a linear one",
            "not_differentiated": self.uncovered,
            "load_seconds": self.load_seconds,
            "codegen_seconds": self.codegen_seconds,
        }


class RoadRunnerPetab:
    """libRoadRunner applying each condition as the harness reads it: the run's own
    state, every parameter of the condition set (``resetAll`` first, so nothing of
    the last condition stays), ``reset()`` (initial assignments follow the
    parameters), the condition's species, then a run from t = 0. A
    pre-equilibration is integrated to ``preeq_max_time`` and its largest scaled
    rate of change recorded. Species are selected by SBML semantics (an amount for
    a species with ``hasOnlySubstanceUnits``, a concentration otherwise)."""

    def __init__(self, prep: dict[str, Any], *, rtol: float, atol: float):
        roadrunner = env.import_roadrunner()

        self.prep = prep
        info = prep["petab"].model
        t0 = time.perf_counter()
        self.rr = roadrunner.RoadRunner(str(prep["files"]["sbml"]))
        self.load_seconds = time.perf_counter() - t0
        self.rr.setIntegrator("cvode")
        integ = self.rr.getIntegrator()
        integ.setValue("relative_tolerance", float(rtol))
        integ.setValue("absolute_tolerance", float(atol))
        integ.setValue("maximum_num_steps", 1_000_000)
        self.sel = [s if info["species"][s]["hosu"] else f"[{s}]" for s in prep["states"]]
        self.rr.timeCourseSelections = ["time"] + self.sel
        self.preeq_residual: list[float] = []

    def _set(self, params: dict[str, float], species: dict[str, float]) -> None:
        info = self.prep["petab"].model
        for p, v in params.items():
            self.rr[p] = float(v)
        for s, v in species.items():
            self.rr[s if info["species"][s]["hosu"] else f"[{s}]"] = float(v)

    def solve(self, cond, over: dict[str, float] | None = None) -> np.ndarray:
        over = over or {}
        rr = self.rr
        rr.resetAll()
        if cond.preeq:
            self._set({**cond.preeq_parameters, **over}, {})
            rr.reset()
            self._set({}, cond.preeq_species)
            rr.simulate(0.0, float(self.prep["proto"]["preeq_max_time"]), 2)
            try:  # a diagnostic: how far from steady the integration ended
                rates = np.abs(np.asarray(rr.model.getFloatingSpeciesConcentrationRates(),
                                          dtype=float))
                state = np.abs(np.asarray(rr.model.getFloatingSpeciesConcentrations(),
                                          dtype=float))
                self.preeq_residual.append(float(np.max(rates / np.maximum(state, 1e-12)))
                                           if rates.size else 0.0)
            except Exception:  # noqa: BLE001 - not part of the result
                self.preeq_residual.append(float("nan"))
            self._set({**cond.parameters, **over}, cond.species)
        else:
            self._set({**cond.parameters, **over}, {})
            rr.reset()
            self._set({}, cond.species)
        sample, rows = _sample_times(cond.times)
        out = np.asarray(rr.simulate(times=sample), dtype=float)
        return out[rows, 1:]

    def config(self) -> dict[str, Any]:
        out = {"integrator": "cvode", "version": env.roadrunner_identity().get("version"),
               "output": "SBML semantics (amount if hasOnlySubstanceUnits, else concentration)",
               "load_seconds": self.load_seconds}
        if self.preeq_residual:
            out["preeq_max_scaled_rate"] = max(self.preeq_residual)
        return out


class FdRoadRunner:
    """Central finite differences of libRoadRunner's solves at steps ``h`` and
    ``h/2`` (``bench.sensitivity``), per condition, at ``fd_rtol``/``fd_atol``."""

    NAME = "fd_roadrunner"

    def __init__(self, prep: dict[str, Any]):
        proto = prep["proto"]
        self.prep = prep
        self.engine = RoadRunnerPetab(prep, rtol=float(proto["fd_rtol"]),
                                      atol=float(proto["fd_atol"]))
        self.h = float(proto["fd_step"])
        self.covered = np.ones(len(prep["params"]), bool)

    def run(self) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        prep = self.prep
        X, fine, coarse = [], [], []
        for cond in prep["petab"].conditions:
            x0 = self.engine.solve(cond)
            sf = np.zeros((*x0.shape, len(prep["params"])))
            sc = np.zeros_like(sf)
            for k, (p, v) in enumerate(zip(prep["params"], prep["values"])):
                d = {}
                for step in (self.h, self.h / 2):
                    up = self.engine.solve(cond, {p: v * (1 + step)})
                    dn = self.engine.solve(cond, {p: v * (1 - step)})
                    d[step] = (up - dn) / (2 * step * v)
                sc[:, :, k], sf[:, :, k] = d[self.h], d[self.h / 2]
            X.append(x0)
            fine.append(sf)
            coarse.append(sc)
        return X, fine, coarse

    def config(self) -> dict[str, Any]:
        p = self.prep["proto"]
        return {**self.engine.config(), "method": "central differences at h and h/2",
                "fd_step": self.h, "fd_rtol": p["fd_rtol"], "fd_atol": p["fd_atol"]}


# --------------------------------------------------------------------------- #
# Outputs and verdicts
# --------------------------------------------------------------------------- #
def output_tensors(prep: dict[str, Any], X: list[np.ndarray], S: list[np.ndarray],
                   y_rows: np.ndarray, sy_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(sig, y)``: the semi-scaled sensitivities ``(n_T, n_out, n_p)`` and values
    ``(n_T, n_out)`` of the states then the observables, the conditions end to end;
    an observable's cell holds its measurement row's value, zero where no row."""
    Xc, Sc = np.concatenate(X), np.concatenate(S)
    n_T, n_x = Xc.shape
    n_o, n_p = len(prep["observables"]), len(prep["params"])
    yo, so = np.zeros((n_T, n_o)), np.zeros((n_T, n_o, n_p))
    for j, t, o in prep["cells"]:
        yo[t, o] = y_rows[j]
        so[t, o] = sy_rows[j]
    sig = sv.scaled(np.concatenate([Sc, so], axis=1), prep["scales"])
    return sig, np.concatenate([Xc, yo], axis=1)


class Run:
    """One engine's outputs, as the verdict reads them."""

    def __init__(self, sig: np.ndarray, y: np.ndarray, covered: np.ndarray,
                 estimate: np.ndarray | None = None):
        self.sig, self.y = sig, y
        self.y_peak = np.max(np.where(np.isfinite(y), np.abs(y), 0.0), axis=0)
        self.covered = np.asarray(covered, bool)
        self.estimate = estimate


def judge(a: Run, b: Run, atol: float) -> sv.Comparison:
    y_peak = np.fmax(a.y_peak, b.y_peak)
    judged = np.broadcast_to(a.covered & b.covered, a.sig.shape[1:]).copy()
    for this, other in ((a, b), (b, a)):
        if this.estimate is not None:
            judged &= sv.fd_resolved(this.estimate, other.sig, this.sig, y_peak, atol=atol,
                                     verdict=VERDICT)
    return sv.compare(a.sig, b.sig, y_peak, atol=atol, verdict=VERDICT, judged=judged)


def output_names(prep: dict[str, Any]) -> list[str]:
    return [*prep["states"], *(f"observable:{o}" for o in prep["observables"])]


def time_labels(prep: dict[str, Any]) -> np.ndarray:
    """The outputs' time axis as times (each condition's own, end to end)."""
    return np.concatenate(prep["times"])


def summary(c: sv.Comparison, prep: dict[str, Any], a: Run, b: Run) -> dict[str, Any]:
    out = c.summary(output_names(prep), prep["params"], time_labels(prep),
                    n_species=len(prep["states"]))
    if "worst" in out:
        t_index = int(c.worst_time[tuple(np.unravel_index(
            int(np.argmax(np.where(c.judged, c.errors, -1.0))), c.errors.shape))])
        k = int(np.searchsorted(prep["offsets"], t_index, side="right") - 1)
        out["worst"]["condition"] = prep["petab"].conditions[k].id
    n_x = len(prep["states"])
    out["trajectory_max_err_over_peak"] = sens.trajectory_error(
        a.y[:, :n_x], b.y[:, :n_x], float(prep["proto"]["atol"]))
    return out


def verdicts(prep: dict[str, Any], bn: Run, refs: dict[str, Run]
             ) -> tuple[dict[str, Any], dict[str, sv.Comparison]]:
    """BNGsim against each reference, with the refutation rule of ``sens.verdicts``."""
    atol = float(prep["proto"]["atol"])
    vs_bn = {name: judge(bn, run, atol) for name, run in refs.items()}
    out = {name: summary(c, prep, bn, refs[name]) for name, c in vs_bn.items()}
    for name, c in vs_bn.items():
        if c.passed or not c.judged.any():
            continue
        vouchers, by = [], []
        for other, c_other in vs_bn.items():
            if other == name:
                continue
            between = judge(refs[other], refs[name], atol)
            vouchers.append((c_other.judged & ~c_other.failing, between.failing))
            if (c_other.judged & ~c_other.failing & between.failing & c.failing).any():
                by.append(other)
        if sv.refuted(c.failing, vouchers):
            out[name]["refuted_by"] = sorted(by)
    return out, vs_bn


def coverage(prep: dict[str, Any], bn: Run, vs_bn: dict[str, sv.Comparison],
             verdict_json: dict[str, Any]) -> dict[str, Any]:
    judged = np.zeros(bn.sig.shape[1:], bool)
    for name, c in vs_bn.items():
        if c.passed and "refuted_by" not in verdict_json[name]:
            judged |= c.judged
    floor = sv.column_scale(np.zeros_like(bn.sig), np.zeros_like(bn.sig), bn.y_peak,
                            atol=float(prep["proto"]["atol"]), verdict=VERDICT)
    significant = sv.peak(bn.sig) >= floor
    missing = significant & ~judged
    return {"n_columns": int(judged.size), "n_significant": int(significant.sum()),
            "n_judged": int((significant & judged).sum()),
            "unjudged": [p for k, p in enumerate(prep["params"]) if missing[:, k].any()]}


def llh_check(bn: float, ref: float | None) -> dict[str, Any]:
    if ref is None or not np.isfinite(ref):
        return {"passed": None, "reason": "no reference value"}
    diff = abs(bn - ref)
    allowed = VERDICT["llh_atol"] + VERDICT["llh_rtol"] * abs(ref)
    return {"passed": bool(diff <= allowed), "bngsim": bn, "reference": ref,
            "abs_difference": diff, "allowed": allowed}


def gradient_check(bn: np.ndarray, ref: np.ndarray, scales: np.ndarray,
                   params: list[str], magnitude: np.ndarray | None = None) -> dict[str, Any]:
    """BNGsim's semi-scaled gradient ``p dL/dp`` against a reference's, as a fraction
    of the gradient's own size. At the nominal parameters -- the authors' fit --
    the gradient is near zero: a sum of the rows' contributions that cancel. So the
    size it is held to is the larger of its largest component and the largest
    sum of the absolute row contributions (``magnitude``, the likelihood's
    ``gradient_magnitude``): an error in the derivative of one row shows at that
    scale, the integrator's error in a cancelled sum does not."""
    a, b = np.asarray(ref, float) * scales, np.asarray(bn, float) * scales
    ok = np.isfinite(a)
    if not ok.any():
        return {"passed": None, "reason": "the reference differentiated no parameter"}
    peak = float(np.max(np.abs(b[ok])))
    size = peak
    if magnitude is not None:
        size = max(size, float(np.max(np.abs(np.asarray(magnitude, float) * scales))))
    if size == 0.0 or not np.isfinite(size):
        return {"passed": bool(np.allclose(a[ok], b[ok])), "max_rel_difference": None}
    rel = np.abs(a - b) / size
    k = int(np.nanargmax(np.where(ok, rel, -1.0)))
    err = float(rel[k])
    return {"passed": bool(err <= VERDICT["gradient_tol"]), "max_rel_difference": err,
            "worst_parameter": params[k], "tol": VERDICT["gradient_tol"],
            "n_parameters": int(ok.sum()), "semi_scaled_peak": peak, "semi_scaled_size": size}


# --------------------------------------------------------------------------- #
# The correctness job
# --------------------------------------------------------------------------- #
def _progress(msg: str) -> None:
    sens.progress(CAPABILITY, msg)


def _pack_rows(y: np.ndarray, sy: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(y, float)[:, None], np.asarray(sy, float)], axis=1)


def _cold_bngsim(prep: dict[str, Any], name: str):
    def fn(stage):
        t0 = time.perf_counter()
        stage("load + codegen (sensitivity RHS)")
        eng = BngsimPetab(prep, **CONFIGURATIONS[name])
        stage("first evaluation (every condition)")
        t1 = time.perf_counter()
        res = eng.run()
        solve = time.perf_counter() - t1
        X, S = eng.tensors(res)
        out = prep["output_model"].evaluate(X, S)
        lik = prep["output_model"].likelihood(out)
        sig, y = output_tensors(prep, X, S, out["y"], out["sy"])
        config = {**eng.config(), "solve_seconds": solve,
                  "cold_seconds": time.perf_counter() - t0,
                  "llh": lik["llh"], "llh_without_log_jacobian": lik["llh_without_log_jacobian"],
                  "gradient": lik["gradient"].tolist(),
                  "gradient_magnitude": lik["gradient_magnitude"].tolist()}
        return np.concatenate([sig, y[:, :, None]], axis=2), config
    return fn


def _cold_amici(prep: dict[str, Any], workdir: Path):
    def fn(stage):
        t0 = time.perf_counter()
        stage("PEtab import + sensitivity code generation + C++ compile")
        eng = AmiciPetab(prep, workdir)
        stage("first evaluation (simulate_petab)")
        t1 = time.perf_counter()
        res = eng.run()
        solve = time.perf_counter() - t1
        X, S = eng.tensors(res)
        y_rows, sy_rows = eng.observables(res)
        sig, y = output_tensors(prep, X, S, y_rows, sy_rows)
        # The harness's observables on AMICI's states: a check of the harness's
        # application of the observable table against AMICI's (recorded).
        mine = prep["output_model"].evaluate(X, [np.nan_to_num(s) for s in S])
        peak = np.maximum(np.abs(y_rows), 1e-300)
        lik = eng.likelihood(res)
        config = {**eng.config(), "covered": [bool(c) for c in eng.covered],
                  "solve_seconds": solve, "cold_seconds": time.perf_counter() - t0,
                  "llh": lik["llh"], "gradient": lik["gradient"].tolist(),
                  "harness_observables_vs_amici_max_rel": float(np.max(
                      np.abs(mine["y"] - y_rows) / np.maximum(peak, np.max(peak) * 1e-12)))}
        return np.concatenate([sig, y[:, :, None]], axis=2), config
    return fn


def _cold_fd(prep: dict[str, Any]):
    def fn(stage):
        t0 = time.perf_counter()
        stage("libRoadRunner load")
        eng = FdRoadRunner(prep)
        stage("perturbed solves")
        X, fine, coarse = eng.run()
        om = prep["output_model"]
        of, oc = om.evaluate(X, fine), om.evaluate(X, coarse)
        sig_f, y = output_tensors(prep, X, fine, of["y"], of["sy"])
        sig_c, _ = output_tensors(prep, X, coarse, oc["y"], oc["sy"])
        lik = om.likelihood(of)
        config = {**eng.config(), "covered": [True] * len(prep["params"]),
                  "cold_seconds": time.perf_counter() - t0, "llh": lik["llh"],
                  "gradient": lik["gradient"].tolist()}
        return np.concatenate([sig_f, sig_c, y[:, :, None]], axis=2), config
    return fn


def _run_from(prep: dict[str, Any], packed: np.ndarray, config: dict[str, Any], *,
              fd: bool = False) -> Run:
    n_p = len(prep["params"])
    y = packed[:, :, -1]
    covered = np.asarray(config.get("covered", [True] * n_p), bool)
    if fd:
        fine, coarse = packed[:, :, :n_p], packed[:, :, n_p:2 * n_p]
        return Run(fine, y, covered, sv.fd_estimate(coarse, fine))
    return Run(packed[:, :, :n_p], y, covered)


def run_correctness(job: Job, problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    try:
        prep = prepare(problem, ctx)
    except Exception as exc:  # noqa: BLE001 - a bad test input, a labelled row
        return {"outcome": "BAD_TEST", "error": f"{type(exc).__name__}: {exc}"[:600]}
    info = prep["info"]
    workdir = Path(ctx["workdir"])
    budget = float(ctx.get("engine_budget_seconds") or engine_budget_seconds(problem))
    info["engine_budget_seconds"] = budget

    def isolated(name: str, fn) -> dict[str, Any]:
        _progress(f"{name}: start (budget {budget:g} s; {threads()} threads in the worker)")
        out = run_isolated(fn, budget, workdir / "isolated" / name)
        _progress(f"{name}: {out.get('error') or 'done'} after {out['seconds']:.1f} s")
        return out

    bn_runs: dict[str, Run] = {}
    bn_errors: dict[str, str] = {}
    configs: dict[str, Any] = {}
    for name in CONFIGURATIONS:
        out = isolated(name, _cold_bngsim(prep, name))
        if "error" in out:
            bn_errors[name] = out["error"]
        else:
            configs[name] = out["config"]
            bn_runs[name] = _run_from(prep, out["trajectory"], out["config"])
    info["bngsim"] = configs.get(DEFAULT_CONFIGURATION, {})
    bn_error = bn_errors.get(DEFAULT_CONFIGURATION)

    references: dict[str, Any] = {}
    ref_runs: dict[str, Run] = {}
    ref_errors: dict[str, str] = {}
    builders = {"amici": lambda: _cold_amici(prep, workdir / "amici"),
                "fd_roadrunner": lambda: _cold_fd(prep)}
    for name in REFERENCES:
        out = isolated(name, builders[name]())
        if "error" in out:
            ref_errors[name] = out["error"]
            continue
        references[name] = out["config"]
        try:
            ref_runs[name] = _run_from(prep, out["trajectory"], out["config"],
                                       fd=name == FdRoadRunner.NAME)
        except Exception as exc:  # noqa: BLE001 - an unreadable result is a failed reference
            ref_errors[name] = f"{type(exc).__name__}: {exc}"[:400]
    info["references"] = references

    verdict_json: dict[str, Any] = {}
    consistent = True
    checks: dict[str, Any] = {}
    if bn_error is None:
        bn = bn_runs[DEFAULT_CONFIGURATION]
        verdict_json, vs_bn = verdicts(prep, bn, ref_runs)
        atol = float(prep["proto"]["atol"])
        others = {}
        for name, run in bn_runs.items():
            if name != DEFAULT_CONFIGURATION:
                others[name] = summary(judge(run, bn, atol), prep, run, bn)
                consistent &= bool(others[name]["passed"])
        info["configurations"] = others
        info["coverage"] = coverage(prep, bn, vs_bn, verdict_json)
        bcfg = configs[DEFAULT_CONFIGURATION]
        g_bn = np.asarray(bcfg["gradient"], float)
        if "amici" in references:
            checks["llh_vs_amici"] = llh_check(bcfg["llh"], references["amici"]["llh"])
            checks["gradient_vs_amici"] = gradient_check(
                g_bn, np.asarray(references["amici"]["gradient"], float), prep["scales"],
                prep["params"], np.asarray(bcfg["gradient_magnitude"], float))
        if "fd_roadrunner" in references:
            checks["llh_vs_fd_roadrunner_nominal"] = llh_check(
                bcfg["llh"], references["fd_roadrunner"]["llh"])
        ref_llh = prep["proto"].get("llh_reference")
        if ref_llh is not None:
            checks["llh_vs_amici_benchmark_reference"] = llh_check(bcfg["llh"],
                                                                   float(ref_llh))
        info["problem_checks"] = checks
    elif len(ref_runs) > 1:
        a, b = list(ref_runs)
        info["references_agreement"] = {
            f"{a}_vs_{b}": summary(judge(ref_runs[a], ref_runs[b],
                                         float(prep["proto"]["atol"])), prep,
                                   ref_runs[a], ref_runs[b])}
    if bn_error is None:
        bn_error = next((f"{name}: {err}" for name, err in bn_errors.items()), None)
    unjudged = info.get("coverage", {}).get("unjudged", [])
    # A likelihood or gradient that disagrees with AMICI is a DIFF unless libRoadRunner's
    # solve (its likelihood at nominal) sides with BNGsim against AMICI.
    llh_am = checks.get("llh_vs_amici", {})
    if llh_am.get("passed") is False:
        rr_side = checks.get("llh_vs_fd_roadrunner_nominal", {}).get("passed")
        if rr_side:
            llh_am["refuted_by"] = ["fd_roadrunner"]
        else:
            consistent = False
    if checks.get("gradient_vs_amici", {}).get("passed") is False and \
            "refuted_by" not in verdict_json.get("amici", {}):
        consistent = False
    result = sens.sens_outcome(bn_error, ref_errors, len(REFERENCES), verdict_json, consistent,
                               unjudged)
    if result == "PASS" and llh_am.get("refuted_by"):
        result = "REFERENCE_FAILED"
    if bn_error and result == "EXCEPTION" and \
            any(part.strip() == "TIMEOUT" for part in bn_error.split(":")[:2]):
        result = "TIMEOUT"
    if bn_error and result in ("EXCEPTION", "BAD_TEST") and \
            any(part.strip() == "UNSUPPORTED" for part in bn_error.split(":")[:2]):
        result = "UNSUPPORTED"
    rec: dict[str, Any] = {"outcome": result, "measurements": info}
    if verdict_json:
        rec["verdicts"] = verdict_json
    if bn_error:
        rec["bngsim_error"] = bn_error
    for name, err in ref_errors.items():
        rec[f"{name}_error"] = err
    rec["not_applicable"] = dict(NOT_WIRED)
    return rec


# --------------------------------------------------------------------------- #
# The timing job
# --------------------------------------------------------------------------- #
class _Timed:
    """An engine's evaluation as the interleave calls it, and what each repetition
    is checked on (its semi-scaled sensitivities, or its trajectory)."""

    def __init__(self, run, capture):
        self.run, self.capture = run, capture


def _role(name: str) -> str:
    if name in CONFIGURATIONS:
        return (f"warm, in-process (system under test, "
                f"{CONFIGURATIONS[name]['sensitivity_method']} corrector, every condition)")
    if name == "bngsim_ode":
        return "warm, in-process (plain solve of every condition: the unit of an FD gradient)"
    if name == "roadrunner":
        return "warm, in-process (plain solve of every condition: the unit of an FD gradient)"
    if name == "amici":
        return "warm, in-process (simulate_petab with forward sensitivities, every condition)"
    if name == "amici_adjoint":
        return ("warm, in-process (simulate_petab with adjoint sensitivities: the PEtab "
                "log-likelihood gradient; timing only)")
    return engine_role(name)


def run_timing(job: Job, problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    try:
        prep = prepare(problem, ctx)
    except Exception as exc:  # noqa: BLE001
        return {"outcome": "BAD_TEST", "error": f"{type(exc).__name__}: {exc}"[:600]}
    info = prep["info"]
    workdir = Path(ctx["workdir"])
    scales = prep["scales"]

    def sens_capture(eng):
        def capture(raw):
            X, S = eng.tensors(raw)
            sig = sv.scaled(np.concatenate(S), scales)
            return np.nan_to_num(sig.reshape(sig.shape[0], -1), nan=0.0)
        return capture

    t0 = time.perf_counter()
    first_engine = BngsimPetab(prep, **CONFIGURATIONS[DEFAULT_CONFIGURATION])
    first = first_engine.run()
    info["bngsim_cold_seconds"] = time.perf_counter() - t0
    anchor = sens_capture(first_engine)(first)
    info["bngsim"] = first_engine.config()
    timed: dict[str, _Timed] = {DEFAULT_CONFIGURATION: _Timed(first_engine.run,
                                                              sens_capture(first_engine))}
    configs: dict[str, Any] = {}
    for name, cfg in CONFIGURATIONS.items():
        if name == DEFAULT_CONFIGURATION:
            continue
        eng = BngsimPetab(prep, **cfg)
        eng.tensors(eng.run())
        timed[name] = _Timed(eng.run, sens_capture(eng))
        configs[name] = eng.config()
    plain = BngsimPetab(prep, sensitivity_method=None)
    plain.run()
    timed["bngsim_ode"] = _Timed(plain.run, lambda raw, e=plain: np.concatenate(e.tensors(raw)[0]))
    configs["bngsim_ode"] = plain.config()
    info["configurations"] = configs

    references: dict[str, Any] = {}
    ref_errors: dict[str, str] = {}
    timed_out = correctness_timeouts(job, ctx, REFERENCES)
    try:
        rr = RoadRunnerPetab(prep, rtol=float(prep["proto"]["rtol"]),
                             atol=float(prep["proto"]["atol"]))
        conds = prep["petab"].conditions

        def rr_run(rr=rr, conds=conds):
            return [rr.solve(c) for c in conds]
        rr_run()
        timed["roadrunner"] = _Timed(rr_run, lambda raw: np.concatenate(raw))
        references["roadrunner"] = rr.config()
    except Exception as exc:  # noqa: BLE001
        ref_errors["roadrunner"] = f"{type(exc).__name__}: {exc}"[:400]
    if "amici" in timed_out:
        ref_errors["amici"] = f"TIMEOUT in the correctness job, so not timed ({timed_out['amici']})"
    else:
        _progress("amici: build + trial run")
        try:
            am = AmiciPetab(prep, workdir / "amici")
            am.tensors(am.run())
            timed["amici"] = _Timed(am.run, sens_capture(am))
            references["amici"] = am.config()
            try:
                adj = AmiciPetab.__new__(AmiciPetab)
                adj.__dict__.update(am.__dict__)
                adj.adjoint = True
                g = adj.run()
                timed["amici_adjoint"] = _Timed(
                    adj.run, lambda raw: np.array([[float(raw["llh"])]]))
                references["amici_adjoint"] = {"shared_build": "amici",
                                               "llh": float(g["llh"])}
            except Exception as exc:  # noqa: BLE001
                ref_errors["amici_adjoint"] = f"{type(exc).__name__}: {exc}"[:400]
        except Exception as exc:  # noqa: BLE001
            ref_errors["amici"] = f"{type(exc).__name__}: {exc}"[:400]
    info["references"] = references

    _progress(f"interleaved timing of {', '.join(timed)}")
    runs = {name: t.run for name, t in timed.items()}
    captures = {name: t.capture for name, t in timed.items()}
    reps = round_robin_interleaved(runs, captures=captures, **job.definition["timing"])
    outputs = {name: reps["engines"][name].pop("outputs") for name in runs}
    repro = {DEFAULT_CONFIGURATION: reproducibility(anchor, outputs[DEFAULT_CONFIGURATION],
                                                    REPRO_TOL)}
    for name in runs:
        if name != DEFAULT_CONFIGURATION:
            outs = outputs[name]
            repro[name] = reproducibility(outs[0], outs[1:], REPRO_TOL)
    reproducible = all(r["passed"] for r in repro.values())
    times_ = absolute_time_stats({name: reps["engines"][name]["seconds"] for name in runs})
    n = len(prep["params"])
    baselines = {name: {"forward_seconds": (n + 1) * times_[name]["median_seconds"],
                        "central_seconds": 2 * n * times_[name]["median_seconds"],
                        "parameters": n}
                 for name in ("bngsim_ode", "roadrunner") if name in times_}
    rec: dict[str, Any] = {
        "outcome": "PASS" if reproducible else "NOT_REPRODUCIBLE",
        "measurements": {**info, "engine_roles": {name: _role(name) for name in runs},
                         "repetitions": reps, "reproducibility": repro, "times": times_,
                         "fd_baselines": baselines},
    }
    for name, err in ref_errors.items():
        rec[f"{name}_error"] = err
    rec["not_applicable"] = dict(NOT_WIRED)
    return rec


def run(job: Job, problem: Problem, ctx: dict[str, Any]) -> dict[str, Any]:
    if job.kind == CORRECTNESS:
        return run_correctness(job, problem, ctx)
    if job.kind == TIMING:
        return run_timing(job, problem, ctx)
    raise ValueError(f"unknown job kind {job.kind!r}")


__all__ = ["PetabError", "jobs", "run", "prepare", "VERDICT", "engines_id"]
