"""A PEtab (v1) problem as the harness applies it: the sens-sbml suite's input.

A PEtab problem is an SBML model plus four tables: the **parameter** table (every
parameter's nominal value and whether it is estimated), the **condition** table
(what each experiment sets: model parameters and species' initial values), the
**measurement** table (which observable was measured when, under which simulation
condition and, where there is one, after which pre-equilibration condition, with
the values that fill the observable's and the noise model's placeholders) and the
**observable** table (each observable's formula over the model, its noise
formula, transformation and noise distribution). The paper's harness read none of
the tables; this module reads all four, with pandas, libsbml and sympy only -- no
simulator and not the ``petab`` library, so the application of the problem that
BNGsim and the finite differences run is the harness's own, and AMICI's PEtab
importer (which reads the problem through ``petab``) is a second, independent one.

What a simulation condition is, as every engine runs it (PEtab v1 semantics):

- **Parameters.** The SBML's own values, overridden by the parameter table's
  nominal value (linear scale) of every SBML parameter the table lists, estimated
  or not, overridden by the condition table's value for the condition (a number,
  or a parameter-table id standing for that parameter's nominal value; an empty
  cell keeps the value below it). ``values(cid)`` gives every parameter any
  condition or the table sets, so an engine reused across conditions leaves none
  of one condition's values in force in the next.
- **Species.** The SBML's initial value -- initial assignments evaluated at the
  condition's parameter values -- unless the condition table gives one. With a
  pre-equilibration, the model is run to its steady state under the
  pre-equilibration condition first, then the simulation condition's parameters
  take effect and the state carries over, except for a species the simulation
  condition sets, which starts again at that value.
- **Outputs.** Each condition is reported at its measurement times (sorted,
  unique), from t = 0.

The **sensitivity set** is the estimated parameters that are SBML global
parameters and enter the ODE as the model's own: not a condition-table column
(that is the experiment's, overridden per condition) and not one the observable
or noise model only (noise and scaling parameters never reach the state). Every
estimated parameter left out is recorded with its reason.

**Observables** are each measurement row's formula evaluated on an engine's state
at the row's time: species (by SBML semantics: an amount for a species with
``hasOnlySubstanceUnits``, a concentration otherwise), the condition's parameter
values, the row's placeholder values, parameter-table values and compartment
sizes, with every assignment-rule target expanded into its rule. Their
sensitivities follow by the chain rule, ``dy/dp = dh/dx S + dh/dp``, the same
arithmetic for every engine. The **log-likelihood** of the measurements is
PEtab's (``petab.v1.calculate.calculate_single_llh``), and also in AMICI's
convention, which leaves out the data-only Jacobian term of a log-transformed
observable (``-log(m)``, ``-log(m ln 10)``), so the two are compared like for like;
its gradient over the sensitivity set is the chain rule through the observable
and the noise scale.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# PEtab's table roles, as a spec's ``petab:`` block names them.
TABLES = ("yaml", "conditions", "measurements", "observables", "parameters")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Functions an observable or noise formula, or an SBML rule it reads, may call.
_FUNCTIONS = {"exp", "log", "ln", "log10", "log2", "sqrt", "abs", "sin", "cos", "tan",
              "pow", "min", "max", "piecewise", "sign", "floor", "ceil", "arcsinh",
              "and", "or", "not", "lt", "le", "gt", "ge", "eq", "ne", "true", "false"}
_PLACEHOLDER_RE = re.compile(r"^(observableParameter|noiseParameter)(\d+)_(.+)$")


class PetabError(ValueError):
    """A PEtab problem the harness cannot apply as written."""


def _read_tsv(path: Path):
    import pandas as pd

    # utf-8-sig: a table may start with a byte-order mark (Giordano's does).
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df.apply(lambda col: col.str.strip())


def _number(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


@dataclass
class Condition:
    """One simulation condition: a (pre-equilibration, simulation) pair of the
    measurement table, with everything an engine sets for it."""

    sim: str
    preeq: str | None
    times: np.ndarray  # sorted unique measurement times
    parameters: dict[str, float]  # every set parameter's value in the simulation
    preeq_parameters: dict[str, float] | None
    species: dict[str, float]  # simulation-condition initial values (SBML units)
    preeq_species: dict[str, float]
    rows: list[int] = field(default_factory=list)  # measurement rows, table order

    @property
    def id(self) -> str:
        return f"{self.preeq}:{self.sim}" if self.preeq else self.sim


@dataclass
class Row:
    """One measurement row, resolved: its condition, time index, observable and
    placeholder values."""

    index: int
    condition: int
    time: float
    time_index: int  # into the condition's ``times``
    observable: str
    measurement: float
    observable_parameters: list[float]
    noise_parameters: list[float]


class PetabProblem:
    """The pinned SBML and tables of one PEtab problem (see the module docstring)."""

    def __init__(self, sbml: Path, conditions: Path, measurements: Path, observables: Path,
                 parameters: Path):
        self.sbml = Path(sbml)
        self.model = read_sbml(self.sbml)
        self.parameter_df = _read_tsv(parameters)
        self.condition_df = _read_tsv(conditions)
        self.measurement_df = _read_tsv(measurements)
        self.observable_df = _read_tsv(observables)
        for df, col, what in ((self.parameter_df, "parameterId", "parameter"),
                              (self.condition_df, "conditionId", "condition"),
                              (self.observable_df, "observableId", "observable")):
            if col not in df.columns:
                raise PetabError(f"the {what} table has no {col} column")
        self.nominal = {}
        self.estimated: list[str] = []
        for _, r in self.parameter_df.iterrows():
            v = _number(r.get("nominalValue", ""))
            if v is None:
                raise PetabError(f"parameter {r['parameterId']}: no numeric nominalValue")
            self.nominal[r["parameterId"]] = v
            if str(r.get("estimate", "0")).strip() in ("1", "1.0", "True", "true"):
                self.estimated.append(r["parameterId"])
        self.scales = {r["parameterId"]: (r.get("parameterScale") or "lin")
                       for _, r in self.parameter_df.iterrows()}
        self.condition_columns = [c for c in self.condition_df.columns
                                  if c not in ("conditionId", "conditionName")]
        kinds = {}
        for c in self.condition_columns:
            if c in self.model["parameters"]:
                kinds[c] = "parameter"
            elif c in self.model["species"]:
                kinds[c] = "species"
            elif c in self.model["compartments"]:
                kinds[c] = "compartment"
            else:
                raise PetabError(f"condition table column {c!r} is not an SBML parameter, "
                                 "species or compartment")
        compartments = [c for c, k in kinds.items() if k == "compartment"]
        if compartments:
            raise PetabError(f"the condition table sets compartment sizes ({compartments}); "
                             "the harness does not apply those")
        self.condition_kinds = kinds
        self._conditions = {r["conditionId"]: r for _, r in self.condition_df.iterrows()}
        self.conditions = self._build_conditions()
        self.rows = self._build_rows()

    # ------------------------------------------------------------------ #
    # Parameters and conditions
    # ------------------------------------------------------------------ #
    def resolve(self, token: str, where: str) -> float:
        """A table cell's value: a number, or a parameter-table id's nominal value."""
        v = _number(token)
        if v is not None:
            return v
        if token in self.nominal:
            return self.nominal[token]
        raise PetabError(f"{where}: {token!r} is neither a number nor a parameter-table id")

    def set_parameters(self) -> list[str]:
        """Every SBML parameter the parameter table or the condition table sets."""
        sbml = self.model["parameters"]
        out = [p for p in self.parameter_df["parameterId"] if p in sbml]
        out += [c for c, k in self.condition_kinds.items() if k == "parameter" and c not in out]
        return out

    def values(self, cid: str) -> dict[str, float]:
        """The value of every set parameter (``set_parameters``) in condition ``cid``."""
        row = self._conditions.get(cid)
        if row is None:
            raise PetabError(f"no condition {cid!r} in the condition table")
        out = {}
        for p in self.set_parameters():
            cell = row.get(p, "") if p in self.condition_kinds else ""
            if cell not in ("", "nan", "NaN"):
                out[p] = self.resolve(cell, f"condition {cid}, {p}")
            elif p in self.nominal:
                out[p] = self.nominal[p]
            else:
                out[p] = float(self.model["parameters"][p])
        return out

    def species_overrides(self, cid: str) -> dict[str, float]:
        """The initial values condition ``cid`` sets, in SBML units (a species with
        ``hasOnlySubstanceUnits`` as an amount, any other as a concentration)."""
        row = self._conditions[cid]
        out = {}
        for c, k in self.condition_kinds.items():
            if k != "species":
                continue
            cell = row.get(c, "")
            if cell in ("", "nan", "NaN"):
                continue
            out[c] = self.resolve(cell, f"condition {cid}, {c}")
        return out

    def _build_conditions(self) -> list[Condition]:
        mdf = self.measurement_df
        has_pre = "preequilibrationConditionId" in mdf.columns
        out: list[Condition] = []
        index: dict[tuple[str | None, str], int] = {}
        for i, r in mdf.iterrows():
            pre = (r["preequilibrationConditionId"] or None) if has_pre else None
            key = (pre, r["simulationConditionId"])
            if key not in index:
                index[key] = len(out)
                sim = key[1]
                out.append(Condition(
                    sim=sim, preeq=pre, times=np.zeros(0),
                    parameters=self.values(sim),
                    preeq_parameters=self.values(pre) if pre else None,
                    species=self.species_overrides(sim),
                    preeq_species=self.species_overrides(pre) if pre else {}))
            out[index[key]].rows.append(int(i))
        for c in out:
            t = np.array([float(mdf.loc[i, "time"]) for i in c.rows])
            if not np.all(np.isfinite(t)):
                raise PetabError(f"condition {c.id}: a measurement at t = inf (a steady-state "
                                 "measurement); the suite measures time courses only")
            if (t < 0).any():
                raise PetabError(f"condition {c.id}: a measurement before t = 0")
            c.times = np.unique(t)
        return out

    def _build_rows(self) -> list[Row]:
        mdf = self.measurement_df
        rows: list[Row] = []
        for k, c in enumerate(self.conditions):
            for i in c.rows:
                r = mdf.loc[i]
                obs = r["observableId"]
                if obs not in set(self.observable_df["observableId"]):
                    raise PetabError(f"measurement row {i}: no observable {obs!r}")
                t = float(r["time"])

                def overrides(col: str) -> list[float]:
                    cell = r.get(col, "") if col in mdf.columns else ""
                    if cell in ("", "nan", "NaN"):
                        return []
                    return [self.resolve(x.strip(), f"measurement row {i}, {col}")
                            for x in cell.split(";")]

                rows.append(Row(index=int(i), condition=k, time=t,
                                time_index=int(np.searchsorted(c.times, t)), observable=obs,
                                measurement=float(r["measurement"]),
                                observable_parameters=overrides("observableParameters"),
                                noise_parameters=overrides("noiseParameters")))
        return rows

    # ------------------------------------------------------------------ #
    # The sensitivity set
    # ------------------------------------------------------------------ #
    def sensitivity_parameters(self) -> tuple[list[str], dict[str, str]]:
        """The estimated parameters that are the model's own SBML parameters, in
        parameter-table order, and every other estimated one with why it is not."""
        sbml = self.model["parameters"]
        chosen, excluded = [], {}
        for p in self.estimated:
            if p not in sbml:
                excluded[p] = ("not an SBML parameter (an observable, noise or condition-table "
                               "parameter: it never reaches the state)")
            elif p in self.condition_kinds:
                excluded[p] = "a condition-table column (the experiment's, set per condition)"
            elif p in self.model["rule_targets"]:
                excluded[p] = "set by an SBML rule"
            elif self.nominal[p] == 0.0:
                excluded[p] = "nominal value zero: no relative step"
            else:
                chosen.append(p)
        return chosen, excluded

    # ------------------------------------------------------------------ #
    # Observables, noise and the likelihood
    # ------------------------------------------------------------------ #
    def output_model(self, state_ids: list[str], params: list[str]) -> "OutputModel":
        return OutputModel(self, state_ids, params)

    def summary(self) -> dict[str, Any]:
        return {
            "conditions": len(self.conditions),
            "with_preequilibration": sum(1 for c in self.conditions if c.preeq),
            "measurements": len(self.rows),
            "observables": int(len(self.observable_df)),
            "estimated": len(self.estimated),
            "condition_columns": dict(self.condition_kinds),
            "measurement_times": int(sum(c.times.size for c in self.conditions)),
        }


def read_sbml(sbml: Path) -> dict[str, Any]:
    """What the harness reads from the SBML with libsbml: global parameters and
    their values, species (compartment, hasOnlySubstanceUnits, constant/boundary),
    compartments and their sizes, assignment rules (target -> formula) and the
    targets of every rule."""
    import libsbml

    doc = libsbml.readSBMLFromFile(str(sbml))
    model = doc.getModel()
    if model is None:
        raise PetabError(f"libsbml cannot read {Path(sbml).name}")
    comps = {c.getId(): {"size": c.getSize() if c.isSetSize() else 1.0,
                         "constant": bool(c.getConstant())}
             for c in model.getListOfCompartments()}
    params = {p.getId(): (p.getValue() if p.isSetValue() else float("nan"))
              for p in model.getListOfParameters()}
    species = {s.getId(): {"compartment": s.getCompartment(),
                           "hosu": bool(s.getHasOnlySubstanceUnits()),
                           "constant": bool(s.getConstant()),
                           "boundary": bool(s.getBoundaryCondition())}
               for s in model.getListOfSpecies()}
    assignment, targets = {}, set()
    for r in model.getListOfRules():
        targets.add(r.getVariable())
        if r.isAssignment():
            assignment[r.getVariable()] = libsbml.formulaToL3String(r.getMath())
    functions = {}
    for f in model.getListOfFunctionDefinitions():
        math_ = f.getMath()
        args = [math_.getChild(i).getName() for i in range(math_.getNumChildren() - 1)]
        functions[f.getId()] = (args, libsbml.formulaToL3String(math_.getChild(
            math_.getNumChildren() - 1)))
    return {"parameters": params, "species": species, "compartments": comps,
            "assignment_rules": assignment, "rule_targets": targets, "functions": functions,
            "events": model.getNumEvents()}


def state_species(model_info: dict[str, Any]) -> list[str]:
    """The species every engine integrates and the verdict compares: neither
    constant, nor a boundary species, nor set by a rule; SBML order."""
    return [s for s, d in model_info["species"].items()
            if not d["constant"] and not d["boundary"] and s not in model_info["rule_targets"]]


def _sympify(text: str, names: set[str]):
    """A PEtab / SBML L3 formula as a sympy expression, every identifier but a known
    function a plain symbol (so ``gamma`` or ``E`` is a model name, not sympy's)."""
    import sympy as sp

    expr = text.replace("^", "**").replace("&&", "&").replace("||", "|")
    local: dict[str, Any] = {}
    for name in set(_IDENT_RE.findall(expr)):
        if name in _FUNCTIONS:
            continue
        local[name] = sp.Symbol(name)
    local.update({"ln": sp.log, "log10": lambda x: sp.log(x, 10),
                  "log2": lambda x: sp.log(x, 2), "pow": sp.Pow, "abs": sp.Abs,
                  "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "min": sp.Min,
                  "max": sp.Max})
    try:
        return sp.parse_expr(expr, local_dict=local, evaluate=True)
    except Exception as exc:  # noqa: BLE001 - a formula the harness cannot read
        raise PetabError(f"cannot read the formula {text!r}: {exc}") from exc


class OutputModel:
    """The problem's observables and noise model on one engine's states.

    ``state_ids`` are the species columns an engine reports (SBML units),
    ``params`` the sensitivity set. ``evaluate(X, S)`` takes the per-condition
    states ``X[c]`` ``(n_t, n_x)`` and sensitivities ``S[c]`` ``(n_t, n_x, n_p)`` and
    returns, per measurement row, the observable ``y``, its sensitivities ``sy``,
    the noise scale ``sigma`` and ``ssigma``."""

    def __init__(self, problem: PetabProblem, state_ids: list[str], params: list[str]):
        import sympy as sp

        self.problem, self.state_ids, self.params = problem, list(state_ids), list(params)
        info = problem.model
        self.x_syms = [sp.Symbol(s) for s in state_ids]
        self.p_syms = [sp.Symbol(p) for p in params]
        odf = problem.observable_df.set_index("observableId")
        self.observables = list(odf.index)
        self.transform = {o: (odf.loc[o].get("observableTransformation") or "lin")
                          for o in self.observables}
        self.distribution = {o: (odf.loc[o].get("noiseDistribution") or "normal")
                             for o in self.observables}
        for o in self.observables:
            if self.transform[o] not in ("lin", "log", "log10"):
                raise PetabError(f"observable {o}: transformation {self.transform[o]!r}")
            if self.distribution[o] not in ("normal", "laplace"):
                raise PetabError(f"observable {o}: noise distribution {self.distribution[o]!r}")
        self._compiled = {}
        for o in self.observables:
            h = self._expand(_sympify(odf.loc[o, "observableFormula"], set()), o)
            g = self._expand(_sympify(odf.loc[o, "noiseFormula"], set()), o)
            self._compiled[o] = (self._compile(h, o), self._compile(g, o))

    def _expand(self, expr, obs: str):
        """Substitute every assignment-rule target (recursively) and SBML function
        definition; leave the placeholders and model names as symbols."""
        import sympy as sp

        info = self.problem.model
        rules = info["assignment_rules"]
        for _ in range(50):
            hits = {s for s in expr.free_symbols if s.name in rules}
            if not hits:
                break
            expr = expr.subs({s: _sympify(rules[s.name], set()) for s in hits})
        else:
            raise PetabError(f"observable {obs}: assignment rules nest too deep")
        for fname, (args, body) in info["functions"].items():
            f = sp.Function(fname)
            if expr.has(f):
                lam = sp.Lambda(tuple(sp.Symbol(a) for a in args), _sympify(body, set()))
                expr = expr.replace(f, lam)
        return expr

    def _compile(self, expr, obs: str):
        """The expression's value and its gradients in the states and the
        sensitivity set, as functions of (x, p, env), env holding every other name."""
        import sympy as sp

        others = sorted((s for s in expr.free_symbols
                         if s not in set(self.x_syms) | set(self.p_syms)), key=lambda s: s.name)
        args = [*self.x_syms, *self.p_syms, *others]
        used_x = [k for k, s in enumerate(self.x_syms) if s in expr.free_symbols]
        used_p = [k for k, s in enumerate(self.p_syms) if s in expr.free_symbols]
        f = sp.lambdify(args, expr, modules="numpy")
        dfx = [(k, sp.lambdify(args, sp.diff(expr, self.x_syms[k]), modules="numpy"))
               for k in used_x]
        dfp = [(k, sp.lambdify(args, sp.diff(expr, self.p_syms[k]), modules="numpy"))
               for k in used_p]
        return {"f": f, "dx": dfx, "dp": dfp, "others": [s.name for s in others],
                "expr": str(expr)[:200]}

    def _env(self, row: Row) -> dict[str, float]:
        prob = self.problem
        cond = prob.conditions[row.condition]
        env: dict[str, float] = {}
        env.update({p: float(v) for p, v in prob.model["parameters"].items()})
        env.update({c: float(d["size"]) for c, d in prob.model["compartments"].items()})
        env.update({p: float(v) for p, v in prob.nominal.items()})
        env.update(cond.parameters)
        for n, v in enumerate(row.observable_parameters, start=1):
            env[f"observableParameter{n}_{row.observable}"] = v
        for n, v in enumerate(row.noise_parameters, start=1):
            env[f"noiseParameter{n}_{row.observable}"] = v
        env["time"] = env["t"] = row.time
        return env

    def _eval(self, comp, x: np.ndarray, p: np.ndarray, env: dict[str, float], s: np.ndarray):
        missing = [n for n in comp["others"] if n not in env]
        if missing:
            raise PetabError(f"a formula reads {missing[:5]}, which nothing gives a value "
                             f"({comp['expr']})")
        args = [*x, *p, *(env[n] for n in comp["others"])]
        value = float(comp["f"](*args))
        grad = np.zeros(len(self.params))
        for k, fn in comp["dx"]:
            grad += float(fn(*args)) * s[k]
        for k, fn in comp["dp"]:
            grad[k] += float(fn(*args))
        return value, grad

    def evaluate(self, X: list[np.ndarray], S: list[np.ndarray]) -> dict[str, np.ndarray]:
        prob = self.problem
        n = len(prob.rows)
        y, sigma = np.zeros(n), np.zeros(n)
        sy, ssigma = np.zeros((n, len(self.params))), np.zeros((n, len(self.params)))
        cond_values = [np.array([c.parameters.get(q, prob.nominal.get(q, np.nan))
                                 for q in self.params]) for c in prob.conditions]
        for j, row in enumerate(prob.rows):
            x = X[row.condition][row.time_index]
            s = S[row.condition][row.time_index]
            env = self._env(row)
            p = cond_values[row.condition]
            h, g = self._compiled[row.observable]
            y[j], sy[j] = self._eval(h, x, p, env, s)
            sigma[j], ssigma[j] = self._eval(g, x, p, env, s)
        return {"y": y, "sy": sy, "sigma": sigma, "ssigma": ssigma}

    def likelihood(self, out: dict[str, np.ndarray]) -> dict[str, Any]:
        """PEtab's log-likelihood of the measurements, AMICI's convention of it, and
        the gradient over the sensitivity set (the same in both conventions)."""
        prob = self.problem
        llh = 0.0
        jac = 0.0  # the data-only term of a log-transformed observable
        grad = np.zeros(len(self.params))
        for j, row in enumerate(prob.rows):
            m, s, sg = row.measurement, out["y"][j], out["sigma"][j]
            ds, dsg = out["sy"][j], out["ssigma"][j]
            tr, dist = self.transform[row.observable], self.distribution[row.observable]
            if tr == "lin":
                tm, ts, dts = m, s, 1.0
                jterm = 0.0
            elif tr == "log":
                tm, ts, dts = math.log(m), math.log(s), 1.0 / s
                jterm = math.log(m)
            else:
                tm, ts, dts = math.log10(m), math.log10(s), 1.0 / (s * math.log(10))
                jterm = math.log(m * math.log(10))
            r = (ts - tm) / sg
            if dist == "normal":
                value = -0.5 * math.log(2 * math.pi * sg ** 2) - 0.5 * r ** 2
                dval_ds = -r / sg * dts
                dval_dsg = -1.0 / sg + r ** 2 / sg
            else:
                value = -math.log(2 * sg) - abs(r)
                dval_ds = -math.copysign(1.0, r) / sg * dts
                dval_dsg = -1.0 / sg + abs(r) / sg
            llh += value - jterm
            jac += jterm
            grad += dval_ds * ds + dval_dsg * dsg
        return {"llh": llh, "llh_amici_convention": llh + jac, "gradient": grad}
