"""A small, independent reader for BioNetGen ``.net`` files.

It reads what the harness itself needs to know about a generated network, without
going through any simulator: the parameters (with their expressions evaluated),
the species and their initial counts, the reactions with their reactants,
products and rate expressions, and the names of any functions.

Its uses:

- the exact first and second moments of a first-order network
  (``bench.stochastic.linear_moments``), a reference that must not share code with
  any engine it judges;
- spotting a reaction with a repeated reactant (``A + A -> ...``), whose exact SSA
  propensity an engine that fires the SBML rate law gets wrong;
- the steady-state capability's engine-free checks: the network's right-hand side
  ``f(x)`` at a state an engine returned (``Network.rhs``), its conservation laws
  (``Network.conservation_laws``), and the network read again with some parameters
  set, as a dose-response point sets them (``parse(..., overrides=...)``);
- for forward sensitivities, the network's structure as written (``structure``):
  which parameters are literals and which reach a reaction rate or an initial
  condition, through derived parameters and functions; the observables as weighted
  sums of species; and a copy of the file with some parameters set to new values
  (``with_parameters``), for finite differences on an engine that reads the ``.net``.

The right-hand side follows BioNetGen's ODE semantics: a reaction's rate is its
rate expression -- a parameter, an expression, or a global function of the
observables -- times the amount of each reactant (a repeated reactant twice); a
``$``-fixed species never changes. Rate laws the reader does not evaluate
(``Sat``, ``MM``, ``Hill``, ``Arrhenius``, ``FunctionProduct``, local functions)
raise ``NetFileError`` when the right-hand side is asked for, never when the file
is read, so reading a network never fails on what only the right-hand side needs.

Expressions are evaluated by walking a parsed Python AST that admits only numbers,
names already defined, arithmetic and a few BNGL math functions, so a model file
can never run code.
"""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class NetFileError(ValueError):
    """A ``.net`` construct this reader does not understand or cannot evaluate."""


@dataclass(frozen=True)
class Reaction:
    index: int
    reactants: tuple[int, ...]  # 0-based species indices; the null species is dropped
    products: tuple[int, ...]
    rate: str  # the rate expression as written (a parameter, an expression, a function)
    label: str = ""


@dataclass
class Network:
    parameters: dict[str, float] = field(default_factory=dict)
    species: list[str] = field(default_factory=list)
    initial: list[float] = field(default_factory=list)
    fixed: list[bool] = field(default_factory=list)  # ``$``-prefixed: held constant
    reactions: list[Reaction] = field(default_factory=list)
    functions: set[str] = field(default_factory=set)
    # Each parameter's expression as written, and each global function's body
    # (None for a local function, one with arguments). Kept as text: only the
    # right-hand side evaluates them.
    parameter_exprs: dict[str, str] = field(default_factory=dict)
    function_bodies: dict[str, str | None] = field(default_factory=dict)
    # Each observable (``begin groups``): (0-based species index, weight) pairs, or
    # the reason the line could not be read.
    groups: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    groups_error: str | None = None

    def is_primary(self, name: str) -> bool:
        """Whether parameter ``name`` is a number in the file, not an expression of
        other parameters (only a primary parameter can be set from outside)."""
        expr = self.parameter_exprs.get(name)
        if expr is None:
            return False
        try:
            float(expr)
        except ValueError:
            return False
        return True

    def repeated_reactants(self) -> list[Reaction]:
        """Reactions that consume one (unfixed) species more than once, ``A + A -> ...``."""
        out = []
        for r in self.reactions:
            live = [i for i in r.reactants if not self.fixed[i]]
            if len(live) != len(set(live)):
                out.append(r)
        return out

    # ------------------------------------------------------------------ #
    # The right-hand side, for the steady-state checks
    # ------------------------------------------------------------------ #
    def stoichiometry(self) -> np.ndarray:
        """The net stoichiometry ``S``, shape ``(n_species, n_reactions)``; a
        ``$``-fixed species' row is zero, since it never changes."""
        s = np.zeros((len(self.species), len(self.reactions)))
        for j, r in enumerate(self.reactions):
            for i in r.reactants:
                s[i, j] -= 1.0
            for i in r.products:
                s[i, j] += 1.0
        s[np.asarray(self.fixed, dtype=bool), :] = 0.0
        return s

    def observables(self, x) -> dict[str, float]:
        """Every observable's value at state ``x``."""
        if self.groups_error:
            raise NetFileError(self.groups_error)
        x = np.asarray(x, dtype=float)
        return {name: float(sum(w * x[i] for i, w in members))
                for name, members in self.groups.items()}

    def function_values(self, x) -> dict[str, float]:
        """Every global function's value at state ``x``; raises NetFileError for a
        function the reader cannot evaluate, when (and only when) it is reached."""
        names: dict[str, float] = {**self.parameters, **self.observables(x)}
        done: dict[str, float] = {}
        calls = re.compile(r"\b(" + "|".join(map(re.escape, self.function_bodies)) + r")\(\s*\)") \
            if self.function_bodies else None

        def value(fname: str, stack: tuple[str, ...]) -> float:
            if fname in done:
                return done[fname]
            if fname in stack:
                raise NetFileError(f"function {fname} reads itself")
            body = self.function_bodies[fname]
            if body is None:
                raise NetFileError(f"local function {fname} (one with arguments) is not evaluated")
            text = calls.sub(r"\1", body) if calls else body
            env = dict(names)
            for other in self.function_bodies:
                if other != fname and re.search(r"\b" + re.escape(other) + r"\b", text):
                    env[other] = value(other, (*stack, fname))
            done[fname] = evaluate(text, env)
            return done[fname]

        for fname in self.function_bodies:
            value(fname, ())
        return done

    def rate_constants(self, x) -> np.ndarray:
        """Each reaction's rate expression at state ``x``: its rate per unit of
        each reactant (a function of the observables makes it depend on ``x``)."""
        for r in self.reactions:
            m = _UNEVALUATED_RATE_RE.match(r.rate)
            if m:
                raise NetFileError(f"reaction {r.index} has a {m.group(1)} rate law, which "
                                   "this reader does not evaluate")
        names = dict(self.parameters)
        if any(names_in(r.rate) & set(self.function_bodies) for r in self.reactions):
            names.update(self.function_values(x))
        cache: dict[str, float] = {}
        out = np.empty(len(self.reactions))
        for j, r in enumerate(self.reactions):
            if r.rate not in cache:
                cache[r.rate] = evaluate(r.rate, names)
            out[j] = cache[r.rate]
        return out

    def rates(self, x) -> np.ndarray:
        """Each reaction's ODE rate at ``x``: its rate expression times the amount of
        each reactant (BioNetGen's mass action, a repeated reactant counted twice)."""
        x = np.asarray(x, dtype=float)
        v = self.rate_constants(x)
        for j, r in enumerate(self.reactions):
            for i in r.reactants:
                v[j] *= x[i]
        return v

    def rhs(self, x) -> np.ndarray:
        """``f(x) = S v(x)``: each species' net rate of change at ``x``."""
        return self.stoichiometry() @ self.rates(x)

    def flux(self, x) -> tuple[np.ndarray, np.ndarray]:
        """``(f(x), g(x))``: each species' net rate of change and its gross flux,
        ``g_i = sum_r |S_ir| |v_r|`` -- what flows in plus what flows out. At a
        steady state ``|f_i|`` is a vanishing part of ``g_i``, in any units."""
        s = self.stoichiometry()
        v = self.rates(x)
        return s @ v, np.abs(s) @ np.abs(v)

    def subspaces(self, rank_tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
        """``(L, Q)``, from an SVD of the stoichiometry, no engine involved: ``L`` an
        orthonormal basis (one row per law) of the vectors with ``L @ S == 0``, every
        linear combination of the amounts that no reaction changes (a fixed species,
        or one no reaction touches, is a law of its own); ``Q`` an orthonormal basis
        (one column per direction) of the directions the reactions move the state in,
        the range of ``S``. Together they span the state space."""
        return subspaces_of(self.stoichiometry(), rank_tol)

    def conservation_laws(self, rank_tol: float = 1e-10) -> np.ndarray:
        """The rows of ``L`` (``subspaces``): the network's conservation laws."""
        return self.subspaces(rank_tol)[0]

    def jacobian(self, x, rel_step: float = 1e-7) -> np.ndarray:
        """``df/dx`` at ``x`` by forward differences of the right-hand side, each
        step ``rel_step`` of the species' amount (or of the state's largest, for a
        species at zero), so no step leaves the non-negative orthant."""
        x = np.asarray(x, dtype=float)
        f0 = self.rhs(x)
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        jac = np.empty((x.size, x.size))
        for i in range(x.size):
            h = rel_step * max(abs(x[i]), 1e-3 * peak, 1e-12)
            y = x.copy()
            y[i] += h
            jac[:, i] = (self.rhs(y) - f0) / h
        return jac

    def pure_sinks(self) -> list[int]:
        """Species some reaction produces and none consumes (and not fixed): a
        write-only accumulator, whose amount grows forever wherever its production
        does not stop, so it has no steady value of its own."""
        consumed = {i for r in self.reactions for i in r.reactants}
        produced = {i for r in self.reactions for i in r.products}
        return sorted(i for i in produced - consumed if not self.fixed[i])


# BNGL math the .net may use in a parameter or rate expression.
_FUNCS = {
    "exp": math.exp, "ln": math.log, "log": math.log, "log10": math.log10, "log2": math.log2,
    "sqrt": math.sqrt, "abs": abs, "min": min, "max": max,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
}
_CONSTS = {"_pi": math.pi, "_e": math.e, "pi": math.pi}
_BINOPS = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
}


def evaluate(expr: str, names: dict[str, float]) -> float:
    """The value of a BNGL arithmetic expression over ``names``.

    Raises NetFileError for anything but numbers, known names, + - * / ^ (or **),
    unary minus and the functions in ``_FUNCS`` -- a function of the state, a
    BNGL ``if``, an unknown name.
    """
    text = expr.strip().replace("^", "**")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise NetFileError(f"cannot parse expression {expr!r}") from exc

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in names:
                return float(names[node.id])
            if node.id in _CONSTS:
                return _CONSTS[node.id]
            raise NetFileError(f"unknown name {node.id!r} in {expr!r}")
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return float(_BINOPS[type(node.op)](ev(node.left), ev(node.right)))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in _FUNCS and not node.keywords:
            return float(_FUNCS[node.func.id](*(ev(a) for a in node.args)))
        raise NetFileError(f"unsupported construct in {expr!r}")

    return ev(tree)


# Rate laws written in a ``.net`` that the right-hand side does not evaluate.
_UNEVALUATED_RATE_RE = re.compile(r"^\s*(?:[-+.\deE]+\*)?(Sat|MM|Hill|Arrhenius|FunctionProduct)\b")
# An identifier not inside a number (the "e" of 2.1e-9 is not a name).
_NAME_RE = re.compile(r"(?<![\w.])[A-Za-z_][A-Za-z0-9_]*")


def names_in(expr: str) -> set[str]:
    """Every identifier an expression mentions. BNGL's own function names (exp, if,
    Sat, ...) come along, but a caller only follows names the network defines."""
    return set(_NAME_RE.findall(expr))


_BLOCK_RE = re.compile(r"^\s*begin\s+(\w+)\s*$", re.IGNORECASE)
_END_RE = re.compile(r"^\s*end\s+(\w+)\s*$", re.IGNORECASE)


def _blocks(text: str) -> dict[str, list[str]]:
    """Each ``begin X ... end X`` block's content lines, comments stripped and
    backslash continuations joined."""
    text = re.sub(r"\\\s*\n", " ", text)
    out: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        m = _BLOCK_RE.match(line)
        if m and current is None:
            current = m.group(1).lower()
            out.setdefault(current, [])
            continue
        m = _END_RE.match(line)
        if m and current is not None and m.group(1).lower() == current:
            current = None
            continue
        if current is not None:
            body = line.split("#", 1)[0].strip()
            if body:
                out[current].append(body)
    return out


def _indices(field_text: str) -> tuple[int, ...]:
    """A reactant/product list ``1,2`` as 0-based indices; ``0`` is the null species."""
    out = []
    for tok in field_text.split(","):
        tok = tok.strip()
        if not tok.isdigit():
            raise NetFileError(f"bad species index list {field_text!r}")
        i = int(tok)
        if i > 0:
            out.append(i - 1)
    return tuple(out)


def subspaces_of(s: np.ndarray, rank_tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """``(L, Q)`` of a stoichiometry matrix ``s`` (species x reactions), as
    ``Network.subspaces`` gives them: ``L`` an orthonormal basis of the left null
    space (one row per conservation law), ``Q`` one of the range (one column per
    direction the reactions move the state in)."""
    s = np.asarray(s, dtype=float)
    n_sp, n_rx = s.shape
    if n_sp == 0:
        return np.zeros((0, 0)), np.zeros((0, 0))
    if n_rx == 0:
        return np.eye(n_sp), np.zeros((n_sp, 0))
    if n_rx >= n_sp:
        _, sv, vh = np.linalg.svd(s.T, full_matrices=False)
        basis = vh
    else:
        u, sv, _ = np.linalg.svd(s, full_matrices=True)
        basis = u.T
    sv = np.concatenate([sv, np.zeros(n_sp - len(sv))])
    rank = int(np.sum(sv > rank_tol * max(float(sv.max(initial=0.0)), 1.0)))
    return basis[rank:].copy(), basis[:rank].T.copy()


def parse(text: str, overrides: Mapping[str, float] | None = None) -> Network:
    """Read a ``.net`` file's parameters, species, reactions, functions and
    observables.

    ``overrides`` sets parameters to values of its own, as BioNetGen's
    ``setParameter`` does before a run: each named parameter takes its value in
    place of its expression, and every later parameter and initial amount is
    evaluated with it, so an initial amount that reads a total moves with it.
    Naming a parameter the file does not have raises NetFileError.
    """
    blocks = _blocks(text)
    net = Network()
    overrides = dict(overrides or {})
    for line in blocks.get("parameters", []):
        parts = line.split(None, 2)
        if len(parts) < 3:
            raise NetFileError(f"bad parameter line {line!r}")
        _, name, expr = parts
        net.parameter_exprs[name] = expr.strip()
        net.parameters[name] = (float(overrides[name]) if name in overrides
                                else evaluate(expr, net.parameters))
    unknown = sorted(set(overrides) - set(net.parameters))
    if unknown:
        raise NetFileError(f"no parameter named {', '.join(map(repr, unknown))}")
    for line in blocks.get("functions", []):
        parts = line.split(None, 2)
        if len(parts) >= 2:
            fname = parts[1].split("(", 1)[0]
            net.functions.add(fname)
            # A global function is written ``name()``; one with arguments is local.
            body = parts[2].strip() if len(parts) == 3 else ""
            net.function_bodies[fname] = body if parts[1].endswith("()") and body else None
    for line in blocks.get("species", []):
        parts = line.split(None, 2)
        if len(parts) < 3:
            raise NetFileError(f"bad species line {line!r}")
        _, name, init = parts
        fixed = name.startswith("$")
        net.species.append(name[1:] if fixed else name)
        net.fixed.append(fixed)
        net.initial.append(evaluate(init, net.parameters))
    raw = blocks.get("reactions", [])
    # The comment carries the rule name; keep it for messages.
    labels = {}
    for line in text.splitlines():
        if "#" in line:
            body, comment = line.split("#", 1)
            labels[body.strip()] = comment.strip()
    for line in raw:
        parts = line.split(None, 3)
        if len(parts) < 4:
            raise NetFileError(f"bad reaction line {line!r}")
        idx, reac, prod, rate = parts
        r = Reaction(int(idx), _indices(reac), _indices(prod), rate.strip(), labels.get(line, ""))
        for i in (*r.reactants, *r.products):
            if i >= len(net.species):
                raise NetFileError(f"reaction {idx} names species {i + 1}, which is not defined")
        net.reactions.append(r)
    for line in blocks.get("groups", []):
        parts = line.split(None, 2)
        try:
            members = []
            for tok in (parts[2].split(",") if len(parts) == 3 else []):
                w, _, i = tok.strip().rpartition("*")
                idx = int(i) - 1
                if not 0 <= idx < len(net.species):
                    raise ValueError(f"species {idx + 1} is not defined")
                members.append((idx, float(w) if w else 1.0))
            net.groups[parts[1]] = members
        except (ValueError, IndexError) as exc:
            # Kept, not raised: only the right-hand side reads the observables.
            net.groups_error = f"bad group line {line!r}: {exc}"
    return net


# A parameters-block line: index, name, the value or expression as written (up to
# any comment), then the comment with the spacing before it.
_PARAM_LINE_RE = re.compile(r"^(\s*\d+\s+)([A-Za-z_]\w*)(\s+)([^#]*?)(\s*(?:#.*)?)$")


def with_parameters(text: str, overrides: Mapping[str, float], *,
                    allow_derived: bool = False) -> str:
    """The ``.net`` text with each parameter in ``overrides`` set to its value, its
    number replaced in the parameters block and every other byte unchanged, as a dose
    is given to an engine that reads the file (run_network), which re-derives every
    parameter, rate and initial amount that depends on it.

    Only a parameter written as a number can be set this way unless
    ``allow_derived``: then a derived parameter's expression is replaced by the
    number too, and it stops depending on what the expression read (BioNetGen's
    ``setParameter`` on a derived parameter). Raises NetFileError for a derived
    parameter otherwise, a value that is not a finite number, or a parameter the file
    does not have."""
    left = dict(overrides)
    bad = sorted(k for k, v in left.items() if not math.isfinite(float(v)))
    if bad:
        raise NetFileError(f"cannot set {', '.join(bad)} to a value that is not a finite number")
    out, block = [], None
    for line in text.splitlines(keepends=True):
        s = line.strip()
        if _BLOCK_RE.match(line):
            block = s.split()[1].lower()
        elif _END_RE.match(line):
            block = None
        elif block == "parameters":
            m = _PARAM_LINE_RE.match(line.rstrip("\n"))
            if m and m.group(2) in left:
                if not allow_derived:
                    try:
                        float(m.group(4))
                    except ValueError:
                        raise NetFileError(f"parameter {m.group(2)!r} is an expression, "
                                           f"{m.group(4)!r}, not a number") from None
                nl = "\n" if line.endswith("\n") else ""
                line = (f"{m.group(1)}{m.group(2)}{m.group(3)}"
                        f"{float(left.pop(m.group(2)))!r}{m.group(5)}{nl}")
        out.append(line)
    if left:
        raise NetFileError(f"no parameter named {', '.join(map(repr, sorted(left)))}")
    return "".join(out)


def read(path: str | Path, overrides: Mapping[str, float] | None = None) -> Network:
    return parse(Path(path).read_text(encoding="utf-8"), overrides)


# --------------------------------------------------------------------------- #
# Structure as written: for choosing and perturbing parameters (sensitivities)
# --------------------------------------------------------------------------- #
_NUMBER_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


@dataclass
class Structure:
    """A ``.net`` as written, nothing evaluated: what each parameter, function,
    rate and initial condition is made of, so a caller can say which parameters a
    reaction rate or an initial condition depends on, and through what."""

    expressions: dict[str, str] = field(default_factory=dict)  # parameter -> as written
    functions: dict[str, str] = field(default_factory=dict)  # function -> body
    rates: list[str] = field(default_factory=list)  # one per reaction
    initial: list[str] = field(default_factory=list)  # one per species
    species: list[str] = field(default_factory=list)
    compartments: dict[str, str] = field(default_factory=dict)  # compartment -> size
    # observable -> [(0-based species index, weight)]: an observable is a weighted
    # sum of species, as BioNetGen's groups block writes it.
    groups: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    # 0-based indices of the species some reaction consumes or produces.
    reacting: set[int] = field(default_factory=set)

    def literal(self, name: str) -> float | None:
        """The value of a parameter written as a number, else None (a derived one)."""
        expr = self.expressions.get(name, "").strip()
        return float(expr) if _NUMBER_RE.match(expr) else None

    def primary(self) -> dict[str, float]:
        """Every parameter written as a number, in file order: the model's own knobs."""
        out = {}
        for name in self.expressions:
            v = self.literal(name)
            if v is not None:
                out[name] = v
        return out

    def closure(self, roots: set[str]) -> set[str]:
        """Every parameter and function the names in ``roots`` depend on, through
        derived parameters and function bodies (the roots included)."""
        seen: set[str] = set()
        stack = list(roots)
        while stack:
            name = stack.pop()
            if name in seen:
                continue
            seen.add(name)
            if name in self.expressions:
                stack.extend(names_in(self.expressions[name]))
            elif name in self.functions:
                stack.extend(names_in(self.functions[name]))
        return seen

    def reaching_rates(self) -> set[str]:
        """Parameters some reaction rate depends on."""
        roots = set().union(*(names_in(r) for r in self.rates)) if self.rates else set()
        return self.closure(roots) & set(self.expressions)

    def reaching_initial(self) -> set[str]:
        """Parameters some species' initial amount depends on."""
        roots = set().union(*(names_in(e) for e in self.initial)) if self.initial else set()
        return self.closure(roots) & set(self.expressions)

    def sizing_compartments(self) -> set[str]:
        """Parameters a compartment's size depends on."""
        roots = set().union(*(names_in(e) for e in self.compartments.values())) \
            if self.compartments else set()
        return self.closure(roots) & set(self.expressions)

    def initial_reading(self, name: str) -> list[int]:
        """The species whose initial amount depends on parameter ``name``."""
        return [i for i, e in enumerate(self.initial)
                if name in self.closure(names_in(e))]

    def inert(self) -> set[int]:
        """The species no reaction consumes or produces and no rate reads (through
        an observable): a species whose amount moves nothing and that nothing moves."""
        roots = set().union(*(names_in(r) for r in self.rates)) if self.rates else set()
        read = self.closure(roots)
        in_rates = {i for g, entries in self.groups.items() if g in read for i, _ in entries}
        return set(range(len(self.species))) - self.reacting - in_rates

    def reads_clock(self) -> bool:
        """Whether some reaction rate depends on the time (BNGL's ``time()``), directly
        or through a function: such a network's dynamics depend on where its clock
        starts, not only on its state."""
        roots = set().union(*(names_in(r) for r in self.rates)) if self.rates else set()
        return "time" in self.closure(roots)

    def observable_weights(self) -> tuple[list[str], np.ndarray]:
        """The observables' names and their ``(n_observables, n_species)`` weights."""
        names = list(self.groups)
        w = np.zeros((len(names), len(self.species)))
        for row, name in enumerate(names):
            for i, weight in self.groups[name]:
                w[row, i] += weight
        return names, w


def _group_entries(text: str) -> list[tuple[int, float]]:
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        weight, _, index = tok.rpartition("*")
        if not index.isdigit():
            raise NetFileError(f"bad group entry {tok!r}")
        out.append((int(index) - 1, float(weight) if weight else 1.0))
    return out


def structure(text: str) -> Structure:
    """Read a ``.net``'s structure as written (``Structure``); nothing is evaluated,
    so it reads any network ``parse`` can, and more."""
    blocks = _blocks(text)
    st = Structure()
    for line in blocks.get("parameters", []):
        parts = line.split(None, 2)
        if len(parts) < 3:
            raise NetFileError(f"bad parameter line {line!r}")
        st.expressions[parts[1]] = parts[2].strip()
    for line in blocks.get("functions", []):
        parts = line.split(None, 2)
        if len(parts) >= 3:
            st.functions[parts[1].split("(", 1)[0]] = parts[2].strip()
    for line in blocks.get("species", []):
        parts = line.split(None, 2)
        if len(parts) < 3:
            raise NetFileError(f"bad species line {line!r}")
        st.species.append(parts[1][1:] if parts[1].startswith("$") else parts[1])
        st.initial.append(parts[2].strip())
    for line in blocks.get("reactions", []):
        parts = line.split(None, 3)
        if len(parts) < 4:
            raise NetFileError(f"bad reaction line {line!r}")
        st.rates.append(parts[3].strip())
        st.reacting.update(_indices(parts[1]), _indices(parts[2]))
    for line in blocks.get("compartments", []):
        parts = line.split()
        if len(parts) >= 4:
            st.compartments[parts[1]] = parts[3]
    for line in blocks.get("groups", []):
        parts = line.split(None, 2)
        if len(parts) == 2:
            st.groups[parts[1]] = []
        elif len(parts) == 3:
            st.groups[parts[1]] = _group_entries(parts[2])
        else:
            raise NetFileError(f"bad group line {line!r}")
        for i, _ in st.groups[parts[1]]:
            if not 0 <= i < len(st.species):
                raise NetFileError(f"group {parts[1]} names species {i + 1}, which is not defined")
    return st


def read_structure(path: str | Path) -> Structure:
    return structure(Path(path).read_text(encoding="utf-8"))


def with_added_parameters(text: str, expressions: Mapping[str, str]) -> str:
    """``text`` with each of ``expressions`` added at the end of the parameters block,
    as a parameter of that name defined by that expression (numbered on from the
    block's last), every other byte unchanged: a value an engine reading the file
    derives from the parameters it was handed. Raises NetFileError for a name the
    file already has, or an expression with a space or a comment."""
    lines = text.splitlines(keepends=True)
    block, last_index, end = None, 0, None
    names = set()
    for n, line in enumerate(lines):
        s = line.strip()
        if _BLOCK_RE.match(line):
            block = s.split()[1].lower()
        elif _END_RE.match(line):
            if block == "parameters":
                end = n
            block = None
        elif block == "parameters":
            m = _PARAM_LINE_RE.match(line.rstrip("\n"))
            if m:
                last_index = max(last_index, int(m.group(1)))
                names.add(m.group(2))
    if end is None:
        raise NetFileError("no parameters block")
    clash = sorted(set(expressions) & names)
    if clash:
        raise NetFileError(f"the file already has parameters named {', '.join(clash)}")
    added = []
    for i, (name, expr) in enumerate(expressions.items(), start=last_index + 1):
        if not expr or any(c.isspace() for c in expr) or "#" in expr:
            raise NetFileError(f"bad parameter expression {expr!r}")
        added.append(f"    {i} {name} {expr}\n")
    return "".join(lines[:end] + added + lines[end:])


# A species-block line: index, name, the initial amount as written (up to any
# comment), then the comment with the spacing before it.
_SPECIES_LINE_RE = re.compile(r"^(\s*\d+\s+)(\S+)(\s+)([^#]*?)(\s*(?:#.*)?)$")


def with_initial_amounts(text: str, amounts) -> str:
    """``text`` with every species' initial amount set to the one in ``amounts`` (one
    per species, in file order), every other byte unchanged.

    The way BNG2.pl continues a simulation in a fresh run_network process: the state
    one run ended in becomes the next run's initial amounts. An amount is a number,
    or an expression of the parameters (a string, written as it is), which the
    engine reading the file evaluates -- a species reset to ``2*S0`` follows S0.
    Raises NetFileError when the count differs from the species block's, for a
    number that is not finite, or for an expression with a space or a comment.
    """
    values: list[float | str] = []
    for a in amounts:
        if isinstance(a, str):
            if not a or any(c.isspace() for c in a) or "#" in a:
                raise NetFileError(f"bad initial-amount expression {a!r}")
            values.append(a)
        else:
            values.append(float(a))
            if not math.isfinite(values[-1]):
                raise NetFileError("an initial amount is not a finite number")
    out, block, i = [], None, 0
    for line in text.splitlines(keepends=True):
        s = line.strip()
        if _BLOCK_RE.match(line):
            block = s.split()[1].lower()
        elif _END_RE.match(line):
            block = None
        elif block == "species":
            m = _SPECIES_LINE_RE.match(line.rstrip("\n"))
            if m and m.group(4):
                if i >= len(values):
                    raise NetFileError(f"more species than the {len(values)} amounts given")
                nl = "\n" if line.endswith("\n") else ""
                v = values[i] if isinstance(values[i], str) else repr(values[i])
                line = f"{m.group(1)}{m.group(2)}{m.group(3)}{v}{m.group(5)}{nl}"
                i += 1
        out.append(line)
    if i != len(values):
        raise NetFileError(f"{len(values)} amounts given for {i} species")
    return "".join(out)
