"""A small, independent reader for BioNetGen ``.net`` files.

It reads what the harness itself needs to know about a generated network, without
going through any simulator: the parameters (with their expressions evaluated),
the species and their initial counts, the reactions with their reactants,
products and rate expressions, and the names of any functions.

Two uses so far, both in the SSA capability:

- the exact first and second moments of a first-order network
  (``bench.stochastic.linear_moments``), a reference that must not share code with
  any engine it judges;
- spotting a reaction with a repeated reactant (``A + A -> ...``), whose exact SSA
  propensity an engine that fires the SBML rate law gets wrong.

Expressions are evaluated by walking a parsed Python AST that admits only numbers,
names already defined, arithmetic and a few BNGL math functions, so a model file
can never run code.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from pathlib import Path


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

    def repeated_reactants(self) -> list[Reaction]:
        """Reactions that consume one (unfixed) species more than once, ``A + A -> ...``."""
        out = []
        for r in self.reactions:
            live = [i for i in r.reactants if not self.fixed[i]]
            if len(live) != len(set(live)):
                out.append(r)
        return out


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


def parse(text: str) -> Network:
    """Read a ``.net`` file's parameters, species, reactions and function names."""
    blocks = _blocks(text)
    net = Network()
    for line in blocks.get("parameters", []):
        parts = line.split(None, 2)
        if len(parts) < 3:
            raise NetFileError(f"bad parameter line {line!r}")
        _, name, expr = parts
        net.parameters[name] = evaluate(expr, net.parameters)
    for line in blocks.get("functions", []):
        parts = line.split(None, 2)
        if len(parts) >= 2:
            net.functions.add(parts[1].split("(", 1)[0])
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
    return net


def read(path: str | Path) -> Network:
    return parse(Path(path).read_text(encoding="utf-8"))
