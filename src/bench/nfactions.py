"""The network-free protocol of a BNGL file: its action steps, read and re-written.

A network-free problem is run the way its authors ran it: BNG2.pl executes the
model's action block, and each ``simulate({method=>"nf", ...})`` (or
``simulate_nf``) step starts a standalone NFsim on the XML BNG2.pl writes, with
any ``setParameter`` between steps applied by BNG2.pl -- something NFsim's own
command line cannot do mid-run. This module reads such an action list into
steps, so that every other engine can replay the same protocol, and writes it
back for BNG2.pl with each step's seed and output suffix replaced.

Only what can be replayed faithfully is accepted: network-free ``simulate``
steps, ``setParameter``, and actions with no effect on a network-free run
(``generate_network``, ``writeXML`` and the like), which are dropped. Anything
else -- an ODE or SSA run, ``parameter_scan``, a concentration change -- is
refused by name, as are NFsim options no other engine can follow.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Actions that change nothing a network-free run reads; dropped from the replay.
INERT_ACTIONS = frozenset({
    "generate_network", "writeXML", "writeSBML", "writeMfile", "writeMexfile",
    "writeModel", "writeNetwork", "writeNET", "writeFile", "visualize", "setOption",
})
NF_METHODS = frozenset({"nf", "nfsim"})
# simulate options every engine can follow (or that do not change the law sampled).
SIM_OPTIONS = frozenset({
    "method", "t_start", "t_end", "n_steps", "seed", "suffix", "prefix", "gml", "utl",
    "complex", "get_final_state", "print_functions", "verbose", "n_output_steps", "param",
})
# Options that hold for the whole run: an engine is built with them once.
RUN_OPTIONS = ("gml", "utl", "complex", "param")
# NFsim flags a ``param`` string may pass, which every engine can follow: -bscb
# (block same-complex binding) and -cb, the same thing in NFsim 1.14.3 (one seed
# gives one trajectory under -cb, -bscb or both; without either NFsim lets a
# bimolecular rule bind within one complex), and -utl N, the traversal limit.
# BNG2.pl appends the string to NFsim's command line as written.
PARAM_SWITCHES = frozenset({"-bscb", "-cb"})
PARAM_VALUED = frozenset({"-utl"})


class ActionError(ValueError):
    """An action list that cannot be replayed network-free."""


@dataclass(frozen=True)
class SetParameter:
    name: str
    value: str  # as written: a number, or an expression in the model's parameters

    def render(self) -> str:
        try:
            float(self.value)
            value = self.value
        except ValueError:
            value = f'"{self.value}"'
        return f'setParameter("{self.name}",{value})'


@dataclass(frozen=True)
class SimulateNF:
    action: str  # "simulate" or "simulate_nf", as written
    options: tuple[tuple[str, str], ...]  # (key, value text) in written order

    def get(self, key: str, default: str | None = None) -> str | None:
        for k, v in self.options:
            if k == key:
                return v
        return default

    def render(self, *, seed: int, suffix: str, get_final_state: bool | None = None) -> str:
        """The step for BNG2.pl, its seed and output suffix replaced (and, if
        given, whether BNG2.pl reads NFsim's final state back)."""
        drop = {"seed", "suffix", "prefix"} | ({"get_final_state"} if get_final_state is not None
                                              else set())
        opts = [(k, v) for k, v in self.options if k not in drop]
        opts += [("suffix", f'"{suffix}"'), ("seed", str(int(seed)))]
        if get_final_state is not None:
            opts.append(("get_final_state", "1" if get_final_state else "0"))
        return f"{self.action}({{" + ",".join(f"{k}=>{v}" for k, v in opts) + "})"


@dataclass(frozen=True)
class Phase:
    """One network-free run of the protocol, and the parameter changes before it."""

    index: int
    sets: tuple[SetParameter, ...]
    duration: float
    n_steps: int
    get_final_state: bool
    step: SimulateNF

    @property
    def n_points(self) -> int:
        return self.n_steps + 1


@dataclass(frozen=True)
class Protocol:
    phases: tuple[Phase, ...]
    gml: int | None = None
    utl: int | None = None
    complex: bool = True
    dropped: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_points(self) -> int:
        return sum(p.n_points for p in self.phases)

    @property
    def single(self) -> bool:
        """One run from the file's seed species, no parameter changed during it."""
        return len(self.phases) == 1

    def describe(self) -> dict:
        return {
            "phases": [{"sets": [[s.name, s.value] for s in p.sets], "duration": p.duration,
                        "n_steps": p.n_steps, "get_final_state": p.get_final_state}
                       for p in self.phases],
            "gml": self.gml, "utl": self.utl, "complex": self.complex,
            "dropped": list(self.dropped),
        }


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
_ACTIONS_BLOCK_RE = re.compile(r"^\s*begin\s+actions\b(.*?)^\s*end\s+actions\b",
                               re.DOTALL | re.MULTILINE | re.IGNORECASE)
_CALL_RE = re.compile(r"^([A-Za-z_]\w*)\s*\((.*)\)\s*;?$", re.DOTALL)


def own_actions(bngl: str) -> str:
    """The text of a BNGL file's ``begin actions ... end actions`` block, or of
    the actions after ``end model`` when there is no such block."""
    m = _ACTIONS_BLOCK_RE.search(bngl)
    if m:
        return m.group(1)
    tail = re.split(r"^\s*end\s+model\b.*$", bngl, flags=re.MULTILINE | re.IGNORECASE)
    if len(tail) > 1:
        return tail[-1]
    raise ActionError("the file has no action block")


def _strip_comment(line: str) -> str:
    out, quote = [], None
    for ch in line:
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#":
            break
        out.append(ch)
    return "".join(out)


def _statements(text: str) -> list[str]:
    lines = [_strip_comment(ln) for ln in text.splitlines()]
    joined = re.sub(r"\\\s*\n", " ", "\n".join(lines))
    out = []
    for ln in joined.splitlines():
        s = ln.strip()
        if s and not re.match(r"^(begin|end)\s+actions\b", s, re.IGNORECASE):
            out.append(s)
    return out


def _split_top(text: str, sep: str) -> list[str]:
    """``text`` split on ``sep`` outside quotes and brackets."""
    parts, depth, quote, cur = [], 0, None, []
    for ch in text:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def _unquote(text: str) -> str:
    t = text.strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
        return t[1:-1]
    return t


def _hash(args: str, where: str) -> tuple[tuple[str, str], ...]:
    a = args.strip()
    if not (a.startswith("{") and a.endswith("}")):
        raise ActionError(f"{where}: expected a {{key=>value, ...}} argument")
    out = []
    for item in _split_top(a[1:-1], ","):
        if "=>" not in item:
            raise ActionError(f"{where}: {item!r} is not key=>value")
        k, v = item.split("=>", 1)
        out.append((_unquote(k), v.strip()))
    return tuple(out)


def parse_steps(text: str) -> tuple[list[SetParameter | SimulateNF], list[str]]:
    """The replayable steps of an action list, and the inert actions dropped."""
    steps: list[SetParameter | SimulateNF] = []
    dropped: list[str] = []
    for stmt in _statements(text):
        m = _CALL_RE.match(stmt)
        if not m:
            raise ActionError(f"cannot read the action {stmt[:80]!r}")
        name, args = m.group(1), m.group(2).strip()
        if name in INERT_ACTIONS:
            dropped.append(name)
        elif name == "setParameter":
            parts = _split_top(args, ",")
            if len(parts) != 2:
                raise ActionError(f"setParameter takes a name and a value: {stmt[:80]!r}")
            steps.append(SetParameter(_unquote(parts[0]), _unquote(parts[1])))
        elif name in ("simulate", "simulate_nf"):
            opts = _hash(args, name)
            keys = [k for k, _ in opts]
            method = _unquote(dict(opts).get("method", '"nf"' if name == "simulate_nf" else ""))
            if method not in NF_METHODS:
                raise ActionError(f"{name}: method {method!r} is not network-free; only "
                                  "network-free runs can be replayed by every NF engine")
            unknown = sorted(set(keys) - SIM_OPTIONS)
            if unknown:
                raise ActionError(f"{name}: option(s) {', '.join(unknown)} cannot be "
                                  "followed by every network-free engine")
            steps.append(SimulateNF(name, opts))
        else:
            raise ActionError(f"the action {name} cannot be replayed network-free "
                              "(only simulate with method nf and setParameter can)")
    return steps, dropped


def _number(text: str | None, what: str) -> float:
    try:
        return float(_unquote(text or ""))
    except ValueError:
        raise ActionError(f"{what}: {text!r} is not a number") from None


def _flag(text: str | None, default: bool) -> bool:
    if text is None:
        return default
    return _number(text, "switch") != 0


def protocol(text: str) -> Protocol:
    """The protocol an action list runs: its network-free phases, each with the
    parameter changes before it, and the options that hold for the whole run."""
    steps, dropped = parse_steps(text)
    phases: list[Phase] = []
    pending: list[SetParameter] = []
    clock = 0.0
    run_opts: dict[str, str | None] = {}
    for step in steps:
        if isinstance(step, SetParameter):
            pending.append(step)
            continue
        where = f"network-free step {len(phases) + 1}"
        n_steps_text = step.get("n_steps", step.get("n_output_steps"))
        if n_steps_text is None or step.get("t_end") is None:
            raise ActionError(f"{where}: needs t_end and n_steps")
        n_steps = _number(n_steps_text, f"{where}.n_steps")
        if n_steps != int(n_steps) or n_steps < 1:
            raise ActionError(f"{where}: n_steps must be a positive integer")
        t_start = _number(step.get("t_start"), f"{where}.t_start") \
            if step.get("t_start") is not None else clock
        t_end = _number(step.get("t_end"), f"{where}.t_end")
        if not t_end > t_start:
            raise ActionError(f"{where}: t_end must be greater than t_start")
        for key in RUN_OPTIONS:
            value = step.get(key)
            if phases and value != run_opts.get(key):
                raise ActionError(f"{where}: {key} differs from the first step's; every "
                                  "engine is built once with these")
            run_opts.setdefault(key, value)
        phases.append(Phase(len(phases), tuple(pending), t_end - t_start, int(n_steps),
                            _flag(step.get("get_final_state"), True), step))
        pending = []
        clock = t_end
    if not phases:
        raise ActionError("the action list runs no network-free simulation")
    def opt_int(key: str) -> int | None:
        v = run_opts.get(key)
        if v is None:
            return None
        x = _number(v, key)
        if x != int(x) or x < 1:
            raise ActionError(f"{key} must be a positive integer")
        return int(x)

    utl = opt_int("utl")
    complex_ = _flag(run_opts.get("complex"), True)
    flags = _param_flags(run_opts.get("param"))
    if "-bscb" in flags or "-cb" in flags:
        complex_ = True
    if "-utl" in flags:
        if utl is not None and utl != flags["-utl"]:
            raise ActionError(f"utl=>{utl} and param -utl {flags['-utl']} disagree")
        utl = flags["-utl"]
    return Protocol(tuple(phases), gml=opt_int("gml"), utl=utl, complex=complex_,
                    dropped=tuple(dropped))


def _param_flags(text: str | None) -> dict[str, int | bool]:
    """The NFsim flags of a ``param`` string, refusing any no other engine can follow."""
    if text is None:
        return {}
    tokens = _unquote(text).split()
    out: dict[str, int | bool] = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in PARAM_SWITCHES:
            out[tok] = True
            i += 1
        elif tok in PARAM_VALUED:
            if i + 1 >= len(tokens):
                raise ActionError(f"param: {tok} needs a value")
            v = _number(tokens[i + 1], f"param {tok}")
            if v != int(v) or v < 1:
                raise ActionError(f"param: {tok} must be a positive integer")
            out[tok] = int(v)
            i += 2
        else:
            raise ActionError(f"param: the NFsim flag {tok!r} cannot be followed by every "
                              f"network-free engine (only {', '.join(sorted(PARAM_SWITCHES | PARAM_VALUED))})")
    return out


def render(proto: Protocol, seeds: list[int], *, suffix: str = "nfp",
           read_back_last: bool = True) -> str:
    """The protocol as BNG2.pl actions: every ``setParameter`` as written and
    every network-free step with ``seeds[k]`` and the output suffix
    ``<suffix><k>``, so each phase writes its own ``.gdat``. With
    ``read_back_last`` false the last step runs with ``get_final_state=>0``:
    no step follows it to use its final state, so BNG2.pl need not read NFsim's
    species file back -- which, for a model of many large complexes, takes BNG2.pl
    far longer than NFsim took to simulate. The trajectory is unchanged."""
    lines = []
    last = len(proto.phases) - 1
    for p in proto.phases:
        lines += [s.render() for s in p.sets]
        keep = None if read_back_last or p.index != last else False
        lines.append(p.step.render(seed=seeds[p.index], suffix=f"{suffix}{p.index}",
                                   get_final_state=keep))
    return "\n".join(lines) + "\n"
