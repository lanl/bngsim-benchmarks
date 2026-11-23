"""Verdicts on forward sensitivities: how far apart two engines' ``dy/dp`` are, and
how well finite differences resolve them. Pure numpy, no simulator.

A forward-sensitivity tensor is ``S[t, i, k] = d y_i(t) / d p_k`` on a time grid,
for outputs ``y_i`` (species amounts, and observables as weighted sums of them) and
parameters ``p_k``. Two engines are compared column by column -- one column is one
output's sensitivity to one parameter over time -- on the **semi-scaled**
sensitivity ``p_k S[t, i, k]``: the change in ``y_i`` per relative change in
``p_k``, in the units of ``y_i``, so that columns for parameters of very different
magnitudes (an Avogadro-scaled constant beside a rate in 1/s) are on one footing.

**The error of a column** is the largest difference over time, as a fraction of the
column's own scale::

    error[i, k] = max_t |p_k (S_a - S_b)| / scale[i, k]
    scale[i, k] = max( max_t |p_k S|  (the larger of the two engines' peaks),
                       floor_rel * max_t |y_i|,
                       floor_atol * atol / tol )

and a comparison passes when every judged column's error is at most ``tol``. The
first term makes the test relative to each sensitivity's peak magnitude. The two
floors stop a negligible column from being judged on its noise: a parameter whose
doubling moves an output by less than ``floor_rel`` of that output's peak has a
sensitivity nothing depends on, and it is held only to ``tol * floor_rel`` of the
output's peak, in absolute terms; and no column is held to less than ``floor_atol``
times the solvers' absolute state tolerance, below which no engine resolves the
state, let alone its derivative. A NaN or an infinity in either column is an error
of infinity: an engine that returns no number has not returned the sensitivity.

**Finite differences.** A central difference ``D(h) = (y(p(1+h)) - y(p(1-h))) /
(2 h p)`` has a truncation error of order ``h^2`` and an error from the integration
of order ``delta / h``, where ``delta`` is the integrator's error in ``y``. Two
steps, ``h`` and ``h/2``, bound both: when truncation dominates, the error of
``D(h/2)`` is a third of ``|D(h) - D(h/2)|``; when the integration dominates, the
difference is the error of two independent quotients, at least that of the finer
one. So ``D(h/2)`` is the reference and ``|D(h) - D(h/2)|`` its error estimate, cell
by cell, and an FD column is **resolved** -- judged at all -- only where that
estimate is at most ``fd_resolved * tol`` of the column's scale, so the
reference's own error cannot decide a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def scaled(S: np.ndarray, values: np.ndarray) -> np.ndarray:
    """``p_k S[t, i, k]``: the semi-scaled sensitivity, in the units of the output."""
    return np.asarray(S, dtype=float) * np.asarray(values, dtype=float)[None, None, :]


def outputs(S: np.ndarray, X: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Species plus observables: ``S`` ``(n_t, n_species, n_p)`` and ``X``
    ``(n_t, n_species)`` extended along the output axis by the observables, each a
    weighted sum of species (``weights`` ``(n_obs, n_species)``). Applied to every
    engine alike, so an observable is judged by the same arithmetic everywhere."""
    S = np.asarray(S, dtype=float)
    X = np.asarray(X, dtype=float)
    if weights.size == 0:
        return S, X
    S_obs = np.einsum("oi,tik->tok", weights, S)
    return np.concatenate([S, S_obs], axis=1), np.concatenate([X, X @ weights.T], axis=1)


def peak(sig: np.ndarray) -> np.ndarray:
    """``(n_out, n_p)``: each column's largest magnitude over time, over its finite
    cells (0 for a column with none)."""
    sig = np.asarray(sig, dtype=float)
    return np.max(np.where(np.isfinite(sig), np.abs(sig), 0.0), axis=0)


def column_scale(sig_a: np.ndarray, sig_b: np.ndarray, y_peak: np.ndarray, *, atol: float,
                 verdict: dict[str, Any]) -> np.ndarray:
    """``(n_out, n_p)``: each column's scale, the larger engine's peak, floored."""
    floor = np.maximum(verdict["floor_rel"] * np.asarray(y_peak, dtype=float)[:, None],
                       verdict["floor_atol"] * float(atol) / verdict["tol"])
    return np.maximum(np.maximum(peak(sig_a), peak(sig_b)), floor)


def column_errors(sig_a: np.ndarray, sig_b: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """``(n_out, n_p)``: each column's largest difference over its scale; infinity
    for a column with a NaN or an infinity in either engine."""
    diff = np.abs(sig_a - sig_b)
    bad = ~np.isfinite(diff).all(axis=0)
    err = np.where(bad, np.inf, np.max(np.where(np.isfinite(diff), diff, 0.0), axis=0) / scale)
    return err


@dataclass
class Comparison:
    """Two engines' sensitivities, column by column."""

    errors: np.ndarray  # (n_out, n_p) error per column (see the module docstring)
    judged: np.ndarray  # (n_out, n_p) bool: the column was compared
    floored: np.ndarray  # (n_out, n_p) bool: the column's scale is a floor
    worst_time: np.ndarray  # (n_out, n_p) index of each column's largest difference
    tol: float

    @property
    def failing(self) -> np.ndarray:
        return self.judged & ~(self.errors <= self.tol)

    @property
    def passed(self) -> bool:
        return bool(self.judged.any()) and not bool(self.failing.any())

    def summary(self, outputs: list[str], params: list[str], times: np.ndarray, *,
                n_species: int, listed: int = 20) -> dict[str, Any]:
        """The comparison as JSON: pass, the worst judged column, and counts."""
        out: dict[str, Any] = {
            "passed": self.passed,
            "tol": self.tol,
            "n_columns": int(self.errors.size),
            "n_judged": int(self.judged.sum()),
            "n_failed": int(self.failing.sum()),
            "n_floored": int((self.judged & self.floored).sum()),
            "params_judged": sorted({params[k] for k in np.nonzero(self.judged.any(axis=0))[0]}),
        }
        if not self.judged.any():
            out.update(max_error=None, reason="no column was judged")
            return out
        err = np.where(self.judged, self.errors, -1.0)
        i, k = np.unravel_index(int(np.argmax(err)), err.shape)
        out["max_error"] = _num(err[i, k])
        out["max_error_species"] = _num(np.max(err[:n_species])) if n_species else None
        out["max_error_observables"] = _num(np.max(err[n_species:])) \
            if err.shape[0] > n_species else None
        out["worst"] = {"output": outputs[i], "parameter": params[k], "error": _num(err[i, k]),
                        "time": float(times[self.worst_time[i, k]]),
                        "floored": bool(self.floored[i, k])}
        fails = np.argwhere(self.failing)
        if fails.size:
            order = np.argsort(-self.errors[self.failing], kind="stable")[:listed]
            out["failed"] = [[outputs[fails[j][0]], params[fails[j][1]],
                              _num(self.errors[tuple(fails[j])])] for j in order]
            out["params_failed"] = sorted({params[c[1]] for c in fails})
        return out


def _num(x: float) -> float | str:
    """A float for JSON; infinity as the string 'inf' (JSON has no infinity)."""
    return float(x) if np.isfinite(x) else "inf"


def compare(sig_a: np.ndarray, sig_b: np.ndarray, y_peak: np.ndarray, *, atol: float,
            verdict: dict[str, Any], judged: np.ndarray | None = None) -> Comparison:
    """Compare two engines' semi-scaled sensitivities ``(n_t, n_out, n_p)``.

    ``judged`` restricts the comparison to some columns (a parameter an engine did
    not differentiate, an FD column it does not resolve); every column otherwise.
    """
    sig_a = np.asarray(sig_a, dtype=float)
    sig_b = np.asarray(sig_b, dtype=float)
    if sig_a.shape != sig_b.shape:
        raise ValueError(f"shapes differ: {sig_a.shape} and {sig_b.shape}")
    scale = column_scale(sig_a, sig_b, y_peak, atol=atol, verdict=verdict)
    diff = np.abs(sig_a - sig_b)
    worst_time = np.argmax(np.where(np.isfinite(diff), diff, np.inf), axis=0)
    return Comparison(
        errors=column_errors(sig_a, sig_b, scale),
        judged=np.ones(scale.shape, bool) if judged is None else np.asarray(judged, bool),
        floored=~(np.maximum(peak(sig_a), peak(sig_b)) >= scale),
        worst_time=worst_time,
        tol=float(verdict["tol"]),
    )


def fd_estimate(sig_h: np.ndarray, sig_h2: np.ndarray) -> np.ndarray:
    """``(n_t, n_out, n_p)``: the two-step estimate of the error of ``D(h/2)``."""
    return np.abs(np.asarray(sig_h, dtype=float) - np.asarray(sig_h2, dtype=float))


def fd_resolved(estimate: np.ndarray, sig_other: np.ndarray, sig_fd: np.ndarray,
                y_peak: np.ndarray, *, atol: float, verdict: dict[str, Any]) -> np.ndarray:
    """``(n_out, n_p)`` bool: the FD columns whose error estimate, at its largest, is
    at most ``fd_resolved * tol`` of the column's scale against ``sig_other``."""
    scale = column_scale(sig_other, sig_fd, y_peak, atol=atol, verdict=verdict)
    est = np.max(np.where(np.isfinite(estimate), estimate, np.inf), axis=0)
    return est <= verdict["fd_resolved"] * verdict["tol"] * scale


def refuted(failing: np.ndarray, vouchers: list[tuple[np.ndarray, np.ndarray]]) -> bool:
    """Whether a reference's disagreement with BNGsim is the reference's own.

    ``failing`` is the reference's failing columns against BNGsim. Each voucher is
    another reference's ``(passes_against_bngsim, fails_against_this_reference)``,
    both column masks. The disagreement is refuted when every failing column is
    vouched for by some other reference that agrees with BNGsim there and disagrees
    with this one there: two independent engines against one.
    """
    if not failing.any() or not vouchers:
        return False
    vouched = np.zeros_like(failing, dtype=bool)
    for agrees_with_bngsim, disagrees_with_ref in vouchers:
        vouched |= agrees_with_bngsim & disagrees_with_ref
    return bool(np.all(vouched[failing]))
