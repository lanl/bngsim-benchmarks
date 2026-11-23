"""Statistics for stochastic simulation: ensemble verdicts and exact reference moments.

An exact stochastic simulation (SSA) of a network has no single right trajectory;
two correct implementations of the same model sample the same probability law. So
a verdict compares ensembles, with a stated false-alarm rate (docs/DESIGN.md §5.3).

**Two ensembles** (BNGsim against a reference engine, ``two_sample_verdict``). Both
are stacks of ``n`` independent trajectories on one time grid, ``(n, n_times,
n_species)``. At every cell (time point, species) two statistics are formed:

- the difference of the ensemble means, standardised by the pooled standard
  deviation, ``(mean_A - mean_B) / sqrt(s2 * (1/n_A + 1/n_B))``;
- the difference of the ensemble variances, standardised by the pooled estimate
  of the variance of a squared deviation, ``(var_A - var_B) / sqrt(tau2 * (1/n_A
  + 1/n_B))`` with ``tau2 = m4 - m2**2`` -- the large-sample variance of a sample
  variance whatever the distribution, where a Gaussian formula (DSMTS's) would be
  wrong for the skewed, discrete counts of a low-copy model.

These are DSMTS's per-time-point tests on means and variances (Evans, Gillespie
& Wilkinson 2008), made two-sample. The verdict's statistic is their maximum
absolute value over every cell, and its null distribution comes from relabelling:
if the engines sample the same law, the pooled trajectories are exchangeable, so
every split of the pool into groups of ``n_A`` and ``n_B`` is equally likely. The
p-value is the share of random relabellings whose maximum is at least the observed
one (Westfall & Young 1993, single-step max-T). That makes the false-alarm rate
**exactly** ``alpha`` (at most), for any ensemble size, any dependence between time
points and species, and any marginal distribution -- zeros, skew and discreteness
included -- with no normal approximation and no Bonferroni conservatism. The
relabellings use a fixed seed, so a verdict is reproducible to the bit.

Relabelling stops early once ``stop_exceedances`` relabellings have beaten the
observed maximum (Besag & Clifford 1991): the p-value is then ``h / k`` and the
comparison passes, having been shown to be far from significant. Only a
comparison that might fail runs every relabelling.

Two kinds of cell are left out, because they carry no information or none the
other test does not: a cell constant across both ensembles (an initial
condition), and, for the variance test only, a cell taking at most two values
(a 0/1 promoter state), whose variance is fixed by its mean.

**One ensemble against exact moments** (``analytic_verdict``), DSMTS-style: the
per-cell ``z = sqrt(n) (mean - mu) / sigma`` with the exact ``mu`` and ``sigma``,
and, for the variance, ``log(s2 / sigma**2)`` over its delta-method standard error
``sqrt((m4 / m2**2 - 1) / n)`` from the sample's kurtosis (DSMTS's ``Y_t`` assumes a
Gaussian kurtosis, which low counts do not have); the comparison fails when the
smallest two-sided normal p-value is at most ``alpha / m`` over the ``m`` tests
(Bonferroni, valid under any dependence). There is nothing to relabel, so this
rests on the central limit theorem: its false-alarm rate is ``alpha``
asymptotically, and DSMTS's advice of 10^4 trajectories applies (a spec that
declares exact moments must ask for that many). A cell whose exact variance is zero is deterministic:
every trajectory must hit the exact mean.

**Exact moments** (``linear_moments``). Where every reaction's propensity is at
most linear in the state (zero- and first-order mass action), the first two
moments obey closed linear ODEs (Gadgil, Lee & Othmer 2005): ``m' = A m + b`` and
``C' = A C + C A^T + sum_r nu_r nu_r^T a_r(m)``. They are solved here exactly, by the
matrix exponential of the augmented linear system, from the network file alone --
no simulator's code is involved.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import netfile

# --------------------------------------------------------------------------- #
# Two ensembles: permutation max-T on means and variances
# --------------------------------------------------------------------------- #
TWO_SAMPLE_TEST = "permutation max-T on per-cell means and variances (Westfall-Young)"
MEANS_TEST = "permutation max-T on per-cell means (Westfall-Young)"
ANALYTIC_TEST = "per-cell z on means and variances, Bonferroni (DSMTS-style)"
_BATCH = 64  # relabellings per matrix product


def _cells(stack: np.ndarray) -> np.ndarray:
    """``(n, n_times, n_species)`` -> ``(n, n_cells)`` float64."""
    a = np.asarray(stack, dtype=float)
    if a.ndim != 3:
        raise ValueError(f"an ensemble is (replicates, times, species), got shape {a.shape}")
    return a.reshape(a.shape[0], -1)


def _two_valued(x: np.ndarray) -> np.ndarray:
    """Per column: does it take at most two distinct values?"""
    lo, hi = x.min(axis=0), x.max(axis=0)
    return np.all((x == lo) | (x == hi), axis=0)


def _locate(flat: int, shape: tuple[int, int], times: np.ndarray | None,
            species: list[str] | None) -> dict[str, Any]:
    t, s = divmod(int(flat), shape[1])
    out: dict[str, Any] = {"time_index": t, "species_index": s}
    if times is not None:
        out["time"] = float(times[t])
    if species is not None:
        out["species"] = species[s]
    return out


def two_sample_verdict(
    a: np.ndarray,
    b: np.ndarray,
    *,
    alpha: float,
    permutations: int,
    seed: int,
    stop_exceedances: int = 20,
    times: np.ndarray | None = None,
    species: list[str] | None = None,
    variance: bool = True,
) -> dict[str, Any]:
    """Do ensembles ``a`` and ``b`` sample the same law? A permutation max-T test.

    ``a`` and ``b`` are ``(n, n_times, n_species)`` on the same grid and species
    order. Returns ``passed`` (p-value above ``alpha``), the ``p_value``, how many
    relabellings were drawn and whether that stopped early, the observed maximum
    ``z_max`` split into its mean and variance parts, the worst cell, and, when every
    relabelling ran, the critical value and how many cells exceed it.

    ``variance=False`` tests the means alone: do the two laws have the same mean at
    every cell? The same relabelling argument makes its false-alarm rate exact, and
    a law that differs from the other only in its spread passes it.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape[1:] != b.shape[1:]:
        return {"passed": False, "reason": f"grids differ: {list(a.shape)} vs {list(b.shape)}"}
    na, nb = a.shape[0], b.shape[0]
    if na < 2 or nb < 2:
        return {"passed": False, "reason": "an ensemble needs at least two replicates"}
    grid = a.shape[1:]
    x = np.concatenate([_cells(a), _cells(b)])
    n = na + nb
    mu = x.mean(axis=0)
    xc = x - mu
    m2 = np.mean(xc**2, axis=0)
    s2 = m2 * n / (n - 1)
    tau2 = np.mean(xc**4, axis=0) - m2**2
    live = s2 > 0
    live_var = live & ~_two_valued(x) & (tau2 > 0) & bool(variance)
    out: dict[str, Any] = {
        "test": TWO_SAMPLE_TEST if variance else MEANS_TEST,
        "alpha": alpha,
        "replicates": [na, nb],
        "n_cells": int(x.shape[1]),
        "n_tested": {"mean": int(live.sum()), "variance": int(live_var.sum())},
    }
    if not live.any():
        # Nothing varies in either ensemble: they agree iff they are identical.
        same = bool(np.array_equal(_cells(a)[0], _cells(b)[0]))
        out.update(passed=same, p_value=1.0 if same else 0.0, permutations=0,
                   stopped_early=False, z_max=0.0 if same else math.inf)
        return out
    cols = np.flatnonzero(live)
    vcols = np.flatnonzero(live_var)
    vpos = np.searchsorted(cols, vcols)  # the variance-tested cells, within cols
    # Centred on the pooled mean (which no relabelling changes), so the sums of
    # squares below do not cancel catastrophically.
    y1 = xc[:, cols]
    y2 = y1[:, vpos] ** 2
    t1, t2 = y1.sum(axis=0), y2.sum(axis=0)
    t1v = t1[vpos]
    scale = 1.0 / na + 1.0 / nb
    sd_mean = np.sqrt(s2[cols] * scale)
    sd_var = np.sqrt(tau2[vcols] * scale)

    def stats(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per relabelling (rows of 0/1 ``w``, 1 = group A): |z| on means and on
        variances. Sums over a group are one matrix product; the other group's
        are the pool's total minus them."""
        s1 = w @ y1
        za = (s1 / na - (t1 - s1) / nb) / sd_mean
        if not vcols.size:
            return np.abs(za), np.zeros((w.shape[0], 0))
        s1v, s2v = s1[:, vpos], w @ y2
        ma, mb = s1v / na, (t1v - s1v) / nb
        va = (s2v - na * ma**2) / (na - 1)
        vb = (t2 - s2v - nb * mb**2) / (nb - 1)
        return np.abs(za), np.abs((va - vb) / sd_var)

    w0 = np.zeros((1, n))
    w0[0, :na] = 1.0
    zm, zv = stats(w0)
    zm, zv = zm[0], zv[0]
    t_obs = float(max(zm.max(), zv.max() if zv.size else 0.0))

    rng = np.random.default_rng(seed)
    base = np.zeros(n)
    base[:na] = 1.0
    maxima: list[np.ndarray] = []
    drawn = exceed = 0
    stopped = False
    while drawn < permutations:
        size = min(_BATCH, permutations - drawn)
        w = rng.permuted(np.tile(base, (size, 1)), axis=1)
        pm, pv = stats(w)
        tmax = pm.max(axis=1)
        if pv.shape[1]:
            tmax = np.maximum(tmax, pv.max(axis=1))
        hits = np.cumsum(tmax >= t_obs)
        if exceed + hits[-1] >= stop_exceedances:
            # Stop at the relabelling that brought the count to the threshold.
            k = int(np.argmax(exceed + hits >= stop_exceedances)) + 1
            drawn += k
            exceed = stop_exceedances
            maxima.append(tmax[:k])
            stopped = True
            break
        drawn += size
        exceed += int(hits[-1])
        maxima.append(tmax)
    p = exceed / drawn if stopped else (1 + exceed) / (1 + drawn)

    worst_m = int(np.argmax(zm))
    worst = {"statistic": "mean", "z": float(zm[worst_m]),
             **_locate(cols[worst_m], grid, times, species)}
    if zv.size and zv.max() > zm.max():
        worst_v = int(np.argmax(zv))
        worst = {"statistic": "variance", "z": float(zv[worst_v]),
                 **_locate(vcols[worst_v], grid, times, species)}
    out.update(
        passed=bool(p > alpha),
        p_value=float(p),
        permutations=int(drawn),
        stopped_early=stopped,
        seed=int(seed),
        z_max=t_obs,
        z_max_mean=float(zm.max()),
        z_max_variance=float(zv.max()) if zv.size else None,
        worst=worst,
    )
    if not stopped:
        null = np.concatenate(maxima)
        crit = float(np.quantile(null, 1.0 - alpha, method="higher"))
        out["critical_z"] = crit
        out["n_flagged"] = {"mean": int((zm > crit).sum()), "variance": int((zv > crit).sum())}
    return out


def parity_core_verdict(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """bngsim's own ensemble verdict on the same two ensembles, for comparison
    only: ``parity_core.differ.ensemble_verdict``, a per-cell K-sigma test on the
    means passing when 99% of cells pass, with K scaled to the replicate count.
    It does not gate this suite (it has no stated false-alarm rate), but a reader
    of both repositories can see where they agree."""
    from ._vendor.parity_core import differ, oracles

    ma, sa = oracles.ensemble_stats(_cells(a))
    mb, sb = oracles.ensemble_stats(_cells(b))
    v = differ.ensemble_verdict(ma, sa, mb, sb, k=differ.ensemble_k_for(min(len(a), len(b))))
    return {k: v[k] for k in ("passed", "frac_pass", "max_z", "k")}


# --------------------------------------------------------------------------- #
# One ensemble against exact moments
# --------------------------------------------------------------------------- #
def _normal_two_sided(z: np.ndarray) -> np.ndarray:
    return np.array([math.erfc(abs(float(v)) / math.sqrt(2.0)) for v in np.ravel(z)]).reshape(
        np.shape(z))


def analytic_verdict(
    x: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    *,
    alpha: float,
    times: np.ndarray | None = None,
    species: list[str] | None = None,
) -> dict[str, Any]:
    """Does ensemble ``x`` (``(n, n_times, n_species)``) have the exact ``mean`` and
    ``var`` (each ``(n_times, n_species)``) at every cell? Bonferroni over per-cell
    normal tests on the mean and the variance."""
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    var = np.clip(np.asarray(var, dtype=float), 0.0, None)
    if x.shape[1:] != mean.shape or mean.shape != var.shape:
        return {"passed": False,
                "reason": f"grids differ: {list(x.shape)} vs {list(mean.shape)}"}
    n = x.shape[0]
    grid = x.shape[1:]
    xs = _cells(x)
    mu, v = mean.ravel(), var.ravel()
    xbar = xs.mean(axis=0)
    dev = xs - xbar
    m2 = np.mean(dev**2, axis=0)
    s2 = m2 * n / (n - 1) if n > 1 else np.zeros_like(m2)
    m4 = np.mean(dev**4, axis=0)
    # A cell whose exact variance is (numerically) zero is deterministic.
    det = v <= 1e-12 * np.maximum(1.0, mu**2)
    det_ok = np.all(np.abs(xs[:, det] - mu[det]) <= 1e-9 * np.maximum(1.0, np.abs(mu[det])),
                    axis=0)
    live = ~det
    zm = np.zeros(mu.size)
    zm[live] = (xbar[live] - mu[live]) / np.sqrt(v[live] / n)
    kurt = np.divide(m4, m2**2, out=np.zeros_like(m4), where=m2 > 0)
    live_var = live & ~_two_valued(xs) & (kurt > 1.0)
    zv = np.zeros(mu.size)
    # On the log scale: a sample variance and the fourth moment that standardises
    # it are low together, which makes ``(s2 - sigma2) / sqrt(tau2 / n)`` too heavy
    # in its lower tail for skewed counts; ``log(s2 / sigma2)`` is closer to normal
    # (measured: Poisson(0.5) counts, 2000 replicates, P(|z| > 3) is 0.0038 against
    # the linear form's 0.0051 and the normal 0.0027). Its delta-method variance
    # is (m4 / m2**2 - 1) / n, positive for any cell with three or more values.
    zv[live_var] = np.log(s2[live_var] / v[live_var]) / np.sqrt((kurt[live_var] - 1.0) / n)
    m = int(live.sum() + live_var.sum())
    out: dict[str, Any] = {
        "test": ANALYTIC_TEST,
        "alpha": alpha,
        "replicates": n,
        "n_cells": int(mu.size),
        "n_tested": {"mean": int(live.sum()), "variance": int(live_var.sum()),
                     "deterministic": int(det.sum())},
        "n_deterministic_failed": int((~det_ok).sum()),
    }
    if m == 0:
        passed = bool(det_ok.all())
        out.update(passed=passed, p_value=1.0 if passed else 0.0,
                   p_min=1.0 if passed else 0.0, z_max=0.0)
        return out
    z_all = np.concatenate([np.abs(zm[live]), np.abs(zv[live_var])])
    p_all = _normal_two_sided(z_all)
    threshold = alpha / m
    k = int(np.argmax(z_all))
    flat = (np.flatnonzero(live).tolist() + np.flatnonzero(live_var).tolist())[k]
    out.update(
        passed=bool(det_ok.all() and p_all.min() > threshold),
        # The family's p-value, comparable to alpha as the permutation test's is:
        # it is above alpha exactly when the smallest cell's is above alpha / m.
        p_value=float(min(1.0, m * p_all.min())) if det_ok.all() else 0.0,
        p_min=float(p_all.min()),
        bonferroni_threshold=threshold,
        n_rejected=int((p_all <= threshold).sum()),
        z_max=float(z_all[k]),
        z_max_mean=float(np.abs(zm[live]).max()),
        z_max_variance=float(np.abs(zv[live_var]).max()) if live_var.any() else None,
        worst={"statistic": "mean" if k < live.sum() else "variance", "z": float(z_all[k]),
               **_locate(flat, grid, times, species)},
    )
    return out


# --------------------------------------------------------------------------- #
# Exact moments of a first-order network
# --------------------------------------------------------------------------- #
# The dense moment system has n + n**2 + 1 unknowns; past this many species the
# matrix exponential stops being cheap, and first-order networks are small.
MAX_LINEAR_SPECIES = 40


def _expm(m: np.ndarray) -> np.ndarray:
    """exp(m) by scaling and squaring with a degree-20 Taylor series: the series is
    taken where ``||m / 2**s||_1 <= 0.5``, so its remainder is under 1e-25."""
    norm = float(np.linalg.norm(m, 1))
    s = max(0, math.ceil(math.log2(norm / 0.5))) if norm > 0.5 else 0
    x = m / (2.0**s)
    eye = np.eye(m.shape[0])
    out, term = eye.copy(), eye.copy()
    for k in range(1, 21):
        term = term @ x / k
        out = out + term
    for _ in range(s):
        out = out @ out
    return out


def linear_moments(net: netfile.Network, t_end: float, n_steps: int
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Exact means and variances, ``(n_steps + 1, n_species)`` each, on the uniform
    grid ``0 .. t_end``, of a network whose propensities are all at most linear.

    Raises ``netfile.NetFileError`` for a network that is not: a reaction with two
    or more (unfixed) reactants, or a rate that is a function or cannot be
    evaluated to a constant. A fixed (``$``) species is a constant: it scales the
    propensities it enters and is never changed.
    """
    n = len(net.species)
    if n == 0:
        raise netfile.NetFileError("the network has no species")
    if n > MAX_LINEAR_SPECIES:
        raise netfile.NetFileError(
            f"{n} species is past the exact-moment solver's {MAX_LINEAR_SPECIES}")
    fixed = np.array(net.fixed, dtype=bool)
    x0 = np.array(net.initial, dtype=float)
    a = np.zeros((n, n))
    b = np.zeros(n)
    noise_lin = np.zeros((n, n, n))  # sum_r c_r nu_r nu_r^T, per linear reactant j
    noise_const = np.zeros((n, n))
    for r in net.reactions:
        name = r.rate.split("(", 1)[0].strip()
        if name in net.functions:
            raise netfile.NetFileError(
                f"reaction {r.index} has a functional rate ({r.rate}); its moments do not close")
        c = netfile.evaluate(r.rate, net.parameters)
        live = [i for i in r.reactants if not fixed[i]]
        for i in r.reactants:
            if fixed[i]:
                c *= x0[i]
        if len(live) > 1:
            raise netfile.NetFileError(
                f"reaction {r.index} is of order {len(live)} ({r.label or r.rate}); "
                "exact moments need every reaction of order 0 or 1")
        nu = np.zeros(n)
        for i in r.reactants:
            nu[i] -= 1
        for i in r.products:
            nu[i] += 1
        nu[fixed] = 0.0
        outer = np.outer(nu, nu)
        if live:
            j = live[0]
            a[:, j] += c * nu
            noise_lin[j] += c * outer
        else:
            b += c * nu
            noise_const += c * outer
    # Augmented state z = [m (n), vec C (n*n, row-major), 1].
    size = n + n * n + 1
    big = np.zeros((size, size))
    one = size - 1
    big[:n, :n] = a
    big[:n, one] = b
    eye = np.eye(n)
    # vec(A C + C A^T) = (A (x) I + I (x) A) vec C for row-major vec.
    big[n:one, n:one] = np.kron(a, eye) + np.kron(eye, a)
    # sum_r nu nu^T a_r(m): linear in m plus a constant.
    big[n:one, :n] = noise_lin.reshape(n, n * n).T
    big[n:one, one] = noise_const.ravel()
    z = np.zeros(size)
    z[:n] = x0
    z[one] = 1.0
    step = _expm(big * (float(t_end) / int(n_steps)))
    means = np.empty((n_steps + 1, n))
    variances = np.empty((n_steps + 1, n))
    for k in range(n_steps + 1):
        means[k] = z[:n]
        variances[k] = np.diag(z[n:one].reshape(n, n))
        z = step @ z
    return means, np.clip(variances, 0.0, None)
