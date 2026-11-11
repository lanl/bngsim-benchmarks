"""The shared cross-engine comparison protocol for every parity suite.

This is the canonical implementation of the verdict logic the bng_parity suite
developed and tuned to its 895/895 golden (`bng_parity/parity_diff.py`), lifted
here so rr_parity (and amici_parity) apply the **identical** protocol rather
than each rolling its own bar. Two pure verdicts, both operating on
already-aligned arrays (the runner matches variables + time points between the
two engines first):

  deterministic_verdict  — ODE / deterministic parity. The ODE-solver error
                           model ``|a-b| <= atol + rtol*|y|`` as a per-cell
                           combined absolute+relative tolerance, a per-species
                           peak-relative significance gate (a column below the
                           model's dynamic range, or never diverging past the
                           ceiling at its own peak scale, is forgiven — the
                           verdict ``dev/investigations/rr_check.py`` trusts),
                           then a fail-fraction budget gated by hard ceilings so
                           a concentrated real divergence is never forgiven. A
                           movement gate (issue #873) holds every species that
                           moves in the reference to MOVE_CEILING of its range
                           over time, whatever its magnitude, so a frozen engine
                           cannot pass by a reservoir or a baseline setting the
                           scale; ``no_signal`` says where nothing moves at all.
  ensemble_verdict       — stochastic (SSA / NF) parity. Per-cell K-sigma test
                           on the difference of ensemble means with a
                           relative-agreement escape hatch and a near-zero
                           skip; pass iff at least ``ENSEMBLE_PASS_FRAC`` of
                           cells pass.

Why a budget + ceilings instead of a single worst-cell ``max_rel_err``: two
independent stiff integrators at the same tolerance legitimately disagree above
a tight relative bar on a handful of transient cells, while a *genuine* engine
divergence shows up as either a large fraction of failing cells or a cell past
the hard ceiling. The budget forgives the former; the ceilings catch the
latter. (See ``oracles.max_rel_err`` for the simpler single-scalar metric; this
module is the richer pass/fail protocol the suites actually gate on.)

The constant values match BNGL particle-space tuning, but the protocol is
domain-independent — what differs between suites is the *integration tolerance*
each engine is run at (rr_parity forces a tight SBML-appropriate default), not
this comparison.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Tuned constants (single source of truth; bng_parity/parity_diff.py mirrors
# these values — migrate it to import from here in a follow-up).
# --------------------------------------------------------------------------- #
# Deterministic per-cell tolerance |a-b| <= ABS_TOL_FILE*file_scale
#   + ABS_TOL_COL*col_peak + REL_TOL*max(|a|,|b|).
REL_TOL = 1e-4  # relative term (rtol*|y|)
ABS_TOL_COL = 1e-6  # absolute term scaled by the per-column (per-variable) peak
ABS_TOL_FILE = 1e-9  # absolute term scaled by the file-wide peak (sub-scale columns)
NEAR_ZERO_FLOOR_REL = 1e-12  # denom floor for the reported relative diff
# Fail-fraction budget: forgive soft cells (past the per-cell tol but within the
# hard ceilings) iff their fraction is within budget AND none breach a ceiling.
FAIL_FRAC_BUDGET = 5e-3
HARD_REL_CEILING = 0.05  # a magnitude-carrying cell past this is never forgiven
HARD_ABS_CEILING_FILE = 1e-2  # absolute ceiling, scaled by file peak (unconditional)
SIGNIF_FLOOR = 1e-3  # peak-relative significance gate: a column whose peak-over-time
#   is below this fraction of the file-wide peak is below the model's dynamic range
#   and cannot carry a hard fail (the per-species verdict the rr_parity triage trusts,
#   dev/investigations/rr_check.py — own/global_peak <= 1e-3 is forgiven).
#   The movement gate below applies the same fraction to the columns' ranges.
# Movement gate (issue #873): a column whose range over time, on either side, is
# past REL_TOL of its peak moves, and one whose range is past SIGNIF_FLOOR of the
# largest such range is significant by movement. A cell of a significant moving
# column that is further from the reference than MOVE_CEILING of that range is a
# hard fail, whatever the magnitude gates above say: a frozen engine is at least
# half the range off, and so is an engine that moves a species the reference
# holds still, and a reservoir or a large baseline no longer hides either.
# 1e-2 is from the #873 census (macOS arm64, BNG 2.9.3): the worst current PASS is
# 7.5e-3 of its movement (ph_schrodinger), p99 4.4e-5, and a 4% error on one
# species is at least 4e-2.
MOVE_CEILING = 1e-2
# A cell whose two values are no further apart than MOVE_ATOL_FACTOR times the
# integration atol the caller passes cannot fail the movement gate: two solvers
# that each hold their local error to atol differ by a few atol on a species near
# zero. On rr_ode (atol 1e-12, BioModels in molar units) the gate without that
# floor failed 8 current PASSes on species at 1e-11 to 1e-15, 1.0 to 3.8 atol
# apart (BIOMD0000000186/187 the widest); the real divergence it found there
# (BIOMD0000000285, an event bngsim did not fire until issue #934) is 2e13 atol
# apart.
MOVE_ATOL_FACTOR = 10.0
# Stochastic ensemble verdict.
ENSEMBLE_K = 3.0  # per-cell sigma threshold on the difference of means (calibrated at N=10)
ENSEMBLE_K_BASE_N = 10  # the replicate count ENSEMBLE_K was calibrated at
ENSEMBLE_PASS_FRAC = 0.99  # pass iff >= this fraction of cells pass
NEAR_ZERO_REL = 1e-9  # ensemble cells below this*scale are skipped (sampling noise)


def deterministic_verdict(
    a: np.ndarray,
    b: np.ndarray,
    *,
    forgive_mask: np.ndarray | None = None,
    atol: float | None = None,
) -> dict:
    """Verdict for two aligned deterministic runs ``a`` (bngsim) vs ``b`` (ref).

    ``a``/``b`` are shape ``(n_time, n_var)`` value arrays (no time column),
    already restricted to the common variables in a shared column order. NaNs
    are handled cell-wise: both-NaN is a zero-diff pass, one-side-NaN is a fail.
    ``forgive_mask`` is an optional boolean mask of cells to forgive *before*
    the budget step (bng_parity passes its step-discontinuity-shift mask here;
    rr_parity passes None) — a strict superset, so omitting it only tightens.

    The per-species peak-relative significance gate (``SIGNIF_FLOOR`` +
    per-column ``HARD_REL_CEILING``) forgives failing cells in any column that is
    not a genuine divergence — below the model's dynamic range, or never
    diverging past the ceiling at its own peak-over-time scale — matching the
    verdict ``dev/investigations/rr_check.py`` uses; a one-side-non-finite cell
    and an absolute-ceiling breach are never forgiven this way. ``max_rel`` is
    reported per column peak (not per cell).

    The movement gate (issue #873) takes each column's range over time as the
    larger of the two sides' ranges: a column whose range is past ``REL_TOL`` of
    its peak moves, and one whose range is past ``SIGNIF_FLOOR`` of the largest
    such range is significant. A cell of such a column further from the
    reference than ``MOVE_CEILING`` of the range is a hard fail that only
    ``forgive_mask`` and the near-zero backstop forgive. It does not need the
    cell to be past the per-cell tolerance: a column that moves by 1e-4 of its
    peak, frozen, is inside it. Both sides count because the hole is symmetric:
    an engine that never moves fails against a reference that does, and an
    engine that moves a species the reference holds still (one it should never
    make, a clamped one it lets drift) fails the same way. ``atol``, when given,
    is the absolute integration tolerance both engines ran at: a cell whose two
    values are no further apart than ``MOVE_ATOL_FACTOR`` times that cannot fail
    the movement gate, since the solvers do not resolve the difference. It
    changes nothing else.

    Returns a dict: ``passed`` (the gate), ``max_rel``/``max_abs`` (worst over
    the genuinely-failing cells, 0 when it passes), and the bucket counts
    (``n_fail``/``n_soft_fail``/``frac_soft_fail``/``n_hard_fail``/
    ``budget_forgiven``/``n_cells``) for the report comment, with
    ``max_err_over_movement`` (the worst cell of a significant moving column, as
    a part of that column's range) and ``no_signal`` (no column moves on either
    side, so the verdict could not tell a frozen engine from a right one).

    When ``forgive_mask`` is given, two more keys audit what it did (issue #316).
    A caller cannot compute either from outside: the mask alone says which cells
    are *inside* it, which is a strict superset of the cells it changed anything
    for — most masked cells were never failing, and of those that were, the
    near-zero backstop or the dynamic-range gate may already have forgiven them.

    * ``n_forgive_rescued`` — cells removed from ``effective_fail`` by this mask
      and by nothing else. "How much of the verdict rested on the mask."
    * ``passed_without_forgive`` — the same verdict **recomputed** with the mask
      empty. "Did the mask turn a fail into a pass."

    Under the current constants the second is implied by the first: a rescued
    cell that is merely *soft* can only exist in a column the significance gate
    calls real, and a column is real only if some cell exceeds
    ``HARD_REL_CEILING`` (0.05) — which is 500x ``REL_TOL``, so that cell is
    itself a hard fail. There is no way to rescue a soft cell without also
    leaving a hard one behind, and a cell the movement gate fails is hard on its
    own. It is computed rather than inferred anyway, because that equivalence is
    a property of constants and a gate ordering, none of which this field should
    depend on; ``test_noise_floor_audit`` pins the equivalence so a future change
    to either is a visible event.

    Both keys are absent when no mask was passed, so a consumer cannot mistake
    "no mask" for "the mask rescued nothing".
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    both_nan = np.isnan(a) & np.isnan(b)
    # A cell where exactly one side is non-finite (NaN or inf while the other is
    # a finite number) is an unambiguous divergence — one engine blew up where
    # the other produced a value. It must be an unconditional hard fail: such a
    # cell has colmag collapsed to 0 below (np.maximum with a non-finite → 0),
    # so without this guard the near-zero backstop would forgive it as
    # underflow, silently PASSing a genuine NaN/inf divergence.
    one_side_nonfinite = np.isfinite(a) != np.isfinite(b)
    absd = np.abs(a - b)
    absd = np.where(both_nan, 0.0, absd)

    finite_mag = np.concatenate([np.abs(a).ravel(), np.abs(b).ravel()])
    finite_mag = finite_mag[np.isfinite(finite_mag)]
    scale = float(finite_mag.max()) if finite_mag.size else 1.0
    zero_floor = max(1e-12, scale * NEAR_ZERO_FLOOR_REL)

    colmag = np.maximum(np.abs(a), np.abs(b))
    colmag = np.where(np.isfinite(colmag), colmag, 0.0)
    col_peak = colmag.max(axis=0) if colmag.size else np.zeros(0)
    col_peak_denom = np.maximum(col_peak, zero_floor)  # per-column denom, always > 0

    absd_clean = np.where(np.isnan(absd), np.inf, absd)  # one-side NaN -> flag

    cell_tol = ABS_TOL_FILE * scale + ABS_TOL_COL * col_peak[np.newaxis, :] + REL_TOL * colmag
    fail_mask = absd_clean > cell_tol

    # Peak-relative significance gate — the per-species verdict the rr_parity
    # triage trusts (dev/investigations/rr_check.py). Judge each column by its
    # OWN peak-over-time, not per cell:
    #   reld_peak = |a-b| / col_peak   normalizes the divergence by the column's
    #     peak magnitude, so a species that was large early and decayed to a
    #     near-zero tail is measured against its own dynamic range, not the
    #     vanishing tail value (per-cell normalization saturates there);
    #   col_significant marks columns carrying >= SIGNIF_FLOOR of the file peak.
    # A column is a genuine ("real") divergence only if it BOTH diverges past the
    # hard relative ceiling at peak scale AND is within the model's dynamic range.
    reld_peak = np.where(np.isnan(absd), np.inf, absd_clean / col_peak_denom[np.newaxis, :])
    col_reldiv_peak = reld_peak.max(axis=0) if reld_peak.size else np.zeros(0)
    col_significant = col_peak > SIGNIF_FLOOR * scale
    col_is_real = (col_reldiv_peak > HARD_REL_CEILING) & col_significant

    hard_abs_fail = absd_clean > HARD_ABS_CEILING_FILE * scale

    # Forgive, before the budget: an injected mask (e.g. step-discontinuity
    # shifts), a near-zero backstop (both sides below the file-scale floor =
    # underflow), and the dynamic-range gate — failing cells in a column that is
    # NOT a real divergence (below the significance floor, or never diverging past
    # the relative ceiling at its own peak scale) are below the model's dynamic
    # range, exactly the cells rr_check forgives. A one-side-non-finite cell (a
    # blow-up on one engine) and an absolute-ceiling breach are NEVER forgiven —
    # regardless of significance.
    forgive = np.zeros_like(fail_mask) if forgive_mask is None else np.asarray(forgive_mask, bool)
    # A cell where both sides are below the file-scale floor is underflow, for the
    # per-cell test and the movement gate alike.
    near_zero_cell = (colmag < zero_floor) & ~one_side_nonfinite
    near_zero_mask = fail_mask & near_zero_cell
    below_dyn_range = (
        fail_mask & (~col_is_real)[np.newaxis, :] & ~one_side_nonfinite & ~hard_abs_fail
    )

    # A relative blow-up is HARD only on a cell whose column carries magnitude
    # (significance gate) and diverges past the ceiling at the column's peak scale
    # (per-column, not per-cell, normalization). The absolute ceiling is
    # unconditional; a one-side-non-finite cell is unconditionally hard. The
    # fail-fraction budget below still forgives a few soft cells within a real
    # column, but the dynamic-range gate above now subsumes the global soft path
    # (a sub-ceiling column is forgiven outright rather than counted against it).
    hard_rel_fail = (reld_peak > HARD_REL_CEILING) & col_significant[np.newaxis, :]

    # Issue #873 — the movement gate. Each column's range over time, over its
    # finite cells, on whichever side moves more; the file's movement scale is the
    # largest range of a column that moves. A frozen copy of the reference is half
    # a range off or more in every moving column, which the magnitude gates above
    # forgave where a clamped reservoir set the file scale (catalysis.bngl), where
    # a nearly constant species did, or where a column moves by under
    # HARD_REL_CEILING of its own peak (a small signal on a large baseline). An
    # engine that moves a column the reference holds still is the same miss seen
    # from the other side, so its range counts too.
    ref_range, ref_peak = _range_and_peak(b)
    eng_range, eng_peak = _range_and_peak(a)
    col_range = np.maximum(ref_range, eng_range)
    col_moves = col_range > REL_TOL * np.maximum(ref_peak, eng_peak)
    move_scale = float(col_range[col_moves].max()) if col_moves.any() else 0.0
    col_moves_significant = col_moves & (col_range > SIGNIF_FLOOR * move_scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        err_over_move = np.where(
            col_moves_significant[np.newaxis, :],
            absd_clean / np.where(col_range > 0.0, col_range, 1.0)[np.newaxis, :],
            0.0,
        )
    # Cells the movement gate leaves alone: underflow on both sides, and a
    # difference the solvers do not resolve (a few atol on a species at 1e-11).
    move_exempt = near_zero_cell.copy()
    if atol is not None:
        move_exempt |= absd_clean <= MOVE_ATOL_FACTOR * float(atol)
    hard_move_fail = (err_over_move > MOVE_CEILING) & ~move_exempt

    def _resolve(forgive_now: np.ndarray) -> tuple[dict, np.ndarray]:
        """Everything downstream of ``forgive``, so the #316 counterfactual is
        the real verdict rather than a re-derivation of it that could drift.
        Returns the verdict and its ``effective_fail``."""
        # The movement gate is a fail of its own, not a cell past the per-cell
        # tolerance: a column that moves by 1e-4 of its peak, frozen, is inside
        # REL_TOL of it at every point.
        move_fail = hard_move_fail & ~forgive_now
        effective_fail = (
            (fail_mask & ~(forgive_now | near_zero_mask | below_dyn_range))
            | move_fail
            | one_side_nonfinite
        )
        hard_fail = effective_fail & (
            hard_rel_fail | hard_abs_fail | move_fail | one_side_nonfinite
        )
        soft_fail = effective_fail & ~hard_fail

        total_cells = int(effective_fail.size)
        n_soft_fail = int(np.sum(soft_fail))
        frac_soft_fail = (n_soft_fail / total_cells) if total_cells else 0.0
        budget_ok = frac_soft_fail <= FAIL_FRAC_BUDGET
        remaining_fail = hard_fail if budget_ok else effective_fail
        n_remaining = int(np.sum(remaining_fail))

        return {
            "passed": n_remaining == 0,
            "max_rel": float(np.max(reld_peak[remaining_fail])) if n_remaining else 0.0,
            "max_abs": float(np.max(absd_clean[remaining_fail])) if n_remaining else 0.0,
            "n_cells": total_cells,
            # A cell the movement gate fails is a failing cell (issue #873), though
            # it can be inside the per-cell tolerance.
            "n_fail": int(np.sum(fail_mask | hard_move_fail)),
            "n_soft_fail": n_soft_fail,
            "frac_soft_fail": frac_soft_fail,
            "n_hard_fail": int(np.sum(hard_fail)),
            "budget_forgiven": n_soft_fail if budget_ok else 0,
            "scale": scale,
            # Issue #873: the worst distance from the reference over the range of
            # its column, over the significant moving columns and the cells not
            # forgiven, and whether there is no such column at all (nothing either
            # side does could tell a frozen engine from a right one).
            "max_err_over_movement": float(
                np.max(np.where(forgive_now | move_exempt, 0.0, err_over_move))
            )
            if err_over_move.size
            else 0.0,
            "no_signal": not bool(col_moves.any()),
        }, effective_fail

    out, effective_fail = _resolve(forgive)
    if forgive_mask is not None:
        # Issue #316 — what the mask actually did, which `forgive.sum()` cannot
        # say. A cell counts as rescued only if it was failing, the mask cleared
        # it, and nothing else would have: the failing cells of the verdict
        # without the mask that are not failing with it. That leaves out a
        # one-side-non-finite cell (re-added unconditionally) and a cell the
        # near-zero backstop or the dynamic-range gate already forgave, and takes
        # in a cell only the movement gate failed (issue #873), which no
        # per-gate formula over `fail_mask` would see.
        bare, bare_fail = _resolve(np.zeros_like(fail_mask))
        out["n_forgive_rescued"] = int(np.sum(bare_fail & ~effective_fail))
        out["passed_without_forgive"] = bool(bare["passed"])
    return out


def ode_row_comment(v: dict, n_sp: int) -> str:
    """The terse row comment the rr and amici runners write for a
    :func:`deterministic_verdict`, and their matrices' ``humanize_comment`` parse:
    the cell tallies, the worst error over movement and, where nothing moves,
    ``NO SIGNAL`` (issue #873). One builder, so the two suites cannot drift."""
    return (
        f"{n_sp} sp; fail {v['n_fail']}/{v['n_cells']} "
        f"(hard {v['n_hard_fail']}, soft {v['n_soft_fail']}, forgiven {v['budget_forgiven']}); "
        f"err/movement {v['max_err_over_movement']:.1e}"
        + ("; NO SIGNAL (nothing moves in either engine)" if v["no_signal"] else "")
    )


def _range_and_peak(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each column's range over time and its peak magnitude, over its finite
    cells (both 0 for a column with none)."""
    finite = np.isfinite(x)
    if not x.shape[0]:
        return np.zeros(x.shape[1]), np.zeros(x.shape[1])
    hi = np.where(finite, x, -np.inf).max(axis=0)
    lo = np.where(finite, x, np.inf).min(axis=0)
    peak = np.where(finite, np.abs(x), 0.0).max(axis=0)
    return np.where(finite.any(axis=0), hi - lo, 0.0), peak


def ensemble_k_for(n_rep: int) -> float:
    """Effect-size-preserving per-cell sigma threshold for an ``n_rep`` ensemble.

    The verdict's per-cell test is ``|Δmean| <= K·combined_sem`` and
    ``sem = std/sqrt(n_rep)``, so for a FIXED true difference the z-score grows like
    ``sqrt(n_rep)``. With a constant ``K`` (calibrated at ``ENSEMBLE_K_BASE_N``), more
    replicates therefore mechanically flag ever-tinier, sub-meaningful differences as
    fails (in the limit N→∞ every model DIFFs). Scaling ``K`` by
    ``sqrt(n_rep / ENSEMBLE_K_BASE_N)`` keeps the gate detecting the same minimum
    *meaningful* (effect-size) difference regardless of N — the stochastic analogue of
    the rr_parity z-tol scaling (GH #190). Returns ``ENSEMBLE_K`` at the calibration N.
    """
    import math

    return ENSEMBLE_K * math.sqrt(max(int(n_rep), 1) / ENSEMBLE_K_BASE_N)


def ensemble_verdict(
    mean_a: np.ndarray,
    sem_a: np.ndarray,
    mean_b: np.ndarray,
    sem_b: np.ndarray,
    *,
    k: float = ENSEMBLE_K,
) -> dict:
    """Verdict for two aligned sampled ensembles via the K-sigma frac-pass test.

    ``mean_*``/``sem_*`` are the per-cell ensemble mean and standard error
    (``std(ddof=1)/sqrt(n)``, e.g. from ``oracles.ensemble_stats``), already
    restricted to the common variables. A cell passes if the difference of
    means is within ``ENSEMBLE_K`` combined standard errors OR within the
    deterministic relative tolerance (the escape hatch for cells that are
    effectively deterministic, where se collapses to 0); near-zero cells are
    skipped. The model passes iff at least ``ENSEMBLE_PASS_FRAC`` of cells pass.

    Returns ``passed``, ``frac_pass``, ``max_z`` (worst per-cell z, for the
    report), ``max_abs_mean_diff``, ``n_cells``, ``n_pass``.
    """
    mean_a, sem_a = np.asarray(mean_a, float), np.asarray(sem_a, float)
    mean_b, sem_b = np.asarray(mean_b, float), np.asarray(sem_b, float)
    if not (mean_a.shape == sem_a.shape == mean_b.shape == sem_b.shape):
        raise ValueError("mean/sem shape mismatch")

    diff = np.abs(mean_a - mean_b)
    se = np.sqrt(sem_a**2 + sem_b**2)  # sem already = std/sqrt(n) per side
    scale = max(
        float(np.nanmax(np.abs(mean_a))) if mean_a.size else 0.0,
        float(np.nanmax(np.abs(mean_b))) if mean_b.size else 0.0,
        1e-12,
    )
    floor = 1e-12 * scale
    near_zero = np.maximum(np.abs(mean_a), np.abs(mean_b)) < NEAR_ZERO_REL * scale
    threshold = k * np.maximum(se, floor)
    rel_floor = np.maximum(np.maximum(np.abs(mean_a), np.abs(mean_b)), scale * NEAR_ZERO_FLOOR_REL)
    rel_ok = diff <= REL_TOL * rel_floor

    both_nan = np.isnan(mean_a) & np.isnan(mean_b)
    either_nan = np.isnan(mean_a) | np.isnan(mean_b)
    cell_pass = (diff <= threshold) | rel_ok
    cell_pass = np.where(both_nan, True, cell_pass)
    cell_pass = np.where(either_nan & ~both_nan, False, cell_pass)
    cell_pass = np.where(near_zero, True, cell_pass)

    n_total = int(cell_pass.size)
    n_pass = int(np.sum(cell_pass))
    frac_pass = n_pass / n_total if n_total else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        z = diff / np.maximum(se, floor)
    max_z = float(np.nanmax(z)) if z.size else 0.0
    return {
        "passed": frac_pass >= ENSEMBLE_PASS_FRAC,
        "frac_pass": frac_pass,
        "max_z": max_z,
        "k": float(k),
        "max_abs_mean_diff": float(np.nanmax(diff)) if diff.size else 0.0,
        "n_cells": n_total,
        "n_pass": n_pass,
    }
