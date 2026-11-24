"""The sensitivity verdict (bench.sensitivity). Synthetic tensors only: no
simulator.

The verdict is checked as a tolerance is: identical tensors pass, a wrong column
fails and is named, a negligible column is judged against its floor rather than
its noise, and a column no engine returned a number for fails. The finite-
difference error estimate is checked against a derivative known in closed form."""

from __future__ import annotations

import numpy as np
import pytest

from bench import sensitivity as sv

VERDICT = {"tol": 1e-3, "floor_rel": 1e-3, "floor_atol": 100.0, "fd_resolved": 0.1}
T = np.linspace(0.0, 10.0, 21)


def _decay(k=0.3, a0=100.0, t=T):
    """A -> B at rate k from A(0) = a0: the state and its exact sensitivities to
    (k, a0), ``(n_t, 2, 2)``."""
    e = np.exp(-k * t)
    X = np.stack([a0 * e, a0 * (1 - e)], axis=1)
    S = np.zeros((t.size, 2, 2))
    S[:, 0, 0], S[:, 1, 0] = -t * a0 * e, t * a0 * e
    S[:, 0, 1], S[:, 1, 1] = e, 1 - e
    return S, X


def _compare(Sa, Sb, X, values=(0.3, 100.0), **kw):
    sa, sb = sv.scaled(Sa, np.array(values)), sv.scaled(Sb, np.array(values))
    return sv.compare(sa, sb, np.abs(X).max(axis=0), atol=1e-8, verdict=VERDICT, **kw)


def test_identical_sensitivities_pass_with_no_error():
    S, X = _decay()
    c = _compare(S, S, X)
    assert c.passed and float(c.errors.max()) == 0.0
    s = c.summary(["A", "B"], ["k", "a0"], T, n_species=2)
    assert s["passed"] and s["max_error"] == 0.0 and s["n_judged"] == 4 and s["n_failed"] == 0


def test_a_one_percent_error_in_one_column_fails_and_is_named():
    S, X = _decay()
    wrong = S.copy()
    wrong[:, 1, 0] *= 1.01
    c = _compare(wrong, S, X)
    assert not c.passed
    s = c.summary(["A", "B"], ["k", "a0"], T, n_species=2)
    assert s["n_failed"] == 1 and s["failed"][0][:2] == ["B", "k"]
    assert s["worst"]["parameter"] == "k" and s["max_error"] == pytest.approx(0.01, rel=0.02)
    assert s["params_failed"] == ["k"]


def test_the_error_is_relative_to_the_columns_peak_not_to_each_cell():
    # Late in the decay dA/dk is tiny; an error there that is huge relative to the
    # cell but small against the column's peak passes.
    S, X = _decay(k=1.5)
    off = S.copy()
    off[-1, 0, 0] += 1e-5 * np.abs(S[:, 0, 0]).max()
    assert abs(off[-1, 0, 0] / S[-1, 0, 0] - 1) > 0.1
    assert _compare(off, S, X, values=(1.5, 100.0)).passed


def test_a_negligible_column_is_judged_against_its_floor_not_its_noise():
    S, X = _decay()
    S = np.concatenate([S, np.zeros((T.size, 2, 1))], axis=2)  # a parameter nothing reads
    noisy = S.copy()
    noisy[:, 0, 2] = 1e-9  # noise far under 1e-3 of A's peak, relative error infinite
    c = _compare(noisy, S, X, values=(0.3, 100.0, 2.0))
    assert c.passed and c.floored[0, 2]
    loud = S.copy()
    loud[:, 0, 2] = 1e-3  # semi-scaled 2e-3: 2e-5 of A's peak, 2% of the floor
    assert not _compare(loud, S, X, values=(0.3, 100.0, 2.0)).passed


def test_no_allowed_error_is_under_a_hundred_state_atols():
    # A species that never moves has no peak to scale by: the atol floor holds.
    S = np.zeros((T.size, 1, 1))
    X = np.zeros((T.size, 1))
    off = S + 0.5e-6  # 50 atols (semi-scaled, p = 1): under the floor
    assert _compare(off, S, X, values=(1.0,)).passed
    assert not _compare(S + 2e-6, S, X, values=(1.0,)).passed


def test_a_column_without_a_number_fails():
    S, X = _decay()
    bad = S.copy()
    bad[3, 1, 1] = np.nan
    c = _compare(bad, S, X)
    assert not c.passed and np.isinf(c.errors[1, 1])
    assert c.summary(["A", "B"], ["k", "a0"], T, n_species=2)["max_error"] == "inf"


def test_only_judged_columns_count():
    S, X = _decay()
    wrong = S.copy()
    wrong[:, :, 0] *= 2
    judged = np.array([[False, True], [False, True]])
    c = _compare(wrong, S, X, judged=judged)
    assert c.passed
    s = c.summary(["A", "B"], ["k", "a0"], T, n_species=2)
    assert s["n_judged"] == 2 and s["params_judged"] == ["a0"]
    assert not _compare(wrong, S, X, judged=np.zeros((2, 2), bool)).passed  # nothing judged


def test_observables_are_weighted_sums_of_species():
    S, X = _decay()
    W = np.array([[1.0, 1.0], [0.0, 2.0]])  # total, and twice B
    S_out, X_out = sv.outputs(S, X, W)
    assert S_out.shape == (T.size, 4, 2) and X_out.shape == (T.size, 4)
    np.testing.assert_allclose(X_out[:, 2], 100.0)
    np.testing.assert_allclose(S_out[:, 2, 0], 0.0, atol=1e-12)  # the total does not move
    np.testing.assert_allclose(S_out[:, 3], 2 * S[:, 1])
    same, _ = sv.outputs(S, X, np.zeros((0, 2)))
    assert same.shape == S.shape


def _central(k, a0, h, which):
    """Central differences of the decay at relative step h, in k or a0."""
    if which == 0:
        up, down = _decay(k * (1 + h), a0)[1], _decay(k * (1 - h), a0)[1]
        return (up - down) / (2 * h * k)
    up, down = _decay(k, a0 * (1 + h))[1], _decay(k, a0 * (1 - h))[1]
    return (up - down) / (2 * h * a0)


def test_the_two_step_estimate_bounds_the_finite_difference_error():
    k, a0, h = 0.3, 100.0, 0.05  # a large step, so truncation dominates
    S, X = _decay(k, a0)
    D_h = np.stack([_central(k, a0, h, 0), _central(k, a0, h, 1)], axis=2)
    D_h2 = np.stack([_central(k, a0, h / 2, 0), _central(k, a0, h / 2, 1)], axis=2)
    values = np.array([k, a0])
    est = sv.fd_estimate(sv.scaled(D_h, values), sv.scaled(D_h2, values))
    # Column by column, at its largest over time (what fd_resolved reads).
    true_err = np.abs(sv.scaled(D_h2 - S, values)).max(axis=0)
    bound = est.max(axis=0)
    assert np.all(true_err <= bound + 1e-12)
    # ... and not by an order of magnitude more than needed (a third, in theory).
    assert true_err[:, 0].max() > 0.2 * bound[:, 0].max()


def test_an_fd_column_is_judged_only_where_its_estimate_resolves_it():
    S, X = _decay()
    values = np.array([0.3, 100.0])
    sig = sv.scaled(S, values)
    est = np.zeros_like(sig)
    est[:, 0, 0] = 1e-2 * np.abs(sig[:, 0, 0]).max()  # 10x the allowed error: unresolved
    resolved = sv.fd_resolved(est, sig, sig, np.abs(X).max(axis=0), atol=1e-8, verdict=VERDICT)
    assert resolved.tolist() == [[False, True], [True, True]]


def test_a_disagreement_is_refuted_only_when_every_failing_column_is_vouched_for():
    failing = np.array([[True, False], [True, False]])
    agrees = np.array([[True, True], [True, True]])
    against = np.array([[True, False], [True, False]])
    assert sv.refuted(failing, [(agrees, against)])
    half = np.array([[True, False], [False, False]])
    assert not sv.refuted(failing, [(agrees, half)])
    # Two vouchers between them can cover the columns.
    other = np.array([[False, False], [True, False]])
    assert sv.refuted(failing, [(agrees, half), (agrees, other)])
    # A voucher that does not itself agree with BNGsim there does not count.
    assert not sv.refuted(failing, [(np.zeros((2, 2), bool), against)])
    assert not sv.refuted(np.zeros((2, 2), bool), [(agrees, against)])
    assert not sv.refuted(failing, [])
