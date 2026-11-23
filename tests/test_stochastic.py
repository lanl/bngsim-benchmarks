"""The stochastic verdicts and the exact reference moments (bench.stochastic), and
the .net reader they stand on (bench.netfile). Synthetic data only: no simulator.

The verdicts are tests with a stated false-alarm rate, so they are checked as
tests are: on data drawn from one law they reject at about alpha, on data from two
laws that differ in the mean or only in the variance they reject, and the same
inputs and seed give the same answer."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bench import netfile
from bench import stochastic as st

IMMIGRATION_DEATH = """# Created by BioNetGen 2.9.3
begin parameters
    1 k                 5.0  # Constant
    2 g                 0.5  # Constant
    3 x0                2*10  # ConstantExpression
end parameters
begin species
    1 X() x0
end species
begin reactions
    1 0 1 k #R_make
    2 1 0 g #R_decay
end reactions
"""


# --------------------------------------------------------------------------- #
# .net reader
# --------------------------------------------------------------------------- #
def test_netfile_evaluates_parameter_expressions_in_order():
    net = netfile.parse(IMMIGRATION_DEATH)
    assert net.parameters == {"k": 5.0, "g": 0.5, "x0": 20.0}
    assert net.species == ["X()"] and net.initial == [20.0] and net.fixed == [False]
    assert [(r.reactants, r.products, r.rate) for r in net.reactions] == [
        ((), (0,), "k"), ((0,), (), "g")]
    assert net.reactions[1].label == "R_decay"


def test_netfile_reads_bngl_powers_functions_and_fixed_species():
    net = netfile.parse("""begin parameters
 1 a 2
 2 b a^3/4
 3 c exp(ln(b))
end parameters
begin functions
 1 rate() a*Obs
end functions
begin species
 1 $Src() 1
 2 A() c
end species
begin reactions
 1 1 1,2 rate #R_make
end reactions
""")
    assert net.parameters["b"] == 2.0 and net.parameters["c"] == pytest.approx(2.0)
    assert net.fixed == [True, False] and net.species == ["Src()", "A()"]
    assert net.functions == {"rate"}


@pytest.mark.parametrize("expr", ["__import__('os')", "a.b", "[1]", "lambda: 1", "Obs + 1"])
def test_netfile_evaluator_refuses_anything_but_arithmetic(expr):
    with pytest.raises(netfile.NetFileError):
        netfile.evaluate(expr, {"a": 1.0})


def test_repeated_reactants_are_found_and_a_fixed_species_does_not_count():
    net = netfile.parse("""begin parameters
 1 k 1
end parameters
begin species
 1 A() 10
 2 $B() 5
end species
begin reactions
 1 1,1 0 0.5*k #R_dimer
 2 2,2 1 k #R_fixed
 3 1,2 0 k #R_hetero
end reactions
""")
    assert [r.index for r in net.repeated_reactants()] == [1]


# --------------------------------------------------------------------------- #
# Exact moments of a first-order network
# --------------------------------------------------------------------------- #
def test_linear_moments_match_immigration_death_in_closed_form():
    # X(t) = Binomial(20, e^-gt) + Poisson(k/g (1 - e^-gt)).
    mean, var = st.linear_moments(netfile.parse(IMMIGRATION_DEATH), 10.0, 100)
    t = np.linspace(0, 10, 101)
    e = np.exp(-0.5 * t)
    np.testing.assert_allclose(mean[:, 0], 20 * e + 10 * (1 - e), rtol=1e-12)
    np.testing.assert_allclose(var[:, 0], 20 * e * (1 - e) + 10 * (1 - e), rtol=1e-11, atol=1e-12)


def test_linear_moments_match_the_dsmts_birth_death_model():
    # DSMTS 001-01: X -> 2X at 0.1, X -> 0 at 0.11, X(0) = 100; its published mean
    # and SD at t = 50 are 60.6531 and 22.3868.
    net = netfile.parse("""begin parameters
 1 lam 0.1
 2 mu 0.11
end parameters
begin species
 1 X() 100
end species
begin reactions
 1 1 1,1 lam
 2 1 0 mu
end reactions
""")
    mean, var = st.linear_moments(net, 50.0, 50)
    assert mean[-1, 0] == pytest.approx(60.6531, abs=1e-4)
    assert math.sqrt(var[-1, 0]) == pytest.approx(22.3868, abs=1e-4)


def test_linear_moments_of_a_chain_carry_the_covariance_through():
    # 0 -> M (v), M -> M + P (b), M -> 0 (d), P -> 0 (e): the stationary protein
    # variance is the burst formula, mean * (1 + b / (d + e)).
    net = netfile.parse("""begin parameters
 1 v 2
 2 b 5
 3 d 1
 4 e 0.1
end parameters
begin species
 1 M() 0
 2 P() 0
end species
begin reactions
 1 0 1 v
 2 1 1,2 b
 3 1 0 d
 4 2 0 e
end reactions
""")
    mean, var = st.linear_moments(net, 400.0, 40)
    p_mean = 2 * 5 / (1 * 0.1)
    assert mean[-1, 1] == pytest.approx(p_mean, rel=1e-9)
    assert var[-1, 1] == pytest.approx(p_mean * (1 + 5 / 1.1), rel=1e-9)
    assert mean[0].tolist() == [0.0, 0.0] and var[0].tolist() == [0.0, 0.0]


@pytest.mark.parametrize("reactions, message", [
    (" 1 1,1 0 k #R_dimer", "order 2"),
    (" 1 0 1 rate #R_make", "functional rate"),
])
def test_linear_moments_refuse_a_network_whose_moments_do_not_close(reactions, message):
    net = netfile.parse(f"""begin parameters
 1 k 1
end parameters
begin functions
 1 rate() k
end functions
begin species
 1 A() 10
end species
begin reactions
{reactions}
end reactions
""")
    with pytest.raises(netfile.NetFileError, match=message):
        st.linear_moments(net, 1.0, 10)


# --------------------------------------------------------------------------- #
# Two ensembles: permutation max-T
# --------------------------------------------------------------------------- #
def _ensemble(rng, n, rate=0.4, n_t=12, n_sp=3, shift=0.0):
    """Low, discrete, time-correlated counts: a random walk of Poisson births and
    binomial deaths, the kind of data a low-copy SSA gives."""
    x = np.zeros((n, n_t, n_sp))
    for t in range(1, n_t):
        x[:, t] = rng.binomial(x[:, t - 1].astype(int), 0.7) + rng.poisson(rate + shift, (n, n_sp))
    return x


def test_two_sample_false_alarm_rate_is_at_most_alpha():
    rng = np.random.default_rng(0)
    rejections = sum(
        not st.two_sample_verdict(_ensemble(rng, 60), _ensemble(rng, 60), alpha=0.05,
                                  permutations=199, seed=i)["passed"]
        for i in range(200))
    # Binomial(200, 0.05): mean 10; above 19 has probability under 0.5%.
    assert rejections <= 19


def test_two_sample_detects_a_shifted_mean():
    rng = np.random.default_rng(1)
    v = st.two_sample_verdict(_ensemble(rng, 2000), _ensemble(rng, 2000, shift=0.08),
                              alpha=1e-3, permutations=999, seed=5)
    assert v["passed"] is False and v["p_value"] <= 1e-3
    assert v["stopped_early"] is False and v["permutations"] == 999
    assert v["n_flagged"]["mean"] > 0 and v["z_max"] > v["critical_z"]
    assert v["worst"]["statistic"] == "mean"


def test_two_sample_detects_a_variance_change_at_the_same_mean():
    rng = np.random.default_rng(2)
    n, n_t = 4000, 6
    a = rng.poisson(20.0, (n, n_t, 1)).astype(float)
    # Same mean, twice the variance: a negative binomial with mean 20, var 40.
    b = rng.negative_binomial(20, 0.5, (n, n_t, 1)).astype(float)
    v = st.two_sample_verdict(a, b, alpha=1e-3, permutations=999, seed=3)
    assert v["passed"] is False
    assert v["worst"]["statistic"] == "variance"
    assert v["z_max_mean"] < 4 < v["z_max_variance"]


def test_two_sample_stops_early_when_far_from_significant_and_is_reproducible():
    rng = np.random.default_rng(3)
    a, b = _ensemble(rng, 500), _ensemble(rng, 500)
    v1 = st.two_sample_verdict(a, b, alpha=1e-3, permutations=1999, seed=11)
    v2 = st.two_sample_verdict(a, b, alpha=1e-3, permutations=1999, seed=11)
    assert v1 == v2
    assert v1["passed"] and v1["stopped_early"] and v1["permutations"] < 1999
    # Besag-Clifford: stopped at the h-th exceedance, p = h / k.
    assert v1["p_value"] == pytest.approx(20 / v1["permutations"])


def test_two_sample_skips_constant_cells_and_two_valued_cells_for_the_variance():
    rng = np.random.default_rng(4)
    n = 300
    a = np.zeros((n, 4, 2))
    b = np.zeros((n, 4, 2))
    a[:, 1:, 0] = rng.integers(0, 2, (n, 3))  # a 0/1 promoter
    b[:, 1:, 0] = rng.integers(0, 2, (n, 3))
    a[:, 1:, 1] = rng.poisson(3, (n, 3))
    b[:, 1:, 1] = rng.poisson(3, (n, 3))
    v = st.two_sample_verdict(a, b, alpha=1e-3, permutations=199, seed=0,
                              times=np.arange(4.0), species=["P", "M"])
    # t = 0 is constant (2 cells); the promoter's 3 cells take two values.
    assert v["n_cells"] == 8
    assert v["n_tested"] == {"mean": 6, "variance": 3}
    assert v["worst"]["species"] in ("P", "M") and "time" in v["worst"]


def test_two_sample_of_identical_constant_ensembles_passes_and_of_different_fails():
    a = np.ones((30, 3, 2))
    assert st.two_sample_verdict(a, a.copy(), alpha=1e-3, permutations=99, seed=0)["passed"]
    # Perfectly separated groups: the most extreme split there is, which a random
    # relabelling of 30 + 30 reproduces with probability 2 / C(60, 30). 99
    # relabellings cannot show p <= 1e-3 (the smallest p they give is 1/100); 1999 can.
    assert st.two_sample_verdict(a, a + 1, alpha=1e-3, permutations=99, seed=0)["p_value"] == 0.01
    v = st.two_sample_verdict(a, a + 1, alpha=1e-3, permutations=1999, seed=0)
    assert not v["passed"] and v["p_value"] == 1 / 2000


def test_two_sample_refuses_mismatched_grids():
    v = st.two_sample_verdict(np.zeros((5, 3, 2)), np.zeros((5, 4, 2)), alpha=1e-3,
                              permutations=9, seed=0)
    assert v["passed"] is False and "grids differ" in v["reason"]


def test_parity_core_verdict_is_reported_beside_it():
    rng = np.random.default_rng(5)
    v = st.parity_core_verdict(_ensemble(rng, 200), _ensemble(rng, 200))
    assert set(v) == {"passed", "frac_pass", "max_z", "k"}


# --------------------------------------------------------------------------- #
# One ensemble against exact moments
# --------------------------------------------------------------------------- #
def test_analytic_false_alarm_rate_is_near_alpha_at_dsmts_size():
    rng = np.random.default_rng(6)
    lam = np.full((5, 2), 4.0)
    rejections = sum(
        not st.analytic_verdict(rng.poisson(4.0, (10_000, 5, 2)).astype(float), lam, lam,
                                alpha=0.05)["passed"]
        for _ in range(100))
    assert rejections <= 12  # Binomial(100, 0.05): above 12 has probability under 0.5%


def test_analytic_detects_a_wrong_mean_and_a_wrong_variance():
    rng = np.random.default_rng(7)
    mu = np.full((5, 1), 10.0)
    x = rng.poisson(10.3, (10_000, 5, 1)).astype(float)
    v = st.analytic_verdict(x, mu, mu, alpha=1e-3)
    assert v["passed"] is False and v["worst"]["statistic"] == "mean"
    assert v["p_value"] <= 1e-3  # the family p-value, comparable to alpha
    # Right mean, variance 20 instead of 10.
    y = rng.negative_binomial(10, 0.5, (10_000, 5, 1)).astype(float)
    w = st.analytic_verdict(y, mu, mu, alpha=1e-3)
    assert w["passed"] is False and w["worst"]["statistic"] == "variance"


def test_analytic_holds_a_deterministic_cell_to_its_exact_value():
    x = np.zeros((10_000, 2, 1))
    x[:, 1, 0] = np.random.default_rng(8).poisson(2.0, 10_000)
    mean = np.array([[0.0], [2.0]])
    var = np.array([[0.0], [2.0]])
    ok = st.analytic_verdict(x, mean, var, alpha=1e-3)
    assert ok["passed"] and ok["p_value"] == pytest.approx(min(1.0, 2 * ok["p_min"]))
    x[17, 0, 0] = 1.0  # one trajectory leaves the initial condition
    v = st.analytic_verdict(x, mean, var, alpha=1e-3)
    assert v["passed"] is False and v["n_deterministic_failed"] == 1
