"""The .net reader's right-hand side, conservation laws and parameter overrides
(bench.netfile), which the steady-state capability's engine-free checks stand on.
Hand-written networks with known answers; no simulator."""

from __future__ import annotations

import numpy as np
import pytest

from bench import netfile

# A + B <-> C, with the totals of A and B set by parameters: two conservation laws.
BINDING = """# Created by BioNetGen 2.9.3
begin parameters
    1 kf                1.0  # Constant
    2 kr                1.0  # Constant
    3 A0                2.0  # Constant
    4 B0                1.0  # Constant
    5 half_B0           B0/2  # ConstantExpression
end parameters
begin species
    1 A() A0
    2 B() B0
    3 C() 0
end species
begin reactions
    1 1,2 3 kf #bind
    2 3 1,2 kr #unbind
end reactions
begin groups
    1 Atot                 1,3
    2 Btot                 2,3
    3 twiceC               2*3
end groups
"""

# The genetic toggle switch of Gardner et al. (2000), as BioNetGen 2.9.3 writes it.
TOGGLE = """# Created by BioNetGen 2.9.3
begin parameters
    1 alpha_1     156.25  # Constant
    2 alpha_2     15.6  # Constant
    3 beta        2.5  # Constant
    4 gamma       1.0  # Constant
    5 k_deg       1.0  # Constant
end parameters
begin functions
    1 syn_R1() alpha_1/(1+(Obs_Tot_R2^beta))
    2 syn_R2() alpha_2/(1+(Obs_Tot_R1^gamma))
end functions
begin species
    1 R1() 0
    2 R2() 0
end species
begin reactions
    1 0 1 syn_R1 #_R1
    2 0 2 syn_R2 #_R2
    3 1 0 k_deg #_R3
    4 2 0 k_deg #_R4
end reactions
begin groups
    1 Obs_Tot_R1           1
    2 Obs_Tot_R2           2
end groups
"""


def test_mass_action_rhs_multiplies_the_rate_by_each_reactant():
    net = netfile.parse(BINDING)
    x = np.array([2.0, 1.0, 0.5])
    bind, unbind = 1.0 * 2.0 * 1.0, 1.0 * 0.5
    np.testing.assert_allclose(net.rates(x), [bind, unbind])
    np.testing.assert_allclose(net.rhs(x), [unbind - bind, unbind - bind, bind - unbind])
    f, g = net.flux(x)
    np.testing.assert_allclose(f, net.rhs(x))
    np.testing.assert_allclose(g, [bind + unbind] * 3)


def test_a_repeated_reactant_counts_twice_and_a_statistical_factor_is_kept():
    net = netfile.parse("""begin parameters
 1 k 3
end parameters
begin species
 1 A() 4
 2 D() 0
end species
begin reactions
 1 1,1 2 0.5*k
end reactions
""")
    assert net.rates([4.0, 0.0]) == pytest.approx([0.5 * 3 * 16])
    np.testing.assert_allclose(net.rhs([4.0, 0.0]), [-2 * 24.0, 24.0])


def test_functional_rates_read_observables_and_other_functions():
    net = netfile.parse(TOGGLE)
    u, v = 3.0, 2.0
    np.testing.assert_allclose(
        net.rhs([u, v]), [156.25 / (1 + v ** 2.5) - u, 15.6 / (1 + u) - v])
    chained = netfile.parse(TOGGLE.replace(
        "2 syn_R2() alpha_2/(1+(Obs_Tot_R1^gamma))",
        "2 syn_R2() 0.5*syn_R1()+Obs_Tot_R1"))
    assert chained.rhs([u, v])[1] == pytest.approx(0.5 * 156.25 / (1 + v ** 2.5) + u - v)


def test_observables_take_their_weights():
    net = netfile.parse(BINDING)
    assert net.observables([2.0, 1.0, 0.5]) == {"Atot": 2.5, "Btot": 1.5, "twiceC": 1.0}


def test_a_fixed_species_never_changes_but_still_drives_its_reactions():
    net = netfile.parse("""begin parameters
 1 k 2
end parameters
begin species
 1 $Src() 5
 2 X() 0
end species
begin reactions
 1 1 1,2 k
end reactions
""")
    np.testing.assert_allclose(net.rhs([5.0, 0.0]), [0.0, 10.0])
    assert net.stoichiometry()[0].tolist() == [0.0]


@pytest.mark.parametrize("rate", ["MM kcat Km", "Sat kcat Km", "0.5*Hill k K n"])
def test_rate_laws_the_reader_does_not_evaluate_fail_only_when_asked(rate):
    net = netfile.parse(f"""begin parameters
 1 kcat 1
 2 Km 1
end parameters
begin species
 1 S() 1
 2 P() 0
end species
begin reactions
 1 1 2 {rate}
end reactions
""")
    assert len(net.reactions) == 1  # reading never fails on it
    with pytest.raises(netfile.NetFileError, match="does not evaluate"):
        net.rhs([1.0, 0.0])


def test_a_local_function_is_read_but_not_evaluated():
    net = netfile.parse(TOGGLE.replace("1 syn_R1() alpha_1", "1 syn_R1(x) alpha_1"))
    assert net.function_bodies["syn_R1"] is None
    with pytest.raises(netfile.NetFileError, match="local function"):
        net.rhs([1.0, 1.0])


def test_unreadable_groups_do_not_stop_the_reader():
    net = netfile.parse(TOGGLE.replace("1 Obs_Tot_R1           1", "1 Obs_Tot_R1   x*1"))
    assert net.species == ["R1()", "R2()"] and net.groups_error
    with pytest.raises(netfile.NetFileError, match="bad group line"):
        net.rhs([1.0, 1.0])


def test_overrides_set_a_parameter_and_everything_read_from_it():
    net = netfile.parse(BINDING, {"B0": 4.0, "kr": 3.0})
    assert net.parameters["half_B0"] == 2.0
    assert net.initial == [2.0, 4.0, 0.0]
    assert net.rates([1.0, 1.0, 1.0])[1] == 3.0
    assert net.is_primary("B0") and not net.is_primary("half_B0")
    with pytest.raises(netfile.NetFileError, match="no parameter named 'nope'"):
        netfile.parse(BINDING, {"nope": 1.0})


def test_with_parameters_rewrites_the_number_and_keeps_the_rest():
    text = netfile.with_parameters(BINDING, {"A0": 0.001, "kf": 1e3})
    assert "    3 A0                0.001  # Constant\n" in text
    assert "    1 kf                1000.0  # Constant\n" in text
    assert netfile.parse(text).initial == [0.001, 1.0, 0.0]
    assert text.replace("0.001", "2.0").replace("1000.0", "1.0") == BINDING
    with pytest.raises(netfile.NetFileError, match="is an expression"):
        netfile.with_parameters(BINDING, {"half_B0": 1.0})
    with pytest.raises(netfile.NetFileError, match="no parameter named"):
        netfile.with_parameters(BINDING, {"nope": 1.0})


def test_conservation_laws_span_what_no_reaction_changes():
    net = netfile.parse(BINDING)
    laws, moving = net.subspaces()
    s = net.stoichiometry()
    assert laws.shape == (2, 3) and moving.shape == (3, 1)
    np.testing.assert_allclose(laws @ s, 0.0, atol=1e-12)
    np.testing.assert_allclose(laws @ laws.T, np.eye(2), atol=1e-12)
    # Total A (A + C) and total B (B + C) are in the span of the laws.
    for total in ([1.0, 0.0, 1.0], [0.0, 1.0, 1.0]):
        np.testing.assert_allclose(laws.T @ (laws @ total), total, atol=1e-12)
    # A network with more reactions than species takes the other SVD branch.
    wide = netfile.parse(BINDING.replace("    2 3 1,2 kr #unbind\n",
                                         "    2 3 1,2 kr #unbind\n    3 3 1,2 kr #again\n"
                                         "    4 1,2 3 kf #again\n"))
    assert wide.conservation_laws().shape == (2, 3)


def test_a_fixed_or_untouched_species_is_a_law_of_its_own():
    net = netfile.parse("""begin parameters
 1 k 1
end parameters
begin species
 1 $Src() 1
 2 X() 0
 3 Idle() 7
end species
begin reactions
 1 1 1,2 k
 2 2 0 k
end reactions
""")
    laws = net.conservation_laws()
    assert laws.shape == (2, 3)
    for unit in ([1.0, 0, 0], [0, 0, 1.0]):
        np.testing.assert_allclose(laws.T @ (laws @ unit), unit, atol=1e-12)


def test_pure_sinks_are_produced_and_never_consumed():
    net = netfile.parse("""begin parameters
 1 k 1
end parameters
begin species
 1 A() 1
 2 Deg() 0
 3 $Src() 1
end species
begin reactions
 1 3 3,1 k
 2 1 2 k
end reactions
""")
    assert net.pure_sinks() == [1]


def test_jacobian_by_differences_matches_the_analytic_one():
    net = netfile.parse(BINDING)
    x = np.array([1.5, 0.5, 0.5])
    a, b, _ = x
    want = np.array([[-b, -a, 1.0], [-b, -a, 1.0], [b, a, -1.0]])
    np.testing.assert_allclose(net.jacobian(x), want, rtol=1e-5, atol=1e-6)
