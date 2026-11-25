"""A ``.net`` read as written (bench.netfile.structure): which parameters are
numbers and which reach a reaction rate, an initial amount or a compartment size,
through derived parameters and functions; the observables' weights; and the file
rewritten with one parameter changed. Synthetic networks only: no simulator."""

from __future__ import annotations

import numpy as np
import pytest

from bench import netfile

NET = """# Created by BioNetGen 2.9.3
begin parameters
    1 NA        6.022e23  # Constant
    2 V         1e-12  # Constant
    3 kon_M     1e6  # Constant
    4 kon       kon_M/(NA*V)  # ConstantExpression
    5 koff      0.1  # Constant
    6 L_conc    1e-9  # Constant
    7 L0        L_conc*NA*V  # ConstantExpression
    8 R0        1000  # Constant
    9 Km        50  # Constant
   10 unused    3  # Constant
   11 Stim      0  # Constant
   12 vol       2.5  # Constant
end parameters
begin compartments
    1 C  3  vol
end compartments
begin species
    1 L() L0
    2 R() R0
    3 L.R() 0
end species
begin reactions
    1 1,2 3 kon #bind
    2 3 1,2 koff #unbind
    3 3 2 f1 #f
end reactions
begin groups
    1 Rtot    2,3
    2 Lbound  2*3
end groups
begin functions
    1 f1() Stim*koff/(Km+Rtot)
end functions
"""


def test_structure_reads_literals_and_expressions_as_written():
    st = netfile.structure(NET)
    assert st.literal("kon_M") == 1e6 and st.literal("kon") is None
    assert list(st.primary()) == ["NA", "V", "kon_M", "koff", "L_conc", "R0", "Km", "unused",
                                  "Stim", "vol"]
    assert st.species == ["L()", "R()", "L.R()"] and st.initial == ["L0", "R0", "0"]
    assert st.compartments == {"C": "vol"}


def test_structure_follows_derived_parameters_and_functions_to_rates_and_initial_amounts():
    st = netfile.structure(NET)
    rates = st.reaching_rates()
    # kon through kon_M/(NA*V); Km and Stim through the function f1.
    assert {"kon", "kon_M", "NA", "V", "koff", "Km", "Stim"} <= rates
    assert not {"L_conc", "R0", "unused", "vol"} & rates
    assert st.reaching_initial() == {"L0", "L_conc", "NA", "V", "R0"}
    assert st.sizing_compartments() == {"vol"}


def test_observables_are_read_with_their_weights():
    names, w = netfile.structure(NET).observable_weights()
    assert names == ["Rtot", "Lbound"]
    np.testing.assert_array_equal(w, [[0, 1, 1], [0, 0, 2]])


def test_an_exponent_is_not_a_name():
    assert netfile.names_in("2.1e-9*f + 3E+2*g") == {"f", "g"}


def test_with_parameters_rewrites_each_line_named_and_nothing_else():
    out = netfile.with_parameters(NET, {"koff": 0.1 * 1.001, "Km": 25.0})
    changed = [(a, b) for a, b in zip(NET.splitlines(), out.splitlines()) if a != b]
    assert [b for _, b in changed] == [f"    5 koff      {0.1 * 1.001!r}  # Constant",
                                       "    9 Km        25.0  # Constant"]
    st = netfile.structure(out)
    assert st.literal("koff") == pytest.approx(0.1001) and st.literal("Km") == 25.0
    # Only the parameters block: the rate "koff" of reaction 2 is untouched.
    assert "    2 3 1,2 koff #unbind" in out
    with pytest.raises(netfile.NetFileError, match="no parameter named 'nope'"):
        netfile.with_parameters(NET, {"nope": 1.0})
    with pytest.raises(netfile.NetFileError, match="not a finite number"):
        netfile.with_parameters(NET, {"koff": float("nan")})


def test_a_derived_parameter_is_overridden_only_when_asked():
    with pytest.raises(netfile.NetFileError, match="is an expression"):
        netfile.with_parameters(NET, {"kon": 2.0})
    out = netfile.with_parameters(NET, {"kon": 2.0}, allow_derived=True)
    assert "    4 kon       2.0  # ConstantExpression" in out.splitlines()
    st = netfile.structure(out)
    assert st.literal("kon") == 2.0 and "kon_M" not in st.reaching_rates()


def test_with_initial_amounts_sets_every_species_and_nothing_else():
    out = netfile.with_initial_amounts(NET, [1.5, 2.0, 0.25])
    st = netfile.structure(out)
    assert st.initial == ["1.5", "2.0", "0.25"]
    assert out.replace("1.5", "L0").replace("2.0", "R0").replace("0.25", "0") == NET
    with pytest.raises(netfile.NetFileError, match="more species than the 2 amounts given"):
        netfile.with_initial_amounts(NET, [1.0, 2.0])
    with pytest.raises(netfile.NetFileError, match="4 amounts given for 3 species"):
        netfile.with_initial_amounts(NET, [1.0, 2.0, 3.0, 4.0])
    with pytest.raises(netfile.NetFileError, match="not a finite number"):
        netfile.with_initial_amounts(NET, [1.0, float("inf"), 0.0])


def test_an_initial_amount_may_be_an_expression_the_engine_evaluates():
    out = netfile.with_initial_amounts(NET, [1.5, "2*R0", 0.0])
    assert netfile.structure(out).initial == ["1.5", "2*R0", "0.0"]
    assert netfile.parse(out).initial[1] == 2000.0
    assert netfile.parse(out, {"R0": 10.0}).initial[1] == 20.0
    for bad in ("2 * R0", "", "R0#x"):
        with pytest.raises(netfile.NetFileError, match="bad initial-amount expression"):
            netfile.with_initial_amounts(NET, [1.0, bad, 0.0])


def test_added_parameters_follow_the_block_and_are_derived_from_it():
    out = netfile.with_added_parameters(NET, {"_bench_reset_0": "2*R0", "_b1": "L0+1"})
    assert "   13 _bench_reset_0 2*R0\n" in out and "   14 _b1 L0+1\n" in out
    net = netfile.parse(netfile.with_parameters(out, {"R0": 7.0}))
    assert net.parameters["_bench_reset_0"] == 14.0
    assert out.replace("    13 _bench_reset_0 2*R0\n    14 _b1 L0+1\n", "") == NET
    with pytest.raises(netfile.NetFileError, match="already has parameters named R0"):
        netfile.with_added_parameters(NET, {"R0": "1"})
    with pytest.raises(netfile.NetFileError, match="bad parameter expression"):
        netfile.with_added_parameters(NET, {"x": "1 + R0"})


def test_a_network_reads_the_clock_only_through_a_rate():
    assert not netfile.structure(NET).reads_clock()
    clocked = NET.replace("Stim*koff/(Km+Rtot)", "Stim*koff*time()")
    assert netfile.structure(clocked).reads_clock()
