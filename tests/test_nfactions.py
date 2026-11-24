"""Reading a network-free protocol from BNGL actions, and writing it back for BNG2.pl."""

from __future__ import annotations

import pytest

from bench import nfactions

CAMKII = r"""
begin model
begin parameters
  k 1
end parameters
end model

begin actions
  # Autophosphorylation of a twelve-subunit holoenzyme
  simulate({method=>"nf",suffix=>"nfr",t_start=>0,t_end=>6,n_steps=>60,\
    seed=>1234,print_functions=>1,get_final_state=>0})
end actions
"""

TWO_PHASE = """
generate_network({overwrite=>1})
setParameter("lig", 0)
simulate({method=>"nf",t_end=>100,n_steps=>10,gml=>500})  # equilibrate
setParameter("lig",1)
simulate_nf({t_end=>160,n_steps=>6,gml=>500,seed=>3})
"""


def test_own_actions_reads_the_files_action_block():
    text = nfactions.own_actions(CAMKII)
    assert "simulate(" in text and "begin model" not in text


def test_a_single_run_is_one_phase_with_the_files_options():
    proto = nfactions.protocol(nfactions.own_actions(CAMKII))
    assert proto.single and len(proto.phases) == 1
    p = proto.phases[0]
    assert (p.duration, p.n_steps, p.n_points, p.get_final_state) == (6.0, 60, 61, False)
    assert proto.complex and proto.gml is None and proto.utl is None
    assert proto.n_points == 61


def test_a_two_phase_protocol_keeps_its_parameter_changes_and_clock():
    proto = nfactions.protocol(TWO_PHASE)
    assert not proto.single and proto.dropped == ("generate_network",)
    first, second = proto.phases
    assert [(s.name, s.value) for s in first.sets] == [("lig", "0")]
    assert [(s.name, s.value) for s in second.sets] == [("lig", "1")]
    # The second run starts where the first ended (no t_start): 100 -> 160.
    assert (first.duration, second.duration) == (100.0, 60.0)
    assert first.get_final_state and proto.gml == 500
    assert proto.n_points == 11 + 7


def test_render_replaces_each_steps_seed_and_suffix_and_keeps_the_rest():
    proto = nfactions.protocol(TWO_PHASE)
    text = nfactions.render(proto, [11, 22], suffix="nfp")
    lines = text.strip().splitlines()
    assert lines[0] == 'setParameter("lig",0)'
    assert lines[1].startswith('simulate({method=>"nf",t_end=>100,n_steps=>10,gml=>500,')
    assert lines[1].endswith('suffix=>"nfp0",seed=>11})')
    assert lines[3] == 'simulate_nf({t_end=>160,n_steps=>6,gml=>500,suffix=>"nfp1",seed=>22})'
    assert "generate_network" not in text and "seed=>3" not in text
    # Rendering reads back as the same protocol.
    again = nfactions.protocol(text)
    assert again.describe() == {**proto.describe(), "dropped": []}


def test_an_expression_value_is_quoted_for_bngl():
    step = nfactions.SetParameter("f", "2*k")
    assert step.render() == 'setParameter("f","2*k")'


@pytest.mark.parametrize("actions, message", [
    ('simulate({method=>"ode",t_end=>1,n_steps=>1})', "not network-free"),
    ('simulate({method=>"nf",t_end=>1,n_steps=>1,equil=>5})', "equil"),
    ('parameter_scan({method=>"nf",parameter=>"k",par_min=>1})', "parameter_scan"),
    ('setConcentration("A()",1)\nsimulate_nf({t_end=>1,n_steps=>1})', "setConcentration"),
    ('simulate({method=>"nf",t_end=>1})', "needs t_end and n_steps"),
    ('simulate({method=>"nf",t_end=>1,n_steps=>0})', "positive integer"),
    ('simulate({method=>"nf",t_start=>5,t_end=>1,n_steps=>2})', "greater than t_start"),
    ('simulate_nf({t_end=>1,n_steps=>1,gml=>5})\nsimulate_nf({t_end=>2,n_steps=>1})',
     "gml differs"),
    ('writeXML()', "runs no network-free simulation"),
    ('setParameter("k")', "a name and a value"),
])
def test_what_cannot_be_replayed_is_refused_by_name(actions, message):
    with pytest.raises(nfactions.ActionError, match=message):
        nfactions.protocol(actions)


def test_param_passes_the_flags_every_engine_can_follow():
    # -bscb is -cb in NFsim 1.14.3: both block same-complex binding (complex).
    p = nfactions.protocol('simulate({method=>"nf",t_end=>10,n_steps=>5,complex=>0,'
                           'param=>"-bscb -utl 5"})')
    assert p.complex and p.utl == 5
    assert nfactions.protocol('simulate_nf({t_end=>1,n_steps=>1,param=>"-cb"})').complex
    assert not nfactions.protocol('simulate_nf({t_end=>1,n_steps=>1,complex=>0})').complex
    # BNG2.pl gets the string as written.
    step = p.phases[0].step.render(seed=3, suffix="x")
    assert 'param=>"-bscb -utl 5"' in step and "seed=>3" in step


@pytest.mark.parametrize("param, message", [
    ('"-bscb -gml 5"', "-gml"),
    ('"-utl"', "needs a value"),
    ('"-utl 0"', "positive integer"),
    ('"-v -bscb"', "'-v'"),
])
def test_param_refuses_flags_some_engine_cannot_follow(param, message):
    with pytest.raises(nfactions.ActionError, match=message):
        nfactions.protocol(f'simulate_nf({{t_end=>1,n_steps=>1,param=>{param}}})')


def test_param_and_utl_must_agree_and_hold_for_every_step():
    with pytest.raises(nfactions.ActionError, match="disagree"):
        nfactions.protocol('simulate_nf({t_end=>1,n_steps=>1,utl=>4,param=>"-utl 5"})')
    with pytest.raises(nfactions.ActionError, match="param differs"):
        nfactions.protocol('simulate_nf({t_end=>1,n_steps=>1,param=>"-bscb"})\n'
                           'simulate_nf({t_end=>2,n_steps=>1})')


def test_a_file_without_actions_is_refused():
    with pytest.raises(nfactions.ActionError, match="no action block"):
        nfactions.own_actions("begin parameters\n k 1\nend parameters\n")
