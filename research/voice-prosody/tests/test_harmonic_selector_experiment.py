import copy
import pytest
from phenometrix_prosody.harmonic_selector_experiment import (
    SEEDS, CANDIDATE, key, natural_decision, analytic_decision)
from phenometrix_prosody.selector_experiment import CONDITIONS

def natural_fixture():
    def row(error,success=.8):
        return {'macro':{'joint_error':error,'reference_voiced_success_rate':success},
                'per_condition':{c:{'joint_error':error} for c in CONDITIONS}}
    result={'original_mlp':row(.1),'original_hybrid':row(.11)}
    for seed in SEEDS:
        result[key(*CANDIDATE,seed)]=row(.094,.81)
        result[key('absolute','acceptable-set',seed)]=row(.097,.805)
    return result

def test_natural_gate_requires_half_point_against_each_frozen_baseline():
    r=natural_fixture(); assert natural_decision(r)['pass']
    r['original_hybrid']['macro']['joint_error']=.098
    assert not natural_decision(r)['checks']['mean_improves_both_by_half_point']
    r=natural_fixture()
    for seed in SEEDS:r[key(*CANDIDATE,seed)]['macro']['joint_error']=.0951
    assert not natural_decision(r)['pass']

@pytest.mark.parametrize('failure',['seed','condition','voiced_success'])
def test_old_gate_components_cannot_be_hidden_by_other_improvements(failure):
    r=natural_fixture();n=key(*CANDIDATE,SEEDS[0])
    if failure=='seed':
        for seed in SEEDS:r[key(*CANDIDATE,seed)]['macro']['joint_error']=.08
        r[n]['macro']['joint_error']=.101
        field='each_seed_improves_original_mlp'
    elif failure=='condition':
        for seed in SEEDS:r[key(*CANDIDATE,seed)]['per_condition']['clip01']['joint_error']=.1101
        field='no_condition_worsens_over_one_point'
    else:
        for seed in SEEDS:r[key(*CANDIDATE,seed)]['macro']['reference_voiced_success_rate']=.7999
        field='voiced_success_not_lower'
    d=natural_decision(r);assert not d['pass'];assert not d['checks'][field]

def analytic_fixture():
    def case(kind):
        methods={key(*CANDIDATE,s):{'all':{'success_within_50c_all_frames':1.},
                 'regions':{'after-observation-change':{'success_within_50c_all_frames':1.}}} for s in SEEDS}
        return {'kind':kind,'methods':methods}
    reg={f'constant-{f}':case('constant') for f in (100,200,300)}
    reg.update({f'observation-change-{f}':case('observation-change') for f in (100,200,300)})
    challenge={f'new-case-{i}':case('observation-change') for i in range(12)}
    return reg,challenge

def test_harmonic_failure_is_gated_per_seed_per_postchange_region():
    reg,challenge=analytic_fixture();assert analytic_decision(reg,challenge)['harmonic_change_pass']
    reg['observation-change-300']['methods'][key(*CANDIDATE,SEEDS[1])]['regions']['after-observation-change']['success_within_50c_all_frames']=0.
    got=analytic_decision(reg,challenge)
    assert got['stationary_pass'] and got['challenge_pass']
    assert not got['harmonic_change_pass']

def test_prospective_case_cannot_be_hidden_in_pooled_success():
    reg,challenge=analytic_fixture()
    challenge['new-case-11']['methods'][key(*CANDIDATE,SEEDS[2])]['regions']['after-observation-change']['success_within_50c_all_frames']=.9499
    assert not analytic_decision(reg,challenge)['challenge_pass']
    del challenge['new-case-11']
    with pytest.raises(ValueError,match='Incomplete'):analytic_decision(reg,challenge)

def test_stationary_gate_still_requires_all_seeds():
    reg,challenge=analytic_fixture()
    reg['constant-300']['methods'][key(*CANDIDATE,SEEDS[2])]['all']['success_within_50c_all_frames']=.94
    assert not analytic_decision(reg,challenge)['stationary_pass']
