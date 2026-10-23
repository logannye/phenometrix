import copy
import json
import numpy as np
import pytest
from phenometrix_prosody.subharmonic_experiment import SEEDS,CANDIDATE,key,natural_decision,analytic_decision,save
from phenometrix_prosody.selector_experiment import CONDITIONS,SYNTHETIC_IDS
from phenometrix_prosody.harmonic_challenge import challenge_spec as previous_spec
from phenometrix_prosody.subharmonic_challenge import challenge_spec

def natural():
    def row(error,success=.8):return {'macro':{'joint_error':error,'reference_voiced_success_rate':success},'per_condition':{c:{'joint_error':error} for c in CONDITIONS}}
    d={'original_mlp':row(.1),'original_hybrid':row(.11)}
    for seed in SEEDS:d[key(CANDIDATE,seed)]=row(.094,.81);d[key('set',seed)]=row(.097,.805)
    return d

def test_half_point_against_both_baselines_unchanged():
    d=natural();assert natural_decision(d)['pass']
    d['original_hybrid']['macro']['joint_error']=.098
    assert not natural_decision(d)['checks']['mean_improves_both_by_half_point']
    d=natural()
    for s in SEEDS:d[key(CANDIDATE,s)]['macro']['joint_error']=.0951
    assert not natural_decision(d)['pass']

@pytest.mark.parametrize('component',['seed','condition','voiced'])
def test_every_natural_component_retained(component):
    d=natural()
    if component=='seed':d[key(CANDIDATE,SEEDS[0])]['macro']['joint_error']=.1
    elif component=='condition':
        for s in SEEDS:d[key(CANDIDATE,s)]['per_condition']['clip01']['joint_error']=.111
    else:
        for s in SEEDS:d[key(CANDIDATE,s)]['macro']['reference_voiced_success_rate']=.799
    assert not natural_decision(d)['pass']

def analytics():
    def case(kind):return {'kind':kind,'methods':{key(CANDIDATE,s):{'all':{'success_within_50c_all_frames':1.},'regions':{'after-observation-change':{'success_within_50c_all_frames':1.}}} for s in SEEDS}}
    reg={n:case('constant' if n.startswith('constant') else 'observation-change' if n.startswith('observation') else 'step' if n.startswith('step') else 'glide') for n in SYNTHETIC_IDS}
    consumed={r['id']:case('observation-change') for r in previous_spec()['records']}
    fresh={r['id']:case('observation-change') for r in challenge_spec()['records']}
    return reg,consumed,fresh

@pytest.mark.parametrize('group,case_name,flag',[(0,'constant-300','stationary_pass'),(0,'observation-change-300','harmonic_change_pass'),(1,'harmonic-balance-f3-p0','consumed_challenge_pass'),(2,'subharmonic-balance-f5-p1','challenge_pass')])
def test_any_failed_case_seed_blocks_gate(group,case_name,flag):
    groups=analytics();assert all(v for k,v in analytic_decision(*groups).items() if k.endswith('_pass'))
    row=groups[group][case_name]['methods'][key(CANDIDATE,SEEDS[1])]
    metric=row['all'] if flag=='stationary_pass' else row['regions']['after-observation-change']
    metric['success_within_50c_all_frames']=.9499
    assert not analytic_decision(*groups)[flag]
    metric['success_within_50c_all_frames']=.95
    assert analytic_decision(*groups)[flag]

@pytest.mark.parametrize('group',[0,1,2])
def test_missing_regression_or_challenge_not_silently_dropped(group):
    groups=analytics();groups[group].pop(next(iter(groups[group])))
    with pytest.raises(ValueError,match='Incomplete'):analytic_decision(*groups)

def test_numpy_boolean_decision_serializes_without_changing_values(tmp_path):
    p=tmp_path/'metrics.json';value={'pass':np.bool_(False),'ratio':np.float64(.95)}
    save(p,value);assert json.loads(p.read_text())=={'pass':False,'ratio':.95}
    with pytest.raises(ValueError):save(p,{'bad':float('nan')})
    with pytest.raises(TypeError):save(p,{'bad':np.arange(3)})
