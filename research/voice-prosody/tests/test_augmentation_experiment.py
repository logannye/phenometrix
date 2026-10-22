import copy
import json
import numpy as np
import pytest
from phenometrix_prosody.augmentation_experiment import SEEDS,CANDIDATE,key,natural_decision,analytic_decision,save
from phenometrix_prosody.selector_experiment import CONDITIONS,SYNTHETIC_IDS
from phenometrix_prosody.harmonic_challenge import challenge_spec as previous_spec
from phenometrix_prosody.subharmonic_challenge import challenge_spec as sub_spec
from phenometrix_prosody.augmentation_cases import challenge_spec

def natural():
    def row(error,success=.8):return {'macro':{'joint_error':error,'reference_voiced_success_rate':success},'per_condition':{c:{'joint_error':error} for c in CONDITIONS}}
    d={'original_mlp':row(.1),'original_hybrid':row(.11)}
    for seed in SEEDS:d[key(CANDIDATE,seed)]=row(.094,.81);d[key('natural-only',seed)]=row(.097,.805)
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
    consumed_sub={r['id']:case('observation-change') for r in sub_spec()['records']}
    return reg,consumed,consumed_sub,fresh

@pytest.mark.parametrize('group,case_name,flag',[(0,'constant-300','stationary_pass'),(0,'observation-change-300','harmonic_change_pass'),(1,'harmonic-balance-f3-p0','consumed_challenge_pass'),(2,'subharmonic-balance-f3-p0','consumed_subharmonic_pass'),(3,'augmentation-balance-f5-p1','challenge_pass')])
def test_any_failed_case_seed_blocks_gate(group,case_name,flag):
    groups=analytics();assert all(v for k,v in analytic_decision(*groups).items() if k.endswith('_pass'))
    row=groups[group][case_name]['methods'][key(CANDIDATE,SEEDS[1])]
    metric=row['all'] if flag=='stationary_pass' else row['regions']['after-observation-change']
    metric['success_within_50c_all_frames']=.9499
    assert not analytic_decision(*groups)[flag]
    metric['success_within_50c_all_frames']=.95
    assert analytic_decision(*groups)[flag]

@pytest.mark.parametrize('group',[0,1,2,3])
def test_missing_regression_or_challenge_not_silently_dropped(group):
    groups=analytics();groups[group].pop(next(iter(groups[group])))
    with pytest.raises(ValueError,match='Incomplete'):analytic_decision(*groups)

def test_numpy_boolean_decision_serializes_without_changing_values(tmp_path):
    p=tmp_path/'metrics.json';value={'pass':np.bool_(False),'ratio':np.float64(.95)}
    save(p,value);assert json.loads(p.read_text())=={'pass':False,'ratio':.95}
    with pytest.raises(ValueError):save(p,{'bad':float('nan')})
    with pytest.raises(TypeError):save(p,{'bad':np.arange(3)})


def test_matched_synthetic_schedule_no_replacement_and_independent_of_natural_rng():
    from phenometrix_prosody.augmentation_experiment import synthetic_order
    for seed in SEEDS:
        seen=[]
        for epoch in range(12):
            ix=synthetic_order(seed,epoch)
            assert ix.shape==(33,512) and len(np.unique(ix))==16896
            assert ix.min()>=0 and ix.max()<18336
            assert np.array_equal(ix,synthetic_order(seed,epoch))
            assert not np.array_equal(ix.ravel(),np.random.default_rng(seed+epoch).permutation(18336)[:16896])
            seen.append(ix)
        assert not np.array_equal(seen[0],seen[1])
    with pytest.raises(ValueError):synthetic_order(SEEDS[0],0,100)


def test_additive_synthetic_term_does_not_reweight_natural_only_loss():
    import torch
    from types import SimpleNamespace
    from phenometrix_prosody.augmentation_experiment import augmentation_loss
    from phenometrix_prosody.selector_model import PitchTargets
    natural=torch.tensor([[1.,0.]],requires_grad=True)
    synthetic=torch.tensor([[0.,1.]],requires_grad=True)
    mask=torch.ones(1,2,dtype=torch.bool)
    target=PitchTargets(torch.tensor([[True,False]]),torch.tensor([0]),torch.tensor([True]))
    nb=SimpleNamespace(features=natural,candidate_mask=mask);sb=SimpleNamespace(features=synthetic,candidate_mask=mask)
    model=lambda b:b.features
    control,base,aux=augmentation_loss(model,nb,target)
    expected=torch.nn.functional.cross_entropy(natural,torch.tensor([0]))
    assert torch.equal(control,base) and torch.allclose(control,expected,rtol=0,atol=1e-7) and aux.item()==0
    total,base,aux=augmentation_loss(model,nb,target,sb,target)
    assert torch.allclose(total,expected+.25*torch.nn.functional.cross_entropy(synthetic,torch.tensor([0])))
    total.backward();assert torch.isfinite(natural.grad).all() and torch.isfinite(synthetic.grad).all()


def test_uncovered_synthetic_rows_have_zero_gradient_without_resampling():
    import torch
    from types import SimpleNamespace
    from phenometrix_prosody.augmentation_experiment import augmentation_loss
    from phenometrix_prosody.selector_model import PitchTargets
    natural=torch.tensor([[1.,0.]],requires_grad=True);synthetic=torch.tensor([[0.,1.],[1.,0.]],requires_grad=True)
    target=PitchTargets(torch.tensor([[True,False]]),torch.tensor([0]),torch.tensor([True]))
    none=PitchTargets(torch.zeros(2,2,dtype=torch.bool),torch.full((2,),-1,dtype=torch.long),torch.zeros(2,dtype=torch.bool))
    nb=SimpleNamespace(features=natural,candidate_mask=torch.ones(1,2,dtype=torch.bool));sb=SimpleNamespace(features=synthetic,candidate_mask=torch.ones(2,2,dtype=torch.bool))
    total,base,aux=augmentation_loss(lambda b:b.features,nb,target,sb,none)
    assert aux.item()==0 and torch.equal(total,base)
    total.backward();assert torch.equal(synthetic.grad,torch.zeros_like(synthetic))
