import copy
from pathlib import Path
import numpy as np
import pytest
from phenometrix_prosody import candidate_coverage_experiment as e
from phenometrix_prosody.selector_model import prepare_selector_batch


def natural():
    result={}
    for name,error,success in [('original_mlp',.1,.7),('original_hybrid',.11,.71),*[(e.key(a,s),.094,.72) for a in e.ARMS for s in e.SEEDS]]:
        result[name]={'macro':{'joint_error':error,'reference_voiced_success_rate':success},'per_condition':{c:{'joint_error':error} for c in e.old.CONDITIONS}}
    return result


def analytic():
    names=[e.key('revised',s) for s in e.SEEDS]
    item=lambda kind:{'kind':kind,'methods':{n:{'all':{'success_within_50c_all_frames':1.},'regions':{'after-observation-change':{'success_within_50c_all_frames':1.}}} for n in names}}
    groups={'regression':{n:item('constant' if n.startswith('constant') else 'observation-change' if n.startswith('observation') else 'glide') for n in e.old.SYNTHETIC_IDS}}
    for group,prefix in [('consumed_harmonic','harmonic-balance'),('consumed_subharmonic','subharmonic-balance'),('consumed_augmentation','augmentation-balance'),('challenge','coverage-balance')]:
        groups[group]={f'{prefix}-f{i}-p{j}':item('observation-change') for i in range(6) for j in range(2)}
    return groups


def batch(frequencies):
    hz=np.full((len(frequencies),6),np.nan);mask=np.zeros_like(hz,bool);x=np.zeros((len(hz),57),np.float32)
    for i,values in enumerate(frequencies):
        for j,value in enumerate(values):
            hz[i,j]=value;mask[i,j]=True;x[i,8*j:8*(j+1)]=[1,np.log2(value/50),.8,0,0,0,0,.8]
        x[i,55]=len(values)/6
    return {r:prepare_selector_batch(x,hz,mask,representation=r) for r in ('fixed-slots','shared-candidates')}


def test_natural_all_components_are_conjunctive():
    result=natural();assert e.natural_decision(result)['pass']
    result[e.key('revised',e.SEEDS[0])]['macro']['joint_error']=.1
    assert not e.natural_decision(result)['checks']['each_seed_improves_original_mlp']


def test_small_gain_never_promotes_even_when_other_components_pass():
    result=natural()
    for seed in e.SEEDS:result[e.key('revised',seed)]['macro']['joint_error']=.098
    d=e.natural_decision(result)
    assert not d['pass'] and not d['checks']['mean_improves_both_by_half_point']
    assert d['checks']['each_seed_improves_original_mlp']


@pytest.mark.parametrize('field', ['condition','voiced'])
def test_condition_and_unconditional_voiced_success_remain_required(field):
    result=natural()
    for s in e.SEEDS:
        if field=='condition':result[e.key('revised',s)]['per_condition']['clip01']['joint_error']=.111
        else:result[e.key('revised',s)]['macro']['reference_voiced_success_rate']=.699
    assert not e.natural_decision(result)['pass']


@pytest.mark.parametrize('group',list(e.GROUPS)[1:])
def test_each_old_and_fresh_case_each_seed_remains_required(group):
    groups=analytic();case=next(iter(groups[group]));groups[group][case]['methods'][e.key('revised',e.SEEDS[1])]['regions']['after-observation-change']['success_within_50c_all_frames']=.949
    decision=e.analytic_decision(groups);assert not decision[group+'_pass']
    assert decision['stationary_pass'] and decision['harmonic_change_pass']


def test_original_whole_and_postchange_gates_are_distinct():
    groups=analytic();groups['regression']['constant-100']['methods'][e.key('revised',e.SEEDS[2])]['all']['success_within_50c_all_frames']=.94
    d=e.analytic_decision(groups);assert not d['stationary_pass'] and d['harmonic_change_pass']


def test_equal_count_substitution_is_rejected():
    groups=analytic();groups['challenge']['unknown']=groups['challenge'].pop('coverage-balance-f0-p0')
    with pytest.raises(ValueError,match='identities'):e.analytic_decision(groups)


def test_missing_old_failure_cannot_disappear_from_denominator():
    groups=analytic();groups['consumed_augmentation'].pop('augmentation-balance-f5-p1')
    with pytest.raises(ValueError,match='inventory'):e.analytic_decision(groups)


def test_coverage_reports_gains_and_losses_even_when_net_is_zero():
    data={'records':[{'start':0,'stop':2,'speaker':'A','condition':'clean'}]}
    labels={'valid':np.array([True,True]),'voiced':np.array([True,True]),'hz':np.array([100.,100.])}
    prepared={'original':batch([[100],[200]]),'revised':batch([[200],[100]])}
    c=e.coverage_transition(data,labels,prepared)['cells']['A/clean']
    for rep in ('fixed-slots','shared-candidates'):
        assert c[rep]['lost']==c[rep]['gained']==1
        assert c[rep]['lost_fraction']==c[rep]['gained_fraction']==.5


def test_fixed_voicing_oracle_cannot_repair_false_negative_or_false_positive():
    data={'records':[{'start':0,'stop':4,'speaker':'A','condition':'clean'}]}
    labels={'valid':np.ones(4,bool),'voiced':np.array([True,True,False,False]),'hz':np.array([100.,100.,np.nan,np.nan])}
    prepared=batch([[100],[100],[100],[100]])
    baseline={'original_mlp':{'cells':{'A/clean':{'eligible_valid_frames':4,'correct_frames':1}}}}
    h=e.oracle_headroom(data,labels,prepared,np.array([.5,.499,.5,.499]),baseline)
    for rep in ('fixed-slots','shared-candidates'):
        c=h[rep]['cells']['A/clean'];assert c['oracle_correct']==2 and c['eligible']==4 and c['gain']==.25


def test_invalid_reference_does_not_enter_oracle_denominator():
    data={'records':[{'start':0,'stop':2,'speaker':'A','condition':'clean'}]}
    labels={'valid':np.array([True,False]),'voiced':np.array([True,True]),'hz':np.array([100.,100.])}
    baseline={'original_mlp':{'cells':{'A/clean':{'eligible_valid_frames':1,'correct_frames':0}}}}
    h=e.oracle_headroom(data,labels,batch([[100],[100]]),np.ones(2),baseline)
    assert h['fixed-slots']['cells']['A/clean']['oracle_correct']==1


def test_reusing_output_is_rejected_before_any_scientific_work(tmp_path):
    with pytest.raises(ValueError,match='Fresh ignored'):e.run(tmp_path)


def test_regional_state_counts_keep_unvoiced_and_unavailable_separate():
    from types import SimpleNamespace
    prepared={'original':batch([[100],[100],[]]),'revised':batch([[100],[100],[]])}
    item={'scenario':{'kind':'constant','initialHz':100},'timing':SimpleNamespace(reference_sample=np.array([1248,1728,2208]),availability_sample=np.array([2208,2688,3168]),common_eligible=np.ones(3,bool)),
          'prepared':prepared,'probability':np.array([.1,.9,.9]),'predictions':{'revised__seed20261022':(np.array([0,1,2]),np.array([np.nan,100.,np.nan]))}}
    region=e.analytic_coverage(item)['all']
    assert region['methods']['revised__seed20261022']=={'unvoiced':1,'voiced':1,'unavailable':1}
    assert region['original_voiced_gate_positive']==2 and region['frames']==3
