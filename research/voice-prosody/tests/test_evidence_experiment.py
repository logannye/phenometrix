"""Policy and numerics fixtures only; never execute scientific generation/fits."""
import copy
import numpy as np
import pytest
import torch
from phenometrix_prosody import evidence_experiment as study
from phenometrix_prosody import selector_experiment as old
from phenometrix_prosody import harmonic_challenge,subharmonic_challenge,augmentation_cases,coverage_cases,expanded_cases,evidence_cases
from phenometrix_prosody.selector_model import build_pitch_targets,selector_loss
from phenometrix_prosody.evidence_selector_model import EvidenceHarmonicSelector
from phenometrix_prosody.expanded_selector_model import ExpandedHarmonicSelector
from test_evidence_selector_model import batch,evidence


def test_inference_retains_exact_old_control_voicing_and_unavailable_states():
    b=batch();e=evidence(b);torch.manual_seed(5);base=ExpandedHarmonicSelector();torch.manual_seed(5);new=EvidenceHarmonicSelector('off')
    probability=np.array([.49,.5,.99]);before=probability.copy()
    expected=old.infer(base,b,probability);actual=study.infer(new,b,e,probability)
    assert np.array_equal(actual[0],expected[0]) and np.array_equal(actual[1],expected[1],equal_nan=True)
    assert actual[0].tolist()==[0,1,2] and np.array_equal(probability,before)
    with pytest.raises(ValueError,match='alignment'):study.infer(new,b,e[:2],probability)


def test_two_loss_denominators_match_explicit_covered_means_and_zero_rows():
    b=batch();e=evidence(b);t=build_pitch_targets(b,[200,125,np.nan],[True,True,False],[True]*3);model=EvidenceHarmonicSelector('periodicity')
    empty=b.index_select([2]);et=t.index_select([2]);ee=e[[2]]
    full,natural,weighted=study.loss(model,b,t,e,empty,et,ee)
    expected=selector_loss(model(b,e),b.candidate_mask,t,objective='acceptable-set')
    torch.testing.assert_close(natural,expected,rtol=0,atol=0);torch.testing.assert_close(full,natural,rtol=0,atol=0)
    assert float(weighted.detach())==0;full.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_diagnostics_include_uncovered_denominators_and_empty_strata():
    b=batch();e=evidence(b);t=build_pitch_targets(b,[200,125,np.nan],[True,True,False],[True]*3);model=EvidenceHarmonicSelector()
    result=study.diagnostics(model,b,t,e,{'broad':np.ones(3,bool),'empty':np.zeros(3,bool),'uncovered':np.array([False,False,True])})
    assert result['broad']['rows']==3 and result['broad']['covered_rows']==2
    assert result['empty']['success_all_rows'] is None and result['empty']['mean_best_acceptable_minus_best_wrong'] is None
    assert result['uncovered']['success_all_rows']==0 and result['uncovered']['covered_rows']==0 and result['uncovered']['conditional_loss']==0


def natural_fixture():
    result={b:{'macro':{'joint_error':.1,'reference_voiced_success_rate':.8},'per_condition':{c:{'joint_error':.1} for c in old.CONDITIONS}} for b in ('original_mlp','original_hybrid')}
    for a in study.ARMS:
        for s in study.SEEDS:result[study.key(a,s)]={'macro':{'joint_error':.09,'reference_voiced_success_rate':.81},'per_condition':{c:{'joint_error':.09} for c in old.CONDITIONS}}
    return result


def test_candidate_fixed_and_all_natural_gates_are_conjunctive():
    x=natural_fixture();assert study.natural_decision(x)['pass']
    for key,value in [('joint_error',.101),('reference_voiced_success_rate',.7)]:
        bad=copy.deepcopy(x);bad[study.key(study.CANDIDATE,study.SEEDS[0])]['macro'][key]=value;assert not study.natural_decision(bad)['pass']
    bad=copy.deepcopy(x)
    for s in study.SEEDS:bad[study.key(study.CANDIDATE,s)]['per_condition']['clean']['joint_error']=.111
    assert not study.natural_decision(bad)['pass']
    bad=copy.deepcopy(x)
    for s in study.SEEDS:bad[study.key(study.CANDIDATE,s)]['macro']['joint_error']=.096
    assert not study.natural_decision(bad)['pass']
    # A much better control never replaces a failing predesignated candidate.
    for s in study.SEEDS:bad[study.key('existing__on',s)]['macro']['joint_error']=0
    assert study.natural_decision(bad)['predesignated_candidate']=='targeted__on' and not study.natural_decision(bad)['pass']


def analytic_fixture():
    groups={'regression':{cid:('constant' if cid.startswith('constant') else 'observation-change' if cid.startswith('observation-change') else 'glide') for cid in old.SYNTHETIC_IDS}}
    for group,module in [('consumed_harmonic',harmonic_challenge),('consumed_subharmonic',subharmonic_challenge),('consumed_augmentation',augmentation_cases),('consumed_coverage',coverage_cases),('consumed_expansion',expanded_cases),('challenge',evidence_cases)]:groups[group]={r['id']:'observation-change' for r in module.challenge_spec()['records']}
    return {g:{cid:{'kind':kind,'methods':{study.key(study.CANDIDATE,s):{'all':{'success_within_50c_all_frames':1.},'regions':{'after-observation-change':{'success_within_50c_all_frames':1.}}} for s in study.SEEDS}} for cid,kind in rows.items()} for g,rows in groups.items()}


def test_each_seed_case_gate_and_exact_configuration_inventory_required():
    x=analytic_fixture();d=study.analytic_decision(x);assert d['pass'] and set(d['passed_by_seed'].values())=={78}
    first=next(iter(x['challenge'].values()));first['methods'][study.key(study.CANDIDATE,study.SEEDS[0])]['regions']['after-observation-change']['success_within_50c_all_frames']=.94
    assert not study.analytic_decision(x)['pass']
    x=analytic_fixture();cid=next(iter(x['challenge']));x['challenge']['wrong-config']=x['challenge'].pop(cid)
    with pytest.raises(ValueError,match='identity'):study.analytic_decision(x)
    x=analytic_fixture();x['challenge'].pop(next(iter(x['challenge'])))
    with pytest.raises(ValueError,match='inventory'):study.analytic_decision(x)


def test_completed_checkpoint_inventory_cannot_be_partial_or_changed(tmp_path):
    with pytest.raises(ValueError,match='Incomplete12'):study.check_checkpoints(tmp_path,{'fits':{}})
    (tmp_path/'checkpoints').mkdir();fits={}
    for a in study.ARMS:
        for s in study.SEEDS:
            name=study.key(a,s);p=tmp_path/'checkpoints'/(name+'.pt');p.write_bytes(b'fixture-only');fits[name]={'weights_sha256':study.ev.sha(p)}
    study.check_checkpoints(tmp_path,{'fits':fits});p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):study.check_checkpoints(tmp_path,{'fits':fits})
