"""Independent decision/phase/checkpoint fixtures; no scientific inputs or fitting."""
import copy
import hashlib
from pathlib import Path
import numpy as np
import pytest
import torch
from phenometrix_prosody import audit_evidence_scores as a
from phenometrix_prosody.harmonic_selector_model import HarmonicSelector


def cells():
    result={}
    for method in a.METHODS:
        error=.10 if method=='original_mlp' else .11 if method=='original_hybrid' else .08
        result[method]={s+'/'+c:{'joint_error':error,'reference_voiced_success_rate':.9}
            for s in a.base.SPEAKERS['development'] for c in a.base.CONDITIONS}
    return result


def gates():
    categories={'stationary':3,'harmonic_change':3,**{g:12 for g in a.GROUP_COUNTS if g!='regression'}}
    return {m:{category+'/'+str(i):{'category':category,'success':.95} for category,n in categories.items() for i in range(n)} for m in a.CANDIDATE}


def test_natural_gate_retains_all_components_and_factorial_contrasts():
    value=a.natural_gate(cells())
    assert value['pass'] and all(value['checks'].values())
    assert value['mean_joint_error']==pytest.approx(.08)
    assert value['joint_error_improvement_over_frozen']==pytest.approx({'original_mlp':.02,'original_hybrid':.03})
    assert set(value['factorial_error_differences'])=={'targeted_minus_existing_off','evidence_on_minus_off_existing','evidence_on_minus_off_targeted','interaction'}
    assert value['factorial_error_differences']['interaction']==pytest.approx(0.)


@pytest.mark.parametrize('failure', ['mean', 'seed', 'condition', 'voiced'])
def test_each_inherited_natural_component_can_reject(failure):
    value=cells()
    if failure=='mean':
        for m in a.CANDIDATE:
            for r in value[m].values():r['joint_error']=.097
    elif failure=='seed':
        for r in value[a.CANDIDATE[0]].values():r['joint_error']=.101
    elif failure=='condition':
        for m in a.CANDIDATE:
            for key,r in value[m].items():
                if key.endswith('/clean'):r['joint_error']=.112
                else:r['joint_error']=.06
    else:
        for r in value[a.CANDIDATE[0]].values():r['reference_voiced_success_rate']=.89
    assert not a.natural_gate(value)['pass']


def test_gate_rejects_missing_cells_or_methods():
    value=cells();del value[a.CANDIDATE[0]]['F01/clean']
    with pytest.raises(ValueError,match='denominator'):a.natural_gate(value)
    value=cells();del value['original_hybrid']
    with pytest.raises(ValueError,match='inventory'):a.natural_gate(value)


def test_all78_mandatory_case_gates_are_retained_per_seed():
    result=a.analytic_gates(gates())
    assert result['pass'] and result['required_per_seed']==78
    assert set(result['passed_by_seed'].values())=={78}
    changed=gates();changed[a.CANDIDATE[1]]['consumed_expansion/0']['success']=.9499
    assert not a.analytic_gates(changed)['pass']


@pytest.mark.parametrize('change', ['missing', 'category', 'seed'])
def test_analytic_gate_inventory_cannot_be_relaxed(change):
    value=gates()
    if change=='missing':del value[a.CANDIDATE[0]]['stationary/0']
    elif change=='category':value[a.CANDIDATE[0]]['stationary/0']['category']='challenge'
    else:del value[a.CANDIDATE[0]]
    with pytest.raises(ValueError):a.analytic_gates(value)


def phase_fixture():
    times=[f'2026-10-23T05:{i:02d}:00+00:00' for i in range(9)]
    m=dict(zip(('sources_frozen_at','input','prerequisite_reference_read_at','prerequisite','training_frozen_at',
        'checkpoints_frozen_at','predictions_frozen_at','final_scoring_reference_read_at','completed_at'),times))
    m['development_references_read_for_final_scoring']=True
    initial={'status':'sources_frozen','fits':{},'development_references_read_for_prerequisites':False,'development_references_read_for_final_scoring':False}
    cache={'status':'input_caches_frozen','reference_content_read':False,'frozen_at':times[1]}
    prerequisite={'pass':True,'frozen_at':times[3]};training={'status':'training_frozen','fits':{}}
    checkpoint={'status':'checkpoints_frozen'}
    prediction={'status':'predictions_frozen','fits':{k:{} for k in a.FIT_NAMES},'development_references_read_for_prerequisites':True,'development_references_read_for_final_scoring':False}
    return m,initial,cache,prerequisite,training,checkpoint,prediction


def test_explicit_gate_reference_read_precedes_fits_and_final_refs_follow_predictions():
    assert len(a.phase_order(*phase_fixture()))==9


@pytest.mark.parametrize('change', ['late_source','early_score','early_fit','early_ref','missing_fit'])
def test_phase_violations_refuse(change):
    args=list(phase_fixture());m,initial,cache,prereq,training,checkpoint,prediction=args
    if change=='late_source':m['sources_frozen_at']=m['checkpoints_frozen_at']
    elif change=='early_score':m['final_scoring_reference_read_at']=m['training_frozen_at']
    elif change=='early_fit':initial['fits']={'unexpected':{}}
    elif change=='early_ref':cache['reference_content_read']=True
    else:del prediction['fits'][a.FIT_NAMES[0]]
    with pytest.raises(ValueError):a.phase_order(*args)


def test_no_go_requires_absence_of_dependent_artifacts(tmp_path):
    m={'fits':{},'fit_elapsed_seconds':0};a.no_fit_artifacts(tmp_path,m)
    (tmp_path/'checkpoints').mkdir();(tmp_path/'checkpoints'/'one.pt').write_bytes(b'fixture')
    with pytest.raises(ValueError,match='checkpoints'):a.no_fit_artifacts(tmp_path,m)


@pytest.mark.parametrize('artifact', ['challenge','checkpoint-freeze.json','training-freeze.json','prediction-freeze.json','metrics.json'])
def test_no_go_cannot_contain_fresh_or_completed_later_stage(tmp_path,artifact):
    (tmp_path/artifact).write_text('fixture')
    with pytest.raises(ValueError,match='forbidden'):a.no_fit_artifacts(tmp_path,{'fits':{}})


def test_no_go_cannot_hide_fit_attempt(tmp_path):
    with pytest.raises(ValueError,match='attempts'):a.no_fit_artifacts(tmp_path,{'fits':{},'fit_attempts':{'one':{'status':'failed'}}})


def test_fresh_metadata_is_exact_without_generation():
    rows=a.fresh_records()
    assert len(rows)==12 and len({r['id'] for r in rows})==12
    assert {r['frequency_hz'] for r in rows}=={109.9,241.1,379.3,563.9,619.7,677.9}
    assert all(r['change_sample']==39840 and r['sample_count']==96000 and r['after']==[0.,1.,.5] for r in rows)


class FixtureAudit:
    def bind(self,path,expected=None):
        path=Path(path)
        if expected is not None:assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
        return path
    def equal(self,x,y,label):assert x==pytest.approx(y)


def checkpoint_fixture(tmp_path,monkeypatch):
    prior=tmp_path/'prior';run=tmp_path/'run';(prior/'checkpoints').mkdir(parents=True);(run/'checkpoints').mkdir(parents=True)
    monkeypatch.setattr(a,'PRIOR',prior)
    old={'fits':{}};m={'fits':{},'fit_attempts':{},'training_frozen_at':'2026-10-23T05:00:00Z',
        'checkpoints_frozen_at':'2026-10-23T06:00:00Z','fit_elapsed_seconds':12.}
    for seed in a.SEEDS:
        torch.manual_seed(seed);model=HarmonicSelector('relative','acceptable-set');state=model.state_dict();state['scaling_fitted']=torch.tensor(True)
        oldname=f'expanded7__seed{seed}';p=prior/'checkpoints'/(oldname+'.pt');torch.save({'state_dict':state},p)
        old['fits'][oldname]={'weights_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        state={**state,'evidence_head.weight':torch.zeros(1,3)}
        h=hashlib.sha256()
        for k,v in sorted(state.items()):h.update(k.encode());h.update(v.numpy().tobytes())
        for arm in a.ARMS:
            name=f'{arm}__seed{seed}';p=run/'checkpoints'/(name+'.pt')
            torch.save({'arm':arm,'evidence_mode':'periodicity' if arm.endswith('__on') else 'off',
                'config':{'representation':'shared-candidates','loss':'acceptable-set','hidden_size':32},'state_dict':state},p)
            m['fits'][name]={'arm':arm,'seed':seed,'optimizer_updates':396,'parameter_count':3012,'initial_tensors_sha256':h.hexdigest(),
                'weights_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'epoch_losses':[1.25]*12,'epoch_base_losses':[1.]*12,
                'epoch_weighted_auxiliary_losses':[.25]*12,'evidence_residual_weights':[[0.,0.,0.]],'elapsed_seconds':1.}
            m['fit_attempts'][name]={'status':'complete','initial_tensors_sha256':h.hexdigest(),'started_at':'2026-10-23T05:10:00Z','completed_at':'2026-10-23T05:11:00Z'}
    return run,m,old


def test_twelve_checkpoint_initializations_scaling_and_off_branch_reconstructed(tmp_path,monkeypatch):
    run,m,old=checkpoint_fixture(tmp_path,monkeypatch)
    result=a.verify_checkpoints(FixtureAudit(),run,m,old)
    assert result['checkpoints']==12 and result['parameters_each']==3012 and result['original_six_scaling_exact']


@pytest.mark.parametrize('change', ['initial','updates','loss','attempt','off_weight'])
def test_checkpoint_evidence_refuses_inconsistent_fit_record(tmp_path,monkeypatch,change):
    run,m,old=checkpoint_fixture(tmp_path,monkeypatch);name=a.FIT_NAMES[0]
    if change=='initial':m['fits'][name]['initial_tensors_sha256']='0'*64
    elif change=='updates':m['fits'][name]['optimizer_updates']=397
    elif change=='loss':m['fits'][name]['epoch_losses'][0]=9.
    elif change=='attempt':m['fit_attempts'][name]['status']='failed'
    else:
        p=run/'checkpoints'/(name+'.pt');saved=torch.load(p,weights_only=True);saved['state_dict']['evidence_head.weight'][0,0]=.1;torch.save(saved,p)
        m['fits'][name]['weights_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError):a.verify_checkpoints(FixtureAudit(),run,m,old)


def toy_evidence():
    hz=np.full((1,7),np.nan);hz[0,0]=300.;mask=np.isfinite(hz);members=np.zeros((1,7,7),bool);members[0,0,0]=True
    grouped={'candidate_hz':hz,'candidate_mask':mask,'member_mask':members}
    feature={'reference_sample':np.array([2208]),'sample_rate_hz':np.array(48000),'pcm_sample_count':np.array(2400)}
    periods=np.zeros((1,7));periods[mask]=160.;length=np.zeros((1,7),np.int32);length[mask]=1759
    evidence={'schema':np.array('phenometrix.candidate-periodicity-evidence.v1'),
        'feature_names':np.array(['period_correlation','half_primitive_period_gap','periodicity_valid']),
        'pcm_sha256':np.array('a'*64),'evidence':np.zeros((1,7,3),np.float32),'correlations':np.zeros((1,7,4)),
        'valid':np.zeros((1,7),bool),'period_samples':periods,'common_length':length,
        **{'grouped_'+k:v.copy() for k,v in grouped.items()},**{k:v.copy() for k,v in feature.items()}}
    return evidence,feature,grouped


def test_evidence_identity_uses_actual_emission_and_same_support():
    evidence,feature,grouped=toy_evidence()
    assert a.verify_evidence(evidence,feature,grouped,'a'*64)==1
    evidence['valid'][0,0]=True;evidence['correlations'][0,0]=[.9,.1,.2,.3];evidence['evidence'][0,0]=[.9,.3,1.]
    assert a.verify_evidence(evidence,feature,grouped,'a'*64)==1


@pytest.mark.parametrize('change', ['hz','time','correlation','period','validity'])
def test_evidence_misalignment_or_changed_arithmetic_refuses(change):
    evidence,feature,grouped=toy_evidence()
    if change=='hz':evidence['grouped_candidate_hz'][0,0]=150.
    elif change=='time':evidence['reference_sample'][0]+=480
    elif change=='correlation':evidence['correlations'][0,0,0]=.1
    elif change=='period':evidence['period_samples'][0,0]+=1
    else:evidence['valid'][0,1]=True
    with pytest.raises(ValueError):a.verify_evidence(evidence,feature,grouped,'a'*64)


def test_repair_has_exact_same58_closure_and_only_named_adapter_change():
    name='research/voice-prosody/phenometrix_prosody/evidence_inputs.py'
    old={f'fixture{i}':'a'*64 for i in range(57)};old[name]='b'*64
    new={**old,name:'c'*64};declared={name:{'before':'b'*64,'after':'c'*64}}
    assert a.repair_sources(old,new,declared)==declared


@pytest.mark.parametrize('change',['extra','missing','undeclared','wrong_before','second_source'])
def test_repair_refuses_broader_or_misrepresented_change(change):
    name='research/voice-prosody/phenometrix_prosody/evidence_inputs.py'
    old={f'fixture{i}':'a'*64 for i in range(57)};old[name]='b'*64
    new={**old,name:'c'*64};declared={name:{'before':'b'*64,'after':'c'*64}}
    if change=='extra':new['extra']='d'*64
    elif change=='missing':del new['fixture0']
    elif change=='undeclared':declared={}
    elif change=='wrong_before':declared[name]['before']='a'*64
    else:new['fixture0']='d'*64
    with pytest.raises(ValueError):a.repair_sources(old,new,declared)
