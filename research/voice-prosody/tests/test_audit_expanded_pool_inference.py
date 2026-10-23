"""Small synthetic tensors only; no completed-run artifacts are opened."""
import copy
from io import BytesIO
import numpy as np
import pytest
import torch

from phenometrix_prosody import audit_expanded_pool_inference as audit
from phenometrix_prosody.expanded_selector_model import ExpandedHarmonicSelector
from phenometrix_prosody.expanded_candidate_features import FEATURE_NAMES_EXPANDED
from phenometrix_prosody.model import FEATURE_NAMES_V2


def feature(k=6, rows=((100.,200.),())):
    n=len(rows);x=np.zeros((n,8*k+9),np.float32);hz=np.full((n,k),np.nan);mask=np.zeros((n,k),bool)
    for r,frequencies in enumerate(rows):
        for j,f in enumerate(frequencies):
            s=[.6+.03*j,.2,0,.4 if j==0 else 0,.5]
            x[r,j*8:(j+1)*8]=[1,np.log2(f/50),*s,max(s)];hz[r,j]=f;mask[r,j]=True
        x[r,8*k:]=[-3,.1,0,0,.2,.7,.8,len(frequencies)/6,1]
    return dict(features=x,candidate_hz=hz,candidate_mask=mask,reference_sample=2208+480*np.arange(n),
        feature_names=np.array(FEATURE_NAMES_V2 if k==6 else FEATURE_NAMES_EXPANDED),sample_rate_hz=np.array(48000),
        pcm_sample_count=np.array(2208+480*(n-1)))


def checkpoint(arm='expanded7',seed=20261022):
    torch.manual_seed(seed);model=ExpandedHarmonicSelector();model.scaling_fitted.fill_(True)
    w=model.state_dict()
    ck={'arm':arm,'context_mode':'relative','candidate_capacity':6 if arm=='original6' else 7,
        'config':dict(audit.CONFIG),'state_dict':w}
    fit={'arm':arm,'seed':seed,'parameter_count':3009,
         'initial_tensors_sha256':audit.equations.initial_tensor_hash('relative','acceptable-set',seed,w)}
    return ck,fit


def phases():
    stamps=[f'2026-10-22T01:00:0{i}+00:00' for i in range(7)]
    common=dict(source_sha256={'source.py':'a'*64},challenge_spec_sha256='b'*64,
                arms=list(audit.ARMS),seeds=list(audit.SEEDS),maximum_fits=6)
    historical={n:{'path':n+'.pt','sha256':'c'*64} for n in audit.OLD_MODELS}
    source={**common,'schema':'phenometrix-expanded-pool-source-freeze.v1','status':'sources_frozen','fits':{},
        'historical_checkpoints':historical,'candidate_policy':audit.POLICY,'sources_frozen_at':stamps[0],
        'development_references_read_for_prerequisites':False,'development_references_read_for_final_scoring':False,
        'reserved_content_read':False}
    fits={n:{'weights_sha256':'d'*64} for n in audit.NEW_MODELS}
    ck={**common,'schema':'phenometrix-expanded-pool-checkpoint-freeze.v1','status':'checkpoints_frozen',
        'frozen_at':stamps[3],'checkpoints':{n:{'sha256':'d'*64} for n in audit.NEW_MODELS}}
    frozen={**source,'schema':'phenometrix-expanded-pool-experiment.v1','status':'predictions_frozen','fits':fits,
        'prediction_sha256':{'one.npz':'e'*64},'development_references_read_for_prerequisites':True,
        'prerequisite_references_read_at':stamps[1],'training_frozen_at':stamps[2],
        'checkpoints_frozen_at':stamps[3],'predictions_frozen_at':stamps[4]}
    manifest={**frozen,'status':'complete','development_references_read_for_final_scoring':True,
        'final_scoring_reference_read_at':stamps[5],'completed_at':stamps[6]}
    return copy.deepcopy((source,ck,frozen,manifest))


def test_phase_validation_accepts_explicit_prerequisites_but_preserves_final_scoring_boundary():
    audit.validate_freezes(*phases())


@pytest.mark.parametrize('change',[
    lambda s,c,f,m:f.update(development_references_read_for_final_scoring=True),
    lambda s,c,f,m:f.update(development_references_read_for_prerequisites=False),
    lambda s,c,f,m:c['checkpoints'].pop(next(iter(c['checkpoints']))),
    lambda s,c,f,m:m.update(prediction_sha256={}),
    lambda s,c,f,m:s.update(reserved_content_read=True),
    lambda s,c,f,m:c.update(frozen_at='2026-10-22T02:00:00+00:00'),
    lambda s,c,f,m:f.update(training_frozen_at='2026-10-22T00:00:00+00:00'),
    lambda s,c,f,m:m.update(status='fitting'),
])
def test_phase_changes_or_missing_models_refuse(change):
    args=phases();change(*args)
    with pytest.raises(ValueError):audit.validate_freezes(*args)


def test_feature_header_refuses_target_before_decoding_and_accepts_both_widths():
    b=BytesIO();np.savez(b,truth_hz=np.array([100.]),features=np.zeros((1,57),np.float32))
    with pytest.raises(ValueError,match='target'):audit.load_feature_bytes(b.getvalue())
    for k in (6,7):
        f=feature(k);b=BytesIO();np.savez(b,**f)
        out=audit.load_feature_bytes(b.getvalue())
        for key in f:np.testing.assert_array_equal(out[key],f[key])


def test_arm_pair_checks_allow_only_added_source_channel_and_count_changes():
    old,new=feature(6),feature(7)
    audit.validate_arm_inputs(old,new)
    new['features'][0,5]=.8;new['features'][0,7]=.8
    audit.validate_arm_inputs(old,new)
    new['features'][0,2]=.1
    with pytest.raises(ValueError,match='weakened'):audit.validate_arm_inputs(old,new)


def test_original_frequency_and_noncount_global_are_not_replaceable():
    old,new=feature(6),feature(7)
    new['candidate_hz'][0,0]=101.
    with pytest.raises(ValueError,match='missing'):audit.validate_arm_inputs(old,new)
    new=feature(7);new['features'][0,60]=.99
    with pytest.raises(ValueError,match='global'):audit.validate_arm_inputs(old,new)


@pytest.mark.parametrize('k',[6,7])
def test_explicit_dynamic_equations_equal_fixture_model_and_do_not_call_forward(k,monkeypatch):
    f=feature(k,rows=((70.,100.,150.,200.,300.,450.) if k==6 else (70.,100.,150.,200.,300.,450.,690.),(),(110.,220.)))
    batch=audit.prepare(f);torch.manual_seed(91);model=ExpandedHarmonicSelector()
    with torch.inference_mode():expected=model(batch)
    monkeypatch.setattr(model,'forward',lambda *_:(_ for _ in ()).throw(AssertionError('forbidden forward')))
    with torch.inference_mode():actual=audit.independent_logits(model.state_dict(),batch.features,batch.candidate_mask)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    assert torch.isfinite(actual).all() and torch.all(actual[1]==-1e9)


def test_checkpoint_arm_capacity_scaling_and_initial_hash_are_checked():
    ck,fit=checkpoint();w,initial=audit.validate_checkpoint('expanded7__seed20261022',ck,fit)
    assert initial==fit['initial_tensors_sha256'] and set(w)==set(ck['state_dict'])
    ck['candidate_capacity']=6
    with pytest.raises(ValueError,match='capacity'):audit.validate_checkpoint('expanded7__seed20261022',ck,fit)
    ck,fit=checkpoint();ck['state_dict']['candidate_scale'][0]=0
    with pytest.raises(ValueError,match='scaling'):audit.validate_checkpoint('expanded7__seed20261022',ck,fit)
    ck,fit=checkpoint();fit['initial_tensors_sha256']='0'*64
    with pytest.raises(ValueError,match='initial'):audit.validate_checkpoint('expanded7__seed20261022',ck,fit)


def prediction_payload():
    out={'original_probability':np.array([.1,.8,.9])}
    for name in audit.METHODS:
        out[name+'__state']=np.array([0,1,2],np.uint8)
        out[name+'__hz']=np.array([np.nan,200.,np.nan])
    return out


def test_all_twelve_prediction_pipelines_and_external_voicing_required():
    p,rows=audit.analytic_predictions(prediction_payload(),3)
    assert len(rows)==12 and p.shape==(3,)
    bad=prediction_payload();bad.pop(next(n for n in bad if n.endswith('__hz')))
    with pytest.raises(ValueError,match='inventory'):audit.analytic_predictions(bad,3)
    bad=prediction_payload();bad[next(n for n in bad if n.endswith('__state'))][0]=2
    with pytest.raises(ValueError,match='voicing'):audit.analytic_predictions(bad,3)


def test_emission_ties_unavailable_and_frozen_voice_are_independent_of_pool_width():
    f=feature(7,rows=((100.,200.),(),(300.,)))
    batch=audit.prepare(f);logits=torch.zeros((3,7));p=np.array([.8,.9,.1])
    state,hz=audit.emit(logits,batch.candidate_hz.numpy(),batch.candidate_mask.numpy(),p)
    np.testing.assert_array_equal(state,[1,2,0]);np.testing.assert_array_equal(hz,[100.,np.nan,np.nan])
    report=audit.compare_arrays((state,hz),state,hz,batch,p)
    assert report['nonmember_emissions']==report['state_mismatches']==report['hz_mismatches']==0
    wrong=hz.copy();wrong[0]=150.
    assert audit.compare_arrays((state,wrong),state,hz,batch,p)['nonmember_emissions']==1


def test_original_voicing_guard_forbids_expanded_input_and_checks_probability_gate():
    weights={'feature_mean':torch.zeros(57),'feature_scale':torch.ones(57),
        'projection.weight':torch.zeros((32,57)),'projection.bias':torch.zeros(32),
        'frame_hidden.weight':torch.zeros((32,32)),'frame_hidden.bias':torch.zeros(32),
        'voicing_head.weight':torch.zeros((1,32)),'voicing_head.bias':torch.zeros(1)}
    p=np.array([.5,.5]);report=audit.probability_check(weights,feature(6),p)
    assert report['max_absolute_difference']==report['gate_mismatches']==0
    with pytest.raises(ValueError,match='original57'):audit.probability_check(weights,feature(7),p)
    with pytest.raises(ValueError,match='parity'):audit.probability_check(weights,feature(6),np.array([.49,.5]))


def test_expected_inventory_and_prior_control_comparison_are_complete():
    assert sum(map(len,audit.GROUP_IDS.values()))==74 and len(audit.METHODS)==12 and len(audit.NEW_MODELS)==6
    per_pipeline=255532+14*156+60*196
    assert per_pipeline*12==3233712
    assert (255532+14*156+48*196)*3==801372
    a=(np.array([1,0],np.uint8),np.array([200.,np.nan]))
    assert audit.preceding.exact_control(a,a)['frames']==2
    b=(a[0],np.array([100.,np.nan]))
    with pytest.raises(ValueError,match='changed'):audit.preceding.exact_control(a,b)


class RecordingLedger:
    def __init__(self): self.calls=[]
    def read(self,path,role,expected=None):
        self.calls.append((path,role,expected));return b'fixture bytes'


def test_final_checkpoint_uses_exact_freeze_binding_without_current_input_entry():
    ledger=RecordingLedger();run=audit.RUNS/'fixture-expanded';name='expanded7__seed20261022'
    path=run/'checkpoints'/(name+'.pt');row={'path':str(path.relative_to(audit.REPO)),'sha256':'a'*64}
    assert audit.new_checkpoint_bytes(ledger,run,name,row,{'weights_sha256':'a'*64})==b'fixture bytes'
    assert ledger.calls==[(path,'weights','a'*64)]
    with pytest.raises(ValueError,match='path/hash'):
        audit.new_checkpoint_bytes(ledger,run,name,row,{'weights_sha256':'b'*64})
    wrong={**row,'path':str((run/'another.pt').relative_to(audit.REPO))}
    with pytest.raises(ValueError,match='path/hash'):
        audit.new_checkpoint_bytes(ledger,run,name,wrong,{'weights_sha256':'a'*64})


def test_inherited_map_must_equal_both_pinned_historical_ledgers_and_all_phases():
    path=audit.RUNS/'old/input.npz';relative=str(path.relative_to(audit.REPO))
    prior={'input_sha256':{str(path):'a'*64}}
    last={'input_sha256':{relative:'a'*64,'research/voice-prosody/runs/other.npz':'b'*64}}
    expected={relative:'a'*64,'research/voice-prosody/runs/other.npz':'b'*64}
    phases=[{'historical_input_sha256':expected.copy()} for _ in range(3)]
    assert audit.historical_bindings(prior,last,*phases)==expected
    phases[1]['historical_input_sha256'].pop(relative)
    with pytest.raises(ValueError,match='inventory'):
        audit.historical_bindings(prior,last,*phases)
    last['input_sha256'][relative]='c'*64
    with pytest.raises(ValueError,match='disagree'):
        audit.historical_bindings(prior,last,*phases)


def test_only_named_consumed_manifests_may_use_verified_inherited_bindings():
    ledger=RecordingLedger();path=audit.RUNS/'harmonic-context-v1/challenge/manifest.json'
    name=str(path.relative_to(audit.REPO));inherited={name:'d'*64}
    assert audit.historical_manifest_bytes(ledger,path,{},inherited)==b'fixture bytes'
    assert ledger.calls==[(path,'manifest','d'*64)]
    with pytest.raises(ValueError,match='identity differs'):
        audit.historical_manifest_bytes(ledger,path,{name:'e'*64},inherited)
    with pytest.raises(ValueError,match='absent'):
        audit.historical_manifest_bytes(ledger,path,{}, {})
    for prohibited in (path.with_name('spec.json'),audit.RUNS/'unrelated/manifest.json'):
        with pytest.raises(ValueError,match='allowed'):
            audit.historical_manifest_bytes(ledger,prohibited,{}, {str(prohibited.relative_to(audit.REPO)):'d'*64})
