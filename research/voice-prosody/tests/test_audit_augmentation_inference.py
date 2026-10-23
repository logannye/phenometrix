import copy
import numpy as np
import pytest
import torch
from phenometrix_prosody import audit_augmentation_inference as audit
from phenometrix_prosody.harmonic_selector_model import HarmonicSelector
from phenometrix_prosody.selector_model import prepare_selector_batch


def fixture():
    x=np.zeros((3,57),np.float32);hz=np.full((3,6),np.nan);mask=np.zeros((3,6),bool)
    for row, frequencies in enumerate(([100,200,300],[150],[])):
        for i,frequency in enumerate(frequencies):
            scores=[.2,.3+i*.05,0,.1,.4]
            x[row,i*8:(i+1)*8]=[1,np.log2(frequency/50),*scores,max(scores)]
            hz[row,i],mask[row,i]=frequency,True
        x[row,48:]=[-3,.2,0,0,.1,.5,.6,len(frequencies)/6,1]
    return prepare_selector_batch(x,hz,mask,representation='shared-candidates')


def checkpoint(arm='natural-only',seed=20261022):
    torch.manual_seed(seed);model=HarmonicSelector('relative','acceptable-set')
    model.fit_scaling(fixture(),split='train');weights=model.state_dict()
    saved={'arm':arm,'context_mode':'relative','config':audit.CONFIG.copy(),'state_dict':weights}
    fit={'arm':arm,'seed':seed,'parameter_count':3009,
         'initial_tensors_sha256':audit.initial_tensor_hash('relative','acceptable-set',seed,weights)}
    return saved,fit,model


@pytest.mark.parametrize('arm',audit.ARMS)
def test_checkpoint_arm_does_not_change_inference_equations(arm):
    saved,fit,model=checkpoint(arm)
    weights,initial=audit.validate_checkpoint(f'{arm}__seed20261022',saved,fit)
    batch=fixture()
    with torch.inference_mode():
        expected=model(batch)
        actual=audit.independent_logits(weights,'relative',batch.features,batch.candidate_mask)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    assert initial==fit['initial_tensors_sha256']


@pytest.mark.parametrize('field,value,match',[
    ('arm','stationary-aug','arm/seed'),('context_mode','absolute','architecture'),
    ('config',{'representation':'shared-candidates','loss':'nearest','hidden_size':32},'architecture'),
])
def test_checkpoint_contract_mutations_rejected(field,value,match):
    saved,fit,_=checkpoint();saved[field]=value
    with pytest.raises(ValueError,match=match):audit.validate_checkpoint('natural-only__seed20261022',saved,fit)


def test_initialization_mismatch_rejected():
    saved,fit,_=checkpoint();fit['initial_tensors_sha256']='0'*64
    with pytest.raises(ValueError,match='initialization'):audit.validate_checkpoint('natural-only__seed20261022',saved,fit)


def saved_predictions():
    saved={'original_probability':np.array([.5,.49,.9])}
    for name in audit.MODELS|audit.PRIOR_MODELS:
        saved[name+'__state']=np.array([1,0,2],np.uint8)
        saved[name+'__hz']=np.array([100,np.nan,np.nan])
    return saved


def test_all_nine_and_three_prior_predictions_require_retained_probabilities():
    saved=saved_predictions();p,outputs=audit.analytic_predictions(saved,3)
    assert set(outputs)==audit.MODELS and len(outputs)==9
    assert p.tolist()==[.5,.49,.9]
    del saved['original_probability']
    with pytest.raises(ValueError,match='inventory'):audit.analytic_predictions(saved,3)


def test_missing_prior_prediction_or_extra_target_field_rejected():
    saved=saved_predictions();saved.pop(next(iter(audit.PRIOR_MODELS))+'__hz')
    with pytest.raises(ValueError,match='inventory'):audit.analytic_predictions(saved,3)
    saved=saved_predictions();saved['truth_hz']=np.array([100,100,100])
    with pytest.raises(ValueError,match='inventory'):audit.analytic_predictions(saved,3)


def test_saved_probability_gate_is_checked_without_new_threshold():
    saved=saved_predictions();saved['original_probability'][0]=.499
    with pytest.raises(ValueError,match='voice gate'):audit.analytic_predictions(saved,3)


def test_domain_inventory_is_exact_and_disjoint():
    assert len(audit.REGRESSION_IDS)==14 and len(audit.CONSUMED_IDS)==len(audit.CONSUMED_SUBHARMONIC_IDS)==len(audit.CHALLENGE_IDS)==12
    assert len(audit.REGRESSION_IDS|audit.CONSUMED_IDS|audit.CONSUMED_SUBHARMONIC_IDS|audit.CHALLENGE_IDS)==50
    assert audit.MODELS=={f'{a}__seed{s}' for a in ('natural-only','stationary-aug','balance-aug') for s in (20261022,20261023,20261024)}


def phase_fixture():
    source={'status':'sources_frozen','fits':{},'source_sha256':{'source.py':'a'*64},
            'training_spec_sha256':'b'*64,'challenge_spec_sha256':'c'*64,
            'sources_frozen_at':'2026-10-22T00:00:00+00:00'}
    ck=copy.deepcopy(source);ck.update(status='checkpoints_frozen',fits={n:{} for n in audit.MODELS},checkpoints_frozen_at='2026-10-22T00:01:00+00:00')
    pf=copy.deepcopy(ck);pf.update(status='predictions_frozen',prediction_sha256={'prediction.npz':'d'*64},
        development_references_read=False,reserved_content_read=False,predictions_frozen_at='2026-10-22T00:02:00+00:00')
    m=copy.deepcopy(pf);m.update(schema='phenometrix-augmentation-experiment.v1',status='complete',completed_at='2026-10-22T00:03:00+00:00')
    return source,ck,pf,m


def test_source_checkpoint_prediction_freezes_are_all_required():
    f=phase_fixture();audit.validate_phase_freezes(*f)
    f[0]['status']='loading'
    with pytest.raises(ValueError,match='freezes'):audit.validate_phase_freezes(*f)


@pytest.mark.parametrize('mutation',['weights','source','specification','prediction','labels','chronology'])
def test_changed_freeze_or_label_boundary_is_rejected(mutation):
    f=phase_fixture()
    if mutation=='weights':f[1]['fits'][next(iter(audit.MODELS))]={'changed':True}
    elif mutation=='source':f[0]['source_sha256']={}
    elif mutation=='specification':f[0]['training_spec_sha256']='0'*64
    elif mutation=='prediction':f[2]['prediction_sha256']={}
    elif mutation=='labels':f[2]['development_references_read']=True
    else:f[0]['sources_frozen_at']='2026-10-22T01:00:00+00:00'
    with pytest.raises(ValueError):audit.validate_phase_freezes(*f)


def test_four_analytic_domains_include_consumed_subharmonics():
    domains=audit.analytic_domains(audit.RUNS/'harmonic-augmentation-v1')
    assert [v[0] for v in domains]==['regression','consumed_challenge','consumed_subharmonic','challenge']
    assert sum(len(v[1]) for v in domains)==50
    assert domains[2][2]==audit.RUNS/'subharmonic-objective-v1/challenge'
    assert domains[3][1]=={f'augmentation-balance-f{i}-p{j}' for i in range(6) for j in range(2)}


def cache_fixture():
    _,ck,pf,_=phase_fixture();path=audit.RUNS/'harmonic-augmentation-v1/checkpoint-freeze.json';sha='e'*64
    cache={'schema':'phenometrix-paired-harmonic-augmentation-cache-v1','split':'prospective-challenge',
        'freeze_kind':'checkpoint','freeze_path':str(path.relative_to(audit.REPO)),'freeze_sha256':sha,
        'checkpoint_freeze_path':str(path.relative_to(audit.REPO)),'checkpoint_freeze_sha256':sha,
        'spec_sha256':pf['challenge_spec_sha256'],'training_spec_sha256':pf['training_spec_sha256'],
        'challenge_spec_sha256':pf['challenge_spec_sha256'],'reference_content_read':False,
        'started_at':'2026-10-22T00:01:10+00:00','completed_at':'2026-10-22T00:01:20+00:00'}
    return cache,path,sha,pf,ck


def test_fresh_cache_requires_new_and_compatibility_freeze_fields():
    c,path,sha,pf,ck=cache_fixture();audit.validate_cache_binding(c,'challenge',path,sha,pf,ck)
    c['freeze_sha256']='f'*64
    with pytest.raises(ValueError,match='binding'):audit.validate_cache_binding(c,'challenge',path,sha,pf,ck)
    c,path,sha,pf,ck=cache_fixture();c['checkpoint_freeze_path']='elsewhere'
    with pytest.raises(ValueError,match='alias'):audit.validate_cache_binding(c,'challenge',path,sha,pf,ck)


def test_prospective_cache_before_checkpoint_or_with_reference_content_rejected():
    c,path,sha,pf,ck=cache_fixture();c['started_at']='2026-10-22T00:00:10+00:00'
    with pytest.raises(ValueError,match='generation'):audit.validate_cache_binding(c,'challenge',path,sha,pf,ck)
    c,path,sha,pf,ck=cache_fixture();c['reference_content_read']=True
    with pytest.raises(ValueError,match='reference'):audit.validate_cache_binding(c,'challenge',path,sha,pf,ck)


def test_feature_archive_rejects_truth_fields_before_array_decode():
    from io import BytesIO
    data=BytesIO();np.savez(data,truth_hz=np.array([100.]))
    with pytest.raises(ValueError,match='target field'):audit.load_feature_bytes(data.getvalue())
