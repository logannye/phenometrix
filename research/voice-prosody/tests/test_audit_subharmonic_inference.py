import copy
import numpy as np
import pytest
import torch
from phenometrix_prosody import audit_subharmonic_inference as audit
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


def checkpoint(arm='set',seed=20261022):
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
    ('arm','alias-margin','arm/seed'),('context_mode','absolute','architecture'),
    ('config',{'representation':'shared-candidates','loss':'nearest','hidden_size':32},'architecture'),
])
def test_checkpoint_contract_mutations_rejected(field,value,match):
    saved,fit,_=checkpoint();saved[field]=value
    with pytest.raises(ValueError,match=match):audit.validate_checkpoint('set__seed20261022',saved,fit)


def test_initialization_mismatch_rejected():
    saved,fit,_=checkpoint();fit['initial_tensors_sha256']='0'*64
    with pytest.raises(ValueError,match='initialization'):audit.validate_checkpoint('set__seed20261022',saved,fit)


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
    assert len(audit.REGRESSION_IDS)==14 and len(audit.CONSUMED_IDS)==len(audit.CHALLENGE_IDS)==12
    assert len(audit.REGRESSION_IDS|audit.CONSUMED_IDS|audit.CHALLENGE_IDS)==38
    assert audit.MODELS=={f'{a}__seed{s}' for a in ('set','all-margin','alias-margin') for s in (20261022,20261023,20261024)}
