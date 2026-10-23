import copy
from io import BytesIO
import numpy as np
import pytest
import torch
from phenometrix_prosody import audit_candidate_coverage_inference as audit


def phases():
    source={'schema':'phenometrix-candidate-coverage-freeze.v1','status':'sources_checkpoints_frozen',
        'source_sha256':{'source.py':'a'*64},'checkpoints':{n:{'path':n,'sha256':'b'*64} for n in audit.CHECKPOINTS},
        'candidate_policy':audit.POLICY,'arms':list(audit.ARMS),'seeds':list(audit.SEEDS),'new_fits':0,
        'frozen_at':'2026-10-22T00:00:00+00:00','challenge_spec_sha256':'c'*64}
    p=copy.deepcopy(source);p.update(schema='phenometrix-candidate-coverage-experiment.v1',status='predictions_frozen',
        optimizer_updates=0,prediction_sha256={'p.npz':'d'*64},development_references_read=False,
        train_references_read=False,reserved_content_read=False,predictions_frozen_at='2026-10-22T00:01:00+00:00')
    m=copy.deepcopy(p);m.update(status='complete',completed_at='2026-10-22T00:02:00+00:00')
    return source,p,m


def test_three_checkpoint_two_policy_freeze_and_exact_domain_inventory():
    audit.validate_freezes(*phases())
    assert len(audit.CHECKPOINTS)==3 and len(audit.METHODS)==6
    assert [len(v) for v in audit.GROUP_IDS.values()]==[14,12,12,12,12]
    assert len(set().union(*audit.GROUP_IDS.values()))==62


@pytest.mark.parametrize('change',[
    lambda f:f[0].update(candidate_policy='other'),
    lambda f:f[1].update(train_references_read=True),
    lambda f:f[1].update(development_references_read=True),
    lambda f:f[2].update(reserved_content_read=True),
    lambda f:f[2].update(optimizer_updates=1),
    lambda f:f[1].update(prediction_sha256={'changed':'0'*64}),
    lambda f:f[0]['checkpoints'].pop(next(iter(audit.CHECKPOINTS))),
    lambda f:f[1].update(predictions_frozen_at='2026-10-21T00:00:00+00:00'),
])
def test_policy_identity_fit_and_reference_boundaries_fail_closed(change):
    f=phases();change(f)
    with pytest.raises(ValueError):audit.validate_freezes(*f)


def minimal_feature():
    return {'reference_sample':np.array([2208]),'feature_names':np.array(['same']),
        'sample_rate_hz':np.array(48000),'pcm_sample_count':np.array(2400),'features':np.zeros((1,57),np.float32)}


def test_arm_candidates_can_change_but_original_global_history_cannot():
    a=minimal_feature();b=copy.deepcopy(a);b['features'][0,1]=1;b['features'][0,55]=.5
    audit.validate_arm_inputs(a,b)
    for col in (48,49,50,51,52,53,54,56):
        bad=copy.deepcopy(b);bad['features'][0,col]=1
        with pytest.raises(ValueError,match='global'):audit.validate_arm_inputs(a,bad)
    b['reference_sample'][0]+=480
    with pytest.raises(ValueError,match='metadata'):audit.validate_arm_inputs(a,b)


def predictions():
    z={'original_probability':np.array([.5,.49,.9])}
    for name in audit.METHODS:
        z[name+'__state']=np.array([1,0,2],np.uint8);z[name+'__hz']=np.array([100,np.nan,np.nan])
    return z


def test_prediction_inventory_has_six_methods_and_one_fixed_voice_vector():
    z=predictions();p,outputs=audit.read_analytic_predictions(z,3)
    assert len(outputs)==6 and p.tolist()==[.5,.49,.9]
    z['truth_hz']=np.array([100,100,100])
    with pytest.raises(ValueError,match='inventory'):audit.read_analytic_predictions(z,3)


def test_saved_voice_cannot_be_recomputed_from_revised_inputs():
    z=predictions();z['original_probability'][0]=.499
    with pytest.raises(ValueError,match='voice'):audit.read_analytic_predictions(z,3)


def test_control_nan_equality_and_any_output_drift_is_visible():
    a=(np.array([1,0,2],np.uint8),np.array([100,np.nan,np.nan]))
    assert audit.exact_control(a,a)['frames']==3
    b=(a[0].copy(),np.array([101,np.nan,np.nan]))
    with pytest.raises(ValueError,match='control changed'):audit.exact_control(a,b)


def test_emission_uses_each_arms_actual_pool_and_shared_original_gate():
    logits=torch.tensor([[0.,1.,-1.,-1.,-1.,-1.],[0.,1.,-1.,-1.,-1.,-1.]])
    mask=np.array([[1,1,0,0,0,0],[1,1,0,0,0,0]],bool);p=np.array([.5,.49])
    first=np.array([[100,200,np.nan,np.nan,np.nan,np.nan]]*2)
    revised=np.array([[100,300,np.nan,np.nan,np.nan,np.nan]]*2)
    a,b=[audit.emit(logits,hz,mask,p) for hz in (first,revised)]
    np.testing.assert_array_equal(a[0],b[0]);assert a[1][0]==200 and b[1][0]==300
    assert np.isnan(a[1][1]) and np.isnan(b[1][1])


def test_feature_archive_rejects_targets_before_decoding():
    b=BytesIO();np.savez(b,truth_hz=np.array([100]),features=np.zeros((1,57),np.float32))
    with pytest.raises(ValueError,match='target'):audit.prior_audit.load_feature_bytes(b.getvalue())
