"""Meaningful oracle, denominator, isolation and fixed-forward checks."""
from pathlib import Path
import numpy as np
import pytest
from phenometrix_prosody import selector_headroom as sh


def example():
    ends=np.arange(2208,11001,480);count=len(ends)
    feature={'reference_sample':ends,'candidate_hz':np.full((count,6),np.nan),
             'candidate_mask':np.zeros((count,6),bool),'features':np.zeros((count,57),np.float32)}
    p=np.full(count,.1);index=np.full(count,-1,np.int16);yin=np.full(count,100.)
    refs=np.full((20,4),np.nan);refs[:5]=[0,0,0,0]
    cases=[(200,.1,[100,200],-1,200),(200,.9,[100,200],0,200),
           (200,.9,[200,400],0,100),(200,.9,[100,200],0,100),
           (200,.9,[100,300],0,100),(0,.1,[100],-1,100),
           (0,.9,[100],0,100),(800,.9,[700],0,700),
           (200,.9,[],-1,200),(200,.9,[200],0,200)]
    for i,(truth,prob,candidates,choice,raw) in enumerate(cases,5):
        j=i-1;refs[i]=[truth,int(truth>0),0,0];p[j]=prob;index[j]=choice;yin[j]=raw
        feature['candidate_hz'][j,:len(candidates)]=candidates
        feature['candidate_mask'][j,:len(candidates)]=True
    state=np.where(p<.5,0,np.where(index>=0,1,2));hz=np.full(count,np.nan)
    for i in np.flatnonzero(state==1):hz[i]=feature['candidate_hz'][i,index[i]]
    mlp={'target_sample':ends-960,'voicing_probability':p,'candidate_index':index,'state':state,'f0_hz':hz}
    return refs,feature,mlp,yin


def test_reference_assisted_selector_cannot_repair_frozen_gate():
    refs,feature,mlp,yin=example();f,excluded=sh.observed_frames(refs,11000,feature,mlp,yin);c=sh.frame_counts(f);r=sh.rates(c)
    assert excluded=={'excluded_invalid_reference':5,'excluded_boundary':5}
    assert (c['eligible_valid_frames'],c['reference_voiced_frames'],c['reference_unvoiced_frames'])==(10,8,2)
    assert c['reference_outside_supported_bounds']==1
    assert c['shared_false_unvoiced_frames']==1
    assert c['six_candidates_coverage_count']==5
    assert c['six_candidates_frozen_voicing_voiced_success_count']==4
    assert c['six_candidates_shared_false_unvoiced_covered_frames']==1
    assert c['six_candidates_both_wrong_recoverable_count']==1
    assert c['six_plus_raw_yin_frozen_voicing_correct_frames']==6
    assert r['six_candidates_joint_error_reduction_vs_mlp']==.2
    assert r['six_candidates_unconstrained_voiced_coverage']==5/8
    assert r['frozen_voicing_perfect_pitch_joint_error_floor']==.2
    assert r['frozen_voicing_perfect_pitch_voiced_success_ceiling']==7/8
    assert c['mlp_unavailable_frames']==1 and c['hybrid_unavailable_frames']==0


def test_paired_categories_partition_unconditional_and_voiced_targets():
    refs,feature,mlp,yin=example();f,_=sh.observed_frames(refs,11000,feature,mlp,yin);c=sh.frame_counts(f)
    keys=('both_correct','mlp_only_correct','hybrid_only_correct','neither_correct')
    assert [c['paired_'+k] for k in keys]==[2,1,2,5]
    assert sum(c['paired_voiced_'+k] for k in keys)==8
    assert c['mlp_correct_frames']==3 and c['hybrid_correct_frames']==4
    bins=sh.f0_masks(f)
    assert np.array_equal(sum(x.astype(int) for x in bins.values()),np.ones(10))
    assert sh.frame_counts(f,bins['voiced_above700'])['eligible_valid_frames']==1


def test_equal_cell_macro_is_not_duration_weighted():
    refs,feature,mlp,yin=example();f,_=sh.observed_frames(refs,11000,feature,mlp,yin)
    a=sh.frame_counts(f,np.arange(10)==5);b=sh.frame_counts(f,np.arange(10)!=5)
    macro=sh.macro([a,b]);pooled=sh.rates(sh.merge([a,b]))
    assert macro['rates']['mlp_joint_error']==pytest.approx((0+7/9)/2)
    assert pooled['mlp_joint_error']==.7
    assert macro['defined_cells']['shared_false_unvoiced_rate']==1


def test_target_mismatch_rejected_not_interpolated():
    refs,feature,mlp,yin=example();mlp['target_sample']=mlp['target_sample']+1
    with pytest.raises(ValueError,match='exact input target'):sh.observed_frames(refs,11000,feature,mlp,yin)


def weights():
    rng=np.random.default_rng(11);config=sh.ModelConfig(kind='framewise-mlp')
    w={'feature_mean':np.zeros(57,np.float32),'feature_scale':np.ones(57,np.float32)}
    for name,out,inn in [('projection',32,57),('frame_hidden',32,32),('voicing_head',1,32),('candidate_head',6,32)]:
        w[name+'.weight']=(rng.normal(size=(out,inn))*.05).astype(np.float32)
        w[name+'.bias']=(rng.normal(size=out)*.05).astype(np.float32)
    return config,w


def test_batched_forward_matches_single_frames_and_has_no_future_dependence():
    _,f,_,yin=example();rng=np.random.default_rng(1);f['features']=rng.normal(size=f['features'].shape).astype(np.float32)
    config,w=weights();batch=sh.model_arrays(f,config,w);single=sh.model_arrays(f,config,w,batch_size=1)
    np.testing.assert_allclose(batch['voicing_probability'],single['voicing_probability'],atol=1e-7,rtol=0)
    for k in ('state','candidate_index','target_sample'):np.testing.assert_array_equal(batch[k],single[k])
    f['features'][7:]=1000;changed=sh.model_arrays(f,config,w)
    for k in batch:np.testing.assert_array_equal(batch[k][:7],changed[k][:7])
    sh.validate_pair(f,batch,yin)


def test_reference_guard_rejects_unfrozen_test_and_unadmitted_speaker(tmp_path):
    ledger=sh.Ledger();path=tmp_path/'records/train/ref_F02_si001.f0';path.parent.mkdir(parents=True);path.write_text('200 1 0 0\n')
    with pytest.raises(ValueError,match='freeze'):ledger.read(path,'reference',split='train')
    ledger.predictions_frozen=True
    assert ledger.read(path,'reference',split='train')==b'200 1 0 0\n'
    with pytest.raises(ValueError,match='speaker'):ledger.read(path.with_name('ref_F05_si001.f0'),'reference',split='train')
    with pytest.raises(ValueError,match='test content'):ledger.read(tmp_path/'sealed-test/anything','features')
    with pytest.raises(ValueError,match='split/path'):ledger.read(path,'reference',split='development')


def test_cached_gate_and_candidate_identity_fail_closed():
    refs,feature,mlp,yin=example();mlp['f0_hz'][5]=150
    with pytest.raises(ValueError,match='selected candidate'):sh.validate_pair(feature,mlp,yin)
