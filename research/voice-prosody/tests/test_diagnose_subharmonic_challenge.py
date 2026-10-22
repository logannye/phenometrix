import numpy as np
from phenometrix_prosody.diagnose_subharmonic_challenge import support_region,coverage,region_counts


def test_exact_new_and_original_observation_boundaries_are_preserved():
    new={'kind':'observation-change','change_sample':39840,'changeStartSeconds':.83}
    assert [support_region(new,e) for e in (39840,39841,41759,41760)]==['before-observation-change','observation-boundary','observation-boundary','after-observation-change']
    old={'kind':'observation-change','changeStartSeconds':.6}
    assert support_region(old,28800)=='observation-boundary'
    assert support_region(old,30720)=='after-observation-change'


def test_step_and_glide_boundaries_keep_full_support():
    glide={'kind':'glide','changeStartSeconds':.4,'changeEndSeconds':1.2}
    assert support_region(glide,21000)=='transition-boundary'
    assert support_region(glide,21120)=='glide-interior'
    assert support_region(glide,57600)=='transition-boundary'
    assert support_region(glide,59520)=='plateau'


def test_vector_truth_coverage_and_unconditional_failure_partition():
    truth=np.array([100,200,300,400.]);mask=np.array([[1,0],[1,1],[0,0],[1,0]],bool)
    hz=np.array([[100,np.nan],[100,200],[np.nan,np.nan],[200,np.nan]])
    cov=coverage(hz,mask,truth)
    assert cov.tolist()==[True,True,False,False]
    result=region_counts(np.ones(4,bool),cov,cov,mask.any(-1),np.array([.7,.8,.6,.2]),truth,
        {'toy':(np.array([1,1,2,0]),np.array([100,100,np.nan,np.nan]))})
    assert result['frames']==4 and result['raw_correct_candidate_frames']==2
    assert result['original_probability_below_half_frames']==1
    assert result['methods']['toy']['success_within_50c_all_frames']==.25
    assert [result['methods']['toy'][k] for k in ('correct_pitch','incorrect_pitch','unvoiced','unavailable')]==[1,1,1,1]
    assert result['methods']['toy']['incorrect_with_grouped_correct_candidate']==1
