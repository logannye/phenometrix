import numpy as np
import pytest
from phenometrix_prosody.diagnose_augmentation_challenge import diagnostic_regions,MODELS,support_region


def test_unconditional_partition_and_fixed_alias_labels_include_all_failures():
    truth=np.array([360.,360.,360.,360.,360.]);state=np.array([1,1,1,0,2]);hz=np.array([360,180,120,np.nan,np.nan])
    include=np.ones(5,bool);cov=np.array([1,1,1,1,0],bool);available=cov.copy();p=np.array([.8,.8,.8,.2,.8])
    result=diagnostic_regions(include,cov,cov,available,p,truth,{'toy':(state,hz)})
    row=result['methods']['toy']
    assert [row[k] for k in ('correct_pitch','incorrect_pitch','unvoiced','unavailable')]==[1,2,1,1]
    assert row['success_within_50c_all_frames']==.2
    assert row['incorrect_alias_counts']=={'quarter':0,'third':1,'half':1,'double':0,'triple':0,'quadruple':0}
    assert row['incorrect_pitch_summary']['min_hz']==120 and row['incorrect_pitch_summary']['max_hz']==180
    assert result['frames']==5 and result['original_probability_below_half_frames']==1


def test_zero_error_case_keeps_explicit_zero_alias_counts():
    ones=np.ones(2,bool);truth=np.array([100.,300.])
    result=diagnostic_regions(ones,ones,ones,ones,np.array([.9,.9]),truth,{'toy':(np.ones(2,int),truth)})
    assert result['methods']['toy']['incorrect_pitch_summary'] is None
    assert not any(result['methods']['toy']['incorrect_alias_counts'].values())
    assert result['methods']['toy']['correct_pitch']==2


def test_new_model_inventory_and_integer_regions_preserved():
    assert len(MODELS)==12
    assert {f'balance-aug__seed{s}' for s in (20261022,20261023,20261024)}<=MODELS
    scenario={'kind':'observation-change','changeStartSeconds':.83,'change_sample':39840}
    assert support_region(scenario,39840)=='before-observation-change'
    assert support_region(scenario,41759)=='observation-boundary'
    assert support_region(scenario,41760)=='after-observation-change'


def test_false_voicing_or_candidate_availability_is_rejected():
    with pytest.raises(ValueError,match='voice gate'):
        diagnostic_regions(np.ones(1,bool),np.ones(1,bool),np.ones(1,bool),np.ones(1,bool),np.array([.2]),np.array([100.]),{'toy':(np.ones(1,int),np.array([100.]))})
