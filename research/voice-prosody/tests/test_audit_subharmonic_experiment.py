import numpy as np
import pytest
from phenometrix_prosody import audit_subharmonic_experiment as a


def test_symmetric_aliases_include_all_factors_and_exclude_acceptable():
    hz=np.array([[100.,200.,300.,400.,600.,1200.],[25.,100/3,50.,100.,175.,np.nan]])
    good=np.array([[False,False,True,False,False,False],[False,False,False,True,False,False]])
    masks=a.independent_negative_masks(hz,good,np.array([True,True]))
    np.testing.assert_array_equal(masks['alias-margin'],[[True,False,False,False,True,True],[True,True,True,False,False,False]])
    assert not np.any(masks['alias-margin']&good)
    assert np.all(masks['alias-margin']<=masks['all-margin'])


def test_aliases_use_any_acceptable_actual_member_not_reference():
    hz=np.array([[100.,101.,202.,305.,np.nan,np.nan]])
    good=np.array([[True,True,False,False,False,False]])
    masks=a.independent_negative_masks(hz,good,np.array([True]))
    assert masks['alias-margin'][0,2] and masks['alias-margin'][0,3]
    assert masks['all-margin'].sum()==2


def test_uncovered_and_padding_are_never_margin_negatives():
    hz=np.array([[100.,200.,300.,np.nan,np.nan,np.nan],[np.nan]*6])
    masks=a.independent_negative_masks(hz,np.zeros((2,6),bool),np.zeros(2,bool))
    assert not any(v.any() for v in masks.values())
    with pytest.raises(ValueError,match='Invalid acceptable'):
        a.independent_negative_masks(hz,np.zeros((2,6),bool),np.array([True,False]))


def test_alias_tolerance_has_no_extra_epsilon():
    hz=np.array([[100.,200*2**(49.999/1200),200*2**(50.001/1200)]])
    good=np.array([[True,False,False]])
    np.testing.assert_array_equal(a.independent_negative_masks(hz,good,np.array([True]))['alias-margin'],[[False,True,False]])


def test_analytic_success_keeps_alias_unvoiced_and_unavailable_in_denominator():
    result=a.analytic_summary(np.full(4,337.1),np.array([1,1,0,2]),np.array([337.1,337.1/3,np.nan,np.nan]),np.array(['after-observation-change']*4),0)
    assert result['frames']==4 and result['success_within_50c_all_frames']==.25
    assert result['pitch_coverage']==.5 and result['nonvoiced_or_unavailable']==2


def test_new_metadata_has_exact_count_support_and_disjoint_frequencies():
    new=a.expected_challenge_records();old=a.historical.expected_challenge_records()
    assert len(new)==12 and len({r['id'] for r in new})==12
    assert not ({r['frequency_hz'] for r in new}&{r['frequency_hz'] for r in old})
    assert all(r['sample_count']==96000 and r['change_sample']==39840 for r in new)
    ends=np.arange(2208,96001,480);target=ends-960;eligible=(target>=3072)&(target+1536<=96000)
    assert eligible.sum()==191 and (eligible&(ends-1920>=39840)).sum()==112


def test_fixed_gate_method_and_case_inventory():
    assert len(a.NEW)==9 and len(a.PRIOR)==3 and len(a.METHODS)==14
    assert a.CANDIDATE==[f'alias-margin__seed{s}' for s in (20261022,20261023,20261024)]
