import numpy as np
import pytest
from phenometrix_prosody.diagnose_harmonic_challenge import decompose, within


def test_partition_preserves_unvoiced_unavailable_and_uncovered_frames():
    state=np.array([1,1,0,2,1,1,1]);hz=np.array([200,100,np.nan,np.nan,400,100,200.])
    included=np.array([True]*6+[False]);raw=np.array([1,1,1,0,0,1,1],bool);group=np.array([1,1,1,0,0,0,1],bool)
    r=decompose(state,hz,200,included,raw,group)
    assert r['frames']==6 and r['success_within_50c_all_frames']==1/6
    assert r['counts']=={'correct_pitch':1,'incorrect_pitch':3,'unvoiced':1,'unavailable':1}
    assert r['outcome_by_candidate_coverage']['raw_correct_candidate_absent']=={'correct_pitch':0,'incorrect_pitch':1,'unvoiced':0,'unavailable':1}
    assert r['outcome_by_candidate_coverage']['correct_candidate_lost_in_grouping']['incorrect_pitch']==1
    assert r['incorrect_pitch_summary']['near_half_f0_frames']==2
    assert r['incorrect_pitch_summary']['near_double_f0_frames']==1


def test_coverage_masks_unavailable_hz_and_keeps_50c_correctness():
    hz=np.array([[200,np.nan],[201,400],[100,400.]])
    mask=np.array([[1,0],[1,1],[1,1]],bool)
    assert within(hz,mask,200).tolist()==[True,True,False]


def test_rejects_invented_grouped_coverage():
    with pytest.raises(ValueError,match='invented'):
        decompose(np.array([1]),np.array([200.]),200,np.array([True]),np.array([False]),np.array([True]))


def test_rejects_false_validity_without_reclassifying_as_silence():
    with pytest.raises(ValueError,match='State and emitted'):
        decompose(np.array([0]),np.array([200.]),200,np.array([True]),np.array([True]),np.array([True]))
