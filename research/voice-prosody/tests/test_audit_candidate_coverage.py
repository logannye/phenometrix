import numpy as np
import pytest
from phenometrix_prosody import audit_candidate_coverage as a


def test_fifty_cents_and_absent_candidates():
    hz=np.array([[200*2**(49.999/1200),np.nan],[200*2**(50.001/1200),200.],[200.,200.]])
    mask=np.array([[True,False],[True,False],[True,True]])
    np.testing.assert_array_equal(a.covered_by_candidates(hz,mask,np.array([200.,200.,np.nan]),[True,True,False]),[True,False,False])
    with pytest.raises(ValueError,match='candidate'):
        a.covered_by_candidates([[np.nan]],[[True]],[200.],[True])


def test_grouping_loss_is_actual_lost_and_nonmember_gain_rejected():
    assert a.coverage_counts([True]*4,[True,True,False,False],[True,False,False,False])=={
        'voiced':4,'raw':2,'grouped':1,'lost':1,'gained':0}
    with pytest.raises(ValueError,match='gained'):
        a.coverage_counts([True,True],[True,False],[False,True])


def test_equal_net_coverage_does_not_hide_eviction():
    c=a.policy_change_counts([True]*4,[True,True,False,False],[True,False,True,False])
    assert c=={'voiced':4,'old_covered':2,'new_covered':2,'retained':1,'lost':1,'gained':1,'neither':1}


def test_oracle_keeps_voicing_errors_and_unconditional_denominator():
    c=a.fixed_voicing_oracle([True]*4+[False],[True,True,False,False,True],
                            [.5,.49,.8,.2,.9],[True,True,False,False,False])
    assert c['eligible_valid_frames']==4 and c['correct_frames']==2 and c['joint_error']==.5
    assert c['shared_false_unvoiced_frames']==c['shared_false_positive_frames']==1
    assert c['oracle_correct_voiced_frames']==1
    with pytest.raises(ValueError,match='outside'):
        a.fixed_voicing_oracle([True],[False],[.1],[True])


def test_paired_feature_contract_allows_only_candidates_and_count():
    f={'features':np.zeros((2,57),np.float32),'candidate_hz':np.full((2,6),np.nan),'candidate_mask':np.zeros((2,6),bool),'reference_sample':np.array([2208,2688]),
       'feature_names':np.arange(57).astype(str),'sample_rate_hz':np.array(48000),'pcm_sample_count':np.array(96000)}
    g={k:v.copy() for k,v in f.items()};g['features'][:,:48]=1;g['features'][:,55]=.5
    assert a.paired_feature_identity(f,g)
    g['features'][0,54]=.1
    with pytest.raises(ValueError,match='non-candidate'):
        a.paired_feature_identity(f,g)
    g={k:v.copy() for k,v in f.items()};g['reference_f0_hz']=np.array([100.,100.])
    with pytest.raises(ValueError,match='field inventory'):
        a.paired_feature_identity(f,g)


def test_exact_frozen_checkpoint_and_method_inventory():
    assert len(a.CHECKPOINT_SHA)==3 and len(a.METHODS)==8 and len(a.SELECTORS)==6
    assert a.CANDIDATE==[f'revised__seed{s}' for s in (20261022,20261023,20261024)]


def test_reference_guard_uses_manifest_role_not_shared_split_directory():
    ref='research/voice-prosody/data/ptdb-tug/records/train/F02/ref_F02_si645.f0'
    mic='research/voice-prosody/data/ptdb-tug/records/train/F02/mic_F02_si645.wav'
    a.no_reference_inputs([mic],[ref])
    with pytest.raises(ValueError,match='Reference content'):
        a.no_reference_inputs([mic,ref],[ref])


def candidate_feature(rows):
    from phenometrix_prosody.model import FEATURE_NAMES_V2
    x=np.zeros((1,57),np.float32);hz=np.full((1,6),np.nan);mask=np.zeros((1,6),bool)
    for j,(frequency,scores) in enumerate(rows):
        hz[0,j]=frequency;mask[0,j]=True
        x[0,j*8:(j+1)*8]=[1,np.log2(frequency/50),*scores,max(scores)]
    x[0,55]=np.float32(len(rows)/6)
    return {'features':x,'candidate_hz':hz,'candidate_mask':mask,'reference_sample':np.array([2208]),
            'feature_names':np.array(FEATURE_NAMES_V2),'sample_rate_hz':np.array(48000),'pcm_sample_count':np.array(2208)}


def test_replacement_reconstructs_collision_without_erasing_other_evidence():
    old=candidate_feature([(100.,[.7,0,0,.9,0]),(200.,[0,.4,0,0,0])])
    new=candidate_feature([(100.,[.7,0,0,0,0]),(200.,[0,.4,0,.6,0])])
    provenance={'old_full40_hz':np.array([100.]),'old_full40_confidence':np.array([.9]),
        'new_full40_hz':np.array([200.]),'new_full40_confidence':np.array([.6]),
        'full40_changed':np.array([True]),'old_frequency_evicted':np.array([False]),'new_frequency_added':np.array([False]),
        'old_candidate_count':np.array([2]),'new_candidate_count':np.array([2]),
        'candidate_policy':np.array('shortest-admitted-full40-v1'),'pcm_sha256':np.array('abc'),'worker_rss_bytes':np.array(1024)}
    assert a.validate_replacement(old,new,provenance,'abc')=={'frames':1,'full40_changed':1,'old_frequency_evicted':0,'new_frequency_added':0}
    new['features'][0,3]=.1
    with pytest.raises(ValueError,match='evidence'):
        a.validate_replacement(old,new,provenance,'abc')


def test_fresh_metadata_keeps_twelve_cases_and_existing_denominators():
    rows=a.fresh_records()
    assert len(rows)==12 and {r['frequency_hz'] for r in rows}=={97.1,213.7,347.3,529.1,627.7,671.3}
    assert all(r['sample_count']==96000 and r['change_sample']==39840 and r['gain']==.055 for r in rows)


def test_equal_cell_grouping_gate_is_not_pooled_or_net():
    result=a.finish_coverage({'F01/clean':{'voiced':100,'raw':100,'grouped':99,'lost':1,'gained':0},
                             'M01/clean':{'voiced':10000,'raw':10000,'grouped':10000,'lost':0,'gained':0}})
    assert result['lost_fraction']==.005 and result['pass'] is True
    result['cells'][0]['lost']=2
    bad=a.finish_coverage({'F01/clean':{'voiced':100,'raw':100,'grouped':98,'lost':2,'gained':0},
                          'M01/clean':{'voiced':10000,'raw':10000,'grouped':10000,'lost':0,'gained':0}})
    assert bad['lost_fraction']==.01 and bad['pass'] is False


def test_step_diagnostic_retains_numerics_and_exact_explanatory_metadata():
    scenario={'changeStartSeconds':.1,'finalHz':200.}
    target=np.array([4320,4800,5280]);end=target+960
    got=a.step_summary(scenario,target,end,np.ones(3,int),np.full(3,200.))
    assert got=={'target_time_first_of_three_settling_lag_ms':-10.,
        'earliest_availability_first_of_three_delay_ms':10.,'settled':True,
        'criterion':'first of 3 consecutive final-frequency outputs within50c; acquisition end>=physical step; no negative-lag clamp'}
