import numpy as np
import pytest
from phenometrix_prosody import audit_augmentation_experiment as a


def test_exact_paired_sampler_is_without_replacement_and_domain_separated():
    actual=a.paired_schedule(20261022,0)
    expected=np.random.default_rng(21261022).permutation(18336)[:16896].reshape(33,512)
    np.testing.assert_array_equal(actual,expected)
    assert actual.shape==(33,512) and len(np.unique(actual))==16896
    assert not np.array_equal(actual,a.paired_schedule(20261022,1))
    with pytest.raises(ValueError):a.paired_schedule(7,0)
    with pytest.raises(ValueError):a.paired_schedule(20261022,12)


def test_synthetic_targets_preserve_uncovered_and_multiple_acceptable():
    hz=np.array([[100.,300.,302.,np.nan,np.nan,np.nan],[100.,150.,np.nan,np.nan,np.nan,np.nan]])
    acceptable,covered=a.synthetic_acceptable(hz,np.array([300.,300.]))
    np.testing.assert_array_equal(acceptable,[[False,True,True,False,False,False],[False]*6])
    np.testing.assert_array_equal(covered,[True,False])


def test_synthetic_tolerance_does_not_widen_fifty_cents():
    acceptable,_=a.synthetic_acceptable(np.array([[200*2**(49.999/1200),200*2**(50.001/1200)]]),np.array([200.]))
    np.testing.assert_array_equal(acceptable,[[True,False]])
    with pytest.raises(ValueError):a.synthetic_acceptable(np.array([[100.]]),np.array([np.nan]))


def metadata():
    scenario=np.repeat(np.arange(96),191);target=np.tile(np.arange(3168,3168+191*480,480),96)
    good=np.zeros((18336,6),bool);good[:,0]=True
    return {'reference_hz':np.full(18336,200.),'scenario_index':scenario,'reference_sample':target,
            'availability_sample':target+960,'region':np.where(target+960<=39840,'before-observation-change',np.where(target+960-1920>=39840,'after-observation-change','observation-boundary')),'covered':good.any(-1),'acceptable':good}


def test_paired_metadata_allows_coverage_difference_but_not_target_shift():
    left=metadata();right={k:v.copy() for k,v in left.items()}
    right['acceptable'][0]=False;right['covered'][0]=False
    assert a.validate_paired_target_metadata(left,right)
    right['reference_sample'][0]+=480
    with pytest.raises(ValueError,match='Unpaired'):a.validate_paired_target_metadata(left,right)


def test_exposure_counts_uncovered_rows_in_unconditional_denominator():
    schedule=a.paired_schedule(20261022,0);m=metadata();m['covered'][schedule[0]]=False
    result=a.sampler_exposure(schedule,m['scenario_index'],m['covered'])
    assert result['selected_rows']==16896 and result['uncovered_zero_rows']==512 and result['covered_rows']==16384
    assert sum(result['scenario_rows'])==16896
    schedule[0,0]=schedule[0,1]
    with pytest.raises(ValueError,match='duplicates'):a.sampler_exposure(schedule,m['scenario_index'],m['covered'])


def test_training_and_fresh_specs_are_metadata_only_paired_and_disjoint():
    training=a.expected_training_records();fresh=a.expected_challenge_records()
    assert len(training['stationary-aug'])==len(training['balance-aug'])==96 and len(fresh)==12
    for left,right in zip(training['stationary-aug'],training['balance-aug']):
        assert {k:v for k,v in left.items() if k not in ('id','arm','after')}=={k:v for k,v in right.items() if k not in ('id','arm','after')}
        assert left['after']==left['before']==[1.,.5,.25] and right['after']==[0.,1.,.5]
    train_hz={v['frequency_hz'] for v in training['balance-aug']}
    assert train_hz.isdisjoint(v['frequency_hz'] for v in fresh)
    assert {337.1,354.5}.isdisjoint(train_hz)
    assert len(a.NEW)==9 and len(a.METHODS)==14 and len(a.SELECTORS)==12


def test_unconditional_analytic_success_keeps_errors_and_absence():
    result=a.analytic_summary(np.full(4,354.5),np.array([1,1,0,2]),np.array([354.5,177.25,np.nan,np.nan]),np.array(['after-observation-change']*4),0)
    assert result['frames']==4 and result['success_within_50c_all_frames']==.25
    assert result['nonvoiced_or_unavailable']==2
