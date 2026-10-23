import numpy as np
import pytest
from phenometrix_prosody import audit_harmonic_experiment as a


def test_analytic_unconditional_success_retains_unvoiced_and_alias():
    s=a.analytic_summary(np.full(3,300.),np.array([1,1,0]),np.array([300.,100.,np.nan]),np.array(['plateau']*3),0)
    assert s['frames']==3 and s['voiced_outputs']==2 and s['nonvoiced_or_unavailable']==1
    assert s['success_within_50c_all_frames']==1/3 and s['pitch_coverage']==2/3
    assert s['p95_absolute_cents']==pytest.approx(1200*np.log2(3))
    assert s['median_signed_equivalent_slope_delay_ms'] is None


def test_independent_step_preserves_negative_target_lag():
    s={'changeStartSeconds':.1,'finalHz':200}
    target=np.array([3840,4320,4800]);end=target+960
    result=a.independent_step(s,target,end,np.ones(3,int),np.full(3,200.))
    assert result=={'target_time_first_of_three_settling_lag_ms':-20.,'earliest_availability_first_of_three_delay_ms':0.,'settled':True}


def test_fixed_challenge_spec_is_metadata_only_and_disjoint():
    records=a.expected_challenge_records();assert len(records)==12
    assert len({r['id'] for r in records})==12
    assert all(r['change_sample']==39840 and r['sample_count']==96000 for r in records)
    assert {r['frequency_hz'] for r in records}=={87.3,137.9,223.7,337.1,467.3,613.7}


def test_regions_use_whole_support_and_fixed_latent_frequency():
    s={'kind':'observation-change','initialHz':337.1,'changeStartSeconds':.83}
    ends=np.array([39648,40128,42048]);assert a.regions(s,ends).tolist()==['before-observation-change','observation-boundary','after-observation-change']
    np.testing.assert_array_equal(a.curve(s,(ends-960)/48000),np.full(3,337.1))
