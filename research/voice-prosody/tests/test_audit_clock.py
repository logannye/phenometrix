import pytest
from phenometrix_prosody.audit_clock import affine_interval, analyze_probes

MAPPING = {'id': 1, 'mainPerformanceMs': 1000., 'audioContextTimeSeconds': 0.}


def probes(rate_ppm=30.):
    rows=[]
    for i in range(8):
        audio=i*60.
        projected=1000.+audio*1000
        residual=7.+audio*rate_ppm/1000
        main=projected+residual
        rows.append({'schemaVersion':'phenometric.voice-clock-observation.v1','probeId':i+1,
            'captureEpoch':1,'clockMappingId':1,'sourceElapsedMs':audio*1000,'contextState':'running',
            'audioClockUncertaintyMs':None,
            'contextRead':{'beforeMs':main-.1,'afterMs':main+.1,'midpointMs':main,
                'audioContextTimeSeconds':audio,'bracketWidthMs':.2,'fixedAnchorResidualMs':residual},
            'outputTimestamp':{'status':'observed','coordinate':'output-device-only-not-capture',
                'beforeMs':main-.1,'afterMs':main+.1,
                'contextTimeSeconds':audio,'performanceTimeMs':main+20.,'fixedAnchorResidualMs':residual+20.},
            'renderObservation':{'status':'observed','coordinate':'render-callback-main-performance-bracket',
                'renderStartFrame':int(audio*48000),'sampleRateHz':48000,'requestSentAtMs':main-2.,
                'responseReceivedAtMs':main+2.,'roundTripMs':4.,
                'fixedAnchorResidualLowerMs':residual-2.,'fixedAnchorResidualUpperMs':residual+2.}})
    return rows


def test_independent_coordinates_recover_rate_without_arrival_data():
    result=analyze_probes(probes(),MAPPING)
    for trend in result['descriptive_trends'].values():
        assert trend['rate_difference_ppm']==pytest.approx(30.)
    feasible=result['all_probe_affine_consistency']
    assert feasible['feasible']
    assert feasible['lower_residual_rate_ppm']<=30<=feasible['upper_residual_rate_ppm']


def test_request_reply_delay_widens_causal_bound_and_is_not_removed():
    rows=probes();r=rows[4]['renderObservation']
    r['responseReceivedAtMs']+=350;r['roundTripMs']+=350;r['fixedAnchorResidualUpperMs']+=350
    result=analyze_probes(rows,MAPPING)
    assert result['one_minute_clock_windows']['4']['render_round_trip_ms']['maximum']==354
    assert result['descriptive_trends']['context_read_residual']['rate_difference_ppm']==pytest.approx(30.)
    assert result['all_probe_affine_consistency']['feasible']


def test_forged_precomputed_clock_residual_is_rejected():
    rows=probes();rows[3]['renderObservation']['fixedAnchorResidualLowerMs']+=1
    with pytest.raises(ValueError,match='independent reconstruction'):
        analyze_probes(rows,MAPPING)


def test_timeouts_are_retained_and_output_is_not_required():
    rows=probes();rows[2]['renderObservation']={'status':'unavailable','reason':'timeout'}
    rows[2]['outputTimestamp']={'status':'unavailable','reason':'unsupported'}
    result=analyze_probes(rows,MAPPING)
    assert result['counts']['probes']==8
    assert result['counts']['render_observed']==7
    assert result['counts']['render_unavailable_timeout']==1
    assert result['counts']['output_unavailable']==1


def test_non_affine_probe_positions_are_not_smoothed_into_a_fit():
    result=affine_interval([(0.,0.,1.),(100.,0.,1.),(200.,30.,31.)])
    assert not result['feasible']


def test_retired_mapping_is_rejected():
    rows=probes();rows[2]['clockMappingId']=0
    with pytest.raises(ValueError,match='mapping identity'):
        analyze_probes(rows,MAPPING)


@pytest.mark.parametrize("field,value", [("captureEpoch",2),("contextState","suspended")])
def test_probe_timeline_changes_are_not_mixed(field,value):
    rows=probes();rows[3][field]=value
    with pytest.raises(ValueError,match='epoch or running state'):
        analyze_probes(rows,MAPPING)


def test_render_position_reversal_is_rejected():
    rows=probes();rows[3]['renderObservation']['renderStartFrame']=0
    with pytest.raises(ValueError,match='sample position/rate'):
        analyze_probes(rows,MAPPING)
