import inspect
import numpy as np
import pytest
from phenometrix_prosody.alignment import (OFFSETS_MS,SUPPORTS_MS,detect_strong_cycles,
    cycle_values,estimate_grid,compare_reference)


def manual_cycles():
    start=np.arange(0,96000,240,dtype=float)
    return {'start':start,'end':start+240,'hz':100+start/1000,'strength':np.ones(len(start))}


def test_constant_cycle_frequency_and_no_gap_filling():
    cycles={'start':np.array([0.,480.,1440.]),'end':np.array([480.,960.,1920.]),'hz':np.array([100.,100.,100.])}
    values=cycle_values(cycles,[240,700,1200,1600])
    np.testing.assert_array_equal(values[[0,1,3]],100)
    assert np.isnan(values[2])
    assert np.isnan(cycle_values(cycles,[1200],(-16,16))[0])
    assert cycle_values(cycles,[480],(-5,5))[0]==100


def test_window_semantics_are_not_clock_shifts():
    cycles=manual_cycles();query=np.arange(5000,90000,480)
    trailing=cycle_values(cycles,query,(-40,0))
    shifted_center=cycle_values(cycles,query-960,(-20,20))
    np.testing.assert_array_equal(trailing,shifted_center)
    assert np.all(trailing<cycle_values(cycles,query,(-20,20)))


def test_offset_sign_and_common_denominators_known_answer():
    cycles=manual_cycles();grid,estimates,transition=estimate_grid(cycles,96000)
    f0=cycle_values(cycles,grid-960)
    refs=np.column_stack([np.nan_to_num(f0),np.isfinite(f0).astype(float),np.ones(len(grid)),np.ones(len(grid))])
    result,errors=compare_reference(refs,estimates,transition)
    assert OFFSETS_MS==(-30,-20,-10,0,10,20,30)
    assert set(SUPPORTS_MS)=={'cycle_interval','centered32','trailing32','centered40','trailing40'}
    assert np.all(errors[('all_common_voiced','cycle_interval',-20)]==0)
    assert np.median(np.abs(errors[('all_common_voiced','cycle_interval',0)]))>0
    assert len({result['rows'][('all_common_voiced',mode,offset)]['n'] for mode in SUPPORTS_MS for offset in OFFSETS_MS})==1


def test_raw_cycle_detector_recovers_strong_synthetic_period_without_labels():
    sr=48000;t=np.arange(sr)/sr
    raw=.2*np.sin(2*np.pi*120*t)+.00001*np.random.default_rng(8).normal(size=len(t))
    for polarity in (1,-1):
        cycles=detect_strong_cycles(raw,polarity=polarity)
        assert len(cycles['hz'])>90
        assert np.median(cycles['hz'])==pytest.approx(120,rel=.005)
    assert set(inspect.signature(detect_strong_cycles).parameters)=={'raw_lar','sample_rate','polarity'}
    assert set(inspect.signature(estimate_grid).parameters)=={'cycles','sample_count','sample_rate'}
    assert not len(detect_strong_cycles(np.zeros(48000))['hz'])
