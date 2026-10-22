"""Independent denominator and gate checks for the bounded selector comparison."""
import copy
import numpy as np
import pytest
from phenometrix_prosody.selector_experiment import (
    score_arrays, rate_counts, natural_decision, model_key, SEEDS, CONDITIONS, load_inputs)
from phenometrix_prosody.metrics import Reference, Prediction, score


def test_vector_counts_match_independent_object_scorer_with_abstentions():
    truth = np.array([100., 100., 100., 100., np.nan, np.nan, np.nan, 200., 200.])
    voiced = np.isfinite(truth)
    valid = np.array([True]*8+[False])
    state = np.array([1, 1, 0, 2, 0, 1, 2, 1, 1], np.uint8)
    hz = np.array([100., 50., np.nan, np.nan, np.nan, 100., np.nan, 220., 200.])
    got = score_arrays({'hz':truth, 'voiced':voiced, 'valid':valid}, state, hz)
    refs = [Reference(i, float(h) if v else None, bool(v), bool(ok)) for i,(h,v,ok) in enumerate(zip(truth,voiced,valid))]
    preds = [Prediction(i, ['unvoiced','voiced','unavailable'][s], float(h) if s==1 else None) for i,(s,h) in enumerate(zip(state,hz))]
    expected = score(refs,preds)
    for key,value in got.items():
        if key in expected:
            assert value == expected[key], key
    assert got['eligible_valid_frames'] == 8
    assert got['reference_voiced_success_rate'] == .2
    assert got['voiced_pitch_error_count'] == 2
    assert got['joint_error'] == .75


def example_results():
    def row(error, success=.8):
        return {'macro':{'joint_error':error,'reference_voiced_success_rate':success},
                'per_condition':{c:{'joint_error':error} for c in CONDITIONS}}
    out={'original_mlp':row(.10),'original_hybrid':row(.11)}
    for seed in SEEDS:
        out[model_key('shared-candidates','acceptable-set',seed)]=row(.09,.81)
        out[model_key('fixed-slots','nearest',seed)]=row(.095,.805)
    return out


def test_decision_requires_every_seed_and_condition_even_when_mean_improves():
    result=example_results()
    assert natural_decision(result)['pass']
    name=model_key('shared-candidates','acceptable-set',SEEDS[0])
    result[name]['macro']['joint_error']=.101
    assert not natural_decision(result)['checks']['each_seed_improves_original_mlp']
    result=example_results()
    for seed in SEEDS:
        result[model_key('shared-candidates','acceptable-set',seed)]['per_condition']['first_difference']['joint_error']=.111
    assert not natural_decision(result)['checks']['no_condition_worsens_over_one_point']


def test_abstention_does_not_improve_unconditional_metric():
    lab={'hz':np.array([100.,100.]),'voiced':np.array([True,True]),'valid':np.array([True,True])}
    baseline=score_arrays(lab,np.array([1,1]),np.array([100.,200.]))
    withheld=score_arrays(lab,np.array([1,2]),np.array([100.,np.nan]))
    assert baseline['joint_error']==withheld['joint_error']==.5
    assert withheld['measurement_coverage']==.5


def test_reserved_split_rejected_before_any_file_access():
    with pytest.raises(ValueError,match='Only train/development'):
        load_inputs('sealed_test')


def test_invalid_prediction_state_rejected():
    with pytest.raises(ValueError,match='Invalid state'):
        score_arrays({'hz':np.array([100.]),'voiced':np.array([True]),'valid':np.array([True])},np.array([1]),np.array([np.nan]))
