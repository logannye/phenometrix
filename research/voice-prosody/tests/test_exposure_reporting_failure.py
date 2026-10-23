"""Document the preserved exposure-study stdout failure; never repair or rerun it.

Only fixed tiny score arrays and the pre-existing mocked lifecycle fixture are
used. The frozen runner stays unchanged. These tests intentionally expect its
terminal TypeError and distinguish successfully saved evidence from execution
status. They are post-run regression documentation, not prospective validation.
"""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from phenometrix_prosody import exposure_study as study
from phenometrix_prosody.evaluate_synthetic import summarize
from phenometrix_prosody.augmentation_experiment import save
from test_exposure_study import analytic_fixture, complete_fixture


def fixed_summary(last_error=49.):
    return summarize([{'signed_cents':v,'equivalent_slope_delay_ms':None}
                      for v in (-20.,0.,last_error)])


def inject_real_score_scalar(groups,score):
    for cases in groups.values():
        for row in cases.values():
            for method in row['methods'].values():
                method['all']=dict(score)
                method['regions']['after-observation-change']=dict(score)
    return groups


@pytest.mark.parametrize('last_error,expected_passed',[(49.,78),(100.,0)])
def test_preserved_numpy_gate_stdout_failure_does_not_corrupt_saved_metrics(tmp_path,last_error,expected_passed):
    score=fixed_summary(last_error)
    assert isinstance(score['success_within_50c_all_frames'],np.floating)
    decision=study.analytic_decision(inject_real_score_scalar(analytic_fixture(),score))
    first_seed=next(iter(decision['by_seed_case']))
    assert isinstance(next(iter(decision['by_seed_case'][first_seed].values())),np.bool_)
    assert isinstance(decision['passed_by_seed'][first_seed],np.integer)
    assert type(decision['pass']) is bool
    # This is the exact unadapted stdout encoder at the archived failing line.
    with pytest.raises(TypeError,match='not JSON serializable'):
        json.dumps(decision,indent=2)
    path=tmp_path/'metrics.json';save(path,{'decision':{'analytic':decision}})
    restored=json.loads(path.read_text())['decision']['analytic']
    assert restored['passed_by_seed']=={seed:expected_passed for seed in decision['by_seed_case']}
    assert restored['pass'] is bool(expected_passed)
    assert all(type(value) is bool for rows in restored['by_seed_case'].values() for value in rows.values())


def test_mock_lifecycle_reproduces_reporting_failure_after_evidence_is_saved(tmp_path,monkeypatch):
    # Existing fake checkpoints/signals/inference remain mocks. Only the score
    # scalar comes from the real historical summarize() rather than literal1.0.
    output,manifest,preparation,events=complete_fixture(tmp_path,monkeypatch)
    fake_scoring=study.harmonic.analytic_scores
    def scoring_with_actual_scalar(predictions,items):
        result=fake_scoring(predictions,items)
        inject_real_score_scalar({'fixture':result},fixed_summary())
        return result
    monkeypatch.setattr(study,'harmonic',SimpleNamespace(
        original_probability=study.harmonic.original_probability,
        analytic_scores=scoring_with_actual_scalar))
    with pytest.raises(TypeError,match='not JSON serializable'):
        study.train_and_evaluate(output,manifest,preparation,study.time.monotonic())
    assert events==['references-train','references-development','fit','fresh','references-development']
    saved=json.loads((output/'metrics.json').read_text())
    assert saved['decision']['analytic']['passed_by_seed']=={study.key(study.CANDIDATE,s):78 for s in study.SEEDS}
    assert len(json.loads((output/'prediction-freeze.json').read_text())['prediction_sha256'])==103
    assert len(manifest['fits'])==12 and manifest['fresh_evaluation_configurations']==12
    assert manifest['metrics_sha256']==study.ev.sha(output/'metrics.json')
    # Direct helper reached its temporary complete write. The real run() outer
    # catch subsequently records failed; this fixture does not call run().
    assert manifest['status']=='complete'
