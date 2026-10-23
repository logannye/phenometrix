import json
import numpy as np
import pytest
from phenometrix_prosody import synthetic_curriculum as sc


def test_frozen_plans_have_bounded_counts_and_disjoint_parameters_without_generating_challenge():
    plans = sc.make_plans()
    assert len(plans['train']['records']) == 128 and plans['train']['duration_seconds'] == 320
    assert len(plans['challenge']['records']) == 32 and plans['challenge']['duration_seconds'] == 80
    train, challenge = plans['train']['records'], plans['challenge']['records']
    assert not {r['id'] for r in train} & {r['id'] for r in challenge}
    assert not {r['seed'] for r in train} & {r['seed'] for r in challenge}
    assert not {sc.parameter_digest(r) for r in train} & {sc.parameter_digest(r) for r in challenge}
    assert sum(r['latent_voiced'] for r in train) == 120
    assert sum(r['latent_voiced'] for r in challenge) == 28
    assert all(len(r['harmonic_amplitudes']) == 6 for r in challenge)
    for record in train + challenge:
        assert record['harmonic_amplitudes'][1] > 0 and record['harmonic_amplitudes'][2] > 0


def test_phase_derivative_matches_train_step_glide_and_constant_truth():
    records = sc.make_plans()['train']['records']
    for kind in ('stationary', 'glide', 'step', 'observation-change'):
        record = next(r for r in records if r['kind'] == kind)
        for t in (.2, .73, 1.33, 2.2):
            derivative = (sc.cycles_at(record, t + 1e-6) - sc.cycles_at(record, t - 1e-6)) / 2e-6
            assert derivative == pytest.approx(sc.frequency_at(record, t), rel=1e-8)


def test_analytic_references_use_nominal_grid_not_feature_endpoint_shift():
    record = next(r for r in sc.make_plans()['train']['records'] if r['kind'] == 'step')
    reference = sc.analytic_reference(record)
    assert reference.shape == (249, 4)
    assert reference[0, 0] == record['initial_hz'] and reference[99, 0] == record['final_hz']
    assert np.all(reference[:, 1] == 1) and np.all(reference[:, 2:] == 0)
    unvoiced = sc.make_plans()['train']['records'][-1]
    assert np.all(sc.analytic_reference(unvoiced) == 0)


def test_train_pcm_reproducible_finite_and_target_suppression_keeps_identifiability():
    record = next(r for r in sc.make_plans()['train']['records'] if r['kind'] == 'stationary' and r['harmonic_amplitudes'][0] == 0)
    first, second = sc.generate_pcm(record), sc.generate_pcm(record)
    assert np.array_equal(first, second) and first.dtype == np.float32 and len(first) == 120000
    assert np.isfinite(first).all() and np.sqrt(np.mean(first ** 2)) > .003


def test_challenge_generation_guard_requires_completed_bound_checkpoint(tmp_path):
    sc.freeze(tmp_path / 'plan')
    with pytest.raises(ValueError, match='sealed'):
        sc.verify_plan(tmp_path / 'plan', 'challenge')
    checkpoint = tmp_path / 'checkpoint.json'
    checkpoint.write_text(json.dumps({'status': 'complete', 'kind': 'framewise-mlp', 'synthetic_plan_sha256': 'wrong'}))
    with pytest.raises(ValueError, match='provenance'):
        sc.verify_plan(tmp_path / 'plan', 'challenge', checkpoint)
    assert not (tmp_path / 'plan' / 'features').exists()
