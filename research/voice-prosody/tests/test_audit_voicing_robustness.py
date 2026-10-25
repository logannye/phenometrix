"""Adversarial fixtures for independent matched-study evidence validation."""
import copy
import random
import numpy as np
import pytest
from phenometrix_prosody import audit_voicing_robustness as a


def weights():
    rng = np.random.default_rng(32)
    result = {'feature_mean': np.zeros(57, np.float32), 'feature_scale': np.ones(57, np.float32), 'scaling_fitted': np.array(True)}
    for name, shape in [('projection', (32, 57)), ('frame_hidden', (32, 32)), ('voicing_head', (1, 32)), ('candidate_head', (6, 32))]:
        result[name + '.weight'] = rng.normal(0, .1, shape).astype(np.float32)
        result[name + '.bias'] = rng.normal(0, .1, shape[0]).astype(np.float32)
    return result


def test_numpy_reconstruction_covers_all_outputs_and_masks():
    import torch
    from phenometrix_prosody.model import ModelConfig, VoiceCandidateModel
    w = weights()
    model = VoiceCandidateModel(ModelConfig(kind='framewise-mlp'))
    model.load_state_dict({k: torch.from_numpy(v) for k, v in w.items()})
    x = np.random.default_rng(5).normal(size=(15, 57)).astype(np.float32)
    mask = np.random.default_rng(9).integers(0, 2, (15, 6)).astype(bool)
    mask[0] = False
    result = a.independent_mlp(w, x, mask)
    with torch.inference_mode():
        expected = model(torch.from_numpy(x), torch.from_numpy(mask), torch.ones(15, 32))
    for key, value in zip(('voicing_logit', 'candidate_logits', 'state'), expected):
        np.testing.assert_allclose(result[key], value.numpy(), atol=1e-6, rtol=0)
    assert np.all(result['candidate_logits'][0] == -1e9)
    assert not result['state'].any()


@pytest.mark.parametrize('name', ['feature_scale', 'scaling_fitted', 'projection.weight'])
def test_reconstruction_rejects_bad_weights(name):
    w = weights()
    if name == 'scaling_fitted':
        w[name] = np.array(False)
    else:
        w[name].flat[0] = np.nan if name.endswith('weight') else 0
    with pytest.raises(ValueError):
        a.independent_mlp(w, np.zeros((1, 57), np.float32), np.ones((1, 6), bool))


def test_hybrid_threshold_bounds_and_unavailable_are_exact():
    status, hz = a.hybrid_arrays(np.array([-.000001, 0, 1, 1, 1, 1]), [180, 50, 700, 49.999, 700.001, np.nan])
    assert status.tolist() == [0, 1, 1, 2, 2, 2]
    assert np.array_equal(hz, [np.nan, 50, 700, np.nan, np.nan, np.nan], equal_nan=True)


def test_full_output_audit_rejects_non_delivered_logit_or_pitch_corruption():
    w = weights()
    feature = {'features': np.zeros((2, 57), np.float32), 'candidate_mask': np.ones((2, 6), bool)}
    saved = a.independent_mlp(w, **{'features': feature['features'], 'candidate_mask': feature['candidate_mask']})
    saved['raw_yin_hz'] = np.array([180., np.nan])
    saved['status'], saved['hz'] = a.hybrid_arrays(saved['voicing_logit'], saved['raw_yin_hz'])
    assert a.verify_numerical_output(saved, w, feature)['state'] == 0
    for key, index in [('candidate_logits', (0, 0)), ('voicing_logit', (1, 0)), ('state', (0, 0))]:
        corrupt = copy.deepcopy(saved)
        corrupt[key][index] += .01
        with pytest.raises(ValueError, match='tolerance'):
            a.verify_numerical_output(corrupt, w, feature)
    corrupt = copy.deepcopy(saved)
    corrupt['hz'][0] = 181
    with pytest.raises(ValueError, match='pitch'):
        a.verify_numerical_output(corrupt, w, feature)


def test_counts_partition_reference_voiced_keep_unsupported_and_unavailable():
    refs = np.tile([180., 1., 0., 0.], (12, 1))
    refs[5] = [0, 0, 0, 0]
    refs[6, 0] = 900
    refs[7, 0] = 0  # invalid voiced reference
    predictions = [
        {'sample': 768 + 480 * 5, 'state': 'voiced', 'f0_hz': 180.},
        {'sample': 768 + 480 * 6, 'state': 'voiced', 'f0_hz': 450.},
        {'sample': 768 + 480 * 8, 'state': 'unvoiced', 'f0_hz': None},
        {'sample': 768 + 480 * 9, 'state': 'voiced', 'f0_hz': 180.},
        {'sample': 768 + 480 * 12, 'state': 'unvoiced', 'f0_hz': None}]
    got, _ = a.count_predictions(refs, 20000, predictions)
    assert got['eligible_valid_frames'] == 6
    assert got['reference_voiced_frames'] == 5
    assert got['reference_unvoiced_frames'] == 1
    assert got['voicing_false_positive_count'] == 1
    assert got['reference_outside_supported_bounds'] == 1
    assert got['voiced_pitch_error_count'] == 1
    assert got['voicing_false_negative_count'] == 1
    assert got['reference_voiced_success_count'] == 1
    assert got['unavailable_reference_voiced_count'] == 2
    assert got['predictions_outside_reference_extent'] == 1
    assert got['excluded_invalid_reference'] == 1


def test_off_grid_and_duplicate_predictions_fail():
    refs = np.tile([180., 1., 0., 0.], (12, 1))
    row = {'sample': 3169, 'state': 'unvoiced', 'f0_hz': None}
    with pytest.raises(ValueError, match='off-grid'):
        a.count_predictions(refs, 20000, [row])
    row['sample'] = 3168
    with pytest.raises(ValueError, match='Duplicate'):
        a.count_predictions(refs, 20000, [row, row])


def model_scores():
    names = ['hybrid_original'] + [f'{arm}-{s}' for s in a.SEEDS for arm in ('C', 'R')]
    out = {}
    for name in names:
        treatment = name.startswith('R-')
        out[name] = {'cells': {s + '/' + c: {
            'voicing_false_negative_or_unavailable_rate': .07 if treatment else .1,
            'voicing_false_positive_rate': .02, 'joint_error': .08 if treatment else .1,
            'reference_voiced_success_rate': .9 if treatment else .87,
        } for s in a.SPEAKERS for c in a.CONDITIONS}}
    analytic = {f'R-{s}': {'status': 'complete', 'pass': True} for s in a.SEEDS}
    return out, analytic


def test_all_gates_and_predetermined_artifact():
    models, analytic = model_scores()
    result = a.development_gates(models, analytic)
    assert result['pass'] and result['export_designation'] == 'R-20261025'
    assert result['effects']['mean_shifted_vs_control_mean']['absolute_percentage_points'] == pytest.approx(3)


@pytest.mark.parametrize('corruption', ['missing_cell', 'undefined', 'zero_baseline', 'missing_analytic', 'failed_analytic', 'bad_export_seed', 'cell_fp', 'clean_joint', 'm05'])
def test_gates_reject_corruption_without_best_seed_switch(corruption):
    models, analytic = model_scores()
    if corruption == 'missing_cell':
        del models['C-20261025']['cells']['F01/clean']
    elif corruption == 'undefined':
        models['R-20261025']['cells']['F01/clean']['voicing_false_positive_rate'] = None
    elif corruption == 'zero_baseline':
        for cell in models['hybrid_original']['cells'].values():
            cell['voicing_false_negative_or_unavailable_rate'] = 0
    elif corruption == 'missing_analytic':
        del analytic['R-20261025']
    elif corruption == 'failed_analytic':
        analytic['R-20261025']['pass'] = False
    elif corruption == 'bad_export_seed':
        for cell in models['R-20261025']['cells'].values():
            cell['voicing_false_negative_or_unavailable_rate'] = .09
    elif corruption == 'cell_fp':
        models['R-20261025']['cells']['F01/clean']['voicing_false_positive_rate'] = .041
    elif corruption == 'clean_joint':
        for key, cell in models['R-20261025']['cells'].items():
            if key.endswith('/clean'):
                cell['joint_error'] = .106
    else:
        models['R-20261025']['cells']['M05/first_difference']['voicing_false_negative_or_unavailable_rate'] = .121
    assert not a.development_gates(models, analytic)['pass']


def test_pooled_cell_weighting_and_undefined_denominators():
    def row(speaker, n, failures):
        counts = dict.fromkeys(a.COUNTS, 0)
        counts.update(eligible_valid_frames=n, reference_voiced_frames=n, correct_frames=n - failures,
            reference_voiced_success_count=n - failures, voicing_false_negative_count=failures,
            voicing_false_negative_or_unavailable_count=failures)
        return {'speaker': speaker, 'condition': 'clean', 'metrics': a.summarize(counts, [])}
    summary = a.aggregate_records([row('A', 1, 1), row('A', 9, 0), row('B', 2, 1)])
    assert summary['overall']['joint_error'] == pytest.approx(.3)
    assert summary['overall']['voicing_false_positive_rate'] is None


def schedule_fixture():
    slots = [{'n_frames': 260, 'valid': np.r_[np.ones(128, bool), np.zeros(132, bool)]},
             {'n_frames': 130, 'valid': np.ones(130, bool)}]
    epochs = []
    for epoch in range(12):
        order = [0, 1]
        random.Random(a.SEEDS[0] + epoch).shuffle(order)
        epochs.append({'epoch': epoch + 1, 'order': order, 'steps': [
            {'indices': order, 'start': 0, 'end': 128, 'update': True},
            {'indices': order, 'start': 128, 'end': 256, 'update': True},
            {'indices': order, 'start': 256, 'end': 260, 'update': False}]})
    return {'epochs': epochs}, slots


def test_schedule_derives_actual_update_count_and_skip_rule():
    schedule, slots = schedule_fixture()
    assert a.validate_schedule(schedule, slots, a.SEEDS[0]) == {'expected_updates': 24, 'epoch_updates': [2] * 12}
    schedule['epochs'][0]['steps'][-1]['update'] = True
    with pytest.raises(ValueError, match='skip rule'):
        a.validate_schedule(schedule, slots, a.SEEDS[0])


def test_fit_pair_refuses_scaling_initialization_and_update_drift():
    w = weights()
    common = {'status': 'complete', 'split': 'train', 'test_or_development_labels_used': False,
        'epochs': 12, 'batch_size': 64, 'tbptt_frames': 128, 'learning_rate': .001, 'gradient_clip_norm': 1.,
        'measurement_delay_ms': 20, 'kind': 'framewise-mlp', 'expected_updates': 24, 'actual_updates': 24,
        'epoch_updates': [2] * 12, 'seed': a.SEEDS[0], 'plan_sha256': 'plan', 'prepare_manifest_sha256': 'prepared',
        'schedule_sha256': 'schedule', 'initialization_sha256': 'init'}
    assert a.validate_fit_pair(common, common, w, w, w, 24)['pass']
    for field, value in [('actual_updates', 396), ('initialization_sha256', 'other')]:
        corrupt = dict(common, **{field: value})
        with pytest.raises(ValueError):
            a.validate_fit_pair(common, corrupt, w, w, w, 24)
    altered = copy.deepcopy(w)
    altered['feature_mean'][0] += 1
    with pytest.raises(ValueError, match='scaling'):
        a.validate_fit_pair(common, common, w, altered, w, 24)


def test_bindings_reject_wrong_hash_and_escape(tmp_path):
    path = tmp_path / 'file'; path.write_bytes(b'actual')
    audit = a.Bindings(tmp_path)
    with pytest.raises(ValueError, match='digest'):
        audit.bind(path, '0' * 64)
    with pytest.raises(ValueError, match='escaped'):
        audit.bind(tmp_path / '..' / 'file')


def test_independent_augmentation_reconstructs_exact_pcm_without_producer_calls():
    from phenometrix_prosody.voicing_robustness_inputs import draw, construct
    pcm = np.random.default_rng(8).normal(0, .15, 4800).astype(np.float32)
    for source in ('a', 'b', 'F02_si645'):
        for slot in range(4):
            params = a.independent_bank_parameters(source, slot)
            assert params == draw(source, slot)
            actual, receipt = construct(pcm, params)
            expected, independent_receipt = a.independent_construct(pcm, params)
            np.testing.assert_array_equal(actual, expected)
            a.compare_values(receipt, independent_receipt, 'receipt')


def test_independent_labels_reconstruct_boundary_uncovered_and_invalid():
    n = 10
    features = {'features': np.zeros((n, 57), np.float32), 'candidate_hz': np.full((n, 6), np.nan),
        'candidate_mask': np.zeros((n, 6), bool), 'reference_sample': np.arange(2208, 2208 + n * 480, 480),
        'pcm_sample_count': np.array(2208 + (n - 1) * 480), 'sample_rate_hz': np.array(48000)}
    features['candidate_hz'][:, 0] = 180.; features['candidate_mask'][:, 0] = True
    refs = np.tile([180., 1., 0., 0.], (15, 1)); refs[6] = [0., 0., 0., 0.]; refs[7, 0] = 900.; refs[8, 0] = np.nan
    labels = a.independent_labels(features, refs)
    assert labels['valid'].sum() == 3
    assert labels['target'][4] == 0
    assert labels['voiced'][5] == 0 and labels['target'][5] == -1
    assert labels['voiced'][6] == 1 and labels['target'][6] == -1
    assert not labels['valid'][7]


def event_output():
    target = np.arange(1248, 143041, 480)
    voiced = (target >= 48000) & (target < 96000)
    return target, voiced.astype(np.uint8), np.where(voiced, 180., np.nan)


def test_events_keep_signed_target_and_availability_separate():
    sample, state, hz = event_output()
    result = a.voicing_event_metrics(sample, state, hz, 180.)
    assert result['pass']
    assert result['events']['onset']['signed_target_delay_ms'] == 6
    assert result['events']['onset']['earliest_availability_delay_ms'] == 26
    state[(sample >= 96000) & (sample <= 106560)] = 2
    assert not a.voicing_event_metrics(sample, state, hz, 180.)['events']['offset']['detected']


def test_event_plateaux_are_individually_gated_and_missing_not_dropped():
    sample, state, hz = event_output()
    active = np.flatnonzero((sample >= 9600) & (sample <= 38400))[:4]
    state[active] = 1; hz[active] = 180.
    result = a.voicing_event_metrics(sample, state, hz, 180.)
    assert result['plateaux']['pre']['false_voiced_rate'] > .05
    assert result['plateaux']['post']['false_voiced_rate'] == 0
    assert not result['pass']
    cases = {str(i): result for i in range(12)}
    assert not a.voicing_event_gate(cases, cases)['pass']


def test_checkpoint_tensor_digest_and_seed_initialization():
    import torch
    from phenometrix_prosody.model import ModelConfig, VoiceCandidateModel
    original = weights()
    torch.manual_seed(a.SEEDS[0]); model = VoiceCandidateModel(ModelConfig(kind='framewise-mlp'))
    expected = {k: v.detach().numpy().copy() for k, v in model.state_dict().items()}
    for key in ('feature_mean', 'feature_scale', 'scaling_fitted'):
        expected[key] = original[key]
    assert a.tensor_digest(expected) == a.tensor_digest(a.independent_initialization(a.SEEDS[0], original))
    assert a.tensor_digest(expected) != a.tensor_digest(a.independent_initialization(a.SEEDS[1], original))


def test_analytic_gate_requires_every_transition_and_challenge_case():
    originals = {f'c{i}': {'kind': 'constant', 'success': .95} for i in range(3)}
    originals.update({f'o{i}': {'kind': 'observation-change', 'success': .95} for i in range(3)})
    originals.update({f's{i}': {'kind': 'step', 'delay_ms': 6.} for i in range(2)})
    originals.update({f'g{i}': {'kind': 'glide', 'delay_ms': 0.} for i in range(6)})
    cases = []
    for cohort in ('previous32-regression', 'fresh32'):
        cases.extend({'cohort': cohort, 'metrics': {'reference_voiced_frames': 100, 'reference_unvoiced_frames': 0, 'reference_voiced_success_rate': .95}} for _ in range(28))
        cases.extend({'cohort': cohort, 'metrics': {'reference_voiced_frames': 0, 'reference_unvoiced_frames': 100, 'voicing_false_positive_rate': .05}} for _ in range(4))
    assert a.independent_analytic_gate(cases, originals, originals)['pass']
    corrupted = copy.deepcopy(originals); corrupted['s0']['delay_ms'] = None
    assert not a.independent_analytic_gate(cases, corrupted, originals)['pass']
    assert not a.independent_analytic_gate(cases[:-1], originals, originals)['pass']
