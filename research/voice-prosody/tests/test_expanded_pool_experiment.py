"""Fixture-only runner contract tests; no study PCM, reference reads or fits."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from phenometrix_prosody import expanded_pool_experiment as e


@pytest.fixture(autouse=True)
def forbid_scientific_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Scientific generation, data access and fitting are forbidden here')
    for module, names in [
        (e.cases, ('generate_challenge',)),
        (e.cases.previous, ('_synthesize_pcm', '_extract_feature')),
        (e.old, ('load_inputs', 'load_labels', 'infer')),
        (e.acquisition, ('expanded_feature',)),
        (torch.optim, ('Adam',)),
    ]:
        for name in names:
            monkeypatch.setattr(module, name, forbidden)


def natural():
    rows = [('original_mlp', .10, .70), ('original_hybrid', .11, .71)]
    rows += [(e.key(arm, seed), .094, .72)
             for arm in (*e.ARMS, 'frozen-original6', 'frozen-expanded7') for seed in e.SEEDS]
    return {name: {'macro': {'joint_error': error, 'reference_voiced_success_rate': success},
                   'per_condition': {c: {'joint_error': error} for c in e.old.CONDITIONS}}
            for name, error, success in rows}


def analytic():
    names = [e.key('expanded7', seed) for seed in e.SEEDS]
    def row(kind):
        return {'kind': kind, 'methods': {name: {
            'all': {'success_within_50c_all_frames': 1.},
            'regions': {'after-observation-change': {'success_within_50c_all_frames': 1.}}}
            for name in names}}
    result = {'regression': {name: row('constant' if name.startswith('constant-')
        else 'observation-change' if name.startswith('observation-change-') else 'glide')
        for name in e.old.SYNTHETIC_IDS}}
    for group, prefix in [('consumed_harmonic', 'harmonic-balance'),
            ('consumed_subharmonic', 'subharmonic-balance'),
            ('consumed_augmentation', 'augmentation-balance'),
            ('consumed_coverage', 'coverage-balance'), ('challenge', 'expanded-balance')]:
        result[group] = {f'{prefix}-f{i}-p{j}': row('observation-change')
                         for i in range(6) for j in range(2)}
    return result


def batch(frequencies, capacity=6, evidence=None):
    hz = np.full((len(frequencies), capacity), np.nan)
    mask = np.zeros_like(hz, bool)
    x = np.zeros((len(frequencies), capacity * 8 + 9), np.float32)
    for i, values in enumerate(frequencies):
        for j, value in enumerate(values):
            score = .8 if evidence is None else evidence[i][j]
            hz[i, j] = value
            mask[i, j] = True
            x[i, j*8:(j+1)*8] = [1, np.log2(value/50), score, 0, 0, 0, 0, score]
        x[i, capacity*8+7] = len(values)/6
    return e.prepare({'features': x, 'candidate_hz': hz, 'candidate_mask': mask})


def test_exact_six_fit_inventory_and_predesignated_arm():
    assert e.ARMS == ('original6', 'expanded7')
    assert e.SEEDS == (20261022, 20261023, 20261024)
    assert e.CANDIDATE == 'expanded7'
    assert {e.key(a, s) for a in e.ARMS for s in e.SEEDS} == e.cases._expected_checkpoints()


def test_every_seed_retains_all_66_individual_gates():
    result = e.analytic_decision(analytic())
    counts = {'stationary': 3, 'harmonic_change': 3, **{g: 12 for g in list(e.GROUPS)[1:]}}
    assert sum(e.GROUPS.values()) == 74
    for seed in e.SEEDS:
        name = e.key('expanded7', seed)
        assert {g: len(result[g+'_by_seed_case'][name]) for g in counts} == counts
        assert sum(len(result[g+'_by_seed_case'][name]) for g in counts) == 66
    assert all(result[g+'_pass'] for g in counts)


@pytest.mark.parametrize('group', list(e.GROUPS)[1:])
@pytest.mark.parametrize('seed', e.SEEDS)
def test_one_failing_old_or_fresh_case_is_never_averaged_away(group, seed):
    rows = analytic()
    case = next(iter(rows[group]))
    rows[group][case]['methods'][e.key('expanded7', seed)]['regions']['after-observation-change']['success_within_50c_all_frames'] = .949
    result = e.analytic_decision(rows)
    assert not result[group+'_pass']
    assert not result[group+'_by_seed_case'][e.key('expanded7', seed)][case]


def test_exact_95_percent_passes_and_next_lower_float_fails():
    rows = analytic()
    cell = rows['challenge']['expanded-balance-f0-p0']['methods'][e.key('expanded7', e.SEEDS[0])]['regions']['after-observation-change']
    cell['success_within_50c_all_frames'] = .95
    assert e.analytic_decision(rows)['challenge_pass']
    cell['success_within_50c_all_frames'] = np.nextafter(.95, -np.inf)
    assert not e.analytic_decision(rows)['challenge_pass']


@pytest.mark.parametrize('case,section,gate', [
    ('constant-300', 'all', 'stationary'),
    ('observation-change-300', 'regions', 'harmonic_change'),
])
def test_original_constant_whole_and_observation_post_gates_are_preserved(case, section, gate):
    rows = analytic()
    cell = rows['regression'][case]['methods'][e.key('expanded7', e.SEEDS[-1])]
    cell = cell['all'] if section == 'all' else cell['regions']['after-observation-change']
    cell['success_within_50c_all_frames'] = .94
    assert not e.analytic_decision(rows)[gate+'_pass']


@pytest.mark.parametrize('group', list(e.GROUPS))
def test_equal_count_identity_substitution_is_rejected(group):
    rows = analytic()
    rows[group]['unplanned-case'] = rows[group].pop(next(iter(rows[group])))
    with pytest.raises(ValueError, match='identity'):
        e.analytic_decision(rows)


def test_old_failure_cannot_be_removed():
    rows = analytic()
    rows['consumed_augmentation'].pop('augmentation-balance-f5-p1')
    with pytest.raises(ValueError, match='inventory'):
        e.analytic_decision(rows)


def test_regression_kind_cannot_transfer_gate_between_whole_and_post():
    rows = analytic()
    rows['regression']['constant-300']['kind'] = 'observation-change'
    with pytest.raises(ValueError, match='gate'):
        e.analytic_decision(rows)


def test_natural_gates_use_new_expanded_arm_not_better_frozen_control():
    rows = natural()
    for seed in e.SEEDS:
        rows[e.key('expanded7', seed)]['macro']['joint_error'] = .098
        rows[e.key('frozen-expanded7', seed)]['macro']['joint_error'] = .01
    result = e.natural_decision(rows)
    assert result['predesignated_candidate'] == 'expanded7'
    assert not result['pass']
    assert not result['checks']['mean_improves_both_by_half_point']


def test_natural_half_point_boundary_is_not_relaxed():
    rows = natural()
    rows['original_mlp']['macro']['joint_error'] = .005
    rows['original_hybrid']['macro']['joint_error'] = .006
    for seed in e.SEEDS:
        rows[e.key('expanded7', seed)]['macro']['joint_error'] = 0.
    assert e.natural_decision(rows)['pass']
    rows[e.key('expanded7', e.SEEDS[0])]['macro']['joint_error'] = 1e-8
    assert not e.natural_decision(rows)['checks']['mean_improves_both_by_half_point']


@pytest.mark.parametrize('field', ['seed_tie', 'condition', 'voiced'])
def test_every_natural_constraint_remains_mandatory(field):
    rows = natural()
    if field == 'seed_tie':
        rows[e.key('expanded7', e.SEEDS[0])]['macro']['joint_error'] = .1
    else:
        for seed in e.SEEDS:
            if field == 'condition':
                rows[e.key('expanded7', seed)]['per_condition']['clip01']['joint_error'] = .110001
            else:
                rows[e.key('expanded7', seed)]['macro']['reference_voiced_success_rate'] = .699999
    assert not e.natural_decision(rows)['pass']


def test_paired_control_drift_is_descriptive_not_a_substitute_gate():
    rows = natural()
    for seed in e.SEEDS:
        rows[e.key('original6', seed)]['macro']['joint_error'] = .096
        rows[e.key('frozen-original6', seed)]['macro']['joint_error'] = .09
    result = e.natural_decision(rows)
    assert result['pass']
    assert result['original_control_drift_from_historical'] == pytest.approx(.006)
    assert result['improvement_over_paired_original'] == pytest.approx(.002)


def test_raw_coverage_retention_raises_on_eviction():
    data = {'records': [{'start': 0, 'stop': 1, 'speaker': 'toy', 'condition': 'clean'}]}
    labels = {'valid': np.ones(1, bool), 'voiced': np.ones(1, bool), 'hz': np.array([100.])}
    pools = {'original6': batch([[100.]]), 'expanded7': batch([[200.]], 7)}
    with pytest.raises(ValueError, match='superset lost'):
        e.coverage_reports(data, labels, pools)


def test_raw_retention_does_not_hide_grouped_representative_loss():
    data = {'records': [{'start': 0, 'stop': 1, 'speaker': 'toy', 'condition': 'clean'}]}
    labels = {'valid': np.ones(1, bool), 'voiced': np.ones(1, bool), 'hz': np.array([97.2])}
    pools = {'original6': batch([[100.]]), 'expanded7': batch([[100., 100.2]], 7, [[.8, .9]])}
    result = e.coverage_reports(data, labels, pools)
    assert result['policy_change']['cells']['toy/clean']['fixed-slots']['lost'] == 0
    assert result['expanded7']['lost_fraction'] == 1.
    assert not result['expanded7']['pass']


def test_regional_diagnostics_preserve_fixed_voice_and_unavailable_states():
    pools = {'original6': batch([[100.], [100.], []]), 'expanded7': batch([[100.], [100.], []], 7)}
    item = {'scenario': {'kind': 'constant', 'initialHz': 100},
        'timing': SimpleNamespace(reference_sample=np.array([3168, 3648, 4128]),
            availability_sample=np.array([4128, 4608, 5088]), common_eligible=np.ones(3, bool)),
        'prepared': pools, 'probability': np.array([.499, .5, .9]),
        'predictions': {'toy': (np.array([0, 1, 2]), np.array([np.nan, 100., np.nan]))}}
    result = e.analytic_coverage(item)['all']
    assert result['frames'] == 3 and result['original_voiced_gate_positive'] == 2
    assert result['methods']['toy'] == {'unvoiced': 1, 'voiced': 1, 'unavailable': 1}
    assert set(result['policies']) == set(e.ARMS)


def checkpoints(tmp_path):
    folder = tmp_path/'checkpoints'
    folder.mkdir()
    fits = {}
    for arm in e.ARMS:
        for seed in e.SEEDS:
            name = e.key(arm, seed)
            path = folder/(name+'.pt')
            path.write_text('Unrelated fixture bytes: '+name)
            fits[name] = {'weights_sha256': e.ev.sha(path)}
    return {'fits': fits}


def test_exact_six_final_weights_required(tmp_path):
    manifest = checkpoints(tmp_path)
    e.check_weights(tmp_path, manifest)
    manifest['fits'].pop(next(iter(manifest['fits'])))
    with pytest.raises(ValueError, match='six-fit inventory'):
        e.check_weights(tmp_path, manifest)


def test_extra_fit_cannot_enter_freeze(tmp_path):
    manifest = checkpoints(tmp_path)
    manifest['fits']['unplanned'] = next(iter(manifest['fits'].values()))
    with pytest.raises(ValueError, match='six-fit inventory'):
        e.check_weights(tmp_path, manifest)


def test_changed_final_weight_rejected(tmp_path):
    manifest = checkpoints(tmp_path)
    (tmp_path/'checkpoints'/(next(iter(manifest['fits']))+'.pt')).write_text('changed')
    with pytest.raises(ValueError, match='checkpoint changed'):
        e.check_weights(tmp_path, manifest)


def test_source_change_rejected(monkeypatch):
    monkeypatch.setattr(e, 'sources', lambda: {'toy.py': 'new'})
    with pytest.raises(ValueError, match='source/plan changed'):
        e.check_sources({'toy.py': 'old'})


def test_output_reuse_and_escape_rejected_before_device_or_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(e, 'RUNS', tmp_path/'runs')
    def unexpected():
        pytest.fail('Device should not be checked for invalid output')
    monkeypatch.setattr(torch.backends.mps, 'is_available', unexpected)
    (tmp_path/'runs'/'old').mkdir(parents=True)
    for path in (tmp_path/'outside', tmp_path/'runs'/'old'):
        with pytest.raises(ValueError, match='Fresh ignored'):
            e.run(path)


def test_unavailable_mps_refuses_before_creating_output(tmp_path, monkeypatch):
    monkeypatch.setattr(e, 'RUNS', tmp_path)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='Local MPS required'):
        e.run(tmp_path/'new')
    assert not (tmp_path/'new').exists()


def test_low_disk_refuses_before_reading_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(e, 'RUNS', tmp_path)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    monkeypatch.setattr(e.shutil, 'disk_usage', lambda _: SimpleNamespace(free=8*1024**3-1))
    with pytest.raises(RuntimeError, match='8 GiB'):
        e.run(tmp_path/'new')
    assert not (tmp_path/'new').exists()


def test_budget_bounds_worker_timeout_by_remaining_deadline(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(e.cov, 'budget', lambda *args: calls.append(args))
    monkeypatch.setattr(e.time, 'monotonic', lambda: 4499.75)
    worker = SimpleNamespace(timeout_seconds=120)
    e.budget(tmp_path, 0, {}, worker)
    assert worker.timeout_seconds == .25 and len(calls) == 1


def test_budget_propagates_resource_failure_before_worker_request(tmp_path, monkeypatch):
    def fail(*args):
        raise MemoryError('fixture resource ceiling')
    monkeypatch.setattr(e.cov, 'budget', fail)
    worker = SimpleNamespace(timeout_seconds=120)
    with pytest.raises(MemoryError, match='resource ceiling'):
        e.budget(tmp_path, 0, {}, worker)
    assert worker.timeout_seconds == 120


def test_training_cache_uses_actual_records_by_arm_schema(tmp_path, monkeypatch):
    # Metadata only: stopping at budget occurs after the schema/inventory check,
    # before reading any feature or PCM file and before the worker is invoked.
    prior = tmp_path/'prior'
    (prior/'training').mkdir(parents=True)
    spec = e.training_cases.training_spec()
    (prior/'training'/'spec.json').write_text(json.dumps(spec))
    cache = {'status': 'complete', 'records': [
        {'id': row['id'], 'arm': arm}
        for arm, rows in spec['records_by_arm'].items() for row in rows]}
    (prior/'training'/'manifest.json').write_text(json.dumps(cache))
    monkeypatch.setattr(e, 'PRIOR', prior)
    monkeypatch.setattr(e, 'bind', lambda m, path, *args: path)
    class ReachedFirstCase(Exception):
        pass
    def stop(*args):
        raise ReachedFirstCase
    monkeypatch.setattr(e, 'budget', stop)
    output = tmp_path/'new'
    output.mkdir()
    with pytest.raises(ReachedFirstCase):
        e.assemble_training(output, {}, object(), 0)
    assert list((output/'training'/'features').iterdir()) == []


def coverage_targets():
    # Pure Boolean fixture: no feature/reference files or study targets loaded.
    original = torch.zeros(132470, dtype=torch.bool)
    original[2:132183] = True
    expanded = original.clone()
    expanded[132183:132468] = True
    return {'original6': SimpleNamespace(covered=original),
            'expanded7': SimpleNamespace(covered=expanded)}


def test_union_keeps_newly_coverable_rows_in_same_sorted_schedule():
    targets = coverage_targets()
    schedule = e.training_schedule(targets)
    assert torch.equal(schedule, torch.arange(2, 132468))
    assert len(schedule) == 132466
    assert len(torch.split(schedule, 4096)) == 33
    assert len(torch.split(schedule, 4096)[-1]) == 1394
    assert int(targets['original6'].covered[schedule].sum()) == 132181


@pytest.mark.parametrize('change', ['lose_original', 'wrong_shape', 'wrong_union', 'same_union_wrong_original'])
def test_union_identity_changes_stop_before_fit(change):
    targets = coverage_targets()
    if change == 'lose_original':
        targets['expanded7'].covered[2] = False
    elif change == 'wrong_shape':
        targets['expanded7'].covered = targets['expanded7'].covered[:-1]
    elif change == 'wrong_union':
        targets['expanded7'].covered[132468] = True
    else:
        # Retaining the union total cannot conceal a changed original/new split.
        targets['original6'].covered[132183] = True
    with pytest.raises(ValueError, match='coverage|schedule/count'):
        e.training_schedule(targets)
