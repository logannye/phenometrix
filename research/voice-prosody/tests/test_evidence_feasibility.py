"""Candidate-array, schedule and phase fixtures; no scientific PCM or acquisition."""
import copy
import json
import numpy as np
import pytest
from phenometrix_prosody import evidence_feasibility as f
from phenometrix_prosody.model import FEATURE_NAMES_V2


@pytest.fixture(autouse=True)
def no_scientific_generation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Scientific generation prohibited in this fixture suite')
    monkeypatch.setattr(f.cases, 'generate_training', forbidden)
    monkeypatch.setattr(f.cases.previous, '_synthesize_pcm', forbidden)
    monkeypatch.setattr(f.expansion, 'CoveragePeakWorker', forbidden)


def toy_record(old_hz=100., added_hz=300.):
    n = 196; x = np.zeros((n, 57), np.float32); hz = np.full((n, 6), np.nan); mask = np.zeros((n, 6), bool)
    x[:, :8] = [1, np.log2(old_hz/50), 0, 0, 0, .9, 0, .9]
    hz[:, 0] = old_hz; mask[:, 0] = True; x[:, 55] = np.float32(1/6); x[:, 56] = 1
    original = dict(features=x, candidate_hz=hz, candidate_mask=mask, reference_sample=np.arange(2208, 96001, 480),
        feature_names=np.array(FEATURE_NAMES_V2), sample_rate_hz=np.array(48000), pcm_sample_count=np.array(96000))
    evidence = dict(pcm_sha256='a'*64, pcm_sample_count=96000, ends=original['reference_sample'].copy(), rss_bytes=1)
    for key, value, score in [('old', old_hz, .9), ('new', added_hz, .7 if added_hz != old_hz else .9)]:
        evidence[key+'_hz'] = np.full(n, value, np.float64)
        evidence[key+'_confidence'] = np.full(n, score, np.float64)
        evidence[key+'_log2_hz_over_50'] = np.full(n, np.log2(value/50), np.float64)
    expanded, provenance = f.expansion.additive_from_evidence(original, evidence)
    record = copy.deepcopy(f.cases.training_spec()['records'][0]); record['frequency_hz'] = 300.
    return original, expanded, provenance, record


def support_table():
    cases = [8*i+4*j+2*k for i in range(6) for j in range(2) for k in range(2)]
    ids = np.repeat(np.array(cases, np.int64), 48); first = np.tile(np.arange(48) < 24, len(cases)); n = len(ids)
    table = {'case_index': ids, 'stratum_added_only': first, 'stratum_counterexample': ~first,
        'stratum_broad': np.ones(n, bool), 'common_eligible': np.ones(n, bool),
        'expanded_group_covered': np.ones(n, bool), 'region': np.where(first, 'before', 'after')}
    return table


def test_full_frame_classification_preserves_truth_time_and_both_pools():
    original, expanded, provenance, record = toy_record(); before = copy.deepcopy(original)
    table = f.classify_record(original, expanded, provenance, record, 0)
    assert table['original_raw_features'].shape == (196, 57) and table['expanded_raw_features'].shape == (196, 65)
    assert table['original_group_hz'].shape == (196, 6) and table['expanded_group_hz'].shape == (196, 7)
    assert table['common_eligible'].sum() == table['stratum_added_only'].sum() == 191
    assert not table['stratum_counterexample'].any()
    assert table['original_raw_covered'].sum() == 0 and table['expanded_group_covered'].sum() == 196
    np.testing.assert_array_equal(table['reference_sample'], original['reference_sample']-960)
    assert (table['truth_hz'] == 300).all()
    assert [int((table['common_eligible'] & (table['region'] == r)).sum()) for r in
        ('before-observation-change', 'observation-boundary', 'after-observation-change')] == [75, 4, 112]
    for key in original: np.testing.assert_array_equal(original[key], before[key])


def test_counterexample_requires_distinct_wrong_added_peak():
    old, new, p, record = toy_record(300., 600.)
    table = f.classify_record(old, new, p, record, 1)
    assert table['stratum_counterexample'].sum() == 191 and not table['stratum_added_only'].any()
    old, new, p, record = toy_record(300., 300.)
    table = f.classify_record(old, new, p, record, 1)
    assert not table['new_frequency_added'].any() and not table['stratum_counterexample'].any()


@pytest.mark.parametrize('change', ['timing', 'policy', 'eviction', 'distinct', 'extent'])
def test_malformed_classification_refuses(change):
    old, new, p, record = toy_record()
    if change == 'timing': new['reference_sample'][0] += 1
    elif change == 'policy': p['candidate_policy'] = np.array('replacement')
    elif change == 'eviction': p['old_frequency_evicted'][0] = True
    elif change == 'distinct': p['new_frequency_added'][0] = False
    else: old['pcm_sample_count'] = np.array(95000)
    with pytest.raises(ValueError): f.classify_record(old, new, p, record, 0)


def test_summary_uses_unconditional_common_denominator_and_no_support_fallback():
    old, new, p, record = toy_record(); table = f.classify_record(old, new, p, record, 0)
    report = f.summarize_table(table, [record])
    assert report['acquisition_rows'] == 196 and report['coverage']['frames'] == 191
    assert report['coverage']['original_raw_covered'] == 0 and report['coverage']['expanded_group_covered'] == 191
    assert report['per_case'][0]['regions']['after-observation-change']['frames'] == 112
    assert not report['feasibility']['pass_result'] and report['schedule_exposure'] == {}
    with pytest.raises(ValueError, match='Insufficient'): f.make_schedules(table, report)


def test_all_declared_schedules_use_full_indices_exact_quotas_and_overlap_accounting():
    table = support_table(); strata = {k: table['stratum_'+k] for k in f.cases.BATCH_COUNTS}
    report = {'feasibility': f.cases.feasibility(strata, table['case_index'])}
    schedules, exposure = f.make_schedules(table, report)
    assert len(schedules) == 72 and len(exposure) == 36
    for seed in f.cases.SEEDS:
        for epoch in range(12):
            key = f'seed{seed}__epoch{epoch}'; ix = schedules[key+'__indices']; origin = schedules[key+'__origin']; row = exposure[key]
            assert ix.shape == origin.shape == (33, 512)
            assert row['selected_rows'] == row['expanded_group_covered_rows'] == 16896
            assert row['expanded_group_uncovered_zero_rows'] == 0
            assert row['requested_strata'] == {'added_only': 4224, 'counterexample': 4224, 'broad': 8448}
            assert row['actual_strata']['broad'] == 16896 and row['duplicate_draws'] > 0
            assert sum(v['selected_rows'] for v in row['per_region'].values()) == 16896
    exact = f.cases.targeted_schedule(strata, table['case_index'], f.cases.SEEDS[0], 0)
    np.testing.assert_array_equal(schedules[f'seed{f.cases.SEEDS[0]}__epoch0__indices'], exact['indices'])


def test_false_claim_of_support_cannot_create_schedule():
    table = support_table(); table['stratum_added_only'][:] = False
    with pytest.raises(ValueError, match='Insufficient'): f.make_schedules(table, {'feasibility': {'pass_result': True}})


def test_complete_closure_includes_preparation_acquisition_and_grouping():
    hashes = f.source_hashes()
    for name in ('evidence_feasibility.py', 'evidence_cases.py', 'expanded_candidate_features.py', 'expanded_selector_model.py', 'selector_model.py'):
        assert 'research/voice-prosody/phenometrix_prosody/'+name in hashes


def test_freeze_must_bind_preparation_closure(monkeypatch):
    monkeypatch.setattr(f.cases, '_validate_freeze', lambda *a, **k: ({'source_sha256': {'one': 'a'}}, {}))
    monkeypatch.setattr(f, 'source_hashes', lambda: {'one': 'a', 'missing': 'b'})
    with pytest.raises(ValueError, match='closure'): f.validate_source_freeze('unused', 'a', generation=True)


@pytest.mark.parametrize('field,value', [('reference_content_read', True), ('model_outputs_read', True), ('fitting_performed', True),
    ('fresh_evaluation_generated', True), ('generation_calls', 2), ('deadline_utc', 'later'), ('status', 'preparing')])
def test_loader_rejects_wrong_phase_before_array_access(tmp_path, monkeypatch, field, value):
    monkeypatch.setattr(f, 'REPO', tmp_path); monkeypatch.setattr(f, 'RUNS', tmp_path/'runs')
    output = tmp_path/'runs'/'prep'; output.mkdir(parents=True); source = tmp_path/'freeze.json'; source.write_text('{}')
    monkeypatch.setattr(f, 'source_hashes', lambda: {})
    monkeypatch.setattr(f, 'validate_source_freeze', lambda *a, **k: ({'deadline_utc': '2026-10-23T12:18:58Z'}, {}))
    manifest = {'schema': f.SCHEMA, 'status': 'complete', 'source_freeze_path': 'freeze.json', 'source_freeze_sha256': 'a',
        'source_sha256': {}, 'training_spec_sha256': f.cases.training_spec_sha256(), 'challenge_spec_sha256': f.cases.challenge_spec_sha256(),
        'deadline_utc': '2026-10-23T12:18:58Z', 'reference_content_read': False, 'model_outputs_read': False,
        'generation_calls': 1, 'fitting_performed': False, 'fresh_evaluation_generated': False}
    manifest[field] = value; (output/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='source-bound'): f.load_preparation(output, source, 'a')


def test_prepare_rejects_bad_freeze_before_output_or_generation(tmp_path, monkeypatch):
    def reject(*args, **kwargs): raise ValueError('missing freeze closure')
    monkeypatch.setattr(f, 'validate_source_freeze', reject)
    with pytest.raises(ValueError, match='closure'): f.prepare(tmp_path/'no-output', tmp_path/'freeze', 'x')
    assert not (tmp_path/'no-output').exists()


@pytest.mark.parametrize('resource', ['time', 'rss', 'disk', 'node'])
def test_resource_ceiling_refuses(tmp_path, monkeypatch, resource):
    monkeypatch.setattr(f.cases, '_remaining', lambda frozen: 100.)
    monkeypatch.setattr(f.time, 'monotonic', lambda: 0.)
    monkeypatch.setattr(f, '_rss_bytes', lambda: f.MAX_PYTHON_RSS+1 if resource == 'rss' else 1)
    if resource == 'disk':
        monkeypatch.setattr(f, 'MAX_BYTES', 1); (tmp_path/'dummy').write_bytes(b'ab')
    class Worker:
        peak_rss_bytes = 2*1024**3+1 if resource == 'node' else 0
        timeout_seconds = 0
    manifest = {'peak_observed_python_rss_bytes': 0, 'peak_observed_node_rss_bytes': 0}
    with pytest.raises(ValueError): f._budget(tmp_path, {}, -4000 if resource == 'time' else 0, manifest, Worker())


def test_table_schema_failure_is_explicit():
    with pytest.raises(ValueError, match='schema'): f.validate_table({'case_index': np.array([0])})
