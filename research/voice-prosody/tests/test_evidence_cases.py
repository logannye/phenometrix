"""Metadata, toy candidate tables and refusal checks; no scientific PCM/features."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from phenometrix_prosody import evidence_cases as e


class FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None): return cls(2026, 10, 23, 5, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def no_scientific_generation(monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError('Scientific PCM/feature generation prohibited in fixture suite')
    monkeypatch.setattr(e.previous, '_synthesize_pcm', forbidden)
    monkeypatch.setattr(e, '_extract_feature', forbidden)
    monkeypatch.setattr(e, 'datetime', FixedDatetime)


def fake_freezes(tmp_path, monkeypatch):
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(e, 'REPO', tmp_path); monkeypatch.setattr(e, 'RUNS', tmp_path/'runs')
    source = tmp_path/'fixture.py'; source.write_text('# fixture source\n')
    closure = {'fixture.py': e.previous._sha(source)}
    monkeypatch.setattr(e, 'source_hashes', lambda: closure.copy())
    run = tmp_path/'runs'/'test-study'; (run/'checkpoints').mkdir(parents=True)
    common = {'arms': list(e.ARMS), 'seeds': list(e.SEEDS), 'maximum_fits': 12,
        'training_spec_sha256': e.training_spec_sha256(), 'challenge_spec_sha256': e.challenge_spec_sha256(),
        'source_sha256': closure, 'deadline_utc': '2026-10-23T12:18:58Z'}
    initial = {**common, 'schema': 'phenometrix-evidence-source-freeze.v1', 'status': 'sources_frozen', 'fits': {}}
    sp = run/'source-freeze.json'; sp.write_text(json.dumps(initial))
    weights = {}
    for name in sorted(e._expected_checkpoints()):
        p = run/'checkpoints'/(name+'.pt'); p.write_bytes(('test-only '+name).encode())
        weights[name] = {'path': str(p.relative_to(tmp_path)), 'sha256': e.previous._sha(p)}
    frozen = {**common, 'schema': 'phenometrix-evidence-checkpoint-freeze.v1', 'status': 'checkpoints_frozen',
        'source_freeze_path': str(sp.relative_to(tmp_path)), 'source_freeze_sha256': e.previous._sha(sp), 'checkpoints': weights}
    cp = run/'checkpoint-freeze.json'; cp.write_text(json.dumps(frozen))
    return sp, initial, cp, frozen


def toy_support():
    # Six carriers x two phases x two gains:24 distinct configurations;48 toy
    # observations each, with disjoint24-row target strata. No waveform is made.
    cases = [8*i+4*j+2*k for i in range(6) for j in range(2) for k in range(2)]
    ids = np.repeat(np.array(cases, np.int64), 48)
    first = np.tile(np.arange(48) < 24, len(cases))
    return {'added_only': first, 'counterexample': ~first, 'broad': np.ones(len(ids), bool)}, ids


def test_fixed512_metadata_and_no_prior_carrier_collision():
    spec = e.training_spec(); rows = spec['records']
    assert len(rows) == len({r['id'] for r in rows}) == 512
    assert len({r['frequency_hz'] for r in rows}) == 64
    assert min(e.TRAIN_FREQUENCIES) == 53.113 and max(e.TRAIN_FREQUENCIES) == 690.988
    assert {tuple(r['phase_radians']) for r in rows} == set(e.TRAIN_PHASES)
    assert {r['gain'] for r in rows} == {.035, .075}
    assert {tuple(r['after']) for r in rows} == {(0., 1., .5), (.2, 1., .5)}
    assert all(r['before'] == [1., .5, .25] and r['sample_count'] == 96000 and r['change_sample'] == 39840 for r in rows)
    assert rows[0]['id'] == 'evidence-train-f0-p0-g0-m0' and rows[-1]['id'] == 'evidence-train-f63-p1-g1-m1'
    assert not set(e.TRAIN_FREQUENCIES) & set(e.historical_carriers())
    assert not set(e.FREQUENCIES) & (set(e.historical_carriers()) | set(e.TRAIN_FREQUENCIES))
    assert {347.9, 601.7, 517.3, 641.9, 633.1} <= set(e.historical_carriers())
    assert spec['common_eligible_rows'] == 97792 and spec['batch_counts'] == {'added_only': 128, 'counterexample': 128, 'broad': 256}
    assert e.training_spec_sha256() == hashlib.sha256(e.previous._canonical(spec)).hexdigest()


def test_exact_fresh_metadata_and_all_retained_gates():
    spec = e.challenge_spec(); rows = spec['records']
    assert len(rows) == 12 and {r['frequency_hz'] for r in rows} == {109.9, 241.1, 379.3, 563.9, 619.7, 677.9}
    assert {tuple(r['phase_radians']) for r in rows} == {(.79, -1.37, .23), (2.17, .61, -.89)}
    assert all(r['gain'] == .057 and r['after'] == [0., 1., .5] for r in rows)
    assert spec['mandatory_case_gates_per_seed'] == 78 and spec['retained_case_count'] == 74 and spec['total_case_count'] == 86
    assert spec['predesignated_candidate'] == 'targeted__on' and 'no frame drops' in spec['prospective_gate']
    assert 'not disjoint support' in spec['novelty_boundary']
    assert e.challenge_spec_sha256() == hashlib.sha256(e.previous._canonical(spec)).hexdigest()


def test_exact_grid_regions_without_pcm():
    r = e.challenge_spec()['records'][0]; s = e.previous._scenario(r)
    ends = np.arange(2208, 96001, 480); targets = ends-960
    selected = ends[(targets >= 3072) & (targets+1536 <= 96000)]
    regions = [e.challenge_region_at(s, int(x)) for x in selected]
    assert len(ends) == 196 and len(selected) == 191
    assert [regions.count(x) for x in ('before-observation-change', 'observation-boundary', 'after-observation-change')] == [75, 4, 112]


def test_strata_use_only_available_correct_added_candidate_and_true_counterexample():
    original = np.array([[100., np.nan], [300., np.nan], [300., np.nan], [100., np.nan], [100., np.nan], [300., np.nan]])
    expanded = np.array([[100., 300.], [300., 600.], [300., np.nan], [100., 300.], [100., 300.], [300., np.nan]])
    groups = expanded.copy(); groups[3, 1] = np.nan
    got = e.stratum_masks(original, np.isfinite(original), expanded, np.isfinite(expanded),
        original, np.isfinite(original), groups, np.isfinite(groups),
        np.array([300., 600., 300., 300., 300., np.nan]), np.full(6, 300.), np.array([True, True, True, True, False, True]))
    assert got['added_only'].tolist() == [True, False, False, False, False, False]
    assert got['counterexample'].tolist() == [False, True, False, False, False, False]
    assert not (got['added_only'] & got['counterexample']).any()
    assert got['broad'].tolist() == [True, True, True, True, False, True]


@pytest.mark.parametrize('bad', ['absent_added', 'eviction', 'nan_truth', 'mask_dtype', 'infinite_added'])
def test_invalid_acquisition_stratum_inputs_refused(bad):
    old = np.array([[300.]]); new = np.array([[300., 600.]]); added = np.array([600.]); truth = np.array([300.]); mask = np.isfinite(old)
    if bad == 'absent_added': added[0] = 500.
    elif bad == 'eviction': new = np.array([[600., np.nan]])
    elif bad == 'nan_truth': truth[0] = np.nan
    elif bad == 'infinite_added': added[0] = np.inf
    else: mask = mask.astype(int)
    with pytest.raises(ValueError): e.stratum_masks(old, mask, new, np.isfinite(new), old, np.isfinite(old), new, np.isfinite(new), added, truth, np.ones(1, bool))


def test_feasibility_floor_exact_and_failure_has_no_fallback():
    strata, ids = toy_support(); result = e.feasibility(strata, ids)
    assert result['pass_result'] and result['strata']['added_only']['rows'] == 576
    assert result['strata']['counterexample']['distinct_cases'] == 24
    assert result['strata']['counterexample']['distinct_carriers'] == 6
    strata['added_only'][:] = False
    assert not e.feasibility(strata, ids)['pass_result']
    with pytest.raises(ValueError, match='Insufficient'): e.targeted_schedule(strata, ids, 20261022, 0)


@pytest.mark.parametrize('field', ['case', 'overlap', 'outside_broad', 'dtype'])
def test_malformed_feasibility_inputs_refused(field):
    strata, ids = toy_support()
    if field == 'case': ids[0] = 512
    elif field == 'overlap': strata['counterexample'][0] = True
    elif field == 'outside_broad': strata['broad'][0] = False
    else: strata['broad'] = strata['broad'].astype(float)
    with pytest.raises(ValueError): e.feasibility(strata, ids)


def test_schedule_reproduces_exact_shape_quotas_case_identity_and_replacement():
    strata, ids = toy_support(); one = e.targeted_schedule(strata, ids, 20261022, 0)
    same = e.targeted_schedule(strata, ids, 20261022, 0); other = e.targeted_schedule(strata, ids, 20261022, 1)
    assert np.array_equal(one['indices'], same['indices']) and np.array_equal(one['origin'], same['origin'])
    assert not np.array_equal(one['indices'], other['indices'])
    assert one['indices'].shape == one['origin'].shape == (33, 512)
    report = e.schedule_exposure(one, strata, ids)
    assert report['selected_rows'] == 16896 and report['requested_strata'] == {'added_only': 4224, 'counterexample': 4224, 'broad': 8448}
    assert report['duplicate_draws'] == 16896-report['unique_rows'] > 0
    assert sum(report['per_case_selected_rows']) == 16896
    assert report['actual_strata']['added_only'] > report['requested_strata']['added_only']
    assert report['actual_strata']['counterexample'] > report['requested_strata']['counterexample']
    assert report['actual_strata']['broad'] == 16896


def test_schedule_cannot_hide_wrong_origin_or_out_of_range_row():
    strata, ids = toy_support(); schedule = e.targeted_schedule(strata, ids, 20261022, 0)
    schedule['origin'][0, 0] = 7
    with pytest.raises(ValueError): e.schedule_exposure(schedule, strata, ids)
    schedule = e.targeted_schedule(strata, ids, 20261022, 0); schedule['indices'][0, 0] = len(ids)
    with pytest.raises(ValueError): e.schedule_exposure(schedule, strata, ids)


def test_full_source_metadata_closure_bound():
    sources = e.source_hashes()
    for name in ('evidence_cases.py', 'expanded_cases.py', 'harmonic_challenge.py', 'synthetic_curriculum.py', 'hybrid_followup.py', 'alias_diagnostic.py'):
        assert 'research/voice-prosody/phenometrix_prosody/'+name in sources
    assert 'apps/capture-web/src/voice-pitch-features-v2.ts' in sources


def test_training_and_exact_twelve_final_checkpoint_freezes(tmp_path, monkeypatch):
    sp, initial, cp, final = fake_freezes(tmp_path, monkeypatch)
    e._validate_freeze(sp, e.previous._sha(sp), challenge=False)
    _, hashes = e._validate_freeze(cp, e.previous._sha(cp), challenge=True)
    assert len(final['checkpoints']) == 12 and all(str(tmp_path/v['path']) in hashes for v in final['checkpoints'].values())


@pytest.mark.parametrize('field,value', [('status', 'fitting'), ('maximum_fits', 9), ('arms', ['targeted__on']),
    ('training_spec_sha256', '0'*64), ('challenge_spec_sha256', '0'*64), ('source_sha256', {}),
    ('deadline_utc', '2026-10-24T12:18:58Z'), ('fits', {'one': {}})])
def test_bad_source_freeze_refuses_before_pcm(tmp_path, monkeypatch, field, value):
    sp, initial, _, _ = fake_freezes(tmp_path, monkeypatch); initial[field] = value; sp.write_text(json.dumps(initial))
    with pytest.raises(ValueError): e.generate_training(tmp_path/'runs'/'output', sp, e.previous._sha(sp))
    assert not (tmp_path/'runs'/'output').exists()


@pytest.mark.parametrize('change', ['missing', 'mutated', 'absolute', 'alias', 'extra', 'partial'])
def test_incomplete_or_wrong_final_checkpoints_refused(tmp_path, monkeypatch, change):
    _, _, cp, final = fake_freezes(tmp_path, monkeypatch); name = sorted(final['checkpoints'])[0]; row = final['checkpoints'][name]
    p = tmp_path/row['path']
    if change == 'missing': p.unlink()
    elif change == 'mutated': p.write_bytes(b'changed')
    elif change == 'absolute': row['path'] = str(p)
    elif change == 'alias': row['path'] = final['checkpoints'][sorted(final['checkpoints'])[1]]['path']
    elif change == 'extra': final['checkpoints']['extra'] = row
    else: del final['checkpoints'][name]
    cp.write_text(json.dumps(final))
    with pytest.raises(ValueError): e.generate_challenge(tmp_path/'runs'/'output', cp, e.previous._sha(cp))
    assert not (tmp_path/'runs'/'output').exists()


def test_expired_deadline_refuses_before_any_generation(tmp_path, monkeypatch):
    class Expired(datetime):
        @classmethod
        def now(cls, tz=None): return cls(2026, 10, 23, 12, 18, 59, tzinfo=timezone.utc)
    sp, _, _, _ = fake_freezes(tmp_path, monkeypatch); monkeypatch.setattr(e, 'datetime', Expired)
    with pytest.raises(ValueError, match='deadline expired'): e.generate_training(tmp_path/'runs'/'output', sp, e.previous._sha(sp))
    assert not (tmp_path/'runs'/'output').exists()


def test_changed_sources_and_linked_initial_freeze_refused(tmp_path, monkeypatch):
    sp, _, cp, final = fake_freezes(tmp_path, monkeypatch)
    sp.write_text('{}')
    with pytest.raises(ValueError, match='Linked initial'): e._validate_freeze(cp, e.previous._sha(cp), challenge=True)
    sp, _, cp, _ = fake_freezes(tmp_path/'again', monkeypatch)
    (tmp_path/'again'/'fixture.py').write_text('changed')
    with pytest.raises(ValueError, match='source changed'): e._validate_freeze(cp, e.previous._sha(cp), challenge=True)


def test_missing_or_wrong_phase_cannot_generate_any_output(tmp_path, monkeypatch):
    sp, _, cp, _ = fake_freezes(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='phase'): e.generate_challenge(tmp_path/'runs'/'fresh', sp, e.previous._sha(sp))
    with pytest.raises(ValueError, match='phase'): e.generate_training(tmp_path/'runs'/'training', cp, e.previous._sha(cp))
    with pytest.raises(ValueError, match='required'): e.generate_training(tmp_path/'runs'/'training', tmp_path/'runs'/'missing', '0'*64)
    assert not (tmp_path/'runs'/'fresh').exists() and not (tmp_path/'runs'/'training').exists()


def test_read_only_freeze_verification_remains_possible_after_deadline(tmp_path, monkeypatch):
    class Expired(datetime):
        @classmethod
        def now(cls, tz=None): return cls(2026, 10, 24, 5, 0, 0, tzinfo=timezone.utc)
    _, _, cp, _ = fake_freezes(tmp_path, monkeypatch); monkeypatch.setattr(e, 'datetime', Expired)
    frozen, hashes = e._validate_freeze(cp, e.previous._sha(cp), challenge=True, enforce_deadline=False)
    assert frozen['status'] == 'checkpoints_frozen' and str(cp) in hashes
    with pytest.raises(ValueError, match='deadline expired'):
        e._validate_freeze(cp, e.previous._sha(cp), challenge=True)
