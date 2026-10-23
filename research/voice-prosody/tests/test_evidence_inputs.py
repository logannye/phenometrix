"""Fixture-only byte binding, provenance, source retention and cache ordering."""
import hashlib
import json
from pathlib import Path
import wave
import numpy as np
import pytest
import torch

from phenometrix_prosody import evidence_inputs as module
from phenometrix_prosody.model import FEATURE_NAMES_V2
from phenometrix_prosody.expanded_candidate_features import FEATURE_NAMES_EXPANDED
from phenometrix_prosody.periodicity_evidence import evidence_from_pcm


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pcm_sha(pcm):
    return hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(count=4000):
    # Literal toy inputs, not a study waveform or a previously observed error.
    time = np.arange(count) / 48000
    pcm = (.1 * np.sin(2 * np.pi * 200 * time)).astype(np.float32)
    ends = 2208 + 480 * np.arange(3)
    def feature(k):
        f = np.zeros((3, 8 * k + 9), np.float32)
        hz = np.full((3, k), np.nan)
        mask = np.zeros((3, k), bool)
        frequencies = [100., 200.] if k == 6 else [100., 200., 200.2]
        for i, frequency in enumerate(frequencies):
            scores = [.4, .2, 0., 0., 0.] if i < 2 else [0., 0., 0., .9, 0.]
            f[:, 8 * i:8 * (i + 1)] = [1, np.log2(frequency / 50), *scores, max(scores)]
            hz[:, i], mask[:, i] = frequency, True
        f[:, 8 * k:] = [-3, .1, 0, 0, .2, .7, .8, len(frequencies) / 6, 1]
        return dict(features=f, candidate_hz=hz, candidate_mask=mask, reference_sample=ends.copy(),
            feature_names=np.array(FEATURE_NAMES_V2 if k == 6 else FEATURE_NAMES_EXPANDED),
            sample_rate_hz=np.array(48000), pcm_sample_count=np.array(count))
    return pcm, feature(6), feature(7)


def test_record_evidence_uses_grouped_emitted_hz_not_raw_slot_or_frequency_average():
    pcm, original, expanded = fixture()
    copied = [{k: v.copy() for k, v in f.items()} for f in (original, expanded)]
    arrays = module.record_evidence(original, expanded, pcm, pcm_sha(pcm))
    # 200 and200.2 are within5c; stronger200.2 is the actual emitted member.
    np.testing.assert_array_equal(arrays['grouped_candidate_hz'][:, :2], [[100., 200.2]] * 3)
    assert arrays['grouped_member_mask'][:, 1, 1:3].all()
    assert not arrays['grouped_candidate_mask'][:, 2:].any()
    direct = evidence_from_pcm(pcm, original['reference_sample'], arrays['grouped_candidate_hz'], arrays['grouped_candidate_mask'])
    np.testing.assert_array_equal(arrays['evidence'], direct.features)
    assert np.all(arrays['period_samples'][:, 1] == 48000 / 200.2)
    for before, after in zip(copied, (original, expanded)):
        for key in before: np.testing.assert_array_equal(before[key], after[key])


@pytest.mark.parametrize('change', [
    lambda f: f['reference_sample'].__setitem__(0, 2209),
    lambda f: f['features'].__setitem__((0, 56), -4),
    lambda f: f['features'].__setitem__((0, 2), .5),
    lambda f: f['candidate_hz'].__setitem__((0, 0), 101),
    lambda f: f.__setitem__('reference_hz', np.array([200., 200., 200.])),
    lambda f: f.__setitem__('feature_names', np.array(FEATURE_NAMES_V2)),
])
def test_changed_source_support_labels_or_schema_refused(change):
    pcm, original, expanded = fixture(); change(expanded)
    with pytest.raises(ValueError): module.record_evidence(original, expanded, pcm, pcm_sha(pcm))


def test_pcm_byte_identity_length_dtype_and_nonfinite_refused():
    pcm, original, expanded = fixture()
    with pytest.raises(ValueError, match='SHA256'): module.record_evidence(original, expanded, pcm, '0' * 64)
    with pytest.raises(ValueError, match='SHA256'): module.record_evidence(original, expanded, pcm, 'bad')
    with pytest.raises(ValueError, match='length'): module.record_evidence(original, expanded, pcm[:-1], pcm_sha(pcm[:-1]))
    with pytest.raises(ValueError, match='Float32'): module.record_evidence(original, expanded, pcm.astype(float), pcm_sha(pcm))
    pcm[20] = np.nan
    with pytest.raises(ValueError, match='finite'): module.record_evidence(original, expanded, pcm, pcm_sha(pcm))


def test_item_path_and_explicit_array_routes_match_and_only_add_provenance(tmp_path):
    pcm, original, expanded = fixture()
    path = tmp_path / 'toy.f32le'; pcm.astype('<f4').tofile(path)
    item = dict(feature=original, expanded_feature=expanded)
    first = module.evidence_for_item(item, path, digest(path))
    assert first.dtype == torch.float32 and first.shape == (3, 7, 3)
    assert set(item) == {'feature', 'expanded_feature', 'periodicity_evidence'}
    second_item = dict(feature=original, expanded_feature=expanded, pcm=pcm)
    second = module.evidence_for_item(second_item, None, digest(path))
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    with pytest.raises(ValueError): module.evidence_for_item(dict(feature=original, expanded_feature=expanded), None, digest(path))
    with pytest.raises(ValueError): module.evidence_for_item(item, path, '0' * 64)
    path.write_bytes(b'abc')
    with pytest.raises(ValueError): module.evidence_for_item(item, path, digest(path))


def local_roots(monkeypatch, tmp_path):
    repo = tmp_path / 'repo'; root = repo / 'research/voice-prosody'; runs = root / 'runs'
    prior = runs / 'expanded-pool-v1'; prior.mkdir(parents=True)
    for name, value in [('REPO', repo), ('ROOT', root), ('RUNS', runs), ('PRIOR', prior)]:
        monkeypatch.setattr(module, name, value)
    monkeypatch.setattr(module, 'source_hashes', lambda: {'fixture-source': '1' * 64})
    return repo, root, runs, prior


def natural_cache(monkeypatch, tmp_path):
    repo, root, runs, prior = local_roots(monkeypatch, tmp_path)
    folder = runs / 'delayed-train-features'; (folder / 'features').mkdir(parents=True)
    pcm, original, expanded = fixture()
    mic = root / 'data/ptdb-tug/MIC/F02_fixture.wav'; mic.parent.mkdir(parents=True)
    pcm16 = np.rint(pcm * 32768).astype('<i2')
    with wave.open(str(mic), 'wb') as wav:
        wav.setparams((1, 2, 48000, len(pcm16), 'NONE', 'not compressed')); wav.writeframes(pcm16.tobytes())
    pcm = pcm16.astype(np.float32) / np.float32(32768)
    identity = 'F02_fixture__clean'
    original_path = folder / 'features' / (identity + '.npz'); np.savez_compressed(original_path, **original)
    expanded_path = prior / 'train/features' / (identity + '.npz'); expanded_path.parent.mkdir(parents=True)
    np.savez_compressed(expanded_path, **expanded)
    record = dict(recording_id=identity, source_recording_id='F02_fixture', speaker='F02', condition='clean',
        sample_count=len(pcm), microphone_sha256=digest(mic), transformed_pcm_sha256=pcm_sha(pcm),
        feature_sha256=digest(original_path), start=0, stop=3)
    source = dict(recording_id='F02_fixture', role='mic', split='train', path='MIC/F02_fixture.wav', sha256=digest(mic))
    # A nonexistent REF sentinel proves the adapter uses metadata but never reads its content.
    acquisition = dict(files=[source, dict(recording_id='F02_fixture', role='ref', split='train',
                                          path='REF/must-not-open.ref', sha256='0' * 64)])
    write_json(folder / 'manifest.json', dict(split='train', recordings=[record]))
    write_json(folder / 'acquisition-manifest.json', acquisition)
    saved = {**record, 'original_feature_path': str(original_path.relative_to(repo)),
        'original_feature_sha256': digest(original_path), 'expanded_feature_path': str(expanded_path.relative_to(repo)),
        'expanded_feature_sha256': digest(expanded_path)}
    cache = dict(status='complete', split='train', records=[saved], original_manifest_path=str((folder/'manifest.json').relative_to(repo)),
                 original_manifest_sha256=digest(folder/'manifest.json'))
    write_json(prior / 'train/manifest.json', cache)
    previous = dict(schema='phenometrix-expanded-pool-experiment.v1', status='complete',
        input_sha256={str((prior/'train/manifest.json').relative_to(repo)): digest(prior/'train/manifest.json')})
    write_json(prior/'manifest.json', previous)
    monkeypatch.setattr(module, 'PRIOR_MANIFEST_SHA256', digest(prior/'manifest.json'))
    data = dict(split='train', folder=folder, acquisition=acquisition, records=[record],
        hashes={str(folder/'acquisition-manifest.json'): digest(folder/'acquisition-manifest.json')},
        raw={k: original[k].copy() for k in ('features', 'candidate_hz', 'candidate_mask')})
    return data, runs/'new-study', {}, prior


def test_natural_cache_binds_existing_bytes_preserves_voice_and_all_frame_order(monkeypatch, tmp_path):
    data, output, manifest, _ = natural_cache(monkeypatch, tmp_path)
    voice = data['raw']['features'].copy(); calls = []
    result = module.cache_natural(data, output, manifest, lambda: calls.append('budget'))
    assert calls == ['budget', 'budget']
    np.testing.assert_array_equal(data['raw']['features'], voice)
    assert result['raw']['features'].shape == (3, 65) and result['evidence'].shape == (3, 7, 3)
    assert len(result['evidence_records']) == 1
    cache = json.loads(result['evidence_manifest_path'].read_text())
    assert cache['status'] == 'complete' and cache['frames'] == 3 and not cache['reference_content_read']
    row = cache['records'][0]; path = module.REPO / row['evidence_path']
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z['reference_sample'], [2208, 2688, 3168])
        np.testing.assert_array_equal(z['evidence'], result['evidence'])
        assert z['correlations'].shape == (3, 7, 4)
    assert manifest['input_sha256'][row['evidence_path']] == digest(path)
    with pytest.raises(FileExistsError): module.cache_natural(data, output, manifest, lambda: None)


def test_natural_cache_rejects_mutated_expanded_feature_and_records_failure(monkeypatch, tmp_path):
    data, output, manifest, prior = natural_cache(monkeypatch, tmp_path)
    path = next((prior/'train/features').glob('*.npz')); path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='identity'): module.cache_natural(data, output, manifest, lambda: None)
    cache = json.loads((output/'evidence/train/manifest.json').read_text())
    assert cache['status'] == 'failed' and not cache['records']


def test_natural_cache_refuses_sealed_before_any_prior_access(monkeypatch):
    monkeypatch.setattr(module, '_prior', lambda manifest: pytest.fail('No sealed access may occur'))
    with pytest.raises(ValueError, match='train/development'):
        module.cache_natural({'split': 'sealed_test'}, Path('/unused'), {}, lambda: None)


@pytest.mark.parametrize('field,value', [('start', 1), ('condition', 'clip01'), ('microphone_sha256', '0'*64)])
def test_natural_record_mismatch_refused(monkeypatch, tmp_path, field, value):
    data, output, manifest, _ = natural_cache(monkeypatch, tmp_path)
    data['records'][0][field] = value
    with pytest.raises(ValueError, match='identity'): module.cache_natural(data, output, manifest, lambda: None)


def test_original_in_memory_input_mismatch_refused(monkeypatch, tmp_path):
    data, output, manifest, _ = natural_cache(monkeypatch, tmp_path)
    data['raw']['features'][0, 0] = 0
    with pytest.raises(ValueError, match='in-memory'): module.cache_natural(data, output, manifest, lambda: None)


def test_prior_ledger_requires_exact_identity_and_conflicts_refuse(monkeypatch, tmp_path):
    repo, _, _, _ = local_roots(monkeypatch, tmp_path)
    path = repo/'input.txt'; path.write_text('fixture'); name = str(path.relative_to(repo)); sha = digest(path)
    with pytest.raises(ValueError, match='lacks'): module._prior_bound({}, {}, path)
    assert module._prior_bound({}, {'historical_input_sha256': {name: sha}}, path) == path
    with pytest.raises(ValueError, match='consistent'):
        module._prior_bound({}, {'input_sha256': {name: sha}, 'historical_input_sha256': {name: '0'*64}}, path)
    with pytest.raises(ValueError, match='identity'):
        module._prior_bound({}, {'input_sha256': {name: sha}}, path, '0'*64)
    with pytest.raises(ValueError, match='escaped'): module._bind({}, tmp_path/'outside', sha)


def test_historical_loaders_reject_incomplete_inventory_before_any_feature_read(monkeypatch, tmp_path):
    repo, _, _, prior = local_roots(monkeypatch, tmp_path)
    training = prior/'training/manifest.json'; write_json(training, dict(status='complete', records=[]))
    analytic = prior/'analytic-inventory.json'; write_json(analytic, dict(regression=[]))
    pinned = dict(input_sha256={str(p.relative_to(repo)): digest(p) for p in (training, analytic)})
    monkeypatch.setattr(module, '_prior', lambda m: pinned)
    monkeypatch.setattr(module, '_npz', lambda path: pytest.fail('No feature read before full inventory guard'))
    with pytest.raises(ValueError, match='96'): module.load_existing_training({})
    with pytest.raises(ValueError, match='74'): module.load_historical_analytic({})


def test_historical_training_loader_preserves_all96_order_and_never_parses_labels(monkeypatch, tmp_path):
    repo, _, _, prior = local_roots(monkeypatch, tmp_path)
    pcm, original, expanded = fixture()
    op, ep, pp = prior/'original.npz', prior/'expanded.npz', prior/'existing.f32le'
    np.savez_compressed(op, **original); np.savez_compressed(ep, **expanded); pcm.tofile(pp)
    rows = [dict(id=f'fixture-{i}', scenario={'id':f'fixture-{i}'},
        original_feature_path=str(op.relative_to(repo)), original_feature_sha256=digest(op),
        expanded_feature_path=str(ep.relative_to(repo)), expanded_feature_sha256=digest(ep),
        pcm_path=str(pp.relative_to(repo)), pcm_sha256=digest(pp)) for i in range(96)]
    path = prior/'training/manifest.json'; write_json(path, dict(status='complete', records=rows))
    pinned = dict(input_sha256={str(p.relative_to(repo)): digest(p) for p in (op, ep, pp, path)})
    monkeypatch.setattr(module, '_prior', lambda m: pinned)
    loaded = module.load_existing_training({})
    assert [item['id'] for item in loaded] == [row['id'] for row in rows]
    assert all(item['pcm'] is None and item['pcm_path'] == pp for item in loaded)
    assert all(item['timing'].availability_sample.tolist() == [2208, 2688, 3168] for item in loaded)


def test_complete74_fixture_resolver_keeps_every_case_and_hashes_unparsed_predictions(monkeypatch, tmp_path):
    repo, _, runs, prior = local_roots(monkeypatch, tmp_path)
    pcm, original, expanded = fixture()
    op, ep, prediction = prior/'original.npz', prior/'expanded.npz', prior/'predictions/never-parse-prediction.npz'
    prediction.parent.mkdir()
    np.savez_compressed(op, **original); np.savez_compressed(ep, **expanded)
    prediction.write_bytes(b'intentionally not an NPZ: only identity may be read')
    pinned = dict(input_sha256={str(p.relative_to(repo)): digest(p) for p in (op, ep)},
        prediction_sha256={str(prediction.relative_to(prior)):digest(prediction)})
    folders = {'consumed_harmonic':runs/'harmonic-context-v1/challenge',
        'consumed_subharmonic':runs/'subharmonic-objective-v1/challenge',
        'consumed_augmentation':runs/'harmonic-augmentation-v1/challenge',
        'consumed_coverage':runs/'candidate-coverage-v1/challenge', 'challenge':prior/'challenge'}
    inventory = {}
    for group, count in [('regression',14), *[(key,12) for key in folders]]:
        rows, pcm_rows = [], []
        for i in range(count):
            identity = f'{group}-fixture-{i}'
            folder = runs/'support-diagnostic-v1/pcm' if group == 'regression' else folders[group]/'pcm'
            folder.mkdir(parents=True, exist_ok=True)
            path = folder/(identity+'.f32le'); pcm.tofile(path)
            pinned['input_sha256'][str(path.relative_to(repo))] = digest(path)
            rows.append(dict(id=identity, scenario={'id':identity}, sample_count=len(pcm),
                original_feature_path=str(op.relative_to(repo)), original_feature_sha256=digest(op),
                expanded_feature_path=str(ep.relative_to(repo)), expanded_feature_sha256=digest(ep),
                pcm_sha256=digest(path), prediction_path=str(prediction.relative_to(prior)), prediction_sha256=digest(prediction)))
            if group != 'regression':
                pcm_rows.append(dict(id=identity, pcm_path=str(path.relative_to(folders[group])), pcm_sha256=digest(path)))
        inventory[group] = rows
        if group != 'regression':
            path = folders[group]/'manifest.json'; write_json(path, dict(status='complete',records=pcm_rows))
            # Exercise the narrowly authenticated inherited-manifest route.
            pinned.setdefault('historical_input_sha256',{})[str(path.relative_to(repo))] = digest(path)
    path = prior/'analytic-inventory.json'; write_json(path, inventory)
    pinned['input_sha256'][str(path.relative_to(repo))] = digest(path)
    monkeypatch.setattr(module, '_prior', lambda m: pinned)
    result = module.load_historical_analytic({})
    assert sum(len(items) for items in result.values()) == 74
    assert 'challenge' not in result and len(result['consumed_expansion']) == 12
    assert [i['id'] for i in result['regression']] == [r['id'] for r in inventory['regression']]
    assert all(item['prior_prediction_path'] == prediction and item['pcm'] is None
               for items in result.values() for item in items)


def test_prior_prediction_uses_only_completed_output_map_not_input_map(monkeypatch, tmp_path):
    repo, _, _, prior = local_roots(monkeypatch, tmp_path)
    path=prior/'predictions/fixture.npz';path.parent.mkdir();path.write_bytes(b'prediction')
    row={'prediction_path':'predictions/fixture.npz','prediction_sha256':digest(path)}
    previous={'prediction_sha256':{row['prediction_path']:digest(path)}}
    assert module._prior_prediction({},previous,row)==path
    with pytest.raises(ValueError,match='output binding'):
        module._prior_prediction({}, {'input_sha256':{str(path.relative_to(repo)):digest(path)}},row)
    with pytest.raises(ValueError,match='output binding'):
        module._prior_prediction({},previous,{**row,'prediction_sha256':'0'*64})
    with pytest.raises(ValueError,match='escaped'):
        module._prior_prediction({},previous,{**row,'prediction_path':'../outside.npz'})
