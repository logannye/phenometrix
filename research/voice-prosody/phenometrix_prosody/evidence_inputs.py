"""Input-only periodicity caches over the previously frozen expanded pool.

Natural microphone reconstruction reuses the exact frozen corruption function.
This module never opens a reference trajectory, generates a NEW waveform,
changes candidates, computes labels, or reads prediction/metric content. Three
historical observation fixtures may be exactly reconstructed and SHA checked.
The run owner
must freeze sources before invoking these functions on scientific inputs.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import torch

from .candidate_coverage_features import _validate as validate_original_feature
from .expanded_candidate_features import FEATURE_NAMES_EXPANDED
from .expanded_selector_model import prepare_expanded_selector_batch
from .periodicity_evidence import evidence_from_pcm, SCHEMA as EVIDENCE_SCHEMA, FEATURE_NAMES
from .features import transform, CONDITIONS
from .pilot import read_mic
from .measurement_alignment import align_feature_timing

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'research/voice-prosody'
RUNS = ROOT / 'runs'
PRIOR = RUNS / 'expanded-pool-v1'
PRIOR_MANIFEST_SHA256 = 'c80c0f95fba8dfa5f327de7cf228200206fea8e8a94f72a26a2d79d1cb27d8fb'
SCHEMA = 'phenometrix-periodicity-input-cache.v1'


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _valid_sha(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('An exact lowercase SHA256 identity is required')
    return value


def _inside(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    if not path.is_relative_to(parent):
        raise ValueError('Input/output path escaped its declared root')
    return path


def _bind(manifest, path, expected=None):
    path = _inside(path, REPO)
    digest = _sha(path)
    if expected is not None and digest != _valid_sha(expected):
        raise ValueError('Input byte identity differs: ' + str(path.relative_to(REPO)))
    name = str(path.relative_to(REPO))
    for mapping in (manifest.get('historical_input_sha256', {}), manifest.setdefault('input_sha256', {})):
        if name in mapping and mapping[name] != digest:
            raise ValueError('Previously bound input changed: ' + name)
    manifest['input_sha256'][name] = digest
    return path


def _npz(path):
    with np.load(path, allow_pickle=False) as handle:
        return {name: handle[name].copy() for name in handle.files}


def _save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def _save_npz(path, arrays):
    with Path(path).open('xb') as handle:
        np.savez_compressed(handle, **arrays)


def validate_feature_pair(original, expanded):
    """Validate unchanged input timing, globals and raw-candidate preservation.

    The exact cached bytes are authenticated separately. This check additionally
    prevents accidentally evaluating new evidence at old slots, averaged Hz,
    changed support, or expanded inputs intended for the original voicing model.
    """
    old_x, old_hz, old_mask, ends, count = validate_original_feature(original)
    if (set(expanded) != set(original)
            or list(expanded.get('feature_names', [])) != list(FEATURE_NAMES_EXPANDED)
            or not np.array_equal(expanded.get('reference_sample'), ends)
            or np.asarray(expanded.get('sample_rate_hz')).shape != ()
            or int(expanded['sample_rate_hz']) != 48000
            or np.asarray(expanded.get('pcm_sample_count')).shape != ()
            or int(expanded['pcm_sample_count']) != count):
        raise ValueError('Expanded/original feature schema, support or PCM length differs')
    x, hz, mask = (np.asarray(expanded[name]) for name in ('features', 'candidate_hz', 'candidate_mask'))
    if (x.dtype != np.float32 or x.shape != (len(old_x), 65) or hz.dtype != np.float64
            or hz.shape != (len(old_x), 7) or mask.dtype != np.bool_ or mask.shape != hz.shape
            or not np.isnan(hz[~mask]).all()):
        raise ValueError('Expected unchanged Float32 expanded65 and Float64/boolean seven-candidate arrays')
    old_blocks, blocks = old_x[:, :48].reshape(-1, 6, 8), x[:, :56].reshape(-1, 7, 8)
    global_indices = [0, 1, 2, 3, 4, 5, 6, 8]
    if not np.array_equal(old_x[:, 48:][:, global_indices], x[:, 56:][:, global_indices]):
        raise ValueError('Non-count global acquisition evidence changed')
    for row in range(len(x)):
        if np.any(np.diff(hz[row, mask[row]]) <= 0):
            raise ValueError('Expanded raw candidates must remain strictly sorted unique Hz')
        for slot in np.flatnonzero(old_mask[row]):
            match = np.flatnonzero(mask[row] & (hz[row] == old_hz[row, slot]))
            if len(match) != 1:
                raise ValueError('Original raw candidate was lost or duplicated')
            current, previous = blocks[row, match[0]], old_blocks[row, slot]
            if (not np.array_equal(current[[0, 1, 2, 3, 4, 6]], previous[[0, 1, 2, 3, 4, 6]])
                    or current[5] < previous[5]):
                raise ValueError('Original source evidence changed outside admitted full40 enhancement')
    grouped = prepare_expanded_selector_batch(x, hz, mask)
    return grouped, ends, count


def record_evidence(original_feature, expanded_feature, pcm, expected_pcm_sha256):
    """Return complete input-only arrays; no caller arrays are mutated."""
    expected_pcm_sha256 = _valid_sha(expected_pcm_sha256)
    samples = np.asarray(pcm)
    if samples.ndim != 1 or samples.dtype != np.float32 or not np.isfinite(samples).all():
        raise ValueError('Expected finite one-dimensional acquired Float32 PCM')
    actual = hashlib.sha256(samples.astype('<f4', copy=False).tobytes()).hexdigest()
    if actual != expected_pcm_sha256:
        raise ValueError('Reconstructed PCM SHA256 differs from frozen acquisition')
    grouped, ends, count = validate_feature_pair(original_feature, expanded_feature)
    if len(samples) != count:
        raise ValueError('PCM length differs from original/expanded feature cache')
    evidence = evidence_from_pcm(samples, ends, grouped.candidate_hz.numpy(), grouped.candidate_mask.numpy())
    return dict(schema=np.array(EVIDENCE_SCHEMA), feature_names=np.array(FEATURE_NAMES),
        evidence=evidence.features, correlations=evidence.correlations, valid=evidence.valid,
        period_samples=evidence.period_samples, common_length=evidence.common_length,
        grouped_candidate_hz=grouped.candidate_hz.numpy().copy(),
        grouped_candidate_mask=grouped.candidate_mask.numpy().copy(),
        grouped_member_mask=grouped.member_mask.numpy().copy(),
        reference_sample=np.asarray(ends).copy(), sample_rate_hz=np.array(48000),
        pcm_sample_count=np.array(count), pcm_sha256=np.array(actual))


def evidence_for_item(item, pcm_path, expected_sha):
    """Return Float32 Torch channels for ALL frames, retaining full provenance.

    ``item`` supplies the original57 ``feature`` and unchanged65
    ``expanded_feature``. A path names existing little-endian Float32 PCM. For
    historical fixtures whose PCM was not retained, None accepts ONLY an
    already-reconstructed ``item['pcm']``; this function never synthesizes it.
    The hash is checked before extraction in both routes. ``periodicity_evidence``
    is the only added item key, so the orchestrator can save all raw correlations.
    """
    expected_sha = _valid_sha(expected_sha)
    if pcm_path is None:
        if 'pcm' not in item or item['pcm'] is None:
            raise ValueError('Existing PCM path or explicit hash-bound PCM array is required')
        pcm = np.asarray(item['pcm'])
    else:
        raw = Path(pcm_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha or len(raw) % 4:
            raise ValueError('PCM file byte identity or Float32 alignment differs')
        pcm = np.frombuffer(raw, dtype='<f4').copy()
    arrays = record_evidence(item['feature'], item['expanded_feature'], pcm, expected_sha)
    item['periodicity_evidence'] = arrays
    return torch.from_numpy(arrays['evidence'])


def _prior(manifest):
    path = _bind(manifest, PRIOR / 'manifest.json', PRIOR_MANIFEST_SHA256)
    value = json.loads(path.read_text())
    if value.get('status') != 'complete' or value.get('schema') != 'phenometrix-expanded-pool-experiment.v1':
        raise ValueError('Expected the exact completed expanded-pool study')
    return value


def _prior_bound(manifest, prior, path, expected=None):
    """Only identities inherited from the exact pinned completed run qualify."""
    path = _inside(path, REPO)
    name = str(path.relative_to(REPO))
    values = []
    for mapping in (prior.get('input_sha256', {}), prior.get('historical_input_sha256', {})):
        value = mapping.get(name, mapping.get(str(path)))
        if value is not None:
            values.append(value)
    if not values or len(set(values)) != 1 or (expected is not None and expected != values[0]):
        raise ValueError('Historical input lacks an exact consistent prior identity: ' + name)
    return _bind(manifest, path, values[0])


def _loaded_item(manifest, prior, row):
    fp = _prior_bound(manifest, prior, REPO / row['original_feature_path'], row['original_feature_sha256'])
    ep = _prior_bound(manifest, prior, REPO / row['expanded_feature_path'], row['expanded_feature_sha256'])
    feature, expanded = _npz(fp), _npz(ep)
    validate_feature_pair(feature, expanded)
    count = int(feature['pcm_sample_count'])
    if 'sample_count' in row and count != row['sample_count']:
        raise ValueError('Historical item sample count changed')
    return dict(id=row['id'], scenario=row['scenario'], feature=feature, expanded_feature=expanded,
        timing=align_feature_timing(feature, measurement_delay_ms=20),
        original_feature_path=fp, expanded_feature_path=ep,
        pcm_sha256=row['pcm_sha256'], record=dict(row), pcm=None)


def _prior_prediction(manifest, prior, row):
    # Generated prior predictions are outputs, hence absent from its input map.
    # Their separate completed-run output map is the authoritative binding.
    path = _inside(PRIOR / row['prediction_path'], PRIOR / 'predictions')
    relative = str(path.relative_to(PRIOR))
    expected = prior.get('prediction_sha256', {}).get(relative)
    if expected is None or expected != row['prediction_sha256']:
        raise ValueError('Prior prediction lacks its exact completed output binding')
    return _bind(manifest, path, expected)


def load_existing_training(manifest):
    """Load the exact 96 existing balance-bank input features and PCM paths.

    This function does not compute analytic target arrays or select a subset.
    Invoke only after the new study's source freeze permits scientific input
    reads. Existing scenario metadata is returned unchanged to the orchestrator.
    """
    prior = _prior(manifest)
    path = _prior_bound(manifest, prior, PRIOR / 'training/manifest.json')
    cache = json.loads(path.read_text())
    if (cache.get('status') != 'complete' or len(cache.get('records', [])) != 96
            or len({r['id'] for r in cache['records']}) != 96):
        raise ValueError('Expected exact existing 96-case training bank')
    result = []
    for row in cache['records']:
        item = _loaded_item(manifest, prior, row)
        item['pcm_path'] = _prior_bound(manifest, prior, REPO / row['pcm_path'], row['pcm_sha256'])
        result.append(item)
    return result


def load_historical_analytic(manifest):
    """Load all 74 consumed inputs; old challenge becomes consumed_expansion.

    No new challenge is selected or generated. Existing PCM is located through
    its exact frozen domain manifest. Three original observation signals had
    no persisted PCM; their source-bound deterministic reconstruction MUST match
    the already-frozen original PCM SHA before returning it. No predicted result
    or reference trajectory is parsed; prior prediction files are only hashed.
    """
    prior = _prior(manifest)
    inventory_path = _prior_bound(manifest, prior, PRIOR / 'analytic-inventory.json')
    inventory = json.loads(inventory_path.read_text())
    expected = dict(regression=14, consumed_harmonic=12, consumed_subharmonic=12,
        consumed_augmentation=12, consumed_coverage=12, challenge=12)
    if set(inventory) != set(expected) or any(len(inventory[g]) != count for g, count in expected.items()):
        raise ValueError('Expected all 74 consumed analytic inputs')
    folders = dict(consumed_harmonic=RUNS / 'harmonic-context-v1/challenge',
        consumed_subharmonic=RUNS / 'subharmonic-objective-v1/challenge',
        consumed_augmentation=RUNS / 'harmonic-augmentation-v1/challenge',
        consumed_coverage=RUNS / 'candidate-coverage-v1/challenge', challenge=PRIOR / 'challenge')
    result = {}
    for group, rows in inventory.items():
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate consumed analytic identity')
        pcm_rows = None
        if group != 'regression':
            folder = folders[group]
            pcm_manifest = json.loads(_prior_bound(manifest, prior, folder / 'manifest.json').read_text())
            pcm_rows = {r['id']: r for r in pcm_manifest.get('records', [])}
            if (pcm_manifest.get('status') != 'complete' or len(pcm_manifest.get('records', [])) != 12
                    or set(pcm_rows) != {r['id'] for r in rows}):
                raise ValueError('Consumed PCM manifest does not match its exact 12-case inventory')
        items = []
        for row in rows:
            item = _loaded_item(manifest, prior, row)
            if group == 'regression' and row['id'].startswith('observation-change-'):
                from .evaluate_synthetic import observation_pcm, OBSERVATION_SCENARIOS
                source = Path(observation_pcm.__code__.co_filename).resolve()
                expected_source = prior['source_sha256'].get(str(source.relative_to(REPO)))
                if expected_source is None or _sha(source) != expected_source:
                    raise ValueError('Historical observation generator source changed')
                known = {r['id']: r for r in OBSERVATION_SCENARIOS}
                if known.get(row['id']) != row['scenario']:
                    raise ValueError('Historical observation reconstruction parameters changed')
                pcm = observation_pcm(row['scenario'])
                if hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest() != row['pcm_sha256']:
                    raise ValueError('Historical observation PCM reconstruction differs')
                item.update(pcm_path=None, pcm=pcm)
            elif group == 'regression':
                item['pcm_path'] = _prior_bound(manifest, prior,
                    RUNS / 'support-diagnostic-v1/pcm' / (row['id'] + '.f32le'), row['pcm_sha256'])
            else:
                pcm_record = pcm_rows[row['id']]
                if pcm_record['pcm_sha256'] != row['pcm_sha256']:
                    raise ValueError('Historical analytic PCM identity differs between inventories')
                item['pcm_path'] = _prior_bound(manifest, prior,
                    _inside(folders[group] / pcm_record['pcm_path'], folders[group]), row['pcm_sha256'])
            item['prior_prediction_path'] = _prior_prediction(manifest, prior, row)
            items.append(item)
        result['consumed_expansion' if group == 'challenge' else group] = items
    return result


def cache_natural(data, output, manifest, budget_callback):
    """Reuse the frozen expanded natural cache and original microphone order.

    ``data`` is selector_experiment.load_inputs(train/development). ``manifest``
    is the new run's mutable input-hash ledger; budget_callback() is called
    before and after every record. Return the original data metadata with
    expanded ``raw`` and concatenated Float32 ``evidence`` aligned to ALL rows.
    Keep the incoming original data separately for original57 voicing.
    """
    split = data.get('split')
    if split not in ('train', 'development'):
        raise ValueError('Evidence cache admits only existing train/development inputs')
    output = _inside(output, RUNS)
    previous = _prior(manifest)
    prior_path = PRIOR / split / 'manifest.json'
    prior_name = str(prior_path.relative_to(REPO))
    prior = json.loads(_bind(manifest, prior_path, previous['input_sha256'][prior_name]).read_text())
    expected_rows = data['records']
    if (prior.get('status') != 'complete' or prior.get('split') != split
            or [r['recording_id'] for r in prior['records']] != [r['recording_id'] for r in expected_rows]):
        raise ValueError('Prior expanded cache must match exact original recording order')
    original_manifest = _bind(manifest, data['folder'] / 'manifest.json', prior['original_manifest_sha256'])
    if original_manifest != (REPO / prior['original_manifest_path']).resolve():
        raise ValueError('Original feature manifest path differs from frozen prior')
    acquisition_path = _bind(manifest, data['folder'] / 'acquisition-manifest.json',
        data['hashes'][str(data['folder'] / 'acquisition-manifest.json')])
    acquisition = json.loads(acquisition_path.read_text())
    if acquisition != data['acquisition']:
        raise ValueError('In-memory microphone inventory differs from its bound bytes')
    micro_rows = [r for r in acquisition['files'] if r['role'] == 'mic' and r['split'] == split]
    micro = {r['recording_id']: r for r in micro_rows}
    if len(micro) != len(micro_rows):
        raise ValueError('Duplicate microphone identities')
    folder = output / 'evidence' / split
    folder.mkdir(parents=True, exist_ok=False)
    cache = dict(schema=SCHEMA, status='extracting', split=split, started_at=_now(),
        source_sha256=source_hashes(), original_manifest_sha256=_sha(original_manifest),
        prior_manifest_path=str((PRIOR / 'manifest.json').relative_to(REPO)),
        prior_manifest_sha256=PRIOR_MANIFEST_SHA256, prior_split_manifest_sha256=_sha(prior_path),
        reference_content_read=False, original_voicing_features_changed=False, records=[])
    _save_json(folder / 'manifest.json', cache)
    pieces, channels, offset = [], [], 0
    try:
        for index, (record, saved) in enumerate(zip(expected_rows, prior['records'])):
            budget_callback()
            for name in ('recording_id', 'source_recording_id', 'condition', 'sample_count', 'microphone_sha256',
                         'transformed_pcm_sha256', 'feature_sha256', 'start', 'stop'):
                if record[name] != saved[name]:
                    raise ValueError('Original/expanded record identity differs: ' + name)
            if record['condition'] not in CONDITIONS or record['start'] != offset:
                raise ValueError('Natural condition or contiguous frame order differs')
            fp = _bind(manifest, REPO / saved['original_feature_path'], saved['original_feature_sha256'])
            if fp != (data['folder'] / 'features' / (record['recording_id'] + '.npz')).resolve():
                raise ValueError('Original feature path differs')
            ep = _bind(manifest, REPO / saved['expanded_feature_path'], saved['expanded_feature_sha256'])
            original, expanded = _npz(fp), _npz(ep)
            for name in ('features', 'candidate_hz', 'candidate_mask'):
                if not np.array_equal(original[name], data['raw'][name][record['start']:record['stop']], equal_nan=True):
                    raise ValueError('Original in-memory voicing/acquisition input differs')
            source = micro[record['source_recording_id']]
            if source['sha256'] != record['microphone_sha256']:
                raise ValueError('Frozen microphone identity differs')
            mic_path = _bind(manifest, ROOT / 'data/ptdb-tug' / source['path'], source['sha256'])
            pcm, _ = transform(read_mic(mic_path), record['condition'], record['source_recording_id'], split)
            _bind(manifest, mic_path, source['sha256'])
            arrays = record_evidence(original, expanded, pcm, record['transformed_pcm_sha256'])
            if len(arrays['evidence']) != record['stop'] - record['start']:
                raise ValueError('Evidence frame count differs from original row offsets')
            target = folder / (record['recording_id'] + '.npz')
            _save_npz(target, arrays); _bind(manifest, target)
            row = {name: record[name] for name in ('recording_id', 'source_recording_id', 'speaker', 'condition', 'start', 'stop')}
            row.update(evidence_path=str(target.relative_to(REPO)), evidence_sha256=_sha(target),
                pcm_sha256=record['transformed_pcm_sha256'], sample_count=len(pcm),
                original_feature_path=str(fp.relative_to(REPO)), original_feature_sha256=_sha(fp),
                expanded_feature_path=str(ep.relative_to(REPO)), expanded_feature_sha256=_sha(ep),
                microphone_path=str(mic_path.relative_to(REPO)), microphone_sha256=source['sha256'])
            cache['records'].append(row)
            channels.append(arrays['evidence'])
            pieces.append({name: expanded[name] for name in ('features', 'candidate_hz', 'candidate_mask')})
            offset = record['stop']
            budget_callback()
            if (index + 1) % 32 == 0 or index + 1 == len(expected_rows):
                _save_json(folder / 'manifest.json', cache)
                print(json.dumps(dict(phase='periodicity_cache', split=split, records=index + 1,
                                      total_records=len(expected_rows), frames=offset)), flush=True)
        if offset != len(data['raw']['features']):
            raise ValueError('Natural evidence omitted original frames')
        cache.update(status='complete', completed_at=_now(), frames=offset)
        _save_json(folder / 'manifest.json', cache); _bind(manifest, folder / 'manifest.json')
    except BaseException as error:
        cache.update(status='failed', failure=repr(error), failed_at=_now())
        _save_json(folder / 'manifest.json', cache)
        raise
    return {**data, 'raw': {name: np.concatenate([p[name] for p in pieces]) for name in pieces[0]},
        'evidence': np.concatenate(channels), 'evidence_records': cache['records'],
        'evidence_manifest_path': folder / 'manifest.json', 'evidence_manifest_sha256': _sha(folder / 'manifest.json')}


def source_hashes():
    from .expanded_selector_model import source_hashes as model_sources
    from .periodicity_evidence import source_hashes as periodicity_sources
    result = model_sources() | periodicity_sources()
    from .evaluate_synthetic import observation_pcm
    for path in (Path(__file__), Path(transform.__code__.co_filename), Path(read_mic.__code__.co_filename),
                 Path(align_feature_timing.__code__.co_filename), Path(observation_pcm.__code__.co_filename)):
        result[str(path.resolve().relative_to(REPO))] = _sha(path)
    return result
