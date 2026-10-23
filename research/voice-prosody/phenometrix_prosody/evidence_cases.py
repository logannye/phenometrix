"""Fixed TRAIN exposure metadata and post-checkpoint prospective challenge.

The training strata use analytic TRAIN truth and acquisition candidates only.
Neither model outputs nor development/test references enter this module. Reading
specifications or importing the module never synthesizes PCM. Historical sources
are unchanged. Public generation entry points enforce byte-bound phase freezes.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np

from . import harmonic_challenge as previous
from . import expanded_cases, augmentation_cases, synthetic_curriculum, hybrid_followup, alias_diagnostic
from .evaluate import validate_feature

REPO, ROOT, RUNS = previous.REPO, previous.ROOT, previous.RUNS
SEEDS = (20261022, 20261023, 20261024)
ARMS = ('existing__off', 'existing__on', 'targeted__off', 'targeted__on')
TRAIN_FREQUENCIES = tuple(round(53.113 + 10.125*i, 3) for i in range(64))
TRAIN_PHASES = ((.17, -.59, 1.23), (1.71, .43, -1.11))
TRAIN_GAINS = (.035, .075)
TRAIN_AFTER = ((0., 1., .5), (.2, 1., .5))
FREQUENCIES = (109.9, 241.1, 379.3, 563.9, 619.7, 677.9)
PHASES = ((.79, -1.37, .23), (2.17, .61, -.89))
BATCH_COUNTS = {'added_only': 128, 'counterexample': 128, 'broad': 256}
FLOOR = {'rows': 512, 'distinct_cases': 24, 'distinct_carriers': 6, 'distinct_phases': 2, 'distinct_gains': 2}
FREQUENCY_BINS = (50., 150., 250., 350., 450., 550., 650., 700.)
TRAIN_MAX_BYTES = 512*1024*1024
CHALLENGE_MAX_BYTES = 40*1024*1024
TRAIN_SCHEMA = 'phenometrix-evidence-train-spec-v1'
CHALLENGE_SCHEMA = 'phenometrix-evidence-challenge-spec-v1'
CACHE_SCHEMA = 'phenometrix-evidence-acquisition-cache-v1'


def _record(identity, frequency, phases, gain, after):
    return {'id': identity, 'frequency_hz': frequency, 'phase_radians': list(phases), 'gain': gain,
        'sample_rate_hz': 48000, 'sample_count': 96000, 'change_sample': 39840,
        'before': [1., .5, .25], 'after': list(after)}


def _shared_spec():
    return {'sample_rate_hz': 48000, 'sample_count': 96000, 'duration_seconds_per_sequence': 2.,
        'feature_version': 'v2', 'measurement_delay_ms': 20, 'support_samples': 1920, 'hop_samples': 480,
        'acquisition_end_grid': '2208+480*k samples', 'reference_relation': 'exclusive_support_end-960',
        'common_mask': 'reference>=3072 and reference+1536<=sample_count',
        'common_eligible_frames_per_sequence': 191, 'fully_post_change_frames_per_sequence': 112,
        'post_region': 'exclusive_support_end-1920>=39840', 'noise_or_normalization': 'none',
        'identifiability': 'nonzero harmonics2and3 preserve shortest period1/F0 for both post mixtures',
        'novelty_boundary': 'fixed carriers differ from historical configured carriers/endpoints and TRAIN carriers; continuous glides and natural speech can cross them; not disjoint support or external generalization'}


def historical_carriers():
    """Only frozen source metadata; no waveform, feature, reference or score reads."""
    plans = synthetic_curriculum.make_plans()
    records = [*plans['train']['records'], *plans['challenge']['records'], *hybrid_followup.fresh_records()]
    values = {float(r[k]) for r in records for k in ('initial_hz', 'final_hz') if r[k] is not None}
    for frequencies in (previous.FREQUENCIES, expanded_cases.subharmonic_challenge.FREQUENCIES,
        augmentation_cases.TRAIN_FREQUENCIES, augmentation_cases.CHALLENGE_FREQUENCIES,
        expanded_cases.coverage_cases.FREQUENCIES, expanded_cases.FREQUENCIES, alias_diagnostic.FREQUENCIES):
        values.update(frequencies)
    values.update((49.99, 50., 50.01, 100., 150., 175., 200., 225., 250., 300., 500., 699.99, 700., 700.01))
    return sorted(values)


def validate_metadata():
    old = set(historical_carriers()); train = set(TRAIN_FREQUENCIES); fresh = set(FREQUENCIES)
    previous._require(len(train) == 64 and len(fresh) == 6 and not train & old and not fresh & old and not train & fresh,
        'TRAIN/fresh configured carrier collision')
    previous._require(all(50 <= f <= 700 for f in train | fresh), 'Configured frequency range differs')
    return {'historical_configured_carriers': len(old), 'train_carriers': 64, 'fresh_carriers': 6,
        'carrier_intersections': [], 'historical_carriers_sha256': hashlib.sha256(previous._canonical(sorted(old))).hexdigest()}


def training_spec():
    records = []
    for i, frequency in enumerate(TRAIN_FREQUENCIES):
        for j, phases in enumerate(TRAIN_PHASES):
            for k, gain in enumerate(TRAIN_GAINS):
                for m, after in enumerate(TRAIN_AFTER):
                    r = _record(f'evidence-train-f{i}-p{j}-g{k}-m{m}', frequency, phases, gain, after)
                    r.update(frequency_index=i, phase_index=j, gain_index=k, mixture_index=m)
                    records.append(r)
    return {**_shared_spec(), 'schema': TRAIN_SCHEMA, 'split': 'train', 'records': records,
        'sequence_count': 512, 'common_eligible_rows': 97792, 'duration_seconds_total': 1024,
        'maximum_output_bytes': TRAIN_MAX_BYTES, 'novelty': validate_metadata(),
        'target_policy': 'analytic constant F0 and voiced=true on all common rows; no model-based mining',
        'strata': {
            'added_only': 'common and no original RAW candidate within50c and distinct added peak within50c and expanded GROUPED candidate within50c',
            'counterexample': 'common and distinct added peak farther than50c and original GROUPED and expanded GROUPED candidates within50c',
            'broad': 'ALL common rows, including uncovered rows; overlaps the targeted strata'},
        'feasibility_floor_per_targeted_stratum': dict(FLOOR), 'frequency_bin_edges_hz': list(FREQUENCY_BINS),
        'batch_counts': dict(BATCH_COUNTS), 'updates_per_epoch': 33, 'epochs': 12,
        'sampling': 'for each batch and stratum, uniform case then uniform qualifying frame within case, both with replacement; concatenate strata in declared order then permute512 positions',
        'sampling_seed': 'numpy.default_rng(seed+2000000+epoch), separate from legacy bank seed+1000000+epoch',
        'duplicate_policy': 'replacement is deliberate; persist every global row index and stratum origin; report unique/duplicate rows plus actual overlapping strata and per-case/region exposure',
        'insufficient_support': 'stop before fitting; no enlargement, alternate generator, relaxed floor or retry',
        'interpretation': 'new TRAIN bank plus fixed sampling package; not isolation of oversampling from broader bank exposure',
        'access': 'TRAIN generation requires exact source/specification freeze; prospective challenge remains ungenerated'}


def challenge_spec():
    records = [_record(f'evidence-balance-f{i}-p{j}', f, p, .057, (0., 1., .5))
        for i, f in enumerate(FREQUENCIES) for j, p in enumerate(PHASES)]
    return {**_shared_spec(), 'schema': CHALLENGE_SCHEMA, 'split': 'prospective-challenge', 'records': records,
        'sequence_count': 12, 'duration_seconds_total': 24, 'maximum_output_bytes': CHALLENGE_MAX_BYTES,
        'novelty': validate_metadata(), 'predesignated_candidate': 'targeted__on',
        'prospective_gate': 'each designated seed: >=95% within50c in EACH fully post-change case; no frame drops',
        'retained_case_count': 74, 'total_case_count': 86, 'mandatory_case_gates_per_seed': 78,
        'retained_gates': 'all inherited natural and candidate prerequisites plus3stationary+3originalchange+60consumedpost+12freshpost',
        'independence': 'one prospectively fixed set of new complete configurations from the familiar failure-motivated noiseless family; no model selection or second fresh set',
        'access': 'no fresh PCM or features until exact linked source and complete twelve-final-checkpoint freezes validate'}


def training_spec_sha256(): return hashlib.sha256(previous._canonical(training_spec())).hexdigest()
def challenge_spec_sha256(): return hashlib.sha256(previous._canonical(challenge_spec())).hexdigest()


def source_hashes():
    sources = {**expanded_cases.source_hashes()}
    for module in (synthetic_curriculum, hybrid_followup, alias_diagnostic):
        p = Path(module.__file__); sources[str(p.relative_to(REPO))] = previous._sha(p)
    sources[str(Path(__file__).relative_to(REPO))] = previous._sha(Path(__file__))
    return sources


def _coverage(hz, mask, truth):
    hz, mask, truth = np.asarray(hz), np.asarray(mask), np.asarray(truth)
    previous._require(hz.ndim == mask.ndim == 2 and hz.shape == mask.shape and mask.dtype == bool
        and truth.shape == (len(hz),) and np.isfinite(truth).all() and np.all((truth >= 50) & (truth <= 700)),
        'Invalid TRAIN frequency/mask/target extent')
    previous._require(np.isfinite(hz[mask]).all() and np.all((hz[mask] >= 50) & (hz[mask] <= 700)),
        'Invalid available TRAIN candidate')
    delta = np.full(hz.shape, np.inf, dtype=np.float64)
    np.log2(np.divide(hz, truth[:, None], where=mask, out=np.ones(hz.shape, np.float64)), out=delta)
    return (mask & (np.abs(1200*delta) <= 50)).any(axis=1)


def stratum_masks(original_hz, original_mask, expanded_hz, expanded_mask,
                   original_group_hz, original_group_mask, expanded_group_hz, expanded_group_mask,
                   added_hz, truth, common):
    """Pure TRAIN truth/acquisition classification; no logits/probability input."""
    truth = np.asarray(truth); common = np.asarray(common); added = np.asarray(added_hz)
    previous._require(common.dtype == bool and common.shape == added.shape == truth.shape, 'TRAIN stratum extent differs')
    original_raw = _coverage(original_hz, original_mask, truth)
    expanded_raw = _coverage(expanded_hz, expanded_mask, truth)
    original_grouped = _coverage(original_group_hz, original_group_mask, truth)
    expanded_grouped = _coverage(expanded_group_hz, expanded_group_mask, truth)
    previous._require(not np.isinf(added).any(), 'Infinite added peak is invalid; onlyNaN denotes absence')
    valid_added = np.isfinite(added)
    previous._require(np.all((added[valid_added] >= 50) & (added[valid_added] <= 700)), 'Added peak range differs')
    original_match = (np.asarray(original_mask) & (np.asarray(original_hz) == added[:, None])).any(1)
    expanded_match = (np.asarray(expanded_mask) & (np.asarray(expanded_hz) == added[:, None])).any(1)
    distinct = valid_added & ~original_match
    previous._require(np.all(~distinct | expanded_match), 'Added candidate absent from expanded pool')
    added_correct = np.zeros(len(truth), bool)
    added_correct[valid_added] = np.abs(1200*np.log2(added[valid_added]/truth[valid_added])) <= 50
    previous._require(not np.any(original_raw & ~expanded_raw), 'Expansion evicted original raw coverage')
    added_only = common & ~original_raw & distinct & added_correct & expanded_grouped
    counter = common & distinct & ~added_correct & original_grouped & expanded_grouped
    previous._require(not np.any(added_only & counter), 'Targeted strata overlap unexpectedly')
    return {'added_only': added_only, 'counterexample': counter, 'broad': common.copy()}


def feasibility(strata, case_index):
    """Return fixed support checks; failure never creates a fallback schedule."""
    case_index = np.asarray(case_index)
    previous._require(case_index.ndim == 1 and np.issubdtype(case_index.dtype, np.integer)
        and np.all((case_index >= 0) & (case_index < 512)), 'Invalid TRAIN case identity')
    previous._require(set(strata) == set(BATCH_COUNTS), 'TRAIN stratum inventory differs')
    for v in strata.values():
        previous._require(np.asarray(v).dtype == bool and np.asarray(v).shape == case_index.shape, 'TRAIN stratum shape/dtype differs')
    previous._require(np.all(~strata['added_only'] | strata['broad']) and np.all(~strata['counterexample'] | strata['broad'])
        and not np.any(strata['added_only'] & strata['counterexample']), 'TRAIN stratum relationships differ')
    records = training_spec()['records']; result = {}
    for name in ('added_only', 'counterexample'):
        selected = np.flatnonzero(strata[name]); cases = np.unique(case_index[selected]); rows = [records[int(i)] for i in cases]
        values = {'rows': int(len(selected)), 'distinct_cases': int(len(cases)),
            'distinct_carriers': len({r['frequency_index'] for r in rows}),
            'distinct_phases': len({r['phase_index'] for r in rows}), 'distinct_gains': len({r['gain_index'] for r in rows})}
        checks = {k: values[k] >= v for k, v in FLOOR.items()}
        values.update(checks=checks, pass_result=all(checks.values()),
            frequency_bin_rows=np.histogram([records[int(i)]['frequency_hz'] for i in case_index[selected]], FREQUENCY_BINS)[0].tolist())
        result[name] = values
    return {'pass_result': bool(all(v['pass_result'] for v in result.values())), 'floor': dict(FLOOR),
        'strata': result, 'broad_rows': int(np.count_nonzero(strata['broad'])),
        'on_failure': 'no fits; no enlargement or revised threshold'}


def targeted_schedule(strata, case_index, seed, epoch):
    """Exactly33x512 row indices/origins; fixed case-balanced replacement."""
    previous._require(seed in SEEDS and isinstance(epoch, int) and 0 <= epoch < 12, 'Unknown seed/epoch')
    proof = feasibility(strata, case_index)
    previous._require(proof['pass_result'], 'Insufficient fixed TRAIN stratum support: no fits')
    case_index = np.asarray(case_index); rng = np.random.default_rng(seed + 2_000_000 + epoch)
    pools = {}
    for name in BATCH_COUNTS:
        indices = np.flatnonzero(strata[name]); cases = np.unique(case_index[indices])
        previous._require(len(cases) > 0, 'Empty TRAIN sampling stratum')
        pools[name] = (cases, {int(c): indices[case_index[indices] == c] for c in cases})
    order = np.empty((33, 512), np.int64); origin = np.empty((33, 512), np.uint8)
    for batch in range(33):
        offset = 0
        for label, (name, count) in enumerate(BATCH_COUNTS.items()):
            cases, per_case = pools[name]
            sampled_cases = rng.choice(cases, size=count, replace=True)
            for j, case in enumerate(sampled_cases):
                candidates = per_case[int(case)]
                order[batch, offset+j] = candidates[rng.integers(len(candidates))]
            origin[batch, offset:offset+count] = label; offset += count
        shuffle = rng.permutation(512); order[batch] = order[batch, shuffle]; origin[batch] = origin[batch, shuffle]
    return {'indices': order, 'origin': origin}


def schedule_exposure(schedule, strata, case_index):
    """Counts saved draws, actual overlaps and repeated rows without concealment."""
    order, origin = np.asarray(schedule['indices']), np.asarray(schedule['origin'])
    case_index = np.asarray(case_index)
    previous._require(order.shape == origin.shape == (33, 512) and order.dtype == np.int64 and origin.dtype == np.uint8,
        'TRAIN schedule schema differs')
    previous._require(np.all((order >= 0) & (order < len(case_index))) and np.isin(origin, [0,1,2]).all(), 'TRAIN schedule indices differ')
    for label, (name, count) in enumerate(BATCH_COUNTS.items()):
        previous._require(np.all((origin == label).sum(1) == count) and np.all(strata[name][order[origin == label]]),
            'TRAIN schedule stratum contract differs')
    flat = order.ravel(); by_case = np.bincount(case_index[flat], minlength=512)
    return {'selected_rows': int(flat.size), 'unique_rows': int(np.unique(flat).size),
        'duplicate_draws': int(flat.size-np.unique(flat).size),
        'requested_strata': {name: int((origin == label).sum()) for label, name in enumerate(BATCH_COUNTS)},
        'actual_strata': {name: int(strata[name][flat].sum()) for name in BATCH_COUNTS},
        'per_case_selected_rows': by_case.tolist()}


def _expected_checkpoints():
    return {f'{arm}__seed{seed}' for arm in ARMS for seed in SEEDS}


def _run_relative(value, reason):
    previous._require(isinstance(value, str) and value and not Path(value).is_absolute(), reason)
    p = previous._inside(REPO, value)
    previous._require(p.is_relative_to(RUNS.resolve()), 'Bound artifact must remain in ignored runs')
    return p


def _remaining(frozen):
    previous._require(frozen.get('deadline_utc') == '2026-10-23T12:18:58Z', 'Authorized session deadline differs')
    remaining = (datetime.fromisoformat(frozen['deadline_utc'].replace('Z', '+00:00'))-datetime.now(timezone.utc)).total_seconds()
    previous._require(remaining > 0, 'Authorized overnight deadline expired')
    return remaining


def _validate_freeze(path, expected_sha, *, challenge, enforce_deadline=True):
    path = previous._inside(RUNS, Path(path).resolve())
    previous._require(path.is_file(), 'A complete declared freeze is required before PCM generation')
    previous._require(previous._sha(path) == expected_sha, 'Freeze identity changed')
    frozen = json.loads(path.read_text())
    kind, status = ('checkpoint', 'checkpoints_frozen') if challenge else ('source', 'sources_frozen')
    previous._require(frozen.get('schema') == f'phenometrix-evidence-{kind}-freeze.v1'
        and frozen.get('status') == status, 'Required source/checkpoint freeze phase differs')
    previous._require(frozen.get('arms') == list(ARMS) and frozen.get('seeds') == list(SEEDS)
        and frozen.get('maximum_fits') == 12, 'Declared arm/seed/fit budget differs')
    previous._require(frozen.get('training_spec_sha256') == training_spec_sha256()
        and frozen.get('challenge_spec_sha256') == challenge_spec_sha256(), 'Frozen TRAIN/fresh specification differs')
    previous._require(frozen.get('deadline_utc') == '2026-10-23T12:18:58Z', 'Authorized session deadline differs')
    if enforce_deadline: _remaining(frozen)
    sources = source_hashes(); closure = frozen.get('source_sha256', {})
    previous._require(isinstance(closure, dict) and all(closure.get(p) == h for p, h in sources.items()),
        'Freeze lacks matching full acquisition/metadata source closure')
    hashes = {str(path): expected_sha}
    for name, expected in closure.items():
        previous._require(isinstance(name, str) and not Path(name).is_absolute(), 'Frozen source path must be repository-relative')
        p = previous._inside(REPO, name)
        previous._require(p.is_file() and previous._sha(p) == expected, 'Frozen scientific source changed or missing')
        hashes[str(p)] = expected
    if not challenge:
        previous._require(frozen.get('fits') == {}, 'TRAIN generation must precede all fitting')
        return frozen, hashes
    source_path = _run_relative(frozen.get('source_freeze_path'), 'Source freeze path must be repository-relative')
    previous._require(source_path.parent == path.parent and source_path.is_file()
        and previous._sha(source_path) == frozen.get('source_freeze_sha256'), 'Linked initial source freeze changed or missing')
    initial, initial_hashes = _validate_freeze(source_path, frozen['source_freeze_sha256'], challenge=False, enforce_deadline=enforce_deadline)
    previous._require(initial['source_sha256'] == closure, 'Initial/final scientific source closure differs')
    hashes.update(initial_hashes)
    checkpoints = frozen.get('checkpoints', {})
    previous._require(isinstance(checkpoints, dict) and set(checkpoints) == _expected_checkpoints(),
        'Exactly twelve declared final checkpoints required')
    for name, binding in checkpoints.items():
        previous._require(isinstance(binding, dict), 'Malformed final checkpoint binding')
        p = _run_relative(binding.get('path'), 'Checkpoint path must be repository-relative')
        previous._require(p == path.parent/'checkpoints'/(name+'.pt'), 'Checkpoint path/name differs from declared fit')
        previous._require(p.is_file() and previous._sha(p) == binding.get('sha256'), 'Final checkpoint changed or missing')
        hashes[str(p)] = binding['sha256']
    return frozen, hashes


def _extract_feature(raw, full, timeout_seconds):
    """Unchanged v2 replay, with bounded subprocess time and retained input hash."""
    subprocess.run(['node', '--max-old-space-size=1536', '--import', 'tsx',
        '../../research/voice-prosody/replay.ts', '--feature-version', 'v2', '--input-f32', str(raw),
        '--sample-rate', '48000', '--output', str(full)], cwd=REPO/'apps/capture-web',
        check=True, capture_output=True, text=True, timeout=timeout_seconds)
    previous._require(full.is_file() and full.stat().st_size <= 4*1024*1024, 'Replay export missing or too large')
    value = json.loads(full.read_text())
    previous._require(value.get('inputSha256') == previous._sha(raw) and value.get('inputSampleCount') == 96000
        and value.get('featureVersion') == 'v2', 'Replay PCM/feature identity differs')
    frames = value['frames']
    hz = np.array([[c['hz'] if c['valid'] else np.nan for c in f['features']['candidates']] for f in frames], np.float64)
    result = {'features': np.array([f['features']['values'] for f in frames], np.float32),
        'candidate_hz': hz, 'candidate_mask': np.isfinite(hz),
        'reference_sample': np.array([f['referenceSample'] for f in frames], np.int64),
        'feature_names': np.array(value['featureNames']), 'sample_rate_hz': np.array(48000),
        'pcm_sample_count': np.array(96000)}
    _timing(result)
    return result


def _timing(feature):
    timing = validate_feature(feature, {'sample_count': 96000})
    previous._require(np.array_equal(timing.availability_sample, np.arange(2208, 96001, 480))
        and int(timing.common_eligible.sum()) == 191, 'Acquisition grid/common support differs')
    return timing


def _generate(output, freeze_path, expected_sha, *, challenge):
    freeze_path = Path(freeze_path).resolve()
    frozen, _ = _validate_freeze(freeze_path, expected_sha, challenge=challenge)
    output = previous._inside(RUNS, Path(output).resolve())
    previous._require(not output.exists(), 'Generation refuses to overwrite prior evidence')
    previous._require(output.parent.exists() and shutil.disk_usage(output.parent).free >= 16*1024**3,
        'At least16GiB free disk required')
    spec = challenge_spec() if challenge else training_spec(); sources = source_hashes()
    prefix = 'checkpoint' if challenge else 'source'; started = time.monotonic()
    output.mkdir()
    manifest = {'schema': CACHE_SCHEMA, 'split': spec['split'], 'status': 'generating', 'started_at': previous._now(),
        'spec_sha256': challenge_spec_sha256() if challenge else training_spec_sha256(),
        'training_spec_sha256': training_spec_sha256(), 'challenge_spec_sha256': challenge_spec_sha256(),
        'freeze_kind': prefix, 'freeze_path': str(freeze_path.relative_to(REPO)), 'freeze_sha256': expected_sha,
        prefix+'_freeze_path': str(freeze_path.relative_to(REPO)), prefix+'_freeze_sha256': expected_sha,
        'source_sha256': sources, 'records': [], 'reference_content_read': False,
        'model_outputs_read': False, 'maximum_output_bytes': spec['maximum_output_bytes'],
        'deadline_utc': frozen['deadline_utc'], 'maximum_generation_seconds': 2700,
        'feature_input': 'finite float32 PCM only enters unchanged v2 replay; TRAIN targets never enter extraction'}
    previous._save(output/'manifest.json', manifest)
    try:
        for name in ('pcm', 'features', 'intermediate', 'source-snapshot'): (output/name).mkdir()
        previous._save(output/'spec.json', spec)
        for name, expected in sources.items():
            snapshot = output/'source-snapshot'/name; snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes((REPO/name).read_bytes())
            previous._require(previous._sha(snapshot) == expected, 'Acquisition source changed during snapshot')
        for index, record in enumerate(spec['records']):
            left = min(_remaining(frozen), 2700-(time.monotonic()-started))
            previous._require(left > 0, 'Bounded acquisition time exceeded')
            raw = output/'pcm'/(record['id']+'.f32le'); full = output/'intermediate'/(record['id']+'.json')
            previous._synthesize_pcm(record).tofile(raw)
            feature = _extract_feature(raw, full, min(60., left)); timing = _timing(feature)
            path = output/'features'/(record['id']+'.npz'); np.savez_compressed(path, **feature)
            export_sha = previous._sha(full); full.unlink()
            manifest['records'].append({'id': record['id'], 'case_index': index,
                'parameters_sha256': hashlib.sha256(previous._canonical(record)).hexdigest(),
                'pcm_path': str(raw.relative_to(output)), 'pcm_sha256': previous._sha(raw),
                'feature_path': str(path.relative_to(output)), 'feature_sha256': previous._sha(path),
                'typescript_export_sha256': export_sha, 'feature_frames': len(feature['features']),
                'common_eligible_frames': int(timing.common_eligible.sum())})
            previous._require(sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) <= spec['maximum_output_bytes'],
                'Acquisition artifact cap exceeded')
            previous._save(output/'manifest.json', manifest)
            if (index+1) % 16 == 0 or index+1 == len(spec['records']):
                print(json.dumps({'phase': 'evidence_input_generation', 'split': spec['split'],
                    'completed_records': index+1, 'total_records': len(spec['records'])}), flush=True)
        (output/'intermediate').rmdir()
        _validate_freeze(freeze_path, expected_sha, challenge=challenge)
        previous._require(source_hashes() == sources, 'Acquisition source changed during generation')
        manifest.update(status='complete', completed_at=previous._now(), elapsed_seconds=time.monotonic()-started)
        previous._save(output/'manifest.json', manifest)
    except BaseException as error:
        manifest.update(status='failed', failed_at=previous._now(), failure=repr(error))
        previous._save(output/'manifest.json', manifest)
        raise
    return _load(output, freeze_path, expected_sha, challenge=challenge)


def _load(output, freeze_path, expected_sha, *, challenge):
    output = previous._inside(RUNS, Path(output).resolve()); freeze_path = Path(freeze_path).resolve()
    _, hashes = _validate_freeze(freeze_path, expected_sha, challenge=challenge, enforce_deadline=False)
    manifest = json.loads((output/'manifest.json').read_text())
    spec = challenge_spec() if challenge else training_spec(); sources = source_hashes(); prefix = 'checkpoint' if challenge else 'source'
    previous._require(json.loads((output/'spec.json').read_text()) == spec, 'Cached acquisition specification differs')
    previous._require(manifest.get('schema') == CACHE_SCHEMA and manifest.get('status') == 'complete'
        and manifest.get('split') == spec['split'] and manifest.get('freeze_kind') == prefix
        and manifest.get('freeze_path') == str(freeze_path.relative_to(REPO)) and manifest.get('freeze_sha256') == expected_sha
        and manifest.get(prefix+'_freeze_path') == str(freeze_path.relative_to(REPO)) and manifest.get(prefix+'_freeze_sha256') == expected_sha
        and manifest.get('source_sha256') == sources and manifest.get('training_spec_sha256') == training_spec_sha256()
        and manifest.get('challenge_spec_sha256') == challenge_spec_sha256()
        and manifest.get('spec_sha256') == hashlib.sha256(previous._canonical(spec)).hexdigest(), 'Cached acquisition provenance differs')
    previous._require([r['id'] for r in manifest['records']] == [r['id'] for r in spec['records']], 'Acquisition inventory differs')
    for name, expected in sources.items():
        snapshot = previous._inside(output/'source-snapshot', name)
        previous._require(previous._sha(snapshot) == expected, 'Acquisition source snapshot changed')
        hashes[str(snapshot)] = expected
    rows = []
    for index, (record, saved) in enumerate(zip(spec['records'], manifest['records'])):
        previous._require(saved['case_index'] == index
            and saved['parameters_sha256'] == hashlib.sha256(previous._canonical(record)).hexdigest(), 'Acquisition parameters/order differ')
        for name, key in (('pcm_path', 'pcm_sha256'), ('feature_path', 'feature_sha256')):
            p = previous._inside(output, saved[name])
            previous._require(previous._sha(p) == saved[key], 'Cached acquisition bytes changed')
            hashes[str(p)] = saved[key]
        with np.load(previous._inside(output, saved['feature_path']), allow_pickle=False) as z:
            feature = {k: z[k].copy() for k in z.files}
        timing = _timing(feature)
        previous._require(saved['feature_frames'] == len(feature['features']) == 196
            and saved['common_eligible_frames'] == 191, 'Cached acquisition frame extent differs')
        rows.append({'id': record['id'], 'case_index': index, 'record': record,
            'scenario': previous._scenario(record), 'timing': timing, 'feature': feature})
    for name in ('manifest.json', 'spec.json'): hashes[str(output/name)] = previous._sha(output/name)
    return rows, hashes


def generate_training(output, freeze_path, expected_sha): return _generate(output, freeze_path, expected_sha, challenge=False)
def load_training(output, freeze_path, expected_sha): return _load(output, freeze_path, expected_sha, challenge=False)
def generate_challenge(output, freeze_path, expected_sha): return _generate(output, freeze_path, expected_sha, challenge=True)
def load_challenge(output, freeze_path, expected_sha): return _load(output, freeze_path, expected_sha, challenge=True)
def challenge_region_at(scenario, end_sample): return previous.challenge_region_at(scenario, end_sample)
