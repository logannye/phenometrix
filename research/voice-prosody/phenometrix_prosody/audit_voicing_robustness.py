"""Independent evidence audit for the prespecified matched voicing intervention.

No fitting, acquisition, augmentation, producer forward/schedule/gate calls or
external outcome access. Historical inputs remain immutable. All score equations
below are independent of the producer; NumPy reconstructs all MLP outputs.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import numpy as np

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'research/voice-prosody'
SEEDS = (20261025, 20261026, 20261027)
CONDITIONS = ('clean', 'noise10db', 'first_difference', 'clip01')
SHIFTED = CONDITIONS[1:]
SPEAKERS = ('F01', 'F03', 'F05', 'F06', 'M01', 'M03', 'M05', 'M06')
BASELINE_SHA = '06110469a9a643fb16806905569ebc939db2c0bdd5b3a6a1593b31789453d10d'
HISTORICAL_IDENTITIES = {
    'delayed-mlp-seed20261022/weights.pt': BASELINE_SHA,
    'delayed-mlp-seed20261022/manifest.json': '73533700a756e295876a3145017b3d263c6af0b73d962ec2bf9b7cdd9ac2d61f',
    'delayed-mlp-export/model.onnx': '3df30ae54f4bf94b34dc7b1ae296864241fab6db7e711a8970198aeefab79c23',
    'delayed-mlp-export/manifest.json': '1a438ebf5454b46deff703bd7aaddcd26f33b0c43b2dd5de6689e3fa485e4769',
    'delayed-mlp-export/weights.npz': '9d2277520dbfdbc45cb3cf3616028ce8bd2448a13360af19074172df582e08ca',
    'delayed-train-features/manifest.json': '35e92d701dcd52891c256fa99b9567c5db4d80e14f4ef71858822b470edf5a87',
    'delayed-development-features/manifest.json': '31f51e93b59f9a70c6f4b8c66cbf20b84fef900e44c75b8134aefad0cade3e8c',
    'hybrid-validation-freeze-v1/freeze.json': 'c5c4288cc04c0d7fb46154977627158336e2b1a5b8b496f8700c7a96dd25c67d',
    'hybrid-validation-v1/metrics.json': 'cfdf88d02977ee6702696726380b337f6ac1ca762bc7193cbe08040ff4fd731d',
    'hybrid-browser-runtime-v1/manifest.json': 'd7404dd13a786243d158693befd670cdf9604d17ad82a540944e7b555b4db486',
}
COUNTS = ('eligible_valid_frames', 'reference_voiced_frames', 'reference_unvoiced_frames',
    'excluded_invalid_reference', 'excluded_boundary', 'reference_outside_supported_bounds',
    'correct_frames', 'voicing_false_positive_count', 'voicing_false_negative_count',
    'unavailable_reference_voiced_count', 'voicing_false_negative_or_unavailable_count',
    'reference_voiced_success_count', 'voiced_pitch_error_count', 'unavailable_frames',
    'jointly_voiced_frames', 'octave_error_count')
RATE_PAIRS = {
    'voicing_false_positive_rate': ('voicing_false_positive_count', 'reference_unvoiced_frames'),
    'voicing_false_negative_rate': ('voicing_false_negative_count', 'reference_voiced_frames'),
    'unavailable_reference_voiced_rate': ('unavailable_reference_voiced_count', 'reference_voiced_frames'),
    'voicing_false_negative_or_unavailable_rate': ('voicing_false_negative_or_unavailable_count', 'reference_voiced_frames'),
    'reference_voiced_success_rate': ('reference_voiced_success_count', 'reference_voiced_frames'),
    'voiced_pitch_error_rate': ('voiced_pitch_error_count', 'reference_voiced_frames'),
    'jointly_voiced_coverage': ('jointly_voiced_frames', 'reference_voiced_frames'),
    'octave_error_rate_reference_voiced': ('octave_error_count', 'reference_voiced_frames'),
}
RATES = tuple(RATE_PAIRS) + ('joint_error', 'measurement_coverage', 'median_jointly_voiced_cents', 'p95_jointly_voiced_cents')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


class Bindings:
    def __init__(self, root=REPO):
        self.root = Path(root).resolve()
        self.files = {}

    def bind(self, path, expected=None):
        path = Path(path)
        path = (self.root / path).resolve() if not path.is_absolute() else path.resolve()
        require(path.is_relative_to(self.root), 'Artifact escaped repository')
        actual = sha(path)
        require(expected is None or actual == expected, 'Artifact digest mismatch: ' + str(path))
        key = str(path.relative_to(self.root))
        require(key not in self.files or self.files[key] == actual, 'Artifact changed during audit: ' + key)
        self.files[key] = actual
        return path

    def json(self, path, expected=None):
        return read_json(self.bind(path, expected))

    def npz(self, path, expected=None):
        with np.load(self.bind(path, expected), allow_pickle=False) as f:
            return {k: v.copy() for k, v in f.items()}


def independent_mlp(weights, features, candidate_mask):
    """Float32 57→32→32 equations, without importing model/forward code."""
    x, mask = np.asarray(features), np.asarray(candidate_mask)
    require(x.ndim == 2 and x.shape[1] == 57 and x.dtype == np.float32 and np.isfinite(x).all(), 'Feature contract differs')
    require(mask.shape == (len(x), 6) and mask.dtype == np.bool_, 'Candidate mask contract differs')
    shapes = {'feature_mean': (57,), 'feature_scale': (57,), 'scaling_fitted': (),
        'projection.weight': (32, 57), 'projection.bias': (32,),
        'frame_hidden.weight': (32, 32), 'frame_hidden.bias': (32,),
        'voicing_head.weight': (1, 32), 'voicing_head.bias': (1,),
        'candidate_head.weight': (6, 32), 'candidate_head.bias': (6,)}
    require(set(weights) == set(shapes), 'MLP tensor inventory differs')
    for name, shape in shapes.items():
        value = np.asarray(weights[name])
        dtype = np.bool_ if name == 'scaling_fitted' else np.float32
        require(value.shape == shape and value.dtype == dtype and np.isfinite(value).all(), 'Invalid model tensor: ' + name)
    require(bool(weights['scaling_fitted']) and (weights['feature_scale'] > 0).all(), 'Invalid frozen scaling')
    h = (x - weights['feature_mean']) / weights['feature_scale']
    for name in ('projection', 'frame_hidden'):
        # Some Accelerate BLAS versions leave spurious FP status flags; verify
        # the actual linear output before applying saturating tanh.
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            linear = h @ weights[name + '.weight'].T + weights[name + '.bias']
        require(np.isfinite(linear).all(), 'Nonfinite independent hidden linear output')
        h = np.tanh(linear)
    logits = h @ weights['voicing_head.weight'].T + weights['voicing_head.bias']
    candidates = h @ weights['candidate_head.weight'].T + weights['candidate_head.bias']
    result = {'voicing_logit': logits.astype(np.float32),
        'candidate_logits': np.where(mask, candidates, np.float32(-1e9)).astype(np.float32),
        'state': np.zeros((len(x), 32), np.float32)}
    require(all(np.isfinite(v).all() for v in result.values()), 'Independent MLP overflow')
    return result


def hybrid_arrays(voicing_logit, raw_yin_hz):
    logits = np.asarray(voicing_logit).reshape(-1)
    hz = np.asarray(raw_yin_hz, dtype=np.float64)
    require(hz.shape == logits.shape and np.isfinite(logits).all(), 'Hybrid extent/logits differ')
    voiced = logits >= 0
    valid = np.isfinite(hz) & (hz >= 50) & (hz <= 700)
    status = np.where(~voiced, 0, np.where(valid, 1, 2)).astype(np.uint8)
    return status, np.where(status == 1, hz, np.nan)


def verify_numerical_output(saved, weights, feature, *, atol=1e-5):
    expected = independent_mlp(weights, feature['features'], feature['candidate_mask'])
    maximum = {}
    for key, value in expected.items():
        actual = np.asarray(saved[key])
        require(actual.shape == value.shape and actual.dtype == value.dtype and np.isfinite(actual).all(), 'Invalid saved ' + key)
        difference = float(np.max(np.abs(actual.astype(np.float64) - value.astype(np.float64)), initial=0))
        maximum[key] = difference
        require(difference <= atol, 'NumPy output tolerance exceeded: ' + key)
    require(np.array_equal(saved['voicing_logit'] >= 0, expected['voicing_logit'] >= 0), 'Threshold decision differs')
    status, hz = hybrid_arrays(expected['voicing_logit'], saved['raw_yin_hz'])
    require(np.array_equal(saved['status'], status), 'Hybrid status differs')
    require(np.array_equal(saved['hz'], hz, equal_nan=True), 'Hybrid pitch differs')
    return maximum


def summarize(counts, errors):
    result = dict(counts)
    for key, (numerator, denominator) in RATE_PAIRS.items():
        result[key] = counts[numerator] / counts[denominator] if counts[denominator] else None
    n = counts['eligible_valid_frames']
    result['joint_error'] = (n - counts['correct_frames']) / n if n else None
    result['measurement_coverage'] = (n - counts['unavailable_frames']) / n if n else None
    ordered = sorted(errors)
    for key, q in [('median_jointly_voiced_cents', .5), ('p95_jointly_voiced_cents', .95)]:
        result[key] = ordered[math.ceil(q * len(ordered)) - 1] if ordered else None
    return result


def count_predictions(reference_rows, sample_count, predictions):
    """Full consumed PTDB reference denominator; no supported-F0 filtering."""
    rows = np.atleast_2d(reference_rows)
    require(rows.ndim == 2 and rows.shape[1] == 4, 'PTDB reference columns differ')
    by_sample = {}
    expected_targets = set(768 + 480 * np.arange(len(rows)))
    for p in predictions:
        sample, state, hz = p['sample'], p['state'], p['f0_hz']
        require(type(sample) is int and sample not in by_sample and sample >= 768 and (sample - 768) % 480 == 0, 'Duplicate/off-grid prediction')
        require(state in ('unvoiced', 'voiced', 'unavailable'), 'Unknown output state')
        require((state == 'voiced' and isinstance(hz, (int, float)) and math.isfinite(hz) and hz > 0)
            or (state != 'voiced' and hz is None), 'State/pitch mismatch')
        require(p.get('correctness_probability') is None, 'Uncalibrated scores cannot be correctness probabilities')
        by_sample[sample] = p
    counts, errors = dict.fromkeys(COUNTS, 0), []
    for index, row in enumerate(rows):
        hz, voiced, _, _ = row
        valid = np.isfinite(row).all() and ((voiced == 0 and hz == 0) or (voiced == 1 and hz > 0))
        sample = 768 + 480 * index
        if not valid:
            counts['excluded_invalid_reference'] += 1
            continue
        if sample < 3072 or sample + 1536 > sample_count:
            counts['excluded_boundary'] += 1
            continue
        p = by_sample.get(sample, {'state': 'unavailable', 'f0_hz': None})
        state = p['state']
        counts['eligible_valid_frames'] += 1
        counts['reference_voiced_frames' if voiced else 'reference_unvoiced_frames'] += 1
        counts['unavailable_frames'] += state == 'unavailable'
        if voiced:
            counts['reference_outside_supported_bounds'] += not 50 <= hz <= 700
            counts['voicing_false_negative_count'] += state == 'unvoiced'
            counts['unavailable_reference_voiced_count'] += state == 'unavailable'
            counts['voicing_false_negative_or_unavailable_count'] += state != 'voiced'
            if state == 'voiced':
                signed = 1200 * math.log2(p['f0_hz'] / hz)
                error = abs(signed)
                errors.append(error)
                counts['jointly_voiced_frames'] += 1
                counts['reference_voiced_success_count'] += error <= 50
                counts['voiced_pitch_error_count'] += error > 50
                counts['correct_frames'] += error <= 50
                octave = round(signed / 1200)
                counts['octave_error_count'] += octave != 0 and abs(signed - 1200 * octave) <= 50
        else:
            counts['voicing_false_positive_count'] += state == 'voiced'
            counts['correct_frames'] += state == 'unvoiced'
    counts = {k: int(v) for k, v in counts.items()}
    require(sum(counts[k] for k in ('voicing_false_negative_count', 'unavailable_reference_voiced_count', 'reference_voiced_success_count', 'voiced_pitch_error_count')) == counts['reference_voiced_frames'], 'Voiced error partition differs')
    metrics = summarize(counts, errors)
    metrics['predictions_outside_reference_extent'] = len(set(by_sample) - expected_targets)
    return metrics, errors


def aggregate_records(records):
    """Pool utterance counts in cells before giving every speaker/condition equal weight."""
    grouped = defaultdict(list)
    for row in records:
        grouped[row['speaker'] + '/' + row['condition']].append(row)
    cells = {}
    for name, rows in sorted(grouped.items()):
        counts = {k: sum(row['metrics'][k] for row in rows) for k in COUNTS}
        errors = [e for row in rows for e in row.get('errors', [])]
        cells[name] = summarize(counts, errors)
    def macro(names):
        chosen = [cells[k] for k in names]
        return {'cell_count': len(chosen), 'defined_cells': {k: sum(c[k] is not None for c in chosen) for k in RATES},
            **{k: sum(c[k] for c in chosen) / len(chosen) if chosen and all(c[k] is not None for c in chosen) else None for k in RATES}}
    return {'cells': cells, 'overall': macro(cells),
        'clean': macro([k for k in cells if k.endswith('/clean')]),
        'shifted': macro([k for k in cells if k.split('/')[1] in SHIFTED]),
        'original_development': macro([k for k in cells if k.split('/')[0] in ('F01', 'F03', 'M01', 'M03')]),
        'formerly_reserved': macro([k for k in cells if k.split('/')[0] in ('F05', 'F06', 'M05', 'M06')])}


def development_gates(models, analytic=None, *, speakers=SPEAKERS, require_m05=True):
    """Prespecified point-estimate gates; undefined required cells never pass."""
    analytic = analytic or {}
    names = ['hybrid_original'] + [f'{a}-{s}' for s in SEEDS for a in ('C', 'R')]
    expected = {s + '/' + c for s in speakers for c in CONDITIONS}
    issues = []
    needed = ('voicing_false_negative_or_unavailable_rate', 'voicing_false_positive_rate', 'joint_error', 'reference_voiced_success_rate')
    for name in names:
        if name not in models or set(models[name].get('cells', {})) != expected:
            issues.append(name + ': missing/extra required cells')
            continue
        for key, cell in models[name]['cells'].items():
            if any(cell.get(k) is None or not math.isfinite(cell[k]) for k in needed):
                issues.append(name + '/' + key + ': undefined required denominator')
    if issues:
        return {'status': 'unassessable', 'pass': False, 'issues': issues}
    def avg(name, metric, conditions):
        return float(np.mean([models[name]['cells'][s + '/' + c][metric] for s in speakers for c in conditions]))
    fn = 'voicing_false_negative_or_unavailable_rate'
    fp = 'voicing_false_positive_rate'
    joint = 'joint_error'
    success = 'reference_voiced_success_rate'
    h = 'hybrid_original'
    checks, effects = {}, {}
    mean_r = float(np.mean([avg(f'R-{s}', fn, SHIFTED) for s in SEEDS]))
    for comparator in ('control_mean', h):
        base = float(np.mean([avg(f'C-{s}', fn, SHIFTED) for s in SEEDS])) if comparator == 'control_mean' else avg(h, fn, SHIFTED)
        if base == 0:
            issues.append('Shifted FN baseline is zero: ' + comparator)
        checks['mean_shifted_vs_' + comparator] = base > 0 and mean_r <= .8 * base
        effects['mean_shifted_vs_' + comparator] = {'absolute_percentage_points': 100 * (base - mean_r), 'relative_reduction': (base - mean_r) / base if base else None}
    for seed in SEEDS:
        r, c = f'R-{seed}', f'C-{seed}'
        for base in (c, h):
            prefix = r + '_vs_' + base
            rfn, bfn = avg(r, fn, SHIFTED), avg(base, fn, SHIFTED)
            checks[prefix + '/shifted_nonregression'] = rfn <= bfn
            if seed == SEEDS[0]:
                if bfn == 0:
                    issues.append('Exported FN baseline is zero: ' + base)
                checks[prefix + '/exported_20pct'] = bfn > 0 and rfn <= .8 * bfn
                effects[prefix] = {'absolute_percentage_points': 100 * (bfn - rfn), 'relative_reduction': (bfn - rfn) / bfn if bfn else None}
                checks[prefix + '/exported_joint'] = avg(r, joint, CONDITIONS) < avg(base, joint, CONDITIONS)
            checks[prefix + '/macro_fp'] = avg(r, fp, CONDITIONS) - avg(base, fp, CONDITIONS) <= .01 + 1e-12
            checks[prefix + '/cell_fp'] = all(models[r]['cells'][k][fp] - models[base]['cells'][k][fp] <= .02 + 1e-12 for k in expected)
            checks[prefix + '/clean_joint'] = avg(r, joint, ('clean',)) - avg(base, joint, ('clean',)) <= .005 + 1e-12
            checks[prefix + '/clean_success'] = avg(base, success, ('clean',)) - avg(r, success, ('clean',)) <= .01 + 1e-12
            if require_m05:
                checks[prefix + '/M05_first_difference'] = models[r]['cells']['M05/first_difference'][fn] - models[base]['cells']['M05/first_difference'][fn] <= .02 + 1e-12
        result = analytic.get(r)
        checks[r + '/analytic_and_transitions'] = isinstance(result, dict) and result.get('pass') is True and result.get('status') == 'complete'
        if result is None or result.get('status') != 'complete':
            issues.append(r + ': analytic/transition evidence incomplete')
    rjoint = float(np.mean([avg(f'R-{s}', joint, CONDITIONS) for s in SEEDS]))
    cjoint = float(np.mean([avg(f'C-{s}', joint, CONDITIONS) for s in SEEDS]))
    checks['mean_joint_vs_control'] = rjoint < cjoint
    checks['mean_joint_vs_hybrid'] = rjoint < avg(h, joint, CONDITIONS)
    status = 'unassessable' if issues else 'pass' if all(checks.values()) else 'failed'
    return {'status': status, 'pass': status == 'pass', 'checks': checks, 'effects': effects, 'issues': issues,
        'interpretation': 'prespecified point-estimate development gates; no population inference',
        'export_designation': 'R-20261025'}


def validate_schedule(schedule, slots, seed):
    """Rebuild padded per-record/chunk inventory and skip rules independently."""
    require(seed in SEEDS and len(slots) > 0, 'Undeclared seed or empty slots')
    require(len(schedule['epochs']) == 12, 'Epoch count differs')
    expected_updates, epoch_updates = 0, []
    for epoch, row in enumerate(schedule['epochs']):
        order = list(range(len(slots)))
        random.Random(seed + epoch).shuffle(order)
        require(row['epoch'] == epoch + 1 and row['order'] == order, 'Paired schedule permutation differs')
        steps = []
        for offset in range(0, len(order), 64):
            indices = order[offset:offset + 64]
            length = max(slots[i]['n_frames'] for i in indices)
            for start in range(0, length, 128):
                end = min(length, start + 128)
                update = any(np.asarray(slots[i]['valid'], dtype=bool)[start:end].any() for i in indices)
                steps.append({'indices': indices, 'start': start, 'end': end, 'update': bool(update)})
        require(row['steps'] == steps, 'Padded update inventory or skip rule differs')
        count = sum(s['update'] for s in steps)
        epoch_updates.append(count)
        expected_updates += count
    return {'expected_updates': expected_updates, 'epoch_updates': epoch_updates}


def validate_fit_pair(control, treatment, control_weights, treatment_weights, original_weights, expected_updates):
    for model, weights in ((control, control_weights), (treatment, treatment_weights)):
        require(model.get('status') == 'complete' and model.get('split') == 'train' and model.get('test_or_development_labels_used') is False, 'Fit provenance incomplete')
        for key, value in {'epochs': 12, 'batch_size': 64, 'tbptt_frames': 128, 'learning_rate': .001, 'gradient_clip_norm': 1., 'measurement_delay_ms': 20, 'kind': 'framewise-mlp'}.items():
            require(model.get(key) == value, 'Fixed training contract differs: ' + key)
        require(model.get('expected_updates') == expected_updates and model.get('actual_updates') == expected_updates and sum(model['epoch_updates']) == expected_updates, 'Actual update count differs')
        for key in ('feature_mean', 'feature_scale', 'scaling_fitted'):
            require(np.array_equal(weights[key], original_weights[key]), 'Frozen original scaling differs: ' + key)
    for key in ('seed', 'plan_sha256', 'prepare_manifest_sha256', 'schedule_sha256', 'initialization_sha256', 'epoch_updates'):
        require(control.get(key) is not None and control.get(key) == treatment.get(key), 'Paired fit binding differs: ' + key)
    require(control['seed'] in SEEDS, 'Undeclared optimization seed')
    return {'pass': True, 'actual_updates_per_arm': expected_updates, 'seed': control['seed']}


def baseline(output, *, repository=REPO):
    """Authenticate consumed historical inputs and quantify fixed-YIN headroom."""
    repository = Path(repository).resolve()
    root = repository / 'research/voice-prosody'
    runs = root / 'runs'
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    audit = Bindings(repository)
    report = {'schema': 'phenometrix-voicing-robustness-baseline-audit.v1', 'status': 'running',
        'started_at': datetime.now(timezone.utc).isoformat(), 'scientific_role': 'reused-development/regression; all eight speakers exposed',
        'fitting': False, 'new_waveform_inference': False, 'source_sha256': {str(Path(__file__).relative_to(repository)): sha(__file__)}}
    write_json(output / 'audit.json', report)
    try:
        for name, expected in HISTORICAL_IDENTITIES.items():
            audit.bind(runs / name, expected)
        audit.bind(repository / 'apps/capture-web/public/.local/voice-mlp/hybrid-config.json', 'be2814796e0c7ce5a3941a3bab8adb16f9107750f25ed6df0a7ada82b768b115')
        manifests = {n: audit.json(runs / n / 'manifest.json') for n in ('hybrid-natural-v1', 'hybrid-validation-v1', 'hybrid-browser-runtime-v1')}
        source_checks = {}
        historical_source_resolution = {}
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip()
        for name, m in manifests.items():
            for path, digest in m['source_sha256'].items():
                current = sha(repository / path)
                if current == digest:
                    audit.bind(path, digest)
                else:
                    blob = subprocess.check_output(['git', 'show', revision + ':' + path], cwd=repository)
                    require(hashlib.sha256(blob).hexdigest() == digest, 'No authenticated historical source: ' + path)
                    historical_source_resolution[path] = {'historical_sha256': digest, 'git_revision': revision, 'current_sha256': current, 'current_source_validated_by_historical_run': False}
            source_checks[name] = len(m['source_sha256'])
        train = audit.json(runs / 'delayed-train-features/manifest.json')
        training = audit.json(runs / 'delayed-mlp-seed20261022/manifest.json')
        require(training['feature_hashes'] == {r['recording_id']: r['feature_sha256'] for r in train['recordings']}, 'Original training features differ')
        require(len(train['recordings']) == 960 and len({r['source_recording_id'] for r in train['recordings']}) == 240 and len({r['speaker'] for r in train['recordings']}) == 12, 'Original TRAIN membership differs')
        for row in train['recordings']:
            audit.bind(runs / 'delayed-train-features/features' / (row['recording_id'] + '.npz'), row['feature_sha256'])
        natural, reserved = manifests['hybrid-natural-v1'], manifests['hybrid-validation-v1']
        development_features = audit.json(runs / 'delayed-development-features/manifest.json')
        require(natural['feature_manifest_sha256'] == sha(runs / 'delayed-development-features/manifest.json'), 'Natural feature manifest differs')
        acquisition = audit.json(runs / 'hybrid-natural-v1/provenance/acquisition-manifest.json', development_features['data_manifest_sha256'])
        reserved_acquisition = audit.json(runs / 'hybrid-validation-v1/provenance/scoring-acquisition-manifest.json', reserved['scoring_acquisition_manifest_sha256'])
        refs = {r['recording_id']: r for r in acquisition['files'] + reserved_acquisition['files'] if r['role'] == 'ref'}
        mic = {r['recording_id']: r for r in acquisition['files'] + reserved_acquisition['files'] if r['role'] == 'mic'}
        models = defaultdict(list)
        source_reference_cache = {}
        details = []
        for cohort, manifest in [('original_development', natural), ('formerly_reserved', reserved)]:
            require(len(manifest['recordings']) == (320 if cohort == 'original_development' else 640), 'Historical evaluation extent differs')
            for row in manifest['recordings']:
                identity, source = row['recording_id'], row['source_recording_id']
                if source not in source_reference_cache:
                    source_reference_cache[source] = np.loadtxt(audit.bind(root / 'data/ptdb-tug' / refs[source]['path'], refs[source]['sha256']))
                    audit.bind(root / 'data/ptdb-tug' / mic[source]['path'], mic[source]['sha256'])
                reference = source_reference_cache[source]
                if cohort == 'original_development':
                    audit.bind(runs / 'delayed-development-features/features' / (identity + '.npz'), row['feature_sha256'])
                    raw = audit.json(runs / 'delayed-development-features/predictions' / (identity + '.json'), row['prediction_sha256'])['yin_evidence']
                    path = audit.bind(runs / 'hybrid-natural-v1' / row['hybrid_prediction_path'], row['hybrid_prediction_sha256'])
                else:
                    audit.bind(runs / 'hybrid-validation-v1' / row['feature_path'], row['feature_sha256'])
                    path = audit.bind(runs / 'hybrid-validation-v1' / row['prediction_path'], row['prediction_sha256'])
                payload = json.loads(gzip.decompress(path.read_bytes()))
                if cohort == 'formerly_reserved':
                    raw = payload['yin_evidence']
                oracle = []
                for y in raw:
                    target = y['sample'] - 960
                    index = (target - 768) // 480
                    require((target - 768) % 480 == 0 and index >= 0, 'Raw YIN/reference timing differs')
                    if index >= len(reference):
                        continue
                    r = reference[index]
                    voiced = r[1] == 1 and r[0] > 0
                    hz = y['raw_f0_hz']
                    valid = hz is not None and math.isfinite(hz) and 50 <= hz <= 700
                    state = 'unvoiced' if not voiced else 'voiced' if valid else 'unavailable'
                    oracle.append({'sample': int(target), 'state': state, 'f0_hz': hz if state == 'voiced' else None})
                methods = {k: v for k, v in payload['predictions'].items() if k in ('dsp', 'yin', 'continuity', 'mlp_original', 'hybrid_original')}
                methods['reference_voicing_fixed_raw_yin_oracle'] = oracle
                for method, predictions in methods.items():
                    counts, errors = count_predictions(reference, row['sample_count'], predictions)
                    models[method].append({'speaker': row['speaker'], 'condition': row['condition'], 'metrics': counts, 'errors': errors})
                    if method in ('hybrid_original', 'reference_voicing_fixed_raw_yin_oracle'):
                        details.append({'recording_id': identity, 'cohort': cohort, 'method': method, 'speaker': row['speaker'], 'condition': row['condition'], 'metrics': counts})
        aggregated = {name: aggregate_records(rows) for name, rows in models.items()}
        baseline_fn = aggregated['hybrid_original']['shifted']['voicing_false_negative_or_unavailable_rate']
        oracle_fn = aggregated['reference_voicing_fixed_raw_yin_oracle']['shifted']['voicing_false_negative_or_unavailable_rate']
        write_json(output / 'records.json', details)
        report.update(status='complete', completed_at=datetime.now(timezone.utc).isoformat(), source_binding_counts=source_checks, historical_source_resolution=historical_source_resolution,
            source_count=len(source_reference_cache), condition_record_count=len(models['hybrid_original']), historical_identities=HISTORICAL_IDENTITIES,
            methods=aggregated, records_sha256=sha(output / 'records.json'), bound_files=audit.files,
            oracle_interpretation='reference voicing replaces only the voicing decision; unchanged saved raw YIN and common denominator; unattainable label-informed upper bound, not a candidate',
            shifted_fn_headroom={'baseline': baseline_fn, 'oracle': oracle_fn, 'required_maximum': .8 * baseline_fn,
                'fixed_yin_makes_20pct_reduction_possible': oracle_fn <= .8 * baseline_fn})
        write_json(output / 'audit.json', report)
        return report
    except BaseException as error:
        report.update(status='failed', failure=repr(error), bound_files=audit.files)
        write_json(output / 'audit.json', report)
        raise


def independent_labels(feature, reference):
    """Reconstruct exact PTDB labels without the training alignment/label helpers."""
    x = feature['features']; count = int(feature['pcm_sample_count'])
    end = np.asarray(feature['reference_sample'])
    require(np.array_equal(end, np.arange(2208, count + 1, 480)), 'Feature availability grid differs')
    require(len(x) == len(end) and int(feature['sample_rate_hz']) == 48000, 'Feature sample count differs')
    require(x.dtype == np.float32 and x.shape == (len(end), 57) and np.isfinite(x).all(), 'Invalid training features')
    hz, mask = feature['candidate_hz'], feature['candidate_mask']
    require(hz.shape == mask.shape == (len(end), 6) and mask.dtype == np.bool_, 'Candidate shape differs')
    require(np.isnan(hz[~mask]).all() and np.isfinite(hz[mask]).all() and ((hz[mask] >= 50) & (hz[mask] <= 700)).all(), 'Candidate validity differs')
    require(all((np.diff(row[active]) > 0).all() for row, active in zip(hz, mask)), 'Candidate ordering differs')
    voice = np.zeros(len(end), np.float32); valid = np.zeros(len(end), bool); target = np.full(len(end), -1, np.int64)
    for i, sample in enumerate(end - 960):
        index = (int(sample) - 768) // 480
        if sample < 3072 or sample + 1536 > count or index < 0 or index >= len(reference):
            continue
        row = reference[index]; f0, voiced = row[:2]
        if not np.isfinite(row).all() or not ((voiced == 0 and f0 == 0) or (voiced == 1 and f0 > 0)):
            continue
        valid[i] = True; voice[i] = voiced
        if voiced:
            error = np.full(6, np.inf)
            error[mask[i]] = np.abs(1200 * np.log2(hz[i][mask[i]] / f0))
            best = int(np.argmin(error))
            if error[best] <= 50:
                target[i] = best
    return {'voiced': voice, 'valid': valid, 'target': target}


def tensor_digest(weights):
    digest = hashlib.sha256()
    for key, value in sorted(weights.items()):
        value = np.asarray(value)
        digest.update(key.encode() + str(value.dtype).encode() + str(value.shape).encode() + value.tobytes())
    return digest.hexdigest()


def checkpoint(audit, path, digest):
    import torch
    saved = torch.load(audit.bind(path, digest), map_location='cpu', weights_only=True)
    require(saved['config'] == {'kind': 'framewise-mlp', 'projection_size': 32, 'hidden_size': 32, 'feature_version': 'v2', 'feature_count': 57, 'candidate_count': 6}, 'Checkpoint architecture differs')
    return {k: v.detach().cpu().numpy().copy() for k, v in saved['state_dict'].items()}


def independent_initialization(seed, original):
    import torch
    torch.manual_seed(seed)
    values = {k: v.copy() for k, v in original.items() if k in ('feature_mean', 'feature_scale', 'scaling_fitted')}
    for name, n_in, n_out in [('projection', 57, 32), ('frame_hidden', 32, 32), ('voicing_head', 32, 1), ('candidate_head', 32, 6)]:
        layer = torch.nn.Linear(n_in, n_out)
        values[name + '.weight'] = layer.weight.detach().numpy().copy()
        values[name + '.bias'] = layer.bias.detach().numpy().copy()
    return values


def independent_bank_parameters(source, slot):
    namespace = 'phenometrix-voicing-robustness-v1-20261025'
    require(slot in range(4), 'Unknown continuous view slot')
    def seed(value):
        return int.from_bytes(hashlib.sha256(f'{namespace}:{source}:{value}'.encode()).digest()[:8], 'little')
    value = seed(slot); rng = np.random.default_rng(value)
    p = {'slot': slot, 'seed': value, 'gain_db': float(rng.uniform(-18, 0)), 'filter': 'none', 'noise': 'none'}
    family = None if slot == 0 else 'emphasis' if slot == 1 else 'lowpass' if slot == 2 else ('emphasis' if rng.random() < .5 else 'lowpass')
    if family == 'emphasis':
        p.update(filter=family, a=float(rng.uniform(.2, 1.)))
    elif family == 'lowpass':
        cutoff = float(np.exp(rng.uniform(np.log(1500.), np.log(10000.))))
        p.update(filter=family, cutoff_hz=cutoff, b=float(np.exp(-2 * np.pi * cutoff / 48000)))
    if slot == 3:
        p.update(noise='white' if rng.random() < .5 else 'ar1', snr_db=float(rng.uniform(5, 25)), ar_coefficient=.8)
    p['noise_seed'] = seed(f'noise-{slot}')
    return p


def independent_construct(pcm, parameters):
    """Full-record numeric reconstruction; no construction-module dependency."""
    from scipy.signal import lfilter
    x = np.asarray(pcm, dtype=np.float64)
    p = parameters
    require(x.ndim == 1 and len(x) and np.isfinite(x).all(), 'Invalid microphone PCM')
    if p['filter'] == 'none':
        y = x.copy()
    elif p['filter'] == 'emphasis':
        a = p['a']
        # Separate products preserve the declared Float64 causal-filter arithmetic.
        y = x * (1 / (1 + a))
        y[1:] += x[:-1] * (-a / (1 + a))
    elif p['filter'] == 'lowpass':
        b = p['b']; y = lfilter([1 - b], [1, -b], x)
    else:
        raise ValueError('Unknown channel filter')
    y *= 10 ** (p['gain_db'] / 20)
    power = float(np.mean(y * y)); noise = np.zeros(len(y), np.float64); scale = 0.
    if p['noise'] != 'none':
        noise = np.random.default_rng(p['noise_seed']).standard_normal(len(y))
        if p['noise'] == 'ar1':
            noise = lfilter([math.sqrt(1 - .8 ** 2)], [1, -.8], noise)
        else:
            require(p['noise'] == 'white', 'Unknown noise family')
        if power:
            scale = math.sqrt(power / (10 ** (p['snr_db'] / 10) * float(np.mean(noise * noise))))
        noise *= scale
    final = (y + noise).astype('<f4')
    require(np.isfinite(final).all(), 'Invalid constructed PCM')
    actual_noise = final.astype(np.float64) - y
    achieved = float(10 * np.log10(power / np.mean(actual_noise * actual_noise))) if power and p['noise'] != 'none' else None
    return final, {'signal_power_before_noise': power, 'noise_scale': scale, 'requested_snr_db': p.get('snr_db'), 'achieved_snr_db': achieved}


def read_pcm16(path):
    import wave
    with wave.open(str(path), 'rb') as source:
        require((source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype()) == (1, 2, 48000, 'NONE'), 'Source PCM format differs')
        return np.frombuffer(source.readframes(source.getnframes()), dtype='<i2').astype(np.float32) / np.float32(32768)


def audit_preparation(audit, run, plan):
    prep = audit.json(run / 'prepare/manifest.json')
    require(prep.get('status') == 'complete' and prep.get('train_only') is True and prep.get('development_or_external_inputs_used') is False, 'Preparation incomplete or contaminated')
    audit.bind(prep['plan']['path'], prep['plan']['sha256'])
    require(prep['plan']['sha256'] == sha(run / 'plan/plan.json') and prep['recipe_sha256'] == plan['recipe_sha256'], 'Preparation plan/recipe differs')
    train = audit.json(ROOT / 'runs/delayed-train-features/manifest.json', HISTORICAL_IDENTITIES['delayed-train-features/manifest.json'])
    acq = audit.json(ROOT / 'runs/delayed-train-features/acquisition-manifest.json', train['data_manifest_sha256'])
    refs = {r['recording_id']: r for r in acq['files'] if r['role'] == 'ref' and r['split'] == 'train'}
    mics = {r['recording_id']: r for r in acq['files'] if r['role'] == 'mic' and r['split'] == 'train'}
    anchors = {(r['source_recording_id'], r['condition']): r for r in train['recordings']}
    sources = sorted({r['source_recording_id'] for r in train['recordings']})
    require(len(sources) == 240 and len(prep['records']) == 1920, 'Training source/view inventory differs')
    records, ref_cache = [], {}
    microphone_source, microphone_pcm = None, None
    for index, row in enumerate(prep['records']):
        source, slot = sources[index // 8], index % 8
        require(row['source_id'] == source and row['slot'] == slot and row['id'] == f'{source}__slot{slot}', 'Record order or source membership differs')
        require(row['speaker'] == mics[source]['speaker'] == refs[source]['speaker'], 'Training speaker differs')
        for role, lookup in [('reference', refs), ('microphone', mics)]:
            wanted = ROOT / 'data/ptdb-tug' / lookup[source]['path']
            require((REPO / row[role]['path']).resolve() == wanted.resolve() and row[role]['sha256'] == lookup[source]['sha256'], 'Input acquisition binding differs')
            audit.bind(row[role]['path'], row[role]['sha256'])
        feature = audit.npz(row['feature']['path'], row['feature']['sha256'])
        saved = audit.npz(row['labels']['path'], row['labels']['sha256'])
        if source not in ref_cache:
            ref_cache[source] = np.loadtxt(REPO / row['reference']['path'])
        labels = independent_labels(feature, ref_cache[source])
        for key in labels:
            require(saved[key].dtype == labels[key].dtype and np.array_equal(saved[key], labels[key]), 'Independent training label differs: ' + key)
        require(row['n_frames'] == len(labels['valid']) and row['valid_frames'] == int(labels['valid'].sum()) and row['covered_candidate_frames'] == int((labels['valid'] & (labels['target'] >= 0)).sum()), 'Training count differs')
        if slot < 4:
            anchor = anchors[source, CONDITIONS[slot]]
            require(row['feature']['sha256'] == anchor['feature_sha256'] and row['construction']['transformed_pcm_sha256'] == anchor['transformed_pcm_sha256'], 'Anchor identity differs')
        else:
            construction = row['construction']; pcm = construction['pcm']
            parameters = independent_bank_parameters(source, slot - 4)
            require(construction['parameters'] == parameters, 'Label-independent bank draw differs')
            if microphone_source != source:
                microphone_source = source; microphone_pcm = read_pcm16(REPO / row['microphone']['path'])
            expected_pcm, expected_receipt = independent_construct(microphone_pcm, parameters)
            require(hashlib.sha256(expected_pcm.tobytes()).hexdigest() == pcm['sha256'], 'Independently constructed PCM differs')
            compare_values(construction, expected_receipt, 'Construction statistics')
            audit.bind(pcm['path'], pcm['sha256'])
            require(pcm['sha256'] == construction['pcm_sha256'] == construction['transformed_pcm_sha256'], 'Transformed PCM binding differs')
            require(construction['sample_count'] == int(feature['pcm_sample_count']) and construction['stored_dtype'] == 'little-endian float32' and construction['construction_dtype'] == 'float64' and construction['saturation'] is False, 'PCM construction convention differs')
            data = np.fromfile(REPO / pcm['path'], dtype='<f4')
            require(len(data) == construction['sample_count'] and np.isfinite(data).all() and float(data.min()) == construction['minimum'] and float(data.max()) == construction['maximum'] and int((np.abs(data) > 1).sum()) == construction['outside_unit_count'], 'Transformed PCM content receipt differs')
        records.append({'n_frames': row['n_frames'], 'valid': labels['valid'], 'voiced': labels['voiced'], 'target': labels['target']})
    expected_r = list(range(1920)); expected_c = [8 * i + s % 4 for i in range(240) for s in range(8)]
    require(prep['slots'] == {'C': expected_c, 'R': expected_r}, 'Matched source/view exposure differs')
    for c, r in zip(expected_c, expected_r):
        require(records[c]['n_frames'] == records[r]['n_frames'] and np.array_equal(records[c]['valid'], records[r]['valid']) and np.array_equal(records[c]['voiced'], records[r]['voiced']), 'Paired supervision or length differs')
    schedules = {}
    for seed in SEEDS:
        binding = prep['schedules'][str(seed)]
        schedule = audit.json(binding['path'], binding['sha256'])
        schedules[seed] = validate_schedule(schedule, [records[i] for i in expected_r], seed)
        require(schedule['expected_updates'] == schedules[seed]['expected_updates'], 'Schedule declared update count differs')
    return prep, schedules, {'records': 1920, 'sources': 240, 'slots_per_arm': 1920, 'schedules': schedules,
        'covered_candidate_frames': {arm: sum(int((records[i]['valid'] & (records[i]['target'] >= 0)).sum()) for i in prep['slots'][arm]) for arm in ('C', 'R')},
        'valid_supervision_frames': {arm: sum(int(records[i]['valid'].sum()) for i in prep['slots'][arm]) for arm in ('C', 'R')}}


def voicing_event_metrics(target_sample, status, hz, fundamental):
    """Fixed onset/offset search, signed target lag and separate availability lag."""
    sample, state, hz = np.asarray(target_sample), np.asarray(status), np.asarray(hz)
    require(sample.ndim == 1 and state.shape == hz.shape == sample.shape and np.issubdtype(sample.dtype, np.integer), 'Event output extent differs')
    require(len(sample) >= 3 and (np.diff(sample) == 480).all() and np.isin(state, [0, 1, 2]).all(), 'Event grid/state differs')
    delays = {}
    for name, event, wanted in [('onset', 48000, 1), ('offset', 96000, 0)]:
        found = None
        for i in range(len(sample) - 2):
            if sample[i] >= event - 960 and sample[i + 2] <= event + 9600 and np.all(state[i:i + 3] == wanted):
                found = int(sample[i]); break
        delays[name] = {'detected': found is not None, 'target_sample': found,
            'signed_target_delay_ms': (found - event) / 48 if found is not None else None,
            'earliest_availability_delay_ms': (found + 960 - event) / 48 if found is not None else None}
    plateau = {}
    for name, lo, hi in [('pre', 9600, 38400), ('post', 105600, 134400), ('voiced', 57600, 86400)]:
        active = (sample >= lo) & (sample <= hi)
        n = int(active.sum()); require(n > 0, 'Missing event plateau')
        if name == 'voiced':
            good = active & (state == 1) & np.isfinite(hz) & (hz > 0)
            errors = np.full(len(hz), np.inf)
            errors[good] = np.abs(1200 * np.log2(hz[good] / fundamental))
            count = int((active & (errors <= 50)).sum())
            plateau[name] = {'frames': n, 'voiced_success_count': count, 'voiced_success_rate': count / n, 'pass': count / n >= .95}
        else:
            count = int((active & (state == 1)).sum())
            plateau[name] = {'frames': n, 'false_voiced_count': count, 'false_voiced_rate': count / n, 'pass': count / n <= .05}
    return {'events': delays, 'plateaux': plateau, 'pass': all(v['detected'] for v in delays.values()) and all(v['pass'] for v in plateau.values())}


def voicing_event_gate(current, baseline):
    require(set(current) == set(baseline) and len(current) == 12, 'Twelve paired event cases required')
    checks = {name: bool(value['pass']) for name, value in current.items()}
    comparison = {}
    for event in ('onset', 'offset'):
        names = [name for name in current if baseline[name]['events'][event]['detected']]
        current_values = [current[name]['events'][event]['signed_target_delay_ms'] for name in names]
        baseline_values = [baseline[name]['events'][event]['signed_target_delay_ms'] for name in names]
        defined = bool(names) and all(value is not None for value in current_values)
        difference = float(np.median(current_values) - np.median(baseline_values)) if defined else None
        comparison[event] = {'paired_baseline_detected_cases': len(names), 'baseline_missing_cases': [n for n in baseline if n not in names], 'extra_median_signed_target_delay_ms': difference}
        checks[event + '_relative_delay'] = not names or (defined and difference <= 10 + 1e-12)
    return {'status': 'complete', 'pass': all(checks.values()), 'checks': checks, 'delay_comparison': comparison}


def compare_values(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict), label + ': object missing')
        for key, value in expected.items():
            require(key in actual, label + ': missing ' + key)
            compare_values(actual[key], value, label + '/' + key)
    elif isinstance(expected, (float, np.floating)):
        require(isinstance(actual, (float, int)) and math.isfinite(actual) and math.isclose(float(actual), float(expected), abs_tol=1e-12, rel_tol=1e-12), label + ': number differs')
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': list extent differs')
        for i, (a, e) in enumerate(zip(actual, expected)):
            compare_values(a, e, label + '/' + str(i))
    else:
        require(actual == expected, label + ': value differs')


def independent_frequency(definition, seconds, original):
    if original:
        first = definition['initialHz']
        if definition['kind'] in ('constant', 'observation-change'):
            return float(first)
        if definition['kind'] == 'step':
            return float(first if seconds < definition['changeStartSeconds'] else definition['finalHz'])
        elapsed = min(definition['changeEndSeconds'] - definition['changeStartSeconds'], max(0., seconds - definition['changeStartSeconds']))
        return float(first + definition['slopeHzPerSecond'] * elapsed)
    if not definition['latent_voiced']:
        return 0.
    first, last = definition['initial_hz'], definition['final_hz']
    if definition['kind'] == 'step':
        return float(first if seconds < definition['change_start_seconds'] else last)
    if definition['kind'] == 'glide':
        t = min(1., max(0., (seconds - definition['change_start_seconds']) / (definition['change_end_seconds'] - definition['change_start_seconds'])))
        return float(first + (last - first) * t)
    return float(first)


def original_analytic_case(definition, sample, status, hz, count):
    eligible = (sample >= 3072) & (sample + 1536 <= count)
    sample, status, hz = sample[eligible], status[eligible], hz[eligible]
    truth = np.array([independent_frequency(definition, t / 48000, True) for t in sample])
    voiced = status == 1
    error = np.full(len(sample), np.inf)
    error[voiced] = np.abs(1200 * np.log2(hz[voiced] / truth[voiced]))
    good = error <= 50
    kind = definition['kind']; result = {'kind': kind}
    if kind == 'constant':
        result['success'] = float(good.mean()) if len(good) else None
    elif kind == 'observation-change':
        after = sample - 960 >= .6 * 48000
        result['success'] = float(good[after].mean()) if after.any() else None
    elif kind == 'step':
        available = sample + 960 >= definition['changeStartSeconds'] * 48000
        selected = np.flatnonzero(available)
        value = None
        for j in range(len(selected) - 2):
            ii = selected[j:j + 3]
            correct = (status[ii] == 1) & (np.abs(1200 * np.log2(np.where(status[ii] == 1, hz[ii], 1.) / definition['finalHz'])) <= 50)
            if correct.all() and sample[ii[-1]] - sample[ii[0]] == 960:
                value = float(sample[ii[0]] / 48 - definition['changeStartSeconds'] * 1000); break
        result['delay_ms'] = value
    else:
        inside = (sample - 960 >= definition['changeStartSeconds'] * 48000) & (sample + 960 < definition['changeEndSeconds'] * 48000) & voiced
        values = (truth[inside] - hz[inside]) / definition['slopeHzPerSecond'] * 1000
        result['delay_ms'] = float(np.median(values)) if len(values) else None
    return result


def independent_analytic_gate(cases, originals, dsp_cases):
    stationary = {key: value['success'] for key, value in originals.items() if value['kind'] == 'constant'}
    post = {key: value['success'] for key, value in originals.items() if value['kind'] == 'observation-change'}
    checks = {'stationary': len(stationary) == 3 and all(v is not None and v >= .95 for v in stationary.values()),
        'postchange': len(post) == 3 and all(v is not None and v >= .95 for v in post.values())}
    differences = {}
    keys = ('median_signed_target_delay_ms', 'median_step_target_settling_lag_ms', 'median_glide_equivalent_delay_ms')
    for key, kinds, n in zip(keys, [('step', 'glide'), ('step',), ('glide',)], [8, 2, 6]):
        names = [name for name, row in originals.items() if row['kind'] in kinds]
        current = [originals[name]['delay_ms'] for name in names]
        baseline = [dsp_cases[name]['delay_ms'] for name in names]
        defined = len(names) == n and all(v is not None for v in current + baseline)
        differences[key] = float(np.median(current) - np.median(baseline)) if defined else None
    checks['transition'] = all(value is not None and value <= 10 for value in differences.values())
    challenge_details = {}
    for cohort, key in [('previous32-regression', 'old_challenge'), ('fresh32', 'challenge')]:
        rows = [row['metrics'] for row in cases if row['cohort'] == cohort]
        voiced = [r for r in rows if r['reference_voiced_frames'] > 0]
        unvoiced = [r for r in rows if r['reference_unvoiced_frames'] > 0]
        successes = [r['reference_voiced_success_rate'] for r in voiced]
        checks[key] = len(voiced) == 28 and len(unvoiced) == 4 and all(v is not None and v >= .8 for v in successes) and float(np.mean(successes)) >= .9 and all(r['voicing_false_positive_rate'] is not None and r['voicing_false_positive_rate'] <= .05 for r in unvoiced)
        challenge_details[cohort] = {'voiced_cases': len(voiced), 'unvoiced_cases': len(unvoiced), 'macro_voiced_success': float(np.mean(successes)) if successes else None}
    return {'status': 'complete', 'pass': all(checks.values()), 'checks': checks, 'stationary': stationary, 'postchange': post,
        'transition_extra_median_ms': differences, 'challenge': challenge_details}


def audit_evaluations(audit, run, checkpoints):
    names = ['hybrid_original'] + [f'{arm}-{seed}' for seed in SEEDS for arm in ('C', 'R')]
    checkpoints = {'hybrid_original': checkpoint(audit, ROOT / 'runs/delayed-mlp-seed20261022/weights.pt', BASELINE_SHA), **checkpoints}
    expected_checkpoints = {name: BASELINE_SHA if name == 'hybrid_original' else sha(run / 'fits' / name / 'weights.pt') for name in names}
    natural, analytic, event_cases, summaries = {}, {}, {}, {}
    max_error = dict.fromkeys(('voicing_logit', 'candidate_logits', 'state'), 0.)
    verified_pairs, ref_cache = 0, {}
    expected_primary = {}
    for source_folder in ('delayed-development-features', 'hybrid-validation-v1'):
        historical = audit.json(ROOT / 'runs' / source_folder / 'manifest.json')
        for row in historical['recordings']:
            expected_primary[row['recording_id']] = row
    execution = audit.json(run / 'execution-freeze.json')
    diagnostic = audit.json(run / 'diagnostic-inputs/manifest.json')
    require(diagnostic['status'] == 'complete' and diagnostic['references_opened'] is False and diagnostic['external_corpus_used'] is False, 'Diagnostic input contract differs')
    declaration = audit.json(run / 'diagnostic-inputs/declaration.json', diagnostic['declaration_sha256'])
    require(diagnostic['plan_sha256'] == sha(run / 'plan/plan.json') and declaration['for_fitting'] is False, 'Diagnostic declaration differs')
    for path, digest in diagnostic['source_sha256'].items():
        audit.bind(run / 'diagnostic-inputs/source-snapshot' / path, digest)
    diagnostic_rows = {r['id']: r for r in diagnostic['records']}
    old_analytic = audit.json(ROOT / 'runs/hybrid-synthetic-v1/manifest.json')
    old_analytic_rows = {r['id']: r for r in old_analytic['records']}
    for group, expected_rows in [('development', 960), ('diagnostics', 1440), ('analytic', 78), ('events', 12)]:
        folder = run / group
        final = audit.json(folder / 'manifest.json')
        require(final['status'] == 'complete' and final['group'] == group, 'Evaluation group incomplete: ' + group)
        frozen = audit.json(folder / 'prediction-freeze.json')
        require(frozen['status'] == 'predictions_frozen' and frozen['reference_values_read_before_prediction_freeze'] is False and frozen['records'] == final['records'], 'Prediction freeze differs: ' + group)
        require(frozen['models'] == final['models'] and set(final['models']) == set(names), 'Evaluated model inventory differs')
        audit.bind(frozen['plan']['path'], frozen['plan']['sha256'])
        require(frozen['plan']['sha256'] == sha(run / 'plan/plan.json'), 'Evaluation plan differs')
        require(len(final['records']) == expected_rows and len({r['id'] for r in final['records']}) == expected_rows, 'Evaluation row inventory differs')
        for path, digest in frozen['source_sha256'].items():
            audit.bind(folder / 'source-snapshot' / path, digest)
            if path in execution['source_sha256']:
                require(digest == execution['source_sha256'][path], 'Evaluation source differs from prefit execution freeze')
        for name, row in frozen['models'].items():
            audit.bind(row['manifest']['path'], row['manifest']['sha256'])
            audit.bind(row['checkpoint']['path'], row['checkpoint']['sha256'])
            require(row['checkpoint']['sha256'] == expected_checkpoints[name], 'Evaluated checkpoint differs')
        saved_metrics = audit.json(final['metrics']['path'], final['metrics']['sha256'])
        audit.bind(saved_metrics['prediction_freeze']['path'], saved_metrics['prediction_freeze']['sha256'])
        require(saved_metrics['prediction_freeze']['sha256'] == sha(folder / 'prediction-freeze.json'), 'Scored prediction freeze differs')
        records = defaultdict(list); original_cases = defaultdict(dict); dsp_cases = {}; cases = defaultdict(list)
        scored = {(r['id'], r['model']): r['metrics'] for r in saved_metrics.get('records', []) if 'model' in r}
        analytic_scored = {r['id']: r['metrics'] for r in saved_metrics.get('records', []) if 'model' not in r}
        for row in frozen['records']:
            if group == 'development':
                require(row['id'] in expected_primary, 'Unexpected primary source')
                historical = expected_primary[row['id']]
                require(row['source_id'] == historical['source_recording_id'] and row['speaker'] == historical['speaker'] and row['condition'] == historical['condition'] and row['sample_count'] == historical['sample_count'] and row['feature']['sha256'] == historical['feature_sha256'] and row['baseline']['sha256'] == historical['prediction_sha256'], 'Primary historical input binding differs')
            elif group in ('diagnostics', 'events'):
                require(row['id'] in diagnostic_rows, 'Unexpected diagnostic/event source')
                original = diagnostic_rows[row['id']]
                require({k: v for k, v in row.items() if k != 'predictions'} == original, 'Diagnostic source declaration differs')
                audit.bind(run / 'diagnostic-inputs' / original['pcm_path'], original['pcm_sha256'])
            else:
                require(row['id'] in old_analytic_rows, 'Unexpected analytic source')
                original = old_analytic_rows[row['id']]
                parameter_bytes = json.dumps(row['definition'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
                require(row['cohort'] == original['cohort'] and row['feature']['sha256'] == original['feature_sha256'] and row['baseline']['sha256'] == original['prediction_sha256'] and hashlib.sha256(parameter_bytes).hexdigest() == original['parameters_sha256'], 'Historical analytic binding differs')
            feature = audit.npz(row['feature']['path'], row['feature']['sha256'])
            raw_path = audit.bind(row['baseline']['path'], row['baseline']['sha256'])
            baseline = json.loads(gzip.decompress(raw_path.read_bytes())) if raw_path.name.endswith('.gz') else read_json(raw_path)
            raw_yin = np.array([r['raw_f0_hz'] for r in baseline['yin_evidence']], np.float64)
            require(np.array_equal([r['sample'] for r in baseline['yin_evidence']], feature['reference_sample']), 'Raw YIN support grid differs')
            predictions = {}
            for name in names:
                saved = audit.npz(row['predictions'][name]['path'], row['predictions'][name]['sha256'])
                require(np.array_equal(saved['raw_yin_hz'], raw_yin, equal_nan=True), 'Saved raw YIN differs from bound source')
                require(np.array_equal(saved['target_sample'], feature['reference_sample'] - 960), 'Prediction target timing differs')
                deltas = verify_numerical_output(saved, checkpoints[name], feature)
                max_error = {k: max(max_error[k], deltas[k]) for k in max_error}
                verified_pairs += len(saved['target_sample'])
                predictions[name] = saved
            if group in ('development', 'diagnostics'):
                ref_path = audit.bind(row['reference']['path'], row['reference']['sha256'])
                if str(ref_path) not in ref_cache:
                    ref_cache[str(ref_path)] = np.loadtxt(ref_path)
                reference = ref_cache[str(ref_path)]
                all_predictions = {name: [{'sample': int(t), 'state': ('unvoiced', 'voiced', 'unavailable')[int(s)], 'f0_hz': float(hz) if s == 1 else None} for t, s, hz in zip(p['target_sample'], p['status'], p['hz'])] for name, p in predictions.items()}
                if group == 'development':
                    path = audit.bind(row['original_predictions']['path'], row['original_predictions']['sha256'])
                    old = json.loads(gzip.decompress(path.read_bytes()))['predictions']
                    all_predictions.update({k: old[k] for k in ('dsp', 'yin', 'continuity', 'mlp_original')})
                for name, payload in all_predictions.items():
                    metric, errors = count_predictions(reference, row['sample_count'], payload)
                    compare_values(scored[row['id'], name], metric, group + '/' + row['id'] + '/' + name)
                    records[name].append({'speaker': row['speaker'], 'condition': row['condition'], 'metrics': metric, 'errors': errors})
            elif group == 'events':
                for name, saved in predictions.items():
                    case = voicing_event_metrics(saved['target_sample'], saved['status'], saved['hz'], row['definition']['frequency_hz'])
                    event_cases.setdefault(name, {})[row['id']] = case
                    reported = {'events': {event: {'detected': value['detected'], 'target_lag_ms': value['signed_target_delay_ms'], 'earliest_availability_delay_ms': value['earliest_availability_delay_ms']} for event, value in case['events'].items()}, 'unvoiced_plateau_false_voicing': {label: case['plateaux'][label]['false_voiced_rate'] for label in ('pre', 'post')}, 'voiced_success_rate': case['plateaux']['voiced']['voiced_success_rate'], 'plateau_pass': all(value['pass'] for value in case['plateaux'].values())}
                    compare_values(analytic_scored[row['id']][name], reported, 'Event metrics/' + row['id'] + '/' + name)
            else:
                definition = row['definition']; original = row['cohort'] == 'original14'
                targets = np.arange(768, row['sample_count'] + 1, 480)
                frequencies = [independent_frequency(definition, t / 48000, original) for t in targets]
                reference = np.array([[f, float(f > 0), 0., 0.] for f in frequencies])
                for name, saved in predictions.items():
                    payload = [{'sample': int(t), 'state': ('unvoiced', 'voiced', 'unavailable')[int(s)], 'f0_hz': float(hz) if s == 1 else None} for t, s, hz in zip(saved['target_sample'], saved['status'], saved['hz'])]
                    metric, _ = count_predictions(reference, row['sample_count'], payload)
                    # Producer's historical scorer lacks the added decomposition fields.
                    keys = ('eligible_valid_frames', 'correct_frames', 'reference_voiced_frames', 'reference_unvoiced_frames', 'voicing_false_positive_count', 'reference_voiced_success_rate', 'voicing_false_positive_rate', 'joint_error')
                    compare_values(analytic_scored[row['id']][name], {k: metric[k] for k in keys}, 'analytic/' + row['id'] + '/' + name)
                    cases[name].append({'cohort': row['cohort'], 'metrics': metric})
                    if original:
                        original_cases[name][row['id']] = original_analytic_case(definition, saved['target_sample'], saved['status'], saved['hz'], row['sample_count'])
                if original:
                    rows = baseline['predictions']['dsp']; sample = np.array([r['sample'] for r in rows])
                    state = np.array([('unvoiced', 'voiced', 'unavailable').index(r['state']) for r in rows])
                    hz = np.array([r['f0_hz'] if r['f0_hz'] is not None else np.nan for r in rows])
                    dsp_cases[row['id']] = original_analytic_case(definition, sample, state, hz, row['sample_count'])
        if group in ('development', 'diagnostics'):
            reconstructed = {name: aggregate_records(rows) for name, rows in records.items()}
            compare_values(saved_metrics['models'], reconstructed, group + '/macro')
            natural[group] = reconstructed
        elif group == 'analytic':
            for name in names:
                analytic[name] = independent_analytic_gate(cases[name], original_cases[name], dsp_cases)
                compare_values(saved_metrics['gates'][name], {'pass': analytic[name]['pass'], 'checks': analytic[name]['checks']}, 'analytic/gates/' + name)
        else:
            for name in names:
                expected = voicing_event_gate(event_cases[name], event_cases['hybrid_original'])
                require(saved_metrics['gates'][name]['pass'] == expected['pass'], 'Independent event gate differs: ' + name)
                analytic[name]['event_gate'] = expected
                analytic[name]['pass'] = analytic[name]['pass'] and expected['pass']
        summaries[group] = {'records': len(frozen['records']), 'metrics_sha256': sha(folder / 'metrics.json')}
    decision = audit.json(run / 'decision.json')
    for group, binding in decision['bindings'].items():
        audit.bind(binding['path'], binding['sha256'])
        require(binding['sha256'] == summaries[group]['metrics_sha256'], 'Decision metric binding differs')
    expected = development_gates(natural['development'], analytic)
    compare_values(decision, {k: expected[k] for k in ('status', 'pass', 'checks', 'effects', 'export_designation')}, 'Development decision')
    require(decision['candidate_passed_development'] == expected['pass'], 'Candidate admission differs')
    return {'groups': summaries, 'verified_frame_model_pairs': verified_pairs, 'maximum_numpy_absolute_error': max_error,
        'analytic': analytic, 'decision': expected, 'development': natural['development']}


def audit_run(run, output=None):
    """Bind authentic prepared inputs and all six fits; never turn absence into pass."""
    run = Path(run).resolve(); output = Path(output) if output is not None else run / 'independent-audit.json'
    require(not output.exists(), 'Refusing to overwrite an independent audit')
    audit = Bindings()
    report = {'schema': 'phenometrix-voicing-robustness-independent-audit.v1', 'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(), 'scientific_pass': False}
    write_json(output, report)
    try:
        plan = audit.json(run / 'plan/plan.json')
        require(plan['schema'] == 'phenometrix-voicing-robustness-plan.v1' and plan['status'] == 'frozen' and plan['candidate'] == 'R-20261025' and plan['seeds'] == list(SEEDS) and plan['arms'] == ['C', 'R'], 'Plan contract differs')
        for path, digest in plan['source_sha256'].items():
            audit.bind(run / 'plan/source-snapshot' / path, digest)
        baseline_report = audit.json(plan['baseline_audit']['path'], plan['baseline_audit']['sha256'])
        require(baseline_report['status'] == 'complete' and baseline_report['shifted_fn_headroom']['fixed_yin_makes_20pct_reduction_possible'] is True, 'Fixed-YIN headroom not authenticated')
        prep, schedules, preparation = audit_preparation(audit, run, plan)
        report['preparation'] = preparation
        fits_path = run / 'fits/manifest.json'
        if not fits_path.is_file():
            report.update(status='complete', fit_status='not_started', scientific_status='unassessable', note='Prepared inputs and matched schedule authenticated; no trained result or scientific qualification')
        else:
            execution = audit.json(run / 'execution-freeze.json')
            require('source_sha256' in execution and execution['source_sha256'].get(str(Path(__file__).relative_to(REPO))) == sha(__file__), 'Auditor differs from prefit execution freeze')
            for path, digest in execution['source_sha256'].items():
                audit.bind(run / 'execution-source-snapshot' / path, digest)
            fits = audit.json(fits_path)
            require(fits['status'] == 'complete' and set(fits['fits']) == {f'{arm}-{s}' for s in SEEDS for arm in ('C', 'R')}, 'Exactly six complete fits required')
            source = audit.json(run / 'fits/source-freeze.json')
            for path, digest in source['source_sha256'].items():
                audit.bind(run / 'fits/source-snapshot' / path, digest)
            original = checkpoint(audit, ROOT / 'runs/delayed-mlp-seed20261022/weights.pt', BASELINE_SHA)
            paired, checkpoints = [], {}
            for seed in SEEDS:
                manifests, values = {}, {}
                expected_initial = independent_initialization(seed, original)
                for arm in ('C', 'R'):
                    name = f'{arm}-{seed}'; binding = fits['fits'][name]
                    fit = audit.json(binding['path'], binding['sha256']); folder = (REPO / binding['path']).parent
                    for key, expected in [('plan', sha(run / 'plan/plan.json')), ('prepare', sha(run / 'prepare/manifest.json')), ('schedule', prep['schedules'][str(seed)]['sha256'])]:
                        audit.bind(fit[key]['path'], fit[key]['sha256'])
                        require(fit[key]['sha256'] == expected, 'Fit input binding differs: ' + key)
                    require(fit['arm'] == arm and fit['scaling_refitted'] is False and fit['scaling_provenance']['sha256'] == BASELINE_SHA, 'Fit arm/scaling provenance differs')
                    initial = checkpoint(audit, folder / 'initial-state.pt', fit['initial_checkpoint_sha256'])
                    require(tensor_digest(initial) == fit['initialization_sha256'] == tensor_digest(expected_initial), 'Initial tensor content or RNG differs')
                    final = checkpoint(audit, folder / 'weights.pt', fit['weights_sha256'])
                    independent_mlp(final, np.zeros((1, 57), np.float32), np.ones((1, 6), bool))
                    require(fit['epoch_updates'] == schedules[seed]['epoch_updates'], 'Actual epoch update inventory differs')
                    require(fit['valid_supervision_frames'] == preparation['valid_supervision_frames'][arm] and fit['covered_candidate_frames'] == preparation['covered_candidate_frames'][arm], 'Fit supervision exposure differs')
                    normalized = {**fit, 'plan_sha256': fit['plan']['sha256'], 'prepare_manifest_sha256': fit['prepare']['sha256'], 'schedule_sha256': fit['schedule']['sha256']}
                    manifests[arm], values[arm], checkpoints[name] = normalized, final, final
                paired.append(validate_fit_pair(manifests['C'], manifests['R'], values['C'], values['R'], original, schedules[seed]['expected_updates']))
            report.update(status='complete', fit_status='six_fits_authenticated', paired_fits=paired, scientific_status='unassessable', note='Matched fit contracts authenticated; prediction reconstruction and independent development gate receipt still required')
        if (run / 'decision.json').is_file():
            evaluation = audit_evaluations(audit, run, checkpoints)
            report.update(evaluation=evaluation, scientific_status=evaluation['decision']['status'], scientific_pass=evaluation['decision']['pass'], note='Inputs, paired fits, all saved model outputs, metrics and decision independently reconstructed')
        report.update(completed_at=datetime.now(timezone.utc).isoformat(), bound_files=audit.files, source_sha256={str(Path(__file__).relative_to(REPO)): sha(__file__)})
        write_json(output, report)
        return report
    except BaseException as error:
        report.update(status='failed', failure=repr(error), bound_files=audit.files)
        write_json(output, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-output', type=Path)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.baseline_output is not None:
        report = baseline(args.baseline_output)
        print(json.dumps({'status': report['status'], 'headroom': report['shifted_fn_headroom']}))
    elif args.run is not None:
        report = audit_run(args.run, args.output)
        print(json.dumps({k: report.get(k) for k in ('status', 'fit_status', 'scientific_status', 'scientific_pass')}))
    else:
        parser.error('--baseline-output or --run required')


if __name__ == '__main__':
    main()
