"""Independent CPU checkpoint/output reproduction, without references or fitting.

Reads frozen inputs and emitted predictions, not metrics or analytic F0 labels.
Uses explicit Torch equations rather than the experiment runner's infer helper
or the HarmonicSelector.forward method. Shared grouping is source-bound and
reused; this is not an independent reconstruction of acquisition/DSP features.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from .selector_model import prepare_selector_batch
from .harmonic_selector_model import HarmonicSelector
from .model import FEATURE_NAMES_V2

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'research/voice-prosody'
RUNS = ROOT / 'runs'
ORIGINAL_SHA = '06110469a9a643fb16806905569ebc939db2c0bdd5b3a6a1593b31789453d10d'
SEEDS = (20261022, 20261023, 20261024)
ARMS = (('absolute', 'nearest'), ('absolute', 'acceptable-set'), ('relative', 'nearest'), ('relative', 'acceptable-set'))
MODELS = {f'{mode}__{objective}__seed{seed}' for seed in SEEDS for mode, objective in ARMS}
REGRESSION_IDS = {'constant-100', 'constant-200', 'constant-300', 'step-100-200', 'step-200-100',
    *(f'glide-{direction}-{speed}' for direction in ('up', 'down') for speed in (50, 100, 200)),
    'observation-change-100', 'observation-change-200', 'observation-change-300'}
CHALLENGE_IDS = {f'harmonic-balance-f{f}-p{p}' for f in range(6) for p in range(2)}
SCALING_BUFFERS = ('candidate_mean', 'candidate_scale', 'global_mean', 'global_scale', 'scaling_fitted')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


class Ledger:
    def __init__(self):
        self.files = {}

    def read(self, path, role, expected=None):
        path = Path(path).resolve()
        require(path.is_relative_to(REPO), 'Input escaped repository')
        require(not any(p in {'sealed-test', 'sealed_test'} for p in path.parts), 'Reserved content excluded')
        require(role in {'manifest', 'freeze', 'source', 'source_snapshot', 'weights', 'feature', 'prediction', 'pcm'},
                'Reference/score/label role is not admitted')
        data = path.read_bytes(); sha = digest(data); name = str(path.relative_to(REPO))
        require(expected is None or sha == expected, 'Input identity mismatch: ' + name)
        require(name not in self.files or self.files[name]['sha256'] == sha, 'Input changed: ' + name)
        self.files[name] = {'sha256': sha, 'bytes': len(data), 'role': role}
        return data

    def json(self, *args, **kwargs):
        return json.loads(self.read(*args, **kwargs))

    def npz(self, *args, **kwargs):
        with np.load(BytesIO(self.read(*args, **kwargs)), allow_pickle=False) as z:
            return {key: z[key].copy() for key in z.files}


def original_probability(weights, features):
    """Original voicing only, independently evaluated with CPU Torch equations."""
    x = torch.as_tensor(features, dtype=torch.float32)
    z = (x - weights['feature_mean']) / weights['feature_scale']
    h = torch.tanh(F.linear(z, weights['projection.weight'], weights['projection.bias']))
    h = torch.tanh(F.linear(h, weights['frame_hidden.weight'], weights['frame_hidden.bias']))
    v = F.linear(h, weights['voicing_head.weight'], weights['voicing_head.bias'])
    require(bool(torch.isfinite(v).all()), 'Nonfinite original voicing logits')
    return torch.sigmoid(v[:, 0].double()).numpy()


def independent_logits(weights, context_mode, features, mask):
    """Independent explicit equations; no call to production/research forward."""
    x = torch.as_tensor(features, dtype=torch.float32)
    valid = torch.as_tensor(mask, dtype=torch.bool)
    raw = x[:, :48].reshape(-1, 6, 8)
    z = (raw - weights['candidate_mean']) / weights['candidate_scale']
    encoded = torch.tanh(F.linear(z, weights['candidate_encoder.weight'], weights['candidate_encoder.bias']))
    # Index dimensions are explicitly [batch, query, neighbor].
    ratios = (raw[:, None, :, 1] - raw[:, :, None, 1]) / math.log2(14.)
    require(context_mode in {'absolute', 'relative'}, 'Unknown context mode')
    if context_mode == 'absolute':
        ratios = torch.zeros_like(ratios)
    edges = valid[:, :, None] & valid[:, None, :]
    neighbors = z[:, None, :, :].expand(-1, 6, -1, -1)
    message_input = torch.cat((neighbors, ratios[..., None]), dim=3)
    message_input = torch.where(edges[..., None], message_input, torch.zeros_like(message_input))
    messages = torch.tanh(F.linear(message_input, weights['message_encoder.weight'], weights['message_encoder.bias']))
    masked_messages = torch.where(edges[..., None], messages, torch.full_like(messages, -torch.inf))
    context = torch.amax(masked_messages, dim=2)
    context = torch.where(valid[..., None], context, torch.zeros_like(context))
    globals_ = ((x[:, 48:] - weights['global_mean']) / weights['global_scale'])[:, None, :].expand(-1, 6, -1)
    combined = torch.cat((encoded, context, globals_), dim=2)
    hidden = torch.tanh(F.linear(combined, weights['shared_hidden.weight'], weights['shared_hidden.bias']))
    logits = F.linear(hidden, weights['shared_head.weight'], weights['shared_head.bias']).squeeze(-1)
    logits = torch.where(valid, logits, torch.full_like(logits, -1e9))
    require(bool(torch.isfinite(logits).all()), 'Nonfinite independent selector logits')
    return logits


def emit(logits, hz, mask, probability):
    """Independent mask, lowest-Hz tie, external-voice and unavailable policy."""
    logits = logits.numpy(); mask = np.asarray(mask, bool)
    safe = np.where(mask, logits, -np.inf)
    tied = mask & (safe == safe.max(axis=1, keepdims=True))
    chosen = np.where(tied, hz, np.inf).min(axis=1)
    state = np.where(probability < .5, 0, np.where(mask.any(axis=1), 1, 2)).astype(np.uint8)
    return state, np.where(state == 1, chosen, np.nan)


def initial_tensor_hash(mode, objective, seed, trained_weights):
    torch.manual_seed(seed)
    initial = HarmonicSelector(mode, objective)
    # Scaling is input-only and fixed across every arm; reuse its frozen bytes,
    # without opening training inputs or fitting preprocessing again.
    with torch.no_grad():
        for name in SCALING_BUFFERS:
            getattr(initial, name).copy_(trained_weights[name])
    sha = hashlib.sha256()
    for name, value in sorted(initial.state_dict().items()):
        sha.update(name.encode()); sha.update(value.detach().cpu().numpy().tobytes())
    return sha.hexdigest()


def validate_feature(f):
    x, hz, mask = f['features'], f['candidate_hz'], f['candidate_mask']
    n = int(f['pcm_sample_count']); ends = f['reference_sample']
    require(x.shape == (len(ends), 57) and x.dtype == np.float32 and np.isfinite(x).all(), 'Invalid acquisition features')
    require(hz.shape == mask.shape == (len(ends), 6) and mask.dtype == bool, 'Invalid candidate dimensions')
    require(np.isfinite(hz[mask]).all() and ((hz[mask] >= 50) & (hz[mask] <= 700)).all()
            and np.isnan(hz[~mask]).all(), 'Candidate validity differs')
    require(np.array_equal(ends, np.arange(2208, n + 1, 480)) and int(f['sample_rate_hz']) == 48000
            and list(f['feature_names']) == list(FEATURE_NAMES_V2), 'Input grid/schema differs')


def prepare(raw):
    b = prepare_selector_batch(raw['features'], raw['candidate_hz'], raw['candidate_mask'], representation='shared-candidates')
    mask, hz = b.candidate_mask.numpy(), b.candidate_hz.numpy()
    # Every grouped output must be one of its declared original members.
    for column in range(6):
        members = b.member_mask[:, column, :].numpy()
        equal = members & raw['candidate_mask'] & (raw['candidate_hz'] == hz[:, column, None])
        require(np.array_equal(equal.any(axis=1), mask[:, column]), 'Grouping invented an output frequency')
    return b


def compare_arrays(saved, state, hz, batch, probability):
    old_state, old_hz = saved
    require(old_state.shape == state.shape and old_hz.shape == hz.shape, 'Frozen output extent differs')
    require(np.isin(old_state, [0, 1, 2]).all(), 'Invalid saved state')
    voiced = old_state == 1
    require(np.isfinite(old_hz[voiced]).all() and np.isnan(old_hz[~voiced]).all(), 'Invalid saved frequency availability')
    member = ((old_hz[:, None] == batch.candidate_hz.numpy()) & batch.candidate_mask.numpy()).any(axis=1)
    raw_member = ((old_hz[:, None] == batch.original_candidate_hz.numpy()) & batch.original_candidate_mask.numpy()).any(axis=1)
    voice_mismatch = int(((old_state == 0) != (probability < .5)).sum())
    return {'frames': len(state), 'state_mismatches': int((old_state != state).sum()),
            'hz_mismatches': int((~((old_hz == hz) | (np.isnan(old_hz) & np.isnan(hz)))).sum()),
            'frozen_voice_gate_mismatches': voice_mismatch,
            'nonmember_emissions': int((voiced & ~(member & raw_member)).sum()),
            'saved_voiced_frames': int(voiced.sum()), 'saved_unavailable_frames': int((old_state == 2).sum())}


def audit(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    require(run.is_relative_to(RUNS) and output.is_relative_to(RUNS), 'Artifacts must remain in ignored runs')
    ledger_path = output.with_name(output.stem + '-inputs.json')
    require(not output.exists() and not ledger_path.exists(), 'Audit refuses overwrite')
    ledger = Ledger(); began = time.monotonic(); torch.set_num_threads(1)
    result = {'schema': 'phenometrix-harmonic-checkpoint-inference-audit.v1', 'status': 'running', 'started_at': now(),
        'device': 'cpu', 'torch': torch.__version__, 'numpy': np.__version__, 'fitting_performed': False,
        'reference_files_or_analytic_target_arrays_read': False, 'truth_used_for_comparison': False, 'challenge_spec_or_metrics_content_opened': False, 'reserved_content_read': False, 'new_gpu_work': False,
        'inference_implementation': 'independent explicit Torch equations; no runner infer or HarmonicSelector.forward',
        'preprocessing_scope': 'source-bound shared grouping reused, with independent original-member validation',
        'comparisons': {}, 'original_voicing_checks': {}, 'initialization_checks': {}}
    try:
        manifest_path = run / 'manifest.json'; checkpoint_path = run / 'checkpoint-freeze.json'; prediction_path = run / 'prediction-freeze.json'
        manifest = ledger.json(manifest_path, 'manifest')
        checkpoints = ledger.json(checkpoint_path, 'freeze')
        frozen = ledger.json(prediction_path, 'freeze')
        require(checkpoints['status'] == 'checkpoints_frozen' and frozen['status'] == 'predictions_frozen', 'Incomplete required freezes')
        require(set(frozen['fits']) == set(checkpoints['fits']) == MODELS and frozen['fits'] == checkpoints['fits'] == manifest['fits'], 'Checkpoint inventory differs')
        require(frozen['source_sha256'] == checkpoints['source_sha256'] == manifest['source_sha256'], 'Source identities differ across phases')
        require(frozen['prediction_sha256'] == manifest['prediction_sha256'], 'Prediction identities changed')
        require(frozen['development_references_read'] is False and frozen['reserved_content_read'] is False, 'Prediction phase label boundary differs')
        result['original_run_status'] = manifest['status']
        result['original_run_failure'] = manifest.get('failure')
        result['identities'] = {name: digest(ledger.read(path, role)) for name, path, role in
            [('manifest_sha256', manifest_path, 'manifest'), ('checkpoint_freeze_sha256', checkpoint_path, 'freeze'), ('prediction_freeze_sha256', prediction_path, 'freeze')]}
        for name, sha in frozen['source_sha256'].items():
            ledger.read(REPO / name, 'source', sha); ledger.read(run / 'source-snapshot' / name, 'source_snapshot', sha)
        ledger.read(Path(__file__), 'source')
        inputs = frozen['input_sha256']
        def bound(path, role, expected=None):
            path = Path(path).resolve()
            require(str(path) in inputs, 'Input absent from prediction-freeze ledger: ' + str(path))
            require(expected is None or expected == inputs[str(path)], 'Nested input hashes disagree')
            return ledger.read(path, role, inputs[str(path)])
        models = {}
        for name in sorted(MODELS):
            fit = frozen['fits'][name]
            ck = torch.load(BytesIO(ledger.read(run / 'checkpoints' / (name + '.pt'), 'weights', fit['weights_sha256'])), map_location='cpu', weights_only=True)
            mode, objective, seed_text = name.split('__'); seed = int(seed_text.removeprefix('seed'))
            require(ck['context_mode'] == mode and ck['config'] == {'representation': 'shared-candidates', 'loss': objective, 'hidden_size': 32}, 'Checkpoint config mismatch')
            w = ck['state_dict']; reference_model = HarmonicSelector(mode, objective)
            reference_model.load_state_dict(w, strict=True)
            require(all(bool(torch.isfinite(t).all()) for t in w.values()) and bool(w['scaling_fitted']), 'Invalid checkpoint values')
            require(bool((w['candidate_scale'] > 0).all()) and bool((w['global_scale'] > 0).all()), 'Invalid scaling')
            require(sum(p.numel() for p in reference_model.parameters()) == 3009, 'Wrong parameter count')
            actual_init = initial_tensor_hash(mode, objective, seed, w)
            require(actual_init == fit['initial_tensors_sha256'], 'Recorded initialization does not reproduce')
            result['initialization_checks'][name] = actual_init
            models[name] = (mode, w)
        for seed in SEEDS:
            require(len({result['initialization_checks'][name] for name in MODELS if name.endswith(str(seed))}) == 1, 'Initial tensors differ within seed')
        original_path = RUNS / 'delayed-mlp-seed20261022/weights.pt'
        original = torch.load(BytesIO(bound(original_path, 'weights', ORIGINAL_SHA)), map_location='cpu', weights_only=True)
        require(original['config']['kind'] == 'framewise-mlp', 'Wrong original voicing model')
        original_weights = original['state_dict']
        def probability_check(identity, f, old_p=None):
            with torch.inference_mode():
                ps = [original_probability(original_weights, f['features'][i:i+4096]) for i in range(0, len(f['features']), 4096)]
            p = np.concatenate(ps)
            require(np.isfinite(p).all(), 'Nonfinite original probability')
            report = {'frames': len(p), 'probability_sha256_float64': digest(p.astype('<f8').tobytes()), 'all_finite': True}
            if old_p is not None:
                require(old_p.shape == p.shape and np.isfinite(old_p).all(), 'Frozen voicing extent/finiteness differs')
                report.update(max_absolute_difference=float(np.abs(old_p-p).max()), gate_mismatches=int(((old_p >= .5) != (p >= .5)).sum()))
                require(report['max_absolute_difference'] <= 1e-6 and report['gate_mismatches'] == 0, 'Original voicing parity failed')
                p = old_p  # exact frozen external policy is authoritative for output reproduction
            result['original_voicing_checks'][identity] = report
            return p
        def verify_dataset(identity, batch, probability, saved_by_model):
            reports = {}
            for name, (mode, w) in models.items():
                states, frequencies = [], []
                with torch.inference_mode():
                    for start in range(0, len(batch.features), 4096):
                        stop = min(start+4096, len(batch.features))
                        logits = independent_logits(w, mode, batch.features[start:stop], batch.candidate_mask[start:stop])
                        state, hz = emit(logits, batch.candidate_hz[start:stop].numpy(), batch.candidate_mask[start:stop].numpy(), probability[start:stop])
                        states.append(state); frequencies.append(hz)
                report = compare_arrays(saved_by_model[name], np.concatenate(states), np.concatenate(frequencies), batch, probability)
                reports[name] = report
            result['comparisons'][identity] = reports
            print(json.dumps({'dataset': identity, 'frames_per_model': len(batch.features), 'models': len(reports),
                'state_or_hz_mismatches': sum(r['state_mismatches']+r['hz_mismatches'] for r in reports.values())}), flush=True)
        # Development acquisition inputs and frozen original probabilities; no reference files.
        dev_root = RUNS / 'delayed-development-features'
        dev_manifest = json.loads(bound(dev_root / 'manifest.json', 'manifest'))
        require(dev_manifest['status'] == 'complete' and dev_manifest['split'] == 'development' and len(dev_manifest['recordings']) == 320, 'Development input inventory differs')
        require({r['speaker'] for r in dev_manifest['recordings']} == {'F01', 'F03', 'M01', 'M03'}, 'Nondevelopment speaker')
        baseline_root = RUNS / 'hybrid-natural-v1'
        baseline = json.loads(bound(baseline_root / 'prediction-freeze.json', 'freeze'))
        baseline_rows = {r['recording_id']: r for r in baseline['recordings']}
        pieces, probabilities = [], []
        for record in dev_manifest['recordings']:
            identity = record['recording_id']; fbytes = bound(dev_root / 'features' / (identity + '.npz'), 'feature', record['feature_sha256'])
            with np.load(BytesIO(fbytes), allow_pickle=False) as z: f = {k:z[k].copy() for k in z.files}
            validate_feature(f); pieces.append(f)
            b = baseline_rows[identity]
            payload = json.loads(gzip.decompress(bound(baseline_root / b['hybrid_prediction_path'], 'prediction', b['hybrid_prediction_sha256'])))
            rows = payload['predictions']['mlp_original']
            require(np.array_equal([r['sample'] for r in rows], f['reference_sample']-960), 'Frozen voice target grid differs')
            probabilities.append(np.asarray([r['voicing_probability'] for r in rows], np.float64))
        raw = {k: np.concatenate([f[k] for f in pieces]) for k in ('features','candidate_hz','candidate_mask')}
        require(len(raw['features']) == 255532, 'Development acquired frame count differs')
        p = probability_check('development', raw, np.concatenate(probabilities)); batch = prepare(raw)
        saved = {}
        for name in MODELS:
            relative = 'predictions/' + name + '.npz'
            z = ledger.npz(run / relative, 'prediction', frozen['prediction_sha256'][relative]); saved[name]=(z['state'],z['hz'])
        verify_dataset('development', batch, p, saved)
        # Reused analytic inputs and frozen original p; analytic truth is not opened.
        regression_root = RUNS / 'delayed-known-truth'
        regression = json.loads(bound(regression_root / 'prediction-freeze.json', 'freeze'))
        require(len(regression['recordings']) == 14 and {r['id'] for r in regression['recordings']} == REGRESSION_IDS, 'Regression inventory differs')
        for record in regression['recordings']:
            identity = record['id']; path = regression_root / 'features' / (identity + '.npz')
            with np.load(BytesIO(bound(path, 'feature', record['feature_sha256'])), allow_pickle=False) as z: f = {k:z[k].copy() for k in z.files}
            validate_feature(f)
            payload = json.loads(gzip.decompress(bound(regression_root / 'predictions' / (identity + '.json.gz'), 'prediction', record['prediction_sha256'])))
            rows = payload['predictions']['mlp']
            require(np.array_equal([r['sample'] for r in rows], f['reference_sample']-960), 'Regression voice grid differs')
            p = probability_check('regression/'+identity, f, np.asarray([r['voicing_probability'] for r in rows], np.float64))
            relative = 'predictions/regression-' + identity + '.npz'
            z = ledger.npz(run / relative, 'prediction', frozen['prediction_sha256'][relative])
            saved = {name:(z[name+'__state'],z[name+'__hz']) for name in MODELS}
            verify_dataset('regression/'+identity, prepare(f), p, saved)
        # New challenge input manifest has hashes, not analytic frequencies. Do
        # not open its spec.json, metrics, scenario truth or reference arrays.
        challenge_root = run / 'challenge'
        challenge = json.loads(bound(challenge_root / 'manifest.json', 'manifest'))
        require(challenge['status'] == 'complete' and challenge['checkpoint_freeze_sha256'] == result['identities']['checkpoint_freeze_sha256']
                and challenge['spec_sha256'] == frozen['challenge_spec_sha256'] and len(challenge['records']) == 12
                and {r['id'] for r in challenge['records']} == CHALLENGE_IDS, 'Challenge freeze/inventory differs')
        for record in challenge['records']:
            identity = record['id']
            pcm = bound(challenge_root / record['pcm_path'], 'pcm', record['pcm_sha256'])
            require(len(pcm) == 96000*4, 'Challenge PCM extent differs')
            with np.load(BytesIO(bound(challenge_root / record['feature_path'], 'feature', record['feature_sha256'])), allow_pickle=False) as z:
                f = {k:z[k].copy() for k in z.files}
            validate_feature(f); require(len(f['features']) == record['feature_frames'], 'Challenge feature extent differs')
            p = probability_check('challenge/'+identity, f)
            relative = 'predictions/challenge-' + identity + '.npz'
            z = ledger.npz(run / relative, 'prediction', frozen['prediction_sha256'][relative])
            saved = {name:(z[name+'__state'],z[name+'__hz']) for name in MODELS}
            verify_dataset('challenge/'+identity, prepare(f), p, saved)
        totals = {name: sum(r[name] for domain in result['comparisons'].values() for r in domain.values())
                  for name in ('frames','state_mismatches','hz_mismatches','frozen_voice_gate_mismatches','nonmember_emissions')}
        result.update(totals=totals, model_count=12, analytic_case_count=26, development_frames_per_model=255532)
        require(len(result['comparisons']) == 27, 'Incomplete inference domain inventory')
        for name, binding in ledger.files.items():
            require(digest((REPO/name).read_bytes()) == binding['sha256'], 'Input bytes changed before audit receipt')
        result.update(status='complete', input_bytes_stable=True,
            pass_result=all(totals[k] == 0 for k in ('state_mismatches','hz_mismatches','frozen_voice_gate_mismatches','nonmember_emissions')))
    except BaseException as error:
        result.update(status='failed', pass_result=False, failure=repr(error))
    finally:
        output.parent.mkdir(parents=True,exist_ok=True)
        ledger_bytes=(json.dumps(ledger.files,sort_keys=True,indent=2)+'\n').encode(); ledger_path.write_bytes(ledger_bytes)
        result.update(completed_at=now(),elapsed_seconds=time.monotonic()-began,input_file_count=len(ledger.files),input_ledger_sha256=digest(ledger_bytes),
                      audit_source_sha256=digest(Path(__file__).read_bytes()))
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'comparisons','original_voicing_checks','initialization_checks'}},indent=2))
    if not result['pass_result']:
        raise SystemExit(1)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); audit(args.run,args.output)
