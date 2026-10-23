"""Postfit descriptive decomposition of frozen prospective challenge predictions.

No fitting, waveform generation, reference-file access, model inference, filtering
of difficult frames, or additional acceptance gate. Saved state 0 reconstructs
the original fixed p(voiced)<.5 decision under the frozen infer() source; actual
probability values were not retained by the experiment runner.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from .selector_model import prepare_selector_batch

REPO = Path(__file__).resolve().parents[3]
SEEDS = (20261022, 20261023, 20261024)
ORIGINAL_WEIGHTS = '06110469a9a643fb16806905569ebc939db2c0bdd5b3a6a1593b31789453d10d'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, reason):
    if not value:
        raise ValueError(reason)


def within(candidate_hz, mask, truth):
    safe = np.where(mask, candidate_hz, truth)
    return ((np.abs(1200 * np.log2(safe / truth)) <= 50) & mask).any(axis=-1)


def decompose(state, hz, truth, included, raw_covered, grouped_covered):
    """Unconditional partition and coverage intersections; every included row stays."""
    state, hz = np.asarray(state), np.asarray(hz)
    require(state.shape == hz.shape == included.shape == raw_covered.shape == grouped_covered.shape,
            'Diagnostic arrays differ')
    require(np.isin(state, [0, 1, 2]).all(), 'Unknown saved prediction state')
    require(np.array_equal(np.isfinite(hz), state == 1), 'State and emitted frequency disagree')
    require(np.all(hz[state == 1] > 0), 'Nonpositive emitted pitch')
    voiced = state == 1
    cents = np.full(len(state), np.inf)
    cents[voiced] = np.abs(1200 * np.log2(hz[voiced] / truth))
    correct = voiced & (cents <= 50)
    categories = {'correct_pitch': correct, 'incorrect_pitch': voiced & ~correct,
                  'unvoiced': state == 0, 'unavailable': state == 2}
    n = int(included.sum())
    require(n > 0, 'Empty post-change region')
    counts = {name: int((included & mask).sum()) for name, mask in categories.items()}
    require(sum(counts.values()) == n, 'Outcome partition lost frames')
    require(not np.any(included & correct & ~grouped_covered), 'Correct emission absent from grouped candidates')
    by_coverage = {}
    coverage_groups = {'grouped_correct_candidate_present': grouped_covered,
                       'correct_candidate_lost_in_grouping': raw_covered & ~grouped_covered,
                       'raw_correct_candidate_absent': ~raw_covered}
    require(not np.any(grouped_covered & ~raw_covered), 'Grouping invented correct candidate')
    for name, mask in coverage_groups.items():
        by_coverage[name] = {outcome: int((included & mask & value).sum()) for outcome, value in categories.items()}
    wrong = hz[included & categories['incorrect_pitch']]
    wrong_summary = None if not len(wrong) else {'min_hz': float(wrong.min()), 'median_hz': float(np.median(wrong)),
        'max_hz': float(wrong.max()), 'near_half_f0_frames': int((np.abs(1200*np.log2(wrong/(truth/2))) <= 50).sum()),
        'near_double_f0_frames': int((np.abs(1200*np.log2(wrong/(truth*2))) <= 50).sum())}
    return {'frames': n, 'counts': counts, 'incorrect_pitch_summary': wrong_summary,
            'success_within_50c_all_frames': counts['correct_pitch'] / n,
            'failure_all_frames': (n-counts['correct_pitch']) / n,
            'outcome_by_candidate_coverage': by_coverage}


def diagnose(run: Path, recovery: Path | None = None):
    run = run.resolve()
    require(run.is_relative_to(REPO / 'research/voice-prosody/runs'), 'Expected ignored local run')
    bound = {}
    def bind(path, expected=None):
        path = path.resolve(); digest = sha(path)
        require(path.is_relative_to(REPO), 'Evidence outside repository')
        require(expected is None or digest == expected, f'Changed evidence: {path.relative_to(REPO)}')
        bound[str(path.relative_to(REPO))] = digest
        return path
    def read(path, expected=None):
        return json.loads(bind(path, expected).read_text())
    manifest = read(run/'manifest.json')
    freeze = read(run/'prediction-freeze.json')
    checkpoint = read(run/'checkpoint-freeze.json')
    require(manifest['status'] in ('failed', 'complete'), 'Wait for bounded run to finish')
    require(freeze['status'] == 'predictions_frozen' and checkpoint['status'] == 'checkpoints_frozen',
            'Complete checkpoint and prediction freezes required')
    require(len(freeze['fits']) == len(checkpoint['fits']) == 12 and freeze['fits'] == checkpoint['fits'], 'Incomplete fit freeze')
    require(freeze['source_sha256'] == checkpoint['source_sha256'] == manifest['source_sha256'], 'Frozen source maps differ')
    for name, digest in freeze['source_sha256'].items():
        bind(REPO/name, digest)
        bind(run/'source-snapshot'/name, digest)
    for name, fit in freeze['fits'].items():
        bind(run/'checkpoints'/(name+'.pt'), fit['weights_sha256'])
    original = bind(REPO/'research/voice-prosody/runs/delayed-mlp-seed20261022/weights.pt', ORIGINAL_WEIGHTS)
    require(freeze['frozen_original_weights_sha256'] == ORIGINAL_WEIGHTS, 'Original voicing identity differs')
    cm = read(run/'challenge/manifest.json', freeze['input_sha256'][str(run/'challenge/manifest.json')])
    spec = read(run/'challenge/spec.json', freeze['input_sha256'][str(run/'challenge/spec.json')])
    spec_digest = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    require(cm['status'] == 'complete' and cm['spec_sha256'] == spec_digest == freeze['challenge_spec_sha256'], 'Challenge freeze differs')
    require(cm['checkpoint_freeze_sha256'] == sha(run/'checkpoint-freeze.json'), 'Challenge checkpoint binding differs')
    require(len(cm['records']) == len(spec['records']) == 12, 'Incomplete prospective challenge')
    require(datetime.fromisoformat(checkpoint['checkpoints_frozen_at']) <= datetime.fromisoformat(cm['started_at']) <=
            datetime.fromisoformat(cm['completed_at']) <= datetime.fromisoformat(freeze['predictions_frozen_at']), 'Freeze chronology differs')
    methods = [f'relative__acceptable-set__seed{s}' for s in SEEDS]
    expected_methods = {f'{mode}__{objective}__seed{s}' for mode in ('absolute', 'relative')
                        for objective in ('nearest', 'acceptable-set') for s in SEEDS}
    expected_methods |= {f'prior__shared-candidates__{objective}__seed{s}'
                         for objective in ('nearest', 'acceptable-set') for s in SEEDS}
    cases = {}
    total = {method: {'frames': 0, 'correct_pitch': 0, 'incorrect_pitch': 0, 'unvoiced': 0, 'unavailable': 0}
             for method in methods}
    for record, saved in zip(spec['records'], cm['records']):
        case = record['id']; require(case == saved['id'], 'Challenge order differs')
        require(record['sample_rate_hz'] == 48000 and record['sample_count'] == 96000 and record['change_sample'] == 39840, 'Integer challenge support differs')
        bind(run/'challenge'/saved['pcm_path'], saved['pcm_sha256'])
        fp = bind(run/'challenge'/saved['feature_path'], saved['feature_sha256'])
        require(freeze['input_sha256'][str(fp)] == saved['feature_sha256'], 'Feature absent from prediction freeze')
        with np.load(fp, allow_pickle=False) as data:
            f = {k: data[k].copy() for k in data.files}
        end = f['reference_sample']
        require(np.array_equal(end, np.arange(2208, 96001, 480)), 'Feature acquisition grid changed')
        require(int(f['sample_rate_hz']) == 48000 and int(f['pcm_sample_count']) == 96000, 'Feature extent differs')
        reference = end - 960; start = end - 1920
        eligible = (reference >= 3072) & (reference + 1536 <= 96000)
        post = eligible & (start >= record['change_sample'])
        raw_mask, raw_hz = f['candidate_mask'], f['candidate_hz']
        batch = prepare_selector_batch(f['features'], raw_hz, raw_mask, representation='shared-candidates')
        group_mask, group_hz = batch.candidate_mask.numpy(), batch.candidate_hz.numpy()
        truth = record['frequency_hz']
        raw_cov, group_cov = within(raw_hz, raw_mask, truth), within(group_hz, group_mask, truth)
        require(not np.any(group_cov & ~raw_cov), 'Grouping added unavailable correct Hz')
        pp = Path('predictions')/('challenge-'+case+'.npz')
        pred_path = bind(run/pp, freeze['prediction_sha256'][str(pp)])
        with np.load(pred_path, allow_pickle=False) as data:
            pred = {k: data[k].copy() for k in data.files}
        require(set(pred) == {m+'__'+field for m in expected_methods for field in ('state', 'hz')}, 'Challenge prediction inventory differs')
        reference_state = pred[methods[0]+'__state']
        for method in expected_methods:
            state, hz = pred[method+'__state'], pred[method+'__hz']
            require(state.shape == hz.shape == end.shape, 'Prediction extent differs')
            require(np.isin(state, [0, 1, 2]).all() and np.array_equal(state == 0, reference_state == 0), 'Frozen voicing disagreement')
            require(np.array_equal(np.isfinite(hz), state == 1), 'Prediction validity differs')
            require(np.array_equal(state == 2, (state != 0) & ~group_mask.any(axis=-1)), 'Unavailable state differs from candidates')
            require(np.all(np.any(group_mask & (group_hz == hz[:,None]), axis=-1)[state == 1]), 'Selected pitch was not an original grouped member')
        n = int(post.sum()); require(n > 0, 'No complete post-change frames')
        outcomes = {}
        for method in methods:
            outcomes[method] = decompose(pred[method+'__state'], pred[method+'__hz'], truth, post, raw_cov, group_cov)
            total[method]['frames'] += n
            for k,v in outcomes[method]['counts'].items(): total[method][k] += v
        cases[case] = {'frequency_hz': truth, 'phase_radians': record['phase_radians'],
            'post_region': {'frames': n, 'first_support_start_sample': int(start[post][0]),
                            'last_support_end_sample_exclusive': int(end[post][-1]),
                            'first_reference_sample': int(reference[post][0]), 'last_reference_sample': int(reference[post][-1])},
            'coverage': {'raw_correct_candidate_frames': int((post & raw_cov).sum()),
                         'grouped_correct_candidate_frames': int((post & group_cov).sum()),
                         'raw_correct_candidate_absent_frames': int((post & ~raw_cov).sum()),
                         'grouping_lost_correct_candidate_frames': int((post & raw_cov & ~group_cov).sum()),
                         'no_raw_candidate_frames': int((post & ~raw_mask.any(axis=-1)).sum()),
                         'raw_fraction': float(raw_cov[post].mean()), 'grouped_fraction': float(group_cov[post].mean())},
            'frozen_voicing_decisions': {'p_below_half_frames': int((post & (reference_state == 0)).sum()),
                                         'p_at_or_above_half_frames': int((post & (reference_state != 0)).sum()),
                                         'agreement_across_saved_methods': len(expected_methods)},
            'designated_seeds': outcomes}
    for value in total.values(): value['success_within_50c_all_frames'] = value['correct_pitch']/value['frames']
    recovery_summary = None
    if recovery is not None:
        recovery = recovery.resolve(); rm = read(recovery/'manifest.json')
        require(rm['status'] == 'complete' and rm['original_manifest_sha256'] == sha(run/'manifest.json')
                and rm['prediction_freeze_sha256'] == sha(run/'prediction-freeze.json')
                and rm['checkpoint_freeze_sha256'] == sha(run/'checkpoint-freeze.json'), 'Score recovery binding differs')
        scored = read(recovery/'metrics.json', rm['metrics_sha256'])
        for case, value in cases.items():
            for method, result in value['designated_seeds'].items():
                prior = scored['challenge'][case]['methods'][method]['regions']['after-observation-change']
                require(prior['frames'] == result['frames'] and prior['success_within_50c_all_frames'] == result['success_within_50c_all_frames']
                        and prior['nonvoiced_or_unavailable'] == result['counts']['unvoiced']+result['counts']['unavailable'], 'Recovered post-region scores differ')
        recovery_summary = {'path': str(recovery.relative_to(REPO)), 'status': rm['status'], 'all36_post_region_scores_agree': True}
    bind(Path(__file__))
    # Recheck bound evidence at end; this report never replaces the original failure.
    for name,digest in bound.items(): require(sha(REPO/name) == digest, 'Bound evidence changed during diagnosis')
    return {'schema': 'phenometrix-harmonic-challenge-diagnostic.v1',
            'scope': 'Descriptive postfit bottlenecks for frozen relative+acceptable-set seeds; no additional gate, fit, waveform or reference access',
            'run_path': str(run.relative_to(REPO)), 'original_run_status': manifest['status'],
            'original_run_failure': manifest.get('failure'), 'score_recovery': recovery_summary,
            'selector_margin_provenance': 'logits and margins not retained; no new selector inference performed',
            'probability_provenance': {'values_retained': False, 'binary_decision': 'state0 iff p(voiced)<0.5 under frozen selector_experiment.infer; verified across all18 saved methods',
                                       'model_weights_path': str(original.relative_to(REPO)), 'model_weights_sha256': ORIGINAL_WEIGHTS,
                                       'model_inference_repeated': False, 'calibration_established': False},
            'support_contract': {'sample_rate_hz': 48000, 'measurement_delay_samples': 960, 'support_samples': 1920,
                                 'end_grid': '2208+480*k', 'reference': 'end-960', 'common_eligible': 'reference>=3072 and reference+1536<=96000',
                                 'post_region': 'support_start=end-1920>=39840', 'correct_pitch': 'finite voiced estimate within inclusive50c of constant analytic F0',
                                 'denominator': 'all common-eligible fully-post-change frames; no voicing or candidate-coverage exclusion'},
            'chronology': {'checkpoints_frozen_at': checkpoint['checkpoints_frozen_at'], 'challenge_started_at': cm['started_at'],
                           'challenge_completed_at': cm['completed_at'], 'predictions_frozen_at': freeze['predictions_frozen_at'],
                           'evidence_boundary': 'stored timestamps and source control flow, not operating-system access logs'},
            'cases': cases, 'pooled_descriptive': total, 'bound_sha256': bound}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recovery', type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite diagnostic evidence')
    result = diagnose(args.run, args.recovery)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), 'sha256': sha(args.output), 'pooled_descriptive': result['pooled_descriptive']}, indent=2))
