"""Independent sealed-validation counts, aggregation, chronology and binding audit.

No fitting or inference. Reference contents are opened only after explicit audit
opt-in and verification of the complete prediction freeze. Completion timestamps
corroborate phase ordering; they are not an operating-system access log.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
METHODS = ('dsp', 'yin', 'continuity', 'mlp_original', 'hybrid_original')
SPEAKERS = ('F05', 'F06', 'M05', 'M06')
CONDITIONS = ('clean', 'noise10db', 'first_difference', 'clip01')
CHECKPOINT = '06110469a9a643fb16806905569ebc939db2c0bdd5b3a6a1593b31789453d10d'
CONTRACT = 'phenometrix.mlp-voicing-yin-pitch.v1'
COUNTS = ('eligible_valid_frames', 'reference_voiced_frames', 'reference_unvoiced_frames', 'excluded_invalid_reference',
          'excluded_boundary', 'reference_outside_supported_bounds', 'correct_frames', 'voicing_false_positive_count',
          'voicing_false_negative_count', 'unavailable_reference_voiced_count', 'voicing_false_negative_or_unavailable_count',
          'reference_voiced_success_count', 'unavailable_frames', 'jointly_voiced_frames', 'octave_error_count')
RATES = ('joint_error', 'reference_voiced_success_rate', 'voicing_false_positive_rate', 'voicing_false_negative_rate',
         'voicing_false_negative_or_unavailable_rate', 'measurement_coverage', 'jointly_voiced_coverage',
         'median_jointly_voiced_cents', 'p95_jointly_voiced_cents', 'octave_error_rate_reference_voiced')


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def inside(root, path):
    p = (root / path).resolve()
    require(p.is_relative_to(root.resolve()), 'Artifact escaped declared root')
    return p


def read(path): return json.loads(Path(path).read_text())


def timestamp(value):
    t = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(t.tzinfo is not None, 'Chronology requires timezone-aware timestamps')
    return t


def compare(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict), label + ': missing object')
        for key, value in expected.items():
            require(key in actual, label + ': missing ' + key)
            compare(actual[key], value, label + '/' + key)
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), label + ': numeric mismatch')
    else:
        require(actual == expected, label + ': mismatch')


def summarize(counts, errors):
    result = dict(counts)
    def ratio(a, b): return a / b if b else None
    n, v, u = (counts[k] for k in ('eligible_valid_frames', 'reference_voiced_frames', 'reference_unvoiced_frames'))
    sorted_errors = sorted(errors)
    result.update(joint_error=ratio(n-counts['correct_frames'], n),
        reference_voiced_success_rate=ratio(counts['reference_voiced_success_count'], v),
        voicing_false_positive_rate=ratio(counts['voicing_false_positive_count'], u),
        voicing_false_negative_rate=ratio(counts['voicing_false_negative_count'], v),
        voicing_false_negative_or_unavailable_rate=ratio(counts['voicing_false_negative_or_unavailable_count'], v),
        measurement_coverage=ratio(n-counts['unavailable_frames'], n), jointly_voiced_coverage=ratio(counts['jointly_voiced_frames'], v),
        median_jointly_voiced_cents=sorted_errors[math.ceil(.5*len(sorted_errors))-1] if sorted_errors else None,
        p95_jointly_voiced_cents=sorted_errors[math.ceil(.95*len(sorted_errors))-1] if sorted_errors else None,
        octave_error_rate_reference_voiced=ratio(counts['octave_error_count'], v), calibration_n=0, brier_score=None,
        reliability_bins=[dict(lower=i/10, upper=(i+1)/10, n=0, mean_probability=None, observed_correct_fraction=None) for i in range(10)])
    return result


def independent_counts(reference_rows, sample_count, predictions):
    """Reconstruct every unconditional count directly, without the production scorer."""
    samples = [p['sample'] for p in predictions]
    require(len(set(samples)) == len(samples), 'Duplicate prediction target')
    by_sample = dict(zip(samples, predictions))
    for p in predictions:
        state, hz = p['state'], p['f0_hz']
        require(state in ('voiced', 'unvoiced', 'unavailable'), 'Unknown prediction state')
        require((state == 'voiced' and isinstance(hz, (int, float)) and math.isfinite(hz) and hz > 0)
                or (state != 'voiced' and hz is None), 'Prediction state/frequency mismatch')
        require(p.get('correctness_probability') is None, 'Unexpected calibrated-correctness claim')
    counts, errors = dict.fromkeys(COUNTS, 0), []
    targets = set()
    for index, row in enumerate(reference_rows):
        require(len(row) == 4, 'Provider reference must have four columns')
        hz, probability, _, _ = row
        sample = 768 + 480 * index; targets.add(sample)
        valid = all(math.isfinite(x) for x in row) and ((probability == 0 and hz == 0) or (probability == 1 and hz > 0))
        if not valid:
            counts['excluded_invalid_reference'] += 1; continue
        if sample < 3072 or sample + 1536 > sample_count:
            counts['excluded_boundary'] += 1; continue
        voiced = probability == 1
        p = by_sample.get(sample, dict(state='unavailable', f0_hz=None)); state = p['state']
        counts['eligible_valid_frames'] += 1
        counts['reference_voiced_frames' if voiced else 'reference_unvoiced_frames'] += 1
        counts['unavailable_frames'] += state == 'unavailable'
        if voiced:
            counts['reference_outside_supported_bounds'] += not 50 <= hz <= 700
            counts['voicing_false_negative_count'] += state == 'unvoiced'
            counts['unavailable_reference_voiced_count'] += state == 'unavailable'
            counts['voicing_false_negative_or_unavailable_count'] += state != 'voiced'
            if state == 'voiced':
                signed = 1200 * math.log2(p['f0_hz'] / hz); error = abs(signed)
                errors.append(error); counts['jointly_voiced_frames'] += 1
                counts['reference_voiced_success_count'] += error <= 50
                counts['correct_frames'] += error <= 50
                octave = round(signed / 1200)
                counts['octave_error_count'] += octave != 0 and abs(signed - octave * 1200) <= 50
        else:
            counts['voicing_false_positive_count'] += state == 'voiced'
            counts['correct_frames'] += state == 'unvoiced'
    return summarize(counts, errors), errors, len(set(samples) - targets)


def aggregate(parts):
    counts = {k: sum(metric[k] for metric, _ in parts) for k in COUNTS}
    errors = [e for _, values in parts for e in values]
    return summarize(counts, errors)


def macro(cells):
    values = {k: [c[k] for c in cells if c[k] is not None] for k in RATES}
    return {'cell_count': len(cells), 'defined_cells': {k: len(v) for k, v in values.items()},
            **{k: sum(v)/len(v) if v else None for k, v in values.items()}}


def independent_bootstrap(records):
    """Different array layout: draw paired prompt indices, then aggregate each cell."""
    prompt_order = {pair: sorted({r['prompt'] for r in records if r['pair'] == pair}) for pair in (5, 6)}
    require(all(len(v) == 40 for v in prompt_order.values()), 'Bootstrap requires 40 prompts per pair')
    lookup = {(r['speaker'], r['condition'], r['prompt']): r['metrics'] for r in records}
    draws = np.random.default_rng(20261022).integers(0, 40, size=(2000, 2, 40))
    values = np.zeros((2000, len(METHODS)))
    for pair_index, pair in enumerate((5, 6)):
        for sex in ('F', 'M'):
            for condition in CONDITIONS:
                for method_index, method in enumerate(METHODS):
                    cells = [lookup[f'{sex}{pair:02}', condition, p][method] for p in prompt_order[pair]]
                    n = np.array([c['eligible_valid_frames'] for c in cells])
                    if np.any(n <= 0):
                        return {'status': 'unavailable', 'reason': 'At least one selected recording has no eligible valid reference; no recording omitted',
                                'used_for_selection': False}
                    errors = n - np.array([c['correct_frames'] for c in cells])
                    indices = draws[:, pair_index, :]
                    values[:, method_index] += errors[indices].sum(axis=1)/n[indices].sum(axis=1)/16
    delta = values[:, :-1] - values[:, -1:]
    bounds = np.percentile(delta, [2.5, 97.5], axis=0)
    return {'seed': 20261022, 'draws': 2000, 'used_for_selection': False,
            'joint_error_advantage_intervals': {m: dict(lower=float(bounds[0, i]), upper=float(bounds[1, i])) for i, m in enumerate(METHODS[:-1])}}


def gate(overall, clean):
    target = overall['hybrid_original']; classical = METHODS[:3]
    if any(overall[m]['defined_cells'][k] != 16 for m in METHODS for k in ('joint_error', 'reference_voiced_success_rate')):
        return {'pass': False, 'status': 'incomplete_required_cell_denominators'}
    reductions = {m: overall[m]['joint_error'] - target['joint_error'] for m in classical}
    loss = max(overall[m]['reference_voiced_success_rate'] for m in classical) - target['reference_voiced_success_rate']
    excess = clean['hybrid_original']['joint_error'] - min(clean[m]['joint_error'] for m in classical)
    passed = all(v >= .01 for v in reductions.values()) and loss <= .02 and excess <= .01
    return dict(status='quality_components_pass_pending_other_gates' if passed else 'quality_components_fail',
                **{'pass': passed}, comparators=list(classical), joint_error_reduction_by_comparator=reductions,
                voiced_success_loss_against_best_simple=loss, clean_joint_error_excess_against_best_simple=excess)


def phase_chronology(execution, freeze_path, run, expected_freeze, runner_path, times, reference_times):
    records = {}
    research = runner_path.parent.parent / 'research/voice-prosody'
    for phase in ('predict','acquire-ref','score'):
        path = execution / (phase+'.json'); row = read(path); log = execution / (phase+'.log')
        require(row.get('schema') == 'phenometrix-local-serialized-natural-phase.v1'
                and row.get('status') == 'complete' and row.get('exit_code') == 0
                and row.get('phase') == phase and row.get('compute') == 'local'
                and row.get('freeze_sha256') == expected_freeze and row.get('runner_sha256') == sha(runner_path)
                and row.get('log_sha256') == sha(log), 'Outer phase identity/log failure: '+phase)
        expected = [str(research/'.venv/bin/python'), '-m', 'phenometrix_prosody.hybrid_validation', phase,
                    '--freeze', str(freeze_path.relative_to(research)), '--output', str(run.relative_to(research))]
        require(row['command'] == expected and Path(row['cwd']).resolve() == research.resolve(), 'Outer phase command changed')
        start, end = timestamp(row['started_at']), timestamp(row['finished_at'])
        require(start <= end, 'Outer phase time reversed')
        records[phase] = dict(start=start,end=end,sha256=sha(path),log_sha256=sha(log))
    predict, acquire, score = (records[p] for p in ('predict','acquire-ref','score'))
    t_start, t_frozen, t_complete = times
    first_ref, last_ref = min(reference_times), max(reference_times)
    require(predict['start'] <= t_start <= t_frozen <= predict['end'] <= acquire['start']
            <= first_ref <= last_ref <= acquire['end'] <= score['start'] <= t_complete <= score['end'],
            'REF command launch or completion order violates prediction freeze')
    return {phase: dict(started_at=row['start'].isoformat(),finished_at=row['end'].isoformat(),
                        record_sha256=row['sha256'],log_sha256=row['log_sha256']) for phase,row in records.items()}


def audit(run, freeze_path, data_root, expected_freeze_sha256, *, allow_sealed_reference_audit=False, repository=REPO, execution=None):
    require(allow_sealed_reference_audit, 'Explicit sealed-reference audit opt-in required')
    require(execution is not None, 'Serialized phase execution evidence required')
    require(sha(freeze_path) == expected_freeze_sha256, 'Final freeze differs from externally recorded identity')
    freeze, final, frozen = read(freeze_path), read(run/'manifest.json'), read(run/'prediction-freeze.json')
    require(freeze['sealed_test_authorized'] is True and freeze['model_sha256'] == CHECKPOINT, 'Wrong sealed authorization/model')
    paths = {k: inside(repository, v) for k, v in freeze['paths'].items()}
    for key in ('inventory', 'configuration', 'protocol', 'preaccess_audit', 'preaccess_acquisition_manifest'):
        require(sha(paths[key]) == freeze[key+'_sha256'], 'Changed frozen ' + key)
    for name, digest in freeze['source_sha256'].items():
        require(sha(inside(repository, name)) == digest, 'Changed frozen source: '+name)
        require(sha(inside(freeze_path.parent/'source-snapshot', name)) == digest, 'Changed frozen source snapshot: '+name)
    for path, field in ((paths['model_directory']/'weights.pt', 'model_sha256'),
                        (paths['model_directory']/'manifest.json', 'model_manifest_sha256'),
                        (paths['training_features']/'manifest.json', 'training_manifest_sha256'),
                        (paths['training_features']/'acquisition-manifest.json', 'training_acquisition_manifest_sha256')):
        require(sha(path) == freeze[field], 'Changed model/training provenance')
    config = read(paths['configuration']); preaccess = read(paths['preaccess_audit'])
    require(config['speakers'] == list(SPEAKERS) and config['conditions'] == list(CONDITIONS)
            and config['methods'] == list(METHODS) and config['per_speaker'] == 40
            and config['transform_split_namespace'] == 'sealed-test' and config['estimator_contract'] == CONTRACT,
            'Unexpected selected experiment')
    microphones = [r for r in config['selected_inventory_records'] if r['role'] == 'mic']
    expected_records = [(m, c) for m in microphones for c in CONDITIONS]
    require(len(microphones) == 160 and len({m['recording_id'] for m in microphones}) == 160
            and len(frozen['recordings']) == len(expected_records) == 640, 'Incomplete selected recording set')
    compare(preaccess['selected_recordings'], [{k: m[k] for k in ('recording_id', 'speaker', 'pair', 'prompt', 'split')} for m in microphones], 'preaccess selection')
    require(preaccess['status'] == 'pass' and preaccess['audio_or_reference_content_opened'] is False, 'Preaccess audit absent')
    require(frozen['status'] == 'predictions_frozen' and final['status'] == 'complete'
            and frozen['references_acquired_or_read_before_prediction_freeze'] is False
            and final['recordings'] == frozen['recordings'] and final['freeze_sha256'] == frozen['freeze_sha256'] == expected_freeze_sha256,
            'Missing complete reference-blind prediction freeze')
    require(frozen['source_sha256'] == freeze['source_sha256'] and frozen['configuration_sha256'] == freeze['configuration_sha256']
            and frozen['model_sha256'] == CHECKPOINT and frozen['methods'] == list(METHODS), 'Prediction freeze identity changed')
    require(sha(run/'provenance/freeze.json') == expected_freeze_sha256
            and sha(run/'provenance/configuration.json') == freeze['configuration_sha256']
            and sha(run/'provenance/protocol.md') == freeze['protocol_sha256'], 'Prediction provenance snapshots changed')
    mic_file, scoring_file = run/'provenance/microphone-acquisition-manifest.json', run/'provenance/scoring-acquisition-manifest.json'
    require(sha(mic_file) == frozen['microphone_manifest_sha256']
            and sha(scoring_file) == final['scoring_acquisition_manifest_sha256'], 'Acquisition provenance snapshot changed')
    mic_rows = [r for r in read(mic_file)['files'] if r['split'] == 'sealed_test']
    score_rows = [r for r in read(scoring_file)['files'] if r['split'] == 'sealed_test']
    require(len(mic_rows) == 160 and all(r['role'] == 'mic' for r in mic_rows), 'References present before prediction freeze')
    lookup = {(r['recording_id'], r['role']): r for r in score_rows}
    require(len(lookup) == len(score_rows) == 320, 'Unexpected/duplicate sealed acquisition')
    expected = {(r['recording_id'], r['role']): r for r in config['selected_inventory_records']}
    require(set(lookup) == set(expected), 'Reference/microphone selection changed')
    mic_lookup = {r['recording_id']: r for r in mic_rows}
    t_freeze, t_start, t_pred, t_complete = map(timestamp, (freeze['frozen_at'], frozen['started_at'], frozen['predictions_frozen_at'], final['completed_at']))
    require(t_freeze <= t_start <= t_pred <= t_complete, 'Experiment chronology reversed')
    first_ref_completion = None
    reference_times = []
    for key, row in lookup.items():
        for field, value in expected[key].items(): require(row.get(field) == value, 'Acquired identity differs from frozen inventory')
        downloaded = timestamp(row['downloaded_at'])
        if row['role'] == 'mic':
            require(row == mic_lookup[row['recording_id']] and t_freeze <= downloaded <= t_start, 'MIC provenance/chronology changed')
        else:
            require(t_pred <= downloaded <= t_complete, 'REF download completion precedes prediction freeze')
            first_ref_completion = min(first_ref_completion, downloaded) if first_ref_completion else downloaded
            reference_times.append(downloaded)
    phases = phase_chronology(execution, freeze_path, run, expected_freeze_sha256, repository/'.local/run-hybrid-natural-stage.py',
                              (t_start,t_pred,t_complete),reference_times)
    # Validate every prediction and feature before opening even one reference file.
    payloads, aligned_frames = {}, 0
    for record, (metadata, condition) in zip(frozen['recordings'], expected_records):
        name, source = record['recording_id'], metadata['recording_id']
        require(name == source+'__'+condition and record['source_recording_id'] == source
                and record['condition'] == condition and record['speaker'] == metadata['speaker']
                and record['pair'] == metadata['pair'] and record['prompt'] == metadata['prompt']
                and record['microphone_sha256'] == lookup[source, 'mic']['sha256'], 'Prediction selection/order changed')
        feature_path, prediction_path = inside(run, record['feature_path']), inside(run, record['prediction_path'])
        require(sha(feature_path) == record['feature_sha256'] and sha(prediction_path) == record['prediction_sha256'], 'Changed feature/prediction bytes')
        with np.load(feature_path, allow_pickle=False) as f:
            ends, n = f['reference_sample'].copy(), int(f['pcm_sample_count'])
            require(int(f['sample_rate_hz']) == 48000 and n == record['sample_count']
                    and np.array_equal(ends, np.arange(2208, n+1, 480)) and f['features'].shape == (len(ends), 57)
                    and f['candidate_hz'].shape == f['candidate_mask'].shape == (len(ends), 6)
                    and np.isfinite(f['features']).all(), 'Acquisition-only feature grid/shape changed')
        with gzip.open(prediction_path, 'rt') as stream: raw = json.load(stream)
        require(raw['recording_id'] == name and raw['availability_sample'] == ends.tolist()
                and set(raw['predictions']) == set(METHODS), 'Prediction identity/methods changed')
        targets = (ends-960).tolist()
        require(all([p['sample'] for p in rows] == targets for rows in raw['predictions'].values())
                and len(raw['yin_evidence']) == len(ends), 'Target grid or YIN evidence changed')
        for index, (model, hybrid, yin) in enumerate(zip(raw['predictions']['mlp_original'], raw['predictions']['hybrid_original'], raw['yin_evidence'])):
            probability, hz = model['voicing_probability'], yin['raw_f0_hz']
            require(isinstance(probability, (int, float)) and math.isfinite(probability) and 0 <= probability <= 1
                    and yin['sample'] == int(ends[index]), 'Invalid hybrid input evidence')
            valid = isinstance(hz, (int, float)) and math.isfinite(hz) and 50 <= hz <= 700
            state = 'unvoiced' if probability < .5 else 'voiced' if valid else 'unavailable'
            require(hybrid['state'] == state and hybrid['f0_hz'] == (hz if state == 'voiced' else None)
                    and hybrid['voicing_probability'] == probability and hybrid['correctness_probability'] is None
                    and hybrid['estimator_contract'] == CONTRACT, 'Hybrid rule differs from frozen head/raw YIN')
        payloads[name] = raw; aligned_frames += len(ends)
    require(sha(run/'metrics.json') == final['metrics_sha256'], 'Reported metrics changed')
    results = read(run/'metrics.json')
    require(results['prediction_freeze_sha256'] == sha(run/'prediction-freeze.json') and results['freeze_sha256'] == expected_freeze_sha256
            and results['production_promotion'] is False, 'Metrics provenance/qualification changed')
    reported = {r['recording_id']: r for r in results['per_record']}
    require(len(reported) == len(results['per_record']) == 640, 'Missing or duplicate reported metric rows')
    refs, groups, independently_scored, comparisons = {}, defaultdict(list), [], 0
    for record in frozen['recordings']:
        name, source, speaker, condition = (record[k] for k in ('recording_id', 'source_recording_id', 'speaker', 'condition'))
        if source not in refs:
            # All prediction hashes/grids have passed before opening any reserved reference.
            for role in ('mic', 'ref'):
                entry = lookup[source, role]
                require(sha(inside(data_root, entry['path'])) == entry['sha256'], 'Acquired bytes differ from frozen source identity')
            path = inside(data_root, lookup[source, 'ref']['path'])
            refs[source] = [tuple(map(float, line.split())) for line in path.read_text().splitlines() if line.strip()]
        actual = reported[name]
        compare(actual, {k: record[k] for k in ('source_recording_id','speaker','pair','prompt','condition','sample_count')}, 'reported record identity')
        require(actual['reference_sha256'] == lookup[source, 'ref']['sha256'], 'Reported reference identity changed')
        per_method = {}
        for method in METHODS:
            metric, errors, outside = independent_counts(refs[source], record['sample_count'], payloads[name]['predictions'][method])
            compare(actual['metrics'][method], metric, name+'/'+method)
            require(actual['predictions_outside_reference_extent'][method] == outside, 'Outside-reference prediction count changed')
            groups[method, speaker, condition].append((metric, errors)); per_method[method] = metric; comparisons += 1
        independently_scored.append({**{k: record[k] for k in ('speaker', 'pair', 'prompt', 'condition')}, 'metrics': per_method})
    cells = {m: {s: {c: aggregate(groups[m,s,c]) for c in CONDITIONS} for s in SPEAKERS} for m in METHODS}
    condition = {m: {c: macro([cells[m][s][c] for s in SPEAKERS]) for c in CONDITIONS} for m in METHODS}
    speaker = {m: {s: macro(list(cells[m][s].values())) for s in SPEAKERS} for m in METHODS}
    pair = {m: {str(p): macro([cells[m][f'{sex}{p:02}'][c] for sex in ('F','M') for c in CONDITIONS]) for p in (5,6)} for m in METHODS}
    overall = {m: macro([cells[m][s][c] for s in SPEAKERS for c in CONDITIONS]) for m in METHODS}
    for field, value in (('per_speaker_condition',cells),('per_condition_equal_speaker',condition),('per_speaker_equal_condition',speaker),
                         ('per_pair_equal_speaker_condition',pair),('equal_speaker_equal_condition',overall)):
        compare(results[field], value, field)
    pooled = {m: {k: sum(cells[m][s][c][k] for s in SPEAKERS for c in CONDITIONS) for k in COUNTS} for m in METHODS}
    for counts in pooled.values():
        counts['joint_error_count'] = counts['eligible_valid_frames'] - counts['correct_frames']
        counts['voiced_pitch_error_count'] = counts['jointly_voiced_frames'] - counts['reference_voiced_success_count']
    compare(results['pooled_counts'], pooled, 'pooled counts')
    pair_advantage = {str(p): {m: pair[m][str(p)]['joint_error'] - pair['hybrid_original'][str(p)]['joint_error'] for m in METHODS[:-1]} for p in (5,6)}
    ranges = {m: {'minimum': min(pair_advantage[str(p)][m] for p in (5,6)), 'maximum': max(pair_advantage[str(p)][m] for p in (5,6))} for m in METHODS[:-1]}
    compare(results['per_pair_joint_error_advantage'], pair_advantage, 'pair effects'); compare(results['pair_advantage_ranges'], ranges, 'pair ranges')
    quality = gate(overall, {m: condition[m]['clean'] for m in METHODS})
    compare(results['quality_components_against_classical'], quality, 'quality gate')
    compare(results['joint_error_change_vs_own_mlp'], overall['hybrid_original']['joint_error']-overall['mlp_original']['joint_error'], 'own MLP difference')
    uncertainty = independent_bootstrap(independently_scored)
    compare(results['conditional_uncertainty'], uncertainty, 'conditional prompt bootstrap')
    samples = {r['source_recording_id']: r['sample_count'] for r in frozen['recordings']}
    totals = dict(source_recording_count=len(samples), condition_record_count=len(frozen['recordings']), unique_source_sample_count=sum(samples.values()),
                  unique_source_seconds=sum(samples.values())/48000, condition_sample_count=4*sum(samples.values()), condition_seconds=4*sum(samples.values())/48000)
    compare(results['acquisition_totals'], totals, 'duration accounting')
    return dict(schema='phenometrix-hybrid-sealed-independent-audit.v1', status='passed', audit_source_sha256=sha(__file__),
        freeze_sha256=expected_freeze_sha256, prediction_freeze_sha256=sha(run/'prediction-freeze.json'), metrics_sha256=sha(run/'metrics.json'),
        verified_source_bindings=len(freeze['source_sha256']), verified_source_recordings=len(refs), verified_condition_records=len(frozen['recordings']),
        verified_exact_acquisition_target_frames=aligned_frames, independent_record_method_count_checks=comparisons,
        independently_reconstructed_gate=quality, independently_reconstructed_conditional_uncertainty=uncertainty,
        serialized_phase_records=phases,
        chronology=dict(protocol_frozen_at=freeze['frozen_at'], predictions_started_at=frozen['started_at'], predictions_frozen_at=frozen['predictions_frozen_at'],
                        earliest_reference_download_completed_at=first_ref_completion.isoformat(), scoring_completed_at=final['completed_at'],
                        reference_absent_from_complete_prediction_input_manifest=True),
        limitation='Saved bytes, serialized command starts and completion timestamps corroborate the frozen phase guards; not an OS access log or network first-byte trace. Counts use the unchanged nominal provider reference, not manually established physiological truth. Prompt intervals condition on four speakers; no population, calibration or production qualification. No new inference or PCM transformation was performed.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run','freeze','data','execution','output'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--freeze-sha256', required=True); p.add_argument('--allow-sealed-reference-audit', action='store_true')
    args = p.parse_args()
    result = audit(args.run.resolve(), args.freeze.resolve(), args.data.resolve(), args.freeze_sha256,
                   allow_sealed_reference_audit=args.allow_sealed_reference_audit, execution=args.execution.resolve())
    with args.output.open('x') as f: json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps({'status':result['status'], 'count_checks':result['independent_record_method_count_checks'], 'output':str(args.output)}))


if __name__ == '__main__': main()
