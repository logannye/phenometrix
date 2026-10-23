"""One source-frozen TRAIN acquisition and fixed exposure feasibility phase.

No fit, checkpoint, model output, natural reference or prospective evaluation
input is read or produced. The generated TRAIN truth is explicit metadata used
only to classify acquisition coverage and freeze the predeclared schedules.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np

from . import evidence_cases as cases
from . import expanded_candidate_features as expansion
from .expanded_selector_model import prepare_expanded_selector_batch, source_hashes as model_source_hashes

REPO, ROOT, RUNS = cases.REPO, cases.ROOT, cases.RUNS
SCHEMA = 'phenometrix-evidence-feasibility.v1'
TABLE_SCHEMA = 'phenometrix-evidence-training-frame-table.v1'
MAX_SECONDS = 3600
MAX_BYTES = 1024**3
MAX_PYTHON_RSS = 16*1024**3
COVERAGE_FIELDS = ('original_raw_covered', 'expanded_raw_covered', 'original_group_covered', 'expanded_group_covered')


def source_hashes():
    result = {**cases.source_hashes(), **expansion.source_hashes(), **model_source_hashes()}
    path = Path(__file__).resolve(); result[str(path.relative_to(REPO))] = cases.previous._sha(path)
    return result


def _require(value, reason): cases.previous._require(value, reason)
def _sha(path): return cases.previous._sha(path)
def _save(path, value): cases.previous._save(path, value)


def _rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value*1024)


def validate_source_freeze(path, expected_sha, *, generation):
    frozen, hashes = cases._validate_freeze(path, expected_sha, challenge=False, enforce_deadline=generation)
    own = source_hashes()
    _require(all(frozen['source_sha256'].get(p) == h for p, h in own.items()),
        'Source freeze lacks the complete feasibility/expansion/grouping closure')
    hashes.update({str(REPO/p): h for p, h in own.items()})
    return frozen, hashes


def _budget(output, frozen, start, manifest, worker=None):
    left = min(cases._remaining(frozen), MAX_SECONDS-(time.monotonic()-start))
    _require(left > 0, 'TRAIN preparation elapsed-time ceiling exceeded')
    rss = _rss_bytes(); manifest['peak_observed_python_rss_bytes'] = max(manifest['peak_observed_python_rss_bytes'], rss)
    _require(rss <= MAX_PYTHON_RSS, 'TRAIN preparation Python RSS exceeded16GiB')
    if worker is not None:
        worker.timeout_seconds = min(60., left)
        manifest['peak_observed_node_rss_bytes'] = max(manifest['peak_observed_node_rss_bytes'], worker.peak_rss_bytes)
        _require(worker.peak_rss_bytes <= 2*1024**3, 'TRAIN preparation Node RSS exceeded2GiB')
    size = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    _require(size <= MAX_BYTES, 'TRAIN preparation artifact ceiling exceeded')
    manifest['observed_output_bytes'] = size
    return left


def classify_record(original, expanded, provenance, record, case_index):
    """Pure array preparation; all candidates are input-only, truth is TRAIN-only."""
    timing = cases._timing(original); n = len(original['features'])
    _require(n == 196 and isinstance(case_index, int) and 0 <= case_index < 512, 'TRAIN acquisition/case extent differs')
    for key in ('reference_sample', 'sample_rate_hz', 'pcm_sample_count'):
        _require(np.array_equal(original[key], expanded[key]), 'Expanded acquisition timing changed')
    _require(str(np.asarray(provenance['candidate_policy']).item()) == expansion.POLICY,
        'Wrong additive candidate policy')
    _require(not np.asarray(provenance['old_frequency_evicted']).any(), 'Original raw candidates were evicted')
    grouped = {}
    for name, feature in (('original', original), ('expanded', expanded)):
        batch = prepare_expanded_selector_batch(feature['features'], feature['candidate_hz'], feature['candidate_mask'],
            representation='shared-candidates')
        grouped[name] = {k: getattr(batch, k).numpy().copy() for k in ('features', 'candidate_hz', 'candidate_mask')}
    truth = np.full(n, record['frequency_hz'], np.float64)
    added = np.asarray(provenance['new_full40_hz'], np.float64)
    masks = cases.stratum_masks(original['candidate_hz'], original['candidate_mask'], expanded['candidate_hz'], expanded['candidate_mask'],
        grouped['original']['candidate_hz'], grouped['original']['candidate_mask'],
        grouped['expanded']['candidate_hz'], grouped['expanded']['candidate_mask'], added, truth, timing.common_eligible)
    distinct = np.isfinite(added) & ~(original['candidate_mask'] & (original['candidate_hz'] == added[:, None])).any(1)
    _require(np.array_equal(distinct, provenance['new_frequency_added']), 'Distinct added-peak provenance differs')
    scenario = cases.previous._scenario(record)
    table = {'case_index': np.full(n, case_index, np.int64), 'availability_sample': timing.availability_sample.copy(),
        'reference_sample': timing.reference_sample.copy(), 'truth_hz': truth, 'common_eligible': timing.common_eligible.copy(),
        'region': np.array([cases.challenge_region_at(scenario, int(end)) for end in timing.availability_sample]),
        'added_hz': added.copy(), 'new_frequency_added': distinct.copy(),
        'old_full40_hz': np.asarray(provenance['old_full40_hz']).copy(),
        'old_full40_confidence': np.asarray(provenance['old_full40_confidence']).copy(),
        'new_full40_confidence': np.asarray(provenance['new_full40_confidence']).copy()}
    for name, feature in (('original', original), ('expanded', expanded)):
        for key, field in (('features', 'features'), ('hz', 'candidate_hz'), ('mask', 'candidate_mask')):
            table[f'{name}_raw_{key}'] = feature[field].copy()
            table[f'{name}_group_{key}'] = grouped[name][field]
        for representation in ('raw', 'group'):
            table[f'{name}_{representation}_covered'] = cases._coverage(table[f'{name}_{representation}_hz'],
                table[f'{name}_{representation}_mask'], truth)
    for name, mask in masks.items(): table['stratum_'+name] = mask
    return table


def summarize_table(table, records):
    common = table['common_eligible']; strata = {name: table['stratum_'+name] for name in cases.BATCH_COUNTS}
    proof = cases.feasibility(strata, table['case_index'])
    def counts(include):
        return {'frames': int(include.sum()), **{k: int((include & table[k]).sum()) for k in COVERAGE_FIELDS},
            'new_frequency_added': int((include & table['new_frequency_added']).sum()),
            'original_grouping_lost': int((include & table['original_raw_covered'] & ~table['original_group_covered']).sum()),
            'expanded_grouping_lost': int((include & table['expanded_raw_covered'] & ~table['expanded_group_covered']).sum()),
            'added_only': int((include & strata['added_only']).sum()), 'counterexample': int((include & strata['counterexample']).sum())}
    per_case = []
    for index, record in enumerate(records):
        selected = common & (table['case_index'] == index)
        per_case.append({'case_index': index, 'id': record['id'], 'frequency_hz': record['frequency_hz'],
            'phase_index': record['phase_index'], 'gain_index': record['gain_index'], 'mixture_index': record['mixture_index'],
            'coverage': counts(selected), 'regions': {r: counts(selected & (table['region'] == r)) for r in sorted(set(table['region']))}})
    return {'schema': SCHEMA, 'table_schema': TABLE_SCHEMA, 'feasibility': proof,
        'acquisition_rows': int(len(common)), 'common_eligible_rows': int(common.sum()), 'case_count': len(records),
        'coverage': counts(common), 'per_case': per_case, 'schedule_exposure': {},
        'inference_inputs': 'raw and grouped acquisition arrays; truth and strata are separate TRAIN supervision fields',
        'claim_boundary': 'TRAIN bank feasibility only; no model training, natural accuracy, fresh evaluation or promotion'}


def make_schedules(table, report):
    """No schedules are produced when the frozen feasibility floor fails."""
    _require(report['feasibility']['pass_result'], 'Insufficient TRAIN support: schedules and fits are forbidden')
    strata = {name: table['stratum_'+name] for name in cases.BATCH_COUNTS}; result = {}; exposure = {}
    for seed in cases.SEEDS:
        for epoch in range(12):
            key = f'seed{seed}__epoch{epoch}'; schedule = cases.targeted_schedule(strata, table['case_index'], seed, epoch)
            for field in ('indices', 'origin'): result[key+'__'+field] = schedule[field]
            item = cases.schedule_exposure(schedule, strata, table['case_index']); selected = schedule['indices'].ravel()
            _require(table['common_eligible'][selected].all(), 'Sampled a noncommon TRAIN frame')
            item['expanded_group_covered_rows'] = int(table['expanded_group_covered'][selected].sum())
            item['expanded_group_uncovered_zero_rows'] = int((~table['expanded_group_covered'][selected]).sum())
            item['per_region'] = {r: {'selected_rows': int((table['region'][selected] == r).sum()),
                'covered_rows': int(((table['region'][selected] == r) & table['expanded_group_covered'][selected]).sum())}
                for r in sorted(set(table['region']))}
            exposure[key] = item
    return result, exposure


def validate_table(table):
    """Validate the full fixed acquisition table without reading labels or models."""
    n = 512*196
    vector_fields = {'case_index', 'availability_sample', 'reference_sample', 'truth_hz', 'common_eligible',
        'region', 'added_hz', 'new_frequency_added', 'old_full40_hz', 'old_full40_confidence', 'new_full40_confidence',
        *COVERAGE_FIELDS, *('stratum_'+x for x in cases.BATCH_COUNTS)}
    matrix_fields = {f'{arm}_{rep}_{field}' for arm in ('original', 'expanded')
        for rep in ('raw', 'group') for field in ('features', 'hz', 'mask')}
    _require(set(table) == vector_fields | matrix_fields, 'Prepared table schema differs')
    _require(all(table[k].shape == (n,) for k in vector_fields), 'Prepared vector extent differs')
    _require(np.array_equal(table['case_index'], np.repeat(np.arange(512), 196)), 'Prepared case ordering differs')
    ends = np.tile(np.arange(2208, 96001, 480), 512)
    _require(np.array_equal(table['availability_sample'], ends) and np.array_equal(table['reference_sample'], ends-960),
        'Prepared timing differs')
    expected_common = (ends-960 >= 3072) & (ends-960+1536 <= 96000)
    _require(table['common_eligible'].dtype == np.bool_ and np.array_equal(table['common_eligible'], expected_common),
        'Prepared common mask differs')
    truth = np.repeat([r['frequency_hz'] for r in cases.training_spec()['records']], 196)
    _require(table['truth_hz'].dtype == np.float64 and np.array_equal(table['truth_hz'], truth), 'Prepared TRAIN truth differs')
    for arm, capacity, width in [('original', 6, 57), ('expanded', 7, 65)]:
        for rep in ('raw', 'group'):
            x, hz, mask = (table[f'{arm}_{rep}_{k}'] for k in ('features', 'hz', 'mask'))
            _require(x.shape == (n, width) and x.dtype == np.float32 and np.isfinite(x).all(), 'Prepared feature matrix differs')
            _require(hz.shape == mask.shape == (n, capacity) and hz.dtype == np.float64 and mask.dtype == np.bool_,
                'Prepared candidate matrix differs')
            _require(np.array_equal(np.isfinite(hz), mask) and np.isnan(hz[~mask]).all(), 'Prepared candidate availability differs')
            _require(((hz[mask] >= 50) & (hz[mask] <= 700)).all(), 'Prepared candidate bounds differ')
    bool_fields = {'new_frequency_added', *COVERAGE_FIELDS, *('stratum_'+x for x in cases.BATCH_COUNTS)}
    _require(all(table[k].dtype == np.bool_ for k in bool_fields), 'Prepared Boolean field dtype differs')
    _require(int(table['common_eligible'].sum()) == 97792, 'Prepared row count differs')


def _relative(path): return str(Path(path).resolve().relative_to(REPO))


def prepare(output, source_freeze_path, expected_sha):
    """Generate the frozen bank once, assemble all rows, then return a bound cache."""
    source_freeze_path = Path(source_freeze_path).resolve()
    frozen, hashes = validate_source_freeze(source_freeze_path, expected_sha, generation=True)
    output = cases.previous._inside(RUNS, Path(output).resolve())
    _require(not output.exists(), 'TRAIN preparation refuses overwrite or implicit resume')
    _require(output.parent.exists(), 'TRAIN preparation parent must already exist')
    output.mkdir(); started = time.monotonic(); sources = source_hashes()
    manifest = {'schema': SCHEMA, 'status': 'preparing', 'started_at': cases.previous._now(),
        'source_freeze_path': _relative(source_freeze_path), 'source_freeze_sha256': expected_sha,
        'source_sha256': sources, 'training_spec_sha256': cases.training_spec_sha256(),
        'challenge_spec_sha256': cases.challenge_spec_sha256(), 'deadline_utc': frozen['deadline_utc'],
        'generation_calls': 0, 'reference_content_read': False, 'model_outputs_read': False,
        'fitting_performed': False, 'fresh_evaluation_generated': False, 'maximum_preparation_seconds': MAX_SECONDS,
        'maximum_output_bytes': MAX_BYTES, 'peak_observed_python_rss_bytes': 0,
        'peak_observed_node_rss_bytes': 0, 'records': [], 'input_sha256': {}}
    def bind(path, expected=None):
        path = Path(path).resolve(); digest = _sha(path)
        _require(expected is None or expected == digest, 'Preparation input bytes changed')
        key = _relative(path)
        _require(key not in manifest['input_sha256'] or manifest['input_sha256'][key] == digest, 'Preparation identity changed')
        manifest['input_sha256'][key] = digest
        return digest
    try:
        for path, expected in hashes.items(): bind(path, expected)
        for name, expected in sources.items():
            snapshot = output/'source-snapshot'/name; snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes((REPO/name).read_bytes()); bind(snapshot, expected)
        _budget(output, frozen, started, manifest); _save(output/'manifest.json', manifest)
        manifest['generation_calls'] = 1; _save(output/'manifest.json', manifest)
        rows, acquisition_hashes = cases.generate_training(output/'acquisition', source_freeze_path, expected_sha)
        for path, expected in acquisition_hashes.items(): bind(path, expected)
        acquisition_manifest = output/'acquisition/manifest.json'
        manifest.update(acquisition_manifest_path=_relative(acquisition_manifest), acquisition_manifest_sha256=bind(acquisition_manifest))
        raw_manifest = json.loads(acquisition_manifest.read_text()); raw_records = {r['id']: r for r in raw_manifest['records']}
        spec_records = cases.training_spec()['records']
        _require(len(rows) == 512 and [r['id'] for r in rows] == [r['id'] for r in spec_records], 'Fixed TRAIN case inventory differs')
        (output/'expanded/features').mkdir(parents=True); (output/'expanded/provenance').mkdir(parents=True)
        pieces = []
        with expansion.CoveragePeakWorker(timeout_seconds=min(60., _budget(output, frozen, started, manifest))) as worker:
            for index, (row, record) in enumerate(zip(rows, spec_records)):
                _budget(output, frozen, started, manifest, worker)
                _require(row['case_index'] == index and row['record'] == record, 'TRAIN parameter identity differs')
                entry = raw_records[row['id']]; pcm = output/'acquisition'/entry['pcm_path']; original_path = output/'acquisition'/entry['feature_path']
                bind(pcm, entry['pcm_sha256']); bind(original_path, entry['feature_sha256'])
                expanded, provenance = expansion.expanded_feature(row['feature'], pcm, worker)
                _require(str(provenance['pcm_sha256'].item()) == entry['pcm_sha256'], 'Expanded PCM identity differs')
                expanded_path = output/'expanded/features'/(row['id']+'.npz'); provenance_path = output/'expanded/provenance'/(row['id']+'.npz')
                np.savez_compressed(expanded_path, **expanded); np.savez_compressed(provenance_path, **provenance)
                table = classify_record(row['feature'], expanded, provenance, record, index); pieces.append(table)
                manifest['records'].append({'case_index': index, 'id': row['id'], 'start': index*196, 'stop': (index+1)*196,
                    'scenario': cases.previous._scenario(record), 'parameters_sha256': entry['parameters_sha256'], 'pcm_path': _relative(pcm), 'pcm_sha256': entry['pcm_sha256'],
                    'original_feature_path': _relative(original_path), 'original_feature_sha256': entry['feature_sha256'],
                    'expanded_feature_path': _relative(expanded_path), 'expanded_feature_sha256': bind(expanded_path),
                    'provenance_path': _relative(provenance_path), 'provenance_sha256': bind(provenance_path)})
                _budget(output, frozen, started, manifest, worker)
                if (index+1) % 16 == 0 or index == 511:
                    _save(output/'manifest.json', manifest)
                    print(json.dumps({'phase': 'TRAIN_candidate_preparation', 'completed_cases': index+1, 'total_cases': 512}), flush=True)
        table = {key: np.concatenate([p[key] for p in pieces]) for key in pieces[0]}; del pieces, rows
        _require(len(table['case_index']) == 100352 and int(table['common_eligible'].sum()) == 97792, 'Full TRAIN row denominator differs')
        table_path = output/'frame-table.npz'; np.savez_compressed(table_path, **table)
        manifest.update(frame_table_path=_relative(table_path), frame_table_sha256=bind(table_path), table_schema=TABLE_SCHEMA)
        report = summarize_table(table, spec_records); manifest['feasibility_pass'] = report['feasibility']['pass_result']
        if manifest['feasibility_pass']:
            schedules, exposure = make_schedules(table, report); report['schedule_exposure'] = exposure
            schedule_path = output/'schedules.npz'; np.savez_compressed(schedule_path, **schedules)
            manifest.update(schedules_path=_relative(schedule_path), schedules_sha256=bind(schedule_path))
        report_path = output/'report.json'; _save(report_path, report)
        manifest.update(report_path=_relative(report_path), report_sha256=bind(report_path))
        _budget(output, frozen, started, manifest)
        validate_source_freeze(source_freeze_path, expected_sha, generation=True)
        _require(source_hashes() == sources, 'Preparation source changed during execution')
        for path, digest in manifest['input_sha256'].items(): _require(_sha(REPO/path) == digest, 'Bound preparation input changed')
        manifest.update(status='complete', completed_at=cases.previous._now(), elapsed_seconds=time.monotonic()-started,
            decision='feasible_for_declared_fits' if manifest['feasibility_pass'] else 'no_fits_insufficient_fixed_support')
        _save(output/'manifest.json', manifest)
    except BaseException as error:
        manifest.update(status='failed', failed_at=cases.previous._now(), failure=repr(error), elapsed_seconds=time.monotonic()-started)
        _save(output/'manifest.json', manifest)
        raise
    return load_preparation(output, source_freeze_path, expected_sha)


def load_preparation(output, source_freeze_path, expected_sha):
    """Read/hash-verify completed TRAIN preparation, including a valid no-fit result."""
    output = cases.previous._inside(RUNS, Path(output).resolve()); source_freeze_path = Path(source_freeze_path).resolve()
    frozen, hashes = validate_source_freeze(source_freeze_path, expected_sha, generation=False)
    manifest_path = output/'manifest.json'; manifest = json.loads(manifest_path.read_text())
    _require(manifest.get('schema') == SCHEMA and manifest.get('status') == 'complete'
        and manifest.get('source_freeze_path') == _relative(source_freeze_path) and manifest.get('source_freeze_sha256') == expected_sha
        and manifest.get('source_sha256') == source_hashes() and manifest.get('training_spec_sha256') == cases.training_spec_sha256()
        and manifest.get('challenge_spec_sha256') == cases.challenge_spec_sha256()
        and manifest.get('deadline_utc') == frozen['deadline_utc']
        and manifest.get('reference_content_read') is False and manifest.get('model_outputs_read') is False
        and manifest.get('generation_calls') == 1 and manifest.get('fitting_performed') is False
        and manifest.get('fresh_evaluation_generated') is False, 'Completed source-bound preparation required')
    for path, digest in manifest['input_sha256'].items():
        p = cases.previous._inside(REPO, path); _require(_sha(p) == digest, 'Bound preparation bytes changed'); hashes[str(p)] = digest
    expected_records = cases.training_spec()['records']
    _require([r['id'] for r in manifest['records']] == [r['id'] for r in expected_records], 'Prepared case inventory differs')
    for index, (row, record) in enumerate(zip(manifest['records'], expected_records)):
        _require(row['case_index'] == index and row['start'] == index*196 and row['stop'] == (index+1)*196
            and row['scenario'] == cases.previous._scenario(record), 'Prepared record index/scenario differs')
        for prefix in ('pcm', 'original_feature', 'expanded_feature', 'provenance'):
            _require(manifest['input_sha256'].get(row[prefix+'_path']) == row[prefix+'_sha256'],
                'Prepared record lacks input identity binding')
    def read_npz(field):
        p = cases.previous._inside(REPO, manifest[field+'_path'])
        _require(p.is_relative_to(output) and _sha(p) == manifest[field+'_sha256'], 'Prepared array binding differs')
        hashes[str(p)] = manifest[field+'_sha256']
        with np.load(p, allow_pickle=False) as z: return {k: z[k].copy() for k in z.files}
    table = read_npz('frame_table')
    validate_table(table)
    p = cases.previous._inside(REPO, manifest['report_path'])
    _require(p.is_relative_to(output) and _sha(p) == manifest['report_sha256'], 'Prepared report binding differs')
    report = json.loads(p.read_text()); hashes[str(p)] = manifest['report_sha256']
    proof = cases.feasibility({k: table['stratum_'+k] for k in cases.BATCH_COUNTS}, table['case_index'])
    _require(proof == report['feasibility'] and proof['pass_result'] == manifest['feasibility_pass'], 'Prepared support decision differs')
    schedules = read_npz('schedules') if manifest['feasibility_pass'] else None
    if schedules is not None:
        expected, exposure = make_schedules(table, report)
        _require(set(schedules) == set(expected) and all(np.array_equal(schedules[k], v) for k, v in expected.items()),
            'Prepared sampling schedule differs')
        _require(exposure == report['schedule_exposure'], 'Prepared exposure accounting differs')
    if not manifest['feasibility_pass']:
        _require('schedules_path' not in manifest and 'schedules_sha256' not in manifest and not (output/'schedules.npz').exists(),
            'Insufficient-support preparation cannot include sampling schedules')
    hashes[str(manifest_path)] = _sha(manifest_path)
    return {'manifest': manifest, 'arrays': table, 'strata': {k: table['stratum_'+k] for k in cases.BATCH_COUNTS},
        'feasibility': proof, 'report': report, 'schedules': schedules, 'hashes': hashes}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--source-freeze', type=Path, required=True)
    parser.add_argument('--expected-source-freeze-sha256', required=True)
    args = parser.parse_args(); result = prepare(args.output, args.source_freeze, args.expected_source_freeze_sha256)
    print(json.dumps({'status': result['manifest']['status'], 'feasibility_pass': result['feasibility']['pass_result'],
        'decision': result['manifest']['decision'], 'report': result['manifest']['report_path']}, indent=2))
