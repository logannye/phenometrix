"""One frozen TRAIN-only carrier-diversity acquisition; never fit or evaluate.

Historical sources are reused without mutation. A support pass is descriptive
TRAIN feasibility and cannot trigger schedules, inference, or model fitting.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time
import numpy as np

from . import evidence_cases as old
from . import expanded_candidate_features as expansion
from .expanded_selector_model import prepare_expanded_selector_batch

REPO, ROOT, RUNS = old.REPO, old.ROOT, old.RUNS
PLAN = ROOT / 'FREQUENCY_DIVERSITY_PLAN.md'
BASELINE = RUNS / 'evidence-factorial-v2'
BASELINE_HASHES = {
    'manifest.json': 'f586e51633e3d784e13d8065412ba55926fd344bc900d5f76672d7603fe3322d',
    'independent-audit.json': '6b79e7c0dcd29f2117a3bcb6f985da482c6fa6ffeb73239192a52920812109d5',
    'feasibility/independent-audit.json': '25ce98cabcb0e48e6fdd04acf35bb6a00adc0d2ad973aa0da04e5636b73483e6',
    'source-freeze.json': 'b4a584b204b513c4c7721117fce0e4e3562db0f644df14e5f25f39688881613b',
    'feasibility/manifest.json': '8090a5cc19452e1c69066b24f7900a8a59d1800e77c9cb2de80487b40a0fdfab',
    'feasibility/report.json': 'f09a7874d1f9e7d6c72266ea7c64c2c980c0b8c0a170bbc8a6af026e33fb0a12',
    'feasibility/frame-table.npz': '49441f16c76ba1d3e68801a83169a9c5e64b4c4b6b5124323937a3378efc9c59',
    'feasibility/acquisition/manifest.json': '1d62a8441f36d82868eaf4d3153aabd916a434ef6784f6b04c23314978809223',
    'feasibility/acquisition/spec.json': 'a996267bd62d5443039ab92e56ffd72298d4736c88362360a798eea5621f631b',
}
EXTRA_BASELINE_HASHES = {
    'evidence-factorial-v1/manifest.json': '208a9549c32c93283d6225546340cc53ac16f29d1b4f6fe1ad491f70831fec25',
    'evidence-factorial-v1/source-freeze.json': 'd498c4400dd1b5b707436e034b0901aef44b1880ba3357f6e4ec2359da0c0c8a',
    'overnight-20261023/repair-v1-to-v2.json': '759dd150be4b7fb2ab8cbcf4fca1cc3a0f3522dfa6dc0291f6381cd5d84c8cb2',
}
SCHEMA = 'phenometrix-frequency-diversity-study.v1'
SPEC_SCHEMA = 'phenometrix-frequency-diversity-train-spec.v1'
REPORT_SCHEMA = 'phenometrix-frequency-diversity-report.v1'
TABLE_SCHEMA = 'phenometrix-frequency-diversity-training-frame-table.v1'
FREEZE_SCHEMA = 'phenometrix-frequency-diversity-source-freeze.v1'
MAX_SECONDS, CHILD_SECONDS = 1800, 60
MAX_OUTPUT_BYTES, MIN_FREE_BYTES = 8*1024**3, 16*1024**3
MAX_PYTHON_RSS, MAX_NODE_RSS = 16*1024**3, 2*1024**3
FLOOR = dict(rows=512, distinct_cases=24, distinct_carriers=6, distinct_phases=2, distinct_gains=2)
BINS = (50., 150., 250., 350., 450., 550., 650., 700.)
STRATA = ('added_only', 'counterexample', 'broad')
CASE_COUNT, FRAMES_PER_CASE, COMMON_PER_CASE = 1536, 196, 191
COVERAGE_FIELDS = ('original_raw_covered', 'expanded_raw_covered', 'original_group_covered', 'expanded_group_covered')


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024*1024), b''): h.update(part)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def now(): return datetime.now(timezone.utc).isoformat()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def relative(path): return str(Path(path).resolve().relative_to(REPO.resolve()))


def training_spec():
    records = []
    phases = ((.17, -.59, 1.23), (1.71, .43, -1.11))
    for i in range(256):
        if i % 4 == 0: continue
        microhz = 53_113_000 + 2_531_250*i
        for j, phase in enumerate(phases):
            for k, gain in enumerate((.035, .075)):
                for m, after in enumerate(((0., 1., .5), (.2, 1., .5))):
                    records.append(dict(id=f'frequency-diversity-train-f{i}-p{j}-g{k}-m{m}',
                        frequency_hz=microhz/1_000_000, frequency_microhz=microhz,
                        frequency_index=i, phase_index=j, gain_index=k, mixture_index=m,
                        phase_radians=list(phase), gain=gain, before=[1., .5, .25], after=list(after),
                        sample_rate_hz=48000, sample_count=96000, change_sample=39840))
    frequencies = {r['frequency_hz'] for r in records}
    baseline = {r['frequency_hz'] for r in old.training_spec()['records']}
    reserved = {r['frequency_hz'] for r in old.challenge_spec()['records']}
    lattice = {(53_113_000+2_531_250*i)/1_000_000 for i in range(256) if i % 4 == 0}
    require(lattice == baseline and not frequencies & baseline and not frequencies & reserved,
        'Nested TRAIN lattice or exact old/reserved carrier exclusion differs; no filtering allowed')
    require(len(records) == CASE_COUNT and len(frequencies) == 192 and all(50 <= f <= 700 for f in frequencies),
        'Fixed new TRAIN inventory differs')
    return dict(schema=SPEC_SCHEMA, split='train', records=records, sequence_count=CASE_COUNT,
        frequency_lattice_microhz=dict(start=53_113_000, step=2_531_250, size=256, include='i%4!=0'),
        sample_rate_hz=48000, sample_count=96000, support_samples=1920, hop_samples=480,
        measurement_delay_ms=20, acquisition_end_grid='2208+480*k samples',
        reference_relation='exclusive_support_end-960', common_mask='reference>=3072 and reference+1536<=sample_count',
        acquisition_rows=CASE_COUNT*FRAMES_PER_CASE, common_eligible_rows=CASE_COUNT*COMMON_PER_CASE,
        regions_per_case={'before-observation-change':75, 'observation-boundary':4, 'after-observation-change':112},
        duration_seconds_total=CASE_COUNT*2, primary_population='new1536 only',
        unchanged_floor=FLOOR.copy(), frequency_bin_edges_hz=list(BINS),
        baseline512='separately bound historical comparison only', combined2048='descriptive only; not primary gate',
        forbidden_operations=['sampling schedules','fitting','model inference','natural references','evaluation generation'],
        novelty='No exact carrier collision with old TRAIN or reserved fresh carriers; not disjoint frequency support or independent evaluation',
        stop='Always terminal after support report, whether support passes or fails')


def source_hashes():
    frozen_path = BASELINE/'source-freeze.json'
    require(sha(frozen_path) == BASELINE_HASHES['source-freeze.json'], 'Historical source freeze changed')
    frozen = json.loads(frozen_path.read_text())
    hashes = dict(frozen['source_sha256'])
    require(len(hashes) == 58, 'Expected exact58 historical scientific source bindings')
    for name, expected in hashes.items(): require(sha(REPO/name) == expected, 'Historical source changed: '+name)
    for path in (Path(__file__).resolve(), ROOT/'tests/test_frequency_diversity.py', PLAN,
                 ROOT/'phenometrix_prosody/audit_frequency_diversity.py', ROOT/'tests/test_audit_frequency_diversity.py',
                 ROOT/'phenometrix_prosody/audit_evidence_feasibility.py', ROOT/'phenometrix_prosody/audit_expanded_pool.py',
                 *(ROOT/'phenometrix_prosody'/name for name in ('audit_augmentation_experiment.py',
                     'audit_candidate_coverage.py','audit_harmonic_experiment.py','audit_harmonic_inference.py',
                     'audit_selector_experiment.py','audit_subharmonic_experiment.py'))):
        require(path.is_file(), 'New module/test/plan must exist before freezing')
        hashes[relative(path)] = sha(path)
    return hashes


def baseline_hashes():
    result = {}
    for name, expected in BASELINE_HASHES.items():
        path = BASELINE/name
        require(sha(path) == expected, 'Historical baseline artifact changed: '+name)
        result[relative(path)] = expected
    for name, expected in EXTRA_BASELINE_HASHES.items():
        path = RUNS/name
        require(sha(path) == expected, 'Historical failed/repair artifact changed: '+name)
        result[relative(path)] = expected
    return result


def recheck(hashes):
    for name, expected in hashes.items(): require(sha(REPO/name) == expected, 'Frozen bytes changed: '+name)


def support_proof(strata, case_index, records):
    indices = np.asarray(case_index)
    require(indices.ndim == 1 and indices.dtype.kind in 'iu' and np.all((indices >= 0) & (indices < len(records))),
        'Invalid contiguous case identity')
    require(set(strata) == set(STRATA) and all(np.asarray(v).dtype == np.bool_ and np.asarray(v).shape == indices.shape for v in strata.values()),
        'Stratum shape/type differs')
    require(not np.any(strata['added_only'] & strata['counterexample'])
        and np.all(~strata['added_only'] | strata['broad']) and np.all(~strata['counterexample'] | strata['broad']),
        'Stratum relationships differ')
    result = {}
    for name in STRATA[:2]:
        selected = np.flatnonzero(strata[name]); unique = np.unique(indices[selected]); metadata = [records[int(i)] for i in unique]
        counts = dict(rows=len(selected), distinct_cases=len(unique),
            distinct_carriers=len({r['frequency_hz'] for r in metadata}),
            distinct_phases=len({tuple(r['phase_radians']) for r in metadata}), distinct_gains=len({r['gain'] for r in metadata}))
        checks = {key:counts[key] >= bound for key,bound in FLOOR.items()}
        carriers = sorted({r['frequency_hz'] for r in metadata})
        case_rows = np.bincount(indices[selected], minlength=len(records))
        frequency_rows = {str(f):sum(int(case_rows[i]) for i,r in enumerate(records) if r['frequency_hz'] == f) for f in carriers}
        phase_rows = {str(j):int(sum(case_rows[i] for i,r in enumerate(records) if r['phase_index'] == j)) for j in sorted({r['phase_index'] for r in records})}
        gain_rows = {str(g):int(sum(case_rows[i] for i,r in enumerate(records) if r['gain'] == g)) for g in sorted({r['gain'] for r in records})}
        result[name] = dict(**counts, checks=checks, pass_result=all(checks.values()),
            frequency_bin_rows=np.histogram([records[int(i)]['frequency_hz'] for i in indices[selected]], BINS)[0].tolist(),
            phase_rows=phase_rows, gain_rows=gain_rows, carrier_hz=carriers, carrier_gaps_hz=np.diff(carriers).tolist(), carrier_rows=frequency_rows,
            maximum_carrier_row_share=max(frequency_rows.values())/len(selected) if len(selected) else None)
    return dict(pass_result=bool(all(v['pass_result'] for v in result.values())), floor=FLOOR.copy(),
        strata=result, broad_rows=int(strata['broad'].sum()), primary_population='new1536 only',
        terminal_policy='No schedules, fitting, model inference or evaluation even if support passes')


def classify_record(original, expanded, provenance, record, case_index):
    timing = old._timing(original); n = len(original['features'])
    require(n == FRAMES_PER_CASE and type(case_index) is int and 0 <= case_index < CASE_COUNT, 'TRAIN case/frame extent differs')
    for k in ('reference_sample','sample_rate_hz','pcm_sample_count'):
        require(np.array_equal(original[k], expanded[k]), 'Expanded acquisition support changed')
    require(str(np.asarray(provenance['candidate_policy']).item()) == expansion.POLICY and not np.asarray(provenance['old_frequency_evicted']).any(),
        'Candidate policy or original retention differs')
    grouped = {name:prepare_expanded_selector_batch(f['features'],f['candidate_hz'],f['candidate_mask'])
        for name,f in (('original',original),('expanded',expanded))}
    truth = np.full(n, record['frequency_hz'], np.float64); added = np.asarray(provenance['new_full40_hz'], np.float64)
    masks = old.stratum_masks(original['candidate_hz'],original['candidate_mask'],expanded['candidate_hz'],expanded['candidate_mask'],
        grouped['original'].candidate_hz.numpy(),grouped['original'].candidate_mask.numpy(),
        grouped['expanded'].candidate_hz.numpy(),grouped['expanded'].candidate_mask.numpy(),added,truth,timing.common_eligible)
    distinct = np.isfinite(added) & ~(original['candidate_mask'] & (original['candidate_hz'] == added[:,None])).any(1)
    require(np.array_equal(distinct, provenance['new_frequency_added']), 'Distinct added peak differs')
    scenario = old.previous._scenario(record)
    table = dict(case_index=np.full(n,case_index,np.int64), availability_sample=timing.availability_sample.copy(),
        reference_sample=timing.reference_sample.copy(), truth_hz=truth, common_eligible=timing.common_eligible.copy(),
        region=np.array([old.challenge_region_at(scenario,int(end)) for end in timing.availability_sample]),
        added_hz=added.copy(), new_frequency_added=distinct.copy(),
        old_full40_hz=np.asarray(provenance['old_full40_hz']).copy(),
        old_full40_confidence=np.asarray(provenance['old_full40_confidence']).copy(),
        new_full40_confidence=np.asarray(provenance['new_full40_confidence']).copy())
    for arm,f in (('original',original),('expanded',expanded)):
        for name,field in (('features','features'),('hz','candidate_hz'),('mask','candidate_mask')):
            table[f'{arm}_raw_{name}'] = f[field].copy()
            table[f'{arm}_group_{name}'] = getattr(grouped[arm],field).numpy().copy()
        for rep in ('raw','group'):
            table[f'{arm}_{rep}_covered'] = old._coverage(table[f'{arm}_{rep}_hz'],table[f'{arm}_{rep}_mask'],truth)
    for name, mask in masks.items(): table['stratum_'+name] = mask
    return table


def support_report(table, records):
    common = table['common_eligible']; strata = {k:table['stratum_'+k] for k in STRATA}
    def counts(mask, part=table):
        return dict(frames=int(mask.sum()), **{k:int((mask & part[k]).sum()) for k in COVERAGE_FIELDS},
            new_frequency_added=int((mask & part['new_frequency_added']).sum()),
            **{a+'_grouping_lost':int((mask & part[a+'_raw_covered'] & ~part[a+'_group_covered']).sum()) for a in ('original','expanded')},
            **{k:int((mask & part['stratum_'+k]).sum()) for k in STRATA[:2]})
    rows = []; regions = sorted(set(table['region']))
    for i,r in enumerate(records):
        lo,hi=i*FRAMES_PER_CASE,(i+1)*FRAMES_PER_CASE
        require(np.all(table['case_index'][lo:hi] == i), 'Per-case reporting requires ordered fixed-size table')
        part={k:v[lo:hi] for k,v in table.items()}; selected=part['common_eligible']
        rows.append(dict(case_index=i, **{k:r[k] for k in ('id','frequency_hz','frequency_index','phase_index','gain_index','mixture_index')},
            coverage=counts(selected,part), regions={region:counts(selected & (part['region'] == region),part) for region in regions}))
    return dict(schema=REPORT_SCHEMA, table_schema=TABLE_SCHEMA, feasibility=support_proof(strata,table['case_index'],records),
        case_count=len(records), acquisition_rows=len(common), common_eligible_rows=int(common.sum()), coverage=counts(common),
        per_region={region:counts(common & (table['region']==region)) for region in regions},
        per_case=rows, schedules_created=0, fits=0, model_inference_performed=False, evaluation_generated=False,
        claim_boundary='TRAIN candidate support only; no model accuracy, calibration, runtime or advancement claim')


def validate_table(table, records):
    n = len(records)*FRAMES_PER_CASE; ends = np.tile(np.arange(2208,96001,480),len(records))
    vectors = {'case_index','availability_sample','reference_sample','truth_hz','common_eligible','region','added_hz',
        'new_frequency_added','old_full40_hz','old_full40_confidence','new_full40_confidence',*COVERAGE_FIELDS,*('stratum_'+s for s in STRATA)}
    matrices = {f'{a}_{r}_{f}' for a in ('original','expanded') for r in ('raw','group') for f in ('features','hz','mask')}
    require(set(table) == vectors|matrices and all(table[k].shape == (n,) for k in vectors),'Frame table schema/extent differs')
    require(all(table[k].dtype == np.bool_ for k in ('common_eligible','new_frequency_added',*COVERAGE_FIELDS,*('stratum_'+s for s in STRATA))), 'Boolean table fields required')
    require(table['truth_hz'].dtype == np.float64 and np.isfinite(table['truth_hz']).all(), 'Finite float64 TRAIN truth required')
    require(np.array_equal(table['case_index'],np.repeat(np.arange(len(records)),196)), 'Contiguous case order differs')
    require(np.array_equal(table['availability_sample'],ends) and np.array_equal(table['reference_sample'],ends-960),'Acquisition/reference grid differs')
    common = (ends-960 >= 3072) & (ends-960+1536 <= 96000)
    require(table['common_eligible'].dtype == np.bool_ and np.array_equal(table['common_eligible'],common),'Common denominator differs')
    require(np.array_equal(table['truth_hz'],np.repeat([r['frequency_hz'] for r in records],196)),'TRAIN truth differs')
    require(np.array_equal(table['region'],np.where(ends<=39840,'before-observation-change',np.where(ends-1920>=39840,'after-observation-change','observation-boundary'))),'Region support differs')
    for arm,k in (('original',6),('expanded',7)):
        for rep in ('raw','group'):
            x,hz,mask = (table[f'{arm}_{rep}_{v}'] for v in ('features','hz','mask'))
            require(x.shape == (n,8*k+9) and x.dtype == np.float32 and np.isfinite(x).all(),'Feature matrix differs')
            require(hz.shape == mask.shape == (n,k) and hz.dtype == np.float64 and mask.dtype == np.bool_
                and np.array_equal(np.isfinite(hz),mask) and np.isnan(hz[~mask]).all(),'Candidate matrix differs')
            require(np.array_equal(table[f'{arm}_{rep}_covered'],old._coverage(hz,mask,table['truth_hz'])),'Saved coverage differs')
    masks = old.stratum_masks(table['original_raw_hz'],table['original_raw_mask'],table['expanded_raw_hz'],table['expanded_raw_mask'],
        table['original_group_hz'],table['original_group_mask'],table['expanded_group_hz'],table['expanded_group_mask'],table['added_hz'],table['truth_hz'],common)
    require(all(np.array_equal(table['stratum_'+s],masks[s]) for s in STRATA),'Saved strata differ')


def _rss(kind):
    value = resource.getrusage(kind).ru_maxrss
    return int(value if sys.platform == 'darwin' else value*1024)


def budget(output, started, deadline, manifest, worker=None):
    left = min(MAX_SECONDS-(time.monotonic()-started), (datetime.fromisoformat(deadline)-datetime.now(timezone.utc)).total_seconds())
    require(left > 0, 'Fixed30-minute scientific deadline reached')
    manifest['peak_observed_python_rss_bytes'] = max(manifest['peak_observed_python_rss_bytes'],_rss(resource.RUSAGE_SELF))
    manifest['peak_observed_completed_child_rss_bytes'] = max(manifest['peak_observed_completed_child_rss_bytes'],_rss(resource.RUSAGE_CHILDREN))
    if worker is not None:
        worker.timeout_seconds = min(CHILD_SECONDS,left)
        manifest['peak_observed_peak_worker_rss_bytes'] = max(manifest['peak_observed_peak_worker_rss_bytes'],worker.peak_rss_bytes)
    require(manifest['peak_observed_python_rss_bytes'] <= MAX_PYTHON_RSS,'Python16GiB RSS limit exceeded')
    require(max(manifest['peak_observed_completed_child_rss_bytes'],manifest['peak_observed_peak_worker_rss_bytes']) <= MAX_NODE_RSS,'Node2GiB observed RSS limit exceeded')
    require(shutil.disk_usage(output).free >= MIN_FREE_BYTES,'Minimum16GiB free disk prerequisite failed')
    size = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    manifest['observed_output_bytes'] = size
    require(size <= MAX_OUTPUT_BYTES,'New output8GiB limit exceeded')
    return left


def preserve_child_output(error, output, bind):
    """Retain bounded captured subprocess failure evidence, never discard silently."""
    result={}
    for name in ('stdout','stderr'):
        value=getattr(error,name,None)
        if value is None and name=='stdout':value=getattr(error,'output',None)
        if value is None:continue
        raw=value.encode('utf-8',errors='replace') if isinstance(value,str) else bytes(value)
        kept=raw[:65536];path=output/('child-failure-'+name+'.txt');path.write_bytes(kept)
        result[name]=dict(path=relative(path),sha256=bind(path),captured_bytes=len(raw),retained_bytes=len(kept),
            truncated=len(raw)>len(kept),full_captured_sha256=hashlib.sha256(raw).hexdigest())
    return result


def supervisor_identity():
    require(os.environ.get('NODE_OPTIONS') == '--max-old-space-size=1536', 'Exact1536MiB Node heap limit required')
    name=os.environ.get('PHENOMETRIX_FREQUENCY_SUPERVISOR_PATH','')
    expected=os.environ.get('PHENOMETRIX_FREQUENCY_SUPERVISOR_SHA256','')
    path=Path(name).resolve()
    require(name and path.is_relative_to(RUNS.resolve()) and path.is_file() and len(expected)==64,
        'Bound external supervisor required before generation')
    require(sha(path)==expected, 'External supervisor bytes changed')
    return {relative(path):expected}


def run(output):
    output = Path(output).resolve()
    require(output.is_relative_to(RUNS.resolve()) and output != RUNS.resolve() and not output.exists(),'Fresh ignored run output required; no overwrite/retry')
    require(output.parent.exists() and shutil.disk_usage(output.parent).free >= MIN_FREE_BYTES,'Parent/free-disk prerequisite failed')
    spec = training_spec(); frozen = source_hashes(); baseline = baseline_hashes(); supervisor = supervisor_identity()
    started = time.monotonic(); frozen_at = datetime.now(timezone.utc); deadline = (frozen_at+timedelta(seconds=MAX_SECONDS)).isoformat()
    freeze = dict(schema=FREEZE_SCHEMA,status='sources_frozen',frozen_at=frozen_at.isoformat(),deadline_utc=deadline,
        maximum_seconds=MAX_SECONDS, source_sha256=frozen, supervisor_sha256=supervisor, baseline_artifact_sha256=baseline, training_spec_sha256=canonical_sha(spec),
        maximum_fits=0, maximum_schedules=0, evaluation_generation_allowed=False, natural_content_allowed=False)
    output.mkdir()
    manifest = dict(schema=SCHEMA,status='sources_frozen',started_at=now(),deadline_utc=deadline,
        source_sha256=frozen,baseline_artifact_sha256=baseline,training_spec_sha256=canonical_sha(spec),
        source_freeze_path=relative(output/'source-freeze.json'),source_freeze_sha256=None,supervisor_sha256=supervisor,
        command=[sys.executable,'-m','phenometrix_prosody.frequency_diversity','--output',str(output)],
        pid=os.getpid(),node_options=os.environ.get('NODE_OPTIONS'),phase_events=[],
        input_sha256={},records=[],generated_configurations=0,expanded_configurations=0,
        fits=0,schedules_created=0,model_inference_performed=False,natural_content_read=False,evaluation_generated=False,
        maximum_seconds=MAX_SECONDS,maximum_output_bytes=MAX_OUTPUT_BYTES,
        peak_observed_python_rss_bytes=0,peak_observed_completed_child_rss_bytes=0,peak_observed_peak_worker_rss_bytes=0)
    def bind(path,expected=None):
        h = sha(path);require(expected is None or expected == h,'Input bytes changed')
        name = relative(path);require(name not in manifest['input_sha256'] or manifest['input_sha256'][name] == h,'Previously bound input changed')
        manifest['input_sha256'][name] = h;return h
    save(output/'manifest.json',manifest)
    try:
        for folder in ('source-snapshot','acquisition/pcm','acquisition/features','acquisition/intermediate','expanded/features','expanded/provenance'):
            (output/folder).mkdir(parents=True,exist_ok=True)
        for name,h in {**frozen,**supervisor}.items():
            dest = output/'source-snapshot'/name; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes((REPO/name).read_bytes())
            require(sha(dest) == h,'Source changed during snapshot')
        save(output/'spec.json',spec);save(output/'source-freeze.json',freeze)
        manifest['source_freeze_sha256']=sha(output/'source-freeze.json')
        for name,expected in {**frozen,**supervisor}.items(): bind(output/'source-snapshot'/name,expected)
        bind(output/'spec.json');bind(output/'source-freeze.json')
        recheck(frozen);recheck(baseline);recheck(supervisor);budget(output,started,deadline,manifest)
        manifest['acquisition_started_at']=now();manifest['phase_events'].append(dict(phase='acquisition_started',at=manifest['acquisition_started_at']))
        save(output/'manifest.json',manifest)
        for i,r in enumerate(spec['records']):
            left = budget(output,started,deadline,manifest)
            pcm = output/'acquisition/pcm'/(r['id']+'.f32le');full = output/'acquisition/intermediate'/(r['id']+'.json')
            old.previous._synthesize_pcm(r).tofile(pcm)
            manifest['generated_configurations'] += 1
            original = old._extract_feature(pcm,full,min(CHILD_SECONDS,budget(output,started,deadline,manifest)))
            fp = output/'acquisition/features'/(r['id']+'.npz');np.savez_compressed(fp,**original)
            export_sha = sha(full);full.unlink()
            manifest['records'].append(dict(case_index=i,id=r['id'],start=i*196,stop=(i+1)*196,
                parameters_sha256=canonical_sha(r),scenario=old.previous._scenario(r),
                pcm_path=relative(pcm),pcm_sha256=bind(pcm),original_feature_path=relative(fp),original_feature_sha256=bind(fp),
                typescript_export_sha256=export_sha,feature_frames=196,common_eligible_frames=191))
            budget(output,started,deadline,manifest)
            if (i+1)%32 == 0:save(output/'manifest.json',manifest);print(json.dumps({'phase':'acquisition','completed':i+1,'total':CASE_COUNT}),flush=True)
        require(manifest['generated_configurations'] == len(manifest['records']) == CASE_COUNT,'Fixed generation inventory differs')
        (output/'acquisition/intermediate').rmdir();manifest['acquisition_completed_at']=now();manifest['status']='acquisition_complete'
        manifest['phase_events'].append(dict(phase='acquisition_complete',at=manifest['acquisition_completed_at']))
        save(output/'manifest.json',manifest);recheck(frozen);recheck(baseline);recheck(supervisor)
        manifest['coverage_started_at']=now();manifest['phase_events'].append(dict(phase='coverage_started',at=manifest['coverage_started_at']))
        pieces=[]
        with expansion.CoveragePeakWorker(timeout_seconds=min(CHILD_SECONDS,budget(output,started,deadline,manifest))) as worker:
            for i,(r,row) in enumerate(zip(spec['records'],manifest['records'])):
                budget(output,started,deadline,manifest,worker);pcm=REPO/row['pcm_path'];fp=REPO/row['original_feature_path']
                bind(pcm,row['pcm_sha256']);bind(fp,row['original_feature_sha256'])
                with np.load(fp,allow_pickle=False) as z:original={k:z[k].copy() for k in z.files}
                expanded,provenance=expansion.expanded_feature(original,pcm,worker)
                require(str(provenance['pcm_sha256']) == row['pcm_sha256'],'Expansion PCM identity differs')
                ep=output/'expanded/features'/(r['id']+'.npz');pp=output/'expanded/provenance'/(r['id']+'.npz')
                np.savez_compressed(ep,**expanded);np.savez_compressed(pp,**provenance)
                pieces.append(classify_record(original,expanded,provenance,r,i))
                row.update(expanded_feature_path=relative(ep),expanded_feature_sha256=bind(ep),provenance_path=relative(pp),provenance_sha256=bind(pp))
                manifest['expanded_configurations']+=1;budget(output,started,deadline,manifest,worker)
                if (i+1)%32 == 0:save(output/'manifest.json',manifest);print(json.dumps({'phase':'expansion','completed':i+1,'total':CASE_COUNT}),flush=True)
        require(manifest['expanded_configurations'] == CASE_COUNT,'Fixed expansion inventory differs')
        table={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]};del pieces
        validate_table(table,spec['records']);tp=output/'frame-table.npz';np.savez_compressed(tp,**table)
        report=support_report(table,spec['records']);rp=output/'report.json';save(rp,report)
        manifest.update(frame_table_path=relative(tp),frame_table_sha256=bind(tp),report_path=relative(rp),report_sha256=bind(rp),
            support_pass=report['feasibility']['pass_result'],decision='support_pass_no_further_operations' if report['feasibility']['pass_result'] else 'support_no_go_no_further_operations')
        manifest['coverage_completed_at']=now();manifest['phase_events'].append(dict(phase='coverage_complete',at=manifest['coverage_completed_at']))
        recheck(frozen);recheck(baseline);recheck(supervisor);recheck(manifest['input_sha256']);budget(output,started,deadline,manifest)
        manifest.update(status='complete',completed_at=now(),elapsed_seconds=time.monotonic()-started)
        save(output/'manifest.json',manifest);return manifest
    except BaseException as error:
        try:manifest['child_failure_output']=preserve_child_output(error,output,bind)
        except Exception as capture_error:manifest['child_failure_capture_error']=repr(capture_error)
        manifest.update(status='failed',failure=repr(error),failed_at=now(),elapsed_seconds=time.monotonic()-started)
        save(output/'manifest.json',manifest);raise


def timeout_signal(signum,frame): raise TimeoutError('Fixed30-minute scientific timeout')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    signal.signal(signal.SIGALRM,timeout_signal);signal.alarm(MAX_SECONDS)
    try:
        result=run(args.output);print(json.dumps({k:result[k] for k in ('status','decision','support_pass','generated_configurations','expanded_configurations','fits','schedules_created')},indent=2))
    finally:signal.alarm(0)
