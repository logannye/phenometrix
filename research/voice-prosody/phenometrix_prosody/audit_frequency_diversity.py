"""Independent saved TRAIN-only frequency-diversity audit; never synthesizes or fits.

The new producer is not imported. Earlier independent candidate-union/grouping
and coverage equations are reused. Numeric carrier identities, support floors,
metadata ordering, and report arithmetic are reconstructed here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from . import audit_evidence_feasibility as independent
from .audit_expanded_pool import validate_expansion, independent_groups

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO/'research/voice-prosody'
RUNS = ROOT/'runs'
FLOOR = dict(independent.FLOOR)
BINS = independent.BINS
STRATA = independent.STRATA
AUDIT_DEPENDENCIES = ('audit_augmentation_experiment','audit_candidate_coverage','audit_evidence_feasibility',
    'audit_expanded_pool','audit_harmonic_experiment','audit_harmonic_inference','audit_selector_experiment','audit_subharmonic_experiment')
PHASES = ((.17, -.59, 1.23), (1.71, .43, -1.11))
GAINS = (.035, .075)
RESERVED = (109.9, 241.1, 379.3, 563.9, 619.7, 677.9)
BASE = 'research/voice-prosody/runs/evidence-factorial-v2'
BASELINE = {
    BASE+'/source-freeze.json': 'b4a584b204b513c4c7721117fce0e4e3562db0f644df14e5f25f39688881613b',
    BASE+'/manifest.json': 'f586e51633e3d784e13d8065412ba55926fd344bc900d5f76672d7603fe3322d',
    BASE+'/feasibility/manifest.json': '8090a5cc19452e1c69066b24f7900a8a59d1800e77c9cb2de80487b40a0fdfab',
    BASE+'/feasibility/report.json': 'f09a7874d1f9e7d6c72266ea7c64c2c980c0b8c0a170bbc8a6af026e33fb0a12',
    BASE+'/feasibility/frame-table.npz': '49441f16c76ba1d3e68801a83169a9c5e64b4c4b6b5124323937a3378efc9c59',
    BASE+'/feasibility/independent-audit.json': '25ce98cabcb0e48e6fdd04acf35bb6a00adc0d2ad973aa0da04e5636b73483e6',
    BASE+'/feasibility/acquisition/manifest.json': '1d62a8441f36d82868eaf4d3153aabd916a434ef6784f6b04c23314978809223',
    BASE+'/feasibility/acquisition/spec.json': 'a996267bd62d5443039ab92e56ffd72298d4736c88362360a798eea5621f631b',
    'research/voice-prosody/runs/evidence-factorial-v1/source-freeze.json': 'd498c4400dd1b5b707436e034b0901aef44b1880ba3357f6e4ec2359da0c0c8a',
    BASE+'/independent-audit.json': '6b79e7c0dcd29f2117a3bcb6f985da482c6fa6ffeb73239192a52920812109d5',
    'research/voice-prosody/runs/evidence-factorial-v1/manifest.json': '208a9549c32c93283d6225546340cc53ac16f29d1b4f6fe1ad491f70831fec25',
    'research/voice-prosody/runs/overnight-20261023/repair-v1-to-v2.json': '759dd150be4b7fb2ab8cbcf4fca1cc3a0f3522dfa6dc0291f6381cd5d84c8cb2',
}
require = independent.require
sha = independent.sha
canonical_sha = independent.canonical_sha
compare = independent.compare
array_equal = independent.array_equal
load_npz = independent.load_npz
Ledger = independent.Ledger


def expected_records():
    """Integer-rational metadata enumeration, not a signal generator."""
    result = []
    for i in range(256):
        if i % 4 == 0:
            continue
        for j, phase in enumerate(PHASES):
            for k, gain in enumerate(GAINS):
                for m, after in enumerate(((0., 1., .5), (.2, 1., .5))):
                    result.append({'id':f'frequency-diversity-train-f{i}-p{j}-g{k}-m{m}',
                        'frequency_hz':(53_113_000+2_531_250*i)/1_000_000,
                        'frequency_microhz':53_113_000+2_531_250*i,
                        'phase_radians':list(phase), 'gain':gain, 'sample_rate_hz':48000,
                        'sample_count':96000, 'change_sample':39840,
                        'before':[1., .5, .25], 'after':list(after),
                        'frequency_index':i, 'phase_index':j, 'gain_index':k, 'mixture_index':m})
    return result


def validate_numeric_metadata(records):
    require(isinstance(records,list) and records, 'Missing TRAIN metadata')
    ids = set(); by_index = {}; by_value = {}
    for row in records:
        index, value = row['frequency_index'], row['frequency_hz']
        require(type(index) is int and isinstance(value,(int,float)) and not isinstance(value,bool)
            and np.isfinite(value) and 50 <= value <= 700, 'Invalid carrier metadata')
        require(row['id'] not in ids, 'Duplicate complete configuration ID'); ids.add(row['id'])
        require(index not in by_index or by_index[index] == value, 'One carrier index has multiple values')
        require(value not in by_value or by_value[value] == index, 'Multiple carrier indices duplicate numeric frequency')
        by_index[index] = value; by_value[value] = index
        require(row['phase_index'] in (0,1) and row['phase_radians'] == list(PHASES[row['phase_index']])
            and row['gain_index'] in (0,1) and row['gain'] == GAINS[row['gain_index']],
            'Phase/gain index is not its actual setting')
    return by_index


def support_report(strata, case_index, records):
    """Count actual carrier/phase/gain values; identifiers cannot create diversity."""
    validate_numeric_metadata(records)
    case_index = np.asarray(case_index)
    require(case_index.ndim == 1 and case_index.dtype.kind in 'iu'
        and np.all((case_index >= 0) & (case_index < len(records))), 'Invalid case identity')
    require(set(strata) == set(STRATA) and all(np.asarray(v).dtype == np.bool_
        and np.asarray(v).shape == case_index.shape for v in strata.values()), 'Invalid stratum extent')
    require(not np.any(strata['added_only'] & strata['counterexample']) and
        all(np.all(~strata[name] | strata['broad']) for name in STRATA[:2]), 'Stratum relationships differ')
    out = {}
    for name in STRATA[:2]:
        selected = np.flatnonzero(strata[name]); cases = np.unique(case_index[selected])
        rows = [records[int(i)] for i in cases]
        counts = {'rows':len(selected), 'distinct_cases':len(cases),
            'distinct_carriers':len({r['frequency_hz'] for r in rows}),
            'distinct_phases':len({tuple(r['phase_radians']) for r in rows}),
            'distinct_gains':len({r['gain'] for r in rows})}
        checks = {k:counts[k] >= minimum for k,minimum in FLOOR.items()}
        out[name] = {**counts, 'checks':checks, 'pass_result':all(checks.values()),
            'frequency_bin_rows':np.histogram([records[int(i)]['frequency_hz'] for i in case_index[selected]],BINS)[0].tolist()}
    return {'pass_result':all(r['pass_result'] for r in out.values()), 'floor':FLOOR.copy(),
        'strata':out, 'broad_rows':int(strata['broad'].sum()),
        'on_failure':'no fits; no enlargement or revised threshold'}


def validate_table(table, records):
    n = len(records)*196
    expected = independent.expected_support(len(records))
    vectors = set(expected) | {'truth_hz','added_hz','new_frequency_added','old_full40_hz',
        'old_full40_confidence','new_full40_confidence'} | {
        p+'_covered' for p in ('original_raw','expanded_raw','original_group','expanded_group')} | {
        'stratum_'+s for s in STRATA}
    matrices = {f'{a}_{r}_{f}' for a in ('original','expanded') for r in ('raw','group') for f in ('features','hz','mask')}
    require(set(table) == vectors | matrices, 'Full table field inventory differs')
    for name,value in expected.items(): array_equal(table[name],value,name)
    array_equal(table['truth_hz'],np.repeat([r['frequency_hz'] for r in records],196),'truth_hz')
    for name in vectors:
        require(table[name].shape == (n,), 'Table vector denominator differs: '+name)
    for name in ('added_hz','old_full40_hz','old_full40_confidence','new_full40_confidence'):
        require(table[name].dtype == np.float64, 'Peak provenance dtype differs')
    for arm,k in (('original',6),('expanded',7)):
        for rep in ('raw','group'):
            x,h,m = (table[f'{arm}_{rep}_{f}'] for f in ('features','hz','mask'))
            require(x.shape == (n,8*k+9) and x.dtype == np.float32 and np.isfinite(x).all(), 'Feature matrix differs')
            require(h.shape == m.shape == (n,k) and h.dtype == np.float64 and m.dtype == np.bool_
                and np.array_equal(np.isfinite(h),m) and np.isnan(h[~m]).all(), 'Candidate matrix differs')
    coverage,strata,distinct = independent.derive_strata(table)
    for name,value in coverage.items(): array_equal(table[name+'_covered'],value,name)
    for name,value in strata.items(): array_equal(table['stratum_'+name],value,name)
    array_equal(table['new_frequency_added'],distinct,'distinct added peak')
    return coverage,strata,distinct


def counts_for(include, coverage, strata, distinct):
    return {'frames':int(include.sum()),
        **{name+'_covered':int((include & value).sum()) for name,value in coverage.items()},
        'new_frequency_added':int((include & distinct).sum()),
        **{name+'_grouping_lost':int((include & coverage[name+'_raw'] & ~coverage[name+'_group']).sum())
            for name in ('original','expanded')},
        **{name:int((include & strata[name]).sum()) for name in STRATA[:2]}}


def independent_summary(table, records):
    coverage,strata,distinct = independent.derive_strata(table)
    common = table['common_eligible']; regions = sorted(set(table['region']))
    case_counts = []
    for i,row in enumerate(records):
        sl = slice(196*i,196*(i+1))
        require(np.all(table['case_index'][sl] == i), 'Per-case fixed-size order differs')
        take = common[sl]; c = {k:v[sl] for k,v in coverage.items()}; t = {k:v[sl] for k,v in strata.items()}; d = distinct[sl]
        case_counts.append({'case_index':i, **{k:row[k] for k in ('id','frequency_hz','phase_index','gain_index','mixture_index')},
            'coverage':counts_for(take,c,t,d),
            'regions':{region:counts_for(take & (table['region'][sl] == region),c,t,d) for region in regions}})
    return {'feasibility':support_report(strata,table['case_index'],records),
        'acquisition_rows':len(common), 'common_eligible_rows':int(common.sum()), 'case_count':len(records),
        'coverage':counts_for(common,coverage,strata,distinct), 'per_case':case_counts}


def carrier_diagnostics(summary):
    """Descriptive concentration; this cannot introduce or waive a support gate."""
    out = {}
    for name in STRATA[:2]:
        by_f = {}; phase = {'0':0,'1':0}; gain = {'0':0,'1':0}; mixture = {'0':0,'1':0}
        regions = {}
        for row in summary['per_case']:
            count = row['coverage'][name]
            if count: by_f[row['frequency_hz']] = by_f.get(row['frequency_hz'],0)+count
            phase[str(row['phase_index'])] += count; gain[str(row['gain_index'])] += count
            mixture[str(row['mixture_index'])] += count
            for region,values in row['regions'].items(): regions[region] = regions.get(region,0)+values[name]
        frequencies = sorted(by_f); total = sum(by_f.values())
        out[name] = {'qualifying_carriers_hz':frequencies,
            'rows_per_qualifying_carrier':[by_f[f] for f in frequencies],
            'adjacent_qualifying_carrier_gaps_hz':[b-a for a,b in zip(frequencies,frequencies[1:])],
            'span_hz':frequencies[-1]-frequencies[0] if frequencies else None,
            'largest_carrier_row_share':max(by_f.values())/total if total else None,
            'per_phase_rows':phase, 'per_gain_rows':gain, 'per_mixture_rows':mixture, 'per_region_rows':regions}
    return out


def combined_summary(old_summary, new_summary):
    """Sum separately audited configurations without promoting a combined gate."""
    old = old_summary['per_case']; new = new_summary['per_case']
    require(not ({r['frequency_hz'] for r in old} & {r['frequency_hz'] for r in new}), 'Old and new carriers overlap')
    coverage = {k:old_summary['coverage'][k]+new_summary['coverage'][k] for k in old_summary['coverage']}
    out = {'case_count':len(old)+len(new),
        'acquisition_rows':old_summary['acquisition_rows']+new_summary['acquisition_rows'],
        'common_eligible_rows':old_summary['common_eligible_rows']+new_summary['common_eligible_rows'],
        'coverage':coverage, 'role':'descriptive only; cannot rescue new-only failure',
        'old_case_count':len(old), 'new_case_count':len(new)}
    strata = {}
    for name in STRATA[:2]:
        qualifying = [r for r in old+new if r['coverage'][name]]
        counts = {'rows':coverage[name], 'distinct_cases':len(qualifying),
            'distinct_carriers':len({r['frequency_hz'] for r in qualifying}),
            'distinct_phases':len({r['phase_index'] for r in qualifying}),
            'distinct_gains':len({r['gain_index'] for r in qualifying})}
        checks = {k:counts[k] >= v for k,v in FLOOR.items()}
        strata[name] = {**counts,'checks':checks,'pass_result':all(checks.values())}
    out['support'] = {'strata':strata,'pass_result':all(x['pass_result'] for x in strata.values()),'floor':FLOOR.copy()}
    return out


def assert_no_dependent_outputs(run):
    run = Path(run)
    banned_dirs = {'checkpoints','predictions','challenge','fresh','natural','development','train','schedules','evidence'}
    banned_files = {'checkpoint-freeze.json','prediction-freeze.json','schedules.npz','training-targets.npz'}
    found = []
    for p in run.rglob('*'):
        parts = p.relative_to(run).parts
        if 'source-snapshot' in parts: continue
        if (p.is_dir() and p.name in banned_dirs) or (p.is_file() and (p.name in banned_files or p.suffix in ('.pt','.pth','.onnx'))):
            found.append(str(p.relative_to(run)))
    require(not found, 'Coverage-only run has dependent outputs: '+repr(found))
    return {'fits':0,'schedules':0,'learned_inference':0,'evaluation_cases':0,'natural_content_read':False}


def validate_source_closure(ledger, source_path, source, old_source):
    require(source.get('schema') == 'phenometrix-frequency-diversity-source-freeze.v1'
        and source.get('status') == 'sources_frozen', 'Initial source freeze differs')
    current = source.get('source_sha256'); old = old_source.get('source_sha256')
    require(isinstance(old,dict) and len(old) == 58 and isinstance(current,dict)
        and all(current.get(p) == h for p,h in old.items()), 'Old 58 scientific sources changed or omitted')
    required = {str(Path(__file__).resolve().relative_to(REPO)),
        'research/voice-prosody/tests/test_audit_frequency_diversity.py',
        'research/voice-prosody/phenometrix_prosody/frequency_diversity.py',
        'research/voice-prosody/tests/test_frequency_diversity.py',
        'research/voice-prosody/FREQUENCY_DIVERSITY_PLAN.md',
        str(Path(independent.__file__).resolve().relative_to(REPO)),
        str(Path(validate_expansion.__code__.co_filename).resolve().relative_to(REPO))}
    required.update('research/voice-prosody/phenometrix_prosody/'+name+'.py' for name in AUDIT_DEPENDENCIES)
    require(required <= set(current), 'Initial source closure omits producer, audit, tests or dependencies')
    for path,digest in current.items():
        ledger.bind(path,digest)
        ledger.bind(str((source_path.parent/'source-snapshot'/path).relative_to(ledger.repo)),digest)
    return len(current)


def validate_chronology(source, manifest, acquisition=None):
    if acquisition is None:
        values = {'sources_frozen_at':source.get('frozen_at'), 'started_at':manifest.get('started_at'),
            **{k:manifest.get(k) for k in ('acquisition_started_at','acquisition_completed_at',
                'coverage_started_at','coverage_completed_at','completed_at')}}
    else:
        values = {'sources_frozen_at':source.get('sources_frozen_at'), 'started_at':manifest.get('started_at'),
            'acquisition_started_at':acquisition.get('started_at'), 'acquisition_completed_at':acquisition.get('completed_at'),
            'completed_at':manifest.get('completed_at')}
    times = [independent.timestamp(v) for v in values.values()]
    require(all(a <= b for a,b in zip(times,times[1:])), 'Freeze/generation/coverage chronology differs')
    require((times[-1]-times[1]).total_seconds() <= 1800, 'Scientific wall-clock bound exceeded')
    return values


def scenario_for(record):
    return {'id':record['id'], 'kind':'observation-change', 'durationSeconds':2.,
        'initialHz':record['frequency_hz'], 'finalHz':record['frequency_hz'],
        'changeStartSeconds':.83, 'changeEndSeconds':.83, 'slopeHzPerSecond':0,
        **{k:record[k] for k in ('change_sample','before','after','phase_radians','gain')}}


def verify_record_files(ledger, run, manifest, table, records):
    """Reconstruct every saved raw union and emitted-member group, no extraction."""
    require([r['id'] for r in manifest['records']] == [r['id'] for r in records], 'Record inventory/order differs')
    verified = 0
    for i,(record,row) in enumerate(zip(records,manifest['records'])):
        sl = slice(196*i,196*(i+1))
        require(row['case_index'] == i and row['start'] == sl.start and row['stop'] == sl.stop,
            'Record slice/case identity differs')
        require(row['parameters_sha256'] == canonical_sha(record), 'Record parameter hash differs')
        compare(row['scenario'],scenario_for(record),'record scenario')
        bound = {}
        for prefix in ('pcm','original_feature','expanded_feature','provenance'):
            path = ledger.bind(row[prefix+'_path'],row[prefix+'_sha256'])
            require(path.is_relative_to(run) and manifest['input_sha256'].get(row[prefix+'_path']) == row[prefix+'_sha256'],
                'Record path escaped its bank or lacks input binding')
            bound[prefix] = path
        require(bound['pcm'].stat().st_size == 96000*4, 'PCM extent differs')
        original = load_npz(bound['original_feature']); expanded = load_npz(bound['expanded_feature'])
        provenance = load_npz(bound['provenance'])
        validate_expansion(original,expanded,provenance,row['pcm_sha256'])
        require(int(original['pcm_sample_count']) == 96000 and int(original['sample_rate_hz']) == 48000,
            'Source PCM coordinate differs')
        array_equal(original['reference_sample'],np.arange(2208,96001,480,dtype=np.int64),'acquisition endpoints')
        for arm,feature in (('original',original),('expanded',expanded)):
            group = independent_groups(feature)
            for rep,values in (('raw',feature),('group',group)):
                for dest,field in (('features','features'),('hz','candidate_hz'),('mask','candidate_mask')):
                    array_equal(table[f'{arm}_{rep}_{dest}'][sl],values[field],record['id']+'/'+arm+'/'+rep+'/'+dest)
        for dest,field in [('added_hz','new_full40_hz'),('new_frequency_added','new_frequency_added'),
            ('old_full40_hz','old_full40_hz'),('old_full40_confidence','old_full40_confidence'),
            ('new_full40_confidence','new_full40_confidence')]:
            array_equal(table[dest][sl],provenance[field],record['id']+'/'+dest)
        verified += 196
    return verified


def baseline_audit(ledger):
    paths = {p:ledger.bind(p,h) for p,h in BASELINE.items()}
    manifest = json.loads(paths[BASE+'/feasibility/manifest.json'].read_text())
    source = json.loads(paths[BASE+'/source-freeze.json'].read_text())
    run = paths[BASE+'/feasibility/manifest.json'].parent
    require(manifest['status'] == 'complete' and manifest['feasibility_pass'] is False
        and manifest['decision'] == 'no_fits_insufficient_fixed_support', 'Historical no-go was changed')
    for audit_file,pass_field in ((BASE+'/feasibility/independent-audit.json','pass'),(BASE+'/independent-audit.json','pass_result')):
        audit = json.loads(paths[audit_file].read_text())
        require(audit.get(pass_field) is True, 'Pinned historical audit is not passing')
    table = load_npz(paths[BASE+'/feasibility/frame-table.npz']); records = independent.expected_records()
    validate_table(table,records)
    expected = independent_summary(table,records)
    reported = json.loads(paths[BASE+'/feasibility/report.json'].read_text())
    for key,value in expected.items(): compare(reported[key],value,'unchanged baseline/'+key)
    frames = verify_record_files(ledger,run,manifest,table,records)
    require(frames == 100352, 'Historical frame denominator changed')
    del table
    return source,expected,{'case_count':512,'acquisition_rows':frames,'common_eligible_rows':97792,
        'regenerated_cases':0,'historical_feasibility_pass':False,'baseline_artifact_sha256':BASELINE.copy()}


def metadata_novelty(records):
    # This old helper only enumerates configured metadata. It never opens cases,
    # outcome files, natural references or any reserved evaluation PCM.
    from .evidence_cases import historical_carriers
    history = sorted(historical_carriers())
    require(canonical_sha(history) == '0c41a9a41062b78fa32748b9089fa3e7ff61024ddaaf1f16dd4799ef8fc2048d',
        'Historical configured metadata inventory changed')
    frequencies = sorted({r['frequency_hz'] for r in records})
    old = {r['frequency_hz'] for r in independent.expected_records()}
    require(not (set(frequencies)&old or set(frequencies)&set(history) or set(frequencies)&set(RESERVED)),
        'New TRAIN metadata collides with configured history or reserved carrier')
    distance,f,r = min((abs(f-r),f,r) for f in frequencies for r in RESERVED)
    return {'new_numeric_carriers':len(frequencies),'historical_configured_carriers':len(history),
        'old_train_carriers':len(old),'reserved_carriers':len(RESERVED),'exact_intersections':[],
        'nearest_reserved_distance_hz':distance,'nearest_new_carrier_hz':f,'nearest_reserved_carrier_hz':r,
        'claim':'Exact configured-carrier separation only; nearby frequencies and continuous support overlap.'}


def expected_report(table, records):
    result = independent_summary(table,records)
    coverage,strata,distinct = independent.derive_strata(table)
    proof = result['feasibility']; proof.pop('on_failure')
    proof.update(primary_population='new1536 only',
        terminal_policy='No schedules, fitting, model inference or evaluation even if support passes')
    diag = carrier_diagnostics(result)
    for name in STRATA[:2]:
        d = diag[name]
        gain = {str(g):sum(row['coverage'][name] for row in result['per_case'] if GAINS[row['gain_index']] == g) for g in GAINS}
        proof['strata'][name].update(phase_rows=d['per_phase_rows'],gain_rows=gain,
            carrier_hz=d['qualifying_carriers_hz'],carrier_gaps_hz=d['adjacent_qualifying_carrier_gaps_hz'],
            carrier_rows={str(f):n for f,n in zip(d['qualifying_carriers_hz'],d['rows_per_qualifying_carrier'])},
            maximum_carrier_row_share=d['largest_carrier_row_share'])
    for row,record in zip(result['per_case'],records): row['frequency_index'] = record['frequency_index']
    common = table['common_eligible']
    result.update(schema='phenometrix-frequency-diversity-report.v1',table_schema='phenometrix-frequency-diversity-training-frame-table.v1',
        per_region={r:counts_for(common & (table['region'] == r),coverage,strata,distinct) for r in sorted(set(table['region']))},
        schedules_created=0,fits=0,model_inference_performed=False,evaluation_generated=False,
        claim_boundary='TRAIN candidate support only; no model accuracy, calibration, runtime or advancement claim')
    return result


def validate_spec(spec):
    records = expected_records()
    compare(spec['records'],records,'independent integer-rational metadata')
    validate_numeric_metadata(records)
    required = {'schema':'phenometrix-frequency-diversity-train-spec.v1','split':'train','sequence_count':1536,
        'frequency_lattice_microhz':{'start':53_113_000,'step':2_531_250,'size':256,'include':'i%4!=0'},
        'sample_rate_hz':48000,'sample_count':96000,'support_samples':1920,'hop_samples':480,'measurement_delay_ms':20,
        'acquisition_end_grid':'2208+480*k samples','reference_relation':'exclusive_support_end-960',
        'common_mask':'reference>=3072 and reference+1536<=sample_count','acquisition_rows':301056,'common_eligible_rows':293376,
        'regions_per_case':{'before-observation-change':75,'observation-boundary':4,'after-observation-change':112},
        'duration_seconds_total':3072,'primary_population':'new1536 only','unchanged_floor':FLOOR.copy(),
        'frequency_bin_edges_hz':list(BINS),'baseline512':'separately bound historical comparison only',
        'combined2048':'descriptive only; not primary gate',
        'forbidden_operations':['sampling schedules','fitting','model inference','natural references','evaluation generation'],
        'novelty':'No exact carrier collision with old TRAIN or reserved fresh carriers; not disjoint frequency support or independent evaluation',
        'stop':'Always terminal after support report, whether support passes or fails'}
    compare({k:v for k,v in spec.items() if k != 'records'},required,'TRAIN specification contract')
    return records


def validate_terminal_declarations(source, manifest):
    compare({k:source.get(k) for k in ('maximum_fits','maximum_schedules','evaluation_generation_allowed','natural_content_allowed')},
        {'maximum_fits':0,'maximum_schedules':0,'evaluation_generation_allowed':False,'natural_content_allowed':False},'freeze scope')
    compare({k:manifest.get(k) for k in ('fits','schedules_created','model_inference_performed','natural_content_read','evaluation_generated')},
        {'fits':0,'schedules_created':0,'model_inference_performed':False,'natural_content_read':False,'evaluation_generated':False},'run scope')
    require(manifest.get('generated_configurations') == manifest.get('expanded_configurations') == 1536,
        'New bank generation/expansion extent differs')
    require(source.get('maximum_seconds') == manifest.get('maximum_seconds') == 1800
        and manifest.get('maximum_output_bytes') == 8*1024**3, 'Resource declaration differs')
    require(manifest.get('node_options') == '--max-old-space-size=1536', 'Node heap bound differs')
    require(0 < manifest.get('elapsed_seconds',0) <= 1800 and 0 < manifest.get('observed_output_bytes',0) <= 8*1024**3,
        'Observed elapsed/output bounds differ')
    require(0 < manifest.get('peak_observed_python_rss_bytes',0) <= 16*1024**3
        and 0 < manifest.get('peak_observed_completed_child_rss_bytes',0) <= 2*1024**3
        and 0 < manifest.get('peak_observed_peak_worker_rss_bytes',0) <= 2*1024**3, 'Observed process RSS bound differs')


def audit_run(run):
    started = time.monotonic(); run = Path(run).resolve()
    require(run.is_relative_to(RUNS.resolve()) and run != RUNS.resolve(), 'Audit requires an ignored local run')
    ledger = Ledger(); relative = lambda p:str(Path(p).resolve().relative_to(REPO))
    manifest_path = ledger.bind(relative(run/'manifest.json')); manifest = json.loads(manifest_path.read_text())
    require(manifest.get('schema') == 'phenometrix-frequency-diversity-study.v1' and manifest.get('status') == 'complete',
        'Completed coverage-only study required')
    source_path = ledger.bind(manifest['source_freeze_path'],manifest['source_freeze_sha256'])
    require(source_path == run/'source-freeze.json', 'Source freeze escaped the new study')
    source = json.loads(source_path.read_text())
    require(source.get('baseline_artifact_sha256') == manifest.get('baseline_artifact_sha256') == BASELINE,
        'Historical baseline exact authority map differs')
    old_source = json.loads(ledger.bind(BASE+'/source-freeze.json',BASELINE[BASE+'/source-freeze.json']).read_text())
    source_count = validate_source_closure(ledger,source_path,source,old_source)
    require(source['source_sha256'] == manifest['source_sha256'], 'Manifest source closure changed')
    require(source.get('deadline_utc') == manifest.get('deadline_utc'), 'Run deadline differs')
    validate_terminal_declarations(source,manifest)
    phases = validate_chronology(source,manifest)
    require(independent.timestamp(manifest['completed_at']) <= independent.timestamp(source['deadline_utc'])
        and (independent.timestamp(source['deadline_utc'])-independent.timestamp(source['frozen_at'])).total_seconds() == 1800,
        'Frozen deadline identity differs')
    compare(manifest['phase_events'],[{'phase':p,'at':manifest[k]} for p,k in (
        ('acquisition_started','acquisition_started_at'),('acquisition_complete','acquisition_completed_at'),
        ('coverage_started','coverage_started_at'),('coverage_complete','coverage_completed_at'))], 'phase event sequence')
    supervisor = source.get('supervisor_sha256')
    require(isinstance(supervisor,dict) and len(supervisor) == 1 and supervisor == manifest.get('supervisor_sha256'),
        'Exact supervisor source binding missing')
    for path,digest in supervisor.items():
        bound = ledger.bind(path,digest); require(bound.is_relative_to(RUNS), 'Supervisor outside ignored local runs')
        ledger.bind(relative(run/'source-snapshot'/path),digest)
    spec_path = ledger.bind(relative(run/'spec.json')); spec = json.loads(spec_path.read_text())
    require(canonical_sha(spec) == source.get('training_spec_sha256') == manifest.get('training_spec_sha256'),
        'Frozen new TRAIN metadata changed')
    records = validate_spec(spec); novelty = metadata_novelty(records)
    def artifact(key, filename):
        path = ledger.bind(manifest[key+'_path'],manifest[key+'_sha256'])
        require(path == run/filename and manifest['input_sha256'].get(relative(path)) == sha(path), 'Output lacks exact input binding')
        return path
    table = load_npz(artifact('frame_table','frame-table.npz')); validate_table(table,records)
    # The file inventory is enumerated from source/records, not a broad directory
    # glob or an untrusted ledger that could lead to natural/evaluation content.
    allowed_inputs = {relative(spec_path),relative(source_path),relative(run/'frame-table.npz'),relative(run/'report.json')}
    allowed_inputs.update(relative(run/'source-snapshot'/p) for p in source['source_sha256'])
    allowed_inputs.update(relative(run/'source-snapshot'/p) for p in supervisor)
    for row,record in zip(manifest['records'],records):
        require(row.get('feature_frames') == 196 and row.get('common_eligible_frames') == 191,
            'Saved per-case denominator differs')
        for prefix,subdir,suffix in (('pcm','acquisition/pcm','.f32le'),('original_feature','acquisition/features','.npz'),
            ('expanded_feature','expanded/features','.npz'),('provenance','expanded/provenance','.npz')):
            expected = relative(run/subdir/(record['id']+suffix))
            require(row.get(prefix+'_path') == expected, 'Record route changed: '+prefix)
            allowed_inputs.add(expected)
        require(isinstance(row.get('typescript_export_sha256'),str) and len(row['typescript_export_sha256']) == 64,
            'Original TypeScript export identity missing')
    require(set(manifest['input_sha256']) == allowed_inputs, 'Unexpected or omitted input path; natural/evaluation reads forbidden')
    for path,digest in manifest['input_sha256'].items(): ledger.bind(path,digest)
    new_frames = verify_record_files(ledger,run,manifest,table,records)
    expected = expected_report(table,records); report_path = artifact('report','report.json')
    compare(json.loads(report_path.read_text()),expected,'independent complete new-bank report')
    passed = expected['feasibility']['pass_result']
    require(type(manifest.get('support_pass')) is bool and manifest['support_pass'] == passed
        and manifest['decision'] == ('support_pass_no_further_operations' if passed else 'support_no_go_no_further_operations'),
        'Terminal new-only decision differs')
    no_dependents = assert_no_dependent_outputs(run)
    del table
    _,old_summary,baseline = baseline_audit(ledger)
    combined = combined_summary(old_summary,expected)
    ledger.recheck()
    return {'schema':'phenometrix-frequency-diversity-independent-audit.v1','status':'complete','pass':True,'pass_result':True,
        'audited_at':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-started,
        'source_freeze_path':relative(source_path),'source_freeze_sha256':sha(source_path),'source_files':source_count,
        'study_manifest_path':relative(manifest_path),'study_manifest_sha256':sha(manifest_path),
        'training_spec_sha256':source['training_spec_sha256'],'report_sha256':manifest['report_sha256'],
        'frame_table_sha256':manifest['frame_table_sha256'],'support_pass':passed,
        'new_bank':{k:expected[k] for k in ('case_count','acquisition_rows','common_eligible_rows','coverage','per_region','feasibility')},
        'new_bank_concentration':carrier_diagnostics(expected),'metadata_novelty':novelty,
        'baseline':baseline,'combined_descriptive':combined,'candidate_frame_pairs_verified':new_frames+baseline['acquisition_rows'],
        'historical_scientific_files_unchanged':58,'zero_dependent_operations':no_dependents,'phase_chronology':phases,
        'bound_artifacts':len(ledger.hashes),'bound_artifact_sha256':ledger.hashes,
        'audit_source_sha256':{p:source['source_sha256'][p] for p in [
            relative(Path(__file__)), 'research/voice-prosody/tests/test_audit_frequency_diversity.py',
            *('research/voice-prosody/phenometrix_prosody/'+name+'.py' for name in AUDIT_DEPENDENCIES)]},
        'limitations':['TRAIN candidate-support feasibility only; no learning, accuracy, calibration, runtime or promotion claim.',
            'Saved source/phase declarations and output inventory are not operating-system access logs.',
            'PCM byte identities and candidate union/grouping are checked; waveform synthesis and full raw ACF extraction are not independently replayed.',
            'Numeric carrier diversity does not establish independent frequency/acoustic support; the bank deliberately refines observed TRAIN support.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); output = args.output.resolve()
    require(output.is_relative_to(RUNS.resolve()) and not output.exists() and output.parent.is_dir(), 'Fresh ignored output required')
    try:
        result = audit_run(args.run)
    except BaseException as error:
        result = {'schema':'phenometrix-frequency-diversity-independent-audit.v1','status':'failed','pass':False,'pass_result':False,
            'failure':repr(error),'audit_source_sha256':sha(Path(__file__))}
        output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
        raise
    output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','pass_result','support_pass','candidate_frame_pairs_verified','bound_artifacts','elapsed_seconds')}))


if __name__ == '__main__': main()
