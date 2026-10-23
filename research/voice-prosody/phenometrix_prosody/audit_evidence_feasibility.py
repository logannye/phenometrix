"""Independent TRAIN feasibility arithmetic; no synthesis, fitting or inference.

Only saved analytic TRAIN inputs are read. No producer stratum, feasibility or
sampling implementation is called. Existing independent pool/grouping validators
are reused explicitly when auditing bound acquisition files.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO/'research/voice-prosody'
RUNS = ROOT/'runs'
SEEDS = (20261022, 20261023, 20261024)
ARMS = ('existing__off', 'existing__on', 'targeted__off', 'targeted__on')
STRATA = ('added_only', 'counterexample', 'broad')
DRAW_COUNTS = (128, 128, 256)
FLOOR = {'rows':512, 'distinct_cases':24, 'distinct_carriers':6,
         'distinct_phases':2, 'distinct_gains':2}
BINS = (50.,150.,250.,350.,450.,550.,650.,700.)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def expected_records():
    """Declared metadata only. This function cannot produce an audio sample."""
    result=[]
    for i in range(64):
        for j,phase in enumerate(((.17,-.59,1.23),(1.71,.43,-1.11))):
            for k,gain in enumerate((.035,.075)):
                for m,after in enumerate(((0.,1.,.5),(.2,1.,.5))):
                    result.append({'id':f'evidence-train-f{i}-p{j}-g{k}-m{m}',
                        'frequency_hz':round(53.113+10.125*i,3),'phase_radians':list(phase),'gain':gain,
                        'sample_rate_hz':48000,'sample_count':96000,'change_sample':39840,
                        'before':[1.,.5,.25],'after':list(after),
                        'frequency_index':i,'phase_index':j,'gain_index':k,'mixture_index':m})
    return result


def expected_support(case_count=512):
    ends=np.arange(2208,96001,480,dtype=np.int64)
    reference=ends-960
    common=(reference>=3072)&(reference+1536<=96000)
    region=np.where(ends<=39840,'before-observation-change',
        np.where(ends-1920>=39840,'after-observation-change','observation-boundary'))
    return {'case_index':np.repeat(np.arange(case_count,dtype=np.int64),196),
        'availability_sample':np.tile(ends,case_count),'reference_sample':np.tile(reference,case_count),
        'common_eligible':np.tile(common,case_count),'region':np.tile(region,case_count)}


def correct_coverage(hz,mask,truth):
    hz=np.asarray(hz);mask=np.asarray(mask);truth=np.asarray(truth)
    require(hz.ndim==2 and hz.shape==mask.shape and mask.dtype==np.bool_
        and truth.shape==(len(hz),),'Candidate/target extent or mask dtype differs')
    require(np.isfinite(truth).all() and np.all((truth>=50)&(truth<=700))
        and np.isfinite(hz[mask]).all() and np.all((hz[mask]>=50)&(hz[mask]<=700)),
        'Invalid available candidate or analytic TRAIN frequency')
    safe=np.where(mask,hz,truth[:,None])
    return np.any(mask & (np.abs(1200*np.log2(safe/truth[:,None]))<=50),axis=1)


def derive_strata(table):
    """Classify from candidates and TRAIN truth, never trust saved flags."""
    truth=np.asarray(table['truth_hz']);common=np.asarray(table['common_eligible'])
    added=np.asarray(table['added_hz'])
    require(common.dtype==np.bool_ and common.shape==truth.shape==added.shape,
        'TRAIN row metadata differs')
    coverage={name:correct_coverage(table[name+'_hz'],table[name+'_mask'],truth)
        for name in ('original_raw','expanded_raw','original_group','expanded_group')}
    require(not np.any(coverage['original_raw']&~coverage['expanded_raw']),
        'Original raw coverage was evicted')
    for prefix in ('original','expanded'):
        require(not np.any(coverage[prefix+'_group']&~coverage[prefix+'_raw']),
            'Grouping invented candidate coverage')
    require(not np.isinf(added).any(),'Infinite added peak is invalid')
    available=np.isfinite(added)
    require(np.all((added[available]>=50)&(added[available]<=700)),'Added peak is outside range')
    old_match=np.any(table['original_raw_mask']&(table['original_raw_hz']==added[:,None]),axis=1)
    new_match=np.any(table['expanded_raw_mask']&(table['expanded_raw_hz']==added[:,None]),axis=1)
    distinct=available&~old_match
    require(np.all(~distinct|new_match),'Distinct added peak absent from expanded pool')
    within=np.zeros(len(truth),bool)
    within[available]=np.abs(1200*np.log2(added[available]/truth[available]))<=50
    strata={'added_only':common&~coverage['original_raw']&distinct&within&coverage['expanded_group'],
        'counterexample':common&distinct&~within&coverage['original_group']&coverage['expanded_group'],
        'broad':common.copy()}
    require(not np.any(strata['added_only']&strata['counterexample']),'Target strata overlap')
    return coverage,strata,distinct


def support_report(strata,case_index,records=None):
    records=expected_records() if records is None else records
    cases=np.asarray(case_index)
    require(cases.ndim==1 and cases.dtype.kind in 'iu' and np.all((cases>=0)&(cases<len(records))),
        'Invalid case identity')
    require(set(strata)==set(STRATA) and all(np.asarray(v).shape==cases.shape
        and np.asarray(v).dtype==np.bool_ for v in strata.values()),'Invalid stratum flags')
    require(np.all(~strata['added_only']|strata['broad']) and np.all(~strata['counterexample']|strata['broad'])
        and not np.any(strata['added_only']&strata['counterexample']),'Stratum relationships differ')
    result={}
    for name in STRATA[:2]:
        selected=np.flatnonzero(strata[name]);unique=np.unique(cases[selected])
        metadata=[records[int(c)] for c in unique]
        counts={'rows':len(selected),'distinct_cases':len(unique),
            **{out:len({r[key] for r in metadata}) for out,key in
                [('distinct_carriers','frequency_index'),('distinct_phases','phase_index'),('distinct_gains','gain_index')]}}
        checks={name:counts[name]>=bound for name,bound in FLOOR.items()}
        result[name]={**counts,'checks':checks,'pass_result':all(checks.values()),
            'frequency_bin_rows':np.histogram([records[int(c)]['frequency_hz'] for c in cases[selected]],BINS)[0].tolist()}
    return {'pass_result':all(x['pass_result'] for x in result.values()),'floor':FLOOR.copy(),
        'strata':result,'broad_rows':int(strata['broad'].sum()),'on_failure':'no fits; no enlargement or revised threshold'}


def reconstruct_schedule(strata,case_index,seed,epoch):
    require(seed in SEEDS and type(epoch) is int and 0<=epoch<12,'Undeclared seed/epoch')
    require(support_report(strata,case_index)['pass_result'],'Insufficient support; no schedule may exist')
    case_index=np.asarray(case_index);rng=np.random.default_rng(seed+2_000_000+epoch)
    eligible=[]
    for name in STRATA:
        rows=np.flatnonzero(strata[name]);cases=np.unique(case_index[rows])
        eligible.append((cases,{int(c):rows[case_index[rows]==c] for c in cases}))
    indices=np.empty((33,512),np.int64);origin=np.empty((33,512),np.uint8)
    for batch in range(33):
        offset=0
        for label,(count,(cases,rows)) in enumerate(zip(DRAW_COUNTS,eligible)):
            selected_cases=rng.choice(cases,size=count,replace=True)
            for case in selected_cases:
                candidates=rows[int(case)]
                indices[batch,offset]=candidates[rng.integers(len(candidates))]
                origin[batch,offset]=label;offset+=1
        perm=rng.permutation(512)
        indices[batch]=indices[batch,perm];origin[batch]=origin[batch,perm]
    return {'indices':indices,'origin':origin}


def exposure(schedule,strata,case_index):
    order=np.asarray(schedule['indices']);origin=np.asarray(schedule['origin']);cases=np.asarray(case_index)
    require(order.shape==origin.shape==(33,512) and order.dtype==np.int64 and origin.dtype==np.uint8,
        'Schedule shape/dtype differs')
    require(np.all((order>=0)&(order<len(cases))) and np.isin(origin,[0,1,2]).all(),'Invalid scheduled index/origin')
    for label,(name,count) in enumerate(zip(STRATA,DRAW_COUNTS)):
        require(np.all((origin==label).sum(axis=1)==count) and np.all(strata[name][order[origin==label]]),
            'Scheduled draw violates requested stratum')
    flat=order.ravel();unique=len(np.unique(flat))
    return {'selected_rows':len(flat),'unique_rows':unique,'duplicate_draws':len(flat)-unique,
        'requested_strata':{name:int((origin==label).sum()) for label,name in enumerate(STRATA)},
        'actual_strata':{name:int(strata[name][flat].sum()) for name in STRATA},
        'per_case_selected_rows':np.bincount(cases[flat],minlength=512).tolist()}


def compare(actual,expected,label='value'):
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and set(actual)==set(expected),label+': field inventory differs')
        for key in expected:compare(actual[key],expected[key],label+'/'+key)
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),label+': length differs')
        for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,label+'/'+str(i))
    else:
        if isinstance(expected,bool):require(type(actual) is bool,label+': Boolean type differs')
        require(actual==expected,label+': value differs')


class Ledger:
    def __init__(self,repo=REPO):
        self.repo=Path(repo).resolve();self.hashes={}

    def bind(self,relative,expected=None):
        require(isinstance(relative,(str,Path)),'Artifact path must be text')
        path=Path(relative)
        require(not path.is_absolute(),'Artifact path must be repository-relative')
        path=(self.repo/path).resolve()
        require(path.is_relative_to(self.repo) and path.is_file(),'Artifact escaped repository or missing')
        actual=sha(path);key=str(path.relative_to(self.repo))
        require(expected is None or actual==expected,'Bound artifact identity differs: '+key)
        require(key not in self.hashes or self.hashes[key]==actual,'Artifact changed during audit: '+key)
        self.hashes[key]=actual
        return path

    def recheck(self):
        for key,expected in self.hashes.items():
            require(sha(self.repo/key)==expected,'Bound artifact changed during audit: '+key)


def validate_phase(ledger,source_path,source,fresh_path,*,feasibility_pass):
    """Check saved phase linkage, not an operating-system access history."""
    expected_names={f'{a}__seed{s}' for a in ARMS for s in SEEDS}
    require(source.get('schema')=='phenometrix-evidence-source-freeze.v1'
        and source.get('status')=='sources_frozen' and source.get('fits')=={},'Invalid initial source freeze')
    require(source.get('arms')==list(ARMS) and source.get('seeds')==list(SEEDS)
        and source.get('maximum_fits')==12,'Source freeze fit budget differs')
    require(source.get('deadline_utc')=='2026-10-23T12:18:58Z','Source freeze deadline differs')
    source_path=Path(source_path);run=source_path.parent;fresh_path=Path(fresh_path)
    weights=set(p.name for p in (run/'checkpoints').glob('*.pt'))
    if not feasibility_pass:
        require(not weights and not fresh_path.exists(),'No-go feasibility has dependent fit/fresh artifacts')
    if not fresh_path.exists():return {'fresh_present':False,'final_checkpoint_bindings_checked':0}
    checkpoint_path=ledger.bind(str((run/'checkpoint-freeze.json').relative_to(ledger.repo)))
    checkpoint=json.loads(checkpoint_path.read_text())
    require(checkpoint.get('schema')=='phenometrix-evidence-checkpoint-freeze.v1'
        and checkpoint.get('status')=='checkpoints_frozen','Fresh content lacks final checkpoint phase')
    require(checkpoint.get('source_freeze_path')==str(source_path.relative_to(ledger.repo))
        and checkpoint.get('source_freeze_sha256')==sha(source_path),'Fresh source freeze link differs')
    for key in ('source_sha256','training_spec_sha256','challenge_spec_sha256','arms','seeds','maximum_fits','deadline_utc'):
        require(checkpoint.get(key)==source.get(key),'Initial/final phase differs: '+key)
    require(set(checkpoint.get('checkpoints',{}))==expected_names and weights=={n+'.pt' for n in expected_names},
        'Fresh generation requires exactly twelve final checkpoints')
    for name,row in checkpoint['checkpoints'].items():
        require(row['path']==str((run/'checkpoints'/(name+'.pt')).relative_to(ledger.repo)),
            'Final checkpoint path differs')
        ledger.bind(row['path'],row['sha256'])
    manifest_path=ledger.bind(str((fresh_path/'manifest.json').relative_to(ledger.repo)))
    fresh=json.loads(manifest_path.read_text())
    require(fresh.get('checkpoint_freeze_path')==str(checkpoint_path.relative_to(ledger.repo))
        and fresh.get('checkpoint_freeze_sha256')==sha(checkpoint_path),'Fresh acquisition checkpoint link differs')
    return {'fresh_present':True,'final_checkpoint_bindings_checked':12}


def array_equal(actual, expected, label):
    actual=np.asarray(actual);expected=np.asarray(expected)
    require(actual.shape==expected.shape and actual.dtype==expected.dtype,
        label+': shape/dtype differs')
    require(np.array_equal(actual,expected,equal_nan=True) if actual.dtype.kind in 'fc'
        else np.array_equal(actual,expected),label+': values differ')


def table_summary(table, records=None):
    records=expected_records() if records is None else records
    coverage,strata,distinct=derive_strata(table)
    common=table['common_eligible'];regions=sorted(set(table['region']))
    def counts(include):
        return {'frames':int(include.sum()),
            **{name+'_covered':int((include&value).sum()) for name,value in coverage.items()},
            'new_frequency_added':int((include&distinct).sum()),
            **{name+'_grouping_lost':int((include&coverage[name+'_raw']&~coverage[name+'_group']).sum())
               for name in ('original','expanded')},
            **{name:int((include&strata[name]).sum()) for name in STRATA[:2]}}
    per_case=[]
    for i,r in enumerate(records):
        selected=common&(table['case_index']==i)
        per_case.append({'case_index':i,**{k:r[k] for k in ('id','frequency_hz','phase_index','gain_index','mixture_index')},
            'coverage':counts(selected),'regions':{region:counts(selected&(table['region']==region)) for region in regions}})
    return {'schema':'phenometrix-evidence-feasibility.v1','table_schema':'phenometrix-evidence-training-frame-table.v1',
        'feasibility':support_report(strata,table['case_index'],records),'acquisition_rows':len(common),
        'common_eligible_rows':int(common.sum()),'case_count':len(records),'coverage':counts(common),
        'per_case':per_case,'schedule_exposure':{},
        'inference_inputs':'raw and grouped acquisition arrays; truth and strata are separate TRAIN supervision fields',
        'claim_boundary':'TRAIN bank feasibility only; no model training, natural accuracy, fresh evaluation or promotion'}


def validate_table(table):
    expected=expected_support();n=100352
    vectors=set(expected)|{'truth_hz','added_hz','new_frequency_added','old_full40_hz',
        'old_full40_confidence','new_full40_confidence'}|{x+'_covered' for x in ('original_raw','expanded_raw','original_group','expanded_group')}|{'stratum_'+s for s in STRATA}
    matrices={f'{a}_{r}_{f}' for a in ('original','expanded') for r in ('raw','group') for f in ('features','hz','mask')}
    require(set(table)==vectors|matrices,'Full table field inventory differs')
    for name,value in expected.items():array_equal(table[name],value,name)
    array_equal(table['truth_hz'],np.repeat([r['frequency_hz'] for r in expected_records()],196),'truth_hz')
    for name in vectors:
        require(table[name].shape==(n,),'Vector row denominator differs: '+name)
    for name in ('added_hz','old_full40_hz','old_full40_confidence','new_full40_confidence'):
        require(table[name].dtype==np.float64,'Peak provenance dtype differs: '+name)
    for a,k in (('original',6),('expanded',7)):
        for r in ('raw','group'):
            x,h,m=(table[f'{a}_{r}_{f}'] for f in ('features','hz','mask'))
            require(x.shape==(n,8*k+9) and x.dtype==np.float32 and np.isfinite(x).all(),'Feature matrix differs')
            require(h.shape==m.shape==(n,k) and h.dtype==np.float64 and m.dtype==np.bool_
                and np.array_equal(np.isfinite(h),m) and np.isnan(h[~m]).all(),'Candidate matrix differs')
    coverage,strata,distinct=derive_strata(table)
    for name,value in coverage.items():array_equal(table[name+'_covered'],value,name+'_covered')
    for name,value in strata.items():array_equal(table['stratum_'+name],value,'stratum_'+name)
    array_equal(table['new_frequency_added'],distinct,'new_frequency_added')
    return coverage,strata


def schedule_report(schedule,strata,table):
    result=exposure(schedule,strata,table['case_index']);selected=schedule['indices'].ravel()
    require(table['common_eligible'][selected].all(),'Schedule includes noncommon rows')
    covered=table['expanded_group_covered'][selected]
    result.update(expanded_group_covered_rows=int(covered.sum()),expanded_group_uncovered_zero_rows=int((~covered).sum()),
        per_region={r:{'selected_rows':int((table['region'][selected]==r).sum()),
            'covered_rows':int(((table['region'][selected]==r)&covered).sum())} for r in sorted(set(table['region']))})
    return result


def timestamp(value):
    require(isinstance(value,str),'Missing phase timestamp')
    try: result=datetime.fromisoformat(value.replace('Z','+00:00'))
    except ValueError as error:raise ValueError('Malformed phase timestamp') from error
    require(result.utcoffset() is not None,'Phase timestamp lacks timezone')
    return result


def chronology(source,manifest,acquisition):
    labels=('sources_frozen_at','preparation_started_at','acquisition_started_at','acquisition_completed_at','preparation_completed_at')
    raw=(source.get('sources_frozen_at'),manifest.get('started_at'),acquisition.get('started_at'),
        acquisition.get('completed_at'),manifest.get('completed_at'))
    times=list(map(timestamp,raw))
    require(all(a<=b for a,b in zip(times,times[1:])),'Source/acquisition/preparation chronology differs')
    require(times[-1]<=timestamp(source['deadline_utc']),'Preparation completed after the declared deadline')
    return dict(zip(labels,raw))


def load_npz(path):
    with np.load(path,allow_pickle=False) as saved:return {k:saved[k].copy() for k in saved.files}


def audit_run(preparation):
    """Audit a completed bound preparation, without running its producer code."""
    from .audit_expanded_pool import validate_expansion,independent_groups
    preparation=Path(preparation).resolve()
    require(preparation.is_relative_to(RUNS.resolve()),'Preparation must be an ignored local run')
    ledger=Ledger();relative=lambda p:str(Path(p).resolve().relative_to(REPO))
    manifest_path=ledger.bind(relative(preparation/'manifest.json'));manifest=json.loads(manifest_path.read_text())
    require(manifest.get('schema')=='phenometrix-evidence-feasibility.v1' and manifest.get('status')=='complete',
        'Completed feasibility preparation required')
    require(manifest.get('generation_calls')==1 and all(manifest.get(k) is False for k in
        ('reference_content_read','model_outputs_read','fitting_performed','fresh_evaluation_generated')),
        'Preparation phase declarations differ')
    source_path=ledger.bind(manifest['source_freeze_path'],manifest['source_freeze_sha256']);source=json.loads(source_path.read_text())
    require(preparation==source_path.parent/'feasibility','Preparation is outside its source-frozen run')
    for key in ('training_spec_sha256','challenge_spec_sha256','deadline_utc'):
        require(manifest.get(key)==source.get(key),'Source/preparation identity differs: '+key)
    require(isinstance(source.get('source_sha256'),dict) and source['source_sha256'],'Missing source closure')
    for name,digest in source['source_sha256'].items():
        ledger.bind(name,digest);ledger.bind(relative(source_path.parent/'source-snapshot'/name),digest)
    for name,digest in manifest['source_sha256'].items():
        require(source['source_sha256'].get(name)==digest,'Preparation source is outside freeze')
        ledger.bind(relative(preparation/'source-snapshot'/name),digest)
    require(manifest['source_sha256'],'Empty preparation source closure')
    for name,digest in manifest['input_sha256'].items():ledger.bind(name,digest)
    def artifact(key):
        path=ledger.bind(manifest[key+'_path'],manifest[key+'_sha256'])
        require(path.is_relative_to(preparation),'Preparation artifact escaped its directory')
        require(manifest['input_sha256'].get(relative(path))==sha(path),'Artifact missing from input ledger')
        return path
    acquisition_path=artifact('acquisition_manifest');acquisition=json.loads(acquisition_path.read_text());folder=acquisition_path.parent
    require(folder==preparation/'acquisition' and acquisition.get('schema')=='phenometrix-evidence-acquisition-cache-v1'
        and acquisition.get('status')=='complete' and acquisition.get('split')=='train','Acquisition phase differs')
    for key in ('training_spec_sha256','challenge_spec_sha256','deadline_utc'):
        require(acquisition.get(key)==source.get(key),'Acquisition source identity differs: '+key)
    require(acquisition.get('freeze_kind')=='source' and acquisition.get('freeze_path')==manifest['source_freeze_path']
        and acquisition.get('freeze_sha256')==manifest['source_freeze_sha256']
        and acquisition.get('source_freeze_path')==manifest['source_freeze_path']
        and acquisition.get('source_freeze_sha256')==manifest['source_freeze_sha256']
        and acquisition.get('reference_content_read') is False and acquisition.get('model_outputs_read') is False,
        'Acquisition lacks exact initial freeze/no-model binding')
    for name,digest in acquisition['source_sha256'].items():
        require(source['source_sha256'].get(name)==digest,'Acquisition source is outside freeze')
        ledger.bind(relative(folder/'source-snapshot'/name),digest)
    spec_path=ledger.bind(relative(folder/'spec.json'));spec=json.loads(spec_path.read_text())
    require(canonical_sha(spec)==source['training_spec_sha256']==acquisition['spec_sha256'],'TRAIN specification differs')
    records=expected_records();compare(spec['records'],records,'independent TRAIN metadata')
    compare({k:spec[k] for k in ('sequence_count','common_eligible_rows','support_samples','hop_samples','measurement_delay_ms')},
        {'sequence_count':512,'common_eligible_rows':97792,'support_samples':1920,'hop_samples':480,'measurement_delay_ms':20},'TRAIN support specification')
    phase_times=chronology(source,manifest,acquisition)
    require([r['id'] for r in manifest['records']]==[r['id'] for r in records]
        and [r['id'] for r in acquisition['records']]==[r['id'] for r in records],'Case order/inventory differs')
    table=load_npz(artifact('frame_table'));coverage,strata=validate_table(table)
    for i,(record,row,rawrow) in enumerate(zip(records,manifest['records'],acquisition['records'])):
        sl=slice(i*196,(i+1)*196)
        require(row['case_index']==rawrow['case_index']==i and row['start']==sl.start and row['stop']==sl.stop,
            'Case slice/order differs')
        scenario={'id':record['id'],'kind':'observation-change','durationSeconds':2.,'initialHz':record['frequency_hz'],
            'finalHz':record['frequency_hz'],'changeStartSeconds':.83,'changeEndSeconds':.83,'slopeHzPerSecond':0,
            **{k:record[k] for k in ('change_sample','before','after','phase_radians','gain')}}
        compare(row.get('scenario'),scenario,'TRAIN scenario')
        require(row['parameters_sha256']==rawrow['parameters_sha256']==canonical_sha(record)
            and rawrow['feature_frames']==196 and rawrow['common_eligible_frames']==191,'Case parameters/denominator differs')
        bound={}
        for prefix in ('pcm','original_feature','expanded_feature','provenance'):
            bound[prefix]=ledger.bind(row[prefix+'_path'],row[prefix+'_sha256'])
            require(bound[prefix].is_relative_to(preparation)
                and manifest['input_sha256'].get(row[prefix+'_path'])==row[prefix+'_sha256'],'Case artifact is unbound')
        require(bound['pcm']==(folder/rawrow['pcm_path']).resolve() and row['pcm_sha256']==rawrow['pcm_sha256']
            and bound['original_feature']==(folder/rawrow['feature_path']).resolve()
            and row['original_feature_sha256']==rawrow['feature_sha256'],'Acquisition/prepared record mismatch')
        require(bound['pcm'].stat().st_size==96000*4,'PCM sample extent differs')
        original=load_npz(bound['original_feature']);expanded=load_npz(bound['expanded_feature']);provenance=load_npz(bound['provenance'])
        validate_expansion(original,expanded,provenance,row['pcm_sha256'])
        require(int(original['pcm_sample_count'])==96000,'Feature PCM extent differs')
        for arm,feature in (('original',original),('expanded',expanded)):
            group=independent_groups(feature)
            for rep,values in (('raw',feature),('group',group)):
                for dest,field in (('features','features'),('hz','candidate_hz'),('mask','candidate_mask')):
                    array_equal(table[f'{arm}_{rep}_{dest}'][sl],values[field],f'{record["id"]}/{arm}/{rep}/{dest}')
        for dest,field in [('added_hz','new_full40_hz'),('new_frequency_added','new_frequency_added'),
            ('old_full40_hz','old_full40_hz'),('old_full40_confidence','old_full40_confidence'),('new_full40_confidence','new_full40_confidence')]:
            array_equal(table[dest][sl],provenance[field],record['id']+'/'+dest)
    expected_report=table_summary(table);passed=expected_report['feasibility']['pass_result']
    require(type(manifest.get('feasibility_pass')) is bool and manifest['feasibility_pass']==passed,'Manifest hides support outcome')
    require(manifest.get('decision')==('feasible_for_declared_fits' if passed else 'no_fits_insufficient_fixed_support'),
        'Feasibility disposition differs')
    if passed:
        schedules=load_npz(artifact('schedules'))
        names={f'seed{s}__epoch{e}__{field}' for s in SEEDS for e in range(12) for field in ('indices','origin')}
        require(set(schedules)==names,'Schedule inventory differs')
        for seed in SEEDS:
            for epoch in range(12):
                name=f'seed{seed}__epoch{epoch}';expected=reconstruct_schedule(strata,table['case_index'],seed,epoch)
                saved={field:schedules[name+'__'+field] for field in ('indices','origin')}
                for field in saved:array_equal(saved[field],expected[field],name+'/'+field)
                expected_report['schedule_exposure'][name]=schedule_report(saved,strata,table)
    else:
        require('schedules_path' not in manifest and 'schedules_sha256' not in manifest and not (preparation/'schedules.npz').exists(),
            'Support no-go must not create schedules')
    report_path=artifact('report');compare(json.loads(report_path.read_text()),expected_report,'independent report')
    phase=validate_phase(ledger,source_path,source,source_path.parent/'challenge',feasibility_pass=passed)
    if phase['fresh_present']:
        cp=json.loads((source_path.parent/'checkpoint-freeze.json').read_text());fresh=json.loads((source_path.parent/'challenge/manifest.json').read_text())
        require(timestamp(manifest['completed_at'])<=timestamp(cp.get('frozen_at'))<=timestamp(fresh.get('started_at'))
            <=timestamp(fresh.get('completed_at'))<=timestamp(source['deadline_utc']),'Final-checkpoint/fresh chronology differs')
        require(fresh.get('schema')=='phenometrix-evidence-acquisition-cache-v1' and fresh.get('status')=='complete'
            and fresh.get('split')=='prospective-challenge' and len(fresh.get('records',[]))==12,'Fresh phase inventory differs')
        freshspec=ledger.bind(relative(source_path.parent/'challenge/spec.json'))
        require(canonical_sha(json.loads(freshspec.read_text()))==source['challenge_spec_sha256']==fresh['spec_sha256'],'Fresh spec binding differs')
        for row in fresh['records']:
            for prefix in ('pcm','feature'):
                path=(source_path.parent/'challenge'/row[prefix+'_path']).resolve()
                require(path.is_relative_to(source_path.parent/'challenge'),'Fresh artifact path escaped')
                ledger.bind(relative(path),row[prefix+'_sha256'])
    require(manifest.get('maximum_preparation_seconds')==3600 and manifest.get('maximum_output_bytes')==1024**3
        and 0<=manifest.get('elapsed_seconds',-1)<=3600
        and 0<manifest.get('peak_observed_python_rss_bytes',0)<=16*1024**3
        and 0<manifest.get('peak_observed_node_rss_bytes',0)<=2*1024**3,'Declared/observed preparation bounds differ')
    ledger.recheck()
    auditor_sources=[Path(__file__).resolve(),Path(validate_expansion.__code__.co_filename).resolve()]
    return {'schema':'phenometrix-evidence-feasibility-audit.v1','pass':True,'audited_at':datetime.now(timezone.utc).isoformat(),
        'preparation_manifest_path':relative(manifest_path),'preparation_manifest_sha256':sha(manifest_path),
        'source_freeze_path':relative(source_path),'source_freeze_sha256':sha(source_path),
        'training_spec_sha256':source['training_spec_sha256'],'challenge_spec_sha256':source['challenge_spec_sha256'],
        'frame_table_sha256':manifest['frame_table_sha256'],'report_sha256':manifest['report_sha256'],
        'schedules_sha256':manifest.get('schedules_sha256'),'bound_artifacts':len(ledger.hashes),
        'bound_artifact_sha256':ledger.hashes,'audit_source_sha256':{relative(p):sha(p) for p in auditor_sources},
        'case_count':512,'acquisition_rows':100352,'common_eligible_rows':97792,
        'feasibility_pass':passed,'feasibility':expected_report['feasibility'],
        'schedule_count':36 if passed else 0,'schedule_draws_verified':36*33*512 if passed else 0,
        'phase_chronology':phase_times,**phase,
        'limitations':['Saved source/manifest chronology is not operating-system access history.',
            'PCM bytes and prior independent expansion/grouping equations are verified; no waveform resynthesis or model inference is performed.',
            'This verifies TRAIN feasibility and sampling, not accuracy, calibration, clinical validity or promotion.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve()
    require(output.is_relative_to(RUNS.resolve()) and not output.exists() and output.parent.exists(),'Fresh ignored audit output required')
    try:receipt=audit_run(args.preparation)
    except BaseException as error:
        output.write_text(json.dumps({'schema':'phenometrix-evidence-feasibility-audit.v1','pass':False,'failure':repr(error),
            'preparation':str(args.preparation)},indent=2)+'\n')
        raise
    output.write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({k:receipt[k] for k in ('pass','feasibility_pass','case_count','common_eligible_rows','schedule_count','bound_artifacts')}))


if __name__=='__main__':main()
