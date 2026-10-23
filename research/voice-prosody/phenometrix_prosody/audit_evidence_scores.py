"""Independent saved-output scoring for the fixed exposure by evidence study.

No fitting, waveform generation, held-out natural content or selector inference.
Reconstructs targets, schedules, coverage and gates from bound arrays. Ordering
is declared-artifact and source evidence, not an operating-system access log.
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
import time
import numpy as np
import torch
from . import audit_selector_experiment as base
from . import audit_harmonic_experiment as historical
from . import audit_candidate_coverage as retention
from . import audit_expanded_pool as pool
from . import audit_evidence_feasibility as feasibility_audit

REPO, ROOT, RUNS = base.REPO, base.ROOT, base.RUNS
SEEDS = (20261022,20261023,20261024)
ARMS = ('existing__off','existing__on','targeted__off','targeted__on')
FIT_NAMES = [f'{a}__seed{s}' for a in ARMS for s in SEEDS]
CONTROLS = [f'frozen-expanded7__seed{s}' for s in SEEDS]
SELECTORS = FIT_NAMES+CONTROLS
METHODS = ['original_mlp','original_hybrid',*SELECTORS]
CANDIDATE = [f'targeted__on__seed{s}' for s in SEEDS]
GROUP_COUNTS = {'regression':14,'consumed_harmonic':12,'consumed_subharmonic':12,
    'consumed_augmentation':12,'consumed_coverage':12,'consumed_expansion':12,'challenge':12}
PRIOR = RUNS/'expanded-pool-v1'
PRIOR_SHA = 'c80c0f95fba8dfa5f327de7cf228200206fea8e8a94f72a26a2d79d1cb27d8fb'
DEADLINE = '2026-10-23T12:18:58Z'
NATURAL_ROWS = 132466
require=base.require
compare_nested=retention.compare_nested
load_npz=historical.load_npz
acceptable_labels=pool.acceptable_labels
independent_groups=pool.independent_groups
nearest_labels=pool.nearest_labels


def bound(audit,manifest,path):
    path=Path(path).resolve(); name=str(path.relative_to(REPO))
    maps=[manifest.get('input_sha256',{}),manifest.get('historical_input_sha256',{})]
    values=[mapping.get(name,mapping.get(str(path))) for mapping in maps]
    values=[v for v in values if v is not None]
    require(values and len(set(values))==1,'Input lacks one consistent bound identity: '+name)
    return audit.bind(path,values[0])


def verify_evidence(evidence,feature,grouped,pcm_sha):
    expected={'schema','feature_names','evidence','correlations','valid','period_samples','common_length',
        'grouped_candidate_hz','grouped_candidate_mask','grouped_member_mask','reference_sample','sample_rate_hz','pcm_sample_count','pcm_sha256'}
    require(set(evidence)==expected and str(evidence['schema'])=='phenometrix.candidate-periodicity-evidence.v1', 'Evidence schema differs')
    require(list(evidence['feature_names'])==['period_correlation','half_primitive_period_gap','periodicity_valid'] and
        str(evidence['pcm_sha256'])==pcm_sha,'Evidence channel/PCM identity differs')
    for name in ('reference_sample','sample_rate_hz','pcm_sample_count'):
        require(np.array_equal(evidence[name],feature[name]),'Evidence acquisition support differs')
    for name in ('candidate_hz','candidate_mask','member_mask'):
        require(np.array_equal(evidence['grouped_'+name],grouped[name],equal_nan=True),'Evidence grouped emitted candidate differs')
    mask=grouped['candidate_mask'];shape=mask.shape
    channels=evidence['evidence'];corr=evidence['correlations'];valid=evidence['valid']
    require(channels.dtype==np.float32 and channels.shape==shape+(3,) and corr.dtype==np.float64 and corr.shape==shape+(4,) and
        valid.dtype==np.bool_ and valid.shape==shape,'Evidence dtype/shape differs')
    require(np.isfinite(channels).all() and np.isfinite(corr).all() and (np.abs(corr)<=1).all() and not valid[~mask].any(), 'Evidence validity/bounds differ')
    require(not np.any(corr[~valid]) and not np.any(channels[~valid]),'Unavailable periodicity is not explicit zero')
    expected_channels=np.stack((corr[...,0],(corr[...,0]-corr[...,1:].max(-1))/2,valid.astype(float)),axis=-1).astype(np.float32)
    require(np.array_equal(channels,expected_channels),'Evidence channel arithmetic differs from saved correlations')
    periods=np.zeros(shape);length=np.zeros(shape,np.int32)
    periods[mask]=48000/grouped['candidate_hz'][mask];length[mask]=1920-np.floor(periods[mask]).astype(np.int32)-1
    require(np.array_equal(evidence['period_samples'],periods) and np.array_equal(evidence['common_length'],length),'Evidence period/common support differs')
    return len(mask)


def natural_gate(cells):
    require(set(cells)==set(METHODS),'Natural method inventory differs')
    def mean(name,field='joint_error',condition=None):
        vals=[v[field] for key,v in cells[name].items() if condition is None or key.endswith('/'+condition)]
        require(len(vals)==(16 if condition is None else 4),'Natural gate cell denominator differs')
        return float(np.mean(vals))
    candidate_error=float(np.mean([mean(m) for m in CANDIDATE])); success=float(np.mean([mean(m,'reference_voiced_success_rate') for m in CANDIDATE]))
    gains={b:mean(b)-candidate_error for b in ('original_mlp','original_hybrid')}
    excess={c:float(np.mean([mean(m,condition=c) for m in CANDIDATE]))-mean('original_mlp',condition=c) for c in base.CONDITIONS}
    checks={'mean_improves_both_by_half_point':all(v>=.005 for v in gains.values()),
        'each_seed_improves_original_mlp':all(mean(m)<mean('original_mlp') for m in CANDIDATE),
        'no_condition_worsens_over_one_point':all(v<=.01 for v in excess.values()),
        'voiced_success_not_lower':success>=mean('original_mlp','reference_voiced_success_rate')}
    means={a:float(np.mean([mean(f'{a}__seed{s}') for s in SEEDS])) for a in ARMS}
    return {'predesignated_candidate':'targeted__on','pass':all(checks.values()),'checks':checks,
        'mean_joint_error':candidate_error,'mean_voiced_success':success,'joint_error_improvement_over_frozen':gains,
        'condition_error_excess_over_mlp':excess,'arm_mean_joint_error':means,
        'factorial_error_differences':{'targeted_minus_existing_off':means['targeted__off']-means['existing__off'],
            'evidence_on_minus_off_existing':means['existing__on']-means['existing__off'],
            'evidence_on_minus_off_targeted':means['targeted__on']-means['targeted__off'],
            'interaction':means['targeted__on']-means['targeted__off']-means['existing__on']+means['existing__off']}}


def analytic_gates(gates):
    require(set(gates)==set(CANDIDATE),'Candidate seed gate inventory differs')
    expected={'stationary':3,'harmonic_change':3,**{g:12 for g in GROUP_COUNTS if g!='regression'}}
    by_seed={}
    for method,rows in gates.items():
        require(len(rows)==78,'Expected78 mandatory case gates per candidate seed')
        counts=defaultdict(int)
        for key,row in rows.items(): counts[row['category']]+=1
        require(dict(counts)==expected,'Mandatory gate category inventory differs')
        by_seed[method]={key:value['success']>=.95 for key,value in rows.items()}
    return {'by_seed_case':by_seed,'passed_by_seed':{m:sum(v.values()) for m,v in by_seed.items()},
        'required_per_seed':78,'pass':all(all(v.values()) for v in by_seed.values())}

def natural_and_coverage(audit,run,manifest,metrics,prior):
    arms=('original6','expanded7')
    predictions={name:load_npz(audit,run/'predictions'/(name+'.npz'),manifest['prediction_sha256']['predictions/'+name+'.npz']) for name in METHODS}
    for values in predictions.values():require(set(values)=={'state','hz'},'Natural prediction fields differ')
    for seed in SEEDS:
        name=f'expanded7__seed{seed}';p='predictions/'+name+'.npz'
        old=load_npz(audit,PRIOR/p,prior['prediction_sha256'][p])
        require(all(np.array_equal(old[k],predictions[f'frozen-expanded7__seed{seed}'][k],equal_nan=True) for k in ('state','hz')),'Frozen expanded natural control changed')
    pp=run/'probabilities-development.npz';p=load_npz(audit,pp,manifest['input_sha256'][str(pp.relative_to(REPO))])
    require(set(p)=={'original_probability'},'DEV probability fields differ');probability=p['original_probability']
    require(probability.ndim==1 and np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all(),'DEV original probability invalid')
    oldfreeze=RUNS/'hybrid-natural-v1/prediction-freeze.json'
    baseline=audit.json(oldfreeze,manifest['input_sha256'][str(oldfreeze.relative_to(REPO))]);baserows={r['recording_id']:r for r in baseline['recordings']}
    totals={m:defaultdict(lambda:{k:0 for k in base.COUNTS}) for m in METHODS}
    all_cov={};headcounts={arm:{rep:defaultdict(lambda:{'eligible':0,'oracle_correct':0}) for rep in ('fixed-slots','shared-candidates')} for arm in arms}
    replacement_totals=defaultdict(int);evidence_rows=0;exclusions={};training_parts={a:{'raw':[],'acceptable':[],'nearest':[]} for a in arms}
    for split in ('train','development'):
        folder,oldcache,metadata=base.load_cache(audit,split)
        cachepath=PRIOR/split/'manifest.json';cache=audit.json(bound(audit,prior,cachepath))
        require(cache['schema']=='phenometrix-expanded-pool-natural-cache.v1' and cache['status']=='complete' and cache['split']==split and cache['reference_content_read'] is False,'Revised natural cache state differs')
        require(cache['source_sha256']==prior['source_sha256'] and cache['candidate_policy']=='add-shortest-admitted-full40-v1','Revised cache policy/source differs')
        require(cache['original_manifest_path']==str((folder/'manifest.json').relative_to(REPO)) and cache['original_manifest_sha256']==base.sha(folder/'manifest.json'),'Original cache binding differs')
        require(len(cache['records'])==len(oldcache['recordings'])==({'train':960,'development':320}[split]),'Natural cache inventory differs')
        historical_cache=audit.json(bound(audit,prior,pool.COVERAGE_PRIOR/split/'manifest.json'))
        historical_rows={r['recording_id']:r for r in historical_cache['records']}
        current=audit.json(bound(audit,manifest,run/'evidence'/split/'manifest.json'))
        require(current['status']=='complete' and current['split']==split and current['reference_content_read'] is False and
            [r['recording_id'] for r in current['records']]==[r['recording_id'] for r in cache['records']], 'Current evidence cache identity/order differs')
        current_rows={r['recording_id']:r for r in current['records']}
        refs_cache={};covcells={arm:defaultdict(lambda:{k:0 for k in ('voiced','raw','grouped','lost','gained')}) for arm in arms}
        changes=defaultdict(lambda:{'voiced':0,**{rep:{k:0 for k in ('lost','gained','both','neither')} for rep in ('fixed-slots','shared-candidates')}})
        offset=0;exclusions[split]={}
        for record,row in zip(oldcache['recordings'],cache['records']):
            for key in ('recording_id','source_recording_id','speaker','condition','sample_count','transformed_pcm_sha256'):
                require(record[key]==row[key],'Paired source/condition identity differs: '+key)
            original=base.load_feature(audit,folder,record)
            require(row['original_feature_path']==str((folder/'features'/(record['recording_id']+'.npz')).relative_to(REPO)) and row['original_feature_sha256']==record['feature_sha256'],'Original feature binding differs')
            revised=load_npz(audit,REPO/row['expanded_feature_path'],row['expanded_feature_sha256']);provenance=load_npz(audit,REPO/row['provenance_path'],row['provenance_sha256'])
            old_row=historical_rows[record['recording_id']]
            require(row['source_provenance_path']==old_row['provenance_path'] and row['source_provenance_sha256']==old_row['provenance_sha256'], 'Historical peak binding differs')
            old_provenance=load_npz(audit,REPO/row['source_provenance_path'],row['source_provenance_sha256'])
            for peak_field in ('old_full40_hz','new_full40_hz','old_full40_confidence','new_full40_confidence'):
                require(np.array_equal(old_provenance[peak_field],provenance[peak_field],equal_nan=True),'Historical peak evidence changed')
            check=pool.validate_expansion(original,revised,provenance,record['transformed_pcm_sha256'])
            for k,v in check.items():replacement_totals[k]+=v
            features={'original6':original,'expanded7':revised};grouped={arm:independent_groups(f) for arm,f in features.items()}
            er=current_rows[record['recording_id']]
            for field in ('original_feature_path','original_feature_sha256','expanded_feature_path','expanded_feature_sha256','start','stop'):
                require(er[field]==row[field],'Current evidence points to a different acquisition: '+field)
            require(er['pcm_sha256']==record['transformed_pcm_sha256'],'Current evidence PCM differs')
            evidence_rows+=verify_evidence(load_npz(audit,REPO/er['evidence_path'],er['evidence_sha256']),revised,grouped['expanded7'],er['pcm_sha256'])
            targets=original['reference_sample']-960;stop=offset+len(targets);sl=slice(offset,stop)
            require(row['start']==offset and row['stop']==stop,'Paired cache row offsets differ')
            source=record['source_recording_id']
            if source not in refs_cache:refs_cache[source]=base.references(audit,metadata[source],record,split)
            refs=refs_cache[source];truth,valid,voiced=retention.ref_arrays(refs,targets);include=valid&voiced
            excludes={'invalid_reference':sum(not r.valid for r in refs),'boundary_reference':sum(r.valid and not r.eligible for r in refs),
                      'predictions_outside_reference_extent':len(set(targets.tolist())-{r.sample for r in refs})}
            exclusions[split][record['recording_id']]=excludes
            cell=record['speaker']+'/'+record['condition'];flags={}
            for arm,f in features.items():
                raw=retention.covered_by_candidates(f['candidate_hz'],f['candidate_mask'],truth,include)
                g=retention.covered_by_candidates(grouped[arm]['candidate_hz'],grouped[arm]['candidate_mask'],truth,include)
                flags[arm]={'fixed-slots':raw,'shared-candidates':g}
                for k,v in retention.coverage_counts(include,raw,g).items():covcells[arm][cell][k]+=v
                if split=='train':
                    acceptable,covered=acceptable_labels(grouped[arm]['candidate_hz'],grouped[arm]['candidate_mask'],truth,valid,voiced)
                    errors=np.where(grouped[arm]['candidate_mask']&include[:,None],
                        np.abs(1200*np.log2(np.where(grouped[arm]['candidate_mask'],grouped[arm]['candidate_hz'],1)/np.where(include,truth,1)[:,None])),np.inf)
                    nearest=np.where(covered,errors.argmin(-1),-1).astype(np.int64)
                    training_parts[arm]['raw'].append(raw);training_parts[arm]['acceptable'].append(acceptable);training_parts[arm]['nearest'].append(nearest)
                if split=='development':
                    for rep,covered in flags[arm].items():
                        oracle=retention.fixed_voicing_oracle(valid,voiced,probability[sl],covered)
                        headcounts[arm][rep][cell]['eligible']+=oracle['eligible_valid_frames']
                        headcounts[arm][rep][cell]['oracle_correct']+=oracle['correct_frames']
            changes[cell]['voiced']+=int(include.sum())
            for rep in ('fixed-slots','shared-candidates'):
                c=retention.policy_change_counts(include,flags['original6'][rep],flags['expanded7'][rep])
                for key in ('lost','gained','neither'):changes[cell][rep][key]+=c[key]
                changes[cell][rep]['both']+=c['retained']
            if split=='development':
                binding=baserows[record['recording_id']]
                cached=json.loads(gzip.decompress(audit.bind(RUNS/'hybrid-natural-v1'/binding['hybrid_prediction_path'],binding['hybrid_prediction_sha256']).read_bytes()))['predictions']
                for m,old_name in [('original_mlp','mlp_original'),('original_hybrid','hybrid_original')]:
                    rows=cached[old_name];require([r['sample'] for r in rows]==targets.tolist(),'Baseline target grid differs')
                    states=np.array([{'unvoiced':0,'voiced':1,'unavailable':2}[r['state']] for r in rows])
                    frequencies=np.array([np.nan if r['f0_hz'] is None else r['f0_hz'] for r in rows])
                    require(np.array_equal(predictions[m]['state'][sl],states) and np.array_equal(predictions[m]['hz'][sl],frequencies,equal_nan=True),'Natural baseline changed')
                require(np.array_equal(probability[sl],[r['voicing_probability'] for r in cached['mlp_original']]),'Original DEV probability changed')
                for method in SELECTORS:
                    state,hz=predictions[method]['state'][sl],predictions[method]['hz'][sl]
                    require(np.array_equal(state==0,probability[sl]<.5),'Frozen original voicing changed')
                    expected_state=np.where(probability[sl]<.5,0,np.where(grouped['expanded7']['candidate_mask'].any(-1),1,2))
                    require(np.array_equal(state,expected_state),'Selector availability differs from actual expanded pool')
                    require(np.all(np.any(grouped['expanded7']['candidate_hz'][state==1]==hz[state==1,None],axis=1)),
                        'Selector emitted a non-member frequency')
                valid_targets={r.sample for r in refs}
                for name,z in predictions.items():
                    objects=[o for o in base.prediction_objects(targets,z['state'][sl],z['hz'][sl]) if o.sample in valid_targets]
                    got=base.object_score(refs,objects)
                    expected=next(r['metrics'] for r in metrics['natural'][name]['per_record'] if r['recording_id']==record['recording_id'])
                    base.compare_metrics(audit,got,expected,name+'/'+record['recording_id'])
                    for k in base.COUNTS:totals[name][cell][k]+=got[k]
            offset=stop
        expected_cells={s+'/'+c for s in base.SPEAKERS[split] for c in base.CONDITIONS}
        require(set(changes)==expected_cells,'Coverage cell inventory differs')
        all_cov[split]={arm:retention.finish_coverage(covcells[arm]) for arm in arms}
        all_cov[split]['policy_change']=retention.finish_transition(changes)
        compare_nested(audit,all_cov[split],metrics['coverage'][split],split+'/coverage')
        if split=='development':require(offset==len(probability) and all(len(v['state'])==offset for v in predictions.values()),'Natural prediction extent differs')
    cells={m:{key:base.count_rates(value) for key,value in rows.items()} for m,rows in totals.items()}
    for name,values in cells.items():
        for cell,value in values.items():base.compare_metrics(audit,value,metrics['natural'][name]['cells'][cell],name+'/'+cell)
        for rate in base.RATES:
            audit.equal(float(np.mean([v[rate] for v in values.values()])),metrics['natural'][name]['macro'][rate],name+'/macro/'+rate)
            for field,keys in [('per_speaker',base.SPEAKERS['development']),('per_condition',base.CONDITIONS)]:
                for key in keys:
                    chosen=[v[rate] for cell,v in values.items() if cell.startswith(key+'/') or cell.endswith('/'+key)]
                    audit.equal(float(np.mean(chosen)),metrics['natural'][name][field][key][rate],name+'/'+field+'/'+key+'/'+rate)
        pooled=base.count_rates({k:sum(v[k] for v in values.values()) for k in base.COUNTS})
        base.compare_metrics(audit,pooled,metrics['natural'][name]['pooled'],name+'/pooled')
    headroom={}
    for arm in arms:
        headroom[arm]={}
        for rep,counts in headcounts[arm].items():
            for cell,c in counts.items():
                require(c['eligible']==cells['original_mlp'][cell]['eligible_valid_frames'],'Oracle denominator differs')
                c['mlp_correct']=cells['original_mlp'][cell]['correct_frames'];c['gain']=(c['oracle_correct']-c['mlp_correct'])/c['eligible']
            gain=float(np.mean([v['gain'] for v in counts.values()]));headroom[arm][rep]={'cells':dict(counts),'mean_gain':gain,'pass':gain>=.01}
    compare_nested(audit,headroom['expanded7'],metrics['headroom'],'headroom')
    compare_nested(audit,exclusions,metrics['exclusions'],'natural exclusions')
    raw_flags={arm:np.concatenate(parts['raw']) for arm,parts in training_parts.items()}
    require(not np.any(raw_flags['original6']&~raw_flags['expanded7']),'Expanded raw TRAIN coverage evicted a candidate')
    schedule=np.flatnonzero(raw_flags['original6']|raw_flags['expanded7'])
    require(len(schedule)==NATURAL_ROWS and raw_flags['original6'].sum()==132181,'Training raw-union inventory differs')
    training={'schedule':schedule}
    for arm,parts in training_parts.items():
        acceptable=np.concatenate(parts['acceptable'])
        training[arm+'__acceptable']=acceptable;training[arm+'__covered']=acceptable.any(-1)
        training[arm+'__nearest']=np.concatenate(parts['nearest'])
    return cells,all_cov,headroom['expanded7'],training,{'methods':len(METHODS),'development_condition_records':320,'train_condition_records':960,
        'development_rows':len(probability),'checkpoint_inference_performed':False,'evidence_frame_rows_checked':evidence_rows,'replacement_reconstruction':dict(replacement_totals)}



def verify_checkpoints(audit,run,manifest,prior):
    from .harmonic_selector_model import HarmonicSelector
    require(set(manifest['fits'])==set(FIT_NAMES),'Exact12 final fits required')
    require(set(manifest['fit_attempts'])==set(FIT_NAMES),'Fit attempts exceed or omit declared inventory')
    shapes={'scaling_fitted':(),'candidate_mean':(8,),'candidate_scale':(8,),'global_mean':(9,),'global_scale':(9,),
        'evidence_head.weight':(1,3)}
    for layer,shape in {'candidate_encoder':(32,8),'message_encoder':(32,9),'shared_hidden':(32,73),'shared_head':(1,32)}.items():
        shapes[layer+'.weight']=shape;shapes[layer+'.bias']=(shape[0],)
    buffers=('scaling_fitted','candidate_mean','candidate_scale','global_mean','global_scale')
    scaling=None;initial={};elapsed=0.
    for seed in SEEDS:
        name=f'expanded7__seed{seed}';old=torch.load(audit.bind(PRIOR/'checkpoints'/(name+'.pt'),prior['fits'][name]['weights_sha256']),map_location='cpu',weights_only=True)['state_dict']
        if scaling is None:scaling={k:old[k] for k in buffers}
        require(all(torch.equal(scaling[k],old[k]) for k in buffers),'Prior scaling differs across seeds')
        torch.manual_seed(seed);model=HarmonicSelector('relative','acceptable-set')
        state=model.state_dict()
        for k in buffers:state[k]=scaling[k].clone()
        state['evidence_head.weight']=torch.zeros((1,3),dtype=torch.float32)
        digest=hashlib.sha256()
        for k,t in sorted(state.items()):digest.update(k.encode());digest.update(t.numpy().tobytes())
        initial[seed]=digest.hexdigest()
        for arm in ARMS:
            name=f'{arm}__seed{seed}';fit=manifest['fits'][name];attempt=manifest['fit_attempts'][name]
            require((fit['arm'],fit['seed'],fit['optimizer_updates'],fit['parameter_count'])==(arm,seed,396,3012),'Fit identity/budget differs')
            require(attempt['status']=='complete' and attempt['initial_tensors_sha256']==fit['initial_tensors_sha256']==initial[seed],
                'Final fit attempt or reconstructed initialization differs')
            require(historical.dt(attempt['started_at'])<historical.dt('2026-10-23T09:48:58Z') and
                historical.dt(manifest['training_frozen_at'])<=historical.dt(attempt['started_at'])<=historical.dt(attempt['completed_at'])<=historical.dt(manifest['checkpoints_frozen_at']),
                'Fit chronology violates fixed phase/deadline')
            for key in ('epoch_losses','epoch_base_losses','epoch_weighted_auxiliary_losses'):
                require(len(fit[key])==12 and all(math.isfinite(x) and x>=-1e-7 for x in fit[key]),'Fit loss summary invalid')
            require(np.allclose(fit['epoch_losses'],np.array(fit['epoch_base_losses'])+np.array(fit['epoch_weighted_auxiliary_losses']),atol=1e-7,rtol=0),'Loss decomposition differs')
            saved=torch.load(audit.bind(run/'checkpoints'/(name+'.pt'),fit['weights_sha256']),map_location='cpu',weights_only=True)
            require(saved['arm']==arm and saved['evidence_mode']==('periodicity' if arm.endswith('__on') else 'off') and
                saved['config']=={'representation':'shared-candidates','loss':'acceptable-set','hidden_size':32},'Saved checkpoint contract differs')
            state=saved['state_dict'];require(set(state)==set(shapes),'Checkpoint tensor inventory differs')
            for k,shape in shapes.items():
                require(tuple(state[k].shape)==shape and state[k].dtype==(torch.bool if k=='scaling_fitted' else torch.float32) and
                    torch.isfinite(state[k]).all(),'Checkpoint tensor shape/dtype/finite value differs')
            require(all(torch.equal(scaling[k],state[k]) for k in buffers),'Original-six scaling changed')
            if arm.endswith('__off'):require(torch.equal(state['evidence_head.weight'],torch.zeros((1,3))),'Off residual changed from zero')
            compare_nested(audit,state['evidence_head.weight'].tolist(),fit['evidence_residual_weights'],name+'/residual')
            elapsed+=fit['elapsed_seconds']
    audit.equal(elapsed,manifest['fit_elapsed_seconds'],'Cumulative fit seconds')
    return {'checkpoints':12,'parameters_each':3012,'initial_tensor_sha256':initial,'original_six_scaling_exact':True,
        'optimizer_updates_each':396,'fit_elapsed_seconds':elapsed,'optimizer_replay_performed':False}


def reconstruct_bank(audit,run,manifest,exposure):
    if exposure=='existing':
        cache=audit.json(bound(audit,manifest,PRIOR/'training/manifest.json'))
        expected=pool.augmentation.expected_training_records()['balance-aug']
        require(cache['status']=='complete' and [r['id'] for r in cache['records']]==[r['id'] for r in expected], 'Existing TRAIN case order differs')
    else:
        cache=audit.json(run/'feasibility/manifest.json',manifest['feasibility_manifest_sha256'])
        expected=feasibility_audit.expected_records()
        require(cache['status']=='complete' and [r['id'] for r in cache['records']]==[r['id'] for r in expected], 'Targeted TRAIN case order differs')
    parts=defaultdict(list)
    for i,(row,record) in enumerate(zip(cache['records'],expected)):
        original=load_npz(audit,REPO/row['original_feature_path'],row['original_feature_sha256'])
        feature=load_npz(audit,REPO/row['expanded_feature_path'],row['expanded_feature_sha256'])
        groups=independent_groups(feature);ends,targets,common=historical.validate_feature(original,96000)
        require(np.array_equal(ends,feature['reference_sample']),'Training evidence support differs')
        scenario=row['scenario'];require(scenario['initialHz']==record['frequency_hz'] and scenario['kind']=='observation-change','TRAIN truth scenario differs')
        ix=np.arange(len(ends)) if exposure=='targeted' else np.flatnonzero(common)
        truth=np.full(len(ix),record['frequency_hz']);a,c,n=nearest_labels(groups['candidate_hz'][ix],groups['candidate_mask'][ix],truth,np.ones(len(ix),bool))
        values={'acceptable':a,'covered':c,'nearest':n,'scenario_index':np.full(len(ix),i,np.int64),
            'reference_hz':truth,'reference_sample':targets[ix],'availability_sample':ends[ix],
            'region':historical.regions(scenario,ends)[ix],'common':common[ix]}
        for k,v in values.items():parts[k].append(v)
    result={k:np.concatenate(v) for k,v in parts.items()}
    require(len(result['covered'])==(18336 if exposure=='existing' else 100352),'Synthetic training extent differs')
    return result


def audit_training(audit,run,manifest,natural_targets):
    schedule=natural_targets['schedule'];expected={k:natural_targets['expanded7__'+k] for k in ('acceptable','covered','nearest')}
    expected['schedule']=schedule
    saved=load_npz(audit,run/'training-targets.npz',manifest['training_targets_sha256'])
    require(set(saved)==set(expected),'Natural target field inventory differs')
    for k,v in expected.items():require(saved[k].dtype==v.dtype and np.array_equal(saved[k],v),'Natural targets differ: '+k)
    require(manifest['scheduled_rows']==len(schedule)==132466,'Natural schedule extent differs')
    banks={a:reconstruct_bank(audit,run,manifest,a) for a in ('existing','targeted')}
    sy=load_npz(audit,run/'synthetic-targets.npz',manifest['synthetic_targets_sha256'])
    require(set(sy)=={a+'__'+k for a in banks for k in ('acceptable','covered','nearest')},'Synthetic target fields differ')
    for arm,b in banks.items():
        for k in ('acceptable','covered','nearest'):require(np.array_equal(sy[arm+'__'+k],b[k]),'Synthetic target differs: '+arm+'/'+k)
    names={f'seed{s}__epoch{e}' for s in SEEDS for e in range(12)}
    schedules={name:load_npz(audit,bound(audit,manifest,run/(name+'-schedules.npz'))) for name in ('natural','existing-synthetic','targeted-synthetic')}
    require(all(set(v)==names for v in schedules.values()),'Schedule field inventory differs')
    prep=audit.json(run/'feasibility/manifest.json',manifest['feasibility_manifest_sha256']);table=load_npz(audit,REPO/prep['frame_table_path'],prep['frame_table_sha256'])
    _,strata,_=feasibility_audit.derive_strata(table)
    require(np.array_equal(table['case_index'],banks['targeted']['scenario_index']) and np.array_equal(table['truth_hz'],banks['targeted']['reference_hz']), 'Targeted table ordering differs')
    coverage={a:{'rows':len(b['covered']),'covered_rows':int(b['covered'].sum())} for a,b in banks.items()}
    for seed in SEEDS:
        for epoch in range(12):
            key=f'seed{seed}__epoch{epoch}';order=pool.natural_order(seed,epoch)
            require(np.array_equal(schedules['natural'][key],order),'Natural row permutation differs')
            reconstruct=feasibility_audit.reconstruct_schedule(strata,table['case_index'],seed,epoch)
            target_order=reconstruct['indices']
            for exposure in ('existing','targeted'):
                want=pool.synthetic_order(seed,epoch) if exposure=='existing' else target_order
                got=schedules[exposure+'-synthetic'][key]
                require(got.dtype==np.int64 and np.array_equal(got,want),'Synthetic sampling package differs')
                b=banks[exposure];flat=got.ravel()
                require(b['common'][flat].all(),'Synthetic schedule included noncommon row')
                observed={'draws':len(flat),'unique_rows':len(np.unique(flat)),'duplicate_draws':len(flat)-len(np.unique(flat)),
                    'covered_rows':int(b['covered'][flat].sum()),'per_case':np.bincount(b['scenario_index'][flat],minlength=96 if exposure=='existing' else 512).tolist(),
                    'per_region':{str(r):int((b['region'][flat]==r).sum()) for r in np.unique(b['region'])}}
                covered=saved['covered'][schedule][order]
                require(all(covered[j:j+4096].any() for j in range(0,len(order),4096)),'Zero covered natural batch would skip fixed update')
                denominators={'natural_covered':int(covered.sum()),'synthetic_covered':int(b['covered'][flat].sum())}
                for mode in ('off','on'):
                    fit=manifest['fits'][f'{exposure}__{mode}__seed{seed}']
                    compare_nested(audit,observed,fit['synthetic_exposure_by_epoch'][epoch],exposure+'/'+mode+'/'+key+'/exposure')
                    compare_nested(audit,denominators,fit['loss_denominators'][epoch],exposure+'/'+mode+'/'+key+'/denominator')
    return {'natural_union_rows':132466,'natural_grouped_zero_rows':int((~saved['covered'][schedule]).sum()),
        'synthetic_banks':coverage,'natural_schedule_arrays':36,'synthetic_schedule_arrays':72,
        'synthetic_draws_per_epoch':16896,'epochs':12,'updates_per_fit':396,'targets_schedules_exposures_exact':True,
        'loss_equation':'natural covered-row acceptable-set CE +0.25 synthetic covered-row acceptable-set CE',
        'counterfactual_scope':'targeted package changes both bank and sampling; no separate oversampling effect claim'}


def fresh_records():
    return [{'id':f'evidence-balance-f{i}-p{j}','frequency_hz':f,'phase_radians':list(p),'gain':.057,
        'sample_rate_hz':48000,'sample_count':96000,'change_sample':39840,'before':[1.,.5,.25],'after':[0.,1.,.5]}
        for i,f in enumerate((109.9,241.1,379.3,563.9,619.7,677.9))
        for j,p in enumerate(((.79,-1.37,.23),(2.17,.61,-.89)))]


def audit_analytic(audit,run,manifest,metrics,prior):
    inventory=audit.json(bound(audit,manifest,run/'analytic-inventory.json'))
    require({g:len(v) for g,v in inventory.items()}==GROUP_COUNTS,'86-case inventory differs')
    previous=audit.json(bound(audit,prior,PRIOR/'analytic-inventory.json'))
    old={(('consumed_expansion' if g=='challenge' else g),r['id']):r for g,rows in previous.items() for r in rows}
    cm=audit.json(run/'challenge/manifest.json');spec=audit.json(run/'challenge/spec.json')
    require(cm['status']=='complete' and cm['reference_content_read'] is False and spec['records']==fresh_records(),'Fresh challenge records differ')
    require(hashlib.sha256(historical.canonical(spec)).hexdigest()==manifest['challenge_spec_sha256']==cm['spec_sha256'],'Fresh metadata identity differs')
    require(cm['freeze_path']==str((run/'checkpoint-freeze.json').relative_to(REPO)) and cm['freeze_sha256']==base.sha(run/'checkpoint-freeze.json'),'Fresh checkpoint linkage differs')
    require(historical.dt(manifest['checkpoints_frozen_at'])<=historical.dt(cm['started_at'])<=historical.dt(cm['completed_at'])<=historical.dt(manifest['predictions_frozen_at']),'Fresh challenge chronology differs')
    fresh={r['id']:r for r in cm['records']};parameters={r['id']:r for r in fresh_records()}
    require(list(fresh)==list(parameters),'Fresh case order differs')
    gates={m:{} for m in CANDIDATE};summaries={};controls=0;frames=0
    for group,rows in inventory.items():
        require(set(metrics[group])=={r['id'] for r in rows} and len({r['id'] for r in rows})==len(rows),'Analytic reported inventory differs')
        for row in rows:
            name=row['id'];scenario=row['scenario'];n=76800 if group=='regression' else 96000
            if group=='challenge':
                source=fresh[name];params=parameters[name]
                require(source['parameters_sha256']==hashlib.sha256(historical.canonical(params)).hexdigest(),'Fresh parameters hash differs')
                require(row['original_feature_path']==str((run/'challenge'/source['feature_path']).relative_to(REPO)) and row['original_feature_sha256']==source['feature_sha256'] and row['pcm_sha256']==source['pcm_sha256'],'Fresh source artifact differs')
                audit.bind(run/'challenge'/source['pcm_path'],source['pcm_sha256'])
                require(scenario=={'id':name,'kind':'observation-change','durationSeconds':2.,'initialHz':params['frequency_hz'],'finalHz':params['frequency_hz'],'changeStartSeconds':.83,'changeEndSeconds':.83,'slopeHzPerSecond':0},'Fresh scenario differs')
            else:
                source=old[(group,name)]
                for key in ('scenario','sample_count','original_feature_path','original_feature_sha256','pcm_sha256'):
                    require(row[key]==source[key],'Consumed analytic identity differs: '+key)
            original=load_npz(audit,REPO/row['original_feature_path'],row['original_feature_sha256'])
            expanded=load_npz(audit,REPO/row['expanded_feature_path'],row['expanded_feature_sha256'])
            pool.validate_pool(original,6);pool.validate_pool(expanded,7)
            ends,targets,eligible=historical.validate_feature(original,n);require(row['sample_count']==n,'Analytic extent differs')
            require(np.array_equal(ends,expanded['reference_sample']),'Analytic expanded support differs')
            groups={'original6':independent_groups(original),'expanded7':independent_groups(expanded)}
            verify_evidence(load_npz(audit,REPO/row['evidence_path'],row['evidence_sha256']),expanded,groups['expanded7'],row['pcm_sha256'])
            z=load_npz(audit,run/row['prediction_path'],row['prediction_sha256'])
            require(row['prediction_sha256']==manifest['prediction_sha256'][row['prediction_path']],'Analytic prediction freeze differs')
            require(set(z)=={'original_probability'}|{m+s for m in SELECTORS for s in ('__state','__hz')},'Analytic prediction fields differ')
            probability=z['original_probability'];require(probability.shape==targets.shape and np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all(),'Analytic probability invalid')
            arrays={m:(z[m+'__state'],z[m+'__hz']) for m in SELECTORS}
            for m,(state,hz) in arrays.items():
                base.prediction_objects(targets,state,hz)
                require(np.array_equal(state,np.where(probability<.5,0,np.where(groups['expanded7']['candidate_mask'].any(1),1,2))),'Analytic fixed voice/availability differs')
                require(np.all(np.any(groups['expanded7']['candidate_hz'][state==1]==hz[state==1,None],axis=1)),'Analytic output is not a grouped actual member')
            if group!='challenge':
                p=PRIOR/source['prediction_path'];require(row['prior_prediction_path']==str(p.relative_to(REPO)),'Prior prediction path differs')
                saved=load_npz(audit,p,source['prediction_sha256'])
                require(np.array_equal(probability,saved['original_probability']),'Consumed original voicing probability differs')
                for seed in SEEDS:
                    for suffix in ('__state','__hz'):
                        require(np.array_equal(z[f'frozen-expanded7__seed{seed}'+suffix],saved[f'expanded7__seed{seed}'+suffix],equal_nan=True),'Frozen expanded analytic control changed')
                    controls+=len(targets)
            truth=historical.curve(scenario,targets/48000);region=historical.regions(scenario,ends);report=metrics[group][name]
            require(report['kind']==scenario['kind'] and set(report['methods'])==set(SELECTORS),'Analytic method/kind differs')
            compare_nested(audit,pool.analytic_coverage({'original6':original,'expanded7':expanded},groups,truth,eligible,region,probability,arrays),report['candidate_coverage'],group+'/'+name+'/coverage')
            case={}
            for method,(state,hz) in arrays.items():
                for label,include in {'all':eligible,**{r:eligible&(region==r) for r in sorted(set(region[eligible]))}}.items():
                    got=historical.analytic_summary(truth[include],state[include],hz[include],region[include],scenario['slopeHzPerSecond'])
                    want=report['methods'][method]['all'] if label=='all' else report['methods'][method]['regions'][label]
                    compare_nested(audit,got,want,group+'/'+name+'/'+method+'/'+label)
                if scenario['kind']=='step':
                    compare_nested(audit,retention.step_summary(scenario,targets[eligible],ends[eligible],state[eligible],hz[eligible]),report['methods'][method]['step_response'],group+'/'+name+'/'+method+'/step')
                else:require(report['methods'][method]['step_response'] is None,'Unexpected step response')
                if method not in CANDIDATE:continue
                if group!='regression':category=group;include=eligible&(region=='after-observation-change')
                elif scenario['kind']=='constant':category='stationary';include=eligible
                elif scenario['kind']=='observation-change':category='harmonic_change';include=eligible&(region=='after-observation-change')
                else:continue
                value=historical.analytic_summary(truth[include],state[include],hz[include],region[include],0)
                gates[method][group+'/'+name]={'category':category,'success':value['success_within_50c_all_frames']}
                case[method]={'frames':value['frames'],'success':value['success_within_50c_all_frames']}
            summaries[group+'/'+name]=case;frames+=len(targets)
    decision=analytic_gates(gates);compare_nested(audit,decision,metrics['decision']['analytic'],'Analytic gate')
    return {'cases':86,'methods_per_case':15,'frames_per_method':frames,'mandatory_cases_per_candidate_seed':78,
        'control_frame_pairs':controls,'gates':decision,'case_results':summaries}

def phase_order(manifest,initial,input_freeze,prerequisite,training,checkpoint,prediction):
    require(initial['status']=='sources_frozen' and initial['fits']=={} and
        input_freeze['status']=='input_caches_frozen' and input_freeze['reference_content_read'] is False and
        prerequisite['pass'] is True and training['status']=='training_frozen' and training['fits']=={} and
        checkpoint['status']=='checkpoints_frozen' and prediction['status']=='predictions_frozen','Required phase state differs')
    times=[manifest['sources_frozen_at'],input_freeze['frozen_at'],manifest['prerequisite_reference_read_at'],
        prerequisite['frozen_at'],manifest['training_frozen_at'],manifest['checkpoints_frozen_at'],
        manifest['predictions_frozen_at'],manifest['final_scoring_reference_read_at'],manifest['completed_at']]
    require(all(historical.dt(a)<=historical.dt(b) for a,b in zip(times,times[1:])),'Source/input/prerequisite/fit/prediction/final-score chronology differs')
    require(initial['development_references_read_for_prerequisites'] is False and
        initial['development_references_read_for_final_scoring'] is False and
        prediction['development_references_read_for_prerequisites'] is True and
        prediction['development_references_read_for_final_scoring'] is False and
        manifest['development_references_read_for_final_scoring'] is True,'Declared reference phases differ')
    require(set(prediction['fits'])==set(FIT_NAMES),'Prediction phase lacks complete fit inventory')
    return times


def no_fit_artifacts(run,manifest):
    require(manifest['fits']=={} and not manifest.get('fit_attempts') and manifest.get('fit_elapsed_seconds',0)==0,
        'No-go state contains fit attempts')
    for name in ('checkpoints','predictions','analytic'):
        require(not any(p.is_file() for p in (run/name).rglob('*')),'No-go contains dependent '+name+' artifacts')
    for name in ('training-freeze.json','checkpoint-freeze.json','prediction-freeze.json','challenge','metrics.json'):
        require(not (run/name).exists(),'No-go contains forbidden dependent artifact: '+name)


def repair_sources(before,after,declared):
    """Only the declared adapter repair may differ across the preserved attempts."""
    require(set(before)==set(after) and len(after)==58,'Repair changed the complete58-source inventory')
    changed={p:{'before':before[p],'after':after[p]} for p in before if before[p]!=after[p]}
    require(changed==declared and set(changed)=={'research/voice-prosody/phenometrix_prosody/evidence_inputs.py'},
        'Repair differs from the one declared adapter change')
    return changed


def audit_repair(audit,run,manifest,initial):
    repair_path=RUNS/'overnight-20261023/repair-v1-to-v2.json'
    repair=audit.json(repair_path,'759dd150be4b7fb2ab8cbcf4fca1cc3a0f3522dfa6dc0291f6381cd5d84c8cb2')
    require(repair['schema']=='phenometrix-evidence-plumbing-repair.v1' and REPO/repair['new_run_path']==run,
        'Repair receipt targets another scientific attempt')
    oldrun=base.inside(RUNS,REPO/repair['old_run_path'])
    old=audit.json(oldrun/'manifest.json',repair['old_manifest_sha256'])
    frozen=audit.json(oldrun/'source-freeze.json',repair['old_source_freeze_sha256'])
    require(old['status']==repair['old_status']=='failed' and old['failure']==repair['failure']=="ValueError('Preparation input bytes changed')",
        'Preserved first failure differs')
    require(old['source_sha256']==frozen['source_sha256'] and frozen['status']=='sources_frozen' and frozen['fits']=={},
        'Preserved first source freeze differs')
    no_fit_artifacts(oldrun,old)
    require(old['development_references_read_for_prerequisites'] is False and old['development_references_read_for_final_scoring'] is False,
        'Failed acquisition crossed reference phases')
    require(repair['completed_fits']==0 and repair['coverage_outcomes_computed'] is False and repair['fresh_evaluation_generated'] is False,
        'Repair receipt claims dependent scientific work')
    require(repair['new_source_sha256']==initial['source_sha256'],'Repair newsource map differs from actual new freeze')
    changed=repair_sources(frozen['source_sha256'],initial['source_sha256'],repair['changed_sources'])
    for path,digest in frozen['source_sha256'].items():
        audit.bind(oldrun/'source-snapshot'/path,digest)
        if path not in changed:audit.bind(REPO/path,digest)
    for field in ('training_spec_sha256','challenge_spec_sha256','arms','seeds','maximum_fits','deadline_utc'):
        require(old[field]==frozen[field]==initial[field],'Repair changed frozen scientific specification: '+field)
    require(repair['training_spec_sha256']==initial['training_spec_sha256'] and repair['challenge_spec_sha256']==initial['challenge_spec_sha256'],
        'Repair specification binding differs')
    oldprep=audit.json(oldrun/'feasibility/manifest.json')
    require(oldprep['status']=='failed' and oldprep['failure']==old['failure'] and oldprep['records']==[] and oldprep['generation_calls']==1,
        'First failure had unreported coverage records')
    for filename in ('frame-table.npz','report.json','schedules.npz'):
        require(not (oldrun/'feasibility'/filename).exists(),'First failure had unreported coverage/schedule outcomes')
    old_acq=audit.json(oldrun/'feasibility/acquisition/manifest.json',repair['old_acquisition_manifest_sha256'])
    new_acq=audit.json(bound(audit,manifest,run/'feasibility/acquisition/manifest.json'))
    require(old_acq['status']==new_acq['status']=='complete' and len(old_acq['records'])==len(new_acq['records'])==repair['training_configurations_generated']==512,
        'Both explicitly counted TRAIN acquisitions are required')
    require(old_acq['freeze_sha256']==repair['old_source_freeze_sha256'] and new_acq['freeze_sha256']==base.sha(run/'source-freeze.json'),
        'An acquisition was rebound to another attempt')
    require(old_acq['spec_sha256']==new_acq['spec_sha256']==initial['training_spec_sha256'],'Restart TRAIN specification changed')
    for p,digest in old_acq['source_sha256'].items():
        require(frozen['source_sha256'].get(p)==digest,'First acquisition escaped its source closure')
        audit.bind(oldrun/'feasibility/acquisition/source-snapshot'/p,digest)
    same_pcm=same_feature_values=0
    for first,second in zip(old_acq['records'],new_acq['records']):
        for field in ('id','case_index','parameters_sha256','pcm_sha256','typescript_export_sha256','feature_frames','common_eligible_frames'):
            require(first[field]==second[field],'Repeated TRAIN input identity differs: '+field)
        for folder,row in ((oldrun,first),(run,second)):
            audit.bind(folder/'feasibility/acquisition'/row['pcm_path'],row['pcm_sha256'])
        x=load_npz(audit,oldrun/'feasibility/acquisition'/first['feature_path'],first['feature_sha256'])
        y=load_npz(audit,run/'feasibility/acquisition'/second['feature_path'],second['feature_sha256'])
        require(set(x)==set(y),'Repeated TRAIN feature field inventory differs')
        for field in x:
            require(x[field].dtype==y[field].dtype and x[field].shape==y[field].shape and
                (np.array_equal(x[field],y[field],equal_nan=True) if x[field].dtype.kind in 'fc' else np.array_equal(x[field],y[field])),
                'Repeated TRAIN feature values changed: '+field)
        same_pcm+=1;same_feature_values+=1
    execution=[]
    for version,attempt in (('v1',old),('v2',manifest)):
        folder=RUNS/('evidence-factorial-execution-'+version)
        record=audit.json(folder/'execution.json',repair['old_execution_sha256'] if version=='v1' else None)
        require(record['schema']=='phenometrix-evidence-execution.v1' and record['status']==('failed' if version=='v1' else 'complete') and
            record['returncode']==(1 if version=='v1' else 0),'Outer execution exit/status differs')
        command=record['command'];expected_run=oldrun if version=='v1' else run
        require(command[:4]==[str(ROOT/'.venv/bin/python'),'-m','phenometrix_prosody.evidence_experiment','--output'] and
            command[4]==str(expected_run) and command[5]=='--preflight' and len(command)==7,'Outer command targets another run')
        preflight=audit.json(Path(command[6]),attempt['input_sha256'][str(Path(command[6]).relative_to(REPO))])
        require(preflight['pass'] is True and preflight['source_sha256']==attempt['source_sha256'],'Outer preflight differs')
        if version=='v2':
            require(preflight['repair_receipt_path']==str(repair_path.relative_to(REPO)) and preflight['repair_receipt_sha256']==base.sha(repair_path),
                'New preflight does not bind the preserved repair')
        audit.bind(folder/'stdout.log',record['stdout_sha256']);audit.bind(folder/'supervisor.py',record['supervisor_sha256'])
        end=attempt['failed_at'] if version=='v1' else attempt['completed_at']
        require(historical.dt(record['started_at'])<=historical.dt(attempt['sources_frozen_at'])<=historical.dt(end)<=historical.dt(record['completed_at']),
            'Outer execution chronology differs')
        execution.append({'version':version,'execution_sha256':base.sha(folder/'execution.json'),'elapsed_seconds':record['elapsed_seconds'],
            'observed_disposable_bytes':record['observed_disposable_bytes'],'status':record['status'],'returncode':record['returncode']})
    require(historical.dt(old['failed_at'])<=historical.dt(repair['recorded_at'])<=historical.dt(initial['sources_frozen_at']),
        'Repair declaration did not precede new source freeze')
    return {'repair_path':str(repair_path.relative_to(REPO)),'repair_sha256':base.sha(repair_path),
        'preserved_failed_manifest_sha256':repair['old_manifest_sha256'],'old_source_freeze_sha256':repair['old_source_freeze_sha256'],
        'new_source_freeze_sha256':base.sha(run/'source-freeze.json'),'changed_sources':changed,
        'both_attempts_zero_fit_no_fresh':not manifest['fits'] and not (run/'challenge').exists(),
        'training_sequences_generated_across_attempts':1024,'paired_pcm_identical':same_pcm,
        'paired_feature_arrays_identical':same_feature_values,'executions':execution,
        'cumulative_outer_execution_seconds':sum(x['elapsed_seconds'] for x in execution),
        'rebindings_or_overwrites_performed':False}


def audit_run(run,output,feasibility_receipt=None):
    started=time.monotonic();run=base.inside(RUNS,run);output=base.inside(run,output)
    require(not output.exists(),'Refuses to overwrite an audit receipt');torch.set_num_threads(1)
    audit=base.Audit();m=audit.json(run/'manifest.json')
    require(m['schema']=='phenometrix-evidence-experiment.v1' and m['status'] in ('complete','feasibility_no_go','prerequisite_no_go'),
        'A terminal scientific result is required; no reads during generation/fitting')
    require(m['arms']==list(ARMS) and m['seeds']==list(SEEDS) and m['maximum_fits']==12 and m['predesignated_candidate']=='targeted__on' and
        m['deadline_utc']==DEADLINE and m['reserved_content_read'] is False,'Declared study/data boundary differs')
    require(0<=m['elapsed_seconds']<=10800 and 0<=m.get('fit_elapsed_seconds',0)<=3600 and
        m['peak_observed_python_rss_bytes']<=16*1024**3 and m['peak_observed_node_rss_bytes']<=2*1024**3 and
        m['peak_observed_mps_allocated_bytes']<=8*1024**3 and m['output_bytes']<=8*1024**3,'Scientific resource ceiling exceeded')
    require(historical.dt(m['completed_at'])<=historical.dt(DEADLINE),'Scientific result exceeds fixed UTC deadline')
    initial=audit.json(run/'source-freeze.json');require(initial['schema']=='phenometrix-evidence-source-freeze.v1' and initial['status']=='sources_frozen' and initial['fits']=={},'Initial source freeze differs')
    for name in ('source_sha256','training_spec_sha256','challenge_spec_sha256','arms','seeds','maximum_fits','deadline_utc','sources_frozen_at'):
        require(initial[name]==m[name],'Initial/final identity differs: '+name)
    prior=audit.json(PRIOR/'manifest.json',PRIOR_SHA)
    require(prior['status']=='complete' and m['prior_manifest']==prior and m['historical_input_sha256']==prior['input_sha256'],'Pinned prior identity differs')
    require(all(m['source_sha256'].get(p)==h for p,h in prior['source_sha256'].items()),'Historical frozen sources changed')
    for p,h in m['source_sha256'].items():audit.bind(REPO/p,h);audit.bind(run/'source-snapshot'/p,h)
    for p,h in m['input_sha256'].items():audit.bind(REPO/p,h)
    repair=audit_repair(audit,run,m,initial)
    prep=audit.json(run/'feasibility/manifest.json',m['feasibility_manifest_sha256'])
    fr=Path(feasibility_receipt).resolve() if feasibility_receipt else run/'feasibility/independent-audit.json'
    proof=audit.json(fr)
    require(proof['schema']=='phenometrix-evidence-feasibility-audit.v1' and proof['pass'] is True and
        proof['preparation_manifest_sha256']==base.sha(run/'feasibility/manifest.json') and
        proof['source_freeze_sha256']==base.sha(run/'source-freeze.json'),'Independent feasibility receipt identity differs')
    compare_nested(audit,proof['feasibility'],m['train_feasibility'],'Independent feasibility decision')
    require(proof['feasibility_pass']==prep['feasibility_pass'],'Independent support outcome differs')
    for p,h in proof['bound_artifact_sha256'].items():audit.bind(REPO/p,h)
    for p,h in proof['audit_source_sha256'].items():audit.bind(REPO/p,h)
    require(historical.dt(m['sources_frozen_at'])<=historical.dt(prep['started_at'])<=historical.dt(prep['completed_at'])<=historical.dt(m['completed_at']),
        'TRAIN feasibility chronology differs')
    result={'schema':'phenometrix-evidence-score-audit.v1','status':'complete','pass_result':True,
        'scientific_status':m['status'],'run_manifest_sha256':base.sha(run/'manifest.json'),
        'feasibility_audit_path':str(fr.relative_to(REPO)),'feasibility_audit_sha256':base.sha(fr),
        'feasibility_pass':proof['feasibility_pass'],'repair_lineage':repair,'fitting_performed':False,'inference_performed':False,'waveform_generation_performed':False,
        'reserved_content_read':False,'limitations':['Artifact/source chronology is not an operating-system access log.',
            'An audit pass validates the declared result and cannot promote a scientific no-go.',
            'Training optimizer execution is not replayed; saved targets, schedules, exposure, tensor identities and update records are checked.']}
    if m['status']!='complete':
        no_fit_artifacts(run,m)
        if m['status']=='feasibility_no_go':
            require(proof['feasibility_pass'] is False and m['development_references_read_for_prerequisites'] is False and
                m['development_references_read_for_final_scoring'] is False,'Feasibility no-go crossed a later reference phase')
        else:
            require(proof['feasibility_pass'] is True,'Prerequisite no-go lacks passed TRAIN support')
            prerequisite=audit.json(run/'prerequisites.json');old_metrics=audit.json(PRIOR/'metrics.json',prior['metrics_sha256'])
            compare_nested(audit,prerequisite['coverage'],old_metrics['coverage'],'Unchanged prerequisite coverage')
            compare_nested(audit,prerequisite['headroom'],old_metrics['headroom']['expanded7'],'Unchanged prerequisite oracle')
            checks={'train_support':True,'raw_fixed_voicing_oracle_headroom':old_metrics['headroom']['expanded7']['fixed-slots']['pass'],
                'train_grouping_loss':old_metrics['coverage']['train']['expanded7']['pass'],
                'development_grouping_loss':old_metrics['coverage']['development']['expanded7']['pass']}
            compare_nested(audit,checks,prerequisite['checks'],'Prerequisite no-go arithmetic')
            require(not all(checks.values()) and prerequisite['pass'] is False and m['development_references_read_for_final_scoring'] is False,
                'Prerequisite no-go decision differs')
        require(repair['both_attempts_zero_fit_no_fresh'],'Both no-fit attempt boundaries must hold')
        result['limitations'][-1]='No fits were performed; postfit scores, initialization reproduction and training exposure checks were not applicable or executed.'
        result.update(no_fit_no_fresh_verified=True,scientific_gate_pass=False,new_fits=0,fresh_cases_generated=0)
    else:
        require(proof['feasibility_pass'] is True,'Fits occurred after insufficient TRAIN support')
        input_freeze=audit.json(run/'input-cache-freeze.json');prerequisite=audit.json(run/'prerequisites.json')
        training=audit.json(run/'training-freeze.json');checkpoint=audit.json(run/'checkpoint-freeze.json');prediction=audit.json(run/'prediction-freeze.json')
        for value in (input_freeze,training,checkpoint,prediction):require(value['source_sha256']==m['source_sha256'],'Scientific source changed across phases')
        phase_order(m,initial,input_freeze,prerequisite,training,checkpoint,prediction)
        require(checkpoint['schema']=='phenometrix-evidence-checkpoint-freeze.v1' and checkpoint['source_freeze_path']==str((run/'source-freeze.json').relative_to(REPO)) and
            checkpoint['source_freeze_sha256']==base.sha(run/'source-freeze.json') and checkpoint['frozen_at']==m['checkpoints_frozen_at'],'Final checkpoint source identity differs')
        for name in ('training_spec_sha256','challenge_spec_sha256','arms','seeds','maximum_fits','deadline_utc'):require(checkpoint[name]==m[name],'Checkpoint contract changed: '+name)
        expected_checkpoints={name:{'path':str((run/'checkpoints'/(name+'.pt')).relative_to(REPO)),'sha256':fit['weights_sha256']} for name,fit in m['fits'].items()}
        require(checkpoint['checkpoints']==expected_checkpoints and prediction['fits']==m['fits'],'Final checkpoint freeze differs')
        require(prediction['prediction_sha256']==m['prediction_sha256'],'Prediction freeze changed')
        for phase in (input_freeze,training,prediction):
            require(all(m['input_sha256'].get(p)==h for p,h in phase['input_sha256'].items()),'Bound input changed across phases')
        reference_paths=set()
        for split in ('train','development'):
            _,_,metadata=base.load_cache(audit,split)
            paths={str((ROOT/'data/ptdb-tug'/r['path']).relative_to(REPO)) for r in metadata.values()};reference_paths|=paths
            refs={str(Path(p).relative_to(REPO)) if Path(p).is_absolute() else p:h for p,h in prerequisite['reference_sha256'][split].items()}
            require(set(refs)==paths,'Prerequisite REF-role inventory differs')
            require(all(m['input_sha256'].get(p)==h and prediction['input_sha256'].get(p)==h for p,h in refs.items()),'Prerequisite REF identity changed')
        retention.no_reference_inputs(input_freeze['input_sha256'],reference_paths)
        inventory=audit.json(bound(audit,m,run/'analytic-inventory.json'))
        expected_predictions={'predictions/'+name+'.npz' for name in METHODS}|{r['prediction_path'] for rows in inventory.values() for r in rows}
        require(set(m['prediction_sha256'])==expected_predictions and len(expected_predictions)==103,'Complete prediction inventory differs')
        for p,h in m['prediction_sha256'].items():audit.bind(run/p,h)
        metrics=audit.json(run/'metrics.json',m['metrics_sha256']);require(metrics['schema']=='phenometrix-evidence-metrics.v1','Metrics schema differs')
        ck=verify_checkpoints(audit,run,m,prior)
        cells,coverage,headroom,targets,natural=natural_and_coverage(audit,run,m,metrics,prior)
        compare_nested(audit,coverage,prerequisite['coverage'],'Prerequisite coverage')
        compare_nested(audit,headroom,prerequisite['headroom'],'Prerequisite oracle')
        compare_nested(audit,metrics['exclusions'],prerequisite['exclusions'],'Prerequisite exclusions')
        checks={'train_support':True,'raw_fixed_voicing_oracle_headroom':headroom['fixed-slots']['pass'],
            'train_grouping_loss':coverage['train']['expanded7']['pass'],'development_grouping_loss':coverage['development']['expanded7']['pass']}
        require(all(checks.values()),'Prerequisites should have stopped fitting')
        compare_nested(audit,checks,prerequisite['checks'],'Prerequisite components');compare_nested(audit,checks,metrics['decision']['prerequisites'],'Final prerequisite components')
        train=audit_training(audit,run,m,targets);analytic=audit_analytic(audit,run,m,metrics,prior)
        gate=natural_gate(cells);compare_nested(audit,gate,{k:metrics['decision'][k] for k in gate},'Natural decision')
        accepted=gate['pass'] and analytic['gates']['pass'] and all(checks.values())
        audit.equal(accepted,metrics['decision']['promising_for_next_study'],'Combined scientific gate')
        controls=3*natural['development_rows']+analytic['control_frame_pairs']
        compare_nested(audit,{'state_hz_frame_pairs':controls,'mismatches':0,'all_exact':True},m['control_reproduction'],'Frozen controls')
        result.update(new_fits=12,fresh_cases_generated=12,checkpoint_checks=ck,training=train,natural=natural,
            analytic=analytic,natural_gate=gate,scientific_gate_pass=accepted,control_frame_pairs=controls)
    modules=[Path(__file__),ROOT/'tests/test_audit_evidence_scores.py',Path(base.__file__),Path(historical.__file__),Path(retention.__file__),
        Path(pool.__file__),Path(feasibility_audit.__file__),Path(pool.augmentation.__file__),Path(base.score.__code__.co_filename)]
    for p in modules:audit.bind(p)
    audit.close();result.update(audited_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
        numeric_comparisons=audit.comparisons,maximum_numeric_difference=audit.max_rate_delta,
        bound_artifacts=len(audit.files),bound_artifact_sha256=audit.files,
        audit_source_sha256={str(p.resolve().relative_to(REPO)):base.sha(p) for p in modules})
    output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--feasibility-audit',type=Path)
    args=parser.parse_args()
    result=audit_run(args.run,args.output,args.feasibility_audit)
    print(json.dumps({k:result[k] for k in ('status','pass_result','scientific_status','scientific_gate_pass','numeric_comparisons','maximum_numeric_difference','elapsed_seconds','bound_artifacts')},indent=2))


if __name__=='__main__':main()
