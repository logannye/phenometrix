"""Independent audit for the bounded physical-augmentation comparison.

No fitting or waveform generation. Natural/analytic scoring is reconstructed
separately from the runner; this module never imports its score or gate logic.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from . import audit_selector_experiment as base
from . import audit_harmonic_experiment as historical
from . import audit_subharmonic_experiment as previous
from .model import ModelConfig, numpy_step

REPO, ROOT, RUNS = base.REPO, base.ROOT, base.RUNS
SEEDS=base.SEEDS
ARMS=('natural-only','stationary-aug','balance-aug')
NEW=[f'{arm}__seed{seed}' for seed in SEEDS for arm in ARMS]
PRIOR=[f'prior__relative__acceptable-set__seed{seed}' for seed in SEEDS]
SELECTORS=NEW+PRIOR
METHODS=['original_mlp','original_hybrid',*SELECTORS]
CANDIDATE=[f'balance-aug__seed{seed}' for seed in SEEDS]
SYNTHETIC_ROWS=18336
SYNTHETIC_BATCH=512
STEPS_PER_EPOCH=33
EPOCHS=12
require=base.require
canonical,dt,load_npz=historical.canonical,historical.dt,historical.load_npz
curve,regions=historical.curve,historical.regions
analytic_summary,independent_step=historical.analytic_summary,historical.independent_step
validate_feature=historical.validate_feature


def paired_schedule(seed,epoch):
    """Independent exact index contract, with no candidate-dependent resampling."""
    require(seed in SEEDS and 0<=epoch<EPOCHS,'Undeclared sampler seed or epoch')
    order=np.random.default_rng(seed+1_000_000+epoch).permutation(SYNTHETIC_ROWS)
    return order[:STEPS_PER_EPOCH*SYNTHETIC_BATCH].reshape(STEPS_PER_EPOCH,SYNTHETIC_BATCH)


def synthetic_acceptable(emitted_hz,reference_hz):
    """All analytic TRAIN rows are voiced; unavailable groups receive no target."""
    hz=np.asarray(emitted_hz,dtype=np.float64);reference=np.asarray(reference_hz,dtype=np.float64)
    require(hz.ndim==2 and reference.shape==hz.shape[:1],'Synthetic target shape differs')
    require(np.isfinite(reference).all() and (reference>0).all(),'Synthetic latent F0 invalid')
    valid=np.isfinite(hz)&(hz>0)
    with np.errstate(invalid='ignore',divide='ignore'):
        acceptable=valid&(np.abs(1200*np.log2(hz/reference[:,None]))<=50)
    return acceptable,acceptable.any(axis=-1)


def validate_paired_target_metadata(left,right):
    """Condition may change coverage, never latent targets or paired row identity."""
    names=('reference_hz','scenario_index','reference_sample','availability_sample','region')
    require(set(left)==set(right)==set(names)|{'covered','acceptable'},'Synthetic target field inventory differs')
    for name in names:
        require(left[name].shape==(SYNTHETIC_ROWS,) and np.array_equal(left[name],right[name]),'Unpaired synthetic metadata: '+name)
    for item in (left,right):
        require(item['acceptable'].shape==(SYNTHETIC_ROWS,6) and item['acceptable'].dtype==np.bool_ and item['covered'].dtype==np.bool_,'Synthetic target mask contract differs')
        require(np.array_equal(item['covered'],item['acceptable'].any(-1)),'Synthetic coverage differs from acceptable set')
        require(np.array_equal(item['availability_sample']-item['reference_sample'],np.full(SYNTHETIC_ROWS,960)),'Synthetic 20ms delay differs')
    require(np.array_equal(np.bincount(left['scenario_index'],minlength=96),np.full(96,191)),'Synthetic scenario row counts differ')
    return True


def sampler_exposure(schedule,scenario_index,covered):
    """All selected rows count; zero-loss uncovered rows are retained."""
    require(schedule.shape==(STEPS_PER_EPOCH,SYNTHETIC_BATCH),'Synthetic schedule shape differs')
    require(schedule.dtype==np.int64 and len(np.unique(schedule))==schedule.size,'Synthetic schedule duplicates or dtype differs')
    require(schedule.min()>=0 and schedule.max()<SYNTHETIC_ROWS,'Synthetic schedule outside cache')
    selected=schedule.reshape(-1)
    return {'selected_rows':len(selected),'covered_rows':int(covered[selected].sum()),
            'uncovered_zero_rows':int((~covered[selected]).sum()),
            'scenario_rows':np.bincount(scenario_index[selected],minlength=96).tolist()}

PRIOR_RUN=RUNS/'harmonic-context-v1'
LAST=RUNS/'subharmonic-objective-v1'
PINS=previous.PINS
PREREQUISITES_SHA=previous.PREREQUISITES_SHA
LAST_MANIFEST_SHA='d3600c00e099e4aa4f426dfbdb46399d968b13189f4a878e69beff39e3f72c17'
TRAIN_FREQUENCIES=(70.25,90.25,110.25,130.25,150.25,170.25,190.25,210.25,230.25,250.25,270.25,290.25,310.25,320.25,330.25,340.25,350.25,360.25,370.25,390.25,430.25,490.25,570.25,650.25)


def expected_training_records():
    records={arm:[] for arm in ARMS[1:]}
    for i,f in enumerate(TRAIN_FREQUENCIES):
        for j,phases in enumerate(((.11,.73,-.29),(-.61,1.07,.41))):
            for k,gain in enumerate((.04,.08)):
                pair=f'augmentation-pair-f{i}-p{j}-g{k}'
                for arm in ARMS[1:]:
                    records[arm].append({'id':f'augmentation-{arm}-f{i}-p{j}-g{k}', 'frequency_hz':f,
                        'phase_radians':list(phases),'gain':gain,'sample_rate_hz':48000,'sample_count':96000,
                        'change_sample':39840,'before':[1.,.5,.25],
                        'after':[1.,.5,.25] if arm=='stationary-aug' else [0.,1.,.5], 'pair_id':pair,'arm':arm})
    return records


def expected_challenge_records():
    return [{'id':f'augmentation-balance-f{i}-p{j}','frequency_hz':f,'phase_radians':list(phases),'gain':.06,
             'sample_rate_hz':48000,'sample_count':96000,'change_sample':39840,'before':[1.,.5,.25],'after':[0.,1.,.5]}
            for i,f in enumerate((93.7,193.9,331.7,366.1,517.3,641.9))
            for j,phases in enumerate(((.43,-1.21,.17),(1.67,.31,-.93)))]

def training_targets(audit,run,manifest):
    folder,cache,metadata=base.load_cache(audit,'train')
    refs={};raw_parts=[];acceptable_parts=[];hz_parts=[]
    for record in cache['recordings']:
        source=record['source_recording_id']
        if source not in refs:refs[source]={r.sample:r for r in base.references(audit,metadata[source],record,'train')}
        f=base.load_feature(audit,folder,record);g=base.representative_hz(f)
        raw=np.zeros(len(g),bool);acceptable=np.zeros(g.shape,bool)
        for i,target in enumerate(f['reference_sample']-960):
            ref=refs[source].get(int(target))
            if ref is not None and ref.valid and ref.eligible and ref.voiced:
                with np.errstate(invalid='ignore',divide='ignore'):
                    raw[i]=np.any(f['candidate_mask'][i] & (np.abs(1200*np.log2(f['candidate_hz'][i]/ref.f0_hz))<=50))
                    acceptable[i]=np.isfinite(g[i]) & (np.abs(1200*np.log2(g[i]/ref.f0_hz))<=50)
        require(not np.any(acceptable.any(-1)&~raw),'Grouping gained non-member coverage')
        raw_parts.append(raw);acceptable_parts.append(acceptable);hz_parts.append(g)
    raw=np.concatenate(raw_parts);schedule=np.flatnonzero(raw)
    acceptable=np.concatenate(acceptable_parts)[schedule];hz=np.concatenate(hz_parts)[schedule]
    covered=acceptable.any(-1)
    expected={'schedule':schedule,'covered':covered,'acceptable':acceptable}
    saved=load_npz(audit,run/'training-targets.npz',manifest['training_target_sha256'])
    require(set(saved)==set(expected),'Training target inventory differs')
    for name,value in expected.items():
        require(saved[name].dtype==value.dtype and np.array_equal(saved[name],value),'Training target differs: '+name)
    audit.equal(len(schedule),132181,'Declared original schedule');audit.equal(int((~covered).sum()),107,'Grouped uncovered schedule')
    audit.equal(len(schedule),manifest['scheduled_rows'],'Scheduled rows');audit.equal(int((~covered).sum()),manifest['grouped_uncovered_scheduled_rows'],'Uncovered rows')
    updates={}
    for seed in SEEDS:
        skipped=0
        for epoch in range(12):
            order=np.random.default_rng(seed+epoch).permutation(len(schedule))
            skipped+=sum(not covered[order[start:start+4096]].any() for start in range(0,len(order),4096))
        updates[seed]=(12*math.ceil(len(schedule)/4096)-skipped,skipped)
        require(updates[seed]==(396,0),'Expected update schedule differs')
    return updates,{'scheduled_rows':len(schedule),'grouped_uncovered_rows':int((~covered).sum()),
                   'target_fields_exact':list(expected),
                   'training_target_sha256':base.sha(run/'training-targets.npz')}

def checkpoint_identity(audit,run,manifest,updates):
    initial={};scaling=None
    for seed in SEEDS:
        for arm in ARMS:
            mode,objective='relative','acceptable-set'
            name=f'{arm}__seed{seed}';fit=manifest['fits'][name]
            require((fit['seed'],fit['arm'])==(seed,arm),'Fit identity differs')
            require(len(fit['epoch_losses'])==12 and all(math.isfinite(x) for x in fit['epoch_losses']),'Epoch losses invalid')
            for field in ('epoch_base_losses','epoch_weighted_auxiliary_losses'):
                require(len(fit[field])==12 and all(math.isfinite(x) and x>=-1e-7 for x in fit[field]),'Loss contributions invalid')
            require(np.allclose(np.array(fit['epoch_losses']),np.array(fit['epoch_base_losses'])+np.array(fit['epoch_weighted_auxiliary_losses']),rtol=0,atol=1e-7),'Epoch loss decomposition differs')
            if arm=='natural-only':require(all(x==0 for x in fit['epoch_weighted_auxiliary_losses']),'Set control has auxiliary loss')
            prior_update=updates[seed]
            audit.equal(fit['optimizer_updates'],prior_update[0],name+'/updates');audit.equal(fit['skipped_batches'],prior_update[1],name+'/skips')
            saved=torch.load(audit.bind(run/'checkpoints'/(name+'.pt'),fit['weights_sha256']),map_location='cpu',weights_only=True)
            require(saved['arm']==arm and saved['context_mode']==mode and saved['config']=={'representation':'shared-candidates','loss':objective,'hidden_size':32},'Saved model configuration differs')
            shapes={'scaling_fitted':(),'candidate_mean':(8,),'candidate_scale':(8,),'global_mean':(9,),'global_scale':(9,)}
            for layer,shape in {'candidate_encoder':(32,8),'shared_hidden':(32,73),'shared_head':(1,32),'message_encoder':(32,9)}.items():
                shapes[layer+'.weight']=shape;shapes[layer+'.bias']=(shape[0],)
            state=saved['state_dict'];require(set(state)==set(shapes),'Tensor inventory differs')
            for field,shape in shapes.items():
                t=state[field];require(tuple(t.shape)==shape,'Tensor shape differs')
                if field=='scaling_fitted':require(t.dtype==torch.bool and bool(t),'Scaling not fitted')
                else:
                    require(t.dtype==torch.float32 and bool(torch.isfinite(t).all()),'Tensor dtype/value differs')
                    if field.endswith('scale'):require(bool((t>0).all()),'Nonpositive scale')
            buffers={k:state[k].numpy() for k in ('candidate_mean','candidate_scale','global_mean','global_scale')}
            if scaling is not None:require(all(np.array_equal(v,scaling[k]) for k,v in buffers.items()),'Scaling differs across arms/seeds')
            scaling=buffers
            audit.equal(sum(v.numel() for k,v in state.items() if k.endswith(('.weight','.bias'))),3009,name+'/tensor parameters')
            audit.equal(fit['parameter_count'],3009,name+'/reported parameters')
            digest=fit['initial_tensors_sha256'];require(len(digest)==64,'Initialization digest missing')
            if seed in initial:require(initial[seed]==digest,'Same-seed initialization identities differ')
            initial[seed]=digest
    return {'new_checkpoints':9,'parameters_each':3009,'same_seed_initialization_hashes':initial,
            'initialization_note':'Equality of declared initialization digests checked here; separate inference audit reconstructs initialization.'}

def grouped_natural_and_prior(audit,run,manifest):
    arrays={m:load_npz(audit,run/'predictions'/(m+'.npz'),manifest['prediction_sha256']['predictions/'+m+'.npz']) for m in SELECTORS}
    prior_manifest=audit.json(PRIOR_RUN/'manifest.json',PINS['manifest.json'])
    for m in PRIOR:
        original=m.removeprefix('prior__');path=PRIOR_RUN/'predictions'/(original+'.npz')
        old=load_npz(audit,path,prior_manifest['prediction_sha256']['predictions/'+original+'.npz'])
        require(all(np.array_equal(arrays[m][k],old[k],equal_nan=True) for k in ('state','hz')),'Prior saved natural output changed')
        audit.bind(PRIOR_RUN/'checkpoints'/(original+'.pt'),prior_manifest['fits'][original]['weights_sha256'])
    folder,cache,_=base.load_cache(audit,'development');offset=0;checked=0
    for record in cache['recordings']:
        f=base.load_feature(audit,folder,record);g=base.representative_hz(f);stop=offset+len(g)
        for m,z in arrays.items():
            state,hz=z['state'][offset:stop],z['hz'][offset:stop];v=state==1
            require(np.all(np.any(g[v]==hz[v,None],axis=1)),'Natural output not an actual grouped representative: '+m)
            checked+=int(v.sum())
        offset=stop
    return {'grouped_representative_emissions_checked':checked,'prior_natural_methods_unchanged':3}

def analytic(audit,run,manifest,metrics,items):
    gates={'stationary_by_seed_case':{m:{} for m in CANDIDATE},'harmonic_change_by_seed_case':{m:{} for m in CANDIDATE},'consumed_challenge_by_seed_case':{m:{} for m in CANDIDATE},'consumed_subharmonic_by_seed_case':{m:{} for m in CANDIDATE},'challenge_by_seed_case':{m:{} for m in CANDIDATE}}
    summaries={};frames=0;emissions=0;maximum_probability_difference=0.
    for group,name,scenario,f,p in items:
        n=96000 if group!='regression' else 76800;end,target,eligible=validate_feature(f,n)
        require(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Invalid original voicing probability')
        grouped=base.representative_hz(f);truth=curve(scenario,target/48000);region=regions(scenario,end)
        coverage=np.any(np.abs(1200*np.log2(grouped/truth[:,None]))<=50,axis=1)&eligible
        report=metrics[group][name];require(set(report['methods'])==set(SELECTORS),'Analytic method inventory differs')
        for k,v in {'eligible_frames':int(eligible.sum()),'covered_frames':int(coverage.sum()),'fraction':float(coverage.sum()/eligible.sum())}.items():audit.equal(v,report['candidate_coverage'][k],name+'/candidate coverage/'+k)
        path=run/'predictions'/(group+'-'+name+'.npz');z=load_npz(audit,path,manifest['prediction_sha256'][str(path.relative_to(run))])
        require(set(z)=={'original_probability'}|{m+s for m in SELECTORS for s in ('__state','__hz')},'Analytic prediction field inventory differs')
        stored=z['original_probability']
        require(stored.shape==p.shape and np.isfinite(stored).all() and ((stored>=0)&(stored<=1)).all(),'Stored probability invalid')
        require(np.array_equal(stored<.5,p<.5),'Independent original voiced threshold differs')
        delta=float(np.max(np.abs(stored-p)))
        maximum_probability_difference=max(maximum_probability_difference,delta)
        require(delta<=5e-6,'Independent original probability differs')
        if group=='regression':require(np.array_equal(stored,p),'Frozen regression probability changed')
        case={}
        for method in SELECTORS:
            state,hz=z[method+'__state'],z[method+'__hz'];base.prediction_objects(target,state,hz)
            require(np.array_equal(state==0,p<.5),'Analytic original voicing changed')
            expected_state=np.where(p<.5,0,np.where(np.isfinite(grouped).any(axis=1),1,2));require(np.array_equal(state,expected_state),'Analytic availability changed')
            v=state==1;require(np.all(np.any(grouped[v]==hz[v,None],axis=1)),'Analytic emitted nonrepresentative frequency')
            emissions+=int(v.sum());frames+=len(state)
            selections={'all':eligible,**{r:eligible&(region==r) for r in sorted(set(region[eligible]))}}
            for label,selection in selections.items():
                got=analytic_summary(truth[selection],state[selection],hz[selection],region[selection],scenario['slopeHzPerSecond'])
                want=report['methods'][method]['all'] if label=='all' else report['methods'][method]['regions'][label]
                require(set(got)==set(want),'Analytic summary inventory differs')
                for k,value in got.items():audit.equal(value,want[k],name+'/'+method+'/'+label+'/'+k)
            if scenario['kind']=='step':
                step=independent_step(scenario,target[eligible],end[eligible],state[eligible],hz[eligible])
                for k,value in step.items():audit.equal(value,report['methods'][method]['step_response'][k],name+'/'+method+'/step/'+k)
            else:require(report['methods'][method]['step_response'] is None,'Unexpected step outcome')
            if method in CANDIDATE:
                if group!='regression':gate=group+'_by_seed_case';selection=eligible&(region=='after-observation-change')
                elif scenario['kind']=='constant':gate='stationary_by_seed_case';selection=eligible
                elif scenario['kind']=='observation-change':gate='harmonic_change_by_seed_case';selection=eligible&(region=='after-observation-change')
                else:continue
                summary=analytic_summary(truth[selection],state[selection],hz[selection],region[selection],0)
                passed=summary['success_within_50c_all_frames']>=.95;gates[gate][method][name]=passed
                case[method]={'frames':summary['frames'],'success':summary['success_within_50c_all_frames'],'pass':passed}
        summaries[group+'/'+name]=case
    for key,count in [('stationary_by_seed_case',3),('harmonic_change_by_seed_case',3),('consumed_challenge_by_seed_case',12),('consumed_subharmonic_by_seed_case',12),('challenge_by_seed_case',12)]:
        require(all(len(cases)==count for cases in gates[key].values()),'Incomplete gate case inventory')
        require(gates[key]==metrics['decision']['analytic'][key],'Analytic per-seed gate differs')
    for flag,key in [('stationary_pass','stationary_by_seed_case'),('harmonic_change_pass','harmonic_change_by_seed_case'),('consumed_challenge_pass','consumed_challenge_by_seed_case'),('consumed_subharmonic_pass','consumed_subharmonic_by_seed_case'),('challenge_pass','challenge_by_seed_case')]:
        gates[flag]=all(all(cases.values()) for cases in gates[key].values());audit.equal(gates[flag],metrics['decision']['analytic'][flag],flag)
    return {'cases':50,'selector_methods_per_case':12,'maximum_original_probability_difference':maximum_probability_difference,'prediction_rows_checked':frames,'grouped_emissions_checked':emissions,'candidate_case_gates':summaries,'gates':gates}

def natural_decision(audit,cells,reported):
    mean=lambda m,k:float(np.mean([v[k] for v in cells[m].values()]))
    error=float(np.mean([mean(n,'joint_error') for n in CANDIDATE]));success=float(np.mean([mean(n,'reference_voiced_success_rate') for n in CANDIDATE]))
    gains={b:mean(b,'joint_error')-error for b in ('original_mlp','original_hybrid')}
    excess={c:float(np.mean([v['joint_error'] for n in CANDIDATE for k,v in cells[n].items() if k.endswith('/'+c)]))-float(np.mean([v['joint_error'] for k,v in cells['original_mlp'].items() if k.endswith('/'+c)])) for c in base.CONDITIONS}
    checks={'mean_improves_both_by_half_point':all(v>=.005 for v in gains.values()),'each_seed_improves_original_mlp':all(mean(n,'joint_error')<mean('original_mlp','joint_error') for n in CANDIDATE),'no_condition_worsens_over_one_point':all(v<=.01 for v in excess.values()),'voiced_success_not_lower':success>=mean('original_mlp','reference_voiced_success_rate')}
    checks={k:bool(v) for k,v in checks.items()};require(checks==reported['checks'],'Natural gate components differ')
    audit.equal(all(checks.values()),reported['pass'],'Natural pass');audit.equal(error,reported['mean_joint_error'],'Natural mean joint error');audit.equal(success,reported['mean_voiced_success'],'Natural mean success')
    for k,v in gains.items():audit.equal(v,reported['joint_error_improvement_over_frozen'][k],'Natural improvement/'+k)
    for k,v in excess.items():audit.equal(v,reported['condition_error_excess_over_mlp'][k],'Natural condition excess/'+k)
    control=float(np.mean([mean(f'natural-only__seed{s}','joint_error') for s in SEEDS]))
    audit.equal(control-error,reported['improvement_over_refitted_natural_control'],'Refitted set control comparison')
    return checks

def cache_record(audit,folder,params,row):
    require(row['id']==params['id'] and row['parameters_sha256']==hashlib.sha256(canonical(params)).hexdigest(),'Cached parameters changed')
    pcm=audit.bind(base.inside(folder,row['pcm_path']),row['pcm_sha256'])
    require(pcm.stat().st_size==384000 and np.isfinite(np.fromfile(pcm,dtype='<f4')).all(),'PCM extent/finiteness differs')
    f=load_npz(audit,base.inside(folder,row['feature_path']),row['feature_sha256'])
    end,target,eligible=validate_feature(f,96000)
    require(len(f['features'])==row['feature_frames'] and int(eligible.sum())==191,'Cached feature extent differs')
    return f,end,target,eligible


def cache_sources(audit,folder,cm,manifest):
    for source,digest in cm['source_sha256'].items():
        require(manifest['source_sha256'].get(source)==digest,'Cache source absent from prefit closure')
        audit.bind(folder/'source-snapshot'/source,digest)


def synthetic_training(audit,run,manifest,sf,cf):
    folder=run/'training'
    cm=audit.json(folder/'manifest.json',manifest['input_sha256'][str(folder/'manifest.json')])
    spec=audit.json(folder/'spec.json',manifest['input_sha256'][str(folder/'spec.json')])
    expected=expected_training_records();require(spec['records_by_arm']==expected,'TRAIN configuration differs')
    require(cm['schema']=='phenometrix-paired-harmonic-augmentation-cache-v1' and cm['status']=='complete' and cm['split']=='train' and cm['reference_content_read'] is False,'TRAIN cache scope differs')
    require(cm['freeze_kind']=='source' and cm['freeze_path']==str((run/'source-freeze.json').relative_to(REPO)) and cm['freeze_sha256']==base.sha(run/'source-freeze.json'),'TRAIN source freeze differs')
    digest=hashlib.sha256(canonical(spec)).hexdigest()
    require(digest==cm['spec_sha256']==cm['training_spec_sha256']==sf['training_spec_sha256']==manifest['training_spec_sha256'],'TRAIN specification hash differs')
    require(cm['challenge_spec_sha256']==sf['challenge_spec_sha256']==manifest['challenge_spec_sha256'],'Challenge specification changed during TRAIN generation')
    require(dt(sf['sources_frozen_at'])<=dt(cm['started_at'])<=dt(cm['completed_at'])<=dt(cf['checkpoints_frozen_at']),'TRAIN generation chronology differs')
    require(cm['source_freeze_path']==cm['freeze_path'] and cm['source_freeze_sha256']==cm['freeze_sha256'],'Explicit source-freeze alias differs')
    cache_sources(audit,folder,cm,manifest)
    flattened=[(arm,p) for arm in ARMS[1:] for p in expected[arm]]
    require([v['id'] for v in cm['records']]==[p['id'] for _,p in flattened],'TRAIN row inventory/order differs')
    pieces={arm:{k:[] for k in ('covered','acceptable','reference_hz','scenario_index','reference_sample','availability_sample','region')} for arm in ARMS[1:]}
    for offset,((arm,params),row) in enumerate(zip(flattened,cm['records'])):
        require(row['arm']==arm and row['pair_id']==params['pair_id'],'TRAIN pair identity differs')
        f,end,target,eligible=cache_record(audit,folder,params,row);grouped=base.representative_hz(f)[eligible]
        reference=np.full(191,params['frequency_hz']);acceptable,covered=synthetic_acceptable(grouped,reference)
        labels=np.where(end[eligible]<=39840,'before-observation-change',np.where(end[eligible]-1920>=39840,'after-observation-change','observation-boundary'))
        values={'covered':covered,'acceptable':acceptable,'reference_hz':reference,'scenario_index':np.full(191,offset%96,np.int64),
                'reference_sample':target[eligible],'availability_sample':end[eligible],'region':labels}
        for k,v in values.items():pieces[arm][k].append(v)
    banks={arm:{k:np.concatenate(v) for k,v in p.items()} for arm,p in pieces.items()}
    validate_paired_target_metadata(banks['stationary-aug'],banks['balance-aug'])
    saved=load_npz(audit,run/'synthetic-targets.npz',manifest['synthetic_target_sha256'])
    require(set(saved)=={arm+'__'+k for arm,p in banks.items() for k in p},'Saved synthetic field inventory differs')
    for arm,p in banks.items():
        for k,v in p.items():require(np.array_equal(v,saved[arm+'__'+k]),'Synthetic target/metadata differs: '+arm+'/'+k)
        pool={'rows':18336,'covered_rows':int(p['covered'].sum()),'uncovered_zero_rows':int((~p['covered']).sum())}
        require(pool==manifest['synthetic_pool'][arm],'Synthetic pool counts differ')
    orders=load_npz(audit,run/'synthetic-schedules.npz',manifest['synthetic_schedule_sha256'])
    require(set(orders)=={f'seed{s}__epoch{e}' for s in SEEDS for e in range(12)},'Synthetic schedule inventory differs')
    all_exposure={}
    for seed in SEEDS:
        require(manifest['fits'][f'natural-only__seed{seed}']['synthetic_exposure_by_epoch']==[],'Natural-only consumed synthetic loss')
        for arm in ARMS[1:]:
            p=banks[arm];epochs=[]
            for epoch in range(12):
                order=orders[f'seed{seed}__epoch{epoch}'];require(np.array_equal(order,paired_schedule(seed,epoch)),'Exact paired sampler differs')
                summary=sampler_exposure(order,p['scenario_index'],p['covered']);ix=order.reshape(-1);covered=p['covered'][ix]
                expected_exposure={'selected_rows':16896,'unique_rows':16896,'covered_rows':summary['covered_rows'],'uncovered_zero_rows':summary['uncovered_zero_rows'],
                    'per_case':[{'case_index':i,'selected_rows':int((p['scenario_index'][ix]==i).sum()),'covered_rows':int(((p['scenario_index'][ix]==i)&covered).sum())} for i in range(96)],
                    'per_region':{label:{'selected_rows':int((p['region'][ix]==label).sum()),'covered_rows':int(((p['region'][ix]==label)&covered).sum())} for label in np.unique(p['region'])}}
                require(expected_exposure==manifest['fits'][f'{arm}__seed{seed}']['synthetic_exposure_by_epoch'][epoch],'Synthetic exposure ledger differs')
                epochs.append(expected_exposure)
            all_exposure[f'{arm}__seed{seed}']=epochs
    return {'cases_per_arm':96,'rows_per_arm':18336,'pool':manifest['synthetic_pool'],'target_fields_exact':list(saved),
            'schedules_exact':36,'selected_rows_per_epoch':16896,'training_spec_sha256':digest,
            'started_at':cm['started_at'],'completed_at':cm['completed_at'],'exposure_by_model_epoch':all_exposure}


def analytic_items(audit,run,manifest,cf):
    original=audit.json(RUNS/'delayed-known-truth/prediction-freeze.json')
    support=audit.json(RUNS/'support-diagnostic-v1/manifest.json',original['original_support_manifest_sha256'])
    specs={s['id']:s for s in support['scenarios']}
    for hz in (100,200,300):
        name=f'observation-change-{hz}'
        specs[name]={'id':name,'kind':'observation-change','durationSeconds':1.6,'initialHz':hz,'finalHz':hz,'changeStartSeconds':.6,'changeEndSeconds':.6,'slopeHzPerSecond':0}
    require(set(specs)==base.SYNTHETIC_IDS,'Original regression inventory differs')
    records=[]
    for row in original['recordings']:
        name=row['id'];f=load_npz(audit,RUNS/'delayed-known-truth/features'/(name+'.npz'),row['feature_sha256'])
        path=audit.bind(RUNS/'delayed-known-truth/predictions'/(name+'.json.gz'),row['prediction_sha256'])
        p=json.loads(gzip.decompress(path.read_bytes()))['predictions']['mlp']
        require([v['sample'] for v in p]==(f['reference_sample']-960).tolist(),'Original probability grid differs')
        records.append(('regression',name,specs[name],f,np.array([v['voicing_probability'] for v in p])))
    saved=torch.load(audit.bind(RUNS/'delayed-mlp-seed20261022/weights.pt',base.WEIGHTS),map_location='cpu',weights_only=True)
    config=ModelConfig(**saved['config']);weights={k:v.numpy() for k,v in saved['state_dict'].items()};receipts={}
    groups=[('consumed_challenge',PRIOR_RUN,historical.expected_challenge_records()),
            ('consumed_subharmonic',LAST,previous.expected_challenge_records()),('challenge',run,expected_challenge_records())]
    for group,parent,expected in groups:
        folder=parent/'challenge';cm=audit.json(folder/'manifest.json',manifest['input_sha256'][str(folder/'manifest.json')]);spec=audit.json(folder/'spec.json',manifest['input_sha256'][str(folder/'spec.json')])
        require(spec['records']==expected and len(cm['records'])==12,'Challenge configurations differ')
        digest=hashlib.sha256(canonical(spec)).hexdigest();require(digest==cm['spec_sha256'],'Canonical challenge hash differs')
        require(cm['status']=='complete' and cm['reference_content_read'] is False,'Challenge state differs')
        if group=='challenge':
            require(cm['freeze_kind']=='checkpoint' and cm['freeze_path']==str((run/'checkpoint-freeze.json').relative_to(REPO)) and cm['freeze_sha256']==base.sha(run/'checkpoint-freeze.json'),'Fresh challenge parent freeze differs')
            require(cm['checkpoint_freeze_path']==cm['freeze_path'] and cm['checkpoint_freeze_sha256']==cm['freeze_sha256'],'Explicit checkpoint-freeze alias differs')
            require(digest==manifest['challenge_spec_sha256']==cf['challenge_spec_sha256']==cm['challenge_spec_sha256'] and cm['training_spec_sha256']==manifest['training_spec_sha256'],'Fresh spec freeze differs')
            require(dt(cf['checkpoints_frozen_at'])<=dt(cm['started_at'])<=dt(cm['completed_at'])<=dt(manifest['predictions_frozen_at']),'Fresh challenge phase order differs')
        else:
            require(cm['checkpoint_freeze_sha256']==base.sha(parent/'checkpoint-freeze.json') and cm['checkpoint_freeze_path']==str((parent/'checkpoint-freeze.json').relative_to(REPO)),'Consumed parent freeze differs')
            require(dt(cm['completed_at'])<dt(manifest['started_at']),'Consumed challenge not historical')
        cache_sources(audit,folder,cm,manifest);hashes={}
        for params,row in zip(expected,cm['records']):
            f,_,_,_=cache_record(audit,folder,params,row)
            logits,_,_=numpy_step(config,weights,f['features'],f['candidate_mask'],np.zeros((len(f['features']),32),np.float32))
            require(np.isfinite(logits).all(),'Original voiced output nonfinite')
            probability=np.exp(-np.logaddexp(0.,-logits[:,0].astype(np.float64)))
            hashes[params['id']]=hashlib.sha256(probability.astype('<f8').tobytes()).hexdigest()
            scenario={'id':params['id'],'kind':'observation-change','durationSeconds':2.,'initialHz':params['frequency_hz'],'finalHz':params['frequency_hz'],'changeStartSeconds':.83,'changeEndSeconds':.83,'slopeHzPerSecond':0}
            records.append((group,params['id'],scenario,f,probability))
        receipts[group]={'started_at':cm['started_at'],'completed_at':cm['completed_at'],'spec_sha256':digest,'independent_numpy_voicing_probability_sha256':hashes}
    require(len(records)==50,'Analytic total case count differs')
    return records,receipts

def audit_run(run,output):
    run=base.inside(RUNS,run);output=base.inside(run,output)
    require(not output.exists(),'Refuses to overwrite audit evidence')
    torch.set_num_threads(1);audit=base.Audit();m=audit.json(run/'manifest.json')
    require(m['schema']=='phenometrix-augmentation-experiment.v1' and m['status']=='complete','Completed augmentation run required')
    require(m['seeds']==list(SEEDS) and m['arms']==list(ARMS) and m['predesignated_candidate']=='balance-aug','Run arms differ')
    require(set(m['fits'])==set(NEW) and m['epochs']==12 and m['batch_size']==4096 and m['learning_rate']==.001 and m['measurement_delay_ms']==20 and m['device']=='mps','Schedule/configuration differs')
    require(m['frozen_original_voicing'] is True and m['frozen_original_weights_sha256']==base.WEIGHTS and m['reserved_content_read'] is False,'Original voicing or scope differs')
    require(m['fit_elapsed_seconds']<=1800 and m['peak_observed_mps_allocated_bytes']<=8*1024**3 and sum(p.stat().st_size for p in run.rglob('*') if p.is_file())<=2*1024**3,'Resource declarations differ')
    require(m['prerequisites_sha256']==PREREQUISITES_SHA and m['prior_run_sha256']==PINS,'Historical pins differ')
    pre=audit.json(RUNS/'selector-prerequisites-v1/prerequisites.json',PREREQUISITES_SHA)
    require(pre['pass'] is True,'Passing historical prerequisites required')
    for path,digest in pre['input_sha256'].items():audit.bind(path,digest)
    for name,digest in PINS.items():audit.bind(PRIOR_RUN/name,digest)
    prior=audit.json(PRIOR_RUN/'manifest.json',PINS['manifest.json'])
    require(prior['status']=='failed' and prior['failure']=="TypeError('Object of type bool is not JSON serializable')",'Historical serialization failure changed')
    for path,digest in m['source_sha256'].items():audit.bind(REPO/path,digest);audit.bind(run/'source-snapshot'/path,digest)
    for path,digest in m['input_sha256'].items():audit.bind(path,digest)
    require(all(m['source_sha256'].get(k)==v for k,v in pre['source_sha256'].items()),'Historical source identity changed')
    for split in ('train','development'):
        for path,digest in pre['reference_sha256'][split].items():
            require(m['input_sha256'].get(path)==digest,'Natural reference pin differs');audit.bind(path,digest)
    sf=audit.json(run/'source-freeze.json');cf=audit.json(run/'checkpoint-freeze.json');pf=audit.json(run/'prediction-freeze.json')
    require(sf['status']=='sources_frozen' and sf['fits']=={} and sf['source_sha256']==m['source_sha256'],'Prefit source freeze differs')
    require(sf['training_spec_sha256']==cf['training_spec_sha256']==pf['training_spec_sha256']==m['training_spec_sha256'] and sf['challenge_spec_sha256']==cf['challenge_spec_sha256']==pf['challenge_spec_sha256']==m['challenge_spec_sha256'],'Specifications changed across freezes')
    require(m['augmentation_configuration']['weight']==.25 and m['augmentation_configuration']['batch_size']==512 and m['augmentation_configuration']['scaling']=='natural TRAIN only','Augmentation configuration differs')
    last=audit.json(LAST/'manifest.json',LAST_MANIFEST_SHA)
    require(last['status']=='complete' and all(m['source_sha256'].get(p)==h for p,h in last['source_sha256'].items()),'Prior frozen closure changed')
    audit.bind(LAST/'checkpoint-freeze.json','ac23ab4fa980bca8873ffb85773c431e87e79ddd1ee334f6bf2b62b12a7ce9ae')
    for field in ('synthetic_target_sha256','synthetic_schedule_sha256'):
        require(cf[field]==pf[field]==m[field],'Synthetic targets/schedules changed across freeze')
    require(cf['status']=='checkpoints_frozen' and pf['status']=='predictions_frozen','Freeze states differ')
    require(cf['fits']==pf['fits']==m['fits'] and cf['source_sha256']==pf['source_sha256']==m['source_sha256'] and pf['prediction_sha256']==m['prediction_sha256'],'Frozen identities differ')
    require(cf['training_target_sha256']==pf['training_target_sha256']==m['training_target_sha256'],'Training targets changed after freeze')
    require(cf['development_references_read'] is False and pf['development_references_read'] is False and m['development_references_read'] is True,'Declared reference chronology differs')
    require(not any('/records/development/' in p for p in pf['input_sha256']),'Development reference hash read before prediction freeze')
    require(all(m['input_sha256'].get(p)==h for p,h in pf['input_sha256'].items()) and all(pf['input_sha256'].get(p)==h for p,h in cf['input_sha256'].items()),'Frozen input identities differ')
    times=[dt(m[k]) for k in ('started_at','sources_frozen_at','checkpoints_frozen_at','predictions_frozen_at','completed_at')]
    require(times==sorted(times),'Freeze time ordering differs')
    require(cf['checkpoints_frozen_at']==m['checkpoints_frozen_at'] and pf['predictions_frozen_at']==m['predictions_frozen_at'],'Freeze timestamp identity differs')
    source=(run/'source-snapshot/research/voice-prosody/phenometrix_prosody/augmentation_experiment.py').read_text()
    require(source.index("save(output/'source-freeze.json',m)")<source.index('cases.generate_training(')<source.index("m['status']='fitting'")<source.index("save(output/'checkpoint-freeze.json',m)")<source.index('challenge,h=generate_challenge(')<source.index('for name,model in models.items():predictions[name]=old.infer')<source.index("save(output/'prediction-freeze.json',m)")<source.index('dl=old.load_labels(dev)'),'Source phase ordering differs')
    expected={'predictions/'+n+'.npz' for n in METHODS}|{'predictions/regression-'+n+'.npz' for n in base.SYNTHETIC_IDS}|{'predictions/consumed_challenge-'+r['id']+'.npz' for r in historical.expected_challenge_records()}|{'predictions/consumed_subharmonic-'+r['id']+'.npz' for r in previous.expected_challenge_records()}|{'predictions/challenge-'+r['id']+'.npz' for r in expected_challenge_records()}
    require(set(m['prediction_sha256'])==expected and {str(p.relative_to(run)) for p in (run/'predictions').glob('*.npz')}==expected,'Prediction inventory differs')
    require({p.stem for p in (run/'checkpoints').glob('*.pt')}==set(NEW),'Checkpoint inventory differs')
    for path,digest in m['prediction_sha256'].items():audit.bind(run/path,digest)
    metrics=audit.json(run/'metrics.json',m['metrics_sha256'])
    require(metrics['schema']=='phenometrix-augmentation-metrics.v1','Metrics schema differs')
    for group,ids in [('regression',base.SYNTHETIC_IDS),('consumed_challenge',{r['id'] for r in historical.expected_challenge_records()}),('consumed_subharmonic',{r['id'] for r in previous.expected_challenge_records()}),('challenge',{r['id'] for r in expected_challenge_records()})]:
        require(set(metrics[group])==ids,'Scored case inventory differs')
    updates,schedule=training_targets(audit,run,m);synthetic=synthetic_training(audit,run,m,sf,cf);checkpoints=checkpoint_identity(audit,run,m,updates)
    members=grouped_natural_and_prior(audit,run,m)
    cells,natural=base.natural(audit,run,m,metrics['natural'],METHODS)
    items,challenge=analytic_items(audit,run,m,cf);analytic_receipt=analytic(audit,run,m,metrics,items)
    checks=natural_decision(audit,cells,metrics['decision'])
    promising=all(checks.values()) and all(analytic_receipt['gates'][k] for k in ('stationary_pass','harmonic_change_pass','consumed_challenge_pass','consumed_subharmonic_pass','challenge_pass'))
    audit.equal(promising,metrics['decision']['promising_for_next_study'],'Combined decision')
    for path in [Path(__file__),Path(base.__file__),Path(historical.__file__),Path(previous.__file__),Path(__file__).with_name('metrics.py'),Path(__file__).with_name('pilot.py'),Path(__file__).with_name('model.py')]:audit.bind(path)
    audit.close()
    receipt={'schema':'phenometrix-augmentation-independent-audit.v1','status':'passed','audited_at':datetime.now(timezone.utc).isoformat(),
        'run_path':str(run.relative_to(REPO)),'run_manifest_sha256':base.sha(run/'manifest.json'),'metrics_sha256':base.sha(run/'metrics.json'),
        'source_freeze_sha256':base.sha(run/'source-freeze.json'),'checkpoint_freeze_sha256':base.sha(run/'checkpoint-freeze.json'),'prediction_freeze_sha256':base.sha(run/'prediction-freeze.json'),
        'audit_source_sha256':base.sha(Path(__file__)),'bound_artifacts':len(audit.files),'numeric_comparisons':audit.comparisons,
        'maximum_numeric_difference':audit.max_rate_delta,'checkpoints':checkpoints,'training_targets':schedule,'synthetic_training':synthetic,'natural':natural|members,
        'analytic':analytic_receipt,'challenge_identity':challenge,'natural_checks':checks,'natural_pass':all(checks.values()),
        'promising_for_next_study':promising,'opened_input_source_sha256':audit.files,
        'limitations':['No fits, new challenge synthesis, reserved/test access, or promotion claim.',
            'Natural scores use object-based metrics.score, not the experiment array scorer; paired synthetic labels and schedules use separate NumPy arithmetic.',
            'Checkpoint bytes/shapes and declared initialization equality are checked; a separate inference audit reproduces selector outputs/initialization.',
            'Original analytic voiced probabilities are independently reconstructed with frozen NumPy weights; 5e-6 cross-backend tolerance and exact threshold decisions are required.',
            'Epoch loss bookkeeping is checked; no optimization replay is performed. Epoch contributions average minibatch objectives weighted by covered rows.',
            'Chronology uses source inspection and frozen manifests/timestamps, not an operating-system access log.',
            'PCM identity, extent and finiteness are checked without regenerating waveforms or TypeScript features.']}
    output.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'passed','comparisons':audit.comparisons,'natural_pass':receipt['natural_pass'],'promising_for_next_study':promising,'output':str(output)}))
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit_run(a.run.resolve(),a.output.resolve())
