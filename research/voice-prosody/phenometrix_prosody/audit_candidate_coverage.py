"""Independent audit of one no-fit, paired candidate-retention intervention.

No optimization, waveform generation, policy selection, or reserved content.
Coverage arithmetic and score/gate reconstruction are separate from the runner.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from . import audit_selector_experiment as base
from . import audit_harmonic_experiment as historical
from . import audit_subharmonic_experiment as subharmonic
from . import audit_augmentation_experiment as augmentation
from . import audit_harmonic_inference as equations

REPO,ROOT,RUNS=base.REPO,base.ROOT,base.RUNS
SEEDS=(20261022,20261023,20261024)
ARMS=('original','revised')
SELECTORS=[f'{arm}__seed{seed}' for arm in ARMS for seed in SEEDS]
METHODS=['original_mlp','original_hybrid',*SELECTORS]
CANDIDATE=[f'revised__seed{seed}' for seed in SEEDS]
CHECKPOINT_SHA={
 'balance-aug__seed20261022':'11fab2c11b86f21c3f2668e451f29bb013709d2d8a24392405fffb8cce2f1b1e',
 'balance-aug__seed20261023':'e16f48f2e39df5da707a0df362e694da8b021a92116eb9fd9c8aa78efcbfd525',
 'balance-aug__seed20261024':'cd0b613f5a26fbff12aaa0f7f1fed28b8cbb3fdbc174582b5a2f8cab433dd265',
}
PRIOR=RUNS/'harmonic-augmentation-v1'
PRIOR_MANIFEST_SHA='d5513b83672f14a711f23a36e5258d348ac9b01a062c5945faaeaf5730d69426'
CHALLENGE_SPEC_SHA='48f8fff5be20297941316ece70fe994c9533148b11596dcfd9ea7607ce024c33'
require=base.require
load_npz=historical.load_npz


def covered_by_candidates(hz,mask,reference,eligible_voiced):
    """All unavailable candidates remain absent; the tolerance stays 50 cents."""
    hz=np.asarray(hz,dtype=np.float64);mask=np.asarray(mask,dtype=bool)
    ref=np.asarray(reference,dtype=np.float64);include=np.asarray(eligible_voiced,dtype=bool)
    require(hz.ndim==2 and mask.shape==hz.shape and ref.shape==include.shape==hz.shape[:1],
            'Coverage dimensions differ')
    require(np.isfinite(hz[mask]).all() and (hz[mask]>0).all(),'Invalid supported candidate')
    require(np.isfinite(ref[include]).all() and (ref[include]>0).all(),'Invalid included reference')
    safe_ref=np.where(include,ref,1.)
    safe_hz=np.where(mask,hz,1.)
    return include & np.any(mask & (np.abs(1200*np.log2(safe_hz/safe_ref[:,None]))<=50),axis=1)


def coverage_counts(eligible_voiced,raw,grouped):
    """Grouping loss counts actual lost rows, never losses minus gains."""
    include=np.asarray(eligible_voiced,dtype=bool)
    raw=np.asarray(raw,dtype=bool);grouped=np.asarray(grouped,dtype=bool)
    require(raw.shape==grouped.shape==include.shape,'Coverage flag dimensions differ')
    require(not np.any((raw|grouped)&~include),'Coverage outside denominator')
    require(not np.any(grouped&~raw),'Original-member grouping gained coverage')
    return {'voiced':int(include.sum()),'raw':int(raw.sum()),'grouped':int(grouped.sum()),
            'lost':int((raw&~grouped).sum()),'gained':0}


def policy_change_counts(eligible_voiced,old,new):
    include=np.asarray(eligible_voiced,dtype=bool);old=np.asarray(old,dtype=bool);new=np.asarray(new,dtype=bool)
    require(include.shape==old.shape==new.shape,'Paired coverage dimensions differ')
    require(not np.any((old|new)&~include),'Paired coverage outside denominator')
    return {'voiced':int(include.sum()),'old_covered':int(old.sum()),'new_covered':int(new.sum()),
            'retained':int((old&new).sum()),'lost':int((old&~new).sum()),
            'gained':int((new&~old).sum()),'neither':int((include&~old&~new).sum())}


def fixed_voicing_oracle(valid,voiced,probability,covered):
    """Reference chooses pitch only; the original binary voicing stays fixed."""
    valid=np.asarray(valid,dtype=bool);voiced=np.asarray(voiced,dtype=bool)
    p=np.asarray(probability,dtype=np.float64);covered=np.asarray(covered,dtype=bool)
    require(valid.shape==voiced.shape==p.shape==covered.shape,'Oracle dimensions differ')
    require(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Original probability invalid')
    require(not np.any(covered&~(valid&voiced)),'Oracle coverage outside voiced denominator')
    gate=p>=.5
    correct=(valid&~voiced&~gate)|(valid&voiced&gate&covered)
    eligible=int(valid.sum());require(eligible>0,'Undefined oracle denominator')
    return {'eligible_valid_frames':eligible,'correct_frames':int(correct.sum()),
            'joint_error':1-float(correct.sum())/eligible,
            'reference_voiced_frames':int((valid&voiced).sum()),
            'shared_false_unvoiced_frames':int((valid&voiced&~gate).sum()),
            'shared_false_positive_frames':int((valid&~voiced&gate).sum()),
            'oracle_correct_voiced_frames':int((valid&voiced&gate&covered).sum())}


def no_reference_inputs(input_paths,reference_paths):
    """Role-qualified REF identities, since legitimate MICs share split folders."""
    require(set(input_paths).isdisjoint(reference_paths),'Reference content hashed before prediction freeze')


def paired_feature_identity(original,revised):
    """The policy may replace candidates/count only; time and globals stay exact."""
    allowed={'features','candidate_hz','candidate_mask','reference_sample','feature_names','sample_rate_hz','pcm_sample_count'}
    require(set(original)==set(revised)==allowed,'Acquisition feature field inventory differs')
    for key in ('reference_sample','feature_names','sample_rate_hz','pcm_sample_count'):
        require(np.array_equal(original[key],revised[key]),'Paired feature identity differs: '+key)
    require(original['features'].shape==revised['features'].shape,'Paired feature extent differs')
    keep=[48,49,50,51,52,53,54,56]
    require(np.array_equal(original['features'][:,keep],revised['features'][:,keep]),
            'Candidate intervention changed non-candidate acquisition evidence')
    return True


def compare_nested(audit,actual,expected,label):
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and set(actual)==set(expected),label+': field inventory differs')
        for key in expected:compare_nested(audit,actual[key],expected[key],label+'/'+str(key))
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),label+': list extent differs')
        for i,(a,e) in enumerate(zip(actual,expected)):compare_nested(audit,a,e,label+'/'+str(i))
    elif isinstance(expected,str):require(actual==expected,label+': string differs')
    else:audit.equal(actual,expected,label)


def step_summary(scenario,target,end,state,hz):
    return {**historical.independent_step(scenario,target,end,state,hz),
            'criterion':'first of 3 consecutive final-frequency outputs within50c; acquisition end>=physical step; no negative-lag clamp'}


def validate_replacement(original,revised,provenance,pcm_sha):
    """Rebuild source-channel replacement from retained peak evidence, not truth."""
    paired_feature_identity(original,revised)
    for f in (original,revised):equations.validate_feature(f)
    n=len(original['features'])
    keys={'old_full40_hz','old_full40_confidence','new_full40_hz','new_full40_confidence',
          'full40_changed','old_frequency_evicted','new_frequency_added','old_candidate_count',
          'new_candidate_count','candidate_policy','pcm_sha256','worker_rss_bytes'}
    require(set(provenance)==keys,'Replacement provenance fields differ')
    require(str(provenance['candidate_policy'])=='shortest-admitted-full40-v1' and str(provenance['pcm_sha256'])==pcm_sha,
            'Replacement provenance policy/PCM differs')
    require(0<int(provenance['worker_rss_bytes'])<=2*1024**3,'Worker memory evidence differs')
    for k in keys-{'candidate_policy','pcm_sha256','worker_rss_bytes'}:require(provenance[k].shape==(n,),'Provenance extent differs')
    oldblocks=original['features'][:,:48].reshape(n,6,8)
    newblocks=revised['features'][:,:48].reshape(n,6,8)
    changed=evicted=added=0
    for i in range(n):
        old_h,new_h=provenance['old_full40_hz'][i],provenance['new_full40_hz'][i]
        old_c,new_c=provenance['old_full40_confidence'][i],provenance['new_full40_confidence'][i]
        require(np.isfinite(old_h)==np.isfinite(new_h),'Policy changed peak availability')
        entries={float(h):block.copy() for h,block,mask in zip(original['candidate_hz'][i],oldblocks[i],original['candidate_mask'][i]) if mask}
        old_peak=[h for h,block in entries.items() if block[5]>0]
        if np.isfinite(old_h):
            require(old_peak==[old_h] and entries[old_h][5]==np.float32(old_c),'Original full40 evidence differs')
            require(50<=new_h<=700 and .35<=new_c<=1 and .35<=old_c<=1,'Peak evidence bounds differ')
            require(new_h>=old_h,'Shortest lag policy returned a longer period')
            entries[old_h][5]=0
            if not entries[old_h][2:7].any():del entries[old_h]
            if new_h not in entries:
                entries[new_h]=np.array([1,np.log2(new_h/50),0,0,0,0,0,0],np.float32)
            entries[new_h][5]=np.float32(new_c)
        else:require(not old_peak and old_c==new_c==0,'Absent peak carries evidence')
        require(len(entries)<=6,'Candidate cap exceeded')
        hz=np.full(6,np.nan);mask=np.zeros(6,bool);blocks=np.zeros((6,8),np.float32)
        for j,(h,b) in enumerate(sorted(entries.items())):
            hz[j]=h;mask[j]=True;b[7]=b[2:7].max();blocks[j]=b
        require(np.array_equal(hz,revised['candidate_hz'][i],equal_nan=True) and np.array_equal(mask,revised['candidate_mask'][i]),
                'Revised pool is not the declared source replacement')
        columns=[0,2,3,4,5,6,7]
        require(np.array_equal(blocks[:,columns],newblocks[i][:,columns]) and np.allclose(blocks[:,1],newblocks[i,:,1],rtol=0,atol=1e-6),
                'Revised feature evidence differs from replacement')
        require(revised['features'][i,55]==np.float32(mask.sum()/6),'Revised count feature differs')
        flags=(bool(np.isfinite(old_h) and (old_h!=new_h or old_c!=new_c)),
               bool(np.isfinite(old_h) and old_h not in entries),
               bool(np.isfinite(new_h) and not np.any(original['candidate_hz'][i,original['candidate_mask'][i]]==new_h)))
        for key,value in zip(('full40_changed','old_frequency_evicted','new_frequency_added'),flags):
            require(bool(provenance[key][i])==value,'Replacement flag differs: '+key)
        require(provenance['old_candidate_count'][i]==original['candidate_mask'][i].sum() and provenance['new_candidate_count'][i]==mask.sum(),
                'Candidate count provenance differs')
        changed+=flags[0];evicted+=flags[1];added+=flags[2]
    return {'frames':n,'full40_changed':changed,'old_frequency_evicted':evicted,'new_frequency_added':added}


def inference_check(models,feature,probability,arrays,grouped):
    from .selector_model import prepare_selector_batch
    batch=prepare_selector_batch(feature['features'],feature['candidate_hz'],feature['candidate_mask'],representation='shared-candidates')
    require(np.array_equal(batch.candidate_hz.numpy(),grouped,equal_nan=True),'Independent grouping representative differs')
    comparisons=0
    for seed,weights in models.items():
        states=[];frequencies=[]
        with torch.inference_mode():
            for start in range(0,len(probability),4096):
                stop=start+4096
                logits=equations.independent_logits(weights,'relative',batch.features[start:stop],batch.candidate_mask[start:stop])
                state,hz=equations.emit(logits,grouped[start:stop],np.isfinite(grouped[start:stop]),probability[start:stop])
                states.append(state);frequencies.append(hz)
        got_state,got_hz=np.concatenate(states),np.concatenate(frequencies)
        state,hz=arrays[seed]
        require(np.array_equal(got_state,state) and np.array_equal(got_hz,hz,equal_nan=True),'Independent frozen-checkpoint output differs')
        require(np.array_equal(state==0,probability<.5),'Frozen voicing threshold differs')
        require(np.array_equal(state,np.where(probability<.5,0,np.where(np.isfinite(grouped).any(-1),1,2))),'Policy-specific availability differs')
        comparisons+=len(state)
    return comparisons


def read_checkpoints(audit,manifest):
    prior=audit.json(PRIOR/'manifest.json',PRIOR_MANIFEST_SHA)
    require(prior['status']=='complete','Prior experiment not complete')
    expected={name:{'path':str((PRIOR/'checkpoints'/(name+'.pt')).relative_to(REPO)),'sha256':sha} for name,sha in CHECKPOINT_SHA.items()}
    require(manifest['checkpoints']==expected,'Checkpoint identities differ from the frozen augmentation study')
    models={}
    for name,row in expected.items():
        require(prior['fits'][name]['weights_sha256']==row['sha256'],'Prior checkpoint manifest identity differs')
        saved=torch.load(audit.bind(REPO/row['path'],row['sha256']),map_location='cpu',weights_only=True)
        require(saved['arm']=='balance-aug' and saved['context_mode']=='relative' and saved['config']=={
            'representation':'shared-candidates','loss':'acceptable-set','hidden_size':32},'Frozen architecture differs')
        weights=saved['state_dict']
        require(all(bool(torch.isfinite(t).all()) for t in weights.values()) and bool(weights['scaling_fitted']),'Invalid checkpoint tensors')
        require(bool((weights['candidate_scale']>0).all()) and bool((weights['global_scale']>0).all()),'Invalid frozen scaling')
        require(sum(t.numel() for k,t in weights.items() if k.endswith(('.weight','.bias')))==3009,'Frozen parameter count differs')
        models[int(name.split('seed')[1])]=weights
    return models,prior


def finish_coverage(cells):
    rows=[]
    for name,c in sorted(cells.items()):
        require(c['voiced']>0,'Undefined coverage cell')
        s,condition=name.split('/')
        rows.append({'speaker':s,'condition':condition,**c,**{label:c[numerator]/c['voiced'] for label,numerator in (
            ('raw_coverage','raw'),('grouped_coverage','grouped'),('lost_fraction','lost'),('gained_fraction','gained'))}})
    result={'cells':rows,**{k:float(np.mean([r[k] for r in rows])) for k in ('raw_coverage','grouped_coverage','lost_fraction','gained_fraction')}}
    result['pass']=result['lost_fraction']<=.005
    return result


def finish_transition(cells):
    result={}
    for name,c in sorted(cells.items()):
        result[name]={'voiced':c['voiced']}
        for rep in ('fixed-slots','shared-candidates'):
            values=c[rep];result[name][rep]={**values,**{k+'_fraction':v/c['voiced'] for k,v in values.items()}}
    return {'cells':result,'macro':{rep:{k+'_fraction':float(np.mean([v[rep][k+'_fraction'] for v in result.values()])) for k in ('lost','gained','both','neither')}
                                    for rep in ('fixed-slots','shared-candidates')}}


def ref_arrays(refs,targets):
    by_sample={r.sample:r for r in refs};hz=np.full(len(targets),np.nan);valid=np.zeros(len(targets),bool);voiced=np.zeros(len(targets),bool)
    for i,t in enumerate(targets):
        r=by_sample.get(int(t))
        if r is not None:
            valid[i]=r.valid and r.eligible;voiced[i]=r.voiced
            if r.f0_hz is not None:hz[i]=r.f0_hz
    return hz,valid,voiced


def natural_and_coverage(audit,run,manifest,metrics,models,prior):
    predictions={name:load_npz(audit,run/'predictions'/(name+'.npz'),manifest['prediction_sha256']['predictions/'+name+'.npz']) for name in METHODS}
    for values in predictions.values():require(set(values)=={'state','hz'},'Natural prediction fields differ')
    for seed in SEEDS:
        name=f'balance-aug__seed{seed}';p='predictions/'+name+'.npz'
        old=load_npz(audit,PRIOR/p,prior['prediction_sha256'][p])
        require(all(np.array_equal(old[k],predictions[f'original__seed{seed}'][k],equal_nan=True) for k in ('state','hz')),'Original-policy natural control changed')
    pp=run/'probabilities-development.npz';p=load_npz(audit,pp,manifest['input_sha256'][str(pp.relative_to(REPO))])
    require(set(p)=={'original_probability'},'DEV probability fields differ');probability=p['original_probability']
    require(probability.ndim==1 and np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all(),'DEV original probability invalid')
    oldfreeze=RUNS/'hybrid-natural-v1/prediction-freeze.json'
    baseline=audit.json(oldfreeze,manifest['input_sha256'][str(oldfreeze.relative_to(REPO))]);baserows={r['recording_id']:r for r in baseline['recordings']}
    totals={m:defaultdict(lambda:{k:0 for k in base.COUNTS}) for m in METHODS}
    all_cov={};headcounts={arm:{rep:defaultdict(lambda:{'eligible':0,'oracle_correct':0}) for rep in ('fixed-slots','shared-candidates')} for arm in ARMS}
    replacement_totals=defaultdict(int);inference_rows=0;exclusions={}
    for split in ('train','development'):
        folder,oldcache,metadata=base.load_cache(audit,split)
        cachepath=run/split/'manifest.json';cache=audit.json(cachepath,manifest['input_sha256'][str(cachepath.relative_to(REPO))])
        require(cache['schema']=='phenometrix-candidate-policy-cache.v1' and cache['status']=='complete' and cache['split']==split and cache['reference_content_read'] is False,'Revised natural cache state differs')
        require(cache['source_sha256']==manifest['source_sha256'] and cache['candidate_policy']=='shortest-admitted-full40-v1','Revised cache policy/source differs')
        require(cache['original_manifest_path']==str((folder/'manifest.json').relative_to(REPO)) and cache['original_manifest_sha256']==base.sha(folder/'manifest.json'),'Original cache binding differs')
        require(len(cache['records'])==len(oldcache['recordings'])==({'train':960,'development':320}[split]),'Natural cache inventory differs')
        refs_cache={};covcells={arm:defaultdict(lambda:{k:0 for k in ('voiced','raw','grouped','lost','gained')}) for arm in ARMS}
        changes=defaultdict(lambda:{'voiced':0,**{rep:{k:0 for k in ('lost','gained','both','neither')} for rep in ('fixed-slots','shared-candidates')}})
        offset=0;exclusions[split]={}
        for record,row in zip(oldcache['recordings'],cache['records']):
            for key in ('recording_id','source_recording_id','speaker','condition','sample_count','transformed_pcm_sha256'):
                require(record[key]==row[key],'Paired source/condition identity differs: '+key)
            original=base.load_feature(audit,folder,record)
            require(row['original_feature_path']==str((folder/'features'/(record['recording_id']+'.npz')).relative_to(REPO)) and row['original_feature_sha256']==record['feature_sha256'],'Original feature binding differs')
            revised=load_npz(audit,REPO/row['feature_path'],row['feature_sha256']);provenance=load_npz(audit,REPO/row['provenance_path'],row['provenance_sha256'])
            check=validate_replacement(original,revised,provenance,record['transformed_pcm_sha256'])
            for k,v in check.items():replacement_totals[k]+=v
            features={'original':original,'revised':revised};grouped={arm:base.representative_hz(f) for arm,f in features.items()}
            targets=original['reference_sample']-960;stop=offset+len(targets);sl=slice(offset,stop)
            require(row['start']==offset and row['stop']==stop,'Paired cache row offsets differ')
            source=record['source_recording_id']
            if source not in refs_cache:refs_cache[source]=base.references(audit,metadata[source],record,split)
            refs=refs_cache[source];truth,valid,voiced=ref_arrays(refs,targets);include=valid&voiced
            excludes={'invalid_reference':sum(not r.valid for r in refs),'boundary_reference':sum(r.valid and not r.eligible for r in refs),
                      'predictions_outside_reference_extent':len(set(targets.tolist())-{r.sample for r in refs})}
            exclusions[split][record['recording_id']]=excludes
            cell=record['speaker']+'/'+record['condition'];flags={}
            for arm,f in features.items():
                raw=covered_by_candidates(f['candidate_hz'],f['candidate_mask'],truth,include)
                g=covered_by_candidates(grouped[arm],np.isfinite(grouped[arm]),truth,include)
                flags[arm]={'fixed-slots':raw,'shared-candidates':g}
                for k,v in coverage_counts(include,raw,g).items():covcells[arm][cell][k]+=v
                if split=='development':
                    for rep,covered in flags[arm].items():
                        oracle=fixed_voicing_oracle(valid,voiced,probability[sl],covered)
                        headcounts[arm][rep][cell]['eligible']+=oracle['eligible_valid_frames']
                        headcounts[arm][rep][cell]['oracle_correct']+=oracle['correct_frames']
            changes[cell]['voiced']+=int(include.sum())
            for rep in ('fixed-slots','shared-candidates'):
                c=policy_change_counts(include,flags['original'][rep],flags['revised'][rep])
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
                for arm in ARMS:
                    arrays={seed:(predictions[f'{arm}__seed{seed}']['state'][sl],predictions[f'{arm}__seed{seed}']['hz'][sl]) for seed in SEEDS}
                    inference_rows+=inference_check(models,features[arm],probability[sl],arrays,grouped[arm])
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
        all_cov[split]={arm:finish_coverage(covcells[arm]) for arm in ARMS}
        all_cov[split]['policy_change']=finish_transition(changes)
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
    for arm in ARMS:
        headroom[arm]={}
        for rep,counts in headcounts[arm].items():
            for cell,c in counts.items():
                require(c['eligible']==cells['original_mlp'][cell]['eligible_valid_frames'],'Oracle denominator differs')
                c['mlp_correct']=cells['original_mlp'][cell]['correct_frames'];c['gain']=(c['oracle_correct']-c['mlp_correct'])/c['eligible']
            gain=float(np.mean([v['gain'] for v in counts.values()]));headroom[arm][rep]={'cells':dict(counts),'mean_gain':gain,'pass':gain>=.01}
    compare_nested(audit,headroom,metrics['headroom'],'headroom')
    compare_nested(audit,exclusions,metrics['exclusions'],'natural exclusions')
    return cells,all_cov,headroom,{'methods':8,'development_condition_records':320,'train_condition_records':960,
        'development_rows':len(probability),'checkpoint_frame_pairs':inference_rows,'replacement_reconstruction':dict(replacement_totals)}


def fresh_records():
    return [{'id':f'coverage-balance-f{i}-p{j}','frequency_hz':f,'phase_radians':list(phases),'gain':.055,
             'sample_rate_hz':48000,'sample_count':96000,'change_sample':39840,'before':[1.,.5,.25],'after':[0.,1.,.5]}
            for i,f in enumerate((97.1,213.7,347.3,529.1,627.7,671.3))
            for j,phases in enumerate(((.29,-.67,1.31),(1.89,.47,-1.43)))]


def expected_analytic(audit,run,manifest):
    original=audit.json(RUNS/'delayed-known-truth/prediction-freeze.json')
    support=audit.json(RUNS/'support-diagnostic-v1/manifest.json',original['original_support_manifest_sha256'])
    specs={r['id']:r for r in support['scenarios']}
    for hz in (100,200,300):
        name=f'observation-change-{hz}';specs[name]={'id':name,'kind':'observation-change','durationSeconds':1.6,
            'initialHz':hz,'finalHz':hz,'changeStartSeconds':.6,'changeEndSeconds':.6,'slopeHzPerSecond':0}
    require(set(specs)==base.SYNTHETIC_IDS,'Original regression configuration inventory differs')
    records={('regression',r['id']):{'scenario':specs[r['id']],
        'original_feature_path':str((RUNS/'delayed-known-truth/features'/(r['id']+'.npz')).relative_to(REPO)),
        'original_feature_sha256':r['feature_sha256'],'pcm_sha256':r['pcm_sha256']} for r in original['recordings']}
    domains=[('consumed_harmonic',RUNS/'harmonic-context-v1/challenge',historical.expected_challenge_records()),
             ('consumed_subharmonic',RUNS/'subharmonic-objective-v1/challenge',subharmonic.expected_challenge_records()),
             ('consumed_augmentation',PRIOR/'challenge',augmentation.expected_challenge_records()),
             ('challenge',run/'challenge',fresh_records())]
    for group,folder,expected in domains:
        cm=audit.json(folder/'manifest.json');spec=audit.json(folder/'spec.json')
        require(cm['status']=='complete' and cm['reference_content_read'] is False and spec['records']==expected,'Challenge inventory/configuration differs')
        digest=hashlib.sha256(historical.canonical(spec)).hexdigest();require(digest==cm['spec_sha256'],'Challenge specification digest differs')
        require({r['id'] for r in cm['records']}=={r['id'] for r in expected},'Challenge cache case inventory differs')
        if group=='challenge':
            require(cm['schema']=='phenometrix-candidate-coverage-challenge-cache-v1' and digest==manifest['challenge_spec_sha256']==CHALLENGE_SPEC_SHA,'Fresh challenge contract differs')
            require(cm['freeze_path']==str((run/'source-checkpoint-freeze.json').relative_to(REPO)) and cm['freeze_sha256']==base.sha(run/'source-checkpoint-freeze.json'),'Fresh challenge source/checkpoint binding differs')
            require(historical.dt(manifest['frozen_at'])<=historical.dt(cm['started_at'])<=historical.dt(cm['completed_at'])<=historical.dt(manifest['predictions_frozen_at']),'Fresh generation chronology differs')
        else:require(historical.dt(cm['completed_at'])<historical.dt(manifest['frozen_at']),'Consumed challenge is not historical')
        for path,sha in cm['source_sha256'].items():audit.bind(REPO/path,sha)
        params={r['id']:r for r in expected}
        for row in cm['records']:
            p=params[row['id']]
            require(row['parameters_sha256']==hashlib.sha256(historical.canonical(p)).hexdigest(),'Challenge parameters hash differs')
            audit.bind(folder/row['pcm_path'],row['pcm_sha256'])
            records[(group,row['id'])]={'scenario':{'id':row['id'],'kind':'observation-change','durationSeconds':2.,
                 'initialHz':p['frequency_hz'],'finalHz':p['frequency_hz'],'changeStartSeconds':.83,'changeEndSeconds':.83,'slopeHzPerSecond':0},
                 'original_feature_path':str((folder/row['feature_path']).relative_to(REPO)),
                 'original_feature_sha256':row['feature_sha256'],'pcm_sha256':row['pcm_sha256']}
    require(len(records)==62,'Expected analytic inventory differs')
    return records


def analytic_coverage_record(features,grouped,truth,eligible,region,probability,arrays):
    flags={arm:{'fixed-slots':covered_by_candidates(f['candidate_hz'],f['candidate_mask'],truth,eligible),
                'shared-candidates':covered_by_candidates(grouped[arm],np.isfinite(grouped[arm]),truth,eligible)}
           for arm,f in features.items()}
    report={}
    for name,include in {'all':eligible,**{r:eligible&(region==r) for r in sorted(set(region[eligible]))}}.items():
        item={'frames':int(include.sum()),'policies':{},'policy_change':{},
              'original_voiced_gate_positive':int((include&(probability>=.5)).sum()),'methods':{}}
        for arm in ARMS:
            a,b=flags[arm]['fixed-slots'],flags[arm]['shared-candidates']
            item['policies'][arm]={'raw_covered':int((include&a).sum()),'grouped_covered':int((include&b).sum()),
                'grouping_lost':int((include&a&~b).sum()),'candidate_unavailable':int((include&~np.isfinite(grouped[arm]).any(-1)).sum())}
        for rep in ('fixed-slots','shared-candidates'):
            a,b=flags['original'][rep],flags['revised'][rep]
            item['policy_change'][rep]={'lost':int((include&a&~b).sum()),'gained':int((include&~a&b).sum()),
                                       'both':int((include&a&b).sum()),'neither':int((include&~a&~b).sum())}
        for m,(state,_) in arrays.items():item['methods'][m]={n:int((include&(state==i)).sum()) for i,n in enumerate(('unvoiced','voiced','unavailable'))}
        report[name]=item
    return report


def analytic(audit,run,manifest,metrics,models,prior):
    path=run/'analytic-inventory.json';inventory=audit.json(path,manifest['input_sha256'][str(path.relative_to(REPO))])
    counts={'regression':14,'consumed_harmonic':12,'consumed_subharmonic':12,'consumed_augmentation':12,'challenge':12}
    require({g:len(rows) for g,rows in inventory.items()}==counts,'Analytic inventory counts differ')
    expected=expected_analytic(audit,run,manifest)
    gates={g+'_by_seed_case':{m:{} for m in CANDIDATE} for g in ('stationary','harmonic_change',*list(counts)[1:])}
    original=torch.load(audit.bind(RUNS/'delayed-mlp-seed20261022/weights.pt',base.WEIGHTS),map_location='cpu',weights_only=True)['state_dict']
    maximum_p_difference=0.;inference_rows=control_rows=0;cases={};replacement=defaultdict(int)
    for group,rows in inventory.items():
        require(set(metrics[group])=={r['id'] for r in rows},'Scored analytic case inventory differs')
        for row in rows:
            name=row['id'];want=expected[(group,name)];scenario=row['scenario']
            for k in ('kind','durationSeconds','initialHz','finalHz','changeStartSeconds','changeEndSeconds','slopeHzPerSecond'):
                require(scenario[k]==want['scenario'][k],'Analytic physical scenario differs: '+k)
            for k in ('original_feature_path','original_feature_sha256','pcm_sha256'):require(row[k]==want[k],'Analytic artifact identity differs: '+k)
            features={'original':load_npz(audit,REPO/row['original_feature_path'],row['original_feature_sha256']),
                      'revised':load_npz(audit,REPO/row['revised_feature_path'],row['revised_feature_sha256'])}
            provenance=load_npz(audit,REPO/row['provenance_path'],row['provenance_sha256'])
            for k,v in validate_replacement(features['original'],features['revised'],provenance,row['pcm_sha256']).items():replacement[k]+=v
            n=76800 if group=='regression' else 96000
            end,target,eligible=historical.validate_feature(features['original'],n)
            require(row['sample_count']==n,'Analytic sample extent differs')
            z=load_npz(audit,run/row['prediction_path'],row['prediction_sha256'])
            require(row['prediction_sha256']==manifest['prediction_sha256'][row['prediction_path']],'Analytic prediction binding differs')
            require(set(z)=={'original_probability'}|{m+s for m in SELECTORS for s in ('__state','__hz')},'Analytic prediction fields differ')
            probability=z['original_probability']
            with torch.inference_mode():actual_p=equations.original_probability(original,features['original']['features'])
            require(probability.shape==actual_p.shape and np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all(),'Analytic probability invalid')
            delta=float(np.max(np.abs(actual_p-probability)));maximum_p_difference=max(maximum_p_difference,delta)
            require(delta<=1e-6 and np.array_equal(actual_p<.5,probability<.5),'Original-feature analytic voicing differs')
            arrays={m:(z[m+'__state'],z[m+'__hz']) for m in SELECTORS}
            grouped={arm:base.representative_hz(f) for arm,f in features.items()}
            for arm in ARMS:inference_rows+=inference_check(models,features[arm],probability,{s:arrays[f'{arm}__seed{s}'] for s in SEEDS},grouped[arm])
            prior_group={'regression':'regression','consumed_harmonic':'consumed_challenge','consumed_subharmonic':'consumed_subharmonic','consumed_augmentation':'challenge'}.get(group)
            if prior_group:
                pp='predictions/'+prior_group+'-'+name+'.npz';old=load_npz(audit,PRIOR/pp,prior['prediction_sha256'][pp])
                require(np.array_equal(old['original_probability'],probability),'Consumed original probability changed')
                for seed in SEEDS:
                    state,hz=arrays[f'original__seed{seed}'];require(np.array_equal(state,old[f'balance-aug__seed{seed}__state']) and np.array_equal(hz,old[f'balance-aug__seed{seed}__hz'],equal_nan=True),'Consumed original control changed')
                    control_rows+=len(state)
            truth=historical.curve(scenario,target/48000);region=historical.regions(scenario,end)
            coverage=analytic_coverage_record(features,grouped,truth,eligible,region,probability,arrays)
            report=metrics[group][name];compare_nested(audit,coverage,report['candidate_coverage'],group+'/'+name+'/coverage')
            require(set(report['methods'])==set(SELECTORS),'Analytic reported method inventory differs')
            case={}
            for method,(state,hz) in arrays.items():
                base.prediction_objects(target,state,hz)
                for label,include in {'all':eligible,**{r:eligible&(region==r) for r in sorted(set(region[eligible]))}}.items():
                    got=historical.analytic_summary(truth[include],state[include],hz[include],region[include],scenario['slopeHzPerSecond'])
                    expected_summary=report['methods'][method]['all'] if label=='all' else report['methods'][method]['regions'][label]
                    compare_nested(audit,got,expected_summary,group+'/'+name+'/'+method+'/'+label)
                if scenario['kind']=='step':
                    got=step_summary(scenario,target[eligible],end[eligible],state[eligible],hz[eligible])
                    compare_nested(audit,got,report['methods'][method]['step_response'],group+'/'+name+'/'+method+'/step')
                else:require(report['methods'][method]['step_response'] is None,'Unexpected analytic step result')
                if method in CANDIDATE:
                    if group!='regression':gate=group+'_by_seed_case';include=eligible&(region=='after-observation-change')
                    elif scenario['kind']=='constant':gate='stationary_by_seed_case';include=eligible
                    elif scenario['kind']=='observation-change':gate='harmonic_change_by_seed_case';include=eligible&(region=='after-observation-change')
                    else:continue
                    got=historical.analytic_summary(truth[include],state[include],hz[include],region[include],0)
                    passed=got['success_within_50c_all_frames']>=.95;gates[gate][method][name]=passed
                    case[method]={'frames':got['frames'],'success':got['success_within_50c_all_frames'],'pass':passed}
            cases[group+'/'+name]=case
    for group,count in {'stationary':3,'harmonic_change':3,**{g:12 for g in list(counts)[1:]}}.items():
        values=gates[group+'_by_seed_case'];require(all(len(v)==count for v in values.values()),'Incomplete analytic gate inventory')
        gates[group+'_pass']=all(all(v.values()) for v in values.values())
    compare_nested(audit,gates,metrics['decision']['analytic'],'analytic gates')
    return {'cases':62,'selectors_per_case':6,'checkpoint_frame_pairs':inference_rows,'original_control_frame_pairs':control_rows,
            'maximum_original_probability_difference':maximum_p_difference,'replacement_reconstruction':dict(replacement),
            'case_gates':cases,'gates':gates}


def natural_decision(audit,cells,reported):
    mean=lambda m,k:float(np.mean([v[k] for v in cells[m].values()]))
    error=float(np.mean([mean(m,'joint_error') for m in CANDIDATE]));success=float(np.mean([mean(m,'reference_voiced_success_rate') for m in CANDIDATE]))
    gains={m:mean(m,'joint_error')-error for m in ('original_mlp','original_hybrid')}
    excess={c:float(np.mean([v['joint_error'] for m in CANDIDATE for key,v in cells[m].items() if key.endswith('/'+c)]))-
                 float(np.mean([v['joint_error'] for key,v in cells['original_mlp'].items() if key.endswith('/'+c)])) for c in base.CONDITIONS}
    checks={'mean_improves_both_by_half_point':all(v>=.005 for v in gains.values()),
            'each_seed_improves_original_mlp':all(mean(m,'joint_error')<mean('original_mlp','joint_error') for m in CANDIDATE),
            'no_condition_worsens_over_one_point':all(v<=.01 for v in excess.values()),
            'voiced_success_not_lower':success>=mean('original_mlp','reference_voiced_success_rate')}
    result={'predesignated_candidate':'revised','pass':all(checks.values()),'checks':checks,'mean_joint_error':error,
            'mean_voiced_success':success,'joint_error_improvement_over_frozen':gains,'condition_error_excess_over_mlp':excess,
            'improvement_over_paired_original':float(np.mean([mean(f'original__seed{s}','joint_error') for s in SEEDS]))-error}
    compare_nested(audit,result,{k:reported[k] for k in result},'natural decision')
    return result


def audit_run(run,output):
    run=base.inside(RUNS,run);output=base.inside(run,output)
    require(not output.exists(),'Refuses to overwrite audit evidence')
    torch.set_num_threads(1);audit=base.Audit();m=audit.json(run/'manifest.json')
    require(m['schema']=='phenometrix-candidate-coverage-experiment.v1' and m['status']=='complete','Completed candidate-coverage experiment required')
    require(m['arms']==list(ARMS) and m['seeds']==list(SEEDS) and m['candidate_policy']=='shortest-admitted-full40-v1','Declared policy comparison differs')
    require(m['new_fits']==m['optimizer_updates']==0 and m['reserved_content_read'] is False,'No-fit/data boundary differs')
    require(m['elapsed_seconds']<=75*60 and m['peak_observed_python_rss_bytes']<=16*1024**3 and m['peak_observed_node_rss_bytes']<=2*1024**3 and m['output_bytes']<=2*1024**3,'Declared local resource budget exceeded')
    for path,sha in m['source_sha256'].items():audit.bind(REPO/path,sha);audit.bind(run/'source-snapshot'/path,sha)
    for path,sha in m['input_sha256'].items():audit.bind(REPO/path,sha)
    models,prior=read_checkpoints(audit,m)
    require(all(m['source_sha256'].get(p)==h for p,h in prior['source_sha256'].items()),'Historical source closure changed')
    if 'historical_input_sha256' in m:
        expected_historical={str(Path(p).relative_to(REPO)):h for p,h in prior['input_sha256'].items()}
        require(m['historical_input_sha256']==expected_historical,'Historical input inventory changed')
        for p,h in m['historical_input_sha256'].items():audit.bind(REPO/p,h)
    frozen=audit.json(run/'source-checkpoint-freeze.json',m['source_checkpoint_freeze_sha256']);pred=audit.json(run/'prediction-freeze.json')
    require(frozen['schema']=='phenometrix-candidate-coverage-freeze.v1' and frozen['status']=='sources_checkpoints_frozen' and pred['status']=='predictions_frozen','Required phase states differ')
    for k in ('source_sha256','checkpoints','arms','seeds','new_fits','candidate_policy','frozen_at','challenge_spec_sha256'):
        require(frozen[k]==pred[k]==m[k],'Identity changed across freezes: '+k)
    require(pred['prediction_sha256']==m['prediction_sha256'] and pred['train_references_read'] is False and pred['development_references_read'] is False,
            'Predictions/declared reference chronology differ')
    require(m['train_references_read'] is True and m['development_references_read'] is True,'Missing final reference phase')
    reference_paths=set()
    for split in ('train','development'):
        _,_,metadata=base.load_cache(audit,split)
        reference_paths.update(str((ROOT/'data/ptdb-tug'/row['path']).relative_to(REPO)) for row in metadata.values())
    no_reference_inputs(pred['input_sha256'],reference_paths)
    require(all(m['input_sha256'].get(p)==h for p,h in pred['input_sha256'].items()),'Prediction-freeze inputs changed')
    require(historical.dt(m['frozen_at'])<=historical.dt(m['started_at'])<=historical.dt(m['predictions_frozen_at'])<=historical.dt(m['completed_at']),'Phase timestamp order differs')
    source=(run/'source-snapshot/research/voice-prosody/phenometrix_prosody/candidate_coverage_experiment.py').read_text()
    require(source.index("save(output/'source-checkpoint-freeze.json',freeze)")<source.index('cache_natural(natural_data[split]')<source.index('fresh,h=cases.generate_challenge(')<source.index("save(output/'prediction-freeze.json',m)")<source.index('labels={split:old.load_labels(data)'),
            'Source prediction/reference ordering differs')
    for path,sha in m['prediction_sha256'].items():audit.bind(run/path,sha)
    metrics=audit.json(run/'metrics.json',m['metrics_sha256']);require(metrics['schema']=='phenometrix-candidate-coverage-metrics.v1','Metrics schema differs')
    inventory=audit.json(run/'analytic-inventory.json')
    expected_predictions={'predictions/'+name+'.npz' for name in METHODS}|{row['prediction_path'] for rows in inventory.values() for row in rows}
    require(set(m['prediction_sha256'])==expected_predictions and len(expected_predictions)==70,'Prediction inventory differs')
    cells,coverage,headroom,natural=natural_and_coverage(audit,run,m,metrics,models,prior)
    analytic_receipt=analytic(audit,run,m,metrics,models,prior);decision=natural_decision(audit,cells,metrics['decision'])
    prerequisites={'raw_fixed_voicing_oracle_headroom':headroom['revised']['fixed-slots']['pass'],
                   'train_grouping_loss':coverage['train']['revised']['pass'],'development_grouping_loss':coverage['development']['revised']['pass']}
    compare_nested(audit,prerequisites,metrics['decision']['prerequisites'],'prerequisites')
    promising=decision['pass'] and all(prerequisites.values()) and all(v for k,v in analytic_receipt['gates'].items() if k.endswith('_pass'))
    audit.equal(promising,metrics['decision']['promising_for_next_study'],'Combined decision')
    control_pairs=3*natural['development_rows']+analytic_receipt['original_control_frame_pairs']
    compare_nested(audit,{'state_hz_frame_pairs':control_pairs,'state_mismatches':0,'hz_mismatches':0,'all_exact':True},m['control_reproduction'],'Original controls')
    audit_test=ROOT/'tests/test_audit_candidate_coverage.py'
    for path in (Path(__file__),audit_test,Path(base.__file__),Path(historical.__file__),Path(subharmonic.__file__),Path(augmentation.__file__),Path(equations.__file__),Path(__file__).with_name('selector_model.py'),Path(__file__).with_name('metrics.py'),Path(__file__).with_name('pilot.py')):audit.bind(path)
    audit.close()
    receipt={'schema':'phenometrix-candidate-coverage-independent-audit.v1','status':'passed','audited_at':datetime.now(timezone.utc).isoformat(),
        'run_path':str(run.relative_to(REPO)),'run_manifest_sha256':base.sha(run/'manifest.json'),'metrics_sha256':base.sha(run/'metrics.json'),
        'source_checkpoint_freeze_sha256':base.sha(run/'source-checkpoint-freeze.json'),'prediction_freeze_sha256':base.sha(run/'prediction-freeze.json'),
        'audit_source_sha256':base.sha(Path(__file__)),'audit_tests_sha256':base.sha(audit_test),'bound_artifacts':len(audit.files),'numeric_comparisons':audit.comparisons,
        'maximum_numeric_difference':audit.max_rate_delta,'checkpoint_sha256':CHECKPOINT_SHA,'natural':natural,'analytic':analytic_receipt,
        'coverage':coverage,'headroom':headroom,'natural_decision':decision,'prerequisites':prerequisites,'promising_for_next_study':promising,
        'original_control_frame_pairs':control_pairs,'total_checkpoint_frame_pairs':natural['checkpoint_frame_pairs']+analytic_receipt['checkpoint_frame_pairs'],
        'opened_input_source_sha256':audit.files,
        'limitations':['No fitting, policy changes, new waveform generation, reserved/test access, or promotion claim.',
          'Natural reference scoring uses object-based metrics.score rather than the runner array scorer; analytic counts and gates use separately implemented arithmetic.',
          'Candidate replacement and genuine source-channel accounting are reconstructed from recorded peaks. This audit does not recompute the signal autocorrelation curves.',
          'Frozen-checkpoint inference uses independent explicit CPU equations; shared preprocessing is source-bound and emitted group representatives are independently checked.',
          'Fresh original voicing is independently reconstructed from original acquisition features, with 1e-6 probability tolerance and exact threshold decisions.',
          'Coverage/headroom are reference-based descriptive bounds; they do not establish learnability, external generalization, calibration, or causal mechanism.',
          'Chronology follows source inspection and frozen manifest identities/timestamps; no operating-system access-log claim.']}
    output.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'passed','comparisons':audit.comparisons,'checkpoint_frame_pairs':receipt['total_checkpoint_frame_pairs'],'promising_for_next_study':promising,'output':str(output)}))
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();audit_run(args.run.resolve(),args.output.resolve())
