"""Independent postfit scoring audit; no fitting, challenge synthesis or test access.

Uses the historical object scorer, separate analytic arithmetic, and independent
member grouping. Declared ordering is checked, not asserted as an OS access log.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from . import audit_selector_experiment as base
from .model import ModelConfig,numpy_step

REPO,ROOT,RUNS=base.REPO,base.ROOT,base.RUNS
SEEDS=base.SEEDS
ARMS=(('absolute','nearest'),('absolute','acceptable-set'),('relative','nearest'),('relative','acceptable-set'))
NEW=[base.key(a,b,s) for s in SEEDS for a,b in ARMS]
PRIOR=['prior__'+base.key('shared-candidates',b,s) for s in SEEDS for b in ('nearest','acceptable-set')]
SELECTORS=NEW+PRIOR
METHODS=['original_mlp','original_hybrid',*SELECTORS]
CANDIDATE=[base.key('relative','acceptable-set',s) for s in SEEDS]
require=base.require


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def dt(value):return datetime.fromisoformat(value)
def load_npz(audit,path,expected=None):
    with np.load(audit.bind(path,expected),allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}

def expected_challenge_records():
    return [{'id':f'harmonic-balance-f{i}-p{j}','frequency_hz':f,'phase_radians':list(p),
             'sample_rate_hz':48000,'sample_count':96000,'change_sample':39840,'gain':.06,
             'before':[1.,.5,.25],'after':[0.,1.,.5]}
            for i,f in enumerate((87.3,137.9,223.7,337.1,467.3,613.7))
            for j,p in enumerate(((.37,-.41,1.13),(1.19,.23,-.73)))]

def curve(scenario,t):
    if scenario['kind'] in ('constant','observation-change'):return np.full(t.shape,float(scenario['initialHz']))
    if scenario['kind']=='step':return np.where(t<scenario['changeStartSeconds'],scenario['initialHz'],scenario['finalHz'])
    elapsed=np.clip(t-scenario['changeStartSeconds'],0,scenario['changeEndSeconds']-scenario['changeStartSeconds'])
    return scenario['initialHz']+scenario['slopeHzPerSecond']*elapsed

def regions(scenario,ends):
    out=[]
    for end in ends:
        start=(int(end)-1920)/48000;stop=int(end)/48000
        if scenario['kind']=='constant':value='plateau'
        elif scenario['kind']=='observation-change':
            change=scenario['changeStartSeconds']
            value='before-observation-change' if stop<change else 'after-observation-change' if start>=change else 'observation-boundary'
        elif scenario['kind']=='glide' and start>=scenario['changeStartSeconds'] and stop<scenario['changeEndSeconds']:
            value='glide-interior'
        else:value='plateau' if stop<scenario['changeStartSeconds'] or start>=scenario['changeEndSeconds'] else 'transition-boundary'
        out.append(value)
    return np.array(out)

def analytic_summary(truth,state,hz,region,slope):
    n=len(truth);voiced=state==1
    error=1200*np.log2(hz[voiced]/truth[voiced]);absolute=np.abs(error)
    delay=(truth[voiced]-hz[voiced])/slope*1000 if slope else np.array([])
    delay=delay[region[voiced]=='glide-interior'] if slope else delay
    return {'frames':n,'voiced_outputs':int(voiced.sum()),'nonvoiced_or_unavailable':int((~voiced).sum()),
        'pitch_coverage':float(voiced.mean()) if n else None,
        'success_within_50c_all_frames':int((absolute<=50).sum())/n if n else None,
        'median_absolute_cents':float(np.median(absolute)) if len(absolute) else None,
        'p95_absolute_cents':float(np.quantile(absolute,.95,method='higher')) if len(absolute) else None,
        'mean_signed_cents':float(np.mean(error)) if len(error) else None,
        'median_signed_equivalent_slope_delay_ms':float(np.median(delay)) if len(delay) else None}

def independent_step(scenario,target,end,state,hz):
    change=scenario['changeStartSeconds']*48000;indices=np.flatnonzero(end>=change);selected=None
    good=(state==1)&np.isfinite(hz)&(np.abs(1200*np.log2(hz/scenario['finalHz']))<=50)
    for position in range(len(indices)-2):
        i,j,k=indices[position:position+3]
        if good[[i,j,k]].all() and target[k]-target[i]==960:selected=i;break
    return {'target_time_first_of_three_settling_lag_ms':None if selected is None else (target[selected]-change)/48,
            'earliest_availability_first_of_three_delay_ms':None if selected is None else (end[selected]-change)/48,
            'settled':selected is not None}

def validate_feature(f,n):
    end=f['reference_sample'];require(np.array_equal(end,np.arange(2208,n+1,480)),'Analytic endpoint grid differs')
    require(int(f['sample_rate_hz'])==48000 and int(f['pcm_sample_count'])==n,'Analytic sample contract differs')
    require(f['features'].shape==(len(end),57) and f['candidate_hz'].shape==f['candidate_mask'].shape==(len(end),6),'Analytic shape differs')
    require(np.isfinite(f['features']).all(),'Nonfinite analytic evidence')
    return end,end-960,(end-960>=3072)&(end-960+1536<=n)

def checkpoint_identity(audit,run,manifest,updates):
    initial={};scaling=None
    for seed in SEEDS:
        for mode,objective in ARMS:
            name=base.key(mode,objective,seed);fit=manifest['fits'][name]
            require((fit['seed'],fit['context_mode'],fit['objective'])==(seed,mode,objective),'Fit identity differs')
            require(len(fit['epoch_losses'])==12 and all(math.isfinite(x) for x in fit['epoch_losses']),'Epoch losses invalid')
            prior_update=updates[base.key('shared-candidates',objective,seed)]
            audit.equal(fit['optimizer_updates'],prior_update[0],name+'/updates');audit.equal(fit['skipped_batches'],prior_update[1],name+'/skips')
            saved=torch.load(audit.bind(run/'checkpoints'/(name+'.pt'),fit['weights_sha256']),map_location='cpu',weights_only=True)
            require(saved['context_mode']==mode and saved['config']=={'representation':'shared-candidates','loss':objective,'hidden_size':32},'Saved model configuration differs')
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
    return {'new_checkpoints':12,'parameters_each':3009,'same_seed_initialization_hashes':initial,
            'initialization_note':'Equality of declared initialization digests checked here; separate inference audit reconstructs initialization.'}

def grouped_natural_and_prior(audit,run,manifest):
    arrays={m:load_npz(audit,run/'predictions'/(m+'.npz'),manifest['prediction_sha256']['predictions/'+m+'.npz']) for m in SELECTORS}
    prior_manifest=audit.json(RUNS/'selector-factorial-v1/manifest.json',manifest['base_experiment_sha256'])
    for m in PRIOR:
        original=m.removeprefix('prior__');path=RUNS/'selector-factorial-v1/predictions'/(original+'.npz')
        old=load_npz(audit,path,prior_manifest['prediction_sha256']['predictions/'+original+'.npz'])
        require(all(np.array_equal(arrays[m][k],old[k],equal_nan=True) for k in ('state','hz')),'Prior saved natural output changed')
        audit.bind(RUNS/'selector-factorial-v1/checkpoints'/(original+'.pt'),prior_manifest['fits'][original]['weights_sha256'])
    folder,cache,_=base.load_cache(audit,'development');offset=0;checked=0
    for record in cache['recordings']:
        f=base.load_feature(audit,folder,record);g=base.representative_hz(f);stop=offset+len(g)
        for m,z in arrays.items():
            state,hz=z['state'][offset:stop],z['hz'][offset:stop];v=state==1
            require(np.all(np.any(g[v]==hz[v,None],axis=1)),'Natural output not an actual grouped representative: '+m)
            checked+=int(v.sum())
        offset=stop
    return {'grouped_representative_emissions_checked':checked,'prior_natural_methods_unchanged':6}

def analytic_items(audit,run,manifest,cf):
    old=audit.json(RUNS/'delayed-known-truth/prediction-freeze.json')
    support=audit.json(RUNS/'support-diagnostic-v1/manifest.json',old['original_support_manifest_sha256'])
    specs={s['id']:s for s in support['scenarios']}
    for hz in (100,200,300):
        name=f'observation-change-{hz}';specs[name]={'id':name,'kind':'observation-change','durationSeconds':1.6,'initialHz':hz,'finalHz':hz,'changeStartSeconds':.6,'changeEndSeconds':.6,'slopeHzPerSecond':0}
    require(set(specs)==base.SYNTHETIC_IDS,'Regression scenario inventory differs')
    records=[]
    for row in old['recordings']:
        name=row['id'];f=load_npz(audit,RUNS/'delayed-known-truth/features'/(name+'.npz'),row['feature_sha256'])
        path=audit.bind(RUNS/'delayed-known-truth/predictions'/(name+'.json.gz'),row['prediction_sha256'])
        p=json.loads(gzip.decompress(path.read_bytes()))['predictions']['mlp'];require([v['sample'] for v in p]==(f['reference_sample']-960).tolist(),'Regression probability target differs')
        records.append(('regression',name,specs[name],f,np.array([v['voicing_probability'] for v in p])))
    cm=audit.json(run/'challenge/manifest.json',manifest['input_sha256'][str(run/'challenge/manifest.json')]);spec=audit.json(run/'challenge/spec.json',manifest['input_sha256'][str(run/'challenge/spec.json')])
    expected=expected_challenge_records();require(spec['records']==expected and len(cm['records'])==12,'Prospective configurations differ')
    spec_hash=hashlib.sha256(canonical(spec)).hexdigest();require(spec_hash==cm['spec_sha256']==manifest['challenge_spec_sha256']==cf['challenge_spec_sha256'],'Challenge spec hashes differ')
    require(cm['status']=='complete' and cm['reference_content_read'] is False and cm['checkpoint_freeze_sha256']==base.sha(run/'checkpoint-freeze.json'),'Challenge freeze binding differs')
    require(cm['checkpoint_freeze_path']==str((run/'checkpoint-freeze.json').relative_to(REPO)),'Challenge parent freeze differs')
    require(dt(cf['checkpoints_frozen_at'])<=dt(cm['started_at'])<=dt(cm['completed_at'])<=dt(manifest['predictions_frozen_at']),'Challenge phase order differs')
    for source,sha in cm['source_sha256'].items():
        require(manifest['source_sha256'].get(source)==sha,'Challenge source not bound before fitting')
        audit.bind(run/'challenge/source-snapshot'/source,sha)
    checkpoint=audit.bind(RUNS/'delayed-mlp-seed20261022/weights.pt',base.WEIGHTS)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True);config=ModelConfig(**saved['config']);weights={k:v.numpy() for k,v in saved['state_dict'].items()}
    probability_hashes={}
    for params,row in zip(expected,cm['records']):
        name=params['id'];require(row['id']==name and row['parameters_sha256']==hashlib.sha256(canonical(params)).hexdigest(),'Challenge parameters changed')
        pcm=audit.bind(base.inside(run/'challenge',row['pcm_path']),row['pcm_sha256']);require(pcm.stat().st_size==96000*4,'Challenge PCM extent differs')
        # Identity is checked without synthesizing any new waveform in this auditor.
        raw=np.fromfile(pcm,dtype='<f4');require(np.isfinite(raw).all(),'Nonfinite challenge PCM')
        f=load_npz(audit,base.inside(run/'challenge',row['feature_path']),row['feature_sha256']);require(len(f['features'])==row['feature_frames'],'Challenge feature count differs')
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            logits,_,_=numpy_step(config,weights,f['features'],f['candidate_mask'],np.zeros((len(f['features']),32),np.float32))
        require(np.isfinite(logits).all(),'Nonfinite independent original voicing')
        p=np.exp(-np.logaddexp(0.,-logits[:,0].astype(np.float64)))
        probability_hashes[name]=hashlib.sha256(p.astype('<f8').tobytes()).hexdigest()
        scenario={'id':name,'kind':'observation-change','durationSeconds':2.,'initialHz':params['frequency_hz'],'finalHz':params['frequency_hz'],'changeStartSeconds':.83,'changeEndSeconds':.83,'slopeHzPerSecond':0}
        records.append(('challenge',name,scenario,f,p))
    return records,{'challenge_started_at':cm['started_at'],'challenge_completed_at':cm['completed_at'],'spec_sha256':spec_hash,'independent_numpy_voicing_probability_sha256':probability_hashes}

def analytic(audit,run,manifest,metrics,items):
    gates={'stationary_by_seed_case':{m:{} for m in CANDIDATE},'harmonic_change_by_seed_case':{m:{} for m in CANDIDATE},'challenge_by_seed_case':{m:{} for m in CANDIDATE}}
    summaries={};frames=0;emissions=0
    for group,name,scenario,f,p in items:
        n=96000 if group=='challenge' else 76800;end,target,eligible=validate_feature(f,n)
        require(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Invalid original voicing probability')
        grouped=base.representative_hz(f);truth=curve(scenario,target/48000);region=regions(scenario,end)
        coverage=np.any(np.abs(1200*np.log2(grouped/truth[:,None]))<=50,axis=1)&eligible
        report=metrics[group][name];require(set(report['methods'])==set(SELECTORS),'Analytic method inventory differs')
        for k,v in {'eligible_frames':int(eligible.sum()),'covered_frames':int(coverage.sum()),'fraction':float(coverage.sum()/eligible.sum())}.items():audit.equal(v,report['candidate_coverage'][k],name+'/candidate coverage/'+k)
        path=run/'predictions'/(group+'-'+name+'.npz');z=load_npz(audit,path,manifest['prediction_sha256'][str(path.relative_to(run))])
        require(set(z)=={m+s for m in SELECTORS for s in ('__state','__hz')},'Analytic prediction field inventory differs')
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
                if group=='challenge':gate='challenge_by_seed_case';selection=eligible&(region=='after-observation-change')
                elif scenario['kind']=='constant':gate='stationary_by_seed_case';selection=eligible
                elif scenario['kind']=='observation-change':gate='harmonic_change_by_seed_case';selection=eligible&(region=='after-observation-change')
                else:continue
                summary=analytic_summary(truth[selection],state[selection],hz[selection],region[selection],0)
                passed=summary['success_within_50c_all_frames']>=.95;gates[gate][method][name]=passed
                case[method]={'frames':summary['frames'],'success':summary['success_within_50c_all_frames'],'pass':passed}
        summaries[group+'/'+name]=case
    for key,count in [('stationary_by_seed_case',3),('harmonic_change_by_seed_case',3),('challenge_by_seed_case',12)]:
        require(all(len(cases)==count for cases in gates[key].values()),'Incomplete gate case inventory')
        require(gates[key]==metrics['decision']['analytic'][key],'Analytic per-seed gate differs')
    for flag,key in [('stationary_pass','stationary_by_seed_case'),('harmonic_change_pass','harmonic_change_by_seed_case'),('challenge_pass','challenge_by_seed_case')]:
        gates[flag]=all(all(cases.values()) for cases in gates[key].values());audit.equal(gates[flag],metrics['decision']['analytic'][flag],flag)
    return {'cases':26,'selector_methods_per_case':18,'prediction_rows_checked':frames,'grouped_emissions_checked':emissions,'candidate_case_gates':summaries,'gates':gates}

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
    control=float(np.mean([mean(base.key('absolute','acceptable-set',s),'joint_error') for s in SEEDS]))
    audit.equal(control-error,reported['improvement_over_equal_parameter_absolute_control'],'Absolute control comparison')
    return checks

def audit_run(run,recovery,output):
    run=base.inside(RUNS,run);recovery=base.inside(RUNS,recovery);output=base.inside(recovery,output);require(not output.exists(),'Refuses to overwrite audit evidence')
    torch.set_num_threads(1);audit=base.Audit();m=audit.json(run/'manifest.json')
    require(m['schema']=='phenometrix-harmonic-context-experiment.v1' and m['status']=='failed' and m['failure']=="TypeError('Object of type bool is not JSON serializable')",'Requires preserved original serialization failure')
    require(not (run/'metrics.json').exists(),'Unexpected original metrics replacement')
    recovered=audit.json(recovery/'manifest.json')
    require(recovered['schema']=='phenometrix-harmonic-score-recovery.v1' and recovered['status']=='complete' and recovered['original_run']==str(run.relative_to(REPO)) and recovered['original_manifest_sha256']==base.sha(run/'manifest.json') and recovered['original_failure']==m['failure'],'Recovery identity differs')
    require(all(recovered[k] is False for k in ('fitting_performed','inference_performed','waveform_generation_performed','reserved_content_read')),'Recovery scope differs')
    for path,sha in recovered['source_sha256'].items():audit.bind(REPO/path,sha)
    for path,sha in recovered['input_sha256'].items():audit.bind(path,sha)
    audit.bind(recovery/'recover_harmonic_scores.py',recovered['source_sha256']['research/voice-prosody/phenometrix_prosody/recover_harmonic_scores.py'])
    require(m['seeds']==list(SEEDS) and m['arms']==[list(x) for x in ARMS] and m['predesignated_candidate']==['relative','acceptable-set'],'Run arms differ')
    require(set(m['fits'])==set(NEW) and m['epochs']==12 and m['batch_size']==4096 and m['learning_rate']==.001 and m['measurement_delay_ms']==20 and m['device']=='mps','Schedule/configuration differs')
    require(m['frozen_original_voicing'] is True and m['reserved_content_read'] is False and m['fit_elapsed_seconds']<=1800 and m['peak_observed_mps_allocated_bytes']<=8*1024**3 and sum(p.stat().st_size for p in run.rglob('*') if p.is_file())<=2*1024**3,'Scope/resource declarations differ')
    pre=audit.json(RUNS/'selector-prerequisites-v1/prerequisites.json',m['prerequisites_sha256']);require(pre['pass'] is True,'Passing historical prerequisites required')
    audit.bind(RUNS/'selector-factorial-v1/manifest.json',m['base_experiment_sha256'])
    for path,sha in m['source_sha256'].items():audit.bind(REPO/path,sha);audit.bind(run/'source-snapshot'/path,sha)
    for path,sha in m['input_sha256'].items():audit.bind(path,sha)
    require(all(m['source_sha256'].get(k)==v for k,v in pre['source_sha256'].items()),'Historical source identity changed')
    cf=audit.json(run/'checkpoint-freeze.json');pf=audit.json(run/'prediction-freeze.json')
    require(cf['status']=='checkpoints_frozen' and pf['status']=='predictions_frozen','Freeze states differ')
    require(cf['fits']==pf['fits']==m['fits'] and cf['source_sha256']==pf['source_sha256']==m['source_sha256'] and pf['prediction_sha256']==m['prediction_sha256'],'Frozen identities differ')
    require(cf['development_references_read'] is False and pf['development_references_read'] is False and m['development_references_read'] is True,'Declared reference chronology differs')
    require(not any('/records/development/' in p for p in pf['input_sha256']),'Development reference hash read before prediction freeze')
    times=[dt(m[k]) for k in ('started_at','checkpoints_frozen_at','predictions_frozen_at','failed_at')]+[dt(recovered[k]) for k in ('started_at','completed_at')];require(times==sorted(times),'Freeze/recovery time ordering differs')
    require(recovered['checkpoint_freeze_sha256']==base.sha(run/'checkpoint-freeze.json') and recovered['prediction_freeze_sha256']==base.sha(run/'prediction-freeze.json'),'Recovery changed freeze identities')
    require(cf['checkpoints_frozen_at']==m['checkpoints_frozen_at'] and pf['predictions_frozen_at']==m['predictions_frozen_at'],'Freeze timestamp identity differs')
    source=(run/'source-snapshot/research/voice-prosody/phenometrix_prosody/harmonic_selector_experiment.py').read_text()
    require(source.index("old.save(output/'checkpoint-freeze.json',m)")<source.index('generate_challenge(') and source.index("old.save(output/'prediction-freeze.json',m)")<source.index('dl=old.load_labels(dev)'),'Source phase ordering differs')
    expected={'predictions/'+n+'.npz' for n in METHODS}|{'predictions/regression-'+n+'.npz' for n in base.SYNTHETIC_IDS}|{'predictions/challenge-'+r['id']+'.npz' for r in expected_challenge_records()}
    require(set(m['prediction_sha256'])==expected and {str(p.relative_to(run)) for p in (run/'predictions').glob('*.npz')}==expected,'Prediction inventory differs')
    require({p.stem for p in (run/'checkpoints').glob('*.pt')}==set(NEW),'Checkpoint inventory differs')
    for path,sha in m['prediction_sha256'].items():audit.bind(run/path,sha)
    metrics=audit.json(recovery/'metrics.json',recovered['metrics_sha256']);require(set(metrics['regression'])==base.SYNTHETIC_IDS and set(metrics['challenge'])=={r['id'] for r in expected_challenge_records()},'Scored case inventory differs')
    updates,schedule=base.training_schedule(audit,m);checkpoints=checkpoint_identity(audit,run,m,updates)
    member_receipt=grouped_natural_and_prior(audit,run,m)
    natural_cells,natural_receipt=base.natural(audit,run,m,metrics['natural'],METHODS)
    items,challenge_receipt=analytic_items(audit,run,m,cf);analytic_receipt=analytic(audit,run,m,metrics,items)
    checks=natural_decision(audit,natural_cells,metrics['decision']);promising=all(checks.values()) and all(analytic_receipt['gates'][k] for k in ('stationary_pass','harmonic_change_pass','challenge_pass'))
    audit.equal(promising,metrics['decision']['promising_for_next_study'],'Combined decision')
    for path in [Path(__file__),Path(base.__file__),Path(__file__).with_name('metrics.py'),Path(__file__).with_name('pilot.py'),Path(__file__).with_name('model.py')]:audit.bind(path)
    audit.close()
    receipt={'schema':'phenometrix-harmonic-experiment-independent-audit.v1','status':'passed','audited_at':datetime.now(timezone.utc).isoformat(),'run_path':str(run.relative_to(REPO)),'run_manifest_sha256':base.sha(run/'manifest.json'),'original_run_status':m['status'],'original_failure':m['failure'],'recovery_path':str(recovery.relative_to(REPO)),'recovery_manifest_sha256':base.sha(recovery/'manifest.json'),'metrics_sha256':base.sha(recovery/'metrics.json'),'checkpoint_freeze_sha256':base.sha(run/'checkpoint-freeze.json'),'prediction_freeze_sha256':base.sha(run/'prediction-freeze.json'),'audit_source_sha256':base.sha(Path(__file__)),'bound_artifacts':len(audit.files),'numeric_comparisons':audit.comparisons,'maximum_numeric_difference':audit.max_rate_delta,'checkpoints':checkpoints,'training_schedule':schedule,'natural':natural_receipt|member_receipt,'analytic':analytic_receipt,'challenge_identity':challenge_receipt,'natural_checks':checks,'natural_pass':all(checks.values()),'promising_for_next_study':promising,'opened_input_source_sha256':audit.files,'limitations':['No refitting, new challenge generation, reserved/test access, or promotion claim.','Natural scores use object-based metrics.score, not the experiment array scorer.','Saved outputs, member frequencies, checkpoint bytes/shapes and declared phases are audited; separate agent reproduces selector inference and initialization.','Chronology is source inspection plus frozen manifests/timestamps, not an OS-level file-access log.','Challenge PCM identities/extent/finiteness are checked; this auditor does not regenerate waveform or TypeScript features.']}
    output.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print(json.dumps({'status':'passed','comparisons':audit.comparisons,'natural_pass':receipt['natural_pass'],'promising_for_next_study':promising,'output':str(output)}));return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--recovery',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit_run(a.run.resolve(),a.recovery.resolve(),a.output.resolve())
