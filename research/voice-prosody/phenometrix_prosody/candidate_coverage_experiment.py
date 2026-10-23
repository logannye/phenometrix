"""Frozen no-fit candidate-retention comparison; all computation stays local."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import resource
import signal
import shutil
import time
import numpy as np
import torch
from . import selector_experiment as old, augmentation_experiment as aug
from . import harmonic_selector_experiment as harmonic, evaluate as ev
from . import coverage_cases as cases
from . import candidate_coverage_features as acquisition
from .features import transform
from .pilot import read_mic
from .harmonic_selector_model import HarmonicSelector
from .selector_model import prepare_selector_batch, build_pitch_targets
from .evaluate_synthetic import OBSERVATION_SCENARIOS, observation_pcm, analytic_frequency
from .diagnose_subharmonic_challenge import support_region

ROOT,RUNS,REPO=old.ROOT,old.RUNS,old.REPO
PLAN=ROOT/'CANDIDATE_COVERAGE_PLAN.md'
PRIOR=RUNS/'harmonic-augmentation-v1'
PRIOR_SHA='d5513b83672f14a711f23a36e5258d348ac9b01a062c5945faaeaf5730d69426'
PRIOR_PREDICTION_FREEZE_SHA='83097642ae265e9a5a21321bc0dc2cdaa977fe0862c30fe5db550d809314dc5e'
ARMS=('original','revised');SEEDS=old.SEEDS;CANDIDATE='revised'
POLICY='shortest-admitted-full40-v1'
GROUPS={'regression':14,'consumed_harmonic':12,'consumed_subharmonic':12,'consumed_augmentation':12,'challenge':12}
MAX_SECONDS=75*60;MAX_BYTES=2*1024**3;MAX_RSS=16*1024**3

def key(arm,seed):return f'{arm}__seed{seed}'
def save(path,value):aug.save(path,value)
def npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sources():
    paths=[PLAN,Path(__file__),Path(acquisition.__file__),ROOT/'coverage_peaks.ts',Path(transform.__code__.co_filename),Path(support_region.__code__.co_filename)]
    return aug.sources()|cases.source_hashes()|acquisition.source_hashes()|{str(p.relative_to(REPO)):ev.sha(p) for p in paths}
def check_sources(frozen):
    if sources()!=frozen:raise ValueError('Frozen candidate study sources changed')
def bind(m,path,expected=None):
    path=Path(path).resolve()
    if not path.is_relative_to(REPO):raise ValueError('Evidence escaped repository')
    digest=ev.sha(path)
    if expected is not None and digest!=expected:raise ValueError('Input changed: '+str(path.relative_to(REPO)))
    name=str(path.relative_to(REPO));historical=m.get('historical_input_sha256',{}).get(name)
    if historical is not None and historical!=digest:raise ValueError('Historical input identity changed: '+name)
    previous=m['input_sha256'].get(name)
    if previous is not None and previous!=digest:raise ValueError('Input changed during run')
    m['input_sha256'][name]=digest;return path

def budget(output,started,m,worker=None):
    elapsed=time.monotonic()-started
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    m['peak_observed_python_rss_bytes']=max(m.get('peak_observed_python_rss_bytes',0),rss)
    if elapsed>MAX_SECONDS:raise TimeoutError('75 minute scientific execution budget exceeded')
    if rss>MAX_RSS:raise MemoryError('16 GiB Python RSS budget exceeded')
    size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    if size>MAX_BYTES:raise RuntimeError('2 GiB new artifact budget exceeded')
    if worker is not None:m['peak_observed_node_rss_bytes']=max(m.get('peak_observed_node_rss_bytes',0),getattr(worker,'peak_rss_bytes',0))
    if m.get('peak_observed_node_rss_bytes',0)>2*1024**3:raise MemoryError('2 GiB Node RSS budget exceeded')

def checkpoint_check(frozen):
    if set(frozen['checkpoints'])!={f'balance-aug__seed{s}' for s in SEEDS}:raise ValueError('Expected exactly three frozen selectors')
    for row in frozen['checkpoints'].values():
        p=(REPO/row['path']).resolve()
        if not p.is_relative_to(PRIOR/'checkpoints') or ev.sha(p)!=row['sha256']:raise ValueError('Frozen checkpoint changed')

def prepare(f):return {rep:prepare_selector_batch(f['features'],f['candidate_hz'],f['candidate_mask'],representation=rep) for rep in ('fixed-slots','shared-candidates')}

def cache_natural(data,output,worker,m,started):
    folder=output/data['split'];folder.mkdir();(folder/'features').mkdir();(folder/'provenance').mkdir();temporary=output/'working.f32le'
    micro={r['recording_id']:r for r in data['acquisition']['files'] if r['role']=='mic' and r['split']==data['split']}
    cache={'schema':'phenometrix-candidate-policy-cache.v1','status':'extracting','split':data['split'],'candidate_policy':POLICY,
           'original_manifest_path':str((data['folder']/'manifest.json').relative_to(REPO)),
           'original_manifest_sha256':ev.sha(data['folder']/'manifest.json'),'source_sha256':m['source_sha256'],
           'started_at':old.now(),'reference_content_read':False,'records':[]}
    save(folder/'manifest.json',cache);pieces=[]
    for i,record in enumerate(data['records']):
        budget(output,started,m,worker)
        worker.timeout_seconds=min(120.,max(.001,MAX_SECONDS-(time.monotonic()-started)))
        source=micro[record['source_recording_id']];path=bind(m,ROOT/'data/ptdb-tug'/source['path'],source['sha256'])
        pcm,_=transform(read_mic(path),record['condition'],record['source_recording_id'],data['split'])
        if len(pcm)!=record['sample_count'] or hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest()!=record['transformed_pcm_sha256']:
            raise ValueError('Transformed microphone PCM differs from original cache')
        pcm.astype('<f4').tofile(temporary)
        original_path=bind(m,data['folder']/'features'/(record['recording_id']+'.npz'),record['feature_sha256']);feature=npz(original_path)
        revised,provenance=acquisition.revised_feature(feature,temporary,worker)
        ev.validate_feature(revised,record)
        fp=folder/'features'/(record['recording_id']+'.npz');pp=folder/'provenance'/(record['recording_id']+'.npz')
        np.savez_compressed(fp,**revised);np.savez_compressed(pp,**provenance)
        saved={k:v for k,v in record.items() if k not in ('targets','eligible')}
        saved.update(original_feature_path=str(original_path.relative_to(REPO)),original_feature_sha256=ev.sha(original_path),
            feature_path=str(fp.relative_to(REPO)),feature_sha256=ev.sha(fp),provenance_path=str(pp.relative_to(REPO)),provenance_sha256=ev.sha(pp))
        cache['records'].append(saved);pieces.append({k:revised[k] for k in ('features','candidate_hz','candidate_mask')})
        temporary.unlink();bind(m,fp);bind(m,pp);budget(output,started,m,worker)
        if (i+1)%32==0 or i+1==len(data['records']):
            save(folder/'manifest.json',cache);budget(output,started,m,worker)
            print(json.dumps({'phase':'candidate_extraction','split':data['split'],'completed':i+1,'total':len(data['records'])}),flush=True)
    cache.update(status='complete',completed_at=old.now());save(folder/'manifest.json',cache);bind(m,folder/'manifest.json')
    return {**data,'raw':{k:np.concatenate([v[k] for v in pieces]) for k in pieces[0]}}

def original_analytic(m):
    support=json.loads(bind(m,RUNS/'support-diagnostic-v1/manifest.json').read_text())
    original_manifest=json.loads(bind(m,RUNS/'delayed-known-truth/manifest.json').read_text());reg={r['id']:r for r in original_manifest['recordings']}
    scenarios={s['id']:s for s in [*support['scenarios'],*OBSERVATION_SCENARIOS]};groups={}
    rows=[]
    for name,scenario in scenarios.items():
        fp=bind(m,RUNS/'delayed-known-truth/features'/(name+'.npz'),reg[name]['feature_sha256']);feature=npz(fp)
        if name.startswith('observation-change'):
            pcm=observation_pcm(scenario)
            if hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest()!=reg[name]['pcm_sha256']:raise ValueError('Original observation PCM changed')
            pcm_path=None
        else:pcm_path=bind(m,RUNS/'support-diagnostic-v1/pcm'/(name+'.f32le'),reg[name]['pcm_sha256']);pcm=None
        rows.append({'id':name,'scenario':scenario,'feature':feature,'original_feature_path':fp,'pcm_path':pcm_path,'pcm':pcm,
                     'pcm_sha256':reg[name]['pcm_sha256'],'timing':ev.validate_feature(feature,{'sample_count':int(feature['pcm_sample_count'])})})
    groups['regression']=rows
    for group,folder in [('consumed_harmonic',RUNS/'harmonic-context-v1/challenge'),('consumed_subharmonic',RUNS/'subharmonic-objective-v1/challenge'),('consumed_augmentation',PRIOR/'challenge')]:
        cache=json.loads(bind(m,folder/'manifest.json').read_text());spec=json.loads(bind(m,folder/'spec.json').read_text());records={r['id']:r for r in spec['records']}
        if cache['status']!='complete' or len(cache['records'])!=12 or set(records)!={r['id'] for r in cache['records']}:raise ValueError('Incomplete consumed analytic domain')
        from .harmonic_challenge import _scenario
        rows=[]
        for r in cache['records']:
            fp=bind(m,folder/r['feature_path'],r['feature_sha256']);feature=npz(fp);pp=bind(m,folder/r['pcm_path'],r['pcm_sha256'])
            rows.append({'id':r['id'],'scenario':_scenario(records[r['id']]),'feature':feature,'original_feature_path':fp,'pcm_path':pp,'pcm':None,
                         'pcm_sha256':r['pcm_sha256'],'timing':ev.validate_feature(feature,{'sample_count':int(feature['pcm_sample_count'])})})
        groups[group]=rows
    return groups

def natural_decision(natural):
    rows=[natural[key(CANDIDATE,s)]['macro'] for s in SEEDS]
    mean=float(np.mean([r['joint_error'] for r in rows]));success=float(np.mean([r['reference_voiced_success_rate'] for r in rows]))
    gains={base:natural[base]['macro']['joint_error']-mean for base in ('original_mlp','original_hybrid')}
    excess={c:float(np.mean([natural[key(CANDIDATE,s)]['per_condition'][c]['joint_error'] for s in SEEDS]))-natural['original_mlp']['per_condition'][c]['joint_error'] for c in old.CONDITIONS}
    checks={'mean_improves_both_by_half_point':all(v>=.005 for v in gains.values()),'each_seed_improves_original_mlp':all(r['joint_error']<natural['original_mlp']['macro']['joint_error'] for r in rows),
            'no_condition_worsens_over_one_point':all(v<=.01 for v in excess.values()),'voiced_success_not_lower':success>=natural['original_mlp']['macro']['reference_voiced_success_rate']}
    return {'predesignated_candidate':CANDIDATE,'pass':all(checks.values()),'checks':checks,'mean_joint_error':mean,'mean_voiced_success':success,
            'joint_error_improvement_over_frozen':gains,'condition_error_excess_over_mlp':excess,
            'improvement_over_paired_original':float(np.mean([natural[key('original',s)]['macro']['joint_error'] for s in SEEDS]))-mean}

def analytic_decision(groups):
    if {g:len(rows) for g,rows in groups.items()}!=GROUPS:raise ValueError('Incomplete 62-case inventory')
    expected={'regression':old.SYNTHETIC_IDS,**{g:{f'{prefix}-f{i}-p{j}' for i in range(6) for j in range(2)} for g,prefix in [('consumed_harmonic','harmonic-balance'),('consumed_subharmonic','subharmonic-balance'),('consumed_augmentation','augmentation-balance'),('challenge','coverage-balance')]}}
    if any(set(groups[g])!=ids for g,ids in expected.items()):raise ValueError('Analytic case identities changed')
    names=[key('revised',s) for s in SEEDS];gates={g:{n:{} for n in names} for g in ('stationary','harmonic_change',*tuple(GROUPS)[1:])}
    for n in names:
        for case,v in groups['regression'].items():
            if v['kind']=='constant':gates['stationary'][n][case]=v['methods'][n]['all']['success_within_50c_all_frames']>=.95
            elif v['kind']=='observation-change':gates['harmonic_change'][n][case]=v['methods'][n]['regions']['after-observation-change']['success_within_50c_all_frames']>=.95
        for group in tuple(GROUPS)[1:]:
            gates[group][n]={case:v['methods'][n]['regions']['after-observation-change']['success_within_50c_all_frames']>=.95 for case,v in groups[group].items()}
    for g,count in {'stationary':3,'harmonic_change':3,**{g:12 for g in tuple(GROUPS)[1:]}}.items():
        if any(len(v)!=count for v in gates[g].values()):raise ValueError('Incomplete analytic gate cases')
    return {**{g+'_by_seed_case':v for g,v in gates.items()},**{g+'_pass':bool(all(all(x.values()) for x in v.values())) for g,v in gates.items()}}

def coverage_transition(data,labels,batches):
    covered={arm:{rep:old.targets(b,labels).covered.numpy() for rep,b in prepared.items()} for arm,prepared in batches.items()}
    cells={}
    for r in data['records']:
        sl=slice(r['start'],r['stop']);include=labels['valid'][sl]&labels['voiced'][sl];cell=cells.setdefault(r['speaker']+'/'+r['condition'],{'voiced':0,**{rep:{'lost':0,'gained':0,'both':0,'neither':0} for rep in ('fixed-slots','shared-candidates')}});cell['voiced']+=int(include.sum())
        for rep in ('fixed-slots','shared-candidates'):
            a,b=covered['original'][rep][sl],covered['revised'][rep][sl]
            for field,mask in [('lost',a&~b),('gained',~a&b),('both',a&b),('neither',~a&~b)]:cell[rep][field]+=int((include&mask).sum())
    for cell in cells.values():
        for rep in ('fixed-slots','shared-candidates'):
            for k in ('lost','gained','both','neither'):cell[rep][k+'_fraction']=cell[rep][k]/cell['voiced']
    return {'cells':cells,'macro':{rep:{k+'_fraction':float(np.mean([v[rep][k+'_fraction'] for v in cells.values()])) for k in ('lost','gained','both','neither')} for rep in ('fixed-slots','shared-candidates')}}

def oracle_headroom(data,labels,prepared,probability,natural):
    results={}
    for rep in ('fixed-slots','shared-candidates'):
        covered=old.targets(prepared[rep],labels).covered.numpy();correct=(labels['voiced']&(probability>=.5)&covered)|(~labels['voiced']&(probability<.5));counts={}
        for r in data['records']:
            sl=slice(r['start'],r['stop']);v=labels['valid'][sl];c=counts.setdefault(r['speaker']+'/'+r['condition'],{'eligible':0,'oracle_correct':0})
            c['eligible']+=int(v.sum());c['oracle_correct']+=int((v&correct[sl]).sum())
        for name,c in counts.items():
            prior=natural['original_mlp']['cells'][name]
            if c['eligible']!=prior['eligible_valid_frames']:raise ValueError('Headroom denominator differs')
            c['mlp_correct']=prior['correct_frames'];c['gain']=(c['oracle_correct']-c['mlp_correct'])/c['eligible']
        mean=float(np.mean([c['gain'] for c in counts.values()]));results[rep]={'cells':counts,'mean_gain':mean,'pass':mean>=.01}
    return results

def analytic_coverage(item):
    timing,scenario=item['timing'],item['scenario'];truth=np.array([analytic_frequency(scenario,int(s)/48000) for s in timing.reference_sample])
    valid=timing.common_eligible;labels={'hz':truth,'voiced':np.ones(len(truth),bool),'valid':valid};regions=np.array([support_region(scenario,int(e)) for e in timing.availability_sample]);cov={}
    for arm,prepared in item['prepared'].items():cov[arm]={rep:old.targets(b,labels).covered.numpy() for rep,b in prepared.items()}
    result={}
    for name,include in {'all':valid,**{r:valid&(regions==r) for r in sorted(set(regions[valid]))}}.items():
        result[name]={'frames':int(include.sum()),'policies':{},'policy_change':{},
                      'original_voiced_gate_positive':int((include&(item['probability']>=.5)).sum()),
                      'methods':{method:{label:int((include&(state==value)).sum()) for label,value in [('unvoiced',0),('voiced',1),('unavailable',2)]} for method,(state,_) in item['predictions'].items()}}
        for arm in ARMS:
            a,b=cov[arm]['fixed-slots'],cov[arm]['shared-candidates'];result[name]['policies'][arm]={'raw_covered':int((include&a).sum()),'grouped_covered':int((include&b).sum()),'grouping_lost':int((include&a&~b).sum()),'candidate_unavailable':int((include&~item['prepared'][arm]['shared-candidates'].candidate_mask.numpy().any(axis=1)).sum())}
        for rep in ('fixed-slots','shared-candidates'):
            a,b=cov['original'][rep],cov['revised'][rep];result[name]['policy_change'][rep]={'lost':int((include&a&~b).sum()),'gained':int((include&~a&b).sum()),'both':int((include&a&b).sum()),'neither':int((include&~a&~b).sum())}
    return result

def run(output):
    output=Path(output).resolve()
    if not output.is_relative_to(RUNS) or output.exists():raise ValueError('Fresh ignored local output required')
    if shutil.disk_usage(ROOT).free<8*1024**3:raise RuntimeError('8 GiB free disk required')
    if ev.sha(PRIOR/'manifest.json')!=PRIOR_SHA or ev.sha(PRIOR/'prediction-freeze.json')!=PRIOR_PREDICTION_FREEZE_SHA:raise ValueError('Pinned prior augmentation changed')
    pm=json.loads((PRIOR/'manifest.json').read_text());pf=json.loads((PRIOR/'prediction-freeze.json').read_text())
    if pm['status']!='complete' or pf['status']!='predictions_frozen' or pm['prediction_sha256']!=pf['prediction_sha256']:raise ValueError('Prior freeze invalid')
    aug.check_sources(pm['source_sha256']);frozen=sources();started=time.monotonic();torch.set_num_threads(1)
    output.mkdir();(output/'source-snapshot').mkdir();(output/'predictions').mkdir();(output/'analytic').mkdir()
    checkpoints={f'balance-aug__seed{s}':{'path':str((PRIOR/'checkpoints'/f'balance-aug__seed{s}.pt').relative_to(REPO)),'sha256':pm['fits'][f'balance-aug__seed{s}']['weights_sha256']} for s in SEEDS}
    freeze={'schema':'phenometrix-candidate-coverage-freeze.v1','status':'sources_checkpoints_frozen','frozen_at':old.now(),'source_sha256':frozen,'checkpoints':checkpoints,'challenge_spec_sha256':cases.challenge_spec_sha256(),'candidate_policy':POLICY,'arms':list(ARMS),'seeds':list(SEEDS),'new_fits':0}
    checkpoint_check(freeze)
    for name,d in frozen.items():
        p=output/'source-snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((REPO/name).read_bytes())
        if ev.sha(p)!=d:raise ValueError('Source snapshot changed')
    save(output/'source-checkpoint-freeze.json',freeze)
    m={'schema':'phenometrix-candidate-coverage-experiment.v1','status':'extracting','started_at':old.now(),**{k:v for k,v in freeze.items() if k not in ('schema','status')},'source_checkpoint_freeze_sha256':ev.sha(output/'source-checkpoint-freeze.json'),'input_sha256':{},'historical_input_sha256':{str(Path(p).relative_to(REPO)):h for p,h in pm['input_sha256'].items()},'new_fits':0,'optimizer_updates':0,'development_references_read':False,'train_references_read':False,'reserved_content_read':False,'execution_device':'local CPU; zero fitting','peak_observed_python_rss_bytes':0,'peak_observed_node_rss_bytes':0}
    save(output/'manifest.json',m)
    try:
        for p in (PRIOR/'manifest.json',PRIOR/'prediction-freeze.json'):bind(m,p)
        for row in checkpoints.values():bind(m,REPO/row['path'],row['sha256'])
        models={}
        for s in SEEDS:
            ck=torch.load(REPO/checkpoints[f'balance-aug__seed{s}']['path'],map_location='cpu',weights_only=True)
            model=HarmonicSelector('relative','acceptable-set');model.load_state_dict(ck['state_dict'],strict=True);models[s]=model.eval()
        natural_data={};revised_data={}
        with acquisition.CoveragePeakWorker() as worker:
            for split in ('train','development'):
                natural_data[split]=old.load_inputs(split)
                for p,h in natural_data[split]['hashes'].items():bind(m,p,h)
                revised_data[split]=cache_natural(natural_data[split],output,worker,m,started)
            dev=natural_data['development'];probability,predictions,h=old.read_natural_baselines(dev)
            for p,d in h.items():bind(m,p,d)
            np.savez_compressed(output/'probabilities-development.npz',original_probability=probability);bind(m,output/'probabilities-development.npz')
            groups=original_analytic(m)
            fresh,h=cases.generate_challenge(output/'challenge',output/'source-checkpoint-freeze.json',ev.sha(output/'source-checkpoint-freeze.json'))
            for p,d in h.items():bind(m,p,d)
            fresh_cache=json.loads((output/'challenge/manifest.json').read_text());fresh_rows={r['id']:r for r in fresh_cache['records']}
            for item in fresh:
                r=fresh_rows[item['id']];item.update(pcm_path=output/'challenge'/r['pcm_path'],pcm=None,pcm_sha256=r['pcm_sha256'],original_feature_path=output/'challenge'/r['feature_path'])
            groups['challenge']=fresh
            analytic_inventory={};control_frames=0
            for group,items in groups.items():
                records=[]
                for item in items:
                    budget(output,started,m,worker)
                    worker.timeout_seconds=min(120.,max(.001,MAX_SECONDS-(time.monotonic()-started)))
                    temporary=output/'working.f32le';pp=item['pcm_path']
                    if pp is None:item['pcm'].astype('<f4').tofile(temporary);pp=temporary
                    revised,provenance=acquisition.revised_feature(item['feature'],pp,worker)
                    if temporary.exists():temporary.unlink()
                    fp=output/'analytic'/(group+'-'+item['id']+'.npz');vp=output/'analytic'/(group+'-'+item['id']+'-provenance.npz')
                    np.savez_compressed(fp,**revised);np.savez_compressed(vp,**provenance);bind(m,fp);bind(m,vp)
                    item['prepared']={'original':prepare(item['feature']),'revised':prepare(revised)};item['batch']=item['prepared']['revised']['shared-candidates']
                    prior_group={'regression':'regression','consumed_harmonic':'consumed_challenge','consumed_subharmonic':'consumed_subharmonic','consumed_augmentation':'challenge'}.get(group)
                    prior_saved=None
                    if prior_group:
                        relative='predictions/'+prior_group+'-'+item['id']+'.npz';prior_saved=npz(bind(m,PRIOR/relative,pf['prediction_sha256'][relative]));item['probability']=prior_saved['original_probability']
                    else:item['probability']=harmonic.original_probability(item['feature'])
                    preds={key(arm,s):old.infer(model,item['prepared'][arm]['shared-candidates'],item['probability']) for arm in ARMS for s,model in models.items()}
                    if prior_saved:
                        for s in SEEDS:
                            state,hz=preds[key('original',s)]
                            if not np.array_equal(state,prior_saved[f'balance-aug__seed{s}__state']) or not np.array_equal(hz,prior_saved[f'balance-aug__seed{s}__hz'],equal_nan=True):raise ValueError('Analytic original control changed')
                            control_frames+=len(state)
                    item['predictions']=preds;p=output/'predictions'/(group+'-'+item['id']+'.npz')
                    np.savez_compressed(p,original_probability=item['probability'],**{n+'__state':a[0] for n,a in preds.items()},**{n+'__hz':a[1] for n,a in preds.items()})
                    records.append({'id':item['id'],'scenario':item['scenario'],'sample_count':int(item['feature']['pcm_sample_count']),'original_feature_path':str(item['original_feature_path'].relative_to(REPO)),'original_feature_sha256':ev.sha(item['original_feature_path']),'revised_feature_path':str(fp.relative_to(REPO)),'revised_feature_sha256':ev.sha(fp),'provenance_path':str(vp.relative_to(REPO)),'provenance_sha256':ev.sha(vp),'pcm_sha256':item['pcm_sha256'],'prediction_path':str(p.relative_to(output)),'prediction_sha256':ev.sha(p)})
                    budget(output,started,m,worker)
                analytic_inventory[group]=records
                print(json.dumps({'phase':'analytic_predictions','group':group,'cases':len(items)}),flush=True)
            prepared_dev={arm:prepare(data['raw']) for arm,data in [('original',dev),('revised',revised_data['development'])]}
            for arm in ARMS:
                for s,model in models.items():predictions[key(arm,s)]=old.infer(model,prepared_dev[arm]['shared-candidates'],probability)
            for s in SEEDS:
                relative=f'predictions/balance-aug__seed{s}.npz';prior_npz=npz(bind(m,PRIOR/relative,pf['prediction_sha256'][relative]));state,hz=predictions[key('original',s)]
                if not np.array_equal(state,prior_npz['state']) or not np.array_equal(hz,prior_npz['hz'],equal_nan=True):raise ValueError('Natural original control changed')
                control_frames+=len(state)
            for name,(state,hz) in predictions.items():np.savez_compressed(output/'predictions'/(name+'.npz'),state=state,hz=hz)
            save(output/'analytic-inventory.json',analytic_inventory);bind(m,output/'analytic-inventory.json')
            m['control_reproduction']={'state_hz_frame_pairs':control_frames,'state_mismatches':0,'hz_mismatches':0,'all_exact':True}
            m['prediction_sha256']={str(p.relative_to(output)):ev.sha(p) for p in sorted((output/'predictions').glob('*.npz'))}
            m.update(status='predictions_frozen',predictions_frozen_at=old.now());budget(output,started,m,worker);check_sources(frozen);checkpoint_check(freeze)
            save(output/'prediction-freeze.json',m);save(output/'manifest.json',m)
        labels={split:old.load_labels(data) for split,data in natural_data.items()};m['train_references_read']=m['development_references_read']=True
        for label in labels.values():
            for p,h in label['hashes'].items():bind(m,p,h)
        natural=old.aggregate(dev,labels['development'],predictions);coverage={}
        for split in ('train','development'):
            original,revised=natural_data[split],revised_data[split];prepared=prepared_dev if split=='development' else {'original':prepare(original['raw']),'revised':prepare(revised['raw'])}
            coverage[split]={arm:old.coverage(original,labels[split],batch) for arm,batch in prepared.items()}
            coverage[split]['policy_change']=coverage_transition(original,labels[split],prepared)
            if split=='train':del prepared
            budget(output,started,m)
        headroom={arm:oracle_headroom(dev,labels['development'],prepared_dev[arm],probability,natural) for arm in ARMS}
        scores={}
        for group,items in groups.items():
            scored=harmonic.analytic_scores({i['id']:i['predictions'] for i in items},items)
            for item in items:scored[item['id']].pop('candidate_coverage');scored[item['id']]['candidate_coverage']=analytic_coverage(item)
            scores[group]=scored
        decision=natural_decision(natural);decision['analytic']=analytic_decision(scores)
        decision['prerequisites']={'raw_fixed_voicing_oracle_headroom':headroom['revised']['fixed-slots']['pass'],'train_grouping_loss':coverage['train']['revised']['pass'],'development_grouping_loss':coverage['development']['revised']['pass']}
        decision['promising_for_next_study']=decision['pass'] and all(decision['prerequisites'].values()) and all(v for k,v in decision['analytic'].items() if k.endswith('_pass'))
        result={'schema':'phenometrix-candidate-coverage-metrics.v1','scope':'paired no-fit inference policy study; consumed natural development,50regressions,12prospective noiseless configurations','natural':natural,'coverage':coverage,'headroom':headroom,**scores,'decision':decision,'exclusions':{s:l['excluded'] for s,l in labels.items()}}
        save(output/'metrics.json',result)
        for p,h in m['prediction_sha256'].items():
            if ev.sha(output/p)!=h:raise ValueError('Frozen prediction changed')
        for p,h in m['input_sha256'].items():
            if ev.sha(REPO/p)!=h:raise ValueError('Frozen input changed')
        check_sources(frozen);checkpoint_check(freeze);budget(output,started,m)
        m.update(status='complete',completed_at=old.now(),elapsed_seconds=time.monotonic()-started,metrics_sha256=ev.sha(output/'metrics.json'),output_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))
        save(output/'manifest.json',m);print(json.dumps(decision,indent=2,default=aug.json_scalar),flush=True);return result
    except BaseException as error:
        m.update(status='failed',failed_at=old.now(),failure=repr(error));save(output/'manifest.json',m);raise

def timeout_signal(signum, frame):
    raise TimeoutError('75 minute scientific execution wall-clock alarm')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();signal.signal(signal.SIGALRM,timeout_signal);signal.setitimer(signal.ITIMER_REAL,MAX_SECONDS)
    try:run(args.output)
    finally:signal.setitimer(signal.ITIMER_REAL,0)
