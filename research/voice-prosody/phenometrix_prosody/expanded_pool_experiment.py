"""One predeclared local six-fit additive-candidate experiment."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time
import numpy as np
import torch
from . import selector_experiment as old, augmentation_experiment as aug
from . import candidate_coverage_experiment as cov, harmonic_selector_experiment as harmonic
from . import evaluate as ev, expanded_cases as cases, augmentation_cases as training_cases
from . import expanded_candidate_features as acquisition, expanded_selector_model as modeling
from .candidate_coverage_features import CoveragePeakWorker
from .harmonic_challenge import _scenario, challenge_region_at
from .selector_model import build_pitch_targets

ROOT,RUNS,REPO=old.ROOT,old.RUNS,old.REPO
PLAN=ROOT/'EXPANDED_POOL_PLAN.md'
LAST=RUNS/'candidate-coverage-v1';PRIOR=RUNS/'harmonic-augmentation-v1'
LAST_SHA='1e2d7fbec0cb2dd71b177a0a5b646eff04d1d6dcf72a1c86eaac8f42f267aeff'
LAST_PRED_SHA='fd0bbe50c3e4be845484c2e77ef47cb538fdee1d2b978e2543a5269079f4b89b'
ARMS=('original6','expanded7');SEEDS=old.SEEDS;CANDIDATE='expanded7'
GROUPS={'regression':14,'consumed_harmonic':12,'consumed_subharmonic':12,
        'consumed_augmentation':12,'consumed_coverage':12,'challenge':12}
MAX_SECONDS=4500;MAX_FIT_SECONDS=1800;EXPECTED_UNION_ROWS=132466
save=aug.save;npz=cov.npz;bind=cov.bind

def key(arm,seed):return f'{arm}__seed{seed}'
def sources():
    paths=[PLAN,Path(__file__)]
    return cov.sources()|cases.source_hashes()|acquisition.source_hashes()|modeling.source_hashes()|{str(p.relative_to(REPO)):ev.sha(p) for p in paths}
def check_sources(expected):
    if sources()!=expected:raise ValueError('Expanded experiment source/plan changed')
def add_hashes(m,hashes):
    for p,h in hashes.items():bind(m,p if Path(p).is_absolute() else REPO/p,h)
def budget(output,started,m,worker=None):
    cov.budget(output,started,m,worker)
    if worker is not None:worker.timeout_seconds=min(120.,max(.001,MAX_SECONDS-(time.monotonic()-started)))
def prepare(feature):
    return {rep:modeling.prepare_expanded_selector_batch(feature['features'],feature['candidate_hz'],feature['candidate_mask'],representation=rep)
            for rep in ('fixed-slots','shared-candidates')}
def state_digest(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def check_weights(output,m):
    expected={key(a,s) for a in ARMS for s in SEEDS}
    if set(m['fits'])!=expected:raise ValueError('Incomplete six-fit inventory')
    for name,row in m['fits'].items():
        if ev.sha(output/'checkpoints'/(name+'.pt'))!=row['weights_sha256']:raise ValueError('Final checkpoint changed')
def check_old_weights(m):
    for row in m['historical_checkpoints'].values():bind(m,REPO/row['path'],row['sha256'])

def cache_natural(data,output,m,started):
    folder=output/data['split'];folder.mkdir();(folder/'features').mkdir();(folder/'provenance').mkdir()
    previous_path=bind(m,LAST/data['split']/'manifest.json')
    previous=json.loads(previous_path.read_text());rows={r['recording_id']:r for r in previous['records']}
    if previous['status']!='complete' or set(rows)!={r['recording_id'] for r in data['records']}:raise ValueError('Prior natural pool inventory differs')
    cache={'schema':'phenometrix-expanded-pool-natural-cache.v1','status':'assembling','split':data['split'],
           'source_sha256':m['source_sha256'],'candidate_policy':acquisition.POLICY,'started_at':old.now(),
           'reference_content_read':False,'original_manifest_path':str((data['folder']/'manifest.json').relative_to(REPO)),
           'original_manifest_sha256':ev.sha(data['folder']/'manifest.json'),'records':[]};pieces=[]
    for i,r in enumerate(data['records']):
        budget(output,started,m);prior=rows[r['recording_id']]
        fp=bind(m,data['folder']/'features'/(r['recording_id']+'.npz'),r['feature_sha256'])
        pp=bind(m,REPO/prior['provenance_path'],prior['provenance_sha256']);feature=npz(fp);provenance=npz(pp)
        if str(provenance['pcm_sha256'])!=r['transformed_pcm_sha256']:raise ValueError('Historical PCM binding differs')
        expanded,new_provenance=acquisition.expanded_from_provenance(feature,provenance)
        ep=folder/'features'/(r['recording_id']+'.npz');vp=folder/'provenance'/(r['recording_id']+'.npz')
        np.savez_compressed(ep,**expanded);np.savez_compressed(vp,**new_provenance);bind(m,ep);bind(m,vp)
        saved={k:v for k,v in r.items() if k not in ('targets','eligible')}
        saved.update(original_feature_path=str(fp.relative_to(REPO)),original_feature_sha256=ev.sha(fp),
            source_provenance_path=str(pp.relative_to(REPO)),source_provenance_sha256=ev.sha(pp),
            expanded_feature_path=str(ep.relative_to(REPO)),expanded_feature_sha256=ev.sha(ep),
            provenance_path=str(vp.relative_to(REPO)),provenance_sha256=ev.sha(vp))
        cache['records'].append(saved);pieces.append({k:expanded[k] for k in ('features','candidate_hz','candidate_mask')})
        if (i+1)%160==0 or i+1==len(data['records']):print(json.dumps({'phase':'expanded_cache','split':data['split'],'completed':i+1,'total':len(data['records'])}),flush=True)
    cache.update(status='complete',completed_at=old.now());save(folder/'manifest.json',cache);bind(m,folder/'manifest.json')
    return {**data,'raw':{k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}}

def coverage_reports(data,labels,prepared):
    result={arm:old.coverage(data,labels,batch) for arm,batch in prepared.items()}
    result['policy_change']=cov.coverage_transition(data,labels,{'original':prepared['original6'],'revised':prepared['expanded7']})
    if any(c['fixed-slots']['lost'] for c in result['policy_change']['cells'].values()):raise ValueError('Raw candidate superset lost correct coverage')
    return result

def natural_decision(natural):
    aliases={**natural,**{key('original',s):natural[key('original6',s)] for s in SEEDS},**{key('revised',s):natural[key('expanded7',s)] for s in SEEDS}}
    result=cov.natural_decision(aliases);result['predesignated_candidate']=CANDIDATE
    result['improvement_over_frozen_expanded']=float(np.mean([natural[key('frozen-expanded7',s)]['macro']['joint_error'] for s in SEEDS]))-result['mean_joint_error']
    result['original_control_drift_from_historical']=float(np.mean([natural[key('original6',s)]['macro']['joint_error']-natural[key('frozen-original6',s)]['macro']['joint_error'] for s in SEEDS]))
    return result

def training_schedule(raw_targets):
    original=raw_targets['original6'].covered;expanded=raw_targets['expanded7'].covered
    if original.shape!=expanded.shape or torch.any(original & ~expanded):
        raise ValueError('Expanded raw targets must preserve original coverage')
    union=original|expanded;schedule=torch.nonzero(union,as_tuple=False).reshape(-1)
    if int(original.sum())!=132181 or int((expanded & ~original).sum())!=285 or len(schedule)!=EXPECTED_UNION_ROWS or (len(schedule)+4095)//4096!=33:
        raise ValueError('Declared union schedule/count changed; stop before fits')
    return schedule

def analytic_decision(groups):
    if set(groups)!=set(GROUPS) or any(len(groups[g])!=n for g,n in GROUPS.items()):raise ValueError('Incomplete74-case inventory')
    from . import harmonic_challenge,subharmonic_challenge,augmentation_cases,coverage_cases
    inventories={'regression':old.SYNTHETIC_IDS,**{g:{r['id'] for r in module.challenge_spec()['records']} for g,module in
        [('consumed_harmonic',harmonic_challenge),('consumed_subharmonic',subharmonic_challenge),('consumed_augmentation',augmentation_cases),('consumed_coverage',coverage_cases),('challenge',cases)]}}
    if any(set(groups[g])!=ids for g,ids in inventories.items()):raise ValueError('Analytic case identity differs')
    names=[key(CANDIDATE,s) for s in SEEDS];gates={g:{n:{} for n in names} for g in ('stationary','harmonic_change',*tuple(GROUPS)[1:])}
    for n in names:
        for cid,row in groups['regression'].items():
            if row['kind']=='constant':gates['stationary'][n][cid]=row['methods'][n]['all']['success_within_50c_all_frames']>=.95
            elif row['kind']=='observation-change':gates['harmonic_change'][n][cid]=row['methods'][n]['regions']['after-observation-change']['success_within_50c_all_frames']>=.95
        for g in tuple(GROUPS)[1:]:gates[g][n]={cid:r['methods'][n]['regions']['after-observation-change']['success_within_50c_all_frames']>=.95 for cid,r in groups[g].items()}
    expected_counts={'stationary':3,'harmonic_change':3,**{g:12 for g in tuple(GROUPS)[1:]}}
    if any(len(gates[g][n])!=count for g,count in expected_counts.items() for n in names):raise ValueError('Expected3 stationary,3 change and12 per challenge group:66 gates per seed')
    return {**{g+'_by_seed_case':v for g,v in gates.items()},**{g+'_pass':all(all(row.values()) for row in v.values()) for g,v in gates.items()}}

def assemble_training(output,m,worker,started):
    folder=PRIOR/'training';cache=json.loads(bind(m,folder/'manifest.json').read_text())
    spec=json.loads(bind(m,folder/'spec.json').read_text());expected=training_cases.training_spec()
    if cache['status']!='complete' or spec!=expected:raise ValueError('Historical TRAIN specification changed')
    records={r['id']:r for r in cache['records'] if r['arm']=='balance-aug'};spec_rows=spec['records_by_arm']['balance-aug']
    if set(records)!={r['id'] for r in spec_rows} or len(records)!=96:raise ValueError('Expected96 balance TRAIN cases')
    target=output/'training';target.mkdir();(target/'features').mkdir();(target/'provenance').mkdir();items=[];saved=[]
    for r in spec_rows:
        budget(output,started,m,worker);row=records[r['id']]
        fp=bind(m,folder/row['feature_path'],row['feature_sha256']);pp=bind(m,folder/row['pcm_path'],row['pcm_sha256'])
        f=npz(fp);expanded,provenance=acquisition.expanded_feature(f,pp,worker)
        ep=target/'features'/(r['id']+'.npz');vp=target/'provenance'/(r['id']+'.npz')
        np.savez_compressed(ep,**expanded);np.savez_compressed(vp,**provenance);bind(m,ep);bind(m,vp)
        items.append({'id':r['id'],'scenario':_scenario(r),'feature':f,'expanded_feature':expanded,'timing':ev.validate_feature(f,{'sample_count':r['sample_count']})})
        saved.append({'id':r['id'],'scenario':_scenario(r),'original_feature_path':str(fp.relative_to(REPO)),'original_feature_sha256':ev.sha(fp),
            'pcm_path':str(pp.relative_to(REPO)),'pcm_sha256':ev.sha(pp),'expanded_feature_path':str(ep.relative_to(REPO)),'expanded_feature_sha256':ev.sha(ep),'provenance_path':str(vp.relative_to(REPO)),'provenance_sha256':ev.sha(vp)})
    save(target/'manifest.json',{'schema':'phenometrix-expanded-pool-training-cache.v1','status':'complete','records':saved,'original_spec_sha256':ev.sha(folder/'spec.json'),'source_sha256':m['source_sha256']});bind(m,target/'manifest.json')
    banks={}
    for arm in ARMS:
        raw={k:[] for k in ('features','candidate_hz','candidate_mask')};meta={k:[] for k in ('reference_hz','scenario_index','reference_sample','availability_sample','region')}
        for i,item in enumerate(items):
            timing=item['timing'];ix=np.flatnonzero(timing.common_eligible)
            if len(ix)!=191:raise ValueError('TRAIN common target count changed')
            feature=item['feature'] if arm=='original6' else item['expanded_feature']
            for k in raw:raw[k].append(feature[k][ix])
            meta['reference_hz'].append(np.full(len(ix),item['scenario']['initialHz']));meta['scenario_index'].append(np.full(len(ix),i,np.int64))
            meta['reference_sample'].append(timing.reference_sample[ix]);meta['availability_sample'].append(timing.availability_sample[ix]);meta['region'].append(np.array([challenge_region_at(item['scenario'],int(e)) for e in timing.availability_sample[ix]]))
        raw={k:np.concatenate(v) for k,v in raw.items()};meta={k:np.concatenate(v) for k,v in meta.items()};b=prepare(raw)['shared-candidates']
        t=old.targets(b,{'hz':meta['reference_hz'],'voiced':np.ones(len(b.features),bool),'valid':np.ones(len(b.features),bool)})
        banks[arm]=(b,t,meta)
    if len(banks['original6'][0].features)!=18336 or any(not np.array_equal(banks['original6'][2][k],banks['expanded7'][2][k]) for k in banks['original6'][2]):raise ValueError('Synthetic paired identity changed')
    return banks

def saved_analytic(output,m):
    inventory=json.loads(bind(m,LAST/'analytic-inventory.json').read_text());groups={}
    for old_group,records in inventory.items():
        group='consumed_coverage' if old_group=='challenge' else old_group;items=[]
        for r in records:
            fp=bind(m,REPO/r['original_feature_path'],r['original_feature_sha256']);pp=bind(m,REPO/r['provenance_path'],r['provenance_sha256'])
            pred_path=bind(m,LAST/r['prediction_path'],r['prediction_sha256']);f=npz(fp);provenance=npz(pp)
            if str(provenance['pcm_sha256'])!=r['pcm_sha256']:raise ValueError('Consumed analytic PCM/provenance identity differs')
            expanded,new_provenance=acquisition.expanded_from_provenance(f,provenance)
            items.append({'id':r['id'],'scenario':r['scenario'],'feature':f,'expanded_feature':expanded,'new_provenance':new_provenance,
                'timing':ev.validate_feature(f,{'sample_count':r['sample_count']}),'original_feature_path':fp,'pcm_sha256':r['pcm_sha256'],
                'source_provenance_path':str(pp.relative_to(REPO)),'source_provenance_sha256':ev.sha(pp),
                'prior_prediction_path':pred_path,'prior_prediction_sha256':ev.sha(pred_path)})
        groups[group]=items
    if set(groups)!=set(GROUPS)-{'challenge'} or sum(len(v) for v in groups.values())!=62:raise ValueError('Expected62consumed cases')
    return groups

def analytic_coverage(item):
    aliased={**item,'prepared':{'original':item['prepared']['original6'],'revised':item['prepared']['expanded7']}}
    result=cov.analytic_coverage(aliased)
    for region in result.values():
        region['policies']={'original6':region['policies']['original'],'expanded7':region['policies']['revised']}
        if region['policy_change']['fixed-slots']['lost']:raise ValueError('Raw analytic coverage was evicted')
    return result

def run(output):
    output=output.resolve();started=time.monotonic()
    if not output.is_relative_to(RUNS) or output.exists():raise ValueError('Fresh ignored run path required')
    if not torch.backends.mps.is_available():raise RuntimeError('Local MPS required for six fits')
    if shutil.disk_usage(ROOT).free<8*1024**3:raise RuntimeError('8 GiB disk required')
    if ev.sha(LAST/'manifest.json')!=LAST_SHA or ev.sha(LAST/'prediction-freeze.json')!=LAST_PRED_SHA or ev.sha(PRIOR/'manifest.json')!=cov.PRIOR_SHA:raise ValueError('Historical run identity changed')
    lm=json.loads((LAST/'manifest.json').read_text());pm=json.loads((PRIOR/'manifest.json').read_text());cov.check_sources(lm['source_sha256']);cov.checkpoint_check(lm)
    frozen=sources();output.mkdir()
    for name in ('source-snapshot','checkpoints','predictions','analytic'):(output/name).mkdir()
    for p,h in frozen.items():
        q=output/'source-snapshot'/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((REPO/p).read_bytes())
        if ev.sha(q)!=h:raise ValueError('Source snapshot changed')
    historical={str(Path(p).resolve().relative_to(REPO)) if Path(p).is_absolute() else p:h for p,h in pm['input_sha256'].items()}
    for p,h in lm['input_sha256'].items():
        if p in historical and historical[p]!=h:raise ValueError('Historical ledgers disagree')
        historical[p]=h
    m={'schema':'phenometrix-expanded-pool-experiment.v1','status':'sources_frozen','started_at':old.now(),'sources_frozen_at':old.now(),
       'source_sha256':frozen,'historical_input_sha256':historical,'historical_checkpoints':lm['checkpoints'],
       'challenge_spec_sha256':cases.challenge_spec_sha256(),'arms':list(ARMS),'seeds':list(SEEDS),'maximum_fits':6,
       'candidate_policy':acquisition.POLICY,'predesignated_candidate':CANDIDATE,'input_sha256':{},'fits':{},
       'device':'mps','torch':torch.__version__,'epochs':12,'batch_size':4096,'learning_rate':.001,
       'development_references_read_for_prerequisites':False,'development_references_read_for_final_scoring':False,
       'reserved_content_read':False,'peak_observed_mps_allocated_bytes':0,'peak_observed_python_rss_bytes':0,'peak_observed_node_rss_bytes':0}
    source_freeze={**m,'schema':'phenometrix-expanded-pool-source-freeze.v1'}
    save(output/'source-freeze.json',source_freeze);save(output/'manifest.json',m)
    try:
        torch.set_num_threads(1);check_old_weights(m)
        for p,h in [(LAST/'manifest.json',LAST_SHA),(LAST/'prediction-freeze.json',LAST_PRED_SHA),(PRIOR/'manifest.json',cov.PRIOR_SHA)]:bind(m,p,h)
        original={};expanded={};prepared={}
        for split in ('train','development'):
            original[split]=old.load_inputs(split);add_hashes(m,original[split]['hashes']);expanded[split]=cache_natural(original[split],output,m,started)
            prepared[split]={'original6':prepare(original[split]['raw']),'expanded7':prepare(expanded[split]['raw'])}
        dev=original['development'];probability,predictions,h=old.read_natural_baselines(dev);add_hashes(m,h)
        np.savez_compressed(output/'probabilities-development.npz',original_probability=probability);bind(m,output/'probabilities-development.npz')
        save(output/'input-cache-freeze.json',{'schema':'phenometrix-expanded-pool-input-cache-freeze.v1','status':'input_caches_frozen','frozen_at':old.now(),'source_sha256':frozen,'input_sha256':m['input_sha256'],'reference_content_read':False})
        bind(m,output/'input-cache-freeze.json');budget(output,started,m)
        labels={s:old.load_labels(data) for s,data in original.items()};m['development_references_read_for_prerequisites']=True;m['prerequisite_references_read_at']=old.now()
        for label in labels.values():add_hashes(m,label['hashes'])
        baseline=old.aggregate(dev,labels['development'],predictions)
        coverage={s:coverage_reports(original[s],labels[s],prepared[s]) for s in original}
        headroom={a:cov.oracle_headroom(dev,labels['development'],prepared['development'][a],probability,baseline) for a in ARMS}
        checks={'raw_fixed_voicing_oracle_headroom':headroom[CANDIDATE]['fixed-slots']['pass'],
            'train_grouping_loss':coverage['train'][CANDIDATE]['pass'],'development_grouping_loss':coverage['development'][CANDIDATE]['pass']}
        prerequisite={'schema':'phenometrix-expanded-pool-prerequisites.v1','status':'complete','frozen_at':old.now(),'pass':all(checks.values()),'checks':checks,
             'source_sha256':frozen,'input_cache_freeze_sha256':ev.sha(output/'input-cache-freeze.json'),'coverage':coverage,'headroom':headroom,
             'reference_sha256':{s:l['hashes'] for s,l in labels.items()},'exclusions':{s:l['excluded'] for s,l in labels.items()},
             'scope':'Consumed DEV references for fixed pass/stop candidate prerequisites only; no model-outcome selection or training feedback.'}
        save(output/'prerequisites.json',prerequisite);bind(m,output/'prerequisites.json')
        if not prerequisite['pass']:
            m.update(status='prerequisite_no_go',completed_at=old.now(),elapsed_seconds=time.monotonic()-started,prerequisites_sha256=ev.sha(output/'prerequisites.json'));save(output/'manifest.json',m);return prerequisite
        raw_targets={a:old.targets(prepared['train'][a]['fixed-slots'],labels['train']) for a in ARMS}
        schedule=training_schedule(raw_targets)
        natural_targets={a:old.targets(prepared['train'][a]['shared-candidates'],labels['train']).index_select(schedule) for a in ARMS}
        natural_batches={a:prepared['train'][a]['shared-candidates'].index_select(schedule) for a in ARMS}
        target_path=output/'training-targets.npz'
        np.savez_compressed(target_path,schedule=schedule.numpy(),**{a+'__'+k:getattr(t,k).numpy() for a,t in natural_targets.items() for k in ('acceptable','covered','nearest')})
        bind(m,target_path);m['training_target_sha256']=ev.sha(target_path);m['scheduled_rows']=len(schedule)
        m['grouped_uncovered_scheduled_rows']={a:int((~t.covered).sum()) for a,t in natural_targets.items()}
        reference_exclusions={s:l['excluded'] for s,l in labels.items()};del labels,baseline,raw_targets,label
        with CoveragePeakWorker() as worker:
            banks=assemble_training(output,m,worker,started);budget(output,started,m,worker)
        synthetic_path=output/'synthetic-targets.npz'
        np.savez_compressed(synthetic_path,**{a+'__'+k:v for a,(_,t,meta) in banks.items() for k,v in {'acceptable':t.acceptable.numpy(),'covered':t.covered.numpy(),'nearest':t.nearest.numpy(),**meta}.items()})
        bind(m,synthetic_path);m['synthetic_target_sha256']=ev.sha(synthetic_path)
        orders={f'seed{s}__epoch{e}':np.random.default_rng(s+e).permutation(len(schedule)).astype(np.int64) for s in SEEDS for e in range(12)}
        syn_orders={f'seed{s}__epoch{e}':aug.synthetic_order(s,e) for s in SEEDS for e in range(12)}
        for name,values in [('natural-schedules.npz',orders),('synthetic-schedules.npz',syn_orders)]:np.savez_compressed(output/name,**values);bind(m,output/name)
        m['synthetic_pool']={a:{'rows':len(b.features),'covered_rows':int(t.covered.sum()),'uncovered_zero_rows':int((~t.covered).sum())} for a,(b,t,_) in banks.items()}
        m['training_frozen_at']=old.now();m['status']='training_frozen';save(output/'training-freeze.json',m);bind(m,output/'training-freeze.json')
        # No development references or analytic truth are available to this fitting loop.
        device_batches={a:b.to('mps') for a,b in natural_batches.items()};device_targets={a:t.to('mps') for a,t in natural_targets.items()}
        device_synthetic={a:(b.to('mps'),t.to('mps')) for a,(b,t,_) in banks.items()};models={};began=time.monotonic()
        for seed in SEEDS:
            initial_identity=None
            for arm in ARMS:
                check_sources(frozen);budget(output,started,m);torch.manual_seed(seed);torch.mps.manual_seed(seed)
                model=modeling.ExpandedHarmonicSelector('relative','acceptable-set');model.fit_scaling(prepared['train']['original6']['shared-candidates'],split='train')
                initial=state_digest(model)
                if initial_identity is not None and initial!=initial_identity:raise ValueError('Paired initialization/scaling differs')
                initial_identity=initial;model.to('mps');optimizer=torch.optim.Adam(model.parameters(),lr=.001)
                name=key(arm,seed);fit_started=time.monotonic();losses=[];base_losses=[];aux_losses=[];exposures=[];updates=0;epoch_covered=[]
                for epoch in range(12):
                    order=orders[f'seed{seed}__epoch{epoch}'];sy=syn_orders[f'seed{seed}__epoch{epoch}'];exposures.append(aug.exposure(banks[arm][1],banks[arm][2],sy));total=base_total=aux_total=denominator=0
                    for start in range(0,len(order),4096):
                        if time.monotonic()-began>MAX_FIT_SECONDS:raise TimeoutError('30 minute total fitting ceiling exceeded')
                        ix=torch.from_numpy(order[start:start+4096].copy());batch=device_batches[arm].index_select(ix);target=device_targets[arm].index_select(ix);n=int(target.covered.sum())
                        if n==0:raise ValueError('Zero-covered batch would change update schedule')
                        sx=sy[start//4096];sb=device_synthetic[arm][0].index_select(sx);st=device_synthetic[arm][1].index_select(sx)
                        optimizer.zero_grad(set_to_none=True);loss,base,aux=aug.augmentation_loss(model,batch,target,sb,st)
                        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                        total+=float(loss.detach().cpu())*n;base_total+=float(base.detach().cpu())*n;aux_total+=float(aux.detach().cpu())*n;denominator+=n;updates+=1
                    torch.mps.synchronize();losses.append(total/denominator);base_losses.append(base_total/denominator);aux_losses.append(aux_total/denominator);epoch_covered.append(denominator)
                    allocation=torch.mps.current_allocated_memory();m['peak_observed_mps_allocated_bytes']=max(m['peak_observed_mps_allocated_bytes'],allocation)
                    if allocation>8*1024**3:raise MemoryError('8 GiB MPS allocation exceeded')
                    budget(output,started,m)
                model=model.cpu().eval();ck=output/'checkpoints'/(name+'.pt');torch.save({'arm':arm,'context_mode':'relative','candidate_capacity':6 if arm=='original6' else 7,'config':asdict(model.config),'state_dict':model.state_dict()},ck)
                if updates!=396:raise ValueError('Expected396updates')
                m['fits'][name]={'seed':seed,'arm':arm,'epoch_losses':losses,'epoch_base_losses':base_losses,'epoch_weighted_auxiliary_losses':aux_losses,
                    'epoch_natural_covered_rows':epoch_covered,'optimizer_updates':updates,'skipped_batches':0,'synthetic_exposure_by_epoch':exposures,
                    'elapsed_seconds':time.monotonic()-fit_started,'parameter_count':sum(p.numel() for p in model.parameters()),'initial_tensors_sha256':initial,'weights_sha256':ev.sha(ck)}
                models[name]=model;save(output/'manifest.json',m);print(json.dumps({'fit':name,'seconds':m['fits'][name]['elapsed_seconds'],'updates':updates}),flush=True)
                del optimizer
        m.update(status='checkpoints_frozen',checkpoints_frozen_at=old.now(),fit_elapsed_seconds=time.monotonic()-began)
        check_weights(output,m);check_sources(frozen);check_old_weights(m)
        checkpoint_freeze={'schema':'phenometrix-expanded-pool-checkpoint-freeze.v1','status':'checkpoints_frozen','frozen_at':m['checkpoints_frozen_at'],
            'source_freeze_path':str((output/'source-freeze.json').relative_to(REPO)),'source_freeze_sha256':ev.sha(output/'source-freeze.json'),
            'source_sha256':frozen,'challenge_spec_sha256':cases.challenge_spec_sha256(),'arms':list(ARMS),'seeds':list(SEEDS),'maximum_fits':6,
            'checkpoints':{n:{'path':str((output/'checkpoints'/(n+'.pt')).relative_to(REPO)),'sha256':v['weights_sha256']} for n,v in m['fits'].items()}}
        save(output/'checkpoint-freeze.json',checkpoint_freeze);save(output/'manifest.json',m);bind(m,output/'checkpoint-freeze.json')
        frozen_models={}
        for seed in SEEDS:
            row=m['historical_checkpoints'][f'balance-aug__seed{seed}'];model=modeling.ExpandedHarmonicSelector('relative','acceptable-set');model.load_state_dict(torch.load(REPO/row['path'],map_location='cpu',weights_only=True)['state_dict']);frozen_models[seed]=model.eval()
        groups=saved_analytic(output,m)
        fresh,h=cases.generate_challenge(output/'challenge',output/'checkpoint-freeze.json',ev.sha(output/'checkpoint-freeze.json'));add_hashes(m,h)
        fresh_rows={r['id']:r for r in json.loads((output/'challenge/manifest.json').read_text())['records']}
        with CoveragePeakWorker() as worker:
            for item in fresh:
                budget(output,started,m,worker);r=fresh_rows[item['id']];f,p=acquisition.expanded_feature(item['feature'],output/'challenge'/r['pcm_path'],worker)
                item.update(expanded_feature=f,new_provenance=p,original_feature_path=output/'challenge'/r['feature_path'],pcm_sha256=r['pcm_sha256'],prior_prediction_path=None)
            budget(output,started,m,worker)
        groups['challenge']=fresh;control_pairs=0;inventory={}
        for group,items in groups.items():
            records=[]
            for item in items:
                budget(output,started,m);fp=output/'analytic'/(group+'-'+item['id']+'.npz');vp=output/'analytic'/(group+'-'+item['id']+'-provenance.npz')
                np.savez_compressed(fp,**item['expanded_feature']);np.savez_compressed(vp,**item['new_provenance']);bind(m,fp);bind(m,vp)
                item['prepared']={'original6':prepare(item['feature']),'expanded7':prepare(item['expanded_feature'])};item['batch']=item['prepared']['expanded7']['shared-candidates']
                prior=npz(item['prior_prediction_path']) if item['prior_prediction_path'] else None
                item['probability']=prior['original_probability'] if prior else harmonic.original_probability(item['feature'])
                item['predictions']={key(a,s):old.infer(models[key(a,s)],item['prepared'][a]['shared-candidates'],item['probability']) for a in ARMS for s in SEEDS}
                for a in ARMS:
                    for s,model in frozen_models.items():item['predictions'][key('frozen-'+a,s)]=old.infer(model,item['prepared'][a]['shared-candidates'],item['probability'])
                if prior:
                    for s in SEEDS:
                        state,hz=item['predictions'][key('frozen-original6',s)]
                        if not np.array_equal(state,prior[key('original',s)+'__state']) or not np.array_equal(hz,prior[key('original',s)+'__hz'],equal_nan=True):raise ValueError('Historical6-pool analytic inference changed')
                        control_pairs+=len(state)
                p=output/'predictions'/(group+'-'+item['id']+'.npz');np.savez_compressed(p,original_probability=item['probability'],**{n+'__state':v[0] for n,v in item['predictions'].items()},**{n+'__hz':v[1] for n,v in item['predictions'].items()})
                record={'id':item['id'],'scenario':item['scenario'],'sample_count':int(item['feature']['pcm_sample_count']),
                    'original_feature_path':str(item['original_feature_path'].relative_to(REPO)),'original_feature_sha256':ev.sha(item['original_feature_path']),
                    'expanded_feature_path':str(fp.relative_to(REPO)),'expanded_feature_sha256':ev.sha(fp),'provenance_path':str(vp.relative_to(REPO)),'provenance_sha256':ev.sha(vp),
                    'pcm_sha256':item['pcm_sha256'],'prediction_path':str(p.relative_to(output)),'prediction_sha256':ev.sha(p)}
                for name in ('source_provenance_path','source_provenance_sha256','prior_prediction_sha256'):
                    if name in item:record[name]=item[name]
                if item['prior_prediction_path']:record['prior_prediction_path']=str(item['prior_prediction_path'].relative_to(REPO))
                records.append(record)
            inventory[group]=records;print(json.dumps({'phase':'analytic_predictions','group':group,'cases':len(items)}),flush=True)
        for a in ARMS:
            for s in SEEDS:
                predictions[key(a,s)]=old.infer(models[key(a,s)],prepared['development'][a]['shared-candidates'],probability)
                predictions[key('frozen-'+a,s)]=old.infer(frozen_models[s],prepared['development'][a]['shared-candidates'],probability)
        prior_freeze=json.loads((LAST/'prediction-freeze.json').read_text())
        for s in SEEDS:
            rel=f'predictions/original__seed{s}.npz';prior=npz(bind(m,LAST/rel,prior_freeze['prediction_sha256'][rel]));state,hz=predictions[key('frozen-original6',s)]
            if not np.array_equal(state,prior['state']) or not np.array_equal(hz,prior['hz'],equal_nan=True):raise ValueError('Historical6-pool natural inference changed')
            control_pairs+=len(state)
        for name,(state,hz) in predictions.items():np.savez_compressed(output/'predictions'/(name+'.npz'),state=state,hz=hz)
        save(output/'analytic-inventory.json',inventory);bind(m,output/'analytic-inventory.json')
        m['control_reproduction']={'state_hz_frame_pairs':control_pairs,'state_mismatches':0,'hz_mismatches':0,'all_exact':True}
        m['prediction_sha256']={str(p.relative_to(output)):ev.sha(p) for p in sorted((output/'predictions').glob('*.npz'))}
        m.update(status='predictions_frozen',predictions_frozen_at=old.now());check_sources(frozen);check_weights(output,m);budget(output,started,m)
        save(output/'prediction-freeze.json',m);save(output/'manifest.json',m)
        dl=old.load_labels(dev);m['development_references_read_for_final_scoring']=True;m['final_scoring_reference_read_at']=old.now()
        if dl['hashes']!=prerequisite['reference_sha256']['development']:raise ValueError('Final DEV references changed from prerequisites')
        add_hashes(m,dl['hashes']);natural=old.aggregate(dev,dl,predictions);scores={}
        for group,items in groups.items():
            scores[group]=harmonic.analytic_scores({i['id']:i['predictions'] for i in items},items)
            for item in items:scores[group][item['id']]['candidate_coverage']=analytic_coverage(item)
        decision=natural_decision(natural);decision['analytic']=analytic_decision(scores);decision['prerequisites']=checks
        decision['promising_for_next_study']=decision['pass'] and all(checks.values()) and all(v for k,v in decision['analytic'].items() if k.endswith('_pass'))
        result={'schema':'phenometrix-expanded-pool-metrics.v1','scope':'Six matched local fits; consumed natural development/62regressions;12prospective noiseless configurations',
            'natural':natural,'coverage':coverage,'headroom':headroom,**scores,'decision':decision,'exclusions':reference_exclusions}
        save(output/'metrics.json',result)
        for p,h in m['input_sha256'].items():
            if ev.sha(REPO/p)!=h:raise ValueError('Bound input changed')
        for p,h in m['prediction_sha256'].items():
            if ev.sha(output/p)!=h:raise ValueError('Frozen prediction changed')
        check_sources(frozen);check_weights(output,m);check_old_weights(m);budget(output,started,m)
        m.update(status='complete',completed_at=old.now(),elapsed_seconds=time.monotonic()-started,metrics_sha256=ev.sha(output/'metrics.json'),output_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))
        save(output/'manifest.json',m);print(json.dumps(decision,indent=2,default=aug.json_scalar),flush=True);return result
    except BaseException as error:
        m.update(status='failed',failed_at=old.now(),failure=repr(error));save(output/'manifest.json',m);raise

def timeout_signal(signum,frame):raise TimeoutError('75 minute experiment deadline reached')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    signal.signal(signal.SIGALRM,timeout_signal);signal.alarm(MAX_SECONDS)
    try:run(args.output)
    finally:signal.alarm(0)
