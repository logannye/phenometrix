"""One prospective local exposure x periodicity study; no adaptive reruns."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import shutil
import signal
import time
import numpy as np
import torch
from . import selector_experiment as old, augmentation_experiment as aug
from . import expanded_pool_experiment as prior, expanded_selector_model as base
from . import candidate_coverage_experiment as cov, harmonic_selector_experiment as harmonic
from . import evidence_cases as cases, evidence_feasibility as feasibility
from . import evidence_inputs as inputs, evidence_selector_model as modeling
from . import expanded_candidate_features as acquisition, evaluate as ev
from .candidate_coverage_features import CoveragePeakWorker
from .selector_model import selector_loss, predict_pitch_hz

ROOT,RUNS,REPO=old.ROOT,old.RUNS,old.REPO
PLAN=ROOT/'EVIDENCE_EXPERIMENT_PLAN.md'
PRIOR=RUNS/'expanded-pool-v1'
ARMS=cases.ARMS;SEEDS=cases.SEEDS;CANDIDATE='targeted__on'
DEADLINE='2026-10-23T12:18:58Z';FIT_DEADLINE='2026-10-23T09:48:58Z'
MAX_SECONDS=3*3600;MAX_FIT_SECONDS=3600;MAX_BYTES=8*1024**3
GROUPS={**{('consumed_expansion' if k=='challenge' else k):v for k,v in prior.GROUPS.items()},'challenge':12}
save=aug.save;npz=cov.npz;bind=cov.bind;key=prior.key

def sources():
    paths=[PLAN,ROOT/'OVERNIGHT_DEVELOPMENT_PLAN.md',Path(__file__),ROOT/'periodicity_evidence.ts',ROOT/'periodicity_evidence_worker.ts']
    return prior.sources()|cases.source_hashes()|feasibility.source_hashes()|inputs.source_hashes()|modeling.source_hashes()|{str(p.relative_to(REPO)):ev.sha(p) for p in paths}

def check_sources(expected):
    if sources()!=expected:raise ValueError('Frozen evidence study source changed')

def remaining(deadline=DEADLINE):
    return (datetime.fromisoformat(deadline.replace('Z','+00:00'))-datetime.now(timezone.utc)).total_seconds()

def budget(output,started,m,worker=None):
    if time.monotonic()-started>MAX_SECONDS or remaining()<=0:raise TimeoutError('Scientific/session deadline reached')
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    m['peak_observed_python_rss_bytes']=max(m.get('peak_observed_python_rss_bytes',0),rss)
    if rss>16*1024**3:raise MemoryError('16GiB observed Python RSS exceeded')
    if shutil.disk_usage(ROOT).free<16*1024**3:raise RuntimeError('16GiB free disk prerequisite failed')
    if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>MAX_BYTES:raise RuntimeError('8GiB study artifacts exceeded')
    if worker is not None:
        m['peak_observed_node_rss_bytes']=max(m.get('peak_observed_node_rss_bytes',0),worker.peak_rss_bytes)
        worker.timeout_seconds=min(120.,max(.001,remaining()),max(.001,MAX_SECONDS-(time.monotonic()-started)))
    if m.get('peak_observed_node_rss_bytes',0)>2*1024**3:raise MemoryError('2GiB Node RSS exceeded')
    if torch.backends.mps.is_available():
        allocation=torch.mps.current_allocated_memory();m['peak_observed_mps_allocated_bytes']=max(m.get('peak_observed_mps_allocated_bytes',0),allocation)
        if allocation>8*1024**3:raise MemoryError('8GiB MPS allocated memory exceeded')

def add_hashes(m,hashes):
    for p,h in hashes.items():bind(m,p if Path(p).is_absolute() else REPO/p,h)

def infer(model,batch,evidence,probability):
    model=model.cpu().eval();pieces=[]
    if evidence.shape!=(len(batch.features),7,3) or len(probability)!=len(batch.features):raise ValueError('Inference alignment differs')
    with torch.inference_mode():
        for start in range(0,len(batch.features),4096):
            ix=torch.arange(start,min(start+4096,len(batch.features)));b=batch.index_select(ix)
            pieces.append(predict_pitch_hz(model(b,evidence[ix]),b).numpy())
    hz=np.concatenate(pieces);state=np.where(probability<.5,0,np.where(np.isfinite(hz),1,2)).astype(np.uint8);hz[state!=1]=np.nan
    return state,hz

def loss(model,batch,target,evidence,sb,st,se):
    natural=selector_loss(model(batch,evidence),batch.candidate_mask,target,objective='acceptable-set')
    auxiliary=.25*selector_loss(model(sb,se),sb.candidate_mask,st,objective='acceptable-set')
    return natural+auxiliary,natural,auxiliary

def diagnostics(model,batch,target,evidence,strata):
    """TRAIN-only descriptions; finite margin only where both classes exist."""
    logits=[];model=model.cpu().eval()
    with torch.inference_mode():
        for start in range(0,len(batch.features),4096):
            ix=torch.arange(start,min(start+4096,len(batch.features)));logits.append(model(batch.index_select(ix),evidence[ix]).numpy())
    logits=np.concatenate(logits);mask=batch.candidate_mask.numpy();acceptable=target.acceptable.numpy();covered=target.covered.numpy()
    wrong=mask&~acceptable;choice=logits.argmax(1);correct=acceptable[np.arange(len(logits)),choice]&covered
    both_available=covered&wrong.any(1);margins=np.zeros(len(logits),np.float64)
    margins[both_available]=np.max(np.where(acceptable[both_available],logits[both_available],-np.inf),axis=1)-np.max(np.where(wrong[both_available],logits[both_available],-np.inf),axis=1)
    result={}
    for name,selected in strata.items():
        selected=np.asarray(selected,bool);ix=np.flatnonzero(selected);eligible=selected&covered;both=eligible&wrong.any(1)
        conditional=selector_loss(torch.from_numpy(logits[ix]),batch.candidate_mask[ix],target.index_select(torch.from_numpy(ix)),objective='acceptable-set')
        result[name]={'rows':int(selected.sum()),'covered_rows':int(eligible.sum()),'selected_correct_rows':int((selected&correct).sum()),
            'success_all_rows':float(correct[selected].mean()) if len(ix) else None,'conditional_loss':float(conditional),
            'both_correct_and_wrong_rows':int(both.sum()),'mean_best_acceptable_minus_best_wrong':float(margins[both].mean()) if both.any() else None}
    return result

def natural_decision(natural):
    rows=[natural[key(CANDIDATE,s)]['macro'] for s in SEEDS];mean=float(np.mean([v['joint_error'] for v in rows]));success=float(np.mean([v['reference_voiced_success_rate'] for v in rows]))
    gains={b:natural[b]['macro']['joint_error']-mean for b in ('original_mlp','original_hybrid')}
    excess={c:float(np.mean([natural[key(CANDIDATE,s)]['per_condition'][c]['joint_error'] for s in SEEDS]))-natural['original_mlp']['per_condition'][c]['joint_error'] for c in old.CONDITIONS}
    checks={'mean_improves_both_by_half_point':all(v>=.005 for v in gains.values()),'each_seed_improves_original_mlp':all(v['joint_error']<natural['original_mlp']['macro']['joint_error'] for v in rows),'no_condition_worsens_over_one_point':all(v<=.01 for v in excess.values()),'voiced_success_not_lower':success>=natural['original_mlp']['macro']['reference_voiced_success_rate']}
    means={a:float(np.mean([natural[key(a,s)]['macro']['joint_error'] for s in SEEDS])) for a in ARMS}
    return {'predesignated_candidate':CANDIDATE,'pass':all(checks.values()),'checks':checks,'mean_joint_error':mean,'mean_voiced_success':success,'joint_error_improvement_over_frozen':gains,'condition_error_excess_over_mlp':excess,'arm_mean_joint_error':means,
        'factorial_error_differences':{'targeted_minus_existing_off':means['targeted__off']-means['existing__off'],'evidence_on_minus_off_existing':means['existing__on']-means['existing__off'],'evidence_on_minus_off_targeted':means['targeted__on']-means['targeted__off'],'interaction':means['targeted__on']-means['targeted__off']-means['existing__on']+means['existing__off']}}

def analytic_decision(groups):
    if set(groups)!=set(GROUPS) or any(len(groups[g])!=n for g,n in GROUPS.items()):raise ValueError('Incomplete86case inventory')
    from . import harmonic_challenge,subharmonic_challenge,augmentation_cases,coverage_cases,expanded_cases
    expected={'regression':old.SYNTHETIC_IDS,**{g:{r['id'] for r in module.challenge_spec()['records']} for g,module in [('consumed_harmonic',harmonic_challenge),('consumed_subharmonic',subharmonic_challenge),('consumed_augmentation',augmentation_cases),('consumed_coverage',coverage_cases),('consumed_expansion',expanded_cases),('challenge',cases)]}}
    if any(set(groups[g])!=ids for g,ids in expected.items()):raise ValueError('Analytic configuration identity differs')
    names=[key(CANDIDATE,s) for s in SEEDS];gates={n:{} for n in names}
    for n in names:
        for group,records in groups.items():
            for cid,row in records.items():
                if group=='regression' and row['kind'] not in ('constant','observation-change'):continue
                score=row['methods'][n]['all'] if group=='regression' and row['kind']=='constant' else row['methods'][n]['regions']['after-observation-change']
                gates[n][group+'/'+cid]=score['success_within_50c_all_frames']>=.95
        if len(gates[n])!=78:raise ValueError('Expected78 mandatory case gates per seed')
    return {'by_seed_case':gates,'passed_by_seed':{n:sum(v.values()) for n,v in gates.items()},'required_per_seed':78,'pass':all(all(v.values()) for v in gates.values())}

def save_evidence(output,m,group,item):
    p=output/'evidence'/group/(item['id']+'.npz');p.parent.mkdir(parents=True,exist_ok=True)
    evidence=item['periodicity_evidence'];np.savez_compressed(p,**evidence);bind(m,p)
    return str(p.relative_to(REPO)),ev.sha(p)

def build_bank(items,output,m,group,guard,*,all_frames=False):
    raw={k:[] for k in ('features','candidate_hz','candidate_mask')};metas={k:[] for k in ('reference_hz','scenario_index','reference_sample','availability_sample','region')};channels=[];records=[]
    for index,item in enumerate(items):
        guard();e=inputs.evidence_for_item(item,item.get('pcm_path'),item['pcm_sha256']);ep,eh=save_evidence(output,m,group,item)
        timing=item['timing'];ix=np.arange(len(e)) if all_frames else np.flatnonzero(timing.common_eligible)
        if not all_frames and len(ix)!=191:raise ValueError('Synthetic common extent changed')
        for k in raw:raw[k].append(item['expanded_feature'][k][ix])
        channels.append(e.numpy()[ix]);metas['reference_hz'].append(np.full(len(ix),item['scenario']['initialHz']));metas['scenario_index'].append(np.full(len(ix),index,np.int64));metas['reference_sample'].append(timing.reference_sample[ix]);metas['availability_sample'].append(timing.availability_sample[ix]);metas['region'].append(np.array([cases.challenge_region_at(item['scenario'],int(end)) for end in timing.availability_sample[ix]]))
        records.append({'id':item['id'],'evidence_path':ep,'evidence_sha256':eh,'pcm_sha256':item['pcm_sha256']})
    batch=prior.prepare({k:np.concatenate(v) for k,v in raw.items()})['shared-candidates'];meta={k:np.concatenate(v) for k,v in metas.items()};evidence=torch.from_numpy(np.concatenate(channels))
    target=old.targets(batch,{'hz':meta['reference_hz'],'voiced':np.ones(len(batch.features),bool),'valid':np.ones(len(batch.features),bool)})
    save(output/'evidence'/group/'manifest.json',{'records':records,'source_sha256':m['source_sha256'],'reference_content_read':False});bind(m,output/'evidence'/group/'manifest.json')
    return batch,target,meta,evidence

def check_checkpoints(output,m):
    if set(m['fits'])!={key(a,s) for s in SEEDS for a in ARMS}:raise ValueError('Incomplete12-fit inventory')
    for name,row in m['fits'].items():
        if ev.sha(output/'checkpoints'/(name+'.pt'))!=row['weights_sha256']:raise ValueError('Final checkpoint changed')


def fit_all(output,m,prepared,natural_target,schedule,banks,diagnostic_strata,orders,synthetic_orders,guard):
    if remaining(FIT_DEADLINE)<=0:raise TimeoutError('New fitting deadline reached')
    natural=prepared['expanded7']['shared-candidates'].index_select(schedule);nt=natural_target.index_select(schedule);ne=torch.from_numpy(prepared['evidence'])[schedule]
    db,dt,de=natural.to('mps'),nt.to('mps'),ne.to('mps')
    device_banks={a:(b.to('mps'),t.to('mps'),e.to('mps')) for a,(b,t,meta,e) in banks.items()};models={};cumulative=0.;diag_b,diag_t,_,diag_e=banks['targeted']
    for seed in SEEDS:
        initialization=None
        for arm in ARMS:
            guard();check_sources(m['source_sha256'])
            if remaining(FIT_DEADLINE)<=0:raise TimeoutError('No new fits after09:48:58UTC')
            torch.manual_seed(seed);torch.mps.manual_seed(seed);model=modeling.EvidenceHarmonicSelector('periodicity' if arm.endswith('__on') else 'off');model.fit_scaling(prepared['original6']['shared-candidates'],split='train')
            digest=prior.state_digest(model)
            if initialization is not None and digest!=initialization:raise ValueError('Paired initialization/scaling mismatch')
            initialization=digest;name=key(arm,seed)
            if name in m.get('fit_attempts',{}):raise ValueError('A fit attempt may not be repeated')
            before=diagnostics(model,diag_b,diag_t,diag_e,diagnostic_strata)
            m.setdefault('fit_attempts',{})[name]={'status':'started','started_at':old.now(),'initial_tensors_sha256':digest};save(output/'manifest.json',m)
            model=model.to('mps').train();optimizer=torch.optim.Adam(model.parameters(),lr=.001);began=time.monotonic();losses=[];base_losses=[];aux_losses=[];denominators=[];exposures=[];updates=0;exposure=arm.split('__')[0]
            sb,st,se=device_banks[exposure]
            for epoch in range(12):
                order=orders[f'seed{seed}__epoch{epoch}'];sy=synthetic_orders[exposure][f'seed{seed}__epoch{epoch}'];total=base_total=aux_total=denom=0;synthetic_denom=0
                for start in range(0,len(order),4096):
                    if cumulative+time.monotonic()-began>MAX_FIT_SECONDS or remaining()<=0:raise TimeoutError('Fit/session ceiling reached')
                    ix=torch.from_numpy(order[start:start+4096].copy());sx=torch.from_numpy(sy[start//4096].copy());b=db.index_select(ix);t=dt.index_select(ix);e=de[ix];sbt=sb.index_select(sx);stt=st.index_select(sx);set_=se[sx];n=int(t.covered.sum());sn=int(stt.covered.sum())
                    if n==0:raise ValueError('Zero covered natural batch would change schedule')
                    optimizer.zero_grad(set_to_none=True);objective,base_loss,aux=loss(model,b,t,e,sbt,stt,set_)
                    if not torch.isfinite(objective):raise ValueError('Nonfinite training loss')
                    objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                    total+=float(objective.detach().cpu())*n;base_total+=float(base_loss.detach().cpu())*n;aux_total+=float(aux.detach().cpu())*n;denom+=n;synthetic_denom+=sn;updates+=1
                torch.mps.synchronize();losses.append(total/denom);base_losses.append(base_total/denom);aux_losses.append(aux_total/denom);denominators.append({'natural_covered':denom,'synthetic_covered':synthetic_denom});guard()
                flat=sy.ravel();meta=banks[exposure][2];exposures.append({'draws':int(len(flat)),'unique_rows':int(len(np.unique(flat))),'duplicate_draws':int(len(flat)-len(np.unique(flat))),'covered_rows':int(banks[exposure][1].covered[flat].sum()),'per_case':np.bincount(meta['scenario_index'][flat],minlength=96 if exposure=='existing' else 512).tolist(),'per_region':{str(r):int((meta['region'][flat]==r).sum()) for r in np.unique(meta['region'])}})
            elapsed=time.monotonic()-began;cumulative+=elapsed
            if updates!=396:raise ValueError('Expected396 updates')
            model=model.cpu().eval();ck=output/'checkpoints'/(name+'.pt');torch.save({'arm':arm,'evidence_mode':model.evidence_mode,'config':asdict(model.config),'state_dict':model.state_dict()},ck)
            after=diagnostics(model,diag_b,diag_t,diag_e,diagnostic_strata)
            m['fits'][name]={'seed':seed,'arm':arm,'parameter_count':3012,'weights_sha256':ev.sha(ck),'initial_tensors_sha256':digest,'epoch_losses':losses,'epoch_base_losses':base_losses,'epoch_weighted_auxiliary_losses':aux_losses,'loss_denominators':denominators,'synthetic_exposure_by_epoch':exposures,'training_diagnostics_before':before,'training_diagnostics_after':after,'optimizer_updates':updates,'elapsed_seconds':elapsed,'evidence_residual_weights':model.evidence_head.weight.detach().numpy().tolist()}
            m['fit_attempts'][name].update(status='complete',completed_at=old.now());m['fit_elapsed_seconds']=cumulative;models[name]=model;save(output/'manifest.json',m);print(json.dumps({'fit':name,'seconds':elapsed,'updates':updates}),flush=True);del optimizer
    return models


def targeted_items(preparation,m):
    result=[]
    for r in preparation['manifest']['records']:
        fp=bind(m,REPO/r['original_feature_path'],r['original_feature_sha256']);ep=bind(m,REPO/r['expanded_feature_path'],r['expanded_feature_sha256']);pp=bind(m,REPO/r['pcm_path'],r['pcm_sha256']);f=npz(fp)
        result.append({'id':r['id'],'scenario':r['scenario'],'feature':f,'expanded_feature':npz(ep),'timing':ev.validate_feature(f,{'sample_count':int(f['pcm_sample_count'])}),'pcm_path':pp,'pcm_sha256':r['pcm_sha256'],'original_feature_path':fp})
    if len(result)!=512:raise ValueError('Expected512 TRAIN configurations')
    return result


def train_and_evaluate(output,m,preparation,started):
    guard=lambda:budget(output,started,m)
    original={};expanded={};prepared={}
    for split in ('train','development'):
        original[split]=old.load_inputs(split);add_hashes(m,original[split]['hashes']);expanded[split]=inputs.cache_natural(original[split],output,m,guard)
        prepared[split]={'original6':prior.prepare(original[split]['raw']),'expanded7':prior.prepare(expanded[split]['raw']),'evidence':expanded[split]['evidence']}
    dev=original['development'];probability,predictions,h=old.read_natural_baselines(dev);add_hashes(m,h)
    np.savez_compressed(output/'probabilities-development.npz',original_probability=probability);bind(m,output/'probabilities-development.npz')
    save(output/'input-cache-freeze.json',{'schema':'phenometrix-evidence-input-cache-freeze.v1','status':'input_caches_frozen','frozen_at':old.now(),'source_sha256':m['source_sha256'],'input_sha256':m['input_sha256'],'reference_content_read':False});bind(m,output/'input-cache-freeze.json')
    labels={s:old.load_labels(d) for s,d in original.items()};m['development_references_read_for_prerequisites']=True;m['prerequisite_reference_read_at']=old.now()
    for label in labels.values():add_hashes(m,label['hashes'])
    baseline=old.aggregate(dev,labels['development'],predictions)
    coverage={s:prior.coverage_reports(original[s],labels[s],{k:prepared[s][k] for k in prior.ARMS}) for s in original}
    headroom=cov.oracle_headroom(dev,labels['development'],prepared['development']['expanded7'],probability,baseline)
    checks={'train_support':preparation['feasibility']['pass_result'],'raw_fixed_voicing_oracle_headroom':headroom['fixed-slots']['pass'],'train_grouping_loss':coverage['train']['expanded7']['pass'],'development_grouping_loss':coverage['development']['expanded7']['pass']}
    prerequisite={'schema':'phenometrix-evidence-prerequisites.v1','pass':all(checks.values()),'checks':checks,'coverage':coverage,'headroom':headroom,'reference_sha256':{s:l['hashes'] for s,l in labels.items()},'exclusions':{s:l['excluded'] for s,l in labels.items()},'frozen_at':old.now()}
    save(output/'prerequisites.json',prerequisite);bind(m,output/'prerequisites.json')
    if not prerequisite['pass']:
        m.update(status='prerequisite_no_go',completed_at=old.now(),elapsed_seconds=time.monotonic()-started);save(output/'manifest.json',m);return prerequisite
    raw_targets={a:old.targets(prepared['train'][a]['fixed-slots'],labels['train']) for a in prior.ARMS};schedule=prior.training_schedule(raw_targets)
    natural_target=old.targets(prepared['train']['expanded7']['shared-candidates'],labels['train'])
    np.savez_compressed(output/'training-targets.npz',schedule=schedule.numpy(),**{k:getattr(natural_target,k).numpy() for k in ('acceptable','covered','nearest')});bind(m,output/'training-targets.npz')
    reference_exclusions={s:l['excluded'] for s,l in labels.items()};del labels,baseline,raw_targets
    banks={'existing':build_bank(inputs.load_existing_training(m),output,m,'existing-training',guard),'targeted':build_bank(targeted_items(preparation,m),output,m,'targeted-training',guard,all_frames=True)}
    table=preparation['arrays'];targeted_bank=banks['targeted']
    if not np.array_equal(targeted_bank[2]['scenario_index'],table['case_index']) or not np.array_equal(targeted_bank[2]['reference_hz'],table['truth_hz']):raise ValueError('Targeted bank/full-table alignment differs')
    if not np.array_equal(targeted_bank[0].candidate_hz.numpy(),table['expanded_group_hz'],equal_nan=True):raise ValueError('Targeted grouping differs from feasibility freeze')
    orders={f'seed{s}__epoch{e}':np.random.default_rng(s+e).permutation(len(schedule)).astype(np.int64) for s in SEEDS for e in range(12)}
    synthetic_orders={'existing':{f'seed{s}__epoch{e}':aug.synthetic_order(s,e) for s in SEEDS for e in range(12)},'targeted':{f'seed{s}__epoch{e}':preparation['schedules'][f'seed{s}__epoch{e}__indices'] for s in SEEDS for e in range(12)}}
    for name,arrays in [('natural-schedules.npz',orders),('existing-synthetic-schedules.npz',synthetic_orders['existing']),('targeted-synthetic-schedules.npz',synthetic_orders['targeted'])]:np.savez_compressed(output/name,**arrays);bind(m,output/name)
    np.savez_compressed(output/'synthetic-targets.npz',**{a+'__'+k:getattr(t,k).numpy() for a,(b,t,meta,e) in banks.items() for k in ('acceptable','covered','nearest')});bind(m,output/'synthetic-targets.npz')
    m.update(status='training_frozen',training_frozen_at=old.now(),scheduled_rows=len(schedule),training_targets_sha256=ev.sha(output/'training-targets.npz'),synthetic_targets_sha256=ev.sha(output/'synthetic-targets.npz'))
    save(output/'training-freeze.json',m);bind(m,output/'training-freeze.json');check_sources(m['source_sha256'])
    models=fit_all(output,m,prepared['train'],natural_target,schedule,banks,preparation['strata'],orders,synthetic_orders,guard)
    check_checkpoints(output,m);check_sources(m['source_sha256']);m.update(status='checkpoints_frozen',checkpoints_frozen_at=old.now())
    freeze={'schema':'phenometrix-evidence-checkpoint-freeze.v1','status':'checkpoints_frozen','frozen_at':m['checkpoints_frozen_at'],'source_freeze_path':str((output/'source-freeze.json').relative_to(REPO)),'source_freeze_sha256':ev.sha(output/'source-freeze.json'),'source_sha256':m['source_sha256'],'training_spec_sha256':cases.training_spec_sha256(),'challenge_spec_sha256':cases.challenge_spec_sha256(),'arms':list(ARMS),'seeds':list(SEEDS),'maximum_fits':12,'deadline_utc':DEADLINE,'checkpoints':{n:{'path':str((output/'checkpoints'/(n+'.pt')).relative_to(REPO)),'sha256':r['weights_sha256']} for n,r in m['fits'].items()}}
    save(output/'checkpoint-freeze.json',freeze);bind(m,output/'checkpoint-freeze.json');save(output/'manifest.json',m)
    # Clear TRAIN-only labels and evidence before evaluation input generation.
    del banks,table,targeted_bank,natural_target,preparation
    groups=inputs.load_historical_analytic(m)
    fresh,h=cases.generate_challenge(output/'challenge',output/'checkpoint-freeze.json',ev.sha(output/'checkpoint-freeze.json'));add_hashes(m,h)
    fresh_records={r['id']:r for r in json.loads((output/'challenge/manifest.json').read_text())['records']}
    with CoveragePeakWorker() as worker:
        for item in fresh:
            budget(output,started,m,worker);r=fresh_records[item['id']];pp=bind(m,output/'challenge'/r['pcm_path'],r['pcm_sha256']);f,p=acquisition.expanded_feature(item['feature'],pp,worker)
            item.update(expanded_feature=f,new_provenance=p,original_feature_path=output/'challenge'/r['feature_path'],pcm_path=pp,pcm_sha256=r['pcm_sha256'],prior_prediction_path=None)
        budget(output,started,m,worker)
    groups['challenge']=fresh
    if {g:len(v) for g,v in groups.items()}!=GROUPS:raise ValueError('Analytic input inventory differs')
    frozen={}
    for s in SEEDS:
        ck=bind(m,PRIOR/'checkpoints'/(key('expanded7',s)+'.pt'));expected=m['prior_manifest']['fits'][key('expanded7',s)]['weights_sha256']
        if ev.sha(ck)!=expected:raise ValueError('Historical expanded checkpoint changed')
        model=base.ExpandedHarmonicSelector('relative','acceptable-set');model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True)['state_dict']);frozen[s]=model.eval()
    inventory={};control_pairs=0
    for group,items in groups.items():
        records=[]
        for item in items:
            guard();evidence=inputs.evidence_for_item(item,item.get('pcm_path'),item['pcm_sha256']);ep,eh=save_evidence(output,m,group,item);b=prior.prepare(item['expanded_feature'])['shared-candidates'];item['prepared']={a:prior.prepare(item['feature'] if a=='original6' else item['expanded_feature']) for a in prior.ARMS};item['batch']=b
            previous=npz(item['prior_prediction_path']) if item.get('prior_prediction_path') else None
            item['probability']=previous['original_probability'] if previous else harmonic.original_probability(item['feature'])
            item['predictions']={n:infer(model,b,evidence,item['probability']) for n,model in models.items()}
            for s,model in frozen.items():
                state,hz=old.infer(model,b,item['probability']);item['predictions'][key('frozen-expanded7',s)]=(state,hz)
                if previous is not None:
                    if not np.array_equal(state,previous[key('expanded7',s)+'__state']) or not np.array_equal(hz,previous[key('expanded7',s)+'__hz'],equal_nan=True):raise ValueError('Frozen expanded analytic inference changed')
                    control_pairs+=len(state)
            fp=output/'analytic'/(group+'-'+item['id']+'.npz');np.savez_compressed(fp,**item['expanded_feature']);bind(m,fp)
            p=output/'predictions'/(group+'-'+item['id']+'.npz');np.savez_compressed(p,original_probability=item['probability'],**{n+'__state':v[0] for n,v in item['predictions'].items()},**{n+'__hz':v[1] for n,v in item['predictions'].items()})
            records.append({'id':item['id'],'scenario':item['scenario'],'sample_count':int(item['feature']['pcm_sample_count']),'original_feature_path':str(item['original_feature_path'].relative_to(REPO)),'original_feature_sha256':ev.sha(item['original_feature_path']),'expanded_feature_path':str(fp.relative_to(REPO)),'expanded_feature_sha256':ev.sha(fp),'evidence_path':ep,'evidence_sha256':eh,'pcm_sha256':item['pcm_sha256'],'prediction_path':str(p.relative_to(output)),'prediction_sha256':ev.sha(p),'prior_prediction_path':str(item['prior_prediction_path'].relative_to(REPO)) if item.get('prior_prediction_path') else None})
        inventory[group]=records;print(json.dumps({'phase':'evidence_analytic_predictions','group':group,'cases':len(items)}),flush=True)
    db=prepared['development']['expanded7']['shared-candidates'];de=torch.from_numpy(prepared['development']['evidence'])
    for n,model in models.items():guard();predictions[n]=infer(model,db,de,probability)
    for s,model in frozen.items():
        state,hz=old.infer(model,db,probability);predictions[key('frozen-expanded7',s)]=(state,hz);p=bind(m,PRIOR/'predictions'/(key('expanded7',s)+'.npz'));previous=npz(p)
        if not np.array_equal(state,previous['state']) or not np.array_equal(hz,previous['hz'],equal_nan=True):raise ValueError('Frozen expanded natural inference changed')
        control_pairs+=len(state)
    for n,(state,hz) in predictions.items():np.savez_compressed(output/'predictions'/(n+'.npz'),state=state,hz=hz)
    save(output/'analytic-inventory.json',inventory);bind(m,output/'analytic-inventory.json')
    m['control_reproduction']={'state_hz_frame_pairs':control_pairs,'mismatches':0,'all_exact':True}
    m['prediction_sha256']={str(p.relative_to(output)):ev.sha(p) for p in sorted((output/'predictions').glob('*.npz'))};m.update(status='predictions_frozen',predictions_frozen_at=old.now());check_sources(m['source_sha256']);check_checkpoints(output,m);guard();save(output/'prediction-freeze.json',m);save(output/'manifest.json',m)
    dl=old.load_labels(dev);m['development_references_read_for_final_scoring']=True;m['final_scoring_reference_read_at']=old.now()
    if dl['hashes']!=prerequisite['reference_sha256']['development']:raise ValueError('Final DEV references changed')
    add_hashes(m,dl['hashes']);natural=old.aggregate(dev,dl,predictions);scores={}
    for g,items in groups.items():
        scores[g]=harmonic.analytic_scores({i['id']:i['predictions'] for i in items},items)
        for item in items:scores[g][item['id']]['candidate_coverage']=prior.analytic_coverage(item)
    decision=natural_decision(natural);decision['analytic']=analytic_decision(scores);decision['prerequisites']=checks;decision['promising_for_next_study']=decision['pass'] and decision['analytic']['pass'] and all(checks.values())
    result={'schema':'phenometrix-evidence-metrics.v1','natural':natural,'coverage':coverage,'headroom':headroom,**scores,'decision':decision,'exclusions':reference_exclusions,'scope':'Twelve fixed local fits; consumed natural DEV and74 regressions;12 fresh noiseless configurations'}
    save(output/'metrics.json',result)
    for p,h in m['input_sha256'].items():
        if ev.sha(REPO/p)!=h:raise ValueError('Bound input changed')
    for p,h in m['prediction_sha256'].items():
        if ev.sha(output/p)!=h:raise ValueError('Frozen prediction changed')
    check_sources(m['source_sha256']);check_checkpoints(output,m);guard();m.update(status='complete',completed_at=old.now(),elapsed_seconds=time.monotonic()-started,metrics_sha256=ev.sha(output/'metrics.json'),output_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()));save(output/'manifest.json',m);print(json.dumps(decision,indent=2),flush=True);return result


def run(output,preflight):
    output=output.resolve();started=time.monotonic()
    if not output.is_relative_to(RUNS) or output.exists():raise ValueError('Fresh ignored output path required; no silent resumption')
    if remaining(FIT_DEADLINE)<=0:raise TimeoutError('Too late to start scientific execution')
    if not torch.backends.mps.is_available():raise RuntimeError('Local MPS is required')
    frozen=sources();receipt=json.loads(preflight.read_text())
    if receipt.get('pass') is not True or receipt.get('source_sha256')!=frozen:raise ValueError('Matching passing preflight required')
    old_manifest=json.loads((PRIOR/'manifest.json').read_text())
    if old_manifest['status']!='complete' or ev.sha(PRIOR/'manifest.json')!=inputs.PRIOR_MANIFEST_SHA256:raise ValueError('Historical expanded run identity differs')
    prior.check_sources(old_manifest['source_sha256']);prior.check_weights(PRIOR,old_manifest)
    output.mkdir()
    for name in ('source-snapshot','checkpoints','predictions','analytic'):(output/name).mkdir()
    for p,h in frozen.items():
        q=output/'source-snapshot'/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((REPO/p).read_bytes())
        if ev.sha(q)!=h:raise ValueError('Source snapshot changed')
    m={'schema':'phenometrix-evidence-experiment.v1','status':'sources_frozen','started_at':old.now(),'sources_frozen_at':old.now(),'source_sha256':frozen,'training_spec_sha256':cases.training_spec_sha256(),'challenge_spec_sha256':cases.challenge_spec_sha256(),'arms':list(ARMS),'seeds':list(SEEDS),'maximum_fits':12,'deadline_utc':DEADLINE,'predesignated_candidate':CANDIDATE,'fits':{},'input_sha256':{},'historical_input_sha256':old_manifest['input_sha256'],'prior_manifest':old_manifest,'device':'mps','torch':torch.__version__,'reserved_content_read':False,'development_references_read_for_prerequisites':False,'development_references_read_for_final_scoring':False,'peak_observed_mps_allocated_bytes':0,'peak_observed_python_rss_bytes':0,'peak_observed_node_rss_bytes':0}
    bind(m,preflight);bind(m,PRIOR/'manifest.json',inputs.PRIOR_MANIFEST_SHA256)
    save(output/'source-freeze.json',{**m,'schema':'phenometrix-evidence-source-freeze.v1'});save(output/'manifest.json',m)
    try:
        torch.set_num_threads(1);budget(output,started,m)
        prepared=feasibility.prepare(output/'feasibility',output/'source-freeze.json',ev.sha(output/'source-freeze.json'));add_hashes(m,prepared['hashes']);budget(output,started,m);check_sources(frozen)
        m['peak_observed_node_rss_bytes']=max(m['peak_observed_node_rss_bytes'],prepared['manifest']['peak_observed_node_rss_bytes']);m['train_feasibility']=prepared['feasibility'];m['feasibility_manifest_sha256']=ev.sha(output/'feasibility/manifest.json');save(output/'manifest.json',m)
        if not prepared['feasibility']['pass_result']:
            m.update(status='feasibility_no_go',completed_at=old.now(),elapsed_seconds=time.monotonic()-started,fit_elapsed_seconds=0,output_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()));save(output/'manifest.json',m)
            print(json.dumps({'scientific_status':'feasibility_no_go','fits':0,'fresh_evaluation_generated':False,'support':prepared['feasibility']},indent=2),flush=True);return prepared['feasibility']
        return train_and_evaluate(output,m,prepared,started)
    except BaseException as error:
        m.update(status='failed',failed_at=old.now(),failure=repr(error),elapsed_seconds=time.monotonic()-started);save(output/'manifest.json',m);raise


def timeout_signal(signum,frame):raise TimeoutError('Bounded local experiment timeout')
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--preflight',type=Path,required=True);args=parser.parse_args()
    signal.signal(signal.SIGALRM,timeout_signal);signal.alarm(max(1,min(MAX_SECONDS,int(remaining()))))
    try:run(args.output,args.preflight)
    finally:signal.alarm(0)
