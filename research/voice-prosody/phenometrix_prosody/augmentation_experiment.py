"""One frozen local harmonic-balance augmentation study; historical evidence stays exact."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from . import selector_experiment as old
from . import harmonic_selector_experiment as prior
from . import evaluate as ev
from .harmonic_selector_model import HarmonicSelector
from .selector_model import prepare_selector_batch, selector_loss
from . import subharmonic_experiment as sub
from . import augmentation_cases as cases

ROOT,RUNS,REPO=old.ROOT,old.RUNS,old.REPO
PLAN=ROOT/'AUGMENTATION_EXPERIMENT_PLAN.md'
PRIOR=RUNS/'harmonic-context-v1'
PREREQUISITES=RUNS/'selector-prerequisites-v1/prerequisites.json'
SEEDS=old.SEEDS
ARMS=('natural-only','stationary-aug','balance-aug')
CANDIDATE='balance-aug'
SYNTHETIC_WEIGHT=.25
SYNTHETIC_BATCH=512
LAST=RUNS/'subharmonic-objective-v1'
LAST_MANIFEST_SHA='d3600c00e099e4aa4f426dfbdb46399d968b13189f4a878e69beff39e3f72c17'
PINS={'manifest.json':'39613f9e09a5ce08e3297df93564c69d1a352e2e1a34f21b391107b03985a91c',
      'checkpoint-freeze.json':'8931d900993e4e737fe6d00a05cc7c1c43f4d8e717c7ca5c76dde5d9b8c6bd2f',
      'prediction-freeze.json':'ae26c900df1664057a14d54397b77363bfdcdb01564508156fc0a46b679c8f7e'}

def key(arm,seed):return f'{arm}__seed{seed}'
def json_scalar(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError(f'Unsupported JSON value: {type(value).__name__}')
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=json_scalar)+'\n')
def sources():
    paths=[PLAN,Path(__file__)]
    return sub.sources()|cases.source_hashes()|{str(p.relative_to(REPO)):ev.sha(p) for p in paths}
def check_sources(expected):
    if sources()!=expected:raise ValueError('Augmentation source/plan changed after freeze')

def check_checkpoints(output,m):
    if set(m['fits'])!={key(a,s) for s in SEEDS for a in ARMS}:raise ValueError('Incomplete nine-fit inventory')
    for name,fit in m['fits'].items():
        if ev.sha(output/'checkpoints'/(name+'.pt'))!=fit['weights_sha256']:raise ValueError('Checkpoint changed')

def natural_decision(natural):
    rows=[natural[key(CANDIDATE,s)]['macro'] for s in SEEDS]
    mean=float(np.mean([v['joint_error'] for v in rows]));success=float(np.mean([v['reference_voiced_success_rate'] for v in rows]))
    gains={b:natural[b]['macro']['joint_error']-mean for b in ('original_mlp','original_hybrid')}
    excess={c:float(np.mean([natural[key(CANDIDATE,s)]['per_condition'][c]['joint_error'] for s in SEEDS]))-natural['original_mlp']['per_condition'][c]['joint_error'] for c in old.CONDITIONS}
    checks={'mean_improves_both_by_half_point':all(v>=.005 for v in gains.values()),
        'each_seed_improves_original_mlp':all(v['joint_error']<natural['original_mlp']['macro']['joint_error'] for v in rows),
        'no_condition_worsens_over_one_point':all(v<=.01 for v in excess.values()),
        'voiced_success_not_lower':success>=natural['original_mlp']['macro']['reference_voiced_success_rate']}
    control=float(np.mean([natural[key('natural-only',s)]['macro']['joint_error'] for s in SEEDS]))
    return {'predesignated_candidate':CANDIDATE,'pass':all(checks.values()),'checks':checks,'mean_joint_error':mean,
        'mean_voiced_success':success,'joint_error_improvement_over_frozen':gains,'condition_error_excess_over_mlp':excess,
        'improvement_over_refitted_natural_control':control-mean}

def analytic_decision(regression,consumed_challenge,consumed_subharmonic,challenge):
    consumed=consumed_challenge
    if set(regression)!=old.SYNTHETIC_IDS:raise ValueError('Incomplete original regression inventory')
    from .harmonic_challenge import challenge_spec as consumed_spec
    from .subharmonic_challenge import challenge_spec as sub_spec
    from .augmentation_cases import challenge_spec
    if set(consumed_subharmonic)!={r['id'] for r in sub_spec()['records']}:raise ValueError('Incomplete consumed subharmonic inventory')
    if set(consumed)!={r['id'] for r in consumed_spec()['records']} or set(challenge)!={r['id'] for r in challenge_spec()['records']}:
        raise ValueError('Incomplete consumed or prospective challenge inventory')
    names=[key(CANDIDATE,s) for s in SEEDS]
    gates={k:{n:{} for n in names} for k in ('stationary','harmonic_change','consumed_challenge','consumed_subharmonic','challenge')}
    for n in names:
        for case,v in regression.items():
            if v['kind']=='constant':gates['stationary'][n][case]=bool(v['methods'][n]['all']['success_within_50c_all_frames']>=.95)
            elif v['kind']=='observation-change':gates['harmonic_change'][n][case]=bool(v['methods'][n]['regions']['after-observation-change']['success_within_50c_all_frames']>=.95)
        for g,items in [('consumed_challenge',consumed),('consumed_subharmonic',consumed_subharmonic),('challenge',challenge)]:
            gates[g][n]={case:bool(v['methods'][n]['regions']['after-observation-change']['success_within_50c_all_frames']>=.95) for case,v in items.items()}
    expected={'stationary':3,'harmonic_change':3,'consumed_challenge':12,'consumed_subharmonic':12,'challenge':12}
    for g,count in expected.items():
        if any(len(v)!=count for v in gates[g].values()):raise ValueError('Incomplete analytic gate inventory')
    return {**{g+'_by_seed_case':v for g,v in gates.items()},**{g+'_pass':all(all(x.values()) for x in v.values()) for g,v in gates.items()}}

def synthetic_order(seed,epoch,rows=18336):
    if rows<33*SYNTHETIC_BATCH:raise ValueError('Synthetic pool is too small for fixed without-replacement schedule')
    return np.random.default_rng(seed+1000000+epoch).permutation(rows)[:33*SYNTHETIC_BATCH].reshape(33,SYNTHETIC_BATCH).astype(np.int64)

def augmentation_loss(model,batch,targets,synthetic_batch=None,synthetic_targets=None):
    natural=selector_loss(model(batch),batch.candidate_mask,targets,objective='acceptable-set')
    auxiliary=natural*0.
    if synthetic_batch is not None:
        if synthetic_targets is None:raise ValueError('Synthetic targets required')
        auxiliary=SYNTHETIC_WEIGHT*selector_loss(model(synthetic_batch),synthetic_batch.candidate_mask,synthetic_targets,objective='acceptable-set')
    return natural+auxiliary,natural,auxiliary

def prepare_synthetic(items):
    # Keep the SAME common-eligible rows, including uncovered zero-loss targets.
    parts={k:[] for k in ('features','candidate_hz','candidate_mask')};meta={k:[] for k in ('reference_hz','scenario_index','reference_sample','availability_sample','region')}
    for index,item in enumerate(items):
        timing=item['timing'];ix=np.flatnonzero(timing.common_eligible)
        if len(ix)!=191:raise ValueError('Synthetic common target extent differs')
        for k in parts:parts[k].append(item['feature'][k][ix])
        meta['reference_hz'].append(np.full(len(ix),item['scenario']['initialHz']))
        meta['scenario_index'].append(np.full(len(ix),index,np.int64))
        meta['reference_sample'].append(timing.reference_sample[ix]);meta['availability_sample'].append(timing.availability_sample[ix])
        from .harmonic_challenge import challenge_region_at
        meta['region'].append(np.array([challenge_region_at(item['scenario'],int(e)) for e in timing.availability_sample[ix]]))
    raw={k:np.concatenate(v) for k,v in parts.items()};meta={k:np.concatenate(v) for k,v in meta.items()}
    b=prepare_selector_batch(**raw,representation='shared-candidates')
    t=old.targets(b,{'hz':meta['reference_hz'],'voiced':np.ones(len(b.features),bool),'valid':np.ones(len(b.features),bool)})
    if len(b.features)!=18336:raise ValueError('Expected96 paired TRAIN cases')
    return b,t,meta

def exposure(target,metadata,order):
    ix=order.reshape(-1);covered=target.covered.numpy()[ix]
    return {'selected_rows':len(ix),'unique_rows':len(np.unique(ix)),'covered_rows':int(covered.sum()),'uncovered_zero_rows':int((~covered).sum()),
      'per_case':[{'case_index':i,'selected_rows':int((metadata['scenario_index'][ix]==i).sum()),'covered_rows':int(((metadata['scenario_index'][ix]==i)&covered).sum())} for i in range(96)],
      'per_region':{k:{'selected_rows':int((metadata['region'][ix]==k).sum()),'covered_rows':int(((metadata['region'][ix]==k)&covered).sum())} for k in np.unique(metadata['region'])}}

def run(output):
    from .augmentation_cases import challenge_spec_sha256,generate_challenge
    from .harmonic_challenge import load_challenge
    output=output.resolve()
    if ARMS!=('natural-only','stationary-aug','balance-aug') or SYNTHETIC_WEIGHT!=.25 or SYNTHETIC_BATCH!=512:raise ValueError('Declared augmentation configuration changed')
    if not output.is_relative_to(RUNS):raise ValueError('Output must remain in ignored local runs')
    if ev.sha(PREREQUISITES)!='feadb83d71f76589c279501a75c70328af54f6cc1afe51b2f22c5a65252fc009':raise ValueError('Prerequisites changed')
    prerequisite=json.loads(PREREQUISITES.read_text());frozen=sources()
    if prerequisite['pass'] is not True or prerequisite['source_sha256']!=old.sources():raise ValueError('Unchanged passing prerequisites required')
    old.validate_headroom(Path(prerequisite['headroom_path']))
    for p,h in prerequisite['input_sha256'].items():
        if ev.sha(p)!=h:raise ValueError('Prerequisite input changed')
    if not torch.backends.mps.is_available():raise RuntimeError('Local MPS required')
    if shutil.disk_usage(ROOT).free<8*1024**3:raise RuntimeError('8 GiB free disk required')
    output.mkdir(parents=True,exist_ok=False)
    for directory in ('checkpoints','predictions','source-snapshot'):(output/directory).mkdir()
    for p,h in frozen.items():
        q=output/'source-snapshot'/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((REPO/p).read_bytes())
        if ev.sha(q)!=h:raise ValueError('Source copy changed')
    m={'schema':'phenometrix-augmentation-experiment.v1','status':'loading','started_at':old.now(),'source_sha256':frozen,
       'prior_run_sha256':PINS,'prerequisites_sha256':ev.sha(PREREQUISITES),'seeds':list(SEEDS),'arms':list(ARMS),'predesignated_candidate':CANDIDATE,
       'challenge_spec_sha256':challenge_spec_sha256(),'training_spec_sha256':cases.training_spec_sha256(),'device':'mps','torch':torch.__version__,'epochs':12,'batch_size':4096,'learning_rate':.001,
       'frozen_original_voicing':True,'frozen_original_weights_sha256':'06110469a9a643fb16806905569ebc939db2c0bdd5b3a6a1593b31789453d10d',
       'augmentation_configuration':{'weight':SYNTHETIC_WEIGHT,'batch_size':SYNTHETIC_BATCH,'objective':'natural covered-row set-CE mean +0.25 synthetic covered-row set-CE mean; one shared optimizer step','scaling':'natural TRAIN only','synthetic_schedule':'seed+1000000+epoch permutation; first16896 of18336 rows; paired between arms'},
       'measurement_delay_ms':20,'development_references_read':False,'reserved_content_read':False,'input_sha256':{},'fits':{},'peak_observed_mps_allocated_bytes':0}
    m['status']='sources_frozen';m['sources_frozen_at']=old.now();save(output/'source-freeze.json',m)
    save(output/'manifest.json',m)
    try:
        torch.set_num_threads(1)
        train=old.load_inputs('train');labels=old.load_labels(train);tb=old.batches(train)
        if labels['hashes']!=prerequisite['reference_sha256']['train']:raise ValueError('Training reference identity changed')
        m['input_sha256'].update(train['hashes']|labels['hashes'])
        rt=old.targets(tb['fixed-slots'],labels);target=old.targets(tb['shared-candidates'],labels)
        schedule=torch.nonzero(rt.covered,as_tuple=False).reshape(-1)
        m['scheduled_rows']=len(schedule);m['grouped_uncovered_scheduled_rows']=int((~target.covered[schedule]).sum())
        if len(schedule)!=132181 or m['grouped_uncovered_scheduled_rows']!=107:raise ValueError('Training schedule changed')
        b=tb['shared-candidates'].index_select(schedule);t=target.index_select(schedule)
        train_path=output/'training-targets.npz'
        np.savez_compressed(train_path,schedule=schedule.numpy(),covered=t.covered.numpy(),acceptable=t.acceptable.numpy())
        m['training_target_sha256']=ev.sha(train_path)
        synthetic_items,h=cases.generate_training(output/'training',output/'source-freeze.json',ev.sha(output/'source-freeze.json'))
        m['input_sha256'].update(h)
        banks={arm:prepare_synthetic(items) for arm,items in synthetic_items.items()}
        if set(banks)!=set(ARMS[1:]):raise ValueError('Paired synthetic arm inventory differs')
        for field in banks['stationary-aug'][2]:
            if not np.array_equal(banks['stationary-aug'][2][field],banks['balance-aug'][2][field]):raise ValueError('Synthetic paired row metadata differs')
        synthetic_path=output/'synthetic-targets.npz'
        np.savez_compressed(synthetic_path,**{arm+'__'+k:v for arm,(_,targets,metadata) in banks.items() for k,v in {'covered':targets.covered.numpy(),'acceptable':targets.acceptable.numpy(),**metadata}.items()})
        m['synthetic_target_sha256']=ev.sha(synthetic_path)
        orders={f'seed{seed}__epoch{epoch}':synthetic_order(seed,epoch) for seed in SEEDS for epoch in range(12)}
        orders_path=output/'synthetic-schedules.npz';np.savez_compressed(orders_path,**orders);m['synthetic_schedule_sha256']=ev.sha(orders_path)
        m['synthetic_pool']={arm:{'rows':len(b.features),'covered_rows':int(t.covered.sum()),'uncovered_zero_rows':int((~t.covered).sum())} for arm,(b,t,_) in banks.items()}
        device_batch=b.to('mps');device_target=t.to('mps')
        device_synthetic={arm:(b.to('mps'),t.to('mps')) for arm,(b,t,_) in banks.items()}
        dev=old.load_inputs('development');db=old.batches(dev)['shared-candidates']
        probability,predictions,baseline_hashes=old.read_natural_baselines(dev);m['input_sha256'].update(dev['hashes']|baseline_hashes)
        previous_models,previous_predictions,h=sub.prior_inputs(dev);predictions.update(previous_predictions);m['input_sha256'].update(h)
        regression,h=old.synthetic_inputs(None,None);m['input_sha256'].update(h)
        from .evaluate_synthetic import OBSERVATION_SCENARIOS
        support=json.loads((RUNS/'support-diagnostic-v1/manifest.json').read_text())
        scenarios={s['id']:s for s in [*support['scenarios'],*OBSERVATION_SCENARIOS]}
        for item in regression:item['batch']=item['batches']['shared-candidates'];item['scenario']=scenarios[item['id']]
        consumed,h=load_challenge(PRIOR/'challenge',PRIOR/'checkpoint-freeze.json',PINS['checkpoint-freeze.json']);m['input_sha256'].update(h)
        sub.add_batch_probability(consumed)
        if ev.sha(LAST/'manifest.json')!=LAST_MANIFEST_SHA:raise ValueError('Previous subharmonic manifest changed')
        lm=json.loads((LAST/'manifest.json').read_text());sub.check_sources(lm['source_sha256']);sub.check_checkpoints(LAST,lm)
        from .subharmonic_challenge import load_challenge as load_subharmonic
        consumed_sub,h=load_subharmonic(LAST/'challenge',LAST/'checkpoint-freeze.json',ev.sha(LAST/'checkpoint-freeze.json'));m['input_sha256'].update(h)
        m['input_sha256'][str(LAST/'manifest.json')]=LAST_MANIFEST_SHA
        sub.add_batch_probability(consumed_sub)
        original_weight=RUNS/'delayed-mlp-seed20261022/weights.pt';m['input_sha256'][str(original_weight)]=ev.sha(original_weight)
        check_sources(frozen)
        m['status']='fitting';save(output/'manifest.json',m);began=time.monotonic();models={}
        for seed in SEEDS:
            initial_identity=None
            for arm in ARMS:
                torch.manual_seed(seed);torch.mps.manual_seed(seed)
                model=HarmonicSelector('relative','acceptable-set');model.fit_scaling(tb['shared-candidates'],split='train')
                digest=hashlib.sha256()
                for n,tensor in sorted(model.state_dict().items()):digest.update(n.encode());digest.update(tensor.detach().cpu().numpy().tobytes())
                initial_sha=digest.hexdigest()
                if initial_identity is not None and initial_identity!=initial_sha:raise ValueError('Arm initial tensors differ')
                initial_identity=initial_sha;model.to('mps');optimizer=torch.optim.Adam(model.parameters(),lr=.001)
                losses=[];base_losses=[];weighted_auxiliary_losses=[];exposures=[];updates=skipped=0;started=time.monotonic();name=key(arm,seed)
                for epoch in range(12):
                    order=torch.from_numpy(np.random.default_rng(seed+epoch).permutation(len(schedule)).astype(np.int64));total=base_total=aux_total=denominator=0
                    synthetic_indices=orders[f'seed{seed}__epoch{epoch}']
                    if arm!='natural-only':exposures.append(exposure(banks[arm][1],banks[arm][2],synthetic_indices))
                    for start in range(0,len(order),4096):
                        if time.monotonic()-began>1800:raise TimeoutError('30 minute fitting limit reached')
                        ix=order[start:start+4096];batch=device_batch.index_select(ix);targets=device_target.index_select(ix);n=int(targets.covered.sum())
                        if not n:skipped+=1;continue
                        optimizer.zero_grad(set_to_none=True)
                        sb=st=None
                        if arm!='natural-only':
                            sx=synthetic_indices[start//4096];sb=device_synthetic[arm][0].index_select(sx);st=device_synthetic[arm][1].index_select(sx)
                        loss,base_loss,auxiliary_loss=augmentation_loss(model,batch,targets,sb,st)
                        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                        total+=float(loss.detach().cpu())*n;base_total+=float(base_loss.detach().cpu())*n;aux_total+=float(auxiliary_loss.detach().cpu())*n;denominator+=n;updates+=1
                    torch.mps.synchronize();losses.append(total/denominator);base_losses.append(base_total/denominator);weighted_auxiliary_losses.append(aux_total/denominator)
                    allocation=torch.mps.current_allocated_memory();m['peak_observed_mps_allocated_bytes']=max(m['peak_observed_mps_allocated_bytes'],allocation)
                    if allocation>8*1024**3:raise MemoryError('8 GiB MPS cap exceeded')
                model=model.cpu().eval();ck=output/'checkpoints'/(name+'.pt')
                torch.save({'arm':arm,'context_mode':'relative','config':asdict(model.config),'state_dict':model.state_dict()},ck)
                m['fits'][name]={'seed':seed,'arm':arm,'epoch_losses':losses,'epoch_base_losses':base_losses,'epoch_weighted_auxiliary_losses':weighted_auxiliary_losses,'optimizer_updates':updates,'skipped_batches':skipped,'synthetic_exposure_by_epoch':exposures,
                    'elapsed_seconds':time.monotonic()-started,'parameter_count':sum(p.numel() for p in model.parameters()),'initial_tensors_sha256':initial_sha,'weights_sha256':ev.sha(ck)}
                if updates!=396 or skipped:raise ValueError('Update schedule changed')
                models[name]=model;save(output/'manifest.json',m);print(json.dumps({'fit':name,'elapsed_seconds':m['fits'][name]['elapsed_seconds'],'optimizer_updates':updates,'final_loss':losses[-1]}),flush=True)
        m.update(status='checkpoints_frozen',checkpoints_frozen_at=old.now(),fit_elapsed_seconds=time.monotonic()-began)
        check_checkpoints(output,m);check_sources(frozen);save(output/'checkpoint-freeze.json',m);save(output/'manifest.json',m)
        challenge,h=generate_challenge(output/'challenge',output/'checkpoint-freeze.json',ev.sha(output/'checkpoint-freeze.json'))
        m['input_sha256'].update(h);sub.add_batch_probability(challenge)
        for name,model in models.items():predictions[name]=old.infer(model,db,probability)
        all_models=previous_models|models
        groups={'regression':regression,'consumed_challenge':consumed,'consumed_subharmonic':consumed_sub,'challenge':challenge}
        ap={g:{item['id']:{name:old.infer(model,item['batch'],item['probability']) for name,model in all_models.items()} for item in items} for g,items in groups.items()}
        hashes={}
        for name,(state,hz) in predictions.items():
            p=output/'predictions'/(name+'.npz');np.savez_compressed(p,state=state,hz=hz);hashes[str(p.relative_to(output))]=ev.sha(p)
        for g,items in groups.items():
            for item in items:
                methods=ap[g][item['id']];p=output/'predictions'/(g+'-'+item['id']+'.npz')
                np.savez_compressed(p,original_probability=item['probability'],**{k+'__state':v[0] for k,v in methods.items()},**{k+'__hz':v[1] for k,v in methods.items()})
                hashes[str(p.relative_to(output))]=ev.sha(p)
        m.update(status='predictions_frozen',predictions_frozen_at=old.now(),prediction_sha256=hashes)
        check_checkpoints(output,m);check_sources(frozen);save(output/'prediction-freeze.json',m);save(output/'manifest.json',m)
        dl=old.load_labels(dev);m['development_references_read']=True
        if dl['hashes']!=prerequisite['reference_sha256']['development']:raise ValueError('Development reference identity changed')
        m['input_sha256'].update(dl['hashes']);natural=old.aggregate(dev,dl,predictions)
        scores={g:prior.analytic_scores(ap[g],items) for g,items in groups.items()}
        decision=natural_decision(natural);decision['analytic']=analytic_decision(**scores)
        decision['promising_for_next_study']=decision['pass'] and all(v for k,v in decision['analytic'].items() if k.endswith('_pass'))
        result={'schema':'phenometrix-augmentation-metrics.v1','scope':'reused natural development and38 analytic regressions;12 prospective synthetic configurations; no external speech',
                'natural':natural,**scores,'decision':decision,'exclusions':dl['excluded']}
        save(output/'metrics.json',result)
        for p,h in hashes.items():
            if ev.sha(output/p)!=h:raise ValueError('Frozen prediction changed')
        for p,h in m['input_sha256'].items():
            if ev.sha(p)!=h:raise ValueError('Input changed during run')
        for path,field in [(train_path,'training_target_sha256'),(synthetic_path,'synthetic_target_sha256'),(orders_path,'synthetic_schedule_sha256')]:
            if ev.sha(path)!=m[field]:raise ValueError('Training targets/schedules changed')
        check_checkpoints(output,m);check_sources(frozen)
        size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if size>2*1024**3:raise RuntimeError('2 GiB artifact cap exceeded')
        m.update(status='complete',completed_at=old.now(),metrics_sha256=ev.sha(output/'metrics.json'),output_bytes=size)
        save(output/'manifest.json',m);print(json.dumps(decision,indent=2,default=json_scalar),flush=True);return result
    except BaseException as error:
        m.update(status='failed',failure=repr(error),failed_at=old.now());save(output/'manifest.json',m);raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
