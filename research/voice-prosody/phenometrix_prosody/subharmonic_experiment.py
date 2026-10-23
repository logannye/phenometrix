"""One fixed local objective study for harmonic aliases; historical runs stay exact."""
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
from .subharmonic_objective import ARMS, MARGIN, MARGIN_WEIGHT, build_negative_masks, subharmonic_loss

ROOT,RUNS,REPO=old.ROOT,old.RUNS,old.REPO
PLAN=ROOT/'SUBHARMONIC_EXPERIMENT_PLAN.md'
PRIOR=RUNS/'harmonic-context-v1'
PREREQUISITES=RUNS/'selector-prerequisites-v1/prerequisites.json'
SEEDS=old.SEEDS
CANDIDATE='alias-margin'
PINS={'manifest.json':'39613f9e09a5ce08e3297df93564c69d1a352e2e1a34f21b391107b03985a91c',
      'checkpoint-freeze.json':'8931d900993e4e737fe6d00a05cc7c1c43f4d8e717c7ca5c76dde5d9b8c6bd2f',
      'prediction-freeze.json':'ae26c900df1664057a14d54397b77363bfdcdb01564508156fc0a46b679c8f7e'}

def key(arm,seed):return f'{arm}__seed{seed}'
def json_scalar(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError(f'Unsupported JSON value: {type(value).__name__}')
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=json_scalar)+'\n')
def sources():
    from .subharmonic_challenge import challenge_source_hashes
    paths=[PLAN,Path(__file__),Path(__file__).with_name('subharmonic_objective.py')]
    return prior.sources()|challenge_source_hashes()|{str(p.relative_to(REPO)):ev.sha(p) for p in paths}
def check_sources(expected):
    if sources()!=expected:raise ValueError('Subharmonic source/plan changed after freeze')
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
    control=float(np.mean([natural[key('set',s)]['macro']['joint_error'] for s in SEEDS]))
    return {'predesignated_candidate':CANDIDATE,'pass':all(checks.values()),'checks':checks,'mean_joint_error':mean,
        'mean_voiced_success':success,'joint_error_improvement_over_frozen':gains,'condition_error_excess_over_mlp':excess,
        'improvement_over_refitted_set_control':control-mean}

def analytic_decision(regression,consumed_challenge,challenge):
    consumed=consumed_challenge
    if set(regression)!=old.SYNTHETIC_IDS:raise ValueError('Incomplete original regression inventory')
    from .harmonic_challenge import challenge_spec as consumed_spec
    from .subharmonic_challenge import challenge_spec
    if set(consumed)!={r['id'] for r in consumed_spec()['records']} or set(challenge)!={r['id'] for r in challenge_spec()['records']}:
        raise ValueError('Incomplete consumed or prospective challenge inventory')
    names=[key(CANDIDATE,s) for s in SEEDS]
    gates={k:{n:{} for n in names} for k in ('stationary','harmonic_change','consumed_challenge','challenge')}
    for n in names:
        for case,v in regression.items():
            if v['kind']=='constant':gates['stationary'][n][case]=bool(v['methods'][n]['all']['success_within_50c_all_frames']>=.95)
            elif v['kind']=='observation-change':gates['harmonic_change'][n][case]=bool(v['methods'][n]['regions']['after-observation-change']['success_within_50c_all_frames']>=.95)
        for g,items in [('consumed_challenge',consumed),('challenge',challenge)]:
            gates[g][n]={case:bool(v['methods'][n]['regions']['after-observation-change']['success_within_50c_all_frames']>=.95) for case,v in items.items()}
    expected={'stationary':3,'harmonic_change':3,'consumed_challenge':12,'challenge':12}
    for g,count in expected.items():
        if any(len(v)!=count for v in gates[g].values()):raise ValueError('Incomplete analytic gate inventory')
    return {**{g+'_by_seed_case':v for g,v in gates.items()},**{g+'_pass':all(all(x.values()) for x in v.values()) for g,v in gates.items()}}

def prior_inputs(dev):
    hashes={str(PRIOR/n):ev.sha(PRIOR/n) for n in PINS}
    if any(hashes[str(PRIOR/n)]!=h for n,h in PINS.items()):raise ValueError('Pinned prior run changed')
    m=json.loads((PRIOR/'manifest.json').read_text());pf=json.loads((PRIOR/'prediction-freeze.json').read_text())
    if m['status']!='failed' or m['failure']!="TypeError('Object of type bool is not JSON serializable')":raise ValueError('Prior failure identity changed')
    prior.check_sources(m['source_sha256']);prior.check_checkpoints(PRIOR,m)
    models={};predictions={}
    for seed in SEEDS:
        oldname=prior.key('relative','acceptable-set',seed);name='prior__'+oldname
        ck=PRIOR/'checkpoints'/(oldname+'.pt');p=PRIOR/'predictions'/(oldname+'.npz')
        if ev.sha(p)!=pf['prediction_sha256'][str(p.relative_to(PRIOR))]:raise ValueError('Prior natural prediction changed')
        saved=torch.load(ck,map_location='cpu',weights_only=True)
        model=HarmonicSelector('relative','acceptable-set');model.load_state_dict(saved['state_dict'],strict=True);model.eval();models[name]=model
        with np.load(p,allow_pickle=False) as z:predictions[name]=(z['state'].copy(),z['hz'].copy())
        if len(predictions[name][0])!=len(dev['raw']['features']):raise ValueError('Prior natural extent differs')
        hashes.update({str(ck):ev.sha(ck),str(p):ev.sha(p)})
    return models,predictions,hashes

def add_batch_probability(items):
    for item in items:
        f=item['feature'];item['batch']=prepare_selector_batch(f['features'],f['candidate_hz'],f['candidate_mask'],representation='shared-candidates')
        item['probability']=prior.original_probability(f)

def run(output):
    from .subharmonic_challenge import challenge_spec_sha256,generate_challenge
    from .harmonic_challenge import load_challenge
    output=output.resolve()
    if ARMS!=('set','all-margin','alias-margin') or MARGIN!=1. or MARGIN_WEIGHT!=.5:raise ValueError('Declared objective configuration changed')
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
    m={'schema':'phenometrix-subharmonic-experiment.v1','status':'loading','started_at':old.now(),'source_sha256':frozen,
       'prior_run_sha256':PINS,'prerequisites_sha256':ev.sha(PREREQUISITES),'seeds':list(SEEDS),'arms':list(ARMS),'predesignated_candidate':CANDIDATE,
       'challenge_spec_sha256':challenge_spec_sha256(),'device':'mps','torch':torch.__version__,'epochs':12,'batch_size':4096,'learning_rate':.001,
       'frozen_original_voicing':True,'frozen_original_weights_sha256':'06110469a9a643fb16806905569ebc939db2c0bdd5b3a6a1593b31789453d10d',
       'margin_configuration':{'margin':MARGIN,'weight':MARGIN_WEIGHT,'factors':[2.,3.,4.,.5,1/3,.25],'alias_tolerance_cents':50.,'eligible_subset':'covered rows with an admitted wrong candidate in the current minibatch','epoch_diagnostics':'covered-row-weighted averages of minibatch base and weighted auxiliary objectives; not a global eligible-negative-row average'},
       'measurement_delay_ms':20,'development_references_read':False,'reserved_content_read':False,'input_sha256':{},'fits':{},'peak_observed_mps_allocated_bytes':0}
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
        neg=build_negative_masks(tb['shared-candidates'],target)
        m['margin_training_rows']={arm:int((mask.any(-1)&target.covered)[schedule].sum()) for arm,mask in neg.items()}
        m['margin_negative_candidates']={arm:int(mask[schedule].sum()) for arm,mask in neg.items()}
        b=tb['shared-candidates'].index_select(schedule);t=target.index_select(schedule)
        train_path=output/'training-targets.npz'
        np.savez_compressed(train_path,schedule=schedule.numpy(),covered=t.covered.numpy(),acceptable=t.acceptable.numpy(),
                            **{k:v[schedule].numpy() for k,v in neg.items()})
        m['training_target_sha256']=ev.sha(train_path)
        device_batch=b.to('mps');device_target=t.to('mps');device_neg={a:v[schedule].to('mps') for a,v in neg.items()}
        dev=old.load_inputs('development');db=old.batches(dev)['shared-candidates']
        probability,predictions,baseline_hashes=old.read_natural_baselines(dev);m['input_sha256'].update(dev['hashes']|baseline_hashes)
        previous_models,previous_predictions,h=prior_inputs(dev);predictions.update(previous_predictions);m['input_sha256'].update(h)
        regression,h=old.synthetic_inputs(None,None);m['input_sha256'].update(h)
        from .evaluate_synthetic import OBSERVATION_SCENARIOS
        support=json.loads((RUNS/'support-diagnostic-v1/manifest.json').read_text())
        scenarios={s['id']:s for s in [*support['scenarios'],*OBSERVATION_SCENARIOS]}
        for item in regression:item['batch']=item['batches']['shared-candidates'];item['scenario']=scenarios[item['id']]
        consumed,h=load_challenge(PRIOR/'challenge',PRIOR/'checkpoint-freeze.json',PINS['checkpoint-freeze.json']);m['input_sha256'].update(h)
        add_batch_probability(consumed)
        original_weight=RUNS/'delayed-mlp-seed20261022/weights.pt';m['input_sha256'][str(original_weight)]=ev.sha(original_weight)
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
                losses=[];base_losses=[];weighted_auxiliary_losses=[];updates=skipped=0;started=time.monotonic();name=key(arm,seed)
                for epoch in range(12):
                    order=torch.from_numpy(np.random.default_rng(seed+epoch).permutation(len(schedule)).astype(np.int64));total=base_total=aux_total=denominator=0
                    for start in range(0,len(order),4096):
                        if time.monotonic()-began>1800:raise TimeoutError('30 minute fitting limit reached')
                        ix=order[start:start+4096];batch=device_batch.index_select(ix);targets=device_target.index_select(ix);n=int(targets.covered.sum())
                        if not n:skipped+=1;continue
                        negatives=device_neg['all-margin' if arm=='set' else arm].index_select(0,ix.to('mps'))
                        optimizer.zero_grad(set_to_none=True)
                        logits=model(batch)
                        loss=subharmonic_loss(logits,batch.candidate_mask,targets,negatives,arm=arm)
                        base_loss=selector_loss(logits,batch.candidate_mask,targets,objective='acceptable-set')
                        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                        total+=float(loss.detach().cpu())*n;base_total+=float(base_loss.detach().cpu())*n;aux_total+=float((loss-base_loss).detach().cpu())*n;denominator+=n;updates+=1
                    torch.mps.synchronize();losses.append(total/denominator);base_losses.append(base_total/denominator);weighted_auxiliary_losses.append(aux_total/denominator)
                    allocation=torch.mps.current_allocated_memory();m['peak_observed_mps_allocated_bytes']=max(m['peak_observed_mps_allocated_bytes'],allocation)
                    if allocation>8*1024**3:raise MemoryError('8 GiB MPS cap exceeded')
                model=model.cpu().eval();ck=output/'checkpoints'/(name+'.pt')
                torch.save({'arm':arm,'context_mode':'relative','config':asdict(model.config),'state_dict':model.state_dict()},ck)
                m['fits'][name]={'seed':seed,'arm':arm,'epoch_losses':losses,'epoch_base_losses':base_losses,'epoch_weighted_auxiliary_losses':weighted_auxiliary_losses,'optimizer_updates':updates,'skipped_batches':skipped,
                    'elapsed_seconds':time.monotonic()-started,'parameter_count':sum(p.numel() for p in model.parameters()),'initial_tensors_sha256':initial_sha,'weights_sha256':ev.sha(ck)}
                if updates!=396 or skipped:raise ValueError('Update schedule changed')
                models[name]=model;save(output/'manifest.json',m);print(json.dumps({'fit':name,**m['fits'][name]}),flush=True)
        m.update(status='checkpoints_frozen',checkpoints_frozen_at=old.now(),fit_elapsed_seconds=time.monotonic()-began)
        check_checkpoints(output,m);check_sources(frozen);save(output/'checkpoint-freeze.json',m);save(output/'manifest.json',m)
        challenge,h=generate_challenge(output/'challenge',output/'checkpoint-freeze.json',ev.sha(output/'checkpoint-freeze.json'))
        m['input_sha256'].update(h);add_batch_probability(challenge)
        for name,model in models.items():predictions[name]=old.infer(model,db,probability)
        all_models=previous_models|models
        groups={'regression':regression,'consumed_challenge':consumed,'challenge':challenge}
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
        result={'schema':'phenometrix-subharmonic-metrics.v1','scope':'reused natural development and26 analytic regressions;12 prospective synthetic configurations; no external speech',
                'natural':natural,**scores,'decision':decision,'exclusions':dl['excluded']}
        save(output/'metrics.json',result)
        for p,h in hashes.items():
            if ev.sha(output/p)!=h:raise ValueError('Frozen prediction changed')
        for p,h in m['input_sha256'].items():
            if ev.sha(p)!=h:raise ValueError('Input changed during run')
        if ev.sha(train_path)!=m['training_target_sha256']:raise ValueError('Training targets changed')
        check_checkpoints(output,m);check_sources(frozen)
        size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if size>2*1024**3:raise RuntimeError('2 GiB artifact cap exceeded')
        m.update(status='complete',completed_at=old.now(),metrics_sha256=ev.sha(output/'metrics.json'),output_bytes=size)
        save(output/'manifest.json',m);print(json.dumps(decision,indent=2,default=json_scalar),flush=True);return result
    except BaseException as error:
        m.update(status='failed',failure=repr(error),failed_at=old.now());save(output/'manifest.json',m);raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
