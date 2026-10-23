"""One bounded local harmonic-context experiment; old experiment stays immutable."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import hashlib
from io import BytesIO
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from . import selector_experiment as old
from . import evaluate as ev
from .selector_model import prepare_selector_batch, build_pitch_targets, selector_loss, predict_pitch_hz
from .harmonic_selector_model import HarmonicSelector
from .model import VoiceCandidateModel, ModelConfig

ROOT, RUNS, REPO = old.ROOT, old.RUNS, old.REPO
PLAN = ROOT / 'HARMONIC_EXPERIMENT_PLAN.md'
SEEDS = old.SEEDS
ARMS = (('absolute','nearest'),('absolute','acceptable-set'),('relative','nearest'),('relative','acceptable-set'))
CANDIDATE = ('relative','acceptable-set')
PRIOR = RUNS / 'selector-factorial-v1'
PREREQUISITES = RUNS / 'selector-prerequisites-v1/prerequisites.json'

def key(mode, objective, seed):
    return f'{mode}__{objective}__seed{seed}'

def sources():
    names = ['HARMONIC_EXPERIMENT_PLAN.md', 'phenometrix_prosody/harmonic_selector_experiment.py',
             'phenometrix_prosody/harmonic_selector_model.py', 'phenometrix_prosody/harmonic_challenge.py']
    out = old.sources()
    out.update({str((ROOT/n).relative_to(REPO)):ev.sha(ROOT/n) for n in names})
    # Bind the complete existing waveform-to-feature replay dependency closure.
    p = RUNS/'delayed-known-truth/prediction-freeze.json'
    for name in json.loads(p.read_text())['source_sha256']:
        out[name] = ev.sha(REPO/name)
    from .harmonic_challenge import challenge_source_hashes
    out.update(challenge_source_hashes())
    return out

def check_sources(expected):
    if sources() != expected:
        raise ValueError('Harmonic experiment sources changed after freeze')

def check_checkpoints(output,manifest):
    expected={key(mode,objective,seed) for seed in SEEDS for mode,objective in ARMS}
    if set(manifest['fits'])!=expected:raise ValueError('Incomplete checkpoint inventory')
    for name,fit in manifest['fits'].items():
        if ev.sha(output/'checkpoints'/(name+'.pt'))!=fit['weights_sha256']:raise ValueError('Checkpoint changed after fit')

def natural_decision(result):
    names=[key(*CANDIDATE,s) for s in SEEDS]
    rows=[result[n]['macro'] for n in names]
    mean=float(np.mean([r['joint_error'] for r in rows]))
    gains={b:result[b]['macro']['joint_error']-mean for b in ('original_mlp','original_hybrid')}
    conditions={c:float(np.mean([result[n]['per_condition'][c]['joint_error'] for n in names]))-result['original_mlp']['per_condition'][c]['joint_error'] for c in old.CONDITIONS}
    success=float(np.mean([r['reference_voiced_success_rate'] for r in rows]))
    checks={'mean_improves_both_by_half_point':all(v>=.005 for v in gains.values()),
            'each_seed_improves_original_mlp':all(r['joint_error']<result['original_mlp']['macro']['joint_error'] for r in rows),
            'no_condition_worsens_over_one_point':all(v<=.01 for v in conditions.values()),
            'voiced_success_not_lower':success>=result['original_mlp']['macro']['reference_voiced_success_rate']}
    control=float(np.mean([result[key('absolute','acceptable-set',s)]['macro']['joint_error'] for s in SEEDS]))
    return {'predesignated_candidate':list(CANDIDATE),'pass':all(checks.values()),'checks':checks,
            'mean_joint_error':mean,'mean_voiced_success':success,'joint_error_improvement_over_frozen':gains,
            'condition_error_excess_over_mlp':conditions,'improvement_over_equal_parameter_absolute_control':control-mean}

def analytic_scores(predictions, items):
    from .evaluate_synthetic import analytic_frequency, summarize, step_response
    result={}
    for item in items:
        scenario,timing=item['scenario'],item['timing']
        name=item['id']; methods={}
        if set(predictions[name]) != set(predictions[items[0]['id']]):
            raise ValueError('Incomplete analytic method inventory')
        for method,(state,hz) in predictions[name].items():
            rows=[]
            for i in np.flatnonzero(timing.common_eligible):
                sample,end=int(timing.reference_sample[i]),int(timing.availability_sample[i])
                truth=analytic_frequency(scenario,sample/48000)
                estimate=float(hz[i]) if state[i]==1 else None
                if 'change_sample' in scenario:
                    from .harmonic_challenge import challenge_region_at
                    region=challenge_region_at(scenario,end)
                elif scenario['kind']=='observation-change':
                    start=(end-1920)/48000; stop=end/48000; change=scenario['changeStartSeconds']
                    region='before-observation-change' if stop<change else 'after-observation-change' if start>=change else 'observation-boundary'
                else:
                    from .evaluate_synthetic import region_at
                    region=region_at(scenario,end)
                slope=scenario['slopeHzPerSecond'] if region=='glide-interior' else 0
                rows.append({'reference_sample':sample,'target_ms':sample/48,'availability_ms':end/48,
                    'region':region,'estimate_hz':estimate,'signed_cents':None if estimate is None else 1200*np.log2(estimate/truth),
                    'equivalent_slope_delay_ms':None if estimate is None or not slope else (truth-estimate)/slope*1000})
            methods[method]={'all':summarize(rows),'regions':{r:summarize([v for v in rows if v['region']==r]) for r in sorted({v['region'] for v in rows})},
                             'step_response':step_response(rows,scenario) if scenario['kind']=='step' else None}
        batch=item['batch']; reference=np.full(len(batch.features),scenario['initialHz'])
        if scenario['kind'] not in {'constant','observation-change'}:
            reference=np.array([analytic_frequency(scenario,int(s)/48000) for s in timing.reference_sample])
        covered=build_pitch_targets(batch,reference,np.ones(len(reference),bool),timing.common_eligible).covered.numpy()
        eligible=timing.common_eligible
        result[name]={'scenario':scenario,'kind':scenario['kind'],'methods':methods,
                      'candidate_coverage':{'eligible_frames':int(eligible.sum()),'covered_frames':int(covered.sum()),
                                            'fraction':float(covered.sum()/eligible.sum())}}
    return result

def analytic_decision(regression, challenge):
    names=[key(*CANDIDATE,s) for s in SEEDS]
    stationary={}; harmonic={}; prospective={}
    for n in names:
        stationary[n]={k:v['methods'][n]['all']['success_within_50c_all_frames']>=.95 for k,v in regression.items() if v['kind']=='constant'}
        harmonic[n]={k:v['methods'][n]['regions']['after-observation-change']['success_within_50c_all_frames']>=.95 for k,v in regression.items() if v['kind']=='observation-change'}
        prospective[n]={k:v['methods'][n]['regions']['after-observation-change']['success_within_50c_all_frames']>=.95 for k,v in challenge.items()}
    if any(len(v)!=3 for v in stationary.values()) or any(len(v)!=3 for v in harmonic.values()) or any(len(v)!=12 for v in prospective.values()):
        raise ValueError('Incomplete fixed analytic gate inventory')
    passed=lambda groups:all(all(v.values()) for v in groups.values())
    return {'stationary_by_seed_case':stationary,'harmonic_change_by_seed_case':harmonic,'challenge_by_seed_case':prospective,
            'stationary_pass':passed(stationary),'harmonic_change_pass':passed(harmonic),'challenge_pass':passed(prospective)}

def load_prior(dev):
    manifest=json.loads((PRIOR/'manifest.json').read_text())
    freeze=json.loads((PRIOR/'prediction-freeze.json').read_text())
    if manifest['status']!='complete' or freeze['status']!='predictions_frozen':
        raise ValueError('Complete prior experiment required')
    for name,h in manifest['source_sha256'].items():
        if ev.sha(REPO/name)!=h:raise ValueError('Historical source changed')
    models={}; predictions={}; hashes={str(PRIOR/'manifest.json'):ev.sha(PRIOR/'manifest.json'),str(PRIOR/'prediction-freeze.json'):ev.sha(PRIOR/'prediction-freeze.json')}
    from .selector_model import PitchSelector, SelectorConfig
    for objective in ('nearest','acceptable-set'):
        for seed in SEEDS:
            name=old.model_key('shared-candidates',objective,seed)
            ck=PRIOR/'checkpoints'/(name+'.pt'); p=PRIOR/'predictions'/(name+'.npz')
            if ev.sha(ck)!=manifest['fits'][name]['weights_sha256'] or ev.sha(p)!=freeze['prediction_sha256'][str(p.relative_to(PRIOR))]:
                raise ValueError('Historical checkpoint/prediction identity changed')
            saved=torch.load(ck,map_location='cpu',weights_only=True)
            model=PitchSelector(SelectorConfig(**saved['config']));model.load_state_dict(saved['state_dict'],strict=True);model.eval()
            method='prior__'+name;models[method]=model
            with np.load(p,allow_pickle=False) as z: predictions[method]=(z['state'].copy(),z['hz'].copy())
            if len(predictions[method][0])!=len(dev['raw']['features']):raise ValueError('Prior prediction extent changed')
            hashes.update({str(ck):ev.sha(ck),str(p):ev.sha(p)})
    return models,predictions,hashes

def original_probability(feature):
    p=RUNS/'delayed-mlp-seed20261022/weights.pt'
    data=p.read_bytes()
    if hashlib.sha256(data).hexdigest()!='06110469a9a643fb16806905569ebc939db2c0bdd5b3a6a1593b31789453d10d':raise ValueError('Frozen original voicing weights changed')
    saved=torch.load(BytesIO(data),map_location='cpu',weights_only=True)
    model=VoiceCandidateModel(ModelConfig(**saved['config']));model.load_state_dict(saved['state_dict'],strict=True);model.eval()
    x=torch.from_numpy(feature['features']); mask=torch.from_numpy(feature['candidate_mask'])
    with torch.inference_mode():
        v,_,_=model(x,mask,model.initial_state(len(x)))
        result=torch.sigmoid(v[:,0].double()).numpy()
    if not np.isfinite(result).all():raise ValueError('Nonfinite original voicing')
    return result

def run(output):
    from .harmonic_challenge import challenge_spec_sha256
    if ev.sha(PRIOR/'manifest.json') != 'df56e53be7399807f068616961ba9eac6ff1d6ce7dd76d6614166d79bbd664f5' or ev.sha(PREREQUISITES) != 'feadb83d71f76589c279501a75c70328af54f6cc1afe51b2f22c5a65252fc009':
        raise ValueError('Previously published prerequisite/run identity changed')
    frozen=sources(); prerequisite=json.loads(PREREQUISITES.read_text())
    if prerequisite.get('pass') is not True or prerequisite['source_sha256']!=old.sources():
        raise ValueError('Unchanged passing historical prerequisites required')
    old.validate_headroom(Path(prerequisite['headroom_path']))
    for name,h in prerequisite['input_sha256'].items():
        if ev.sha(name)!=h:raise ValueError('Prerequisite acquisition changed')
    if not torch.backends.mps.is_available():raise RuntimeError('Local MPS required; no fallback')
    if shutil.disk_usage(ROOT).free<8*1024**3:raise RuntimeError('At least 8 GiB free disk required')
    output.mkdir(parents=True,exist_ok=False)
    for directory in ('checkpoints','predictions','source-snapshot'): (output/directory).mkdir()
    for name in frozen:
        p=output/'source-snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((REPO/name).read_bytes())
    m={'schema':'phenometrix-harmonic-context-experiment.v1','status':'loading','started_at':old.now(),
       'source_sha256':frozen,'base_experiment_sha256':ev.sha(PRIOR/'manifest.json'),'prerequisites_sha256':ev.sha(PREREQUISITES),
       'seeds':list(SEEDS),'arms':[list(a) for a in ARMS],'predesignated_candidate':list(CANDIDATE),'challenge_spec_sha256':challenge_spec_sha256(),
       'device':'mps','torch':torch.__version__,'epochs':12,'batch_size':4096,'learning_rate':.001,
       'frozen_original_voicing':True,'frozen_original_weights_sha256':'06110469a9a643fb16806905569ebc939db2c0bdd5b3a6a1593b31789453d10d','measurement_delay_ms':20,'development_references_read':False,
       'reserved_content_read':False,'input_sha256':{},'fits':{},'peak_observed_mps_allocated_bytes':0}
    old.save(output/'manifest.json',m)
    try:
        torch.set_num_threads(1)
        train=old.load_inputs('train'); labels=old.load_labels(train); tb=old.batches(train)
        if labels['hashes']!=prerequisite['reference_sha256']['train']:raise ValueError('Training reference identity changed')
        m['input_sha256'].update(train['hashes']|labels['hashes'])
        raw_targets=old.targets(tb['fixed-slots'],labels);target=old.targets(tb['shared-candidates'],labels)
        schedule=torch.nonzero(raw_targets.covered,as_tuple=False).reshape(-1)
        m['scheduled_rows']=len(schedule);m['grouped_uncovered_scheduled_rows']=int((~target.covered[schedule]).sum())
        if len(schedule)!=132181 or m['grouped_uncovered_scheduled_rows']!=107:raise ValueError('Fixed training schedule changed')
        device_batch=tb['shared-candidates'].index_select(schedule).to('mps');device_target=target.index_select(schedule).to('mps')
        dev=old.load_inputs('development'); db=old.batches(dev)['shared-candidates']
        probability,predictions,baseline_hashes=old.read_natural_baselines(dev)
        m['input_sha256'].update(dev['hashes']|baseline_hashes)
        prior_models,prior_predictions,prior_hashes=load_prior(dev);predictions.update(prior_predictions);m['input_sha256'].update(prior_hashes)
        regression,reg_hashes=old.synthetic_inputs(None,None)
        m['input_sha256'].update(reg_hashes)
        from .evaluate_synthetic import OBSERVATION_SCENARIOS
        support_path=RUNS/'support-diagnostic-v1/manifest.json'
        scenarios={s['id']:s for s in [*json.loads(support_path.read_text())['scenarios'],*OBSERVATION_SCENARIOS]}
        for item in regression:
            item['batch']=item['batches']['shared-candidates'];item['scenario']=scenarios[item['id']]
        m['status']='fitting';old.save(output/'manifest.json',m);began=time.monotonic();models={}
        for seed in SEEDS:
            initial_identity=None
            for mode,objective in ARMS:
                torch.manual_seed(seed);torch.mps.manual_seed(seed)
                model=HarmonicSelector(mode,objective);model.fit_scaling(tb['shared-candidates'],split='train')
                digest=hashlib.sha256()
                for tensor_name,tensor in sorted(model.state_dict().items()):
                    digest.update(tensor_name.encode());digest.update(tensor.detach().cpu().numpy().tobytes())
                initial_sha=digest.hexdigest()
                if initial_identity is not None and initial_sha!=initial_identity:raise ValueError('Same-seed initial tensors differ across arms')
                initial_identity=initial_sha;model.to('mps')
                optimizer=torch.optim.Adam(model.parameters(),lr=.001)
                losses=[];updates=skipped=0;started=time.monotonic();name=key(mode,objective,seed)
                for epoch in range(12):
                    order=torch.from_numpy(np.random.default_rng(seed+epoch).permutation(len(schedule)).astype(np.int64));total=denominator=0
                    for start in range(0,len(order),4096):
                        if time.monotonic()-began>1800:raise TimeoutError('30 minute fitting limit reached')
                        ix=order[start:start+4096];b=device_batch.index_select(ix);t=device_target.index_select(ix);n=int(t.covered.sum())
                        if not n:skipped+=1;continue
                        optimizer.zero_grad(set_to_none=True);loss=selector_loss(model(b),b.candidate_mask,t,objective=objective)
                        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                        total+=float(loss.detach().cpu())*n;denominator+=n;updates+=1
                    torch.mps.synchronize();losses.append(total/denominator)
                    allocation=torch.mps.current_allocated_memory();m['peak_observed_mps_allocated_bytes']=max(allocation,m['peak_observed_mps_allocated_bytes'])
                    if allocation>8*1024**3:raise MemoryError('8 GiB MPS allocation cap exceeded')
                model=model.cpu().eval();ck=output/'checkpoints'/(name+'.pt')
                torch.save({'context_mode':mode,'config':asdict(model.config),'state_dict':model.state_dict()},ck)
                m['fits'][name]={'seed':seed,'context_mode':mode,'objective':objective,'epoch_losses':losses,'optimizer_updates':updates,'skipped_batches':skipped,
                                'elapsed_seconds':time.monotonic()-started,'parameter_count':sum(p.numel() for p in model.parameters()),'initial_tensors_sha256':initial_sha,'weights_sha256':ev.sha(ck)}
                if updates!=396 or skipped:raise ValueError('Fixed optimizer-update schedule changed')
                models[name]=model;old.save(output/'manifest.json',m);print(json.dumps({'fit':name,**m['fits'][name]}),flush=True)
        m.update(status='checkpoints_frozen',checkpoints_frozen_at=old.now(),fit_elapsed_seconds=time.monotonic()-began)
        check_sources(frozen);old.save(output/'checkpoint-freeze.json',m);old.save(output/'manifest.json',m)
        # Fresh analytic waveforms are generated only after every final checkpoint is immutable.
        from .harmonic_challenge import generate_challenge
        challenge,challenge_hashes=generate_challenge(output/'challenge',output/'checkpoint-freeze.json',ev.sha(output/'checkpoint-freeze.json'))
        m['input_sha256'].update(challenge_hashes)
        for item in challenge:
            f=item['feature'];item['batch']=prepare_selector_batch(f['features'],f['candidate_hz'],f['candidate_mask'],representation='shared-candidates')
            item['probability']=original_probability(f)
        all_models=prior_models|models
        for name,model in models.items():predictions[name]=old.infer(model,db,probability)
        analytic_predictions={group:{item['id']:{name:old.infer(model,item['batch'],item['probability']) for name,model in all_models.items()} for item in items}
                              for group,items in (('regression',regression),('challenge',challenge))}
        hashes={}
        for name,(state,hz) in predictions.items():
            p=output/'predictions'/(name+'.npz');np.savez_compressed(p,state=state,hz=hz);hashes[str(p.relative_to(output))]=ev.sha(p)
        for group,cases in analytic_predictions.items():
            for name,methods in cases.items():
                p=output/'predictions'/(group+'-'+name+'.npz')
                np.savez_compressed(p,**{k+'__state':v[0] for k,v in methods.items()},**{k+'__hz':v[1] for k,v in methods.items()})
                hashes[str(p.relative_to(output))]=ev.sha(p)
        m.update(status='predictions_frozen',predictions_frozen_at=old.now(),prediction_sha256=hashes)
        check_checkpoints(output,m);check_sources(frozen);old.save(output/'prediction-freeze.json',m);old.save(output/'manifest.json',m)
        # No development reference bytes enter this process before this freeze.
        dl=old.load_labels(dev);m['development_references_read']=True
        if dl['hashes']!=prerequisite['reference_sha256']['development']:raise ValueError('Development reference identity changed')
        m['input_sha256'].update(dl['hashes'])
        natural=old.aggregate(dev,dl,predictions)
        reg_scores=analytic_scores(analytic_predictions['regression'],regression);challenge_scores=analytic_scores(analytic_predictions['challenge'],challenge)
        decision=natural_decision(natural);decision['analytic']=analytic_decision(reg_scores,challenge_scores)
        decision['promising_for_next_study']=decision['pass'] and all(decision['analytic'][x] for x in ('stationary_pass','harmonic_change_pass','challenge_pass'))
        result={'schema':'phenometrix-harmonic-context-metrics.v1','scope':'reused natural development and old analytic regression; prospective synthetic challenge, no external speech',
                'natural':natural,'regression':reg_scores,'challenge':challenge_scores,'decision':decision,'exclusions':dl['excluded']}
        old.save(output/'metrics.json',result)
        for name,h in hashes.items():
            if ev.sha(output/name)!=h:raise ValueError('Predictions changed after freeze')
        for name,h in m['input_sha256'].items():
            if ev.sha(name)!=h:raise ValueError('Input changed during experiment')
        check_checkpoints(output,m);check_sources(frozen)
        size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if size>2*1024**3:raise RuntimeError('2 GiB artifact limit exceeded')
        m.update(status='complete',completed_at=old.now(),metrics_sha256=ev.sha(output/'metrics.json'),output_bytes=size)
        old.save(output/'manifest.json',m);print(json.dumps(decision,indent=2),flush=True)
        return result
    except BaseException as e:
        m.update(status='failed',failure=repr(e),failed_at=old.now());old.save(output/'manifest.json',m);raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    run(args.output.resolve())
