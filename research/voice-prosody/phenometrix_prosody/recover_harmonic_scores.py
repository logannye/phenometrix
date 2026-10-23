"""Score-only recovery of a JSON serialization failure; never fit or predict."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from . import harmonic_selector_experiment as experiment
from . import selector_experiment as old
from . import evaluate as ev
from .selector_model import prepare_selector_batch
from .harmonic_challenge import load_challenge
from .evaluate_synthetic import OBSERVATION_SCENARIOS

def json_scalar(value):
    """Normalize scalar representation only; arrays/objects remain rejected."""
    if isinstance(value,np.generic):return value.item()
    raise TypeError(f'Unsupported JSON value: {type(value).__name__}')

def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=json_scalar)+'\n')

def run(run_path,output):
    run_path=run_path.resolve();output=output.resolve()
    if not run_path.is_relative_to(old.RUNS) or not output.is_relative_to(old.RUNS):raise ValueError('Only ignored research runs admitted')
    output.mkdir(parents=True,exist_ok=False)
    frozen=json.loads((run_path/'manifest.json').read_text())
    cf=json.loads((run_path/'checkpoint-freeze.json').read_text())
    pf=json.loads((run_path/'prediction-freeze.json').read_text())
    if frozen['status']!='failed' or frozen['failure']!="TypeError('Object of type bool is not JSON serializable')":raise ValueError('Only the recorded serialization failure is recoverable')
    if (run_path/'metrics.json').exists():raise ValueError('Original metrics artifact unexpectedly exists')
    if cf['status']!='checkpoints_frozen' or pf['status']!='predictions_frozen' or cf['fits']!=pf['fits'] or pf['fits']!=frozen['fits']:raise ValueError('Incomplete or changed prediction/checkpoint freezes')
    experiment.check_sources(frozen['source_sha256']);experiment.check_checkpoints(run_path,frozen)
    inputs={str(run_path/n):ev.sha(run_path/n) for n in ('manifest.json','checkpoint-freeze.json','prediction-freeze.json')}
    for p,h in frozen['input_sha256'].items():
        if ev.sha(p)!=h:raise ValueError('Frozen input changed')
        inputs[p]=h
    for p,h in pf['prediction_sha256'].items():
        if ev.sha(run_path/p)!=h:raise ValueError('Frozen prediction changed')
        inputs[str(run_path/p)]=h
    sources=frozen['source_sha256']|{str(Path(__file__).relative_to(old.REPO)):ev.sha(__file__)}
    receipt={'schema':'phenometrix-harmonic-score-recovery.v1','status':'scoring','started_at':old.now(),
        'original_run':str(run_path.relative_to(old.REPO)),'original_failure':frozen['failure'],
        'checkpoint_freeze_sha256':inputs[str(run_path/'checkpoint-freeze.json')],
        'prediction_freeze_sha256':inputs[str(run_path/'prediction-freeze.json')],
        'original_manifest_sha256':inputs[str(run_path/'manifest.json')],
        'source_sha256':sources,'input_sha256':inputs,
        'change':'JSON default converts NumPy scalar to equivalent Python scalar; scoring equations/gates unchanged',
        'fitting_performed':False,'inference_performed':False,'waveform_generation_performed':False,
        'reserved_content_read':False,'device':'cpu'}
    save(output/'manifest.json',receipt)
    (output/'recover_harmonic_scores.py').write_bytes(Path(__file__).read_bytes())
    try:
        dev=old.load_inputs('development')
        for p,h in dev['hashes'].items():
            if inputs.get(p)!=h:raise ValueError('Development input not bound to failed run')
        names=['original_mlp','original_hybrid',*[f'prior__{old.model_key("shared-candidates",o,s)}' for o in ('nearest','acceptable-set') for s in old.SEEDS],
               *[experiment.key(mode,o,s) for s in old.SEEDS for mode,o in experiment.ARMS]]
        predictions={}
        for name in names:
            with np.load(run_path/'predictions'/(name+'.npz'),allow_pickle=False) as z:predictions[name]=(z['state'].copy(),z['hz'].copy())
        # References are opened only after the already completed all-prediction freeze.
        labels=old.load_labels(dev)
        for p,h in labels['hashes'].items():
            if inputs.get(p)!=h:raise ValueError('Reference identity not bound to failed scoring attempt')
        natural=old.aggregate(dev,labels,predictions)
        regression,_=old.synthetic_inputs(None,None)
        support=json.loads((old.RUNS/'support-diagnostic-v1/manifest.json').read_text())
        scenarios={s['id']:s for s in [*support['scenarios'],*OBSERVATION_SCENARIOS]}
        for item in regression:item['batch']=item['batches']['shared-candidates'];item['scenario']=scenarios[item['id']]
        challenge,_=load_challenge(run_path/'challenge',run_path/'checkpoint-freeze.json',receipt['checkpoint_freeze_sha256'])
        for item in challenge:
            f=item['feature'];item['batch']=prepare_selector_batch(f['features'],f['candidate_hz'],f['candidate_mask'],representation='shared-candidates')
        scores={}
        for group,items in (('regression',regression),('challenge',challenge)):
            case_predictions={}
            for item in items:
                with np.load(run_path/'predictions'/(group+'-'+item['id']+'.npz'),allow_pickle=False) as z:
                    methods=[n for n in names if n not in ('original_mlp','original_hybrid')]
                    if set(z.files)!={n+s for n in methods for s in ('__state','__hz')}:raise ValueError('Incomplete analytic prediction inventory')
                    case_predictions[item['id']]={n:(z[n+'__state'].copy(),z[n+'__hz'].copy()) for n in methods}
            scores[group]=experiment.analytic_scores(case_predictions,items)
        decision=experiment.natural_decision(natural)
        decision['analytic']=experiment.analytic_decision(scores['regression'],scores['challenge'])
        decision['promising_for_next_study']=decision['pass'] and all(decision['analytic'][x] for x in ('stationary_pass','harmonic_change_pass','challenge_pass'))
        result={'schema':'phenometrix-harmonic-context-metrics.v1','scope':'reused natural development and old analytic regression; prospective synthetic challenge, no external speech',
                'natural':natural,**scores,'decision':decision,'exclusions':labels['excluded']}
        save(output/'metrics.json',result)
        for p,h in inputs.items():
            if ev.sha(p)!=h:raise ValueError('Input changed during score recovery')
        experiment.check_sources(frozen['source_sha256']);experiment.check_checkpoints(run_path,frozen)
        for p,h in sources.items():
            if ev.sha(old.REPO/p)!=h:raise ValueError('Source changed during score recovery')
        receipt.update(status='complete',completed_at=old.now(),metrics_sha256=ev.sha(output/'metrics.json'))
        save(output/'manifest.json',receipt)
        print(json.dumps(decision,indent=2,default=json_scalar),flush=True)
        return result
    except BaseException as error:
        receipt.update(status='failed',failure=repr(error),failed_at=old.now());save(output/'manifest.json',receipt);raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.run,a.output)
