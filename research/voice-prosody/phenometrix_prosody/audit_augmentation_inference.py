"""Independent nine-checkpoint CPU replay of the frozen augmentation experiment.

No fitting, target/reference files, analytic truth arrays, challenge specification,
metrics, synthetic training targets/inputs, or waveform generation. Explicit forward/emission equations
are reused from the earlier independent auditor, never the experiment runner.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import gzip
from io import BytesIO
import json
from pathlib import Path
import time
import numpy as np
import torch
from . import audit_harmonic_inference as equations
from .audit_harmonic_inference import (
    Ledger, digest, require, now, original_probability, independent_logits,
    emit, initial_tensor_hash, validate_feature, prepare, compare_arrays,
)
from .harmonic_selector_model import HarmonicSelector

REPO, ROOT, RUNS = equations.REPO, equations.ROOT, equations.RUNS
ORIGINAL_SHA = equations.ORIGINAL_SHA
SEEDS = (20261022, 20261023, 20261024)
ARMS = ('natural-only', 'stationary-aug', 'balance-aug')
MODELS = {f'{arm}__seed{seed}' for arm in ARMS for seed in SEEDS}
PRIOR_MODELS = {f'prior__relative__acceptable-set__seed{seed}' for seed in SEEDS}
REGRESSION_IDS = equations.REGRESSION_IDS
CONSUMED_IDS = {f'harmonic-balance-f{f}-p{p}' for f in range(6) for p in range(2)}
CONSUMED_SUBHARMONIC_IDS = {f'subharmonic-balance-f{f}-p{p}' for f in range(6) for p in range(2)}
CHALLENGE_IDS = {f'augmentation-balance-f{f}-p{p}' for f in range(6) for p in range(2)}
CONFIG = {'representation': 'shared-candidates', 'loss': 'acceptable-set', 'hidden_size': 32}


def validate_checkpoint(name, checkpoint, fit):
    require(name in MODELS, 'Unknown new checkpoint identity')
    arm, seed_text = name.split('__'); seed = int(seed_text.removeprefix('seed'))
    require(checkpoint.get('arm') == arm and fit.get('arm') == arm and fit.get('seed') == seed,
            'Checkpoint arm/seed differs')
    require(checkpoint.get('context_mode') == 'relative' and checkpoint.get('config') == CONFIG,
            'Checkpoint inference architecture differs')
    weights = checkpoint['state_dict']
    model = HarmonicSelector('relative', 'acceptable-set')
    model.load_state_dict(weights, strict=True)
    require(all(bool(torch.isfinite(t).all()) for t in weights.values()) and bool(weights['scaling_fitted']),
            'Invalid checkpoint values')
    require(bool((weights['candidate_scale'] > 0).all()) and bool((weights['global_scale'] > 0).all()),
            'Invalid scaling')
    require(sum(p.numel() for p in model.parameters()) == 3009 and fit['parameter_count'] == 3009,
            'Parameter count differs')
    initial = initial_tensor_hash('relative', 'acceptable-set', seed, weights)
    require(initial == fit['initial_tensors_sha256'], 'Recorded initialization does not reproduce')
    return weights, initial


def analytic_predictions(saved, frames):
    expected = {'original_probability'} | {name+'__'+field for name in MODELS | PRIOR_MODELS for field in ('state','hz')}
    require(set(saved) == expected, 'Analytic prediction/probability inventory differs')
    probability = saved['original_probability']
    require(probability.shape == (frames,) and np.isfinite(probability).all()
            and np.all((probability >= 0) & (probability <= 1)), 'Invalid saved original probability')
    result = {}
    for name in MODELS | PRIOR_MODELS:
        state, hz = saved[name+'__state'], saved[name+'__hz']
        require(state.shape == hz.shape == probability.shape and np.isin(state,[0,1,2]).all(),
                'Analytic output dimensions/states differ')
        require(np.array_equal(np.isfinite(hz), state == 1), 'Analytic frequency availability differs')
        require(np.array_equal(state == 0, probability < .5), 'Saved analytic original voice gate differs')
        if name in MODELS:
            result[name] = (state, hz)
    return probability, result


FEATURE_FIELDS = {'features', 'candidate_hz', 'candidate_mask', 'reference_sample',
                  'feature_names', 'sample_rate_hz', 'pcm_sample_count'}


def load_feature_bytes(content):
    """Reject target-bearing archives before decoding arrays for inference."""
    with np.load(BytesIO(content), allow_pickle=False) as z:
        require(set(z.files) == FEATURE_FIELDS, 'Unexpected feature or analytic target field')
        feature = {k:z[k].copy() for k in z.files}
    validate_feature(feature)
    return feature


def analytic_domains(run):
    return [('regression',REGRESSION_IDS,RUNS/'delayed-known-truth',None),
            ('consumed_challenge',CONSUMED_IDS,RUNS/'harmonic-context-v1/challenge',RUNS/'harmonic-context-v1/checkpoint-freeze.json'),
            ('consumed_subharmonic',CONSUMED_SUBHARMONIC_IDS,RUNS/'subharmonic-objective-v1/challenge',RUNS/'subharmonic-objective-v1/checkpoint-freeze.json'),
            ('challenge',CHALLENGE_IDS,run/'challenge',run/'checkpoint-freeze.json')]


def validate_phase_freezes(source, checkpoints, frozen, manifest):
    require(manifest['schema']=='phenometrix-augmentation-experiment.v1' and manifest['status']=='complete',
            'Wait for the completed frozen augmentation run')
    require(source['status']=='sources_frozen' and checkpoints['status']=='checkpoints_frozen'
            and frozen['status']=='predictions_frozen', 'Incomplete required freezes')
    require(source['fits']=={} and set(frozen['fits'])==set(checkpoints['fits'])==MODELS
            and frozen['fits']==checkpoints['fits']==manifest['fits'], 'Checkpoint inventory differs')
    require(source['source_sha256']==frozen['source_sha256']==checkpoints['source_sha256']==manifest['source_sha256'],
            'Source identities differ across phases')
    require(frozen['prediction_sha256']==manifest['prediction_sha256'], 'Prediction identities changed')
    require(frozen['development_references_read'] is False and frozen['reserved_content_read'] is False,
            'Prediction phase label boundary differs')
    for field in ('training_spec_sha256','challenge_spec_sha256','sources_frozen_at'):
        require(source[field]==checkpoints[field]==frozen[field]==manifest[field], 'Source/specification freeze differs')
    require(datetime.fromisoformat(source['sources_frozen_at'])<=datetime.fromisoformat(checkpoints['checkpoints_frozen_at'])
            <=datetime.fromisoformat(frozen['predictions_frozen_at'])<=datetime.fromisoformat(manifest['completed_at']),
            'Freeze chronology differs')


def validate_cache_binding(cache, group, expected_checkpoint, expected_sha, frozen, checkpoints):
    """Input identity/chronology only; no challenge specification is opened."""
    if group=='challenge':
        require(cache['schema']=='phenometrix-paired-harmonic-augmentation-cache-v1'
                and cache['split']=='prospective-challenge' and cache['freeze_kind']=='checkpoint'
                and cache['freeze_path']==str(expected_checkpoint.relative_to(REPO))
                and cache['freeze_sha256']==expected_sha, 'Augmentation challenge checkpoint binding differs')
        require(cache['spec_sha256']==cache['challenge_spec_sha256']==frozen['challenge_spec_sha256']
                and cache['training_spec_sha256']==frozen['training_spec_sha256'], 'Prospective specification identity differs')
        require(datetime.fromisoformat(checkpoints['checkpoints_frozen_at'])<=datetime.fromisoformat(cache['started_at'])
                <=datetime.fromisoformat(cache['completed_at'])<=datetime.fromisoformat(frozen['predictions_frozen_at']),
                'Prospective generation freeze differs')
    require(cache['checkpoint_freeze_sha256']==expected_sha
            and cache['checkpoint_freeze_path']==str(expected_checkpoint.relative_to(REPO)),
            'Challenge checkpoint alias binding differs')
    require(cache['reference_content_read'] is False, 'Unexpected reference-bearing cache')


def audit(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    require(run.is_relative_to(RUNS) and output.is_relative_to(RUNS), 'Artifacts must remain in ignored runs')
    ledger_path = output.with_name(output.stem+'-inputs.json')
    require(not output.exists() and not ledger_path.exists(), 'Audit refuses overwrite')
    ledger = Ledger(); began = time.monotonic(); torch.set_num_threads(1)
    result = {'schema':'phenometrix-augmentation-checkpoint-inference-audit.v1', 'status':'running', 'started_at':now(),
        'device':'cpu', 'torch':torch.__version__, 'numpy':np.__version__, 'fitting_performed':False,
        'reference_files_or_analytic_target_arrays_read':False, 'truth_used_for_comparison':False,
        'challenge_spec_or_metrics_content_opened':False, 'reserved_content_read':False, 'new_gpu_work':False,
        'waveform_generation_performed':False, 'synthetic_training_inputs_or_targets_read':False,
        'inference_implementation':'explicit Torch equations from audit_harmonic_inference; no runner infer or HarmonicSelector.forward',
        'preprocessing_scope':'source-bound shared grouping reused, with independent original-member validation',
        'comparisons':{}, 'original_voicing_checks':{}, 'initialization_checks':{}}
    try:
        manifest_path = run/'manifest.json'; checkpoint_path = run/'checkpoint-freeze.json'; prediction_path = run/'prediction-freeze.json'
        source_path = run/'source-freeze.json'
        manifest = ledger.json(manifest_path,'manifest')
        source = ledger.json(source_path,'freeze')
        checkpoints = ledger.json(checkpoint_path,'freeze'); frozen = ledger.json(prediction_path,'freeze')
        validate_phase_freezes(source,checkpoints,frozen,manifest)
        result['identities'] = {name:digest(ledger.read(path,role)) for name,path,role in
            [('manifest_sha256',manifest_path,'manifest'),('source_freeze_sha256',source_path,'freeze'),
             ('checkpoint_freeze_sha256',checkpoint_path,'freeze'),('prediction_freeze_sha256',prediction_path,'freeze')]}
        for name, sha in frozen['source_sha256'].items():
            ledger.read(REPO/name,'source',sha); ledger.read(run/'source-snapshot'/name,'source_snapshot',sha)
        ledger.read(Path(__file__),'source'); ledger.read(Path(equations.__file__),'source')
        inputs = frozen['input_sha256']
        def bound(path,role,expected=None):
            path = Path(path).resolve()
            require(str(path) in inputs, 'Input absent from prediction-freeze ledger: '+str(path))
            require(expected is None or expected == inputs[str(path)], 'Nested input hashes disagree')
            return ledger.read(path,role,inputs[str(path)])
        models = {}
        for name in sorted(MODELS):
            fit = frozen['fits'][name]
            checkpoint = torch.load(BytesIO(ledger.read(run/'checkpoints'/(name+'.pt'),'weights',fit['weights_sha256'])),
                                    map_location='cpu',weights_only=True)
            weights, initial = validate_checkpoint(name,checkpoint,fit)
            models[name] = weights; result['initialization_checks'][name] = initial
        for seed in SEEDS:
            require(len({result['initialization_checks'][name] for name in MODELS if name.endswith(str(seed))}) == 1,
                    'Initial tensors differ within seed')
        original = torch.load(BytesIO(bound(RUNS/'delayed-mlp-seed20261022/weights.pt','weights',ORIGINAL_SHA)),
                              map_location='cpu',weights_only=True)
        require(original['config']['kind'] == 'framewise-mlp' and frozen['frozen_original_weights_sha256'] == ORIGINAL_SHA,
                'Original voice identity differs')
        original_weights = original['state_dict']
        def probability_check(identity,feature,saved_probability):
            with torch.inference_mode():
                p = np.concatenate([original_probability(original_weights,feature['features'][i:i+4096])
                                    for i in range(0,len(feature['features']),4096)])
            require(saved_probability.shape == p.shape and np.isfinite(p).all() and np.isfinite(saved_probability).all()
                    and np.all((saved_probability>=0)&(saved_probability<=1)), 'Original probability extent/range differs')
            report = {'frames':len(p), 'recomputed_sha256_float64':digest(p.astype('<f8').tobytes()),
                      'frozen_sha256_float64':digest(saved_probability.astype('<f8').tobytes()),
                      'max_absolute_difference':float(np.abs(p-saved_probability).max()),
                      'gate_mismatches':int(((p>=.5)!=(saved_probability>=.5)).sum())}
            require(report['max_absolute_difference'] <= 1e-6 and report['gate_mismatches'] == 0,
                    'Original voicing parity failed')
            result['original_voicing_checks'][identity] = report
            return saved_probability
        def verify_dataset(identity,batch,probability,saved_by_model):
            reports = {}
            for name,weights in models.items():
                states, frequencies = [], []
                with torch.inference_mode():
                    for start in range(0,len(batch.features),4096):
                        stop = min(start+4096,len(batch.features))
                        logits = independent_logits(weights,'relative',batch.features[start:stop],batch.candidate_mask[start:stop])
                        state,hz = emit(logits,batch.candidate_hz[start:stop].numpy(),batch.candidate_mask[start:stop].numpy(),probability[start:stop])
                        states.append(state); frequencies.append(hz)
                reports[name] = compare_arrays(saved_by_model[name],np.concatenate(states),np.concatenate(frequencies),batch,probability)
            result['comparisons'][identity] = reports
            print(json.dumps({'dataset':identity,'frames_per_model':len(batch.features),'models':len(reports),
                  'state_or_hz_mismatches':sum(r['state_mismatches']+r['hz_mismatches'] for r in reports.values())}),flush=True)
        dev_root = RUNS/'delayed-development-features'
        dev_manifest = json.loads(bound(dev_root/'manifest.json','manifest'))
        require(dev_manifest['status']=='complete' and dev_manifest['split']=='development' and len(dev_manifest['recordings'])==320,
                'Development input inventory differs')
        require({r['speaker'] for r in dev_manifest['recordings']}=={'F01','F03','M01','M03'}, 'Nondevelopment speaker')
        baseline_root = RUNS/'hybrid-natural-v1'
        baseline = json.loads(bound(baseline_root/'prediction-freeze.json','freeze'))
        baseline_rows = {r['recording_id']:r for r in baseline['recordings']}
        pieces, probabilities = [], []
        for record in dev_manifest['recordings']:
            identity = record['recording_id']
            feature = load_feature_bytes(bound(dev_root/'features'/(identity+'.npz'),'feature',record['feature_sha256']))
            pieces.append(feature)
            row = baseline_rows[identity]
            payload = json.loads(gzip.decompress(bound(baseline_root/row['hybrid_prediction_path'],'prediction',row['hybrid_prediction_sha256'])))
            old_rows = payload['predictions']['mlp_original']
            require(np.array_equal([r['sample'] for r in old_rows],feature['reference_sample']-960), 'Frozen voice target grid differs')
            probabilities.append(np.asarray([r['voicing_probability'] for r in old_rows],np.float64))
        raw = {k:np.concatenate([f[k] for f in pieces]) for k in ('features','candidate_hz','candidate_mask')}
        require(len(raw['features'])==255532, 'Development acquired frame count differs')
        probability = probability_check('development',raw,np.concatenate(probabilities)); batch = prepare(raw)
        saved = {}
        for name in MODELS:
            relative = 'predictions/'+name+'.npz'
            z = ledger.npz(run/relative,'prediction',frozen['prediction_sha256'][relative])
            require(set(z)=={'state','hz'}, 'Unexpected natural prediction fields')
            saved[name] = (z['state'],z['hz'])
        verify_dataset('development',batch,probability,saved)
        # These manifests contain input identities only; no challenge spec or
        # analytic frequency/reference arrays are opened or passed to equations.
        domains = analytic_domains(run)
        for group, identities, folder, expected_checkpoint in domains:
            records = None
            if group!='regression':
                cache = json.loads(bound(folder/'manifest.json','manifest'))
                require(cache['status']=='complete' and len(cache['records'])==12
                        and {r['id'] for r in cache['records']}==identities, 'Challenge cache inventory differs')
                expected_sha = result['identities']['checkpoint_freeze_sha256'] if group=='challenge' else inputs[str(expected_checkpoint)]
                validate_cache_binding(cache,group,expected_checkpoint,expected_sha,frozen,checkpoints)
                records = {r['id']:r for r in cache['records']}
            for identity in sorted(identities):
                if records is None:
                    content = bound(folder/'features'/(identity+'.npz'),'feature')
                else:
                    record = records[identity]
                    pcm = bound(folder/record['pcm_path'],'pcm',record['pcm_sha256'])
                    require(len(pcm)==96000*4, 'Challenge PCM extent differs')
                    content = bound(folder/record['feature_path'],'feature',record['feature_sha256'])
                feature = load_feature_bytes(content)
                if records is not None:
                    require(len(feature['features'])==record['feature_frames'], 'Challenge feature extent differs')
                relative = 'predictions/'+group+'-'+identity+'.npz'
                z = ledger.npz(run/relative,'prediction',frozen['prediction_sha256'][relative])
                old_p,saved = analytic_predictions(z,len(feature['features']))
                p = probability_check(group+'/'+identity,feature,old_p)
                verify_dataset(group+'/'+identity,prepare(feature),p,saved)
        totals = {name:sum(r[name] for domain in result['comparisons'].values() for r in domain.values())
                  for name in ('frames','state_mismatches','hz_mismatches','frozen_voice_gate_mismatches','nonmember_emissions')}
        result.update(totals=totals,model_count=9,analytic_case_count=50,development_frames_per_model=255532)
        require(len(result['comparisons'])==51, 'Incomplete inference domain inventory')
        for name,binding in ledger.files.items():
            require(digest((REPO/name).read_bytes())==binding['sha256'], 'Input bytes changed before audit receipt')
        result.update(status='complete',input_bytes_stable=True,
            pass_result=all(totals[k]==0 for k in ('state_mismatches','hz_mismatches','frozen_voice_gate_mismatches','nonmember_emissions')))
    except BaseException as error:
        result.update(status='failed',pass_result=False,failure=repr(error))
    finally:
        output.parent.mkdir(parents=True,exist_ok=True)
        ledger_bytes=(json.dumps(ledger.files,sort_keys=True,indent=2)+'\n').encode(); ledger_path.write_bytes(ledger_bytes)
        result.update(completed_at=now(),elapsed_seconds=time.monotonic()-began,input_file_count=len(ledger.files),
                      input_ledger_sha256=digest(ledger_bytes),audit_source_sha256=digest(Path(__file__).read_bytes()))
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'comparisons','original_voicing_checks','initialization_checks'}},indent=2))
    if not result['pass_result']: raise SystemExit(1)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); audit(args.run,args.output)
