"""Target-free CPU reproduction of the bounded expanded-pool experiment.

Explicit forward/emission equations, with source-bound grouping and independent
member checks. No runner inference, model.forward, references, analytic labels,
scenario inventories, challenge specification, metrics or training arrays.
"""
from __future__ import annotations
import argparse
from datetime import datetime
from io import BytesIO
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F

from . import audit_harmonic_inference as equations
from . import audit_augmentation_inference as historical
from . import audit_candidate_coverage_inference as preceding
from . import audit_expanded_pool as pool_checks
from .expanded_selector_model import prepare_expanded_selector_batch
from .harmonic_selector_model import HarmonicSelector
from .audit_harmonic_inference import Ledger, digest, require, now, emit, compare_arrays

REPO, ROOT, RUNS = equations.REPO, equations.ROOT, equations.RUNS
SEEDS = (20261022, 20261023, 20261024)
ARMS = ('original6', 'expanded7')
PIPELINES = ARMS + ('frozen-original6', 'frozen-expanded7')
METHODS = {f'{arm}__seed{s}' for arm in PIPELINES for s in SEEDS}
NEW_MODELS = {f'{arm}__seed{s}' for arm in ARMS for s in SEEDS}
OLD_MODELS = {f'balance-aug__seed{s}' for s in SEEDS}
POLICY = 'add-shortest-admitted-full40-v1'
LAST = RUNS / 'candidate-coverage-v1'
LAST_SHA = '1e2d7fbec0cb2dd71b177a0a5b646eff04d1d6dcf72a1c86eaac8f42f267aeff'
LAST_PRED_SHA = 'fd0bbe50c3e4be845484c2e77ef47cb538fdee1d2b978e2543a5269079f4b89b'
PRIOR = RUNS / 'harmonic-augmentation-v1'
PRIOR_SHA = preceding.PRIOR_MANIFEST_SHA
GROUP_IDS = {**{k:v for k,v in preceding.GROUP_IDS.items() if k != 'challenge'},
    'consumed_coverage': preceding.GROUP_IDS['challenge'],
    'challenge': {f'expanded-balance-f{i}-p{j}' for i in range(6) for j in range(2)}}
FEATURE_FIELDS = historical.FEATURE_FIELDS
CONFIG = historical.CONFIG
HISTORICAL_MANIFEST_PATHS = frozenset((RUNS/'harmonic-context-v1/challenge/manifest.json',
    RUNS/'subharmonic-objective-v1/challenge/manifest.json', PRIOR/'challenge/manifest.json',
    LAST/'challenge/manifest.json'))


def historical_bindings(prior, last, source, frozen, manifest):
    """Reconstruct the inherited map from the two exact pinned old manifests."""
    combined = {}
    for payload in (prior, last):
        for path, sha in payload['input_sha256'].items():
            p = Path(path)
            if p.is_absolute():
                require(p.is_relative_to(REPO), 'Historical identity escaped repository')
                name = str(p.relative_to(REPO))
            else:
                require('..' not in p.parts, 'Historical identity escaped repository')
                name = str(p)
            require(name not in combined or combined[name] == sha, 'Historical identity maps disagree')
            combined[name] = sha
    for payload in (source, frozen, manifest):
        require(payload['historical_input_sha256'] == combined, 'Historical input inventory changed')
    return combined


def historical_manifest_bytes(ledger, path, current, inherited):
    """Allow only four named consumed input manifests through the inherited map."""
    path = Path(path).resolve()
    require(path in HISTORICAL_MANIFEST_PATHS, 'Manifest is not an allowed consumed input cache')
    name = str(path.relative_to(REPO))
    if name in current:
        require(name not in inherited or current[name] == inherited[name], 'Current/historical identity differs')
        sha = current[name]
    else:
        require(name in inherited, 'Consumed manifest is absent from inherited identity map')
        sha = inherited[name]
    return ledger.read(path, 'manifest', sha)


def new_checkpoint_bytes(ledger, run, name, row, fit):
    """Final checkpoints are authorized by their exact final-freeze records."""
    require(name in NEW_MODELS, 'Unknown new checkpoint identity')
    path = Path(run).resolve()/'checkpoints'/(name+'.pt')
    require(path.is_relative_to(RUNS) and row['path'] == str(path.relative_to(REPO))
        and row['sha256'] == fit['weights_sha256'], 'Final checkpoint path/hash differs')
    return ledger.read(path, 'weights', row['sha256'])


def validate_freezes(source, checkpoints, frozen, manifest):
    require(source['schema'] == 'phenometrix-expanded-pool-source-freeze.v1'
        and source['status'] == 'sources_frozen' and source['fits'] == {}, 'Invalid initial source freeze')
    require(checkpoints['schema'] == 'phenometrix-expanded-pool-checkpoint-freeze.v1'
        and checkpoints['status'] == 'checkpoints_frozen', 'Incomplete final checkpoint freeze')
    require(frozen['schema'] == manifest['schema'] == 'phenometrix-expanded-pool-experiment.v1'
        and frozen['status'] == 'predictions_frozen' and manifest['status'] == 'complete', 'Wait for completed run')
    for field in ('source_sha256', 'challenge_spec_sha256', 'arms', 'seeds', 'maximum_fits'):
        require(source[field] == checkpoints[field] == frozen[field] == manifest[field], 'Phase identity changed: ' + field)
    for field in ('historical_checkpoints', 'candidate_policy', 'sources_frozen_at'):
        require(source[field] == frozen[field] == manifest[field], 'Source identity changed: ' + field)
    require(source['arms'] == list(ARMS) and source['seeds'] == list(SEEDS)
        and source['maximum_fits'] == 6 and source['candidate_policy'] == POLICY, 'Declared experiment differs')
    require(set(source['historical_checkpoints']) == OLD_MODELS and set(checkpoints['checkpoints']) == NEW_MODELS
        and set(frozen['fits']) == NEW_MODELS and frozen['fits'] == manifest['fits'], 'Checkpoint inventory changed')
    for name in NEW_MODELS:
        require(checkpoints['checkpoints'][name]['sha256'] == frozen['fits'][name]['weights_sha256'], 'Final weight hashes differ')
    require(frozen['prediction_sha256'] == manifest['prediction_sha256'], 'Frozen prediction hashes differ')
    require(source['development_references_read_for_prerequisites'] is False
        and frozen['development_references_read_for_prerequisites'] is True
        and source['development_references_read_for_final_scoring'] is False
        and frozen['development_references_read_for_final_scoring'] is False
        and manifest['development_references_read_for_final_scoring'] is True, 'Declared consumed-DEV phase boundary differs')
    require(all(p['reserved_content_read'] is False for p in (source, frozen, manifest)), 'Reserved content excluded')
    timestamps = [source['sources_frozen_at'], frozen['prerequisite_references_read_at'], frozen['training_frozen_at'],
                  checkpoints['frozen_at'], frozen['predictions_frozen_at'], manifest['final_scoring_reference_read_at'], manifest['completed_at']]
    require(all(a <= b for a,b in zip(map(datetime.fromisoformat,timestamps), list(map(datetime.fromisoformat,timestamps))[1:])),
        'Source/train/checkpoint/prediction/scoring chronology differs')
    require(checkpoints['frozen_at'] == frozen['checkpoints_frozen_at'], 'Checkpoint timestamp differs')


def load_feature_bytes(content):
    # Read archive keys before decoding arrays, so target-bearing NPZs are refused.
    with np.load(BytesIO(content), allow_pickle=False) as z:
        require(set(z.files) == FEATURE_FIELDS, 'Unexpected feature or target array')
        feature = {k:z[k].copy() for k in z.files}
    k = feature['candidate_hz'].shape[-1]
    pool_checks.validate_pool(feature, k)
    return feature


def validate_arm_inputs(original, expanded):
    for field in ('reference_sample', 'sample_rate_hz', 'pcm_sample_count'):
        require(np.array_equal(original[field], expanded[field]), 'Acquisition metadata changed: ' + field)
    n = len(original['features'])
    require(expanded['features'].shape == (n, 65) and original['features'].shape == (n, 57), 'Paired pool shape differs')
    globals_kept = [0,1,2,3,4,5,6,8]
    require(np.array_equal(original['features'][:,48:][:,globals_kept], expanded['features'][:,56:][:,globals_kept]),
        'Noncount global acquisition features changed')
    a = original['features'][:,:48].reshape(n,6,8)
    b = expanded['features'][:,:56].reshape(n,7,8)
    for j in range(6):
        old_valid = original['candidate_mask'][:,j]
        match = expanded['candidate_mask'] & (expanded['candidate_hz'] == original['candidate_hz'][:,j,None])
        require(np.array_equal(match.sum(-1), old_valid.astype(np.int64)), 'Original raw Hz is missing or duplicated')
        ix = np.flatnonzero(old_valid); destination = match[ix].argmax(-1)
        require(np.array_equal(a[ix,j][:,[0,1,2,3,4,6]], b[ix,destination][:,[0,1,2,3,4,6]])
            and np.all(b[ix,destination,5] >= a[ix,j,5]) and np.all(b[ix,destination,7] >= a[ix,j,7]),
            'Expansion weakened original source evidence')


def prepare(raw):
    batch = prepare_expanded_selector_batch(raw['features'],raw['candidate_hz'],raw['candidate_mask'],representation='shared-candidates')
    k = batch.candidate_mask.shape[-1]
    for j in range(k):
        members = batch.member_mask[:,j].numpy()
        equal = members & raw['candidate_mask'] & (raw['candidate_hz'] == batch.candidate_hz[:,j,None].numpy())
        require(np.array_equal(equal.any(-1), batch.candidate_mask[:,j].numpy()), 'Grouping invented an emitted frequency')
    require(np.array_equal(batch.member_mask.numpy().sum(1), raw['candidate_mask'].astype(np.int64)),
        'Grouping omitted or duplicated a raw member')
    return batch


def independent_logits(weights, features, mask):
    """Dynamic explicit CPU equations, independent of every model.forward call."""
    x = torch.as_tensor(features, dtype=torch.float32)
    valid = torch.as_tensor(mask, dtype=torch.bool)
    k = valid.shape[-1]
    require(k in (6,7) and x.ndim == valid.ndim == 2 and x.shape == (len(valid),8*k+9), 'Inference pool shape differs')
    raw = x[:,:8*k].reshape(-1,k,8)
    z = (raw-weights['candidate_mean'])/weights['candidate_scale']
    embedding = torch.tanh(F.linear(z,weights['candidate_encoder.weight'],weights['candidate_encoder.bias']))
    ratios = (raw[:,None,:,1]-raw[:,:,None,1])/math.log2(14.)
    edges = valid[:,:,None] & valid[:,None,:]
    neighbors = z[:,None,:,:].expand(-1,k,-1,-1)
    messages_in = torch.cat((neighbors,ratios[...,None]),dim=3)
    messages_in = torch.where(edges[...,None],messages_in,torch.zeros_like(messages_in))
    messages = torch.tanh(F.linear(messages_in,weights['message_encoder.weight'],weights['message_encoder.bias']))
    context = torch.amax(torch.where(edges[...,None],messages,torch.full_like(messages,-torch.inf)),dim=2)
    context = torch.where(valid[...,None],context,torch.zeros_like(context))
    globals_ = ((x[:,8*k:]-weights['global_mean'])/weights['global_scale'])[:,None,:].expand(-1,k,-1)
    hidden = torch.tanh(F.linear(torch.cat((embedding,context,globals_),dim=2),weights['shared_hidden.weight'],weights['shared_hidden.bias']))
    logits = F.linear(hidden,weights['shared_head.weight'],weights['shared_head.bias']).squeeze(-1)
    logits = torch.where(valid,logits,torch.full_like(logits,-1e9))
    require(bool(torch.isfinite(logits).all()), 'Nonfinite independent selector logits')
    return logits


def validate_checkpoint(name, checkpoint, fit):
    require(name in NEW_MODELS, 'Unknown new checkpoint identity')
    arm, seed_text = name.split('__'); seed = int(seed_text.removeprefix('seed'))
    require(checkpoint['arm'] == fit['arm'] == arm and fit['seed'] == seed
        and checkpoint['candidate_capacity'] == (6 if arm == 'original6' else 7), 'Checkpoint arm/capacity/seed differs')
    require(checkpoint['config'] == CONFIG and checkpoint['context_mode'] == 'relative', 'Checkpoint model config differs')
    weights = checkpoint['state_dict']; shape_model = HarmonicSelector('relative','acceptable-set')
    shape_model.load_state_dict(weights,strict=True)
    require(sum(p.numel() for p in shape_model.parameters()) == fit['parameter_count'] == 3009, 'Parameter count differs')
    require(all(torch.isfinite(w).all().item() for w in weights.values()) and bool(weights['scaling_fitted'])
        and bool((weights['candidate_scale']>0).all()) and bool((weights['global_scale']>0).all()), 'Invalid checkpoint/scaling values')
    initial = equations.initial_tensor_hash('relative','acceptable-set',seed,weights)
    require(initial == fit['initial_tensors_sha256'], 'Recorded initial tensors do not reproduce')
    return weights, initial


def analytic_predictions(saved, frames):
    expected = {'original_probability'} | {n+'__'+field for n in METHODS for field in ('state','hz')}
    require(set(saved) == expected, 'Analytic pipeline inventory differs')
    p = saved['original_probability']
    require(p.shape == (frames,) and np.isfinite(p).all() and np.all((p>=0)&(p<=1)), 'Saved original probability differs')
    result = {}
    for name in METHODS:
        state,hz = saved[name+'__state'],saved[name+'__hz']
        require(state.shape == hz.shape == p.shape and state.dtype == np.uint8 and hz.dtype == np.float64
            and np.isin(state,[0,1,2]).all(), 'Prediction state/extent/dtype differs')
        require(np.array_equal(np.isfinite(hz),state==1) and np.isnan(hz[state!=1]).all(), 'Frequency availability differs')
        require(np.array_equal(state==0,p<.5), 'Original voicing threshold changed')
        result[name] = state,hz
    return p,result


def probability_check(weights, original, saved):
    require(original['features'].ndim == 2 and original['features'].shape[1] == 57, 'Original voicing requires original57 input')
    with torch.inference_mode():
        actual = np.concatenate([equations.original_probability(weights,original['features'][i:i+4096])
                                 for i in range(0,len(original['features']),4096)])
    require(saved.shape == actual.shape and np.isfinite(saved).all() and np.isfinite(actual).all()
        and np.all((saved>=0)&(saved<=1)), 'Original probability extent/finiteness differs')
    maximum = float(np.abs(saved-actual).max()); mismatch = int(np.count_nonzero((saved>=.5)!=(actual>=.5)))
    require(maximum <= 1e-6 and mismatch == 0, 'Original voicing parity failed')
    return {'frames':len(saved),'max_absolute_difference':maximum,'gate_mismatches':mismatch,
            'saved_sha256_float64':digest(saved.astype('<f8').tobytes()),'recomputed_sha256_float64':digest(actual.astype('<f8').tobytes())}


def audit(run, output):
    run,output = Path(run).resolve(),Path(output).resolve()
    require(run.is_relative_to(RUNS) and output.is_relative_to(RUNS), 'Ignored local artifacts required')
    ledger_path = output.with_name(output.stem+'-inputs.json')
    require(not output.exists() and not ledger_path.exists(), 'Refuses overwrite')
    ledger = Ledger(); began = time.monotonic(); torch.set_num_threads(1)
    result = {'schema':'phenometrix-expanded-pool-inference-audit.v1','status':'running','started_at':now(),
        'device':'cpu','torch':torch.__version__,'numpy':np.__version__,'fitting_performed':False,
        'reference_files_or_analytic_target_arrays_read':False,'truth_used_for_comparison':False,
        'challenge_spec_metrics_or_analytic_inventory_content_opened':False,'reserved_content_read':False,
        'training_features_or_targets_read':False,'new_gpu_work':False,'waveform_generation_performed':False,
        'inference_implementation':'independent explicit dynamic Torch equations; no runner infer or model.forward',
        'preprocessing_scope':'source-bound grouping plus independent every-member/emitted-member validation',
        'voicing_contract':'frozen original MLP on original57 inputs only for all12 pipelines',
        'comparisons':{},'original_voicing_checks':{},'initialization_checks':{},'prior_control_checks':{}}
    try:
        manifest = ledger.json(run/'manifest.json','manifest')
        source = ledger.json(run/'source-freeze.json','freeze')
        checkpoint_freeze = ledger.json(run/'checkpoint-freeze.json','freeze')
        frozen = ledger.json(run/'prediction-freeze.json','freeze')
        validate_freezes(source,checkpoint_freeze,frozen,manifest)
        result['identities'] = {p:digest(ledger.read(run/p,'manifest' if p=='manifest.json' else 'freeze'))
            for p in ('manifest.json','source-freeze.json','checkpoint-freeze.json','prediction-freeze.json')}
        require(checkpoint_freeze['source_freeze_path'] == str((run/'source-freeze.json').relative_to(REPO))
            and checkpoint_freeze['source_freeze_sha256'] == result['identities']['source-freeze.json'], 'Linked source freeze differs')
        for name,h in frozen['source_sha256'].items():
            ledger.read(REPO/name,'source',h); ledger.read(run/'source-snapshot'/name,'source_snapshot',h)
        for p in (Path(__file__),Path(equations.__file__),Path(historical.__file__),Path(preceding.__file__),
                  Path(pool_checks.__file__),ROOT/'tests/test_audit_expanded_pool_inference.py'):
            ledger.read(p,'source')
        inputs = frozen['input_sha256']
        def bound(path,role,expected=None):
            path = Path(path).resolve(); name = str(path.relative_to(REPO))
            require(name in inputs, 'Input absent from prediction freeze: '+name)
            require(expected is None or expected==inputs[name], 'Nested input identity differs')
            return ledger.read(path,role,inputs[name])
        def feature(path,expected=None): return load_feature_bytes(bound(path,'feature',expected))
        def saved(relative): return ledger.npz(run/relative,'prediction',frozen['prediction_sha256'][relative])
        last = json.loads(bound(LAST/'manifest.json','manifest',LAST_SHA))
        last_frozen = json.loads(bound(LAST/'prediction-freeze.json','freeze',LAST_PRED_SHA))
        prior = json.loads(bound(PRIOR/'manifest.json','manifest',PRIOR_SHA))
        require(last['status'] == prior['status'] == 'complete'
            and last['prediction_sha256'] == last_frozen['prediction_sha256']
            and source['historical_checkpoints'] == last['checkpoints'], 'Historical completed inputs differ')
        inherited = historical_bindings(prior,last,source,frozen,manifest)
        models = {}; historical_models = {}
        for name in sorted(NEW_MODELS):
            row = checkpoint_freeze['checkpoints'][name]; path = run/'checkpoints'/(name+'.pt')
            require(row['path'] == str(path.relative_to(REPO)), 'Checkpoint path differs')
            ck = torch.load(BytesIO(new_checkpoint_bytes(ledger,run,name,row,frozen['fits'][name])),map_location='cpu',weights_only=True)
            w,initial = validate_checkpoint(name,ck,frozen['fits'][name]); models[name] = w
            result['initialization_checks'][name] = initial
        for seed in SEEDS:
            name = f'balance-aug__seed{seed}'; row = source['historical_checkpoints'][name]
            require(row['path'] == str((PRIOR/'checkpoints'/(name+'.pt')).relative_to(REPO))
                and row['sha256'] == prior['fits'][name]['weights_sha256'], 'Historical checkpoint differs')
            ck = torch.load(BytesIO(bound(REPO/row['path'],'weights',row['sha256'])),map_location='cpu',weights_only=True)
            w,initial = historical.validate_checkpoint(name,ck,prior['fits'][name]); historical_models[seed] = w
            result['initialization_checks'][name] = initial
            require(result['initialization_checks'][f'original6__seed{seed}'] == result['initialization_checks'][f'expanded7__seed{seed}'],
                'Paired initial parameter/scaling tensors differ')
            for arm in ARMS:
                for field in equations.SCALING_BUFFERS:
                    require(torch.equal(models[f'{arm}__seed{seed}'][field],w[field]), 'Original-six TRAIN scaling changed: '+field)
        voice = torch.load(BytesIO(bound(RUNS/'delayed-mlp-seed20261022/weights.pt','weights',equations.ORIGINAL_SHA)),
                           map_location='cpu',weights_only=True)
        require(voice['config']['kind']=='framewise-mlp', 'Original voice model differs')
        def verify(identity,raws,probability,predictions):
            result['original_voicing_checks'][identity] = probability_check(voice['state_dict'],raws['original6'],probability)
            reports = {}
            for arm in ARMS:
                batch = prepare(raws[arm])
                for seed in SEEDS:
                    for prefix,weights in [('',models[f'{arm}__seed{seed}']),('frozen-',historical_models[seed])]:
                        state_rows,hz_rows = [],[]
                        with torch.inference_mode():
                            for start in range(0,len(batch.features),4096):
                                stop = min(start+4096,len(batch.features))
                                logits = independent_logits(weights,batch.features[start:stop],batch.candidate_mask[start:stop])
                                s,h = emit(logits,batch.candidate_hz[start:stop].numpy(),batch.candidate_mask[start:stop].numpy(),probability[start:stop])
                                state_rows.append(s); hz_rows.append(h)
                        name = f'{prefix}{arm}__seed{seed}'
                        reports[name] = compare_arrays(predictions[name],np.concatenate(state_rows),np.concatenate(hz_rows),batch,probability)
            result['comparisons'][identity] = reports
            print(json.dumps({'dataset':identity,'methods':len(reports),'frames_per_method':len(probability)}),flush=True)
        original_cache = RUNS/'delayed-development-features'
        original_manifest = json.loads(bound(original_cache/'manifest.json','manifest'))
        expanded_manifest = json.loads(bound(run/'development/manifest.json','manifest'))
        require(original_manifest['status']==expanded_manifest['status']=='complete'
            and original_manifest['split']==expanded_manifest['split']=='development'
            and expanded_manifest['candidate_policy']==POLICY and expanded_manifest['reference_content_read'] is False,
            'Development cache contract differs')
        require(expanded_manifest['original_manifest_sha256']==digest(bound(original_cache/'manifest.json','manifest'))
            and expanded_manifest['source_sha256']==source['source_sha256'], 'Expanded natural source identity differs')
        require(len(original_manifest['recordings'])==len(expanded_manifest['records'])==320
            and {r['speaker'] for r in original_manifest['recordings']}=={'F01','F03','M01','M03'}, 'Development inventory differs')
        pieces = {arm:[] for arm in ARMS}
        for old_record,new_record in zip(original_manifest['recordings'],expanded_manifest['records']):
            require(old_record['recording_id']==new_record['recording_id']
                and new_record['original_feature_sha256']==old_record['feature_sha256'], 'Natural acquisition order/identity differs')
            old_path = original_cache/'features'/(old_record['recording_id']+'.npz')
            require(new_record['original_feature_path']==str(old_path.relative_to(REPO)), 'Original feature path differs')
            old_f = feature(old_path,old_record['feature_sha256'])
            new_f = feature(REPO/new_record['expanded_feature_path'],new_record['expanded_feature_sha256'])
            validate_arm_inputs(old_f,new_f)
            pieces['original6'].append(old_f); pieces['expanded7'].append(new_f)
        raw = {a:{k:np.concatenate([r[k] for r in pieces[a]]) for k in ('features','candidate_hz','candidate_mask')} for a in ARMS}
        require(len(raw['original6']['features'])==255532, 'Development frame count differs')
        with np.load(BytesIO(bound(run/'probabilities-development.npz','prediction')),allow_pickle=False) as z:
            require(set(z.files)=={'original_probability'}, 'Probability archive contains other fields')
            probability = z['original_probability'].copy()
        predictions = {}
        for name in METHODS:
            z = saved('predictions/'+name+'.npz'); require(set(z)=={'state','hz'}, 'Natural prediction inventory differs')
            predictions[name] = z['state'],z['hz']
        for seed in SEEDS:
            rel=f'predictions/original__seed{seed}.npz'; z=ledger.npz(LAST/rel,'prediction',last_frozen['prediction_sha256'][rel])
            result['prior_control_checks'][f'development/seed{seed}'] = preceding.exact_control(predictions[f'frozen-original6__seed{seed}'],(z['state'],z['hz']))
        verify('development',raw,probability,predictions)
        folders = {'regression':RUNS/'delayed-known-truth','consumed_harmonic':RUNS/'harmonic-context-v1/challenge',
            'consumed_subharmonic':RUNS/'subharmonic-objective-v1/challenge','consumed_augmentation':PRIOR/'challenge',
            'consumed_coverage':LAST/'challenge','challenge':run/'challenge'}
        for group,ids in GROUP_IDS.items():
            folder = folders[group]; records = None
            if group != 'regression':
                cache_bytes = (bound(folder/'manifest.json','manifest') if group=='challenge' else
                               historical_manifest_bytes(ledger,folder/'manifest.json',inputs,inherited))
                cache = json.loads(cache_bytes)
                require(cache['status']=='complete' and cache['reference_content_read'] is False
                    and len(cache['records'])==12 and {r['id'] for r in cache['records']}==ids, 'Analytic input inventory differs')
                records = {r['id']:r for r in cache['records']}
                if group=='challenge':
                    require(cache['schema']=='phenometrix-expanded-pool-challenge-cache-v1'
                        and cache['freeze_path']==str((run/'checkpoint-freeze.json').relative_to(REPO))
                        and cache['freeze_sha256']==result['identities']['checkpoint-freeze.json']
                        and cache['spec_sha256']==source['challenge_spec_sha256'], 'Fresh challenge freeze differs')
                    require(datetime.fromisoformat(checkpoint_freeze['frozen_at'])<=datetime.fromisoformat(cache['started_at'])
                        <=datetime.fromisoformat(cache['completed_at'])<=datetime.fromisoformat(frozen['predictions_frozen_at']), 'Fresh generation chronology differs')
            for identity in sorted(ids):
                old_f = feature(folder/'features'/(identity+'.npz'),records[identity]['feature_sha256'] if records else None)
                new_f = feature(run/'analytic'/(group+'-'+identity+'.npz'))
                validate_arm_inputs(old_f,new_f)
                require(len(old_f['features']) == (156 if group=='regression' else 196), 'Analytic acquisition extent differs')
                p,predictions = analytic_predictions(saved('predictions/'+group+'-'+identity+'.npz'),len(old_f['features']))
                if group != 'challenge':
                    old_group = 'challenge' if group=='consumed_coverage' else group
                    rel='predictions/'+old_group+'-'+identity+'.npz'
                    previous = ledger.npz(LAST/rel,'prediction',last_frozen['prediction_sha256'][rel])
                    require(np.array_equal(p,previous['original_probability']), 'Consumed original probability changed')
                    for seed in SEEDS:
                        result['prior_control_checks'][f'{group}/{identity}/seed{seed}'] = preceding.exact_control(
                            predictions[f'frozen-original6__seed{seed}'],(previous[f'original__seed{seed}__state'],previous[f'original__seed{seed}__hz']))
                verify(group+'/'+identity,{'original6':old_f,'expanded7':new_f},p,predictions)
        fields=('frames','state_mismatches','hz_mismatches','frozen_voice_gate_mismatches','nonmember_emissions')
        totals={f:sum(r[f] for domain in result['comparisons'].values() for r in domain.values()) for f in fields}
        control_frames=sum(v['frames'] for v in result['prior_control_checks'].values())
        require(len(result['comparisons'])==75 and totals['frames']==3233712, 'Incomplete pipeline/dataset/frame inventory')
        require(len(result['prior_control_checks'])==189 and control_frames==801372, 'Incomplete historical control inventory')
        require(frozen['control_reproduction']=={'state_hz_frame_pairs':control_frames,'state_mismatches':0,'hz_mismatches':0,'all_exact':True},
            'Runner historical control receipt differs')
        for name,binding in ledger.files.items():
            require(digest((REPO/name).read_bytes())==binding['sha256'], 'Input changed before receipt')
        result.update(status='complete',input_bytes_stable=True,totals=totals,distinct_checkpoint_count=9,new_checkpoint_count=6,
            historical_checkpoint_count=3,pipeline_count=12,analytic_case_count=74,historical_control_frame_pairs=control_frames,
            pass_result=all(totals[k]==0 for k in fields if k!='frames'))
    except BaseException as error:
        result.update(status='failed',pass_result=False,failure=repr(error))
    finally:
        output.parent.mkdir(parents=True,exist_ok=True)
        content=(json.dumps(ledger.files,sort_keys=True,indent=2)+'\n').encode(); ledger_path.write_bytes(content)
        result.update(completed_at=now(),elapsed_seconds=time.monotonic()-began,input_file_count=len(ledger.files),
            input_ledger_sha256=digest(content),audit_source_sha256=digest(Path(__file__).read_bytes()))
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'comparisons','prior_control_checks','original_voicing_checks','initialization_checks'}},indent=2))
    if not result['pass_result']: raise SystemExit(1)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); audit(args.run,args.output)
