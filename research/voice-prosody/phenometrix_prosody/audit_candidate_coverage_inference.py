"""Independent explicit-equation CPU replay of the no-fit candidate-policy study.

No fitting, waveform generation, acquisition replay, target/reference arrays,
metrics or analytic specifications. Shared grouping remains source-bound.
"""
from __future__ import annotations
import argparse
from datetime import datetime
from io import BytesIO
import json
from pathlib import Path
import time
import numpy as np
import torch
from . import audit_harmonic_inference as equations
from .audit_harmonic_inference import (Ledger,digest,require,now,original_probability,
    independent_logits,emit,prepare,compare_arrays)
from . import audit_augmentation_inference as prior_audit

REPO,ROOT,RUNS=equations.REPO,equations.ROOT,equations.RUNS
SEEDS=(20261022,20261023,20261024)
ARMS=('original','revised')
METHODS={f'{arm}__seed{seed}' for arm in ARMS for seed in SEEDS}
CHECKPOINTS={f'balance-aug__seed{seed}' for seed in SEEDS}
POLICY='shortest-admitted-full40-v1'
PRIOR=RUNS/'harmonic-augmentation-v1'
PRIOR_MANIFEST_SHA='d5513b83672f14a711f23a36e5258d348ac9b01a062c5945faaeaf5730d69426'
PRIOR_FREEZE_SHA='83097642ae265e9a5a21321bc0dc2cdaa977fe0862c30fe5db550d809314dc5e'
GROUP_IDS={'regression':equations.REGRESSION_IDS,
    'consumed_harmonic':{f'harmonic-balance-f{i}-p{j}' for i in range(6) for j in range(2)},
    'consumed_subharmonic':{f'subharmonic-balance-f{i}-p{j}' for i in range(6) for j in range(2)},
    'consumed_augmentation':{f'augmentation-balance-f{i}-p{j}' for i in range(6) for j in range(2)},
    'challenge':{f'coverage-balance-f{i}-p{j}' for i in range(6) for j in range(2)}}
PRIOR_GROUPS={'regression':'regression','consumed_harmonic':'consumed_challenge',
              'consumed_subharmonic':'consumed_subharmonic','consumed_augmentation':'challenge'}


def validate_freezes(source,frozen,manifest):
    require(source['schema']=='phenometrix-candidate-coverage-freeze.v1'
        and source['status']=='sources_checkpoints_frozen'
        and frozen['schema']==manifest['schema']=='phenometrix-candidate-coverage-experiment.v1'
        and frozen['status']=='predictions_frozen' and manifest['status']=='complete','Wait for completed required freezes')
    for name in ('source_sha256','checkpoints','challenge_spec_sha256','candidate_policy','arms','seeds','new_fits','frozen_at'):
        require(source[name]==frozen[name]==manifest[name],'Phase identity changed: '+name)
    require(set(source['checkpoints'])==CHECKPOINTS and source['candidate_policy']==POLICY
        and source['arms']==list(ARMS) and source['seeds']==list(SEEDS),'Declared policy/checkpoints differ')
    require(source['new_fits']==frozen['optimizer_updates']==manifest['optimizer_updates']==0,'Unexpected fitting')
    require(frozen['prediction_sha256']==manifest['prediction_sha256'],'Prediction bytes changed')
    require(all(frozen[k] is False for k in ('development_references_read','train_references_read','reserved_content_read')),
        'Prediction label boundary differs')
    require(manifest['reserved_content_read'] is False,'Reserved content excluded')
    require(datetime.fromisoformat(source['frozen_at'])<=datetime.fromisoformat(frozen['predictions_frozen_at'])
        <=datetime.fromisoformat(manifest['completed_at']),'Freeze chronology differs')


def validate_arm_inputs(original,revised):
    for field in ('reference_sample','feature_names','sample_rate_hz','pcm_sample_count'):
        require(np.array_equal(original[field],revised[field]),'Policy changed acquisition metadata: '+field)
    columns=[48,49,50,51,52,53,54,56]
    require(np.array_equal(original['features'][:,columns],revised['features'][:,columns]),
        'Policy changed noncandidate global inputs')


def read_analytic_predictions(saved,frames):
    expected={'original_probability'}|{n+'__'+field for n in METHODS for field in ('state','hz')}
    require(set(saved)==expected,'Analytic prediction inventory differs')
    p=saved['original_probability']
    require(p.shape==(frames,) and np.isfinite(p).all() and np.all((p>=0)&(p<=1)),
        'Original probability extent/range differs')
    result={}
    for name in METHODS:
        state,hz=saved[name+'__state'],saved[name+'__hz']
        require(state.shape==hz.shape==p.shape and np.isin(state,[0,1,2]).all(),'Invalid state extent')
        require(np.array_equal(np.isfinite(hz),state==1) and np.isnan(hz[state!=1]).all(),'Invalid emitted frequency availability')
        require(np.array_equal(state==0,p<.5),'Original frozen voice gate differs')
        result[name]=(state,hz)
    return p,result


def exact_control(saved,prior):
    a,b=saved;c,d=prior
    require(a.shape==c.shape and b.shape==d.shape and a.dtype==c.dtype and b.dtype==d.dtype,'Control extent/dtype differs')
    states=int(np.count_nonzero(a!=c));frequencies=int(np.count_nonzero(~((b==d)|(np.isnan(b)&np.isnan(d)))))
    require(states==frequencies==0,'Prior frozen original control changed')
    return {'frames':len(a),'state_mismatches':states,'hz_mismatches':frequencies}


def audit(run,output):
    run,output=Path(run).resolve(),Path(output).resolve()
    require(run.is_relative_to(RUNS) and output.is_relative_to(RUNS),'Ignored local artifacts required')
    ledger_path=output.with_name(output.stem+'-inputs.json')
    require(not output.exists() and not ledger_path.exists(),'Refuses overwrite')
    ledger=Ledger();began=time.monotonic();torch.set_num_threads(1)
    result={'schema':'phenometrix-candidate-coverage-inference-audit.v1','status':'running','started_at':now(),
        'device':'cpu','torch':torch.__version__,'numpy':np.__version__,'fitting_performed':False,
        'reference_files_or_analytic_target_arrays_read':False,'truth_used_for_comparison':False,
        'challenge_spec_or_metrics_content_opened':False,'reserved_content_read':False,
        'new_gpu_work':False,'waveform_generation_performed':False,'acquisition_replayed':False,
        'inference_implementation':'explicit independent Torch equations; no runner infer or HarmonicSelector.forward',
        'preprocessing_scope':'shared grouping source-bound, independent emitted-member validation; arm-specific cached candidate pools',
        'voicing_contract':'frozen original probability from ORIGINAL raw feature pool for both arms',
        'comparisons':{},'original_voicing_checks':{},'initialization_checks':{},'prior_control_checks':{}}
    try:
        manifest=ledger.json(run/'manifest.json','manifest')
        source=ledger.json(run/'source-checkpoint-freeze.json','freeze')
        frozen=ledger.json(run/'prediction-freeze.json','freeze')
        validate_freezes(source,frozen,manifest)
        result['identities']={p:digest(ledger.read(run/p,'freeze' if 'freeze' in p else 'manifest'))
            for p in ('manifest.json','source-checkpoint-freeze.json','prediction-freeze.json')}
        require(frozen['source_checkpoint_freeze_sha256']==result['identities']['source-checkpoint-freeze.json'],'Outer freeze SHA differs')
        for name,h in frozen['source_sha256'].items():
            ledger.read(REPO/name,'source',h);ledger.read(run/'source-snapshot'/name,'source_snapshot',h)
        for p in (Path(__file__),Path(equations.__file__),Path(prior_audit.__file__),
                  ROOT/'tests/test_audit_candidate_coverage_inference.py'):ledger.read(p,'source')
        inputs=frozen['input_sha256']
        def bound(path,role,expected=None):
            path=Path(path).resolve();name=str(path.relative_to(REPO))
            require(name in inputs,'Input absent from prediction freeze: '+name)
            require(expected is None or expected==inputs[name],'Nested identity differs')
            return ledger.read(path,role,inputs[name])
        def feature(path,expected=None):return prior_audit.load_feature_bytes(bound(path,'feature',expected))
        def saved(relative):return ledger.npz(run/relative,'prediction',frozen['prediction_sha256'][relative])
        previous= json.loads(bound(PRIOR/'manifest.json','manifest',PRIOR_MANIFEST_SHA))
        pf=json.loads(bound(PRIOR/'prediction-freeze.json','freeze',PRIOR_FREEZE_SHA))
        require(previous['status']=='complete' and previous['prediction_sha256']==pf['prediction_sha256'],'Prior completed run differs')
        models={}
        for seed in SEEDS:
            name=f'balance-aug__seed{seed}';row=source['checkpoints'][name]
            require(row['path']==str((PRIOR/'checkpoints'/(name+'.pt')).relative_to(REPO))
                and row['sha256']==previous['fits'][name]['weights_sha256'],'Frozen checkpoint identity differs')
            ck=torch.load(BytesIO(bound(REPO/row['path'],'weights',row['sha256'])),map_location='cpu',weights_only=True)
            weights,initial=prior_audit.validate_checkpoint(name,ck,previous['fits'][name]);models[seed]=weights
            result['initialization_checks'][name]=initial
        original=torch.load(BytesIO(bound(RUNS/'delayed-mlp-seed20261022/weights.pt','weights',equations.ORIGINAL_SHA)),map_location='cpu',weights_only=True)
        require(original['config']['kind']=='framewise-mlp','Original voice architecture differs')
        def verify_probability(identity,raw,p):
            with torch.inference_mode():actual=np.concatenate([original_probability(original['state_dict'],raw['features'][i:i+4096]) for i in range(0,len(raw['features']),4096)])
            require(p.shape==actual.shape and np.isfinite(p).all() and np.isfinite(actual).all() and np.all((p>=0)&(p<=1)),'Original probability extent/range differs')
            delta=float(np.abs(p-actual).max());mismatches=int(np.count_nonzero((p>=.5)!=(actual>=.5)))
            require(delta<=1e-6 and mismatches==0,'Original voice parity failed')
            result['original_voicing_checks'][identity]={'frames':len(p),'max_absolute_difference':delta,'gate_mismatches':mismatches,
                'saved_sha256_float64':digest(p.astype('<f8').tobytes()),'recomputed_sha256_float64':digest(actual.astype('<f8').tobytes())}
        def verify(identity,raws,p,predictions):
            reports={}
            for arm in ARMS:
                batch=prepare(raws[arm])
                for seed,weights in models.items():
                    state_rows,hz_rows=[],[]
                    with torch.inference_mode():
                        for start in range(0,len(batch.features),4096):
                            stop=min(start+4096,len(batch.features));logits=independent_logits(weights,'relative',batch.features[start:stop],batch.candidate_mask[start:stop])
                            state,hz=emit(logits,batch.candidate_hz[start:stop].numpy(),batch.candidate_mask[start:stop].numpy(),p[start:stop]);state_rows.append(state);hz_rows.append(hz)
                    name=f'{arm}__seed{seed}'
                    reports[name]=compare_arrays(predictions[name],np.concatenate(state_rows),np.concatenate(hz_rows),batch,p)
            result['comparisons'][identity]=reports
            print(json.dumps({'dataset':identity,'methods':len(reports),'frames_per_method':len(p)}),flush=True)
        original_cache=RUNS/'delayed-development-features'
        old_manifest=json.loads(bound(original_cache/'manifest.json','manifest'))
        revised_manifest=json.loads(bound(run/'development/manifest.json','manifest'))
        require(old_manifest['status']==revised_manifest['status']=='complete' and old_manifest['split']==revised_manifest['split']=='development'
            and revised_manifest['candidate_policy']==POLICY and revised_manifest['reference_content_read'] is False,'Development cache contract differs')
        require(revised_manifest['original_manifest_sha256']==digest(bound(original_cache/'manifest.json','manifest'))
            and revised_manifest['source_sha256']==source['source_sha256'],'Development cache source identity differs')
        require(len(old_manifest['recordings'])==len(revised_manifest['records'])==320
            and {r['speaker'] for r in old_manifest['recordings']}=={'F01','F03','M01','M03'},'Development inventory differs')
        pieces={arm:[] for arm in ARMS}
        for a,b in zip(old_manifest['recordings'],revised_manifest['records']):
            require(a['recording_id']==b['recording_id'] and b['original_feature_sha256']==a['feature_sha256'],'Development order/identity differs')
            orig_path=original_cache/'features'/(a['recording_id']+'.npz')
            require(b['original_feature_path']==str(orig_path.relative_to(REPO)),'Original feature path differs')
            raw={'original':feature(orig_path,a['feature_sha256']),'revised':feature(REPO/b['feature_path'],b['feature_sha256'])}
            validate_arm_inputs(raw['original'],raw['revised'])
            for arm in ARMS:pieces[arm].append(raw[arm])
        raw={arm:{k:np.concatenate([r[k] for r in pieces[arm]]) for k in ('features','candidate_hz','candidate_mask')} for arm in ARMS}
        require(len(raw['original']['features'])==255532,'Development frame count differs')
        with np.load(BytesIO(bound(run/'probabilities-development.npz','prediction')),allow_pickle=False) as z:
            require(set(z.files)=={'original_probability'},'Development probability inventory differs');probability=z['original_probability'].copy()
        verify_probability('development',raw['original'],probability)
        predictions={}
        for name in METHODS:
            z=saved('predictions/'+name+'.npz');require(set(z)=={'state','hz'},'Natural output inventory differs');predictions[name]=(z['state'],z['hz'])
        for seed in SEEDS:
            rel=f'predictions/balance-aug__seed{seed}.npz'
            z=ledger.npz(PRIOR/rel,'prediction',pf['prediction_sha256'][rel])
            result['prior_control_checks'][f'development/seed{seed}']=exact_control(predictions[f'original__seed{seed}'],(z['state'],z['hz']))
        verify('development',raw,probability,predictions)
        folders={'regression':RUNS/'delayed-known-truth','consumed_harmonic':RUNS/'harmonic-context-v1/challenge',
            'consumed_subharmonic':RUNS/'subharmonic-objective-v1/challenge','consumed_augmentation':PRIOR/'challenge','challenge':run/'challenge'}
        for group,ids in GROUP_IDS.items():
            folder=folders[group];cache=None;records=None
            if group!='regression':
                cache=json.loads(bound(folder/'manifest.json','manifest'))
                require(cache['status']=='complete' and cache['reference_content_read'] is False
                    and len(cache['records'])==12 and {r['id'] for r in cache['records']}==ids,'Analytic input inventory differs')
                records={r['id']:r for r in cache['records']}
                if group=='challenge':
                    require(cache['schema']=='phenometrix-candidate-coverage-challenge-cache-v1'
                        and cache['freeze_path']==str((run/'source-checkpoint-freeze.json').relative_to(REPO))
                        and cache['freeze_sha256']==result['identities']['source-checkpoint-freeze.json']
                        and cache['spec_sha256']==source['challenge_spec_sha256'],'Fresh generation freeze differs')
                    require(datetime.fromisoformat(source['frozen_at'])<=datetime.fromisoformat(cache['started_at'])
                        <=datetime.fromisoformat(cache['completed_at'])<=datetime.fromisoformat(frozen['predictions_frozen_at']),'Fresh generation chronology differs')
            for identity in sorted(ids):
                orig=folder/'features'/(identity+'.npz')
                old_feature=feature(orig,records[identity]['feature_sha256'] if records else None)
                new_feature=feature(run/'analytic'/(group+'-'+identity+'.npz'))
                validate_arm_inputs(old_feature,new_feature)
                p,predictions=read_analytic_predictions(saved('predictions/'+group+'-'+identity+'.npz'),len(old_feature['features']))
                verify_probability(group+'/'+identity,old_feature,p)
                if group in PRIOR_GROUPS:
                    rel='predictions/'+PRIOR_GROUPS[group]+'-'+identity+'.npz'
                    old_saved=ledger.npz(PRIOR/rel,'prediction',pf['prediction_sha256'][rel])
                    require(np.array_equal(p,old_saved['original_probability']),'Consumed probability changed')
                    for seed in SEEDS:
                        name=f'balance-aug__seed{seed}'
                        result['prior_control_checks'][f'{group}/{identity}/seed{seed}']=exact_control(predictions[f'original__seed{seed}'],(old_saved[name+'__state'],old_saved[name+'__hz']))
                verify(group+'/'+identity,{'original':old_feature,'revised':new_feature},p,predictions)
        fields=('frames','state_mismatches','hz_mismatches','frozen_voice_gate_mismatches','nonmember_emissions')
        totals={k:sum(row[k] for domain in result['comparisons'].values() for row in domain.values()) for k in fields}
        require(len(result['comparisons'])==63 and totals['frames']==1602744,'Incomplete arm/dataset/frame inventory')
        require(len(result['prior_control_checks'])==153 and sum(v['frames'] for v in result['prior_control_checks'].values())==794316,'Incomplete historical control inventory')
        for name,binding in ledger.files.items():require(digest((REPO/name).read_bytes())==binding['sha256'],'Input changed before receipt')
        result.update(status='complete',input_bytes_stable=True,totals=totals,checkpoint_count=3,policy_count=2,analytic_case_count=62,
            historical_control_frame_pairs=794316,pass_result=all(totals[k]==0 for k in fields if k!='frames'))
    except BaseException as error:result.update(status='failed',pass_result=False,failure=repr(error))
    finally:
        output.parent.mkdir(parents=True,exist_ok=True);raw=(json.dumps(ledger.files,sort_keys=True,indent=2)+'\n').encode();ledger_path.write_bytes(raw)
        result.update(completed_at=now(),elapsed_seconds=time.monotonic()-began,input_file_count=len(ledger.files),
            input_ledger_sha256=digest(raw),audit_source_sha256=digest(Path(__file__).read_bytes()))
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'comparisons','prior_control_checks','original_voicing_checks','initialization_checks'}},indent=2))
    if not result['pass_result']:raise SystemExit(1)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();audit(args.run,args.output)
