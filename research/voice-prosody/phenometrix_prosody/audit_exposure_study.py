"""Independent frozen exposure-study audit, using saved inputs and CPU equations.

Never fits, synthesizes, acquires data or changes a scientific result. Reuses
historical independent score/group equations; never calls the new producer's
loss, schedule, forward or decision functions. No global monkeypatching.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import signal
import resource
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from . import audit_evidence_scores as old
from . import audit_frequency_diversity as bank_audit
from . import audit_harmonic_inference as voice_equations

base, historical, pool, retention = old.base, old.historical, old.pool, old.retention
REPO, ROOT, RUNS = old.REPO, old.ROOT, old.RUNS
SEEDS, ARMS, FIT_NAMES, METHODS = old.SEEDS, old.ARMS, old.FIT_NAMES, old.METHODS
PRIOR, PRIOR_SHA = old.PRIOR, old.PRIOR_SHA
require, bound, load_npz, compare_nested = old.require, old.bound, old.load_npz, old.compare_nested
BANK = RUNS/'frequency-diversity-v1'
BANK_IDENTITIES = {
    'manifest.json':'8349835c4bac48332f629219063470ece8d84958e520790b300ab083cc9eb8a7',
    'source-freeze.json':'f1183405fe8a556d9293688037107e6099db480fd9e6ce496cae133912bcc918',
    'report.json':'3338e407866bc87dac2bb4f6dcdbf244a5852ab7c890aacef37c1cee36d4522c',
    'frame-table.npz':'201a2b648dcd0714ea0c73c50648ab881be387ff000f50492fe309293f26d04c',
    'spec.json':'f25019b3b9deb6d699f440fd672cac5aabe6a6c70ccf6842c91fdb537a72fba4',
    'independent-audit.json':'3587af269f38689d2b04d1db04e5cb524efc8cd3533d91b88e3edb1b921f570e',
}
SAMPLE_ROWS = (4,80,194)
CORRELATION_ATOL = 2e-13
CHANNEL_ATOL = 1e-7
PROBABILITY_ATOL = 1e-6
MAX_AUDIT_SECONDS = 1800
MAX_PYTHON_RSS = 16*1024**3


def source_hashes():
    """Explicit transitive independent interpretation closure, plus fixture bytes."""
    names=set(bank_audit.AUDIT_DEPENDENCIES)|{'audit_evidence_scores','audit_frequency_diversity',
        'audit_exposure_study','metrics','harmonic_selector_model'}
    paths=[Path(importlib.import_module('.'+n,__package__).__file__).resolve() for n in sorted(names)]
    paths.extend(ROOT/'tests'/n for n in ('test_audit_exposure_study.py','test_audit_evidence_scores.py','test_evidence_selector_model.py'))
    return {str(p.relative_to(REPO)):base.sha(p) for p in paths}


def scalar_periodicity(window,hz,mask):
    """Independent scalar interpolation/dot replay; no extraction kernel call."""
    x=np.asarray(window);hz=np.asarray(hz);mask=np.asarray(mask)
    require(x.shape==(1920,) and x.dtype==np.float32 and np.isfinite(x).all(),'Scalar PCM contract differs')
    require(hz.ndim==1 and mask.dtype==np.bool_ and mask.shape==hz.shape and len(hz) in (6,7),'Scalar candidates differ')
    require(np.isfinite(hz[mask]).all() and ((hz[mask]>=50)&(hz[mask]<=700)).all(),'Scalar active Hz invalid')
    x=x.astype(np.float64);x=x-x.mean()
    corr=np.zeros((len(hz),4),np.float64);valid=np.zeros(len(hz),bool)
    period=np.zeros(len(hz),np.float64);length=np.zeros(len(hz),np.int32)
    for k in np.flatnonzero(mask):
        period[k]=48000/hz[k];length[k]=1920-math.floor(period[k])-1
        a=x[:length[k]];ea=float(np.dot(a,a));energies=[ea];values=[]
        for factor in (1.,.5,1./3.,.25):
            lag=period[k]*factor;lo=math.floor(lag);fraction=lag-lo
            # An independently sliced scalar interpolation avoids the producer's
            # padded/vectorized 1920-position reduction and indexing machinery.
            b=(1-fraction)*x[lo:lo+length[k]]+fraction*x[lo+1:lo+length[k]+1]
            eb=float(np.dot(b,b));energies.append(eb)
            values.append(float(np.dot(a,b))/math.sqrt(ea*eb) if ea*eb>0 else 0.)
        valid[k]=all(e>=1e-12 for e in energies)
        if valid[k]:corr[k]=np.clip(values,-1.,1.)
    channels=np.stack((corr[:,0],(corr[:,0]-corr[:,1:].max(-1))/2,valid),axis=-1).astype(np.float32)
    return {'correlations':corr,'valid':valid,'period_samples':period,'common_length':length,'evidence':channels}


def compare_scalar(saved,got):
    for key in ('valid','period_samples','common_length'):
        require(np.array_equal(saved[key],got[key]),'Raw periodicity '+key+' differs')
    dc=float(np.max(np.abs(saved['correlations']-got['correlations']),initial=0))
    de=float(np.max(np.abs(saved['evidence'].astype(float)-got['evidence'].astype(float)),initial=0))
    require(dc<=CORRELATION_ATOL and de<=CHANNEL_ATOL,'Raw periodicity qualification tolerance exceeded')
    return dc,de


def targeted_schedule(strata,case_index,records,seed,epoch):
    require(seed in SEEDS and type(epoch) is int and 0<=epoch<12,'Undeclared schedule seed/epoch')
    require(bank_audit.support_report(strata,case_index,records)['pass_result'],'Insufficient numeric support for sampling')
    rng=np.random.default_rng(seed+2_000_000+epoch);pools=[]
    for name in ('added_only','counterexample','broad'):
        selected=np.flatnonzero(strata[name]);cases=np.unique(case_index[selected])
        pools.append((cases,{int(c):selected[case_index[selected]==c] for c in cases}))
    indices=np.empty((33,512),np.int64);origin=np.empty((33,512),np.uint8)
    for batch in range(33):
        offset=0
        for label,(count,(cases,rows)) in enumerate(zip((128,128,256),pools)):
            for case in rng.choice(cases,size=count,replace=True):
                candidates=rows[int(case)];indices[batch,offset]=candidates[rng.integers(len(candidates))]
                origin[batch,offset]=label;offset+=1
        permutation=rng.permutation(512)
        indices[batch]=indices[batch,permutation];origin[batch]=origin[batch,permutation]
    return {'indices':indices,'origin':origin}


def sampling_exposure(schedule,strata,table,case_count):
    selected=schedule['indices'].ravel();origin=schedule['origin']
    require(origin.shape==schedule['indices'].shape==(33,512),'Schedule extent differs')
    for label,count in enumerate((128,128,256)):
        require(np.all((origin==label).sum(-1)==count),'Requested per-batch stratum count differs')
    require(table['common_eligible'][selected].all(),'Sampling included noncommon rows')
    unique=len(np.unique(selected));covered=table['expanded_group_covered'][selected]
    names=('added_only','counterexample','broad')
    return {'selected_rows':len(selected),'unique_rows':unique,'duplicate_draws':len(selected)-unique,
        'requested_strata':{name:int((origin==label).sum()) for label,name in enumerate(names)},
        'actual_strata':{name:int(strata[name][selected].sum()) for name in names},
        'per_case_selected_rows':np.bincount(table['case_index'][selected],minlength=case_count).tolist(),
        'expanded_group_covered_rows':int(covered.sum()),'expanded_group_uncovered_zero_rows':int((~covered).sum()),
        'per_region':{str(r):{'selected_rows':int((table['region'][selected]==r).sum()),
            'covered_rows':int(((table['region'][selected]==r)&covered).sum())} for r in sorted(set(table['region']))}}


class ExposureAudit(base.Audit):
    def __init__(self):
        super().__init__();self.started=time.monotonic();self.peak_rss_bytes=0
    def bind(self,path,expected=None):
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        self.peak_rss_bytes=max(self.peak_rss_bytes,rss)
        require(rss<=MAX_PYTHON_RSS and time.monotonic()-self.started<=MAX_AUDIT_SECONDS,'Independent audit resource ceiling exceeded')
        return super().bind(path,expected)


class AuditLedger:
    """Adapt the older independent bank validator without changing global state."""
    def __init__(self,audit):self.audit=audit
    def bind(self,path,expected=None):
        p=Path(path);require(not p.is_absolute(),'Expected repository-relative bank identity')
        return self.audit.bind(REPO/p,expected)


def audit_bank(audit,manifest):
    docs={name:(load_npz(audit,BANK/name,digest) if name.endswith('.npz') else audit.json(BANK/name,digest))
        for name,digest in BANK_IDENTITIES.items()}
    cache=docs['manifest.json'];freeze=docs['source-freeze.json'];table=docs['frame-table.npz'];spec=docs['spec.json']
    records=bank_audit.expected_records();bank_audit.validate_spec(spec)
    require(spec['records']==records,'Targeted metadata differs from independent enumeration')
    require(manifest['training_spec_sha256']==freeze['training_spec_sha256'],'Current TRAIN spec changed')
    require(len(freeze['source_sha256'])==71 and cache['source_sha256']==freeze['source_sha256'],'Saved historical71 source closure differs')
    require(cache['status']=='complete' and docs['independent-audit.json']['pass'] is True,'Saved bank lacks completed independent qualification')
    for path,digest in freeze['source_sha256'].items():
        require(manifest['source_sha256'].get(path)==digest,'A prior71 frozen scientific source changed')
        audit.bind(REPO/path,digest)
    for name,digest in BANK_IDENTITIES.items():
        require(manifest['input_sha256'].get(str((BANK/name).relative_to(REPO)))==digest,'Bank identity missing from current input freeze')
    coverage,strata,distinct=bank_audit.validate_table(table,records)
    frames=bank_audit.verify_record_files(AuditLedger(audit),BANK,cache,table,records)
    expected=bank_audit.expected_report(table,records)
    bank_audit.compare(docs['report.json'],expected,'Saved TRAIN report')
    support=bank_audit.support_report(strata,table['case_index'],records)
    return cache,table,records,strata,support,{'case_count':len(records),'acquisition_rows':frames,
        'common_eligible_rows':int(table['common_eligible'].sum()),'support':support,
        'bound_saved_bank_sha256':BANK_IDENTITIES,'regenerated_cases':0}


def independent_logits(weights,grouped,evidence,mode):
    require(mode in ('off','periodicity'),'Undeclared evidence mode')
    x=np.asarray(evidence);mask=grouped['candidate_mask']
    require(x.dtype==np.float32 and x.shape==mask.shape+(3,) and np.isfinite(x).all(),'Inference evidence shape/dtype differs')
    require(not np.any(x[~mask]),'Evidence on a padded candidate')
    logits=pool.independent_logits(weights,grouped)
    values=torch.from_numpy(x if mode=='periodicity' else np.zeros_like(x))
    logits=logits+F.linear(values,weights['evidence_head.weight']).squeeze(-1)
    logits=torch.where(torch.from_numpy(mask),logits,torch.full_like(logits,-1e9))
    require(torch.isfinite(logits).all(),'Independent selector equation produced nonfinite output')
    return logits


def check_saved_output(state,hz,want_state,want_hz):
    require(state.shape==hz.shape==want_state.shape==want_hz.shape,'Prediction array extent differs')
    require(np.array_equal(state,want_state) and np.array_equal(hz,want_hz,equal_nan=True),
        'Independent CPU equation state/Hz differs from saved output')
    return len(state)


def verify_inference(audit,run,manifest,checkpoint_weights):
    voice=torch.load(audit.bind(RUNS/'delayed-mlp-seed20261022/weights.pt',voice_equations.ORIGINAL_SHA),map_location='cpu',weights_only=True)
    require(voice['config']['kind']=='framewise-mlp','Original voicing model contract differs')
    pairs=voice_frames=0;max_probability_difference=0.;datasets=[]
    def verify(identity,original,expanded,evidence,probability,predictions):
        nonlocal pairs,voice_frames,max_probability_difference
        groups=pool.independent_groups(expanded);n=len(probability)
        require(probability.shape==(len(original['features']),) and np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all(),'Original probability extent/values differ')
        with torch.inference_mode():
            for start in range(0,n,2048):
                sl=slice(start,min(start+2048,n));p=voice_equations.original_probability(voice['state_dict'],original['features'][sl])
                delta=float(np.max(np.abs(p-probability[sl]),initial=0));max_probability_difference=max(max_probability_difference,delta)
                require(delta<=PROBABILITY_ATOL and np.array_equal(p<.5,probability[sl]<.5),'Independent original voicing differs')
                g={k:v[sl] for k,v in groups.items()};voice_frames+=len(p)
                for name,weights in checkpoint_weights.items():
                    mode='periodicity' if name.split('__seed')[0].endswith('__on') else 'off'
                    logits=independent_logits(weights,g,evidence[sl],mode)
                    state,hz=voice_equations.emit(logits,g['candidate_hz'],g['candidate_mask'],probability[sl])
                    saved=predictions[name];pairs+=check_saved_output(state,hz,saved[0][sl],saved[1][sl])
        datasets.append({'id':identity,'frames':n,'new_models':len(checkpoint_weights)})
    # Predictions are bound before reconstruction. Read original57 only for the
    # unchanged voice equation and expanded65 only for candidate selection.
    p=load_npz(audit,bound(audit,manifest,run/'probabilities-development.npz'))['original_probability']
    saved={n:load_npz(audit,run/'predictions'/(n+'.npz'),manifest['prediction_sha256']['predictions/'+n+'.npz']) for n in FIT_NAMES}
    cache=audit.json(bound(audit,manifest,run/'evidence/development/manifest.json'))
    for row in cache['records']:
        original=load_npz(audit,REPO/row['original_feature_path'],row['original_feature_sha256'])
        expanded=load_npz(audit,REPO/row['expanded_feature_path'],row['expanded_feature_sha256'])
        e=load_npz(audit,REPO/row['evidence_path'],row['evidence_sha256']);sl=slice(row['start'],row['stop'])
        old.verify_evidence(e,expanded,pool.independent_groups(expanded),row['pcm_sha256'])
        verify('development/'+row['recording_id'],original,expanded,e['evidence'],p[sl],{n:(s['state'][sl],s['hz'][sl]) for n,s in saved.items()})
    inventory=audit.json(bound(audit,manifest,run/'analytic-inventory.json'))
    for group,rows in inventory.items():
        for row in rows:
            original=load_npz(audit,REPO/row['original_feature_path'],row['original_feature_sha256'])
            expanded=load_npz(audit,REPO/row['expanded_feature_path'],row['expanded_feature_sha256'])
            e=load_npz(audit,REPO/row['evidence_path'],row['evidence_sha256'])
            old.verify_evidence(e,expanded,pool.independent_groups(expanded),row['pcm_sha256'])
            s=load_npz(audit,run/row['prediction_path'],row['prediction_sha256'])
            verify(group+'/'+row['id'],original,expanded,e['evidence'],s['original_probability'],{n:(s[n+'__state'],s[n+'__hz']) for n in FIT_NAMES})
    require(len(datasets)==406 and pairs==12*voice_frames,'Inference dataset/model denominator differs')
    return {'datasets':len(datasets),'new_models':12,'frame_model_comparisons':pairs,'original_voicing_frames':voice_frames,
        'state_mismatches':0,'hz_mismatches':0,'voicing_gate_mismatches':0,'maximum_original_probability_difference':max_probability_difference,
        'probability_absolute_tolerance':PROBABILITY_ATOL,'explicit_cpu_equations':True,'dataset_counts':datasets}


def verify_checkpoints(audit,run,manifest,prior,prior_dir=PRIOR):
    from .harmonic_selector_model import HarmonicSelector
    require(set(manifest['fits'])==set(FIT_NAMES),'Exact12 final fits required')
    require(set(manifest['fit_attempts'])==set(FIT_NAMES),'Fit attempts exceed or omit declared inventory')
    shapes={'scaling_fitted':(),'candidate_mean':(8,),'candidate_scale':(8,),'global_mean':(9,),'global_scale':(9,),
        'evidence_head.weight':(1,3)}
    for layer,shape in {'candidate_encoder':(32,8),'message_encoder':(32,9),'shared_hidden':(32,73),'shared_head':(1,32)}.items():
        shapes[layer+'.weight']=shape;shapes[layer+'.bias']=(shape[0],)
    buffers=('scaling_fitted','candidate_mean','candidate_scale','global_mean','global_scale')
    scaling=None;initial={};elapsed=0.;weights={};previous_completion=historical.dt(manifest['training_frozen_at'])
    for seed in SEEDS:
        name=f'expanded7__seed{seed}';old=torch.load(audit.bind(prior_dir/'checkpoints'/(name+'.pt'),prior['fits'][name]['weights_sha256']),map_location='cpu',weights_only=True)['state_dict']
        if scaling is None:scaling={k:old[k] for k in buffers}
        require(all(torch.equal(scaling[k],old[k]) for k in buffers),'Prior scaling differs across seeds')
        torch.manual_seed(seed);model=HarmonicSelector('relative','acceptable-set')
        state=model.state_dict()
        for k in buffers:state[k]=scaling[k].clone()
        state['evidence_head.weight']=torch.zeros((1,3),dtype=torch.float32)
        digest=hashlib.sha256()
        for k,t in sorted(state.items()):digest.update(k.encode());digest.update(t.numpy().tobytes())
        initial[seed]=digest.hexdigest()
        for arm in ARMS:
            name=f'{arm}__seed{seed}';fit=manifest['fits'][name];attempt=manifest['fit_attempts'][name]
            require((fit['arm'],fit['seed'],fit['optimizer_updates'],fit['parameter_count'])==(arm,seed,396,3012),'Fit identity/budget differs')
            require(attempt['status']=='complete' and attempt['initial_tensors_sha256']==fit['initial_tensors_sha256']==initial[seed],
                'Final fit attempt or reconstructed initialization differs')
            require(historical.dt(attempt['started_at'])<historical.dt(manifest['deadline_utc']) and
                historical.dt(manifest['training_frozen_at'])<=historical.dt(attempt['started_at'])<=historical.dt(attempt['completed_at'])<=historical.dt(manifest['checkpoints_frozen_at']),
                'Fit chronology violates fixed phase/deadline')
            require(previous_completion<=historical.dt(attempt['started_at']),'Fixed seed/arm fit sequence overlaps or changes order')
            previous_completion=historical.dt(attempt['completed_at'])
            for key in ('epoch_losses','epoch_base_losses','epoch_weighted_auxiliary_losses'):
                require(len(fit[key])==12 and all(math.isfinite(x) and x>=-1e-7 for x in fit[key]),'Fit loss summary invalid')
            require(np.allclose(fit['epoch_losses'],np.array(fit['epoch_base_losses'])+np.array(fit['epoch_weighted_auxiliary_losses']),atol=1e-7,rtol=0),'Loss decomposition differs')
            saved=torch.load(audit.bind(run/'checkpoints'/(name+'.pt'),fit['weights_sha256']),map_location='cpu',weights_only=True)
            require(saved['arm']==arm and saved['evidence_mode']==('periodicity' if arm.endswith('__on') else 'off') and
                saved['config']=={'representation':'shared-candidates','loss':'acceptable-set','hidden_size':32},'Saved checkpoint contract differs')
            state=saved['state_dict'];weights[name]=state;require(set(state)==set(shapes),'Checkpoint tensor inventory differs')
            require(sum(t.numel() for k,t in state.items() if k not in buffers)==3012,'Actual learned tensor capacity differs')
            for k,shape in shapes.items():
                require(tuple(state[k].shape)==shape and state[k].dtype==(torch.bool if k=='scaling_fitted' else torch.float32) and
                    torch.isfinite(state[k]).all(),'Checkpoint tensor shape/dtype/finite value differs')
            require(all(torch.equal(scaling[k],state[k]) for k in buffers),'Original-six scaling changed')
            if arm.endswith('__off'):require(torch.equal(state['evidence_head.weight'],torch.zeros((1,3))),'Off residual changed from zero')
            compare_nested(audit,state['evidence_head.weight'].tolist(),fit['evidence_residual_weights'],name+'/residual')
            elapsed+=fit['elapsed_seconds']
    audit.equal(elapsed,manifest['fit_elapsed_seconds'],'Cumulative fit seconds')
    return weights,{'checkpoints':12,'parameters_each':3012,'initial_tensor_sha256':initial,'original_six_scaling_exact':True,
        'optimizer_updates_each':396,'fit_elapsed_seconds':elapsed,'optimizer_replay_performed':False}


def reconstruct_banks(audit,run,manifest,cache,table,records):
    """Every training target derives from emitted grouped Hz, never fitted logits."""
    acceptable,covered,nearest=pool.nearest_labels(table['expanded_group_hz'],table['expanded_group_mask'],table['truth_hz'],np.ones(len(table['truth_hz']),bool))
    target={'acceptable':acceptable,'covered':covered,'nearest':nearest,'scenario_index':table['case_index'],
        'reference_hz':table['truth_hz'],'reference_sample':table['reference_sample'],
        'availability_sample':table['availability_sample'],'region':table['region'],'common':table['common_eligible']}
    existing=old.reconstruct_bank(audit,run,manifest,'existing')
    samples=active=0;max_corr=max_channel=0.;evidence_frames=0
    for exposure,source_rows in (('targeted',cache['records']),('existing',audit.json(bound(audit,manifest,PRIOR/'training/manifest.json'))['records'])):
        current=audit.json(bound(audit,manifest,run/'evidence'/(exposure+'-training')/'manifest.json'))
        require(current['source_sha256']==manifest['source_sha256'] and current['reference_content_read'] is False and
            [r['id'] for r in current['records']]==[r['id'] for r in source_rows],'TRAIN evidence inventory differs')
        for row,er in zip(source_rows,current['records']):
            feature=load_npz(audit,REPO/row['expanded_feature_path'],row['expanded_feature_sha256'])
            groups=pool.independent_groups(feature)
            e=load_npz(audit,REPO/er['evidence_path'],er['evidence_sha256'])
            require(er['pcm_sha256']==row['pcm_sha256'],'TRAIN evidence uses different PCM')
            evidence_frames+=old.verify_evidence(e,feature,groups,row['pcm_sha256'])
            if exposure!='targeted':continue
            pcm=np.frombuffer(audit.bind(REPO/row['pcm_path'],row['pcm_sha256']).read_bytes(),dtype='<f4')
            require(pcm.shape==(96000,) and np.isfinite(pcm).all(),'Saved TRAIN PCM extent differs')
            for i in SAMPLE_ROWS:
                end=int(feature['reference_sample'][i]);want=scalar_periodicity(pcm[end-1920:end],groups['candidate_hz'][i],groups['candidate_mask'][i])
                dc,de=compare_scalar({k:e[k][i] for k in want},want)
                max_corr=max(max_corr,dc);max_channel=max(max_channel,de);samples+=1;active+=int(groups['candidate_mask'][i].sum())
    require(samples==4608,'Fixed raw-periodicity replay denominator differs')
    return {'existing':existing,'targeted':target},{'acquired_rows_per_case':list(SAMPLE_ROWS),'common_row_ordinals':[0,76,190],
        'cases':1536,'frames':samples,'active_candidate_windows':active,'correlation_absolute_tolerance':CORRELATION_ATOL,
        'channel_absolute_tolerance':CHANNEL_ATOL,'maximum_correlation_difference':max_corr,'maximum_channel_difference':max_channel,
        'exact_valid_period_anchor':True,'all_training_evidence_arithmetic_frames':evidence_frames,
        'scope':'Saved synthetic TRAIN PCM only; three fixed acquired windows per new-bank case; no full extractor or natural PCM replay'}


def audit_training(audit,run,manifest,natural_targets,banks,table,records,strata):
    schedule=natural_targets['schedule'];expected={k:natural_targets['expanded7__'+k] for k in ('acceptable','covered','nearest')};expected['schedule']=schedule
    saved=load_npz(audit,run/'training-targets.npz',manifest['training_targets_sha256'])
    require(set(saved)==set(expected),'Natural target field inventory differs')
    for k,v in expected.items():require(saved[k].dtype==v.dtype and np.array_equal(saved[k],v),'Natural targets differ: '+k)
    require(manifest['scheduled_rows']==len(schedule)==132466,'Natural union schedule extent differs')
    sy=load_npz(audit,run/'synthetic-targets.npz',manifest['synthetic_targets_sha256'])
    require(set(sy)=={a+'__'+k for a in banks for k in ('acceptable','covered','nearest')},'Synthetic target fields differ')
    for arm,b in banks.items():
        for k in ('acceptable','covered','nearest'):
            require(sy[arm+'__'+k].dtype==b[k].dtype and np.array_equal(sy[arm+'__'+k],b[k]),'Synthetic supervision differs: '+arm+'/'+k)
    keys={f'seed{s}__epoch{e}' for s in SEEDS for e in range(12)}
    schedules={name:load_npz(audit,bound(audit,manifest,run/(name+'-schedules.npz'))) for name in ('natural','existing-synthetic','targeted-synthetic')}
    require(all(set(v)==keys for v in schedules.values()),'Schedule array inventory differs')
    sampling=load_npz(audit,bound(audit,manifest,run/'targeted-sampling-schedules.npz'))
    require(set(sampling)=={k+'__'+s for k in keys for s in ('indices','origin')},'Targeted origin schedule inventory differs')
    exposures=audit.json(bound(audit,manifest,run/'targeted-schedule-exposure.json'))
    require(set(exposures)==keys,'Targeted schedule exposure inventory differs')
    for seed in SEEDS:
        for epoch in range(12):
            key=f'seed{seed}__epoch{epoch}';order=pool.natural_order(seed,epoch)
            require(np.array_equal(schedules['natural'][key],order),'Natural permutation differs')
            target=targeted_schedule(strata,table['case_index'],records,seed,epoch)
            for field,values in target.items():
                require(sampling[key+'__'+field].dtype==values.dtype and np.array_equal(sampling[key+'__'+field],values),'Targeted sampling/origin differs')
            compare_nested(audit,sampling_exposure(target,strata,table,1536),exposures[key],key+'/sampling exposure')
            for exposure in ('existing','targeted'):
                want=pool.synthetic_order(seed,epoch) if exposure=='existing' else target['indices']
                got=schedules[exposure+'-synthetic'][key]
                require(got.dtype==np.int64 and np.array_equal(got,want),'Synthetic fit order differs')
                b=banks[exposure];flat=got.ravel();require(b['common'][flat].all(),'Noncommon synthetic rows trained')
                observed={'draws':len(flat),'unique_rows':len(np.unique(flat)),'duplicate_draws':len(flat)-len(np.unique(flat)),
                    'covered_rows':int(b['covered'][flat].sum()),'per_case':np.bincount(b['scenario_index'][flat],minlength=96 if exposure=='existing' else 1536).tolist(),
                    'per_region':{str(r):int((b['region'][flat]==r).sum()) for r in np.unique(b['region'])}}
                covered=saved['covered'][schedule][order]
                require(all(covered[j:j+4096].any() for j in range(0,len(order),4096)),'A batch would skip the fixed optimizer update')
                denoms={'natural_covered':int(covered.sum()),'synthetic_covered':int(b['covered'][flat].sum())}
                for mode in ('off','on'):
                    fit=manifest['fits'][f'{exposure}__{mode}__seed{seed}']
                    compare_nested(audit,observed,fit['synthetic_exposure_by_epoch'][epoch],key+'/'+exposure+'/'+mode+'/exposure')
                    compare_nested(audit,denoms,fit['loss_denominators'][epoch],key+'/'+exposure+'/'+mode+'/loss denominators')
    return {'natural_union_rows':132466,'natural_grouped_zero_rows':int((~saved['covered'][schedule]).sum()),
        'synthetic_banks':{a:{'rows':len(b['covered']),'covered_rows':int(b['covered'].sum())} for a,b in banks.items()},
        'natural_schedule_arrays':36,'synthetic_schedule_arrays':72,'targeted_origin_arrays':36,
        'synthetic_draws_per_epoch':16896,'epochs':12,'updates_per_fit':396,'targets_schedules_exposures_exact':True,
        'optimizer_replay_performed':False,'train_logit_diagnostics_independently_recomputed':False,'counterfactual_scope':'bank and sampling change together; no isolated oversampling claim'}


def validate_resources(m):
    start=historical.dt(m['sources_frozen_at']);deadline=historical.dt(m['deadline_utc'])
    require((deadline-start).total_seconds()==10800,'Scientific deadline is not exactly three hours from source freeze')
    terminal=historical.dt(m['failed_at'] if m['status']=='failed' else m['completed_at'])
    require(start<=terminal<=deadline,'Scientific terminal timestamp exceeds frozen wall budget')
    require(0<=m['elapsed_seconds']<=10800 and 0<=m.get('fit_elapsed_seconds',0)<=3600,'Scientific wall/fit budget exceeded')
    for key,limit in [('peak_observed_python_rss_bytes',16*1024**3),('peak_observed_node_rss_bytes',2*1024**3),('peak_observed_mps_allocated_bytes',8*1024**3),('output_bytes',8*1024**3)]:
        require(0<=m[key]<=limit,'Observed scientific resource ceiling exceeded: '+key)
    return {'scientific_seconds':m['elapsed_seconds'],'fit_seconds':m.get('fit_elapsed_seconds',0),
        **{k:m[k] for k in ('peak_observed_python_rss_bytes','peak_observed_node_rss_bytes','peak_observed_mps_allocated_bytes','output_bytes')},
        'scope':'Observed process/resource samples, not continuous whole-machine bounds'}


def failed_inventory(run,m):
    """Failure is preserved, never reclassified as completed qualification."""
    require(m['status']=='failed' and isinstance(m.get('failure'),str) and m['failure'],'Failure lacks its recorded exception')
    require(set(m.get('fits',{}))<=set(FIT_NAMES) and set(m.get('fit_attempts',{}))<=set(FIT_NAMES),'Failure attempted undeclared fits')
    checkpoints={p.name for p in (run/'checkpoints').glob('*.pt')}
    require(checkpoints<={n+'.pt' for n in FIT_NAMES},'Failure contains undeclared checkpoints')
    if (run/'challenge').exists():
        require((run/'checkpoint-freeze.json').is_file() and set(m['fits'])==set(FIT_NAMES),'Fresh inputs existed before every checkpoint')
    require(not m.get('development_references_read_for_final_scoring') or (run/'prediction-freeze.json').is_file(),'Final scoring preceded prediction freeze')
    return {'status':'failed','failure':m['failure'],'completed_fits':len(m.get('fits',{})),
        'attempted_fits':len(m.get('fit_attempts',{})),'fresh_directory_present':(run/'challenge').exists(),
        'qualification_complete':False,'no_retry_performed_by_auditor':True}


def validate_fresh_declaration(cache,spec,manifest,freeze_relative,freeze_sha):
    require(cache['schema']=='phenometrix-exposure-reserved-input-cache.v1' and cache['status']=='complete' and
        cache['reference_content_read'] is False and cache['model_outputs_read'] is False,'Fresh cache phase/schema differs')
    require(spec['records']==old.fresh_records() and len(cache['records'])==12 and
        [r['id'] for r in cache['records']]==[r['id'] for r in spec['records']],'Reserved complete configuration inventory differs')
    require(hashlib.sha256(historical.canonical(spec)).hexdigest()==manifest['challenge_spec_sha256']==cache['spec_sha256']==cache['challenge_spec_sha256'],
        'Fresh specification bytes differ')
    require(cache['source_sha256']==manifest['source_sha256'] and cache['deadline_utc']==manifest['deadline_utc'] and
        cache['training_bank_manifest_sha256']==BANK_IDENTITIES['manifest.json'],'Fresh source/bank/deadline differs')
    require(cache['freeze_kind']=='checkpoint' and cache['freeze_path']==cache['checkpoint_freeze_path']==freeze_relative and
        cache['freeze_sha256']==cache['checkpoint_freeze_sha256']==freeze_sha,'Fresh checkpoint alias differs')
    require(historical.dt(manifest['checkpoints_frozen_at'])<=historical.dt(cache['started_at'])<=historical.dt(cache['completed_at'])<=
        historical.dt(manifest['predictions_frozen_at']),'Fresh generation preceded final checkpoints or followed predictions')
    for i,(record,parameters) in enumerate(zip(cache['records'],spec['records'])):
        require(record['case_index']==i and record['parameters_sha256']==hashlib.sha256(historical.canonical(parameters)).hexdigest() and
            record['feature_frames']==196 and record['common_eligible_frames']==191,'Fresh metadata/support differs')
    return True


def validate_phase_bindings(audit,run,m,initial):
    phases={n:audit.json(run/(n+'.json')) for n in ('input-cache-freeze','prerequisites','training-freeze','checkpoint-freeze','prediction-freeze')}
    inp,prereq,training,checkpoint,prediction=(phases[n] for n in phases)
    old.phase_order(m,initial,inp,prereq,training,checkpoint,prediction)
    for phase in (inp,training,checkpoint,prediction):
        require(phase['source_sha256']==m['source_sha256'],'Source changed between scientific phases')
    require(checkpoint['schema']=='phenometrix-exposure-checkpoint-freeze.v1' and checkpoint['source_freeze_path']==str((run/'source-freeze.json').relative_to(REPO)) and
        checkpoint['source_freeze_sha256']==base.sha(run/'source-freeze.json') and checkpoint['frozen_at']==m['checkpoints_frozen_at'],'Checkpoint source linkage differs')
    for key in ('training_spec_sha256','training_bank_manifest_sha256','challenge_spec_sha256','arms','seeds','maximum_fits','deadline_utc'):
        require(checkpoint[key]==m[key],'Checkpoint contract changed: '+key)
    expected={n:{'path':str((run/'checkpoints'/(n+'.pt')).relative_to(REPO)),'sha256':f['weights_sha256']} for n,f in m['fits'].items()}
    require(set(expected)==set(FIT_NAMES) and checkpoint['checkpoints']==expected and prediction['fits']==m['fits'],'Final checkpoint inventory differs')
    require(prediction['prediction_sha256']==m['prediction_sha256'],'Frozen predictions changed')
    for phase in (inp,training,prediction):
        require(all(m['input_sha256'].get(p)==h for p,h in phase['input_sha256'].items()),'Input changed between phases')
    required_training={'training-targets.npz','synthetic-targets.npz','natural-schedules.npz','existing-synthetic-schedules.npz',
        'targeted-synthetic-schedules.npz','targeted-sampling-schedules.npz','targeted-schedule-exposure.json'}
    for name in required_training:
        p=run/name;require(training['input_sha256'].get(str(p.relative_to(REPO)))==base.sha(p),'Schedule/target was not frozen before fitting: '+name)
    references=set()
    for split in ('train','development'):
        _,_,metadata=base.load_cache(audit,split)
        paths={str((ROOT/'data/ptdb-tug'/r['path']).relative_to(REPO)) for r in metadata.values()};references|=paths
        refs={str(Path(p).relative_to(REPO)) if Path(p).is_absolute() else p:h for p,h in prereq['reference_sha256'][split].items()}
        require(set(refs)==paths,'Prerequisite REF-role inventory differs')
        require(all(m['input_sha256'].get(p)==h and prediction['input_sha256'].get(p)==h for p,h in refs.items()),'Prerequisite reference identity changed')
    retention.no_reference_inputs(inp['input_sha256'],references)
    inventory=audit.json(bound(audit,m,run/'analytic-inventory.json'))
    require({k:len(v) for k,v in inventory.items()}==old.GROUP_COUNTS,'Analytic inventory differs')
    expected_predictions={'predictions/'+n+'.npz' for n in METHODS}|{r['prediction_path'] for rows in inventory.values() for r in rows}
    require(set(m['prediction_sha256'])==expected_predictions and len(expected_predictions)==103,'Prediction freeze omitted a case/method')
    for p,h in m['prediction_sha256'].items():audit.bind(run/p,h)
    cm=audit.json(bound(audit,m,run/'challenge/manifest.json'));spec=audit.json(bound(audit,m,run/'challenge/spec.json'))
    validate_fresh_declaration(cm,spec,m,str((run/'checkpoint-freeze.json').relative_to(REPO)),base.sha(run/'checkpoint-freeze.json'))
    return prereq


def audit_run(run,output):
    started=time.monotonic();run=base.inside(RUNS,run);output=base.inside(run,output)
    require(not output.exists(),'Refuses to overwrite a prior audit receipt');torch.set_num_threads(1)
    audit=ExposureAudit();m=audit.json(run/'manifest.json')
    require(m['schema']=='phenometrix-exposure-study.v1' and m['status'] in ('complete','prerequisite_no_go','failed'),
        'Only a terminal new scientific attempt may be audited')
    require(m['arms']==list(ARMS) and m['seeds']==list(SEEDS) and m['maximum_fits']==12 and m['predesignated_candidate']=='targeted__on',
        'Declared arm/seed/candidate inventory differs')
    require(m['reserved_content_read'] is False and m['training_generation_configurations']==0,'Study accessed new natural content or regenerated TRAIN')
    resources=validate_resources(m)
    initial=audit.json(run/'source-freeze.json')
    require(initial['schema']=='phenometrix-exposure-source-freeze.v1' and initial['status']=='sources_frozen' and initial['fits']=={} and not initial.get('fit_attempts'),'Initial source freeze differs')
    for key in ('source_sha256','supervisor_sha256','training_spec_sha256','training_bank_manifest_sha256','challenge_spec_sha256','arms','seeds','maximum_fits','deadline_utc','sources_frozen_at'):
        require(initial[key]==m[key],'Initial/final identity differs: '+key)
    require(m['training_bank_manifest_sha256']==BANK_IDENTITIES['manifest.json'] and m['challenge_spec_sha256']=='23987c7d14dbd4490f1c73ec4eef2394d30aae5518021188c6962ae260022462','Frozen bank/reserved specification differs')
    own=source_hashes();require(all(m['source_sha256'].get(p)==h for p,h in own.items()),'Independent audit interpretation was not source-frozen')
    for p,h in {**m['source_sha256'],**m['supervisor_sha256']}.items():audit.bind(REPO/p,h);audit.bind(run/'source-snapshot'/p,h)
    for p,h in m['input_sha256'].items():audit.bind(REPO/p,h)
    prior=audit.json(PRIOR/'manifest.json',PRIOR_SHA)
    require(prior['status']=='complete' and m['prior_manifest']==prior and m['historical_input_sha256']==prior['input_sha256'],'Pinned prior experiment differs')
    require(all(m['source_sha256'].get(p)==h for p,h in prior['source_sha256'].items()),'Historical source changed')
    result={'schema':'phenometrix-exposure-independent-audit.v1','status':'complete','pass_result':True,
        'scientific_status':m['status'],'run_manifest_sha256':base.sha(run/'manifest.json'),
        'source_freeze_sha256':base.sha(run/'source-freeze.json'),'scientific_resources':resources,
        'training_generation_performed':False,'waveform_generation_performed':False,'fitting_performed':False,
        'natural_reserved_content_read':False,'inference_performed':False,'scientific_gate_pass':False,
        'limitations':['Artifact/source chronology is not an operating-system access log.',
          'Audit pass validates the declared result; it does not waive any scientific gate or promote a learned model.',
          'Optimizer updates and TRAIN before/after logit diagnostics are not replayed; exact targets, schedules, exposure and checkpoint records are checked.',
          'Exposure changes the bank and sampling together; no isolated oversampling effect is identified.',
          'Raw-periodicity replay is restricted to the frozen 4,608 synthetic TRAIN windows, not all features.']}
    if m['status']=='failed':
        result['failure_inventory']=failed_inventory(run,m)
        result['qualification_complete']=False
    else:
        cache,table,records,strata,support,bank_report=audit_bank(audit,m)
        expected_feasibility=bank_audit.expected_report(table,records)['feasibility']
        compare_nested(audit,expected_feasibility,m['train_feasibility'],'Saved new-bank TRAIN support')
        result['targeted_bank']=bank_report
        if m['status']=='prerequisite_no_go':
            old.no_fit_artifacts(run,m)
            require(support['pass_result'] and m['development_references_read_for_prerequisites'] is True and
                m['development_references_read_for_final_scoring'] is False,'Prerequisite no-go crossed invalid reference phase')
            prereq=audit.json(bound(audit,m,run/'prerequisites.json'))
            previous=audit.json(PRIOR/'metrics.json',prior['metrics_sha256'])
            compare_nested(audit,prereq['coverage'],previous['coverage'],'Unchanged prerequisite coverage')
            compare_nested(audit,prereq['headroom'],previous['headroom']['expanded7'],'Unchanged prerequisite headroom')
            checks={'train_support':True,'raw_fixed_voicing_oracle_headroom':prereq['headroom']['fixed-slots']['pass'],
                'train_grouping_loss':prereq['coverage']['train']['expanded7']['pass'],'development_grouping_loss':prereq['coverage']['development']['expanded7']['pass']}
            compare_nested(audit,checks,prereq['checks'],'No-go prerequisite components')
            require(prereq['pass'] is False and not all(checks.values()),'No-go does not follow failed prerequisites')
            require(historical.dt(m['sources_frozen_at'])<=historical.dt(m['prerequisite_reference_read_at'])<=historical.dt(prereq['frozen_at'])<=historical.dt(m['completed_at']),'No-go reference chronology differs')
            result.update(no_fit_no_fresh_verified=True,new_fits=0,fresh_cases_generated=0)
        else:
            require(support['pass_result'] and m['fresh_evaluation_configurations']==12,'Fits/fresh phase lacks passed TRAIN support')
            prereq=validate_phase_bindings(audit,run,m,initial)
            metrics=audit.json(run/'metrics.json',m['metrics_sha256']);require(metrics['schema']=='phenometrix-exposure-metrics.v1','Metric schema differs')
            weights,checkpoint_report=verify_checkpoints(audit,run,m,prior)
            inference=verify_inference(audit,run,m,weights)
            cells,coverage,headroom,targets,natural=old.natural_and_coverage(audit,run,m,metrics,prior)
            compare_nested(audit,coverage,prereq['coverage'],'Prerequisite coverage')
            compare_nested(audit,headroom,prereq['headroom'],'Prerequisite headroom')
            compare_nested(audit,metrics['exclusions'],prereq['exclusions'],'Reference exclusions')
            checks={'train_support':True,'raw_fixed_voicing_oracle_headroom':headroom['fixed-slots']['pass'],
                'train_grouping_loss':coverage['train']['expanded7']['pass'],'development_grouping_loss':coverage['development']['expanded7']['pass']}
            require(all(checks.values()),'Fitting crossed a failed prerequisite')
            compare_nested(audit,checks,prereq['checks'],'Prerequisite components');compare_nested(audit,checks,metrics['decision']['prerequisites'],'Final prerequisite components')
            banks,raw_replay=reconstruct_banks(audit,run,m,cache,table,records)
            training=audit_training(audit,run,m,targets,banks,table,records,strata)
            analytic=old.audit_analytic(audit,run,m,metrics,prior)
            gate=old.natural_gate(cells);compare_nested(audit,gate,{k:metrics['decision'][k] for k in gate},'Natural gate')
            accepted=gate['pass'] and analytic['gates']['pass'] and all(checks.values())
            audit.equal(accepted,metrics['decision']['promising_for_next_study'],'Combined scientific decision')
            controls=3*natural['development_rows']+analytic['control_frame_pairs']
            compare_nested(audit,{'state_hz_frame_pairs':controls,'mismatches':0,'all_exact':True},m['control_reproduction'],'Historical frozen controls')
            natural['checkpoint_inference_performed']=True
            result.update(new_fits=12,fresh_cases_generated=12,checkpoint_checks=checkpoint_report,inference=inference,inference_performed=True,
                training=training,raw_periodicity_replay=raw_replay,natural=natural,analytic=analytic,natural_gate=gate,
                scientific_gate_pass=accepted,control_frame_pairs=controls,qualification_complete=True)
    audit.close()
    result.update(audited_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
        numeric_comparisons=audit.comparisons,maximum_numeric_difference=audit.max_rate_delta,
        bound_artifacts=len(audit.files),bound_artifact_sha256=audit.files,audit_source_sha256=own,peak_observed_python_rss_bytes=audit.peak_rss_bytes)
    output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run=base.inside(RUNS,args.run);output=base.inside(run,args.output)
    require(not output.exists(),'Refuses to overwrite a prior audit receipt')
    def stop(signum,frame):raise TimeoutError('Independent audit thirty-minute wall limit reached')
    signal.signal(signal.SIGALRM,stop);signal.alarm(MAX_AUDIT_SECONDS)
    began=time.monotonic()
    try:
        result=audit_run(run,output)
        print(json.dumps({k:result[k] for k in ('status','pass_result','scientific_status','scientific_gate_pass','numeric_comparisons','maximum_numeric_difference','elapsed_seconds','bound_artifacts')},indent=2))
    except BaseException as error:
        if not output.exists():
            output.write_text(json.dumps({'schema':'phenometrix-exposure-independent-audit.v1','status':'failed','pass_result':False,
                'failure':repr(error),'elapsed_seconds':time.monotonic()-began,'audit_source_sha256':source_hashes(),
                'run_manifest_sha256':base.sha(run/'manifest.json'),'scientific_result_modified':False},indent=2,sort_keys=True)+'\n')
        raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
