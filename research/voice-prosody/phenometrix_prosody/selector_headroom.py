"""Train/development-only retrospective selector headroom; no fitting or test access.

All original training forward predictions are frozen before this run reads labels.
Development predictions are reused byte-for-byte. Oracles see reference outcomes
and are upper bounds, never learned/selectable estimators or validation results.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import platform
import importlib.metadata
import numpy as np
from .model import ModelConfig, numpy_step, FEATURE_NAMES_V2

REPO = Path(__file__).resolve().parents[3]
RESEARCH = REPO/'research/voice-prosody'
WEIGHTS = '06110469a9a643fb16806905569ebc939db2c0bdd5b3a6a1593b31789453d10d'
CONDITIONS = ('clean','noise10db','first_difference','clip01')
SPEAKERS = {'train':tuple(f'{sex}{i:02}' for sex in ('F','M') for i in (2,4,7,8,9,10)),
            'development':('F01','F03','M01','M03')}
PINS = {'delayed-train-features/manifest.json':'35e92d701dcd52891c256fa99b9567c5db4d80e14f4ef71858822b470edf5a87',
        'delayed-development-features/manifest.json':'31f51e93b59f9a70c6f4b8c66cbf20b84fef900e44c75b8134aefad0cade3e8c',
        'delayed-development-evaluation/manifest.json':'c1d178ad5eac9f337e770f325041e02d04e326f56d687d4c77460063d07fc2a5',
        'delayed-development-evaluation/prediction-freeze.json':'64358ddf2a654a6c20ef3023c3eccdda92612707d465cb422a20163e99258d2e',
        'hybrid-natural-v1/manifest.json':'380b43dc1968bccccb114fab38cb3b6f0eced50e3e26d148e56eeb9e27159439',
        'hybrid-natural-v1/prediction-freeze.json':'3a7be525cb503f48d146e1af2f6bcc8feff62825038c9e8d5ed273e75eea6bb0',
        'delayed-mlp-seed20261022/manifest.json':'73533700a756e295876a3145017b3d263c6af0b73d962ec2bf9b7cdd9ac2d61f'}
POOLS = ('six_candidates','raw_yin','selected_mlp_or_raw_yin','six_plus_raw_yin')
F0_BINS = ('unvoiced','voiced_below50','voiced_50_100','voiced_100_150','voiced_150_200',
           'voiced_200_300','voiced_300_400','voiced_400_700','voiced_above700')


def require(value, reason):
    if not value: raise ValueError(reason)


def now(): return datetime.now(timezone.utc).isoformat()

def digest(data): return hashlib.sha256(data).hexdigest()

def write(path, value):
    with Path(path).open('x') as f: json.dump(value,f,separators=(',',':'),allow_nan=False);f.write('\n')


def inside(root, relative):
    p=(root/relative).resolve();require(p.is_relative_to(root.resolve()),'Artifact escaped root');return p


class Ledger:
    def __init__(self): self.files={};self.labels_started=False;self.predictions_frozen=False
    def read(self,path,role,expected=None,split=None):
        path=Path(path).resolve()
        require(not any(p in ('sealed_test','sealed-test') for p in path.parts),'Consumed test content is excluded')
        if role=='reference':
            require(self.predictions_frozen,'All predictions must freeze before first label read')
            require(split in SPEAKERS and f'/records/{split}/' in str(path),'Reference split/path not admitted')
            require(path.name.split('_')[1] in SPEAKERS[split],'Reference speaker not admitted')
            self.labels_started=True
        data=path.read_bytes();sha=digest(data)
        require(expected is None or sha==expected,'Input bytes changed: '+str(path))
        key=str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)
        if key in self.files: require(self.files[key]['sha256']==sha,'Input changed during audit')
        self.files[key]={'sha256':sha,'bytes':len(data),'role':role}
        return data
    def json(self,*args,**kwargs):return json.loads(self.read(*args,**kwargs))


def model_arrays(feature,config,weights,batch_size=4096):
    """Framewise fixed-weight NumPy forward pass; never receives reference labels."""
    require(config.kind=='framewise-mlp','Only frozen framewise model allowed')
    x,mask=feature['features'],feature['candidate_mask'];ps=[];indices=[]
    for start in range(0,len(x),batch_size):
        values=x[start:start+batch_size];valid=mask[start:start+batch_size]
        logits,choices,_=numpy_step(config,weights,values,valid,np.zeros((len(values),config.hidden_size),np.float32))
        require(np.isfinite(logits).all() and np.isfinite(choices).all(),'Nonfinite frozen forward pass')
        probability=np.exp(-np.logaddexp(0.,-logits[:,0].astype(np.float64)))
        index=np.argmax(np.where(valid,choices,-np.inf),axis=1)
        index[(probability<.5)|~valid.any(axis=1)]=-1
        ps.extend(probability);indices.extend(index)
    p=np.array(ps,np.float64);index=np.array(indices,np.int16)
    state=np.where(p<.5,0,np.where(index>=0,1,2)).astype(np.int8)
    hz=np.full(len(p),np.nan,np.float64);voiced=index>=0
    hz[voiced]=feature['candidate_hz'][np.flatnonzero(voiced),index[voiced]]
    return dict(target_sample=feature['reference_sample']-960,voicing_probability=p,candidate_index=index,state=state,f0_hz=hz)


def cached_arrays(rows,targets):
    require([r['sample'] for r in rows]==targets.tolist(),'Cached prediction target mismatch')
    p=np.array([r['voicing_probability'] for r in rows],np.float64)
    state=np.array([{'unvoiced':0,'voiced':1,'unavailable':2}[r['state']] for r in rows],np.int8)
    hz=np.array([r['f0_hz'] if r['f0_hz'] is not None else np.nan for r in rows],np.float64)
    index=np.array([r.get('candidate_index') if r.get('candidate_index') is not None else -1 for r in rows],np.int16)
    require(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Invalid cached voicing probability')
    require(all(r.get('correctness_probability') is None for r in rows),'Correctness is not calibrated')
    return dict(target_sample=targets,voicing_probability=p,candidate_index=index,state=state,f0_hz=hz)


def load_feature(ledger,folder,record):
    data=ledger.read(folder/'features'/(record['recording_id']+'.npz'),'features',record['feature_sha256'])
    with np.load(BytesIO(data),allow_pickle=False) as f:feature={k:f[k].copy() for k in f.files}
    x,hz,mask=feature['features'],feature['candidate_hz'],feature['candidate_mask'];ends=feature['reference_sample'];n=record['sample_count']
    require(int(feature['sample_rate_hz'])==48000 and int(feature['pcm_sample_count'])==n
            and np.array_equal(ends,np.arange(2208,n+1,480)) and list(feature['feature_names'])==list(FEATURE_NAMES_V2)
            and x.shape==(len(ends),57) and x.dtype==np.float32 and np.isfinite(x).all()
            and hz.shape==mask.shape==(len(ends),6) and mask.dtype==bool and np.isfinite(hz[mask]).all()
            and ((hz[mask]>=50)&(hz[mask]<=700)).all() and np.isnan(hz[~mask]).all(),'Feature identity/grid/shape violated')
    baseline=ledger.json(folder/'predictions'/(record['recording_id']+'.json'),'cached_raw_yin',record['prediction_sha256'])
    require(baseline['recording_id']==record['recording_id'] and [r['sample'] for r in baseline['yin_evidence']]==ends.tolist(),'Raw YIN endpoint mismatch')
    return feature,np.array([r['raw_f0_hz'] for r in baseline['yin_evidence']],np.float64)


def validate_pair(feature,mlp,raw_yin,hybrid=None):
    p,state,hz,index=(mlp[k] for k in ('voicing_probability','state','f0_hz','candidate_index'))
    mask=feature['candidate_mask'];expected=np.where(p<.5,0,np.where(mask.any(axis=1),1,2))
    require(np.array_equal(state,expected),'Frozen original voicing/availability differs')
    voiced=state==1
    require(((index[voiced]>=0)&(index[voiced]<6)).all() and mask[np.flatnonzero(voiced),index[voiced]].all()
            and np.array_equal(hz[voiced],feature['candidate_hz'][np.flatnonzero(voiced),index[voiced]])
            and np.isnan(hz[~voiced]).all(),'Original MLP pitch is not its selected candidate')
    valid_yin=np.isfinite(raw_yin)&(raw_yin>=50)&(raw_yin<=700)
    hybrid_state=np.where(p<.5,0,np.where(valid_yin,1,2));hybrid_hz=np.where(hybrid_state==1,raw_yin,np.nan)
    if hybrid is not None:
        require([r['sample'] for r in hybrid]==mlp['target_sample'].tolist(),'Hybrid target mismatch')
        require(all(r['voicing_probability']==float(p[i]) and r['correctness_probability'] is None
                    and r['state']==('unvoiced','voiced','unavailable')[hybrid_state[i]]
                    and r['f0_hz']==(float(raw_yin[i]) if hybrid_state[i]==1 else None) for i,r in enumerate(hybrid)),
                'Cached hybrid differs from original frozen gate/raw YIN')
    return hybrid_state.astype(np.int8),hybrid_hz


def observed_frames(reference_rows,n,feature,mlp,raw_yin):
    reference_rows=np.asarray(reference_rows,np.float64)
    require(reference_rows.ndim==2 and reference_rows.shape[1]==4,'Expected four provider reference columns')
    hz,probability=reference_rows[:,:2].T
    valid=np.isfinite(reference_rows).all(axis=1)&(((hz==0)&(probability==0))|((hz>0)&(probability==1)))
    targets=768+480*np.arange(len(hz));eligible=(targets>=3072)&(targets+1536<=n)
    selected=np.flatnonzero(valid&eligible);at=np.searchsorted(mlp['target_sample'],targets[selected])
    require((at<len(mlp['target_sample'])).all() and np.array_equal(mlp['target_sample'][at],targets[selected]),'Eligible reference has no exact input target')
    hy_state,hy_hz=validate_pair(feature,mlp,raw_yin)
    truth=hz[selected];v=probability[selected]==1;p=mlp['voicing_probability'][at];g=p>=.5
    def pitch_good(values):
        good=np.zeros(len(selected),bool);ok=v&np.isfinite(values)&(values>0)
        good[ok]=np.abs(1200*np.log2(values[ok]/truth[ok]))<=50;return good
    original=mlp['f0_hz'][at];yin=raw_yin[at]
    yin_valid=np.isfinite(yin)&(yin>=50)&(yin<=700)
    candidate=feature['candidate_hz'][at];mask=feature['candidate_mask'][at]
    good=np.zeros(candidate.shape,bool)
    for i in range(6):good[:,i]=mask[:,i]&pitch_good(candidate[:,i])
    six=good.any(axis=1);yg=yin_valid&pitch_good(yin);og=pitch_good(original)
    pools={'six_candidates':six,'raw_yin':yg,'selected_mlp_or_raw_yin':og|yg,'six_plus_raw_yin':six|yg}
    ms,hs=mlp['state'][at],hy_state[at]
    mc=(~v&(ms==0))|(v&(ms==1)&og);hc=(~v&(hs==0))|(v&(hs==1)&yg)
    def octave(values,state):
        result=np.zeros(len(selected),bool);ok=v&(state==1);signed=1200*np.log2(values[ok]/truth[ok]);nearest=np.rint(signed/1200)
        result[ok]=(nearest!=0)&(np.abs(signed-nearest*1200)<=50);return result
    frame={'v':v,'g':g,'truth':truth,'mlp_state':ms,'hybrid_state':hs,'mlp_correct':mc,'hybrid_correct':hc,
           'mlp_octave':octave(original,ms),'hybrid_octave':octave(yin,hs),'six_available':mask.any(axis=1),
           'raw_yin_available':yin_valid,**{'coverage_'+k:values for k,values in pools.items()}}
    excluded={'excluded_invalid_reference':int((~valid).sum()),'excluded_boundary':int((valid&~eligible).sum())}
    return frame,excluded


def frame_counts(f,select=None):
    if select is None:select=np.ones(len(f['v']),bool)
    v,g,mc,hc=(f[k][select] for k in ('v','g','mlp_correct','hybrid_correct'));u=~v
    ms,hs=f['mlp_state'][select],f['hybrid_state'][select]
    count=lambda x:int(np.sum(x))
    c={'eligible_valid_frames':len(v),'reference_voiced_frames':count(v),'reference_unvoiced_frames':count(u),
       'reference_outside_supported_bounds':count(v&((f['truth'][select]<50)|(f['truth'][select]>700))),
       'mlp_correct_frames':count(mc),'hybrid_correct_frames':count(hc),'mlp_voiced_success_count':count(v&mc),'hybrid_voiced_success_count':count(v&hc),
       'mlp_false_positive_count':count(u&(ms==1)),'hybrid_false_positive_count':count(u&(hs==1)),
       'mlp_unavailable_frames':count(ms==2),'hybrid_unavailable_frames':count(hs==2),
       'shared_false_unvoiced_frames':count(v&~g),'reference_unvoiced_gate_positive_frames':count(u&g),
       'frozen_voicing_perfect_pitch_correct_frames':count((u&~g)|(v&g)),
       'frozen_voicing_perfect_pitch_voiced_success_count':count(v&g),
       'mlp_voiced_pitch_error_count':count(v&(ms==1)&~mc),'hybrid_voiced_pitch_error_count':count(v&(hs==1)&~hc),
       'mlp_octave_error_count':count(f['mlp_octave'][select]),'hybrid_octave_error_count':count(f['hybrid_octave'][select]),
       'six_no_available_pitch_on_voiced':count(v&~f['six_available'][select]),'yin_no_available_pitch_on_voiced':count(v&~f['raw_yin_available'][select])}
    for name,mask in {'both_correct':mc&hc,'mlp_only_correct':mc&~hc,'hybrid_only_correct':~mc&hc,'neither_correct':~mc&~hc}.items():
        c['paired_'+name]=count(mask);c['paired_voiced_'+name]=count(v&mask)
    for pool in POOLS:
        hit=f['coverage_'+pool][select]
        c[pool+'_coverage_count']=count(v&hit)
        c[pool+'_frozen_voicing_voiced_success_count']=count(v&g&hit)
        c[pool+'_frozen_voicing_correct_frames']=count((u&~g)|(v&g&hit))
        c[pool+'_shared_false_unvoiced_covered_frames']=count(v&~g&hit)
        c[pool+'_both_wrong_recoverable_count']=count(v&g&hit&~mc&~hc)
    return c


def rates(c):
    n,v,u=(c.get(k,0) for k in ('eligible_valid_frames','reference_voiced_frames','reference_unvoiced_frames'))
    ratio=lambda a,b:a/b if b else None
    r={}
    for method in ('mlp','hybrid'):
        r[method+'_joint_error']=ratio(n-c[method+'_correct_frames'],n)
        r[method+'_voiced_success_rate']=ratio(c[method+'_voiced_success_count'],v)
        r[method+'_false_positive_rate']=ratio(c[method+'_false_positive_count'],u)
        r[method+'_coverage']=ratio(n-c[method+'_unavailable_frames'],n)
        r[method+'_octave_error_rate_reference_voiced']=ratio(c[method+'_octave_error_count'],v)
    r['shared_false_unvoiced_rate']=ratio(c['shared_false_unvoiced_frames'],v)
    r['frozen_voicing_perfect_pitch_joint_error_floor']=ratio(n-c['frozen_voicing_perfect_pitch_correct_frames'],n)
    r['frozen_voicing_perfect_pitch_voiced_success_ceiling']=ratio(c['frozen_voicing_perfect_pitch_voiced_success_count'],v)
    for pool in POOLS:
        r[pool+'_unconstrained_voiced_coverage']=ratio(c[pool+'_coverage_count'],v)
        r[pool+'_frozen_voicing_joint_error']=ratio(n-c[pool+'_frozen_voicing_correct_frames'],n)
        r[pool+'_frozen_voicing_voiced_success_rate']=ratio(c[pool+'_frozen_voicing_voiced_success_count'],v)
        for method in ('mlp','hybrid'):
            r[pool+'_joint_error_reduction_vs_'+method]=ratio(c[pool+'_frozen_voicing_correct_frames']-c[method+'_correct_frames'],n)
    return r


def merge(parts):
    return {k:sum(p.get(k,0) for p in parts) for k in set().union(*(p.keys() for p in parts))} if parts else {}


def macro(parts):
    row=[rates(p) for p in parts];names=list(row[0])
    vals={k:[p[k] for p in row if p[k] is not None] for k in names}
    return {'cell_count':len(row),'defined_cells':{k:len(v) for k,v in vals.items()},
            'rates':{k:float(np.mean(v)) if v else None for k,v in vals.items()}}


def f0_masks(f):
    v,h=f['v'],f['truth'];return dict(zip(F0_BINS,[~v,v&(h<50),v&(h>=50)&(h<100),v&(h>=100)&(h<150),
        v&(h>=150)&(h<200),v&(h>=200)&(h<300),v&(h>=300)&(h<400),v&(h>=400)&(h<=700),v&(h>700)]))


def aggregate_split(records,speakers):
    cells={s:{c:merge([r['counts'] for r in records if r['speaker']==s and r['condition']==c]) for c in CONDITIONS} for s in speakers}
    all_cells=[cells[s][c] for s in speakers for c in CONDITIONS]
    output={'source_recordings':len({r['source_recording_id'] for r in records}),'condition_records':len(records),
            'pooled':{'counts':merge(all_cells),'rates':rates(merge(all_cells))},'equal_speaker_condition':macro(all_cells),
            'per_speaker_condition':{s:{c:{'counts':cells[s][c],'rates':rates(cells[s][c])} for c in CONDITIONS} for s in speakers},
            'per_speaker':{s:{'pooled_counts':merge(list(cells[s].values())),'equal_condition':macro(list(cells[s].values()))} for s in speakers},
            'per_condition':{c:{'pooled_counts':merge([cells[s][c] for s in speakers]),'equal_speaker':macro([cells[s][c] for s in speakers])} for c in CONDITIONS},
            'by_reference_f0':{}}
    for band in F0_BINS:
        groups={s:{c:merge([r['f0'][band] for r in records if r['speaker']==s and r['condition']==c]) for c in CONDITIONS} for s in speakers}
        flat=[groups[s][c] for s in speakers for c in CONDITIONS]
        output['by_reference_f0'][band]={'pooled_counts':merge(flat),'pooled_rates':rates(merge(flat)),'equal_speaker_condition':macro(flat),
            'per_condition_equal_speaker':{c:macro([groups[s][c] for s in speakers]) for c in CONDITIONS}}
    return output


def run(output,public_output,data_root):
    require(output.resolve().is_relative_to(RESEARCH/'runs'),'Raw audit outputs must stay in ignored research runs')
    require(not output.exists() and not public_output.exists(),'Refuses to overwrite evidence')
    ledger=Ledger();runs=RESEARCH/'runs'
    source_paths=[Path(__file__),RESEARCH/'phenometrix_prosody/__init__.py',RESEARCH/'tests/test_selector_headroom.py',RESEARCH/'phenometrix_prosody/model.py',
        RESEARCH/'phenometrix_prosody/measurement_alignment.py',RESEARCH/'phenometrix_prosody/metrics.py',RESEARCH/'phenometrix_prosody/pilot.py',
        RESEARCH/'phenometrix_prosody/evaluate.py',RESEARCH/'phenometrix_prosody/hybrid_followup.py',RESEARCH/'phenometrix_prosody/hybrid_natural.py',
        RESEARCH/'SELECTOR_EXPERIMENT_PLAN.md',RESEARCH/'pyproject.toml',RESEARCH/'uv.lock',REPO/'pnpm-lock.yaml']
    for path in source_paths:ledger.read(path,'source')
    pinned={name:ledger.json(runs/name,'manifest',pin) for name,pin in PINS.items()}
    caches={split:pinned[f'delayed-{split}-features/manifest.json'] for split in SPEAKERS}
    acquisitions={}
    for split,cache in caches.items():
        require(cache['status']=='complete' and cache['split']==split and cache['reference_or_laryngograph_content_read'] is False
                and tuple(cache['conditions'])==CONDITIONS and cache['measurement_delay_ms']==20,'Input-only cache contract changed')
        require(sorted({r['speaker'] for r in cache['recordings']})==list(SPEAKERS[split]),'Admitted speaker selection changed')
        acquisitions[split]=ledger.json(runs/f'delayed-{split}-features/acquisition-manifest.json','acquisition_metadata',cache['data_manifest_sha256'])
        sources={r['recording_id']:r for r in acquisitions[split]['files'] if r['split']==split and r['role']=='mic'}
        for r in cache['recordings']:
            require(r['source_recording_id'] in sources and r['speaker']==sources[r['source_recording_id']]['speaker']
                    and r['microphone_sha256']==sources[r['source_recording_id']]['sha256'] and r['condition'] in CONDITIONS
                    and r['recording_id']==r['source_recording_id']+'__'+r['condition'],'Cache source/split mismatch')
    model_manifest=pinned['delayed-mlp-seed20261022/manifest.json'];train=caches['train']
    require(model_manifest['test_or_development_labels_used'] is False and model_manifest['split']=='train'
            and model_manifest['weights_sha256']==WEIGHTS and model_manifest['feature_run_manifest_sha256']==PINS['delayed-train-features/manifest.json']
            and model_manifest['feature_hashes']=={r['recording_id']:r['feature_sha256'] for r in train['recordings']},'Original model training identity changed')
    train_refs={r['recording_id']:r['sha256'] for r in acquisitions['train']['files'] if r['role']=='ref' and r['split']=='train'
                and r['recording_id'] in {x['source_recording_id'] for x in train['recordings']}}
    require(model_manifest['reference_hashes']==train_refs,'Original training reference metadata changed')
    import torch
    saved=torch.load(BytesIO(ledger.read(runs/'delayed-mlp-seed20261022/weights.pt','checkpoint',WEIGHTS)),map_location='cpu',weights_only=True)
    config=ModelConfig(**saved['config']);weights={k:v.detach().cpu().numpy().copy() for k,v in saved['state_dict'].items()}
    require(config.kind=='framewise-mlp' and config.feature_version=='v2' and bool(weights['scaling_fitted'])
            and np.isfinite(weights['feature_scale']).all() and (weights['feature_scale']>0).all(),'Frozen scaling/model contract invalid')
    evf=pinned['delayed-development-evaluation/prediction-freeze.json'];hyf=pinned['hybrid-natural-v1/prediction-freeze.json']
    for value in (evf,hyf):
        require(value['status']=='predictions_frozen' and value['split']=='development'
                and value['reference_content_read_before_prediction_freeze'] is False
                and value['feature_manifest_sha256']==PINS['delayed-development-features/manifest.json'],'Development freeze changed')
    require(evf['models']['mlp']['weights_sha256']==WEIGHTS and hyf['models']['original']['weights_sha256']==WEIGHTS,'Wrong development model')
    output.mkdir(parents=True);(output/'train-predictions').mkdir()
    declaration={'schema':'phenometrix-selector-headroom-declaration.v1','created_at':now(),'splits':['train','development'],
        'model_weights_sha256':WEIGHTS,'input_manifest_sha256':PINS,'oracle_tolerance_cents':50,'voicing_threshold':.5,
        'raw_yin':'Historical cached Python librosa raw YIN; not substituted by repaired browser float64 implementation',
        'pools':list(POOLS),'reference_f0_strata':list(F0_BINS),'prefit_gate':'development equal16cell original-six fixed-voicing oracle joint-error reduction versus original MLP >=0.01',
        'source_sha256':{k:v['sha256'] for k,v in ledger.files.items() if v['role']=='source'},
        'new_training':False,'consumed_test_content_access':False,'train_inference':'Original frozen model, batched NumPy; labels absent from forward pass',
        'development_inference':'No new forward pass; existing frozen original predictions'}
    write(output/'declaration.json',declaration)
    frozen={'schema':'phenometrix-selector-headroom-input-freeze.v1','started_at':now(),'records':[],'reference_content_read':False}
    for split in ('train','development'):
        cache=caches[split];folder=runs/f'delayed-{split}-features'
        for i,record in enumerate(cache['recordings']):
            feature,yin=load_feature(ledger,folder,record)
            if split=='train':
                arrays=model_arrays(feature,config,weights);validate_pair(feature,arrays,yin)
                path=output/'train-predictions'/(record['recording_id']+'.npz');np.savez_compressed(path,**arrays)
                identity={'path':str(path.relative_to(output)),'sha256':digest(path.read_bytes())}
            else:
                er,hr=evf['recordings'][i],hyf['recordings'][i]
                require(all(er.get(k)==v and hr.get(k)==v for k,v in record.items()),'Development source/order changed')
                ep=ledger.read(inside(runs/'delayed-development-evaluation',er['evaluated_prediction_path']),'cached_original_prediction',er['evaluated_prediction_sha256'])
                hp=ledger.read(inside(runs/'hybrid-natural-v1',hr['hybrid_prediction_path']),'cached_hybrid_prediction',hr['hybrid_prediction_sha256'])
                original,hybrid=json.loads(gzip.decompress(ep)),json.loads(gzip.decompress(hp))
                require(original['recording_id']==hybrid['recording_id']==record['recording_id']
                        and original['availability_sample']==hybrid['availability_sample']==feature['reference_sample'].tolist()
                        and original['predictions']['mlp']==hybrid['predictions']['mlp_original'],'Frozen original/hybrid disagreement')
                arrays=cached_arrays(original['predictions']['mlp'],feature['reference_sample']-960)
                validate_pair(feature,arrays,yin,hybrid['predictions']['hybrid_original'])
                identity={'original_sha256':er['evaluated_prediction_sha256'],'hybrid_sha256':hr['hybrid_prediction_sha256']}
            frozen['records'].append({'split':split,'recording_id':record['recording_id'],'frames':len(feature['reference_sample']),**identity})
            if (i+1)%100==0:print(f'Frozen inputs {split} {i+1}/{len(cache["recordings"])}',flush=True)
    frozen.update(status='predictions_frozen',predictions_frozen_at=now(),declaration_sha256=digest((output/'declaration.json').read_bytes()))
    write(output/'prediction-freeze.json',frozen);ledger.predictions_frozen=True
    records_by_split={};prediction_lookup={(r['split'],r['recording_id']):r for r in frozen['records']}
    first_label_at=now()
    with (output/'per-record-counts.jsonl').open('x') as out:
        for split in ('train','development'):
            records_by_split[split]=[];cache=caches[split];folder=runs/f'delayed-{split}-features'
            metadata={r['recording_id']:r for r in acquisitions[split]['files'] if r['role']=='ref' and r['split']==split};refs={}
            for i,record in enumerate(cache['recordings']):
                source=record['source_recording_id'];feature,yin=load_feature(ledger,folder,record)
                if split=='train':
                    binding=prediction_lookup[split,record['recording_id']]
                    raw=ledger.read(inside(output,binding['path']),'new_frozen_train_prediction',binding['sha256'])
                    with np.load(BytesIO(raw),allow_pickle=False) as f:arrays={k:f[k].copy() for k in f.files}
                else:
                    er=evf['recordings'][i];raw=ledger.read(inside(runs/'delayed-development-evaluation',er['evaluated_prediction_path']),'cached_original_prediction',er['evaluated_prediction_sha256'])
                    arrays=cached_arrays(json.loads(gzip.decompress(raw))['predictions']['mlp'],feature['reference_sample']-960)
                if source not in refs:
                    meta=metadata[source];require(meta['speaker']==record['speaker'] and meta['split']==split,'Reference metadata changed')
                    raw=ledger.read(inside(data_root,meta['path']),'reference',meta['sha256'],split)
                    refs[source]=np.loadtxt(BytesIO(raw),ndmin=2)
                frames,excluded=observed_frames(refs[source],record['sample_count'],feature,arrays,yin)
                counts={**frame_counts(frames),**excluded}
                result={k:record[k] for k in ('recording_id','source_recording_id','speaker','condition')}
                result.update(counts=counts,f0={name:frame_counts(frames,mask) for name,mask in f0_masks(frames).items()})
                records_by_split[split].append(result);out.write(json.dumps({'split':split,**result},separators=(',',':'))+'\n')
    summaries={s:aggregate_split(records_by_split[s],SPEAKERS[s]) for s in SPEAKERS}
    dev=summaries['development']['equal_speaker_condition'];gap=dev['rates']['six_candidates_joint_error_reduction_vs_mlp']
    require(dev['defined_cells']['six_candidates_joint_error_reduction_vs_mlp']==16,'Incomplete development prefit denominator')
    # Verify all tracked input bytes again, including labels and immutable predictions.
    for name,entry in ledger.files.items():require(digest(inside(REPO,name).read_bytes())==entry['sha256'],'Input changed during audit: '+name)
    write(output/'opened-input-ledger.json',{'files':ledger.files,'reference_content_read_before_prediction_freeze':False})
    report={'schema':'phenometrix-selector-headroom.v1','status':'complete','created_at':now(),'scope':'Retrospective train/development only; consumed test excluded',
        'declaration_sha256':digest((output/'declaration.json').read_bytes()),'prediction_freeze_sha256':digest((output/'prediction-freeze.json').read_bytes()),
        'source_sha256':declaration['source_sha256'],'input_manifest_sha256':PINS,'model_weights_sha256':WEIGHTS,
        'input_ledger':{'path':str((output/'opened-input-ledger.json').relative_to(REPO)),'sha256':digest((output/'opened-input-ledger.json').read_bytes()),'files':len(ledger.files)},
        'raw_record_counts':{'path':str((output/'per-record-counts.jsonl').relative_to(REPO)),'sha256':digest((output/'per-record-counts.jsonl').read_bytes())},
        'environment':{'python':platform.python_version(),**{p:importlib.metadata.version(p) for p in ('numpy','torch')}},
        'chronology':{'predictions_frozen_at':frozen['predictions_frozen_at'],'first_label_phase_at':first_label_at},
        'contracts':{'measurement_delay_ms':20,'support':'[target-20ms,target+20ms); available at exclusive support end',
          'common_mask':'nominal16+10k ms targets; t>=64ms and t+32ms<=N','pitch_tolerance_cents':50,'voicing_threshold':.5,
          'f0_strata':{'labels':list(F0_BINS),'edges':'lower-inclusive upper-exclusive except400..700 includes700; outside50..700 stays in denominator'},
          'aggregation':'pool recording counts within each speaker/condition, then equal48train/equal16development cells; missing stratum denominators explicit',
          'frozen_voicing_oracle':'Reference chooses a covered pitch only when original p(voiced)>=.5; cannot fix false-unvoiced or reference-unvoiced gate-positive errors',
          'unconstrained_coverage':'Reference-voiced proposal hit rate ignoring original voicing; separate from any realizable selector. selected_mlp_or_raw_yin uses the actually emitted MLP pitch (missing when gate is off); six_candidates and raw_yin are input-only ungated proposals.',
          'perfect_pitch_ceiling':'Assumes perfect unbounded pitch at every gate-positive voiced target, even outside supported50..700Hz; theoretical voicing-only ceiling',
          'raw_yin':declaration['raw_yin']},
        'splits':summaries,'prefit_gate':{'split':'development','metric':'equal16cell original-six frozen-voicing oracle joint-error reduction vs original MLP',
          'minimum_absolute_reduction':.01,'observed_absolute_reduction':gap,'pass':gap>=.01,'grouping_coverage_gate':'Not assessed by this audit; separate experiment prerequisite'},
        'limitations':['Oracles use reference outcomes; headroom does not demonstrate learnability, calibrated confidence, or prospective improvement.',
          'Training scores are resubstitution; development is repeatedly inspected exploratory data. No fresh holdout evaluation or test reference access.',
          'Nominal provider reference validity/alignment retained; no offset fitting, reference repair or outcome-based exclusions.',
          'No new training or model search. Original train forward pass only; development reused exactly.']}
    write(output/'summary.json',report);write(public_output,report)
    print(json.dumps({'status':'complete','prefit_gate':report['prefit_gate'],'public_output':str(public_output),'input_files':len(ledger.files)}))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--public-output',type=Path,default=REPO/'benchmarks/voice-prosody/selector-headroom.json')
    p.add_argument('--data',type=Path,default=RESEARCH/'data/ptdb-tug');a=p.parse_args()
    run(a.output.resolve(),a.public_output.resolve(),a.data.resolve())


if __name__=='__main__':main()
