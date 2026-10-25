"""Prespecified matched voicing study, with explicit immutable phase receipts."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import platform
from pathlib import Path
import random
import shutil
import subprocess
import time
import numpy as np
import torch
from . import voicing_robustness_inputs as vi
from .pilot import REPO, RESEARCH, digest, write_json, read_mic, source_identity
from .model import ModelConfig, VoiceCandidateModel, model_metadata
from .train import training_labels, supervised_loss
from .hybrid_validation import extract_input

ROOT = RESEARCH / 'runs'
TRAIN = ROOT / 'delayed-train-features'
ORIGINAL = ROOT / 'delayed-mlp-seed20261022'
DATA = RESEARCH / 'data/ptdb-tug'
PINS = {
 'delayed-mlp-seed20261022/weights.pt':'06110469a9a643fb16806905569ebc939db2c0bdd5b3a6a1593b31789453d10d',
 'delayed-mlp-seed20261022/manifest.json':'73533700a756e295876a3145017b3d263c6af0b73d962ec2bf9b7cdd9ac2d61f',
 'delayed-mlp-export/model.onnx':'3df30ae54f4bf94b34dc7b1ae296864241fab6db7e711a8970198aeefab79c23',
 'delayed-train-features/manifest.json':'35e92d701dcd52891c256fa99b9567c5db4d80e14f4ef71858822b470edf5a87',
 'delayed-development-features/manifest.json':'31f51e93b59f9a70c6f4b8c66cbf20b84fef900e44c75b8134aefad0cade3e8c',
 'hybrid-validation-freeze-v1/freeze.json':'c5c4288cc04c0d7fb46154977627158336e2b1a5b8b496f8700c7a96dd25c67d',
 'hybrid-validation-v1/metrics.json':'cfdf88d02977ee6702696726380b337f6ac1ca762bc7193cbe08040ff4fd731d',
 'hybrid-browser-runtime-v1/manifest.json':'d7404dd13a786243d158693befd670cdf9604d17ad82a540944e7b555b4db486',
}


def now(): return datetime.now(timezone.utc).isoformat()
def rel(path): return str(Path(path).resolve().relative_to(REPO))
def load(path): return json.loads(Path(path).read_text())
def require(ok, message):
    if not ok: raise ValueError(message)
def file_record(path): return dict(path=rel(path), sha256=digest(path))
def verify_file(row):
    p=(REPO/row['path']).resolve()
    require(p.is_relative_to(REPO) and digest(p)==row['sha256'], 'Bound artifact changed: '+row['path'])
    return p


def scientific_sources():
    names=set(source_identity()['source_sha256'])
    names.update('research/voice-prosody/phenometrix_prosody/'+s+'.py' for s in (
        'voicing_robustness','voicing_robustness_inputs','train','model','measurement_alignment',
        'hybrid_validation','hybrid_followup','evaluate','features'))
    names.add('research/voice-prosody/VOICING_ROBUSTNESS_PLAN.md')
    return {name:digest(REPO/name) for name in sorted(names)}


def snapshot(folder, sources):
    for name,sha in sources.items():
        p=folder/'source-snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes((REPO/name).read_bytes());require(digest(p)==sha,'Snapshot changed')


def check_sources(sources):
    for name,sha in sources.items(): require(digest(REPO/name)==sha,'Frozen source changed: '+name)


def inventory(output):
    output.mkdir(parents=True,exist_ok=False)
    pins={name:dict(expected=expected,actual=digest(ROOT/name)) for name,expected in PINS.items()}
    require(all(r['expected']==r['actual'] for r in pins.values()),'Historical artifact identity mismatch')
    train=load(TRAIN/'manifest.json'); acq=load(TRAIN/'acquisition-manifest.json')
    require(digest(TRAIN/'acquisition-manifest.json')==train['data_manifest_sha256'],'Acquisition changed')
    require(train['status']=='complete' and train['split']=='train' and len(train['recordings'])==960,'TRAIN cache differs')
    row=dict(schema='phenometrix-voicing-robustness-inventory.v1',status='complete',at=now(),pins=pins,
        source=source_identity(),train_sources=len({r['source_recording_id'] for r in train['recordings']}),
        train_speakers=sorted({r['speaker'] for r in train['recordings']}),
        platform=platform.platform(),python=platform.python_version(),
        dependencies={k:importlib.metadata.version(k) for k in ('torch','numpy','onnx','onnxruntime','librosa','scipy')},
        mps_available=torch.backends.mps.is_available(),disk_free_bytes=shutil.disk_usage(REPO).free)
    write_json(output/'inventory.json',row);return row


def freeze(output):
    output.mkdir(parents=True,exist_ok=False)
    inv=inventory(output/'inventory')
    require(inv['mps_available'],'Required MPS backend unavailable; no backend substitution')
    recipe=vi.bank_recipe()
    sources=scientific_sources();snapshot(output,sources)
    baseline=ROOT/'voicing-robustness-baseline-v3/audit.json'
    require(baseline.is_file(),'Authenticated fixed-YIN headroom receipt required before freeze')
    plan=dict(baseline_audit=file_record(baseline),schema='phenometrix-voicing-robustness-plan.v1',status='frozen',at=now(),recipe=recipe,
        recipe_sha256=vi.canonical_sha(recipe),source_sha256=sources,pins=PINS,
        inventory=file_record(output/'inventory/inventory.json'),seeds=list(vi.SEEDS),arms=list(vi.ARMS),candidate='R-20261025',
        max_completed_fits=6,epochs=12,batch_size=64,chunk_length=128,learning_rate=.001,gradient_clip_norm=1.,
        architecture=asdict(ModelConfig(kind='framewise-mlp')),scaling='copy authenticated original TRAIN mean/scale; never refit',
        objective='unweighted BCE(valid) + CE(valid covered within 50 cents)',final_epoch_only=True,
        all_ptdb_exposed=True,external_access='documentation/metadata only until audited development passes',
        model_development_after_failed_gate=False,model_threshold=.5,pitch='unchanged raw YIN, inclusive 50-700 Hz',
        voicing_event_spec=dict(seconds=3,harmonic_interval=[1,2],fundamentals=[90,180,360],phases=[0,.7],
            amplitudes=[.1,.05,.025],backgrounds=['silence','white_noise_sd_0.003'],seed_namespace=vi.NAMESPACE+':events',
            event_search_ms=[-20,200],consecutive_frames=3,unvoiced_plateaux=[[.2,.8],[2.2,2.8]],voiced_plateau=[1.2,1.8]),
        limits=dict(new_artifact_bytes=4*1024**3,minimum_free_bytes=15*1024**3))
    write_json(output/'plan.json',plan);return plan


def read_plan(folder):
    p=load(folder/'plan.json')
    require(p['schema']=='phenometrix-voicing-robustness-plan.v1' and p['status']=='frozen','Frozen plan required')
    require(p['recipe_sha256']==vi.canonical_sha(vi.bank_recipe()),'Bank recipe changed')
    check_sources(p['source_sha256'])
    return p


def load_feature(path):
    with np.load(path,allow_pickle=False) as data: return {k:data[k].copy() for k in data.files}


def prepare(plan_folder):
    plan=read_plan(plan_folder);out=plan_folder.parent/'prepare';out.mkdir(exist_ok=False)
    for name in ('features','pcm','labels','intermediate','schedules'): (out/name).mkdir()
    m=dict(schema='phenometrix-voicing-robustness-inputs.v1',status='running',started_at=now(),
        plan=file_record(plan_folder/'plan.json'),recipe_sha256=plan['recipe_sha256'],records=[],slots={'C':[],'R':[]},schedules={},
        train_only=True,development_or_external_inputs_used=False)
    write_json(out/'manifest.json',m)
    began=time.monotonic()
    try:
        source=load(TRAIN/'manifest.json');acq=load(TRAIN/'acquisition-manifest.json')
        require(digest(TRAIN/'manifest.json')==PINS['delayed-train-features/manifest.json'],'TRAIN identity changed')
        require(digest(TRAIN/'acquisition-manifest.json')==source['data_manifest_sha256'],'TRAIN acquisition changed')
        refs={r['recording_id']:r for r in acq['files'] if r['role']=='ref' and r['split']=='train'}
        mics={r['recording_id']:r for r in acq['files'] if r['role']=='mic' and r['split']=='train'}
        anchors={(r['source_recording_id'],r['condition']):r for r in source['recordings']}
        ids=sorted({r['source_recording_id'] for r in source['recordings']})
        require(len(ids)==240 and len({r['speaker'] for r in source['recordings']})==12,'TRAIN source membership differs')
        valid_masks=[]
        for source_index,identity in enumerate(ids):
            mic=DATA/mics[identity]['path'];ref=DATA/refs[identity]['path']
            require(digest(mic)==mics[identity]['sha256'] and digest(ref)==refs[identity]['sha256'],'TRAIN source bytes changed')
            clean=read_mic(mic);reference=np.loadtxt(ref);records=[]
            for slot in range(8):
                if slot<4:
                    anchor=anchors[identity,vi.ANCHORS[slot]];fp=TRAIN/'features'/(anchor['recording_id']+'.npz')
                    require(digest(fp)==anchor['feature_sha256'],'Anchor feature bytes changed')
                    feature=load_feature(fp);receipt=dict(anchor=anchor['recording_id'],transformed_pcm_sha256=anchor['transformed_pcm_sha256'])
                else:
                    pcm,receipt=vi.construct(clean,vi.draw(identity,slot-4))
                    pcm_path=out/'pcm'/f'{identity}__slot{slot}.f32le';pcm.tofile(pcm_path)
                    feature,_,binding=extract_input(pcm,f'{identity}__slot{slot}',out/'intermediate')
                    require(binding['transformed_pcm_sha256']==receipt['pcm_sha256'],'PCM byte convention changed')
                    receipt.update(binding,pcm=file_record(pcm_path))
                    fp=out/'features'/f'{identity}__slot{slot}.npz';np.savez_compressed(fp,**feature)
                require(list(feature['feature_names'])==model_metadata(VoiceCandidateModel(ModelConfig(kind='framewise-mlp')))['feature_names'],'Feature order differs')
                voiced,valid,target=training_labels(feature,reference,measurement_delay_ms=20)
                lp=out/'labels'/f'{identity}__slot{slot}.npz';np.savez_compressed(lp,voiced=voiced,valid=valid,target=target)
                record=dict(id=f'{identity}__slot{slot}',source_id=identity,speaker=mics[identity]['speaker'],slot=slot,
                    feature=file_record(fp),reference=file_record(ref),microphone=file_record(mic),labels=file_record(lp),
                    n_frames=len(voiced),valid_frames=int(valid.sum()),covered_candidate_frames=int(((target>=0)&valid).sum()),construction=receipt)
                m['records'].append(record);records.append(len(m['records'])-1)
                if slot==0: base_valid=valid.copy();n_frames=len(voiced)
                require(len(voiced)==n_frames and np.array_equal(valid,base_valid),'Paired length/validity masks differ')
            for slot in range(8):
                m['slots']['C'].append(records[slot%4]);m['slots']['R'].append(records[slot]);valid_masks.append(base_valid)
            if source_index%20==0:
                write_json(out/'manifest.json',m)
                print(json.dumps(dict(phase='prepare',sources=source_index+1,total=240,elapsed_seconds=time.monotonic()-began)),flush=True)
                require(shutil.disk_usage(REPO).free>plan['limits']['minimum_free_bytes'],'Free disk floor reached')
        require(len(m['slots']['C'])==1920 and len(m['records'])==1920,'Matched slot inventory differs')
        for seed in vi.SEEDS:
            schedule=dict(schema='phenometrix-voicing-robustness-schedule.v1',seed=seed,epochs=[],slot_count=1920)
            for epoch in range(12):
                order=list(range(1920));random.Random(seed+epoch).shuffle(order);steps=[]
                for offset in range(0,len(order),64):
                    indices=order[offset:offset+64];length=max(len(valid_masks[i]) for i in indices)
                    for start in range(0,length,128):
                        end=min(length,start+128)
                        steps.append(dict(indices=indices,start=start,end=end,update=any(bool(valid_masks[i][start:end].any()) for i in indices)))
                schedule['epochs'].append(dict(epoch=epoch+1,order=order,steps=steps))
            schedule['expected_updates']=sum(s['update'] for e in schedule['epochs'] for s in e['steps'])
            sp=out/'schedules'/f'{seed}.json';write_json(sp,schedule);m['schedules'][str(seed)]=file_record(sp)
        check_sources(plan['source_sha256']);m.update(status='complete',finished_at=now(),elapsed_seconds=time.monotonic()-began)
        write_json(out/'manifest.json',m);return m
    except BaseException as error:
        m.update(status='failed',failure=repr(error),failed_at=now());write_json(out/'manifest.json',m);raise


def initial_model(seed):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    if torch.backends.mps.is_available():torch.mps.manual_seed(seed)
    model=VoiceCandidateModel(ModelConfig(kind='framewise-mlp'))
    require(digest(ORIGINAL/'weights.pt')==PINS['delayed-mlp-seed20261022/weights.pt'],'Original scaling provenance changed')
    original=torch.load(ORIGINAL/'weights.pt',map_location='cpu',weights_only=True)['state_dict']
    with torch.no_grad():
        for name in ('feature_mean','feature_scale','scaling_fitted'):getattr(model,name).copy_(original[name])
    require(bool(model.scaling_fitted),'Original scaling must be fitted')
    return model


def tensor_sha(state):
    import hashlib
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        a=v.detach().cpu().numpy();h.update(k.encode()+str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
    return h.hexdigest()


def fit(plan_folder):
    plan=read_plan(plan_folder);base=plan_folder.parent;prep=load(base/'prepare/manifest.json')
    require(prep['status']=='complete' and verify_file(prep['plan'])==plan_folder/'plan.json','Complete bound preparation required')
    require(torch.backends.mps.is_available(),'MPS required, refusing silent CPU substitution')
    torch.set_num_threads(1)
    fits=base/'fits';fits.mkdir(exist_ok=False)
    # Freeze final training/audit source independently from the earlier bank recipe.
    source=scientific_sources()
    for name in ('audit_voicing_robustness.py',):
        path=Path(__file__).with_name(name);require(path.is_file(),'Independent audit must exist before fits');source[rel(path)]=digest(path)
    snapshot(fits,source);write_json(fits/'source-freeze.json',dict(source_sha256=source,at=now(),prepare=file_record(base/'prepare/manifest.json')))
    records=[]
    for row in prep['records']:
        feature=load_feature(verify_file(row['feature']));labels=load_feature(verify_file(row['labels']))
        verify_file(row['reference']);verify_file(row['microphone'])
        records.append(dict(x=torch.from_numpy(feature['features']),mask=torch.from_numpy(feature['candidate_mask']),
            voiced=torch.from_numpy(labels['voiced']),valid=torch.from_numpy(labels['valid']),target=torch.from_numpy(labels['target'])))
    receipt=dict(schema='phenometrix-voicing-robustness-fits.v1',status='running',started_at=now(),fits={})
    write_json(fits/'manifest.json',receipt)
    try:
        for seed in vi.SEEDS:
            schedule=load(verify_file(prep['schedules'][str(seed)]))
            init=initial_model(seed);initial=tensor_sha(init.state_dict())
            for arm in vi.ARMS:
                name=f'{arm}-{seed}';out=fits/name;out.mkdir()
                model=initial_model(seed);require(tensor_sha(model.state_dict())==initial,'Paired initialization differs')
                torch.save({'config':asdict(model.config),'state_dict':model.state_dict()},out/'initial-state.pt')
                seqs=[records[i] for i in prep['slots'][arm]]
                m=dict(schema='phenometrix-local-voice-training-v2',study_schema='phenometrix-voicing-robustness-fit.v1',status='running',split='train',
                    started_at=now(),device='mps',torch=torch.__version__,kind='framewise-mlp',seed=seed,arm=arm,epochs=12,batch_size=64,tbptt_frames=128,
                    measurement_delay_ms=20,learning_rate=.001,gradient_clip_norm=1.,test_or_development_labels_used=False,
                    feature_run_manifest_sha256=digest(base/'prepare/manifest.json'),prepare=file_record(base/'prepare/manifest.json'),plan=file_record(plan_folder/'plan.json'),
                    schedule=prep['schedules'][str(seed)],initialization_sha256=initial,initial_checkpoint_sha256=digest(out/'initial-state.pt'),
                    scaling_provenance=file_record(ORIGINAL/'weights.pt'),scaling_refitted=False,source_sha256=source,
                    expected_updates=schedule['expected_updates'],actual_updates=0,epoch_updates=[],epoch_losses=[],
                    sequence_count=len(seqs),feature_frames=sum(len(s['x']) for s in seqs),valid_supervision_frames=sum(int(s['valid'].sum()) for s in seqs),
                    covered_candidate_frames=sum(int(((s['target']>=0)&s['valid']).sum()) for s in seqs),
                    conditions=list(vi.ANCHORS)+(['anchor-repeat'] if arm=='C' else ['continuous-slots-0-3']),
                    augmentation_provenance=dict(arm=arm,recipe_sha256=plan['recipe_sha256'],prepare_manifest_sha256=digest(base/'prepare/manifest.json')))
                write_json(out/'manifest.json',m);began=time.monotonic();model.to('mps');optimizer=torch.optim.Adam(model.parameters(),lr=.001)
                try:
                    for ep in schedule['epochs']:
                        losses=[];updates=0;previous=None;batch=None
                        for step in ep['steps']:
                            if step['indices']!=previous:
                                group=[seqs[i] for i in step['indices']];length=max(len(s['x']) for s in group)
                                batch={}
                                for key,pad in [('x',0.),('mask',False),('voiced',0.),('valid',False),('target',-1)]:
                                    shape=(len(group),length,*group[0][key].shape[1:]);buf=torch.full(shape,pad,dtype=group[0][key].dtype)
                                    for i,s in enumerate(group):buf[i,:len(s[key])]=s[key]
                                    batch[key]=buf.to('mps')
                                previous=step['indices']
                            start,end=step['start'],step['end'];include=batch['valid'][:,start:end]
                            require(bool(include.any())==step['update'],'Frozen skip inventory changed')
                            if not step['update']:continue
                            optimizer.zero_grad(set_to_none=True)
                            v,c,_=model(batch['x'][:,start:end],batch['mask'][:,start:end],torch.zeros((len(previous),end-start,32),device='mps'))
                            loss=supervised_loss(v.squeeze(-1),c,batch['voiced'][:,start:end],include,batch['target'][:,start:end])
                            require(bool(torch.isfinite(loss)),'Nonfinite training loss')
                            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                            updates+=1;m['actual_updates']+=1;losses.append(float(loss.detach().cpu()))
                        m['epoch_updates'].append(updates);m['epoch_losses'].append(dict(epoch=ep['epoch'],mean_chunk_loss=float(np.mean(losses)),elapsed_seconds=time.monotonic()-began))
                        write_json(out/'manifest.json',m);print(json.dumps(dict(fit=name,**m['epoch_losses'][-1],updates=updates)),flush=True)
                    torch.mps.synchronize();require(m['actual_updates']==m['expected_updates'],'Update count differs')
                    model.cpu().eval();torch.save(dict(config=asdict(model.config),state_dict=model.state_dict()),out/'weights.pt')
                    check_sources(source);m.update(status='complete',finished_at=now(),elapsed_seconds=time.monotonic()-began,weights_sha256=digest(out/'weights.pt'),model=model_metadata(model))
                    write_json(out/'manifest.json',m);receipt['fits'][name]=file_record(out/'manifest.json');write_json(fits/'manifest.json',receipt)
                except BaseException as error:
                    m.update(status='failed',failure=repr(error),failed_at=now());write_json(out/'manifest.json',m);raise
                finally:
                    del model,optimizer;torch.mps.empty_cache()
        receipt.update(status='complete',finished_at=now());write_json(fits/'manifest.json',receipt);return receipt
    except BaseException as error:
        receipt.update(status='failed',failure=repr(error),failed_at=now());write_json(fits/'manifest.json',receipt);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['inventory','freeze','prepare','fit']);p.add_argument('--output',type=Path);p.add_argument('--plan',type=Path)
    a=p.parse_args()
    if a.phase in ('inventory','freeze'):
        require(a.output is not None,'--output required');result=globals()[a.phase](a.output.resolve())
    else:
        require(a.plan is not None,'--plan required');result=globals()[a.phase](a.plan.resolve())
    print(json.dumps(dict(phase=a.phase,status=result['status']),allow_nan=False))

if __name__=='__main__': main()
