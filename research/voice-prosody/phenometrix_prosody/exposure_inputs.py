"""Exact saved TRAIN-bank adapter and checkpoint-gated reserved input acquisition.

No historical source globals are changed. The new1536 TRAIN bank is read only;
only the twelve already-declared reserved configurations may be synthesized,
and only after a complete new source/checkpoint freeze passes validation.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
from . import frequency_diversity as bank
from . import evidence_cases as cases
from . import candidate_coverage_experiment as ledger

REPO,ROOT,RUNS=bank.REPO,bank.ROOT,bank.RUNS
TRAIN=RUNS/'frequency-diversity-v1'
PINS={
 'manifest.json':'8349835c4bac48332f629219063470ece8d84958e520790b300ab083cc9eb8a7',
 'source-freeze.json':'f1183405fe8a556d9293688037107e6099db480fd9e6ce496cae133912bcc918',
 'spec.json':'f25019b3b9deb6d699f440fd672cac5aabe6a6c70ccf6842c91fdb537a72fba4',
 'report.json':'3338e407866bc87dac2bb4f6dcdbf244a5852ab7c890aacef37c1cee36d4522c',
 'frame-table.npz':'201a2b648dcd0714ea0c73c50648ab881be387ff000f50492fe309293f26d04c',
 'independent-audit.json':'3587af269f38689d2b04d1db04e5cb524efc8cd3533d91b88e3edb1b921f570e',
}
ARMS=cases.ARMS;SEEDS=cases.SEEDS
SCHEMA='phenometrix-exposure-reserved-input-cache.v1'
BATCH_COUNTS={'added_only':128,'counterexample':128,'broad':256}
require=bank.require;sha=bank.sha;save=ledger.save;canonical_sha=bank.canonical_sha


def load_saved_training(manifest, guard):
    """Validate exact completed/audited new1536 bank; never acquire TRAIN again."""
    paths={name:ledger.bind(manifest,TRAIN/name,h) for name,h in PINS.items()}
    value=json.loads(paths['manifest.json'].read_text());freeze=json.loads(paths['source-freeze.json'].read_text())
    audit=json.loads(paths['independent-audit.json'].read_text());spec=json.loads(paths['spec.json'].read_text())
    report=json.loads(paths['report.json'].read_text())
    require(value.get('status')=='complete' and value.get('schema')==bank.SCHEMA and value.get('support_pass') is True,
        'Exact completed new1536 support-pass bank required')
    require(value.get('fits')==value.get('schedules_created')==0 and value.get('generated_configurations')==1536,
        'Saved TRAIN bank phase differs')
    require(audit.get('pass_result') is True and audit.get('support_pass') is True
        and audit.get('study_manifest_sha256')==PINS['manifest.json'] and audit.get('report_sha256')==PINS['report.json']
        and audit.get('frame_table_sha256')==PINS['frame-table.npz'], 'Exact independent TRAIN audit differs')
    require(value['source_sha256']==freeze['source_sha256'] and len(freeze['source_sha256'])==71,
        'Historical71 source closure differs')
    for p,h in freeze['source_sha256'].items():
        ledger.bind(manifest,REPO/p,h);ledger.bind(manifest,TRAIN/'source-snapshot'/p,h)
    require(spec==bank.training_spec() and canonical_sha(spec)==value['training_spec_sha256'], 'Saved TRAIN specification differs')
    require([r['id'] for r in value['records']]==[r['id'] for r in spec['records']], 'Exact new1536 TRAIN order differs')
    guard();arrays=ledger.npz(paths['frame-table.npz']);bank.validate_table(arrays,spec['records'])
    require(bank.support_report(arrays,spec['records'])==report and report['feasibility']['pass_result'],
        'Saved TRAIN support reconstruction differs')
    for row in value['records']:
        guard()
        for prefix in ('pcm','original_feature','expanded_feature','provenance'):
            path=(REPO/row[prefix+'_path']).resolve();h=row[prefix+'_sha256']
            require(path.is_relative_to(TRAIN) and value['input_sha256'].get(row[prefix+'_path'])==h,
                'TRAIN record escaped exact saved artifact map')
            ledger.bind(manifest,path,h)
    return dict(manifest=value,arrays=arrays,records=spec['records'],feasibility=report['feasibility'],
        strata={name:arrays['stratum_'+name] for name in BATCH_COUNTS},training_spec_sha256=canonical_sha(spec),
        identity={name:dict(path=str(path.relative_to(REPO)),sha256=PINS[name]) for name,path in paths.items()})


def targeted_schedule(strata,case_index,records,seed,epoch):
    """Same case-first RNG/order, now scoped to actual new1536 metadata."""
    require(seed in SEEDS and type(epoch) is int and 0<=epoch<12,'Unknown fixed seed/epoch')
    require(bank.support_proof(strata,case_index,records)['pass_result'],'Insufficient new-bank TRAIN support')
    case_index=np.asarray(case_index);rng=np.random.default_rng(seed+2_000_000+epoch);pools={}
    for name in BATCH_COUNTS:
        indices=np.flatnonzero(strata[name]);represented=np.unique(case_index[indices])
        pools[name]=(represented,{int(c):indices[case_index[indices]==c] for c in represented})
    order=np.empty((33,512),np.int64);origin=np.empty((33,512),np.uint8)
    for batch in range(33):
        offset=0
        for label,(name,count) in enumerate(BATCH_COUNTS.items()):
            represented,per_case=pools[name];sampled=rng.choice(represented,size=count,replace=True)
            for j,case in enumerate(sampled):
                candidates=per_case[int(case)];order[batch,offset+j]=candidates[rng.integers(len(candidates))]
            origin[batch,offset:offset+count]=label;offset+=count
        shuffle=rng.permutation(512);order[batch]=order[batch,shuffle];origin[batch]=origin[batch,shuffle]
    return dict(indices=order,origin=origin)


def schedule_exposure(schedule,strata,table,case_count):
    order,origin=schedule['indices'],schedule['origin'];case_index=table['case_index']
    require(order.shape==origin.shape==(33,512) and order.dtype==np.int64 and origin.dtype==np.uint8,
        'Targeted schedule schema differs')
    require(np.all((order>=0)&(order<len(case_index))) and np.isin(origin,[0,1,2]).all(),'Targeted schedule indices differ')
    for label,(name,count) in enumerate(BATCH_COUNTS.items()):
        require(np.all((origin==label).sum(1)==count) and np.all(strata[name][order[origin==label]]),'Targeted stratum draw differs')
    flat=order.ravel();require(table['common_eligible'][flat].all(),'Noncommon TRAIN row sampled')
    return dict(selected_rows=int(len(flat)),unique_rows=int(len(np.unique(flat))),duplicate_draws=int(len(flat)-len(np.unique(flat))),
        requested_strata={name:int((origin==label).sum()) for label,name in enumerate(BATCH_COUNTS)},
        actual_strata={name:int(strata[name][flat].sum()) for name in BATCH_COUNTS},
        per_case_selected_rows=np.bincount(case_index[flat],minlength=case_count).tolist(),
        expanded_group_covered_rows=int(table['expanded_group_covered'][flat].sum()),
        expanded_group_uncovered_zero_rows=int((~table['expanded_group_covered'][flat]).sum()),
        per_region={str(r):dict(selected_rows=int((table['region'][flat]==r).sum()),
            covered_rows=int(((table['region'][flat]==r)&table['expanded_group_covered'][flat]).sum())) for r in np.unique(table['region'])})


def make_schedules(preparation):
    arrays=preparation['arrays'];strata=preparation['strata'];records=preparation['records'];out={};exposure={}
    require(preparation['feasibility']['pass_result'],'Failed support cannot create schedules')
    for seed in SEEDS:
        for epoch in range(12):
            k=f'seed{seed}__epoch{epoch}';s=targeted_schedule(strata,arrays['case_index'],records,seed,epoch)
            for field in ('indices','origin'):out[k+'__'+field]=s[field]
            exposure[k]=schedule_exposure(s,strata,arrays,len(records))
    return out,exposure



def required_source_names():
    p=TRAIN/'source-freeze.json';require(sha(p)==PINS['source-freeze.json'],'Historical source freeze differs')
    historical=set(json.loads(p.read_text())['source_sha256'])
    require(len(historical)==71,'Historical71 closure differs')
    from . import audit_exposure_study as independent
    return historical|set(independent.source_hashes())|{str((ROOT/q).relative_to(REPO)) for q in (
        'EXPOSURE_STUDY_PLAN.md','phenometrix_prosody/exposure_study.py','phenometrix_prosody/exposure_inputs.py',
        'tests/test_exposure_study.py','tests/test_exposure_inputs.py','phenometrix_prosody/audit_exposure_study.py',
        'tests/test_audit_exposure_study.py','phenometrix_prosody/audit_evidence_scores.py','phenometrix_prosody/evaluate_curriculum.py')}


def validate_checkpoint_freeze(path,expected_sha,*,enforce_deadline=True):
    path=Path(path).resolve();require(path.is_relative_to(RUNS) and sha(path)==expected_sha,'New checkpoint freeze identity differs')
    frozen=json.loads(path.read_text())
    require(frozen.get('schema')=='phenometrix-exposure-checkpoint-freeze.v1' and frozen.get('status')=='checkpoints_frozen',
        'Complete new checkpoint freeze required')
    require(frozen.get('arms')==list(ARMS) and frozen.get('seeds')==list(SEEDS) and frozen.get('maximum_fits')==12,
        'Twelve declared arms/seeds required')
    require(frozen.get('training_bank_manifest_sha256')==PINS['manifest.json']
        and frozen.get('challenge_spec_sha256')==cases.challenge_spec_sha256(),'Frozen TRAIN/fresh identity differs')
    initial_path=(REPO/frozen.get('source_freeze_path','')).resolve()
    require(initial_path==path.parent/'source-freeze.json' and sha(initial_path)==frozen.get('source_freeze_sha256'),
        'Linked new initial source freeze differs')
    initial=json.loads(initial_path.read_text())
    require(initial.get('schema')=='phenometrix-exposure-source-freeze.v1' and initial.get('status')=='sources_frozen'
        and initial.get('fits')=={} and initial.get('source_sha256')==frozen.get('source_sha256')
        and initial.get('deadline_utc')==frozen.get('deadline_utc')
        and initial.get('training_bank_manifest_sha256')==PINS['manifest.json']
        and initial.get('challenge_spec_sha256')==cases.challenge_spec_sha256(),'Initial source/spec phase differs')
    if enforce_deadline:
        require((datetime.fromisoformat(frozen['deadline_utc'])-datetime.now(timezone.utc)).total_seconds()>0,
            'New scientific deadline expired')
    require(required_source_names()<=set(frozen.get('source_sha256',{})),'Incomplete bound scientific source closure')
    for p,h in frozen['source_sha256'].items():
        source=(REPO/p).resolve();require(source.is_relative_to(REPO) and p==str(source.relative_to(REPO)) and sha(source)==h,'Frozen scientific source changed')
        require(sha(path.parent/'source-snapshot'/p)==h,'Frozen source snapshot changed')
    require(set(frozen.get('checkpoints',{}))=={f'{a}__seed{s}' for a in ARMS for s in SEEDS},'Exactly12 final checkpoints required')
    for name,row in frozen['checkpoints'].items():
        checkpoint=(REPO/row['path']).resolve()
        require(checkpoint==path.parent/'checkpoints'/(name+'.pt') and row['path']==str(checkpoint.relative_to(REPO)) and sha(checkpoint)==row['sha256'],'Final checkpoint bytes changed')
    return frozen


def generate_challenge(output,freeze_path,expected_sha,guard):
    """One sequential twelve-case acquisition; exact unchanged reserved metadata."""
    frozen=validate_checkpoint_freeze(freeze_path,expected_sha);output=Path(output).resolve()
    require(output==Path(freeze_path).resolve().parent/'challenge' and not output.exists(),'Fresh challenge directory required; no retries')
    spec=cases.challenge_spec();output.mkdir()
    for name in ('pcm','features','intermediate'):(output/name).mkdir()
    manifest=dict(schema=SCHEMA,status='generating',split=spec['split'],started_at=cases.previous._now(),
        source_sha256=frozen['source_sha256'],spec_sha256=cases.challenge_spec_sha256(),challenge_spec_sha256=cases.challenge_spec_sha256(),
        training_bank_manifest_sha256=PINS['manifest.json'],freeze_kind='checkpoint',freeze_path=str(Path(freeze_path).resolve().relative_to(REPO)),
        freeze_sha256=expected_sha,checkpoint_freeze_path=str(Path(freeze_path).resolve().relative_to(REPO)),checkpoint_freeze_sha256=expected_sha,
        records=[],generated_configurations=0,reference_content_read=False,model_outputs_read=False,deadline_utc=frozen['deadline_utc'])
    save(output/'manifest.json',manifest);save(output/'spec.json',spec);items=[];hashes={}
    try:
        for index,record in enumerate(spec['records']):
            left=guard();validate_checkpoint_freeze(freeze_path,expected_sha)
            raw=output/'pcm'/(record['id']+'.f32le');full=output/'intermediate'/(record['id']+'.json')
            cases.previous._synthesize_pcm(record).tofile(raw)
            manifest['generated_configurations']=index+1;save(output/'manifest.json',manifest)
            feature=cases._extract_feature(raw,full,min(60.,guard()));timing=cases._timing(feature)
            fp=output/'features'/(record['id']+'.npz');np.savez_compressed(fp,**feature);export_sha=sha(full);full.unlink()
            row=dict(id=record['id'],case_index=index,parameters_sha256=canonical_sha(record),pcm_path=str(raw.relative_to(output)),pcm_sha256=sha(raw),
                feature_path=str(fp.relative_to(output)),feature_sha256=sha(fp),typescript_export_sha256=export_sha,feature_frames=196,common_eligible_frames=191)
            manifest['records'].append(row);save(output/'manifest.json',manifest)
            items.append(dict(id=record['id'],case_index=index,record=record,scenario=cases.previous._scenario(record),timing=timing,feature=feature))
            hashes[str(raw)]=row['pcm_sha256'];hashes[str(fp)]=row['feature_sha256'];guard()
        require(len(items)==12,'Exactly12 reserved configurations required');(output/'intermediate').rmdir()
        validate_checkpoint_freeze(freeze_path,expected_sha);manifest.update(status='complete',completed_at=cases.previous._now());save(output/'manifest.json',manifest)
    except BaseException as error:
        manifest.update(status='failed',failure=repr(error),failed_at=cases.previous._now())
        for name in ('stdout','stderr'):
            value=getattr(error,name,None)
            if value is not None:
                raw=value.encode() if isinstance(value,str) else bytes(value);p=output/('failure-'+name+'.log');p.write_bytes(raw[:65536])
                manifest[name+'_capture']=dict(path=p.name,captured_bytes=len(raw),retained_bytes=min(len(raw),65536),sha256=sha(p))
        save(output/'manifest.json',manifest);raise
    for name in ('manifest.json','spec.json'):hashes[str(output/name)]=sha(output/name)
    hashes[str(Path(freeze_path).resolve())]=expected_sha
    return items,hashes
