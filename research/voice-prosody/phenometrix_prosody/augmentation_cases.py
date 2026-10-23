"""Paired TRAIN augmentation and prospectively frozen evaluation inputs.

The stationary and balance arms differ only in their post-change coefficients.
Training PCM requires an exact source/specification freeze; evaluation PCM also
requires all nine final checkpoints. No fitting, natural references, model output
or outcome-based parameter selection occurs here. Historical helpers stay exact.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from . import harmonic_challenge as previous
from .evaluate import validate_feature

REPO, ROOT, RUNS = previous.REPO, previous.ROOT, previous.RUNS
ARMS = ('natural-only', 'stationary-aug', 'balance-aug')
TRAIN_ARMS = ('stationary-aug', 'balance-aug')
SEEDS = (20261022, 20261023, 20261024)
TRAIN_FREQUENCIES = tuple(f+.25 for f in
    (70,90,110,130,150,170,190,210,230,250,270,290,310,320,330,340,350,360,370,390,430,490,570,650))
TRAIN_PHASES = ((.11,.73,-.29),(-.61,1.07,.41))
TRAIN_GAINS = (.04,.08)
CHALLENGE_FREQUENCIES = (93.7,193.9,331.7,366.1,517.3,641.9)
CHALLENGE_PHASES = ((.43,-1.21,.17),(1.67,.31,-.93))
TRAIN_MAX_BYTES = 128*1024*1024
CHALLENGE_MAX_BYTES = 40*1024*1024
TRAIN_SCHEMA = 'phenometrix-paired-harmonic-augmentation-train-spec-v1'
CHALLENGE_SCHEMA = 'phenometrix-paired-harmonic-augmentation-challenge-spec-v1'
CACHE_SCHEMA = 'phenometrix-paired-harmonic-augmentation-cache-v1'


def _record(identity,frequency,phases,gain,after,*,pair_id=None,arm=None):
    result={'id':identity,'frequency_hz':frequency,'phase_radians':list(phases),'gain':gain,
            'sample_rate_hz':48000,'sample_count':96000,'change_sample':39840,
            'before':[1.,.5,.25],'after':list(after)}
    if pair_id is not None: result['pair_id']=pair_id
    if arm is not None: result['arm']=arm
    return result


def _shared_spec():
    return {'sample_rate_hz':48000,'sample_count':96000,'duration_seconds_per_sequence':2.,
        'feature_version':'v2','measurement_delay_ms':20,'acquisition_end_grid':'2208+480*k samples',
        'reference_relation':'exclusive_support_end-960','common_mask':'reference>=3072 and reference+1536<=sample_count',
        'support_samples':1920,'hop_samples':480,'common_eligible_frames_per_sequence':191,
        'post_region':'exclusive_support_end-1920>=39840','post_frames_per_sequence':112,
        'noise_or_normalization':'none','harmonic_identifiability':'nonzero harmonics2and3 preserve shortest period1/F0',
        'disjointness':'TRAIN and fresh fixed carriers, endpoints, phases and complete configurations differ from consumed evaluation; continuous historical glides cross frequency ranges',
        'claim_boundary':'failure-motivated noiseless generator family; not external speech or broad channel robustness'}


def training_spec():
    records={arm:[] for arm in TRAIN_ARMS}
    for i,frequency in enumerate(TRAIN_FREQUENCIES):
        for j,phases in enumerate(TRAIN_PHASES):
            for k,gain in enumerate(TRAIN_GAINS):
                pair=f'augmentation-pair-f{i}-p{j}-g{k}'
                for arm in TRAIN_ARMS:
                    after=(1.,.5,.25) if arm=='stationary-aug' else (0.,1.,.5)
                    records[arm].append(_record(f'augmentation-{arm}-f{i}-p{j}-g{k}',frequency,phases,gain,after,pair_id=pair,arm=arm))
    return {**_shared_spec(),'schema':TRAIN_SCHEMA,'split':'train','records_by_arm':records,
        'pair_count':96,'sequence_count_per_arm':96,'common_eligible_rows_per_arm':18336,
        'duration_seconds_per_arm':192,'pairing':'same ordered pair_id, F0, phase, gain, before spectrum, change time and acquisition grid; only after spectrum differs',
        'stationary_control':'after equals before; there is no repeated after-mixture factor',
        'target_policy':'analytic constant F0 and voiced=true on common-eligible rows; uncovered candidates retain zero pitch loss and are counted',
        'access':'no TRAIN PCM or features before exact source/specification freeze',
        'maximum_output_bytes':TRAIN_MAX_BYTES}


def challenge_spec():
    records=[_record(f'augmentation-balance-f{i}-p{j}',frequency,phases,.06,(0.,1.,.5))
             for i,frequency in enumerate(CHALLENGE_FREQUENCIES) for j,phases in enumerate(CHALLENGE_PHASES)]
    return {**_shared_spec(),'schema':CHALLENGE_SCHEMA,'split':'prospective-challenge','records':records,
        'sequence_count':12,'duration_seconds_total':24,'predesignated_candidate':'balance-aug',
        'prospective_gate':'each predesignated-candidate seed: >=95% success within50c in EACH fully post-change case; no frame drops',
        'retained_gates':'all inherited natural, original analytic and consumed challenge requirements remain unchanged',
        'reporting':'whole/pre/boundary/post outcomes, raw/grouped candidate coverage and voiced/unavailable counts',
        'access':'no evaluation PCM or features until complete source-bound nine-checkpoint freeze',
        'maximum_output_bytes':CHALLENGE_MAX_BYTES}


def training_spec_sha256():return hashlib.sha256(previous._canonical(training_spec())).hexdigest()
def challenge_spec_sha256():return hashlib.sha256(previous._canonical(challenge_spec())).hexdigest()


def source_hashes():
    return {**previous.challenge_source_hashes(),str(Path(__file__).relative_to(REPO)):previous._sha(Path(__file__))}


def _expected_fits():return {f'{arm}__seed{seed}' for arm in ARMS for seed in SEEDS}


def _validate_freeze(path,expected_sha,*,challenge):
    path=previous._inside(RUNS,Path(path).resolve())
    previous._require(path.is_file(),'A complete declared freeze is required before PCM generation')
    previous._require(previous._sha(path)==expected_sha,'Freeze identity changed')
    frozen=json.loads(path.read_text());status='checkpoints_frozen' if challenge else 'sources_frozen'
    previous._require(frozen.get('status')==status,'Required freeze phase is incomplete')
    previous._require(frozen.get('training_spec_sha256')==training_spec_sha256()
                      and frozen.get('challenge_spec_sha256')==challenge_spec_sha256(),'Predeclared augmentation specification differs')
    previous._require(frozen.get('arms')==list(ARMS) and frozen.get('seeds')==list(SEEDS),'Declared arm/seed schedule differs')
    sources=source_hashes()
    previous._require(all(frozen.get('source_sha256',{}).get(p)==h for p,h in sources.items()),
                      'Freeze lacks matching augmentation/replay source closure')
    hashes={str(path):expected_sha,**{str(REPO/p):h for p,h in sources.items()}}
    if challenge:
        previous._require(set(frozen.get('fits',{}))==_expected_fits(),'Exactly the declared nine final checkpoints are required')
        for name,fit in frozen['fits'].items():
            path_=previous._inside(path.parent,Path('checkpoints')/(name+'.pt'))
            previous._require(path_.is_file() and previous._sha(path_)==fit.get('weights_sha256'),'Checkpoint changed or missing')
            hashes[str(path_)]=fit['weights_sha256']
    return frozen,hashes


def _scenario(record):
    scenario=previous._scenario(record)
    if 'pair_id' in record:scenario.update(pair_id=record['pair_id'],arm=record['arm'])
    return scenario


def challenge_region_at(scenario,end_sample):return previous.challenge_region_at(scenario,end_sample)


def _records(spec,challenge):
    if challenge:return [('challenge',record) for record in spec['records']]
    return [(arm,record) for arm in TRAIN_ARMS for record in spec['records_by_arm'][arm]]


def _generate(output,freeze_path,expected_sha,*,challenge):
    freeze_path=Path(freeze_path).resolve();_validate_freeze(freeze_path,expected_sha,challenge=challenge)
    output=previous._inside(RUNS,Path(output).resolve())
    previous._require(not output.exists(),'Generation refuses to overwrite prior evidence')
    previous._require(output.parent.exists() and shutil.disk_usage(output.parent).free>=1024*1024*1024,
                      'At least 1 GiB free required for bounded generation')
    spec=challenge_spec() if challenge else training_spec();limit=spec['maximum_output_bytes'];sources=source_hashes()
    output.mkdir()
    manifest={'schema':CACHE_SCHEMA,'split':spec['split'],'status':'generating','started_at':previous._now(),
        'spec_sha256':challenge_spec_sha256() if challenge else training_spec_sha256(),
        'training_spec_sha256':training_spec_sha256(),'challenge_spec_sha256':challenge_spec_sha256(),
        'freeze_kind':'checkpoint' if challenge else 'source','freeze_path':str(freeze_path.relative_to(REPO)),
        'freeze_sha256':expected_sha,'source_sha256':sources,'records':[],
        'reference_content_read':False,'feature_input':'only finite float32 PCM and sample rate enter unchanged TypeScript replay',
        'maximum_output_bytes':limit}
    prefix='checkpoint' if challenge else 'source'
    manifest[prefix+'_freeze_path']=manifest['freeze_path']
    manifest[prefix+'_freeze_sha256']=expected_sha
    previous._save(output/'manifest.json',manifest)
    try:
        for folder in ('pcm','features','intermediate','source-snapshot'):(output/folder).mkdir()
        previous._save(output/'spec.json',spec)
        for name,h in sources.items():
            destination=output/'source-snapshot'/name;destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes((REPO/name).read_bytes())
            previous._require(previous._sha(destination)==h,'Source snapshot changed')
        records=_records(spec,challenge)
        for arm,record in records:
            pcm=output/'pcm'/(record['id']+'.f32le');full=output/'intermediate'/(record['id']+'.json')
            previous._synthesize_pcm(record).tofile(pcm)
            feature=previous._extract_feature(pcm,full);timing=validate_feature(feature,{'sample_count':96000})
            previous._require(int(timing.common_eligible.sum())==191,'Fixed eligible row count changed')
            path=output/'features'/(record['id']+'.npz');np.savez_compressed(path,**feature)
            exported=previous._sha(full);full.unlink()
            manifest['records'].append({'id':record['id'],'arm':arm,'pair_id':record.get('pair_id'),
                'parameters_sha256':hashlib.sha256(previous._canonical(record)).hexdigest(),
                'pcm_path':str(pcm.relative_to(output)),'pcm_sha256':previous._sha(pcm),
                'feature_path':str(path.relative_to(output)),'feature_sha256':previous._sha(path),
                'typescript_export_sha256':exported,'feature_frames':len(feature['features']),'common_eligible_frames':191})
            previous._require(sum(p.stat().st_size for p in output.rglob('*') if p.is_file())<=limit,'Generation artifact cap exceeded')
            previous._save(output/'manifest.json',manifest)
            completed=len(manifest['records'])
            if completed % 16 == 0 or completed == len(records):
                print(json.dumps({'phase':'generating_augmentation_cases','split':spec['split'],
                                  'completed_records':completed,'total_records':len(records)}),flush=True)
        (output/'intermediate').rmdir();_validate_freeze(freeze_path,expected_sha,challenge=challenge)
        previous._require(source_hashes()==sources,'Source changed during generation')
        manifest.update(status='complete',completed_at=previous._now());previous._save(output/'manifest.json',manifest)
    except BaseException as error:
        manifest.update(status='failed',failed_at=previous._now(),failure=repr(error));previous._save(output/'manifest.json',manifest);raise
    return _load(output,freeze_path,expected_sha,challenge=challenge)


def _load(output,freeze_path,expected_sha,*,challenge):
    output=previous._inside(RUNS,Path(output).resolve());freeze_path=Path(freeze_path).resolve()
    _,hashes=_validate_freeze(freeze_path,expected_sha,challenge=challenge)
    manifest_path=output/'manifest.json';manifest=json.loads(manifest_path.read_text())
    spec=challenge_spec() if challenge else training_spec();sources=source_hashes();records=_records(spec,challenge)
    previous._require(json.loads((output/'spec.json').read_text())==spec,'Cached specification changed')
    previous._require(manifest.get('schema')==CACHE_SCHEMA and manifest.get('status')=='complete'
        and manifest.get('split')==spec['split'] and manifest.get('freeze_kind')==('checkpoint' if challenge else 'source')
        and manifest.get('freeze_sha256')==expected_sha and manifest.get('freeze_path')==str(freeze_path.relative_to(REPO))
        and manifest.get(('checkpoint' if challenge else 'source')+'_freeze_sha256')==expected_sha
        and manifest.get(('checkpoint' if challenge else 'source')+'_freeze_path')==str(freeze_path.relative_to(REPO))
        and manifest.get('source_sha256')==sources and manifest.get('training_spec_sha256')==training_spec_sha256()
        and manifest.get('challenge_spec_sha256')==challenge_spec_sha256()
        and manifest.get('spec_sha256')==(challenge_spec_sha256() if challenge else training_spec_sha256()),'Cache provenance differs')
    previous._require(len(manifest['records'])==len(records)
                      and [r['id'] for r in manifest['records']]==[r['id'] for _,r in records],'Cache inventory differs')
    for name,h in sources.items():
        snapshot=previous._inside(output/'source-snapshot',name);previous._require(previous._sha(snapshot)==h,'Source snapshot changed')
        hashes[str(snapshot)]=h
    rows=[] if challenge else {arm:[] for arm in TRAIN_ARMS}
    for (arm,record),saved in zip(records,manifest['records']):
        previous._require(saved['arm']==arm and saved['pair_id']==record.get('pair_id')
                          and saved['parameters_sha256']==hashlib.sha256(previous._canonical(record)).hexdigest(),'Cached pair or parameter identity differs')
        for path_key,hash_key in (('pcm_path','pcm_sha256'),('feature_path','feature_sha256')):
            p=previous._inside(output,saved[path_key]);previous._require(previous._sha(p)==saved[hash_key],'Cached acquisition changed');hashes[str(p)]=saved[hash_key]
        with np.load(previous._inside(output,saved['feature_path']),allow_pickle=False) as z:feature={k:z[k].copy() for k in z.files}
        timing=validate_feature(feature,{'sample_count':96000})
        previous._require(len(feature['features'])==saved['feature_frames'] and int(timing.common_eligible.sum())==saved['common_eligible_frames']==191,'Cached frame extent changed')
        row={'id':record['id'],'scenario':_scenario(record),'timing':timing,'feature':feature}
        if challenge:rows.append(row)
        else:row['pair_id']=record['pair_id'];rows[arm].append(row)
    if not challenge:
        for left,right in zip(rows[TRAIN_ARMS[0]],rows[TRAIN_ARMS[1]]):
            previous._require(left['pair_id']==right['pair_id'] and np.array_equal(left['timing'].reference_sample,right['timing'].reference_sample)
                              and np.array_equal(left['timing'].common_eligible,right['timing'].common_eligible),'Paired acquisition grid differs')
        previous._require(all(sum(int(r['timing'].common_eligible.sum()) for r in rows[arm])==18336 for arm in TRAIN_ARMS),'Paired eligible row pool differs')
    hashes[str(manifest_path)]=previous._sha(manifest_path);hashes[str(output/'spec.json')]=previous._sha(output/'spec.json')
    return rows,hashes


def generate_training(output,source_freeze_path,expected_sha):return _generate(output,source_freeze_path,expected_sha,challenge=False)
def load_training(output,source_freeze_path,expected_sha):return _load(output,source_freeze_path,expected_sha,challenge=False)
def generate_challenge(output,checkpoint_freeze_path,expected_sha):return _generate(output,checkpoint_freeze_path,expected_sha,challenge=True)
def load_challenge(output,checkpoint_freeze_path,expected_sha):return _load(output,checkpoint_freeze_path,expected_sha,challenge=True)
