import hashlib
import json
from pathlib import Path
import wave

import numpy as np
import pytest
import torch

from phenometrix_prosody.comparator_cache import (TRANSFORM_SOURCE, frozen_worklist,
    reconstruct_pcm, run)
from phenometrix_prosody.features import CONDITIONS, transform
from phenometrix_prosody.pilot import REPO, digest, read_mic


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


@pytest.fixture
def cache(tmp_path):
    data, features = tmp_path/'data', tmp_path/'features'
    data.mkdir();features.mkdir()
    mic = data/'mic_F04_si7.wav'
    x=(.2*np.sin(2*np.pi*170*np.arange(4800)/48000)*32767).astype('<i2')
    with wave.open(str(mic),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(48000);f.writeframes(x.tobytes())
    source={'recording_id':'F04_si7','speaker':'F04','pair':4,'prompt':'si7','role':'mic',
            'split':'train','primary_eligible':True,'path':mic.name,'sha256':digest(mic)}
    # These paths do not exist. Any attempted sensor/reference content read fails.
    acquisition={'files':[source,{**source,'role':'ref','path':'DO-NOT-OPEN.ref'},
                           {**source,'role':'lar','path':'DO-NOT-OPEN.lar'}]}
    save(features/'acquisition-manifest.json',acquisition)
    save(features/'microphone-selection.json',[source])
    snapshot=features/'source-snapshot'/TRANSFORM_SOURCE
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    snapshot.write_bytes((REPO/TRANSFORM_SOURCE).read_bytes())
    pcm=read_mic(mic);records=[]
    for condition in CONDITIONS:
        changed,corruption=transform(pcm,condition,source['recording_id'],'train')
        records.append({'recording_id':source['recording_id']+'__'+condition,
            'source_recording_id':source['recording_id'],'speaker':'F04','pair':4,'prompt':'si7',
            'condition':condition,'sample_count':len(pcm),'microphone_sha256':source['sha256'],
            'transformed_pcm_sha256':hashlib.sha256(changed.astype('<f4').tobytes()).hexdigest(),
            'corruption':corruption})
    m={'schema':'phenometrix-input-only-feature-cache-v1','status':'complete','split':'train',
       'conditions':list(CONDITIONS),'measurement_delay_ms':20,'reference_or_laryngograph_content_read':False,
       'data_manifest_sha256':digest(features/'acquisition-manifest.json'),
       'identity':{'source_sha256':{TRANSFORM_SOURCE:digest(snapshot)}},'recordings':records}
    save(features/'manifest.json',m)
    return data,features,m,source


def test_frozen_selection_and_exact_corruption_reconstruction(cache):
    data,features,m,source=cache
    pinned,sha,work=frozen_worklist(features)
    assert sha==digest(features/'manifest.json')
    assert [r['condition'] for _,r in work]==list(CONDITIONS)
    for source,record in work:
        pcm=reconstruct_pcm(data,source,record,'train')
        assert hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest()==record['transformed_pcm_sha256']
    changed=dict(work[0][1],transformed_pcm_sha256='0'*64)
    with pytest.raises(ValueError,match='corruption differs'):
        reconstruct_pcm(data,source,changed,'train')
    with pytest.raises(ValueError,match='non-microphone'):
        reconstruct_pcm(data,dict(source,role='ref'),work[0][1],'train')


@pytest.mark.parametrize('field,value,match',[
    ('status','running','completed'),('split','sealed_test','sealed-test'),
    ('reference_or_laryngograph_content_read',True,'input-only'),
    ('measurement_delay_ms',0,'20 ms')])
def test_refuses_incomplete_sealed_or_changed_contract(cache,field,value,match):
    _,features,m,_=cache;m[field]=value;save(features/'manifest.json',m)
    with pytest.raises(ValueError,match=match):frozen_worklist(features)


def test_refuses_reordered_or_missing_conditions(cache):
    _,features,m,_=cache
    m['recordings'][0],m['recordings'][1]=m['recordings'][1],m['recordings'][0]
    save(features/'manifest.json',m)
    with pytest.raises(ValueError,match='order'):frozen_worklist(features)
    m['recordings'].pop();save(features/'manifest.json',m)
    with pytest.raises(ValueError,match='condition coverage'):frozen_worklist(features)


def test_runner_freezes_nominal_centered_predictions_without_outcomes(cache,tmp_path):
    data,features,_,_=cache;out=tmp_path/'predictions';calls=[]
    def backend(frames):
        calls.append(tuple(frames.shape))
        values=torch.full((len(frames),360),.02)
        values[:,110]=.9
        return values
    manifest=run(features,data,out,device='cpu',batch_size=3,_test_backend=backend)
    assert calls and manifest['status']=='complete'
    assert manifest['execution_backend']=='synthetic_test_fixture'
    assert manifest['scoring_performed'] is False
    assert manifest['reference_or_laryngograph_content_read'] is False
    assert manifest['declared_lookahead_ms']==32 and manifest['score_target_shift_ms']==0
    assert len(manifest['recordings'])==4
    freeze=json.loads((out/'prediction-freeze.json').read_text())
    assert digest(out/'prediction-freeze.json')==manifest['prediction_freeze_sha256']
    for r in manifest['recordings']:
        p=out/'predictions'/(r['recording_id']+'.json')
        assert digest(p)==freeze['prediction_sha256'][r['recording_id']]
        rows=json.loads(p.read_text())['predictions']
        assert [r['sample'] for r in rows]==list(range(768,4800,480))
        assert all(r['available_at_sample']==r['sample']+1536 for r in rows)
        assert all(r['declared_lookahead_ms']==32 for r in rows)
