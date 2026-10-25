"""Synthetic-only diagnostic input tests; one bounded real TS/YIN integration."""
import copy
import hashlib
import os
import numpy as np
import pytest
from phenometrix_prosody import voicing_robustness_diagnostics as vd
from phenometrix_prosody import voicing_robustness_inputs as vi
from phenometrix_prosody import evaluate as ev
from phenometrix_prosody.pilot import RESEARCH


def spec():
    return dict(seconds=3,harmonic_interval=[1,2],fundamentals=[90,180,360],phases=[0,.7],
        amplitudes=[.1,.05,.025],backgrounds=['silence','white_noise_sd_0.003'],seed_namespace=vi.NAMESPACE+':events')


def test_event_inventory_and_common_phase_equations():
    records=vd.event_specs(spec())
    assert len(records)==12 and len({r['id'] for r in records})==12
    assert len({r['seed'] for r in records})==12
    for row in records:
        assert row['seed']==int.from_bytes(hashlib.sha256(f"{vi.NAMESPACE}:events:{row['id']}".encode()).digest()[:8],'little')
        pcm,receipt=vd.construct_event(row)
        noise=np.random.default_rng(row['seed']).standard_normal(144000)*row['noise_sd']
        reference=noise.copy()
        for n in (0,1,12000,47999):
            expected=sum(amp*np.sin(2*np.pi*h*row['fundamental_hz']*n/48000+row['common_phase_radians'])
                for h,amp in enumerate([.1,.05,.025],1))
            np.testing.assert_allclose(pcm[48000+n], np.float32(noise[48000+n]+expected),atol=1e-7)
        np.testing.assert_array_equal(pcm[:48000],noise[:48000].astype(np.float32))
        np.testing.assert_array_equal(pcm[96000:],noise[96000:].astype(np.float32))
        assert receipt['pcm_sha256']==hashlib.sha256(pcm.tobytes()).hexdigest()
        np.testing.assert_array_equal(pcm,vd.construct_event(row)[0])


@pytest.mark.parametrize('change',[{'fundamentals':[91,180,360]},{'phases':[0,.8]}, {'seconds':4}, {'amplitudes':[.2,.05,.025]}, {'seed_namespace':'different'}])
def test_event_contract_cannot_drift(change):
    with pytest.raises(ValueError): vd.event_specs({**spec(),**change})


def synthetic_metadata(group):
    _,_,split,speakers,count=vd.GROUPS[group]
    rows=[];files=[]
    for speaker in speakers:
        for index in range(count):
            source=f'{speaker}_synthetic{index}'
            for condition in vi.ANCHORS:
                rows.append(dict(recording_id=source+'__'+condition,source_recording_id=source,
                    speaker=speaker,condition=condition,sample_count=48000,microphone_sha256='0'*64))
            files.append(dict(role='mic',recording_id=source,split=split,speaker=speaker,sha256='0'*64,
                primary_eligible=True,path=f'records/{split}/{speaker}/{source}.wav'))
            files.append(dict(role='ref',recording_id=source,split=split,speaker=speaker,sha256='2'*64,
                path=f'records/{split}/{speaker}/{source}.f0'))
    return dict(status='complete',recordings=rows),dict(files=files)


def test_exact_consumed_membership_from_metadata_only():
    for group,count in [('original_development',80),('formerly_reserved',160)]:
        manifest,acq=synthetic_metadata(group)
        result=vd.selected_sources(manifest,acq,group=group)
        assert len(result)==count
        assert [r['source_recording_id'] for r in result]==sorted(r['source_recording_id'] for r in result)
        assert all(r['microphone']['split']!='train' for r in result)


@pytest.mark.parametrize('corrupt',['count','speaker','duplicate','split','hash','missing','path'])
def test_rejects_selection_or_acquisition_corruption(corrupt):
    manifest,acq=synthetic_metadata('original_development')
    if corrupt=='count': manifest['recordings'].pop()
    elif corrupt=='speaker': manifest['recordings'][0]['speaker']='F02'
    elif corrupt=='duplicate': manifest['recordings'][1]=copy.deepcopy(manifest['recordings'][0])
    elif corrupt=='split': acq['files'][0]['split']='train'
    elif corrupt=='hash': acq['files'][0]['sha256']='1'*64
    elif corrupt=='missing': acq['files'].pop()
    elif corrupt=='path': acq['files'][0]['path']='../../outside.wav'
    with pytest.raises(ValueError):vd.selected_sources(manifest,acq,group='original_development')


def test_actual_ts_feature_and_yin_preparation_on_synthetic_pcm(tmp_path):
    for name in ('pcm','features','baselines','intermediate'):(tmp_path/name).mkdir()
    os.environ.setdefault('NUMBA_CACHE_DIR',str(RESEARCH/'runs/.numba-cache'))
    row=vd.event_specs(spec())[0]
    pcm,construction=vd.construct_event(row)
    artifacts=vd.prepare_pcm(pcm,identity='synthetic-integration',output=tmp_path)
    assert artifacts['pcm_sha256']==construction['pcm_sha256']
    assert artifacts['pcm_sha256']==hashlib.sha256((tmp_path/artifacts['pcm_path']).read_bytes()).hexdigest()
    with np.load(tmp_path/artifacts['feature_path'],allow_pickle=False) as features:
        assert features['features'].shape[1]==57
        endpoints=features['reference_sample']
        assert endpoints[0]==2208
        assert np.all(np.diff(endpoints)==480)
    base=ev.read_predictions(tmp_path/artifacts['baseline_path'])
    assert [r['sample'] for r in base['yin_evidence']]==endpoints.tolist()
    for rows in base['predictions'].values():
        assert [r['sample'] for r in rows]==(endpoints-960).tolist()
    assert all('raw_f0_hz' in r for r in base['yin_evidence'])
    assert not list((tmp_path/'intermediate').iterdir())
    with pytest.raises(ValueError,match='overwrite'):
        vd.prepare_pcm(pcm,identity='synthetic-integration',output=tmp_path)
