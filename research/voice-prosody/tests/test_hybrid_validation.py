import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
from phenometrix_prosody import hybrid_validation as hv
from phenometrix_prosody import evaluate as ev
from phenometrix_prosody.model import FEATURE_NAMES_V2


def inventory():
    rows=[]
    for speaker in hv.SPEAKERS:
        for j in range(hv.PER_SPEAKER+2):
            prompt=f"si{int(speaker[1:])*1000+j}"
            for role in ('mic','ref'):
                name=f'{role}_{speaker}_{prompt}.'+('wav' if role=='mic' else 'f0')
                rows.append(dict(recording_id=speaker+'_'+prompt,speaker=speaker,pair=int(speaker[1:]),prompt=prompt,
                    role=role,filename=name,url='https://example.invalid/'+name,split='sealed_test',primary_eligible=True))
    return dict(seed=hv.acquisition.SEED,split_assignment=hv.acquisition.split_assignment(),audit={'no_shared_prompt_gate':True},records=rows)


def test_metadata_selection_is_exact_and_original_only():
    c=hv.fixed_configuration(inventory())
    assert c['source_recording_count']==160 and c['condition_record_count']==640
    assert c['transform_split_namespace']=='sealed-test' and c['methods']==list(hv.METHODS)
    assert 'hybrid_augmented' not in c['methods'] and c==hv.fixed_configuration(inventory())
    bad=inventory();bad['audit']['no_shared_prompt_gate']=False
    with pytest.raises(ValueError):hv.fixed_configuration(bad)


def strict_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(hv,'REPO',tmp_path);monkeypatch.setattr(hv,'required_sources',lambda:['source.py'])
    paths={k:tmp_path/n for k,n in [('inventory','inventory.json'),('configuration','configuration.json'),('protocol','protocol.md'),
        ('preaccess_audit','audit.json'),('preaccess_acquisition_manifest','pre.json')]}
    for k,p in paths.items():ev.write_json(p,inventory() if k=='inventory' else {})
    (tmp_path/'source.py').write_text('# fixed\n')
    for folder in ('model','training'):(tmp_path/folder).mkdir()
    (tmp_path/'model/weights.pt').write_bytes(b'fixture')
    for n in ['model/manifest.json','training/manifest.json','training/acquisition-manifest.json']:ev.write_json(tmp_path/n,{})
    monkeypatch.setitem(hv.WEIGHT_HASHES,'original',ev.sha(tmp_path/'model/weights.pt'))
    ev.write_json(paths['configuration'],hv.fixed_configuration(inventory()))
    f=dict(schema='phenometrix-prosody-freeze-v1',experiment=hv.EXPERIMENT,sealed_test_authorized=True,source_revision='fixture',
        paths={**{k:p.name for k,p in paths.items()},'model_directory':'model','training_features':'training'},
        source_sha256={'source.py':ev.sha(tmp_path/'source.py')})
    for k,p in paths.items():f[k+'_sha256']=ev.sha(p)
    for field,n in [('model_sha256','model/weights.pt'),('model_manifest_sha256','model/manifest.json'),
        ('training_manifest_sha256','training/manifest.json'),('training_acquisition_manifest_sha256','training/acquisition-manifest.json')]:f[field]=ev.sha(tmp_path/n)
    path=tmp_path/'freeze.json';ev.write_json(path,f);return path,f


def test_actual_bound_bytes_and_authorization_verified(tmp_path,monkeypatch):
    path,f=strict_fixture(tmp_path,monkeypatch);hv.read_freeze(path)
    for name in ['protocol.md','inventory.json','source.py','model/weights.pt','configuration.json','audit.json']:
        p=tmp_path/name;old=p.read_bytes();p.write_bytes(old+b' ')
        with pytest.raises(ValueError):hv.read_freeze(path)
        p.write_bytes(old)
    f['sealed_test_authorized']=False;ev.write_json(path,f)
    with pytest.raises(ValueError,match='authorization'):hv.read_freeze(path)


def test_missing_source_or_escaping_path_refused(tmp_path,monkeypatch):
    path,f=strict_fixture(tmp_path,monkeypatch);f['source_sha256']={};ev.write_json(path,f)
    with pytest.raises(ValueError,match='omits'):hv.read_freeze(path)
    f['paths']['inventory']='../outside';ev.write_json(path,f)
    with pytest.raises(ValueError,match='escaped'):hv.read_freeze(path)


def small_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(hv,'PER_SPEAKER',1);c=hv.fixed_configuration(inventory())
    data=tmp_path/'data';data.mkdir();out=tmp_path/'output';freeze_path=tmp_path/'freeze.json';freeze_path.write_text('{}\n')
    protocol=tmp_path/'protocol.md';protocol.write_text('fixed\n');cp=tmp_path/'config.json';ev.write_json(cp,c)
    f=dict(inventory_sha256='a'*64,source_sha256={},configuration_sha256=ev.sha(cp),model_sha256='b'*64,model_manifest_sha256='c'*64)
    paths=dict(configuration=cp,protocol=protocol,training_features=tmp_path/'training',model_directory=tmp_path/'model')
    monkeypatch.setattr(hv,'read_freeze',lambda _:(copy.deepcopy(f),copy.deepcopy(c),paths))
    monkeypatch.setattr(ev,'read_cache_manifest',lambda _:({},{}))
    monkeypatch.setattr(ev,'load_frozen_model',lambda *_:(SimpleNamespace(kind='framewise-mlp'),{},{'model':{}}))
    mic=[];ref=[]
    for metadata in c['selected_inventory_records']:
        r={**metadata,'path':f"records/sealed_test/{metadata['speaker']}/{metadata['filename']}"}
        p=data/r['path'];p.parent.mkdir(parents=True,exist_ok=True)
        if r['role']=='mic':p.write_bytes(b'synthetic microphone');r['sha256']=ev.sha(p);mic.append(r)
        else:r['sha256']=None;ref.append(r)
    ev.write_json(data/'manifest.json',dict(inventory_sha256=f['inventory_sha256'],files=mic))
    monkeypatch.setattr(hv,'read_mic',lambda _:np.zeros(9600,np.float32))
    feature=dict(features=np.zeros((8,57),np.float32),candidate_hz=np.full((8,6),np.nan),candidate_mask=np.zeros((8,6),bool),
        reference_sample=2208+np.arange(8,dtype=np.int64)*480,feature_names=np.array(FEATURE_NAMES_V2),sample_rate_hz=np.array(48000),pcm_sample_count=np.array(9600))
    def extract(pcm,*_):
        t=ev.validate_feature(feature,{'sample_count':len(pcm)});rows=[dict(sample=int(x),state='unvoiced',f0_hz=None,correctness_probability=None) for x in t.reference_sample]
        return feature,rows,dict(transformed_pcm_sha256='d'*64,typescript_export_sha256='e'*64)
    monkeypatch.setattr(hv,'extract_input',extract)
    monkeypatch.setattr(hv,'yin_predictions',lambda pcm,ends:([ev.Prediction(int(e),'unvoiced',None) for e in ends],[dict(sample=int(e),raw_f0_hz=200.) for e in ends]))
    monkeypatch.setattr(ev,'model_predictions',lambda f,*_:[dict(sample=int(e-960),state='unavailable',f0_hz=None,voicing_probability=.9,candidate_index=None) for e in f['reference_sample']])
    return freeze_path,data,out,f,mic,ref


def add_refs(data,f,mic,ref):
    for r in ref:
        p=data/r['path'];np.savetxt(p,np.tile([200.,1.,0.,0.],(20,1)));r['sha256']=ev.sha(p)
    ev.write_json(data/'manifest.json',dict(inventory_sha256=f['inventory_sha256'],files=mic+ref))


def test_all_predictions_frozen_before_first_reference_read(tmp_path,monkeypatch):
    path,data,out,f,mic,ref=small_fixture(tmp_path,monkeypatch);real=np.loadtxt
    def read(p,*a,**k):
        frozen=json.loads((out/'prediction-freeze.json').read_text())
        assert frozen['status']=='predictions_frozen' and len(frozen['recordings'])==16
        assert frozen['references_acquired_or_read_before_prediction_freeze'] is False
        return real(p,*a,**k)
    monkeypatch.setattr(np,'loadtxt',read);hv.predict(path,data,out)
    assert not any((data/r['path']).exists() for r in ref)
    add_refs(data,f,mic,ref);result=hv.score(path,data,out)
    assert result['production_promotion'] is False and result['conditional_uncertainty']['draws']==2000
    with pytest.raises(FileExistsError):hv.score(path,data,out)


def test_early_reference_acquisition_refused_before_mic_read(tmp_path,monkeypatch):
    path,data,out,f,mic,ref=small_fixture(tmp_path,monkeypatch);add_refs(data,f,mic,ref)
    monkeypatch.setattr(hv,'read_mic',lambda _:pytest.fail('opened microphone before phase guard'))
    with pytest.raises(ValueError,match='References must not'):hv.predict(path,data,out)
    assert not out.exists()


def test_changed_prediction_refused_before_reference(tmp_path,monkeypatch):
    path,data,out,f,mic,ref=small_fixture(tmp_path,monkeypatch);hv.predict(path,data,out);add_refs(data,f,mic,ref)
    file=next((out/'predictions').iterdir());file.write_bytes(file.read_bytes()+b'changed')
    monkeypatch.setattr(np,'loadtxt',lambda *_:pytest.fail('reference opened before prediction integrity'))
    with pytest.raises(ValueError,match='predictions changed'):hv.score(path,data,out)


def test_ref_acquisition_cannot_bypass_prediction_freeze(tmp_path,monkeypatch):
    path,data,out,f,mic,ref=small_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(hv.acquisition,'acquire',lambda *_:pytest.fail('download before prediction freeze'))
    with pytest.raises(FileNotFoundError):hv.acquire_phase('acquire-ref',path,data,out,1000)


def test_prompt_bootstrap_preserves_known_paired_advantage(monkeypatch):
    monkeypatch.setattr(hv,'PER_SPEAKER',2);rows=[]
    for s in hv.SPEAKERS:
        for c in hv.CONDITIONS:
            for p in ['a','b']:
                rows.append(dict(speaker=s,pair=int(s[1:]),condition=c,prompt=p,metrics={m:dict(eligible_valid_frames=100,correct_frames=90 if m=='hybrid_original' else 80) for m in hv.METHODS}))
    a=hv.conditional_uncertainty(rows);assert a==hv.conditional_uncertainty(rows)
    for v in a['joint_error_advantage_intervals'].values():assert v['lower']==pytest.approx(.1) and v['upper']==pytest.approx(.1)
