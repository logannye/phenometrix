"""Actual specs are metadata only; refusal tests generate no PCM/features."""
import copy
import json
from pathlib import Path
import pytest
from phenometrix_prosody import augmentation_cases as cases


def fake_freeze(tmp_path,monkeypatch,*,challenge=False):
    monkeypatch.setattr(cases,'REPO',tmp_path);monkeypatch.setattr(cases,'RUNS',tmp_path)
    monkeypatch.setattr(cases,'source_hashes',lambda:{'toy.py':'a'*64})
    run=tmp_path/'run';run.mkdir()
    value={'status':'checkpoints_frozen' if challenge else 'sources_frozen','source_sha256':{'toy.py':'a'*64},
           'training_spec_sha256':cases.training_spec_sha256(),'challenge_spec_sha256':cases.challenge_spec_sha256(),
           'arms':list(cases.ARMS),'seeds':list(cases.SEEDS)}
    if challenge:
        (run/'checkpoints').mkdir();value['fits']={}
        for name in cases._expected_fits():
            p=run/'checkpoints'/(name+'.pt');p.write_bytes(b'toy checkpoint only')
            value['fits'][name]={'weights_sha256':cases.previous._sha(p)}
    path=run/'freeze.json';path.write_text(json.dumps(value))
    return path,value


def test_exact96_pairs_differ_only_in_after_spectrum_and_identity():
    spec=cases.training_spec();a,b=[spec['records_by_arm'][arm] for arm in cases.TRAIN_ARMS]
    assert len(a)==len(b)==96 and len({r['pair_id'] for r in a})==96
    assert spec['common_eligible_rows_per_arm']==96*191==18336
    for left,right in zip(a,b):
        assert left['pair_id']==right['pair_id']
        assert left['after']==left['before']==[1,.5,.25] and right['after']==[0,1,.5]
        assert {k:v for k,v in left.items() if k not in {'id','arm','after'}}=={k:v for k,v in right.items() if k not in {'id','arm','after'}}
    assert {r['gain'] for r in a}=={.04,.08}
    assert {tuple(r['phase_radians']) for r in a}=={(.11,.73,-.29),(-.61,1.07,.41)}
    assert {r['frequency_hz'] for r in a}=={f+.25 for f in (70,90,110,130,150,170,190,210,230,250,270,290,310,320,330,340,350,360,370,390,430,490,570,650)}


def test_fresh12_are_off_training_grid_and_consumed_fixed_carriers():
    spec=cases.challenge_spec();records=spec['records']
    assert len(records)==12 and len({r['id'] for r in records})==12
    assert {r['frequency_hz'] for r in records}=={93.7,193.9,331.7,366.1,517.3,641.9}
    assert {tuple(r['phase_radians']) for r in records}=={(.43,-1.21,.17),(1.67,.31,-.93)}
    consumed={50,100,150,200,300,700,87.3,137.9,223.7,337.1,467.3,613.7,79.7,151.3,319.7,354.5,503.9,659.3,
              66.7,85.9,121.3,177.7,247.9,353.3,479.7,623.1,83,167,217,389,321,573,149,443,89,179,227,431,337,619,137,509,127,263,421,593}
    assert not consumed&set(cases.TRAIN_FREQUENCIES)
    assert not consumed&set(cases.CHALLENGE_FREQUENCIES)
    assert not set(cases.TRAIN_FREQUENCIES)&set(cases.CHALLENGE_FREQUENCIES)
    assert not set(cases.TRAIN_PHASES)&set(cases.CHALLENGE_PHASES)
    assert all(r['gain']==.06 and r['after']==[0,1,.5] and r['change_sample']==39840 for r in records)
    assert all(r['id']==f'augmentation-balance-f{i//2}-p{i%2}' for i,r in enumerate(records))


def test_spec_hashes_commit_pair_physics_and_evaluation_before_generation():
    train=cases.training_spec();train['records_by_arm']['balance-aug'][0]['after'][2]=.25
    assert cases.training_spec_sha256()!=cases.hashlib.sha256(cases.previous._canonical(train)).hexdigest()
    assert cases.training_spec_sha256()!=cases.challenge_spec_sha256()
    assert cases.source_hashes().items()>=cases.previous.challenge_source_hashes().items()
    assert str(Path(cases.__file__).relative_to(cases.REPO)) in cases.source_hashes()


@pytest.mark.parametrize('challenge',[False,True])
def test_missing_freeze_refuses_before_creating_output(tmp_path,monkeypatch,challenge):
    monkeypatch.setattr(cases,'RUNS',tmp_path)
    generator=cases.generate_challenge if challenge else cases.generate_training
    with pytest.raises(ValueError,match='freeze is required'):generator(tmp_path/'output',tmp_path/'absent','0'*64)
    assert not (tmp_path/'output').exists()


def test_source_freeze_does_not_authorize_evaluation(tmp_path,monkeypatch):
    path,value=fake_freeze(tmp_path,monkeypatch)
    cases._validate_freeze(path,cases.previous._sha(path),challenge=False)
    with pytest.raises(ValueError,match='phase is incomplete'):
        cases.generate_challenge(tmp_path/'output',path,cases.previous._sha(path))
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('field,value,match',[
    ('status','fitting','phase'),('training_spec_sha256','0'*64,'specification'),
    ('challenge_spec_sha256','0'*64,'specification'),('source_sha256',{},'source closure'),
    ('arms',['natural-only'],'schedule'),('seeds',[20261022],'schedule'),
])
def test_invalid_training_freeze_blocks_all_pcm(tmp_path,monkeypatch,field,value,match):
    path,frozen=fake_freeze(tmp_path,monkeypatch);frozen[field]=value;path.write_text(json.dumps(frozen))
    with pytest.raises(ValueError,match=match):cases.generate_training(tmp_path/'output',path,cases.previous._sha(path))
    assert not (tmp_path/'output').exists()


def test_exact_nine_checkpoint_gate_and_missing_weight(tmp_path,monkeypatch):
    path,frozen=fake_freeze(tmp_path,monkeypatch,challenge=True)
    cases._validate_freeze(path,cases.previous._sha(path),challenge=True)
    assert set(frozen['fits'])=={f'{a}__seed{s}' for a in ('natural-only','stationary-aug','balance-aug') for s in (20261022,20261023,20261024)}
    name=next(iter(frozen['fits']));(path.parent/'checkpoints'/(name+'.pt')).unlink()
    with pytest.raises(ValueError,match='Checkpoint changed or missing'):
        cases.generate_challenge(tmp_path/'output',path,cases.previous._sha(path))
    assert not (tmp_path/'output').exists()


def test_incomplete_checkpoint_inventory_and_wrong_filehash_rejected(tmp_path,monkeypatch):
    path,frozen=fake_freeze(tmp_path,monkeypatch,challenge=True)
    with pytest.raises(ValueError,match='identity'):cases._validate_freeze(path,'0'*64,challenge=True)
    frozen['fits'].pop(next(iter(frozen['fits'])));path.write_text(json.dumps(frozen))
    with pytest.raises(ValueError,match='nine final'):cases._validate_freeze(path,cases.previous._sha(path),challenge=True)


def test_pair_scenarios_keep_exact_integer_support():
    record=cases.training_spec()['records_by_arm']['stationary-aug'][0]
    scenario=cases._scenario(record)
    assert scenario['pair_id']==record['pair_id'] and scenario['arm']=='stationary-aug'
    assert scenario['initialHz']==scenario['finalHz']==70.25
    assert [cases.challenge_region_at(scenario,e) for e in (39840,39841,41759,41760)]==[
        'before-observation-change','observation-boundary','observation-boundary','after-observation-change']
