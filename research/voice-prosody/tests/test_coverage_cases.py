"""Metadata and authorization only; never synthesize the prospective waveforms."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from phenometrix_prosody import coverage_cases as challenge


def fake_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(challenge, 'REPO', tmp_path)
    monkeypatch.setattr(challenge, 'RUNS', tmp_path/'runs')
    source=tmp_path/'toy.py';source.write_text('# unrelated test-only source\n')
    sources={'toy.py':challenge.previous._sha(source)}
    monkeypatch.setattr(challenge, 'source_hashes', lambda:sources.copy())
    run=tmp_path/'runs'/'prior';(run/'checkpoints').mkdir(parents=True)
    checkpoints={}
    for name in sorted(challenge._expected_checkpoints()):
        p=run/'checkpoints'/(name+'.pt');p.write_bytes(('test-only '+name).encode())
        checkpoints[name]={'path':str(p.relative_to(tmp_path)),'sha256':challenge.previous._sha(p)}
    value={'schema':'phenometrix-candidate-coverage-freeze.v1','status':'sources_checkpoints_frozen',
           'challenge_spec_sha256':challenge.challenge_spec_sha256(),'source_sha256':sources,'checkpoints':checkpoints}
    path=tmp_path/'runs'/'source-checkpoint-freeze.json';path.write_text(json.dumps(value))
    return path,value


@pytest.fixture(autouse=True)
def prohibit_prospective_pcm(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Prospective PCM/feature generation forbidden in this suite')
    monkeypatch.setattr(challenge.previous,'_synthesize_pcm',forbidden)
    monkeypatch.setattr(challenge.previous,'_extract_feature',forbidden)


def test_fixed_metadata_and_gate():
    spec=challenge.challenge_spec();records=spec['records']
    assert len(records)==12 and len({r['id'] for r in records})==12
    assert {r['frequency_hz'] for r in records}=={97.1,213.7,347.3,529.1,627.7,671.3}
    assert {tuple(r['phase_radians']) for r in records}=={(.29,-.67,1.31),(1.89,.47,-1.43)}
    assert all(r['id']==f'coverage-balance-f{i//2}-p{i%2}' for i,r in enumerate(records))
    assert all((r['sample_rate_hz'],r['sample_count'],r['change_sample'],r['gain'])==(48000,96000,39840,.055) for r in records)
    assert all(r['before']==[1,.5,.25] and r['after']==[0,1,.5] for r in records)
    assert spec['frozen_selector_arm']=='balance-aug' and '>=95%' in spec['prospective_gate']
    assert 'no frame drops' in spec['prospective_gate'] and 'all fifty' in spec['independence']
    assert 'not disjoint frequency support' in spec['novelty_boundary']
    assert challenge.challenge_spec_sha256()==hashlib.sha256(challenge.previous._canonical(spec)).hexdigest()


def test_carriers_avoid_all_prior_configured_endpoints_and_train_carriers():
    # These APIs construct metadata only, never PCM. Hybrid step endpoints were
    # easy to miss in earlier manual lists; reconstruct them from frozen source.
    from phenometrix_prosody import augmentation_cases as aug, subharmonic_challenge as sub
    from phenometrix_prosody import synthetic_curriculum as curriculum, hybrid_followup as hybrid
    plans=curriculum.make_plans()
    records=[*plans['train']['records'],*plans['challenge']['records'],*hybrid.fresh_records()]
    historical={float(r[key]) for r in records for key in ('initial_hz','final_hz') if r[key] is not None}
    historical.update(challenge.previous.FREQUENCIES);historical.update(sub.FREQUENCIES)
    historical.update(aug.TRAIN_FREQUENCIES);historical.update(aug.CHALLENGE_FREQUENCIES)
    # Fixed support-diagnostic and hybrid-parity carrier/endpoints, inspected in
    # source without calling waveform-producing analytic_sources().
    historical.update((49.99,50,50.01,100,150,175,200,225,250,300,500,699.99,700,700.01))
    assert 347.9 in historical and 601.7 in historical
    assert not set(challenge.FREQUENCIES)&historical
    old_phases=set(challenge.previous.PHASES)|set(sub.PHASES)|set(aug.TRAIN_PHASES)|set(aug.CHALLENGE_PHASES)
    assert not set(challenge.PHASES)&old_phases
    assert .055 not in aug.TRAIN_GAINS


def test_source_closure_includes_old_replay_and_new_helper():
    sources=challenge.source_hashes()
    assert sources.items()>=challenge.previous.challenge_source_hashes().items()
    assert str(Path(challenge.__file__).relative_to(challenge.REPO)) in sources
    assert 'research/voice-prosody/phenometrix_prosody/harmonic_challenge.py' in sources
    assert 'research/voice-prosody/replay.ts' in sources
    assert 'apps/capture-web/src/voice-pitch-features-v2.ts' in sources


def test_exact_existing_three_checkpoint_inventory(tmp_path,monkeypatch):
    path,frozen=fake_freeze(tmp_path,monkeypatch)
    assert set(frozen['checkpoints'])=={f'balance-aug__seed{s}' for s in (20261022,20261023,20261024)}
    _,hashes=challenge._validate_freeze(path,challenge.previous._sha(path))
    assert all(str(tmp_path/c['path']) in hashes for c in frozen['checkpoints'].values())


def test_missing_freeze_prevents_output(tmp_path,monkeypatch):
    monkeypatch.setattr(challenge,'RUNS',tmp_path)
    with pytest.raises(ValueError,match='freeze is required'):
        challenge.generate_challenge(tmp_path/'out',tmp_path/'missing.json','0'*64)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('field,value,match',[
    ('schema','other','schema'),('status','fitting','must freeze'),
    ('challenge_spec_sha256','0'*64,'specification'),('source_sha256',{},'source closure'),
    ('checkpoints',{},'three declared'),
])
def test_incomplete_freeze_refuses_output_before_pcm(tmp_path,monkeypatch,field,value,match):
    path,frozen=fake_freeze(tmp_path,monkeypatch);frozen[field]=value;path.write_text(json.dumps(frozen))
    with pytest.raises(ValueError,match=match):
        challenge.generate_challenge(tmp_path/'runs'/'out',path,challenge.previous._sha(path))
    assert not (tmp_path/'runs'/'out').exists()


def test_changed_freeze_digest_rejected(tmp_path,monkeypatch):
    path,_=fake_freeze(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='identity changed'):
        challenge._validate_freeze(path,'0'*64)


@pytest.mark.parametrize('remove',[False,True])
def test_missing_or_changed_existing_checkpoint(tmp_path,monkeypatch,remove):
    path,frozen=fake_freeze(tmp_path,monkeypatch)
    weight=tmp_path/next(iter(frozen['checkpoints'].values()))['path']
    if remove:weight.unlink()
    else:weight.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint changed or missing'):
        challenge._validate_freeze(path,challenge.previous._sha(path))


@pytest.mark.parametrize('kind',['absolute','escape','wrong_name'])
def test_checkpoint_path_contract(tmp_path,monkeypatch,kind):
    path,frozen=fake_freeze(tmp_path,monkeypatch);entry=next(iter(frozen['checkpoints'].values()))
    if kind=='absolute':entry['path']=str(tmp_path/entry['path']);match='repository-relative'
    elif kind=='escape':entry['path']='../escaped.pt';match='escaped'
    else:entry['path']='runs/prior/checkpoints/other.pt';match='declared existing'
    path.write_text(json.dumps(frozen))
    with pytest.raises(ValueError,match=match):challenge._validate_freeze(path,challenge.previous._sha(path))


def test_extra_checkpoint_rejected(tmp_path,monkeypatch):
    path,frozen=fake_freeze(tmp_path,monkeypatch)
    frozen['checkpoints']['stationary-aug__seed20261022']=next(iter(frozen['checkpoints'].values()))
    path.write_text(json.dumps(frozen))
    with pytest.raises(ValueError,match='three declared'):challenge._validate_freeze(path,challenge.previous._sha(path))


def test_refuse_overwrite_before_pcm(tmp_path,monkeypatch):
    path,_=fake_freeze(tmp_path,monkeypatch);out=tmp_path/'runs'/'out';out.mkdir();(out/'keep').write_text('prior evidence')
    with pytest.raises(ValueError,match='overwrite'):challenge.generate_challenge(out,path,challenge.previous._sha(path))
    assert (out/'keep').read_text()=='prior evidence'


def test_integer_support_counts_unchanged_without_generating_pcm():
    scenario=challenge.previous._scenario(challenge.challenge_spec()['records'][0])
    assert scenario['initialHz']==scenario['finalHz']==97.1
    assert challenge.challenge_region_at(scenario,39840)=='before-observation-change'
    assert challenge.challenge_region_at(scenario,39841)=='observation-boundary'
    assert challenge.challenge_region_at(scenario,41759)=='observation-boundary'
    assert challenge.challenge_region_at(scenario,41760)=='after-observation-change'
    endpoints=np.arange(2208,96001,480);reference=endpoints-960
    eligible=endpoints[(reference>=3072)&(reference+1536<=96000)]
    regions=[challenge.challenge_region_at(scenario,int(e)) for e in eligible]
    assert len(endpoints)==196 and len(eligible)==191
    assert regions.count('before-observation-change')==75
    assert regions.count('observation-boundary')==4
    assert regions.count('after-observation-change')==112
