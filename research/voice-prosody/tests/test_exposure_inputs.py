"""Fixed metadata, candidate-table and mocked acquisition tests; no scientific run."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from phenometrix_prosody import exposure_inputs as x
from test_frequency_diversity import toy_support, toy_record


def namespace(module, **changes):
    return SimpleNamespace(**({k:getattr(module,k) for k in dir(module) if not k.startswith('__')}|changes))


@pytest.fixture(autouse=True)
def no_scientific_processes(monkeypatch):
    def forbidden(*a, **kw):raise AssertionError('Scientific synthesis/replay forbidden in fixtures')
    monkeypatch.setattr(x,'cases',namespace(x.cases, previous=namespace(x.cases.previous,_synthesize_pcm=forbidden),_extract_feature=forbidden))


def fake_freeze(tmp_path, monkeypatch):
    repo=tmp_path;run=repo/'runs'/'fixture';(run/'checkpoints').mkdir(parents=True)
    monkeypatch.setattr(x,'REPO',repo);monkeypatch.setattr(x,'RUNS',repo/'runs')
    source=repo/'fixture.py';source.write_text('# source fixture\n');closure={'fixture.py':x.sha(source)}
    snapshot=run/'source-snapshot/fixture.py';snapshot.parent.mkdir();snapshot.write_bytes(source.read_bytes())
    monkeypatch.setattr(x,'required_source_names',lambda:{'fixture.py'})
    shared=dict(source_sha256=closure,deadline_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
        training_bank_manifest_sha256=x.PINS['manifest.json'],challenge_spec_sha256=x.cases.challenge_spec_sha256())
    initial=dict(shared,schema='phenometrix-exposure-source-freeze.v1',status='sources_frozen',fits={})
    sp=run/'source-freeze.json';x.save(sp,initial)
    checkpoints={}
    for a in x.ARMS:
        for seed in x.SEEDS:
            name=f'{a}__seed{seed}';p=run/'checkpoints'/(name+'.pt');p.write_bytes(name.encode())
            checkpoints[name]=dict(path=str(p.relative_to(repo)),sha256=x.sha(p))
    final=dict(shared,schema='phenometrix-exposure-checkpoint-freeze.v1',status='checkpoints_frozen',
        source_freeze_path=str(sp.relative_to(repo)),source_freeze_sha256=x.sha(sp),arms=list(x.ARMS),seeds=list(x.SEEDS),maximum_fits=12,checkpoints=checkpoints)
    cp=run/'checkpoint-freeze.json';x.save(cp,final)
    return run,sp,initial,cp,final


def test_new_bank_pins_and_fixed_challenge_metadata_are_separate():
    assert set(x.PINS)=={'manifest.json','source-freeze.json','spec.json','report.json','frame-table.npz','independent-audit.json'}
    assert len(x.bank.training_spec()['records'])==1536
    assert len(x.cases.training_spec()['records'])==512
    assert len(x.cases.challenge_spec()['records'])==12
    assert all(len(h)==64 for h in x.PINS.values())


def test_exact_case_first_sampling_is_reproducible_and_keeps_broad_uncovered():
    records,ids,strata=toy_support();one=x.targeted_schedule(strata,ids,records,20261022,0)
    same=x.targeted_schedule(strata,ids,records,20261022,0);different=x.targeted_schedule(strata,ids,records,20261022,1)
    np.testing.assert_array_equal(one['indices'],same['indices']);np.testing.assert_array_equal(one['origin'],same['origin'])
    assert not np.array_equal(one['indices'],different['indices'])
    assert one['indices'].shape==one['origin'].shape==(33,512)
    # An independent literal sampler confirms RNG draws and concatenation order.
    rng=np.random.default_rng(20261022+2_000_000);expected=[];expected_origin=[]
    for _ in range(33):
        row=[];origin=[]
        for label,(name,count) in enumerate(x.BATCH_COUNTS.items()):
            eligible=np.flatnonzero(strata[name]);available=np.unique(ids[eligible])
            chosen=rng.choice(available,count,replace=True)
            for case in chosen:
                pool=eligible[ids[eligible]==case];row.append(pool[rng.integers(len(pool))]);origin.append(label)
        shuffle=rng.permutation(512);expected.append(np.asarray(row)[shuffle]);expected_origin.append(np.asarray(origin)[shuffle])
    np.testing.assert_array_equal(one['indices'],expected);np.testing.assert_array_equal(one['origin'],expected_origin)
    table=dict(case_index=ids,common_eligible=np.ones(len(ids),bool),expanded_group_covered=np.arange(len(ids))%7!=0,region=np.where(np.arange(len(ids))%2,'after','before'))
    report=x.schedule_exposure(one,strata,table,len(records))
    assert report['requested_strata']==dict(added_only=4224,counterexample=4224,broad=8448)
    assert report['selected_rows']==16896 and report['expanded_group_uncovered_zero_rows']>0
    assert sum(report['per_case_selected_rows'])==16896 and report['actual_strata']['broad']==16896


@pytest.mark.parametrize('bad',['seed','epoch','support','origin','noncommon'])
def test_schedule_refuses_wrong_identity_or_stratum(bad):
    records,ids,strata=toy_support()
    if bad in ('seed','epoch','support'):
        if bad=='support':strata['added_only'][0]=False
        with pytest.raises(ValueError):x.targeted_schedule(strata,ids,records,1 if bad=='seed' else 20261022,12 if bad=='epoch' else 0)
    else:
        s=x.targeted_schedule(strata,ids,records,20261022,0)
        table=dict(case_index=ids,common_eligible=np.ones(len(ids),bool),expanded_group_covered=np.ones(len(ids),bool),region=np.zeros(len(ids),int))
        if bad=='origin':s['origin'][0,0]=3
        else:table['common_eligible'][s['indices'][0,0]]=False
        with pytest.raises(ValueError):x.schedule_exposure(s,strata,table,len(records))


def test_bad_saved_bank_bytes_refused_before_array_read(tmp_path,monkeypatch):
    p=tmp_path/'manifest.json';p.write_text('wrong bytes');monkeypatch.setattr(x,'TRAIN',tmp_path)
    calls=[]
    def bind(m,path,expected):
        calls.append(path.name)
        if x.sha(path)!=expected:raise ValueError('Hash differs')
        return path
    monkeypatch.setattr(x,'ledger',namespace(x.ledger,bind=bind,npz=lambda p:pytest.fail('No arrays after bad identity')))
    with pytest.raises(ValueError,match='Hash'):x.load_saved_training({},lambda:None)
    assert calls==['manifest.json']


def test_complete_checkpoint_gate_and_source_snapshots(tmp_path,monkeypatch):
    run,sp,initial,cp,final=fake_freeze(tmp_path,monkeypatch)
    assert x.validate_checkpoint_freeze(cp,x.sha(cp))==final
    (run/'source-snapshot/fixture.py').write_text('changed')
    with pytest.raises(ValueError,match='snapshot'):x.validate_checkpoint_freeze(cp,x.sha(cp))


@pytest.mark.parametrize('bad',['missing_checkpoint','changed_checkpoint','extra_checkpoint','empty_source','wrong_train','wrong_fresh','expired','bad_initial','wrong_path'])
def test_checkpoint_failure_precedes_output_or_synthesis(tmp_path,monkeypatch,bad):
    run,sp,initial,cp,final=fake_freeze(tmp_path,monkeypatch);name=next(iter(final['checkpoints']))
    if bad=='missing_checkpoint':final['checkpoints'].pop(name)
    elif bad=='changed_checkpoint':(tmp_path/final['checkpoints'][name]['path']).write_bytes(b'changed')
    elif bad=='extra_checkpoint':final['checkpoints']['extra']=final['checkpoints'][name]
    elif bad=='empty_source':
        initial['source_sha256']={};final['source_sha256']={};x.save(sp,initial);final['source_freeze_sha256']=x.sha(sp)
    elif bad=='wrong_train':final['training_bank_manifest_sha256']='0'*64
    elif bad=='wrong_fresh':final['challenge_spec_sha256']='0'*64
    elif bad=='expired':
        initial['deadline_utc']=final['deadline_utc']=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat();x.save(sp,initial);final['source_freeze_sha256']=x.sha(sp)
    elif bad=='bad_initial':initial['fits']={'partial':{}};x.save(sp,initial);final['source_freeze_sha256']=x.sha(sp)
    else:final['checkpoints'][name]['path']='elsewhere.pt'
    x.save(cp,final)
    with pytest.raises((ValueError,FileNotFoundError)):x.generate_challenge(run/'challenge',cp,x.sha(cp),lambda:60.)
    assert not (run/'challenge').exists()


def test_full_twelve_mock_acquisitions_are_checkpoint_gated_and_nonretryable(tmp_path,monkeypatch):
    run,sp,initial,cp,final=fake_freeze(tmp_path,monkeypatch);feature=toy_record()[0];calls=[]
    # Fixed zero PCM is a lifecycle fixture, not the analytical generator.
    def synth(record):
        assert len(x.validate_checkpoint_freeze(cp,x.sha(cp))['checkpoints'])==12
        calls.append(record['id']);return np.zeros(96000,np.float32)
    def extract(raw,full,timeout):
        assert timeout==17.;assert raw.stat().st_size==384000;full.write_text('fixed toy export')
        return deepcopy(feature)
    monkeypatch.setattr(x,'cases',namespace(x.cases,previous=namespace(x.cases.previous,_synthesize_pcm=synth),_extract_feature=extract))
    items,hashes=x.generate_challenge(run/'challenge',cp,x.sha(cp),lambda:17.)
    manifest=json.loads((run/'challenge/manifest.json').read_text())
    assert manifest['status']=='complete' and manifest['generated_configurations']==len(items)==12
    assert calls==[r['id'] for r in x.cases.challenge_spec()['records']]
    assert all(i['scenario']['kind']=='observation-change' and len(i['timing'].common_eligible)==196 for i in items)
    assert len(hashes)==27 and not (run/'challenge/intermediate').exists()
    with pytest.raises(ValueError,match='no retries'):x.generate_challenge(run/'challenge',cp,x.sha(cp),lambda:17.)


def test_partial_child_failure_preserves_actual_generated_count_and_bounded_logs(tmp_path,monkeypatch):
    import subprocess
    run,sp,initial,cp,final=fake_freeze(tmp_path,monkeypatch)
    def fail(*a):raise subprocess.CalledProcessError(1,['fixture'],output=b'output',stderr=b'e'*70000)
    monkeypatch.setattr(x,'cases',namespace(x.cases,previous=namespace(x.cases.previous,_synthesize_pcm=lambda _:np.zeros(96000,np.float32)),_extract_feature=fail))
    with pytest.raises(subprocess.CalledProcessError):x.generate_challenge(run/'challenge',cp,x.sha(cp),lambda:60.)
    saved=json.loads((run/'challenge/manifest.json').read_text())
    assert saved['status']=='failed' and saved['generated_configurations']==1 and saved['records']==[]
    assert saved['stderr_capture']['captured_bytes']==70000 and saved['stderr_capture']['retained_bytes']==65536
    with pytest.raises(ValueError,match='no retries'):x.generate_challenge(run/'challenge',cp,x.sha(cp),lambda:60.)
