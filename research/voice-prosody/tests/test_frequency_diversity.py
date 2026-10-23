"""Fixed metadata/candidate fixtures only; no scientific PCM, replay or model run."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from phenometrix_prosody import frequency_diversity as f
from phenometrix_prosody.model import FEATURE_NAMES_V2


@pytest.fixture(autouse=True)
def forbid_scientific_work(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('No scientific synthesis, replay or subprocess in fixture tests')
    monkeypatch.setattr(f.old.previous, '_synthesize_pcm', forbidden)
    monkeypatch.setattr(f.old, '_extract_feature', forbidden)
    monkeypatch.setattr(f.expansion, 'CoveragePeakWorker', forbidden)


def toy_record(old_hz=100., added_hz=300.):
    n=196; x=np.zeros((n,57),np.float32); hz=np.full((n,6),np.nan); mask=np.zeros((n,6),bool)
    x[:,:8]=[1,np.log2(old_hz/50),0,0,0,.9,0,.9];hz[:,0]=old_hz;mask[:,0]=True
    x[:,55]=np.float32(1/6);x[:,56]=1
    original=dict(features=x,candidate_hz=hz,candidate_mask=mask,reference_sample=np.arange(2208,96001,480),
        feature_names=np.array(FEATURE_NAMES_V2),sample_rate_hz=np.array(48000),pcm_sample_count=np.array(96000))
    evidence=dict(pcm_sha256='a'*64,pcm_sample_count=96000,ends=original['reference_sample'].copy(),rss_bytes=1)
    for name,value,score in [('old',old_hz,.9),('new',added_hz,.7 if added_hz!=old_hz else .9)]:
        evidence[name+'_hz']=np.full(n,value,np.float64);evidence[name+'_confidence']=np.full(n,score,np.float64)
        evidence[name+'_log2_hz_over_50']=np.full(n,np.log2(value/50),np.float64)
    expanded,provenance=f.expansion.additive_from_evidence(original,evidence)
    record=deepcopy(f.training_spec()['records'][0]);record['frequency_hz']=300.
    return original,expanded,provenance,record


def toy_support():
    records=[dict(id=f'toy-{a}-{b}-{c}',frequency_hz=100.+10*a,frequency_index=a,
        phase_index=b,phase_radians=[b,0.,0.],gain=.03+.01*c,gain_index=c,mixture_index=0)
        for a in range(6) for b in range(2) for c in range(2)]
    ids=np.tile(np.arange(24,dtype=np.int64),22)[:512]
    ids=np.concatenate((ids,ids));first=np.arange(len(ids))<512
    return records,ids,dict(added_only=first,counterexample=~first,broad=np.ones(len(ids),bool))


def test_exact_nested_metadata_and_contiguous_order_without_generation():
    spec=f.training_spec(); rows=spec['records']
    assert len(rows)==1536 and len({r['id'] for r in rows})==1536
    assert [r['frequency_index'] for r in rows[::8]]==[i for i in range(256) if i%4]
    assert [r['frequency_microhz'] for r in rows[::8]]==[53113000+2531250*i for i in range(256) if i%4]
    assert len({r['frequency_hz'] for r in rows})==192
    assert (rows[0]['frequency_hz'],rows[-1]['frequency_hz'])==(55.64425,698.58175)
    assert rows[0]['id']=='frequency-diversity-train-f1-p0-g0-m0'
    assert rows[-1]['id']=='frequency-diversity-train-f255-p1-g1-m1'
    assert [(r['phase_index'],r['gain_index'],r['mixture_index']) for r in rows[:8]]==[(j,k,m) for j in range(2) for k in range(2) for m in range(2)]
    assert spec['acquisition_rows']==301056 and spec['common_eligible_rows']==293376
    assert spec['duration_seconds_total']==3072 and 'Always terminal' in spec['stop']
    assert f.canonical_sha(spec)==f.canonical_sha(f.training_spec())


def test_metadata_collision_rejected_not_filtered(monkeypatch):
    monkeypatch.setattr(f.old,'challenge_spec',lambda:{'records':[{'frequency_hz':55.64425}]})
    with pytest.raises(ValueError,match='no filtering'):f.training_spec()


def test_support_exact_floors_and_numeric_frequency_identity():
    records,ids,strata=toy_support(); got=f.support_proof(strata,ids,records)
    assert got['pass_result'] and got['strata']['added_only']['rows']==512
    assert got['strata']['added_only']['distinct_cases']==24
    assert got['strata']['added_only']['distinct_carriers']==6
    assert sum(got['strata']['added_only']['frequency_bin_rows'])==512
    assert sum(got['strata']['added_only']['phase_rows'].values())==512
    assert sum(got['strata']['added_only']['gain_rows'].values())==512
    records[0]['frequency_hz']=records[1]['frequency_hz']=100.
    # Many distinct IDs and indices cannot manufacture extra physical carriers.
    for row in records:row['frequency_hz']=100.
    got=f.support_proof(strata,ids,records)
    assert not got['pass_result'] and got['strata']['added_only']['distinct_carriers']==1


@pytest.mark.parametrize('dimension',['rows','cases','carriers','phases','gains'])
def test_any_unchanged_floor_failure_is_terminal_no_go(dimension):
    records,ids,strata=toy_support()
    if dimension=='rows':strata['added_only'][0]=False
    if dimension=='cases':ids[ids==23]=22
    if dimension=='carriers':
        for r in records:
            if r['frequency_hz']==150.:r['frequency_hz']=140.
    if dimension=='phases':
        for r in records:r['phase_radians']=[0.,0.,0.]
    if dimension=='gains':
        for r in records:r['gain']=.03
    got=f.support_proof(strata,ids,records)
    assert not got['pass_result'] and 'No schedules' in got['terminal_policy']


@pytest.mark.parametrize('bad',['overlap','outside_broad','integer_mask','negative_case','past_case'])
def test_invalid_strata_rejected(bad):
    records,ids,strata=toy_support()
    if bad=='overlap':strata['counterexample'][0]=True
    elif bad=='outside_broad':strata['broad'][0]=False
    elif bad=='integer_mask':strata['broad']=strata['broad'].astype(int)
    elif bad=='negative_case':ids[0]=-1
    else:ids[0]=len(records)
    with pytest.raises(ValueError):f.support_proof(strata,ids,records)


def test_full_frame_toy_preserves_candidates_and_delayed_target():
    original,expanded,provenance,record=toy_record();before=deepcopy(original)
    table=f.classify_record(original,expanded,provenance,record,0);f.validate_table(table,[record])
    report=f.support_report(table,[record])
    assert table['stratum_added_only'].sum()==191 and not table['stratum_counterexample'].any()
    assert table['expanded_raw_features'].shape==(196,65)
    assert np.array_equal(table['reference_sample'],original['reference_sample']-960)
    assert [report['per_region'][r]['frames'] for r in ('before-observation-change','observation-boundary','after-observation-change')]==[75,4,112]
    assert report['fits']==report['schedules_created']==0 and not report['feasibility']['pass_result']
    for key in original:np.testing.assert_array_equal(original[key],before[key])


def test_counterexample_and_existing_exact_collision():
    original,expanded,p,r=toy_record(300.,600.);t=f.classify_record(original,expanded,p,r,0)
    assert t['stratum_counterexample'].sum()==191 and not t['stratum_added_only'].any()
    original,expanded,p,r=toy_record(300.,300.);t=f.classify_record(original,expanded,p,r,0)
    assert not t['stratum_counterexample'].any() and not t['new_frequency_added'].any()


@pytest.mark.parametrize('bad',['time','policy','eviction','distinct','case'])
def test_classification_rejects_corruption(bad):
    o,e,p,r=toy_record();i=0
    if bad=='time':e['reference_sample'][0]+=1
    elif bad=='policy':p['candidate_policy']=np.array('replacement')
    elif bad=='eviction':p['old_frequency_evicted'][0]=True
    elif bad=='distinct':p['new_frequency_added'][0]=False
    else:i=1536
    with pytest.raises(ValueError):f.classify_record(o,e,p,r,i)


@pytest.mark.parametrize('bad',['truth','reference','coverage','mask','schema','boolean'])
def test_table_validator_detects_saved_corruption(bad):
    o,e,p,r=toy_record();t=f.classify_record(o,e,p,r,0)
    if bad=='truth':t['truth_hz'][0]=301.
    elif bad=='reference':t['reference_sample'][0]+=1
    elif bad=='coverage':t['expanded_raw_covered'][0]=False
    elif bad=='mask':t['expanded_raw_mask'][0,0]=False
    elif bad=='schema':t['extra']=np.zeros(196)
    else:t['stratum_added_only']=t['stratum_added_only'].astype(int)
    with pytest.raises(ValueError):f.validate_table(t,[r])


def test_source_binding_and_snapshot_file_are_required(tmp_path,monkeypatch):
    repo=tmp_path;root=repo/'research';base=root/'runs'/'baseline';base.mkdir(parents=True)
    monkeypatch.setattr(f,'REPO',repo);monkeypatch.setattr(f,'ROOT',root);monkeypatch.setattr(f,'BASELINE',base)
    closure={}
    for i in range(58):
        path=repo/f'old-{i}.py';path.write_text(str(i));closure[path.name]=f.sha(path)
    freeze=base/'source-freeze.json';f.save(freeze,{'source_sha256':closure})
    monkeypatch.setattr(f,'BASELINE_HASHES',{'source-freeze.json':f.sha(freeze)})
    own=root/'producer.py';own.write_text('fixture');monkeypatch.setattr(f,'__file__',str(own))
    plan=root/'plan.md';plan.write_text('fixture');monkeypatch.setattr(f,'PLAN',plan)
    for name in ('tests/test_frequency_diversity.py','tests/test_audit_frequency_diversity.py','phenometrix_prosody/audit_frequency_diversity.py','phenometrix_prosody/audit_evidence_feasibility.py','phenometrix_prosody/audit_expanded_pool.py',
            *('phenometrix_prosody/'+x for x in ('audit_augmentation_experiment.py','audit_candidate_coverage.py','audit_harmonic_experiment.py',
            'audit_harmonic_inference.py','audit_selector_experiment.py','audit_subharmonic_experiment.py'))):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture')
    assert len(f.source_hashes())==71
    (repo/'old-0.py').write_text('mutated')
    with pytest.raises(ValueError,match='Historical source changed'):f.source_hashes()


def test_baseline_and_recheck_fail_on_byte_change(tmp_path,monkeypatch):
    monkeypatch.setattr(f,'REPO',tmp_path);monkeypatch.setattr(f,'BASELINE',tmp_path)
    p=tmp_path/'old.json';p.write_text('{}');expected=f.sha(p)
    monkeypatch.setattr(f,'BASELINE_HASHES',{'old.json':expected});monkeypatch.setattr(f,'EXTRA_BASELINE_HASHES',{})
    assert f.baseline_hashes()=={'old.json':expected}
    p.write_text('{"changed":true}')
    with pytest.raises(ValueError):f.baseline_hashes()
    with pytest.raises(ValueError):f.recheck({'old.json':expected})


def fixture_budget(tmp_path,monkeypatch):
    monkeypatch.setattr(f,'_rss',lambda kind:1234)
    monkeypatch.setattr(f.shutil,'disk_usage',lambda p:SimpleNamespace(free=32*1024**3))
    return dict(peak_observed_python_rss_bytes=0,peak_observed_completed_child_rss_bytes=0,peak_observed_peak_worker_rss_bytes=0)


def test_deadline_and_worker_timeout_are_bounded(tmp_path,monkeypatch):
    m=fixture_budget(tmp_path,monkeypatch);w=SimpleNamespace(timeout_seconds=999,peak_rss_bytes=100)
    left=f.budget(tmp_path,f.time.monotonic(),(datetime.now(timezone.utc)+timedelta(seconds=12)).isoformat(),m,w)
    assert 0<left<=12 and 0<w.timeout_seconds<=12
    with pytest.raises(ValueError,match='deadline'):
        f.budget(tmp_path,f.time.monotonic()-1801,(datetime.now(timezone.utc)+timedelta(seconds=1800)).isoformat(),m)


@pytest.mark.parametrize('kind',['memory','node','disk','output'])
def test_resource_limits_fail_closed(tmp_path,monkeypatch,kind):
    m=fixture_budget(tmp_path,monkeypatch)
    if kind=='memory':m['peak_observed_python_rss_bytes']=f.MAX_PYTHON_RSS+1
    if kind=='node':m['peak_observed_peak_worker_rss_bytes']=f.MAX_NODE_RSS+1
    if kind=='disk':monkeypatch.setattr(f.shutil,'disk_usage',lambda p:SimpleNamespace(free=1))
    if kind=='output':(tmp_path/'blob').write_bytes(b'ab');monkeypatch.setattr(f,'MAX_OUTPUT_BYTES',1)
    with pytest.raises(ValueError):f.budget(tmp_path,f.time.monotonic(),(datetime.now(timezone.utc)+timedelta(seconds=30)).isoformat(),m)


def test_supervisor_binding_and_exact_heap_option(tmp_path,monkeypatch):
    monkeypatch.setattr(f,'REPO',tmp_path);monkeypatch.setattr(f,'RUNS',tmp_path)
    p=tmp_path/'supervisor.py';p.write_text('fixture')
    monkeypatch.setenv('NODE_OPTIONS','--max-old-space-size=1536')
    monkeypatch.setenv('PHENOMETRIX_FREQUENCY_SUPERVISOR_PATH',str(p))
    monkeypatch.setenv('PHENOMETRIX_FREQUENCY_SUPERVISOR_SHA256',f.sha(p))
    assert f.supervisor_identity()=={'supervisor.py':f.sha(p)}
    p.write_text('changed')
    with pytest.raises(ValueError,match='bytes changed'):f.supervisor_identity()
    monkeypatch.delenv('NODE_OPTIONS')
    with pytest.raises(ValueError,match='heap'):f.supervisor_identity()


def test_no_overwrite_or_outside_run_directory_before_generation(tmp_path,monkeypatch):
    runs=tmp_path/'runs';runs.mkdir();old=runs/'old';old.mkdir()
    monkeypatch.setattr(f,'RUNS',runs)
    for target in (old,tmp_path/'outside',runs):
        with pytest.raises(ValueError,match='no overwrite'):f.run(target)


def test_bad_source_refused_before_output_or_generation(tmp_path,monkeypatch):
    runs=tmp_path/'runs';runs.mkdir();monkeypatch.setattr(f,'RUNS',runs)
    monkeypatch.setattr(f.shutil,'disk_usage',lambda p:SimpleNamespace(free=32*1024**3))
    def refuse():raise ValueError('bad source fixture')
    monkeypatch.setattr(f,'source_hashes',refuse)
    with pytest.raises(ValueError,match='bad source'):f.run(runs/'new')
    assert not (runs/'new').exists()


def test_postfreeze_failure_retains_failed_attempt_and_blocks_generation(tmp_path,monkeypatch):
    runs=tmp_path/'runs';runs.mkdir();monkeypatch.setattr(f,'REPO',tmp_path);monkeypatch.setattr(f,'RUNS',runs)
    monkeypatch.setattr(f.shutil,'disk_usage',lambda p:SimpleNamespace(free=32*1024**3))
    p=tmp_path/'source.py';p.write_text('fixed fixture')
    monkeypatch.setattr(f,'source_hashes',lambda:{'source.py':f.sha(p)})
    monkeypatch.setattr(f,'baseline_hashes',lambda:{})
    monkeypatch.setattr(f,'supervisor_identity',lambda:{})
    def refuse(*a,**kw):raise ValueError('deadline fixture')
    monkeypatch.setattr(f,'budget',refuse)
    output=runs/'attempt'
    with pytest.raises(ValueError,match='deadline'):f.run(output)
    m=json.loads((output/'manifest.json').read_text());freeze=json.loads((output/'source-freeze.json').read_text())
    assert m['status']=='failed' and m['generated_configurations']==m['fits']==m['schedules_created']==0
    assert m['source_freeze_sha256']==f.sha(output/'source-freeze.json')
    assert (output/'source-snapshot/source.py').read_text()=='fixed fixture'
    assert 1799<=(datetime.fromisoformat(freeze['deadline_utc'])-datetime.fromisoformat(freeze['frozen_at'])).total_seconds()<=1800
    with pytest.raises(ValueError,match='no overwrite'):f.run(output)


def test_captured_subprocess_failure_is_retained_bounded_without_process(tmp_path,monkeypatch):
    import subprocess
    monkeypatch.setattr(f,'REPO',tmp_path)
    error=subprocess.CalledProcessError(3,['fixture'],output='actual child stdout',stderr=b'x'*70000)
    result=f.preserve_child_output(error,tmp_path,f.sha)
    assert (tmp_path/'child-failure-stdout.txt').read_text()=='actual child stdout'
    assert result['stdout']['captured_bytes']==19 and not result['stdout']['truncated']
    assert result['stderr']['captured_bytes']==70000 and result['stderr']['retained_bytes']==65536
    assert result['stderr']['truncated'] and result['stderr']['sha256']==f.sha(tmp_path/'child-failure-stderr.txt')
    assert f.preserve_child_output(ValueError('fixture'),tmp_path,f.sha)=={}
