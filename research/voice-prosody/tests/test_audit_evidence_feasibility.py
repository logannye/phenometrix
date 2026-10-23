"""Independent-auditor toy tables only; no study files or waveform generation."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from phenometrix_prosody import audit_evidence_feasibility as a


def table():
    old=np.array([[100.,np.nan],[300.,np.nan],[300.,np.nan],[100.,np.nan],
                  [100.,np.nan],[300.,np.nan]])
    expanded=np.column_stack((old[:,0],[300.,600.,np.nan,300.,300.,np.nan]))
    grouped=expanded.copy();grouped[3,1]=np.nan
    return {'truth_hz':np.full(6,300.),'common_eligible':np.array([1,1,1,1,0,1],bool),
        'added_hz':np.array([300.,600.,300.,300.,300.,np.nan]),
        'original_raw_hz':old,'original_raw_mask':np.isfinite(old),
        'expanded_raw_hz':expanded,'expanded_raw_mask':np.isfinite(expanded),
        'original_group_hz':old.copy(),'original_group_mask':np.isfinite(old),
        'expanded_group_hz':grouped,'expanded_group_mask':np.isfinite(grouped)}


def support():
    #24 toy cases span six carriers, both phases and gains. These are indices,
    #not generated scientific waveforms or observed training-frame outcomes.
    case=np.repeat([8*f+4*p+2*g for f in range(6) for p in range(2) for g in range(2)],48).astype(np.int64)
    which=np.tile(np.arange(48)<24,24)
    return {'added_only':which,'counterexample':~which,'broad':np.ones(len(case),bool)},case


def test_declared_metadata_is_independently_reconstructed():
    records=a.expected_records()
    assert len(records)==512 and len({r['id'] for r in records})==512
    assert records[0]['frequency_hz']==53.113 and records[-1]['frequency_hz']==690.988
    assert records[0]['id']=='evidence-train-f0-p0-g0-m0'
    assert records[-1]['id']=='evidence-train-f63-p1-g1-m1'
    assert {tuple(r['after']) for r in records}=={(0.,1.,.5),(.2,1.,.5)}


def test_support_preserves_all196_and_common191_per_case():
    s=a.expected_support()
    assert len(s['case_index'])==100352 and s['common_eligible'].sum()==97792
    for c in (0,511):
        rows=s['case_index']==c;common=rows&s['common_eligible']
        assert rows.sum()==196 and common.sum()==191
        assert np.array_equal(s['reference_sample'][rows],s['availability_sample'][rows]-960)
        assert {r:int(np.sum(s['region'][common]==r)) for r in set(s['region'])}=={
            'before-observation-change':75,'observation-boundary':4,'after-observation-change':112}


def test_strata_distinguish_missing_correct_candidate_wrong_added_peak_and_grouping():
    coverage,strata,distinct=a.derive_strata(table())
    assert strata['added_only'].tolist()==[True,False,False,False,False,False]
    assert strata['counterexample'].tolist()==[False,True,False,False,False,False]
    assert strata['broad'].tolist()==[True,True,True,True,False,True]
    assert distinct.tolist()==[True,True,False,True,True,False]
    assert coverage['expanded_raw'][3] and not coverage['expanded_group'][3]


@pytest.mark.parametrize('kind',['infinite_added','missing_added','eviction','invented_group','bad_common'])
def test_stratum_integrity_refuses_invalid_evidence(kind):
    t=table()
    if kind=='infinite_added':t['added_hz'][0]=np.inf
    elif kind=='missing_added':t['expanded_raw_mask'][0,1]=False
    elif kind=='eviction':t['expanded_raw_mask'][2,0]=False
    elif kind=='invented_group':t['original_group_hz'][0,0]=300.
    else:t['common_eligible']=t['common_eligible'].astype(int)
    with pytest.raises(ValueError):a.derive_strata(t)


def test_masked_nonfinite_candidate_is_absent_not_zero_cent_match():
    hz=np.array([[np.nan,np.nan],[np.nan,300.]])
    mask=np.array([[False,False],[False,True]])
    assert a.correct_coverage(hz,mask,np.array([300.,300.])).tolist()==[False,True]


def test_exact_declared_floors_pass_and_one_short_fails():
    strata,cases=support();report=a.support_report(strata,cases)
    assert report['pass_result']
    assert report['strata']['added_only']['rows']==576
    assert report['strata']['added_only']['distinct_cases']==24
    assert report['strata']['added_only']['distinct_carriers']==6
    strata['added_only'][np.flatnonzero(strata['added_only'])[511:]]=False
    report=a.support_report(strata,cases)
    assert not report['pass_result'] and not report['strata']['added_only']['checks']['rows']


def test_floor_failure_cannot_create_replacement_schedule():
    strata,cases=support();strata['counterexample'][:]=False
    with pytest.raises(ValueError,match='Insufficient support'):
        a.reconstruct_schedule(strata,cases,20261022,0)


def test_deterministic_case_balanced_replacement_and_exact_origins():
    strata,cases=support();one=a.reconstruct_schedule(strata,cases,20261022,0)
    two=a.reconstruct_schedule(strata,cases,20261022,0)
    other=a.reconstruct_schedule(strata,cases,20261023,0)
    np.testing.assert_array_equal(one['indices'],two['indices'])
    np.testing.assert_array_equal(one['origin'],two['origin'])
    assert not np.array_equal(one['indices'],other['indices'])
    out=a.exposure(one,strata,cases)
    assert out['selected_rows']==16896 and out['duplicate_draws']>0
    assert out['requested_strata']=={'added_only':4224,'counterexample':4224,'broad':8448}
    assert out['actual_strata']['broad']==16896
    assert out['actual_strata']['added_only']>4224 and out['actual_strata']['counterexample']>4224
    assert sum(out['per_case_selected_rows'])==16896


@pytest.mark.parametrize('kind',['index','origin','membership','batch_count'])
def test_exposure_rejects_hidden_row_or_origin_changes(kind):
    strata,cases=support();saved=a.reconstruct_schedule(strata,cases,20261022,0)
    if kind=='index':saved['indices'][0,0]=len(cases)
    elif kind=='origin':saved['origin'][0,0]=3
    elif kind=='membership':
        i=np.flatnonzero(saved['origin'][0]==0)[0]
        saved['indices'][0,i]=np.flatnonzero(strata['counterexample'])[0]
    else:
        i=np.flatnonzero(saved['origin'][0]==0)[0];saved['origin'][0,i]=2
    with pytest.raises(ValueError):a.exposure(saved,strata,cases)


def source_fixture(tmp_path):
    run=tmp_path/'runs'/'toy';run.mkdir(parents=True)
    source={'schema':'phenometrix-evidence-source-freeze.v1','status':'sources_frozen','fits':{},
        'arms':list(a.ARMS),'seeds':list(a.SEEDS),'maximum_fits':12,
        'deadline_utc':'2026-10-23T12:18:58Z','source_sha256':{},
        'training_spec_sha256':'a'*64,'challenge_spec_sha256':'b'*64}
    path=run/'source-freeze.json';path.write_text(json.dumps(source))
    return a.Ledger(tmp_path),path,source


def test_readonly_phase_audit_does_not_require_unexpired_deadline(tmp_path):
    ledger,path,source=source_fixture(tmp_path)
    assert a.validate_phase(ledger,path,source,path.parent/'challenge',feasibility_pass=False)=={
        'fresh_present':False,'final_checkpoint_bindings_checked':0}


@pytest.mark.parametrize('kind',['checkpoint','fresh'])
def test_no_go_refuses_any_dependent_work(tmp_path,kind):
    ledger,path,source=source_fixture(tmp_path)
    if kind=='checkpoint':
        (path.parent/'checkpoints').mkdir();(path.parent/'checkpoints'/'unplanned.pt').write_bytes(b'toy')
    else:(path.parent/'challenge').mkdir()
    with pytest.raises(ValueError,match='No-go'):
        a.validate_phase(ledger,path,source,path.parent/'challenge',feasibility_pass=False)


def test_fresh_directory_without_twelve_final_checkpoint_freeze_refuses(tmp_path):
    ledger,path,source=source_fixture(tmp_path);(path.parent/'challenge').mkdir()
    with pytest.raises(ValueError,match='missing'):
        a.validate_phase(ledger,path,source,path.parent/'challenge',feasibility_pass=True)


@pytest.mark.parametrize('kind',['escape','wrong_digest','later_change'])
def test_artifact_identity_and_repository_boundary(tmp_path,kind):
    ledger=a.Ledger(tmp_path);p=tmp_path/'source';p.write_bytes(b'toy')
    if kind=='escape':
        with pytest.raises(ValueError):ledger.bind('../outside')
    elif kind=='wrong_digest':
        with pytest.raises(ValueError,match='identity'):ledger.bind('source','0'*64)
    else:
        ledger.bind('source');p.write_bytes(b'changed')
        with pytest.raises(ValueError,match='changed'):ledger.recheck()


def test_nested_report_does_not_allow_missing_failure_or_extra_unchecked_field():
    expected={'pass_result':False,'rows':[0,1]}
    a.compare(copy.deepcopy(expected),expected)
    for altered in ({'pass_result':True,'rows':[0,1]}, {'pass_result':False}, {**expected,'ignored':1}):
        with pytest.raises(ValueError):a.compare(altered,expected)


def test_array_comparison_preserves_nan_masks_and_exact_dtype():
    a.array_equal(np.array([np.nan,300.]),np.array([np.nan,300.]),'Hz')
    with pytest.raises(ValueError,match='dtype'):
        a.array_equal(np.array([1],np.float32),np.array([1],np.float64),'feature')
    with pytest.raises(ValueError,match='values'):
        a.array_equal(np.array([np.nan,300.]),np.array([0.,300.]),'Hz')


def test_report_boolean_cannot_be_integer_surrogate():
    with pytest.raises(ValueError,match='Boolean'):
        a.compare({'pass_result':1},{'pass_result':True})


def chronology_fixture():
    return ({'sources_frozen_at':'2026-10-23T05:00:00Z','deadline_utc':'2026-10-23T12:18:58Z'},
        {'started_at':'2026-10-23T05:01:00Z','completed_at':'2026-10-23T05:05:00Z'},
        {'started_at':'2026-10-23T05:02:00Z','completed_at':'2026-10-23T05:04:00Z'})


def test_source_acquisition_preparation_chronology_is_explicit():
    source,manifest,acquisition=chronology_fixture()
    result=a.chronology(source,manifest,acquisition)
    assert list(result)==['sources_frozen_at','preparation_started_at','acquisition_started_at',
        'acquisition_completed_at','preparation_completed_at']


@pytest.mark.parametrize('kind',['source_after','acquisition_after','late','no_zone','missing'])
def test_chronology_refuses_missing_or_inverted_phase_evidence(kind):
    source,manifest,acquisition=chronology_fixture()
    if kind=='source_after':source['sources_frozen_at']='2026-10-23T05:03:00Z'
    elif kind=='acquisition_after':acquisition['completed_at']='2026-10-23T05:06:00Z'
    elif kind=='late':manifest['completed_at']='2026-10-23T12:18:59Z'
    elif kind=='no_zone':source['sources_frozen_at']='2026-10-23T05:00:00'
    else:source.pop('sources_frozen_at')
    with pytest.raises(ValueError):a.chronology(source,manifest,acquisition)


def test_summary_keeps_uncovered_noncommon_and_grouping_losses_visible():
    t=table();t.update(case_index=np.zeros(6,np.int64),region=np.array(['pre','post','post','post','pre','post']))
    records=[a.expected_records()[0]]
    summary=a.table_summary(t,records)
    assert summary['acquisition_rows']==6 and summary['common_eligible_rows']==5
    assert summary['coverage']['frames']==5
    assert summary['coverage']['added_only']==1 and summary['coverage']['counterexample']==1
    assert summary['coverage']['expanded_grouping_lost']==1
    assert summary['per_case'][0]['regions']['post']['frames']==4
    assert not summary['feasibility']['pass_result'] and summary['schedule_exposure']=={}


def final_phase_fixture(tmp_path):
    ledger,path,source=source_fixture(tmp_path);run=path.parent
    (run/'checkpoints').mkdir();bindings={}
    for arm in a.ARMS:
        for seed in a.SEEDS:
            name=f'{arm}__seed{seed}';p=run/'checkpoints'/(name+'.pt')
            p.write_bytes(('toy checkpoint identity '+name).encode())
            bindings[name]={'path':str(p.relative_to(tmp_path)),'sha256':a.sha(p)}
    checkpoint={k:source[k] for k in ('source_sha256','training_spec_sha256','challenge_spec_sha256',
        'arms','seeds','maximum_fits','deadline_utc')}
    checkpoint.update(schema='phenometrix-evidence-checkpoint-freeze.v1',status='checkpoints_frozen',
        source_freeze_path=str(path.relative_to(tmp_path)),source_freeze_sha256=a.sha(path),checkpoints=bindings)
    cp=run/'checkpoint-freeze.json';cp.write_text(json.dumps(checkpoint))
    fresh=run/'challenge';fresh.mkdir()
    (fresh/'manifest.json').write_text(json.dumps({'checkpoint_freeze_path':str(cp.relative_to(tmp_path)),
        'checkpoint_freeze_sha256':a.sha(cp)}))
    return ledger,path,source,fresh,cp,checkpoint


def test_fresh_phase_requires_all_twelve_named_bound_checkpoints(tmp_path):
    ledger,path,source,fresh,_,_=final_phase_fixture(tmp_path)
    assert a.validate_phase(ledger,path,source,fresh,feasibility_pass=True)=={
        'fresh_present':True,'final_checkpoint_bindings_checked':12}


@pytest.mark.parametrize('kind',['missing','extra','modified','source_link','arm','spec'])
def test_fresh_phase_refuses_partial_or_changed_freeze(tmp_path,kind):
    ledger,path,source,fresh,cp,checkpoint=final_phase_fixture(tmp_path)
    if kind=='missing':next((path.parent/'checkpoints').iterdir()).unlink()
    elif kind=='extra':(path.parent/'checkpoints'/'extra.pt').write_bytes(b'toy')
    elif kind=='modified':next((path.parent/'checkpoints').iterdir()).write_bytes(b'changed')
    else:
        if kind=='source_link':checkpoint['source_freeze_sha256']='0'*64
        elif kind=='arm':checkpoint['arms']=['substitute']
        else:checkpoint['challenge_spec_sha256']='0'*64
        cp.write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError):a.validate_phase(ledger,path,source,fresh,feasibility_pass=True)
