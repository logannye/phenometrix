"""Known-answer fixtures only; no study generation, model or saved study arrays."""
import copy
from pathlib import Path

import numpy as np
import pytest

from phenometrix_prosody import audit_frequency_diversity as a


def toy_table():
    old = np.array([[100.,np.nan],[300.,np.nan],[300.,np.nan],[100.,np.nan],[100.,np.nan]])
    expanded = np.column_stack((old[:,0],[300.,600.,np.nan,300.,300.]))
    grouped = expanded.copy(); grouped[3,1] = np.nan
    return {'truth_hz':np.full(5,300.), 'common_eligible':np.array([1,1,1,1,0],bool),
        'added_hz':np.array([300.,600.,300.,300.,300.]),
        'original_raw_hz':old,'original_raw_mask':np.isfinite(old),
        'expanded_raw_hz':expanded,'expanded_raw_mask':np.isfinite(expanded),
        'original_group_hz':old.copy(),'original_group_mask':np.isfinite(old),
        'expanded_group_hz':grouped,'expanded_group_mask':np.isfinite(grouped)}


def support():
    records = a.expected_records()[:48]  # Six carriers x eight complete settings.
    selected_cases = [8*f+4*p+2*g for f in range(6) for p in range(2) for g in range(2)]
    cases = np.repeat(selected_cases,48).astype(np.int64)
    selected = np.tile(np.arange(48)<24,len(selected_cases))
    return {'added_only':selected,'counterexample':~selected,'broad':np.ones(len(cases),bool)},cases,records


def test_rational_grid_is_fixed_new_only_cartesian_product():
    rows=a.expected_records();values=a.validate_numeric_metadata(rows)
    assert len(rows)==1536 and len(values)==192
    assert set(values)=={i for i in range(256) if i%4}
    assert rows[0]['frequency_hz']==55.64425 and rows[-1]['frequency_hz']==698.58175
    assert rows[0]['id']=='frequency-diversity-train-f1-p0-g0-m0'
    assert rows[-1]['id']=='frequency-diversity-train-f255-p1-g1-m1'
    assert not set(values.values()) & set(a.RESERVED)
    old={r['frequency_hz'] for r in a.independent.expected_records()}
    assert old=={(53_113_000+2_531_250*i)/1_000_000 for i in range(0,256,4)}
    assert not old & set(values.values())
    assert min(abs(f-r) for f in values.values() for r in a.RESERVED)==pytest.approx(.34425)


@pytest.mark.parametrize('kind',['duplicate_id','same_index_different_value','duplicate_numeric_carrier','fake_phase','fake_gain'])
def test_identifiers_cannot_manufacture_carrier_or_setting_diversity(kind):
    rows=a.expected_records()[:16]
    if kind=='duplicate_id':rows[8]['id']=rows[0]['id']
    elif kind=='same_index_different_value':rows[1]['frequency_hz']+=1
    elif kind=='duplicate_numeric_carrier':
        for row in rows[8:]:row['frequency_hz']=rows[0]['frequency_hz']
    elif kind=='fake_phase':rows[0]['phase_radians']=[0.,0.,0.]
    else:rows[0]['gain']=.99
    with pytest.raises(ValueError):a.validate_numeric_metadata(rows)


def test_actual_numeric_carrier_support_and_all_five_floors():
    strata,cases,records=support();out=a.support_report(strata,cases,records)
    assert out['pass_result'] and out['floor']=={'rows':512,'distinct_cases':24,'distinct_carriers':6,'distinct_phases':2,'distinct_gains':2}
    assert out['strata']['added_only']['rows']==576
    assert out['strata']['added_only']['distinct_cases']==24
    assert out['strata']['added_only']['distinct_carriers']==6
    assert sum(out['strata']['added_only']['frequency_bin_rows'])==576


def test_one_fewer_required_row_fails_without_changing_floor():
    strata,cases,records=support();strata['added_only'][np.flatnonzero(strata['added_only'])[511:]]=False
    out=a.support_report(strata,cases,records)
    assert not out['pass_result'] and not out['strata']['added_only']['checks']['rows']
    assert out['floor']['rows']==512


@pytest.mark.parametrize('kind',['negative_case','float_case','stratum_int','outside_broad','overlap'])
def test_support_refuses_invalid_row_identity_or_strata(kind):
    s,c,r=support()
    if kind=='negative_case':c[0]=-1
    elif kind=='float_case':c=c.astype(float)
    elif kind=='stratum_int':s['added_only']=s['added_only'].astype(int)
    elif kind=='outside_broad':s['broad'][0]=False
    else:s['counterexample'][0]=True
    with pytest.raises(ValueError):a.support_report(s,c,r)


def test_independent_strata_preserve_absence_grouping_loss_and_noncommon_rows():
    cov,s,d=a.independent.derive_strata(toy_table())
    assert s['added_only'].tolist()==[True,False,False,False,False]
    assert s['counterexample'].tolist()==[False,True,False,False,False]
    assert d.tolist()==[True,True,False,True,True]
    c=a.counts_for(np.array([1,1,1,1,0],bool),cov,s,d)
    assert c['frames']==4 and c['expanded_grouping_lost']==1
    assert c['added_only']==c['counterexample']==1


def test_inclusive50_cent_rule_and_missing_candidates():
    target=np.full(4,300.);hz=np.array([[300*2**(49.99999/1200)],[300*2**(50/1200)],[300*2**(50.00001/1200)],[np.nan]])
    # The declared floating-point log-cent rule is reconstructed, including exact
    # same arithmetic at the boundary; no epsilon is added by the auditor.
    expected=np.isfinite(hz[:,0]) & (np.abs(1200*np.log2(hz[:,0]/target))<=50)
    np.testing.assert_array_equal(a.independent.correct_coverage(hz,np.isfinite(hz),target),expected)
    assert expected[0] and not expected[2] and not expected[3]
    # The nominal mathematical boundary may round just outside; no tolerance is invented.


def test_timing_grid_has196_acquired191_common_and112_fully_post():
    t=a.independent.expected_support(2)
    assert len(t['case_index'])==392 and t['common_eligible'].sum()==382
    for i in range(2):
        m=(t['case_index']==i)&t['common_eligible']
        assert {r:int(np.sum(t['region'][m]==r)) for r in set(t['region'])}=={
            'before-observation-change':75,'observation-boundary':4,'after-observation-change':112}


def summary_fixture(frequency=100.,case_count=2):
    rows=[]
    for i in range(case_count):
        counts={'frames':191,'original_raw_covered':100,'expanded_raw_covered':191,
            'original_group_covered':100,'expanded_group_covered':191,'new_frequency_added':150,
            'original_grouping_lost':0,'expanded_grouping_lost':0,'added_only':10+i,'counterexample':20+i}
        rows.append({'case_index':i,'id':str(i),'frequency_hz':frequency+i,
            'phase_index':i%2,'gain_index':i%2,'mixture_index':i%2,'coverage':counts,
            'regions':{'post':counts.copy()}})
    return {'per_case':rows,'coverage':{k:sum(r['coverage'][k] for r in rows) for k in rows[0]['coverage']},
        'acquisition_rows':196*case_count,'common_eligible_rows':191*case_count}


def test_concentration_and_combined_support_are_descriptive_not_primary_override():
    one=summary_fixture();diag=a.carrier_diagnostics(one)
    assert diag['added_only']['qualifying_carriers_hz']==[100.,101.]
    assert diag['added_only']['adjacent_qualifying_carrier_gaps_hz']==[1.]
    assert diag['added_only']['largest_carrier_row_share']==11/21
    combined=a.combined_summary(one,summary_fixture(200.))
    assert combined['case_count']==4 and combined['coverage']['added_only']==42
    assert combined['common_eligible_rows']==764 and not combined['support']['pass_result']
    assert 'cannot rescue' in combined['role']
    with pytest.raises(ValueError,match='overlap'):a.combined_summary(one,one)


@pytest.mark.parametrize('name',['schedules.npz','checkpoint-freeze.json','prediction-freeze.json','weights.pt','model.onnx','predictions','challenge','natural'])
def test_no_fitting_evaluation_or_schedule_outputs_permitted_even_if_support_passes(tmp_path,name):
    p=tmp_path/name
    if '.' in name:p.write_bytes(b'toy')
    else:p.mkdir()
    with pytest.raises(ValueError,match='dependent'):a.assert_no_dependent_outputs(tmp_path)


def test_source_snapshot_source_filenames_do_not_count_as_scientific_outputs(tmp_path):
    p=tmp_path/'source-snapshot'/'research'/'model.onnx';p.parent.mkdir(parents=True);p.write_bytes(b'source toy')
    assert a.assert_no_dependent_outputs(tmp_path)['fits']==0


def chronology():
    return ({'sources_frozen_at':'2026-10-23T10:00:00Z'},
        {'started_at':'2026-10-23T10:00:01Z','completed_at':'2026-10-23T10:05:00Z'},
        {'started_at':'2026-10-23T10:00:02Z','completed_at':'2026-10-23T10:04:00Z'})


def test_chronology_has_freeze_before_generation_without_wallclock_access_claim():
    assert a.validate_chronology(*chronology())['sources_frozen_at']=='2026-10-23T10:00:00Z'


@pytest.mark.parametrize('kind',['late_freeze','reverse_acquisition','over_budget','missing','timezone'])
def test_invalid_phase_evidence_refuses(kind):
    s,m,g=chronology()
    if kind=='late_freeze':s['sources_frozen_at']='2026-10-23T10:03:00Z'
    elif kind=='reverse_acquisition':g['completed_at']='2026-10-23T10:06:00Z'
    elif kind=='over_budget':m['completed_at']='2026-10-23T10:30:02Z'
    elif kind=='missing':s.pop('sources_frozen_at')
    else:s['sources_frozen_at']='2026-10-23T10:00:00'
    with pytest.raises(ValueError):a.validate_chronology(s,m,g)


def test_full_prospective_metadata_matches_independent_enumeration_without_signal_generation():
    from phenometrix_prosody import frequency_diversity as producer
    rows=a.validate_spec(producer.training_spec())
    assert rows==a.expected_records()


def tiny_report_table():
    records=a.expected_records()[:8]
    table=a.independent.expected_support(8)
    n=len(table['case_index']);truth=np.repeat([r['frequency_hz'] for r in records],196)
    old=np.column_stack((np.full(n,50.),np.full(n,np.nan)))
    # Alternate fully declared toy cases between added-only and wrong-added.
    counter=table['case_index']%2==1
    old[counter,0]=truth[counter]
    added=np.where(counter,2*truth,truth)
    new=np.column_stack((old[:,0],added))
    table.update(truth_hz=truth,added_hz=added)
    for arm,hz in (('original',old),('expanded',new)):
        for rep in ('raw','group'):
            table[f'{arm}_{rep}_hz']=hz.copy();table[f'{arm}_{rep}_mask']=np.isfinite(hz)
    cov,s,distinct=a.independent.derive_strata(table)
    table.update({k+'_covered':v for k,v in cov.items()})
    table.update({'stratum_'+k:v for k,v in s.items()});table['new_frequency_added']=distinct
    return table,records


def test_toy_complete_report_matches_independent_counts_and_all_descriptives():
    from phenometrix_prosody import frequency_diversity as producer
    table,records=tiny_report_table()
    a.compare(producer.support_report(table,records),a.expected_report(table,records),'toy cross-implementation')
    assert a.expected_report(table,records)['feasibility']['strata']['added_only']['distinct_carriers']==1
    assert not a.expected_report(table,records)['feasibility']['pass_result']


def test_actual_top_level_phase_schema_and_missing_coverage_stage():
    source={'frozen_at':'2026-10-23T10:00:00Z'}
    manifest={'started_at':'2026-10-23T10:00:01Z','acquisition_started_at':'2026-10-23T10:00:02Z',
        'acquisition_completed_at':'2026-10-23T10:02:00Z','coverage_started_at':'2026-10-23T10:02:01Z',
        'coverage_completed_at':'2026-10-23T10:04:00Z','completed_at':'2026-10-23T10:04:01Z'}
    assert a.validate_chronology(source,manifest)['coverage_started_at']==manifest['coverage_started_at']
    del manifest['coverage_started_at']
    with pytest.raises(ValueError,match='timestamp'):a.validate_chronology(source,manifest)


def terminal_fixture():
    source={'maximum_fits':0,'maximum_schedules':0,'evaluation_generation_allowed':False,'natural_content_allowed':False,'maximum_seconds':1800}
    manifest={'fits':0,'schedules_created':0,'model_inference_performed':False,'natural_content_read':False,'evaluation_generated':False,
        'generated_configurations':1536,'expanded_configurations':1536,'maximum_seconds':1800,'maximum_output_bytes':8*1024**3,
        'node_options':'--max-old-space-size=1536','elapsed_seconds':1.,'observed_output_bytes':1,
        'peak_observed_python_rss_bytes':1,'peak_observed_completed_child_rss_bytes':1,'peak_observed_peak_worker_rss_bytes':1}
    return source,manifest


def test_terminal_scope_and_budget_are_exact_even_on_support_pass():
    a.validate_terminal_declarations(*terminal_fixture())


@pytest.mark.parametrize('field',['fits','schedules_created','model_inference_performed','natural_content_read','evaluation_generated','generated_configurations','elapsed_seconds','node_options'])
def test_terminal_scope_or_budget_change_refuses(field):
    source,manifest=terminal_fixture()
    manifest[field]={'fits':1,'schedules_created':1,'model_inference_performed':True,'natural_content_read':True,
        'evaluation_generated':True,'generated_configurations':1535,'elapsed_seconds':1801.,'node_options':'unbounded'}[field]
    with pytest.raises(ValueError):a.validate_terminal_declarations(source,manifest)
