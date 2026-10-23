"""Toy aggregate fixtures only. No scientific input, PCM, inference or browser."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
from phenometrix_prosody import local_evidence_report as r


def counts(frames=10):
    return {'frames':frames,'original_raw_covered':frames//2,'expanded_raw_covered':frames,
        'original_group_covered':frames//2,'expanded_group_covered':frames-1,
        'new_frequency_added':frames//2,'original_grouping_lost':0,'expanded_grouping_lost':1,
        'added_only':frames//2-1,'counterexample':2}


def report():
    c=counts();return {'schema':'phenometrix-evidence-feasibility.v1','case_count':1,
        'acquisition_rows':12,'common_eligible_rows':10,'coverage':c,
        'feasibility':{'pass_result':False,'floor':{'rows':512},'strata':{
            'added_only':{'rows':4,'checks':{'rows':False}},'counterexample':{'rows':2,'checks':{'rows':False}}}},
        'per_case':[{'case_index':0,'id':'toy-case','frequency_hz':123.456,'phase_index':0,
            'gain_index':0,'mixture_index':0,'coverage':dict(c),'regions':{'toy-region':dict(c)}}],
        'schedule_exposure':{}}


def model_counts():
    score={'frames':10,'voiced_outputs':8,'nonvoiced_or_unavailable':2,'success_within_50c_all_frames':.5}
    coverage={'frames':10,'original_voiced_gate_positive':9,
        'methods':{'toy':{'voiced':8,'unvoiced':1,'unavailable':1}},
        'policies':{'expanded7':{'raw_covered':9,'grouped_covered':8,'grouping_lost':1,'candidate_unavailable':1}}}
    return score,coverage


def payload():
    return {'schema':r.SCHEMA,'created_at':'2026-10-23T00:00:00Z','support':r.support_view(report()),
        'model':None,'study':None,'support_audit':None,'evidence_artifacts_hash_verified':0,
        'source_freeze_sha256':'a'*64,'training_spec_sha256':'b'*64,'challenge_spec_sha256':'c'*64,
        'inputs_sha256':{},'claim_boundary':'Toy fixture; not study data.','renderer_role':'Presentation only',
        'repair_receipt':None}


def test_support_preserves_coverage_and_absence_without_claiming_model_accuracy():
    value=r.support_view(report());c=value['coverage']
    assert c['original_raw_absent']==5 and c['expanded_raw_absent']==0
    assert c['expanded_group_absent']==1 and c['expanded_grouping_lost']==1
    assert value['case_count']==1 and not value['feasibility']['pass_result']
    assert 'No model outcome' in value['meaning']


@pytest.mark.parametrize('kind',['global','region','group_gain','group_loss','integer'])
def test_support_rejects_inconsistent_aggregate_counts(kind):
    value=report()
    if kind=='global':value['coverage']['frames']=11
    elif kind=='region':value['per_case'][0]['regions']['toy-region']['added_only']=3
    elif kind=='group_gain':value['coverage']['original_group_covered']=6
    elif kind=='group_loss':value['coverage']['expanded_grouping_lost']=0
    else:value['coverage']['frames']=10.0
    with pytest.raises(ValueError):r.support_view(value)


def test_model_states_are_exact_but_ranking_overlap_is_bounded():
    score,cov=model_counts();value=r.model_region(score,cov,'toy')
    assert value['correct_pitch']==5 and value['incorrect_pitch']==3
    assert value['unvoiced']==1 and value['unavailable']==1
    assert value['raw_correct_candidate_absent']==1 and value['grouping_loss']==1
    assert (value['ranking_error_lower_bound'],value['ranking_error_upper_bound'])==(1,3)


def test_full_correct_candidate_coverage_identifies_wrong_ranking_count():
    score,cov=model_counts();cov['policies']['expanded7'].update(raw_covered=10,grouped_covered=10,grouping_lost=0,candidate_unavailable=0)
    cov['methods']['toy'].update(unvoiced=2,unavailable=0);cov['original_voiced_gate_positive']=8
    value=r.model_region(score,cov,'toy')
    assert value['ranking_error_lower_bound']==value['ranking_error_upper_bound']==3


def test_missing_state_coverage_stays_unavailable_not_zero():
    score,_=model_counts();value=r.model_region(score,None,'toy')
    assert value['incorrect_pitch']==3 and value['unvoiced'] is None and value['unavailable'] is None
    assert value['ranking_error_lower_bound'] is None


@pytest.mark.parametrize('kind',['rate','states','coverage','voicing','fractional'])
def test_model_rejects_changed_denominators_or_unidentifiable_integer_score(kind):
    score,cov=model_counts()
    if kind=='rate':score['success_within_50c_all_frames']=1.1
    elif kind=='states':cov['methods']['toy']['unvoiced']=2
    elif kind=='coverage':cov['policies']['expanded7']['grouping_lost']=0
    elif kind=='voicing':cov['original_voiced_gate_positive']=8
    else:score['success_within_50c_all_frames']=.123
    with pytest.raises(ValueError):r.model_region(score,cov,'toy')


def test_zero_region_is_preserved_with_no_rate():
    score={'frames':0,'voiced_outputs':0,'nonvoiced_or_unavailable':0,'success_within_50c_all_frames':None}
    value=r.model_region(score,None,'toy')
    assert value['frames']==0 and value['success_rate'] is None


def test_html_is_self_contained_and_untrusted_strings_cannot_close_data_element():
    value=payload();value['support']['cases'][0]['id']='</script><img src=x onerror=alert(1)>'
    value['claim_boundary']='<script>alert("injection")</script>'
    page=r.render_html(value)
    assert '</script><img' not in page and '\\u003c/script' in page
    assert '&lt;script&gt;' in page and "connect-src 'none'" in page
    assert '<script src=' not in page and '<link ' not in page
    assert 'No completed model study was supplied' in page
    assert 'Not assessed' in page


def test_study_success_never_implies_browser_or_production_promotion():
    value=payload();value['model']={'decision':{'promising_for_next_study':True}}
    page=r.render_html(value)
    assert 'All scientific gates passed; research qualification remains separate' in page


def test_write_report_binds_snapshots_and_refuses_overwrite(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'RUNS',tmp_path);monkeypatch.setattr(r,'REPO',tmp_path)
    p=tmp_path/'toy-aggregate.json';p.write_text('{"count":1}')
    bindings=r.Bindings(tmp_path);bindings.read(p,snapshot='toy.json');value=payload();value['inputs_sha256']=dict(bindings.hashes)
    out=tmp_path/'report';manifest=r.write_report(out,value,bindings)
    assert manifest['status']=='complete' and not manifest['network_required']
    for path,digest in manifest['artifacts_sha256'].items():assert r.sha(out/path)==digest
    assert (out/'input-snapshot/toy.json').read_bytes()==p.read_bytes()
    assert (out/'source-snapshot/local_evidence_report.py').exists()
    assert (out/'index.html').exists() and (out/'report-data.json').exists()
    with pytest.raises(ValueError):r.write_report(out,value,bindings)


def test_changed_bound_input_refuses_report_before_output_creation(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'RUNS',tmp_path);monkeypatch.setattr(r,'REPO',tmp_path)
    p=tmp_path/'toy.json';p.write_text('{}');b=r.Bindings(tmp_path);b.read(p);p.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='changed'):r.write_report(tmp_path/'report',payload(),b)
    assert not (tmp_path/'report').exists()


def test_bindings_reject_nonfinite_json_and_outside_repository(tmp_path):
    b=r.Bindings(tmp_path);p=tmp_path/'bad.json';p.write_text('{"count":NaN}')
    with pytest.raises(ValueError,match='Nonfinite'):b.read(p)
    with pytest.raises(ValueError):b.bind(tmp_path.parent/'outside')


def collection_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'REPO',tmp_path);monkeypatch.setattr(r,'RUNS',tmp_path/'runs')
    run=tmp_path/'runs/toy';prep=run/'feasibility';prep.mkdir(parents=True)
    source={'status':'sources_frozen','fits':{},'training_spec_sha256':'a'*64,'challenge_spec_sha256':'b'*64,
        'deadline_utc':'2026-10-23T12:18:58Z'}
    sourcepath=run/'source-freeze.json';sourcepath.write_text(json.dumps(source))
    reportpath=prep/'report.json';reportpath.write_text(json.dumps(report()))
    manifest={'schema':'phenometrix-evidence-feasibility.v1','status':'complete','feasibility_pass':False,
        **{k:source[k] for k in ('training_spec_sha256','challenge_spec_sha256','deadline_utc')},
        'source_freeze_path':str(sourcepath.relative_to(tmp_path)),'source_freeze_sha256':r.sha(sourcepath),
        'report_path':str(reportpath.relative_to(tmp_path)),'report_sha256':r.sha(reportpath)}
    (prep/'manifest.json').write_text(json.dumps(manifest))
    return run,prep,manifest


def test_collect_support_only_with_explicit_bound_repair_receipt(tmp_path,monkeypatch):
    run,prep,manifest=collection_fixture(tmp_path,monkeypatch)
    repair=tmp_path/'repair.json';repair.write_text(json.dumps({'status':'recorded','failed_attempt':'toy-v1','fits':0,'fresh_cases':0}))
    value,bound=r.collect(prep,repair_receipt=repair,expected_repair_sha256=r.sha(repair))
    assert value['model'] is None and value['study'] is None and value['repair_receipt']['fits']==0
    assert bound.snapshots['repair-receipt.json']==repair.read_bytes()
    assert value['support_audit'] is None and value['evidence_artifacts_hash_verified']==0
    with pytest.raises(ValueError,match='hash differs'):
        r.collect(prep,repair_receipt=repair,expected_repair_sha256='0'*64)


def test_present_failed_or_mismatched_audit_cannot_be_displayed_as_pass(tmp_path,monkeypatch):
    _,prep,m=collection_fixture(tmp_path,monkeypatch)
    p=prep/'independent-audit.json';receipt={'schema':'phenometrix-evidence-feasibility-audit.v1','pass':False,
        'preparation_manifest_sha256':r.sha(prep/'manifest.json'),'report_sha256':m['report_sha256'],'feasibility_pass':False}
    p.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='audit'):r.collect(prep)
    receipt['pass']=True;p.write_text(json.dumps(receipt));value,_=r.collect(prep)
    assert value['support_audit']['pass'] is True
    receipt['report_sha256']='0'*64;p.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='audit'):r.collect(prep)


def test_incomplete_study_cannot_populate_model_outcomes(tmp_path,monkeypatch):
    run,prep,_=collection_fixture(tmp_path,monkeypatch)
    (run/'manifest.json').write_text(json.dumps({'schema':'phenometrix-evidence-experiment.v1','status':'training_frozen'}))
    with pytest.raises(ValueError,match='completed study'):r.collect(prep,study=run)
