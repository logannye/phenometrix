"""Versioned forensic admission/provenance fixtures, no scientific inputs."""
import copy
import inspect
import json
from types import SimpleNamespace
import numpy as np
import pytest
from phenometrix_prosody import audit_exposure_forensic_v2 as a
from phenometrix_prosody import audit_exposure_strict_analytic as s


def test_exact_analytic_body_has_only_declared_predicate_replacement():
    before=inspect.getsource(s.old.audit_analytic)
    old="require(scenario=={'id':name,'kind':'observation-change','durationSeconds':2.,'initialHz':params['frequency_hz'],'finalHz':params['frequency_hz'],'changeStartSeconds':.83,'changeEndSeconds':.83,'slopeHzPerSecond':0},'Fresh scenario differs')"
    new="require(scenario==expected_scenario(params),'Fresh scenario differs')"
    assert before.count(old)==1
    assert inspect.getsource(s.audit_analytic)==before.replace(old,new)


def test_all_twelve_full_scenarios_have_exact_independent_physical_fields():
    records=s.fresh_records()
    assert len(records)==12
    for p in records:
        got=s.expected_scenario(p)
        assert set(got)=={'id','kind','durationSeconds','initialHz','finalHz','changeStartSeconds',
            'changeEndSeconds','slopeHzPerSecond','change_sample','before','after','phase_radians','gain'}
        assert got=={'id':p['id'],'kind':'observation-change','durationSeconds':2.,'initialHz':p['frequency_hz'],
            'finalHz':p['frequency_hz'],'changeStartSeconds':.83,'changeEndSeconds':.83,'slopeHzPerSecond':0,
            'change_sample':39840,'before':[1.,.5,.25],'after':[0.,1.,.5],
            'phase_radians':p['phase_radians'],'gain':.057}


@pytest.mark.parametrize('field',list(s.expected_scenario(s.fresh_records()[0])))
@pytest.mark.parametrize('mode',['remove','change'])
def test_every_scenario_field_is_required_and_exact(field,mode):
    params=s.fresh_records()[0];actual=copy.deepcopy(s.expected_scenario(params))
    if mode=='remove':actual.pop(field)
    else:actual[field]='different'
    with pytest.raises(ValueError,match='Fresh scenario differs'):
        s.require(actual==s.expected_scenario(params),'Fresh scenario differs')


def test_extra_scenario_key_is_not_silently_dropped():
    params=s.fresh_records()[0];actual=s.expected_scenario(params)|{'extra':True}
    with pytest.raises(ValueError):s.require(actual==s.expected_scenario(params),'Fresh scenario differs')


@pytest.mark.parametrize('field',['frequency_hz','phase_radians','gain','before','after','change_sample','sample_count','sample_rate_hz','id'])
def test_physical_parameters_cannot_leave_frozen_enumeration(field):
    p=copy.deepcopy(s.fresh_records()[0]);p[field]='changed'
    with pytest.raises(ValueError):s.expected_scenario(p)


def test_combined_budget_counts_both_previous_attempts():
    assert a.remaining_budget(3.,167.)==1630.


@pytest.mark.parametrize('pair',[(-1,1),(1,-1),(float('nan'),1),(1,float('inf')),(900,900),(1801,0)])
def test_invalid_or_exhausted_budget_refuses(pair):
    with pytest.raises(ValueError):a.remaining_budget(*pair)


def declarations():
    m={'source_sha256':{'frozen':'a'*64},'failed_at':'2026-10-23T18:33:55+00:00'}
    artifacts={'manifest.json':'b'*64};sources={'new.py':'c'*64};lineage=dict(a.PRIOR_FILES)
    d={'schema':'phenometrix-exposure-posthoc-forensic-declaration.v2','posthoc':True,'original_scientific_status':'failed',
        'declared_at':'2026-10-23T19:00:00+00:00','artifact_sha256':artifacts.copy(),'audit_source_sha256':sources.copy(),
        'scientific_source_sha256':m['source_sha256'].copy(),'prior_attempt_sha256':lineage.copy(),
        'first_audit_elapsed_seconds':3.,'prior_forensic_elapsed_seconds':167.,'maximum_forensic_seconds':1630.,
        'maximum_python_rss_bytes':16*1024**3,'scenario_fields':list(s.expected_scenario(s.fresh_records()[0])),
        'source_status_or_metric_repair_allowed':False,'fitting_or_generation_allowed':False}
    return d,m,artifacts,sources,lineage


def test_distinct_declaration_preserves_failed_status_without_mutation():
    args=declarations();before=copy.deepcopy(args)
    assert a.validate_declaration(*args)
    assert args==before


@pytest.mark.parametrize('field',['schema','posthoc','original_scientific_status','declared_at','artifact_sha256',
    'audit_source_sha256','scientific_source_sha256','prior_attempt_sha256','maximum_forensic_seconds',
    'maximum_python_rss_bytes','scenario_fields','source_status_or_metric_repair_allowed','fitting_or_generation_allowed'])
def test_declaration_mismatch_cannot_bypass_previous_failure_or_budget(field):
    args=declarations();d=args[0]
    if field.endswith('sha256'):d[field]={'changed':'d'*64}
    elif field=='declared_at':d[field]='2026-10-23T18:00:00+00:00'
    elif field=='maximum_forensic_seconds':d[field]=1631.
    elif field=='maximum_python_rss_bytes':d[field]=32*1024**3
    elif field=='scenario_fields':d[field]=d[field][:-1]
    elif field in ('posthoc',):d[field]=False
    elif field.endswith('allowed'):d[field]=True
    else:d[field]='changed'
    with pytest.raises(ValueError):a.validate_declaration(*args)


def test_even_self_consistent_changed_prior_source_hash_is_refused():
    args=declarations();key=next(iter(a.PRIOR_FILES));args[-1][key]='e'*64;args[0]['prior_attempt_sha256'][key]='e'*64
    with pytest.raises(ValueError):a.validate_declaration(*args)


def test_completed_stage_persists_after_later_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'REPO',tmp_path)
    ledger=SimpleNamespace(comparisons=17,max_rate_delta=1e-16,peak_rss_bytes=1000,bind=lambda p:None)
    stages=a.Stages(tmp_path/'stages','a'*64,ledger)
    stages.add('first',{'valid':np.bool_(True),'count':np.int64(3)})
    with pytest.raises(ValueError):raise ValueError('later stage fails')
    found=a.completed_stages(stages.folder,'a'*64)
    assert len(found)==1 and found[0]['stage']=='first' and found[0]['numeric_comparisons_so_far']==17
    assert json.loads((stages.folder/'01-first.json').read_text())['result']=={'valid':True,'count':3}
    with pytest.raises(FileExistsError):a.Stages(stages.folder,'a'*64,ledger)
    with pytest.raises(ValueError):a.completed_stages(stages.folder,'b'*64)


def test_large_array_stage_metadata_retains_hash_not_raw_values():
    z=np.arange(12,dtype=np.float32).reshape(3,4)
    result=a.array_identities({'targets':z})
    assert result['targets']['shape']==[3,4] and result['targets']['dtype']=='float32'
    assert set(result['targets'])=={'shape','dtype','bytes_sha256'}


def test_receipt_serialization_accepts_numpy_scalar_but_refuses_raw_array(tmp_path):
    a.write_once(tmp_path/'receipt.json',{'pass':np.bool_(False)})
    with pytest.raises(FileExistsError):a.write_once(tmp_path/'receipt.json',{})
    with pytest.raises(TypeError):a.scalar(np.zeros(2))
