"""Posthoc wrapper admission/provenance fixtures; no scientific files read."""
import copy
import pytest
from phenometrix_prosody import audit_exposure_forensic as a


def failed_manifest():
    return {'schema':'phenometrix-exposure-study.v1','status':'failed','failure':a.EXPECTED_FAILURE,
        'completed_at':'2026-10-23T18:33:55.042829+00:00','failed_at':'2026-10-23T18:33:55.283251+00:00',
        'fits':{n:{} for n in a.frozen.FIT_NAMES},'fit_attempts':{n:{'status':'complete'} for n in a.frozen.FIT_NAMES},
        'prediction_sha256':{str(i):'a'*64 for i in range(103)},'fresh_evaluation_configurations':12,
        'training_generation_configurations':0,'metrics_sha256':'b'*64,
        'development_references_read_for_final_scoring':True,'reserved_content_read':False,'source_sha256':{'frozen.py':'c'*64}}


def test_actual_failed_manifest_admitted_without_mutation_or_status_shim():
    m=failed_manifest();before=copy.deepcopy(m)
    result=a.require_saved_failed_result(m)
    assert m==before and result['scientific_status']=='failed' and not result['scientific_status_changed']
    assert result['scientific_failure']==a.EXPECTED_FAILURE


@pytest.mark.parametrize('field',['schema','status','failure','completed_at','fits','fit_attempts','prediction_sha256',
    'fresh_evaluation_configurations','training_generation_configurations','metrics_sha256',
    'development_references_read_for_final_scoring','reserved_content_read'])
def test_incomplete_or_reclassified_attempt_refused(field):
    m=failed_manifest()
    if field=='schema':m[field]='other'
    elif field=='status':m[field]='complete'
    elif field=='failure':m[field]='unrelated failure'
    elif field=='completed_at':m[field]='2026-10-23T18:34:00+00:00'
    elif field=='fits':m[field].pop(next(iter(m[field])))
    elif field=='fit_attempts':m[field][next(iter(m[field]))]['status']='started'
    elif field=='prediction_sha256':m[field].pop(next(iter(m[field])))
    elif field=='fresh_evaluation_configurations':m[field]=11
    elif field=='training_generation_configurations':m[field]=1
    elif field=='metrics_sha256':m[field]='missing'
    elif field=='development_references_read_for_final_scoring':m[field]=False
    else:m[field]=True
    with pytest.raises(ValueError):a.require_saved_failed_result(m)


def declaration_fixture():
    m=failed_manifest();artifacts={name:'a'*64 for name in a.ARTIFACTS};sources={'wrapper.py':'b'*64}
    d={'schema':'phenometrix-exposure-posthoc-forensic-declaration.v1','posthoc':True,'original_scientific_status':'failed',
        'declared_at':'2026-10-23T19:00:00+00:00','artifact_sha256':artifacts.copy(),'audit_source_sha256':sources.copy(),
        'scientific_source_sha256':m['source_sha256'].copy(),'first_audit_elapsed_seconds':3.,'maximum_forensic_seconds':1797.,
        'maximum_python_rss_bytes':16*1024**3}
    return d,m,artifacts,sources


def test_posthoc_declaration_requires_exact_sources_and_remaining_combined_budget():
    assert a.validate_declaration(*declaration_fixture())


@pytest.mark.parametrize('field',['schema','posthoc','original_scientific_status','declared_at','artifact_sha256',
    'audit_source_sha256','scientific_source_sha256','first_audit_elapsed_seconds','maximum_forensic_seconds','maximum_python_rss_bytes'])
def test_posthoc_declaration_changes_refuse(field):
    args=declaration_fixture();d=args[0]
    if field=='schema':d[field]='not-posthoc'
    elif field=='posthoc':d[field]=False
    elif field=='original_scientific_status':d[field]='complete'
    elif field=='declared_at':d[field]='2026-10-23T18:00:00+00:00'
    elif field.endswith('sha256'):d[field]={'changed':'c'*64}
    elif field=='first_audit_elapsed_seconds':d[field]=1800
    elif field=='maximum_forensic_seconds':d[field]=1800
    else:d[field]=32*1024**3
    with pytest.raises(ValueError):a.validate_declaration(*args)


def test_frozen_helper_identities_are_pinned_and_not_changed_by_import():
    assert a.base.sha(a.frozen.__file__)==a.FROZEN_AUDITOR_SHA
    assert a.base.sha(a.ROOT/'tests/test_audit_exposure_study.py')==a.FROZEN_TEST_SHA
    hashes=a.source_hashes()
    assert hashes['research/voice-prosody/phenometrix_prosody/audit_exposure_study.py']==a.FROZEN_AUDITOR_SHA
    assert 'research/voice-prosody/phenometrix_prosody/audit_exposure_forensic.py' in hashes
    assert 'research/voice-prosody/tests/test_audit_exposure_forensic.py' in hashes
