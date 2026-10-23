"""Toy-only audit corruption, schedule, equation and phase fixtures."""
import copy
from datetime import datetime,timedelta,timezone
import numpy as np
import pytest
import torch
from phenometrix_prosody import audit_exposure_study as a
from phenometrix_prosody import exposure_inputs as producer
from phenometrix_prosody import exposure_study as runner
from phenometrix_prosody.periodicity_evidence import periodicity_evidence
from phenometrix_prosody.evidence_selector_model import EvidenceHarmonicSelector
from test_evidence_selector_model import batch,evidence
from test_audit_evidence_scores import cells,gates,phase_fixture


def toy_schedule_inputs():
    records=a.bank_audit.expected_records()[:64]
    cases=np.repeat(np.arange(64,dtype=np.int64),32)
    odd=np.tile(np.arange(32)%2==1,64)
    strata={'added_only':~odd,'counterexample':odd,'broad':np.ones(len(cases),bool)}
    table={'case_index':cases,'common_eligible':strata['broad'],'expanded_group_covered':np.ones(len(cases),bool),
        'region':np.where(odd,'after-observation-change','before-observation-change')}
    return records,cases,strata,table


def test_exact_bank_and_reserved_metadata_are_enumerated_without_inputs():
    records=a.bank_audit.expected_records()
    assert len(records)==1536 and len({r['frequency_hz'] for r in records})==192
    assert records[0]['frequency_hz']==55.64425 and records[-1]['frequency_hz']==698.58175
    assert a.BANK_IDENTITIES==producer.PINS
    assert a.old.fresh_records()==runner.cases.challenge_spec()['records']
    assert a.old.GROUP_COUNTS==runner.GROUPS
    assert a.ARMS==runner.ARMS and a.SEEDS==runner.SEEDS
    assert len(a.FIT_NAMES)==12 and len(a.METHODS)==17


@pytest.mark.parametrize('seed',a.SEEDS)
@pytest.mark.parametrize('epoch',[0,7,11])
def test_case_uniform_then_row_uniform_schedule_matches_frozen_design(seed,epoch):
    records,cases,strata,table=toy_schedule_inputs()
    expected=a.targeted_schedule(strata,cases,records,seed,epoch)
    actual=producer.targeted_schedule(strata,cases,records,seed,epoch)
    for k in expected:np.testing.assert_array_equal(actual[k],expected[k])
    assert expected['indices'].dtype==np.int64 and expected['origin'].dtype==np.uint8
    report=a.sampling_exposure(expected,strata,table,len(records))
    assert report==producer.schedule_exposure(actual,strata,table,len(records))
    assert report['selected_rows']==16896 and report['duplicate_draws']>0
    assert report['requested_strata']=={'added_only':4224,'counterexample':4224,'broad':8448}


@pytest.mark.parametrize('mutation',['too_few_rows','duplicate_numeric_carrier','phase_alias','seed','epoch'])
def test_schedule_refuses_broken_support_and_names(mutation):
    records,cases,strata,table=toy_schedule_inputs();seed=a.SEEDS[0];epoch=0
    if mutation=='too_few_rows':strata['added_only'][400:]=False
    elif mutation=='duplicate_numeric_carrier':records[8]['frequency_hz']=records[0]['frequency_hz']
    elif mutation=='phase_alias':records[0]['phase_radians']=records[4]['phase_radians']
    elif mutation=='seed':seed=1
    else:epoch=12
    with pytest.raises(ValueError):a.targeted_schedule(strata,cases,records,seed,epoch)


@pytest.mark.parametrize('frequency',[50.,137.25,333.123456789,700.])
@pytest.mark.parametrize('amplitude',[0.,.1,1e-8])
def test_scalar_raw_periodicity_matches_existing_kernel_on_fixed_toys(frequency,amplitude):
    t=np.arange(1920)/48000
    window=(amplitude*(np.sin(2*np.pi*193.75*t+.13)+.5*np.sin(4*np.pi*193.75*t+.4))).astype(np.float32)
    hz=np.array([frequency,100.,200.,np.nan,np.nan,np.nan,np.nan]);mask=np.isfinite(hz)
    actual=periodicity_evidence(window,hz,mask)
    saved={k:getattr(actual,'features' if k=='evidence' else k) for k in ('correlations','valid','period_samples','common_length','evidence')}
    expected=a.scalar_periodicity(window,hz,mask)
    dc,de=a.compare_scalar(saved,expected)
    assert dc<=2e-13 and de<=1e-7


@pytest.mark.parametrize('field',['correlations','evidence','valid','period_samples','common_length'])
def test_scalar_comparison_refuses_corruption(field):
    hz=np.array([200.,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]);mask=np.isfinite(hz)
    window=np.sin(np.arange(1920)*.02).astype(np.float32)
    x=a.scalar_periodicity(window,hz,mask);y={k:v.copy() for k,v in x.items()}
    if field=='valid':y[field][0]=not y[field][0]
    else:y[field].flat[0]+=1e-4 if field in ('correlations','evidence') else 1
    with pytest.raises(ValueError):a.compare_scalar(y,x)


def test_raw_sample_coordinates_are_frozen_and_cover_three_regions():
    ends=2208+480*np.arange(196);target=ends-960
    common=(target>=3072)&(target+1536<=96000)
    assert np.flatnonzero(common)[[0,76,190]].tolist()==list(a.SAMPLE_ROWS)
    assert ends[4]<=39840 and ends[80]-1920<39840<ends[80] and ends[194]-1920>=39840


@pytest.mark.parametrize('mode',['off','periodicity'])
@pytest.mark.parametrize('k',[6,7])
def test_explicit_cpu_equations_match_model_on_fixture_only(mode,k):
    torch.manual_seed(11);model=EvidenceHarmonicSelector(mode);model.fit_scaling(batch(6),split='train')
    with torch.no_grad():model.evidence_head.weight.copy_(torch.tensor([[.3,-.2,.1]]))
    b=batch(k);e=evidence(b);grouped={n:getattr(b,n).numpy() for n in ('features','candidate_hz','candidate_mask')}
    with torch.inference_mode():
        got=a.independent_logits(model.state_dict(),grouped,e.numpy(),mode)
        want=model(b,e)
    torch.testing.assert_close(got,want,rtol=0,atol=1e-7)
    p=np.array([.8,.2,.8]);s,h=a.voice_equations.emit(got,grouped['candidate_hz'],grouped['candidate_mask'],p)
    assert s.tolist()==[1,0,2] and np.isnan(h[1:]).all()
    assert a.check_saved_output(s,h,s.copy(),h.copy())==3


@pytest.mark.parametrize('field',['state','hz'])
def test_prediction_corruption_is_not_hidden_by_mean_metrics(field):
    s=np.array([1,0,2],np.uint8);h=np.array([200.,np.nan,np.nan]);t=s.copy();z=h.copy()
    if field=='state':t[0]=0
    else:z[0]=100.
    with pytest.raises(ValueError):a.check_saved_output(s,h,t,z)


def test_natural_and_analytic_gates_match_producer_toy_object_contract():
    values=cells();independent=a.old.natural_gate(values)
    natural={m:{'macro':{'joint_error':np.mean([v['joint_error'] for v in rows.values()]),
        'reference_voiced_success_rate':np.mean([v['reference_voiced_success_rate'] for v in rows.values()])},
        'per_condition':{c:{'joint_error':np.mean([v['joint_error'] for k,v in rows.items() if k.endswith('/'+c)])} for c in a.base.CONDITIONS}}
        for m,rows in values.items()}
    assert runner.natural_decision(natural)==independent
    assert a.old.analytic_gates(gates())['passed_by_seed']=={n:78 for n in a.old.CANDIDATE}
    broken=gates();broken[a.old.CANDIDATE[1]]['challenge/0']['success']=.9499
    assert not a.old.analytic_gates(broken)['pass']


def resource_fixture():
    start=datetime(2026,10,24,tzinfo=timezone.utc)
    return {'status':'complete','sources_frozen_at':start.isoformat(),'deadline_utc':(start+timedelta(hours=3)).isoformat(),
        'completed_at':(start+timedelta(minutes=5)).isoformat(),'elapsed_seconds':300.,'fit_elapsed_seconds':100.,
        'peak_observed_python_rss_bytes':1,'peak_observed_node_rss_bytes':1,'peak_observed_mps_allocated_bytes':1,'output_bytes':1}


def test_new_run_deadline_is_relative_not_expired_historical_deadline():
    assert a.validate_resources(resource_fixture())['fit_seconds']==100
    assert a.old.phase_order(*phase_fixture())


@pytest.mark.parametrize('field',['deadline_utc','completed_at','fit_elapsed_seconds','peak_observed_python_rss_bytes','output_bytes'])
def test_resource_or_deadline_violation_refuses(field):
    m=resource_fixture()
    if field=='deadline_utc':m[field]='2026-10-24T04:00:00+00:00'
    elif field=='completed_at':m[field]='2026-10-24T04:00:00+00:00'
    elif field=='fit_elapsed_seconds':m[field]=3601
    else:m[field]=32*1024**3
    with pytest.raises(ValueError):a.validate_resources(m)


def test_failed_inventory_keeps_failure_without_claiming_model_qualification(tmp_path):
    m={'status':'failed','failure':'fixture failure','fits':{},'fit_attempts':{}}
    result=a.failed_inventory(tmp_path,m)
    assert not result['qualification_complete'] and result['completed_fits']==0
    (tmp_path/'challenge').mkdir()
    with pytest.raises(ValueError):a.failed_inventory(tmp_path,m)


def test_no_fit_no_go_cannot_hide_partial_checkpoints(tmp_path):
    a.old.no_fit_artifacts(tmp_path,{'fits':{},'fit_elapsed_seconds':0})
    (tmp_path/'checkpoints').mkdir();(tmp_path/'checkpoints'/'unexpected.pt').write_bytes(b'fixture')
    with pytest.raises(ValueError):a.old.no_fit_artifacts(tmp_path,{'fits':{},'fit_elapsed_seconds':0})


def test_audit_dependency_closure_includes_unexercised_historical_score_driver():
    names=a.source_hashes()
    for name in ('audit_evidence_scores','audit_frequency_diversity','audit_harmonic_inference','audit_expanded_pool','audit_exposure_study'):
        assert 'research/voice-prosody/phenometrix_prosody/'+name+'.py' in names
    assert 'research/voice-prosody/tests/test_audit_exposure_study.py' in names


def fresh_fixture():
    spec=runner.cases.challenge_spec();source={'fixture.py':'a'*64}
    m={'challenge_spec_sha256':runner.cases.challenge_spec_sha256(),'source_sha256':source,
        'deadline_utc':'2026-10-24T03:00:00+00:00','checkpoints_frozen_at':'2026-10-24T00:05:00+00:00',
        'predictions_frozen_at':'2026-10-24T00:10:00+00:00'}
    path='research/voice-prosody/runs/fixture/checkpoint-freeze.json';digest='b'*64
    cache={'schema':'phenometrix-exposure-reserved-input-cache.v1','status':'complete','reference_content_read':False,'model_outputs_read':False,
        'spec_sha256':m['challenge_spec_sha256'],'challenge_spec_sha256':m['challenge_spec_sha256'],'source_sha256':source,
        'deadline_utc':m['deadline_utc'],'training_bank_manifest_sha256':a.BANK_IDENTITIES['manifest.json'],
        'freeze_kind':'checkpoint','freeze_path':path,'checkpoint_freeze_path':path,'freeze_sha256':digest,'checkpoint_freeze_sha256':digest,
        'started_at':'2026-10-24T00:06:00+00:00','completed_at':'2026-10-24T00:07:00+00:00',
        'records':[{'id':r['id'],'case_index':i,'parameters_sha256':a.hashlib.sha256(a.historical.canonical(r)).hexdigest(),
            'feature_frames':196,'common_eligible_frames':191} for i,r in enumerate(spec['records'])]}
    return cache,spec,m,path,digest


def test_complete_fresh_cache_schema_is_validated_without_generating_waveforms():
    assert a.validate_fresh_declaration(*fresh_fixture())


@pytest.mark.parametrize('field',['schema','source_sha256','freeze_sha256','checkpoint_freeze_path','started_at','model_outputs_read','records'])
def test_fresh_identity_or_phase_corruption_refuses(field):
    args=fresh_fixture();cache=args[0]
    if field=='source_sha256':cache[field]={'different.py':'a'*64}
    elif field=='started_at':cache[field]='2026-10-24T00:04:00+00:00'
    elif field=='model_outputs_read':cache[field]=True
    elif field=='records':cache[field]=cache[field][1:]
    else:cache[field]='wrong'
    with pytest.raises(ValueError):a.validate_fresh_declaration(*args)


class FixtureAudit:
    def bind(self,path,expected=None):
        if expected is not None:assert a.base.sha(path)==expected
        return path
    def equal(self,x,y,label):assert x==pytest.approx(y)


def checkpoint_fixture(tmp_path):
    from phenometrix_prosody.harmonic_selector_model import HarmonicSelector
    prior=tmp_path/'prior';run=tmp_path/'run';(prior/'checkpoints').mkdir(parents=True);(run/'checkpoints').mkdir(parents=True)
    previous={'fits':{}};m={'fits':{},'fit_attempts':{},'training_frozen_at':'2026-10-24T00:05:00+00:00',
        'checkpoints_frozen_at':'2026-10-24T01:00:00+00:00','deadline_utc':'2026-10-24T03:00:00+00:00','fit_elapsed_seconds':12.}
    for seed in a.SEEDS:
        torch.manual_seed(seed);model=HarmonicSelector('relative','acceptable-set');state=model.state_dict();state['scaling_fitted']=torch.tensor(True)
        name=f'expanded7__seed{seed}';p=prior/'checkpoints'/(name+'.pt');torch.save({'state_dict':state},p)
        previous['fits'][name]={'weights_sha256':a.base.sha(p)}
        state={**state,'evidence_head.weight':torch.zeros(1,3)};digest=a.hashlib.sha256()
        for k,v in sorted(state.items()):digest.update(k.encode());digest.update(v.numpy().tobytes())
        for arm in a.ARMS:
            sequence=len(m['fits']);start=datetime(2026,10,24,0,10,tzinfo=timezone.utc)+timedelta(seconds=sequence*3)
            name=f'{arm}__seed{seed}';p=run/'checkpoints'/(name+'.pt')
            torch.save({'arm':arm,'evidence_mode':'periodicity' if arm.endswith('__on') else 'off',
                'config':{'representation':'shared-candidates','loss':'acceptable-set','hidden_size':32},'state_dict':state},p)
            m['fits'][name]={'arm':arm,'seed':seed,'optimizer_updates':396,'parameter_count':3012,'initial_tensors_sha256':digest.hexdigest(),
                'weights_sha256':a.base.sha(p),'epoch_losses':[1.25]*12,'epoch_base_losses':[1.]*12,'epoch_weighted_auxiliary_losses':[.25]*12,
                'evidence_residual_weights':[[0.,0.,0.]],'elapsed_seconds':1.}
            m['fit_attempts'][name]={'status':'complete','initial_tensors_sha256':digest.hexdigest(),'started_at':start.isoformat(),'completed_at':(start+timedelta(seconds=1)).isoformat()}
    return prior,run,m,previous


def test_new_relative_deadline_checkpoint_initialization_and_scaling(tmp_path):
    prior,run,m,previous=checkpoint_fixture(tmp_path)
    weights,report=a.verify_checkpoints(FixtureAudit(),run,m,previous,prior)
    assert len(weights)==12 and report['parameters_each']==3012 and report['optimizer_updates_each']==396


@pytest.mark.parametrize('field',['updates','initial','loss','deadline','off_weight'])
def test_checkpoint_fit_record_corruption_refuses(tmp_path,field):
    prior,run,m,previous=checkpoint_fixture(tmp_path);name=a.FIT_NAMES[0]
    if field=='updates':m['fits'][name]['optimizer_updates']=397
    elif field=='initial':m['fits'][name]['initial_tensors_sha256']='0'*64
    elif field=='loss':m['fits'][name]['epoch_losses'][0]=3.
    elif field=='deadline':m['deadline_utc']='2026-10-24T00:00:00+00:00'
    else:
        p=run/'checkpoints'/(name+'.pt');saved=torch.load(p,weights_only=True);saved['state_dict']['evidence_head.weight'][0,0]=1.
        torch.save(saved,p);m['fits'][name]['weights_sha256']=a.base.sha(p)
    with pytest.raises(ValueError):a.verify_checkpoints(FixtureAudit(),run,m,previous,prior)


def test_fit_sequence_cannot_overlap_or_switch_arms(tmp_path):
    prior,run,m,previous=checkpoint_fixture(tmp_path)
    second=f'existing__on__seed{a.SEEDS[0]}'
    m['fit_attempts'][second]['started_at']=m['fit_attempts'][a.FIT_NAMES[0]]['started_at']
    with pytest.raises(ValueError,match='sequence'):a.verify_checkpoints(FixtureAudit(),run,m,previous,prior)
