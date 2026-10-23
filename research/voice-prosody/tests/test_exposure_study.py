"""Policy and numerics fixtures only; never execute scientific generation/fits."""
import copy
import numpy as np
import pytest
import torch
from phenometrix_prosody import exposure_study as study
from phenometrix_prosody import selector_experiment as old
from phenometrix_prosody import harmonic_challenge,subharmonic_challenge,augmentation_cases,coverage_cases,expanded_cases,evidence_cases
from phenometrix_prosody.selector_model import build_pitch_targets,selector_loss
from phenometrix_prosody.evidence_selector_model import EvidenceHarmonicSelector
from phenometrix_prosody.expanded_selector_model import ExpandedHarmonicSelector
from test_evidence_selector_model import batch,evidence


def test_inference_retains_exact_old_control_voicing_and_unavailable_states():
    b=batch();e=evidence(b);torch.manual_seed(5);base=ExpandedHarmonicSelector();torch.manual_seed(5);new=EvidenceHarmonicSelector('off')
    probability=np.array([.49,.5,.99]);before=probability.copy()
    expected=old.infer(base,b,probability);actual=study.infer(new,b,e,probability)
    assert np.array_equal(actual[0],expected[0]) and np.array_equal(actual[1],expected[1],equal_nan=True)
    assert actual[0].tolist()==[0,1,2] and np.array_equal(probability,before)
    with pytest.raises(ValueError,match='alignment'):study.infer(new,b,e[:2],probability)


def test_two_loss_denominators_match_explicit_covered_means_and_zero_rows():
    b=batch();e=evidence(b);t=build_pitch_targets(b,[200,125,np.nan],[True,True,False],[True]*3);model=EvidenceHarmonicSelector('periodicity')
    empty=b.index_select([2]);et=t.index_select([2]);ee=e[[2]]
    full,natural,weighted=study.loss(model,b,t,e,empty,et,ee)
    expected=selector_loss(model(b,e),b.candidate_mask,t,objective='acceptable-set')
    torch.testing.assert_close(natural,expected,rtol=0,atol=0);torch.testing.assert_close(full,natural,rtol=0,atol=0)
    assert float(weighted.detach())==0;full.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_diagnostics_include_uncovered_denominators_and_empty_strata():
    b=batch();e=evidence(b);t=build_pitch_targets(b,[200,125,np.nan],[True,True,False],[True]*3);model=EvidenceHarmonicSelector()
    result=study.diagnostics(model,b,t,e,{'broad':np.ones(3,bool),'empty':np.zeros(3,bool),'uncovered':np.array([False,False,True])})
    assert result['broad']['rows']==3 and result['broad']['covered_rows']==2
    assert result['empty']['success_all_rows'] is None and result['empty']['mean_best_acceptable_minus_best_wrong'] is None
    assert result['uncovered']['success_all_rows']==0 and result['uncovered']['covered_rows']==0 and result['uncovered']['conditional_loss']==0


def natural_fixture():
    result={b:{'macro':{'joint_error':.1,'reference_voiced_success_rate':.8},'per_condition':{c:{'joint_error':.1} for c in old.CONDITIONS}} for b in ('original_mlp','original_hybrid')}
    for a in study.ARMS:
        for s in study.SEEDS:result[study.key(a,s)]={'macro':{'joint_error':.09,'reference_voiced_success_rate':.81},'per_condition':{c:{'joint_error':.09} for c in old.CONDITIONS}}
    return result


def test_candidate_fixed_and_all_natural_gates_are_conjunctive():
    x=natural_fixture();assert study.natural_decision(x)['pass']
    for key,value in [('joint_error',.101),('reference_voiced_success_rate',.7)]:
        bad=copy.deepcopy(x);bad[study.key(study.CANDIDATE,study.SEEDS[0])]['macro'][key]=value;assert not study.natural_decision(bad)['pass']
    bad=copy.deepcopy(x)
    for s in study.SEEDS:bad[study.key(study.CANDIDATE,s)]['per_condition']['clean']['joint_error']=.111
    assert not study.natural_decision(bad)['pass']
    bad=copy.deepcopy(x)
    for s in study.SEEDS:bad[study.key(study.CANDIDATE,s)]['macro']['joint_error']=.096
    assert not study.natural_decision(bad)['pass']
    # A much better control never replaces a failing predesignated candidate.
    for s in study.SEEDS:bad[study.key('existing__on',s)]['macro']['joint_error']=0
    assert study.natural_decision(bad)['predesignated_candidate']=='targeted__on' and not study.natural_decision(bad)['pass']


def analytic_fixture():
    groups={'regression':{cid:('constant' if cid.startswith('constant') else 'observation-change' if cid.startswith('observation-change') else 'glide') for cid in old.SYNTHETIC_IDS}}
    for group,module in [('consumed_harmonic',harmonic_challenge),('consumed_subharmonic',subharmonic_challenge),('consumed_augmentation',augmentation_cases),('consumed_coverage',coverage_cases),('consumed_expansion',expanded_cases),('challenge',evidence_cases)]:groups[group]={r['id']:'observation-change' for r in module.challenge_spec()['records']}
    return {g:{cid:{'kind':kind,'methods':{study.key(study.CANDIDATE,s):{'all':{'success_within_50c_all_frames':1.},'regions':{'after-observation-change':{'success_within_50c_all_frames':1.}}} for s in study.SEEDS}} for cid,kind in rows.items()} for g,rows in groups.items()}


def test_each_seed_case_gate_and_exact_configuration_inventory_required():
    x=analytic_fixture();d=study.analytic_decision(x);assert d['pass'] and set(d['passed_by_seed'].values())=={78}
    first=next(iter(x['challenge'].values()));first['methods'][study.key(study.CANDIDATE,study.SEEDS[0])]['regions']['after-observation-change']['success_within_50c_all_frames']=.94
    assert not study.analytic_decision(x)['pass']
    x=analytic_fixture();cid=next(iter(x['challenge']));x['challenge']['wrong-config']=x['challenge'].pop(cid)
    with pytest.raises(ValueError,match='identity'):study.analytic_decision(x)
    x=analytic_fixture();x['challenge'].pop(next(iter(x['challenge'])))
    with pytest.raises(ValueError,match='inventory'):study.analytic_decision(x)


def test_completed_checkpoint_inventory_cannot_be_partial_or_changed(tmp_path):
    with pytest.raises(ValueError,match='Incomplete12'):study.check_checkpoints(tmp_path,{'fits':{}})
    (tmp_path/'checkpoints').mkdir();fits={}
    for a in study.ARMS:
        for s in study.SEEDS:
            name=study.key(a,s);p=tmp_path/'checkpoints'/(name+'.pt');p.write_bytes(b'fixture-only');fits[name]={'weights_sha256':study.ev.sha(p)}
    study.check_checkpoints(tmp_path,{'fits':fits});p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):study.check_checkpoints(tmp_path,{'fits':fits})


def test_no_overwrite_and_outside_output_refused_before_inputs(tmp_path,monkeypatch):
    runs=tmp_path/'runs';runs.mkdir();existing=runs/'existing';existing.mkdir();monkeypatch.setattr(study,'RUNS',runs)
    for path in (runs,existing,tmp_path/'outside'):
        with pytest.raises(ValueError,match='Fresh ignored'):study.run(path,tmp_path/'absent.json')


def test_terminal_resources_include_manifest_and_partial_fit_progress(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    monkeypatch.setattr(study,'ROOT',tmp_path)
    monkeypatch.setattr(study,'torch',SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda:False))))
    m={'status':'failed','fit_attempts':{'fixture':{'status':'failed'}}}
    study.fit_progress(m,'fixture',study.time.monotonic()-.1,4.,7,0)
    assert m['fit_elapsed_seconds']>=4.1 and m['fit_attempts']['fixture']['optimizer_updates']==7
    (tmp_path/'partial.bin').write_bytes(b'preserve me');study.finish_manifest(tmp_path,m)
    saved=json.loads((tmp_path/'manifest.json').read_text())
    assert saved['output_bytes']==sum(p.stat().st_size for p in tmp_path.rglob('*') if p.is_file())
    assert saved['peak_observed_python_rss_bytes']>0 and saved['terminal_free_disk_bytes']>0


def test_expired_budget_and_allocated_not_driver_memory_are_guarded(tmp_path,monkeypatch):
    from datetime import datetime,timedelta,timezone
    from types import SimpleNamespace
    m={'deadline_utc':(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()}
    with pytest.raises(TimeoutError):study.budget(tmp_path,study.time.monotonic(),m)
    m['deadline_utc']=(datetime.now(timezone.utc)+timedelta(minutes=1)).isoformat();m['_last_disk_check']=study.time.monotonic()
    mock=SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda:True)),mps=SimpleNamespace(current_allocated_memory=lambda:100,driver_allocated_memory=lambda:9*1024**3))
    monkeypatch.setattr(study,'torch',mock);assert study.budget(tmp_path,study.time.monotonic(),m)>0
    assert m['peak_observed_mps_allocated_bytes']==100 and m['peak_observed_mps_driver_allocated_bytes']==9*1024**3
    mock.mps.current_allocated_memory=lambda:8*1024**3+1
    with pytest.raises(MemoryError,match='MPS allocated'):study.budget(tmp_path,study.time.monotonic(),m)


def test_toy_receipt_requires_exact_speed_memory_and_source_bindings(tmp_path,monkeypatch):
    import json
    p=tmp_path/'toy.json';source=tmp_path/'kernel.py';source.write_text('fixture')
    value={'pass_result':True,'frames_per_second':200.,'observed_process_peak_rss_bytes':1024**3,
        'specification':{'minimum_frames_per_second':200,'maximum_observed_process_rss_bytes':1024**3},'source_sha256':{'kernel.py':study.ev.sha(source)}}
    monkeypatch.setattr(study,'REPO',tmp_path);monkeypatch.setattr(study,'TOY',p)
    def bind(m,path,expected=None):
        assert expected is None or study.ev.sha(path)==expected;return path
    monkeypatch.setattr(study,'bind',bind)
    for bad in (None,'memory','speed','nan'):
        v=copy.deepcopy(value)
        if bad=='memory':v['observed_process_peak_rss_bytes']+=1
        elif bad=='speed':v['frames_per_second']=199.99
        elif bad=='nan':v['frames_per_second']=float('nan')
        p.write_text(json.dumps(v));monkeypatch.setattr(study,'TOY_SHA',study.ev.sha(p))
        if bad:
            with pytest.raises(ValueError):study.verify_toy_receipt({})
        else:assert study.verify_toy_receipt({})['reused_without_rerun']


def complete_fixture(tmp_path,monkeypatch,*,prerequisite_pass=True):
    """Exercise the real orchestrator with fixed tiny artifacts and no fit/signal."""
    from datetime import datetime,timedelta,timezone
    import json
    from types import SimpleNamespace
    def ns(module,**overrides):return SimpleNamespace(**({k:getattr(module,k) for k in dir(module) if not k.startswith('__')}|overrides))
    repo=tmp_path;output=repo/'runs/new';previous=repo/'runs/prior'
    for p in (output/'checkpoints',output/'predictions',output/'analytic',previous/'checkpoints',previous/'predictions'):p.mkdir(parents=True)
    monkeypatch.setattr(study,'REPO',repo);monkeypatch.setattr(study,'ROOT',repo);monkeypatch.setattr(study,'PRIOR',previous)
    b=batch();e=evidence(b);t=build_pitch_targets(b,[200,125,np.nan],[True,True,False],[True]*3)
    probability=np.array([.75,.25,.75]);state=np.array([1,0,2],np.uint8);hz=np.array([200.,np.nan,np.nan]);events=[]
    feature={'features':b.features.numpy(),'candidate_hz':b.candidate_hz.numpy(),'candidate_mask':b.candidate_mask.numpy(),'pcm_sample_count':np.array(96000)}
    original_file=repo/'original.npz';np.savez_compressed(original_file,**feature)
    source=repo/'fixture.py';source.write_text('fixed source');closure={'fixture.py':study.ev.sha(source)}
    m={'source_sha256':closure,'input_sha256':{},'fits':{},'fit_attempts':{},'training_spec_sha256':'fixture-training',
        'deadline_utc':(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),'prior_manifest':{'fits':{}},'fresh_evaluation_configurations':0,'fit_elapsed_seconds':0}
    study.save(output/'source-freeze.json',{'schema':'phenometrix-exposure-source-freeze.v1','fits':{},'source_sha256':closure})
    for seed in study.SEEDS:
        name=study.key('expanded7',seed);p=previous/'checkpoints'/(name+'.pt');torch.save({'state_dict':{}},p)
        m['prior_manifest']['fits'][name]={'weights_sha256':study.ev.sha(p)}
        np.savez_compressed(previous/'predictions'/(name+'.npz'),state=state,hz=hz)
    def bind(manifest,path,expected=None):
        from pathlib import Path
        p=Path(path);h=study.ev.sha(p)
        if expected is not None and h!=expected:raise ValueError('Fixture binding changed')
        manifest['input_sha256'][str(p.relative_to(repo))]=h;return p
    monkeypatch.setattr(study,'bind',bind);monkeypatch.setattr(study,'budget',lambda *a:60.);monkeypatch.setattr(study,'check_sources',lambda expected:None)
    original={s:{'split':s,'raw':feature,'hashes':{}} for s in ('train','development')}
    def load_labels(d):
        events.append('references-'+d['split'])
        if events.count('references-development')==2:
            freeze=json.loads((output/'prediction-freeze.json').read_text())
            assert freeze['status']=='predictions_frozen' and len(freeze['prediction_sha256'])==103
            assert len(freeze['fits'])==12 and freeze['fresh_evaluation_configurations']==12
        return {'hashes':{},'excluded':{}}
    def aggregate(d,l,p):
        if len(p)>2:assert len(p)==17
        return {name:{'macro':{'joint_error':.09 if name not in ('original_mlp','original_hybrid') else .1,'reference_voiced_success_rate':.81 if name not in ('original_mlp','original_hybrid') else .8},'per_condition':{c:{'joint_error':.09 if name not in ('original_mlp','original_hybrid') else .1} for c in old.CONDITIONS}} for name in p}
    monkeypatch.setattr(study,'old',ns(study.old,load_inputs=lambda s:original[s],load_labels=load_labels,
        read_natural_baselines=lambda d:(probability.copy(),{a:(state.copy(),hz.copy()) for a in ('original_mlp','original_hybrid')},{}),
        aggregate=aggregate,targets=lambda *a:t,infer=lambda *a:(state.copy(),hz.copy())))
    monkeypatch.setattr(study,'prior',ns(study.prior,prepare=lambda raw:{'fixed-slots':b,'shared-candidates':b},
        coverage_reports=lambda *a:{'original6':{'pass':True},'expanded7':{'pass':prerequisite_pass}},training_schedule=lambda _:torch.arange(3),analytic_coverage=lambda item:{'fixture':True}))
    monkeypatch.setattr(study,'cov',ns(study.cov,oracle_headroom=lambda *a:{'fixed-slots':{'pass':True}}))
    fixture_scores=analytic_fixture();groups={}
    for group,rows in fixture_scores.items():
        if group=='challenge':continue
        items=[]
        for cid,row in rows.items():
            p=previous/(group+'-'+cid+'.npz');np.savez_compressed(p,original_probability=probability,**{study.key('expanded7',s)+'__state':state for s in study.SEEDS},**{study.key('expanded7',s)+'__hz':hz for s in study.SEEDS})
            items.append({'id':cid,'scenario':{'kind':row['kind'],'initialHz':200.},'feature':feature,'expanded_feature':feature,'original_feature_path':original_file,'pcm_sha256':'a'*64,'prior_prediction_path':p})
        groups[group]=items
    def evid(item,*args):
        item['periodicity_evidence']={'features':e.numpy(),'valid':b.candidate_mask.numpy()};return e
    monkeypatch.setattr(study,'inputs',ns(study.inputs,cache_natural=lambda *a:{'raw':feature,'evidence':e.numpy()},
        load_existing_training=lambda m:[],load_historical_analytic=lambda m:groups,evidence_for_item=evid))
    meta={'scenario_index':np.arange(3),'reference_hz':np.array([200.,125.,np.nan]),'region':np.array(['before','boundary','after'])}
    monkeypatch.setattr(study,'build_bank',lambda *a,**kw:(b,t,meta,e));monkeypatch.setattr(study,'targeted_items',lambda *a:[])
    prep={'feasibility':{'pass_result':True},'arrays':{'case_index':meta['scenario_index'],'truth_hz':meta['reference_hz'],'expanded_group_hz':b.candidate_hz.numpy()},'strata':{'broad':np.ones(3,bool)}}
    # Target reference is finite on the empty candidate fixture as in real TRAIN.
    meta['reference_hz'][2]=300.;prep['arrays']['truth_hz']=meta['reference_hz']
    schedules={f'seed{s}__epoch{ep}__{field}':np.zeros((33,512),np.int64 if field=='indices' else np.uint8) for s in study.SEEDS for ep in range(12) for field in ('indices','origin')}
    def fit(*args):
        events.append('fit')
        assert events==['references-train','references-development','fit']
        assert (output/'training-freeze.json').is_file() and (output/'targeted-sampling-schedules.npz').is_file()
        for s in study.SEEDS:
            for a in study.ARMS:
                name=study.key(a,s);p=output/'checkpoints'/(name+'.pt');p.write_bytes(name.encode());m['fits'][name]={'weights_sha256':study.ev.sha(p)}
        return {n:object() for n in m['fits']}
    monkeypatch.setattr(study,'fit_all',fit)
    def fresh(out,freeze_path,digest,guard):
        events.append('fresh');freeze=json.loads(freeze_path.read_text())
        assert freeze['schema']=='phenometrix-exposure-checkpoint-freeze.v1' and len(freeze['checkpoints'])==12
        assert study.ev.sha(freeze_path)==digest and events[-2]=='fit'
        out.mkdir();rows=[];items=[]
        for record in study.cases.challenge_spec()['records']:
            cid=record['id'];raw=out/(cid+'.f32le');raw.write_bytes(b'fixed toy PCM');fp=out/(cid+'.npz');np.savez_compressed(fp,**feature)
            rows.append({'id':cid,'pcm_path':raw.name,'pcm_sha256':study.ev.sha(raw),'feature_path':fp.name})
            items.append({'id':cid,'scenario':{'kind':'observation-change','initialHz':record['frequency_hz']},'feature':feature})
        study.save(out/'manifest.json',{'records':rows});return items,{}
    monkeypatch.setattr(study,'bank_adapter',ns(study.bank_adapter,make_schedules=lambda p:(schedules,{}),generate_challenge=fresh))
    class Worker:
        peak_rss_bytes=1
        def __init__(self,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
    monkeypatch.setattr(study,'CoveragePeakWorker',Worker);monkeypatch.setattr(study,'acquisition',ns(study.acquisition,expanded_feature=lambda *a:(feature,{})))
    class FrozenModel:
        def __init__(self,*a):pass
        def load_state_dict(self,*a):pass
        def eval(self):return self
    monkeypatch.setattr(study,'base',ns(study.base,ExpandedHarmonicSelector=FrozenModel))
    monkeypatch.setattr(study,'infer',lambda *a:(state.copy(),hz.copy()))
    def analytic_scores(preds,items):
        assert all(len(p)==15 for p in preds.values())
        result={}
        for item in items:
            row=copy.deepcopy(next(g[item['id']] for g in fixture_scores.values() if item['id'] in g))
            metric=next(iter(row['methods'].values()));row['methods']={n:copy.deepcopy(metric) for n in preds[item['id']]};result[item['id']]=row
        return result
    monkeypatch.setattr(study,'harmonic',ns(study.harmonic,original_probability=lambda f:probability.copy(),analytic_scores=analytic_scores))
    monkeypatch.setattr(study,'torch',ns(torch,backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda:False))))
    return output,m,prep,events


def test_complete_postfit_path_saves_all86_cases17_methods_before_final_reference_read(tmp_path,monkeypatch):
    import json
    output,m,prep,events=complete_fixture(tmp_path,monkeypatch)
    result=study.train_and_evaluate(output,m,prep,study.time.monotonic())
    assert events==['references-train','references-development','fit','fresh','references-development']
    assert m['status']=='complete' and len(m['fits'])==12 and m['fresh_evaluation_configurations']==12
    assert result['schema']=='phenometrix-exposure-metrics.v1' and result['decision']['analytic']['required_per_seed']==78
    assert result['decision']['promising_for_next_study'] and len(result['natural'])==17
    inventory=json.loads((output/'analytic-inventory.json').read_text())
    assert {k:len(v) for k,v in inventory.items()}==study.GROUPS
    freeze=json.loads((output/'prediction-freeze.json').read_text());assert not freeze.get('development_references_read_for_final_scoring',False)
    assert len(freeze['prediction_sha256'])==103 and m['control_reproduction']['mismatches']==0
    assert m['output_bytes']==sum(p.stat().st_size for p in output.rglob('*') if p.is_file())


def test_prerequisite_no_go_never_fits_generates_or_reads_final_references(tmp_path,monkeypatch):
    output,m,prep,events=complete_fixture(tmp_path,monkeypatch,prerequisite_pass=False)
    result=study.train_and_evaluate(output,m,prep,study.time.monotonic())
    assert not result['pass'] and m['status']=='prerequisite_no_go'
    assert events==['references-train','references-development'] and m['fits']=={} and m['fresh_evaluation_configurations']==0
    assert not (output/'checkpoint-freeze.json').exists() and m['output_bytes']>0


def test_interrupted_first_fit_retains_attempt_without_creating_checkpoint(tmp_path,monkeypatch):
    from types import SimpleNamespace
    def ns(module,**overrides):return SimpleNamespace(**({k:getattr(module,k) for k in dir(module) if not k.startswith('__')}|overrides))
    class FixtureBatch:
        def index_select(self,*a):return self
        def to(self,*a):return self
    class DeviceFailure:
        def fit_scaling(self,*a,**kw):pass
        def to(self,*a):raise RuntimeError('fixed device-allocation failure')
    b=FixtureBatch();prepared={'expanded7':{'shared-candidates':b},'original6':{'shared-candidates':b},'evidence':np.zeros((1,7,3),np.float32)}
    banks={a:(b,b,{},torch.zeros(1,7,3)) for a in ('existing','targeted')}
    # No actual device is used: tensor.to is a fixture-only identity here.
    monkeypatch.setattr(torch.Tensor,'to',lambda self,*a,**kw:self)
    monkeypatch.setattr(study,'modeling',SimpleNamespace(EvidenceHarmonicSelector=lambda mode:DeviceFailure()))
    monkeypatch.setattr(study,'prior',ns(study.prior,state_digest=lambda model:'fixture-initial'))
    monkeypatch.setattr(study,'torch',ns(torch,mps=SimpleNamespace(manual_seed=lambda seed:None)))
    monkeypatch.setattr(study,'check_sources',lambda _:None);monkeypatch.setattr(study,'remaining',lambda _:100.)
    monkeypatch.setattr(study,'diagnostics',lambda *a:{})
    m={'fits':{},'source_sha256':{}}
    with pytest.raises(RuntimeError,match='device-allocation'):
        study.fit_all(tmp_path,m,prepared,b,torch.tensor([0]),banks,{}, {},{},lambda:100.)
    attempt=m['fit_attempts'][study.key(study.ARMS[0],study.SEEDS[0])]
    assert attempt['status']=='failed' and attempt['optimizer_updates']==0 and attempt['completed_epochs']==0
    assert attempt['elapsed_seconds']>=0 and m['fit_elapsed_seconds']==attempt['elapsed_seconds'] and m['fits']=={}
    assert not (tmp_path/'checkpoints').exists()


@pytest.mark.parametrize('changed',['source','input'])
def test_prerequisite_no_go_still_rechecks_final_identities(tmp_path,monkeypatch,changed):
    from types import SimpleNamespace
    output,m,prep,events=complete_fixture(tmp_path,monkeypatch,prerequisite_pass=False)
    if changed=='source':
        def reject(_):raise ValueError('Frozen source changed fixture')
        monkeypatch.setattr(study,'check_sources',reject)
    else:
        p=tmp_path/'bound-input.txt';p.write_text('original');m['input_sha256'][p.name]=study.ev.sha(p)
        previous=study.cov.oracle_headroom
        def changed_after_read(*args):p.write_text('changed');return previous(*args)
        monkeypatch.setattr(study,'cov',SimpleNamespace(oracle_headroom=changed_after_read))
    with pytest.raises(ValueError,match='changed'):study.train_and_evaluate(output,m,prep,study.time.monotonic())
    assert m.get('status')!='prerequisite_no_go' and m['fits']=={} and 'fresh' not in events
