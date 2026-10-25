import copy
import numpy as np
from phenometrix_prosody.voicing_robustness_evaluate import forward,event_measure,producer_gates,SPEAKERS,NAMES
from phenometrix_prosody.model import VoiceCandidateModel,ModelConfig
from phenometrix_prosody.metrics import Prediction
from phenometrix_prosody.audit_voicing_robustness import development_gates,verify_numerical_output


def test_actual_model_all_outputs_independent_equations():
    model=VoiceCandidateModel(ModelConfig(kind='framewise-mlp')).eval()
    model.scaling_fitted.fill_(True)
    rng=np.random.default_rng(30)
    feature={'features':rng.normal(size=(8,57)).astype(np.float32),'candidate_mask':rng.random((8,6))>.5,
             'reference_sample':np.arange(8)*480+2208,'pcm_sample_count':7000,'sample_rate_hz':48000}
    result=forward(model,feature,np.array([50,700,np.nan,701,49,180,360,90.]))
    weights={k:v.detach().numpy() for k,v in model.state_dict().items()}
    verify_numerical_output(result,weights,feature)


def ideal_events():
    return [Prediction(t,'voiced' if 48000<=t<96000 else 'unvoiced',180. if 48000<=t<96000 else None) for t in range(1248,144000,480)]


def test_events_target_availability_missing_and_separate_plateaux():
    rows=ideal_events();m=event_measure(rows,180)
    assert m['events']['onset']['target_lag_ms']==6
    assert m['events']['onset']['earliest_availability_delay_ms']==26
    assert m['plateau_pass']
    pre=[i for i,r in enumerate(rows) if 9600<=r.sample<=38400]
    for i in pre[:5]: rows[i]=Prediction(rows[i].sample,'voiced',180.)
    assert not event_measure(rows,180)['plateau_pass'] # pooled pre/post would hide >5% in pre
    rows=[Prediction(r.sample,'unavailable',None) if r.sample>=96000 else r for r in ideal_events()]
    assert not event_measure(rows,180)['events']['offset']['detected']


def fixtures():
    models={}
    for name in NAMES:
        treatment=name.startswith('R-')
        cell=dict(voicing_false_negative_or_unavailable_rate=.1 if treatment else .2,
                  voicing_false_positive_rate=.01,joint_error=.1 if treatment else .2,reference_voiced_success_rate=.9 if treatment else .8)
        models[name]={'cells':{s+'/'+c:dict(cell) for s in SPEAKERS for c in ('clean','noise10db','first_difference','clip01')}}
    analytic={n:{'status':'complete','pass':True} for n in NAMES}
    return models,analytic


def test_producer_auditor_gate_agreement_and_falsifiers():
    for mutation in ('none','FP','clean','M05','analytic','zero'):
        m,a=fixtures()
        if mutation=='FP':m['R-20261025']['cells']['F01/noise10db']['voicing_false_positive_rate']=.04
        if mutation=='clean':m['R-20261026']['cells']['F01/clean']['joint_error']=1.
        if mutation=='M05':m['R-20261027']['cells']['M05/first_difference']['voicing_false_negative_or_unavailable_rate']=.6
        if mutation=='analytic':a['R-20261025']['pass']=False
        if mutation=='zero':
            for c in m['hybrid_original']['cells'].values():c['voicing_false_negative_or_unavailable_rate']=0.
        p=producer_gates(m,a);q=development_gates(m,a)
        assert p['checks']==q['checks'] and p['status']==q['status'] and p['effects']==q['effects']
        assert p['pass']==(mutation=='none')


def test_prediction_phase_freezes_module_source_paths(tmp_path, monkeypatch):
    from phenometrix_prosody import voicing_robustness_evaluate as evaluator
    monkeypatch.setattr(evaluator, 'models', lambda base: ({}, {}))
    monkeypatch.setattr(evaluator, 'primary_inputs', lambda: [])
    # Exercise the actual prediction-phase source freeze without corpus or fits.
    monkeypatch.setattr(evaluator.study, 'file_record', lambda path: {'path':str(path), 'sha256':'synthetic-binding'})
    result = evaluator.predict(tmp_path, 'development')
    assert result['status'] == 'predictions_frozen'
    source = 'research/voice-prosody/phenometrix_prosody/voicing_robustness_evaluate.py'
    assert (tmp_path/'development/source-snapshot'/source).is_file()
