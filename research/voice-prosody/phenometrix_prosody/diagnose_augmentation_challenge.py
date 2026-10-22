"""Postfit analytic coverage/state diagnostic for paired augmentation.

No fitting, inference, natural references, new labels or waveform generation.
All fifty analytic cases and all twelve saved selectors remain in the report.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .selector_model import prepare_selector_batch
from .evaluate_synthetic import analytic_frequency, OBSERVATION_SCENARIOS
from . import diagnose_subharmonic_challenge as previous_diagnostic
from .diagnose_subharmonic_challenge import require, sha, support_region, coverage, region_counts

REPO=Path(__file__).resolve().parents[3]
RUNS=REPO/'research/voice-prosody/runs'
SEEDS=(20261022,20261023,20261024)
MODELS={f'{arm}__seed{seed}' for arm in ('natural-only','stationary-aug','balance-aug') for seed in SEEDS}
MODELS|={f'prior__relative__acceptable-set__seed{seed}' for seed in SEEDS}
ALIAS_FACTORS={'quarter':.25,'third':1/3,'half':.5,'double':2.,'triple':3.,'quadruple':4.}


def diagnostic_regions(include,raw_covered,grouped_covered,any_candidate,probability,truth,predictions):
    result=region_counts(include,raw_covered,grouped_covered,any_candidate,probability,truth,predictions)
    for name,(state,hz) in predictions.items():
        voiced=(state==1);error=np.full(len(state),np.inf)
        error[voiced]=1200*np.log2(hz[voiced]/truth[voiced])
        wrong=include&voiced&(np.abs(error)>50)
        result['methods'][name]['incorrect_pitch_summary']=None if not wrong.any() else {
            'min_hz':float(hz[wrong].min()),'median_hz':float(np.median(hz[wrong])),
            'max_hz':float(hz[wrong].max()),'median_signed_cents':float(np.median(error[wrong]))}
        result['methods'][name]['incorrect_alias_counts']={alias:int((np.abs(error[wrong]-1200*np.log2(factor))<=50).sum())
                                                        for alias,factor in ALIAS_FACTORS.items()}
    return result


def diagnose(run):
    run=Path(run).resolve();require(run.is_relative_to(RUNS),'Expected ignored local run')
    bindings={}
    def bind(path,expected=None):
        path=Path(path).resolve();require(path.is_relative_to(REPO),'Evidence escaped repository')
        h=sha(path);require(expected is None or h==expected,'Evidence identity changed: '+str(path.relative_to(REPO)))
        bindings[str(path.relative_to(REPO))]=h;return path
    def read(path,expected=None):return json.loads(bind(path,expected).read_text())
    m=read(run/'manifest.json');frozen=read(run/'prediction-freeze.json');ck=read(run/'checkpoint-freeze.json')
    require(m['schema']=='phenometrix-augmentation-experiment.v1' and m['status']=='complete','Wait for complete experiment')
    require(frozen['status']=='predictions_frozen' and ck['status']=='checkpoints_frozen','Required freezes absent')
    require(m['fits']==frozen['fits']==ck['fits'] and set(m['fits'])=={n for n in MODELS if not n.startswith('prior__')},'Checkpoint inventory differs')
    require(m['source_sha256']==frozen['source_sha256']==ck['source_sha256'],'Source phases differ')
    for name,h in frozen['source_sha256'].items():bind(REPO/name,h);bind(run/'source-snapshot'/name,h)
    for name,fit in m['fits'].items():bind(run/'checkpoints'/(name+'.pt'),fit['weights_sha256'])
    inputs=frozen['input_sha256']
    def input_path(path):
        path=Path(path).resolve();require(str(path) in inputs,'Input absent from prediction freeze')
        return bind(path,inputs[str(path)])
    def load_npz(path):
        with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    original_weight=input_path(RUNS/'delayed-mlp-seed20261022/weights.pt')
    require(sha(original_weight)==frozen['frozen_original_weights_sha256']=='06110469a9a643fb16806905569ebc939db2c0bdd5b3a6a1593b31789453d10d','Original voice identity differs')
    for seed in SEEDS:input_path(RUNS/f'harmonic-context-v1/checkpoints/relative__acceptable-set__seed{seed}.pt')
    metrics=read(run/'metrics.json',m['metrics_sha256'])
    support=json.loads(input_path(RUNS/'support-diagnostic-v1/manifest.json').read_text())
    scenarios={r['id']:r for r in [*support['scenarios'],*OBSERVATION_SCENARIOS]}
    require(len(scenarios)==14,'Regression scenario inventory differs')
    groups={};compared=0
    domains=[('regression',RUNS/'delayed-known-truth',None),
             ('consumed_challenge',RUNS/'harmonic-context-v1/challenge','harmonic-balance'),
             ('consumed_subharmonic',RUNS/'subharmonic-objective-v1/challenge','subharmonic-balance'),
             ('challenge',run/'challenge','augmentation-balance')]
    for group,folder,prefix in domains:
        records=None
        if prefix:
            cm=json.loads(input_path(folder/'manifest.json').read_text());spec=json.loads(input_path(folder/'spec.json').read_text())
            digest=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            require(cm['status']=='complete' and cm['spec_sha256']==digest,'Challenge spec/cache differs')
            expected={f'{prefix}-f{i}-p{j}' for i in range(6) for j in range(2)}
            require({r['id'] for r in spec['records']}=={r['id'] for r in cm['records']}==expected,'Challenge inventory differs')
            records={r['id']:r for r in cm['records']}
            spec_records={r['id']:r for r in spec['records']}
            case_scenarios={r['id']:{'id':r['id'],'kind':'observation-change','initialHz':r['frequency_hz'],'finalHz':r['frequency_hz'],
                'changeStartSeconds':r['change_sample']/r['sample_rate_hz'],'changeEndSeconds':r['change_sample']/r['sample_rate_hz'],
                'change_sample':r['change_sample'],'slopeHzPerSecond':0} for r in spec['records']}
        else:case_scenarios=scenarios
        cases={}
        require(set(metrics[group])==set(case_scenarios),'Scored case inventory differs')
        for identity,scenario in case_scenarios.items():
            if records:
                r=records[identity];input_path(folder/r['pcm_path']);fp=input_path(folder/r['feature_path'])
                require(sha(fp)==r['feature_sha256'],'Feature/cache hash differs')
            else:fp=input_path(folder/'features'/(identity+'.npz'))
            feature=load_npz(fp);end=feature['reference_sample'];n=int(feature['pcm_sample_count'])
            require(int(feature['sample_rate_hz'])==48000 and np.array_equal(end,np.arange(2208,n+1,480)),'Acquisition timing differs')
            reference=end-960;eligible=(reference>=3072)&(reference+1536<=n)
            truth=np.array([analytic_frequency(scenario,int(t)/48000) for t in reference],np.float64)
            labels=np.array([support_region(scenario,e) for e in end])
            batch=prepare_selector_batch(feature['features'],feature['candidate_hz'],feature['candidate_mask'],representation='shared-candidates')
            raw_cov=coverage(feature['candidate_hz'],feature['candidate_mask'],truth)
            grouped_cov=coverage(batch.candidate_hz.numpy(),batch.candidate_mask.numpy(),truth)
            any_candidate=batch.candidate_mask.numpy().any(axis=-1)
            relative='predictions/'+group+'-'+identity+'.npz';pred=load_npz(bind(run/relative,frozen['prediction_sha256'][relative]))
            require(set(pred)=={'original_probability'}|{name+'__'+field for name in MODELS for field in ('state','hz')},'Analytic output inventory differs')
            p=pred['original_probability'];require(p.shape==end.shape and np.isfinite(p).all() and np.all((p>=0)&(p<=1)),'Invalid frozen probability')
            methods={name:(pred[name+'__state'],pred[name+'__hz']) for name in sorted(MODELS)}
            for state,hz in methods.values():
                members=(batch.candidate_mask.numpy()&(batch.candidate_hz.numpy()==hz[:,None])).any(axis=-1)
                require(members[state==1].all(),'Saved emitted pitch not a grouped original member')
            masks={'all':eligible,**{label:eligible&(labels==label) for label in sorted(set(labels[eligible]))}}
            regions={region:diagnostic_regions(include,raw_cov,grouped_cov,any_candidate,p,truth,methods) for region,include in masks.items()}
            for region,result in regions.items():
                for method,out in result['methods'].items():
                    old=metrics[group][identity]['methods'][method]
                    old=old['all'] if region=='all' else old['regions'][region]
                    require(old['frames']==result['frames'] and old['voiced_outputs']==out['correct_pitch']+out['incorrect_pitch']
                            and old['nonvoiced_or_unavailable']==out['unvoiced']+out['unavailable']
                            and old['success_within_50c_all_frames']==out['success_within_50c_all_frames'],'Frozen metric/state decomposition differs')
                    compared+=1
            cov=metrics[group][identity]['candidate_coverage']
            require(cov['eligible_frames']==regions['all']['frames'] and cov['covered_frames']==regions['all']['grouped_correct_candidate_frames'],'Frozen grouped coverage differs')
            cases[identity]={'kind':scenario['kind'],'acquired_frames':len(end),'common_eligible_frames':int(eligible.sum()),
                'excluded_common_boundary_frames':int((~eligible).sum()),
                'original_probability_sha256_float64':hashlib.sha256(p.astype('<f8').tobytes()).hexdigest(),'regions':regions}
            if prefix:
                cases[identity]['frequency_hz']=spec_records[identity]['frequency_hz'];cases[identity]['phase_radians']=spec_records[identity]['phase_radians']
        groups[group]=cases
    require({g:len(v) for g,v in groups.items()}=={'regression':14,'consumed_challenge':12,'consumed_subharmonic':12,'challenge':12},'Incomplete fifty-case inventory')
    bind(Path(__file__));bind(Path(previous_diagnostic.__file__))
    for name,h in bindings.items():require(sha(REPO/name)==h,'Input changed during diagnostic')
    return {'schema':'phenometrix-augmentation-analytic-diagnostic.v1','status':'complete',
        'scope':'Postfit descriptive raw/grouped candidate coverage and all saved selector outcomes; no extra gate, fit, inference or waveform',
        'run_path':str(run.relative_to(REPO)),'natural_reference_media_or_target_arrays_read':False,
        'analytic_truth':'Frozen analytic scenarios and unchanged analytic_frequency function; used only for postfit scoring',
        'support_contract':{'sample_rate_hz':48000,'support_samples':1920,'end_grid':'2208+480*k','reference':'end-960',
            'eligible':'reference>=3072 and reference+1536<=N','observation_boundary':'new/consumed: end<=change before, end-1920>=change after; original convention strict end<change before',
            'denominator':'every common-eligible frame in whole case or reported region; no voiced/candidate-coverage filtering',
            'correctness':'inclusive50c against analytic F0 at declared reference'},
        'case_counts':{g:len(cases) for g,cases in groups.items()},'model_count':len(MODELS),
        'compared_case_region_model_scores':compared,'all_reported_metrics_agree':True,'groups':groups,'bound_sha256':bindings}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require(not args.output.exists(),'Refusing overwrite');result=diagnose(args.run)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(args.output),'sha256':sha(args.output),'case_counts':result['case_counts'],
                      'model_count':result['model_count'],'score_comparisons':result['compared_case_region_model_scores']},indent=2))
