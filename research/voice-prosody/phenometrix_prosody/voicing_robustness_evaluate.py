"""Frozen input-only predictions then consumed-development/regression scoring."""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import json
import math
import numpy as np
import torch
from . import evaluate as ev
from . import evaluate_synthetic as known
from . import evaluate_curriculum as curriculum
from . import synthetic_curriculum as synthesis
from . import voicing_robustness as study
from . import voicing_robustness_inputs as vi
from .model import VoiceCandidateModel,ModelConfig
from .measurement_alignment import align_feature_timing
from .metrics import Reference,Prediction,score
from .pilot import REPO,RESEARCH,references_from_array,write_json,digest

NAMES=['hybrid_original']+[f'{a}-{s}' for s in vi.SEEDS for a in vi.ARMS]
SPEAKERS=('F01','F03','F05','F06','M01','M03','M05','M06')
RATES=('joint_error','voicing_false_positive_rate','voicing_false_negative_rate','unavailable_reference_voiced_rate',
       'voicing_false_negative_or_unavailable_rate','reference_voiced_success_rate','voiced_pitch_error_rate',
       'measurement_coverage','jointly_voiced_coverage','median_jointly_voiced_cents','p95_jointly_voiced_cents','octave_error_rate_reference_voiced')


def require(ok,msg):study.require(ok,msg)
def load(path):return study.load(path)
def bind(path,expected):require(digest(path)==expected,'Input identity changed: '+str(path));return path


def models(base):
    fitted=load(base/'fits/manifest.json');require(fitted['status']=='complete' and set(fitted['fits'])==set(NAMES[1:]),'Exactly six completed fits required')
    result={};identities={}
    for name in NAMES:
        folder=study.ORIGINAL if name=='hybrid_original' else base/'fits'/name
        m=load(folder/'manifest.json');bind(folder/'weights.pt',m['weights_sha256'])
        if name!='hybrid_original':study.verify_file(fitted['fits'][name])
        saved=torch.load(folder/'weights.pt',map_location='cpu',weights_only=True)
        model=VoiceCandidateModel(ModelConfig(**saved['config']));model.load_state_dict(saved['state_dict'],strict=True);model.eval()
        result[name]=model;identities[name]=dict(manifest=study.file_record(folder/'manifest.json'),checkpoint=study.file_record(folder/'weights.pt'))
    return result,identities


def primary_inputs():
    acq=load(study.DATA/'manifest.json');refs={r['recording_id']:r for r in acq['files'] if r['role']=='ref'}
    result=[]
    for folder,cohort in [(study.ROOT/'delayed-development-features','original_development'),(study.ROOT/'hybrid-validation-v1','formerly_reserved')]:
        manifest=load(folder/'manifest.json')
        for row in manifest['recordings']:
            identity=row['recording_id'];source=row['source_recording_id'];reference=refs[source]
            require(reference['speaker'] in SPEAKERS and reference['split']!='train','Non-TRAIN development membership differs')
            fp=folder/'features'/(identity+'.npz');bind(fp,row['feature_sha256'])
            pp=folder/('predictions/'+identity+('.json' if cohort=='original_development' else '.json.gz'));bind(pp,row['prediction_sha256'])
            original_folder=study.ROOT/'hybrid-natural-v1' if cohort=='original_development' else folder
            om=load(original_folder/'manifest.json');original=next(r for r in om['recordings'] if r['recording_id']==identity)
            op=original_folder/(original['hybrid_prediction_path'] if cohort=='original_development' else original['prediction_path'])
            bind(op,original['hybrid_prediction_sha256'] if cohort=='original_development' else original['prediction_sha256'])
            result.append(dict(id=identity,source_id=source,speaker=row['speaker'],condition=row['condition'],cohort=cohort,sample_count=row['sample_count'],
                feature=study.file_record(fp),baseline=study.file_record(pp),original_predictions=study.file_record(op),
                reference=dict(path=study.rel(study.DATA/reference['path']),sha256=reference['sha256'])))
    require(len(result)==960 and len({r['source_id'] for r in result})==240,'Development source inventory differs')
    return result


def analytic_inputs():
    folder=study.ROOT/'hybrid-synthetic-v1';manifest=load(folder/'manifest.json')
    prior=load(study.ROOT/'synthetic-curriculum-plan/challenge-plan.json')
    new=load(study.ROOT/'hybrid-followup-plan-v1/plan.json')
    support=load(study.ROOT/'support-diagnostic-v1/manifest.json')
    parameters={r['id']:r for r in prior['records']+new['fresh_challenge_records']}
    scenarios={r['id']:r for r in support['scenarios']};scenarios.update({r['id']:r for r in known.OBSERVATION_SCENARIOS})
    result=[]
    for row in manifest['records']:
        identity=row['id'];fp=folder/'features'/(identity+'.npz');pp=folder/'predictions'/(identity+'.json.gz')
        bind(fp,row['feature_sha256']);bind(pp,row['prediction_sha256'])
        definition=scenarios[identity] if row['cohort']=='original14' else parameters[identity]
        require(synthesis.parameter_digest(definition)==row['parameters_sha256'],'Analytic specification changed')
        result.append(dict(id=identity,cohort=row['cohort'],sample_count=row['pcm_sample_count'],definition=definition,
            feature=study.file_record(fp),baseline=study.file_record(pp)))
    require(len(result)==78,'Original hybrid regression inventory differs')
    return result


def forward(model,feature,raw_yin):
    x=torch.from_numpy(feature['features']);mask=torch.from_numpy(feature['candidate_mask']);n=len(x)
    with torch.no_grad():v,c,s=model(x,mask,torch.zeros((n,32)))
    v,c,s=(a.numpy() for a in (v,c,s));require(all(np.isfinite(a).all() for a in (v,c,s)),'Nonfinite inference')
    # Sigmoid at 0.5 is exactly logit>=0. No fitted probability calibration.
    valid=np.isfinite(raw_yin)&(raw_yin>=50)&(raw_yin<=700)
    status=np.where(v[:,0]<0,0,np.where(valid,1,2)).astype(np.uint8)
    return dict(target_sample=align_feature_timing(feature,measurement_delay_ms=20).reference_sample,
        voicing_logit=v,candidate_logits=c,state=s,raw_yin_hz=raw_yin,status=status,hz=np.where(status==1,raw_yin,np.nan))


def predict(base,group):
    out=base/group;out.mkdir(exist_ok=False);(out/'predictions').mkdir()
    torch.set_num_threads(1);loaded,identities=models(base)
    if group=='development':rows=primary_inputs()
    elif group=='analytic':rows=analytic_inputs()
    else:
        src=base/'diagnostic-inputs/manifest.json';prepared=load(src);require(prepared['status']=='complete','Complete diagnostic/event inputs required')
        rows=[r for r in prepared['records'] if (r['group']=='diagnostic' if group=='diagnostics' else r['group']=='events')]
    sources=study.scientific_sources()
    for module in (__file__,known.__file__,curriculum.__file__,synthesis.__file__):sources[study.rel(module)]=digest(Path(module))
    study.snapshot(out,sources)
    manifest=dict(schema='phenometrix-voicing-robustness-evaluation.v1',group=group,status='predicting',started_at=study.now(),
        source_sha256=sources,models=identities,plan=study.file_record(base/'plan/plan.json'),records=[],
        reference_values_read_before_prediction_freeze=False,role='consumed development/regression; never independent held-out PTDB')
    write_json(out/'manifest.json',manifest)
    try:
        for idx,row in enumerate(rows):
            feature=study.load_feature(study.verify_file(row['feature']));baseline=ev.read_predictions(study.verify_file(row['baseline'])) if row['baseline']['path'].endswith('.gz') else load(study.verify_file(row['baseline']))
            yin=np.array([r['raw_f0_hz'] for r in baseline['yin_evidence']],np.float64)
            require(np.array_equal([r['sample'] for r in baseline['yin_evidence']],feature['reference_sample']),'Raw YIN support mismatch')
            saved={}
            for name,model in loaded.items():
                payload=forward(model,feature,yin);path=out/'predictions'/f'{row["id"]}__{name}.npz';np.savez_compressed(path,**payload);saved[name]=study.file_record(path)
                if name=='hybrid_original' and 'original_predictions' in row:
                    old=ev.read_predictions(study.verify_file(row['original_predictions']))['predictions']['hybrid_original']
                    require([r['state'] for r in old]==[['unvoiced','voiced','unavailable'][int(s)] for s in payload['status']], 'Reconstructed frozen hybrid states differ')
            manifest['records'].append(dict(**row,predictions=saved))
            if idx%40==0:write_json(out/'manifest.json',manifest);print(json.dumps(dict(phase='predict',group=group,records=idx+1,total=len(rows))),flush=True)
        study.check_sources(sources);manifest.update(status='predictions_frozen',predictions_frozen_at=study.now())
        write_json(out/'prediction-freeze.json',manifest);write_json(out/'manifest.json',manifest);return manifest
    except BaseException as error:
        manifest.update(status='failed',failure=repr(error),failed_at=study.now());write_json(out/'manifest.json',manifest);raise


def prediction_rows(saved):
    return [Prediction(int(t),('unvoiced','voiced','unavailable')[int(s)],float(hz) if s==1 else None)
            for t,s,hz in zip(saved['target_sample'],saved['status'],saved['hz'])]


def enrich(metric):
    nv=metric['reference_voiced_frames'];metric['unavailable_reference_voiced_rate']=metric['unavailable_reference_voiced_count']/nv if nv else None
    metric['voiced_pitch_error_count']=metric['jointly_voiced_frames']-metric['reference_voiced_success_count']
    metric['voiced_pitch_error_rate']=metric['voiced_pitch_error_count']/nv if nv else None
    return metric


def aggregate(groups):
    cells={key:enrich(ev.combine_scores(items)) for key,items in groups.items()}
    def macro(keys):
        chosen=[cells[k] for k in keys]
        return dict(cell_count=len(chosen),defined_cells={k:sum(r[k] is not None for r in chosen) for k in RATES},
            **{k:float(np.mean([r[k] for r in chosen])) if chosen and all(r[k] is not None for r in chosen) else None for k in RATES})
    return dict(cells=cells,overall=macro(cells),clean=macro([k for k in cells if k.endswith('/clean')]),
        shifted=macro([k for k in cells if k.split('/')[1] in vi.ANCHORS[1:]]),
        original_development=macro([k for k in cells if k.split('/')[0] in ('F01','F03','M01','M03')]),
        formerly_reserved=macro([k for k in cells if k.split('/')[0] in ('F05','F06','M05','M06')]))


def score_natural(base,group):
    out=base/group;manifest=load(out/'prediction-freeze.json');require(manifest['status']=='predictions_frozen','Prediction freeze required')
    results={};records=[]
    try:
        for method in NAMES+(['dsp','yin','continuity','mlp_original'] if group=='development' else []):
            grouped=defaultdict(list)
            for row in manifest['records']:
                vals=np.loadtxt(study.verify_file(row['reference']))
                refs=[replace(r,eligible=r.sample>=3072 and r.sample+1536<=row['sample_count']) for r in references_from_array(vals,row['sample_count'])]
                if method in NAMES:
                    saved=study.load_feature(study.verify_file(row['predictions'][method]));preds=prediction_rows(saved)
                else:
                    old=ev.read_predictions(study.verify_file(row['original_predictions']));preds=ev.prediction_objects(old['predictions'][method])
                targets={r.sample for r in refs};extra=sum(p.sample not in targets for p in preds);preds=[p for p in preds if p.sample in targets]
                metrics=enrich(score(refs,preds));metrics['predictions_outside_reference_extent']=extra
                records.append(dict(id=row['id'],speaker=row['speaker'],condition=row['condition'],model=method,metrics=metrics))
                grouped[row['speaker']+'/'+row['condition']].append((refs,preds,row['sample_count']))
            results[method]=aggregate(grouped)
        result=dict(schema='phenometrix-voicing-robustness-metrics.v1',status='complete',group=group,models=results,records=records,
            prediction_freeze=study.file_record(out/'prediction-freeze.json'),denominator='all eligible valid references; missing output unavailable; outside 50-700Hz retained',
            aggregation='pool utterances within speaker/condition then equal cells; never average predictions across seeds')
        write_json(out/'metrics.json',result);manifest.update(status='complete',metrics=study.file_record(out/'metrics.json'),finished_at=study.now());write_json(out/'manifest.json',manifest);return result
    except BaseException as error:
        manifest.update(status='failed',failure=repr(error),failed_at=study.now());write_json(out/'manifest.json',manifest);raise


def event_measure(rows,frequency):
    by_sample={p.sample:p for p in rows};events={}
    for name,event,wanted in [('onset',48000,'voiced'),('offset',96000,'unvoiced')]:
        targets=sorted(t for t in by_sample if event-960<=t<=event+9600)
        first=next((t for t in targets if all(t+480*i<=event+9600 and t+480*i in by_sample and by_sample[t+480*i].state==wanted for i in range(3))),None)
        events[name]=dict(detected=first is not None,target_lag_ms=(first-event)/48 if first is not None else None,
            earliest_availability_delay_ms=(first+960-event)/48 if first is not None else None)
    pre=[p for p in rows if 9600<=p.sample<=38400]
    post=[p for p in rows if 105600<=p.sample<=134400]
    unvoiced=pre+post
    plateau_fp={label:sum(p.state=='voiced' for p in subset)/len(subset) if subset else None for label,subset in [('pre',pre),('post',post)]}
    voiced=[p for p in rows if 57600<=p.sample<=86400]
    fp=sum(p.state=='voiced' for p in unvoiced)/len(unvoiced) if unvoiced else None
    success=sum(p.state=='voiced' and abs(1200*math.log2(p.f0_hz/frequency))<=50 for p in voiced)/len(voiced) if voiced else None
    return dict(events=events,false_voicing_rate=fp,unvoiced_plateau_false_voicing=plateau_fp,voiced_success_rate=success,plateau_pass=all(v is not None and v<=.05 for v in plateau_fp.values()) and success is not None and success>=.95)


def score_analytic(base,group):
    out=base/group;m=load(out/'prediction-freeze.json');require(m['status']=='predictions_frozen','Predictions must freeze before scoring')
    per_case=[];scenarios={name:[] for name in NAMES};events={name:[] for name in NAMES}
    for row in m['records']:
        methods={}
        for name in NAMES:
            saved=study.load_feature(study.verify_file(row['predictions'][name]));preds=prediction_rows(saved)
            if group=='events':
                frequency=row['definition']['frequency_hz'];metrics=event_measure(preds,frequency);events[name].append(dict(id=row['id'],**metrics));methods[name]=metrics;continue
            definition=row['definition']
            frequency=lambda t:known.analytic_frequency(definition,t/48000) if row['cohort']=='original14' else synthesis.frequency_at(definition,t/48000)
            refs=[Reference(t,frequency(t) or None,frequency(t)>0,True,t>=3072 and t+1536<=row['sample_count']) for t in range(768,row['sample_count']+1,480)]
            methods[name]=score(refs,preds)
            if row['cohort']=='original14':
                historical=ev.read_predictions(study.verify_file(row['baseline']));raw=dict(availability_sample=saved['target_sample'].tolist(),predictions={})
                raw['availability_sample']=[int(t)+960 for t in saved['target_sample']]
                # Only the mlp fixed slot is changed; delayed DSP defines transition baseline.
                original=historical['predictions']
                for key in known.METHODS:raw['predictions'][key]=[vars(p) for p in preds] if key=='mlp' else original.get(key,original['dsp'])
                metric,_=known.score_predictions(definition,raw);scenarios[name].append(dict(scenario=definition,methods=metric))
        per_case.append(dict(id=row['id'],cohort=row.get('cohort',group),metrics=methods))
    gates={}
    for name in NAMES:
        if group=='events':
            rows=events[name];checks={'all_events_detected':len(rows)==12 and all(r['events'][k]['detected'] for r in rows for k in ('onset','offset')),
                'all_plateaux_pass':len(rows)==12 and all(r['plateau_pass'] for r in rows)}
            comparisons={};h={r['id']:r for r in events['hybrid_original']}
            for event in ('onset','offset'):
                common=[r for r in rows if h[r['id']]['events'][event]['detected']]
                defined=all(r['events'][event]['detected'] for r in common)
                baseline=float(np.median([h[r['id']]['events'][event]['target_lag_ms'] for r in common])) if common else None
                candidate=float(np.median([r['events'][event]['target_lag_ms'] for r in common])) if common and defined else None
                checks[event+'_relative_delay']=True if not common else candidate is not None and candidate-baseline<=10+1e-12
                comparisons[event]=dict(common_baseline_cases=len(common),baseline_median_signed_lag_ms=baseline,candidate_median_signed_lag_ms=candidate,
                    original_missed_events=[r['id'] for r in h.values() if not r['events'][event]['detected']])
            gates[name]=dict(status='complete',pass_=all(checks.values()),checks=checks,comparisons=comparisons,cases=rows)
            gates[name]['pass']=gates[name].pop('pass_')
        else:
            cases=curriculum.original_case_gates({'scenarios':scenarios[name]});transition=known.transition_gate(scenarios[name])['model_components']['mlp']
            old=curriculum.challenge_gate([r for r in per_case if r['cohort']=='previous32-regression'],name)
            challenge=curriculum.challenge_gate([r for r in per_case if r['cohort']=='fresh32'],name)
            checks=dict(stationary=cases['stationary_component_pass'],postchange=cases['observation_component_pass'],
                transition=transition['transition_lag_component_pass'],old_challenge=old['challenge_component_pass'],challenge=challenge['challenge_component_pass'])
            gates[name]=dict(status='complete',**{'pass':all(checks.values())},checks=checks,original_cases=cases,transition=transition,previous_challenge=old,challenge=challenge)
    result=dict(schema='phenometrix-voicing-robustness-analytic.v1',status='complete',group=group,gates=gates,records=per_case,
        prediction_freeze=study.file_record(out/'prediction-freeze.json'))
    write_json(out/'metrics.json',result);m.update(status='complete',metrics=study.file_record(out/'metrics.json'),finished_at=study.now());write_json(out/'manifest.json',m);return result


def producer_gates(models,analytic):
    """Producer formulas independent of the separate auditor's gate implementation."""
    checks={};effects={};issues=[];fn='voicing_false_negative_or_unavailable_rate';fp='voicing_false_positive_rate';joint='joint_error';success='reference_voiced_success_rate'
    expected={s+'/'+c for s in SPEAKERS for c in vi.ANCHORS}
    for name in NAMES:
        require(set(models[name]['cells'])==expected,'Incomplete development cells')
        for cell in models[name]['cells'].values():
            if any(cell[k] is None for k in (fn,fp,joint,success)):issues.append(name+': undefined denominator')
    if issues:return dict(status='unassessable',**{'pass':False},issues=issues)
    def mean(name,key,conditions):return float(np.mean([models[name]['cells'][s+'/'+c][key] for s in SPEAKERS for c in conditions]))
    h='hybrid_original';shifted=vi.ANCHORS[1:]
    for metric,domain in [(fn,shifted),(joint,vi.ANCHORS)]:
        r=float(np.mean([mean(f'R-{s}',metric,domain) for s in vi.SEEDS]));c=float(np.mean([mean(f'C-{s}',metric,domain) for s in vi.SEEDS]))
        for label,b in [('control_mean',c),(h,mean(h,metric,domain))]:
            if metric==fn:
                checks['mean_shifted_vs_'+label]=b>0 and r<=.8*b
                if b==0:issues.append('zero shifted baseline')
                effects['mean_shifted_vs_'+label]=dict(absolute_percentage_points=100*(b-r),relative_reduction=(b-r)/b if b else None)
            else:checks['mean_joint_vs_'+('control' if label=='control_mean' else 'hybrid')]=r<b
    for seed in vi.SEEDS:
        r=f'R-{seed}'
        for b in (f'C-{seed}',h):
            prefix=r+'_vs_'+b;rfn=mean(r,fn,shifted);bfn=mean(b,fn,shifted)
            checks[prefix+'/shifted_nonregression']=rfn<=bfn
            if seed==vi.SEEDS[0]:
                checks[prefix+'/exported_20pct']=bfn>0 and rfn<=.8*bfn
                checks[prefix+'/exported_joint']=mean(r,joint,vi.ANCHORS)<mean(b,joint,vi.ANCHORS)
                effects[prefix]=dict(absolute_percentage_points=100*(bfn-rfn),relative_reduction=(bfn-rfn)/bfn if bfn else None)
                if bfn==0:issues.append('zero exported baseline')
            checks[prefix+'/macro_fp']=mean(r,fp,vi.ANCHORS)-mean(b,fp,vi.ANCHORS)<=.01+1e-12
            checks[prefix+'/cell_fp']=all(models[r]['cells'][k][fp]-models[b]['cells'][k][fp]<=.02+1e-12 for k in expected)
            checks[prefix+'/clean_joint']=mean(r,joint,['clean'])-mean(b,joint,['clean'])<=.005+1e-12
            checks[prefix+'/clean_success']=mean(b,success,['clean'])-mean(r,success,['clean'])<=.01+1e-12
            checks[prefix+'/M05_first_difference']=models[r]['cells']['M05/first_difference'][fn]-models[b]['cells']['M05/first_difference'][fn]<=.02+1e-12
        checks[r+'/analytic_and_transitions']=analytic[r]['status']=='complete' and analytic[r]['pass']
    status='unassessable' if issues else 'pass' if all(checks.values()) else 'failed'
    return dict(status=status,**{'pass':status=='pass'},checks=checks,effects=effects,issues=issues,export_designation='R-20261025')


def decide(base):
    metrics=load(base/'development/metrics.json');old=load(base/'analytic/metrics.json');events=load(base/'events/metrics.json')
    analytic={name:dict(status='complete',**{'pass':old['gates'][name]['pass'] and events['gates'][name]['pass']}) for name in NAMES}
    result=producer_gates(metrics['models'],analytic)
    result.update(schema='phenometrix-voicing-robustness-decision.v1',at=study.now(),models=metrics['models'],analytic=analytic,
        candidate_passed_development=result['pass'],external_assessment_admitted=False,
        external_status='blocked: retained KEELE reference contract',
        bindings={k:study.file_record(base/k/'metrics.json') for k in ('development','analytic','events','diagnostics')})
    path=base/'decision.json';require(not path.exists(),'Never overwrite a scientific decision');write_json(path,result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['predict','score','decide']);p.add_argument('--run',type=Path,required=True);p.add_argument('--group',choices=['development','analytic','events','diagnostics']);a=p.parse_args();base=a.run.resolve()
    try:
        if a.phase=='predict':result=predict(base,a.group)
        elif a.phase=='score':result=score_natural(base,a.group) if a.group in ('development','diagnostics') else score_analytic(base,a.group)
        else:result=decide(base)
        print(json.dumps(dict(phase=a.phase,group=a.group,status=result['status']),allow_nan=False))
    except BaseException as error:
        # Retain even terminal reporting exceptions without rewriting successful scientific evidence.
        receipt=base/f'{a.phase}-{a.group or "decision"}-failure-{study.now().replace(":","-")}.json'
        if base.is_dir():write_json(receipt,dict(status='failed',phase=a.phase,group=a.group,error=repr(error),at=study.now(),source=study.file_record(Path(__file__))))
        raise

if __name__=='__main__':main()
