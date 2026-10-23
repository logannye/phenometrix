"""Independent count, alignment and gate audit; no new model inference."""
from __future__ import annotations
import argparse, gzip, hashlib, json, math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
import numpy as np
from . import synthetic_curriculum as sc


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frequency(d, t):
    if d.get("latent_voiced") is False:
        return None
    a=d.get("initial_hz",d.get("initialHz")); b=d.get("final_hz",d.get("finalHz"))
    start=d.get("change_start_seconds",d.get("changeStartSeconds")); end=d.get("change_end_seconds",d.get("changeEndSeconds"))
    if d["kind"]=="step": return a if t<start else b
    if d["kind"]=="glide": return a+(b-a)*min(1.,max(0.,(t-start)/(end-start)))
    return a


def audit(run, plan_folder):
    research=Path(__file__).resolve().parents[1]; repo=research.parents[1]
    plan=json.loads((plan_folder/'plan.json').read_text()); manifest=json.loads((run/'manifest.json').read_text())
    freeze=json.loads((run/'prediction-freeze.json').read_text()); results=json.loads((run/'metrics.json').read_text())
    assert sha(plan_folder/'plan.json')==json.loads((plan_folder/'declaration.json').read_text())['plan_sha256']==manifest['plan_sha256']
    assert manifest['status']=='complete' and freeze['status']=='predictions_frozen'
    assert freeze['analytic_references_read_before_prediction_freeze'] is False
    assert sha(run/'metrics.json')==manifest['metrics_sha256'] and manifest['records']==freeze['records']
    assert len(manifest['records'])==78
    for name,expected in plan['source_sha256'].items(): assert sha(repo/name)==expected,name
    for m in plan['models'].values(): assert sha(research/'runs'/m['directory']/'weights.pt')==m['weights_sha256']
    support=research/'runs/support-diagnostic-v1'
    definitions={r['id']:r for r in json.loads((support/'manifest.json').read_text())['scenarios']}
    for hz in (100,200,300): definitions[f'observation-change-{hz}']={'kind':'observation-change','initialHz':hz,'finalHz':hz,'changeStartSeconds':.6,'changeEndSeconds':.6}
    definitions.update({r['id']:r for r in json.loads((research/'runs/synthetic-curriculum-plan/challenge-plan.json').read_text())['records']})
    definitions.update({r['id']:r for r in plan['fresh_challenge_records']})
    reported={r['id']:r for r in results['per_case']}; comparisons=frames=0
    fresh={m:[] for m in ('hybrid_original','hybrid_augmented')}
    lag={m:{'step':[],'glide':[]} for m in ('dsp','hybrid_original','hybrid_augmented')}
    for record in manifest['records']:
        name=record['id']; d=definitions[name]
        if not name.startswith('observation-change-'): assert sc.parameter_digest(d)==record['parameters_sha256']
        fpath=run/'features'/(name+'.npz'); ppath=run/'predictions'/(name+'.json.gz')
        assert sha(fpath)==record['feature_sha256'] and sha(ppath)==record['prediction_sha256']
        with np.load(fpath,allow_pickle=False) as f:
            ends=f['reference_sample'].copy(); hz=f['candidate_hz'].copy(); masks=f['candidate_mask'].copy(); n=int(f['pcm_sample_count'])
            assert int(f['sample_rate_hz'])==48000 and n==record['pcm_sample_count']
        assert ends[0]==2208 and np.all(np.diff(ends)==480)
        if record['cohort']=='original14':
            if name.startswith('observation-change-'):
                t=np.arange(76800,dtype=np.float64)/48000; phase=2*np.pi*d['initialHz']*t
                pcm=(.075*np.where(t<.6,np.sin(phase)+.5*np.sin(2*phase)+.25*np.sin(3*phase),np.sin(2*phase)+.5*np.sin(3*phase))).astype(np.float32)
            else: pcm=np.fromfile(support/'pcm'/(name+'.f32le'),dtype='<f4')
        else: pcm=sc.generate_pcm(d)
        assert hashlib.sha256(pcm.tobytes()).hexdigest()==record['pcm_sha256'],name
        with gzip.open(ppath,'rt') as f: raw=json.load(f)
        assert raw['availability_sample']==ends.tolist()
        target=ends-960; include=np.flatnonzero((target>=3072)&(target+1536<=n))
        truths={int(i):frequency(d,target[i]/48000) for i in include}
        voiced=sum(v is not None for v in truths.values())
        oracle=sum(any(abs(1200*math.log2(h/truths[int(i)]))<=50 for h in hz[i,masks[i]]) for i in include if truths[int(i)] is not None)
        assert reported[name]['oracle']['covered_frames']==oracle and reported[name]['oracle']['reference_voiced_frames']==voiced
        for method,rows in raw['predictions'].items():
            assert [r['sample'] for r in rows]==target.tolist()
            good=fp=fn=unavailable=success=0
            for i in include:
                r=rows[i]; truth=truths[int(i)]; pred=r['f0_hz'] if r['state']=='voiced' else None
                unavailable+=r['state']=='unavailable'
                if truth is None: fp+=r['state']=='voiced'; good+=r['state']=='unvoiced'
                else:
                    fn+=r['state']=='unvoiced'; correct=pred is not None and abs(1200*math.log2(pred/truth))<=50
                    good+=correct; success+=correct
                if method in fresh:
                    p=raw['predictions']['mlp_'+method.removeprefix('hybrid_')][i]['voicing_probability']; y=raw['yin_evidence'][i]
                    assert y['sample']==int(ends[i]); v=y['raw_f0_hz']
                    state='unvoiced' if p<.5 else 'voiced' if math.isfinite(v) and 50<=v<=700 else 'unavailable'
                    assert r['state']==state and r['f0_hz']==(v if state=='voiced' else None)
            actual=reported[name]['metrics'][method]
            check={'eligible_valid_frames':len(include),'reference_voiced_frames':voiced,'correct_frames':good,'voicing_false_positive_count':fp,
                'voicing_false_negative_count':fn,'unavailable_frames':unavailable,'reference_voiced_success_count':success}
            for key,value in check.items(): assert actual[key]==value,(name,method,key)
            assert abs(actual['joint_error']-(1-good/len(include)))<1e-12
            comparisons+=1
            if record['cohort']=='fresh32' and method in fresh: fresh[method].append((voiced,success,fp,len(include)-voiced))
            if record['cohort']=='original14' and method in lag and d['kind'] in ('step','glide'):
                if d['kind']=='step':
                    eligible=[i for i in include if ends[i]/48000>=d['changeStartSeconds']]
                    ok=[rows[i]['state']=='voiced' and abs(1200*math.log2(rows[i]['f0_hz']/d['finalHz']))<=50 for i in eligible]
                    j=next(j for j in range(len(eligible)-2) if all(ok[j:j+3]))
                    value=target[eligible[j]]/48-d['changeStartSeconds']*1000
                else:
                    slope=(d['finalHz']-d['initialHz'])/(d['changeEndSeconds']-d['changeStartSeconds'])
                    values=[(frequency(d,target[i]/48000)-rows[i]['f0_hz'])/slope*1000 for i in include
                        if ends[i]-1920>=d['changeStartSeconds']*48000 and ends[i]<d['changeEndSeconds']*48000 and rows[i]['state']=='voiced']
                    value=median(values)
                lag[method][d['kind']].append(value)
        frames+=len(ends)
    gates={}
    for method,rows in fresh.items():
        v=[s/n for n,s,fp,u in rows if n]; u=[fp/u for n,s,fp,u in rows if u]; g=results['gates'][method]['fresh_challenge']
        assert len(v)==28 and len(u)==4
        assert abs(mean(v)-g['equal_scenario_reference_voiced_success'])<1e-12 and abs(min(v)-g['minimum_scenario_reference_voiced_success'])<1e-12
        passed=mean(v)>=.9 and min(v)>=.8 and max(u)<=.05; assert passed==g['challenge_component_pass']
        extra={}
        for group,key in [('step','median_step_target_settling_lag_ms'),('glide','median_glide_equivalent_delay_ms'),('all','median_signed_target_delay_ms')]:
            a=lag[method]['step']+lag[method]['glide'] if group=='all' else lag[method][group]
            b=lag['dsp']['step']+lag['dsp']['glide'] if group=='all' else lag['dsp'][group]
            assert len(a)==len(b)=={'step':2,'glide':6,'all':8}[group]
            extra[key]=median(a)-median(b)
            assert abs(extra[key]-results['gates'][method]['original_transition']['extra_median_target_delay_vs_delayed_dsp_ms'][key])<1e-10
        assert all(v<=10 for v in extra.values())==results['gates'][method]['original_transition']['transition_lag_component_pass']
        gates[method]={'fresh_challenge_pass':passed,'extra_signed_transition_medians_ms':extra}
    return {'schema':'phenometrix-hybrid-synthetic-independent-audit.v1','status':'pass','created_at':datetime.now(timezone.utc).isoformat(),
        'audit_source_sha256':sha(__file__),'plan_sha256':sha(plan_folder/'plan.json'),'metrics_sha256':sha(run/'metrics.json'),
        'prediction_freeze_sha256':sha(run/'prediction-freeze.json'),'verified_source_bindings':len(plan['source_sha256']),
        'verified_pcm_feature_prediction_records':78,'verified_acquisition_frame_alignment':frames,'independent_case_method_count_checks':comparisons,
        'verified_gates':gates,'limitations':'Byte identities and analytic arithmetic only; no new inference, natural holdout, or deployment qualification.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','plan','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); result=audit(a.run.resolve(),a.plan.resolve())
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
