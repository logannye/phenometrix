"""Versioned strict scenario admission for the saved exposure forensic audit.

Copied from frozen audit_evidence_scores.audit_analytic; the ONLY function-body
change is its fresh-scenario equality RHS. It now compares all13 physical and
timing fields to independently enumerated frozen parameters. Score equations,
masks, cases, methods, gates, hashes and chronology checks are unchanged.
No old module global, source, input, or result is modified.
"""
from __future__ import annotations
import hashlib
import numpy as np
from . import audit_evidence_scores as old
REPO,PRIOR=old.REPO,old.PRIOR
GROUP_COUNTS,CANDIDATE,SELECTORS=old.GROUP_COUNTS,old.CANDIDATE,old.SELECTORS
base,historical,pool,retention=old.base,old.historical,old.pool,old.retention
require,bound,load_npz,compare_nested=old.require,old.bound,old.load_npz,old.compare_nested
fresh_records,analytic_gates,independent_groups,verify_evidence=old.fresh_records,old.analytic_gates,old.independent_groups,old.verify_evidence
SEEDS=old.SEEDS


def expected_scenario(params):
    require(params in fresh_records(),'Fresh parameters are outside the frozen independent12-case enumeration')
    return {'id':params['id'],'kind':'observation-change','durationSeconds':2.,
        'initialHz':params['frequency_hz'],'finalHz':params['frequency_hz'],
        'changeStartSeconds':.83,'changeEndSeconds':.83,'slopeHzPerSecond':0,
        'change_sample':params['change_sample'],'before':params['before'],'after':params['after'],
        'phase_radians':params['phase_radians'],'gain':params['gain']}


def audit_analytic(audit,run,manifest,metrics,prior):
    inventory=audit.json(bound(audit,manifest,run/'analytic-inventory.json'))
    require({g:len(v) for g,v in inventory.items()}==GROUP_COUNTS,'86-case inventory differs')
    previous=audit.json(bound(audit,prior,PRIOR/'analytic-inventory.json'))
    old={(('consumed_expansion' if g=='challenge' else g),r['id']):r for g,rows in previous.items() for r in rows}
    cm=audit.json(run/'challenge/manifest.json');spec=audit.json(run/'challenge/spec.json')
    require(cm['status']=='complete' and cm['reference_content_read'] is False and spec['records']==fresh_records(),'Fresh challenge records differ')
    require(hashlib.sha256(historical.canonical(spec)).hexdigest()==manifest['challenge_spec_sha256']==cm['spec_sha256'],'Fresh metadata identity differs')
    require(cm['freeze_path']==str((run/'checkpoint-freeze.json').relative_to(REPO)) and cm['freeze_sha256']==base.sha(run/'checkpoint-freeze.json'),'Fresh checkpoint linkage differs')
    require(historical.dt(manifest['checkpoints_frozen_at'])<=historical.dt(cm['started_at'])<=historical.dt(cm['completed_at'])<=historical.dt(manifest['predictions_frozen_at']),'Fresh challenge chronology differs')
    fresh={r['id']:r for r in cm['records']};parameters={r['id']:r for r in fresh_records()}
    require(list(fresh)==list(parameters),'Fresh case order differs')
    gates={m:{} for m in CANDIDATE};summaries={};controls=0;frames=0
    for group,rows in inventory.items():
        require(set(metrics[group])=={r['id'] for r in rows} and len({r['id'] for r in rows})==len(rows),'Analytic reported inventory differs')
        for row in rows:
            name=row['id'];scenario=row['scenario'];n=76800 if group=='regression' else 96000
            if group=='challenge':
                source=fresh[name];params=parameters[name]
                require(source['parameters_sha256']==hashlib.sha256(historical.canonical(params)).hexdigest(),'Fresh parameters hash differs')
                require(row['original_feature_path']==str((run/'challenge'/source['feature_path']).relative_to(REPO)) and row['original_feature_sha256']==source['feature_sha256'] and row['pcm_sha256']==source['pcm_sha256'],'Fresh source artifact differs')
                audit.bind(run/'challenge'/source['pcm_path'],source['pcm_sha256'])
                require(scenario==expected_scenario(params),'Fresh scenario differs')
            else:
                source=old[(group,name)]
                for key in ('scenario','sample_count','original_feature_path','original_feature_sha256','pcm_sha256'):
                    require(row[key]==source[key],'Consumed analytic identity differs: '+key)
            original=load_npz(audit,REPO/row['original_feature_path'],row['original_feature_sha256'])
            expanded=load_npz(audit,REPO/row['expanded_feature_path'],row['expanded_feature_sha256'])
            pool.validate_pool(original,6);pool.validate_pool(expanded,7)
            ends,targets,eligible=historical.validate_feature(original,n);require(row['sample_count']==n,'Analytic extent differs')
            require(np.array_equal(ends,expanded['reference_sample']),'Analytic expanded support differs')
            groups={'original6':independent_groups(original),'expanded7':independent_groups(expanded)}
            verify_evidence(load_npz(audit,REPO/row['evidence_path'],row['evidence_sha256']),expanded,groups['expanded7'],row['pcm_sha256'])
            z=load_npz(audit,run/row['prediction_path'],row['prediction_sha256'])
            require(row['prediction_sha256']==manifest['prediction_sha256'][row['prediction_path']],'Analytic prediction freeze differs')
            require(set(z)=={'original_probability'}|{m+s for m in SELECTORS for s in ('__state','__hz')},'Analytic prediction fields differ')
            probability=z['original_probability'];require(probability.shape==targets.shape and np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all(),'Analytic probability invalid')
            arrays={m:(z[m+'__state'],z[m+'__hz']) for m in SELECTORS}
            for m,(state,hz) in arrays.items():
                base.prediction_objects(targets,state,hz)
                require(np.array_equal(state,np.where(probability<.5,0,np.where(groups['expanded7']['candidate_mask'].any(1),1,2))),'Analytic fixed voice/availability differs')
                require(np.all(np.any(groups['expanded7']['candidate_hz'][state==1]==hz[state==1,None],axis=1)),'Analytic output is not a grouped actual member')
            if group!='challenge':
                p=PRIOR/source['prediction_path'];require(row['prior_prediction_path']==str(p.relative_to(REPO)),'Prior prediction path differs')
                saved=load_npz(audit,p,source['prediction_sha256'])
                require(np.array_equal(probability,saved['original_probability']),'Consumed original voicing probability differs')
                for seed in SEEDS:
                    for suffix in ('__state','__hz'):
                        require(np.array_equal(z[f'frozen-expanded7__seed{seed}'+suffix],saved[f'expanded7__seed{seed}'+suffix],equal_nan=True),'Frozen expanded analytic control changed')
                    controls+=len(targets)
            truth=historical.curve(scenario,targets/48000);region=historical.regions(scenario,ends);report=metrics[group][name]
            require(report['kind']==scenario['kind'] and set(report['methods'])==set(SELECTORS),'Analytic method/kind differs')
            compare_nested(audit,pool.analytic_coverage({'original6':original,'expanded7':expanded},groups,truth,eligible,region,probability,arrays),report['candidate_coverage'],group+'/'+name+'/coverage')
            case={}
            for method,(state,hz) in arrays.items():
                for label,include in {'all':eligible,**{r:eligible&(region==r) for r in sorted(set(region[eligible]))}}.items():
                    got=historical.analytic_summary(truth[include],state[include],hz[include],region[include],scenario['slopeHzPerSecond'])
                    want=report['methods'][method]['all'] if label=='all' else report['methods'][method]['regions'][label]
                    compare_nested(audit,got,want,group+'/'+name+'/'+method+'/'+label)
                if scenario['kind']=='step':
                    compare_nested(audit,retention.step_summary(scenario,targets[eligible],ends[eligible],state[eligible],hz[eligible]),report['methods'][method]['step_response'],group+'/'+name+'/'+method+'/step')
                else:require(report['methods'][method]['step_response'] is None,'Unexpected step response')
                if method not in CANDIDATE:continue
                if group!='regression':category=group;include=eligible&(region=='after-observation-change')
                elif scenario['kind']=='constant':category='stationary';include=eligible
                elif scenario['kind']=='observation-change':category='harmonic_change';include=eligible&(region=='after-observation-change')
                else:continue
                value=historical.analytic_summary(truth[include],state[include],hz[include],region[include],0)
                gates[method][group+'/'+name]={'category':category,'success':value['success_within_50c_all_frames']}
                case[method]={'frames':value['frames'],'success':value['success_within_50c_all_frames']}
            summaries[group+'/'+name]=case;frames+=len(targets)
    decision=analytic_gates(gates);compare_nested(audit,decision,metrics['decision']['analytic'],'Analytic gate')
    return {'cases':86,'methods_per_case':15,'frames_per_method':frames,'mandatory_cases_per_candidate_seed':78,
        'control_frame_pairs':controls,'gates':decision,'case_results':summaries}
