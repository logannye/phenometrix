"""Posthoc qualification of saved outputs from an explicitly FAILED study.

The scientific run is not repaired or reclassified. This separate wrapper calls
unchanged prospectively frozen audit equations on the original failed manifest.
It never fits, creates waveforms, rewrites metrics, or substitutes a status.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import signal
import time
import torch
from . import audit_exposure_study as frozen

REPO,ROOT,RUNS=frozen.REPO,frozen.ROOT,frozen.RUNS
base,old=frozen.base,frozen.old
require=frozen.require
EXPECTED_FAILURE="TypeError('Object of type bool is not JSON serializable')"
FROZEN_AUDITOR_SHA='baf722bdc39e910e7030ba237ec7b6dbed04c0f78ae3e77bf2b82a84bf3b1c8b'
FROZEN_TEST_SHA='b6cefad201cf71cbabd4a2c686eb5b1b304092cbf3e70645eddda91b5788fa0a'
FIRST_EXECUTION=RUNS/'exposure-study-validation-v1/audit-failed-terminal-attempt1.execution.json'
ARTIFACTS=('manifest.json','source-freeze.json','input-cache-freeze.json','prerequisites.json','training-freeze.json',
    'checkpoint-freeze.json','prediction-freeze.json','metrics.json','independent-audit.json')


def now():return datetime.now(timezone.utc).isoformat()


def source_hashes():
    result=frozen.source_hashes()
    for path in (Path(__file__).resolve(),ROOT/'tests/test_audit_exposure_forensic.py'):
        result[str(path.relative_to(REPO))]=base.sha(path)
    return result


def require_saved_failed_result(m):
    require(m['schema']=='phenometrix-exposure-study.v1' and m['status']=='failed' and m['failure']==EXPECTED_FAILURE,
        'Requires the preserved final-print failure, never a status shim')
    require(frozen.historical.dt(m['completed_at'])<=frozen.historical.dt(m['failed_at']),
        'Failure did not follow saved scientific completion fields')
    require(set(m['fits'])==set(frozen.FIT_NAMES) and set(m['fit_attempts'])==set(frozen.FIT_NAMES),
        'Failed attempt lacks the complete declared checkpoint inventory')
    require(all(r['status']=='complete' for r in m['fit_attempts'].values()),'An incomplete fit cannot be qualified as saved final output')
    require(len(m['prediction_sha256'])==103 and m['fresh_evaluation_configurations']==12 and
        m['training_generation_configurations']==0 and isinstance(m['metrics_sha256'],str) and len(m['metrics_sha256'])==64,
        'Saved score/prediction/fresh inventory is incomplete')
    require(m['development_references_read_for_final_scoring'] is True and m['reserved_content_read'] is False,
        'Reference-phase declarations differ')
    return {'scientific_status':'failed','scientific_failure':m['failure'],
        'saved_completed_at':m['completed_at'],'failed_at':m['failed_at'],
        'scientific_status_changed':False,'qualification_scope':'Posthoc verification of already-saved checkpoints, predictions and scores only'}


def validate_declaration(declaration,manifest,artifacts,current_sources):
    require(declaration['schema']=='phenometrix-exposure-posthoc-forensic-declaration.v1' and
        declaration['posthoc'] is True and declaration['original_scientific_status']=='failed','Distinct posthoc declaration required')
    require(declaration['artifact_sha256']==artifacts and declaration['audit_source_sha256']==current_sources and
        declaration['scientific_source_sha256']==manifest['source_sha256'],'Posthoc source or artifact binding differs')
    require(frozen.historical.dt(declaration['declared_at'])>=frozen.historical.dt(manifest['failed_at']),
        'Posthoc declaration cannot claim prospective timing')
    first=declaration['first_audit_elapsed_seconds'];remaining=declaration['maximum_forensic_seconds']
    require(0<=first<1800 and 0<remaining<=1800-first and declaration['maximum_python_rss_bytes']==16*1024**3,
        'Combined audit wall/RSS ceiling differs')
    return True


def declare(run,output):
    run=base.inside(RUNS,run);output=base.inside(RUNS,output)
    require(not output.exists(),'Refuses to overwrite posthoc declaration')
    m=json.loads((run/'manifest.json').read_text());require_saved_failed_result(m)
    require(base.sha(Path(frozen.__file__))==FROZEN_AUDITOR_SHA and base.sha(ROOT/'tests/test_audit_exposure_study.py')==FROZEN_TEST_SHA,
        'Prospectively frozen auditor/test changed')
    first=json.loads(FIRST_EXECUTION.read_text());receipt=json.loads((run/'independent-audit.json').read_text())
    require(first['status']=='complete' and first['returncode']==0 and first['audit_sha256']==base.sha(run/'independent-audit.json') and
        receipt['pass_result'] is True and receipt['scientific_status']=='failed' and receipt['qualification_complete'] is False,
        'First frozen failed-branch audit is missing or did not pass')
    value={'schema':'phenometrix-exposure-posthoc-forensic-declaration.v1','posthoc':True,'declared_at':now(),
        'run_path':str(run.relative_to(REPO)),'original_scientific_status':'failed','original_failure':m['failure'],
        'artifact_sha256':{name:base.sha(run/name) for name in ARTIFACTS},'audit_source_sha256':source_hashes(),
        'scientific_source_sha256':m['source_sha256'],'first_execution_path':str(FIRST_EXECUTION.relative_to(REPO)),
        'first_execution_sha256':base.sha(FIRST_EXECUTION),'first_audit_elapsed_seconds':first['elapsed_seconds'],
        'maximum_forensic_seconds':1800-first['elapsed_seconds'],'maximum_python_rss_bytes':16*1024**3,
        'fitting_or_generation_allowed':False,'source_status_or_metric_repair_allowed':False,
        'scope':'Invoke unchanged frozen verification helpers against the actual failed manifest; no status substitution'}
    validate_declaration(value,m,value['artifact_sha256'],value['audit_source_sha256'])
    output.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    return value


def audit_run(run,output,declaration_path):
    began=time.monotonic();run=base.inside(RUNS,run);output=base.inside(run,output);declaration_path=base.inside(RUNS,declaration_path)
    require(not output.exists(),'Refuses to overwrite a forensic receipt');torch.set_num_threads(1)
    audit=frozen.ExposureAudit();m=audit.json(run/'manifest.json');preserved=require_saved_failed_result(m)
    declaration=audit.json(declaration_path);artifacts={name:base.sha(audit.bind(run/name)) for name in ARTIFACTS}
    validate_declaration(declaration,m,artifacts,source_hashes())
    require(declaration['run_path']==str(run.relative_to(REPO)),'Declaration targets another attempt')
    first=audit.json(REPO/declaration['first_execution_path'],declaration['first_execution_sha256'])
    require(first['elapsed_seconds']==declaration['first_audit_elapsed_seconds'] and first['status']=='complete' and first['returncode']==0,
        'First audit execution changed')
    terminal=audit.json(run/'independent-audit.json')
    require(terminal['pass_result'] is True and terminal['scientific_status']=='failed' and terminal['qualification_complete'] is False and
        terminal['run_manifest_sha256']==artifacts['manifest.json'],'Preserved terminal audit differs')
    for p,h in declaration['audit_source_sha256'].items():audit.bind(REPO/p,h)
    for p,h in {**m['source_sha256'],**m['supervisor_sha256']}.items():
        audit.bind(REPO/p,h);audit.bind(run/'source-snapshot'/p,h)
    for p,h in m['input_sha256'].items():audit.bind(REPO/p,h)
    require(base.sha(Path(frozen.__file__))==FROZEN_AUDITOR_SHA and base.sha(ROOT/'tests/test_audit_exposure_study.py')==FROZEN_TEST_SHA,
        'Frozen audit helpers changed')
    scientific_resources=frozen.validate_resources(m)
    initial=audit.json(run/'source-freeze.json')
    require(initial['schema']=='phenometrix-exposure-source-freeze.v1' and initial['status']=='sources_frozen' and initial['fits']=={},
        'Original source freeze differs')
    for key in ('source_sha256','supervisor_sha256','training_spec_sha256','training_bank_manifest_sha256','challenge_spec_sha256',
                'arms','seeds','maximum_fits','deadline_utc','sources_frozen_at'):
        require(initial[key]==m[key],'Original source/final contract differs: '+key)
    prior=audit.json(frozen.PRIOR/'manifest.json',frozen.PRIOR_SHA)
    require(prior['status']=='complete' and m['prior_manifest']==prior and m['historical_input_sha256']==prior['input_sha256'],
        'Pinned historical experiment differs')
    require(all(m['source_sha256'].get(p)==h for p,h in prior['source_sha256'].items()),'Historical source bytes changed')
    cache,table,records,strata,support,bank_report=frozen.audit_bank(audit,m)
    expected_feasibility=frozen.bank_audit.expected_report(table,records)['feasibility']
    frozen.compare_nested(audit,expected_feasibility,m['train_feasibility'],'Saved new-bank TRAIN support')
    require(support['pass_result'],'Saved fits crossed failed support')
    # Helpers receive m exactly as read. No copy changes its failed status and
    # no module-global contract or historical function is monkeypatched.
    prereq=frozen.validate_phase_bindings(audit,run,m,initial)
    metrics=audit.json(run/'metrics.json',m['metrics_sha256'])
    require(metrics['schema']=='phenometrix-exposure-metrics.v1','Saved metrics schema differs')
    weights,checkpoint_report=frozen.verify_checkpoints(audit,run,m,prior)
    inference=frozen.verify_inference(audit,run,m,weights)
    cells,coverage,headroom,targets,natural=old.natural_and_coverage(audit,run,m,metrics,prior)
    frozen.compare_nested(audit,coverage,prereq['coverage'],'Prerequisite coverage')
    frozen.compare_nested(audit,headroom,prereq['headroom'],'Prerequisite headroom')
    frozen.compare_nested(audit,metrics['exclusions'],prereq['exclusions'],'Reference exclusions')
    checks={'train_support':True,'raw_fixed_voicing_oracle_headroom':headroom['fixed-slots']['pass'],
        'train_grouping_loss':coverage['train']['expanded7']['pass'],'development_grouping_loss':coverage['development']['expanded7']['pass']}
    require(all(checks.values()),'Fitting crossed a failed prerequisite')
    frozen.compare_nested(audit,checks,prereq['checks'],'Prerequisite components')
    frozen.compare_nested(audit,checks,metrics['decision']['prerequisites'],'Final prerequisite components')
    banks,raw_replay=frozen.reconstruct_banks(audit,run,m,cache,table,records)
    training=frozen.audit_training(audit,run,m,targets,banks,table,records,strata)
    analytic=old.audit_analytic(audit,run,m,metrics,prior)
    gate=old.natural_gate(cells);frozen.compare_nested(audit,gate,{k:metrics['decision'][k] for k in gate},'Natural decision')
    accepted=gate['pass'] and analytic['gates']['pass'] and all(checks.values())
    audit.equal(accepted,metrics['decision']['promising_for_next_study'],'Combined saved scientific decision')
    controls=3*natural['development_rows']+analytic['control_frame_pairs']
    frozen.compare_nested(audit,{'state_hz_frame_pairs':controls,'mismatches':0,'all_exact':True},m['control_reproduction'],'Historical frozen controls')
    natural['checkpoint_inference_performed']=True
    audit.close();elapsed=time.monotonic()-began
    require(elapsed<=declaration['maximum_forensic_seconds'],'Combined independent audit wall ceiling exceeded')
    result={'schema':'phenometrix-exposure-posthoc-forensic-audit.v1','status':'complete','pass_result':True,
        'posthoc':True,**preserved,'saved_artifact_checks_pass':True,'scientific_gate_pass':accepted,
        'declaration_path':str(declaration_path.relative_to(REPO)),'declaration_sha256':base.sha(declaration_path),
        'original_run_manifest_sha256':artifacts['manifest.json'],'original_metrics_sha256':artifacts['metrics.json'],
        'original_terminal_audit_sha256':artifacts['independent-audit.json'],'original_artifact_sha256':artifacts,
        'new_fits':0,'new_waveforms':0,'checkpoint_inference_performed':True,'targeted_bank':bank_report,
        'scientific_resources':scientific_resources,'checkpoint_checks':checkpoint_report,'inference':inference,
        'training':training,'raw_periodicity_replay':raw_replay,'natural':natural,'analytic':analytic,'natural_gate':gate,
        'control_frame_pairs':controls,'numeric_comparisons':audit.comparisons,'maximum_numeric_difference':audit.max_rate_delta,
        'audited_at':now(),'elapsed_seconds':elapsed,'combined_audit_elapsed_seconds':elapsed+first['elapsed_seconds'],
        'peak_observed_python_rss_bytes':audit.peak_rss_bytes,'bound_artifacts':len(audit.files),'bound_artifact_sha256':audit.files,
        'audit_source_sha256':declaration['audit_source_sha256'],'original_scientific_source_sha256':m['source_sha256'],
        'limitations':['This wrapper was declared after the failed run and observed outcomes; it is not a prospective scientific rerun.',
            'The original scientific status remains failed. Passing saved-artifact verification does not repair finalization or relax any advancement gate.',
            'Declared artifact chronology is not an operating-system reference-access log.',
            'Optimization and TRAIN before/after logit diagnostics are not independently replayed.',
            'Raw periodicity replay covers only the predeclared4608 synthetic TRAIN windows; no full candidate extractor or natural PCM replay.',
            'Natural development is consumed; synthetic configuration freshness is not independent frequency support; no clinical or deployment qualification.']}
    output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('declare','audit'))
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--declaration',type=Path);args=parser.parse_args()
    if args.action=='declare':
        value=declare(args.run,args.output);print(json.dumps({'posthoc':value['posthoc'],'declaration_sha256':base.sha(args.output)}));return
    require(args.declaration is not None,'An immutable posthoc declaration is required')
    output=base.inside(base.inside(RUNS,args.run),args.output);require(not output.exists(),'Refuses overwrite')
    declaration=json.loads(args.declaration.read_text());seconds=int(declaration['maximum_forensic_seconds'])
    require(0<seconds<=1800,'Invalid forensic wall budget')
    def stop(signum,frame):raise TimeoutError('Combined independent audit wall budget reached')
    signal.signal(signal.SIGALRM,stop);signal.alarm(seconds);began=time.monotonic()
    try:
        result=audit_run(args.run,output,args.declaration)
        print(json.dumps({k:result[k] for k in ('status','pass_result','scientific_status','scientific_gate_pass','posthoc','numeric_comparisons','maximum_numeric_difference','elapsed_seconds','bound_artifacts')},indent=2))
    except BaseException as error:
        if not output.exists():
            output.write_text(json.dumps({'schema':'phenometrix-exposure-posthoc-forensic-audit.v1','status':'failed','pass_result':False,
                'posthoc':True,'failure':repr(error),'scientific_status':'failed','scientific_status_changed':False,
                'declaration_sha256':base.sha(args.declaration),'audit_source_sha256':source_hashes(),'elapsed_seconds':time.monotonic()-began},indent=2,sort_keys=True)+'\n')
        raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
