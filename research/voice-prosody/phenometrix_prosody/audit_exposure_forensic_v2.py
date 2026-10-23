"""Posthoc v2 saved-output audit; original failed run and first audit preserved.

The sole analytic change is strict 13-field scenario admission in the separate
adapter. Frozen numerical equations, gates and scientific bytes remain intact.
Every returned verification stage is recorded before the next stage starts.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch
from . import audit_exposure_forensic as previous
from . import audit_exposure_strict_analytic as strict

frozen=previous.frozen
REPO,ROOT,RUNS=previous.REPO,previous.ROOT,previous.RUNS
base,old,require=previous.base,previous.old,previous.require
require_saved_failed_result=previous.require_saved_failed_result
FROZEN_AUDITOR_SHA,FROZEN_TEST_SHA=previous.FROZEN_AUDITOR_SHA,previous.FROZEN_TEST_SHA
FIRST_EXECUTION=previous.FIRST_EXECUTION
PRIOR_EXECUTION=RUNS/'exposure-study-validation-v1/forensic-audit-attempt1.execution.json'
ARTIFACTS=previous.ARTIFACTS+('forensic-audit.json',)
STAGE_FOLDER='forensic-v2-stages'
PRIOR_FILES={
 'research/voice-prosody/runs/exposure-study-validation-v1/audit-failed-terminal-attempt1.execution.json':'fe7f0e22721e9cc2df1a7c80eeb2e627d133bbba8d12767da498bf68395d1133',
 'research/voice-prosody/runs/exposure-study-validation-v1/forensic-audit-attempt1.execution.json':'d4799328522414a89c4c603a2f5491fc968a029f66dbfc9d0177d3f82791bd14',
 'research/voice-prosody/phenometrix_prosody/audit_exposure_forensic.py':'6c58d44c6a5ccc607139a300a9c2fc7c669acf633d36868f158bd2b72950f819',
 'research/voice-prosody/tests/test_audit_exposure_forensic.py':'08360be60b2faa280ead1cbb41155422cd73300917cf3ad6d952a4309d243739',
 'research/voice-prosody/runs/exposure-study-validation-v1/forensic-declaration.json':'347213637a2bd7c90ebe6f6340818cba47e3e04c82047b567723c7154cff666a',
 'research/voice-prosody/runs/exposure-study-v1/forensic-audit.json':'187258fb3e3db3038c805289e0f1ce8fdd62a8aa853a74375723a8fa05357777',
 'research/voice-prosody/runs/exposure-study-validation-v1/forensic-audit-attempt1.stderr.log':'7e0c8b0175d5d8be4e70160942182b59a83a6552e7761ff17861e9938f2ae85c',
}


def now():return datetime.now(timezone.utc).isoformat()


def scalar(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError('Unsupported receipt value: '+type(value).__name__)


def write_once(path,value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False,default=scalar)+'\n')


def source_hashes():
    result=previous.source_hashes()
    for path in (Path(__file__).resolve(),Path(strict.__file__).resolve(),ROOT/'tests/test_audit_exposure_forensic_v2.py'):
        result[str(path.relative_to(REPO))]=base.sha(path)
    return result


def remaining_budget(first_elapsed,prior_elapsed):
    require(np.isfinite(first_elapsed) and np.isfinite(prior_elapsed) and first_elapsed>=0 and prior_elapsed>=0,
        'Invalid prior elapsed time')
    remaining=1800-first_elapsed-prior_elapsed
    require(remaining>0,'Combined independent audit budget exhausted')
    return remaining


def lineage_files():
    paths=set(PRIOR_FILES)|{str(FIRST_EXECUTION.relative_to(REPO)),str(PRIOR_EXECUTION.relative_to(REPO))}
    for prefix in ('audit-failed-terminal-attempt1','forensic-audit-attempt1'):
        for suffix in ('stdout.log','stderr.log','supervisor.py'):
            paths.add(str((PRIOR_EXECUTION.parent/(prefix+'.'+suffix)).relative_to(REPO)))
    return {path:base.sha(REPO/path) for path in sorted(paths)}


def validate_previous(first,prior,failed,terminal):
    require(first['status']=='complete' and first['returncode']==0 and terminal['pass_result'] is True and
        terminal['scientific_status']=='failed' and terminal['qualification_complete'] is False,'Frozen terminal audit differs')
    require(prior['status']=='failed' and prior['returncode']==1 and prior['stop_reason'] is None and
        failed['status']=='failed' and failed['pass_result'] is False and
        failed['failure']=="ValueError('Fresh scenario differs')" and failed['scientific_status']=='failed' and
        failed['scientific_status_changed'] is False,'Preserved strict-scenario audit failure differs')
    return remaining_budget(first['elapsed_seconds'],prior['elapsed_seconds'])


def validate_declaration(d,m,artifacts,sources,lineage):
    require(d['schema']=='phenometrix-exposure-posthoc-forensic-declaration.v2' and d['posthoc'] is True and
        d['original_scientific_status']=='failed','Distinct v2 posthoc declaration required')
    require(d['artifact_sha256']==artifacts and d['audit_source_sha256']==sources and
        d['scientific_source_sha256']==m['source_sha256'] and d['prior_attempt_sha256']==lineage,'Source or artifact binding differs')
    require(all(lineage.get(p)==h for p,h in PRIOR_FILES.items()),'Prior forensic source or failure bytes changed')
    require(frozen.historical.dt(d['declared_at'])>=frozen.historical.dt(m['failed_at']), 'Posthoc chronology differs')
    require(d['maximum_forensic_seconds']==remaining_budget(d['first_audit_elapsed_seconds'],d['prior_forensic_elapsed_seconds']) and
        d['maximum_python_rss_bytes']==16*1024**3,'Combined remaining wall/RSS budget differs')
    require(d['scenario_fields']==list(strict.expected_scenario(strict.fresh_records()[0])) and
        d['source_status_or_metric_repair_allowed'] is False and d['fitting_or_generation_allowed'] is False,
        'Strict scenario or no-repair contract differs')
    return True


def declare(run,output):
    run=base.inside(RUNS,run);output=base.inside(RUNS,output)
    require(not output.exists(),'Refuses to overwrite posthoc declaration')
    m=json.loads((run/'manifest.json').read_text());require_saved_failed_result(m)
    require(base.sha(Path(frozen.__file__))==FROZEN_AUDITOR_SHA and base.sha(ROOT/'tests/test_audit_exposure_study.py')==FROZEN_TEST_SHA,
        'Prospectively frozen auditor/test changed')
    first=json.loads(FIRST_EXECUTION.read_text());prior=json.loads(PRIOR_EXECUTION.read_text())
    failed=json.loads((run/'forensic-audit.json').read_text());terminal=json.loads((run/'independent-audit.json').read_text())
    remaining=validate_previous(first,prior,failed,terminal)
    require(first['audit_sha256']==base.sha(run/'independent-audit.json') and prior['audit_sha256']==base.sha(run/'forensic-audit.json'),
        'Prior audit execution/receipt binding differs')
    value={'schema':'phenometrix-exposure-posthoc-forensic-declaration.v2','posthoc':True,'declared_at':now(),
        'run_path':str(run.relative_to(REPO)),'original_scientific_status':'failed','original_failure':m['failure'],
        'artifact_sha256':{name:base.sha(run/name) for name in ARTIFACTS},'audit_source_sha256':source_hashes(),
        'scientific_source_sha256':m['source_sha256'],'prior_attempt_sha256':lineage_files(),
        'first_audit_elapsed_seconds':first['elapsed_seconds'],'prior_forensic_elapsed_seconds':prior['elapsed_seconds'],
        'maximum_forensic_seconds':remaining,'maximum_python_rss_bytes':16*1024**3,
        'fitting_or_generation_allowed':False,'source_status_or_metric_repair_allowed':False,
        'scenario_fields':list(strict.expected_scenario(strict.fresh_records()[0])),
        'exact_analytic_function_change':'Fresh-scenario equality RHS only: eight-field literal becomes independently enumerated thirteen-field expected_scenario(params).',
        'scope':'Posthoc verification of saved outputs against the actual failed manifest; all prior attempts preserved.'}
    validate_declaration(value,m,value['artifact_sha256'],value['audit_source_sha256'],value['prior_attempt_sha256'])
    write_once(output,value);return value


class Stages:
    def __init__(self,folder,declaration_sha256,audit):
        self.folder=Path(folder);self.folder.mkdir(exist_ok=False)
        self.declaration_sha256=declaration_sha256;self.audit=audit;self.index=[];self.started=time.monotonic()
    def add(self,name,result):
        require(name.replace('-','').isalnum(),'Unsafe stage name')
        path=self.folder/(f'{len(self.index)+1:02d}-'+name+'.json')
        value={'schema':'phenometrix-exposure-forensic-stage.v2','status':'complete','stage':name,
            'declaration_sha256':self.declaration_sha256,'recorded_at':now(),'elapsed_seconds':time.monotonic()-self.started,
            'numeric_comparisons_so_far':self.audit.comparisons,'maximum_numeric_difference_so_far':self.audit.max_rate_delta,
            'peak_observed_python_rss_bytes':self.audit.peak_rss_bytes,'result':result}
        write_once(path,value);self.audit.bind(path)
        self.index.append({'stage':name,'path':str(path.relative_to(REPO)),'sha256':base.sha(path)})
        return value


def completed_stages(folder,declaration_sha256):
    result=[]
    if not folder.exists():return result
    for path in sorted(folder.glob('*.json')):
        value=json.loads(path.read_text())
        require(value['schema']=='phenometrix-exposure-forensic-stage.v2' and value['status']=='complete' and
            value['declaration_sha256']==declaration_sha256,'Completed stage identity differs')
        result.append({'stage':value['stage'],'path':str(path.relative_to(REPO)),'sha256':base.sha(path),
            'numeric_comparisons_so_far':value['numeric_comparisons_so_far'],
            'maximum_numeric_difference_so_far':value['maximum_numeric_difference_so_far']})
    return result


def array_identities(value):
    if isinstance(value,np.ndarray):
        return {'shape':list(value.shape),'dtype':str(value.dtype),'bytes_sha256':hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
    if isinstance(value,dict):return {k:array_identities(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [array_identities(v) for v in value]
    return value


def audit_run(run,output,declaration_path):
    began=time.monotonic();run=base.inside(RUNS,run);output=base.inside(run,output);declaration_path=base.inside(RUNS,declaration_path)
    require(not output.exists(),'Refuses to overwrite a forensic receipt');torch.set_num_threads(1)
    audit=frozen.ExposureAudit();m=audit.json(run/'manifest.json');preserved=require_saved_failed_result(m)
    declaration=audit.json(declaration_path);artifacts={name:base.sha(audit.bind(run/name)) for name in ARTIFACTS}
    lineage=lineage_files()
    validate_declaration(declaration,m,artifacts,source_hashes(),lineage)
    require(declaration['run_path']==str(run.relative_to(REPO)),'Declaration targets another attempt')
    for p,h in lineage.items():audit.bind(REPO/p,h)
    first=audit.json(FIRST_EXECUTION);prior_execution=audit.json(PRIOR_EXECUTION)
    terminal=audit.json(run/'independent-audit.json');failed=audit.json(run/'forensic-audit.json')
    validate_previous(first,prior_execution,failed,terminal)
    require(first['elapsed_seconds']==declaration['first_audit_elapsed_seconds'] and
        prior_execution['elapsed_seconds']==declaration['prior_forensic_elapsed_seconds'] and
        first['audit_sha256']==artifacts['independent-audit.json'] and prior_execution['audit_sha256']==artifacts['forensic-audit.json'],
        'Prior execution identities or durations differ')
    require(terminal['run_manifest_sha256']==artifacts['manifest.json'],'Preserved terminal audit differs')
    stages=Stages(run/STAGE_FOLDER,base.sha(declaration_path),audit)
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
    stages.add('provenance',{'scientific_resources':scientific_resources,'scientific_status':'failed','source_files':len(m['source_sha256'])})
    cache,table,records,strata,support,bank_report=frozen.audit_bank(audit,m)
    expected_feasibility=frozen.bank_audit.expected_report(table,records)['feasibility']
    frozen.compare_nested(audit,expected_feasibility,m['train_feasibility'],'Saved new-bank TRAIN support')
    require(support['pass_result'],'Saved fits crossed failed support')
    stages.add('training-bank',{'bank':bank_report,'support':support})
    # Helpers receive m exactly as read. No copy changes its failed status and
    # no module-global contract or historical function is monkeypatched.
    prereq=frozen.validate_phase_bindings(audit,run,m,initial)
    stages.add('phase-bindings',{'prerequisites':prereq})
    metrics=audit.json(run/'metrics.json',m['metrics_sha256'])
    require(metrics['schema']=='phenometrix-exposure-metrics.v1','Saved metrics schema differs')
    weights,checkpoint_report=frozen.verify_checkpoints(audit,run,m,prior)
    stages.add('checkpoints',checkpoint_report)
    inference=frozen.verify_inference(audit,run,m,weights)
    stages.add('inference',inference)
    cells,coverage,headroom,targets,natural=old.natural_and_coverage(audit,run,m,metrics,prior)
    frozen.compare_nested(audit,coverage,prereq['coverage'],'Prerequisite coverage')
    frozen.compare_nested(audit,headroom,prereq['headroom'],'Prerequisite headroom')
    frozen.compare_nested(audit,metrics['exclusions'],prereq['exclusions'],'Reference exclusions')
    checks={'train_support':True,'raw_fixed_voicing_oracle_headroom':headroom['fixed-slots']['pass'],
        'train_grouping_loss':coverage['train']['expanded7']['pass'],'development_grouping_loss':coverage['development']['expanded7']['pass']}
    require(all(checks.values()),'Fitting crossed a failed prerequisite')
    frozen.compare_nested(audit,checks,prereq['checks'],'Prerequisite components')
    frozen.compare_nested(audit,checks,metrics['decision']['prerequisites'],'Final prerequisite components')
    stages.add('natural-prerequisites',{'natural':natural,'coverage':coverage,'headroom':headroom,'checks':checks,'cells':cells,'target_array_identities':array_identities(targets)})
    banks,raw_replay=frozen.reconstruct_banks(audit,run,m,cache,table,records)
    stages.add('raw-periodicity-replay',raw_replay)
    training=frozen.audit_training(audit,run,m,targets,banks,table,records,strata)
    stages.add('training-targets-schedules-exposure',training)
    analytic=strict.audit_analytic(audit,run,m,metrics,prior)
    stages.add('analytic-scores',analytic)
    gate=old.natural_gate(cells);frozen.compare_nested(audit,gate,{k:metrics['decision'][k] for k in gate},'Natural decision')
    accepted=gate['pass'] and analytic['gates']['pass'] and all(checks.values())
    audit.equal(accepted,metrics['decision']['promising_for_next_study'],'Combined saved scientific decision')
    controls=3*natural['development_rows']+analytic['control_frame_pairs']
    frozen.compare_nested(audit,{'state_hz_frame_pairs':controls,'mismatches':0,'all_exact':True},m['control_reproduction'],'Historical frozen controls')
    stages.add('final-gates-controls',{'natural_gate':gate,'scientific_gate_pass':accepted,'control_frame_pairs':controls})
    natural['checkpoint_inference_performed']=True
    audit.close();elapsed=time.monotonic()-began
    require(elapsed<=declaration['maximum_forensic_seconds'],'Combined independent audit wall ceiling exceeded')
    result={'schema':'phenometrix-exposure-posthoc-forensic-audit.v2','status':'complete','pass_result':True,
        'posthoc':True,**preserved,'saved_artifact_checks_pass':True,'scientific_gate_pass':accepted,
        'declaration_path':str(declaration_path.relative_to(REPO)),'declaration_sha256':base.sha(declaration_path),
        'original_run_manifest_sha256':artifacts['manifest.json'],'original_metrics_sha256':artifacts['metrics.json'],
        'original_terminal_audit_sha256':artifacts['independent-audit.json'],'original_artifact_sha256':artifacts,
        'previous_forensic_audit_sha256':artifacts['forensic-audit.json'],'completed_stages':stages.index,
        'prior_attempt_sha256':lineage,'strict_scenario_fields':declaration['scenario_fields'],
        'new_fits':0,'new_waveforms':0,'checkpoint_inference_performed':True,'targeted_bank':bank_report,
        'scientific_resources':scientific_resources,'checkpoint_checks':checkpoint_report,'inference':inference,
        'training':training,'raw_periodicity_replay':raw_replay,'natural':natural,'analytic':analytic,'natural_gate':gate,
        'control_frame_pairs':controls,'numeric_comparisons':audit.comparisons,'maximum_numeric_difference':audit.max_rate_delta,
        'audited_at':now(),'elapsed_seconds':elapsed,'combined_audit_elapsed_seconds':elapsed+first['elapsed_seconds']+prior_execution['elapsed_seconds'],
        'peak_observed_python_rss_bytes':audit.peak_rss_bytes,'bound_artifacts':len(audit.files),'bound_artifact_sha256':audit.files,
        'audit_source_sha256':declaration['audit_source_sha256'],'original_scientific_source_sha256':m['source_sha256'],
        'limitations':['This wrapper was declared after the failed run and observed outcomes; it is not a prospective scientific rerun.',
            'The original scientific status remains failed. Passing saved-artifact verification does not repair finalization or relax any advancement gate.',
            'Declared artifact chronology is not an operating-system reference-access log.',
            'Optimization and TRAIN before/after logit diagnostics are not independently replayed.',
            'Raw periodicity replay covers only the predeclared4608 synthetic TRAIN windows; no full candidate extractor or natural PCM replay.',
            'Natural development is consumed; synthetic configuration freshness is not independent frequency support; no clinical or deployment qualification.']}
    write_once(output,result)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('declare','audit'))
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--declaration',type=Path);args=parser.parse_args()
    if args.action=='declare':
        value=declare(args.run,args.output);print(json.dumps({'posthoc':True,'declaration_sha256':base.sha(args.output)}));return
    require(args.declaration is not None,'An immutable posthoc declaration is required')
    run=base.inside(RUNS,args.run);output=base.inside(run,args.output);require(not output.exists(),'Refuses overwrite')
    require(not (run/STAGE_FOLDER).exists(),'Refuses an existing v2 stage folder')
    declaration=json.loads(args.declaration.read_text());seconds=int(declaration['maximum_forensic_seconds'])
    require(0<seconds<=1800,'Invalid forensic wall budget')
    def stop(signum,frame):raise TimeoutError('Combined independent audit wall budget reached')
    signal.signal(signal.SIGALRM,stop);signal.alarm(seconds);began=time.monotonic()
    try:
        result=audit_run(run,output,args.declaration)
        print(json.dumps({k:result[k] for k in ('status','pass_result','scientific_status','scientific_gate_pass','posthoc','numeric_comparisons','maximum_numeric_difference','elapsed_seconds','bound_artifacts')},indent=2,default=scalar))
    except BaseException as error:
        if not output.exists():
            write_once(output,{'schema':'phenometrix-exposure-posthoc-forensic-audit.v2','status':'failed','pass_result':False,
                'posthoc':True,'failure':repr(error),'scientific_status':'failed','scientific_status_changed':False,
                'declaration_sha256':base.sha(args.declaration),'audit_source_sha256':source_hashes(),
                'completed_stages':completed_stages(run/STAGE_FOLDER,base.sha(args.declaration)),
                'elapsed_seconds':time.monotonic()-began})
        raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
