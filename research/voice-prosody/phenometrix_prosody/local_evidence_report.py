"""Render an offline, aggregate-only research report from frozen local evidence.

No generation, fitting, inference, reference trajectories or media decoding.
The report is a presentation of bound saved aggregates, not a new score auditor.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path

REPO=Path(__file__).resolve().parents[3]
RUNS=REPO/'research/voice-prosody/runs'
SCHEMA='phenometrix-local-evidence-report.v1'
GROUPS=('regression','consumed_harmonic','consumed_subharmonic','consumed_augmentation',
    'consumed_coverage','consumed_expansion','challenge')
LIMIT=16*1024*1024


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def integer(value,name):
    require(type(value) is int and value>=0,'Invalid integer '+name)
    return value


def support_counts(value):
    names=('frames','original_raw_covered','expanded_raw_covered','original_group_covered',
        'expanded_group_covered','new_frequency_added','original_grouping_lost','expanded_grouping_lost',
        'added_only','counterexample')
    require(set(value)==set(names),'Support aggregate fields differ')
    counts={k:integer(value[k],k) for k in names};n=counts['frames']
    require(all(v<=n for v in counts.values()),'Support count exceeds denominator')
    require(counts['original_group_covered']<=counts['original_raw_covered']<=counts['expanded_raw_covered']
        and counts['expanded_group_covered']<=counts['expanded_raw_covered'],'Coverage nesting differs')
    for prefix in ('original','expanded'):
        require(counts[prefix+'_grouping_lost']==counts[prefix+'_raw_covered']-counts[prefix+'_group_covered'],
            'Grouping loss disagrees with raw/group coverage')
    return {**counts,'original_raw_absent':n-counts['original_raw_covered'],
        'expanded_raw_absent':n-counts['expanded_raw_covered'],
        'expanded_group_absent':n-counts['expanded_group_covered']}


def support_view(report):
    require(report.get('schema')=='phenometrix-evidence-feasibility.v1','Unexpected feasibility report')
    require(type(report.get('feasibility',{}).get('pass_result')) is bool,'Missing support decision')
    rows=[]
    for row in report['per_case']:
        require(isinstance(row['id'],str) and isinstance(row['frequency_hz'],(int,float))
            and math.isfinite(row['frequency_hz']),'Invalid case metadata')
        meta={k:row[k] for k in ('case_index','id','frequency_hz','phase_index','gain_index','mixture_index')}
        for key in ('case_index','phase_index','gain_index','mixture_index'):integer(meta[key],key)
        rows.append({**meta,'regions':{'all':support_counts(row['coverage']),
            **{name:support_counts(counts) for name,counts in row['regions'].items()}}})
    require(len(rows)==report['case_count'] and len({r['id'] for r in rows})==len(rows),'Case inventory differs')
    coverage=support_counts(report['coverage'])
    require(sum(r['regions']['all']['frames'] for r in rows)==report['common_eligible_rows']==coverage['frames'],
        'Support denominator does not reconcile')
    for key in coverage:
        require(sum(r['regions']['all'][key] for r in rows)==coverage[key],'Support aggregate does not reconcile: '+key)
    for row in rows:
        for key in coverage:
            require(sum(r[key] for name,r in row['regions'].items() if name!='all')==row['regions']['all'][key],
                'Case regions do not reconcile: '+row['id']+'/'+key)
    return {'feasibility':report['feasibility'],'acquisition_rows':integer(report['acquisition_rows'],'acquisition_rows'),
        'common_eligible_rows':report['common_eligible_rows'],'case_count':report['case_count'],
        'coverage':coverage,'cases':rows,'schedule_count':len(report['schedule_exposure']),
        'meaning':'TRAIN support and exposure only. No model outcome is inferred from candidate coverage.'}


def model_region(score,coverage,method):
    """Keep outcome categories exact; ranking overlap is unknown unless identifiable."""
    n=integer(score['frames'],'frames');voiced=integer(score['voiced_outputs'],'voiced_outputs')
    abstained=integer(score['nonvoiced_or_unavailable'],'nonvoiced_or_unavailable')
    require(voiced+abstained==n,'Output states do not sum to all-frame denominator')
    rate=score['success_within_50c_all_frames']
    if n:
        require(isinstance(rate,(int,float)) and math.isfinite(rate) and 0<=rate<=1,'Invalid unconditional success rate')
        correct=round(rate*n)
        require(abs(rate*n-correct)<=1e-8 and correct<=voiced,'Success rate is not an integer count')
    else:
        require(rate is None,'Empty region must have unavailable rate');correct=0
    wrong=voiced-correct
    result={'frames':n,'correct_pitch':correct,'incorrect_pitch':wrong,'unvoiced':None,'unavailable':None,
        'nonvoiced_or_unavailable':abstained,'success_rate':rate,'raw_correct_candidate_absent':None,
        'grouping_loss':None,'grouped_correct_candidate_absent':None,'candidate_set_empty':None,
        'frozen_voicing_gate_positive':None,'ranking_error_lower_bound':None,'ranking_error_upper_bound':None}
    if coverage is None:return result
    require(coverage['frames']==n,'Candidate/output region denominators differ')
    states=coverage['methods'][method]
    require(set(states)=={'unvoiced','voiced','unavailable'} and all(type(x) is int and 0<=x<=n for x in states.values())
        and states['voiced']==voiced and states['unvoiced']+states['unavailable']==abstained,'Saved state counts differ')
    policy=coverage['policies']['expanded7'];raw=integer(policy['raw_covered'],'raw_covered');group=integer(policy['grouped_covered'],'grouped_covered')
    require(0<=group<=raw<=n and policy['grouping_lost']==raw-group,'Candidate coverage disagrees')
    require(correct<=group,'Correct outputs exceed grouped coverage')
    empty=integer(policy['candidate_unavailable'],'candidate_unavailable');gate=integer(coverage['original_voiced_gate_positive'],'voicing positive')
    require(empty<=n-group and gate<=n and states['unvoiced']==n-gate,'Availability/voicing counts disagree')
    # Aggregates do not identify the overlap of wrong outputs with covered frames.
    # Exact ranking failures are available when every frame has a correct candidate.
    result.update(unvoiced=states['unvoiced'],unavailable=states['unavailable'],raw_correct_candidate_absent=n-raw,
        grouping_loss=raw-group,grouped_correct_candidate_absent=n-group,candidate_set_empty=empty,
        frozen_voicing_gate_positive=gate,ranking_error_lower_bound=max(0,wrong-(n-group)),
        ranking_error_upper_bound=min(wrong,group-correct))
    return result


def model_view(metrics,inventory):
    require(metrics.get('schema')=='phenometrix-evidence-metrics.v1','Unexpected model metrics')
    require(set(inventory)==set(GROUPS),'Analytic inventory groups differ')
    rows=[]
    for group in GROUPS:
        require({r['id'] for r in inventory[group]}==set(metrics[group]),'Analytic aggregate inventory differs')
        for case_id,row in metrics[group].items():
            scenario=row['scenario'];cov=row.get('candidate_coverage',{})
            for method,values in row['methods'].items():
                regions={'all':values['all'],**values['regions']}
                for region,score in regions.items():
                    rows.append({'group':group,'id':case_id,'kind':row['kind'],'method':method,'region':region,
                        'initial_hz':scenario.get('initialHz'),'final_hz':scenario.get('finalHz'),
                        **model_region(score,cov.get(region),method)})
    natural={name:{'joint_error':r['macro']['joint_error'],'reference_voiced_success_rate':r['macro']['reference_voiced_success_rate']}
        for name,r in metrics['natural'].items()}
    return {'decision':metrics['decision'],'natural':natural,'cases':rows,
        'meaning':'All-frame, inclusive50-cent success. Candidate absence and output states overlap; they are not additive causes. Ranking bounds are aggregate constraints, not per-frame attribution.'}


class Bindings:
    def __init__(self,repo=None):self.repo=Path(REPO if repo is None else repo).resolve();self.hashes={};self.snapshots={}
    def bind(self,path,expected=None):
        path=Path(path).resolve();require(path.is_relative_to(self.repo) and path.is_file(),'Input escaped repository or is missing')
        digest=sha(path);name=str(path.relative_to(self.repo))
        require(expected is None or digest==expected,'Input hash differs: '+name)
        require(name not in self.hashes or self.hashes[name]==digest,'Input changed during report')
        self.hashes[name]=digest;return path
    def read(self,path,expected=None,snapshot=None):
        path=self.bind(path,expected);require(path.stat().st_size<=LIMIT,'Aggregate JSON exceeds16MiB')
        raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==self.hashes[str(path.relative_to(self.repo))],'Input changed during read')
        value=json.loads(raw,parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON '+value)))
        if snapshot:self.snapshots[snapshot]=raw
        return value
    def recheck(self):
        for name,digest in self.hashes.items():require(sha(self.repo/name)==digest,'Input changed before report completion')


def collect(preparation,study=None,repair_receipt=None,expected_repair_sha256=None):
    preparation=Path(preparation).resolve();require(preparation.is_relative_to(RUNS.resolve()),'Expected ignored local preparation')
    bound=Bindings();m=bound.read(preparation/'manifest.json',snapshot='feasibility-manifest.json')
    require(m.get('schema')=='phenometrix-evidence-feasibility.v1' and m.get('status')=='complete','Completed support preparation required')
    source=bound.read(REPO/m['source_freeze_path'],m['source_freeze_sha256'],snapshot='source-freeze.json')
    require(source.get('status')=='sources_frozen' and source.get('fits')=={},'Initial source freeze differs')
    for field in ('training_spec_sha256','challenge_spec_sha256','deadline_utc'):
        require(m.get(field)==source.get(field),'Preparation source identity differs')
    report=bound.read(REPO/m['report_path'],m['report_sha256'],snapshot='feasibility-report.json')
    support=support_view(report)
    require(type(m.get('feasibility_pass')) is bool and support['feasibility']['pass_result']==m['feasibility_pass'],'Support outcome differs')
    audit_path=preparation/'independent-audit.json';audit=None
    if audit_path.exists():
        receipt=bound.read(audit_path,snapshot='feasibility-audit.json')
        require(receipt.get('schema')=='phenometrix-evidence-feasibility-audit.v1' and receipt.get('pass') is True
            and receipt.get('preparation_manifest_sha256')==sha(preparation/'manifest.json')
            and receipt.get('report_sha256')==m['report_sha256'] and receipt.get('feasibility_pass')==m['feasibility_pass'],
            'Present feasibility audit is failed or does not bind these aggregates')
        audit={'pass':True,'sha256':sha(audit_path),'path':str(audit_path.relative_to(REPO))}
    model=None;evidence_count=0;study_metadata=None
    if study is not None:
        study=Path(study).resolve();require(study==preparation.parent,'Study and preparation must belong to the same run')
        sm=bound.read(study/'manifest.json',snapshot='study-manifest.json')
        require(sm.get('schema')=='phenometrix-evidence-experiment.v1' and sm.get('status')=='complete','Model report requires a completed study')
        require(m['feasibility_pass'] and sm.get('feasibility_manifest_sha256')==sha(preparation/'manifest.json'),
            'Completed model study lacks exact feasible preparation')
        frozen=bound.read(study/'prediction-freeze.json',snapshot='prediction-freeze.json')
        checkpoint=bound.read(study/'checkpoint-freeze.json',snapshot='checkpoint-freeze.json')
        require(frozen.get('status')=='predictions_frozen' and checkpoint.get('status')=='checkpoints_frozen'
            and sm['fits']==frozen['fits'] and sm['source_sha256']==frozen['source_sha256']==checkpoint['source_sha256']
            and sm['prediction_sha256']==frozen['prediction_sha256'],'Study freeze identities differ')
        expected={f'{arm}__seed{seed}' for arm in ('existing__off','existing__on','targeted__off','targeted__on') for seed in (20261022,20261023,20261024)}
        require(set(sm['fits'])==set(checkpoint['checkpoints'])==expected,'Exactly twelve frozen fits required')
        metrics=bound.read(study/'metrics.json',sm['metrics_sha256'],snapshot='metrics.json')
        invpath=study/'analytic-inventory.json';name=str(invpath.relative_to(REPO))
        inventory=bound.read(invpath,frozen['input_sha256'][name],snapshot='analytic-inventory.json')
        for rows in inventory.values():
            for row in rows:
                # Hash provenance only; never open NPZ arrays or decoded media.
                require(frozen['input_sha256'].get(row['evidence_path'])==row['evidence_sha256'],'Evidence absent from input freeze')
                bound.bind(REPO/row['evidence_path'],row['evidence_sha256']);evidence_count+=1
                require(frozen['prediction_sha256'].get(row['prediction_path'])==row['prediction_sha256'],'Analytic prediction binding differs')
        model=model_view(metrics,inventory)
        study_metadata={'status':sm['status'],'predesignated_candidate':sm['predesignated_candidate'],
            'sources_frozen_at':sm['sources_frozen_at'],'checkpoints_frozen_at':sm['checkpoints_frozen_at'],
            'predictions_frozen_at':sm['predictions_frozen_at'],'completed_at':sm['completed_at'],'fit_count':len(sm['fits'])}
    repair=None
    require((repair_receipt is None)==(expected_repair_sha256 is None),'Repair path and expected SHA256 must be supplied together')
    if repair_receipt is not None:
        require(isinstance(expected_repair_sha256,str) and len(expected_repair_sha256)==64 and all(c in '0123456789abcdef' for c in expected_repair_sha256),'Exact lowercase repair SHA256 required')
        repair=bound.read(repair_receipt,expected_repair_sha256,snapshot='repair-receipt.json')
        require(isinstance(repair,dict),'Repair receipt must be a JSON object')
    bound.recheck()
    payload={'repair_receipt':repair,'schema':SCHEMA,'created_at':datetime.now(timezone.utc).isoformat(),'support':support,'model':model,
        'study':study_metadata,'support_audit':audit,'evidence_artifacts_hash_verified':evidence_count,
        'source_freeze_sha256':m['source_freeze_sha256'],'training_spec_sha256':m['training_spec_sha256'],
        'challenge_spec_sha256':m['challenge_spec_sha256'],'inputs_sha256':bound.hashes,
        'claim_boundary':'Local research inspection only. No calibrated probability, clinical claim, promotion or production change. No raw media or natural reference trajectories are embedded.',
        'renderer_role':'Presentation with aggregate reconciliation and hash checks; independent scoring and inference audits remain separate.'}
    return payload,bound


PAGE=r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'; img-src data:; base-uri 'none'; form-action 'none'">
<title>PhenoMetrix · Evidence inspection</title>
<style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#162431;background:#f3f5f7;font-size:15px}*{box-sizing:border-box}body{margin:0}header{background:#102c42;color:#fff;padding:28px max(24px,calc((100vw - 1360px)/2))}header small{letter-spacing:.12em;text-transform:uppercase;color:#c3d6e6}h1{font-size:30px;letter-spacing:-.03em;margin:10px 0}header p{max-width:900px;color:#dce5ed;line-height:1.55}main{max-width:1410px;margin:auto;padding:24px}h2{font-size:21px;margin:0 0 12px}h3{font-size:16px}p{line-height:1.55}.panel{background:#fff;border:1px solid #dce2e8;border-radius:9px;padding:22px;margin-bottom:20px}.status{font-size:19px;font-weight:650}.warn{color:#914412}.good{color:#165f46}.muted{color:#5c6d7c}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:16px 0}.card{background:#f4f7fa;border-radius:6px;padding:15px}.card b{display:block;font-size:24px;font-variant-numeric:tabular-nums;margin-top:6px}.controls{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}label{font-size:13px;color:#4b5b69;display:flex;flex-direction:column;gap:6px}select,input{font:inherit;color:#162431;border:1px solid #bac6d1;border-radius:5px;padding:8px;background:white;min-width:130px}input{min-width:240px}.scroll{overflow:auto;max-height:620px;border:1px solid #e2e7ec;border-radius:5px}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}th,td{padding:10px 12px;text-align:right;border-bottom:1px solid #e5e9ed;white-space:nowrap}th{position:sticky;top:0;background:#edf2f6;font-weight:600;z-index:1}th:first-child,td:first-child{text-align:left}tbody tr:nth-child(even){background:#fafbfd}tbody tr:hover{background:#edf4fa}.tag{display:inline-block;font-size:12px;padding:3px 7px;border-radius:4px;background:#eaf0f5;color:#314d63}details{margin:12px 0}summary{cursor:pointer;font-weight:600}code,pre{font:12px ui-monospace,monospace;overflow-wrap:anywhere}pre{white-space:pre-wrap;max-height:360px;overflow:auto;background:#f4f7fa;padding:16px;border-radius:5px}.note{border-left:3px solid #b6c8d8;padding-left:14px;font-size:14px}.fail{background:#fff1e8!important}footer{font-size:13px;color:#617282;margin:20px 0}.empty{padding:18px;color:#617282}.subtle{font-size:13px;color:#617282}a{color:#1d5c8f}@media(max-width:650px){main{padding:12px}.panel{padding:15px}h1{font-size:25px}header{padding:22px}input,select{max-width:100%;min-width:110px}}
</style></head><body><header><small>PhenoMetrix / local research</small><h1>Where does the evidence run out?</h1><p>A saved, inspectable view of TRAIN coverage and fixed exposure. When completed model evidence is supplied, candidate support and output failures are shown separately. All calculations use the saved all-frame denominators.</p></header><main>
<section class="panel"><div id="status" class="status">__STATUS__</div><p class="muted">__CLAIM__</p><div class="cards" id="summary"></div><p id="audit" class="subtle"></p><noscript>This report contains embedded JSON and aggregate snapshots. Enable JavaScript locally to filter tables; no network access is required or allowed.</noscript></section>
<section class="panel"><h2>Feasibility before fitting</h2><p class="muted">Both targeted strata must meet every fixed support floor. Broad draws include all eligible rows and overlap both strata. Candidate coverage is not model accuracy.</p><div class="scroll"><table id="floors"></table></div><p id="sampling" class="subtle"></p></section>
<section class="panel"><h2>TRAIN support by complete configuration</h2><p class="note">“Absent” means no candidate within 50 cents. Grouping loss removes an otherwise correct raw candidate. These input facts do not establish a voicing or ranking failure. No training rows are excluded from the displayed eligibility denominator.</p>
<div class="controls"><label>Carrier (Hz)<select id="carrier"></select></label><label>Phase<select id="phase"></select></label><label>Gain<select id="gain"></select></label><label>After mixture<select id="mixture"></select></label><label>Support region<select id="supportRegion"></select></label><label>Case ID contains<input id="search" type="search" placeholder="Filter configuration IDs"></label></div>
<div class="cards" id="selection"></div><p id="supportCount" class="subtle"></p><div class="scroll"><table id="support"></table></div><p class="subtle">Before: exclusive end ≤39,840. Boundary: support crosses39,840. After: end−1,920 ≥39,840. Every eligible frame remains in its region.</p></section>
<section class="panel" id="models"><h2>Model outcomes</h2><p id="modelNotice" class="note"></p><div id="modelContents" hidden><h3>Natural DEV aggregates</h3><div class="scroll"><table id="natural"></table></div><p class="subtle">Consumed development data. Joint error and unconditional reference-voiced success are distinct measures; these results do not establish new-population or clinical validity.</p><h3>Analytic cases and failure categories</h3><div class="controls"><label>Method<select id="method"></select></label><label>Case family<select id="family"></select></label><label>Region<select id="modelRegion"></select></label></div><p id="modelCount" class="subtle"></p><div class="scroll"><table id="outcomes"></table></div><details><summary>How to read the categories</summary><p>Correct pitch, incorrect pitch, unvoiced and unavailable outputs partition all eligible frames when state counts are available. Raw candidate absence, grouping loss and an empty candidate set describe input support and may overlap output categories. They must not be summed as independent causes.</p><p>Ranking error bounds constrain incorrect voiced outputs that occurred while a correct grouped candidate was available. When the bounds differ, aggregate data cannot identify their exact overlap. The report does not label every wrong output as a ranking error. Logits and acoustic evidence are not calibrated correctness probabilities.</p></details><details><summary>Saved advancement decision</summary><pre id="decision"></pre></details></div></section>
<section class="panel" id="repairPanel" hidden><h2>Preserved engineering failure</h2><p class="note">This separately bound receipt preserves the failed attempt and any authorized restart. It is not a scientific gate result, and no failed attempt is silently replaced.</p><pre id="repairReceipt"></pre></section>
<section class="panel"><h2>Inspect the evidence</h2><p class="muted">The companion report-data.json is the exact embedded aggregate payload. input-snapshot/ contains only the bound JSON reports and manifests; source-snapshot/ contains this renderer. The report manifest binds their bytes and this HTML. Raw media, natural references, feature arrays and predictions are not copied into this report.</p><p id="identities" class="subtle"></p><details><summary>Input identities (SHA256)</summary><pre id="bindings"></pre></details><p class="note">Hash checks establish byte identity. This renderer reconciles saved aggregates; it does not replace independent arithmetic, inference or source-chronology audits. Model sections are absent unless a completed study and its prediction freeze were supplied.</p></section><footer id="footer"></footer></main>
<script id="report-data" type="application/json">__DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('report-data').textContent);
const el=id=>document.getElementById(id),fmt=n=>n===null||n===undefined?'Not assessed':Number(n).toLocaleString('en-US'),pct=n=>n===null?'Not assessed':(100*n).toFixed(3)+'%';
function text(id,value){el(id).textContent=value}
function cards(id,rows){el(id).replaceChildren(...rows.map(([label,value])=>{const p=document.createElement('div');p.className='card';p.append(document.createTextNode(label));const b=document.createElement('b');b.textContent=fmt(value);p.append(b);return p}))}
function table(id,headers,rows,failed=()=>false){const target=el(id);target.replaceChildren();const head=document.createElement('thead'),tr=document.createElement('tr');headers.forEach(h=>{const th=document.createElement('th');th.textContent=h;tr.append(th)});head.append(tr);target.append(head);const body=document.createElement('tbody');rows.forEach(row=>{const tr=document.createElement('tr');if(failed(row))tr.className='fail';row.forEach(value=>{const td=document.createElement('td');td.textContent=value;tr.append(td)});body.append(tr)});if(!rows.length){const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=headers.length;td.textContent='No matching records; no aggregate has been inferred.';tr.append(td);body.append(tr)}target.append(body)}
function options(id,values,all=true){const s=el(id);s.replaceChildren();if(all){const o=document.createElement('option');o.value='';o.textContent='All';s.append(o)}values.forEach(v=>{const o=document.createElement('option');o.value=String(v[0]??v);o.textContent=String(v[1]??v);s.append(o)})}
const support=data.support,passed=support.feasibility.pass_result;
el('status').className='status '+(passed?'good':'warn');
cards('summary',[['Complete configurations',support.case_count],['Acquired frames',support.acquisition_rows],['Eligible TRAIN frames',support.common_eligible_rows],['Model fits loaded',data.study?.fit_count??0]]);
text('audit',data.support_audit?'Independent feasibility audit passed and binds these saved aggregates.':'Independent feasibility audit not supplied. This presentation is not an audit pass.');
const floor=support.feasibility.floor,fr=[];Object.entries(support.feasibility.strata).forEach(([name,v])=>Object.entries(floor).forEach(([dimension,min])=>fr.push([name,dimension,fmt(v[dimension]),fmt(min),v.checks[dimension]?'Met':'FAILED'])));
table('floors',['Stratum','Required dimension','Observed','Minimum','Result'],fr,row=>row[4]==='FAILED');
text('sampling',passed?`${fmt(support.schedule_count)} fixed seed/epoch schedules are recorded. Targeted batches draw128 added-only,128 counterexample and256 broad rows, with case-balanced replacement; overlapping strata and duplicate draws remain explicit in the saved report.`:'Support floor failed. No training schedule or model conclusion is implied; dependent fitting and fresh evaluation remain forbidden.');
const unique=key=>[...new Set(support.cases.map(r=>r[key]))].sort((a,b)=>a-b);
options('carrier',unique('frequency_hz').map(v=>[v,v]));options('phase',unique('phase_index').map(v=>[v,'Phase '+v]));options('gain',unique('gain_index').map(v=>[v,['.035','.075'][v]??v]));options('mixture',unique('mixture_index').map(v=>[v,['(0,1,.5)','(.2,1,.5)'][v]??v]));
options('supportRegion',[['all','All eligible'],['before-observation-change','Before'],['observation-boundary','Boundary'],['after-observation-change','After']],false);
function drawSupport(){const choices=[['carrier','frequency_hz'],['phase','phase_index'],['gain','gain_index'],['mixture','mixture_index']];const region=el('supportRegion').value,q=el('search').value.toLowerCase();const rows=support.cases.filter(r=>choices.every(([id,key])=>el(id).value===''||String(r[key])===el(id).value)&&r.id.toLowerCase().includes(q));const total={frames:0,added_only:0,counterexample:0,expanded_raw_absent:0};rows.forEach(r=>Object.keys(total).forEach(k=>total[k]+=r.regions[region][k]));cards('selection',[['Selected eligible frames',total.frames],['Added-only rows',total.added_only],['Counterexample rows',total.counterexample],['Expanded raw absence',total.expanded_raw_absent]]);text('supportCount',`${rows.length} of${support.case_count} configurations · region: ${region}. Counts are unconditional within the selected eligible region.`);table('support',['Case','Hz','Phase','Gain','Mixture','Frames','Original raw absent','Expanded raw absent','Original grouping loss','Expanded grouping loss','Added-only','Counterexample'],rows.map(r=>{const c=r.regions[region];return[r.id,r.frequency_hz,r.phase_index,r.gain_index,r.mixture_index,fmt(c.frames),fmt(c.original_raw_absent),fmt(c.expanded_raw_absent),fmt(c.original_grouping_lost),fmt(c.expanded_grouping_lost),fmt(c.added_only),fmt(c.counterexample)]}))}
['carrier','phase','gain','mixture','supportRegion'].forEach(id=>el(id).addEventListener('change',drawSupport));el('search').addEventListener('input',drawSupport);drawSupport();
if(!data.model){text('modelNotice','No completed model study was supplied. Wrong ranking, voicing decisions, output unavailability, natural accuracy and advancement gates are not assessed. Candidate absence and grouping loss above are TRAIN input facts only.')}else{el('modelContents').hidden=false;text('modelNotice',data.model.meaning);table('natural',['Method','Joint error','Unconditional voiced success'],Object.entries(data.model.natural).map(([n,r])=>[n,pct(r.joint_error),pct(r.reference_voiced_success_rate)]));const rows=data.model.cases;options('method',[...new Set(rows.map(r=>r.method))].sort().map(v=>[v,v]),false);options('family',[...new Set(rows.map(r=>r.group))].map(v=>[v,v]));options('modelRegion',[...new Set(rows.map(r=>r.region))].map(v=>[v,v]),false);el('modelRegion').value='all';function drawModels(){const selected=rows.filter(r=>r.method===el('method').value&&(!el('family').value||r.group===el('family').value)&&r.region===el('modelRegion').value).sort((a,b)=>(a.success_rate??-1)-(b.success_rate??-1)||a.id.localeCompare(b.id));text('modelCount',`${selected.length} matching case regions. Sorted by unconditional success; every saved failed or zero-support region is retained.`);table('outcomes',['Family / case','Hz','Frames','Correct','Wrong pitch','Unvoiced','Unavailable','Raw absent','Grouping loss','Empty set','Ranking bounds','Success'],selected.map(r=>[r.group+' / '+r.id,r.initial_hz,fmt(r.frames),fmt(r.correct_pitch),fmt(r.incorrect_pitch),fmt(r.unvoiced),fmt(r.unavailable),fmt(r.raw_correct_candidate_absent),fmt(r.grouping_loss),fmt(r.candidate_set_empty),r.ranking_error_lower_bound===null?'Not assessed':r.ranking_error_lower_bound+'–'+r.ranking_error_upper_bound,pct(r.success_rate)]))}['method','family','modelRegion'].forEach(id=>el(id).addEventListener('change',drawModels));drawModels();text('decision',JSON.stringify(data.model.decision,null,2))}
if(data.repair_receipt){el('repairPanel').hidden=false;text('repairReceipt',JSON.stringify(data.repair_receipt,null,2))}text('bindings',JSON.stringify(data.inputs_sha256,null,2));text('identities',`Source freeze: ${data.source_freeze_sha256}\nTRAIN specification: ${data.training_spec_sha256}\nFresh specification: ${data.challenge_spec_sha256}\nEvidence artifacts hash-verified without embedding arrays: ${data.evidence_artifacts_hash_verified}`);text('footer',`Created ${data.created_at}. Offline report; network access disabled. ${data.renderer_role}`);
</script></body></html>'''


def render_html(payload):
    support=payload['support']
    status=('TRAIN support floors met; model outcomes not loaded' if support['feasibility']['pass_result']
        else 'No-go: fixed TRAIN support floor failed')
    if payload['model'] is not None:
        status=('All scientific gates passed; research qualification remains separate' if payload['model']['decision'].get('promising_for_next_study')
            else 'No-go: the predesignated model did not pass every scientific gate')
    encoded=json.dumps(payload,separators=(',',':'),ensure_ascii=False,allow_nan=False)
    # Raw-text script elements must not admit an embedded closing tag or HTML entity.
    encoded=encoded.replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    return PAGE.replace('__STATUS__',html.escape(status)).replace('__CLAIM__',html.escape(payload['claim_boundary'])).replace('__DATA__',encoded)


def write_report(output,payload,bound):
    output=Path(output).resolve()
    require(output.is_relative_to(RUNS.resolve()) and not output.exists() and output.parent.exists(),'Fresh ignored report directory required')
    require(all(not Path(REPO/name).resolve().is_relative_to(output) for name in bound.hashes),'Report output overlaps input evidence')
    bound.recheck();source=Path(__file__).resolve();source_digest=sha(source)
    payload={**payload,'renderer_source_sha256':source_digest}
    data=json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n';page=render_html(payload)
    require(len(data.encode())<=LIMIT and len(page.encode())<=2*LIMIT,'Aggregate report size ceiling exceeded')
    output.mkdir();(output/'input-snapshot').mkdir();(output/'source-snapshot').mkdir()
    manifest={'schema':SCHEMA,'status':'writing','created_at':payload['created_at'],
        'renderer_source_sha256':source_digest,'inputs_sha256':dict(bound.hashes),'artifacts_sha256':{},
        'raw_media_embedded':False,'natural_reference_trajectories_embedded':False,'network_required':False}
    try:
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        for name,raw in bound.snapshots.items():
            require(Path(name).name==name,'Snapshot path must be a basename')
            (output/'input-snapshot'/name).write_bytes(raw)
        (output/'source-snapshot'/'local_evidence_report.py').write_bytes(source.read_bytes())
        require(sha(output/'source-snapshot'/'local_evidence_report.py')==source_digest,'Renderer changed during snapshot')
        (output/'report-data.json').write_text(data);(output/'index.html').write_text(page)
        bound.recheck();require(sha(source)==source_digest,'Renderer changed during report')
        manifest['artifacts_sha256']={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='manifest.json'}
        manifest['status']='complete';(output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    except BaseException as error:
        manifest.update(status='failed',failure=repr(error));(output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');raise
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preparation',type=Path,required=True)
    parser.add_argument('--study',type=Path);parser.add_argument('--repair-receipt',type=Path);parser.add_argument('--expected-repair-sha256');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();payload,bound=collect(args.preparation,args.study,args.repair_receipt,args.expected_repair_sha256);manifest=write_report(args.output,payload,bound)
    print(json.dumps({'status':manifest['status'],'html':str(args.output.resolve()/'index.html'),
        'html_sha256':manifest['artifacts_sha256']['index.html'],'input_count':len(manifest['inputs_sha256'])}))


if __name__=='__main__':main()
