"""Read-only frozen synthetic/checkpoint diagnosis; outputs stay in this ignored folder."""
from pathlib import Path
from io import BytesIO
import json,gzip,hashlib,datetime
import numpy as np
import torch
from phenometrix_prosody.selector_model import SelectorConfig,PitchSelector,prepare_selector_batch
ROOT=Path(__file__).resolve().parents[4];RUNS=ROOT/'research/voice-prosody/runs';OUT=Path(__file__).resolve().parent
ledger={}
def read(path,expected=None):
 p=Path(path).resolve();assert p.is_relative_to(ROOT)
 assert not any(k in str(p) for k in ['sealed-test','sealed_test','hybrid-validation'])
 raw=p.read_bytes();sha=hashlib.sha256(raw).hexdigest();assert expected is None or sha==expected,str(p)
 ledger[str(p.relative_to(ROOT))]=sha;return raw

def loadz(path,expected=None):
 with np.load(BytesIO(read(path,expected)),allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def stats(values):
 values=np.asarray(values);finite=values[np.isfinite(values)]
 return {'min':float(np.min(finite)) if len(finite) else None,'median':float(np.median(finite)) if len(finite) else None,'max':float(np.max(finite)) if len(finite) else None,'finite_count':int(len(finite)),'nonfinite_unopposed_count':int(len(values)-len(finite))}
def finite_lists(x):return [None if not np.isfinite(v) else float(v) for v in x]

read(__file__)
manifest=json.loads(read(RUNS/'selector-factorial-v1/manifest.json'))
read(RUNS/'selector-factorial-v1/checkpoint-freeze.json');read(RUNS/'selector-factorial-v1/prediction-freeze.json')
read(RUNS/'selector-factorial-v1/metrics.json',manifest['metrics_sha256'])
for source in ['selector_model.py','evaluate_synthetic.py','model.py']:
 path=ROOT/'research/voice-prosody/phenometrix_prosody'/source;raw=read(path,manifest['source_sha256'][str(path.relative_to(ROOT))])
 saved=OUT/'source-snapshot'/source;saved.parent.mkdir(exist_ok=True)
 if saved.exists():assert saved.read_bytes()==raw
 else:saved.write_bytes(raw)
read(ROOT/'research/voice-prosody/phenometrix_prosody/__init__.py')
torch.set_num_threads(1)
models={}
for objective in ['nearest','acceptable-set']:
 for seed in [20261022,20261023,20261024]:
  key=f'shared-candidates__{objective}__seed{seed}';sha=manifest['fits'][key]['weights_sha256']
  saved=torch.load(BytesIO(read(RUNS/'selector-factorial-v1/checkpoints'/f'{key}.pt',sha)),weights_only=True,map_location='cpu')
  model=PitchSelector(SelectorConfig(**saved['config']));model.load_state_dict(saved['state_dict']);model.eval();models[key]=model
results={};checks=0
for kind in ['constant','observation-change']:
 for hz in [100,200,300]:
  name=f'{kind}-{hz}';path=RUNS/'delayed-known-truth/features'/f'{name}.npz';f=loadz(path,manifest['input_sha256'][str(path)])
  p=RUNS/'delayed-known-truth/predictions'/f'{name}.json.gz';old=json.loads(gzip.decompress(read(p,manifest['input_sha256'][str(p)])))
  probability=np.array([r['voicing_probability'] for r in old['predictions']['mlp']]);end=f['reference_sample'];target=end-960
  n=int(f['pcm_sample_count']);eligible=(target>=3072)&(target+1536<=n)
  batch=prepare_selector_batch(f['features'],f['candidate_hz'],f['candidate_mask'],representation='shared-candidates')
  grouped=batch.candidate_hz.numpy();mask=batch.candidate_mask.numpy();correct=mask&(np.abs(1200*np.log2(grouped/hz))<=50)
  relative=f'predictions/synthetic-{name}.npz';frozen=loadz(RUNS/'selector-factorial-v1'/relative,manifest['prediction_sha256'][relative])
  regions={'all':eligible}
  if kind=='observation-change':regions.update(before=eligible&(end<28800),boundary=eligible&(end>=28800)&(end<30720),after=eligible&(end>=30720))
  result={'source_features_sha256':ledger[str(path.relative_to(ROOT))],'target_hz':hz,'reference_is_analytic':True,'regions':{},'models':{}}
  for region,select in regions.items():
   result['regions'][region]={'frames':int(select.sum()),'correct_group_present':int(correct[select].any(axis=1).sum()),'original_voicing_positive':int((probability[select]>=.5).sum()),'group_count':stats(mask[select].sum(axis=1))}
  for key,m in models.items():
   with torch.no_grad():logits=m(batch).numpy()
   choices=np.argmax(logits,axis=1);selected=grouped[np.arange(len(choices)),choices]
   states=np.where(probability<.5,0,np.where(mask.any(axis=1),1,2));selected=np.where(states==1,selected,np.nan)
   assert np.array_equal(states,frozen[key+'__state']) and np.allclose(selected,frozen[key+'__hz'],atol=0,rtol=0,equal_nan=True),key
   checks+=len(end)
   margins=np.max(np.where(correct,logits,-np.inf),axis=1)-np.max(np.where(mask&~correct,logits,-np.inf),axis=1)
   per={}
   for region,select in regions.items():
    good=(states==1)&(np.abs(1200*np.log2(selected/hz))<=50)
    picks,counts=np.unique(np.round(selected[select],6),return_counts=True)
    per[region]={'correct_frames':int(good[select].sum()),'frames':int(select.sum()),'correct_vs_wrong_logit_margin':stats(margins[select]),'selected_hz_counts':dict(zip(map(str,picks),map(int,counts)))}
   result['models'][key]=per
  if name=='observation-change-300':
   snapshots=[]
   for desired in [596,606,616,626,636,646,656,806,1206]:
    i=int(np.argmin(abs(end/48-desired)));valid=mask[i];x=batch.features[i,:48].reshape(6,8)
    sample={'availability_ms':float(end[i]/48),'target_ms':float(target[i]/48),'original_p_voiced':float(probability[i]),'raw_candidate_hz':finite_lists(f['candidate_hz'][i]),'group_hz':grouped[i,valid].tolist(),'group_members':batch.member_mask[i,valid].int().tolist(),'candidate_evidence':x[valid].tolist(),'global_evidence':batch.features[i,48:].tolist(),'models':{}}
    for key,m in models.items():
     with torch.no_grad():
      row=batch.index_select(np.array([i]));l=m(row)[0].numpy();cz=((x-m.candidate_mean)/m.candidate_scale)[valid]
      gz=(row.features[0,48:]-m.global_mean)/m.global_scale
      embedding=torch.tanh(m.candidate_encoder(cz))
     sample['models'][key]={'logits':l[valid].tolist(),'correct_vs_wrong_margin':float(l[correct[i]].max()-l[valid&~correct[i]].max()),'candidate_z':cz.tolist(),'global_z':gz.tolist(),'candidate_embedding_abs_above_099_fraction':float((embedding.abs()>.99).float().mean())}
    snapshots.append(sample)
   result['transition_snapshots']=snapshots
   m=models['shared-candidates__acceptable-set__seed20261023'];result['train_scaling']={k:getattr(m,k).tolist() for k in ['candidate_mean','candidate_scale','global_mean','global_scale']}
  results[name]=result
for path,sha in ledger.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha
output={'schema':'phenometrix-harmonic-balance-frozen-diagnosis.v1','status':'complete','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_fitting':True,'no_reserved_content':True,'cpu_prediction_frame_checks':checks,'fixtures':results,'opened_sha256':ledger,'evidence_columns':['valid','log2_hz_over_50','autocorrelation','amdf','zero_crossing','acf40_refined','acf20_refined','maximum'],'global_columns':['log_rms','delta_log_rms','dc_offset','clipped_fraction','spectral_flux','legacy_confidence','legacy_agreement','group_count_over6','history_present'],'limits':['This diagnoses six existing analytic regression fixtures only.','Training scaling z-scores are descriptive; they do not establish joint distribution support.','Six distinct frozen checkpoints are compared; no causal attribution to loss alone is possible.','No parameters, features, thresholds, candidates, references or gates are changed.']}
p=OUT/'diagnosis.json';assert not p.exists();p.write_text(json.dumps(output,separators=(',',':'),allow_nan=False)+'\n')
print(json.dumps({'status':'complete','output':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'cpu_prediction_frame_checks':checks,'input_files':len(ledger)}))
