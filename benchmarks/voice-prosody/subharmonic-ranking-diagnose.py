"""Reproduce the historical 337.1-Hz and TRAIN-only ranking diagnosis.

Requires the unchanged ignored research artifacts and installed research .venv.
This portable entry point derives from the exact executed source whose SHA256 is
e1fe86fbfd7dd4d204efcfede1b78318523e9b8330117be2a9b311a8ed1d4397. Only repository/output
path handling and the historical prediction-freeze pin are strengthened here.
No fitting, waveform generation, development references or test access occurs.
Run from the repository root with a NEW output directory under research runs:

research/voice-prosody/.venv/bin/python benchmarks/voice-prosody/subharmonic-ranking-diagnose.py --output research/voice-prosody/runs/subharmonic-ranking-diagnosis-reproduction-v1
"""
from pathlib import Path
import sys,json,hashlib,time
from datetime import datetime,timezone
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[2];ROOT=REPO/'research/voice-prosody';sys.path.insert(0,str(ROOT))
from phenometrix_prosody import selector_experiment as old
from phenometrix_prosody.harmonic_selector_model import HarmonicSelector
from phenometrix_prosody.selector_model import prepare_selector_batch,build_pitch_targets
from phenometrix_prosody.audit_harmonic_inference import original_probability
from phenometrix_prosody.model import ModelConfig
from io import BytesIO

OUT=ROOT/'runs/subharmonic-diagnosis-v1';RUN=ROOT/'runs/harmonic-context-v1'
FACTOR=(2.,3.,4.,.5,1/3,.25);SEEDS=(20261022,20261023,20261024)
ledger={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p,expected=None):
 p=Path(p);data=p.read_bytes();h=hashlib.sha256(data).hexdigest()
 if expected is not None and h!=expected:raise ValueError('Identitychanged:'+str(p))
 ledger[str(p.relative_to(REPO))]=h;return data
def stats(x):
 x=np.asarray(x);x=x[np.isfinite(x)]
 return {'n':len(x),'min':float(x.min()),'median':float(np.median(x)),'max':float(x.max())} if len(x) else {'n':0,'min':None,'median':None,'max':None}
def npz(p,h):
 with np.load(BytesIO(read(p,h)),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def alias_masks(hz,mask,acceptable):
 negative=mask&~acceptable;good=acceptable[:,None,:];pairs=negative[:,:,None]&good
 safe=np.where(mask,hz,1.)
 cents=1200*np.log2(safe[:,:,None]/safe[:,None,:]);by={}
 for f in FACTOR:by[str(f)]=pairs&(np.abs(cents-1200*np.log2(f))<=50)
 union=np.any(np.stack([v.any(axis=2) for v in by.values()]),axis=0)
 return negative,union,by
def forward_details(model,batch):
 x=batch.features;mask=batch.candidate_mask;z=model._standardized_candidates(batch)
 embedding=torch.tanh(model.candidate_encoder(z));mi=model.message_inputs(batch)
 messages=torch.tanh(model.message_encoder(mi));edges=mask.unsqueeze(-1)&mask.unsqueeze(-2)
 scores=torch.where(edges[...,None],messages,torch.full_like(messages,-torch.inf))
 context=scores.max(dim=-2).values;context=torch.where(mask[...,None],context,torch.zeros_like(context))
 glob=((x[...,48:]-model.global_mean)/model.global_scale).unsqueeze(-2).expand(len(x),6,9)
 logits=model.shared_head(torch.tanh(model.shared_hidden(torch.cat((embedding,context,glob),dim=-1)))).squeeze(-1)
 logits=torch.where(mask,logits,torch.full_like(logits,-1e9))
 return logits.numpy(),context.numpy(),scores.argmax(dim=-2).numpy()
def summarize_ranking(logits,mask,acceptable,alias):
 chosen=np.argmax(np.where(mask,logits,-np.inf),axis=1);row=np.arange(len(chosen))
 good=acceptable[row,chosen];wrong=~good;bad=mask&~acceptable
 bestgood=np.max(np.where(acceptable,logits,-np.inf),axis=1)
 bestbad=np.max(np.where(bad,logits,-np.inf),axis=1)
 bestalias=np.max(np.where(alias,logits,-np.inf),axis=1)
 return {'rows':len(chosen),'wrong_selected':int(wrong.sum()),'selected_wrong_alias':int((wrong&alias[row,chosen]).sum()),
  'nonalias_wrong':int((wrong&~alias[row,chosen]).sum()),'good_minus_bestnegative':stats(bestgood-bestbad),
  'good_minus_bestalias':stats(bestgood-bestalias),'all_negative_margin_below1_rows':int((bad.any(axis=1)&((bestgood-bestbad)<1)).sum()),
  'alias_margin_below1_rows':int((alias.any(axis=1)&((bestgood-bestalias)<1)).sum())}

def main():
 if (OUT/'diagnosis.json').exists():raise ValueError('Nooverwrite')
 torch.set_num_threads(1);began=time.monotonic();m=json.loads(read(RUN/'prediction-freeze.json','ae26c900df1664057a14d54397b77363bfdcdb01564508156fc0a46b679c8f7e'))
 for name,h in m['source_sha256'].items():read(REPO/name,h)
 models={}
 for seed in SEEDS:
  name=f'relative__acceptable-set__seed{seed}';ck=torch.load(BytesIO(read(RUN/'checkpoints'/(name+'.pt'),m['fits'][name]['weights_sha256'])),map_location='cpu',weights_only=True)
  model=HarmonicSelector('relative','acceptable-set');model.load_state_dict(ck['state_dict']);model.eval();models[seed]=model
 op=ROOT/'runs/delayed-mlp-seed20261022/weights.pt';ow=torch.load(BytesIO(read(op,m['frozen_original_weights_sha256'])),map_location='cpu',weights_only=True)['state_dict']
 challenge=json.loads(read(RUN/'challenge/manifest.json',m['input_sha256'][str(RUN/'challenge/manifest.json')]))
 report={'schema':'phenometrix-subharmonic-training-diagnosis.v1','started_at':datetime.now(timezone.utc).isoformat(),
 'scope':'existing337.1challenge+TRAIN-onlylabels;nofit/newPCM/devrefs','alias_definition':'valid nonacceptable emitted candidate within50c of factor times ANY acceptable emitted candidate',
 'factors':list(FACTOR),'challenge':{},'train':{},'models':{str(s):m['fits'][f'relative__acceptable-set__seed{s}']['weights_sha256'] for s in SEEDS}}
 for record in challenge['records']:
  if not record['id'].startswith('harmonic-balance-f3-'):continue
  identity=record['id'];f=npz(RUN/'challenge'/record['feature_path'],record['feature_sha256']);b=prepare_selector_batch(f['features'],f['candidate_hz'],f['candidate_mask'],representation='shared-candidates')
  end=f['reference_sample'];target=end-960;post=(end-1920>=39840)&(target>=3072)&(target+1536<=int(f['pcm_sample_count']))
  labels=build_pitch_targets(b,np.full(len(end),337.1),np.ones(len(end),bool),np.ones(len(end),bool));acceptable=labels.acceptable.numpy();mask=b.candidate_mask.numpy();hz=b.candidate_hz.numpy()
  negative,alias,factor=alias_masks(hz,mask,acceptable)
  predpath='predictions/challenge-'+identity+'.npz';saved=npz(RUN/predpath,m['prediction_sha256'][predpath]);indices=np.flatnonzero(post);center=int(indices[len(indices)//2])
  with torch.inference_mode():p=original_probability(ow,f['features'])
  item={'post_frames':int(post.sum()),'post_correct_candidate_coverage':int(labels.covered.numpy()[post].sum()),
   'post_original_voicing_probability':stats(p[post]),'representative_frame':{'index':center,'target_sample':int(target[center]),'support_end':int(end[center]),
    'raw_hz':f['candidate_hz'][center].tolist(),'raw_feature_blocks':f['features'][center,:48].reshape(6,8).tolist(),
    'grouped_hz':hz[center].tolist(),'grouped_blocks':b.features[center,:48].reshape(6,8).tolist(),'globals':b.features[center,48:].tolist()},'seeds':{}}
  for seed,model in models.items():
   with torch.inference_mode():logits,context,winners=forward_details(model,b)
   chosen=np.argmax(np.where(mask,logits,-np.inf),axis=1);estimate=hz[np.arange(len(end)),chosen];estimate[p<.5]=np.nan
   key=f'relative__acceptable-set__seed{seed}';oldhz=saved[key+'__hz'];assert np.all((estimate==oldhz)|(np.isnan(estimate)&np.isnan(oldhz)))
   rank=summarize_ranking(logits[post],mask[post],acceptable[post],alias[post]);bestgood=np.argmax(np.where(acceptable[center],logits[center],-np.inf));bestalias=np.argmax(np.where(alias[center],logits[center],-np.inf))
   rank['center']={'logits':logits[center].tolist(),'selected_hz':float(hz[center,chosen[center]]),'best_correct_index':int(bestgood),'best_alias_index':int(bestalias),
    'correct_minus_alias_logit':float(logits[center,bestgood]-logits[center,bestalias]),'correct_alias_context_l2':float(np.linalg.norm(context[center,bestgood]-context[center,bestalias])),
    'correct_context_neighbor_channel_counts':np.bincount(winners[center,bestgood],minlength=6).tolist(),'alias_context_neighbor_channel_counts':np.bincount(winners[center,bestalias],minlength=6).tolist()}
   different=0
   for i in indices:
    ws=winners[i,mask[i]];different+=int(np.any(ws!=ws[:1],axis=0).sum())
   rank['query_dependent_argmax_channel_occurrences']=different;rank['post_channels_checked']=int(post.sum()*32)
   original_mode=model.context_mode;model.context_mode='absolute'
   with torch.inference_mode():zero_logits=model(b).numpy()
   model.context_mode=original_mode
   rank['counterfactual_zero_ratio_same_weights']=summarize_ranking(zero_logits[post],mask[post],acceptable[post],alias[post])
   item['seeds'][str(seed)]=rank
  report['challenge'][identity]=item
 print(json.dumps({'challenge':report['challenge']},allow_nan=True),flush=True)
 # Only training inputs and training references enter the following section.
 train=old.load_inputs('train');lab=old.load_labels(train);raw=train['raw']
 ledger.update({str(Path(k).relative_to(REPO)):v for k,v in (train['hashes']|lab['hashes']).items()})
 b=prepare_selector_batch(raw['features'],raw['candidate_hz'],raw['candidate_mask'],representation='shared-candidates')
 targets=build_pitch_targets(b,lab['hz'],lab['voiced'],lab['valid']);ix=torch.nonzero(targets.covered).reshape(-1);subset=b.index_select(ix);a=targets.acceptable[ix].numpy();mask=subset.candidate_mask.numpy();hz=subset.candidate_hz.numpy()
 negative,alias,by=alias_masks(hz,mask,a);ref=lab['hz'][ix.numpy()]
 def counts(select):
  return {'covered_rows':int(select.sum()),'any_negative_rows':int((select&negative.any(axis=1)).sum()),'alias_rows':int((select&alias.any(axis=1)).sum()),
   'negative_candidate_count':int(negative[select].sum()),'alias_candidate_count':int(alias[select].sum()),
   'acceptable_count_histogram':{str(i):int((select&(a.sum(axis=1)==i)).sum()) for i in range(1,7)},
   'candidate_count_histogram':{str(i):int((select&(mask.sum(axis=1)==i)).sum()) for i in range(1,7)},
   'factors':{f:{'rows':int((select&v.any(axis=(1,2))).sum()),'negative_candidates':int(v[select].any(axis=2).sum()),'pairs':int(v[select].sum())} for f,v in by.items()}}
 bins={'below100':ref<100,'100to200':(ref>=100)&(ref<200),'200to300':(ref>=200)&(ref<300),'300plus':ref>=300}
 report['train']={'acquired_frames':len(raw['features']),'valid_voiced_frames':int((lab['valid']&lab['voiced']).sum()),'all':counts(np.ones(len(ref),bool)),
  'f0_strata':{name:counts(select) for name,select in bins.items()},'frozen_model_rankings':{}}
 for seed,model in models.items():
  with torch.inference_mode():logits=np.concatenate([model(subset.index_select(torch.arange(start,min(start+4096,len(ref))))).numpy() for start in range(0,len(ref),4096)])
  report['train']['frozen_model_rankings'][str(seed)]={'all':summarize_ranking(logits,mask,a,alias),
   'f0_strata':{name:summarize_ranking(logits[select],mask[select],a[select],alias[select]) for name,select in bins.items()}}
 print(json.dumps({'train':report['train']},allow_nan=False),flush=True)
 for name,h in ledger.items():assert sha(REPO/name)==h
 report['source_sha256']=sha(Path(__file__));report['input_sha256']=ledger;report['elapsed_seconds']=time.monotonic()-began
 # Missing candidate metadata uses null rather than nonstandard JSON NaN.
 def clean(v):
  if isinstance(v,float) and not np.isfinite(v):return None
  if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
  if isinstance(v,list):return [clean(x) for x in v]
  return v
 (OUT/'diagnosis.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n')
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args();OUT=args.output.resolve();runs=(ROOT/'runs').resolve()
 if not OUT.is_relative_to(runs) or OUT==runs:parser.error('Output must be a new subdirectory of research/voice-prosody/runs')
 if OUT.exists():parser.error('Refusing to overwrite an existing output directory')
 OUT.mkdir(parents=True);main()
