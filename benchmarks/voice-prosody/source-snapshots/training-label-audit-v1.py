from pathlib import Path
import json,hashlib
from collections import Counter,defaultdict
import numpy as np
R=Path(__file__).resolve().parents[3]; B=R/'research/voice-prosody/runs'; DATA=R/'research/voice-prosody/data/ptdb-tug'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def empty():return {'frames':0,'valid':0,'voiced':0,'unvoiced':0,'covered':0,'multiple_good':0,'target_slot':Counter(),'candidate_count':Counter(),'voiced_candidate_count':Counter(),'good_count':Counter(),'hz_bins':defaultdict(Counter),'slot_by_hz_bin':defaultdict(Counter),'extra_good_candidates_not_target':0,'near_duplicate_good_pairs_5c':0}
def add(s,p,refpath,fhash,rhash):
 assert sha(p)==fhash and sha(refpath)==rhash
 f=np.load(p,allow_pickle=False); x=f['features'];hz=f['candidate_hz'];mask=f['candidate_mask']; ends=f['reference_sample'];n=len(x);s['frames']+=n
 t=ends-960;common=(t>=3072)&(t+1536<=int(f['pcm_sample_count']))
 ri=(t-768)//480; rr=np.loadtxt(refpath); inrange=(ri>=0)&(ri<len(rr)); usable=common&inrange
 rv=np.zeros((n,4));rv[inrange]=rr[ri[inrange]]
 valid=usable&np.isfinite(rv).all(axis=1)&(((rv[:,1]==0)&(rv[:,0]==0))|((rv[:,1]==1)&(rv[:,0]>0)))
 voice=valid&(rv[:,1]==1)&(rv[:,0]>0)
 err=np.full(hz.shape,np.inf);np.log2(np.divide(hz,rv[:,0,None],out=np.ones(hz.shape),where=mask&voice[:,None]),out=err,where=mask&voice[:,None]);err=np.abs(1200*err);err[~mask|~voice[:,None]]=np.inf
 good=err<=50;count=good.sum(1);covered=voice&(count>0);target=err.argmin(1)
 s['valid']+=int(valid.sum());s['voiced']+=int(voice.sum());s['unvoiced']+=int((valid&~voice).sum());s['covered']+=int(covered.sum());s['multiple_good']+=int((count>1).sum());s['extra_good_candidates_not_target']+=int((count[covered]-1).sum())
 s['target_slot'].update(map(int,target[covered]));s['candidate_count'].update(map(int,mask.sum(1)[valid]));s['voiced_candidate_count'].update(map(int,mask.sum(1)[voice]));s['good_count'].update(map(int,count[voice]))
 for low,hi in [(0,100),(100,200),(200,250),(250,300),(300,400),(400,500),(500,701),(701,10000)]:
  sel=voice&(rv[:,0]>=low)&(rv[:,0]<hi);k=f'[{low},{hi})';s['hz_bins'][k].update(voiced=int(sel.sum()),covered=int((sel&covered).sum()),multiple_good=int((sel&(count>1)).sum()));s['slot_by_hz_bin'][k].update(map(int,target[sel&covered]))
 for a in range(6):
  for b in range(a+1,6):
   sel=good[:,a]&good[:,b];s['near_duplicate_good_pairs_5c']+=int((np.abs(1200*np.log2(hz[sel,a]/hz[sel,b]))<=5).sum())
 return f
m=json.loads((B/'delayed-train-features/manifest.json').read_text()); ac=json.loads((B/'delayed-train-features/acquisition-manifest.json').read_text());refs={r['recording_id']:r for r in ac['files'] if r['role']=='ref' and r['split']=='train'}
natural=empty()
for row in m['recordings']:
 rr=refs[row['source_recording_id']];add(natural,B/'delayed-train-features/features'/(row['recording_id']+'.npz'),DATA/rr['path'],row['feature_sha256'],rr['sha256'])
syn=empty();sm=json.loads((B/'synthetic-train-curriculum/manifest.json').read_text())
for r in sm['records']:add(syn,B/'synthetic-train-curriculum'/r['feature_path'],B/'synthetic-train-curriculum'/r['reference_path'],r['feature_sha256'],r['reference_sha256'])
print(json.dumps({'natural':natural,'synthetic_unique':syn,'note':'Synthetic each repeated4; disjoint synthetic identities, not independent participants. Counts use train only; no dev or sealed outcomes.'},indent=2))
