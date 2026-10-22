"""Compare frozen checkpoint tensors and saved outputs; no forward passes/labels."""
from pathlib import Path
from io import BytesIO
import argparse
import hashlib,json
from datetime import datetime, timezone
import numpy as np
import torch
REPO=Path(__file__).resolve().parents[2];ROOT=REPO/'research/voice-prosody'
NEW=ROOT/'runs/harmonic-augmentation-v1';OLD=ROOT/'runs/harmonic-context-v1'
DEFAULT_OUTPUT=NEW/'control-audit.json';ledger={}
def read(path,expected=None):
 data=path.read_bytes();h=hashlib.sha256(data).hexdigest()
 if expected is not None and h!=expected:raise ValueError('Identity mismatch: '+str(path))
 name=str(path.relative_to(REPO))
 if name in ledger and ledger[name]!=h:raise ValueError('Changed during read: '+name)
 ledger[name]=h;return data

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
 output=parser.parse_args().output
 if output.exists():raise ValueError('Refuses to overwrite receipt')
 torch.set_num_threads(1)
 n=json.loads(read(NEW/'manifest.json'));o=json.loads(read(OLD/'prediction-freeze.json','ae26c900df1664057a14d54397b77363bfdcdb01564508156fc0a46b679c8f7e'))
 assert n['status']=='complete'
 rows={}
 for seed in (20261022,20261023,20261024):
  new=f'natural-only__seed{seed}';old=f'relative__acceptable-set__seed{seed}'
  ck={}
  for role,run,m,name in [('refitted',NEW,n,new),('prior',OLD,o,old)]:
   ck[role]=torch.load(BytesIO(read(run/'checkpoints'/(name+'.pt'),m['fits'][name]['weights_sha256'])),map_location='cpu',weights_only=True)['state_dict']
  assert set(ck['refitted'])==set(ck['prior']);tensors={}
  for key,a in ck['refitted'].items():
   b=ck['prior'][key];assert a.dtype==b.dtype and a.shape==b.shape
   assert torch.isfinite(a).all() and torch.isfinite(b).all()
   tensors[key]={'shape':list(a.shape),'dtype':str(a.dtype),'identical':torch.equal(a,b),'max_absolute_difference':float((a.double()-b.double()).abs().max()) if a.numel() else 0}
  arrays={}
  for role,run,m,name in [('refitted',NEW,n,new),('prior',OLD,o,old)]:
   path='predictions/'+name+'.npz'
   with np.load(BytesIO(read(run/path,m['prediction_sha256'][path])),allow_pickle=False) as z:arrays[role]={k:z[k].copy() for k in ('state','hz')}
  a,b=arrays['refitted'],arrays['prior'];assert a['state'].shape==b['state'].shape==a['hz'].shape==b['hz'].shape
  assert a['state'].dtype==b['state'].dtype and a['hz'].dtype==b['hz'].dtype
  both=np.isfinite(a['hz'])&np.isfinite(b['hz']);equal=(a['hz']==b['hz'])|(np.isnan(a['hz'])&np.isnan(b['hz']))
  rows[str(seed)]={'refitted_method':new,'prior_method':old,'initial_tensor_hash_identical':n['fits'][new]['initial_tensors_sha256']==o['fits'][old]['initial_tensors_sha256'],
    'tensors':tensors,'all_state_tensors_identical':all(t['identical'] for t in tensors.values()),
    'max_tensor_absolute_difference':max(t['max_absolute_difference'] for t in tensors.values()),
    'development_frames':len(a['state']),'state_mismatches':int(np.count_nonzero(a['state']!=b['state'])),
    'hz_mismatches_nan_equal':int(np.count_nonzero(~equal)),'hz_availability_mismatches':int(np.count_nonzero(np.isfinite(a['hz'])!=np.isfinite(b['hz']))),
    'jointly_finite_hz_frames':int(both.sum()),'max_finite_hz_absolute_difference':float(np.max(np.abs(a['hz'][both]-b['hz'][both]))) if both.any() else 0}
 for path,h in ledger.items():assert hashlib.sha256((REPO/path).read_bytes()).hexdigest()==h
 report={'schema':'phenometrix.augmentation-refitted-control-audit.v1','status':'complete','audited_at':datetime.now(timezone.utc).isoformat(),
   'scope':'Stored checkpoint tensors and frozen development states/Hz only; no inference, references, labels, fitting, or new waveforms.',
   'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seeds':rows,'bound_input_sha256':ledger,
   'all_three_controls_exactly_reproduce':all(r['initial_tensor_hash_identical'] and r['all_state_tensors_identical'] and r['state_mismatches']==r['hz_mismatches_nan_equal']==0 for r in rows.values()),
   'total_saved_development_frame_pairs':sum(r['development_frames'] for r in rows.values()),
   'limitations':['Comparison reproduces the baseline control numerically; it does not independently replay optimization or inference.',
                   'Prior prediction freeze is pinned by its historical SHA256. New complete-run manifest is bound as observed, and all checkpoint/prediction bytes are checked against each manifest.']}
 output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ('seeds','bound_input_sha256','limitations')}))
 for seed,r in rows.items():print(seed,json.dumps({k:v for k,v in r.items() if k!='tensors'}))
if __name__=='__main__':main()
