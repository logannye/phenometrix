"""Input-only, fixed-slot full40 candidate replacement; no references or models.

The original v2 layout is retained solely for frozen-selector compatibility.
Every consuming run must separately bind POLICY; these are not original v2 bytes.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import time
import numpy as np
from .model import FEATURE_NAMES_V2

REPO = Path(__file__).resolve().parents[3]
KERNEL = REPO / 'research/voice-prosody/coverage_peaks.ts'
POLICY = 'shortest-admitted-full40-v1'
MAX_SAMPLES, MAX_ENDS = 48000 * 120, 12000
MAX_RESPONSE_BYTES, MAX_RSS_BYTES = 8 * 1024 * 1024, 2 * 1024**3
FEATURE_KEYS = frozenset(('features','candidate_hz','candidate_mask','reference_sample',
                          'feature_names','sample_rate_hz','pcm_sample_count'))


def source_hashes():
    paths = (Path(__file__), KERNEL, Path(__file__).with_name('model.py'),
             REPO/'apps/capture-web/src/dsp-primitives.ts', REPO/'apps/capture-web/src/voice-dsp.ts',
             REPO/'apps/capture-web/src/voice-pitch-features.ts', REPO/'apps/capture-web/src/voice-pitch-features-v2.ts')
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


class CoveragePeakWorker:
    """One bounded persistent Node process; deadlines fail closed and terminate it."""
    def __init__(self, *, timeout_seconds=120.):
        if not np.isfinite(timeout_seconds) or timeout_seconds <= 0: raise ValueError('Positive finite timeout required')
        self.timeout_seconds = float(timeout_seconds)
        self.process = None
        self.peak_rss_bytes = 0
        self._request_id = 0
        self._buffer = bytearray()

    def __enter__(self):
        if self.process is not None: raise ValueError('Worker already entered')
        self.process = subprocess.Popen(['node','--import','tsx',str(KERNEL)],
            cwd=REPO/'apps/capture-web', stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0)
        return self

    def close(self):
        p, self.process = self.process, None
        if p is None: return
        if p.poll() is None:
            p.terminate()
            try: p.wait(timeout=2)
            except subprocess.TimeoutExpired: p.kill(); p.wait(timeout=2)
        for stream in (p.stdin,p.stdout,p.stderr):
            if stream is not None: stream.close()
        self._buffer.clear()

    def __exit__(self, *_): self.close()

    def _response(self):
        deadline = time.monotonic() + self.timeout_seconds
        while b'\n' not in self._buffer:
            remaining = deadline-time.monotonic()
            if remaining <= 0: raise TimeoutError('Candidate worker response deadline exceeded')
            ready, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not ready: raise TimeoutError('Candidate worker response deadline exceeded')
            block = os.read(self.process.stdout.fileno(), 65536)
            if not block:
                ready_err, _, _ = select.select([self.process.stderr], [], [], 0)
                detail = os.read(self.process.stderr.fileno(),4096).decode(errors='replace') if ready_err else ''
                raise RuntimeError('Candidate worker ended without response: '+detail)
            self._buffer.extend(block)
            if len(self._buffer) > MAX_RESPONSE_BYTES: raise ValueError('Candidate worker response exceeds bound')
        line, _, rest = self._buffer.partition(b'\n'); self._buffer = bytearray(rest)
        return json.loads(line)

    def inspect(self, pcm_path, ends):
        if self.process is None: raise ValueError('Worker must be entered')
        ends = np.asarray(ends)
        if ends.ndim != 1 or ends.dtype.kind not in 'iu' or len(ends)>MAX_ENDS or np.any(np.diff(ends.astype(np.int64))<=0):
            raise ValueError('Increasing integer endpoint vector required')
        path = Path(pcm_path).resolve()
        if path.suffix != '.f32le': raise ValueError('Little-endian float32 PCM file required')
        size = path.stat().st_size
        if size==0 or size%4 or size//4>MAX_SAMPLES: raise ValueError('PCM size exceeds bound')
        raw = path.read_bytes()
        if not raw or len(raw)%4 or len(raw)//4>MAX_SAMPLES: raise ValueError('PCM size exceeds bound')
        count = len(raw)//4
        if np.any(ends<1920) or np.any(ends>count): raise ValueError('Complete-support endpoints required')
        expected_sha = hashlib.sha256(raw).hexdigest()
        self._request_id += 1
        request = {'id':self._request_id,'pcm_path':str(path),'sample_rate_hz':48000,'ends':ends.tolist()}
        try:
            self.process.stdin.write((json.dumps(request,separators=(',',':'))+'\n').encode())
            self.process.stdin.flush()
            value = self._response()
            if value.get('id') != self._request_id or value.get('ok') is not True:
                raise ValueError('Candidate worker rejected input: '+str(value.get('error','invalid response')))
            if (value.get('policy')!=POLICY or value.get('pcm_sha256')!=expected_sha or value.get('pcm_sample_count')!=count
                    or value.get('ends')!=ends.tolist() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha):
                raise ValueError('Candidate worker input identity changed')
            rss = value.get('rss_bytes')
            if type(rss) is not int or not 0<rss<=MAX_RSS_BYTES: raise ValueError('Candidate worker RSS exceeds bound')
            self.peak_rss_bytes = max(self.peak_rss_bytes,rss)
            result = {'pcm_sha256':expected_sha,'pcm_sample_count':count,'ends':ends.copy(),'rss_bytes':rss}
            for name, key in (('old','old'),('new','revised')):
                peaks = value.get(key)
                if not isinstance(peaks,list) or len(peaks)!=len(ends): raise ValueError('Invalid peak inventory')
                hz=np.full(len(ends),np.nan); confidence=np.zeros(len(ends)); logs=np.zeros(len(ends))
                for i,p in enumerate(peaks):
                    if p is None: continue
                    if (set(p)!= {'hz','confidence','lagSamples','log2HzOver50'} or
                            not all(isinstance(v,(int,float)) and np.isfinite(v) for v in p.values()) or
                            not 50<=p['hz']<=700 or not .35<=p['confidence']<=1 or
                            p['hz']!=8000/p['lagSamples']): raise ValueError('Invalid peak evidence')
                    hz[i],confidence[i],logs[i]=p['hz'],p['confidence'],p['log2HzOver50']
                result[name+'_hz'],result[name+'_confidence'],result[name+'_log2_hz_over_50']=hz,confidence,logs
            return result
        except BaseException:
            self.close()
            raise


def _validate(feature):
    if set(feature)!=FEATURE_KEYS: raise ValueError('Only acquisition feature fields allowed')
    x,hz,mask=(np.asarray(feature[k]) for k in ('features','candidate_hz','candidate_mask'))
    n=len(x)
    if (x.shape!=(n,57) or x.dtype!=np.float32 or hz.shape!=(n,6) or hz.dtype!=np.float64
            or mask.shape!=(n,6) or mask.dtype!=np.bool_ or not np.isfinite(x).all()
            or not np.array_equal(feature['feature_names'],FEATURE_NAMES_V2)
            or np.asarray(feature['sample_rate_hz']).shape!=() or np.asarray(feature['sample_rate_hz']).dtype.kind not in 'iu'
            or int(feature['sample_rate_hz'])!=48000):
        raise ValueError('Invalid original feature schema/dtypes')
    if not np.isfinite(hz[mask]).all() or np.any((hz[mask]<50)|(hz[mask]>700)) or not np.isnan(hz[~mask]).all():
        raise ValueError('Invalid original candidate frequencies')
    blocks=x[:,:48].reshape(n,6,8)
    if (not np.array_equal(blocks[:,:,0],mask.astype(np.float32)) or np.any(blocks[~mask]!=0)
            or np.any((blocks[:,:,2:7]<0)|(blocks[:,:,2:7]>1))
            or np.any(mask & ~np.any(blocks[:,:,2:7]>0,axis=2))
            or not np.array_equal(blocks[:,:,7],blocks[:,:,2:7].max(axis=2))
            or not np.array_equal(x[:,55],(mask.sum(axis=1)/6).astype(np.float32))):
        raise ValueError('Invalid original candidate evidence')
    for f,m in zip(hz,mask):
        if np.any(np.diff(f[m])<=0) or not np.array_equal(m,np.arange(6)<m.sum()):
            raise ValueError('Sorted packed original candidates required')
    ends=np.asarray(feature['reference_sample'])
    count=np.asarray(feature['pcm_sample_count'])
    if (ends.shape!=(n,) or ends.dtype.kind not in 'iu' or count.shape!=() or count.dtype.kind not in 'iu'
            or not np.array_equal(ends,2208+480*np.arange(n)) or int(count)<=0
            or (n and ends[-1]>int(count))): raise ValueError('Invalid original acquisition grid')
    return x,hz,mask,ends,int(count)


def revised_feature(feature, pcm_path, worker):
    """Return fresh same-layout arrays plus replacement evidence, never mutate input."""
    x,hz,mask,ends,count=_validate(feature)
    evidence=worker.inspect(pcm_path,ends)
    if evidence['pcm_sample_count']!=count or not np.array_equal(evidence['ends'],ends):
        raise ValueError('PCM/cache acquisition identity differs')
    result={k:np.asarray(v).copy() for k,v in feature.items()}
    new_x=np.zeros_like(x);new_x[:,48:]=x[:,48:]
    new_hz=np.full_like(hz,np.nan);new_mask=np.zeros_like(mask)
    changed=np.zeros(len(x),bool);evicted=np.zeros(len(x),bool);added=np.zeros(len(x),bool)
    old_blocks=x[:,:48].reshape(-1,6,8)
    for i,blocks in enumerate(old_blocks):
        old_f,old_c=evidence['old_hz'][i],evidence['old_confidence'][i]
        new_f,new_c=evidence['new_hz'][i],evidence['new_confidence'][i]
        present=np.flatnonzero(blocks[:,5]>0)
        if np.isnan(old_f):
            if len(present) or old_c!=0: raise ValueError('Cached old full40 evidence does not match recomputation')
        elif (len(present)!=1 or hz[i,present[0]]!=old_f or blocks[present[0],5]!=np.float32(old_c)):
            raise ValueError('Cached old full40 evidence does not match recomputation')
        if np.isnan(new_f)!=(np.isnan(old_f)):
            raise ValueError('Peak policies must have identical availability')
        candidates={float(hz[i,j]):blocks[j].copy() for j in np.flatnonzero(mask[i])}
        if np.isfinite(old_f):
            candidates[old_f][5]=0
            if not np.any(candidates[old_f][2:7]>0): del candidates[old_f]
        if np.isfinite(new_f):
            if new_f not in candidates:
                block=np.zeros(8,np.float32);block[0]=1
                block[1]=evidence['new_log2_hz_over_50'][i]
                candidates[new_f]=block
            candidates[new_f][5]=np.float32(new_c)
        if len(candidates)>6: raise ValueError('Fixed candidate bound exceeded')
        for j,(f,block) in enumerate(sorted(candidates.items())):
            block[7]=np.max(block[2:7]);new_x[i,j*8:(j+1)*8]=block
            new_hz[i,j],new_mask[i,j]=f,True
        new_x[i,55]=np.float32(len(candidates)/6)
        changed[i]=np.isfinite(old_f) and (old_f!=new_f or old_c!=new_c)
        evicted[i]=np.isfinite(old_f) and old_f not in candidates
        added[i]=np.isfinite(new_f) and not np.any(hz[i,mask[i]]==new_f)
    result.update(features=new_x,candidate_hz=new_hz,candidate_mask=new_mask)
    _validate(result)
    provenance={f'{label}_full40_{field}':np.asarray(evidence[f'{label}_{field}']).copy()
                for label in ('old','new') for field in ('hz','confidence')}
    provenance.update(full40_changed=changed,old_frequency_evicted=evicted,new_frequency_added=added,
        old_candidate_count=mask.sum(axis=1),new_candidate_count=new_mask.sum(axis=1),
        candidate_policy=np.array(POLICY),pcm_sha256=np.array(evidence['pcm_sha256']),
        worker_rss_bytes=np.array(evidence.get('rss_bytes',0),np.int64))
    return result,provenance
