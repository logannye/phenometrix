"""Toy-window tests for the independent TypeScript acoustic kernel.

No study data, TRAIN bank, fresh challenge generation, fitting, or labels.
"""
import json
import hashlib
from pathlib import Path
import subprocess

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = r"""
import periodicity from '../../research/voice-prosody/periodicity_evidence.ts';
const {inspectPeriodicityEvidence}=periodicity;
let raw='';for await(const part of process.stdin)raw+=part;
const request=JSON.parse(raw);
try {
  let pcm=new Float32Array(request.pcm);
  const hz=Float64Array.from(request.hz.map(x=>x===null?NaN:x));
  const mask=request.mask;
  if(request.mutate==='nan-pcm')pcm[100]=NaN;
  if(request.mutate==='infinite-pcm')pcm[100]=Infinity;
  if(request.mutate==='double-pcm')pcm=new Float64Array(pcm);
  if(request.mutate==='infinite-hz')hz[0]=Infinity;
  if(request.mutate==='outside-window') {
    const whole=new Float32Array(6000);whole.fill(17);
    whole.set(pcm,1217);whole.fill(-31,1217+1920);
    pcm=whole.subarray(1217,1217+1920);
  }
  const before=Array.from(pcm);
  const out=inspectPeriodicityEvidence(pcm,request.sample_rate??48000,hz,mask);
  console.log(JSON.stringify({
    ...Object.fromEntries(Object.entries(out).map(([key,value])=>[key,Array.from(value)])),
    unchanged:before.every((x,i)=>Object.is(x,pcm[i]))}));
} catch(error) { console.log(JSON.stringify({error:String(error)})); }
"""


def inspect(pcm, hz=(200.,), mask=None, *, pad=True, **extra):
    hz=list(hz);mask=list(mask if mask is not None else [True]*len(hz));count=len(hz)
    if pad and len(hz)<6 and len(mask)==len(hz):
        hz += [None]*(6-count);mask += [False]*(6-count)
    request={'pcm':np.asarray(pcm).tolist(), 'hz':hz, 'mask':mask, **extra}
    run=subprocess.run(['node','--import','tsx','--input-type=module','-e',SCRIPT],
        input=json.dumps(request,allow_nan=False),text=True,capture_output=True,
        cwd=REPO/'apps/capture-web',check=True,timeout=15)
    result=json.loads(run.stdout)
    if pad and count<6 and 'error' not in result:
        for field,width in [('channels',3),('correlations',4),('valid',1),('periodSamples',1),('anchorLength',1)]:
            result[field]=result[field][:count*width]
    return result


def toy_tone(amplitude=.1):
    # Fixed familiar unit-test carrier, not a scientific evaluation waveform.
    return (amplitude*np.sin(2*np.pi*200*np.arange(1920)/48000)).astype(np.float32)


def test_known_period_and_octave_alias_contrast():
    r=inspect(toy_tone(),[100.,200.,400.])
    c=np.asarray(r['correlations']).reshape(3,4)
    channels=np.asarray(r['channels']).reshape(3,3)
    assert r['valid']==[1,1,1] and r['unchanged']
    np.testing.assert_allclose(c[0,:2],[1,1],atol=2e-15,rtol=0)
    np.testing.assert_allclose(c[1,:2],[1,-1],atol=2e-15,rtol=0)
    assert c[2,0]==pytest.approx(-1,abs=2e-15)
    assert abs(channels[0,1])<1e-7
    assert channels[1,1]>.49 and channels[2,1]<-.4
    assert r['periodSamples']==[480.,240.,120.]
    assert r['anchorLength']==[1439,1679,1799]


def test_fractional_interpolation_uses_same_anchor_for_all_four_shifts():
    x=np.random.default_rng(1729).normal(0,.2,1920).astype(np.float32)
    r=inspect(x,[183.25,611.5])
    centered=x.astype(np.float64)-x.astype(np.float64).mean()
    expected=[]
    for f in (183.25,611.5):
        period=48000/f;length=1920-int(np.floor(period))-1
        anchor=centered[:length]
        for divisor in (1,2,3,4):
            shifted=np.interp(np.arange(length)+period/divisor,np.arange(1920),centered)
            expected.append(np.dot(anchor,shifted)/np.sqrt(np.dot(anchor,anchor)*np.dot(shifted,shifted)))
    np.testing.assert_allclose(r['correlations'],expected,rtol=0,atol=2e-14)
    c=np.asarray(expected).reshape(2,4)
    channels=np.column_stack((c[:,0],(c[:,0]-c[:,1:].max(axis=1))/2,np.ones(2))).astype(np.float32)
    np.testing.assert_array_equal(np.asarray(r['channels']).reshape(2,3),channels)


@pytest.mark.parametrize('pcm',[np.zeros(1920,np.float32),np.full(1920,.25,np.float32),toy_tone(1e-9)])
def test_silence_dc_and_low_energy_zero_all_evidence(pcm):
    r=inspect(pcm,[50.,200.,700.])
    assert r['valid']==[0,0,0]
    assert r['channels']==[0]*9 and r['correlations']==[0]*12
    assert r['periodSamples']==[960.,240.,48000/700]
    assert r['anchorLength']==[959,1679,1851]


def test_sufficient_small_energy_remains_valid():
    assert inspect(toy_tone(1e-7))['valid']==[1]


def test_any_invalid_shifted_energy_zeros_already_computed_correlations():
    x=np.zeros(1920,np.float32)
    x[[0,1800]]=1;x[[1,1801]]=-1
    # Anchor and T-shift have energy; T/2 and shorter shifted spans have none.
    r=inspect(x)
    assert r['valid']==[0] and r['correlations']==[0]*4 and r['channels']==[0]*3


def test_masked_candidate_is_not_interpreted_and_empty_valid_set_is_bounded():
    r=inspect(toy_tone(),[None,200.,None],[0,1,False])
    assert r['valid']==[0,1,0]
    for field,width in [('channels',3),('correlations',4)]:
        assert r[field][:width]==[0]*width and r[field][-width:]==[0]*width
    assert r['periodSamples']==[0,240.,0] and r['anchorLength']==[0,1679,0]
    r=inspect(toy_tone(),[None]*6,[False]*6)
    assert all(all(x==0 for x in r[name]) for name in ['channels','correlations','valid','periodSamples','anchorLength'])
    assert 'six or seven' in inspect(toy_tone(),[],[],pad=False)['error']


def test_candidate_permutation_and_duplicate_identity():
    f=[50.,100.,200.,400.,700.,200.,None];mask=[True]*6+[False]
    r=inspect(toy_tone(),f,mask);order=[6,2,0,5,3,1,4]
    p=inspect(toy_tone(),[f[i] for i in order],[mask[i] for i in order])
    for field,width in [('channels',3),('correlations',4),('valid',1),('periodSamples',1),('anchorLength',1)]:
        a=np.asarray(r[field]).reshape(7,width)
        np.testing.assert_array_equal(np.asarray(p[field]).reshape(7,width),a[order])
        np.testing.assert_array_equal(a[2],a[5])


def test_exact_window_support_is_independent_of_past_and_future_backing_buffer():
    x=toy_tone()
    assert inspect(x,[183.25,200.])==inspect(x,[183.25,200.],mutate='outside-window')


def test_last_required_sample_can_affect_fractional_period_evidence():
    x=toy_tone();a=inspect(x,[200.5]);x[-1]+=.5;b=inspect(x,[200.5])
    assert a['correlations']!=b['correlations']


def test_float64_centering_preserves_binary_exact_gain_and_dc_invariance():
    x=np.tile(np.array([.125,.25,-.125,-.25],np.float32),480)
    a=inspect(x,[183.25,200.]);b=inspect(x*2+.5,[183.25,200.])
    np.testing.assert_array_equal(a['channels'],b['channels'])
    np.testing.assert_allclose(a['correlations'],b['correlations'],atol=1e-15,rtol=0)


def test_finite_large_float32_values_do_not_overflow_correlation_arithmetic():
    r=inspect(toy_tone()*np.float32(1e38),[50.,200.,700.])
    assert 'error' not in r and r['valid']==[1,1,1]
    assert np.isfinite(r['channels']).all() and np.isfinite(r['correlations']).all()
    assert np.max(np.abs(r['correlations']))<=1


@pytest.mark.parametrize('length',[0,1919,1921])
def test_wrong_window_length_refuses(length):
    assert 'exactly 1920' in inspect(np.zeros(length,np.float32))['error']


@pytest.mark.parametrize('mutate,match',[
    ('nan-pcm','Finite PCM'),('infinite-pcm','Finite PCM'),
    ('double-pcm','Float32'),('infinite-hz','50..700'),
])
def test_nonfinite_and_wrong_dtype_refuse(mutate,match):
    assert match in inspect(toy_tone(),mutate=mutate)['error']


@pytest.mark.parametrize('hz,mask,match',[
    ([49.99],[True],'50..700'),([700.01],[True],'50..700'),
    ([None],[True],'50..700'),([200.],[2],'mask'),
    ([200.],[],'matching'),([200.]*8,[True]*8,'seven'),
])
def test_candidate_contract_refuses(hz,mask,match):
    assert match in inspect(toy_tone(),hz,mask)['error']


def test_wrong_sample_rate_refuses():
    assert '48000' in inspect(toy_tone(),sample_rate=44100)['error']


WORKER = REPO/'research/voice-prosody/periodicity_evidence_worker.ts'


def request(path, ends=(1920,), identity=1):
    return {'id':identity,'pcm_path':str(path),'sample_rate_hz':48000,'ends':list(ends),
            'candidate_hz':[[50.,100.,200.,400.,700.,None,None] for _ in ends],
            'candidate_mask':[[True]*5+[False]*2 for _ in ends]}


def run_worker(requests):
    data=''.join(json.dumps(r,allow_nan=False)+'\n' for r in requests)
    return subprocess.run(['node','--import','tsx',str(WORKER)],input=data,
        capture_output=True,text=True,cwd=REPO/'apps/capture-web',timeout=20)


def pcm_file(tmp_path, name='tone', values=None):
    p=tmp_path/(name+'.f32le')
    np.asarray(toy_tone() if values is None else values,dtype='<f4').tofile(p)
    return p


def test_serial_worker_hash_support_and_no_cross_request_history(tmp_path):
    tone=pcm_file(tmp_path)
    silence=pcm_file(tmp_path,'silence',np.zeros(1920,np.float32))
    run=run_worker([request(tone),request(silence,identity=2)])
    assert run.returncode==0,run.stderr
    a,b=[json.loads(line) for line in run.stdout.splitlines()]
    assert (a['id'],b['id'])==(1,2) and a['ok'] and b['ok']
    assert a['schema']=='phenometrix.candidate-periodicity-evidence.v1'
    assert a['pcm_sha256']==hashlib.sha256(tone.read_bytes()).hexdigest()
    assert a['pcm_sample_count']==1920 and a['ends']==[1920]
    assert len(a['channels'][0])==21 and len(a['correlations'][0])==28
    assert a['valid']==[[1]*5+[0]*2] and b['valid']==[[0]*7]
    assert b['channels']==[[0]*21] and b['correlations']==[[0]*28]
    assert a['common_length'][0]==[959,1439,1679,1799,1851,0,0]
    assert 0<a['rss_bytes']<=2*1024**3
    assert 0<=a['kernel_elapsed_ms']<=a['elapsed_ms']<120000


def test_worker_future_samples_do_not_change_earlier_window_evidence(tmp_path):
    x=np.concatenate((toy_tone(),np.zeros(480,np.float32)))
    a=pcm_file(tmp_path,'first',x)
    x[1920:]=np.linspace(-.5,.5,480,dtype=np.float32)
    b=pcm_file(tmp_path,'second',x)
    run=run_worker([request(a,(1920,2400)),request(b,(1920,2400),identity=2)])
    assert run.returncode==0,run.stderr
    first,second=[json.loads(line) for line in run.stdout.splitlines()]
    assert first['pcm_sha256']!=second['pcm_sha256']
    assert first['channels'][0]==second['channels'][0]
    assert first['correlations'][0]==second['correlations'][0]
    assert first['correlations'][1]!=second['correlations'][1]


@pytest.mark.parametrize('ends',[(1919,),(1921,),(1920,1920),(1920.5,),tuple([1920]*12001)])
def test_worker_rejects_bad_or_excess_endpoints(tmp_path,ends):
    run=run_worker([request(pcm_file(tmp_path),ends)])
    result=json.loads(run.stdout)
    assert run.returncode!=0 and not result['ok']
    assert 'endpoint' in result['error']


@pytest.mark.parametrize('change,reason',[
    (lambda r:r.update(reference_hz=[200.]),'Unknown request field'),
    (lambda r:r.update(sample_rate_hz=44100),'48000'),
    (lambda r:r.update(candidate_hz=[]),'inventory'),
    (lambda r:r['candidate_hz'][0].pop(),'Seven grouped'),
    (lambda r:r['candidate_mask'][0].__setitem__(0,2),'mask'),
    (lambda r:r['candidate_hz'][0].__setitem__(0,49.99),'in-range'),
    (lambda r:r.update(timeout_ms=120001),'deadline'),
    (lambda r:r.update(timeout_ms=.000001),'deadline exceeded'),
])
def test_worker_contract_and_deadline_fail_closed(tmp_path,change,reason):
    r=request(pcm_file(tmp_path));change(r)
    run=run_worker([r,request(pcm_file(tmp_path,'later'),identity=2)])
    lines=run.stdout.splitlines()
    assert run.returncode!=0 and len(lines)==1
    failure=json.loads(lines[0]);assert not failure['ok'] and reason in failure['error']


def test_worker_checks_file_size_before_read(tmp_path):
    p=tmp_path/'oversize.f32le'
    with p.open('wb') as stream:stream.truncate((48000*120+1)*4)
    run=run_worker([request(p)])
    assert run.returncode!=0 and 'PCM size' in json.loads(run.stdout)['error']


def test_worker_nonfinite_pcm_refuses(tmp_path):
    x=toy_tone();x[100]=np.nan
    run=run_worker([request(pcm_file(tmp_path,values=x))])
    assert run.returncode!=0 and 'Finite PCM' in json.loads(run.stdout)['error']


def test_worker_request_buffer_is_bounded_before_json_parse():
    run=subprocess.run(['node','--import','tsx',str(WORKER)],input='x'*(16*1024*1024+1),
        capture_output=True,text=True,cwd=REPO/'apps/capture-web',timeout=20)
    assert run.returncode!=0 and 'exceeds 16 MiB' in run.stderr


def test_browser_kernel_has_no_node_or_transport_dependency():
    source=(REPO/'research/voice-prosody/periodicity_evidence.ts').read_text()
    assert 'node:' not in source and 'process.' not in source and 'require(' not in source


@pytest.mark.parametrize('kind',['tone','noise','silence','low-energy','one-empty-shift'])
@pytest.mark.parametrize('capacity',[6,7])
def test_independent_numpy_kernel_parity_on_fixed_toy_windows(kind,capacity):
    from phenometrix_prosody.periodicity_evidence import periodicity_evidence
    if kind=='tone':x=toy_tone()
    elif kind=='noise':x=np.random.default_rng(392).normal(0,.15,1920).astype(np.float32)
    elif kind=='silence':x=np.zeros(1920,np.float32)
    elif kind=='low-energy':x=toy_tone(1e-9)
    else:
        x=np.zeros(1920,np.float32);x[[0,1800]]=1;x[[1,1801]]=-1
    hz=np.array([50.,100.,183.25,200.,611.5,700.,np.nan][:capacity])
    mask=np.isfinite(hz)
    py=periodicity_evidence(x,hz,mask)
    ts=inspect(x,[float(f) if np.isfinite(f) else None for f in hz],mask.tolist())
    np.testing.assert_allclose(np.array(ts['correlations']).reshape(capacity,4),py.correlations,atol=2e-13,rtol=0)
    np.testing.assert_allclose(np.array(ts['channels']).reshape(capacity,3),py.features,atol=1e-7,rtol=0)
    np.testing.assert_array_equal(ts['valid'],py.valid)
    np.testing.assert_array_equal(ts['anchorLength'],py.common_length)
    np.testing.assert_array_equal(ts['periodSamples'],py.period_samples)
