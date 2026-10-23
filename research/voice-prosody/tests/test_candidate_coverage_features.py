"""Known-answer algebra and input-only worker checks; no scientific cases."""
import copy
import json
import subprocess
from pathlib import Path
import numpy as np
import pytest
from phenometrix_prosody import candidate_coverage_features as cc


def feature(candidates=()):
    x=np.zeros((1,57),np.float32);hz=np.full((1,6),np.nan);mask=np.zeros((1,6),bool)
    for j,(f,scores) in enumerate(sorted(candidates)):
        x[0,j*8:(j+1)*8]=[1,np.log2(f/50),*scores,max(scores)]
        hz[0,j],mask[0,j]=f,True
    x[0,48:]=[-3,.2,.003,.01,.4,.7,.9,len(candidates)/6,1]
    return dict(features=x,candidate_hz=hz,candidate_mask=mask,reference_sample=np.array([2208]),
        feature_names=np.array(cc.FEATURE_NAMES_V2),sample_rate_hz=np.array(48000),pcm_sample_count=np.array(2400))


class FakeWorker:
    def __init__(self,old,new):self.old,self.new=old,new
    def inspect(self,path,ends):
        d=dict(pcm_sha256='a'*64,pcm_sample_count=2400,ends=ends.copy(),rss_bytes=1234)
        for k,p in [('old',self.old),('new',self.new)]:
            f,c=(np.nan,0) if p is None else p
            d[k+'_hz']=np.array([f]);d[k+'_confidence']=np.array([c])
            d[k+'_log2_hz_over_50']=np.array([np.log2(f/50) if np.isfinite(f) else 0])
        return d


def run(f,old,new):return cc.revised_feature(f,Path('unused.f32le'),FakeWorker(old,new))


def test_absence_is_preserved_without_inventing_peak_or_zero_frequency():
    f=feature();out,p=run(f,None,None)
    for k in f:np.testing.assert_array_equal(out[k],f[k])
    assert np.isnan(p['old_full40_hz']).all() and not p['full40_changed'].any()
    assert not out['candidate_mask'].any()


def test_replacement_retains_other_sources_and_copies_inputs():
    f=feature([(100,[.8,.7,0,.9,0]),(200,[0,0,0,0,.6])]);before=copy.deepcopy(f)
    out,p=run(f,(100,.9),(300,.75));b=out['features'][0,:48].reshape(6,8)
    np.testing.assert_array_equal(out['candidate_hz'][0,:3],[100,200,300])
    np.testing.assert_array_equal(b[0,2:7],np.array([.8,.7,0,0,0],np.float32))
    assert b[2,5]==np.float32(.75) and b[2,7]==np.float32(.75)
    assert p['full40_changed'][0] and p['new_frequency_added'][0] and not p['old_frequency_evicted'][0]
    assert out['features'][0,55]==np.float32(3/6)
    np.testing.assert_array_equal(out['features'][:,[48,49,50,51,52,53,54,56]],f['features'][:,[48,49,50,51,52,53,54,56]])
    for k in f:np.testing.assert_array_equal(f[k],before[k]);assert not np.shares_memory(out[k],f[k])


def test_source_only_old_candidate_is_evicted_and_collision_merged():
    f=feature([(100,[0,0,0,.9,0]),(300,[.6,0,.8,0,0])])
    out,p=run(f,(100,.9),(300,.75));b=out['features'][0,:48].reshape(6,8)
    assert out['candidate_hz'][0,0]==300 and out['candidate_mask'].sum()==1
    np.testing.assert_array_equal(b[0,2:7],np.array([.6,0,.8,.75,0],np.float32))
    assert b[0,7]==np.float32(.8) and p['old_frequency_evicted'][0] and not p['new_frequency_added'][0]
    assert out['features'][0,55]==np.float32(1/6)


def test_unchanged_peak_is_byte_identical():
    f=feature([(100,[.8,0,0,.9,0]),(300,[0,0,0,0,.7])])
    out,p=run(f,(100,.9),(100,.9))
    for k in f:np.testing.assert_array_equal(out[k],f[k])
    assert not p['full40_changed'].any() and not p['old_frequency_evicted'].any()


@pytest.mark.parametrize('old',[(101,.9),(100,.8),None])
def test_wrong_cached_old_full40_identity_refuses(old):
    f=feature([(100,[0,0,0,.9,0])])
    with pytest.raises(ValueError,match='old full40'):run(f,old,(300,.7))


def test_two_original_full40_peaks_refuse():
    f=feature([(100,[0,0,0,.9,0]),(200,[0,0,0,.8,0])])
    with pytest.raises(ValueError,match='old full40'):run(f,(100,.9),(300,.7))


@pytest.mark.parametrize('old,new',[(None,(300,.7)),((100,.9),None)])
def test_policy_cannot_change_peak_availability(old,new):
    f=feature([] if old is None else [(100,[0,0,0,.9,0])])
    with pytest.raises(ValueError,match='availability'):run(f,old,new)


def test_fixed_six_slot_bound_refuses_if_cache_impossible_for_adapter():
    f=feature([(100,[.8,0,0,.9,0])]+[(150+50*i,[.7,0,0,0,0]) for i in range(5)])
    with pytest.raises(ValueError,match='bound'):run(f,(100,.9),(600,.7))


@pytest.mark.parametrize('change',[
    lambda f:f.update(reference_f0=np.array([100.])),
    lambda f:f['features'].__setitem__((0,0),0),
    lambda f:f['features'].__setitem__((0,55),.99),
    lambda f:f['features'].__setitem__((0,7),.01),
    lambda f:f['candidate_hz'].__setitem__((0,0),900),
    lambda f:f['reference_sample'].__setitem__(0,2300),
])
def test_schema_or_evidence_corruption_refuses_before_worker(change):
    f=feature([(100,[0,0,0,.9,0])]);change(f)
    class Never:
        def inspect(self,*_):raise AssertionError('must refuse before worker')
    with pytest.raises(ValueError):cc.revised_feature(f,Path('unused'),Never())


def test_worker_silence_tone_persistence_and_original_v2_arithmetic(tmp_path):
    # Existing unit-test carrier only; no challenge/regression synthesis.
    silence=tmp_path/'silence.f32le';np.zeros(2400,'<f4').tofile(silence)
    tone=tmp_path/'tone.f32le';(.12*np.sin(2*np.pi*200*np.arange(2400)/48000)).astype('<f4').tofile(tone)
    ends=np.array([1920,2208,2400])
    with cc.CoveragePeakWorker() as worker:
        a=worker.inspect(silence,ends);b=worker.inspect(tone,ends)
        assert np.isnan(a['old_hz']).all() and np.isnan(a['new_hz']).all()
        assert np.all(np.isfinite(b['new_hz'])) and worker.peak_rss_bytes>0
        script="""import fs from 'node:fs';import {inspectRefinedVoiceLagEvidenceV2} from './src/voice-dsp.ts';
const b=fs.readFileSync(process.argv[1]);const x=new Float32Array(b.buffer,b.byteOffset,b.length/4);
console.log(JSON.stringify([1920,2208,2400].map(e=>inspectRefinedVoiceLagEvidenceV2(x.subarray(e-1920,e),48000).full40[0]??null)));"""
        r=subprocess.run(['node','--import','tsx','--input-type=module','-e',script,str(tone)],
            cwd=cc.REPO/'apps/capture-web',check=True,capture_output=True,text=True)
        old=json.loads(r.stdout)
        np.testing.assert_array_equal(b['old_hz'],[p['hz'] for p in old])
        np.testing.assert_array_equal(b['old_confidence'],[p['confidence'] for p in old])
    assert worker.process is None


@pytest.mark.parametrize('ends',[np.array([1919]),np.array([2500]),np.array([2208,1920]),np.array([2208.])])
def test_worker_rejects_bad_support_before_send(tmp_path,ends):
    path=tmp_path/'zero.f32le';np.zeros(2400,'<f4').tofile(path)
    with cc.CoveragePeakWorker() as worker:
        with pytest.raises(ValueError):worker.inspect(path,ends)


def test_worker_nonfinite_pcm_fails_closed(tmp_path):
    path=tmp_path/'nonfinite.f32le';x=np.zeros(2400,'<f4');x[0]=np.nan;x.tofile(path)
    with cc.CoveragePeakWorker() as worker:
        with pytest.raises(ValueError,match='Finite PCM'):worker.inspect(path,np.array([2208]))
        assert worker.process is None


def test_worker_deadline_closes_process(tmp_path,monkeypatch):
    path=tmp_path/'zero.f32le';np.zeros(2400,'<f4').tofile(path)
    with cc.CoveragePeakWorker(timeout_seconds=.001) as worker:
        monkeypatch.setattr(cc.select,'select',lambda *_:([],[],[]))
        with pytest.raises(TimeoutError):worker.inspect(path,np.array([2208]))
        assert worker.process is None


def test_new_policy_retains_earliest_300hz_peak_when_old_ranks_period_multiple(tmp_path):
    # Historical unit carrier, not a new scientific challenge configuration.
    path=tmp_path/'tone300.f32le'
    (.12*np.sin(2*np.pi*300*np.arange(2400)/48000)).astype('<f4').tofile(path)
    with cc.CoveragePeakWorker() as worker:r=worker.inspect(path,np.array([2208]))
    assert abs(r['new_hz'][0]-300)<2
    assert r['new_hz'][0]>2*r['old_hz'][0]
    assert r['new_confidence'][0]<r['old_confidence'][0]


def test_large_pcm_refused_before_read(tmp_path):
    path=tmp_path/'large.f32le'
    with path.open('wb') as f:f.truncate((cc.MAX_SAMPLES+1)*4)
    with cc.CoveragePeakWorker() as worker:
        with pytest.raises(ValueError,match='size'):worker.inspect(path,np.array([2208]))


def test_response_size_bound_fails_closed(tmp_path,monkeypatch):
    path=tmp_path/'zero.f32le';np.zeros(2400,'<f4').tofile(path)
    monkeypatch.setattr(cc,'MAX_RESPONSE_BYTES',1)
    with cc.CoveragePeakWorker() as worker:
        with pytest.raises(ValueError,match='response exceeds'):worker.inspect(path,np.array([2208]))
        assert worker.process is None


def test_observed_rss_limit_fails_closed(tmp_path,monkeypatch):
    path=tmp_path/'zero.f32le';np.zeros(2400,'<f4').tofile(path)
    monkeypatch.setattr(cc,'MAX_RSS_BYTES',1)
    with cc.CoveragePeakWorker() as worker:
        with pytest.raises(ValueError,match='RSS exceeds'):worker.inspect(path,np.array([2208]))
        assert worker.process is None
