"""Bounded input-only fixtures for browser hybrid arithmetic parity, never fitting.

Analytic cases and one full recording per development speaker are fixed in source.
This helper never opens a corpus reference or admits sealed-speaker content.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from . import evaluate as ev
from .hybrid_followup import CONTRACT, WEIGHT_HASHES, _feature, hybrid_predictions
from .model import numpy_step
from .pilot import REPO, RESEARCH, read_mic, yin_predictions


def analytic_sources():
    t = np.arange(24000, dtype=np.float64) / 48000
    result = [("silence", np.zeros_like(t)), ("dc-015", np.full_like(t, .015)),
              ("noise-fixed", .02 * np.random.default_rng(20261022).standard_normal(len(t)))]
    for hz in (49.99, 50., 50.01, 699.99, 700., 700.01):
        result.append(("sine-" + str(hz).replace('.', '_'), .075 * np.sin(2 * np.pi * hz * t + .37)))
    phase = 2 * np.pi * 300 * t
    result.append(("harmonic-300", .075 * (np.sin(phase) + .5 * np.sin(2 * phase) + .25 * np.sin(3 * phase))))
    phase = 2 * np.pi * 200 * t + .37
    result.append(("weak-fundamental-200", .075 * (.1 * np.sin(phase) + np.sin(2 * phase) + .5 * np.sin(3 * phase))))
    result.append(("missing-fundamental-200", .075 * (np.sin(2 * phase) + .5 * np.sin(3 * phase))))
    phase = 2 * np.pi * (100 * np.minimum(t, .25) + 300 * np.maximum(t - .25, 0))
    result.append(("step-100-300", .075 * (np.sin(phase) + .5 * np.sin(2 * phase) + .25 * np.sin(3 * phase))))
    phase = 2 * np.pi * (100 * t + 400 * t * t)
    result.append(("glide-100-500", .075 * (np.sin(phase) + .5 * np.sin(2 * phase) + .25 * np.sin(3 * phase))))
    return [(name, value.astype(np.float32)) for name, value in result]


def build(output, data_root):
    jobs = [(name, pcm, {'kind': 'analytic-source'}) for name, pcm in analytic_sources()]
    acquisition = json.loads((data_root / 'manifest.json').read_text())
    rows = []
    for speaker in ('F01', 'F03', 'M01', 'M03'):
        choices = sorted((r for r in acquisition['files'] if r['split'] == 'development' and r['role'] == 'mic'
                          and r['speaker'] == speaker), key=lambda r:r['recording_id'])
        if not choices: raise ValueError('Missing declared development speaker; no replacement')
        rows.append(choices[0])
    for r in rows:
        path = (data_root / r['path']).resolve()
        if not path.is_relative_to(data_root.resolve()) or ev.sha(path) != r['sha256']:
            raise ValueError('Development microphone identity changed')
        pcm = read_mic(path)
        if len(pcm) > 960000:
            raise ValueError('Declared full source exceeds20s; do not truncate or replace it')
        jobs.append(('development-' + r['recording_id'], pcm, {'kind': 'previously-acquired-development',
            'source_recording_id':r['recording_id'], 'microphone_sha256':r['sha256'],
            'clip_start_sample':0,'clip_end_sample_exclusive':len(pcm),'full_source':True}))
    return write_fixtures(output, jobs, 'development-full-source-v2')


def write_fixtures(output, jobs, cohort, additional_sources=None):
    output.mkdir(parents=True, exist_ok=False)
    (output / 'pcm').mkdir(); (output / 'intermediate').mkdir()
    os.environ.setdefault('NUMBA_CACHE_DIR', str(RESEARCH / 'runs/.numba-cache'))
    training = RESEARCH / 'runs/delayed-train-features'
    config, weights, model = ev.load_frozen_model(RESEARCH / 'runs/delayed-mlp-seed20261022', training, ev.read_cache_manifest(training)[0])
    if model['weights_sha256'] != WEIGHT_HASHES['original']:
        raise ValueError('Only original frozen MLP permitted')
    if any(len(pcm)>960000 for _,pcm,_ in jobs): raise ValueError('Full-source parity bound20s exceeded')
    inputs = []
    for name, pcm, _ in jobs:
        path = output / 'pcm' / (name + '.f32le'); pcm.astype('<f4').tofile(path)
        inputs.append({'id': name, 'path': str(path)})
    ev.write_json(output / 'intermediate/input-list.json', inputs)
    subprocess.run(['node','--import','tsx','../../research/voice-prosody/hybrid_features.ts',
        str(output / 'intermediate/input-list.json'),str(output / 'intermediate/features.json')], cwd=REPO / 'apps/capture-web',check=True)
    exported = {r['id']:r['frames'] for r in json.loads((output / 'intermediate/features.json').read_text())}
    records = []
    for name, pcm, source in jobs:
        feature = _feature(exported[name],len(pcm)); timing=ev.validate_feature(feature,{'sample_count':len(pcm)})
        _, yin=yin_predictions(pcm,feature['reference_sample']); learned=ev.model_predictions(feature,config,weights)
        hybrid=hybrid_predictions(feature,learned,yin); state=np.zeros((1,32),np.float32); frames=[]
        for i,end in enumerate(feature['reference_sample']):
            voicing, logits, state=numpy_step(config,weights,feature['features'][i:i+1],feature['candidate_mask'][i:i+1],state)
            raw=yin[i]['raw_f0_hz']; probability=hybrid[i]['voicing_probability']; tolerance=max(.01,1e-4*abs(raw))
            frames.append({'supportEndSampleExclusive':int(end),'referenceSample':int(end-960),
                'featureValues':feature['features'][i].tolist(),
                'candidateHz':[float(h) if m else None for h,m in zip(feature['candidate_hz'][i],feature['candidate_mask'][i])],
                'candidateMask':feature['candidate_mask'][i].tolist(),'voicingLogit':float(voicing[0,0]),
                'candidateLogits':logits[0].tolist(),'voicingProbability':probability,'rawYinHz':raw,
                'state':hybrid[i]['state'],'f0Hz':hybrid[i]['f0_hz'],
                'nearBoundary':abs(probability-.5)<=1e-5 or abs(raw-50)<=tolerance or abs(raw-700)<=tolerance})
        file=output/'pcm'/(name+'.f32le')
        records.append({'id':name,'source':source,'pcmPath':str(file.relative_to(output)),'pcmSha256':ev.sha(file),
            'sampleRateHz':48000,'sampleCount':len(pcm),'historyResetBeforeRecord':True,
            'windowEndSamples':timing.availability_sample.tolist(),'frames':frames})
    manifest={'schema':'phenometrix.hybrid-browser-parity.v1','created_at':datetime.now(timezone.utc).isoformat(),
        'estimatorContract':CONTRACT,'weightsSha256':model['weights_sha256'],'sourceSha256':ev.sha(__file__),
        'fixtureCohort':cohort,'modelConfig':model['model']['config'],'additionalSourceSha256':additional_sources or {},
        'featureExporterSha256':ev.sha(RESEARCH/'hybrid_features.ts'),
        'dependencies':{n:importlib.metadata.version(n) for n in ('numpy','librosa','scipy','torch')},
        'sampleRateHz':48000,'windowSamples':1920,'hopSamples':480,'measurementDelayMs':20,
        'pitchTolerance':'max(0.01Hz,1e-4*abs(referenceHz))','voicingBoundaryTolerance':1e-5,
        'boundaryPolicy':'List each boundary/tie discrepancy; no unconditional parity pass from an exception',
        'reference_content_read':False,'sealed_content_read':cohort=='sealed-input-only-parity-v1','records':records}
    (output/'generator-source.py').write_bytes(Path(__file__).read_bytes())
    ev.write_json(output/'manifest.json',manifest)
    for p in (output/'intermediate').iterdir():p.unlink()
    (output/'intermediate').rmdir()
    return {'manifest_sha256':ev.sha(output/'manifest.json'),'records':len(records),'frames':sum(len(r['frames']) for r in records)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,default=RESEARCH/'data/ptdb-tug');a=p.parse_args()
    print(json.dumps(build(a.output.resolve(),a.data.resolve())))

if __name__=='__main__':main()
