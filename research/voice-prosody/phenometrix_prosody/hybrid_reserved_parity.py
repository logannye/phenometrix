"""Authorized sealed MICROPHONE-only arithmetic parity; no reference access.

Uses the first fixed SHA-selected source per reserved speaker, all four declared
conditions, full recordings capped at20s. It cannot change the frozen estimator
or validation selection, and refuses replacement/truncation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from . import evaluate as ev
from .features import CONDITIONS, transform
from .hybrid_parity import write_fixtures
from .hybrid_validation import SPEAKERS, acquired_selection, inside, read_freeze
from .pilot import RESEARCH, read_mic


def build(freeze_path, data_root, output):
    freeze, configuration, _ = read_freeze(freeze_path)
    _, files = acquired_selection(data_root, freeze, configuration, predicting=True)
    selected = [next(r for r in configuration['selected_inventory_records'] if r['role']=='mic' and r['speaker']==speaker)
                for speaker in SPEAKERS]
    jobs=[]
    for metadata in selected:
        entry=files[metadata['recording_id'],'mic']; path=inside(data_root,entry['path'])
        if ev.sha(path)!=entry['sha256']: raise ValueError('Frozen microphone identity changed')
        clean=read_mic(path)
        if len(clean)>960000: raise ValueError('Fixed source exceeds20s; no replacement or truncation')
        for condition in CONDITIONS:
            pcm, corruption=transform(clean,condition,metadata['recording_id'],'sealed-test')
            jobs.append((metadata['recording_id']+'__'+condition,pcm,{'kind':'authorized-sealed-input-only',
                'source_recording_id':metadata['recording_id'],'microphone_sha256':entry['sha256'],
                'full_source':True,'clip_start_sample':0,'clip_end_sample_exclusive':len(clean),
                'corruption':corruption,'freeze_sha256':ev.sha(freeze_path)}))
    result=write_fixtures(output,jobs,'sealed-input-only-parity-v1',{'hybrid_reserved_parity.py':ev.sha(__file__)})
    (output/'reserved-generator-source.py').write_bytes(Path(__file__).read_bytes())
    read_freeze(freeze_path); acquired_selection(data_root,freeze,configuration,predicting=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,default=RESEARCH/'data/ptdb-tug');a=p.parse_args()
    print(json.dumps(build(a.freeze.resolve(),a.data.resolve(),a.output.resolve())))

if __name__=='__main__':main()
