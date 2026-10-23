"""Scorer-only fixed-delay diagnostic; never changes original end-target results.

The 20 ms arm explicitly estimates reference time t from support[t-20,t+20).
It requires20ms future support relative to the target and is not zero-lookahead.
Every recording uses exactly the same declared shift; no time-offset search.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from .metrics import Prediction, score, oracle_coverage
from .pilot import digest, references_from_array


def run(data_root, original, revised, centered, end_anchored, output):
    output.mkdir(parents=True,exist_ok=False)
    selection=json.loads((original/"manifest.json").read_text())
    if selection["split"]!="train":raise ValueError("Training-only diagnosis.")
    data=json.loads((data_root/"manifest.json").read_text())
    refs={r["recording_id"]:r for r in data["files"] if r["split"]=="train" and r["role"]=="ref"}
    groups={};per_record=[]
    for record in selection["recordings"]:
        identity=record["recording_id"];entry=refs[identity]
        path=data_root/entry["path"]
        if digest(path)!=entry["sha256"]:raise ValueError("Reference bytes changed.")
        references=[replace(r,eligible=r.eligible and r.sample>=3072)
                    for r in references_from_array(np.loadtxt(path),record["sample_count"])]
        targets={r.sample for r in references}
        base=json.loads((original/"predictions"/(identity+".json")).read_text())
        predictions={}
        for name in ["dsp","yin"]:
            for delay in [0,20]:
                shift=delay*48
                predictions[f"{name}_delay{delay}ms"]=[Prediction(r["sample"]-shift,r["state"],r["f0_hz"])
                    for r in base[name] if r["sample"]-shift in targets]
        for name,folder in [("crepe_centered32ms",centered),("crepe_modified_end64ms",end_anchored)]:
            rows=json.loads((folder/"predictions"/(identity+".json")).read_text())
            predictions[name]=[Prediction(r["sample"],r["state"],r["f0_hz"]) for r in rows if r["sample"] in targets]
        feature=np.load(revised/"features"/(identity+".npz"),allow_pickle=False)
        candidates={f"v2_oracle_delay{delay}ms":{int(t)-48*delay:list(hz[mask])
            for t,hz,mask in zip(feature["reference_sample"],feature["candidate_hz"],feature["candidate_mask"])}
            for delay in [0,20]}
        rows={name:score(references,pred) for name,pred in predictions.items()}
        oracles={name:oracle_coverage(references,values) for name,values in candidates.items()}
        per_record.append({"recording_id":identity,"scores":rows,"oracles":oracles})
        groups.setdefault(record["speaker"],[]).append((references,predictions,candidates))
    per_speaker={}
    for speaker,records in groups.items():
        references=[];predictions={};candidates={}
        for i,(refs,preds,options) in enumerate(records):
            delta=i*10**8
            references.extend(replace(r,sample=r.sample+delta) for r in refs)
            for name,values in preds.items():predictions.setdefault(name,[]).extend(replace(p,sample=p.sample+delta) for p in values)
            for name,values in options.items():candidates.setdefault(name,{}).update({sample+delta:hz for sample,hz in values.items()})
        per_speaker[speaker]={"scores":{name:score(references,pred) for name,pred in predictions.items()},
            "oracles":{name:oracle_coverage(references,values) for name,values in candidates.items()}}
    first=next(iter(per_speaker.values()))
    metrics=["joint_error","reference_voiced_success_rate","voicing_false_positive_rate","voicing_false_negative_or_unavailable_rate"]
    macro={name:{metric:float(np.mean([row["scores"][name][metric] for row in per_speaker.values()]))
        for metric in metrics} for name in first["scores"]}
    oracle_macro={name:float(np.mean([row["oracles"][name]["coverage"] for row in per_speaker.values()])) for name in first["oracles"]}
    result={"schema":"phenometrix-fixed-delay-diagnostic-v1","status":"complete","split":"train",
        "recording_count":len(per_record),"speaker_count":len(groups),"fixed_new_delay_ms":20,
        "shared_boundary":"target>=64ms and target+32ms<=recording end; valid original provider rows",
        "primary_original_gate":"FAILED; this explicitly changed target-availability contract cannot retrospectively pass it",
        "input_run_manifests":{str(folder):digest(folder/"manifest.json") for folder in [original,revised,centered,end_anchored]},
        "source_sha256":digest(Path(__file__)),"speaker_macro":macro,"oracle_speaker_macro":oracle_macro,
        "per_speaker":per_speaker,"per_recording":per_record,
        "limitations":["Exploratory training-only diagnostic; no held-out/model-training result",
            "No per-recording shift chosen; one20ms point motivated by analytic support lag",
            "Unknown provider high-pass delay compensation remains unresolved",
            "Oracle uses labels only to bound achievable candidate selection; not a runnable estimator"]}
    (output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"speaker_macro":macro,"oracle_speaker_macro":oracle_macro},indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["data","original","revised","centered","end-anchored","output"]:p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();run(a.data,a.original,a.revised,a.centered,a.end_anchored,a.output)
