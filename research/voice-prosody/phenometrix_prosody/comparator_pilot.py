"""Run the pinned delayed comparator on a frozen training-only pilot selection."""
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .comparator import TinyComparator, TinyConfig, TINY_SHA256, TORCHCREPE_COMMIT
from .pilot import read_mic, digest, references_from_array, source_identity
from .metrics import Prediction, score


def run(data_root,selection,output,anchor="centered"):
    output.mkdir(parents=True,exist_ok=False)
    (output/"predictions").mkdir()
    original=json.loads(selection.read_text())
    if original["split"]!="train":raise ValueError("Only the frozen training pilot is allowed.")
    inventory=json.loads((data_root/"manifest.json").read_text())
    inputs={r["recording_id"]:r for r in inventory["files"] if r["role"]=="mic" and r["split"]=="train"}
    references={r["recording_id"]:r for r in inventory["files"] if r["role"]=="ref" and r["split"]=="train"}
    if not torch.backends.mps.is_available():raise RuntimeError("Local MPS required for comparator pilot.")
    config=TinyConfig(device="mps",batch_size=128,anchor=anchor)
    estimator=TinyComparator(config)
    manifest={"schema":"phenometrix-delayed-comparator-pilot-v1","status":"running","split":"train",
        "evidence":"exploratory-training-only-quality-comparison-no-runtime-claim",
        "selection_manifest_sha256":digest(selection),"model_sha256":TINY_SHA256,
        "source_commit":TORCHCREPE_COMMIT,"config":asdict(config),"source":source_identity(),
        "source_adapter_sha256":digest(Path(__file__).with_name("comparator.py")),
        "source_runner_sha256":digest(Path(__file__)),"records":[],"started_at":datetime.now(timezone.utc).isoformat()}
    def save(): (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    save()
    try:
        # Finish all acquisition-only predictions before reading the first target.
        for record in original["recordings"]:
            identity=record["recording_id"];entry=inputs[identity];path=data_root/entry["path"]
            if digest(path)!=record["microphone_sha256"]:raise ValueError("Frozen microphone identity changed.")
            pcm=read_mic(path);predictions=estimator.predict(pcm)
            prediction_path=output/"predictions"/(identity+".json")
            prediction_path.write_text(json.dumps(predictions,allow_nan=False)+"\n")
            manifest["records"].append({"recording_id":identity,"speaker":entry["speaker"],
                "pair":entry["pair"],"pcm_sample_count":len(pcm),"microphone_sha256":entry["sha256"],
                "prediction_sha256":digest(prediction_path)})
            save();print(json.dumps({"predicted":identity,"frames":len(predictions)}),flush=True)
        manifest["predictions_completed_before_reference_read"]=True;save()
        per_record=[];by_speaker={}
        for record in manifest["records"]:
            identity=record["recording_id"];entry=references[identity];path=data_root/entry["path"]
            if digest(path)!=entry["sha256"]:raise ValueError("Reference identity changed.")
            refs=references_from_array(np.loadtxt(path),record["pcm_sample_count"])
            target_set={r.sample for r in refs}
            raw=json.loads((output/"predictions"/(identity+".json")).read_text())
            pred=[Prediction(r["sample"],r["state"],r["f0_hz"]) for r in raw if r["sample"] in target_set]
            result=score(refs,pred);per_record.append({"recording_id":identity,"scores":result,
                "reference_sha256":entry["sha256"]})
            by_speaker.setdefault(record["speaker"],[]).append((refs,pred))
        per_speaker={}
        # Offset sample coordinates only in scorer aggregation to keep recordings distinct.
        for speaker,rows in by_speaker.items():
            all_refs=[];all_pred=[]
            for index,(refs,pred) in enumerate(rows):
                shift=index*10**8
                all_refs.extend(type(r)(r.sample+shift,r.f0_hz,r.voiced,r.valid,r.eligible) for r in refs)
                all_pred.extend(Prediction(r.sample+shift,r.state,r.f0_hz) for r in pred)
            per_speaker[speaker]=score(all_refs,all_pred)
        keys=["joint_error","reference_voiced_success_rate","voicing_false_positive_rate",
            "voicing_false_negative_or_unavailable_rate","octave_error_rate_reference_voiced"]
        scores={"recording_count":len(per_record),"speaker_count":len(per_speaker),
            "declared_future_support_ms":32 if anchor=="centered" else 0,
            "support_contract": "centered64ms" if anchor=="centered" else "modified-end-anchored64ms","held_out_result":False,
            "speaker_macro":{k:float(np.mean([s[k] for s in per_speaker.values()])) for k in keys},
            "per_speaker":per_speaker,"per_recording":per_record,
            "limitations":["Nominal provider timing; high-pass delay unresolved",
            "Frame-local resampling differs from upstream whole-record resampy",
            "Exact pretrained training-file manifest unavailable",
            "Fixed diagnostic voicing gate; not development-calibrated"]}
        (output/"scores.json").write_text(json.dumps(scores,indent=2)+"\n")
        manifest.update(status="complete",finished_at=datetime.now(timezone.utc).isoformat(),
            scores_sha256=digest(output/"scores.json"));save()
        print(json.dumps(scores["speaker_macro"],indent=2))
    except BaseException as error:
        manifest.update(status="failed",failure=repr(error));save();raise

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data",type=Path,required=True)
    p.add_argument("--selection",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--anchor",choices=["centered","end_anchored"],default="centered")
    a=p.parse_args();run(a.data.resolve(),a.selection.resolve(),a.output.resolve(),a.anchor)
