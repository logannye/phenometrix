"""Freeze the predetermined PTDB fallback workload without reading references.

Creates a new ignored run directory. Existing historically sealed records are
explicitly admitted here as observed development; historical loaders are intact.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from .pilot import read_mic

SCHEMA = "phenometrix.natural-browser-inputs.v1"
SPEAKERS = ("F01", "F03", "F05", "F06", "M01", "M03", "M05", "M06")
CONDITIONS = ("clean", "first_difference")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def ptdb_prefix(pcm):
    pcm = np.asarray(pcm)
    if pcm.ndim != 1 or not np.isfinite(pcm).all():
        raise ValueError("Finite mono source required")
    return np.concatenate((np.zeros(192, np.float32), pcm.astype(np.float32)))

def first_sources(acquisition):
    result = []
    for speaker in SPEAKERS:
        rows = sorted((r for r in acquisition["files"] if r["role"] == "mic" and r["speaker"] == speaker
                       and r["split"] in ("development", "sealed_test") and r["primary_eligible"]), key=lambda r:r["recording_id"])
        if not rows:
            raise ValueError("Missing predetermined speaker; do not replace")
        result.append(rows[0])
    return result

def prepare(data, output, decision):
    decision_bytes = decision.read_bytes()
    disposition = json.loads(decision_bytes)
    if disposition.get("external_assessment_admitted") is not False or disposition.get("candidate_passed_development") is not False:
        raise ValueError("PTDB H/H fallback needs explicit bound no-admission/no-candidate decision")
    acquisition_path = data / "manifest.json"
    acquisition = json.loads(acquisition_path.read_text())
    selected = first_sources(acquisition)
    output.mkdir(parents=True, exist_ok=False)
    (output / "pcm").mkdir()
    (output / "decision.json").write_bytes(decision_bytes)
    (output / "acquisition-manifest.json").write_bytes(acquisition_path.read_bytes())
    manifest = dict(schema=SCHEMA, status="preparing", createdAt=datetime.now(timezone.utc).isoformat(),
        adapter="ptdb-prefix-192-v1", adapterSourceSha256=sha(__file__), decisionSha256=sha(decision),
        acquisitionManifestSha256=sha(acquisition_path), sampleRateHz=48000, prefixSamples=192,
        referenceOriginSamples=768, firstCommonEligibleOriginalTarget=3168, supportSamples=1920, hopSamples=480, delaySamples=960,
        playlist=["H", "H"], selection="First metadata-ordered complete acquired source per eight non-TRAIN speakers; no outcome selection",
        referenceContentRead=False, dataRole="previously-observed-development-browser-implementation-only", records=[])
    (output / "selection.json").write_text(json.dumps(selected, indent=2) + "\n")
    for row in selected:
        path = (data / row["path"]).resolve()
        if not path.is_relative_to(data.resolve()) or sha(path) != row["sha256"]:
            raise ValueError("Source authentication failed")
        pcm = read_mic(path)
        ref = [r for r in acquisition["files"] if r["role"] == "ref" and r["recording_id"] == row["recording_id"]]
        if len(ref) != 1:
            raise ValueError("Exactly one authenticated reference identity required")
        for condition in CONDITIONS:
            transformed = pcm if condition == "clean" else np.diff(pcm, prepend=np.float32(0)).astype(np.float32)
            prefixed = ptdb_prefix(transformed)
            if not 1920 <= len(prefixed) <= 48000 * 120:
                raise ValueError("Whole source outside declared runtime bound; no truncation/replacement")
            name = row["recording_id"] + "__" + condition
            target = output / "pcm" / (name + ".f32le")
            prefixed.astype("<f4").tofile(target)
            manifest["records"].append(dict(id=name, speaker=row["speaker"], sourceRecordingId=row["recording_id"], condition=condition,
                pcmPath=str(target.relative_to(output)), pcmSha256=sha(target), sampleCount=len(prefixed), originalSampleCount=len(pcm),
                microphoneSha256=row["sha256"], reference=ref[0], adapter="ptdb-prefix-192-v1", fullSource=True))
    index = np.zeros(48137, np.float32)
    index[:16] = (np.arange(16, dtype=np.float32) + 1) / 32
    index[16:] = .1 * np.sin(2 * np.pi * 200 * np.arange(len(index)-16) / 48000)
    target = output / "pcm/index-origin.f32le"; index.astype("<f4").tofile(target)
    manifest["indexFixture"] = dict(id="index-origin", pcmPath=str(target.relative_to(output)), pcmSha256=sha(target),
        sampleCount=len(index), adapter="synthetic-index-v1")
    manifest["status"] = "frozen"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return dict(status="frozen", manifestSha256=sha(output / "manifest.json"), recordings=len(manifest["records"]))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--decision", type=Path, required=True); a=p.parse_args()
    print(json.dumps(prepare(a.data.resolve(), a.output.resolve(), a.decision.resolve())))

if __name__ == "__main__": main()
