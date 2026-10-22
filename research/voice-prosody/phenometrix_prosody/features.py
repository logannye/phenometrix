"""Input-only, versioned TypeScript feature caching for the delayed experiment."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from .metrics import Prediction
from .pilot import RESEARCH, REPO, RATE, ORIGIN, HOP, FIRST_INDEX, digest, write_json, read_mic, yin_predictions, source_identity

CONDITIONS = ("clean", "noise10db", "first_difference", "clip01")


def transform(pcm, condition, identity, split):
    """Offline corruption construction only; no reference content or inference scaling."""
    if condition not in CONDITIONS or split not in {"train", "development", "sealed-test"}:
        raise ValueError("Unknown condition or split")
    seed = int.from_bytes(hashlib.sha256(f"20261022:{split}:{identity}:{condition}".encode()).digest()[:8], "little")
    metadata = {"condition": condition, "seed": seed, "target_preserved": True}
    if condition == "clean":
        result = pcm.copy()
    elif condition == "noise10db":
        noise = np.random.default_rng(seed).standard_normal(len(pcm))
        power = float(np.mean(np.square(pcm.astype(np.float64))))
        noise *= np.sqrt(power / (10 * max(1e-30, float(np.mean(noise ** 2)))))
        result = (pcm + noise).astype(np.float32)
        actual_noise = result.astype(np.float64) - pcm
        achieved = None if power == 0 else float(10 * np.log10(power / np.mean(actual_noise ** 2)))
        metadata.update(requested_snr_db=10, achieved_snr_db=achieved,
                        construction="whole clean recording power; offline corruption only")
    elif condition == "first_difference":
        result = np.diff(pcm, prepend=np.float32(0)).astype(np.float32)
        metadata["filter"] = "y[n]=x[n]-x[n-1], x[-1]=0; no normalization"
    else:
        result = np.clip(pcm, -.1, .1).astype(np.float32)
        metadata["clip_amplitude"] = .1
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite transformed PCM")
    return result, metadata


def cache(data_root, output, split, conditions=CONDITIONS):
    if split not in {"train", "development"}:
        raise ValueError("This pre-freeze cache refuses sealed test content")
    if not conditions or len(set(conditions)) != len(conditions) or any(c not in CONDITIONS for c in conditions):
        raise ValueError("Unique declared conditions required")
    data_path = data_root / "manifest.json"
    data_bytes = data_path.read_bytes()
    rows = sorted((r for r in json.loads(data_bytes)["files"] if r["role"] == "mic" and r["split"] == split), key=lambda r: r["recording_id"])
    if not rows or any(not r["primary_eligible"] for r in rows):
        raise ValueError("Expected eligible source recordings")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("features", "predictions", "intermediate", "source-snapshot"):
        (output / name).mkdir()
    os.environ.setdefault("NUMBA_CACHE_DIR", str(RESEARCH / "runs" / ".numba-cache"))
    identity = source_identity()
    identity["source_sha256"][str(Path(__file__).relative_to(REPO))] = digest(Path(__file__))
    for name, expected in identity["source_sha256"].items():
        target = output / "source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / name).read_bytes())
        if digest(target) != expected:
            raise ValueError("Source snapshot mismatch")
    (output / "acquisition-manifest.json").write_bytes(data_bytes)
    write_json(output / "microphone-selection.json", rows)
    manifest = {"schema": "phenometrix-input-only-feature-cache-v1", "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(), "split": split, "feature_version": "v2",
        "conditions": list(conditions), "identity": identity, "data_manifest_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "reference_or_laryngograph_content_read": False, "measurement_delay_ms": 20,
        "time_contract": "feature end e; target e-960 samples; earliest availability e; 48kHz; 20ms future support relative to target",
        "recordings": []}
    write_json(output / "manifest.json", manifest)
    try:
        for row in rows:
            mic_path = data_root / row["path"]
            if digest(mic_path) != row["sha256"]:
                raise ValueError("Microphone bytes changed")
            clean = read_mic(mic_path)
            for condition in conditions:
                source_id = row["recording_id"]
                record_id = source_id + "__" + condition
                pcm, corruption = transform(clean, condition, source_id, split)
                raw = output / "intermediate" / (record_id + ".f32le")
                full = output / "intermediate" / (record_id + ".json")
                pcm.astype("<f4").tofile(raw)
                subprocess.run(["node", "--import", "tsx", "../../research/voice-prosody/replay.ts", "--feature-version", "v2",
                    "--input-f32", str(raw), "--sample-rate", str(RATE), "--output", str(full)],
                    cwd=REPO / "apps/capture-web", check=True, capture_output=True, text=True)
                export_sha = digest(full)
                export = json.loads(full.read_text())
                frames = export["frames"]
                ends = np.array([f["referenceSample"] for f in frames], dtype=np.int64)
                if not np.array_equal(ends, ORIGIN + np.arange(FIRST_INDEX, FIRST_INDEX + len(ends)) * HOP):
                    raise ValueError("Unexpected acquisition grid")
                hz = np.array([[c["hz"] if c["valid"] else np.nan for c in f["features"]["candidates"]] for f in frames], dtype=np.float64)
                file = output / "features" / (record_id + ".npz")
                np.savez_compressed(file, features=np.array([f["features"]["values"] for f in frames], np.float32),
                    candidate_hz=hz, candidate_mask=np.isfinite(hz), reference_sample=ends,
                    feature_names=np.array(export["featureNames"]), sample_rate_hz=RATE, pcm_sample_count=len(pcm))
                dsp = []
                for f in frames:
                    b = f["features"]["baseline"]
                    periodic = b["f0Hz"] is not None and b["f0Confidence"] >= .55 and b["estimatorAgreement"] >= .7
                    dsp.append(asdict(Prediction(f["referenceSample"], "voiced" if periodic else "unvoiced", b["f0Hz"] if periodic else None)))
                yin, yin_evidence = yin_predictions(pcm, ends)
                prediction = output / "predictions" / (record_id + ".json")
                write_json(prediction, {"recording_id": record_id, "dsp": dsp, "yin": [asdict(p) for p in yin], "yin_evidence": yin_evidence})
                manifest["recordings"].append({"recording_id": record_id, "source_recording_id": source_id,
                    "speaker": row["speaker"], "pair": row["pair"], "prompt": row["prompt"], "condition": condition,
                    "sample_count": len(pcm), "microphone_sha256": row["sha256"], "transformed_pcm_sha256": digest(raw),
                    "feature_sha256": digest(file), "prediction_sha256": digest(prediction),
                    "typescript_export_sha256": export_sha, "corruption": corruption})
                raw.unlink(); full.unlink()
                write_json(output / "manifest.json", manifest)
                print(f"Cached {record_id}: {len(ends)} frames", flush=True)
        for name, expected in identity["source_sha256"].items():
            if digest(REPO / name) != expected:
                raise ValueError("Relevant source changed during cache construction")
        (output / "intermediate").rmdir()
        manifest.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat())
        write_json(output / "manifest.json", manifest)
        return manifest
    except BaseException as error:
        manifest.update(status="failed", failure=repr(error))
        write_json(output / "manifest.json", manifest)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["train", "development"], required=True)
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    a = p.parse_args(); cache(a.data.resolve(), a.output.resolve(), a.split, tuple(a.conditions.split(",")))

if __name__ == "__main__":
    main()
