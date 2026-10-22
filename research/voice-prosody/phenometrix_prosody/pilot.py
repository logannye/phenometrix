"""Bounded training-only DSP/YIN diagnosis; predictions finish before labels load."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import wave

import numpy as np

from .metrics import Prediction, Reference, cents, oracle_coverage, score

RESEARCH = Path(__file__).resolve().parents[1]
REPO = RESEARCH.parents[1]
RATE = 48000
WINDOW = 1920
HOP = 480
ORIGIN = 768
FIRST_INDEX = 3
FIRST_START = ORIGIN + FIRST_INDEX * HOP - WINDOW


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_mic(path: Path):
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, RATE, "NONE"):
            raise ValueError("Pilot only supports audited mono PCM16 48 kHz microphone files")
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / np.float32(32768)
    return pcm


def yin_predictions(pcm: np.ndarray, targets: np.ndarray):
    # Imported after callers set a single writable, ignored Numba cache.
    import librosa
    f0 = librosa.yin(pcm[FIRST_START:], fmin=50, fmax=700, sr=RATE,
                     frame_length=WINDOW, hop_length=HOP, center=False)
    if len(f0) != len(targets):
        raise ValueError("YIN frame count does not match exact TypeScript target grid")
    predictions = []
    evidence = []
    for estimate, sample in zip(f0, targets):
        window = pcm[int(sample) - WINDOW:int(sample)].astype(np.float64)
        window -= window.mean()
        rms = float(np.sqrt(np.mean(window * window)))
        lag = int(np.clip(round(RATE / float(estimate)), 1, WINDOW - 1))
        left, right = window[:-lag], window[lag:]
        correlation = float(np.dot(left, right) / max(1e-12, np.sqrt(np.dot(left, left) * np.dot(right, right))))
        voiced = rms >= 0.003 and correlation >= 0.55
        predictions.append(Prediction(int(sample), "voiced" if voiced else "unvoiced", float(estimate) if voiced else None))
        evidence.append({"sample": int(sample), "raw_f0_hz": float(estimate), "window_rms": rms,
                         "normalized_lag_correlation": correlation})
    return predictions, evidence


def references_from_array(values: np.ndarray, sample_count: int):
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("Expected four audited PTDB reference columns")
    references = []
    for index, row in enumerate(values):
        hz, probability, rms, correlation = (float(value) for value in row)
        valid = all(math.isfinite(value) for value in row) and (
            (probability == 0 and hz == 0) or (probability == 1 and hz > 0))
        voiced = probability == 1 and hz > 0
        target = ORIGIN + index * HOP
        eligible = target >= WINDOW and target + 1536 <= sample_count
        references.append(Reference(target, hz if voiced else None, voiced, valid, eligible))
    return references


def source_identity():
    paths = ["apps/capture-web/src/voice-dsp.ts", "apps/capture-web/src/dsp-primitives.ts",
             "apps/capture-web/src/voice-pitch-features.ts", "apps/capture-web/src/voice-pitch-features-v2.ts", "apps/capture-web/src/voice-replay.ts",
             "research/voice-prosody/replay.ts", "research/voice-prosody/phenometrix_prosody/pilot.py",
             "research/voice-prosody/phenometrix_prosody/metrics.py", "research/voice-prosody/uv.lock", "pnpm-lock.yaml"]
    return {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True),
            "source_sha256": {name: digest(REPO / name) for name in paths},
            "dependencies": {name: importlib.metadata.version(name) for name in ["numpy", "librosa", "scipy", "numba"]}}


def run_pilot(data_root: Path, output: Path, *, feature_version: str = "v1", selection_manifest: Path | None = None):
    if feature_version not in {"v1", "v2"}:
        raise ValueError("Unknown feature version")
    output.mkdir(parents=True, exist_ok=False)
    (output / "features").mkdir()
    (output / "predictions").mkdir()
    temporary = output / "intermediate"
    temporary.mkdir()
    os.environ.setdefault("NUMBA_CACHE_DIR", str(RESEARCH / "runs" / ".numba-cache"))
    identity = source_identity()
    manifest_path = data_root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    data_manifest = json.loads(manifest_bytes)
    selected = None if selection_manifest is None else json.loads(selection_manifest.read_text())
    selected_ids = None if selected is None else {row["recording_id"] for row in selected["recordings"]}
    microphones = sorted((row for row in data_manifest["files"] if row["role"] == "mic" and
        (selected_ids is None or row["recording_id"] in selected_ids)), key=lambda row: row["recording_id"])
    if selected_ids is not None and {row["recording_id"] for row in microphones} != selected_ids:
        raise ValueError("Selection manifest recordings missing from acquisition manifest")
    if not microphones or any(row["split"] != "train" or not row["primary_eligible"] for row in microphones):
        raise ValueError("This diagnosis runner refuses non-training or universal-prompt recordings")
    mic_ids = {row["recording_id"] for row in microphones}
    reference_rows = {row["recording_id"]: row for row in data_manifest["files"] if row["role"] == "ref" and row["recording_id"] in mic_ids}
    if selected is not None:
        selected_mic = {row["recording_id"]: row["microphone_sha256"] for row in selected["recordings"]}
        selected_ref = {row["recording_id"]: row["reference_sha256"] for row in
            json.loads((selection_manifest.parent / "scores.json").read_text())["per_recording"]}
        if any(row["sha256"] != selected_mic[row["recording_id"]] for row in microphones) or any(
            row["sha256"] != selected_ref[key] for key, row in reference_rows.items()):
            raise ValueError("Original microphone/reference identity changed")
    write_json(output / "data-selection-manifest.json", {"microphones": microphones, "references": list(reference_rows.values())})
    if len(reference_rows) != len(microphones):
        raise ValueError("Each training microphone needs exactly one reference role")
    run_manifest = {"schema_version": "phenometrix.voice-training-pilot.v1", "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(), "identity": identity,
        "data_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "selection_manifest_sha256": None if selection_manifest is None else digest(selection_manifest), "split": "train", "condition": "clean",
        "selection_status": "initial-diagnostic-no-model-or-threshold-selection",
        "feature_version": feature_version, "v2_oracle_prerequisite": 0.75 if feature_version == "v2" else None,
        "reference_timing": "nominal 16+10*k ms; trailing40ms; shared centered64ms eligibility",
        "classical_gate": "existing periodic flag: f0 exists and confidence>=0.55 and agreement>=0.7",
        "yin_gate": "present centered-window RMS>=0.003 and normalized correlation at rounded YIN lag>=0.55",
        "yin_parameters": {"version": "0.11.0", "center": False, "sr": RATE,
            "frame_length": WINDOW, "hop_length": HOP, "fmin": 50, "fmax": 700,
            "trough_threshold": 0.1},
        "timing_claim": "No measured runtime comparison; sequential offline accuracy diagnosis only",
        "recordings": []}
    write_json(output / "manifest.json", run_manifest)
    try:
        # Phase 1: complete every prediction, without opening any reference or laryngograph.
        for row in microphones:
            recording_id = row["recording_id"]
            mic_path = data_root / row["path"]
            if digest(mic_path) != row["sha256"]:
                raise ValueError(f"Microphone identity mismatch: {recording_id}")
            pcm = read_mic(mic_path)
            raw_path = temporary / f"{recording_id}.f32le"
            full_path = temporary / f"{recording_id}.json"
            pcm.astype("<f4").tofile(raw_path)
            subprocess.run(["node", "--import", "tsx", "../../research/voice-prosody/replay.ts",
                "--feature-version", feature_version, "--input-f32", str(raw_path), "--sample-rate", str(RATE), "--output", str(full_path)],
                cwd=REPO / "apps" / "capture-web", check=True, capture_output=True, text=True)
            feature_export_sha = digest(full_path)
            feature_export = json.loads(full_path.read_text())
            frames = feature_export["frames"]
            targets = np.array([frame["referenceSample"] for frame in frames], dtype=np.int64)
            expected_targets = ORIGIN + np.arange(FIRST_INDEX, FIRST_INDEX + len(targets)) * HOP
            if not np.array_equal(targets, expected_targets):
                raise ValueError("TypeScript target-grid mismatch")
            candidate_hz = np.array([[candidate["hz"] if candidate["valid"] else np.nan
                                     for candidate in frame["features"]["candidates"]] for frame in frames], dtype=np.float64)
            candidate_mask = np.isfinite(candidate_hz)
            features = np.array([frame["features"]["values"] for frame in frames], dtype=np.float32)
            feature_path = output / "features" / f"{recording_id}.npz"
            np.savez_compressed(feature_path, features=features, candidate_hz=candidate_hz,
                candidate_mask=candidate_mask, reference_sample=targets,
                feature_names=np.array(feature_export["featureNames"]), sample_rate_hz=RATE,
                pcm_sample_count=len(pcm))
            dsp = []
            dsp_evidence = []
            for frame in frames:
                baseline = frame["features"]["baseline"]
                periodic = baseline["f0Hz"] is not None and baseline["f0Confidence"] >= 0.55 and baseline["estimatorAgreement"] >= 0.7
                dsp.append(Prediction(frame["referenceSample"], "voiced" if periodic else "unvoiced",
                                      baseline["f0Hz"] if periodic else None))
                dsp_evidence.append({"sample": frame["referenceSample"], "raw_f0_hz": baseline["f0Hz"],
                    "confidence": baseline["f0Confidence"], "agreement": baseline["estimatorAgreement"],
                    "rms": baseline["rms"]})
            yin, yin_evidence = yin_predictions(pcm, targets)
            prediction_path = output / "predictions" / f"{recording_id}.json"
            write_json(prediction_path, {"recording_id": recording_id, "microphone_sha256": row["sha256"],
                "sample_count": len(pcm), "feature_sha256": digest(feature_path),
                "dsp": [asdict(value) for value in dsp], "yin": [asdict(value) for value in yin],
                "dsp_evidence": dsp_evidence, "yin_evidence": yin_evidence})
            run_manifest["recordings"].append({"recording_id": recording_id, "speaker": row["speaker"],
                "pair": row["pair"], "prompt": row["prompt"], "sample_count": len(pcm),
                "microphone_sha256": row["sha256"], "feature_sha256": digest(feature_path),
                "typescript_feature_export_sha256": feature_export_sha,
                "raw_f32le_sha256": feature_export["inputSha256"], "prediction_sha256": digest(prediction_path)})
            raw_path.unlink()
            full_path.unlink()
            write_json(output / "manifest.json", run_manifest)
            print(f"Predicted {recording_id}: {len(targets)} complete causal frames", flush=True)
        temporary.rmdir()
        # Freeze the complete prediction bundle before loading scorer-only labels.
        write_json(output / "prediction-freeze.json", {"completed_at": datetime.now(timezone.utc).isoformat(),
            "recordings": run_manifest["recordings"], "label_content_read": False})
        by_speaker = defaultdict(lambda: {"references": [], "dsp": [], "yin": [], "candidates": {}})
        per_recording = []
        error_traces = []
        for row in run_manifest["recordings"]:
            recording_id = row["recording_id"]
            ref_row = reference_rows[recording_id]
            if ref_row["split"] != "train":
                raise ValueError("Scorer refuses non-training reference")
            ref_path = data_root / ref_row["path"]
            if digest(ref_path) != ref_row["sha256"]:
                raise ValueError(f"Reference identity mismatch: {recording_id}")
            reference = references_from_array(np.loadtxt(ref_path), row["sample_count"])
            reference_samples = {value.sample for value in reference}
            raw = json.loads((output / "predictions" / f"{recording_id}.json").read_text())
            # Array alignment never fits a shift. Beyond-reference predictions are retained in raw output but not scored.
            predictions = {name: [Prediction(**value) for value in raw[name] if value["sample"] in reference_samples]
                           for name in ["dsp", "yin"]}
            cache = np.load(output / "features" / f"{recording_id}.npz", allow_pickle=False)
            candidates = {int(sample): [float(hz) for hz, valid in zip(hz_row, mask) if valid]
                          for sample, hz_row, mask in zip(cache["reference_sample"], cache["candidate_hz"], cache["candidate_mask"])}
            scores = {name: score(reference, values) for name, values in predictions.items()}
            oracle = oracle_coverage(reference, candidates)
            per_recording.append({"recording_id": recording_id, "speaker": row["speaker"],
                "pair": row["pair"], "scores": scores, "candidate_oracle": oracle,
                "reference_sha256": ref_row["sha256"],
                "prediction_targets_beyond_reference": len(raw["dsp"]) - len(predictions["dsp"])})
            group = by_speaker[row["speaker"]]
            # Rebase recording coordinates in aggregation only; never synthesize time continuity.
            offset = sum(item.eligible or not item.eligible for item in group["references"]) * (RATE * 60)
            group["references"].extend(Reference(value.sample + offset, value.f0_hz, value.voiced,
                                                   value.valid, value.eligible) for value in reference)
            for name in ["dsp", "yin"]:
                group[name].extend(Prediction(value.sample + offset, value.state, value.f0_hz) for value in predictions[name])
            group["candidates"].update({sample + offset: values for sample, values in candidates.items()})
            lookup = {value.sample: value for value in predictions["dsp"]}
            errors = []
            for value in reference:
                if not (value.valid and value.eligible and value.voiced):
                    continue
                estimate = lookup.get(value.sample)
                error = None if estimate is None or estimate.f0_hz is None else abs(cents(estimate.f0_hz, value.f0_hz))
                covered = any(abs(cents(hz, value.f0_hz)) <= 50 for hz in candidates.get(value.sample, []))
                if error is None or error > 50:
                    errors.append({"sample": value.sample, "time_ms": value.sample / 48,
                        "reference_hz": value.f0_hz, "dsp_state": estimate.state if estimate else "unavailable",
                        "dsp_hz": estimate.f0_hz if estimate else None, "absolute_cents": error,
                        "candidate_hz": candidates.get(value.sample, []), "oracle_covered": covered})
            # Fixed deterministic representative rows: first five errors of each failure category per recording.
            categories = {"candidate_missing": [e for e in errors if not e["oracle_covered"]],
                "resolver_or_gate_missed": [e for e in errors if e["oracle_covered"]],
                "octave": [e for e in errors if e["absolute_cents"] is not None and abs(e["absolute_cents"] - 1200) <= 50]}
            error_traces.append({"recording_id": recording_id, "counts": {key: len(value) for key, value in categories.items()},
                                 "examples": {key: value[:5] for key, value in categories.items()}})
        speaker_scores = {speaker: {"scores": {name: score(group["references"], group[name]) for name in ["dsp", "yin"]},
            "candidate_oracle": oracle_coverage(group["references"], group["candidates"])}
            for speaker, group in sorted(by_speaker.items())}
        macro = {}
        for name in ["dsp", "yin"]:
            macro[name] = {metric: float(np.mean([value["scores"][name][metric] for value in speaker_scores.values()
                                              if value["scores"][name][metric] is not None]))
                          for metric in ["joint_error", "reference_voiced_success_rate", "voicing_false_positive_rate",
                                         "voicing_false_negative_or_unavailable_rate", "octave_error_rate_reference_voiced"]}
        macro["candidate_oracle_coverage"] = float(np.mean([value["candidate_oracle"]["coverage"] for value in speaker_scores.values()]))
        report = {"evidence_status": "clean-training-only-initial-diagnostic", "recording_count": len(per_recording),
            "speaker_count": len(speaker_scores), "duration_seconds": sum(row["sample_count"] for row in run_manifest["recordings"]) / RATE,
            "speaker_macro": macro, "per_speaker": speaker_scores, "per_recording": per_recording}
        write_json(output / "scores.json", report)
        write_json(output / "error-traces.json", error_traces)
        after = source_identity()
        if identity["source_sha256"] != after["source_sha256"]:
            raise RuntimeError("Relevant source changed while pilot ran; preserve outputs as failed")
        run_manifest.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
            "scores_sha256": digest(output / "scores.json"), "error_traces_sha256": digest(output / "error-traces.json")})
        write_json(output / "manifest.json", run_manifest)
        return report
    except Exception as error:
        run_manifest.update({"status": "failed", "failure": f"{type(error).__name__}: {error}",
                             "failed_at": datetime.now(timezone.utc).isoformat()})
        write_json(output / "manifest.json", run_manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=RESEARCH / "data" / "ptdb-tug")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--feature-version", choices=["v1", "v2"], default="v1")
    args = parser.parse_args()
    result = run_pilot(args.data_root.resolve(), args.output.resolve(), feature_version=args.feature_version, selection_manifest=args.selection_manifest)
    print(json.dumps({key: result[key] for key in ["evidence_status", "recording_count", "speaker_count", "duration_seconds", "speaker_macro"]}, indent=2))


if __name__ == "__main__":
    main()
