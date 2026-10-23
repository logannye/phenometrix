"""Reproduce the bounded, reference-free all-training frozen NumPy/Torch check.

Run from the repository with its existing research Python, passing --repo.
No source/model/plan mutation, fitting, reference read, or NumPy forward occurs.
A fresh output directory is required; the fixed source is preserved separately.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

WEIGHTS = "06110469a9a643fb16806905569ebc939db2c0bdd5b3a6a1593b31789453d10d"
EXPECTED_RECORDS = 960
EXPECTED_FRAMES = 676384
PROBABILITY_ABSOLUTE_TOLERANCE = 1e-6


def sha(data):
    return hashlib.sha256(data).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def run(repo: Path, output: Path):
    repo = repo.resolve(); output = output.resolve()
    research = repo / "research/voice-prosody"
    if not output.is_relative_to(research / "runs"):
        raise ValueError("Receipt output must stay under ignored research runs")
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "full-torch-parity.json"
    ledger_path = output / "full-torch-parity-input-ledger.json"
    if receipt_path.exists() or ledger_path.exists():
        raise ValueError("Refusing to overwrite a completed or failed audit artifact")
    sys.path.insert(0, str(research))
    from phenometrix_prosody.model import VoiceCandidateModel, ModelConfig, FEATURE_NAMES_V2
    ledger = {}

    def read(path, role, expected=None):
        path = path.resolve()
        if not path.is_relative_to(repo):
            raise ValueError("Input escaped repository")
        if any(part in {"sealed_test", "sealed-test"} for part in path.parts) or role not in {
            "model_source", "headroom_source", "declaration", "prediction_freeze", "feature_manifest",
            "model_manifest", "weights", "feature", "prediction", "verification_source"}:
            raise ValueError("Input role/path not admitted; no references may be read")
        data = path.read_bytes(); digest = sha(data)
        if expected is not None and digest != expected:
            raise ValueError(f"Input identity mismatch: {path.relative_to(repo)}")
        key = str(path.relative_to(repo))
        if key in ledger and ledger[key]["sha256"] != digest:
            raise ValueError("Input changed during audit")
        ledger[key] = {"role": role, "sha256": digest, "bytes": len(data)}
        return data

    started = now(); began = time.monotonic()
    result = {"schema": "phenometrix-selector-full-train-torch-parity.v1", "started_at": started,
        "status": "running", "device": "cpu", "torch_threads": 1, "torch": torch.__version__,
        "numpy": np.__version__, "python": platform.python_version(),
        "reference_content_read": False, "development_or_test_content_read": False, "fitting_performed": False,
        "warnings_suppressed": False, "probability_absolute_tolerance": PROBABILITY_ABSOLUTE_TOLERANCE,
        "warning_interpretation": "This output check does not establish the cause of the prior NumPy matmul warnings.",
        "candidate_comparison_scope": "All stored emitted indices/states/Hz. Frozen index is -1 on unvoiced frames; ungated candidate argmax on those frames was not retained and is not claimed verified.",
        "frames": 0, "records": 0, "finite_frozen_probability_frames": 0, "finite_torch_probability_frames": 0,
        "candidate_present_frames": 0, "frozen_voiced_with_candidate_frames": 0, "frozen_unvoiced_frames": 0,
        "frozen_unavailable_frames": 0, "max_absolute_probability_difference": 0.,
        "probability_tolerance_exceedances": 0, "gate_mismatches": 0, "state_mismatches": 0,
        "emitted_candidate_index_mismatches": 0, "emitted_hz_mismatches": 0,
        "mismatch_examples": []}
    try:
        torch.set_num_threads(1)
        folder = research / "runs/selector-headroom-v1"
        declaration_path = folder / "declaration.json"
        freeze_path = folder / "prediction-freeze.json"
        freeze_bytes = read(freeze_path, "prediction_freeze")
        freeze = json.loads(freeze_bytes)
        if freeze.get("schema") != "phenometrix-selector-headroom-input-freeze.v1" or freeze.get("status") != "predictions_frozen":
            raise ValueError("Incomplete prediction freeze")
        declaration_bytes = read(declaration_path, "declaration", freeze["declaration_sha256"])
        declaration = json.loads(declaration_bytes)
        if declaration["model_weights_sha256"] != WEIGHTS:
            raise ValueError("Original frozen checkpoint binding changed")
        pins = declaration["input_manifest_sha256"]
        feature_folder = research / "runs/delayed-train-features"
        feature_manifest_bytes = read(feature_folder / "manifest.json", "feature_manifest", pins["delayed-train-features/manifest.json"])
        manifest = json.loads(feature_manifest_bytes)
        model_folder = research / "runs/delayed-mlp-seed20261022"
        model_manifest_bytes = read(model_folder / "manifest.json", "model_manifest", pins["delayed-mlp-seed20261022/manifest.json"])
        model_manifest = json.loads(model_manifest_bytes)
        if (manifest["status"] != "complete" or manifest["split"] != "train"
                or model_manifest["status"] != "complete" or model_manifest["kind"] != "framewise-mlp"
                or model_manifest["feature_run_manifest_sha256"] != sha(feature_manifest_bytes)):
            raise ValueError("Training feature/checkpoint identity mismatch")
        model_source = research / "phenometrix_prosody/model.py"
        model_source_key = str(model_source.relative_to(repo))
        read(model_source, "model_source", declaration["source_sha256"][model_source_key])
        if declaration["source_sha256"][model_source_key] != model_manifest["source_sha256"][model_source_key]:
            raise ValueError("Current model equations differ from training snapshot")
        headroom_source = research / "phenometrix_prosody/selector_headroom.py"
        read(headroom_source, "headroom_source", declaration["source_sha256"][str(headroom_source.relative_to(repo))])
        verification_source = Path(__file__).resolve()
        read(verification_source, "verification_source")
        weight_bytes = read(model_folder / "weights.pt", "weights", WEIGHTS)
        if model_manifest["weights_sha256"] != WEIGHTS:
            raise ValueError("Model manifest does not identify original weights")
        checkpoint = torch.load(BytesIO(weight_bytes), map_location="cpu", weights_only=True)
        if checkpoint["config"] != model_manifest["model"]["config"]:
            raise ValueError("Checkpoint config differs from manifest")
        config = ModelConfig(**checkpoint["config"])
        if config.kind != "framewise-mlp" or config.feature_version != "v2":
            raise ValueError("Only original framewise v2 model admitted")
        model = VoiceCandidateModel(config).cpu().eval()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        if not all(torch.isfinite(value).all() for value in model.state_dict().values()) or not bool(model.scaling_fitted) or not torch.all(model.feature_scale > 0):
            raise ValueError("Invalid frozen weights/scaling")
        records = [r for r in freeze["records"] if r["split"] == "train"]
        inputs = {r["recording_id"]: r for r in manifest["recordings"]}
        if (len(records) != EXPECTED_RECORDS or len(inputs) != EXPECTED_RECORDS
                or len({r["recording_id"] for r in records}) != EXPECTED_RECORDS
                or {r["recording_id"] for r in records} != set(inputs)):
            raise ValueError("Incomplete or duplicated training inventory")
        admitted_speakers = {f"{sex}{i:02d}" for sex in "FM" for i in (2, 4, 7, 8, 9, 10)}
        if {r["speaker"] for r in inputs.values()} != admitted_speakers:
            raise ValueError("Nontraining speaker in input inventory")
        for record in records:
            identity = record["recording_id"]; meta = inputs[identity]
            feature_path = feature_folder / "features" / (identity + ".npz")
            feature_bytes = read(feature_path, "feature", meta["feature_sha256"])
            if model_manifest["feature_hashes"][identity] != meta["feature_sha256"]:
                raise ValueError("Training feature not bound to original checkpoint")
            prediction_path = (folder / record["path"]).resolve()
            if not prediction_path.is_relative_to(folder / "train-predictions"):
                raise ValueError("Prediction escaped exact training folder")
            prediction_bytes = read(prediction_path, "prediction", record["sha256"])
            with np.load(BytesIO(feature_bytes), allow_pickle=False) as z:
                f = {key: z[key].copy() for key in z.files}
            with np.load(BytesIO(prediction_bytes), allow_pickle=False) as z:
                old = {key: z[key].copy() for key in z.files}
            x, mask, hz = f["features"], f["candidate_mask"], f["candidate_hz"]
            n = len(x)
            if (n != record["frames"] or x.shape != (n, 57) or x.dtype != np.float32 or mask.shape != (n, 6)
                    or mask.dtype != bool or hz.shape != (n, 6) or not np.isfinite(x).all()
                    or list(f["feature_names"]) != list(FEATURE_NAMES_V2)
                    or not np.isfinite(hz[mask]).all() or not ((hz[mask] >= 50) & (hz[mask] <= 700)).all()
                    or not np.isnan(hz[~mask]).all() or int(f["pcm_sample_count"]) != meta["sample_count"]
                    or int(f["sample_rate_hz"]) != 48000
                    or not np.array_equal(f["reference_sample"], np.arange(2208, meta["sample_count"]+1, 480))
                    or not np.array_equal(old["target_sample"], f["reference_sample"] - 960)
                    or any(old[key].shape != (n,) for key in ("voicing_probability", "candidate_index", "state", "f0_hz"))):
                raise ValueError("Malformed input/output shape, identity or grid")
            probabilities, choices = [], []
            with torch.inference_mode():
                for start in range(0, n, 4096):
                    xt = torch.from_numpy(x[start:start+4096]); mt = torch.from_numpy(mask[start:start+4096])
                    v, logits, state_out = model(xt, mt, model.initial_state(len(xt)))
                    if not torch.isfinite(v).all() or not torch.isfinite(logits).all() or not torch.isfinite(state_out).all():
                        raise ValueError("Nonfinite Torch output")
                    probabilities.append(torch.sigmoid(v[:, 0].double()).numpy())
                    choices.append(torch.where(mt, logits, torch.full_like(logits, -torch.inf)).argmax(dim=1).numpy())
            p = np.concatenate(probabilities); index = np.concatenate(choices)
            index[(p < .5) | ~mask.any(axis=1)] = -1
            state = np.where(p < .5, 0, np.where(index >= 0, 1, 2)).astype(np.int8)
            emitted = np.full(n, np.nan); voiced = index >= 0
            emitted[voiced] = hz[np.flatnonzero(voiced), index[voiced]]
            oldp = old["voicing_probability"]
            if not np.isfinite(oldp).all() or np.any((oldp < 0) | (oldp > 1)) or not np.isfinite(p).all():
                raise ValueError("Nonfinite or invalid probability")
            absolute = np.abs(p - oldp)
            bad_gate = (p >= .5) != (oldp >= .5)
            bad_state = state != old["state"]
            bad_index = index != old["candidate_index"]
            bad_hz = ~((emitted == old["f0_hz"]) | (np.isnan(emitted) & np.isnan(old["f0_hz"])))
            result["records"] += 1; result["frames"] += n
            result["finite_frozen_probability_frames"] += int(np.isfinite(oldp).sum())
            result["finite_torch_probability_frames"] += int(np.isfinite(p).sum())
            result["candidate_present_frames"] += int(mask.any(axis=1).sum())
            result["frozen_voiced_with_candidate_frames"] += int((old["state"] == 1).sum())
            result["frozen_unvoiced_frames"] += int((old["state"] == 0).sum())
            result["frozen_unavailable_frames"] += int((old["state"] == 2).sum())
            result["max_absolute_probability_difference"] = max(result["max_absolute_probability_difference"], float(absolute.max()))
            result["probability_tolerance_exceedances"] += int((absolute > PROBABILITY_ABSOLUTE_TOLERANCE).sum())
            for name, bad in [("gate_mismatches", bad_gate), ("state_mismatches", bad_state),
                              ("emitted_candidate_index_mismatches", bad_index), ("emitted_hz_mismatches", bad_hz)]:
                result[name] += int(bad.sum())
            for frame in np.flatnonzero(bad_gate | bad_state | bad_index | bad_hz)[:max(0, 10-len(result["mismatch_examples"]))]:
                result["mismatch_examples"].append({"recording_id": identity, "frame_index": int(frame),
                    "gate": bool(bad_gate[frame]), "state": bool(bad_state[frame]),
                    "candidate_index": bool(bad_index[frame]), "emitted_hz": bool(bad_hz[frame])})
            if result["records"] % 160 == 0:
                print(json.dumps({"records": result["records"], "frames": result["frames"],
                                  "max_probability_error": result["max_absolute_probability_difference"]}), flush=True)
        if result["frames"] != EXPECTED_FRAMES:
            raise ValueError("Acquired frame count differs from fixed inventory")
        # Rehash all exact inputs, without opening any reference or waveform.
        for path, binding in ledger.items():
            if sha((repo / path).read_bytes()) != binding["sha256"]:
                raise ValueError("Input changed before receipt freeze")
        result.update(status="complete", input_bytes_stable=True,
            pass_result=all(result[key] == 0 for key in ("probability_tolerance_exceedances", "gate_mismatches", "state_mismatches",
                                                        "emitted_candidate_index_mismatches", "emitted_hz_mismatches")))
        result["identities"] = {"prediction_freeze_sha256": sha(freeze_bytes), "declaration_sha256": sha(declaration_bytes),
            "feature_manifest_sha256": sha(feature_manifest_bytes), "model_manifest_sha256": sha(model_manifest_bytes),
            "weights_sha256": WEIGHTS, "model_source_sha256": ledger[model_source_key]["sha256"],
            "verification_source_sha256": ledger[str(verification_source.relative_to(repo))]["sha256"]}
    except BaseException as error:
        result.update(status="failed", pass_result=False, failure=repr(error))
    finally:
        ledger_bytes = (json.dumps(ledger, sort_keys=True, indent=2) + "\n").encode()
        ledger_path.write_bytes(ledger_bytes)
        result.update(completed_at=now(), elapsed_seconds=time.monotonic()-began,
            input_ledger_sha256=sha(ledger_bytes), input_file_count=len(ledger),
            verified_feature_files=sum(v["role"] == "feature" for v in ledger.values()),
            verified_prediction_files=sum(v["role"] == "prediction" for v in ledger.values()))
        receipt_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
    if not result["pass_result"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.repo, args.output)
