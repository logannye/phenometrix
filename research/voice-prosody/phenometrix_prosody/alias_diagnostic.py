"""Versioned no-fit diagnosis of frozen subharmonic selections; no test data."""
from __future__ import annotations
import argparse
import base64
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from . import evaluate as ev
from .model import FEATURE_NAMES_V2, VoiceCandidateModel, numpy_step
from .pilot import REPO, RESEARCH, yin_predictions

FREQUENCIES = [75, 100, 125, 150, 175, 200, 225, 250, 275, 290, 295, 299, 299.99, 299.999,
               300, 300.001, 300.01, 301, 305, 310, 325, 350, 375, 400, 450, 500, 550, 600]
PROFILES = {"fundamental": [1, 0, 0], "harmonic": [1, .5, .25], "missing-fundamental": [0, 1, .5]}
MODEL_DIRS = {"mlp": "delayed-mlp-seed20261022", "gru": "delayed-gru-seed20261022",
              "augmented_mlp": "delayed-augmented-mlp-seed20261022"}


def infer(feature, config, weights, *, reset_every_frame=False):
    state = np.zeros((1, 32), np.float32)
    rows = []
    for x, mask, hz in zip(feature["features"], feature["candidate_mask"], feature["candidate_hz"]):
        if reset_every_frame:
            state.fill(0)
        voicing, logits, state = numpy_step(config, weights, x[None], mask[None], state)
        p = float(np.exp(-np.logaddexp(0., -float(voicing[0, 0]))))
        slot = int(np.argmax(np.where(mask, logits[0], -np.inf))) if mask.any() else None
        selected = float(hz[slot]) if slot is not None and p >= .5 else None
        rows.append({"voicing_probability": p, "candidate_logits": logits[0].tolist(), "slot": slot,
                     "f0_hz": selected, "state": "unvoiced" if p < .5 else "voiced" if slot is not None else "unavailable"})
    return rows


def intervention(feature, weights, name):
    """Artificial input interventions, never a production feature contract."""
    out = {k: v.copy() for k, v in feature.items()}
    x, hz, mask = out["features"], out["candidate_hz"], out["candidate_mask"]
    if name.startswith("remove:"):
        slot = int(name.split(":")[1])
        for i in range(len(x)):
            if not mask[i, slot]:
                continue
            keep = [j for j in range(6) if mask[i, j] and j != slot]
            blocks = x[i, :48].reshape(6, 8).copy()
            frequencies = hz[i].copy()
            x[i, :48] = 0; hz[i] = np.nan; mask[i] = False
            for dest, source in enumerate(keep):
                x[i, dest * 8:(dest + 1) * 8] = blocks[source]
                hz[i, dest] = frequencies[source]; mask[i, dest] = True
            x[i, 55] = len(keep) / 6
    elif name.startswith("mask:"):
        mask[:, int(name.split(":")[1])] = False
    elif name.startswith("swap:"):
        a = int(name.split(":")[1]); b = a + 1
        blocks = x[:, :48].reshape(-1, 6, 8)
        blocks[:, [a, b]] = blocks[:, [b, a]].copy()
        hz[:, [a, b]] = hz[:, [b, a]].copy(); mask[:, [a, b]] = mask[:, [b, a]].copy()
    elif name.startswith("mean:"):
        j = int(name.split(":")[1]); x[:, j] = weights["feature_mean"][j]
    elif name == "global-means":
        x[:, 48:] = weights["feature_mean"][48:]
    elif name == "scaled-clip3":
        x[:] = weights["feature_mean"] + np.clip((x - weights["feature_mean"]) / weights["feature_scale"], -3, 3) * weights["feature_scale"]
    elif name not in {"baseline", "state-reset-every-frame"}:
        raise ValueError("Unknown declared intervention")
    return out


def summary(rows, truth, eligible):
    picked = [r for r, use in zip(rows, eligible) if use]
    hz = np.array([r["f0_hz"] if r["f0_hz"] is not None else np.nan for r in picked])
    error = 1200 * np.log2(hz / truth)
    finite = np.isfinite(error)
    hist = {str(i): sum(r["slot"] == i for r in picked) for i in range(6)}
    return {"frames": len(picked), "voiced": int(finite.sum()), "success50": float(np.mean(np.abs(error) <= 50)),
            "octave_error": float(np.mean(finite & (np.abs(error) >= 1150))),
            "median_signed_cents": float(np.nanmedian(error)) if finite.any() else None,
            "selected_slot_counts": hist}


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "features").mkdir(); (output / "source-snapshot").mkdir()
    training = RESEARCH / "runs/delayed-train-features"
    training_manifest, _ = ev.read_cache_manifest(training)
    models = {name: ev.load_frozen_model(RESEARCH / "runs" / folder, training, training_manifest)
              for name, folder in MODEL_DIRS.items()}
    source_names = ["research/voice-prosody/ALIAS_FOLLOWUP_PLAN.md", "research/voice-prosody/alias_sweep.ts",
        "research/voice-prosody/phenometrix_prosody/alias_diagnostic.py", "research/voice-prosody/phenometrix_prosody/model.py",
        "research/voice-prosody/phenometrix_prosody/evaluate.py", "research/voice-prosody/phenometrix_prosody/pilot.py",
        "research/voice-prosody/phenometrix_prosody/measurement_alignment.py", "apps/capture-web/src/voice-replay.ts",
        "apps/capture-web/src/voice-pitch-features.ts", "apps/capture-web/src/voice-pitch-features-v2.ts",
        "apps/capture-web/src/voice-dsp.ts", "apps/capture-web/src/dsp-primitives.ts"]
    sources = {name: ev.sha(REPO / name) for name in source_names}
    for name in sources:
        destination = output / "source-snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes((REPO / name).read_bytes())
    scenarios = [{"id": f"sweep-{i:03}", "hz": hz, "profile": profile, "harmonics": harmonics, "phase": phase}
        for i, (hz, profile, harmonics, phase) in enumerate((hz, p, h, phase)
            for hz in FREQUENCIES for p, h in PROFILES.items() for phase in [0, .37])]
    plan = {"schema": "phenometrix-subharmonic-diagnostic-plan.v1", "declared_at": datetime.now(timezone.utc).isoformat(),
        "fitting": False, "natural_hybrid_scoring": False, "sealed_test_access": False, "source_sha256": sources,
        "scenarios": scenarios, "models": {name: {"directory": MODEL_DIRS[name], "weights_sha256": m[2]["weights_sha256"]} for name, m in models.items()},
        "measurement_delay_ms": 20, "sample_rate_hz": 48000, "duration_seconds_per_case": .4,
        "interventions": ["baseline", "global-means", "scaled-clip3", "state-reset-every-frame"] +
            [f"remove:{i}" for i in range(6)] + [f"mask:{i}" for i in range(6)] +
            [f"swap:{i}" for i in range(5)] + [f"mean:{i}" for i in range(57)]}
    ev.write_json(output / "plan.json", plan)
    raw_file = output / "acquisition-temporary.json"
    subprocess.run(["node", "--import", "tsx", "../../research/voice-prosody/alias_sweep.ts",
                    str(output / "plan.json"), str(raw_file)], cwd=REPO / "apps/capture-web", check=True)
    acquired = json.loads(raw_file.read_text())
    if acquired["featureNames"] != list(FEATURE_NAMES_V2):
        raise ValueError("Feature names changed")
    predictions, features, identities = {}, {}, []
    for record in acquired["recordings"]:
        frames = record["frames"]
        hz = np.array([[c["hz"] if c["valid"] else np.nan for c in f["candidates"]] for f in frames], np.float64)
        feature = {"features": np.array([f["values"] for f in frames], np.float32), "candidate_hz": hz,
            "candidate_mask": np.isfinite(hz), "reference_sample": np.array([f["end"] for f in frames], np.int64),
            "feature_names": np.array(FEATURE_NAMES_V2), "sample_rate_hz": np.array(48000), "pcm_sample_count": np.array(19200)}
        ev.validate_feature(feature, {"sample_count": 19200})
        path = output / "features" / (record["id"] + ".npz")
        np.savez_compressed(path, **feature); features[record["id"]] = feature
        pcm = np.frombuffer(base64.b64decode(record["pcmBase64"]), dtype="<f4")
        if hashlib.sha256(pcm.tobytes()).hexdigest() != record["pcmSha256"]:
            raise ValueError("PCM transfer identity changed")
        yin, _ = yin_predictions(pcm, feature["reference_sample"])
        dsp = []
        for f in frames:
            b = f["baseline"]; voiced = b["f0Hz"] is not None and b["f0Confidence"] >= .55 and b["estimatorAgreement"] >= .7
            dsp.append({"f0_hz": b["f0Hz"] if voiced else None, "state": "voiced" if voiced else "unvoiced", "slot": None})
        predictions[record["id"]] = {name: infer(feature, config, weights) for name, (config, weights, _) in models.items()}
        predictions[record["id"]].update(dsp=dsp, yin=[{**asdict(row), "slot": None} for row in yin])
        identities.append({"id": record["id"], "pcm_sha256": record["pcmSha256"], "feature_sha256": ev.sha(path)})
    # Prediction freeze precedes analytic-frequency scoring, including all sweep cases.
    prediction_file = output / "predictions.json.gz"
    with gzip.GzipFile(filename="", mode="wb", fileobj=prediction_file.open("wb"), mtime=0) as gz:
        gz.write(json.dumps(predictions, allow_nan=False).encode())
    ev.write_json(output / "prediction-freeze.json", {"plan_sha256": ev.sha(output / "plan.json"),
        "predictions_sha256": ev.sha(prediction_file), "records": identities})
    raw_file.unlink()
    sweep = []
    for scenario in scenarios:
        feature = features[scenario["id"]]
        eligible = ev.validate_feature(feature, {"sample_count": 19200}).common_eligible
        error = np.abs(1200 * np.log2(feature["candidate_hz"] / scenario["hz"]))
        oracle = float(np.mean(np.any(error[eligible] <= 50, axis=1)))
        sweep.append({**scenario, "candidate_oracle50": oracle,
            "candidate_count_histogram": {str(k): int(np.sum(feature["candidate_mask"][eligible].sum(1) == k)) for k in range(7)},
            "models": {name: summary(rows, scenario["hz"], eligible) for name, rows in predictions[scenario["id"]].items()}})
    original_path = RESEARCH / "runs/delayed-known-truth/features/constant-300.npz"
    original = {k: v.copy() for k, v in np.load(original_path, allow_pickle=False).items()}
    eligible = ev.validate_feature(original, {"sample_count": 76800}).common_eligible
    first = int(np.flatnonzero(eligible)[0])
    counterfactuals, parity, probes = {}, {}, {}
    for name, (config, weights, _) in models.items():
        baseline = infer(original, config, weights)
        counterfactuals[name] = {}
        for change in plan["interventions"]:
            changed = intervention(original, weights, change)
            rows = infer(changed, config, weights, reset_every_frame=change == "state-reset-every-frame")
            counterfactuals[name][change] = {**summary(rows, 300, eligible), "first_frame": rows[first]}
        torch_model = VoiceCandidateModel(config)
        torch_model.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()}, strict=True)
        torch_model.eval(); state = torch_model.initial_state(); largest = 0.
        with torch.no_grad():
            for i, x in enumerate(original["features"]):
                v, c, state = torch_model(torch.from_numpy(x[None]), torch.from_numpy(original["candidate_mask"][i:i+1]), state)
                largest = max(largest, float(np.max(np.abs(c.numpy()[0] - baseline[i]["candidate_logits"]))))
        parity[name] = {"maximum_candidate_logit_absolute_error": largest, "frames": len(baseline)}
        standardized = (original["features"][first] - weights["feature_mean"]) / weights["feature_scale"]
        activation = standardized @ weights["projection.weight"].T + weights["projection.bias"]
        probes[name] = {"first_reference_ms": (int(original["reference_sample"][first]) - 960) / 48,
            "candidate_hz": [None if not np.isfinite(v) else float(v) for v in original["candidate_hz"][first]],
            "feature_values": original["features"][first].tolist(), "standardized": standardized.tolist(),
            "projection_preactivation": activation.tolist(), "projection_absolute_over3": int((np.abs(activation) > 3).sum()),
            "prediction": baseline[first]}
    result = {"schema": "phenometrix-subharmonic-diagnostic-result.v1", "status": "complete", "plan_sha256": ev.sha(output / "plan.json"),
        "prediction_freeze_sha256": ev.sha(output / "prediction-freeze.json"), "original_feature_sha256": ev.sha(original_path),
        "sweep": sweep, "counterfactuals": counterfactuals, "original_probes": probes, "torch_numpy_parity": parity,
        "warning": "Diagnostic artificial interventions are not valid feature contracts, fitted estimators or qualification evidence."}
    for name, digest in sources.items():
        if ev.sha(REPO / name) != digest:
            raise ValueError("Diagnostic source changed during run")
    ev.write_json(output / "results.json", result)
    print(json.dumps({"status": "complete", "scenarios": len(sweep), "parity": parity, "results_sha256": ev.sha(output / "results.json")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); run(args.output.resolve())

if __name__ == "__main__":
    main()
