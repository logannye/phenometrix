"""One frozen analytic training curriculum and a disjoint held-out challenge.

No fitting or candidate/architecture changes. Challenge generation is blocked
until a completed augmented checkpoint binds the frozen training plan. Parameter
plans are immutable; labels never enter TypeScript acquisition feature extraction.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import numpy as np
from .pilot import REPO, source_identity
from .evaluate import validate_feature

RATE = 48000
DURATION = 2.5
MAX_BYTES = 40 * 1024 * 1024


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameter_digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def write_json(path, value, *, fresh=False):
    with Path(path).open("x" if fresh else "w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def harmonic_coefficients(rng, regime, count):
    amplitudes = rng.uniform(.025, .2, count)
    if regime == "fundamental":
        amplitudes[:3] = [rng.uniform(.7, 1.2), rng.uniform(.2, .65), rng.uniform(.12, .4)]
    elif regime == "weak-fundamental":
        amplitudes[:3] = [rng.uniform(.05, .18), rng.uniform(.65, 1.), rng.uniform(.4, .8)]
    elif regime == "missing-fundamental":
        amplitudes[:3] = [0., rng.uniform(.55, 1.), rng.uniform(.35, .8)]
    elif regime == "bright":
        amplitudes[:3] = [rng.uniform(.3, .6), rng.uniform(.5, .9), rng.uniform(.4, .8)]
    else:
        raise ValueError("Unknown fixed harmonic regime")
    return amplitudes.tolist()


def scenario(split, index, kind, initial, final=None, regime="fundamental", after=None):
    identity = f"synthetic-{split}-{index:03d}"
    seed = int.from_bytes(hashlib.sha256(f"20261025:{split}:{identity}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    harmonics = 5 if split == "train" else 6
    count = round(DURATION * RATE)
    phase = rng.uniform(-np.pi, np.pi, harmonics).tolist()
    coefficients = harmonic_coefficients(rng, regime, harmonics)
    after_coefficients = harmonic_coefficients(rng, after, harmonics) if after else None
    change_start = {"stationary": 0., "glide": .5 if split == "train" else .35,
                    "step": 1. if split == "train" else .87,
                    "observation-change": 1.25 if split == "train" else 1.17,
                    "white-noise": 0., "difference-noise": 0.}[kind]
    change_end = (2. if split == "train" else 2.05) if kind == "glide" else change_start
    unvoiced = kind in {"white-noise", "difference-noise"}
    return {"id": identity, "split": split, "kind": kind, "duration_seconds": DURATION,
            "sample_rate_hz": RATE, "pcm_sample_count": count, "seed": seed,
            "initial_hz": initial, "final_hz": initial if final is None else final,
            "change_start_seconds": change_start, "change_end_seconds": change_end,
            "harmonic_amplitudes": coefficients, "harmonic_amplitudes_after": after_coefficients,
            "harmonic_phase_radians": phase, "amplitude_gain": float(rng.uniform(.018, .085)),
            "aperiodic_rms": ([.008, .015, .03, .06][index % 4] if split == "train" else [.011, .023][index % 2]) if unvoiced else None,
            "additive_snr_db": None if unvoiced else (20 if index % 4 == 0 else None) if split == "train" else (14 if index % 2 else None),
            "latent_voiced": not unvoiced,
            "noise_semantics": "offline whole-source clean power sets additive noise; no inference normalization; target is analytic latent carrier"}


def make_plans():
    frequencies = np.geomspace(60., 600., 12).tolist()
    train = []
    def add(kind, initial, final=None, regime="fundamental", after=None):
        train.append(scenario("train", len(train), kind, initial, final, regime, after))
    for hz in frequencies:
        for regime in ("fundamental", "weak-fundamental", "missing-fundamental", "bright"):
            add("stationary", hz, regime=regime)
    for kind, distance in (("glide", 4), ("step", 5)):
        for index, initial in enumerate(frequencies):
            final = frequencies[(index + distance) % len(frequencies)]
            add(kind, initial, final)
            add(kind, final, initial, regime="bright")
    for hz in frequencies:
        add("observation-change", hz, regime="fundamental", after="missing-fundamental")
        add("observation-change", hz, regime="weak-fundamental", after="bright")
    for kind in ("white-noise", "difference-noise"):
        for _ in range(4):
            add(kind, None)
    challenge = []
    def challenge_add(kind, initial, final=None, regime="bright", after=None):
        challenge.append(scenario("challenge", len(challenge), kind, initial, final, regime, after))
    for i, hz in enumerate((66.7, 85.9, 121.3, 177.7, 247.9, 353.3, 479.7, 623.1)):
        challenge_add("stationary", hz, regime="fundamental" if i % 2 == 0 else "missing-fundamental")
    for initial, final in ((83, 167), (167, 83), (217, 389), (389, 217), (321, 573), (573, 321), (149, 443), (443, 149)):
        challenge_add("glide", initial, final)
    for initial, final in ((89, 179), (179, 89), (227, 431), (431, 227), (337, 619), (619, 337), (137, 509), (509, 137)):
        challenge_add("step", initial, final)
    for hz in (127, 263, 421, 593):
        challenge_add("observation-change", hz, regime="fundamental", after="missing-fundamental")
    for kind in ("white-noise", "difference-noise"):
        for _ in range(2):
            challenge_add(kind, None)
    shared = {"schema": "phenometrix-analytic-curriculum-plan-v1", "measurement_delay_ms": 20,
              "feature_version": "v2", "duration_seconds_per_sequence": DURATION,
              "analytic_reference_grid": "16+10*k ms; four columns F0,voiced,0,0; zero placeholders are not measured evidence",
              "phase": "exact piecewise integral of latent instantaneous F0; independent fixed phase per harmonic",
              "disjointness": "unique identity, seed, complete generator parameters and trajectories; continuous train glides may cross challenge frequencies",
              "harmonic_identifiability": "fundamental-suppressed regimes retain nonzero harmonics2and3",
              "maximum_generated_output_bytes": MAX_BYTES}
    return {split: {**shared, "split": split, "sequence_count": len(records),
                    "duration_seconds": len(records) * DURATION, "records": records}
            for split, records in (("train", train), ("challenge", challenge))}


def freeze(folder):
    plans = make_plans()
    folder.mkdir(parents=True, exist_ok=False)
    timestamp = datetime.now(timezone.utc).isoformat()
    hashes = {}
    for split, plan in plans.items():
        plan["declared_at"] = timestamp
        plan["generator_sha256"] = digest(__file__)
        write_json(folder / f"{split}-plan.json", plan, fresh=True)
        hashes[split] = digest(folder / f"{split}-plan.json")
    decision = {"schema": "phenometrix-single-augmented-mlp-decision-v1", "declared_at": timestamp,
        "status": "frozen_before_generation_and_augmented_fit", "plan_sha256": hashes,
        "retained_prior_failures": ["original endpoint candidate gate", "first GRU marginal-gain gate", "first learned constant300Hz selection failures"],
        "new_configuration_count": 1, "model": "same framewise MLP32/projection32/v2features; no candidate or architecture change; no GRU retry",
        "training": {"seed": 20261022, "epochs": 12, "optimizer": "Adam", "learning_rate": .001,
            "batch_size": 64, "tbptt_frames": 128, "synthetic_sequence_repetitions": 4,
            "repetition_warning": "Repeated deterministic examples change weight, not independent sample count",
            "scaling": "train only, natural plus same4fold synthetic weighting"},
        "challenge_access": "No challenge PCM, features or analytic references generated until completed augmented checkpoint binds train plan SHA",
        "gates": {"primary_natural_development_and_runtime": "unchanged",
            "each_original_stationary_success_at_50c_minimum": .95,
            "each_original_postchange_observation_success_at_50c_minimum": .95,
            "new_challenge_equal_scenario_voiced_success_at_50c_minimum": .90,
            "each_new_challenge_voiced_scenario_success_at_50c_minimum": .80,
            "new_challenge_unvoiced_false_positive_rate_maximum": .05,
            "transition_lag": "combined,step-only,glide-only extra median<=10ms versus delayedDSP; all8 defined"},
        "stop_rule": "One final augmented configuration. Report all original and new challenge cases. Failed gates remain no-go; no selection of challenge subgroups or architecture search."}
    write_json(folder / "selection-decision.json", decision, fresh=True)
    return {"folder": str(folder), "plan_sha256": hashes, "selection_decision_sha256": digest(folder / "selection-decision.json")}


def frequency_at(record, time):
    if not record["latent_voiced"]:
        return 0.
    initial, final = record["initial_hz"], record["final_hz"]
    if record["kind"] == "step":
        return initial if time < record["change_start_seconds"] else final
    if record["kind"] == "glide":
        fraction = min(1., max(0., (time - record["change_start_seconds"]) /
                               (record["change_end_seconds"] - record["change_start_seconds"])))
        return initial + (final - initial) * fraction
    return initial


def cycles_at(record, time):
    if not record["latent_voiced"]:
        raise ValueError("Aperiodic source has no latent cycle phase")
    initial, final = record["initial_hz"], record["final_hz"]
    if record["kind"] == "step":
        return initial * time + (final - initial) * np.maximum(0., time - record["change_start_seconds"])
    if record["kind"] == "glide":
        duration = record["change_end_seconds"] - record["change_start_seconds"]
        elapsed = np.clip(time - record["change_start_seconds"], 0., duration)
        slope = (final - initial) / duration
        return initial * time + .5 * slope * elapsed ** 2 + slope * duration * np.maximum(0., time - record["change_end_seconds"])
    return initial * time


def generate_pcm(record):
    rng = np.random.default_rng(record["seed"])
    count = record["pcm_sample_count"]
    if not record["latent_voiced"]:
        values = rng.standard_normal(count)
        if record["kind"] == "difference-noise":
            values = np.diff(values, prepend=0.)
        values *= record["aperiodic_rms"] / np.sqrt(np.mean(values ** 2))
    else:
        time = np.arange(count, dtype=np.float64) / RATE
        phase = 2 * np.pi * cycles_at(record, time)
        values = np.zeros(count, np.float64)
        for index, (amplitude, shift) in enumerate(zip(record["harmonic_amplitudes"], record["harmonic_phase_radians"])):
            gain = amplitude
            if record["harmonic_amplitudes_after"] is not None:
                gain = np.where(time < record["change_start_seconds"], amplitude, record["harmonic_amplitudes_after"][index])
            values += gain * np.sin((index + 1) * phase + shift)
        values *= record["amplitude_gain"]
        if record["additive_snr_db"] is not None:
            noise = rng.standard_normal(count)
            noise *= np.sqrt(np.mean(values ** 2) / (10 ** (record["additive_snr_db"] / 10) * np.mean(noise ** 2)))
            values += noise
    pcm = values.astype("<f4")
    if not np.isfinite(pcm).all():
        raise ValueError("Nonfinite analytic source")
    return pcm


def analytic_reference(record):
    samples = np.arange(768, record["pcm_sample_count"] + 1, 480)
    values = np.zeros((len(samples), 4), np.float64)
    if record["latent_voiced"]:
        values[:, 0] = [frequency_at(record, sample / RATE) for sample in samples]
        values[:, 1] = 1
    return values


def verify_plan(folder, split, checkpoint_manifest=None):
    declaration = json.loads((folder / "selection-decision.json").read_text())
    file = folder / f"{split}-plan.json"
    if digest(file) != declaration["plan_sha256"][split]:
        raise ValueError("Frozen plan identity changed")
    plan = json.loads(file.read_text())
    if plan["generator_sha256"] != digest(__file__) or plan["split"] != split:
        raise ValueError("Frozen generator or split changed")
    if split == "challenge":
        if checkpoint_manifest is None:
            raise ValueError("Challenge remains sealed until a completed augmented checkpoint is provided")
        checkpoint = json.loads(checkpoint_manifest.read_text())
        supplement = checkpoint.get("synthetic_curriculum")
        if (checkpoint.get("status") != "complete" or checkpoint.get("kind") != "framewise-mlp"
                or not supplement or supplement.get("challenge_content_or_outcomes_used") is not False):
            raise ValueError("Challenge requires completed augmented MLP provenance bound to the frozen train plan")
        supplement_file = checkpoint_manifest.parent / "synthetic-curriculum-manifest.json"
        if digest(supplement_file) != supplement.get("manifest_sha256"):
            raise ValueError("Augmented checkpoint supplement identity changed")
        curriculum = json.loads(supplement_file.read_text())
        if (curriculum.get("plan_sha256") != declaration["plan_sha256"]["train"]
                or curriculum.get("split") != "train" or curriculum.get("status") != "complete"
                or digest(checkpoint_manifest.parent / "weights.pt") != checkpoint.get("weights_sha256")):
            raise ValueError("Challenge requires completed augmented MLP provenance bound to the frozen train plan")
    return plan


def output_bytes(output):
    return sum(path.stat().st_size for path in output.rglob("*") if path.is_file())


def generate(folder, split, output, checkpoint_manifest=None):
    plan = verify_plan(folder, split, checkpoint_manifest)
    if shutil.disk_usage(output.parent).free < 512 * 1024 * 1024:
        raise ValueError("Insufficient free storage for bounded generation")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("features", "references", "intermediate", "source-snapshot", "provenance"):
        (output / name).mkdir()
    identity = source_identity()
    for source in (Path(__file__), Path(__file__).with_name("evaluate.py"), Path(__file__).with_name("measurement_alignment.py")):
        identity["source_sha256"][str(source.relative_to(REPO))] = digest(source)
    for name, expected in identity["source_sha256"].items():
        path = output / "source-snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((REPO / name).read_bytes())
        if digest(path) != expected:
            raise ValueError("Source snapshot changed")
    (output / "provenance" / f"{split}-plan.json").write_bytes((folder / f"{split}-plan.json").read_bytes())
    (output / "provenance" / "selection-decision.json").write_bytes((folder / "selection-decision.json").read_bytes())
    manifest = {"schema": "phenometrix-analytic-curriculum-cache-v1", "status": "generating-input-features", "split": split,
        "feature_version": "v2", "measurement_delay_ms": 20, "started_at": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": digest(folder / f"{split}-plan.json"), "selection_decision_sha256": digest(folder / "selection-decision.json"),
        "generator_sha256": digest(__file__), "identity": identity, "analytic_reference_content_created_before_feature_freeze": False,
        "reference_format": "analytic truth on16+10k-ms grid: F0Hz,voicedBoolean,0,0; final columns placeholders; not PTDB measurements",
        "duration_seconds": plan["duration_seconds"], "unique_sequence_count": plan["sequence_count"],
        "maximum_generated_output_bytes": MAX_BYTES, "records": []}
    if checkpoint_manifest is not None:
        manifest["frozen_augmented_checkpoint_manifest_sha256"] = digest(checkpoint_manifest)
    write_json(output / "manifest.json", manifest)
    try:
        # First extract every input-only feature file; no reference arrays yet.
        for record in plan["records"]:
            name = record["id"]
            pcm = generate_pcm(record)
            raw, full = output / "intermediate" / (name + ".f32le"), output / "intermediate" / (name + ".json")
            pcm.tofile(raw)
            subprocess.run(["node", "--import", "tsx", "../../research/voice-prosody/replay.ts", "--feature-version", "v2",
                            "--input-f32", str(raw), "--sample-rate", str(RATE), "--output", str(full)],
                           cwd=REPO / "apps/capture-web", check=True, capture_output=True, text=True)
            replay = json.loads(full.read_text())
            frames = replay["frames"]
            hz = np.array([[c["hz"] if c["valid"] else np.nan for c in f["features"]["candidates"]] for f in frames], np.float64)
            feature = {"features": np.array([f["features"]["values"] for f in frames], np.float32), "candidate_hz": hz,
                "candidate_mask": np.isfinite(hz), "reference_sample": np.array([f["referenceSample"] for f in frames], np.int64),
                "feature_names": np.array(replay["featureNames"]), "sample_rate_hz": np.array(RATE), "pcm_sample_count": np.array(len(pcm))}
            validate_feature(feature, {"sample_count": len(pcm)})
            path = output / "features" / (name + ".npz")
            np.savez_compressed(path, **feature)
            manifest["records"].append({"id": name, "kind": record["kind"], "feature_path": str(path.relative_to(output)),
                "feature_sha256": digest(path), "reference_path": f"references/{name}.txt", "reference_sha256": None,
                "pcm_sample_count": len(pcm), "pcm_sha256": digest(raw), "feature_frames": len(frames),
                "parameters_sha256": parameter_digest(record), "generator_sha256": plan["generator_sha256"]})
            raw.unlink(); full.unlink()
            if output_bytes(output) > MAX_BYTES:
                raise ValueError("Generated curriculum exceeds the frozen40MiB artifact budget")
            write_json(output / "manifest.json", manifest)
            if len(manifest["records"]) % 16 == 0:
                print(json.dumps({"split": split, "feature_sequences": len(manifest["records"]), "of": len(plan["records"])}), flush=True)
        (output / "intermediate").rmdir()
        manifest.update(status="input-features-frozen", input_features_frozen_at=datetime.now(timezone.utc).isoformat())
        write_json(output / "input-feature-freeze.json", manifest)
        # Only now create scorer/training-supervision reference arrays.
        for record, saved in zip(plan["records"], manifest["records"]):
            path = output / saved["reference_path"]
            np.savetxt(path, analytic_reference(record), fmt="%.17g")
            saved["reference_sha256"] = digest(path)
        for name, expected in identity["source_sha256"].items():
            if digest(REPO / name) != expected:
                raise ValueError("Feature generator source changed during generation")
        manifest.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat(), output_bytes_before_final_manifest=output_bytes(output))
        if manifest["output_bytes_before_final_manifest"] > MAX_BYTES:
            raise ValueError("Generated curriculum exceeds the frozen40MiB artifact budget")
        write_json(output / "manifest.json", manifest)
        return manifest
    except BaseException as error:
        manifest.update(status="failed", failure=repr(error))
        write_json(output / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freezing = commands.add_parser("freeze")
    freezing.add_argument("--output", type=Path, required=True)
    generating = commands.add_parser("generate")
    generating.add_argument("--plan", type=Path, required=True)
    generating.add_argument("--split", choices=["train", "challenge"], required=True)
    generating.add_argument("--output", type=Path, required=True)
    generating.add_argument("--checkpoint-manifest", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args.output.resolve())
    else:
        result = generate(args.plan.resolve(), args.split, args.output.resolve(),
                          args.checkpoint_manifest.resolve() if args.checkpoint_manifest else None)
        result = {key: result[key] for key in ("status", "split", "unique_sequence_count", "duration_seconds", "plan_sha256", "output_bytes_before_final_manifest")}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
