"""Frozen held-out analytic challenge evaluation; no fitting or threshold selection."""
from __future__ import annotations
import argparse
from dataclasses import replace, asdict
from datetime import datetime, timezone
import json
import hashlib
import os
import subprocess
from pathlib import Path
import numpy as np
from . import evaluate as ev
from .measurement_alignment import align_feature_timing
from .metrics import score, oracle_coverage
from .pilot import references_from_array, REPO, RESEARCH, yin_predictions
from . import synthetic_curriculum as sc


def challenge_gate(records, method):
    voiced = [r for r in records if r["metrics"][method]["reference_voiced_frames"] > 0]
    unvoiced = [r for r in records if r["metrics"][method]["reference_unvoiced_frames"] > 0]
    successes = [r["metrics"][method]["reference_voiced_success_rate"] for r in voiced]
    denominator = sum(r["metrics"][method]["reference_unvoiced_frames"] for r in unvoiced)
    false_positives = sum(r["metrics"][method]["voicing_false_positive_count"] for r in unvoiced)
    average = float(np.mean(successes)) if successes else None
    fp = false_positives / denominator if denominator else None
    failed = [r["id"] for r in voiced if r["metrics"][method]["reference_voiced_success_rate"] < .8]
    failed_unvoiced = [r["id"] for r in unvoiced if r["metrics"][method]["voicing_false_positive_rate"] > .05]
    passed = (len(voiced) == 28 and len(unvoiced) == 4 and average >= .9 and not failed and fp <= .05 and not failed_unvoiced)
    return {"voiced_scenario_count": len(voiced), "unvoiced_scenario_count": len(unvoiced),
            "equal_scenario_reference_voiced_success": average,
            "minimum_scenario_reference_voiced_success": min(successes) if successes else None,
            "voiced_scenarios_below_80percent": failed,
            "unvoiced_scenarios_above_5percent_false_positive": failed_unvoiced,
            "unvoiced_false_positive_count": false_positives, "unvoiced_reference_frames": denominator,
            "unvoiced_false_positive_rate": fp, "challenge_component_pass": passed, "final_keep": False}


def original_case_gates(known_truth, method="mlp"):
    stationary = {r["scenario"]["id"]: r["methods"][method]["regions"]["all"]["success_within_50c_all_frames"]
                  for r in known_truth["scenarios"] if r["scenario"]["kind"] == "constant"}
    observations = {r["scenario"]["id"]: r["methods"][method]["regions"]["after-observation-change"]["success_within_50c_all_frames"]
                    for r in known_truth["scenarios"] if r["scenario"]["kind"] == "observation-change"}
    return {"stationary_success": stationary, "postchange_observation_success": observations,
            "stationary_component_pass": len(stationary) == 3 and all(v is not None and v >= .95 for v in stationary.values()),
            "observation_component_pass": len(observations) == 3 and all(v is not None and v >= .95 for v in observations.values())}


def evaluate(curriculum, training_features, augmented, original, output):
    manifest_path = curriculum / "manifest.json"
    source = json.loads(manifest_path.read_text())
    if (source.get("status") != "complete" or source.get("split") != "challenge"
            or source.get("measurement_delay_ms") != 20 or source.get("feature_version") != "v2"
            or source.get("unique_sequence_count") != 32
            or source.get("frozen_augmented_checkpoint_manifest_sha256") != ev.sha(augmented / "manifest.json")):
        raise ValueError("Requires completed challenge bound to the frozen augmented checkpoint")
    decision_file = curriculum / "provenance" / "selection-decision.json"
    plan_file = curriculum / "provenance" / "challenge-plan.json"
    if ev.sha(plan_file) != source["plan_sha256"] or ev.sha(decision_file) != source["selection_decision_sha256"]:
        raise ValueError("Frozen challenge declaration changed")
    declaration = json.loads(decision_file.read_text())
    parameters = {r["id"]: r for r in json.loads(plan_file.read_text())["records"]}
    clarification = RESEARCH / "runs/synthetic-curriculum-plan/challenge-evaluation-decision.json"
    if declaration["plan_sha256"]["challenge"] != source["plan_sha256"]:
        raise ValueError("Challenge differs from pre-fit declaration")
    training_manifest, _ = ev.read_cache_manifest(training_features)
    if training_manifest["split"] != "train":
        raise ValueError("Checkpoint provenance must use training cache")
    folders = {"augmented": augmented, "original": original}
    models = {name: ev.load_frozen_model(folder, training_features, training_manifest) for name, folder in folders.items()}
    if any(model[0].kind != "framewise-mlp" for model in models.values()):
        raise ValueError("Challenge comparison is the frozen same-family MLP pair")
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("predictions", "provenance", "source-snapshot", "intermediate"):
        (output / directory).mkdir()
    for name, file in {"challenge-cache-manifest.json": manifest_path,
                       "selection-decision.json": decision_file, "challenge-evaluation-decision.json": clarification, "challenge-plan.json": plan_file,
                       **{name + "-training-manifest.json": folder / "manifest.json" for name, folder in folders.items()}}.items():
        (output / "provenance" / name).write_bytes(file.read_bytes())
    paths = [Path(__file__), Path(ev.__file__), Path(__file__).with_name("measurement_alignment.py"),
             Path(__file__).with_name("model.py"), Path(__file__).with_name("metrics.py"), Path(__file__).with_name("pilot.py"),
             Path(sc.__file__), RESEARCH / "replay.ts"]
    paths += [REPO / name for name in source["identity"]["source_sha256"] if name.startswith("apps/")]
    os.environ.setdefault("NUMBA_CACHE_DIR", str(RESEARCH / "runs" / ".numba-cache"))
    identities = {str(p.relative_to(REPO)): ev.sha(p) for p in paths}
    for file in paths:
        target = output / "source-snapshot" / file.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(file.read_bytes())
    manifest = {"schema": "phenometrix-held-out-synthetic-evaluation-v1", "status": "predicting",
        "started_at": datetime.now(timezone.utc).isoformat(), "split": "challenge", "measurement_delay_ms": 20,
        "challenge_manifest_sha256": ev.sha(manifest_path), "selection_decision_sha256": ev.sha(decision_file),
        "challenge_evaluation_decision_sha256": ev.sha(clarification),
        "source_sha256": identities, "reference_content_read_before_prediction_freeze": False,
        "models": {name: {"manifest_sha256": ev.sha(folders[name] / "manifest.json"),
                           "weights_sha256": model[2]["weights_sha256"]} for name, model in models.items()}, "records": []}
    ev.write_json(output / "manifest.json", manifest)
    try:
        for record in source["records"]:
            file = (curriculum / record["feature_path"]).resolve()
            if not file.is_relative_to(curriculum) or ev.sha(file) != record["feature_sha256"]:
                raise ValueError("Challenge input identity or path changed")
            with np.load(file, allow_pickle=False) as data:
                feature = {key: data[key].copy() for key in data.files}
            timing = ev.validate_feature(feature, {"sample_count": record["pcm_sample_count"]})
            predictions = {name: ev.model_predictions(feature, config, weights) for name, (config, weights, _) in models.items()}
            parameter = parameters[record["id"]]
            if sc.parameter_digest(parameter) != record["parameters_sha256"]:
                raise ValueError("Frozen source parameters changed")
            pcm = sc.generate_pcm(parameter)
            if hashlib.sha256(pcm.tobytes()).hexdigest() != record["pcm_sha256"]:
                raise ValueError("Reconstructed source differs from the frozen challenge PCM")
            raw_path = output / "intermediate" / (record["id"] + ".f32le")
            replay_path = output / "intermediate" / (record["id"] + ".json")
            pcm.tofile(raw_path)
            subprocess.run(["node", "--import", "tsx", "../../research/voice-prosody/replay.ts", "--feature-version", "v2",
                "--input-f32", str(raw_path), "--sample-rate", "48000", "--output", str(replay_path)],
                cwd=REPO / "apps/capture-web", check=True, capture_output=True, text=True)
            frames = json.loads(replay_path.read_text())["frames"]
            replay_values = np.array([f["features"]["values"] for f in frames], np.float32)
            replay_hz = np.array([[c["hz"] if c["valid"] else np.nan for c in f["features"]["candidates"]] for f in frames], np.float64)
            if not np.array_equal(replay_values, feature["features"]) or not np.array_equal(replay_hz, feature["candidate_hz"], equal_nan=True):
                raise ValueError("Replayed features differ from the frozen challenge acquisition cache")
            dsp = []
            for frame in frames:
                b = frame["features"]["baseline"]
                voiced = b["f0Hz"] is not None and b["f0Confidence"] >= .55 and b["estimatorAgreement"] >= .7
                dsp.append({"sample": frame["referenceSample"], "state": "voiced" if voiced else "unvoiced", "f0_hz": b["f0Hz"] if voiced else None})
            yin, _ = yin_predictions(pcm, feature["reference_sample"])
            predictions.update(dsp=ev.delayed_baseline(dsp, feature), yin=ev.delayed_baseline([asdict(r) for r in yin], feature),
                               continuity=ev.continuity_predictions(feature))
            raw_path.unlink(); replay_path.unlink()
            path = output / "predictions" / (record["id"] + ".json.gz")
            ev.write_predictions(path, {"id": record["id"], "time_contract": timing.metadata(),
                "availability_sample": timing.availability_sample.tolist(), "predictions": predictions})
            manifest["records"].append({**record, "prediction_sha256": ev.sha(path)})
        (output / "intermediate").rmdir()
        manifest.update(status="predictions_frozen", predictions_frozen_at=datetime.now(timezone.utc).isoformat())
        ev.write_json(output / "prediction-freeze.json", manifest)
        ev.write_json(output / "manifest.json", manifest)
        results = []
        for record in manifest["records"]:
            reference_path = (curriculum / record["reference_path"]).resolve()
            if not reference_path.is_relative_to(curriculum) or ev.sha(reference_path) != record["reference_sha256"]:
                raise ValueError("Analytic challenge reference identity changed")
            references = references_from_array(np.loadtxt(reference_path), record["pcm_sample_count"])
            references = [replace(r, eligible=r.sample >= 3072 and r.sample + 1536 <= record["pcm_sample_count"]) for r in references]
            path = output / "predictions" / (record["id"] + ".json.gz")
            if ev.sha(path) != record["prediction_sha256"]:
                raise ValueError("Challenge prediction changed after reference-free freeze")
            frozen = ev.read_predictions(path)["predictions"]
            samples = {r.sample for r in references}
            metrics = {name: score(references, [p for p in ev.prediction_objects(rows) if p.sample in samples]) for name, rows in frozen.items()}
            feature_path = curriculum / record["feature_path"]
            if ev.sha(feature_path) != record["feature_sha256"]:
                raise ValueError("Challenge candidates changed")
            with np.load(feature_path, allow_pickle=False) as feature:
                timing = align_feature_timing(feature, measurement_delay_ms=20)
                candidates = {int(sample): hz[mask].tolist() for sample, hz, mask in zip(timing.reference_sample, feature["candidate_hz"], feature["candidate_mask"])}
            results.append({"id": record["id"], "kind": record["kind"], "metrics": metrics, "candidate_oracle": oracle_coverage(references, candidates)})
        result = {"schema": "phenometrix-held-out-synthetic-scores-v1", "measurement_delay_ms": 20,
            "evidence": "disjoint frozen synthetic configurations, not new speakers or natural-data validation",
            "per_scenario": results, "gates": {name: challenge_gate(results, name) for name in (*models, "dsp", "yin", "continuity")},
            "known_failure_warning": "Original14-case stationary/observation and natural-data/runtime gates are separate; no overall keep is inferred here"}
        ev.write_json(output / "metrics.json", result)
        for name, expected in identities.items():
            if ev.sha(REPO / name) != expected:
                raise ValueError("Challenge evaluator source changed")
        manifest.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat(), metrics_sha256=ev.sha(output / "metrics.json"))
        ev.write_json(output / "manifest.json", manifest)
        return result
    except BaseException as error:
        manifest.update(status="failed", failure=repr(error))
        ev.write_json(output / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--training-features", type=Path, required=True)
    parser.add_argument("--augmented", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.curriculum.resolve(), args.training_features.resolve(), args.augmented.resolve(), args.original.resolve(), args.output.resolve())
    print(json.dumps(result["gates"], indent=2))


if __name__ == "__main__":
    main()
