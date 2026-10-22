"""Frozen no-fit MLP-voicing/raw-YIN decomposition and fresh analytic challenge."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np
from . import evaluate as ev
from . import evaluate_synthetic as known
from . import evaluate_curriculum as ec
from . import synthetic_curriculum as sc
from .measurement_alignment import align_feature_timing
from .metrics import Reference, score, oracle_coverage
from .model import FEATURE_NAMES_V2
from .pilot import REPO, RESEARCH, yin_predictions

CONTRACT = "phenometrix.mlp-voicing-yin-pitch.v1"
VARIANTS = {"original": "hybrid_original", "augmented": "hybrid_augmented"}
MODEL_DIRS = {"original": "delayed-mlp-seed20261022", "augmented": "delayed-augmented-mlp-seed20261022"}
WEIGHT_HASHES = {"original": "06110469a9a643fb16806905569ebc939db2c0bdd5b3a6a1593b31789453d10d",
                 "augmented": "9c18d9aca2829071dfa869a99db6c53e891216b6cf17d7f1cf59f0166d33aee2"}
EVALUATIONS = {"original": "delayed-development-evaluation", "augmented": "delayed-augmented-development-evaluation"}


def hybrid_predictions(feature, model_rows, yin_evidence):
    """Acquisition only. YIN rows at e; model output and new prediction at e−960."""
    timing = align_feature_timing(feature, measurement_delay_ms=20)
    if len(model_rows) != len(timing.reference_sample) or len(yin_evidence) != len(model_rows):
        raise ValueError("Hybrid evidence length differs from exact acquisition grid")
    result = []
    for model, yin, target, end in zip(model_rows, yin_evidence, timing.reference_sample, timing.availability_sample):
        if model["sample"] != int(target) or yin["sample"] != int(end):
            raise ValueError("Hybrid evidence sample does not match target/availability contract")
        probability = model["voicing_probability"]
        if not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid raw voicing probability")
        raw = yin.get("raw_f0_hz")
        valid_pitch = isinstance(raw, (int, float)) and math.isfinite(raw) and 50 <= raw <= 700
        state = "unvoiced" if probability < .5 else "voiced" if valid_pitch else "unavailable"
        result.append({"sample": int(target), "state": state, "f0_hz": float(raw) if state == "voiced" else None,
            "voicing_probability": float(probability), "correctness_probability": None, "estimator_contract": CONTRACT})
    return result


def fresh_records():
    records = []
    def add(kind, initial, final=None, regime="fundamental", after=None):
        i = len(records)
        seed = int.from_bytes(hashlib.sha256(f"20261027:hybrid-fresh-v1:{i}".encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        amplitudes = [1., 0., 0., 0., 0., 0.] if regime == "pure" else sc.harmonic_coefficients(rng, regime, 6)
        r = {"id": f"hybrid-fresh-{i:03}", "split": "fresh-challenge", "kind": kind,
             "duration_seconds": 2.2, "sample_rate_hz": 48000, "pcm_sample_count": 105600,
             "seed": seed, "initial_hz": initial, "final_hz": initial if final is None else final,
             "change_start_seconds": .41 if kind == "glide" else .93 if kind == "step" else 1.07 if kind == "observation-change" else 0.,
             "change_end_seconds": 1.79 if kind == "glide" else .93 if kind == "step" else 1.07 if kind == "observation-change" else 0.,
             "harmonic_amplitudes": amplitudes,
             "harmonic_amplitudes_after": sc.harmonic_coefficients(rng, after, 6) if after else None,
             "harmonic_phase_radians": rng.uniform(-np.pi, np.pi, 6).tolist(),
             "amplitude_gain": [.006, .012, .03, .075][i % 4],
             "aperiodic_rms": [.009, .021, .043, .065][i % 4] if initial is None else None,
             "additive_snr_db": [None, 20, None, 10][i % 4] if initial is not None else None,
             "latent_voiced": initial is not None,
             "noise_semantics": "whole-source power sets additive noise; no inference normalization"}
        records.append(r)
    for i, hz in enumerate([71.3, 93.7, 133.1, 187.9, 241.7, 313.7, 467.3, 587.9]):
        add("stationary", hz, regime=["pure", "fundamental", "weak-fundamental", "missing-fundamental"][i % 4])
    for initial, final in [(99.3, 191.7), (191.7, 99.3), (223.9, 369.1), (369.1, 223.9),
                           (331.7, 557.9), (557.9, 331.7), (143.1, 487.7), (487.7, 143.1)]:
        add("glide", initial, final, regime="weak-fundamental" if len(records) % 2 else "fundamental")
    for initial, final in [(81.7, 173.3), (173.3, 81.7), (231.7, 409.3), (409.3, 231.7),
                           (347.9, 601.7), (601.7, 347.9), (157.3, 521.9), (521.9, 157.3)]:
        add("step", initial, final, regime="bright")
    for hz in [151.7, 283.9, 433.7, 571.1]:
        add("observation-change", hz, after="missing-fundamental")
    for kind in ["white-noise", "white-noise", "difference-noise", "difference-noise"]:
        add(kind, None)
    return records


def freeze(folder):
    folder.mkdir(parents=True, exist_ok=False)
    names = ["research/voice-prosody/HYBRID_FOLLOWUP_PLAN.md", "research/voice-prosody/hybrid_features.ts"]
    names += ["research/voice-prosody/phenometrix_prosody/" + n + ".py" for n in
              ["hybrid_followup", "hybrid_natural", "model", "evaluate", "evaluate_synthetic", "evaluate_curriculum",
               "synthetic_curriculum", "measurement_alignment", "pilot", "metrics"]]
    names += ["research/voice-prosody/tests/test_hybrid_followup.py", "research/voice-prosody/tests/test_hybrid_natural.py"]
    names += ["apps/capture-web/src/" + n + ".ts" for n in
              ["voice-dsp", "dsp-primitives", "voice-replay", "voice-pitch-features", "voice-pitch-features-v2"]]
    sources = {name: ev.sha(REPO / name) for name in names}
    models = {}
    for name, directory in MODEL_DIRS.items():
        path = RESEARCH / "runs" / directory
        if ev.sha(path / "weights.pt") != WEIGHT_HASHES[name]:
            raise ValueError("Frozen checkpoint differs")
        models[name] = {"directory": directory, "weights_sha256": WEIGHT_HASHES[name], "manifest_sha256": ev.sha(path / "manifest.json")}
    plan = {"schema": "phenometrix-hybrid-followup-plan.v1", "declared_at": datetime.now(timezone.utc).isoformat(),
        "estimator_contract": CONTRACT, "source_sha256": sources, "models": models,
        "natural_evaluations": {name: {"directory": directory, "manifest_sha256": ev.sha(RESEARCH / "runs" / directory / "manifest.json"),
            "prediction_freeze_sha256": ev.sha(RESEARCH / "runs" / directory / "prediction-freeze.json")} for name, directory in EVALUATIONS.items()},
        "synthetic_inputs": {name: ev.sha(RESEARCH / "runs" / name) for name in [
            "support-diagnostic-v1/manifest.json", "delayed-known-truth/manifest.json",
            "synthetic-challenge-curriculum/manifest.json", "delayed-augmented-challenge/manifest.json",
            "synthetic-curriculum-plan/challenge-plan.json"]},
        "natural_features": {"directory": "delayed-development-features", "manifest_sha256": ev.sha(RESEARCH / "runs/delayed-development-features/manifest.json")},
        "selection": "Prefer hybrid_original if both independently pass; secondary hybrid_augmented only if original fails and secondary passes; no automatic production promotion",
        "fitting": False, "threshold_search": False, "sealed_test_access": False, "voicing_threshold": .5,
        "measurement_delay_ms": 20, "raw_yin_inclusive_hz": [50, 700], "extra_rms_or_correlation_gate": False,
        "gates": {"natural_joint_error_reduction_vs_each_dsp_yin_continuity": .01,
            "maximum_voiced_success_loss_vs_best_classical": .02, "maximum_clean_joint_error_excess_vs_best_classical": .01,
            "each_original_stationary_minimum": .95, "each_original_postchange_observation_minimum": .95,
            "fresh_equal_voiced_scenario_minimum": .9, "fresh_each_voiced_scenario_minimum": .8,
            "fresh_each_unvoiced_false_positive_maximum": .05,
            "original_transition_extra_median_ms_combined_steps_glides": 10, "all_original_transition_delays_required": 8},
        "fresh_challenge_records": fresh_records(), "fresh_challenge_scenarios": 32,
        "fresh_challenge_total_seconds": 70.4, "fresh_challenge_outcomes_before_freeze": False}
    ev.write_json(folder / "plan.json", plan)
    ev.write_json(folder / "declaration.json", {"schema": "phenometrix-hybrid-followup-freeze.v1", "plan_sha256": ev.sha(folder / "plan.json")})
    for name in sources:
        path = folder / "source-snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes((REPO / name).read_bytes())
    return {"plan_sha256": ev.sha(folder / "plan.json"), "source_count": len(sources)}


def read_frozen_plan(folder):
    folder = Path(folder)
    declaration = json.loads((folder / "declaration.json").read_text())
    if ev.sha(folder / "plan.json") != declaration["plan_sha256"]:
        raise ValueError("Follow-up plan changed after declaration")
    plan = json.loads((folder / "plan.json").read_text())
    if plan["schema"] != "phenometrix-hybrid-followup-plan.v1" or plan["fresh_challenge_records"] != fresh_records():
        raise ValueError("Unknown follow-up plan or challenge parameters")
    for name, expected in plan["source_sha256"].items():
        if ev.sha(REPO / name) != expected:
            raise ValueError("Frozen follow-up source changed: " + name)
    for name, expected in plan["synthetic_inputs"].items():
        if ev.sha(RESEARCH / "runs" / name) != expected:
            raise ValueError("Frozen regression input manifest changed")
    for name, metadata in plan["models"].items():
        path = RESEARCH / "runs" / metadata["directory"]
        if ev.sha(path / "manifest.json") != metadata["manifest_sha256"] or ev.sha(path / "weights.pt") != WEIGHT_HASHES[name]:
            raise ValueError("Follow-up checkpoint changed")
    return plan


def _feature(frames, count):
    hz = np.array([[c["hz"] if c["valid"] else np.nan for c in f["candidates"]] for f in frames], np.float64)
    return {"features": np.array([f["values"] for f in frames], np.float32), "candidate_hz": hz,
        "candidate_mask": np.isfinite(hz), "reference_sample": np.array([f["end"] for f in frames], np.int64),
        "feature_names": np.array(FEATURE_NAMES_V2), "sample_rate_hz": np.array(48000), "pcm_sample_count": np.array(count)}


def _dsp(frames, feature):
    rows = []
    for frame in frames:
        b = frame["baseline"]; voiced = b["f0Hz"] is not None and b["f0Confidence"] >= .55 and b["estimatorAgreement"] >= .7
        rows.append({"sample": frame["end"], "state": "voiced" if voiced else "unvoiced", "f0_hz": b["f0Hz"] if voiced else None})
    return ev.delayed_baseline(rows, feature)


def synthetic(plan_folder, output):
    plan = read_frozen_plan(plan_folder)
    train_folder = RESEARCH / "runs/delayed-train-features"
    training, _ = ev.read_cache_manifest(train_folder)
    models = {name: ev.load_frozen_model(RESEARCH / "runs" / row["directory"], train_folder, training) for name, row in plan["models"].items()}
    output.mkdir(parents=True, exist_ok=False)
    for name in ["features", "predictions", "intermediate", "provenance"]:
        (output / name).mkdir()
    (output / "provenance/plan.json").write_bytes((plan_folder / "plan.json").read_bytes())
    manifest = {"schema": "phenometrix-hybrid-synthetic-evaluation.v1", "status": "predicting", "plan_sha256": ev.sha(plan_folder / "plan.json"),
        "models": plan["models"], "estimator_contract": CONTRACT, "started_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": plan["source_sha256"], "analytic_references_read_before_prediction_freeze": False, "records": []}
    ev.write_json(output / "manifest.json", manifest)
    jobs = []
    # Old inputs and old raw predictions are verified; no reference files opened.
    support = RESEARCH / "runs/support-diagnostic-v1"
    support_manifest = json.loads((support / "manifest.json").read_text())
    old = RESEARCH / "runs/delayed-known-truth"
    old_manifest = json.loads((old / "manifest.json").read_text())
    if old_manifest["original_support_manifest_sha256"] != ev.sha(support / "manifest.json"):
        raise ValueError("Original regression source identity changed")
    definitions = {s["id"]: s for s in support_manifest["scenarios"]}
    definitions.update({s["id"]: s for s in known.OBSERVATION_SCENARIOS})
    for record in old_manifest["recordings"]:
        name = record["id"]; scenario = definitions[name]
        pcm = known.observation_pcm(scenario) if scenario["kind"] == "observation-change" else np.fromfile(support / "pcm" / (name + ".f32le"), dtype="<f4")
        feature_path = old / "features" / (name + ".npz")
        prediction_path = old / "predictions" / (name + ".json.gz")
        if hashlib.sha256(pcm.tobytes()).hexdigest() != record["pcm_sha256"] or ev.sha(feature_path) != record["feature_sha256"] or ev.sha(prediction_path) != record["prediction_sha256"]:
            raise ValueError("Original analytic input identity changed")
        feature = {k: v.copy() for k, v in np.load(feature_path, allow_pickle=False).items()}
        baseline = ev.read_predictions(prediction_path)["predictions"]
        jobs.append({"id": name, "cohort": "original14", "scenario": scenario, "pcm": pcm, "feature": feature,
                     "dsp": baseline["dsp"], "gru": baseline["gru"], "source_feature_sha256": ev.sha(feature_path)})
    curriculum = RESEARCH / "runs/synthetic-challenge-curriculum"
    previous = RESEARCH / "runs/delayed-augmented-challenge"
    curriculum_manifest = json.loads((curriculum / "manifest.json").read_text())
    previous_manifest = json.loads((previous / "manifest.json").read_text())
    if previous_manifest["challenge_manifest_sha256"] != ev.sha(curriculum / "manifest.json"):
        raise ValueError("Old challenge identity changed")
    prior = {r["id"]: r for r in previous_manifest["records"]}
    parameters_path = RESEARCH / "runs/synthetic-curriculum-plan/challenge-plan.json"
    if ev.sha(parameters_path) != curriculum_manifest["plan_sha256"]:
        raise ValueError("Old challenge parameters changed")
    parameters = {r["id"]: r for r in json.loads(parameters_path.read_text())["records"]}
    for record in curriculum_manifest["records"]:
        name = record["id"]; parameter = parameters[name]; pcm = sc.generate_pcm(parameter)
        feature_path = curriculum / record["feature_path"]
        prediction_path = previous / "predictions" / (name + ".json.gz")
        if sc.parameter_digest(parameter) != record["parameters_sha256"] or hashlib.sha256(pcm.tobytes()).hexdigest() != record["pcm_sha256"] or ev.sha(feature_path) != record["feature_sha256"] or ev.sha(prediction_path) != prior[name]["prediction_sha256"]:
            raise ValueError("Old challenge acquisition content changed")
        feature = {k: v.copy() for k, v in np.load(feature_path, allow_pickle=False).items()}
        baseline = ev.read_predictions(prediction_path)["predictions"]
        jobs.append({"id": name, "cohort": "previous32-regression", "parameters": parameter, "pcm": pcm,
                     "feature": feature, "dsp": baseline["dsp"], "source_feature_sha256": ev.sha(feature_path)})
    fresh_input = []
    for record in plan["fresh_challenge_records"]:
        pcm = sc.generate_pcm(record)
        file = output / "intermediate" / (record["id"] + ".f32le"); pcm.tofile(file)
        fresh_input.append({"id": record["id"], "path": str(file)})
        jobs.append({"id": record["id"], "cohort": "fresh32", "parameters": record, "pcm": pcm})
    ev.write_json(output / "intermediate/input-list.json", fresh_input)
    subprocess.run(["node", "--import", "tsx", "../../research/voice-prosody/hybrid_features.ts",
        str(output / "intermediate/input-list.json"), str(output / "intermediate/features.json")], cwd=REPO / "apps/capture-web", check=True)
    exported = {r["id"]: r["frames"] for r in json.loads((output / "intermediate/features.json").read_text())}
    for job in jobs:
        name, pcm = job["id"], job["pcm"]
        if job["cohort"] == "fresh32":
            job["feature"] = _feature(exported[name], len(pcm)); job["dsp"] = _dsp(exported[name], job["feature"])
        feature = job["feature"]; timing = ev.validate_feature(feature, {"sample_count": len(pcm)})
        feature_path = output / "features" / (name + ".npz"); np.savez_compressed(feature_path, **feature)
        yin, evidence = yin_predictions(pcm, feature["reference_sample"])
        predictions = {"dsp": job["dsp"], "yin": ev.delayed_baseline([asdict(r) for r in yin], feature), "continuity": ev.continuity_predictions(feature)}
        for variant, (config, weights, _) in models.items():
            rows = ev.model_predictions(feature, config, weights)
            predictions["mlp_" + variant] = rows
            predictions[VARIANTS[variant]] = hybrid_predictions(feature, rows, evidence)
        if "gru" in job:
            predictions["gru"] = job["gru"]
        file = output / "predictions" / (name + ".json.gz")
        ev.write_predictions(file, {"id": name, "availability_sample": timing.availability_sample.tolist(),
                                   "time_contract": timing.metadata(), "predictions": predictions, "yin_evidence": evidence})
        manifest["records"].append({"id": name, "cohort": job["cohort"], "pcm_sample_count": len(pcm),
            "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(), "feature_sha256": ev.sha(feature_path),
            "source_feature_sha256": job.get("source_feature_sha256"), "prediction_sha256": ev.sha(file),
            "parameters_sha256": sc.parameter_digest(job.get("parameters", job.get("scenario")))})
    manifest.update(status="predictions_frozen", predictions_frozen_at=datetime.now(timezone.utc).isoformat())
    ev.write_json(output / "prediction-freeze.json", manifest); ev.write_json(output / "manifest.json", manifest)
    for file in (output / "intermediate").iterdir():
        file.unlink()
    (output / "intermediate").rmdir()
    results, original_summaries = [], {v: [] for v in VARIANTS.values()}
    for job, record in zip(jobs, manifest["records"]):
        file = output / "predictions" / (job["id"] + ".json.gz")
        if ev.sha(file) != record["prediction_sha256"]:
            raise ValueError("Synthetic output changed after freeze")
        raw = ev.read_predictions(file); predictions = raw["predictions"]
        count = len(job["pcm"]); feature = job["feature"]
        def frequency(sample):
            return known.analytic_frequency(job["scenario"], sample / 48000) if job["cohort"] == "original14" else sc.frequency_at(job["parameters"], sample / 48000)
        references = [Reference(int(t), frequency(t) or None, frequency(t) > 0, True, t >= 3072 and t + 1536 <= count)
                      for t in range(768, count + 1, 480)]
        methods = {name: score(references, ev.prediction_objects(rows)) for name, rows in predictions.items()}
        timing = align_feature_timing(feature, measurement_delay_ms=20)
        candidates = {int(t): h[m].tolist() for t, h, m in zip(timing.reference_sample, feature["candidate_hz"], feature["candidate_mask"])}
        results.append({"id": job["id"], "cohort": job["cohort"], "metrics": methods, "oracle": oracle_coverage(references, candidates)})
        if job["cohort"] == "original14":
            for variant in VARIANTS.values():
                # Reuse the original signed-lag definition with mlp as its fixed slot.
                payload = {**raw, "predictions": {**predictions, "mlp": predictions[variant]}}
                metrics, _ = known.score_predictions(job["scenario"], payload)
                original_summaries[variant].append({"scenario": job["scenario"], "methods": metrics})
    gates = {}
    for variant, scenarios in original_summaries.items():
        gates[variant] = {"original_cases": ec.original_case_gates({"scenarios": scenarios}),
            "original_transition": known.transition_gate(scenarios)["model_components"]["mlp"],
            "fresh_challenge": ec.challenge_gate([r for r in results if r["cohort"] == "fresh32"], variant),
            "previous_challenge_regression": ec.challenge_gate([r for r in results if r["cohort"] == "previous32-regression"], variant),
            "final_keep": False}
    result = {"schema": "phenometrix-hybrid-synthetic-metrics.v1", "estimator_contract": CONTRACT,
        "per_case": results, "gates": gates, "original_scenarios_for_transition": original_summaries,
        "selection": plan["selection"], "final_keep": False, "natural_gate_and_runtime_separate": True}
    read_frozen_plan(plan_folder)
    ev.write_json(output / "metrics.json", result)
    manifest.update(status="complete", metrics_sha256=ev.sha(output / "metrics.json"), completed_at=datetime.now(timezone.utc).isoformat())
    ev.write_json(output / "manifest.json", manifest)
    return {"gates": gates, "metrics_sha256": manifest["metrics_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze"); f.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("synthetic"); s.add_argument("--plan", type=Path, required=True); s.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.output.resolve()) if args.command == "freeze" else synthetic(args.plan.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
