"""Fixed known-truth transition/observation checks for the 20 ms measurement task.

No fitting, architecture selection, threshold search or candidate revision. The
11 original analytic PCM files are reused exactly. Three additional fixed
observation changes retain harmonics 2 and 3 while suppressing the fundamental.
Every estimator output is frozen before analytic target arrays are opened.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import subprocess
import numpy as np
from . import evaluate as ev
from .metrics import cents
from .model import FEATURE_NAMES_V2
from .pilot import REPO, RESEARCH, yin_predictions

METHODS = ("dsp", "yin", "continuity", "mlp", "gru")
OBSERVATION_SCENARIOS = tuple({"id": f"observation-change-{hz}", "kind": "observation-change",
    "durationSeconds": 1.6, "initialHz": hz, "finalHz": hz, "changeStartSeconds": .6,
    "changeEndSeconds": .6, "slopeHzPerSecond": 0} for hz in (100, 200, 300))


def observation_pcm(scenario):
    if scenario not in OBSERVATION_SCENARIOS:
        raise ValueError("Only the three fixed observation changes are declared")
    time = np.arange(76800, dtype=np.float64) / 48000
    phase = 2 * np.pi * scenario["initialHz"] * time
    before = np.sin(phase) + .5 * np.sin(2 * phase) + .25 * np.sin(3 * phase)
    after = np.sin(2 * phase) + .5 * np.sin(3 * phase)
    return (.075 * np.where(time < .6, before, after)).astype(np.float32)


def analytic_frequency(scenario, time):
    if scenario["kind"] in {"constant", "observation-change"}:
        return float(scenario["initialHz"])
    if scenario["kind"] == "step":
        return float(scenario["initialHz"] if time < scenario["changeStartSeconds"] else scenario["finalHz"])
    elapsed = min(scenario["changeEndSeconds"] - scenario["changeStartSeconds"], max(0., time - scenario["changeStartSeconds"]))
    return scenario["initialHz"] + scenario["slopeHzPerSecond"] * elapsed


def region_at(scenario, end_sample):
    start, end = (end_sample - 1920) / 48000, end_sample / 48000
    if scenario["kind"] == "constant":
        return "plateau"
    if scenario["kind"] == "observation-change":
        return "before-observation-change" if end < .6 else "after-observation-change" if start >= .6 else "observation-boundary"
    if scenario["kind"] == "glide" and start >= scenario["changeStartSeconds"] and end < scenario["changeEndSeconds"]:
        return "glide-interior"
    return "plateau" if end < scenario["changeStartSeconds"] or start >= scenario["changeEndSeconds"] else "transition-boundary"


def summarize(rows):
    errors = np.array([r["signed_cents"] for r in rows if r["signed_cents"] is not None])
    delays = [r["equivalent_slope_delay_ms"] for r in rows if r["equivalent_slope_delay_ms"] is not None]
    correct = sum(abs(v) <= 50 for v in errors)
    return {"frames": len(rows), "voiced_outputs": len(errors), "nonvoiced_or_unavailable": len(rows) - len(errors),
            "pitch_coverage": len(errors) / len(rows) if rows else None,
            "success_within_50c_all_frames": correct / len(rows) if rows else None,
            "median_absolute_cents": float(np.median(np.abs(errors))) if len(errors) else None,
            "p95_absolute_cents": float(np.quantile(np.abs(errors), .95, method="higher")) if len(errors) else None,
            "mean_signed_cents": float(np.mean(errors)) if len(errors) else None,
            "median_signed_equivalent_slope_delay_ms": float(np.median(delays)) if delays else None}


def step_response(rows, scenario):
    """Final-value response can anticipate the reported target because delay=20ms.

    Start when the physical change could first be available, not at target>=step.
    A first-of-three final-value response at target before the change has negative
    target lag and remains a pitch error in the ordinary target-time frame score.
    """
    change_ms = scenario["changeStartSeconds"] * 1000
    eligible = [r for r in rows if r["availability_ms"] >= change_ms]
    final_correct = [r["estimate_hz"] is not None and abs(cents(r["estimate_hz"], scenario["finalHz"])) <= 50 for r in eligible]
    first = next((i for i in range(len(eligible) - 2) if all(final_correct[i:i + 3])
                  and eligible[i + 2]["reference_sample"] - eligible[i]["reference_sample"] == 960), None)
    row = None if first is None else eligible[first]
    return {"target_time_first_of_three_settling_lag_ms": None if row is None else row["target_ms"] - change_ms,
            "earliest_availability_first_of_three_delay_ms": None if row is None else row["availability_ms"] - change_ms,
            "settled": row is not None, "criterion": "first of 3 consecutive final-frequency outputs within50c; acquisition end>=physical step; no negative-lag clamp"}


def score_predictions(scenario, raw):
    traces, result = [], {}
    for method in METHODS:
        rows = []
        for prediction, end in zip(raw["predictions"][method], raw["availability_sample"]):
            sample = prediction["sample"]
            if sample != end - 960:
                raise ValueError("Synthetic prediction target differs from declared delay")
            if sample < 3072 or sample + 1536 > 76800:
                continue
            truth = analytic_frequency(scenario, sample / 48000)
            estimate = prediction["f0_hz"] if prediction["state"] == "voiced" else None
            region = region_at(scenario, end)
            slope = scenario["slopeHzPerSecond"] if region == "glide-interior" else 0
            row = {"scenario": scenario["id"], "method": method, "reference_sample": sample,
                   "target_ms": sample / 48, "availability_ms": end / 48, "region": region,
                   "target_hz": truth, "estimate_hz": estimate, "state": prediction["state"],
                   "signed_cents": None if estimate is None else cents(estimate, truth),
                   "equivalent_slope_delay_ms": None if estimate is None or not slope else (truth - estimate) / slope * 1000}
            rows.append(row)
        regions = ["all", *sorted({r["region"] for r in rows})]
        result[method] = {"regions": {region: summarize(rows if region == "all" else [r for r in rows if r["region"] == region]) for region in regions},
                          "step_response": step_response(rows, scenario) if scenario["kind"] == "step" else None}
        traces.extend(rows)
    return result, traces


def transition_gate(scenarios):
    by_method = {m: {} for m in METHODS}
    for row in scenarios:
        scenario = row["scenario"]
        for method in METHODS:
            metric = row["methods"][method]
            if scenario["kind"] == "step":
                value = metric["step_response"]["target_time_first_of_three_settling_lag_ms"]
            elif scenario["kind"] == "glide":
                value = metric["regions"]["glide-interior"]["median_signed_equivalent_slope_delay_ms"]
            else:
                continue
            by_method[method][scenario["id"]] = value
    summaries = {}
    for method, values in by_method.items():
        defined = [v for v in values.values() if v is not None]
        steps = [v for name, v in values.items() if name.startswith("step-") and v is not None]
        glides = [v for name, v in values.items() if name.startswith("glide-") and v is not None]
        summaries[method] = {"per_scenario_signed_target_delay_ms": values, "defined_scenarios": len(defined),
            "median_signed_target_delay_ms": float(np.median(defined)) if len(defined) == 8 else None,
            "median_step_target_settling_lag_ms": float(np.median(steps)) if len(steps) == 2 else None,
            "median_glide_equivalent_delay_ms": float(np.median(glides)) if len(glides) == 6 else None}
    components = {}
    for method in ("gru", "mlp"):
        differences = {}
        for name in ("median_signed_target_delay_ms", "median_step_target_settling_lag_ms", "median_glide_equivalent_delay_ms"):
            candidate, baseline = summaries[method][name], summaries["dsp"][name]
            differences[name] = None if candidate is None or baseline is None else candidate - baseline
        passed = all(value is not None and value <= 10 for value in differences.values())
        components[method] = {"extra_median_target_delay_vs_delayed_dsp_ms": differences, "transition_lag_component_pass": passed}
    return {"aggregation": "equal-scenario median of2 step settling lags and6 glide median signed equivalent delays; all8 required; step-only andglide-only medians additionally gated",
            "measurement_delay_ms_reported_separately": 20, "methods": summaries, "model_components": components,
            "final_keep": False, "warning": "This component cannot establish a model keep decision without natural-data quality and runtime gates"}


def run(support_run, training_features, model_folders, output):
    training_manifest, _ = ev.read_cache_manifest(training_features)
    if training_manifest["split"] != "train":
        raise ValueError("Requires train-only checkpoint provenance")
    models = {name: ev.load_frozen_model(folder, training_features, training_manifest) for name, folder in model_folders.items()}
    if set(models) != {"gru", "mlp"} or models["gru"][0].kind != "causal-gru" or models["mlp"][0].kind != "framewise-mlp":
        raise ValueError("Requires both frozen family members")
    old_manifest = json.loads((support_run / "manifest.json").read_text())
    if old_manifest["schemaVersion"] != "phenometrix.support-diagnostic.v1" or len(old_manifest["recordings"]) != 11:
        raise ValueError("Requires the frozen original11-scenario support run")
    old_scenarios = {s["id"]: s for s in old_manifest["scenarios"]}
    if len(old_scenarios) != 11 or {s["kind"] for s in old_scenarios.values()} != {"constant", "glide", "step"}:
        raise ValueError("Original scenario declaration changed")
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("intermediate", "features", "predictions", "source-snapshot", "provenance"):
        (output / directory).mkdir()
    os.environ.setdefault("NUMBA_CACHE_DIR", str(RESEARCH / "runs" / ".numba-cache"))
    paths = [Path(__file__), Path(ev.__file__), Path(__file__).with_name("measurement_alignment.py"),
             Path(__file__).with_name("model.py"), Path(__file__).with_name("pilot.py"),
             RESEARCH / "replay.ts"]
    paths += [REPO / path for path in old_manifest["sourceSha256"] if path.startswith("apps/")]
    source_hashes = {str(p.relative_to(REPO)): ev.sha(p) for p in paths}
    for path in paths:
        relative = str(path.relative_to(REPO))
        if relative.startswith("apps/") and source_hashes[relative] != old_manifest["sourceSha256"][relative]:
            raise ValueError("Original candidate/DSP source changed")
        target = output / "source-snapshot" / path.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    for name, path in {"original-support-manifest.json": support_run / "manifest.json",
                       "training-feature-manifest.json": training_features / "manifest.json",
                       **{name + "-training-manifest.json": folder / "manifest.json" for name, folder in model_folders.items()}}.items():
        (output / "provenance" / name).write_bytes(path.read_bytes())
    selection_file = RESEARCH / "selection-decision.json"
    (output / "provenance" / "selection-decision.json").write_bytes(selection_file.read_bytes())
    manifest = {"schema": "phenometrix-delayed-known-truth-evaluation-v1", "selection_decision_sha256": ev.sha(selection_file), "status": "predicting",
        "measurement_delay_ms": 20, "sample_rate_hz": 48000, "started_at": datetime.now(timezone.utc).isoformat(),
        "original_support_manifest_sha256": ev.sha(support_run / "manifest.json"),
        "training_feature_manifest_sha256": ev.sha(training_features / "manifest.json"), "source_sha256": source_hashes,
        "observation_change": {"frequencies_hz": [100, 200, 300], "at_seconds": .6, "duration_seconds": 1.6,
            "harmonic_amplitudes_before": [1, .5, .25], "harmonic_amplitudes_after": [0, 1, .5], "amplitude": .075,
            "identifiability": "harmonics2and3 retained; gcd2,3=1; unchanged analytic fundamental; no added noise or normalization"},
        "models": {name: {"manifest_sha256": ev.sha(model_folders[name] / "manifest.json"), "weights_sha256": model[2]["weights_sha256"]} for name, model in models.items()},
        "analytic_target_arrays_read_before_prediction_freeze": False, "recordings": []}
    ev.write_json(output / "manifest.json", manifest)
    try:
        # Input phase: old target arrays are never opened here; predictors see PCM/features only.
        inputs = [(r["id"], r) for r in old_manifest["recordings"]] + [(s["id"], None) for s in OBSERVATION_SCENARIOS]
        for name, original in inputs:
            if original is not None:
                raw = (support_run / "pcm" / (name + ".f32le")).read_bytes()
                if hashlib.sha256(raw).hexdigest() != original["pcmSha256"]:
                    raise ValueError("Original synthetic PCM changed")
                pcm = np.frombuffer(raw, dtype="<f4").copy()
            else:
                pcm = observation_pcm(next(s for s in OBSERVATION_SCENARIOS if s["id"] == name))
                raw = pcm.astype("<f4").tobytes()
            raw_path, full_path = output / "intermediate" / (name + ".f32le"), output / "intermediate" / (name + ".json")
            raw_path.write_bytes(raw)
            subprocess.run(["node", "--import", "tsx", "../../research/voice-prosody/replay.ts", "--feature-version", "v2",
                            "--input-f32", str(raw_path), "--sample-rate", "48000", "--output", str(full_path)],
                           cwd=REPO / "apps/capture-web", check=True, capture_output=True, text=True)
            replay = json.loads(full_path.read_text())
            frames = replay["frames"]
            hz = np.array([[c["hz"] if c["valid"] else np.nan for c in f["features"]["candidates"]] for f in frames], np.float64)
            feature = {"features": np.array([f["features"]["values"] for f in frames], np.float32), "candidate_hz": hz,
                "candidate_mask": np.isfinite(hz), "reference_sample": np.array([f["referenceSample"] for f in frames], np.int64),
                "feature_names": np.array(replay["featureNames"]), "sample_rate_hz": np.array(48000), "pcm_sample_count": np.array(len(pcm))}
            timing = ev.validate_feature(feature, {"sample_count": len(pcm)})
            feature_path = output / "features" / (name + ".npz")
            np.savez_compressed(feature_path, **feature)
            dsp = []
            for frame in frames:
                b = frame["features"]["baseline"]
                voiced = b["f0Hz"] is not None and b["f0Confidence"] >= .55 and b["estimatorAgreement"] >= .7
                dsp.append({"sample": frame["referenceSample"], "state": "voiced" if voiced else "unvoiced", "f0_hz": b["f0Hz"] if voiced else None})
            yin, _ = yin_predictions(pcm, feature["reference_sample"])
            predictions = {"dsp": ev.delayed_baseline(dsp, feature), "yin": ev.delayed_baseline([asdict(r) for r in yin], feature),
                           "continuity": ev.continuity_predictions(feature)}
            for method, (config, weights, _) in models.items():
                predictions[method] = ev.model_predictions(feature, config, weights)
            prediction_path = output / "predictions" / (name + ".json.gz")
            ev.write_predictions(prediction_path, {"scenario_id": name, "time_contract": timing.metadata(),
                "availability_sample": timing.availability_sample.tolist(), "predictions": predictions})
            manifest["recordings"].append({"id": name, "pcm_sha256": hashlib.sha256(raw).hexdigest(),
                "feature_sha256": ev.sha(feature_path), "prediction_sha256": ev.sha(prediction_path), "frames": len(frames)})
            raw_path.unlink(); full_path.unlink()
        (output / "intermediate").rmdir()
        manifest.update(status="predictions_frozen", predictions_frozen_at=datetime.now(timezone.utc).isoformat())
        ev.write_json(output / "prediction-freeze.json", manifest)
        ev.write_json(output / "manifest.json", manifest)
        # Scorer phase: known analytic truth is accessed only after all outputs exist.
        scenarios, traces = [], []
        for record in manifest["recordings"]:
            name = record["id"]
            if name in old_scenarios:
                target_path = support_run / "analytic-targets" / (name + ".json")
                truth = json.loads(target_path.read_text())
                scenario = truth["scenario"]
                if scenario != old_scenarios[name]:
                    raise ValueError("Original analytic scenario changed")
                for target in truth["targets"]:
                    expected = analytic_frequency(scenario, (target["sample"] - 960) / 48000)
                    if abs(target["midpointHz"] - expected) > 1e-9:
                        raise ValueError("Analytic label does not match independent derivative")
                record["analytic_target_sha256"] = ev.sha(target_path)
            else:
                scenario = next(s for s in OBSERVATION_SCENARIOS if s["id"] == name)
            file = output / "predictions" / (name + ".json.gz")
            if ev.sha(file) != record["prediction_sha256"]:
                raise ValueError("Prediction changed after freeze")
            metrics, local = score_predictions(scenario, ev.read_predictions(file))
            scenarios.append({"scenario": scenario, "methods": metrics})
            traces.extend(local)
        result = {"schema": "phenometrix-delayed-known-truth-scores-v1", "measurement_delay_ms": 20,
            "scenario_count": len(scenarios), "evidence": "fixed noiseless synthetic examples only; no speech/generalization/clinical claim",
            "scenarios": scenarios, "transition_gate": transition_gate(scenarios),
            "overall_by_region": {method: {region: summarize([r for r in traces if r["method"] == method and (region == "all" or r["region"] == region)])
                for region in sorted({r["region"] for r in traces} | {"all"})} for method in METHODS}}
        ev.write_json(output / "metrics.json", result)
        ev.write_predictions(output / "scored-traces.json.gz", traces)
        for name, expected in source_hashes.items():
            if ev.sha(REPO / name) != expected:
                raise ValueError("Known-truth evaluator source changed during run")
        manifest.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat(), metrics_sha256=ev.sha(output / "metrics.json"),
                        scored_traces_sha256=ev.sha(output / "scored-traces.json.gz"))
        ev.write_json(output / "manifest.json", manifest)
        return result
    except BaseException as error:
        manifest.update(status="failed", failure=repr(error))
        ev.write_json(output / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-run", type=Path, required=True)
    parser.add_argument("--training-features", type=Path, required=True)
    parser.add_argument("--gru", type=Path, required=True)
    parser.add_argument("--mlp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.support_run.resolve(), args.training_features.resolve(),
                 {"gru": args.gru.resolve(), "mlp": args.mlp.resolve()}, args.output.resolve())
    print(json.dumps({"scenario_count": result["scenario_count"], "transition_gate": result["transition_gate"]}, indent=2))


if __name__ == "__main__":
    main()
