"""Frozen-checkpoint evaluation for the separate declared 20 ms measurement task.

All acquisition-only predictions are persisted before any reference file is read.
Neural inference uses independent NumPy equations, with zero state per recording.
Raw voicing probabilities are not F0 correctness probabilities or calibration.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import gzip
import json
from pathlib import Path
import re
import numpy as np
import torch
from .measurement_alignment import align_feature_timing
from .metrics import Prediction, Reference, score, oracle_coverage
from .model import ModelConfig, VoiceCandidateModel, model_metadata, numpy_step, FEATURE_NAMES_V2
from .pilot import references_from_array, REPO

METRICS = (
    "joint_error", "reference_voiced_success_rate", "voicing_false_positive_rate",
    "voicing_false_negative_rate", "voicing_false_negative_or_unavailable_rate",
    "measurement_coverage", "jointly_voiced_coverage", "median_jointly_voiced_cents",
    "p95_jointly_voiced_cents", "octave_error_rate_reference_voiced",
)
SIMPLE_METHODS = ("dsp", "yin", "continuity", "mlp")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")


def write_predictions(path, value):
    """Lossless deterministic compression keeps raw research outputs bounded."""
    raw = (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()
    Path(path).write_bytes(gzip.compress(raw, compresslevel=6, mtime=0))


def read_predictions(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def read_cache_manifest(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    if (manifest.get("schema") != "phenometrix-input-only-feature-cache-v1"
            or manifest.get("status") != "complete" or manifest.get("split") not in {"train", "development"}
            or manifest.get("feature_version") != "v2" or manifest.get("measurement_delay_ms") != 20
            or manifest.get("reference_or_laryngograph_content_read") is not False):
        raise ValueError("Requires a complete input-only v2 train/development cache with declared 20 ms delay")
    acquisition = folder / "acquisition-manifest.json"
    if sha(acquisition) != manifest["data_manifest_sha256"]:
        raise ValueError("Frozen acquisition manifest identity changed")
    data = json.loads(acquisition.read_text())  # identities only; no reference content
    records, conditions = manifest["recordings"], manifest["conditions"]
    if not records or not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("Empty records or invalid condition declaration")
    if len({r["recording_id"] for r in records}) != len(records):
        raise ValueError("Duplicate feature-cache recording")
    microphones = {r["recording_id"]: r for r in data["files"]
                   if r["role"] == "mic" and r["split"] == manifest["split"]}
    reference_metadata = {r["recording_id"]: r for r in data["files"]
                          if r["role"] == "ref" and r["split"] == manifest["split"]}
    observed = defaultdict(set)
    for r in records:
        source, condition = r["source_recording_id"], r["condition"]
        if (source not in microphones or source not in reference_metadata
                or r["recording_id"] != source + "__" + condition or condition not in conditions
                or r["microphone_sha256"] != microphones[source]["sha256"]
                or r["speaker"] != microphones[source]["speaker"]
                or r["speaker"] != reference_metadata[source]["speaker"]):
            raise ValueError("Cache source identity, speaker, split or condition mismatch")
        observed[source].add(condition)
    if any(c != set(conditions) for c in observed.values()):
        raise ValueError("Each source must include all declared conditions")
    return manifest, data


def load_frozen_model(folder, training_folder, training_manifest):
    manifest = json.loads((folder / "manifest.json").read_text())
    if (manifest.get("status") != "complete" or manifest.get("split") != "train"
            or manifest.get("measurement_delay_ms") != 20
            or manifest.get("test_or_development_labels_used") is not False
            or manifest.get("feature_run_manifest_sha256") != sha(training_folder / "manifest.json")):
        raise ValueError("Checkpoint lacks matching completed train-only 20 ms provenance")
    expected_features = {r["recording_id"]: r["feature_sha256"] for r in training_manifest["recordings"]}
    if manifest.get("feature_hashes") != expected_features:
        raise ValueError("Checkpoint training feature identities differ from frozen train cache")
    acquisition = json.loads((training_folder / "acquisition-manifest.json").read_text())
    source_ids = {r["source_recording_id"] for r in training_manifest["recordings"]}
    reference_identities = {r["recording_id"]: r["sha256"] for r in acquisition["files"]
                            if r["role"] == "ref" and r["split"] == "train" and r["recording_id"] in source_ids}
    if manifest.get("reference_hashes") != reference_identities:
        raise ValueError("Checkpoint training reference identities differ from frozen acquisition metadata")
    supplement = manifest.get("synthetic_curriculum")
    if supplement is not None:
        supplement_file = folder / "synthetic-curriculum-manifest.json"
        if sha(supplement_file) != supplement.get("manifest_sha256"):
            raise ValueError("Synthetic training supplement snapshot identity changed")
        curriculum = json.loads(supplement_file.read_text())
        if (curriculum.get("status") != "complete" or curriculum.get("split") != "train"
                or curriculum.get("feature_version") != "v2" or curriculum.get("measurement_delay_ms") != 20
                or manifest.get("kind") != "framewise-mlp"
                or supplement.get("challenge_content_or_outcomes_used") is not False
                or supplement.get("repetitions_for_training_and_scaling") != 4):
            raise ValueError("Synthetic supplement violates the frozen augmented-MLP contract")
        records = curriculum["records"]
        identities = {r["id"]: {"feature_sha256": r["feature_sha256"], "reference_sha256": r["reference_sha256"]} for r in records}
        if (len(identities) != len(records) or supplement.get("unique_sequence_count") != len(records)
                or supplement.get("feature_and_reference_sha256") != identities):
            raise ValueError("Synthetic supplement record identities differ from training provenance")
    checkpoint = folder / "weights.pt"
    if sha(checkpoint) != manifest["weights_sha256"]:
        raise ValueError("Checkpoint weights changed")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = ModelConfig(**saved["config"])
    if config.feature_version != "v2" or config.projection_size != 32 or config.hidden_size != 32:
        raise ValueError("Checkpoint does not match the frozen v2/32 experiment")
    validator = VoiceCandidateModel(config)
    expected_state = validator.state_dict()
    actual = saved["state_dict"]
    if set(actual) != set(expected_state):
        raise ValueError("Unexpected checkpoint tensor names")
    for name, value in actual.items():
        if value.shape != expected_state[name].shape or value.dtype != expected_state[name].dtype:
            raise ValueError("Checkpoint tensor shape/dtype mismatch")
        if value.dtype != torch.bool and not bool(torch.isfinite(value).all()):
            raise ValueError("Nonfinite checkpoint tensor")
    validator.load_state_dict(actual, strict=True)
    if not bool(validator.scaling_fitted) or not bool((validator.feature_scale > 0).all()):
        raise ValueError("Checkpoint requires positive train-fitted scaling buffers")
    if manifest.get("model") != model_metadata(validator) or manifest.get("kind") != config.kind:
        raise ValueError("Checkpoint model contract differs from manifest")
    return config, {k: v.detach().cpu().numpy().copy() for k, v in actual.items()}, manifest


def validate_feature(feature, record):
    if list(feature["feature_names"]) != list(FEATURE_NAMES_V2):
        raise ValueError("Feature ordering mismatch")
    x, hz, mask = feature["features"], feature["candidate_hz"], feature["candidate_mask"]
    if (x.ndim != 2 or x.shape[1] != 57 or x.dtype != np.float32 or not np.isfinite(x).all()
            or hz.shape != (len(x), 6) or hz.dtype != np.float64
            or mask.shape != hz.shape or mask.dtype != np.bool_
            or int(feature["pcm_sample_count"]) != record["sample_count"]
            or int(feature["sample_rate_hz"]) != 48000):
        raise ValueError("Invalid acquisition feature contract")
    if (not np.isfinite(hz[mask]).all() or np.any(hz[mask] < 50) or np.any(hz[mask] > 700)
            or not np.isnan(hz[~mask]).all()):
        raise ValueError("Candidate absence/bounds contract violated")
    for row, valid in zip(hz, mask):
        if np.any(np.diff(row[valid]) <= 0):
            raise ValueError("Candidates must be strictly increasing Hz")
    timing = align_feature_timing(feature, measurement_delay_ms=20)
    if len(x) and int(timing.availability_sample[0]) != 2208:
        raise ValueError("Feature cache omits the recording's initial causal warmup frames")
    return timing


def model_predictions(feature, config, weights):
    """No reference argument. Carries every acquired frame, including warmup."""
    timing = align_feature_timing(feature, measurement_delay_ms=20)
    state = np.zeros((1, config.hidden_size), np.float32)
    rows = []
    for i, sample in enumerate(timing.reference_sample):
        mask = feature["candidate_mask"][i:i + 1]
        voicing, logits, state = numpy_step(config, weights, feature["features"][i:i + 1], mask, state)
        if not all(np.isfinite(a).all() for a in (voicing, logits, state)):
            raise ValueError("Nonfinite independent NumPy prediction")
        probability = float(np.exp(-np.logaddexp(0., -float(voicing[0, 0]))))
        chosen = None
        if probability < .5:
            prediction = Prediction(int(sample), "unvoiced", None)
        elif not mask.any():
            prediction = Prediction(int(sample), "unavailable", None)
        else:
            chosen = int(np.argmax(np.where(mask[0], logits[0], -np.inf)))
            prediction = Prediction(int(sample), "voiced", float(feature["candidate_hz"][i, chosen]))
        rows.append({**asdict(prediction), "voicing_probability": probability, "candidate_index": chosen})
    return rows


def continuity_predictions(feature):
    """Frozen evidence/continuity rule; reset after every nonvoiced output."""
    timing = align_feature_timing(feature, measurement_delay_ms=20)
    names = list(feature["feature_names"])
    score_indices = [names.index(f"candidate_{i}_maximum_score") for i in range(6)]
    rms_index = names.index("log_rms_floor_1e-8")
    previous, rows = None, []
    for i, sample in enumerate(timing.reference_sample):
        hz, mask = feature["candidate_hz"][i], feature["candidate_mask"][i]
        evidence = feature["features"][i, score_indices].astype(np.float64)
        log_rms = feature["features"][i, rms_index]
        chosen = None
        if not mask.any() or log_rms < np.float32(np.log(.003)) or float(np.max(evidence[mask])) < .55:
            prediction = Prediction(int(sample), "unvoiced", None)
            previous = None
        else:
            penalized = evidence.copy()
            if previous is not None:
                penalized[mask] -= .25 * np.minimum(np.abs(1200 * np.log2(hz[mask] / previous)) / 1200, 1)
            chosen = int(np.argmax(np.where(mask, penalized, -np.inf)))
            previous = float(hz[chosen])
            prediction = Prediction(int(sample), "voiced", previous)
        rows.append({**asdict(prediction), "candidate_index": chosen})
    return rows


def delayed_baseline(rows, feature):
    timing = align_feature_timing(feature, measurement_delay_ms=20)
    if [r["sample"] for r in rows] != timing.availability_sample.tolist():
        raise ValueError("Baseline and feature acquisition timestamps differ")
    for row in rows:
        if row["state"] not in {"voiced", "unvoiced", "unavailable"}:
            raise ValueError("Invalid baseline state")
        hz = row["f0_hz"]
        if (row["state"] == "voiced" and (hz is None or not np.isfinite(hz) or hz <= 0)
                or row["state"] != "voiced" and hz is not None):
            raise ValueError("Invalid baseline frequency/state pairing")
    return [asdict(Prediction(int(target), r["state"], r["f0_hz"]))
            for r, target in zip(rows, timing.reference_sample)]


def prediction_objects(rows):
    return [Prediction(r["sample"], r["state"], r["f0_hz"]) for r in rows]


def macro(cells):
    """Equal cell weights, with missing-denominator counts always explicit."""
    output = {"cell_count": len(cells), "defined_cells": {}}
    for name in METRICS:
        values = [c[name] for c in cells if c[name] is not None]
        output[name] = float(np.mean(values)) if values else None
        output["defined_cells"][name] = len(values)
    return output


def combine_scores(items):
    references, predictions, offset = [], [], 0
    for refs, preds, sample_count in items:
        references.extend(replace(r, sample=r.sample + offset) for r in refs)
        predictions.extend(replace(p, sample=p.sample + offset) for p in preds)
        offset += sample_count + 48000
    return score(references, predictions)


def quality_components(candidate, comparators, overall, clean):
    required = {candidate, *comparators}
    if not required <= set(overall) or not required <= set(clean):
        return {"status": "incomplete_comparators", "pass": False}
    needed = [(overall[m], metric) for m in required for metric in ("joint_error", "reference_voiced_success_rate")]
    needed += [(clean[m], "joint_error") for m in required]
    if any(cell.get(metric) is None for cell, metric in needed):
        return {"status": "incomplete_metric_denominators", "pass": False}
    reductions = {m: overall[m]["joint_error"] - overall[candidate]["joint_error"] for m in comparators}
    success_loss = max(overall[m]["reference_voiced_success_rate"] for m in comparators) - overall[candidate]["reference_voiced_success_rate"]
    clean_excess = clean[candidate]["joint_error"] - min(clean[m]["joint_error"] for m in comparators)
    passed = all(v >= .01 for v in reductions.values()) and success_loss <= .02 and clean_excess <= .01
    return {"status": "quality_components_pass_pending_other_gates" if passed else "quality_components_fail", "pass": passed,
            "comparators": list(comparators), "joint_error_reduction_by_comparator": reductions,
            "voiced_success_loss_against_best_simple": success_loss,
            "clean_joint_error_excess_against_best_simple": clean_excess}


def quality_gate(overall, clean):
    """Pre-development selection clarification; quality alone never establishes keep."""
    gru = quality_components("gru", SIMPLE_METHODS, overall, clean)
    mlp = quality_components("mlp", ("dsp", "yin", "continuity"), overall, clean)
    classical = [m for m in ("dsp", "yin", "continuity") if m in overall and overall[m]["joint_error"] is not None]
    best_classical = min(classical, key=lambda m: (overall[m]["joint_error"], m)) if classical else None
    provisional = "gru" if gru["pass"] else "mlp" if mlp["pass"] else best_classical
    return {"gru": gru, "mlp": mlp, "provisional_selection_from_quality_only": provisional,
            "minimum_joint_error_classical": best_classical, "keep": False,
            "selection_note": "GRU if every gate passes, otherwise independently qualified MLP, otherwise classical comparator with limitations; no production integration",
            "requires_separate_checks": ["all8 known-truth delays defined; combined, step-only and glide-only extra median lag <=10ms versus delayed DSP",
                                         "complete worker service and drop/coverage runtime gate", "frozen later confirmation and calibration gates"]}


def evaluate(feature_folder, training_folder, data_root, model_folders, output):
    cache, acquisition = read_cache_manifest(feature_folder)
    train, _ = read_cache_manifest(training_folder)
    if train["split"] != "train":
        raise ValueError("Training provenance must be a train cache")
    if cache["conditions"] != train["conditions"]:
        raise ValueError("Evaluation and training conditions differ")
    if cache["split"] == "train":
        if sha(feature_folder / "manifest.json") != sha(training_folder / "manifest.json"):
            raise ValueError("Training evaluation must use the exact checkpoint feature cache")
    else:
        for key in ("speaker", "source_recording_id"):
            if {r[key] for r in cache["recordings"]} & {r[key] for r in train["recordings"]}:
                raise ValueError("Development overlaps training source or speaker")
    models = {}
    for name, folder in model_folders.items():
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", name) or name in {"dsp", "yin", "continuity"}:
            raise ValueError("Invalid or reserved model name")
        models[name] = load_frozen_model(folder, training_folder, train)
    if set(models) != {"gru", "mlp"} or models["gru"][0].kind != "causal-gru" or models["mlp"][0].kind != "framewise-mlp":
        raise ValueError("Frozen comparison requires named gru and mlp checkpoints")
    output.mkdir(parents=True, exist_ok=False)
    (output / "predictions").mkdir()
    provenance = output / "provenance"
    provenance.mkdir()
    for name, path in {"evaluation-feature-manifest.json": feature_folder / "manifest.json",
                       "training-feature-manifest.json": training_folder / "manifest.json",
                       "evaluation-acquisition-manifest.json": feature_folder / "acquisition-manifest.json",
                       "training-acquisition-manifest.json": training_folder / "acquisition-manifest.json",
                       **{name + "-training-manifest.json": path / "manifest.json" for name, path in model_folders.items()}}.items():
        (provenance / name).write_bytes(path.read_bytes())
    sources = [Path(__file__), Path(__file__).with_name("measurement_alignment.py"), Path(__file__).with_name("model.py"),
               Path(__file__).with_name("metrics.py"), Path(__file__).with_name("pilot.py")]
    source_hashes = {str(p.relative_to(REPO)): sha(p) for p in sources}
    for p in sources:
        target = output / "source-snapshot" / p.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    selection_file = Path(__file__).parents[1] / "selection-decision.json"
    (provenance / "selection-decision.json").write_bytes(selection_file.read_bytes())
    manifest = {"schema": "phenometrix-delayed-evaluation-v1", "selection_decision_sha256": sha(selection_file), "status": "predicting", "split": cache["split"],
        "measurement_delay_ms": 20, "started_at": datetime.now(timezone.utc).isoformat(),
        "feature_manifest_sha256": sha(feature_folder / "manifest.json"),
        "training_feature_manifest_sha256": sha(training_folder / "manifest.json"),
        "source_sha256": source_hashes, "conditions": cache["conditions"],
        "inference": "independent NumPy equations; zero state per recording; every acquisition frame carried",
        "reference_content_read_before_prediction_freeze": False,
        "prediction_encoding": "deterministic gzip of complete UTF-8 JSON; lossless; mtime=0",
        "probability_semantics": "raw voicing probability only; correctness_probability null; no calibration",
        "models": {name: {"manifest_sha256": sha(model_folders[name] / "manifest.json"),
                          "weights_sha256": value[2]["weights_sha256"], "model": value[2]["model"]}
                   for name, value in models.items()}, "recordings": []}
    write_json(output / "manifest.json", manifest)
    try:
        # PHASE 1: this loop opens no reference/laryngograph content.
        for record in cache["recordings"]:
            identity = record["recording_id"]
            file = feature_folder / "features" / (identity + ".npz")
            baseline_file = feature_folder / "predictions" / (identity + ".json")
            if sha(file) != record["feature_sha256"] or sha(baseline_file) != record["prediction_sha256"]:
                raise ValueError("Acquisition-only cached input identity changed")
            with np.load(file, allow_pickle=False) as stored:
                feature = {key: stored[key].copy() for key in stored.files}
            timing = validate_feature(feature, record)
            baseline = json.loads(baseline_file.read_text())
            if baseline["recording_id"] != identity:
                raise ValueError("Baseline recording identity mismatch")
            predictions = {name: delayed_baseline(baseline[name], feature) for name in ("dsp", "yin")}
            predictions["continuity"] = continuity_predictions(feature)
            for name, (config, weights, _) in models.items():
                predictions[name] = model_predictions(feature, config, weights)
            prediction_file = output / "predictions" / (identity + ".json.gz")
            write_predictions(prediction_file, {"recording_id": identity, "time_contract": timing.metadata(),
                "availability_sample": timing.availability_sample.tolist(), "predictions": predictions})
            manifest["recordings"].append({**record, "evaluated_prediction_sha256": sha(prediction_file), "evaluated_prediction_path": "predictions/" + identity + ".json.gz"})
        manifest.update(status="predictions_frozen", predictions_frozen_at=datetime.now(timezone.utc).isoformat())
        write_json(output / "prediction-freeze.json", manifest)
        write_json(output / "manifest.json", manifest)
        # PHASE 2: all model/baseline outputs are fixed before the first label load.
        refs_by_source = {r["recording_id"]: r for r in acquisition["files"]
                          if r["role"] == "ref" and r["split"] == cache["split"]}
        methods = ["dsp", "yin", "continuity", *models]
        groups = {method: defaultdict(list) for method in methods}
        oracle_groups = defaultdict(list)
        reference_cache, per_record = {}, []
        for record in manifest["recordings"]:
            identity, source = record["recording_id"], record["source_recording_id"]
            if source not in reference_cache:
                metadata = refs_by_source[source]
                reference_file = data_root / metadata["path"]
                if sha(reference_file) != metadata["sha256"]:
                    raise ValueError("Frozen reference identity changed")
                refs = references_from_array(np.loadtxt(reference_file), record["sample_count"])
                reference_cache[source] = [replace(r, eligible=r.sample >= 3072 and r.sample + 1536 <= record["sample_count"]) for r in refs]
            refs = reference_cache[source]
            record["reference_sha256"] = refs_by_source[source]["sha256"]
            prediction_file = output / "predictions" / (identity + ".json.gz")
            if sha(prediction_file) != record["evaluated_prediction_sha256"]:
                raise ValueError("Prediction changed after reference-free freeze")
            frozen = read_predictions(prediction_file)["predictions"]
            targets = {r.sample for r in refs}
            metrics, outside = {}, {}
            for method in methods:
                all_predictions = prediction_objects(frozen[method])
                predictions = [p for p in all_predictions if p.sample in targets]
                outside[method] = len(all_predictions) - len(predictions)
                metrics[method] = score(refs, predictions)
                groups[method][(record["speaker"], record["condition"])].append((refs, predictions, record["sample_count"]))
            feature_file = feature_folder / "features" / (identity + ".npz")
            if sha(feature_file) != record["feature_sha256"]:
                raise ValueError("Feature candidates changed after prediction freeze")
            with np.load(feature_file, allow_pickle=False) as feature:
                timing = align_feature_timing(feature, measurement_delay_ms=20)
                candidates = {int(t): hz[mask].tolist() for t, hz, mask in zip(timing.reference_sample, feature["candidate_hz"], feature["candidate_mask"])}
            oracle = oracle_coverage(refs, candidates)
            oracle_groups[(record["speaker"], record["condition"])].append(oracle)
            per_record.append({"recording_id": identity, "source_recording_id": source, "speaker": record["speaker"],
                "condition": record["condition"], "metrics": metrics, "candidate_oracle": oracle,
                "predictions_outside_reference_extent": outside})
        speakers = sorted({r["speaker"] for r in cache["recordings"]})
        cells, by_condition, by_speaker, overall = {}, {}, {}, {}
        for method in methods:
            cells[method] = {speaker: {condition: combine_scores(groups[method][speaker, condition])
                                      for condition in cache["conditions"]} for speaker in speakers}
            by_condition[method] = {condition: macro([cells[method][speaker][condition] for speaker in speakers])
                                    for condition in cache["conditions"]}
            by_speaker[method] = {speaker: macro(list(cells[method][speaker].values())) for speaker in speakers}
            overall[method] = macro([cells[method][s][c] for s in speakers for c in cache["conditions"]])
        oracle_cells = {s: {} for s in speakers}
        for (speaker, condition), rows in oracle_groups.items():
            denominator = sum(r["reference_voiced_frames"] for r in rows)
            numerator = sum(r["covered_frames"] for r in rows)
            oracle_cells[speaker][condition] = {"reference_voiced_frames": denominator, "covered_frames": numerator,
                                                "coverage": numerator / denominator if denominator else None}
        oracle_values = [oracle_cells[s][c]["coverage"] for s in speakers for c in cache["conditions"]]
        result = {"schema": "phenometrix-delayed-evaluation-metrics-v1", "split": cache["split"],
            "measurement_delay_ms": 20, "mask": "t>=64ms and t+32ms<=N; exact16+10k-ms target grid",
            "aggregation": "pool recordings within speaker-condition, then weight every speaker-condition cell equally",
            "per_record": per_record, "per_speaker_condition": cells, "per_condition_equal_speaker": by_condition,
            "per_speaker_equal_condition": by_speaker, "equal_speaker_equal_condition": overall,
            "candidate_oracle_per_speaker_condition": oracle_cells,
            "candidate_oracle_equal_speaker_equal_condition": float(np.mean([v for v in oracle_values if v is not None])) if any(v is not None for v in oracle_values) else None,
            "candidate_oracle_defined_cells": sum(v is not None for v in oracle_values),
            "quality_gate": quality_gate(overall, {m: by_condition[m]["clean"] for m in methods})}
        write_json(output / "metrics.json", result)
        if sha(feature_folder / "manifest.json") != manifest["feature_manifest_sha256"]:
            raise ValueError("Feature manifest changed during evaluation")
        for name, expected in source_hashes.items():
            if sha(REPO / name) != expected:
                raise ValueError("Evaluator source changed during evaluation")
        manifest.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat(), metrics_sha256=sha(output / "metrics.json"))
        write_json(output / "manifest.json", manifest)
        return result
    except BaseException as error:
        manifest.update(status="failed", failure=repr(error))
        write_json(output / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--training-features", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--gru", type=Path, required=True)
    parser.add_argument("--mlp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.features.resolve(), args.training_features.resolve(), args.data.resolve(),
                      {"gru": args.gru.resolve(), "mlp": args.mlp.resolve()}, args.output.resolve())
    print(json.dumps({"split": result["split"], "metrics": result["equal_speaker_equal_condition"], "quality_gate": result["quality_gate"]}, indent=2))


if __name__ == "__main__":
    main()
