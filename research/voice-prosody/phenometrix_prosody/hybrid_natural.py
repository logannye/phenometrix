"""No-fit hybrid evaluation from frozen development probabilities and raw YIN.

Both fixed variants are persisted before this process opens any reference file.
Reused development is exploratory. This runner never admits the sealed test.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from . import evaluate as ev
from .hybrid_followup import hybrid_predictions, read_frozen_plan
from .pilot import REPO

VARIANTS = ("original", "augmented")
METHODS = ("dsp", "yin", "continuity", "gru", "mlp_original", "mlp_augmented",
           "hybrid_original", "hybrid_augmented")
CONDITIONS = ("clean", "noise10db", "first_difference", "clip01")
SPEAKERS = ("F01", "F03", "M01", "M03")


def _declared_path(entry):
    path = Path(entry["directory"])
    return (path if path.is_absolute() else REPO / "research/voice-prosody/runs" / path).resolve()


def _inside(folder, relative):
    path = (folder / relative).resolve()
    if not path.is_relative_to(folder.resolve()):
        raise ValueError("Artifact escaped its declared directory")
    return path


def _load_sources(plan_folder, feature_folder, evaluations):
    plan = read_frozen_plan(plan_folder)
    if set(evaluations) != set(VARIANTS):
        raise ValueError("Both frozen hybrid variants are required")
    if (feature_folder.resolve() != _declared_path(plan["natural_features"])
            or ev.sha(feature_folder / "manifest.json") != plan["natural_features"]["manifest_sha256"]):
        raise ValueError("Feature cache differs from frozen plan")
    cache, acquisition = ev.read_cache_manifest(feature_folder)
    if (cache["split"] != "development" or tuple(cache["conditions"]) != CONDITIONS
            or sorted({r["speaker"] for r in cache["recordings"]}) != list(SPEAKERS)):
        raise ValueError("Only the declared four-speaker development cache is permitted; sealed test refused")
    expected = cache["recordings"]
    sources = {}
    for variant in VARIANTS:
        folder = evaluations[variant]
        binding = plan["natural_evaluations"][variant]
        if (folder.resolve() != _declared_path(binding)
                or ev.sha(folder / "manifest.json") != binding["manifest_sha256"]
                or ev.sha(folder / "prediction-freeze.json") != binding["prediction_freeze_sha256"]):
            raise ValueError("Frozen evaluation identity differs from plan")
        final = json.loads((folder / "manifest.json").read_text())
        frozen = json.loads((folder / "prediction-freeze.json").read_text())
        for manifest, state in ((final, "complete"), (frozen, "predictions_frozen")):
            if (manifest.get("schema") != "phenometrix-delayed-evaluation-v1"
                    or manifest.get("status") != state or manifest.get("split") != "development"
                    or manifest.get("measurement_delay_ms") != 20
                    or manifest.get("reference_content_read_before_prediction_freeze") is not False
                    or manifest.get("feature_manifest_sha256") != plan["natural_features"]["manifest_sha256"]
                    or manifest.get("conditions") != list(CONDITIONS)
                    or len(manifest.get("recordings", [])) != len(expected)):
                raise ValueError("Requires complete matching input-only development predictions")
            model = manifest["models"]["mlp"]
            if (model["manifest_sha256"] != plan["models"][variant]["manifest_sha256"]
                    or model["weights_sha256"] != plan["models"][variant]["weights_sha256"]
                    or model["model"]["config"]["kind"] != "framewise-mlp"):
                raise ValueError("Frozen probabilities belong to the wrong model")
            for actual, wanted in zip(manifest["recordings"], expected):
                if any(actual.get(k) != v for k, v in wanted.items()):
                    raise ValueError("Frozen evaluation has changed source, order or condition")
        for a, b in zip(final["recordings"], frozen["recordings"]):
            if any(a.get(k) != b.get(k) for k in ("evaluated_prediction_sha256", "evaluated_prediction_path")):
                raise ValueError("Evaluation prediction identity differs from its input-only freeze")
        sources[variant] = frozen
    return plan, cache, acquisition, sources


def _read_input_record(feature_folder, evaluations, sources, record, index):
    identity = record["recording_id"]
    feature_file = _inside(feature_folder, "features/" + identity + ".npz")
    baseline_file = _inside(feature_folder, "predictions/" + identity + ".json")
    if ev.sha(feature_file) != record["feature_sha256"] or ev.sha(baseline_file) != record["prediction_sha256"]:
        raise ValueError("Frozen feature or raw YIN input changed")
    with np.load(feature_file, allow_pickle=False) as stored:
        feature = {k: stored[k].copy() for k in stored.files}
    timing = ev.validate_feature(feature, record)
    baseline = json.loads(baseline_file.read_text())
    if baseline["recording_id"] != identity or "yin_evidence" not in baseline:
        raise ValueError("Raw input-only YIN evidence missing or misidentified")
    frozen = {}
    for variant in VARIANTS:
        row = sources[variant]["recordings"][index]
        path = _inside(evaluations[variant], row["evaluated_prediction_path"])
        if ev.sha(path) != row["evaluated_prediction_sha256"]:
            raise ValueError("Frozen model prediction bytes changed")
        value = ev.read_predictions(path)
        if (value["recording_id"] != identity or value["time_contract"] != timing.metadata()
                or value["availability_sample"] != timing.availability_sample.tolist()):
            raise ValueError("Model prediction time or identity contract differs")
        for rows in value["predictions"].values():
            if [r["sample"] for r in rows] != timing.reference_sample.tolist():
                raise ValueError("Cached model prediction grid differs")
        frozen[variant] = value["predictions"]
    for name in ("dsp", "yin", "continuity", "gru"):
        if frozen["original"][name] != frozen["augmented"][name]:
            raise ValueError("Shared frozen baseline predictions differ between evaluations")
    for name in ("dsp", "yin"):
        if frozen["original"][name] != ev.delayed_baseline(baseline[name], feature):
            raise ValueError("Frozen evaluation baseline differs from feature-cache prediction")
    rows = {name: frozen["original"][name] for name in ("dsp", "yin", "continuity", "gru")}
    for variant in VARIANTS:
        rows["mlp_" + variant] = frozen[variant]["mlp"]
        rows["hybrid_" + variant] = hybrid_predictions(feature, frozen[variant]["mlp"], baseline["yin_evidence"])
    return feature, timing, rows


def freeze_predictions(plan_folder, feature_folder, evaluations, output):
    plan, cache, _, sources = _load_sources(plan_folder, feature_folder, evaluations)
    output.mkdir(parents=True, exist_ok=False)
    (output / "predictions").mkdir()
    (output / "provenance").mkdir()
    copies = {"plan.json": plan_folder / "plan.json", "declaration.json": plan_folder / "declaration.json",
              "feature-manifest.json": feature_folder / "manifest.json",
              "acquisition-manifest.json": feature_folder / "acquisition-manifest.json"}
    for variant, folder in evaluations.items():
        copies[variant + "-evaluation-manifest.json"] = folder / "manifest.json"
        copies[variant + "-prediction-freeze.json"] = folder / "prediction-freeze.json"
    for name, path in copies.items():
        (output / "provenance" / name).write_bytes(path.read_bytes())
    manifest = {"schema": "phenometrix-no-fit-hybrid-natural-v1", "status": "predicting", "split": "development",
        "scope": "Previously inspected development, exploratory; no independent holdout or promotion claim",
        "plan_sha256": ev.sha(plan_folder / "plan.json"), "source_sha256": plan["source_sha256"],
        "feature_manifest_sha256": ev.sha(feature_folder / "manifest.json"),
        "source_evaluation_manifests": {v: ev.sha(p / "manifest.json") for v, p in evaluations.items()},
        "models": plan["models"], "measurement_delay_ms": 20, "methods": list(METHODS),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "reference_content_read_before_prediction_freeze": False, "new_neural_inference": False,
        "probability_semantics": "frozen uncalibrated voicing only; correctness probability null",
        "conditions": cache["conditions"], "recordings": []}
    ev.write_json(output / "manifest.json", manifest)
    try:
        for i, record in enumerate(cache["recordings"]):
            _, timing, predictions = _read_input_record(feature_folder, evaluations, sources, record, i)
            relative = "predictions/" + record["recording_id"] + ".json.gz"
            path = output / relative
            ev.write_predictions(path, {"recording_id": record["recording_id"], "time_contract": timing.metadata(),
                "availability_sample": timing.availability_sample.tolist(), "predictions": predictions})
            manifest["recordings"].append({**record, "hybrid_prediction_path": relative,
                "hybrid_prediction_sha256": ev.sha(path),
                "source_model_prediction_sha256": {v: sources[v]["recordings"][i]["evaluated_prediction_sha256"] for v in VARIANTS}})
        _load_sources(plan_folder, feature_folder, evaluations)
        manifest.update(status="predictions_frozen", predictions_frozen_at=datetime.now(timezone.utc).isoformat())
        ev.write_json(output / "prediction-freeze.json", manifest)
        ev.write_json(output / "manifest.json", manifest)
        return manifest
    except BaseException as error:
        manifest.update(status="failed", failure=repr(error))
        ev.write_json(output / "manifest.json", manifest)
        raise


def score_frozen(plan_folder, feature_folder, evaluations, data_root, output):
    plan, cache, acquisition, sources = _load_sources(plan_folder, feature_folder, evaluations)
    frozen_file = output / "prediction-freeze.json"
    frozen = json.loads(frozen_file.read_text())
    if (frozen.get("schema") != "phenometrix-no-fit-hybrid-natural-v1" or frozen.get("status") != "predictions_frozen"
            or frozen.get("split") != "development" or frozen.get("methods") != list(METHODS)
            or frozen.get("reference_content_read_before_prediction_freeze") is not False
            or frozen.get("plan_sha256") != ev.sha(plan_folder / "plan.json")
            or frozen.get("feature_manifest_sha256") != ev.sha(feature_folder / "manifest.json")
            or frozen.get("source_sha256") != plan["source_sha256"]
            or len(frozen.get("recordings", [])) != len(cache["recordings"])):
        raise ValueError("Requires the complete two-variant input-only freeze")
    if (output / "metrics.json").exists():
        raise FileExistsError("Scoring refuses to overwrite existing evidence")
    # Verify every prediction and source before opening the first reference.
    for i, (record, expected) in enumerate(zip(frozen["recordings"], cache["recordings"])):
        if any(record.get(k) != v for k, v in expected.items()):
            raise ValueError("Prediction freeze changed record order or source identity")
        _read_input_record(feature_folder, evaluations, sources, expected, i)
        if ev.sha(_inside(output, record["hybrid_prediction_path"])) != record["hybrid_prediction_sha256"]:
            raise ValueError("Hybrid prediction changed after freeze")
    refs_meta = {r["recording_id"]: r for r in acquisition["files"] if r["role"] == "ref" and r["split"] == "development"}
    groups = {m: defaultdict(list) for m in METHODS}
    reference_cache, per_record = {}, []
    for record in frozen["recordings"]:
        source = record["source_recording_id"]
        if source not in reference_cache:
            metadata = refs_meta[source]
            path = _inside(data_root, metadata["path"])
            if ev.sha(path) != metadata["sha256"]:
                raise ValueError("Development reference identity changed")
            refs = ev.references_from_array(np.loadtxt(path), record["sample_count"])
            reference_cache[source] = [replace(r, eligible=r.sample >= 3072 and r.sample + 1536 <= record["sample_count"]) for r in refs]
        refs = reference_cache[source]
        value = ev.read_predictions(_inside(output, record["hybrid_prediction_path"]))
        targets = {r.sample for r in refs}
        metrics, outside = {}, {}
        for method in METHODS:
            all_rows = ev.prediction_objects(value["predictions"][method])
            rows = [p for p in all_rows if p.sample in targets]
            outside[method] = len(all_rows) - len(rows)
            metrics[method] = ev.score(refs, rows)
            groups[method][record["speaker"], record["condition"]].append((refs, rows, record["sample_count"]))
        per_record.append({"recording_id": record["recording_id"], "speaker": record["speaker"],
            "condition": record["condition"], "reference_sha256": refs_meta[source]["sha256"],
            "metrics": metrics, "predictions_outside_reference_extent": outside})
    cells, conditions, speakers, overall = {}, {}, {}, {}
    for method in METHODS:
        cells[method] = {s: {c: ev.combine_scores(groups[method][s, c]) for c in CONDITIONS} for s in SPEAKERS}
        conditions[method] = {c: ev.macro([cells[method][s][c] for s in SPEAKERS]) for c in CONDITIONS}
        speakers[method] = {s: ev.macro(list(cells[method][s].values())) for s in SPEAKERS}
        overall[method] = ev.macro([cells[method][s][c] for s in SPEAKERS for c in CONDITIONS])
    result = {"schema": "phenometrix-no-fit-hybrid-natural-metrics-v1", "split": "development",
        "evidence": "Adaptive reuse of previously scored development; exploratory, not a new holdout",
        "plan_sha256": frozen["plan_sha256"], "prediction_freeze_sha256": ev.sha(frozen_file),
        "measurement_delay_ms": 20, "mask": "t>=64ms and t+32ms<=N; exact16+10k-ms nominal target grid",
        "aggregation": "pool recording counts within speaker-condition; equal16cell macro",
        "per_record": per_record, "per_speaker_condition": cells,
        "per_condition_equal_speaker": conditions, "per_speaker_equal_condition": speakers,
        "equal_speaker_equal_condition": overall,
        "quality_components_against_classical": {"hybrid_"+v: ev.quality_components("hybrid_"+v,
            ("dsp", "yin", "continuity"), overall, {m: conditions[m]["clean"] for m in METHODS}) for v in VARIANTS},
        "joint_error_change_vs_own_mlp": {v: overall["hybrid_"+v]["joint_error"]-overall["mlp_"+v]["joint_error"] for v in VARIANTS},
        "keep": False, "limitation": "Natural quality alone cannot qualify either hybrid; fresh synthetic, transition, runtime and later independent confirmation remain separate."}
    ev.write_json(output / "metrics.json", result)
    manifest = {**frozen, "status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
                "metrics_sha256": ev.sha(output / "metrics.json")}
    ev.write_json(output / "manifest.json", manifest)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "features", "original-evaluation", "augmented-evaluation", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--data", type=Path)
    p.add_argument("--phase", choices=("predict", "score", "all"), default="all")
    a = p.parse_args()
    evaluations = {"original": a.original_evaluation.resolve(), "augmented": a.augmented_evaluation.resolve()}
    if a.phase in ("score", "all") and a.data is None:
        p.error("--data is required for scoring")
    if a.phase in ("predict", "all"):
        freeze_predictions(a.plan.resolve(), a.features.resolve(), evaluations, a.output.resolve())
    if a.phase in ("score", "all"):
        score_frozen(a.plan.resolve(), a.features.resolve(), evaluations, a.data.resolve(), a.output.resolve())
    print(json.dumps({"status": "predictions_frozen" if a.phase == "predict" else "complete", "output": str(a.output)}))


if __name__ == "__main__":
    main()
