"""Strict, separately authorized sealed-speaker validation of one frozen hybrid.

Only predict opens selected microphone bytes. Only score opens references, after
checking a complete immutable prediction freeze. The generic pre-freeze cache and
evaluator guards remain closed to this split. No fitting or threshold selection.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import numpy as np
from . import data as acquisition
from . import evaluate as ev
from .features import CONDITIONS, transform
from .hybrid_followup import CONTRACT, WEIGHT_HASHES, hybrid_predictions, _feature, _dsp
from .pilot import REPO, RESEARCH, read_mic, yin_predictions

EXPERIMENT = "phenometrix-hybrid-independent-validation.v1"
SPEAKERS = ("F05", "F06", "M05", "M06")
PAIRS = (5, 6)
PER_SPEAKER = 40
METHODS = ("dsp", "yin", "continuity", "mlp_original", "hybrid_original")
CLASSICAL = ("dsp", "yin", "continuity")
HASH = re.compile(r"[0-9a-f]{64}")


def fixed_configuration(inventory):
    """Metadata only; the selection is fixed before any sealed content access."""
    if inventory.get("seed") != acquisition.SEED or inventory.get("split_assignment") != acquisition.split_assignment():
        raise ValueError("Inventory split does not match the original fixed assignment")
    records = acquisition.select_records(inventory, "sealed_test", PER_SPEAKER, ("mic", "ref"))
    selected = [r for r in records if r["role"] == "mic"]
    if (len(selected) != len(SPEAKERS) * PER_SPEAKER or
            any(sum(r["speaker"] == s for r in selected) != PER_SPEAKER for s in SPEAKERS)):
        raise ValueError("Incomplete fixed sealed speaker inventory")
    for pair in PAIRS:
        a = {r["prompt"] for r in selected if r["speaker"] == f"F{pair:02}"}
        b = {r["prompt"] for r in selected if r["speaker"] == f"M{pair:02}"}
        if a != b:
            raise ValueError("Each fixed F/M pair must have identical selected prompts")
    return {"schema": "phenometrix-hybrid-validation-configuration.v1", "experiment": EXPERIMENT,
        "split": "sealed_test", "speakers": list(SPEAKERS), "per_speaker": PER_SPEAKER,
        "selection": "ascending SHA256(20261022:integer_pair:prompt), then prompt; exclude sa1/sa2",
        "selected_inventory_records": records, "source_recording_count": len(selected),
        "conditions": list(CONDITIONS), "condition_record_count": len(selected) * len(CONDITIONS),
        "transform_split_namespace": "sealed-test", "methods": list(METHODS),
        "estimator_contract": CONTRACT, "model_weights_sha256": WEIGHT_HASHES["original"],
        "voicing_threshold": .5, "raw_yin_inclusive_hz": [50, 700], "measurement_delay_ms": 20,
        "mask": "t>=64ms and t+32ms<=N; exact16+10k-ms nominal target grid",
        "gates": {"joint_error_reduction_vs_each_classical": .01,
            "maximum_voiced_success_loss_vs_best_classical": .02,
            "maximum_clean_joint_error_excess_vs_best_classical": .01},
        "bootstrap": {"seed": 20261022, "draws": 2000, "interval_percentiles": [2.5, 97.5],
            "unit": "paired prompt within each fixed pair; both sexes and all conditions resampled together",
            "scope": "conditional on these four speakers and two pairs; not a population interval",
            "used_for_selection": False}, "refitting": False, "threshold_search": False,
        "secondary_model_switch": False, "production_promotion": False}


def required_sources():
    names = ["hybrid_validation", "hybrid_followup", "features", "data", "evaluate", "model", "pilot",
             "metrics", "measurement_alignment", "evaluate_synthetic", "evaluate_curriculum", "synthetic_curriculum", "sealed_metadata_audit"]
    result = [f"research/voice-prosody/phenometrix_prosody/{n}.py" for n in names]
    result += ["research/voice-prosody/tests/test_hybrid_validation.py", "research/voice-prosody/tests/test_sealed_metadata_audit.py", "research/voice-prosody/replay.ts",
               "research/voice-prosody/uv.lock", "research/voice-prosody/pyproject.toml", "pnpm-lock.yaml"]
    result += [f"apps/capture-web/src/{n}.ts" for n in
               ("voice-replay", "voice-dsp", "dsp-primitives", "voice-pitch-features", "voice-pitch-features-v2")]
    return sorted(result)


def inside(folder, relative):
    path = (folder / relative).resolve()
    if not path.is_relative_to(folder.resolve()):
        raise ValueError("Path escaped its bound directory")
    return path


def read_freeze(path):
    """Verify real bytes, not merely nonempty digest strings, before data access."""
    freeze = json.loads(Path(path).read_text())
    if (freeze.get("schema") != "phenometrix-prosody-freeze-v1" or freeze.get("experiment") != EXPERIMENT
            or freeze.get("sealed_test_authorized") is not True or not freeze.get("source_revision")):
        raise ValueError("Explicit final sealed-test authorization is absent")
    paths = {k: inside(REPO, v) for k, v in freeze["paths"].items()}
    bindings = {"inventory": "inventory_sha256", "configuration": "configuration_sha256", "protocol": "protocol_sha256",
                "preaccess_audit": "preaccess_audit_sha256", "preaccess_acquisition_manifest": "preaccess_acquisition_manifest_sha256"}
    for role, field in bindings.items():
        if not HASH.fullmatch(freeze.get(field, "")) or ev.sha(paths[role]) != freeze[field]:
            raise ValueError(f"Bound {role} bytes differ from the final freeze")
    sources = freeze.get("source_sha256", {})
    if not set(required_sources()) <= sources.keys():
        raise ValueError("Final freeze omits required implementation/dependency identities")
    for name, digest in sources.items():
        if not HASH.fullmatch(digest) or ev.sha(inside(REPO, name)) != digest:
            raise ValueError("Bound source bytes changed: " + name)
    checks = [(paths["model_directory"] / "weights.pt", "model_sha256"),
              (paths["model_directory"] / "manifest.json", "model_manifest_sha256"),
              (paths["training_features"] / "manifest.json", "training_manifest_sha256"),
              (paths["training_features"] / "acquisition-manifest.json", "training_acquisition_manifest_sha256")]
    for file, field in checks:
        if not HASH.fullmatch(freeze.get(field, "")) or ev.sha(file) != freeze[field]:
            raise ValueError("Bound model/training provenance bytes changed: " + field)
    if freeze["model_sha256"] != WEIGHT_HASHES["original"]:
        raise ValueError("Only the frozen original MLP is admitted")
    inventory = json.loads(paths["inventory"].read_text())
    configuration = json.loads(paths["configuration"].read_text())
    if configuration != fixed_configuration(inventory):
        raise ValueError("Configuration differs from the fixed original-only experiment")
    acquisition.authorize_split("sealed_test", paths["inventory"], path)
    return freeze, configuration, paths


def acquired_selection(data_root, freeze, configuration, *, predicting, allow_incomplete=False):
    """Read acquisition identities only; no audio or reference bytes here."""
    manifest_file = data_root / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    if manifest.get("inventory_sha256") != freeze["inventory_sha256"]:
        raise ValueError("Acquisition metadata belongs to another inventory")
    expected = {(r["recording_id"], r["role"]): r for r in configuration["selected_inventory_records"]}
    files = {}
    for row in manifest["files"]:
        if row["split"] != "sealed_test":
            continue
        key = row["recording_id"], row["role"]
        if key not in expected or key in files:
            raise ValueError("Unexpected or duplicate sealed acquisition identity")
        if predicting and row["role"] != "mic":
            raise ValueError("References must not be acquired before the complete prediction freeze")
        wanted = expected[key]
        for field, value in wanted.items():
            if row.get(field) != value:
                raise ValueError("Sealed acquired source identity differs from inventory: " + field)
        relative = f"records/sealed_test/{row['speaker']}/{row['filename']}"
        if row["path"] != relative or not HASH.fullmatch(row.get("sha256", "")):
            raise ValueError("Sealed source path/hash invalid")
        inside(data_root, relative)
        files[key] = row
    required = {k for k in expected if k[1] == "mic" or not predicting}
    if (not allow_incomplete and set(files) != required) or (allow_incomplete and not set(files) <= required):
        raise ValueError("Acquisition is not exactly the frozen complete selection")
    return manifest_file, files


def extract_input(pcm, identity, intermediate):
    """Shared production-derived TypeScript features, without any target input."""
    raw, full = intermediate / (identity + ".f32le"), intermediate / (identity + ".json")
    pcm.astype("<f4").tofile(raw)
    subprocess.run(["node", "--import", "tsx", "../../research/voice-prosody/replay.ts", "--feature-version", "v2",
        "--input-f32", str(raw), "--sample-rate", "48000", "--output", str(full)],
        cwd=REPO / "apps/capture-web", check=True, capture_output=True, text=True)
    exported = json.loads(full.read_text())
    frames = [{"end": f["referenceSample"], **f["features"]} for f in exported["frames"]]
    feature = _feature(frames, len(pcm))
    identity = {"transformed_pcm_sha256": ev.sha(raw), "typescript_export_sha256": ev.sha(full)}
    dsp = _dsp(frames, feature)
    raw.unlink(); full.unlink()
    return feature, dsp, identity


def predict(freeze_path, data_root, output):
    freeze, configuration, paths = read_freeze(freeze_path)
    acquisition_file, files = acquired_selection(data_root, freeze, configuration, predicting=True)
    train, _ = ev.read_cache_manifest(paths["training_features"])
    config, weights, model_manifest = ev.load_frozen_model(paths["model_directory"], paths["training_features"], train)
    if config.kind != "framewise-mlp":
        raise ValueError("Independent validation permits only the original framewise MLP")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("features", "predictions", "intermediate", "provenance"):
        (output / name).mkdir()
    for name, path in {"freeze.json": freeze_path, "configuration.json": paths["configuration"],
                       "protocol.md": paths["protocol"], "microphone-acquisition-manifest.json": acquisition_file}.items():
        (output / "provenance" / name).write_bytes(Path(path).read_bytes())
    os.environ.setdefault("NUMBA_CACHE_DIR", str(RESEARCH / "runs/.numba-cache"))
    manifest = {"schema": EXPERIMENT, "status": "predicting", "split": "sealed_test",
        "started_at": datetime.now(timezone.utc).isoformat(), "freeze_sha256": ev.sha(freeze_path),
        "configuration_sha256": freeze["configuration_sha256"], "source_sha256": freeze["source_sha256"],
        "microphone_manifest_sha256": ev.sha(acquisition_file), "model_sha256": freeze["model_sha256"],
        "model_manifest_sha256": freeze["model_manifest_sha256"], "model": model_manifest["model"],
        "measurement_delay_ms": 20, "methods": list(METHODS), "conditions": list(CONDITIONS),
        "references_acquired_or_read_before_prediction_freeze": False, "recordings": []}
    ev.write_json(output / "manifest.json", manifest)
    try:
        microphones = [r for r in configuration["selected_inventory_records"] if r["role"] == "mic"]
        for metadata in microphones:
            source = metadata["recording_id"]; entry = files[source, "mic"]
            file = inside(data_root, entry["path"])
            if ev.sha(file) != entry["sha256"]:
                raise ValueError("Microphone bytes changed")
            clean = read_mic(file)
            for condition in CONDITIONS:
                identity = source + "__" + condition
                pcm, corruption = transform(clean, condition, source, "sealed-test")
                feature, dsp, input_hashes = extract_input(pcm, identity, output / "intermediate")
                timing = ev.validate_feature(feature, {"sample_count": len(pcm)})
                yin, evidence = yin_predictions(pcm, feature["reference_sample"])
                learned = ev.model_predictions(feature, config, weights)
                predictions = {"dsp": dsp, "yin": ev.delayed_baseline([asdict(r) for r in yin], feature),
                    "continuity": ev.continuity_predictions(feature), "mlp_original": learned,
                    "hybrid_original": hybrid_predictions(feature, learned, evidence)}
                feature_path = output / "features" / (identity + ".npz")
                np.savez_compressed(feature_path, **feature)
                prediction_path = output / "predictions" / (identity + ".json.gz")
                ev.write_predictions(prediction_path, {"recording_id": identity, "time_contract": timing.metadata(),
                    "availability_sample": timing.availability_sample.tolist(), "predictions": predictions,
                    "yin_evidence": evidence})
                manifest["recordings"].append({"recording_id": identity, "source_recording_id": source,
                    "speaker": metadata["speaker"], "pair": metadata["pair"], "prompt": metadata["prompt"],
                    "condition": condition, "sample_count": len(pcm), "microphone_sha256": entry["sha256"],
                    "feature_path": str(feature_path.relative_to(output)), "feature_sha256": ev.sha(feature_path),
                    "prediction_path": str(prediction_path.relative_to(output)), "prediction_sha256": ev.sha(prediction_path),
                    **input_hashes, "corruption": corruption})
                ev.write_json(output / "manifest.json", manifest)
                print("Predicted " + identity, flush=True)
        read_freeze(freeze_path)
        # Concurrent reference acquisition would violate the declared ordering.
        acquired_selection(data_root, freeze, configuration, predicting=True)
        if ev.sha(acquisition_file) != manifest["microphone_manifest_sha256"]:
            raise ValueError("Acquisition metadata changed during prediction")
        (output / "intermediate").rmdir()
        manifest.update(status="predictions_frozen", predictions_frozen_at=datetime.now(timezone.utc).isoformat())
        ev.write_json(output / "prediction-freeze.json", manifest)
        ev.write_json(output / "manifest.json", manifest)
        return manifest
    except BaseException as error:
        manifest.update(status="failed", failure=repr(error)); ev.write_json(output / "manifest.json", manifest)
        raise


def validate_prediction_freeze(freeze_path, data_root, output):
    freeze, configuration, paths = read_freeze(freeze_path)
    frozen = json.loads((output / "prediction-freeze.json").read_text())
    expected = [(r, c) for r in configuration["selected_inventory_records"] if r["role"] == "mic" for c in CONDITIONS]
    if (frozen.get("schema") != EXPERIMENT or frozen.get("status") != "predictions_frozen"
            or frozen.get("freeze_sha256") != ev.sha(freeze_path) or frozen.get("methods") != list(METHODS)
            or frozen.get("source_sha256") != freeze["source_sha256"]
            or frozen.get("configuration_sha256") != freeze["configuration_sha256"]
            or frozen.get("model_sha256") != freeze["model_sha256"]
            or frozen.get("references_acquired_or_read_before_prediction_freeze") is not False
            or len(frozen.get("recordings", [])) != len(expected)):
        raise ValueError("Complete frozen original-only predictions are required before reference access")
    if ev.sha(output / "provenance/freeze.json") != ev.sha(freeze_path):
        raise ValueError("Prediction provenance freeze changed")
    mic_manifest = output / "provenance/microphone-acquisition-manifest.json"
    if ev.sha(mic_manifest) != frozen["microphone_manifest_sha256"]:
        raise ValueError("Prediction microphone manifest changed")
    microphone_rows = {(r["recording_id"], r["role"]): r for r in json.loads(mic_manifest.read_text())["files"]}
    # Verify every input feature and prediction before opening any reference.
    for record, (metadata, condition) in zip(frozen["recordings"], expected):
        source = metadata["recording_id"]
        if (record["recording_id"] != source + "__" + condition or record["source_recording_id"] != source
                or record["condition"] != condition or record["speaker"] != metadata["speaker"]
                or record["pair"] != metadata["pair"] or record["prompt"] != metadata["prompt"]
                or record["microphone_sha256"] != microphone_rows[source, "mic"]["sha256"]):
            raise ValueError("Frozen record selection or order changed")
        fp, pp = inside(output, record["feature_path"]), inside(output, record["prediction_path"])
        if ev.sha(fp) != record["feature_sha256"] or ev.sha(pp) != record["prediction_sha256"]:
            raise ValueError("Frozen features or predictions changed")
        with np.load(fp, allow_pickle=False) as stored:
            feature = {k: stored[k].copy() for k in stored.files}
        timing = ev.validate_feature(feature, record); value = ev.read_predictions(pp)
        if (value["recording_id"] != record["recording_id"] or value["time_contract"] != timing.metadata()
                or value["availability_sample"] != timing.availability_sample.tolist() or set(value["predictions"]) != set(METHODS)):
            raise ValueError("Frozen prediction identity/time contract changed")
        for rows in value["predictions"].values():
            if [r["sample"] for r in rows] != timing.reference_sample.tolist():
                raise ValueError("Frozen prediction target grid changed")
        if value["predictions"]["hybrid_original"] != hybrid_predictions(feature, value["predictions"]["mlp_original"], value["yin_evidence"]):
            raise ValueError("Frozen hybrid differs from the declared component rule")
    return freeze, configuration, frozen


def conditional_uncertainty(records):
    """Paired-prompt resampling conditional on fixed speakers; never frame resampling."""
    prompts = {pair: sorted({r["prompt"] for r in records if r["pair"] == pair}) for pair in PAIRS}
    if any(len(v) != PER_SPEAKER for v in prompts.values()):
        raise ValueError("Bootstrap requires the exact complete prompt selection")
    lookup = {(r["speaker"], r["condition"], r["prompt"]): r for r in records}
    counts = np.zeros((len(METHODS), len(PAIRS), 2, len(CONDITIONS), PER_SPEAKER, 2), np.float64)
    for m, method in enumerate(METHODS):
        for p, pair in enumerate(PAIRS):
            for sex, prefix in enumerate(("F", "M")):
                for c, condition in enumerate(CONDITIONS):
                    for j, prompt in enumerate(prompts[pair]):
                        cell = lookup[f"{prefix}{pair:02}", condition, prompt]["metrics"][method]
                        n = cell["eligible_valid_frames"]
                        counts[m, p, sex, c, j] = [n - cell["correct_frames"], n]
    if np.any(counts[..., 1] <= 0):
        return {"status": "unavailable", "reason": "At least one selected recording has no eligible valid reference; no recording omitted",
                "used_for_selection": False}
    rng = np.random.default_rng(20261022); draws = np.empty((2000, len(METHODS)))
    for i in range(2000):
        totals = []
        for pair in range(len(PAIRS)):
            indices = rng.integers(0, PER_SPEAKER, PER_SPEAKER)
            selected = np.take(counts[:, pair], indices, axis=3).sum(axis=3)
            totals.append(selected[..., 0] / selected[..., 1])
        draws[i] = np.stack(totals, axis=1).mean(axis=(1, 2, 3))
    target = METHODS.index("hybrid_original")
    return {"schema": "phenometrix-conditional-prompt-bootstrap.v1", "seed": 20261022, "draws": 2000,
        "scope": "Conditional on the four selected speakers/two pairs; not a population interval",
        "unit": "prompt within pair, both sexes and all conditions together", "used_for_selection": False,
        "joint_error_advantage_intervals": {m: {"lower": float(np.percentile(draws[:, i] - draws[:, target], 2.5)),
            "upper": float(np.percentile(draws[:, i] - draws[:, target], 97.5))}
            for i, m in enumerate(METHODS) if m != "hybrid_original"}}


def score(freeze_path, data_root, output):
    if (output / "metrics.json").exists():
        raise FileExistsError("Refuses to overwrite sealed outcomes")
    freeze, configuration, frozen = validate_prediction_freeze(freeze_path, data_root, output)
    acquisition_file, files = acquired_selection(data_root, freeze, configuration, predicting=False)
    for record in frozen["recordings"]:
        if files[record["source_recording_id"], "mic"]["sha256"] != record["microphone_sha256"]:
            raise ValueError("Microphone identity changed since prediction freeze")
    refs_cache, grouped, records = {}, {m: defaultdict(list) for m in METHODS}, []
    for record in frozen["recordings"]:
        source = record["source_recording_id"]
        if source not in refs_cache:
            entry = files[source, "ref"]; path = inside(data_root, entry["path"])
            if ev.sha(path) != entry["sha256"]:
                raise ValueError("Reference bytes changed")
            refs = ev.references_from_array(np.loadtxt(path), record["sample_count"])
            refs_cache[source] = [replace(r, eligible=r.sample >= 3072 and r.sample + 1536 <= record["sample_count"]) for r in refs]
        refs = refs_cache[source]; targets = {r.sample for r in refs}
        raw = ev.read_predictions(inside(output, record["prediction_path"]))
        metrics, outside = {}, {}
        for method in METHODS:
            all_rows = ev.prediction_objects(raw["predictions"][method]); rows = [r for r in all_rows if r.sample in targets]
            outside[method] = len(all_rows) - len(rows); metrics[method] = ev.score(refs, rows)
            grouped[method][record["speaker"], record["condition"]].append((refs, rows, record["sample_count"]))
        records.append({k: record[k] for k in ("recording_id", "source_recording_id", "speaker", "pair", "prompt", "condition", "sample_count")} |
            {"reference_sha256": files[source, "ref"]["sha256"], "metrics": metrics, "predictions_outside_reference_extent": outside})
    cells, conditions, speakers, pairs, overall = {}, {}, {}, {}, {}
    for method in METHODS:
        cells[method] = {s: {c: ev.combine_scores(grouped[method][s, c]) for c in CONDITIONS} for s in SPEAKERS}
        conditions[method] = {c: ev.macro([cells[method][s][c] for s in SPEAKERS]) for c in CONDITIONS}
        speakers[method] = {s: ev.macro(list(cells[method][s].values())) for s in SPEAKERS}
        pairs[method] = {str(p): ev.macro([cells[method][f"{sex}{p:02}"][c] for sex in ("F", "M") for c in CONDITIONS]) for p in PAIRS}
        overall[method] = ev.macro([cells[method][s][c] for s in SPEAKERS for c in CONDITIONS])
    count_fields = ("eligible_valid_frames", "reference_voiced_frames", "reference_unvoiced_frames",
        "excluded_invalid_reference", "excluded_boundary", "reference_outside_supported_bounds", "correct_frames",
        "voicing_false_positive_count", "voicing_false_negative_count", "unavailable_reference_voiced_count",
        "voicing_false_negative_or_unavailable_count", "reference_voiced_success_count", "unavailable_frames",
        "jointly_voiced_frames", "octave_error_count")
    pooled = {m: {k: sum(r["metrics"][m][k] for r in records) for k in count_fields} for m in METHODS}
    for counts in pooled.values():
        counts["joint_error_count"] = counts["eligible_valid_frames"] - counts["correct_frames"]
        counts["voiced_pitch_error_count"] = counts["jointly_voiced_frames"] - counts["reference_voiced_success_count"]
    unique_samples = {r["source_recording_id"]: r["sample_count"] for r in records}
    acquisition_totals = {"source_recording_count": len(unique_samples), "condition_record_count": len(records),
        "unique_source_sample_count": sum(unique_samples.values()), "unique_source_seconds": sum(unique_samples.values()) / 48000,
        "condition_sample_count": sum(r["sample_count"] for r in records),
        "condition_seconds": sum(r["sample_count"] for r in records) / 48000,
        "note": "Four conditions repeat each source; neither repeated samples nor frames are independent participants"}
    result = {"schema": "phenometrix-hybrid-independent-validation-metrics.v1", "split": "sealed_test",
        "freeze_sha256": ev.sha(freeze_path), "prediction_freeze_sha256": ev.sha(output / "prediction-freeze.json"),
        "measurement_delay_ms": 20, "mask": configuration["mask"], "per_record": records,
        "acquisition_totals": acquisition_totals, "pooled_counts": pooled,
        "per_speaker_condition": cells, "per_condition_equal_speaker": conditions, "per_speaker_equal_condition": speakers,
        "per_pair_equal_speaker_condition": pairs, "equal_speaker_equal_condition": overall,
        "quality_components_against_classical": ev.quality_components("hybrid_original", CLASSICAL, overall,
            {m: conditions[m]["clean"] for m in METHODS}),
        "joint_error_change_vs_own_mlp": overall["hybrid_original"]["joint_error"] - overall["mlp_original"]["joint_error"],
        "conditional_uncertainty": conditional_uncertainty(records), "production_promotion": False,
        "limitation": "Fixed 160-recording subset, four speakers/two prompt pairs; no population, calibration or clinical qualification."}
    if any(overall[m]["defined_cells"][metric] != len(SPEAKERS) * len(CONDITIONS)
           for m in METHODS for metric in ("joint_error", "reference_voiced_success_rate")):
        result["quality_components_against_classical"] = {"pass": False, "status": "incomplete_required_cell_denominators"}
    result["per_pair_joint_error_advantage"] = {str(pair): {m: pairs[m][str(pair)]["joint_error"] - pairs["hybrid_original"][str(pair)]["joint_error"]
        for m in METHODS if m != "hybrid_original"} for pair in PAIRS}
    result["pair_advantage_ranges"] = {m: {"minimum": min(result["per_pair_joint_error_advantage"][str(p)][m] for p in PAIRS),
        "maximum": max(result["per_pair_joint_error_advantage"][str(p)][m] for p in PAIRS)} for m in METHODS if m != "hybrid_original"}
    read_freeze(freeze_path)
    (output / "provenance/scoring-acquisition-manifest.json").write_bytes(acquisition_file.read_bytes())
    ev.write_json(output / "metrics.json", result)
    ev.write_json(output / "manifest.json", {**frozen, "status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
        "metrics_sha256": ev.sha(output / "metrics.json"), "scoring_acquisition_manifest_sha256": ev.sha(acquisition_file)})
    return result


def prepare_freeze(data_root, output, protocol, audit_path, extra_sources=()):
    """Public metadata/model/source freeze only; no sealed data access."""
    inventory_path = data_root / "metadata/inventory.json"
    configuration = fixed_configuration(json.loads(inventory_path.read_text()))
    audit = json.loads(audit_path.read_text())
    selected = [{k: r[k] for k in ("recording_id", "speaker", "pair", "prompt", "split")}
                for r in configuration["selected_inventory_records"] if r["role"] == "mic"]
    if (audit.get("status") != "pass" or audit.get("audio_or_reference_content_opened") is not False
            or audit.get("selected_recordings") != selected or audit.get("inventory_sha256") != ev.sha(inventory_path)
            or audit.get("acquisition_manifest_sha256") != ev.sha(data_root / "manifest.json")
            or audit.get("audit_source_sha256") != ev.sha(Path(__file__).with_name("sealed_metadata_audit.py"))):
        raise ValueError("Pre-access metadata audit is absent, stale, or selects different sources")
    model = RESEARCH / "runs/delayed-mlp-seed20261022"
    training = RESEARCH / "runs/delayed-train-features"
    train, _ = ev.read_cache_manifest(training)
    config, _, _ = ev.load_frozen_model(model, training, train)
    if config.kind != "framewise-mlp" or ev.sha(model / "weights.pt") != WEIGHT_HASHES["original"]:
        raise ValueError("Frozen original MLP identity mismatch")
    if not output.is_relative_to(RESEARCH / "runs"):
        raise ValueError("Freeze output must be in the repository's ignored research runs")
    sources = {name: ev.sha(inside(REPO, name)) for name in sorted(set(required_sources()) | set(extra_sources))}
    output.mkdir(parents=True, exist_ok=False)
    ev.write_json(output / "configuration.json", configuration)
    (output / "preaccess-acquisition-manifest.json").write_bytes((data_root / "manifest.json").read_bytes())
    paths = {"inventory": inventory_path, "configuration": output / "configuration.json", "protocol": protocol,
        "model_directory": model, "training_features": training, "preaccess_audit": audit_path,
        "preaccess_acquisition_manifest": output / "preaccess-acquisition-manifest.json"}
    freeze = {"schema": "phenometrix-prosody-freeze-v1", "experiment": EXPERIMENT, "sealed_test_authorized": True,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "source_revision_role": "base commit plus exact bound working-tree source hashes",
        "working_tree_status": subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True),
        "paths": {k: str(p.relative_to(REPO)) for k, p in paths.items()}, "source_sha256": sources,
        "inventory_sha256": ev.sha(inventory_path), "protocol_sha256": ev.sha(protocol),
        "configuration_sha256": ev.sha(output / "configuration.json"), "model_sha256": ev.sha(model / "weights.pt"),
        "model_manifest_sha256": ev.sha(model / "manifest.json"), "training_manifest_sha256": ev.sha(training / "manifest.json"),
        "training_acquisition_manifest_sha256": ev.sha(training / "acquisition-manifest.json"),
        "preaccess_acquisition_manifest_sha256": ev.sha(data_root / "manifest.json"), "preaccess_audit_sha256": ev.sha(audit_path)}
    for name in sources:
        target = output / "source-snapshot" / name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / name).read_bytes())
    ev.write_json(output / "freeze.json", freeze)
    read_freeze(output / "freeze.json")
    return {"freeze_sha256": ev.sha(output / "freeze.json"), "source_count": len(sources), "configuration": configuration}


def acquire_phase(phase, freeze_path, data_root, output, max_bytes):
    freeze, configuration, _ = read_freeze(freeze_path)
    if phase == "acquire-ref":
        _, _, frozen = validate_prediction_freeze(freeze_path, data_root, output)
        _, existing = acquired_selection(data_root, freeze, configuration, predicting=False, allow_incomplete=True)
        for record in frozen["recordings"]:
            if existing.get((record["source_recording_id"], "mic"), {}).get("sha256") != record["microphone_sha256"]:
                raise ValueError("Microphone selection changed before reference acquisition")
        roles = ("ref",)
    else:
        acquired_selection(data_root, freeze, configuration, predicting=True, allow_incomplete=True)
        roles = ("mic",)
    # The legacy acquisition tool's automatic audit is safe only after the phase guard.
    return acquisition.acquire(data_root, "sealed_test", PER_SPEAKER, roles, max_bytes, freeze_path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "acquire-mic", "predict", "acquire-ref", "score"))
    p.add_argument("--freeze", type=Path)
    p.add_argument("--data", type=Path, default=RESEARCH / "data/ptdb-tug")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--protocol", type=Path, default=RESEARCH / "HYBRID_VALIDATION_PLAN.md")
    p.add_argument("--preaccess-audit", type=Path)
    p.add_argument("--extra-source", action="append", default=[])
    p.add_argument("--max-bytes", type=int, default=2 << 30)
    a = p.parse_args(); data, output = a.data.resolve(), a.output.resolve()
    if a.phase == "freeze":
        if a.preaccess_audit is None: p.error("freeze requires --preaccess-audit")
        result = prepare_freeze(data, output, a.protocol.resolve(), a.preaccess_audit.resolve(), a.extra_source)
    else:
        if a.freeze is None: p.error("this phase requires --freeze")
        freeze = a.freeze.resolve()
        if a.phase.startswith("acquire-"): result = acquire_phase(a.phase, freeze, data, output, a.max_bytes)
        else: result = (predict if a.phase == "predict" else score)(freeze, data, output)
    print(json.dumps({"phase": a.phase, "status": result.get("status", "complete"), "output": str(output),
                      **({"freeze_sha256": result["freeze_sha256"]} if "freeze_sha256" in result else {})}))

if __name__ == "__main__":
    main()
