"""Independent post-fit audit using the established object-based metrics.score.

No fitting, selector inference, or reserved/test content access occurs here. The
receipt binds saved outputs/checkpoints and reconstructs natural scores; it does
not claim an OS-level reference-access log or checkpoint inference reproduction.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from .metrics import Prediction, Reference, score, cents
from .pilot import references_from_array

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "research/voice-prosody"
RUNS = ROOT / "runs"
SEEDS = (20261022, 20261023, 20261024)
ARMS = (("fixed-slots", "nearest"), ("fixed-slots", "acceptable-set"),
        ("shared-candidates", "nearest"), ("shared-candidates", "acceptable-set"))
CONDITIONS = ("clean", "noise10db", "first_difference", "clip01")
SPEAKERS = {"train": tuple(f"{sex}{i:02d}" for sex in "FM" for i in (2, 4, 7, 8, 9, 10)),
            "development": ("F01", "F03", "M01", "M03")}
COUNTS = ("eligible_valid_frames", "reference_voiced_frames", "reference_unvoiced_frames", "correct_frames",
          "reference_voiced_success_count", "voicing_false_positive_count", "voicing_false_negative_count",
          "unavailable_frames", "unavailable_reference_voiced_count", "octave_error_count", "voiced_pitch_error_count")
RATES = ("joint_error", "reference_voiced_success_rate", "voicing_false_positive_rate",
         "voicing_false_negative_rate", "voicing_false_negative_or_unavailable_rate", "measurement_coverage",
         "octave_error_rate_reference_voiced")
WEIGHTS = "06110469a9a643fb16806905569ebc939db2c0bdd5b3a6a1593b31789453d10d"
SYNTHETIC_IDS = {"constant-100", "constant-200", "constant-300", "step-100-200", "step-200-100",
                 *(f"glide-{d}-{v}" for d in ("up", "down") for v in (50, 100, 200)),
                 "observation-change-100", "observation-change-200", "observation-change-300"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def inside(root, path):
    result = (root / path).resolve()
    require(result.is_relative_to(root.resolve()), "Path escaped declared root")
    require(not any(p in {"sealed_test", "sealed-test", "test", "reserved"} for p in result.parts),
            "Reserved/test content is excluded")
    return result


class Audit:
    def __init__(self):
        self.files = {}
        self.comparisons = 0
        self.max_rate_delta = 0.0

    def bind(self, path, expected=None):
        path = inside(REPO, Path(path).resolve())
        actual = sha(path)
        require(expected is None or actual == expected, "Artifact changed: " + str(path.relative_to(REPO)))
        key = str(path.relative_to(REPO))
        require(key not in self.files or self.files[key] == actual, "Artifact changed during audit: " + key)
        self.files[key] = actual
        return path

    def json(self, path, expected=None):
        return json.loads(self.bind(path, expected).read_text())

    def close(self):
        for name, expected in self.files.items():
            require(sha(REPO / name) == expected, "Bound artifact changed during audit: " + name)

    def equal(self, actual, expected, label):
        self.comparisons += 1
        if actual is None or expected is None:
            require(actual is expected, label + ": definedness differs")
        elif isinstance(actual, (bool, int, np.integer)) and isinstance(expected, (bool, int, np.integer)):
            require(actual == expected, label + ": integer/boolean differs")
        else:
            delta = abs(float(actual) - float(expected))
            require(math.isfinite(delta) and delta <= 1e-12, label + ": numeric differs")
            self.max_rate_delta = max(self.max_rate_delta, delta)


def key(rep, loss, seed):
    return f"{rep}__{loss}__seed{seed}"


def load_cache(audit, split):
    require(split in SPEAKERS, "Only train/development allowed")
    folder = RUNS / f"delayed-{split}-features"
    manifest = audit.json(folder / "manifest.json")
    require(manifest.get("status") == "complete" and manifest.get("split") == split
            and manifest.get("feature_version") == "v2" and manifest.get("measurement_delay_ms") == 20
            and tuple(manifest["conditions"]) == CONDITIONS, "Cache contract differs")
    require(set(r["speaker"] for r in manifest["recordings"]) == set(SPEAKERS[split]), "Speaker split differs")
    acquisition = audit.json(folder / "acquisition-manifest.json", manifest["data_manifest_sha256"])
    refs = {r["recording_id"]: r for r in acquisition["files"] if r["role"] == "ref" and r["split"] == split}
    return folder, manifest, refs


def load_feature(audit, folder, record):
    path = audit.bind(folder / "features" / (record["recording_id"] + ".npz"), record["feature_sha256"])
    with np.load(path, allow_pickle=False) as data:
        f = {k: data[k].copy() for k in data.files}
    n = record["sample_count"]
    require(int(f["sample_rate_hz"]) == 48000 and int(f["pcm_sample_count"]) == n, "Feature support metadata differs")
    require(np.array_equal(f["reference_sample"], np.arange(2208, n + 1, 480)), "Acquisition endpoint grid differs")
    require(f["features"].shape == (len(f["reference_sample"]), 57)
            and f["candidate_hz"].shape == f["candidate_mask"].shape == (len(f["reference_sample"]), 6), "Feature shape differs")
    return f


def references(audit, metadata, record, split):
    require(metadata["split"] == split and metadata["speaker"] == record["speaker"]
            and metadata["speaker"] in SPEAKERS[split], "Reference speaker or split differs")
    path = inside(ROOT / "data/ptdb-tug", metadata["path"])
    require(f"/records/{split}/" in str(path), "Reference content path not admitted")
    audit.bind(path, metadata["sha256"])
    rows = references_from_array(np.loadtxt(path), record["sample_count"])
    return [replace(r, eligible=r.sample >= 3072 and r.sample + 1536 <= record["sample_count"]) for r in rows]


def representative_hz(feature):
    """Independent NumPy grouping for coverage and emitted-member checks."""
    hz, masks = feature["candidate_hz"], feature["candidate_mask"]
    evidence = feature["features"][:, :48].reshape(-1, 6, 8)[:, :, 7]
    out = np.full(hz.shape, np.nan)
    for row, mask in enumerate(masks):
        groups = []
        for index in np.flatnonzero(mask):
            if not groups or 1200 * np.log2(hz[row, index] / hz[row, groups[-1][0]]) > 5:
                groups.append([index])
            else:
                groups[-1].append(index)
        for slot, members in enumerate(groups):
            chosen = min(members, key=lambda i: (-float(evidence[row, i]), hz[row, i]))
            out[row, slot] = hz[row, chosen]
    return out


def training_schedule(audit, manifest):
    folder, cache, meta = load_cache(audit, "train")
    ref_cache, raw_parts, grouped_parts = {}, [], []
    for record in cache["recordings"]:
        source = record["source_recording_id"]
        if source not in ref_cache:
            ref_cache[source] = {r.sample: r for r in references(audit, meta[source], record, "train")}
        f = load_feature(audit, folder, record)
        grouped = representative_hz(f)
        raw_ok, grouped_ok = np.zeros(len(grouped), bool), np.zeros(len(grouped), bool)
        for i, target in enumerate(f["reference_sample"] - 960):
            r = ref_cache[source].get(int(target))
            if r is not None and r.valid and r.eligible and r.voiced:
                raw_ok[i] = any(abs(cents(float(h), r.f0_hz)) <= 50 for h in f["candidate_hz"][i, f["candidate_mask"][i]])
                grouped_ok[i] = any(abs(cents(float(h), r.f0_hz)) <= 50 for h in grouped[i, np.isfinite(grouped[i])])
        require(not np.any(grouped_ok & ~raw_ok), "Grouping gained non-member coverage")
        raw_parts.append(raw_ok); grouped_parts.append(grouped_ok)
    raw, grouped = np.concatenate(raw_parts), np.concatenate(grouped_parts)
    covered = grouped[raw]
    audit.equal(int(raw.sum()), manifest["scheduled_rows"], "Original-covered training schedule")
    audit.equal(int((~covered).sum()), manifest["grouped_uncovered_scheduled_rows"], "Grouped-uncovered scheduled rows")
    expected = {}
    for seed in SEEDS:
        for rep, objective in ARMS:
            skipped = 0
            for epoch in range(12):
                order = np.random.default_rng(seed + epoch).permutation(len(covered))
                if rep == "shared-candidates":
                    skipped += sum(not covered[order[start:start + 4096]].any() for start in range(0, len(order), 4096))
            expected[key(rep, objective, seed)] = (12 * math.ceil(len(covered) / 4096) - skipped, skipped)
    return expected, {"scheduled_rows": int(raw.sum()), "grouped_uncovered_rows": int((~covered).sum())}


def count_rates(c):
    ratio = lambda a, b: a / b if b else None
    n, v, u = (c[k] for k in COUNTS[:3])
    return {**c, "joint_error": ratio(n-c["correct_frames"], n),
            "reference_voiced_success_rate": ratio(c["reference_voiced_success_count"], v),
            "voicing_false_positive_rate": ratio(c["voicing_false_positive_count"], u),
            "voicing_false_negative_rate": ratio(c["voicing_false_negative_count"], v),
            "voicing_false_negative_or_unavailable_rate": ratio(c["voicing_false_negative_count"]+c["unavailable_reference_voiced_count"], v),
            "measurement_coverage": ratio(n-c["unavailable_frames"], n),
            "octave_error_rate_reference_voiced": ratio(c["octave_error_count"], v)}


def object_score(refs, predictions):
    result = score(refs, predictions)
    result["voiced_pitch_error_count"] = result["jointly_voiced_frames"] - result["reference_voiced_success_count"]
    return result


def compare_metrics(audit, actual, expected, label):
    require(set(expected) == set(COUNTS) | set(RATES), label + ": metric inventory differs")
    for name in (*COUNTS, *RATES):
        audit.equal(actual[name], expected[name], label + "/" + name)


def prediction_objects(targets, state, hz):
    require(state.shape == hz.shape == targets.shape and np.isin(state, [0, 1, 2]).all(), "Prediction array contract differs")
    require(np.isfinite(hz[state == 1]).all() and (hz[state == 1] > 0).all() and np.isnan(hz[state != 1]).all(), "Prediction absence contract differs")
    return [Prediction(int(t), ("unvoiced", "voiced", "unavailable")[int(s)], float(h) if s == 1 else None)
            for t, s, h in zip(targets, state, hz)]


def natural(audit, run, manifest, reported, methods):
    folder, cache, meta = load_cache(audit, "development")
    arrays = {}
    for method in methods:
        p = run / "predictions" / (method + ".npz")
        audit.bind(p, manifest["prediction_sha256"][str(p.relative_to(run))])
        with np.load(p, allow_pickle=False) as z:
            require(set(z.files) == {"state", "hz"}, "Unexpected prediction fields")
            arrays[method] = (z["state"].copy(), z["hz"].copy())
    require(set(reported) == set(methods), "Natural method inventory differs")
    freeze_path = RUNS / "hybrid-natural-v1/prediction-freeze.json"
    old = audit.json(freeze_path, manifest["input_sha256"][str(freeze_path)])
    old_rows = {r["recording_id"]: r for r in old["recordings"]}
    buckets = defaultdict(list)
    offset = 0
    ref_cache = {}
    fixed_voicing_frames = 0
    for record in cache["recordings"]:
        f = load_feature(audit, folder, record)
        targets = f["reference_sample"] - 960; end = offset + len(targets)
        sl = slice(offset, end); source = record["source_recording_id"]
        if source not in ref_cache:
            ref_cache[source] = references(audit, meta[source], record, "development")
        refs = ref_cache[source]; valid_targets = {r.sample for r in refs}
        binding = old_rows[record["recording_id"]]
        p = inside(RUNS / "hybrid-natural-v1", binding["hybrid_prediction_path"])
        audit.bind(p, binding["hybrid_prediction_sha256"])
        cached = json.loads(gzip.decompress(p.read_bytes()))["predictions"]
        probability = np.array([r["voicing_probability"] for r in cached["mlp_original"]])
        require(np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all(), "Invalid original probability")
        for method, old_method in (("original_mlp", "mlp_original"), ("original_hybrid", "hybrid_original")):
            rows = cached[old_method]
            require([r["sample"] for r in rows] == targets.tolist(), "Original baseline target grid differs")
            original_state = np.array([{"unvoiced": 0, "voiced": 1, "unavailable": 2}[r["state"]] for r in rows])
            original_hz = np.array([np.nan if r["f0_hz"] is None else r["f0_hz"] for r in rows])
            require(np.array_equal(arrays[method][0][sl], original_state)
                    and np.array_equal(arrays[method][1][sl], original_hz, equal_nan=True), "Original baseline output changed")
        grouped = representative_hz(f)
        per_method = {}
        for method in methods:
            state, hz = arrays[method][0][sl], arrays[method][1][sl]
            require(np.array_equal(state == 0, probability < .5), "Original voicing changed: " + method)
            if method not in {"original_mlp", "original_hybrid"}:
                require(np.array_equal(state, arrays["original_mlp"][0][sl]), "Selector changed original availability")
                candidates = grouped if method.startswith("shared-candidates") else f["candidate_hz"]
                require(np.all(np.any(candidates[state == 1] == hz[state == 1, None], axis=1)), "Selector emitted a non-member frequency")
            predictions = [p for p in prediction_objects(targets, state, hz) if p.sample in valid_targets]
            result = object_score(refs, predictions)
            expected = next(r["metrics"] for r in reported[method]["per_record"] if r["recording_id"] == record["recording_id"])
            compare_metrics(audit, result, expected, method + "/" + record["recording_id"])
            per_method[method] = predictions
        buckets[(record["speaker"], record["condition"])].append((refs, per_method, record["sample_count"]))
        fixed_voicing_frames += len(targets); offset = end
    require(all(len(state) == offset for state, hz in arrays.values()), "Trailing or missing prediction rows")
    results = {m: {} for m in methods}
    for (speaker, condition), pieces in sorted(buckets.items()):
        cell = speaker + "/" + condition
        refs, predictions, base = [], {m: [] for m in methods}, 0
        for local_refs, local_predictions, count in pieces:
            refs.extend(replace(r, sample=r.sample + base) for r in local_refs)
            for m in methods:
                predictions[m].extend(replace(p, sample=p.sample + base) for p in local_predictions[m])
            base += count + 48000
        for m in methods:
            value = object_score(refs, predictions[m]); results[m][cell] = value
            compare_metrics(audit, value, reported[m]["cells"][cell], m + "/" + cell)
    for m in methods:
        cells = results[m]
        require(set(cells) == {s + "/" + c for s in SPEAKERS["development"] for c in CONDITIONS}, "Cell inventory differs")
        for metric in RATES:
            audit.equal(np.mean([v[metric] for v in cells.values()]), reported[m]["macro"][metric], m + "/macro/" + metric)
            for c in CONDITIONS:
                audit.equal(np.mean([v[metric] for k, v in cells.items() if k.endswith("/" + c)]), reported[m]["per_condition"][c][metric], m + "/condition/" + c + "/" + metric)
            for s in SPEAKERS["development"]:
                audit.equal(np.mean([v[metric] for k, v in cells.items() if k.startswith(s + "/")]), reported[m]["per_speaker"][s][metric], m + "/speaker/" + s + "/" + metric)
        pooled = count_rates({k: sum(v[k] for v in cells.values()) for k in COUNTS})
        compare_metrics(audit, pooled, reported[m]["pooled"], m + "/pooled")
    return results, {"methods": len(methods), "cells_per_method": len(buckets), "condition_recordings": len(cache["recordings"]),
                     "acquired_rows_per_method": offset, "frozen_voicing_rows_checked_per_method": fixed_voicing_frames,
                     "eligible_valid_frames_per_method": sum(v["eligible_valid_frames"] for v in results[methods[0]].values())}


def check_checkpoints(audit, run, manifest, expected_updates):
    for seed in SEEDS:
        for rep, objective in ARMS:
            name = key(rep, objective, seed); fit = manifest["fits"][name]
            require((fit["seed"], fit["representation"], fit["objective"]) == (seed, rep, objective), "Fit identity differs")
            require(len(fit["epoch_losses"]) == 12 and all(math.isfinite(x) for x in fit["epoch_losses"]), "Invalid epoch losses")
            audit.equal(fit["optimizer_updates"], expected_updates[name][0], name + "/updates")
            audit.equal(fit["skipped_batches"], expected_updates[name][1], name + "/skips")
            p = audit.bind(run / "checkpoints" / (name + ".pt"), fit["weights_sha256"])
            saved = torch.load(p, map_location="cpu", weights_only=True)
            require(saved["config"] == {"representation": rep, "loss": objective, "hidden_size": 32}, "Checkpoint configuration differs")
            if rep == "fixed-slots":
                layers = {"projection": (32, 57), "frame_hidden": (32, 32), "candidate_head": (6, 32)}
                buffers = {"feature_mean": (57,), "feature_scale": (57,)}
            else:
                layers = {"candidate_encoder": (32, 8), "shared_hidden": (32, 73), "shared_head": (1, 32)}
                buffers = {"candidate_mean": (8,), "candidate_scale": (8,), "global_mean": (9,), "global_scale": (9,)}
            shapes = {"scaling_fitted": ()} | buffers
            for layer, shape in layers.items():
                shapes[layer + ".weight"] = shape; shapes[layer + ".bias"] = (shape[0],)
            state = saved["state_dict"]
            require(set(state) == set(shapes), "Checkpoint tensor inventory differs")
            for n, shape in shapes.items():
                require(tuple(state[n].shape) == shape, "Checkpoint tensor shape differs")
                if n == "scaling_fitted":
                    require(state[n].dtype == torch.bool and bool(state[n]), "Scaling not fitted")
                else:
                    require(state[n].dtype == torch.float32 and bool(torch.isfinite(state[n]).all()), "Checkpoint dtype/value differs")
                    if n.endswith("scale"):
                        require(bool((state[n] > 0).all()), "Invalid scaling buffer")
            params = sum(state[n].numel() for n in state if n.endswith((".weight", ".bias")))
            audit.equal(params, fit["parameter_count"], name + "/parameters")
            audit.equal(params, 3110 if rep == "fixed-slots" else 2689, name + "/declared parameters")


def synthetic(audit, run, manifest, reported, methods):
    old = audit.json(RUNS / "delayed-known-truth/prediction-freeze.json")
    support = audit.json(RUNS / "support-diagnostic-v1/manifest.json", old["original_support_manifest_sha256"])
    require(len(old["recordings"]) == 14 and {r["id"] for r in old["recordings"]} == SYNTHETIC_IDS, "Synthetic inventory differs")
    require(set(reported["scenarios"]) == SYNTHETIC_IDS, "Reported synthetic inventory differs")
    scenarios = {r["id"]: r for r in support["scenarios"]}
    scenarios.update({f"observation-change-{h}": {"kind": "observation-change", "initialHz": h} for h in (100, 200, 300)})
    passes = {key("shared-candidates", "acceptable-set", s): True for s in SEEDS}
    stationary = {}
    for row in old["recordings"]:
        name = row["id"]; scenario = scenarios[name]
        fp = audit.bind(RUNS / "delayed-known-truth/features" / (name + ".npz"), row["feature_sha256"])
        with np.load(fp, allow_pickle=False) as z:
            target = z["reference_sample"] - 960; count = int(z["pcm_sample_count"])
        old_p = audit.bind(RUNS / "delayed-known-truth/predictions" / (name + ".json.gz"), row["prediction_sha256"])
        old_voice = np.array([r["voicing_probability"] for r in json.loads(gzip.decompress(old_p.read_bytes()))["predictions"]["mlp"]])
        file = run / "predictions" / ("synthetic-" + name + ".npz")
        audit.bind(file, manifest["prediction_sha256"][str(file.relative_to(run))])
        with np.load(file, allow_pickle=False) as z:
            require(set(z.files) == {m + suffix for m in methods for suffix in ("__state", "__hz")}, "Synthetic method inventory differs")
            for method in methods:
                state, hz = z[method + "__state"], z[method + "__hz"]
                require(np.array_equal(state == 0, old_voice < .5), "Synthetic frozen voicing changed")
                preds = prediction_objects(target, state, hz)
                refs = []
                for sample in target:
                    t = float(sample) / 48000
                    if scenario["kind"] in {"constant", "observation-change"}:
                        truth = scenario["initialHz"]
                    elif scenario["kind"] == "step":
                        truth = scenario["initialHz"] if t < scenario["changeStartSeconds"] else scenario["finalHz"]
                    else:
                        elapsed = min(scenario["changeEndSeconds"]-scenario["changeStartSeconds"], max(0, t-scenario["changeStartSeconds"]))
                        truth = scenario["initialHz"] + scenario["slopeHzPerSecond"] * elapsed
                    refs.append(Reference(int(sample), float(truth), True, True, sample >= 3072 and sample + 1536 <= count))
                value = score(refs, preds); original = reported["scenarios"][name]["methods"][method]["all"]
                audit.equal(value["eligible_valid_frames"], original["frames"], name + "/frames")
                audit.equal(value["reference_voiced_success_rate"], original["success_within_50c_all_frames"], name + "/success")
                audit.equal(value["jointly_voiced_frames"], original["voiced_outputs"], name + "/voiced")
                if scenario["kind"] == "constant":
                    stationary.setdefault(name, {})[method] = value["reference_voiced_success_rate"]
                    if method in passes:
                        passes[method] &= value["reference_voiced_success_rate"] >= .95
    require(set(stationary) == {"constant-100", "constant-200", "constant-300"}, "Missing stationary scenario")
    require(passes == reported["stationary_candidate_pass_by_seed"] and all(passes.values()) == reported["stationary_candidate_pass"], "Stationary decision differs")
    return {"scenarios": 14, "methods_per_scenario": len(methods), "stationary_success": stationary,
            "stationary_candidate_pass_by_seed": passes, "stationary_candidate_pass": all(passes.values())}


def audit_run(run, prerequisites, grouping_failure, output):
    run = inside(RUNS, run); output = Path(output).resolve()
    require(not output.exists(), "Refuses to overwrite audit evidence")
    audit = Audit(); manifest = audit.json(run / "manifest.json")
    require(manifest.get("status") == "complete" and manifest.get("schema") == "phenometrix-selector-factorial-v1", "Requires completed selector run")
    require(manifest["seeds"] == list(SEEDS) and manifest["arms"] == [list(a) for a in ARMS]
            and manifest["epochs"] == 12 and manifest["batch_size"] == 4096 and manifest["learning_rate"] == .001
            and manifest["measurement_delay_ms"] == 20 and manifest["device"] == "mps", "Run configuration differs")
    names = [key(rep, loss, seed) for seed in SEEDS for rep, loss in ARMS]
    methods = ["original_mlp", "original_hybrid", *names]
    require(set(manifest["fits"]) == set(names), "Expected exactly 12 final fits")
    require(manifest["frozen_original_weights_sha256"] == WEIGHTS and manifest["reserved_content_read"] is False, "Original model or scope differs")
    require(manifest["fit_elapsed_seconds"] <= 1800 and manifest["peak_observed_mps_allocated_bytes"] <= 8*1024**3, "Fit resource cap exceeded")
    prerequisite = audit.json(prerequisites, manifest["prerequisites_sha256"])
    require(prerequisite["status"] == "complete" and prerequisite["pass"] is True
            and prerequisite["source_sha256"] == manifest["source_sha256"], "Prerequisite identity differs")
    for name, expected in manifest["source_sha256"].items():
        audit.bind(REPO / name, expected); audit.bind(run / "source-snapshot" / name, expected)
    for name, expected in manifest["input_sha256"].items():
        audit.bind(name, expected)
    cf = audit.json(run / "checkpoint-freeze.json"); pf = audit.json(run / "prediction-freeze.json")
    require(cf["fits"] == pf["fits"] == manifest["fits"] and cf["source_sha256"] == pf["source_sha256"] == manifest["source_sha256"], "Freeze fit/source identities differ")
    require(pf["prediction_sha256"] == manifest["prediction_sha256"], "Prediction identities changed after freeze")
    require(cf["development_references_read"] is False and pf["development_references_read"] is False
            and manifest["development_references_read"] is True, "Declared chronology differs")
    times = [datetime.fromisoformat(manifest[k]) for k in ("started_at", "checkpoints_frozen_at", "predictions_frozen_at", "completed_at")]
    require(times == sorted(times) and cf["checkpoints_frozen_at"] == manifest["checkpoints_frozen_at"]
            and pf["predictions_frozen_at"] == manifest["predictions_frozen_at"], "Freeze timestamps out of order")
    require(not any("/records/development/" in p for p in pf["input_sha256"]), "Development reference bytes bound before freeze")
    source = (run / "source-snapshot/research/voice-prosody/phenometrix_prosody/selector_experiment.py").read_text()
    require(source.index('save(output/"prediction-freeze.json",manifest)') < source.index('dev_labels=load_labels(dev)'), "Declared code ordering differs")
    expected_predictions = {"predictions/" + n + ".npz" for n in methods} | {"predictions/synthetic-" + n + ".npz" for n in SYNTHETIC_IDS}
    require(set(manifest["prediction_sha256"]) == expected_predictions
            and {str(p.relative_to(run)) for p in (run / "predictions").glob("*.npz")} == expected_predictions, "Prediction inventory differs")
    require({p.name for p in (run / "checkpoints").glob("*.pt")} == {n + ".pt" for n in names}, "Checkpoint inventory differs")
    metrics = audit.json(run / "metrics.json", manifest["metrics_sha256"])
    updates, schedule = training_schedule(audit, manifest)
    check_checkpoints(audit, run, manifest, updates)
    natural_cells, natural_receipt = natural(audit, run, manifest, metrics["natural"], methods)
    synthetic_receipt = synthetic(audit, run, manifest, metrics["synthetic"], names)
    mean = lambda method, metric: float(np.mean([v[metric] for v in natural_cells[method].values()]))
    candidate_names = [key("shared-candidates", "acceptable-set", s) for s in SEEDS]
    candidate_error = float(np.mean([mean(n, "joint_error") for n in candidate_names]))
    improvements = {b: mean(b, "joint_error") - candidate_error for b in ("original_mlp", "original_hybrid")}
    excess = {c: float(np.mean([v["joint_error"] for n in candidate_names for k, v in natural_cells[n].items() if k.endswith("/" + c)]))
                 - float(np.mean([v["joint_error"] for k, v in natural_cells["original_mlp"].items() if k.endswith("/" + c)])) for c in CONDITIONS}
    checks = {"mean_improves_both_by_half_point": all(v >= .005 for v in improvements.values()),
              "each_seed_improves_original_mlp": all(mean(n, "joint_error") < mean("original_mlp", "joint_error") for n in candidate_names),
              "no_condition_worsens_over_one_point": all(v <= .01 for v in excess.values()),
              "voiced_success_not_lower": np.mean([mean(n, "reference_voiced_success_rate") for n in candidate_names]) >= mean("original_mlp", "reference_voiced_success_rate")}
    checks = {k: bool(v) for k, v in checks.items()}
    require(checks == metrics["decision"]["checks"] and all(checks.values()) == metrics["decision"]["pass"], "Natural decision differs")
    require((all(checks.values()) and synthetic_receipt["stationary_candidate_pass"]) == metrics["decision"]["promising_for_next_study"], "Combined decision differs")
    failure = audit.json(grouping_failure)
    require(failure.get("status") == "failed" and failure.get("stage") == "grouping prerequisite", "Preflight failure record differs")
    require(not list(Path(grouping_failure).parent.glob("checkpoints/*")) and not (Path(grouping_failure).parent / "manifest.json").exists(), "Failed preflight unexpectedly contains fit evidence")
    audit.bind(Path(__file__)); audit.bind(Path(__file__).with_name("metrics.py")); audit.bind(Path(__file__).with_name("pilot.py"))
    audit.close()
    receipt = {"schema": "phenometrix-selector-factorial-independent-audit.v1", "status": "passed",
               "audited_at": datetime.now(timezone.utc).isoformat(), "run_path": str(run.relative_to(REPO)),
               "run_manifest_sha256": sha(run / "manifest.json"), "metrics_sha256": sha(run / "metrics.json"),
               "prediction_freeze_sha256": sha(run / "prediction-freeze.json"), "checkpoint_freeze_sha256": sha(run / "checkpoint-freeze.json"),
               "audit_source_sha256": sha(Path(__file__)), "object_scorer_sha256": sha(Path(__file__).with_name("metrics.py")),
               "bound_artifacts": len(audit.files), "numeric_comparisons": audit.comparisons, "max_numeric_difference": audit.max_rate_delta,
               "fit_count": 12, "epochs_each": 12, "fit_counts_recomputed": True, "schedule": schedule,
               "natural": natural_receipt, "synthetic": synthetic_receipt, "natural_checks": checks,
               "natural_gate_pass": all(checks.values()), "promising_for_next_study": all(checks.values()) and synthetic_receipt["stationary_candidate_pass"],
               "chronology_evidence": "Source inspection, declared timestamps and freeze manifests; not an OS access log.",
               "earlier_preflight": {"failure_record_sha256": sha(grouping_failure), "record_path": str(Path(grouping_failure).relative_to(REPO)),
                   "scope": "Assistant-recorded grouping-v1 source-mismatch rejection; no fitting artifacts in that directory. Original rejection was not independently reconstructed."},
               "scope": "Saved output scoring and byte/tensor identity; no refit, no independent checkpoint inference reproduction, no reserved/test content, no promotion claim."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": receipt["status"], "receipt": str(output.relative_to(REPO)), "comparisons": audit.comparisons,
                      "natural_gate_pass": receipt["natural_gate_pass"], "stationary_gate_pass": synthetic_receipt["stationary_candidate_pass"]}))
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--prerequisites", type=Path, required=True)
    p.add_argument("--grouping-failure", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    audit_run(args.run.resolve(), args.prerequisites.resolve(), args.grouping_failure.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
