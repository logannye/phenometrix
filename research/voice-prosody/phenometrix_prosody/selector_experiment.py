"""Frozen, local-only step-1 factorial experiment; never opens reserved data."""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from . import evaluate as ev
from .pilot import REPO, references_from_array
from .measurement_alignment import align_feature_timing
from .selector_model import (SelectorConfig, PitchSelector, prepare_selector_batch,
                             build_pitch_targets, selector_loss, predict_pitch_hz)

ROOT = REPO / "research/voice-prosody"
RUNS = ROOT / "runs"
PLAN = ROOT / "SELECTOR_EXPERIMENT_PLAN.md"
SEEDS = (20261022, 20261023, 20261024)
ARMS = (("fixed-slots", "nearest"), ("fixed-slots", "acceptable-set"),
        ("shared-candidates", "nearest"), ("shared-candidates", "acceptable-set"))
CANDIDATE = "shared-candidates__acceptable-set"
CONDITIONS = ("clean", "noise10db", "first_difference", "clip01")
COUNTS = ("eligible_valid_frames", "reference_voiced_frames", "reference_unvoiced_frames",
          "correct_frames", "reference_voiced_success_count", "voicing_false_positive_count",
          "voicing_false_negative_count", "unavailable_frames", "unavailable_reference_voiced_count",
          "octave_error_count", "voiced_pitch_error_count")
SOURCES = ("selector_experiment.py", "selector_model.py", "evaluate.py", "model.py",
           "measurement_alignment.py", "pilot.py", "metrics.py", "evaluate_synthetic.py", "selector_headroom.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sources():
    return {str(p.relative_to(REPO)): ev.sha(p) for p in
            [PLAN, *(Path(__file__).with_name(n) for n in SOURCES)]}


def assert_sources(expected):
    if sources() != expected:
        raise ValueError("Experiment source/plan changed after freeze")


def load_inputs(split):
    if split not in {"train", "development"}:
        raise ValueError("Only train/development are admitted")
    folder = RUNS / f"delayed-{split}-features"
    manifest, acquisition = ev.read_cache_manifest(folder)
    assignment = json.loads((ROOT / "config/ptdb-split.json").read_text())
    expected = {f"{sex}{pair:02d}" for pair in assignment["assignment"][split] for sex in "FM"}
    if (manifest["split"] != split or set(r["speaker"] for r in manifest["recordings"]) != expected
            or tuple(manifest["conditions"]) != CONDITIONS):
        raise ValueError("Cache does not match the fixed train/development assignment")
    pieces, records, offset = [], [], 0
    hashes = {str(folder / n): ev.sha(folder / n) for n in ("manifest.json", "acquisition-manifest.json")}
    hashes[str(ROOT / "config/ptdb-split.json")] = ev.sha(ROOT / "config/ptdb-split.json")
    for record in manifest["recordings"]:
        path = folder / "features" / (record["recording_id"] + ".npz")
        if ev.sha(path) != record["feature_sha256"]:
            raise ValueError("Input feature identity changed")
        hashes[str(path)] = record["feature_sha256"]
        with np.load(path, allow_pickle=False) as z:
            f = {k: z[k].copy() for k in z.files}
        timing = ev.validate_feature(f, record)
        size = len(f["features"])
        pieces.append(f)
        records.append({**record, "start": offset, "stop": offset + size,
                        "targets": timing.reference_sample, "eligible": timing.common_eligible})
        offset += size
    raw = {k: np.concatenate([f[k] for f in pieces]) for k in
           ("features", "candidate_hz", "candidate_mask")}
    return {"split": split, "folder": folder, "manifest": manifest, "acquisition": acquisition,
            "records": records, "raw": raw, "hashes": hashes}


def load_labels(data):
    """Reference reads are isolated from fitting/prediction input loading."""
    metadata = {r["recording_id"]: r for r in data["acquisition"]["files"]
                if r["role"] == "ref" and r["split"] == data["split"]}
    size = len(data["raw"]["features"])
    hz, voiced, valid = np.full(size, np.nan), np.zeros(size, bool), np.zeros(size, bool)
    loaded, hashes, excluded = {}, {}, {}
    data_root = (ROOT / "data/ptdb-tug").resolve()
    for record in data["records"]:
        source = record["source_recording_id"]
        if source not in loaded:
            meta = metadata[source]
            path = (data_root / meta["path"]).resolve()
            if not path.is_relative_to(data_root) or ev.sha(path) != meta["sha256"]:
                raise ValueError("Reference path or identity changed")
            refs = references_from_array(np.loadtxt(path), record["sample_count"])
            loaded[source] = {r.sample: r for r in refs}
            hashes[str(path)] = meta["sha256"]
        refs = loaded[source]
        targets = set(record["targets"].tolist())
        if any(r.valid and r.sample >= 3072 and r.sample + 1536 <= record["sample_count"]
               and r.sample not in targets for r in refs.values()):
            raise ValueError("Eligible reference target absent from input grid")
        excluded[record["recording_id"]] = {
            "invalid_reference": sum(not r.valid for r in refs.values()),
            "boundary_reference": sum(r.valid and not (r.sample >= 3072 and r.sample + 1536 <= record["sample_count"])
                                      for r in refs.values()),
            "predictions_outside_reference_extent": len(targets - set(refs))}
        for j, target in enumerate(record["targets"]):
            r = refs.get(int(target)); i = record["start"] + j
            if r is not None:
                valid[i] = r.valid and bool(record["eligible"][j])
                voiced[i] = r.voiced
                if r.f0_hz is not None:
                    hz[i] = r.f0_hz
    return {"hz": hz, "voiced": voiced, "valid": valid, "hashes": hashes, "excluded": excluded}


def batches(data):
    r = data["raw"]
    return {rep: prepare_selector_batch(r["features"], r["candidate_hz"], r["candidate_mask"], representation=rep)
            for rep in ("fixed-slots", "shared-candidates")}


def targets(batch, labels):
    return build_pitch_targets(batch, labels["hz"], labels["voiced"], labels["valid"])


def coverage(data, labels, prepared):
    covered = {rep: targets(b, labels).covered.cpu().numpy() for rep, b in prepared.items()}
    groups = defaultdict(lambda: {"voiced": 0, "raw": 0, "grouped": 0, "lost": 0, "gained": 0})
    for r in data["records"]:
        sl = slice(r["start"], r["stop"]); g = groups[(r["speaker"], r["condition"])]
        ref = labels["valid"][sl] & labels["voiced"][sl]
        a, b = covered["fixed-slots"][sl], covered["shared-candidates"][sl]
        for key, value in {"voiced": ref.sum(), "raw": a.sum(), "grouped": b.sum(),
                           "lost": (a & ~b).sum(), "gained": (b & ~a).sum()}.items():
            g[key] += int(value)
    cells = [{"speaker": s, "condition": c, **g,
              "raw_coverage": g["raw"] / g["voiced"], "grouped_coverage": g["grouped"] / g["voiced"],
              "lost_fraction": g["lost"] / g["voiced"], "gained_fraction": g["gained"] / g["voiced"]}
             for (s, c), g in sorted(groups.items()) if g["voiced"]]
    if len(cells) != len(groups):
        raise ValueError("Undefined coverage denominator")
    # Gate actual lost frames, not net loss concealed by gains.
    if any(c["gained"] for c in cells):
        raise ValueError("Grouping cannot gain coverage when emitting an original member")
    lost = float(np.mean([c["lost_fraction"] for c in cells]))
    return {"cells": cells, "raw_coverage": float(np.mean([c["raw_coverage"] for c in cells])),
            "grouped_coverage": float(np.mean([c["grouped_coverage"] for c in cells])),
            "lost_fraction": lost, "gained_fraction": float(np.mean([c["gained_fraction"] for c in cells])),
            "pass": lost <= .005}


def rate_counts(count):
    out = dict(count)
    def ratio(a, b):
        return a / b if b else None
    n, nv, nu = (out[k] for k in COUNTS[:3])
    out.update(joint_error=ratio(n-out["correct_frames"], n),
        reference_voiced_success_rate=ratio(out["reference_voiced_success_count"], nv),
        voicing_false_positive_rate=ratio(out["voicing_false_positive_count"], nu),
        voicing_false_negative_rate=ratio(out["voicing_false_negative_count"], nv),
        voicing_false_negative_or_unavailable_rate=ratio(out["voicing_false_negative_count"]+out["unavailable_reference_voiced_count"], nv),
        measurement_coverage=ratio(n-out["unavailable_frames"], n),
        octave_error_rate_reference_voiced=ratio(out["octave_error_count"], nv))
    return out


def score_arrays(labels, state, hz):
    valid, voiced, truth = labels["valid"], labels["voiced"], labels["hz"]
    if len(state) != len(valid) or len(hz) != len(valid):
        raise ValueError("Prediction length mismatch")
    if np.any(~np.isin(state, [0, 1, 2])) or np.any((state == 1) & (~np.isfinite(hz) | (hz <= 0))):
        raise ValueError("Invalid state or voiced frequency")
    if np.any((state != 1) & np.isfinite(hz)):
        raise ValueError("Nonvoiced frequency must be absent")
    vv, uv = valid & voiced, valid & ~voiced
    both = vv & (state == 1)
    error = np.full(len(state), np.nan)
    error[both] = 1200*np.log2(hz[both]/truth[both])
    success = both & (np.abs(error) <= 50)
    octaves = np.zeros(len(state), bool)
    nearest = np.rint(error[both] / 1200)
    octaves[both] = (nearest != 0) & (np.abs(error[both]-1200*nearest) <= 50)
    vals = (valid.sum(), vv.sum(), uv.sum(), (success | (uv & (state == 0))).sum(),
            success.sum(), (uv & (state == 1)).sum(), (vv & (state == 0)).sum(),
            (valid & (state == 2)).sum(), (vv & (state == 2)).sum(), octaves.sum(),
            (both & ~success).sum())
    return rate_counts(dict(zip(COUNTS, map(int, vals))))


def macro(cells):
    keys = [k for k in cells[0] if k not in COUNTS]
    if any(c[k] is None for c in cells for k in keys):
        raise ValueError("Undefined primary cell denominator")
    return {"cell_count": len(cells), **{k: float(np.mean([c[k] for c in cells])) for k in keys}}


def aggregate(data, labels, predictions):
    output = {}
    for method, (state, hz) in predictions.items():
        grouped = defaultdict(lambda: dict.fromkeys(COUNTS, 0))
        records = []
        for r in data["records"]:
            sl = slice(r["start"], r["stop"])
            lab = {k: labels[k][sl] for k in ("hz", "voiced", "valid")}
            result = score_arrays(lab, state[sl], hz[sl])
            cell = grouped[(r["speaker"], r["condition"])]
            for k in COUNTS:
                cell[k] += result[k]
            records.append({"recording_id": r["recording_id"], "metrics": result})
        cells = {f"{s}/{c}": rate_counts(count) for (s, c), count in sorted(grouped.items())}
        output[method] = {"macro": macro(list(cells.values())), "cells": cells,
                         "per_condition": {c: macro([v for k, v in cells.items() if k.endswith('/'+c)]) for c in CONDITIONS},
                         "per_speaker": {s: macro([v for k, v in cells.items() if k.startswith(s+'/')])
                                         for s in sorted({r["speaker"] for r in data["records"]})},
                         "pooled": rate_counts({k: sum(c[k] for c in cells.values()) for k in COUNTS}),
                         "per_record": records}
    return output


def model_key(rep, loss, seed):
    return f"{rep}__{loss}__seed{seed}"


def natural_decision(result):
    names = [model_key("shared-candidates", "acceptable-set", seed) for seed in SEEDS]
    rows = [result[n]["macro"] for n in names]
    mean_error = float(np.mean([r["joint_error"] for r in rows]))
    improvements = {b: result[b]["macro"]["joint_error"]-mean_error for b in ("original_mlp", "original_hybrid")}
    conditions = {c: float(np.mean([result[n]["per_condition"][c]["joint_error"] for n in names]))
                      -result["original_mlp"]["per_condition"][c]["joint_error"] for c in CONDITIONS}
    success = float(np.mean([r["reference_voiced_success_rate"] for r in rows]))
    checks = {"mean_improves_both_by_half_point": all(v >= .005 for v in improvements.values()),
              "each_seed_improves_original_mlp": all(r["joint_error"] < result["original_mlp"]["macro"]["joint_error"] for r in rows),
              "no_condition_worsens_over_one_point": all(v <= .01 for v in conditions.values()),
              "voiced_success_not_lower": success >= result["original_mlp"]["macro"]["reference_voiced_success_rate"]}
    control_error = float(np.mean([result[model_key("fixed-slots", "nearest", s)]["macro"]["joint_error"] for s in SEEDS]))
    return {"predesignated_candidate": CANDIDATE, "pass": all(checks.values()), "checks": checks,
            "mean_joint_error": mean_error, "mean_voiced_success": success,
            "joint_error_improvement_over_frozen": improvements, "condition_error_excess_over_mlp": conditions,
            "improvement_over_new_fixed_slot_nearest_mean": control_error-mean_error,
            "interpretation": "exploratory development only; no fresh holdout, calibration or production promotion"}


def validate_headroom(path):
    from .selector_headroom import PINS, WEIGHTS, SPEAKERS
    audit = json.loads(path.read_text())
    if (audit.get("schema") != "phenometrix-selector-headroom.v1" or audit.get("status") != "complete"
            or audit.get("input_manifest_sha256") != PINS or audit.get("model_weights_sha256") != WEIGHTS):
        raise ValueError("Headroom audit lacks the fixed completed input/model contract")
    for name, expected in PINS.items():
        if ev.sha(RUNS/name) != expected:
            raise ValueError("Headroom input manifest changed")
    if ev.sha(RUNS/"delayed-mlp-seed20261022/weights.pt") != WEIGHTS:
        raise ValueError("Headroom original weights changed")
    if not audit.get("source_sha256"):
        raise ValueError("Headroom source bindings missing")
    for name, expected in audit["source_sha256"].items():
        source = (REPO/name).resolve()
        if not source.is_relative_to(REPO) or ev.sha(source) != expected:
            raise ValueError("Headroom audit source changed")
    for field in ("input_ledger", "raw_record_counts"):
        artifact = (REPO/audit[field]["path"]).resolve()
        if not artifact.is_relative_to(RUNS) or ev.sha(artifact) != audit[field]["sha256"]:
            raise ValueError("Headroom provenance artifact changed")
    cells = audit["splits"]["development"]["per_speaker_condition"]
    if set(cells) != set(SPEAKERS["development"]) or any(set(v) != set(CONDITIONS) for v in cells.values()):
        raise ValueError("Headroom requires all 16 development cells")
    gaps=[]
    for conditions in cells.values():
        for cell in conditions.values():
            c=cell["counts"]
            if c["eligible_valid_frames"]<=0:raise ValueError("Undefined headroom denominator")
            gaps.append((c["six_candidates_frozen_voicing_correct_frames"]-c["mlp_correct_frames"])/c["eligible_valid_frames"])
    observed=float(np.mean(gaps)); gate=audit["prefit_gate"]
    if (gate.get("split")!="development" or gate.get("minimum_absolute_reduction")!=.01
            or abs(gate.get("observed_absolute_reduction",float('inf'))-observed)>1e-12
            or gate.get("pass") is not (observed>=.01)):
        raise ValueError("Headroom gate differs from fixed rule or underlying counts")
    return gate


def grouping_audit(output):
    output.mkdir(parents=True, exist_ok=False)
    frozen = sources()
    result = {"schema": "phenometrix-selector-grouping-v1", "started_at": now(),
              "source_sha256": frozen, "plan_sha256": ev.sha(PLAN),
              "coverage": {}, "input_sha256": {}, "reference_sha256": {}}
    for split in ("train", "development"):
        data = load_inputs(split); labels = load_labels(data); prepared = batches(data)
        result["coverage"][split] = coverage(data, labels, prepared)
        result["input_sha256"].update(data["hashes"])
        result["reference_sha256"][split] = labels["hashes"]
        print(json.dumps({"split": split, "coverage": {k:v for k,v in result["coverage"][split].items() if k != "cells"}}), flush=True)
    assert_sources(frozen)
    result.update(status="complete", completed_at=now())
    save(output / "grouping.json", result)
    return result


def prerequisites(output, headroom, grouping):
    output.mkdir(parents=True, exist_ok=False)
    result = json.loads(grouping.read_text())
    if (result.get("schema") != "phenometrix-selector-grouping-v1" or result.get("status") != "complete"
            or result["source_sha256"] != sources() or result["plan_sha256"] != ev.sha(PLAN)
            or set(result["coverage"]) != {"train", "development"}):
        raise ValueError("Requires the source-matched complete grouping audit")
    for name, expected in result["input_sha256"].items():
        if ev.sha(name) != expected:raise ValueError("Grouping acquisition input changed")
    # This is still the explicit prerequisite audit, not the fitting/prediction process.
    for entries in result["reference_sha256"].values():
        for name, expected in entries.items():
            if ev.sha(name) != expected:raise ValueError("Grouping reference changed")
    for split, cov in result["coverage"].items():
        expected_cells=48 if split=="train" else 16
        if len(cov["cells"]) != expected_cells:raise ValueError("Incomplete grouping cells")
        lost=float(np.mean([c["lost"]/c["voiced"] for c in cov["cells"]]))
        if (any(c["gained"] for c in cov["cells"]) or abs(lost-cov["lost_fraction"])>1e-12
                or cov["pass"] is not (lost <= .005)):
            raise ValueError("Grouping gate differs from counts")
    gate = validate_headroom(headroom)
    result.update(schema="phenometrix-selector-prerequisites-v1", headroom_path=str(headroom),
                  headroom_sha256=ev.sha(headroom), grouping_sha256=ev.sha(grouping), headroom_gate=gate,
                  completed_at=now())
    result["pass"] = bool(gate["pass"]) and all(v["pass"] for v in result["coverage"].values())
    save(output / "prerequisites.json", result)
    print(json.dumps({"prerequisites_pass":result["pass"], "headroom":gate}),flush=True)
    return result


def read_natural_baselines(data):
    folder = RUNS / "hybrid-natural-v1"
    freeze = json.loads((folder / "prediction-freeze.json").read_text())
    if freeze["split"] != "development" or freeze["feature_manifest_sha256"] != ev.sha(data["folder"] / "manifest.json"):
        raise ValueError("Baseline freeze differs from development inputs")
    original_folder = RUNS / "delayed-mlp-seed20261022"
    original = json.loads((original_folder / "manifest.json").read_text())
    binding = freeze["models"]["original"]
    if (binding["manifest_sha256"] != ev.sha(original_folder / "manifest.json")
            or binding["weights_sha256"] != original["weights_sha256"]
            or ev.sha(original_folder / "weights.pt") != binding["weights_sha256"]):
        raise ValueError("Baseline voicing belongs to a different original checkpoint")
    n = len(data["raw"]["features"])
    probability = np.zeros(n)
    outputs = {m: (np.zeros(n, np.uint8), np.full(n, np.nan)) for m in ("original_mlp", "original_hybrid")}
    hashes = {str(folder / "prediction-freeze.json"): ev.sha(folder / "prediction-freeze.json")}
    if len(freeze["recordings"]) != len(data["records"]):
        raise ValueError("Incomplete baseline freeze")
    for r, frozen in zip(data["records"], freeze["recordings"]):
        if r["recording_id"] != frozen["recording_id"] or r["feature_sha256"] != frozen["feature_sha256"]:
            raise ValueError("Baseline source identity mismatch")
        p = folder / frozen["hybrid_prediction_path"]
        if not p.resolve().is_relative_to(folder.resolve()) or ev.sha(p) != frozen["hybrid_prediction_sha256"]:
            raise ValueError("Baseline prediction identity changed")
        hashes[str(p)] = frozen["hybrid_prediction_sha256"]
        payload = ev.read_predictions(p)
        for name, key in (("original_mlp", "mlp_original"), ("original_hybrid", "hybrid_original")):
            rows = payload["predictions"][key]
            if [v["sample"] for v in rows] != r["targets"].tolist():
                raise ValueError("Baseline target grid mismatch")
            sl = slice(r["start"], r["stop"])
            outputs[name][0][sl] = [{"unvoiced":0,"voiced":1,"unavailable":2}[v["state"]] for v in rows]
            outputs[name][1][sl] = [np.nan if v["f0_hz"] is None else v["f0_hz"] for v in rows]
            if name == "original_mlp":
                probability[sl] = [v["voicing_probability"] for v in rows]
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Invalid frozen voicing probability")
    for state, hz in outputs.values():
        if not np.array_equal(state == 0, probability < .5):
            raise ValueError("Frozen baseline voicing decision differs from original probability")
    hashes[str(original_folder / "manifest.json")] = ev.sha(original_folder / "manifest.json")
    hashes[str(original_folder / "weights.pt")] = original["weights_sha256"]
    return probability, outputs, hashes


def infer(model, batch, probability):
    model = model.cpu().eval()
    with torch.inference_mode():
        pieces = []
        for start in range(0, len(batch.features), 4096):
            b = batch.index_select(torch.arange(start, min(start+4096, len(batch.features))))
            pieces.append(predict_pitch_hz(model(b), b).cpu().numpy())
        hz = np.concatenate(pieces)
    state = np.where(probability < .5, 0, np.where(np.isfinite(hz), 1, 2)).astype(np.uint8)
    hz[state != 1] = np.nan
    return state, hz


SYNTHETIC_IDS = {"constant-100", "constant-200", "constant-300", "step-100-200", "step-200-100",
                 *(f"glide-{direction}-{speed}" for direction in ("up", "down") for speed in (50,100,200)),
                 "observation-change-100", "observation-change-200", "observation-change-300"}


def synthetic_inputs(config, weights):
    folder = RUNS / "delayed-known-truth"
    manifest = json.loads((folder / "prediction-freeze.json").read_text())
    support_path = RUNS / "support-diagnostic-v1/manifest.json"
    original_folder = RUNS / "delayed-mlp-seed20261022"
    if (manifest.get("schema") != "phenometrix-delayed-known-truth-evaluation-v1"
            or manifest.get("status") != "predictions_frozen" or manifest.get("measurement_delay_ms") != 20
            or len(manifest.get("recordings", [])) != 14
            or {r["id"] for r in manifest["recordings"]} != SYNTHETIC_IDS
            or manifest["original_support_manifest_sha256"] != ev.sha(support_path)
            or manifest["models"]["mlp"]["weights_sha256"] != ev.sha(original_folder/"weights.pt")
            or manifest["models"]["mlp"]["manifest_sha256"] != ev.sha(original_folder/"manifest.json")):
        raise ValueError("Incomplete or changed frozen synthetic regression inventory")
    output, hashes = [], {str(folder / "prediction-freeze.json"): ev.sha(folder / "prediction-freeze.json"),
                          str(support_path):ev.sha(support_path)}
    for row in manifest["recordings"]:
        p = folder / "features" / (row["id"] + ".npz")
        if ev.sha(p) != row["feature_sha256"]:
            raise ValueError("Synthetic regression features changed")
        hashes[str(p)] = row["feature_sha256"]
        with np.load(p, allow_pickle=False) as z:
            f = {k: z[k].copy() for k in z.files}
        timing = ev.validate_feature(f, {"sample_count": int(f["pcm_sample_count"])})
        prediction_path = folder/"predictions"/(row["id"]+".json.gz")
        if ev.sha(prediction_path) != row["prediction_sha256"]:
            raise ValueError("Frozen synthetic original prediction changed")
        hashes[str(prediction_path)] = row["prediction_sha256"]
        old = ev.read_predictions(prediction_path)["predictions"]["mlp"]
        if [r["sample"] for r in old] != timing.reference_sample.tolist():
            raise ValueError("Synthetic original target grid differs")
        probability = np.array([r["voicing_probability"] for r in old])
        if not np.isfinite(probability).all() or np.any((probability<0)|(probability>1)):
            raise ValueError("Invalid original synthetic voicing probability")
        prepared = {rep: prepare_selector_batch(f["features"], f["candidate_hz"], f["candidate_mask"], representation=rep)
                    for rep in ("fixed-slots", "shared-candidates")}
        output.append({"id": row["id"], "timing": timing, "probability": probability, "batches": prepared})
    return output, hashes


def synthetic_scores(predictions, inputs, support_sha256):
    from .evaluate_synthetic import OBSERVATION_SCENARIOS, analytic_frequency, region_at, summarize, step_response
    path = RUNS / "support-diagnostic-v1/manifest.json"
    if ev.sha(path) != support_sha256:
        raise ValueError("Synthetic support metadata changed after prediction freeze")
    if set(predictions) != SYNTHETIC_IDS or len(inputs) != 14 or {r["id"] for r in inputs} != SYNTHETIC_IDS:
        raise ValueError("Incomplete synthetic scoring inventory")
    support = json.loads(path.read_text())
    scenarios = {s["id"]:s for s in [*support["scenarios"], *OBSERVATION_SCENARIOS]}
    results = {}
    for item in inputs:
        name, timing = item["id"], item["timing"]
        scenario = scenarios[name]; methods = {}
        for method, (state, hz) in predictions[name].items():
            rows = []
            for i in np.flatnonzero(timing.common_eligible):
                sample, end = int(timing.reference_sample[i]), int(timing.availability_sample[i])
                truth = analytic_frequency(scenario, sample/48000)
                estimate = float(hz[i]) if state[i] == 1 else None
                region = region_at(scenario, end)
                slope = scenario["slopeHzPerSecond"] if region == "glide-interior" else 0
                rows.append({"reference_sample": sample,"target_ms":sample/48,"availability_ms":end/48,
                    "region":region,"estimate_hz":estimate,"signed_cents":None if estimate is None else 1200*np.log2(estimate/truth),
                    "equivalent_slope_delay_ms":None if estimate is None or not slope else (truth-estimate)/slope*1000})
            methods[method] = {"all": summarize(rows), "regions": {region:summarize([r for r in rows if r["region"]==region])
                              for region in sorted({r["region"] for r in rows})},
                              "step_response":step_response(rows,scenario) if scenario["kind"]=="step" else None}
        results[name] = {"kind":scenario["kind"],"methods":methods}
    passes = {model_key("shared-candidates","acceptable-set",s): all(v["methods"][model_key("shared-candidates","acceptable-set",s)]["all"]["success_within_50c_all_frames"] >= .95
               for v in results.values() if v["kind"] == "constant") for s in SEEDS}
    return {"scenarios":results,"stationary_candidate_pass_by_seed":passes,"stationary_candidate_pass":all(passes.values()),
            "scope":"reused analytic regression; not fresh generalization", "support_manifest_sha256":ev.sha(path)}


def run(output, prerequisite_path):
    prerequisite = json.loads(prerequisite_path.read_text())
    if prerequisite.get("status") != "complete" or prerequisite.get("pass") is not True:
        raise ValueError("Complete passing prerequisites required")
    frozen = sources()
    if prerequisite["source_sha256"] != frozen or prerequisite["plan_sha256"] != ev.sha(PLAN):
        raise ValueError("Source or plan changed since prerequisites")
    if ev.sha(prerequisite["headroom_path"]) != prerequisite["headroom_sha256"]:
        raise ValueError("Headroom audit changed")
    validate_headroom(Path(prerequisite["headroom_path"]))
    for name, expected in prerequisite["input_sha256"].items():
        if ev.sha(name) != expected:
            raise ValueError("Prerequisite input identity changed")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Local MPS is required; no remote or CPU fit fallback")
    if shutil.disk_usage(ROOT).free < 8*1024**3:
        raise RuntimeError("At least 8 GiB free disk required")
    output.mkdir(parents=True, exist_ok=False)
    for p in ("checkpoints", "predictions", "source-snapshot"):
        (output/p).mkdir()
    for name in frozen:
        p=output/"source-snapshot"/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes((REPO/name).read_bytes())
    manifest={"schema":"phenometrix-selector-factorial-v1","status":"loading","started_at":now(),
              "source_sha256":frozen,"prerequisites_sha256":ev.sha(prerequisite_path),"device":"mps",
              "torch":torch.__version__,"seeds":list(SEEDS),"arms":[list(a) for a in ARMS],
              "epochs":12,"batch_size":4096,"learning_rate":.001,"measurement_delay_ms":20,
              "frozen_original_voicing":True,"development_references_read":False,"reserved_content_read":False,
              "fits":{},"input_sha256":{},"peak_observed_mps_allocated_bytes":0}
    save(output/"manifest.json",manifest)
    try:
        train=load_inputs("train"); train_labels=load_labels(train); train_batches=batches(train)
        if train_labels["hashes"] != prerequisite["reference_sha256"]["train"]:
            raise ValueError("Training reference identities differ from prerequisites")
        manifest["input_sha256"].update(train["hashes"]|train_labels["hashes"])
        train_targets={rep:targets(b,train_labels) for rep,b in train_batches.items()}
        schedule=torch.nonzero(train_targets["fixed-slots"].covered,as_tuple=False).reshape(-1)
        manifest["scheduled_rows"]=len(schedule)
        manifest["grouped_uncovered_scheduled_rows"]=int((~train_targets["shared-candidates"].covered[schedule]).sum())
        device_batches={rep:b.index_select(schedule).to("mps") for rep,b in train_batches.items()}
        device_targets={rep:t.index_select(schedule).to("mps") for rep,t in train_targets.items()}
        dev=load_inputs("development"); prepared_dev=batches(dev)
        probability, predictions, baseline_hashes=read_natural_baselines(dev)
        manifest["input_sha256"].update(dev["hashes"]|baseline_hashes)
        train_manifest,_=ev.read_cache_manifest(RUNS/"delayed-train-features")
        config,weights,original=ev.load_frozen_model(RUNS/"delayed-mlp-seed20261022",RUNS/"delayed-train-features",train_manifest)
        manifest["frozen_original_weights_sha256"]=original["weights_sha256"]
        synthetic,synthetic_hashes=synthetic_inputs(config,weights)
        manifest["input_sha256"].update(synthetic_hashes)
        synthetic_predictions={s["id"]:{} for s in synthetic}
        manifest["status"]="fitting"; save(output/"manifest.json",manifest)
        began=time.monotonic()
        models={}
        for seed in SEEDS:
            for rep,objective in ARMS:
                torch.manual_seed(seed); torch.mps.manual_seed(seed)
                model=PitchSelector(SelectorConfig(representation=rep,loss=objective))
                model.fit_scaling(train_batches[rep],split="train"); model=model.to("mps")
                optimizer=torch.optim.Adam(model.parameters(),lr=.001)
                b,t=device_batches[rep],device_targets[rep]
                losses=[]; updates=skipped=0; started=time.monotonic()
                key=model_key(rep,objective,seed)
                for epoch in range(12):
                    order=torch.from_numpy(np.random.default_rng(seed+epoch).permutation(len(schedule)).astype(np.int64))
                    total=denominator=0
                    for start in range(0,len(order),4096):
                        if time.monotonic()-began>1800:
                            raise TimeoutError("Declared 30 minute fit budget reached")
                        ix=order[start:start+4096]; bt=b.index_select(ix); target=t.index_select(ix)
                        covered=int(target.covered.sum().item())
                        if not covered:
                            skipped+=1; continue
                        optimizer.zero_grad(set_to_none=True)
                        logits=model(bt)
                        loss=selector_loss(logits,bt.candidate_mask,target,objective=objective)
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite selector loss")
                        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
                        total+=float(loss.detach().cpu())*covered; denominator+=covered; updates+=1
                    torch.mps.synchronize()
                    losses.append(total/denominator)
                    allocation=torch.mps.current_allocated_memory()
                    manifest["peak_observed_mps_allocated_bytes"]=max(allocation,manifest["peak_observed_mps_allocated_bytes"])
                    if allocation>8*1024**3:
                        raise MemoryError("Declared 8 GiB MPS allocation cap exceeded")
                model=model.cpu().eval(); path=output/"checkpoints"/(key+".pt")
                torch.save({"config":asdict(model.config),"state_dict":model.state_dict()},path)
                fit={"seed":seed,"representation":rep,"objective":objective,"epoch_losses":losses,
                     "optimizer_updates":updates,"skipped_batches":skipped,"elapsed_seconds":time.monotonic()-started,
                     "parameter_count":sum(p.numel() for p in model.parameters()),"weights_sha256":ev.sha(path)}
                manifest["fits"][key]=fit; models[key]=model
                save(output/"manifest.json",manifest)
                print(json.dumps({"fit":key,**fit}),flush=True)
        manifest["checkpoints_frozen_at"]=now(); manifest["fit_elapsed_seconds"]=time.monotonic()-began
        assert_sources(frozen); save(output/"checkpoint-freeze.json",manifest)
        for key,model in models.items():
            rep=model.config.representation
            predictions[key]=infer(model,prepared_dev[rep],probability)
            for item in synthetic:
                synthetic_predictions[item["id"]][key]=infer(model,item["batches"][rep],item["probability"])
        prediction_hashes={}
        for key,(state,hz) in predictions.items():
            p=output/"predictions"/(key+".npz");np.savez_compressed(p,state=state,hz=hz)
            prediction_hashes[str(p.relative_to(output))]=ev.sha(p)
        for name,methods in synthetic_predictions.items():
            p=output/"predictions"/("synthetic-"+name+".npz")
            np.savez_compressed(p,**{key+"__state":v[0] for key,v in methods.items()},**{key+"__hz":v[1] for key,v in methods.items()})
            prediction_hashes[str(p.relative_to(output))]=ev.sha(p)
        manifest.update(status="predictions_frozen",predictions_frozen_at=now(),prediction_sha256=prediction_hashes)
        save(output/"prediction-freeze.json",manifest);save(output/"manifest.json",manifest)
        # Final numerical predictions now fixed. Only here does this runner open development labels.
        dev_labels=load_labels(dev);manifest["development_references_read"]=True
        if dev_labels["hashes"] != prerequisite["reference_sha256"]["development"]:
            raise ValueError("Development reference identities differ from prerequisites")
        manifest["input_sha256"].update(dev_labels["hashes"])
        result=aggregate(dev,dev_labels,predictions); synth=synthetic_scores(synthetic_predictions,synthetic,synthetic_hashes[str(RUNS/"support-diagnostic-v1/manifest.json")])
        decision=natural_decision(result)
        decision["stationary_regression_pass"]=synth["stationary_candidate_pass"]
        decision["promising_for_next_study"]=decision["pass"] and decision["stationary_regression_pass"]
        summary={"schema":"phenometrix-selector-factorial-metrics-v1","scope":"reused development and analytic regression; no independent test",
                 "measurement_delay_ms":20,"aggregation":"pool each speaker-condition then equal-cell mean",
                 "natural":result,"synthetic":synth,"decision":decision,"exclusions":dev_labels["excluded"]}
        save(output/"metrics.json",summary)
        for name,expected in prediction_hashes.items():
            if ev.sha(output/name)!=expected:raise ValueError("Predictions changed after freeze")
        assert_sources(frozen)
        size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if size>2*1024**3:raise RuntimeError("Declared 2 GiB output cap exceeded")
        manifest.update(status="complete",completed_at=now(),metrics_sha256=ev.sha(output/"metrics.json"),output_bytes=size)
        save(output/"manifest.json",manifest)
        print(json.dumps(decision,indent=2),flush=True)
        return summary
    except BaseException as e:
        manifest.update(status="failed",failure=repr(e),failed_at=now());save(output/"manifest.json",manifest);raise


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("grouping");a.add_argument("--output",type=Path,required=True)
    a=sub.add_parser("prerequisites");a.add_argument("--headroom",type=Path,required=True);a.add_argument("--grouping",type=Path,required=True);a.add_argument("--output",type=Path,required=True)
    a=sub.add_parser("run");a.add_argument("--prerequisites",type=Path,required=True);a.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.command=="grouping": grouping_audit(a.output.resolve())
    elif a.command=="prerequisites": prerequisites(a.output.resolve(),a.headroom.resolve(),a.grouping.resolve())
    else: run(a.output.resolve(),a.prerequisites.resolve())

if __name__=="__main__":main()
