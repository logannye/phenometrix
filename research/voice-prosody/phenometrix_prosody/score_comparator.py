"""Scorer-only comparison of already frozen centered pretrained predictions."""
import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from .evaluate import read_cache_manifest, combine_scores, macro
from .metrics import Prediction, score
from .pilot import digest, references_from_array, write_json


def run(feature_run, predictions_run, data_root, output):
    cache, acquisition = read_cache_manifest(feature_run)
    manifest = json.loads((predictions_run / "manifest.json").read_text())
    freeze_path = predictions_run / "prediction-freeze.json"
    if (manifest.get("status") != "complete" or manifest.get("split") != cache["split"]
            or manifest.get("input_cache_manifest_sha256") != digest(feature_run / "manifest.json")
            or manifest.get("reference_or_laryngograph_content_read") is not False
            or manifest.get("scoring_performed") is not False
            or manifest.get("execution_backend") != "pinned_torchcrepe_tiny"
            or manifest.get("declared_lookahead_ms") != 32 or manifest.get("score_target_shift_ms") != 0
            or digest(freeze_path) != manifest.get("prediction_freeze_sha256")):
        raise ValueError("Requires an unchanged input-only pinned centered comparator freeze")
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("prediction_sha256") != {r["recording_id"]: r["prediction_sha256"] for r in manifest["recordings"]}:
        raise ValueError("Prediction manifest differs from reference-free freeze")
    expected = {r["recording_id"]: r for r in cache["recordings"]}
    if len(manifest["recordings"]) != len(expected) or {r["recording_id"] for r in manifest["recordings"]} != set(expected):
        raise ValueError("Prediction cohort differs from feature cohort")
    # Verify every frozen prediction before loading any label.
    for r in manifest["recordings"]:
        original = expected[r["recording_id"]]
        for key in ("source_recording_id", "speaker", "pair", "condition", "sample_count", "microphone_sha256", "transformed_pcm_sha256"):
            if r[key] != original[key]:
                raise ValueError("Prediction source/condition differs from frozen feature input")
        if digest(predictions_run / "predictions" / (r["recording_id"] + ".json")) != r["prediction_sha256"]:
            raise ValueError("Prediction bytes changed")
    output.mkdir(parents=True, exist_ok=False)
    refs_by_source = {r["recording_id"]: r for r in acquisition["files"] if r["role"] == "ref" and r["split"] == cache["split"]}
    groups, per_record, ref_cache, ref_hashes = defaultdict(list), [], {}, {}
    for record in manifest["recordings"]:
        source = record["source_recording_id"]
        if source not in ref_cache:
            metadata = refs_by_source[source]
            file = data_root / metadata["path"]
            if digest(file) != metadata["sha256"]:
                raise ValueError("Frozen reference changed")
            ref_cache[source] = [replace(r, eligible=r.sample >= 3072 and r.sample + 1536 <= record["sample_count"])
                for r in references_from_array(np.loadtxt(file), record["sample_count"])]
            ref_hashes[source] = metadata["sha256"]
        refs = ref_cache[source]
        targets = {r.sample for r in refs}
        raw = json.loads((predictions_run / "predictions" / (record["recording_id"] + ".json")).read_text())["predictions"]
        pred = [Prediction(r["sample"], r["state"], r["f0_hz"]) for r in raw if r["sample"] in targets]
        per_record.append({"recording_id": record["recording_id"], "speaker": record["speaker"],
            "condition": record["condition"], "metrics": score(refs, pred)})
        groups[record["speaker"], record["condition"]].append((refs, pred, record["sample_count"]))
    speakers = sorted({r["speaker"] for r in manifest["recordings"]})
    cells = {s: {c: combine_scores(groups[s, c]) for c in cache["conditions"]} for s in speakers}
    result = {"schema": "phenometrix-centered-comparator-scores-v1", "status": "complete", "split": cache["split"],
        "created_at": datetime.now(timezone.utc).isoformat(), "declared_lookahead_ms": 32,
        "shared_mask": "target>=64ms and target+32ms<=N; original nominal16+10k-ms targets",
        "feature_manifest_sha256": digest(feature_run / "manifest.json"),
        "prediction_manifest_sha256": digest(predictions_run / "manifest.json"),
        "prediction_freeze_sha256": digest(freeze_path), "reference_sha256": ref_hashes,
        "scorer_sha256": digest(Path(__file__)), "per_record": per_record, "per_speaker_condition": cells,
        "per_condition_equal_speaker": {c: macro([cells[s][c] for s in speakers]) for c in cache["conditions"]},
        "equal_speaker_equal_condition": macro([cells[s][c] for s in speakers for c in cache["conditions"]]),
        "interpretation": "Pretrained comparison has32ms future support, versus20ms for the candidate experiment; no browser runtime comparison"}
    write_json(output / "metrics.json", result)
    print(json.dumps({"overall": result["equal_speaker_equal_condition"], "conditions": result["per_condition_equal_speaker"]}, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("features", "predictions", "data", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args(); run(a.features.resolve(), a.predictions.resolve(), a.data.resolve(), a.output.resolve())
