"""Input-only centered tiny predictions for a frozen train/development cache.

This module never opens reference or LAR content and has no scoring operation.
It preserves the nominal target grid and centered 32 ms future support, even
when the feature cache describes a different 20 ms measurement-delay method.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path

import numpy as np
import torch

from .comparator import (TinyComparator, TinyConfig, TORCHCREPE_COMMIT,
                         TINY_SHA256, MODEL_SOURCE_SHA256, target_grid)
from .data import split_assignment
from .features import CONDITIONS, transform
from .pilot import REPO, digest, read_mic, write_json

METHOD = "torchcrepe-tiny-0.0.24-centered-frame-local-v1"
TRANSFORM_SOURCE = "research/voice-prosody/phenometrix_prosody/features.py"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def frozen_worklist(feature_run):
    """Validate only cache/acquisition metadata; do not open any acoustic outcome."""
    feature_run = Path(feature_run)
    raw = (feature_run / "manifest.json").read_bytes()
    cache = json.loads(raw)
    if cache.get("schema") != "phenometrix-input-only-feature-cache-v1" or cache.get("status") != "complete":
        raise ValueError("A completed input-only feature cache is required")
    split = cache.get("split")
    if split not in {"train", "development"}:
        raise ValueError("Pre-freeze comparator refuses sealed-test or unknown splits")
    if cache.get("reference_or_laryngograph_content_read") is not False:
        raise ValueError("Feature cache must explicitly declare input-only construction")
    if cache.get("measurement_delay_ms") != 20:
        raise ValueError("Expected the declared 20 ms feature cache; comparator remains centered with 32 ms lookahead")
    conditions = cache.get("conditions", [])
    if not conditions or len(set(conditions)) != len(conditions) or any(c not in CONDITIONS for c in conditions):
        raise ValueError("Unknown or repeated corruption condition")
    acquisition = feature_run / "acquisition-manifest.json"
    if digest(acquisition) != cache["data_manifest_sha256"]:
        raise ValueError("Frozen acquisition metadata changed")
    rows = sorted((r for r in json.loads(acquisition.read_bytes())["files"]
                   if r["role"] == "mic" and r["split"] == split), key=lambda r: r["recording_id"])
    if not rows or len({r["recording_id"] for r in rows}) != len(rows):
        raise ValueError("Missing or duplicate microphone identities")
    allowed = set(split_assignment()[split])
    for r in rows:
        if (not r["primary_eligible"] or r["pair"] not in allowed or
            int(r["speaker"][1:]) != r["pair"] or r["prompt"] in {"sa1", "sa2"}):
            raise ValueError("Microphone split/group/primary-eligibility mismatch")
    if json.loads((feature_run / "microphone-selection.json").read_text()) != rows:
        raise ValueError("Microphone selection differs from frozen acquisition metadata")
    expected = [(r, c) for r in rows for c in conditions]
    records = cache.get("recordings", [])
    if len(records) != len(expected):
        raise ValueError("Completed cache has incomplete condition coverage")
    work = []
    for record, (source, condition) in zip(records, expected):
        identity = source["recording_id"] + "__" + condition
        if (record["recording_id"] != identity or record["source_recording_id"] != source["recording_id"]
            or record["condition"] != condition or record["microphone_sha256"] != source["sha256"]
            or any(record[key] != source[key] for key in ("speaker", "pair", "prompt"))):
            raise ValueError("Source/condition order or identity differs from frozen cache")
        work.append((source, record))
    source_hashes = cache["identity"]["source_sha256"]
    if TRANSFORM_SOURCE not in source_hashes:
        raise ValueError("Corruption implementation is not pinned in feature cache")
    for relative, expected_sha in source_hashes.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Invalid source snapshot path")
        if digest(feature_run / "source-snapshot" / path) != expected_sha or digest(REPO / path) != expected_sha:
            raise ValueError(f"Frozen/current source differs: {relative}")
    return cache, hashlib.sha256(raw).hexdigest(), work


def reconstruct_pcm(data_root, source, record, split):
    """Read only the pinned MIC, reconstruct corruption, and verify exact bytes."""
    root = Path(data_root).resolve()
    path = (root / source["path"]).resolve()
    if not path.is_relative_to(root) or source["role"] != "mic" or source["split"] != split:
        raise ValueError("Refusing an out-of-root or non-microphone input")
    if digest(path) != source["sha256"]:
        raise ValueError("Pinned microphone bytes changed")
    pcm = read_mic(path)
    transformed, corruption = transform(pcm, record["condition"], source["recording_id"], split)
    raw_hash = hashlib.sha256(transformed.astype("<f4").tobytes()).hexdigest()
    if len(transformed) != record["sample_count"] or raw_hash != record["transformed_pcm_sha256"]:
        raise ValueError("Reconstructed corruption differs from frozen feature-cache PCM")
    if corruption != record["corruption"]:
        raise ValueError("Reconstructed corruption metadata differs")
    return transformed


def run(feature_run, data_root, output, device="mps", batch_size=128, *, _test_backend=None):
    feature_run, data_root, output = map(Path, (feature_run, data_root, output))
    cache, cache_sha, work = frozen_worklist(feature_run)
    if _test_backend is not None and device != "cpu":
        raise ValueError("Synthetic test backends must use CPU")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Local Apple MPS is unavailable; no remote fallback")
    config = TinyConfig(device=device, batch_size=batch_size, anchor="centered")
    comparator = TinyComparator(config, backend=_test_backend)
    output.mkdir(parents=True, exist_ok=False)
    (output / "predictions").mkdir()
    (output / "source-snapshot").mkdir()
    sources = {}
    for name in ("comparator_cache.py", "comparator.py", "features.py", "pilot.py"):
        path = Path(__file__).with_name(name)
        sources[name] = digest(path)
        (output / "source-snapshot" / name).write_bytes(path.read_bytes())
    manifest = {"schema": "phenometrix-centered-tiny-cache-predictions-v1", "status": "running",
        "started_at": utc_now(), "split": cache["split"], "conditions": cache["conditions"],
        "input_feature_cache": str(feature_run.resolve()), "input_cache_manifest_sha256": cache_sha,
        "input_data_manifest_sha256": cache["data_manifest_sha256"],
        "source_sha256": sources, "method": METHOD, "config": asdict(config),
        "time_contract": "nominal target t on16+10*k ms; source[t-32ms,t+32ms); input availabilityt+32ms",
        "declared_lookahead_ms": 32, "score_target_shift_ms": 0,
        "feature_method_measurement_delay_ms": 20,
        "reference_or_laryngograph_content_read": False, "scoring_performed": False,
        "execution_backend": "synthetic_test_fixture" if _test_backend is not None else "pinned_torchcrepe_tiny",
        "model": {"package_version": "0.0.24", "commit": TORCHCREPE_COMMIT,
                  "tiny_weight_sha256": TINY_SHA256, "model_source_sha256": MODEL_SOURCE_SHA256},
        "dependencies": {name: metadata.version(name) for name in ("numpy", "scipy", "torch", "torchaudio", "torchcrepe")},
        "recordings": []}
    write_json(output / "manifest.json", manifest)
    try:
        for source, cached in work:
            pcm = reconstruct_pcm(data_root, source, cached, cache["split"])
            predictions = comparator.predict(pcm, sample_rate=48000)
            if [p["sample"] for p in predictions] != list(target_grid(len(pcm))):
                raise ValueError("Comparator changed the nominal target grid")
            if any(p["method"] != METHOD or p["declared_lookahead_ms"] != 32
                   or p["available_at_sample"] != p["sample"] + 1536 for p in predictions):
                raise ValueError("Comparator changed the declared centered support")
            identity = cached["recording_id"]
            file = output / "predictions" / (identity + ".json")
            write_json(file, {"recording_id": identity, "source_recording_id": source["recording_id"],
                             "split": cache["split"], "condition": cached["condition"], "predictions": predictions})
            manifest["recordings"].append({"recording_id": identity,
                "source_recording_id": source["recording_id"], "speaker": source["speaker"],
                "pair": source["pair"], "prompt": source["prompt"], "condition": cached["condition"],
                "sample_count": len(pcm), "microphone_sha256": source["sha256"],
                "transformed_pcm_sha256": cached["transformed_pcm_sha256"],
                "prediction_sha256": digest(file), "prediction_count": len(predictions)})
            write_json(output / "manifest.json", manifest)
            print(f"Predicted {identity}: {len(predictions)} nominal targets", flush=True)
        if digest(feature_run / "manifest.json") != cache_sha:
            raise ValueError("Input feature cache manifest changed during prediction")
        for name, expected_sha in sources.items():
            if digest(Path(__file__).with_name(name)) != expected_sha:
                raise ValueError("Comparator/transform source changed during prediction")
        freeze = {"schema": "phenometrix-input-only-comparator-freeze-v1", "created_at": utc_now(),
            "input_cache_manifest_sha256": cache_sha, "source_sha256": sources,
            "scoring_performed": False, "reference_or_laryngograph_content_read": False,
            "execution_backend": manifest["execution_backend"],
            "prediction_sha256": {r["recording_id"]: r["prediction_sha256"] for r in manifest["recordings"]}}
        write_json(output / "prediction-freeze.json", freeze)
        manifest.update(status="complete", completed_at=utc_now(), prediction_freeze_sha256=digest(output / "prediction-freeze.json"))
        write_json(output / "manifest.json", manifest)
        return manifest
    except BaseException as error:
        manifest.update(status="failed", failure=repr(error))
        write_json(output / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    run(args.features.resolve(), args.data.resolve(), args.output.resolve(), args.device, args.batch_size)


if __name__ == "__main__":
    main()
