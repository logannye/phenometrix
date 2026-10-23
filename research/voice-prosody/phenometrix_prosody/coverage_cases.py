"""Prospective coverage challenge after source/three existing-checkpoint freeze.

Only metadata, authorization and input cache provenance live here. Unchanged
harmonic synthesis and v2 replay provide baseline acquisition inputs; a separate
research wrapper may consume those inputs under its own bound source contract.
This module never fits, reads natural references or changes production capture.
Inspecting metadata does not generate PCM. All fifty prior cases stay regression.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from . import harmonic_challenge as previous
from .evaluate import validate_feature

REPO, ROOT, RUNS = previous.REPO, previous.ROOT, previous.RUNS
FREQUENCIES = (97.1, 213.7, 347.3, 529.1, 627.7, 671.3)
PHASES = ((.29, -.67, 1.31), (1.89, .47, -1.43))
SEEDS = (20261022, 20261023, 20261024)
MAX_BYTES = previous.MAX_BYTES
SPEC_SCHEMA = "phenometrix-candidate-coverage-challenge-spec-v1"
CACHE_SCHEMA = "phenometrix-candidate-coverage-challenge-cache-v1"


def challenge_spec():
    """Prospectively fixed metadata; no waveform, feature or outcome access."""
    records = []
    for index, frequency in enumerate(FREQUENCIES):
        for phase_index, phases in enumerate(PHASES):
            records.append({"id": f"coverage-balance-f{index}-p{phase_index}", "frequency_hz": frequency,
                "phase_radians": list(phases), "sample_rate_hz": 48000, "sample_count": 96000,
                "change_sample": 39840, "gain": .055, "before": [1., .5, .25], "after": [0., 1., .5]})
    return {"schema": SPEC_SCHEMA, "split": "prospective-challenge", "records": records,
            "sequence_count": 12, "measurement_delay_ms": 20, "feature_version": "v2",
            "acquisition_end_grid": "2208+480*k samples", "reference_relation": "exclusive_support_end-960",
            "common_mask": "reference>=3072 and reference+1536<=sample_count",
            "change": "sample39840 (0.83s); constant latent F0; nonzero harmonics2and3 remain",
            "post_region": "fully post-change 40ms support: support_start_sample>=39840",
            "frozen_selector_arm": "balance-aug",
            "prospective_gate": "each predesignated-candidate seed: >=95% success within50c in EACH post region; no frame drops",
            "whole_and_other_regions": "report all/regions, raw and grouped candidate coverage and frozen unvoiced/unavailable counts",
            "noise_or_normalization": "none", "duration_seconds_total": 24,
            "independence": "prospectively frozen new complete frequency/phase/gain configurations within the same failure-motivated noiseless family; all fifty prior cases remain consumed regression",
            "novelty_boundary": "fixed carriers/endpoints differ from historical configured carriers/endpoints and TRAIN carriers; continuous historical glides and natural speech can cross these frequencies; not disjoint frequency support or external generalization",
            "identifiability": "nonzero harmonics2and3 preserve shortest period1/F0; no single-harmonic or even-only post signal",
            "access": "no waveform or feature generation until source freeze binds all three existing balance-aug checkpoints; no new fits",
            "maximum_output_bytes": MAX_BYTES}


def challenge_spec_sha256():
    return hashlib.sha256(previous._canonical(challenge_spec())).hexdigest()


def source_hashes():
    """Bind this helper, unchanged harmonic helper and the full v2 replay closure."""
    return {**previous.challenge_source_hashes(),
            str(Path(__file__).relative_to(REPO)): previous._sha(Path(__file__))}


def _expected_checkpoints():
    return {f"balance-aug__seed{seed}" for seed in SEEDS}


def _validate_freeze(path, expected_sha256):
    path = previous._inside(RUNS, Path(path).resolve())
    previous._require(path.is_file(), "A complete source/checkpoint freeze is required before challenge generation")
    previous._require(previous._sha(path) == expected_sha256, "Freeze identity changed")
    frozen = json.loads(path.read_text())
    previous._require(frozen.get("schema") == "phenometrix-candidate-coverage-freeze.v1",
                      "Unknown candidate coverage freeze schema")
    previous._require(frozen.get("status") == "sources_checkpoints_frozen",
                      "Sources and all existing checkpoints must freeze before challenge generation")
    previous._require(frozen.get("challenge_spec_sha256") == challenge_spec_sha256(),
                      "Challenge specification differs from predeclared specification")
    checkpoints = frozen.get("checkpoints", {})
    previous._require(isinstance(checkpoints, dict) and set(checkpoints) == _expected_checkpoints(),
                      "Exactly the three declared balance-aug checkpoints are required")
    sources = source_hashes()
    previous._require(all(frozen.get("source_sha256", {}).get(name) == value for name, value in sources.items()),
                      "Freeze lacks matching new/previous challenge and replay source closure")
    hashes = {str(path): expected_sha256, **{str(REPO / n): h for n, h in sources.items()}}
    resolved = set()
    for name, checkpoint in checkpoints.items():
        relative = checkpoint.get("path") if isinstance(checkpoint, dict) else None
        previous._require(isinstance(relative, str) and relative and not Path(relative).is_absolute(),
                          "Checkpoint path must be repository-relative")
        weight = previous._inside(REPO, relative)
        previous._require(weight.is_relative_to(RUNS.resolve()) and weight.name == name + ".pt",
                          "Checkpoint path must identify the declared existing run checkpoint")
        previous._require(weight not in resolved, "Checkpoint bindings must be distinct")
        resolved.add(weight)
        previous._require(weight.is_file() and previous._sha(weight) == checkpoint.get("sha256"),
                          "Frozen challenge checkpoint changed or missing")
        hashes[str(weight)] = checkpoint["sha256"]
    return frozen, hashes


def challenge_region_at(scenario, end_sample):
    """Use the same exact integer exclusive-support region contract."""
    return previous.challenge_region_at(scenario, end_sample)


def generate_challenge(output, freeze_path, expected_freeze_sha256):
    """Return (rows, absolute-path hashes) only after the source/checkpoint freeze."""
    freeze_path = Path(freeze_path).resolve()
    _validate_freeze(freeze_path, expected_freeze_sha256)
    output = previous._inside(RUNS, Path(output).resolve())
    previous._require(not output.exists(), "Challenge generation refuses to overwrite prior evidence")
    previous._require(output.parent.exists() and shutil.disk_usage(output.parent).free >= 512 * 1024 * 1024,
                      "Insufficient bounded-generation storage")
    output.mkdir()
    spec = challenge_spec(); sources = source_hashes()
    manifest = {"schema": CACHE_SCHEMA, "status": "generating", "started_at": previous._now(),
                "spec_sha256": challenge_spec_sha256(), "freeze_path": str(freeze_path.relative_to(REPO)),
                "freeze_sha256": expected_freeze_sha256, "source_sha256": sources,
                "records": [], "reference_content_read": False,
                "feature_input": "only finite float32 PCM and sample rate enter unchanged TypeScript replay",
                "maximum_output_bytes": MAX_BYTES}
    previous._save(output / "manifest.json", manifest)
    try:
        for name in ("pcm", "features", "intermediate", "source-snapshot"):
            (output / name).mkdir()
        previous._save(output / "spec.json", spec)
        for name, expected in sources.items():
            target = output / "source-snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((REPO / name).read_bytes())
            previous._require(previous._sha(target) == expected, "Challenge source snapshot changed")
        for record in spec["records"]:
            raw = output / "pcm" / (record["id"] + ".f32le")
            full = output / "intermediate" / (record["id"] + ".json")
            previous._synthesize_pcm(record).tofile(raw)
            feature = previous._extract_feature(raw, full)
            feature_path = output / "features" / (record["id"] + ".npz")
            np.savez_compressed(feature_path, **feature)
            export_sha = previous._sha(full); full.unlink()
            manifest["records"].append({"id": record["id"],
                "parameters_sha256": hashlib.sha256(previous._canonical(record)).hexdigest(),
                "pcm_path": str(raw.relative_to(output)), "pcm_sha256": previous._sha(raw),
                "feature_path": str(feature_path.relative_to(output)), "feature_sha256": previous._sha(feature_path),
                "typescript_export_sha256": export_sha, "feature_frames": len(feature["features"])})
            previous._require(sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) <= MAX_BYTES,
                              "Challenge output cap exceeded")
            previous._save(output / "manifest.json", manifest)
        print(f"coverage challenge: generated {len(manifest['records'])}/12 input records", flush=True)
        (output / "intermediate").rmdir()
        _validate_freeze(freeze_path, expected_freeze_sha256)
        previous._require(source_hashes() == sources, "Challenge source changed during generation")
        manifest.update(status="complete", completed_at=previous._now())
        previous._save(output / "manifest.json", manifest)
    except BaseException as error:
        manifest.update(status="failed", failed_at=previous._now(), failure=repr(error))
        previous._save(output / "manifest.json", manifest)
        raise
    return load_challenge(output, freeze_path, expected_freeze_sha256)


def load_challenge(output, freeze_path, expected_freeze_sha256):
    output = previous._inside(RUNS, Path(output).resolve())
    freeze_path = Path(freeze_path).resolve()
    _, hashes = _validate_freeze(freeze_path, expected_freeze_sha256)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    spec = challenge_spec(); sources = source_hashes()
    previous._require(json.loads((output / "spec.json").read_text()) == spec, "Cached challenge specification changed")
    previous._require(manifest.get("schema") == CACHE_SCHEMA and manifest.get("status") == "complete"
                      and manifest.get("spec_sha256") == challenge_spec_sha256()
                      and manifest.get("freeze_sha256") == expected_freeze_sha256
                      and manifest.get("freeze_path") == str(freeze_path.relative_to(REPO))
                      and manifest.get("source_sha256") == sources, "Challenge cache provenance differs")
    previous._require(len(manifest["records"]) == 12 and [r["id"] for r in manifest["records"]]
                      == [r["id"] for r in spec["records"]], "Challenge cache inventory differs")
    for name, expected in sources.items():
        snapshot = previous._inside(output / "source-snapshot", name)
        previous._require(previous._sha(snapshot) == expected, "Challenge source snapshot changed")
        hashes[str(snapshot)] = expected
    rows = []
    for record, saved in zip(spec["records"], manifest["records"]):
        previous._require(saved["parameters_sha256"] == hashlib.sha256(previous._canonical(record)).hexdigest(),
                          "Challenge parameters changed")
        for path_key, hash_key in (("pcm_path", "pcm_sha256"), ("feature_path", "feature_sha256")):
            path = previous._inside(output, saved[path_key])
            previous._require(previous._sha(path) == saved[hash_key], "Challenge input bytes changed")
            hashes[str(path)] = saved[hash_key]
        with np.load(previous._inside(output, saved["feature_path"]), allow_pickle=False) as data:
            feature = {k: data[k].copy() for k in data.files}
        timing = validate_feature(feature, {"sample_count": record["sample_count"]})
        previous._require(len(feature["features"]) == saved["feature_frames"], "Challenge frame inventory changed")
        previous._require(np.array_equal(timing.availability_sample, np.arange(2208, 96001, 480))
                          and int(timing.common_eligible.sum()) == 191,
                          "Challenge acquisition phase or common support changed")
        rows.append({"id": record["id"], "scenario": previous._scenario(record), "timing": timing, "feature": feature})
    hashes[str(manifest_path)] = previous._sha(manifest_path)
    hashes[str(output / "spec.json")] = previous._sha(output / "spec.json")
    return rows, hashes
