"""Prospective harmonic-balance inputs, generated only after all 12 fits freeze.

This module never fits, opens natural references, or changes acquisition features.
Importing it and reading the specification do not synthesize challenge waveforms.
The challenge probes one fixed harmonic transition over new frequency/phase
configurations; it does not establish arbitrary channel robustness.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from .evaluate import validate_feature
from .measurement_alignment import align_feature_timing

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "research/voice-prosody"
RUNS = ROOT / "runs"
FREQUENCIES = (87.3, 137.9, 223.7, 337.1, 467.3, 613.7)
PHASES = ((.37, -.41, 1.13), (1.19, .23, -.73))
SEEDS = (20261022, 20261023, 20261024)
MAX_BYTES = 40 * 1024 * 1024
SPEC_SCHEMA = "phenometrix-harmonic-balance-challenge-spec-v1"
CACHE_SCHEMA = "phenometrix-harmonic-balance-challenge-cache-v1"


def _require(value, reason):
    if not value:
        raise ValueError(reason)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _inside(root, path):
    p = (root / path).resolve()
    _require(p.is_relative_to(root.resolve()), "Challenge path escaped declared root")
    return p


def _save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def challenge_spec():
    """Metadata only, suitable for the pre-fit frozen protocol/manifest."""
    records = []
    for index, frequency in enumerate(FREQUENCIES):
        for phase_index, phases in enumerate(PHASES):
            records.append({"id": f"harmonic-balance-f{index}-p{phase_index}", "frequency_hz": frequency,
                "phase_radians": list(phases), "sample_rate_hz": 48000, "sample_count": 96000,
                "change_sample": 39840, "gain": .06, "before": [1., .5, .25], "after": [0., 1., .5]})
    return {"schema": SPEC_SCHEMA, "split": "prospective-challenge", "records": records,
            "sequence_count": 12, "measurement_delay_ms": 20, "feature_version": "v2",
            "acquisition_end_grid": "2208+480*k samples", "reference_relation": "exclusive_support_end-960",
            "common_mask": "reference>=3072 and reference+1536<=sample_count",
            "change": "sample39840 (0.83s); constant latent F0; nonzero harmonics2and3 remain",
            "post_region": "fully post-change 40ms support: support_start_sample>=39840",
            "prospective_gate": "each predesignated-candidate seed: >=95% success within50c in EACH post region; no frame drops",
            "whole_and_other_regions": "report all/regions, candidate coverage and frozen unvoiced/unavailable counts",
            "noise_or_normalization": "none", "duration_seconds_total": 24,
            "independence": "new complete frequency/phase configurations; original14 and prior synthetic challenges remain reused regression",
            "access": "no waveform or feature generation until complete source-bound12-checkpoint freeze",
            "maximum_output_bytes": MAX_BYTES}


def challenge_spec_sha256():
    return hashlib.sha256(_canonical(challenge_spec())).hexdigest()


def _source_paths():
    python = [Path(__file__), *(Path(__file__).with_name(n) for n in
        ("evaluate.py", "model.py", "measurement_alignment.py", "pilot.py", "metrics.py"))]
    relative = ["research/voice-prosody/replay.ts", "research/voice-prosody/fixtures.ts", "research/voice-prosody/oracle.ts",
                "apps/capture-web/src/voice-replay.ts", "apps/capture-web/src/voice-dsp.ts",
                "apps/capture-web/src/dsp-primitives.ts", "apps/capture-web/src/voice-pitch-features.ts",
                "apps/capture-web/src/voice-pitch-features-v2.ts", "apps/capture-web/package.json",
                "research/voice-prosody/pyproject.toml", "research/voice-prosody/uv.lock", "pnpm-lock.yaml"]
    return [*python, *(REPO / p for p in relative)]


def challenge_source_hashes():
    return {str(p.relative_to(REPO)): _sha(p) for p in _source_paths()}


def _expected_fits():
    return {f"{context}__{objective}__seed{seed}" for seed in SEEDS
            for context in ("absolute", "relative") for objective in ("nearest", "acceptable-set")}


def _validate_freeze(path, expected_sha256):
    path = _inside(RUNS, Path(path).resolve())
    _require(path.is_file(), "A complete checkpoint freeze is required before challenge generation")
    _require(_sha(path) == expected_sha256, "Checkpoint freeze identity changed")
    frozen = json.loads(path.read_text())
    _require(frozen.get("status") == "checkpoints_frozen", "All checkpoints must freeze before challenge generation")
    _require(frozen.get("challenge_spec_sha256") == challenge_spec_sha256(), "Challenge specification differs from pre-fit declaration")
    _require(set(frozen.get("fits", {})) == _expected_fits(), "Exactly the declared12 final fits are required")
    sources = challenge_source_hashes()
    _require(all(frozen.get("source_sha256", {}).get(name) == value for name, value in sources.items()),
             "Checkpoint freeze lacks matching challenge/replay source closure")
    hashes = {str(path): expected_sha256, **{str(REPO / n): h for n, h in sources.items()}}
    for name, fit in frozen["fits"].items():
        weight = _inside(path.parent, Path("checkpoints") / (name + ".pt"))
        _require(weight.is_file() and _sha(weight) == fit.get("weights_sha256"), "Frozen challenge checkpoint changed or missing")
        hashes[str(weight)] = fit["weights_sha256"]
    return frozen, hashes


def _synthesize_pcm(record):
    """Pure arithmetic; public entry points enforce the freeze before calling."""
    t = np.arange(record["sample_count"], dtype=np.float64) / record["sample_rate_hz"]
    base = 2 * np.pi * record["frequency_hz"] * t
    before = np.arange(record["sample_count"]) < record["change_sample"]
    signal = np.zeros(record["sample_count"], np.float64)
    for harmonic, phase in enumerate(record["phase_radians"], 1):
        gain = np.where(before, record["before"][harmonic-1], record["after"][harmonic-1])
        signal += gain * np.sin(harmonic * base + phase)
    result = (record["gain"] * signal).astype("<f4")
    _require(np.isfinite(result).all(), "Nonfinite challenge PCM")
    return result


def _extract_feature(raw, full):
    subprocess.run(["node", "--import", "tsx", "../../research/voice-prosody/replay.ts", "--feature-version", "v2",
                    "--input-f32", str(raw), "--sample-rate", "48000", "--output", str(full)],
                   cwd=REPO / "apps/capture-web", check=True, capture_output=True, text=True)
    value = json.loads(full.read_text())
    _require(value.get("inputSha256") == _sha(raw) and value.get("inputSampleCount") == 96000
             and value.get("featureVersion") == "v2", "Replay input identity differs")
    frames = value["frames"]
    hz = np.array([[c["hz"] if c["valid"] else np.nan for c in f["features"]["candidates"]] for f in frames], np.float64)
    feature = {"features": np.array([f["features"]["values"] for f in frames], np.float32),
               "candidate_hz": hz, "candidate_mask": np.isfinite(hz),
               "reference_sample": np.array([f["referenceSample"] for f in frames], np.int64),
               "feature_names": np.array(value["featureNames"]), "sample_rate_hz": np.array(48000),
               "pcm_sample_count": np.array(96000)}
    validate_feature(feature, {"sample_count": 96000})
    return feature


def _scenario(record):
    change = record["change_sample"] / record["sample_rate_hz"]
    return {"id": record["id"], "kind": "observation-change", "durationSeconds": 2.,
            "initialHz": record["frequency_hz"], "finalHz": record["frequency_hz"],
            "changeStartSeconds": change, "changeEndSeconds": change, "slopeHzPerSecond": 0,
            "change_sample": record["change_sample"], "before": record["before"], "after": record["after"],
            "phase_radians": record["phase_radians"], "gain": record["gain"]}


def challenge_region_at(scenario, end_sample):
    """Classify exclusive support by the declared integer transition, never .6s."""
    change = scenario["change_sample"]
    if end_sample <= change:
        return "before-observation-change"
    if end_sample - 1920 >= change:
        return "after-observation-change"
    return "observation-boundary"


def generate_challenge(output, checkpoint_freeze_path, expected_checkpoint_freeze_sha256):
    """Return (rows, absolute-path hashes); no references or model outcomes used."""
    freeze_path = Path(checkpoint_freeze_path).resolve()
    _, hashes = _validate_freeze(freeze_path, expected_checkpoint_freeze_sha256)
    output = _inside(RUNS, Path(output).resolve())
    _require(not output.exists(), "Challenge generation refuses to overwrite prior evidence")
    _require(output.parent.exists() and shutil.disk_usage(output.parent).free >= 512 * 1024 * 1024, "Insufficient bounded-generation storage")
    output.mkdir()
    for name in ("pcm", "features", "intermediate", "source-snapshot"):
        (output / name).mkdir()
    spec = challenge_spec(); sources = challenge_source_hashes()
    _save(output / "spec.json", spec)
    for name, expected in sources.items():
        target = output / "source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes((REPO / name).read_bytes())
        _require(_sha(target) == expected, "Challenge source snapshot changed")
    manifest = {"schema": CACHE_SCHEMA, "status": "generating", "started_at": _now(), "spec_sha256": challenge_spec_sha256(),
                "checkpoint_freeze_path": str(freeze_path.relative_to(REPO)), "checkpoint_freeze_sha256": expected_checkpoint_freeze_sha256,
                "source_sha256": sources, "records": [], "reference_content_read": False,
                "feature_input": "only finite float32 PCM and sample rate enter unchanged TypeScript replay",
                "maximum_output_bytes": MAX_BYTES}
    _save(output / "manifest.json", manifest)
    try:
        for record in spec["records"]:
            raw = output / "pcm" / (record["id"] + ".f32le")
            full = output / "intermediate" / (record["id"] + ".json")
            _synthesize_pcm(record).tofile(raw)
            feature = _extract_feature(raw, full)
            feature_path = output / "features" / (record["id"] + ".npz")
            np.savez_compressed(feature_path, **feature)
            export_sha = _sha(full); full.unlink()
            manifest["records"].append({"id": record["id"], "parameters_sha256": hashlib.sha256(_canonical(record)).hexdigest(),
                "pcm_path": str(raw.relative_to(output)), "pcm_sha256": _sha(raw), "feature_path": str(feature_path.relative_to(output)),
                "feature_sha256": _sha(feature_path), "typescript_export_sha256": export_sha, "feature_frames": len(feature["features"])})
            _require(sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) <= MAX_BYTES, "Challenge output cap exceeded")
            _save(output / "manifest.json", manifest)
        (output / "intermediate").rmdir()
        _validate_freeze(freeze_path, expected_checkpoint_freeze_sha256)
        _require(challenge_source_hashes() == sources, "Challenge source changed during generation")
        manifest.update(status="complete", completed_at=_now())
        _save(output / "manifest.json", manifest)
    except BaseException as error:
        manifest.update(status="failed", failed_at=_now(), failure=repr(error)); _save(output / "manifest.json", manifest)
        raise
    return load_challenge(output, freeze_path, expected_checkpoint_freeze_sha256)


def load_challenge(output, checkpoint_freeze_path, expected_checkpoint_freeze_sha256):
    output = _inside(RUNS, Path(output).resolve())
    _, hashes = _validate_freeze(checkpoint_freeze_path, expected_checkpoint_freeze_sha256)
    manifest_path = output / "manifest.json"; manifest = json.loads(manifest_path.read_text())
    spec = challenge_spec()
    _require(json.loads((output / "spec.json").read_text()) == spec, "Cached challenge specification changed")
    _require(manifest.get("schema") == CACHE_SCHEMA and manifest.get("status") == "complete"
             and manifest.get("spec_sha256") == challenge_spec_sha256()
             and manifest.get("checkpoint_freeze_sha256") == expected_checkpoint_freeze_sha256
             and manifest.get("source_sha256") == challenge_source_hashes(), "Challenge cache provenance differs")
    _require(len(manifest["records"]) == 12 and [r["id"] for r in manifest["records"]] == [r["id"] for r in spec["records"]], "Challenge cache inventory differs")
    rows = []
    for record, saved in zip(spec["records"], manifest["records"]):
        _require(saved["parameters_sha256"] == hashlib.sha256(_canonical(record)).hexdigest(), "Challenge parameters changed")
        for path_key, hash_key in (("pcm_path", "pcm_sha256"), ("feature_path", "feature_sha256")):
            path = _inside(output, saved[path_key]); _require(_sha(path) == saved[hash_key], "Challenge input bytes changed")
            hashes[str(path)] = saved[hash_key]
        with np.load(_inside(output, saved["feature_path"]), allow_pickle=False) as data:
            feature = {k: data[k].copy() for k in data.files}
        timing = validate_feature(feature, {"sample_count": record["sample_count"]})
        _require(len(feature["features"]) == saved["feature_frames"], "Challenge frame inventory changed")
        rows.append({"id": record["id"], "scenario": _scenario(record), "timing": timing, "feature": feature})
    hashes[str(manifest_path)] = _sha(manifest_path); hashes[str(output / "spec.json")] = _sha(output / "spec.json")
    return rows, hashes
