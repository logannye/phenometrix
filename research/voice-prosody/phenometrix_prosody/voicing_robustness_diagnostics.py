"""Fixed reused-development diagnostics and analytic events; input-only preparation.

No reference arrays, fitting, model selection or learned predictions are used.
The original sealed loader is unchanged: this versioned route explicitly admits
only the previously consumed 80 + 160 recordings as development regressions.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from . import evaluate as ev
from . import voicing_robustness_inputs as vi
from .hybrid_validation import extract_input
from .measurement_alignment import align_feature_timing
from .pilot import REPO, RESEARCH, digest, read_mic, source_identity, yin_predictions

GROUPS = {
    "original_development": ("delayed-development-features", "acquisition-manifest.json", "development", ("F01", "F03", "M01", "M03"), 20),
    "formerly_reserved": ("hybrid-validation-v1", "provenance/microphone-acquisition-manifest.json", "sealed_test", ("F05", "F06", "M05", "M06"), 40),
}
DEV_MANIFEST_SHA256 = "31f51e93b59f9a70c6f4b8c66cbf20b84fef900e44c75b8134aefad0cade3e8c"
RATE = 48000


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _load(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _bound(path):
    return {"path": str(path.resolve().relative_to(REPO)), "sha256": digest(path)}


def selected_sources(manifest, acquisition, *, group, reference_acquisition=None):
    """Metadata-only exact historical membership; no rank by new difficulty."""
    _require(group in GROUPS, "Unknown diagnostic source group")
    _, _, split, speakers, per_speaker = GROUPS[group]
    _require(manifest.get("status") == "complete", "Completed historical cache required")
    rows = manifest.get("recordings", [])
    _require(len(rows) == len(speakers) * per_speaker * len(vi.ANCHORS), "Historical view count differs")
    records = {}
    for row in rows:
        source = row["source_recording_id"]
        _require(row["recording_id"] == source + "__" + row["condition"], "Historical source/condition identity differs")
        _require(row["condition"] in vi.ANCHORS and row["speaker"] in speakers, "Unexpected historical source/condition")
        entry = records.setdefault(source, {"source_recording_id": source, "speaker": row["speaker"],
            "sample_count": row["sample_count"], "microphone_sha256": row["microphone_sha256"], "conditions": set()})
        _require(row["condition"] not in entry["conditions"], "Duplicate historical condition")
        _require(all(entry[k] == row[k] for k in ("speaker", "sample_count", "microphone_sha256")), "Historical source binding differs between views")
        entry["conditions"].add(row["condition"])
    _require(all(row["conditions"] == set(vi.ANCHORS) for row in records.values()), "Incomplete historical condition set")
    _require(all(sum(r["speaker"] == s for r in records.values()) == per_speaker for s in speakers), "Historical speaker counts differ")
    mics = {}
    for row in acquisition["files"]:
        if row["role"] != "mic" or row["recording_id"] not in records:
            continue
        _require(row["recording_id"] not in mics, "Duplicate acquired microphone role")
        mics[row["recording_id"]] = row
    _require(set(mics) == set(records), "Missing acquired microphone role")
    refs = {}
    for row in (reference_acquisition or acquisition)["files"]:
        if row["role"] == "ref" and row["recording_id"] in records:
            _require(row["recording_id"] not in refs, "Duplicate reference metadata role")
            refs[row["recording_id"]] = row
    _require(set(refs) == set(records), "Missing reference metadata role")
    result = []
    for identity in sorted(records):
        meta, mic = records[identity], mics[identity]
        _require(mic["split"] == split and mic["speaker"] == meta["speaker"] and mic["sha256"] == meta["microphone_sha256"], "Historical microphone role/split/hash mismatch")
        _require(mic.get("primary_eligible") is True, "Unexpected universal/ineligible prompt")
        path = Path(mic["path"])
        _require(not path.is_absolute() and ".." not in path.parts, "Microphone path escapes corpus root")
        ref = refs[identity]
        _require(ref["split"] == split and ref["speaker"] == meta["speaker"], "Reference metadata role/split mismatch")
        refpath = Path(ref["path"])
        _require(not refpath.is_absolute() and ".." not in refpath.parts, "Reference path escapes corpus root")
        result.append({k: v for k, v in meta.items() if k != "conditions"} | {"group": group, "microphone": mic, "reference_metadata": ref})
    return result


def event_specs(spec):
    """Exactly the prospective 12-event family, not data-derived expansion."""
    fixed = dict(seconds=3, harmonic_interval=[1, 2], fundamentals=[90, 180, 360], phases=[0, .7],
        amplitudes=[.1, .05, .025], backgrounds=["silence", "white_noise_sd_0.003"], seed_namespace=vi.NAMESPACE + ":events")
    _require(all(spec.get(k) == v for k, v in fixed.items()), "Frozen event construction differs")
    result = []
    for hz in fixed["fundamentals"]:
        for phase_index, phase in enumerate(fixed["phases"]):
            for background in fixed["backgrounds"]:
                identity = f"voicing-event-f{hz:03}-p{phase_index}-{background}"
                seed = int.from_bytes(hashlib.sha256(f"{fixed['seed_namespace']}:{identity}".encode()).digest()[:8], "little")
                result.append(dict(id=identity, sample_rate_hz=RATE, sample_count=3 * RATE,
                    onset_sample=RATE, offset_sample=2 * RATE, fundamental_hz=hz,
                    harmonic_amplitudes=fixed["amplitudes"], common_phase_radians=phase,
                    background=background, noise_sd=0 if background == "silence" else .003,
                    seed=seed, seed_conversion="SHA256 UTF-8 namespace:event_id; first 8 bytes little endian; NumPy PCG64",
                    phase_origin="elapsed time from tone onset; same phase added to each harmonic"))
    return result


def construct_event(spec):
    """Abrupt analytic onset/offset plus fixed full-record independent background."""
    _require(spec["sample_rate_hz"] == RATE and spec["sample_count"] == 3 * RATE
             and spec["onset_sample"] == RATE and spec["offset_sample"] == 2 * RATE,
             "Event sample geometry differs")
    _require(spec["fundamental_hz"] in (90, 180, 360) and spec["common_phase_radians"] in (0, .7)
             and spec["harmonic_amplitudes"] == [.1, .05, .025], "Event harmonic family differs")
    _require(spec["background"] in ("silence", "white_noise_sd_0.003") and
             spec["noise_sd"] == (0 if spec["background"] == "silence" else .003), "Event noise family differs")
    pcm = np.zeros(spec["sample_count"], dtype=np.float64)
    if spec["noise_sd"]:
        pcm += np.random.default_rng(spec["seed"]).standard_normal(len(pcm)) * spec["noise_sd"]
    local_time = np.arange(RATE, dtype=np.float64) / RATE
    tone = np.zeros(RATE, dtype=np.float64)
    for harmonic, amplitude in enumerate(spec["harmonic_amplitudes"], 1):
        tone += amplitude * np.sin(2 * np.pi * harmonic * spec["fundamental_hz"] * local_time + spec["common_phase_radians"])
    pcm[RATE:2 * RATE] += tone
    pcm = pcm.astype("<f4")
    return pcm, {"parameters": spec, "arithmetic": "float64 then little-endian float32",
        "sample_count": len(pcm), "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
        "outside_unit_count": int((np.abs(pcm) > 1).sum()), "minimum": float(pcm.min()), "maximum": float(pcm.max()),
        "noise_scaling": "fixed sd 0.003 without realized-RMS renormalization", "saturation": False}


def prepare_pcm(pcm, *, identity, output):
    """One real TS feature/YIN path shared by synthetic tests and all records."""
    pcm = np.asarray(pcm, dtype="<f4")
    _require(pcm.ndim == 1 and len(pcm) >= 2208 and np.isfinite(pcm).all(), "Finite full-recording PCM required")
    raw = output / "pcm" / (identity + ".f32le")
    _require(not raw.exists(), "Refuse PCM overwrite")
    pcm.tofile(raw)
    feature, dsp, binding = extract_input(pcm, identity, output / "intermediate")
    _require(binding["transformed_pcm_sha256"] == digest(raw), "TS input differs from stored PCM")
    timing = align_feature_timing(feature, measurement_delay_ms=20)
    yin, evidence = yin_predictions(pcm, feature["reference_sample"])
    _require(len(evidence) == len(timing.reference_sample), "Raw YIN feature extent differs")
    _require([r["sample"] for r in evidence] == timing.availability_sample.tolist(), "Raw YIN support grid differs")
    baselines = dict(dsp=dsp, yin=ev.delayed_baseline([asdict(r) for r in yin], feature),
                     continuity=ev.continuity_predictions(feature))
    for name, rows in baselines.items():
        _require([r["sample"] for r in rows] == timing.reference_sample.tolist(), "Baseline target grid differs: " + name)
    feature_path = output / "features" / (identity + ".npz")
    np.savez_compressed(feature_path, **feature)
    baseline_path = output / "baselines" / (identity + ".json.gz")
    ev.write_predictions(baseline_path, dict(recording_id=identity, time_contract=timing.metadata(),
        availability_sample=timing.availability_sample.tolist(), predictions=baselines, yin_evidence=evidence))
    return dict(sample_count=len(pcm), feature_path=str(feature_path.relative_to(output)), feature_sha256=digest(feature_path),
        baseline_path=str(baseline_path.relative_to(output)), baseline_sha256=digest(baseline_path),
        pcm_path=str(raw.relative_to(output)), pcm_sha256=digest(raw), **binding)


def _sources():
    source = dict(source_identity()["source_sha256"])
    for name in ("voicing_robustness_diagnostics", "voicing_robustness_inputs", "hybrid_validation", "hybrid_followup", "measurement_alignment", "evaluate"):
        p = Path(__file__).with_name(name + ".py")
        source[str(p.relative_to(REPO))] = digest(p)
    return source


def prepare(plan_folder: Path, output: Path | None = None):
    plan_folder = plan_folder.resolve()
    plan_path = plan_folder / "plan.json"
    plan = _load(plan_path)
    _require(plan.get("schema") == "phenometrix-voicing-robustness-plan.v1" and plan.get("status") == "frozen", "Frozen robustness plan required")
    _require(plan["recipe_sha256"] == vi.canonical_sha(vi.bank_recipe()), "Frozen acoustic bank recipe differs")
    _require(plan.get("all_ptdb_exposed") is True and plan["recipe"]["diagnostics"] == list(vi.DIAGNOSTICS), "Consumed source/diagnostic contract absent")
    events = event_specs(plan["voicing_event_spec"])
    source_rows, acquisitions = [], {}
    for group, (folder, acquisition_name, _, _, _) in GROUPS.items():
        base = RESEARCH / "runs" / folder
        manifest_path, acquisition_path = base / "manifest.json", base / acquisition_name
        manifest, acq = _load(manifest_path), _load(acquisition_path)
        expected = manifest["data_manifest_sha256"] if group == "original_development" else manifest["microphone_manifest_sha256"]
        _require(digest(acquisition_path) == expected, "Historical microphone acquisition manifest changed")
        if group == "original_development":
            _require(digest(manifest_path) == DEV_MANIFEST_SHA256, "Frozen development cache identity differs")
        ref_acq_path = acquisition_path if group == "original_development" else base / "provenance/scoring-acquisition-manifest.json"
        ref_acq = acq if group == "original_development" else _load(ref_acq_path)
        if group == "formerly_reserved":
            _require(digest(ref_acq_path) == manifest["scoring_acquisition_manifest_sha256"], "Reference acquisition metadata changed")
        source_rows.extend(selected_sources(manifest, acq, group=group, reference_acquisition=ref_acq))
        acquisitions[group] = dict(manifest=_bound(manifest_path), acquisition=_bound(acquisition_path), reference_metadata=_bound(ref_acq_path))
    _require(len(source_rows) == 240 and len({r['source_recording_id'] for r in source_rows}) == 240, "Consumed diagnostic membership differs")
    source = _sources()
    output = output or plan_folder.parent / "diagnostic-inputs"
    output.mkdir(parents=True, exist_ok=False)
    for name in ("pcm", "features", "baselines", "intermediate", "source-snapshot"):
        (output / name).mkdir()
    declaration = dict(schema="phenometrix-voicing-robustness-diagnostics-declaration.v1", at=_now(),
        plan=_bound(plan_path), source_sha256=source, acquisitions=acquisitions, selected_sources=source_rows,
        diagnostics=list(vi.DIAGNOSTICS), events=events, original_development_source_count=80,
        formerly_reserved_source_count=160, natural_view_count=1440, synthetic_event_count=12,
        references_opened=False, external_corpus_used=False, for_fitting=False,
        interpretation="All natural sources previously consumed; descriptive development/regression only")
    _write(output / "declaration.json", declaration)
    for name, sha in source.items():
        dest = output / "source-snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, dest)
        _require(digest(dest) == sha, "Source snapshot changed during copy")
    manifest = dict(schema="phenometrix-voicing-robustness-diagnostic-inputs.v1", status="running", started_at=_now(),
        declaration_sha256=digest(output / "declaration.json"), plan_sha256=digest(plan_path),
        source_sha256=source, references_opened=False, external_corpus_used=False,
        learned_model_inference=False, records=[])
    _write(output / "manifest.json", manifest)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(RESEARCH / "runs/.numba-cache"))
    began = time.monotonic()
    try:
        for index, row in enumerate(source_rows):
            mic = RESEARCH / "data/ptdb-tug" / row["microphone"]["path"]
            _require(digest(mic) == row["microphone_sha256"], "Historical microphone bytes changed")
            clean = read_mic(mic)
            _require(len(clean) == row["sample_count"], "Historical microphone sample count changed")
            for condition in vi.DIAGNOSTICS:
                identity = row["source_recording_id"] + "__diagnostic_" + condition
                pcm, construction = vi.construct(clean, vi.diagnostic_draw(row["source_recording_id"], condition))
                artifacts = prepare_pcm(pcm, identity=identity, output=output)
                _require(artifacts["pcm_sha256"] == construction["pcm_sha256"], "Diagnostic construction bytes changed")
                reference_path = RESEARCH / "data/ptdb-tug" / row["reference_metadata"]["path"]
                manifest["records"].append(dict(kind="natural_diagnostic", recording_id=identity, id=identity,
                    source_recording_id=row["source_recording_id"], source_id=row["source_recording_id"],
                    speaker=row["speaker"], cohort=row["group"], group="diagnostic",
                    condition=condition, microphone_sha256=row["microphone_sha256"], construction=construction,
                    feature=_bound(output / artifacts["feature_path"]), baseline=_bound(output / artifacts["baseline_path"]),
                    reference=dict(path=str(reference_path.relative_to(REPO)), sha256=row["reference_metadata"]["sha256"]), **artifacts))
            if index % 5 == 0 or index == len(source_rows) - 1:
                _write(output / "manifest.json", manifest)
                print(json.dumps(dict(phase="diagnostic-inputs", sources=index + 1, total=240,
                    records=len(manifest["records"]), elapsed_seconds=time.monotonic() - began)), flush=True)
                _require(shutil.disk_usage(REPO).free >= plan["limits"]["minimum_free_bytes"], "Free disk floor reached")
                size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
                _require(size <= plan["limits"]["new_artifact_bytes"], "Diagnostic artifact cap reached")
        for spec in events:
            pcm, construction = construct_event(spec)
            artifacts = prepare_pcm(pcm, identity=spec["id"], output=output)
            manifest["records"].append(dict(kind="synthetic_event", recording_id=spec["id"], id=spec["id"],
                source_recording_id=spec["id"], source_id=spec["id"], speaker=None, group="events", cohort="synthetic_event",
                condition=spec["background"], event=spec, definition={"frequency_hz": spec["fundamental_hz"], **spec},
                construction=construction, feature=_bound(output / artifacts["feature_path"]),
                baseline=_bound(output / artifacts["baseline_path"]), **artifacts))
        _require(len(manifest["records"]) == 1452, "Diagnostic/event output inventory differs")
        for name, sha in source.items():
            _require(digest(REPO / name) == sha, "Preparation source changed: " + name)
        _require(digest(plan_path) == manifest["plan_sha256"], "Study plan changed")
        manifest.update(status="complete", completed_at=_now(), elapsed_seconds=time.monotonic() - began)
        _write(output / "manifest.json", manifest)
        return manifest
    except BaseException as error:
        manifest.update(status="failed", failed_at=_now(), failure=repr(error), elapsed_seconds=time.monotonic() - began)
        _write(output / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = prepare(args.plan, args.output)
    print(json.dumps(dict(status=result["status"], records=len(result["records"]))))


if __name__ == "__main__":
    main()
