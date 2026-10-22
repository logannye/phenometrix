"""Descriptive scorer-only strata for the frozen 24-recording v2 pilot."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .metrics import cents
from .pilot import RESEARCH, references_from_array


def stratify(pilot: Path, data_root: Path):
    manifest = json.loads((pilot / "manifest.json").read_text())
    scores = json.loads((pilot / "scores.json").read_text())
    if manifest["status"] != "complete" or manifest["feature_version"] != "v2" or manifest["split"] != "train":
        raise ValueError("Requires the completed frozen v2 training pilot")
    expected_reference = {row["recording_id"]: row["reference_sha256"] for row in scores["per_recording"]}
    source_manifest = json.loads((data_root / "manifest.json").read_text())
    reference_files = {row["recording_id"]: row for row in source_manifest["files"] if row["role"] == "ref"}
    rows = []
    for recording in manifest["recordings"]:
        name = recording["recording_id"]
        reference_path = data_root / reference_files[name]["path"]
        if reference_files[name]["split"] != "train" or hashlib.sha256(reference_path.read_bytes()).hexdigest() != expected_reference[name]:
            raise ValueError("Frozen reference identity mismatch")
        feature_path = pilot / "features" / f"{name}.npz"
        if hashlib.sha256(feature_path.read_bytes()).hexdigest() != recording["feature_sha256"]:
            raise ValueError("Frozen acquisition feature identity mismatch")
        cache = np.load(feature_path, allow_pickle=False)
        candidates = {int(sample): [float(hz) for hz, valid in zip(hz_row, mask) if valid]
                      for sample, hz_row, mask in zip(cache["reference_sample"], cache["candidate_hz"], cache["candidate_mask"])}
        references = references_from_array(np.loadtxt(reference_path), recording["sample_count"])
        for index, target in enumerate(references):
            if not (target.valid and target.eligible and target.voiced):
                continue
            adjacent = references[max(0, index - 1):index] + references[index + 1:index + 2]
            boundary = len(adjacent) != 2 or not all(row.valid and row.voiced for row in adjacent)
            slope = None if boundary else (adjacent[1].f0_hz - adjacent[0].f0_hz) / .02
            absolute_slope = None if slope is None else abs(slope)
            f0_bin = "below-100" if target.f0_hz < 100 else "100-through-200" if target.f0_hz <= 200 else "above-200"
            slope_bin = "adjacent-unvoiced-or-invalid" if boundary else (
                "0-through-200" if absolute_slope <= 200 else "200-through-500" if absolute_slope <= 500 else
                "500-through-1000" if absolute_slope <= 1000 else "above-1000")
            frequencies = candidates.get(target.sample, [])
            closest = min(frequencies, key=lambda hz: (abs(cents(hz, target.f0_hz)), hz), default=None)
            error = None if closest is None else abs(cents(closest, target.f0_hz))
            rows.append({"recording_id": name, "speaker": recording["speaker"], "sample": target.sample,
                         "time_ms": target.sample / 48, "reference_hz": target.f0_hz, "f0_bin": f0_bin,
                         "central_reference_slope_hz_per_second": slope, "slope_bin": slope_bin,
                         "adjacent_reference_boundary": boundary, "candidate_hz": frequencies,
                         "candidate_absent": closest is None, "nearest_hz": closest,
                         "nearest_absolute_cents": error, "oracle_correct_50c": error is not None and error <= 50})

    def summarize(values):
        by_speaker = {}
        for speaker in sorted({row["speaker"] for row in values}):
            subset = [row for row in values if row["speaker"] == speaker]
            by_speaker[speaker] = {"n": len(subset), "covered": sum(row["oracle_correct_50c"] for row in subset),
                                  "coverage": sum(row["oracle_correct_50c"] for row in subset) / len(subset),
                                  "candidate_absent": sum(row["candidate_absent"] for row in subset)}
        errors = [row["nearest_absolute_cents"] for row in values if row["nearest_absolute_cents"] is not None]
        failed_errors = [value for value in errors if value > 50]
        return {"n": len(values), "covered": sum(row["oracle_correct_50c"] for row in values),
            "coverage": sum(row["oracle_correct_50c"] for row in values) / len(values) if values else None,
            "candidate_absent": sum(row["candidate_absent"] for row in values),
            "median_nearest_absolute_cents_with_candidate": float(np.median(errors)) if errors else None,
            "median_failed_nearest_absolute_cents_with_candidate": float(np.median(failed_errors)) if failed_errors else None,
            "p95_nearest_absolute_cents_with_candidate": float(np.quantile(errors, .95, method="higher")) if errors else None,
            "speaker_macro_coverage_among_represented": float(np.mean([value["coverage"] for value in by_speaker.values()])) if by_speaker else None,
            "per_speaker": by_speaker}

    interior = [row for row in rows if not row["adjacent_reference_boundary"]]
    boundary = [row for row in rows if row["adjacent_reference_boundary"]]
    f0_bins = ["below-100", "100-through-200", "above-200"]
    slope_bins = ["0-through-200", "200-through-500", "500-through-1000", "above-1000"]
    report = {"schema_version": "phenometrix.natural-error-stratification.v1", "status": "complete",
        "purpose": "Descriptive frozen training errors only; not model selection or a changed scoring target",
        "reference_slope": "central difference of scorer-only labels at t-10ms and t+10ms; never an inference feature",
        "boundary_definition": "current reference is eligible valid voiced; either neighboring reference is absent, invalid or unvoiced",
        "strata_population": "F0 and slope tables exclude the separately shown adjacent-reference-boundary group",
        "oracle_definition": "closest v2 candidate to endpoint reference; absence remains failure",
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ["natural_error_stratification.py", "pilot.py", "metrics.py"]},
        "pilot_manifest_sha256": hashlib.sha256((pilot / "manifest.json").read_bytes()).hexdigest(),
        "all": summarize(rows), "reference_boundaries": summarize(boundary), "non_boundary": summarize(interior),
        "by_f0": {band: summarize([row for row in interior if row["f0_bin"] == band]) for band in f0_bins},
        "by_absolute_slope": {band: summarize([row for row in interior if row["slope_bin"] == band]) for band in slope_bins},
        "cross_strata": {f"{f0}/{slope}": summarize([row for row in interior if row["f0_bin"] == f0 and row["slope_bin"] == slope])
                         for f0 in f0_bins for slope in slope_bins}}
    report["first_failure_examples"] = {band: [row for row in rows if row["slope_bin"] == band and not row["oracle_correct_50c"]][:3]
                                         for band in [*slope_bins, "adjacent-unvoiced-or-invalid"]}
    for name, value in [("natural-strata.json", report), ("natural-strata-rows.json", rows)]:
        with (pilot / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=RESEARCH / "data" / "ptdb-tug")
    args = parser.parse_args()
    report = stratify(args.pilot.resolve(), args.data_root.resolve())
    compact = lambda rows: {key: {metric: value[metric] for metric in ["n", "coverage", "candidate_absent", "median_nearest_absolute_cents_with_candidate"]}
                            for key, value in rows.items()}
    print(json.dumps({"all": compact({"all": report["all"]}), "boundaries": compact({"boundaries": report["reference_boundaries"]}),
                      "f0": compact(report["by_f0"]), "absolute_slope": compact(report["by_absolute_slope"])}, indent=2))


if __name__ == "__main__":
    main()
