"""Score fixed analytic source changes without changing their endpoint target."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from .metrics import cents
from .pilot import RESEARCH, yin_predictions


def write_json(path: Path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def summarize(rows):
    emitted = [row for row in rows if row["estimate_hz"] is not None]
    errors = np.array([row["signed_cents"] for row in emitted], dtype=float)
    correct = sum(abs(row["signed_cents"]) <= 50 for row in emitted)
    delays = [row["equivalent_linear_slope_delay_ms"] for row in emitted
              if row["equivalent_linear_slope_delay_ms"] is not None]
    return {"n": len(rows), "emitted": len(emitted),
            "pitch_coverage": len(emitted) / len(rows) if rows else None,
            "correct_50c_fraction_all": correct / len(rows) if rows else None,
            "median_absolute_cents": float(np.median(np.abs(errors))) if len(errors) else None,
            "p95_absolute_cents": float(np.quantile(np.abs(errors), .95, method="higher")) if len(errors) else None,
            "mean_signed_cents": float(np.mean(errors)) if len(errors) else None,
            "median_signed_cents": float(np.median(errors)) if len(errors) else None,
            "median_equivalent_linear_slope_delay_ms": float(np.median(delays)) if delays else None,
            "expected_lag_sign_fraction": sum(row["signed_cents"] * row["slope_hz_per_second"] < 0
                for row in emitted if row["slope_hz_per_second"] != 0) /
                sum(row["slope_hz_per_second"] != 0 for row in emitted)
                if any(row["slope_hz_per_second"] != 0 for row in emitted) else None}


def score_diagnostic(run: Path):
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["schemaVersion"] != "phenometrix.support-diagnostic.v1":
        raise ValueError("Unexpected diagnostic manifest")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(RESEARCH / "runs" / ".numba-cache"))
    yin_dir = run / "yin-predictions"
    yin_dir.mkdir(exist_ok=False)
    # All YIN predictions complete before opening any analytic-target file.
    for recording in manifest["recordings"]:
        name = recording["id"]
        raw = (run / "pcm" / f"{name}.f32le").read_bytes()
        if hashlib.sha256(raw).hexdigest() != recording["pcmSha256"]:
            raise ValueError("Synthetic PCM digest mismatch")
        pcm = np.frombuffer(raw, dtype="<f4")
        prediction_path = run / "predictions" / f"{name}.json"
        if hashlib.sha256(prediction_path.read_bytes()).hexdigest() != recording["predictionSha256"]:
            raise ValueError("DSP prediction digest mismatch")
        acquired = json.loads(prediction_path.read_text())
        targets = np.array([frame["sample"] for frame in acquired["frames"]], dtype=np.int64)
        predictions, evidence = yin_predictions(pcm, targets)
        write_json(yin_dir / f"{name}.json", {"predictions": [asdict(value) for value in predictions], "evidence": evidence})
    write_json(run / "yin-prediction-freeze.json", {"analytic_target_files_opened": False,
        "files_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(yin_dir.glob("*.json"))}})
    traces = []
    scenarios = []
    slope_summaries = []
    for recording in manifest["recordings"]:
        name = recording["id"]
        truth = json.loads((run / "analytic-targets" / f"{name}.json").read_text())
        acquired = json.loads((run / "predictions" / f"{name}.json").read_text())
        yin = json.loads((yin_dir / f"{name}.json").read_text())
        prediction_by_sample = {frame["sample"]: frame for frame in acquired["frames"]}
        yin_by_sample = {frame["sample"]: frame for frame in yin["predictions"]}
        local = []
        for target in truth["targets"]:
            if not target["eligible"]:
                continue
            frame = prediction_by_sample[target["sample"]]
            for target_name in ["endpointHz", "midpointHz", "windowMeanHz"]:
                reference = target[target_name]
                oracle = min((candidate["hz"] for candidate in frame["candidates"]),
                             key=lambda value: (abs(cents(value, reference)), value), default=None)
                estimates = {"legacy_dsp": frame["baseline"]["f0Hz"],
                             "v2_oracle": oracle,
                             "causal_yin": yin_by_sample[target["sample"]]["f0_hz"]}
                for method, estimate in estimates.items():
                    slope = target["slopeHzPerSecond"]
                    row = {"scenario": name, "sample": target["sample"], "time_ms": target["timeSeconds"] * 1000,
                           "region": target["region"], "slope_hz_per_second": slope,
                           "method": method, "target": target_name, "target_hz": reference,
                           "estimate_hz": estimate, "signed_cents": None if estimate is None else cents(estimate, reference),
                           "equivalent_linear_slope_delay_ms": None if estimate is None or slope == 0 else
                               (reference - estimate) / slope * 1000,
                           "target_aware_oracle_bound": method == "v2_oracle"}
                    local.append(row)
        summaries = {method: {target: {region: summarize([row for row in local if row["method"] == method and
            row["target"] == target and (region == "all" or row["region"] == region)])
            for region in ["all", "plateau", "changing-support", "boundary"]}
            for target in ["endpointHz", "midpointHz", "windowMeanHz"]}
            for method in ["legacy_dsp", "v2_oracle", "causal_yin"]}
        response = None
        if truth["scenario"]["kind"] == "step":
            step_ms = truth["scenario"]["changeStartSeconds"] * 1000
            response = {}
            for method in ["legacy_dsp", "v2_oracle", "causal_yin"]:
                after = [row for row in local if row["method"] == method and row["target"] == "endpointHz" and row["time_ms"] >= step_ms]
                correct = [row["signed_cents"] is not None and abs(row["signed_cents"]) <= 50 for row in after]
                first = next((i for i, value in enumerate(correct) if value), None)
                settled = next((i for i in range(len(correct) - 2) if all(correct[i:i+3])), None)
                response[method] = {"first_correct_after_step_ms": None if first is None else after[first]["time_ms"] - step_ms,
                    "first_of_three_consecutive_correct_ms": None if settled is None else after[settled]["time_ms"] - step_ms,
                    "declaration": "50 cents; three consecutive outputs; grid quantization retained; oracle is unattainable selection ceiling"}
        if truth["scenario"]["kind"] == "glide":
            slope_summaries.append({"scenario": name, "slope_hz_per_second": truth["scenario"]["slopeHzPerSecond"],
                "methods": {method: summaries[method]["endpointHz"]["changing-support"] for method in summaries}})
        scenarios.append({"scenario": truth["scenario"], "summaries": summaries, "step_response": response})
        traces.extend(local)
    report = {"schema_version": "phenometrix.support-diagnostic-scores.v1", "status": "complete",
        "evidence": "Fixed noiseless synthetic signals with exact analytic phase; no natural-speech or trained-model claim",
        "primary_target": "instantaneous frequency at exclusive support endpoint",
        "alternate_target_warning": "Midpoint/window-mean results diagnose support integration; they do not change primary scoring or authorize output relabeling",
        "oracle_warning": "The v2 oracle selects using each analysis target; it is a candidate ceiling, not an executable estimator",
        "source_sha256": {"support_diagnostic.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          "pilot.py": hashlib.sha256(Path(__file__).with_name("pilot.py").read_bytes()).hexdigest()},
        "scenario_count": len(scenarios), "duration_seconds": sum(row["scenario"]["durationSeconds"] for row in scenarios),
        "scenarios": scenarios, "slope_bins": slope_summaries,
        "overall": {method: {target: {region: summarize([row for row in traces if row["method"] == method and
            row["target"] == target and (region == "all" or row["region"] == region)])
            for region in ["all", "plateau", "changing-support", "boundary"]}
            for target in ["endpointHz", "midpointHz", "windowMeanHz"]}
            for method in ["legacy_dsp", "v2_oracle", "causal_yin"]}}
    write_json(run / "scored-traces.json", traces)
    write_json(run / "scores.json", report)
    write_json(run / "evaluation-manifest.json", {"status": "complete",
        "scores_sha256": hashlib.sha256((run / "scores.json").read_bytes()).hexdigest(),
        "traces_sha256": hashlib.sha256((run / "scored-traces.json").read_bytes()).hexdigest(),
        "source_sha256": report["source_sha256"]})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    report = score_diagnostic(args.run.resolve())
    print(json.dumps({"scenario_count": report["scenario_count"], "duration_seconds": report["duration_seconds"],
        "changing_support_endpoint": {method: targets["endpointHz"]["changing-support"] for method, targets in report["overall"].items()},
        "step_responses": {row["scenario"]["id"]: row["step_response"] for row in report["scenarios"] if row["step_response"]}}, indent=2))


if __name__ == "__main__":
    main()
