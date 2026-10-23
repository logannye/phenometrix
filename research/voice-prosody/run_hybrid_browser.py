"""Local sequential browser validation; never invokes hosted computation.

Run after correctness checks and model preparation. Saves an immutable plan and
source identities before the first replay. Raw telemetry remains under ignored runs/.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "research/voice-prosody"
PLAN = (("01-dsp", "dsp", 60_000), ("02-hybrid", "hybrid", 60_000),
        ("03-hybrid", "hybrid", 60_000), ("04-dsp", "dsp", 60_000),
        ("05-hybrid-soak", "hybrid", 1_800_000))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    folder = args.output.resolve()
    if not folder.is_relative_to((RESEARCH / "runs").resolve()):
        raise ValueError("Evidence must stay under ignored research runs")
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if config.get("estimator") != "phenometrix.mlp-voicing-yin-pitch.v1":
        raise ValueError("Exact frozen hybrid configuration required")
    folder.mkdir(parents=True, exist_ok=False)
    paths = list((ROOT / "apps/capture-web/src").glob("*.ts"))
    paths += [ROOT / p for p in (
        "apps/capture-web/public/voice-capture-worklet.js",
        "apps/capture-web/e2e/voice-research.spec.ts",
        "apps/capture-web/e2e/voice-hybrid.spec.ts",
        "apps/capture-web/playwright.research.config.ts",
        "apps/capture-web/vite.research.config.ts", "pnpm-lock.yaml",
        "research/voice-prosody/run_hybrid_browser.py",
        "research/voice-prosody/HYBRID_VALIDATION_PLAN.md",
        "research/voice-prosody/YIN_NUMERICAL_REPAIR_PLAN.md")]
    paths.append(config_path)
    source = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    def validate():
        changed = [p for p, expected in source.items() if sha(ROOT / p) != expected]
        if changed:
            raise ValueError("Frozen measured source changed: " + repr(changed))
    plan = [{"name": n, "method": m, "duration_ms": d, "worker_stall_at_ms": d // 3,
             "main_stall_at_ms": 2*d // 3, "worker_stall_ms": 300, "main_stall_ms": 350}
            for n, m, d in PLAN]
    manifest = {"schema": "phenometrix-hybrid-local-browser-validation.v1", "status": "planned",
        "created_at": now(), "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": source, "configuration": config, "plan": plan,
        "execution": {"location": "local", "hardware": "Apple M4 Max, 128 GiB unified memory",
            "wasm_threads": 1, "onnxruntime_web": "1.23.2", "concurrent_agent_workloads": False,
            "ordinary_desktop_and_thermal_state": "uncontrolled", "microphone": False,
            "input": "repeated two-second 200/400Hz synthetic waveform",
            "browser_heap_growth_measured": False,
            "scope": "complete shared DSP+hybrid research path; not isolated inference cost"}, "runs": []}
    write(folder / "plan.json", manifest)
    manifest.update(status="running", started_at=now())
    write(folder / "manifest.json", manifest)
    print(json.dumps({"output": str(folder), "status": "running"}), flush=True)
    try:
        for entry in plan:
            validate()
            target = folder / entry["name"]
            target.mkdir()
            env = os.environ.copy()
            env.update(VOICE_RESEARCH_OUTPUT_DIR=str(target), VOICE_RESEARCH_METHOD=entry["method"],
                VOICE_RESEARCH_MODEL_CONFIG=str(config_path),
                VOICE_RESEARCH_DURATION_MS=str(entry["duration_ms"]),
                VOICE_RESEARCH_OVERLOAD_AT_MS=str(entry["worker_stall_at_ms"]),
                VOICE_RESEARCH_OVERLOAD_DURATION_MS="300",
                VOICE_RESEARCH_MAIN_STALL_AT_MS=str(entry["main_stall_at_ms"]),
                VOICE_RESEARCH_MAIN_STALL_DURATION_MS="350", VOICE_RESEARCH_CLOCK_DIAGNOSTICS="1")
            command = ["caffeinate", "-i", "pnpm", "exec", "playwright", "test", "--config",
                "playwright.research.config.ts", "--grep", "real AudioWorklet/worker replay",
                "--workers=1", "--reporter=json"]
            row = {**entry, "started_at": now(), "command": command, "source_sha256": source,
                   "status": "running", "source_changed": []}
            write(target / "manifest.json", row)
            print(json.dumps({"run": entry["name"], "status": "running"}), flush=True)
            with (target / "playwright-report.json").open("x") as stream:
                result = subprocess.run(command, cwd=ROOT / "apps/capture-web", env=env,
                                        stdout=stream, stderr=subprocess.STDOUT)
            validate()
            row.update(finished_at=now(), exit_code=result.returncode, source_stable=True,
                status="complete" if result.returncode == 0 else "failed",
                artifacts={p.name: sha(p) for p in target.iterdir() if p.is_file() and p.name != "manifest.json"})
            write(target / "manifest.json", row)
            manifest["runs"].append(row)
            write(folder / "manifest.json", manifest)
            print(json.dumps({"run": entry["name"], "status": "finished", "exit_code": result.returncode}), flush=True)
            if result.returncode:
                raise RuntimeError("Replay failed; no outcome is discarded: " + entry["name"])
        manifest.update(status="complete", finished_at=now())
    except BaseException as error:
        manifest.update(status="failed", finished_at=now(), error=repr(error))
        write(folder / "manifest.json", manifest)
        raise
    write(folder / "manifest.json", manifest)
    print(json.dumps({"output": str(folder), "status": "complete"}), flush=True)


if __name__ == "__main__":
    main()
