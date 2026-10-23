/** Fixed analytic support experiment. Target labels never enter DSP extraction. */
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { replayVoicePcmV1 } from "../../apps/capture-web/src/voice-replay.js";

export interface SupportScenario {
  id: string;
  kind: "constant" | "glide" | "step";
  durationSeconds: number;
  initialHz: number;
  finalHz: number;
  changeStartSeconds: number;
  changeEndSeconds: number;
  slopeHzPerSecond: number;
}

export const SUPPORT_DIAGNOSTIC_SCENARIOS: readonly SupportScenario[] = [
  ...[100, 200, 300].map(hz => ({ id: `constant-${hz}`, kind: "constant" as const,
    durationSeconds: 1.6, initialHz: hz, finalHz: hz, changeStartSeconds: 0,
    changeEndSeconds: 0, slopeHzPerSecond: 0 })),
  ...[-200, -100, -50, 50, 100, 200].map(slope => ({ id: `glide-${slope < 0 ? "down" : "up"}-${Math.abs(slope)}`,
    kind: "glide" as const, durationSeconds: 1.6, initialHz: 200 - slope / 2,
    finalHz: 200 + slope / 2, changeStartSeconds: .25, changeEndSeconds: 1.25,
    slopeHzPerSecond: slope })),
  ...[[100, 200], [200, 100]].map(([initial, final]) => ({ id: `step-${initial}-${final}`,
    kind: "step" as const, durationSeconds: 1.6, initialHz: initial, finalHz: final,
    changeStartSeconds: .6, changeEndSeconds: .6, slopeHzPerSecond: 0 }))
];

export function frequencyAt(scenario: SupportScenario, timeSeconds: number): number {
  if (scenario.kind === "constant") return scenario.initialHz;
  if (scenario.kind === "step") return timeSeconds < scenario.changeStartSeconds ? scenario.initialHz : scenario.finalHz;
  return scenario.initialHz + scenario.slopeHzPerSecond * Math.min(
    scenario.changeEndSeconds - scenario.changeStartSeconds, Math.max(0, timeSeconds - scenario.changeStartSeconds));
}

/** Exact integral of frequency in cycles, including the continuous step phase. */
export function cyclesAt(scenario: SupportScenario, timeSeconds: number): number {
  if (scenario.kind === "constant") return scenario.initialHz * timeSeconds;
  if (scenario.kind === "step") return scenario.initialHz * timeSeconds +
    (scenario.finalHz - scenario.initialHz) * Math.max(0, timeSeconds - scenario.changeStartSeconds);
  const duration = scenario.changeEndSeconds - scenario.changeStartSeconds;
  const elapsed = Math.min(duration, Math.max(0, timeSeconds - scenario.changeStartSeconds));
  return scenario.initialHz * timeSeconds + .5 * scenario.slopeHzPerSecond * elapsed ** 2 +
    scenario.slopeHzPerSecond * duration * Math.max(0, timeSeconds - scenario.changeEndSeconds);
}

export function supportDiagnosticPcm(scenario: SupportScenario, rate = 48000): Float32Array {
  return Float32Array.from({ length: Math.round(scenario.durationSeconds * rate) }, (_, sample) => {
    const phase = 2 * Math.PI * cyclesAt(scenario, sample / rate);
    return .075 * (Math.sin(phase) + .5 * Math.sin(2 * phase) + .25 * Math.sin(3 * phase));
  });
}

function main() {
  const args = process.argv.slice(2);
  if (args.length !== 2 || args[0] !== "--output") throw new Error("Usage: support_diagnostic.ts --output <fresh-directory>");
  const output = resolve(args[1]);
  mkdirSync(output, { recursive: false });
  for (const directory of ["pcm", "predictions", "analytic-targets"]) mkdirSync(join(output, directory));
  const rate = 48000;
  const digest = (bytes: Uint8Array | string) => createHash("sha256").update(bytes).digest("hex");
  const recordings: { id: string; pcmSha256: string; predictionSha256: string; sampleCount: number; frameCount: number }[] = [];
  const endpoints = new Map<string, readonly number[]>();
  // Phase 1: all methods receive microphone-like PCM only. No analytic target array exists yet.
  for (const scenario of SUPPORT_DIAGNOSTIC_SCENARIOS) {
    const pcm = supportDiagnosticPcm(scenario, rate);
    const bytes = Buffer.alloc(pcm.length * 4);
    pcm.forEach((value, i) => bytes.writeFloatLE(value, i * 4));
    writeFileSync(join(output, "pcm", `${scenario.id}.f32le`), bytes, { flag: "wx" });
    const replay = replayVoicePcmV1(pcm, { sampleRateHz: rate, featureVersion: "v2", streamId: scenario.id }, [127, 960, 3, 211]);
    const frames = replay.map(frame => {
      const baseline = frame.features.baseline;
      const voiced = baseline.f0Hz !== null && baseline.f0Confidence >= .55 && baseline.estimatorAgreement >= .7;
      return { sample: frame.referenceSample, supportStartSample: frame.supportStartSample,
        supportEndSampleExclusive: frame.supportEndSampleExclusive,
        baseline: { state: voiced ? "voiced" : "unvoiced", f0Hz: voiced ? baseline.f0Hz : null,
          rawF0Hz: baseline.f0Hz, confidence: baseline.f0Confidence, agreement: baseline.estimatorAgreement },
        candidates: frame.features.candidates.filter(candidate => candidate.valid).map(candidate => ({ hz: candidate.hz!,
          sourceMask: candidate.sourceMask, scores: candidate.confidenceBySource })) };
    });
    const prediction = JSON.stringify({ schemaVersion: "phenometrix.support-diagnostic-predictions.v1", id: scenario.id,
      sampleRateHz: rate, sampleCount: pcm.length, featureVersion: "v2", frames }) + "\n";
    writeFileSync(join(output, "predictions", `${scenario.id}.json`), prediction, { flag: "wx" });
    endpoints.set(scenario.id, frames.map(frame => frame.sample));
    recordings.push({ id: scenario.id, pcmSha256: digest(bytes), predictionSha256: digest(prediction),
      sampleCount: pcm.length, frameCount: frames.length });
  }
  writeFileSync(join(output, "prediction-freeze.json"), JSON.stringify({ completedAt: new Date().toISOString(),
    analyticTargetArraysCreated: false, recordings }, null, 2) + "\n", { flag: "wx" });
  // Phase 2: independent scorer-only analytic labels, never returned to predictors.
  for (const scenario of SUPPORT_DIAGNOSTIC_SCENARIOS) {
    const targets = endpoints.get(scenario.id)!.map(sample => {
      const end = sample / rate;
      const start = (sample - 1920) / rate;
      const midpoint = (start + end) / 2;
      const rampInterior = scenario.kind === "glide" && start >= scenario.changeStartSeconds && end < scenario.changeEndSeconds;
      const plateau = scenario.kind === "constant" || end < scenario.changeStartSeconds || start >= scenario.changeEndSeconds;
      return { sample, timeSeconds: end, eligible: sample + 1536 <= Math.round(scenario.durationSeconds * rate),
        region: rampInterior ? "changing-support" : plateau ? "plateau" : "boundary",
        endpointHz: frequencyAt(scenario, end), midpointHz: frequencyAt(scenario, midpoint),
        windowMeanHz: (cyclesAt(scenario, end) - cyclesAt(scenario, start)) / (end - start),
        slopeHzPerSecond: rampInterior ? scenario.slopeHzPerSecond : 0 };
    });
    writeFileSync(join(output, "analytic-targets", `${scenario.id}.json`), JSON.stringify({ scenario, targets }) + "\n", { flag: "wx" });
  }
  const research = dirname(fileURLToPath(import.meta.url));
  const repo = resolve(research, "../..");
  const sources = ["research/voice-prosody/support_diagnostic.ts", "apps/capture-web/src/voice-dsp.ts",
    "apps/capture-web/src/dsp-primitives.ts", "apps/capture-web/src/voice-pitch-features.ts",
    "apps/capture-web/src/voice-pitch-features-v2.ts", "apps/capture-web/src/voice-replay.ts"];
  const manifest = { schemaVersion: "phenometrix.support-diagnostic.v1", status: "predictions-and-analytic-targets-complete",
    purpose: "Known-truth support-lag diagnosis only; no candidate revision, model training or reference-data claim",
    sampleRateHz: rate, harmonicAmplitudes: [1, .5, .25], amplitude: .075, noise: "none",
    phase: "analytic-integral-at-sample-n/rate", referenceGrid: "16+10*k ms; trailing40ms support",
    primaryTarget: "endpointHz", alternateTargets: "midpointHz/windowMeanHz are diagnostic only; no timing relabeling",
    methods: ["preserved-classical-periodic-gate", "v2-candidate-oracle", "causal-YIN-pending-independent-Python-prediction"],
    scenarios: SUPPORT_DIAGNOSTIC_SCENARIOS, recordings,
    sourceSha256: Object.fromEntries(sources.map(path => [path, digest(readFileSync(join(repo, path)))])) };
  writeFileSync(join(output, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n", { flag: "wx" });
  console.log(JSON.stringify({ output, scenarios: recordings.length, frames: recordings.reduce((sum, row) => sum + row.frameCount, 0) }));
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) main();
