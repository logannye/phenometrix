import { expect, test } from "@playwright/test";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import type { VoiceResearchModelConfig, VoiceResearchPrediction } from "../src/voice-research-hook.js";
const configPath = process.env.VOICE_RESEARCH_MODEL_CONFIG;
const output = (name: string, fallback: string) => process.env.VOICE_RESEARCH_OUTPUT_DIR ? resolve(process.env.VOICE_RESEARCH_OUTPUT_DIR, name) : fallback;
const sha = (bytes: string | Buffer) => createHash("sha256").update(bytes).digest("hex");
function configuration() {
  if (!configPath) throw new Error("Explicit hybrid configuration path required.");
  const text = readFileSync(resolve(configPath), "utf8");
  const config = JSON.parse(text) as VoiceResearchModelConfig;
  if (config.estimator !== "phenometrix.mlp-voicing-yin-pitch.v1") throw new Error("Frozen hybrid contract required.");
  return { config, configDocument: { text, sha256: sha(text) } };
}

test("frozen hybrid bounded PCM hook matches Python raw YIN and NumPy including development speech", async ({ page }, testInfo) => {
  test.skip(!configPath, "Provide explicit VOICE_RESEARCH_MODEL_CONFIG for the research-only hybrid.");
  test.setTimeout(180_000);
  const configured = configuration();
  const path = resolve(process.env.VOICE_HYBRID_PARITY_MANIFEST ?? "../../research/voice-prosody/runs/hybrid-browser-parity-v2/manifest.json");
  const manifestBytes = readFileSync(path);
  const manifest = JSON.parse(manifestBytes.toString());
  expect(manifest.schema).toBe("phenometrix.hybrid-browser-parity.v1");
  expect(manifest.estimatorContract).toBe(configured.config.estimator);
  expect(manifest.sampleRateHz).toBe(48000);
  expect(manifest.records.length).toBeLessThanOrEqual(64);
  await page.goto("/research-voice.html");
  await page.waitForFunction(() => typeof window.runVoiceHybridPcmParity === "function");
  const failures: Array<Record<string, unknown>> = [];
  const boundaryExceptions: Array<Record<string, unknown>> = [];
  const records: Array<Record<string, unknown>> = [];
  let frameCount = 0;
  let maximumRawYinErrorHz = 0;
  const close = (a: number, b: number, atol = 1e-6, rtol = 1e-5) => Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= atol + rtol * Math.abs(b);
  for (const record of manifest.records) {
    const pcmPath = resolve(dirname(path), record.pcmPath);
    if (!pcmPath.startsWith(dirname(path) + "/")) throw new Error("Parity PCM escapes fixture directory.");
    const pcm = readFileSync(pcmPath);
    expect(sha(pcm)).toBe(record.pcmSha256);
    expect(pcm.byteLength).toBeLessThanOrEqual(960000 * 4);
    const result = await page.evaluate(options => window.runVoiceHybridPcmParity(options), { ...configured,
      pcmBytes: Array.from(pcm), pcmSha256: record.pcmSha256, windowEndSamples: record.windowEndSamples });
    records.push({ id: record.id, ...result });
    if (result.status !== "complete") { failures.push({ id: record.id, reason: result.reason }); continue; }
    const frames = result.frames as VoiceResearchPrediction[];
    if (frames.length !== record.frames.length) failures.push({ id: record.id, reason: "frame-count" });
    for (let i = 0; i < frames.length; i += 1) {
      frameCount += 1;
      const actual = frames[i], expected = record.frames[i];
      const mismatch = (reason: string, detail?: unknown) => failures.push({ id: record.id, index: i, reason, detail });
      if (actual.referenceSample !== expected.referenceSample || actual.supportEndSampleExclusive !== expected.supportEndSampleExclusive ||
          actual.supportStartSample !== expected.supportEndSampleExclusive - 1920 || actual.lastRequiredInputSample !== expected.supportEndSampleExclusive - 1) mismatch("support");
      if (actual.configurationSha256 !== configured.configDocument.sha256 || actual.checkpointSha256 !== manifest.weightsSha256 ||
          actual.modelManifestSha256 !== configured.config.manifestSha256 || actual.modelSha256 !== configured.config.modelSha256 ||
          actual.estimatorContract !== manifest.estimatorContract) mismatch("identity");
      if (actual.correctnessProbability !== null || actual.candidateIndex !== null || actual.candidateHeadRole !== "diagnostic-only-unused-by-hybrid") mismatch("hybrid-semantics");
      if (!close(actual.voicingLogit, expected.voicingLogit) || !close(actual.voicingProbability, expected.voicingProbability, 1e-6, 0)) mismatch("voicing", { actual: actual.voicingLogit, expected: expected.voicingLogit });
      if (actual.featureValues.length !== 57 || actual.featureValues.some((value, j) => !close(value, expected.featureValues[j]))) mismatch("features");
      if (actual.candidateLogits.length !== 6 || actual.candidateLogits.some((value, j) => !close(value, expected.candidateLogits[j]))) mismatch("candidate-logits");
      if (JSON.stringify(actual.candidateMask) !== JSON.stringify(expected.candidateMask)) mismatch("candidate-mask");
      const raw = actual.rawYin?.rawF0Hz;
      const error = raw === undefined ? Infinity : Math.abs(raw - expected.rawYinHz);
      maximumRawYinErrorHz = Math.max(maximumRawYinErrorHz, error);
      const tolerance = Math.max(.01, 1e-4 * Math.abs(expected.rawYinHz));
      if (!Number.isFinite(error) || error > tolerance) mismatch("raw-yin", { actual: raw, expected: expected.rawYinHz, error, tolerance });
      const stateDiffers = actual.state !== expected.state || (actual.f0Hz === null) !== (expected.f0Hz === null);
      if (stateDiffers) {
        const detail = { id: record.id, index: i, actual: actual.state, expected: expected.state,
          actualRawYinHz: raw, expectedRawYinHz: expected.rawYinHz, probability: expected.voicingProbability };
        if (expected.nearBoundary) boundaryExceptions.push(detail); else mismatch("hybrid-state", detail);
      }
      if (actual.f0Hz !== null && expected.f0Hz !== null && Math.abs(actual.f0Hz - expected.f0Hz) > tolerance) mismatch("final-pitch");
    }
  }
  const result = { status: failures.length ? "failed" : boundaryExceptions.length ? "boundary-exceptions" : "complete",
    evidence: "bounded-pcm-hook-arithmetic-parity-not-paced-audio", manifestSha256: sha(manifestBytes),
    configurationSha256: configured.configDocument.sha256, estimatorContract: manifest.estimatorContract,
    frameCount, maximumRawYinErrorHz, failures, boundaryExceptions, records, timingBenchmark: false };
  const target = output("browser-hybrid-pcm-parity.json", testInfo.outputPath("browser-hybrid-pcm-parity.json"));
  mkdirSync(dirname(target), { recursive: true }); writeFileSync(target, JSON.stringify(result) + "\n", { flag: "wx" });
  expect(failures, JSON.stringify(failures.slice(0, 20))).toHaveLength(0);
  // Boundary exceptions remain visible and do not silently become an unconditional pass.
  expect(boundaryExceptions, JSON.stringify(boundaryExceptions)).toHaveLength(0);
});

test("frozen hybrid runs actual worklet 300 Hz regression with explicit uncertainty", async ({ page }, testInfo) => {
  test.skip(!configPath, "Provide explicit VOICE_RESEARCH_MODEL_CONFIG for the research-only hybrid.");
  test.setTimeout(45_000);
  const configured = configuration();
  await page.goto("/research-voice.html");
  await page.waitForFunction(() => typeof window.runVoiceResearchReplay === "function");
  const result = await page.evaluate(options => window.runVoiceResearchReplay({ durationMs: 2200,
    syntheticTone: "300hz-harmonic", modelConfig: options.config, modelConfigDocument: options.configDocument,
    retainedFrameLimit: 300, retainedBlockLimit: 150 }), configured);
  const target = output("browser-hybrid-300hz.json", testInfo.outputPath("browser-hybrid-300hz.json"));
  mkdirSync(dirname(target), { recursive: true }); writeFileSync(target, JSON.stringify({ ...result, timingBenchmark: false }) + "\n", { flag: "wx" });
  expect(result.status, JSON.stringify(result.errors)).toBe("complete");
  expect(result.workerStopAcknowledged).toBe(true); expect(result.transportStop.status).toBe("complete");
  expect(result.summary.researchFrameCount).toBe(result.summary.frameCount);
  expect(Number(result.summary.researchVoicedFrameCount)).toBeGreaterThan(100);
  expect(result.summary.researchCorrectKnownPitchFrameCount).toBe(result.summary.researchVoicedFrameCount);
  for (const row of result.frames) {
    const frame = row.research as VoiceResearchPrediction;
    expect(frame.correctnessProbability).toBeNull(); expect(frame.candidateIndex).toBeNull();
    expect(frame.estimatorContract).toBe("phenometrix.mlp-voicing-yin-pitch.v1");
    expect(frame.configurationSha256).toBe(configured.configDocument.sha256);
    expect(frame.referenceSample).toBe(frame.supportEndSampleExclusive - 960);
    expect(frame.f0Hz).toBeCloseTo(300, 0);
  }
});
