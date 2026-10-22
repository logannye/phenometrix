import { expect, test } from "@playwright/test";
import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import type { VoiceResearchModelConfig } from "../src/voice-research-hook.js";
import { createHash } from "node:crypto";
import { dirname, resolve } from "node:path";
import type { ReplayResult, ReplayTelemetryBatch } from "../src/research-voice-replay.js";

const outputPath = (name: string, fallback: string) => process.env.VOICE_RESEARCH_OUTPUT_DIR
  ? resolve(process.env.VOICE_RESEARCH_OUTPUT_DIR, name) : fallback;

test("real AudioWorklet/worker replay bounds overload and recovers segments", async ({ page }, testInfo) => {
  const method = process.env.VOICE_RESEARCH_METHOD ?? "dsp";
  if (method !== "dsp" && method !== "trained-mlp") throw new Error("Invalid research method.");
  const modelConfig = method === "trained-mlp"
    ? JSON.parse(readFileSync(resolve("public/.local/voice-mlp/config.json"), "utf8")) as VoiceResearchModelConfig : undefined;
  const durationMs = Number(process.env.VOICE_RESEARCH_DURATION_MS ?? 3_500);
  const overloadAtMs = Number(process.env.VOICE_RESEARCH_OVERLOAD_AT_MS ?? 1_200);
  const overloadDurationMs = Number(process.env.VOICE_RESEARCH_OVERLOAD_DURATION_MS ?? 300);
  if (!Number.isFinite(durationMs) || durationMs < 3_000 || durationMs > 31 * 60 * 1_000 ||
      !Number.isFinite(overloadAtMs) || overloadAtMs < 0 ||
      !Number.isFinite(overloadDurationMs) || overloadDurationMs < 150 || overloadDurationMs > 1_000 ||
      overloadAtMs + overloadDurationMs + 1_000 > durationMs) throw new Error("Invalid overload replay configuration.");
  test.setTimeout(durationMs + 45_000);
  const rawPath = outputPath("browser-raw.ndjson", testInfo.outputPath("browser-raw.ndjson"));
  mkdirSync(dirname(rawPath), { recursive: true });
  writeFileSync(rawPath, "", { flag: "wx" });
  let expectedBatch = 1;
  let rawRecordCount = 0;
  await page.exposeBinding("recordVoiceResearchBatch", (_source, batch: ReplayTelemetryBatch) => {
    if (batch.schemaVersion !== "phenometric.voice-replay-telemetry.v1" || batch.sequence !== expectedBatch ||
        batch.records.length > 100) throw new Error("Unexpected replay batch ordering or capacity.");
    expectedBatch += 1;
    rawRecordCount += batch.records.length;
    appendFileSync(rawPath, `${JSON.stringify(batch)}\n`);
  });
  await page.goto("/research-voice.html");
  await page.waitForFunction(() => typeof window.runVoiceResearchReplay === "function");
  const result = await page.evaluate(options => window.runVoiceResearchReplay({ ...options, streamRawTelemetry: true,
    retainedFrameLimit: 120, retainedBlockLimit: 80 }), { durationMs, overloadAtMs, overloadDurationMs, modelConfig }) as ReplayResult;
  let validationStatus = "failed";
  try {
    expect(result.status, JSON.stringify(result.errors)).toBe("complete");
    expect(result.workerStopAcknowledged).toBe(true);
    expect(result.transportStop.status).toBe("complete");
    expect(result.pacedSourceDurationMs).toBeGreaterThanOrEqual(durationMs - 10);
    expect(result.setupDurationMs).toBeGreaterThanOrEqual(0);
    expect(result.teardownDurationMs).toBeGreaterThanOrEqual(0);
    expect(result.blockTimings.length).toBeGreaterThan(50);
    expect(result.blockTimings.length).toBeLessThanOrEqual(80);
    expect(result.frames.length).toBeGreaterThan(100);
    expect(result.frames.length).toBeLessThanOrEqual(120);
    const diagnostic = result.diagnostics as { maximumInFlightBlocks: number; droppedTransportBlockCount: number;
      droppedTransportSampleCount: number; lostBlockCount: number; staleTransportBlockCount: number };
    expect(diagnostic.maximumInFlightBlocks).toBeLessThanOrEqual(4);
    expect(diagnostic.droppedTransportBlockCount).toBeGreaterThan(0);
    expect(diagnostic.droppedTransportSampleCount).toBeGreaterThan(0);
    expect(diagnostic.lostBlockCount).toBeGreaterThan(0);
    expect(diagnostic.staleTransportBlockCount).toBeGreaterThan(0);
    expect(Number(result.summary.segmentCount)).toBeGreaterThan(1);
    expect(Number(result.summary.periodicFrameCount)).toBeGreaterThan(100);
    expect(result.summary.correctKnownPitchFrameCount).toBe(result.summary.periodicFrameCount);
    expect(Number(result.summary.maximumKnownPitchErrorHz)).toBeLessThanOrEqual(2);
    const raw = result.summary.rawTelemetry as { status: string; streamedRecordCount: number; maximumPendingRecords: number };
    expect(raw.status).toBe("complete");
    expect(raw.maximumPendingRecords).toBeLessThanOrEqual(200);
    expect(raw.streamedRecordCount).toBe(rawRecordCount);
    expect(rawRecordCount).toBe(Number(result.summary.frameCount) + Number(result.summary.discardedResultFrameCount) + Number(result.summary.blockCount));
    expect(Number(result.summary.frameCount)).toBeGreaterThan(result.frames.length);
    expect(Number(result.summary.blockCount)).toBeGreaterThan(result.blockTimings.length);
    if (result.transportStop.status === "complete") {
      expect(result.transportStop.diagnostics.droppedSampleCount).toBeGreaterThanOrEqual(diagnostic.droppedTransportSampleCount);
      expect(result.transportStop.diagnostics.maximumInFlightBlocks).toBe(4);
    }
    for (const frame of result.frames) {
      const timing = frame.timing as Record<string, number | string | null>;
      expect(Number(timing.supportEndSampleExclusive) - Number(timing.supportStartSample)).toBe(1920);
      expect(timing.lastRequiredInputSample).toBe(Number(timing.supportEndSampleExclusive) - 1);
      expect(Number(timing.workerServiceMs)).toBeGreaterThanOrEqual(0);
      if (timing.status === "valid") {
        expect(Number(timing.mainResultArrivalMs)).toBeGreaterThanOrEqual(Number(timing.workerReadyAtMs));
        expect(Number(timing.sampleAvailabilityToMainResultMs)).toBeGreaterThanOrEqual(0);
      } else expect(timing.sampleAvailabilityToMainResultMs).toBeNull();
    }
    validationStatus = "passed";
  } finally {
    const evidence = { ...result, validationStatus,
      runPurpose: durationMs >= 30 * 60 * 1_000 ? "sustained-functional-soak" : "functional-regression",
      replayConfiguration: { method, durationMs, overloadAtMs, overloadDurationMs, modelConfig },
      timingBenchmark: false,
      timingLimitation: "Synthetic replay; exclusive hardware conditions and hardware input delay are not established by this test.",
      rawArtifact: { file: "browser-raw.ndjson", sha256: createHash("sha256").update(readFileSync(rawPath)).digest("hex"),
        batchCount: expectedBatch - 1, recordCount: rawRecordCount } };
    await testInfo.attach("actual-browser-replay", { body: JSON.stringify(evidence), contentType: "application/json" });
    const resultPath = outputPath("browser-replay.json", testInfo.outputPath("browser-replay.json"));
    writeFileSync(resultPath, `${JSON.stringify(evidence)}\n`, { flag: "wx" });
  }
});

test("exported recurrent graph matches independent Python fixtures in browser WASM", async ({ page }, testInfo) => {
  await page.goto("/research-voice.html");
  await page.waitForFunction(() => typeof window.runVoiceModelExportParity === "function");
  const result = await page.evaluate(() => window.runVoiceModelExportParity());
  await testInfo.attach("browser-export-parity", { body: JSON.stringify(result), contentType: "application/json" });
  const target = outputPath("browser-export-parity.json", testInfo.outputPath("browser-export-parity.json"));
  mkdirSync(dirname(target), { recursive: true });
  writeFileSync(target, `${JSON.stringify(result)}\n`, { flag: "wx" });
  expect(result.status, JSON.stringify(result)).toBe("complete");
  expect(result.steps).toBe(24);
  expect(result.maximumAbsoluteError).toBeLessThanOrEqual(1e-6);
  expect(result.decisionMismatches).toBe(0);
  expect(result.allMaskedSteps).toBeGreaterThan(0);
  expect(result.outputContract).toMatchObject({ dtype: "float32", voicingLogitShape: [1, 1] });
});


test("telemetry sink failure marks replay incomplete and stops collection", async ({ page }) => {
  await page.exposeBinding("recordVoiceResearchBatch", () => { throw new Error("deliberate sink failure"); });
  await page.goto("/research-voice.html");
  await page.waitForFunction(() => typeof window.runVoiceResearchReplay === "function");
  const result = await page.evaluate(() => window.runVoiceResearchReplay({ durationMs: 2_000,
    streamRawTelemetry: true, retainedFrameLimit: 20, retainedBlockLimit: 10 }));
  expect(result.status).toBe("failed");
  expect(result.errors.some(error => error.startsWith("telemetry-sink-failed:"))).toBe(true);
  expect(result.pacedSourceDurationMs).toBeLessThan(2_000);
  expect(result.frames.length).toBeLessThanOrEqual(20);
  expect(result.blockTimings.length).toBeLessThanOrEqual(10);
  expect(result.workerStopAcknowledged).toBe(true);
  expect((result.summary.rawTelemetry as { status: string }).status).toBe("failed");
});

test("blocked main thread bounds result credits and discards stale queued frames before recovery", async ({ page }, testInfo) => {
  await page.goto("/research-voice.html");
  await page.waitForFunction(() => typeof window.runVoiceResearchReplay === "function");
  const result = await page.evaluate(() => window.runVoiceResearchReplay({ durationMs: 2_200,
    mainStallAtMs: 600, mainStallDurationMs: 350, retainedFrameLimit: 80, retainedBlockLimit: 50 }));
  await testInfo.attach("blocked-main-result-transport", { body: JSON.stringify(result), contentType: "application/json" });
  const target = outputPath("browser-blocked-main.json", testInfo.outputPath("browser-blocked-main.json"));
  mkdirSync(dirname(target), { recursive: true });
  writeFileSync(target, `${JSON.stringify({ ...result, runPurpose: "functional-regression", timingBenchmark: false })}\n`, { flag: "wx" });
  expect(result.status, JSON.stringify(result.errors)).toBe("complete");
  expect(result.workerStopAcknowledged).toBe(true);
  expect(result.transportStop.status).toBe("complete");
  expect(result.diagnostics).toMatchObject({ maximumInFlightResultBatches: 4 });
  const diagnostics = result.diagnostics as { resultBackpressureBlockCount: number; resultBackpressureSampleCount: number;
    hostDiscardedFrameCount: number; hostStaleFrameCount: number; lastResultBackpressureSpan: {
      supportStartSample: number; supportEndSampleExclusive: number; droppedSampleCount: number } };
  expect(diagnostics.resultBackpressureBlockCount).toBeGreaterThan(0);
  expect(diagnostics.resultBackpressureSampleCount).toBeGreaterThan(0);
  expect(diagnostics.lastResultBackpressureSpan.supportEndSampleExclusive).toBeGreaterThan(diagnostics.lastResultBackpressureSpan.supportStartSample);
  expect(diagnostics.hostStaleFrameCount).toBeGreaterThan(0);
  expect(diagnostics.hostDiscardedFrameCount).toBe(result.summary.discardedResultFrameCount);
  expect(Number(result.summary.segmentCount)).toBeGreaterThan(1);
  expect(Number(result.summary.staleResultFrameCount)).toBeGreaterThan(0);
  expect(result.summary.correctKnownPitchFrameCount).toBe(result.summary.periodicFrameCount);
  for (const frame of result.frames) {
    const timing = frame.timing as { status: string; sampleAvailabilityToMainResultMs: number | null };
    if (timing.status === "valid") expect(timing.sampleAvailabilityToMainResultMs).toBeLessThanOrEqual(100);
    else expect(timing.sampleAvailabilityToMainResultMs).toBeNull();
  }
});


test("research-only trained MLP runs through actual worklet and matches independent NumPy", async ({ page }, testInfo) => {
  test.setTimeout(45_000);
  const config = JSON.parse(readFileSync(resolve("public/.local/voice-mlp/config.json"), "utf8")) as VoiceResearchModelConfig;
  await page.goto("/research-voice.html");
  await page.waitForFunction(() => typeof window.runVoiceResearchReplay === "function");
  const result = await page.evaluate(modelConfig => window.runVoiceResearchReplay({ durationMs: 2_200,
    modelConfig, retainedFrameLimit: 300, retainedBlockLimit: 150 }), config);
  const target = outputPath("browser-trained-mlp.json", testInfo.outputPath("browser-trained-mlp.json"));
  mkdirSync(dirname(target), { recursive: true });
  writeFileSync(target, `${JSON.stringify({ ...result, runPurpose: "functional-parity", timingBenchmark: false })}\n`, { flag: "wx" });
  expect(result.status, JSON.stringify(result.errors)).toBe("complete");
  expect(result.researchModel).toMatchObject({ qualification: "unqualified-functional-fixture", modelSha256: config.modelSha256 });
  expect(result.workerStopAcknowledged).toBe(true);
  expect(result.transportStop.status).toBe("complete");
  expect(Number(result.summary.researchFrameCount)).toBe(Number(result.summary.frameCount));
  expect(Number(result.summary.frameCount)).toBeGreaterThan(100);
  expect(result.summary.correctKnownPitchFrameCount).toBe(result.summary.periodicFrameCount);
  expect(result.diagnostics).toMatchObject({ researchQueueDroppedBlockCount: 0 });
  const diagnostic = result.diagnostics as { researchMaximumQueuedBlocks: number; maximumInFlightResultBatches: number };
  expect(diagnostic.researchMaximumQueuedBlocks).toBeLessThanOrEqual(3);
  expect(diagnostic.maximumInFlightResultBatches).toBeLessThanOrEqual(4);
  for (const frame of result.frames) {
    const prediction = frame.research as { featureValues: number[]; candidateProbabilities: number[]; correctnessProbability: null };
    expect(prediction.featureValues).toHaveLength(57);
    expect(prediction.candidateProbabilities).toHaveLength(6);
    expect(prediction.correctnessProbability).toBeNull();
  }
  const parity = JSON.parse(execFileSync(resolve("../../research/voice-prosody/.venv/bin/python"), [
    resolve("scripts/verify-voice-mlp-stream.py"), resolve("../../research/voice-prosody/runs/delayed-mlp-export"), target
  ], { encoding: "utf8", timeout: 30_000 }));
  expect(parity.status).toBe("complete");
  expect(parity.steps).toBe(result.frames.length);
  const parityTarget = outputPath("browser-trained-mlp-numpy-parity.json", testInfo.outputPath("browser-trained-mlp-numpy-parity.json"));
  writeFileSync(parityTarget, `${JSON.stringify(parity)}\n`, { flag: "wx" });
  await testInfo.attach("trained-mlp-stream-parity", { body: JSON.stringify(parity), contentType: "application/json" });
});
