import { VOICE_RENDER_FRESHNESS_VERSION, type VoiceRenderLease, type VoiceRenderFreshnessPolicy } from "./voice-render-freshness.js";
import { CausalVoiceReplay } from "./voice-replay.js";
import type { VoicePitchFeatureHistoryV1 } from "./voice-pitch-features.js";
import type { VoicePitchFeaturesV2 } from "./voice-pitch-features-v2.js";
import type { VoiceResearchHook, VoiceResearchPrediction } from "./voice-research-hook.js";
import { extractVoicePitchFeaturesV2 } from "./voice-pitch-features-v2.js";
import { afterEach, describe, expect, it, vi } from "vitest";
import { requestedAudioCaptureSettings, VOICE_WORKER_MESSAGE_VERSION, VOICE_WORKLET_MESSAGE_VERSION,
  type VoiceWorkerRequest, type VoiceWorkerResponse, type VoiceWorkletPcmBlock } from "./voice-worker-protocol.js";
import type { VoiceClockMapping } from "./voice-stream-timing.js";

async function fixture(initiallySuspended = false, autoAcknowledge = true, hook?: VoiceResearchHook,
  freshnessPolicy: VoiceRenderFreshnessPolicy = "unqualified-legacy-fixture") {
  const responses: VoiceWorkerResponse[] = [];
  const wireResponses: VoiceWorkerResponse[] = [];
  const recycled: { transportId: number; buffer: ArrayBuffer }[] = [];
  let request!: (event: { data: VoiceWorkerRequest }) => void;
  let pcm!: (event: { data: unknown }) => void;
  let now = 45;
  vi.stubGlobal("performance", { timeOrigin: 107_000, now: () => { const result = now; now += 0.25; return result; } });
  vi.stubGlobal("self", { addEventListener: (_type: string, listener: typeof request) => { request = listener; },
    postMessage: (response: VoiceWorkerResponse) => {
      wireResponses.push(response);
      if (response.type === "result-batch") responses.push(...response.records);
      else if (response.type === "result-batch-complete") {
        responses.push({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "block-timing",
          captureEpoch: response.captureEpoch, timing: response.timing });
        if (autoAcknowledge) request({ data: { schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "result-ack",
          captureEpoch: response.captureEpoch, batchId: response.batchId, discardedFrameCount: 0, staleFrameCount: 0 } });
      } else responses.push(response);
    }, close() {} });
  const implementation = await import("./voice-worker.js");
  if (hook) implementation.configureVoiceResearchHook(hook);
  const port = { addEventListener: (_type: string, listener: typeof pcm) => { pcm = listener; },
    start() {}, close() {}, postMessage: (message: typeof recycled[number]) => recycled.push(message) } as unknown as MessagePort;
  const mapping: VoiceClockMapping = { id: 0, mainTimeOriginMs: 100_000, mainPerformanceMs: 7_000,
    audioContextTimeSeconds: 5, anchorReadUncertaintyMs: 0.1, audioClockUncertaintyMs: null };
  const send = (message: VoiceWorkerRequest) => request({ data: message });
  send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "initialize", freshnessPolicy, captureEpoch: 1,
    port, clockMapping: initiallySuspended ? null : mapping, audioContextSampleRateHz: 48_000, taskContext: "ambient-speech-turn",
    noiseFloorRms: 0.001, captureSettings: requestedAudioCaptureSettings(48_000, 1,
      { echoCancellation: false, noiseSuppression: false, autoGainControl: false }) });
  let transportId = 0;
  const block = (overrides: Partial<VoiceWorkletPcmBlock> = {}) => {
    const value: VoiceWorkletPcmBlock = {
      schemaVersion: VOICE_WORKLET_MESSAGE_VERSION, type: "pcm-block", captureEpoch: 1,
      sequence: 1, absoluteSampleIndex: 0, acquisitionAudioTimeSeconds: -999, endAudioFrameExclusive: 241_920,
      clockMappingId: 0, transportId: ++transportId, sampleRateHz: 48_000, channelCount: 1,
      droppedBlockCount: 0, droppedSampleCount: 0, inFlightBlockCount: 1,
      buffer: Float32Array.from({ length: 1920 }, (_, i) => 0.05 * Math.sin(i * 2 * Math.PI * 150 / 48_000)).buffer,
      ...overrides
    };
    now = (value.endAudioFrameExclusive / value.sampleRateHz - 5) * 1_000 + 5;
    pcm({ data: value });
    return value;
  };
  const frames = () => responses.filter((response): response is Extract<VoiceWorkerResponse, { type: "signal-frame" }> => response.type === "signal-frame");
  return { responses, wireResponses, recycled, send, block, frames, mapping, setNow: (value: number) => { now = value; }, rawPcm: pcm };
}

describe("actual DSP worker with simulated transport queues", () => {
  afterEach(() => { vi.unstubAllGlobals(); vi.resetModules(); });

  const renderLease = (issuedAtMainMs = 7000, overrides: Partial<VoiceRenderLease> = {}): VoiceRenderLease => ({
    schemaVersion: VOICE_RENDER_FRESHNESS_VERSION, leaseId: 1, captureEpoch: 1, clockMappingId: 0,
    mainTimeOriginMs: 100000, issuedAtMainMs, expiresAtMainMs: issuedAtMainMs + 100, ...overrides });

  it("withholds missing leases, expires cross-origin queued PCM, and recovers without analyzing the stale backlog", async () => {
    const f = await fixture(false, true, undefined, "required-render-lease-v1");
    f.block();
    expect(f.frames()).toHaveLength(0);
    f.block({ sequence: 2, absoluteSampleIndex: 1920, endAudioFrameExclusive: 243840, renderLease: renderLease(7040) });
    expect(f.frames()).toHaveLength(1);
    const first = f.frames()[0];
    const old = { ...f.block({ captureEpoch: 0 }), captureEpoch: 1, sequence: 3,
      absoluteSampleIndex: 3840, endAudioFrameExclusive: 245760, renderLease: renderLease(7080) };
    f.setNow(400); // Worker origin107000 +400 -> main7400, not local400 or audio120.
    f.rawPcm({ data: old });
    expect(f.frames()).toHaveLength(1);
    expect(f.responses.at(-1)).toMatchObject({ type: "block-timing", timing: {
      disposition: "freshness-withheld", reason: "expired-lease", analyzedFrameCount: 0 } });
    f.block({ sequence: 11, absoluteSampleIndex: 18240, endAudioFrameExclusive: 260160, renderLease: renderLease(7400) });
    expect(f.frames()).toHaveLength(2);
    expect(f.frames()[1].frame).toMatchObject({ absoluteSampleIndex: 18240, spectralFlux: 0 });
    expect(f.frames()[1].frame.trackSegmentId).not.toBe(first.frame.trackSegmentId);
    f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "dispose", captureEpoch: 1 });
    expect(f.responses.at(-1)).toMatchObject({ diagnostics: { freshnessWithheldBlockCount: 2,
      freshnessWithheldSampleCount: 3840, freshnessWithheldReasons: { "missing-lease": 1, "expired-lease": 1 } } });
  });

  it("does not gate fresh input on positive drift or negative fixed audio projections", async () => {
    const f = await fixture(false, true, undefined, "required-render-lease-v1");
    f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "clock-mapping", captureEpoch: 1,
      clockMapping: { ...f.mapping, mainPerformanceMs: 0 } });
    f.block({ renderLease: renderLease() });
    expect(f.frames()).toHaveLength(1);
    expect(f.frames()[0].timing.rawBlockEndToWorkerReceiptMs).toBeGreaterThan(7000);
    f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "clock-mapping", captureEpoch: 1,
      clockMapping: { ...f.mapping, id: 1, mainPerformanceMs: 9000 } });
    f.block({ sequence: 2, absoluteSampleIndex: 1920, endAudioFrameExclusive: 243840, clockMappingId: 1,
      renderLease: renderLease(7040, { leaseId: 2, clockMappingId: 1 }) });
    expect(f.frames()).toHaveLength(2);
    expect(f.frames()[1].timing).toMatchObject({ status: "unavailable", projectionSemantics: "unqualified-fixed-audio-clock-projection" });
    expect(f.frames()[1].timing.rawSampleAvailabilityToWorkerReadyMs).toBeLessThan(0);
  });

  it("lease renewal preserves the contiguous DSP sample grid and segment", async () => {
    const f = await fixture(false, true, undefined, "required-render-lease-v1");
    const pcm = Float32Array.from({ length: 960 }, (_, i) => .05 * Math.sin(i * 2 * Math.PI * 150 / 48000));
    for (let i = 0; i < 4; i += 1) f.block({ sequence: i + 1, absoluteSampleIndex: i * 960,
      endAudioFrameExclusive: 240000 + (i + 1) * 960, buffer: pcm.slice().buffer,
      renderLease: renderLease(7000 + i * 20, { leaseId: i + 1 }) });
    expect(f.frames().map(frame => frame.frame.tMs)).toEqual([0, 10, 20, 30, 40]);
    expect(new Set(f.frames().map(frame => frame.frame.trackSegmentId)).size).toBe(1);
    expect(f.frames().map(frame => frame.renderLease!.leaseId)).toEqual([2, 3, 3, 4, 4]);
  });

  it("uses an atomic reference at startup and the first PCM block of a renewed epoch", async () => {
    const f = await fixture();
    f.block();
    const initial = f.frames()[0];
    expect(initial.frame.snrDb).toBeCloseTo(20 * Math.log10(initial.frame.rms / 0.001), 8);
    f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "reset", captureEpoch: 2,
      clockMapping: { ...f.mapping, id: 1 }, taskContext: "ambient-speech-turn", noiseFloorRms: 0.01 });
    f.block(); // Late old-epoch PCM is recycled, never interpreted under the new reference.
    f.block({ captureEpoch: 2, clockMappingId: 1 });
    expect(f.frames()).toHaveLength(2);
    expect(f.recycled).toHaveLength(3);
    const next = f.frames()[1];
    expect(next.captureEpoch).toBe(2);
    expect(next.frame.snrDb).toBeCloseTo(20 * Math.log10(next.frame.rms / 0.01), 8);
    expect(initial.frame.snrDb - next.frame.snrDb).toBeCloseTo(20, 8);
    expect(next.frame.trackSegmentId).not.toBe(initial.frame.trackSegmentId);
  });

  it("uses exact sample support and corrected clock origins while retaining legacy window-start tMs", async () => {
    const f = await fixture();
    f.block();
    const output = f.frames()[0];
    expect(output.frame.tMs).toBe(0);
    expect(output.frame.absoluteSampleIndex).toBe(0);
    expect(output.frame.acquiredAtMs).toBeCloseTo(7_040, 8);
    expect(output.timing).toMatchObject({ supportStartSample: 0, supportEndSampleExclusive: 1920,
      referenceSample: 1920, lastRequiredInputSample: 1919, referenceTimeMs: 40,
      workerReceivedAtMs: 7_045.25, coordinate: "main-performance-ms" });
    expect(output.processingLatencyMs).toBeCloseTo(6.25, 8);
    expect(output.timing.workerServiceMs).toBeCloseTo(0.5, 8);
    expect(output.processingLatencyMs).toBeGreaterThan(output.timing.workerServiceMs);
    const blockTiming = f.responses.find(response => response.type === "block-timing");
    expect(blockTiming?.type).toBe("block-timing");
    if (blockTiming?.type === "block-timing") {
      expect(blockTiming.timing.analyzedFrameCount).toBe(1);
      expect(blockTiming.timing.workerServiceMs).toBeGreaterThan(output.timing.workerServiceMs);
      expect(blockTiming.timing.workerCompletedAtMs).toBeGreaterThan(output.timing.workerReadyAtMs);
    }
  });

  it("clears spectral and nucleus history and starts a new segment after a sample gap", async () => {
    const f = await fixture();
    f.block();
    const initial = f.frames()[0];
    expect(initial.frame.syllabicNucleus).toBe(true);
    f.block({ sequence: 3, absoluteSampleIndex: 2880, endAudioFrameExclusive: 244_800,
      droppedBlockCount: 1, droppedSampleCount: 960 });
    const renewed = f.frames()[1];
    expect(renewed.frame.tMs).toBe(60);
    expect(renewed.frame.trackSegmentId).not.toBe(initial.frame.trackSegmentId);
    expect(renewed.frame.spectralFlux).toBe(0);
    // Only 60 ms after the first nucleus: this is true only if refractory history reset.
    expect(renewed.frame.syllabicNucleus).toBe(true);
    expect(renewed.frame.f0Hz).toBe(initial.frame.f0Hz);
    f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "dispose", captureEpoch: 1 });
    const disposed = f.responses.at(-1);
    expect(disposed?.type).toBe("disposed");
    if (disposed?.type === "disposed") {
      expect(disposed.diagnostics.lostBlockCount).toBe(1);
      expect(disposed.diagnostics.droppedTransportSampleCount).toBe(960);
    }
  });

  it("invalidates suspension mappings, rejects stale mapping blocks, and reacquires a fresh segment", async () => {
    const f = await fixture();
    f.block();
    f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "clock-mapping", captureEpoch: 1, clockMapping: null });
    f.block({ sequence: 2, absoluteSampleIndex: 1920, endAudioFrameExclusive: 243_840 });
    expect(f.frames()).toHaveLength(1);
    f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "clock-mapping", captureEpoch: 1,
      clockMapping: { ...f.mapping, id: 2 } });
    f.block({ sequence: 3, absoluteSampleIndex: 3840, endAudioFrameExclusive: 245_760, clockMappingId: 0 });
    f.block({ sequence: 4, absoluteSampleIndex: 5760, endAudioFrameExclusive: 247_680, clockMappingId: 2 });
    expect(f.frames()).toHaveLength(2);
    expect(f.recycled).toHaveLength(4);
    expect(f.frames()[1].frame.trackSegmentId).not.toBe(f.frames()[0].frame.trackSegmentId);
    expect(f.frames()[1].frame.spectralFlux).toBe(0);
  });

  it("waits for a running clock mapping after suspended initialization", async () => {
    const f = await fixture(true);
    expect(f.responses[0].type).toBe("ready");
    f.block();
    expect(f.frames()).toHaveLength(0);
    expect(f.recycled).toHaveLength(1);
    f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "clock-mapping", captureEpoch: 1,
      clockMapping: { ...f.mapping, id: 1 } });
    f.block({ sequence: 2, absoluteSampleIndex: 1920, endAudioFrameExclusive: 243_840, clockMappingId: 1 });
    expect(f.frames()).toHaveLength(1);
    expect(f.frames()[0].frame.absoluteSampleIndex).toBe(1920);
    expect(f.frames()[0].frame.spectralFlux).toBe(0);
  });

  it("discards regressing blocks and clears all feature history before accepting contiguous recovery", async () => {
    const f = await fixture();
    f.block();
    const initial = f.frames()[0];
    f.block(); // Duplicate sequence/sample support must never be analyzed twice.
    expect(f.frames()).toHaveLength(1);
    f.block({ sequence: 2, absoluteSampleIndex: 1920, endAudioFrameExclusive: 243_840 });
    expect(f.frames()).toHaveLength(2);
    const recovered = f.frames()[1];
    expect(recovered.frame.trackSegmentId).not.toBe(initial.frame.trackSegmentId);
    expect(recovered.frame.absoluteSampleIndex).toBe(1920);
    expect(recovered.frame.spectralFlux).toBe(0);
    expect(recovered.frame.syllabicNucleus).toBe(true);
    f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "dispose", captureEpoch: 1 });
    expect(f.responses.at(-1)).toMatchObject({ type: "disposed", diagnostics: { timestampRegressionCount: 1 } });
  });

  it("rebuilds bounded rate-dependent state and changes segment on sample-rate change", async () => {
    const f = await fixture();
    f.block();
    f.block({ sequence: 2, absoluteSampleIndex: 1920, endAudioFrameExclusive: 224_028, sampleRateHz: 44_100,
      buffer: Float32Array.from({ length: 1764 }, (_, i) => 0.05 * Math.sin(i * 2 * Math.PI * 150 / 44_100)).buffer });
    expect(f.frames()).toHaveLength(2);
    expect(f.frames()[1].frame.trackSegmentId).not.toBe(f.frames()[0].frame.trackSegmentId);
    expect(f.frames()[1].frame.sampleRateHz).toBe(48_000); // Acquisition metadata is not the resampled PCM rate.
    expect(f.frames()[1].timing.sampleRateHz).toBe(44_100);
    expect(f.frames()[1].frame.spectralFlux).toBe(0);
  });

  it("keeps the acoustic sequence invariant under uneven contiguous chunks", async () => {
    const f = await fixture();
    const waveform = Float32Array.from({ length: 5760 }, (_, i) =>
      0.05 * Math.sin(i * 2 * Math.PI * (i < 2880 ? 150 : 180) / 48_000));
    const replay = (sizes: number[], epoch: number, clockMappingId: number) => {
      let offset = 0;
      for (const [index, length] of sizes.entries()) {
        f.block({ captureEpoch: epoch, clockMappingId, sequence: index + 1,
          absoluteSampleIndex: offset, endAudioFrameExclusive: 240_000 + offset + length,
          buffer: waveform.slice(offset, offset + length).buffer });
        offset += length;
      }
    };
    replay([960, 960, 960, 960, 960, 960], 1, 0);
    const acoustic = (frame: ReturnType<typeof f.frames>[number]) => ({
      tMs: frame.frame.tMs, f0Hz: frame.frame.f0Hz, rms: frame.frame.rms,
      spectralFlux: frame.frame.spectralFlux, syllabicNucleus: frame.frame.syllabicNucleus,
      f0Confidence: frame.frame.f0Confidence, estimatorAgreement: frame.frame.estimatorAgreement,
      start: frame.timing.supportStartSample, end: frame.timing.supportEndSampleExclusive
    });
    const expected = f.frames().map(acoustic);
    f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "reset", captureEpoch: 2,
      clockMapping: { ...f.mapping, id: 1 }, taskContext: "ambient-speech-turn" });
    replay([127, 833, 1300, 600, 1400, 1500], 2, 1);
    expect(f.frames().filter(frame => frame.captureEpoch === 2).map(acoustic)).toEqual(expected);
  });

  it("changes segment identity even when a reset repeats the same epoch", async () => {
    const f = await fixture();
    f.block();
    f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "reset", captureEpoch: 1,
      clockMapping: { ...f.mapping, id: 1 }, taskContext: "ambient-speech-turn" });
    f.block({ clockMappingId: 1 });
    expect(f.frames()[1].frame.trackSegmentId).not.toBe(f.frames()[0].frame.trackSegmentId);
    expect(f.frames()[1].frame.syllabicNucleus).toBe(f.frames()[0].frame.syllabicNucleus);
  });

  it("discards stale queued audio using mapped age, recycles it, and resumes in a fresh segment", async () => {
    const f = await fixture();
    const first = f.block();
    const stale = { ...first, sequence: 2, absoluteSampleIndex: 1920,
      endAudioFrameExclusive: 243_840, transportId: 99 };
    f.setNow(380); // Main time 7380, block availability7080: actual mapped age300ms.
    f.rawPcm({ data: stale });
    expect(f.frames()).toHaveLength(1);
    expect(f.recycled.at(-1)?.transportId).toBe(99);
    expect(f.responses.at(-1)).toMatchObject({ type: "block-timing", timing: {
      disposition: "stale-dropped", reason: "audio-timeline-age-exceeded", analyzedFrameCount: 0
    } });
    f.block({ sequence: 3, absoluteSampleIndex: 3840, endAudioFrameExclusive: 245_760 });
    expect(f.frames()).toHaveLength(2);
    expect(f.frames()[1].frame.trackSegmentId).not.toBe(f.frames()[0].frame.trackSegmentId);
    expect(f.frames()[1].frame.spectralFlux).toBe(0);
    expect(f.frames()[1].frame.tMs).toBe(80);
    f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "dispose", captureEpoch: 1 });
    const result = f.responses.at(-1);
    if (result?.type !== "disposed") throw new Error("Missing final diagnostics");
    expect(result.diagnostics.staleTransportBlockCount).toBe(1);
    expect(result.diagnostics.staleTransportSampleCount).toBe(1920);
    expect(result.diagnostics.lostBlockCount).toBe(1);
  });

  it("returns stale-port buffers only to their original sender after reinitialization", async () => {
    const f = await fixture();
    const first = f.block();
    let nextPcm!: (event: MessageEvent<unknown>) => void;
    const nextRecycled: unknown[] = [];
    const nextPort = { addEventListener: (_type: string, listener: typeof nextPcm) => { nextPcm = listener; },
      start() {}, close() {}, postMessage: (value: unknown) => nextRecycled.push(value) } as unknown as MessagePort;
    f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "initialize", freshnessPolicy: "unqualified-legacy-fixture", captureEpoch: 1,
      port: nextPort, clockMapping: f.mapping, audioContextSampleRateHz: 48_000,
      taskContext: "ambient-speech-turn", captureSettings: requestedAudioCaptureSettings(48_000, 1,
        { echoCancellation: false, noiseSuppression: false, autoGainControl: false }) });
    f.setNow(45);
    f.rawPcm({ data: first }); // Same epoch/id/token, but from the retired port.
    expect(f.frames()).toHaveLength(1);
    expect(nextRecycled).toHaveLength(0);
    expect(f.recycled).toHaveLength(2);
    nextPcm({ data: first } as MessageEvent<unknown>);
    expect(f.frames()).toHaveLength(2);
    expect(nextRecycled).toHaveLength(1);
  });


  it("keeps a suspended reset unmeasured and ignores a retired epoch's dispose request", async () => {
    const f = await fixture();
    f.block();
    f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "reset", captureEpoch: 2,
      clockMapping: null, taskContext: "ambient-speech-turn" });
    f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "dispose", captureEpoch: 1 });
    expect(f.responses.some(response => response.type === "disposed")).toBe(false);
    f.block({ captureEpoch: 2, clockMappingId: 1 });
    expect(f.frames()).toHaveLength(1);
    f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "clock-mapping", captureEpoch: 2,
      clockMapping: { ...f.mapping, id: 1 } });
    f.block({ captureEpoch: 2, clockMappingId: 1 });
    expect(f.frames()).toHaveLength(2);
    expect(f.frames()[1].captureEpoch).toBe(2);
    expect(f.frames()[1].frame.spectralFlux).toBe(0);
  });

  it("bounds blocked-main result batches, drops input without pending scalar growth, and recovers a new segment", async () => {
    const f = await fixture(false, false);
    for (let i = 0; i < 6; i += 1) f.block({ sequence: i + 1, absoluteSampleIndex: i * 1920,
      endAudioFrameExclusive: 241_920 + i * 1920 });
    const batches = f.wireResponses.filter(response => response.type === "result-batch");
    expect(batches).toHaveLength(4);
    expect(batches.every(batch => batch.type === "result-batch" && batch.records.length <= 5)).toBe(true);
    expect(f.wireResponses.filter(response => response.type === "result-batch-complete")).toHaveLength(4);
    expect(f.recycled).toHaveLength(6);
    const first = batches[0];
    if (first.type !== "result-batch") throw new Error("Missing batch");
    const ack = { schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "result-ack" as const,
      captureEpoch: 1, batchId: first.batchId, discardedFrameCount: 1, staleFrameCount: 1 };
    f.send({ ...ack, captureEpoch: 99 });
    f.block({ sequence: 7, absoluteSampleIndex: 11520, endAudioFrameExclusive: 253_440 });
    expect(f.wireResponses.filter(response => response.type === "result-batch")).toHaveLength(4);
    f.send(ack);
    f.send(ack); // Duplicate ACK cannot create a second credit.
    f.block({ sequence: 8, absoluteSampleIndex: 13440, endAudioFrameExclusive: 255_360 });
    const recovered = f.frames().at(-1)!;
    expect(recovered.frame.trackSegmentId).not.toBe(f.frames()[0].frame.trackSegmentId);
    expect(recovered.frame.absoluteSampleIndex).toBe(13440);
    expect(recovered.frame.spectralFlux).toBe(0);
    f.block({ sequence: 9, absoluteSampleIndex: 15360, endAudioFrameExclusive: 257_280 });
    expect(f.wireResponses.filter(response => response.type === "result-batch")).toHaveLength(5);
    f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "dispose", captureEpoch: 1 });
    expect(f.responses.at(-1)).toMatchObject({ type: "disposed", diagnostics: {
      inFlightResultBatchCount: 4, maximumInFlightResultBatches: 4,
      resultBackpressureBlockCount: 4, resultBackpressureSampleCount: 7680,
      resultBackpressureEpisodeCount: 2, hostDiscardedFrameCount: 1, hostStaleFrameCount: 1,
      lastResultBackpressureSpan: { supportStartSample: 15360, supportEndSampleExclusive: 17280,
        droppedBlockCount: 1, droppedSampleCount: 1920 }
    } }); // Final control response does not need a result credit.
  });

  it("preserves result credits across reset and accepts only each token's originating epoch", async () => {
    const f = await fixture(false, false);
    for (let i = 0; i < 4; i += 1) f.block({ sequence: i + 1, absoluteSampleIndex: i * 1920,
      endAudioFrameExclusive: 241_920 + i * 1920 });
    const first = f.wireResponses.find(response => response.type === "result-batch")!;
    if (first.type !== "result-batch") throw new Error("Missing batch");
    f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "reset", captureEpoch: 2,
      clockMapping: { ...f.mapping, id: 1 }, taskContext: "ambient-speech-turn" });
    f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "result-ack", captureEpoch: 2,
      batchId: first.batchId, discardedFrameCount: 0, staleFrameCount: 0 });
    f.block({ captureEpoch: 2, clockMappingId: 1 });
    expect(f.frames().filter(frame => frame.captureEpoch === 2)).toHaveLength(0);
    f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "result-ack", captureEpoch: 1,
      batchId: first.batchId, discardedFrameCount: 1, staleFrameCount: 1 });
    f.block({ captureEpoch: 2, clockMappingId: 1, sequence: 2, absoluteSampleIndex: 1920, endAudioFrameExclusive: 243_840 });
    expect(f.frames().filter(frame => frame.captureEpoch === 2)).toHaveLength(1);
    f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "dispose", captureEpoch: 2 });
    expect(f.responses.at(-1)).toMatchObject({ type: "disposed", diagnostics: { hostDiscardedFrameCount: 0,
      maximumInFlightResultBatches: 4, resultBackpressureBlockCount: 1 } });
  });

  it("rejects oversized PCM support explicitly without truncation or producing a result batch", async () => {
    const f = await fixture();
    f.block({ buffer: new Float32Array(1921).buffer });
    expect(f.frames()).toHaveLength(0);
    expect(f.recycled).toHaveLength(1);
    expect(f.wireResponses.some(response => response.type === "result-batch")).toBe(false);
    f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "dispose", captureEpoch: 1 });
    expect(f.responses.at(-1)).toMatchObject({ type: "disposed", diagnostics: { oversizedPcmBlockCount: 1 } });
  });

  it("frees retired-mapping ACK credit without resetting fresh same-epoch feature support", async () => {
    const f = await fixture(false, false);
    f.block();
    const batch = f.wireResponses.find(response => response.type === "result-batch")!;
    if (batch.type !== "result-batch") throw new Error("Missing batch");
    f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "clock-mapping", captureEpoch: 1,
      clockMapping: { ...f.mapping, id: 1 } });
    const samples = Float32Array.from({ length: 960 }, (_, i) => .05 * Math.sin(i * 2 * Math.PI * 150 / 48000));
    f.block({ sequence: 2, clockMappingId: 1, absoluteSampleIndex: 1920, endAudioFrameExclusive: 242880, buffer: samples.slice().buffer });
    f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "result-ack", captureEpoch: 1,
      batchId: batch.batchId, discardedFrameCount: 1, staleFrameCount: 0 });
    f.block({ sequence: 3, clockMappingId: 1, absoluteSampleIndex: 2880, endAudioFrameExclusive: 243840, buffer: samples.slice().buffer });
    expect(f.frames()).toHaveLength(2);
    expect(f.frames()[1].frame.absoluteSampleIndex).toBe(1920);
  });

  it("bounds queued research PCM and cancels an in-flight model across reset without late delivery", async () => {
    const resolutions: Array<() => void> = [];
    const dispose = vi.fn(async () => {});
    const reset = vi.fn();
    const hook: VoiceResearchHook = {
      prepare: (samples, rate) => extractVoicePitchFeaturesV2(samples, rate), reset, dispose,
      predict: () => new Promise(resolve => resolutions.push(() => resolve({} as VoiceResearchPrediction)))
    };
    const f = await fixture(false, true, hook);
    for (let i = 0; i < 5; i += 1) f.block({ sequence: i + 1, absoluteSampleIndex: i * 1920, endAudioFrameExclusive: 241920 + i * 1920 });
    expect(resolutions).toHaveLength(1);
    expect(f.frames()).toHaveLength(0);
    expect(f.recycled).toHaveLength(1); // Overflow, with one active and three pending buffers.
    f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "reset", captureEpoch: 2,
      clockMapping: { ...f.mapping, id: 1 }, taskContext: "ambient-speech-turn" });
    expect(f.recycled).toHaveLength(5);
    resolutions[0]();
    await new Promise(resolve => setTimeout(resolve, 0));
    expect(f.frames()).toHaveLength(0);
    f.block({ captureEpoch: 2, clockMappingId: 1 });
    expect(resolutions).toHaveLength(2);
    resolutions[1]();
    await new Promise(resolve => setTimeout(resolve, 0));
    expect(f.frames()).toHaveLength(1);
    expect(f.frames()[0].captureEpoch).toBe(2);
    expect(f.frames()[0].frame.absoluteSampleIndex).toBe(0);
    expect(f.frames()[0].frame.f0Hz).toBeGreaterThan(140);
    expect(f.recycled).toHaveLength(6);
    f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "dispose", captureEpoch: 2 });
    expect(dispose).toHaveBeenCalledOnce();
  });

  it("natural completion fence waits for acoustic inference and the main result acknowledgement", async () => {
    let release!: (value: VoiceResearchPrediction) => void;
    const hook: VoiceResearchHook = { prepare: (samples, rate) => extractVoicePitchFeaturesV2(samples, rate), reset() {},
      predict: () => new Promise(resolve => { release = resolve; }), async dispose() {} };
    const f = await fixture(false, false, hook);
    f.block();
    f.rawPcm({ data: { type: "research-natural-end", captureEpoch: 1, fenceId: 1 } });
    const drained = () => f.wireResponses.filter(row => (row as unknown as { type: string }).type === "research-natural-drained");
    expect(drained()).toHaveLength(0);
    release({} as VoiceResearchPrediction);
    await new Promise(resolve => setTimeout(resolve, 15));
    expect(drained()).toHaveLength(0);
    const batch = f.wireResponses.find(row => row.type === "result-batch") as Extract<VoiceWorkerResponse, { type: "result-batch" }>;
    f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "result-ack", captureEpoch: 1,
      batchId: batch.batchId, discardedFrameCount: 0, staleFrameCount: 0 });
    await new Promise(resolve => setTimeout(resolve, 15));
    expect(drained()).toEqual([expect.objectContaining({ status: "drained", pumpIdle: true,
      diagnostics: expect.objectContaining({ inFlightResultBatchCount: 0, researchCancelledFrameCount: 0 }) })]);
  });

  it("acknowledges disposal immediately during pending inference and recycles active PCM only once", async () => {
    let finish!: () => void;
    const hook: VoiceResearchHook = { prepare: (samples, rate) => extractVoicePitchFeaturesV2(samples, rate), reset() {},
      dispose: vi.fn(async () => {}), predict: () => new Promise(resolve => { finish = () => resolve({} as VoiceResearchPrediction); }) };
    const f = await fixture(false, true, hook);
    f.block();
    f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "dispose", captureEpoch: 1 });
    expect(f.responses.at(-1)).toMatchObject({ type: "disposed", diagnostics: { researchCancelledBlockCount: 1,
      researchCancelledFrameCount: 1, inFlightResultBatchCount: 0 } });
    expect(f.recycled).toHaveLength(1);
    finish();
    await new Promise(resolve => setTimeout(resolve, 0));
    expect(f.frames()).toHaveLength(0);
    expect(f.recycled).toHaveLength(1);
  });

  it.each(["epoch", "mapping"])("suppresses a retired research rejection after %s reset", async resetKind => {
    let reject!: (error: Error) => void;
    const hook: VoiceResearchHook = { prepare: (pcm, rate) => extractVoicePitchFeaturesV2(pcm, rate),
      reset() {}, dispose: async () => {}, predict: () => new Promise((_resolve, fail) => { reject = fail; }) };
    const f = await fixture(false, true, hook);
    f.block();
    if (resetKind === "epoch") f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "reset", captureEpoch: 2,
      clockMapping: { ...f.mapping, id: 1 }, taskContext: "ambient-speech-turn" });
    else f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "clock-mapping", captureEpoch: 1,
      clockMapping: { ...f.mapping, id: 1 } });
    reject(new Error("retired inference rejection"));
    await new Promise(resolve => setTimeout(resolve, 0));
    expect(f.responses.some(response => response.type === "error")).toBe(false);
    expect(f.frames()).toHaveLength(0);
    expect(f.recycled).toHaveLength(1);
  });

  it("fails a current research model once and bounds subsequent error transport", async () => {
    const predict = vi.fn(async () => { throw new Error("current inference failure"); });
    const hook: VoiceResearchHook = { prepare: (pcm, rate) => extractVoicePitchFeaturesV2(pcm, rate),
      reset() {}, dispose: async () => {}, predict };
    const f = await fixture(false, true, hook);
    f.block();
    await new Promise(resolve => setTimeout(resolve, 0));
    for (let i = 0; i < 8; i += 1) f.block({ sequence: i + 2 });
    expect(predict).toHaveBeenCalledOnce();
    expect(f.responses.filter(response => response.type === "error")).toHaveLength(1);
    expect(f.recycled).toHaveLength(9);
    f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "dispose", captureEpoch: 1 });
    expect(f.responses.at(-1)?.type).toBe("disposed");
  });

  it("research preparation matches native 40ms offline features and preserves every legacy DSP field", async () => {
    const samples = Float32Array.from({ length: 960 * 12 }, (_, i) => {
      const amplitude = i < 4800 ? .05 : .1;
      return amplitude * Math.sin(i * 2 * Math.PI * 200 / 48000) + .015 * Math.sin(i * 2 * Math.PI * 400 / 48000);
    });
    const baseline = await fixture();
    for (let i = 0; i < 12; i += 1) baseline.block({ sequence: i + 1, absoluteSampleIndex: i * 960,
      endAudioFrameExclusive: 240000 + (i + 1) * 960, buffer: samples.slice(i * 960, (i + 1) * 960).buffer });
    const original = baseline.frames().map(response => response.frame);
    vi.resetModules();
    let history: VoicePitchFeatureHistoryV1 | null = null;
    const prepared: VoicePitchFeaturesV2[] = [];
    const hook: VoiceResearchHook = {
      prepare(pcm, rate) { const features = extractVoicePitchFeaturesV2(pcm, rate, history);
        history = features.nextHistory; prepared.push(features); return features; },
      reset() { history = null; }, dispose: async () => {}, predict: async () => ({} as VoiceResearchPrediction)
    };
    const research = await fixture(false, true, hook);
    for (let i = 0; i < 12; i += 1) {
      research.block({ sequence: i + 1, absoluteSampleIndex: i * 960, endAudioFrameExclusive: 240000 + (i + 1) * 960,
        buffer: samples.slice(i * 960, (i + 1) * 960).buffer });
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    expect(research.frames().map(response => response.frame)).toEqual(original);
    const offline = new CausalVoiceReplay({ sampleRateHz: 48000, featureVersion: "v2", referenceOriginMs: 0 }).push(samples);
    expect(offline[0].supportEndSampleExclusive).toBe(1920);
    expect(prepared).toHaveLength(offline.length);
    for (let i = 0; i < prepared.length; i += 1) {
      expect(prepared[i]).toEqual(offline[i].features);
      expect(research.frames()[i].timing.supportEndSampleExclusive).toBe(offline[i].supportEndSampleExclusive);
      expect(research.frames()[i].frame.tMs).toBe(offline[i].legacyWindowStartMs);
    }
  });

  it("reports impossible negative timeline latency instead of publishing a plausible service time", async () => {
    const f = await fixture();
    const block = f.block();
    f.send({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "reset", captureEpoch: 2,
      clockMapping: { ...f.mapping, id: 1 }, taskContext: "ambient-speech-turn" });
    f.setNow(0);
    f.rawPcm({ data: { ...block, captureEpoch: 2, clockMappingId: 1, transportId: 99 } });
    expect(f.frames()).toHaveLength(2);
    expect(f.frames()[1].processingLatencyMs).toBeNull();
    expect(f.frames()[1].timing.status).toBe("unavailable");
    expect(f.frames()[1].timing.rawSampleAvailabilityToWorkerReadyMs).toBeLessThan(0);
    expect(f.recycled.at(-1)?.transportId).toBe(99);
  });
});
