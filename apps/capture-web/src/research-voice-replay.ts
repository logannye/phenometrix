import type { VoiceResearchModelConfig } from "./voice-research-hook.js";
import { VoiceResultBatchReceiver } from "./voice-result-transport.js";
import { captureVoiceClockMapping, type VoiceFrameTiming } from "./voice-stream-timing.js";
import { VOICE_WORKER_MESSAGE_VERSION, isCurrentVoiceTransportDiagnostics, requestedAudioCaptureSettings,
  type VoiceTransportStopResult, type VoiceWorkerResponse } from "./voice-worker-protocol.js";

export interface ReplayTelemetryBatch {
  schemaVersion: "phenometric.voice-replay-telemetry.v1";
  sequence: number;
  records: Array<{ kind: "frame" | "block-timing"; value: Record<string, unknown> }>;
}
export interface ReplayOptions {
  durationMs?: number;
  /** Research route only; the production capture entry never supplies a model. */
  modelConfig?: VoiceResearchModelConfig;
  overloadAtMs?: number;
  overloadDurationMs?: number;
  mainStallAtMs?: number;
  mainStallDurationMs?: number;
  /** Requires window.recordVoiceResearchBatch, normally a Playwright-to-Node file sink. */
  streamRawTelemetry?: boolean;
  retainedFrameLimit?: number;
  retainedBlockLimit?: number;
}
export interface ReplayResult {
  status: "complete" | "failed";
  evidence: "paced-synthetic-browser-replay";
  durationMs: number;
  requestedSourceDurationMs: number;
  pacedSourceDurationMs: number;
  setupDurationMs: number;
  teardownDurationMs: number;
  sampleRateHz: number;
  researchModel: { qualification: "unqualified-functional-fixture"; modelSha256: string; loadMs: number } | null;
  browser: string;
  frames: Array<Record<string, unknown>>;
  errors: string[];
  diagnostics: unknown;
  transportStop: VoiceTransportStopResult;
  workerStopAcknowledged: boolean;
  blockTimings: Array<Record<string, unknown>>;
  maximumServiceMs: number | null;
  summary: Record<string, unknown>;
}
const quantiles = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  const q = (fraction: number) => sorted.length ? sorted[Math.ceil(sorted.length * fraction) - 1] : null;
  return { n: sorted.length, p50: q(.5), p95: q(.95), p99: q(.99), maximum: q(1) };
};
const boundedLimit = (value: number | undefined, fallback: number) => {
  const limit = value ?? fallback;
  if (!Number.isSafeInteger(limit) || limit < 0 || limit > 10_000) throw new Error("Invalid retained telemetry limit.");
  return limit;
};

/** Real-time synthetic replay, never microphone capture or a hardware-input latency test. */
export async function runVoiceResearchReplay(options: ReplayOptions = {}): Promise<ReplayResult> {
  const durationMs = options.durationMs ?? 4_000;
  if (!Number.isFinite(durationMs) || durationMs < 1_000 || durationMs > 31 * 60 * 1_000) throw new Error("Invalid replay duration.");
  if (options.mainStallAtMs !== undefined && (!Number.isFinite(options.mainStallAtMs) || options.mainStallAtMs < 0 ||
      !Number.isFinite(options.mainStallDurationMs ?? 300) || (options.mainStallDurationMs ?? 300) < 0 ||
      (options.mainStallDurationMs ?? 300) > 1_000)) throw new Error("Invalid main-thread stall.");
  const frameLimit = boundedLimit(options.retainedFrameLimit, 2_000);
  const blockLimit = boundedLimit(options.retainedBlockLimit, 1_000);
  if (options.streamRawTelemetry && !window.recordVoiceResearchBatch) throw new Error("Missing replay telemetry sink.");
  const context = new AudioContext({ sampleRate: 48_000 });
  const worker = options.modelConfig
    ? new Worker(new URL("./research-voice-trained-worker.ts", import.meta.url), { type: "module" })
    : new Worker(new URL("./research-voice-stall-worker.ts", import.meta.url), { type: "module" });
  let researchModel: ReplayResult["researchModel"] = null;
  const channel = new MessageChannel();
  const frames: Array<Record<string, unknown>> = [];
  const blockTimings: Array<Record<string, unknown>> = [];
  const errors: string[] = [];
  const statisticsCapacity = 200_000; // >31 min at the declared 100 Hz hop; no full-frame growth.
  const statistics = { frameService: [] as number[], blockService: [] as number[],
    processedBlockService: [] as number[], staleBlockService: [] as number[],
    arrival: [] as number[], receipt: [] as number[], modelInference: [] as number[], modelReady: [] as number[], modelArrival: [] as number[], targetReady: [] as number[] };
  const segments: string[] = [];
  let previousSegment = "";
  let segmentCount = 0;
  let frameCount = 0;
  let researchFrameCount = 0;
  let blockCount = 0;
  let invalidTimingCount = 0;
  let discardedResultFrameCount = 0;
  let staleResultFrameCount = 0;
  let periodicFrameCount = 0;
  let correctKnownPitchFrameCount = 0;
  let maximumKnownPitchErrorHz = 0;
  const blockFrameCounts: Record<string, number> = {};
  let diagnostics: unknown = null;
  let transportStop: VoiceTransportStopResult = { status: "unavailable", captureEpoch: 1, reason: "worklet-stop-timeout" };
  let workerStopAcknowledged = false;
  let transportStopAcknowledged = false;
  let node: AudioWorkletNode | null = null;
  let source: AudioBufferSourceNode | null = null;
  let mainStallTimer: ReturnType<typeof setTimeout> | undefined;
  let overloadTimer: ReturnType<typeof setTimeout> | undefined;
  let pacedTimer: ReturnType<typeof setTimeout> | undefined;
  let workerDisposed: (() => void) | undefined;
  let transportDisposed: (() => void) | undefined;
  let ready: (() => void) | undefined;
  let initializationFailed: ((error: Error) => void) | undefined;
  let finishPaced: (() => void) | undefined;
  let sourceStartedAtMs = 0;
  let sourceStoppedAtMs = 0;
  let stopping = false;
  const began = performance.now();
  const fail = (reason: string) => {
    if (!errors.includes(reason)) errors.push(reason);
    finishPaced?.();
  };
  const addStatistic = (values: number[], value: number | null) => {
    if (value === null || !Number.isFinite(value)) return;
    if (values.length === statisticsCapacity) { fail("numeric-statistics-capacity-exceeded"); return; }
    values.push(value);
  };
  const retain = (values: Array<Record<string, unknown>>, count: number, limit: number, item: Record<string, unknown>) => {
    if (limit > 0) values[(count - 1) % limit] = item;
  };
  const chronological = (values: Array<Record<string, unknown>>, count: number, limit: number) => {
    if (count <= limit || limit === 0) return values;
    const first = count % limit;
    return [...values.slice(first), ...values.slice(0, first)];
  };

  const telemetryQueue: ReplayTelemetryBatch["records"] = [];
  let telemetryInFlight: Promise<void> | null = null;
  let telemetryFailed = false;
  let telemetryFinishing = false;
  let telemetryBatchSequence = 0;
  let streamedRecordCount = 0;
  let maximumPendingTelemetryRecords = 0;
  const pumpTelemetry = (): void => {
    if (!options.streamRawTelemetry || telemetryFailed || telemetryInFlight || telemetryQueue.length === 0 ||
        (!telemetryFinishing && telemetryQueue.length < 100)) return;
    const records = telemetryQueue.splice(0, 100);
    const batch: ReplayTelemetryBatch = { schemaVersion: "phenometric.voice-replay-telemetry.v1",
      sequence: ++telemetryBatchSequence, records };
    telemetryInFlight = Promise.resolve().then(() => window.recordVoiceResearchBatch!(batch)).then(() => {
      streamedRecordCount += records.length;
    }).catch(error => {
      telemetryFailed = true;
      fail(`telemetry-sink-failed:${String(error)}`);
    }).finally(() => { telemetryInFlight = null; pumpTelemetry(); });
  };
  const emitTelemetry = (kind: "frame" | "block-timing", value: Record<string, unknown>) => {
    if (!options.streamRawTelemetry || telemetryFailed) return;
    if (telemetryQueue.length >= 200) {
      telemetryFailed = true;
      fail("telemetry-sink-backpressure");
      return;
    }
    telemetryQueue.push({ kind, value });
    maximumPendingTelemetryRecords = Math.max(maximumPendingTelemetryRecords, telemetryQueue.length);
    pumpTelemetry();
  };
  const finishTelemetry = async () => {
    telemetryFinishing = true;
    pumpTelemetry();
    let timeout: ReturnType<typeof setTimeout> | undefined;
    const drained = async () => {
      while (!telemetryFailed && (telemetryInFlight || telemetryQueue.length)) {
        pumpTelemetry();
        if (telemetryInFlight) await telemetryInFlight;
      }
    };
    await Promise.race([drained(), new Promise<void>(resolve => {
      timeout = setTimeout(() => { telemetryFailed = true; fail("telemetry-sink-timeout"); resolve(); }, 5_000);
    })]);
    if (timeout) clearTimeout(timeout);
  };

  const handleResponse = (message: VoiceWorkerResponse) => {
    if (message.schemaVersion !== VOICE_WORKER_MESSAGE_VERSION || message.captureEpoch !== 1) return;
    if (message.type === "ready") ready?.();
    if (message.type === "error") fail(message.reason);
    if (message.type === "diagnostics" || message.type === "disposed") diagnostics = message.diagnostics;
    if (message.type === "disposed") { workerStopAcknowledged = true; workerDisposed?.(); }
    if (message.type === "block-timing") {
      const item = { ...message.timing };
      blockCount += 1;
      retain(blockTimings, blockCount, blockLimit, item);
      addStatistic(statistics.blockService, message.timing.workerServiceMs);
      addStatistic(message.timing.disposition === "stale-dropped" ? statistics.staleBlockService : statistics.processedBlockService,
        message.timing.workerServiceMs);
      const key = `${message.timing.disposition}:${message.timing.analyzedFrameCount}`;
      blockFrameCounts[key] = (blockFrameCounts[key] ?? 0) + 1;
      emitTelemetry("block-timing", item);
    }
    if (message.type === "signal-frame") {
      const arrivalMs = message.timing.mainResultArrivalMs ?? performance.now();
      const elapsed = arrivalMs - message.timing.sampleAvailabilityMs;
      const valid = message.timing.status === "valid" && Number.isFinite(elapsed) && elapsed >= 0 &&
        arrivalMs >= message.timing.workerReadyAtMs;
      const timing: VoiceFrameTiming = { ...message.timing, status: valid ? "valid" : "unavailable",
        ...(valid ? {} : { unavailableReason: "inconsistent-clock-mapping" as const }),
        mainResultArrivalMs: arrivalMs, rawSampleAvailabilityToMainResultMs: elapsed,
        sampleAvailabilityToMainResultMs: valid ? elapsed : null };
      const item = { ...message.frame, timing, ...(message.research ? { research: message.research } : {}) };
      if (message.research) {
        researchFrameCount += 1;
        addStatistic(statistics.modelInference, message.research.inferenceServiceMs);
        addStatistic(statistics.modelReady, message.research.elapsedFromFrameStartMs ?? null);
        addStatistic(statistics.modelArrival, message.research.sampleAvailabilityToModelReadyMs ?? null);
        addStatistic(statistics.targetReady, message.research.targetReferenceToModelReadyMs ?? null);
      }
      frameCount += 1;
      retain(frames, frameCount, frameLimit, item);
      invalidTimingCount += Number(!valid);
      addStatistic(statistics.frameService, timing.workerServiceMs);
      addStatistic(statistics.arrival, timing.sampleAvailabilityToMainResultMs ?? null);
      addStatistic(statistics.receipt, timing.blockEndToWorkerReceiptMs);
      if (message.frame.trackSegmentId !== previousSegment) {
        segmentCount += 1;
        if (segments.length < 64) segments.push(message.frame.trackSegmentId);
        previousSegment = message.frame.trackSegmentId;
      }
      if (message.frame.periodic && message.frame.f0Hz !== null) {
        periodicFrameCount += 1;
        const error = Math.abs(message.frame.f0Hz - 200);
        maximumKnownPitchErrorHz = Math.max(maximumKnownPitchErrorHz, error);
        correctKnownPitchFrameCount += Number(error <= 2);
      }
      emitTelemetry("frame", item);
    }
  };
  const contextStateChanged = () => {
    if (!stopping && sourceStartedAtMs > 0 && context.state !== "running") {
      worker.postMessage({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "clock-mapping",
        captureEpoch: 1, clockMapping: null });
      fail(`unexpected-audio-context-state:${context.state}`);
    }
  };
  const resultReceiver = new VoiceResultBatchReceiver(message => worker.postMessage(message));
  worker.addEventListener("message", (event: MessageEvent<VoiceWorkerResponse>) => {
    const received = resultReceiver.receive(event.data, 1, performance.now(), 1);
    for (const discarded of received.discardedFrames) {
      discardedResultFrameCount += 1;
      staleResultFrameCount += Number(discarded.discardReason === "result-age-exceeded");
      emitTelemetry("frame", { ...discarded.frame, timing: discarded.timing,
        deliveryDisposition: "discarded", discardReason: discarded.discardReason });
    }
    for (const message of received.messages) handleResponse(message);
  });
  worker.addEventListener("error", event => {
    const reason = `replay-worker-error:${event.message}`;
    fail(reason);
    initializationFailed?.(new Error(reason));
  });
  context.addEventListener("statechange", contextStateChanged);
  try {
    if (options.modelConfig) {
      researchModel = await new Promise<NonNullable<ReplayResult["researchModel"]>>((resolve, reject) => {
        const timer = setTimeout(() => { worker.removeEventListener("message", listener);
          reject(new Error("Research model initialization timeout.")); }, 30_000);
        const listener = (event: MessageEvent) => {
          if (event.data?.type !== "research-model-ready" && event.data?.type !== "research-model-error") return;
          clearTimeout(timer); worker.removeEventListener("message", listener);
          if (event.data.type === "research-model-error") reject(new Error(event.data.reason));
          else resolve({ qualification: event.data.qualification, modelSha256: event.data.modelSha256, loadMs: event.data.loadMs });
        };
        worker.addEventListener("message", listener);
        worker.postMessage({ type: "configure-research-model", config: options.modelConfig });
      });
    }
    await context.audioWorklet.addModule("/voice-capture-worklet.js");
    await context.resume();
    node = new AudioWorkletNode(context, "phenometric-voice-capture", { numberOfInputs: 1, numberOfOutputs: 1,
      outputChannelCount: [1] });
    node.addEventListener("processorerror", () => fail("replay-worklet-processor-error"));
    node.port.onmessage = event => {
      if (!isCurrentVoiceTransportDiagnostics(event.data, 1)) return;
      transportStop = { status: "complete", diagnostics: event.data };
      transportStopAcknowledged = true;
      transportDisposed?.();
    };
    node.connect(context.destination); // Output is zero: the synthetic replay is not played aloud.
    const mapping = captureVoiceClockMapping(context, performance, 1);
    const initialized = new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("Replay worker did not initialize.")), 5_000);
      ready = () => { clearTimeout(timer); resolve(); };
      initializationFailed = error => { clearTimeout(timer); reject(error); };
    });
    worker.postMessage({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "initialize", captureEpoch: 1,
      port: channel.port1, clockMapping: mapping, audioContextSampleRateHz: context.sampleRate,
      captureSettings: requestedAudioCaptureSettings(context.sampleRate, 1,
        { echoCancellation: false, noiseSuppression: false, autoGainControl: false }),
      taskContext: "ambient-speech-turn", noiseFloorRms: .002 }, [channel.port1]);
    node.port.postMessage({ type: "attach-port", port: channel.port2 }, [channel.port2]);
    node.port.postMessage({ type: "capture-epoch", captureEpoch: 1, clockMappingId: 1 });
    await initialized;
    if (context.state !== "running") throw new Error(`Audio context not running:${context.state}`);
    const buffer = context.createBuffer(1, context.sampleRate * 2, context.sampleRate);
    const samples = buffer.getChannelData(0);
    for (let i = 0; i < samples.length; i += 1) samples[i] = .12 * Math.sin(2 * Math.PI * 200 * i / context.sampleRate)
      + .03 * Math.sin(2 * Math.PI * 400 * i / context.sampleRate);
    source = context.createBufferSource();
    source.buffer = buffer; source.loop = true; source.connect(node);
    sourceStartedAtMs = performance.now();
    source.start();
    if (options.overloadAtMs !== undefined) overloadTimer = setTimeout(() => worker.postMessage({
      type: "research-stall", durationMs: options.overloadDurationMs ?? 250 }), options.overloadAtMs);
    if (options.mainStallAtMs !== undefined) mainStallTimer = setTimeout(() => {
      const until = performance.now() + (options.mainStallDurationMs ?? 300);
      while (performance.now() < until) { /* Controlled main-thread blocking; functional test only. */ }
    }, options.mainStallAtMs);
    await new Promise<void>(resolve => {
      finishPaced = resolve;
      if (errors.length) resolve();
      else pacedTimer = setTimeout(resolve, durationMs);
    });
    if (pacedTimer) clearTimeout(pacedTimer);
    sourceStoppedAtMs = performance.now();
    stopping = true;
    source.stop(); source.disconnect();
    const workerAcknowledgement = new Promise<void>(resolve => { workerDisposed = resolve; });
    const transportAcknowledgement = new Promise<void>(resolve => { transportDisposed = resolve; });
    node.port.postMessage({ type: "dispose" }); node.disconnect();
    worker.postMessage({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "dispose", captureEpoch: 1 });
    let stopTimeout: ReturnType<typeof setTimeout> | undefined;
    await Promise.race([Promise.all([workerAcknowledgement, transportAcknowledgement]), new Promise<void>(resolve => {
      stopTimeout = setTimeout(resolve, 3_000);
    })]);
    if (stopTimeout) clearTimeout(stopTimeout);
    if (!workerStopAcknowledged) fail("replay-worker-stop-timeout");
    if (!transportStopAcknowledged) fail("replay-worklet-stop-timeout");
    await finishTelemetry();
    const completedAtMs = performance.now();
    const frameService = quantiles(statistics.frameService);
    return { status: errors.length ? "failed" : "complete", evidence: "paced-synthetic-browser-replay",
      durationMs: completedAtMs - began, requestedSourceDurationMs: durationMs,
      pacedSourceDurationMs: sourceStoppedAtMs - sourceStartedAtMs,
      setupDurationMs: sourceStartedAtMs - began, teardownDurationMs: completedAtMs - sourceStoppedAtMs,
      sampleRateHz: context.sampleRate, researchModel, browser: navigator.userAgent,
      frames: chronological(frames, frameCount, frameLimit), errors, diagnostics, transportStop, workerStopAcknowledged,
      blockTimings: chronological(blockTimings, blockCount, blockLimit), maximumServiceMs: frameService.maximum,
      summary: { frameCount, blockCount, researchFrameCount, modelInferenceServiceMs: quantiles(statistics.modelInference),
        modelElapsedFromFrameStartMs: quantiles(statistics.modelReady),
        sampleAvailabilityToModelReadyMs: quantiles(statistics.modelArrival),
        targetReferenceToModelReadyMs: quantiles(statistics.targetReady),
        frameServiceScope: options.modelConfig ? "shared-feature-and-baseline-preparation-excludes-inference" : "baseline-dsp",
        completeBlockServiceScope: "complete-admitted-block-including-optional-model-completion-and-data-post-excludes-completion-post", retainedFrameLimit: frameLimit, retainedBlockLimit: blockLimit,
        numericStatisticsCapacityPerMetric: statisticsCapacity, frameServiceMs: frameService,
        completeBlockServiceMs: quantiles(statistics.blockService),
        completeProcessedBlockServiceMs: quantiles(statistics.processedBlockService),
        staleBlockServiceMs: quantiles(statistics.staleBlockService), blockFrameCounts,
        frameServiceBudgetMs: 5, frameServiceBudgetExceedances: statistics.frameService.filter(value => value > 5).length,
        frameCadenceMs: 10, frameCadenceDeadlineMisses: statistics.frameService.filter(value => value > 10).length,
        completeBlockServiceBudgetMs: 10, completeBlockServiceBudgetExceedances: statistics.processedBlockService.filter(value => value > 10).length,
        completeBlockCadenceMs: 20, completeBlockDeadlineMisses: statistics.blockService.filter(value => value > 20).length,
        discardedResultFrameCount, staleResultFrameCount,
        invalidFrameTimingCount: invalidTimingCount, sampleAvailabilityToMainResultMs: quantiles(statistics.arrival),
        blockEndToWorkerReceiptMs: quantiles(statistics.receipt), segments, segmentCount,
        periodicFrameCount, correctKnownPitchFrameCount, maximumKnownPitchErrorHz,
        rawTelemetry: { enabled: !!options.streamRawTelemetry, status: telemetryFailed ? "failed" : "complete",
          batchCount: telemetryBatchSequence, streamedRecordCount, maximumPendingRecords: maximumPendingTelemetryRecords,
          pendingRecordLimit: 200, maximumBatchRecords: 100 },
        note: "Whole replay numeric statistics with bounded retained frames; raw batch delivery is optional. No hardware acquisition, natural-speech, or display latency claim." } };
  } finally {
    stopping = true;
    if (mainStallTimer) clearTimeout(mainStallTimer);
    if (overloadTimer) clearTimeout(overloadTimer);
    if (pacedTimer) clearTimeout(pacedTimer);
    context.removeEventListener("statechange", contextStateChanged);
    try { source?.stop(); } catch { /* already stopped */ }
    source?.disconnect(); node?.port.postMessage({ type: "dispose" }); node?.disconnect();
    node?.port.close(); channel.port1.close(); channel.port2.close(); worker.terminate();
    await context.close();
  }
}

export async function runVoiceModelExportParity(): Promise<Record<string, unknown>> {
  const worker = new Worker(new URL("./research-voice-model-worker.ts", import.meta.url), { type: "module" });
  try { return await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("Export parity timed out.")), 30_000);
    worker.onmessage = event => { clearTimeout(timer); resolve(event.data); };
    worker.onerror = event => { clearTimeout(timer); reject(new Error(event.message)); };
    worker.postMessage({ type: "run-export-parity" });
  }); } finally { worker.terminate(); }
}
declare global { interface Window {
  runVoiceResearchReplay: typeof runVoiceResearchReplay;
  runVoiceModelExportParity: typeof runVoiceModelExportParity;
  recordVoiceResearchBatch?: (batch: ReplayTelemetryBatch) => Promise<void>;
} }
window.runVoiceModelExportParity = runVoiceModelExportParity;
window.runVoiceResearchReplay = runVoiceResearchReplay;
