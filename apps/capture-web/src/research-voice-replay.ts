import { NaturalSourceCompletionFence } from "./voice-natural-completion.js";
import { VoiceRenderLeaseIssuer, VOICE_RENDER_LEASE_RENEWAL_MS } from "./voice-render-freshness.js";
import { VoiceClockProbeTracker, type VoiceClockObservation } from "./voice-clock-probe.js";
import type { VoiceResearchModelConfig, VoiceResearchConfigDocument, VoiceResearchIdentity } from "./voice-research-hook.js";
import { VoiceResultBatchReceiver } from "./voice-result-transport.js";
import { captureVoiceClockMapping, type VoiceFrameTiming } from "./voice-stream-timing.js";
import { VOICE_WORKER_MESSAGE_VERSION, isCurrentVoiceTransportDiagnostics, requestedAudioCaptureSettings,
  type VoiceTransportStopResult, type VoiceWorkerResponse } from "./voice-worker-protocol.js";

export interface ReplayTelemetryBatch {
  schemaVersion: "phenometric.voice-replay-telemetry.v1";
  sequence: number;
  records: Array<{ kind: "frame" | "block-timing" | "clock-probe"; value: Record<string, unknown> }>;
}
export interface NaturalPcmSource {
  id: string; pcmUrl: string; pcmSha256: string; sampleCount: number;
  adapter: "ptdb-prefix-192-v1" | "keele-zero-phase-v1" | "synthetic-index-v1";
}
export interface ReplayOptions {
  /** Full finite 48 kHz float32 source; local research assets only. */
  naturalPcm?: NaturalPcmSource;
  durationMs?: number;
  /** Independent 1 Hz clock diagnostics; never re-anchors or changes admission. */
  clockDiagnostics?: boolean;
  /** Research route only; the production capture entry never supplies a model. */
  modelConfig?: VoiceResearchModelConfig;
  modelConfigDocument?: VoiceResearchConfigDocument;
  syntheticTone?: "200hz" | "300hz-harmonic";
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
  evidence: "paced-synthetic-browser-replay" | "finite-natural-pcm-browser-replay";
  naturalSource: Record<string, unknown> | null;
  durationMs: number;
  requestedSourceDurationMs: number;
  pacedSourceDurationMs: number;
  setupDurationMs: number;
  teardownDurationMs: number;
  sampleRateHz: number;
  researchModel: (VoiceResearchIdentity & { qualification: "unqualified-functional-fixture"; loadMs: number }) | null;
  browser: string;
  frames: Array<Record<string, unknown>>;
  errors: string[];
  diagnostics: unknown;
  transportStop: VoiceTransportStopResult;
  workerStopAcknowledged: boolean;
  blockTimings: Array<Record<string, unknown>>;
  maximumServiceMs: number | null;
  clockProbes: VoiceClockObservation[];
  clockDiagnostics: { enabled: boolean; intervalMs: 1000; maximumPending: 1; retainedLimit: 64;
    fixedMapping: ReturnType<typeof captureVoiceClockMapping> | null };
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
  if (options.syntheticTone !== undefined && options.syntheticTone !== "200hz" && options.syntheticTone !== "300hz-harmonic") throw new Error("Unknown synthetic tone.");
  const knownPitchHz = options.naturalPcm ? null : options.syntheticTone === "300hz-harmonic" ? 300 : 200;
  let naturalSamples: Float32Array | null = null;
  if (options.naturalPcm) {
    const fixture = options.naturalPcm;
    if (!options.modelConfig || options.syntheticTone || options.durationMs !== undefined ||
        !["ptdb-prefix-192-v1", "keele-zero-phase-v1", "synthetic-index-v1"].includes(fixture.adapter) ||
        !Number.isSafeInteger(fixture.sampleCount) || fixture.sampleCount < 1920 || fixture.sampleCount > 48000 * 120 ||
        !/^[a-f0-9]{64}$/.test(fixture.pcmSha256)) throw new Error("Invalid natural PCM contract.");
    const url = new URL(fixture.pcmUrl, location.href);
    if (url.origin !== location.origin) throw new Error("Natural PCM must stay local to the research origin.");
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Natural PCM HTTP ${response.status}`);
    const bytes = await response.arrayBuffer();
    const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), v => v.toString(16).padStart(2, "0")).join("");
    if (digest !== fixture.pcmSha256 || bytes.byteLength !== fixture.sampleCount * 4) throw new Error("Natural PCM identity mismatch.");
    naturalSamples = new Float32Array(bytes);
    if (!naturalSamples.every(Number.isFinite)) throw new Error("Nonfinite natural PCM.");
    if (fixture.adapter === "ptdb-prefix-192-v1" && naturalSamples.subarray(0, 192).some(v => v !== 0)) throw new Error("Missing exact PTDB zero prefix.");
  }
  const durationMs = naturalSamples ? naturalSamples.length / 48 : options.durationMs ?? 4_000;
  if (!Number.isFinite(durationMs) || durationMs < (naturalSamples ? 40 : 1_000) || durationMs > 31 * 60 * 1_000) throw new Error("Invalid replay duration.");
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
  const leaseIssuer = new VoiceRenderLeaseIssuer(performance);
  let leaseTimer: ReturnType<typeof setInterval> | undefined;
  let initialLeaseAcknowledged: (() => void) | undefined;
  const stopLeases = () => { if (leaseTimer !== undefined) clearInterval(leaseTimer); leaseTimer = undefined; leaseIssuer.stop(); };
  const channel = new MessageChannel();
  const frames: Array<Record<string, unknown>> = [];
  const blockTimings: Array<Record<string, unknown>> = [];
  const errors: string[] = [];
  const clockProbes: VoiceClockObservation[] = [];
  let clockProbeTracker: VoiceClockProbeTracker | null = null;
  let clockProbeCount = 0;
  let clockProbeObservedCount = 0;
  let clockProbeTimer: ReturnType<typeof setInterval> | undefined;
  let fixedClockMapping: ReturnType<typeof captureVoiceClockMapping> | null = null;
  const statisticsCapacity = 200_000; // >31 min at the declared 100 Hz hop; no full-frame growth.
  const statistics = { frameService: [] as number[], blockService: [] as number[],
    processedBlockService: [] as number[], staleBlockService: [] as number[], freshnessWithheldBlockService: [] as number[],
    renderAgeUpperBound: [] as number[],
    arrival: [] as number[], receipt: [] as number[], yinService: [] as number[], modelInference: [] as number[], modelReady: [] as number[], modelArrival: [] as number[], targetReady: [] as number[] };
  const segments: string[] = [];
  let previousSegment = "";
  let segmentCount = 0;
  let frameCount = 0;
  let researchFrameCount = 0;
  let discardedResearchFrameCount = 0;
  let researchVoicedFrameCount = 0;
  let researchCorrectKnownPitchFrameCount = 0;
  let blockCount = 0;
  let invalidTimingCount = 0;
  let certifiedFrameCount = 0;
  const hostWithheldReasons: Record<string, number> = {};
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
  let naturalArmed: ((value: Record<string, unknown>) => void) | undefined;
  const naturalCompletion = new NaturalSourceCompletionFence();
  let naturalCredits: Record<string, unknown> | null = null;
  let naturalCompletionReceipt: { status: "complete" | "timeout"; missing: string[] } | null = null;
  let naturalOrigin: Record<string, unknown> | null = null;
  let naturalDrain: Record<string, unknown> | null = null;
  let sourceEnded = false;
  let scheduledStartAudioFrame: number | null = null;
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
  const emitTelemetry = (kind: "frame" | "block-timing" | "clock-probe", value: Record<string, unknown>) => {
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

  const recordClockProbe = (observation: VoiceClockObservation | null) => {
    if (!observation) return;
    clockProbeCount += 1;
    clockProbeObservedCount += Number(observation.renderObservation.status === "observed");
    clockProbes[(clockProbeCount - 1) % 64] = observation;
    emitTelemetry("clock-probe", { ...observation });
  };
  const stopClockProbes = () => {
    if (clockProbeTimer) clearInterval(clockProbeTimer);
    clockProbeTimer = undefined;
    recordClockProbe(clockProbeTracker?.expire("stopped") ?? null);
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
      addStatistic(message.timing.disposition === "processed" ? statistics.processedBlockService :
        message.timing.disposition === "freshness-withheld" ? statistics.freshnessWithheldBlockService : statistics.staleBlockService,
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
      const item = { ...message.frame, captureEpoch: message.captureEpoch, timing, renderLease: message.renderLease ?? null,
        ...(message.research ? { research: message.research } : {}) };
      if (timing.renderFreshness?.status === "certified") {
        certifiedFrameCount += 1;
        addStatistic(statistics.renderAgeUpperBound, timing.renderFreshness.renderAgeUpperBoundMs);
      } else fail("delivered-frame-without-render-freshness-certificate");
      if (message.research) {
        researchFrameCount += 1;
        if (message.research.rawYin) addStatistic(statistics.yinService, message.research.rawYin.serviceMs ?? null);
        if (message.research.state === "voiced") {
          researchVoicedFrameCount += 1;
          researchCorrectKnownPitchFrameCount += Number(message.research.f0Hz !== null && knownPitchHz !== null && Math.abs(message.research.f0Hz - knownPitchHz) <= 2);
        }
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
      if (knownPitchHz !== null && message.frame.periodic && message.frame.f0Hz !== null) {
        periodicFrameCount += 1;
        const error = Math.abs(message.frame.f0Hz - knownPitchHz);
        maximumKnownPitchErrorHz = Math.max(maximumKnownPitchErrorHz, error);
        correctKnownPitchFrameCount += Number(error <= 2);
      }
      emitTelemetry("frame", item);
    }
  };
  const contextStateChanged = () => {
    if (!stopping && sourceStartedAtMs > 0 && context.state !== "running") {
      leaseIssuer.invalidate();
      worker.postMessage({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "clock-mapping",
        captureEpoch: 1, clockMapping: null });
      fail(`unexpected-audio-context-state:${context.state}`);
    }
  };
  const resultReceiver = new VoiceResultBatchReceiver(message => worker.postMessage(message),
    { policy: "required-render-lease-v1", issuer: leaseIssuer, mainTimeOriginMs: performance.timeOrigin });
  worker.addEventListener("message", (event: MessageEvent<VoiceWorkerResponse>) => {
    const control = event.data as unknown as Record<string, unknown>;
    if (options.naturalPcm && control.type === "research-natural-drained" && control.captureEpoch === 1) {
      naturalDrain = control;
      if (control.eventOverflow !== 0) fail("natural-event-ledger-overflow");
      naturalCompletion.receive("worker"); return;
    }
    let received: ReturnType<VoiceResultBatchReceiver["receive"]>;
    try { received = resultReceiver.receive(event.data, 1, performance.now(), 1); }
    catch { fail("invalid-voice-result-transport"); return; }
    for (const discarded of received.discardedFrames) {
      discardedResultFrameCount += 1;
      staleResultFrameCount += Number(discarded.discardReason === "result-age-exceeded" || discarded.discardReason === "expired-lease");
      hostWithheldReasons[discarded.discardReason] = (hostWithheldReasons[discarded.discardReason] ?? 0) + 1;
      discardedResearchFrameCount += Number(!!discarded.research);
      emitTelemetry("frame", { ...discarded.frame, captureEpoch: discarded.captureEpoch,
        ...(discarded.research ? { research: discarded.research } : {}), timing: discarded.timing, renderLease: discarded.renderLease ?? null,
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
          else resolve({ qualification: event.data.qualification, modelSha256: event.data.modelSha256, loadMs: event.data.loadMs,
            configurationSha256: event.data.configurationSha256, modelManifestSha256: event.data.modelManifestSha256,
            checkpointSha256: event.data.checkpointSha256, estimatorContract: event.data.estimatorContract });
        };
        worker.addEventListener("message", listener);
        worker.postMessage({ type: "configure-research-model", config: options.modelConfig, configDocument: options.modelConfigDocument });
      });
    }
    await context.audioWorklet.addModule("/voice-capture-worklet.js");
    await context.resume();
    node = new AudioWorkletNode(context, "phenometric-voice-capture", { numberOfInputs: 1, numberOfOutputs: 1,
      outputChannelCount: [1] });
    node.addEventListener("processorerror", () => fail("replay-worklet-processor-error"));
    node.port.onmessage = event => {
      if (options.naturalPcm && event.data?.type === "research-natural-armed") { naturalArmed?.(event.data); return; }
      if (options.naturalPcm && event.data?.type === "research-natural-credits-drained") {
        naturalCredits = event.data;
        if (event.data.captureEpoch !== 1 || event.data.scheduledStartAudioFrame !== scheduledStartAudioFrame ||
            event.data.sourceSampleCount !== naturalSamples!.length || event.data.inFlightBlockCount !== 0 || event.data.pendingSampleCount !== 0)
          fail("natural-credit-drain-mismatch");
        naturalCompletion.receive("credits");
        return;
      }
      if (options.naturalPcm && event.data?.type === "research-natural-ended") {
        naturalOrigin = event.data;
        naturalCompletion.receive("origin");
        if (event.data.timeline?.schemaVersion !== "phenometric.voice-natural-timeline.v2" ||
            event.data.timeline?.status !== "complete" || event.data.timeline?.overflow !== 0)
          fail(`natural-source-timeline-failed:${event.data.timeline?.error ?? "missing-receipt"}`);
        if (event.data.observedStartAudioFrame !== scheduledStartAudioFrame || event.data.sourceOffset !== 0 ||
            event.data.captureEndSampleExclusive !== naturalSamples!.length) fail("natural-source-origin-mismatch");
        return;
      }
      if (leaseIssuer.acknowledge(event.data)) {
        if (event.data.disposition === "accepted") initialLeaseAcknowledged?.();
        return;
      }
      if (event.data?.type === "clock-probe-result") {
        recordClockProbe(clockProbeTracker?.receive(event.data) ?? null);
        return;
      }
      if (!isCurrentVoiceTransportDiagnostics(event.data, 1)) return;
      transportStop = { status: "complete", diagnostics: event.data };
      transportStopAcknowledged = true;
      transportDisposed?.();
    };
    node.connect(context.destination); // Output is zero: the synthetic replay is not played aloud.
    const mapping = captureVoiceClockMapping(context, performance, 1);
    fixedClockMapping = mapping;
    const initialized = new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("Replay worker did not initialize.")), 5_000);
      ready = () => { clearTimeout(timer); resolve(); };
      initializationFailed = error => { clearTimeout(timer); reject(error); };
    });
    worker.postMessage({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "initialize", captureEpoch: 1, freshnessPolicy: "required-render-lease-v1",
      port: channel.port1, clockMapping: mapping, audioContextSampleRateHz: context.sampleRate,
      captureSettings: requestedAudioCaptureSettings(context.sampleRate, 1,
        { echoCancellation: false, noiseSuppression: false, autoGainControl: false }),
      taskContext: "ambient-speech-turn", noiseFloorRms: .002 }, [channel.port1]);
    node.port.postMessage({ type: "attach-port", port: channel.port2 }, [channel.port2]);
    node.port.postMessage({ type: "capture-epoch", captureEpoch: 1, clockMappingId: 1 });
    await initialized;
    const renewLease = () => {
      if (stopping || context.state !== "running") return;
      const request = leaseIssuer.issue(1, 1);
      if (request) node!.port.postMessage(request);
    };
    // Start the native 40 ms frame grid only after an accepted initial lease.
    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => { initialLeaseAcknowledged = undefined;
        reject(new Error("Initial render freshness lease timed out.")); }, 5_000);
      initialLeaseAcknowledged = () => { clearTimeout(timeout); initialLeaseAcknowledged = undefined; resolve(); };
      renewLease();
    });
    leaseTimer = setInterval(renewLease, VOICE_RENDER_LEASE_RENEWAL_MS);
    if (context.state !== "running") throw new Error(`Audio context not running:${context.state}`);
    if (naturalSamples && context.sampleRate !== 48000) throw new Error("Natural source requires exact 48 kHz rendering.");
    const buffer = context.createBuffer(1, naturalSamples?.length ?? context.sampleRate * 2, context.sampleRate);
    const samples = buffer.getChannelData(0);
    if (naturalSamples) samples.set(naturalSamples);
    else for (let i = 0; i < samples.length; i += 1) samples[i] = options.syntheticTone === "300hz-harmonic"
      ? .075 * (Math.sin(2 * Math.PI * 300 * i / context.sampleRate) + .5 * Math.sin(2 * Math.PI * 600 * i / context.sampleRate) + .25 * Math.sin(2 * Math.PI * 900 * i / context.sampleRate))
      : .12 * Math.sin(2 * Math.PI * 200 * i / context.sampleRate) + .03 * Math.sin(2 * Math.PI * 400 * i / context.sampleRate);
    source = context.createBufferSource();
    source.buffer = buffer; source.loop = !naturalSamples;
    if (naturalSamples) {
      scheduledStartAudioFrame = Math.ceil((context.currentTime * context.sampleRate + 12000) / 1920) * 1920;
      await new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error("Natural source arm timed out.")), 1000);
        naturalArmed = receipt => { clearTimeout(timeout); naturalArmed = undefined;
          if (receipt.status !== "armed") reject(new Error("Natural source arm rejected.")); else resolve(); };
        node!.port.postMessage({ type: "research-natural-source", captureEpoch: 1,
          startAudioFrame: scheduledStartAudioFrame, sampleCount: naturalSamples!.length });
      });
      if (context.currentTime * context.sampleRate >= scheduledStartAudioFrame) throw new Error("Natural source missed scheduled start.");
      source.connect(node);
      source.onended = () => { sourceEnded = true; finishPaced?.(); };
      sourceStartedAtMs = performance.now();
      source.start(scheduledStartAudioFrame / context.sampleRate);
    } else {
      source.connect(node); sourceStartedAtMs = performance.now(); source.start();
    }
    if (options.clockDiagnostics) {
      clockProbeTracker = new VoiceClockProbeTracker(mapping, 1, performance);
      const probe = () => {
        recordClockProbe(clockProbeTracker!.expire("timeout"));
        const request = clockProbeTracker!.begin(context, performance.now() - sourceStartedAtMs);
        if (request) node!.port.postMessage(request);
      };
      probe();
      clockProbeTimer = setInterval(probe, 1_000);
    }
    if (options.overloadAtMs !== undefined) overloadTimer = setTimeout(() => worker.postMessage({
      type: "research-stall", durationMs: options.overloadDurationMs ?? 250 }), options.overloadAtMs);
    if (options.mainStallAtMs !== undefined) mainStallTimer = setTimeout(() => {
      const until = performance.now() + (options.mainStallDurationMs ?? 300);
      while (performance.now() < until) { /* Controlled main-thread blocking; functional test only. */ }
    }, options.mainStallAtMs);
    await new Promise<void>(resolve => {
      finishPaced = resolve;
      if (errors.length) resolve();
      else pacedTimer = setTimeout(() => {
        if (naturalSamples) fail("natural-source-ended-timeout");
        resolve();
      }, durationMs + (naturalSamples ? 1500 : 0));
    });
    if (pacedTimer) clearTimeout(pacedTimer);
    sourceStoppedAtMs = performance.now();
    if (naturalSamples) {
      // Keep leases, acknowledgements, clock probes and the telemetry sink alive.
      // onended is only the source endpoint, never an inference completion claim.
      naturalCompletionReceipt = await naturalCompletion.wait(6000);
      if (naturalCompletionReceipt.status !== "complete") fail("natural-drain-timeout:" + naturalCompletionReceipt.missing.join(","));
      const drain = naturalDrain as Record<string, unknown> | null;
      if (!sourceEnded || !naturalOrigin || drain?.status !== "drained" || drain?.pumpIdle !== true) fail("natural-completion-fence-failed");
    }
    stopping = true;
    stopClockProbes();
    if (leaseTimer !== undefined) clearInterval(leaseTimer);
    leaseTimer = undefined;
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
    return { status: errors.length ? "failed" : "complete", evidence: naturalSamples ? "finite-natural-pcm-browser-replay" : "paced-synthetic-browser-replay",
      naturalSource: options.naturalPcm ? { ...options.naturalPcm, scheduledStartAudioFrame, sourceEnded,
        origin: naturalOrigin, drain: naturalDrain, credits: naturalCredits, completion: naturalCompletionReceipt,
        drainTimeoutMs: 5000, mainDrainTimeoutMs: 6000,
        finalPartialPolicy: "discard-retain-count-no-padding", sourceOffset: 0,
        referenceOffsetSamples: options.naturalPcm.adapter === "ptdb-prefix-192-v1" ? 192 : 0 } : null,
      durationMs: completedAtMs - began, requestedSourceDurationMs: durationMs,
      pacedSourceDurationMs: sourceStoppedAtMs - sourceStartedAtMs,
      setupDurationMs: sourceStartedAtMs - began, teardownDurationMs: completedAtMs - sourceStoppedAtMs,
      sampleRateHz: context.sampleRate, researchModel, browser: navigator.userAgent,
      frames: chronological(frames, frameCount, frameLimit), errors, diagnostics, transportStop, workerStopAcknowledged,
      blockTimings: chronological(blockTimings, blockCount, blockLimit), maximumServiceMs: frameService.maximum,
      clockProbes: clockProbeCount <= 64 ? clockProbes : [...clockProbes.slice(clockProbeCount % 64), ...clockProbes.slice(0, clockProbeCount % 64)],
      clockDiagnostics: { enabled: !!options.clockDiagnostics, intervalMs: 1000, maximumPending: 1, retainedLimit: 64,
        fixedMapping: options.clockDiagnostics ? fixedClockMapping : null },
      summary: { frameCount, blockCount, researchFrameCount, discardedResearchFrameCount, researchVoicedFrameCount, researchCorrectKnownPitchFrameCount, knownPitchHz, knownPitchDiagnosticToleranceHz: 2, rawYinServiceMs: quantiles(statistics.yinService), clockProbeCount, clockProbeObservedCount, modelInferenceServiceMs: quantiles(statistics.modelInference),
        modelElapsedFromFrameStartMs: quantiles(statistics.modelReady),
        sampleAvailabilityToModelReadyMs: quantiles(statistics.modelArrival),
        targetReferenceToModelReadyMs: quantiles(statistics.targetReady),
        frameServiceScope: options.modelConfig ? "shared-feature-baseline-and-optional-yin-preparation-excludes-inference" : "baseline-dsp",
        completeBlockServiceScope: "complete-admitted-block-including-optional-model-completion-and-data-post-excludes-completion-post", retainedFrameLimit: frameLimit, retainedBlockLimit: blockLimit,
        numericStatisticsCapacityPerMetric: statisticsCapacity, frameServiceMs: frameService,
        completeBlockServiceMs: quantiles(statistics.blockService),
        completeProcessedBlockServiceMs: quantiles(statistics.processedBlockService),
        staleBlockServiceMs: quantiles(statistics.staleBlockService),
        freshnessWithheldBlockServiceMs: quantiles(statistics.freshnessWithheldBlockService), blockFrameCounts,
        freshnessPolicy: "required-render-lease-v1", certifiedFrameCount,
        renderAgeUpperBoundMs: quantiles(statistics.renderAgeUpperBound), hostWithheldReasons,
        freshnessLeaseControl: { ...leaseIssuer.statistics, renewalMs: VOICE_RENDER_LEASE_RENEWAL_MS,
          maximumAgeMs: 100, pendingRequestLimit: 1, ledgerEntryLimit: 8 },
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
        note: naturalSamples ? "Finite full-source natural PCM through scheduled AudioBufferSource, worklet, model worker and main delivery. Descriptive software timing; no hardware acquisition or display latency claim." : "Whole replay numeric statistics with bounded retained frames; raw batch delivery is optional. No hardware acquisition, natural-speech, or display latency claim." } };
  } finally {
    stopping = true;
    stopLeases();
    stopClockProbes();
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
/** Separate bounded arithmetic fixture; does not claim natural-audio worklet timing. */
export async function runVoiceHybridPcmParity(options: { config: VoiceResearchModelConfig;
  configDocument: VoiceResearchConfigDocument; pcmBytes: number[]; pcmSha256: string; windowEndSamples: number[] }): Promise<Record<string, unknown>> {
  if (options.pcmBytes.length > 960000 * 4) throw new Error("Oversized parity PCM.");
  const worker = new Worker(new URL("./research-voice-hybrid-parity-worker.ts", import.meta.url), { type: "module" });
  const pcm = Uint8Array.from(options.pcmBytes).buffer;
  try { return await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("Hybrid arithmetic parity timed out.")), 60_000);
    worker.onmessage = event => { clearTimeout(timer); resolve(event.data); };
    worker.onerror = event => { clearTimeout(timer); reject(new Error(event.message)); };
    worker.postMessage({ config: options.config, configDocument: options.configDocument, pcm, pcmSha256: options.pcmSha256,
      windowEndSamples: options.windowEndSamples }, [pcm]);
  }); } finally { worker.terminate(); }
}
declare global { interface Window {
  runVoiceResearchReplay: typeof runVoiceResearchReplay;
  runVoiceModelExportParity: typeof runVoiceModelExportParity;
  runVoiceHybridPcmParity: typeof runVoiceHybridPcmParity;
  recordVoiceResearchBatch?: (batch: ReplayTelemetryBatch) => Promise<void>;
} }
window.runVoiceHybridPcmParity = runVoiceHybridPcmParity;
window.runVoiceModelExportParity = runVoiceModelExportParity;
window.runVoiceResearchReplay = runVoiceResearchReplay;
