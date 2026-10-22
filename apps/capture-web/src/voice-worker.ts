/// <reference lib="webworker" />

import type {
  AudioStreamDiagnostics,
  VoiceTaskContext
} from "@phenometrix/contracts";
import {
  evaluateVoiceQuality,
  type AmbientVoiceFrame
} from "@phenometrix/ambient-core";
import {
  BoundedPcmRingBuffer,
  analyzeVoiceWindow
} from "./voice-dsp.js";
import {
  VOICE_DSP_PROCESSOR_REF,
  VOICE_MAXIMUM_BLOCK_AGE_MS,
  VOICE_SIGNAL_FRAME_VERSION,
  VOICE_WORKER_MESSAGE_VERSION,
  VOICE_WORKLET_MESSAGE_VERSION,
  audioPipelineProvenance,
  type VoiceWorkerRequest,
  type VoiceWorkerResponse,
  type VoiceWorkerBatchRecord,
  VOICE_MAXIMUM_RESULT_BATCHES, VOICE_MAXIMUM_RESULT_RECORDS, VOICE_MAXIMUM_PCM_BLOCK_MS,
  type VoiceWorkletPcmBlock
} from "./voice-worker-protocol.js";

import {
  VOICE_TIMING_CONTRACT_VERSION, audioTimeInMainClock,
  performanceTimeInMainClock, validVoiceClockMapping, validateVoiceDurations,
  type VoiceClockMapping, type VoiceFrameTiming
} from "./voice-stream-timing.js";

import { BoundedVoiceAsyncPump } from "./voice-async-pump.js";
import type { VoiceResearchHook } from "./voice-research-hook.js";
import type { VoicePitchFeaturesV2 } from "./voice-pitch-features-v2.js";

const worker = self as DedicatedWorkerGlobalScope;
let researchHook: VoiceResearchHook | null = null;
let featureGeneration = 0;
let workerDisposed = false;
let researchFailed = false;
let researchCancelledBlockCount = 0;
let researchCancelledFrameCount = 0;
let researchQueueDroppedBlockCount = 0;
let researchQueueDroppedSampleCount = 0;
let activeResearchCancel: (() => void) | null = null;
interface ResearchPcmRequest { event: MessageEvent<unknown>; sourcePort: MessagePort; receivedAtLocalMs: number; recycled: boolean }
let pendingResearchFrames: Array<{ record: Extract<VoiceWorkerResponse, { type: "signal-frame" }>;
  features: VoicePitchFeaturesV2; startedAt: number }> | null = null;

/** Only the separate research worker imports and calls this before initialization. */
export function configureVoiceResearchHook(hook: VoiceResearchHook): void {
  if (port || researchHook || workerDisposed) throw new Error("Research hook must be configured once before initialization.");
  researchHook = hook;
}
function recycleResearchRequest(request: ResearchPcmRequest): void {
  if (request.recycled) return;
  request.recycled = true;
  const block = request.event.data as VoiceWorkletPcmBlock;
  if (block?.buffer instanceof ArrayBuffer && Number.isSafeInteger(block.transportId)) {
    request.sourcePort.postMessage({ type: "recycle", transportId: block.transportId, buffer: block.buffer }, [block.buffer]);
  }
}
const researchPump = new BoundedVoiceAsyncPump<ResearchPcmRequest>(
  request => handleBlock(request.event, request.sourcePort, request),
  request => {
    const block = request.event.data as VoiceWorkletPcmBlock;
    if (block.captureEpoch === captureEpoch) {
      researchQueueDroppedBlockCount += 1;
      researchQueueDroppedSampleCount += block.buffer instanceof ArrayBuffer ? block.buffer.byteLength / 4 : 0;
    }
    recycleResearchRequest(request);
  },
  error => {
    if (workerDisposed || researchFailed) return;
    researchFailed = true;
    cancelResearchWork();
    post({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION,
      type: "error", captureEpoch, reason: `research-model-failed:${String(error)}` });
  }
);
function receivePcm(event: MessageEvent<unknown>, sourcePort: MessagePort): void {
  if (!researchHook) { void handleBlock(event, sourcePort); return; }
  const request = { event, sourcePort, receivedAtLocalMs: performance.now(), recycled: false };
  if (researchFailed) { recycleResearchRequest(request); return; }
  researchPump.submit(request);
}
function cancelResearchWork(): void {
  activeResearchCancel?.();
  researchPump.flush();
}


let captureEpoch = 0;
let port: MessagePort | null = null;
let ring: BoundedPcmRingBuffer | null = null;
let sampleRateHz = 48_000;
let acquisitionSampleRateHz = 48_000;
let taskContext: VoiceTaskContext = "quiet-calibration";
let clockMapping: VoiceClockMapping | null = null;
let segmentNumber = 0;
let invalidClockMappingCount = 0;
let droppedTransportBlockCount = 0;
let droppedTransportSampleCount = 0;
let maximumInFlightBlocks = 0;
let staleTransportBlockCount = 0;
let staleTransportSampleCount = 0;
let browserProcessing = {
  echoCancellation: false,
  noiseSuppression: false,
  autoGainControl: false
};
let noiseFloorRms = 0.002;
let lastBlockSequence = 0;
let lastAbsoluteSampleIndex = 0;
let lastBlockAcquiredAtMs: number | null = null;
let nextAnalysisSampleIndex = 0;
let processedFrameCount = 0;
let receivedBlockCount = 0;
let lostBlockCount = 0;
let maximumBlockGapMs = 0;
let currentBlockGapMs = 0;
let timestampRegressionCount = 0;
let priorBands: readonly number[] | null = null;
let lastNucleusAtMs = Number.NEGATIVE_INFINITY;
let priorNucleusScore = 0;
const latencyValues: number[] = [];
const recentBlockContinuity: Array<{
  atMs: number;
  received: number;
  lost: number;
}> = [];

let nextResultBatchId = 0;
const inFlightResults = new Map<number, { epoch: number; mappingId: number; segment: number }>(); // Origins survive resets.
let pendingResultRecords: VoiceWorkerBatchRecord[] | null = null;
let maximumInFlightResultBatches = 0;
let resultBackpressureBlockCount = 0;
let resultBackpressureSampleCount = 0;
let resultBackpressureEpisodeCount = 0;
let lastResultBackpressureSpan: AudioStreamDiagnostics["lastResultBackpressureSpan"];
let resultBackpressured = false;
let hostDiscardedFrameCount = 0;
let hostStaleFrameCount = 0;
let oversizedPcmBlockCount = 0;

function post(message: VoiceWorkerResponse): void {
  if (pendingResultRecords && (message.type === "signal-frame" || message.type === "diagnostics")) {
    if (pendingResultRecords.length >= VOICE_MAXIMUM_RESULT_RECORDS) throw new Error("voice-result-batch-capacity-exceeded");
    pendingResultRecords.push(message);
  } else worker.postMessage(message);
}

function diagnostics(): AudioStreamDiagnostics {
  const sortedLatency = [...latencyValues].sort(
    (left, right) => left - right
  );
  return {
    receivedBlockCount,
    processedFrameCount,
    lostBlockCount,
    lostBlockFraction:
      lostBlockCount /
      Math.max(1, receivedBlockCount + lostBlockCount),
    maximumBlockGapMs,
    p95FeatureLatencyMs:
      sortedLatency.length === 0
        ? 0
        : sortedLatency[
            Math.min(
              sortedLatency.length - 1,
              Math.floor(sortedLatency.length * 0.95)
            )
          ],
    timestampRegressionCount,
    ringBufferCapacitySamples: ring?.capacity ?? 0,
    invalidClockMappingCount, droppedTransportBlockCount,
    droppedTransportSampleCount, maximumInFlightBlocks,
    ...(researchHook ? { researchCancelledBlockCount, researchCancelledFrameCount,
      researchQueueDroppedBlockCount, researchQueueDroppedSampleCount,
      researchMaximumQueuedBlocks: researchPump.maximumQueued } : {}),
    inFlightResultBatchCount: inFlightResults.size, maximumInFlightResultBatches, resultBackpressureBlockCount, resultBackpressureSampleCount,
    resultBackpressureEpisodeCount,
    ...(lastResultBackpressureSpan ? { lastResultBackpressureSpan: { ...lastResultBackpressureSpan } } : {}),
    hostDiscardedFrameCount, hostStaleFrameCount, oversizedPcmBlockCount,
    staleTransportBlockCount, staleTransportSampleCount, latencyStatisticSampleCount: latencyValues.length
  };
}

function resetFeatureState(): void {
  featureGeneration += 1;
  researchHook?.reset();
  ring?.clear();
  nextAnalysisSampleIndex = 0;
  priorBands = null;
  lastNucleusAtMs = Number.NEGATIVE_INFINITY;
  priorNucleusScore = 0;
  segmentNumber += 1;
}

function reset(epoch: number, nextNoiseFloorRms?: number): void {
  // PCM uses a separate MessagePort. Updating the epoch and reference in one
  // worker turn prevents a new-epoch frame from carrying the prior noise floor.
  if (nextNoiseFloorRms !== undefined) noiseFloorRms = nextNoiseFloorRms;
  cancelResearchWork();
  captureEpoch = epoch;
  resetFeatureState();
  invalidClockMappingCount = 0;
  droppedTransportBlockCount = 0;
  droppedTransportSampleCount = 0;
  maximumInFlightBlocks = 0;
  staleTransportBlockCount = 0;
  staleTransportSampleCount = 0;
  pendingResultRecords = null;
  maximumInFlightResultBatches = inFlightResults.size;
  resultBackpressureBlockCount = 0;
  resultBackpressureSampleCount = 0;
  resultBackpressureEpisodeCount = 0;
  lastResultBackpressureSpan = undefined;
  resultBackpressured = false;
  hostDiscardedFrameCount = 0;
  hostStaleFrameCount = 0;
  oversizedPcmBlockCount = 0;
  researchCancelledBlockCount = 0;
  researchCancelledFrameCount = 0;
  researchQueueDroppedBlockCount = 0;
  researchQueueDroppedSampleCount = 0;
  lastBlockSequence = 0;
  lastAbsoluteSampleIndex = 0;
  lastBlockAcquiredAtMs = null;
  nextAnalysisSampleIndex = 0;
  processedFrameCount = 0;
  receivedBlockCount = 0;
  lostBlockCount = 0;
  maximumBlockGapMs = 0;
  currentBlockGapMs = 0;
  timestampRegressionCount = 0;
  priorBands = null;
  lastNucleusAtMs = Number.NEGATIVE_INFINITY;
  priorNucleusScore = 0;
  latencyValues.length = 0;
  recentBlockContinuity.length = 0;
}

function acquiredAtMs(block: VoiceWorkletPcmBlock): number {
  if (!clockMapping) throw new Error("invalid-voice-clock-mapping");
  return audioTimeInMainClock(block.endAudioFrameExclusive / block.sampleRateHz, clockMapping);
}

function mainNow(): number {
  if (!clockMapping) throw new Error("invalid-voice-clock-mapping");
  return performanceTimeInMainClock(performance.now(), performance.timeOrigin, clockMapping);
}

function reportClockFailure(): void {
  invalidClockMappingCount += 1;
  clockMapping = null;
  resetFeatureState();
  post({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "error", captureEpoch,
    reason: "invalid-voice-clock-duration" });
}

function analyzeAvailable(block: VoiceWorkletPcmBlock, workerReceivedAtMs: number): void {
  if (!ring || !clockMapping) return;
  const windowSamples = Math.round(sampleRateHz * 0.04);
  const hopSamples = Math.round(sampleRateHz * 0.01);
  const absoluteEnd =
    block.absoluteSampleIndex +
    new Float32Array(block.buffer).length;
  if (nextAnalysisSampleIndex === 0) {
    nextAnalysisSampleIndex =
      block.absoluteSampleIndex + windowSamples;
  }
  while (
    absoluteEnd >= nextAnalysisSampleIndex &&
    ring.availableSamples() >= windowSamples
  ) {
    const startedAt = performance.now();
    const workerStartedAtMs = mainNow();
    const trailingSampleCount = Math.max(
      0,
      absoluteEnd - nextAnalysisSampleIndex
    );
    const samples = ring.endingBeforeLatest(
      windowSamples,
      trailingSampleCount
    );
    if (!samples) break;
    const prepared = researchHook?.prepare(samples, sampleRateHz);
    const analysis = prepared?.baseline ?? analyzeVoiceWindow(samples, sampleRateHz, priorBands);
    const tMs =
      ((nextAnalysisSampleIndex - windowSamples) / sampleRateHz) *
      1000;
    const snrDb =
      analysis.rms <= 0
        ? 0
        : Math.max(
            0,
            20 *
              Math.log10(
                analysis.rms / Math.max(0.0001, noiseFloorRms)
              )
          );
    const continuityFloorMs = acquiredAtMs(block) - 2_000;
    while (
      recentBlockContinuity.length > 0 &&
      recentBlockContinuity[0].atMs < continuityFloorMs
    ) {
      recentBlockContinuity.shift();
    }
    const recentReceived = recentBlockContinuity.reduce(
      (sum, item) => sum + item.received,
      0
    );
    const recentLost = recentBlockContinuity.reduce(
      (sum, item) => sum + item.lost,
      0
    );
    const lostBlockFraction =
      recentLost / Math.max(1, recentReceived + recentLost);
    const blockGapMs = currentBlockGapMs;
    const frameAcquiredAtMs =
      acquiredAtMs(block) -
      (trailingSampleCount / sampleRateHz) * 1_000;
    const periodic =
      analysis.f0Hz !== null &&
      analysis.f0Confidence >= 0.55 &&
      analysis.estimatorAgreement >= 0.7;
    const speechActive =
      analysis.rms >= Math.max(0.003, noiseFloorRms * 2.8) &&
      (periodic ||
        (analysis.spectralFlux > 0.02 &&
          analysis.rms >= noiseFloorRms * 4));
    const nucleusScore =
      analysis.rms * (1 + Math.min(2, analysis.spectralFlux * 20));
    const syllabicNucleus =
      speechActive &&
      tMs - lastNucleusAtMs >= 100 &&
      nucleusScore > Math.max(0.012, priorNucleusScore * 1.15);
    if (syllabicNucleus) lastNucleusAtMs = tMs;
    priorNucleusScore =
      0.7 * priorNucleusScore + 0.3 * nucleusScore;
    priorBands = analysis.bandEnergies;
    processedFrameCount += 1;
    const frame: AmbientVoiceFrame = {
      schemaVersion: VOICE_SIGNAL_FRAME_VERSION,
      tMs,
      acquiredAtMs: frameAcquiredAtMs,
      captureEpoch,
      sequence: processedFrameCount,
      absoluteSampleIndex:
        nextAnalysisSampleIndex - windowSamples,
      taskContext,
      speechActive,
      periodic,
      trackSegmentId: `audio-${captureEpoch}-${segmentNumber}`,
      rms: analysis.rms,
      f0Hz: analysis.f0Hz,
      f0Confidence: analysis.f0Confidence,
      estimatorAgreement: analysis.estimatorAgreement,
      spectralFlux: analysis.spectralFlux,
      cepstralPeakProminenceDb: analysis.cepstralPeakProminenceDb,
      syllabicNucleus,
      clippedSampleFraction: analysis.clippedSampleFraction,
      dcOffset: analysis.dcOffset,
      snrDb,
      sampleRateHz: acquisitionSampleRateHz,
      blockGapMs,
      lostBlockFraction,
      browserProcessing: { ...browserProcessing },
      qualityReasons: [],
      processorRef: VOICE_DSP_PROCESSOR_REF
    };
    frame.qualityReasons =
      evaluateVoiceQuality(frame).reasonCodes;
    const workerServiceMs = performance.now() - startedAt;
    const workerReadyAtMs = mainNow();
    const processingLatencyMs = workerReadyAtMs - frameAcquiredAtMs;
    const blockEndAvailabilityMs = acquiredAtMs(block);
    let validTiming = true;
    try {
      validateVoiceDurations({ workerServiceMs,
        sampleAvailabilityToWorkerReadyMs: processingLatencyMs,
        blockEndToWorkerReceiptMs: workerReceivedAtMs - blockEndAvailabilityMs });
    } catch {
      validTiming = false;
      invalidClockMappingCount += 1;
    }
    const timing: VoiceFrameTiming = {
      schemaVersion: VOICE_TIMING_CONTRACT_VERSION,
      clockMappingId: clockMapping.id,
      coordinate: "main-performance-ms",
      status: validTiming ? "valid" : "unavailable",
      ...(validTiming ? {} : { unavailableReason: "inconsistent-clock-mapping" as const }),
      supportStartSample: nextAnalysisSampleIndex - windowSamples,
      supportEndSampleExclusive: nextAnalysisSampleIndex,
      referenceSample: nextAnalysisSampleIndex,
      lastRequiredInputSample: nextAnalysisSampleIndex - 1,
      referenceTimeMs: nextAnalysisSampleIndex / sampleRateHz * 1_000,
      sampleRateHz,
      sampleAvailabilityMs: frameAcquiredAtMs,
      blockEndAvailabilityMs, workerReceivedAtMs, workerStartedAtMs, workerReadyAtMs,
      workerServiceMs,
      sampleAvailabilityToWorkerReadyMs: validTiming ? processingLatencyMs : null,
      rawSampleAvailabilityToWorkerReadyMs: processingLatencyMs,
      blockEndToWorkerReceiptMs: validTiming ? workerReceivedAtMs - blockEndAvailabilityMs : null,
      rawBlockEndToWorkerReceiptMs: workerReceivedAtMs - blockEndAvailabilityMs,
      anchorReadUncertaintyMs: clockMapping.anchorReadUncertaintyMs,
      audioClockUncertaintyMs: null
    };
    if (validTiming) latencyValues.push(processingLatencyMs);
    if (latencyValues.length > 500) latencyValues.shift();
    const record: Extract<VoiceWorkerResponse, { type: "signal-frame" }> = {
      schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "signal-frame", captureEpoch, frame,
      processingLatencyMs: validTiming ? processingLatencyMs : null, timing
    };
    post(record);
    if (prepared) pendingResearchFrames?.push({ record, features: prepared, startedAt });
    if (processedFrameCount % 100 === 0) {
      post({
        schemaVersion: VOICE_WORKER_MESSAGE_VERSION,
        type: "diagnostics",
        captureEpoch,
        diagnostics: diagnostics()
      });
    }
    nextAnalysisSampleIndex += hopSamples;
  }
}

function handleBlock(event: MessageEvent<unknown>, sourcePort: MessagePort, researchRequest?: ResearchPcmRequest): void | Promise<void> {
  const blockServiceStartedAt = performance.now();
  const block = event.data as VoiceWorkletPcmBlock;
  // Recycle known transport buffers even when the epoch/mapping is stale.
  // Otherwise a cross-port reset race would permanently consume a credit.
  const recycle = (): void => {
    if (researchRequest) { recycleResearchRequest(researchRequest); return; }
    if (block?.buffer instanceof ArrayBuffer && Number.isSafeInteger(block.transportId)) {
      sourcePort.postMessage({ type: "recycle", transportId: block.transportId, buffer: block.buffer }, [block.buffer]);
    }
  };
  if (
    workerDisposed || sourcePort !== port || block?.schemaVersion !== VOICE_WORKLET_MESSAGE_VERSION || block.type !== "pcm-block" ||
    block.captureEpoch !== captureEpoch || block.channelCount !== 1 ||
    !(block.buffer instanceof ArrayBuffer) || block.buffer.byteLength % 4 !== 0 ||
    !Number.isSafeInteger(block.absoluteSampleIndex) || block.absoluteSampleIndex < 0 ||
    !Number.isSafeInteger(block.sequence) || block.sequence < 1 ||
    !Number.isSafeInteger(block.endAudioFrameExclusive) || block.endAudioFrameExclusive < 0 ||
    !Number.isFinite(block.sampleRateHz) || block.sampleRateHz <= 0 ||
    !clockMapping || block.clockMappingId !== clockMapping.id
  ) {
    recycle();
    return;
  }
  if (block.buffer.byteLength / 4 > Math.ceil(block.sampleRateHz * VOICE_MAXIMUM_PCM_BLOCK_MS / 1_000)) {
    oversizedPcmBlockCount += 1;
    resetFeatureState();
    recycle();
    return; // Reject oversize support explicitly; never truncate or silently analyze a prefix.
  }
  const workerReceivedAtMs = researchRequest
    ? performanceTimeInMainClock(researchRequest.receivedAtLocalMs, performance.timeOrigin, clockMapping) : mainNow();
  const admittedAtMs = researchRequest ? mainNow() : workerReceivedAtMs;
  const samples = new Float32Array(block.buffer);
  const atMs = acquiredAtMs(block);
  const blockAgeMs = admittedAtMs - atMs;
  // A structurally valid mapping with inconsistent/negative duration cannot
  // establish age. Retain unavailable timing; never clamp it into a bound.
  const stale = Number.isFinite(blockAgeMs) && blockAgeMs > VOICE_MAXIMUM_BLOCK_AGE_MS;
  const endSampleExclusive = block.absoluteSampleIndex + samples.length;
  let resultBatchId = 0;
  const finishBlock = (analyzedFrameCount: number, disposition: "processed" | "stale-dropped",
    records: VoiceWorkerBatchRecord[] = pendingResultRecords ?? []): void => {
    recycle();
    if (records === pendingResultRecords) pendingResultRecords = null;
    // Post the bounded data envelope first so complete-block service includes its
    // structured clone/post cost. The completion envelope is the only excluded post.
    worker.postMessage({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "result-batch", captureEpoch,
      batchId: resultBatchId, clockMappingId: block.clockMappingId, records } satisfies VoiceWorkerResponse);
    const workerCompletedAtMs = mainNow();
    const workerServiceMs = performance.now() - blockServiceStartedAt;
    post({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "result-batch-complete", captureEpoch,
      batchId: resultBatchId, clockMappingId: block.clockMappingId,
      timing: { schemaVersion: VOICE_TIMING_CONTRACT_VERSION,
        captureEpoch, trackSegmentId: `audio-${captureEpoch}-${segmentNumber}`,
        blockSequence: block.sequence, supportStartSample: block.absoluteSampleIndex,
        supportEndSampleExclusive: endSampleExclusive, analyzedFrameCount, disposition,
        ...(disposition === "stale-dropped" ? { reason: "audio-timeline-age-exceeded" as const } : {}),
        workerReceivedAtMs, workerCompletedAtMs, workerServiceMs } });
  };
  droppedTransportBlockCount = Math.max(droppedTransportBlockCount, block.droppedBlockCount);
  droppedTransportSampleCount = Math.max(droppedTransportSampleCount, block.droppedSampleCount);
  maximumInFlightBlocks = Math.max(maximumInFlightBlocks, block.inFlightBlockCount);
  if (
    (lastBlockSequence > 0 && block.sequence <= lastBlockSequence) ||
    (lastAbsoluteSampleIndex > 0 &&
      block.absoluteSampleIndex < lastAbsoluteSampleIndex)
  ) {
    timestampRegressionCount += 1;
    resetFeatureState();
    recycle();
    return;
  }
  if (
    lastBlockAcquiredAtMs !== null &&
    atMs < lastBlockAcquiredAtMs
  ) {
    timestampRegressionCount += 1;
    resetFeatureState();
  }
  const lostSincePrior =
    lastBlockSequence > 0
      ? Math.max(0, block.sequence - lastBlockSequence - 1)
      : block.sequence - 1;
  const noResultCredit = inFlightResults.size >= VOICE_MAXIMUM_RESULT_BATCHES;
  const dropped = stale || noResultCredit;
  lostBlockCount += lostSincePrior + (dropped ? 1 : 0);
  const expectedBlockMs =
    (samples.length / block.sampleRateHz) * 1000;
  const gapMs =
    lastBlockAcquiredAtMs === null
      ? 0
      : Math.max(
          0,
          atMs - lastBlockAcquiredAtMs - expectedBlockMs
        );
  maximumBlockGapMs = Math.max(maximumBlockGapMs, gapMs);
  currentBlockGapMs = gapMs;
  const sampleDiscontinuity =
    lastAbsoluteSampleIndex > 0 &&
    block.absoluteSampleIndex !== lastAbsoluteSampleIndex;
  const rateChanged = sampleRateHz !== block.sampleRateHz;
  if (rateChanged) {
    sampleRateHz = block.sampleRateHz;
    ring?.clear();
    ring = new BoundedPcmRingBuffer(Math.round(sampleRateHz * 2));
  }
  if (lostSincePrior > 0 || sampleDiscontinuity || gapMs > 40 || rateChanged) resetFeatureState();
  recentBlockContinuity.push({
    atMs,
    received: dropped ? 0 : 1,
    lost: lostSincePrior + (dropped ? 1 : 0)
  });
  while (recentBlockContinuity.length > 512 ||
      (recentBlockContinuity.length > 0 && recentBlockContinuity[0].atMs < atMs - 2_000)) {
    recentBlockContinuity.shift();
  }
  lastBlockAcquiredAtMs = atMs;
  lastBlockSequence = block.sequence;
  lastAbsoluteSampleIndex =
    block.absoluteSampleIndex + samples.length;
  if (noResultCredit) {
    if (!resultBackpressured) {
      resultBackpressureEpisodeCount += 1;
      lastResultBackpressureSpan = { supportStartSample: block.absoluteSampleIndex,
        supportEndSampleExclusive: endSampleExclusive, sampleRateHz: block.sampleRateHz,
        droppedBlockCount: 0, droppedSampleCount: 0 };
    }
    resultBackpressured = true;
    resultBackpressureBlockCount += 1;
    resultBackpressureSampleCount += samples.length;
    if (lastResultBackpressureSpan) {
      lastResultBackpressureSpan.supportEndSampleExclusive = endSampleExclusive;
      lastResultBackpressureSpan.droppedBlockCount += 1;
      lastResultBackpressureSpan.droppedSampleCount += samples.length;
    }
    resetFeatureState();
    recycle();
    return; // No scalar message or retained pending record while result credits are exhausted.
  }
  const resumedAfterResultBackpressure = resultBackpressured;
  resultBackpressured = false;
  resultBatchId = ++nextResultBatchId;
  inFlightResults.set(resultBatchId, { epoch: captureEpoch, mappingId: block.clockMappingId, segment: segmentNumber });
  maximumInFlightResultBatches = Math.max(maximumInFlightResultBatches, inFlightResults.size);
  pendingResultRecords = [];
  pendingResearchFrames = researchHook ? [] : null;
  if (stale) {
    staleTransportBlockCount += 1;
    staleTransportSampleCount += samples.length;
    resetFeatureState();
    finishBlock(0, "stale-dropped");
    return;
  }
  receivedBlockCount += 1;
  sampleRateHz = block.sampleRateHz;
  const priorFrameCount = processedFrameCount;
  ring?.push(samples);
  try { analyzeAvailable(block, workerReceivedAtMs); }
  catch (error) {
    pendingResultRecords = null; pendingResearchFrames = null;
    inFlightResults.delete(resultBatchId); recycle(); resetFeatureState(); throw error;
  }
  if (resumedAfterResultBackpressure && !pendingResultRecords.some(record => record.type === "diagnostics")) {
    post({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "diagnostics", captureEpoch, diagnostics: diagnostics() });
  }
  const analyzedFrameCount = processedFrameCount - priorFrameCount;
  const preparedFrames = pendingResearchFrames;
  pendingResearchFrames = null;
  if (!researchHook || !researchRequest || !preparedFrames?.length) {
    finishBlock(analyzedFrameCount, "processed");
    return;
  }
  const hook = researchHook;
  const generation = featureGeneration;
  const records = pendingResultRecords;
  pendingResultRecords = null;
  let cancelled = false;
  const cancel = (): void => {
    if (cancelled) return;
    cancelled = true;
    inFlightResults.delete(resultBatchId);
    if (block.captureEpoch === captureEpoch) {
      researchCancelledBlockCount += 1;
      researchCancelledFrameCount += analyzedFrameCount;
      receivedBlockCount -= 1;
      lostBlockCount += 1;
      const continuity = recentBlockContinuity.at(-1);
      if (continuity?.atMs === atMs) { continuity.received = 0; continuity.lost += 1; }
    }
    recycle();
  };
  activeResearchCancel = cancel;
  const stillCurrent = () => !cancelled && !workerDisposed && generation === featureGeneration &&
    block.captureEpoch === captureEpoch && block.clockMappingId === clockMapping?.id;
  return (async () => {
    try {
      for (const prepared of preparedFrames) {
        if (!stillCurrent()) { cancel(); return; }
        const prediction = await hook.predict(prepared.features, prepared.record.timing);
        if (!stillCurrent()) { cancel(); return; }
        const modelReadyAtMs = mainNow();
        const modelLatency = modelReadyAtMs - prepared.record.timing.sampleAvailabilityMs;
        const modelTimingValid = prepared.record.timing.status === "valid" && Number.isFinite(modelLatency) && modelLatency >= 0;
        prepared.record.research = { ...prediction, modelReadyAtMs,
          rawSampleAvailabilityToModelReadyMs: modelLatency,
          sampleAvailabilityToModelReadyMs: modelTimingValid ? modelLatency : null,
          targetReferenceToModelReadyMs: modelTimingValid ? modelLatency + 20 : null,
          elapsedFromFrameStartMs: performance.now() - prepared.startedAt };
      }
      if (!stillCurrent()) { cancel(); return; }
      finishBlock(analyzedFrameCount, "processed", records);
    } catch (error) {
      const current = stillCurrent();
      cancel();
      if (!current) return; // A retired inference cannot fail the renewed epoch or mapping.
      resetFeatureState();
      throw error;
    } finally {
      if (activeResearchCancel === cancel) activeResearchCancel = null;
    }
  })();
}

worker.addEventListener("message", (event: MessageEvent<VoiceWorkerRequest>) => {
  const message = event.data;
  if (message?.schemaVersion !== VOICE_WORKER_MESSAGE_VERSION) return;
  if (message.type === "result-ack") {
    const origin = inFlightResults.get(message.batchId);
    if (!origin || origin.epoch !== message.captureEpoch ||
        !Number.isSafeInteger(message.discardedFrameCount) || message.discardedFrameCount < 0 || message.discardedFrameCount > 4 ||
        !Number.isSafeInteger(message.staleFrameCount) || message.staleFrameCount < 0 ||
        message.staleFrameCount > message.discardedFrameCount) return;
    inFlightResults.delete(message.batchId);
    if (message.captureEpoch === captureEpoch && message.discardedFrameCount > 0) {
      hostDiscardedFrameCount += message.discardedFrameCount;
      hostStaleFrameCount += message.staleFrameCount;
      if (origin.mappingId === clockMapping?.id && origin.segment === segmentNumber) resetFeatureState();
    }
    return;
  }
  if ((message.type === "initialize" || message.type === "reset") && message.noiseFloorRms !== undefined
    && (!Number.isFinite(message.noiseFloorRms) || message.noiseFloorRms < 0.0001 || message.noiseFloorRms > 1)) return;
  if (message.type === "initialize") {
    port?.close();
    port = message.port;
    if ((message.clockMapping !== null && !validVoiceClockMapping(message.clockMapping)) ||
        !Number.isFinite(message.audioContextSampleRateHz) || message.audioContextSampleRateHz <= 0) {
      post({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "error", captureEpoch: message.captureEpoch,
        reason: "invalid-voice-clock-mapping" });
      return;
    }
    clockMapping = message.clockMapping;
    sampleRateHz = message.audioContextSampleRateHz;
    acquisitionSampleRateHz = message.captureSettings.actual.sampleRate;
    browserProcessing = {
      ...message.captureSettings.actual.browserProcessing
    };
    taskContext = message.taskContext;
    ring = new BoundedPcmRingBuffer(
      Math.round(sampleRateHz * 2)
    );
    reset(message.captureEpoch, message.noiseFloorRms);
    const initializedPort = port;
    port.addEventListener("message", event => receivePcm(event, initializedPort));
    port.start();
    post({
      schemaVersion: VOICE_WORKER_MESSAGE_VERSION,
      type: "ready",
      captureEpoch,
      provenance: { ...audioPipelineProvenance(), acquisitionSampleRateHz, renderedSampleRateHz: sampleRateHz }
    });
    return;
  }
  if (message.type === "clock-mapping" && message.captureEpoch === captureEpoch) {
    cancelResearchWork();
    resetFeatureState();
    clockMapping = validVoiceClockMapping(message.clockMapping) ? message.clockMapping : null;
    lastBlockAcquiredAtMs = null;
    return;
  }
  if (message.type === "set-task" && message.captureEpoch === captureEpoch) {
    taskContext = message.taskContext;
    return;
  }
  if (
    message.type === "set-noise-floor" &&
    message.captureEpoch === captureEpoch &&
    Number.isFinite(message.noiseFloorRms)
  ) {
    noiseFloorRms = Math.max(0.0001, message.noiseFloorRms);
    return;
  }
  if (message.type === "reset") {
    if (message.clockMapping !== null && !validVoiceClockMapping(message.clockMapping)) { reportClockFailure(); return; }
    clockMapping = message.clockMapping;
    taskContext = message.taskContext;
    reset(message.captureEpoch, message.noiseFloorRms);
    return;
  }
  if (message.type === "dispose" && message.captureEpoch === captureEpoch) {
    workerDisposed = true;
    cancelResearchWork();
    researchPump.dispose();
    void researchHook?.dispose().catch(() => {});
    const finalDiagnostics = diagnostics();
    port?.close();
    port = null;
    resetFeatureState();
    clockMapping = null;
    ring = null;
    post({
      schemaVersion: VOICE_WORKER_MESSAGE_VERSION,
      type: "disposed",
      captureEpoch,
      diagnostics: finalDiagnostics
    });
    worker.close();
  }
});
