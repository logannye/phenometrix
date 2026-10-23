import { VOICE_RENDER_FRESHNESS_VERSION, type VoiceRenderFreshnessPolicy, type VoiceRenderLease } from "./voice-render-freshness.js";
import type { VoiceResearchPrediction } from "./voice-research-hook.js";
import type {
  AudioCaptureSettings,
  AudioPipelineProvenance,
  AudioStreamDiagnostics,
  BrowserAudioProcessingState,
  VoiceTaskContext
} from "@phenometrix/contracts";
import type { VoiceClockMapping, VoiceFrameTiming, VoiceBlockTiming } from "./voice-stream-timing.js";
import { VOICE_TIMING_CONTRACT_VERSION } from "./voice-stream-timing.js";
import type { VoiceSignalFrameV1 } from "@phenometrix/ambient-core";

export const VOICE_MAXIMUM_BLOCK_AGE_MS = 100;
export const VOICE_MAXIMUM_RESULT_AGE_MS = 100;
export const VOICE_MAXIMUM_RESULT_BATCHES = 4;
export const VOICE_MAXIMUM_RESULT_RECORDS = 5; // At most four frames and one diagnostic; timing arrives in the completion envelope.
export const VOICE_MAXIMUM_PCM_BLOCK_MS = 40;

export const VOICE_WORKER_MESSAGE_VERSION =
  "phenometric.voice-worker-message.v3" as const;
export const VOICE_SIGNAL_FRAME_VERSION =
  "phenometric.voice-signal-frame.v1" as const;
export const VOICE_WORKLET_MESSAGE_VERSION =
  "phenometric.voice-worklet-message.v2" as const;
/*
 * 1.1: the pitch path decimates through an anti-aliased filter instead of a
 * boxcar average, band energies come from one windowed FFT instead of a
 * rectangular-window direct evaluation, and cepstral peak prominence is
 * measured. Pitch values shift slightly as a result, so sessions analysed
 * under 1.0 and 1.1 are not interchangeable.
 *
 * The frame schema version stays v1: the new fields are optional additions, so
 * the shape is still compatible. What changed is the values, and that is what
 * the algorithm and processor versions exist to say. They are separate for
 * exactly this reason.
 */
export const VOICE_DSP_ALGORITHM_VERSION =
  "voice-analysis-1.1" as const;
export const VOICE_DSP_PROCESSOR_REF =
  "browser-voice-dsp@1.1" as const;
export type VoiceWorkerRequest =
  | {
      schemaVersion: typeof VOICE_WORKER_MESSAGE_VERSION;
      type: "result-ack";
      captureEpoch: number;
      batchId: number;
      discardedFrameCount: number;
      staleFrameCount: number;
    }
  | {
      schemaVersion: typeof VOICE_WORKER_MESSAGE_VERSION;
      type: "initialize";
      freshnessPolicy: VoiceRenderFreshnessPolicy;
      captureEpoch: number;
      port: MessagePort;
      /** Null while the audio context is not running. */
      clockMapping: VoiceClockMapping | null;
      audioContextSampleRateHz: number;
      captureSettings: AudioCaptureSettings;
      taskContext: VoiceTaskContext;
      /** Applied atomically before this epoch can process PCM. */
      noiseFloorRms?: number;
    }
  | {
      schemaVersion: typeof VOICE_WORKER_MESSAGE_VERSION;
      type: "clock-mapping";
      captureEpoch: number;
      clockMapping: VoiceClockMapping | null;
    }
  | {
      schemaVersion: typeof VOICE_WORKER_MESSAGE_VERSION;
      type: "set-task";
      captureEpoch: number;
      taskContext: VoiceTaskContext;
    }
  | {
      schemaVersion: typeof VOICE_WORKER_MESSAGE_VERSION;
      type: "set-noise-floor";
      captureEpoch: number;
      noiseFloorRms: number;
    }
  | {
      schemaVersion: typeof VOICE_WORKER_MESSAGE_VERSION;
      type: "reset";
      captureEpoch: number;
      /** Null keeps a reset during suspension invalid until a running-context anchor arrives. */
      clockMapping: VoiceClockMapping | null;
      taskContext: VoiceTaskContext;
      /** Applied in the same worker turn as the epoch change. */
      noiseFloorRms?: number;
    }
  | {
      schemaVersion: typeof VOICE_WORKER_MESSAGE_VERSION;
      type: "dispose";
      captureEpoch: number;
    }
  ;

export type VoiceWorkerResponse =
  | {
      schemaVersion: typeof VOICE_WORKER_MESSAGE_VERSION;
      type: "result-batch";
      captureEpoch: number;
      batchId: number;
      clockMappingId: number;
      records: VoiceWorkerBatchRecord[];
    }
  | {
      schemaVersion: typeof VOICE_WORKER_MESSAGE_VERSION;
      type: "result-batch-complete";
      captureEpoch: number;
      batchId: number;
      clockMappingId: number;
      timing: VoiceBlockTiming;
    }
  | {
      schemaVersion: typeof VOICE_WORKER_MESSAGE_VERSION;
      type: "ready";
      captureEpoch: number;
      provenance: AudioPipelineProvenance;
    }
  | {
      schemaVersion: typeof VOICE_WORKER_MESSAGE_VERSION;
      type: "signal-frame";
      captureEpoch: number;
      frame: VoiceSignalFrameV1;
      /** Audio timeline support-end to worker-ready, excluding hardware acquisition. */
      processingLatencyMs: number | null;
      timing: VoiceFrameTiming;
      research?: VoiceResearchPrediction;
      renderLease?: VoiceRenderLease | null;
    }
  | {
      schemaVersion: typeof VOICE_WORKER_MESSAGE_VERSION;
      type: "block-timing";
      captureEpoch: number;
      timing: VoiceBlockTiming;
    }
  | {
      schemaVersion: typeof VOICE_WORKER_MESSAGE_VERSION;
      type: "diagnostics";
      captureEpoch: number;
      diagnostics: AudioStreamDiagnostics;
    }
  | {
      schemaVersion: typeof VOICE_WORKER_MESSAGE_VERSION;
      type: "disposed";
      captureEpoch: number;
      diagnostics: AudioStreamDiagnostics;
    }
  | {
      schemaVersion: typeof VOICE_WORKER_MESSAGE_VERSION;
      type: "error";
      captureEpoch: number;
      reason: string;
    }
  ;

export type VoiceWorkerBatchRecord = Extract<VoiceWorkerResponse, { type: "signal-frame" | "diagnostics" }>;

export interface VoiceWorkletPcmBlock {
  schemaVersion: typeof VOICE_WORKLET_MESSAGE_VERSION;
  type: "pcm-block";
  captureEpoch: number;
  sequence: number;
  absoluteSampleIndex: number;
  /** Legacy render-quantum timestamp; never used as the block endpoint. */
  acquisitionAudioTimeSeconds: number;
  endAudioFrameExclusive: number;
  clockMappingId: number;
  transportId: number;
  droppedBlockCount: number;
  droppedSampleCount: number;
  inFlightBlockCount: number;
  sampleRateHz: number;
  channelCount: 1;
  buffer: ArrayBuffer;
  /** Lease present before this block's first observed input; never refreshed at block end. */
  renderLease?: VoiceRenderLease | null;
}

export interface VoiceWorkletTransportDiagnostics {
  schemaVersion: typeof VOICE_WORKLET_MESSAGE_VERSION;
  type: "transport-diagnostics";
  captureEpoch: number;
  final: true;
  sequence: number;
  sampleClockEndExclusive: number;
  droppedBlockCount: number;
  droppedSampleCount: number;
  discardedPartialSampleCount: number;
  inFlightBlockCountAtDispose: number;
  maximumInFlightBlocks: 4;
}

export type VoiceTransportStopResult =
  | { status: "complete"; diagnostics: VoiceWorkletTransportDiagnostics }
  | { status: "unavailable"; captureEpoch: number; reason: "worklet-stop-timeout" | "worklet-dispose-failed" };

export function isCurrentVoiceTransportDiagnostics(value: unknown, captureEpoch: number): value is VoiceWorkletTransportDiagnostics {
  if (!value || typeof value !== "object") return false;
  const item = value as VoiceWorkletTransportDiagnostics;
  return item.schemaVersion === VOICE_WORKLET_MESSAGE_VERSION && item.type === "transport-diagnostics" &&
    item.captureEpoch === captureEpoch && item.final === true && item.maximumInFlightBlocks === 4 &&
    [item.sequence, item.sampleClockEndExclusive, item.droppedBlockCount, item.droppedSampleCount,
      item.discardedPartialSampleCount, item.inFlightBlockCountAtDispose].every(number =>
        Number.isSafeInteger(number) && number >= 0) && item.inFlightBlockCountAtDispose <= 4;
}

export function isCurrentVoiceWorkerResponse(
  value: unknown,
  captureEpoch: number
): value is VoiceWorkerResponse {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as {
    schemaVersion?: unknown;
    captureEpoch?: unknown;
  };
  return (
    candidate.schemaVersion === VOICE_WORKER_MESSAGE_VERSION &&
    candidate.captureEpoch === captureEpoch
  );
}

export function audioPipelineProvenance(freshnessPolicy: VoiceRenderFreshnessPolicy = "required-render-lease-v1"): AudioPipelineProvenance {
  return {
    processorRef: VOICE_DSP_PROCESSOR_REF,
    runtime: "audio-worklet-voice-worker",
    workletSchemaVersion: VOICE_WORKLET_MESSAGE_VERSION,
    workerSchemaVersion: VOICE_WORKER_MESSAGE_VERSION,
    signalFrameSchemaVersion: VOICE_SIGNAL_FRAME_VERSION,
    analysisWindowMs: 40,
    analysisHopMs: 10,
    ringBufferSeconds: 2,
    algorithmVersion: VOICE_DSP_ALGORITHM_VERSION,
    timingContractVersion: VOICE_TIMING_CONTRACT_VERSION,
    transportMaxInFlightBlocks: 4,
    ...(freshnessPolicy === "unqualified-legacy-fixture" ? { transportMaxBlockAgeMs: VOICE_MAXIMUM_BLOCK_AGE_MS } : {}),
    renderFreshnessContractVersion: VOICE_RENDER_FRESHNESS_VERSION,
    renderFreshnessPolicy: freshnessPolicy,
    renderFreshnessMaximumAgeMs: 100,
    renderLeaseRenewalMs: 25,
    renderLeaseMaximumPending: 1,
    renderLeaseLedgerCapacity: 8,
    resultMaxInFlightBatches: VOICE_MAXIMUM_RESULT_BATCHES,
    resultMaxRecordsPerBatch: VOICE_MAXIMUM_RESULT_RECORDS + 1,
    ...(freshnessPolicy === "unqualified-legacy-fixture" ? { resultMaxAgeMs: VOICE_MAXIMUM_RESULT_AGE_MS } : {}),
    resultDropPolicy: freshnessPolicy === "required-render-lease-v1"
      ? "drop-input-at-result-cap;require-main-render-lease;reset-on-result-loss"
      : "unqualified-legacy-fixed-projection-age-gate;drop-input-at-result-cap;reset-on-result-loss",
    transportDropPolicy: freshnessPolicy === "required-render-lease-v1"
      ? "drop-new-at-cap;require-render-lease-at-worker;reset-on-gap"
      : "unqualified-legacy-fixed-projection-age-gate;drop-new-at-cap;reset-on-gap",
    latencyStatisticWindowFrames: 500
  };
}

export function requestedAudioCaptureSettings(
  sampleRate: number,
  channelCount: number,
  browserProcessing: BrowserAudioProcessingState
): AudioCaptureSettings {
  return {
    requested: {
      channelCount: 1,
      sampleRate: 48_000,
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false
    },
    actual: {
      channelCount,
      sampleRate,
      browserProcessing: { ...browserProcessing }
    }
  };
}
