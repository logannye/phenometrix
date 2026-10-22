import type { BrowserAudioProcessingState } from "./measurement.js";

export interface VisualPipelineProvenance {
  processorRef: string;
  runtime: "mediapipe-tasks-vision";
  mediaPipeVersion: string;
  modelAsset: string;
  modelSha256: string;
  delegate: "GPU" | "CPU";
  geometryVersion: string;
}

export interface VideoCaptureSettings {
  requested: {
    width: number;
    height: number;
    frameRate: number;
  };
  actual: {
    width: number;
    height: number;
    frameRate: number | null;
  };
  facingMode?: string;
  coordinateSpace: "normalized-unmirrored-image";
  displayMirrored: true;
  lateralityConvention: "subject-anatomical";
}

export interface AudioCaptureSettings {
  requested: {
    channelCount: 1;
    sampleRate: 48000;
    echoCancellation: false;
    noiseSuppression: false;
    autoGainControl: false;
  };
  actual: {
    channelCount: number;
    sampleRate: number;
    browserProcessing: BrowserAudioProcessingState;
  };
}

/**
 * The window/hop/ring literals describe the pitch and VAD analysis path and are
 * pinned so a mid-stream change cannot go unnoticed. A future analysis path with
 * different framing (formants, cepstrum) declares its own provenance rather than
 * widening these.
 */
export interface AudioPipelineProvenance {
  processorRef: string;
  runtime: "audio-worklet-voice-worker";
  workletSchemaVersion: "phenometric.voice-worklet-message.v1" | "phenometric.voice-worklet-message.v2";
  workerSchemaVersion: "phenometric.voice-worker-message.v1" | "phenometric.voice-worker-message.v2" | "phenometric.voice-worker-message.v3";
  signalFrameSchemaVersion: "phenometric.voice-signal-frame.v1";
  analysisWindowMs: 40;
  analysisHopMs: 10;
  ringBufferSeconds: 2;
  algorithmVersion: string;
  /** Versioned timing semantics are independent of pitch arithmetic. */
  timingContractVersion?: string;
  renderFreshnessContractVersion?: string;
  renderFreshnessPolicy?: "required-render-lease-v1" | "unqualified-legacy-fixture";
  renderFreshnessMaximumAgeMs?: number;
  renderLeaseRenewalMs?: number;
  renderLeaseMaximumPending?: number;
  renderLeaseLedgerCapacity?: number;
  /** Actual input track rate; this determines acquisition quality eligibility. */
  acquisitionSampleRateHz?: number;
  /** AudioContext PCM rate after any browser resampling, before DSP decimation. */
  renderedSampleRateHz?: number;
  transportMaxInFlightBlocks?: number;
  /** Provisional bound on admitted mapped audio-timeline age, not hardware latency. */
  transportMaxBlockAgeMs?: number;
  transportDropPolicy?: string;
  resultMaxInFlightBatches?: number;
  /** Includes the separately posted final block-timing record. */
  resultMaxRecordsPerBatch?: number;
  /** Main-thread support-end age bound only when the clock mapping is consistent. */
  resultMaxAgeMs?: number;
  resultDropPolicy?: string;
  /** Rolling latency window; never a whole-session tail statistic. */
  latencyStatisticWindowFrames?: number;
}

export interface AudioStreamDiagnostics {
  receivedBlockCount: number;
  processedFrameCount: number;
  lostBlockCount: number;
  lostBlockFraction: number;
  maximumBlockGapMs: number;
  p95FeatureLatencyMs: number;
  timestampRegressionCount: number;
  ringBufferCapacitySamples: number;
  invalidClockMappingCount?: number;
  droppedTransportBlockCount?: number;
  droppedTransportSampleCount?: number;
  maximumInFlightBlocks?: number;
  staleTransportBlockCount?: number;
  staleTransportSampleCount?: number;
  freshnessWithheldBlockCount?: number;
  freshnessWithheldSampleCount?: number;
  freshnessWithheldReasons?: Record<string, number>;
  freshnessPolicy?: "required-render-lease-v1" | "unqualified-legacy-fixture";
  inFlightResultBatchCount?: number;
  maximumInFlightResultBatches?: number;
  resultBackpressureBlockCount?: number;
  resultBackpressureSampleCount?: number;
  resultBackpressureEpisodeCount?: number;
  /** Span of the latest backpressure episode; counts exclude any other gaps inside its span. */
  lastResultBackpressureSpan?: { supportStartSample: number; supportEndSampleExclusive: number;
    sampleRateHz: number; droppedBlockCount: number; droppedSampleCount: number };
  hostDiscardedFrameCount?: number;
  hostStaleFrameCount?: number;
  oversizedPcmBlockCount?: number;
  researchCancelledBlockCount?: number;
  researchCancelledFrameCount?: number;
  researchQueueDroppedBlockCount?: number;
  researchQueueDroppedSampleCount?: number;
  researchMaximumQueuedBlocks?: number;
  /** p95FeatureLatencyMs is undefined as a measurement when this is zero. */
  latencyStatisticSampleCount?: number;
}
