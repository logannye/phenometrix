import { extractVoicePitchFeaturesV2, type VoicePitchFeaturesV2 } from "./voice-pitch-features-v2.js";
import { BoundedPcmRingBuffer } from "./voice-dsp.js";
import {
  extractVoicePitchFeaturesV1,
  type VoicePitchFeatureHistoryV1,
  type VoicePitchFeaturesV1
} from "./voice-pitch-features.js";

export const VOICE_REPLAY_VERSION = "phenometrix.causal-voice-replay.v1";
export interface VoiceReplayOptions {
  sampleRateHz: number;
  featureVersion?: "v1" | "v2";
  /** Default audited nominal provider grid: 16 + 10*k ms. */
  referenceOriginMs?: number;
  streamId?: string;
  epoch?: number;
}
export interface VoiceReplayFrameV1 {
  schemaVersion: typeof VOICE_REPLAY_VERSION;
  streamId: string;
  epoch: number;
  segmentId: string;
  referenceIndex: number;
  sourceSampleRateHz: number;
  supportStartSample: number;
  supportEndSampleExclusive: number;
  referenceSample: number;
  referenceTimeMs: number;
  nominalReferenceTimeMs: number;
  referenceQuantizationErrorMs: number;
  lastRequiredInputSample: number;
  lookaheadSamples: 0;
  /** Historical tMs semantics are retained explicitly, not relabeled. */
  legacyWindowStartMs: number;
  baselineProcessorId: "browser-voice-dsp@1.1";
  baselinePitch: { state: "available" | "unavailable"; f0Hz: number | null;
    reason: "legacy-pitch-unresolved" | null };
  features: VoicePitchFeaturesV1 | VoicePitchFeaturesV2;
}

/**
 * Sequential acquisition-only replay; no reference trajectories enter this class.
 * Storage is one 40 ms window plus bounded features. Arbitrary input chunk sizes
 * do not alter window phase. This is offline replay, not browser timing evidence.
 */
export class CausalVoiceReplay {
  readonly sampleRateHz: number;
  readonly windowSamples: number;
  readonly hopSamples: number;
  readonly referenceOriginSamples: number;
  private readonly featureVersion: "v1" | "v2";
  private readonly referenceOriginMs: number;
  private readonly streamId: string;
  private readonly epoch: number;
  private readonly ring: BoundedPcmRingBuffer;
  private cursor = 0;
  private nextReferenceIndex = 0;
  private segmentOrdinal = 0;
  private history: VoicePitchFeatureHistoryV1 | null = null;

  constructor(options: VoiceReplayOptions) {
    this.sampleRateHz = options.sampleRateHz;
    this.featureVersion = options.featureVersion ?? "v1";
    this.windowSamples = this.sampleRateHz * 0.04;
    this.hopSamples = this.sampleRateHz * 0.01;
    const originMs = options.referenceOriginMs ?? 16;
    this.referenceOriginMs = originMs;
    this.referenceOriginSamples = Math.round(this.sampleRateHz * originMs / 1000);
    if (!Number.isInteger(this.sampleRateHz) || this.sampleRateHz < 8000 ||
        !Number.isInteger(this.windowSamples) || !Number.isInteger(this.hopSamples) ||
        !Number.isFinite(originMs) || originMs < 0 ||
        !Number.isSafeInteger(this.referenceOriginSamples)) {
      throw new Error("Replay requires an integer rate with exact 40/10 ms sample counts and a finite phase.");
    }
    this.streamId = options.streamId ?? "public-research-replay";
    this.epoch = options.epoch ?? 0;
    if (!Number.isSafeInteger(this.epoch) || this.epoch < 0) throw new Error("Invalid replay epoch.");
    this.ring = new BoundedPcmRingBuffer(this.windowSamples);
    this.scheduleFirstCompleteWindow();
  }

  private scheduleFirstCompleteWindow(): void {
    this.nextReferenceIndex = Math.max(0, Math.ceil(
      (this.cursor + this.windowSamples - this.referenceOriginSamples) / this.hopSamples
    ));
  }

  /** Discontinuities discard all signal and feature history and create a segment. */
  resetAt(startSample: number): void {
    if (!Number.isSafeInteger(startSample) || startSample < this.cursor) {
      throw new Error("Replay reset sample must not move backwards.");
    }
    this.cursor = startSample;
    this.segmentOrdinal += 1;
    this.ring.clear();
    this.history = null;
    this.scheduleFirstCompleteWindow();
  }

  get retainedSampleCount(): number { return this.ring.availableSamples(); }
  get nextInputSample(): number { return this.cursor; }

  push(samples: Float32Array, startSample = this.cursor): VoiceReplayFrameV1[] {
    if (!Number.isSafeInteger(startSample) || startSample < this.cursor ||
        !Number.isSafeInteger(startSample + samples.length)) throw new Error("Invalid replay sample interval.");
    // Validate the entire block before changing state, so a rejected block is atomic.
    for (const value of samples) if (!Number.isFinite(value)) throw new Error("PCM must be finite.");
    if (startSample !== this.cursor) this.resetAt(startSample);
    const output: VoiceReplayFrameV1[] = [];
    let offset = 0;
    while (offset < samples.length) {
      const endpoint = this.referenceOriginSamples + this.nextReferenceIndex * this.hopSamples;
      const count = Math.min(samples.length - offset, endpoint - this.cursor);
      this.ring.push(samples.subarray(offset, offset + count));
      this.cursor += count;
      offset += count;
      if (this.cursor !== endpoint) continue;
      const window = this.ring.latest(this.windowSamples);
      if (window === null) throw new Error("Replay window scheduling invariant failed.");
      const features = this.featureVersion === "v2"
        ? extractVoicePitchFeaturesV2(window, this.sampleRateHz, this.history)
        : extractVoicePitchFeaturesV1(window, this.sampleRateHz, this.history);
      this.history = features.nextHistory;
      const start = endpoint - this.windowSamples;
      output.push({ schemaVersion: VOICE_REPLAY_VERSION, streamId: this.streamId, epoch: this.epoch,
        segmentId: `${this.streamId}:${this.epoch}:${this.segmentOrdinal}`,
        referenceIndex: this.nextReferenceIndex, sourceSampleRateHz: this.sampleRateHz,
        supportStartSample: start, supportEndSampleExclusive: endpoint,
        referenceSample: endpoint, referenceTimeMs: endpoint * 1000 / this.sampleRateHz,
        nominalReferenceTimeMs: this.referenceOriginMs + this.nextReferenceIndex * 10,
        referenceQuantizationErrorMs: endpoint * 1000 / this.sampleRateHz -
          (this.referenceOriginMs + this.nextReferenceIndex * 10),
        lastRequiredInputSample: endpoint - 1, lookaheadSamples: 0,
        legacyWindowStartMs: start * 1000 / this.sampleRateHz,
        baselineProcessorId: "browser-voice-dsp@1.1",
        baselinePitch: features.baseline.f0Hz === null
          ? { state: "unavailable", f0Hz: null, reason: "legacy-pitch-unresolved" }
          : { state: "available", f0Hz: features.baseline.f0Hz, reason: null }, features });
      this.nextReferenceIndex += 1;
    }
    return output;
  }
}

export function replayVoicePcmV1(
  samples: Float32Array, options: VoiceReplayOptions, chunkSizes: readonly number[] = [960]
): VoiceReplayFrameV1[] {
  if (chunkSizes.length === 0 || chunkSizes.some(size => !Number.isInteger(size) || size <= 0)) {
    throw new Error("Replay chunk sizes must be positive integers.");
  }
  const replay = new CausalVoiceReplay(options);
  const frames: VoiceReplayFrameV1[] = [];
  let offset = 0;
  let chunk = 0;
  while (offset < samples.length) {
    const end = Math.min(samples.length, offset + chunkSizes[chunk % chunkSizes.length]);
    frames.push(...replay.push(samples.subarray(offset, end)));
    offset = end;
    chunk += 1;
  }
  return frames;
}
