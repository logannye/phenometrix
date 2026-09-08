import type { VoiceSignalFrameV1 } from "@phenometrix/ambient-core";

export const PASSIVE_NOISE_PROCESSOR_REF = "passive-screened-noise@1.0.0" as const;
export const PASSIVE_NOISE_LIMITS = Object.freeze({
  analysisWindowMs: 40, maximumFrameGapMs: 40, minimumCandidateDurationMs: 2_000,
  minimumContrastDurationMs: 400, maximumEvidenceAgeMs: 30_000,
  minimumRms: 0.0001, maximumCandidateRms: 0.002,
  minimumContrastRms: 0.012, minimumContrastRatio: 8,
  maximumCandidateFlux: 0.01, minimumMedianCppDropDb: 3,
  maximumCandidatePitchConfidence: 0.3, maximumCandidateEstimatorAgreement: 0.3,
  minimumContrastPitchConfidence: 0.8, minimumContrastEstimatorAgreement: 0.8,
  minimumContrastCppDb: 8, maximumCandidateRmsRatio: 1.5
});

export interface PassiveNoiseReference {
  noiseFloorRms: number;
  durationMs: number;
  processorRef: typeof PASSIVE_NOISE_PROCESSOR_REF;
  qualification: "engineering-only";
}

/**
 * Conservative acoustic reference, not a validated speech/non-speech classifier.
 * Uses neither speechActive, periodic, snrDb nor qualityReasons: those may depend
 * on the noise floor being estimated. Only scalar DSP features are retained.
 * Strong preceding periodic activity provides contrast; it does not identify its
 * speaker. A whisper/caregiver cannot be conclusively excluded by these features.
 */
export class PassiveNoiseCalibration {
  private identity: string | null = null;
  private lastAtMs: number | null = null;
  private lastSelectedAtMs = -Infinity;
  private contrastAtMs = -Infinity;
  private contrastRms = 0;
  private contrastCppDb = 0;
  private contrastSamples: { rms: number; cppDb: number }[] = [];
  private candidates: { rms: number; cppDb: number }[] = [];

  reset(): void {
    this.identity = null; this.lastAtMs = null; this.lastSelectedAtMs = -Infinity;
    this.contrastAtMs = -Infinity; this.contrastRms = 0; this.contrastCppDb = 0;
    this.contrastSamples = []; this.candidates = [];
  }

  observe(frame: VoiceSignalFrameV1): PassiveNoiseReference | null {
    if (!this.acquisitionUsable(frame)) { this.reset(); return null; }
    const identity = JSON.stringify([frame.captureEpoch, frame.trackSegmentId, frame.processorRef,
      frame.sampleRateHz, frame.browserProcessing]);
    if (this.identity !== identity || (this.lastAtMs !== null
      && (frame.acquiredAtMs <= this.lastAtMs || frame.acquiredAtMs - this.lastAtMs > PASSIVE_NOISE_LIMITS.maximumFrameGapMs))) this.reset();
    this.identity = identity; this.lastAtMs = frame.acquiredAtMs;
    if (frame.acquiredAtMs - this.contrastAtMs > PASSIVE_NOISE_LIMITS.maximumEvidenceAgeMs) {
      this.contrastRms = 0; this.candidates = [];
    }
    const contrast = frame.rms >= PASSIVE_NOISE_LIMITS.minimumContrastRms
      && frame.f0Hz !== null && frame.f0Hz >= 70 && frame.f0Hz <= 500
      && frame.f0Confidence >= PASSIVE_NOISE_LIMITS.minimumContrastPitchConfidence
      && frame.estimatorAgreement >= PASSIVE_NOISE_LIMITS.minimumContrastEstimatorAgreement
      && frame.cepstralPeakProminenceDb! >= PASSIVE_NOISE_LIMITS.minimumContrastCppDb;
    const candidate = frame.rms <= PASSIVE_NOISE_LIMITS.maximumCandidateRms
      && frame.f0Hz === null && frame.f0Confidence <= PASSIVE_NOISE_LIMITS.maximumCandidatePitchConfidence
      && frame.estimatorAgreement <= PASSIVE_NOISE_LIMITS.maximumCandidateEstimatorAgreement
      && frame.spectralFlux! <= PASSIVE_NOISE_LIMITS.maximumCandidateFlux
      && frame.cepstralPeakProminenceDb! < this.contrastCppDb;
    // Every hop can invalidate a run; only disjoint 40 ms analysis windows add exposure.
    if (!contrast) this.contrastSamples = [];
    if (!candidate || this.contrastRms < frame.rms * PASSIVE_NOISE_LIMITS.minimumContrastRatio) this.candidates = [];
    if (frame.acquiredAtMs - this.lastSelectedAtMs < PASSIVE_NOISE_LIMITS.analysisWindowMs) return null;
    this.lastSelectedAtMs = frame.acquiredAtMs;
    if (contrast) {
      this.contrastSamples.push({ rms: frame.rms, cppDb: frame.cepstralPeakProminenceDb! });
      this.contrastSamples = this.contrastSamples.slice(-PASSIVE_NOISE_LIMITS.minimumContrastDurationMs / PASSIVE_NOISE_LIMITS.analysisWindowMs);
      if (this.contrastSamples.length * PASSIVE_NOISE_LIMITS.analysisWindowMs >= PASSIVE_NOISE_LIMITS.minimumContrastDurationMs) {
        this.contrastRms = Math.min(...this.contrastSamples.map(value => value.rms));
        this.contrastCppDb = Math.min(...this.contrastSamples.map(value => value.cppDb));
        this.contrastAtMs = frame.acquiredAtMs;
      }
      return null;
    }
    if (!candidate || this.contrastRms < frame.rms * PASSIVE_NOISE_LIMITS.minimumContrastRatio) return null;
    this.candidates.push({ rms: frame.rms, cppDb: frame.cepstralPeakProminenceDb! });
    const maximumSamples = PASSIVE_NOISE_LIMITS.minimumCandidateDurationMs / PASSIVE_NOISE_LIMITS.analysisWindowMs;
    this.candidates = this.candidates.slice(-maximumSamples);
    const amplitudes = this.candidates.map(value => value.rms);
    if (Math.max(...amplitudes) / Math.min(...amplitudes) > PASSIVE_NOISE_LIMITS.maximumCandidateRmsRatio) {
      this.candidates = [this.candidates[this.candidates.length - 1]]; return null;
    }
    if (this.candidates.length < maximumSamples) return null;
    // Compare this implementation's CPP against the preceding signal, rather than
    // borrowing clinical CPP cutoffs from a different acoustic implementation.
    const cppValues = this.candidates.map(value => value.cppDb).sort((a, b) => a - b);
    if (this.contrastCppDb - cppValues[Math.floor(cppValues.length / 2)] < PASSIVE_NOISE_LIMITS.minimumMedianCppDropDb) return null;
    const sorted = amplitudes.sort((a, b) => a - b);
    // Upper quartile is conservative against overstating speech SNR.
    const noiseFloorRms = sorted[Math.floor((sorted.length - 1) * 0.75)];
    this.reset();
    return { noiseFloorRms, durationMs: maximumSamples * PASSIVE_NOISE_LIMITS.analysisWindowMs,
      processorRef: PASSIVE_NOISE_PROCESSOR_REF, qualification: "engineering-only" };
  }

  private acquisitionUsable(frame: VoiceSignalFrameV1): boolean {
    return [frame.acquiredAtMs, frame.rms, frame.f0Confidence, frame.estimatorAgreement,
      frame.spectralFlux, frame.cepstralPeakProminenceDb, frame.clippedSampleFraction,
      frame.dcOffset, frame.blockGapMs, frame.lostBlockFraction, frame.sampleRateHz].every(value => typeof value === "number" && Number.isFinite(value))
      && frame.sampleRateHz >= 16_000 && frame.rms >= PASSIVE_NOISE_LIMITS.minimumRms
      && frame.rms <= 1 && frame.f0Confidence >= 0 && frame.f0Confidence <= 1
      && frame.estimatorAgreement >= 0 && frame.estimatorAgreement <= 1 && frame.spectralFlux! >= 0
      && frame.clippedSampleFraction === 0 && Math.abs(frame.dcOffset) <= 0.005
      && frame.blockGapMs >= 0 && frame.blockGapMs <= PASSIVE_NOISE_LIMITS.maximumFrameGapMs
      && frame.lostBlockFraction === 0 && !!frame.processorRef && !!frame.trackSegmentId
      && frame.browserProcessing.echoCancellation === false && frame.browserProcessing.noiseSuppression === false
      && frame.browserProcessing.autoGainControl === false;
  }
}
