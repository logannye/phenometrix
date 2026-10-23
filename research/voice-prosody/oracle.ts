import type { VoiceReplayFrameV1 } from "../../apps/capture-web/src/voice-replay.js";

export interface VoiceOracleReferenceV1 {
  referenceIndex: number;
  valid: boolean;
  voiced: boolean;
  f0Hz: number | null;
}

/**
 * Scorer only. Predictions must be complete before reference rows are passed.
 * Common boundary eligibility includes the causal 40 ms and centered 64 ms
 * comparator supports. At nonintegral sample endpoints use ceil for support.
 * Out-of-range reference F0 remains in the all-voiced denominator.
 */
export function scoreVoiceCandidateOracleV1(
  predictions: readonly VoiceReplayFrameV1[],
  references: readonly VoiceOracleReferenceV1[],
  recordingSampleCount: number,
  toleranceCents = 50
) {
  if (!Number.isSafeInteger(recordingSampleCount) || recordingSampleCount < 0 ||
      !Number.isFinite(toleranceCents) || toleranceCents <= 0) throw new Error("Invalid oracle scorer configuration.");
  const byIndex = new Map<number, VoiceOracleReferenceV1>();
  for (const row of references) {
    if (!Number.isSafeInteger(row.referenceIndex) || row.referenceIndex < 0 || byIndex.has(row.referenceIndex)) {
      throw new Error("Reference indices must be unique nonnegative integers.");
    }
    if (row.valid && row.voiced && (row.f0Hz === null || !Number.isFinite(row.f0Hz) || row.f0Hz <= 0)) {
      throw new Error("Valid voiced references require finite positive F0.");
    }
    byIndex.set(row.referenceIndex, row);
  }
  let eligibleFrames = 0;
  let boundaryExcludedFrames = 0;
  let missingReferenceFrames = 0;
  let invalidReferenceFrames = 0;
  let voicedFrames = 0;
  let outOfRangeVoicedFrames = 0;
  let oracleCoveredVoicedFrames = 0;
  let legacyCorrectVoicedFrames = 0;
  const seen = new Set<number>();
  for (const prediction of predictions) {
    if (seen.has(prediction.referenceIndex)) throw new Error("Duplicate prediction reference index.");
    seen.add(prediction.referenceIndex);
    if (prediction.supportStartSample < 0 || prediction.referenceSample +
        Math.ceil(prediction.sourceSampleRateHz * 0.032) > recordingSampleCount) {
      boundaryExcludedFrames += 1;
      continue;
    }
    const reference = byIndex.get(prediction.referenceIndex);
    if (reference === undefined) { missingReferenceFrames += 1; continue; }
    if (!reference.valid) { invalidReferenceFrames += 1; continue; }
    eligibleFrames += 1;
    if (!reference.voiced) continue;
    voicedFrames += 1;
    const hz = reference.f0Hz!;
    if (hz < 50 || hz > 700) outOfRangeVoicedFrames += 1;
    const within = (estimate: number | null) => estimate !== null && estimate > 0 &&
      1200 * Math.abs(Math.log2(estimate / hz)) <= toleranceCents;
    if (prediction.features.candidates.some(candidate => candidate.valid && within(candidate.hz))) {
      oracleCoveredVoicedFrames += 1;
    }
    if (within(prediction.baselinePitch.f0Hz)) legacyCorrectVoicedFrames += 1;
  }
  return {
    schemaVersion: "phenometrix.voice-candidate-oracle.v1", toleranceCents,
    eligibility: "causal-40ms-and-centered-64ms-supports-exist",
    eligibleFrames, boundaryExcludedFrames, missingReferenceFrames, invalidReferenceFrames,
    voicedFrames, outOfRangeVoicedFrames, oracleCoveredVoicedFrames, legacyCorrectVoicedFrames,
    oracleCoverage: voicedFrames === 0 ? null : oracleCoveredVoicedFrames / voicedFrames,
    legacyCorrectVoicedFraction: voicedFrames === 0 ? null : legacyCorrectVoicedFrames / voicedFrames
  };
}
