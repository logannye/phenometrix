import { inspectRefinedVoiceLagEvidenceV2 } from "./voice-dsp.js";
import { extractVoicePitchFeaturesV1, type VoicePitchFeatureHistoryV1,
  type VoicePitchFeaturesV1 } from "./voice-pitch-features.js";

export const VOICE_PITCH_FEATURE_VERSION_V2 = "phenometrix.voice-pitch-features.v2";
export const VOICE_PITCH_CANDIDATE_COUNT_V2 = 6;
export const VOICE_PITCH_FEATURE_NAMES_V2 = Object.freeze([
  ...Array.from({ length: 6 }, (_, i) => [`candidate_${i}_valid`, `candidate_${i}_log2_hz_over_50`,
    `candidate_${i}_autocorrelation_score`, `candidate_${i}_amdf_score`, `candidate_${i}_zero_crossing_score`,
    `candidate_${i}_acf40_refined_score`, `candidate_${i}_acf20_refined_score`, `candidate_${i}_maximum_score`]).flat(),
  "log_rms_floor_1e-8", "delta_log_rms", "dc_offset", "clipped_sample_fraction", "spectral_flux",
  "legacy_confidence_score", "legacy_estimator_agreement", "valid_candidate_fraction", "history_present"
]);
export interface VoicePitchCandidateV2 {
  valid: boolean;
  hz: number | null;
  sourceMask: number; // AC=1, AMDF=2, ZC=4, refined full40=8, refined suffix20=16
  confidenceBySource: readonly [number, number, number, number, number];
}
export interface VoicePitchFeaturesV2 extends Omit<VoicePitchFeaturesV1, "schemaVersion" | "candidates" | "evidence"> {
  schemaVersion: typeof VOICE_PITCH_FEATURE_VERSION_V2;
  candidates: readonly VoicePitchCandidateV2[];
  evidence: VoicePitchFeaturesV1["evidence"] & {
    refinedLagEvidence: ReturnType<typeof inspectRefinedVoiceLagEvidenceV2>;
  };
}

/**
 * One predeclared candidate revision after the original training-only oracle.
 * Up to three original hypotheses plus one refined 40 ms peak and two refined
 * final-20 ms peaks. The short support is restricted to >=100 Hz (two cycles).
 * Whole-window preprocessing remains the exact legacy window-local decimator.
 * No reference, previous true F0, time shift or learned weight enters generation.
 */
export function extractVoicePitchFeaturesV2(samples: Float32Array, sampleRate: number,
  history: VoicePitchFeatureHistoryV1 | null = null): VoicePitchFeaturesV2 {
  const legacy = extractVoicePitchFeaturesV1(samples, sampleRate, history);
  const refined = inspectRefinedVoiceLagEvidenceV2(samples, sampleRate);
  const candidates = new Map<number, { sourceMask: number; scores: [number, number, number, number, number] }>();
  for (const candidate of legacy.candidates) {
    if (candidate.valid) candidates.set(candidate.hz!, { sourceMask: candidate.sourceMask,
      scores: [...candidate.confidenceBySource, 0, 0] });
  }
  const add = (candidate: { hz: number; confidence: number }, source: 3 | 4) => {
    const entry = candidates.get(candidate.hz) ?? { sourceMask: 0, scores: [0, 0, 0, 0, 0] };
    entry.sourceMask |= 1 << source;
    entry.scores[source] = candidate.confidence;
    candidates.set(candidate.hz, entry);
  };
  refined.full40.forEach(candidate => add(candidate, 3));
  refined.suffix20.forEach(candidate => add(candidate, 4));
  const slots: VoicePitchCandidateV2[] = [...candidates.entries()].sort(([a], [b]) => a - b)
    .map(([hz, entry]) => ({ valid: true, hz, sourceMask: entry.sourceMask, confidenceBySource: entry.scores }));
  const count = slots.length;
  if (count > 6) throw new Error("Candidate revision exceeded declared bound.");
  while (slots.length < 6) slots.push({ valid: false, hz: null, sourceMask: 0, confidenceBySource: [0, 0, 0, 0, 0] });
  const values = slots.flatMap(candidate => candidate.valid
    ? [1, Math.log2(candidate.hz! / 50), ...candidate.confidenceBySource, Math.max(...candidate.confidenceBySource)]
    : [0, 0, 0, 0, 0, 0, 0, 0]);
  const global = legacy.values.slice(-9);
  global[7] = count / 6;
  values.push(...global);
  return { ...legacy, schemaVersion: VOICE_PITCH_FEATURE_VERSION_V2, featureNames: VOICE_PITCH_FEATURE_NAMES_V2,
    candidates: slots, values, evidence: { ...legacy.evidence, refinedLagEvidence: refined } };
}
