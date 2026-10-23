import {
  analyzeVoiceWindow,
  inspectVoicePitchEvidenceV1,
  type VoicePitchEvidenceV1,
  type VoiceWindowAnalysis
} from "./voice-dsp.js";

export const VOICE_PITCH_FEATURE_VERSION = "phenometrix.voice-pitch-features.v1";
export const VOICE_PITCH_CANDIDATE_COUNT = 3;
export const VOICE_PITCH_BOUNDS_HZ = Object.freeze([50, 700] as const);
const sources = ["autocorrelation", "amdf", "zeroCrossing"] as const;
export type VoicePitchSource = typeof sources[number];

/** Candidate order is strictly increasing Hz; exact duplicates merge sources. */
export interface VoicePitchCandidateV1 {
  valid: boolean;
  hz: number | null;
  sourceMask: number; // autocorrelation=1, AMDF=2, zero crossing=4
  confidenceBySource: readonly [number, number, number];
}

export const VOICE_PITCH_FEATURE_NAMES = Object.freeze([
  ...Array.from({ length: VOICE_PITCH_CANDIDATE_COUNT }, (_, i) => [
    `candidate_${i}_valid`, `candidate_${i}_log2_hz_over_50`,
    `candidate_${i}_autocorrelation_score`, `candidate_${i}_amdf_score`,
    `candidate_${i}_zero_crossing_score`, `candidate_${i}_maximum_score`
  ]).flat(),
  "log_rms_floor_1e-8", "delta_log_rms", "dc_offset", "clipped_sample_fraction",
  "spectral_flux", "legacy_confidence_score", "legacy_estimator_agreement",
  "valid_candidate_fraction", "history_present"
]);

export interface VoicePitchFeatureHistoryV1 {
  bandEnergies: readonly number[];
  logRms: number;
}

export interface VoicePitchFeaturesV1 {
  schemaVersion: typeof VOICE_PITCH_FEATURE_VERSION;
  featureNames: typeof VOICE_PITCH_FEATURE_NAMES;
  values: readonly number[];
  candidates: readonly VoicePitchCandidateV1[];
  evidence: VoicePitchEvidenceV1;
  baseline: VoiceWindowAnalysis;
  nextHistory: VoicePitchFeatureHistoryV1;
}

/**
 * Scores retain their heuristic meaning. No field is a voicing probability or
 * calibrated correctness probability. Initial candidates are only the three
 * legacy hypotheses, so oracle coverage must be checked before model training.
 * Reference labels, eligibility masks and laryngograph signals are not inputs.
 */
export function extractVoicePitchFeaturesV1(
  samples: Float32Array,
  sampleRate: number,
  history: VoicePitchFeatureHistoryV1 | null = null
): VoicePitchFeaturesV1 {
  const evidence = inspectVoicePitchEvidenceV1(samples, sampleRate);
  if (history !== null && (
    !Number.isFinite(history.logRms) || history.bandEnergies.length !== 16 ||
    history.bandEnergies.some(value => !Number.isFinite(value) || value < 0)
  )) throw new Error("Feature history must contain finite prior acquisition features.");
  const baseline = analyzeVoiceWindow(samples, sampleRate, history?.bandEnergies ?? null);
  const byFrequency = new Map<number, { sourceMask: number; scores: [number, number, number] }>();
  sources.forEach((source, index) => {
    const candidate = evidence[source];
    if (candidate.hz === null || !Number.isFinite(candidate.hz) ||
        candidate.hz < VOICE_PITCH_BOUNDS_HZ[0] || candidate.hz > VOICE_PITCH_BOUNDS_HZ[1]) return;
    const entry = byFrequency.get(candidate.hz) ?? { sourceMask: 0, scores: [0, 0, 0] };
    entry.sourceMask |= 1 << index;
    entry.scores[index] = candidate.confidence;
    byFrequency.set(candidate.hz, entry);
  });
  const candidates: VoicePitchCandidateV1[] = [...byFrequency.entries()]
    .sort(([left], [right]) => left - right)
    .map(([hz, entry]) => ({ valid: true, hz, sourceMask: entry.sourceMask,
      confidenceBySource: entry.scores }));
  const validCount = candidates.length;
  while (candidates.length < VOICE_PITCH_CANDIDATE_COUNT) {
    candidates.push({ valid: false, hz: null, sourceMask: 0, confidenceBySource: [0, 0, 0] });
  }
  const logRms = Math.log(Math.max(1e-8, baseline.rms));
  const values = candidates.flatMap(candidate => candidate.valid ? [
    1, Math.log2(candidate.hz! / 50), ...candidate.confidenceBySource,
    Math.max(...candidate.confidenceBySource)
  ] : [0, 0, 0, 0, 0, 0]);
  values.push(logRms, history === null ? 0 : logRms - history.logRms,
    baseline.dcOffset, baseline.clippedSampleFraction, baseline.spectralFlux,
    baseline.f0Confidence, baseline.estimatorAgreement,
    validCount / VOICE_PITCH_CANDIDATE_COUNT, history === null ? 0 : 1);
  if (values.some(value => !Number.isFinite(value))) throw new Error("Nonfinite voice feature.");
  return {
    schemaVersion: VOICE_PITCH_FEATURE_VERSION,
    featureNames: VOICE_PITCH_FEATURE_NAMES,
    values,
    candidates,
    evidence,
    baseline,
    nextHistory: { bandEnergies: [...baseline.bandEnergies], logRms }
  };
}

/** No all-masked softmax: selection returns null when no candidate is valid. */
export function selectVoicePitchCandidateV1(
  candidates: readonly VoicePitchCandidateV1[], logits: readonly number[]
): VoicePitchCandidateV1 | null {
  if (candidates.length !== VOICE_PITCH_CANDIDATE_COUNT || logits.length !== candidates.length ||
      logits.some(value => !Number.isFinite(value))) throw new Error("Invalid candidate logits.");
  let selected: VoicePitchCandidateV1 | null = null;
  let best = -Infinity;
  candidates.forEach((candidate, index) => {
    // Strict comparison selects the lowest-frequency valid slot on a tie.
    if (candidate.valid && logits[index] > best) {
      selected = candidate;
      best = logits[index];
    }
  });
  return selected;
}
