/** Research-only, window-local evidence for already observed candidate periods.
 *
 * No candidate generation, voicing decision, labels, history, or future input.
 * The caller supplies grouped emitted frequencies, preserving their ordering.
 * This module deliberately has no Node or browser-platform dependencies.
 */
export const PERIODICITY_EVIDENCE_SCHEMA = 'phenometrix.candidate-periodicity-evidence.v1';
export const PERIODICITY_WINDOW_SAMPLES = 1920;
export const PERIODICITY_SAMPLE_RATE_HZ = 48000;
export const PERIODICITY_MAX_CANDIDATES = 7;
export const PERIODICITY_MIN_ENERGY = 1e-12;
const PERIOD_FACTORS = [1, .5, 1 / 3, .25] as const;

export interface PeriodicityEvidence {
  /** Candidate-major Float32 [C(T), (C(T)-max proper-subperiod C)/2, valid]. */
  channels: Float32Array;
  /** Candidate-major Float64 correlations at [T, T/2, T/3, T/4]. */
  correlations: Float64Array;
  valid: Uint8Array;
  /** Zero for masked candidates; retained for active low-energy candidates. */
  periodSamples: Float64Array;
  anchorLength: Int32Array;
}

/** Exact 40-ms Float32 support. All arithmetic before channels is Float64. */
export function inspectPeriodicityEvidence(
  pcm: Float32Array,
  sampleRateHz: number,
  candidateHz: ArrayLike<number>,
  candidateMask: ArrayLike<boolean | number>,
): PeriodicityEvidence {
  if (!(pcm instanceof Float32Array) || pcm.length !== PERIODICITY_WINDOW_SAMPLES) {
    throw new RangeError('Periodicity evidence requires exactly 1920 Float32 PCM samples');
  }
  if (sampleRateHz !== PERIODICITY_SAMPLE_RATE_HZ) {
    throw new RangeError('Periodicity evidence requires 48000 Hz');
  }
  const count = candidateHz.length;
  if ((count !== 6 && count !== PERIODICITY_MAX_CANDIDATES)
      || candidateMask.length !== count) {
    throw new RangeError('Periodicity evidence requires matching candidate/mask lengths of six or seven');
  }
  for (let c = 0; c < count; c += 1) {
    const mask = candidateMask[c];
    if (mask !== false && mask !== true && mask !== 0 && mask !== 1) {
      throw new RangeError('Candidate mask must contain only booleans or zero/one');
    }
    if (mask && (!Number.isFinite(candidateHz[c]) || candidateHz[c] < 50 || candidateHz[c] > 700)) {
      throw new RangeError('Active grouped candidate frequency must be finite within 50..700 Hz');
    }
  }
  let mean = 0;
  for (let i = 0; i < pcm.length; i += 1) {
    if (!Number.isFinite(pcm[i])) throw new RangeError('Finite PCM required');
    mean += pcm[i];
  }
  mean /= pcm.length;
  const centered = new Float64Array(pcm.length);
  for (let i = 0; i < pcm.length; i += 1) centered[i] = pcm[i] - mean;
  const result: PeriodicityEvidence = {
    channels: new Float32Array(count * 3), correlations: new Float64Array(count * 4),
    valid: new Uint8Array(count), periodSamples: new Float64Array(count),
    anchorLength: new Int32Array(count),
  };
  for (let c = 0; c < count; c += 1) {
    if (!candidateMask[c]) continue;
    const period = sampleRateHz / candidateHz[c];
    const length = pcm.length - Math.floor(period) - 1;
    result.periodSamples[c] = period;
    result.anchorLength[c] = length;
    let anchorEnergy = 0;
    for (let i = 0; i < length; i += 1) anchorEnergy += centered[i] * centered[i];
    if (anchorEnergy < PERIODICITY_MIN_ENERGY) continue;
    const correlations = [0, 0, 0, 0];
    let allEnergiesValid = true;
    for (let s = 0; s < 4; s += 1) {
      const shift = period * PERIOD_FACTORS[s];
      const integer = Math.floor(shift);
      const fraction = shift - integer;
      let product = 0;
      let shiftedEnergy = 0;
      for (let i = 0; i < length; i += 1) {
        const shifted = (1 - fraction) * centered[i + integer]
          + fraction * centered[i + integer + 1];
        product += centered[i] * shifted;
        shiftedEnergy += shifted * shifted;
      }
      if (shiftedEnergy < PERIODICITY_MIN_ENERGY) {
        allEnergiesValid = false;
        break;
      }
      correlations[s] = Math.max(-1, Math.min(1, product / Math.sqrt(anchorEnergy * shiftedEnergy)));
    }
    // An invalid subperiod invalidates all evidence rather than fabricating a
    // high primitive-period contrast from unavailable shifted samples.
    if (!allEnergiesValid) continue;
    result.correlations.set(correlations, c * 4);
    result.channels[c * 3] = correlations[0];
    result.channels[c * 3 + 1] = (correlations[0]
      - Math.max(correlations[1], correlations[2], correlations[3])) / 2;
    result.channels[c * 3 + 2] = 1;
    result.valid[c] = 1;
  }
  return result;
}
