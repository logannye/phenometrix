import { fftInPlace } from "./dsp-primitives.js";

/** Research-only YIN formula port: float32 input, consistent float64 internals.
 * Numerical equivalence to librosa 0.11 float32 is checked, not assumed bit-exact. */
export const RAW_YIN_IMPLEMENTATION_VERSION = "phenometrix.raw-yin-librosa011-semantics.float64.ts.v2" as const;
export const HYBRID_ESTIMATOR_CONTRACT = "phenometrix.mlp-voicing-yin-pitch.v1" as const;
export const ORIGINAL_MLP_WEIGHTS_SHA256 = "06110469a9a643fb16806905569ebc939db2c0bdd5b3a6a1593b31789453d10d";
export interface RawYinEvidence {
  implementationVersion: typeof RAW_YIN_IMPLEMENTATION_VERSION;
  referenceLibrary: "librosa@0.11.0";
  rawF0Hz: number;
  selectedPeriod: number;
  parabolicShift: number;
  thresholdTrough: boolean;
  serviceMs?: number;
}

const FRAME = 1920;
const MIN_PERIOD = 68; // floor(48000 / 700)
const MAX_PERIOD = 960; // ceil(48000 / 50)
const FFT_SIZE = 4096; // Linear autocorrelation, no circular overlap within the searched lags.
const TINY_FLOAT64 = 2.2250738585072014e-308;

/** One fixed scratch set per research worker; no retained PCM or temporal YIN state. */
export class RawYin48k {
  private readonly real = new Float64Array(FFT_SIZE);
  private readonly imaginary = new Float64Array(FFT_SIZE);
  private readonly acf = new Float64Array(MAX_PERIOD + 1);
  private readonly prefix = new Float64Array(FRAME);
  private readonly difference = new Float64Array(MAX_PERIOD + 1);
  private readonly normalized = new Float64Array(MAX_PERIOD - MIN_PERIOD + 1);

  process(samples: Float32Array): RawYinEvidence {
    if (samples.length !== FRAME || !samples.every(Number.isFinite)) {
      throw new Error("Raw research YIN requires exactly 1920 finite float32 samples at 48kHz.");
    }
    this.real.fill(0); this.imaginary.fill(0); this.real.set(samples);
    fftInPlace(this.real, this.imaginary);
    for (let bin = 0; bin < FFT_SIZE; bin += 1) {
      this.real[bin] = this.real[bin] ** 2 + this.imaginary[bin] ** 2;
      this.imaginary[bin] = 0;
    }
    // The power spectrum is real/even, so the real part of the forward transform
    // equals the inverse transform after division by N. No window or DC removal.
    fftInPlace(this.real, this.imaginary);
    for (let lag = 0; lag <= MAX_PERIOD; lag += 1) this.acf[lag] = this.real[lag] / FFT_SIZE;

    // Keep FFT, energy, difference and cumulative normalization consistently f64.
    // No input-specific tie epsilon or preferred-period override is applied.
    let energy = 0;
    for (let sample = 0; sample < FRAME; sample += 1) {
      energy = energy + samples[sample] * samples[sample];
      this.prefix[sample] = energy;
    }
    this.prefix[0] = 0; // librosa writes yin_frames[0]=0 before evaluating its difference RHS.
    this.difference[0] = 0;
    let cumulative = 0;
    for (let lag = 1; lag <= MAX_PERIOD; lag += 1) {
      this.difference[lag] = 2 * (this.acf[0] - this.acf[lag]) - this.prefix[lag - 1];
      cumulative = cumulative + this.difference[lag];
      if (lag >= MIN_PERIOD) this.normalized[lag - MIN_PERIOD] =
        this.difference[lag] / (cumulative / lag + TINY_FLOAT64);
    }
    let selected = 0;
    let thresholdTrough = false;
    for (let index = 0; index < this.normalized.length; index += 1) {
      const value = this.normalized[index];
      if (value < this.normalized[selected]) selected = index; // First global minimum on ties.
      const trough = index === 0 ? value < this.normalized[1]
        : index === this.normalized.length - 1 ? value < this.normalized[index - 1]
        : value < this.normalized[index - 1] && value <= this.normalized[index + 1];
      if (trough && value < 0.1) { selected = index; thresholdTrough = true; break; }
    }
    let parabolicShift = 0;
    if (selected > 0 && selected < this.normalized.length - 1) {
      const a = this.normalized[selected + 1] + this.normalized[selected - 1] - 2 * this.normalized[selected];
      const b = (this.normalized[selected + 1] - this.normalized[selected - 1]) / 2;
      if (Math.abs(b) < Math.abs(a)) parabolicShift = -b / a;
    }
    const selectedPeriod = MIN_PERIOD + selected;
    return { implementationVersion: RAW_YIN_IMPLEMENTATION_VERSION, referenceLibrary: "librosa@0.11.0",
      rawF0Hz: 48000 / (selectedPeriod + parabolicShift), selectedPeriod, parabolicShift, thresholdTrough };
  }

  clear(): void {
    this.real.fill(0); this.imaginary.fill(0); this.acf.fill(0);
    this.prefix.fill(0); this.difference.fill(0); this.normalized.fill(0);
  }
}

export function rawYin48k(samples: Float32Array): RawYinEvidence { return new RawYin48k().process(samples); }

/** Frozen hybrid rule. The candidate head and prior RMS/correlation gates are unused. */
export function combineMlpVoicingAndRawYin(probability: number, rawF0Hz: number | null): {
  state: "voiced" | "unvoiced" | "unavailable"; f0Hz: number | null;
} {
  if (!Number.isFinite(probability) || probability < 0 || probability > 1) throw new Error("Invalid research voicing probability.");
  if (probability < 0.5) return { state: "unvoiced", f0Hz: null };
  if (rawF0Hz === null || !Number.isFinite(rawF0Hz) || rawF0Hz < 50 || rawF0Hz > 700) {
    return { state: "unavailable", f0Hz: null };
  }
  return { state: "voiced", f0Hz: rawF0Hz };
}
