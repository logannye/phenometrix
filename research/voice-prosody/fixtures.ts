/** Analytic mechanism fixtures, not evidence of natural-speech performance. */
export interface SyntheticVoiceFixture {
  id: string;
  sampleRateHz: number;
  pcm: Float32Array;
  transitionSample: number;
  referenceF0HzAtSample: (sample: number) => number;
}
export function makeVoiceMechanismFixture(
  kind: "steady" | "harmonic-change" | "source-step" | "source-glide" | "silence",
  sampleRateHz = 48_000, durationSeconds = 0.6
): SyntheticVoiceFixture {
  const length = Math.round(sampleRateHz * durationSeconds);
  const transitionSample = Math.floor(length / 2);
  if (!Number.isInteger(sampleRateHz) || sampleRateHz < 8000 || length <= 0) {
    throw new Error("Invalid synthetic fixture duration/rate.");
  }
  const referenceF0HzAtSample = (sample: number): number => {
    if (kind === "silence") return 0;
    if (kind === "source-step") return sample < transitionSample ? 150 : 300;
    if (kind === "source-glide") return 150 + 150 * sample / Math.max(1, length - 1);
    return 150;
  };
  const pcm = new Float32Array(length);
  let phase = 0;
  for (let sample = 0; sample < length; sample += 1) {
    const hz = referenceF0HzAtSample(sample);
    phase += 2 * Math.PI * hz / sampleRateHz;
    if (kind === "silence") continue;
    const fundamental = kind === "harmonic-change" && sample >= transitionSample ? 0 : 1;
    // Phase-continuous excitation; changing observation harmonics preserves F0.
    pcm[sample] = 0.075 * (fundamental * Math.sin(phase) +
      0.7 * Math.sin(2 * phase) + 0.35 * Math.sin(3 * phase));
  }
  return { id: `synthetic-${kind}-v1`, sampleRateHz, pcm, transitionSample,
    referenceF0HzAtSample };
}
