import { describe, expect, it } from "vitest";
import { analyzeVoiceWindow, inspectVoicePitchEvidenceV1 } from "./voice-dsp.js";
import { extractVoicePitchFeaturesV1, selectVoicePitchCandidateV1,
  VOICE_PITCH_FEATURE_NAMES } from "./voice-pitch-features.js";

function sine(hz: number, rate = 48_000) {
  return Float32Array.from({ length: Math.round(rate * 0.04) },
    (_, i) => 0.1 * Math.sin(2 * Math.PI * hz * i / rate));
}

describe("research pitch evidence preserves legacy arithmetic", () => {
  it.each([
    [44_100, 50, 50, 0.9999467195132351],
    [44_100, 220, 222.22222222222223, 0.9508864797483425],
    [44_100, 700, 713.7247967918678, 0.9708010623260522],
    [48_000, 50, 50, 0.9999434782973968],
    [48_000, 220, 222.22222222222223, 0.9510357139950203],
    [48_000, 700, 713.5820903501547, 0.9722189837775772]
  ])("retains pre-adapter outputs at rate %d and frequency %d", (rate, hz, expectedHz, score) => {
    const samples = sine(hz, rate);
    const before = analyzeVoiceWindow(samples, rate);
    const originalPcm = samples.slice();
    const features = extractVoicePitchFeaturesV1(samples, rate);
    expect(features.baseline).toEqual(before);
    expect(samples).toEqual(originalPcm);
    expect(features.baseline.f0Hz).toBeCloseTo(expectedHz, 8);
    expect(features.baseline.f0Confidence).toBeCloseTo(score, 8);
    // The legacy 700 Hz fixture exceeds 700 Hz; research bounds must not rewrite it.
    expect(features.candidates.every(candidate => candidate.hz === null || candidate.hz <= 700)).toBe(true);
  });

  it("pads no-candidate silence explicitly and never creates an all-masked selection", () => {
    const features = extractVoicePitchFeaturesV1(new Float32Array(1920), 48_000);
    expect(features.values).toHaveLength(27);
    expect(features.featureNames).toEqual(VOICE_PITCH_FEATURE_NAMES);
    expect(features.values.every(Number.isFinite)).toBe(true);
    expect(features.candidates).toHaveLength(3);
    expect(features.candidates.every(candidate => !candidate.valid && candidate.hz === null)).toBe(true);
    expect(selectVoicePitchCandidateV1(features.candidates, [0, 0, 0])).toBeNull();
    expect(features.baseline.f0Hz).toBeNull();
  });

  it("orders unique candidates and breaks valid-logit ties at the lowest frequency", () => {
    const features = extractVoicePitchFeaturesV1(sine(220), 48_000);
    const valid = features.candidates.filter(candidate => candidate.valid);
    expect(valid.length).toBeGreaterThan(0);
    for (let i = 1; i < valid.length; i++) expect(valid[i].hz!).toBeGreaterThan(valid[i - 1].hz!);
    expect(selectVoicePitchCandidateV1(features.candidates, [0, 0, 0])).toBe(valid[0]);
    expect(valid.some(candidate => candidate.sourceMask === 3)).toBe(true);
  });

  it("requires finite acquisition evidence and explicit feature history", () => {
    expect(() => inspectVoicePitchEvidenceV1(new Float32Array([NaN]), 48000)).toThrow(/finite/);
    expect(() => inspectVoicePitchEvidenceV1(sine(220), 4000)).toThrow(/8000/);
    expect(() => extractVoicePitchFeaturesV1(sine(220), 48000,
      { bandEnergies: [0], logRms: 0 })).toThrow(/history/);
    const first = extractVoicePitchFeaturesV1(sine(220), 48000);
    const second = extractVoicePitchFeaturesV1(sine(330), 48000, first.nextHistory);
    expect(first.values.at(-1)).toBe(0);
    expect(second.values.at(-1)).toBe(1);
    expect(second.baseline).toEqual(analyzeVoiceWindow(sine(330), 48000, first.nextHistory.bandEnergies));
  });
});
