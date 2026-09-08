import { describe, expect, it } from "vitest";
import { PassiveNoiseCalibration, PASSIVE_NOISE_PROCESSOR_REF } from "./passive-noise-calibration.js";
import { voiceFrame } from "../../../packages/encounter-capture/src/test-fixtures.js";
import type { VoiceSignalFrameV1 } from "@phenometrix/ambient-core";
import { analyzeVoiceWindow } from "./voice-dsp.js";

function frame(tMs: number, kind: "contrast" | "candidate", overrides: Partial<VoiceSignalFrameV1> = {}): VoiceSignalFrameV1 {
  return { ...voiceFrame(tMs), acquiredAtMs: tMs, spectralFlux: 0.005,
    ...(kind === "contrast" ? { rms: 0.04, f0Hz: 150, f0Confidence: 0.95, estimatorAgreement: 0.95, cepstralPeakProminenceDb: 12 }
      : { rms: 0.001, f0Hz: null, f0Confidence: 0.1, estimatorAgreement: 0.1, cepstralPeakProminenceDb: 2 }), ...overrides };
}
function contrast(estimator: PassiveNoiseCalibration, start = 0) {
  for (let t = start; t < start + 500; t += 10) estimator.observe(frame(t, "contrast"));
}

describe("passive screened noise reference (engineering qualification only)", () => {
  it("accepts a deterministic engineering fixture through the actual DSP without a clinical noise-detection claim", () => {
    const estimator = new PassiveNoiseCalibration();
    let seed = 123;
    const pcm = Float32Array.from({ length: 48_000 * 3 }, (_, i) => {
      if (i < 48_000 * 0.8) return 0.05 * Math.sin(i * 2 * Math.PI * 150 / 48_000);
      seed = (1664525 * seed + 1013904223) >>> 0;
      return (seed / 4294967296 * 2 - 1) * 0.0017;
    });
    let prior: number[] | null = null;
    let reference = null;
    // Match the actual worker's overlapping 40 ms windows and 10 ms hop.
    for (let t = 0; t <= 2_960; t += 10) {
      const samples = pcm.subarray(t * 48, t * 48 + 1920);
      const { bandEnergies, ...dsp } = analyzeVoiceWindow(samples, 48_000, prior);
      prior = bandEnergies;
      reference = estimator.observe({ ...frame(t, "candidate"), ...dsp }) ?? reference;
    }
    expect(reference).toMatchObject({ durationMs: 2_000, qualification: "engineering-only" });
    expect(reference?.noiseFloorRms).toBeGreaterThan(0.0009);
    expect(reference?.noiseFloorRms).toBeLessThan(0.0011);
  });
  it("uses disjoint acoustic evidence after observed contrast and returns only scalar provenance", () => {
    const estimator = new PassiveNoiseCalibration(); contrast(estimator);
    for (let t = 500; t < 2_480; t += 10) expect(estimator.observe(frame(t, "candidate"))).toBeNull();
    expect(estimator.observe(frame(2_480, "candidate"))).toEqual({ noiseFloorRms: 0.001, durationMs: 2_000,
      processorRef: PASSIVE_NOISE_PROCESSOR_REF, qualification: "engineering-only" });
  });
  it("does not trust floor-dependent speech activity, SNR or quality labels", () => {
    const estimator = new PassiveNoiseCalibration(); contrast(estimator);
    let result = null;
    for (let t = 500; t <= 2_500; t += 10) result = estimator.observe(frame(t, "candidate",
      { speechActive: true, periodic: true, snrDb: 80, qualityReasons: ["signal-too-quiet"] })) ?? result;
    expect(result?.noiseFloorRms).toBe(0.001);
  });
  it("withholds for indefinitely quiet, periodic or low-intensity unvoiced signals without contrast", () => {
    for (const kind of ["contrast", "candidate"] as const) {
      const estimator = new PassiveNoiseCalibration();
      for (let t = 0; t < 60_000; t += 10) expect(estimator.observe(frame(t, kind))).toBeNull();
    }
  });
  it.each([
    { rms: 0 }, { rms: 0.003 }, { rms: Number.NaN }, { spectralFlux: undefined },
    { cepstralPeakProminenceDb: null }, { cepstralPeakProminenceDb: 12 }, { cepstralPeakProminenceDb: 10 }, { spectralFlux: 0.2 },
    { f0Hz: 160 }, { f0Confidence: 0.5 }, { estimatorAgreement: 0.5 },
    { clippedSampleFraction: 0.001 }, { dcOffset: 0.1 }, { lostBlockFraction: 0.01 },
    { browserProcessing: { echoCancellation: false, noiseSuppression: true, autoGainControl: false } }
  ])("rejects ambiguous or corrupted candidate input %j", overrides => {
    const estimator = new PassiveNoiseCalibration(); contrast(estimator);
    for (let t = 500; t < 4_000; t += 10) expect(estimator.observe(frame(t, "candidate", overrides))).toBeNull();
  });
  it("invalidates contrast on a timestamp gap or capture identity change", () => {
    const estimator = new PassiveNoiseCalibration(); contrast(estimator);
    for (let t = 1_000; t < 5_000; t += 10) expect(estimator.observe(frame(t, "candidate"))).toBeNull();
    contrast(estimator, 5_000);
    for (let t = 5_500; t < 9_000; t += 10) expect(estimator.observe(frame(t, "candidate", { captureEpoch: 2 }))).toBeNull();
  });
  it("does not count duplicated overlapping analysis windows as two seconds of exposure", () => {
    const estimator = new PassiveNoiseCalibration(); contrast(estimator);
    for (let t = 500; t < 1_000; t += 1) expect(estimator.observe(frame(t, "candidate"))).toBeNull();
  });
  it("requires stable candidate amplitude and recent contrast", () => {
    const estimator = new PassiveNoiseCalibration(); contrast(estimator);
    for (let t = 500; t < 35_000; t += 10) expect(estimator.observe(frame(t, "candidate", { rms: t % 80 < 40 ? 0.0002 : 0.001 }))).toBeNull();
    for (let t = 35_000; t < 38_000; t += 10) expect(estimator.observe(frame(t, "candidate"))).toBeNull();
  });
  it("explicit reset removes all accumulated evidence", () => {
    const estimator = new PassiveNoiseCalibration(); contrast(estimator); estimator.reset();
    for (let t = 500; t < 4_000; t += 10) expect(estimator.observe(frame(t, "candidate"))).toBeNull();
  });
});
