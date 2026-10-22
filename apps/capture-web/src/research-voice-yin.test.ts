import { describe, expect, it } from "vitest";
import { RawYin48k, combineMlpVoicingAndRawYin } from "./research-voice-yin.js";

describe("frozen research hybrid decision", () => {
  it("uses raw voicing first and inclusive 50–700 Hz bounds without candidate or RMS gating", () => {
    expect(combineMlpVoicingAndRawYin(.499999, 300)).toEqual({ state: "unvoiced", f0Hz: null });
    expect(combineMlpVoicingAndRawYin(.49, NaN)).toEqual({ state: "unvoiced", f0Hz: null });
    for (const hz of [50, 300, 700]) expect(combineMlpVoicingAndRawYin(.5, hz)).toEqual({ state: "voiced", f0Hz: hz });
    for (const hz of [49.99999, 700.00001, NaN, Infinity, null]) expect(combineMlpVoicingAndRawYin(.5, hz)).toEqual({ state: "unavailable", f0Hz: null });
    for (const p of [-1, 1.1, NaN]) expect(() => combineMlpVoicingAndRawYin(p, 300)).toThrow();
  });
  it("preserves raw silence fallback and never turns raw YIN into its own voicing gate", () => {
    const result = new RawYin48k().process(new Float32Array(1920));
    expect(result.selectedPeriod).toBe(68);
    expect(result.parabolicShift).toBe(0);
    expect(result.thresholdTrough).toBe(false);
    expect(result.rawF0Hz).toBe(48000 / 68);
    expect(combineMlpVoicingAndRawYin(.8, result.rawF0Hz).state).toBe("unavailable");
  });
  it("uses exact finite support and no previous-window state", () => {
    const yin = new RawYin48k();
    expect(() => yin.process(new Float32Array(1919))).toThrow();
    const invalid = new Float32Array(1920); invalid[1919] = NaN;
    expect(() => yin.process(invalid)).toThrow();
    const input = Float32Array.from({ length: 1920 }, (_, i) => .12 * Math.sin(2 * Math.PI * 300 * i / 48000));
    const before = Float32Array.from(input);
    const original = yin.process(input);
    yin.process(new Float32Array(1920));
    expect(yin.process(input)).toEqual(original);
    yin.clear(); expect(yin.process(input)).toEqual(original);
    expect(input).toEqual(before);
  });
});
