import { describe, expect, it } from "vitest";
import { CausalVoiceReplay, replayVoicePcmV1 } from "./voice-replay.js";
import { makeVoiceMechanismFixture } from "../../../research/voice-prosody/fixtures.js";
import { scoreVoiceCandidateOracleV1 } from "../../../research/voice-prosody/oracle.js";

const shortFixture = () => makeVoiceMechanismFixture("source-step", 48_000, 0.12);

describe("causal public-data replay", () => {
  it("uses the reference grid with complete trailing support and explicit legacy start", () => {
    const fixture = shortFixture();
    const frames = replayVoicePcmV1(fixture.pcm, { sampleRateHz: 48_000 });
    expect(frames[0]).toMatchObject({ referenceIndex: 3, referenceSample: 2208,
      referenceTimeMs: 46, supportStartSample: 288, supportEndSampleExclusive: 2208,
      lastRequiredInputSample: 2207, legacyWindowStartMs: 6, lookaheadSamples: 0 });
    const native = replayVoicePcmV1(fixture.pcm, { sampleRateHz: 48_000, referenceOriginMs: 40 });
    expect(native[0]).toMatchObject({ referenceIndex: 0, referenceSample: 1920,
      referenceTimeMs: 40, legacyWindowStartMs: 0 });
  });

  it("is exactly invariant to uneven input chunks", () => {
    const fixture = shortFixture();
    const options = { sampleRateHz: 48_000 };
    expect(replayVoicePcmV1(fixture.pcm, options, [1, 3, 127, 3000, 17]))
      .toEqual(replayVoicePcmV1(fixture.pcm, options, [fixture.pcm.length]));
  });

  it("future suffix interventions cannot change prefix predictions or state", () => {
    const fixture = makeVoiceMechanismFixture("steady", 48_000, 0.16);
    const alternate = fixture.pcm.slice();
    const prefixLength = 3840;
    alternate.fill(0, prefixLength);
    const original = replayVoicePcmV1(fixture.pcm, { sampleRateHz: 48_000 });
    const changed = replayVoicePcmV1(alternate, { sampleRateHz: 48_000 });
    expect(changed.filter(frame => frame.referenceSample <= prefixLength))
      .toEqual(original.filter(frame => frame.referenceSample <= prefixLength));
  });

  it("keeps storage bounded and resets every feature on gaps", () => {
    const fixture = shortFixture();
    const replay = new CausalVoiceReplay({ sampleRateHz: 48_000, streamId: "fixture" });
    const first = replay.push(fixture.pcm);
    expect(replay.retainedSampleCount).toBe(replay.windowSamples);
    const next = replay.push(fixture.pcm, 9600);
    expect(next[0].segmentId).not.toEqual(first[0].segmentId);
    expect(next[0].supportStartSample).toBeGreaterThanOrEqual(9600);
    expect(next[0].features.values.at(-1)).toBe(0);
    expect(next[0].features.baseline.spectralFlux).toBe(0);
    expect(() => replay.push(fixture.pcm, 0)).toThrow(/interval/);
  });

  it("rejects a nonfinite block without partially advancing replay state", () => {
    const replay = new CausalVoiceReplay({ sampleRateHz: 48000 });
    const pcm = new Float32Array(3000);
    pcm[2999] = NaN;
    expect(() => replay.push(pcm)).toThrow(/finite/);
    expect(replay.nextInputSample).toBe(0);
    expect(replay.retainedSampleCount).toBe(0);
  });

  it("keeps unresolved pitch unavailable and does not invent unvoiced evidence", () => {
    const frames = replayVoicePcmV1(new Float32Array(4800), { sampleRateHz: 48000 });
    expect(frames[0].baselinePitch).toEqual({ state: "unavailable", f0Hz: null,
      reason: "legacy-pitch-unresolved" });
  });
});

describe("scorer-only candidate coverage", () => {
  it("changing labels changes scores but cannot change acquisition predictions", () => {
    const fixture = makeVoiceMechanismFixture("steady", 48000, 0.12);
    const frames = replayVoicePcmV1(fixture.pcm, { sampleRateHz: 48000 });
    const before = JSON.stringify(frames);
    const references = frames.map(frame => ({ referenceIndex: frame.referenceIndex,
      valid: true, voiced: true, f0Hz: 150 }));
    const score = scoreVoiceCandidateOracleV1(frames, references, fixture.pcm.length);
    const changed = scoreVoiceCandidateOracleV1(frames,
      references.map(row => ({ ...row, f0Hz: 900 })), fixture.pcm.length);
    expect(score.oracleCoverage).toBe(1);
    expect(changed.oracleCoverage).toBe(0);
    expect(changed.outOfRangeVoicedFrames).toBe(changed.voicedFrames);
    expect(JSON.stringify(frames)).toBe(before);
    expect(score.boundaryExcludedFrames).toBeGreaterThan(0);
  });

  it("reports absent or invalid references without treating them as unvoiced", () => {
    const fixture = shortFixture();
    const frames = replayVoicePcmV1(fixture.pcm, { sampleRateHz: 48000 });
    const score = scoreVoiceCandidateOracleV1(frames, [{ referenceIndex: frames[0].referenceIndex,
      valid: false, voiced: false, f0Hz: null }], fixture.pcm.length);
    expect(score.invalidReferenceFrames).toBe(1);
    expect(score.missingReferenceFrames).toBeGreaterThan(0);
    expect(score.voicedFrames).toBe(0);
    expect(score.oracleCoverage).toBeNull();
  });
});
