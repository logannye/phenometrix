import { describe, expect, it } from "vitest";
import { extractVoicePitchFeaturesV1 } from "./voice-pitch-features.js";
import { extractVoicePitchFeaturesV2 } from "./voice-pitch-features-v2.js";
import { replayVoicePcmV1 } from "./voice-replay.js";
import { makeVoiceMechanismFixture } from "../../../research/voice-prosody/fixtures.js";

describe("single refined-lag candidate revision", () => {
  it("retains every v1 candidate and the unchanged classical output", () => {
    const fixture = makeVoiceMechanismFixture("steady", 48000, .04);
    const v1 = extractVoicePitchFeaturesV1(fixture.pcm, 48000);
    const v2 = extractVoicePitchFeaturesV2(fixture.pcm, 48000);
    expect(v2.baseline).toEqual(v1.baseline);
    expect(v2.values).toHaveLength(57);
    expect(v2.candidates).toHaveLength(6);
    for (const candidate of v1.candidates.filter(candidate => candidate.valid)) {
      expect(v2.candidates.some(other => other.hz === candidate.hz)).toBe(true);
    }
    expect(v2.evidence.refinedLagEvidence.full40.length).toBeLessThanOrEqual(1);
    expect(v2.evidence.refinedLagEvidence.suffix20.length).toBeLessThanOrEqual(2);
    expect(v2.evidence.refinedLagEvidence.suffix20.every(candidate => candidate.hz >= 100)).toBe(true);
  });
  it("does not invent candidates on silence", () => {
    const result = extractVoicePitchFeaturesV2(new Float32Array(1920), 48000);
    expect(result.values.every(Number.isFinite)).toBe(true);
    expect(result.candidates.every(candidate => !candidate.valid && candidate.hz === null)).toBe(true);
  });
  it("is chunk invariant and future independent after the representation revision", () => {
    const fixture = makeVoiceMechanismFixture("source-glide", 48000, .12);
    const options = { sampleRateHz: 48000, featureVersion: "v2" as const };
    const expected = replayVoicePcmV1(fixture.pcm, options, [fixture.pcm.length]);
    expect(replayVoicePcmV1(fixture.pcm, options, [1, 129, 871, 11])).toEqual(expected);
    const alternate = fixture.pcm.slice();
    alternate.fill(0, 3840);
    expect(replayVoicePcmV1(alternate, options).filter(frame => frame.referenceSample <= 3840))
      .toEqual(expected.filter(frame => frame.referenceSample <= 3840));
  });
});
