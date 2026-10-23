import { describe, expect, it } from "vitest";
import { SUPPORT_DIAGNOSTIC_SCENARIOS, cyclesAt, frequencyAt, supportDiagnosticPcm }
  from "../../../research/voice-prosody/support_diagnostic.js";

describe("analytic support diagnostic ground truth", () => {
  it("predeclares three constants, six slopes, and two steps", () => {
    expect(SUPPORT_DIAGNOSTIC_SCENARIOS).toHaveLength(11);
    expect(SUPPORT_DIAGNOSTIC_SCENARIOS.filter(scenario => scenario.kind === "glide")
      .map(scenario => scenario.slopeHzPerSecond)).toEqual([-200, -100, -50, 50, 100, 200]);
  });
  it("integrates each smooth segment with the declared instantaneous derivative", () => {
    for (const scenario of SUPPORT_DIAGNOSTIC_SCENARIOS) {
      for (const time of [.1, .4, .8, 1.1, 1.5]) {
        const derivative = (cyclesAt(scenario, time + 1e-6) - cyclesAt(scenario, time - 1e-6)) / 2e-6;
        expect(derivative).toBeCloseTo(frequencyAt(scenario, time), 5);
      }
      expect(supportDiagnosticPcm(scenario)).toHaveLength(76800);
    }
  });
  it("keeps phase continuous through source steps and averages complete support analytically", () => {
    const step = SUPPORT_DIAGNOSTIC_SCENARIOS.find(scenario => scenario.id === "step-100-200")!;
    expect(cyclesAt(step, .6)).toBe(60);
    expect(frequencyAt(step, .6)).toBe(200);
    expect((cyclesAt(step, .62) - cyclesAt(step, .58)) / .04).toBeCloseTo(150, 10);
  });
});
