import { describe, expect, it } from "vitest";
import { audioTimeInMainClock, captureVoiceClockMapping, performanceTimeInMainClock,
  validateVoiceDurations, validVoiceClockMapping } from "./voice-stream-timing.js";

const mapping = { id: 3, mainTimeOriginMs: 100_000, mainPerformanceMs: 7_000,
  audioContextTimeSeconds: 5, anchorReadUncertaintyMs: 0.2, audioClockUncertaintyMs: null } as const;

describe("voice clock coordinates", () => {
  it("maps audio and different execution-context origins into the main performance clock", () => {
    const available = audioTimeInMainClock(5.04, mapping);
    const workerReady = performanceTimeInMainClock(52, 107_000, mapping);
    expect(available).toBeCloseTo(7_040, 8);
    expect(workerReady).toBe(7_052);
    expect(workerReady - available).toBeCloseTo(12, 8);
    expect(52 - available).toBeLessThan(0); // The former cross-context subtraction.
  });
  it("records the read bracket without claiming known audio-clock accuracy", () => {
    let call = 0;
    const captured = captureVoiceClockMapping({ currentTime: 5 }, {
      timeOrigin: 100_000, now: () => [7_000, 7_002][call++]
    }, 1);
    expect(captured.mainPerformanceMs).toBe(7_001);
    expect(captured.anchorReadUncertaintyMs).toBe(1);
    expect(captured.audioClockUncertaintyMs).toBeNull();
  });
  it("fails invalid mappings and negative durations instead of clamping them", () => {
    expect(validVoiceClockMapping({ ...mapping, mainTimeOriginMs: NaN })).toBe(false);
    expect(() => audioTimeInMainClock(Infinity, mapping)).toThrow("invalid-voice-clock-mapping");
    expect(() => validateVoiceDurations({ workerServiceMs: 2,
      sampleAvailabilityToWorkerReadyMs: -3, blockEndToWorkerReceiptMs: 1 }))
      .toThrow("invalid-voice-clock-duration");
    expect(() => validateVoiceDurations({ workerServiceMs: 2,
      sampleAvailabilityToWorkerReadyMs: 3, blockEndToWorkerReceiptMs: NaN }))
      .toThrow("invalid-voice-clock-duration");
  });
});
