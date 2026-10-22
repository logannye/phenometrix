import { describe, expect, it } from "vitest";
import { VoiceClockProbeTracker, VOICE_CLOCK_PROBE_VERSION } from "./voice-clock-probe.js";

const mapping = { id: 3, mainTimeOriginMs: 100_000, mainPerformanceMs: 7_000,
  audioContextTimeSeconds: 5, anchorReadUncertaintyMs: .2, audioClockUncertaintyMs: null } as const;
function fixture() {
  let now = 7_010;
  const context = { currentTime: 5.008, state: "running", sampleRate: 48_000,
    getOutputTimestamp: () => ({ contextTime: 4.99, performanceTime: 7_005 }) };
  const tracker = new VoiceClockProbeTracker(mapping, 7, { now: () => now });
  const response = (probeId: number) => ({ schemaVersion: VOICE_CLOCK_PROBE_VERSION, type: "clock-probe-result",
    probeId, captureEpoch: 7, clockMappingId: 3, renderStartFrame: 240_480, sampleRateHz: 48_000 });
  return { tracker, context, response, setNow: (value: number) => { now = value; } };
}
describe("independent render-clock diagnostic probes", () => {
  it("records raw paired reads and causal render brackets without reanchoring or treating output as capture", () => {
    const f = fixture(); const request = f.tracker.begin(f.context, 10)!;
    f.setNow(7_014);
    const result = f.tracker.receive(f.response(request.probeId))!;
    expect(result.contextRead.fixedAnchorResidualMs).toBeCloseTo(2, 8);
    expect(result.outputTimestamp).toMatchObject({ status: "observed", coordinate: "output-device-only-not-capture" });
    expect(result.renderObservation).toMatchObject({ status: "observed", requestSentAtMs: 7_010,
      responseReceivedAtMs: 7_014, roundTripMs: 4 });
    if (result.renderObservation.status !== "observed") throw new Error("Missing render observation");
    expect(result.renderObservation.fixedAnchorResidualLowerMs).toBeCloseTo(0, 8);
    expect(result.renderObservation.fixedAnchorResidualUpperMs).toBeCloseTo(4, 8);
    expect(result.audioClockUncertaintyMs).toBeNull();
    expect(mapping.mainPerformanceMs).toBe(7_000);
  });
  it("bounds requests through timeout and ignores duplicate, wrong epoch, and wrong mapping replies", () => {
    const f = fixture(); const first = f.tracker.begin(f.context, 10)!;
    expect(f.tracker.begin(f.context, 11)).toBeNull();
    expect(f.tracker.receive({ ...f.response(first.probeId), captureEpoch: 99 })).toBeNull();
    expect(f.tracker.receive({ ...f.response(first.probeId), clockMappingId: 99 })).toBeNull();
    expect(f.tracker.expire("timeout")?.renderObservation).toMatchObject({ status: "unavailable", reason: "timeout" });
    expect(f.tracker.expire("timeout")).toBeNull();
    expect(f.tracker.begin(f.context, 1010)).toBeNull(); // No timeout-generated extra request in a blocked port.
    expect(f.tracker.receive(f.response(first.probeId))).toBeNull(); // Sole credit returns, no duplicate record.
    const second = f.tracker.begin(f.context, 2010)!;
    expect(second.probeId).toBe(first.probeId + 1);
    expect(f.tracker.receive(f.response(first.probeId))).toBeNull();
    expect(f.tracker.expire("stopped")?.renderObservation).toMatchObject({ status: "unavailable", reason: "stopped" });
    expect(f.tracker.receive(f.response(second.probeId))).toBeNull();
  });
  it("preserves negative residuals and widening callback brackets rather than fitting away delayed delivery", () => {
    const f = fixture(); f.context.currentTime = 5.02;
    const request = f.tracker.begin(f.context, 10)!;
    f.setNow(7_360);
    const result = f.tracker.receive(f.response(request.probeId))!;
    expect(result.contextRead.fixedAnchorResidualMs).toBeCloseTo(-10, 8);
    expect(result.renderObservation).toMatchObject({ status: "observed", roundTripMs: 350 });
    if (result.renderObservation.status !== "observed") throw new Error("Missing render observation");
    expect(result.renderObservation.fixedAnchorResidualUpperMs).toBeCloseTo(350, 8);
  });
  it("keeps zero or failed output timestamps explicitly unavailable", () => {
    const f = fixture(); f.context.getOutputTimestamp = () => ({ contextTime: 0, performanceTime: 0 });
    const request = f.tracker.begin(f.context, 10)!;
    expect(f.tracker.receive(f.response(request.probeId))?.outputTimestamp).toEqual({ status: "unavailable", reason: "zero-or-invalid" });
    f.context.getOutputTimestamp = () => { throw new Error("unsupported device"); };
    const next = f.tracker.begin(f.context, 20)!;
    expect(f.tracker.receive(f.response(next.probeId))?.outputTimestamp).toEqual({ status: "unavailable", reason: "read-failed" });
  });
});
