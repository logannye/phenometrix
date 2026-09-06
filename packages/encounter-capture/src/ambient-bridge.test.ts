import { afterEach, describe, expect, it, vi } from "vitest";
import { AmbientEncounterWindow } from "./ambient-bridge.js";
import { RotatingAmbientEncounter } from "./rotating-window.js";
import { authorization, EPOCH, voiceFrame } from "./test-fixtures.js";
import type { AcceptedDerived } from "./controller.js";
import type { DurableObservationV1 } from "@phenometrix/contracts";

const options = () => ({ authorization: authorization(), observationId: "test-observation", revisionId: "test-revision",
  startedAtMs: EPOCH, adapterId: "synthetic-test", adapterVersion: "1", sourceKind: "synthetic" as const,
  deviceClass: "synthetic-test", clockUncertaintyMs: 1 });
afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks(); });
function accepted(tMs: number): AcceptedDerived {
  return { encounterId: "test-encounter", subjectRef: "test-patient", platformParticipantId: "123", captureEpoch: 1,
    continuityId: "test-1", modality: "voice", acquiredAtMs: EPOCH + tMs, batch: { voiceFrames: [voiceFrame(tMs)] } };
}

describe("concrete ambient-core bridge", () => {
  it("runs real deterministic extraction on synthetic derived primitives and erases them on finalization", async () => {
    const window = new AmbientEncounterWindow(options()); window.setCalibration({ noiseDurationMs: 2_000 });
    for (let t = 0; t < 30_000; t += 10) expect(window.append(accepted(t))).toBe(true);
    const observation = await window.finalize(EPOCH + 30_000);
    expect(observation.metrics.find(m => m.metricCode === "ambient.voice.f0.median")).toMatchObject({ status: "measured", value: 150 });
    expect(observation.capture.rawMediaRetained).toBe(false); expect(observation.windows.length).toBeGreaterThan(0);
    expect(window.frameCount).toBe(0); expect(window.append(accepted(31_000))).toBe(false);
    expect(JSON.stringify(observation)).not.toContain("voiceFrames");
    expect(observation.metrics.every(m => m.temporalSamples.length === 0)).toBe(true);
  });
  it("does not manufacture measurements when processors/calibration/data are absent", async () => {
    const observation = await new AmbientEncounterWindow(options()).finalize(EPOCH + 100);
    expect(observation.metrics).toHaveLength(27);
    expect(observation.metrics.every(m => m.status === "withheld" && m.value === null)).toBe(true);
  });
  it("cannot return a durable observation when withdrawal races asynchronous provenance hashing", async () => {
    const abort = new AbortController();
    const digest = crypto.subtle.digest.bind(crypto.subtle);
    vi.spyOn(crypto.subtle, "digest").mockImplementationOnce(async (...args) => {
      const value = await digest(...args); abort.abort(); return value;
    });
    const window = new AmbientEncounterWindow(options());
    await expect(window.finalize(EPOCH + 100, EPOCH + 100, abort.signal)).rejects.toThrow();
    expect(window.frameCount).toBe(0);
  });
  it("emits no voice metric or window when only facial analysis was authorized", async () => {
    const config = options(); config.authorization.modalities = ["face"];
    const window = new AmbientEncounterWindow(config);
    expect(window.append(accepted(0))).toBe(false);
    const observation = await window.finalize(EPOCH + 100);
    expect(observation.metrics).toHaveLength(20); expect(observation.metrics.every(m => m.modality === "face")).toBe(true);
    expect(observation.windows).toEqual([]);
  });
  it("bounds derived memory and rejects mismatched participant and clocks", () => {
    const window = new AmbientEncounterWindow({ ...options(), maxFrames: 1 });
    expect(window.append({ ...accepted(0), subjectRef: "wrong" })).toBe(false);
    const bad = accepted(0); bad.batch.voiceFrames![0].acquiredAtMs = EPOCH - 100;
    expect(window.append(bad)).toBe(false); expect(window.append(accepted(0))).toBe(true);
    expect(window.append(accepted(10))).toBe(false); expect(window.frameCount).toBe(1); window.discard();
  });
  it("rotates windows automatically during a longer visit and rebases the second window", async () => {
    const observations: DurableObservationV1[] = [];
    const capture = new RotatingAmbientEncounter({ ...options(), onObservation: o => { observations.push(o); } });
    capture.setCalibration({ noiseDurationMs: 2_000 });
    for (let t = 0; t < 30_000; t += 10) capture.append(accepted(t));
    capture.advance(EPOCH + 300_000);
    for (let t = 300_000; t < 330_000; t += 10) capture.append(accepted(t));
    const final = await capture.finish(EPOCH + 330_000);
    expect(observations).toHaveLength(2); expect(final.observationId).toBe("test-observation.window-1");
    expect(final.startedAt).toBe(new Date(EPOCH + 300_000).toISOString());
    expect(final.metrics.find(m => m.metricCode === "ambient.voice.f0.median")).toMatchObject({ status: "measured", value: 150 });
    expect(final.windows.every(w => w.endMs <= 30_000)).toBe(true);
  });
  it("keeps a long suspension as an observation gap rather than inventing intervening windows", async () => {
    const observations: DurableObservationV1[] = [];
    const capture = new RotatingAmbientEncounter({ ...options(), onObservation: o => { observations.push(o); } });
    capture.advance(EPOCH + 1_200_000); await capture.finish(EPOCH + 1_201_000);
    expect(observations).toHaveLength(2); expect(observations[1].startedAt).toBe(new Date(EPOCH + 1_200_000).toISOString());
  });
  it("surfaces a failed persistence callback without exposing primitive frames", async () => {
    let failures = 0;
    const capture = new RotatingAmbientEncounter({ ...options(), onObservation: () => { throw new Error("offline"); }, onFailure: () => { failures++; } });
    await expect(capture.finish(EPOCH + 1_000)).rejects.toThrow(); expect(failures).toBe(1);
  });
  it("does not invent a zero-duration record when ending exactly at the window boundary", async () => {
    const observations: DurableObservationV1[] = [];
    const capture = new RotatingAmbientEncounter({ ...options(), onObservation: o => { observations.push(o); } });
    capture.advance(EPOCH + 300_000); await capture.finish(EPOCH + 300_000);
    expect(observations).toHaveLength(1); expect(observations[0].observationId).toBe("test-observation");
  });
  it("keeps processing fingerprints stable across observation and encounter identifiers", async () => {
    const first = options(); const second = options(); second.authorization.encounterId = second.authorization.binding.encounterId = "other-encounter";
    second.observationId = "other-observation"; second.revisionId = "other-revision";
    const a = await new AmbientEncounterWindow(first).finalize(EPOCH + 100);
    const b = await new AmbientEncounterWindow(second).finalize(EPOCH + 100);
    expect(a.capture.processorFingerprint).toBe(b.capture.processorFingerprint);
  });
  it("suppresses not-yet-delivered observations after discard", async () => {
    const delivered = vi.fn();
    const capture = new RotatingAmbientEncounter({ ...options(), onObservation: delivered });
    capture.advance(EPOCH + 300_000); capture.discard();
    await new Promise(resolve => setTimeout(resolve, 20)); expect(delivered).not.toHaveBeenCalled();
  });
  it("aborts an in-flight host delivery after its timeout and refuses further capture", async () => {
    vi.useFakeTimers(); let signal: AbortSignal | null = null;
    const capture = new RotatingAmbientEncounter({ ...options(), deliveryTimeoutMs: 100,
      onObservation: (_o, value) => { signal = value; return new Promise(() => {}); } });
    capture.advance(EPOCH + 300_000);
    await vi.waitFor(() => expect(signal).not.toBeNull());
    await vi.advanceTimersByTimeAsync(101); expect(signal!.aborted).toBe(true);
    expect(capture.append(accepted(300_000))).toBe(false);
  });
  it("requires derived-retention permission at the durable boundary", () => {
    const config = options(); config.authorization.consent.permissions.derivedRetention = false;
    expect(() => new AmbientEncounterWindow(config)).toThrow("authorization");
  });
});
