import { describe, it, expect, vi, afterEach } from "vitest";
import type { EncounterClient } from "@phenometrix/encounter-client";
import type { IntegratedEncounterOptions, HostEncounterEvent } from "./integrated-encounter.js";
import { startIntegratedEncounter } from "./integrated-encounter.js";
import { createEmbeddedEncounter } from "./embedded-encounter.js";
import { AMBIENT_LOCAL_PROTOCOL_REF } from "@phenometrix/contracts";

const measurementProtocolRef = { id: AMBIENT_LOCAL_PROTOCOL_REF.packId, version: AMBIENT_LOCAL_PROTOCOL_REF.version, contentSha256: AMBIENT_LOCAL_PROTOCOL_REF.contentSha256 };

vi.mock("./embedded-encounter.js", () => ({ createEmbeddedEncounter: vi.fn() }));
afterEach(() => { vi.useRealTimers(); vi.clearAllMocks(); });
function fixture() {
  let event: (value: HostEncounterEvent) => void = () => {};
  const capture = { start: vi.fn(async () => true), finish: vi.fn(async () => undefined), withdraw: vi.fn(), discard: vi.fn(), updateClockCalibration: vi.fn(() => true), controller: { updateAuthorization: vi.fn() } };
  vi.mocked(createEmbeddedEncounter).mockReturnValue(capture as unknown as ReturnType<typeof createEmbeddedEncounter>);
  const clock = { localMeasuredAtMs: Date.now(), localExpiresAtMs: Date.now() + 60_000, utcOffsetMs: 0, uncertaintyMs: 10, sourceId: "utc-monitor", sourceKind: "monitored-utc" as const, measuredRoundTripMs: 10 };
  const client = { calibrateClock: vi.fn(async () => clock), captureContext: vi.fn(async () => ({ measurementProtocolRef, binding: { platformParticipantId: "participant-track" }, consent: { permissions: { modalities: ["face"] } } })), ingestObservation: vi.fn(async () => ({})) } as unknown as EncounterClient;
  const statuses: string[] = [];
  const options: IntegratedEncounterOptions = { client, encounterId: "encounter", platformParticipantId: "participant-track", stream: {} as MediaStream,
    subscribeLifecycle: listener => { event = listener; return vi.fn(); }, onStatus: status => statuses.push(status) };
  return { options, capture, client, statuses, event: (value: HostEncounterEvent) => event(value) };
}
describe("normal appointment composition", () => {
  it("starts from the host event and delivers observations without a new patient action", async () => {
    const f = fixture(); const handle = await startIntegratedEncounter(f.options);
    expect(f.client.captureContext).toHaveBeenCalledOnce(); expect(f.capture.start).toHaveBeenCalledOnce();
    const options = vi.mocked(createEmbeddedEncounter).mock.calls[0][0];
    await options.onObservation?.({ revisionId: "synthetic" } as never, new AbortController().signal);
    expect(f.client.ingestObservation).toHaveBeenCalledOnce();
    f.event("ended"); expect(f.capture.finish).toHaveBeenCalledOnce(); expect(f.capture.withdraw).not.toHaveBeenCalled(); handle.stop();
  });
  it("never starts when the host stream does not match the enrolled participant", async () => {
    const f = fixture(); await startIntegratedEncounter({ ...f.options, platformParticipantId: "caregiver-track" });
    expect(createEmbeddedEncounter).not.toHaveBeenCalled(); expect(f.statuses).toContain("unavailable");
  });
  it("abstains before attaching workers when the encounter requires an unsupported measurement protocol", async () => {
    const f = fixture();
    const context = await f.client.captureContext("encounter");
    vi.mocked(f.client.captureContext).mockResolvedValue({ ...context, measurementProtocolRef: { ...measurementProtocolRef, version: "unsupported" } });
    await startIntegratedEncounter(f.options);
    expect(createEmbeddedEncounter).not.toHaveBeenCalled(); expect(f.statuses).toContain("unavailable");
  });
  it("ends authorization refresh and capture on withdrawal", async () => {
    vi.useFakeTimers(); const f = fixture(); await startIntegratedEncounter(f.options);
    f.event("consent-withdrawn"); await vi.advanceTimersByTimeAsync(60_000);
    expect(f.capture.withdraw).toHaveBeenCalledOnce(); expect(f.client.captureContext).toHaveBeenCalledOnce(); expect(f.capture.finish).not.toHaveBeenCalled();
  });
  it("does not start after the appointment ends while context is loading", async () => {
    const f = fixture(); let resolve!: (value: unknown) => void;
    vi.mocked(f.client.captureContext).mockImplementationOnce(() => new Promise(done => { resolve = done as typeof resolve; }));
    const starting = startIntegratedEncounter(f.options); await vi.waitFor(() => expect(resolve).toBeTypeOf("function")); f.event("ended");
    resolve({ measurementProtocolRef, binding: { platformParticipantId: "participant-track" }, consent: { permissions: { modalities: ["face"] } } });
    await starting; expect(createEmbeddedEncounter).not.toHaveBeenCalled();
  });
  it("suspends measurement on host call-quality pressure without finishing or retrying capture", async () => {
    vi.useFakeTimers(); const f = fixture(); await startIntegratedEncounter(f.options);
    f.event("resource-pressure"); await vi.advanceTimersByTimeAsync(60_000);
    expect(f.capture.withdraw).toHaveBeenCalledOnce(); expect(f.capture.finish).not.toHaveBeenCalled();
    expect(f.client.captureContext).toHaveBeenCalledOnce();
  });
  it("renews clock and authorization automatically and propagates capture failure immediately", async () => {
    vi.useFakeTimers(); const f = fixture(); await startIntegratedEncounter(f.options);
    await vi.advanceTimersByTimeAsync(20_000);
    expect(f.client.calibrateClock).toHaveBeenCalledTimes(2); expect(f.capture.updateClockCalibration).toHaveBeenCalledOnce();
    const options = vi.mocked(createEmbeddedEncounter).mock.calls[0][0];
    options.onEvent?.({ type: "stopped", reason: "processor-failed" } as never);
    await vi.advanceTimersByTimeAsync(60_000);
    expect(f.capture.withdraw).toHaveBeenCalledOnce(); expect(f.client.calibrateClock).toHaveBeenCalledTimes(2);
  });
  it("renews a nearly expired join-time clock before its deadline", async () => {
    vi.useFakeTimers(); const f = fixture();
    const clock = await f.client.calibrateClock("encounter");
    vi.mocked(f.client.calibrateClock).mockClear().mockResolvedValueOnce({ ...clock!, localExpiresAtMs: Date.now() + 5_000 });
    await startIntegratedEncounter(f.options);
    await vi.advanceTimersByTimeAsync(2_500);
    expect(f.client.calibrateClock).toHaveBeenCalledTimes(2); expect(f.capture.updateClockCalibration).toHaveBeenCalledOnce();
    f.event("binding-lost");
  });
  it("does not silently adopt a new timestamp origin after starting without clock qualification", async () => {
    vi.useFakeTimers(); const f = fixture();
    vi.mocked(f.client.calibrateClock).mockResolvedValueOnce(null);
    await startIntegratedEncounter(f.options); await vi.advanceTimersByTimeAsync(20_000);
    expect(vi.mocked(createEmbeddedEncounter).mock.calls[0][0].clockCalibration).toBeUndefined();
    expect(f.capture.updateClockCalibration).not.toHaveBeenCalled(); f.event("binding-lost");
  });
  it("retries the same observation after a transient failure and propagates exhausted delivery", async () => {
    vi.useFakeTimers(); const f = fixture(); await startIntegratedEncounter(f.options);
    const options = vi.mocked(createEmbeddedEncounter).mock.calls[0][0], observation = { revisionId: "same-revision" } as never;
    vi.mocked(f.client.ingestObservation).mockRejectedValueOnce(new Error("encounter-request-503"));
    const retry = options.onObservation!(observation, new AbortController().signal);
    await vi.advanceTimersByTimeAsync(250); await retry;
    expect(f.client.ingestObservation).toHaveBeenCalledTimes(2);
    expect(vi.mocked(f.client.ingestObservation).mock.calls.every(call => call[0] === observation)).toBe(true);
    vi.mocked(f.client.ingestObservation).mockRejectedValue(new Error("encounter-request-503"));
    const failed = expect(options.onObservation!(observation, new AbortController().signal)).rejects.toThrow("503");
    await vi.advanceTimersByTimeAsync(250); await failed;
    expect(f.statuses.filter(status => status === "delivered")).toHaveLength(1);
    expect(f.statuses).toContain("delivery-failed"); f.event("binding-lost");
  });
  it("does not retry rejected authority or acknowledge a canceled delivery", async () => {
    const f = fixture(); await startIntegratedEncounter(f.options);
    const options = vi.mocked(createEmbeddedEncounter).mock.calls[0][0];
    vi.mocked(f.client.ingestObservation).mockRejectedValue(new Error("encounter-request-403"));
    await expect(options.onObservation!({} as never, new AbortController().signal)).rejects.toThrow("403");
    expect(f.client.ingestObservation).toHaveBeenCalledOnce(); expect(f.capture.withdraw).toHaveBeenCalledOnce();
    expect(f.statuses).not.toContain("delivered");
  });
  it("retains withdrawal handling while the final observation is being delivered", async () => {
    const f = fixture(); let finish!: () => void;
    f.capture.finish.mockImplementation(() => new Promise(resolve => { finish = resolve as () => void; }));
    await startIntegratedEncounter(f.options); f.event("ended"); f.event("consent-withdrawn");
    expect(f.capture.withdraw).toHaveBeenCalledOnce(); finish();
  });
});
