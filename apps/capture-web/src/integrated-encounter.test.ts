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
  const capture = { start: vi.fn(async () => true), finish: vi.fn(async () => undefined), withdraw: vi.fn(), discard: vi.fn(), controller: { updateAuthorization: vi.fn() } };
  vi.mocked(createEmbeddedEncounter).mockReturnValue(capture as unknown as ReturnType<typeof createEmbeddedEncounter>);
  const client = { captureContext: vi.fn(async () => ({ measurementProtocolRef, binding: { platformParticipantId: "participant-track" }, consent: { permissions: { modalities: ["face"] } } })), ingestObservation: vi.fn(async () => ({})) } as unknown as EncounterClient;
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
    const starting = startIntegratedEncounter(f.options); f.event("ended");
    resolve({ measurementProtocolRef, binding: { platformParticipantId: "participant-track" }, consent: { permissions: { modalities: ["face"] } } });
    await starting; expect(createEmbeddedEncounter).not.toHaveBeenCalled();
  });
  it("suspends measurement on host call-quality pressure without finishing or retrying capture", async () => {
    vi.useFakeTimers(); const f = fixture(); await startIntegratedEncounter(f.options);
    f.event("resource-pressure"); await vi.advanceTimersByTimeAsync(60_000);
    expect(f.capture.withdraw).toHaveBeenCalledOnce(); expect(f.capture.finish).not.toHaveBeenCalled();
    expect(f.client.captureContext).toHaveBeenCalledOnce();
  });
});
