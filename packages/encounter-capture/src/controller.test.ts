import { afterEach, describe, expect, it, vi } from "vitest";
import { EncounterCaptureController, type AcceptedDerived, type CaptureEvent } from "./controller.js";
import { EMBEDDED_LOCAL_CAPTURE_CAPABILITIES, EmbeddedLocalAdapter } from "./embedded-local.js";
import { ZOOM_RTMS_CAPTURE_CAPABILITIES, ZoomRtmsAdapter } from "./zoom-rtms.js";
import { transientBytes, transientFrame, type CaptureSourceSink, type MediaPacket, type MediaProcessor } from "./media.js";
import { authorization, EPOCH, voiceFrame } from "./test-fixtures.js";

afterEach(() => vi.useRealTimers());
function setup(processor?: MediaProcessor) {
  vi.useFakeTimers(); vi.setSystemTime(EPOCH);
  let sink!: CaptureSourceSink;
  const close = vi.fn();
  const factory = vi.fn(async (_context, next: CaptureSourceSink) => { sink = next; return { stop: close }; });
  const derived: AcceptedDerived[] = []; const events: CaptureEvent[] = [];
  const adapter = new EmbeddedLocalAdapter(factory);
  const controller = new EncounterCaptureController({ adapter, processorFactory: processor ? () => processor : undefined,
    onDerived: d => derived.push(d), onEvent: e => events.push(e) });
  return { controller, factory, close, derived, events, sink: () => sink };
}
function packet(overrides: Partial<MediaPacket> = {}): MediaPacket {
  return { encounterId: "test-encounter", platformParticipantId: "123", attribution: "individual-track",
    modality: "voice", trackId: "test-track", sequence: 1, acquiredAtMs: EPOCH + 10, format: "pcm-s16le",
    sampleRateHz: 48_000, channelCount: 1, media: transientBytes(new Uint8Array([1, 2, 3, 4])), ...overrides };
}
const flush = async () => { for (let n = 0; n < 8; n++) await Promise.resolve(); };

describe("host-driven encounter lifecycle (synthetic packets)", () => {
  it("rejects unsupported requested modalities before constructing processors or attaching sources", async () => {
    vi.useFakeTimers(); vi.setSystemTime(EPOCH);
    const factory = vi.fn(async () => ({ stop() {} }));
    const processorFactory = vi.fn();
    const events: CaptureEvent[] = [];
    const adapter = new EmbeddedLocalAdapter(factory, ["face"]);
    const controller = new EncounterCaptureController({ adapter, processorFactory, onDerived() {}, onEvent: event => events.push(event) });
    expect(await controller.start(authorization())).toBe(false);
    expect(factory).not.toHaveBeenCalled(); expect(processorFactory).not.toHaveBeenCalled();
    expect(events).toEqual([expect.objectContaining({ type: "withheld", reason: "modality-unavailable", modality: "voice" })]);
    expect(controller.status).toBe("idle");
    const faceOnly = authorization(); faceOnly.modalities = ["face"];
    expect(await controller.start(faceOnly)).toBe(true); expect(factory).toHaveBeenCalledOnce();
    controller.stop();
  });
  it("declares transport capabilities separately from technical processors and clinical qualification", () => {
    const local = new EmbeddedLocalAdapter(async () => ({ stop() {} }));
    const rtms = new ZoomRtmsAdapter({ meetingUuid: "test", streamId: "test", signalingUrl: "wss://example.zoom.us", signature: "0".repeat(64) });
    expect(local.capabilities).toMatchObject({ processingLocation: "patient-device", sourceKind: "patient-local-pre-codec",
      rawFormats: [], derivedBranch: true, requiresQualifiedProcessor: false, clinicallyQualifiedMetricCodes: [] });
    expect(rtms.capabilities).toEqual(ZOOM_RTMS_CAPTURE_CAPABILITIES);
    expect(rtms.capabilities).toMatchObject({ processingLocation: "deployment-server", sourceKind: "platform-patient-track",
      rawFormats: ["pcm-s16le", "jpeg"], derivedBranch: false, technicalMeasurementAvailability: "requires-qualified-processor",
      requiresQualifiedProcessor: true, clinicallyQualifiedMetricCodes: [] });
    for (const capabilities of [local.capabilities, rtms.capabilities]) {
      expect(Object.isFrozen(capabilities)).toBe(true);
      expect(Object.isFrozen(capabilities.modalities)).toBe(true);
      expect(Object.isFrozen(capabilities.rawFormats)).toBe(true);
      expect(Object.isFrozen(capabilities.clinicallyQualifiedMetricCodes)).toBe(true);
    }
  });
  it.each(["consent", "binding", "scope", "encounter", "modality"])("never attaches a source with invalid %s", async invalid => {
    const test = setup(); const auth = authorization();
    if (invalid === "consent") auth.consent.permissions.capture = false;
    if (invalid === "binding") auth.binding.status = "unverified";
    if (invalid === "scope") auth.consent.scope = { ...auth.consent.scope, participantId: "other" };
    if (invalid === "encounter") auth.encounterActive = false;
    if (invalid === "modality") auth.consent.permissions.modalities = ["face"];
    expect(await test.controller.start(auth)).toBe(false); expect(test.factory).not.toHaveBeenCalled();
  });
  it("keeps capturing while the clinician tab is hidden and emits actual derived events", async () => {
    const test = setup(); await test.controller.start(authorization()); test.controller.setPageVisibility(false);
    const { media: _media, format: _format, ...base } = packet();
    test.sink().derived({ ...base, batch: { voiceFrames: [voiceFrame()] } });
    expect(test.derived).toHaveLength(1); expect(test.controller.status).toBe("active");
    expect(test.events.at(-1)).toMatchObject({ type: "accepted", count: 1 }); test.controller.stop();
  });
  it.each(["mixed", "unknown"] as const)("disposes %s media without processing or using active-speaker guesses", async attribution => {
    const process = vi.fn(async () => ({ voiceFrames: [voiceFrame()] }));
    const test = setup({ supports: () => true, process, dispose() {} }); await test.controller.start(authorization());
    const bytes = new Uint8Array([1, 2]); test.sink().packet(packet({ attribution, media: transientBytes(bytes) }));
    await flush(); expect(process).not.toHaveBeenCalled(); expect([...bytes]).toEqual([0, 0]); test.controller.stop();
  });
  it.each(["wrong-participant", "wrong-encounter", "stale", "too-large", "unavailable"])("rejects %s packets", async kind => {
    const test = setup(); await test.controller.start(authorization());
    const p = packet();
    if (kind === "wrong-participant") p.platformParticipantId = "456";
    if (kind === "wrong-encounter") p.encounterId = "other";
    if (kind === "stale") p.acquiredAtMs = EPOCH - 1;
    if (kind === "too-large") p.media = transientBytes(new Uint8Array(4_194_305));
    if (kind === "unavailable") test.controller.setAvailability("voice", false);
    test.sink().packet(p); expect(p.media.released).toBe(true); expect(test.derived).toHaveLength(0); test.controller.stop();
  });
  it("permits voice while camera is off and rejects face outputs", async () => {
    const test = setup(); await test.controller.start(authorization()); test.sink().availability("face", false);
    const { media: _media, format: _format, ...base } = packet();
    test.sink().derived({ ...base, modality: "face", batch: {} });
    test.sink().derived({ ...base, batch: { voiceFrames: [voiceFrame()] } });
    expect(test.derived.map(d => d.modality)).toEqual(["voice"]); test.controller.stop();
  });
  it("bounds backpressure to one in-flight packet per lane and invalidates continuity", async () => {
    let resolve!: (result: { voiceFrames: ReturnType<typeof voiceFrame>[] }) => void;
    const test = setup({ supports: () => true, process: () => new Promise(r => { resolve = r; }), dispose() {} });
    await test.controller.start(authorization());
    const first = packet(); const second = packet({ sequence: 2, acquiredAtMs: EPOCH + 20 });
    test.sink().packet(first); await flush(); test.sink().packet(second);
    expect(test.controller.inFlightCount).toBe(1); expect(second.media.released).toBe(true);
    resolve({ voiceFrames: [voiceFrame()] }); await flush();
    expect(first.media.released).toBe(true); expect(test.derived).toHaveLength(0); test.controller.stop();
  });
  it.each(["withdraw", "end", "ambiguity"])("aborts, erases, and discards a late result after %s", async action => {
    let resolve!: (result: { voiceFrames: ReturnType<typeof voiceFrame>[] }) => void;
    const dispose = vi.fn();
    const test = setup({ supports: () => true, process: () => new Promise(r => { resolve = r; }), dispose });
    await test.controller.start(authorization()); const p = packet(); test.sink().packet(p); await flush();
    if (action === "withdraw") test.controller.withdraw();
    if (action === "end") test.controller.encounterEnded();
    if (action === "ambiguity") test.controller.updateAuthorization({ ...authorization(), binding: { ...authorization().binding, status: "unverified" } });
    expect(p.media.released).toBe(true); expect(dispose).toHaveBeenCalledTimes(1); expect(test.close).toHaveBeenCalledTimes(1);
    resolve({ voiceFrames: [voiceFrame()] }); await flush(); expect(test.derived).toHaveLength(0);
  });
  it("fails closed if a processor hangs and releases raw media", async () => {
    const dispose = vi.fn(); const test = setup({ supports: () => true, process: () => new Promise(() => {}), dispose });
    await test.controller.start(authorization()); const p = packet(); test.sink().packet(p); await flush();
    await vi.advanceTimersByTimeAsync(2_001); expect(test.controller.status).toBe("stopped"); expect(p.media.released).toBe(true);
    expect(dispose).toHaveBeenCalledOnce();
  });
  it("stops at consent expiry even if no new packets arrive", async () => {
    const test = setup(); const auth = authorization(); auth.consent.expiresAt = new Date(EPOCH + 50).toISOString();
    await test.controller.start(auth); await vi.advanceTimersByTimeAsync(51);
    expect(test.controller.status).toBe("stopped"); expect(test.close).toHaveBeenCalledOnce();
  });
  it("continues encounters longer than a measurement window", async () => {
    const test = setup(); await test.controller.start(authorization()); await vi.advanceTimersByTimeAsync(300_001);
    expect(test.controller.status).toBe("active"); test.controller.stop();
  });
  it("isolates throwing host factories, callbacks and teardown while remaining stopped", async () => {
    vi.useFakeTimers(); vi.setSystemTime(EPOCH);
    const factoryFailure = new EncounterCaptureController({ adapter: { adapterId: "test", adapterVersion: "1", capabilities: EMBEDDED_LOCAL_CAPTURE_CAPABILITIES, async start() {}, stop() { throw Error(); } },
      processorFactory() { throw Error(); }, onDerived() {} });
    expect(await factoryFailure.start(authorization())).toBe(false); expect(factoryFailure.status).toBe("stopped");
    let sink!: CaptureSourceSink;
    const dispose = vi.fn(() => { throw Error(); });
    const controller = new EncounterCaptureController({ adapter: { adapterId: "test", adapterVersion: "1",
      capabilities: EMBEDDED_LOCAL_CAPTURE_CAPABILITIES,
      async start(_context, next) { sink = next; }, stop() { throw Error(); } },
      processorFactory: () => ({ supports: () => true, async process() { return {}; }, dispose }),
      onDerived() { throw Error(); }, onEvent() { throw Error(); } });
    await controller.start(authorization());
    const { media: _media, format: _format, ...base } = packet();
    expect(() => sink.derived({ ...base, batch: { voiceFrames: [voiceFrame()] } })).not.toThrow();
    expect(controller.status).toBe("stopped"); expect(dispose).toHaveBeenCalledOnce(); expect(() => controller.stop()).not.toThrow();
  });
  it("disposes media when a processor capability check throws", async () => {
    const test = setup({ supports() { throw Error(); }, async process() { return {}; }, dispose() {} });
    await test.controller.start(authorization()); const p = packet();
    expect(() => test.sink().packet(p)).not.toThrow(); expect(p.media.released).toBe(true); test.controller.stop();
  });
  it("cleans up a branch that finishes constructing after withdrawal", async () => {
    vi.useFakeTimers(); vi.setSystemTime(EPOCH);
    let resolve!: (branch: { stop(): void }) => void; const stop = vi.fn();
    const adapter = new EmbeddedLocalAdapter(() => new Promise(r => { resolve = r; }));
    const controller = new EncounterCaptureController({ adapter, onDerived() {} });
    const start = controller.start(authorization()); controller.withdraw(); resolve({ stop });
    expect(await start).toBe(false); expect(stop).toHaveBeenCalledOnce();
  });
  it("closes native frames exactly once", () => {
    const close = vi.fn(); const lease = transientFrame({ close }, 100); lease.release(); lease.release();
    expect(close).toHaveBeenCalledOnce(); expect(lease.frame).toBeUndefined();
  });
});
