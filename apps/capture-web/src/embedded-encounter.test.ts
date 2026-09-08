import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { DurableObservationV1, EncounterClockCalibrationV1 } from "@phenometrix/contracts";
import type { VoiceCaptureStartOptions } from "./voice-capture.js";
import type { VisualWorkerRequest, VisualWorkerFrameMessage } from "./face-worker-protocol.js";
import { syntheticFacialFrame } from "../../../packages/ambient-core/src/test-helpers.js";
import { createEmbeddedEncounter } from "./embedded-encounter.js";
import { authorization, EPOCH, voiceFrame } from "../../../packages/encounter-capture/src/test-fixtures.js";

const mocked = vi.hoisted(() => ({ startVoice: vi.fn(), loadVoice: vi.fn(), loadFace: vi.fn() }));
vi.mock("./voice-capture.js", () => ({ startVoiceCapturePipeline: mocked.startVoice }));
vi.mock("./static-assets.js", () => ({ loadAndVerifyVoiceStaticAssets: mocked.loadVoice, loadAndVerifyFaceStaticAssets: mocked.loadFace }));
vi.mock("./ambient-finalization.js", async () => {
  const { finalizeAmbientMetrics } = await import("@phenometrix/ambient-core");
  return { finalizeMetricsInWorker: async (input: Parameters<typeof finalizeAmbientMetrics>[0]) => finalizeAmbientMetrics(input) };
});

class HostTrack extends EventTarget {
  readyState = "live";
  enabled = true;
  muted = false;
  readonly stop = vi.fn();
  constructor(readonly kind: "audio" | "video", private settings: MediaTrackSettings = {}) { super(); }
  getSettings() { return this.settings; }
}
class HostStream {
  constructor(readonly tracks: HostTrack[]) {}
  getAudioTracks() { return this.tracks.filter(t => t.kind === "audio"); }
  getVideoTracks() { return this.tracks.filter(t => t.kind === "video"); }
}
let pipelineOptions: VoiceCaptureStartOptions;
const stop = vi.fn(async () => {});
const reset = vi.fn();
const setNoiseFloor = vi.fn();

beforeEach(() => {
  vi.useFakeTimers({ toFake: ["Date", "setTimeout", "clearTimeout", "setInterval", "clearInterval"] });
  vi.setSystemTime(EPOCH);
  vi.spyOn(performance, "now").mockImplementation(() => Date.now() - EPOCH);
  vi.stubGlobal("MediaStream", HostStream);
  vi.stubGlobal("navigator", { mediaDevices: { getUserMedia: vi.fn(() => { throw new Error("Device prompts forbidden"); }) } });
  mocked.startVoice.mockReset(); stop.mockClear(); reset.mockClear(); setNoiseFloor.mockClear();
  mocked.loadVoice.mockReset().mockResolvedValue({ manifest: { assets: { voiceWorklet: { sha256: "a".repeat(64) } } },
    voiceWorkletUrl: "/verified-voice-worklet.js" });
  mocked.loadFace.mockReset().mockResolvedValue({ manifest: { assets: { faceModel: {
    sha256: "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff" } } },
    faceModelUrl: "/models/face_landmarker.task", mediaPipeRootUrl: "/mediapipe" });
  mocked.startVoice.mockImplementation(async (options: VoiceCaptureStartOptions) => {
    pipelineOptions = options;
    return { stop, reset, setNoiseFloor, setTask() {}, captureEpoch: 1 };
  });
});
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.useRealTimers(); });

function setup(noise = true) {
  const track = new HostTrack("audio", { deviceId: "opaque-test-microphone", sampleRate: 48_000, channelCount: 1,
    echoCancellation: false, noiseSuppression: false, autoGainControl: false });
  const stream = new HostStream([track]);
  const audioContext = { state: "running", sampleRate: 48_000, close: vi.fn() };
  const auth = authorization(); auth.modalities = ["voice"];
  const observations: DurableObservationV1[] = [];
  const options = { authorization: auth, stream: stream as unknown as MediaStream,
    audioContext: audioContext as unknown as AudioContext, observationId: "embedded-test-observation", revisionId: "embedded-test-revision",
    ...(noise ? { noiseCalibration: { noiseFloorRms: 0.001, durationMs: 2_000 } } : {}),
    onObservation: (o: DurableObservationV1) => { observations.push(o); } };
  return { track, stream, audioContext, observations, options };
}
function emitVoice(from: number, until: number, captureEpoch = 1) {
  for (let t = from; t < until; t += 10) {
    vi.setSystemTime(EPOCH + t);
    pipelineOptions.callbacks.onFrame({ ...voiceFrame(t), acquiredAtMs: t, captureEpoch }, 1);
  }
}
function clockCalibration(localMeasuredAtMs = EPOCH, utcOffsetMs = 0, uncertaintyMs = 10): EncounterClockCalibrationV1 {
  return { localMeasuredAtMs, localExpiresAtMs: localMeasuredAtMs + 60_000, utcOffsetMs, uncertaintyMs,
    sourceId: "test-clock", sourceKind: "monitored-utc", measuredRoundTripMs: 10 };
}
function emitPassiveReference(start = 0, captureEpoch = 1) {
  for (let t = start; t < start + 2_500; t += 10) {
    const contrast = t - start < 500;
    vi.setSystemTime(EPOCH + t);
    pipelineOptions.callbacks.onFrame({ ...voiceFrame(t), acquiredAtMs: t, captureEpoch,
      spectralFlux: 0.005, speechActive: true, snrDb: 50,
      ...(contrast ? { rms: 0.04, f0Hz: 150, f0Confidence: 0.95, estimatorAgreement: 0.95, cepstralPeakProminenceDb: 12 }
        : { rms: 0.001, f0Hz: null, f0Confidence: 0.1, estimatorAgreement: 0.1, cepstralPeakProminenceDb: 2 }) }, 1);
  }
}

describe("embedded host bridge with simulated worker outputs (no live microphone/model claims)", () => {
  it("does not attach workers or inspect tracks before valid authorization", async () => {
    const test = setup(); test.options.authorization.consent.permissions.capture = false;
    const inspect = vi.spyOn(test.stream, "getAudioTracks");
    const capture = createEmbeddedEncounter(test.options);
    expect(await capture.start()).toBe(false); expect(inspect).not.toHaveBeenCalled();
    expect(mocked.startVoice).not.toHaveBeenCalled(); expect(navigator.mediaDevices.getUserMedia).not.toHaveBeenCalled();
    expect(mocked.loadVoice).not.toHaveBeenCalled(); expect(mocked.loadFace).not.toHaveBeenCalled();
  });
  it("connects the host track to the existing voice pipeline and runs real ambient extraction", async () => {
    const test = setup(); const capture = createEmbeddedEncounter(test.options);
    expect(await capture.start()).toBe(true);
    expect(pipelineOptions.stream.getAudioTracks()[0]).toBe(test.track);
    expect(pipelineOptions.audioContext).toBe(test.audioContext);
    expect(pipelineOptions.workletUrl).toBe("/verified-voice-worklet.js");
    expect(pipelineOptions.taskContext).toBe("ambient-speech-turn"); expect(pipelineOptions.noiseFloorRms).toBe(0.001);
    emitVoice(0, 30_000); vi.setSystemTime(EPOCH + 30_000);
    const result = await capture.finish();
    expect(result.metrics).toHaveLength(7); expect(result.metrics.every(m => m.modality === "voice")).toBe(true);
    expect(result.metrics.find(m => m.metricCode === "ambient.voice.f0.median")).toMatchObject({ status: "measured", value: 150 });
    expect(test.observations).toEqual([result]); expect(result.capture.sourceKind).toBe("patient-local-pre-codec");
    expect(result.capture.clockUncertaintyMs).toBe(Number.MAX_SAFE_INTEGER);
    expect(JSON.stringify(result)).not.toContain("opaque-test-microphone");
    expect(stop).toHaveBeenCalledOnce(); expect(test.track.stop).not.toHaveBeenCalled(); expect(test.audioContext.close).not.toHaveBeenCalled();
    expect(navigator.mediaDevices.getUserMedia).not.toHaveBeenCalled();
  });
  it("automatically delivers bounded windows throughout a longer encounter, including a hidden tab", async () => {
    const test = setup(); const capture = createEmbeddedEncounter(test.options);
    await capture.start(); capture.controller.setPageVisibility(false);
    emitVoice(0, 330_000);
    await vi.waitFor(() => expect(test.observations).toHaveLength(1));
    expect(capture.controller.status).toBe("active");
    vi.setSystemTime(EPOCH + 330_000);
    const result = await capture.finish();
    expect(test.observations).toHaveLength(2); expect(result.startedAt).toBe(new Date(EPOCH + 300_000).toISOString());
    expect(result.windows.every(w => w.endMs <= 30_000)).toBe(true);
    expect(result.metrics.find(m => m.metricCode === "ambient.voice.f0.median")).toMatchObject({ status: "measured", value: 150 });
  });
  it("abstains without a measured noise calibration instead of inventing a baseline", async () => {
    const test = setup(false); const capture = createEmbeddedEncounter(test.options); await capture.start();
    emitVoice(0, 30_000); vi.setSystemTime(EPOCH + 30_000);
    const result = await capture.finish(); expect(result.metrics.every(m => m.status === "withheld")).toBe(true);
    expect(setNoiseFloor).not.toHaveBeenCalled();
  });
  it("becomes ready from passively screened scalar features without keeping pre-calibration measurements", async () => {
    const test = setup(false); const capture = createEmbeddedEncounter(test.options); await capture.start();
    emitPassiveReference(); expect(setNoiseFloor).not.toHaveBeenCalled();
    expect(reset).toHaveBeenCalledWith(2, "ambient-speech-turn", 0.001);
    emitVoice(2_500, 32_500, 2); vi.setSystemTime(EPOCH + 32_500);
    const result = await capture.finish();
    expect(result.metrics.find(m => m.metricCode === "ambient.voice.f0.median")).toMatchObject({ status: "measured", value: 150 });
    expect(result.capture.audioNoiseCalibration).toEqual({ method: "passive-screened", algorithmVersion: "passive-screened-noise@1.0.0", qualification: "engineering-only" });
    expect(result.metrics.find(m => m.status === "measured")?.processorRef).toContain("passive-screened-noise@1.0.0:engineering-only");
    expect(result.windows.every(window => window.startMs >= 2_500)).toBe(true);
    expect(navigator.mediaDevices.getUserMedia).not.toHaveBeenCalled();
  });
  it("invalidates an existing noise reference when mute or acquisition loss breaks continuity", async () => {
    const test = setup(); const capture = createEmbeddedEncounter(test.options); await capture.start();
    emitVoice(0, 40_000); vi.setSystemTime(EPOCH + 40_000);
    test.track.muted = true; test.track.dispatchEvent(new Event("mute"));
    test.track.muted = false; test.track.dispatchEvent(new Event("unmute"));
    emitVoice(40_000, 70_000, 3); vi.setSystemTime(EPOCH + 70_000);
    const result = await capture.finish(); expect(result.metrics.every(m => m.status === "withheld")).toBe(true);
    expect(result.capture.audioNoiseCalibration).toBeUndefined();
    expect(test.observations).toHaveLength(2);
    const first = test.observations.find(observation => observation.observationId === test.options.observationId)!;
    expect(first.endedAt).toBe(result.startedAt);
    expect(first.metrics.find(metric => metric.metricCode === "ambient.voice.f0.median"))
      .toMatchObject({ status: "measured", value: 150 });
    expect(first.capture.audioNoiseCalibration?.method).toBe("host-supplied");
  });
  it("renews the same UTC source without changing acquisition timestamps or reducing prior uncertainty", async () => {
    const test = setup(); const capture = createEmbeddedEncounter({ ...test.options, clockCalibration: clockCalibration() });
    await capture.start(); emitVoice(0, 30_000); vi.setSystemTime(EPOCH + 30_000);
    expect(capture.updateClockCalibration(clockCalibration(EPOCH + 30_000, 15))).toBe(true);
    emitVoice(30_000, 70_000); vi.setSystemTime(EPOCH + 70_000);
    expect(capture.controller.status).toBe("active");
    expect(capture.updateClockCalibration(clockCalibration(EPOCH + 70_000, 14, 2))).toBe(true);
    const result = await capture.finish();
    expect(result.endedAt).toBe(new Date(EPOCH + 70_000).toISOString());
    expect(result.capture.clockUncertaintyMs).toBe(27);
    expect(result.capture.clockSource).toEqual({ sourceId: "test-clock", kind: "monitored-utc" });
    expect(result.metrics.find(m => m.metricCode === "ambient.voice.f0.median")?.status).toBe("measured");
  });
  it.each(["source", "kind", "step", "stale", "too-long"])("fails closed on a %s clock renewal", async kind => {
    const test = setup(); const capture = createEmbeddedEncounter({ ...test.options, clockCalibration: clockCalibration() });
    await capture.start(); vi.setSystemTime(EPOCH + 100);
    const next = clockCalibration(EPOCH + 100);
    if (kind === "source") next.sourceId = "other-source";
    if (kind === "kind") next.sourceKind = "synthetic";
    if (kind === "step") next.utcOffsetMs = 50;
    if (kind === "stale") next.localMeasuredAtMs = EPOCH;
    if (kind === "too-long") next.localExpiresAtMs += 1;
    expect(capture.updateClockCalibration(next)).toBe(false); expect(capture.controller.status).toBe("stopped");
    await expect(capture.finish()).rejects.toThrow("discarded");
  });
  it("expires an unrenewed clock without packets and does not revive expired capture", async () => {
    const test = setup(); const capture = createEmbeddedEncounter({ ...test.options, clockCalibration: clockCalibration() });
    await capture.start(); await vi.advanceTimersByTimeAsync(60_001);
    expect(capture.controller.status).toBe("stopped"); expect(stop).toHaveBeenCalledOnce();
    expect(capture.updateClockCalibration(clockCalibration(EPOCH + 60_001))).toBe(false);
    await expect(capture.finish()).rejects.toThrow("discarded");
  });
  it.each(["packet", "finish"])("fails closed at %s when a throttled expiry timer has not run", async boundary => {
    const test = setup(); const capture = createEmbeddedEncounter({ ...test.options, clockCalibration: clockCalibration() });
    await capture.start(); vi.setSystemTime(EPOCH + 60_001);
    if (boundary === "packet") pipelineOptions.callbacks.onFrame({ ...voiceFrame(60_001), acquiredAtMs: 60_001 }, 1);
    await expect(capture.finish()).rejects.toThrow("discarded");
    expect(capture.controller.status).toBe("stopped"); expect(test.observations).toEqual([]);
  });
  it("cannot adopt a new nonzero UTC offset after starting an unqualified encounter", async () => {
    const test = setup(); const capture = createEmbeddedEncounter(test.options); await capture.start();
    vi.setSystemTime(EPOCH + 100);
    expect(capture.updateClockCalibration(clockCalibration(EPOCH + 100, 1_000))).toBe(false);
    expect(capture.controller.status).toBe("stopped");
  });
  it("invalidates worker epochs on mute and ignores late callbacks after withdrawal", async () => {
    const test = setup(); const capture = createEmbeddedEncounter(test.options); await capture.start();
    test.track.muted = true; test.track.dispatchEvent(new Event("mute"));
    expect(reset).toHaveBeenCalledWith(2, "ambient-speech-turn");
    test.track.muted = false; test.track.dispatchEvent(new Event("unmute"));
    expect(reset).toHaveBeenCalledWith(3, "ambient-speech-turn");
    capture.withdraw(); emitVoice(0, 100);
    expect(capture.controller.status).toBe("stopped"); expect(test.observations).toEqual([]);
    expect(stop).toHaveBeenCalledOnce(); expect(test.track.stop).not.toHaveBeenCalled();
    await expect(capture.finish()).rejects.toThrow("discarded");
  });
  it("keeps voice running when the authorized camera modality is unavailable", async () => {
    const test = setup(); test.options.authorization.modalities = ["face", "voice"];
    const capture = createEmbeddedEncounter(test.options); expect(await capture.start()).toBe(true);
    emitVoice(0, 30_000); vi.setSystemTime(EPOCH + 30_000); const result = await capture.finish();
    expect(result.metrics.filter(m => m.modality === "face").every(m => m.status === "withheld")).toBe(true);
    expect(result.metrics.find(m => m.metricCode === "ambient.voice.f0.median")?.status).toBe("measured");
  });
  it("connects host video to the real frame pump and worker protocol, rejects multiple faces and stale post-mute results", async () => {
    const test = setup(); test.options.authorization.modalities = ["face"];
    const track = new HostTrack("video", { width: 1280, height: 720, frameRate: 30 }); test.stream.tracks.push(track);
    let onPresented!: (now: number, metadata: Record<string, number>) => void;
    const cancelVideoFrameCallback = vi.fn();
    const video = { videoWidth: 1280, videoHeight: 720, srcObject: test.stream,
      requestVideoFrameCallback(callback: typeof onPresented) { onPresented = callback; return 1; }, cancelVideoFrameCallback };
    const close = vi.fn(); vi.stubGlobal("createImageBitmap", vi.fn(async () => ({ close })));
    let worker!: SimulatedWorker;
    class SimulatedWorker {
      onmessage: ((event: { data: unknown }) => void) | null = null;
      onerror = null;
      readonly requests: VisualWorkerRequest[] = [];
      readonly terminate = vi.fn();
      constructor() { worker = this; }
      postMessage(message: VisualWorkerRequest) {
        this.requests.push(message);
        if (message.type === "initialize") queueMicrotask(() => this.onmessage?.({ data: {
          type: "ready", captureEpoch: message.captureEpoch } }));
      }
      reply(count: number) {
        const request = this.requests.at(-1) as VisualWorkerFrameMessage;
        // Test worker consumes transferred bitmap ownership, as production face-worker does in finally.
        request.bitmap.close();
        this.onmessage?.({ data: { type: "frame", captureEpoch: request.captureEpoch, sequence: request.sequence,
          acquiredAtMs: request.acquiredAtMs, faceCount: count,
          frame: syntheticFacialFrame(request.tMs, "ambient-frontal", { acquiredAtMs: request.acquiredAtMs }) } });
      }
    }
    vi.stubGlobal("Worker", SimulatedWorker);
    const events: { type: string; reason?: string }[] = [];
    const capture = createEmbeddedEncounter({ ...test.options, video: video as unknown as HTMLVideoElement, onEvent: e => events.push(e) });
    expect(await capture.start()).toBe(true);
    expect(worker.requests[0]).toMatchObject({ type: "initialize", assets: { modelUrl: "/models/face_landmarker.task" } });
    expect(mocked.startVoice).not.toHaveBeenCalled();
    const present = async (t: number) => {
      vi.setSystemTime(EPOCH + t); onPresented(t, { presentationTime: t, width: 1280, height: 720 });
      for (let n = 0; n < 5; n++) await Promise.resolve();
    };
    await present(100); worker.reply(2);
    expect(events.at(-1)).toMatchObject({ type: "withheld", reason: "multiple-faces" });
    await present(200); vi.setSystemTime(EPOCH + 201); track.muted = true; track.dispatchEvent(new Event("mute"));
    vi.setSystemTime(EPOCH + 202); track.muted = false; track.dispatchEvent(new Event("unmute")); worker.reply(1);
    expect(events.filter(e => e.type === "accepted")).toHaveLength(0);
    await present(300); worker.reply(1);
    expect(events.filter(e => e.type === "accepted")).toHaveLength(0); // Baseline acquisition is not measurement evidence.
    video.srcObject = new HostStream([new HostTrack("video")]); await present(400);
    expect(capture.controller.status).toBe("stopped"); await expect(capture.finish()).rejects.toThrow("discarded");
    expect(worker.terminate).toHaveBeenCalledOnce(); expect(cancelVideoFrameCallback).toHaveBeenCalled();
    expect(track.stop).not.toHaveBeenCalled(); expect(close).toHaveBeenCalledTimes(3);
  });
  it("discards pending windows on automatic consent expiry or host authorization loss", async () => {
    const test = setup(); test.options.authorization.consent.expiresAt = new Date(EPOCH + 50).toISOString();
    const capture = createEmbeddedEncounter(test.options); await capture.start();
    await vi.advanceTimersByTimeAsync(51); expect(capture.controller.status).toBe("stopped");
    await expect(capture.finish()).rejects.toThrow("discarded"); expect(test.observations).toEqual([]);
  });
  it("does not create a durable capture branch without retention permission", async () => {
    const test = setup(); test.options.authorization.consent.permissions.derivedRetention = false;
    const capture = createEmbeddedEncounter(test.options); expect(await capture.start()).toBe(false);
    expect(mocked.startVoice).not.toHaveBeenCalled(); expect(test.observations).toEqual([]);
  });
  it("abstains from a lane whose verified static assets are unavailable", async () => {
    mocked.loadVoice.mockRejectedValue(new Error("asset-integrity-failed"));
    const test = setup(); const capture = createEmbeddedEncounter(test.options); await capture.start();
    expect(mocked.startVoice).not.toHaveBeenCalled(); vi.setSystemTime(EPOCH + 100);
    expect((await capture.finish()).metrics.every(m => m.status === "withheld")).toBe(true);
  });
  it("stops and discards when acquisition settings change instead of retaining stale provenance", async () => {
    const test = setup(); const capture = createEmbeddedEncounter(test.options); await capture.start();
    test.track.getSettings().sampleRate = 44_100; emitVoice(0, 10);
    expect(capture.controller.status).toBe("stopped"); await expect(capture.finish()).rejects.toThrow("discarded");
  });
});
