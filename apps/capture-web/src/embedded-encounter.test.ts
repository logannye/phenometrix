import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { DurableObservationV1 } from "@phenometrix/contracts";
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
    expect(pipelineOptions.taskContext).toBe("ambient-speech-turn"); expect(setNoiseFloor).toHaveBeenCalledWith(0.001);
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
    emitVoice(0, 30_000);
    await vi.advanceTimersByTimeAsync(271_010);
    await vi.waitFor(() => expect(test.observations).toHaveLength(1));
    expect(capture.controller.status).toBe("active");
    emitVoice(300_000, 330_000); vi.setSystemTime(EPOCH + 330_000);
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
    expect(events.filter(e => e.type === "accepted")).toHaveLength(1);
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
