import { afterEach, describe, expect, it, vi } from "vitest";
import { requestedAudioCaptureSettings, VOICE_WORKER_MESSAGE_VERSION, VOICE_WORKLET_MESSAGE_VERSION,
  type VoiceWorkerRequest, type VoiceWorkerResponse } from "./voice-worker-protocol.js";

describe("actual DSP worker with simulated transport queues", () => {
  afterEach(() => { vi.unstubAllGlobals(); vi.resetModules(); });
  it("uses an atomic reference at startup and the first PCM block of a renewed epoch", async () => {
    const responses: VoiceWorkerResponse[] = [];
    let request!: (event: { data: VoiceWorkerRequest }) => void;
    let pcm!: (event: { data: unknown }) => void;
    vi.stubGlobal("self", { addEventListener: (_type: string, listener: typeof request) => { request = listener; },
      postMessage: (response: VoiceWorkerResponse) => responses.push(response), close() {} });
    await import("./voice-worker.js");
    const port = { addEventListener: (_type: string, listener: typeof pcm) => { pcm = listener; },
      start() {}, close() {}, postMessage() {} } as unknown as MessagePort;
    request({ data: { schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "initialize", captureEpoch: 1,
      port, sessionOriginPerformanceMs: 0, audioContextOriginSeconds: 0, taskContext: "ambient-speech-turn",
      noiseFloorRms: 0.001, captureSettings: requestedAudioCaptureSettings(48_000, 1,
        { echoCancellation: false, noiseSuppression: false, autoGainControl: false }) } });
    const sendPcm = (captureEpoch: number) => pcm({ data: { schemaVersion: VOICE_WORKLET_MESSAGE_VERSION,
      type: "pcm-block", captureEpoch, sequence: 1, absoluteSampleIndex: 0, acquisitionAudioTimeSeconds: 0.04,
      sampleRateHz: 48_000, channelCount: 1,
      buffer: Float32Array.from({ length: 1920 }, (_, i) => 0.05 * Math.sin(i * 2 * Math.PI * 150 / 48_000)).buffer } });
    sendPcm(1);
    const initial = responses.find(response => response.type === "signal-frame");
    expect(initial?.type).toBe("signal-frame");
    if (initial?.type !== "signal-frame") throw new Error("Missing actual DSP output");
    expect(initial.frame.snrDb).toBeCloseTo(20 * Math.log10(initial.frame.rms / 0.001), 8);
    request({ data: { schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "reset", captureEpoch: 2,
      taskContext: "ambient-speech-turn", noiseFloorRms: 0.01 } });
    sendPcm(1); // Late old-epoch PCM must not be interpreted under the new reference.
    sendPcm(2); // The PCM port can run immediately after the single reset message.
    const renewed = responses.filter(response => response.type === "signal-frame");
    expect(renewed).toHaveLength(2);
    const next = renewed[1];
    if (next.type !== "signal-frame") throw new Error("Missing renewed DSP output");
    expect(next.captureEpoch).toBe(2);
    expect(next.frame.snrDb).toBeCloseTo(20 * Math.log10(next.frame.rms / 0.01), 8);
    expect(initial.frame.snrDb - next.frame.snrDb).toBeCloseTo(20, 8);
  });
});
