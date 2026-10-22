import { afterEach, describe, expect, it, vi } from "vitest";
import { startVoiceCapturePipeline } from "./voice-capture.js";

describe("startVoiceCapturePipeline", () => {
  afterEach(() => vi.unstubAllGlobals());

  function callbacks(onFailure = vi.fn()) {
    return {
      onFrame() {},
      onReady() {},
      onDiagnostics() {},
      onFailure
    };
  }

  function messageChannelStub(): void {
    class MessagePortMock {
      readonly postMessage = vi.fn();
      readonly close = vi.fn();
    }
    vi.stubGlobal("MessageChannel", class {
      readonly port1 = new MessagePortMock();
      readonly port2 = new MessagePortMock();
    });
  }

  function workerStub(): {
    terminate: ReturnType<typeof vi.fn>;
    sent: Record<string, unknown>[];
  } {
    const sent: Record<string, unknown>[] = [];
    const terminate = vi.fn();
    class WorkerMock extends EventTarget {
      postMessage(message: { type?: string; captureEpoch?: number }): void {
        sent.push(message);
        if (message.type !== "dispose") return;
        const event = new Event("message") as MessageEvent;
        Object.defineProperty(event, "data", {
          value: {
            schemaVersion: "phenometric.voice-worker-message.v2",
            type: "disposed",
            captureEpoch: message.captureEpoch,
            diagnostics: {
              receivedBlockCount: 0,
              droppedBlockCount: 0,
              processedWindowCount: 0,
              emittedFrameCount: 0,
              maximumRingBufferSamples: 96_000
            }
          }
        });
        this.dispatchEvent(event);
      }

      terminate(): void {
        terminate();
      }
    }
    vi.stubGlobal("Worker", WorkerMock);
    return { terminate, sent };
  }

  function audioNode(disconnect = vi.fn()): AudioNode {
    return {
      connect() {
        return this;
      },
      disconnect
    } as unknown as AudioNode;
  }

  it("does not create processor resources when startup is cancelled during addModule", async () => {
    let resolveAddModule!: () => void;
    const addModule = vi.fn(() => new Promise<void>((resolve) => {
      resolveAddModule = resolve;
    }));
    const workerConstructor = vi.fn();
    vi.stubGlobal("Worker", workerConstructor);
    const startup = new AbortController();

    const pipeline = startVoiceCapturePipeline({
      stream: {} as MediaStream,
      audioContext: {
        audioWorklet: { addModule },
        sampleRate: 48_000
      } as unknown as AudioContext,
      captureSettings: {} as never,
      captureEpoch: 1,
      taskContext: "quiet-calibration",
      workletUrl: "/voice-capture-worklet.js",
      startupSignal: startup.signal,
      callbacks: callbacks()
    });

    startup.abort();
    resolveAddModule();

    await expect(pipeline).rejects.toThrow("voice-capture-startup-aborted");
    expect(workerConstructor).not.toHaveBeenCalled();
  });

  it("reports AudioWorklet processor errors and releases the pipeline", async () => {
    messageChannelStub();
    const worker = workerStub();
    const source = audioNode();
    const gain = Object.assign(audioNode(), { gain: { value: 1 } });
    const destination = audioNode();
    let worklet!: AudioWorkletNodeMock;
    class AudioWorkletNodeMock extends EventTarget {
      readonly port = { postMessage: vi.fn(), close: vi.fn() };
      readonly disconnect = vi.fn();
      connect(): AudioNode {
        return gain;
      }

      constructor() {
        super();
        worklet = this;
      }
    }
    vi.stubGlobal("AudioWorkletNode", AudioWorkletNodeMock);
    const onFailure = vi.fn();
    const onTransportStopped = vi.fn();

    const pipeline = await startVoiceCapturePipeline({
      stream: {} as MediaStream,
      audioContext: {
        audioWorklet: { addModule: vi.fn(async () => undefined) },
        createMediaStreamSource: () => source,
        createGain: () => gain,
        destination,
        currentTime: 0,
        sampleRate: 48_000,
        addEventListener() {},
        removeEventListener() {}
      } as unknown as AudioContext,
      captureSettings: {} as never,
      captureEpoch: 9,
      taskContext: "quiet-calibration",
      workletUrl: "/voice-capture-worklet.js",
      callbacks: { ...callbacks(onFailure), onTransportStopped }
    });

    worklet.dispatchEvent(new Event("processorerror"));
    expect(onFailure).toHaveBeenCalledOnce();
    expect(onFailure).toHaveBeenCalledWith(
      "audio-worklet-processor-failed"
    );

    await pipeline.stop();
    expect(worker.terminate).toHaveBeenCalledOnce();
    expect(worklet.disconnect).toHaveBeenCalledOnce();
    expect(onTransportStopped).toHaveBeenCalledWith({ status: "unavailable", captureEpoch: 9, reason: "worklet-stop-timeout" });
  });

  it("cleans up a worker and partial graph when worklet construction fails", async () => {
    messageChannelStub();
    const worker = workerStub();
    const disconnectSource = vi.fn();
    const source = audioNode(disconnectSource);
    vi.stubGlobal("AudioWorkletNode", class {
      constructor() {
        throw new Error("worklet-construction-failed");
      }
    });

    await expect(startVoiceCapturePipeline({
      stream: {} as MediaStream,
      audioContext: {
        audioWorklet: { addModule: vi.fn(async () => undefined) },
        createMediaStreamSource: () => source,
        currentTime: 0,
        sampleRate: 48_000,
        addEventListener() {},
        removeEventListener() {}
      } as unknown as AudioContext,
      captureSettings: {} as never,
      captureEpoch: 10,
      taskContext: "quiet-calibration",
      workletUrl: "/voice-capture-worklet.js",
      callbacks: callbacks()
    })).rejects.toThrow("worklet-construction-failed");

    expect(disconnectSource).toHaveBeenCalledOnce();
    expect(worker.terminate).toHaveBeenCalledOnce();
  });
  it("re-establishes audio clock mappings across suspend/resume and removes the listener on stop", async () => {
    messageChannelStub();
    const worker = workerStub();
    let now = 1_000;
    vi.stubGlobal("performance", { timeOrigin: 100_000, now: () => now });
    const node = audioNode();
    const context = Object.assign(new EventTarget(), {
      audioWorklet: { addModule: vi.fn(async () => undefined) },
      createMediaStreamSource: () => node,
      createGain: () => Object.assign(audioNode(), { gain: { value: 1 } }),
      destination: node, currentTime: 5, sampleRate: 48_000, state: "running"
    });
    class Worklet extends EventTarget {
      readonly port = { postMessage: vi.fn(), close: vi.fn() };
      connect() {} disconnect() {}
    }
    vi.stubGlobal("AudioWorkletNode", Worklet);
    const pipeline = await startVoiceCapturePipeline({
      stream: {} as MediaStream, audioContext: context as unknown as AudioContext,
      captureSettings: {} as never, captureEpoch: 9, taskContext: "quiet-calibration",
      workletUrl: "/voice-capture-worklet.js", callbacks: callbacks()
    });
    expect(worker.sent[0]).toMatchObject({ type: "initialize", audioContextSampleRateHz: 48_000,
      clockMapping: { id: 0, mainTimeOriginMs: 100_000, mainPerformanceMs: 1_000, audioContextTimeSeconds: 5 } });
    context.state = "suspended";
    context.dispatchEvent(new Event("statechange"));
    expect(worker.sent.at(-1)).toMatchObject({ type: "clock-mapping", clockMapping: null });
    pipeline.reset(10, "ambient-speech-turn", 0.002);
    expect(worker.sent.at(-1)).toMatchObject({ type: "reset", captureEpoch: 10, clockMapping: null });
    context.state = "running"; context.currentTime = 5.5; now = 2_000;
    context.dispatchEvent(new Event("statechange"));
    expect(worker.sent.at(-1)).toMatchObject({ type: "clock-mapping",
      clockMapping: { id: 3, mainPerformanceMs: 2_000, audioContextTimeSeconds: 5.5 } });
    pipeline.reset(11, "ambient-speech-turn", 0.002);
    expect(worker.sent.at(-1)).toMatchObject({ type: "reset", captureEpoch: 11,
      clockMapping: { id: 4 }, noiseFloorRms: 0.002 });
    await pipeline.stop();
    const requestsAtStop = worker.sent.length;
    context.dispatchEvent(new Event("statechange"));
    expect(worker.sent).toHaveLength(requestsAtStop);
  });

  it("initializes suspended contexts without a valid clock anchor", async () => {
    messageChannelStub();
    const worker = workerStub();
    const node = audioNode();
    const context = Object.assign(new EventTarget(), {
      audioWorklet: { addModule: vi.fn(async () => undefined) }, createMediaStreamSource: () => node,
      createGain: () => Object.assign(audioNode(), { gain: { value: 1 } }),
      destination: node, currentTime: 0, sampleRate: 48_000, state: "suspended"
    });
    class Worklet extends EventTarget {
      readonly port = { postMessage: vi.fn(), close: vi.fn() };
      connect() {} disconnect() {}
    }
    vi.stubGlobal("AudioWorkletNode", Worklet);
    const pipeline = await startVoiceCapturePipeline({
      stream: {} as MediaStream, audioContext: context as unknown as AudioContext,
      captureSettings: {} as never, captureEpoch: 1, taskContext: "ambient-speech-turn",
      workletUrl: "/voice-capture-worklet.js", callbacks: callbacks()
    });
    expect(worker.sent[0]).toMatchObject({ type: "initialize", clockMapping: null });
    context.state = "running";
    context.dispatchEvent(new Event("statechange"));
    expect(worker.sent.at(-1)).toMatchObject({ type: "clock-mapping", clockMapping: { id: 1 } });
    await pipeline.stop();
  });

  it("waits for a current-epoch final transport snapshot and makes stop idempotent", async () => {
    messageChannelStub();
    workerStub();
    const node = audioNode();
    const context = Object.assign(new EventTarget(), {
      audioWorklet: { addModule: vi.fn(async () => undefined) }, createMediaStreamSource: () => node,
      createGain: () => Object.assign(audioNode(), { gain: { value: 1 } }),
      destination: node, currentTime: 0, sampleRate: 48_000, state: "running"
    });
    const disposePosts = vi.fn();
    class Worklet extends EventTarget {
      readonly port = {
        onmessage: null as ((event: { data: unknown }) => void) | null,
        postMessage: (message: { type?: string }) => {
          if (message.type !== "dispose") return;
          disposePosts();
          const snapshot = { schemaVersion: "phenometric.voice-worklet-message.v2", type: "transport-diagnostics",
            final: true, sequence: 6, sampleClockEndExclusive: 5888, droppedBlockCount: 2,
            droppedSampleCount: 2048, discardedPartialSampleCount: 128,
            inFlightBlockCountAtDispose: 4, maximumInFlightBlocks: 4 };
          this.port.onmessage?.({ data: { ...snapshot, captureEpoch: 8 } });
          this.port.onmessage?.({ data: { ...snapshot, captureEpoch: 9 } });
        }, close() {}
      };
      connect() {} disconnect() {}
    }
    vi.stubGlobal("AudioWorkletNode", Worklet);
    const onTransportStopped = vi.fn();
    const pipeline = await startVoiceCapturePipeline({ stream: {} as MediaStream,
      audioContext: context as unknown as AudioContext, captureSettings: {} as never,
      captureEpoch: 9, taskContext: "quiet-calibration", workletUrl: "/voice-capture-worklet.js",
      callbacks: { ...callbacks(), onTransportStopped }
    });
    const firstStop = pipeline.stop();
    expect(pipeline.stop()).toBe(firstStop);
    await firstStop;
    expect(disposePosts).toHaveBeenCalledOnce();
    expect(onTransportStopped).toHaveBeenCalledExactlyOnceWith({ status: "complete", diagnostics: expect.objectContaining({
      captureEpoch: 9, droppedSampleCount: 2048, discardedPartialSampleCount: 128
    }) });
  });

  it("releases every processor resource when a stop-notification callback throws", async () => {
    messageChannelStub();
    const worker = workerStub();
    const source = audioNode();
    let worklet!: Worklet;
    class Worklet extends EventTarget {
      readonly port = { onmessage: null, postMessage() {}, close: vi.fn() };
      readonly disconnect = vi.fn();
      constructor() { super(); worklet = this; }
      connect() {}
    }
    vi.stubGlobal("AudioWorkletNode", Worklet);
    const context = Object.assign(new EventTarget(), {
      audioWorklet: { addModule: vi.fn(async () => undefined) }, createMediaStreamSource: () => source,
      createGain: () => Object.assign(audioNode(), { gain: { value: 1 } }),
      destination: audioNode(), currentTime: 0, sampleRate: 48_000, state: "running"
    });
    const pipeline = await startVoiceCapturePipeline({ stream: {} as MediaStream,
      audioContext: context as unknown as AudioContext, captureSettings: {} as never,
      captureEpoch: 9, taskContext: "quiet-calibration", workletUrl: "/voice-capture-worklet.js",
      callbacks: { ...callbacks(), onTransportStopped() { throw new Error("notification-failed"); } }
    });
    await expect(pipeline.stop()).rejects.toThrow("notification-failed");
    expect(worker.terminate).toHaveBeenCalledOnce();
    expect(worklet.disconnect).toHaveBeenCalledOnce();
    expect(worklet.port.close).toHaveBeenCalledOnce();
  });

});
