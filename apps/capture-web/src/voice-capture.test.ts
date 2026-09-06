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
  } {
    const terminate = vi.fn();
    class WorkerMock extends EventTarget {
      postMessage(message: { type?: string; captureEpoch?: number }): void {
        if (message.type !== "dispose") return;
        const event = new Event("message") as MessageEvent;
        Object.defineProperty(event, "data", {
          value: {
            schemaVersion: "phenometric.voice-worker-message.v1",
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
    return { terminate };
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

    const pipeline = await startVoiceCapturePipeline({
      stream: {} as MediaStream,
      audioContext: {
        audioWorklet: { addModule: vi.fn(async () => undefined) },
        createMediaStreamSource: () => source,
        createGain: () => gain,
        destination,
        currentTime: 0
      } as unknown as AudioContext,
      captureSettings: {} as never,
      captureEpoch: 9,
      taskContext: "quiet-calibration",
      workletUrl: "/voice-capture-worklet.js",
      callbacks: callbacks(onFailure)
    });

    worklet.dispatchEvent(new Event("processorerror"));
    expect(onFailure).toHaveBeenCalledOnce();
    expect(onFailure).toHaveBeenCalledWith(
      "audio-worklet-processor-failed"
    );

    await pipeline.stop();
    expect(worker.terminate).toHaveBeenCalledOnce();
    expect(worklet.disconnect).toHaveBeenCalledOnce();
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
        currentTime: 0
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
});
