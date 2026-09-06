import type {
  AudioCaptureSettings,
  AudioPipelineProvenance,
  AudioStreamDiagnostics,
  VoiceTaskContext
} from "@phenometrix/contracts";
import type { VoiceSignalFrameV1 } from "@phenometrix/ambient-core";
import {
  isCurrentVoiceWorkerResponse,
  VOICE_WORKER_MESSAGE_VERSION,
  type VoiceWorkerRequest,
  type VoiceWorkerResponse
} from "./voice-worker-protocol.js";

export interface VoiceCaptureCallbacks {
  onFrame(frame: VoiceSignalFrameV1, processingLatencyMs: number): void;
  onReady(provenance: AudioPipelineProvenance): void;
  onDiagnostics(diagnostics: AudioStreamDiagnostics): void;
  onFailure(reason: string): void;
}

export interface VoiceCaptureStartOptions {
  stream: MediaStream;
  audioContext: AudioContext;
  captureSettings: AudioCaptureSettings;
  captureEpoch: number;
  taskContext: VoiceTaskContext;
  workletUrl: string;
  startupSignal?: AbortSignal;
  callbacks: VoiceCaptureCallbacks;
}

export interface VoiceCapturePipeline {
  setTask(taskContext: VoiceTaskContext): void;
  setNoiseFloor(noiseFloorRms: number): void;
  reset(captureEpoch: number, taskContext: VoiceTaskContext): void;
  stop(): Promise<void>;
  readonly captureEpoch: number;
}

function request(
  value: Record<string, unknown>
): VoiceWorkerRequest {
  return {
    schemaVersion: VOICE_WORKER_MESSAGE_VERSION,
    ...value
  } as VoiceWorkerRequest;
}

export async function startVoiceCapturePipeline(
  options: VoiceCaptureStartOptions
): Promise<VoiceCapturePipeline> {
  if (!options.audioContext.audioWorklet) {
    throw new Error("audio-worklet-unavailable");
  }
  await options.audioContext.audioWorklet.addModule(
    options.workletUrl
  );
  if (options.startupSignal?.aborted) {
    throw new Error("voice-capture-startup-aborted");
  }
  let worker: Worker | null = null;
  let source: MediaStreamAudioSourceNode | null = null;
  let worklet: AudioWorkletNode | null = null;
  let mutedOutput: GainNode | null = null;
  let channel: MessageChannel | null = null;
  let captureEpoch = options.captureEpoch;
  let stopped = false;
  let disposedResolver: (() => void) | null = null;
  let stopPromise: Promise<void> | null = null;

  const releaseResources = (): void => {
    for (const node of [source, worklet, mutedOutput]) {
      try {
        node?.disconnect();
      } catch {
        // A partially constructed graph can already be disconnected.
      }
    }
    for (const port of [channel?.port1, channel?.port2, worklet?.port]) {
      try {
        port?.close();
      } catch {
        // Transferred or failed ports can already be detached.
      }
    }
    try {
      worker?.terminate();
    } catch {
      // Construction failure can leave a non-operational worker handle.
    }
    disposedResolver = null;
  };

  try {
    worker = new Worker(
      new URL("./voice-worker.ts", import.meta.url),
      { type: "module" }
    );
    const activeWorker = worker;
    source = options.audioContext.createMediaStreamSource(options.stream);
    const activeSource = source;
    worklet = new AudioWorkletNode(
      options.audioContext,
      /*
       * Deliberately NOT renamed with the rest of PhenoMetrix.
       *
       * This is a runtime registration identifier, the same category as the
       * "phenometric.<name>.vN" schema strings that were left alone for the same
       * reason. Worse, the worklet is served from public/ at a fixed unhashed
       * URL, so browsers cache it hard: a stale copy registers the old name while
       * a fresh bundle asks for the new one, AudioWorkletNode throws, and the
       * voice lane dies silently -- live telemetry simply stops. Renaming it was
       * cosmetic and cost exactly that.
       */
      "phenometric-voice-capture",
      {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
        channelCount: 1,
        channelCountMode: "explicit"
      }
    );
    const activeWorklet = worklet;
    mutedOutput = options.audioContext.createGain();
    const activeMutedOutput = mutedOutput;
    activeMutedOutput.gain.value = 0;
    activeSource.connect(activeWorklet);
    activeWorklet.connect(activeMutedOutput);
    activeMutedOutput.connect(options.audioContext.destination);

    channel = new MessageChannel();
    const activeChannel = channel;

    activeWorker.addEventListener(
      "message",
      (event: MessageEvent<VoiceWorkerResponse>) => {
        if (!isCurrentVoiceWorkerResponse(event.data, captureEpoch)) {
          return;
        }
        if (event.data.type === "disposed") {
          options.callbacks.onDiagnostics(event.data.diagnostics);
          disposedResolver?.();
          disposedResolver = null;
          return;
        }
        if (stopped) return;
        if (event.data.type === "ready") {
          options.callbacks.onReady(event.data.provenance);
        } else if (event.data.type === "signal-frame") {
          options.callbacks.onFrame(
            event.data.frame,
            event.data.processingLatencyMs
          );
        } else if (event.data.type === "diagnostics") {
          options.callbacks.onDiagnostics(event.data.diagnostics);
        } else if (event.data.type === "error") {
          options.callbacks.onFailure(event.data.reason);
        }
      }
    );
    activeWorker.addEventListener("error", () => {
      if (!stopped) {
        options.callbacks.onFailure("voice-worker-unavailable");
      }
    });
    activeWorklet.addEventListener("processorerror", () => {
      if (!stopped) {
        options.callbacks.onFailure("audio-worklet-processor-failed");
      }
    }, { once: true });

    activeWorklet.port.postMessage(
      { type: "attach-port", port: activeChannel.port1 },
      [activeChannel.port1]
    );
    activeWorklet.port.postMessage({
      type: "capture-epoch",
      captureEpoch
    });
    activeWorker.postMessage(
      request({
        type: "initialize",
        captureEpoch,
        port: activeChannel.port2,
        sessionOriginPerformanceMs: performance.now(),
        audioContextOriginSeconds: options.audioContext.currentTime,
        captureSettings: options.captureSettings,
        taskContext: options.taskContext
      }),
      [activeChannel.port2]
    );

    return {
      get captureEpoch() {
        return captureEpoch;
      },
      setTask(taskContext) {
        if (stopped) return;
        activeWorker.postMessage(
          request({ type: "set-task", captureEpoch, taskContext })
        );
      },
      setNoiseFloor(noiseFloorRms) {
        if (stopped || !Number.isFinite(noiseFloorRms)) return;
        activeWorker.postMessage(
          request({
            type: "set-noise-floor",
            captureEpoch,
            noiseFloorRms
          })
        );
      },
      reset(nextEpoch, taskContext) {
        if (stopped) return;
        captureEpoch = nextEpoch;
        activeWorklet.port.postMessage({
          type: "capture-epoch",
          captureEpoch
        });
        activeWorker.postMessage(
          request({ type: "reset", captureEpoch, taskContext })
        );
      },
      stop() {
        if (stopPromise) return stopPromise;
        stopped = true;
        stopPromise = (async () => {
          const acknowledgement = new Promise<void>((resolve) => {
            disposedResolver = resolve;
          });
          try {
            activeWorklet.port.postMessage({ type: "dispose" });
          } catch {
            // A failed worklet can already have released its message port.
          }
          try {
            activeWorker.postMessage(
              request({ type: "dispose", captureEpoch })
            );
          } catch {
            disposedResolver?.();
            disposedResolver = null;
          }
          let timeout: ReturnType<typeof setTimeout> | undefined;
          await Promise.race([
            acknowledgement,
            new Promise<void>((resolve) => {
              timeout = setTimeout(resolve, 500);
            })
          ]);
          if (timeout !== undefined) clearTimeout(timeout);
          releaseResources();
        })();
        return stopPromise;
      }
    };
  } catch (error) {
    try {
      worklet?.port.postMessage({ type: "dispose" });
    } catch {
      // The processor may not have completed construction.
    }
    releaseResources();
    throw error;
  }
}
