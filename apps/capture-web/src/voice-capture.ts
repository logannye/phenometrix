import { VoiceResultBatchReceiver } from "./voice-result-transport.js";
import type {
  AudioCaptureSettings,
  AudioPipelineProvenance,
  AudioStreamDiagnostics,
  VoiceTaskContext
} from "@phenometrix/contracts";
import { captureVoiceClockMapping, type VoiceFrameTiming, type VoiceBlockTiming } from "./voice-stream-timing.js";
import type { VoiceSignalFrameV1 } from "@phenometrix/ambient-core";
import {
  isCurrentVoiceWorkerResponse,
  isCurrentVoiceTransportDiagnostics,
  type VoiceTransportStopResult,
  VOICE_WORKER_MESSAGE_VERSION,
  type VoiceWorkerRequest,
  type VoiceWorkerResponse
} from "./voice-worker-protocol.js";

export interface VoiceCaptureCallbacks {
  onFrame(frame: VoiceSignalFrameV1, processingLatencyMs: number | null, timing?: VoiceFrameTiming): void;
  onReady(provenance: AudioPipelineProvenance): void;
  onDiagnostics(diagnostics: AudioStreamDiagnostics): void;
  onBlockTiming?(timing: VoiceBlockTiming): void;
  onTransportStopped?(result: VoiceTransportStopResult): void;
  onFailure(reason: string): void;
}

export interface VoiceCaptureStartOptions {
  stream: MediaStream;
  audioContext: AudioContext;
  captureSettings: AudioCaptureSettings;
  captureEpoch: number;
  taskContext: VoiceTaskContext;
  noiseFloorRms?: number;
  workletUrl: string;
  startupSignal?: AbortSignal;
  callbacks: VoiceCaptureCallbacks;
}

export interface VoiceCapturePipeline {
  setTask(taskContext: VoiceTaskContext): void;
  setNoiseFloor(noiseFloorRms: number): void;
  reset(captureEpoch: number, taskContext: VoiceTaskContext, noiseFloorRms?: number): void;
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
  if (options.noiseFloorRms !== undefined && (!Number.isFinite(options.noiseFloorRms)
    || options.noiseFloorRms < 0.0001 || options.noiseFloorRms > 1)) throw new Error("invalid-noise-floor");
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
  let clockMappingId = 0;
  let removeClockListener: (() => void) | null = null;
  let disposedResolver: (() => void) | null = null;
  let transportDisposedResolver: (() => void) | null = null;
  let transportDisposed = false;
  let stopPromise: Promise<void> | null = null;

  const releaseResources = (): void => {
    removeClockListener?.();
    removeClockListener = null;
    if (worklet) worklet.port.onmessage = null;
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
    transportDisposedResolver = null;
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
    activeWorklet.port.onmessage = (event: MessageEvent<unknown>) => {
      if (!stopped || transportDisposed || !isCurrentVoiceTransportDiagnostics(event.data, captureEpoch)) return;
      transportDisposed = true;
      const resolve = transportDisposedResolver;
      transportDisposedResolver = null;
      resolve?.();
      options.callbacks.onTransportStopped?.({ status: "complete", diagnostics: event.data });
    };
    mutedOutput = options.audioContext.createGain();
    const activeMutedOutput = mutedOutput;
    activeMutedOutput.gain.value = 0;
    activeSource.connect(activeWorklet);
    activeWorklet.connect(activeMutedOutput);
    activeMutedOutput.connect(options.audioContext.destination);

    channel = new MessageChannel();
    const activeChannel = channel;

    const resultReceiver = new VoiceResultBatchReceiver(message => activeWorker.postMessage(message));
    const handleWorkerResponse = (data: VoiceWorkerResponse): void => {
      if (!isCurrentVoiceWorkerResponse(data, captureEpoch)) {
        return;
      }
      if (data.type === "disposed") {
        const resolve = disposedResolver;
        disposedResolver = null;
        resolve?.();
        options.callbacks.onDiagnostics(data.diagnostics);
        return;
      }
      if (stopped) return;
      if (data.type === "ready") {
        options.callbacks.onReady(data.provenance);
      } else if (data.type === "signal-frame") {
        const mainResultArrivalMs = data.timing.mainResultArrivalMs ?? performance.now();
        const sampleAvailabilityToMainResultMs = mainResultArrivalMs - data.timing.sampleAvailabilityMs;
        const valid = data.timing.status === "valid" && Number.isFinite(sampleAvailabilityToMainResultMs) &&
          sampleAvailabilityToMainResultMs >= 0 && mainResultArrivalMs >= data.timing.workerReadyAtMs;
        options.callbacks.onFrame(data.frame, valid ? data.processingLatencyMs : null, {
          ...data.timing, mainResultArrivalMs,
          status: valid ? "valid" : "unavailable",
          ...(valid ? {} : { unavailableReason: "inconsistent-clock-mapping" as const }),
          sampleAvailabilityToMainResultMs: valid ? sampleAvailabilityToMainResultMs : null,
          rawSampleAvailabilityToMainResultMs: sampleAvailabilityToMainResultMs
        });
      } else if (data.type === "block-timing") {
        options.callbacks.onBlockTiming?.(data.timing);
      } else if (data.type === "diagnostics") {
        options.callbacks.onDiagnostics(data.diagnostics);
      } else if (data.type === "error") {
        options.callbacks.onFailure(data.reason);
      }
    };
    activeWorker.addEventListener("message", (event: MessageEvent<VoiceWorkerResponse>) => {
      const received = resultReceiver.receive(event.data, captureEpoch, performance.now(), clockMappingId);
      for (const message of received.messages) handleWorkerResponse(message);
    });
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
    const initialClockMapping = options.audioContext.state === "running"
      ? captureVoiceClockMapping(options.audioContext, performance, clockMappingId) : null;
    activeWorklet.port.postMessage({
      type: "capture-epoch", captureEpoch, clockMappingId
    });
    activeWorker.postMessage(
      request({
        type: "initialize",
        captureEpoch,
        port: activeChannel.port2,
        clockMapping: initialClockMapping,
        audioContextSampleRateHz: options.audioContext.sampleRate,
        captureSettings: options.captureSettings,
        taskContext: options.taskContext,
        ...(options.noiseFloorRms === undefined ? {} : { noiseFloorRms: options.noiseFloorRms })
      }),
      [activeChannel.port2]
    );

    const updateClockMapping = (): void => {
      if (stopped) return;
      clockMappingId += 1;
      const clockMapping = options.audioContext.state === "running"
        ? captureVoiceClockMapping(options.audioContext, performance, clockMappingId) : null;
      activeWorklet.port.postMessage({ type: "clock-mapping", clockMappingId });
      activeWorker.postMessage(request({ type: "clock-mapping", captureEpoch, clockMapping }));
    };
    options.audioContext.addEventListener("statechange", updateClockMapping);
    removeClockListener = () => options.audioContext.removeEventListener("statechange", updateClockMapping);

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
      reset(nextEpoch, taskContext, noiseFloorRms) {
        if (stopped) return;
        if (noiseFloorRms !== undefined && (!Number.isFinite(noiseFloorRms)
          || noiseFloorRms < 0.0001 || noiseFloorRms > 1)) throw new Error("invalid-noise-floor");
        captureEpoch = nextEpoch;
        clockMappingId += 1;
        const clockMapping = options.audioContext.state === "running"
          ? captureVoiceClockMapping(options.audioContext, performance, clockMappingId) : null;
        activeWorklet.port.postMessage({ type: "capture-epoch", captureEpoch, clockMappingId });
        activeWorker.postMessage(
          request({ type: "reset", captureEpoch, taskContext, clockMapping,
            ...(noiseFloorRms === undefined ? {} : { noiseFloorRms }) })
        );
      },
      stop() {
        if (stopPromise) return stopPromise;
        stopped = true;
        stopPromise = (async () => {
          let timeout: ReturnType<typeof setTimeout> | undefined;
          try {
            const workerAcknowledgement = new Promise<void>((resolve) => { disposedResolver = resolve; });
            const transportAcknowledgement = new Promise<void>((resolve) => {
              if (transportDisposed) resolve();
              else transportDisposedResolver = resolve;
            });
            const transportUnavailable = (reason: "worklet-stop-timeout" | "worklet-dispose-failed"): void => {
              if (transportDisposed) return;
              transportDisposed = true;
              const resolve = transportDisposedResolver;
              transportDisposedResolver = null;
              resolve?.();
              options.callbacks.onTransportStopped?.({ status: "unavailable", captureEpoch, reason });
            };
            try {
              activeWorklet.port.postMessage({ type: "dispose" });
            } catch {
              // A failed worklet can already have released its message port.
              transportUnavailable("worklet-dispose-failed");
            }
            try {
              activeWorker.postMessage(
                request({ type: "dispose", captureEpoch })
              );
            } catch {
              disposedResolver?.();
              disposedResolver = null;
            }
            await Promise.race([
              Promise.all([workerAcknowledgement, transportAcknowledgement]),
              new Promise<void>((resolve) => {
                timeout = setTimeout(resolve, 500);
              })
            ]);
            transportUnavailable("worklet-stop-timeout");
          } finally {
            if (timeout !== undefined) clearTimeout(timeout);
            releaseResources();
          }
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
