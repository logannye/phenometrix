import { EmbeddedLocalAdapter, EncounterCaptureController, RotatingAmbientEncounter,
  type CaptureAuthorization, type CaptureEvent, type CaptureSourceContext, type CaptureSourceSink,
  type LocalCaptureBranch } from "@phenometrix/encounter-capture";
import { calculateSha256Hex, type FaceCalibration, type DurableObservationV1 } from "@phenometrix/contracts";
import type { AmbientFacialFrame } from "@phenometrix/ambient-core";
import { startVoiceCapturePipeline, type VoiceCapturePipeline } from "./voice-capture.js";
import { requestedAudioCaptureSettings } from "./voice-worker-protocol.js";
import { finalizeMetricsInWorker } from "./ambient-finalization.js";
import { createVideoCaptureSettings, createVisualWorkerInitializeMessage, createVisualWorkerFrameMessage,
  FACE_LANDMARKER_MODEL_SHA256, type VisualWorkerResponse } from "./face-worker-protocol.js";
import { LatestFrameScheduler, VideoFramePump } from "./visual-frame-pump.js";
import { classifyFaceCalibration } from "./capture-calibration.js";
import { loadAndVerifyFaceStaticAssets, loadAndVerifyVoiceStaticAssets,
  type ResolvedFaceStaticAssets, type ResolvedVoiceStaticAssets } from "./static-assets.js";

export interface EmbeddedEncounterOptions {
  authorization: CaptureAuthorization;
  /** The normal telehealth client's own participant track, never a mixed remote stream. */
  stream: MediaStream;
  /** Host video element already displaying that participant's unmirrored source. */
  video?: HTMLVideoElement;
  /** Host-owned running context; this module never prompts for audio activation. */
  audioContext?: AudioContext;
  observationId: string;
  revisionId: string;
  /** Application base containing its integrity manifest and immutable worker assets. */
  assetBaseUrl?: string;
  /** Host/server clock estimate. Validity timestamps use the local Date.now clock. */
  clockCalibration?: { localMeasuredAtMs: number; localExpiresAtMs: number; utcOffsetMs: number; uncertaintyMs: number };
  /** Measured by host calibration, if available. Absence abstains from voice metrics. */
  noiseCalibration?: { noiseFloorRms: number; durationMs: number };
  onEvent?(event: CaptureEvent): void;
  onObservation?(observation: DurableObservationV1, signal: AbortSignal): void | Promise<void>;
}

/**
 * Concrete embedding entry point using the existing MediaPipe and DSP workers.
 * No getUserMedia, new login, capture buttons, visibility listeners or track.stop.
 * The host must stop it on encounter end, withdrawal or participant ambiguity.
 */
export function createEmbeddedEncounter(options: EmbeddedEncounterOptions) {
  const authorization = structuredClone(options.authorization);
  const clock = options.clockCalibration ? { ...options.clockCalibration } : null;
  if (clock && (Object.values(clock).some(v => !Number.isFinite(v)) || clock.uncertaintyMs < 0
    || clock.localMeasuredAtMs > Date.now() || clock.localExpiresAtMs <= Date.now())) throw new Error("Invalid host clock calibration");
  const now = () => Date.now() + (clock?.utcOffsetMs ?? 0);
  let window: RotatingAmbientEncounter | null = null;
  let rotationTimer: ReturnType<typeof setInterval> | null = null;
  const adapter = new EmbeddedLocalAdapter(async (context, sink, signal) => {
    if (clock && Date.now() >= clock.localExpiresAtMs) throw new Error("Host clock calibration expired");
    const clockTimer = clock ? setTimeout(() => controller.stop("transport-interrupted"),
      Math.min(2_147_483_647, clock.localExpiresAtMs - Date.now())) : null;
    signal.addEventListener("abort", () => { if (clockTimer) clearTimeout(clockTimer); }, { once: true });
    const performanceOrigin = performance.now();
    const unixOrigin = now();
    const toUnix = (time: number) => unixOrigin + time - performanceOrigin;
    const assets = await verifiedAssets(options.assetBaseUrl, context, signal);
    const acquisitionFingerprint = await fingerprintHostTracks(options.stream, context.modalities, authorization.binding.scope, assets);
    if (signal.aborted) return { stop() {} };
    window = new RotatingAmbientEncounter({ authorization,
      observationId: options.observationId, revisionId: options.revisionId, startedAtMs: context.startedAtMs,
      adapterId: "embedded-local", adapterVersion: "1.0.0", sourceKind: "patient-local-pre-codec",
      deviceClass: "browser-host-track:hardware-attestation-unavailable",
      // Sentinel uncertainty explicitly prevents treatment-timing qualification without a host clock estimate.
      clockUncertaintyMs: clock?.uncertaintyMs ?? Number.MAX_SAFE_INTEGER, acquisitionFingerprint,
      finalizeMetrics: finalizeMetricsInWorker,
      onObservation: options.onObservation,
      onFailure: () => controller.stop("processor-failed") });
    rotationTimer = setInterval(() => {
      if ((clock && Date.now() >= clock.localExpiresAtMs) || Math.abs(now() - toUnix(performance.now())) > 1_000) {
        controller.stop("transport-interrupted"); return;
      }
      try { window?.advance(now()); } catch { controller.stop("processor-failed"); }
    }, 1_000);
    signal.addEventListener("abort", () => { if (rotationTimer) clearInterval(rotationTimer); }, { once: true });
    if (validNoiseCalibration(options.noiseCalibration))
      window.setCalibration({ noiseDurationMs: options.noiseCalibration.durationMs });
    return attachBrowserWorkers(options, assets, context, sink, signal, toUnix,
      calibration => window?.setCalibration({ face: calibration }));
  });
  const controller = new EncounterCaptureController({ adapter, now, onEvent: event => {
    if (event.type === "stopped" && event.reason !== "encounter-inactive") window?.discard();
    options.onEvent?.(event);
  },
    onDerived: derived => { window?.append(derived); } });
  return {
    controller,
    start: () => controller.start(authorization),
    async finish() {
      controller.encounterEnded();
      if (!window) throw new Error("Encounter was not started");
      return window.finish(now());
    },
    withdraw() { controller.withdraw(); window?.discard(); },
    /** No derived record is created when the host abandons an encounter. */
    discard() { controller.stop(); window?.discard(); }
  };
}

function validNoiseCalibration(value: EmbeddedEncounterOptions["noiseCalibration"]): value is NonNullable<EmbeddedEncounterOptions["noiseCalibration"]> {
  return !!value && Number.isFinite(value.durationMs) && value.durationMs >= 2_000
    && Number.isFinite(value.noiseFloorRms) && value.noiseFloorRms > 0;
}

interface VerifiedAssets { voice: ResolvedVoiceStaticAssets | null; face: ResolvedFaceStaticAssets | null }
async function verifiedAssets(base: string | undefined, context: CaptureSourceContext, signal: AbortSignal): Promise<VerifiedAssets> {
  const documentBase = base ?? (typeof document === "undefined" ? "" : document.baseURI);
  const fetcher: typeof fetch = (input, init) => fetch(input, { ...init, signal: AbortSignal.any([signal, AbortSignal.timeout(15_000)]) });
  const [voice, face] = await Promise.all([
    context.modalities.includes("voice") ? loadAndVerifyVoiceStaticAssets(documentBase, fetcher).catch(() => null) : null,
    context.modalities.includes("face") ? loadAndVerifyFaceStaticAssets(documentBase, fetcher).catch(() => null) : null
  ]);
  return { voice, face: face?.manifest.assets.faceModel.sha256 === FACE_LANDMARKER_MODEL_SHA256 ? face : null };
}

async function fingerprintHostTracks(stream: MediaStream, modalities: readonly ("face" | "voice")[],
  scope: CaptureAuthorization["binding"]["scope"], assets: VerifiedAssets) {
  const audio = modalities.includes("voice") ? stream.getAudioTracks()[0]?.getSettings() : undefined;
  const video = modalities.includes("face") ? stream.getVideoTracks()[0]?.getSettings() : undefined;
  // Include opaque browser device IDs only in hash input, never a durable identifier or log.
  // These are browser-provided settings, not proof of a consistent physical device.
  return calculateSha256Hex(JSON.stringify({
    scope,
    assets: { voice: assets.voice?.manifest ?? null, face: assets.face?.manifest ?? null },
    audio: audio ? { device: audio.deviceId ?? "unknown", sampleRate: audio.sampleRate ?? "unknown",
      channelCount: audio.channelCount ?? "unknown", echoCancellation: audio.echoCancellation ?? "unknown",
      noiseSuppression: audio.noiseSuppression ?? "unknown", autoGainControl: audio.autoGainControl ?? "unknown" } : null,
    video: video ? { device: video.deviceId ?? "unknown", width: video.width ?? "unknown", height: video.height ?? "unknown",
      frameRate: video.frameRate ?? "unknown", facingMode: video.facingMode ?? "unknown" } : null
  }));
}

async function attachBrowserWorkers(options: EmbeddedEncounterOptions, assets: VerifiedAssets, context: CaptureSourceContext,
  sink: CaptureSourceSink, signal: AbortSignal, toUnix: (time: number) => number,
  onFaceCalibration: (calibration: FaceCalibration | null) => void): Promise<LocalCaptureBranch> {
  let stopped = false;
  let voice: VoiceCapturePipeline | null = null;
  let faceWorker: Worker | null = null;
  let pump: VideoFramePump<ImageBitmap> | null = null;
  let faceCalibration: FaceCalibration | null = null;
  let faceCalibrationFrames: AmbientFacialFrame[] = [];
  let faceInvalidatedThroughMs = -Infinity;
  const faceEpoch = 1;
  let voiceEpoch = 1;
  let voiceSequence = 0;
  const cleanup: (() => void)[] = [];
  const branch: LocalCaptureBranch = { stop() {
    if (stopped) return;
    stopped = true;
    for (const remove of cleanup) { try { remove(); } catch { /* Continue disposing other owned resources. */ } }
    try { pump?.stop(); } catch { /* Failed presentation callback. */ } pump = null;
    try { faceWorker?.terminate(); } catch { /* Already unavailable. */ } faceWorker = null;
    // Existing voice pipeline disconnects/terminates within its 500 ms disposal bound.
    void voice?.stop().catch(() => {}); voice = null;
    faceCalibrationFrames = []; faceCalibration = null;
  } };
  signal.addEventListener("abort", () => branch.stop(), { once: true });

  const audioTrack = options.stream.getAudioTracks()[0];
  const videoTrack = options.stream.getVideoTracks()[0];
  const settingsIdentity = (track: MediaStreamTrack) => JSON.stringify(track.getSettings());
  const originalAudioSettings = audioTrack ? settingsIdentity(audioTrack) : null;
  const originalVideoSettings = videoTrack ? settingsIdentity(videoTrack) : null;
  const sourceMatches = (modality: "face" | "voice") => {
    const track = modality === "voice" ? audioTrack : videoTrack;
    if (!track) return false;
    const tracks = modality === "voice" ? options.stream.getAudioTracks() : options.stream.getVideoTracks();
    if (tracks.length !== 1 || tracks[0] !== track || settingsIdentity(track) !== (modality === "voice" ? originalAudioSettings : originalVideoSettings)) return false;
    if (modality === "voice") return true;
    const displayed = options.video?.srcObject;
    return displayed instanceof MediaStream && displayed.getVideoTracks().length === 1 && displayed.getVideoTracks()[0] === track;
  };
  const verifySource = (modality: "face" | "voice") => {
    try { if (sourceMatches(modality)) return true; } catch { /* Device/source inspection failed. */ }
    sink.interruption("participant-source-changed"); return false;
  };
  const enabled = (track?: MediaStreamTrack) => !!track && track.readyState === "live" && track.enabled && !track.muted;
  const watch = (track: MediaStreamTrack | undefined, modality: "voice" | "face", changed: () => void) => {
    if (!track) { sink.availability(modality, false); return; }
    let wasEnabled = enabled(track);
    const listener = () => { wasEnabled = enabled(track); sink.availability(modality, wasEnabled); changed(); };
    const monitor = setInterval(() => { if (!verifySource(modality)) return; if (enabled(track) !== wasEnabled) listener(); }, 250);
    cleanup.push(() => clearInterval(monitor));
    for (const event of ["mute", "unmute", "ended"]) {
      track.addEventListener(event, listener);
      cleanup.push(() => track.removeEventListener(event, listener));
    }
    sink.availability(modality, enabled(track));
  };

  try {
    if (context.modalities.includes("voice") && assets.voice && audioTrack && sourceMatches("voice") && options.audioContext?.state === "running") {
      const settings = audioTrack.getSettings();
      try { voice = await startVoiceCapturePipeline({ stream: new MediaStream([audioTrack]), audioContext: options.audioContext,
        captureSettings: requestedAudioCaptureSettings(options.audioContext.sampleRate, settings.channelCount ?? 1,
          { echoCancellation: settings.echoCancellation ?? true, noiseSuppression: settings.noiseSuppression ?? true,
            autoGainControl: settings.autoGainControl ?? true }),
        captureEpoch: voiceEpoch, taskContext: "ambient-speech-turn", workletUrl: assets.voice.voiceWorkletUrl,
        startupSignal: signal,
        callbacks: {
          onFrame(frame) {
            if (stopped || !verifySource("voice") || !enabled(audioTrack) || frame.captureEpoch !== voiceEpoch) return;
            const acquiredAtMs = toUnix(frame.acquiredAtMs);
            sink.derived({ encounterId: context.encounterId, platformParticipantId: context.platformParticipantId,
              attribution: "individual-track", modality: "voice", trackId: `local-audio-${voiceEpoch}`,
              sequence: ++voiceSequence, acquiredAtMs,
              batch: { voiceFrames: [{ ...frame, acquiredAtMs, tMs: acquiredAtMs - context.startedAtMs }] } });
          },
          onReady() {}, onDiagnostics() {},
          onFailure() { sink.availability("voice", false); void voice?.stop(); }
        } });
      if (validNoiseCalibration(options.noiseCalibration)) voice.setNoiseFloor(options.noiseCalibration.noiseFloorRms);
      watch(audioTrack, "voice", () => voice?.reset(++voiceEpoch, "ambient-speech-turn"));
      const clockStateChanged = () => { if (options.audioContext?.state !== "running") sink.interruption("audio-clock-discontinuity"); };
      options.audioContext.addEventListener?.("statechange", clockStateChanged);
      cleanup.push(() => options.audioContext?.removeEventListener?.("statechange", clockStateChanged));
      } catch { sink.availability("voice", false); }
    } else sink.availability("voice", false);

    if (signal.aborted) { branch.stop(); return branch; }
    if (context.modalities.includes("face") && assets.face && videoTrack && options.video && sourceMatches("face")) {
      try {
      const video = options.video;
      const settings = videoTrack.getSettings();
      faceWorker = new Worker(new URL("./face-worker.ts", import.meta.url), { type: "module" });
      const worker = faceWorker;
      let pendingPresentation: { sequence: number; acquiredAtMs: number } | null = null;
      let inferenceTimeout: ReturnType<typeof setTimeout> | null = null;
      const failFace = () => {
        sink.availability("face", false); pump?.stop(); worker.terminate();
        if (inferenceTimeout) clearTimeout(inferenceTimeout);
      };
      cleanup.push(() => { if (inferenceTimeout) clearTimeout(inferenceTimeout); });
      const scheduler = new LatestFrameScheduler<ImageBitmap>({ captureEpoch: faceEpoch, onSubmit(frame) {
        if (stopped || !verifySource("face") || !enabled(videoTrack)) throw new Error("Frame unavailable");
        const acquiredAtMs = toUnix(frame.acquisitionTimestampMs);
        const message = createVisualWorkerFrameMessage({ ...frame, acquisitionTimestampMs: acquiredAtMs },
          { tMs: acquiredAtMs - context.startedAtMs, taskContext: "ambient-frontal", calibration: faceCalibration });
        pendingPresentation = { sequence: frame.sequence, acquiredAtMs: frame.acquisitionTimestampMs };
        inferenceTimeout = setTimeout(failFace, 2_000);
        worker.postMessage(message, [frame.frame]);
      } });
      await new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error("Face worker startup timed out")), 10_000);
        const abort = () => { clearTimeout(timeout); reject(new Error("Face startup aborted")); };
        signal.addEventListener("abort", abort, { once: true });
        cleanup.push(() => { clearTimeout(timeout); signal.removeEventListener("abort", abort); });
        worker.onmessage = (event: MessageEvent<VisualWorkerResponse>) => {
          if (stopped || event.data.captureEpoch !== faceEpoch) return;
          const message = event.data;
          if (message.type === "ready") { clearTimeout(timeout); signal.removeEventListener("abort", abort); resolve(); }
          else if (message.type === "frame") {
            // Scheduler clocks use browser presentation time; worker provenance uses Unix milliseconds.
            if (inferenceTimeout) clearTimeout(inferenceTimeout);
            if (!pendingPresentation || pendingPresentation.sequence !== message.sequence) return;
            scheduler.accept({ captureEpoch: faceEpoch, sequence: message.sequence, acquisitionTimestampMs: pendingPresentation.acquiredAtMs });
            pendingPresentation = null;
            if (!verifySource("face") || !enabled(videoTrack) || message.acquiredAtMs <= faceInvalidatedThroughMs) return;
            const frame = { ...message.frame, faceCount: message.faceCount };
            if (message.faceCount !== 1) {
              faceCalibrationFrames = []; faceCalibration = null; onFaceCalibration(null);
            } else if (!faceCalibration) {
              faceCalibrationFrames.push(frame);
              faceCalibrationFrames = faceCalibrationFrames.filter(f => frame.acquiredAtMs - f.acquiredAtMs <= 5_000).slice(-180);
              const assessed = classifyFaceCalibration(faceCalibrationFrames);
              if (assessed.quality === "strong") { faceCalibration = assessed.calibration; onFaceCalibration(faceCalibration); faceCalibrationFrames = []; }
            }
            sink.derived({ encounterId: context.encounterId, platformParticipantId: context.platformParticipantId,
              attribution: "individual-track", modality: "face", trackId: `local-video-${faceEpoch}`,
              sequence: message.sequence, acquiredAtMs: message.acquiredAtMs, batch: { faceFrames: [frame] } });
          } else if (message.type === "error") { clearTimeout(timeout); failFace(); reject(new Error("Face processor unavailable")); }
        };
        worker.onerror = () => { clearTimeout(timeout); failFace(); reject(new Error("Face worker unavailable")); };
        worker.postMessage(createVisualWorkerInitializeMessage(faceEpoch,
          createVideoCaptureSettings({ width: settings.width ?? video.videoWidth, height: settings.height ?? video.videoHeight,
            frameRate: settings.frameRate ?? null }), {
            mediaPipeRootUrl: assets.face!.mediaPipeRootUrl, modelUrl: assets.face!.faceModelUrl,
            modelSha256: assets.face!.manifest.assets.faceModel.sha256 }));
      });
      if (signal.aborted) { branch.stop(); return branch; }
      pump = new VideoFramePump({ source: video, scheduler,
        capture: () => verifySource("face") ? createImageBitmap(video) : Promise.reject(new Error("Patient video source unavailable")),
        taskContextAtAcquisition: () => "ambient-frontal" });
      watch(videoTrack, "face", () => {
        // Loss/reacquisition invalidates the in-memory calibration. Worker handles track continuity too.
        faceInvalidatedThroughMs = toUnix(performance.now());
        faceCalibrationFrames = []; faceCalibration = null; onFaceCalibration(null);
      });
      pump.start();
      } catch { faceWorker?.terminate(); faceWorker = null; sink.availability("face", false); }
    } else sink.availability("face", false);
    return branch;
  } catch {
    branch.stop();
    throw new Error("Embedded worker startup unavailable");
  }
}
