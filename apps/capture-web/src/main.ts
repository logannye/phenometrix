import {
  AMBIENT_FACE_ALGORITHM_VERSION,
  AMBIENT_FACE_TASK_CONTEXT,
  AMBIENT_VOICE_ALGORITHM_VERSION,
  AMBIENT_VOICE_TASK_CONTEXT,
  type AmbientFacialFrame,
  type AmbientVoiceFrame
} from "@phenometrix/ambient-core";
import {
  AMBIENT_LOCAL_CONSENT_TEXT,
  AMBIENT_LOCAL_PROTOCOL_PACK,
  AMBIENT_LOCAL_PROTOCOL_REF,
  REPORT_BOUNDARY_STATEMENT,
  REPORT_SOURCE_DISCLOSURE,
  type AudioPipelineProvenance,
  type ConsentRecordV1,
  type FaceCalibration,
  type PostEncounterReportV1,
  type ProcessorProvenanceV1,
  type VisualPipelineProvenance,
  type WithheldReasonCode
} from "@phenometrix/contracts";
import {
  buildConditionEvidenceCard,
  buildPostEncounterReport
} from "@phenometrix/evidence-core";
import { InMemoryEventJournal } from "@phenometrix/event-log";
import { comparePreviousVisit } from "@phenometrix/trajectory-core";
import { buildAmbientObservation } from "./ambient-core-adapter.js";
import {
  AMBIENT_CAPTURE_LIMIT_MS,
  AMBIENT_SETUP_TIMEOUT_MS,
  createAmbientWorkflowState,
  reduceAmbientWorkflow,
  type AmbientWorkflowEffect,
  type AmbientWorkflowEvent,
  type AmbientWorkflowState
} from "./ambient-workflow.js";
import { classifyFaceCalibration } from "./capture-calibration.js";
import { ConditionDemoController } from "./condition-demo-controller.js";
import {
  clearConditionEvidenceCard,
  renderConditionEvidenceCard,
  type ConditionEvidenceViewElements
} from "./condition-evidence-view.js";
import {
  cleanupAudioLaneResources,
  cleanupFaceLaneResources
} from "./capture-lane-cleanup.js";
import {
  CaptureRuntime,
  withTimeout,
  type DerivedCaptureSnapshot
} from "./capture-runtime.js";
import {
  VISUAL_WORKER_MESSAGE_VERSION,
  createVideoCaptureSettings,
  createVisualWorkerFrameMessage,
  createVisualWorkerInitializeMessage,
  visualWorkerMessage,
  type VisualWorkerResponse
} from "./face-worker-protocol.js";
import { FaceOverlayController } from "./face-overlay-controller.js";
import { LiveVoiceVisualizer } from "./live-voice-visualizer.js";
import {
  loadAndVerifyFaceStaticAssets,
  loadAndVerifyVoiceStaticAssets,
  type ResolvedFaceStaticAssets,
  type ResolvedVoiceStaticAssets
} from "./static-assets.js";
import {
  LatestFrameScheduler,
  VideoFramePump,
  type ScheduledVisualFrame
} from "./visual-frame-pump.js";
import {
  startVoiceCapturePipeline,
  type VoiceCapturePipeline
} from "./voice-capture.js";
import {
  requestedAudioCaptureSettings
} from "./voice-worker-protocol.js";

function element<T extends HTMLElement>(id: string): T {
  const value = document.getElementById(id);
  if (!value) throw new Error(`Missing required element #${id}`);
  return value as T;
}

// Detected once on the main thread, where matchMedia exists. The face-worker
// runs in a DedicatedWorkerGlobalScope with no matchMedia, so this presentation
// preference is threaded into the worker via the attach-overlay message.
const prefersReducedMotion =
  typeof window.matchMedia === "function" &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const welcomeView = element<HTMLElement>("welcome-view");
const captureView = element<HTMLElement>("capture-view");
const messageView = element<HTMLElement>("message-view");
const reportView = element<HTMLElement>("report-view");
const consentForm = element<HTMLFormElement>("consent-form");
const consentCheckbox = element<HTMLInputElement>("consent-checkbox");
const consentText = element<HTMLElement>("consent-text");
const affectedSideFieldset = element<HTMLFieldSetElement>("affected-side-fieldset");
const affectedSideLeft = element<HTMLInputElement>("affected-side-left");
const affectedSideRight = element<HTMLInputElement>("affected-side-right");
const startButton = element<HTMLButtonElement>("start-button");
const finishButton = element<HTMLButtonElement>("finish-button");
const discardButton = element<HTMLButtonElement>("discard-button");
const resetButton = element<HTMLButtonElement>("reset-button");
const messageResetButton = element<HTMLButtonElement>("message-reset-button");
const messageTitle = element<HTMLElement>("message-title");
const messageDetail = element<HTMLElement>("message-detail");
const phaseLabel = element<HTMLElement>("phase-label");
const privacyState = element<HTMLElement>("privacy-state");
const captureEyebrow = element<HTMLElement>("capture-eyebrow");
const captureTitle = element<HTMLElement>("capture-title");
const captureInstruction = element<HTMLElement>("capture-instruction");
const captureStatus = element<HTMLElement>("capture-status");
const audioLaneState = element<HTMLElement>("audio-lane-state");
const audioLaneDetail = element<HTMLElement>("audio-lane-detail");
const faceLaneState = element<HTMLElement>("face-lane-state");
const faceLaneDetail = element<HTMLElement>("face-lane-detail");
const cameraPreview = element<HTMLVideoElement>("camera-preview");
const landmarkOverlay = element<HTMLCanvasElement>("landmark-overlay");
const faceMeshStatus = element<HTMLElement>("face-mesh-status");
const cameraPlaceholder = element<HTMLElement>("camera-placeholder");
const sessionClock = element<HTMLTimeElement>("session-clock");
const reportBoundary = element<HTMLElement>("report-boundary");
const reportSource = element<HTMLElement>("report-source");
const reportSections = element<HTMLElement>("report-sections");
const conditionSideBadge = element<HTMLElement>("condition-side-badge");
const conditionStatus = element<HTMLElement>("condition-status");
const acceptReferenceButton = element<HTMLButtonElement>("accept-reference-button");
const followUpButton = element<HTMLButtonElement>("follow-up-button");
const conditionAcceptButton = element<HTMLButtonElement>("condition-accept-button");
const conditionDismissButton = element<HTMLButtonElement>("condition-dismiss-button");
const conditionEvidenceView: ConditionEvidenceViewElements = {
  card: element<HTMLElement>("condition-card"),
  summary: element<HTMLElement>("condition-card-summary"),
  rows: element<HTMLElement>("condition-card-rows"),
  reviewState: element<HTMLElement>("condition-review-state"),
  acceptButton: conditionAcceptButton,
  dismissButton: conditionDismissButton
};
const faceOverlay = new FaceOverlayController(
  landmarkOverlay,
  faceMeshStatus
);
const liveVoiceVisualizer = new LiveVoiceVisualizer({
  levelGauge: element<HTMLCanvasElement>("voice-level-gauge"),
  pitchGauge: element<HTMLCanvasElement>("voice-pitch-gauge"),
  energyCanvas: element<HTMLCanvasElement>("voice-energy-chart"),
  pitchCanvas: element<HTMLCanvasElement>("voice-pitch-chart"),
  clarityCanvas: element<HTMLCanvasElement>("voice-clarity-chart"),
  state: element<HTMLElement>("voice-live-state"),
  level: element<HTMLElement>("voice-level-value"),
  pitch: element<HTMLElement>("voice-pitch-value"),
  snr: element<HTMLElement>("voice-snr-value"),
  confidence: element<HTMLElement>("voice-confidence-value"),
  agreement: element<HTMLElement>("voice-agreement-value"),
  quality: element<HTMLElement>("voice-quality-state")
});

consentText.textContent = AMBIENT_LOCAL_CONSENT_TEXT;

let workflow: AmbientWorkflowState = createAmbientWorkflowState();
let runtime = new CaptureRuntime();
const conditionDemo = new ConditionDemoController();
let sessionId = "";
let subjectRef = "";
let consentRecord: ConsentRecordV1 | null = null;
let sessionStartedAtIso = "";
let observationStartedAtIso = "";
let observationStartedAtPerformanceMs = 0;
let observationEndedAtIso = "";
let audioStream: MediaStream | null = null;
let videoStream: MediaStream | null = null;
let audioContext: AudioContext | null = null;
let voicePipeline: VoiceCapturePipeline | null = null;
let voiceStartupAbortController: AbortController | null = null;
let audioLaneTeardownPromise: Promise<void> | null = null;
let faceWorker: Worker | null = null;
let faceScheduler: LatestFrameScheduler<ImageBitmap> | null = null;
let facePump: VideoFramePump<ImageBitmap> | null = null;
let faceDisposedResolver: (() => void) | null = null;
let faceLaneTeardownPromise: Promise<void> | null = null;
let setupTimer: ReturnType<typeof setTimeout> | null = null;
let captureLimitTimer: ReturnType<typeof setTimeout> | null = null;
let clockTimer: ReturnType<typeof setInterval> | null = null;
let pendingDisposal: Promise<void> | null = null;
let quietStartedAtMs: number | null = null;
let quietCalibrationRms: number[] = [];
let noiseCalibrationDurationMs = 0;
let faceCalibrationFrames: AmbientFacialFrame[] = [];
let faceCalibration: FaceCalibration | null = null;
let lastFaceCalibrationAtMs: number | null = null;
let audioCalibrationResolved = false;
let faceCalibrationResolved = false;
let lastVoiceFrameTMs = 0;
let voiceObservationOriginMs = 0;
let audioProvenance: AudioPipelineProvenance | null = null;
let visualProvenance: VisualPipelineProvenance | null = null;
let processorProvenance: ProcessorProvenanceV1[] = [];
let voiceStaticAssetsPromise: Promise<ResolvedVoiceStaticAssets> | null = null;
let faceStaticAssetsPromise: Promise<ResolvedFaceStaticAssets> | null = null;
let journal: InMemoryEventJournal | null = null;
let audioLaneFailureReason: string | null = null;
let faceLaneFailureReason: string | null = null;
type LaneContractFailureReason = Extract<
  WithheldReasonCode,
  "modality-unavailable" | "processor-unavailable" | "asset-integrity-failed"
>;
let audioLaneContractFailureReason: LaneContractFailureReason | null = null;
let faceLaneContractFailureReason: LaneContractFailureReason | null = null;
let faceCalibrationGuidance: string | null = null;

function verifiedVoiceAssets(): Promise<ResolvedVoiceStaticAssets> {
  if (!voiceStaticAssetsPromise) {
    const pending = loadAndVerifyVoiceStaticAssets(document.baseURI);
    voiceStaticAssetsPromise = pending;
    void pending.catch(() => {
      if (voiceStaticAssetsPromise === pending) voiceStaticAssetsPromise = null;
    });
  }
  return voiceStaticAssetsPromise;
}

function verifiedFaceAssets(): Promise<ResolvedFaceStaticAssets> {
  if (!faceStaticAssetsPromise) {
    const pending = loadAndVerifyFaceStaticAssets(document.baseURI);
    faceStaticAssetsPromise = pending;
    void pending.catch(() => {
      if (faceStaticAssetsPromise === pending) faceStaticAssetsPromise = null;
    });
  }
  return faceStaticAssetsPromise;
}

function nowIso(): string {
  return new Date().toISOString();
}

function selectedAffectedSide(): "left" | "right" | null {
  if (affectedSideLeft.checked) return "left";
  if (affectedSideRight.checked) return "right";
  return null;
}

function updateStartButton(): void {
  startButton.disabled =
    !consentCheckbox.checked || selectedAffectedSide() === null;
}

function createSessionIdentity(): void {
  const affectedSide = selectedAffectedSide();
  if (!affectedSide) throw new Error("condition-demo-affected-side-required");
  sessionId = crypto.randomUUID();
  subjectRef = conditionDemo.context?.subjectRef ?? `subject-${crypto.randomUUID()}`;
  conditionDemo.startParticipant(subjectRef, affectedSide);
  sessionStartedAtIso = nowIso();
  consentRecord = {
    schemaVersion: "phenometric.consent-record.v1",
    consentId: `consent-${crypto.randomUUID()}`,
    sessionId,
    documentVersion: AMBIENT_LOCAL_PROTOCOL_PACK.consentDocument.version,
    documentSha256: AMBIENT_LOCAL_PROTOCOL_PACK.consentDocument.contentSha256,
    recordedAt: sessionStartedAtIso,
    scopes: {
      cameraCapture: true,
      microphoneCapture: true,
      localInMemoryAnalysis: true
    },
    localParticipantAssertion: true,
    withdrawnAt: null
  };
  journal = new InMemoryEventJournal({
    sessionId,
    subjectRef,
    protocolRef: AMBIENT_LOCAL_PROTOCOL_REF
  });
  journal.append({
    sessionId,
    subjectRef,
    protocolRef: AMBIENT_LOCAL_PROTOCOL_REF,
    actor: { kind: "application", id: "capture-web", version: "1.0.0" },
    type: "consent.recorded",
    stage: "requesting-permission",
    summary: "Local in-memory consent recorded.",
    payload: { consentId: consentRecord.consentId },
    evidenceRefs: []
  });
}

function clearTimers(): void {
  if (setupTimer !== null) clearTimeout(setupTimer);
  if (captureLimitTimer !== null) clearTimeout(captureLimitTimer);
  if (clockTimer !== null) clearInterval(clockTimer);
  setupTimer = null;
  captureLimitTimer = null;
  clockTimer = null;
}

function isCurrent(generation: number): boolean {
  return generation === workflow.generation &&
    ["requesting-permission", "calibrating", "observing"].includes(
      workflow.phase
    );
}

function laneCopy(state: AmbientWorkflowState["audioLane"]): {
  label: string;
  detail: string;
} {
  if (state === "requesting") return { label: "Requesting", detail: "Waiting for browser permission." };
  if (state === "calibrating") return { label: "Calibrating", detail: "Checking local signal quality." };
  if (state === "measurable") return { label: "Ready", detail: "Derived measurements may be available after sufficient evidence." };
  if (state === "not-measurable") return { label: "Not measurable", detail: "This lane will be shown as unavailable in the report." };
  return { label: "Off", detail: "The device is not active." };
}

function streamIsLive(stream: MediaStream | null): boolean {
  return Boolean(
    stream?.getTracks().some((track) => track.readyState === "live")
  );
}

function readableFailure(error: unknown, fallback: string): string {
  if (error instanceof DOMException) return `${error.name}: ${error.message}`;
  if (error instanceof Error) return error.message || error.name;
  return fallback;
}

function processorFailureReason(error: unknown): LaneContractFailureReason {
  return readableFailure(error, "processor-unavailable").startsWith("asset-")
    ? "asset-integrity-failed"
    : "processor-unavailable";
}

function disposeJournal(): void {
  journal?.dispose();
  journal = null;
}

function stopAudioLaneResources(): Promise<void> {
  if (audioLaneTeardownPromise) return audioLaneTeardownPromise;
  const stream = audioStream;
  const context = audioContext;
  const pipeline = voicePipeline;
  const startupAbortController = voiceStartupAbortController;
  audioStream = null;
  audioContext = null;
  voicePipeline = null;
  voiceStartupAbortController = null;
  audioLaneTeardownPromise = cleanupAudioLaneResources({
    cancelPendingStartup: () => startupAbortController?.abort(),
    stream,
    pipeline,
    audioContext: context
  });
  return audioLaneTeardownPromise;
}

function stopFaceLaneResources(generation: number): Promise<void> {
  if (faceLaneTeardownPromise) return faceLaneTeardownPromise;
  const stream = videoStream;
  const pump = facePump;
  const scheduler = faceScheduler;
  const worker = faceWorker;
  videoStream = null;
  facePump = null;
  faceScheduler = null;
  faceWorker = null;
  faceLaneTeardownPromise = cleanupFaceLaneResources({
    stream,
    pump,
    scheduler,
    video: cameraPreview,
    deactivate: () => {
      cameraPlaceholder.hidden = false;
    },
    disposeWorker: () => faceWorkerDisposed(generation, worker)
  });
  return faceLaneTeardownPromise;
}

function markLaneFailed(
  generation: number,
  lane: "audio" | "face",
  detail: string,
  reason: LaneContractFailureReason
): void {
  if (!isCurrent(generation)) return;
  const existingReason = lane === "audio"
    ? audioLaneContractFailureReason
    : faceLaneContractFailureReason;
  if (existingReason !== null) return;

  if (lane === "audio") {
    audioLaneFailureReason = detail;
    audioLaneContractFailureReason = reason;
    audioCalibrationResolved = true;
    // A track or processor can fail while addModule is still pending. Prevent
    // that stale startup from constructing a worker and audio graph after the
    // lane has already terminalized.
    liveVoiceVisualizer.setUnavailable();
    void stopAudioLaneResources();
  } else {
    faceLaneFailureReason = detail;
    faceLaneContractFailureReason = reason;
    faceCalibrationResolved = true;
    faceOverlay.markUnavailable();
    cameraPlaceholder.textContent = "Camera unavailable";
    void stopFaceLaneResources(generation);
  }
  const remainingLaneActive = lane === "audio"
    ? faceLaneContractFailureReason === null && streamIsLive(videoStream)
    : audioLaneContractFailureReason === null && streamIsLive(audioStream);
  dispatch({
    type: "lane-failed",
    generation,
    lane,
    remainingLaneActive,
    atMs: performance.now()
  });
}

function renderWorkflow(): void {
  const active = [
    "requesting-permission",
    "calibrating",
    "observing",
    "finalizing"
  ].includes(workflow.phase);
  const devicesLive = streamIsLive(audioStream) || streamIsLive(videoStream);
  welcomeView.hidden = workflow.phase !== "idle";
  captureView.hidden = !active;
  reportView.hidden = workflow.phase !== "report";
  if (workflow.phase === "discarded" || workflow.phase === "error") {
    messageView.hidden = false;
    messageTitle.textContent = workflow.phase === "error" ? "Session unavailable" : "Session discarded";
    messageDetail.textContent = workflow.phase === "error"
      ? "The local capture could not continue. No report was created; local cleanup is in progress."
      : "No report was created. Local session cleanup is in progress.";
  } else {
    messageView.hidden = true;
  }

  const labels: Record<AmbientWorkflowState["phase"], string> = {
    idle: "Ready",
    "requesting-permission": "Requesting devices",
    calibrating: "Technical setup",
    observing: "Ambient session",
    finalizing: "Finalizing locally",
    report: "Report ready",
    discarded: "Devices off",
    error: "Devices off"
  };
  phaseLabel.textContent = labels[workflow.phase];
  privacyState.textContent = devicesLive ? "Devices active · local only" : "Devices off";
  privacyState.classList.toggle("is-live", devicesLive);
  finishButton.disabled = workflow.phase !== "observing";

  const audioCopy = laneCopy(workflow.audioLane);
  const audioCapturingWithoutCalibration =
    workflow.phase === "observing" &&
    workflow.audioLane === "not-measurable" &&
    audioLaneContractFailureReason === null &&
    streamIsLive(audioStream);
  audioLaneState.textContent = audioCapturingWithoutCalibration
    ? "On"
    : audioCopy.label;
  audioLaneState.dataset.state = workflow.audioLane;
  audioLaneDetail.textContent =
    audioCapturingWithoutCalibration
      ? "Local capture is on, but technical calibration was incomplete; voice metrics may be Not measurable."
      : workflow.audioLane === "not-measurable" && audioLaneFailureReason
      ? `Unavailable: ${audioLaneFailureReason}. The camera lane can continue.`
      : audioCopy.detail;
  const faceCopy = laneCopy(workflow.faceLane);
  const faceCapturingWithoutCalibration =
    workflow.phase === "observing" &&
    workflow.faceLane === "not-measurable" &&
    faceLaneContractFailureReason === null &&
    streamIsLive(videoStream);
  faceLaneState.textContent = faceCapturingWithoutCalibration
    ? "On"
    : faceCopy.label;
  faceLaneState.dataset.state = workflow.faceLane;
  faceLaneDetail.textContent =
    faceCapturingWithoutCalibration
      ? `Local capture is on, but technical calibration was incomplete; face metrics may be Not measurable.${faceCalibrationGuidance ? ` ${faceCalibrationGuidance}` : ""}`
      : workflow.faceLane === "not-measurable" && faceLaneFailureReason
      ? `Unavailable: ${faceLaneFailureReason}. The microphone lane can continue.`
      : workflow.faceLane === "calibrating" && faceCalibrationGuidance
        ? faceCalibrationGuidance
      : faceCopy.detail;

  if (workflow.phase === "observing") {
    captureEyebrow.textContent = "Ambient session";
    captureTitle.textContent = "Continue the ordinary conversation";
    captureInstruction.textContent = "No exercises or scripted prompts are required. End whenever you are ready.";
    captureStatus.textContent = "Only derived, content-free engineering measurements are retained in session memory.";
  } else if (workflow.phase === "finalizing") {
    captureEyebrow.textContent = "Local finalization";
    captureTitle.textContent = "Turning devices off";
    captureInstruction.textContent = "The report appears only after camera, microphone, and processors have stopped.";
    captureStatus.textContent = "Clearing transient media buffers.";
  } else {
    captureEyebrow.textContent = "Technical setup";
    captureTitle.textContent = "Preparing local signals";
    captureInstruction.textContent = "Face the camera and allow a brief quiet moment for calibration.";
  }
}

function dispatch(event: AmbientWorkflowEvent): void {
  if (
    event.type === "calibration-resolved" &&
    event.measurable &&
    workflow.phase === "calibrating" &&
    journal
  ) {
    journal.append({
      sessionId,
      subjectRef,
      protocolRef: AMBIENT_LOCAL_PROTOCOL_REF,
      actor: {
        kind: "processor",
        id: event.lane === "audio" ? "voice-analysis" : "facial-analysis",
        version:
          event.lane === "audio"
            ? AMBIENT_VOICE_ALGORITHM_VERSION
            : AMBIENT_FACE_ALGORITHM_VERSION
      },
      type: "capture.lane.ready",
      stage: "calibrating",
      summary: `${event.lane === "audio" ? "Voice" : "Face"} lane passed technical calibration.`,
      payload: { modality: event.lane === "audio" ? "voice" : "face" },
      evidenceRefs: []
    });
  }
  const transition = reduceAmbientWorkflow(workflow, event);
  workflow = transition.state;
  renderWorkflow();
  for (const effect of transition.effects) void executeEffect(effect);
}

async function requestLane(
  generation: number,
  lane: "audio" | "face"
): Promise<void> {
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("media-devices-unavailable");
    const stream = await navigator.mediaDevices.getUserMedia(
      lane === "audio"
        ? {
            video: false,
            audio: {
              channelCount: 1,
              sampleRate: 48_000,
              echoCancellation: false,
              noiseSuppression: false,
              autoGainControl: false
            }
          }
        : {
            audio: false,
            video: {
              width: { ideal: 1280 },
              height: { ideal: 720 },
              frameRate: { ideal: 30 },
              facingMode: "user"
            }
          }
    );
    if (!isCurrent(generation)) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }
    if (lane === "audio") audioStream = stream;
    else videoStream = stream;
    // CaptureRuntime must own each granted stream as soon as that individual
    // permission resolves. The other lane may remain pending indefinitely.
    attachRuntimeHandles(generation);
    stream.getTracks().forEach((track) => {
      track.addEventListener("ended", () => {
        markLaneFailed(
          generation,
          lane,
          `${lane}-track-ended`,
          "modality-unavailable"
        );
      }, { once: true });
    });
    dispatch({
      type: "permission-resolved",
      generation,
      lane,
      available: true,
      atMs: performance.now()
    });
  } catch (error) {
    if (!isCurrent(generation)) return;
    const reason = readableFailure(error, `${lane}-permission-unavailable`);
    if (lane === "audio") {
      audioLaneFailureReason = reason;
      audioLaneContractFailureReason = "modality-unavailable";
      liveVoiceVisualizer.setUnavailable();
    }
    else {
      faceLaneFailureReason = reason;
      faceLaneContractFailureReason = "modality-unavailable";
    }
    dispatch({
      type: "permission-resolved",
      generation,
      lane,
      available: false,
      atMs: performance.now()
    });
  }
}

function median(values: readonly number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}

function handleVoiceFrame(frameInput: AmbientVoiceFrame): void {
  const frame: AmbientVoiceFrame = {
    ...frameInput,
    speechActive: frameInput.speechActive,
    periodic: frameInput.periodic,
    trackSegmentId: frameInput.trackSegmentId
  };
  liveVoiceVisualizer.push(frame);
  lastVoiceFrameTMs = frame.tMs;
  if (
    !audioCalibrationResolved &&
    (workflow.phase === "calibrating" || workflow.phase === "observing")
  ) {
    const quiet = !frame.speechActive && frame.blockGapMs <= 40 &&
      frame.clippedSampleFraction <= 0.01 && Math.abs(frame.dcOffset) <= 0.02;
    if (!quiet) {
      quietStartedAtMs = null;
      quietCalibrationRms = [];
      return;
    }
    quietStartedAtMs ??= frame.tMs;
    quietCalibrationRms.push(frame.rms);
    const duration = frame.tMs - quietStartedAtMs;
    if (duration >= 2_000) {
      audioCalibrationResolved = true;
      noiseCalibrationDurationMs = duration;
      voicePipeline?.setNoiseFloor(Math.max(0.0001, median(quietCalibrationRms)));
      dispatch({
        type: "calibration-resolved",
        generation: workflow.generation,
        lane: "audio",
        measurable: true,
        atMs: performance.now()
      });
    }
    return;
  }
  if (workflow.phase === "observing") {
    runtime.addVoiceFrame({
      ...frame,
      tMs: Math.max(0, frame.tMs - voiceObservationOriginMs),
      taskContext: AMBIENT_VOICE_TASK_CONTEXT
    });
  }
}

async function startAudioLane(generation: number): Promise<void> {
  if (!audioStream) {
    markLaneFailed(
      generation,
      "audio",
      "audio-track-unavailable",
      "modality-unavailable"
    );
    return;
  }
  let startupAbortController: AbortController | null = null;
  try {
    const assets = await verifiedVoiceAssets();
    if (
      !isCurrent(generation) ||
      audioLaneContractFailureReason !== null
    ) return;
    audioContext = new AudioContext({ sampleRate: 48_000, latencyHint: "interactive" });
    attachRuntimeHandles(generation);
    const track = audioStream.getAudioTracks()[0];
    const settings = track?.getSettings() ?? {};
    const browserProcessing = {
      echoCancellation: settings.echoCancellation ?? false,
      noiseSuppression: settings.noiseSuppression ?? false,
      autoGainControl: settings.autoGainControl ?? false
    };
    const captureSettings = requestedAudioCaptureSettings(
      settings.sampleRate ?? audioContext.sampleRate,
      settings.channelCount ?? 1,
      browserProcessing
    );
    startupAbortController = new AbortController();
    voiceStartupAbortController = startupAbortController;
    attachRuntimeHandles(generation);
    const startedPipeline = await startVoiceCapturePipeline({
      stream: audioStream,
      audioContext,
      captureSettings,
      captureEpoch: generation,
      taskContext: "quiet-calibration",
      workletUrl: assets.voiceWorkletUrl,
      startupSignal: startupAbortController.signal,
      callbacks: {
        onReady(provenance) {
          if (!isCurrent(generation) || audioLaneContractFailureReason !== null) {
            return;
          }
          audioProvenance = provenance;
          processorProvenance.push({
            modality: "voice",
            processorRef: provenance.processorRef,
            runtime: provenance.runtime,
            runtimeVersion: provenance.algorithmVersion,
            assetPath: assets.manifest.assets.voiceWorklet.path,
            assetSha256: assets.manifest.assets.voiceWorklet.sha256,
            assetIntegrityVerified: true
          });
        },
        onFrame(frame) {
          if (isCurrent(generation) && audioLaneContractFailureReason === null) {
            handleVoiceFrame(frame as AmbientVoiceFrame);
          }
        },
        onDiagnostics() {},
        onFailure(reason) {
          console.error("Audio processor failed:", reason);
          markLaneFailed(
            generation,
            "audio",
            reason,
            "processor-unavailable"
          );
        }
      }
    });
    if (voiceStartupAbortController === startupAbortController) {
      voiceStartupAbortController = null;
    }
    if (
      !isCurrent(generation) ||
      audioLaneContractFailureReason !== null
    ) {
      const latePipelineStop = startedPipeline.stop();
      const priorTeardown = audioLaneTeardownPromise ?? Promise.resolve();
      audioLaneTeardownPromise = Promise.all([
        priorTeardown,
        latePipelineStop
      ]).then(() => undefined);
      if (!isCurrent(generation)) {
        // CaptureRuntime may already be disposing. A late attachment makes the
        // local pipeline part of its pending-cleanup barrier as well.
        runtime.attach({
          stopVoicePipeline: () => audioLaneTeardownPromise!
        });
      }
      await audioLaneTeardownPromise;
      return;
    }
    voicePipeline = startedPipeline;
  } catch (error) {
    if (voiceStartupAbortController === startupAbortController) {
      voiceStartupAbortController = null;
    }
    if (!isCurrent(generation)) return;
    console.error("Audio lane setup failed:", error);
    markLaneFailed(
      generation,
      "audio",
      readableFailure(error, "audio-processor-unavailable"),
      processorFailureReason(error)
    );
  }
}

function handleFaceFrame(
  frameInput: AmbientFacialFrame,
  faceCount: number
): void {
  faceOverlay.updateFaceCount(faceCount);
  const frame: AmbientFacialFrame = {
    ...frameInput,
    faceCount,
    trackSegmentId: `face-${workflow.generation}`,
    qualityReasons:
      faceCount > 1
        ? [...new Set([...frameInput.qualityReasons, "multiple-faces" as const])]
        : frameInput.qualityReasons
  };
  if (
    !faceCalibrationResolved &&
    (workflow.phase === "calibrating" || workflow.phase === "observing")
  ) {
    if (faceCount !== 1) {
      faceCalibrationGuidance =
        faceCount > 1
          ? "Only one face can be visible during setup."
          : "Move fully into view and face the camera.";
    } else if (
      frame.pose === null ||
      Math.abs(frame.pose.yawDegrees) > 7 ||
      Math.abs(frame.pose.pitchDegrees) > 10 ||
      Math.abs(frame.pose.rollDegrees) > 5
    ) {
      faceCalibrationGuidance = "Face the camera directly and hold your head level.";
    } else if (frame.qualityReasons.includes("illumination-out-of-range")) {
      faceCalibrationGuidance = "Use even front lighting without strong backlight.";
    } else if (frame.qualityReasons.includes("blur")) {
      faceCalibrationGuidance = "Hold still briefly so the camera image is sharp.";
    } else if (
      frame.qualityReasons.includes("frame-rate-below-minimum") ||
      frame.qualityReasons.includes("too-many-skipped-frames")
    ) {
      faceCalibrationGuidance = "Camera analysis is stabilizing; keep this tab visible.";
    } else {
      faceCalibrationGuidance = "Signal detected. Hold this position briefly.";
    }
    faceLaneDetail.textContent = faceCalibrationGuidance;
    if (
      faceCount !== 1 ||
      (lastFaceCalibrationAtMs !== null &&
        frame.acquiredAtMs - lastFaceCalibrationAtMs > 200)
    ) {
      faceCalibrationFrames = [];
    }
    lastFaceCalibrationAtMs = frame.acquiredAtMs;
    if (faceCount === 1) faceCalibrationFrames.push(frame);
    const result = classifyFaceCalibration(faceCalibrationFrames);
    if (result.quality === "strong" && result.calibration) {
      faceCalibrationResolved = true;
      faceCalibration = result.calibration;
      faceCalibrationGuidance = null;
      dispatch({
        type: "calibration-resolved",
        generation: workflow.generation,
        lane: "face",
        measurable: true,
        atMs: performance.now()
      });
    }
    return;
  }
  if (workflow.phase === "observing") runtime.addFaceFrame(frame);
}

function faceWorkerDisposed(
  generation: number,
  worker: Worker | null
): Promise<void> {
  faceOverlay.clear();
  if (!worker) {
    faceOverlay.releaseWorker();
    return Promise.resolve();
  }
  const acknowledgement = new Promise<void>((resolve) => {
    faceDisposedResolver = resolve;
    try {
      worker.postMessage(
        visualWorkerMessage({ type: "dispose", captureEpoch: generation })
      );
    } catch {
      resolve();
    }
  });
  return withTimeout(acknowledgement, 500, () => undefined)
    .catch(() => undefined)
    .then(() => {
      worker.terminate();
      faceDisposedResolver = null;
      faceOverlay.releaseWorker();
    });
}

async function startFaceLane(generation: number): Promise<void> {
  if (!videoStream) {
    markLaneFailed(
      generation,
      "face",
      "face-track-unavailable",
      "modality-unavailable"
    );
    return;
  }
  try {
    const assets = await verifiedFaceAssets();
    if (
      !isCurrent(generation) ||
      faceLaneContractFailureReason !== null
    ) return;
    const stream = videoStream;
    if (!stream) return;
    cameraPreview.srcObject = stream;
    await cameraPreview.play();
    if (
      !isCurrent(generation) ||
      faceLaneContractFailureReason !== null
    ) return;
    cameraPlaceholder.hidden = true;
    const track = stream.getVideoTracks()[0];
    const settings = track?.getSettings() ?? {};
    const captureSettings = createVideoCaptureSettings({
      width: settings.width ?? (cameraPreview.videoWidth || 1280),
      height: settings.height ?? (cameraPreview.videoHeight || 720),
      frameRate: settings.frameRate,
      facingMode: settings.facingMode
    });
    let workerReady = false;
    let rejectWorkerReady: ((reason?: unknown) => void) | null = null;
    const ready = new Promise<void>((resolve, reject) => {
      rejectWorkerReady = reject;
      const worker = new Worker(new URL("./face-worker.ts", import.meta.url), {
        type: "module"
      });
      faceWorker = worker;
      attachRuntimeHandles(generation);
      worker.addEventListener("error", () => {
        markLaneFailed(
          generation,
          "face",
          "face-worker-unavailable",
          "processor-unavailable"
        );
        reject(new Error("face-worker-unavailable"));
      }, { once: true });
      worker.addEventListener("message", (event: MessageEvent<VisualWorkerResponse>) => {
        const message = event.data;
        if (
          message.schemaVersion !== VISUAL_WORKER_MESSAGE_VERSION ||
          message.captureEpoch !== generation
        ) return;
        if (message.type === "ready") {
          workerReady = true;
          if (!isCurrent(generation) || faceLaneContractFailureReason !== null) {
            resolve();
            return;
          }
          visualProvenance = message.provenance;
          processorProvenance.push({
            modality: "face",
            processorRef: message.provenance.processorRef,
            runtime: message.provenance.runtime,
            runtimeVersion: message.provenance.mediaPipeVersion,
            assetPath: assets.manifest.assets.faceModel.path,
            assetSha256: assets.manifest.assets.faceModel.sha256,
            assetIntegrityVerified: true
          });
          resolve();
        } else if (
          message.type === "frame" &&
          isCurrent(generation) &&
          faceLaneContractFailureReason === null
        ) {
          faceScheduler?.accept({
            captureEpoch: message.captureEpoch,
            sequence: message.sequence,
            acquisitionTimestampMs: message.acquiredAtMs
          });
          handleFaceFrame(message.frame as AmbientFacialFrame, message.faceCount);
        } else if (message.type === "overlay-status") {
          faceOverlay.acknowledge(message.captureEpoch, message.attached);
        } else if (message.type === "discarded") {
          faceScheduler?.discard({
            captureEpoch: message.captureEpoch,
            sequence: message.sequence,
            acquisitionTimestampMs: message.acquiredAtMs
          });
        } else if (message.type === "error") {
          if (message.sequence !== null && message.acquiredAtMs !== null) {
            faceScheduler?.fail({
              captureEpoch: message.captureEpoch,
              sequence: message.sequence,
              acquisitionTimestampMs: message.acquiredAtMs
            });
          }
          if (!message.recoverable) {
            markLaneFailed(
              generation,
              "face",
              message.code,
              "processor-unavailable"
            );
            reject(new Error(message.code));
          }
        } else if (message.type === "disposed") {
          faceDisposedResolver?.();
          faceDisposedResolver = null;
          if (!workerReady) rejectWorkerReady?.(new Error("face-worker-disposed"));
        }
      });
    });
    const initializedWorker = faceWorker;
    if (!initializedWorker) throw new Error("face-worker-unavailable");
    faceOverlay.attach(initializedWorker, generation, prefersReducedMotion);
    initializedWorker.postMessage(
      createVisualWorkerInitializeMessage(generation, captureSettings, {
        mediaPipeRootUrl: assets.mediaPipeRootUrl,
        modelUrl: assets.faceModelUrl,
        modelSha256: assets.manifest.assets.faceModel.sha256
      })
    );
    const readyBeforeTimeout = await withTimeout(
      ready.then(() => true),
      10_000,
      () => false
    );
    if (!readyBeforeTimeout) throw new Error("face-worker-ready-timeout");
    if (
      !isCurrent(generation) ||
      faceLaneContractFailureReason !== null ||
      !faceWorker
    ) return;
    faceScheduler = new LatestFrameScheduler<ImageBitmap>({
      captureEpoch: generation,
      onSubmit(scheduled: ScheduledVisualFrame<ImageBitmap>) {
        if (!faceWorker) throw new Error("face-worker-unavailable");
        const message = createVisualWorkerFrameMessage(scheduled, {
          tMs: Math.max(0, scheduled.acquisitionTimestampMs - observationStartedAtPerformanceMs),
          taskContext: AMBIENT_FACE_TASK_CONTEXT,
          calibration: faceCalibration
        });
        faceWorker.postMessage(message, [scheduled.frame]);
      }
    });
    facePump = new VideoFramePump<ImageBitmap>({
      source: cameraPreview,
      scheduler: faceScheduler,
      capture: async () => createImageBitmap(cameraPreview),
      taskContextAtAcquisition: () => AMBIENT_FACE_TASK_CONTEXT
    });
    facePump.start();
  } catch (error) {
    if (!isCurrent(generation)) return;
    console.error("Face lane setup failed:", error);
    markLaneFailed(
      generation,
      "face",
      readableFailure(error, "face-processor-unavailable"),
      processorFailureReason(error)
    );
  }
}

function attachRuntimeHandles(generation: number): void {
  const streams: MediaStream[] = [];
  if (audioStream) streams.push(audioStream);
  if (videoStream) streams.push(videoStream);
  runtime.attach({
    cancelPendingStartup: () => voiceStartupAbortController?.abort(),
    stopFacePump: () => facePump?.stop(),
    stopVoicePipeline: () => stopAudioLaneResources(),
    disposeFaceWorker: () => stopFaceLaneResources(generation),
    streams,
    video: cameraPreview,
    disconnectAudio: () => undefined,
    ...(audioContext ? { audioContext } : {}),
    cancelTimers: clearTimers
  });
}

async function beginCalibration(generation: number): Promise<void> {
  const modalities = [
    ...(workflow.audioLane === "calibrating" ? ["voice" as const] : []),
    ...(workflow.faceLane === "calibrating" ? ["face" as const] : [])
  ];
  if (journal && modalities.length > 0) {
    journal.append({
      sessionId,
      subjectRef,
      protocolRef: AMBIENT_LOCAL_PROTOCOL_REF,
      actor: { kind: "application", id: "capture-web", version: "1.0.0" },
      type: "capture.calibration.started",
      stage: "calibrating",
      summary: "Technical calibration started for available local signals.",
      payload: { modalities },
      evidenceRefs: []
    });
  }
  attachRuntimeHandles(generation);
  setupTimer = setTimeout(() => {
    if (!isCurrent(generation)) return;
    dispatch({ type: "setup-timeout", generation, atMs: performance.now() });
  }, AMBIENT_SETUP_TIMEOUT_MS);
  const starts: Promise<void>[] = [];
  if (workflow.audioLane === "calibrating") starts.push(startAudioLane(generation));
  if (workflow.faceLane === "calibrating") starts.push(startFaceLane(generation));
  await Promise.allSettled(starts);
  if (isCurrent(generation)) attachRuntimeHandles(generation);
}

function beginObservation(generation: number): void {
  if (!isCurrent(generation)) return;
  if (setupTimer !== null) clearTimeout(setupTimer);
  setupTimer = null;
  observationStartedAtIso = nowIso();
  observationStartedAtPerformanceMs = performance.now();
  voiceObservationOriginMs = lastVoiceFrameTMs;
  voicePipeline?.setTask(AMBIENT_VOICE_TASK_CONTEXT);
  journal?.append({
    sessionId,
    subjectRef,
    protocolRef: AMBIENT_LOCAL_PROTOCOL_REF,
    actor: { kind: "application", id: "capture-web", version: "1.0.0" },
    type: "capture.started",
    stage: "observing",
    summary: "Ambient local observation started.",
    payload: { startedAt: observationStartedAtIso },
    evidenceRefs: []
  });
  updateClock();
  clockTimer = setInterval(updateClock, 250);
  captureLimitTimer = setTimeout(() => {
    dispatch({ type: "capture-limit-reached", generation });
  }, AMBIENT_CAPTURE_LIMIT_MS);
}

function updateClock(): void {
  const elapsed = workflow.phase === "observing"
    ? Math.min(AMBIENT_CAPTURE_LIMIT_MS, performance.now() - observationStartedAtPerformanceMs)
    : 0;
  const totalSeconds = Math.floor(elapsed / 1_000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  sessionClock.textContent = `${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
  sessionClock.dateTime = `PT${totalSeconds}S`;
}

function clearSessionReferences(): void {
  audioStream = null;
  videoStream = null;
  audioContext = null;
  voicePipeline = null;
  voiceStartupAbortController = null;
  audioLaneTeardownPromise = null;
  faceWorker = null;
  faceScheduler = null;
  facePump = null;
  faceDisposedResolver = null;
  faceLaneTeardownPromise = null;
  audioProvenance = null;
  visualProvenance = null;
  sessionId = "";
  subjectRef = "";
  consentRecord = null;
  sessionStartedAtIso = "";
  observationStartedAtIso = "";
  observationStartedAtPerformanceMs = 0;
  observationEndedAtIso = "";
  quietStartedAtMs = null;
  quietCalibrationRms = [];
  noiseCalibrationDurationMs = 0;
  faceCalibrationFrames = [];
  faceCalibration = null;
  lastFaceCalibrationAtMs = null;
  audioCalibrationResolved = false;
  faceCalibrationResolved = false;
  lastVoiceFrameTMs = 0;
  voiceObservationOriginMs = 0;
  processorProvenance = [];
  audioLaneFailureReason = null;
  faceLaneFailureReason = null;
  audioLaneContractFailureReason = null;
  faceLaneContractFailureReason = null;
  faceCalibrationGuidance = null;
}

async function finalizeSession(generation: number): Promise<void> {
  liveVoiceVisualizer.reset();
  faceOverlay.clear();
  journal?.append({
    sessionId,
    subjectRef,
    protocolRef: AMBIENT_LOCAL_PROTOCOL_REF,
    actor: { kind: "application", id: "capture-web", version: "1.0.0" },
    type: "capture.finalizing",
    stage: "finalizing",
    summary: "Capture stopped before deterministic local finalization.",
    payload: {
      reason:
        workflow.terminalReason === "capture-limit-reached"
          ? "maximum-duration"
          : "manual"
    },
    evidenceRefs: []
  });
  observationEndedAtIso = nowIso();
  const durationMs = Math.min(
    AMBIENT_CAPTURE_LIMIT_MS,
    Math.max(0, performance.now() - observationStartedAtPerformanceMs)
  );
  let disposal: Promise<DerivedCaptureSnapshot | null> | null =
    runtime.dispose(true);
  await Promise.all([
    audioLaneTeardownPromise ?? Promise.resolve(),
    faceLaneTeardownPromise ?? Promise.resolve()
  ]);
  let snapshot: DerivedCaptureSnapshot | null = await disposal;
  disposal = null;
  if (generation !== workflow.generation || workflow.phase !== "finalizing") return;
  if (!snapshot || !consentRecord) {
    disposeJournal();
    conditionDemo.clear();
    dispatch({ type: "finalization-failed", generation, reason: "derived-snapshot-unavailable" });
    return;
  }
  try {
    const observation = buildAmbientObservation({
      sessionId,
      subjectRef,
      consent: consentRecord,
      startedAt: observationStartedAtIso || sessionStartedAtIso,
      endedAt: observationEndedAtIso,
      durationMs,
      voiceFrames: snapshot.voice as AmbientVoiceFrame[],
      faceFrames: snapshot.face as AmbientFacialFrame[],
      noiseCalibrationDurationMs,
      faceCalibration: faceCalibration
        ? {
            durationMs: faceCalibration.durationMs,
            baselineBoxWidthPixels: faceCalibration.baselineBoxWidthPixels,
            baselineBoxHeightPixels: faceCalibration.baselineBoxHeightPixels
          }
        : null,
      voiceLaneAvailable: audioLaneContractFailureReason === null,
      faceLaneAvailable: faceLaneContractFailureReason === null,
      voiceLaneFailureReason: audioLaneContractFailureReason,
      faceLaneFailureReason: faceLaneContractFailureReason,
      processors: processorProvenance
    });
    // ObservationV3 is now the only retained source artifact. Explicitly drop
    // the last local binding to derived frame arrays before report/card work or
    // any result view is rendered.
    snapshot = null;
    for (const outcome of observation.metricOutcomes) {
      if (outcome.status === "measured") {
        const measurement = observation.measurements.find(
          (candidate) => candidate.aggregateId === outcome.aggregateId
        );
        if (!measurement) throw new Error(`Missing measurement for ${outcome.metricCode}`);
        journal?.append({
          sessionId,
          subjectRef,
          protocolRef: AMBIENT_LOCAL_PROTOCOL_REF,
          actor: {
            kind: "processor",
            id: outcome.modality === "voice" ? "voice-analysis" : "facial-analysis",
            version: outcome.algorithmVersion
          },
          type: "measurement.recorded",
          stage: "finalizing",
          summary: `${outcome.label} produced a technically qualified measurement.`,
          payload: {
            measurementId: measurement.measurementId,
            aggregateId: outcome.aggregateId,
            metricCode: outcome.metricCode
          },
          evidenceRefs: outcome.evidence.refs
        });
      } else {
        journal?.append({
          sessionId,
          subjectRef,
          protocolRef: AMBIENT_LOCAL_PROTOCOL_REF,
          actor: {
            kind: "processor",
            id: outcome.modality === "voice" ? "voice-analysis" : "facial-analysis",
            version: outcome.algorithmVersion
          },
          type: "measurement.withheld",
          stage: "finalizing",
          summary: `${outcome.label} was not measurable under the protocol contract.`,
          payload: {
            outcomeId: outcome.outcomeId,
            aggregateId: outcome.aggregateId,
            metricCode: outcome.metricCode,
            reasonCode: outcome.reasonCode
          },
          evidenceRefs: outcome.evidence.refs
        });
      }
    }
    journal?.append({
      sessionId,
      subjectRef,
      protocolRef: AMBIENT_LOCAL_PROTOCOL_REF,
      actor: { kind: "application", id: "report-builder", version: "1.0.0" },
      type: "observation.created",
      stage: "finalizing",
      summary: "Validated ObservationV3 created in session memory.",
      payload: { observationId: observation.observationId },
      evidenceRefs: []
    });
    const report = buildPostEncounterReport(
      observation,
      AMBIENT_LOCAL_PROTOCOL_PACK,
      {
        generatedAt: observationEndedAtIso,
        events: [...(journal?.snapshot() ?? [])]
      }
    );
    journal?.append({
      sessionId,
      subjectRef,
      protocolRef: AMBIENT_LOCAL_PROTOCOL_REF,
      actor: { kind: "application", id: "report-builder", version: "1.0.0" },
      type: "report.created",
      stage: "report",
      summary: "Deterministic post-encounter report created in session memory.",
      payload: { reportId: report.reportId, observationId: observation.observationId },
      evidenceRefs: []
    });
    conditionDemo.recordFinalizedObservation(observation);
    const acceptedReference = conditionDemo.acceptedReference;
    const currentContext = conditionDemo.context;
    if (
      acceptedReference &&
      currentContext &&
      acceptedReference.observation.observationId !== observation.observationId
    ) {
      const comparison = comparePreviousVisit({
        currentContext,
        acceptedReference,
        currentObservation: observation
      });
      const card = buildConditionEvidenceCard({
        comparison,
        generatedAt: observationEndedAtIso
      });
      conditionDemo.recordCard(card);
    }
    disposeJournal();
    renderReport(report);
    clearSessionReferences();
    dispatch({ type: "finalization-completed", generation });
  } catch (error) {
    console.error(error);
    disposeJournal();
    clearSessionReferences();
    conditionDemo.clear();
    dispatch({ type: "finalization-failed", generation, reason: "report-validation-failed" });
  }
}

async function discardSession(): Promise<void> {
  liveVoiceVisualizer.reset();
  faceOverlay.clear();
  const reason = workflow.terminalReason === "consent-withdrawn"
    ? "consent-withdrawn"
    : workflow.terminalReason === "page-hidden" || workflow.terminalReason === "page-unloaded"
      ? "document-hidden"
      : "user-cancelled";
  if (journal && !journal.disposed) {
    journal.append({
      sessionId,
      subjectRef,
      protocolRef: AMBIENT_LOCAL_PROTOCOL_REF,
      actor: { kind: "application", id: "capture-web", version: "1.0.0" },
      type: "capture.discarded",
      stage: "discarded",
      summary: "Session discarded and in-memory data cleared.",
      payload: { reason },
      evidenceRefs: []
    });
  }
  const disposal = runtime.dispose(false);
  await Promise.all([
    audioLaneTeardownPromise ?? Promise.resolve(),
    faceLaneTeardownPromise ?? Promise.resolve()
  ]);
  await disposal;
  disposeJournal();
  clearSessionReferences();
  conditionDemo.clear();
  if (workflow.phase === "discarded" || workflow.phase === "error") {
    messageDetail.textContent =
      "No report was created. Devices are off and all local session data was cleared.";
  }
}

async function executeEffect(effect: AmbientWorkflowEffect): Promise<void> {
  if (effect.type === "request-media") {
    void requestLane(effect.generation, "audio");
    void requestLane(effect.generation, "face");
  } else if (effect.type === "begin-calibration") {
    await beginCalibration(effect.generation);
  } else if (effect.type === "begin-observation") {
    beginObservation(effect.generation);
  } else if (effect.type === "finalize") {
    await finalizeSession(effect.generation);
  } else if (effect.type === "dispose") {
    const disposal = discardSession();
    pendingDisposal = disposal;
    try {
      await disposal;
    } finally {
      if (pendingDisposal === disposal) pendingDisposal = null;
    }
  }
}

function formatValue(value: number, unit: string): string {
  if (unit === "ratio") return value.toFixed(3);
  if (unit.includes("second") || unit.includes("minute") || unit === "Hz") {
    return value.toFixed(2);
  }
  return value.toFixed(3);
}

function appendTraceItem(container: HTMLElement, label: string, value: string): void {
  const item = document.createElement("div");
  const name = document.createElement("span");
  const code = document.createElement("code");
  name.textContent = label;
  code.textContent = value;
  item.append(name, code);
  container.append(item);
}

function renderConditionPanel(): void {
  const context = conditionDemo.context;
  const reference = conditionDemo.acceptedReference;
  const observation = conditionDemo.latestObservation;
  const card = conditionDemo.latestCard;
  conditionSideBadge.textContent = context
    ? `Asserted affected side · subject-${context.assertedAffectedSide.side}`
    : "Affected side not set";

  if (!context || !observation) {
    conditionStatus.textContent =
      "Complete a live session before creating a comparison reference.";
    acceptReferenceButton.hidden = true;
    followUpButton.hidden = true;
    clearConditionEvidenceCard(conditionEvidenceView);
    return;
  }

  if (!reference) {
    conditionStatus.textContent =
      "This live session is ready to be explicitly accepted as the page-memory comparison reference.";
    acceptReferenceButton.hidden = false;
    followUpButton.hidden = true;
    clearConditionEvidenceCard(conditionEvidenceView);
    return;
  }

  acceptReferenceButton.hidden = true;
  if (observation.observationId === reference.observation.observationId) {
    conditionStatus.textContent =
      "Reference accepted in page memory. Capture a second live session to calculate raw compatible-session differences.";
    followUpButton.hidden = false;
    clearConditionEvidenceCard(conditionEvidenceView);
    return;
  }

  followUpButton.hidden = true;
  if (!card) {
    conditionStatus.textContent =
      "No condition comparison card was produced for this session.";
    clearConditionEvidenceCard(conditionEvidenceView);
    return;
  }

  conditionStatus.textContent =
    "Follow-up comparison ready. Inspect measured, withheld, and incompatible rows before reviewing this research card.";
  renderConditionEvidenceCard(conditionEvidenceView, card);
}

function renderReport(report: PostEncounterReportV1): void {
  reportBoundary.textContent = `${REPORT_BOUNDARY_STATEMENT} Prototype measurements are not clinically validated.`;
  reportSource.textContent = REPORT_SOURCE_DISCLOSURE;
  reportSections.replaceChildren();
  for (const section of report.sections) {
    const card = document.createElement("article");
    card.className = "report-section";
    card.dataset.section = section.sectionId;
    const heading = document.createElement("h2");
    heading.textContent = section.label;
    card.append(heading);
    if (section.sectionId === "capture-quality") {
      const facts = document.createElement("div");
      facts.className = "quality-facts";
      for (const fact of section.qualityFacts) {
        const node = document.createElement("div");
        node.className = "quality-fact";
        const label = document.createElement("span");
        label.textContent = fact.label;
        const value = document.createElement("strong");
        value.textContent = `${fact.value}${fact.unit ? ` ${fact.unit}` : ""}`;
        node.append(label, value);
        facts.append(node);
      }
      card.append(facts);
    } else {
      const list = document.createElement("div");
      list.className = "metric-list";
      for (const outcome of section.outcomes) {
        const row = document.createElement("article");
        row.className = "metric-row";
        row.dataset.metricCode = outcome.metricCode;
        const metricLabel = document.createElement("div");
        metricLabel.className = "metric-label";
        const label = document.createElement("strong");
        label.textContent = outcome.label;
        const context = document.createElement("span");
        context.textContent = outcome.context;
        metricLabel.append(label, context);
        const metricValue = document.createElement("div");
        metricValue.className = `metric-value${outcome.status === "withheld" ? " is-withheld" : ""}`;
        const value = document.createElement("strong");
        const unit = document.createElement("span");
        if (outcome.status === "measured") {
          value.textContent = formatValue(outcome.value, outcome.unit);
          unit.textContent = outcome.unit;
        } else {
          value.textContent = "Not measurable";
          unit.textContent = outcome.detail;
        }
        metricValue.append(value, unit);
        const evidence = document.createElement("div");
        evidence.className = "metric-evidence";
        evidence.textContent = `${(outcome.evidence.eligibleDurationMs / 1_000).toFixed(1)} s eligible · ${outcome.evidence.sampleCount} samples · ${outcome.evidence.windowCount} windows`;
        const details = document.createElement("details");
        details.className = "metric-details";
        const summary = document.createElement("summary");
        summary.textContent = "Measurement details";
        const trace = document.createElement("div");
        trace.className = "trace-grid";
        appendTraceItem(trace, "Aggregate ID", outcome.aggregateId);
        appendTraceItem(trace, "Algorithm", outcome.algorithmVersion);
        appendTraceItem(trace, "Processor", outcome.processorRef);
        appendTraceItem(trace, "Track segment", outcome.trackSegmentId);
        appendTraceItem(trace, "Technical quality", outcome.technicalQualityScore === null ? "Not available" : outcome.technicalQualityScore.toFixed(3));
        appendTraceItem(trace, "Technical dispersion", outcome.technicalDispersion === null ? "Not available" : outcome.technicalDispersion.toFixed(3));
        appendTraceItem(trace, "Evidence refs", outcome.evidence.refs.map((ref) => ref.kind === "event" ? ref.eventId : ref.kind === "window" ? ref.windowId : ref.kind === "measurement" ? ref.measurementId : ref.aggregateId).join(", "));
        if (outcome.status === "withheld") appendTraceItem(trace, "Withheld reason", outcome.reasonCode);
        details.append(summary, trace);
        row.append(metricLabel, metricValue, evidence, details);
        list.append(row);
      }
      card.append(list);
    }
    reportSections.append(card);
  }
  renderConditionPanel();
}

async function resetApplication(preserveConditionDemo = false): Promise<void> {
  const disposal = pendingDisposal;
  if (disposal) {
    resetButton.disabled = true;
    messageResetButton.disabled = true;
    try {
      await disposal;
    } finally {
      resetButton.disabled = false;
      messageResetButton.disabled = false;
    }
  }
  if (!preserveConditionDemo) conditionDemo.clear();
  disposeJournal();
  clearTimers();
  workflow = reduceAmbientWorkflow(workflow, { type: "reset" }).state;
  runtime = new CaptureRuntime();
  sessionId = "";
  subjectRef = "";
  consentRecord = null;
  sessionStartedAtIso = "";
  observationStartedAtIso = "";
  observationStartedAtPerformanceMs = 0;
  observationEndedAtIso = "";
  quietStartedAtMs = null;
  quietCalibrationRms = [];
  noiseCalibrationDurationMs = 0;
  faceCalibrationFrames = [];
  faceCalibration = null;
  lastFaceCalibrationAtMs = null;
  audioCalibrationResolved = false;
  faceCalibrationResolved = false;
  lastVoiceFrameTMs = 0;
  voiceObservationOriginMs = 0;
  processorProvenance = [];
  audioLaneFailureReason = null;
  faceLaneFailureReason = null;
  audioLaneContractFailureReason = null;
  faceLaneContractFailureReason = null;
  faceCalibrationGuidance = null;
  liveVoiceVisualizer.reset();
  faceOverlay.resetCanvas();
  consentCheckbox.checked = false;
  const retainedSide = conditionDemo.context?.assertedAffectedSide.side ?? null;
  affectedSideLeft.checked = retainedSide === "left";
  affectedSideRight.checked = retainedSide === "right";
  affectedSideFieldset.disabled = retainedSide !== null;
  updateStartButton();
  cameraPlaceholder.hidden = false;
  cameraPlaceholder.textContent = "Camera is preparing";
  sessionClock.textContent = "00:00";
  reportSections.replaceChildren();
  clearConditionEvidenceCard(conditionEvidenceView);
  renderWorkflow();
}

consentCheckbox.addEventListener("change", () => {
  dispatch({ type: "consent-changed", consented: consentCheckbox.checked });
  updateStartButton();
});

affectedSideLeft.addEventListener("change", updateStartButton);
affectedSideRight.addEventListener("change", updateStartButton);

consentForm.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!consentCheckbox.checked || selectedAffectedSide() === null) return;
  affectedSideFieldset.disabled = true;
  createSessionIdentity();
  journal?.append({
    sessionId,
    subjectRef,
    protocolRef: AMBIENT_LOCAL_PROTOCOL_REF,
    actor: { kind: "application", id: "capture-web", version: "1.0.0" },
    type: "capture.permission.requested",
    stage: "requesting-permission",
    summary: "Requested local camera and microphone permission.",
    payload: { modalities: ["voice", "face"] },
    evidenceRefs: []
  });
  dispatch({ type: "start-requested", atMs: performance.now() });
});

finishButton.addEventListener("click", () => {
  dispatch({ type: "finish-requested", generation: workflow.generation });
});

discardButton.addEventListener("click", () => {
  dispatch({ type: "discard-requested", reason: "participant-discarded" });
});

resetButton.addEventListener("click", () => {
  void resetApplication(false);
});
messageResetButton.addEventListener("click", () => {
  void resetApplication(false);
});

acceptReferenceButton.addEventListener("click", () => {
  try {
    conditionDemo.acceptLatestAsReference({
      referenceId: `reference-${crypto.randomUUID()}`,
      acceptedAt: nowIso()
    });
    renderConditionPanel();
  } catch (error) {
    console.error(error);
    conditionStatus.textContent =
      "The current observation could not be accepted as a comparison reference.";
  }
});

followUpButton.addEventListener("click", () => {
  try {
    conditionDemo.prepareFollowUp();
    void resetApplication(true);
  } catch (error) {
    console.error(error);
    conditionStatus.textContent =
      "A valid accepted reference is required before a follow-up capture.";
  }
});

conditionAcceptButton.addEventListener("click", () => {
  try {
    const card = conditionDemo.reviewLatestCard("accept-card", nowIso());
    renderConditionEvidenceCard(conditionEvidenceView, card);
  } catch (error) {
    console.error(error);
  }
});

conditionDismissButton.addEventListener("click", () => {
  try {
    const card = conditionDemo.reviewLatestCard("dismiss-card", nowIso());
    renderConditionEvidenceCard(conditionEvidenceView, card);
  } catch (error) {
    console.error(error);
  }
});

document.addEventListener("visibilitychange", () => {
  if (!document.hidden) return;
  const phaseBeforeVisibilityLoss = workflow.phase;
  conditionDemo.clear();
  dispatch({ type: "visibility-lost" });
  if (phaseBeforeVisibilityLoss === "idle" || phaseBeforeVisibilityLoss === "report") {
    void resetApplication(false);
  }
});

window.addEventListener("pagehide", () => {
  if (["requesting-permission", "calibrating", "observing", "finalizing"].includes(workflow.phase)) {
    dispatch({ type: "discard-requested", reason: "page-unloaded" });
  }
});

if (!window.isSecureContext) {
  captureStatus.textContent = "Camera and microphone require a secure context.";
}
renderWorkflow();
