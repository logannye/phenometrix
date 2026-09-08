import { finalizeAmbientMetrics, AMBIENT_MAX_CAPTURE_DURATION_MS,
  type AmbientFacialFrame, type AmbientVoiceFrame, type AmbientFaceCalibration } from "@phenometrix/ambient-core";
import { AMBIENT_LOCAL_PROTOCOL_REF, DurableObservationV1Schema, calculateSha256Hex,
  type DurableObservationV1 } from "@phenometrix/contracts";
import { authorizationFailure, type AcceptedDerived, type CaptureAuthorization } from "./controller.js";

export interface AmbientWindowOptions {
  authorization: CaptureAuthorization;
  observationId: string;
  revisionId: string;
  startedAtMs: number;
  adapterId: string;
  adapterVersion: string;
  sourceKind: DurableObservationV1["capture"]["sourceKind"];
  deviceClass: string;
  clockUncertaintyMs: number;
  clockSource?: DurableObservationV1["capture"]["clockSource"];
  audioNoiseCalibration?: DurableObservationV1["capture"]["audioNoiseCalibration"];
  /** Hash of relevant acquisition settings and opaque device identity; never raw hardware IDs. */
  acquisitionFingerprint?: string;
  /** Browser hosts supply a dedicated worker; server workers may use the pure default. */
  finalizeMetrics?: (input: Parameters<typeof finalizeAmbientMetrics>[0], signal: AbortSignal) => Promise<ReturnType<typeof finalizeAmbientMetrics>>;
  maxFrames?: number;
}

/** Bounded transient derived primitives; only terminal, schema-checked measurements leave it. */
export class AmbientEncounterWindow {
  private voice: AmbientVoiceFrame[] = [];
  private face: AmbientFacialFrame[] = [];
  private noiseDurationMs = 0;
  private faceCalibration: AmbientFaceCalibration | null = null;
  private closed = false;
  private readonly finalizationAbort = new AbortController();
  private readonly maxFrames: number;
  private captureReadiness: Pick<DurableObservationV1["capture"], "clockUncertaintyMs" | "clockSource" | "audioNoiseCalibration">;
  constructor(private readonly options: AmbientWindowOptions) {
    if (authorizationFailure(options.authorization, options.startedAtMs) || !options.authorization.consent.permissions.derivedRetention)
      throw new Error("Durable observation authorization required");
    this.options = { ...options, authorization: structuredClone(options.authorization) };
    this.captureReadiness = { clockUncertaintyMs: options.clockUncertaintyMs,
      ...(options.clockSource ? { clockSource: structuredClone(options.clockSource) } : {}),
      ...(options.audioNoiseCalibration ? { audioNoiseCalibration: structuredClone(options.audioNoiseCalibration) } : {}) };
    this.maxFrames = options.maxFrames ?? 60_000;
    if (!Number.isSafeInteger(this.maxFrames) || this.maxFrames < 1) throw new Error("Invalid frame limit");
  }
  get frameCount() { return this.voice.length + this.face.length; }
  setCaptureReadiness(input: { clockUncertaintyMs?: number; clockSource?: DurableObservationV1["capture"]["clockSource"];
    audioNoiseCalibration?: DurableObservationV1["capture"]["audioNoiseCalibration"] | null }): void {
    if (this.closed) return;
    if (input.clockUncertaintyMs !== undefined) {
      if (!Number.isFinite(input.clockUncertaintyMs) || input.clockUncertaintyMs < 0) throw new Error("Invalid clock uncertainty");
      this.captureReadiness.clockUncertaintyMs = Math.max(this.captureReadiness.clockUncertaintyMs, input.clockUncertaintyMs);
    }
    if (input.clockSource) {
      if (this.captureReadiness.clockSource && (this.captureReadiness.clockSource.sourceId !== input.clockSource.sourceId
        || this.captureReadiness.clockSource.kind !== input.clockSource.kind))
        throw new Error("Clock source changed inside an observation");
      this.captureReadiness.clockSource = structuredClone(input.clockSource);
    }
    if (input.audioNoiseCalibration !== undefined) {
      if (input.audioNoiseCalibration === null) delete this.captureReadiness.audioNoiseCalibration;
      else this.captureReadiness.audioNoiseCalibration = structuredClone(input.audioNoiseCalibration);
    }
  }
  setCalibration(input: { noiseDurationMs?: number; face?: AmbientFaceCalibration | null }): void {
    if (this.closed) return;
    if (input.noiseDurationMs !== undefined) this.noiseDurationMs = input.noiseDurationMs;
    if (input.face !== undefined) this.faceCalibration = input.face;
  }
  append(value: AcceptedDerived): boolean {
    if (this.closed || value.encounterId !== this.options.authorization.encounterId
      || value.subjectRef !== this.options.authorization.binding.scope.participantId
      || value.platformParticipantId !== this.options.authorization.binding.platformParticipantId
      || !this.options.authorization.modalities.includes(value.modality)
      || (value.modality === "face" && !!value.batch.voiceFrames?.length)
      || (value.modality === "voice" && !!value.batch.faceFrames?.length)) return false;
    const added = (value.batch.voiceFrames?.length ?? 0) + (value.batch.faceFrames?.length ?? 0);
    if (this.frameCount + added > this.maxFrames) return false;
    const validTime = (frame: { acquiredAtMs: number; tMs: number }) =>
      Number.isFinite(frame.acquiredAtMs) && Number.isFinite(frame.tMs) && frame.tMs >= 0
      && frame.tMs <= AMBIENT_MAX_CAPTURE_DURATION_MS
      && Math.abs(frame.acquiredAtMs - this.options.startedAtMs - frame.tMs) <= 1;
    if (!(value.batch.voiceFrames ?? []).every(validTime) || !(value.batch.faceFrames ?? []).every(validTime)) return false;
    this.voice.push(...structuredClone(value.batch.voiceFrames ?? []));
    this.face.push(...structuredClone(value.batch.faceFrames ?? []));
    return true;
  }
  discard(): void { this.finalizationAbort.abort(); this.voice = []; this.face = []; this.faceCalibration = null; this.noiseDurationMs = 0; this.closed = true; }

  async finalize(endedAtMs: number, recordedAtMs = endedAtMs, externalSignal?: AbortSignal): Promise<DurableObservationV1> {
    if (this.closed) throw new Error("Ambient window already closed");
    if (!Number.isFinite(endedAtMs) || endedAtMs < this.options.startedAtMs
      || endedAtMs - this.options.startedAtMs > AMBIENT_MAX_CAPTURE_DURATION_MS) throw new Error("Use bounded ambient windows of at most five minutes");
    this.closed = true;
    const captureReadiness = structuredClone(this.captureReadiness);
    try {
      const protocol = AMBIENT_LOCAL_PROTOCOL_REF;
      const duration = endedAtMs - this.options.startedAtMs;
      // Facial source evidence uses complete fixed 5-second bins. A partial last
      // bin must not claim time after encounter end or a calibration boundary.
      const faceEndMs = Math.floor(duration / 5_000) * 5_000;
      const input = { identity: { sessionId: this.options.observationId,
        protocolVersion: protocol.version, protocolContentSha256: protocol.contentSha256,
        // The core's analysis clock and tMs are both relative to this bounded window.
        // Unix acquisition timestamps remain separate provenance, never the analysis origin.
        sessionStartedAtMs: 0 },
      // Core voice evidence ends one 10 ms hop beyond its last frame timestamp.
      voice: { frames: this.voice.filter(frame => frame.tMs + 10 <= duration), noiseCalibrationDurationMs: this.noiseDurationMs },
      face: { frames: this.face.filter(frame => frame.tMs < faceEndMs), calibration: this.faceCalibration } };
      const signal = externalSignal ? AbortSignal.any([externalSignal, this.finalizationAbort.signal]) : this.finalizationAbort.signal;
      signal.throwIfAborted();
      const result = this.options.finalizeMetrics ? await this.options.finalizeMetrics(input, signal) : finalizeAmbientMetrics(input);
      signal.throwIfAborted();
      // Release primitive arrays before asynchronous digest/delivery work.
      input.voice.frames = []; input.face.frames = [];
      this.voice = []; this.face = []; this.faceCalibration = null; this.noiseDurationMs = 0;
      const outcomes = result.outcomes.filter(outcome => this.options.authorization.modalities.includes(outcome.modality));
      const windows: DurableObservationV1["windows"] = [];
      const windowMap = new Map<string, string>();
      for (const outcome of outcomes) for (const source of outcome.evidence.sourceWindowRefs) {
        if (windowMap.has(source)) continue;
        const parts = source.split(":");
        const startMs = Number(parts.at(-2)); const endMs = Number(parts.at(-1));
        if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || startMs < 0 || endMs <= startMs || endMs > duration)
          throw new Error("Derived window outside encounter");
        const windowId = `window-${(await calculateSha256Hex(source)).slice(0, 24)}`;
        signal.throwIfAborted();
        windowMap.set(source, windowId);
        windows.push({ windowId, startMs, endMs, status: "eligible", reasonCodes: [] });
      }
      const processors = [...new Set(outcomes.flatMap(o => o.evidence.processorRefs))].sort();
      const processorFingerprint = await calculateSha256Hex(JSON.stringify({ processors,
        acquisition: this.options.acquisitionFingerprint ?? "acquisition-settings-unknown" }));
      signal.throwIfAborted();
      return DurableObservationV1Schema.parse({ schemaVersion: "phenometric.durable-observation.v1",
        observationId: this.options.observationId, revisionId: this.options.revisionId, supersedesRevisionId: null,
        scope: this.options.authorization.binding.scope, encounterId: this.options.authorization.encounterId,
        consentId: this.options.authorization.consent.consentId, bindingId: this.options.authorization.binding.bindingId,
        recordedAt: new Date(recordedAtMs).toISOString(), startedAt: new Date(this.options.startedAtMs).toISOString(),
        endedAt: new Date(endedAtMs).toISOString(), status: "available",
        measurementProtocolRef: { id: protocol.packId, version: protocol.version, contentSha256: protocol.contentSha256 },
        capture: { adapterId: this.options.adapterId, adapterVersion: this.options.adapterVersion,
          sourceKind: this.options.sourceKind, pipelineVersion: "ambient-bridge.1.1.0",
          processorFingerprint, deviceClass: this.options.deviceClass,
          ...captureReadiness, rawMediaRetained: false },
        windows,
        metrics: outcomes.map(o => ({ metricCode: o.code, modality: o.modality, unit: o.unit,
          context: o.identity.context, algorithmVersion: o.identity.algorithmVersion,
          processorRef: o.evidence.processorRefs.join("+") || "unavailable", status: o.status,
          value: o.status === "measured" ? o.value : null, reasonCodes: o.status === "measured" ? [] : [o.reasonCode],
          usableDurationMs: o.evidence.eligibleDurationMs, technicalQualityScore: o.technicalQualityScore,
          sourceWindowIds: o.evidence.sourceWindowRefs.map(ref => windowMap.get(ref)!), temporalSamples: [] })) });
    } finally { this.discard(); }
  }
}
