import type {
  CaptureModality, CaptureSourceContext, DerivedBatch, DerivedPacket,
  EncounterCaptureAdapter, MediaPacket, MediaProcessor
} from "./media.js";
import {
  DerivedDataConsentV1Schema, ParticipantBindingV1Schema,
  type DerivedDataConsentV1, type ParticipantBindingV1
} from "@phenometrix/contracts";

/** Host assertions must originate in authenticated encounter/consent workflows. */
export interface CaptureAuthorization {
  encounterId: string;
  encounterActive: boolean;
  binding: ParticipantBindingV1;
  consent: DerivedDataConsentV1;
  modalities: readonly CaptureModality[];
}

export type CaptureReason = "consent-required" | "encounter-inactive" | "participant-unresolved"
  | "wrong-encounter" | "wrong-participant" | "attribution-ambiguous" | "modality-unavailable"
  | "stale-packet" | "invalid-packet" | "backpressure" | "processor-unavailable"
  | "processor-failed" | "multiple-faces" | "capture-stopped" | "transport-interrupted";

export interface CaptureEvent {
  schemaVersion: "phenometric.encounter-capture-event.v1";
  sequence: number;
  atMs: number;
  type: "started" | "stopped" | "withheld" | "accepted" | "availability";
  reason?: CaptureReason;
  modality?: CaptureModality;
  /** Derived summary only; never a packet, frame, name, or credential. */
  count?: number;
}

export interface AcceptedDerived {
  encounterId: string;
  subjectRef: string;
  platformParticipantId: string;
  captureEpoch: number;
  continuityId: string;
  modality: CaptureModality;
  acquiredAtMs: number;
  batch: DerivedBatch;
}

export interface CaptureControllerOptions {
  adapter: EncounterCaptureAdapter;
  processorFactory?: () => MediaProcessor;
  now?: () => number;
  maxPacketBytes?: number;
  maxPacketAgeMs?: number;
  processorTimeoutMs?: number;
  onDerived(value: AcceptedDerived): void;
  onEvent?(event: CaptureEvent): void;
}

export function authorizationFailure(a: CaptureAuthorization, now: number): CaptureReason | null {
  if (!a.encounterActive || !a.encounterId) return "encounter-inactive";
  if (!ParticipantBindingV1Schema.safeParse(a.binding).success || a.binding.status !== "verified"
    || a.binding.verificationMethod === "unverified" || a.binding.revokedAt !== null
    || Date.parse(a.binding.recordedAt) > now || a.binding.encounterId !== a.encounterId) return "participant-unresolved";
  const c = a.consent;
  if (!DerivedDataConsentV1Schema.safeParse(c).success || !c.permissions.capture || !c.permissions.derivedAnalysis
    || c.withdrawnAt !== null || Date.parse(c.grantedAt) > now || (c.expiresAt !== null && Date.parse(c.expiresAt) <= now)
    || c.scope.tenantId !== a.binding.scope.tenantId || c.scope.studyId !== a.binding.scope.studyId
    || c.scope.participantId !== a.binding.scope.participantId || !a.modalities.length
    || a.modalities.some(m => (m !== "face" && m !== "voice") || !c.permissions.modalities.includes(m))) return "consent-required";
  return null;
}

/** Host lifecycle, independent of page visibility and active-speaker events. */
export class EncounterCaptureController {
  private epoch = 0;
  private eventSequence = 0;
  private authorization: CaptureAuthorization | null = null;
  private phase: "idle" | "starting" | "active" | "stopped" = "idle";
  private processor: MediaProcessor | null = null;
  private readonly now: () => number;
  private readonly limits: { bytes: number; age: number; timeout: number };
  private startedAt = 0;
  private availability = { face: true, voice: true };
  private continuity = { face: 0, voice: 0 };
  private lastAt = { face: -Infinity, voice: -Infinity };
  private readonly pending = new Map<CaptureModality, { packet: MediaPacket; abort: AbortController; timer: ReturnType<typeof setTimeout> }>();
  private expiryTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(private readonly options: CaptureControllerOptions) {
    this.now = options.now ?? Date.now;
    this.limits = { bytes: options.maxPacketBytes ?? 4_194_304, age: options.maxPacketAgeMs ?? 2_000,
      timeout: options.processorTimeoutMs ?? 2_000 };
    if (Object.values(this.limits).some(v => !Number.isFinite(v) || v <= 0)) throw new Error("Invalid capture resource limit");
  }

  get status() { return this.phase; }
  get inFlightCount() { return this.pending.size; }

  async start(authorization: CaptureAuthorization): Promise<boolean> {
    if (this.phase === "starting" || this.phase === "active") throw new Error("Capture already active");
    const failure = authorizationFailure(authorization, this.now());
    if (failure) { this.emit("withheld", failure); return false; }
    const unsupported = authorization.modalities.find(modality => !this.options.adapter.capabilities.modalities.includes(modality));
    if (unsupported) { this.emit("withheld", "modality-unavailable", unsupported); return false; }
    this.authorization = structuredClone(authorization);
    this.startedAt = this.now();
    this.phase = "starting";
    const epoch = ++this.epoch;
    this.availability = { face: true, voice: true };
    this.lastAt = { face: -Infinity, voice: -Infinity };
    this.watchConsentExpiry();
    if (this.phase !== "starting" || epoch !== this.epoch) return false;
    const context: CaptureSourceContext = {
      encounterId: authorization.encounterId, platformParticipantId: authorization.binding.platformParticipantId,
      modalities: [...authorization.modalities], startedAtMs: this.startedAt
    };
    try {
      this.processor = this.options.processorFactory?.() ?? null;
      if (!this.authorization || authorizationFailure(this.authorization, this.now())) {
        this.stop("consent-required"); return false;
      }
      await this.options.adapter.start(context, {
        packet: p => { if (epoch !== this.epoch) this.release(p); else this.receive(p); },
        derived: p => { if (epoch === this.epoch) this.receiveDerived(p); },
        availability: (m, available) => { if (epoch === this.epoch) this.setAvailability(m, available); },
        interruption: () => { if (epoch === this.epoch) this.stop("transport-interrupted"); }
      });
      if (epoch !== this.epoch || this.phase !== "starting") return false;
      this.phase = "active";
      this.emit("started");
      return true;
    } catch {
      if (epoch === this.epoch) this.stop("transport-interrupted");
      return false;
    }
  }

  /** Any changed binding/consent requires fresh authorization and a new epoch. */
  updateAuthorization(next: CaptureAuthorization): void {
    if (!this.authorization) return;
    if (JSON.stringify(next) !== JSON.stringify(this.authorization)) {
      this.stop(authorizationFailure(next, this.now()) ?? "participant-unresolved");
    }
  }

  withdraw(): void { this.stop("consent-required"); }
  encounterEnded(): void { this.stop("encounter-inactive"); }
  /** Visibility is presentation state, deliberately not an encounter end event. */
  setPageVisibility(_visible: boolean): void {}

  setAvailability(modality: CaptureModality, available: boolean): void {
    if (this.availability[modality] === available) return;
    this.availability[modality] = available;
    this.breakContinuity(modality);
    this.emit("availability", available ? undefined : "modality-unavailable", modality);
  }

  stop(reason: CaptureReason = "capture-stopped"): void {
    if (this.phase === "stopped" || this.phase === "idle") return;
    this.phase = "stopped";
    if (this.expiryTimer) clearTimeout(this.expiryTimer);
    this.expiryTimer = null;
    ++this.epoch;
    this.breakContinuity("face"); this.breakContinuity("voice");
    try { this.options.adapter.stop(); } catch { /* Source failure cannot prevent remaining disposal. */ }
    try { this.processor?.dispose(); } catch { /* Dispose remaining controller-owned resources. */ }
    this.processor = null; this.authorization = null;
    this.emit("stopped", reason);
  }

  private gate(p: Omit<DerivedPacket, "batch">): CaptureReason | null {
    const a = this.authorization;
    if (this.phase !== "active" || !a) return "capture-stopped";
    const invalidAuthorization = authorizationFailure(a, this.now());
    if (invalidAuthorization) return invalidAuthorization;
    if (p.encounterId !== a.encounterId) return "wrong-encounter";
    if (p.attribution !== "individual-track") return "attribution-ambiguous";
    if (p.platformParticipantId !== a.binding.platformParticipantId) return "wrong-participant";
    if (!a.modalities.includes(p.modality) || !this.availability[p.modality]) return "modality-unavailable";
    if (!p.trackId || !Number.isSafeInteger(p.sequence) || p.sequence < 0 || !Number.isFinite(p.acquiredAtMs)) return "invalid-packet";
    if (p.acquiredAtMs < Math.max(this.startedAt, Date.parse(a.consent.grantedAt))
      || this.now() - p.acquiredAtMs > this.limits.age || p.acquiredAtMs > this.now() + 1_000
      || p.acquiredAtMs <= this.lastAt[p.modality]) return "stale-packet";
    return null;
  }

  private receive(packet: MediaPacket): void {
    let failure = this.gate(packet) ?? (packet.media.released || !Number.isFinite(packet.media.byteLength)
      || packet.media.byteLength <= 0 || packet.media.byteLength > this.limits.bytes ? "invalid-packet" : null)
      ?? (this.pending.has(packet.modality) ? "backpressure" : null);
    if (!failure) {
      try { if (!this.processor?.supports(packet)) failure = "processor-unavailable"; }
      catch { failure = "processor-failed"; }
    }
    if (failure) {
      this.release(packet);
      if (failure === "backpressure" || failure === "attribution-ambiguous") this.continuity[packet.modality]++;
      this.emit("withheld", failure, packet.modality); return;
    }
    const epoch = this.epoch;
    const continuity = this.continuity[packet.modality];
    const abort = new AbortController();
    const timer = setTimeout(() => { if (epoch === this.epoch) this.stop("processor-failed"); }, this.limits.timeout);
    this.pending.set(packet.modality, { packet, abort, timer });
    this.lastAt[packet.modality] = packet.acquiredAtMs;
    const context = { signal: abort.signal, captureEpoch: epoch,
      continuityId: this.continuityId(packet.modality), sessionStartedAtMs: this.startedAt };
    Promise.resolve().then(() => {
      if (abort.signal.aborted) return {};
      return this.processor!.process(packet, context);
    }).then(batch => {
      if (epoch === this.epoch && this.authorization && !authorizationFailure(this.authorization, this.now())
        && continuity === this.continuity[packet.modality] && !abort.signal.aborted) {
        this.accept(packet, batch, context.continuityId);
      }
    }).catch(() => { if (epoch === this.epoch) { this.breakContinuity(packet.modality); this.emit("withheld", "processor-failed", packet.modality); } })
      .finally(() => {
        clearTimeout(timer); this.release(packet);
        if (this.pending.get(packet.modality)?.packet === packet) this.pending.delete(packet.modality);
      });
  }

  private receiveDerived(packet: DerivedPacket): void {
    const failure = this.gate(packet);
    if (failure) { this.emit("withheld", failure, packet.modality); return; }
    this.lastAt[packet.modality] = packet.acquiredAtMs;
    this.accept(packet, packet.batch, this.continuityId(packet.modality));
  }

  private accept(packet: Omit<DerivedPacket, "batch">, batch: DerivedBatch, continuityId: string): void {
    const frames = packet.modality === "face" ? batch.faceFrames : batch.voiceFrames;
    if (!frames?.length || frames.length > 1_000 || (packet.modality === "face" && batch.voiceFrames?.length)
      || (packet.modality === "voice" && batch.faceFrames?.length)) { this.emit("withheld", "invalid-packet", packet.modality); return; }
    if (frames.some(frame => !Number.isFinite(frame.acquiredAtMs) || !Number.isFinite(frame.tMs)
      || frame.tMs < 0 || frame.acquiredAtMs < this.startedAt || frame.acquiredAtMs > packet.acquiredAtMs + 1_000
      || this.now() - frame.acquiredAtMs > this.limits.age
      || !frame.processorRef || frame.schemaVersion !== (packet.modality === "voice"
        ? "phenometric.voice-signal-frame.v1" : "phenometric.facial-kinematics-frame.v1"))) {
      this.emit("withheld", "invalid-packet", packet.modality); return;
    }
    if (batch.faceFrames?.some(f => f.faceCount !== 1)) {
      this.breakContinuity("face"); this.emit("withheld", "multiple-faces", "face"); return;
    }
    // Preserve worker-derived values; replace only capture identity/continuity supplied by this gate.
    const remap = <T extends { acquiredAtMs: number; tMs: number; captureEpoch: number; trackSegmentId?: string }>(f: T): T =>
      ({ ...f, captureEpoch: this.epoch, trackSegmentId: `${packet.trackId}:${continuityId}` });
    try { this.options.onDerived({ encounterId: this.authorization!.encounterId, subjectRef: this.authorization!.binding.scope.participantId,
      platformParticipantId: this.authorization!.binding.platformParticipantId, captureEpoch: this.epoch,
      continuityId, modality: packet.modality, acquiredAtMs: packet.acquiredAtMs,
      batch: { ...(batch.voiceFrames ? { voiceFrames: batch.voiceFrames.map(remap) } : {}),
        ...(batch.faceFrames ? { faceFrames: batch.faceFrames.map(remap) } : {}) } });
    } catch { this.stop("processor-failed"); return; }
    this.emit("accepted", undefined, packet.modality, frames.length);
  }

  private continuityId(modality: CaptureModality) { return `${this.epoch}:${modality}:${this.continuity[modality]}`; }
  private breakContinuity(modality: CaptureModality): void {
    this.continuity[modality]++;
    const pending = this.pending.get(modality);
    if (pending) { pending.abort.abort(); clearTimeout(pending.timer); this.release(pending.packet); this.pending.delete(modality); }
  }
  private release(packet: MediaPacket) { try { packet.media.release(); } catch { /* A failing native close must not interrupt other cleanup. */ } }
  private watchConsentExpiry(): void {
    const expiry = this.authorization?.consent.expiresAt;
    if (!expiry) return;
    const remaining = Date.parse(expiry) - this.now();
    if (remaining <= 0) { this.stop("consent-required"); return; }
    this.expiryTimer = setTimeout(() => this.watchConsentExpiry(), Math.min(2_147_483_647, remaining));
  }
  private emit(type: CaptureEvent["type"], reason?: CaptureReason, modality?: CaptureModality, count?: number): void {
    try { this.options.onEvent?.({ schemaVersion: "phenometric.encounter-capture-event.v1", sequence: ++this.eventSequence,
      atMs: this.now(), type, ...(reason ? { reason } : {}), ...(modality ? { modality } : {}), ...(count !== undefined ? { count } : {}) });
    } catch { /* A presentation callback must never interrupt cleanup. */ }
  }
}
