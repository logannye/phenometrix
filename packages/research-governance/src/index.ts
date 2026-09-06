/** Separate from routine capture. No built-in recording or media persistence. */
export interface ResearchScope { tenantId: string; studyId: string; participantId: string }
export interface ResearchConsent {
  id: string; scope: ResearchScope; validFrom: number; validUntil: number;
  withdrawnAt: number | null; retainClips: boolean; annotateClips: boolean;
}
export interface ResearchClip {
  id: string; scope: ResearchScope; consentId: string; observationId: string;
  createdAt: number; expiresAt: number; durationMs: number; capturedAt: number; endedAt: number; mimeType: string;
  status: "pending-upload" | "available" | "deletion-pending" | "deleted";
}
export interface ResearchClipPorts {
  /** Supply an encrypted store, isolated from clinical measurements. Keys are opaque UUIDs. */
  blobs: { put(key: string, bytes: Uint8Array, options: { expiresAt: number; signal: AbortSignal }): Promise<void>; get(key: string): Promise<Uint8Array>; delete(key: string): Promise<void> };
  metadata: { put(clip: ResearchClip): Promise<void>; get(id: string): Promise<ResearchClip | null>; list(scope: ResearchScope): Promise<ResearchClip[]> };
  consent(id: string): Promise<ResearchConsent | null>;
  observation(id: string): Promise<{ scope: ResearchScope; startedAt: number; endedAt: number } | null>;
  /** This callback must validate authenticated study/subject membership, not caller-supplied roles. */
  authorize(actorId: string, scope: ResearchScope, action: "retain" | "annotate" | "delete"): Promise<boolean>;
  audit(event: { action: "retained" | "read" | "deleted"; actorId: string; clipId: string; scope: ResearchScope; at: number }): Promise<void>;
}
export interface ResearchClipPolicy {
  enabled: boolean; maxBytes: number; maxDurationMs: number; retentionMs: number;
}
export const RESEARCH_CLIPS_DISABLED: ResearchClipPolicy = Object.freeze({
  enabled: false, maxBytes: 0, maxDurationMs: 0, retentionMs: 0
});

function matches(a: ResearchScope, b: ResearchScope): boolean {
  return a.tenantId === b.tenantId && a.studyId === b.studyId && a.participantId === b.participantId;
}
function deny(): never { throw new Error("research-clip-access-unavailable"); }

/**
 * A policy-enforcing optional media boundary. The service does not enable this
 * until institution-specific encrypted storage, consent and retention ports exist.
 * Accepted byte buffers transfer ownership here and are zeroed on every exit.
 */
export class ResearchClipVault {
  constructor(private ports: ResearchClipPorts, private policy: ResearchClipPolicy = RESEARCH_CLIPS_DISABLED,
    private now: () => number = Date.now) {
    if (policy.enabled && [policy.maxBytes, policy.maxDurationMs, policy.retentionMs].some(x => !Number.isSafeInteger(x) || x <= 0)) {
      throw new Error("invalid-research-retention-policy");
    }
    this.policy = Object.freeze({ ...policy });
  }

  private async consent(id: string, scope: ResearchScope, annotate = false): Promise<ResearchConsent> {
    const value = await this.ports.consent(id);
    const now = this.now();
    if (!this.policy.enabled || !value || value.id !== id || !matches(value.scope, scope) || !value.retainClips ||
      (annotate && !value.annotateClips) || value.withdrawnAt !== null ||
      !Number.isFinite(value.validFrom) || !Number.isFinite(value.validUntil) ||
      value.validFrom > now || value.validUntil <= now) deny();
    return structuredClone(value);
  }

  private async source(clip: Pick<ResearchClip, "scope" | "consentId" | "observationId" | "capturedAt" | "endedAt">, annotate = false): Promise<ResearchConsent> {
    const observation = await this.ports.observation(clip.observationId);
    // Resolve the grant last, so a slow source lookup cannot bypass withdrawal.
    const consent = await this.consent(clip.consentId, clip.scope, annotate);
    if (!observation || !matches(observation.scope, clip.scope) || !Number.isFinite(clip.capturedAt) ||
      !Number.isFinite(observation.startedAt) || !Number.isFinite(observation.endedAt) || observation.endedAt <= observation.startedAt ||
      !Number.isFinite(clip.endedAt) || clip.capturedAt < Math.max(consent.validFrom, observation.startedAt) ||
      clip.endedAt > Math.min(consent.validUntil, observation.endedAt, this.now()) || clip.endedAt <= clip.capturedAt) deny();
    return consent;
  }

  private available(clip: ResearchClip | null, id: string, scope: ResearchScope): ResearchClip {
    if (!clip || clip.id !== id || !matches(clip.scope, scope) || clip.status !== "available" ||
      !Number.isFinite(clip.createdAt) || !Number.isFinite(clip.expiresAt) || clip.createdAt > this.now() ||
      clip.expiresAt <= this.now() || clip.expiresAt > clip.createdAt + this.policy.retentionMs ||
      clip.durationMs !== clip.endedAt - clip.capturedAt || clip.durationMs <= 0 || clip.durationMs > this.policy.maxDurationMs) deny();
    return structuredClone(clip);
  }

  async retain(input: { actorId: string; scope: ResearchScope; consentId: string; observationId: string;
    durationMs: number; capturedAt: number; endedAt: number; mimeType: string; bytes: Uint8Array }): Promise<ResearchClip> {
    input = { ...input, scope: { ...input.scope } };
    const id = globalThis.crypto.randomUUID();
    let pending: ResearchClip | null = null;
    try {
      if (!this.policy.enabled || !(await this.ports.authorize(input.actorId, input.scope, "retain")) ||
        !input.observationId || !Number.isFinite(input.durationMs) || input.durationMs <= 0 ||
        input.durationMs > this.policy.maxDurationMs || input.bytes.byteLength === 0 ||
        input.bytes.byteLength > this.policy.maxBytes || !["video/webm", "video/mp4", "audio/wav"].includes(input.mimeType)) deny();
      const consent = await this.source(input);
      if (input.durationMs !== input.endedAt - input.capturedAt) deny();
      const createdAt = this.now();
      const clip: ResearchClip = { id, scope: { ...input.scope }, consentId: consent.id,
        observationId: input.observationId, createdAt, expiresAt: Math.min(createdAt + this.policy.retentionMs, consent.validUntil),
        durationMs: input.durationMs, capturedAt: input.capturedAt, endedAt: input.endedAt, mimeType: input.mimeType, status: "pending-upload" };
      // Durable cleanup intent precedes any possible physical upload. The provider
      // must honor cancellation and object expiry, including incomplete uploads.
      await this.ports.metadata.put(structuredClone(clip)); pending = clip;
      const remaining = Math.min(clip.expiresAt, createdAt + 30_000) - this.now();
      if (remaining <= 0) deny();
      const uploadSignal = AbortSignal.timeout(Math.ceil(remaining));
      await this.ports.blobs.put(id, input.bytes, { expiresAt: clip.expiresAt, signal: uploadSignal });
      uploadSignal.throwIfAborted();
      const current = await this.ports.metadata.get(id);
      if (!current || current.id !== id || current.status !== "pending-upload" || !matches(current.scope, input.scope)) deny();
      if (!(await this.ports.authorize(input.actorId, input.scope, "retain"))) deny();
      await this.source(input);
      if (clip.expiresAt <= this.now()) deny();
      const available: ResearchClip = { ...clip, status: "available" };
      await this.ports.metadata.put(structuredClone(available));
      await this.ports.audit({ action: "retained", actorId: input.actorId, clipId: id, scope: input.scope, at: this.now() });
      if (!(await this.ports.authorize(input.actorId, input.scope, "retain"))) deny();
      await this.source(input);
      if (clip.expiresAt <= this.now()) deny();
      return available;
    } catch (error) {
      if (pending) {
        // If a store is unavailable the original pending intent survives for the
        // retention worker. Never erase cleanup intent before physical deletion.
        try { await this.ports.metadata.put({ ...pending, status: "deletion-pending" }); } catch { /* pending lease remains */ }
        await this.ports.blobs.delete(id);
        await this.ports.metadata.put({ ...pending, status: "deleted" });
        await this.ports.audit({ action: "deleted", actorId: input.actorId, clipId: id, scope: input.scope, at: this.now() });
      }
      throw error;
    } finally { input.bytes.fill(0); }
  }

  /** Media is available only within the callback; do not copy it into logs or analytics. */
  async withClip<T>(input: { actorId: string; scope: ResearchScope; clipId: string }, consume: (bytes: Uint8Array, clip: ResearchClip) => Promise<T>): Promise<T> {
    input = { ...input, scope: { ...input.scope } };
    const clip = this.available(await this.ports.metadata.get(input.clipId), input.clipId, input.scope);
    if (!(await this.ports.authorize(input.actorId, input.scope, "annotate"))) deny();
    await this.source(clip, true);
    const bytes = await this.ports.blobs.get(clip.id);
    try {
      await this.ports.audit({ action: "read", actorId: input.actorId, clipId: clip.id, scope: input.scope, at: this.now() });
      // Audit and blob retrieval may be slow. Recheck the exact still-live clip,
      // membership and consent immediately before releasing temporary bytes.
      const current = this.available(await this.ports.metadata.get(input.clipId), input.clipId, input.scope);
      if ((["consentId", "observationId", "createdAt", "expiresAt", "durationMs", "capturedAt", "endedAt", "mimeType"] as const).some(key => current[key] !== clip[key]) ||
        !(await this.ports.authorize(input.actorId, input.scope, "annotate"))) deny();
      await this.source(clip, true);
      if (clip.expiresAt <= this.now()) deny();
      return await consume(bytes, clip);
    } finally { bytes.fill(0); }
  }

  /** Physical deletion precedes tombstoning, so a failed provider deletion can be retried. */
  async purgeUnavailable(actorId: string, scope: ResearchScope): Promise<number> {
    scope = { ...scope };
    if (!(await this.ports.authorize(actorId, scope, "delete"))) deny();
    let deleted = 0;
    for (const clip of await this.ports.metadata.list(scope)) {
      if (!matches(clip.scope, scope) || clip.status === "deleted") continue;
      if (clip.status === "pending-upload" && clip.createdAt + 30_000 > this.now() && clip.expiresAt > this.now()) continue;
      const consent = await this.ports.consent(clip.consentId);
      if (clip.status === "available" && this.policy.enabled && Number.isFinite(clip.expiresAt) && clip.expiresAt > this.now() && consent && consent.id === clip.consentId && matches(consent.scope, scope) &&
        consent.retainClips && consent.withdrawnAt === null && Number.isFinite(consent.validFrom) && consent.validFrom <= Math.min(this.now(), clip.capturedAt) &&
        Number.isFinite(consent.validUntil) && consent.validUntil > this.now() && consent.validUntil >= clip.endedAt) continue;
      await this.ports.metadata.put({ ...clip, status: "deletion-pending" });
      await this.ports.blobs.delete(clip.id);
      await this.ports.metadata.put({ ...clip, status: "deleted" });
      await this.ports.audit({ action: "deleted", actorId, clipId: clip.id, scope, at: this.now() });
      deleted++;
    }
    return deleted;
  }
}
