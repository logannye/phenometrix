/** Software render freshness; this is not a physical acquisition timestamp. */
export const VOICE_RENDER_FRESHNESS_VERSION = "phenometric.voice-render-freshness.v1" as const;
export const VOICE_RENDER_LEASE_RENEWAL_MS = 25;
export const VOICE_RENDER_LEASE_MAX_AGE_MS = 100;
export const VOICE_RENDER_LEASE_LEDGER_CAPACITY = 8;
export type VoiceRenderFreshnessPolicy = "required-render-lease-v1" | "unqualified-legacy-fixture";
export interface VoiceRenderLeaseIdentity {
  captureEpoch: number;
  clockMappingId: number;
  mainTimeOriginMs: number;
}
export interface VoiceRenderLease extends VoiceRenderLeaseIdentity {
  schemaVersion: typeof VOICE_RENDER_FRESHNESS_VERSION;
  leaseId: number;
  issuedAtMainMs: number;
  expiresAtMainMs: number;
}
export interface VoiceRenderLeaseRequest {
  type: "render-freshness-lease";
  lease: VoiceRenderLease;
}
export interface VoiceRenderLeaseAck {
  schemaVersion: typeof VOICE_RENDER_FRESHNESS_VERSION;
  type: "render-freshness-lease-ack";
  leaseId: number;
  captureEpoch: number;
  clockMappingId: number;
  disposition: "accepted" | "retired";
}
export type VoiceRenderLeaseFailure = "missing-lease" | "malformed-lease" | "retired-epoch" |
  "retired-clock-mapping" | "clock-origin-mismatch" | "future-lease" | "expired-lease" | "unrecognized-lease";
export type VoiceRenderLeaseAssessment = { status: "current"; issuedToNowMs: number }
  | { status: "withheld"; reason: VoiceRenderLeaseFailure };
export interface VoiceRenderFreshnessCertificate {
  schemaVersion: typeof VOICE_RENDER_FRESHNESS_VERSION;
  status: "certified";
  basis: "main-lease-before-block-first-input";
  scope: "frame-last-required-rendered-input";
  leaseId: number;
  issuedAtMainMs: number;
  resultReceivedAtMainMs: number;
  maximumAgeMs: 100;
  renderAgeUpperBoundMs: number;
  timestampPrecisionMs: null;
}
export type VoiceRenderFreshnessResult = VoiceRenderFreshnessCertificate |
  { schemaVersion: typeof VOICE_RENDER_FRESHNESS_VERSION; status: "withheld"; reason: VoiceRenderLeaseFailure };

export function validVoiceRenderLease(value: unknown): value is VoiceRenderLease {
  if (!value || typeof value !== "object") return false;
  const lease = value as VoiceRenderLease;
  return lease.schemaVersion === VOICE_RENDER_FRESHNESS_VERSION &&
    Number.isSafeInteger(lease.leaseId) && lease.leaseId > 0 &&
    Number.isSafeInteger(lease.captureEpoch) && lease.captureEpoch >= 0 &&
    Number.isSafeInteger(lease.clockMappingId) && lease.clockMappingId >= 0 &&
    Number.isFinite(lease.mainTimeOriginMs) &&
    Number.isFinite(lease.issuedAtMainMs) && lease.issuedAtMainMs >= 0 &&
    Number.isFinite(lease.expiresAtMainMs) &&
    lease.expiresAtMainMs === lease.issuedAtMainMs + VOICE_RENDER_LEASE_MAX_AGE_MS;
}

/** This check uses no audio clock, fixed projection, result latency, or fitted rate. */
export function assessVoiceRenderLease(lease: unknown, nowInMainMs: number,
  identity: VoiceRenderLeaseIdentity): VoiceRenderLeaseAssessment {
  const fail = (reason: VoiceRenderLeaseFailure): VoiceRenderLeaseAssessment => ({ status: "withheld", reason });
  if (lease === null || lease === undefined) return fail("missing-lease");
  if (!validVoiceRenderLease(lease)) return fail("malformed-lease");
  if (lease.captureEpoch !== identity.captureEpoch) return fail("retired-epoch");
  if (lease.clockMappingId !== identity.clockMappingId) return fail("retired-clock-mapping");
  if (lease.mainTimeOriginMs !== identity.mainTimeOriginMs) return fail("clock-origin-mismatch");
  if (!Number.isFinite(nowInMainMs) || nowInMainMs < lease.issuedAtMainMs) return fail("future-lease");
  if (nowInMainMs > lease.expiresAtMainMs) return fail("expired-lease");
  return { status: "current", issuedToNowMs: nowInMainMs - lease.issuedAtMainMs };
}

/** Snapshot once at PCM block start; later renewal cannot qualify earlier support. */
export function snapshotVoiceRenderLease(lease: VoiceRenderLease | null): VoiceRenderLease | null {
  return lease === null ? null : { ...lease };
}

/** Main-realm issuer and verifier. One outstanding control request, at most eight identities. */
export class VoiceRenderLeaseIssuer {
  private nextId = 0;
  private pending: { lease: VoiceRenderLease; timeoutReported: boolean } | null = null;
  private readonly issued = new Map<number, VoiceRenderLease>();
  private stopped = false;
  readonly statistics = { issuedCount: 0, acknowledgedCount: 0, retiredAckCount: 0,
    expiredPendingCount: 0, maximumPendingRequests: 0, maximumLedgerEntries: 0 };
  constructor(private readonly clock: Pick<Performance, "now" | "timeOrigin">) {}
  issue(captureEpoch: number, clockMappingId: number): VoiceRenderLeaseRequest | null {
    if (this.stopped) return null;
    const now = this.clock.now();
    if (this.pending) {
      if (now > this.pending.lease.expiresAtMainMs && !this.pending.timeoutReported) {
        this.pending.timeoutReported = true;
        this.statistics.expiredPendingCount += 1;
      }
      return null; // Expiry cannot mint another transport credit.
    }
    const lease: VoiceRenderLease = { schemaVersion: VOICE_RENDER_FRESHNESS_VERSION,
      leaseId: ++this.nextId, captureEpoch, clockMappingId, mainTimeOriginMs: this.clock.timeOrigin,
      issuedAtMainMs: now, expiresAtMainMs: now + VOICE_RENDER_LEASE_MAX_AGE_MS };
    if (!validVoiceRenderLease(lease)) throw new Error("invalid-render-lease-issuance");
    this.pending = { lease, timeoutReported: false };
    this.issued.set(lease.leaseId, lease);
    while (this.issued.size > VOICE_RENDER_LEASE_LEDGER_CAPACITY) this.issued.delete(this.issued.keys().next().value!);
    this.statistics.issuedCount += 1;
    this.statistics.maximumPendingRequests = 1;
    this.statistics.maximumLedgerEntries = Math.max(this.statistics.maximumLedgerEntries, this.issued.size);
    return { type: "render-freshness-lease", lease: { ...lease } };
  }
  acknowledge(value: unknown): boolean {
    const ack = value as Partial<VoiceRenderLeaseAck> | null;
    const pending = this.pending?.lease;
    if (!ack || !pending || ack.schemaVersion !== VOICE_RENDER_FRESHNESS_VERSION ||
        ack.type !== "render-freshness-lease-ack" || ack.leaseId !== pending.leaseId ||
        ack.captureEpoch !== pending.captureEpoch || ack.clockMappingId !== pending.clockMappingId ||
        (ack.disposition !== "accepted" && ack.disposition !== "retired")) return false;
    this.pending = null;
    this.statistics.acknowledgedCount += 1;
    this.statistics.retiredAckCount += Number(ack.disposition === "retired");
    return true;
  }
  /** Reset revokes verification but retains the outstanding request until its retired ACK. */
  invalidate(): void { this.issued.clear(); }
  certify(lease: unknown, arrivalInMainMs: number, identity: VoiceRenderLeaseIdentity): VoiceRenderFreshnessResult {
    const assessment = assessVoiceRenderLease(lease, arrivalInMainMs, identity);
    if (assessment.status === "withheld") return { schemaVersion: VOICE_RENDER_FRESHNESS_VERSION, ...assessment };
    const value = lease as VoiceRenderLease;
    const original = this.issued.get(value.leaseId);
    if (this.stopped || !original || original.captureEpoch !== value.captureEpoch ||
        original.clockMappingId !== value.clockMappingId || original.mainTimeOriginMs !== value.mainTimeOriginMs ||
        original.issuedAtMainMs !== value.issuedAtMainMs || original.expiresAtMainMs !== value.expiresAtMainMs) {
      return { schemaVersion: VOICE_RENDER_FRESHNESS_VERSION, status: "withheld", reason: "unrecognized-lease" };
    }
    return { schemaVersion: VOICE_RENDER_FRESHNESS_VERSION, status: "certified",
      basis: "main-lease-before-block-first-input", scope: "frame-last-required-rendered-input",
      leaseId: value.leaseId, issuedAtMainMs: value.issuedAtMainMs, resultReceivedAtMainMs: arrivalInMainMs,
      maximumAgeMs: 100, renderAgeUpperBoundMs: assessment.issuedToNowMs, timestampPrecisionMs: null };
  }
  stop(): void { this.stopped = true; this.pending = null; this.issued.clear(); }
}
