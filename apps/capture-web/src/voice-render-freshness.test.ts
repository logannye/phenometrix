import { describe, expect, it } from "vitest";
import { assessVoiceRenderLease, snapshotVoiceRenderLease, VoiceRenderLeaseIssuer,
  VOICE_RENDER_FRESHNESS_VERSION, type VoiceRenderLease } from "./voice-render-freshness.js";
const identity = { captureEpoch: 7, clockMappingId: 3, mainTimeOriginMs: 100_000 };
const lease: VoiceRenderLease = { schemaVersion: VOICE_RENDER_FRESHNESS_VERSION, leaseId: 1,
  ...identity, issuedAtMainMs: 1000, expiresAtMainMs: 1100 };
const ack = (value: VoiceRenderLease, disposition: "accepted" | "retired" = "accepted") => ({
  schemaVersion: VOICE_RENDER_FRESHNESS_VERSION, type: "render-freshness-lease-ack", leaseId: value.leaseId,
  captureEpoch: value.captureEpoch, clockMappingId: value.clockMappingId, disposition });

describe("same-main-clock render freshness lease contract", () => {
  it("accepts exact expiry and withholds missing, expired, future, and malformed evidence", () => {
    expect(assessVoiceRenderLease(lease, 1100, identity)).toEqual({ status: "current", issuedToNowMs: 100 });
    for (const [value, now, reason] of [[null, 1001, "missing-lease"], [lease, 1100.001, "expired-lease"],
      [lease, 999, "future-lease"], [{ ...lease, expiresAtMainMs: 1200 }, 1001, "malformed-lease"]] as const) {
      expect(assessVoiceRenderLease(value, now, identity)).toEqual({ status: "withheld", reason });
    }
  });
  it("requires the original epoch, mapping and performance origin", () => {
    expect(assessVoiceRenderLease(lease, 1050, { ...identity, captureEpoch: 8 })).toMatchObject({ reason: "retired-epoch" });
    expect(assessVoiceRenderLease(lease, 1050, { ...identity, clockMappingId: 4 })).toMatchObject({ reason: "retired-clock-mapping" });
    expect(assessVoiceRenderLease(lease, 1050, { ...identity, mainTimeOriginMs: 0 })).toMatchObject({ reason: "clock-origin-mismatch" });
  });
  it("keeps the oldest lease across a partial block and both output-frame endpoints", () => {
    let active = lease;
    const blockLease = snapshotVoiceRenderLease(active);
    active = { ...lease, leaseId: 2, issuedAtMainMs: 1090, expiresAtMainMs: 1190 };
    // A new lease midway through this block cannot rescue either earlier endpoint.
    expect(assessVoiceRenderLease(blockLease, 1101, identity)).toMatchObject({ reason: "expired-lease" });
    expect(assessVoiceRenderLease(blockLease, 1111, identity)).toMatchObject({ reason: "expired-lease" });
    expect(assessVoiceRenderLease(snapshotVoiceRenderLease(active), 1111, identity)).toMatchObject({ status: "current" });
    expect(snapshotVoiceRenderLease(null)).toBeNull();
  });
  it("retains one credit through timeout and reset until the matching retired ACK", () => {
    let now = 1000;
    const issuer = new VoiceRenderLeaseIssuer({ now: () => now, timeOrigin: identity.mainTimeOriginMs });
    const first = issuer.issue(7, 3)!.lease;
    now = 2000;
    for (let i = 0; i < 100; i += 1) expect(issuer.issue(8, 4)).toBeNull();
    issuer.invalidate();
    expect(issuer.issue(8, 4)).toBeNull();
    expect(issuer.acknowledge({ ...ack(first), leaseId: 99 })).toBe(false);
    expect(issuer.acknowledge(ack(first, "retired"))).toBe(true);
    const second = issuer.issue(8, 4)!.lease;
    expect(issuer.acknowledge(ack(first))).toBe(false);
    expect(issuer.issue(8, 4)).toBeNull();
    expect(issuer.statistics).toMatchObject({ issuedCount: 2, expiredPendingCount: 1, maximumPendingRequests: 1 });
    issuer.stop();
    expect(issuer.acknowledge(ack(second))).toBe(false);
    expect(issuer.issue(8, 4)).toBeNull();
  });
  it("verifies exact issuance, bounds its ledger, and cannot qualify an evicted or forged lease", () => {
    let now = 1000;
    const issuer = new VoiceRenderLeaseIssuer({ now: () => now, timeOrigin: identity.mainTimeOriginMs });
    const first = issuer.issue(7, 3)!.lease;
    expect(issuer.certify(first, 1050, identity)).toMatchObject({ status: "certified", renderAgeUpperBoundMs: 50,
      timestampPrecisionMs: null, scope: "frame-last-required-rendered-input" });
    expect(issuer.certify({ ...first, issuedAtMainMs: 1001, expiresAtMainMs: 1101 }, 1050, identity)).toMatchObject({ reason: "unrecognized-lease" });
    issuer.acknowledge(ack(first));
    for (let i = 0; i < 9; i += 1) {
      now += 1;
      const next = issuer.issue(7, 3)!;
      issuer.acknowledge(ack(next.lease));
    }
    expect(issuer.statistics.maximumLedgerEntries).toBe(8);
    expect(issuer.certify(first, 1050, identity)).toMatchObject({ reason: "unrecognized-lease" });
  });
});
