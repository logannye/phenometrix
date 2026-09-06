import { describe, it, expect, vi } from "vitest";
import { ResearchClipVault, type ResearchClip, type ResearchClipPorts, type ResearchConsent } from "./index.js";

function fixture(enabled = true) {
  const scope = { tenantId: "synthetic-tenant", studyId: "synthetic-study", participantId: "synthetic-subject" };
  let now = 100;
  const consent: ResearchConsent = { id: "consent", scope, validFrom: 0, validUntil: 1000, withdrawnAt: null, retainClips: true, annotateClips: true };
  const blobs = new Map<string, Uint8Array>();
  const metadata = new Map<string, ResearchClip>();
  const audit: string[] = [];
  const ports: ResearchClipPorts = {
    blobs: { put: async (key, data) => { blobs.set(key, data.slice()); }, get: async key => blobs.get(key)!.slice(), delete: async key => { blobs.delete(key); } },
    metadata: { put: async clip => { metadata.set(clip.id, clip); }, get: async id => metadata.get(id) ?? null, list: async () => [...metadata.values()] },
    consent: async () => consent, observation: async () => ({ scope, startedAt: 0, endedAt: 100 }),
    authorize: async actor => actor === "researcher", audit: async event => { audit.push(event.action); }
  };
  const vault = new ResearchClipVault(ports, { enabled, maxBytes: 100, maxDurationMs: 1000, retentionMs: 200 }, () => now);
  const retain = () => vault.retain({ scope, actorId: "researcher", consentId: "consent", observationId: "synthetic-observation", capturedAt: 10, endedAt: 90, durationMs: 80, mimeType: "video/webm", bytes: new Uint8Array([1, 2, 3]) });
  return { scope, consent, blobs, metadata, audit, vault, ports, retain, setTime: (time: number) => { now = time; } };
}

describe("separate research-media governance", () => {
  it("is disabled unless an explicit policy enables storage", async () => {
    const f = fixture(false); await expect(f.retain()).rejects.toThrow(); expect(f.blobs.size).toBe(0);
  });
  it("checks research consent separately and zeroes bytes even on denial", async () => {
    const f = fixture(); f.consent.retainClips = false;
    const bytes = new Uint8Array([7]);
    await expect(f.vault.retain({ scope: f.scope, actorId: "researcher", consentId: "consent", observationId: "obs", capturedAt: 10, endedAt: 90, durationMs: 80, mimeType: "video/webm", bytes })).rejects.toThrow();
    expect([...bytes]).toEqual([0]); expect(f.blobs.size).toBe(0);
  });
  it("separates tenant scope and expires access without exposing bytes", async () => {
    const f = fixture(); const clip = await f.retain();
    await expect(f.vault.withClip({ scope: { ...f.scope, tenantId: "other" }, actorId: "researcher", clipId: clip.id }, async () => 1)).rejects.toThrow();
    f.setTime(301);
    await expect(f.vault.withClip({ scope: f.scope, actorId: "researcher", clipId: clip.id }, async () => 1)).rejects.toThrow();
    expect(await f.vault.purgeUnavailable("researcher", f.scope)).toBe(1);
    expect(f.blobs.size).toBe(0); expect(f.metadata.get(clip.id)?.status).toBe("deleted");
  });
  it("revocation blocks future reads and removes physical media", async () => {
    const f = fixture(); const clip = await f.retain(); f.consent.withdrawnAt = 101;
    await expect(f.vault.withClip({ scope: f.scope, actorId: "researcher", clipId: clip.id }, async () => 1)).rejects.toThrow();
    expect(await f.vault.purgeUnavailable("researcher", f.scope)).toBe(1);
  });
  it("audits access and clears annotation buffers even when consumers throw", async () => {
    const f = fixture(); const clip = await f.retain(); let buffer: Uint8Array | undefined;
    await expect(f.vault.withClip({ scope: f.scope, actorId: "researcher", clipId: clip.id }, async bytes => { buffer = bytes; throw new Error("annotation failed"); })).rejects.toThrow();
    expect([...buffer!]).toEqual([0, 0, 0]); expect(f.audit).toEqual(["retained", "read"]);
  });
  it("compensates when consent changes during an upload", async () => {
    const f = fixture(); const put = f.ports.blobs.put;
    f.ports.blobs.put = async (key, bytes, options) => { await put(key, bytes, options); f.consent.withdrawnAt = 101; };
    await expect(f.retain()).rejects.toThrow(); expect(f.blobs.size).toBe(0);
  });
  it("requires source ownership and consent covering capture, not only upload", async () => {
    const f = fixture(); f.consent.validFrom = 50;
    await expect(f.retain()).rejects.toThrow(); expect(f.blobs.size).toBe(0);
    f.consent.validFrom = 0; f.ports.observation = async () => ({ scope: { ...f.scope, participantId: "other" }, startedAt: 0, endedAt: 100 });
    await expect(f.retain()).rejects.toThrow(); expect(f.blobs.size).toBe(0);
  });
  it("retains cleanup intent when a compensating physical deletion fails", async () => {
    const f = fixture(); const put = f.ports.blobs.put, remove = f.ports.blobs.delete;
    f.ports.blobs.put = async (key, bytes, options) => { await put(key, bytes, options); f.consent.withdrawnAt = 100; };
    f.ports.blobs.delete = async () => { throw new Error("store unavailable"); };
    await expect(f.retain()).rejects.toThrow();
    expect([...f.metadata.values()][0].status).toBe("deletion-pending");
    f.ports.blobs.delete = remove;
    expect(await f.vault.purgeUnavailable("researcher", f.scope)).toBe(1); expect(f.blobs.size).toBe(0);
  });
  it("rejects substituted consent IDs and non-finite source times", async () => {
    const f = fixture(); f.consent.id = "another-consent";
    await expect(f.retain()).rejects.toThrow("access-unavailable");
    f.consent.id = "consent"; f.ports.observation = async () => ({ scope: f.scope, startedAt: NaN, endedAt: NaN });
    await expect(f.retain()).rejects.toThrow("access-unavailable"); expect(f.blobs.size).toBe(0);
  });
  it("does not release bytes after expiry, withdrawal or role loss during auditing", async () => {
    for (const change of ["expiry", "withdrawal", "membership"] as const) {
      const f = fixture(), clip = await f.retain(), consume = vi.fn(async () => 1);
      let bytes: Uint8Array | undefined;
      const get = f.ports.blobs.get;
      f.ports.blobs.get = async key => { bytes = await get(key); return bytes; };
      f.ports.audit = async () => {
        if (change === "expiry") f.setTime(301);
        if (change === "withdrawal") f.consent.withdrawnAt = 100;
        if (change === "membership") f.ports.authorize = async () => false;
      };
      await expect(f.vault.withClip({ scope: f.scope, actorId: "researcher", clipId: clip.id }, consume)).rejects.toThrow("access-unavailable");
      expect(consume).not.toHaveBeenCalled(); expect([...bytes!]).toEqual([0, 0, 0]);
    }
  });
  it("rejects substituted clip metadata and cleans media when its exact grant is unavailable", async () => {
    const f = fixture(), clip = await f.retain();
    f.ports.metadata.get = async () => ({ ...clip, id: "different-clip" });
    await expect(f.vault.withClip({ scope: f.scope, actorId: "researcher", clipId: clip.id }, async () => 1)).rejects.toThrow("access-unavailable");
    f.consent.id = "another-consent";
    expect(await f.vault.purgeUnavailable("researcher", f.scope)).toBe(1); expect(f.blobs.size).toBe(0);
  });
  it("expires pending uploads even before the upload lease ends and compensates late completion", async () => {
    const f = fixture(), put = f.ports.blobs.put;
    f.ports.blobs.put = async (key, bytes, options) => {
      f.setTime(301);
      expect(await f.vault.purgeUnavailable("researcher", f.scope)).toBe(1);
      await put(key, bytes, options);
    };
    await expect(f.retain()).rejects.toThrow("access-unavailable");
    expect(f.blobs.size).toBe(0); expect([...f.metadata.values()][0].status).toBe("deleted");
  });
  it("compensates if retention expires while the retained event is being audited", async () => {
    const f = fixture();
    f.ports.audit = async event => { if (event.action === "retained") f.setTime(301); };
    await expect(f.retain()).rejects.toThrow("access-unavailable");
    expect(f.blobs.size).toBe(0); expect([...f.metadata.values()][0].status).toBe("deleted");
  });
});
