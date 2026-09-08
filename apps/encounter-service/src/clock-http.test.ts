import { afterEach, describe, expect, it } from "vitest";
import { randomBytes } from "node:crypto";
import type { Server } from "node:http";
import { EncounterClockSampleV1Schema } from "@phenometrix/contracts";
import { createAuthenticator, signDevelopmentSession, type Authenticate } from "./auth.js";
import { createEncounterHttpServer } from "./http.js";
import { createClockSourceProvider, type EncounterClockSourceProvider } from "./clock-source.js";
import { EncounterService } from "./service.js";
import { MemoryRepository } from "./repository.memory.js";
import { seedSyntheticEpisode, SYNTHETIC_EPISODE_ID, SYNTHETIC_PRINCIPAL, SYNTHETIC_SCOPE } from "./synthetic.js";

const servers: Server[] = [];
afterEach(async () => { await Promise.all(servers.splice(0).map(server => new Promise<void>(resolve => server.close(() => resolve())))); });
async function start(options: { provider?: EncounterClockSourceProvider; sourceTimeoutMs?: number; authenticate?: Authenticate } = {}) {
  const service = new EncounterService(new MemoryRepository(), { mode: "synthetic" });
  await seedSyntheticEpisode(service);
  const secret = randomBytes(32), issuer = "synthetic-clock", audience = "service";
  let provider = options.provider;
  const server = createEncounterHttpServer({ service, allowedOrigins: [],
    authenticate: options.authenticate ?? createAuthenticator({ mode: "synthetic", issuer, audience, developmentSecret: secret }),
    clockSource: async input => provider ? provider(input) : null, clockSourceTimeoutMs: options.sourceTimeoutMs ?? 250 });
  servers.push(server); await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
  const address = server.address(); if (!address || typeof address === "string") throw new Error("missing port");
  const url = `http://127.0.0.1:${address.port}/v1/episodes/${SYNTHETIC_EPISODE_ID}/capture-context/synthetic-encounter-1/clock`;
  const token = await signDevelopmentSession(SYNTHETIC_PRINCIPAL, secret, issuer, audience);
  const headers = { Authorization: `Bearer ${token}` };
  return { service, url, headers, setProvider: (value: EncounterClockSourceProvider) => { provider = value; } };
}

describe("authenticated encounter-clock probe", () => {
  it("returns server-boundary timestamps with explicit synthetic quality and no caching", async () => {
    const f = await start({ provider: createClockSourceProvider({ mode: "synthetic" }) });
    const before = Date.now(), response = await fetch(f.url, { headers: f.headers }), after = Date.now();
    expect(response.status).toBe(200); expect(response.headers.get("cache-control")).toBe("no-store");
    const sample = EncounterClockSampleV1Schema.parse(await response.json());
    expect(sample.scope).toEqual(SYNTHETIC_SCOPE); expect(sample.episodeId).toBe(SYNTHETIC_EPISODE_ID); expect(sample.encounterId).toBe("synthetic-encounter-1");
    expect(sample.serverReceivedAtMs).toBeGreaterThanOrEqual(before); expect(sample.serverSentAtMs).toBeLessThanOrEqual(after);
    expect(sample.source?.kind).toBe("synthetic"); expect(sample.source!.validUntilMs - sample.serverSentAtMs).toBeLessThanOrEqual(60_000);
    expect((await fetch(f.url)).status).toBe(401);
    expect((await fetch(f.url.replace("synthetic-encounter-1", "unbound-encounter"), { headers: f.headers })).status).toBe(403);
  });
  it("returns no accuracy claim for absent, expired, overlong or stalled sources", async () => {
    const f = await start({ sourceTimeoutMs: 10 });
    for (const provider of [undefined,
      async () => ({ sourceId: "expired", kind: "monitored-utc" as const, maximumUtcErrorMs: 1, validUntilMs: Date.now() - 1 }),
      async () => ({ sourceId: "overlong", kind: "monitored-utc" as const, maximumUtcErrorMs: 1, validUntilMs: Date.now() + 120_000 }),
      () => new Promise<null>(() => {})]) {
      if (provider) f.setProvider(provider);
      const response = await fetch(f.url, { headers: f.headers });
      expect(response.status).toBe(200); expect((await response.json()).source).toBeNull();
    }
  });
  it("rechecks capture consent after the provider await", async () => {
    const f = await start();
    f.setProvider(async () => {
      const state = await f.service.repository.getState(SYNTHETIC_SCOPE.tenantId, SYNTHETIC_EPISODE_ID);
      const record = state!.records.find(item => item.kind === "consent")!;
      await f.service.write(SYNTHETIC_PRINCIPAL, SYNTHETIC_EPISODE_ID, "consents", { data: { ...record.payload, withdrawnAt: new Date().toISOString() }, supersedesRecordId: record.id }, "clock-withdraw");
      return { sourceId: "synthetic", kind: "synthetic", maximumUtcErrorMs: 0, validUntilMs: Date.now() + 1000 };
    });
    expect((await fetch(f.url, { headers: f.headers })).status).toBe(403);
  });
  it("revalidates authentication after source resolution and denies revoked capture permissions", async () => {
    let calls = 0;
    const f = await start({ authenticate: async () => ({ ...SYNTHETIC_PRINCIPAL, permissions: ++calls > 1 ? ["evidence:read"] : SYNTHETIC_PRINCIPAL.permissions }),
      provider: createClockSourceProvider({ mode: "synthetic" }) });
    expect((await fetch(f.url, { headers: f.headers })).status).toBe(403); expect(calls).toBe(2);
  });
});
