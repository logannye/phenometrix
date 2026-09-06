import { it, expect } from "vitest";
import { randomBytes } from "node:crypto";
import { createEncounterClient } from "../../../packages/encounter-client/src/index.js";
import { buildTreatmentResponseEvidence } from "../../../packages/evidence-core/src/treatment-response.js";
import { createAuthenticator, signDevelopmentSession } from "./auth.js";
import { createEncounterHttpServer } from "./http.js";
import { EncounterService } from "./service.js";
import { MemoryRepository } from "./repository.memory.js";
import { runOneJob } from "./worker.js";
import { seedSyntheticEpisode, syntheticFixture, SYNTHETIC_PRINCIPAL, SYNTHETIC_SCOPE, SYNTHETIC_EPISODE_ID } from "./synthetic.js";

it("delivers a normal host capture through authenticated ingestion, analysis and chart evidence", async () => {
  const now = new Date().toISOString();
  const service = new EncounterService(new MemoryRepository(), { mode: "synthetic" });
  await seedSyntheticEpisode(service, now);
  const secret = randomBytes(32), issuer = "test-host", audience = "encounter-service";
  const token = await signDevelopmentSession(SYNTHETIC_PRINCIPAL, secret, issuer, audience);
  const server = createEncounterHttpServer({ service, allowedOrigins: [], authenticate: createAuthenticator({ mode: "synthetic", issuer, audience, developmentSecret: secret }) });
  await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
  try {
    const address = server.address();
    if (!address || typeof address === "string") throw new Error("missing-address");
    const client = createEncounterClient({ baseUrl: `http://127.0.0.1:${address.port}`, episodeId: SYNTHETIC_EPISODE_ID, scope: SYNTHETIC_SCOPE, accessToken: async () => token });
    expect((await client.loadEvidence()).run).toBeNull();
    const context = await client.captureContext("synthetic-encounter-4");
    expect(context.binding.scope).toEqual(SYNTHETIC_SCOPE);
    const fixture = await syntheticFixture(now);
    const observation = { ...fixture.sessions[3]!.observation, observationId: "host-capture", revisionId: "host-capture-v1" };
    const first = await client.ingestObservation(observation);
    expect(await client.ingestObservation(observation)).toEqual({ ...first, duplicate: true });
    expect(await runOneJob(service.repository)).toBe("published");
    const loaded = await client.loadEvidence();
    expect(loaded.run?.rows).toHaveLength(4);
    const evidence = await buildTreatmentResponseEvidence({ run: loaded.run!, expectedScope: SYNTHETIC_SCOPE, reviews: loaded.reviews });
    expect(evidence.status).toBe("generated");
    expect(evidence.encounterCount).toBe(4);
    expect(evidence.run.fittedModels).toBe(false);
    // Withdrawal hides generated evidence before the recomputation worker runs.
    const state = await service.repository.getState(SYNTHETIC_SCOPE.tenantId, SYNTHETIC_EPISODE_ID);
    const consentRecord = state!.records.find(record => record.kind === "consent")!;
    await service.write(SYNTHETIC_PRINCIPAL, SYNTHETIC_EPISODE_ID, "consents", {
      data: { ...fixture.consent, withdrawnAt: new Date().toISOString() }, supersedesRecordId: consentRecord.id
    }, "withdraw-host-consent");
    await expect(client.loadEvidence()).rejects.toThrow("encounter-request-403");
    await expect(client.captureContext("synthetic-encounter-4")).rejects.toThrow("encounter-request-403");
  } finally {
    await new Promise<void>(resolve => server.close(() => resolve()));
    secret.fill(0);
  }
});
