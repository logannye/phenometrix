import { randomBytes } from "node:crypto";
import { createAuthenticator, signDevelopmentSession, type Principal } from "../../encounter-service/src/auth.js";
import { createEncounterHttpServer } from "../../encounter-service/src/http.js";
import { EncounterService } from "../../encounter-service/src/service.js";
import { MemoryRepository } from "../../encounter-service/src/repository.memory.js";
import { runOneJob } from "../../encounter-service/src/worker.js";
import { createHfsTreatmentResponseProtocol, createHfsTreatmentResponseSpecification } from "../../../packages/condition-profiles/src/hfs-treatment-response.js";
import type { DerivedDataConsentV1, DurableObservationV1 } from "../../../packages/contracts/src/treatment-response.js";

/**
 * Test-only fixture, never imported by an application entry point. Every identity,
 * clinical event, device, and face frame is fabricated. Live source-kind rules
 * exercise unchanged embedded payloads, with a synthetic signed test issuer and
 * an isolated in-memory repository; this is not a live deployment configuration.
 */
export async function createConnectedServiceFixture(startedAtMs: number) {
  let nowMs = startedAtMs;
  const scope = { tenantId: "synthetic-browser-tenant", studyId: "synthetic-browser-study", participantId: "synthetic-browser-patient" };
  const episodeId = "synthetic-browser-episode", encounterId = "synthetic-browser-encounter", platformParticipantId = "synthetic-host-track";
  const principal: Principal = { sub: "synthetic-browser-clinician", tenantId: scope.tenantId,
    role: "clinician", subjectRefs: [scope.participantId], studyIds: [scope.studyId], dataClass: "synthetic",
    permissions: ["episode:create", "capture:write", "consent:write", "clinical:write", "evidence:read", "evidence:review"] };
  const protocol = await createHfsTreatmentResponseProtocol();
  const specification = await createHfsTreatmentResponseSpecification({ scope, anchorTreatmentId: "synthetic-browser-injection" });
  const repository = new MemoryRepository();
  const service = new EncounterService(repository, { mode: "live", now: () => new Date(nowMs).toISOString(), allowedProtocolDigests: [protocol.contentSha256] });
  await service.createEpisode(principal, { id: episodeId, scope, dataClass: "synthetic", protocol, specification });
  const grantTime = new Date(startedAtMs - 2 * 86400000).toISOString();
  const consent: DerivedDataConsentV1 = { consentId: "synthetic-browser-consent", scope,
    documentRef: { id: "synthetic-browser-grant", version: "1", contentSha256: "a".repeat(64) }, grantedAt: grantTime, expiresAt: null, withdrawnAt: null,
    permissions: { capture: true, derivedAnalysis: true, derivedRetention: true, researchClips: false, modalities: ["face"] } };
  await service.write(principal, episodeId, "consents", { data: consent }, "synthetic-grant");
  await service.write(principal, episodeId, "bindings", { data: {
    bindingId: "synthetic-browser-binding", scope, encounterId, platformParticipantId,
    status: "verified", verificationMethod: "clinic-enrollment", recordedAt: grantTime, revokedAt: null
  } }, "synthetic-binding");
  const injection = new Date(startedAtMs - 86400000).toISOString();
  await service.write(principal, episodeId, "clinical-events", { data: {
    schemaVersion: "phenometric.treatment-revision.v1", treatmentId: "synthetic-browser-injection", revisionId: "synthetic-browser-injection-v1", supersedesRevisionId: null,
    scope, courseId: episodeId, cycleId: "synthetic-cycle", recordedAt: injection,
    effectiveTime: { earliest: injection, latest: injection, precision: "instant" }, status: "administered", verification: "verified",
    kind: "botulinum-injection", product: "Fabricated test product", dose: null, route: null, sites: [],
    sourceRef: { system: "synthetic-browser-fixture", resourceId: "injection", versionId: "1" }
  } }, "synthetic-treatment");
  const secret = randomBytes(32), issuer = "synthetic-browser-issuer", audience = "synthetic-browser-service";
  const token = await signDevelopmentSession(principal, secret, issuer, audience);
  const server = createEncounterHttpServer({ service, allowedOrigins: ["http://127.0.0.1:4175"], now: () => nowMs,
    // Fabricated monitored source; proves propagation/renewal, not UTC clock accuracy.
    clockSource: async () => ({ sourceId: "synthetic-browser-monitored-clock", kind: "monitored-utc", maximumUtcErrorMs: 5, validUntilMs: nowMs + 60000 }),
    authenticate: createAuthenticator({ mode: "synthetic", issuer, audience, developmentSecret: secret }) });
  await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("Synthetic service address unavailable");
  return {
    service, repository, scope, episodeId, encounterId, platformParticipantId, token,
    baseUrl: `http://127.0.0.1:${address.port}`,
    setNow(value: number) { nowMs = value; },
    async observations(): Promise<DurableObservationV1[]> {
      const state = await repository.getState(scope.tenantId, episodeId);
      return state!.records.filter(record => record.kind === "session").map(record => record.payload as DurableObservationV1);
    },
    async analyze() { return runOneJob(repository, { now: () => new Date(nowMs).toISOString() }); },
    async withdraw() {
      const state = await repository.getState(scope.tenantId, episodeId);
      const grant = state!.records.find(record => record.kind === "consent")!;
      await service.write(principal, episodeId, "consents", { data: { ...consent, withdrawnAt: new Date(nowMs).toISOString() }, supersedesRecordId: grant.id }, "synthetic-withdrawal");
    },
    async close() { await new Promise<void>(resolve => server.close(() => resolve())); secret.fill(0); }
  };
}
