import { describe, it, expect, vi } from "vitest";
import { sealTreatmentResponseArtifact, TreatmentResponseRunV1Schema, DurableObservationV1Schema } from "@phenometrix/contracts";
import { createEncounterClient } from "./index.js";
const scope = { tenantId: "tenant", studyId: "study", participantId: "patient" };
const ref = { id: "protocol", version: "1", contentSha256: "a".repeat(64) };
const pending = (revision = 1) => ({ episode: { id: "episode", tenantId: "tenant", subjectRef: "patient", revision, metadata: { studyId: "study" } }, inputRevision: revision, status: "pending", snapshot: null });
const fromResponse = (value: unknown) => createEncounterClient({ baseUrl: "https://clinic.example", episodeId: "episode", scope, accessToken: async () => "token", fetch: (async () => new Response(JSON.stringify(value))) as typeof fetch });
async function ready() {
  const run = await sealTreatmentResponseArtifact(TreatmentResponseRunV1Schema.parse({ schemaVersion: "phenometric.treatment-response-run.v1", runId: "run", contentSha256: "0".repeat(64), scope,
    snapshotId: "snapshot", snapshotSha256: "b".repeat(64), protocolRef: ref, specificationRef: ref, engineVersion: "1", generatedAt: "2026-01-01T00:00:00Z",
    status: "anchor-unavailable", anchor: { treatmentId: "treatment", revisionId: null, cycleId: null, effectiveTime: null, reasonCodes: ["missing"] },
    rows: [{ metricCode: "aperture", label: "Aperture", unit: "ratio", status: "no-comparable-data", baseline: { status: "insufficient-data", value: null, observationRevisionIds: [], encounterCount: 0, compatibilityFingerprint: null, reasonCodes: ["missing"] },
      points: [], exclusions: [], phaseCoverage: [], uncertainty: { measurementError: "unknown", minimumDetectableChange: "unknown", confidenceInterval: null } }],
    contextFlags: [], clinicalClaim: "descriptive-only", fittedModels: false }));
  return { ...pending(), status: "ready", episode: { ...pending().episode, metadata: { studyId: "study", protocol: { protocolId: ref.id, version: ref.version, contentSha256: ref.contentSha256 }, specification: { specificationId: ref.id, version: ref.version, contentSha256: ref.contentSha256 } } },
    snapshot: { episodeId: "episode", inputRevision: 1, analysis: run }, reviews: [] as unknown[] };
}
describe("host-authorized client", () => {
  it("uses a bearer header and returns no stale evidence while analysis is pending", async () => {
    const client = createEncounterClient({ baseUrl: "https://clinic.example/phenometrix", episodeId: "episode", scope, accessToken: async () => "synthetic-token",
      fetch: (async (url, options) => {
        expect(String(url)).toBe("https://clinic.example/phenometrix/v1/episodes/episode/evidence");
        expect(options?.headers).toMatchObject({ Authorization: "Bearer synthetic-token" });
        expect(options?.redirect).toBe("error");
        return new Response(JSON.stringify(pending()));
      }) as typeof fetch });
    expect(await client.loadEvidence()).toEqual({ run: null, reviews: [] });
  });
  it("rejects mixed patient context before returning data to the host", async () => {
    const client = createEncounterClient({ baseUrl: "https://clinic.example", episodeId: "episode", scope, accessToken: async () => "synthetic-token",
      fetch: (async () => new Response(JSON.stringify({ episode: { id: "episode", tenantId: "tenant", subjectRef: "other", metadata: { studyId: "study" } }, inputRevision: 1, status: "pending" }))) as typeof fetch });
    await expect(client.loadEvidence()).rejects.toThrow("scope");
  });
  it("rejects insecure remote endpoints and URL-borne credentials", () => {
    for (const baseUrl of ["http://remote.example", "https://user:secret@clinic.example", "https://clinic.example?token=secret"]) {
      expect(() => createEncounterClient({ baseUrl, episodeId: "episode", scope, accessToken: async () => "token" })).toThrow();
    }
  });
  it("rejects unknown statuses and contradictory ready/current revision envelopes", async () => {
    for (const payload of [{ ...pending(), status: "surprise" }, { ...pending(), status: "ready" }, { ...pending(), inputRevision: 2 }, { ...pending(), snapshot: {} }]) {
      await expect(fromResponse(payload).loadEvidence()).rejects.toThrow();
    }
  });
  it("prevents an older concurrent response from replacing a newer accepted revision", async () => {
    let finishFirst!: (response: Response) => void, calls = 0;
    const client = createEncounterClient({ baseUrl: "https://clinic.example", episodeId: "episode", scope, accessToken: async () => "token",
      fetch: (() => ++calls === 1 ? new Promise<Response>(resolve => { finishFirst = resolve; }) : Promise.resolve(new Response(JSON.stringify(pending(2))))) as typeof fetch });
    const older = client.loadEvidence();
    await vi.waitFor(() => expect(calls).toBe(1));
    expect(await client.loadEvidence()).toEqual({ run: null, reviews: [] });
    finishFirst(new Response(JSON.stringify(pending(1))));
    await expect(older).rejects.toThrow("stale");
  });
  it("bounds token retrieval and prevents canceled or late fetch responses from being delivered", async () => {
    const fetcher = vi.fn();
    const waiting = createEncounterClient({ baseUrl: "https://clinic.example", episodeId: "episode", scope, timeoutMs: 10, accessToken: () => new Promise(() => {}), fetch: fetcher });
    await expect(waiting.loadEvidence()).rejects.toMatchObject({ name: "TimeoutError" });
    expect(fetcher).not.toHaveBeenCalled();
    const controller = new AbortController();
    let finish!: (response: Response) => void;
    const delayed = createEncounterClient({ baseUrl: "https://clinic.example", episodeId: "episode", scope, accessToken: async () => "token",
      fetch: (() => new Promise<Response>(resolve => { finish = resolve; })) as typeof fetch });
    const result = delayed.loadEvidence(controller.signal);
    await vi.waitFor(() => expect(finish).toBeTypeOf("function"));
    controller.abort();
    await expect(result).rejects.toMatchObject({ name: "AbortError" });
    finish(new Response(JSON.stringify(pending())));
  });
  it("validates sealed provenance and binds reviews to the returned run", async () => {
    const payload = await ready(), run = payload.snapshot.analysis;
    const review = { schemaVersion: "phenometric.treatment-response-review.v1", reviewId: "review", scope, runId: run.runId, runSha256: run.contentSha256, actorId: "clinician", recordedAt: "2026-02-01T00:00:00Z", disposition: "acknowledged", supersedesReviewId: null, reasonCode: null };
    payload.reviews = [review, { ...review, reviewId: "old-review", runId: "old-run" }];
    expect((await fromResponse(payload).loadEvidence()).reviews).toEqual([review]);
    payload.reviews = [{ ...review, runSha256: "c".repeat(64) }];
    await expect(fromResponse(payload).loadEvidence()).rejects.toThrow("review-mismatch");
    payload.reviews = [{ ...review, scope: { ...scope, participantId: "other" } }];
    await expect(fromResponse(payload).loadEvidence()).rejects.toThrow("scope-mismatch");
    payload.reviews = []; payload.episode.metadata.protocol.contentSha256 = "d".repeat(64);
    await expect(fromResponse(payload).loadEvidence()).rejects.toThrow("provenance-mismatch");
  });
  it("rejects revoked capture contexts before handing them to capture", async () => {
    const consent = { consentId: "consent", scope, documentRef: ref, grantedAt: "2020-01-01T00:00:00Z", expiresAt: null, withdrawnAt: "2021-01-01T00:00:00Z", permissions: { capture: true, derivedAnalysis: true, derivedRetention: true, researchClips: false, modalities: ["face"] } };
    const binding = { bindingId: "binding", scope, encounterId: "encounter", platformParticipantId: "platform", status: "verified", verificationMethod: "authenticated-patient-portal", recordedAt: "2020-01-01T00:00:00Z", revokedAt: null };
    const payload = { scope, episodeId: "episode", inputRevision: 1, consent, binding, measurementProtocolRef: ref };
    await expect(fromResponse(payload).captureContext("encounter")).rejects.toThrow("capture-unavailable");
    expect((await fromResponse({ ...payload, consent: { ...consent, withdrawnAt: null } }).captureContext("encounter")).measurementProtocolRef).toEqual(ref);
  });
  it("requires ingestion acknowledgements to identify the exact submitted record", async () => {
    const observation = DurableObservationV1Schema.parse({ schemaVersion: "phenometric.durable-observation.v1", observationId: "observation", revisionId: "revision", supersedesRevisionId: null, scope,
      encounterId: "encounter", consentId: "consent", bindingId: "binding", recordedAt: "2026-01-01T00:01:00Z", startedAt: "2026-01-01T00:00:00Z", endedAt: "2026-01-01T00:01:00Z", status: "available", measurementProtocolRef: ref,
      capture: { adapterId: "patient-web", adapterVersion: "1", sourceKind: "patient-local-pre-codec", pipelineVersion: "1", processorFingerprint: "a".repeat(64), deviceClass: "desktop", clockUncertaintyMs: 0, rawMediaRetained: false },
      windows: [{ windowId: "window", startMs: 0, endMs: 60_000, status: "eligible", reasonCodes: [] }],
      metrics: [{ metricCode: "ambient.face.eye_aperture.left", unit: "eye-width-ratio", modality: "face", context: "ambient-frontal", algorithmVersion: "1", processorRef: "face.1", status: "measured", value: 0.4,
        reasonCodes: [], usableDurationMs: 60_000, technicalQualityScore: 0.9, sourceWindowIds: ["window"] }] });
    const response = { record: { episodeId: "episode", kind: "session", logicalId: "observation", idempotencyKey: "revision", payload: observation, revision: 1 }, duplicate: false };
    expect(await fromResponse(response).ingestObservation(observation)).toEqual(response);
    await expect(fromResponse({ ...response, record: { ...response.record, episodeId: "other" } }).ingestObservation(observation)).rejects.toThrow("ingest-mismatch");
    await expect(fromResponse({ ...response, record: { ...response.record, payload: { ...observation, observationId: "other" } } }).ingestObservation(observation)).rejects.toThrow("ingest-mismatch");
  });
});
