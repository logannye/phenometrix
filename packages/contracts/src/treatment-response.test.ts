import { describe, expect, it } from "vitest";
import {
  ClinicalEffectiveTimeV1Schema, DerivedDataConsentV1Schema, DurableObservationV1Schema,
  ParticipantBindingV1Schema, TreatmentResponseReviewV1Schema,
  canonicalTreatmentResponseJson, sealTreatmentResponseArtifact, verifyTreatmentResponseArtifact
} from "./treatment-response.js";

const scope = { tenantId: "clinic", studyId: "pilot", participantId: "participant" };
const ref = { id: "protocol", version: "1", contentSha256: "a".repeat(64) };
const start = "2026-09-01T12:00:00.000Z";
const end = "2026-09-01T12:01:00.000Z";
function observation() {
  return DurableObservationV1Schema.parse({
    schemaVersion: "phenometric.durable-observation.v1", observationId: "observation", revisionId: "observation-r1", supersedesRevisionId: null,
    scope, encounterId: "encounter", consentId: "consent", bindingId: "binding", recordedAt: end, startedAt: start, endedAt: end,
    status: "available", measurementProtocolRef: ref,
    capture: { adapterId: "client", adapterVersion: "1", sourceKind: "patient-local-pre-codec", pipelineVersion: "1", processorFingerprint: "b".repeat(64), deviceClass: "desktop", clockUncertaintyMs: 50, rawMediaRetained: false },
    windows: [{ windowId: "window-1", startMs: 0, endMs: 30_000, status: "eligible", reasonCodes: [] }, { windowId: "window-2", startMs: 30_000, endMs: 60_000, status: "eligible", reasonCodes: [] }],
    metrics: [{ metricCode: "ambient.face.eye_aperture.left", modality: "face", unit: "eye-width-ratio", context: "ambient-frontal", algorithmVersion: "1", processorRef: "face",
      status: "measured", value: 0.3, reasonCodes: [], usableDurationMs: 60_000, technicalQualityScore: 0.9, sourceWindowIds: ["window-1", "window-2"] }]
  });
}

describe("governed treatment-response contracts", () => {
  it("rejects raw media, opaque extra state and contradictory measured/withheld values", () => {
    expect(DurableObservationV1Schema.safeParse({ ...observation(), rawVideo: "pixels" }).success).toBe(false);
    const raw = observation(); raw.capture.rawMediaRetained = true as false;
    expect(DurableObservationV1Schema.safeParse(raw).success).toBe(false);
    const withheld = observation(); withheld.metrics[0].status = "withheld"; withheld.metrics[0].reasonCodes = ["no-signal"];
    expect(DurableObservationV1Schema.safeParse(withheld).success).toBe(false);
    withheld.metrics[0].value = null; expect(DurableObservationV1Schema.safeParse(withheld).success).toBe(true);
  });

  it("requires real eligible window attribution and computes duration from a union", () => {
    const duplicatedCoverage = observation(); duplicatedCoverage.windows[1].startMs = 0; duplicatedCoverage.windows[1].endMs = 30_000;
    expect(DurableObservationV1Schema.safeParse(duplicatedCoverage).success).toBe(false);
    const missing = observation(); missing.metrics[0].sourceWindowIds = ["unknown"];
    expect(DurableObservationV1Schema.safeParse(missing).success).toBe(false);
    const outside = observation(); outside.windows[0].endMs = 70_000;
    expect(DurableObservationV1Schema.safeParse(outside).success).toBe(false);
  });

  it("retains explicit derived temporal samples only inside their own referenced windows", () => {
    const sampled = observation(); sampled.metrics[0].temporalSamples = [{ sampleId: "sample-1", startMs: 0, endMs: 5_000, value: 0.29, sourceWindowIds: ["window-1"] }];
    expect(DurableObservationV1Schema.safeParse(sampled).success).toBe(true);
    sampled.metrics[0].temporalSamples[0].sourceWindowIds = ["window-2"];
    expect(DurableObservationV1Schema.safeParse(sampled).success).toBe(false);
    expect(observation().metrics[0].temporalSamples).toEqual([]);
  });

  it("requires purpose and modality-specific consent without assuming microphone permission", () => {
    const consent = { consentId: "consent", scope, documentRef: ref, grantedAt: start, expiresAt: null, withdrawnAt: null,
      permissions: { capture: true, derivedAnalysis: true, derivedRetention: true, researchClips: false, modalities: ["face"] } };
    expect(DerivedDataConsentV1Schema.parse(consent).permissions.modalities).toEqual(["face"]);
    expect(DerivedDataConsentV1Schema.safeParse({ ...consent, permissions: { ...consent.permissions, modalities: [] } }).success).toBe(false);
    expect(DerivedDataConsentV1Schema.safeParse({ ...consent, expiresAt: "2026-08-01T00:00:00Z" }).success).toBe(false);
    expect(DerivedDataConsentV1Schema.safeParse({ ...consent, permissions: { ...consent.permissions, modalities: ["face", "face"] } }).success).toBe(false);
  });

  it("rejects contradictory identity binding or fabricated precision", () => {
    expect(ParticipantBindingV1Schema.safeParse({ bindingId: "binding", scope, encounterId: "encounter", platformParticipantId: "platform-patient", status: "verified", verificationMethod: "unverified", recordedAt: start, revokedAt: null }).success).toBe(false);
    expect(ClinicalEffectiveTimeV1Schema.safeParse({ precision: "unknown", earliest: start, latest: end }).success).toBe(false);
    expect(ClinicalEffectiveTimeV1Schema.safeParse({ precision: "instant", earliest: start, latest: end }).success).toBe(false);
    expect(ClinicalEffectiveTimeV1Schema.safeParse({ precision: "day", earliest: start, latest: "2026-09-05T00:00:00Z" }).success).toBe(false);
  });

  it("canonicalizes key order and detects edited artifact content", async () => {
    expect(canonicalTreatmentResponseJson({ z: 2, a: [1, { b: true, a: null }] })).toBe(canonicalTreatmentResponseJson({ a: [1, { a: null, b: true }], z: 2 }));
    expect(() => canonicalTreatmentResponseJson({ value: undefined })).toThrow();
    expect(() => canonicalTreatmentResponseJson({ value: Infinity })).toThrow();
    const sealed = await sealTreatmentResponseArtifact({ contentSha256: "0".repeat(64), id: "artifact", value: 1 });
    expect(await verifyTreatmentResponseArtifact(sealed)).toBe(true);
    expect(await verifyTreatmentResponseArtifact({ ...sealed, value: 2 })).toBe(false);
  });

  it("binds review to a specific run and forbids a clinical execution action", () => {
    const review = { schemaVersion: "phenometric.treatment-response-review.v1", reviewId: "review", scope, runId: "run", runSha256: "c".repeat(64), actorId: "clinician", recordedAt: end, disposition: "acknowledged", supersedesReviewId: null, reasonCode: null };
    expect(TreatmentResponseReviewV1Schema.safeParse(review).success).toBe(true);
    expect(TreatmentResponseReviewV1Schema.safeParse({ ...review, disposition: "execute-treatment" }).success).toBe(false);
  });
});
