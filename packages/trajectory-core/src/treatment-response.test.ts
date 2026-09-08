import { describe, expect, it } from "vitest";
import {
  DurableObservationV1Schema, TreatmentResponseProtocolV1Schema, TreatmentResponseSpecificationV1Schema,
  sealTreatmentResponseArtifact, verifyTreatmentResponseArtifact,
  type DurableObservationV1, type TreatmentResponseSnapshotInputV1,
  type TreatmentRevisionV1
} from "@phenometrix/contracts";
import { analyzeTreatmentResponse, createTreatmentResponseSnapshot } from "./treatment-response.js";

const SCOPE = { tenantId: "tenant-a", studyId: "hfs-pilot", participantId: "participant-a" };
const REF = { id: "ambient-local-observation", version: "1.0.0", contentSha256: "a".repeat(64) };
const ORIGIN = Date.parse("2026-08-01T12:00:00Z");
const date = (day: number): string => new Date(ORIGIN + day * 86_400_000).toISOString();

function observation(id: string, day: number, value = 0.3): DurableObservationV1 {
  return DurableObservationV1Schema.parse({
    schemaVersion: "phenometric.durable-observation.v1", observationId: id, revisionId: `${id}-r1`, supersedesRevisionId: null,
    scope: SCOPE, encounterId: `encounter-${id}`, consentId: "consent-a", bindingId: `binding-${id}`,
    startedAt: date(day), endedAt: new Date(Date.parse(date(day)) + 60_000).toISOString(),
    recordedAt: new Date(Date.parse(date(day)) + 65_000).toISOString(), status: "available", measurementProtocolRef: REF,
    capture: { adapterId: "patient-web", adapterVersion: "1.0.0", sourceKind: "patient-local-pre-codec",
      pipelineVersion: "face.1", processorFingerprint: "b".repeat(64), deviceClass: "desktop", clockUncertaintyMs: 0, rawMediaRetained: false },
    windows: [{ windowId: `${id}-w1`, startMs: 0, endMs: 60_000, status: "eligible", reasonCodes: [] }],
    metrics: [{ metricCode: "ambient.face.eye_aperture.left", unit: "eye-width-ratio", modality: "face", context: "ambient-frontal",
      algorithmVersion: "1.0.0", processorRef: "face.1", status: "measured", value, reasonCodes: [], usableDurationMs: 60_000,
      technicalQualityScore: 0.9, sourceWindowIds: [`${id}-w1`] }]
  });
}

function treatment(): TreatmentRevisionV1 {
  return { schemaVersion: "phenometric.treatment-revision.v1", treatmentId: "injection-1", revisionId: "injection-1-r1", supersedesRevisionId: null,
    scope: SCOPE, courseId: "course-1", cycleId: "cycle-1", recordedAt: date(0), effectiveTime: { earliest: date(0), latest: date(0), precision: "instant" },
    status: "administered", verification: "verified", kind: "botulinum-injection", product: "onabotulinumtoxinA", dose: { value: 20, unit: "product-specific-unit" }, route: "intramuscular",
    sites: [{ site: "orbicularis-oculi", laterality: "left", dose: { value: 20, unit: "product-specific-unit" } }],
    sourceRef: { system: "test-fhir", resourceId: "medication-administration-1", versionId: "1" } };
}

function history(observations = [observation("before-1", -14, 0.2), observation("before-2", -7, 0.4), observation("after", 7, 0.5)]): TreatmentResponseSnapshotInputV1 {
  return { scope: SCOPE, asOf: date(100), observations, treatments: [treatment()], contexts: [],
    consents: [{ consentId: "consent-a", scope: SCOPE, documentRef: { id: "research-consent", version: "1", contentSha256: "c".repeat(64) },
      grantedAt: date(-100), expiresAt: null, withdrawnAt: null, permissions: { capture: true, derivedAnalysis: true, derivedRetention: true, researchClips: false, modalities: ["face"] } }],
    bindings: observations.map(record => ({ bindingId: record.bindingId, scope: SCOPE, encounterId: record.encounterId, platformParticipantId: `platform-${record.observationId}`,
      status: "verified", verificationMethod: "authenticated-patient-portal", recordedAt: date(-100), revokedAt: null })) };
}

async function protocol() {
  return sealTreatmentResponseArtifact(TreatmentResponseProtocolV1Schema.parse({ schemaVersion: "phenometric.treatment-response-protocol.v1", protocolId: "hfs-test", version: "1", contentSha256: "0".repeat(64),
    status: "research-only", condition: "established-hemifacial-spasm", intendedUse: "Descriptive research", targetPopulation: "Consented research participants", measurementProtocolRef: REF,
    permittedSourceKinds: ["patient-local-pre-codec"], metrics: [{ metricCode: "ambient.face.eye_aperture.left", label: "Left eye aperture", unit: "eye-width-ratio", modality: "face", context: "ambient-frontal", algorithmVersion: "1.0.0", minimumUsableDurationMs: 30_000, minimumTechnicalQualityScore: 0.5 }],
    referenceStandard: "Not validated", humanWorkflow: "Nonblocking clinician inspection", clinicalValidation: "none", validatedClaim: "none", uncertainty: "analytical-repeatability-and-minimum-detectable-change-unknown", prohibitedClaims: ["causal-effect"] }));
}

async function setup(input = history()) {
  const pack = await protocol();
  const specification = await sealTreatmentResponseArtifact(TreatmentResponseSpecificationV1Schema.parse({ schemaVersion: "phenometric.treatment-response-specification.v1", specificationId: "spec-1", version: "1", contentSha256: "0".repeat(64), scope: SCOPE,
    protocolRef: { id: pack.protocolId, version: pack.version, contentSha256: pack.contentSha256 }, anchorTreatmentId: "injection-1",
    baseline: { startDay: -30, endDay: 0, minimumObservations: 2, minimumEncounters: 2 }, followupEndDay: 112,
    phases: [{ phaseId: "early", startDay: 0, endDay: 14, minimumCount: 1 }, { phaseId: "later", startDay: 14, endDay: 112, minimumCount: 1 }],
    maximumClockUncertaintyMs: 1_000, concurrentTreatmentPolicy: "flag", sourceCompatibility: "exact", baselineAggregation: "median-of-encounter-medians", fittedModels: false, clinicalClaim: "descriptive-only" }));
  return { snapshot: await createTreatmentResponseSnapshot(input), protocol: pack, specification };
}

describe("treatment-aligned descriptive analysis", () => {
  it("replays identically from unordered history, freezes artifacts and preserves exact point provenance", async () => {
    const first = await setup();
    const shuffled = history(); shuffled.observations.reverse(); shuffled.bindings.reverse();
    const second = await setup(shuffled);
    expect(second.snapshot).toEqual(first.snapshot);
    const run = await analyzeTreatmentResponse(first);
    expect(await analyzeTreatmentResponse(second)).toEqual(run);
    expect(await verifyTreatmentResponseArtifact(run)).toBe(true);
    expect(Object.isFrozen(run.rows[0].points[0])).toBe(true);
    expect(run.rows[0].baseline.value).toBeCloseTo(0.3);
    expect(run.rows[0].baseline.observationRevisionIds).toEqual(["before-1-r1", "before-2-r1"]);
    expect(run.rows[0].points.at(-1)).toMatchObject({ value: 0.5, delta: 0.5 - run.rows[0].baseline.value!, context: "ambient-frontal", technicalQualityScore: 0.9, usableDurationMs: 60_000, processorRef: "face.1", sourceWindowIds: ["after-w1"] });
    expect(run.rows[0].phaseCoverage).toEqual([{ phaseId: "early", count: 1, minimumCount: 1, status: "sufficient" }, { phaseId: "later", count: 0, minimumCount: 1, status: "missing" }]);
    expect(run.clinicalClaim).toBe("descriptive-only"); expect(run.fittedModels).toBe(false);
  });

  it.each(["missing", "unknown-date", "planned", "unverified"])("keeps calendar measurements with unavailable anchor: %s", async mode => {
    const input = history();
    if (mode === "missing") input.treatments = [];
    if (mode === "unknown-date") input.treatments[0].effectiveTime = { earliest: null, latest: null, precision: "unknown" };
    if (mode === "planned") input.treatments[0].status = "planned";
    if (mode === "unverified") input.treatments[0].verification = "unverified";
    const run = await analyzeTreatmentResponse(await setup(input));
    expect(run.status).toBe("anchor-unavailable"); expect(run.rows[0].points).toHaveLength(3);
    expect(run.rows[0].points.every(point => point.delta === null && point.daysSinceTreatment === null && point.phaseIds.length === 0)).toBe(true);
  });

  it("requires separate baseline encounters and does not mistake repeat captures for phase coverage", async () => {
    const input = history();
    input.observations[1].encounterId = input.observations[0].encounterId;
    input.bindings[1].encounterId = input.observations[0].encounterId;
    const run = await analyzeTreatmentResponse(await setup(input));
    expect(run.rows[0].baseline).toMatchObject({ status: "insufficient-data", value: null, encounterCount: 1 });
    expect(run.rows[0].points.every(point => point.delta === null)).toBe(true);
  });

  it("uses encounter medians so repeated captures cannot overweight the baseline", async () => {
    const input = history([observation("before-1", -14, 0.1), observation("before-1b", -14, 0.3), observation("before-1c", -14, 0.5), observation("before-2", -7, 0.8), observation("after", 7)]);
    for (const index of [1, 2]) { input.observations[index].encounterId = input.observations[0].encounterId; input.bindings[index].encounterId = input.observations[0].encounterId; }
    const run = await analyzeTreatmentResponse(await setup(input)); expect(run.rows[0].baseline.value).toBe(0.55);
  });

  it("shows withholding and version/context/pipeline incompatibility explicitly", async () => {
    const input = history([observation("before-1", -14), observation("before-2", -7), observation("withheld", 5), observation("new-version", 6), observation("changed-pipeline", 7), observation("different-context", 8)]);
    Object.assign(input.observations[2].metrics[0], { status: "withheld", value: null, reasonCodes: ["insufficient-events"] });
    input.observations[3].metrics[0].algorithmVersion = "2.0.0";
    input.observations[4].capture.processorFingerprint = "d".repeat(64);
    input.observations[5].metrics[0].context = "prompted-fixation";
    const run = await analyzeTreatmentResponse(await setup(input));
    const reasons = Object.fromEntries(run.rows[0].exclusions.map(exclusion => [exclusion.observationId, exclusion.reasonCodes]));
    expect(reasons.withheld).toContain("metric-withheld"); expect(reasons["new-version"]).toContain("metric-definition-mismatch");
    expect(reasons["changed-pipeline"]).toEqual(["source-incompatible"]); expect(reasons["different-context"]).toContain("metric-definition-mismatch");
    expect(run.rows[0].status).toBe("insufficient-followup"); expect(run.status).toBe("insufficient-data");
  });

  it("requires applicable modality consent and verified encounter binding", async () => {
    const input = history(); input.consents[0].permissions.modalities = ["voice"];
    input.bindings[0].status = "unverified"; input.bindings[1].encounterId = "different-encounter";
    const run = await analyzeTreatmentResponse(await setup(input)); expect(run.rows[0].points).toHaveLength(0);
    expect(run.rows[0].exclusions.find(exclusion => exclusion.observationId === "before-1")?.reasonCodes).toContain("binding-not-verified");
    expect(run.rows[0].exclusions.find(exclusion => exclusion.observationId === "before-2")?.reasonCodes).toContain("binding-encounter-mismatch");
    expect(run.rows[0].exclusions.every(exclusion => exclusion.reasonCodes.includes("consent-not-applicable"))).toBe(true);
  });
  it("never treats a synthetic clock as qualification for live acquisition", async () => {
    const input = history();
    input.observations[2].capture.clockSource = { sourceId: "fixture-clock", kind: "synthetic" };
    const run = await analyzeTreatmentResponse(await setup(input));
    expect(run.rows[0].points.map(point => point.observationId)).not.toContain("after");
    expect(run.rows[0].exclusions.find(exclusion => exclusion.observationId === "after")?.reasonCodes).toContain("clock-uncertain");
  });

  it("withdrawal blocks later analysis and late consent cannot authorize earlier capture", async () => {
    const withdrawn = history(); withdrawn.consents[0].withdrawnAt = date(8);
    expect((await analyzeTreatmentResponse(await setup(withdrawn))).rows[0].points).toHaveLength(0);
    const late = history(); late.consents[0].grantedAt = date(0);
    expect((await analyzeTreatmentResponse(await setup(late))).rows[0].points.map(point => point.observationId)).toEqual(["after"]);
  });

  it("retains original as-of replay while a corrected injection changes new alignment", async () => {
    const input = history();
    input.treatments.push({ ...treatment(), revisionId: "injection-1-r2", supersedesRevisionId: "injection-1-r1", recordedAt: date(20), effectiveTime: { earliest: date(2), latest: date(2), precision: "instant" } });
    const early = { ...input, asOf: date(10) };
    const oldRun = await analyzeTreatmentResponse(await setup(early));
    const newRun = await analyzeTreatmentResponse(await setup(input));
    expect(oldRun.anchor.revisionId).toBe("injection-1-r1"); expect(newRun.anchor.revisionId).toBe("injection-1-r2");
    expect(oldRun.rows[0].points.at(-1)?.daysSinceTreatment?.minimum).toBe(7);
    expect(newRun.rows[0].points.at(-1)?.daysSinceTreatment?.minimum).toBe(5);
    expect(oldRun.runId).not.toBe(newRun.runId);
    expect(oldRun.contextFlags.some(flag => flag.kind === "source-correction")).toBe(false);
    expect(newRun.contextFlags).toContainEqual({ revisionId: "injection-1-r2", kind: "source-correction", reason: "analysis-input-history-includes-corrected-source" });
  });

  it("supersedes corrected observations without counting them twice", async () => {
    const input = history(); const original = input.observations[2];
    input.observations.push({ ...original, revisionId: "after-r2", supersedesRevisionId: original.revisionId, recordedAt: date(9), metrics: [{ ...original.metrics[0], value: 0.6 }] });
    const run = await analyzeTreatmentResponse(await setup(input));
    expect(run.rows[0].points.filter(point => point.observationId === "after")).toHaveLength(1);
    expect(run.rows[0].points.at(-1)?.value).toBe(0.6);
    expect(run.rows[0].exclusions.find(exclusion => exclusion.revisionId === "after-r1")?.reasonCodes).toContain("superseded");
  });

  it("refuses revision forks, cross-scope records and tampered artifacts", async () => {
    const forked = history();
    forked.treatments.push({ ...treatment(), revisionId: "injection-r2a", supersedesRevisionId: "injection-1-r1" }, { ...treatment(), revisionId: "injection-r2b", supersedesRevisionId: "injection-1-r1" });
    await expect(createTreatmentResponseSnapshot(forked)).rejects.toThrow("revision-fork");
    const crossScope = history(); crossScope.observations[0].scope = { ...SCOPE, tenantId: "other-tenant" };
    await expect(createTreatmentResponseSnapshot(crossScope)).rejects.toThrow("Cross-scope");
    const original = await setup(); const tampered = structuredClone(original); tampered.snapshot.observations[0].metrics[0].value = 999;
    await expect(analyzeTreatmentResponse(tampered)).rejects.toThrow("digest-mismatch");
  });

  it("keeps date-only injection intervals rather than inventing a precise treatment time", async () => {
    const input = history(); input.treatments[0].effectiveTime = { earliest: "2026-08-01T00:00:00Z", latest: "2026-08-01T23:59:59.999Z", precision: "day" };
    const run = await analyzeTreatmentResponse(await setup(input)); const point = run.rows[0].points.at(-1)!;
    expect(point.daysSinceTreatment!.minimum).toBeCloseTo(6.5); expect(point.daysSinceTreatment!.maximum).toBeGreaterThan(7.5);
  });

  it("flags carryover and concurrent changes without interpreting them as causal adjustment", async () => {
    const input = history(); input.treatments.push({ ...treatment(), treatmentId: "previous-cycle", revisionId: "previous-cycle-r1", cycleId: "cycle-0", recordedAt: date(-70), effectiveTime: { earliest: date(-70), latest: date(-70), precision: "instant" } });
    input.contexts.push({ schemaVersion: "phenometric.clinical-context-revision.v1", contextId: "illness", revisionId: "illness-r1", supersedesRevisionId: null, scope: SCOPE, recordedAt: date(3), effectiveTime: { earliest: date(3), latest: date(3), precision: "instant" }, status: "active", kind: "intercurrent-illness", valueCode: "intercurrent-illness-present", sourceRef: { system: "ehr", resourceId: "context-1", versionId: "1" } });
    const config = await setup(input); const run = await analyzeTreatmentResponse(config);
    expect(run.contextFlags.map(flag => flag.reason)).toContain("prior-treatment-carryover-not-estimated");
    expect(run.contextFlags.map(flag => flag.reason)).toContain("clinical-context-not-causally-adjusted");
    const censored = await analyzeTreatmentResponse({ ...config, specification: await sealTreatmentResponseArtifact({ ...config.specification, concurrentTreatmentPolicy: "exclude-after-change" }) });
    expect(censored.rows[0].exclusions.find(exclusion => exclusion.observationId === "after")?.reasonCodes).toContain("concurrent-treatment-change");
  });

  it("ends index-cycle eligibility at the next documented injection even under flag-only confound policy", async () => {
    const input = history([observation("before-1", -14), observation("before-2", -7), observation("after", 7), observation("next-cycle", 30)]);
    input.treatments.push({ ...treatment(), treatmentId: "injection-2", revisionId: "injection-2-r1", cycleId: "cycle-2", recordedAt: date(28), effectiveTime: { earliest: date(28), latest: date(28), precision: "instant" } });
    const run = await analyzeTreatmentResponse(await setup(input));
    expect(run.rows[0].points.map(point => point.observationId)).toContain("after");
    expect(run.rows[0].exclusions.find(exclusion => exclusion.observationId === "next-cycle")?.reasonCodes).toContain("subsequent-treatment-cycle");
  });

  it("fails closed on numerical overflow without manufacturing an infinite delta", async () => {
    const input = history([observation("before-1", -14, 1e308), observation("before-2", -7, 1e308), observation("after", 7, -1e308)]);
    const run = await analyzeTreatmentResponse(await setup(input));
    expect(run.rows[0].exclusions.find(exclusion => exclusion.observationId === "after")?.reasonCodes).toEqual(["non-finite-delta"]);
  });
});
