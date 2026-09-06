import { z } from "zod";
import { calculateSha256Hex } from "./ambient-protocol.js";

const Id = z.string().min(1).max(160).regex(/^[A-Za-z0-9._:-]+$/);
const Timestamp = z.string().datetime({ offset: true });
const Digest = z.string().regex(/^[a-f0-9]{64}$/);
const NonNegative = z.number().finite().nonnegative();
const Quality = z.number().finite().min(0).max(1);

export const TreatmentResponseScopeV1Schema = z.object({
  tenantId: Id, studyId: Id, participantId: Id
}).strict();
export type TreatmentResponseScopeV1 = z.infer<typeof TreatmentResponseScopeV1Schema>;

export const VersionedArtifactRefV1Schema = z.object({
  id: Id, version: z.string().min(1), contentSha256: Digest
}).strict();
export type VersionedArtifactRefV1 = z.infer<typeof VersionedArtifactRefV1Schema>;

export const ClinicalSourceRefV1Schema = z.object({
  system: z.string().min(1), resourceId: z.string().min(1), versionId: z.string().min(1)
}).strict();
export type ClinicalSourceRefV1 = z.infer<typeof ClinicalSourceRefV1Schema>;

export const DerivedDataConsentV1Schema = z.object({
  consentId: Id, scope: TreatmentResponseScopeV1Schema,
  documentRef: VersionedArtifactRefV1Schema,
  grantedAt: Timestamp, expiresAt: Timestamp.nullable(), withdrawnAt: Timestamp.nullable(),
  permissions: z.object({
    capture: z.boolean(), derivedAnalysis: z.boolean(), derivedRetention: z.boolean(),
    researchClips: z.boolean(), modalities: z.array(z.enum(["face", "voice"]))
  }).strict()
}).strict().superRefine((consent, ctx) => {
  const granted = Date.parse(consent.grantedAt);
  if ((consent.expiresAt !== null && Date.parse(consent.expiresAt) <= granted) || (consent.withdrawnAt !== null && Date.parse(consent.withdrawnAt) < granted)) ctx.addIssue({ code: "custom", message: "Consent expiry and withdrawal must not precede its grant." });
  if (new Set(consent.permissions.modalities).size !== consent.permissions.modalities.length || (consent.permissions.capture && consent.permissions.modalities.length === 0)) ctx.addIssue({ code: "custom", message: "Capture consent must enumerate unique permitted modalities." });
});
export type DerivedDataConsentV1 = z.infer<typeof DerivedDataConsentV1Schema>;

export const ParticipantBindingV1Schema = z.object({
  bindingId: Id, scope: TreatmentResponseScopeV1Schema, encounterId: Id,
  platformParticipantId: Id,
  status: z.enum(["verified", "unverified", "revoked"]),
  verificationMethod: z.enum(["clinic-enrollment", "authenticated-patient-portal", "synthetic-fixture", "unverified"]),
  recordedAt: Timestamp, revokedAt: Timestamp.nullable()
}).strict().superRefine((binding, ctx) => {
  if ((binding.status === "verified" && binding.verificationMethod === "unverified") || (binding.status === "revoked" && binding.revokedAt === null) || (binding.revokedAt !== null && Date.parse(binding.revokedAt) < Date.parse(binding.recordedAt))) ctx.addIssue({ code: "custom", message: "Binding status, verification and revocation chronology must agree." });
});
export type ParticipantBindingV1 = z.infer<typeof ParticipantBindingV1Schema>;

export const DurableMetricV1Schema = z.object({
  metricCode: z.string().min(1), unit: z.string().min(1), context: z.string().min(1),
  modality: z.enum(["face", "voice"]),
  algorithmVersion: z.string().min(1), processorRef: z.string().min(1),
  status: z.enum(["measured", "withheld"]), value: z.number().finite().nullable(),
  reasonCodes: z.array(z.string().min(1)), usableDurationMs: NonNegative,
  technicalQualityScore: Quality.nullable(), sourceWindowIds: z.array(Id),
  temporalSamples: z.array(z.object({
    sampleId: Id, startMs: NonNegative, endMs: NonNegative, value: z.number().finite(),
    sourceWindowIds: z.array(Id).min(1)
  }).strict()).default([])
}).strict().superRefine((metric, ctx) => {
  if (metric.status === "measured" && (metric.value === null || metric.technicalQualityScore === null || metric.sourceWindowIds.length === 0 || metric.reasonCodes.length > 0)) {
    ctx.addIssue({ code: "custom", message: "Measured metrics require a value, quality and windows, without withholding reasons." });
  }
  if (metric.status === "withheld" && (metric.value !== null || metric.reasonCodes.length === 0)) {
    ctx.addIssue({ code: "custom", message: "Withheld metrics require reasons and cannot carry a value." });
  }
});
export type DurableMetricV1 = z.infer<typeof DurableMetricV1Schema>;

export const DurableObservationV1Schema = z.object({
  schemaVersion: z.literal("phenometric.durable-observation.v1"),
  observationId: Id, revisionId: Id, supersedesRevisionId: Id.nullable(),
  scope: TreatmentResponseScopeV1Schema, encounterId: Id,
  consentId: Id, bindingId: Id,
  recordedAt: Timestamp, startedAt: Timestamp, endedAt: Timestamp,
  status: z.enum(["available", "entered-in-error"]),
  measurementProtocolRef: VersionedArtifactRefV1Schema,
  capture: z.object({
    adapterId: Id, adapterVersion: z.string().min(1),
    sourceKind: z.enum(["patient-local-pre-codec", "platform-patient-track", "research-upload", "synthetic"]),
    pipelineVersion: z.string().min(1), processorFingerprint: Digest,
    deviceClass: z.string().min(1), clockUncertaintyMs: NonNegative,
    rawMediaRetained: z.literal(false)
  }).strict(),
  windows: z.array(z.object({
    windowId: Id, startMs: NonNegative, endMs: NonNegative,
    status: z.enum(["eligible", "withheld"]), reasonCodes: z.array(z.string().min(1))
  }).strict()),
  metrics: z.array(DurableMetricV1Schema).min(1)
}).strict().superRefine((observation, ctx) => {
  const duration = Date.parse(observation.endedAt) - Date.parse(observation.startedAt);
  if (duration < 0 || Date.parse(observation.recordedAt) < Date.parse(observation.endedAt)) {
    ctx.addIssue({ code: "custom", message: "Observation chronology is invalid." });
  }
  const windows = new Map(observation.windows.map(window => [window.windowId, window]));
  if (windows.size !== observation.windows.length || new Set(observation.metrics.map(metric => metric.metricCode)).size !== observation.metrics.length) {
    ctx.addIssue({ code: "custom", message: "Window IDs and terminal metric codes must be unique." });
  }
  for (const window of observation.windows) {
    if (window.endMs <= window.startMs || window.endMs > duration || (window.status === "withheld" && window.reasonCodes.length === 0)) {
      ctx.addIssue({ code: "custom", message: "Evidence window falls outside the observation or lacks reasons." });
    }
  }
  for (const metric of observation.metrics) {
    if (metric.usableDurationMs > duration || new Set(metric.sourceWindowIds).size !== metric.sourceWindowIds.length || metric.sourceWindowIds.some(id => !windows.has(id) || (metric.status === "measured" && windows.get(id)?.status !== "eligible"))) {
      ctx.addIssue({ code: "custom", message: "Metric duration or source-window attribution is invalid." });
    }
    const eligibleIntervals = metric.sourceWindowIds.flatMap(id => {
      const window = windows.get(id);
      return window?.status === "eligible" ? [[window.startMs, window.endMs] as const] : [];
    }).sort((a, b) => a[0] - b[0]);
    const merged: [number, number][] = [];
    for (const [start, end] of eligibleIntervals) {
      const last = merged.at(-1);
      if (last && start <= last[1]) last[1] = Math.max(last[1], end);
      else merged.push([start, end]);
    }
    if (metric.status === "measured" && metric.usableDurationMs > merged.reduce((sum, [start, end]) => sum + end - start, 0)) ctx.addIssue({ code: "custom", message: "Usable duration cannot exceed the union of its eligible windows." });
    if (new Set(metric.temporalSamples.map(sample => sample.sampleId)).size !== metric.temporalSamples.length) ctx.addIssue({ code: "custom", message: "Temporal sample IDs must be unique." });
    for (const sample of metric.temporalSamples) {
      const supporting = sample.sourceWindowIds.flatMap(id => {
        const window = windows.get(id);
        return window?.status === "eligible" ? [[window.startMs, window.endMs] as const] : [];
      }).sort((a, b) => a[0] - b[0]);
      let supportedUntil = sample.startMs;
      for (const [start, end] of supporting) { if (start > supportedUntil) break; supportedUntil = Math.max(supportedUntil, end); }
      if (metric.status !== "measured" || sample.endMs <= sample.startMs || supportedUntil < sample.endMs || new Set(sample.sourceWindowIds).size !== sample.sourceWindowIds.length || sample.sourceWindowIds.some(id => !metric.sourceWindowIds.includes(id))) ctx.addIssue({ code: "custom", message: "Temporal sample requires eligible, attributable evidence within the encounter." });
    }
  }
  if (observation.revisionId === observation.supersedesRevisionId) ctx.addIssue({ code: "custom", message: "A revision cannot supersede itself." });
});
export type DurableObservationV1 = z.infer<typeof DurableObservationV1Schema>;

export const ClinicalEffectiveTimeV1Schema = z.object({
  earliest: Timestamp.nullable(), latest: Timestamp.nullable(),
  precision: z.enum(["instant", "day", "unknown"])
}).strict().superRefine((time, ctx) => {
  if (time.precision === "unknown") {
    if (time.earliest !== null || time.latest !== null) ctx.addIssue({ code: "custom", message: "Unknown times cannot invent bounds." });
  } else if (!time.earliest || !time.latest || Date.parse(time.latest) < Date.parse(time.earliest) || (time.precision === "instant" && Date.parse(time.latest) !== Date.parse(time.earliest)) || (time.precision === "day" && (Date.parse(time.latest) <= Date.parse(time.earliest) || Date.parse(time.latest) - Date.parse(time.earliest) > 26 * 60 * 60 * 1_000))) {
    ctx.addIssue({ code: "custom", message: "Known times require ordered bounds and exact instants require equal bounds." });
  }
});
export type ClinicalEffectiveTimeV1 = z.infer<typeof ClinicalEffectiveTimeV1Schema>;

const Dose = z.object({ value: z.number().finite().positive(), unit: z.string().min(1) }).strict();
export const TreatmentRevisionV1Schema = z.object({
  schemaVersion: z.literal("phenometric.treatment-revision.v1"),
  treatmentId: Id, revisionId: Id, supersedesRevisionId: Id.nullable(),
  scope: TreatmentResponseScopeV1Schema, courseId: Id, cycleId: Id.nullable(),
  recordedAt: Timestamp, effectiveTime: ClinicalEffectiveTimeV1Schema,
  status: z.enum(["administered", "planned", "entered-in-error"]),
  verification: z.enum(["verified", "unverified"]),
  kind: z.enum(["botulinum-injection", "concurrent-treatment"]),
  product: z.string().min(1), dose: Dose.nullable(), route: z.string().min(1).nullable(),
  sites: z.array(z.object({
    site: z.string().min(1), laterality: z.enum(["left", "right", "bilateral", "unspecified"]),
    dose: Dose.nullable()
  }).strict()),
  sourceRef: ClinicalSourceRefV1Schema
}).strict().refine(value => value.revisionId !== value.supersedesRevisionId, "A revision cannot supersede itself.");
export type TreatmentRevisionV1 = z.infer<typeof TreatmentRevisionV1Schema>;

export const ClinicalContextRevisionV1Schema = z.object({
  schemaVersion: z.literal("phenometric.clinical-context-revision.v1"),
  contextId: Id, revisionId: Id, supersedesRevisionId: Id.nullable(),
  scope: TreatmentResponseScopeV1Schema, recordedAt: Timestamp,
  effectiveTime: ClinicalEffectiveTimeV1Schema,
  status: z.enum(["active", "entered-in-error"]),
  kind: z.enum(["medication-change", "intercurrent-illness", "procedure", "patient-context"]),
  valueCode: z.string().min(1), sourceRef: ClinicalSourceRefV1Schema
}).strict().refine(value => value.revisionId !== value.supersedesRevisionId, "A revision cannot supersede itself.");
export type ClinicalContextRevisionV1 = z.infer<typeof ClinicalContextRevisionV1Schema>;

export const TreatmentResponseProtocolV1Schema = z.object({
  schemaVersion: z.literal("phenometric.treatment-response-protocol.v1"),
  protocolId: Id, version: z.string().min(1), contentSha256: Digest,
  status: z.literal("research-only"), condition: z.string().min(1),
  intendedUse: z.string().min(1), targetPopulation: z.string().min(1),
  measurementProtocolRef: VersionedArtifactRefV1Schema,
  permittedSourceKinds: z.array(DurableObservationV1Schema.shape.capture.shape.sourceKind).min(1),
  metrics: z.array(z.object({
    metricCode: z.string().min(1), label: z.string().min(1), unit: z.string().min(1),
    modality: z.enum(["face", "voice"]),
    context: z.string().min(1), algorithmVersion: z.string().min(1),
    minimumUsableDurationMs: NonNegative, minimumTechnicalQualityScore: Quality
  }).strict()).min(1),
  referenceStandard: z.string().min(1), humanWorkflow: z.string().min(1),
  clinicalValidation: z.literal("none"), validatedClaim: z.literal("none"),
  uncertainty: z.literal("analytical-repeatability-and-minimum-detectable-change-unknown"),
  prohibitedClaims: z.array(z.string().min(1)).min(1)
}).strict().refine(protocol => new Set(protocol.metrics.map(metric => metric.metricCode)).size === protocol.metrics.length, "Metric codes must be unique.");
export type TreatmentResponseProtocolV1 = z.infer<typeof TreatmentResponseProtocolV1Schema>;

export const TreatmentResponseSpecificationV1Schema = z.object({
  schemaVersion: z.literal("phenometric.treatment-response-specification.v1"),
  specificationId: Id, version: z.string().min(1), contentSha256: Digest,
  scope: TreatmentResponseScopeV1Schema, protocolRef: VersionedArtifactRefV1Schema,
  anchorTreatmentId: Id,
  baseline: z.object({
    startDay: z.number().finite().negative(), endDay: z.number().finite().max(0),
    minimumObservations: z.number().int().positive(), minimumEncounters: z.number().int().positive()
  }).strict(),
  followupEndDay: z.number().finite().positive(),
  phases: z.array(z.object({
    phaseId: Id, startDay: z.number().finite().nonnegative(), endDay: z.number().finite().positive(),
    minimumCount: z.number().int().positive()
  }).strict()),
  maximumClockUncertaintyMs: NonNegative,
  concurrentTreatmentPolicy: z.enum(["flag", "exclude-after-change"]),
  sourceCompatibility: z.literal("exact"),
  baselineAggregation: z.literal("median-of-encounter-medians"),
  fittedModels: z.literal(false), clinicalClaim: z.literal("descriptive-only")
}).strict().superRefine((spec, ctx) => {
  if (spec.baseline.endDay <= spec.baseline.startDay) ctx.addIssue({ code: "custom", message: "Baseline window must have positive duration." });
  if (new Set(spec.phases.map(phase => phase.phaseId)).size !== spec.phases.length) ctx.addIssue({ code: "custom", message: "Phase IDs must be unique." });
  const phases = [...spec.phases].sort((a, b) => a.startDay - b.startDay);
  phases.forEach((phase, index) => {
    if (phase.endDay <= phase.startDay || phase.endDay > spec.followupEndDay || (index > 0 && phase.startDay < phases[index - 1].endDay)) ctx.addIssue({ code: "custom", message: "Coverage windows must be ordered, disjoint and within follow-up." });
  });
});
export type TreatmentResponseSpecificationV1 = z.infer<typeof TreatmentResponseSpecificationV1Schema>;

export const TreatmentResponseSnapshotInputV1Schema = z.object({
  scope: TreatmentResponseScopeV1Schema, asOf: Timestamp,
  observations: z.array(DurableObservationV1Schema),
  consents: z.array(DerivedDataConsentV1Schema), bindings: z.array(ParticipantBindingV1Schema),
  treatments: z.array(TreatmentRevisionV1Schema), contexts: z.array(ClinicalContextRevisionV1Schema)
}).strict().superRefine((snapshot, ctx) => {
  const scope = canonicalTreatmentResponseJson(snapshot.scope);
  for (const name of ["observations", "consents", "bindings", "treatments", "contexts"] as const) {
    if (snapshot[name].some(record => canonicalTreatmentResponseJson(record.scope) !== scope)) ctx.addIssue({ code: "custom", path: [name], message: "Cross-scope history is forbidden." });
    const records: readonly { revisionId?: string; consentId?: string; bindingId?: string }[] = snapshot[name];
    const ids = records.map(record => record.revisionId ?? record.consentId ?? record.bindingId);
    if (new Set(ids).size !== ids.length) ctx.addIssue({ code: "custom", path: [name], message: "Record IDs must be unique within a snapshot." });
  }
});
export type TreatmentResponseSnapshotInputV1 = z.infer<typeof TreatmentResponseSnapshotInputV1Schema>;

export const TreatmentResponseSnapshotV1Schema = TreatmentResponseSnapshotInputV1Schema.safeExtend({
  schemaVersion: z.literal("phenometric.treatment-response-snapshot.v1"),
  snapshotId: Id, contentSha256: Digest
});
export type TreatmentResponseSnapshotV1 = z.infer<typeof TreatmentResponseSnapshotV1Schema>;

export const TreatmentResponseExclusionCodeV1Schema = z.enum([
  "superseded", "entered-in-error", "consent-missing", "consent-not-applicable", "binding-missing",
  "binding-not-verified", "binding-encounter-mismatch", "source-kind-not-permitted", "clock-uncertain",
  "measurement-protocol-mismatch", "metric-missing", "metric-definition-mismatch", "metric-withheld",
  "insufficient-usable-duration", "technical-quality-failed", "source-incompatible", "outside-analysis-window",
  "treatment-time-overlap", "concurrent-treatment-change", "subsequent-treatment-cycle", "non-finite-delta", "baseline-unavailable", "anchor-unavailable"
]);
export type TreatmentResponseExclusionCodeV1 = z.infer<typeof TreatmentResponseExclusionCodeV1Schema>;

export const TreatmentResponsePointV1Schema = z.object({
  observationId: Id, revisionId: Id, encounterId: Id, startedAt: Timestamp, endedAt: Timestamp,
  value: z.number().finite(), delta: z.number().finite().nullable(),
  modality: z.enum(["face", "voice"]), context: z.string().min(1), algorithmVersion: z.string().min(1),
  processorRef: z.string().min(1), usableDurationMs: NonNegative, technicalQualityScore: Quality,
  captureAdapterId: Id, captureAdapterVersion: z.string().min(1),
  sourceKind: DurableObservationV1Schema.shape.capture.shape.sourceKind, processorFingerprint: Digest,
  daysSinceTreatment: z.object({ minimum: z.number().finite(), maximum: z.number().finite() }).strict().nullable(),
  phaseIds: z.array(Id), sourceWindowIds: z.array(Id).min(1)
}).strict();
export type TreatmentResponsePointV1 = z.infer<typeof TreatmentResponsePointV1Schema>;

export const TreatmentResponseRowV1Schema = z.object({
  metricCode: z.string().min(1), label: z.string().min(1), unit: z.string().min(1),
  status: z.enum(["observed", "insufficient-baseline", "insufficient-followup", "no-comparable-data"]),
  baseline: z.object({
    status: z.enum(["available", "insufficient-data"]), value: z.number().finite().nullable(),
    observationRevisionIds: z.array(Id), encounterCount: z.number().int().nonnegative(),
    compatibilityFingerprint: Digest.nullable(), reasonCodes: z.array(z.string().min(1))
  }).strict(),
  points: z.array(TreatmentResponsePointV1Schema),
  exclusions: z.array(z.object({ observationId: Id, revisionId: Id, reasonCodes: z.array(TreatmentResponseExclusionCodeV1Schema).min(1) }).strict()),
  phaseCoverage: z.array(z.object({
    phaseId: Id, count: z.number().int().nonnegative(), minimumCount: z.number().int().positive(),
    status: z.enum(["sufficient", "sparse", "missing"])
  }).strict()),
  uncertainty: z.object({ measurementError: z.literal("unknown"), minimumDetectableChange: z.literal("unknown"), confidenceInterval: z.null() }).strict()
}).strict().superRefine((row, ctx) => {
  if ((row.baseline.status === "available") !== (row.baseline.value !== null)) ctx.addIssue({ code: "custom", message: "Baseline status and value must agree." });
  for (const point of row.points) {
    if ((point.daysSinceTreatment !== null && point.daysSinceTreatment.maximum < point.daysSinceTreatment.minimum) || (row.baseline.value === null ? point.delta !== null : point.delta !== point.value - row.baseline.value) || (point.daysSinceTreatment === null && (point.phaseIds.length > 0 || point.delta !== null))) ctx.addIssue({ code: "custom", message: "Point timing or baseline delta is inconsistent." });
  }
  for (const phase of row.phaseCoverage) {
    const expected = phase.count === 0 ? "missing" : phase.count < phase.minimumCount ? "sparse" : "sufficient";
    if (phase.status !== expected) ctx.addIssue({ code: "custom", message: "Phase status must reflect its observation count." });
  }
  const pointIds = new Set(row.points.map(point => point.revisionId));
  if (pointIds.size !== row.points.length || new Set(row.exclusions.map(exclusion => exclusion.revisionId)).size !== row.exclusions.length || row.exclusions.some(exclusion => pointIds.has(exclusion.revisionId)) || row.baseline.observationRevisionIds.some(id => !pointIds.has(id))) ctx.addIssue({ code: "custom", message: "Points, exclusions and baseline membership must have unique, consistent source revisions." });
});
export type TreatmentResponseRowV1 = z.infer<typeof TreatmentResponseRowV1Schema>;

export const TreatmentResponseRunV1Schema = z.object({
  schemaVersion: z.literal("phenometric.treatment-response-run.v1"),
  runId: Id, contentSha256: Digest, scope: TreatmentResponseScopeV1Schema,
  snapshotId: Id, snapshotSha256: Digest,
  protocolRef: VersionedArtifactRefV1Schema, specificationRef: VersionedArtifactRefV1Schema,
  engineVersion: z.string().min(1), generatedAt: Timestamp,
  status: z.enum(["available", "insufficient-data", "anchor-unavailable"]),
  anchor: z.object({
    treatmentId: Id, revisionId: Id.nullable(), cycleId: Id.nullable(),
    effectiveTime: ClinicalEffectiveTimeV1Schema.nullable(), reasonCodes: z.array(z.string().min(1))
  }).strict(),
  rows: z.array(TreatmentResponseRowV1Schema).min(1),
  contextFlags: z.array(z.object({ revisionId: Id, kind: z.string().min(1), reason: z.string().min(1) }).strict()),
  clinicalClaim: z.literal("descriptive-only"), fittedModels: z.literal(false)
}).strict();
export type TreatmentResponseRunV1 = z.infer<typeof TreatmentResponseRunV1Schema>;

export const TreatmentResponseReviewV1Schema = z.object({
  schemaVersion: z.literal("phenometric.treatment-response-review.v1"),
  reviewId: Id, scope: TreatmentResponseScopeV1Schema, runId: Id, runSha256: Digest,
  actorId: Id, recordedAt: Timestamp,
  disposition: z.enum(["acknowledged", "dismissed", "correction-requested"]),
  supersedesReviewId: Id.nullable(), reasonCode: z.string().min(1).nullable()
}).strict();
export type TreatmentResponseReviewV1 = z.infer<typeof TreatmentResponseReviewV1Schema>;

/** Canonical JSON for the JSON-only records in this subsystem; rejects lossy values. */
export function canonicalTreatmentResponseJson(value: unknown): string {
  if (value === null || typeof value === "string" || typeof value === "boolean") return JSON.stringify(value);
  if (typeof value === "number" && Number.isFinite(value)) return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalTreatmentResponseJson).join(",")}]`;
  if (value && typeof value === "object" && (Object.getPrototypeOf(value) === Object.prototype || Object.getPrototypeOf(value) === null)) {
    return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${canonicalTreatmentResponseJson((value as Record<string, unknown>)[key])}`).join(",")}}`;
  }
  throw new Error("treatment-response-non-json-value");
}

export async function treatmentResponseDigest(value: unknown): Promise<string> {
  return calculateSha256Hex(canonicalTreatmentResponseJson(value));
}

/** Content-address an artifact which has an independently assigned logical ID. */
export async function sealTreatmentResponseArtifact<T extends { contentSha256: string }>(value: T): Promise<T> {
  const { contentSha256: _digest, ...content } = value;
  return { ...value, contentSha256: await treatmentResponseDigest(content) };
}

export async function verifyTreatmentResponseArtifact<T extends { contentSha256: string }>(value: T): Promise<boolean> {
  return (await sealTreatmentResponseArtifact(value)).contentSha256 === value.contentSha256;
}
