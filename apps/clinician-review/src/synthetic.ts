/** Explicit, local-only preview fixtures. Never imported by the production mount API. */
import { sealTreatmentResponseArtifact, type TreatmentResponseRunV1, type TreatmentResponseRowV1 } from "@phenometrix/contracts";
export const SYNTHETIC_SCOPE = { tenantId: "synthetic-clinic", studyId: "synthetic-study", participantId: "synthetic-participant" };
const reference = { id: "synthetic-fixture", version: "1.0.0", contentSha256: "a".repeat(64) };
const anchor = "2026-06-15T12:00:00.000Z";

export async function syntheticRun(labelOverride?: string): Promise<TreatmentResponseRunV1> {
  const rows: TreatmentResponseRowV1[] = [
    { metricCode: "ambient.face.eye_aperture.left", label: labelOverride ?? "Left eye aperture", unit: "ratio", values: [.22, .24, .28, .27, .25] },
    { metricCode: "ambient.face.lid_closure_completeness.left", label: "Left spontaneous closure completeness", unit: "ratio", values: [.92, .94, .89, .93, .92] }
  ].map(metric => {
    const baseline = (metric.values[0] + metric.values[1]) / 2;
    return {
      metricCode: metric.metricCode, label: metric.label, unit: metric.unit, status: "observed",
      baseline: { status: "available", value: baseline, observationRevisionIds: ["synthetic-r0", "synthetic-r1"], encounterCount: 2, compatibilityFingerprint: "b".repeat(64), reasonCodes: [] },
      points: [-7, -2, 7, 28, 70].map((day, i) => {
        const startedAt = new Date(Date.parse(anchor) + day * 86400000).toISOString();
        return { observationId: `synthetic-o${i}`, revisionId: `synthetic-r${i}`, encounterId: `synthetic-visit${i}`,
          startedAt, endedAt: new Date(Date.parse(startedAt) + 60000).toISOString(), value: metric.values[i], delta: metric.values[i] - baseline,
          modality: "face", context: "ambient-frontal", algorithmVersion: "1.1.0", processorRef: "synthetic-processor",
          usableDurationMs: 55000, technicalQualityScore: .9, captureAdapterId: "synthetic-adapter", captureAdapterVersion: "1.0.0",
          sourceKind: "synthetic", processorFingerprint: "b".repeat(64),
          daysSinceTreatment: { minimum: day, maximum: day }, phaseIds: day < 0 ? [] : [day < 15 ? "Days 0–14" : day < 45 ? "Days 15–44" : "Days 45–90"].map(x => x.replaceAll(" ", "-").replaceAll("–", "-")), sourceWindowIds: [`synthetic-window${i}`] };
      }),
      exclusions: [{ observationId: "synthetic-missing", revisionId: "synthetic-withheld", reasonCodes: ["metric-withheld"] }],
      phaseCoverage: [{ phaseId: "Days-0-14", count: 1, minimumCount: 2, status: "sparse" }, { phaseId: "Days-15-44", count: 1, minimumCount: 2, status: "sparse" }, { phaseId: "Days-45-90", count: 1, minimumCount: 2, status: "sparse" }],
      uncertainty: { measurementError: "unknown", minimumDetectableChange: "unknown", confidenceInterval: null }
    };
  });
  return sealTreatmentResponseArtifact({
    schemaVersion: "phenometric.treatment-response-run.v1", runId: "synthetic-run", contentSha256: "0".repeat(64), scope: SYNTHETIC_SCOPE,
    snapshotId: "synthetic-snapshot", snapshotSha256: "c".repeat(64), protocolRef: reference, specificationRef: reference,
    engineVersion: "synthetic-preview", generatedAt: "2026-08-24T12:05:00.000Z", status: "available",
    anchor: { treatmentId: "synthetic-injection", revisionId: "synthetic-injection-r1", cycleId: "synthetic-cycle", effectiveTime: { earliest: anchor, latest: anchor, precision: "instant" }, reasonCodes: [] },
    rows, contextFlags: [], clinicalClaim: "descriptive-only", fittedModels: false
  });
}

export async function syntheticUnavailableAnchorRun(): Promise<TreatmentResponseRunV1> {
  const run = await syntheticRun();
  run.runId = "synthetic-unverified-run"; run.status = "anchor-unavailable";
  run.anchor.reasonCodes = ["treatment-not-verified"];
  for (const row of run.rows) {
    row.status = "insufficient-baseline";
    row.baseline = { status: "insufficient-data", value: null, observationRevisionIds: [], encounterCount: 0, compatibilityFingerprint: null, reasonCodes: ["anchor-unavailable"] };
    for (const point of row.points) { point.delta = null; point.daysSinceTreatment = null; point.phaseIds = []; }
    for (const phase of row.phaseCoverage) { phase.count = 0; phase.status = "missing"; }
  }
  return sealTreatmentResponseArtifact(run);
}
