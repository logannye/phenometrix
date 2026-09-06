import {
  AMBIENT_LOCAL_PROTOCOL_PACK, AMBIENT_LOCAL_PROTOCOL_REF,
  TreatmentResponseProtocolV1Schema, TreatmentResponseSpecificationV1Schema,
  sealTreatmentResponseArtifact,
  type TreatmentResponseProtocolV1, type TreatmentResponseSpecificationV1,
  type TreatmentResponseScopeV1
} from "@phenometrix/contracts";

export const HFS_TREATMENT_RESPONSE_PROTOCOL_ID = "hfs-ambient-treatment-alignment-research";
export const HFS_TREATMENT_RESPONSE_METRIC_CODES = [
  "ambient.face.eye_aperture.left", "ambient.face.eye_aperture.right",
  "ambient.face.lid_closure_completeness.left", "ambient.face.lid_closure_completeness.right"
] as const;

/** Existing engineering measurements only; no claim that these detect spasms or treatment benefit. */
export async function createHfsTreatmentResponseProtocol(): Promise<TreatmentResponseProtocolV1> {
  const content = TreatmentResponseProtocolV1Schema.parse({
    schemaVersion: "phenometric.treatment-response-protocol.v1",
    protocolId: HFS_TREATMENT_RESPONSE_PROTOCOL_ID, version: "1.0.0", contentSha256: "0".repeat(64),
    status: "research-only", condition: "established-hemifacial-spasm",
    intendedUse: "Describe technically qualified eye-aperture and eyelid-closure measurements across routine encounters relative to documented botulinum administration dates.",
    targetPopulation: "Consented adults with clinician-established hemifacial spasm receiving botulinum injections in a clinician research pilot.",
    measurementProtocolRef: { id: AMBIENT_LOCAL_PROTOCOL_REF.packId, version: AMBIENT_LOCAL_PROTOCOL_REF.version, contentSha256: AMBIENT_LOCAL_PROTOCOL_REF.contentSha256 },
    permittedSourceKinds: ["patient-local-pre-codec", "synthetic"],
    metrics: HFS_TREATMENT_RESPONSE_METRIC_CODES.map(metricCode => {
      const definition = AMBIENT_LOCAL_PROTOCOL_PACK.metrics.find(metric => metric.code === metricCode)!;
      return { metricCode, label: definition.label, unit: definition.unit, modality: definition.modality, context: definition.context,
        algorithmVersion: definition.algorithmVersion,
        // These are engineering observation sufficiency gates, not validated clinical thresholds.
        minimumUsableDurationMs: 30_000, minimumTechnicalQualityScore: 0.5 };
    }),
    referenceStandard: "Prospective clinician adjudication and separately consented research annotation are required; no clinical reference-standard validation is implemented.",
    humanWorkflow: "Nonblocking clinician inspection of source-linked measurements during the existing care workflow. No automatic treatment recommendation or execution.",
    clinicalValidation: "none", validatedClaim: "none",
    uncertainty: "analytical-repeatability-and-minimum-detectable-change-unknown",
    prohibitedClaims: ["spasm-diagnosis", "spasm-frequency", "clinical-severity", "treatment-efficacy", "causal-effect", "peak-response", "wearing-off", "dose-recommendation"]
  });
  return sealTreatmentResponseArtifact(content);
}

export async function createHfsTreatmentResponseSpecification(input: {
  scope: TreatmentResponseScopeV1;
  anchorTreatmentId: string;
  specificationId?: string;
}): Promise<TreatmentResponseSpecificationV1> {
  const protocol = await createHfsTreatmentResponseProtocol();
  return sealTreatmentResponseArtifact(TreatmentResponseSpecificationV1Schema.parse({
    schemaVersion: "phenometric.treatment-response-specification.v1",
    specificationId: input.specificationId ?? `hfs-spec-${input.anchorTreatmentId}`, version: "1.0.0", contentSha256: "0".repeat(64),
    scope: input.scope, protocolRef: { id: protocol.protocolId, version: protocol.version, contentSha256: protocol.contentSha256 },
    anchorTreatmentId: input.anchorTreatmentId,
    baseline: { startDay: -56, endDay: 0, minimumObservations: 2, minimumEncounters: 2 },
    followupEndDay: 112,
    phases: [
      { phaseId: "days-0-to-14", startDay: 0, endDay: 14, minimumCount: 1 },
      { phaseId: "days-14-to-42", startDay: 14, endDay: 42, minimumCount: 1 },
      { phaseId: "days-42-to-112", startDay: 42, endDay: 112, minimumCount: 1 }
    ],
    maximumClockUncertaintyMs: 1_000, concurrentTreatmentPolicy: "flag",
    sourceCompatibility: "exact", baselineAggregation: "median-of-encounter-medians",
    fittedModels: false, clinicalClaim: "descriptive-only"
  }));
}
