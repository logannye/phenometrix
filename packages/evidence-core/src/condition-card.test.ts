import { describe, expect, it } from "vitest";
import {
  AMBIENT_LOCAL_PROTOCOL_PACK,
  AMBIENT_LOCAL_PROTOCOL_REF,
  CONDITION_DEMO_BOUNDARY_STATEMENT,
  CONDITION_DEMO_METRIC_CODES,
  CONDITION_DEMO_PROHIBITED_CLAIMS,
  CONDITION_EVIDENCE_SOURCE_DISCLOSURE,
  PREVIOUS_VISIT_CLAIM_BOUNDARY,
  PreviousVisitComparisonV1Schema,
  type ConditionDemoMetricCode,
  type ExcludedPreviousVisitMetricSourceV1,
  type MeasuredMetricOutcomeV1,
  type MeasuredPreviousVisitMetricSourceV1,
  type PreviousVisitComparisonRowV1,
  type PreviousVisitComparisonV1,
  type ProcessorProvenanceV1,
  type WithheldMetricOutcomeV1
} from "@phenometrix/contracts";
import {
  UNILATERAL_FACIAL_MOVEMENT_PROFILE
} from "@phenometrix/condition-profiles";
import { buildConditionEvidenceCard } from "./condition-card.js";

const PROCESSOR_DIGEST = "b".repeat(64);
const REFERENCE_OBSERVATION_ID = "observation-reference";
const REFERENCE_SESSION_ID = "session-reference";
const CURRENT_OBSERVATION_ID = "observation-current";
const CURRENT_SESSION_ID = "session-current";
const SUBJECT_REF = "subject-demo";

const METRIC_DETAILS: Readonly<
  Record<
    ConditionDemoMetricCode,
    {
      displayLabel: string;
      displayRole: "primary" | "experimental";
      laterality:
        | "subject-left-minus-subject-right"
        | "explicit-subject-left"
        | "explicit-subject-right";
      unit: string;
      reportSection: "symmetry" | "blink-behavior" | "expression-dynamics";
    }
  >
> = {
  "ambient.face.rest_mouth_corner_asymmetry.signed": {
    displayLabel:
      "Resting mouth difference (subject-left minus subject-right)",
    displayRole: "primary",
    laterality: "subject-left-minus-subject-right",
    unit: "inter-eye-normalized-distance",
    reportSection: "symmetry"
  },
  "ambient.face.rest_eye_aperture_asymmetry.signed": {
    displayLabel:
      "Resting eye-aperture difference (subject-left minus subject-right)",
    displayRole: "primary",
    laterality: "subject-left-minus-subject-right",
    unit: "eye-width-ratio",
    reportSection: "symmetry"
  },
  "ambient.face.lid_closure_completeness.left": {
    displayLabel: "Subject-left lid closure completeness",
    displayRole: "primary",
    laterality: "explicit-subject-left",
    unit: "closure-ratio",
    reportSection: "blink-behavior"
  },
  "ambient.face.lid_closure_completeness.right": {
    displayLabel: "Subject-right lid closure completeness",
    displayRole: "primary",
    laterality: "explicit-subject-right",
    unit: "closure-ratio",
    reportSection: "blink-behavior"
  },
  "ambient.face.spontaneous_excursion_asymmetry.median": {
    displayLabel:
      "Spontaneous excursion difference (subject-left minus subject-right)",
    displayRole: "experimental",
    laterality: "subject-left-minus-subject-right",
    unit: "signed-excursion-ratio",
    reportSection: "expression-dynamics"
  },
  "ambient.face.oculo_oral_synkinesis_index": {
    displayLabel: "Oculo-oral coupling difference",
    displayRole: "experimental",
    laterality: "subject-left-minus-subject-right",
    unit: "signed-coupling-ratio",
    reportSection: "expression-dynamics"
  }
};

const PROFILE = UNILATERAL_FACIAL_MOVEMENT_PROFILE;

const PROCESSOR: ProcessorProvenanceV1 = {
  modality: "face",
  processorRef: "facial-analysis-1.1.0",
  runtime: "mediapipe-tasks-vision",
  runtimeVersion: "0.10.35",
  assetPath: "/models/face_landmarker.task",
  assetSha256: PROCESSOR_DIGEST,
  assetIntegrityVerified: true
};

function outcomeBase(
  metricCode: ConditionDemoMetricCode,
  observationId: string,
  sessionId: string
) {
  const details = METRIC_DETAILS[metricCode];
  const definition = AMBIENT_LOCAL_PROTOCOL_PACK.metrics.find(
    (candidate) => candidate.code === metricCode
  );
  if (!definition) throw new Error(`Missing protocol metric ${metricCode}.`);
  const suffix = metricCode.replaceAll(".", "-");
  const aggregateId = `aggregate-${suffix}`;
  return {
    outcomeId: `outcome-${suffix}`,
    aggregateId,
    metricCode,
    label: definition.label,
    modality: "face" as const,
    context: "ambient-frontal" as const,
    unit: details.unit,
    reportSection: details.reportSection,
    algorithmVersion: "1.1.0",
    processorRef: PROCESSOR.processorRef,
    trackSegmentId: "face-track-1",
    technicalVerification: "automated-test" as const,
    clinicalValidation: "none" as const,
    evidence: {
      eligibleDurationMs: 30_000,
      activeDurationMs: 0,
      segmentCount: 1,
      windowCount: 1,
      binCount: 6,
      eventCount: 3,
      sampleCount: 900,
      coverage: null,
      qualityFacts: { observationSpanMs: 30_000 },
      refs: [
        {
          schemaVersion: "phenometric.evidence-ref.v1" as const,
          kind: "aggregate" as const,
          sessionId,
          observationId,
          protocolRef: AMBIENT_LOCAL_PROTOCOL_REF,
          aggregateId,
          metricCode,
          modality: "face" as const,
          context: "ambient-frontal" as const,
          unit: details.unit,
          trackSegmentId: "face-track-1"
        }
      ]
    }
  };
}

function measuredOutcome(
  metricCode: ConditionDemoMetricCode,
  observationId: string,
  sessionId: string,
  value: number,
  technicalQualityScore: number
): MeasuredMetricOutcomeV1 {
  return {
    ...outcomeBase(metricCode, observationId, sessionId),
    status: "measured",
    value,
    technicalQualityScore,
    technicalDispersion: 0.01
  };
}

function withheldOutcome(
  metricCode: ConditionDemoMetricCode,
  observationId: string,
  sessionId: string
): WithheldMetricOutcomeV1 {
  return {
    ...outcomeBase(metricCode, observationId, sessionId),
    status: "withheld",
    reasonCode: "insufficient-events",
    detail: "Not enough technically qualified spontaneous events.",
    technicalQualityScore: null,
    technicalDispersion: null
  };
}

function source(
  outcome: MeasuredMetricOutcomeV1,
  observationId: string,
  sessionId: string
): MeasuredPreviousVisitMetricSourceV1 {
  return {
    observationId,
    sessionId,
    status: "measured",
    outcome,
    processor: PROCESSOR
  };
}

function excludedSource(
  outcome: MeasuredMetricOutcomeV1 | WithheldMetricOutcomeV1,
  observationId: string,
  sessionId: string
): ExcludedPreviousVisitMetricSourceV1 {
  const trace = {
    observationId,
    sessionId,
    outcomeId: outcome.outcomeId,
    aggregateId: outcome.aggregateId,
    metricCode: outcome.metricCode as ConditionDemoMetricCode,
    modality: "face" as const,
    context: "ambient-frontal" as const,
    unit: outcome.unit,
    algorithmVersion: outcome.algorithmVersion,
    processorRef: outcome.processorRef,
    trackSegmentId: outcome.trackSegmentId,
    evidence: outcome.evidence,
    processor: PROCESSOR
  };
  return outcome.status === "measured"
    ? {
        ...trace,
        status: "measured",
        technicalQualityScore: outcome.technicalQualityScore,
        withheldReasonCode: null
      }
    : {
        ...trace,
        status: "withheld",
        technicalQualityScore: outcome.technicalQualityScore,
        withheldReasonCode: outcome.reasonCode
      };
}

function measuredRow(
  metricCode: ConditionDemoMetricCode,
  index: number
): PreviousVisitComparisonRowV1 {
  const referenceValue = 0.1 + index * 0.01;
  const currentValue = 0.16 + index * 0.01;
  const referenceOutcome = measuredOutcome(
    metricCode,
    REFERENCE_OBSERVATION_ID,
    REFERENCE_SESSION_ID,
    referenceValue,
    0.96 - index * 0.03
  );
  const currentOutcome = measuredOutcome(
    metricCode,
    CURRENT_OBSERVATION_ID,
    CURRENT_SESSION_ID,
    currentValue,
    0.94 - index * 0.04
  );
  return {
    metricCode,
    status: "measured",
    decision: "included",
    compatibilityReasonCodes: [],
    nativeUnit: METRIC_DETAILS[metricCode].unit,
    reference: source(
      referenceOutcome,
      REFERENCE_OBSERVATION_ID,
      REFERENCE_SESSION_ID
    ),
    current: source(
      currentOutcome,
      CURRENT_OBSERVATION_ID,
      CURRENT_SESSION_ID
    ),
    delta: currentValue - referenceValue
  };
}

function buildComparison(
  rowOverride?: (
    row: PreviousVisitComparisonRowV1,
    metricCode: ConditionDemoMetricCode,
    index: number
  ) => PreviousVisitComparisonRowV1
): PreviousVisitComparisonV1 {
  const rows = CONDITION_DEMO_METRIC_CODES.map((metricCode, index) => {
    const row = measuredRow(metricCode, index);
    return rowOverride ? rowOverride(row, metricCode, index) : row;
  });
  return PreviousVisitComparisonV1Schema.parse({
    schemaVersion: "phenometric.previous-visit-comparison.v1",
    comparisonId: "comparison-reference-current",
    generatedAt: "2026-08-21T16:01:00.000Z",
    demoContext: {
      schemaVersion: "phenometric.condition-demo-context.v1",
      profileRef: {
        profileId: PROFILE.profileId,
        version: PROFILE.version,
        contentSha256: PROFILE.contentSha256
      },
      subjectRef: SUBJECT_REF,
      assertedAffectedSide: {
        side: "left",
        source: "participant-asserted",
        verified: false
      },
      sourceProtocolRef: AMBIENT_LOCAL_PROTOCOL_REF,
      allowlistedMetricCodes: CONDITION_DEMO_METRIC_CODES,
      boundaryStatement: CONDITION_DEMO_BOUNDARY_STATEMENT,
      prohibitedClaims: CONDITION_DEMO_PROHIBITED_CLAIMS,
      persistence: "page-memory-only"
    },
    acceptedReferenceId: "accepted-reference-1",
    referenceObservation: {
      observationId: REFERENCE_OBSERVATION_ID,
      sessionId: REFERENCE_SESSION_ID,
      startedAt: "2026-08-20T16:00:00.000Z",
      endedAt: "2026-08-20T16:01:00.000Z",
      protocolRef: AMBIENT_LOCAL_PROTOCOL_REF,
      captureAdapter: { id: "browser-local-media", version: "1.1.0" }
    },
    currentObservation: {
      observationId: CURRENT_OBSERVATION_ID,
      sessionId: CURRENT_SESSION_ID,
      startedAt: "2026-08-21T16:00:00.000Z",
      endedAt: "2026-08-21T16:01:00.000Z",
      protocolRef: AMBIENT_LOCAL_PROTOCOL_REF,
      captureAdapter: { id: "browser-local-media", version: "1.1.0" }
    },
    rows,
    analyticalRepeatability: "unknown",
    minimumDetectableChange: "unknown",
    claimBoundary: PREVIOUS_VISIT_CLAIM_BOUNDARY,
    persistence: "page-memory-only"
  });
}

function minimumQuality(
  rows: readonly PreviousVisitComparisonRowV1[],
  sourceName: "reference" | "current"
): number | null {
  const values = rows.flatMap((row) => {
    const metricSource = row[sourceName];
    if (metricSource?.status !== "measured") return [];
    return [
      "outcome" in metricSource
        ? metricSource.outcome.technicalQualityScore
        : metricSource.technicalQualityScore
    ];
  });
  return values.length === 0 ? null : Math.min(...values);
}

describe("buildConditionEvidenceCard", () => {
  it("builds the fixed six-row card and preserves every source outcome", () => {
    const comparison = buildComparison();
    const first = buildConditionEvidenceCard({
      comparison,
      generatedAt: "2026-08-21T16:01:03.000Z"
    });
    const second = buildConditionEvidenceCard({
      comparison,
      generatedAt: "2026-08-21T16:01:03.000Z"
    });

    expect(first).toEqual(second);
    expect(first.cardId).toBe(`card-${comparison.comparisonId}`);
    expect(first.rows).toHaveLength(6);
    first.rows.forEach((row, index) => {
      expect(row.metric.metricCode).toBe(CONDITION_DEMO_METRIC_CODES[index]);
      expect(row.comparison).toEqual(comparison.rows[index]);
      if (row.comparison.status === "measured") {
        expect(row.comparison.delta).toBe(
          row.comparison.current.outcome.value -
            row.comparison.reference.outcome.value
        );
      }
    });
    expect(first.rows[5].metric.displayLabel).toBe(
      "Oculo-oral coupling difference"
    );
    expect(first.rows[5].metric.displayLabel.toLowerCase()).not.toContain(
      "synkinesis"
    );
    expect(first.qualitySummary).toEqual({
      measuredComparisonCount: 6,
      withheldComparisonCount: 0,
      incompatibleComparisonCount: 0,
      referenceMinimumTechnicalQualityScore: minimumQuality(
        comparison.rows,
        "reference"
      ),
      currentMinimumTechnicalQualityScore: minimumQuality(
        comparison.rows,
        "current"
      )
    });
    expect(first.review).toEqual({
      status: "pending",
      action: null,
      actor: null,
      recordedAt: null
    });
    expect(first.boundaryStatement).toBe(CONDITION_DEMO_BOUNDARY_STATEMENT);
    expect(first.sourceDisclosure).toBe(
      CONDITION_EVIDENCE_SOURCE_DISCLOSURE
    );
    expect(first).not.toHaveProperty("headline");
    expect(first).not.toHaveProperty("summary");
    expect(first).not.toHaveProperty("narrative");
    expect(first).not.toHaveProperty("interpretation");
  });

  it("keeps withheld and incompatible rows excluded with no delta", () => {
    const comparison = buildComparison((row, metricCode, index) => {
      if (index === 1 && row.status === "measured") {
        return {
          metricCode,
          status: "withheld",
          decision: "excluded",
          compatibilityReasonCodes: ["current-withheld"],
          nativeUnit: METRIC_DETAILS[metricCode].unit,
          reference: excludedSource(
            row.reference.outcome,
            REFERENCE_OBSERVATION_ID,
            REFERENCE_SESSION_ID
          ),
          current: excludedSource(
            withheldOutcome(
              metricCode,
              CURRENT_OBSERVATION_ID,
              CURRENT_SESSION_ID
            ),
            CURRENT_OBSERVATION_ID,
            CURRENT_SESSION_ID
          ),
          delta: null
        };
      }
      if (index === 4) {
        return {
          metricCode,
          status: "incompatible",
          decision: "excluded",
          compatibilityReasonCodes: ["context-mismatch"],
          nativeUnit: null,
          reference: null,
          current: null,
          delta: null
        };
      }
      return row;
    });

    const card = buildConditionEvidenceCard({
      comparison,
      generatedAt: "2026-08-21T16:01:03.000Z"
    });

    expect(card.rows[1].comparison).toMatchObject({
      status: "withheld",
      decision: "excluded",
      delta: null
    });
    const withheld = card.rows[1].comparison;
    if (withheld.status !== "withheld") {
      throw new Error("Expected withheld comparison.");
    }
    expect(withheld.reference).not.toHaveProperty("outcome");
    expect(withheld.current).not.toHaveProperty("outcome");
    expect(withheld.reference).not.toHaveProperty("value");
    expect(withheld.current).not.toHaveProperty("value");
    expect(card.rows[4].comparison).toMatchObject({
      status: "incompatible",
      decision: "excluded",
      delta: null
    });
    expect(card.qualitySummary).toEqual({
      measuredComparisonCount: 4,
      withheldComparisonCount: 1,
      incompatibleComparisonCount: 1,
      referenceMinimumTechnicalQualityScore: minimumQuality(
        comparison.rows,
        "reference"
      ),
      currentMinimumTechnicalQualityScore: minimumQuality(
        comparison.rows,
        "current"
      )
    });
  });

  it("rejects a value that no longer agrees with its source-derived delta", () => {
    const comparison = structuredClone(buildComparison());
    const row = comparison.rows[0];
    if (row.status !== "measured") throw new Error("Expected measured row.");
    row.current.outcome.value += 1;

    expect(() =>
      buildConditionEvidenceCard({
        comparison,
        generatedAt: "2026-08-21T16:01:03.000Z"
      })
    ).toThrow();
  });

  it("rejects a profile context or generation time outside the comparison", () => {
    const comparison = buildComparison();
    const mismatchedComparison = structuredClone(comparison);
    mismatchedComparison.demoContext.profileRef.version = "1.0.1";

    expect(() =>
      buildConditionEvidenceCard({
        comparison: mismatchedComparison,
        generatedAt: "2026-08-21T16:01:03.000Z"
      })
    ).toThrow(/profile does not match/);
    expect(() =>
      buildConditionEvidenceCard({
        comparison,
        generatedAt: "2026-08-21T16:00:01.000Z"
      })
    ).toThrow(/cannot be generated before/);
  });

  it("accepts only a contract-valid fixed review disposition", () => {
    const comparison = buildComparison();
    const card = buildConditionEvidenceCard({
      comparison,
      generatedAt: "2026-08-21T16:01:03.000Z",
      review: {
        status: "accepted",
        action: "accept-card",
        actor: "local-demo-user",
        recordedAt: "2026-08-21T16:01:04.000Z"
      }
    });

    expect(card.review.status).toBe("accepted");
  });
});
