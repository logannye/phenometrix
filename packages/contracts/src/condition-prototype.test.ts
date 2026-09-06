import { describe, expect, it } from "vitest";
import {
  AMBIENT_LOCAL_PROTOCOL_PACK,
  AMBIENT_LOCAL_PROTOCOL_REF,
  CONDITION_DEMO_BOUNDARY_STATEMENT,
  CONDITION_DEMO_METRIC_CODES,
  CONDITION_DEMO_PROHIBITED_CLAIMS,
  CONDITION_EVIDENCE_SOURCE_DISCLOSURE,
  PREVIOUS_VISIT_CLAIM_BOUNDARY,
  AcceptedReferenceV1Schema,
  ConditionDemoContextV1Schema,
  ConditionDemoProfileV1Schema,
  ConditionEvidenceCardV1Schema,
  WithheldPreviousVisitComparisonRowV1Schema,
  ObservationV3Schema,
  PreviousVisitComparisonV1Schema,
  createAggregateId,
  type ConditionDemoContextV1,
  type MetricOutcomeV1,
  type ObservationV3,
  type PreviousVisitComparisonV1,
  type ProcessorProvenanceV1
} from "./index.js";

const PROFILE_DIGEST = "a".repeat(64);
const PROCESSOR_DIGEST = "b".repeat(64);

const profileMetricDefinitions = [
  {
    metricCode: "ambient.face.rest_mouth_corner_asymmetry.signed",
    displayLabel:
      "Resting mouth difference (subject-left minus subject-right)",
    displayRole: "primary",
    laterality: "subject-left-minus-subject-right"
  },
  {
    metricCode: "ambient.face.rest_eye_aperture_asymmetry.signed",
    displayLabel:
      "Resting eye-aperture difference (subject-left minus subject-right)",
    displayRole: "primary",
    laterality: "subject-left-minus-subject-right"
  },
  {
    metricCode: "ambient.face.lid_closure_completeness.left",
    displayLabel: "Subject-left lid closure completeness",
    displayRole: "primary",
    laterality: "explicit-subject-left"
  },
  {
    metricCode: "ambient.face.lid_closure_completeness.right",
    displayLabel: "Subject-right lid closure completeness",
    displayRole: "primary",
    laterality: "explicit-subject-right"
  },
  {
    metricCode: "ambient.face.spontaneous_excursion_asymmetry.median",
    displayLabel:
      "Spontaneous excursion difference (subject-left minus subject-right)",
    displayRole: "experimental",
    laterality: "subject-left-minus-subject-right"
  },
  {
    metricCode: "ambient.face.oculo_oral_synkinesis_index",
    displayLabel: "Oculo-oral coupling difference",
    displayRole: "experimental",
    laterality: "subject-left-minus-subject-right"
  }
] as const;

function demoContext(): ConditionDemoContextV1 {
  return ConditionDemoContextV1Schema.parse({
    schemaVersion: "phenometric.condition-demo-context.v1",
    profileRef: {
      profileId: "unilateral-facial-movement-research-demo",
      version: "1.0.0",
      contentSha256: PROFILE_DIGEST
    },
    subjectRef: "demo-subject-1",
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
  });
}

const processor: ProcessorProvenanceV1 = {
  modality: "face",
  processorRef: "facial-analysis@1.1.0",
  runtime: "mediapipe-face-landmarker",
  runtimeVersion: "0.10.22",
  assetPath: "/assets/face_landmarker.task",
  assetSha256: PROCESSOR_DIGEST,
  assetIntegrityVerified: true
};

function observation(
  sessionId: string,
  observationId: string,
  startedAt: string,
  endedAt: string,
  valueOffset = 0
): ObservationV3 {
  const trackSegmentId = `${sessionId}:face-track`;
  const windowId = `${sessionId}:face-window`;
  const outcomes: MetricOutcomeV1[] = CONDITION_DEMO_METRIC_CODES.map(
    (metricCode, index) => {
      const definition = AMBIENT_LOCAL_PROTOCOL_PACK.metrics.find(
        (candidate) => candidate.code === metricCode
      );
      if (!definition) throw new Error(`Missing ${metricCode}.`);
      const aggregateId = createAggregateId({
        protocolPackId: AMBIENT_LOCAL_PROTOCOL_REF.packId,
        protocolVersion: AMBIENT_LOCAL_PROTOCOL_REF.version,
        sessionId,
        metricCode,
        context: definition.context,
        unit: definition.unit,
        algorithmVersion: definition.algorithmVersion,
        processorRef: processor.processorRef,
        trackSegmentId
      });
      return {
        outcomeId: `${sessionId}:outcome:${index}`,
        aggregateId,
        metricCode,
        label: definition.label,
        modality: definition.modality,
        context: definition.context,
        unit: definition.unit,
        reportSection: definition.reportSection,
        algorithmVersion: definition.algorithmVersion,
        processorRef: processor.processorRef,
        trackSegmentId,
        technicalVerification: "automated-test",
        clinicalValidation: "none",
        status: "measured",
        value: index + 1 + valueOffset,
        technicalQualityScore: valueOffset === 0 ? 0.9 : 0.8,
        technicalDispersion: 0.01,
        evidence: {
          eligibleDurationMs: 60_000,
          activeDurationMs: 60_000,
          segmentCount: 1,
          windowCount: 1,
          binCount: 12,
          eventCount: 3,
          sampleCount: 1_440,
          coverage: 0.95,
          qualityFacts: { usableBins: 12, expressionEventCount: 3 },
          refs: [
            {
              schemaVersion: "phenometric.evidence-ref.v1",
              kind: "window",
              sessionId,
              observationId,
              protocolRef: AMBIENT_LOCAL_PROTOCOL_REF,
              windowId,
              modality: "face",
              context: "ambient-frontal",
              trackSegmentId
            }
          ]
        }
      };
    }
  );

  return ObservationV3Schema.parse({
    schemaVersion: "phenometric.encounter-observation.v3",
    containsPHI: false,
    retention: {
      rawMedia: false,
      rawAudio: false,
      rawVideo: false,
      transcript: false,
      embeddings: false,
      persisted: false
    },
    observationId,
    sessionId,
    subjectRef: "demo-subject-1",
    protocolRef: AMBIENT_LOCAL_PROTOCOL_REF,
    consent: {
      schemaVersion: "phenometric.consent-record.v1",
      consentId: `${sessionId}:consent`,
      sessionId,
      documentVersion: "ambient-local-consent.v1",
      documentSha256:
        AMBIENT_LOCAL_PROTOCOL_PACK.consentDocument.contentSha256,
      recordedAt: startedAt,
      scopes: {
        cameraCapture: true,
        microphoneCapture: true,
        localInMemoryAnalysis: true
      },
      localParticipantAssertion: true,
      withdrawnAt: null
    },
    source: {
      role: "local-participant",
      sourceSessionRef: sessionId,
      audioAttribution: "user-asserted-local-participant",
      speakerAttribution: "unverified-local-input",
      audioInput: "microphone",
      faceAttribution: "single-visible-face",
      identityVerified: false
    },
    startedAt,
    endedAt,
    durationMs: 60_000,
    captureAdapter: { id: "browser-capture", version: "1.0.0" },
    processors: [processor],
    windows: [
      {
        windowId,
        sessionId,
        modality: "face",
        context: "ambient-frontal",
        trackSegmentId,
        processorRef: processor.processorRef,
        startMs: 0,
        endMs: 60_000,
        technicalQualityScore: 0.9,
        status: "eligible",
        reasonCodes: []
      }
    ],
    measurements: [],
    metricOutcomes: outcomes,
    qualitySummary: {
      voice: {
        state: "unavailable",
        eligibleDurationMs: 0,
        technicalQualityScore: null,
        reasonCodes: ["modality-unavailable"]
      },
      face: {
        state: "ready",
        eligibleDurationMs: 60_000,
        technicalQualityScore: 0.9,
        reasonCodes: []
      },
      totalWindowCount: 1,
      eligibleWindowCount: 1,
      withheldWindowCount: 0
    }
  });
}

function comparison(): PreviousVisitComparisonV1 {
  const reference = observation(
    "session-reference",
    "observation-reference",
    "2026-08-21T16:00:00.000Z",
    "2026-08-21T16:01:00.000Z"
  );
  const current = observation(
    "session-current",
    "observation-current",
    "2026-08-21T16:05:00.000Z",
    "2026-08-21T16:06:00.000Z",
    1
  );
  return PreviousVisitComparisonV1Schema.parse({
    schemaVersion: "phenometric.previous-visit-comparison.v1",
    comparisonId: "previous:observation-reference:observation-current",
    generatedAt: current.endedAt,
    demoContext: demoContext(),
    acceptedReferenceId: "reference-1",
    referenceObservation: {
      observationId: reference.observationId,
      sessionId: reference.sessionId,
      startedAt: reference.startedAt,
      endedAt: reference.endedAt,
      protocolRef: reference.protocolRef,
      captureAdapter: reference.captureAdapter
    },
    currentObservation: {
      observationId: current.observationId,
      sessionId: current.sessionId,
      startedAt: current.startedAt,
      endedAt: current.endedAt,
      protocolRef: current.protocolRef,
      captureAdapter: current.captureAdapter
    },
    rows: CONDITION_DEMO_METRIC_CODES.map((metricCode) => {
      const referenceOutcome = reference.metricOutcomes.find(
        (outcome) => outcome.metricCode === metricCode
      );
      const currentOutcome = current.metricOutcomes.find(
        (outcome) => outcome.metricCode === metricCode
      );
      if (
        referenceOutcome?.status !== "measured" ||
        currentOutcome?.status !== "measured"
      ) {
        throw new Error(`Missing measured outcome ${metricCode}.`);
      }
      return {
        metricCode,
        nativeUnit: referenceOutcome.unit,
        status: "measured",
        decision: "included",
        compatibilityReasonCodes: [],
        reference: {
          observationId: reference.observationId,
          sessionId: reference.sessionId,
          status: "measured",
          outcome: referenceOutcome,
          processor
        },
        current: {
          observationId: current.observationId,
          sessionId: current.sessionId,
          status: "measured",
          outcome: currentOutcome,
          processor
        },
        delta: currentOutcome.value - referenceOutcome.value
      };
    }),
    analyticalRepeatability: "unknown",
    minimumDetectableChange: "unknown",
    claimBoundary: PREVIOUS_VISIT_CLAIM_BOUNDARY,
    persistence: "page-memory-only"
  });
}

describe("condition demo contracts", () => {
  it("accepts the strict condition profile, context, and accepted reference", () => {
    expect(
      ConditionDemoProfileV1Schema.safeParse({
        schemaVersion: "phenometric.condition-demo-profile.v1",
        profileId: "unilateral-facial-movement-research-demo",
        version: "1.0.0",
        contentSha256: PROFILE_DIGEST,
        status: "nonclinical-research-demo",
        displayName: "Unilateral Facial Movement Research Demo",
        sourceProtocolRef: AMBIENT_LOCAL_PROTOCOL_REF,
        intendedUse:
          "Within-page repeatability demonstration for a participant with previously established unilateral peripheral facial palsy.",
        targetPopulation:
          "Adults participating in a nonclinical research demonstration who assert a previously established unilateral peripheral facial palsy.",
        affectedSidePolicy: {
          source: "participant-asserted",
          verified: false,
          automaticInference: false
        },
        metrics: profileMetricDefinitions,
        boundaryStatement: CONDITION_DEMO_BOUNDARY_STATEMENT,
        prohibitedClaims: CONDITION_DEMO_PROHIBITED_CLAIMS,
        clinicalValidation: "none",
        validatedClaim: "none",
        persistence: "page-memory-only"
      }).success
    ).toBe(true);

    const source = observation(
      "session-reference",
      "observation-reference",
      "2026-08-21T16:00:00.000Z",
      "2026-08-21T16:01:00.000Z"
    );
    expect(
      AcceptedReferenceV1Schema.safeParse({
        schemaVersion: "phenometric.accepted-reference.v1",
        referenceId: "reference-1",
        demoContext: demoContext(),
        observation: source,
        acceptedAt: "2026-08-21T16:02:00.000Z",
        localReviewerAction: {
          action: "use-as-in-memory-reference",
          actor: "local-demo-user",
          status: "accepted"
        },
        persistence: "page-memory-only"
      }).success
    ).toBe(true);
  });

  it("rejects unknown metrics, missing side provenance, and claim-like fields", () => {
    const unknownMetric = structuredClone(demoContext());
    unknownMetric.allowlistedMetricCodes[0] =
      "ambient.face.brow_height_asymmetry.signed" as typeof unknownMetric.allowlistedMetricCodes[0];
    expect(ConditionDemoContextV1Schema.safeParse(unknownMetric).success).toBe(
      false
    );

    const missingProvenance = structuredClone(demoContext()) as Record<
      string,
      unknown
    >;
    missingProvenance.assertedAffectedSide = { side: "left" };
    expect(
      ConditionDemoContextV1Schema.safeParse(missingProvenance).success
    ).toBe(false);

    const freeText = {
      ...demoContext(),
      interpretation: "Facial function improved."
    };
    expect(ConditionDemoContextV1Schema.safeParse(freeText).success).toBe(false);
  });

  it("rejects invalid accepted-reference provenance", () => {
    const source = observation(
      "session-reference",
      "observation-reference",
      "2026-08-21T16:00:00.000Z",
      "2026-08-21T16:01:00.000Z"
    );
    const input = {
      schemaVersion: "phenometric.accepted-reference.v1",
      referenceId: "reference-1",
      demoContext: demoContext(),
      observation: source,
      acceptedAt: "2026-08-21T16:02:00.000Z",
      localReviewerAction: {
        action: "use-as-in-memory-reference",
        actor: "local-demo-user",
        status: "accepted"
      },
      persistence: "page-memory-only"
    };

    const wrongSubject = structuredClone(input);
    wrongSubject.observation.subjectRef = "different-subject";
    expect(AcceptedReferenceV1Schema.safeParse(wrongSubject).success).toBe(
      false
    );

    const missingMetric = structuredClone(input);
    missingMetric.observation.metricOutcomes.pop();
    expect(AcceptedReferenceV1Schema.safeParse(missingMetric).success).toBe(
      false
    );
  });
});

describe("previous-visit comparison and evidence-card contracts", () => {
  it("round-trips a finite, traceable comparison and structured card", () => {
    const artifact = comparison();
    expect(PreviousVisitComparisonV1Schema.parse(artifact)).toEqual(artifact);

    const card = {
      schemaVersion: "phenometric.condition-evidence-card.v1",
      cardId: "card-1",
      generatedAt: artifact.generatedAt,
      comparison: artifact,
      rows: profileMetricDefinitions.map((metric, index) => ({
        metric,
        comparison: artifact.rows[index]
      })),
      qualitySummary: {
        measuredComparisonCount: 6,
        withheldComparisonCount: 0,
        incompatibleComparisonCount: 0,
        referenceMinimumTechnicalQualityScore: 0.9,
        currentMinimumTechnicalQualityScore: 0.8
      },
      review: {
        status: "pending",
        action: null,
        actor: null,
        recordedAt: null
      },
      boundaryStatement: CONDITION_DEMO_BOUNDARY_STATEMENT,
      sourceDisclosure: CONDITION_EVIDENCE_SOURCE_DISCLOSURE,
      persistence: "page-memory-only",
      exportAvailable: false
    };
    expect(ConditionEvidenceCardV1Schema.safeParse(card).success).toBe(true);
  });

  it("rejects non-finite or manufactured deltas and duplicate rows", () => {
    const nonfinite = structuredClone(comparison());
    if (nonfinite.rows[0]?.status !== "measured") {
      throw new Error("Expected a measured row.");
    }
    nonfinite.rows[0].delta = Number.POSITIVE_INFINITY;
    expect(PreviousVisitComparisonV1Schema.safeParse(nonfinite).success).toBe(
      false
    );

    const manufactured = structuredClone(comparison());
    if (manufactured.rows[0]?.status !== "measured") {
      throw new Error("Expected a measured row.");
    }
    manufactured.rows[0].delta = 99;
    expect(PreviousVisitComparisonV1Schema.safeParse(manufactured).success).toBe(
      false
    );

    const duplicate = structuredClone(comparison());
    duplicate.rows[1] = duplicate.rows[0];
    expect(PreviousVisitComparisonV1Schema.safeParse(duplicate).success).toBe(
      false
    );
  });

  it("rejects forged canonical metric metadata and source evidence bindings", () => {
    const wrongUnit = structuredClone(comparison());
    const wrongUnitRow = wrongUnit.rows[0];
    if (wrongUnitRow?.status !== "measured") {
      throw new Error("Expected a measured row.");
    }
    wrongUnitRow.nativeUnit = "diagnosis-points";
    wrongUnitRow.reference.outcome.unit = "diagnosis-points";
    wrongUnitRow.current.outcome.unit = "diagnosis-points";
    expect(PreviousVisitComparisonV1Schema.safeParse(wrongUnit).success).toBe(
      false
    );

    const wrongLabel = structuredClone(comparison());
    const wrongLabelRow = wrongLabel.rows[0];
    if (wrongLabelRow?.status !== "measured") {
      throw new Error("Expected a measured row.");
    }
    wrongLabelRow.reference.outcome.label = "Facial severity score";
    wrongLabelRow.current.outcome.label = "Facial severity score";
    expect(PreviousVisitComparisonV1Schema.safeParse(wrongLabel).success).toBe(
      false
    );

    const wrongEvidenceObservation = structuredClone(comparison());
    const wrongEvidenceRow = wrongEvidenceObservation.rows[0];
    if (wrongEvidenceRow?.status !== "measured") {
      throw new Error("Expected a measured row.");
    }
    wrongEvidenceRow.reference.outcome.evidence.refs[0].observationId =
      "observation-other";
    expect(
      PreviousVisitComparisonV1Schema.safeParse(wrongEvidenceObservation)
        .success
    ).toBe(false);
  });

  it("rejects measured artifacts mutated across protocol, adapter, or chronology boundaries", () => {
    const wrongProtocol = structuredClone(comparison());
    wrongProtocol.currentObservation.protocolRef.version = "9.9.9";

    const wrongAdapter = structuredClone(comparison());
    wrongAdapter.currentObservation.captureAdapter.version = "2.0.0";

    const overlapping = structuredClone(comparison());
    overlapping.currentObservation.startedAt =
      overlapping.referenceObservation.endedAt;

    const generatedEarly = structuredClone(comparison());
    generatedEarly.generatedAt = "2026-08-21T16:05:59.999Z";

    for (const mutated of [
      wrongProtocol,
      wrongAdapter,
      overlapping,
      generatedEarly
    ]) {
      expect(PreviousVisitComparisonV1Schema.safeParse(mutated).success).toBe(
        false
      );
    }
  });

  it("preserves excluded source traceability without admitting a value", () => {
    const artifact = comparison();
    const included = artifact.rows[0];
    if (included?.status !== "measured") {
      throw new Error("Expected a measured row.");
    }
    const trace = (
      source: typeof included.reference,
      status: "measured" | "withheld"
    ) => ({
      observationId: source.observationId,
      sessionId: source.sessionId,
      status,
      outcomeId: source.outcome.outcomeId,
      aggregateId: source.outcome.aggregateId,
      metricCode: source.outcome.metricCode,
      modality: source.outcome.modality,
      context: source.outcome.context,
      unit: source.outcome.unit,
      algorithmVersion: source.outcome.algorithmVersion,
      processorRef: source.outcome.processorRef,
      trackSegmentId: source.outcome.trackSegmentId,
      evidence: source.outcome.evidence,
      processor: source.processor,
      technicalQualityScore: source.outcome.technicalQualityScore,
      withheldReasonCode:
        status === "withheld" ? "insufficient-events" : null
    });
    const excluded = {
      metricCode: included.metricCode,
      nativeUnit: included.nativeUnit,
      status: "withheld",
      decision: "excluded",
      compatibilityReasonCodes: ["reference-withheld"],
      reference: trace(included.reference, "withheld"),
      current: trace(included.current, "measured"),
      delta: null
    };

    expect(
      WithheldPreviousVisitComparisonRowV1Schema.safeParse(excluded).success
    ).toBe(true);
    expect(
      WithheldPreviousVisitComparisonRowV1Schema.safeParse({
        ...excluded,
        current: { ...excluded.current, value: 7 }
      }).success
    ).toBe(false);

    expect(
      WithheldPreviousVisitComparisonRowV1Schema.safeParse({
        ...excluded,
        reference: {
          ...excluded.reference,
          withheldDetail: "Facial palsy severity improved after treatment."
        }
      }).success
    ).toBe(false);
  });

  it("rejects an evidence card generated before its comparison", () => {
    const artifact = comparison();
    const card = {
      schemaVersion: "phenometric.condition-evidence-card.v1",
      cardId: "card-early",
      generatedAt: "2026-08-21T16:05:59.999Z",
      comparison: artifact,
      rows: profileMetricDefinitions.map((metric, index) => ({
        metric,
        comparison: artifact.rows[index]
      })),
      qualitySummary: {
        measuredComparisonCount: 6,
        withheldComparisonCount: 0,
        incompatibleComparisonCount: 0,
        referenceMinimumTechnicalQualityScore: 0.9,
        currentMinimumTechnicalQualityScore: 0.8
      },
      review: {
        status: "pending",
        action: null,
        actor: null,
        recordedAt: null
      },
      boundaryStatement: CONDITION_DEMO_BOUNDARY_STATEMENT,
      sourceDisclosure: CONDITION_EVIDENCE_SOURCE_DISCLOSURE,
      persistence: "page-memory-only",
      exportAvailable: false
    };

    expect(ConditionEvidenceCardV1Schema.safeParse(card).success).toBe(false);
  });

  it("rejects free clinical narrative and unsafe display labels", () => {
    const artifact = comparison();
    const base = {
      schemaVersion: "phenometric.condition-evidence-card.v1",
      cardId: "card-1",
      generatedAt: artifact.generatedAt,
      comparison: artifact,
      rows: profileMetricDefinitions.map((metric, index) => ({
        metric,
        comparison: artifact.rows[index]
      })),
      qualitySummary: {
        measuredComparisonCount: 6,
        withheldComparisonCount: 0,
        incompatibleComparisonCount: 0,
        referenceMinimumTechnicalQualityScore: 0.9,
        currentMinimumTechnicalQualityScore: 0.8
      },
      review: {
        status: "pending",
        action: null,
        actor: null,
        recordedAt: null
      },
      boundaryStatement: CONDITION_DEMO_BOUNDARY_STATEMENT,
      sourceDisclosure: CONDITION_EVIDENCE_SOURCE_DISCLOSURE,
      persistence: "page-memory-only",
      exportAvailable: false
    };

    expect(
      ConditionEvidenceCardV1Schema.safeParse({
        ...base,
        narrative: "The participant improved."
      }).success
    ).toBe(false);

    const unsafeLabel = structuredClone(base) as unknown as {
      rows: Array<{ metric: { displayLabel: string } }>;
    };
    unsafeLabel.rows[0].metric.displayLabel = "Facial palsy improved";
    expect(ConditionEvidenceCardV1Schema.safeParse(unsafeLabel).success).toBe(
      false
    );
  });
});
