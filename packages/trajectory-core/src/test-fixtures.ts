import {
  AMBIENT_LOCAL_PROTOCOL_PACK,
  AMBIENT_LOCAL_PROTOCOL_REF,
  CONDITION_DEMO_BOUNDARY_STATEMENT,
  CONDITION_DEMO_METRIC_CODES,
  CONDITION_DEMO_PROHIBITED_CLAIMS,
  ConditionDemoContextV1Schema,
  AcceptedReferenceV1Schema,
  UNILATERAL_FACIAL_MOVEMENT_PROFILE_ID,
  createAggregateId,
  createMeasurementId,
  type AcceptedReferenceV1,
  type AmbientMeasurementContext,
  type AmbientModality,
  type ConditionDemoContextV1,
  type ConditionDemoMetricCode,
  type MetricCode,
  type ObservationV3,
  type ProcessorProvenanceV1,
  type ProtocolRef
} from "@phenometrix/contracts";
import type { ComparePreviousVisitInput } from "./compare-previous.js";

const PROFILE_DIGEST = "a".repeat(64);
const FACE_ASSET_DIGEST = "b".repeat(64);

interface MetricOverride {
  value?: number;
  unit?: string;
  context?: AmbientMeasurementContext;
  modality?: AmbientModality;
  algorithmVersion?: string;
  withheldReason?: "insufficient-events" | "quality-threshold-failed";
}

export interface ObservationFixtureOptions {
  sessionId: string;
  observationId: string;
  subjectRef?: string;
  startedAt: string;
  endedAt: string;
  valueOffset?: number;
  protocolRef?: ProtocolRef;
  captureAdapter?: { id: string; version: string };
  processor?: Partial<ProcessorProvenanceV1>;
  processors?: ProcessorProvenanceV1[];
  metricOverrides?: Partial<Record<ConditionDemoMetricCode, MetricOverride>>;
  omittedMetric?: ConditionDemoMetricCode;
}

export function createObservationFixture(
  options: ObservationFixtureOptions
): ObservationV3 {
  const protocolRef = options.protocolRef ?? AMBIENT_LOCAL_PROTOCOL_REF;
  const durationMs = Date.parse(options.endedAt) - Date.parse(options.startedAt);
  const processor: ProcessorProvenanceV1 = {
    modality: "face",
    processorRef: "face-geometry@1.0.0",
    runtime: "mediapipe-tasks-vision",
    runtimeVersion: "0.10.35",
    assetPath: "assets/face_landmarker.task",
    assetSha256: FACE_ASSET_DIGEST,
    assetIntegrityVerified: true,
    ...options.processor
  };
  const definitions = CONDITION_DEMO_METRIC_CODES.flatMap((metricCode) => {
    if (metricCode === options.omittedMetric) return [];
    const definition = AMBIENT_LOCAL_PROTOCOL_PACK.metrics.find(
      (candidate) => candidate.code === metricCode
    );
    if (!definition) throw new Error(`Missing fixture definition ${metricCode}.`);
    return [definition];
  });

  const artifacts = definitions.map((definition, index) => {
    const override = options.metricOverrides?.[
      definition.code as ConditionDemoMetricCode
    ];
    const context = override?.context ?? definition.context;
    const modality = override?.modality ?? definition.modality;
    const unit = override?.unit ?? definition.unit;
    const algorithmVersion =
      override?.algorithmVersion ?? definition.algorithmVersion;
    const trackSegmentId = "face-track-1";
    const windowId = `window:${options.sessionId}:${index}`;
    const identity = {
      protocolPackId: protocolRef.packId,
      protocolVersion: protocolRef.version,
      sessionId: options.sessionId,
      metricCode: definition.code,
      context,
      unit,
      algorithmVersion,
      processorRef: processor.processorRef,
      trackSegmentId
    };
    const aggregateId = createAggregateId(identity);
    const measurementId = createMeasurementId(identity, windowId, 0);
    const withheldReason = override?.withheldReason;
    const value = override?.value ?? index / 100 + (options.valueOffset ?? 0);
    const window = {
      windowId,
      sessionId: options.sessionId,
      modality,
      context,
      trackSegmentId,
      processorRef: processor.processorRef,
      startMs: 0,
      endMs: durationMs,
      technicalQualityScore: withheldReason ? 0.4 : 0.9,
      status: withheldReason ? ("withheld" as const) : ("eligible" as const),
      reasonCodes: withheldReason ? [withheldReason] : []
    };
    const common = {
      outcomeId: `outcome:${options.sessionId}:${index}`,
      aggregateId,
      metricCode: definition.code,
      label: definition.label,
      modality,
      context,
      unit,
      reportSection: definition.reportSection,
      algorithmVersion,
      processorRef: processor.processorRef,
      trackSegmentId,
      technicalVerification: "automated-test" as const,
      clinicalValidation: "none" as const,
      evidence: {
        eligibleDurationMs: withheldReason ? 0 : durationMs,
        activeDurationMs: 0,
        segmentCount: 0,
        windowCount: 1,
        binCount: withheldReason ? 0 : 1,
        eventCount: 3,
        sampleCount: withheldReason ? 0 : 100,
        coverage: null,
        qualityFacts: {
          expressionEventCount: 3,
          coupledExpressionEventCount: 3
        },
        refs: [
          {
            schemaVersion: "phenometric.evidence-ref.v1" as const,
            kind: "window" as const,
            sessionId: options.sessionId,
            observationId: options.observationId,
            protocolRef,
            windowId,
            modality,
            context,
            trackSegmentId
          },
          ...(!withheldReason
            ? [
                {
                  schemaVersion: "phenometric.evidence-ref.v1" as const,
                  kind: "measurement" as const,
                  sessionId: options.sessionId,
                  observationId: options.observationId,
                  protocolRef,
                  measurementId,
                  metricCode: definition.code,
                  modality,
                  context,
                  unit,
                  trackSegmentId
                }
              ]
            : []),
          {
            schemaVersion: "phenometric.evidence-ref.v1" as const,
            kind: "aggregate" as const,
            sessionId: options.sessionId,
            observationId: options.observationId,
            protocolRef,
            aggregateId,
            metricCode: definition.code,
            modality,
            context,
            unit,
            trackSegmentId
          }
        ]
      }
    };
    const outcome = withheldReason
      ? {
          ...common,
          status: "withheld" as const,
          reasonCode: withheldReason,
          detail: "Fixture outcome was intentionally withheld.",
          technicalQualityScore: 0.4,
          technicalDispersion: null
        }
      : {
          ...common,
          status: "measured" as const,
          value,
          technicalQualityScore: 0.9,
          technicalDispersion: 0.01
        };
    const measurement = withheldReason
      ? null
      : {
          measurementId,
          aggregateId,
          sessionId: options.sessionId,
          metricCode: definition.code,
          label: definition.label,
          modality,
          context,
          unit,
          value,
          technicalQualityScore: 0.9,
          algorithmVersion,
          processorRef: processor.processorRef,
          trackSegmentId,
          ordinal: 0,
          sourceWindowRefs: [windowId]
        };
    return { window, outcome, measurement };
  });
  const eligibleWindowCount = artifacts.filter(
    ({ window }) => window.status === "eligible"
  ).length;

  return {
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
    observationId: options.observationId,
    sessionId: options.sessionId,
    subjectRef: options.subjectRef ?? "demo-subject",
    protocolRef,
    consent: {
      schemaVersion: "phenometric.consent-record.v1",
      consentId: `consent:${options.sessionId}`,
      sessionId: options.sessionId,
      documentVersion: "ambient-local-consent.v1",
      documentSha256:
        AMBIENT_LOCAL_PROTOCOL_PACK.consentDocument.contentSha256,
      recordedAt: options.startedAt,
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
      sourceSessionRef: options.sessionId,
      audioAttribution: "user-asserted-local-participant",
      speakerAttribution: "unverified-local-input",
      audioInput: "microphone",
      faceAttribution: "single-visible-face",
      identityVerified: false
    },
    startedAt: options.startedAt,
    endedAt: options.endedAt,
    durationMs,
    captureAdapter: options.captureAdapter ?? {
      id: "browser-local-media",
      version: "1.1.0"
    },
    processors: options.processors ?? [processor],
    windows: artifacts.map(({ window }) => window),
    measurements: artifacts.flatMap(({ measurement }) =>
      measurement ? [measurement] : []
    ),
    metricOutcomes: artifacts.map(({ outcome }) => outcome),
    qualitySummary: {
      voice: {
        state: "unavailable",
        eligibleDurationMs: 0,
        technicalQualityScore: null,
        reasonCodes: ["modality-unavailable"]
      },
      face: {
        state: eligibleWindowCount > 0 ? "ready" : "withheld",
        eligibleDurationMs: eligibleWindowCount > 0 ? durationMs : 0,
        technicalQualityScore: eligibleWindowCount > 0 ? 0.9 : null,
        reasonCodes: eligibleWindowCount > 0 ? [] : ["insufficient-events"]
      },
      totalWindowCount: artifacts.length,
      eligibleWindowCount,
      withheldWindowCount: artifacts.length - eligibleWindowCount
    }
  };
}

export interface ContextFixtureOptions {
  subjectRef?: string;
  side?: "left" | "right";
  profileId?: string;
  profileVersion?: string;
  profileDigest?: string;
  protocolRef?: ConditionDemoContextV1["sourceProtocolRef"];
}

export function createContextFixture(
  options: ContextFixtureOptions = {}
): ConditionDemoContextV1 {
  return ConditionDemoContextV1Schema.parse({
    schemaVersion: "phenometric.condition-demo-context.v1",
    profileRef: {
      profileId:
        options.profileId ?? UNILATERAL_FACIAL_MOVEMENT_PROFILE_ID,
      version: options.profileVersion ?? "1.0.0",
      contentSha256: options.profileDigest ?? PROFILE_DIGEST
    },
    subjectRef: options.subjectRef ?? "demo-subject",
    assertedAffectedSide: {
      side: options.side ?? "left",
      source: "participant-asserted",
      verified: false
    },
    sourceProtocolRef: options.protocolRef ?? AMBIENT_LOCAL_PROTOCOL_REF,
    allowlistedMetricCodes: CONDITION_DEMO_METRIC_CODES,
    boundaryStatement: CONDITION_DEMO_BOUNDARY_STATEMENT,
    prohibitedClaims: CONDITION_DEMO_PROHIBITED_CLAIMS,
    persistence: "page-memory-only"
  });
}

export function createAcceptedReferenceFixture(
  observation: ObservationV3,
  demoContext: ConditionDemoContextV1
): AcceptedReferenceV1 {
  return AcceptedReferenceV1Schema.parse({
    schemaVersion: "phenometric.accepted-reference.v1",
    referenceId: `reference:${observation.sessionId}`,
    demoContext,
    observation,
    acceptedAt: observation.endedAt,
    localReviewerAction: {
      action: "use-as-in-memory-reference",
      actor: "local-demo-user",
      status: "accepted"
    },
    persistence: "page-memory-only"
  });
}

export function createComparisonInput(
  options: {
    referenceObservation?: ObservationV3;
    currentObservation?: ObservationV3;
    referenceContext?: ConditionDemoContextV1;
    currentContext?: ConditionDemoContextV1;
  } = {}
): ComparePreviousVisitInput {
  const referenceObservation =
    options.referenceObservation ??
    createObservationFixture({
      sessionId: "session-reference",
      observationId: "observation-reference",
      startedAt: "2026-08-21T10:00:00.000Z",
      endedAt: "2026-08-21T10:01:00.000Z"
    });
  const currentObservation =
    options.currentObservation ??
    createObservationFixture({
      sessionId: "session-current",
      observationId: "observation-current",
      startedAt: "2026-08-21T10:02:00.000Z",
      endedAt: "2026-08-21T10:03:00.000Z",
      valueOffset: 0.005
    });
  const referenceContext =
    options.referenceContext ??
    createContextFixture({
      subjectRef: referenceObservation.subjectRef,
      protocolRef: referenceObservation.protocolRef
    });
  const currentContext =
    options.currentContext ??
    createContextFixture({
      subjectRef: currentObservation.subjectRef,
      protocolRef: currentObservation.protocolRef
    });
  return {
    currentContext,
    acceptedReference: createAcceptedReferenceFixture(
      referenceObservation,
      referenceContext
    ),
    currentObservation
  };
}

export function metricOverride(
  metricCode: MetricCode,
  override: MetricOverride
): Partial<Record<ConditionDemoMetricCode, MetricOverride>> {
  return { [metricCode]: override };
}
