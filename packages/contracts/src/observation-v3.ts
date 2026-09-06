import { z } from "zod";
import {
  createAggregateId,
  createMeasurementId
} from "./identity.js";
import {
  AmbientMeasurementContextSchema,
  AmbientModalitySchema,
  MetricCodeSchema,
  ProtocolRefSchema,
  ReportSectionIdSchema
} from "./protocol.js";

const IdSchema = z.string().min(1).max(160).regex(/^[A-Za-z0-9._:-]+$/);
const IsoTimestampSchema = z.string().datetime({ offset: true });
const Sha256Schema = z.string().regex(/^[a-f0-9]{64}$/);
const FiniteNonNegativeSchema = z.number().finite().nonnegative();
const TechnicalQualityScoreSchema = z.number().finite().min(0).max(1);

export const ConsentRecordV1Schema = z
  .object({
    schemaVersion: z.literal("phenometric.consent-record.v1"),
    consentId: IdSchema,
    sessionId: IdSchema,
    documentVersion: z.literal("ambient-local-consent.v1"),
    documentSha256: Sha256Schema,
    recordedAt: IsoTimestampSchema,
    scopes: z
      .object({
        cameraCapture: z.literal(true),
        microphoneCapture: z.literal(true),
        localInMemoryAnalysis: z.literal(true)
      })
      .strict(),
    localParticipantAssertion: z.literal(true),
    withdrawnAt: z.null()
  })
  .strict();
export type ConsentRecordV1 = z.infer<typeof ConsentRecordV1Schema>;

export const SourceAttributionV1Schema = z
  .object({
    role: z.literal("local-participant"),
    sourceSessionRef: IdSchema,
    audioAttribution: z.literal("user-asserted-local-participant"),
    speakerAttribution: z.literal("unverified-local-input"),
    audioInput: z.literal("microphone"),
    faceAttribution: z.literal("single-visible-face"),
    identityVerified: z.literal(false)
  })
  .strict();
export type SourceAttributionV1 = z.infer<
  typeof SourceAttributionV1Schema
>;

const EvidenceRefBaseShape = {
  schemaVersion: z.literal("phenometric.evidence-ref.v1"),
  sessionId: IdSchema,
  observationId: IdSchema,
  protocolRef: ProtocolRefSchema
} as const;

export const WindowEvidenceRefSchema = z
  .object({
    ...EvidenceRefBaseShape,
    kind: z.literal("window"),
    windowId: IdSchema,
    modality: AmbientModalitySchema,
    context: AmbientMeasurementContextSchema,
    trackSegmentId: IdSchema
  })
  .strict();

export const MeasurementEvidenceRefSchema = z
  .object({
    ...EvidenceRefBaseShape,
    kind: z.literal("measurement"),
    measurementId: IdSchema,
    metricCode: MetricCodeSchema,
    modality: AmbientModalitySchema,
    context: AmbientMeasurementContextSchema,
    unit: z.string().min(1),
    trackSegmentId: IdSchema
  })
  .strict();

export const AggregateEvidenceRefSchema = z
  .object({
    ...EvidenceRefBaseShape,
    kind: z.literal("aggregate"),
    aggregateId: IdSchema,
    metricCode: MetricCodeSchema,
    modality: AmbientModalitySchema,
    context: AmbientMeasurementContextSchema,
    unit: z.string().min(1),
    trackSegmentId: IdSchema
  })
  .strict();

export const EventEvidenceRefSchema = z
  .object({
    ...EvidenceRefBaseShape,
    kind: z.literal("event"),
    eventId: z.string().uuid()
  })
  .strict();

export const EvidenceRefSchema = z.discriminatedUnion("kind", [
  WindowEvidenceRefSchema,
  MeasurementEvidenceRefSchema,
  AggregateEvidenceRefSchema,
  EventEvidenceRefSchema
]);
export type EvidenceRef = z.infer<typeof EvidenceRefSchema>;
export type WindowEvidenceRef = z.infer<typeof WindowEvidenceRefSchema>;
export type MeasurementEvidenceRef = z.infer<
  typeof MeasurementEvidenceRefSchema
>;
export type AggregateEvidenceRef = z.infer<
  typeof AggregateEvidenceRefSchema
>;
export type EventEvidenceRef = z.infer<typeof EventEvidenceRefSchema>;

export const EvidenceWindowV1Schema = z
  .object({
    windowId: IdSchema,
    sessionId: IdSchema,
    modality: AmbientModalitySchema,
    context: AmbientMeasurementContextSchema,
    trackSegmentId: IdSchema,
    processorRef: z.string().min(1),
    startMs: FiniteNonNegativeSchema,
    endMs: FiniteNonNegativeSchema,
    technicalQualityScore: TechnicalQualityScoreSchema,
    status: z.enum(["eligible", "withheld"]),
    reasonCodes: z.array(z.string().min(1))
  })
  .strict()
  .refine((window) => window.endMs > window.startMs, {
    path: ["endMs"],
    message: "Window end must be after its start."
  });
export type EvidenceWindowV1 = z.infer<typeof EvidenceWindowV1Schema>;

export const MeasurementV3Schema = z
  .object({
    measurementId: IdSchema,
    aggregateId: IdSchema,
    sessionId: IdSchema,
    metricCode: MetricCodeSchema,
    label: z.string().min(1),
    modality: AmbientModalitySchema,
    context: AmbientMeasurementContextSchema,
    unit: z.string().min(1),
    value: z.number().finite(),
    technicalQualityScore: TechnicalQualityScoreSchema,
    algorithmVersion: z.string().min(1),
    processorRef: z.string().min(1),
    trackSegmentId: IdSchema,
    ordinal: z.number().int().nonnegative(),
    sourceWindowRefs: z.array(IdSchema).min(1)
  })
  .strict();
export type MeasurementV3 = z.infer<typeof MeasurementV3Schema>;

export const WithheldReasonCodeSchema = z.enum([
  "modality-unavailable",
  "processor-unavailable",
  "asset-integrity-failed",
  "quality-threshold-failed",
  "no-usable-signal",
  "insufficient-duration",
  "insufficient-active-speech",
  "insufficient-pitched-speech",
  "insufficient-segments",
  "insufficient-bins",
  "insufficient-events",
  "insufficient-nuclei",
  "insufficient-pitch-bins",
  "pitch-estimator-disagreement",
  "multiple-faces",
  "pose-out-of-range",
  "face-scale-out-of-range",
  "insufficient-exposure",
  "insufficient-frame-cadence",
  "session-ended-early"
]);
export type WithheldReasonCode = z.infer<
  typeof WithheldReasonCodeSchema
>;

export const MetricEvidenceSummaryV1Schema = z
  .object({
    eligibleDurationMs: FiniteNonNegativeSchema,
    activeDurationMs: FiniteNonNegativeSchema,
    segmentCount: z.number().int().nonnegative(),
    windowCount: z.number().int().nonnegative(),
    binCount: z.number().int().nonnegative(),
    eventCount: z.number().int().nonnegative(),
    sampleCount: z.number().int().nonnegative(),
    coverage: z.number().finite().min(0).max(1).nullable(),
    qualityFacts: z.record(
      z.string().min(1),
      z.union([z.string().min(1), z.number().finite(), z.boolean()])
    ),
    refs: z.array(EvidenceRefSchema).min(1)
  })
  .strict();
export type MetricEvidenceSummaryV1 = z.infer<
  typeof MetricEvidenceSummaryV1Schema
>;

const MetricOutcomeBaseShape = {
  outcomeId: IdSchema,
  aggregateId: IdSchema,
  metricCode: MetricCodeSchema,
  label: z.string().min(1),
  modality: AmbientModalitySchema,
  context: AmbientMeasurementContextSchema,
  unit: z.string().min(1),
  reportSection: ReportSectionIdSchema.exclude(["capture-quality"]),
  algorithmVersion: z.string().min(1),
  processorRef: z.string().min(1),
  trackSegmentId: IdSchema,
  technicalVerification: z.literal("automated-test"),
  clinicalValidation: z.literal("none"),
  evidence: MetricEvidenceSummaryV1Schema
} as const;

export const MeasuredMetricOutcomeV1Schema = z
  .object({
    ...MetricOutcomeBaseShape,
    status: z.literal("measured"),
    value: z.number().finite(),
    technicalQualityScore: TechnicalQualityScoreSchema,
    technicalDispersion: FiniteNonNegativeSchema.nullable()
  })
  .strict();

export const WithheldMetricOutcomeV1Schema = z
  .object({
    ...MetricOutcomeBaseShape,
    status: z.literal("withheld"),
    reasonCode: WithheldReasonCodeSchema,
    detail: z.string().min(1).max(240),
    technicalQualityScore: TechnicalQualityScoreSchema.nullable(),
    technicalDispersion: z.null()
  })
  .strict();

export const MetricOutcomeV1Schema = z.discriminatedUnion("status", [
  MeasuredMetricOutcomeV1Schema,
  WithheldMetricOutcomeV1Schema
]);
export type MeasuredMetricOutcomeV1 = z.infer<
  typeof MeasuredMetricOutcomeV1Schema
>;
export type WithheldMetricOutcomeV1 = z.infer<
  typeof WithheldMetricOutcomeV1Schema
>;
export type MetricOutcomeV1 = z.infer<typeof MetricOutcomeV1Schema>;

export const ProcessorProvenanceV1Schema = z
  .object({
    modality: AmbientModalitySchema,
    processorRef: z.string().min(1),
    runtime: z.string().min(1),
    runtimeVersion: z.string().min(1),
    assetPath: z.string().min(1).nullable(),
    assetSha256: Sha256Schema.nullable(),
    assetIntegrityVerified: z.boolean()
  })
  .strict();
export type ProcessorProvenanceV1 = z.infer<
  typeof ProcessorProvenanceV1Schema
>;

const LaneQualitySummarySchema = z
  .object({
    state: z.enum(["ready", "unavailable", "withheld"]),
    eligibleDurationMs: FiniteNonNegativeSchema,
    technicalQualityScore: TechnicalQualityScoreSchema.nullable(),
    reasonCodes: z.array(z.string().min(1))
  })
  .strict();

export const ObservationV3Schema = z
  .object({
    schemaVersion: z.literal("phenometric.encounter-observation.v3"),
    containsPHI: z.literal(false),
    retention: z
      .object({
        rawMedia: z.literal(false),
        rawAudio: z.literal(false),
        rawVideo: z.literal(false),
        transcript: z.literal(false),
        embeddings: z.literal(false),
        persisted: z.literal(false)
      })
      .strict(),
    observationId: IdSchema,
    sessionId: IdSchema,
    subjectRef: IdSchema,
    protocolRef: ProtocolRefSchema,
    consent: ConsentRecordV1Schema,
    source: SourceAttributionV1Schema,
    startedAt: IsoTimestampSchema,
    endedAt: IsoTimestampSchema,
    durationMs: FiniteNonNegativeSchema.max(300_000),
    captureAdapter: z
      .object({ id: z.string().min(1), version: z.string().min(1) })
      .strict(),
    processors: z.array(ProcessorProvenanceV1Schema),
    windows: z.array(EvidenceWindowV1Schema),
    measurements: z.array(MeasurementV3Schema),
    metricOutcomes: z.array(MetricOutcomeV1Schema).min(1),
    qualitySummary: z
      .object({
        voice: LaneQualitySummarySchema,
        face: LaneQualitySummarySchema,
        totalWindowCount: z.number().int().nonnegative(),
        eligibleWindowCount: z.number().int().nonnegative(),
        withheldWindowCount: z.number().int().nonnegative()
      })
      .strict()
  })
  .strict()
  .superRefine((observation, context) => {
    const start = Date.parse(observation.startedAt);
    const end = Date.parse(observation.endedAt);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) {
      context.addIssue({
        code: "custom",
        path: ["endedAt"],
        message: "Observation end must not precede its start."
      });
    } else if (Math.abs(end - start - observation.durationMs) > 1_000) {
      context.addIssue({
        code: "custom",
        path: ["durationMs"],
        message: "Observation duration does not match its timestamps."
      });
    }
    if (observation.consent.sessionId !== observation.sessionId) {
      context.addIssue({
        code: "custom",
        path: ["consent", "sessionId"],
        message: "Consent must belong to the observation session."
      });
    }
    const consentRecordedAt = Date.parse(observation.consent.recordedAt);
    if (consentRecordedAt > start) {
      context.addIssue({
        code: "custom",
        path: ["consent", "recordedAt"],
        message: "Consent must be recorded no later than the observation start."
      });
    }
    if (observation.source.sourceSessionRef !== observation.sessionId) {
      context.addIssue({
        code: "custom",
        path: ["source", "sourceSessionRef"],
        message: "Source attribution must belong to the observation session."
      });
    }
    const unique = (
      values: string[],
      path: (string | number)[],
      label: string
    ): void => {
      if (new Set(values).size !== values.length) {
        context.addIssue({
          code: "custom",
          path,
          message: `${label} must be unique.`
        });
      }
    };
    unique(
      observation.windows.map((window) => window.windowId),
      ["windows"],
      "Window IDs"
    );
    unique(
      observation.measurements.map(
        (measurement) => measurement.measurementId
      ),
      ["measurements"],
      "Measurement IDs"
    );
    unique(
      observation.metricOutcomes.map((outcome) => outcome.outcomeId),
      ["metricOutcomes"],
      "Outcome IDs"
    );
    unique(
      observation.metricOutcomes.map((outcome) => outcome.aggregateId),
      ["metricOutcomes"],
      "Aggregate IDs"
    );
    unique(
      observation.metricOutcomes.map((outcome) => outcome.metricCode),
      ["metricOutcomes"],
      "Terminal metric codes"
    );
    const eligibleWindowCount = observation.windows.filter(
      (window) => window.status === "eligible"
    ).length;
    const withheldWindowCount = observation.windows.length - eligibleWindowCount;
    if (
      observation.qualitySummary.totalWindowCount !==
        observation.windows.length ||
      observation.qualitySummary.eligibleWindowCount !==
        eligibleWindowCount ||
      observation.qualitySummary.withheldWindowCount !==
        withheldWindowCount
    ) {
      context.addIssue({
        code: "custom",
        path: ["qualitySummary"],
        message: "Quality-summary window counts do not match the observation."
      });
    }
    observation.windows.forEach((window, index) => {
      if (
        window.sessionId !== observation.sessionId ||
        window.endMs > observation.durationMs
      ) {
        context.addIssue({
          code: "custom",
          path: ["windows", index],
          message: "Window falls outside the observation session."
        });
      }
    });
    for (const outcome of observation.metricOutcomes) {
      const expectedAggregateId = createAggregateId({
        protocolPackId: observation.protocolRef.packId,
        protocolVersion: observation.protocolRef.version,
        sessionId: observation.sessionId,
        metricCode: outcome.metricCode,
        context: outcome.context,
        unit: outcome.unit,
        algorithmVersion: outcome.algorithmVersion,
        processorRef: outcome.processorRef,
        trackSegmentId: outcome.trackSegmentId
      });
      if (outcome.aggregateId !== expectedAggregateId) {
        context.addIssue({
          code: "custom",
          path: ["metricOutcomes", outcome.metricCode, "aggregateId"],
          message: "Aggregate ID does not match its canonical identity."
        });
      }
    }
    for (const measurement of observation.measurements) {
      const identity = {
        protocolPackId: observation.protocolRef.packId,
        protocolVersion: observation.protocolRef.version,
        sessionId: observation.sessionId,
        metricCode: measurement.metricCode,
        context: measurement.context,
        unit: measurement.unit,
        algorithmVersion: measurement.algorithmVersion,
        processorRef: measurement.processorRef,
        trackSegmentId: measurement.trackSegmentId
      };
      const expectedAggregateId = createAggregateId(identity);
      const expectedMeasurementId = createMeasurementId(
        identity,
        measurement.sourceWindowRefs[0],
        measurement.ordinal
      );
      if (measurement.aggregateId !== expectedAggregateId) {
        context.addIssue({
          code: "custom",
          path: ["measurements", measurement.measurementId, "aggregateId"],
          message: "Measurement aggregate ID does not match its canonical identity."
        });
      }
      if (measurement.measurementId !== expectedMeasurementId) {
        context.addIssue({
          code: "custom",
          path: ["measurements", measurement.measurementId, "measurementId"],
          message: "Measurement ID does not match its canonical identity."
        });
      }
    }
  });
export type ObservationV3 = z.infer<typeof ObservationV3Schema>;
