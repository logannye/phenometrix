import { z } from "zod";
import {
  AMBIENT_LOCAL_PROTOCOL_PACK,
  AMBIENT_LOCAL_PROTOCOL_REF
} from "./ambient-protocol.js";
import {
  CONDITION_DEMO_METRIC_CODES,
  ConditionDemoContextV1Schema,
  ConditionDemoMetricCodeSchema,
  PREVIOUS_VISIT_CLAIM_BOUNDARY
} from "./condition-demo.js";
import {
  MeasuredMetricOutcomeV1Schema,
  MetricEvidenceSummaryV1Schema,
  ProcessorProvenanceV1Schema,
  WithheldReasonCodeSchema
} from "./observation-v3.js";
import {
  AmbientMeasurementContextSchema,
  AmbientModalitySchema,
  ProtocolRefSchema
} from "./protocol.js";

const IdSchema = z.string().min(1).max(160).regex(/^[A-Za-z0-9._:-]+$/);
const IsoTimestampSchema = z.string().datetime({ offset: true });

export const PreviousVisitCompatibilityReasonCodeSchema = z.enum([
  "subject-mismatch",
  "profile-id-mismatch",
  "profile-version-mismatch",
  "profile-digest-mismatch",
  "asserted-side-mismatch",
  "protocol-pack-id-mismatch",
  "protocol-version-mismatch",
  "protocol-digest-mismatch",
  "metric-missing",
  "metric-definition-mismatch",
  "modality-mismatch",
  "context-mismatch",
  "unit-mismatch",
  "algorithm-version-mismatch",
  "processor-ref-mismatch",
  "processor-provenance-missing",
  "processor-provenance-ambiguous",
  "processor-runtime-mismatch",
  "processor-runtime-version-mismatch",
  "processor-asset-path-mismatch",
  "processor-asset-digest-mismatch",
  "processor-asset-integrity-mismatch",
  "capture-adapter-id-mismatch",
  "capture-adapter-version-mismatch",
  "reference-withheld",
  "current-withheld",
  "reference-not-before-current",
  "non-finite-delta"
]);
export type PreviousVisitCompatibilityReasonCode = z.infer<
  typeof PreviousVisitCompatibilityReasonCodeSchema
>;

export const PreviousVisitObservationRefV1Schema = z
  .object({
    observationId: IdSchema,
    sessionId: IdSchema,
    startedAt: IsoTimestampSchema,
    endedAt: IsoTimestampSchema,
    protocolRef: ProtocolRefSchema,
    captureAdapter: z
      .object({
        id: z.string().min(1),
        version: z.string().min(1)
      })
      .strict()
  })
  .strict()
  .refine(
    (observation) =>
      Date.parse(observation.endedAt) >= Date.parse(observation.startedAt),
    {
      path: ["endedAt"],
      message: "Comparison observation end cannot precede its start."
    }
  );
export type PreviousVisitObservationRefV1 = z.infer<
  typeof PreviousVisitObservationRefV1Schema
>;

export const MeasuredPreviousVisitMetricSourceV1Schema = z
  .object({
    observationId: IdSchema,
    sessionId: IdSchema,
    status: z.literal("measured"),
    outcome: MeasuredMetricOutcomeV1Schema,
    processor: ProcessorProvenanceV1Schema
  })
  .strict()
  .superRefine((source, context) => {
    if (source.processor.processorRef !== source.outcome.processorRef) {
      context.addIssue({
        code: "custom",
        path: ["processor", "processorRef"],
        message: "Source processor provenance must match the metric outcome."
      });
    }
  });

const ExcludedSourceBaseShape = {
  observationId: IdSchema,
  sessionId: IdSchema,
  outcomeId: IdSchema,
  aggregateId: IdSchema,
  metricCode: ConditionDemoMetricCodeSchema,
  modality: AmbientModalitySchema,
  context: AmbientMeasurementContextSchema,
  unit: z.string().min(1),
  algorithmVersion: z.string().min(1),
  processorRef: z.string().min(1),
  trackSegmentId: IdSchema,
  evidence: MetricEvidenceSummaryV1Schema,
  processor: ProcessorProvenanceV1Schema
} as const;

const ExcludedMeasuredPreviousVisitMetricSourceV1Schema = z
  .object({
    ...ExcludedSourceBaseShape,
    status: z.literal("measured"),
    technicalQualityScore: z.number().finite().min(0).max(1),
    withheldReasonCode: z.null()
  })
  .strict()
  .superRefine((source, context) => {
    if (source.processor.processorRef !== source.processorRef) {
      context.addIssue({
        code: "custom",
        path: ["processor", "processorRef"],
        message: "Source processor provenance must match the source trace."
      });
    }
  });

const ExcludedWithheldPreviousVisitMetricSourceV1Schema = z
  .object({
    ...ExcludedSourceBaseShape,
    status: z.literal("withheld"),
    technicalQualityScore: z.number().finite().min(0).max(1).nullable(),
    withheldReasonCode: WithheldReasonCodeSchema
  })
  .strict()
  .superRefine((source, context) => {
    if (source.processor.processorRef !== source.processorRef) {
      context.addIssue({
        code: "custom",
        path: ["processor", "processorRef"],
        message: "Source processor provenance must match the source trace."
      });
    }
  });

export const ExcludedPreviousVisitMetricSourceV1Schema = z.discriminatedUnion(
  "status",
  [
    ExcludedMeasuredPreviousVisitMetricSourceV1Schema,
    ExcludedWithheldPreviousVisitMetricSourceV1Schema
  ]
);
export type MeasuredPreviousVisitMetricSourceV1 = z.infer<
  typeof MeasuredPreviousVisitMetricSourceV1Schema
>;
export type ExcludedPreviousVisitMetricSourceV1 = z.infer<
  typeof ExcludedPreviousVisitMetricSourceV1Schema
>;

type ComparisonMetricSourceV1 =
  | MeasuredPreviousVisitMetricSourceV1
  | ExcludedPreviousVisitMetricSourceV1;

interface ContractValidationIssue {
  path: (string | number)[];
  message: string;
}

function canonicalMetricDefinition(
  metricCode: z.infer<typeof ConditionDemoMetricCodeSchema>
) {
  const definition = AMBIENT_LOCAL_PROTOCOL_PACK.metrics.find(
    (candidate) => candidate.code === metricCode
  );
  if (!definition) {
    throw new Error(`Missing canonical condition metric ${metricCode}.`);
  }
  return definition;
}

function sourceTrace(source: ComparisonMetricSourceV1) {
  if ("outcome" in source) {
    return {
      metricCode: source.outcome.metricCode,
      aggregateId: source.outcome.aggregateId,
      modality: source.outcome.modality,
      context: source.outcome.context,
      unit: source.outcome.unit,
      algorithmVersion: source.outcome.algorithmVersion,
      processorRef: source.outcome.processorRef,
      trackSegmentId: source.outcome.trackSegmentId,
      evidence: source.outcome.evidence
    };
  }
  return {
    metricCode: source.metricCode,
    aggregateId: source.aggregateId,
    modality: source.modality,
    context: source.context,
    unit: source.unit,
    algorithmVersion: source.algorithmVersion,
    processorRef: source.processorRef,
    trackSegmentId: source.trackSegmentId,
    evidence: source.evidence
  };
}

function sourceTraceFieldPath(
  source: ComparisonMetricSourceV1,
  field: string
): (string | number)[] {
  return "outcome" in source ? ["outcome", field] : [field];
}

function canonicalSourceIssues(
  source: ComparisonMetricSourceV1,
  definition: ReturnType<typeof canonicalMetricDefinition>
): ContractValidationIssue[] {
  const trace = sourceTrace(source);
  const issues: ContractValidationIssue[] = [];
  for (const [field, expected] of [
    ["modality", definition.modality],
    ["context", definition.context],
    ["unit", definition.unit],
    ["algorithmVersion", definition.algorithmVersion]
  ] as const) {
    if (trace[field] !== expected) {
      issues.push({
        path: sourceTraceFieldPath(source, field),
        message: `Comparison source ${field} must match the canonical protocol metric definition.`
      });
    }
  }
  if ("outcome" in source) {
    for (const [field, expected] of [
      ["label", definition.label],
      ["reportSection", definition.reportSection],
      ["technicalVerification", definition.technicalVerification],
      ["clinicalValidation", definition.clinicalValidation]
    ] as const) {
      if (source.outcome[field] !== expected) {
        issues.push({
          path: ["outcome", field],
          message: `Measured source ${field} must match the canonical protocol metric definition.`
        });
      }
    }
  }
  return issues;
}

function compatibleSourcePairIssues(
  reference: ComparisonMetricSourceV1,
  current: ComparisonMetricSourceV1
): ContractValidationIssue[] {
  const referenceTrace = sourceTrace(reference);
  const currentTrace = sourceTrace(current);
  const issues: ContractValidationIssue[] = [];
  for (const field of [
    "metricCode",
    "modality",
    "context",
    "unit",
    "algorithmVersion",
    "processorRef"
  ] as const) {
    if (referenceTrace[field] !== currentTrace[field]) {
      issues.push({
        path: ["current", ...sourceTraceFieldPath(current, field)],
        message: `Measured or withheld sources must have matching ${field}.`
      });
    }
  }
  for (const [name, source, trace] of [
    ["reference", reference, referenceTrace],
    ["current", current, currentTrace]
  ] as const) {
    if (source.processor.modality !== trace.modality) {
      issues.push({
        path: [name, "processor", "modality"],
        message: "Source processor modality must match the metric trace."
      });
    }
    if (
      !source.processor.assetIntegrityVerified ||
      source.processor.assetPath === null ||
      source.processor.assetSha256 === null
    ) {
      issues.push({
        path: [name, "processor", "assetIntegrityVerified"],
        message:
          "Measured or withheld sources require a verified, identified processor asset."
      });
    }
  }
  for (const field of [
    "modality",
    "processorRef",
    "runtime",
    "runtimeVersion",
    "assetPath",
    "assetSha256",
    "assetIntegrityVerified"
  ] as const) {
    if (reference.processor[field] !== current.processor[field]) {
      issues.push({
        path: ["current", "processor", field],
        message: `Measured or withheld sources must have matching processor ${field}.`
      });
    }
  }
  return issues;
}

function sameProtocolRef(
  left: { packId: string; version: string; contentSha256: string },
  right: { packId: string; version: string; contentSha256: string }
): boolean {
  return (
    left.packId === right.packId &&
    left.version === right.version &&
    left.contentSha256 === right.contentSha256
  );
}

function evidenceBindingIssues(
  source: ComparisonMetricSourceV1,
  observation: PreviousVisitObservationRefV1
): ContractValidationIssue[] {
  const trace = sourceTrace(source);
  const issues: ContractValidationIssue[] = [];
  const evidencePrefix = "outcome" in source
    ? ["outcome", "evidence", "refs"]
    : ["evidence", "refs"];
  trace.evidence.refs.forEach((ref, index) => {
    const path = [...evidencePrefix, index];
    if (
      ref.sessionId !== source.sessionId ||
      ref.observationId !== source.observationId ||
      !sameProtocolRef(ref.protocolRef, observation.protocolRef)
    ) {
      issues.push({
        path,
        message:
          "Evidence reference session, observation, and protocol must match its declared comparison source."
      });
    }
    if (ref.kind === "event") return;
    if (
      ref.modality !== trace.modality ||
      ref.context !== trace.context ||
      ref.trackSegmentId !== trace.trackSegmentId
    ) {
      issues.push({
        path,
        message:
          "Evidence reference modality, context, and track must match its metric trace."
      });
    }
    if (ref.kind === "window") return;
    if (ref.metricCode !== trace.metricCode || ref.unit !== trace.unit) {
      issues.push({
        path,
        message:
          "Measurement and aggregate evidence must match the source metric and unit."
      });
    }
    if (ref.kind === "aggregate" && ref.aggregateId !== trace.aggregateId) {
      issues.push({
        path: [...path, "aggregateId"],
        message: "Aggregate evidence must match the source aggregate."
      });
    }
  });
  return issues;
}

const ComparisonRowBaseShape = {
  metricCode: ConditionDemoMetricCodeSchema,
  nativeUnit: z.string().min(1).nullable()
} as const;

export const MeasuredPreviousVisitComparisonRowV1Schema = z
  .object({
    ...ComparisonRowBaseShape,
    status: z.literal("measured"),
    decision: z.literal("included"),
    compatibilityReasonCodes: z.tuple([]),
    nativeUnit: z.string().min(1),
    reference: MeasuredPreviousVisitMetricSourceV1Schema,
    current: MeasuredPreviousVisitMetricSourceV1Schema,
    delta: z.number().finite()
  })
  .strict()
  .superRefine((row, context) => {
    const definition = canonicalMetricDefinition(row.metricCode);
    if (row.nativeUnit !== definition.unit) {
      context.addIssue({
        code: "custom",
        path: ["nativeUnit"],
        message: "Measured row native unit must match the canonical protocol metric."
      });
    }
    const sources = [row.reference, row.current] as const;
    for (const [index, source] of sources.entries()) {
      const sourceName = index === 0 ? "reference" : "current";
      if (source.outcome.metricCode !== row.metricCode) {
        context.addIssue({
          code: "custom",
          path: [sourceName, "outcome", "metricCode"],
          message: "Comparison source metric must match the row metric."
        });
      }
      if (source.outcome.unit !== row.nativeUnit) {
        context.addIssue({
          code: "custom",
          path: [sourceName, "outcome", "unit"],
          message: "Measured comparison sources must use the row native unit."
        });
      }
      for (const issue of canonicalSourceIssues(source, definition)) {
        context.addIssue({
          code: "custom",
          path: [sourceName, ...issue.path],
          message: issue.message
        });
      }
    }
    for (const issue of compatibleSourcePairIssues(
      row.reference,
      row.current
    )) {
      context.addIssue({
        code: "custom",
        path: issue.path,
        message: issue.message
      });
    }
    if (row.delta !== row.current.outcome.value - row.reference.outcome.value) {
      context.addIssue({
        code: "custom",
        path: ["delta"],
        message: "Delta must equal current minus reference in the native unit."
      });
    }
  });

export const WithheldPreviousVisitComparisonRowV1Schema = z
  .object({
    ...ComparisonRowBaseShape,
    status: z.literal("withheld"),
    decision: z.literal("excluded"),
    compatibilityReasonCodes: z
      .array(PreviousVisitCompatibilityReasonCodeSchema)
      .min(1),
    nativeUnit: z.string().min(1),
    reference: ExcludedPreviousVisitMetricSourceV1Schema,
    current: ExcludedPreviousVisitMetricSourceV1Schema,
    delta: z.null()
  })
  .strict()
  .superRefine((row, context) => {
    const definition = canonicalMetricDefinition(row.metricCode);
    if (row.nativeUnit !== definition.unit) {
      context.addIssue({
        code: "custom",
        path: ["nativeUnit"],
        message: "Withheld row native unit must match the canonical protocol metric."
      });
    }
    const referenceWithheld = row.reference.status === "withheld";
    const currentWithheld = row.current.status === "withheld";
    if (!referenceWithheld && !currentWithheld) {
      context.addIssue({
        code: "custom",
        path: ["status"],
        message: "A withheld row requires a withheld source outcome."
      });
    }
    if (
      referenceWithheld !==
      row.compatibilityReasonCodes.includes("reference-withheld")
    ) {
      context.addIssue({
        code: "custom",
        path: ["compatibilityReasonCodes"],
        message: "Reference-withheld status must have an exact reason code."
      });
    }
    if (
      currentWithheld !==
      row.compatibilityReasonCodes.includes("current-withheld")
    ) {
      context.addIssue({
        code: "custom",
        path: ["compatibilityReasonCodes"],
        message: "Current-withheld status must have an exact reason code."
      });
    }
    for (const [name, source] of [
      ["reference", row.reference],
      ["current", row.current]
    ] as const) {
      if (source.metricCode !== row.metricCode) {
        context.addIssue({
          code: "custom",
          path: [name, "metricCode"],
          message: "Comparison source metric must match the row metric."
        });
      }
      if (source.unit !== row.nativeUnit) {
        context.addIssue({
          code: "custom",
          path: [name, "unit"],
          message: "Withheld comparison sources must use the row native unit."
        });
      }
      for (const issue of canonicalSourceIssues(source, definition)) {
        context.addIssue({
          code: "custom",
          path: [name, ...issue.path],
          message: issue.message
        });
      }
    }
    for (const issue of compatibleSourcePairIssues(
      row.reference,
      row.current
    )) {
      context.addIssue({
        code: "custom",
        path: issue.path,
        message: issue.message
      });
    }
  });

export const IncompatiblePreviousVisitComparisonRowV1Schema = z
  .object({
    ...ComparisonRowBaseShape,
    status: z.literal("incompatible"),
    decision: z.literal("excluded"),
    compatibilityReasonCodes: z
      .array(PreviousVisitCompatibilityReasonCodeSchema)
      .min(1),
    reference: ExcludedPreviousVisitMetricSourceV1Schema.nullable(),
    current: ExcludedPreviousVisitMetricSourceV1Schema.nullable(),
    delta: z.null()
  })
  .strict()
  .superRefine((row, context) => {
    const definition = canonicalMetricDefinition(row.metricCode);
    let visibleMetricDefinitionMismatch =
      row.nativeUnit !== null && row.nativeUnit !== definition.unit;
    for (const [name, source] of [
      ["reference", row.reference],
      ["current", row.current]
    ] as const) {
      if (source !== null && source.metricCode !== row.metricCode) {
        context.addIssue({
          code: "custom",
          path: [name, "metricCode"],
          message: "Comparison source metric must match the row metric."
        });
      }
      if (
        source !== null &&
        canonicalSourceIssues(source, definition).length > 0
      ) {
        visibleMetricDefinitionMismatch = true;
      }
    }
    if (
      visibleMetricDefinitionMismatch &&
      !row.compatibilityReasonCodes.includes("metric-definition-mismatch")
    ) {
      context.addIssue({
        code: "custom",
        path: ["compatibilityReasonCodes"],
        message:
          "A visible canonical metric-definition mismatch requires its stable reason code."
      });
    }
  });

export const PreviousVisitComparisonRowV1Schema = z.discriminatedUnion(
  "status",
  [
    MeasuredPreviousVisitComparisonRowV1Schema,
    WithheldPreviousVisitComparisonRowV1Schema,
    IncompatiblePreviousVisitComparisonRowV1Schema
  ]
);
export type MeasuredPreviousVisitComparisonRowV1 = z.infer<
  typeof MeasuredPreviousVisitComparisonRowV1Schema
>;
export type WithheldPreviousVisitComparisonRowV1 = z.infer<
  typeof WithheldPreviousVisitComparisonRowV1Schema
>;
export type IncompatiblePreviousVisitComparisonRowV1 = z.infer<
  typeof IncompatiblePreviousVisitComparisonRowV1Schema
>;
export type PreviousVisitComparisonRowV1 = z.infer<
  typeof PreviousVisitComparisonRowV1Schema
>;

export const PreviousVisitComparisonV1Schema = z
  .object({
    schemaVersion: z.literal("phenometric.previous-visit-comparison.v1"),
    comparisonId: IdSchema,
    generatedAt: IsoTimestampSchema,
    demoContext: ConditionDemoContextV1Schema,
    acceptedReferenceId: IdSchema,
    referenceObservation: PreviousVisitObservationRefV1Schema,
    currentObservation: PreviousVisitObservationRefV1Schema,
    rows: z.array(PreviousVisitComparisonRowV1Schema).length(6),
    analyticalRepeatability: z.literal("unknown"),
    minimumDetectableChange: z.literal("unknown"),
    claimBoundary: z.literal(PREVIOUS_VISIT_CLAIM_BOUNDARY),
    persistence: z.literal("page-memory-only")
  })
  .strict()
  .superRefine((comparison, context) => {
    const protocolRefs = [
      comparison.demoContext.sourceProtocolRef,
      comparison.referenceObservation.protocolRef,
      comparison.currentObservation.protocolRef
    ] as const;
    const embeddedCompatibilityReasons: PreviousVisitCompatibilityReasonCode[] = [];
    if (!protocolRefs.every((ref) => ref.packId === protocolRefs[0].packId)) {
      embeddedCompatibilityReasons.push("protocol-pack-id-mismatch");
    }
    if (!protocolRefs.every((ref) => ref.version === protocolRefs[0].version)) {
      embeddedCompatibilityReasons.push("protocol-version-mismatch");
    }
    if (
      !protocolRefs.every(
        (ref) => ref.contentSha256 === protocolRefs[0].contentSha256
      )
    ) {
      embeddedCompatibilityReasons.push("protocol-digest-mismatch");
    }
    if (
      comparison.referenceObservation.captureAdapter.id !==
      comparison.currentObservation.captureAdapter.id
    ) {
      embeddedCompatibilityReasons.push("capture-adapter-id-mismatch");
    }
    if (
      comparison.referenceObservation.captureAdapter.version !==
      comparison.currentObservation.captureAdapter.version
    ) {
      embeddedCompatibilityReasons.push("capture-adapter-version-mismatch");
    }
    if (
      Date.parse(comparison.referenceObservation.endedAt) >=
      Date.parse(comparison.currentObservation.startedAt)
    ) {
      embeddedCompatibilityReasons.push("reference-not-before-current");
    }
    if (
      Date.parse(comparison.generatedAt) <
      Date.parse(comparison.currentObservation.endedAt)
    ) {
      context.addIssue({
        code: "custom",
        path: ["generatedAt"],
        message: "Comparison generation cannot precede the current observation."
      });
    }

    comparison.rows.forEach((row, index) => {
      if (row.metricCode !== CONDITION_DEMO_METRIC_CODES[index]) {
        context.addIssue({
          code: "custom",
          path: ["rows", index, "metricCode"],
          message: "Comparison rows must follow the profile metric order."
        });
      }
      if (
        row.status !== "incompatible" &&
        !sameProtocolRef(
          comparison.demoContext.sourceProtocolRef,
          AMBIENT_LOCAL_PROTOCOL_REF
        )
      ) {
        context.addIssue({
          code: "custom",
          path: ["rows", index, "status"],
          message:
            "Measured and withheld condition rows require the canonical ambient protocol version and digest."
        });
      }
      for (const reason of embeddedCompatibilityReasons) {
        if (
          row.status !== "incompatible" ||
          !row.compatibilityReasonCodes.includes(reason)
        ) {
          context.addIssue({
            code: "custom",
            path: ["rows", index, "compatibilityReasonCodes"],
            message: `Embedded comparison provenance requires ${reason}.`
          });
        }
      }
      for (const [name, source, expected] of [
        ["reference", row.reference, comparison.referenceObservation],
        ["current", row.current, comparison.currentObservation]
      ] as const) {
        if (
          source !== null &&
          (source.observationId !== expected.observationId ||
            source.sessionId !== expected.sessionId)
        ) {
          context.addIssue({
            code: "custom",
            path: ["rows", index, name],
            message: "Row source must belong to the declared observation."
          });
        }
        if (source !== null) {
          for (const issue of evidenceBindingIssues(source, expected)) {
            context.addIssue({
              code: "custom",
              path: ["rows", index, name, ...issue.path],
              message: issue.message
            });
          }
        }
      }
    });
  });
export type PreviousVisitComparisonV1 = z.infer<
  typeof PreviousVisitComparisonV1Schema
>;
