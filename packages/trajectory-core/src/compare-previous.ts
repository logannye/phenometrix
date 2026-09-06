import {
  AcceptedReferenceV1Schema,
  ConditionDemoMetricCodeSchema,
  ConditionDemoContextV1Schema,
  ObservationV3Schema,
  PREVIOUS_VISIT_CLAIM_BOUNDARY,
  PreviousVisitComparisonV1Schema,
  type AcceptedReferenceV1,
  type ConditionDemoContextV1,
  type ExcludedPreviousVisitMetricSourceV1,
  type MeasuredPreviousVisitMetricSourceV1,
  type ObservationV3,
  type PreviousVisitComparisonRowV1,
  type PreviousVisitComparisonV1,
  type PreviousVisitObservationRefV1
} from "@phenometrix/contracts";
import {
  evaluateMetricCompatibility,
  orderCompatibilityReasons,
  previousVisitGlobalCompatibilityReasons
} from "./compatibility.js";
import type { ResolvedMetricSource } from "./compatibility.js";
import { assertConditionSourceBindings } from "./source-validation.js";

export interface ComparePreviousVisitInput {
  currentContext: ConditionDemoContextV1;
  acceptedReference: AcceptedReferenceV1;
  currentObservation: ObservationV3;
}

const FNV_64_OFFSET = 0xcbf29ce484222325n;
const FNV_64_PRIME = 0x100000001b3n;
const UINT64_MASK = 0xffffffffffffffffn;

function fnv1a64(value: string): string {
  let hash = FNV_64_OFFSET;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= BigInt(value.charCodeAt(index));
    hash = (hash * FNV_64_PRIME) & UINT64_MASK;
  }
  return hash.toString(16).padStart(16, "0");
}

/**
 * Creates a short, deterministic artifact identifier without using a clock or
 * randomness. The source identifiers remain present in the comparison itself;
 * this hash is only an artifact key, not an integrity digest.
 */
export function createPreviousVisitComparisonId(
  acceptedReference: AcceptedReferenceV1,
  currentObservation: ObservationV3,
  currentContext: ConditionDemoContextV1
): string {
  const identity = [
    acceptedReference.referenceId,
    acceptedReference.observation.observationId,
    acceptedReference.demoContext.profileRef.profileId,
    acceptedReference.demoContext.profileRef.version,
    acceptedReference.demoContext.profileRef.contentSha256,
    acceptedReference.demoContext.subjectRef,
    acceptedReference.demoContext.assertedAffectedSide.side,
    acceptedReference.demoContext.sourceProtocolRef.packId,
    acceptedReference.demoContext.sourceProtocolRef.version,
    acceptedReference.demoContext.sourceProtocolRef.contentSha256,
    currentObservation.observationId,
    currentContext.profileRef.profileId,
    currentContext.profileRef.version,
    currentContext.profileRef.contentSha256,
    currentContext.subjectRef,
    currentContext.assertedAffectedSide.side,
    currentContext.sourceProtocolRef.packId,
    currentContext.sourceProtocolRef.version,
    currentContext.sourceProtocolRef.contentSha256
  ].join("\u0000");
  return `comparison:previous-visit:${fnv1a64(identity)}`;
}

function observationRef(
  observation: ObservationV3
): PreviousVisitObservationRefV1 {
  return {
    observationId: observation.observationId,
    sessionId: observation.sessionId,
    startedAt: observation.startedAt,
    endedAt: observation.endedAt,
    protocolRef: observation.protocolRef,
    captureAdapter: observation.captureAdapter
  };
}

function measuredSource(
  source: ResolvedMetricSource
): MeasuredPreviousVisitMetricSourceV1 {
  if (source.outcome.status !== "measured") {
    throw new Error("An included comparison source must be measured.");
  }
  return {
    observationId: source.observationId,
    sessionId: source.sessionId,
    status: "measured",
    outcome: source.outcome,
    processor: source.processor
  };
}

function excludedSource(source: null): null;
function excludedSource(
  source: ResolvedMetricSource
): ExcludedPreviousVisitMetricSourceV1;
function excludedSource(
  source: ResolvedMetricSource | null
): ExcludedPreviousVisitMetricSourceV1 | null;
function excludedSource(
  source: ResolvedMetricSource | null
): ExcludedPreviousVisitMetricSourceV1 | null {
  if (source === null) return null;
  const common = {
    observationId: source.observationId,
    sessionId: source.sessionId,
    outcomeId: source.outcome.outcomeId,
    aggregateId: source.outcome.aggregateId,
    metricCode: ConditionDemoMetricCodeSchema.parse(
      source.outcome.metricCode
    ),
    modality: source.outcome.modality,
    context: source.outcome.context,
    unit: source.outcome.unit,
    algorithmVersion: source.outcome.algorithmVersion,
    processorRef: source.outcome.processorRef,
    trackSegmentId: source.outcome.trackSegmentId,
    evidence: source.outcome.evidence,
    processor: source.processor
  };
  return source.outcome.status === "measured"
    ? {
        ...common,
        status: "measured",
        technicalQualityScore: source.outcome.technicalQualityScore,
        withheldReasonCode: null
      }
    : {
        ...common,
        status: "withheld",
        technicalQualityScore: source.outcome.technicalQualityScore,
        withheldReasonCode: source.outcome.reasonCode
      };
}

function comparisonRow(
  input: ComparePreviousVisitInput,
  metricCode: ConditionDemoContextV1["allowlistedMetricCodes"][number],
  globalReasons: ReturnType<typeof previousVisitGlobalCompatibilityReasons>
): PreviousVisitComparisonRowV1 {
  const evaluation = evaluateMetricCompatibility(
    input,
    metricCode,
    globalReasons
  );

  if (evaluation.status === "incompatible") {
    return {
      metricCode,
      status: "incompatible",
      decision: "excluded",
      compatibilityReasonCodes: evaluation.reasonCodes,
      nativeUnit: evaluation.nativeUnit,
      reference: excludedSource(evaluation.reference),
      current: excludedSource(evaluation.current),
      delta: null
    };
  }

  if (
    evaluation.reference === null ||
    evaluation.current === null ||
    evaluation.nativeUnit === null
  ) {
    throw new Error(
      `Compatible ${metricCode} sources must resolve to a native unit and exact provenance.`
    );
  }

  if (evaluation.status === "withheld") {
    return {
      metricCode,
      status: "withheld",
      decision: "excluded",
      compatibilityReasonCodes: evaluation.reasonCodes,
      nativeUnit: evaluation.nativeUnit,
      reference: excludedSource(evaluation.reference),
      current: excludedSource(evaluation.current),
      delta: null
    };
  }

  if (
    evaluation.reference.outcome.status !== "measured" ||
    evaluation.current.outcome.status !== "measured"
  ) {
    throw new Error(
      `Compatible ${metricCode} sources must both be measured outcomes.`
    );
  }

  const delta =
    evaluation.current.outcome.value - evaluation.reference.outcome.value;
  if (!Number.isFinite(delta)) {
    return {
      metricCode,
      status: "incompatible",
      decision: "excluded",
      compatibilityReasonCodes: orderCompatibilityReasons([
        "non-finite-delta"
      ]),
      nativeUnit: evaluation.nativeUnit,
      reference: excludedSource(evaluation.reference),
      current: excludedSource(evaluation.current),
      delta: null
    };
  }

  return {
    metricCode,
    status: "measured",
    decision: "included",
    compatibilityReasonCodes: [],
    nativeUnit: evaluation.nativeUnit,
    reference: measuredSource(evaluation.reference),
    current: measuredSource(evaluation.current),
    delta
  };
}

/**
 * Compares a current ObservationV3 with one explicitly accepted prior
 * ObservationV3. The function is synchronous, pure, and deterministic: it
 * performs no storage, clock access, inference, interpolation, or clinical
 * interpretation.
 */
export function comparePreviousVisit(
  input: ComparePreviousVisitInput
): PreviousVisitComparisonV1 {
  const currentContext = ConditionDemoContextV1Schema.parse(
    input.currentContext
  );
  const acceptedReference = AcceptedReferenceV1Schema.parse(
    input.acceptedReference
  );
  const currentObservation = ObservationV3Schema.parse(
    input.currentObservation
  );
  const validatedInput: ComparePreviousVisitInput = {
    currentContext,
    acceptedReference,
    currentObservation
  };
  assertConditionSourceBindings(acceptedReference.observation);
  assertConditionSourceBindings(currentObservation);
  const globalReasons =
    previousVisitGlobalCompatibilityReasons(validatedInput);

  return PreviousVisitComparisonV1Schema.parse({
    schemaVersion: "phenometric.previous-visit-comparison.v1",
    comparisonId: createPreviousVisitComparisonId(
      acceptedReference,
      currentObservation,
      currentContext
    ),
    generatedAt: currentObservation.endedAt,
    demoContext: currentContext,
    acceptedReferenceId: acceptedReference.referenceId,
    referenceObservation: observationRef(acceptedReference.observation),
    currentObservation: observationRef(currentObservation),
    rows: currentContext.allowlistedMetricCodes.map((metricCode) =>
      comparisonRow(validatedInput, metricCode, globalReasons)
    ),
    analyticalRepeatability: "unknown",
    minimumDetectableChange: "unknown",
    claimBoundary: PREVIOUS_VISIT_CLAIM_BOUNDARY,
    persistence: "page-memory-only"
  });
}
