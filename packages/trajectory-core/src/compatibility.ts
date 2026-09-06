import {
  PreviousVisitCompatibilityReasonCodeSchema,
  type AcceptedReferenceV1,
  type ConditionDemoContextV1,
  type ConditionDemoMetricCode,
  type MetricOutcomeV1,
  type ObservationV3,
  type PreviousVisitCompatibilityReasonCode,
  type ProcessorProvenanceV1
} from "@phenometrix/contracts";
import { conditionMetricMatchesCanonicalDefinition } from "./source-validation.js";

export interface PreviousVisitCompatibilityInput {
  currentContext: ConditionDemoContextV1;
  acceptedReference: AcceptedReferenceV1;
  currentObservation: ObservationV3;
}

export interface MetricCompatibilityEvaluation {
  metricCode: ConditionDemoMetricCode;
  status: "compatible" | "withheld" | "incompatible";
  reasonCodes: PreviousVisitCompatibilityReasonCode[];
  nativeUnit: string | null;
  reference: ResolvedMetricSource | null;
  current: ResolvedMetricSource | null;
}

export interface ResolvedMetricSource {
  observationId: string;
  sessionId: string;
  outcome: MetricOutcomeV1;
  processor: ProcessorProvenanceV1;
}

function allEqual<T>(values: readonly T[]): boolean {
  return values.every((value) => value === values[0]);
}

function protocolReasons(
  refs: readonly {
    packId: string;
    version: string;
    contentSha256: string;
  }[]
): PreviousVisitCompatibilityReasonCode[] {
  const reasons: PreviousVisitCompatibilityReasonCode[] = [];
  if (!allEqual(refs.map((ref) => ref.packId))) {
    reasons.push("protocol-pack-id-mismatch");
  }
  if (!allEqual(refs.map((ref) => ref.version))) {
    reasons.push("protocol-version-mismatch");
  }
  if (!allEqual(refs.map((ref) => ref.contentSha256))) {
    reasons.push("protocol-digest-mismatch");
  }
  return reasons;
}

/**
 * Returns encounter-level reasons in the contract enum's stable order. These
 * reasons apply to every allowlisted metric; no row is silently dropped.
 */
export function previousVisitGlobalCompatibilityReasons(
  input: PreviousVisitCompatibilityInput
): PreviousVisitCompatibilityReasonCode[] {
  const referenceContext = input.acceptedReference.demoContext;
  const referenceObservation = input.acceptedReference.observation;
  const currentContext = input.currentContext;
  const currentObservation = input.currentObservation;
  const reasons: PreviousVisitCompatibilityReasonCode[] = [];

  if (
    !allEqual([
      referenceContext.subjectRef,
      referenceObservation.subjectRef,
      currentContext.subjectRef,
      currentObservation.subjectRef
    ])
  ) {
    reasons.push("subject-mismatch");
  }

  if (
    referenceContext.profileRef.profileId !==
    currentContext.profileRef.profileId
  ) {
    reasons.push("profile-id-mismatch");
  }
  if (
    referenceContext.profileRef.version !==
    currentContext.profileRef.version
  ) {
    reasons.push("profile-version-mismatch");
  }
  if (
    referenceContext.profileRef.contentSha256 !==
    currentContext.profileRef.contentSha256
  ) {
    reasons.push("profile-digest-mismatch");
  }
  if (
    referenceContext.assertedAffectedSide.side !==
    currentContext.assertedAffectedSide.side
  ) {
    reasons.push("asserted-side-mismatch");
  }

  reasons.push(
    ...protocolReasons([
      referenceContext.sourceProtocolRef,
      referenceObservation.protocolRef,
      currentContext.sourceProtocolRef,
      currentObservation.protocolRef
    ])
  );

  if (
    referenceObservation.captureAdapter.id !==
    currentObservation.captureAdapter.id
  ) {
    reasons.push("capture-adapter-id-mismatch");
  }
  if (
    referenceObservation.captureAdapter.version !==
    currentObservation.captureAdapter.version
  ) {
    reasons.push("capture-adapter-version-mismatch");
  }

  const referenceEndedAt = Date.parse(referenceObservation.endedAt);
  const referenceAcceptedAt = Date.parse(input.acceptedReference.acceptedAt);
  const currentStartedAt = Date.parse(currentObservation.startedAt);
  if (
    referenceEndedAt >= currentStartedAt ||
    referenceAcceptedAt >= currentStartedAt
  ) {
    reasons.push("reference-not-before-current");
  }

  return orderCompatibilityReasons(reasons);
}

interface ResolvedSource {
  source: ResolvedMetricSource | null;
  processor: ProcessorProvenanceV1 | null;
  reasonCodes: PreviousVisitCompatibilityReasonCode[];
}

function resolveSource(
  observation: ObservationV3,
  outcome: MetricOutcomeV1
): ResolvedSource {
  const processors = observation.processors.filter(
    (processor) => processor.processorRef === outcome.processorRef
  );
  if (processors.length === 0) {
    return {
      source: null,
      processor: null,
      reasonCodes: ["processor-provenance-missing"]
    };
  }
  if (processors.length !== 1) {
    return {
      source: null,
      processor: null,
      reasonCodes: ["processor-provenance-ambiguous"]
    };
  }

  const processor = processors[0];
  const source: ResolvedMetricSource = {
    observationId: observation.observationId,
    sessionId: observation.sessionId,
    outcome,
    processor
  };
  return {
    source,
    processor,
    reasonCodes:
      processor.modality === outcome.modality
        ? []
        : ["modality-mismatch"]
  };
}

function processorReasons(
  referenceOutcome: MetricOutcomeV1,
  currentOutcome: MetricOutcomeV1,
  referenceProcessor: ProcessorProvenanceV1,
  currentProcessor: ProcessorProvenanceV1
): PreviousVisitCompatibilityReasonCode[] {
  const reasons: PreviousVisitCompatibilityReasonCode[] = [];
  if (referenceOutcome.processorRef !== currentOutcome.processorRef) {
    reasons.push("processor-ref-mismatch");
  }
  if (referenceProcessor.runtime !== currentProcessor.runtime) {
    reasons.push("processor-runtime-mismatch");
  }
  if (referenceProcessor.runtimeVersion !== currentProcessor.runtimeVersion) {
    reasons.push("processor-runtime-version-mismatch");
  }
  if (referenceProcessor.assetPath !== currentProcessor.assetPath) {
    reasons.push("processor-asset-path-mismatch");
  }
  if (referenceProcessor.assetSha256 !== currentProcessor.assetSha256) {
    reasons.push("processor-asset-digest-mismatch");
  }
  if (
    !referenceProcessor.assetIntegrityVerified ||
    !currentProcessor.assetIntegrityVerified ||
    referenceProcessor.assetPath === null ||
    currentProcessor.assetPath === null ||
    referenceProcessor.assetSha256 === null ||
    currentProcessor.assetSha256 === null
  ) {
    reasons.push("processor-asset-integrity-mismatch");
  }
  return reasons;
}

/**
 * Orders and de-duplicates reason codes according to the contract enum. Stable
 * ordering makes replay byte-for-byte deterministic for identical inputs.
 */
export function orderCompatibilityReasons(
  reasons: readonly PreviousVisitCompatibilityReasonCode[]
): PreviousVisitCompatibilityReasonCode[] {
  const reasonSet = new Set(reasons);
  return PreviousVisitCompatibilityReasonCodeSchema.options.filter((reason) =>
    reasonSet.has(reason)
  );
}

/**
 * Evaluates one allowlisted outcome. Structural incompatibility wins over a
 * withheld status; otherwise a withheld outcome remains visibly withheld and
 * never produces a value or delta.
 */
export function evaluateMetricCompatibility(
  input: PreviousVisitCompatibilityInput,
  metricCode: ConditionDemoMetricCode,
  globalReasonCodes: readonly PreviousVisitCompatibilityReasonCode[] =
    previousVisitGlobalCompatibilityReasons(input)
): MetricCompatibilityEvaluation {
  const referenceObservation = input.acceptedReference.observation;
  const currentObservation = input.currentObservation;
  const referenceOutcome = referenceObservation.metricOutcomes.find(
    (outcome) => outcome.metricCode === metricCode
  );
  const currentOutcome = currentObservation.metricOutcomes.find(
    (outcome) => outcome.metricCode === metricCode
  );
  const reasons: PreviousVisitCompatibilityReasonCode[] = [
    ...globalReasonCodes
  ];

  if (!referenceOutcome || !currentOutcome) {
    reasons.push("metric-missing");
    const reference = referenceOutcome
      ? resolveSource(referenceObservation, referenceOutcome)
      : null;
    const current = currentOutcome
      ? resolveSource(currentObservation, currentOutcome)
      : null;
    if (reference) reasons.push(...reference.reasonCodes);
    if (current) reasons.push(...current.reasonCodes);
    return {
      metricCode,
      status: "incompatible",
      reasonCodes: orderCompatibilityReasons(reasons),
      nativeUnit:
        referenceOutcome?.unit === currentOutcome?.unit
          ? referenceOutcome?.unit ?? null
          : null,
      reference: reference?.source ?? null,
      current: current?.source ?? null
    };
  }

  if (referenceOutcome.context !== currentOutcome.context) {
    reasons.push("context-mismatch");
  }
  if (referenceOutcome.modality !== currentOutcome.modality) {
    reasons.push("modality-mismatch");
  }
  if (referenceOutcome.unit !== currentOutcome.unit) {
    reasons.push("unit-mismatch");
  }
  if (referenceOutcome.algorithmVersion !== currentOutcome.algorithmVersion) {
    reasons.push("algorithm-version-mismatch");
  }
  if (
    !conditionMetricMatchesCanonicalDefinition(
      referenceObservation,
      input.acceptedReference.demoContext,
      referenceOutcome
    ) ||
    !conditionMetricMatchesCanonicalDefinition(
      currentObservation,
      input.currentContext,
      currentOutcome
    )
  ) {
    reasons.push("metric-definition-mismatch");
  }

  const reference = resolveSource(referenceObservation, referenceOutcome);
  const current = resolveSource(currentObservation, currentOutcome);
  reasons.push(...reference.reasonCodes, ...current.reasonCodes);

  if (reference.processor && current.processor) {
    reasons.push(
      ...processorReasons(
        referenceOutcome,
        currentOutcome,
        reference.processor,
        current.processor
      )
    );
  }

  const orderedReasons = orderCompatibilityReasons(reasons);
  const nativeUnit =
    referenceOutcome.unit === currentOutcome.unit
      ? referenceOutcome.unit
      : null;
  if (orderedReasons.length > 0) {
    return {
      metricCode,
      status: "incompatible",
      reasonCodes: orderedReasons,
      nativeUnit,
      reference: reference.source,
      current: current.source
    };
  }

  const withheldReasons: PreviousVisitCompatibilityReasonCode[] = [];
  if (referenceOutcome.status === "withheld") {
    withheldReasons.push("reference-withheld");
  }
  if (currentOutcome.status === "withheld") {
    withheldReasons.push("current-withheld");
  }
  if (withheldReasons.length > 0) {
    return {
      metricCode,
      status: "withheld",
      reasonCodes: orderCompatibilityReasons(withheldReasons),
      nativeUnit: referenceOutcome.unit,
      reference: reference.source,
      current: current.source
    };
  }

  return {
    metricCode,
    status: "compatible",
    reasonCodes: [],
    nativeUnit: referenceOutcome.unit,
    reference: reference.source,
    current: current.source
  };
}
