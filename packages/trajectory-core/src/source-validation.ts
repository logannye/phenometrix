import {
  AMBIENT_LOCAL_PROTOCOL_PACK,
  AMBIENT_LOCAL_PROTOCOL_REF,
  CONDITION_DEMO_METRIC_CODES,
  type ConditionDemoContextV1,
  type MetricOutcomeV1,
  type ObservationV3
} from "@phenometrix/contracts";

function sameProtocol(
  left: { packId: string; version: string; contentSha256: string },
  right: { packId: string; version: string; contentSha256: string }
): boolean {
  return (
    left.packId === right.packId &&
    left.version === right.version &&
    left.contentSha256 === right.contentSha256
  );
}

function exactOutcomeDefinitionErrors(outcome: MetricOutcomeV1): string[] {
  const definition = AMBIENT_LOCAL_PROTOCOL_PACK.metrics.find(
    (candidate) => candidate.code === outcome.metricCode
  );
  if (!definition) return [`${outcome.metricCode} is not registered.`];
  const errors: string[] = [];
  for (const [field, actual, expected] of [
    ["label", outcome.label, definition.label],
    ["modality", outcome.modality, definition.modality],
    ["context", outcome.context, definition.context],
    ["unit", outcome.unit, definition.unit],
    ["report section", outcome.reportSection, definition.reportSection],
    ["algorithm version", outcome.algorithmVersion, definition.algorithmVersion],
    [
      "technical verification",
      outcome.technicalVerification,
      definition.technicalVerification
    ],
    ["clinical validation", outcome.clinicalValidation, definition.clinicalValidation]
  ] as const) {
    if (actual !== expected) {
      errors.push(`${outcome.metricCode} ${field} is not canonical.`);
    }
  }
  if (
    outcome.status === "withheld" &&
    !definition.withheldReasonCodes.includes(outcome.reasonCode)
  ) {
    errors.push(`${outcome.metricCode} withheld reason is not canonical.`);
  }
  return errors;
}

function exactEvidenceErrors(
  observation: ObservationV3,
  outcome: MetricOutcomeV1
): string[] {
  const errors: string[] = [];
  const windowRefIds = new Set<string>();
  let measurementRefCount = 0;
  let matchingAggregateRefCount = 0;

  for (const ref of outcome.evidence.refs) {
    const prefix = `${outcome.metricCode} ${ref.kind} evidence`;
    if (
      ref.sessionId !== observation.sessionId ||
      ref.observationId !== observation.observationId ||
      !sameProtocol(ref.protocolRef, observation.protocolRef)
    ) {
      errors.push(`${prefix} is bound to another observation.`);
      continue;
    }

    if (ref.kind === "window") {
      windowRefIds.add(ref.windowId);
      const windows = observation.windows.filter(
        (candidate) => candidate.windowId === ref.windowId
      );
      const window = windows.length === 1 ? windows[0] : null;
      if (
        window === null ||
        window.sessionId !== observation.sessionId ||
        window.modality !== outcome.modality ||
        window.context !== outcome.context ||
        window.processorRef !== outcome.processorRef ||
        window.trackSegmentId !== outcome.trackSegmentId ||
        ref.modality !== outcome.modality ||
        ref.context !== outcome.context ||
        ref.trackSegmentId !== outcome.trackSegmentId
      ) {
        errors.push(`${prefix} does not resolve to the exact source window.`);
      }
      continue;
    }

    if (ref.kind === "measurement") {
      measurementRefCount += 1;
      const measurements = observation.measurements.filter(
        (candidate) => candidate.measurementId === ref.measurementId
      );
      const measurement = measurements.length === 1 ? measurements[0] : null;
      if (
        outcome.status !== "measured" ||
        measurement === null ||
        measurement.aggregateId !== outcome.aggregateId ||
        measurement.sessionId !== observation.sessionId ||
        measurement.metricCode !== outcome.metricCode ||
        measurement.label !== outcome.label ||
        measurement.modality !== outcome.modality ||
        measurement.context !== outcome.context ||
        measurement.unit !== outcome.unit ||
        measurement.value !== outcome.value ||
        measurement.technicalQualityScore !== outcome.technicalQualityScore ||
        measurement.algorithmVersion !== outcome.algorithmVersion ||
        measurement.processorRef !== outcome.processorRef ||
        measurement.trackSegmentId !== outcome.trackSegmentId ||
        ref.metricCode !== outcome.metricCode ||
        ref.modality !== outcome.modality ||
        ref.context !== outcome.context ||
        ref.unit !== outcome.unit ||
        ref.trackSegmentId !== outcome.trackSegmentId
      ) {
        errors.push(`${prefix} does not resolve to the exact measurement.`);
      }
      continue;
    }

    if (ref.kind === "aggregate") {
      if (
        ref.aggregateId === outcome.aggregateId &&
        ref.metricCode === outcome.metricCode &&
        ref.modality === outcome.modality &&
        ref.context === outcome.context &&
        ref.unit === outcome.unit &&
        ref.trackSegmentId === outcome.trackSegmentId
      ) {
        matchingAggregateRefCount += 1;
      } else {
        errors.push(`${prefix} does not resolve to the exact aggregate.`);
      }
    }
  }

  if (windowRefIds.size !== outcome.evidence.windowCount) {
    errors.push(`${outcome.metricCode} evidence window count is inconsistent.`);
  }
  if (matchingAggregateRefCount !== 1) {
    errors.push(`${outcome.metricCode} requires one exact aggregate reference.`);
  }
  if (outcome.status === "measured" && measurementRefCount < 1) {
    errors.push(`${outcome.metricCode} measured outcome lacks measurement evidence.`);
  }
  if (outcome.status === "withheld" && measurementRefCount !== 0) {
    errors.push(`${outcome.metricCode} withheld outcome has measurement evidence.`);
  }

  for (const measurement of observation.measurements.filter(
    (candidate) => candidate.aggregateId === outcome.aggregateId
  )) {
    for (const sourceWindowRef of measurement.sourceWindowRefs) {
      if (!windowRefIds.has(sourceWindowRef)) {
        errors.push(
          `${outcome.metricCode} measurement cites a window outside its outcome evidence.`
        );
      }
    }
  }
  return errors;
}

/**
 * Revalidates the six condition sources against the canonical ambient pack.
 * Pairwise equality is insufficient: two equally fabricated units or
 * algorithms must not become a comparable measurement.
 *
 * Noncanonical protocol references are left to the compatibility engine so it
 * can emit stable mismatch rows. A source claiming the canonical protocol,
 * however, must conform to that protocol before any values are compared.
 */
export function conditionMetricMatchesCanonicalDefinition(
  observation: ObservationV3,
  context: ConditionDemoContextV1,
  outcome: MetricOutcomeV1
): boolean {
  if (
    !sameProtocol(observation.protocolRef, AMBIENT_LOCAL_PROTOCOL_REF) ||
    !sameProtocol(context.sourceProtocolRef, AMBIENT_LOCAL_PROTOCOL_REF)
  ) {
    return true;
  }
  return exactOutcomeDefinitionErrors(outcome).length === 0;
}

/** Rejects broken source links while leaving compatibility mismatches visible. */
export function assertConditionSourceBindings(
  observation: ObservationV3
): void {
  const errors: string[] = [];
  for (const metricCode of CONDITION_DEMO_METRIC_CODES) {
    const outcomes = observation.metricOutcomes.filter(
      (candidate) => candidate.metricCode === metricCode
    );
    if (outcomes.length === 0) continue;
    if (outcomes.length > 1) {
      errors.push(`${metricCode} resolves to more than one terminal outcome.`);
      continue;
    }
    const [outcome] = outcomes;
    errors.push(...exactEvidenceErrors(observation, outcome));
  }

  if (errors.length > 0) {
    throw new Error(
      `Condition source binding failed: ${errors.join(" ")}`
    );
  }
}
