import {
  finalizeAmbientMetrics,
  type FaceScreeningDiagnostics,
  type VoiceScreeningDiagnostics,
  type AmbientFaceCalibration,
  type AmbientFacialFrame,
  type AmbientMetricEvidence,
  type AmbientMetricOutcome,
  type AmbientVoiceFrame
} from "@phenometrix/ambient-core";
import {
  AMBIENT_LOCAL_PROTOCOL_PACK,
  AMBIENT_LOCAL_PROTOCOL_REF,
  ObservationV3Schema,
  WithheldReasonCodeSchema,
  createAggregateId,
  createMeasurementId,
  type ConsentRecordV1,
  type EvidenceRef,
  type EvidenceWindowV1,
  type MetricCode,
  type MetricDefinition,
  type MetricOutcomeV1,
  type ObservationV3,
  type ProcessorProvenanceV1,
  type WithheldReasonCode
} from "@phenometrix/contracts";

export interface AmbientObservationBuildInput {
  sessionId: string;
  subjectRef: string;
  consent: ConsentRecordV1;
  startedAt: string;
  endedAt: string;
  durationMs: number;
  voiceFrames: readonly AmbientVoiceFrame[];
  faceFrames: readonly AmbientFacialFrame[];
  noiseCalibrationDurationMs: number;
  faceCalibration: AmbientFaceCalibration | null;
  voiceLaneAvailable: boolean;
  faceLaneAvailable: boolean;
  voiceLaneFailureReason?: AmbientLaneFailureReason | null;
  faceLaneFailureReason?: AmbientLaneFailureReason | null;
  processors: readonly ProcessorProvenanceV1[];
}

export type AmbientLaneFailureReason = Extract<
  WithheldReasonCode,
  "modality-unavailable" | "processor-unavailable" | "asset-integrity-failed"
>;

function safeId(value: string): string {
  return value.replace(/[^A-Za-z0-9._:-]/gu, "-").slice(0, 150);
}

export interface ParsedAmbientSourceWindowRef {
  modality: "voice" | "face";
  captureEpoch: number;
  trackSegmentId: string;
  startMs: number;
  endMs: number;
}

export function parseAmbientSourceWindowRef(
  sourceRef: string,
  expectedModality: "voice" | "face",
  observationDurationMs: number
): ParsedAmbientSourceWindowRef {
  const parts = sourceRef.split(":");
  if (parts.length < 5) {
    throw new Error(`Malformed ambient source window reference: ${sourceRef}`);
  }
  const modality = parts[0];
  const captureEpoch = Number(parts[1]);
  const trackSegmentId = parts.slice(2, -2).join(":");
  const startMs = Number(parts.at(-2));
  const endMs = Number(parts.at(-1));
  if (
    (modality !== "voice" && modality !== "face") ||
    modality !== expectedModality ||
    !Number.isInteger(captureEpoch) ||
    captureEpoch < 0 ||
    trackSegmentId.length === 0 ||
    !Number.isFinite(startMs) ||
    !Number.isFinite(endMs) ||
    startMs < 0 ||
    endMs <= startMs ||
    endMs > observationDurationMs
  ) {
    throw new Error(`Invalid ambient source window reference: ${sourceRef}`);
  }
  return {
    modality,
    captureEpoch,
    trackSegmentId,
    startMs,
    endMs
  };
}

function metricDefinition(code: string): MetricDefinition {
  const definition = AMBIENT_LOCAL_PROTOCOL_PACK.metrics.find(
    (candidate) => candidate.code === code
  );
  if (!definition) throw new Error(`Unregistered ambient metric: ${code}`);
  return definition;
}

function primaryProcessor(
  outcome: AmbientMetricOutcome,
  processors: readonly ProcessorProvenanceV1[]
): string {
  if (outcome.evidence.processorRefs.length === 1) {
    return outcome.evidence.processorRefs[0];
  }
  if (outcome.evidence.processorRefs.length > 1) {
    return `${outcome.modality}-mixed-provenance`;
  }
  return processors.find((processor) => processor.modality === outcome.modality)
    ?.processorRef ?? `${outcome.modality}-processor-unavailable`;
}

function primaryTrack(outcome: AmbientMetricOutcome): string {
  if (outcome.evidence.trackSegmentIds.length === 1) {
    return outcome.evidence.trackSegmentIds[0];
  }
  return outcome.evidence.trackSegmentIds.length > 1
    ? `${outcome.modality}-mixed-provenance`
    : `${outcome.modality}-track-unavailable`;
}

function measuredProcessorFailureReason(
  outcome: AmbientMetricOutcome,
  processorRef: string,
  processors: readonly ProcessorProvenanceV1[]
): AmbientLaneFailureReason | null {
  const matches = processors.filter(
    (processor) => processor.processorRef === processorRef
  );
  if (
    matches.length !== 1 ||
    matches[0].modality !== outcome.modality
  ) return "processor-unavailable";
  const processor = matches[0];
  return processor.assetIntegrityVerified &&
    processor.assetPath !== null &&
    processor.assetSha256 !== null
    ? null
    : "asset-integrity-failed";
}

function unavailableProcessorRef(
  modality: "voice" | "face",
  processors: readonly ProcessorProvenanceV1[]
): string {
  const base = `${modality}-processor-unavailable`;
  let candidate = base;
  let suffix = 1;
  while (
    processors.some((processor) => processor.processorRef === candidate)
  ) {
    candidate = `${base}:${suffix}`;
    suffix += 1;
  }
  return candidate;
}

/**
 * The events this particular metric is built from.
 *
 * This was previously `Math.max` over the pause, speech-run, nucleus, and
 * blink counters, which conflated four unrelated counts into one number: a
 * voice pause gate could be satisfied by a syllable count, and once face
 * events joined the same maximum a facial count could satisfy a voice gate.
 * Counting only the metric's own events keeps the number honest and keeps
 * each gate answerable by its own evidence.
 */
/**
 * Prints why the session measured what it measured.
 *
 * A report full of abstentions looks identical whether the camera saw nobody or
 * saw a face that never held still long enough for a bin to qualify, and those
 * call for opposite fixes. Console only: no storage, no export, no contract
 * surface. Counts and pose percentiles, never a per-frame series.
 */
function reportCaptureDiagnostics(
  extraction: { diagnostics?: unknown; events?: unknown },
  faceFrameCount: number
): void {
  const all = extraction.diagnostics as
    | { face?: FaceScreeningDiagnostics; voice?: VoiceScreeningDiagnostics }
    | undefined;
  const diagnostics = all?.face;
  const voice = all?.voice;
  if (!diagnostics) return;
  const events = extraction.events as
    | { blinks?: readonly unknown[]; expressions?: readonly unknown[];
        pauses?: readonly unknown[]; speechRuns?: readonly unknown[] }
    | undefined;

  const pct = (part: number, whole: number) =>
    whole > 0 ? `${((part / whole) * 100).toFixed(1)}%` : "n/a";
  const lines: string[] = [
    `frames delivered to extractor: ${faceFrameCount}`,
    `frames passing every gate:     ${diagnostics.usableFrameCount} (${pct(diagnostics.usableFrameCount, diagnostics.frameCount)})`,
    `bins accepted:                 ${diagnostics.binsAccepted} of ${diagnostics.binsConsidered}`
  ];
  const pose = diagnostics.pose;
  if (pose) {
    lines.push(
      "absolute pose in degrees (limits yaw 7 / pitch 10 / roll 5):",
      `  yaw    p50 ${pose.yawP50.toFixed(1)}   p95 ${pose.yawP95.toFixed(1)}`,
      `  pitch  p50 ${pose.pitchP50.toFixed(1)}   p95 ${pose.pitchP95.toFixed(1)}`,
      `  roll   p50 ${pose.rollP50.toFixed(1)}   p95 ${pose.rollP95.toFixed(1)}`
    );
  }
  const gates = Object.entries(diagnostics.frameGateFailures)
    .sort(([, a], [, b]) => b - a);
  lines.push(
    gates.length === 0
      ? "no frame failed any gate"
      : "frames failing each gate (one frame can fail several):"
  );
  for (const [reason, count] of gates) {
    lines.push(`  ${reason.padEnd(20)} ${String(count).padStart(6)}  ${pct(count, diagnostics.frameCount)}`);
  }
  const rejections = Object.entries(diagnostics.binRejections)
    .sort(([, a], [, b]) => b - a);
  lines.push(
    rejections.length === 0
      ? "no bin was rejected"
      : "bin rejections by first failing check:"
  );
  for (const [reason, count] of rejections) {
    lines.push(`  ${reason.padEnd(20)} ${String(count).padStart(6)}`);
  }
  const curve = diagnostics.acceptanceCurve ?? [];
  if (curve.length > 0) {
    lines.push(
      "if the all-or-nothing frame gate became fractional:",
      "  threshold   bins   lost to gap   lost to span   lost to samples"
    );
    for (const point of curve) {
      lines.push(
        `  ${point.threshold.toFixed(2).padStart(8)}` +
          `${String(point.binsAccepted).padStart(7)}` +
          `${String(point.lostToGap).padStart(14)}` +
          `${String(point.lostToSpan ?? 0).padStart(15)}` +
          `${String(point.lostToSampleCount).padStart(17)}`
      );
    }
  }
  const perBin = diagnostics.bins ?? [];
  if (perBin.length > 0) {
    lines.push("per bin: usable fraction / largest gap between usable frames:");
    for (const bin of perBin) {
      lines.push(
        `  bin ${String(bin.index).padStart(3)}  ` +
          `${(bin.usableFraction * 100).toFixed(1).padStart(5)}%  ` +
          `${String(bin.usableFrameCount).padStart(4)}/${String(bin.frameCount).padEnd(4)}  ` +
          `gap ${bin.maxUsableGapMs.toFixed(0).padStart(5)} ms  ` +
          `span ${(bin.usableSpanMs ?? 0).toFixed(0).padStart(5)} ms`
      );
    }
  }
  if (voice) {
    const req = voice.requirements;
    lines.push(
      "voice lane:",
      `  frames ${voice.frameCount}, usable ${voice.usableFrameCount}, ` +
        `speech-active ${voice.speechActiveFrameCount}, periodic ${voice.periodicFrameCount}`,
      `  segments ${voice.segmentsAccepted} (needs ${req.minimumSegments})`,
      `  eligible ${(voice.eligibleDurationMs / 1000).toFixed(1)}s ` +
        `(needs ${(req.minimumEligibleSpanMs / 1000).toFixed(0)}s), ` +
        `active speech ${(voice.activeSpeechMs / 1000).toFixed(1)}s ` +
        `(needs ${(req.minimumActiveSpeechMs / 1000).toFixed(0)}s)`
    );
    const vg = Object.entries(voice.frameGateFailures).sort(([, a], [, b]) => b - a);
    if (vg.length > 0) {
      lines.push("  frames failing each voice gate:");
      for (const [reason, count] of vg) {
        lines.push(`    ${reason.padEnd(22)} ${String(count).padStart(6)}`);
      }
    }
  }
  lines.push(
    "tier-2 events:",
    `  blinks       ${events?.blinks?.length ?? 0}`,
    `  expressions  ${events?.expressions?.length ?? 0}`,
    `  pauses       ${events?.pauses?.length ?? 0}`,
    `  speech runs  ${events?.speechRuns?.length ?? 0}`
  );
  const report = `PhenoMetrix capture diagnostics\n${lines.join("\n")}`;
  // eslint-disable-next-line no-console
  console.log(report);
}

function relevantEventCount(
  code: MetricCode,
  evidence: AmbientMetricEvidence
): number {
  switch (code) {
    case "ambient.voice.pause_rate":
    case "ambient.voice.pause_duration.median":
      return evidence.pauseCount ?? 0;
    case "ambient.voice.speech_run_duration.median":
      return evidence.speechRunCount ?? 0;
    case "ambient.voice.acoustic_nucleus_rate":
      return evidence.nucleusCount ?? 0;
    case "ambient.face.blink_rate.bilateral":
      return evidence.blinkCount ?? 0;
    case "ambient.face.spontaneous_event_rate":
    case "ambient.face.spontaneous_excursion.p90":
    case "ambient.face.spontaneous_excursion_asymmetry.median":
      return evidence.expressionEventCount ?? 0;
    case "ambient.face.oculo_oral_synkinesis_index":
      return evidence.coupledExpressionEventCount ?? 0;
    default:
      return 0;
  }
}

function qualityFacts(evidence: AmbientMetricEvidence): Record<string, number> {
  const facts: Record<string, number> = {
    eligibleDurationMs: evidence.eligibleDurationMs,
    sampleCount: evidence.sampleCount,
    segmentCount: evidence.segmentCount,
    qualifyingBinCount: evidence.qualifyingBinCount
  };
  const optional: Array<[string, number | undefined]> = [
    ["activeSpeechDurationMs", evidence.activeSpeechDurationMs],
    ["pitchedDurationMs", evidence.pitchedDurationMs],
    ["pitchCoverage", evidence.pitchCoverage],
    ["timingCoverage", evidence.timingCoverage],
    ["pauseCount", evidence.pauseCount],
    ["speechRunCount", evidence.speechRunCount],
    ["nucleusCount", evidence.nucleusCount],
    ["frontalExposureMs", evidence.frontalExposureMs],
    ["blinkCount", evidence.blinkCount],
    ["expressionEventCount", evidence.expressionEventCount],
    ["coupledExpressionEventCount", evidence.coupledExpressionEventCount],
    // Worst-case gate facts. Each is re-verified against the pack threshold in
    // evidence-core; a metric measured without the fact its own pack entry
    // requires now fails provenance rather than silently passing.
    ["estimatorQuality", evidence.estimatorQuality],
    ["estimatorAgreement", evidence.estimatorAgreement],
    ["segmentSpanMs", evidence.segmentSpanMs],
    ["activeSpeechPerSegmentMs", evidence.activeSpeechPerSegmentMs],
    ["validBinsPerSegment", evidence.validBinsPerSegment],
    ["cadenceHz", evidence.cadenceHz],
    ["p95FrameGapMs", evidence.p95FrameGapMs],
    ["maximumFrameGapMs", evidence.maximumFrameGapMs],
    ["dataPerBinMs", evidence.dataPerBinMs],
    ["samplesPerBin", evidence.samplesPerBin],
    ["binSpanMs", evidence.binSpanMs],
    ["observationSpanMs", evidence.observationSpanMs]
  ];
  for (const [name, value] of optional) {
    if (value !== undefined && Number.isFinite(value)) facts[name] = value;
  }
  return facts;
}

function windowsFor(
  outcome: AmbientMetricOutcome,
  definition: MetricDefinition,
  sessionId: string,
  processorRef: string,
  trackSegmentId: string,
  measured: boolean,
  withheldReason: WithheldReasonCode | null,
  observationDurationMs: number
): EvidenceWindowV1[] {
  const observedStart = Math.max(0, outcome.evidence.observedStartMs ?? 0);
  const observedEnd = Math.max(
    observedStart + 1,
    outcome.evidence.observedEndMs ?? observedStart + 1
  );
  const sources = outcome.evidence.sourceWindowRefs.length > 0
    ? outcome.evidence.sourceWindowRefs.map((sourceRef) => ({
        sourceRef,
        ...parseAmbientSourceWindowRef(
          sourceRef,
          definition.modality,
          observationDurationMs
        )
      }))
    : [{
        sourceRef: `${outcome.modality}-unavailable`,
        startMs: observedStart,
        endMs: Math.min(observationDurationMs, observedEnd)
      }];
  return sources.map((source, index) => ({
    windowId: safeId(`window:${definition.code}:${index}:${source.sourceRef}`),
    sessionId,
    modality: definition.modality,
    context: definition.context,
    trackSegmentId,
    processorRef,
    startMs: source.startMs,
    endMs: source.endMs,
    technicalQualityScore:
      measured && outcome.status === "measured" ? outcome.technicalQualityScore : 0,
    status: measured ? "eligible" : "withheld",
    reasonCodes: measured ? [] : [withheldReason ?? "quality-threshold-failed"]
  }));
}

export function contractReason(
  outcome: AmbientMetricOutcome,
  definition: MetricDefinition,
  laneAvailable: boolean,
  laneFailureReason: AmbientLaneFailureReason = "modality-unavailable"
): WithheldReasonCode {
  if (!laneAvailable) {
    if (!definition.withheldReasonCodes.includes(laneFailureReason)) {
      throw new Error(
        `Ambient lane failure reason ${laneFailureReason} is not registered for ${definition.code}.`
      );
    }
    return laneFailureReason;
  }
  if (outcome.status === "measured") return "quality-threshold-failed";
  const reason = WithheldReasonCodeSchema.parse(outcome.reasonCode);
  if (!definition.withheldReasonCodes.includes(reason)) {
    throw new Error(
      `Ambient extractor reason ${reason} is not registered for ${definition.code}.`
    );
  }
  return reason;
}

function outcomeArtifacts(
  outcome: AmbientMetricOutcome,
  input: AmbientObservationBuildInput
): {
  outcome: MetricOutcomeV1;
  windows: EvidenceWindowV1[];
  measurement: ObservationV3["measurements"][number] | null;
} {
  const definition = metricDefinition(outcome.code);
  const attributedProcessorRef = primaryProcessor(outcome, input.processors);
  const trackSegmentId = primaryTrack(outcome);
  const exactAttribution =
    outcome.evidence.processorRefs.length === 1 &&
    outcome.evidence.trackSegmentIds.length === 1;
  const processorFailureReason =
    outcome.status === "measured" && exactAttribution
      ? measuredProcessorFailureReason(
          outcome,
          attributedProcessorRef,
          input.processors
        )
      : null;
  const projectAsMeasured =
    outcome.status === "measured" &&
    exactAttribution &&
    processorFailureReason === null;
  const globallyResolvedProcessors = input.processors.filter(
    (processor) => processor.processorRef === attributedProcessorRef
  );
  const processorRef =
    projectAsMeasured ||
    (
      globallyResolvedProcessors.length === 1 &&
      globallyResolvedProcessors[0].modality === outcome.modality
    )
      ? attributedProcessorRef
      : unavailableProcessorRef(outcome.modality, input.processors);
  const laneAvailable =
    definition.modality === "voice"
      ? input.voiceLaneAvailable
      : input.faceLaneAvailable;
  const laneFailureReason =
    definition.modality === "voice"
      ? input.voiceLaneFailureReason
      : input.faceLaneFailureReason;
  const withheldReason = projectAsMeasured
    ? null
    : processorFailureReason ??
      contractReason(
        outcome,
        definition,
        laneAvailable,
        laneFailureReason ?? "modality-unavailable"
      );
  const identity = {
    protocolPackId: AMBIENT_LOCAL_PROTOCOL_PACK.packId,
    protocolVersion: AMBIENT_LOCAL_PROTOCOL_PACK.version,
    sessionId: input.sessionId,
    metricCode: definition.code,
    context: definition.context,
    unit: definition.unit,
    algorithmVersion: definition.algorithmVersion,
    processorRef,
    trackSegmentId
  };
  const aggregateId = createAggregateId(identity);
  const windows = windowsFor(
    outcome,
    definition,
    input.sessionId,
    processorRef,
    trackSegmentId,
    projectAsMeasured,
    withheldReason,
    Math.min(300_000, Math.max(0, input.durationMs))
  );
  const observationId = safeId(`observation:${input.sessionId}`);
  const windowRefs: EvidenceRef[] = windows.map((window) => ({
    schemaVersion: "phenometric.evidence-ref.v1",
    kind: "window",
    sessionId: input.sessionId,
    observationId,
    protocolRef: AMBIENT_LOCAL_PROTOCOL_REF,
    windowId: window.windowId,
    modality: definition.modality,
    context: definition.context,
    trackSegmentId
  }));
  const aggregateRef: EvidenceRef = {
    schemaVersion: "phenometric.evidence-ref.v1",
    kind: "aggregate",
    sessionId: input.sessionId,
    observationId,
    protocolRef: AMBIENT_LOCAL_PROTOCOL_REF,
    aggregateId,
    metricCode: definition.code,
    modality: definition.modality,
    context: definition.context,
    unit: definition.unit,
    trackSegmentId
  };
  const measurementId = createMeasurementId(identity, windows[0].windowId, 0);
  const measurementRef: EvidenceRef = {
    schemaVersion: "phenometric.evidence-ref.v1",
    kind: "measurement",
    sessionId: input.sessionId,
    observationId,
    protocolRef: AMBIENT_LOCAL_PROTOCOL_REF,
    measurementId,
    metricCode: definition.code,
    modality: definition.modality,
    context: definition.context,
    unit: definition.unit,
    trackSegmentId
  };
  const evidence = {
    eligibleDurationMs: outcome.evidence.eligibleDurationMs,
    activeDurationMs: outcome.evidence.activeSpeechDurationMs ?? 0,
    segmentCount: outcome.evidence.segmentCount,
    // `windowsFor` always emits a bounded fallback window when the extractor
    // has no qualifying source windows. The count describes the ObservationV3
    // evidence that is actually referenced, not only the extractor's input
    // list, so source-binding validation remains exact for short/withheld
    // sessions as well as measured sessions.
    windowCount: windows.length,
    binCount: outcome.evidence.qualifyingBinCount,
    eventCount: relevantEventCount(definition.code, outcome.evidence),
    sampleCount: outcome.evidence.sampleCount,
    coverage: outcome.evidence.pitchCoverage ?? null,
    qualityFacts: qualityFacts(outcome.evidence),
    refs:
      projectAsMeasured
        ? [...windowRefs, measurementRef, aggregateRef]
        : [...windowRefs, aggregateRef]
  };
  const common = {
    outcomeId: outcome.identity.outcomeId,
    aggregateId,
    metricCode: definition.code,
    label: definition.label,
    modality: definition.modality,
    context: definition.context,
    unit: definition.unit,
    reportSection: definition.reportSection,
    algorithmVersion: definition.algorithmVersion,
    processorRef,
    trackSegmentId,
    technicalVerification: "automated-test" as const,
    clinicalValidation: "none" as const,
    evidence
  };
  if (projectAsMeasured && outcome.status === "measured") {
    return {
      outcome: {
        ...common,
        status: "measured",
        value: outcome.value,
        technicalQualityScore: outcome.technicalQualityScore,
        technicalDispersion: outcome.technicalDispersion
      },
      windows,
      measurement: {
        measurementId,
        ordinal: 0,
        aggregateId,
        sessionId: input.sessionId,
        metricCode: definition.code,
        label: definition.label,
        modality: definition.modality,
        context: definition.context,
        unit: definition.unit,
        value: outcome.value,
        technicalQualityScore: outcome.technicalQualityScore,
        algorithmVersion: definition.algorithmVersion,
        processorRef,
        trackSegmentId,
        sourceWindowRefs: windows.map((window) => window.windowId)
      }
    };
  }
  return {
    outcome: {
      ...common,
      status: "withheld",
      reasonCode: withheldReason ?? "quality-threshold-failed",
      detail:
        !laneAvailable || processorFailureReason !== null
          ? laneFailureDetail(
              definition.modality,
              withheldReason ?? "modality-unavailable"
            )
          : outcome.status === "measured"
            ? "Measurement was withheld because processor or track provenance was mixed or missing."
            : outcome.detail.slice(0, 240),
      technicalQualityScore: null,
      technicalDispersion: null
    },
    windows,
    measurement: null
  };
}

function laneFailureDetail(
  modality: "voice" | "face",
  reason: WithheldReasonCode
): string {
  if (reason === "asset-integrity-failed") {
    return `The required local ${modality} asset could not be loaded and verified.`;
  }
  if (reason === "processor-unavailable") {
    return `The local ${modality} processor was unavailable in this session.`;
  }
  return `The ${modality} capture device was unavailable in this session.`;
}

function ensureProcessorRefs(
  processors: readonly ProcessorProvenanceV1[],
  outcomes: readonly MetricOutcomeV1[]
): ProcessorProvenanceV1[] {
  const refCounts = new Map<string, number>();
  for (const processor of processors) {
    refCounts.set(
      processor.processorRef,
      (refCounts.get(processor.processorRef) ?? 0) + 1
    );
  }
  const byRef = new Map<string, ProcessorProvenanceV1>();
  for (const processor of processors) {
    if (refCounts.get(processor.processorRef) === 1) {
      byRef.set(processor.processorRef, processor);
    }
  }
  for (const outcome of outcomes) {
    if (outcome.status === "measured") continue;
    if (byRef.has(outcome.processorRef)) continue;
    byRef.set(outcome.processorRef, {
      modality: outcome.modality,
      processorRef: outcome.processorRef,
      runtime: "unavailable",
      runtimeVersion: "1.0.0",
      assetPath: null,
      assetSha256: null,
      assetIntegrityVerified: false
    });
  }
  return [...byRef.values()];
}

export function buildAmbientObservation(
  input: AmbientObservationBuildInput
): ObservationV3 {
  const extraction = finalizeAmbientMetrics({
    identity: {
      sessionId: input.sessionId,
      protocolVersion: AMBIENT_LOCAL_PROTOCOL_PACK.version,
      protocolContentSha256: AMBIENT_LOCAL_PROTOCOL_PACK.contentSha256,
      sessionStartedAtMs: 0
    },
    voice: {
      frames: input.voiceFrames,
      noiseCalibrationDurationMs: input.noiseCalibrationDurationMs
    },
    face: {
      frames: input.faceFrames,
      calibration: input.faceCalibration
    }
  });

  reportCaptureDiagnostics(extraction, input.faceFrames.length);
  const artifacts = extraction.outcomes.map((outcome) =>
    outcomeArtifacts(outcome, input)
  );
  const outcomes = artifacts.map((artifact) => artifact.outcome);
  const windows = artifacts.flatMap((artifact) => artifact.windows);
  const measurements = artifacts.flatMap((artifact) =>
    artifact.measurement ? [artifact.measurement] : []
  );
  const laneSummary = (
    modality: "voice" | "face",
    available: boolean,
    failureReason: AmbientLaneFailureReason | null | undefined
  ) => {
    const laneOutcomes = outcomes.filter((outcome) => outcome.modality === modality);
    const measured = laneOutcomes.filter((outcome) => outcome.status === "measured");
    return {
      state: measured.length > 0
        ? "ready" as const
        : !available
          ? "unavailable" as const
          : "withheld" as const,
      eligibleDurationMs: Math.max(0, ...laneOutcomes.map((outcome) => outcome.evidence.eligibleDurationMs)),
      technicalQualityScore:
        measured.length === 0
          ? null
          : measured.reduce((sum, outcome) => sum + outcome.technicalQualityScore, 0) /
            measured.length,
      reasonCodes: [...new Set([
        ...laneOutcomes.flatMap((outcome) =>
          outcome.status === "withheld" ? [outcome.reasonCode] : []
        ),
        ...(!available && failureReason ? [failureReason] : [])
      ])]
    };
  };
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
    observationId: safeId(`observation:${input.sessionId}`),
    sessionId: input.sessionId,
    subjectRef: input.subjectRef,
    protocolRef: AMBIENT_LOCAL_PROTOCOL_REF,
    consent: input.consent,
    source: {
      role: "local-participant",
      sourceSessionRef: input.sessionId,
      audioAttribution: "user-asserted-local-participant",
      speakerAttribution: "unverified-local-input",
      audioInput: "microphone",
      faceAttribution: "single-visible-face",
      identityVerified: false
    },
    startedAt: input.startedAt,
    endedAt: input.endedAt,
    durationMs: Math.min(300_000, Math.max(0, input.durationMs)),
    captureAdapter: { id: "browser-local-media", version: "1.1.0" },
    processors: ensureProcessorRefs(input.processors, outcomes),
    windows,
    measurements,
    metricOutcomes: outcomes,
    qualitySummary: {
      voice: laneSummary(
        "voice",
        input.voiceLaneAvailable,
        input.voiceLaneFailureReason
      ),
      face: laneSummary(
        "face",
        input.faceLaneAvailable,
        input.faceLaneFailureReason
      ),
      totalWindowCount: windows.length,
      eligibleWindowCount: windows.filter((window) => window.status === "eligible").length,
      withheldWindowCount: windows.filter((window) => window.status === "withheld").length
    }
  });
}
