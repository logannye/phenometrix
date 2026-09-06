import { describe, expect, it } from "vitest";
import {
  AMBIENT_LOCAL_PROTOCOL_PACK,
  AMBIENT_LOCAL_PROTOCOL_REF,
  createAggregateId,
  createMeasurementId,
  type MetricOutcomeV1,
  type ObservationV3
} from "@phenometrix/contracts";
import {
  ALGORITHM_PARAMETER_REQUIREMENT_KEYS,
  buildPostEncounterReport,
  evidenceFactFor,
  validateObservationProvenance
} from "./report.js";

function createObservation(): ObservationV3 {
  const windows = [
    {
      windowId: "voice-unavailable-window",
      sessionId: "session-report",
      modality: "voice" as const,
      context: "ambient-speech-turn" as const,
      trackSegmentId: "voice-track-1",
      processorRef: "voice-dsp@1.0.0",
      startMs: 0,
      endMs: 1,
      technicalQualityScore: 0,
      status: "withheld" as const,
      reasonCodes: ["modality-unavailable"]
    },
    {
      windowId: "face-unavailable-window",
      sessionId: "session-report",
      modality: "face" as const,
      context: "ambient-frontal" as const,
      trackSegmentId: "face-track-1",
      processorRef: "face-geometry@1.0.0",
      startMs: 0,
      endMs: 1,
      technicalQualityScore: 0,
      status: "withheld" as const,
      reasonCodes: ["modality-unavailable"]
    }
  ];
  const metricOutcomes: MetricOutcomeV1[] =
    AMBIENT_LOCAL_PROTOCOL_PACK.metrics.map((definition) => {
      const trackSegmentId =
        definition.modality === "voice" ? "voice-track-1" : "face-track-1";
      const windowId =
        definition.modality === "voice"
          ? "voice-unavailable-window"
          : "face-unavailable-window";
      const processorRef =
        definition.modality === "voice"
          ? "voice-dsp@1.0.0"
          : "face-geometry@1.0.0";
      const aggregateId = createAggregateId({
        protocolPackId: AMBIENT_LOCAL_PROTOCOL_REF.packId,
        protocolVersion: AMBIENT_LOCAL_PROTOCOL_REF.version,
        sessionId: "session-report",
        metricCode: definition.code,
        context: definition.context,
        unit: definition.unit,
        algorithmVersion: definition.algorithmVersion,
        processorRef,
        trackSegmentId
      });
      return {
        outcomeId: `outcome-${definition.code}`,
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
        technicalVerification: "automated-test",
        clinicalValidation: "none",
        status: "withheld",
        reasonCode: "modality-unavailable",
        detail: "The modality was not available during this session.",
        technicalQualityScore: null,
        technicalDispersion: null,
        evidence: {
          eligibleDurationMs: 0,
          activeDurationMs: 0,
          segmentCount: 0,
          windowCount: 0,
          binCount: 0,
          eventCount: 0,
          sampleCount: 0,
          coverage: null,
          qualityFacts: {},
          refs: [
            {
              schemaVersion: "phenometric.evidence-ref.v1",
              kind: "window",
              sessionId: "session-report",
              observationId: "observation-report",
              protocolRef: AMBIENT_LOCAL_PROTOCOL_REF,
              windowId,
              modality: definition.modality,
              context: definition.context,
              trackSegmentId
            }
          ]
        }
      };
    });

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
    observationId: "observation-report",
    sessionId: "session-report",
    subjectRef: "subject-session-report",
    protocolRef: AMBIENT_LOCAL_PROTOCOL_REF,
    consent: {
      schemaVersion: "phenometric.consent-record.v1",
      consentId: "consent-report",
      sessionId: "session-report",
      documentVersion: "ambient-local-consent.v1",
      documentSha256:
        AMBIENT_LOCAL_PROTOCOL_PACK.consentDocument.contentSha256,
      recordedAt: "2026-07-20T16:00:00.000Z",
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
      sourceSessionRef: "session-report",
      audioAttribution: "user-asserted-local-participant",
      speakerAttribution: "unverified-local-input",
      audioInput: "microphone",
      faceAttribution: "single-visible-face",
      identityVerified: false
    },
    startedAt: "2026-07-20T16:00:00.000Z",
    endedAt: "2026-07-20T16:00:01.000Z",
    durationMs: 1_000,
    captureAdapter: { id: "browser", version: "1.0.0" },
    processors: [
      {
        modality: "voice",
        processorRef: "voice-dsp@1.0.0",
        runtime: "audio-worklet-worker",
        runtimeVersion: "1.0.0",
        assetPath: "assets/voice-capture-worklet.js",
        assetSha256: "1".repeat(64),
        assetIntegrityVerified: true
      },
      {
        modality: "face",
        processorRef: "face-geometry@1.0.0",
        runtime: "mediapipe-tasks-vision",
        runtimeVersion: "0.10.35",
        assetPath: "assets/face_landmarker.task",
        assetSha256: "0".repeat(64),
        assetIntegrityVerified: true
      }
    ],
    windows,
    measurements: [],
    metricOutcomes,
    qualitySummary: {
      voice: {
        state: "unavailable",
        eligibleDurationMs: 0,
        technicalQualityScore: null,
        reasonCodes: ["modality-unavailable"]
      },
      face: {
        state: "unavailable",
        eligibleDurationMs: 0,
        technicalQualityScore: null,
        reasonCodes: ["modality-unavailable"]
      },
      totalWindowCount: 2,
      eligibleWindowCount: 0,
      withheldWindowCount: 2
    }
  };
}

function createQualifiedZeroPauseRateObservation(): ObservationV3 {
  const observation = createObservation();
  observation.endedAt = "2026-07-20T16:00:30.000Z";
  observation.durationMs = 30_000;
  observation.windows[0] = {
    ...observation.windows[0],
    endMs: 10_000,
    technicalQualityScore: 0.95,
    status: "eligible",
    reasonCodes: []
  };
  const voiceWindowIds = [
    observation.windows[0].windowId,
    "voice-eligible-window-2",
    "voice-eligible-window-3"
  ];
  observation.windows.push(
    {
      ...observation.windows[0],
      windowId: voiceWindowIds[1],
      startMs: 10_000,
      endMs: 20_000
    },
    {
      ...observation.windows[0],
      windowId: voiceWindowIds[2],
      startMs: 20_000,
      endMs: 30_000
    }
  );
  observation.qualitySummary.voice = {
    state: "ready",
    eligibleDurationMs: 30_000,
    technicalQualityScore: 0.95,
    reasonCodes: ["modality-unavailable"]
  };
  observation.qualitySummary.totalWindowCount = 4;
  observation.qualitySummary.eligibleWindowCount = 3;
  observation.qualitySummary.withheldWindowCount = 1;

  const outcomeIndex = observation.metricOutcomes.findIndex(
    (outcome) => outcome.metricCode === "ambient.voice.pause_rate"
  );
  const withheld = observation.metricOutcomes[outcomeIndex];
  const identity = {
    protocolPackId: observation.protocolRef.packId,
    protocolVersion: observation.protocolRef.version,
    sessionId: observation.sessionId,
    metricCode: withheld.metricCode,
    context: withheld.context,
    unit: withheld.unit,
    algorithmVersion: withheld.algorithmVersion,
    processorRef: withheld.processorRef,
    trackSegmentId: withheld.trackSegmentId
  };
  const measurementId = createMeasurementId(
    identity,
    "voice-unavailable-window",
    0
  );
  observation.measurements.push({
    measurementId,
    aggregateId: withheld.aggregateId,
    sessionId: observation.sessionId,
    metricCode: withheld.metricCode,
    label: withheld.label,
    modality: withheld.modality,
    context: withheld.context,
    unit: withheld.unit,
    value: 0,
    technicalQualityScore: 0.95,
    algorithmVersion: withheld.algorithmVersion,
    processorRef: withheld.processorRef,
    trackSegmentId: withheld.trackSegmentId,
    ordinal: 0,
    sourceWindowRefs: voiceWindowIds
  });
  observation.metricOutcomes[outcomeIndex] = {
    outcomeId: withheld.outcomeId,
    aggregateId: withheld.aggregateId,
    metricCode: withheld.metricCode,
    label: withheld.label,
    modality: withheld.modality,
    context: withheld.context,
    unit: withheld.unit,
    reportSection: withheld.reportSection,
    algorithmVersion: withheld.algorithmVersion,
    processorRef: withheld.processorRef,
    trackSegmentId: withheld.trackSegmentId,
    technicalVerification: "automated-test",
    clinicalValidation: "none",
    status: "measured",
    value: 0,
    technicalQualityScore: 0.95,
    technicalDispersion: 0,
    evidence: {
      eligibleDurationMs: 30_000,
      activeDurationMs: 15_000,
      segmentCount: 3,
      windowCount: 3,
      binCount: 0,
      eventCount: 0,
      sampleCount: 3_000,
      coverage: 0.9,
      qualityFacts: {
        timingCoverage: 0.9,
        activeSpeechDurationMs: 15_000,
        segmentCount: 3,
        pauseCount: 0,
        // Named for the statistic, not the gate: these are the worst observed
        // per-segment values, which the pack's `minimum*` thresholds are then
        // checked against.
        segmentSpanMs: 2_000,
        activeSpeechPerSegmentMs: 1_000
      },
      refs: [
        ...voiceWindowIds.map((windowId) => ({
          schemaVersion: "phenometric.evidence-ref.v1" as const,
          kind: "window" as const,
          sessionId: observation.sessionId,
          observationId: observation.observationId,
          protocolRef: observation.protocolRef,
          windowId,
          modality: withheld.modality,
          context: withheld.context,
          trackSegmentId: withheld.trackSegmentId
        })),
        {
          schemaVersion: "phenometric.evidence-ref.v1",
          kind: "measurement",
          sessionId: observation.sessionId,
          observationId: observation.observationId,
          protocolRef: observation.protocolRef,
          measurementId,
          metricCode: withheld.metricCode,
          modality: withheld.modality,
          context: withheld.context,
          unit: withheld.unit,
          trackSegmentId: withheld.trackSegmentId
        }
      ]
    }
  };
  return observation;
}

describe("post-encounter report", () => {
  it("rejects a ready lane summary when every lane outcome is withheld", () => {
    const observation = createObservation();
    observation.qualitySummary.voice = {
      state: "ready",
      eligibleDurationMs: 300_000,
      technicalQualityScore: 1,
      reasonCodes: []
    };

    const result = validateObservationProvenance(
      observation,
      AMBIENT_LOCAL_PROTOCOL_PACK
    );

    expect(result.status).toBe("fail");
    expect(result.errors).toEqual(expect.arrayContaining([
      "voice quality-summary eligible duration does not match its outcomes.",
      "voice quality-summary cannot be ready without a measured outcome.",
      "voice quality-summary score must be null without a measured outcome.",
      "voice quality-summary omits withheld reason modality-unavailable."
    ]));
  });

  it("rejects eligible-duration claims unsupported by resolved windows", () => {
    const observation = createQualifiedZeroPauseRateObservation();
    observation.windows[0].endMs = 1;

    const result = validateObservationProvenance(
      observation,
      AMBIENT_LOCAL_PROTOCOL_PACK
    );
    expect(result.status).toBe("fail");
    expect(result.errors).toContain(
      "ambient.voice.pause_rate claimed eligible duration is not supported by its resolved windows."
    );
  });

  it("does not let overlapping windows inflate eligible-duration support", () => {
    const observation = createQualifiedZeroPauseRateObservation();
    observation.endedAt = "2026-07-20T16:01:30.000Z";
    observation.durationMs = 90_000;
    observation.windows[0].endMs = 10_000;

    const pauseRate = observation.metricOutcomes.find(
      (outcome) => outcome.metricCode === "ambient.voice.pause_rate"
    );
    if (!pauseRate || pauseRate.status !== "measured") {
      throw new Error("Missing measured pause-rate outcome.");
    }
    const windowRef = pauseRate.evidence.refs.find(
      (ref) => ref.kind === "window"
    );
    if (!windowRef || windowRef.kind !== "window") {
      throw new Error("Missing pause-rate window evidence.");
    }

    const overlappingWindowIds = [observation.windows[0].windowId];
    observation.windows = [observation.windows[0], observation.windows[1]];
    for (let ordinal = 2; ordinal <= 9; ordinal += 1) {
      const windowId = `voice-overlap-${ordinal}`;
      overlappingWindowIds.push(windowId);
      observation.windows.push({
        ...observation.windows[0],
        windowId
      });
    }
    pauseRate.evidence.refs = [
      ...overlappingWindowIds.map((windowId) => ({
        ...windowRef,
        windowId
      })),
      ...pauseRate.evidence.refs.filter((ref) => ref.kind !== "window")
    ];
    pauseRate.evidence.eligibleDurationMs = 90_000;
    pauseRate.evidence.segmentCount = 9;
    pauseRate.evidence.windowCount = 9;
    observation.measurements[0].sourceWindowRefs = overlappingWindowIds;
    observation.qualitySummary.voice.eligibleDurationMs = 90_000;
    observation.qualitySummary.totalWindowCount = 10;
    observation.qualitySummary.eligibleWindowCount = 9;
    observation.qualitySummary.withheldWindowCount = 1;

    const result = validateObservationProvenance(
      observation,
      AMBIENT_LOCAL_PROTOCOL_PACK
    );
    expect(result.status).toBe("fail");
    expect(result.errors).toContain(
      "ambient.voice.pause_rate claimed eligible duration is not supported by its resolved windows."
    );
  });

  it("does not manufacture duration from a frame-gap allowance per tiny window", () => {
    const observation = createQualifiedZeroPauseRateObservation();
    const pauseRate = observation.metricOutcomes.find(
      (outcome) => outcome.metricCode === "ambient.voice.pause_rate"
    );
    if (!pauseRate || pauseRate.status !== "measured") {
      throw new Error("Missing measured pause-rate outcome.");
    }
    const firstWindow = observation.windows[0];
    const withheldFaceWindow = observation.windows[1];
    const firstWindowRef = pauseRate.evidence.refs.find(
      (ref) => ref.kind === "window"
    );
    if (!firstWindowRef || firstWindowRef.kind !== "window") {
      throw new Error("Missing pause-rate window evidence.");
    }

    const tinyWindows = Array.from({ length: 750 }, (_, ordinal) => ({
      ...firstWindow,
      windowId:
        ordinal === 0 ? firstWindow.windowId : `voice-tiny-${ordinal}`,
      startMs: ordinal * 2,
      endMs: ordinal * 2 + 1
    }));
    const tinyWindowRefs = tinyWindows.map((window) => ({
      ...firstWindowRef,
      windowId: window.windowId
    }));
    pauseRate.evidence.refs = [
      ...tinyWindowRefs,
      ...pauseRate.evidence.refs.filter((ref) => ref.kind !== "window")
    ];
    pauseRate.evidence.windowCount = tinyWindows.length;
    observation.measurements[0].sourceWindowRefs = tinyWindows.map(
      (window) => window.windowId
    );
    observation.windows = [...tinyWindows, withheldFaceWindow];
    observation.qualitySummary.totalWindowCount = 751;
    observation.qualitySummary.eligibleWindowCount = 750;
    observation.qualitySummary.withheldWindowCount = 1;

    const result = validateObservationProvenance(
      observation,
      AMBIENT_LOCAL_PROTOCOL_PACK
    );

    expect(result.status).toBe("fail");
    expect(result.errors).toContain(
      "ambient.voice.pause_rate claimed eligible duration is not supported by its resolved windows."
    );
  });

  it("derives a face observation-span gate from resolved windows", () => {
    const observation = createObservation();
    observation.endedAt = "2026-07-20T16:00:30.000Z";
    observation.durationMs = 30_000;
    const outcomeIndex = observation.metricOutcomes.findIndex(
      (outcome) => outcome.metricCode === "ambient.face.brow_height.left"
    );
    const withheld = observation.metricOutcomes[outcomeIndex];
    const baseWindow = observation.windows[1];
    const baseRef = withheld.evidence.refs.find(
      (ref) => ref.kind === "window"
    );
    if (!baseRef || baseRef.kind !== "window") {
      throw new Error("Missing face window evidence.");
    }
    const faceWindows = Array.from({ length: 3 }, (_, ordinal) => ({
      ...baseWindow,
      windowId:
        ordinal === 0 ? baseWindow.windowId : `face-clustered-${ordinal}`,
      startMs: ordinal * 5_000,
      endMs: (ordinal + 1) * 5_000,
      technicalQualityScore: 0.95,
      status: "eligible" as const,
      reasonCodes: []
    }));
    observation.windows = [observation.windows[0], ...faceWindows];

    const identity = {
      protocolPackId: observation.protocolRef.packId,
      protocolVersion: observation.protocolRef.version,
      sessionId: observation.sessionId,
      metricCode: withheld.metricCode,
      context: withheld.context,
      unit: withheld.unit,
      algorithmVersion: withheld.algorithmVersion,
      processorRef: withheld.processorRef,
      trackSegmentId: withheld.trackSegmentId
    };
    const measurementId = createMeasurementId(
      identity,
      faceWindows[0].windowId,
      0
    );
    observation.measurements.push({
      measurementId,
      ordinal: 0,
      aggregateId: withheld.aggregateId,
      sessionId: observation.sessionId,
      metricCode: withheld.metricCode,
      label: withheld.label,
      modality: withheld.modality,
      context: withheld.context,
      unit: withheld.unit,
      value: 0.5,
      technicalQualityScore: 0.95,
      algorithmVersion: withheld.algorithmVersion,
      processorRef: withheld.processorRef,
      trackSegmentId: withheld.trackSegmentId,
      sourceWindowRefs: faceWindows.map((window) => window.windowId)
    });
    observation.metricOutcomes[outcomeIndex] = {
      outcomeId: withheld.outcomeId,
      aggregateId: withheld.aggregateId,
      metricCode: withheld.metricCode,
      label: withheld.label,
      modality: withheld.modality,
      context: withheld.context,
      unit: withheld.unit,
      reportSection: withheld.reportSection,
      algorithmVersion: withheld.algorithmVersion,
      processorRef: withheld.processorRef,
      trackSegmentId: withheld.trackSegmentId,
      technicalVerification: withheld.technicalVerification,
      clinicalValidation: withheld.clinicalValidation,
      status: "measured",
      value: 0.5,
      technicalQualityScore: 0.95,
      technicalDispersion: 0,
      evidence: {
        eligibleDurationMs: 15_000,
        activeDurationMs: 0,
        segmentCount: 1,
        windowCount: 3,
        binCount: 3,
        eventCount: 0,
        sampleCount: 450,
        coverage: null,
        qualityFacts: { observationSpanMs: 30_000 },
        refs: [
          ...faceWindows.map((window) => ({
            ...baseRef,
            windowId: window.windowId
          })),
          {
            schemaVersion: "phenometric.evidence-ref.v1" as const,
            kind: "measurement" as const,
            sessionId: observation.sessionId,
            observationId: observation.observationId,
            protocolRef: observation.protocolRef,
            measurementId,
            metricCode: withheld.metricCode,
            modality: withheld.modality,
            context: withheld.context,
            unit: withheld.unit,
            trackSegmentId: withheld.trackSegmentId
          }
        ]
      }
    };
    observation.qualitySummary.face = {
      state: "ready",
      eligibleDurationMs: 15_000,
      technicalQualityScore: 0.95,
      reasonCodes: []
    };
    observation.qualitySummary.totalWindowCount = 4;
    observation.qualitySummary.eligibleWindowCount = 3;
    observation.qualitySummary.withheldWindowCount = 1;

    const result = validateObservationProvenance(
      observation,
      AMBIENT_LOCAL_PROTOCOL_PACK
    );

    expect(result.status).toBe("fail");
    expect(result.errors).toContain(
      "ambient.face.brow_height.left claimed observation span is not supported by its resolved windows."
    );
    expect(result.errors).toContain(
      "ambient.face.brow_height.left resolved windows do not satisfy minimumObservationSpanMs."
    );
  });

  it("derives a blink exposure gate from resolved eligible duration", () => {
    const observation = createObservation();
    observation.endedAt = "2026-07-20T16:01:00.000Z";
    observation.durationMs = 60_000;
    const faceWindow = observation.windows[1];
    faceWindow.endMs = 15_000;
    faceWindow.technicalQualityScore = 0.95;
    faceWindow.status = "eligible";
    faceWindow.reasonCodes = [];

    const outcomeIndex = observation.metricOutcomes.findIndex(
      (outcome) => outcome.metricCode === "ambient.face.blink_rate.bilateral"
    );
    const withheld = observation.metricOutcomes[outcomeIndex];
    const identity = {
      protocolPackId: observation.protocolRef.packId,
      protocolVersion: observation.protocolRef.version,
      sessionId: observation.sessionId,
      metricCode: withheld.metricCode,
      context: withheld.context,
      unit: withheld.unit,
      algorithmVersion: withheld.algorithmVersion,
      processorRef: withheld.processorRef,
      trackSegmentId: withheld.trackSegmentId
    };
    const measurementId = createMeasurementId(
      identity,
      faceWindow.windowId,
      0
    );
    observation.measurements.push({
      measurementId,
      ordinal: 0,
      aggregateId: withheld.aggregateId,
      sessionId: observation.sessionId,
      metricCode: withheld.metricCode,
      label: withheld.label,
      modality: withheld.modality,
      context: withheld.context,
      unit: withheld.unit,
      value: 0,
      technicalQualityScore: 0.95,
      algorithmVersion: withheld.algorithmVersion,
      processorRef: withheld.processorRef,
      trackSegmentId: withheld.trackSegmentId,
      sourceWindowRefs: [faceWindow.windowId]
    });
    const windowRef = withheld.evidence.refs.find(
      (ref) => ref.kind === "window"
    );
    if (!windowRef || windowRef.kind !== "window") {
      throw new Error("Missing blink window evidence.");
    }
    observation.metricOutcomes[outcomeIndex] = {
      outcomeId: withheld.outcomeId,
      aggregateId: withheld.aggregateId,
      metricCode: withheld.metricCode,
      label: withheld.label,
      modality: withheld.modality,
      context: withheld.context,
      unit: withheld.unit,
      reportSection: withheld.reportSection,
      algorithmVersion: withheld.algorithmVersion,
      processorRef: withheld.processorRef,
      trackSegmentId: withheld.trackSegmentId,
      technicalVerification: withheld.technicalVerification,
      clinicalValidation: withheld.clinicalValidation,
      status: "measured",
      value: 0,
      technicalQualityScore: 0.95,
      technicalDispersion: 0,
      evidence: {
        eligibleDurationMs: 15_000,
        activeDurationMs: 0,
        segmentCount: 1,
        windowCount: 1,
        binCount: 3,
        eventCount: 0,
        sampleCount: 450,
        coverage: null,
        qualityFacts: {
          frontalExposureMs: 60_000,
          blinkCount: 1,
          cadenceHz: 30,
          p95FrameGapMs: 50
        },
        refs: [
          windowRef,
          {
            schemaVersion: "phenometric.evidence-ref.v1",
            kind: "measurement",
            sessionId: observation.sessionId,
            observationId: observation.observationId,
            protocolRef: observation.protocolRef,
            measurementId,
            metricCode: withheld.metricCode,
            modality: withheld.modality,
            context: withheld.context,
            unit: withheld.unit,
            trackSegmentId: withheld.trackSegmentId
          }
        ]
      }
    };
    observation.qualitySummary.face = {
      state: "ready",
      eligibleDurationMs: 15_000,
      technicalQualityScore: 0.95,
      reasonCodes: []
    };
    observation.qualitySummary.eligibleWindowCount = 1;
    observation.qualitySummary.withheldWindowCount = 1;

    const result = validateObservationProvenance(
      observation,
      AMBIENT_LOCAL_PROTOCOL_PACK
    );

    expect(result.status).toBe("fail");
    expect(result.errors).toContain(
      "ambient.face.blink_rate.bilateral claimed exposure is not supported by its resolved windows."
    );
    expect(result.errors).toContain(
      "ambient.face.blink_rate.bilateral resolved windows do not satisfy minimumExposureMs."
    );
    expect(result.errors).toContain(
      "ambient.face.blink_rate.bilateral face bin count does not match its evidence windows."
    );
    expect(result.errors).toContain(
      "ambient.face.blink_rate.bilateral frontal exposure does not match eligible duration."
    );
    expect(result.errors).toContain(
      "ambient.face.blink_rate.bilateral value does not match its blink count and frontal exposure."
    );
  });

  it("projects all registered metrics exactly once without narrative", () => {
    const observation = createObservation();
    const report = buildPostEncounterReport(
      observation,
      AMBIENT_LOCAL_PROTOCOL_PACK,
      { generatedAt: "2026-07-20T16:00:02.000Z" }
    );
    const outcomes = report.sections.flatMap((section) => section.outcomes);

    expect(report.sections).toHaveLength(10);
    expect(outcomes).toHaveLength(27);
    expect(new Set(outcomes.map((outcome) => outcome.metricCode)).size).toBe(
      27
    );
    expect(outcomes.every((outcome) => outcome.status === "withheld")).toBe(
      true
    );
    expect(report).not.toHaveProperty("headline");
    expect(report).not.toHaveProperty("summary");
    expect(report.exportAvailable).toBe(false);
  });

  it("rejects evidence that only matches the metric code", () => {
    const observation = createObservation();
    const first = observation.metricOutcomes[0];
    const ref = first.evidence.refs[0];
    if (ref.kind !== "window") throw new Error("Expected window evidence.");
    ref.context = "ambient-frontal";

    const result = validateObservationProvenance(
      observation,
      AMBIENT_LOCAL_PROTOCOL_PACK
    );
    expect(result.status).toBe("fail");
    expect(result.errors.join(" ")).toMatch(/metadata|registry/);
  });

  it("rejects missing and duplicate terminal metric outcomes", () => {
    const observation = createObservation();
    observation.metricOutcomes.pop();
    expect(
      validateObservationProvenance(
        observation,
        AMBIENT_LOCAL_PROTOCOL_PACK
      ).status
    ).toBe("fail");

    const duplicated = createObservation();
    duplicated.metricOutcomes[1] = structuredClone(
      duplicated.metricOutcomes[0]
    );
    expect(
      validateObservationProvenance(
        duplicated,
        AMBIENT_LOCAL_PROTOCOL_PACK
      ).status
    ).toBe("fail");
  });

  it("accepts a qualified zero but rejects the same value without its denominator", () => {
    const qualified = createQualifiedZeroPauseRateObservation();
    expect(
      validateObservationProvenance(
        qualified,
        AMBIENT_LOCAL_PROTOCOL_PACK
      ).status
    ).toBe("pass");
    const report = buildPostEncounterReport(
      qualified,
      AMBIENT_LOCAL_PROTOCOL_PACK,
      { generatedAt: "2026-07-20T16:00:31.000Z" }
    );
    expect(
      report.sections
        .flatMap((section) => section.outcomes)
        .find((outcome) => outcome.metricCode === "ambient.voice.pause_rate")
    ).toMatchObject({ status: "measured", value: 0 });

    const insufficient = createQualifiedZeroPauseRateObservation();
    const pauseRate = insufficient.metricOutcomes.find(
      (outcome) => outcome.metricCode === "ambient.voice.pause_rate"
    );
    if (!pauseRate) throw new Error("Missing pause-rate outcome.");
    pauseRate.evidence.eligibleDurationMs = 29_999;
    const validation = validateObservationProvenance(
      insufficient,
      AMBIENT_LOCAL_PROTOCOL_PACK
    );
    expect(validation.status).toBe("fail");
    expect(validation.errors.join(" ")).toMatch(/minimumEligibleSpanMs/);
  });

  it("requires exact verified processor provenance for a measured outcome", () => {
    const unverified = createQualifiedZeroPauseRateObservation();
    unverified.processors[0].assetIntegrityVerified = false;
    unverified.processors[0].assetPath = null;
    unverified.processors[0].assetSha256 = null;
    expect(
      validateObservationProvenance(
        unverified,
        AMBIENT_LOCAL_PROTOCOL_PACK
      ).errors
    ).toContain(
      "ambient.voice.pause_rate measured processor lacks verified asset provenance."
    );

    const duplicate = createQualifiedZeroPauseRateObservation();
    duplicate.processors.push({ ...duplicate.processors[0] });
    expect(
      validateObservationProvenance(
        duplicate,
        AMBIENT_LOCAL_PROTOCOL_PACK
      ).errors
    ).toContain(
      "ambient.voice.pause_rate measured processor reference must resolve exactly once."
    );

    const wrongModality = createQualifiedZeroPauseRateObservation();
    wrongModality.processors[0].modality = "face";
    expect(
      validateObservationProvenance(
        wrongModality,
        AMBIENT_LOCAL_PROTOCOL_PACK
      ).errors
    ).toContain(
      "ambient.voice.pause_rate measured processor modality does not match the outcome."
    );
  });

  it("rejects impossible measured duration, segment, coverage, and rate facts", () => {
    const observation = createQualifiedZeroPauseRateObservation();
    const pauseRate = observation.metricOutcomes.find(
      (outcome) => outcome.metricCode === "ambient.voice.pause_rate"
    );
    if (!pauseRate || pauseRate.status !== "measured") {
      throw new Error("Missing measured pause-rate outcome.");
    }
    pauseRate.evidence.activeDurationMs = 31_000;
    pauseRate.evidence.segmentCount = 2;
    pauseRate.evidence.qualityFacts.pitchedDurationMs = 32_000;
    pauseRate.evidence.qualityFacts.pitchCoverage = 0.25;
    pauseRate.evidence.coverage = 0.5;
    pauseRate.evidence.qualityFacts.pauseCount = 1;
    pauseRate.value = 0;
    observation.measurements[0].value = 0;

    const validation = validateObservationProvenance(
      observation,
      AMBIENT_LOCAL_PROTOCOL_PACK
    );

    expect(validation.errors).toEqual(
      expect.arrayContaining([
        "ambient.voice.pause_rate active duration exceeds eligible duration.",
        "ambient.voice.pause_rate voice segment count does not match its evidence windows.",
        "ambient.voice.pause_rate pitched duration exceeds active-speech duration.",
        "ambient.voice.pause_rate pitch coverage does not match its duration denominator.",
        "ambient.voice.pause_rate pitch coverage fields disagree.",
        "ambient.voice.pause_rate pauseCount does not match its event count.",
        "ambient.voice.pause_rate value does not match its pause count and eligible duration."
      ])
    );
  });

  it("rejects fractional event counters even when they reproduce the claimed rate", () => {
    const observation = createQualifiedZeroPauseRateObservation();
    const pauseRate = observation.metricOutcomes.find(
      (outcome) => outcome.metricCode === "ambient.voice.pause_rate"
    );
    if (!pauseRate || pauseRate.status !== "measured") {
      throw new Error("Missing measured pause-rate outcome.");
    }
    pauseRate.evidence.qualityFacts.pauseCount = 0.5;
    pauseRate.value = 1;
    observation.measurements[0].value = 1;

    const validation = validateObservationProvenance(
      observation,
      AMBIENT_LOCAL_PROTOCOL_PACK
    );

    expect(validation.status).toBe("fail");
    expect(validation.errors).toContain(
      "ambient.voice.pause_rate is missing a nonnegative integer pauseCount."
    );
  });

  it("binds a measured outcome value to its referenced measurement", () => {
    const observation = createQualifiedZeroPauseRateObservation();
    observation.measurements[0].value = 1;

    const validation = validateObservationProvenance(
      observation,
      AMBIENT_LOCAL_PROTOCOL_PACK
    );

    expect(validation.status).toBe("fail");
    expect(validation.errors.join(" ")).toMatch(/numeric value/);
  });

  it("binds a measured outcome quality score to its referenced measurement", () => {
    const observation = createQualifiedZeroPauseRateObservation();
    observation.measurements[0].technicalQualityScore = 0.5;

    const validation = validateObservationProvenance(
      observation,
      AMBIENT_LOCAL_PROTOCOL_PACK
    );

    expect(validation.status).toBe("fail");
    expect(validation.errors.join(" ")).toMatch(/technical quality score/);
  });

  it("rejects a referenced measurement backed by a withheld unrelated window", () => {
    const observation = createQualifiedZeroPauseRateObservation();
    const unrelatedWindowId = "voice-withheld-measurement-source";
    observation.windows.push({
      ...observation.windows[0],
      windowId: unrelatedWindowId,
      endMs: 1,
      technicalQualityScore: 0,
      status: "withheld",
      reasonCodes: ["quality-threshold-failed"]
    });
    observation.qualitySummary.totalWindowCount = 5;
    observation.qualitySummary.withheldWindowCount = 2;

    const measurement = observation.measurements[0];
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
    const measurementId = createMeasurementId(
      identity,
      unrelatedWindowId,
      measurement.ordinal
    );
    measurement.measurementId = measurementId;
    measurement.sourceWindowRefs = [unrelatedWindowId];

    const pauseRate = observation.metricOutcomes.find(
      (outcome) => outcome.metricCode === "ambient.voice.pause_rate"
    );
    if (!pauseRate || pauseRate.status !== "measured") {
      throw new Error("Missing measured pause-rate outcome.");
    }
    const measurementRef = pauseRate.evidence.refs.find(
      (ref) => ref.kind === "measurement"
    );
    if (!measurementRef || measurementRef.kind !== "measurement") {
      throw new Error("Missing pause-rate measurement evidence.");
    }
    measurementRef.measurementId = measurementId;

    const validation = validateObservationProvenance(
      observation,
      AMBIENT_LOCAL_PROTOCOL_PACK
    );

    expect(validation.status).toBe("fail");
    expect(validation.errors).toContain(
      `Measurement ${measurementId} source window is not eligible.`
    );
    expect(validation.errors).toContain(
      `Measurement ${measurementId} source window is not present in the outcome window evidence.`
    );
    expect(validation.errors).toContain(
      "ambient.voice.pause_rate measurement source windows do not exactly match its outcome window evidence."
    );
  });

  it("allows plural measurements whose source subsets exactly cover the outcome windows", () => {
    const observation = createQualifiedZeroPauseRateObservation();
    const secondWindowId = "voice-eligible-window-2";
    const thirdWindowId = "voice-eligible-window-3";

    const pauseRate = observation.metricOutcomes.find(
      (outcome) => outcome.metricCode === "ambient.voice.pause_rate"
    );
    if (!pauseRate || pauseRate.status !== "measured") {
      throw new Error("Missing measured pause-rate outcome.");
    }
    const firstWindowRef = pauseRate.evidence.refs.find(
      (ref) => ref.kind === "window"
    );
    const firstMeasurementRef = pauseRate.evidence.refs.find(
      (ref) => ref.kind === "measurement"
    );
    if (!firstWindowRef || firstWindowRef.kind !== "window") {
      throw new Error("Missing pause-rate window evidence.");
    }
    if (!firstMeasurementRef || firstMeasurementRef.kind !== "measurement") {
      throw new Error("Missing pause-rate measurement evidence.");
    }
    const firstMeasurement = observation.measurements[0];
    firstMeasurement.sourceWindowRefs = [observation.windows[0].windowId];
    const secondOrdinal = 1;
    const secondMeasurementId = createMeasurementId(
      {
        protocolPackId: observation.protocolRef.packId,
        protocolVersion: observation.protocolRef.version,
        sessionId: observation.sessionId,
        metricCode: firstMeasurement.metricCode,
        context: firstMeasurement.context,
        unit: firstMeasurement.unit,
        algorithmVersion: firstMeasurement.algorithmVersion,
        processorRef: firstMeasurement.processorRef,
        trackSegmentId: firstMeasurement.trackSegmentId
      },
      secondWindowId,
      secondOrdinal
    );
    observation.measurements.push({
      ...firstMeasurement,
      measurementId: secondMeasurementId,
      ordinal: secondOrdinal,
      sourceWindowRefs: [secondWindowId, thirdWindowId]
    });
    pauseRate.evidence.refs.push({
      ...firstMeasurementRef,
      measurementId: secondMeasurementId
    });

    expect(
      validateObservationProvenance(
        observation,
        AMBIENT_LOCAL_PROTOCOL_PACK
      )
    ).toEqual({ status: "pass", errors: [] });
  });

  it("rejects measurements that are not referenced by a terminal outcome", () => {
    const observation = createQualifiedZeroPauseRateObservation();
    const firstMeasurement = observation.measurements[0];
    const unreferencedMeasurementId = createMeasurementId(
      {
        protocolPackId: observation.protocolRef.packId,
        protocolVersion: observation.protocolRef.version,
        sessionId: observation.sessionId,
        metricCode: firstMeasurement.metricCode,
        context: firstMeasurement.context,
        unit: firstMeasurement.unit,
        algorithmVersion: firstMeasurement.algorithmVersion,
        processorRef: firstMeasurement.processorRef,
        trackSegmentId: firstMeasurement.trackSegmentId
      },
      firstMeasurement.sourceWindowRefs[0],
      1
    );
    observation.measurements.push({
      ...firstMeasurement,
      measurementId: unreferencedMeasurementId,
      ordinal: 1
    });

    const validation = validateObservationProvenance(
      observation,
      AMBIENT_LOCAL_PROTOCOL_PACK
    );

    expect(validation.status).toBe("fail");
    expect(validation.errors).toContain(
      `Measurement ${unreferencedMeasurementId} is not referenced by a terminal outcome.`
    );
  });

  it("rejects a tampered pack even when its declared digest is unchanged", () => {
    const tampered = structuredClone(AMBIENT_LOCAL_PROTOCOL_PACK);
    tampered.metrics[0].evidenceRequirements.minimumSegments = 1;
    const result = validateObservationProvenance(
      createObservation(),
      tampered
    );
    expect(result.status).toBe("fail");
    expect(result.errors).toContain(
      "Supplied protocol pack is not the canonical active pack."
    );
  });
});

describe("pack requirement coverage", () => {
  /**
   * Every fact the adapter can emit, set to a value that satisfies any
   * threshold in either direction. The point is not whether a gate passes but
   * whether the requirement is *reachable* at all.
   */
  const ALL_FACTS: Record<string, number> = {
    eligibleDurationMs: 1,
    sampleCount: 1,
    segmentCount: 1,
    qualifyingBinCount: 1,
    activeSpeechDurationMs: 1,
    observationSpanMs: 1,
    pitchedDurationMs: 1,
    pitchCoverage: 1,
    timingCoverage: 1,
    pauseCount: 1,
    speechRunCount: 1,
    nucleusCount: 1,
    frontalExposureMs: 1,
    blinkCount: 1,
    expressionEventCount: 1,
    coupledExpressionEventCount: 1,
    estimatorQuality: 1,
    estimatorAgreement: 1,
    segmentSpanMs: 1,
    activeSpeechPerSegmentMs: 1,
    validBinsPerSegment: 1,
    cadenceHz: 1,
    p95FrameGapMs: 1,
    maximumFrameGapMs: 1,
    dataPerBinMs: 1,
    samplesPerBin: 1,
    binSpanMs: 1
  };

  function fullyEvidencedOutcome(metricCode: string): MetricOutcomeV1 {
    return {
      status: "measured",
      value: 100,
      metricCode,
      evidence: {
        eligibleDurationMs: 1,
        activeDurationMs: 1,
        segmentCount: 1,
        windowCount: 1,
        binCount: 1,
        eventCount: 1,
        sampleCount: 1,
        coverage: 1,
        qualityFacts: ALL_FACTS,
        refs: []
      }
    } as unknown as MetricOutcomeV1;
  }

  it("resolves every requirement the pack declares", () => {
    const unreachable: string[] = [];
    for (const metric of AMBIENT_LOCAL_PROTOCOL_PACK.metrics) {
      for (const requirement of Object.keys(metric.evidenceRequirements)) {
        if (ALGORITHM_PARAMETER_REQUIREMENT_KEYS.has(requirement)) continue;
        const fact = evidenceFactFor(
          fullyEvidencedOutcome(metric.code),
          requirement
        );
        if (fact === undefined) {
          unreachable.push(`${metric.code}:${requirement}`);
        }
      }
    }
    // A requirement the pack publishes but the report cannot resolve is a gate
    // nothing enforces at this boundary. Since a missing fact now fails
    // provenance rather than passing silently, an unreachable requirement
    // would withhold every metric that declares it.
    expect(unreachable).toEqual([]);
  });

  it("declares every algorithm parameter that the pack actually uses", () => {
    const declared = new Set<string>();
    for (const metric of AMBIENT_LOCAL_PROTOCOL_PACK.metrics) {
      for (const requirement of Object.keys(metric.evidenceRequirements)) {
        declared.add(requirement);
      }
    }
    // Guards the other direction: an algorithm-parameter exemption left behind
    // after its requirement was removed would silently exempt a future
    // requirement that happened to reuse the name.
    const stale = [...ALGORITHM_PARAMETER_REQUIREMENT_KEYS].filter(
      (key) => !declared.has(key)
    );
    expect(stale).toEqual([]);
  });
});
