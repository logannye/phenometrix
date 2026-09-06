import { describe, expect, it } from "vitest";
import type { AmbientFacialFrame } from "@phenometrix/ambient-core";
import {
  AMBIENT_LOCAL_PROTOCOL_PACK,
  AcceptedReferenceV1Schema,
  type ConsentRecordV1,
  type ObservationV3,
  type ProcessorProvenanceV1
} from "@phenometrix/contracts";
import { createUnilateralFacialMovementDemoContext } from "@phenometrix/condition-profiles";
import {
  buildConditionEvidenceCard,
  buildPostEncounterReport,
  validateObservationProvenance
} from "@phenometrix/evidence-core";
import { comparePreviousVisit } from "@phenometrix/trajectory-core";
import { buildAmbientObservation } from "./ambient-core-adapter.js";

const SUBJECT_REF = "subject-condition-integration";
const FACE_PROCESSOR: ProcessorProvenanceV1 = {
  modality: "face",
  processorRef: "mediapipe-face-landmarker@condition-integration",
  runtime: "mediapipe-tasks-vision",
  runtimeVersion: "0.10.35",
  assetPath: "models/face_landmarker.task",
  assetSha256: "9".repeat(64),
  assetIntegrityVerified: true
};
const FACE_CALIBRATION = {
  durationMs: 1_500,
  baselineBoxWidthPixels: 384,
  baselineBoxHeightPixels: 360
} as const;

function consent(sessionId: string, recordedAt: string): ConsentRecordV1 {
  return {
    schemaVersion: "phenometric.consent-record.v1",
    consentId: `consent-${sessionId}`,
    sessionId,
    documentVersion: "ambient-local-consent.v1",
    documentSha256: AMBIENT_LOCAL_PROTOCOL_PACK.consentDocument.contentSha256,
    recordedAt,
    scopes: {
      cameraCapture: true,
      microphoneCapture: true,
      localInMemoryAnalysis: true
    },
    localParticipantAssertion: true,
    withdrawnAt: null
  };
}

/**
 * Generates compact, v3-native face geometry rather than a result artifact.
 * The mouth corners are anatomical mirrors around y=0.1, so changing the sign
 * swaps which subject side sits higher. The optional left-lid shortfall models
 * one-sided incomplete closure while preserving the right-eye trajectory.
 */
function generatedFaceFrames(input: {
  restingMouthDifference: number;
  leftClosureAperture: number;
  durationMs?: number;
  cadenceHz?: number;
}): AmbientFacialFrame[] {
  const durationMs = input.durationMs ?? 30_000;
  const cadenceHz = input.cadenceHz ?? 30;
  const stepMs = 1_000 / cadenceHz;
  return Array.from(
    { length: Math.round(durationMs / stepMs) },
    (_, index) => {
      const tMs = index * stepMs;
      const blinkPhase = tMs % 10_000;
      const closed = blinkPhase >= 1_000 && blinkPhase < 1_100;
      return {
        schemaVersion: "phenometric.facial-kinematics-frame.v1",
        tMs,
        acquiredAtMs: tMs,
        sequence: index + 1,
        captureEpoch: 1,
        taskContext: "ambient-frontal",
        faceCount: 1,
        trackSegmentId: "condition-face-track",
        faceVisible: true,
        boundingBox: {
          x: 0.35,
          y: 0.2,
          width: 0.3,
          height: 0.5,
          widthPixels: 384,
          heightPixels: 360,
          edgeMarginFraction: 0.1
        },
        anatomicalLaterality: "subject-anatomical",
        pose: { yawDegrees: 0, pitchDegrees: 0, rollDegrees: 0 },
        eyeAperture: closed
          ? { left: input.leftClosureAperture, right: 0.1 }
          : { left: 0.3, right: 0.3 },
        browHeight: { left: 0.55, right: 0.55 },
        mouthCorners: {
          left: {
            x: 0.3,
            y: 0.1 - input.restingMouthDifference / 2
          },
          right: {
            x: -0.3,
            y: 0.1 + input.restingMouthDifference / 2
          }
        },
        mouthApertureRatio: 0.08,
        regionalMovementSpeed: 0.02,
        imageQuality: {
          illuminationMean: 0.55,
          darkClippingFraction: 0.02,
          brightClippingFraction: 0.02,
          sharpness: 0.002
        },
        analyzedFrameRate: cadenceHz,
        interResultGapMs: index === 0 ? null : stepMs,
        skippedFrameFraction: 0,
        processingLatencyMs: 4,
        qualityReasons: [],
        processorRef: FACE_PROCESSOR.processorRef
      };
    }
  );
}

function observation(input: {
  sessionId: string;
  startedAt: string;
  endedAt: string;
  restingMouthDifference: number;
  leftClosureAperture: number;
}): ObservationV3 {
  return buildAmbientObservation({
    sessionId: input.sessionId,
    subjectRef: SUBJECT_REF,
    consent: consent(input.sessionId, input.startedAt),
    startedAt: input.startedAt,
    endedAt: input.endedAt,
    durationMs: 30_000,
    voiceFrames: [],
    faceFrames: generatedFaceFrames({
      restingMouthDifference: input.restingMouthDifference,
      leftClosureAperture: input.leftClosureAperture
    }),
    noiseCalibrationDurationMs: 0,
    faceCalibration: FACE_CALIBRATION,
    voiceLaneAvailable: false,
    faceLaneAvailable: true,
    processors: [FACE_PROCESSOR]
  });
}

describe("unilateral condition pipeline integration", () => {
  it("carries mirrored sign crossing and one-sided closure through report, comparison, and card", () => {
    const reference = observation({
      sessionId: "condition-reference-session",
      startedAt: "2026-08-21T10:00:00.000Z",
      endedAt: "2026-08-21T10:00:30.000Z",
      restingMouthDifference: 0.04,
      leftClosureAperture: 0.1
    });
    const current = observation({
      sessionId: "condition-current-session",
      startedAt: "2026-08-21T10:01:00.000Z",
      endedAt: "2026-08-21T10:01:30.000Z",
      restingMouthDifference: -0.04,
      leftClosureAperture: 0.24
    });

    for (const value of [reference, current]) {
      expect(
        validateObservationProvenance(value, AMBIENT_LOCAL_PROTOCOL_PACK)
      ).toEqual({ status: "pass", errors: [] });
      const report = buildPostEncounterReport(
        value,
        AMBIENT_LOCAL_PROTOCOL_PACK,
        { generatedAt: value.endedAt }
      );
      expect(report.sections.flatMap((section) => section.outcomes)).toHaveLength(
        27
      );
    }

    const context = createUnilateralFacialMovementDemoContext({
      subjectRef: SUBJECT_REF,
      assertedAffectedSide: "left"
    });
    const acceptedReference = AcceptedReferenceV1Schema.parse({
      schemaVersion: "phenometric.accepted-reference.v1",
      referenceId: "condition-reference",
      demoContext: context,
      observation: reference,
      acceptedAt: "2026-08-21T10:00:45.000Z",
      localReviewerAction: {
        action: "use-as-in-memory-reference",
        actor: "local-demo-user",
        status: "accepted"
      },
      persistence: "page-memory-only"
    });
    const comparison = comparePreviousVisit({
      currentContext: context,
      acceptedReference,
      currentObservation: current
    });
    const card = buildConditionEvidenceCard({
      comparison,
      generatedAt: current.endedAt
    });

    const mouth = card.rows.find(
      (row) =>
        row.metric.metricCode ===
        "ambient.face.rest_mouth_corner_asymmetry.signed"
    );
    expect(mouth?.comparison.status).toBe("measured");
    if (!mouth || mouth.comparison.status !== "measured") {
      throw new Error("Expected a measured resting-mouth comparison.");
    }
    expect(mouth.comparison.reference.outcome.value).toBeCloseTo(0.04, 6);
    expect(mouth.comparison.current.outcome.value).toBeCloseTo(-0.04, 6);
    expect(mouth.comparison.current.outcome.value).toBeCloseTo(
      -mouth.comparison.reference.outcome.value,
      12
    );
    expect(mouth.comparison.reference).toMatchObject({
      observationId: reference.observationId,
      sessionId: reference.sessionId
    });
    expect(mouth.comparison.current).toMatchObject({
      observationId: current.observationId,
      sessionId: current.sessionId
    });
    expect(mouth.comparison.delta).toBeCloseTo(-0.08, 6);
    expect(mouth.comparison.delta).toBe(
      mouth.comparison.current.outcome.value -
        mouth.comparison.reference.outcome.value
    );

    const leftClosure = card.rows.find(
      (row) =>
        row.metric.metricCode ===
        "ambient.face.lid_closure_completeness.left"
    );
    const rightClosure = card.rows.find(
      (row) =>
        row.metric.metricCode ===
        "ambient.face.lid_closure_completeness.right"
    );
    expect(leftClosure?.comparison.status).toBe("measured");
    expect(rightClosure?.comparison.status).toBe("measured");
    if (
      !leftClosure ||
      leftClosure.comparison.status !== "measured" ||
      !rightClosure ||
      rightClosure.comparison.status !== "measured"
    ) {
      throw new Error("Expected measured left and right closure comparisons.");
    }
    expect(leftClosure.comparison.current.outcome.value).toBeLessThan(
      leftClosure.comparison.reference.outcome.value
    );
    expect(leftClosure.comparison.delta).toBeLessThan(0);
    expect(rightClosure.comparison.delta).toBeCloseTo(0, 12);

    expect(card.qualitySummary).toMatchObject({
      measuredComparisonCount: 4,
      withheldComparisonCount: 2,
      incompatibleComparisonCount: 0
    });
    expect(card.persistence).toBe("page-memory-only");
    expect(card.exportAvailable).toBe(false);
  });
});
