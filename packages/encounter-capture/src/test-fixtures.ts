import type { AmbientVoiceFrame } from "@phenometrix/ambient-core";
import type { CaptureAuthorization } from "./controller.js";

export const EPOCH = 1_700_000_000_000;
export function authorization(): CaptureAuthorization {
  const scope = { tenantId: "test-tenant", studyId: "test-study", participantId: "test-patient" };
  return { encounterId: "test-encounter", encounterActive: true, modalities: ["voice", "face"],
    binding: { bindingId: "test-binding", scope, encounterId: "test-encounter", platformParticipantId: "123",
      status: "verified", verificationMethod: "clinic-enrollment", recordedAt: new Date(EPOCH - 1_000).toISOString(), revokedAt: null },
    consent: { consentId: "test-consent", scope, documentRef: { id: "test-consent-doc", version: "1", contentSha256: "a".repeat(64) },
      grantedAt: new Date(EPOCH - 1_000).toISOString(), expiresAt: null, withdrawnAt: null,
      permissions: { capture: true, derivedAnalysis: true, derivedRetention: true, researchClips: false, modalities: ["face", "voice"] } } };
}
export function voiceFrame(tMs = 10): AmbientVoiceFrame {
  return { schemaVersion: "phenometric.voice-signal-frame.v1", tMs, acquiredAtMs: EPOCH + tMs,
    captureEpoch: 1, sequence: tMs / 10 + 1, absoluteSampleIndex: tMs * 48,
    taskContext: "ambient-speech-turn", speechActive: true, periodic: true, trackSegmentId: "test-track",
    rms: 0.08, f0Hz: 150, f0Confidence: 0.95, estimatorAgreement: 0.95, syllabicNucleus: tMs % 500 === 0,
    clippedSampleFraction: 0, dcOffset: 0, snrDb: 30, sampleRateHz: 48_000, blockGapMs: 10, lostBlockFraction: 0,
    browserProcessing: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    qualityReasons: [], processorRef: "synthetic-test-processor" };
}
