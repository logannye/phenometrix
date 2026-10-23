import { describe, expect, it } from "vitest";
import {
  isCurrentVoiceWorkerResponse,
  isCurrentVoiceTransportDiagnostics,
  VOICE_WORKER_MESSAGE_VERSION
} from "./voice-worker-protocol.js";

describe("voice worker response guard", () => {
  it("accepts only the current capture epoch", () => {
    const response = {
      schemaVersion: VOICE_WORKER_MESSAGE_VERSION,
      type: "disposed",
      captureEpoch: 4,
      diagnostics: {
        receivedBlockCount: 0,
        processedFrameCount: 0,
        lostBlockCount: 0,
        lostBlockFraction: 0,
        maximumBlockGapMs: 0,
        p95FeatureLatencyMs: 0,
        timestampRegressionCount: 0,
        ringBufferCapacitySamples: 0
      }
    };
    expect(isCurrentVoiceWorkerResponse(response, 4)).toBe(true);
    expect(isCurrentVoiceWorkerResponse(response, 5)).toBe(false);
  });

  it("rejects unversioned or malformed messages", () => {
    expect(
      isCurrentVoiceWorkerResponse(
        { captureEpoch: 2, type: "signal-frame" },
        2
      )
    ).toBe(false);
    expect(isCurrentVoiceWorkerResponse(null, 2)).toBe(false);
  });
  it("requires complete finite transport snapshots for the current epoch", () => {
    const snapshot = { schemaVersion: "phenometric.voice-worklet-message.v2", type: "transport-diagnostics",
      captureEpoch: 4, final: true, sequence: 2, sampleClockEndExclusive: 2048,
      droppedBlockCount: 0, droppedSampleCount: 128, discardedPartialSampleCount: 128,
      inFlightBlockCountAtDispose: 2, maximumInFlightBlocks: 4 };
    expect(isCurrentVoiceTransportDiagnostics(snapshot, 4)).toBe(true);
    expect(isCurrentVoiceTransportDiagnostics(snapshot, 5)).toBe(false);
    expect(isCurrentVoiceTransportDiagnostics({ ...snapshot, droppedSampleCount: NaN }, 4)).toBe(false);
    expect(isCurrentVoiceTransportDiagnostics({ ...snapshot, inFlightBlockCountAtDispose: 5 }, 4)).toBe(false);
  });

});
