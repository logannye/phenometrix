import { describe, expect, it, vi } from "vitest";
import { VoiceResultBatchReceiver } from "./voice-result-transport.js";
import { VOICE_WORKER_MESSAGE_VERSION, type VoiceWorkerResponse } from "./voice-worker-protocol.js";

function envelopes(batchId: number, epoch = 1, segment = "audio-1-1", status = "valid", availability = 100) {
  const record = { schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "signal-frame", captureEpoch: epoch,
    frame: { trackSegmentId: segment }, processingLatencyMs: 5,
    timing: { status, sampleAvailabilityMs: availability, workerReadyAtMs: availability + 5 } } as VoiceWorkerResponse;
  const batch = { schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "result-batch", captureEpoch: epoch,
    batchId, clockMappingId: 0, records: [record] } as VoiceWorkerResponse;
  const complete = { schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "result-batch-complete", captureEpoch: epoch,
    batchId, clockMappingId: 0, timing: { workerServiceMs: 1 } } as VoiceWorkerResponse;
  return { batch, complete };
}

describe("bounded main-thread result receiver", () => {
  it("rejects stale results and same-segment continuation, acknowledges before delivery, then recovers a new segment", () => {
    const ack = vi.fn(); const receiver = new VoiceResultBatchReceiver(ack);
    const first = envelopes(1);
    receiver.receive(first.batch, 1, 300);
    const stale = receiver.receive(first.complete, 1, 300);
    expect(stale.discardedFrames[0].discardReason).toBe("result-age-exceeded");
    expect(ack).toHaveBeenLastCalledWith(expect.objectContaining({ batchId: 1, discardedFrameCount: 1, staleFrameCount: 1 }));
    const next = envelopes(2, 1, "audio-1-1", "valid", 295);
    receiver.receive(next.batch, 1, 300);
    expect(receiver.receive(next.complete, 1, 300).discardedFrames[0].discardReason).toBe("prior-result-loss");
    const renewed = envelopes(3, 1, "audio-1-2", "valid", 295);
    receiver.receive(renewed.batch, 1, 300);
    const accepted = receiver.receive(renewed.complete, 1, 300);
    expect(accepted.messages[0].type).toBe("signal-frame");
    expect(ack).toHaveBeenCalledTimes(3); // Already acknowledged before application callbacks can throw.
    expect(() => { for (const _ of accepted.messages) throw new Error("callback-failed"); }).toThrow("callback-failed");
    expect(ack).toHaveBeenCalledTimes(3);
  });

  it("preserves unavailable timing instead of claiming or enforcing a clock-derived age bound", () => {
    const receiver = new VoiceResultBatchReceiver(() => {});
    for (const [batchId, status, availability] of [[1, "unavailable", 0], [2, "valid", 400]] as const) {
      const pair = envelopes(batchId, 1, "audio-1-1", status, availability);
      receiver.receive(pair.batch, 1, 300);
      const result = receiver.receive(pair.complete, 1, 300);
      expect(result.discardedFrames).toHaveLength(0);
      expect(result.messages[0]).toMatchObject({ type: "signal-frame", processingLatencyMs: null,
        timing: { status: "unavailable", sampleAvailabilityToMainResultMs: null } });
    }
  });

  it("rejects queued results from a retired mapping even when capture epoch is reused", () => {
    const ack = vi.fn(); const receiver = new VoiceResultBatchReceiver(ack);
    const old = envelopes(1);
    receiver.receive(old.batch, 1, 110, 0);
    const result = receiver.receive(old.complete, 1, 110, 1);
    expect(result.messages).toHaveLength(0);
    expect(result.discardedFrames[0].discardReason).toBe("retired-clock-mapping");
    expect(ack).toHaveBeenCalledWith(expect.objectContaining({ captureEpoch: 1, batchId: 1,
      discardedFrameCount: 1, staleFrameCount: 0 }));
  });

  it("ignores duplicate pending and completed envelopes without redelivery or a second ACK", () => {
    const ack = vi.fn(); const receiver = new VoiceResultBatchReceiver(ack);
    const pair = envelopes(1);
    receiver.receive(pair.batch, 1, 110, 0);
    receiver.receive(pair.batch, 1, 110, 0);
    expect(receiver.receive(pair.complete, 1, 110, 0).messages[0].type).toBe("signal-frame");
    expect(receiver.receive(pair.batch, 1, 110, 0).messages).toHaveLength(0);
    expect(receiver.receive(pair.complete, 1, 110, 0).messages).toHaveLength(0);
    expect(ack).toHaveBeenCalledTimes(1);
  });

  it("bounds envelope assembly and returns a retired epoch's token without delivering its records", () => {
    const ack = vi.fn(); const receiver = new VoiceResultBatchReceiver(ack);
    for (let i = 1; i <= 4; i += 1) receiver.receive(envelopes(i).batch, 2, 300);
    expect(() => receiver.receive(envelopes(5).batch, 2, 300)).toThrow("invalid-voice-result-batch");
    expect(receiver.receive(envelopes(1).complete, 2, 300).messages).toHaveLength(0);
    expect(ack).toHaveBeenCalledWith(expect.objectContaining({ captureEpoch: 1, batchId: 1, discardedFrameCount: 0 }));
    receiver.receive(envelopes(5, 2).batch, 2, 300);
    expect(() => receiver.receive(envelopes(5, 1).complete, 2, 300)).toThrow("unmatched-voice-result-completion");
  });
});
