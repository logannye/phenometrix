import { VoiceRenderLeaseIssuer, type VoiceRenderLease } from "./voice-render-freshness.js";
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
    const ack = vi.fn(); const receiver = new VoiceResultBatchReceiver(ack, { policy: "unqualified-legacy-fixture" });
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
    const receiver = new VoiceResultBatchReceiver(() => {}, { policy: "unqualified-legacy-fixture" });
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
    const ack = vi.fn(); const receiver = new VoiceResultBatchReceiver(ack, { policy: "unqualified-legacy-fixture" });
    const old = envelopes(1);
    receiver.receive(old.batch, 1, 110, 0);
    const result = receiver.receive(old.complete, 1, 110, 1);
    expect(result.messages).toHaveLength(0);
    expect(result.discardedFrames[0].discardReason).toBe("retired-clock-mapping");
    expect(ack).toHaveBeenCalledWith(expect.objectContaining({ captureEpoch: 1, batchId: 1,
      discardedFrameCount: 1, staleFrameCount: 0 }));
  });

  it("ignores duplicate pending and completed envelopes without redelivery or a second ACK", () => {
    const ack = vi.fn(); const receiver = new VoiceResultBatchReceiver(ack, { policy: "unqualified-legacy-fixture" });
    const pair = envelopes(1);
    receiver.receive(pair.batch, 1, 110, 0);
    receiver.receive(pair.batch, 1, 110, 0);
    expect(receiver.receive(pair.complete, 1, 110, 0).messages[0].type).toBe("signal-frame");
    expect(receiver.receive(pair.batch, 1, 110, 0).messages).toHaveLength(0);
    expect(receiver.receive(pair.complete, 1, 110, 0).messages).toHaveLength(0);
    expect(ack).toHaveBeenCalledTimes(1);
  });

  it("bounds envelope assembly and returns a retired epoch's token without delivering its records", () => {
    const ack = vi.fn(); const receiver = new VoiceResultBatchReceiver(ack, { policy: "unqualified-legacy-fixture" });
    for (let i = 1; i <= 4; i += 1) receiver.receive(envelopes(i).batch, 2, 300);
    expect(() => receiver.receive(envelopes(5).batch, 2, 300)).toThrow("invalid-voice-result-batch");
    expect(receiver.receive(envelopes(1).complete, 2, 300).messages).toHaveLength(0);
    expect(ack).toHaveBeenCalledWith(expect.objectContaining({ captureEpoch: 1, batchId: 1, discardedFrameCount: 0 }));
    receiver.receive(envelopes(5, 2).batch, 2, 300);
    expect(() => receiver.receive(envelopes(5, 1).complete, 2, 300)).toThrow("unmatched-voice-result-completion");
  });
});


describe("required render lease host verification", () => {
  const identity = { captureEpoch: 1, clockMappingId: 0, mainTimeOriginMs: 100000 };
  function fixture() {
    let now = 100;
    const issuer = new VoiceRenderLeaseIssuer({ now: () => now, timeOrigin: identity.mainTimeOriginMs });
    const ack = vi.fn();
    const receiver = new VoiceResultBatchReceiver(ack, { policy: "required-render-lease-v1", issuer,
      mainTimeOriginMs: identity.mainTimeOriginMs });
    const issue = (time: number) => {
      now = time;
      const lease = issuer.issue(1, 0)!.lease;
      issuer.acknowledge({ schemaVersion: lease.schemaVersion, type: "render-freshness-lease-ack",
        leaseId: lease.leaseId, captureEpoch: 1, clockMappingId: 0, disposition: "accepted" });
      return lease;
    };
    const deliver = (id: number, lease: VoiceRenderLease | null, arrival: number, projectedAvailability: number,
      segment = "audio-1-1") => {
      const pair = envelopes(id, 1, segment, "valid", projectedAvailability);
      if (pair.batch.type !== "result-batch") throw new Error("fixture-shape");
      Object.assign(pair.batch.records[0], { renderLease: lease });
      receiver.receive(pair.batch, 1, arrival, 0);
      return receiver.receive(pair.complete, 1, arrival, 0);
    };
    return { issuer, ack, receiver, issue, deliver };
  }
  it("certifies same-origin render freshness independently of large and negative audio projections", () => {
    const f = fixture(); const lease = f.issue(100);
    for (const [id, projection] of [[1, -10000], [2, 10000]]) {
      const result = f.deliver(id, lease, 150, projection);
      expect(result.discardedFrames).toHaveLength(0);
      expect(result.messages[0]).toMatchObject({ type: "signal-frame", timing: {
        renderFreshness: { status: "certified", renderAgeUpperBoundMs: 50, issuedAtMainMs: 100,
          resultReceivedAtMainMs: 150, timestampPrecisionMs: null } } });
    }
    expect(f.ack).toHaveBeenCalledTimes(2);
  });
  it("expires blocked-main results even when the audio projection falsely suggests zero backlog", () => {
    const f = fixture();
    const lease = f.issue(100);
    const expired = f.deliver(1, lease, 450, 445);
    expect(expired.discardedFrames[0]).toMatchObject({ discardReason: "expired-lease",
      timing: { renderFreshness: { status: "withheld", reason: "expired-lease" } } });
    expect(f.ack).toHaveBeenLastCalledWith(expect.objectContaining({ staleFrameCount: 1, discardedFrameCount: 1 }));
    const current = f.issue(450);
    expect(f.deliver(2, current, 470, 465).discardedFrames[0].discardReason).toBe("prior-result-loss");
    const recovered = f.deliver(3, current, 480, 475, "audio-1-2");
    expect(recovered.messages[0]).toMatchObject({ timing: { renderFreshness: { status: "certified" } } });
    expect(f.ack).toHaveBeenCalledTimes(3); // The callback below cannot hold a result credit.
    expect(() => { for (const _ of recovered.messages) throw new Error("consumer-failed"); }).toThrow("consumer-failed");
    expect(f.ack).toHaveBeenCalledTimes(3);
  });
  it("fails closed for missing, forged, retired and disposed lease evidence", () => {
    const f = fixture(); const lease = f.issue(100);
    expect(f.deliver(1, null, 150, 145).discardedFrames[0].discardReason).toBe("missing-lease");
    expect(f.deliver(2, { ...lease, leaseId: 99 }, 150, 145, "audio-1-2").discardedFrames[0].discardReason).toBe("unrecognized-lease");
    expect(f.deliver(3, { ...lease, mainTimeOriginMs: 107000 }, 150, 145, "audio-1-3").discardedFrames[0].discardReason).toBe("clock-origin-mismatch");
    f.issuer.invalidate();
    expect(f.deliver(4, lease, 150, 145, "audio-1-4").discardedFrames[0].discardReason).toBe("unrecognized-lease");
    const next = f.issue(150); f.issuer.stop();
    expect(f.deliver(5, next, 160, 155, "audio-1-5").discardedFrames[0].discardReason).toBe("unrecognized-lease");
    expect(f.ack).toHaveBeenCalledTimes(5);
    expect(() => f.receiver.receive(envelopes(6).batch.type === "result-batch"
      ? (envelopes(6).batch as Extract<VoiceWorkerResponse, { type: "result-batch" }>).records[0] : null!, 1, 160, 0))
      .toThrow("unbatched-voice-result-frame");
  });
});
