import { VOICE_MAXIMUM_RESULT_AGE_MS, VOICE_MAXIMUM_RESULT_BATCHES, VOICE_MAXIMUM_RESULT_RECORDS,
  VOICE_WORKER_MESSAGE_VERSION, type VoiceWorkerRequest, type VoiceWorkerResponse } from "./voice-worker-protocol.js";

/** Bounded assembly of the two ordered envelopes; completion follows the measured data post. */
export class VoiceResultBatchReceiver {
  private readonly pending = new Map<number, Extract<VoiceWorkerResponse, { type: "result-batch" }>>();
  private lastCompletedBatchId = 0;
  private epoch: number | null = null;
  private rejectedSegment: string | null = null;

  constructor(private readonly acknowledge: (request: VoiceWorkerRequest) => void) {}

  receive(message: VoiceWorkerResponse, currentEpoch: number, arrivalMs: number, currentClockMappingId?: number): {
    messages: VoiceWorkerResponse[];
    discardedFrames: Array<Extract<VoiceWorkerResponse, { type: "signal-frame" }> & { discardReason: "result-age-exceeded" | "prior-result-loss" | "retired-clock-mapping" }>;
  } {
    if (this.epoch !== currentEpoch) { this.epoch = currentEpoch; this.rejectedSegment = null; }
    const empty = { messages: [] as VoiceWorkerResponse[], discardedFrames: [] as ReturnType<VoiceResultBatchReceiver["receive"]>["discardedFrames"] };
    if (message.schemaVersion !== VOICE_WORKER_MESSAGE_VERSION) return empty;
    if (message.type === "result-batch") {
      if (message.batchId <= this.lastCompletedBatchId || this.pending.has(message.batchId)) return empty;
      if (!Number.isSafeInteger(message.batchId) || message.batchId < 1 ||
          !Array.isArray(message.records) || message.records.length > VOICE_MAXIMUM_RESULT_RECORDS ||
          this.pending.has(message.batchId) || this.pending.size >= VOICE_MAXIMUM_RESULT_BATCHES) {
        throw new Error("invalid-voice-result-batch");
      }
      this.pending.set(message.batchId, message);
      return empty;
    }
    if (message.type !== "result-batch-complete") return { ...empty, messages: [message] };
    if (message.batchId <= this.lastCompletedBatchId) return empty;
    const batch = this.pending.get(message.batchId);
    if (!batch || batch.captureEpoch !== message.captureEpoch || batch.clockMappingId !== message.clockMappingId) throw new Error("unmatched-voice-result-completion");
    this.pending.delete(message.batchId);
    this.lastCompletedBatchId = message.batchId;
    let staleFrameCount = 0;
    if (batch.captureEpoch === currentEpoch) {
      for (const record of batch.records) {
        const retiredMapping = currentClockMappingId !== undefined && batch.clockMappingId !== currentClockMappingId;
        if (record.type !== "signal-frame") { if (!retiredMapping) empty.messages.push(record); continue; }
        if (retiredMapping) {
          empty.discardedFrames.push({ ...record, discardReason: "retired-clock-mapping" });
          continue;
        }
        const elapsed = arrivalMs - record.timing.sampleAvailabilityMs;
        const valid = record.timing.status === "valid" && Number.isFinite(elapsed) && elapsed >= 0 &&
          arrivalMs >= record.timing.workerReadyAtMs;
        const frame = { ...record, processingLatencyMs: valid ? record.processingLatencyMs : null,
          timing: { ...record.timing, mainResultArrivalMs: arrivalMs,
            status: valid ? "valid" as const : "unavailable" as const,
            ...(valid ? {} : { unavailableReason: "inconsistent-clock-mapping" as const }),
            sampleAvailabilityToMainResultMs: valid ? elapsed : null,
            rawSampleAvailabilityToMainResultMs: elapsed } };
        const stale = valid && elapsed > VOICE_MAXIMUM_RESULT_AGE_MS;
        if (stale || this.rejectedSegment === record.frame.trackSegmentId) {
          this.rejectedSegment = record.frame.trackSegmentId;
          staleFrameCount += Number(stale);
          empty.discardedFrames.push({ ...frame, discardReason: stale ? "result-age-exceeded" : "prior-result-loss" });
        } else {
          this.rejectedSegment = null;
          empty.messages.push(frame);
        }
      }
      if (currentClockMappingId === undefined || batch.clockMappingId === currentClockMappingId) {
        empty.messages.push({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "block-timing",
          captureEpoch: message.captureEpoch, timing: message.timing });
      }
    }
    // Return credit after deciding the discard disposition, before any application callback.
    // Retired-epoch tokens still release their own credit, never a newer token's credit.
    this.acknowledge({ schemaVersion: VOICE_WORKER_MESSAGE_VERSION, type: "result-ack",
      captureEpoch: batch.captureEpoch, batchId: batch.batchId,
      discardedFrameCount: empty.discardedFrames.length, staleFrameCount });
    return empty;
  }
}
