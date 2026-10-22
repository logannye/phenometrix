/** Clock coordinates used by the transport; no hardware-input latency is implied. */
export const VOICE_TIMING_CONTRACT_VERSION = "phenometric.voice-stream-timing.v1" as const;

export interface VoiceClockMapping {
  id: number;
  mainTimeOriginMs: number;
  mainPerformanceMs: number;
  audioContextTimeSeconds: number;
  /** Half the bracket around the audio-clock read; audio-clock accuracy is unknown. */
  anchorReadUncertaintyMs: number;
  audioClockUncertaintyMs: null;
}

export interface VoiceFrameTiming {
  schemaVersion: typeof VOICE_TIMING_CONTRACT_VERSION;
  clockMappingId: number;
  coordinate: "main-performance-ms";
  status: "valid" | "unavailable";
  unavailableReason?: "inconsistent-clock-mapping";
  supportStartSample: number;
  supportEndSampleExclusive: number;
  /** Boundary after the last observed sample, not an observed future sample. */
  referenceSample: number;
  lastRequiredInputSample: number;
  sampleRateHz: number;
  referenceTimeMs: number;
  sampleAvailabilityMs: number;
  blockEndAvailabilityMs: number;
  workerReceivedAtMs: number;
  workerStartedAtMs: number;
  workerReadyAtMs: number;
  workerServiceMs: number;
  sampleAvailabilityToWorkerReadyMs: number | null;
  rawSampleAvailabilityToWorkerReadyMs: number;
  /** Combined transport/scheduling delay; not an independently measured queue delay. */
  blockEndToWorkerReceiptMs: number | null;
  rawBlockEndToWorkerReceiptMs: number;
  anchorReadUncertaintyMs: number;
  audioClockUncertaintyMs: null;
  mainResultArrivalMs?: number;
  sampleAvailabilityToMainResultMs?: number | null;
  rawSampleAvailabilityToMainResultMs?: number;
}

/** Whole accepted PCM handler, excluding only this timing message's construction/post. */
export interface VoiceBlockTiming {
  schemaVersion: typeof VOICE_TIMING_CONTRACT_VERSION;
  captureEpoch: number;
  trackSegmentId: string;
  blockSequence: number;
  supportStartSample: number;
  supportEndSampleExclusive: number;
  analyzedFrameCount: number;
  disposition: "processed" | "stale-dropped";
  reason?: "audio-timeline-age-exceeded";
  workerReceivedAtMs: number;
  workerCompletedAtMs: number;
  workerServiceMs: number;
}

export function validVoiceClockMapping(mapping: VoiceClockMapping | null | undefined): mapping is VoiceClockMapping {
  return !!mapping && Number.isSafeInteger(mapping.id) && mapping.id >= 0 &&
    [mapping.mainTimeOriginMs, mapping.mainPerformanceMs, mapping.audioContextTimeSeconds,
      mapping.anchorReadUncertaintyMs].every(Number.isFinite) &&
    mapping.mainPerformanceMs >= 0 && mapping.audioContextTimeSeconds >= 0 &&
    mapping.anchorReadUncertaintyMs >= 0 && mapping.audioClockUncertaintyMs === null;
}

export function captureVoiceClockMapping(
  context: Pick<AudioContext, "currentTime">,
  clock: Pick<Performance, "now" | "timeOrigin">,
  id: number
): VoiceClockMapping {
  const before = clock.now();
  const audioContextTimeSeconds = context.currentTime;
  const after = clock.now();
  const mapping: VoiceClockMapping = {
    id, mainTimeOriginMs: clock.timeOrigin,
    mainPerformanceMs: (before + after) / 2,
    audioContextTimeSeconds,
    anchorReadUncertaintyMs: (after - before) / 2,
    audioClockUncertaintyMs: null
  };
  if (!validVoiceClockMapping(mapping)) throw new Error("invalid-voice-clock-mapping");
  return mapping;
}

export function audioTimeInMainClock(audioSeconds: number, mapping: VoiceClockMapping): number {
  if (!validVoiceClockMapping(mapping) || !Number.isFinite(audioSeconds)) {
    throw new Error("invalid-voice-clock-mapping");
  }
  return mapping.mainPerformanceMs + (audioSeconds - mapping.audioContextTimeSeconds) * 1_000;
}

export function performanceTimeInMainClock(
  localTimeMs: number, localTimeOriginMs: number, mapping: VoiceClockMapping
): number {
  if (!validVoiceClockMapping(mapping) || !Number.isFinite(localTimeMs) || !Number.isFinite(localTimeOriginMs)) {
    throw new Error("invalid-voice-clock-mapping");
  }
  // Subtract origins first to avoid adding a small timestamp to Unix-epoch magnitude.
  return localTimeMs + (localTimeOriginMs - mapping.mainTimeOriginMs);
}

export function validateVoiceDurations(timing: { workerServiceMs: number;
  sampleAvailabilityToWorkerReadyMs: number; blockEndToWorkerReceiptMs: number }): void {
  if (Object.values(timing).some(value => !Number.isFinite(value) || value < 0)) {
    throw new Error("invalid-voice-clock-duration");
  }
}
