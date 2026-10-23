import { audioTimeInMainClock, type VoiceClockMapping } from "./voice-stream-timing.js";

export const VOICE_CLOCK_PROBE_VERSION = "phenometric.voice-clock-probe.v1" as const;
export interface VoiceRenderClockProbeRequest {
  schemaVersion: typeof VOICE_CLOCK_PROBE_VERSION;
  type: "clock-probe";
  probeId: number;
  captureEpoch: number;
  clockMappingId: number;
}
export interface VoiceRenderClockProbeResponse extends Omit<VoiceRenderClockProbeRequest, "type"> {
  type: "clock-probe-result";
  renderStartFrame: number;
  sampleRateHz: number;
}
export interface VoiceClockObservation {
  schemaVersion: "phenometric.voice-clock-observation.v1";
  probeId: number;
  captureEpoch: number;
  clockMappingId: number;
  sourceElapsedMs: number;
  contextState: string;
  contextRead: { beforeMs: number; afterMs: number; midpointMs: number; audioContextTimeSeconds: number;
    bracketWidthMs: number; fixedAnchorResidualMs: number };
  outputTimestamp: { status: "observed"; coordinate: "output-device-only-not-capture";
    beforeMs: number; afterMs: number; contextTimeSeconds: number; performanceTimeMs: number;
    fixedAnchorResidualMs: number } | { status: "unavailable"; reason: "unsupported" | "zero-or-invalid" | "read-failed" };
  renderObservation: { status: "observed"; coordinate: "render-callback-main-performance-bracket";
    renderStartFrame: number; sampleRateHz: number; requestSentAtMs: number; responseReceivedAtMs: number;
    roundTripMs: number; fixedAnchorResidualLowerMs: number; fixedAnchorResidualUpperMs: number }
    | { status: "unavailable"; reason: "timeout" | "stopped"; requestSentAtMs: number; observedAtMs: number };
  /** Neither the context read nor the output timestamp establishes capture-clock accuracy. */
  audioClockUncertaintyMs: null;
}
interface ProbeContext { currentTime: number; state: string; sampleRate: number;
  getOutputTimestamp?: () => { contextTime?: number; performanceTime?: number } }
type ProbeClock = Pick<Performance, "now">;

/** One outstanding diagnostic request. It never changes mapping, admission, or signal state. */
export class VoiceClockProbeTracker {
  private nextId = 0;
  private pending: { request: VoiceRenderClockProbeRequest; sentAtMs: number; expectedRateHz: number; expired: boolean;
    partial: Omit<VoiceClockObservation, "renderObservation"> } | null = null;
  constructor(private readonly mapping: VoiceClockMapping, private readonly captureEpoch: number,
    private readonly clock: ProbeClock) {}
  begin(context: ProbeContext, sourceElapsedMs: number): VoiceRenderClockProbeRequest | null {
    if (this.pending) return null;
    const beforeMs = this.clock.now();
    const audioContextTimeSeconds = context.currentTime;
    const afterMs = this.clock.now();
    const midpointMs = (beforeMs + afterMs) / 2;
    let outputTimestamp: VoiceClockObservation["outputTimestamp"] = { status: "unavailable", reason: "unsupported" };
    if (context.getOutputTimestamp) {
      try {
        const outputBefore = this.clock.now();
        const value = context.getOutputTimestamp();
        const outputAfter = this.clock.now();
        if (typeof value.contextTime === "number" && typeof value.performanceTime === "number" &&
            Number.isFinite(value.contextTime) && value.contextTime > 0 &&
            Number.isFinite(value.performanceTime) && value.performanceTime > 0) {
          outputTimestamp = { status: "observed", coordinate: "output-device-only-not-capture",
            beforeMs: outputBefore, afterMs: outputAfter, contextTimeSeconds: value.contextTime,
            performanceTimeMs: value.performanceTime,
            fixedAnchorResidualMs: value.performanceTime - audioTimeInMainClock(value.contextTime, this.mapping) };
        } else outputTimestamp = { status: "unavailable", reason: "zero-or-invalid" };
      } catch { outputTimestamp = { status: "unavailable", reason: "read-failed" }; }
    }
    const request: VoiceRenderClockProbeRequest = { schemaVersion: VOICE_CLOCK_PROBE_VERSION,
      type: "clock-probe", probeId: ++this.nextId, captureEpoch: this.captureEpoch, clockMappingId: this.mapping.id };
    this.pending = { request, sentAtMs: this.clock.now(), expectedRateHz: context.sampleRate, expired: false,
      partial: { schemaVersion: "phenometric.voice-clock-observation.v1", probeId: request.probeId,
        captureEpoch: this.captureEpoch, clockMappingId: this.mapping.id, sourceElapsedMs, contextState: context.state,
        contextRead: { beforeMs, afterMs, midpointMs, audioContextTimeSeconds, bracketWidthMs: afterMs - beforeMs,
          fixedAnchorResidualMs: midpointMs - audioTimeInMainClock(audioContextTimeSeconds, this.mapping) },
        outputTimestamp, audioClockUncertaintyMs: null } };
    return request;
  }
  receive(value: unknown): VoiceClockObservation | null {
    const receivedAtMs = this.clock.now();
    const response = value as Partial<VoiceRenderClockProbeResponse> | null;
    const pending = this.pending;
    if (!pending || !response || response.schemaVersion !== VOICE_CLOCK_PROBE_VERSION ||
        response.type !== "clock-probe-result" || response.probeId !== pending.request.probeId ||
        response.captureEpoch !== pending.request.captureEpoch || response.clockMappingId !== pending.request.clockMappingId ||
        !Number.isSafeInteger(response.renderStartFrame) || response.renderStartFrame! < 0 ||
        response.sampleRateHz !== pending.expectedRateHz) return null;
    this.pending = null;
    if (pending.expired) return null; // A late response releases the sole probe credit without duplicate evidence.
    const mappedRenderMs = audioTimeInMainClock(response.renderStartFrame! / response.sampleRateHz, this.mapping);
    return { ...pending.partial, renderObservation: { status: "observed",
      coordinate: "render-callback-main-performance-bracket", renderStartFrame: response.renderStartFrame!,
      sampleRateHz: response.sampleRateHz, requestSentAtMs: pending.sentAtMs, responseReceivedAtMs: receivedAtMs,
      roundTripMs: receivedAtMs - pending.sentAtMs,
      fixedAnchorResidualLowerMs: pending.sentAtMs - mappedRenderMs,
      fixedAnchorResidualUpperMs: receivedAtMs - mappedRenderMs } };
  }
  expire(reason: "timeout" | "stopped"): VoiceClockObservation | null {
    if (!this.pending) return null;
    const pending = this.pending;
    if (reason === "stopped") this.pending = null;
    if (pending.expired) return null;
    // Timeout reports unavailability but does not mint another transport credit.
    // A matching late response must arrive before another request can be sent.
    pending.expired = true;
    return { ...pending.partial, renderObservation: { status: "unavailable", reason,
      requestSentAtMs: pending.sentAtMs, observedAtMs: this.clock.now() } };
  }
}
