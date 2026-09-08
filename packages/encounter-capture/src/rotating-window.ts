import { AMBIENT_MAX_CAPTURE_DURATION_MS, type AmbientFaceCalibration } from "@phenometrix/ambient-core";
import type { DurableObservationV1 } from "@phenometrix/contracts";
import { AmbientEncounterWindow, type AmbientWindowOptions } from "./ambient-bridge.js";
import type { AcceptedDerived } from "./controller.js";

export interface RotatingAmbientOptions extends AmbientWindowOptions {
  onObservation?(observation: DurableObservationV1, signal: AbortSignal): void | Promise<void>;
  onFailure?(): void;
  deliveryTimeoutMs?: number;
}

/** Five-minute measurement windows inside an uncapped host encounter. No synthetic gap filling. */
export class RotatingAmbientEncounter {
  private index = 0;
  private startMs: number;
  private current: AmbientEncounterWindow;
  private calibration: { noiseDurationMs?: number; face?: AmbientFaceCalibration | null } = {};
  private readonly pending = new Set<Promise<DurableObservationV1>>();
  private readonly deliveries = new Set<AbortController>();
  private lastDelivery: Promise<DurableObservationV1> | null = null;
  private discarded = false;
  private finished = false;
  private deliveryFailed = false;
  private captureReadiness: Parameters<AmbientEncounterWindow["setCaptureReadiness"]>[0] = {};
  constructor(private readonly options: RotatingAmbientOptions) {
    this.startMs = options.startedAtMs;
    this.current = this.createWindow();
  }
  get currentWindowStartMs() { return this.startMs; }
  setCaptureReadiness(input: Parameters<AmbientEncounterWindow["setCaptureReadiness"]>[0]): void {
    if (this.discarded || this.finished) return;
    this.current.setCaptureReadiness(input);
    this.captureReadiness = { ...this.captureReadiness, ...structuredClone(input),
      ...(input.clockUncertaintyMs === undefined ? {} : { clockUncertaintyMs:
        Math.max(this.captureReadiness.clockUncertaintyMs ?? this.options.clockUncertaintyMs, input.clockUncertaintyMs) }) };
  }
  setCalibration(input: typeof this.calibration): void {
    this.calibration = { ...this.calibration, ...input };
    this.current.setCalibration(input);
  }
  /** Close existing evidence before a source or calibration boundary; never rewrite it later. */
  flushBoundary(nowMs: number): void {
    if (this.discarded || this.finished || this.deliveryFailed) return;
    this.advance(nowMs);
    if (this.discarded || this.deliveryFailed || this.current.frameCount === 0 || nowMs <= this.startMs) return;
    this.lastDelivery = this.deliver(this.current, nowMs);
    if (this.discarded || this.deliveryFailed) return;
    this.startMs = nowMs; this.index++; this.current = this.createWindow();
    this.current.setCaptureReadiness(this.captureReadiness); this.current.setCalibration(this.calibration);
  }
  append(value: AcceptedDerived): boolean {
    if (this.discarded || this.finished || this.deliveryFailed) return false;
    const frames = value.modality === "voice" ? value.batch.voiceFrames : value.batch.faceFrames;
    if (!frames) return false;
    let accepted = true;
    for (const frame of frames) {
      this.advance(frame.acquiredAtMs);
      const rebased = { ...frame, tMs: frame.acquiredAtMs - this.startMs };
      accepted = this.current.append({ ...value,
        batch: value.modality === "voice" ? { voiceFrames: [rebased as NonNullable<typeof value.batch.voiceFrames>[number]] }
          : { faceFrames: [rebased as NonNullable<typeof value.batch.faceFrames>[number]] } }) && accepted;
    }
    return accepted;
  }
  /** Host clock tick allows all-withheld windows when a modality is unavailable. */
  advance(nowMs: number): void {
    if (this.discarded || this.finished || this.deliveryFailed || nowMs - this.startMs < AMBIENT_MAX_CAPTURE_DURATION_MS) return;
    const endMs = this.startMs + AMBIENT_MAX_CAPTURE_DURATION_MS;
    this.lastDelivery = this.deliver(this.current, endMs);
    if (this.discarded || this.deliveryFailed) return;
    // A suspended runtime can miss many windows. Preserve the gap, do not backfill observations.
    this.startMs = nowMs - endMs >= AMBIENT_MAX_CAPTURE_DURATION_MS ? nowMs : endMs;
    this.index++;
    this.current = this.createWindow();
    this.current.setCaptureReadiness(this.captureReadiness);
    this.current.setCalibration(this.calibration);
  }
  async finish(nowMs: number): Promise<DurableObservationV1> {
    if (this.discarded || this.finished || this.deliveryFailed) throw new Error("Encounter already finished, discarded, or delivery failed");
    if (nowMs - this.startMs > AMBIENT_MAX_CAPTURE_DURATION_MS) this.advance(nowMs);
    this.finished = true;
    let final: Promise<DurableObservationV1>;
    if (nowMs === this.startMs && this.index > 0 && this.current.frameCount === 0 && this.lastDelivery) {
      this.current.discard(); final = this.lastDelivery;
    } else final = this.deliver(this.current, nowMs);
    await Promise.all([...this.pending]);
    if (this.deliveryFailed) throw new Error("Derived observation delivery failed");
    return final;
  }
  discard(): void {
    this.discarded = true; this.current.discard();
    for (const delivery of this.deliveries) delivery.abort();
  }

  private createWindow() {
    const suffix = this.index === 0 ? "" : `.window-${this.index}`;
    return new AmbientEncounterWindow({ ...this.options, startedAtMs: this.startMs,
      observationId: this.options.observationId + suffix, revisionId: this.options.revisionId + suffix });
  }
  private deliver(window: AmbientEncounterWindow, endedAtMs: number): Promise<DurableObservationV1> {
    if (this.pending.size >= 2) {
      window.discard(); this.fail();
      const failure = Promise.reject<DurableObservationV1>(new Error("Observation delivery backpressure"));
      // Rotation can run from a synchronous media callback without an awaiting caller.
      void failure.catch(() => {});
      return failure;
    }
    let timeout: ReturnType<typeof setTimeout> | undefined;
    const abort = new AbortController(); this.deliveries.add(abort);
    const work = window.finalize(endedAtMs, endedAtMs, abort.signal).then(async observation => {
      if (!this.discarded && !abort.signal.aborted) await this.options.onObservation?.(observation, abort.signal);
      return observation;
    });
    const delivery = Promise.race([work, new Promise<DurableObservationV1>((_resolve, reject) => {
      timeout = setTimeout(() => reject(new Error("Observation delivery timed out")), this.options.deliveryTimeoutMs ?? 10_000);
    })]);
    this.pending.add(delivery);
    // Attach rejection handling even when rotation was triggered from a synchronous frame callback.
    void delivery.then(() => {}, () => this.fail()).finally(() => {
      if (timeout) clearTimeout(timeout);
      this.deliveries.delete(abort);
      this.pending.delete(delivery);
    });
    return delivery;
  }
  private fail() {
    if (this.deliveryFailed) return;
    this.deliveryFailed = true; this.discard();
    try { this.options.onFailure?.(); } catch { /* host error callback */ }
  }
}
