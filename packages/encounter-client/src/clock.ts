import { EncounterClockCalibrationV1Schema, type EncounterClockCalibrationV1, type EncounterClockSampleV1 } from "@phenometrix/contracts";

export interface ClockProbe {
  sample: EncounterClockSampleV1;
  localSentAtMs: number;
  localReceivedAtMs: number;
  elapsedMs: number;
}

/** Engineering bound: full round trip, monitored server UTC error, 2 ms
 * timestamp quantization, and 100 ppm local drift over at most 60 seconds.
 * A deployment must qualify these assumptions on its supported devices.
 */
export function calibrationFromProbe(probe: ClockProbe): EncounterClockCalibrationV1 | null {
  const { sample, localSentAtMs, localReceivedAtMs, elapsedMs } = probe;
  if (!sample.source) return null;
  const wallElapsed = localReceivedAtMs - localSentAtMs;
  if (![localSentAtMs, localReceivedAtMs, elapsedMs].every(Number.isFinite) || elapsedMs < 0 || elapsedMs > 2_000 ||
    wallElapsed < 0 || Math.abs(wallElapsed - elapsedMs) > 10) throw new Error("encounter-clock-discontinuity");
  const serverElapsed = sample.serverSentAtMs - sample.serverReceivedAtMs;
  if (serverElapsed < 0 || serverElapsed > elapsedMs + 2) throw new Error("encounter-clock-invalid-round-trip");
  const utcOffsetMs = ((sample.serverReceivedAtMs - localSentAtMs) + (sample.serverSentAtMs - localReceivedAtMs)) / 2;
  const uncertaintyMs = Math.ceil(Math.max(elapsedMs, wallElapsed) / 2 + sample.source.maximumUtcErrorMs + 8);
  // Account for uncertainty when translating the source's expiration to local time.
  const localExpiresAtMs = Math.min(localReceivedAtMs + 60_000, sample.source.validUntilMs - utcOffsetMs - uncertaintyMs);
  if (localExpiresAtMs <= localReceivedAtMs) return null;
  return EncounterClockCalibrationV1Schema.parse({ localMeasuredAtMs: localReceivedAtMs, localExpiresAtMs,
    utcOffsetMs, uncertaintyMs, sourceId: sample.source.sourceId, sourceKind: sample.source.kind, measuredRoundTripMs: elapsedMs });
}

/** Inconsistent sources or disjoint offset bounds cannot establish a clock. */
export function selectClockCalibration(samples: EncounterClockCalibrationV1[], localNowMs = samples.at(-1)?.localMeasuredAtMs ?? 0): EncounterClockCalibrationV1 | null {
  if (!samples.length) return null;
  const first = samples[0];
  if (samples.some(sample => sample.sourceId !== first.sourceId || sample.sourceKind !== first.sourceKind) ||
    Math.max(...samples.map(sample => sample.utcOffsetMs - sample.uncertaintyMs)) > Math.min(...samples.map(sample => sample.utcOffsetMs + sample.uncertaintyMs)))
    throw new Error("encounter-clock-inconsistent-source");
  const current = samples.filter(sample => sample.localMeasuredAtMs <= localNowMs && sample.localExpiresAtMs > localNowMs);
  return current.length ? current.reduce((best, sample) => sample.uncertaintyMs < best.uncertaintyMs ? sample : best) : null;
}
