import { z } from "zod";
import { TreatmentResponseScopeV1Schema } from "./treatment-response.js";

const Id = z.string().min(1).max(160).regex(/^[A-Za-z0-9._:-]+$/);
const Milliseconds = z.number().finite().nonnegative().max(Number.MAX_SAFE_INTEGER);
export const EncounterClockSourceV1Schema = z.object({
  sourceId: Id,
  kind: z.enum(["synthetic", "monitored-utc"])
}).strict();
export type EncounterClockSourceV1 = z.infer<typeof EncounterClockSourceV1Schema>;

/** A timestamp source must declare a current external bound on its UTC error. */
export const EncounterClockSampleV1Schema = z.object({
  schemaVersion: z.literal("phenometric.encounter-clock-sample.v1"),
  scope: TreatmentResponseScopeV1Schema, episodeId: Id, encounterId: Id,
  serverReceivedAtMs: Milliseconds, serverSentAtMs: Milliseconds,
  source: EncounterClockSourceV1Schema.extend({
    maximumUtcErrorMs: Milliseconds, validUntilMs: Milliseconds
  }).nullable()
}).strict().superRefine((sample, ctx) => {
  if (sample.serverSentAtMs < sample.serverReceivedAtMs || (sample.source && sample.source.validUntilMs <= sample.serverSentAtMs))
    ctx.addIssue({ code: "custom", message: "Clock sample chronology or source validity is invalid." });
});
export type EncounterClockSampleV1 = z.infer<typeof EncounterClockSampleV1Schema>;

/** Local timestamps retain the caller's wall clock; offset maps them to estimated UTC. */
export const EncounterClockCalibrationV1Schema = z.object({
  localMeasuredAtMs: Milliseconds, localExpiresAtMs: Milliseconds,
  utcOffsetMs: z.number().finite(), uncertaintyMs: Milliseconds,
  sourceId: Id, sourceKind: z.enum(["synthetic", "monitored-utc"]),
  measuredRoundTripMs: Milliseconds
}).strict().superRefine((value, ctx) => {
  if (value.localExpiresAtMs <= value.localMeasuredAtMs || value.localExpiresAtMs - value.localMeasuredAtMs > 60_000)
    ctx.addIssue({ code: "custom", message: "Clock calibration must have a positive validity period of at most one minute." });
});
export type EncounterClockCalibrationV1 = z.infer<typeof EncounterClockCalibrationV1Schema>;
