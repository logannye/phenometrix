import { open } from "node:fs/promises";
import { z } from "zod";
import type { EncounterClockSampleV1 } from "@phenometrix/contracts";

export type EncounterClockSource = NonNullable<EncounterClockSampleV1["source"]>;
export type EncounterClockSourceProvider = (input: { signal: AbortSignal }) => Promise<EncounterClockSource | null>;

/** Written atomically by an external, trusted UTC synchronization monitor. */
export const ClockAttestationSchema = z.object({
  sourceId: z.string().min(1).max(160).regex(/^[A-Za-z0-9._:-]+$/),
  validatedAtMs: z.number().int().nonnegative().safe(),
  expiresAtMs: z.number().int().nonnegative().safe(),
  maximumUtcErrorMs: z.number().finite().nonnegative()
}).strict();

/** Reading the host clock alone never establishes its UTC accuracy. */
export function createClockSourceProvider(options: {
  mode: "synthetic" | "live"; attestationPath?: string; now?: () => number;
}): EncounterClockSourceProvider {
  const now = options.now ?? Date.now;
  if (options.mode === "synthetic") return async ({ signal }) => {
    signal.throwIfAborted();
    return { sourceId: "synthetic-local-clock", kind: "synthetic", maximumUtcErrorMs: 0, validUntilMs: now() + 30_000 };
  };
  return async ({ signal }) => {
    if (!options.attestationPath) return null;
    signal.throwIfAborted();
    let file: Awaited<ReturnType<typeof open>> | undefined;
    const buffer = Buffer.alloc(8193);
    try {
      file = await open(options.attestationPath, "r");
      const { bytesRead } = await file.read(buffer, 0, buffer.length, 0);
      signal.throwIfAborted();
      if (bytesRead > 8192) return null;
      const parsed = ClockAttestationSchema.safeParse(JSON.parse(buffer.subarray(0, bytesRead).toString("utf8")));
      if (!parsed.success) return null;
      const attestation = parsed.data, current = now();
      if (attestation.validatedAtMs > current || attestation.expiresAtMs <= current ||
        attestation.expiresAtMs <= attestation.validatedAtMs || attestation.expiresAtMs - attestation.validatedAtMs > 60_000) return null;
      return { sourceId: attestation.sourceId, kind: "monitored-utc", maximumUtcErrorMs: attestation.maximumUtcErrorMs, validUntilMs: attestation.expiresAtMs };
    } catch { return null; }
    finally { buffer.fill(0); await file?.close(); }
  };
}
