import { afterEach, describe, expect, it } from "vitest";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { createClockSourceProvider } from "./clock-source.js";

const directories: string[] = [];
afterEach(async () => { await Promise.all(directories.splice(0).map(path => rm(path, { recursive: true, force: true }))); });
const request = () => ({ signal: AbortSignal.timeout(1000) });

describe("trusted server-clock source", () => {
  it("labels synthetic time explicitly and never invents live clock accuracy", async () => {
    expect(await createClockSourceProvider({ mode: "synthetic", now: () => 1000 })(request())).toEqual({ sourceId: "synthetic-local-clock", kind: "synthetic", maximumUtcErrorMs: 0, validUntilMs: 31_000 });
    expect(await createClockSourceProvider({ mode: "live" })(request())).toBeNull();
    expect(await createClockSourceProvider({ mode: "live", attestationPath: "/missing/phenometrix-clock" })(request())).toBeNull();
  });
  it("reloads bounded monitor attestations and rejects expired, future or invalid claims", async () => {
    const directory = await mkdtemp(join(tmpdir(), "phenometrix-clock-")); directories.push(directory);
    const path = join(directory, "attestation.json");
    let now = 1000;
    const provider = createClockSourceProvider({ mode: "live", attestationPath: path, now: () => now });
    const valid = { sourceId: "clinic-ntp-monitor", validatedAtMs: 500, expiresAtMs: 1500, maximumUtcErrorMs: 8 };
    await writeFile(path, JSON.stringify(valid));
    expect(await provider(request())).toEqual({ sourceId: "clinic-ntp-monitor", kind: "monitored-utc", maximumUtcErrorMs: 8, validUntilMs: 1500 });
    now = 1500; expect(await provider(request())).toBeNull(); now = 1000;
    for (const invalid of [{ ...valid, validatedAtMs: 1200 }, { ...valid, expiresAtMs: 61_001 }, { ...valid, maximumUtcErrorMs: -1 }, { ...valid, extra: "untrusted" }]) {
      await writeFile(path, JSON.stringify(invalid)); expect(await provider(request())).toBeNull();
    }
    await writeFile(path, "x".repeat(8193)); expect(await provider(request())).toBeNull();
    await writeFile(path, "{"); expect(await provider(request())).toBeNull();
    await writeFile(path, JSON.stringify({ ...valid, sourceId: "replacement-monitor" }));
    expect((await provider(request()))?.sourceId).toBe("replacement-monitor");
  });
});
