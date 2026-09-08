import { describe, it, expect, vi, afterEach } from "vitest";
import { EncounterClockCalibrationV1Schema, EncounterClockSampleV1Schema, type EncounterClockSampleV1 } from "@phenometrix/contracts";
import { calibrationFromProbe, selectClockCalibration } from "./clock.js";
import { createEncounterClient } from "./index.js";

const scope = { tenantId: "tenant", studyId: "study", participantId: "patient" };
const sample: EncounterClockSampleV1 = {
  schemaVersion: "phenometric.encounter-clock-sample.v1", scope, episodeId: "episode", encounterId: "encounter",
  serverReceivedAtMs: 1_070, serverSentAtMs: 1_080,
  source: { sourceId: "utc-monitor", kind: "monitored-utc", maximumUtcErrorMs: 5, validUntilMs: 31_000 }
};
const probe = { sample, localSentAtMs: 1_000, localReceivedAtMs: 1_050, elapsedMs: 50 };
afterEach(() => vi.restoreAllMocks());
describe("automatic clock qualification", () => {
  it("bounds UTC offset without assuming symmetric transport or perfect server time", () => {
    const result = calibrationFromProbe(probe)!;
    expect(result).toMatchObject({ utcOffsetMs: 50, uncertaintyMs: 38, measuredRoundTripMs: 50, sourceKind: "monitored-utc" });
    expect(result.localExpiresAtMs).toBe(30_912);
    expect(result.localExpiresAtMs + result.utcOffsetMs + result.uncertaintyMs).toBe(sample.source!.validUntilMs);
  });
  it("never derives accuracy from unmonitored server wall time", () => {
    expect(calibrationFromProbe({ ...probe, sample: { ...sample, source: null } })).toBeNull();
    expect(calibrationFromProbe({ ...probe, sample: { ...sample, source: { ...sample.source!, validUntilMs: 1_080 } } })).toBeNull();
  });
  it.each([
    { localReceivedAtMs: 2_000 }, { elapsedMs: -1 }, { elapsedMs: 2_001 }, { elapsedMs: Number.NaN },
    { sample: { ...sample, serverSentAtMs: 1_069 } }, { sample: { ...sample, serverSentAtMs: 1_200 } }
  ])("rejects clock steps or physically inconsistent probes %j", change => {
    expect(() => calibrationFromProbe({ ...probe, ...change })).toThrow();
  });
  it("chooses the narrowest consistent bound and rejects disjoint or mixed sources", () => {
    const first = calibrationFromProbe(probe)!;
    const narrow = { ...first, utcOffsetMs: first.utcOffsetMs + 1, uncertaintyMs: 20 };
    expect(selectClockCalibration([first, narrow])).toEqual(narrow);
    expect(() => selectClockCalibration([first, { ...narrow, utcOffsetMs: 500 }])).toThrow("inconsistent");
    expect(() => selectClockCalibration([first, { ...narrow, sourceKind: "synthetic" }])).toThrow("inconsistent");
    expect(selectClockCalibration([])).toBeNull();
    expect(selectClockCalibration([{ ...first, localExpiresAtMs: 1_500 }, { ...narrow, localMeasuredAtMs: 1_600 }], 1_800)).toEqual({ ...narrow, localMeasuredAtMs: 1_600 });
    expect(selectClockCalibration([first], first.localExpiresAtMs)).toBeNull();
  });
  it("rejects invalid sample chronology, source expiry and overlong local validity in shared contracts", () => {
    expect(EncounterClockSampleV1Schema.safeParse({ ...sample, serverSentAtMs: 0 }).success).toBe(false);
    expect(EncounterClockSampleV1Schema.safeParse({ ...sample, source: { ...sample.source!, validUntilMs: 1_000 } }).success).toBe(false);
    const valid = calibrationFromProbe(probe)!;
    expect(EncounterClockCalibrationV1Schema.safeParse({ ...valid, localExpiresAtMs: valid.localMeasuredAtMs + 60_001 }).success).toBe(false);
  });
  it("authenticates three probes, binds the exact encounter, and adjusts consent checks to bounded UTC", async () => {
    const wall = 1_800_000_000_000;
    vi.spyOn(Date, "now").mockReturnValue(wall);
    vi.spyOn(performance, "now").mockReturnValue(100);
    const ref = { id: "protocol", version: "1", contentSha256: "a".repeat(64) };
    const fetcher = vi.fn(async (url: RequestInfo | URL, options?: RequestInit) => {
      expect(options?.headers).toMatchObject({ Authorization: "Bearer host-token" });
      expect(options?.cache).toBe("no-store");
      const payload = String(url).endsWith("/clock") ? { ...sample, serverReceivedAtMs: wall + 60_000, serverSentAtMs: wall + 60_000,
        source: { ...sample.source!, validUntilMs: wall + 90_000 } } : {
        scope, episodeId: "episode", inputRevision: 1, measurementProtocolRef: ref,
        consent: { consentId: "consent", scope, documentRef: ref, grantedAt: new Date(wall + 30_000).toISOString(), expiresAt: null, withdrawnAt: null,
          permissions: { capture: true, derivedAnalysis: true, derivedRetention: true, researchClips: false, modalities: ["face"] } },
        binding: { bindingId: "binding", scope, encounterId: "encounter", platformParticipantId: "platform", status: "verified", verificationMethod: "authenticated-patient-portal", recordedAt: new Date(wall + 30_000).toISOString(), revokedAt: null }
      };
      return new Response(JSON.stringify(payload));
    });
    const client = createEncounterClient({ baseUrl: "https://clinic.example", episodeId: "episode", scope, accessToken: async () => "host-token", fetch: fetcher as typeof fetch });
    expect(await client.calibrateClock("encounter")).toMatchObject({ utcOffsetMs: 60_000, sourceId: "utc-monitor" });
    expect(fetcher).toHaveBeenCalledTimes(3);
    await expect(client.captureContext("encounter")).resolves.toMatchObject({ encounterId: "encounter" });
    fetcher.mockImplementation(async () => new Response(JSON.stringify({ ...sample, encounterId: "caregiver" })));
    await expect(client.calibrateClock("encounter")).rejects.toThrow("scope-mismatch");
  });
  it("honors withdrawal during a probe even if the network ignores its signal", async () => {
    const abort = new AbortController();
    const client = createEncounterClient({ baseUrl: "https://clinic.example", episodeId: "episode", scope, accessToken: async () => "token", fetch: vi.fn(() => new Promise(() => {})) as typeof fetch });
    const pending = client.calibrateClock("encounter", abort.signal); abort.abort();
    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
  });
});
