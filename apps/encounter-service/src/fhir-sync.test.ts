import { describe, it, expect, vi } from "vitest";
import { fetchFhirPatientContext, synchronizeFhirPatientContext, type FhirReadSource } from "./fhir-sync.js";
import { EncounterService } from "./service.js";
import { MemoryRepository } from "./repository.memory.js";
import { syntheticFixture, seedSyntheticEpisode, SYNTHETIC_PRINCIPAL, SYNTHETIC_EPISODE_ID, SYNTHETIC_SCOPE } from "./synthetic.js";
import { ServiceError } from "./errors.js";

function response(data: unknown) { return new Response(JSON.stringify(data), { headers: { "content-type": "application/fhir+json" } }); }
const empty = { resourceType: "Bundle", type: "searchset", entry: [] };
function source(fn: (url: URL, init?: RequestInit) => Response): FhirReadSource {
  return { baseUrl: "https://clinical.example/fhir", accessToken: async () => "synthetic-token",
    fetch: (async (url: string | URL | Request, init?: RequestInit) => fn(new URL(String(url)), init)) as typeof fetch };
}
async function synchronizationFixture() {
  const now = new Date().toISOString(), fixture = await syntheticFixture(now);
  const service = new EncounterService(new MemoryRepository(), { mode: "synthetic", fhirSources: {
    configured: { sourceSystem: "https://clinical.example/fhir", treatmentCodes: [{ system: "urn:synthetic", code: "toxin", kind: "botulinum-injection", product: "Synthetic product" }] }
  } });
  await seedSyntheticEpisode(service, now);
  let reads = 0;
  const clinicalSource = source(url => {
    reads++;
    return response(url.pathname.endsWith("MedicationAdministration") ? { ...empty, entry: [{ resource: {
      resourceType: "MedicationAdministration", id: "synthetic-admin", meta: { versionId: "1", lastUpdated: now }, status: "completed",
      subject: { reference: "Patient/synthetic-patient" }, medicationCodeableConcept: { coding: [{ system: "urn:synthetic", code: "toxin" }] }, effectiveDateTime: fixture.treatment.effectiveTime.earliest
    } }] } : empty);
  });
  const input = { service, principal: SYNTHETIC_PRINCIPAL, episodeId: SYNTHETIC_EPISODE_ID, sourceId: "configured", source: clinicalSource, eventId: "synthetic-event" };
  const capture = async (index: number) => {
    const observation = { ...fixture.sessions[0]!.observation, observationId: `concurrent-${index}`, revisionId: `concurrent-${index}-v1` };
    await service.write(SYNTHETIC_PRINCIPAL, SYNTHETIC_EPISODE_ID, "sessions", { data: observation }, observation.revisionId);
  };
  return { service, input, capture, reads: () => reads };
}
describe("automatic clinical-source connector", () => {
  it("reads patient-scoped structured resources using existing integration credentials", async () => {
    const requests: URL[] = [];
    const value = await fetchFhirPatientContext(source((url, init) => {
      requests.push(url); expect(init?.redirect).toBe("error"); expect(init?.headers).toMatchObject({ authorization: "Bearer synthetic-token" });
      return response(empty);
    }), "Patient/synthetic");
    expect(requests).toHaveLength(5); expect(requests.every(url => url.searchParams.get("patient") === "Patient/synthetic")).toBe(true);
    expect(value.entry).toEqual([]);
  });
  it("rejects pagination to a different origin before sending credentials", async () => {
    let calls = 0;
    await expect(fetchFhirPatientContext(source(() => { calls++; return response({ ...empty, link: [{ relation: "next", url: "https://attacker.example/fhir" }] }); }), "Patient/synthetic")).rejects.toThrow();
    expect(calls).toBe(1);
  });
  it("bounds responses, pagination loops and invalid patient references", async () => {
    await expect(fetchFhirPatientContext({ ...source(() => response(empty)), maxResponseBytes: 8 }, "Patient/synthetic")).rejects.toThrow();
    await expect(fetchFhirPatientContext(source(url => response({ ...empty, link: [{ relation: "next", url: url.href }] })), "Patient/synthetic")).rejects.toThrow();
    await expect(fetchFhirPatientContext(source(() => response(empty)), "Patient/x?other=y")).rejects.toThrow();
  });
  it("cannot label records fetched from a different EHR as the registered source", async () => {
    const service = new EncounterService(new MemoryRepository(), { mode: "synthetic", fhirSources: {
      configured: { sourceSystem: "https://registered.example/fhir", treatmentCodes: [] }
    } });
    await service.createEpisode(SYNTHETIC_PRINCIPAL, (await syntheticFixture()).episode);
    let requests = 0;
    await expect(synchronizeFhirPatientContext({ service, principal: SYNTHETIC_PRINCIPAL, episodeId: SYNTHETIC_EPISODE_ID,
      sourceId: "configured", eventId: "test-event", source: source(() => { requests++; return response(empty); })
    })).rejects.toMatchObject({ code: "fhir-source-mismatch" });
    expect(requests).toBe(0);
  });
  it("retries a concurrent capture conflict without refetching or duplicating the source event", async () => {
    const f = await synchronizationFixture(), original = f.service.importFhir.bind(f.service);
    let attempts = 0;
    const imports = vi.spyOn(f.service, "importFhir").mockImplementation(async (...args) => {
      if (++attempts === 1) await f.capture(attempts);
      return original(...args);
    });
    const result = await synchronizeFhirPatientContext(f.input);
    expect(attempts).toBe(2); expect(f.reads()).toBe(5); expect(result.records).toHaveLength(1);
    expect(imports.mock.calls.map(call => call[3])).toEqual(["source-event:synthetic-event", "source-event:synthetic-event"]);
    const state = await f.service.repository.getState(SYNTHETIC_SCOPE.tenantId, SYNTHETIC_EPISODE_ID);
    const imported = state!.records.filter(record => record.idempotencyKey.startsWith("fhir:"));
    expect(imported).toHaveLength(1); expect(imported[0]!.sourceProvenance?.versionId).toBe("1");
    expect((await synchronizeFhirPatientContext(f.input)).duplicate).toBe(true);
  });
  it("revalidates membership after a conflict and never retries revoked authorization", async () => {
    const f = await synchronizationFixture(), original = f.service.importFhir.bind(f.service);
    let validations = 0;
    const imports = vi.spyOn(f.service, "importFhir").mockImplementation(async (...args) => {
      await f.capture(1); return original(...args);
    });
    await expect(synchronizeFhirPatientContext({ ...f.input, resolvePrincipal: async () => {
      validations++;
      return { ...SYNTHETIC_PRINCIPAL, permissions: validations > 2 ? ["evidence:read"] : SYNTHETIC_PRINCIPAL.permissions };
    } })).rejects.toMatchObject({ status: 403, code: "permission-denied" });
    expect(imports).toHaveBeenCalledTimes(1); expect(validations).toBe(3); expect(f.reads()).toBe(5);
  });
  it("bounds revision-conflict retries and leaves a clear retryable failure", async () => {
    const f = await synchronizationFixture(), original = f.service.importFhir.bind(f.service);
    let attempts = 0;
    vi.spyOn(f.service, "importFhir").mockImplementation(async (...args) => {
      await f.capture(++attempts); return original(...args);
    });
    await expect(synchronizeFhirPatientContext(f.input)).rejects.toMatchObject({ status: 409, code: "stale-revision" });
    expect(attempts).toBe(3); expect(f.reads()).toBe(5);
    expect((await f.service.repository.getState(SYNTHETIC_SCOPE.tenantId, SYNTHETIC_EPISODE_ID))!.records.filter(record => record.idempotencyKey.startsWith("fhir:"))).toHaveLength(0);
  });
  it("does not retry authorization or other reconciliation errors", async () => {
    for (const error of [new ServiceError(403, "permission-denied", "Denied"), new ServiceError(409, "fhir-source-order", "Reconcile")]) {
      const f = await synchronizationFixture();
      const imports = vi.spyOn(f.service, "importFhir").mockRejectedValue(error);
      await expect(synchronizeFhirPatientContext(f.input)).rejects.toBe(error);
      expect(imports).toHaveBeenCalledTimes(1);
    }
  });
  it("bounds source token and body waits even when custom dependencies ignore cancellation", async () => {
    await expect(fetchFhirPatientContext({ ...source(() => response(empty)), accessToken: () => new Promise(() => {}), timeoutMs: 10 }, "Patient/synthetic")).rejects.toMatchObject({ name: "TimeoutError" });
    await expect(fetchFhirPatientContext({ ...source(() => new Response(new ReadableStream())), timeoutMs: 10 }, "Patient/synthetic")).rejects.toMatchObject({ name: "TimeoutError" });
  });
});
