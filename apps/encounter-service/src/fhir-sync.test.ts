import { describe, it, expect } from "vitest";
import { fetchFhirPatientContext, synchronizeFhirPatientContext, type FhirReadSource } from "./fhir-sync.js";
import { EncounterService } from "./service.js";
import { MemoryRepository } from "./repository.memory.js";
import { syntheticFixture, SYNTHETIC_PRINCIPAL, SYNTHETIC_EPISODE_ID } from "./synthetic.js";

function response(data: unknown) { return new Response(JSON.stringify(data), { headers: { "content-type": "application/fhir+json" } }); }
const empty = { resourceType: "Bundle", type: "searchset", entry: [] };
function source(fn: (url: URL, init?: RequestInit) => Response): FhirReadSource {
  return { baseUrl: "https://clinical.example/fhir", accessToken: async () => "synthetic-token",
    fetch: (async (url: string | URL | Request, init?: RequestInit) => fn(new URL(String(url)), init)) as typeof fetch };
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
});
