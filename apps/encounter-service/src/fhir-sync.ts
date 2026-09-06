import { authorize, type Principal } from "./auth.js";
import { invariant } from "./errors.js";
import type { EncounterService } from "./service.js";

export interface FhirReadSource {
  /** Deployment configuration, never a URL accepted from a patient request. */
  baseUrl: string;
  accessToken: () => Promise<string>;
  fetch?: typeof globalThis.fetch;
  maxPages?: number;
  maxResponseBytes?: number;
  timeoutMs?: number;
}

async function boundedJson(response: Response, maxBytes: number): Promise<Record<string, unknown>> {
  invariant(response.body && response.ok, 502, "fhir-unavailable", "The configured clinical source is unavailable.");
  const reader = response.body.getReader(); const chunks: Uint8Array[] = []; let size = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read(); if (done) break;
      size += value.byteLength;
      if (size > maxBytes) { await reader.cancel(); invariant(false, 502, "fhir-response-limit", "Clinical source response exceeds the configured limit."); }
      chunks.push(value);
    }
    const bytes = new Uint8Array(size); let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
    try {
      const result: unknown = JSON.parse(new TextDecoder().decode(bytes));
      invariant(result && typeof result === "object" && !Array.isArray(result), 502, "fhir-invalid-response", "Clinical source returned an invalid bundle.");
      return result as Record<string, unknown>;
    } finally { bytes.fill(0); }
  } finally { reader.releaseLock(); for (const chunk of chunks) chunk.fill(0); }
}

/** Read-only structured source connector. No narrative extraction or patient questions. */
export async function fetchFhirPatientContext(source: FhirReadSource, patientReference: string): Promise<Record<string, unknown>> {
  const base = new URL(source.baseUrl.endsWith("/") ? source.baseUrl : `${source.baseUrl}/`);
  invariant(base.protocol === "https:" && !base.username && !base.password && !base.search && !base.hash,
    500, "fhir-source-config", "The clinical source requires an HTTPS base URL.");
  invariant(/^Patient\/[A-Za-z0-9.-]{1,64}$/.test(patientReference), 422, "fhir-patient-reference", "A source-bound Patient reference is required.");
  const maxPages = source.maxPages ?? 20, maxBytes = source.maxResponseBytes ?? 2_000_000;
  invariant(Number.isSafeInteger(maxPages) && maxPages > 0 && Number.isSafeInteger(maxBytes) && maxBytes > 0,
    500, "fhir-source-config", "Clinical source limits must be positive integers.");
  const token = await source.accessToken();
  invariant(token.length > 0 && !/[\r\n]/.test(token), 500, "fhir-source-auth", "Clinical source authentication is unavailable.");
  const entries: unknown[] = []; let pages = 0;
  const visited = new Set<string>();
  const allowed = (url: URL) => url.origin === base.origin && url.pathname.startsWith(base.pathname)
    && !url.username && !url.password && !url.hash;
  const request = source.fetch ?? globalThis.fetch;
  for (const type of ["MedicationAdministration", "Procedure", "MedicationRequest", "MedicationStatement", "Encounter"]) {
    let next: URL | null = new URL(type, base);
    next.searchParams.set("patient", patientReference); next.searchParams.set("_count", "100");
    while (next) {
      invariant(allowed(next) && !visited.has(next.href), 502, "fhir-pagination-invalid", "Clinical pagination left its configured source or repeated a page.");
      invariant(++pages <= maxPages, 502, "fhir-page-limit", "Clinical source pagination exceeded the configured limit.");
      visited.add(next.href);
      const response = await request(next, { headers: { authorization: `Bearer ${token}`, accept: "application/fhir+json" },
        redirect: "error", signal: AbortSignal.timeout(source.timeoutMs ?? 15_000) });
      const bundle = await boundedJson(response, maxBytes);
      invariant(bundle.resourceType === "Bundle" && (bundle.entry === undefined || Array.isArray(bundle.entry)),
        502, "fhir-invalid-response", "Clinical source returned an invalid bundle.");
      entries.push(...(bundle.entry as unknown[] ?? []));
      const links = Array.isArray(bundle.link) ? bundle.link as Record<string, unknown>[] : [];
      const nextLinks = links.filter(link => link.relation === "next");
      invariant(nextLinks.length <= 1 && (!nextLinks.length || typeof nextLinks[0].url === "string"),
        502, "fhir-pagination-invalid", "Clinical pagination links are ambiguous.");
      next = nextLinks.length ? new URL(nextLinks[0].url as string, base) : null;
    }
  }
  return { resourceType: "Bundle", type: "collection", entry: entries };
}

/** Call from the clinic's existing encounter/record-change event handler. */
export async function synchronizeFhirPatientContext(input: {
  service: EncounterService; principal: Principal; episodeId: string; sourceId: string;
  source: FhirReadSource; eventId: string;
}) {
  const current = await input.service.getEpisode(input.principal, input.episodeId);
  authorize(input.principal, "clinical:write", current.episode.subjectRef, current.episode.dataClass);
  invariant(["clinician", "integration"].includes(input.principal.role), 403, "clinical-authority", "Clinical synchronization requires an authorized clinical integration.");
  const patient = current.episode.metadata.fhirPatientReference;
  invariant(typeof patient === "string", 422, "fhir-binding-required", "The episode has no source-bound patient reference.");
  const configuredSource = input.service.getFhirSourceConfiguration(input.sourceId);
  const canonicalBase = (value: string) => new URL(value.endsWith("/") ? value : `${value}/`).href;
  invariant(canonicalBase(configuredSource.sourceSystem) === canonicalBase(input.source.baseUrl),
    500, "fhir-source-mismatch", "The fetch source must match the registered clinical provenance source.");
  const bundle = await fetchFhirPatientContext(input.source, patient);
  return input.service.importFhir(input.principal, input.episodeId,
    { source: input.sourceId, bundle, expectedRevision: current.inputRevision }, `source-event:${input.eventId}`);
}
