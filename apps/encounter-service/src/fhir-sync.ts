import { authorize, type Principal } from "./auth.js";
import { invariant, ServiceError } from "./errors.js";
import type { EncounterService } from "./service.js";
import { withinDeadline } from "./deadline.js";

export interface FhirReadSource {
  /** Deployment configuration, never a URL accepted from a patient request. */
  baseUrl: string;
  accessToken: () => Promise<string>;
  fetch?: typeof globalThis.fetch;
  maxPages?: number;
  maxResponseBytes?: number;
  timeoutMs?: number;
}

async function boundedJson(response: Response, maxBytes: number, signal: AbortSignal): Promise<Record<string, unknown>> {
  invariant(response.body && response.ok, 502, "fhir-unavailable", "The configured clinical source is unavailable.");
  const reader = response.body.getReader(); const chunks: Uint8Array[] = []; let size = 0;
  try {
    for (;;) {
      const { done, value } = await withinDeadline(reader.read(), signal); if (done) break;
      size += value.byteLength;
      if (size > maxBytes) { void reader.cancel().catch(() => {}); invariant(false, 502, "fhir-response-limit", "Clinical source response exceeds the configured limit."); }
      chunks.push(value);
    }
    const bytes = new Uint8Array(size); let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
    try {
      const result: unknown = JSON.parse(new TextDecoder().decode(bytes));
      invariant(result && typeof result === "object" && !Array.isArray(result), 502, "fhir-invalid-response", "Clinical source returned an invalid bundle.");
      return result as Record<string, unknown>;
    } finally { bytes.fill(0); }
  } finally { if (signal.aborted) void reader.cancel().catch(() => {}); reader.releaseLock(); for (const chunk of chunks) chunk.fill(0); }
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
  const timeoutMs = source.timeoutMs ?? 15_000;
  invariant(Number.isSafeInteger(timeoutMs) && timeoutMs > 0 && timeoutMs <= 120_000, 500, "fhir-source-config", "Clinical source timeout must be between 1 and 120000 milliseconds.");
  // One deadline covers token retrieval, all pages and streamed response bodies.
  const signal = AbortSignal.timeout(timeoutMs);
  const token = await withinDeadline(source.accessToken(), signal);
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
      signal.throwIfAborted();
      const response = await withinDeadline(request(next, { headers: { authorization: `Bearer ${token}`, accept: "application/fhir+json" },
        redirect: "error", signal }), signal);
      const bundle = await boundedJson(response, maxBytes, signal);
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
  /** The host may revalidate its signed session/membership before each attempt. */
  resolvePrincipal?: () => Promise<Principal>;
  maxAttempts?: number;
}) {
  const maxAttempts = input.maxAttempts ?? 3;
  invariant(Number.isSafeInteger(maxAttempts) && maxAttempts >= 1 && maxAttempts <= 5, 500, "fhir-retry-config", "Clinical synchronization permits one to five import attempts.");
  invariant(/^[A-Za-z0-9._:-]{1,140}$/.test(input.eventId), 422, "fhir-event-id", "A bounded source event identity is required.");
  const identity = structuredClone(input.principal);
  const canonicalBase = (value: string) => new URL(value.endsWith("/") ? value : `${value}/`).href;
  const authorizeAttempt = async () => {
    const principal = input.resolvePrincipal ? await withinDeadline(input.resolvePrincipal(), AbortSignal.timeout(2_000)) : structuredClone(input.principal);
    invariant(principal.sub === identity.sub && principal.tenantId === identity.tenantId, 403, "fhir-session-changed", "Synchronization cannot switch authenticated identity.");
    const current = await input.service.getEpisode(principal, input.episodeId);
    authorize(principal, "clinical:write", current.episode.subjectRef, current.episode.dataClass);
    invariant(["clinician", "integration"].includes(principal.role), 403, "clinical-authority", "Clinical synchronization requires an authorized clinical integration.");
    const patient = current.episode.metadata.fhirPatientReference;
    invariant(typeof patient === "string", 422, "fhir-binding-required", "The episode has no source-bound patient reference.");
    const configuredSource = input.service.getFhirSourceConfiguration(input.sourceId);
    invariant(canonicalBase(configuredSource.sourceSystem) === canonicalBase(input.source.baseUrl),
      500, "fhir-source-mismatch", "The fetch source must match the registered clinical provenance source.");
    return { principal, current, patient, configuredSource };
  };
  const initial = await authorizeAttempt();
  const bundle = await fetchFhirPatientContext(input.source, initial.patient);
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    // Re-resolve current scope, source binding and authorization after network I/O
    // and every conflict. Import itself checks them again before the transaction.
    const current = await authorizeAttempt();
    invariant(current.patient === initial.patient && current.current.episode.subjectRef === initial.current.episode.subjectRef &&
      current.current.episode.metadata.studyId === initial.current.episode.metadata.studyId &&
      JSON.stringify(current.configuredSource) === JSON.stringify(initial.configuredSource), 409, "fhir-binding-changed", "Clinical source binding changed during synchronization.");
    try {
      return await input.service.importFhir(current.principal, input.episodeId,
        { source: input.sourceId, bundle, expectedRevision: current.current.inputRevision }, `source-event:${input.eventId}`);
    } catch (error) {
      if (!(error instanceof ServiceError) || error.code !== "stale-revision" || attempt + 1 === maxAttempts) throw error;
    }
  }
  throw new Error("unreachable-fhir-retry");
}
