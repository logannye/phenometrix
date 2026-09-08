import {
  DerivedDataConsentV1Schema, ParticipantBindingV1Schema, DurableObservationV1Schema,
  TreatmentResponseScopeV1Schema, VersionedArtifactRefV1Schema,
  TreatmentResponseRunV1Schema, TreatmentResponseReviewV1Schema, canonicalTreatmentResponseJson, verifyTreatmentResponseArtifact,
  EncounterClockSampleV1Schema, type EncounterClockCalibrationV1,
  type DurableObservationV1, type TreatmentResponseScopeV1
} from "@phenometrix/contracts";
import { calibrationFromProbe, selectClockCalibration } from "./clock.js";
export { calibrationFromProbe, selectClockCalibration } from "./clock.js";

export interface EncounterClientOptions {
  baseUrl: string; episodeId: string; scope: TreatmentResponseScopeV1;
  /** Reuse the host's existing signed scoped session. This client never displays login UI. */
  accessToken: () => Promise<string>;
  fetch?: typeof fetch; timeoutMs?: number;
}
function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("encounter-invalid-response");
  return value as Record<string, unknown>;
}
function id(value: string): string {
  if (!/^[A-Za-z0-9._:-]{1,160}$/.test(value)) throw new Error("encounter-invalid-id"); return value;
}
function withinDeadline<T>(promise: Promise<T>, signal: AbortSignal): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const aborted = () => reject(signal.reason);
    signal.addEventListener("abort", aborted, { once: true });
    promise.then(value => {
      signal.removeEventListener("abort", aborted);
      if (signal.aborted) reject(signal.reason); else resolve(value);
    }, error => { signal.removeEventListener("abort", aborted); reject(error); });
    if (signal.aborted) { signal.removeEventListener("abort", aborted); aborted(); }
  });
}

export function createEncounterClient(options: EncounterClientOptions) {
  const base = new URL(options.baseUrl.endsWith("/") ? options.baseUrl : `${options.baseUrl}/`);
  if ((base.protocol !== "https:" && !(base.protocol === "http:" && ["127.0.0.1", "localhost", "[::1]"].includes(base.hostname))) || base.username || base.password || base.search || base.hash)
    throw new Error("encounter-invalid-service-url");
  const episodeId = id(options.episodeId), prefix = `v1/episodes/${episodeId}`;
  const scope = TreatmentResponseScopeV1Schema.parse(options.scope);
  const { accessToken, fetch: fetcher = globalThis.fetch, timeoutMs = 10_000 } = options;
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs <= 0 || timeoutMs > 2_147_483_647) throw new Error("encounter-invalid-timeout");
  let latestRevision = -1;
  let clock: EncounterClockCalibrationV1 | null = null;
  const revision = (value: unknown): number => {
    if (!Number.isSafeInteger(value) || Number(value) < 0) throw new Error("encounter-invalid-revision");
    return Number(value);
  };
  const acceptRevision = (value: unknown) => {
    const current = revision(value);
    if (current < latestRevision) throw new Error("encounter-stale-evidence");
    latestRevision = current;
    return current;
  };
  const matches = (value: unknown) => canonicalTreatmentResponseJson(value) === canonicalTreatmentResponseJson(scope);
  const request = async (path: string, method: "GET" | "POST", body?: unknown, key?: string, signal?: AbortSignal) => {
    const timeout = AbortSignal.timeout(timeoutMs);
    const requestSignal = signal ? AbortSignal.any([signal, timeout]) : timeout;
    requestSignal.throwIfAborted();
    const token = await withinDeadline(accessToken(), requestSignal);
    if (!token || /[\r\n]/.test(token)) throw new Error("encounter-session-unavailable");
    const result = await withinDeadline(fetcher(new URL(path, base), {
      method, headers: { Authorization: `Bearer ${token}`, ...(body === undefined ? {} : { "Content-Type": "application/json" }), ...(key ? { "Idempotency-Key": key } : {}) },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      signal: requestSignal, redirect: "error", cache: "no-store"
    }), requestSignal);
    if (!result.ok) throw new Error(`encounter-request-${result.status}`);
    const payload = record(await withinDeadline(result.json(), requestSignal));
    return { payload, requestSignal };
  };
  return {
    /** Authenticated, scoped probes use the host session and show no UI. Missing
     * UTC monitoring remains unqualified; server wall time alone is insufficient. */
    async calibrateClock(encounterId: string, signal?: AbortSignal): Promise<EncounterClockCalibrationV1 | null> {
      const samples: EncounterClockCalibrationV1[] = [];
      const probeSignal = AbortSignal.any([AbortSignal.timeout(2_000), ...(signal ? [signal] : [])]);
      for (let index = 0; index < 3; index++) {
        const localSentAtMs = Date.now(), monotonicStart = performance.now();
        const { payload } = await request(`${prefix}/capture-context/${id(encounterId)}/clock`, "GET", undefined, undefined, probeSignal);
        const localReceivedAtMs = Date.now(), elapsedMs = performance.now() - monotonicStart;
        const sample = EncounterClockSampleV1Schema.parse(payload);
        if (!matches(sample.scope) || sample.episodeId !== episodeId || sample.encounterId !== encounterId) throw new Error("encounter-clock-scope-mismatch");
        const calibrated = calibrationFromProbe({ sample, localSentAtMs, localReceivedAtMs, elapsedMs });
        if (!calibrated) return null; // The previous bound may remain valid; it is never extended here.
        samples.push(calibrated);
      }
      probeSignal.throwIfAborted();
      clock = selectClockCalibration(samples, Date.now());
      return clock;
    },
    async captureContext(encounterId: string, signal?: AbortSignal) {
      const { payload: result, requestSignal } = await request(`${prefix}/capture-context/${id(encounterId)}`, "GET", undefined, undefined, signal);
      if (!matches(result.scope) || result.episodeId !== episodeId) throw new Error("encounter-scope-mismatch");
      revision(result.inputRevision);
      const consent = DerivedDataConsentV1Schema.parse(result.consent);
      const binding = ParticipantBindingV1Schema.parse(result.binding);
      if (!matches(consent.scope) || !matches(binding.scope) || binding.encounterId !== encounterId) throw new Error("encounter-binding-mismatch");
      const localNow = Date.now();
      const validClock = clock && localNow >= clock.localMeasuredAtMs && localNow < clock.localExpiresAtMs ? clock : null;
      const now = localNow + (validClock?.utcOffsetMs ?? 0), error = validClock?.uncertaintyMs ?? 0;
      if (!consent.permissions.capture || !consent.permissions.derivedAnalysis || !consent.permissions.derivedRetention || consent.withdrawnAt !== null ||
        Date.parse(consent.grantedAt) > now - error || (consent.expiresAt !== null && Date.parse(consent.expiresAt) <= now + error) ||
        binding.status !== "verified" || binding.revokedAt !== null || Date.parse(binding.recordedAt) > now - error) throw new Error("encounter-capture-unavailable");
      const measurementProtocolRef = VersionedArtifactRefV1Schema.parse(result.measurementProtocolRef);
      requestSignal.throwIfAborted();
      return { consent, binding, encounterId, inputRevision: acceptRevision(result.inputRevision), measurementProtocolRef };
    },
    async ingestObservation(value: DurableObservationV1, signal?: AbortSignal) {
      const observation = DurableObservationV1Schema.parse(value);
      if (!matches(observation.scope)) throw new Error("encounter-scope-mismatch");
      // A bounded UTC estimate can be slightly ahead of the server. Wait until
      // even its earliest possible UTC has reached recordedAt, rather than
      // weakening the service's immutable knowledge-time rule.
      const now = Date.now();
      if (clock && now >= clock.localMeasuredAtMs && now < clock.localExpiresAtMs) {
        const delayMs = Math.max(0, Math.ceil(Date.parse(observation.recordedAt) - (now + clock.utcOffsetMs - clock.uncertaintyMs)));
        if (delayMs > 2_000 || now + delayMs >= clock.localExpiresAtMs) throw new Error("encounter-clock-delivery-unavailable");
        if (delayMs > 0) {
          const delaySignal = AbortSignal.any([AbortSignal.timeout(timeoutMs), ...(signal ? [signal] : [])]);
          await new Promise<void>((resolve, reject) => {
            const aborted = () => { clearTimeout(timer); delaySignal.removeEventListener("abort", aborted); reject(delaySignal.reason); };
            const timer = setTimeout(() => { delaySignal.removeEventListener("abort", aborted); resolve(); }, delayMs);
            delaySignal.addEventListener("abort", aborted, { once: true });
            if (delaySignal.aborted) aborted();
          });
        }
      }
      const { payload: result, requestSignal } = await request(`${prefix}/sessions`, "POST", { data: observation }, id(observation.revisionId), signal);
      const saved = record(result.record);
      if (saved.episodeId !== episodeId || saved.kind !== "session" || saved.logicalId !== observation.observationId || saved.idempotencyKey !== observation.revisionId ||
        typeof result.duplicate !== "boolean" || canonicalTreatmentResponseJson(DurableObservationV1Schema.parse(saved.payload)) !== canonicalTreatmentResponseJson(observation)) throw new Error("encounter-ingest-mismatch");
      const savedRevision = revision(saved.revision);
      requestSignal.throwIfAborted();
      latestRevision = Math.max(latestRevision, savedRevision);
      return result;
    },
    /** Directly usable as the chart panel's load callback. Pending evidence stays empty. */
    async loadEvidence(signal?: AbortSignal) {
      const { payload: result, requestSignal } = await request(`${prefix}/evidence`, "GET", undefined, undefined, signal);
      const episode = record(result.episode), metadata = record(episode.metadata);
      if (episode.id !== episodeId || !matches({ tenantId: episode.tenantId, studyId: metadata.studyId, participantId: episode.subjectRef })) throw new Error("encounter-scope-mismatch");
      revision(result.inputRevision);
      if (episode.revision !== result.inputRevision) throw new Error("encounter-stale-evidence");
      if (!["ready", "pending", "failed", "empty"].includes(String(result.status))) throw new Error("encounter-invalid-status");
      if (result.status !== "ready") {
        if (result.snapshot !== null) throw new Error("encounter-stale-evidence");
        requestSignal.throwIfAborted();
        acceptRevision(result.inputRevision);
        return { run: null, reviews: [] };
      }
      const snapshot = record(result.snapshot);
      if (snapshot.episodeId !== episodeId || snapshot.inputRevision !== result.inputRevision) throw new Error("encounter-stale-evidence");
      const run = TreatmentResponseRunV1Schema.parse(snapshot.analysis);
      if (!matches(run.scope)) throw new Error("encounter-scope-mismatch");
      const protocol = record(metadata.protocol), specification = record(metadata.specification);
      if (canonicalTreatmentResponseJson(run.protocolRef) !== canonicalTreatmentResponseJson({ id: protocol.protocolId, version: protocol.version, contentSha256: protocol.contentSha256 }) ||
        canonicalTreatmentResponseJson(run.specificationRef) !== canonicalTreatmentResponseJson({ id: specification.specificationId, version: specification.version, contentSha256: specification.contentSha256 })) throw new Error("encounter-provenance-mismatch");
      if (!(await withinDeadline(verifyTreatmentResponseArtifact(run), requestSignal))) throw new Error("encounter-run-integrity");
      if (!Array.isArray(result.reviews)) throw new Error("encounter-invalid-reviews");
      const reviews = result.reviews.map(review => TreatmentResponseReviewV1Schema.parse(review));
      if (reviews.some(review => !matches(review.scope))) throw new Error("encounter-scope-mismatch");
      if (reviews.some(review => review.runId === run.runId && review.runSha256 !== run.contentSha256)) throw new Error("encounter-review-mismatch");
      if (new Set(reviews.map(review => review.reviewId)).size !== reviews.length || reviews.some(review => review.runId === run.runId && Date.parse(review.recordedAt) < Date.parse(run.generatedAt))) throw new Error("encounter-invalid-reviews");
      requestSignal.throwIfAborted();
      acceptRevision(result.inputRevision);
      return { run, reviews: reviews.filter(review => review.runId === run.runId && review.runSha256 === run.contentSha256) };
    }
  };
}
export type EncounterClient = ReturnType<typeof createEncounterClient>;
