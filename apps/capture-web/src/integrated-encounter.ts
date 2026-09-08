import type { EncounterClient } from "@phenometrix/encounter-client";
import { createEmbeddedEncounter, type EmbeddedEncounterOptions } from "./embedded-encounter.js";
import type { CaptureAuthorization } from "@phenometrix/encounter-capture";
import { AMBIENT_LOCAL_PROTOCOL_REF } from "@phenometrix/contracts";

export type HostEncounterEvent = "ended" | "consent-withdrawn" | "binding-lost" | "resource-pressure";
export interface IntegratedEncounterOptions extends Omit<EmbeddedEncounterOptions, "authorization" | "observationId" | "revisionId" | "onObservation" | "clockCalibration"> {
  client: EncounterClient;
  encounterId: string;
  /** Participant ID supplied by the existing telehealth host for this exact local stream. */
  platformParticipantId: string;
  modalities?: readonly ("face" | "voice")[];
  /** Includes the host's existing call-quality/resource-pressure signal. */
  subscribeLifecycle: (listener: (event: HostEncounterEvent) => void) => () => void;
  onStatus?: (status: "started" | "unavailable" | "delivered" | "delivery-failed" | "stopped") => void;
}

/**
 * Application composition for an existing telehealth host. All activity follows
 * host events and existing credentials; this entry point creates no interface.
 */
export async function startIntegratedEncounter(options: IntegratedEncounterOptions): Promise<{ stop(): void }> {
  const authorizationAbort = new AbortController();
  const deliveryAbort = new AbortController();
  let capture: ReturnType<typeof createEmbeddedEncounter> | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let clockExpiresAtMs: number | null = null;
  let terminated = false, refreshing = false, finishing = false, hasClock = false;
  let unsubscribe = () => {};
  const status = (value: Parameters<NonNullable<IntegratedEncounterOptions["onStatus"]>>[0]) => {
    try { options.onStatus?.(value); } catch { /* Host telemetry must not interrupt the visit. */ }
  };
  const terminate = (event: HostEncounterEvent) => {
    if (terminated) {
      if (finishing && event !== "ended") { deliveryAbort.abort(); capture?.withdraw(); }
      return;
    }
    terminated = true; authorizationAbort.abort(); if (timer) clearTimeout(timer);
    if (event === "ended") {
      // Finishing/delivery is asynchronous and does not delay the telehealth end.
      // Keep withdrawal subscribed until this bounded delivery has settled.
      finishing = true;
      void Promise.resolve(capture?.finish()).catch(() => { if (!deliveryAbort.signal.aborted) status("delivery-failed"); }).finally(() => {
        finishing = false;
        try { unsubscribe(); } catch { /* Host owns its event bus. */ }
      });
    } else {
      try { unsubscribe(); } catch { /* Host owns its event bus. */ }
      deliveryAbort.abort(); capture?.withdraw();
    }
    status("stopped");
  };
  const context = async (): Promise<CaptureAuthorization> => {
    const resolved = await options.client.captureContext(options.encounterId, authorizationAbort.signal);
    if (resolved.binding.platformParticipantId !== options.platformParticipantId) throw new Error("host-participant-mismatch");
    if (resolved.measurementProtocolRef.id !== AMBIENT_LOCAL_PROTOCOL_REF.packId
      || resolved.measurementProtocolRef.version !== AMBIENT_LOCAL_PROTOCOL_REF.version
      || resolved.measurementProtocolRef.contentSha256 !== AMBIENT_LOCAL_PROTOCOL_REF.contentSha256) throw new Error("host-measurement-protocol-unsupported");
    const requested = options.modalities ?? ["face"];
    if (requested.some(modality => !resolved.consent.permissions.modalities.includes(modality))) throw new Error("modality-not-consented");
    return { encounterId: options.encounterId, encounterActive: !terminated, binding: resolved.binding, consent: resolved.consent, modalities: requested };
  };
  try {
    unsubscribe = options.subscribeLifecycle(terminate);
    if (terminated) { unsubscribe(); return { stop: () => {} }; }
    const clockCalibration = await options.client.calibrateClock(options.encounterId, authorizationAbort.signal);
    hasClock = clockCalibration !== null;
    clockExpiresAtMs = clockCalibration?.localExpiresAtMs ?? null;
    const authorization = await context();
    if (terminated) return { stop: () => {} };
    const captureId = `capture-${crypto.randomUUID()}`;
    capture = createEmbeddedEncounter({ ...options, ...(clockCalibration ? { clockCalibration } : {}), authorization, observationId: captureId, revisionId: `${captureId}-v1`,
      onEvent: event => {
        if (event.type === "stopped" && event.reason !== "encounter-inactive") terminate("binding-lost");
        try { options.onEvent?.(event); } catch { /* Host telemetry is isolated. */ }
      },
      onObservation: async (observation, captureSignal) => {
        const signal = captureSignal ? AbortSignal.any([captureSignal, deliveryAbort.signal]) : deliveryAbort.signal;
        let failure: unknown;
        for (let attempt = 0; attempt < 2; attempt++) {
          signal.throwIfAborted();
          const attemptSignal = AbortSignal.any([signal, AbortSignal.timeout(2_500)]);
          try {
            await options.client.ingestObservation(observation, attemptSignal);
            attemptSignal.throwIfAborted(); status("delivered"); return;
          }
          catch (error) {
            failure = error;
            if (signal.aborted) throw error;
            const code = error instanceof Error ? /^encounter-request-(\d+)$/.exec(error.message)?.[1] : undefined;
            if (code === "401" || code === "403") { terminate("consent-withdrawn"); throw error; }
            if (code && Number(code) < 500 && code !== "408" && code !== "429") break;
            // Scope/integrity validation failures are not transient network errors.
            if (error instanceof Error && error.message.startsWith("encounter-") && !code) break;
            if (attempt === 0) await retryDelay(signal);
          }
        }
        status("delivery-failed");
        // Propagate exhaustion so rotating capture discards pending work and stops.
        throw failure ?? new Error("encounter-delivery-failed");
      }
    });
    if (!(await capture.start()) || terminated) { capture.discard(); terminate("binding-lost"); return { stop: () => {} }; }
    status("started");
    // Direct host withdrawal events stop immediately. This bounded refresh also
    // checks changes made elsewhere in the clinical record; failure stops analysis.
    const refresh = () => {
      if (refreshing || terminated) return; refreshing = true;
      void (async () => {
        try {
          const renewal = await options.client.calibrateClock(options.encounterId, authorizationAbort.signal);
          if (terminated) return;
          if (renewal && hasClock) {
            if (!capture?.updateClockCalibration(renewal)) { terminate("binding-lost"); return; }
            clockExpiresAtMs = renewal.localExpiresAtMs;
          }
        } catch (error) {
          if (error instanceof Error && /encounter-(clock-|request-40[13])/.test(error.message)) throw error;
          // A transient probe failure does not extend the prior calibration.
          // The embedded controller still stops when its existing bound expires.
        }
        return context();
      })().then(next => {
        if (terminated || !next) return;
        capture?.controller.updateAuthorization(next);
        if (capture?.controller.status === "stopped") terminate("binding-lost");
      })
        .catch(() => { if (!terminated) terminate("binding-lost"); }).finally(() => { refreshing = false; scheduleRefresh(); });
    };
    const scheduleRefresh = () => {
      if (terminated) return;
      // Join-time accuracy attestations may be close to expiry. Renew midway
      // through the remaining bound, capped at the authorization refresh limit.
      const delay = clockExpiresAtMs === null ? 20_000 : Math.min(20_000, Math.max(250, (clockExpiresAtMs - Date.now()) / 2));
      timer = setTimeout(refresh, delay);
    };
    scheduleRefresh();
  } catch {
    status("unavailable"); terminate("binding-lost");
  }
  return { stop: () => terminate("binding-lost") };
}

function retryDelay(signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const aborted = () => { clearTimeout(timer); signal.removeEventListener("abort", aborted); reject(signal.reason); };
    const timer = setTimeout(() => { signal.removeEventListener("abort", aborted); resolve(); }, 250);
    signal.addEventListener("abort", aborted, { once: true });
    if (signal.aborted) aborted();
  });
}
