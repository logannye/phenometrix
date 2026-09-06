import type { EncounterClient } from "@phenometrix/encounter-client";
import { createEmbeddedEncounter, type EmbeddedEncounterOptions } from "./embedded-encounter.js";
import type { CaptureAuthorization } from "@phenometrix/encounter-capture";
import { AMBIENT_LOCAL_PROTOCOL_REF } from "@phenometrix/contracts";

export type HostEncounterEvent = "ended" | "consent-withdrawn" | "binding-lost" | "resource-pressure";
export interface IntegratedEncounterOptions extends Omit<EmbeddedEncounterOptions, "authorization" | "observationId" | "revisionId" | "onObservation"> {
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
  let timer: ReturnType<typeof setInterval> | null = null;
  let terminated = false, refreshing = false;
  let unsubscribe = () => {};
  const status = (value: Parameters<NonNullable<IntegratedEncounterOptions["onStatus"]>>[0]) => {
    try { options.onStatus?.(value); } catch { /* Host telemetry must not interrupt the visit. */ }
  };
  const terminate = (event: HostEncounterEvent) => {
    if (terminated) return;
    terminated = true; authorizationAbort.abort(); if (timer) clearInterval(timer);
    try { unsubscribe(); } catch { /* Host owns its event bus. */ }
    if (event === "ended") {
      // Finishing/delivery is asynchronous and does not delay the telehealth end.
      void capture?.finish().catch(() => status("delivery-failed"));
    } else {
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
    const authorization = await context();
    if (terminated) return { stop: () => {} };
    const captureId = `capture-${crypto.randomUUID()}`;
    capture = createEmbeddedEncounter({ ...options, authorization, observationId: captureId, revisionId: `${captureId}-v1`,
      onObservation: async (observation, captureSignal) => {
        const signal = captureSignal ? AbortSignal.any([captureSignal, deliveryAbort.signal]) : deliveryAbort.signal;
        for (let attempt = 0; attempt < 2; attempt++) {
          if (signal.aborted) return;
          try { await options.client.ingestObservation(observation, signal); status("delivered"); return; }
          catch (error) {
            if (error instanceof Error && error.message === "encounter-request-403") { terminate("consent-withdrawn"); break; }
          }
        }
        status("delivery-failed");
      }
    });
    if (!(await capture.start()) || terminated) { capture.discard(); terminate("binding-lost"); return { stop: () => {} }; }
    status("started");
    // Direct host withdrawal events stop immediately. This bounded refresh also
    // checks changes made elsewhere in the clinical record; failure stops analysis.
    timer = setInterval(() => {
      if (refreshing || terminated) return; refreshing = true;
      void context().then(next => {
        if (terminated) return;
        capture?.controller.updateAuthorization(next);
        if (capture?.controller.status === "stopped") terminate("binding-lost");
      })
        .catch(() => { if (!terminated) terminate("binding-lost"); }).finally(() => { refreshing = false; });
    }, 30_000);
  } catch {
    status("unavailable"); terminate("binding-lost");
  }
  return { stop: () => terminate("binding-lost") };
}
