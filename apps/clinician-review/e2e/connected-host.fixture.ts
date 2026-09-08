import { createEncounterClient } from "../../../packages/encounter-client/src/index.js";
import type { TreatmentResponseScopeV1 } from "../../../packages/contracts/src/treatment-response.js";
import { syntheticFacialFrame } from "../../../packages/ambient-core/src/test-helpers.js";
import { startIntegratedEncounter, type HostEncounterEvent } from "../../capture-web/src/integrated-encounter.js";
import type { VisualWorkerRequest } from "../../capture-web/src/face-worker-protocol.js";
import { mountTreatmentResponsePanel } from "../src/panel.js";

/** Browser-only, explicitly fabricated media. The controller, finalization worker,
 * signed HTTP client/service, trajectory worker and panel are production code. */
export async function startSyntheticConnectedHost(input: {
  baseUrl: string; token: string; episodeId: string; encounterId: string;
  platformParticipantId: string; scope: TreatmentResponseScopeV1; assetBaseUrl: string;
}) {
  const stats = { deviceRequests: 0, trackStops: 0, bitmapsCreated: 0, bitmapsClosed: 0,
    finalizationWorkersStarted: 0, finalizationWorkersActive: 0, maximumFinalizationWorkers: 0,
    acceptedFrames: 0, faceWorkerStops: 0, clockResponses: 0,
    inFlightRequests: 0, maximumInFlightRequests: 0, clockProbesInFlight: 0,
    clockProbeCyclesCompleted: 0, clockProbeCyclesFailed: 0, authorizationRequestsInFlight: 0,
    requests: [] as string[], statuses: [] as string[], unavailability: [] as string[], events: [] as string[] };
  let observer: (event: HostEncounterEvent) => void = () => {};
  let nextFrame: ((now: number, metadata: { presentationTime: number; width: number; height: number }) => void) | null = null;
  class SyntheticTrack extends EventTarget {
    kind = "video"; readyState = "live"; enabled = true; muted = false;
    getSettings() { return { deviceId: "synthetic-camera", width: 1280, height: 720, frameRate: 25 }; }
    stop() { stats.trackStops++; }
  }
  const track = new SyntheticTrack();
  class SyntheticStream {
    constructor(private readonly tracks: SyntheticTrack[]) {}
    getVideoTracks() { return this.tracks; }
    getAudioTracks() { return []; }
  }
  const stream = new SyntheticStream([track]);
  const video = { srcObject: stream, videoWidth: 1280, videoHeight: 720,
    requestVideoFrameCallback(callback: NonNullable<typeof nextFrame>) { nextFrame = callback; return 1; },
    cancelVideoFrameCallback() { nextFrame = null; } };
  const nativeWorker = window.Worker;
  class SyntheticFaceWorker {
    onmessage: ((event: { data: unknown }) => void) | null = null;
    onerror = null; private stopped = false;
    postMessage(message: VisualWorkerRequest) {
      if (message.type === "initialize") queueMicrotask(() => this.onmessage?.({ data: { type: "ready", captureEpoch: message.captureEpoch } }));
      if (message.type !== "frame") return;
      message.bitmap.close();
      const aperture = message.tMs % 2000 < 120 ? .03 : .3;
      queueMicrotask(() => {
        if (this.stopped) return;
        this.onmessage?.({ data: { type: "frame", captureEpoch: message.captureEpoch, sequence: message.sequence,
          acquiredAtMs: message.acquiredAtMs, faceCount: 1,
          frame: syntheticFacialFrame(message.tMs, "ambient-frontal", { acquiredAtMs: message.acquiredAtMs,
            captureEpoch: message.captureEpoch, sequence: message.sequence, interResultGapMs: 40,
            analyzedFrameRate: 25, eyeAperture: { left: aperture, right: aperture } }) } });
      });
    }
    terminate() { if (!this.stopped) stats.faceWorkerStops++; this.stopped = true; }
  }
  window.Worker = class {
    constructor(url: string | URL, options?: WorkerOptions) {
      if (String(url).includes("face-worker")) return new SyntheticFaceWorker() as unknown as Worker;
      const worker = new nativeWorker(url, options);
      if (String(url).includes("ambient-finalization-worker")) {
        stats.finalizationWorkersStarted++; stats.finalizationWorkersActive++;
        stats.maximumFinalizationWorkers = Math.max(stats.maximumFinalizationWorkers, stats.finalizationWorkersActive);
        const terminate = worker.terminate.bind(worker); let stopped = false;
        worker.terminate = () => { if (!stopped) stats.finalizationWorkersActive--; stopped = true; terminate(); };
      }
      return worker;
    }
  } as typeof Worker;
  window.MediaStream = SyntheticStream as unknown as typeof MediaStream;
  window.createImageBitmap = (async () => {
    stats.bitmapsCreated++; let closed = false;
    return { close() { if (!closed) stats.bitmapsClosed++; closed = true; } } as ImageBitmap;
  }) as typeof createImageBitmap;
  navigator.mediaDevices.getUserMedia = async () => { stats.deviceRequests++; throw new Error("Synthetic harness forbids device access"); };
  const transport: typeof fetch = async (resource, options) => {
    const pathname = new URL(String(resource), location.href).pathname;
    stats.requests.push(`${options?.method ?? "GET"} ${pathname}`);
    stats.inFlightRequests++;
    stats.maximumInFlightRequests = Math.max(stats.maximumInFlightRequests, stats.inFlightRequests);
    try {
      const response = await fetch(resource, options);
      // Fetch resolves at headers. Keep the real request active through its body
      // so a simulated clock step cannot turn body transit into fictional RTT.
      const body = await response.arrayBuffer();
      if (pathname.endsWith("/clock")) stats.clockResponses++;
      return new Response(body, { status: response.status, statusText: response.statusText, headers: response.headers });
    } finally { stats.inFlightRequests--; }
  };
  const underlyingClient = createEncounterClient({ ...input, accessToken: async () => input.token, fetch: transport });
  const client = { ...underlyingClient,
    async calibrateClock(...args: Parameters<typeof underlyingClient.calibrateClock>) {
      stats.clockProbesInFlight++;
      try {
        const result = await underlyingClient.calibrateClock(...args);
        if (result) stats.clockProbeCyclesCompleted++;
        return result;
      } catch (error) { stats.clockProbeCyclesFailed++; throw error; }
      finally { stats.clockProbesInFlight--; }
    },
    async captureContext(...args: Parameters<typeof underlyingClient.captureContext>) {
      stats.authorizationRequestsInFlight++;
      try { return await underlyingClient.captureContext(...args); }
      finally { stats.authorizationRequestsInFlight--; }
    }
  };
  document.body.replaceChildren();
  const description = document.createElement("p"); description.textContent = "Synthetic connected encounter acceptance — fabricated media, no hardware or clinical performance claim.";
  const chart = document.createElement("div"); chart.id = "connected-chart"; document.body.append(description, chart);
  const panel = mountTreatmentResponsePanel(chart, { scope: input.scope,
    load: signal => client.loadEvidence(signal), onUnavailable: reason => stats.unavailability.push(reason) });
  const capture = await startIntegratedEncounter({ client, encounterId: input.encounterId,
    platformParticipantId: input.platformParticipantId, stream: stream as unknown as MediaStream,
    video: video as unknown as HTMLVideoElement, assetBaseUrl: new URL(input.assetBaseUrl, location.href).href, modalities: ["face"],
    subscribeLifecycle: listener => { observer = listener; return () => { observer = () => {}; }; },
    onEvent: event => { if (event.type === "accepted") stats.acceptedFrames += event.count ?? 0;
      else { const label = `${event.type}:${event.reason ?? ""}`; if (!stats.events.includes(label)) stats.events.push(label); } },
    onStatus: status => stats.statuses.push(status)
  });
  const present = () => {
    if (track.enabled && !track.muted) nextFrame?.(performance.now(), { presentationTime: performance.now(), width: 1280, height: 720 });
  };
  let presented: ReturnType<typeof setInterval> | null = setInterval(present, 40);
  await panel.refresh();
  return {
    stats, refresh: () => panel.refresh(),
    async transportState() {
      // Two native message turns let chained probe/context promises start their
      // next request. This barrier never advances Date or performance clocks.
      for (let turn = 0; turn < 2; turn++) await new Promise<void>(resolve => {
        const channel = new MessageChannel();
        channel.port1.onmessage = () => { channel.port1.close(); channel.port2.close(); resolve(); };
        channel.port2.postMessage(null);
      });
      return { requests: stats.inFlightRequests, probes: stats.clockProbesInFlight, authorization: stats.authorizationRequestsInFlight };
    },
    camera(available: boolean) {
      if (presented) clearInterval(presented);
      presented = available ? setInterval(present, 40) : null;
      track.muted = !available; track.dispatchEvent(new Event(available ? "unmute" : "mute"));
    },
    chartVisible(visible: boolean) { chart.hidden = !visible; Object.defineProperty(document, "visibilityState", { configurable: true, value: visible ? "visible" : "hidden" }); document.dispatchEvent(new Event("visibilitychange")); },
    event(event: HostEncounterEvent) { observer(event); },
    async wrongScope() {
      const wrong = createEncounterClient({ ...input, scope: { ...input.scope, participantId: "synthetic-other-patient" }, accessToken: async () => input.token });
      try { await wrong.loadEvidence(); return "accepted"; } catch (error) { return error instanceof Error ? error.message : "rejected"; }
    },
    dispose() { if (presented) clearInterval(presented); capture.stop(); panel.dispose(); }
  };
}
