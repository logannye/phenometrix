import type { CaptureAdapterCapabilities, CaptureModality, CaptureSourceContext, CaptureSourceSink, EncounterCaptureAdapter } from "./media.js";

export const EMBEDDED_LOCAL_CAPTURE_CAPABILITIES = Object.freeze({
  processingLocation: "patient-device", sourceKind: "patient-local-pre-codec",
  modalities: Object.freeze(["face", "voice"] as const), rawFormats: Object.freeze([]),
  derivedBranch: true, technicalMeasurementAvailability: "host-derived-primitives",
  requiresQualifiedProcessor: false, clinicallyQualifiedMetricCodes: Object.freeze([])
} satisfies CaptureAdapterCapabilities);

export interface LocalCaptureBranch { stop(): void }
export type LocalCaptureFactory = (context: CaptureSourceContext, sink: CaptureSourceSink,
  signal: AbortSignal) => Promise<LocalCaptureBranch>;

/** Attaches an analysis branch to a host-owned stream, never acquires devices. */
export class EmbeddedLocalAdapter implements EncounterCaptureAdapter {
  readonly adapterId = "embedded-local";
  readonly adapterVersion = "1.0.0";
  readonly capabilities: CaptureAdapterCapabilities;
  private branch: LocalCaptureBranch | null = null;
  private abort: AbortController | null = null;
  private generation = 0;
  constructor(private readonly factory: LocalCaptureFactory, modalities: readonly CaptureModality[] = EMBEDDED_LOCAL_CAPTURE_CAPABILITIES.modalities) {
    if (!modalities.length || modalities.some(modality => !EMBEDDED_LOCAL_CAPTURE_CAPABILITIES.modalities.includes(modality)))
      throw new Error("Unsupported embedded modality");
    this.capabilities = Object.freeze({ ...EMBEDDED_LOCAL_CAPTURE_CAPABILITIES,
      modalities: Object.freeze([...new Set(modalities)]) });
  }
  async start(context: CaptureSourceContext, sink: CaptureSourceSink): Promise<void> {
    if (this.abort) throw new Error("Local adapter already started");
    const abort = this.abort = new AbortController();
    const generation = ++this.generation;
    const branch = await this.factory(context, {
      packet: packet => { if (abort.signal.aborted) packet.media.release(); else sink.packet(packet); },
      derived: packet => { if (!abort.signal.aborted) sink.derived(packet); },
      availability: (m, available) => { if (!abort.signal.aborted) sink.availability(m, available); },
      interruption: reason => { if (!abort.signal.aborted) sink.interruption(reason); }
    }, abort.signal);
    if (generation !== this.generation || abort.signal.aborted) { branch.stop(); return; }
    this.branch = branch;
  }
  stop(): void {
    ++this.generation;
    this.abort?.abort(); this.abort = null;
    this.branch?.stop(); this.branch = null;
  }
}
