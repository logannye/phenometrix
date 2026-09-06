import type { AmbientFacialFrame, AmbientVoiceFrame } from "@phenometrix/ambient-core";
import type { DurableObservationV1 } from "@phenometrix/contracts";

export type CaptureModality = "face" | "voice";

/** Declares the source boundary, not clinical validity or a promise that a metric is measurable. */
export interface CaptureAdapterCapabilities {
  readonly processingLocation: "patient-device" | "deployment-server";
  readonly sourceKind: DurableObservationV1["capture"]["sourceKind"];
  readonly modalities: readonly CaptureModality[];
  /** Raw packet formats emitted by this adapter, not formats its transport could hypothetically carry. */
  readonly rawFormats: readonly MediaPacket["format"][];
  readonly derivedBranch: boolean;
  readonly technicalMeasurementAvailability: "host-derived-primitives" | "requires-qualified-processor";
  readonly requiresQualifiedProcessor: boolean;
  /** Empty for all current adapters. A protocol independently authorizes any future clinical claim. */
  readonly clinicallyQualifiedMetricCodes: readonly string[];
}

/** Ownership transfers to the controller. Never serialize or persist a lease. */
export interface TransientMedia {
  readonly byteLength: number;
  readonly released: boolean;
  readonly bytes?: Uint8Array;
  readonly frame?: { close(): void };
  release(): void;
}

/** Wipes the exact owned view, including rejected and late packets. */
export function transientBytes(bytes: Uint8Array): TransientMedia {
  let released = false;
  return {
    byteLength: bytes.byteLength,
    get released() { return released; },
    get bytes() { return released ? undefined : bytes; },
    release() { if (!released) { bytes.fill(0); released = true; } }
  };
}

export function transientFrame(frame: { close(): void }, byteLength: number): TransientMedia {
  let released = false;
  return {
    byteLength,
    get released() { return released; },
    get frame() { return released ? undefined : frame; },
    release() { if (!released) { released = true; frame.close(); } }
  };
}

export interface MediaPacket {
  encounterId: string;
  platformParticipantId: string | null;
  attribution: "individual-track" | "mixed" | "unknown";
  modality: CaptureModality;
  trackId: string;
  sequence: number;
  acquiredAtMs: number;
  format: "pcm-s16le" | "jpeg" | "png" | "h264" | "video-frame";
  sampleRateHz?: number;
  channelCount?: number;
  media: TransientMedia;
}

export interface DerivedBatch {
  voiceFrames?: readonly AmbientVoiceFrame[];
  faceFrames?: readonly AmbientFacialFrame[];
}

/** Trusted worker output only; never a network deserialization boundary. */
export interface DerivedPacket extends Omit<MediaPacket, "format" | "sampleRateHz" | "channelCount" | "media"> {
  batch: DerivedBatch;
}

export interface ProcessingContext {
  signal: AbortSignal;
  captureEpoch: number;
  /** Distinct after camera off, packet loss, attribution loss, or processor reset. */
  continuityId: string;
  sessionStartedAtMs: number;
}

/** A worker bridge must terminate work and erase its own raw buffers on dispose. */
export interface MediaProcessor {
  supports(packet: MediaPacket): boolean;
  process(packet: MediaPacket, context: ProcessingContext): Promise<DerivedBatch>;
  dispose(): void;
}

export interface CaptureSourceContext {
  encounterId: string;
  platformParticipantId: string;
  modalities: readonly CaptureModality[];
  startedAtMs: number;
}

export interface CaptureSourceSink {
  packet(packet: MediaPacket): void;
  derived(packet: DerivedPacket): void;
  availability(modality: CaptureModality, available: boolean): void;
  interruption(reason: string): void;
}

export interface EncounterCaptureAdapter {
  readonly adapterId: string;
  readonly adapterVersion: string;
  readonly capabilities: CaptureAdapterCapabilities;
  /** Starts only after host authorization. Resolve once media negotiation is ready. */
  start(context: CaptureSourceContext, sink: CaptureSourceSink): Promise<void>;
  /** Synchronously detach sources, close transports, and dispose owned media. */
  stop(): void;
}
