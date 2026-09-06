import type { CaptureAdapterCapabilities, CaptureSourceContext, CaptureSourceSink, EncounterCaptureAdapter } from "./media.js";
import { transientBytes } from "./media.js";

export const ZOOM_RTMS_CAPTURE_CAPABILITIES = Object.freeze({
  processingLocation: "deployment-server", sourceKind: "platform-patient-track",
  modalities: Object.freeze(["face", "voice"] as const), rawFormats: Object.freeze(["pcm-s16le", "jpeg"] as const),
  derivedBranch: false, technicalMeasurementAvailability: "requires-qualified-processor",
  requiresQualifiedProcessor: true, clinicallyQualifiedMetricCodes: Object.freeze([])
} satisfies CaptureAdapterCapabilities);

export interface RtmsSocket {
  onopen: (() => void) | null;
  onmessage: ((event: { data: unknown }) => void) | null;
  onclose: (() => void) | null;
  onerror: (() => void) | null;
  send(data: string): void;
  close(): void;
}

export interface ZoomRtmsOptions {
  /** Values from an authenticated meeting.rtms_started webhook, not arbitrary user input. */
  meetingUuid: string;
  streamId: string;
  signalingUrl: string;
  /** Server-generated HMAC. Never place app credentials in a patient browser. */
  signature: string;
  socketFactory?: (url: string) => RtmsSocket;
  /** Deployment allowlist applies to signaling AND returned media URLs. */
  allowUrl?: (url: URL) => boolean;
  handshakeTimeoutMs?: number;
  maxMessageCharacters?: number;
}

/** Official HMAC input: client_id,meeting_uuid,rtms_stream_id. Run server-side. */
export async function createZoomRtmsSignature(clientId: string, meetingUuid: string, streamId: string,
  clientSecret: string): Promise<string> {
  const encoder = new TextEncoder();
  const keyBytes = encoder.encode(clientSecret);
  try {
    const key = await crypto.subtle.importKey("raw", keyBytes, { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
    const digest = await crypto.subtle.sign("HMAC", key, encoder.encode(`${clientId},${meetingUuid},${streamId}`));
    return [...new Uint8Array(digest)].map(b => b.toString(16).padStart(2, "0")).join("");
  } finally { keyBytes.fill(0); }
}

function object(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : null;
}

/**
 * RTMS WebSocket reference integration. No transcript, recording, mixed audio,
 * active-speaker switching, automatic reconnect, or implicit participant choice.
 */
export class ZoomRtmsAdapter implements EncounterCaptureAdapter {
  readonly adapterId = "zoom-rtms-reference";
  readonly adapterVersion = "1.0.0";
  readonly capabilities = ZOOM_RTMS_CAPTURE_CAPABILITIES;
  private sockets: RtmsSocket[] = [];
  private signalSocket: RtmsSocket | null = null;
  private context: CaptureSourceContext | null = null;
  private sink: CaptureSourceSink | null = null;
  private generation = 0;
  private sequence = 0;
  private videoSubscribed = false;
  private videoOn = false;
  private subscriptionTimestamp: number | null = null;
  private lastSubscriptionTimestamp = 0;
  private readonly heartbeatTimers = new Map<RtmsSocket, ReturnType<typeof setTimeout>>();
  private ready = new Set<"face" | "voice">();
  private timer: ReturnType<typeof setTimeout> | null = null;
  private resolve: (() => void) | null = null;
  private reject: ((reason: Error) => void) | null = null;

  constructor(private readonly options: ZoomRtmsOptions) {}

  async start(context: CaptureSourceContext, sink: CaptureSourceSink): Promise<void> {
    if (this.context) throw new Error("RTMS already started");
    if (!/^[1-9]\d*$/.test(context.platformParticipantId) || Number(context.platformParticipantId) > 0xffffffff)
      throw new Error("RTMS requires a verified numeric participant ID");
    if (!this.options.meetingUuid || !this.options.streamId || !/^[a-f0-9]{64}$/.test(this.options.signature))
      throw new Error("RTMS handshake configuration unavailable");
    this.context = { ...context, modalities: [...context.modalities] };
    this.sink = sink;
    const generation = ++this.generation;
    this.sequence = 0; this.ready.clear(); this.videoSubscribed = false; this.videoOn = false;
    this.lastSubscriptionTimestamp = 0;
    sink.availability("face", false);
    const completion = new Promise<void>((resolve, reject) => { this.resolve = resolve; this.reject = reject; });
    this.timer = setTimeout(() => this.fail(), this.options.handshakeTimeoutMs ?? 10_000);
    try {
      this.signalSocket = this.open(this.options.signalingUrl, generation, () => {
        this.send(this.signalSocket!, { msg_type: 1, protocol_version: 1, sequence: 1,
          meeting_uuid: this.options.meetingUuid, rtms_stream_id: this.options.streamId,
          signature: this.options.signature, buffer_data: false });
      }, message => this.signalMessage(message, generation));
    } catch { this.fail(); }
    return completion;
  }

  stop(): void {
    const reject = this.reject;
    this.reject = null; this.resolve = null;
    ++this.generation;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    for (const timer of this.heartbeatTimers.values()) clearTimeout(timer);
    this.heartbeatTimers.clear();
    for (const socket of this.sockets) {
      socket.onopen = socket.onmessage = socket.onclose = socket.onerror = null;
      try { socket.close(); } catch { /* already disconnected */ }
    }
    this.sockets = []; this.signalSocket = null; this.context = null; this.sink = null;
    this.ready.clear(); this.videoSubscribed = false; this.subscriptionTimestamp = null;
    reject?.(new Error("RTMS capture stopped"));
  }

  private open(url: string, generation: number, opened: () => void,
    message: (value: Record<string, unknown>) => void): RtmsSocket {
    const parsed = new URL(url);
    const allowed = this.options.allowUrl ?? (u => u.hostname.endsWith(".zoom.us") || u.hostname.endsWith(".zoom.com"));
    if (parsed.protocol !== "wss:" || parsed.username || parsed.password || !allowed(parsed)) throw new Error("Untrusted RTMS URL");
    const socket = this.options.socketFactory?.(url) ?? new WebSocket(url) as unknown as RtmsSocket;
    this.sockets.push(socket);
    socket.onopen = () => { if (generation === this.generation) { try { this.watchHeartbeat(socket); opened(); } catch { this.fail(); } } };
    socket.onmessage = event => {
      if (generation !== this.generation) return;
      this.watchHeartbeat(socket);
      // Binary framing and payload encryption need separate qualified decoders.
      if (typeof event.data !== "string" || event.data.length > (this.options.maxMessageCharacters ?? 6_000_000)) { this.fail(); return; }
      try {
        const parsedMessage = object(JSON.parse(event.data));
        if (!parsedMessage) throw new Error("Invalid RTMS message");
        if (parsedMessage.msg_type === 12) this.send(socket, { msg_type: 13, timestamp: parsedMessage.timestamp });
        else message(parsedMessage);
      } catch { this.fail(); }
    };
    socket.onclose = socket.onerror = () => { if (generation === this.generation) this.fail(); };
    return socket;
  }

  private signalMessage(message: Record<string, unknown>, generation: number): void {
    if (message.msg_type === 2) {
      if (message.status_code !== 0 || this.sockets.length > 1) { this.fail(); return; }
      const urls = object(object(message.media_server)?.server_urls);
      for (const modality of this.context!.modalities) {
        const url = urls?.[modality === "voice" ? "audio" : "video"];
        if (typeof url !== "string") { this.fail(); return; }
        this.connectMedia(modality, url, generation);
      }
      this.send(this.signalSocket!, { msg_type: 5, events: [
        { event_type: 8, subscribe: true }, { event_type: 9, subscribe: true }, { event_type: 4, subscribe: true }
      ] });
    } else if (message.msg_type === 6) {
      const event = object(message.event);
      if (event?.event_type === 7) { this.fail(); return; }
      if (event?.event_type === 4 && Array.isArray(event.participants)
        && event.participants.some(p => String(object(p)?.user_id) === this.context!.platformParticipantId)) { this.fail(); return; }
      // Deliberately ignore active-speaker events (type 2).
      if (event?.event_type !== 8 && event?.event_type !== 9) return;
      const participants = event.participants;
      if (!Array.isArray(participants) || !participants.some(p => String(object(p)?.user_id) === this.context!.platformParticipantId)) return;
      this.videoOn = event.event_type === 8;
      this.videoSubscribed = false;
      this.subscriptionTimestamp = null;
      this.sink!.availability("face", false);
      if (this.videoOn && this.ready.has("face")) this.subscribeVideo();
    } else if (message.msg_type === 29) {
      if (String(message.user_id) !== this.context!.platformParticipantId || message.timestamp !== this.subscriptionTimestamp) return;
      this.videoSubscribed = message.status_code === 0 && this.videoOn;
      this.sink!.availability("face", this.videoSubscribed);
    } else if ((message.msg_type === 8 && message.state !== 1)
      || (message.msg_type === 9 && message.state !== 1 && message.state !== 2)) {
      // Resume requires fresh consent/binding and a new capture epoch; no buffered replay.
      this.fail();
    }
  }

  private connectMedia(modality: "face" | "voice", url: string, generation: number): void {
    const voice = modality === "voice";
    let socket: RtmsSocket;
    socket = this.open(url, generation, () => {
      this.send(socket, { msg_type: 3, protocol_version: 1, sequence: 0,
        meeting_uuid: this.options.meetingUuid, rtms_stream_id: this.options.streamId,
        signature: this.options.signature, payload_encryption: false, media_type: voice ? 1 : 2,
        media_params: voice
          ? { audio: { content_type: 2, sample_rate: 3, channel: 1, codec: 1, data_opt: 2, send_rate: 20 } }
          : { video: { content_type: 3, codec: 5, resolution: 2, fps: 5, data_opt: 4 } } });
    }, message => {
      if (message.msg_type === 4) {
        const parameters = object(object(message.media_params)?.[voice ? "audio" : "video"]);
        if (message.status_code !== 0 || message.payload_encrypted === true || message.payload_encryption === true
          || !parameters || parameters.codec !== (voice ? 1 : 5) || parameters.data_opt !== (voice ? 2 : 4)
          || (voice && (parameters.sample_rate !== 3 || parameters.channel !== 1 || parameters.content_type !== 2))) { this.fail(); return; }
        this.ready.add(modality);
        if (!voice && this.videoOn) this.subscribeVideo();
        if (this.context!.modalities.every(m => this.ready.has(m))) {
          this.send(this.signalSocket!, { msg_type: 7, rtms_stream_id: this.options.streamId });
          if (this.timer) clearTimeout(this.timer);
          this.timer = null;
          this.resolve?.(); this.resolve = null; this.reject = null;
        }
      } else if (message.msg_type === (voice ? 14 : 15) && this.ready.has(modality)) {
        this.mediaMessage(modality, message);
      }
    });
  }

  private subscribeVideo(): void {
    this.subscriptionTimestamp = this.lastSubscriptionTimestamp = Math.max(Date.now(), this.lastSubscriptionTimestamp + 1);
    this.send(this.signalSocket!, { msg_type: 28, user_id: Number(this.context!.platformParticipantId),
      subscribe: true, timestamp: this.subscriptionTimestamp });
  }

  private mediaMessage(modality: "face" | "voice", message: Record<string, unknown>): void {
    const content = object(message.content);
    // Drop nonpatient media before base64 decoding, and never infer identity from display names.
    if (!content || String(content.user_id) !== this.context!.platformParticipantId
      || (modality === "face" && !this.videoSubscribed)) return;
    if (typeof content.data !== "string" || !Number.isSafeInteger(content.length) || Number(content.length) <= 0
      || Number(content.length) > 4_194_304 || !Number.isSafeInteger(content.timestamp)
      || !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(content.data)) throw new Error("Invalid RTMS media");
    const bytes = Uint8Array.from(atob(content.data), c => c.charCodeAt(0));
    if (bytes.byteLength !== content.length || (modality === "voice" && bytes.byteLength % 2 !== 0)) { bytes.fill(0); throw new Error("Invalid media length"); }
    const media = transientBytes(bytes);
    try { this.sink!.packet({ encounterId: this.context!.encounterId, platformParticipantId: this.context!.platformParticipantId,
      attribution: "individual-track", modality, trackId: `rtms-${this.options.streamId}-${modality}`,
      sequence: ++this.sequence, acquiredAtMs: Number(content.timestamp), format: modality === "voice" ? "pcm-s16le" : "jpeg",
      ...(modality === "voice" ? { sampleRateHz: 48_000, channelCount: 1 } : {}), media });
    } catch (error) { media.release(); throw error; }
  }

  private watchHeartbeat(socket: RtmsSocket) {
    const previous = this.heartbeatTimers.get(socket); if (previous) clearTimeout(previous);
    this.heartbeatTimers.set(socket, setTimeout(() => this.fail(), 65_000));
  }

  private send(socket: RtmsSocket, value: Record<string, unknown>) { socket.send(JSON.stringify(value)); }
  private fail(): void {
    const sink = this.sink;
    this.stop();
    sink?.interruption("rtms-unavailable");
  }
}
