import { afterEach, describe, expect, it, vi } from "vitest";
import { createHmac } from "node:crypto";
import { createZoomRtmsSignature, ZoomRtmsAdapter, type RtmsSocket } from "./zoom-rtms.js";
import type { MediaPacket } from "./media.js";

class SyntheticSocket implements RtmsSocket {
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: unknown }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  readonly sent: Record<string, unknown>[] = [];
  closed = false;
  send(data: string) { this.sent.push(JSON.parse(data)); }
  close() { this.closed = true; }
  opened() { this.onopen?.(); }
  received(message: unknown) { this.onmessage?.({ data: JSON.stringify(message) }); }
}
afterEach(() => vi.useRealTimers());
function setup() {
  const sockets: SyntheticSocket[] = []; const packets: MediaPacket[] = [];
  const availability = vi.fn(); const interruption = vi.fn();
  const adapter = new ZoomRtmsAdapter({ meetingUuid: "synthetic-meeting", streamId: "synthetic-stream",
    signature: "a".repeat(64), signalingUrl: "wss://test.zoom.us/signaling",
    socketFactory: () => { const socket = new SyntheticSocket(); sockets.push(socket); return socket; } });
  const start = adapter.start({ encounterId: "test-encounter", platformParticipantId: "123", modalities: ["voice", "face"], startedAtMs: 1 },
    { packet: p => { packets.push(p); p.media.release(); }, derived() {}, availability, interruption });
  const signal = sockets[0]; signal.opened();
  signal.received({ msg_type: 2, status_code: 0, media_server: { server_urls: { audio: "wss://test.zoom.us/audio", video: "wss://test.zoom.us/video" } } });
  const audio = sockets[1]; const video = sockets[2]; audio.opened(); video.opened();
  return { adapter, sockets, signal, audio, video, packets, availability, interruption, start };
}
function handshakes(test: ReturnType<typeof setup>, audioOverride: Record<string, unknown> = {}) {
  test.audio.received({ msg_type: 4, status_code: 0, payload_encrypted: false,
    media_params: { audio: { content_type: 2, sample_rate: 3, channel: 1, codec: 1, data_opt: 2, ...audioOverride } } });
  test.video.received({ msg_type: 4, status_code: 0, payload_encrypted: false,
    media_params: { video: { content_type: 3, codec: 5, data_opt: 4, fps: 5 } } });
}
const media = (userId: number, type = 14) => ({ msg_type: type,
  content: { user_id: userId, user_name: "Synthetic test actor", data: "AQIDBA==", length: 4, timestamp: 1_700_000_000_000 } });

describe("Zoom RTMS protocol reference (mock WebSockets, no live Zoom claim)", () => {
  it("generates the official HMAC without logging or retaining credential bytes", async () => {
    expect(await createZoomRtmsSignature("test-client", "test-meeting", "test-stream", "test-secret"))
      .toBe(createHmac("sha256", "test-secret").update("test-client,test-meeting,test-stream").digest("hex"));
  });
  it("negotiates individual audio and subscribed video, requests no transcript, and answers keepalives", async () => {
    const test = setup(); handshakes(test); await test.start;
    expect(test.signal.sent[0]).toMatchObject({ msg_type: 1, sequence: 1, buffer_data: false });
    expect(test.audio.sent[0]).toMatchObject({ media_type: 1, media_params: { audio: { data_opt: 2, sample_rate: 3 } } });
    expect(test.video.sent[0]).toMatchObject({ media_type: 2, media_params: { video: { data_opt: 4 } } });
    expect(test.signal.sent.some(m => m.msg_type === 7)).toBe(true);
    test.signal.received({ msg_type: 12, timestamp: 123 }); test.audio.received({ msg_type: 12, timestamp: 456 });
    expect(test.signal.sent.at(-1)).toEqual({ msg_type: 13, timestamp: 123 });
    expect(test.audio.sent.at(-1)).toEqual({ msg_type: 13, timestamp: 456 }); test.adapter.stop();
  });
  it("rejects fallback mixed audio instead of applying active-speaker metadata", async () => {
    const test = setup(); const outcome = expect(test.start).rejects.toThrow(); handshakes(test, { data_opt: 1 });
    await outcome; expect(test.interruption).toHaveBeenCalledOnce(); expect(test.sockets.every(s => s.closed)).toBe(true);
  });
  it("filters other participants before decode and never uses display names or active-speaker changes", async () => {
    const test = setup(); handshakes(test); await test.start;
    test.signal.received({ msg_type: 6, event: { event_type: 2, user_id: 456 } });
    test.audio.received(media(456)); test.audio.received(media(0)); test.audio.received(media(123));
    expect(test.packets).toHaveLength(1); expect(test.packets[0].platformParticipantId).toBe("123");
    expect(test.signal.sent.some(m => m.msg_type === 28)).toBe(false); test.adapter.stop();
  });
  it("subscribes only to the selected patient's video, respects camera off, and rejects stale acknowledgements", async () => {
    const test = setup(); handshakes(test); await test.start;
    test.signal.received({ msg_type: 6, event: { event_type: 8, participants: [{ user_id: 456 }] } });
    test.video.received(media(123, 15)); expect(test.packets).toHaveLength(0);
    test.signal.received({ msg_type: 6, event: { event_type: 8, participants: [{ user_id: 123 }] } });
    const subscription = test.signal.sent.at(-1)!; expect(subscription).toMatchObject({ msg_type: 28, user_id: 123 });
    test.signal.received({ msg_type: 29, user_id: 123, status_code: 0, timestamp: subscription.timestamp });
    test.video.received(media(123, 15)); expect(test.packets).toHaveLength(1);
    test.signal.received({ msg_type: 6, event: { event_type: 9, participants: [{ user_id: 123 }] } });
    test.signal.received({ msg_type: 29, user_id: 123, status_code: 0, timestamp: subscription.timestamp });
    test.video.received(media(123, 15)); expect(test.packets).toHaveLength(1); test.adapter.stop();
  });
  it("allows initial active states, but terminates on pause or network interruption without replay", async () => {
    const test = setup(); handshakes(test); await test.start;
    test.signal.received({ msg_type: 8, state: 1 }); test.signal.received({ msg_type: 9, state: 2 });
    expect(test.sockets.every(s => !s.closed)).toBe(true);
    test.signal.received({ msg_type: 8, state: 5 }); expect(test.sockets.every(s => s.closed)).toBe(true);
    expect(test.interruption).toHaveBeenCalledOnce();
  });
  it("rejects unapproved returned websocket endpoints without connecting", async () => {
    const sockets: SyntheticSocket[] = [];
    const adapter = new ZoomRtmsAdapter({ meetingUuid: "test", streamId: "test", signature: "a".repeat(64),
      signalingUrl: "wss://test.zoom.us", socketFactory: () => { const s = new SyntheticSocket(); sockets.push(s); return s; } });
    const start = adapter.start({ encounterId: "test", platformParticipantId: "123", modalities: ["voice"], startedAtMs: 1 },
      { packet: p => p.media.release(), derived() {}, availability() {}, interruption() {} });
    const outcome = expect(start).rejects.toThrow(); sockets[0].opened();
    sockets[0].received({ msg_type: 2, status_code: 0, media_server: { server_urls: { audio: "wss://attacker.invalid" } } });
    await outcome; expect(sockets).toHaveLength(1);
  });
  it("fails closed after 65 seconds without messages on any negotiated socket", async () => {
    vi.useFakeTimers(); const test = setup(); handshakes(test); await test.start;
    await vi.advanceTimersByTimeAsync(65_001);
    expect(test.sockets.every(s => s.closed)).toBe(true); expect(test.interruption).toHaveBeenCalledOnce();
  });
});
