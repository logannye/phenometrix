import { afterEach, describe, it, expect, vi } from "vitest";
import { createHmac } from "node:crypto";
import { createZoomWebhookHandler, type ZoomWebhookHost } from "./zoom-webhook.js";
import { authorization, EPOCH } from "./test-fixtures.js";
afterEach(() => vi.useRealTimers());

async function signed(event: string, payload: object, timestamp = Math.floor(EPOCH / 1000)) {
  const body = JSON.stringify({ event, event_ts: EPOCH, payload });
  const key = await crypto.subtle.importKey("raw", new TextEncoder().encode("test-webhook-secret"), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const digest = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(`v0:${timestamp}:${body}`));
  const signature = [...new Uint8Array(digest)].map(value => value.toString(16).padStart(2, "0")).join("");
  return new Request("https://service.example/zoom", { method: "POST", headers: { "Content-Type": "application/json", "x-zm-request-timestamp": String(timestamp), "x-zm-signature": `v0=${signature}` }, body });
}
function fixture() {
  const route = { accountId: "enrolled-account", meetingUuid: "scheduled-instance", authorization: authorization() };
  const host: ZoomWebhookHost = { webhookSecret: "test-webhook-secret", now: () => EPOCH,
    resolveEncounter: vi.fn(async () => route), started: vi.fn(async () => {}), stopped: vi.fn(async () => {}) };
  const payload = { account_id: route.accountId, meeting_uuid: route.meetingUuid, rtms_stream_id: "stream", server_urls: "wss://rtms.zoom.us/signal" };
  return { host, route, payload, handler: createZoomWebhookHandler(host) };
}
describe("signed Zoom encounter ingress", () => {
  it("rejects forged and expired requests before resolving enrollment", async () => {
    const f = fixture(), request = await signed("meeting.rtms_started", f.payload);
    request.headers.set("x-zm-signature", `v0=${"0".repeat(64)}`);
    expect((await f.handler(request)).status).toBe(401);
    expect((await f.handler(await signed("meeting.rtms_started", f.payload, Math.floor(EPOCH / 1000) - 301))).status).toBe(401);
    expect(f.host.resolveEncounter).not.toHaveBeenCalled();
  });
  it("starts only the exact eligible encounter and provides a stable retry identity", async () => {
    const f = fixture();
    expect((await f.handler(await signed("meeting.rtms_started", f.payload))).status).toBe(200);
    expect((await f.handler(await signed("meeting.rtms_started", f.payload))).status).toBe(200);
    const calls = vi.mocked(f.host.started).mock.calls;
    expect(calls[0].slice(0, 4)).toEqual(calls[1].slice(0, 4));
    expect(calls[0][4].aborted).toBe(false);
    expect(calls[0][0].authorization.binding.platformParticipantId).toBe("123");
    f.route.authorization.consent.withdrawnAt = new Date(EPOCH).toISOString();
    await f.handler(await signed("meeting.rtms_started", f.payload));
    expect(f.host.started).toHaveBeenCalledTimes(2);
    expect((await f.handler(await signed("meeting.rtms_stopped", f.payload))).status).toBe(200);
    expect(f.host.stopped).toHaveBeenCalledWith("scheduled-instance", "stream", expect.any(AbortSignal));
    await f.handler(await signed("meeting.rtms_stopped", f.payload));
    expect(f.host.stopped).toHaveBeenCalledTimes(2);
  });
  it("ignores unrelated accounts and rejects unexpected signaling hosts", async () => {
    const f = fixture();
    await f.handler(await signed("meeting.rtms_started", { ...f.payload, account_id: "unrelated" }));
    expect((await f.handler(await signed("meeting.rtms_started", { ...f.payload, server_urls: "wss://attacker.example" }))).status).toBe(422);
    expect(f.host.started).not.toHaveBeenCalled();
  });
  it("answers an authenticated URL validation challenge without connecting media", async () => {
    const f = fixture();
    const response = await f.handler(await signed("endpoint.url_validation", { plainToken: "challenge" }));
    expect(await response.json()).toEqual({ plainToken: "challenge",
      encryptedToken: createHmac("sha256", "test-webhook-secret").update("challenge").digest("hex") });
    expect(f.host.resolveEncounter).not.toHaveBeenCalled();
  });
  it.each(["resolveEncounter", "started", "stopped"] as const)("returns retryable 503 if the external %s port fails", async port => {
    const f = fixture(); vi.mocked(f.host[port]).mockRejectedValueOnce(new Error("host unavailable"));
    const response = await f.handler(await signed(port === "stopped" ? "meeting.rtms_stopped" : "meeting.rtms_started", f.payload));
    expect(response.status).toBe(503); expect(response.headers.get("Retry-After")).toBe("1");
    expect(await response.json()).toEqual({});
  });
  it("does not enqueue when consent expires during enrollment lookup", async () => {
    const f = fixture(); let now = EPOCH; f.host.now = () => now;
    f.route.authorization.consent.expiresAt = new Date(EPOCH + 50).toISOString();
    vi.mocked(f.host.resolveEncounter).mockImplementation(async () => { now += 100; return f.route; });
    expect((await f.handler(await signed("meeting.rtms_started", f.payload))).status).toBe(200);
    expect(f.host.started).not.toHaveBeenCalled();
  });
  it("preserves semantic retry identity across re-signed deliveries and scopes it by account", async () => {
    const f = fixture();
    await f.handler(await signed("meeting.rtms_started", f.payload));
    await f.handler(await signed("meeting.rtms_started", f.payload, Math.floor(EPOCH / 1000) + 1));
    const first = vi.mocked(f.host.started).mock.calls[0][3].idempotencyKey;
    expect(vi.mocked(f.host.started).mock.calls[1][3].idempotencyKey).toBe(first);
    f.route.accountId = "other-account";
    await f.handler(await signed("meeting.rtms_started", { ...f.payload, account_id: "other-account" }));
    expect(vi.mocked(f.host.started).mock.calls[2][3].idempotencyKey).not.toBe(first);
  });
  it("bounds host callback latency and signals cancellation before the response deadline", async () => {
    const f = fixture(); const request = await signed("meeting.rtms_started", f.payload);
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    vi.mocked(f.host.started).mockImplementation(() => new Promise(() => {}));
    const pending = f.handler(request);
    await vi.waitFor(() => expect(f.host.started).toHaveBeenCalledOnce());
    const signal = vi.mocked(f.host.started).mock.calls[0][4];
    await vi.advanceTimersByTimeAsync(2_501);
    expect((await pending).status).toBe(503); expect(signal.aborted).toBe(true);
  });
  it("times out a stalled request body, cancels it without waiting, and wipes received chunks", async () => {
    const f = fixture(), signedRequest = await signed("meeting.rtms_started", f.payload);
    const chunk = new Uint8Array([123]);
    const cancel = vi.fn(() => new Promise<void>(() => {}));
    const body = new ReadableStream<Uint8Array>({ start(controller) { controller.enqueue(chunk); }, cancel });
    const request = new Request(signedRequest.url, { method: "POST", headers: signedRequest.headers, body, duplex: "half" } as RequestInit);
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const pending = f.handler(request); await vi.advanceTimersByTimeAsync(2_501);
    expect((await pending).status).toBe(503); expect(cancel).toHaveBeenCalledOnce(); expect([...chunk]).toEqual([0]);
    expect(f.host.resolveEncounter).not.toHaveBeenCalled();
  });
  it("rejects declared and streamed oversized bodies before any host callback", async () => {
    const f = fixture(); const declared = await signed("meeting.rtms_started", f.payload);
    declared.headers.set("Content-Length", "64001"); expect((await f.handler(declared)).status).toBe(413);
    const original = await signed("meeting.rtms_started", f.payload); const chunk = new Uint8Array(64_001).fill(1);
    const body = new ReadableStream<Uint8Array>({ start(controller) { controller.enqueue(chunk); controller.close(); } });
    const streamed = new Request(original.url, { method: "POST", headers: original.headers, body, duplex: "half" } as RequestInit);
    expect((await f.handler(streamed)).status).toBe(413); expect(chunk.every(value => value === 0)).toBe(true);
    expect(f.host.resolveEncounter).not.toHaveBeenCalled();
  });
  it("preserves client-error status for malformed authenticated events", async () => {
    const f = fixture();
    expect((await f.handler(await signed("meeting.rtms_started", {}))).status).toBe(400);
    expect((await f.handler(await signed("meeting.rtms_started", { ...f.payload, server_urls: "not a URL" }))).status).toBe(422);
    expect(f.host.started).not.toHaveBeenCalled();
  });
});
