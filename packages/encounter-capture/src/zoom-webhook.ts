import { authorizationFailure, type CaptureAuthorization } from "./controller.js";

export interface ZoomEncounterRoute {
  accountId: string;
  meetingUuid: string;
  authorization: CaptureAuthorization;
}
export interface ZoomWebhookHost {
  webhookSecret: string;
  now?: () => number;
  /** Overall response budget; at most 2.5 seconds for Zoom's three-second deadline. */
  requestTimeoutMs?: number;
  /** External enrollment authority; never display-name or facial matching. Honor cancellation. */
  resolveEncounter: (accountId: string, meetingUuid: string, signal: AbortSignal) => Promise<ZoomEncounterRoute | null>;
  /**
   * External transaction: atomically deduplicate AND enqueue, rejecting stopped-stream tombstones.
   * Resolve only once committed. Honor cancellation before commit. A timeout after commit is safe
   * only because retries use the same key. Never connect media inline in this callback.
   */
  started: (route: ZoomEncounterRoute, streamId: string, signalingUrl: string,
    delivery: { idempotencyKey: string; expiresAtMs: number }, signal: AbortSignal) => Promise<void>;
  /**
   * External idempotent stop/tombstone transaction for this app's matching bound stream.
   * Persist terminal state so a delayed started retry cannot reopen it. Never suppress retries.
   */
  stopped: (meetingUuid: string, streamId: string, signal: AbortSignal) => Promise<void>;
}

class WebhookFailure extends Error { constructor(readonly status: number) { super("webhook-unavailable"); } }
function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new WebhookFailure(400);
  return value as Record<string, unknown>;
}
function string(value: unknown) {
  if (typeof value !== "string" || !value.length || value.length > 2048) throw new WebhookFailure(400);
  return value;
}
const encoder = new TextEncoder();
const hex = (bytes: ArrayBuffer) => [...new Uint8Array(bytes)].map(b => b.toString(16).padStart(2, "0")).join("");

/**
 * Request/Response handler factory, not a deployed endpoint or queue implementation.
 * Host ports require a real enrollment store and transactional queue.
 * https://developers.zoom.us/docs/api/webhooks/ documents signatures, the three-second
 * response deadline, and retry of 5xx responses (4xx failures are not retried).
 */
export function createZoomWebhookHandler(host: ZoomWebhookHost) {
  if (!host.webhookSecret) throw new Error("webhook-secret-required");
  const timeoutMs = host.requestTimeoutMs ?? 2_500;
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0 || timeoutMs > 2_500) throw new Error("invalid-webhook-deadline");
  return async (request: Request): Promise<Response> => {
    const reply = (status: number, body: object = {}) => Response.json(body, { status,
      headers: { "Cache-Control": "no-store", ...(status === 405 ? { Allow: "POST" } : {}),
        ...(status === 503 ? { "Retry-After": "1" } : {}) } });
    if (request.method !== "POST") return reply(405);
    const abort = new AbortController();
    const disconnected = () => abort.abort(new WebhookFailure(503));
    request.signal.addEventListener("abort", disconnected, { once: true });
    if (request.signal.aborted) disconnected();
    const timer = setTimeout(() => abort.abort(new WebhookFailure(503)), timeoutMs);
    const bounded = <T>(work: () => Promise<T>): Promise<T> => new Promise((resolve, reject) => {
      if (abort.signal.aborted) { reject(abort.signal.reason); return; }
      const cancelled = () => reject(abort.signal.reason);
      abort.signal.addEventListener("abort", cancelled, { once: true });
      const remove = () => abort.signal.removeEventListener("abort", cancelled);
      try { work().then(value => { remove(); resolve(value); }, error => { remove(); reject(error); }); }
      catch (error) { remove(); reject(error); }
    });
    const now = () => {
      const value = (host.now ?? Date.now)();
      if (!Number.isFinite(value)) throw new WebhookFailure(503);
      return value;
    };
    let chunks: Uint8Array[] = [], raw: Uint8Array | null = null, signedBytes: Uint8Array | null = null;
    let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
    try {
      const timestamp = request.headers.get("x-zm-request-timestamp") ?? "";
      const signature = request.headers.get("x-zm-signature") ?? "";
      const fresh = () => Math.abs(now() - Number(timestamp) * 1000) <= 300_000;
      if (!/^\d{10}$/.test(timestamp) || !fresh() || !/^v0=[a-f0-9]{64}$/.test(signature)) return reply(401);
      if (request.headers.get("content-type")?.split(";")[0]?.trim().toLowerCase() !== "application/json") return reply(415);
      const declaredSize = request.headers.get("content-length");
      if (declaredSize !== null && (!/^\d+$/.test(declaredSize) || !Number.isSafeInteger(Number(declaredSize)))) return reply(400);
      if (declaredSize !== null && Number(declaredSize) > 64_000) return reply(413);
      if (!request.body) return reply(400);
      reader = request.body.getReader();
      let size = 0;
      while (true) {
        const next = await bounded(() => reader!.read());
        if (next.done) break;
        chunks.push(next.value); size += next.value.byteLength;
        if (size > 64_000) return reply(413);
      }
      raw = new Uint8Array(size); let at = 0;
      for (const chunk of chunks) { raw.set(chunk, at); at += chunk.length; }
      const prefix = encoder.encode(`v0:${timestamp}:`);
      signedBytes = new Uint8Array(prefix.length + raw.length); signedBytes.set(prefix); signedBytes.set(raw, prefix.length);
      const secret = encoder.encode(host.webhookSecret);
      let key: CryptoKey;
      try { key = await bounded(() => crypto.subtle.importKey("raw", secret, { name: "HMAC", hash: "SHA-256" }, false, ["sign", "verify"])); }
      finally { secret.fill(0); }
      const received = Uint8Array.from(signature.slice(3).match(/../g)!, value => Number.parseInt(value, 16));
      if (!(await bounded(() => crypto.subtle.verify("HMAC", key, received, signedBytes!))) || !fresh()) return reply(401);
      let event: Record<string, unknown>;
      try { event = object(JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(raw))); }
      catch { return reply(400); }
      const payload = object(event.payload);
      if (event.event === "endpoint.url_validation") {
        const plainToken = string(payload.plainToken);
        return reply(200, { plainToken, encryptedToken: hex(await bounded(() => crypto.subtle.sign("HMAC", key, encoder.encode(plainToken)))) });
      }
      if (event.event !== "meeting.rtms_started" && event.event !== "meeting.rtms_stopped") return reply(200);
      const meetingUuid = string(payload.meeting_uuid), streamId = string(payload.rtms_stream_id);
      if (event.event === "meeting.rtms_stopped") {
        await bounded(() => host.stopped(meetingUuid, streamId, abort.signal)); return reply(200);
      }
      const accountId = string(payload.account_id);
      const route = await bounded(() => host.resolveEncounter(accountId, meetingUuid, abort.signal));
      if (!fresh()) return reply(401);
      // Recheck time after enrollment lookup, not only at request arrival.
      if (!route || route.accountId !== accountId || route.meetingUuid !== meetingUuid
        || authorizationFailure(route.authorization, now()) || !route.authorization.consent.permissions.derivedRetention) return reply(200);
      let signaling: URL[];
      try {
        signaling = string(payload.server_urls).split(",").map(value => new URL(value.trim()))
          .filter(url => url.protocol === "wss:" && !url.username && !url.password
            && (url.hostname.endsWith(".zoom.us") || url.hostname.endsWith(".zoom.com")));
      } catch { return reply(422); }
      if (signaling.length !== 1) return reply(422);
      // Semantic identity is stable across re-signed retries, unambiguous, and account-scoped.
      const digest = hex(await bounded(() => crypto.subtle.digest("SHA-256",
        encoder.encode(JSON.stringify([event.event, accountId, meetingUuid, streamId])))));
      if (authorizationFailure(route.authorization, now())) return reply(200);
      await bounded(() => host.started(route, streamId, signaling[0]!.href,
        { idempotencyKey: digest, expiresAtMs: now() + 86_400_000 }, abort.signal));
      return reply(200);
    } catch (error) { return reply(error instanceof WebhookFailure ? error.status : 503); }
    finally {
      clearTimeout(timer); request.signal.removeEventListener("abort", disconnected);
      // Do not await a potentially hanging underlying stream cancel implementation.
      if (reader) { void reader.cancel().catch(() => {}); try { reader.releaseLock(); } catch { /* Pending stream cancellation. */ } }
      else if (request.body && !request.body.locked) void request.body.cancel().catch(() => {});
      raw?.fill(0); signedBytes?.fill(0); for (const chunk of chunks) chunk.fill(0); chunks = [];
    }
  };
}
