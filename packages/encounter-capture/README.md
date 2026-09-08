# Encounter capture integration

This package adds host-controlled capture to the research prototype. It does not identify a patient from their appearance, diagnose a condition, or establish clinical validity.

## Available paths

| Path | Implemented behavior | Qualification boundary |
| --- | --- | --- |
| Embedded local browser | The existing telehealth host supplies its patient `MediaStream`, associated video element, and running `AudioContext`. The concrete bridge in `apps/capture-web/src/embedded-encounter.ts` uses the existing MediaPipe worker, audio worklet/DSP worker, frame scheduler, calibration assessment, and `ambient-core` extraction. | Host enrollment must bind this exact local participant. Static assets must pass the existing integrity loaders. Real-world capture/device/model and clinical validation remain separate. |
| Zoom RTMS reference | Signed-webhook boundary plus actual signaling/media WebSocket handshakes, HMAC signatures, separate participant audio, explicit patient-video subscription, keepalives, disconnect handling, and owned transient packets. | A deployment must supply authenticated enrollment resolution, credentials, an atomic deduplicate-and-enqueue transaction, and a qualified server media decoder/processor. No live Zoom connection or server measurement processor is claimed by the mock protocol tests. |

`createEmbeddedEncounter` attaches an analysis branch without calling `getUserMedia`, starting a login flow, adding buttons, or stopping the host's tracks. `startIntegratedEncounter` in capture-web composes it with the encounter-service client, host lifecycle events, and derived observation delivery. It is the application entry point for an existing telehealth host; the legacy standalone demo remains separate.

## Authorization, attribution, and lifecycle

`EncounterCaptureController` requires an active encounter, scoped consent, permitted modalities, and a verified binding to a specific platform participant. A display name or active speaker is never an identity source. The generic controller supports transient analysis; `AmbientEncounterWindow` additionally requires permission to retain derived observations.

The browser bridge verifies that `video.srcObject` contains the exact host video track. It checks this at acquisition/result boundaries and polls source/settings changes. Replaced tracks, changed settings, audio-clock discontinuity, loss of authorization, and consent expiry stop capture and discard pending windows. Multiple faces invalidate calibration and withhold face data. Camera availability is independent of voice availability. Page visibility has no application-level effect; browser suspension/throttling can still create missing observations.

Normal encounter end permits finalizing the already authorized window. Withdrawal, binding loss, expiry, interruption, and explicit discard suppress undelivered observations and abort their delivery signals. A host persistence callback must honor that signal, and the server must independently recheck current consent/binding: a browser cannot recall a write already accepted by another process.

## Bounded data and measurements

- Raw packets have a 4 MiB default size limit, two-second maximum age, one pending item per modality, and a two-second processor deadline. Rejections, timeout, and stop release media; owned byte views are zeroed and native frames are closed. Processor implementations must erase their own working copies on disposal.
- The embedded bridge uses the existing bounded worker buffers and transfers each image once. It retains bounded derived primitives for at most a five-minute measurement window. The whole encounter can continue for any duration through automatic rotation and `onObservation(observation, signal)` callbacks.
- Completed-window extraction runs in a dedicated browser worker with an eight-second deadline and cancellation. Up to two observation deliveries may be pending, with a ten-second deadline. Delivery failure aborts further capture. A runtime suspension remains a gap; the code does not invent intervening data.
- Terminal output contains only authorized modalities, scalar measurements or explicit withheld reasons, analysis-window references, and provenance. It contains no media, transcript, landmark array, or voice-frame array. `temporalSamples` is empty because the current extractor does not provide qualified per-bin measurements; no episode timing is fabricated.

Acquisition timestamps use a fixed monotonic encounter map, separate from the window-relative analysis origin. An optional host `EncounterClockCalibrationV1` supplies the local measurement/expiry times, UTC offset, uncertainty, source identity/kind, and measured round-trip time. `updateClockCalibration(next)` renews the same source for at most 60 seconds without rewriting timestamps or reducing the accumulated uncertainty. Source changes, inconsistent offset bounds, wall-clock discontinuity, expiry, and suspended audio clocks stop capture. A failed external refresh may retain the prior bound until expiry. Without an initial estimate, the bridge records an unqualified uncertainty sentinel (`Number.MAX_SAFE_INTEGER`); it cannot adopt a new offset mid-encounter. Device IDs are included only in a participant-scoped hash with actual track settings and verified asset manifests; no raw hardware ID enters a durable observation. Browser settings are not hardware attestation.

Voice readiness uses an existing host noise reference when available. Otherwise, the versioned `passive-screened-noise@1.0.0` estimator waits for strong periodic acoustic contrast followed by two seconds of stable low-amplitude, low-pitch-confidence evidence with reduced cepstral structure. It counts disjoint analysis windows and ignores floor-dependent speech/SNR labels. Only bounded scalar features are retained. Digital silence, missing features, browser audio enhancement or unknown settings, insufficient contrast, instability, and acquisition loss withhold readiness. This is an engineering screening heuristic, not validated nonspeech detection: whispers and other nearby speakers cannot be excluded conclusively. Both reference methods are marked `engineering-only`; passive results require separate clinical qualification.

Face calibration uses actual worker primitives. Pre-calibration frames are excluded; later calibration never qualifies earlier frames retroactively. Camera or audio continuity loss closes existing evidence before resetting the affected calibration, preserving earlier valid segments. Renewed metadata applies to current/future windows; already-finalizing observations retain their snapshot. Incomplete terminal facial bins are omitted. No patient calibration action or extra media prompt is introduced.

## Zoom deployment notes

The webhook handler uses standard `Request`/`Response`, bounded signed request bodies, timestamp tolerance, and an external host-owned atomic deduplicate-and-enqueue transaction. Its enrollment resolver must authorize the specific meeting/stream/participant before connection. These required host ports are not a deployed endpoint. Host callbacks honor cancellation and preserve stopped-stream tombstones. Secrets and RTMS signature generation belong on the server.

The transport requests 48 kHz mono L16 separate audio and JPEG video for an explicitly subscribed individual. JPEG is limited to 5 fps and therefore cannot meet higher-cadence blink requirements. Mixed audio, active-speaker video fallback, unexpected codecs/encryption, unapproved endpoints, interrupted state, and missing keepalives fail closed. There is no automatic reconnect or buffered replay. Platform media must be labeled `platform-patient-track`, never `patient-local-pre-codec`.

The raw-media processor interface is intentionally unimplemented for Zoom: absent a qualified processor, the controller disposes packets and emits `processor-unavailable`. H264, alternate audio formats, encrypted payloads, transcription, screen sharing, and cloud recording are unsupported. JavaScript cannot erase immutable WebSocket/base64 strings; their references are callback-local, while decoded owned byte buffers are explicitly wiped. Deployments must also disable raw-payload logging and recording in their infrastructure.

Protocol references checked against primary Zoom documentation:

- [WebSocket connection quickstart](https://developers.zoom.us/docs/rtms/meetings/quickstart-rest-api/)
- [Event/message reference](https://developers.zoom.us/docs/rtms/event-reference/)
- [Media parameter definitions](https://developers.zoom.us/docs/rtms/media-parameter-definition/)
- [Enum definitions](https://developers.zoom.us/docs/rtms/data-types/)
- [Handling media data](https://developers.zoom.us/docs/rtms/meetings/media/)
- [Official raw connection sample](https://github.com/zoom/rtms-samples/blob/main/RTMS_CONNECTION_FLOW.md)

## Verification

Run `pnpm --filter @phenometrix/encounter-capture test:unit` and `pnpm --filter @phenometrix/capture-web test:unit`. Controller tests use synthetic media leases; Zoom tests use explicit mock WebSockets. Embedded tests simulate worker outputs while exercising the real controller, frame pump/protocol, window rotation, and ambient-core extraction. They prove wiring, deterministic calculations, and rejection/disposal behavior, not microphone/model accuracy, live Zoom interoperability, or clinical performance.
