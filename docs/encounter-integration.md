# Treatment-response measurement in the existing encounter

The routine product entry point is an authorized telehealth host. After normal
enrollment and integration setup, the patient and provider conduct their usual
appointment. No PhenoMetrix recording control, calibration screen, exercise,
questionnaire, treatment-date entry, baseline selection, or mandatory review is
introduced. Missing evidence stays missing. The standalone capture demo remains
an optional engineering/research surface.

This repository implements the software path and synthetic acceptance tests.
It does not establish clinical validity, qualify a Zoom deployment, provision an
EHR, or authorize use of real patient information.

## Implemented boundaries

```mermaid
flowchart LR
  H[Existing telehealth host] --> B[Scoped encounter binding and consent]
  B --> A[Embedded local or Zoom RTMS adapter]
  A --> W[Bounded background processing]
  W --> O[Durable derived observation]
  F[Structured clinical source] --> I[FHIR importer]
  I --> D[(PostgreSQL revision history)]
  O --> D
  D --> J[Transactional jobs and pure analysis]
  J --> C[Evidence panel in existing chart]
```

| Component | Implemented responsibility |
| --- | --- |
| `packages/encounter-capture` | Host lifecycle, exact participant binding, consent checks, bounded media ownership, load shedding, rotating natural-observation windows, RTMS reference transport and signed webhook handler |
| `apps/capture-web` embedded entry | Existing MediaPipe/DSP workers attached to host-owned tracks; opportunistic face calibration; no device permission requests or capture UI |
| `packages/contracts` | New durable observation, consent, binding, treatment/context revision, analysis and review contracts; ephemeral `ObservationV3` is not a persistence format |
| `apps/encounter-service` | Authenticated scoped API, PostgreSQL history, FHIR R4 import, worker leases, immutable analysis publication and optional review history |
| `packages/trajectory-core` | Automatic eligible references, interval treatment alignment, compatible descriptive differences, explicit exclusions and coverage |
| `packages/encounter-client` | Existing-session credentials, automatic capture delivery, exact-scope evidence loading and request cancellation |
| `apps/clinician-review` | Embeddable passive evidence panel; synthetic standalone preview for development |
| `packages/research-governance` | Separately consented clip retention/access/annotation boundary, disabled by default |

## Run the synthetic system

From the repository root:

```sh
pnpm install --frozen-lockfile
docker compose -f compose.encounter.yml up -d --wait
PHENOMETRIX_MODE=synthetic \
DATABASE_URL=postgresql://phenometrix:synthetic-development@127.0.0.1:55439/phenometrix_synthetic \
ALLOWED_ORIGINS=http://127.0.0.1:4173,http://127.0.0.1:4175 \
pnpm dev:encounter
```

The service at `http://127.0.0.1:4318` migrates the database, seeds a fabricated
HFS episode and runs the analysis worker. `POST /dev/session` supplies a scoped
short-lived synthetic token; it is unavailable in live mode. Source records,
jobs and reviews survive restarts. Development signing keys intentionally do
not. A local PostgreSQL 16 installation can replace Docker using a database
explicitly named `test`, `synthetic`, or `dev`.

`ALLOWED_ORIGINS` must match the exact browser origins of the actual embedding
hosts, including scheme and port. The example permits the two local development
origins; add the local host application's origin if it uses another port.

In another terminal, `pnpm dev:evidence` opens the fabricated chart preview at
`http://127.0.0.1:4175`. The preview has no camera, microphone, patient data, or
login and uses a fixed synthetic fixture. Production hosts mount the panel
with `client.loadEvidence`; the authenticated service-to-panel data path has a
separate integration test.

`pnpm dev` continues to run the previous two-capture local demo at port 4173.
That research demo is not the required entry point for ordinary telehealth.

## Embed in an existing supported host

The host must already have an authenticated appointment-to-participant mapping,
an applicable grant and a verified binding stored through the service. It
supplies a participant-specific **local, pre-codec** stream. Do not label a
decoded conferencing feed as a local camera. Hosts without this access use a
separately qualified platform adapter.

```ts
import { createEncounterClient } from "@phenometrix/encounter-client";
import { startIntegratedEncounter } from "@phenometrix/capture-web/embedded";
import { mountTreatmentResponsePanel } from "@phenometrix/clinician-review";

const client = createEncounterClient({
  baseUrl: configuration.encounterServiceUrl,
  episodeId: enrollment.episodeId,
  scope: enrollment.scope,
  accessToken: () => existingSession.scopedEncounterToken()
});

// Called by the existing telehealth client's authorized encounter-start event.
const capture = await startIntegratedEncounter({
  client,
  encounterId: appointment.encounterId,
  platformParticipantId: patient.platformParticipantId,
  stream: patient.localStream,
  video: patient.unmirroredSourceVideo,
  assetBaseUrl: configuration.phenometrixAssetBase,
  modalities: ["face"],
  subscribeLifecycle: listener => hostEvents.subscribe(appointment, listener)
});

const panel = mountTreatmentResponsePanel(existingChartElement, {
  scope: enrollment.scope,
  load: signal => client.loadEvidence(signal)
});
await panel.refresh(); // Initial host-driven chart load; no provider action.
// Use existing chart refresh events to call panel.refresh().
// On chart teardown: panel.dispose(). On abandoned capture: capture.stop().
```

Identifiers and host methods above are integration ports, not globals supplied
by this repository. The trusted host issues `ended`, `consent-withdrawn`,
`binding-lost`, and `resource-pressure` lifecycle events. Existing call-quality
signals suspend the measurement branch without restarting it during the visit.
Withdrawal and ambiguous binding discard
pending capture and cancel delivery. Clock and external consent refresh runs at
most 20 seconds apart, with earlier renewal for short-lived accuracy evidence;
a host withdrawal event stops immediately, including during final delivery.
Requests have timeouts,
and the API checks current authorization transactionally on every write.

The entry point automatically obtains three authenticated clock probes before
resolving capture context. No provider or patient clock entry is required.
For live timing qualification, configure the service's trusted UTC-monitor
attestation described in the [service runbook](../apps/encounter-service/README.md).
Transport timing alone does not prove UTC accuracy. The client includes the
full round trip, attested server error, 2 ms timestamp resolution and 100 ppm
local drift over at most 60 seconds in its engineering bound. These assumptions
require deployment qualification. Expired or inconsistent renewal stops
measurement; it never changes earlier timestamps or reduces their uncertainty.
A missing initial attestation permits unqualified capture, with treatment
alignment withheld. Synthetic clock sources cannot qualify live observations.

Voice capture can use an existing host noise reference or wait automatically
for the versioned passive noise screen. It requires natural acoustic contrast
followed by sufficient stable low-amplitude evidence; it does not use the
noise-dependent speech/SNR labels to bootstrap itself. Digital silence, unknown
or enhanced audio settings, instability, and insufficient evidence withhold
readiness. This remains an engineering heuristic, not validated nonspeech or
speaker detection. No audio, transcript, or raw calibration buffer is persisted.
Reference method and version are retained; clinical qualification remains open.

Camera/audio continuity loss closes earlier evidence before resetting the
affected calibration. Recalibration cannot qualify earlier frames. Transient
upload failures retry the identical observation revision once with a 250 ms
delay and 2.5-second attempt bounds; exhaustion stops measurement and remains
visible to host telemetry. No retry prompt is shown to the appointment.

`pnpm build` produces both the existing demo and `apps/capture-web/dist-embedded`
with its workers and static assets, plus `apps/clinician-review/dist-library`.
Serve the embedded output at a stable asset base. Existing manifest verification
checks downloaded model/worklet/WASM self-consistency; independently trusted,
execution-bound artifact delivery remains a deployment qualification item.

Completed-window extraction runs in a dedicated worker with cancellation and an
eight-second deadline, rather than calculating on the telehealth UI thread.
Opening a chart or switching tabs does not reset encounter state. Browser
throttling or suspension can reduce captured coverage; background continuity
must be measured on supported devices. The host retains ownership of tracks
and AudioContext. PhenoMetrix never stops a telehealth track or closes the
host's audio context.

## Zoom RTMS reference integration

The adapter supports the documented signaling/media handshake, individual
audio, explicit single-patient video subscription, heartbeat handling and
termination. It rejects mixed/unknown attribution, unexpected codecs/framing,
untrusted URLs and unsupported participant configurations. It does not follow
active-speaker changes. Reconnection requires fresh authorization and a new
capture epoch, rather than replaying buffered media.

`createZoomWebhookHandler` verifies the original signed request bytes, limits
body size, rejects stale request timestamps, answers URL-validation challenges
and resolves only an exact configured account/meeting instance to enrollment.
Its host callback must atomically deduplicate and enqueue using the supplied
event key, preserve a stopped-stream tombstone, and return promptly. The ingress
server must enforce request timeouts; the adapter never exposes app secrets to
a browser. These deployment-owned callbacks are deliberately required, with no
accept-all meeting fallback.

The reference adapter is tested with synthetic protocol exchanges. A live
server decoder/measurement processor, Zoom app credentials, host/account
permissions and verified scheduling/participant mapping still require target
deployment integration. The HFS specification **does not qualify RTMS-derived
measurements by default**. Provider codecs, denoising, camera changes and
sampling characteristics must be validated independently.

Use encounter eligibility to initiate streams through authorized host controls;
do not enable capture for every meeting hosted by a clinician. Platform notices
remain explicit. Zoom documents account/host auto-start controls and individual
video subscriptions in its [stream lifecycle documentation](https://developers.zoom.us/docs/rtms/meetings/work-with-streams/).
Signature/validation behavior follows [Zoom webhooks](https://developers.zoom.us/docs/api/webhooks/)
and the [RTMS event schema](https://developers.zoom.us/docs/api/rtms/events/).

## Clinical records and response analysis

Deployment-configured code mappings import actual administrations/procedures,
prescriptions and medication statements. The sync connector also fetches
Encounter resources, but the current normalizer excludes them as unsupported;
appointment and participant binding comes from the existing host integration.
Actual treatment remains
distinct from orders and reported use. Unknown dose, formulation, sites,
laterality and timing remain unknown; no cross-product unit conversion occurs.
The importer preserves source version, content hash, original timing and
trusted receipt time. Conflicting or ambiguously ordered corrections fail with
explicit reconciliation errors; they never create patient prompts.

`synchronizeFhirPatientContext` performs bounded authenticated patient-scoped
reads from a configured HTTPS FHIR source. Invoke it from existing encounter or
record-change events, not a provider button. Remote pagination cannot escape
the configured origin/path. See the [service runbook](../apps/encounter-service/README.md)
for terminology configuration, API contracts and authentication requirements.
Concurrent observation writes no longer abort a valid clinical import on the
first revision conflict: synchronization retries up to three import attempts
with the same source event and already-fetched bundle. Each attempt rechecks
authorization and source/patient binding. The host can supply its existing
session revalidation callback; revoked access and changed binding are not retried.

The current HFS protocol selects four existing engineering metrics: left/right
resting eye aperture and lid closure completeness. It does not implement an
HFS spasm classifier, weakness diagnosis or clinical severity score. Independent
clinician annotations remain separate from ambient measurements.

References are selected by versioned protocol rules. The default HFS protocol
requires at least two eligible observations from two separate baseline encounters;
one visit remains descriptive-only. Protocols permitting one reference label it
as a single reference. Baseline support counts independent encounters, not
five-minute windows. Pre-cycle references retain prior exposure
uncertainty. Unknown or unverified treatment timing leaves calendar observations
visible without treatment alignment. Missing visits produce gaps. No fitted
curve, onset, recurrence, durability or forecast is enabled, and a difference
after treatment does not establish causation.

All analyses are content-addressed immutable snapshots. Corrections enqueue a
new revision; a stale worker cannot publish over newer input. The panel checks
scope and digest and distinguishes generated/reviewed/correction-requested
evidence. Source corrections create new records; generated values are never
edited in place. Inspection and review are optional existing-chart activities.

## Media and qualification gates

Ordinary raw media stays in bounded transient processing buffers and has no
HTTP upload or persistence path. Do not configure logging, crash dumps or
telemetry to retain payloads. Optional `ResearchClipVault` requires separate
consent, observation ownership, capture-time eligibility, access authority,
encrypted object storage, durable cleanup metadata and an audit sink. Pending
uploads carry cleanup intent before bytes are written. Withdrawal/expiry denies
access immediately and the configured retention worker physically deletes media.

Before enrolling a real pilot, independently demonstrate: correct patient track
selection for caregivers/shared links/multiple faces; no added in-visit actions;
no adverse call-performance effect; natural-context repeatability; codec/device
effects; clock mapping; routine-only missingness and treatment-cycle coverage.
Qualify each measurement/acquisition context. Stronger claims remain disabled
until that evidence exists. This follows the fit-for-purpose approach in
[FDA digital health guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/digital-health-technologies-remote-data-acquisition-clinical-investigations).

## Verification

```sh
pnpm verify
TEST_DATABASE_URL=postgresql://phenometrix:synthetic-development@127.0.0.1:55439/phenometrix_synthetic \
pnpm test:postgres
```

`verify` covers structure/assets, unit tests, typechecking, both application and
embedded builds, browser tests and the optional Python voice service. The
separate PostgreSQL suite uses an isolated disposable schema in the specified
test database; CI runs it against PostgreSQL 16. It tests transactions,
concurrency, immutable history, reconnect/recovery and stale-worker fencing.
Synthetic host integration tests cover authenticated capture ingestion through
analysis and chart evidence, with no manual capture, baseline or review step.
The connected browser test runs the real controller, finalization worker,
signed HTTP service, trajectory engine and chart panel across multiple windows,
camera gaps, hidden-chart use, a transient upload failure, scope refusal and
withdrawal. It fabricates media/model outputs and UTC monitoring; it establishes
software behavior, not actual hardware throughput, call quality or clinical accuracy.
