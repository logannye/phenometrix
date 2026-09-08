# Existing-host service client

`createEncounterClient({baseUrl, episodeId, scope, accessToken})` reuses a scoped
session supplied by the clinic's existing host application. It never opens a
login, captures media, adds reminders, or requests patient information.

- `captureContext(encounterId)` resolves current enrollment consent and binding.
- `calibrateClock(encounterId, signal)` validates three scoped authenticated
  probes against a current server UTC-error attestation. It returns a bounded,
  expiring estimate or `null` when no accuracy evidence is available.
- `ingestObservation(observation)` submits a new durable record with its revision
  as the idempotency key; no raw-media request is supported.
- `loadEvidence(signal)` maps current service evidence to the chart panel loader.
  Pending, withdrawn, and stale results are not presented as current evidence.

HTTPS is required except for loopback development. Credentials stay in bearer
headers, redirects are rejected, requests have bounded deadlines, and returned
scope/binding/revision identities are checked. The host owns token renewal and
its existing authenticated session; never store tokens in URL queries or logs.

Deadlines include token retrieval, response parsing and integrity verification;
canceled work cannot deliver a late result. A client instance rejects evidence or
capture contexts older than a revision it has already accepted. Capture contexts
include the episode's measurement protocol reference. Ingest acknowledgements
must identify the exact submitted observation, and displayed runs must match the
episode's sealed protocol/specification; reviews are bound to the returned run.

Clock estimates conservatively include full round-trip time, the attested source
error, timestamp resolution and bounded drift. Disjoint bounds, wall-clock
steps, source changes and expired candidates cannot establish qualification.
Consent timing uses the current estimate conservatively. Fresh observations
may wait up to two seconds before upload so an admissible positive timestamp
error cannot violate the server's knowledge-time rule. Waiting is cancellable;
recorded measurements and timestamps are never rewritten. Automatic renewal and
upload retry/stop orchestration live in `startIntegratedEncounter`.
