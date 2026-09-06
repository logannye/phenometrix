# Existing-host service client

`createEncounterClient({baseUrl, episodeId, scope, accessToken})` reuses a scoped
session supplied by the clinic's existing host application. It never opens a
login, captures media, adds reminders, or requests patient information.

- `captureContext(encounterId)` resolves current enrollment consent and binding.
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
