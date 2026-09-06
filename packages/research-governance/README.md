# Optional research-media boundary

Routine capture never calls this package and never retains raw media.
`ResearchClipVault` is disabled by default. Enabling it requires explicit positive
retention/size/duration limits, an encrypted blob-store adapter, durable metadata,
current research consent, authenticated study/participant authorization, and an
audit sink. No provider credentials or media-store implementation are bundled.

A trusted resolver verifies ownership of the referenced derived observation.
The clip's entire capture interval must be covered by its separate research
grant. Cleanup metadata is committed before upload; failed or interrupted
uploads retain a deletion intent. The blob adapter must honor the supplied
expiry and cancellation signal, including its own working copies.
An upload's lease cannot outlast object expiry. A provider that completes after
cleanup is rejected and compensated; physical expiry and abandoned uploads still
require a storage provider that honors the adapter contract.

Retention and annotation permissions are separate. Every read checks current
consent and expiry after asynchronous retrieval/auditing and immediately before
the callback; callbacks receive temporary byte buffers cleared on exit.
Scope mismatches fail closed. `purgeUnavailable` deletes expired or withdrawn
media before recording a tombstone, permitting retry after a failed physical
deletion. Wire that method into the deployment's retention worker and consent
withdrawal handler. Access denial is immediate; physical deletion depends on that
worker and the configured storage provider's behavior, including backup policy.

The caller transfers ownership of buffers passed to `retain`. Annotation code
must not copy bytes to logs, analytics, unrelated stores, or model-training jobs.
The vault cannot erase copies made outside its boundary. Clinical derived
observations remain separate; institutional policy governs retention of already
completed annotations and analyses.
