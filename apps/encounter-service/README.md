# Encounter service

Versioned derived observations and clinical source revisions for a clinician
research pilot. The HTTP service and worker connect to PostgreSQL. Raw media
upload is disabled; separately consented research media belongs to the
independent governed media path.

## Synthetic development

Use a local database explicitly named `test`, `synthetic`, or `dev`. These commands
create only fabricated participant records:

```sh
PHENOMETRIX_MODE=synthetic \
DATABASE_URL=postgresql://logannye@127.0.0.1:55439/phenometrix_test \
ALLOWED_ORIGINS=http://127.0.0.1:4173,http://127.0.0.1:4175 \
pnpm --filter @phenometrix/encounter-service dev
```

The launcher applies migrations, seeds `synthetic-hfs-episode` if absent, and
runs the HTTP service and one worker at `http://127.0.0.1:4318`. A separate worker
can run with the same environment using `pnpm --filter
@phenometrix/encounter-service worker`. Database restarts retain inputs, jobs,
evidence and review history. Development signing keys are ephemeral, so signed
development sessions must be refreshed after an application restart.

`POST /dev/session` returns a short-lived signed synthetic session and episode
ID, only in explicit synthetic mode and only on loopback. Never put that token
in a URL or persist it in source files. Use `Authorization: Bearer <token>` for
all `/v1` requests. The in-memory repository is for isolated tests, not a
fallback when PostgreSQL is unavailable.

Configure `ALLOWED_ORIGINS` for the exact existing host origins. The example
uses the local capture and chart development ports; replace or extend them for
the actual embedding application. A service's default port-5173 origins do not
authorize the port-4173/4175 applications automatically.

## Host API

| Method and path | Input or result |
| --- | --- |
| `POST /v1/episodes` | `{id, scope, dataClass, protocol, specification, fhirPatientReference?}`; protocol and specification must be sealed versioned contracts |
| `GET /v1/episodes/:id` | Episode metadata, current input revision and source counts |
| `GET /v1/episodes/:id/capture-context/:encounterId` | Exact scoped current grant, verified encounter binding and measurement protocol |
| `POST /v1/episodes/:id/consents` | `{data: DerivedDataConsentV1, expectedRevision?, supersedesRecordId?}` |
| `POST /v1/episodes/:id/bindings` | Same envelope containing `ParticipantBindingV1` |
| `POST /v1/episodes/:id/sessions` | Same envelope containing `DurableObservationV1` |
| `POST /v1/episodes/:id/clinical-events` | Same envelope containing `TreatmentRevisionV1` or `ClinicalContextRevisionV1` |
| `POST /v1/episodes/:id/annotations` | Separate routine-care reference or governed clip annotation, excluded from ambient metrics |
| `POST /v1/episodes/:id/fhir` | `{source, bundle, expectedRevision?}` for a deployment-configured FHIR R4 source |
| `GET /v1/episodes/:id/evidence` | `{episode, inputRevision, status, snapshot, reviews, jobs}` |
| `GET /v1/episodes/:id/history` | Same envelope plus authorized historical snapshots and routine-care reference annotations |
| `POST /v1/episodes/:id/reviews` | `{snapshotId, inputRevision, disposition, supersedesReviewId?, reasonCode?}` |

All source/review POST requests require an `Idempotency-Key`. Same-key retries
return the original record. Different content with the same key is rejected.
Corrections retain the previous record and must explicitly identify its
predecessor. Requests checked before a concurrent change fail with a revision
conflict, including a consent withdrawal racing capture ingestion.

`snapshot.analysis` is `TreatmentResponseRunV1`; `reviews` contains shared
`TreatmentResponseReviewV1` objects. The server binds reviewer identity, scope,
run ID and content digest. Pending revisions expose no stale current snapshot.
The history endpoint filters old evidence against the current consent of its
actual measurement sources; an unrelated new grant cannot reveal withdrawn
evidence. Blinded clip annotations require a configured exact-clip authority
callback that verifies ownership, expiry and annotation permission. They are
disabled by default and do not appear in routine evidence history.

## Persistence and workers

Each accepted batch acquires an episode row lock, commits immutable source
records and a job in the same transaction, and supersedes stale pending/running
jobs. A worker claims a bounded lease with `FOR UPDATE SKIP LOCKED`; a unique
lease token and input revision fence publication. Crashed workers can be
reclaimed. Three analysis failures leave an explicit failed job. Analyses
preserve complete revision chains, the explicit analysis time and input hash.
Old evidence is never rewritten when a source is corrected.

Database triggers enforce append-only input/evidence/review rows and immutable
episode identity, protocol and specification. Grants/bindings have immutable
envelope versions: reconstruct a prior analysis by selecting records through
its input revision and resolving authorization versions within that revision.
The clinical engine independently validates scope, source provenance, consent,
binding, modality, revision order, protocol compatibility and data sufficiency.
Evidence and capture-context reads have a separate append-only access audit
with scoped actor, time, action, result and source revision. Audit records do
not include tokens or clinical payloads and do not generate analysis jobs.

## Live integration boundary

Live launch fails unless all of these are explicitly configured:

- `PHENOMETRIX_MODE=live` and `PHENOMETRIX_LIVE_ENABLED=1`.
- PostgreSQL `DATABASE_URL` with `sslmode=verify-full` and a trusted CA.
- `OIDC_ISSUER`, `OIDC_AUDIENCE`, and an HTTPS `OIDC_JWKS_URL`.
- Exact HTTPS browser origins in `ALLOWED_ORIGINS`.

Only asymmetric RS256/ES256 sessions from that configured issuer and audience
are accepted. The trusted identity integration must issue signed `tenantId`,
`studyIds`, `subjectRefs`, `role`, `permissions` and `dataClass` claims alongside
`sub`, `iat` and `exp`. This is an identity integration boundary, not an
implemented patient login or identity-proofing service. Unsigned identity
headers are never accepted. Hosts must not forward an unrestricted IdP token
without participant/study authorization mapping.

The launcher registers the current HFS protocol digest. Programmatic live
hosts must explicitly configure `allowedProtocolDigests`; a self-created hash
does not authorize a new live protocol. No cloud deployment, institutional
authorization, real patient enrollment or clinical validation is supplied by
these controls.

## Clinical source configuration

`FHIR_SOURCE_CONFIG_PATH` can point to an uncommitted deployment-owned JSON
file mapping source names to:

```json
{
  "example": {
    "sourceSystem": "https://ehr.example/fhir",
    "dateOffset": "-07:00",
    "treatmentCodes": [
      {"system": "urn:example-only", "code": "synthetic-injection", "product": "Synthetic product", "kind": "botulinum-injection"}
    ]
  }
}
```

The example coding is synthetic. Real terminology mappings require institution
review. Unmapped names and free-text narratives never establish an injection.
`MedicationAdministration` and mapped `Procedure` resources count as actual
administration only when complete. Requests remain planned; medication
statements remain reported clinical context. Product-specific dose units are
preserved without conversion. Remote `lastUpdated`, content hash and original
effective date are source provenance; trusted receipt time is knowledge time.
Source-version corrections and entered-in-error retractions remain explicit.
Previously unseen older or ambiguously ordered versions require reconciliation.
A correction that changes clinical record class or makes a previously used
source unmapped/incomplete is surfaced as an explicit reconciliation error;
it is never silently skipped or inserted as an invalid revision chain.

Exact date-times preserve their timezone. Day-only values need a configured
source offset and remain day intervals, not exact midnight injections. Partial
dates, missing timezone and unsupported intervals remain unknown. A fixed
offset must not be reused across dates with different daylight-saving offsets;
such deployments should supply full source date-times or use unknown timing.
The separate `fhir-sync` connector supports bounded configured-source reads;
real EHR credentials, scopes, patient matching and provisioning remain external.

## Verification

```sh
pnpm --filter @phenometrix/encounter-service test:unit
pnpm --filter @phenometrix/encounter-service typecheck
TEST_DATABASE_URL=postgresql://logannye@127.0.0.1:55439/phenometrix_test \
pnpm --filter @phenometrix/encounter-service test:postgres
```

PostgreSQL tests allocate and remove a uniquely named test schema. They verify
concurrent idempotency, transactional rollback, worker fencing, database
immutability, authorization-sensitive writes, durable reviews and reconnect
behavior against actual PostgreSQL rather than a mocked SQL client.
