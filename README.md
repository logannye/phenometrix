# PhenoMetrix

> Nonclinical research prototype. Not a medical device. No validated diagnostic,
> treatment, emergency-detection, or disease-severity claims. Local development
> uses synthetic participants; real-patient deployment requires separate
> institutional authorization and integration work.

PhenoMetrix derives quality-aware face and voice measurements from ordinary
telehealth conversation. Its three capabilities are **Ambient Capture**,
**Personal Trajectory**, and **Clinician Evidence Card**.

The default integration direction is an analysis branch inside the patient's
existing telehealth client, imported treatment context, and evidence available
without interrupting care. The repository implements that integration substrate
alongside the original local demo. It does not include a provisioned Zoom
installation or a connected clinical EHR.

See the [integration runbook](docs/encounter-integration.md),
[architecture](docs/architecture.md), [development baseline](docs/development-baseline.md),
and [validation boundaries](docs/validation.md). The
[platform vision](docs/telehealth-platform-vision.md) describes the broader direction.

## Implemented encounter path

```text
existing clinic identity, encounter binding, and modality-specific consent
  → authorized branch on the host's patient media tracks
  → local face/voice workers and bounded transient primitives
  → up to five-minute derived observations throughout the encounter
  → scoped API + append-only PostgreSQL history + leased analysis worker
  → immutable history snapshot + protocol + treatment-alignment specification
  → descriptive points, qualified baseline differences, and phase coverage
  → nonblocking evidence panel and durable clinician review
```

`createEmbeddedEncounter()` uses a host-owned stream, video element, and running
audio context. It does not request devices, open another login, add a capture
screen, or stop capture merely because the page becomes hidden. Encounter
lifecycle, source attribution, authorization, and resource failure govern
capture. Host tracks retain their existing owner. `startIntegratedEncounter()`
connects capture to the scoped service client; hosts must supply their actual
authenticated encounter integration.

Treatment-time qualification also requires a measured host clock calibration;
the default timing uncertainty is explicitly unqualified.

The service implements signed tenant/study/participant authorization, explicit
grants and bindings, idempotent ingestion, append-only corrections, transactional
analysis jobs, worker leases, stale-publication protection, and source-linked
reviews. A FHIR R4 normalizer and configured-source sync preserve administration
versus order status, dose units, revisions, and date precision. EHR credentials,
terminology mappings, patient matching, and scheduling remain external.

A Zoom RTMS reference adapter implements transport and participant-attribution
boundaries. Transport tests do not qualify Zoom-derived measurements. The initial
HFS protocol excludes platform patient tracks pending acquisition validation.

## Initial treatment-alignment research protocol

The new protocol targets research with adults who already have a clinician
diagnosis of hemifacial spasm and receive botulinum injections. It compares four
existing engineering measurements: left/right eye aperture and left/right lid
closure completeness. These are **not spasm counts, spasm burden, clinical
severity, or proof of treatment response**.

The deterministic core uses a sealed protocol/specification and immutable
history. It selects a compatible pretreatment baseline, aggregates by encounter,
preserves source versions and exclusions, flags concurrent treatments and
possible carryover, and stops index-cycle comparisons at a subsequent documented
injection. Unknown treatment timing permits technically qualified calendar
observations but disables treatment alignment and baseline deltas.

Sparse data remains sparse. There are no fitted curves, interpolated
observations, causal effects, peak-response or wearing-off estimates, dose
recommendations, or automatic clinical actions. Measurement error and minimum
detectable change are unknown. Quality thresholds and windows are engineering
choices awaiting validation.

## Existing local demo

`pnpm dev` retains the unilateral facial movement research demonstration:

```text
participant-asserted affected side + consent
  → independent camera/microphone permission and calibration
  → first live capture → ObservationV3 + report after media disposal
  → explicit acceptance as the page-memory reference
  → second independently consented live capture
  → strict six-row comparison + page-local accept/dismiss
  → discard on reset, visibility loss, or reload
```

This demo has no server requests, durable history, retained recordings,
transcripts, embeddings, or authenticated review. Voice contributes to its
generic 27-outcome report; the condition comparison selects six facial metrics.
Its affected side and identity are unverified. Review buttons update page memory.
The new durable contract is separate; legacy ObservationV3 remains unchanged.

Both paths reuse the versioned `ambient-core` registry: 7 voice and 20 facial
metrics. Measurements require qualified source windows; otherwise they are
withheld with reasons. Native landmarks, image frames, and raw audio stay outside
observation contracts. Facial geometry is derived in the worker.

## Privacy and clinical boundaries

- Capture, analysis, retention, and permitted modalities require explicit
  authorization. An HFS session can authorize face only.
- Routine capture retains derived observations through the governed service;
  it does not upload raw media, transcripts, voiceprints, or embeddings.
- Optional research clips use a separate, disabled-by-default governance
  package requiring separate consent, retention policy, encrypted storage,
  authorization, and audit adapters. No production media store is bundled.
- Evidence access checks current source consent. Corrections create new
  versions and analyses instead of rewriting prior evidence.
- Software controls do not establish patient identity proofing, institutional
  approval, measurement validity, or clinical readiness.

Acute stroke screening remains a standing product exclusion. See
[safety.md](docs/safety.md). No metric supports disease grading or equivalence
to House-Brackmann, Sunnybrook, or an HFS clinical scale.

## Run and verify

Use Node 22.12+ on the Node 22 line (or Node 24+), pnpm 9.12.3, and current
Chrome on macOS for the browser prototype.

```bash
pnpm install --frozen-lockfile
pnpm dev
```

The legacy demo runs at `http://127.0.0.1:4173`. Start the synthetic evidence
preview with `pnpm dev:evidence` at `http://127.0.0.1:4175`. The durable synthetic
service requires PostgreSQL; follow the [integration runbook](docs/encounter-integration.md)
and [service README](apps/encounter-service/README.md) before `pnpm dev:encounter`.
Do not use real patient data in synthetic development.

```bash
pnpm verify
pnpm test:postgres
git diff --check
```

`pnpm verify` runs structure/digest checks, unit tests, TypeScript, production
builds, browser suites, and isolated Python tests. It requires Chrome and
uv/Python 3.11. PostgreSQL tests run separately against a configured test database.
Mocked media tests do not establish real-device or platform support. The optional
[WavLM service](services/voice-inference/README.md) remains disabled by default
and disconnected from browser capture.

## Repository map

| Path | Responsibility |
| --- | --- |
| `apps/capture-web` | Legacy local demo and embedded host capture entry points |
| `apps/encounter-service` | Scoped API, PostgreSQL, workers, FHIR boundaries |
| `apps/clinician-review` | Embeddable evidence panel and synthetic preview |
| `packages/ambient-core` | Deterministic face and voice extraction |
| `packages/contracts` | Legacy v3 and separate durable contracts |
| `packages/condition-profiles` | Facial movement demo and HFS research protocol |
| `packages/encounter-capture` | Capture controller, ambient bridge, RTMS reference |
| `packages/encounter-client` | Existing-host scoped API client |
| `packages/trajectory-core` | Legacy pair comparison and multi-visit analysis |
| `packages/evidence-core` | Source-preserving reports and evidence projections |
| `packages/research-governance` | Optional research-media boundary |
| `packages/event-log` | Legacy session-only journal |
| `services/voice-inference` | Optional disconnected WavLM service |
| `protocols`, `examples` | Archival guided/v2 artifacts |
