# Validation

PhenoMetrix has technical verification only. Automated tests do not establish
analytical validity, clinical validity, clinical utility, or regulatory
fitness.

## Required commands

From the repository root, `pnpm verify` runs workspace checks, both browser
suites, and the optional voice service's isolated tests. It requires uv/Python
in addition to the browser prerequisites. PostgreSQL integration tests run
separately against a configured test database. `pnpm demo:smoke` runs only the
legacy capture browser suite; it does not cover the new evidence preview.

```bash
pnpm install --frozen-lockfile
pnpm run check
pnpm test
pnpm test:browser
pnpm demo:smoke
pnpm test:postgres
uv sync --project services/voice-inference --extra dev --locked
uv run --project services/voice-inference --extra dev python -m pytest services/voice-inference/tests
git diff --check
```

`pnpm run check` validates repository structure, exactly three capability
directories, absence of tracked media, required manifests, committed static-asset
digests, and the legacy condition profile digest. New treatment-response
protocol/specification and snapshot/run digests are also exercised by their
unit tests and verified at runtime boundaries.

`pnpm test` runs all workspace unit tests, TypeScript typechecks, and the
production build. Browser and Python tests are separate because they have
different runtimes and CI jobs.

The legacy Playwright lifecycle suite replaces media devices, audio processing,
workers, and image capture with deterministic mocks. The clinician panel suite
uses fabricated evidence. Embedded-host unit/integration tests simulate media
and worker boundaries while exercising real derived-data extraction and service
composition. None proves live MediaPipe/AudioWorklet performance, real-hardware
repeatability, or a deployed telehealth integration.

PostgreSQL tests use an actual configured database and isolated test schemas.
They are distinct from the in-memory repository's unit tests. See the
[service README](../apps/encounter-service/README.md) for database configuration
and the [integration runbook](encounter-integration.md) for end-to-end setup.

## Encounter architecture coverage

- Strict durable observation, consent, binding, treatment/context revision,
  protocol/specification, snapshot/run, and review contracts; scope and digest
  validation; evidence-window and temporal consistency.
- Deterministic history ordering/replay, explicit correction chains,
  encounter-weighted baseline selection, exact source compatibility, unknown
  treatment anchors, sparse coverage, concurrent treatment and cycle boundaries,
  and explicit exclusions without fitted or causal estimates.
- Face-only consent, exact participant attribution, stale worker suppression,
  consent expiry/revocation, bounded raw ownership, rotating window rebasing,
  persistence failure, and preservation of host-owned tracks/audio contexts.
- RTMS handshake/participant transport fixtures and unsupported-media rejection;
  these do not qualify RTMS measurements for the HFS protocol.
- Signed scoped authorization, idempotent source writes, immutable histories,
  revision conflicts, consent-sensitive evidence access, leased worker fencing,
  failed jobs, and exact-run durable review.
- FHIR patient/source checks, completed versus planned treatments, original
  effective dates and trusted receipt time, date precision, dose units, version
  ordering, corrections, and configured-source synchronization.
- Evidence projection, calendar/treatment-time presentation, source/quality
  inspection, stale refresh/abort handling, safe text rendering, and synthetic
  preview behavior.
- Disabled-by-default research-media governance, separate retention/annotation
  permissions, scope/observation binding, expiry/withdrawal access, transient
  buffers, audit, and deletion retry behavior with test adapters.

These are engineering contract checks, not evidence that the four HFS metrics
measure spasm frequency, burden, clinical severity, or treatment benefit.

## Retained local-demo coverage

- voice and face quality thresholds and abstention;
- deterministic 7-voice/20-face metric registry ordering;
- exact reason-code projection and source-window intervals;
- strict ObservationV3, protocol, report, consent, evidence-ref, and workflow
  event schemas;
- canonical aggregate and measurement identities;
- provenance and report grounding;
- the immutable unilateral movement profile, its six fixed labels/order, and
  participant-asserted/unverified affected-side policy;
- exact previous-session compatibility across subject, profile, protocol,
  asserted side, capture adapter, metric, and processor provenance;
- one terminal measured/withheld/incompatible result for each of six condition
  rows, with raw `current - reference` only for compatible measured sources;
- a generated 30-second-per-session facial-geometry integration path through
  the real extractor, ObservationV3 adapter, report builder, comparator, and
  card builder, including signed mouth reversal and one-sided closure;
- trace-only excluded sources, so withheld or incompatible rows cannot expose
  measured values or a delta;
- deterministic evidence-card source preservation, quality summary, safe
  **Oculo-oral coupling difference** label, and page-local review schema;
- session journal ordering, causal boundaries, replay, and disposal;
- capture lifecycle races, teardown order, and bounded setup/session timers;
- asset-path and runtime digest verification;
- a static scan that rejects prohibited clinical interpretation language in
  the condition-demo UI source;
- browser consent, affected-side gating, permission denial, independent
  calibration, report display, explicit reference acceptance, second live
  capture, six-row card, page-local accept/dismiss, reset, no-upload/no-storage
  behavior, discard, withdrawal, pagehide/restoration without a visibility
  event, and late-stream cleanup; and
- optional WavLM health, CORS, request validation, and transient summary output
  using a deterministic fake adapter.

## Embedded-host acceptance still required

Exercise the actual authorized telehealth host on named hardware before any
live pilot; do not substitute the standalone demo for this acceptance:

1. Confirm existing enrollment, signed scope, modality-specific consent, and
   exact patient-track binding without a new capture screen or device prompt.
2. Verify host tracks continue normally when analysis ends, fails, or withdraws.
   Test track replacement, source-element replacement, mute/unmute, multiple
   faces, permission loss, and suspended audio contexts.
3. Verify hidden-page operation, long encounters, window rebasing, bounded
   memory, calibrated clock uncertainty, missed intervals, and worker failure.
4. Exercise remote consent changes, in-flight delivery cancellation, server
   revision conflicts, disconnect/retry, worker crashes, and database restarts.
5. Verify chart evidence belongs to the authenticated participant, suppresses
   stale current results, preserves source quality/exclusions, and binds review
   to the exact run. Missing evidence must not block routine care.
6. Reconcile actual clinical source terminology, patient identity, dose units,
   source corrections, timezone precision, concurrent therapies, and cycles.
7. Keep RTMS sources excluded from the HFS protocol until platform acquisition
   and metrics are qualified. Verify any separately enabled research-media
   deployment's consent, storage, audit, deletion, and backup policies.

## Legacy demo hardware acceptance

In current Chrome on the target MacBook:

1. Select a participant-asserted affected side, consent, and complete one live
   camera-and-microphone session.
2. Confirm each lane calibrates or abstains independently.
3. Confirm the complete facial mesh tracks exactly one face and clears for zero
   or multiple faces.
4. Confirm quiet, unvoiced noise, and voiced speech produce the expected live
   voice state, energy response, and periodic-only pitch trace.
5. Confirm the end button is enabled only during ambient observation.
6. Verify both live displays clear and camera/microphone indicators turn off
   before the report appears.
7. Confirm the first report contains exactly 27 terminal outcomes and 10
   sections.
8. Explicitly accept it as the page-memory reference. Start the follow-up and
   confirm the affected side is preserved and locked while consent is required
   again.
9. Complete the second live camera-and-microphone session; confirm the generic
   microphone lane remains present and teardown precedes the second report.
10. Confirm the second report again has 27 outcomes/10 sections and the
    condition card has exactly six rows in profile order.
11. For every condition row, confirm the terminal status is measured, withheld,
    or incompatible. Only measured rows may show reference/current values and a
    raw native-unit `current - reference` delta.
12. Confirm **Oculo-oral coupling difference** is used as the safe display
    label and no copy assigns improvement, worsening, recovery, progression,
    severity, diagnosis, or clinical significance.
13. Accept or dismiss the card, then choose **New participant · discard all**
    and confirm the reference, review, side assertion, and report state clear.
14. Discard another session and verify no report appears.
15. Confirm mid-session consent withdrawal is performed through Discard and
   verify that it stops all tracks.
16. Accept a new reference, then hide or reload the page and confirm no
    comparison state remains.

Do not save or commit live media or health-related artifacts during local-demo
acceptance. Separately authorized research retention uses its governed path,
not this demo. Manual acceptance applies only to the tested Chrome/macOS setup;
localhost/HTTPS permission behavior, actual camera and microphone devices,
AudioWorklet, workers, `OffscreenCanvas`, WebGL, and hardware acceleration must
all be exercised. Passing mocked browser tests does not extend support to other
browsers or hardware.

## Not validated

No metric has established reference-standard accuracy, repeatability, normative
ranges, minimum detectable change, disease association, subgroup performance,
or clinical workflow utility. The two-capture and multi-visit arithmetic are
technically tested; their deltas have no established health meaning. Every
active metric and the HFS protocol remain clinically unvalidated. No study has
established peak response, wearing off, causal efficacy, dosing guidance, or
equivalence to an HFS or facial-palsy clinical scale for this implementation.
