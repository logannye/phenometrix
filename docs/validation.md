# Validation

PhenoMetrix has technical verification only. Automated tests do not establish
analytical validity, clinical validity, clinical utility, or regulatory
fitness.

## Required commands

```bash
pnpm install --frozen-lockfile
pnpm run check
pnpm test
pnpm test:browser
pnpm demo:smoke
uv sync --project services/voice-inference --extra dev --locked
uv run --project services/voice-inference --extra dev python -m pytest services/voice-inference/tests
git diff --check
```

`pnpm run check` validates the active ambient-v3 and condition-demo structure,
exactly three capability directories, absence of tracked media, required JSON
manifests, the committed static-asset digests, and the content digest of the
unilateral facial movement demo profile.

`pnpm test` runs all workspace unit tests, TypeScript typechecks, and the
production build. Browser and Python tests are separate because they have
different runtimes and CI jobs.

The checked-in Playwright lifecycle suite replaces media devices, audio
processing, workers, and image capture with deterministic mocks. It verifies
the browser workflow, teardown contract, and two-capture page-memory condition
flow, but it does not prove real MediaPipe/AudioWorklet measurement or
real-hardware repeatability.

## Automated coverage

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
  behavior, discard, withdrawal, and late-stream cleanup; and
- optional WavLM health, CORS, request validation, and transient summary output
  using a deterministic fake adapter.

## Manual hardware acceptance

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

No live media or resulting health-related artifact may be saved or committed.
Manual acceptance applies only to current Chrome on the target macOS hardware;
localhost/HTTPS permission behavior, actual camera and microphone devices,
AudioWorklet, workers, `OffscreenCanvas`, WebGL, and hardware acceleration must
all be exercised. Passing mocked browser tests does not extend support to other
browsers or hardware.

## Not validated

No metric has reference-standard accuracy, repeatability, normative ranges,
minimum detectable change, disease association, subgroup performance, or
clinical workflow evidence. The two-capture arithmetic is technically tested,
but its deltas have no established health meaning. Every active metric remains
`clinicalValidation: "none"`, and the condition profile has no validated claim.
