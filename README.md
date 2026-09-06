# PhenoMetrix

> Nonclinical research prototype. Not a medical device. Not for diagnosis,
> treatment, emergency detection, or use with protected health information.

PhenoMetrix derives bounded, quality-aware face and voice measurements from an
ordinary conversation, in the browser, without recording it.

A clinician watching a video visit reads a great deal from how someone looks and
sounds — facial symmetry, blink rate, vocal effort, how long they can speak
before drawing breath. Almost none of it reaches the record, because it is hard
to quantify consistently and impossible to quantify the same way twice. The aim
here is to turn those transient observations into measurements that can be
compared: to the other side of the same face, to the start of the same session,
and eventually to the same patient last month.

Everything is built on one rule: **measure only technically qualified signal,
report `Not measurable` otherwise, and dispose of the media before showing the
report.** An abstention with a reason code is a first-class result, not a
failure.

## Why contrast, not absolute value

Every measurement here earns its keep by cancelling a confound rather than by
being accurate in isolation:

| Contrast | Cancels | Status |
|---|---|---|
| Left vs. right, within one frame | lighting, camera, distance, individual anatomy | implemented |
| Early vs. late, within one session | all of the above, plus mood, medication timing, effort | substrate in place |
| Previous live session vs. current live session | all of the above, plus a within-page reference | implemented narrowly; no durable visit history |

Absolute values across people are where the confounds live. A measurement that
compares a face to itself reduces that problem, which is why the first
condition-oriented measurement work targets unilateral facial nerve palsy —
an indication where the finding *is* an asymmetry.

## Current implementation

The implemented browser path can run one generic ambient session or the full
two-capture unilateral facial movement research demonstration:

```text
participant-asserted affected side + consent
  → independent camera and microphone permission
  → bounded technical calibration
  → first live ambient observation (up to five minutes)
  → ObservationV3 + session-only structured report, after media disposal
  → explicit acceptance as the page-memory reference
  → a second, independently consented and calibrated live observation
  → second ObservationV3 + structured report, after media disposal
  → strict six-row current-versus-reference comparison
  → page-memory-only research evidence card and optional accept/dismiss
  → discard all on reset, visibility loss, or reload
```

There are no exercises, scripted prompts, LLM calls, server APIs, retained
recordings, transcripts, embeddings, durable persistence, export, or clinical
interpretation in this path. The microphone remains part of the generic
ambient capture during both sessions; the condition comparison allowlists six
face metrics and does not compare voice outcomes.

### Ambient Capture

`apps/capture-web` uses two independent local processing lanes:

- Audio is captured in 20 ms worklet blocks and analyzed in a worker using 40
  ms windows with a 10 ms hop. Only compact `VoiceSignalFrameV1` values cross
  into application state. Those same derived frames drive an eight-second live
  level and pitch display; the display is not a provisional report.
- MediaPipe Face Landmarker runs in a worker. Native video frames, landmarks,
  and transformation matrices remain inside that boundary. The worker draws its
  complete 478-point mesh and contours directly onto a transferred presentation
  canvas, while only compact `FacialKinematicsFrameV1` geometry and quality
  values are emitted.

  Blendshapes are deliberately **not** computed. They are the obvious shortcut
  to Action Unit intensities and the wrong instrument for this: the rig is
  trained for avatar retargeting and carries a symmetry prior that suppresses
  exactly the left-right difference being measured. Action Units are derived
  geometrically from landmarks instead.

Permission, calibration, measurement, and abstention are independent by
modality. One lane can continue when the other is unavailable.

### Three tiers

Session metrics are the smallest of three representations, not the only one:

| Tier | Rate | Content | Boundary |
|---|---|---|---|
| Substrate | ~100 Hz voice / analyzed cadence face | per-frame geometric and acoustic vectors | extractor memory only; not retained |
| Event | per blink, expression, breath-group | kinematic parameters | extractor memory only; not retained |
| Summary | per session | the 27 published metrics | crosses any boundary |

The design principle for a future durable substrate is to **store physical
quantities, not clinical constructs**. The current browser persists none of
these tiers: per-frame and event records exist transiently inside the extractor
and are discarded after the session report is built. A construct like a palsy grade is terminal; a quantity like
"nasolabial angle, left versus right, over time" recombines. Every clinical
scale is a function of quantities, so an archive of quantities can produce a
scale invented after the data was collected — an archive of constructs cannot.

Concretely: a blink is not a count. It is a closing edge, a closed interval, and
a reopening edge. Reduced rate is hypomimia, shallow depth is incomplete
closure, delayed reopening is fatigable — one waveform, three findings, none of
them recoverable from a number.

### Active metric registry

The immutable `ambient-local-observation` protocol pack is content-addressed by
a SHA-256 digest over its own canonical form, and contains exactly 27
nonclinical metrics across 10 report sections: pitch, speech timing, eye
geometry, mouth geometry, symmetry, expression dynamics, brow geometry,
movement, blink behaviour, and capture quality.

Every metric carries its unit, context, algorithm version, evidence
requirements, permitted withheld reasons, technical-verification status, and
`clinicalValidation: "none"`. Every evidence requirement the pack publishes is
re-verified at the report boundary against the same statistic the extractor
enforced; a metric that cannot produce the evidence its own pack entry demands
fails provenance rather than passing quietly.

### Observation and report

`buildAmbientObservation()` converts extractor outcomes into the strict
`phenometric.encounter-observation.v3` schema. Each terminal metric outcome is
either measured or withheld and resolves to exact evidence windows, processor
and track provenance, and a deterministic aggregate identity.

`buildPostEncounterReport()` validates that provenance against the active
protocol pack and creates a ten-section structured report for each capture.
The report is screen-only and exists only in page memory.

For the condition demo, the first ObservationV3 can be explicitly accepted
from its report screen as the one in-memory reference.
`@phenometrix/trajectory-core` then compares the second ObservationV3 with that
reference only when subject, condition profile,
asserted side, protocol, capture adapter, metric context/unit/algorithm, and
processor provenance are compatible. Each of the six rows terminates as
`measured`, `withheld`, or `incompatible`. Only a measured row exposes both
source values and the raw native-unit `current - reference` difference;
withheld and incompatible rows expose no delta.

`@phenometrix/evidence-core` projects that comparison into a deterministic
six-row card with fixed profile labels, source traces, quality counts, and a
page-local pending/accepted/dismissed state. It does not generate narrative or
assign clinical meaning.

## Capability status

1. **Ambient Capture:** implemented as the local v3 prototype described above;
   the same generic camera-and-microphone workflow produces each live
   ObservationV3.
2. **Personal Trajectory:** implemented only as one deterministic comparison of
   an explicitly accepted live reference with a later live observation in the
   same page. There is no importer, durable participant identity, baseline,
   trend, or multi-visit history.
3. **Clinician Evidence Card:** implemented as the deterministic 27-outcome
   session report plus the fixed six-row condition card. Accept/dismiss is a
   local research-demo action, not authenticated clinician approval or a
   durable review record.

The superseded v2 `trajectory-core` implementation was removed in July 2026.
The current package is a new ObservationV3-native, strictly compatible
previous-session comparator; it does not restore the old unordered history
model.

The restored `services/voice-inference` WavLM service is an optional,
disabled-by-default research surface. The browser does not import or call it.

## Privacy and safety boundary

- Consent is required before device access.
- Camera and microphone permissions are requested separately.
- Raw media is not uploaded or written to storage.
- PCM, spectral arrays, transcripts, embeddings, native landmarks, and native
  video frames are excluded from ObservationV3 and report contracts.
- Device tracks, workers, audio nodes, timers, derived frame buffers, and the
  in-memory event journal are disposed on finish, discard (which is also the
  in-session consent-withdrawal path), visibility loss, or reset.
- Identity is not verified and speaker attribution is explicitly unverified.
- `Not measurable` is a valid terminal result; missing evidence is never
  imputed as a measurement.

## Deliberately not implemented

- durable or more-than-one-reference history, baseline, trend, or import;
- retained evidence snippets or clips;
- narrative generation or authenticated/durable clinician review;
- authentication, PHI workflows, EHR/FHIR integration, or export;
- diagnosis, progression classification, risk prediction, or treatment advice;
- analytical or clinical validation against a reference standard;
- clinical validation of any protocol pack, including facial palsy;
- voluntary-movement grading (House-Brackmann / Sunnybrook equivalence).

## Refused capability

Acute stroke screening is a standing product boundary, not a gap. Forehead
sparing—the discriminator between central and peripheral facial weakness—is not
measurable here, and an ambient capture that abstains on low quality is the
wrong shape for an emergency instrument. See `docs/safety.md`.

## First condition-oriented measurement substrate

The runnable condition-oriented profile is the **Unilateral Facial Movement
Research Demo** for an adult who asserts a previously established unilateral
peripheral facial palsy. It uses spontaneous expression captured ambiently
rather than elicited movement. The affected side is participant-asserted and
unverified; the application never infers it.

The profile selects exactly six face metrics from the generic, nonclinical
`ambient-local-observation` pack, in a fixed order: resting mouth difference,
resting eye-aperture difference, subject-left and subject-right lid closure
completeness, spontaneous excursion difference, and experimental oculo-oral
coupling difference. The last label is deliberately safer than the internal
metric code: the UI does not claim that synkinesis was detected.

This is a versioned nonclinical demo profile, not a clinical protocol pack. It
has no reference standard, validated claim, repeatability estimate, minimum
detectable change, governed clinical workflow, or scale equivalence. Raw
current-minus-reference differences cannot be interpreted as improvement,
worsening, recovery, progression, severity, or clinically meaningful change.

Full design, including the new primitives required, the contract implications,
and why equivalence to House-Brackmann and Sunnybrook is explicitly not
claimed: `docs/superpowers/specs/2026-07-24-facial-palsy-protocol-pack-design.md`.

## Run locally

Requirements: Node.js 22.12+ on the Node 22 line (or Node 24+), pnpm 9.12.3,
and current Chrome on macOS.

```bash
pnpm install --frozen-lockfile
pnpm dev
```

Open `http://127.0.0.1:4173`. Camera and microphone access requires localhost
or HTTPS. Select the participant-asserted affected side, complete and accept a
first live session as the in-memory reference, then complete the second live
session to see the condition card. Consent is collected again for the second
capture. `Ctrl-C` stops the Vite development server.

The live implementation currently targets current Chrome on macOS with working
camera and microphone hardware, but the two-capture condition flow has not yet
completed the named-hardware manual acceptance checklist. Permission policy,
secure-context rules, AudioWorklet, worker, `OffscreenCanvas`,
WebGL/hardware-acceleration, and device behavior can vary across environments;
no real-hardware support statement is made yet.

The optional WavLM research service has separate instructions in
`services/voice-inference/README.md`; starting it does not alter browser
behavior.

## Validate

```bash
pnpm run check
pnpm test
pnpm test:browser
pnpm demo:smoke
uv sync --project services/voice-inference --extra dev --locked
uv run --project services/voice-inference --extra dev python -m pytest services/voice-inference/tests
```

`pnpm test` runs structure and static-asset checks, the protocol-pack digest
check, all unit tests, TypeScript typechecking, and the production build.
Browser smoke tests and the optional Python service remain separate CI jobs.

The pack digest is regenerated with `pnpm --filter
@phenometrix/capture-web exec tsx ../../scripts/protocol-digest.mjs --write`
and enforced by `--check` in the repository gate. Changing any pack
content — including the consent wording, whose SHA is a field inside the pack —
requires regenerating it and bumping the pack version, deliberately.

## Repository map

```text
apps/capture-web/          static ambient browser application
apps/clinician-review/     documentation-only future surface
packages/ambient-core/     deterministic face and voice extractors
packages/condition-profiles/ versioned unilateral movement demo profile
packages/contracts/        v3, comparison, card, and provenance schemas
packages/evidence-core/    report and deterministic condition-card builders
packages/event-log/        session-only workflow journal
packages/trajectory-core/  strict two-observation comparison engine
services/voice-inference/  optional disconnected WavLM research service
agents/                    exactly three capability boundary documents
protocols/ and examples/   archival guided/v2 demo artifacts
```

See `docs/architecture.md`, `docs/safety.md`, and `docs/validation.md` before
changing an active boundary.
