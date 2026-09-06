# Unilateral Facial Palsy Research Prototype — Implementation Plan

Status: implemented; automated gates passed, manual hardware acceptance pending

Date: 2026-08-21

Target: a useful, locally runnable, nonclinical prototype in roughly 5–7 focused engineering days

## Implementation result — 2026-08-21

The toy prototype is implemented end to end. Two independently consented live
captures now produce validated ObservationV3 artifacts, an explicitly accepted
page-memory reference, a strict six-row current-minus-reference comparison, and
a deterministic research evidence card with page-local accept/dismiss state.

The implementation also added the versioned condition profile and digest gate,
strict comparison/card contracts, a v3-native trajectory package, canonical
source-binding checks, fixed display labels, generated-frame integration
coverage, two-capture Playwright coverage, static prohibited-language checks,
and destructive cleanup for reset, discard, withdrawal, visibility loss, and
reload. The generated 30-second sign-crossing scenario produces four measured
rows, two visibly withheld rows, and zero incompatible rows; the resting mouth
value crosses from `+0.04` to `-0.04` with a raw delta of `-0.08`.

Implementation deviation: condition state and rendering were separated into
`condition-demo-controller.ts` and `condition-evidence-view.ts`, while the
existing capture finalization boundary remained in `main.ts` rather than being
moved into the optional `encounter-finalizer.ts` named below. The runtime still
drops derived frame arrays and destroys media/native resources before result
rendering, and the behavior is covered at unit and browser levels.

Automated status at completion: all 420 workspace unit tests, TypeScript
typechecks, production build, structure/digest/static-copy gates, and all 10
Playwright scenarios pass. The packaged `pnpm demo:smoke` command also exits
cleanly. The manual checks in section 8 have not been performed on named real
camera/microphone hardware, so this is a runnable research prototype—not a
hardware-accepted or clinically validated release.

## 1. Executive decision

Build the first condition-oriented prototype around **previously established unilateral peripheral facial palsy**. The user-facing name should be **Unilateral Facial Movement Research Demo** so the software does not imply that it diagnoses facial palsy or grades its severity.

The prototype will reuse the existing ambient browser capture and its facial engineering measurements. It will add one narrow longitudinal function: compare a genuinely live current capture with one explicitly accepted, compatible prior capture from the same in-memory demo participant. The only analytic is the raw numerical difference between the two sessions, with quality, missingness, and provenance visible.

This is the smallest increment that connects the project's three capabilities:

```text
Ambient Capture
  -> validated ObservationV3
  -> accepted in-memory reference observation
  -> compatible current-minus-reference comparison
  -> deterministic research evidence card
```

It is not the separate, clinically validated facial-palsy protocol pack described in the historical design. It is a condition-oriented research view over the existing `ambient-local-observation` nonclinical pack.

## 2. Why this condition

This condition is the best fit for the repository as it exists:

- The live face pipeline, quality gates, abstention behavior, provenance, and relevant scalar metrics already exist.
- Left-versus-right facial measurements are within-frame contrasts, so both sides share the same camera, lighting, pose, and time point.
- The repository already contains a reviewed facial-palsy measurement design and eleven related engineering primitives.
- A useful demonstration does not require a new model, backend, database, LLM, population reference range, or clinical scoring system.

Validated clinician instruments such as Sunnybrook and eFACE are appropriate future research comparators, but this prototype must not claim to reproduce or equal either instrument.

## 3. Current starting point

Already implemented:

- browser consent, independent device setup, and bounded calibration;
- live ambient capture with local face and voice processing;
- deterministic face and voice extraction with reason-coded abstention;
- a validated `ObservationV3` with exact protocol, algorithm, processor, window, measurement, and aggregate provenance;
- a deterministic 27-outcome post-encounter report;
- device/worker teardown before report display;
- session-memory-only retention and explicit discard/reset behavior;
- extensive unit coverage and mocked browser smoke coverage.

Missing at plan approval:

- a condition/demo context and participant-asserted anatomical side;
- stable anonymous subject linkage across two captures in one page lifetime;
- explicit acceptance of a prior observation as the comparison reference;
- a v3-native compatibility and comparison engine;
- a condition-focused evidence card and review action;
- browser tests for a two-capture path;
- real-browser/hardware acceptance for the condition view.

Known limitations that remain true at prototype completion:

- The metrics have technical tests but no clinical validation, test-retest error, or minimum detectable change.
- Browser automation mocks media and workers; it does not establish real MediaPipe or hardware performance.
- The current expression-event detector and baseline differ from the historical facial-palsy design.
- The application does not verify identity or the asserted affected side.
- There is no durable participant identity, persistence, import/export, authentication, PHI handling, or EHR integration.

## 4. Prototype user flow

1. The user selects the sole condition demo and reads the nonclinical/emergency boundary.
2. The user identifies the side previously understood to be affected. Store this as `participant-asserted` and `unverified`; never infer it from measurements.
3. The user consents and completes a 60–90 second live ambient capture. Do not prompt a smile, brow raise, or eye closure.
4. The application stops devices and processors, creates `ObservationV3`, validates provenance, and renders the existing report.
5. The user may select **Use as in-memory reference**. Only the structured observation is retained in page memory.
6. The user starts a second capture. It receives a new `sessionId` and consent record but keeps the same anonymous `subjectRef` and demo context.
7. After teardown, the application compares compatible outcomes and renders a focused research evidence card.
8. The user may accept or dismiss the card for the current page session.
9. **New participant**, **discard all**, page reload, or tab loss clears the reference and all derived state.

The two captures may be performed minutes apart for a repeatability demonstration. The interface must say that their numerical difference is not evidence of health change.

## 5. Condition-focused measurements and analytics

### 5.1 Primary rows

Use only existing outcomes:

| Existing metric | Display role |
| --- | --- |
| `ambient.face.rest_mouth_corner_asymmetry.signed` | Primary signed anatomical mouth measure |
| `ambient.face.rest_eye_aperture_asymmetry.signed` | Primary signed anatomical eye measure |
| `ambient.face.lid_closure_completeness.left` | Subject-left closure measure; tag the asserted side visually |
| `ambient.face.lid_closure_completeness.right` | Subject-right closure measure; tag the asserted side visually |
| `ambient.face.spontaneous_excursion_asymmetry.median` | Experimental dynamic measure; show only when measured |
| `ambient.face.oculo_oral_synkinesis_index` | Experimental dynamic measure, displayed only as **Oculo-oral coupling difference** |

`ambient.face.brow_height_asymmetry.signed` may remain available in the underlying engineering report, but it is not a prototype headline because the demo must not imply central-versus-peripheral discrimination or stroke screening.

Do not headline `ambient.face.spontaneous_excursion.p90`; it is an absolute larger-side magnitude and does not meet the historical condition design's within-face rule. Event rate is supporting evidence, not a condition result.

### 5.2 Permitted analytics

For each compatible measured row, show:

- anatomical sign convention (`subject-left minus subject-right`) or explicit left/right label;
- prior value;
- current value;
- raw `current - prior` difference in the native unit;
- current and prior technical quality;
- eligible duration, usable-bin count, expression-event count where relevant;
- exact source outcome, measurement, aggregate, and evidence-window references;
- an exact reason when either result is withheld or the prior is incompatible.

Do not calculate or display:

- a composite facial-function score;
- percent change for signed, near-zero quantities;
- trend slopes from two points;
- normative ranges, thresholds, alerts, or traffic-light status;
- `improved`, `worsened`, `recovered`, `progressed`, or `clinically meaningful`;
- diagnosis, severity, prognosis, cause, treatment, or emergency guidance;
- `synkinesis detected`, `lagophthalmos`, `forehead sparing`, or scale equivalence.

Dynamic metrics must preserve the current protocol's `insufficient-events` abstention. Missing data remains a visible gap and is never imputed.

## 6. Contracts and architecture

Add contracts before UI wiring.

### 6.1 Demo context

Add `packages/contracts/src/condition-demo.ts` with a strict, versioned `ConditionDemoContextV1` containing:

- a condition-profile ID/version/digest;
- the anonymous `subjectRef`;
- the participant-asserted side (`left` or `right`) and `verified: false`;
- the source `ProtocolRef`;
- the allowlisted metric codes;
- the exact nonclinical boundary and prohibited-claim list;
- `persistence: "page-memory-only"`.

This is a research/demo profile, not a clinical protocol pack. It must reference rather than rename the generic measurement pack.

### 6.2 Accepted reference

Add `AcceptedReferenceV1` containing the demo context, the validated source observation, acceptance timestamp, and local reviewer action. Acceptance must not change the observation or manufacture a measurement.

### 6.3 Previous-visit comparison

Add `packages/contracts/src/previous-visit-comparison.ts` with:

- current and reference observation IDs and timestamps;
- inclusion/exclusion decision for every allowlisted metric;
- exact compatibility reason codes;
- measured, withheld, or incompatible terminal status;
- prior/current values and native-unit delta only when both are eligible;
- source evidence references for both sessions;
- `analyticalRepeatability: "unknown"` and `minimumDetectableChange: "unknown"` literals;
- a literal claim boundary that forbids health-state interpretation.

### 6.4 Research evidence card

Add a strict `ConditionEvidenceCardV1` contract. It should contain structured rows, the comparison artifact, quality summary, review status, and source references. Narrative text must come from fixed templates; no LLM is needed.

### 6.5 Compatibility policy

A reference outcome is comparable only when all of the following match:

- `subjectRef`;
- condition-profile ID/version/digest and asserted side;
- measurement protocol pack ID/version/content digest;
- metric code, context, and unit;
- algorithm version;
- processor reference and verified asset identity;
- capture-adapter ID/version;
- measured status in both observations.

Every mismatch receives a stable reason code. Never bridge protocol or algorithm versions by assumption.

## 7. Dependency-ordered implementation

### Milestone 0 — Freeze the reviewed baseline (0.5 day)

Tasks:

- Reconcile and checkpoint the current review fixes before layering trajectory work onto the large pending diff.
- Run `pnpm run check`, `pnpm test`, `pnpm test:browser`, and `pnpm demo:smoke`.
- Record the current protocol digest and known browser-test limitations.

Exit criteria:

- The existing single-capture flow passes all repository gates.
- No condition code is added until lifecycle teardown and report provenance are green.

### Milestone 1 — Contracts and condition profile (0.5–1 day)

Files:

- `packages/contracts/src/condition-demo.ts`
- `packages/contracts/src/previous-visit-comparison.ts`
- `packages/contracts/src/condition-evidence-card.ts`
- `packages/contracts/src/condition-prototype.test.ts`
- `packages/contracts/src/index.ts`
- `packages/condition-profiles/src/unilateral-facial-movement.ts`
- `packages/condition-profiles/src/index.ts`

Tasks:

- Define the contracts in section 6.
- Publish the allowlist and safe display labels in one versioned profile.
- Add canonical digest generation and validation for the profile.
- Reject unknown metrics, missing side provenance, non-finite deltas, duplicate rows, and claim-like free text.

Exit criteria:

- Schemas round-trip valid artifacts and fail closed on invalid provenance or prohibited fields.
- The existing measurement pack, thresholds, and metric labels are not relabeled as clinically validated.

### Milestone 2 — Minimal Personal Trajectory slice (1–1.5 days)

Create `packages/trajectory-core` as a fresh v3-native package:

- `src/compatibility.ts`
- `src/compare-previous.ts`
- `src/index.ts`
- package, TypeScript, and Vitest configuration
- focused unit tests

Tasks:

- Validate both observations and the demo profile at the boundary.
- Require the reference to precede the current observation.
- Apply the exact compatibility policy.
- Calculate only deterministic native-unit differences.
- Preserve withheld values and incompatibilities without fallback or interpolation.
- Return exact evidence references and reason codes.

Exit criteria:

- Tests cover left/right side preservation, time ordering, subject mismatch, protocol/digest mismatch, algorithm mismatch, processor mismatch, adapter mismatch, withheld prior/current outcomes, non-finite inputs, and deterministic replay.
- No code path emits a directional health interpretation.

### Milestone 3 — Evidence-card builder (0.5–1 day)

Files:

- `packages/evidence-core/src/condition-card.ts`
- `packages/evidence-core/src/condition-card.test.ts`
- `packages/evidence-core/src/index.ts`

Tasks:

- Combine the current report, accepted reference, and comparison artifact.
- Render fixed measurement-only statements.
- Include quality, missingness, compatibility decisions, and exact traceability.
- Add page-session accept/dismiss review state without a clinician signature claim.

Exit criteria:

- Every displayed number resolves to a validated source outcome.
- A withheld or incompatible row cannot produce a value or delta.
- The internal `synkinesis_index` code is always presented with the safe display label.

### Milestone 4 — Two-capture browser flow (1.5–2 days)

Refactor the current 1,500-line entry point while integrating:

- `apps/capture-web/src/encounter-finalizer.ts`
- `apps/capture-web/src/condition-demo-controller.ts`
- `apps/capture-web/src/condition-evidence-view.ts`
- updates to `main.ts`, `index.html`, and `styles.css`

Tasks:

- Separate observation/report finalization from rendering and state disposal.
- Keep an accepted structured reference in page memory after all media/native state is destroyed.
- Reuse the anonymous `subjectRef` for a follow-up capture while generating a new `sessionId` and consent record.
- Add `Use as reference`, `Capture follow-up`, `Accept card`, `Dismiss card`, and `New participant / discard all` actions.
- Add the emergency boundary: not for sudden facial weakness or stroke screening.
- Keep the full engineering report inspectable below the focused card.

Exit criteria:

- Two live captures link only through the explicit in-memory demo context.
- Devices, workers, frames, timers, and native observations are gone before either report/card appears.
- Reload, reset, discard, consent withdrawal, visibility loss, or new participant clears reference state.
- No local storage, session storage, IndexedDB, network write, or export is introduced.

Implementation note: the current generic capture pack includes both camera and microphone. The prototype may initially reuse that consented flow, but voice outcomes must never influence the condition card. A camera-only observation/consent contract should be designed as the next privacy-hardening increment rather than silently misrepresenting the current v3 source contract.

### Milestone 5 — Fixtures, browser tests, and documentation (1–1.5 days)

Tasks:

- Add v3-native, code-generated facial-geometry fixtures. Do not reuse archival guided/v2 fixtures or patient media.
- Run fixtures through the real extractor, adapter, report builder, comparator, and card builder rather than hard-coding UI values.
- Add mirrored left/right cases, no-expression abstention, pose/scale rejection, one-sided closure, incompatible history, and sign-crossing comparison cases.
- Extend Playwright for the two-capture state machine and prohibited-language assertions.
- Update the root README, architecture, safety, validation, operator guide, demo experience, and all three capability status READMEs.

Exit criteria:

- `pnpm run check`, `pnpm test`, `pnpm test:browser`, and `pnpm demo:smoke` pass.
- Mirrored fixtures yield exact sign inverses where the metric definition requires it.
- `Not measurable` remains a terminal, visible result.
- Static checks find no prohibited clinical language in the condition UI.

## 8. Manual acceptance and go/no-go

For the first runnable release, narrow the support statement to the exact Mac model and current Chrome version actually tested.

Required manual checks:

- three consecutive nominal 60–90 second camera sessions;
- anatomical-left/right verification despite the mirrored preview;
- dim/back lighting, excessive pose, distance change, zero/multiple faces, camera denial/interruption, tab hiding, discard, reset, and maximum-duration behavior;
- expected abstention when fewer than three expression events occur;
- a five-minute memory/performance soak with no unbounded growth;
- keyboard operation, visible focus, screen-reader labels, and non-color-only state;
- comprehension checks that `Not measurable` is not a normal result and a raw delta is not health improvement or worsening.

Go only if:

- all automated and manual gates pass;
- every value/delta is traceable to both observations;
- laterality is correct in live and mirrored synthetic cases;
- the reference survives only within the intended page lifetime;
- teardown and deletion behavior pass every terminal path;
- the UI contains no diagnostic, severity, recovery, stroke, or treatment claim.

No-go for a longitudinal health-change claim until same-subject repeatability, measurement error, and minimum detectable change are characterized under the real ambient protocol.

## 9. Explicitly deferred beyond this toy prototype

- A separately versioned, clinically validated facial-palsy protocol pack.
- Corrections/calibration for the historical expression-event and resting-baseline divergences.
- Affected-minus-unaffected transforms or automatic affected-side inference.
- Robust multi-visit baselines, MAD, slopes, uncertainty, or meaningful-change thresholds.
- Durable local or server persistence, import/export, identity verification, PHI, authentication, or EHR/FHIR integration.
- Population norms, House-Brackmann/Sunnybrook/eFACE score reproduction, or patient-reported instruments.
- LLM narrative, clinical signature, treatment suggestions, messaging, orders, or actions.
- Acute facial-weakness triage or stroke screening.

After the toy prototype is accepted, the next engineering decision should be whether to prioritize (a) camera-only contracts and privacy hardening, (b) durable derived-observation history plus robust trajectory statistics, or (c) correcting the condition-specific measurement algorithm. Those should not be bundled into this first runnable increment.

## 10. Definition of done

The prototype is done when a developer can run one command, open the browser, complete and accept one live reference capture, complete a second live capture, and inspect a deterministic unilateral-facial-movement evidence card containing condition-relevant current values and raw compatible-session differences—or exact reasons why each value was not measurable or comparable. All processing and comparison remain local and page-memory-only, all media/native state is destroyed before results appear, and the product makes no clinical interpretation.

## 11. Future validation anchors

Future clinical research may compare the prototype's measurements against:

- Ross, Fradet, and Nedzelski's Sunnybrook facial grading system development: <https://pubmed.ncbi.nlm.nih.gov/8649870/>
- Banks et al.'s eFACE clinician-graded instrument: <https://pubmed.ncbi.nlm.nih.gov/26218397/>
- Video-versus-in-person eFACE reliability work: <https://pubmed.ncbi.nlm.nih.gov/28006048/>
- Mehta et al.'s Synkinesis Assessment Questionnaire for a separate patient-reported research layer: <https://pubmed.ncbi.nlm.nih.gov/17473697/>

These are external validation anchors only. The toy prototype neither implements nor claims equivalence to them.
