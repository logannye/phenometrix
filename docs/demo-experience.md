# Live demonstration experience

The current demonstration is two genuinely live ambient sessions in one browser
page. The first can be explicitly accepted as the comparison reference; the
second produces a fixed six-row unilateral facial movement research card. It
contains no guided exercise sequence or operator-only synthetic capture mode.

## 1. Welcome and consent

The opening page states that the application is a nonclinical prototype, uses
local processing, saves no recording or result, and does not verify identity,
affected side, or clinical meaning. It is for a participant who asserts a
previously established unilateral peripheral facial palsy, not for new or
sudden facial weakness.

The participant selects subject-left or subject-right as the previously
understood affected side. This assertion is unverified and the application
never infers it. The setup button remains disabled until one side is selected
and the local camera-and-microphone consent checkbox is checked.

## 2. Independent setup

The browser requests microphone and camera separately. Each lane displays its
own requesting, calibrating, ready, or not-measurable state. A camera failure
does not invalidate a usable voice lane, and vice versa.

Audio calibration asks for a brief quiet moment. Face calibration asks for one
well-lit, frontal face. Setup is bounded to 15 seconds.

## 3. First live ambient session

When at least one lane qualifies—or a capture-capable lane reaches the bounded
setup timeout—the interface switches to **Ambient session**. The participant
continues the ordinary conversation without prompts. The session ends manually
or at five minutes.

When one face is visible, the camera preview shows all 478 MediaPipe landmarks,
the full tessellation, and the eye, iris, brow, lip, and face-oval contours.
This mesh is drawn inside the face worker, follows the mirrored preview, and is
never retained or returned as landmark coordinates.

The panel beside the video shows an eight-second rolling energy and periodic
pitch display plus current level, pitch, SNR, F0 confidence, estimator
agreement, activity state, and signal-quality codes. These are presentation
views of existing derived frames, not provisional report metrics. Nonperiodic
speech or noise moves the energy trace while leaving a gap in the pitch trace.

The microphone is still part of the generic ambient capture. Its outcomes
appear in the full session report, but voice is not part of the six-row
condition comparison.

## 4. First local finalization and reference acceptance

The interface explicitly shows that devices and processors are being stopped.
Only after disposal does it build the first ObservationV3 and its generic
report.

The condition panel then offers **Use this session as in-memory reference**.
This is an explicit page-local action, not automatic baseline selection. The
reference is not written to local storage, IndexedDB, a server, or an export.

## 5. Second live ambient session

After reference acceptance, **Capture follow-up session** returns to the
welcome screen. The participant context and asserted side remain in page memory
and the side control is locked, but consent must be checked again. Camera and
microphone permission/setup, calibration, capture, teardown, ObservationV3,
and generic report creation run again from live browser input.

This is a second session in the same page, not an imported or durable clinical
visit. Reloading, hiding the page, or choosing **New participant · discard all**
clears the reference and follow-up state.

## 6. Reports and condition evidence card

Each live capture has a generic report with 10 sections and 27 metric outcomes.
Each metric is a measured engineering value or `Not measurable`, with technical
evidence and provenance details.

After the second report, the condition panel shows exactly six rows in fixed
order: resting mouth difference, resting eye-aperture difference, subject-left
and subject-right lid closure completeness, spontaneous excursion difference,
and **Oculo-oral coupling difference**. The last label must not be presented as
a synkinesis finding.

Strict compatibility is evaluated per row. A measured row shows its reference
value, current value, and raw native-unit `current - reference` difference. A
withheld or incompatible row shows its reason codes and no delta. No row is
silently omitted, and excluded source traces cannot expose a measured value.
Repeatability and minimum detectable change are unknown.

The card starts as **Not reviewed**. **Accept research card** or **Dismiss
research card** changes only page-memory state; neither is authenticated
clinician approval, a durable audit record, or an export.

There is no narrative generation, clinical interpretation, persistence,
export, imported history, multi-visit baseline, or trend. No value or raw
difference indicates improvement, worsening, recovery, progression, severity,
diagnosis, or clinically meaningful health change.

## Browser and hardware boundary

The runnable target is current Chrome on macOS, opened from localhost or HTTPS,
with a working camera and microphone and permission to use them. The demo relies
on browser media capture, AudioWorklet, workers, `OffscreenCanvas`, and
WebGL/hardware acceleration. Behavior on other browsers, operating systems,
virtual cameras, unusual microphones, or restricted enterprise browser
configurations is not validated. A presentation mesh may be unavailable even
when face extraction continues.

## Honest failure demonstrations

- Deny one device and show that the other lane continues.
- Deny both devices and show that no report is created.
- End a short session and show specific reason-coded abstentions.
- Accept the first live session's ObservationV3 as the in-memory reference,
  complete a second live capture, and show all six
  measured/withheld/incompatible card rows.
- Accept or dismiss the card, then reset and show that both the review and
  reference are gone.
- Discard mid-session (the in-session consent-withdrawal path) and show that
  devices turn off without a report.
- Hide or reload the page and show that condition-demo memory is cleared.
- Make voiced sound, unvoiced sound, and background noise and show the live
  voice state and traces changing without creating a recording.
- Move within the camera frame and show the dense mesh tracking one face; add a
  second face and show the presentation clearing rather than choosing an
  identity.
