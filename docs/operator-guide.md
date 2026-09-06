# PhenoMetrix operator guide

## Environment

- macOS with current Chrome, hardware acceleration, and a working camera and
  microphone;
- Node.js 22.12+ on the Node 22 line, or Node 24+;
- pnpm 9.12.3; and
- localhost or HTTPS for camera and microphone access.

The browser application has no runtime environment variables and no server
credential. Do not add an API key or PHI to `.env` files.

## Install and run

```bash
pnpm install --frozen-lockfile
pnpm dev
```

Open `http://127.0.0.1:4173` in Chrome.

## Rehearsal

1. Confirm the welcome page says **Nonclinical research prototype**, **Not for
   new or sudden facial weakness**, and that raw media and results are not
   uploaded or saved.
2. Select the participant-asserted subject-left or subject-right affected side.
   Confirm the application says that it does not infer or verify that side.
3. Read the consent statement, check the box, and begin technical setup.
4. Allow the microphone and camera. Stay quiet briefly, face the camera, and
   wait for their independent lane states.
5. Confirm the dense facial mesh follows one face. Speak, make an unvoiced
   sound, and pause; confirm energy responds continuously while pitch appears
   only for periodic sound.
6. Continue a normal conversation; do not perform scripted exercises. Treat
   the live voice panel as a signal preview, not report output.
7. End the first session. Verify that both devices turn off and the mesh and
   voice history clear before the 10-section, 27-outcome report appears.
8. Inspect one measured and one not-measurable generic metric, including its
   exact reason and evidence references.
9. Choose **Use this session as in-memory reference**, then **Capture follow-up
   session**. Confirm the asserted-side control remains selected and locked,
   while consent must be checked again.
10. Complete a second live setup and ambient session. The microphone remains
    part of this generic capture even though the condition card uses face
    metrics only.
11. End the second session and verify teardown again precedes its generic
    report and the condition comparison.
12. Confirm the condition card has exactly six rows. Each must be measured,
    withheld, or incompatible. A measured row shows reference, current, and raw
    `current - reference` in the native unit; an excluded row has a reason and
    no delta.
13. Confirm the sixth row is labeled **Oculo-oral coupling difference**, not a
    statement that synkinesis was detected. Confirm the page states that
    repeatability and minimum detectable change are unknown and assigns no
    direction of health change.
14. Accept or dismiss the research card and confirm the state says it applies
    only to this page session.
15. Choose **New participant · discard all** and verify the affected side,
    reference, current report, comparison, and review state are gone.

Also rehearse camera denial, microphone denial, both-device denial, and
discard, which is also the in-session consent-withdrawal path. A face-denied
session may still produce a generic voice report but cannot yield measured
condition rows. Reload or hide the page after accepting a reference and confirm
that page-memory condition state is cleared.

## Automated smoke

```bash
pnpm demo:smoke
```

The Playwright fixture replaces browser APIs only inside the test page. It
covers the two-capture condition flow and page-memory clearing, but it does not
establish real-camera or real-microphone repeatability. The production bundle
has no synthetic-capture query mode.

## Optional WavLM research service

The sidecar is not required for the demo and the browser does not call it. To
test it independently:

```bash
uv sync --project services/voice-inference --extra dev --locked
uv run --project services/voice-inference --extra dev python -m pytest services/voice-inference/tests
```

Follow `services/voice-inference/README.md` only for isolated research use. It
must remain disabled by default and bound to loopback.

## Common failures

- **Asset integrity failure:** run `pnpm --filter @phenometrix/capture-web
  verify:assets`; rebuild the manifest only after intentionally reviewing an
  asset change.
- **Permission denied:** reset the site permission in Chrome and start a new
  session.
- **Follow-up state disappeared:** visibility loss, reload, and **New
  participant · discard all** intentionally clear the in-memory reference and
  review. Complete both captures without leaving or hiding the page.
- **Lane remains not measurable:** use quieter audio, even front lighting, one
  face, and keep the tab visible. Never force a measurement.
- **Mesh display unavailable:** use current Chrome with hardware acceleration
  enabled. Face extraction may continue even when the presentation canvas is
  unavailable.
- **Energy moves but pitch is blank:** pitch intentionally appears only for a
  sufficiently periodic signal; noise and unvoiced sound create pitch gaps.
- **No report:** both devices were unavailable, the session was discarded, or
  final provenance validation failed. Inspect the browser console without
  weakening the gate.
- **No measured condition delta:** inspect the row status and reason codes.
  Withheld and incompatible rows deliberately have no delta; never substitute
  a value or relax compatibility.

The intended hardware target is current Chrome on macOS; the named-hardware
acceptance checklist is still pending. Secure-context permission policy, device drivers,
AudioWorklet, workers, `OffscreenCanvas`, WebGL, and acceleration behavior vary
elsewhere; a successful automated fixture is not evidence that another browser
or device is supported.
