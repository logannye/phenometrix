# Capture web application

This is the static ambient-v3 browser application and the runnable
two-live-capture unilateral facial movement research demo.

## Runtime responsibilities

- record explicit session-local consent;
- request microphone and camera independently;
- verify the committed local asset manifest;
- run bounded voice and face calibration;
- maintain independent worker-backed capture lanes;
- draw a presentation-only 478-point face mesh and a bounded live voice signal
  dashboard during capture;
- collect compact derived frames for at most five minutes;
- finalize all 27 deterministic metrics into ObservationV3;
- validate and render the session-only structured report;
- preserve an explicitly accepted first ObservationV3 as the one
  page-memory reference;
- repeat the complete live consent, permission, calibration, capture,
  finalization, and media-disposal lifecycle for a second ObservationV3;
- run the strict six-metric face comparison and render its deterministic
  evidence card;
- record an optional accept/dismiss action only in page memory; and
- dispose devices, workers, timers, buffers, and journal state.

The affected side is selected by the participant and is not inferred or
verified. The microphone remains active as an independent generic ambient lane
in both captures, so each full report can still contain voice outcomes; the
condition card allowlists exactly six face metrics and does not compare voice.

The application has no server route, API key, LLM, persistence layer, export,
guided task mode, or synthetic production capture mode. The accepted
reference, latest observation, comparison, card, and card review exist only in
JavaScript memory for the current page. **New participant · discard all**,
visibility loss, and reload clear them.

## Condition-demo sequence

1. Select the previously understood subject-left or subject-right affected
   side, consent, and complete a live session.
2. Inspect the generic 27-outcome report and explicitly choose **Use this
   session as in-memory reference**.
3. Choose **Capture follow-up session**, consent again, and complete a second
   independent live setup and capture.
4. Inspect the fixed six-row condition card. A row is `measured`, `withheld`,
   or `incompatible`; only measured rows show a raw native-unit
   `current - reference` difference.
5. Optionally accept or dismiss the research card for this page session. That
   control is not clinician sign-off and is not saved.

The condition card assigns no direction of health change. Repeatability and
minimum detectable change are unknown, and the safe UI label **Oculo-oral
coupling difference** must not be restated as a synkinesis finding.

## Worker boundaries

The voice worker receives PCM from the AudioWorklet and emits compact signal
frames. The face worker owns native MediaPipe results and emits compact
kinematics. It also draws the dense mesh directly to a transferred
`OffscreenCanvas`; landmark coordinates never return to the application. The
voice dashboard retains at most eight seconds/800 derived level and pitch
points and is cleared with the capture lifecycle. Neither live visualization
is application, ObservationV3, or report evidence.

## Commands

From the repository root:

```bash
pnpm dev
pnpm --filter @phenometrix/capture-web verify:assets
pnpm --filter @phenometrix/capture-web test:unit
pnpm --filter @phenometrix/capture-web typecheck
pnpm --filter @phenometrix/capture-web test:browser
pnpm --filter @phenometrix/capture-web build
```

The Playwright suite injects media and worker mocks with `page.addInitScript`.
Those fixtures are not compiled into the production application.

In development, open `http://127.0.0.1:4173/` in current Chrome on macOS. The
browser UI owns consent, capture start, end/discard, reference acceptance,
page-local card review, and reset. Camera and microphone access requires
localhost or HTTPS and real device permission. Other browsers, operating
systems, camera/microphone combinations, and configurations without working
AudioWorklet, workers, `OffscreenCanvas`, WebGL, or hardware acceleration have
not been accepted as supported hardware targets. Stop the Vite process with
`Ctrl-C` when development is complete.
