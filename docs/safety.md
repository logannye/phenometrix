# Safety foundations

## Demonstration boundary

PhenoMetrix is a nonclinical engineering prototype. It must not be used for
diagnosis, treatment, emergency detection, progression classification,
population screening, or decisions about a person. It must not process PHI.

## Required behavior

- Obtain explicit consent before requesting devices.
- Request camera and microphone independently.
- Show when devices are active and when they are off.
- Treat identity and speaker attribution as unverified.
- Treat the affected side as participant-asserted and unverified; never infer
  it from the measurements.
- Process native media locally and ephemerally.
- Emit only compact derived frames across worker boundaries.
- Keep live face and voice visualizations presentation-only, bounded, and
  coupled to device teardown.
- Require versioned protocol, algorithm, processor, track, and evidence
  provenance for every measurement.
- Prefer a specific `Not measurable` outcome over imputation or a generic
  quality result.
- Compare only one explicitly accepted reference ObservationV3 with one later
  live ObservationV3 after strict context and provenance compatibility checks.
- Keep exactly six condition rows visible as measured, withheld, or
  incompatible; calculate a raw `current - reference` delta only for a
  compatible pair of measured outcomes.
- Keep withheld and incompatible sources trace-only, without measured values or
  deltas.
- Stop all tracks and processors before showing a report.
- Clear derived frames and workflow events on discard, withdrawal, reset, or
  reload.

## Forbidden active-observation data

ObservationV3 and the report must never contain raw audio/video, device labels
or identifiers, PCM, waveforms, FFT bins, cepstra, MFCCs, formants,
spectrograms, transcripts, embeddings, voiceprints, native face landmarks,
blendshapes, transformation matrices, bitmaps, screenshots, mesh pixels, media
streams, or worker-owned canvases.

## Measurement boundary

The active protocol contains only deterministic engineering measurements. A
value may be emitted only after its metric-specific evidence and quality gates
pass with exact processor and track attribution. Otherwise the extractor emits
a registered withheld reason.

No metric has clinical validation. Interface copy must not convert a metric,
quality score, absence, or trajectory into a disease or treatment statement.
The internal `ambient.face.oculo_oral_synkinesis_index` code must be displayed
only with the profile's fixed **Oculo-oral coupling difference** label; it is
not evidence that synkinesis was detected.

## Comparison boundary

The unilateral facial movement profile is a within-page repeatability research
demo for a participant who asserts a previously established unilateral
peripheral facial palsy. It is not for new or sudden facial weakness. It
provides no diagnosis, severity grade, prognosis, cause, treatment advice,
emergency guidance, automatic affected-side inference, or equivalence to a
validated scale.

The comparison engine must match subject, profile, asserted side, protocol,
capture adapter, chronology, metric context/unit/algorithm, and processor/asset
provenance. Incompatibility is a terminal reason-coded result, not permission
to coerce units, choose another baseline, or silently drop a row. The only
allowed arithmetic is the native-unit raw difference `current - reference`.
Analytical repeatability and minimum detectable change are unknown, so neither
the sign nor magnitude may be described as improvement, worsening, recovery,
progression, or clinically meaningful health change.

## Retention boundary

The current application keeps derived measurements, report data, workflow
events, the accepted reference ObservationV3, the current ObservationV3,
comparison, evidence card, and accept/dismiss state only in memory for the
current page. It does not write local storage, session storage, IndexedDB, a
server, a retained clip, or an export file. Reference acceptance and card
review are therefore temporary research-demo actions, not a durable baseline,
clinical sign-off, or audit record. New-participant reset, visibility loss, or
reload clears the condition state.

Each of the two live captures independently requests and processes camera and
microphone input. Native media is disposed before that capture's ObservationV3
and report are shown. The microphone remains in the generic ambient pipeline,
but its outcomes are not among the six condition rows.

The live voice chart holds no PCM and at most eight seconds of derived display
points. The face mesh is drawn on a worker-owned canvas without returning
native landmarks. Both displays clear when capture stops and are excluded from
ObservationV3 and report contracts.

## Browser and hardware boundary

Current Chrome on macOS is the intended first live environment, but the
two-capture condition flow has not yet completed the named-hardware manual
acceptance checklist. Localhost or HTTPS, explicit media permission, functional
camera and microphone devices, AudioWorklet, workers, `OffscreenCanvas`, and
WebGL/hardware acceleration are environmental dependencies. Browser or device
failure must produce a lane failure, abstention, or missing presentation mesh;
it must never be reinterpreted as a participant finding. Automated browser
fixtures do not validate real-hardware measurement or repeatability.

## Optional WavLM service

The restored WavLM sidecar is a separate research-only service. It is disabled
by default, loopback-only, accepts no file path or URL, does not log request
bodies, and retains neither PCM nor returned summaries. The browser does not
call it, and its output is not an active metric or evidence artifact.

## Refused capability: acute stroke screening

Acute stroke screening must not be built on this system, and this is recorded
as a standing boundary rather than left as an absence—facial droop is one of
three FAST items, so the idea is structurally recurrent and will be proposed
again.

Two reasons, either sufficient:

1. The prototype publishes unvalidated brow geometry, but it does not measure
   forehead sparing with a reference-standard method and cannot distinguish
   central from peripheral facial weakness. Sensitivity adequate for an acute
   triage decision is a far higher bar than the trend-oriented engineering
   measurement this system is designed to explore.
2. Capture is ambient, unsupervised, and deliberately quality-gated to abstain.
   Those are the opposite of the properties an acute screening instrument
   requires, and abstention in an emergency context is itself a hazard.

A missed stroke is catastrophic and indefensible. No configuration, protocol
pack, or downstream consumer may present a facial asymmetry measurement as
stroke triage information.

## Deferred governance

Authentication, authorization, PHI handling, durable audit logs, retention
policy, incident response, clinical validation, authenticated/governed human
review, EHR integration, and regulated deployment controls must be designed
before any production or clinical use.
