# Clinician Evidence Card

## Goal

Produce an inspectable artifact whose statements resolve to protocol-valid
measurements and evidence.

## Current implementation

`@phenometrix/evidence-core` now builds two deterministic artifacts:

- the generic report validates each ObservationV3 against the canonical
  protocol pack, resolves measurement/window/aggregate provenance, orders 27
  outcomes into 10 sections, and preserves the nonclinical boundary; and
- the unilateral facial movement card embeds the exact six-row
  previous-session comparison, joins only the profile's fixed display labels,
  derives measured/withheld/incompatible counts and minimum source quality, and
  preserves fixed boundary and source-disclosure text.

Every card value remains traceable to the embedded comparison and source
outcomes. A measured row displays reference, current, and raw native-unit
`current - reference`; a withheld or incompatible row has no delta and retains
only trace metadata without a measured metric value. The internal
synkinesis-oriented metric code is shown with the fixed safe label
**Oculo-oral coupling difference**, never as a finding that synkinesis was
detected.

The card begins pending and the browser can mark it accepted or dismissed for
the current page session. This is an unauthenticated research-demo control, not
clinician approval, signature, or durable review. There is no LLM narrative,
free-text conclusion, persistence, export, or EHR integration. The separate
`apps/clinician-review` directory remains documentation-only.

## Hard boundary

The Evidence Card capability cannot create measurements, expand a permitted
claim, diagnose, classify progression, infer cause, propose treatment, sign,
order, prescribe, message a patient, or execute an action. It cannot interpret
the sign or magnitude of a difference as improvement, worsening, recovery,
severity, or clinically meaningful health change.
