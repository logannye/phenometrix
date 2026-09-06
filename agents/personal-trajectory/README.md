# Personal Trajectory

## Goal

Compare six allowlisted current face measurements only with one explicitly
accepted, compatible reference observation for the same page-memory
participant context.

## Current status

The current milestone implements a deliberately narrow ObservationV3-native
slice in `@phenometrix/trajectory-core`:

- the first live browser observation must be explicitly accepted as the sole
  in-memory reference;
- a second live observation is the current source; imported or synthetic
  production observations are not accepted by the UI;
- subject reference, condition-profile ID/version/digest, participant-asserted
  side, protocol ID/version/digest, capture-adapter ID/version, chronology,
  metric context/modality/unit/algorithm, and processor runtime/asset/integrity
  provenance are matched fail-closed;
- each of the profile's six metrics terminates as `measured`, `withheld`, or
  `incompatible`, with stable reason codes and no silently omitted row; and
- only two compatible measured sources yield the native-unit raw difference
  `current - reference`.

Withheld and incompatible rows have no delta. Their available source objects
are trace-only and cannot contain measured numeric values. Analytical
repeatability and minimum detectable change are explicitly `unknown`.

The accepted reference, context, current observation, and comparison exist
only in browser page memory. New-participant reset, visibility loss, or reload
clears them. There is no stable participant identity, storage, importer,
reference selection across history, more-than-two-session model, baseline,
dispersion, trend, or uncertainty estimate.

The superseded v2 contracts, implementation, tests, and synthetic fixture were
removed on 2026-07-24. The current package reuses the name but not the v2
unordered-prior design.

## Hard boundary

Personal Trajectory does not use population norms, diagnose progression, infer
cause, recommend treatment, infer the affected side, or assign clinical
significance. The sign or magnitude of a raw delta must not be presented as
improvement, worsening, recovery, progression, severity, or clinically
meaningful health change. The current research profile has no validated claim
and cannot permit that interpretation.
