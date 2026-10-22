# Follow-up protocol: clock qualification and pitch-alias diagnosis

Declared 22 October 2026 against PR #33 source commit
`da5dfe752062bfce2a0b3d5a950da217e118690b`, after the initial results and before
new diagnostic outcomes. The user authorized both follow-ups. Earlier failures,
raw artifacts, selection decisions and the unopened sealed test remain intact.
This is a new exploratory development round, not retrospective qualification of
the original checkpoints.

## Clock investigation

The old 30-minute replay showed increasing mapped pre-receipt age while handler
service and receipt-to-main delivery stayed stable. One audio/performance anchor
assumed equal clock rates. The first task is to measure the clocks independently
of PCM/result arrival, not to subtract an empirically convenient latency trend.

Record bounded, timestamped observations of the render clock, bracketed main
performance reads, and (when available) separately labeled output-device
`getOutputTimestamp()` pairs. The latter describes output-device position and is
not a measurement of microphone capture availability. Retain the original fixed
anchor projection for comparison. Any revised mapping must preserve observation
identity, raw evidence, unknown hardware precision, unavailable durations,
discontinuities, and genuine injected worker/main stalls. It must not use a
result's arrival to make that same result appear timely.

Run a short functional check first, then an isolated local diagnostic sufficient
to compare clock-rate change with the historical trend. Freeze any implementation
and its source identities before a new 30-minute paced confirmation. No model
fitting or other agent benchmark runs concurrently with a timed confirmation.
Ordinary desktop load, hardware input latency and thermal state remain outside
the controlled experiment. Record these limits explicitly.

A timing fix is accepted only when deterministic tests preserve stall detection,
missing/invalid clock evidence and reset behavior, and actual-browser evidence
shows bounded credits, complete stop acknowledgments, no lost telemetry, and
known-tone recovery. Complete-block p99 must remain at most 10 ms and no processed
block may exceed its 20 ms cadence in the confirmation run. Report every timing
exclusion and every five-minute window, including residual trend; do not claim a
physical latency bound from these software observations. Any additional numerical
clock criterion will be declared with the chosen observable contract before its
confirmation, rather than selected from the final run.

## Pitch investigation

The original stationary 300 Hz case remains a failure for all three learned
checkpoints despite correct candidate coverage. Inspect exact candidates,
feature scaling, training labels, logits, voicing, and frozen inference first.
Use a frozen frequency/profile/phase sweep and explicitly labeled feature or
slot counterfactuals to distinguish numerical rank instability, learned index
priors, observation ambiguity, and missing candidate support. A counterfactual
feature vector is diagnostic evidence, not a physically generated recording.

No new fitting is authorized by this protocol's first diagnostic phase. Before
an intervention is evaluated, freeze its source, semantic contract and criteria.
Prefer a small deterministic change or an interpretable decomposition over a
larger model. Any new fit must use the existing local environment and GPU, with
bounded epochs, seed count and disk outputs recorded in an additional plan.

The 14 original scenarios and the previous 32-case challenge are now development
regressions. A subsequent claim of generalization requires a new generator plan
frozen before the intervention's outcomes, disjoint parameters, and explicit
identifiability limits. Natural development remains the same four speakers and
four conditions; transformed recordings do not increase subject count. Preserve
per-speaker and per-condition scores, voicing and pitch components, coverage,
transition behavior, and all failed cases. Do not acquire or score sealed-test
audio, tune calibration on development, or change the production estimator as a
consequence of average improvement alone.

All computation, tests and analysis stay on the local Apple M4 Max machine.
Public changes include source, aggregate evidence and reproducible commands.
Audio, full trajectories, weights and raw timing rows remain ignored locally.
