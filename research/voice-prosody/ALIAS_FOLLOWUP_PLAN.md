# Frozen subharmonic diagnosis, version 1

This is a new, explicitly authorized diagnostic after the original bounded trial
ended in no-go. The old gates and results remain failed. No fitting, production
promotion, candidate revision, threshold selection, natural-data hybrid scoring,
or sealed-test access is part of this diagnostic.

Before running, freeze the three existing checkpoint identities, input feature
and generator sources, and the following sweep. Use the existing local environment,
CPU inference, and the unchanged v2 TypeScript acquisition features. This is not a
runtime benchmark. The fixed sweep has 28 frequencies from 75 through 600 Hz,
including 299.99, 299.999, 300, 300.001 and 300.01 Hz; three harmonic profiles
`[1,0,0]`, `[1,.5,.25]`, `[0,1,.5]`; initial fundamental phases 0 and 0.37 radians;
gain .075; and 0.4 seconds at 48 kHz. There are 168 independent resets and 67.2
seconds of noiseless generated input. Harmonics 2 and 3 preserve an identifiable
fundamental when harmonic 1 is absent. Exact frequencies are in the run plan.

Predict every acquired frame before using known frequency in a score. Report
all scenarios and original/augmented MLP and original GRU results, candidate
oracle coverage, voicing, octave errors, candidate counts and selected ranks.
Use target time e−20 ms, availability e, and the existing common support mask.
Include preserved DSP and fixed YIN as diagnostic context. All three frozen
models receive identical raw float32 features and boolean masks, with their own
embedded training scaling and zero state at every recording boundary.

On the already-scored constant-300 regression, repeat inference under declared
counterfactuals: remove each one candidate and repack the sorted slots; mask that
candidate without changing input features; swap adjacent slot blocks while
preserving their frequencies/masks; neutralize each input coordinate to its own
training mean; neutralize all global features; clamp standardized coordinates
at ±3; and reset recurrent state at each frame. These are deliberately artificial
interventions to localize dependence, not executable estimators or valid accuracy
improvements. In particular, slot permutation violates the sorted-slot contract.
All interventions are reported, including failures. No intervention is selected
for deployment from this sweep.

Check exact Torch versus independent NumPy inference on the unchanged original
regression, the single-nearest-candidate loss versus the set of candidates within
50 cents, padding, candidate-source/rank patterns, and training-distribution
coverage. Prediction parity rules out an implementation discrepancy, not a bad
learned decision. Training-label ambiguity and feature sensitivity require direct
evidence before they are described as causes.

A follow-up intervention and prospective decision gates must be declared separately
before any new fitting or natural-data scoring. The existing development set and
original synthetic challenge are now diagnostic/regression sets; neither can be
presented as an untouched evaluation after this investigation. Preserve the sealed
test and require independent challenge definitions for any subsequent qualification.
