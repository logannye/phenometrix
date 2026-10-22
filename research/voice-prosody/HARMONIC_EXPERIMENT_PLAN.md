# Bounded harmonic-context experiment

Declared 2026-10-22 before any new fitting or generation of the prospective
challenge. This follows the user's request to address the harmonic-balance
failure without relaxing gates. Historical source, checkpoints, gates and
failures remain immutable. Production stays DSP.

## Evidence and hypothesis

The reused observation-change-300 fixture has a known seed-specific failure:
shared/acceptable-set seed 20261023 misses all 95 fully post-change targets,
whereas the other two acceptable-set and all three shared/nearest seeds pass.
The correct near-300 Hz candidate remains available, with suffix-correlation
evidence, while the near-100 Hz threefold-period alternative has strong AMDF and
full-window correlation. The failing seed gives that alternative a slightly
higher logit. This is a ranking failure, not an absent candidate or a voiced
rejection. Passing seeds prevent a claim that the existing pooling is inherently
incapable. Acceptable-set loss never labels this subharmonic acceptable.

Test whether explicitly relating another candidate's frequency to the queried
candidate helps the shared scorer compare competing estimator evidence. This
changes the inductive bias using existing acquisition evidence. It supplies no
new waveform feature, frequency candidate, target adjustment or temporal state.

## Single fixed comparison

Four arms cross absolute versus relative candidate context with nearest versus
acceptable-set objectives. Each new model has 3,009 learned parameters. Keep the
existing grouped inputs, train-only scaling, tanh candidate encoder Linear(8,32),
and shared tanh Linear(73,32) then Linear(32,1) scoring head. Replace the previous
broadcast context with a separate shared message encoder Linear(9,32), tanh,
then elementwise maximum over all valid neighbor candidates, including self.

For query i and neighbor j, the message input consists of the eight standardized
fields of candidate j and one relative-frequency scalar. The relative arm uses
`(raw_log2_hz_j - raw_log2_hz_i) / log2(700/50)`; the absolute control sets this
scalar to zero. All-empty context is zero. Both arms use identical parameter
shapes, initialization and optimization schedules. The 32 weights on the ratio
column receive zero gradients in the absolute control, so equal effective
functional capacity is not claimed. There is no slot identity,
hard preferred harmonic, source deletion or added raw evidence. The comparison
isolates access to relative-frequency input within this added context encoder;
comparison with the older 2,689-parameter shared model also changes capacity.

Use the existing losses unchanged. The predesignated candidate is **relative
context with acceptable-set loss**. Nearest-target arms are diagnostic controls;
no winning arm, seed or epoch may replace the designated candidate after scoring.
The six older shared checkpoints are frozen comparators, never refitted.

## Immutable data and learning schedule

Use only the same 240 natural training sources/12 speakers and 80 development
sources/four disjoint speakers under clean, noise10db, first_difference and clip01.
No synthetic training, augmentation, source dropout, test/reserved access or
new reference acquisition. Natural development is consumed exploratory evidence.
Reuse and verify the passing step-1 oracle/grouping prerequisites by exact hashes;
the unchanged grouping must retain those coverage bounds. A prior passing oracle
is a reference-assisted ceiling, not evidence that the intervention will learn it.

Keep the six original candidates, 5-cent grouping, original-member representative,
40 ms support, 10 ms hop, target at support end minus 20 ms, 50-cent tolerance,
and common eligible reference mask. Frozen original MLP voicing at threshold 0.5
is identical for every arm. Invalid/boundary references are excluded and counted;
all valid eligible unvoiced and uncovered frames retain their evaluation role.

Run seeds 20261022, 20261023, 20261024, in that order; within each seed run
absolute/nearest, absolute/acceptable-set, relative/nearest, relative/acceptable-set.
Use PyTorch default initialization immediately after setting the seed, 12 epochs,
Adam 0.001, batch 4096, gradient norm clipping 1.0, and the previous seed-plus-epoch
NumPy permutation. All arms use the same 132,181 originally covered training rows;
107 grouped-uncovered rows remain and contribute zero pitch loss. Scaling uses
input-only training frames. No early stopping, checkpoint selection or retuning.
Freeze all twelve final checkpoints before generating the new challenge or making
new development predictions. Freeze all predictions before opening development
reference bytes in the fitting/scoring process. This cannot restore independent
holdout status to already inspected development data.

## Gates, unchanged and strengthened

The original natural advancement gate remains exactly:

- Three-seed mean joint error improves by at least **0.5 percentage points against
  both original MLP and original hybrid**, using equal speaker/condition cells.
- Every candidate seed improves over the original MLP.
- No condition's mean joint error worsens by more than 1 percentage point versus MLP.
- Mean unconditional reference-voiced success does not decrease versus MLP.

Retain the original >=95% success requirement on every original stationary case
for every candidate seed. Add >=95% success on **each fully post-change region**
of observation-change-100, -200 and -300 for **every candidate seed**. No averaging
across seeds or frequencies may hide the known failure. Keep all 14 reused cases,
including transition boundaries, in their whole-case reports.

The prospective challenge has exactly twelve configurations: frequencies
87.3, 137.9, 223.7, 337.1, 467.3 and 613.7 Hz, crossed with harmonic phase vectors
(0.37,-0.41,1.13) and (1.19,0.23,-0.73) radians. At 48 kHz, duration 2.0 seconds,
gain 0.06 and change time 0.83 seconds, the first three harmonic coefficients
change from (1,0.5,0.25) to (0,1,0.5). Phases and latent F0 remain continuous;
there is no noise, normalization or synthesized dropout. These full configurations
are disjoint from the earlier evaluated waveforms. Generate their PCM/features
only after checkpoint freeze. References cannot enter the scorer forward pass.

Require >=95% success on each challenge's fully post-change region for every
candidate seed as an additional prospective synthetic component. Report whole-case,
pre-change and boundary results, candidate coverage, and unvoiced/unavailable
failures without exclusions or threshold changes. This limited noiseless generator
family is not independent natural speech or broad channel robustness.

Call the designated candidate promising only if **all** natural, stationary,
old post-change and prospective challenge gates pass. Report paired results
against the equal-parameter absolute control and all older shared models regardless
of decision. A repaired old fixture alone is not advancement.

## Local budget, provenance and stopping

One fitting process on this Mac's GPU through the existing PyTorch MPS environment;
CPU for preprocessing, inference and independent scoring. Maximum twelve new
fits, 30 minutes total fitting, 8 GiB observed MPS allocation and 2 GiB new raw
artifacts. Require 8 GiB free disk before execution. No installations, Rust/native
compilation, hosted compute or browser integration. Tests may inspect challenge
specification and rejection paths, but must not generate its actual waveforms
before the final checkpoint freeze.

Freeze source/plan/input/checkpoint/prediction identities, retain all fitted arms
and any failed attempt. Numerical or plumbing errors do not authorize scientific
retuning or silently discarding completed fits. Never overwrite a run directory.
Stop after this comparison, including a no-go if any gate fails. No extra seed,
loss weight, architecture, threshold or fit may be tried in response to results.
Publish code, aggregate evidence and the full outcome; raw speech, features,
reference arrays, per-frame predictions and weights remain ignored and local.
