# Bounded candidate-coverage experiment

## Question and fixed comparison

The preceding augmentation study repaired 337.1/354.5 Hz ranking failures but
failed at 517.3/641.9 Hz because all retained candidates missed the fundamental.
Inspection of the four consumed signals finds a correct refined full-window
local maximum on all 448 fully post-change targets. Its integer-lag correlation
ranks tenth or twelfth, below long-period multiples. The correct maximum is the
shortest admitted peak in those windows. This diagnosis motivates one policy;
it is not fresh evaluation or evidence of general natural-speech benefit.

Compare exactly two acquisition policies using the same three frozen
`balance-aug` checkpoints from `harmonic-augmentation-v1`, seeds 20261022,
20261023 and 20261024:

- `original`: unchanged six-candidate v2 policy, including the highest-scoring
  full-40-ms refined autocorrelation maximum.
- **`revised`, predesignated:** replace only that full-window hypothesis with the
  shortest-lag already-admitted local maximum. Keep the existing >=0.35
  integer-lag correlation threshold, 50–700 Hz bounds, parabolic refinement,
  exact DC removal/8-kHz decimation, and actual integer-lag confidence score.
  If no peak is admitted, the source remains absent. No new threshold,
  frequency-specific exception, harmonic arithmetic or outcome-based fallback.

All original legacy hypotheses and both score-ranked suffix-20-ms hypotheses
remain. Merge exact-equal Hz by source-wise maximum, sort by Hz, and pad to six.
A candidate contributed only by the displaced full-window source can disappear;
its eviction is explicit. Source channel 3 still means observed full-window ACF.
The other source channels and eight non-count global fields are unchanged. The
raw valid-candidate fraction and later grouped fraction reflect the actual set.

The selector remains the same 3,009-parameter relative-context network, with its
original fitted natural-TRAIN scaling and weights. Complete-span 5-cent grouping,
original-member emission, 50-cent correctness tolerance, 40-ms support, 10-ms
hop and 20-ms measurement delay remain unchanged. Original MLP voicing at 0.5
continues to consume ORIGINAL v2 inputs; revised features never enter that head.
No training, scaling fit, checkpoint choice, alternate arm, second candidate
revision, calibration or browser integration is permitted in this experiment.
This tests a changed inference pipeline with frozen weights; it does not claim
that the selector is optimal for the revised candidate distribution.

## Data, source freeze and chronology

Historical sources/plans/checkpoints/predictions remain immutable. Pin the
completed augmentation manifest and its prediction freeze. Freeze this plan,
the full new/old computational source closure, the three checkpoint paths and
SHA256 identities, and the fresh specification before new extraction or fresh
PCM generation. Snapshot those exact source bytes in an ignored run directory.
All three checkpoints already predate this experiment; there are zero new fits.

Use the same 240 natural TRAIN sources from twelve speakers and eighty consumed
development sources from four disjoint speakers, under clean, noise10db,
first_difference and clip01. Reconstruct each transformed PCM from hash-verified
local microphone data with the original transformation and verify its exact
recorded SHA256. Do not read reserved/test speech or acquire external speech.

For every original cached input frame, recompute the full-window peak curve
through the new research-only TypeScript module. Require its original highest
peak and float32 source score to reproduce the original cached v2 evidence
exactly. Copy the other source/global values from the original input cache and
bind its hash. This validates the changed peak path against v2; it is not an
independent replay of every unchanged DSP feature. Save revised feature arrays,
original/new peak provenance, source evictions, exact grids and input hashes.
The kernel receives only finite PCM and acquisition endpoints, never references,
scenario F0, labels, model output or desired frequencies.

Keep all fifty existing analytic cases: fourteen original stationary/step/glide/
observation cases, twelve harmonic-context cases, twelve subharmonic-objective
cases, and all twelve augmentation cases including both phases at 517.3/641.9 Hz.
Reuse exact saved PCM; the three older observation-change waveforms may be
reconstructed only if their bytes match the frozen original PCM hashes.

Fresh evaluation consists of twelve prospectively fixed complete configurations:
frequencies `97.1,213.7,347.3,529.1,627.7,671.3` Hz crossed with phase triples
`(0.29,-0.67,1.31)` and `(1.89,0.47,-1.43)`, gain 0.055. Each signal has 96,000
samples at 48 kHz and changes harmonic coefficients from `(1,0.5,0.25)` to
`(0,1,0.5)` at sample 39,840. Harmonics 2 and 3 preserve the fundamental period.
There is no noise or normalization. Each retains 191 common-eligible targets,
including 112 fully post-change targets. The fixed carriers/configurations
avoid prior fixed evaluation carriers/endpoints and the augmentation TRAIN grid;
continuous historical glides and natural speech can cross those frequencies.
This is a failure-motivated noiseless family, not external generalization.

Generate fresh PCM only after the source/checkpoint/specification freeze. Save
all six selector pipelines' development and 62 analytic-case predictions and
original voiced probabilities, then freeze their hashes before this run reads
natural TRAIN/development references or computes analytic scores. Old control
predictions must reproduce saved augmentation outputs exactly on all overlapping
inputs. Source/manifests chronology is not an operating-system access audit and
does not restore independence to consumed development/regressions.

## Unchanged gates and complete reporting

Recompute the inherited prerequisites for the revised candidates: development
raw-candidate fixed-original-voicing oracle joint-error improvement >=1 percentage point over
original MLP, and actual raw-to-grouped lost voiced-target fraction <=0.5 points
on TRAIN and development, equally weighted across speaker/condition cells.
Do not substitute net coverage change for actual lost frames. Report raw and
grouped original-to-revised gains AND losses by cell and case/region. These
checks occur after prediction freeze; no training is being qualified or run.

Natural advancement still requires three-seed mean joint error improve by >=0.5
percentage points against BOTH original MLP and hybrid; each seed beats MLP; no
condition mean exceeds MLP error by more than 1 point; mean unconditional
reference-voiced success is not lower than MLP. Keep the same speaker/condition
weighting, reference eligibility, unavailable/unvoiced outcomes and denominators.

Every revised-policy seed must achieve >=95% success within 50 cents on EACH
original stationary whole case; EACH original 100/200/300 Hz fully post-change
case; EACH of the 36 consumed harmonic/subharmonic/augmentation post-change
cases; and EACH of the twelve new post-change cases. No case averaging may
conceal a failure. Retain all original transition outcomes and whole/pre/boundary/
post metrics, including failed controls and cases. All prerequisites, natural
and analytic gates must pass for a promising result; promotion is outside scope.

## Local budget and stopping rule

One scientific execution, two fixed policies, three frozen selectors, zero fits,
zero parameter/scaling updates. Use the existing local environment on Apple M4
Max/128 GiB RAM. One TypeScript peak worker and one Python process; CPU handles
signal extraction and frozen inference. No native builds, installation, remote
compute or additional environment. The GPU is not needed for this no-fit test.

Require >=8 GiB free disk before starting. Bound execution to 75 minutes, 2 GiB
new run artifacts and 16 GiB peak observed Python RSS; report the Node worker's
observed memory separately and stop it above 2 GiB RSS. These are process-memory
observations, not a claim about total machine memory. Per-request worker timeout
is at most 120 seconds, capped by remaining global time. The CLI also arms a
75-minute wall-clock alarm; the execution supervisor terminates its isolated
process group if that deadline is exceeded. Stop at the declared comparison, including a no-go. Preserve
failed attempts; a documented plumbing repair cannot retune the policy or drop
completed outcomes. Never overwrite an earlier run or evidence artifact.

Independently reconstruct scoring, coverage gains/losses, gates and checkpoint
inference. Verify source/input/checkpoint/prediction identities and control
reproduction. Publish code, aggregate metrics and audit receipts. Raw audio,
PCM, weights, features, reference arrays and natural per-frame outputs remain
local and ignored. Keep warnings and exclusions visible. Production stays DSP;
prior browser timings do not qualify this new offline candidate policy. Update
the high-level README and existing PR with `[skip ci]`, then verify zero hosted
workflow runs for the pushed revision.
