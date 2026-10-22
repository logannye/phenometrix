# Bounded subharmonic-ranking objective experiment

Declared 2026-10-22 before any new fit or prospective waveform generation.
This follows the 337.1 Hz failure of the predesignated harmonic-context model.
Historical plans, sources, predictions and failed gates remain exact. Production
remains DSP. The user authorized one bounded experiment with unchanged gates and
all computation on this machine.

## Hypothesis and scope

The prior relative/acceptable-set seed 20261023 selects approximately 112.37 Hz
instead of 337.1 Hz on all 112 fully post-change targets in both phase cases.
Correct candidates and positive original voiced decisions are present throughout.
Other seeds succeed. This is a reproducible ranking error, not candidate absence
or evidence that the architecture can never represent a successful choice.

Test whether explicitly separating acceptable candidates from strong wrong
competitors during natural-only training improves this failure. A conditional
harmonic-alias penalty concentrates optimization on existing alias-bearing rows;
it supplies no missing waveform examples. It bundles negative selection and row
reweighting, so a gain must not be attributed solely to harmonic structure.
This study changes only the training objective. Candidate features, architecture,
parameter count, data, timing, grouping, voiced decisions and evaluation remain
fixed. No special rule names 337.1 Hz, a preferred octave or a preferred direction.

## Three fixed objectives, nine fits

Reuse the exact `HarmonicSelector('relative','acceptable-set')` architecture:
3,009 parameters, original grouped candidate representation and train-only
scaling. Every arm starts with identical same-seed tensors. All inference uses
the unchanged candidate logits and original-member frequency emission. There
is no margin term or reference input at inference.

Run arms in this order within each seed:

1. **set:** unchanged acceptable-set cross entropy.
2. **all-margin:** set loss plus the fixed auxiliary margin against all valid
   nonacceptable candidates.
3. **alias-margin (predesignated):** set loss plus the same margin against
   nonacceptable harmonic-alias candidates only.

An acceptable candidate is exactly the existing emitted frequency within 50
cents of a valid voiced TRAIN reference. For each wrong candidate, define alias
membership if it is within 50 cents of any acceptable emitted candidate times
one of `{2,3,4,1/2,1/3,1/4}`. Use CPU float64 frequencies for this TRAIN-only mask.
Both upward and downward errors are penalized; acceptable candidates never become
negatives. No new acceptable label is introduced and no reference is inferred
from an alias. Uncovered rows retain zero pitch loss.

For each covered row with at least one negative admitted by its arm:

`hinge = max(0, 1 + max_negative_logit - max_acceptable_logit)`.

The auxiliary is **0.5 times the mean hinge over those rows in the minibatch**.
Rows without an admitted negative contribute to the unchanged base loss only;
an empty auxiliary subset returns finite, connected zero. Use `torch.amax` to
share gradients at exact maxima ties. All-padding/uncovered data must yield zero
auxiliary gradient and padded logits must never enter the maxima. The margin of
1.0 and weight of 0.5 are fixed, not tuned. The alias subset and its minibatch
normalization deliberately reweight training. Report eligible row/candidate
counts and each epoch's base and weighted-auxiliary contributions, summarized as
covered-row-weighted averages of minibatch objectives rather than a global
eligible-negative-row mean; do not claim
that all-margin versus alias-margin isolates only harmonic structure.

Use seeds 20261022, 20261023, 20261024, 12 epochs, Adam 0.001, batch size 4096, gradient
norm clipping 1.0 and the existing seed-plus-epoch permutation. Retain all 132,181
originally covered scheduled rows, including 107 grouped-uncovered rows; require
396 optimizer updates with zero skipped batches per fit. No early stopping,
checkpoint selection, refit, extra seed or outcome-based arm substitution.
The three prior relative/acceptable checkpoints remain frozen comparators.

## Data, measurement and chronology

Use the same 240 natural training sources / 12 speakers and 80 development sources / 
four disjoint speakers under clean, noise10db, first_difference and clip01.
No synthetic training, augmentation, new acquisition, reserved/test content or
external speech. Development is consumed exploratory evidence.

Retain original six candidates, 5-cent complete-span grouping, original-member
representatives, source evidence, scaling, 40 ms support, 10 ms hop, 20 ms measurement
delay, 50-cent tolerance and common reference mask. Retain original MLP voiced
threshold 0.5 exactly. Count invalid/boundary exclusions and include valid unvoiced,
unavailable and uncovered outcomes in the original denominators.

Verify the previously passing oracle/grouping prerequisites by exact artifact
and source hashes. Their original headroom >=1 point and grouping loss <=0.5 point
train/development requirements remain in force. This study does not alter either
candidate representation or its inherited coverage.

Freeze all nine final checkpoints before generating fresh challenge PCM/features
or making any new development predictions. Freeze ALL natural and analytic
predictions before reading development references in the fitting/scoring process.
Reference chronology is an implementation/manifests claim, not an operating-system
access log; it does not restore independence to consumed data. Save original voiced
probabilities alongside analytic predictions for independent checks.

## All existing gates retained; new prospective challenge added

Natural advancement remains: three-seed mean joint error, equally averaging
speaker/condition cells, improves by >=0.5 percentage points against BOTH original
MLP and hybrid; each seed improves over original MLP; no condition mean worsens
by >1 point versus MLP; mean unconditional reference-voiced success does not decrease.

Every designated seed must achieve >=95% success within 50 cents separately on:

- Each original stationary 100/200/300 Hz whole case.
- Each original 100/200/300 Hz fully post-change region.
- Each of the previous 12 harmonic-context challenge post-change regions,
  including both 337.1 Hz failures. These are now consumed regressions.
- Each of the new 12 prospective challenge post-change regions below.

Report all 14 original analytic cases and all old/new challenge whole cases,
pre-change and boundary regions, candidate coverage, voiced/unavailable outcomes.
No averaging across seeds/cases, frame dropping, frequency exception, metric
substitution or threshold adjustment can conceal a failure. Call the designated
candidate promising only if ALL natural and analytic gates pass.

New challenge frequencies: 79.7,151.3,319.7,354.5,503.9,659.3 Hz, crossed with phase
vectors (0.19,-0.83,0.67) and(1.43,0.59,-1.07) radians. All complete configurations
differ from the consumed 12; two frequencies bracket the known 337.1 Hz failure.
Keep 48 kHz, 2.0 seconds, gain 0.06, change_sample 39840 (0.83 seconds), and harmonic
coefficients (1,0.5,0.25)->(0,1,0.5), with continuous phase and constant latent F0.
No noise or normalization. Both remaining harmonics 2 and 3 preserve the fundamental
period; no single-harmonic or even-only ambiguous target is introduced. Each case
has 191 common-eligible targets and 112 fully post-change targets. Generate only
after the exact nine-checkpoint/source/specification freeze. This failure-motivated
synthetic check is prospective within one generator family, not external speech
or broad channel robustness. Metadata/rejection tests must not synthesize these
waveforms before the freeze.

## Local budget, provenance and stopping

One local fitting process, existing Python/Torch MPS environment; CPU inference,
preprocessing, scoring and audits. At most **nine new fits**, 30 minutes total fitting,
8 GiB observed MPS allocation,2 GiB new experiment outputs, and8 GiB free disk before
execution. No installation, Rust/native build, hosted computation or new environment.
Bind source/plan/input/checkpoint/prediction identities and keep all attempts.

The historical JSON serialization failure and its separate recovery remain
unchanged. The new runner explicitly serializes NumPy scalars to equivalent
Python scalars and tests boolean gate output before fitting. Plumbing failures
never authorize scientific retuning or discarding completed fits. Refuse to
overwrite any run directory. Stop after the declared comparison, including a
no-go when any gate fails. Publish code, aggregate results, diagnosis and audits;
raw speech, PCM, references, features, weights and natural frame outputs stay
ignored and local. Production promotion, calibration and browser qualification
are outside this experiment.
