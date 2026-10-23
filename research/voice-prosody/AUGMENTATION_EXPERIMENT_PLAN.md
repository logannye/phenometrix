# Bounded harmonic-balance training augmentation

Declared 2026-10-22 before new waveform generation or fitting. This study targets
consumed 337.1 and 354.5 Hz subharmonic ranking failures. The user authorizes all
computation locally and publication of aggregate evidence, with unchanged gates.
Historical sources, plans, checkpoints, failed decisions and recovery records
remain immutable. Production remains versioned DSP.

## Hypothesis and fixed comparison

Correct candidates and positive frozen voicing already exist in the failing
post-change frames. Natural TRAIN has only 517 grouped-covered frames at F0
>=300 Hz. Test whether physically generated examples with a missing fundamental
teach the existing selector to rank the retained fundamental-period candidate.
This is a training-support hypothesis, not a proven diagnosis or clinical claim.

Nine fits: three fixed seeds 20261022, 20261023 and 20261024, with these arms in
order within each seed:

1. `natural-only`: unchanged acceptable-set cross entropy on natural TRAIN.
2. `stationary-aug`: natural loss plus 0.25 times synthetic stationary loss.
3. **`balance-aug`, predesignated**: natural loss plus 0.25 times synthetic
   harmonic-balance loss.

All use exactly `HarmonicSelector('relative','acceptable-set')`, 3,009 parameters,
identical same-seed initial tensors and natural-TRAIN-only scaling. Keep the
six candidates, complete-span 5-cent grouping, original-member frequency emission,
50-cent acceptable set, original MLP voicing threshold 0.5, 40 ms causal support,
10 ms hop and 20 ms measurement delay. No new acquisition feature, inference-time
rule, objective margin, calibration or temporal model.

Retain all 132,181 original natural scheduled rows, including 107 grouped-uncovered
zero-loss rows. Each arm uses the same seed-plus-epoch natural permutation,
12 epochs, batch size 4096, Adam 0.001 and gradient clipping at norm 1.0. Require
396 optimizer updates and zero skipped natural minibatches. No early stopping,
checkpoint choice, extra epoch, fit, seed, or observed-arm substitution.

Each augmented optimizer step adds a synthetic minibatch of 512 rows. The loss
is `mean_set_CE(natural covered rows) + 0.25 * mean_set_CE(synthetic covered rows)`.
An empty covered synthetic subset contributes connected zero. Uncovered synthetic
rows stay in the fixed sample schedule and are counted; never resample to replace
them. There are no additional optimizer steps. The additive term changes gradient
magnitude/clipping and adds examples/compute; this is not an equal-compute claim.
The same-gain stationary control also differs in energy after the balance change,
so the contrast tests a physical augmentation package, not harmonic content
isolated from amplitude or coverage-dependent loss normalization.

## Paired synthetic TRAIN data and exposure

Generate 96 configurations per synthetic arm: these 24 frequencies in Hz,
crossed with two phase vectors and two gains:

`70.25,90.25,110.25,130.25,150.25,170.25,190.25,210.25,230.25,250.25,270.25,290.25,
310.25,320.25,330.25,340.25,350.25,360.25,370.25,390.25,430.25,490.25,570.25,650.25`.

TRAIN phases are `(0.11,0.73,-0.29)` and `(-0.61,1.07,0.41)` radians; gains are
0.04 and 0.08. Both arms use 48 kHz, 96,000 samples, constant latent F0,
continuous phase, before coefficients `(1,0.5,0.25)` and change sample 39,840.
Stationary coefficients remain unchanged; balance coefficients become `(0,1,0.5)`.
Nonzero harmonics 2 and 3 preserve shortest period 1/F0 after the fundamental
vanishes. No noise, normalization, single-harmonic or even-only ambiguity.

Pairs share frequency, phase, gain and time grid; only the after coefficients
change. The exact 337.1 and 354.5 Hz cases are excluded from training. Fixed TRAIN
carriers/full configurations differ from old and fresh evaluation; consumed
continuous glides can cross these frequencies, so support-disjointness is not
claimed. This targeted synthetic family is not natural-speech diversity.

Use every common-eligible row from each sequence, 191 per case and 18,336 per arm.
For seed/epoch, independently permute these rows with NumPy RNG seed
`seed + 1_000_000 + epoch`, take the first 16,896 rows and reshape to 33 x 512.
This schedule samples without replacement within an epoch and is identical in
both augmented arms. Persist all schedules and targets, including reference F0,
scenario index, reference/availability samples and region. Record per-epoch
selected, covered, uncovered-zero, case and region exposure, plus natural and
weighted synthetic minibatch loss contributions. Coverage differences can change
effective loss weighting and must remain visible.

## Natural data and chronology

Keep 240 natural TRAIN sources from twelve speakers and eighty development
sources from four disjoint speakers, each under clean, noise10db,
first_difference and clip01. Development is consumed exploratory evidence.
No new natural acquisition, reserved/test content or external speech.

Require exact source/artifact matches for the existing passing oracle/grouping
prerequisites: >=1 percentage point oracle headroom and <=0.5 point actual grouping
loss on natural TRAIN/development. Candidate support is unchanged. Synthetic
coverage is reported, not silently used to alter those inherited prerequisites.

Freeze plan, source closure and both specifications before generating TRAIN PCM.
Bind raw PCM, unchanged TypeScript replay exports, features, target arrays and
schedules. Freeze all nine final checkpoints before generating ANY fresh
challenge PCM/features or making new development predictions. Then freeze all
natural and fifty analytic-case predictions before the runner reads development
references. Save original voiced probabilities alongside analytic predictions.
This is source/manifests chronology, not an operating-system access log, and does
not restore statistical independence to already-consumed development/regressions.

## Every existing gate retained, fresh cases added

Natural advancement requires three-seed mean joint error, equally averaged across
speaker/condition cells, improve by >=0.5 percentage points against BOTH original
MLP and hybrid; each seed beats original MLP; no condition mean is >1 point worse
than MLP; mean unconditional reference-voiced success is not lower than MLP.

Every designated seed must reach >=95% within 50 cents separately on:

- Each original stationary 100/200/300 Hz whole case.
- Each original 100/200/300 Hz fully post-change region.
- Each of all twelve consumed harmonic-context post-change cases, including
  both 337.1 Hz cases.
- Each of all twelve consumed subharmonic-objective post-change cases, including
  both 354.5 Hz cases.
- Each of the twelve fresh prospective post-change cases below.

Fresh frequencies are `93.7,193.9,331.7,366.1,517.3,641.9` Hz, crossed with phases
`(0.43,-1.21,0.17)` and `(1.67,0.31,-0.93)` radians. Gain is 0.06. Keep 48 kHz,
2 seconds, change at 0.83 seconds and `(1,0.5,0.25)` to `(0,1,0.5)` coefficients.
Each has 191 common-eligible targets, including 112 fully post-change targets.
All frequencies and phase vectors are absent from the TRAIN grid and previous
challenge specifications. These are new configurations within the same noiseless
family, not an independent natural-speech or broad robustness evaluation.

Retain all fourteen original cases, all twenty-four consumed cases and twelve
fresh cases, with whole/pre/boundary/post results, candidate coverage,
voiced/unavailable counts and unchanged common denominators. Keep valid unvoiced,
unavailable and uncovered outcomes in natural scoring. No averaging away an
individual case/seed failure, dropping frames, changing masks/tolerances,
frequency exceptions or substituting metrics. The designated candidate is
promising only if ALL natural and analytic gates pass.

## Local budget, verification and stop

One existing Python/Torch MPS training process; CPU generation, inference,
scoring and independent audits. At most nine new fits, 30 minutes total fitting,
8 GiB observed MPS allocation, 2 GiB new experiment outputs and 8 GiB free disk
before execution. No installations, native builds, remote computation or extra
environment. Keep all attempts and refuse overwritten run directories. Stop
at the declared comparison, including a no-go; plumbing repair never permits
retuning or discarding completed fits.

Independently reconstruct schedules/targets, scores/gates and saved-checkpoint
inference; verify source/input/checkpoint/prediction hashes and chronology.
Historical warnings remain visible. Publish code, aggregate results and audit
receipts; raw speech, generated PCM, feature/reference arrays, weights and natural
frame outputs stay local and ignored. Browser qualification and production
promotion are outside this bounded experiment. Use `[skip ci]` when publishing
and verify that no hosted workflow runs were triggered.
