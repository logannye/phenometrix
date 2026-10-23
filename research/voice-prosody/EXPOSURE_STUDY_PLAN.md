# Prospective exposure × acoustic-evidence study on the refined TRAIN bank

Base revision: `5e33e118a6da247bc6cabe05ad91eb9d1fbcf95a`.
The user authorizes one bounded local four-arm, three-seed study using the new
TRAIN bank, unchanged gates, preserved failures, and fresh evaluation inputs.
This is a new run, not a restart of the expired overnight attempt. Preserve all
historical sources, runs and no-go decisions. Freeze this plan, new runner and
adapters, independent auditor, tests, complete dependency closure, exact input
identities and supervisor before any scientific feature preparation or fitting.
No scientific source repair, parameter sweep, completed-fit repeat, alternate
fresh set, adaptive bank enlargement or outcome-based substitution is permitted.

## Question, arms and model

Cross existing versus coverage-targeted TRAIN exposure with periodicity evidence
off versus on. Execute seeds **20261022, 20261023, 20261024**, each with arms
**existing__off, existing__on, targeted__off, targeted__on** in that order. There
are at most **twelve fits**. Predesignate **targeted__on** for every advancement
gate; other arms are explanatory controls and cannot replace it after results.
Report the exposure effect in evidence-off models, the evidence effect in each
exposure condition, and their difference-in-differences interaction. This tests
a bank-plus-sampling intervention, not an isolated oversampling effect.

Use the exact previously frozen expanded seven-candidate pool, grouping,
relative-context acceptable-set scorer and original 57-input MLP voicing. The
base scorer has 3,009 parameters. A shared bias-free three-weight residual starts
at zero; both modes allocate 3,012 parameters. Off supplies exact zero channels
and has zero residual gradients. Effective information and trainable capacity
therefore differ despite matched allocated capacity. Match CPU base
initialization, original-six TRAIN-only scaling, and seed RNG streams. Do not
add candidates, state, pitch vetoes, frequency rules, calibration or voicing
inputs. Preserve 40 ms trailing support, 10 ms hop and 20 ms measurement delay.

The unchanged acoustic kernel evaluates each actual grouped frequency f on the
last 1,920 Float32 PCM samples at 48 kHz. Center in Float64; T=48000/f,
L=1920-floor(T)-1. Interpolate shifts T, T/2, T/3 and T/4 over the same anchor,
normalize Float64 dot products and clamp correlations to [-1,1]. All five
energies must be at least 1e-12. Otherwise correlations and channels are zero
and validity false. Model channels are Float32 C(T),
[C(T)-max(C(T/2),C(T/3),C(T/4))]/2, and validity. Padded slots stay zero. Retain
period, anchor length, raw correlations, validity and exact emitted frequency.
The inherited source-bound single 1,000-frame toy feasibility receipt passes
>=200 frames/s and <=1 GiB RSS; verify its unchanged source and receipt instead
of repeating or selecting a timing trial. Rerun the existing fixed toy parity
and causality tests as part of the preflight suite.

## Exact TRAIN inputs and sampling

The existing exposure bank is the exact 96-case balance bank from
`expanded-pool-v1`: 18,336 common-eligible rows. Preserve its seed+1,000,000+epoch
permutation without replacement, using the first 16,896 rows as 33×512 batches.

The targeted bank is **only the 1,536 configurations** already acquired in
`frequency-diversity-v1`, not the old-plus-new combination. Bind its manifest,
source freeze, complete report, table, independent audit and every PCM/original/
expanded/provenance artifact. Do not regenerate it. Its fixed lattice is
(53,113,000+2,531,250*i)/1,000,000 Hz for i=0..255 with i%4!=0, crossed with two
phase settings, two gains and two harmonic changes. Retain all 301,056 acquired
rows and 293,376 common-eligible rows. The previously audited added-only stratum
has 11,456 rows, 104 configurations and 13 numeric carriers; counterexamples have
79,576 rows, 716 configurations and 90 carriers. Independently recheck these
identities and all unchanged support floors before permitting fits.

Each targeted stratum still requires >=512 rows, >=24 complete configurations
represented, >=6 numeric carrier frequencies, >=2 phases and >=2 gains, at the
inclusive 50-cent correctness threshold. Added-only means no correct original
raw candidate, a distinct correct added peak, and a correct expanded grouped
candidate. Counterexample means a wrong distinct added peak and correct
original/expanded grouped candidates. Broad includes all common-eligible rows,
including uncovered rows. Targeted strata are disjoint; broad overlaps both.

For each seed and epoch, RNG seed+2,000,000+epoch draws 128 added-only,
128 counterexample and 256 broad rows per batch. Within each stratum choose a
case uniformly and then a qualifying frame uniformly, both with replacement;
concatenate strata in that order and permute 512 positions. Save all 36
seed/epoch schedules, origin labels, repeated/unique rows, actual overlapping
strata, case/region exposure and loss denominators before fitting. No model
scores, DEV labels or fresh cases influence this sampling.

Preserve all 1,844 uncovered rows at 696.0505 and 698.58175 Hz in the broad
population. Uncovered rows contribute connected zero loss and remain in exposure
accounting. Do not filter or repair these known TRAIN failures. The prior
512-case four-carrier no-go stays a separate historical result, without reuse
as extra targeted examples. No natural-data acquisition or reserved/test access.

## Prerequisites, fitting and chronology

Use the exact consumed natural TRAIN and development inputs from the frozen
expanded-pool study. Recheck DEV raw fixed-voicing oracle headroom >=1
percentage point over the original MLP, and actual raw-to-grouped lost
reference-voiced fraction <=0.5 percentage points in each of TRAIN and DEV,
under existing equal-speaker/condition weighting.
Record the consumed DEV reference read for prerequisites; the study is not
globally label-blinded. A failed prerequisite or feature contract terminates
dependent science before fits and fresh input generation.

Freeze references, prepared feature inputs, target arrays, and every schedule.
Natural raw-union training has 132,466 rows, permuted with NumPy seed+epoch,
33 batches of at most 4,096 rows per epoch, paired across all four arms.
Each fit performs exactly twelve epochs and 396 optimizer updates: Adam lr .001,
gradient norm clipping at 1, natural covered-row acceptable-set CE mean plus
0.25 times the synthetic covered-row acceptable-set CE mean. No optimization,
stopping or checkpoint choice uses DEV or analytic evaluation outcomes. Retain
only the prescribed final epoch-12 checkpoint as the scientific result, while
preserving failures and incomplete artifacts. Persist fit-start/complete state
and cumulative elapsed fitting time; an interrupted fit is not silently retried.

Record TRAIN-only per-stratum correct-selection rate, conditional loss and
best-acceptable-minus-best-wrong logit margin before and after every fit.
These are descriptions, not selection criteria. Keep natural training and
synthetic row identities aligned with their emitted candidate evidence. Clear
TRAIN-only targets before the evaluation phase. Preserve exact historical
frozen inference as a control; MPS refit differences are descriptive and never
justify replacing a completed run.

## Fresh evaluation and unchanged advancement

The original twelve reserved configurations remain unconsumed. Freeze their
exact specification before fitting: frequencies **109.9, 241.1, 379.3, 563.9,
619.7, 677.9 Hz**, phases **(.79,-1.37,.23)** and **(2.17,.61,-.89)**, gain .057,
before harmonics (1,.5,.25), after (0,1,.5), 96,000 samples at 48 kHz, change
sample 39,840; IDs `evidence-balance-f{i}-p{j}`. Their source metadata and exact
configuration exclusion checks are unchanged. Freshness refers to complete
ungenerated configurations, not independent frequency support: the nearest
refined TRAIN carrier is only 0.34425 Hz from a reserved carrier.

Generate these twelve inputs once, only after every one of the twelve final
checkpoint hashes is frozen and reverified. Retain all 74 consumed analytic
regressions, including failures at 337.1, 354.5, 517.3, 641.9 and 633.1 Hz. Freeze
all model/control predictions for consumed natural DEV and all 86 analytic
cases before final scoring. No adaptation or second fresh set after outcomes.
All new cases become consumed regressions after this evaluation.

The predesignated candidate must pass every existing gate: mean natural joint
error improves >=0.5 percentage points against BOTH original MLP and hybrid;
each seed beats MLP; no condition mean worsens >1 point against MLP; mean
unconditional reference-voiced success is not lower than MLP. Every seed must
separately attain >=95% within 50 cents on each of three stationary whole cases,
three original post-change cases, sixty consumed harmonic-change cases and
twelve fresh post-change cases: **78 conjunctive analytic gates per seed**.
Report all other whole/pre/boundary/post/step/glide outcomes. No frame drops,
averaging failures away, tolerance changes or gate relaxation. Any failure
preserves default DSP and the learned-model promotion no-go. No export/browser
qualification is part of this study, even if these scientific gates pass.

## Independent verification and bounds

The frozen auditor independently reconstructs metadata, retained candidates,
grouping, strata, schedules, targets, loss denominators, checkpoint/phase
chronology, original voicing, all twelve model equations and saved output
states, and natural/analytic scores and gate decisions. It does not rerun
optimization or independently replay the TRAIN logit-diagnostic summaries,
which remain hash-bound producer descriptions. Saved feature-channel arithmetic and existing independent toy
TypeScript/NumPy parity remain separate from raw-signal replay.

Additionally freeze a bounded raw-periodicity audit: for EACH of the 1,536
saved targeted TRAIN cases independently recompute three windows at common-row
ordinals 0, 76 and 190 (acquired indices 4, 80 and 194), covering pre-change,
boundary and post-change. Use scalar Float64 centering, interpolation and dot
products over the saved PCM and actual grouped candidates; do not call the
producer's periodicity kernel. Use the established absolute tolerances 2e-13
for correlations and 1e-7 for Float32 channels, with exact validity, period and
anchor checks. This covers 4,608 frames; it is not a replay of all natural
features or the original candidate extractor. A disagreement is a recorded
qualification failure, without tolerance change or scientific retry.

All processing, fitting, inference, tests and audits run locally in the existing
Mac environment. Use MPS for fitting, one fitting process, <=60 minutes
cumulative fitting, and <=3 hours scientific wall time from initial source
freeze; an external watchdog ends at 3 hours plus 30 seconds. Observe Python
RSS <=16 GiB, MPS allocated memory <=8 GiB, Node/launcher RSS <=2 GiB, new run
plus supervisor outputs <=8 GiB, free disk >=16 GiB. Bound child requests by
60 seconds and the remaining scientific budget. Use the exclusive retained
process lease, no concurrent study or duplicate fit, with process-group and
observed detached-descendant cleanup. These are sampled resource observations,
not whole-machine or sub-poll memory guarantees. Bound the separate independent
post-run audit to 30 minutes with the same Python RSS ceiling.

Before execution, pass fixed toy tests for the new adapters, schedule/metadata
identity, complete-path schemas, zero-residual/optimizer parity, numerical
causality, timeout/no-overwrite controls, audit corruption detection and
supervisor lifecycle. Preserve preflight failures and repairs before freezing.
Afterward run independent audits, full research tests and `npm run check`.
Publish only coherent source, tests, aggregate receipts and documentation.
Update high-level README, detailed results and PR #33; commit/push with
`[skip ci]`, verify zero Actions runs across all event types, and do not merge.
Keep raw media, features, references, frame outputs and weights local/ignored.
