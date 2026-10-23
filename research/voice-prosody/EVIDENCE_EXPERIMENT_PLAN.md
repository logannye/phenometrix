# Prospective exposure × acoustic-evidence study

Frozen before scientific generation. This is the one study authorized in
`OVERNIGHT_DEVELOPMENT_PLAN.md`, based on revision
`e15b40b5cd39b383fa5a6c64a0d832e9611166ab`. The authorized window ends
2026-10-23 12:18:58 UTC. A failed prerequisite ends dependent science without
revision, fresh evaluation generation, or fits. Historical files remain exact.

## Question and fixed comparison

The added seventh candidate corrected candidate absence without correcting
selection. The existing 96-case harmonic-balance TRAIN bank supplied no
added-only correct candidates. Cross existing versus coverage-targeted TRAIN
exposure with absent versus present candidate-conditioned periodicity evidence.
The arms, in execution order within each seed, are `existing__off`,
`existing__on`, `targeted__off`, `targeted__on`. Seeds are 20261022, 20261023,
20261024. Predesignate `targeted__on`; the other arms are controls, never an
outcome-selected substitute. Maximum twelve fits, with no repeated completed
fit or additional sweep. Both data exposure and acoustic information may matter;
a negative interaction is also a valid result.

Every arm uses the same expanded seven-candidate pool and relative-context
acceptable-set scorer. Preserve original six raw candidates, original source
scores, deterministic grouping, the actual grouped emitted frequencies,
original 57-input MLP voicing, all eligibility and timing. The base has 3,009
parameters. Add one shared bias-free Linear(3,1) residual to each candidate
logit, initialized to three zeros. Both modes allocate 3,012 parameters. Off
feeds exact zeros and has zero residual gradients: allocated tensor capacity is
matched, effective input information and trainable capacity intentionally differ.
Base initialization and the caller RNG stream match the old model. No pitch
veto, hard-coded target frequency, new candidate or extra state is introduced.

## Fixed acoustic contract

For each actual grouped frequency f, use only the existing trailing 1,920
Float32 samples at 48 kHz. Remove the full-window mean in Float64. T=48000/f;
common anchor length L=1920-floor(T)-1. At lags
T*[1,.5,1/3,.25], linearly interpolate the shifted signal over this same anchor.
Compute normalized correlation from Float64 sums, clamp to [-1,1]. Require
anchor and all four shifted energies >=1e-12. Otherwise all correlations and
channels are zero with false validity. Preserve period, anchor length, all four
correlations and the explicit validity bit in local provenance. Padded
candidates have all-zero evidence and metadata. No active pre-support candidate
is accepted. Model channels are Float32 C(T),
[C(T)-max(C(T/2),C(T/3),C(T/4))]/2, and validity. No added fitted scaling.
Candidate grouping precedes this computation; evidence from different emitted
frequencies is never maximum-merged. The pure TypeScript and independent NumPy
implementations must agree on toy fixtures before extraction. A single fixed
1,000-frame toy extraction must reach >=200 frames/s with observed RSS<=1 GiB;
this is an extraction feasibility screen, not browser qualification.

## TRAIN specification and coverage-only prerequisites

The fixed new bank contains 512 two-second cases: carriers
round(53.113+10.125*i,3), i=0..63; phase triples (.17,-.59,1.23) and
(1.71,.43,-1.11); gains .035 and .075; coefficients change from (1,.5,.25)
to either (0,1,.5) or (.2,1,.5). Every recording has 96,000 samples at48kHz
and change sample39,840. There is no noise or normalization. Exact metadata
and hashes live in `evidence_cases.py`. No generator revision follows outcomes.
Metadata novelty excludes prior configured carriers/endpoints and old TRAIN
carriers; it does not assert independent frequency support or new populations.

Strata use only analytic TRAIN truth and candidate coverage at50 cents:

- Added-only: no original raw correct candidate; a distinct added raw peak is
  correct; the expanded grouped pool still offers a correct candidate.
- Counterexample: a distinct added raw peak is wrong; original grouped and
  expanded grouped pools both offer a correct candidate.
- Broad: every common-eligible TRAIN row, including uncovered rows. Duplicate
  old frequencies do not count as new peaks. Targeted strata are disjoint;
  broad overlaps both and all such overlap is reported.

Each targeted stratum must contain >=512 rows, >=24 complete configurations,
>=6 carriers, >=2 phase settings and >=2 gains. Report fixed100-Hz bin support
without adding a bin gate. Insufficient support means no fits, no larger bank,
no relaxed floor, and no fresh evaluation generation. Preserve all cases and
frame-table masks for an independent audit. Each case has196 acquired frames,
191 common-eligible targets:75 before,4 boundary,112 after change.

If feasible, every targeted synthetic batch draws128 added-only,128
counterexample,256 broad rows. For each stratum sample a case uniformly, then a
qualifying frame uniformly, with replacement; concatenate strata in that order
and permute512 positions. RNG seed=fit seed+2,000,000+epoch. Save33x512 indices
and origin labels for all36 seed/epoch pairs before fits. Report repeated rows,
case/region exposure, actual overlapping categories and covered-loss denominators.
This is the combined intervention of a new bank plus sampling, not an isolated
estimate of oversampling versus broad new-bank exposure.

The existing arm reuses the frozen96-case balance bank, all18,336 common rows,
with the old seed+1,000,000+epoch permutation, first16,896 rows reshaped33x512,
without replacement. Its original frequency/phase/gain configuration is exact.

## Matched training and chronology

Use the same132,466 natural raw-union schedule as the expanded-pool study,
33 batches of at most4,096 per epoch, twelve epochs,396 updates per fit.
Each seed/epoch uses NumPy seed+epoch permutation of this schedule, paired
across all four arms. Original-six natural TRAIN scaling only; identical
base/scaling initialization in all four arms. Adam learning rate.001,
gradient norm clip1, natural covered-row acceptable-set CE mean plus.25 times
the synthetic covered-row acceptable-set CE mean. Uncovered rows contribute
connected zero to loss and remain in recorded exposure. No optimization,
checkpoint or stopping choice uses DEV or analytic evaluation outcomes.
Every final epoch12 checkpoint is retained. Persist fit-start and fit-complete
records; interrupted fits stay failed and are not silently repeated.

Before fitting, recheck the consumed natural-data prerequisites without changing
thresholds: raw fixed-voicing oracle headroom >=1 percentage point and actual
raw-to-grouped lost reference-voiced fraction <=.5 points in both TRAIN and DEV,
using existing equal speaker/condition weighting. Record this explicit consumed
DEV reference read; the whole study is not globally label-blinded. Freeze the
references, prepared inputs, target arrays and all schedules. Fitting functions
receive TRAIN only. Preserve exact old input/weight/source hashes, and verify
old frozen inference. Exact CPU base initialization and fixture optimizer parity
are preflight contracts. Historical MPS refit equality is reported descriptively,
not an excuse to tune or replace a completed fit.

Record TRAIN-only per-stratum acceptable-vs-best-wrong logit margins, selection
accuracy and conditional loss before/after each fit. These diagnostics explain
learning and weighting; they do not select models or replace advancement gates.

## Fresh inputs, scoring and unchanged advancement

Retain all74 consumed analytic cases from `expanded-pool-v1`, including both
phases at337.1,354.5,517.3,641.9 and633.1Hz. Reserve twelve fresh configurations:
109.9,241.1,379.3,563.9,619.7,677.9Hz crossed with phase triples
(.79,-1.37,.23) and(2.17,.61,-.89); gain.057; before(1,.5,.25), after(0,1,.5);
96,000 samples,48kHz,change39,840. Freeze exact metadata before TRAIN generation.
Generate fresh PCM/features only after all twelve final checkpoint hashes freeze.
Freeze every pipeline's natural and all86 analytic predictions before final
scoring. No adaptation, second fresh set or rerun of fits after these outcomes.

The predesignated candidate must meet every unchanged gate: mean natural joint
error improves >=.5 percentage points against BOTH original MLP and hybrid;
each seed beats MLP; no condition mean worsens more than1 point against MLP;
mean unconditional reference-voiced success is not lower than MLP. Every seed
must separately reach >=95% within50 cents on each of3 stationary whole cases,
3 original post-change cases,60 consumed harmonic-change cases and12 fresh
post-change cases:78 gates per seed. No frame drops or averaging failures away.
Preserve all other step/glide/whole/pre/boundary/post results. Maintain40ms
support,10ms hop and20ms measurement delay. All prerequisites/natural/analytic
gates are conjunctive. Failure retains default DSP and blocks model promotion.

All natural speakers have contributed observed results. No new natural-data
acquisition, reserved/test access, untouched-speaker or clinical-validity claim.
The noiseless fresh family assesses these complete configurations only.

## Engineering, independent audits and hard bounds

Freeze exact source closure and source snapshots before scientific generation.
Engineering fixes after that freeze must be documented separately; they cannot
change this scientific policy, discard completed fits, or reuse a fresh set.
Independent audits reconstruct expansion/grouping, strata/support/schedules,
feature arithmetic, model equations, output states, scores and gate decisions.
A local failure report separates absent candidate, grouping loss, voicing,
wrong ranking and unavailable output. Raw logits are not calibrated confidence.
Only an all-gates pass permits a separately frozen browser export/replay attempt.
Feature-kernel parity and harness improvements are allowed after a no-go.

All processing, fitting, inference, tests and replay run on this Mac, using the
existing environment and MPS where appropriate. One fitting process, at most
60 cumulative fit minutes; observed Python RSS<=16GiB, Node-worker RSS<=2GiB,
MPS allocation<=8GiB; new disposable outputs<=8GiB; free disk>=16GiB. Bound every
subprocess and total scientific run by both fixed timeout and session deadline.
No installs, environments, native/Rust builds or remote compute. Stop new fitting
at09:48:58UTC and optional browser runs at10:48:58UTC. Reserve final90min for
checks/report/shipping. Preserve incomplete work at12:18:58UTC.

Update high-level README, detailed results, aggregate receipts and PR33 only
with coherent validated work. Run full research tests and `npm run check`;
commit/push with `[skip ci]` and verify zero GitHub Actions runs across all event
types. Keep media, features, references, weights and frame outputs local/ignored.
No merge and no default DSP change. Stop the heartbeat when the scope completes.
