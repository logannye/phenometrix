# Step 1: candidate representation and acceptable-set learning

Declared 2026-10-22 before this experiment's headroom outcomes or new fits.
The user authorized the previously recommended step 1. Earlier failed gates and
published results remain historical evidence; this is a new, bounded diagnostic
experiment on the existing 20 ms measurement-delay task. It does not authorize
production promotion, a temporal model, calibration, or another test-set access.

## Question and staged gates

Can correcting candidate slot dependence, duplicate representation, and the
exclusive target objective recover pitch-selection errors using the same
acquisition evidence? First measure train/development headroom. The reference-
assisted oracle is an upper bound, never a deployable predictor or evidence of
learnability. Distinguish original-MLP versus raw-YIN complementary errors,
any-candidate coverage, and shared frozen-voicing failures.

Proceed to the bounded fits only if development's existing-candidate oracle,
under the frozen original voicing decisions, offers at least 1 percentage point
of equal-speaker/equal-condition joint-error improvement over the original MLP.
The set preprocessing must lose no more than 0.5 percentage points of candidate
coverage on either training or development (same equal-cell aggregation). Stop
and document a no-go if either prerequisite fails; do not change these gates in
response to outcomes. No training is required to compute either prerequisite.

## Data and support

Use only completed delayed-train-features and delayed-development-features,
matching acquisition manifests/references, and the original frozen MLP checkpoint
in delayed-mlp-seed20261022. All opened data and source identities are hashed.
The train and development speaker sets must be disjoint and match the frozen
PTDB split. No reserved/test audio, feature, prediction, or reference files are
opened. Existing aggregate reports may provide historical context only.

Use all four existing conditions: clean, noise10db, first_difference, clip01.
Use natural training only, with no augmentation or new candidate generator.
The previously used synthetic suites are regression data only and are never
included in fitting. Sparse natural support above 300 Hz remains a limitation.
Preserve 40 ms support, 10 ms hop, target at support end minus 20 ms, and the
existing shared validity/boundary mask. No offset fitting or reference repair.
A separate prerequisite audit may read development labels to measure headroom
and grouping coverage before fitting. The fitting/prediction runner freezes all
new development predictions before it opens development reference content for
scoring. Development has already been inspected;
this ordering does not restore independent holdout status.

## Fixed factorial comparison

Four pitch-only models cross representation and objective:

| Representation | Objective |
|---|---|
| Original fixed six slots, 57 inputs | Exclusive nearest acceptable candidate |
| Original fixed six slots, 57 inputs | Total probability of acceptable candidates |
| Shared candidate scorer with symmetric context and duplicate grouping | Exclusive nearest acceptable group representative |
| Shared candidate scorer with symmetric context and duplicate grouping | Total probability of acceptable group representatives |

Frozen original MLP p(voiced), threshold 0.5, is used identically for every
method. It is not retrained. New selectors cannot fix shared false-unvoiced
errors. Compare with the original MLP and original-voicing/raw-YIN hybrid.
Changing from joint voicing/pitch learning to pitch-only learning is common to
all four new arms; the fixed-slot nearest arm controls for that change.

Shared scoring uses train-only pooled candidate scaling: tanh Linear(8,32)
per group, masked elementwise-max context (all-empty context is zero), then
concatenate 32 group + 32 context + 9 global values, tanh Linear(73,32), and
shared Linear(32,1), totaling 2,689 parameters. Fixed slots use tanh Linear(57,32),
tanh Linear(32,32), Linear(32,6), totaling 3,110 parameters. Both use PyTorch
default initialization immediately after the declared seed. There is no state. Fixed-slot scoring uses the existing 32-by-32 framewise structure
without a voicing head. Report parameter counts; parameter equality is not
claimed. The representation intervention bundles grouping, scaling and shared
scoring, so its effect cannot be assigned to any one subcomponent.

Group sorted frequencies greedily with total group span at most 5 cents (no
single-link chaining). Merge evidence by per-source maximum. Emit the member
with greatest original maximum-evidence score, breaking ties by lower Hz.
Preserve member/source provenance. Recompute candidate fraction after grouping.
Exact duplicate replication must not change predictions; mere reordering must
not change frequency decisions (with deterministic ties). Nearby nonidentical
candidates may change grouped evidence or the representative and are not claimed
to be strictly invariant. Compute grouping coverage changes explicitly.

Both target objectives use the actually emitted frequency. Acceptable means
absolute pitch error <=50 cents. Acceptable-set loss is minus the log summed
probability on acceptable candidates. Candidate-absent, unvoiced and invalid
frames receive no invented pitch target. They remain in evaluation denominators.

## Fixed local budget and output controls

One local Python process, existing virtual environment, PyTorch MPS on this Mac.
Run all four arms for each of seeds 20261022, 20261023 and 20261024, in declared
order, 12 fixed epochs, Adam 0.001, gradient norm clip 1.0, shuffled frame batches
of 4096, and no checkpoint/epoch selection. All arms see the same original-six-candidate-covered, valid reference-voiced
training rows in the same seed-specific order (irrespective of frozen predicted
voicing). Grouped-uncovered rows remain in that schedule with zero pitch loss
and explicit counts. No arm independently filters its rows. Report optimizer
updates per arm; an all-uncovered batch is skipped and counted. Scaling uses input-only train
features. All 12 final checkpoints are retained and frozen before development
scoring. Do not add configurations, fit more seeds, or retune after results.

Cap fitting at 30 minutes total, observed MPS allocation at 8 GiB, and new raw
artifacts at 2 GiB. A failed or interrupted run remains visible and is not
silently replaced by a differently configured run. Check free disk before
execution. No package installation, Rust compilation, or remote computation.
Checkpoint, frame predictions and references stay in ignored runs directories;
only code, plans, aggregate metrics and reports may be committed.

## Outcomes and decision

Primary exploratory endpoint: joint error with equal weight for every
speaker/condition cell. Also report unconditional voiced success, false positives,
false negatives/unavailable, octave errors, candidate coverage, and each condition
and speaker. Report every seed and all four arms. The predesignated candidate is
shared-candidates plus acceptable-set, not the best observed arm or seed.

Call the candidate promising for the next temporal study only if its three-seed
mean joint error is at least 0.5 percentage points below BOTH original MLP and
hybrid, each seed improves over original MLP, no condition's mean joint error
worsens by more than 1 point versus original MLP, and mean unconditional voiced
success does not decrease. Report whether it also beats the new fixed-slot
nearest arm; without that comparison an improvement cannot be attributed to the
representation/objective changes.

Run existing analytic regression fixtures for all completed arms after checkpoint
freeze, retaining every result including 300 Hz. Require >=95% success on every
stationary case for every predesignated-candidate seed before claiming the known
stationary failure is resolved. These reused fixtures do not prove new-case
generalization. Do not start temporal modeling, calibration, fresh participant
evaluation, or production integration in this step. Report any failures as such.
