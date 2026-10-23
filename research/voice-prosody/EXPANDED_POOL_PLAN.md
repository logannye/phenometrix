# Bounded additive candidate pool and selector retraining

## Question and fixed comparison

The preceding replacement policy restored correct candidates at 517.3/641.9 Hz
but displaced useful natural-speech hypotheses and left selection incorrect.
This study preserves the original pool and asks whether training on the added
evidence improves selection without relaxing any advancement gate.

Predesignate `expanded7`: retain every original v2 candidate frequency and source
score, and add the shortest-lag already-admitted full40 autocorrelation maximum.
Merge exact-equal frequencies by source-wise maximum, sort by frequency and pad
to seven slots. There are no frequency exceptions, new admission thresholds,
harmonic arithmetic or outcome-based fallback. The original strongest full40,
legacy and suffix hypotheses remain. Same-frequency added evidence may increase
a source score; no original frequency or source evidence is removed or weakened.

The new research schema has seven eight-field candidate blocks plus nine globals.
Candidate/group count keeps the historical unit `count/6`, which may reach 7/6;
it is not a fraction of the new capacity. Other eight globals are copied exactly.
The original six-slot schema remains unchanged. A new variable-width shared
selector implements the same 3,009-parameter relative-context equations, with
exact six-slot parity required in preflight. Complete-span five-cent grouping,
source-score maxima and original-member representative selection remain unchanged.
Raw coverage cannot decrease, but regrouping can change representatives and
must be audited separately. Original MLP voicing at 0.5 always consumes original
57-field inputs. Timing remains 40-ms support, 10-ms hop, 20-ms measurement delay;
correctness remains within 50 cents.

Exactly six new fits cross `original6` and `expanded7` with seeds 20261022,
20261023 and 20261024. The original arm is a matched refit control. Also evaluate
the three previous balance-augmentation checkpoints on BOTH pools without fits,
named `frozen-original6` and `frozen-expanded7`. These controls distinguish added
input evidence from retraining. Both new arms start with identical parameter
tensors and the same original-six natural-TRAIN scaling; no expanded-pool,
synthetic, development or evaluation scaling is fitted. Network size, objective,
optimizer and augmentation dose do not change. No alternate arm replaces the
predesignated candidate after outcomes are known.

## Frozen inputs and training schedule

Bind the completed candidate-coverage and augmentation manifests, prediction
freezes, historical input identities and three old checkpoints. Freeze this plan,
the new/old computational source closure and the fresh specification before
assembling expanded caches or fitting. Historical sources and evidence remain
immutable. Reuse exact original features and hash-bound peak provenance from
the preceding run for all 960 natural TRAIN and 320 development condition-inputs
and 62 consumed analytic cases. Validate the old full40 peak/score against the
original blocks and preserve every old candidate. This does not independently
recompute all historical waveform features or peak curves.

Reuse the 96 existing balance-augmentation TRAIN PCM/features, unchanged at
24 frequencies, two phases and two gains. Obtain only the additional observed
peak through the unchanged local TypeScript peak worker. No new TRAIN waveform
family or targeted evaluation carrier is introduced. Keep 191 common-eligible
rows per case, all 18,336 rows, including any uncovered zero-loss rows.

Natural TRAIN retains 240 recordings from twelve speakers under clean,
noise10db, first_difference and clip01; development retains eighty recordings
from four disjoint speakers under the same conditions. All natural development
and old regressions are consumed research evidence. No reserved/test content,
new natural acquisition or external service is used.

Use the UNION of raw-covered natural TRAIN targets under both pools as the
shared schedule, preserving newly coverable examples. Derive and freeze it before
fitting. Prior consumed coverage predicts exactly 132,466 rows: the original
132,181 plus 285 newly covered rows. Stop before fitting if that identity fails
or the schedule requires other than 33 batches of size at most 4,096. Both arms
use the same twelve NumPy permutations, seed `seed+epoch`, including rows on
which their own grouped targets are uncovered and contribute zero loss. Because
this changes minibatch composition, exact historical fitted-weight reproduction
is not expected; report control drift. Six-pool frozen inference must still
reproduce the historical state/frequency outputs exactly.

Each fit uses twelve epochs, 396 Adam updates, learning rate .001, gradient
norm clipping at 1, and no skipped batches or checkpoint selection. Each update
minimizes covered-row acceptable-set cross entropy on the natural batch plus
0.25 times the same loss on 512 synthetic rows. The paired synthetic order is
the first 16,896 rows of the 18,336-row permutation from
`seed+1_000_000+epoch`, reshaped to 33 by 512. Persist every schedule, target,
covered/uncovered count, case/region exposure and loss contribution. Effective
conditional-mean weighting can differ when candidate coverage changes; report it.

## Prerequisites, prediction freeze and fresh evaluation

Freeze all input-only natural caches and original voiced probabilities first.
Then compute the inherited candidate prerequisites from exact natural TRAIN/DEV
references: development RAW fixed-original-voicing oracle joint-error headroom
at least one percentage point over original MLP; actual raw-to-grouped lost
reference-voiced fraction at most 0.5 points on TRAIN and DEV, equally weighted
over speaker/condition cells. Report raw/grouped original-to-expanded gains and
losses. Stop on any prerequisite failure, without a policy revision or fit.

This explicitly declared prerequisite phase reads consumed DEV references before
training for pass/stop checks. Freeze that receipt, target arrays and schedules
before fitting; no DEV reference values enter model inputs, training labels,
scaling or loss. Final DEV scoring reloads those exact references only after all
predictions freeze. This is source/manifests chronology, not an operating-system
access audit, and does not claim global DEV-label blinding or independence.

Freeze all six final checkpoints and source identities before generating ANY
fresh evaluation PCM/features. The twelve new complete configurations cross
103.7, 229.1, 361.7, 541.9, 633.1 and 683.7 Hz with phase triples
(.53,-.97,1.47) and (2.03,.71,-1.29), gain .052. Keep 48 kHz, 96,000 samples,
change sample39,840, coefficients (1,.5,.25) to (0,1,.5), no noise/normalization.
Each has 191 common targets, 112 fully post-change. These fixed carriers and
configurations avoid prior configured carriers/endpoints and TRAIN grids;
continuous glides and natural speech can cross them. The family remains a
failure-motivated noiseless synthetic family, not external generalization.

Retain all 62 prior analytic cases, including 337.1/354.5 Hz repairs and both
phases of the 517.3/641.9 Hz failures. Save all twelve pipelines' development and
74 analytic-case predictions plus original voiced probabilities. Freeze their
hashes before final DEV or analytic scoring. Retain every whole/pre/boundary/post
and step/glide result, candidate availability, grouped coverage, voicing state,
unavailable output and denominator. No failed case or frame is removed.

## Unchanged advancement gates and stopping rule

Natural advancement requires three-seed mean joint error improve by at least
0.5 percentage points against BOTH original MLP and hybrid; each seed beats
MLP; no condition mean exceeds MLP by more than one point; mean unconditional
reference-voiced success is not below MLP. Use the original eligibility, scoring
and equal speaker/condition weighting. Every expanded7 seed must separately
achieve at least 95% within50 cents on EACH of three original stationary whole
cases, three original post-change cases, all 48 consumed harmonic-change cases
and twelve fresh post-change cases: 66 mandatory case gates per seed. All
prerequisites, natural and analytic gates must pass for a promising result.

One bounded scientific execution, at most six fits, one local Python/MPS training
process and one sequential TypeScript peak worker; CPU handles assembly, inference,
scoring and audits. Use the existing M4 Max/128 GiB environment. Require 8 GiB
free disk, cap execution at 75 minutes, total fitting at 30 minutes, observed
Python RSS at16 GiB, Node-worker RSS at2 GiB, MPS allocation at8 GiB, and new
run artifacts at2 GiB. These process/allocator observations do not bound total
machine memory. Use per-request deadlines and an external wall-time supervisor.
No installation, native build, remote compute or extra environment.

Stop at the fixed comparison, including a no-go. Preserve failed attempts;
documented plumbing repair cannot retune the policy, discard or repeat completed
fits. Independently reconstruct grouping/retention, targets/schedules, scores,
gates and checkpoint equations. Publish code, aggregates and audit receipts;
raw audio, PCM, features, references, weights and natural frame predictions stay
local and ignored. Update the high-level README and existing PR with `[skip ci]`
and verify zero hosted runs. Browser qualification and production promotion
remain outside this experiment.
