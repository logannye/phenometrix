# Expanded candidate pool: coverage preserved, ranking still fails

**The six-fit experiment is complete and remains a no-go under unchanged gates.**
Preserving all original candidates fixes the earlier eviction problem, but
retraining the selector does not fix the 517.3/641.9 Hz failures. Mean natural
development joint error is **8.4589%**, compared with **8.4020%** for the matched
original-pool refit. Two fresh 633.1 Hz configurations also fail despite complete
expanded-pool coverage. No estimator is promoted and the default remains DSP.

The [fixed plan](EXPANDED_POOL_PLAN.md),
[aggregate evidence](../../benchmarks/voice-prosody/expanded-pool-summary.json)
and [local validation receipt](../../benchmarks/voice-prosody/expanded-pool-local-validation.json)
preserve the hypotheses, controls, failures and exact source identities.

## What changed and what stayed fixed

The new research-only acquisition schema retains every original candidate
frequency and source score, then adds at most one shortest-lag already-admitted
full-window autocorrelation maximum. Exact-frequency duplicates merge source
scores by maximum; no old score decreases. The pool grows from six to at most
seven candidates. Admission remains 0.35, bounds 50–700 Hz, and interpolation
and observed-confidence semantics are unchanged. Complete-span five-cent
grouping still emits an actual member frequency. Raw coverage cannot decrease;
regrouping can change representatives and is audited separately.

The variable-width scorer has the same **3,009 parameters** and exact six-slot
preflight parity. Both new arms use identical seed initialization and original-six
natural-TRAIN scaling. Original MLP voicing always uses the original 57-field
features and 0.5 threshold. Support remains 40 ms, hop 10 ms, measurement delay
20 ms, and correctness within 50 cents. There are no targeted frequency rules.

Exactly six local MPS fits cross original-six and expanded-seven pools with
seeds 20261022–20261024. Each uses twelve epochs and 396 Adam updates at learning
rate .001, gradient clipping at 1, and natural covered-set cross entropy plus
0.25 times the same loss on 512 synthetic rows per update. No checkpoint or arm
is selected after evaluation. Three historical balance-augmentation checkpoints
are also evaluated on both pools without fitting, producing twelve selector
pipelines in total.

## Matched training and evaluation boundary

Natural TRAIN contains 240 recordings from twelve speakers; development contains
eighty recordings from four disjoint speakers. Each has four fixed conditions.
The shared natural schedule is exactly **132,466** raw-covered rows: 132,181
originally covered plus 285 newly covered. Both arms use the same twelve
permutations and 33 batches per epoch. Grouped uncovered rows remain in the
schedule with zero loss: 392 for original-six and 105 for expanded-seven.
Changing the shared schedule changes minibatch composition; the original refit
is a matched control, not a required bit-exact reproduction of old training.

The unchanged synthetic bank has 96 harmonic-balance sequences at 24 frequencies,
two phases and two gains. **Both pools already cover all 18,336 synthetic targets.**
Thus this expansion adds no newly coverable synthetic training rows. Each epoch
uses the same 16,896 distinct rows in both arms. Target masks, schedules, per-case
and per-region exposures, initial tensor hashes, losses and six final checkpoints
remain bound to the run. Conditional loss denominators differ when natural
coverage differs; synthetic denominators are identical.

A 45-file source closure and specification freeze precede input-cache assembly.
Historical natural and consumed-case peak provenance is reused and hash-checked;
this is not a new full waveform replay of all historical features. Input caches
freeze before consumed TRAIN/development reference content is read for the
predeclared candidate prerequisites. Those pass/stop checks all pass. References
do not enter model inputs, scaling or development-derived training labels.
Final development scoring reloads the same references after all predictions
freeze. This is source/manifests chronology, not an operating-system access audit
or a claim of globally blinded development evaluation.

All six checkpoints freeze before twelve new analytic configurations are generated.
They cross 103.7, 229.1, 361.7, 541.9, 633.1 and 683.7 Hz with two new phase
triples and gain .052. Each two-second signal removes its first harmonic at
sample 39,840 while harmonics two and three preserve the fundamental period.
Each has 191 common targets, including 112 fully post-change targets. All 62
previous cases remain, for 74 total. These new complete configurations belong to
a familiar noiseless family; natural speech and historical glides can cross their
frequencies. All natural development and earlier cases remain consumed evidence.

## Candidate coverage and selection

Counts below are per case's 112 fully post-change targets. Selection outcomes
hold separately for every new and historical selector seed.

| Cases | Original raw/grouped coverage | Expanded raw/grouped coverage | Correct original selections | Correct expanded selections |
| --- | ---: | ---: | ---: | ---: |
| 517.3 Hz, either phase | 0/112 | 112/112 | 0/112 | 0/112 |
| 641.9 Hz, either phase | 0/112 | 112/112 | 0/112 | 0/112 |
| Fresh 633.1 Hz, phase 0 | 24/112 | 112/112 | 24/112 | 24/112 |
| Fresh 633.1 Hz, phase 1 | 23/112 | 112/112 | 23/112 | 23/112 |
| Other ten fresh configurations | 112/112 | 112/112 | 112/112 | 112/112 |

All these windows pass the unchanged voicing gate and have available outputs.
The remaining error is selection: correct expanded candidates exist, but the
model chooses incorrectly. The targeted old failures have approximately
one-octave downward errors. The fresh 633.1 Hz failures show that this problem
extends beyond the two previously diagnosed carriers.

The repaired 337.1/354.5 Hz cases, all original stationary/change gates, all
24 consumed harmonic/subharmonic cases and all twelve consumed candidate-coverage
cases retain their passes. Expanded-seven passes **60 of 66 mandatory case gates
per seed**; the four old target cases and two new 633.1 Hz cases fail. Gates are
conjunctive, so the passes cannot cancel those failures. Whole, pre-change,
boundary, post-change, step and glide outcomes remain in the aggregate evidence.
The new failures are retained as named regression evidence for any future study.

## Natural speech and unchanged advancement gates

Joint error includes incorrect voicing, unavailable outputs and voiced pitch
errors beyond 50 cents. Results equally weight speaker/condition cells, then
selector seeds; they are exploratory results on reused development speech.

| Method | Mean joint error | Unconditional reference-voiced success |
| --- | ---: | ---: |
| Original MLP | 8.6349% | 75.4057% |
| Original hybrid | 8.9224% | 74.1933% |
| Historical weights, original-six | 8.4018% | 76.3961% |
| Historical weights, expanded-seven | 8.4339% | 76.2644% |
| Matched refit, original-six | **8.4020%** | 76.3962% |
| Retrained expanded-seven | **8.4589%** | 76.1446% |

Expanded-seven seed errors are 8.4643%, 8.4751% and 8.4373%. Mean error is
0.0569 percentage points worse than the paired refit and 0.0251 points worse
than historical weights on the expanded pool. Original-refit drift from the
historical original-pool mean is only +0.00018 points. No statistical superiority
claim follows from these three seeds on reused data.

The candidate improves over original MLP by **0.1760 points** and hybrid by
**0.4635 points**; both miss the required 0.5-point gain. Each seed beats MLP,
all condition means improve over MLP, and unconditional voiced success exceeds
MLP. These component passes do not override the failed aggregate and case gates.

| Split / representation | Original coverage | Expanded coverage | Lost targets | Gained targets |
| --- | ---: | ---: | ---: | ---: |
| TRAIN / raw | 86.3825% | 86.5736% | 0 | 285 |
| TRAIN / grouped | 86.3135% | 86.5059% | 3 | 290 |
| Development / raw | 88.1304% | 88.2399% | 0 | 65 |
| Development / grouped | 88.0885% | 88.1981% | 0 | 65 |

Coverage percentages are macro-averaged; target counts are pooled. The three
TRAIN grouping losses arise from changed representatives, not raw eviction.
Expanded raw-to-grouped loss is 0.0677 points on TRAIN and 0.0418 points on
development, below the unchanged 0.5-point prerequisite. Fixed-original-voicing
raw oracle headroom is **1.3297 points**, above the one-point prerequisite.
Reference exclusions remain 4,800 boundary targets and 2,880 predictions outside
reference extent on TRAIN, and 1,600/960 on development; no invalid reference
values are excluded. No denominator is trimmed to improve a result.

## Local execution and independent verification

The single scientific run completed in **158.90 seconds**, including **61.69
seconds of fitting**, on the local Apple M4 Max/128 GiB machine. It used the
existing Python 3.11.15, PyTorch 2.9.1 and NumPy 2.2.6 environment. Peak observed
Python RSS was **4.73 GiB**, Node-worker RSS **158.33 MiB**, and MPS allocation
**79.42 MiB**. Completion artifacts occupied **125.74 MiB** before audit receipts.
The external supervisor exited normally within its 75-minute deadline. No
package installation, native build, remote computation or additional fit occurred.
Process/allocator observations do not establish whole-machine memory bounds.

The 133-check preflight suite and a tiny MPS forward/backward probe passed before
the run; the probe performed zero optimizer updates. The full local suite now
passes **635 research tests** and `npm run check`. Three historical NumPy
matrix-multiplication warnings remain recorded in the validation receipt; they
are not treated as diagnosed or resolved by passing output checks.

The [independent score/training audit](../../benchmarks/voice-prosody/expanded-pool-independent-audit.json)
passes **167,060 numeric comparisons**, maximum difference **3.33e-16**. It
reconstructs candidate retention/grouping, targets, schedules, training exposure,
natural and analytic scores, chronology and all unchanged gates. All 8,864 bound
artifacts remain stable. The [target-free inference audit](../../benchmarks/voice-prosody/expanded-pool-inference-audit.json)
independently reconstructs explicit CPU equations for all nine distinct
checkpoints and twelve pipelines, verifying **3,233,712 frame/pipeline outputs**
with zero state, frequency, voicing-gate or candidate-membership mismatches.
**801,372 historical-control state/frequency pairs** reproduce exactly.

Original voicing reconstruction covers 269,476 frames; the maximum probability
difference is 2.44e-7 with zero threshold-decision changes. The target-free audit
reads no reference values, analytic targets, challenge specification, score
content or training arrays. Both full audits pass on their first execution;
preflight-only schema and ledger corrections preceded those executions. No
scientific source, fit, prediction or gate was changed after the freeze.

Published artifacts contain code, aggregate metrics and audit receipts. Raw
speech, generated PCM, references, features, checkpoints and natural per-record
and frame outputs remain local and ignored. Browser performance from earlier
studies does not qualify this new offline model.

## Decision and next question

The additive pool solves raw candidate eviction, but this fixed retraining
recipe does not solve selection. The study stops with a no-go, retains every
failure, and leaves the default capture path unchanged.

The training bank's complete original-pool coverage is a useful limit on this
result: it does not directly teach many examples in which only the newly admitted
candidate is correct. A separately declared study could first measure and add
TRAIN examples with that evidence pattern, keeping these failures as regressions
and reserving new complete configurations. Whether such examples suffice, or
whether the scorer needs more discriminative signal evidence, remains untested.
Independent natural speech, browser qualification, uncertainty calibration and
clinical validation remain separate requirements.
