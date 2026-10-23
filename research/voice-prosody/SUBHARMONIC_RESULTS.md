# Subharmonic-ranking objective results

The fixed nine-fit experiment is complete. **The predesignated alias-margin model remains a no-go.** It partially repairs the failing 337.1 Hz seed from 0/112 to **56/112 and 58/112** correct post-change targets across the two phases, below the unchanged 95% gate. Natural development joint error worsens from the refitted control's **8.4492% to 8.5126%**. New 354.5 Hz cases expose an F0/2 failure in every new arm and seed, despite correct candidates being available. No gate, candidate, model architecture, voiced threshold or training schedule was relaxed.

## What the diagnosis established

The [prefit diagnosis](../../benchmarks/voice-prosody/subharmonic-ranking-diagnosis.json) binds the old checkpoints and TRAIN-only analysis. At a saved failing center frame, two acceptable candidates near 333.33 and 337.01 Hz have about 50.3% combined softmax mass, while the wrong near-112.38 Hz candidate has 46.9% and beats either acceptable candidate individually (best acceptable about 41.2%). These are uncalibrated selector probabilities, not correctness confidence. Set loss still penalizes the wrong candidate; the finite checkpoint nevertheless makes the wrong argmax choice. The new hinge explicitly compares the best acceptable logit with the strongest admitted wrong logit.

The old seed-23 median correct-minus-alias margins are approximately -0.1271 and -0.1301; passing seeds have positive margins. Correct candidates and original positive voiced decisions are present in all 112 post-change frames of both old cases. Training has 132,074 grouped-covered rows, but only **517 at reference F0 >=300 Hz**, including 126 with a one-third-frequency negative. Overall, 47,116 rows have harmonic aliases. This motivates a training-support hypothesis; it does not establish that augmentation or another architecture will solve the problem.

The current affine/tanh/max context also has query-independent winning neighbors per channel in exact arithmetic: subtracting the query's scalar frequency contribution occurs equally for every neighbor before the monotone transformation. The final scorer remains query-dependent, and passing seeds show this is not proof of representational impossibility. No architecture change was made in this study.

## Frozen comparison and natural results

The [plan](SUBHARMONIC_EXPERIMENT_PLAN.md) predesignates alias-margin before fitting. All three arms use the same 3,009-parameter relative-context model, identical same-seed initial tensors, the same natural data and original MLP voicing. Set retains the original acceptable-set loss. All-margin and alias-margin add **0.5 times a one-logit hinge**, averaged over covered minibatch rows with admitted wrong candidates. All-margin uses all wrong candidates; alias-margin uses wrong candidates within 50 cents of factors 2, 3, 4 or their reciprocals of any acceptable emitted candidate. Both error directions are included. There is no frequency-specific rule or inference-time margin/reference input.

| Objective | Seed 22 error | Seed 23 error | Seed 24 error | Mean error | Mean voiced success |
| --- | ---: | ---: | ---: | ---: | ---: |
| set | 8.4341% | 8.4673% | 8.4460% | 8.4492% | 76.2170% |
| all-margin | 8.4767% | 8.4852% | 8.3572% | 8.4397% | 76.2506% |
| alias-margin | 8.5140% | 8.5268% | 8.4969% | 8.5126% | 75.9527% |

Original MLP joint error is 8.6349%, and hybrid error is 8.9224%. The designated model gains only **0.1224 and 0.4098 percentage points**, respectively, below the required 0.5 against both. Each candidate seed still beats original MLP, every condition improves on average, and unconditional voiced success improves over MLP's 75.4057%. Those component passes do not override the failed minimum improvement. The candidate is **0.0634 points worse than the refitted set control**. The better observed all-margin mean is diagnostic; it does not replace the designated arm and also has analytic failures.

All-margin covers 97,181 training rows and 150,898 negative candidates; alias-margin covers 47,116 rows and 55,652 negatives. Their contrast bundles negative selection and row reweighting. Every fit retains the same 132,181 scheduled rows, including 107 grouped-uncovered rows, twelve epochs and 396 optimizer updates without skips. Per-epoch base/weighted-auxiliary contributions are recorded in the aggregate artifact; they are covered-row-weighted averages of minibatch objectives, not a global eligible-negative-row mean.

## The consumed challenge remains a required regression

Designated-candidate fully post-change outcomes, with 112 targets per case:

| Case | Seed 22 | Seed 23 | Seed 24 |
| --- | ---: | ---: | ---: |
| 87.3 Hz / p0 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 87.3 Hz / p1 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 137.9 Hz / p0 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 137.9 Hz / p1 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 223.7 Hz / p0 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 223.7 Hz / p1 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 337.1 Hz / p0 | 100.000% (112/112) | 50.000% (56/112) | 100.000% (112/112) |
| 337.1 Hz / p1 | 100.000% (112/112) | 51.786% (58/112) | 100.000% (112/112) |
| 467.3 Hz / p0 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 467.3 Hz / p1 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 613.7 Hz / p0 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 613.7 Hz / p1 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |

The two 337.1 Hz seed-23 outcomes improve to 50.000% and 51.786%, but both fail. Other seeds and frequencies retain 100% post-change success. Set reproduces the old 0/112 failure. All-margin keeps seed 23 at 0/112 and introduces a seed-22 regression to 14/112 in both phases. Candidate pass counts are 34/36 seed/case combinations for set and alias-margin, and 32/36 for all-margin; aggregate pass counts cannot hide a failed individual gate.

## Prospective frequency/phase challenge

All nine checkpoints froze before generation. Six new frequencies cross phases (0.19,-0.83,0.67) and (1.43,0.59,-1.07) radians, retaining 48 kHz, 2 seconds, gain 0.06, change at 0.83 seconds and harmonic coefficients (1,0.5,0.25) to (0,1,0.5). Harmonics 2 and 3 remain nonzero. This is a failure-motivated prospective check of new configurations within the same noiseless family, not independent natural speech.

| Case | Seed 22 | Seed 23 | Seed 24 |
| --- | ---: | ---: | ---: |
| 79.7 Hz / p0 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 79.7 Hz / p1 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 151.3 Hz / p0 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 151.3 Hz / p1 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 319.7 Hz / p0 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 319.7 Hz / p1 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 354.5 Hz / p0 | 0.000% (0/112) | 0.000% (0/112) | 0.000% (0/112) |
| 354.5 Hz / p1 | 0.000% (0/112) | 0.000% (0/112) | 0.000% (0/112) |
| 503.9 Hz / p0 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 503.9 Hz / p1 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 659.3 Hz / p0 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 659.3 Hz / p1 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |

**Every new arm and seed fails both 354.5 Hz post-change cases at 0/112**, selecting approximately F0/2. The other ten new cases pass at 100% for every new arm/seed, so each arm passes 30/36 combinations but fails the per-case gate. Both phases have 112/112 raw and grouped correct-candidate coverage, 112/112 positive original voiced decisions, zero grouping loss and zero unavailable post-change frames. All three prior selectors also fail these new cases. The separate [coverage/state diagnostic](../../benchmarks/voice-prosody/subharmonic-challenge-diagnostic.json) records raw/grouped coverage and voiced/unavailable outcomes in every region without changing denominators.

## Original analytic regressions

All original stationary and 100/200/300 Hz post-change gates pass for every designated seed. All fourteen original whole-case results remain reported, including transitions:

| Case | Seed 22 | Seed 23 | Seed 24 |
| --- | ---: | ---: | ---: |
| constant-100 | 100.000% (151/151) | 100.000% (151/151) | 100.000% (151/151) |
| constant-200 | 100.000% (151/151) | 100.000% (151/151) | 100.000% (151/151) |
| constant-300 | 100.000% (151/151) | 100.000% (151/151) | 100.000% (151/151) |
| glide-down-200 | 99.338% (150/151) | 99.338% (150/151) | 99.338% (150/151) |
| glide-down-100 | 100.000% (151/151) | 100.000% (151/151) | 100.000% (151/151) |
| glide-down-50 | 100.000% (151/151) | 100.000% (151/151) | 100.000% (151/151) |
| glide-up-50 | 100.000% (151/151) | 100.000% (151/151) | 100.000% (151/151) |
| glide-up-100 | 100.000% (151/151) | 100.000% (151/151) | 100.000% (151/151) |
| glide-up-200 | 100.000% (151/151) | 100.000% (151/151) | 100.000% (151/151) |
| step-100-200 | 98.675% (149/151) | 98.675% (149/151) | 98.675% (149/151) |
| step-200-100 | 99.338% (150/151) | 99.338% (150/151) | 99.338% (150/151) |
| observation-change-100 | 100.000% (151/151) | 100.000% (151/151) | 100.000% (151/151) |
| observation-change-200 | 100.000% (151/151) | 100.000% (151/151) | 100.000% (151/151) |
| observation-change-300 | 100.000% (151/151) | 100.000% (151/151) | 100.000% (151/151) |

The [aggregate evidence](../../benchmarks/voice-prosody/subharmonic-objective-summary.json) retains all fourteen natural methods and all twelve selectors across every original, consumed and new analytic case. Reference masks, 50-cent scoring, unavailable outcomes and source identities remain unchanged.

## Execution, verification and decision

All computation ran locally on the Apple M4 Max with 128 GiB RAM. Nine MPS fits took **98.49 seconds**, with peak observed MPS tensor allocation **39.19 MiB**. This is allocator telemetry, not total process memory. CPU handled preprocessing, inference, scoring and audits. No package installation, native build, remote compute, extra fit, new seed or threshold search occurred. The run completed normally; the prior harmonic experiment's separate serialization failure/recovery remains immutable. The unchanged read-only NumPy mask warning during scoring is retained, and the target builder only reads that mask.

The [independent scorer](../../benchmarks/voice-prosody/subharmonic-independent-audit.json) reconstructs **101,599 comparisons with zero difference**. The [explicit-equation checkpoint replay](../../benchmarks/voice-prosody/subharmonic-inference-audit.json) checks **2,361,780 frame/model pairs** with zero state, frequency, voiced-gate or candidate-membership mismatches. A separate analytic diagnostic agrees on 1,728 case/region/model scores. The [refitted-control audit](../../benchmarks/voice-prosody/subharmonic-refitted-control-audit.json) verifies every old/new control tensor and all 766,596 stored development frame pairs are identical. Its [executed source snapshot](../../benchmarks/voice-prosody/subharmonic-refitted-controls.py) retains the original paths and refuses to overwrite the recorded receipt.

Historical NumPy matmul warnings recurred in the independent original-voicing check. Outputs remained finite, with exact threshold decisions and maximum probability difference 1.39e-7 against stored values. Warnings are retained; output parity does not diagnose their cause.

The final local suite passes **306 research tests** and `npm run check`, retaining the three historical NumPy warnings. Publication whitespace checks preserve one trailing space at line 75 of the frozen plan to retain its prefit source identity; all other staged files pass. Exact source identities and logs are recorded in the [local validation receipt](../../benchmarks/voice-prosody/subharmonic-local-validation.json). Audit passes verify the outcome, not model advancement. Raw audio, generated PCM, feature/reference arrays, weights and natural frame outputs remain local and ignored.

**Natural advancement, consumed-challenge and prospective-challenge gates fail.** Original stationary and post-change gates pass. Stop at the declared comparison; do not promote this loss, choose another observed arm or refit. The next justified study is a separately bounded test of physical harmonic-balance training coverage across a predeclared frequency/phase range, retaining natural-data controls and fresh prospective cases. Its benefit remains unestablished. Production remains DSP; calibration, browser qualification, external speech and clinical use remain outside this result.
