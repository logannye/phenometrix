# Harmonic-context experiment: repair with a remaining generalization failure

**The designated relative-context model repairs the known 300 Hz harmonic-balance failure in all three seeds, but fails the unchanged natural-improvement gate and the new prospective challenge gate.** Production remains DSP. No additional fit or threshold search followed these outcomes.

## Diagnosis and controlled intervention

The [frozen diagnosis](../../benchmarks/voice-prosody/harmonic-ranking-diagnosis.json) reproduces all 5,616 old checkpoint/frame decisions it checks. After the original 300 Hz harmonic change, correct candidates and voiced decisions remain available on all 95 targets. In the failing seed, the best correct logit is 4.0212 but the approximately 100 Hz alternative scores 4.1728. Strong AMDF and full-window correlation support that threefold-period alternative. Passing old seeds show that this is a learned ranking failure, not proof that max pooling cannot represent the correct choice.

The [predeclared plan](HARMONIC_EXPERIMENT_PLAN.md) tests a shared neighbor-message encoder. Its query-specific context includes each neighbor’s standardized estimator evidence and relative log frequency. An absolute-context control zeros only the frequency-ratio input. Both allocate 3,009 parameters and start from byte-identical initial tensors within each seed; the 32 ratio-column weights receive zero gradients in the absolute arm, so equal effective functional capacity is not claimed. The old 2,689-parameter shared models remain frozen comparators.

The two context settings cross nearest and acceptable-set losses, using seeds 20261022/23/24. All twelve fits use the same 132,181 natural training rows, including 107 grouped-uncovered rows with zero pitch loss, 12 epochs and 396 optimizer updates without skipped batches. Training/development speakers, source recordings and selected prompts remain disjoint; the same 240 training sources/12 speakers and 80 development sources/four speakers appear under four fixed conditions. No synthetic fitting, candidate addition, grouping change, voicing retraining, temporal state or reference adjustment occurs. Development remains reused exploratory data.

## Natural speech and the original failure

Joint error is pooled within each speaker/condition cell, then averaged equally. Unconditional voiced success retains every valid eligible reference-voiced frame. Invalid/boundary references are excluded and counted. Seeds describe optimization variation, not participant uncertainty.

| Context | Objective | Joint error by seed | Mean joint error | Mean voiced success | Original 300 Hz post-change success by seed |
| --- | --- | ---: | ---: | ---: | ---: |
| absolute | nearest | 8.4428% / 8.3730% / 8.4353% | 8.4170% | 76.3406% | 100% / 100% / 100% |
| absolute | acceptable-set | 8.4737% / 8.4909% / 8.4794% | 8.4813% | 76.0757% | 100% / 0% / 100% |
| relative | nearest | 8.3999% / 8.3770% / 8.3988% | 8.3919% | 76.4474% | 100% / 100% / 100% |
| relative | acceptable-set | 8.4341% / 8.4673% / 8.4460% | 8.4492% | 76.2170% | 100% / 100% / 100% |

The relative/acceptable candidate scores **8.4492%** versus the original MLP’s **8.6349%** and hybrid’s **8.9224%**. Gains are **0.1858** and **0.4733 percentage points**, both below the required 0.5. Every candidate seed beats MLP; each condition improves on average; mean voiced success rises from 75.4057% to 76.2170%. Those component passes do not override the failed minimum-gain requirement.

The equal-parameter absolute/acceptable control scores 8.4813%, so the relative input improves mean error by only **0.0322 points**. It changes the previously failing seed from 0/95 to 95/95 on the old 300 Hz post-change region. All candidate seeds pass all original stationary and 100/200/300 Hz post-change cases. The nearest-loss arms remain diagnostic controls and are not substituted for the designated candidate.

## Prospective frequency/phase challenge

All twelve final checkpoints froze before generating twelve new 2-second waveforms. Six frequencies cross phase vectors P0=(0.37,-0.41,1.13) and P1=(1.19,0.23,-0.73) radians. At 0.83 seconds the first three harmonic amplitudes change from (1,0.5,0.25) to (0,1,0.5), preserving the latent F0. Each fully post-change region contains 112 scored targets; whole-case reports retain 191 targets, including boundary frames. These are new noiseless configurations of one generator family, not independent natural speech.

Designated-candidate post-change success, with seeds 20261022/23/24:

| F0 | Phase | Seed 22 | Seed 23 | Seed 24 |
| ---: | --- | ---: | ---: | ---: |
| 87.3 Hz | P0 | 100% | 100% | 100% |
| 87.3 Hz | P1 | 100% | 100% | 100% |
| 137.9 Hz | P0 | 100% | 100% | 100% |
| 137.9 Hz | P1 | 100% | 100% | 100% |
| 223.7 Hz | P0 | 100% | 100% | 100% |
| 223.7 Hz | P1 | 100% | 100% | 100% |
| 337.1 Hz | P0 | 100% | 0% | 100% |
| 337.1 Hz | P1 | 100% | 0% | 100% |
| 467.3 Hz | P0 | 100% | 100% | 100% |
| 467.3 Hz | P1 | 100% | 100% | 100% |
| 613.7 Hz | P0 | 100% | 100% | 100% |
| 613.7 Hz | P1 | 100% | 100% | 100% |

**Seed 20261023 fails all 112 post-change targets in both 337.1 Hz phase cases**, selecting approximately F0/3. The other 34 seed/case combinations pass at 100%, but the per-case/per-seed gate fails. The [postfit bottleneck audit](../../benchmarks/voice-prosody/harmonic-challenge-diagnostic.json) verifies 100% raw and grouped correct-candidate coverage and unchanged voiced decisions in every post-change region. The residual failure is selector choice, not missing candidates or a voicing rejection. Probability values were not retained for the new challenge; binary voicing is reconstructed from agreement across all eighteen frozen selector outputs.

Across the four new arms, passing seed/case counts are 34/36 absolute/nearest, 34/36 absolute/acceptable, 32/36 relative/nearest and 34/36 relative/acceptable. Thus the repair of the old failure does not establish broad robustness. All eighteen selectors’ full metrics and every original regression are retained in the [aggregate evidence](../../benchmarks/voice-prosody/harmonic-context-summary.json).

## Every original analytic regression

Designated-candidate whole-case success within 50 cents, including boundary and nonvoiced/unavailable outcomes in the denominator:

| Reused case | Seed 22 | Seed 23 | Seed 24 |
| --- | ---: | ---: | ---: |
| constant-100 | 100.000% | 100.000% | 100.000% |
| constant-200 | 100.000% | 100.000% | 100.000% |
| constant-300 | 100.000% | 100.000% | 100.000% |
| glide-down-200 | 99.338% | 99.338% | 99.338% |
| glide-down-100 | 100.000% | 100.000% | 100.000% |
| glide-down-50 | 100.000% | 100.000% | 100.000% |
| glide-up-50 | 100.000% | 100.000% | 100.000% |
| glide-up-100 | 100.000% | 100.000% | 100.000% |
| glide-up-200 | 100.000% | 100.000% | 100.000% |
| step-100-200 | 98.675% | 98.675% | 98.675% |
| step-200-100 | 99.338% | 99.338% | 99.338% |
| observation-change-100 | 100.000% | 100.000% | 100.000% |
| observation-change-200 | 100.000% | 100.000% | 100.000% |
| observation-change-300 | 100.000% | 100.000% | 100.000% |

## Decision, execution and recovery

The inherited natural-improvement gate fails. The inherited stationary gate and strengthened old post-change gate pass. The added prospective challenge gate fails. The final decision is **no-go for advancement**. Historical failed gates remain failed, and no winning arm, seed or threshold replaces the predesignated candidate. This completes the declared bounded experiment; calibration, browser qualification, fresh natural-speech evaluation and production integration remain unperformed.

All computation ran locally. Twelve MPS fits took **57.07 seconds**; peak observed MPS allocator usage was **35.10 MiB**, not whole-process memory. CPU performed preprocessing, frozen inference and scoring. No packages were installed, native builds performed or hosted jobs used. Raw data, generated PCM, weights and per-frame outputs remain ignored and local.

The original run correctly froze every checkpoint and prediction, then failed while serializing NumPy boolean gate values to JSON. Its failed manifest, source snapshot and execution log remain unchanged; there is no replacement original metrics file. A separate `recover_harmonic_scores` process recomputed the same scores from the frozen arrays and converted NumPy scalars to equivalent Python scalars solely at JSON serialization. It performed **zero additional fits, inference passes or waveform generations**, and did not alter the equations, masks, thresholds or gate definitions.

The original and recovery scoring retained a PyTorch warning when a read-only NumPy eligibility mask was wrapped as a tensor. The unchanged target builder only reads that mask; no mutation is performed. The historical NumPy matmul warnings also remain visible in the research tests. These warnings are documented, not suppressed. Independent arithmetic and checkpoint replay checks are recorded in the local validation receipt.

## Local verification

The final local suite passes **244 research tests** and `npm run check`. Three
historical NumPy matmul warnings remain visible. The [independent score
audit](../../benchmarks/voice-prosody/harmonic-independent-audit.json) reconstructs
138,382 numeric comparisons, with maximum difference 1.39e-17, and independently
confirms every gate decision. The [checkpoint inference
audit](../../benchmarks/voice-prosody/harmonic-inference-audit.json) evaluates
explicit Torch equations on 3,120,816 frame/model pairs across all twelve
checkpoints, development inputs and all 26 analytic cases. It finds zero state,
frequency, frozen-voicing-gate or original-candidate-membership mismatches and
reconstructs every same-seed initialization digest. It reads no reference labels
or analytic target arrays and performs no fitting.

The [local validation receipt](../../benchmarks/voice-prosody/harmonic-local-validation.json)
binds final source bytes, check-log hashes, the original failed run and the
separate recovery. Passing software and evidence-integrity checks establish
that the recorded result is reproducible; the model advancement decision remains
no-go. The product/browser suites were not rerun for these offline-only changes.
