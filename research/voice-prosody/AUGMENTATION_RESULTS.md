# Harmonic-balance training augmentation results

The fixed nine-fit study repairs the known **337.1 and 354.5 Hz post-change failures: 112/112 correct in both phases and all three seeds**, using either stationary or harmonic-balance augmentation. The designated balance model modestly improves natural development joint error from **8.4492% to 8.4018%**. **The overall advancement decision remains NO-GO:** its improvement over original MLP is below the unchanged 0.5-point requirement, and fresh 517.3/641.9 Hz cases expose missing correct candidates. No gates were relaxed, no new fit followed the result, and production remains DSP.

## What this comparison establishes

The [frozen plan](AUGMENTATION_EXPERIMENT_PLAN.md) compares natural-only training, stationary synthetic augmentation and predesignated harmonic-balance augmentation. Both augmented arms repair the targeted failures, and their natural errors differ by only **0.0013 percentage points**. The result supports the value of added synthetic training support in this experiment. It does **not** demonstrate that missing-fundamental transitions specifically caused the gain. No significance, broad generalization or clinical benefit is established from three seeds.

The fresh failures also separate two different bottlenecks. At 337.1/354.5 Hz, correct candidates are available and augmentation improves their ranking. At fresh 517.3/641.9 Hz after the harmonic change, **no raw or grouped candidate is within 50 cents in any of the 112 targets**. A selector constrained to emit an existing candidate cannot succeed on those post-change targets, regardless of its ranking. This is a candidate-support ceiling under the current contract. Before the change, correct candidates exist but some predictions still fail; the full regional record retains those ranking errors too.

## Frozen training comparison

Architecture remains the 3,009-parameter relative-context selector with acceptable-set loss. Each seed starts with identical tensors across arms. Scaling uses natural TRAIN only. Candidate features, grouping, frequency emission, frozen original MLP voicing, 40 ms support, 10 ms hop, 20 ms measurement delay and 50-cent tolerance are unchanged.

Natural data remain 240 sources from twelve TRAIN speakers and eighty sources from four disjoint development speakers, each under four fixed conditions. Every fit keeps the same 132,181 natural scheduled rows, including 107 grouped-uncovered zero-loss rows, twelve epochs and **396 optimizer updates with zero skips**. There is no early stopping or best-checkpoint/seed selection.

Paired synthetic TRAIN uses 96 sequences per arm: 24 predeclared frequencies from 70.25 to 650.25 Hz, two phases and gains 0.04/0.08. Exact 337.1/354.5 Hz evaluation cases are excluded. All sequences are 2 seconds at 48 kHz with constant latent F0. Stationary signals retain harmonic coefficients (1,0.5,0.25); balance signals change to (0,1,0.5) at 0.83 seconds. Nonzero harmonics 2 and 3 preserve the fundamental period. No noise or normalization is added.

Each synthetic pool has **18,336/18,336 covered common-eligible rows**; no synthetic uncovered rows were discarded. Both arms receive the same 33 x 512 row schedule per epoch, sampled without replacement with a separate fixed RNG namespace. That is 16,896 synthetic rows per epoch and 202,752 row presentations per augmented fit. Targets, schedules and per-case/per-region exposures are saved and independently reconstructed.

Each augmented update uses natural set loss plus **0.25 times synthetic set loss** within one optimizer step. The same natural batch and update count are retained; added input/compute and gradient magnitude/clipping can differ. The physical augmentation changes energy as well as harmonic content. Coverage-normalized objectives could weight the conditions differently; here both synthetic pools have complete coverage. Recorded epoch loss contributions weight minibatch objectives by natural covered-row counts and are not a global synthetic-row loss mean.

## Natural development results

| Arm | Seed 22 error | Seed 23 error | Seed 24 error | Mean error | Mean voiced success |
| --- | ---: | ---: | ---: | ---: | ---: |
| natural-only | 8.4341% | 8.4673% | 8.4460% | 8.4492% | 76.2170% |
| stationary-aug | 8.4357% | 8.3969% | 8.3768% | 8.4031% | 76.3968% |
| balance-aug | 8.4234% | 8.3851% | 8.3970% | 8.4018% | 76.3961% |

Joint error counts incorrect voicing, unavailable outcomes and pitch errors above 50 cents, equally averaged across speaker/condition cells. Original MLP scores **8.6349%**, hybrid **8.9224%**. The designated model improves by **0.2331 points versus MLP and 0.5206 points versus hybrid**. The requirement is at least 0.5 points against **both**, so the natural gate fails. Its improvement versus the refitted natural-only control is **0.0474 points**.

Each designated seed beats original MLP, every condition mean improves, and mean unconditional reference-voiced success rises from original MLP's 75.4057% to **76.3961%**. Those component passes do not override the minimum-gain failure. Development is consumed exploratory evidence, not fresh natural-speech validation.

## Targeted failures and all consumed challenge regressions

The natural-only control retains the old failures: seed 23 scores 0/112 at 337.1 Hz in both phases, and every seed scores 0/112 at 354.5 Hz. **Both augmentation arms reach 112/112 for all twelve targeted seed/phase/frequency combinations.** All 24 consumed challenge cases now pass at 100% in every designated seed. The stationary control also passes all 72 consumed case/seed combinations, so the repair cannot be attributed uniquely to spectral-transition training.

Designated balance-augmentation results retain every consumed case:

| Case | Seed 22 | Seed 23 | Seed 24 |
| --- | ---: | ---: | ---: |
| 87.3 Hz / p0 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 87.3 Hz / p1 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 137.9 Hz / p0 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 137.9 Hz / p1 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 223.7 Hz / p0 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 223.7 Hz / p1 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 337.1 Hz / p0 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 337.1 Hz / p1 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 467.3 Hz / p0 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 467.3 Hz / p1 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 613.7 Hz / p0 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 613.7 Hz / p1 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |

| Case | Seed 22 | Seed 23 | Seed 24 |
| --- | ---: | ---: | ---: |
| 79.7 Hz / p0 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 79.7 Hz / p1 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 151.3 Hz / p0 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 151.3 Hz / p1 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 319.7 Hz / p0 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 319.7 Hz / p1 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 354.5 Hz / p0 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 354.5 Hz / p1 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 503.9 Hz / p0 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 503.9 Hz / p1 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 659.3 Hz / p0 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 659.3 Hz / p1 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |

## Fresh prospective cases

All nine checkpoints froze before generating these twelve sequences. Complete frequency/phase/gain configurations are prospective. Frequencies are absent from synthetic TRAIN and the two immediately preceding harmonic/subharmonic fixed challenge grids, but earlier studies used 93.7 Hz stationary signals and 331.7 Hz glide endpoints. The new phases and gain 0.06 also differ from TRAIN. The family remains noiseless, with the same physical harmonic transition and timing. Historical glides cross TRAIN frequencies, so this is configuration disjointness, not disjoint instantaneous frequency support across all prior work.

| Case | Seed 22 | Seed 23 | Seed 24 |
| --- | ---: | ---: | ---: |
| 93.7 Hz / p0 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 93.7 Hz / p1 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 193.9 Hz / p0 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 193.9 Hz / p1 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 331.7 Hz / p0 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 331.7 Hz / p1 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 366.1 Hz / p0 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 366.1 Hz / p1 | 100.000% (112/112) | 100.000% (112/112) | 100.000% (112/112) |
| 517.3 Hz / p0 | 0.000% (0/112) | 0.000% (0/112) | 0.000% (0/112) |
| 517.3 Hz / p1 | 0.000% (0/112) | 0.000% (0/112) | 0.000% (0/112) |
| 641.9 Hz / p0 | 0.000% (0/112) | 0.000% (0/112) | 0.000% (0/112) |
| 641.9 Hz / p1 | 0.000% (0/112) | 0.000% (0/112) | 0.000% (0/112) |

The first eight cases pass at 100% for every arm/seed, including natural-only. **Both phases at 517.3 and 641.9 Hz fail at 0/112 for every new and prior selector.** Every such post-change target has positive original voicing, but zero raw/grouped correct-candidate coverage. There are no grouping losses or unavailable outputs; models emit approximately F0/2. Whole-case coverage is 77/191, and all whole/pre/boundary/post outcomes remain recorded. No failed targets are removed from any denominator.

The predesignated arm passes 24/36 fresh case/seed combinations, exactly as the stationary and natural-only controls do. This aggregate count does not satisfy the per-case, per-seed >=95% gate. The [coverage/state diagnostic](../../benchmarks/voice-prosody/augmentation-challenge-diagnostic.json) explains the support ceiling and retains pre-change ranking errors.

## Original analytic regressions

All original stationary and 100/200/300 Hz fully post-change gates pass. All fourteen whole-case results, including transitions, remain visible:

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

## Local execution and verification

All computation ran on the local Apple M4 Max with 128 GiB RAM. Nine MPS fits took **68.45 seconds**, with peak observed MPS allocation **45.03 MiB**; that is allocator telemetry, not total process memory. Outputs at completion occupied 96.16 MiB. CPU handled generation through the unchanged TypeScript extractor, inference, scoring and independent audits. No package installation, native build, remote computation or extra fit occurred.

Before fitting, 37 local preflight tests passed and toy CPU/MPS losses matched exactly; the maximum gradient difference was 7.45e-8 with no optimizer updates or scientific waveform generation. One initial toy test's overly strict cross-expression float32 equality was corrected before the preflight; no scientific result was used. All 33 frozen source hashes remain unchanged.

The [independent scorer](../../benchmarks/voice-prosody/augmentation-independent-audit.json) reconstructs **106,820 comparisons**, maximum difference 4.16e-17, plus exact natural/synthetic targets, schedules and exposure. The [explicit-equation inference audit](../../benchmarks/voice-prosody/augmentation-inference-audit.json) verifies **2,382,948 frame/model pairs** with zero state, frequency, voiced-gate or candidate-membership mismatches. The separate analytic diagnostic agrees on **2,304 case/region/model scores**. The [refitted-control comparison](../../benchmarks/voice-prosody/augmentation-refitted-control-audit.json) finds exact equality of all three prior control tensors and **766,596 saved development state/frequency pairs**; this checks stored results, without independently replaying optimization.

The unchanged read-only NumPy mask warning and historical NumPy matmul warnings remain documented. Finite output agreement does not diagnose warning causes. The final local suite passes **371 research tests** with three historical warnings, plus `npm run check`. Source identities and exact audit hashes are in the [local validation receipt](../../benchmarks/voice-prosody/augmentation-local-validation.json). The [aggregate evidence](../../benchmarks/voice-prosody/augmentation-summary.json) retains all fourteen natural methods and twelve selectors across all fifty analytic cases. Raw audio, generated PCM, references, features, weights and natural frame outputs remain local and ignored.

## Decision and next question

The targeted regression repair is real within these fixtures, but **natural advancement and fresh-challenge gates fail**. All original stationary, harmonic-change and consumed-challenge gates pass. No threshold is relaxed, no alternate arm replaces the designated candidate, and no model is promoted.

The next separately declared study should first diagnose and improve candidate support around the newly exposed frequencies, retaining these failures as regressions and reserving fresh configurations. More selector training alone cannot repair an absent candidate under the unchanged emission contract. Natural-speech improvement remains a separate unmet requirement. External speech, calibration, browser qualification and clinical validation remain outside this result.
