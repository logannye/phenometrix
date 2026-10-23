# Candidate coverage: support repaired, ranking still fails

**The bounded experiment is complete and remains a no-go.** The revised candidate policy restores a correct pitch candidate on all 448 targeted post-change windows at 517.3 and 641.9 Hz. All three frozen selectors still choose an incorrect candidate on every one of those windows. Natural development error increases from **8.4018% to 8.4571%**, and candidate coverage declines. All twelve fresh analytic configurations pass under both policies. No gate is relaxed, no additional fitting occurs, and production remains DSP.

The useful result is a separation of two bottlenecks: the original candidate-retention rule discards a valid fundamental, and recovering that candidate alone does not teach the frozen selector to prefer it. The [prospective plan](CANDIDATE_COVERAGE_PLAN.md), [aggregate evidence](../../benchmarks/voice-prosody/candidate-coverage-summary.json), and [local validation receipt](../../benchmarks/voice-prosody/candidate-coverage-local-validation.json) preserve the complete comparison.

## Why the candidate was missing

The four consumed failures contain an accurately interpolated full-window autocorrelation peak near the fundamental on every fully post-change target. At the representative 517.3 Hz window, the true peak is at 517.4681 Hz, only 0.562 cents high, but ranks tenth by integer-lag correlation. At 641.9 Hz it is at 642.4412 Hz, 1.459 cents high, and ranks twelfth. Longer period multiples align more closely with the 8-kHz integer lag grid and receive higher correlation scores. The full-window path keeps only its strongest peak; the suffix path also misses the fundamental in its two retained peaks.

The correct peak is the shortest admitted full-window maximum on all 448 targeted windows. This motivated exactly one predeclared policy: replace the strongest full-window candidate with the shortest-lag already-admitted maximum. The 0.35 admission threshold, 50–700 Hz bounds, interpolation, observed confidence score and six-candidate capacity remain unchanged. Legacy candidates and both suffix candidates remain. A hypothesis supplied only by the displaced full-window source can disappear, so the experiment records losses as well as gains.

The [consumed-input diagnosis](../../benchmarks/voice-prosody/candidate-coverage-diagnosis.json) is a post-run reproduction of the pre-freeze diagnosis. All numeric records match the original exactly; only creation time and the portable script's source binding differ. Both receipt hashes are retained. This is diagnosis of known failures, not fresh validation.

## Fixed comparison and evaluation boundary

The original and revised policies use the same three previously trained 3,009-parameter balance-augmentation selectors, seeds 20261022–20261024, with unchanged natural-TRAIN scaling. The original MLP makes the voicing decision from original features for both policies. Grouping still requires a complete span within five cents and emits an original member frequency. Correctness remains within 50 cents, with 40-ms support, 10-ms hop and 20-ms measurement delay. There are **zero new fits, optimizer updates or scaling updates**.

Natural evaluation reuses 240 TRAIN recordings from twelve speakers and eighty consumed development recordings from four disjoint speakers, each under four fixed conditions. All 960 TRAIN and 320 development input transformations match their historical PCM hashes. The recomputed original full-window peak and stored float32 score exactly match the original feature cache on every acquired frame. The other unchanged feature channels are copied and hash-bound; this is not a new independent replay of the entire DSP pipeline.

All fifty earlier analytic cases remain, including both phases of the 517.3/641.9 Hz failures. Twelve prospective cases cross 97.1, 213.7, 347.3, 529.1, 627.7 and 671.3 Hz with two new phase triples and gain 0.055. Their two-second signals remove the first harmonic at sample 39,840 while harmonics two and three preserve the fundamental period. Each has 191 common-eligible targets, including 112 fully post-change targets. Complete configurations are new, but the noiseless family and timing are familiar; historical glides and natural speech can cross these frequencies.

A 40-file source closure, including the plan, plus checkpoint identities and the challenge specification froze before extraction and new waveform generation. All predictions froze before this run read natural reference content or computed analytic scores. Exact historical control reproduction covers **794,316 state/frequency pairs**. The chronology is source-level evidence, not an operating-system access audit, and does not make consumed data independent.

## Targeted regressions and fresh cases

Both phases have the same outcome at each frequency below. Counts refer to each case's 112 fully post-change targets; selector counts hold for every seed independently.

| Cases | Original raw/grouped coverage | Revised raw/grouped coverage | Original correct selections | Revised correct selections |
| --- | ---: | ---: | ---: | ---: |
| 517.3 Hz, phases 0 and 1 | 0/112 | 112/112 | 0/112 | 0/112 |
| 641.9 Hz, phases 0 and 1 | 0/112 | 112/112 | 0/112 | 0/112 |
| All twelve fresh configurations | 112/112 | 112/112 | 112/112 | 112/112 |

Original voicing remains positive on all targeted post-change windows, with no unavailable outputs or grouping losses. The revised model therefore fails at selection despite having a correct available candidate. All twelve targeted case/seed combinations remain below the unchanged 95% gate. The failures stay in the regression set.

All 24 earlier harmonic-context and subharmonic-objective cases, including the repaired 337.1 and 354.5 Hz cases, retain 112/112 correct post-change outputs in every revised seed. The other eight consumed augmentation cases also retain 112/112. All original stationary and harmonic-change gates pass. The fourteen original whole-case outcomes are unchanged: descending 200-Hz/s glide and 200→100 Hz step score 150/151, the 100→200 Hz step scores 149/151, and the remaining eleven cases score 151/151. Whole, pre-change, boundary, post-change and transition outcomes for every method and all 62 cases remain in the aggregate evidence.

The twelve fresh cases pass for every original and revised seed. That preserves performance on those configurations; it supplies no evidence of improvement over the control or generalization to independent natural speech.

## Natural speech and coverage tradeoff

Joint error counts incorrect voicing, unavailable output and voiced pitch errors beyond 50 cents on the unchanged common reference grid. Numbers below equally weight the sixteen development speaker/condition cells, then the three selector seeds. Development has already been consumed for research decisions.

| Method | Joint error | Unconditional reference-voiced success |
| --- | ---: | ---: |
| Original MLP | 8.6349% | 75.4057% |
| Original hybrid | 8.9224% | 74.1933% |
| Frozen balance selectors, original candidates | 8.4018% | 76.3961% |
| Same frozen selectors, revised candidates | **8.4571%** | **76.1689%** |

Revised seed errors are 8.4887%, 8.4279% and 8.4547%. Their mean improves over MLP by only **0.1779 percentage points** and over hybrid by **0.4653 points**; both miss the required 0.5-point improvement. Error rises **0.0553 points** relative to the paired original-policy selectors. Each seed still beats MLP, each condition mean improves over MLP, and unconditional voiced success remains above MLP. These component passes cannot override the failed advancement criterion.

Candidate coverage is measured on reference-voiced targets, independent of whether the fixed voicing head permits emission. Original-to-revised losses and gains below are separate actual frame sets, macro-averaged across speaker/condition cells.

| Split / representation | Original coverage | Revised coverage | Lost coverage | Gained coverage |
| --- | ---: | ---: | ---: | ---: |
| TRAIN / raw | 86.3825% | 85.9882% | 0.5853 points | 0.1910 points |
| TRAIN / grouped | 86.3135% | 85.9375% | 0.5807 points | 0.2047 points |
| Development / raw | 88.1304% | 87.6029% | 0.6371 points | 0.1096 points |
| Development / grouped | 88.0885% | 87.5615% | 0.6446 points | 0.1175 points |

The unchanged prerequisites still pass. Revised fixed-original-voicing raw-candidate oracle headroom is **1.2442 points**, above the 1-point minimum; grouped headroom is 1.2362 points. Actual revised raw-to-grouped loss is **0.0507 points** on TRAIN and **0.0414 points** on development, below 0.5. This grouping-loss test measures a different event from the larger original-to-revised eviction losses above. We do not reinterpret it as a passing policy-eviction gate.

Reference exclusions remain explicit: TRAIN has 4,800 boundary-reference targets and 2,880 predictions outside reference extent across 960 condition-inputs; development has 1,600 and 960 across 320. Both have zero invalid-reference targets. Failed analytic targets remain in their denominators.

## Local execution and verification

One local CPU execution on Apple M4 Max with 128 GiB RAM took **227.35 seconds** inside the experiment, or 227.97 seconds including the supervisor. Peak observed Python RSS was **3.45 GiB** and peak observed Node-worker RSS **409.14 MiB**. These are process observations, not whole-machine memory bounds. Completion artifacts occupied **88.99 MiB** before later audit receipts. The 75-minute, 16-GiB Python, 2-GiB worker and 2-GiB output ceilings were respected. No GPU fitting was needed; no remote computation, package installation or native build occurred.

The pre-freeze fixture suite passed 63 tests. The [independent inference auditor](../../benchmarks/voice-prosody/candidate-coverage-inference-audit.json) reconstructs explicit network equations on both candidate pools and verifies **1,602,744 frame/method pairs**, with zero state, frequency, voicing-gate or candidate-membership mismatches. It reads no reference, analytic truth, challenge specification or score content. The [independent scoring auditor](../../benchmarks/voice-prosody/candidate-coverage-independent-audit.json) verifies **77,238 numeric comparisons**, with maximum difference **1.11e-16**, reconstructing natural metrics, candidate coverage and eviction, analytic regions, gates and control equality. Two auditor-only repairs were required: replace a directory-based reference-access guard that incorrectly classified microphone WAVs with exact reference-role paths, and validate an unchanged explanatory step-criterion string alongside independently computed numbers. Both failed attempts, source snapshots and repairs are retained. No scientific source, metric formula, experiment or prediction changed or reran.

The final local suite passes **460 research tests** and `npm run check`. Three historical NumPy warnings and exact source/artifact identities are recorded in the local validation receipt. Historical NumPy warnings are retained rather than treated as diagnosed by finite output agreement. Source hashes remained unchanged through scoring and auditing. Raw speech, generated PCM, features, references, checkpoints and natural frame predictions remain local and ignored.

## Decision and next question

The candidate-retention diagnosis is supported, but this fixed replacement policy does not produce a deployable improvement. It sacrifices natural coverage, leaves the targeted ranking failures unresolved and fails the unchanged natural and per-case gates. The bounded study stops here with the default unchanged.

A separately declared next study should preserve useful existing hypotheses while admitting missing fundamentals, measure the resulting coverage losses before training, and then train a selector on that candidate distribution. It must retain these failures, reserve new complete evaluation cases and satisfy the same natural-speech and per-case advancement gates. Browser qualification, independent natural speech, calibrated uncertainty and clinical validation remain separate requirements.
